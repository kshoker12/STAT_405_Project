library(arrow)
library(dplyr)
library(tidyr)
library(cmdstanr)
library(data.table)   # for fast grouped ops

# ── Read parquet ──────────────────────────────────────────────────────────────
df_raw <- read_parquet("processing/processed_data/full_task_all.parquet")

# Read K from metadata — arrow stores custom metadata as raw bytes
pq_meta <- read_parquet_metadata("processing/processed_data/full_task_all.parquet")
K       <- as.integer(rawToChar(pq_meta$metadata[["K"]]))

# ── Defensive cast of list-column from Arrow ──────────────────────────────────
# Arrow can return large_list or chunked types; force to plain R integer lists
df_raw <- df_raw |>
  mutate(skill_indices = lapply(skill_indices, as.integer))

# ── Basic cleaning ────────────────────────────────────────────────────────────
df <- df_raw |>
  filter(!is.na(TimeTaken), TimeTaken >= 5, TimeTaken <= 1800) |>
  mutate(
    log_time = log(TimeTaken),
    is_fast  = as.integer(TimeTaken < 20)
  )

# ── Stratified sample ─────────────────────────────────────────────────────────
set.seed(42)
N_TARGET <- 200000L

anchors <- df |>
  group_by(UserId, IsCorrect) |>
  slice_sample(n = 5) |>
  ungroup()

remaining <- df |> filter(!AnswerId %in% anchors$AnswerId)
top_up    <- remaining |>
  slice_sample(n = max(0L, min(N_TARGET - nrow(anchors), nrow(remaining))))

sample_df <- bind_rows(anchors, top_up) |>
  slice_sample(n = min(N_TARGET, n())) |>
  # ── Remap IDs AFTER sampling so indices are always contiguous ────────────
  mutate(
    player_idx   = as.integer(factor(UserId)),
    question_idx = as.integer(factor(QuestionId))
  )

n_players   <- max(sample_df$player_idx)
n_questions <- max(sample_df$question_idx)

# Verify contiguity — gaps here would silently misalign the skill arrays
stopifnot(n_players   == n_distinct(sample_df$player_idx))
stopifnot(n_questions == n_distinct(sample_df$question_idx))

cat("Sample:    ", nrow(sample_df), "rows\n")
cat("Players:   ", n_players, "\n")
cat("Questions: ", n_questions, "\n")

# ── Build question → new index mapping ───────────────────────────────────────
# Keep ONE row per original QuestionId with its skill_indices
# This is the key fix: look up by original QuestionId, not by question_idx,
# then join the new question_idx in explicitly
q_map <- sample_df |>
  distinct(QuestionId, question_idx) |>           # original → new index
  left_join(                                       # attach skill_indices
    df_raw |>
      distinct(QuestionId, skill_indices),         # from unfiltered source
    by = "QuestionId"
  ) |>
  arrange(question_idx)

# Confirm no gaps or duplicates in q_map
stopifnot(nrow(q_map) == n_questions)
stopifnot(!anyDuplicated(q_map$question_idx))

# ── Build sparse CSR skill structure ─────────────────────────────────────────
# Vectorised approach — no row-by-row loop
build_sparse_skills <- function(q_map, K, n_questions) {

  # Expand list-column to long format in one shot with data.table
  dt <- data.table(
    question_idx = q_map$question_idx,
    skills       = q_map$skill_indices
  )

  # Unnest: each row becomes one (question_idx, skill) pair
  dt_long <- dt[, .(skill = unlist(skills, use.names = FALSE)),
                  by = question_idx]

  # Bounds check
  dt_long <- dt_long[!is.na(skill) & skill >= 1L & skill <= K]

  # Sort by question_idx to build contiguous CSR blocks
  setorder(dt_long, question_idx)

  skill_flat <- dt_long$skill

  # Count skills per question (including questions with zero skills)
  counts_dt        <- dt_long[, .N, by = question_idx]
  skill_count      <- integer(n_questions)       # initialise all zeros
  skill_count[counts_dt$question_idx] <- counts_dt$N

  # Start positions: cumsum of counts + 1 (1-indexed for Stan)
  skill_start      <- integer(n_questions)
  skill_start[1]   <- 1L
  if (n_questions > 1)
    skill_start[2:n_questions] <- cumsum(skill_count)[1:(n_questions - 1)] + 1L

  # Questions with zero skills: start is meaningless but must be >= 1
  # Stan will never index into skill_flat for these (skill_count == 0 guard)
  skill_start[skill_count == 0L] <- 1L

  list(
    n_skill_entries = length(skill_flat),
    skill_flat      = as.array(as.integer(skill_flat)),
    skill_start     = as.array(skill_start),
    skill_count     = as.array(skill_count)
  )
}

sparse <- build_sparse_skills(q_map, K, n_questions)

# Sanity check CSR structure
stopifnot(all(sparse$skill_flat >= 1L))
stopifnot(all(sparse$skill_flat <= K))
stopifnot(all(sparse$skill_start >= 1L))
stopifnot(all(sparse$skill_count >= 0L))
with(sparse, stopifnot(
  sum(skill_count) == n_skill_entries
))

# ── Player covariates ─────────────────────────────────────────────────────────
# One row per player in player_idx order — vectorised, no loop
player_meta <- sample_df |>
  distinct(player_idx, Gender, Age) |>
  arrange(player_idx)

# Verify one row per player (no conflicting metadata)
stopifnot(nrow(player_meta) == n_players)

age_raw  <- player_meta$Age
age_mean <- mean(age_raw, na.rm = TRUE)
age_sd   <- sd(age_raw,   na.rm = TRUE)
age_sd   <- ifelse(is.na(age_sd) | age_sd == 0, 1.0, age_sd)

age_z    <- ifelse(is.na(age_raw), 0.0, (age_raw - age_mean) / age_sd)

# Gender: 0=unknown, 1=reference, 2=other — match Stan transformed data block
gender   <- as.integer(player_meta$Gender)
gender[is.na(gender)] <- 0L

# age_known is NOT passed — Stan model does not use it
# (Stan uses age_z with missing filled as 0 = population mean)

# ── Assemble Stan data list ───────────────────────────────────────────────────
stan_data <- c(
  list(
    N           = nrow(sample_df),
    n_players   = n_players,
    n_questions = n_questions,
    n_skills    = K,

    player_id   = as.array(sample_df$player_idx),
    question_id = as.array(sample_df$question_idx),
    response    = as.array(as.integer(sample_df$IsCorrect)),
    is_fast     = as.array(sample_df$is_fast),
    log_time    = as.array(sample_df$log_time),

    age_z       = as.array(age_z),
    gender      = as.array(gender)
  ),
  sparse
)

# ── Final validation ──────────────────────────────────────────────────────────
stopifnot(max(stan_data$player_id)   == stan_data$n_players)
stopifnot(max(stan_data$question_id) == stan_data$n_questions)
stopifnot(length(stan_data$age_z)    == stan_data$n_players)
stopifnot(length(stan_data$gender)   == stan_data$n_players)
stopifnot(length(stan_data$skill_start) == stan_data$n_questions)
stopifnot(length(stan_data$skill_count) == stan_data$n_questions)

cat("n_skill_entries:", stan_data$n_skill_entries, "\n")
cat("Skill sparsity: ",
    round(1 - stan_data$n_skill_entries / (n_questions * K), 3), "\n")

# ── Compile and sample ────────────────────────────────────────────────────────
model <- cmdstan_model(
  "irt_model.stan",
  cpp_options = list(
    STAN_THREADS    = TRUE,
    STAN_CPP_OPTIMS = TRUE
  )
)

fit <- model$variational(
  data = stan_data,
  output_samples = 1000,
  iter = 10000,
  tol_rel_obj = 0.001,
  refresh = 500
)

# fit <- model$sample(
#   data              = stan_data,
#   chains            = 4,
#   parallel_chains   = 4,
#   threads_per_chain = 4,     # reduce_sum uses this — set to available_cores/4
#   iter_warmup       = 500,
#   iter_sampling     = 500,
#   adapt_delta       = 0.9,
#   max_treedepth     = 12,
#   refresh           = 50
# )

fit$cmdstan_diagnose()
fit$summary(c("mu_skill", "sigma_skill", "beta_age", "beta_gender",
              "gamma_base", "gamma_fast_delta", "alpha",
              "mu_log_time", "sigma_log_time")) |> print(n = 40)