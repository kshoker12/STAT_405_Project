library(cmdstanr)

file <- file.path("./bayesian_model/main.stan")
mod <- cmdstan_model(file)
export_dir <- "./bayesian_model/stan_exports"

interaction_data <- read.csv(file.path(export_dir, "interaction_data.csv"))
question_skill_requirements <- as.matrix(
  read.csv(file.path(export_dir, "question_skill_requirements.csv"))
)

n_players <- max(interaction_data$player_id)
n_questions <- max(interaction_data$question_id)
n_skills <- ncol(question_skill_requirements)
N <- nrow(interaction_data)

stan_data <- list(
  N = N,
  n_players = n_players,
  n_questions = n_questions,
  n_skills = n_skills,
  player_id = interaction_data$player_id,
  question_id = interaction_data$question_id,
  question_skill_requirements = question_skill_requirements,
  response = interaction_data$correct
)
fit_preview <- mod$sample(
  data = stan_data,
  chains = 1,
  parallel_chains = 1,
  iter_warmup = 50,
  iter_sampling = 100,
  refresh = 10,
  show_messages = TRUE
)
fit_preview$summary()
