// ============================================================================
// Multidimensional IRT model with:
//   - Sparse skill requirements (CSR format)
//   - Normalised skill effect (average not sum)
//   - 3PL-style guessing floor tied to fast-response indicator
//   - Age + gender as hierarchical covariates on skill
//   - Log response time as auxiliary likelihood (log-normal mixture)
//   - reduce_sum for within-chain parallelism
// ============================================================================

functions {

  // --------------------------------------------------------------------------
  // Compute the skill effect for one observation using sparse CSR structure.
  // Returns the AVERAGE skill value across required skills (not sum),
  // so questions requiring different numbers of skills are comparable.
  // --------------------------------------------------------------------------
  real sparse_skill_effect(
      matrix user_skill,          // [n_players, n_skills]
      int    player,
      array[] int skill_flat,     // flattened skill indices
      array[] int skill_start,    // start position per question (1-indexed)
      array[] int skill_count,    // number of skills per question
      int    question
  ) {
    int cnt = skill_count[question];
    if (cnt == 0) return 0.0;

    real total = 0.0;
    int  start = skill_start[question];
    for (s in start:(start + cnt - 1))
      total += user_skill[player, skill_flat[s]];
    return total / cnt;
  }

  // --------------------------------------------------------------------------
  // Partial log-likelihood for reduce_sum.
  // Slices over the N observations dimension.
  // --------------------------------------------------------------------------
  real partial_log_lik(
      // Sliced data (reduce_sum slices these automatically)
      array[] int response_slice,
      int         start,
      int         end,

      // Remaining data passed through as-is
      array[] int  player_id,
      array[] int  question_id,
      array[] int  is_fast,
      vector       log_time,

      // Sparse skill data
      array[] int  skill_flat,
      array[] int  skill_start,
      array[] int  skill_count,

      // Parameters
      matrix       user_skill,          // [n_players, n_skills]
      vector       question_difficulty, // [n_questions]
      real         gamma_base,          // guessing floor (base)
      real         gamma_fast_delta,    // extra floor for fast responses
      real         alpha,               // upper asymptote
      real         beta_time,           // effect of log-time on engagement
      real         mu_log_time,         // log-normal mean for RT
      real         sigma_log_time       // log-normal sd for RT
  ) {
    real lp = 0.0;

    for (n in start:end) {
      int  p = player_id[n];
      int  q = question_id[n];

      // ── Skill effect (sparse, averaged) ───────────────────────────────────
      real skill_eff = sparse_skill_effect(
          user_skill, p, skill_flat, skill_start, skill_count, q
      );

      // ── 3PL probability of correct ────────────────────────────────────────
      // Guessing floor rises for fast responses (likely guessing regime)
      real gamma     = gamma_base + gamma_fast_delta * is_fast[n];
      real p_skill   = inv_logit(skill_eff - question_difficulty[q]);
      real p_correct = gamma + (alpha - gamma) * p_skill;

      // Clamp to valid probability range (numerical safety)
      p_correct = fmin(fmax(p_correct, 1e-6), 1.0 - 1e-6);

      lp += bernoulli_lpmf(response_slice[n - start + 1] | p_correct);

      // ── Response time: mixture of engaged (log-normal) and guess (flat) ───
      // Engagement probability increases with log time
      real p_engaged    = inv_logit(beta_time * log_time[n]);
      real rt_lp_engage = normal_lpdf(log_time[n] | mu_log_time, sigma_log_time);
      real rt_lp_guess  = normal_lpdf(log_time[n] | 3.0, 1.5); // ~20s, vague

      lp += log_mix(p_engaged, rt_lp_engage, rt_lp_guess);
    }
    return lp;
  }

}

// ============================================================================
data {
  // Dimensions
  int<lower=1> N;
  int<lower=1> n_players;
  int<lower=1> n_questions;
  int<lower=1> n_skills;

  // Observations
  array[N] int<lower=1, upper=n_players>   player_id;
  array[N] int<lower=1, upper=n_questions> question_id;
  array[N] int<lower=0, upper=1>           response;
  array[N] int<lower=0, upper=1>           is_fast;
  vector[N]                                log_time;

  // Sparse skill structure (CSR format)
  int<lower=0>                             n_skill_entries;
  // When n_skill_entries==0 (no questions have skills) bounds are unchecked
  // safely because sparse_skill_effect guards with cnt==0 before indexing
  array[n_skill_entries] int               skill_flat;   // values in [1, n_skills]
  array[n_questions]     int<lower=1>      skill_start;
  array[n_questions]     int<lower=0>      skill_count;

  // Player covariates
  vector[n_players]                        age_z;    // standardised, missing → 0
  array[n_players] int<lower=0, upper=2>   gender;   // 0=unknown, 1=ref, 2=other
}

// ============================================================================
transformed data {
  // Pre-compute gender indicator (gender==2 vs reference==1, unknown→0)
  vector[n_players] gender_ind;
  for (p in 1:n_players)
    gender_ind[p] = (gender[p] == 2) ? 1.0 : 0.0;
}

// ============================================================================
parameters {
  // ── Population skill distribution ─────────────────────────────────────────
  vector[n_skills]            mu_skill;
  vector<lower=0>[n_skills]   sigma_skill;

  // ── Skill regression on player covariates ─────────────────────────────────
  vector[n_skills]  beta_age;
  vector[n_skills]  beta_gender;

  // ── Raw (non-centred) player skills — avoids funnel geometry ──────────────
  matrix[n_players, n_skills] user_skill_raw;

  // ── Question parameters ────────────────────────────────────────────────────
  vector[n_questions] question_difficulty;

  // ── 3PL asymptotes ────────────────────────────────────────────────────────
  // Design constraint: gamma_base + gamma_fast_delta must always < alpha
  // We enforce this through the upper bounds:
  //   gamma_base      in [0,    0.40]
  //   gamma_fast_delta in [0,   0.15]  → max combined floor = 0.55
  //   alpha            in [0.6, 1.0]  → guaranteed alpha > max floor
  // Using logit-scale parameters avoids truncated-beta prior issues
  real<lower=0,    upper=0.40> gamma_base;
  real<lower=0,    upper=0.15> gamma_fast_delta;
  real<lower=0.60, upper=1.0>  alpha;

  // ── Response time parameters ───────────────────────────────────────────────
  real                      mu_log_time;
  real<lower=0>             sigma_log_time;
  real                      beta_time;      // log-time → engagement probability
}

// ============================================================================
transformed parameters {
  // ── Non-centred parameterisation of player skills ─────────────────────────
  // user_skill[p,k] = mu_p[k] + sigma_skill[k] * user_skill_raw[p,k]
  // where mu_p[k] = mu_skill[k] + beta_age[k]*age_z[p] + beta_gender[k]*gender_ind[p]
  matrix[n_players, n_skills] user_skill;

  for (p in 1:n_players) {
    // mu_p is a vector[n_skills] — all three terms are vectorised
    vector[n_skills] mu_p = mu_skill
                          + beta_age    * age_z[p]
                          + beta_gender * gender_ind[p];
    // Vectorised row assignment — no inner k loop needed
    user_skill[p] = (mu_p + sigma_skill .* user_skill_raw[p]')';
  }
}

// ============================================================================
model {
  // ── Priors ─────────────────────────────────────────────────────────────────

  // Population skill
  mu_skill    ~ normal(0, 1);
  sigma_skill ~ exponential(1);

  // Covariate effects — regularising priors
  beta_age    ~ normal(0, 0.5);
  beta_gender ~ normal(0, 0.5);

  // Non-centred player skills — standard normal in raw space
  to_vector(user_skill_raw) ~ std_normal();

  // Question difficulty
  question_difficulty ~ normal(0, 1);

  // 3PL asymptotes — priors on the unconstrained scale
  // gamma_base: centred near 0.25 (4-choice floor), soft via normal on logit scale
  // Use uniform within bounds — the bounds already encode the domain knowledge,
  // so a flat prior within them is honest; beta(5,15) was implicitly truncated
  // which distorted its shape. If you want informative priors, use:
  //   target += normal_lpdf(gamma_base | 0.25, 0.05);
  // which is a proper normal on the parameter scale within its bounds.
  target += normal_lpdf(gamma_base        | 0.25, 0.08);
  target += normal_lpdf(gamma_fast_delta  | 0.05, 0.05);
  target += normal_lpdf(alpha             | 0.90, 0.05);

  // Response time
  mu_log_time    ~ normal(4.5, 1);  // ~90s typical engaged time
  sigma_log_time ~ exponential(1);
  beta_time      ~ normal(1, 0.5);  // positive: more time → more engaged

  // ── Likelihood via reduce_sum (within-chain parallelism) ──────────────────
  target += reduce_sum(
    partial_log_lik,
    response,         // sliced argument
    1,                // grainsize (1 = auto-tune)

    // remaining data
    player_id,
    question_id,
    is_fast,
    log_time,
    skill_flat,
    skill_start,
    skill_count,

    // parameters
    user_skill,
    question_difficulty,
    gamma_base,
    gamma_fast_delta,
    alpha,
    beta_time,
    mu_log_time,
    sigma_log_time
  );
}

// ============================================================================
generated quantities {
  // ── Per-observation log-likelihood (for LOO-CV) ───────────────────────────
  vector[N] log_lik;

  // ── Posterior predictive accuracy ─────────────────────────────────────────
  array[N] int y_rep;

  for (n in 1:N) {
    int  p = player_id[n];
    int  q = question_id[n];

    real skill_eff = sparse_skill_effect(
        user_skill, p, skill_flat, skill_start, skill_count, q
    );

    real gamma     = gamma_base + gamma_fast_delta * is_fast[n];
    real p_correct = gamma + (alpha - gamma) * inv_logit(skill_eff - question_difficulty[q]);
    p_correct      = fmin(fmax(p_correct, 1e-6), 1.0 - 1e-6);

    log_lik[n] = bernoulli_lpmf(response[n] | p_correct);
    y_rep[n]   = bernoulli_rng(p_correct);
  }
}
