// ============================================================================
// Naive (Level-1 skill) IRT model — Variational Inference variant
//
// Identical to naive_model/model.stan except:
//   - generated quantities block is commented out (avoid computing N quantities
//     on every VI draw sample — enable only for diagnostic/model-checking runs)
//
// This model uses K=9 broad Level-1 skills (Number, Algebra, Geometry, etc.)
// rather than the K=69 fine-grained Level-2 skills in the primary model.
// ============================================================================

functions {

  real sparse_skill_effect(
      matrix user_skill,
      int    player,
      array[] int skill_flat,
      array[] int skill_start,
      array[] int skill_count,
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

  real partial_log_lik(
      array[] int response_slice,
      int         start,
      int         end,

      array[] int  player_id,
      array[] int  question_id,
      array[] int  is_fast,
      vector       log_time,

      array[] int  skill_flat,
      array[] int  skill_start,
      array[] int  skill_count,

      matrix       user_skill,
      vector       question_difficulty,
      real         gamma_base,
      real         gamma_fast_delta,
      real         alpha,
      real         beta_time,
      real         mu_log_time,
      real         sigma_log_time
  ) {
    real lp = 0.0;

    for (n in start:end) {
      int  p = player_id[n];
      int  q = question_id[n];

      real skill_eff = sparse_skill_effect(
          user_skill, p, skill_flat, skill_start, skill_count, q
      );

      real gamma     = gamma_base + gamma_fast_delta * is_fast[n];
      real p_skill   = inv_logit(skill_eff - question_difficulty[q]);
      real p_correct = gamma + (alpha - gamma) * p_skill;
      p_correct = fmin(fmax(p_correct, 1e-6), 1.0 - 1e-6);

      lp += bernoulli_lpmf(response_slice[n - start + 1] | p_correct);

      real p_engaged    = inv_logit(beta_time * log_time[n]);
      real rt_lp_engage = normal_lpdf(log_time[n] | mu_log_time, sigma_log_time);
      real rt_lp_guess  = normal_lpdf(log_time[n] | 3.0, 1.5);

      lp += log_mix(p_engaged, rt_lp_engage, rt_lp_guess);
    }
    return lp;
  }

}

// ============================================================================
data {
  int<lower=1> N;
  int<lower=1> n_players;
  int<lower=1> n_questions;
  int<lower=1> n_skills;          // = 9 for the naive Level-1 model

  array[N] int<lower=1, upper=n_players>   player_id;
  array[N] int<lower=1, upper=n_questions> question_id;
  array[N] int<lower=0, upper=1>           response;
  array[N] int<lower=0, upper=1>           is_fast;
  vector[N]                                log_time;

  int<lower=0>                             n_skill_entries;
  array[n_skill_entries] int               skill_flat;
  array[n_questions]     int<lower=1>      skill_start;
  array[n_questions]     int<lower=0>      skill_count;

  vector[n_players]                        age_z;
  array[n_players] int<lower=0, upper=2>   gender;
}

// ============================================================================
transformed data {
  vector[n_players] gender_ind;
  for (p in 1:n_players)
    gender_ind[p] = (gender[p] == 2) ? 1.0 : 0.0;
}

// ============================================================================
parameters {
  vector[n_skills]            mu_skill;
  vector<lower=0>[n_skills]   sigma_skill;

  vector[n_skills]  beta_age;
  vector[n_skills]  beta_gender;

  matrix[n_players, n_skills] user_skill_raw;

  vector[n_questions] question_difficulty;

  real<lower=0,    upper=0.40> gamma_base;
  real<lower=0,    upper=0.15> gamma_fast_delta;
  real<lower=0.60, upper=1.0>  alpha;

  real                      mu_log_time;
  real<lower=0>             sigma_log_time;
  real                      beta_time;
}

// ============================================================================
transformed parameters {
  matrix[n_players, n_skills] user_skill;

  for (p in 1:n_players) {
    vector[n_skills] mu_p = mu_skill
                          + beta_age    * age_z[p]
                          + beta_gender * gender_ind[p];
    user_skill[p] = (mu_p + sigma_skill .* user_skill_raw[p]')';
  }
}

// ============================================================================
model {
  mu_skill    ~ normal(0, 1);
  sigma_skill ~ exponential(1);

  beta_age    ~ normal(0, 0.5);
  beta_gender ~ normal(0, 0.5);

  to_vector(user_skill_raw) ~ std_normal();

  question_difficulty ~ normal(0, 1);

  target += normal_lpdf(gamma_base        | 0.25, 0.08);
  target += normal_lpdf(gamma_fast_delta  | 0.05, 0.05);
  target += normal_lpdf(alpha             | 0.90, 0.05);

  mu_log_time    ~ normal(4.5, 1);
  sigma_log_time ~ exponential(1);
  beta_time      ~ normal(1, 0.5);

  target += reduce_sum(
    partial_log_lik,
    response,
    1,

    player_id,
    question_id,
    is_fast,
    log_time,
    skill_flat,
    skill_start,
    skill_count,

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