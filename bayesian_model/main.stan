data {
    int<lower=1> N;
    int<lower=1> n_players;
    int<lower=1> n_questions;
    int<lower=1> n_skills;
    array[N] int<lower=1, upper=n_players> player_id;
    array[N] int<lower=1, upper=n_questions> question_id;
    array[n_questions] vector[n_skills] question_skill_requirements;
    array[N] int<lower=0, upper=1> response;
}

parameters {
    vector[n_skills] mu_skill;
    real<lower=1e-6> sigma_skill;
    array[n_players] vector[n_skills] user_skill;
    vector[n_questions] question_difficulty;
}

model {
    mu_skill ~ normal(0, 1);
    sigma_skill ~ exponential(1);

    for (p in 1:n_players) {
        user_skill[p] ~ normal(mu_skill, sigma_skill);
    }

    question_difficulty ~ normal(0, 1);

    for (n in 1:N) {
        real skill_effect = dot_product(user_skill[player_id[n]], question_skill_requirements[question_id[n]]);
        real difficulty_effect = question_difficulty[question_id[n]];
        response[n] ~ bernoulli_logit(skill_effect - difficulty_effect);
    }
}
