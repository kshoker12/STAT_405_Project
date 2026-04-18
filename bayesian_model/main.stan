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
    vector[n_skills] population_mean_per_skill;
    vector<lower=0>[n_skills] population_sd_per_skill;
    array[n_players] vector[n_skills] player_skill;
    vector[n_questions] question_difficulty;
}

model {
    population_mean_per_skill ~ normal(0, 5);
    population_sd_per_skill ~ exponential(1);

    for (p in 1:n_players) {
        player_skill[p] ~ normal(population_mean_per_skill, population_sd_per_skill);
    }

    question_difficulty ~ normal(0, 5);

    for (n in 1:N) {
        real skill_effect = dot_product(player_skill[player_id[n]], question_skill_requirements[question_id[n]]);
        real difficulty_effect = question_difficulty[question_id[n]];
        response[n] ~ bernoulli_logit(skill_effect - difficulty_effect);
    }
}
