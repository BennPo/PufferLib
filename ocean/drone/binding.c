#include "drone.h"
#include "render.h"

#define OBS_SIZE 26
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;
    env->task = (int)dict_get(kwargs, "task")->value;
    env->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    env->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    env->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    env->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->race_oob_penalty = dict_get(kwargs, "race_oob_penalty")->value;
    env->ring_collision_penalty = dict_get(kwargs, "ring_collision_penalty")->value;
    env->race_difficulty = dict_get(kwargs, "race_difficulty")->value;
    env->race_min_spacing = dict_get(kwargs, "race_min_spacing")->value;
    env->race_max_spacing = dict_get(kwargs, "race_max_spacing")->value;
    env->race_min_turn_angle = dict_get(kwargs, "race_min_turn_angle")->value;
    env->race_max_turn_angle = dict_get(kwargs, "race_max_turn_angle")->value;
    env->race_min_height_delta = dict_get(kwargs, "race_min_height_delta")->value;
    env->race_max_height_delta = dict_get(kwargs, "race_max_height_delta")->value;
    env->hover_target_dist = dict_get(kwargs, "hover_target_dist")->value;
    env->hover_dist = dict_get(kwargs, "hover_dist")->value;
    env->hover_omega = dict_get(kwargs, "hover_omega")->value;
    env->hover_vel = dict_get(kwargs, "hover_vel")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collisions", log->ring_collision);
    dict_set(out, "collisions", log->collisions);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "ema_dist", log->ema_dist);
    dict_set(out, "ema_vel", log->ema_vel);
    dict_set(out, "ema_omega", log->ema_omega);
}
