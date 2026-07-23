// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

#define HORIZON 8192

typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    DroneTask task;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    // reward scaling
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;
    float angular_damping;
    float race_oob_penalty;
    float ring_collision_penalty;
    float race_clean_pass_bonus;
    float race_aperture_alignment_coef;
    float race_corner_speed_control_coef;
    float race_corner_speed_gate_dist;
    float race_lookahead_segment_coef;
    float race_lookahead_gate_dist;
    int race_course_mode;
    float race_min_spacing;
    float race_max_spacing;
    // hover task parameters
    float hover_target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;
};

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
        env->agents[i].buffer_idx = 0;
    }

    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout) {
    Drone* agent = &env->agents[idx];
    float race_progress = 0.0f;
    if (agent->buffer_size > 0) {
        race_progress = agent->prev_race_progress / (float)agent->buffer_size;
    }

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;
    env->log.ring_collision += agent->ring_collisions;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;

    if (env->task == RACE) {
        env->log.score += agent->rings_passed;
        env->log.perf += race_progress;
    } else {
        env->log.score += agent->hover_score;
        env->log.perf += agent->hover_ema;
    }
    env->log.rings_passed += agent->rings_passed;
    env->log.ema_dist += agent->ema_dist;
    env->log.ema_vel += agent->ema_vel;
    env->log.ema_omega += agent->ema_omega;
    env->log.ema_abs_omega_x += agent->ema_abs_omega_x;
    env->log.ema_abs_omega_y += agent->ema_abs_omega_y;
    env->log.ema_abs_omega_z += agent->ema_abs_omega_z;
    env->log.omega_x_saturation += agent->omega_x_saturation / (float)agent->episode_length;
    env->log.ema_abs_roll_command += agent->ema_abs_roll_command;
    env->log.race_corner_speed_penalty += agent->race_corner_speed_penalty;
    env->log.race_lookahead_reward += agent->race_lookahead_reward;
    if (agent->episode_length > 0) {
        env->log.race_speed += agent->race_speed / (float)agent->episode_length;
    }

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
    agent->ring_collisions = 0.0f;
    agent->race_corner_speed_penalty = 0.0f;
    agent->race_lookahead_reward = 0.0f;
    agent->race_speed = 0.0f;
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_drone_observations(&env->agents[i], env->observations + i*29);
    }
}

void reset_agent(DroneEnv* env, Drone* agent) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->ring_collisions = 0.0f;
    agent->prev_race_progress = 0.0f;
    agent->prev_race_alignment = 0.0f;
    agent->prev_race_segment_progress = 0.0f;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;
    agent->ema_dist = 0.0f;
    agent->ema_vel = 0.0f;
    agent->ema_omega = 0.0f;
    agent->ema_abs_omega_x = 0.0f;
    agent->ema_abs_omega_y = 0.0f;
    agent->ema_abs_omega_z = 0.0f;
    agent->omega_x_saturation = 0.0f;
    agent->ema_abs_roll_command = 0.0f;
    agent->race_corner_speed_penalty = 0.0f;
    agent->race_lookahead_reward = 0.0f;
    agent->race_speed = 0.0f;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;
    agent->buffer_idx = 0;

    init_drone(agent, &env->rng, 0.05f);
    agent->params.k_ang_damp = env->angular_damping;

    if (env->task == RACE) {
        Target* start_ring = &env->ring_buffer[0];
        Vec3 start = sub3(start_ring->pos, scalmul3(start_ring->normal, 4.0f));
        Vec3 jitter = {rndf(-1.0f, 1.0f, &env->rng), rndf(-1.0f, 1.0f, &env->rng), rndf(-0.5f, 0.5f, &env->rng)};
        start = add3(start, jitter);
        agent->state.pos = (Vec3){
            clampf(start.x, -MARGIN_X, MARGIN_X),
            clampf(start.y, -MARGIN_Y, MARGIN_Y),
            clampf(start.z, -MARGIN_Z, MARGIN_Z),
        };
    } else {
        agent->state.pos =
            (Vec3){rndf(-MARGIN_X, MARGIN_X, &env->rng), rndf(-MARGIN_Y, MARGIN_Y, &env->rng), rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
    }

    agent->prev_pos = agent->state.pos;
    if (env->task != RACE) {
        agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
    }
}

void sync_agent_progress(DroneEnv* env, Drone* agent) {
    if (env->task == RACE) {
        agent->prev_race_progress =
            race_absolute_progress(agent->state.pos, agent->rings_passed, agent->target);
        agent->prev_race_alignment = race_aperture_alignment(agent, agent->target);
        agent->prev_race_segment_progress = race_lookahead_segment_progress(
            agent->state.pos, agent->target, next_race_target(agent));
    }
}

RaceConfig race_config(DroneEnv* env) {
    float min_spacing = env->race_min_spacing;
    float max_spacing = env->race_max_spacing;
    if (env->race_course_mode == RACE_COURSE_EXTREME) {
        min_spacing = 12.0f;
        max_spacing = 18.0f;
    }

    return (RaceConfig){
        .course_mode = env->race_course_mode,
        .min_spacing = min_spacing,
        .max_spacing = max_spacing,
    };
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(&env->rng, env->ring_buffer, env->max_rings, race_config(env));
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent);
        set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
        sync_agent_progress(env, agent);
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, &env->actions[4 * i]);
        agent->episode_length++;

        const float telemetry_alpha = 0.01f;
        agent->ema_abs_omega_x = (1.0f - telemetry_alpha) * agent->ema_abs_omega_x
            + telemetry_alpha * fabsf(agent->state.omega.x);
        agent->ema_abs_omega_y = (1.0f - telemetry_alpha) * agent->ema_abs_omega_y
            + telemetry_alpha * fabsf(agent->state.omega.y);
        agent->ema_abs_omega_z = (1.0f - telemetry_alpha) * agent->ema_abs_omega_z
            + telemetry_alpha * fabsf(agent->state.omega.z);
        if (fabsf(agent->state.omega.x) >= 0.99f * agent->params.max_omega) {
            agent->omega_x_saturation += 1.0f;
        }
        float* action = &env->actions[4 * i];
        float roll_command = (action[2] + action[3]) - (action[0] + action[1]);
        agent->ema_abs_roll_command = (1.0f - telemetry_alpha) * agent->ema_abs_roll_command
            + telemetry_alpha * fabsf(roll_command);

        bool oob = false;
        bool timeout = (agent->episode_length >= HORIZON);
        bool ring_collision_reset = false;
        bool race_complete = false;
        float reward = 0.0f;
        if (env->task == RACE) {
            oob = fabsf(agent->state.pos.x) > MARGIN_X
               || fabsf(agent->state.pos.y) > MARGIN_Y
               || fabsf(agent->state.pos.z) > MARGIN_Z;

            Target* current_target = agent->target;
            Target* next_target = next_race_target(agent);
            int ring_state = check_ring(agent, current_target);
            float current_alignment = race_aperture_alignment(agent, current_target);
            float current_segment_progress = race_lookahead_segment_progress(
                agent->state.pos, current_target, next_target);
            float current_gate_quality = race_lookahead_gate_quality(
                agent, current_target, env->race_lookahead_gate_dist);
            float current_speed = norm3(agent->state.vel);
            agent->race_speed += current_speed;

            bool apply_lookahead = next_target != NULL
                && env->race_lookahead_segment_coef > 0.0f
                && !oob
                && !timeout
                && ring_state != -1;
            if (apply_lookahead) {
                float lookahead_reward = env->race_lookahead_segment_coef
                    * current_gate_quality
                    * (current_segment_progress - agent->prev_race_segment_progress);
                reward += lookahead_reward;
                agent->race_lookahead_reward += lookahead_reward;
            }
            agent->prev_race_segment_progress = current_segment_progress;

            if (ring_state == 1) {
                agent->rings_passed += 1;
                if (race_target_is_final(agent)) {
                    race_complete = true;
                } else {
                    agent->buffer_idx += 1;
                    set_target_race(agent);
                    agent->prev_race_segment_progress = race_lookahead_segment_progress(
                        agent->state.pos, agent->target, next_race_target(agent));
                }
                reward += env->race_clean_pass_bonus;
            } else if (ring_state == -1) {
                agent->ring_collisions += 1.0f;
                ring_collision_reset = true;
            }

            float current_progress = race_complete
                ? (float)agent->buffer_size
                : race_absolute_progress(agent->state.pos, agent->rings_passed, agent->target);
            reward += current_progress - agent->prev_race_progress;
            agent->prev_race_progress = current_progress;

            if (ring_state == 0 && !oob && !timeout) {
                reward += env->race_aperture_alignment_coef * (current_alignment - agent->prev_race_alignment);
                agent->prev_race_alignment = current_alignment;
                float speed_penalty = compute_race_corner_speed_penalty(
                    agent, current_target, env->race_corner_speed_control_coef,
                    env->race_corner_speed_gate_dist);
                reward -= speed_penalty;
                agent->race_corner_speed_penalty += speed_penalty;
            } else if (ring_state == 1 && !race_complete) {
                agent->prev_race_alignment = race_aperture_alignment(agent, agent->target);
            }

            if (oob) {
                reward -= env->race_oob_penalty;
            }
            if (ring_collision_reset) {
                reward -= env->ring_collision_penalty;
            }
        } else {
            oob = norm3(sub3(agent->target->pos, agent->state.pos)) > (env->hover_target_dist + 1.0f);

            float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));
            float omega = norm3(agent->state.omega);
            float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));

            reward = env->alpha_dist * (prev_dist - curr_dist)
                   + env->alpha_hover * curr
                   + env->alpha_shaping * (curr - agent->prev_potential)
                   - env->alpha_omega * omega;

            agent->prev_potential = curr;

            float h = check_hover(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            agent->hover_score += h;
            agent->hover_ema = (1.0f - 0.02f) * agent->hover_ema + 0.02f * h;
            agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * curr_dist;
            agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * norm3(agent->state.vel);
            agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
        }

        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool reset = oob || timeout || ring_collision_reset || race_complete;
        env->terminals[i] = reset ? 1.0f : 0.0f;

        if (reset) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent);
            set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
            sync_agent_progress(env, agent);
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].target);
    }

    free(env->agents);
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
}
