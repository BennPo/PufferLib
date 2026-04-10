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

#define HORIZON 1024

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

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
    agent->ring_collisions = 0.0f;
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_drone_observations(&env->agents[i], env->observations + i*23);
    }
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->ring_collisions = 0.0f;
    agent->prev_race_progress = 0.0f;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;
    agent->ema_dist = 0.0f;
    agent->ema_vel = 0.0f;
    agent->ema_omega = 0.0f;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;
    agent->buffer_idx = 0;

    init_drone(agent, &env->rng, 0.05f);

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
        agent->prev_race_progress = race_absolute_progress(agent->state.pos, agent->rings_passed, agent->target);
    }
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(&env->rng, env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
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

        bool oob = false;
        bool timeout = (agent->episode_length >= HORIZON);
        float reward = 0.0f;
        if (env->task == RACE) {
            oob = fabsf(agent->state.pos.x) > MARGIN_X
               || fabsf(agent->state.pos.y) > MARGIN_Y
               || fabsf(agent->state.pos.z) > MARGIN_Z;

            int ring_state = check_ring(agent, agent->target);
            if (ring_state == 1) {
                agent->rings_passed += 1;
                agent->buffer_idx = (agent->buffer_idx + 1) % agent->buffer_size;
                set_target_race(agent);
            } else if (ring_state == -1) {
                agent->ring_collisions += 1.0f;
            }

            float current_progress = race_absolute_progress(agent->state.pos, agent->rings_passed, agent->target);
            reward = current_progress - agent->prev_race_progress;
            agent->prev_race_progress = current_progress;
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

        bool reset = oob || timeout;
        env->terminals[i] = reset ? 1.0f : 0.0f;

        if (reset) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent, i);
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
