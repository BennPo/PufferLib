// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Crazyflie Physical Constants
// https://github.com/arplaboratory/learning-to-fly
#define BASE_MASS 0.027f         // kg
#define BASE_IXX 3.85e-6f        // kgm²
#define BASE_IYY 3.85e-6f        // kgm²
#define BASE_IZZ 5.9675e-6f      // kgm²
#define BASE_ARM_LEN 0.0396f     // m
#define BASE_K_THRUST 3.16e-10f  // thrust coefficient
#define BASE_K_DRAG 0.005964552f // yaw moment constant
#define BASE_GRAVITY 9.81f       // m/s^2
#define BASE_MAX_RPM 21702.0f    // RPM
#define BASE_K_MOT 0.15f         // s (RPM time constant)

#define BASE_K_ANG_DAMP 0.0f // angular damping coefficient
#define BASE_B_DRAG 0.0f     // linear drag coefficient
#define BASE_MAX_VEL 20.0f   // m/s
#define BASE_MAX_OMEGA 20.0f // rad/s

// Simulation properties
#define GRID_X 90.0f
#define GRID_Y 90.0f
#define GRID_Z 10.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define MARGIN_Z (GRID_Z - 1)
#define RING_RADIUS 2.0f
#define V_TARGET 0.05f

// Core Parameters
#define DT 0.002f // 500 Hz
#define ACTION_SUBSTEPS 5
#define ACTION_DT (DT * (float)ACTION_SUBSTEPS) // 100 Hz

#define DT_RNG 0.0f

// Corner to corner distance
#define MAX_DIST                                                                                   \
    sqrtf((2 * GRID_X) * (2 * GRID_X) + (2 * GRID_Y) * (2 * GRID_Y) + (2 * GRID_Z) * (2 * GRID_Z))

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float rings_passed;
    float collisions;
    float oob;
    float ring_collision;
    float timeout;
    float score;
    float perf;
    float ema_dist;
    float ema_vel;
    float ema_omega;
    float race_corner_speed_penalty;
    float race_lookahead_reward;
    float race_speed;
    float n;
};

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat orientation;
    Vec3 normal;
    float radius;
} Target;

typedef struct {
    int course_mode;
    float min_spacing;
    float max_spacing;
} RaceConfig;

typedef struct {
    float max_turn_radians;
    float min_height_delta;
    float max_height_delta;
} RaceCoursePreset;

enum {
    RACE_COURSE_STRAIGHT = 1,
    RACE_COURSE_RANDOM = 2,
};

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float rpms[4]; // motor RPMs
} State;

typedef struct {
    Vec3 vel;         // Derivative of position
    Vec3 v_dot;       // Derivative of velocity
    Quat q_dot;       // Derivative of quaternion
    Vec3 w_dot;       // Derivative of angular velocity
    float rpm_dot[4]; // Derivative of motor RPMs
} StateDerivative;

typedef struct {
    float mass;       // kg
    float ixx;        // kgm^2
    float iyy;        // kgm^2
    float izz;        // kgm^2
    float arm_len;    // m
    float k_thrust;   // thrust coefficient (T = k * rpm^2)
    float k_ang_damp; // angular damping coefficient
    float k_drag;     // yaw moment constant (torque-to-thrust ratio style)
    float b_drag;     // linear drag coefficient
    float gravity;    // m/s^2 (positive, world gravity points -z)
    float max_rpm;    // RPM
    float max_vel;    // m/s (observation clamp)
    float max_omega;  // rad/s (observation clamp)
    float k_mot;      // s (motor RPM time constant)
} Params;

typedef struct {
    // core state and parameters
    State state;
    Params params;
    Vec3 prev_pos;

    // current target
    Target* target;

    // target buffer
    Target* buffer;
    int buffer_idx;
    int buffer_size;

    // logging utils
    float prev_race_progress;
    float episode_return;
    int episode_length;
    float score;
    float collisions;
    int rings_passed;
    float ring_collisions;
    float hover_score;
    float prev_potential;
    float prev_race_alignment;
    float prev_race_segment_progress;
    float hover_ema;
    float ema_dist;
    float ema_vel;
    float ema_omega;
    float race_corner_speed_penalty;
    float race_lookahead_reward;
    float race_speed;
} Drone;

static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float rndf(float a, float b, unsigned int* rng) {
    return a + ((float)rand_r(rng) / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline Quat add_quat(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline Quat scalmul_quat(Quat a, float b) {
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline Vec3 normalize3(Vec3 a) {
    float n = norm3(a);
    if (n <= 1e-6f) {
        return (Vec3){1.0f, 0.0f, 0.0f};
    }
    return scalmul3(a, 1.0f / n);
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * clampf(t, 0.0f, 1.0f);
}

static inline void clamp3(Vec3* vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = (Quat){0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = (Quat){q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

static inline float rpm_hover(const Params* p) {
    // total thrust = m*g = 4 * k_thrust * rpm^2
    return sqrtf((p->mass * p->gravity) / (4.0f * p->k_thrust));
}

static inline float rpm_min_for_centered_hover(const Params* p) {
    // choose min_rpm so that action=0 -> (min+max)/2 == hover
    float min_rpm = 2.0f * rpm_hover(p) - p->max_rpm;
    if (min_rpm < 0.0f) min_rpm = 0.0f;
    if (min_rpm > p->max_rpm) min_rpm = p->max_rpm;
    return min_rpm;
}

static inline void init_drone(Drone* drone, unsigned int* rng, float dr) {
    drone->params.arm_len = BASE_ARM_LEN * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.mass = BASE_MASS * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.ixx = BASE_IXX * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.iyy = BASE_IYY * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.izz = BASE_IZZ * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_thrust = BASE_K_THRUST * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_ang_damp = BASE_K_ANG_DAMP * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_drag = BASE_K_DRAG * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.b_drag = BASE_B_DRAG * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.gravity = BASE_GRAVITY * rndf(0.99f, 1.01f, rng);

    drone->params.max_rpm = BASE_MAX_RPM;
    drone->params.max_vel = BASE_MAX_VEL;
    drone->params.max_omega = BASE_MAX_OMEGA;

    drone->params.k_mot = BASE_K_MOT * rndf(1.0f - dr, 1.0f + dr, rng);

    float hover = rpm_hover(&drone->params);
    for (int i = 0; i < 4; i++)
        drone->state.rpms[i] = hover;

    drone->state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    drone->prev_pos = drone->state.pos;
    drone->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
}

static inline void compute_derivatives(State* state, Params* params, float* actions,
                                       StateDerivative* derivatives) {
    float min_rpm = rpm_min_for_centered_hover(params);

    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        float u = (actions[i] + 1.0f) * 0.5f; // [0,1]
        target_rpms[i] = min_rpm + u * (params->max_rpm - min_rpm);
    }

    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);
    }

    // motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = state->rpms[i];
        if (rpm < 0.0f) rpm = 0.0f;
        T[i] = params->k_thrust * rpm * rpm;
    }

    // body frame net force
    Vec3 F_prop_body = (Vec3){0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame force -> world frame force
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    // world frame linear drag
    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    // linear acceleration
    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    // quaternion rates
    Quat omega_q = (Quat){0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // body frame torques (plus copter)
    // Vec3 Tau_prop;
    // Tau_prop.x = params->arm_len*(T[1] - T[3]);
    // Tau_prop.y = params->arm_len*(T[2] - T[0]);
    // Tau_prop.z = params->k_drag*(T[0] - T[1] + T[2] - T[3]);

    // body frame torques (cross copter)
    // M1=FR, M2=BR, M3=BL, M4=FL
    // https://www.bitcraze.io/documentation/hardware/crazyflie_2_1_brushless/crazyflie_2_1_brushless-datasheet.pdf
    float arm_factor = params->arm_len / sqrtf(2.0f);
    Vec3 Tau_prop;
    Tau_prop.x = arm_factor * ((T[2] + T[3]) - (T[0] + T[1]));
    Tau_prop.y = arm_factor * ((T[1] + T[2]) - (T[0] + T[3]));
    Tau_prop.z = params->k_drag * (-T[0] + T[1] - T[2] + T[3]);

    // torque from angular damping
    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;

    // gyroscopic torque
    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    // angular velocity rates
    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static inline void step(State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    for (int i = 0; i < 4; i++) {
        output->rpms[i] = initial->rpms[i] + deriv->rpm_dot[i] * dt;
    }
    quat_normalize(&output->quat);
}

static inline void rk4_step(State* state, Params* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives(state, params, actions, &k1);

    step(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k2);

    step(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k3);

    step(state, &k3, dt, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] +=
            (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

static inline void move_drone(Drone* drone, float* actions) {
    clamp4(actions, -1.0f, 1.0f);

    for (int s = 0; s < ACTION_SUBSTEPS; s++) {
        rk4_step(&drone->state, &drone->params, actions, DT);

        clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
        clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);

        for (int i = 0; i < 4; i++) {
            drone->state.rpms[i] = clampf(drone->state.rpms[i], 0.0f, drone->params.max_rpm);
        }
    }
}

static inline Target make_race_ring(Vec3 pos, Vec3 normal, float radius) {
    Target ring = (Target){0};
    ring.pos = pos;
    ring.normal = normalize3(normal);
    ring.radius = radius;
    return ring;
}

static inline Vec3 direction_from_yaw_pitch(float yaw, float pitch) {
    float cp = cosf(pitch);
    return normalize3((Vec3){cp * cosf(yaw), cp * sinf(yaw), sinf(pitch)});
}

static inline bool ring_point_in_bounds(Vec3 pos, float clearance) {
    return fabsf(pos.x) <= MARGIN_X - clearance
        && fabsf(pos.y) <= MARGIN_Y - clearance
        && fabsf(pos.z) <= MARGIN_Z - clearance;
}

static inline void make_straight_race(Target* ring_buffer, int num_rings, float min_spacing) {
    float clearance = 2.0f * RING_RADIUS;
    float exit_dist = RING_RADIUS + 4.0f;
    Vec3 dir = {1.0f, 0.0f, 0.0f};

    float min_x = -MARGIN_X + clearance + 0.1f;
    float max_x = MARGIN_X - RING_RADIUS - 0.1f - exit_dist;
    float spacing = min_spacing;
    if (num_rings > 1) {
        spacing = fminf(spacing, (max_x - min_x) / (float)(num_rings - 1));
    }

    float length = spacing * (float)(num_rings - 1);
    float start_x = clampf(-0.5f * length, min_x, max_x - length);
    for (int i = 0; i < num_rings; i++) {
        Vec3 pos = {start_x + spacing * (float)i, 0.0f, 0.0f};
        ring_buffer[i] = make_race_ring(pos, dir, RING_RADIUS);
    }
}

static inline float race_config_min_spacing(RaceConfig config) {
    return fmaxf(2.0f * RING_RADIUS + 1.0f, config.min_spacing);
}

static inline float race_config_max_spacing(RaceConfig config) {
    float min_spacing = race_config_min_spacing(config);
    return fmaxf(min_spacing, config.max_spacing);
}

static inline float race_ring_clearance(void) {
    return 2.0f * RING_RADIUS;
}

static inline float race_planar_turn_angle(Vec3 incoming, Vec3 outgoing) {
    float incoming_xy = sqrtf(incoming.x * incoming.x + incoming.y * incoming.y);
    float outgoing_xy = sqrtf(outgoing.x * outgoing.x + outgoing.y * outgoing.y);
    if (incoming_xy <= 1e-5f || outgoing_xy <= 1e-5f) {
        return 0.0f;
    }

    float denom = incoming_xy * outgoing_xy;
    float turn_dot = clampf((incoming.x * outgoing.x + incoming.y * outgoing.y) / denom, -1.0f, 1.0f);
    float turn_cross = clampf((incoming.x * outgoing.y - incoming.y * outgoing.x) / denom, -1.0f, 1.0f);
    return fabsf(atan2f(turn_cross, turn_dot));
}

static inline RaceCoursePreset race_course_preset(int course_mode) {
    switch (course_mode) {
        case RACE_COURSE_RANDOM:
            return (RaceCoursePreset){
                .max_turn_radians = (float)M_PI * 0.5f - 1e-4f,
                .min_height_delta = 0.25f,
                .max_height_delta = 4.0f,
            };
        case RACE_COURSE_STRAIGHT:
        default:
            return (RaceCoursePreset){
                .max_turn_radians = (float)M_PI / 10.0f,
                .min_height_delta = 0.0f,
                .max_height_delta = 0.5f,
            };
    }
}

static inline void assign_race_ring_normals_open(Target* ring_buffer, int num_rings) {
    if (num_rings <= 0) return;
    if (num_rings == 1) {
        ring_buffer[0] = make_race_ring(ring_buffer[0].pos, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS);
        return;
    }

    for (int i = 0; i < num_rings; i++) {
        Vec3 pos = ring_buffer[i].pos;

        Vec3 normal;
        if (i == 0) {
            normal = normalize3(sub3(ring_buffer[1].pos, pos));
        } else if (i == num_rings - 1) {
            normal = normalize3(sub3(pos, ring_buffer[i - 1].pos));
        } else {
            Vec3 incoming = normalize3(sub3(pos, ring_buffer[i - 1].pos));
            Vec3 outgoing = normalize3(sub3(ring_buffer[i + 1].pos, pos));
            normal = add3(incoming, outgoing);
            if (norm3(normal) <= 1e-5f) {
                normal = outgoing;
            }
        }

        ring_buffer[i] = make_race_ring(pos, normal, RING_RADIUS);
    }
}

static inline bool race_course_is_valid_open(Target* ring_buffer, int num_rings, RaceConfig config, bool check_spacing) {
    if (num_rings <= 0) return false;

    float min_spacing = race_config_min_spacing(config);
    float max_spacing = race_config_max_spacing(config);
    RaceCoursePreset preset = race_course_preset(config.course_mode);
    float max_height_delta = fmaxf(0.0f, preset.max_height_delta);
    float min_visible_height = fminf(fmaxf(0.0f, preset.min_height_delta), max_height_delta);
    float clearance = race_ring_clearance();
    float max_observed_dz = 0.0f;

    for (int i = 0; i < num_rings; i++) {
        Target* ring = &ring_buffer[i];

        if (!ring_point_in_bounds(ring->pos, clearance + 0.05f)) {
            return false;
        }

        float normal_len = norm3(ring->normal);
        if (fabsf(normal_len - 1.0f) > 1e-3f) {
            return false;
        }
    }

    if (num_rings == 1) {
        return true;
    }

    for (int i = 0; i < num_rings - 1; i++) {
        Target* ring = &ring_buffer[i];
        Target* next = &ring_buffer[i + 1];
        Vec3 delta = sub3(next->pos, ring->pos);
        float dist = norm3(delta);
        if (dist <= 1e-5f) {
            return false;
        }

        if (check_spacing && (dist < min_spacing - 1e-3f || dist > max_spacing + 1e-3f)) {
            return false;
        }

        float abs_dz = fabsf(delta.z);
        max_observed_dz = fmaxf(max_observed_dz, abs_dz);
        if (abs_dz > max_height_delta + 1e-3f) {
            return false;
        }
        if (abs_dz > 0.75f * dist + 1e-3f) {
            return false;
        }

        Vec3 outgoing = normalize3(delta);
        if (i > 0) {
            Vec3 incoming = normalize3(sub3(ring->pos, ring_buffer[i - 1].pos));
            if (dot3(incoming, outgoing) <= 1e-4f) {
                return false;
            }
            if (race_planar_turn_angle(incoming, outgoing) > preset.max_turn_radians + 1e-3f) {
                return false;
            }
        }

        if (dot3(delta, ring->normal) <= 1e-3f) {
            return false;
        }
        if (dot3(scalmul3(delta, -1.0f), next->normal) >= -1e-3f) {
            return false;
        }
        if (dot3(ring->normal, next->normal) <= 1e-4f) {
            return false;
        }
    }

    if (num_rings > 1 && min_visible_height > 1e-6f && max_height_delta > 1e-6f
            && max_observed_dz < min_visible_height - 1e-3f) {
        return false;
    }

    return true;
}

static inline bool make_preset_open_race(unsigned int* rng, Target* ring_buffer, int num_rings, RaceConfig config) {
    if (num_rings <= 0) return false;

    RaceCoursePreset preset = race_course_preset(config.course_mode);
    float min_spacing = race_config_min_spacing(config);
    float max_spacing = race_config_max_spacing(config);
    float max_height_delta = fmaxf(0.0f, preset.max_height_delta);
    float max_turn_radians = clampf(preset.max_turn_radians, 0.0f, (float)M_PI * 0.5f - 1e-4f);
    float min_height_delta = fminf(fmaxf(0.0f, preset.min_height_delta), max_height_delta);
    bool require_visible_height = min_height_delta > 1e-6f;
    float clearance = race_ring_clearance();
    float xy_limit = fminf(MARGIN_X - clearance - 0.1f, MARGIN_Y - clearance - 0.1f);
    float z_limit = fmaxf(0.0f, MARGIN_Z - clearance - 0.1f);
    float z_delta_limit = fminf(max_height_delta, z_limit * 1.25f);
    if (xy_limit <= 1e-6f) return false;

    if (num_rings == 1) {
        ring_buffer[0] = make_race_ring((Vec3){0.0f, 0.0f, 0.0f}, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS);
        return true;
    }

    for (int course_attempt = 0; course_attempt < 512; course_attempt++) {
        float start_window = fminf(xy_limit * 0.35f, 0.5f * min_spacing * (float)num_rings);
        Vec3 pos = {
            rndf(-start_window, start_window, rng),
            rndf(-start_window, start_window, rng),
            rndf(-0.35f * z_limit, 0.35f * z_limit, rng),
        };
        float yaw = rndf(-(float)M_PI, (float)M_PI, rng);
        Vec3 dir = direction_from_yaw_pitch(yaw, 0.0f);
        ring_buffer[0] = make_race_ring(pos, dir, RING_RADIUS);

        bool course_valid = true;
        bool forced_height = false;
        for (int i = 1; i < num_rings; i++) {
            bool need_height = require_visible_height && !forced_height
                && max_height_delta > 1e-6f && i >= num_rings / 2;
            bool placed_ring = false;

            for (int attempt = 0; attempt < 128; attempt++) {
                float spacing = rndf(min_spacing, max_spacing, rng);
                // Negative yaw turns left by convention; positive yaw turns right.
                float turn = rndf(-max_turn_radians, max_turn_radians, rng);
                float candidate_yaw = yaw + turn;
                float local_max_dz = fminf(z_delta_limit, spacing * 0.60f);
                float dz = 0.0f;
                if (local_max_dz > 1e-6f) {
                    float min_dz = need_height ? fminf(min_height_delta, local_max_dz) : 0.0f;
                    float abs_dz = rndf(min_dz, local_max_dz, rng);
                    float sign = rndf(0.0f, 1.0f, rng) < 0.5f ? -1.0f : 1.0f;
                    dz = sign * abs_dz;
                    if (pos.z + dz > z_limit) dz = -abs_dz;
                    if (pos.z + dz < -z_limit) dz = abs_dz;
                }

                float pitch = asinf(clampf(dz / fmaxf(spacing, 1e-6f), -0.75f, 0.75f));
                Vec3 candidate_dir = direction_from_yaw_pitch(candidate_yaw, pitch);
                if (dot3(dir, candidate_dir) <= 1e-4f) {
                    continue;
                }

                Vec3 next = add3(pos, scalmul3(candidate_dir, spacing));
                if (!ring_point_in_bounds(next, clearance + 0.05f)) {
                    continue;
                }

                float segment_dz = fabsf(next.z - pos.z);
                if (require_visible_height && segment_dz >= min_height_delta - 1e-3f) {
                    forced_height = true;
                }
                pos = next;
                dir = candidate_dir;
                yaw = candidate_yaw;
                ring_buffer[i] = make_race_ring(pos, dir, RING_RADIUS);
                placed_ring = true;
                break;
            }

            if (!placed_ring) {
                course_valid = false;
                break;
            }
        }

        if (!course_valid) {
            continue;
        }

        assign_race_ring_normals_open(ring_buffer, num_rings);
        if (race_course_is_valid_open(ring_buffer, num_rings, config, true)) {
            return true;
        }
    }

    return false;
}

static inline void reset_rings(unsigned int* rng, Target* ring_buffer, int num_rings, RaceConfig config) {
    if (num_rings <= 0) return;

    float min_spacing = race_config_min_spacing(config);
    if (!make_preset_open_race(rng, ring_buffer, num_rings, config)) {
        make_straight_race(ring_buffer, num_rings, min_spacing);
    }
}

static inline bool race_target_is_final(Drone* agent) {
    return agent->buffer_size <= 1 || agent->buffer_idx >= agent->buffer_size - 1;
}

static inline int check_ring(Drone* drone, Target* ring) {
    // previous dot product negative if on the 'entry' side of the ring's plane
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);
    float new_dot = dot3(sub3(drone->state.pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    // if we have crossed the plane of the ring
    if (valid_dir || invalid_dir) {
        // find intesection with ring's plane
        Vec3 dir = sub3(drone->state.pos, drone->prev_pos);
        float denom = dot3(ring->normal, dir);
        if (fabsf(denom) < 1e-9f) return 0;

        float t = -prev_dot / denom;
        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float dist = norm3(sub3(intersection, ring->pos));

        if (dist < (ring->radius - 0.5f) && valid_dir) {
            return 1;
        } else if (dist < ring->radius + 0.5f) {
            return -1;
        }
    }

    return 0;
}

static inline float race_aperture_alignment(Drone* drone, Target* ring) {
    Vec3 offset = sub3(drone->state.pos, ring->pos);
    float signed_plane = dot3(offset, ring->normal);
    float entry_side_gate = signed_plane <= 0.0f ? 1.0f : 0.0f;
    float near_plane_gate = clampf(1.0f - fabsf(signed_plane) / 8.0f, 0.0f, 1.0f);

    Vec3 radial = sub3(offset, scalmul3(ring->normal, signed_plane));
    float safe_radius = fmaxf(ring->radius - 0.5f, 1e-3f);
    float centeredness = clampf(1.0f - norm3(radial) / safe_radius, 0.0f, 1.0f);

    float speed = norm3(drone->state.vel);
    float forward_direction = 0.0f;
    if (speed > 1e-6f) {
        Vec3 vel_dir = scalmul3(drone->state.vel, 1.0f / speed);
        forward_direction = clampf(dot3(vel_dir, ring->normal), 0.0f, 1.0f);
    }

    return entry_side_gate * near_plane_gate * centeredness * forward_direction;
}

static inline float race_lookahead_segment_progress(Vec3 pos, Target* current, Target* next) {
    if (current == NULL || next == NULL) {
        return 0.0f;
    }

    Vec3 segment = sub3(next->pos, current->pos);
    float segment_len = norm3(segment);
    if (segment_len <= 1e-6f) {
        return 0.0f;
    }

    Vec3 segment_dir = scalmul3(segment, 1.0f / segment_len);
    float progress = dot3(sub3(pos, current->pos), segment_dir) / segment_len;
    return clampf(progress, 0.0f, 1.0f);
}

static inline float race_lookahead_gate_quality(Drone* drone, Target* ring, float gate_dist) {
    if (drone == NULL || ring == NULL || gate_dist <= 0.0f) {
        return 0.0f;
    }

    Vec3 offset = sub3(drone->state.pos, ring->pos);
    float signed_plane = dot3(offset, ring->normal);
    float near_plane_gate = clampf(1.0f - fabsf(signed_plane) / gate_dist, 0.0f, 1.0f);

    Vec3 radial = sub3(offset, scalmul3(ring->normal, signed_plane));
    float safe_radius = fmaxf(ring->radius - 0.5f, 1e-3f);
    float centeredness = clampf(1.0f - norm3(radial) / safe_radius, 0.0f, 1.0f);

    float speed = norm3(drone->state.vel);
    float forward_through_current_ring = 0.0f;
    if (speed > 1e-6f) {
        Vec3 vel_dir = scalmul3(drone->state.vel, 1.0f / speed);
        forward_through_current_ring = clampf(dot3(vel_dir, ring->normal), 0.0f, 1.0f);
    }

    return near_plane_gate * centeredness * forward_through_current_ring;
}

float hover_potential(Drone* agent, float hover_dist, float hover_omega, float hover_vel) {
    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    float d = 1.0f / (1.0f + dist / hover_dist);
    float v = 1.0f / (1.0f + vel / hover_vel);
    float w = 1.0f / (1.0f + omega / hover_omega);

    return d * (0.7f + 0.15f * v + 0.15f * w);
}

float check_hover(Drone* agent, float hover_dist, float hover_omega, float hover_vel) {
    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    float d = dist / (hover_dist * 10.0f);
    float v = vel / (hover_vel * 10.0f);
    float w = omega / (hover_omega * 10.0f);

    float score = 1.0f - 0.7f * d - 0.15f * v - 0.15f * w;
    return score > 0.0f ? score : 0.0f;
}

static inline Target* next_race_target(Drone* agent) {
    if (agent->buffer == NULL || agent->buffer_size <= 0) {
        return agent->target;
    }
    if (race_target_is_final(agent)) {
        return NULL;
    }

    return &agent->buffer[agent->buffer_idx + 1];
}

static inline float compute_race_corner_speed_penalty(Drone* agent, Target* ring, float coef, float gate_dist) {
    if (coef <= 0.0f || gate_dist <= 0.0f || agent->buffer == NULL || agent->buffer_size <= 1) {
        return 0.0f;
    }

    Target* next = next_race_target(agent);
    if (next == NULL) {
        return 0.0f;
    }

    Vec3 next_dir = normalize3(sub3(next->pos, ring->pos));
    float dot = clampf(dot3(ring->normal, next_dir), -1.0f, 1.0f);
    float raw_turn = 0.5f * (1.0f - dot);
    float turn_severity = clampf((raw_turn - 0.15f) / 0.85f, 0.0f, 1.0f);
    if (turn_severity <= 0.0f) {
        return 0.0f;
    }

    float signed_plane = dot3(sub3(agent->state.pos, ring->pos), ring->normal);
    if (signed_plane > 0.0f) {
        return 0.0f;
    }

    float near_ring_gate = clampf(1.0f + signed_plane / gate_dist, 0.0f, 1.0f);
    if (near_ring_gate <= 0.0f) {
        return 0.0f;
    }

    const float straight_speed = 12.0f;
    const float corner_speed = 4.0f;
    float desired_speed = lerpf(straight_speed, corner_speed, turn_severity);
    float speed = norm3(agent->state.vel);
    float excess = fmaxf(0.0f, speed - desired_speed);
    float normalized_excess = excess / fmaxf(agent->params.max_vel, 1e-6f);
    return coef * near_ring_gate * turn_severity * normalized_excess * normalized_excess;
}

static inline float race_target_max_dist(void) {
    return sqrtf((2.0f * MARGIN_X) * (2.0f * MARGIN_X)
               + (2.0f * MARGIN_Y) * (2.0f * MARGIN_Y)
               + (2.0f * MARGIN_Z) * (2.0f * MARGIN_Z));
}

static inline float race_target_proximity(Vec3 pos, Target* ring) {
    float max_dist = race_target_max_dist();
    if (max_dist <= 1e-6f) {
        return 1.0f;
    }

    float dist = norm3(sub3(pos, ring->pos));
    // Choose k so a distance of 5 units maps to 0.5 proximity.
    const float log_k = (max_dist - 10.0f) / 25.0f;
    float proximity = 1.0f - log1pf(log_k * dist) / log1pf(log_k * max_dist);
    return clampf(proximity, 0.0f, 1.0f);
}

static inline float race_absolute_progress(Vec3 pos, int rings_passed, Target* ring) {
    return (float)rings_passed + race_target_proximity(pos, ring);
}

void compute_drone_observations(Drone* agent, float* observations) {
    int idx = 0;

    // choose the hemisphere with w >= 0
    // to avoid observation sign ambiguity
    Quat q = agent->state.quat;
    //if (q.w < 0.0f) {q.w=-q.w; q.x=-q.x; q.y=-q.y; q.z=-q.z;}

    Quat q_inv = quat_inverse(q);
    Vec3 linear_vel_body = quat_rotate(q_inv, agent->state.vel);
    Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);
    Vec3 to_next_target = (Vec3){0.0f, 0.0f, 0.0f};
    if (agent->buffer != NULL && agent->buffer_size > 1) {
        Target* next_target = next_race_target(agent);
        if (next_target != NULL) {
            Vec3 to_next_target_world = sub3(next_target->pos, agent->state.pos);
            to_next_target = quat_rotate(q_inv, to_next_target_world);
        }
    }

    // we should probably clamp the overall velocity
    float denom = agent->params.max_vel * 1.7320508f; // sqrt(3)
    observations[idx++] = linear_vel_body.x / denom;
    observations[idx++] = linear_vel_body.y / denom;
    observations[idx++] = linear_vel_body.z / denom;

    observations[idx++] = agent->state.omega.x / agent->params.max_omega;
    observations[idx++] = agent->state.omega.y / agent->params.max_omega;
    observations[idx++] = agent->state.omega.z / agent->params.max_omega;

    observations[idx++] = q.w;
    observations[idx++] = q.x;
    observations[idx++] = q.y;
    observations[idx++] = q.z;

    // this is body frame so we have to be careful about scaling
    // because distances are relative to the drone orientation
    observations[idx++] = tanhf(to_target.x * 0.1f);
    observations[idx++] = tanhf(to_target.y * 0.1f);
    observations[idx++] = tanhf(to_target.z * 0.1f);

    observations[idx++] = tanhf(to_next_target.x * 0.1f);
    observations[idx++] = tanhf(to_next_target.y * 0.1f);
    observations[idx++] = tanhf(to_next_target.z * 0.1f);

    Vec3 normal_body = quat_rotate(q_inv, agent->target->normal);
    observations[idx++] = normal_body.x;
    observations[idx++] = normal_body.y;
    observations[idx++] = normal_body.z;

    // rpms should always be last in the obs
    observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;

    // Signed world position gives direct boundary context across all tasks.
    observations[idx++] = clampf(agent->state.pos.x / MARGIN_X, -1.0f, 1.0f);
    observations[idx++] = clampf(agent->state.pos.y / MARGIN_Y, -1.0f, 1.0f);
    observations[idx++] = clampf(agent->state.pos.z / MARGIN_Z, -1.0f, 1.0f);
}
