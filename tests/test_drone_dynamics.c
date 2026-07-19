#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/drone/dronelib.h"

static Params test_params(void) {
    Params params = {
        .mass = BASE_MASS,
        .ixx = BASE_IXX,
        .iyy = BASE_IYY,
        .izz = BASE_IZZ,
        .arm_len = BASE_ARM_LEN,
        .k_thrust = BASE_K_THRUST,
        .k_ang_damp = 2.0e-6f,
        .k_drag = BASE_K_DRAG,
        .gravity = BASE_GRAVITY,
        .max_rpm = BASE_MAX_RPM,
        .k_mot = BASE_K_MOT,
    };
    return params;
}

static State hover_state(const Params* params) {
    State state = {.quat = {1.0f, 0.0f, 0.0f, 0.0f}};
    float hover = rpm_hover(params);
    for (int i = 0; i < 4; i++) state.rpms[i] = hover;
    return state;
}

static void test_xy_torque_symmetry(void) {
    Params params = test_params();
    State roll = hover_state(&params);
    State pitch = hover_state(&params);
    float delta = 1000.0f;

    // Roll: left motors (M3/M4) above hover, right motors (M1/M2) below.
    roll.rpms[0] -= delta;
    roll.rpms[1] -= delta;
    roll.rpms[2] += delta;
    roll.rpms[3] += delta;

    // Pitch: rear motors (M2/M3) above hover, front motors (M1/M4) below.
    pitch.rpms[0] -= delta;
    pitch.rpms[1] += delta;
    pitch.rpms[2] += delta;
    pitch.rpms[3] -= delta;

    float actions[4] = {0};
    StateDerivative roll_deriv = {0};
    StateDerivative pitch_deriv = {0};
    compute_derivatives(&roll, &params, actions, &roll_deriv);
    compute_derivatives(&pitch, &params, actions, &pitch_deriv);

    assert(fabsf(roll_deriv.w_dot.x - pitch_deriv.w_dot.y) < 1e-4f);
    assert(fabsf(roll_deriv.w_dot.y) < 1e-6f);
    assert(fabsf(pitch_deriv.w_dot.x) < 1e-6f);
}

static void test_angular_damping_opposes_rotation(void) {
    Params params = test_params();
    State state = hover_state(&params);
    state.omega.x = 5.0f;

    float actions[4] = {0};
    StateDerivative deriv = {0};
    compute_derivatives(&state, &params, actions, &deriv);

    assert(deriv.w_dot.x < 0.0f);
    assert(fabsf(deriv.w_dot.x + params.k_ang_damp * 5.0f / params.ixx) < 1e-5f);
}

int main(void) {
    test_xy_torque_symmetry();
    test_angular_damping_opposes_rotation();
    puts("drone dynamics checks passed");
    return 0;
}
