#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/drone/dronelib.h"

static void assert_near(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-6f);
}

static void assert_clean_center_pass(Target* ring) {
    Drone drone = {0};
    drone.prev_pos = sub3(ring->pos, scalmul3(ring->normal, 1.0f));
    drone.state.pos = add3(ring->pos, scalmul3(ring->normal, 1.0f));

    int state = check_ring(&drone, ring);
    assert(state == 1);
}

static void assert_course(Target* rings, int num_rings, RaceConfig config) {
    assert(race_course_is_valid_open(rings, num_rings, config, true));

    for (int i = 0; i < num_rings; i++) {
        assert_clean_center_pass(&rings[i]);
    }

    if (config.course_mode == RACE_COURSE_EXTREME) {
        float min_spacing = race_config_min_spacing(config);
        float max_spacing = race_config_max_spacing(config);
        for (int i = 0; i < num_rings - 1; i++) {
            float spacing = norm3(sub3(rings[i + 1].pos, rings[i].pos));
            assert(spacing >= min_spacing - 1e-3f);
            assert(spacing <= max_spacing + 1e-3f);
        }
        return;
    }

    RaceCoursePreset preset = race_course_preset(config.course_mode);
    float max_dz = 0.0f;

    if (num_rings > 1) {
        Vec3 first_out = normalize3(sub3(rings[1].pos, rings[0].pos));
        assert(dot3(first_out, rings[0].normal) > 0.999f);

        Vec3 final_in = normalize3(sub3(rings[num_rings - 1].pos, rings[num_rings - 2].pos));
        assert(dot3(final_in, rings[num_rings - 1].normal) > 0.999f);
    }

    for (int i = 0; i < num_rings - 1; i++) {
        Vec3 delta = sub3(rings[i + 1].pos, rings[i].pos);
        float abs_dz = fabsf(delta.z);
        max_dz = fmaxf(max_dz, abs_dz);
        assert(abs_dz <= preset.max_height_delta + 1e-3f);

        if (i > 0) {
            Vec3 incoming = normalize3(sub3(rings[i].pos, rings[i - 1].pos));
            Vec3 outgoing = normalize3(delta);
            Vec3 expected = normalize3(add3(incoming, outgoing));
            assert(dot3(expected, rings[i].normal) > 0.999f);

            float turn_angle = race_planar_turn_angle(incoming, outgoing);
            assert(turn_angle <= preset.max_turn_radians + 1e-3f);
        }
    }

    assert(max_dz >= preset.min_height_delta - 1e-3f);
}

static void assert_extreme_distribution(RaceConfig config) {
    bool saw_positive_dz = false;
    bool saw_negative_dz = false;
    bool saw_small_dz = false;
    bool saw_large_dz = false;
    bool saw_backward_ring = false;
    bool saw_opposed_normals = false;

    for (unsigned int seed = 1; seed <= 512; seed++) {
        Target rings[10] = {0};
        unsigned int rng = seed * 2654435761u;
        reset_rings(&rng, rings, 10, config);
        assert_course(rings, 10, config);

        for (int i = 0; i < 9; i++) {
            Vec3 segment = normalize3(sub3(rings[i + 1].pos, rings[i].pos));
            float dz = rings[i + 1].pos.z - rings[i].pos.z;
            saw_positive_dz |= dz > 1.0f;
            saw_negative_dz |= dz < -1.0f;
            saw_small_dz |= fabsf(dz) < 0.25f;
            saw_large_dz |= fabsf(dz) > 4.0f;
            saw_backward_ring |= dot3(segment, rings[i + 1].normal) < 0.0f;
            saw_opposed_normals |= dot3(rings[i].normal, rings[i + 1].normal) < 0.0f;
        }
    }

    assert(saw_positive_dz);
    assert(saw_negative_dz);
    assert(saw_small_dz);
    assert(saw_large_dz);
    assert(saw_backward_ring);
    assert(saw_opposed_normals);
}

static void assert_extreme_accepts_unrestricted_geometry(RaceConfig config) {
    Target rings[3] = {
        make_race_ring((Vec3){0.0f, 0.0f, 0.0f}, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS),
        make_race_ring((Vec3){16.0f, 0.0f, 0.0f}, (Vec3){-1.0f, 0.0f, 0.0f}, RING_RADIUS),
        make_race_ring((Vec3){0.0f, 0.0f, 0.0f}, (Vec3){0.0f, 0.0f, 1.0f}, RING_RADIUS),
    };
    assert(race_course_is_valid_open(rings, 3, config, true));

    RaceConfig constrained = config;
    constrained.course_mode = RACE_COURSE_RANDOM;
    assert(!race_course_is_valid_open(rings, 3, constrained, true));
}

static void assert_final_ring_is_finish(Target* rings, int num_rings) {
    Drone drone = {0};
    drone.buffer = rings;
    drone.buffer_size = num_rings;
    drone.buffer_idx = num_rings - 1;
    assert(race_target_is_final(&drone));
    assert(next_race_target(&drone) == NULL);

    if (num_rings > 1) {
        drone.buffer_idx = num_rings - 2;
        assert(!race_target_is_final(&drone));
        assert(next_race_target(&drone) == &rings[num_rings - 1]);
    }
}

static void assert_lookahead_helpers(void) {
    // Use a simple straight course so the expected lookahead values are exact.
    Target rings[3] = {
        make_race_ring((Vec3){0.0f, 0.0f, 0.0f}, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS),
        make_race_ring((Vec3){10.0f, 0.0f, 0.0f}, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS),
        make_race_ring((Vec3){20.0f, 0.0f, 0.0f}, (Vec3){1.0f, 0.0f, 0.0f}, RING_RADIUS),
    };
    const float gate_dist = 8.0f;

    Drone drone = {0};
    drone.state.pos = (Vec3){0.1f, 0.0f, 0.0f};
    drone.state.vel = (Vec3){5.0f, 0.0f, 0.0f};
    assert(race_lookahead_gate_quality(&drone, &rings[0], gate_dist) > 0.9f);

    drone.state.pos = (Vec3){0.1f, RING_RADIUS, 0.0f};
    assert_near(race_lookahead_gate_quality(&drone, &rings[0], gate_dist), 0.0f);

    drone.state.pos = (Vec3){9.0f, 0.0f, 0.0f};
    assert_near(race_lookahead_gate_quality(&drone, &rings[0], gate_dist), 0.0f);

    drone.state.pos = (Vec3){0.1f, 0.0f, 0.0f};
    drone.state.vel = (Vec3){-5.0f, 0.0f, 0.0f};
    assert_near(race_lookahead_gate_quality(&drone, &rings[0], gate_dist), 0.0f);

    drone.state.vel = (Vec3){0.0f, 5.0f, 0.0f};
    assert_near(race_lookahead_gate_quality(&drone, &rings[0], gate_dist), 0.0f);

    float p_before = race_lookahead_segment_progress((Vec3){-1.0f, 0.0f, 0.0f}, &rings[0], &rings[1]);
    float p_start = race_lookahead_segment_progress(rings[0].pos, &rings[0], &rings[1]);
    float p_mid = race_lookahead_segment_progress((Vec3){5.0f, 0.0f, 0.0f}, &rings[0], &rings[1]);
    float p_end = race_lookahead_segment_progress(rings[1].pos, &rings[0], &rings[1]);
    float p_after = race_lookahead_segment_progress((Vec3){11.0f, 0.0f, 0.0f}, &rings[0], &rings[1]);
    assert_near(p_before, 0.0f);
    assert_near(p_start, 0.0f);
    assert_near(p_mid, 0.5f);
    assert_near(p_end, 1.0f);
    assert_near(p_after, 1.0f);
    assert_near(race_lookahead_segment_progress(rings[0].pos, &rings[0], NULL), 0.0f);
    assert_near(race_lookahead_segment_progress(rings[0].pos, NULL, &rings[1]), 0.0f);

    drone.state.pos = (Vec3){0.25f, 0.0f, 0.0f};
    drone.state.vel = (Vec3){5.0f, 0.0f, 0.0f};
    drone.prev_race_segment_progress = race_lookahead_segment_progress(drone.state.pos, &rings[0], &rings[1]);
    drone.state.pos = (Vec3){0.5f, 0.0f, 0.0f};
    float old_progress = race_lookahead_segment_progress(drone.state.pos, &rings[0], &rings[1]);
    assert(old_progress - drone.prev_race_segment_progress > 0.0f);

    drone.prev_race_segment_progress = race_lookahead_segment_progress(drone.state.pos, &rings[1], &rings[2]);
    float reset_progress = race_lookahead_segment_progress(drone.state.pos, &rings[1], &rings[2]);
    assert_near(reset_progress, drone.prev_race_segment_progress);
}

static void assert_generated_courses(RaceConfig config) {
    const int ring_counts[] = {5, 10, 16};
    for (int c = 0; c < 3; c++) {
        int num_rings = ring_counts[c];
        for (unsigned int seed = 1; seed <= 512; seed++) {
            Target rings[32] = {0};
            unsigned int rng = seed * 2654435761u + (unsigned int)num_rings;
            reset_rings(&rng, rings, num_rings, config);
            assert_course(rings, num_rings, config);
            assert_final_ring_is_finish(rings, num_rings);
        }
    }
}

int main(void) {
    RaceConfig straight_config = {
        .course_mode = RACE_COURSE_STRAIGHT,
        .min_spacing = 7.0f,
        .max_spacing = 16.0f,
    };
    RaceConfig random_config = {
        .course_mode = RACE_COURSE_RANDOM,
        .min_spacing = 7.0f,
        .max_spacing = 16.0f,
    };
    RaceConfig extreme_config = {
        .course_mode = RACE_COURSE_EXTREME,
        .min_spacing = 16.0f,
        .max_spacing = 24.0f,
    };
    RaceConfig unknown_config = {
        .course_mode = 999,
        .min_spacing = 7.0f,
        .max_spacing = 16.0f,
    };

    assert_lookahead_helpers();
    assert_generated_courses(straight_config);
    assert_generated_courses(random_config);
    assert_generated_courses(extreme_config);
    assert_extreme_distribution(extreme_config);
    assert_extreme_accepts_unrestricted_geometry(extreme_config);
    assert_generated_courses(unknown_config);

    puts("drone race geometry checks passed");
    return 0;
}
