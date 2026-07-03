#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../ocean/drone/dronelib.h"

static void assert_clean_center_pass(Target* ring) {
    Drone drone = {0};
    drone.prev_pos = sub3(ring->pos, scalmul3(ring->normal, 1.0f));
    drone.state.pos = add3(ring->pos, scalmul3(ring->normal, 1.0f));

    int state = check_ring(&drone, ring);
    assert(state == 1);
}

static void assert_course(Target* rings, int num_rings, RaceConfig config) {
    assert(race_course_is_valid_open(rings, num_rings, config, true));

    RaceCoursePreset preset = race_course_preset(config.course_mode);
    float max_dz = 0.0f;
    for (int i = 0; i < num_rings; i++) {
        assert_clean_center_pass(&rings[i]);
    }

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
    RaceConfig unknown_config = {
        .course_mode = 999,
        .min_spacing = 7.0f,
        .max_spacing = 16.0f,
    };

    assert_generated_courses(straight_config);
    assert_generated_courses(random_config);
    assert_generated_courses(unknown_config);

    puts("drone race geometry checks passed");
    return 0;
}
