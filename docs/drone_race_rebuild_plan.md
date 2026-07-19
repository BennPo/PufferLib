# Rebuild Drone Race Course Generation

## Summary

Rebuild the drone race course generator around open connected race modes. Mode
`1` creates a gentle straight-ish path, and mode `2` creates a more varied
random 3D path. Both modes guarantee that every ring can be flown through
directly.

Mode `3` is an opt-in extreme course. Each center is sampled 16-24 units from
the previous center when launched with the recommended spacing overrides, and
each ring normal is sampled independently over the sphere. Unlike modes `1`
and `2`, mode `3` deliberately does not guarantee a direct fly-through from the
previous ring; backward-facing and otherwise opposed gates are valid.

The core rule is that gates may be randomly spaced and angled, but every ring
must pass geometry checks that keep its normal aligned with the route. The drone
should be able to fly from one ring to the next through the aperture, without
needing to go around a ring to approach it from the valid side. The final ring
is the finish line; it is not compared against the first ring.

## Key Changes

- Replace sliding race difficulty with fixed `race_course_mode` values.
- Keep public spacing controls while using internal turn and height presets.
- Make race mode default to the visibly varied random 3D course when `task = 7`.
- Validate every adjacent ring pair in the course, without any final-to-first
  wraparound checks.
- End the race episode successfully when the final ring is passed.
- Preserve the existing race reward, observation layout, ring pass detection,
  and rendering behavior unless compile cleanup requires small local fixes.

Recommended race defaults:

```ini
race_course_mode = 2
race_min_spacing = 7.0
race_max_spacing = 16.0
```

## Implementation

- Implement the generator in `ocean/drone/dronelib.h`, replacing the internals
  of `reset_rings`.
- Start each course near the arena center with a random heading.
- Generate each next ring from the previous ring using:
  - random spacing from `race_min_spacing` to `race_max_spacing`,
  - one signed random yaw angle, where negative values are left turns and
    positive values are right turns,
  - random vertical pitch or height delta from the selected course preset.
- Use fixed internal presets:
  - mode `1`: max turn about `18` degrees, max vertical delta `0.5`,
  - mode `2`: max turn just under `90` degrees, max vertical delta `4.0`.
- Accept the first sampled candidate that stays in bounds and preserves the
  adjacent fly-through geometry; otherwise retry the same ring placement.
- Do not score candidates for center pull, edge distance, or preferred turn
  direction.
- Keep all ring centers inside the race bounds with enough clearance for the
  ring radius.
- Compute each ring normal from the open route geometry after positions are
  chosen:
  - first ring normal points toward ring `1`,
  - middle ring normals use the normalized bisector of incoming and outgoing
    path directions,
  - final ring normal points along the incoming segment from the previous ring.
- Reject and regenerate any course candidate that violates the hard checks.
- Use the existing straight open race fallback only if repeated random
  generation attempts fail.
- On a clean final-ring pass, increment `rings_passed`, mark the episode
  terminal, log full course progress, and reset onto a fresh generated course.

Required hard checks for connected modes `1` and `2`:

- Adjacent ring normals must be less than 90 degrees apart:
  `dot(previous.normal, next.normal) > 0`.
- Each real segment `i -> i + 1` must satisfy configured spacing, vertical
  pitch, height-delta, and in-bounds constraints.
- For middle rings, incoming and outgoing path directions must not require a
  turn of 90 degrees or more.
- Each next ring center must be on the current ring's exit side, and each
  current ring center must be on the next ring's entry side.
- No spacing, normal, turn, height, or entry/exit check is performed from the
  final ring back to ring `0`.

Mode `3` keeps only the bounds, configured adjacent-center spacing, and
unit-normal checks from this list. Its turn, normal alignment, and entry/exit
geometry are intentionally unrestricted.

## Test Plan

- Add a focused geometry test that generates many courses across different RNG
  seeds and validates:
  - all ring centers are in bounds,
  - every spacing value is within the configured range,
  - every ring normal is normalized,
  - all adjacent normal angles are under 90 degrees,
  - every real segment has valid entry and exit side geometry,
  - all vertical pitch and height-delta constraints hold,
  - no wraparound segment is required from the last ring to the first ring.
- Include a direct pass-through check using `check_ring`: simulate crossing each
  ring through its center from the entry side to the exit side and assert that
  it returns a clean pass.
- Include a final-ring completion check so the final target is treated as a
  finish line instead of wrapping to ring `0`.
- Compile and run the geometry test with `clang` so it does not depend on the
  renderer.
- Build the drone environment with `./build.sh drone --cpu`, or the normal
  local build if raylib/CUDA dependencies are available.
- Visually inspect race mode in the standalone renderer to confirm the generated
  track is three-dimensional and moves around the map.

## Assumptions

- The implementation should edit the top-level source tree under
  `/home/benn/PufferLib`, not the nested untracked `PufferLib/` copy.
- The race should be open: after the last ring, the episode completes and
  resets.
- "Random path" means a connected random route through the arena, not
  independent random ring placement.
- Course modes replace the old difficulty slider and public turn/height tuning
  keys.
