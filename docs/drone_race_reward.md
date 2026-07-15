# Drone Race Reward

Use one progress value:

```text
absolute_progress = ring_index + proximity_to_current_target
```

Then define the base reward as:

```text
progress_reward = absolute_progress_after - absolute_progress_before
```

Where:

- `ring_index` is the number of rings already passed
- `proximity_to_current_target` is a number in `[0, 1]` measuring closeness to the current target ring

For the current drone env, define:

```text
dist = ||drone_pos - ring_pos||
d_max = distance from one in-bounds corner to the opposite corner
k = (d_max - 10) / 25
proximity_to_current_target =
    clamp(1 - log(1 + k * dist) / log(1 + k * d_max), 0, 1)
```

The race bounds are:

```text
x in [-MARGIN_X, MARGIN_X]
y in [-MARGIN_Y, MARGIN_Y]
z in [-MARGIN_Z, MARGIN_Z]
```

So:

```text
d_max = sqrt((2 * MARGIN_X)^2
           + (2 * MARGIN_Y)^2
           + (2 * MARGIN_Z)^2)
```

This choice of `k` gives:

```text
proximity_to_current_target(0) = 1
proximity_to_current_target(5) = 0.5
proximity_to_current_target(d_max) = 0
```

This means:

- moving closer to the current ring gives positive reward
- moving away gives negative reward
- changes closer to the ring are weighted more heavily than changes far away
- passing a ring gives a one-time progress jump plus `race_clean_pass_bonus`
- hovering in place gives about `0`

On top of that base progress reward, the current env also applies an explicit
out-of-bounds penalty:

```text
reward =
    progress_reward
    + race_clean_pass_bonus * 1[clean_pass]
    + race_aperture_alignment_coef * alignment_delta
    + race_lookahead_segment_coef * gate_quality * segment_progress_delta
    - race_oob_penalty * 1[oob]
    - ring_collision_penalty * 1[ring_collision]
```

`race_oob_penalty` is only applied on the terminal step that exits bounds.
`ring_collision_penalty` is only applied on the terminal step that hits a ring
rim. `race_clean_pass_bonus` is added only when the drone cleanly passes through
the active ring. `alignment_delta` is only applied on ordinary non-terminal
approach steps, so clean passes and rim collisions are not shaped by the
aperture term.

`segment_progress_delta` measures progress along the line from the current ring
to the next ring. It is multiplied by `gate_quality`, so the lookahead term only
matters when the drone is near the current ring aperture, centered, and moving
through the current ring in the correct direction. This avoids rewarding raw
distance to the next ring, which can encourage corner cutting.

The aperture alignment value is:

```text
alignment =
    entry_side_gate
  * near_plane_gate
  * centeredness
  * forward_direction
```

Where:

- `entry_side_gate = 1` on the entry side of the ring plane, otherwise `0`
- `near_plane_gate` fades from `1` at the ring plane to `0` eight units away
- `centeredness` is `1` at the ring centerline and `0` at the safe aperture rim
- `forward_direction` is the positive component of velocity along the ring normal

The reward uses the change in this value, not the raw value, so the policy is
rewarded for improving its approach instead of sitting aligned in front of a
ring.

The default `race_oob_penalty`, `ring_collision_penalty`,
`race_clean_pass_bonus`, `race_aperture_alignment_coef`, and lookahead knobs
live in `config/drone.ini`. The recommended first experiment values are
`race_clean_pass_bonus = 0.5`, `race_aperture_alignment_coef = 0.1`, and
`race_lookahead_segment_coef = 0.05`.

The drone observation now has `26` floats. The final three entries are signed
normalized world position:

```text
pos.x / MARGIN_X
pos.y / MARGIN_Y
pos.z / MARGIN_Z
```

These values are clamped to `[-1, 1]` and are exposed on all drone tasks, not
just race, so policies can directly reason about arena-relative position.

When a ring is passed:

- increment `ring_index`
- switch the target to the next ring
- compute `absolute_progress_after` using the new target

When the drone hits the ring rim instead of making a clean pass:

- record a ring collision
- end the episode under the same reset flow used for other race terminals
- do not increment `ring_index`
- do not switch the target to the next ring

Out of bounds should end the episode under standard terminal semantics.
In practice, terminal semantics alone were not enough for stable transfer from
hover, because the policy could still learn to grab some race progress and then
crash. The explicit OOB penalty is there to make "progress, then die" less
attractive than bounded flight.
