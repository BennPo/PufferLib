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
- passing a ring gives a one-time jump of about `+1`
- hovering in place gives about `0`

On top of that base progress reward, the current env also applies two race
stability terms:

```text
reward =
    progress_reward
    - race_boundary_penalty * boundary_risk(pos_after, race_boundary_margin)
    - race_oob_penalty * 1[oob]
```

Where:

- `boundary_risk` is `0` when the drone has enough clearance to all bounds
- `boundary_risk` grows as the drone approaches the nearest world boundary
- `race_oob_penalty` is only applied on the terminal step that exits bounds

The default knobs live in `config/drone.ini`:

- `race_oob_penalty`
- `race_boundary_penalty`
- `race_boundary_margin`

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
crash. The explicit OOB penalty and boundary shaping are there to make
"progress, then die" less attractive than bounded flight.
