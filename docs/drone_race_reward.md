# Drone Race Reward

Use one progress value:

```text
absolute_progress = ring_index + proximity_to_current_target
```

Then define reward as:

```text
reward = absolute_progress_after - absolute_progress_before
```

Where:

- `ring_index` is the number of rings already passed
- `proximity_to_current_target` is a number in `[0, 1]` measuring closeness to the current target ring

For the current drone env, define:

```text
dist = ||drone_pos - ring_pos||
d_max = distance from one in-bounds corner to the opposite corner
proximity_to_current_target = clamp(1 - dist / d_max, 0, 1)
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

This means:

- moving closer to the current ring gives positive reward
- moving away gives negative reward
- passing a ring gives a one-time jump of about `+1`
- hovering in place gives about `0`

When a ring is passed:

- increment `ring_index`
- switch the target to the next ring
- compute `absolute_progress_after` using the new target

Out of bounds should end the episode under standard terminal semantics.

This is usually enough punishment by itself, because the drone loses all future reward.
