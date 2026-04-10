# Drone Race Reward

Use one reward only:

```text
reward = pass_bonus + proximity_after - proximity_before
```

Where:

- `pass_bonus = 1` on a valid ring pass, otherwise `0`
- `proximity` is a number in `[0, 1]` that measures how close the drone is to its current target ring

## Proximity

Define:

```text
proximity(pos, ring) = 1 - dist(pos, ring.pos) / max_dist(ring)
```

Where:

- `dist(pos, ring.pos)` is Euclidean distance to the ring center
- `max_dist(ring)` is the maximum possible in-bounds distance from that ring to any corner of the race volume

Because `max_dist(ring)` is the true in-bounds maximum, `proximity` is naturally in `[0, 1]` while the drone is in bounds.

## Pass Step

When the drone passes a ring:

- add `+1`
- advance the checkpoint index
- compute `proximity_after` using the next ring

So passing a ring gives an immediate jump in reward and then smoothly rewards getting closer to the following ring.

## Why This Is Good

- Passing two rings is worth more than passing one.
- Moving closer helps, moving away hurts.
- Oscillating in place does not create net reward.
- No speed or omega reward terms are needed.
- The reward directly matches course progress.
