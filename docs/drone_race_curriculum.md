# Hover-To-Race Curriculum

This playbook explains how to warm-start drone race training from an existing
hover checkpoint.

The goal is simple: reuse a policy that already knows how to stabilize the
drone, then adapt it to race reward and race targets. The policy architecture
does not change. Only the task and the training schedule change.

## What You Need

- a hover checkpoint path such as `<hover_checkpoint.bin>`
- the normal drone training entrypoint: `python -m pufferlib.pufferl ...`
- a build of the drone env

This repo already supports continuing training from existing weights with
`--load-model-path`.

## Important Repo Facts

- `config/drone.ini` defaults to hover with `task = 1`
- hover is task `1`
- race is task `7`
- drone observations now have `26` floats, with the final three slots holding
  normalized world position `x`, `y`, and `z`
- older drone `.bin` checkpoints trained with the previous `23`-float
  observation layout are intentionally incompatible with the current model
- the current race reward is the progress-delta objective with explicit OOB
  penalty and boundary shaping described in
  `docs/drone_race_reward.md`
- the current drone binding only exposes the env keys already present in
  `config/drone.ini`, so this curriculum should stay within the existing CLI
  and config surface

## Stage 0: Validate The Hover Checkpoint

Before starting race finetuning, make sure the hover checkpoint is actually a
good low-level controller.

Run hover eval:

```bash
python -m pufferlib.pufferl eval drone \
  --task 1 \
  --load-model-path <hover_checkpoint.bin>
```

What to look for:

- stable behavior instead of immediate collapse
- `env/oob ~= 0`
- `env/timeout ~= 1`
- high or near-peak `env/perf`

If the checkpoint is not stable on hover, do not use it as the curriculum
starting point.

For hover, timing out is usually a success signal, not a failure signal. A
good hover policy survives to the episode horizon instead of drifting away from
its target and resetting early.

Do not automatically use the final hover checkpoint. Prefer the earliest
checkpoint that is already stable and is close to peak hover `env/perf`. That
usually transfers better than the most specialized late-run checkpoint.

## Why Hover Survives While Race Goes OOB

This confusion is common and worth making explicit.

Hover and race do not use the same reward:

- hover rewards local stabilization and penalizes unstable motion
- race rewards progress toward and through rings

Hover and race also do not use the same OOB condition:

- hover OOB means the drone drifted too far from the hover target
- race OOB means the drone exited the world bounds

Race now also treats ring-rim collisions as immediate episode resets. That
means some early retrains may shift failure mass from `env/oob` into
`env/ring_collisions` before clean passes improve.

That means a hover checkpoint can transfer low-level stabilization without yet
knowing how to navigate forward aggressively while staying in bounds. Early
race finetuning often looks like:

- `env/perf` rises a bit
- `env/rings_passed` rises a bit
- `env/oob` stays pinned near `1`

That usually means the policy learned to lunge for progress before crashing,
not to fly the course in a stable way.

## Stage 1: Short Race Warm Start And Recovery Protocol

Switch to race, keep the course short, and use a smaller timestep budget than a
full race run.

This stage is only for adaptation. The point is to let the policy learn race
targets and race reward without having to solve the full task immediately.

Do not try only one hover checkpoint here. Try two:

- an earlier stable hover checkpoint
- a later stable hover checkpoint

Recommended starting command:

```bash
python -m pufferlib.pufferl train drone \
  --task 7 \
  --max-rings 3 \
  --train.total-timesteps 20000000 \
  --checkpoint-interval 50 \
  --train.learning-rate 5e-4 \
  --train.ent-coef 1e-5 \
  --load-model-path <hover_checkpoint.bin>
```

What success looks like:

- `env/oob` starts trending down
- `env/ring_collisions` starts trending down after an initial spike
- `env/rings_passed` starts moving up
- `env/score` and `env/perf` improve together
- `env/timeout` begins to appear instead of every episode ending OOB
- `loss/entropy` stops climbing aggressively while score is flat

Do not promote a Stage 1 checkpoint just because score went up once. Prefer a
checkpoint that is more stable, not just more aggressive.

If around halfway through the warm start `env/oob` is still above `0.95` and
`env/timeout` is still near `0`, stop that run and try the next hover
checkpoint instead of letting a clearly crash-dominated run continue.

After the ring-collision reset change, also watch `env/ring_collisions`. It is
normal for that metric to rise at first because former rim-hit episodes now end
immediately. What matters is that it begins to fall as `env/rings_passed`
improves.

If both initial Stage 1 runs fail, use this fallback tuning ladder:

1. rerun the better hover checkpoint with higher exploration

```bash
python -m pufferlib.pufferl train drone \
  --task 7 \
  --max-rings 3 \
  --train.total-timesteps 20000000 \
  --checkpoint-interval 50 \
  --train.learning-rate 5e-4 \
  --train.ent-coef 3e-5 \
  --load-model-path <hover_checkpoint.bin>
```

2. if that still fails, rerun once more with smaller updates

```bash
python -m pufferlib.pufferl train drone \
  --task 7 \
  --max-rings 3 \
  --train.total-timesteps 20000000 \
  --checkpoint-interval 50 \
  --train.learning-rate 2e-4 \
  --train.ent-coef 3e-5 \
  --load-model-path <hover_checkpoint.bin>
```

Cap the training-only recovery path at four short Stage 1 runs. If all four are
still crash-dominated, stop adjusting only training commands and move to a
code-change plan instead.

## Stage 2: Full Race Finetune

Once the warm start is producing better race behavior, continue training on the
full race setup.

Recommended example:

```bash
python -m pufferlib.pufferl train drone \
  --task 7 \
  --max-rings 10 \
  --train.total-timesteps 200000000 \
  --checkpoint-interval 50 \
  --train.learning-rate 5e-4 \
  --train.ent-coef 1e-5 \
  --load-model-path <stage1_checkpoint.bin>
```

This stage is where the policy should move from "occasionally reaches rings"
toward "stays in bounds and keeps making progress through the course."

If `env/oob` climbs back above `0.90` for a long stretch while `env/timeout`
stays near `0`, do not keep promoting later checkpoints from that run. Restart
from the original Stage 1 promotion checkpoint, or from the next-best Stage 1
checkpoint, and try again.

## Stage 3: Promote Only When Behavior Improves

Advance to the next checkpoint based on behavior, not just on the latest saved
file.

A good promotion checkpoint should show:

- lower `env/oob`
- lower `env/ring_collisions`
- higher `env/rings_passed`
- improving `env/score`
- improving `env/perf`
- `env/timeout` beginning to appear instead of nearly every episode ending OOB
- `loss/entropy` flattening or starting to fall instead of rising while score
  is flat

A checkpoint with high score but `env/oob ~= 1` is usually not a good promotion
candidate. That often means the policy learned to lunge for reward before
crashing, not to control the full race.

Do not automatically use the last checkpoint from a run.

When comparing Stage 1 checkpoints, rank them in this order:

1. lowest `env/oob`
2. highest `env/perf`
3. highest `env/rings_passed`
4. highest `env/timeout`

Only promote a Stage 1 checkpoint into full race if it passes all of these:

- `env/oob < 0.85`
- `env/timeout > 0.05`
- `env/perf` clearly improved over the first saved Stage 1 checkpoint
- `env/rings_passed >= 0.8`

Stage 2 should always start from the best Stage 1 checkpoint, never just the
last saved Stage 1 checkpoint.

## Recommended Automation

If this curriculum is automated later, it should be implemented as a
trainer-side schedule, not as a new env task such as `task = 8`.

That is the cleaner design because:

- hover and race are still the real env tasks
- the curriculum is a training procedure, not a new physics mode
- checkpoints stay easy to compare, resume, and ablate
- it remains easy to run plain hover-only and plain race-only baselines

The recommended shape is something like:

```text
--curriculum hover_race
--curriculum.hover_timesteps <stage0_steps>
--curriculum.race_timesteps <stage1_steps>
--curriculum.warm_start_max_rings <small>
```

With behavior like:

1. train hover with `task = 1`
2. save or select the best hover checkpoint
3. switch to `task = 7`
4. continue training from that checkpoint on a shorter race setup
5. promote into full race finetuning

This should live in the trainer or config layer, not inside the drone env.

## How To Choose Checkpoints Between Stages

Use the best checkpoint from the previous stage, not the most recent one.

For this curriculum, "best" means:

- it crashes less often
- it collides with rings less often
- it passes more rings
- it improves score without keeping `env/oob` pinned near `1`
- it starts producing non-trivial `env/timeout`

In practice:

- Stage 1 should choose the best hover-to-race adaptation checkpoint
- Stage 2 should start from that selected Stage 1 checkpoint

## Failure Patterns

These patterns are common and worth checking early:

- `env/oob` stuck at `1`
  - the policy is still not learning boundary-safe control
- `env/ring_collisions` stuck high
  - the policy is reaching rings but not learning clean passes
- `loss/entropy` rising while score is flat
  - the policy is staying too random instead of becoming more precise
- `env/rings_passed` rising a little while OOB stays pinned
  - the policy is learning to grab reward opportunistically, not to fly the
    course in a stable way

If those patterns hold for a long stretch, do not keep promoting checkpoints
blindly. Go back to the last stable checkpoint and rerun the shorter stage.

If repeated short warm starts still show those patterns after trying an earlier
stable hover checkpoint, a later stable hover checkpoint, and the Stage 1
fallback tuning ladder, stop the training-only loop and switch to a code-change
plan.

## Minimal Workflow Summary

1. Evaluate the hover checkpoint with `--task 1`.
2. Choose two stable hover checkpoints, usually one earlier and one later.
3. Run short race warm starts with `--task 7`, `--max-rings 3`, and a smaller
   timestep budget.
4. Promote checkpoints using OOB-first criteria, not latest-checkpoint logic.
5. Continue full race finetuning only after Stage 1 is less crash-dominated.
6. If repeated short warm starts still fail, switch to a code-change plan.
