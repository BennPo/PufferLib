#!/usr/bin/env python3
import argparse
import math
import sys
from typing import Any

import wandb


SUMMARY_KEYS = [
    "env/score",
    "env/perf",
    "env/rings_passed",
    "env/ring_collisions",
    "env/collisions",
    "env/oob",
    "env/timeout",
    "env/episode_return",
    "env/episode_length",
    "env/ema_dist",
    "env/ema_vel",
    "env/ema_omega",
    "env/race_lookahead_reward",
    "loss/approx_kl",
    "loss/explained_variance",
    "loss/entropy",
]

TREND_KEYS = [
    "env/score",
    "env/rings_passed",
    "env/oob",
    "env/timeout",
    "env/episode_return",
    "env/episode_length",
    "env/race_lookahead_reward",
]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return None


def nested_get(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def history_first_last(history_rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    first = None
    last = None
    for row in history_rows:
        value = as_float(row.get(key))
        if value is None:
            continue
        if first is None:
            first = value
        last = value
    return first, last


def fetch_history(run: Any) -> list[dict[str, Any]]:
    try:
        rows = run.history(keys=TREND_KEYS, samples=500, pandas=False)
    except Exception:
        return []
    return list(rows) if rows is not None else []


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def detect_findings(config: dict[str, Any], summary: dict[str, Any], trends: dict[str, tuple[float | None, float | None]]) -> list[str]:
    findings: list[str] = []

    oob = as_float(summary.get("env/oob"))
    timeout = as_float(summary.get("env/timeout"))
    rings = as_float(summary.get("env/rings_passed"))
    episode_length = as_float(summary.get("env/episode_length"))
    episode_return = as_float(summary.get("env/episode_return"))
    approx_kl = as_float(summary.get("loss/approx_kl"))
    explained_variance = as_float(summary.get("loss/explained_variance"))
    entropy = as_float(summary.get("loss/entropy"))

    if oob is not None and timeout is not None and oob > 0.8 and timeout < 0.2:
        findings.append(
            f"OOB still dominates endings (`env/oob={fmt(oob)}`, `env/timeout={fmt(timeout)}`), so the policy is mostly learning to survive slightly longer rather than finish runs."
        )

    if rings is not None and rings < 2.0:
        findings.append(
            f"Mean rings passed is still low (`env/rings_passed={fmt(rings)}`), which suggests the policy is plateauing around one-ring behavior."
        )

    first_rings, last_rings = trends.get("env/rings_passed", (None, None))
    if first_rings is not None and last_rings is not None and last_rings - first_rings < 0.5:
        findings.append(
            f"Ring progress improved only a little over the run (`{fmt(first_rings)} -> {fmt(last_rings)}`), which points to weak or misaligned task reward."
        )

    first_oob, last_oob = trends.get("env/oob", (None, None))
    if first_oob is not None and last_oob is not None and last_oob > 0.8:
        findings.append(
            f"OOB decreased only marginally (`{fmt(first_oob)} -> {fmt(last_oob)}`), so boundary avoidance is still not learned."
        )

    if episode_length is not None and episode_return is not None and oob is not None and oob > 0.8:
        findings.append(
            f"Return is rising with episode length (`return={fmt(episode_return)}`, `length={fmt(episode_length)}`) while OOB stays high, which often means the agent found a partial strategy that is still crash-dominated."
        )

    if approx_kl is not None and explained_variance is not None and entropy is not None:
        if approx_kl < 0.02 and explained_variance > 0.9:
            findings.append(
                f"Optimization looks numerically stable (`approx_kl={fmt(approx_kl)}`, `explained_variance={fmt(explained_variance)}`, `entropy={fmt(entropy)}`), so the bottleneck is more likely environment/reward design than PPO instability."
            )

    learning_rate = as_float(nested_get(config, "train.learning_rate"))
    ent_coef = as_float(nested_get(config, "train.ent_coef"))
    if learning_rate is not None and ent_coef is not None and learning_rate > 0.005 and ent_coef < 1e-4:
        findings.append(
            f"Training settings are aggressive for a control task (`learning_rate={learning_rate:g}`, `ent_coef={ent_coef:g}`), which can lock in a mediocre local policy early."
        )

    return findings


def print_run_report(run: Any) -> None:
    config = dict(run.config)
    summary = dict(run.summary)

    history_rows = fetch_history(run)
    trends = {key: history_first_last(history_rows, key) for key in TREND_KEYS}
    findings = detect_findings(config, summary, trends)

    print(f"Run: {run.name or run.id} ({run.id})")
    print(f"State: {run.state}")
    print(f"Project: {run.project}")
    print(f"Created: {run.created_at}")
    print(f"Task: {nested_get(config, 'env.task')}")

    for key in SUMMARY_KEYS:
        value = as_float(summary.get(key))
        if value is not None:
            print(f"{key}: {fmt(value)}")

    print("Trends:")
    for key in TREND_KEYS:
        first, last = trends[key]
        print(f"  {key}: {fmt(first)} -> {fmt(last)}")

    print("Findings:")
    if findings:
        for finding in findings:
            print(f"  - {finding}")
    else:
        print("  - No obvious failure pattern detected from the fetched metrics.")
    print()


def infer_entities(api: wandb.Api, explicit_entity: str | None) -> list[str]:
    if explicit_entity:
        return [explicit_entity]

    viewer = getattr(api, "viewer", None)
    candidates: list[str] = []
    username = getattr(viewer, "username", None)
    if username:
        candidates.append(username)

    for team in getattr(viewer, "teams", []) or []:
        if team not in candidates:
            candidates.append(team)

    entity = getattr(viewer, "entity", None)
    if entity and entity not in candidates:
        candidates.append(entity)

    if not candidates:
        raise SystemExit("Could not infer W&B entity. Pass --entity explicitly.")

    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the most recent W&B runs for a project.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/user/org")
    parser.add_argument("--project", type=str, default="drone-experiments", help="W&B project name")
    parser.add_argument("--count", type=int, default=3, help="Number of recent runs to inspect")
    parser.add_argument("--include-running", action="store_true", help="Include currently running runs")
    args = parser.parse_args()

    api = wandb.Api()
    selected = []
    chosen_path = None
    last_error = None
    for entity in infer_entities(api, args.entity):
        path = f"{entity}/{args.project}"
        try:
            runs = api.runs(path, order="-created_at")
            current = []
            for run in runs:
                if not args.include_running and run.state == "running":
                    continue
                current.append(run)
                if len(current) >= args.count:
                    break
            if current:
                selected = current
                chosen_path = path
                break
        except Exception as exc:
            last_error = exc

    if not selected or chosen_path is None:
        if last_error is not None:
            raise SystemExit(str(last_error))
        raise SystemExit(f"No runs found for project {args.project}")

    print(f"Analyzing {len(selected)} most recent runs from {chosen_path}")
    print()
    for run in selected:
        print_run_report(run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
