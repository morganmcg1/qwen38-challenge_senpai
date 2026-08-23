#!/usr/bin/env python3
"""E154 R2 delay-injection boundedness analysis.

Reads the per-arm run directories written by `research/e154_r2_session.sh`
and answers one question: when the host stalls for `delta` microseconds
inside a scored round, how much of that stall does the round absorb?

  absorbed fully  (slope 0) -> the round was waiting on the GPU, so the
                               host had at least `delta` of slack
  absorbed not at all (slope 1) -> the round was waiting on the host

The same probe runs on the device side with a dependent matmul chain. The
two slopes together separate GPU-bound from dispatch-bound.

harness=local. Never an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

RECORD_PREFIX = "e154-arm: "


def parse_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        row = {}
        for field in line[len(RECORD_PREFIX):].split():
            if "=" not in field:
                continue
            key, _, value = field.partition("=")
            try:
                row[key] = float(value) if "." in value or "e" in value.lower() \
                    else int(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def parse_meta(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    meta = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            meta[key.strip()] = value.strip()
    return meta


def load_arm(run_dir: pathlib.Path) -> dict | None:
    meta = parse_meta(run_dir / "meta.txt")
    if not meta:
        return None
    report = {}
    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}
    return {
        "dir": run_dir.name,
        "arm": meta.get("arm", "?"),
        "position": int(meta.get("position", 0)),
        "meta": meta,
        "report": report,
        "rounds": parse_records(run_dir / "rounds.txt"),
    }


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Slope, intercept, and standard error of the slope."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(residual / (n - 2) / sxx) if n > 2 else float("nan")
    return slope, intercept, se


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = p / 100.0 * (len(ordered) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": pct(5),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "sd": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def level_curve(rounds: list[dict], level_key: str) -> dict:
    """Mean wall clock per injection level, differenced against level zero.

    Levels cycle by drafting-round index, so every level is interleaved
    across the leg and monotone thermal drift is shared. Draft count `d`
    changes the round's real work, so the curve is also reported on the
    modal `d` alone.
    """
    by_level: dict[float, list[dict]] = defaultdict(list)
    for row in rounds:
        by_level[float(row.get(level_key, 0))].append(row)
    if 0.0 not in by_level:
        return {"error": f"no zero level present for {level_key}"}

    modal_d = Counter(r.get("d") for r in rounds).most_common(1)[0][0]
    out = {"level_key": level_key, "modal_d": modal_d, "levels": []}
    base_all = statistics.fmean(
        [r["wall_us"] for r in by_level[0.0]])
    base_modal = [r["wall_us"] for r in by_level[0.0] if r.get("d") == modal_d]

    for level in sorted(by_level):
        rows = by_level[level]
        walls = [r["wall_us"] for r in rows]
        modal = [r["wall_us"] for r in rows if r.get("d") == modal_d]
        entry = {
            "level": level,
            "n": len(rows),
            "mean_wall_us": statistics.fmean(walls),
            "sd_wall_us": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "delta_wall_us": statistics.fmean(walls) - base_all,
            "mean_thread_cpu_us": statistics.fmean(
                [r["thread_cpu_us"] for r in rows]),
            "mean_process_cpu_us": statistics.fmean(
                [r["process_cpu_us"] for r in rows]),
            "n_modal_d": len(modal),
        }
        if modal and base_modal:
            entry["delta_wall_us_modal_d"] = (
                statistics.fmean(modal) - statistics.fmean(base_modal))
        out["levels"].append(entry)

    xs = [e["level"] for e in out["levels"]]
    ys = [e["delta_wall_us"] for e in out["levels"]]
    slope, intercept, se = ols(xs, ys)
    out["absorption_slope"] = slope
    out["absorption_slope_se"] = se
    out["absorption_intercept_us"] = intercept

    # Per-round regression, with draft count as a covariate through the
    # modal-d subset. Far more degrees of freedom than the six level means.
    rows_modal = [r for r in rounds if r.get("d") == modal_d]
    s2, _, se2 = ols(
        [float(r.get(level_key, 0)) for r in rows_modal],
        [r["wall_us"] for r in rows_modal])
    out["per_round_slope_modal_d"] = s2
    out["per_round_slope_modal_d_se"] = se2
    out["per_round_n_modal_d"] = len(rows_modal)

    # Slack knee: the largest level whose measured cost stays inside noise.
    noise = max(
        (e["sd_wall_us"] for e in out["levels"] if e["level"] == 0.0),
        default=0.0)
    knee = 0.0
    for entry in out["levels"]:
        if entry["level"] == 0.0:
            continue
        if entry["delta_wall_us"] <= noise:
            knee = entry["level"]
        else:
            break
    out["slack_us_per_round"] = knee
    out["zero_level_sd_us"] = noise
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.runs_dir)
    arms = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if args.label and not child.name.startswith(args.label):
            continue
        arm = load_arm(child)
        if arm:
            arms.append(arm)
    if not arms:
        print(f"no arms under {root}", file=sys.stderr)
        return 1

    result = {
        "experiment": "e154-r2-round-boundedness",
        "harness": "local",
        "official_or_ranked_score": False,
        "runs_dir": str(root),
        "arms": [],
    }

    all_zero_walls: list[float] = []
    for arm in arms:
        meta = arm["meta"]
        entry = {
            "dir": arm["dir"],
            "arm": arm["arm"],
            "position": arm["position"],
            "arm_env": meta.get("arm_env"),
            "tokens": meta.get("tokens"),
            "all_tokens_matched": meta.get("all_tokens_matched"),
            "residual_divergence_count": meta.get("residual_divergence_count"),
            "round_count": meta.get("round_count"),
            "seconds_per_token": meta.get("seconds_per_token"),
            "effective_mean_draft_len": meta.get("effective_mean_draft_len"),
            "accepted_draft_rate": meta.get("accepted_draft_rate"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "cool_gate_passed_real_gate": meta.get(
                "cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "instrument_records": len(arm["rounds"]),
        }
        rounds = arm["rounds"]
        if rounds:
            entry["wall_clock_us"] = describe([r["wall_us"] for r in rounds])
            entry["thread_cpu_us"] = describe(
                [r["thread_cpu_us"] for r in rounds])
            entry["process_cpu_us"] = describe(
                [r["process_cpu_us"] for r in rounds])
            entry["draft_count_histogram"] = dict(
                sorted(Counter(r.get("d") for r in rounds).items()))
            entry["accepted_histogram"] = dict(
                sorted(Counter(r.get("acc") for r in rounds).items()))
            injected = {float(r.get("cpu_injected_us", 0)) for r in rounds}
            chains = {int(r.get("gpu_chain_steps", 0)) for r in rounds}
            if injected - {0.0}:
                entry["cpu_curve"] = level_curve(rounds, "cpu_injected_us")
            if chains - {0}:
                entry["gpu_curve"] = level_curve(rounds, "gpu_chain_steps")
            if arm["arm"] == "zero":
                all_zero_walls.extend(r["wall_us"] for r in rounds)
        report = arm["report"]
        for key in ("first_block_seconds", "seed_prefill_seconds",
                    "p50_block_request_seconds_after_first",
                    "max_block_request_seconds_after_first"):
            if key in report:
                entry[key] = report[key]
        result["arms"].append(entry)

    result["e154_absolute_round_wall_clock_us"] = describe(all_zero_walls)
    result["e154_host_syncs_per_round"] = 1

    # Token neutrality: the instrument must not change what is generated.
    matched = {a["all_tokens_matched"] for a in result["arms"]}
    rounds_seen = {a["round_count"] for a in result["arms"]}
    edl = {a["effective_mean_draft_len"] for a in result["arms"]}
    result["e154_instrument_is_token_neutral"] = {
        "all_tokens_matched": sorted(matched),
        "round_count": sorted(rounds_seen),
        "effective_mean_draft_len": sorted(edl),
        "verdict": (
            matched == {"true"} and len(rounds_seen) == 1 and len(edl) == 1),
    }

    # Zero-arm drift check. The session is void if the control moves more
    # than the effect it is supposed to measure.
    zero_arms = [a for a in result["arms"] if a["arm"] == "zero"
                 and a.get("wall_clock_us")]
    if len(zero_arms) > 1:
        means = [a["wall_clock_us"]["mean"] for a in zero_arms]
        result["zero_control_drift_us"] = max(means) - min(means)
        result["zero_control_means_us"] = means

    temps = [float(a["gpu_temp_entry_c"]) for a in result["arms"]
             if a.get("gpu_temp_entry_c")]
    if temps:
        result["entry_temperature_spread_c"] = max(temps) - min(temps)
        result["entry_temperatures_c"] = temps

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
