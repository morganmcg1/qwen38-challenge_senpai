#!/usr/bin/env python3
"""Reduce the E78 rung 2b session to one decision.

  python3 research/e78_analyze.py [--out research/e78-artifacts/rung2b.json]

The stop rule was written before the session ran:

  * primary metric  = absolute candidate `mtp_seconds_per_token`;
  * baseline        = the `a_ship` groups measured in the SAME session;
  * session null    = the within-arm difference between the early and the late
                      position group, which sees the same drift and the same
                      rebuilds as a real arm contrast but no dispatch change;
  * useful effect   = a hybrid beats `a_ship` by more than twice the largest
                      session null, in the predicted direction.

Anything smaller is reported as `not useful`. The threshold is not retuned
after the fact.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import subprocess

RUNS = pathlib.Path(".mlxfast-private/e78/runs")
ARMS = pathlib.Path(".mlxfast-private/e78/arms")

# tag -> (arm, position group). `warm` is declared discarded and never enters a
# contrast.
LAYOUT = {
    "a1": ("a_ship", "early"),
    "b1": ("b_crown", "early"),
    "c1": ("c_hybrid24928", "early"),
    "d1": ("d_hybrid8192", "early"),
    "d2": ("d_hybrid8192", "late"),
    "c2": ("c_hybrid24928", "late"),
    "b2": ("b_crown", "late"),
    "a2": ("a_ship", "late"),
}
BASELINE_ARM = "a_ship"
PRIMARY = "mtp_seconds_per_token"
SECONDARY = ("serial_seconds_per_token", "mtp_decode_speedup",
             "effective_mean_draft_len", "accepted_draft_rate")

# The arm ladder is the per-family-group attribution: each step adds exactly one
# family group to the set that runs at IPG 3.
LADDER = (
    ("d_hybrid8192 - a_ship", "d_hybrid8192", "a_ship",
     "n=5120: mlp.down, linear_attn.out_proj, full_attn.o_proj"),
    ("c_hybrid24928 - d_hybrid8192", "c_hybrid24928", "d_hybrid8192",
     "n=14336 qkv_proj and n=16480 linear_attn.in_proj"),
    ("b_crown - c_hybrid24928", "b_crown", "c_hybrid24928",
     "n=34816 mlp.gate_up, n=98336 compact_draft_vocab, n=248320 lm_head"),
)


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def width_histogram(report: pathlib.Path) -> dict[str, int] | None:
    if not report.exists():
        return None
    doc = json.loads(report.read_text())
    if doc.get("is_serial_control"):
        return None
    lengths = doc.get("effective_draft_lengths")
    if not lengths:
        return None
    counts = collections.Counter(int(x) + 1 for x in lengths)
    return {str(k): counts[k] for k in sorted(counts)}


def collect() -> dict:
    groups: dict[str, dict] = {}
    for tag, (arm, position) in LAYOUT.items():
        run_dir = RUNS / tag
        meta = read_meta(run_dir / "meta.txt")
        scores = sorted(run_dir.glob("score-*.json"))
        legs = []
        for index, path in enumerate(scores, start=1):
            metrics = json.loads(path.read_text())["metrics"]
            legs.append({
                "index": index,
                "metrics": {k: metrics.get(k) for k in (PRIMARY, *SECONDARY)},
                "all_tokens_matched": bool(metrics.get("all_tokens_matched")),
                "residual_divergence_count":
                    metrics.get("residual_divergence_count"),
                "width_histogram": width_histogram(
                    run_dir / f"reports/leg-{index}/04-mtp-timed.json"),
                "thermal_after": meta.get(f"leg{index}_thermal_after"),
            })
        wandb_path = run_dir / "wandb.json"
        groups[tag] = {
            "arm": arm,
            "position": position,
            "legs": legs,
            "thermal_before": meta.get("thermal_before"),
            "thermal_after": meta.get("thermal_after"),
            "cool_gate_requested": meta.get("cool_gate_requested"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "worker_sha256": meta.get("worker_sha256"),
            "metallib_fingerprint": meta.get("metallib_fingerprint"),
            "twin_digests": meta.get("twin_digests"),
            "dispatch": {
                key: value for key, value in meta.items()
                if key.startswith("e78_binary_assert_")
            },
            "wandb": json.loads(wandb_path.read_text())
            if wandb_path.exists() else None,
        }
    return groups


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e78-artifacts/rung2b.json")
    args = ap.parse_args()

    groups = collect()
    missing = [tag for tag, g in groups.items() if not g["legs"]]

    per_arm: dict[str, list[float]] = collections.defaultdict(list)
    per_arm_position: dict[tuple[str, str], list[float]] = \
        collections.defaultdict(list)
    secondary: dict[str, dict[str, list[float]]] = \
        collections.defaultdict(lambda: collections.defaultdict(list))
    histograms: set[str] = set()
    unmatched = []

    for tag, group in groups.items():
        for leg in group["legs"]:
            value = leg["metrics"].get(PRIMARY)
            if isinstance(value, (int, float)):
                per_arm[group["arm"]].append(value)
                per_arm_position[(group["arm"], group["position"])].append(value)
            for key in SECONDARY:
                other = leg["metrics"].get(key)
                if isinstance(other, (int, float)):
                    secondary[group["arm"]][key].append(other)
            if leg["width_histogram"] is not None:
                histograms.add(json.dumps(leg["width_histogram"],
                                          sort_keys=True))
            if not leg["all_tokens_matched"]:
                unmatched.append(f"{tag}/leg-{leg['index']}")

    baseline = mean(per_arm.get(BASELINE_ARM, []))

    # The null is a same-arm, cross-position contrast: identical dispatch table,
    # identical rebuild path, different place in the session.
    nulls = {}
    for arm in per_arm:
        early = mean(per_arm_position.get((arm, "early"), []))
        late = mean(per_arm_position.get((arm, "late"), []))
        if early is not None and late is not None:
            nulls[arm] = abs(late - early)
    session_null = max(nulls.values()) if nulls else None

    arms = {}
    for arm, values in sorted(per_arm.items()):
        value = mean(values)
        arms[arm] = {
            "legs": len(values),
            PRIMARY: value,
            f"{PRIMARY}_min": min(values),
            f"{PRIMARY}_max": max(values),
            f"{PRIMARY}_stdev": statistics.stdev(values)
            if len(values) > 1 else None,
            "delta_vs_baseline": None if baseline is None else value - baseline,
            "pct_vs_baseline": None if baseline is None
            else 100.0 * (value - baseline) / baseline,
            "session_null": nulls.get(arm),
            **{key: mean(vals) for key, vals in secondary[arm].items()},
        }

    ladder = []
    for name, high, low, families in LADDER:
        a, b = mean(per_arm.get(high, [])), mean(per_arm.get(low, []))
        ladder.append({
            "step": name,
            "families": families,
            "delta_seconds_per_token": None if a is None or b is None
            else a - b,
            "pct_of_baseline": None if a is None or b is None
            or baseline is None else 100.0 * (a - b) / baseline,
        })

    candidates = {arm: arms[arm]["delta_vs_baseline"]
                  for arm in ("c_hybrid24928", "d_hybrid8192") if arm in arms}
    threshold = None if session_null is None else 2.0 * session_null
    winners = [] if threshold is None else [
        arm for arm, delta in candidates.items()
        if delta is not None and delta < -threshold]

    gate_qualified = all(
        group.get("gate_qualified_for_timing") == "true"
        for group in groups.values() if group["legs"])

    checks = {
        "all_groups_measured": not missing,
        "all_tokens_matched": not unmatched,
        "width_histogram_identical_across_arms": len(histograms) == 1,
        "baseline_present": baseline is not None,
        "session_null_present": session_null is not None,
    }
    verdict = ("local winner" if winners
               else "not useful" if all(checks.values())
               else "invalid")

    record = {
        "experiment": "e78",
        "rung": "2b",
        "harness": "local",
        "host_chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip(),
        "head_sha": subprocess.run(["git", "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "primary_metric": PRIMARY,
        "primary_metric_direction": "minimize",
        "baseline_arm": BASELINE_ARM,
        "baseline_seconds_per_token": baseline,
        "session_null_seconds_per_token": session_null,
        "session_null_by_arm": nulls,
        "useful_effect_threshold_seconds_per_token": threshold,
        "arms": arms,
        "ladder": ladder,
        "winners": winners,
        "checks": checks,
        "missing_groups": missing,
        "unmatched_legs": unmatched,
        "width_histograms_seen": [json.loads(h) for h in sorted(histograms)],
        "gate_qualified_for_timing": gate_qualified,
        "cool_gate_passed_real_gate": gate_qualified,
        "official_or_ranked_score": False,
        "verdict": verdict,
        "groups": groups,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"e78_analyze: baseline {BASELINE_ARM} {PRIMARY}={baseline}")
    for arm, row in arms.items():
        print(f"e78_analyze: {arm:16s} {row[PRIMARY]:.9f} "
              f"delta={row['delta_vs_baseline']:+.9f} "
              f"({row['pct_vs_baseline']:+.4f} %) legs={row['legs']}")
    print(f"e78_analyze: session null {session_null} "
          f"threshold {threshold}")
    for step in ladder:
        print(f"e78_analyze: ladder {step['step']}: "
              f"{step['delta_seconds_per_token']} ({step['families']})")
    print(f"e78_analyze: checks {checks}")
    print(f"e78_analyze: verdict {verdict} winners={winners} -> {out}")

    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
