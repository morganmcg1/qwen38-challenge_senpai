#!/usr/bin/env python3
"""E187 Stage 1: score the GDN recurrent-state storage gate.

    usage: research/e187_gate_analyze.py --baseline GOLDEN.json \
               --arm NAME=GOLDEN.json [--arm NAME=GOLDEN.json ...] \
               [--output REPORT.json] [--wandb] [--run-name NAME]

Every golden is one offline `mtp-verify --generate` chain on the public
fixture. The baseline arm is the pinned fp32 state; each other arm stores the
state differently or carries an injected perturbation. All numbers are
`harness=local` offline replay evidence. Nothing here is timed and nothing
here is a score.

Definitions.
  * flip           an arm row whose argmax differs from the baseline row.
  * top-two flip   an arm row whose ordered top-2 token pair differs.
  * margin         top1_logit - top2_logit of one row.
  * bf16 ulp       2**(floor(log2(|top1_logit|)) - 7), the natural grain of the
                   bf16 logits the row is read from.
  * erosion        baseline margin minus arm margin, in bf16 ulps. Positive
                   erosion means the arm moved the row closer to a tie.

Rows are only comparable while both chains carry the same prefix, so the
comparison window ends at the first flip. That window is reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_rows(path: Path) -> dict:
    doc = json.loads(path.read_text())
    rows = doc["rows"]
    return {
        "path": str(path),
        "seed_token": doc.get("reference_seed_token"),
        "self_consistent": doc.get("reference_self_consistent"),
        "emitted": doc.get("emitted_tokens") or [],
        "argmax": [r["sequential_argmax"] for r in rows],
        "top2_tokens": [r.get("top2_tokens") or [] for r in rows],
        "top2_logits": [r.get("top2_logits") or [] for r in rows],
    }


def bf16_ulp(value: float) -> float:
    if value == 0.0 or not math.isfinite(value):
        return float("nan")
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def margins(arm: dict) -> list[float | None]:
    out: list[float | None] = []
    for pair in arm["top2_logits"]:
        out.append(pair[0] - pair[1] if len(pair) >= 2 else None)
    return out


def margins_in_ulps(arm: dict) -> list[float | None]:
    out: list[float | None] = []
    for pair in arm["top2_logits"]:
        if len(pair) < 2:
            out.append(None)
            continue
        ulp = bf16_ulp(pair[0])
        out.append((pair[0] - pair[1]) / ulp if ulp == ulp else None)
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def baseline_margin_profile(base: dict) -> dict:
    ulps = [m for m in margins_in_ulps(base) if m is not None]
    raw = [m for m in margins(base) if m is not None]
    buckets = {}
    for edge in (0, 1, 2, 4, 8, 16, 32, 64):
        buckets[f"rows_margin_le_{edge}_ulp"] = sum(1 for m in ulps if m <= edge)
    return {
        "rows": len(ulps),
        "min_margin_ulp": min(ulps) if ulps else None,
        "p01_margin_ulp": percentile(ulps, 0.01),
        "p10_margin_ulp": percentile(ulps, 0.10),
        "median_margin_ulp": percentile(ulps, 0.50),
        "min_margin_abs": min(raw) if raw else None,
        "median_margin_abs": percentile(raw, 0.50),
        **buckets,
    }


def compare(base: dict, arm: dict, name: str) -> dict:
    n = min(len(base["argmax"]), len(arm["argmax"]))
    first_flip = None
    flips = []
    top2_flips = []
    for i in range(n):
        if base["argmax"][i] != arm["argmax"][i]:
            flips.append(i)
            if first_flip is None:
                first_flip = i
        if base["top2_tokens"][i] != arm["top2_tokens"][i]:
            top2_flips.append(i)

    # Rows stay comparable only while the two chains share their prefix.
    window = n if first_flip is None else first_flip + 1

    base_ulp = margins_in_ulps(base)
    arm_ulp = margins_in_ulps(arm)
    base_abs = margins(base)
    arm_abs = margins(arm)

    erosion_ulp: list[float] = []
    erosion_abs: list[float] = []
    residual_ulp: list[float] = []
    identical_top1 = 0
    identical_rows = 0
    first_divergence: int | None = None
    for i in range(window):
        if base_ulp[i] is None or arm_ulp[i] is None:
            continue
        erosion_ulp.append(base_ulp[i] - arm_ulp[i])
        erosion_abs.append(base_abs[i] - arm_abs[i])
        residual_ulp.append(arm_ulp[i])
        if base["top2_logits"][i][:1] == arm["top2_logits"][i][:1]:
            identical_top1 += 1
        if base["top2_logits"][i] == arm["top2_logits"][i]:
            identical_rows += 1
        elif first_divergence is None:
            first_divergence = i

    abs_erosion = [abs(e) for e in erosion_ulp]
    max_erosion = max(abs_erosion) if abs_erosion else None
    min_residual = min(residual_ulp) if residual_ulp else None

    # How close did the arm come to a flip, and how large is the top-1 logit
    # perturbation the storage change produced?
    top1_delta_ulp: list[float] = []
    for i in range(window):
        if not base["top2_logits"][i] or not arm["top2_logits"][i]:
            continue
        ulp = bf16_ulp(base["top2_logits"][i][0])
        if ulp == ulp:
            top1_delta_ulp.append(
                abs(arm["top2_logits"][i][0] - base["top2_logits"][i][0]) / ulp
            )
    residual_buckets = {
        f"arm_rows_margin_le_{edge}_ulp": sum(1 for m in residual_ulp if m <= edge)
        for edge in (0, 1, 2, 4, 8, 16, 32, 64)
    }
    rows_erosion_exceeds_margin = sum(
        1
        for i in range(len(erosion_ulp))
        if erosion_ulp[i] >= (base_ulp[i] if base_ulp[i] is not None else 0)
    )

    # Cliff distance. A row flips when the erosion it draws exceeds its own
    # margin. Pairing every measured margin with every measured erosion treats
    # the two as independent, which is the honest way to ask how close this
    # fixture came to a flip that a different prompt would have taken.
    margins_sample = [m for m in base_ulp if m is not None]
    flip_pairs = 0
    if margins_sample and erosion_ulp:
        ordered_erosion = sorted(erosion_ulp)
        for margin in margins_sample:
            low, high = 0, len(ordered_erosion)
            while low < high:
                mid = (low + high) // 2
                if ordered_erosion[mid] > margin:
                    high = mid
                else:
                    low = mid + 1
            flip_pairs += len(ordered_erosion) - low
    flip_probability = (
        flip_pairs / (len(margins_sample) * len(erosion_ulp))
        if margins_sample and erosion_ulp
        else None
    )

    # Drift versus position: mean absolute erosion over 64-row buckets.
    drift = []
    for start in range(0, window, 64):
        chunk = abs_erosion[start : start + 64]
        if chunk:
            drift.append(
                {
                    "start": start,
                    "rows": len(chunk),
                    "mean_abs_erosion_ulp": sum(chunk) / len(chunk),
                    "max_abs_erosion_ulp": max(chunk),
                }
            )

    histogram = {}
    for edge in (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32):
        histogram[f"abs_erosion_le_{edge}_ulp"] = sum(
            1 for e in abs_erosion if e <= edge
        )

    headroom = None
    if max_erosion not in (None, 0) and min_residual is not None:
        headroom = min_residual / max_erosion

    return {
        "arm": name,
        "path": arm["path"],
        "rows_compared": n,
        "comparison_window": window,
        "self_consistent": arm["self_consistent"],
        "seed_token_matches_baseline": arm["seed_token"] == base["seed_token"],
        "flip_count": len(flips),
        "first_flip_index": first_flip,
        "flip_indices_head": flips[:20],
        "top2_order_flip_count": len(top2_flips),
        "top2_order_first_flip_index": top2_flips[0] if top2_flips else None,
        "rows_bit_identical": identical_rows,
        "first_divergence_index": first_divergence,
        "rows_top1_logit_identical": identical_top1,
        "max_abs_erosion_ulp": max_erosion,
        "mean_abs_erosion_ulp": (
            sum(abs_erosion) / len(abs_erosion) if abs_erosion else None
        ),
        "p99_abs_erosion_ulp": percentile(abs_erosion, 0.99),
        "min_residual_margin_ulp": min_residual,
        "headroom_ratio_min_residual_over_max_erosion": headroom,
        "max_top1_logit_delta_ulp": max(top1_delta_ulp) if top1_delta_ulp else None,
        "mean_top1_logit_delta_ulp": (
            sum(top1_delta_ulp) / len(top1_delta_ulp) if top1_delta_ulp else None
        ),
        "p99_top1_logit_delta_ulp": percentile(top1_delta_ulp, 0.99),
        "rows_where_erosion_reaches_baseline_margin": rows_erosion_exceeds_margin,
        "independent_pairing_flip_probability_per_row": flip_probability,
        "expected_flips_per_4096_row_ranked_run": (
            None if flip_probability is None else flip_probability * 4096
        ),
        "arm_margin_histogram": residual_buckets,
        "erosion_histogram": histogram,
        "drift_by_position": drift,
    }


def verdict(result: dict, criterion: float) -> str:
    if result["flip_count"] > 0 or result["top2_order_flip_count"] > 0:
        return "fail"
    headroom = result["headroom_ratio_min_residual_over_max_erosion"]
    if headroom is None:
        return "pass"
    return "pass" if headroom >= criterion else "unclear"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--criterion", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="e187-gate")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    base = load_rows(args.baseline)
    report = {
        "experiment": "e187-bf16-gdn-state-gate",
        "stage": 1,
        "harness": "local",
        "timed": False,
        "baseline": {
            "path": str(args.baseline),
            "self_consistent": base["self_consistent"],
            "rows": len(base["argmax"]),
            "margin_profile": baseline_margin_profile(base),
        },
        "arms": [],
    }
    for spec in args.arm:
        name, _, path = spec.partition("=")
        result = compare(base, load_rows(Path(path)), name)
        result["verdict"] = verdict(result, args.criterion)
        result["criterion_min_residual_over_max_erosion"] = args.criterion
        report["arms"].append(result)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            name=args.run_name,
            notes=args.notes,
            job_type="numerical-gate",
            tags=["e187", "gdn-state", "bf16", "harness=local", "offline-replay"],
            config={
                "experiment": "e187-bf16-gdn-state-gate",
                "stage": 1,
                "harness": "local",
                "timed": False,
                "fixture": "public_longcopy_gate_english_512_256",
                "criterion_min_residual_over_max_erosion": args.criterion,
                "baseline_path": str(args.baseline),
                "arms": [spec.split("=", 1)[0] for spec in args.arm],
            },
        )
        summary = {
            f"baseline/{k}": v
            for k, v in report["baseline"]["margin_profile"].items()
        }
        for result in report["arms"]:
            name = result["arm"]
            for key in (
                "flip_count",
                "first_flip_index",
                "top2_order_flip_count",
                "top2_order_first_flip_index",
                "comparison_window",
                "rows_bit_identical",
                "first_divergence_index",
                "rows_top1_logit_identical",
                "max_abs_erosion_ulp",
                "mean_abs_erosion_ulp",
                "p99_abs_erosion_ulp",
                "min_residual_margin_ulp",
                "headroom_ratio_min_residual_over_max_erosion",
                "max_top1_logit_delta_ulp",
                "mean_top1_logit_delta_ulp",
                "p99_top1_logit_delta_ulp",
                "rows_where_erosion_reaches_baseline_margin",
                "independent_pairing_flip_probability_per_row",
                "expected_flips_per_4096_row_ranked_run",
                "verdict",
            ):
                summary[f"{name}/{key}"] = result[key]
            for key, value in result["erosion_histogram"].items():
                summary[f"{name}/hist/{key}"] = value
            for key, value in result["arm_margin_histogram"].items():
                summary[f"{name}/hist/{key}"] = value
            for point in result["drift_by_position"]:
                run.log(
                    {
                        f"{name}/drift_mean_abs_erosion_ulp": point[
                            "mean_abs_erosion_ulp"
                        ],
                        f"{name}/drift_max_abs_erosion_ulp": point[
                            "max_abs_erosion_ulp"
                        ],
                        "position": point["start"],
                    }
                )
        run.summary.update(summary)
        artifact = wandb.Artifact("e187-gate-report", type="analysis")
        with artifact.new_file("e187-gate-report.json") as handle:
            handle.write(text)
        run.log_artifact(artifact)
        print(f"wandb_run_url={run.url}")
        print(f"wandb_run_id={run.id}")
        run.finish()


if __name__ == "__main__":
    main()
