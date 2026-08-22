#!/usr/bin/env python3
"""E130 rung 10a: read the counterbalanced wired-residency ladder.

Three arms, measured on one binary that differs only by environment:

    none   no wiring, the shipped behaviour on a 48 GiB host
    s64    wiredLimit = active + 64 MiB, the shipped M5 formula
    s512   wiredLimit = active + 512 MiB, the proposed fix

`none` against `s512` is the POSITIVE CONTROL and is reported first. If wiring
the tower moves nothing on this hardware then the instrument cannot see wiring
effects at all, and `s64` against `s512` carries no information. `s64` against
`s512` is the treatment, and it is the exact change rung 10b ships.

THE HEADLINE IS ABSOLUTE CANDIDATE SECONDS PER TOKEN, NOT THE LOCAL RATIO.
Both local legs run the same candidate binary and the wiring environment
reaches the serial control worker as well as the MTP worker, so a wiring effect
moves both legs and partly cancels in `mtp_decode_speedup`. The serial leg is
reported as a co-timed control, not as a denominator.

The session is a palindrome, none s64 s512 s512 s64 none, so each arm is
measured once in each half and a monotone drift over the session cancels to
first order in the arm mean. With two observations per arm the half difference
is also the only available spread, so it is reported rather than a standard
error that two points cannot support.

Usage
-----
    python3 research/e130_rung10a_read.py --prefix e130-r10a \
        --out research/e130-artifacts/rung10a-wiring-ladder.json --wandb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ORDER = ["none", "s64", "s512", "s512", "s64", "none"]


def read_meta(path: Path) -> dict:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def swap_fields(blob: str) -> dict:
    return {
        k: int(v) for k, v in re.findall(r"([a-z_]+)=([0-9]+)", blob or "")
    }


def read_leg(root: Path, prefix: str, index: int, arm: str) -> dict:
    tag = f"{prefix}-{index}-{arm}"
    out = root / tag
    meta = read_meta(out / "meta.txt")
    leg = {
        "tag": tag,
        "index": index,
        "half": 1 if index <= 3 else 2,
        "arm": arm,
        "exit": meta.get("exit"),
        "wired_residency_active": meta.get("wired_residency_active"),
        "wired_outcome_line": meta.get("wired_outcome_line"),
        "wired_slack_mb": meta.get("wired_slack_mb"),
        "wired_gate_gib": meta.get("wired_gate_gib"),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "worker_sha256": meta.get("worker_sha256"),
        "base_sha": meta.get("base_sha"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
    }

    entry = swap_fields(meta.get("swap_entry", ""))
    exit_ = swap_fields(meta.get("swap_exit", ""))
    leg["swap_delta"] = {
        key: exit_.get(key, 0) - entry.get(key, 0)
        for key in sorted(set(entry) | set(exit_))
    }
    # A leg that swapped measured the pager, not the kernel.
    leg["swapped"] = leg["swap_delta"].get("swapouts", 0) > 0

    score_path = out / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
        metrics = score.get("metrics", {})
        leg["mtp_seconds_per_token"] = metrics.get("mtp_seconds_per_token")
        leg["serial_seconds_per_token"] = metrics.get("serial_seconds_per_token")
        leg["mtp_decode_speedup"] = metrics.get("mtp_decode_speedup")
        leg["all_tokens_matched"] = metrics.get("all_tokens_matched")
        leg["decode_tokens"] = metrics.get("decode_tokens")
    return leg


def arm_summary(legs: list[dict], arm: str, field: str) -> dict:
    values = [
        (leg["half"], leg[field])
        for leg in legs
        if leg["arm"] == arm and leg.get(field) is not None
    ]
    if not values:
        return {"arm": arm, "n": 0}
    numbers = [v for _, v in values]
    mean = sum(numbers) / len(numbers)
    by_half = {half: value for half, value in values}
    return {
        "arm": arm,
        "n": len(numbers),
        "mean": mean,
        "values": numbers,
        "half_1": by_half.get(1),
        "half_2": by_half.get(2),
        "half_spread": (
            abs(by_half[1] - by_half[2]) if 1 in by_half and 2 in by_half else None
        ),
        "half_spread_pct": (
            100.0 * abs(by_half[1] - by_half[2]) / mean
            if 1 in by_half and 2 in by_half else None
        ),
    }


def contrast(a: dict, b: dict, name: str, note: str) -> dict:
    """Percent change from `a` to `b`; negative means `b` is faster."""
    if not a.get("n") or not b.get("n"):
        return {"name": name, "usable": False, "note": note}
    delta = 100.0 * (b["mean"] - a["mean"]) / a["mean"]
    # With two points per arm the pooled half spread is the only honest scale.
    spreads = [s for s in (a.get("half_spread_pct"), b.get("half_spread_pct"))
               if s is not None]
    scale = max(spreads) if spreads else None
    return {
        "name": name,
        "usable": True,
        "note": note,
        "baseline_arm": a["arm"],
        "candidate_arm": b["arm"],
        "baseline_mean": a["mean"],
        "candidate_mean": b["mean"],
        "delta_pct": delta,
        "candidate_is_faster": delta < 0,
        "largest_half_spread_pct": scale,
        "exceeds_half_spread": abs(delta) > scale if scale is not None else None,
    }


def binary_identity(legs: list[dict]) -> dict:
    """One binary must serve every arm, or the arms are not comparable.

    `worker_sha256` is the causal identity here, not `base_sha`. A commit that
    touches only `Tests/` or `research/` between legs moves the recorded commit
    without rebuilding the worker, so the commit set is reported for audit while
    only the binary hash is allowed to gate the read.
    """
    workers = sorted({leg["worker_sha256"] for leg in legs
                      if leg.get("worker_sha256")})
    bases = sorted({leg["base_sha"] for leg in legs if leg.get("base_sha")})
    return {
        "one_binary_served_every_arm": len(workers) == 1,
        "worker_sha256_values": workers,
        "base_sha_values": bases,
        "base_sha_moved_mid_session": len(bases) > 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e130-r10a")
    ap.add_argument("--root", type=Path, default=Path("research/out"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    legs = [
        read_leg(args.root, args.prefix, index + 1, arm)
        for index, arm in enumerate(ORDER)
    ]

    field = "mtp_seconds_per_token"
    arms = {arm: arm_summary(legs, arm, field) for arm in ("none", "s64", "s512")}
    serial = {arm: arm_summary(legs, arm, "serial_seconds_per_token")
              for arm in ("none", "s64", "s512")}

    control = contrast(
        arms["none"], arms["s512"], "positive control: no wiring vs 512 MiB",
        "if this does not fire, the instrument cannot see wiring on this host "
        "and the treatment contrast carries no information",
    )
    treatment = contrast(
        arms["s64"], arms["s512"], "treatment: 64 MiB vs 512 MiB",
        "the exact change rung 10b ships, measured causally on GPU",
    )
    serial_control = contrast(
        serial["s64"], serial["s512"], "serial co-timed control: 64 MiB vs 512 MiB",
        "the serial worker wires too, so this is not a denominator",
    )

    result = {
        "experiment": "e130-rung10a",
        "question": "does raising the wired residency slack lower absolute "
                    "candidate seconds per token",
        "headline_metric": field,
        "headline_is_absolute_not_a_ratio": True,
        "order": ORDER,
        "counterbalancing": "palindrome, one observation per arm per half",
        "cool_gate_passed_real_gate": "false",
        "gate_qualified_for_timing": "false",
        "legs": legs,
        "binary_identity": binary_identity(legs),
        "arms_candidate": arms,
        "arms_serial": serial,
        "positive_control": control,
        "treatment": treatment,
        "serial_co_timed_control": serial_control,
        "any_leg_swapped": any(leg.get("swapped") for leg in legs),
        "all_tokens_matched": all(
            leg.get("all_tokens_matched") for leg in legs
            if "all_tokens_matched" in leg
        ),
        "entry_temperature_spread_c": (
            max(t for leg in legs if (t := leg.get("gpu_temp_entry_c")) is not None)
            - min(t for leg in legs if (t := leg.get("gpu_temp_entry_c")) is not None)
        ) if any(leg.get("gpu_temp_entry_c") is not None for leg in legs) else None,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")

    print("=== E130 rung 10a, absolute candidate seconds per token ===")
    for arm in ("none", "s64", "s512"):
        summary = arms[arm]
        if not summary.get("n"):
            print(f"  {arm:5s}  no usable leg")
            continue
        print(f"  {arm:5s}  n={summary['n']}  mean {summary['mean']:.8f}  "
              f"half1 {summary['half_1']}  half2 {summary['half_2']}  "
              f"half spread {summary['half_spread_pct']:.4f} %")
    print()
    for item in (control, treatment, serial_control):
        if not item.get("usable"):
            print(f"  {item['name']}: NOT USABLE")
            continue
        print(f"  {item['name']}: {item['delta_pct']:+.4f} % "
              f"(spread scale {item['largest_half_spread_pct']:.4f} %, "
              f"exceeds spread {item['exceeds_half_spread']})")
    print()
    identity = result["binary_identity"]
    print(f"  one binary        {identity['one_binary_served_every_arm']} "
          f"({len(identity['worker_sha256_values'])} worker hash, "
          f"{len(identity['base_sha_values'])} commit)")
    print(f"  any leg swapped   {result['any_leg_swapped']}")
    print(f"  tokens matched    {result['all_tokens_matched']}")
    print(f"  entry temp spread {result['entry_temperature_spread_c']} C")
    print("  cool_gate_passed_real_gate=false gate_qualified_for_timing=false")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130rung10a",
            name="e130rung10a",
            resume="allow",
            config={"experiment": "e130-rung10a", "order": ORDER},
            save_code=True,
        )
        payload = {
            "e130_rung10a_none_mtp_spt": arms["none"].get("mean"),
            "e130_rung10a_s64_mtp_spt": arms["s64"].get("mean"),
            "e130_rung10a_s512_mtp_spt": arms["s512"].get("mean"),
            "e130_rung10a_control_delta_pct": control.get("delta_pct"),
            "e130_rung10a_treatment_delta_pct": treatment.get("delta_pct"),
            "e130_rung10a_serial_control_delta_pct": serial_control.get("delta_pct"),
            "e130_rung10a_any_leg_swapped": int(bool(result["any_leg_swapped"])),
            "e130_rung10a_entry_temp_spread_c": result["entry_temperature_spread_c"],
        }
        run.log({k: v for k, v in payload.items() if v is not None})
        if args.out is not None:
            artifact = wandb.Artifact("e130-rung10a-wiring-ladder", type="analysis")
            artifact.add_file(str(args.out))
            run.log_artifact(artifact)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
