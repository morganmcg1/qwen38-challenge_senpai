"""E134 item 1 -- score the warm-refill arms collected by e134_warm_session.sh.

Zero GPU. Reads .mlxfast-private/e134/runs-<arm>/<leg>/ and reports, per leg:

  seed_prefill_seconds   the first timed forward after warm
  first_block_seconds    the first scored round, as the parent timed it
  E@1 width-matched      round 1 minus the median of the tail rounds that
                         dispatch the same width, from trace.txt
  a mid-leg control that must return approximately zero
  warm shapes_ms and refill_ms, from the untimed warm telemetry line

and then the arm contrasts that answer the mechanism question:

  clear - base           what the residency Memory.clearCache() costs
  clearrefill - clear    what the refill returns of that cost
  refill - base          the shipped-default change on a host whose residency
                         path never runs, which must be approximately zero

Usage:
  python3 e134_warm_arms.py --json e134-artifacts/item1-warm-arms.json
"""

import argparse
import json
import os
import statistics

import e134_rung5_round1 as r5

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRIV = os.path.join(ROOT, ".mlxfast-private", "e134")

ARMS = ["base", "refill", "clear", "clearrefill"]
CONTRASTS = [
    ("clear", "base", "cost of the residency allocator clear"),
    ("clearrefill", "clear", "what the refill returns of that cost"),
    ("clearrefill", "base", "ranked-like state with the fix, against shipped"),
    ("refill", "base", "shipped default on a host with no residency clear"),
]


def read_meta(path):
    meta = {}
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            key, _, value = line.rstrip("\n").partition("=")
            if key:
                meta[key] = value
    return meta


def leg_record(arm, leg, directory):
    meta_path = os.path.join(directory, "meta.txt")
    trace_path = os.path.join(directory, "trace.txt")
    if not os.path.exists(meta_path):
        return None
    meta = read_meta(meta_path)
    if meta.get("exit") != "0":
        return None
    record = {
        "arm": arm,
        "leg": leg,
        "prompt": r5.LEG_PROMPT.get(leg),
        "all_tokens_matched": meta.get("all_tokens_matched"),
        "residual_divergence_count": meta.get("residual_divergence_count"),
        "seed_prefill_s": float(meta["seed_prefill_seconds"]),
        "first_block_s": float(meta["first_block_seconds"]),
        "decode_s": float(meta["decode_seconds"]),
        "p50_block_s": float(meta["p50_block_request_seconds"]),
        "spt": float(meta["parent_measured_seconds_per_token"]),
        "round_count": int(meta["round_count"]),
        "mean_draft": float(meta["effective_mean_draft_len"]),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "warm_shapes_ms": meta.get("warm_shapes_ms"),
        "warm_refill_ms": meta.get("refill_ms"),
        "warm_refill_on": meta.get("refill"),
        "warm_emulated_clear": meta.get("emulated_clear"),
        "cache_after_sizing": meta.get("cache_after_sizing"),
        "cache_end": meta.get("cache_end"),
    }
    if os.path.exists(trace_path):
        matched = r5.analyse_leg(leg, trace_path, seed=0, match_width=True)
        pooled = r5.analyse_leg(leg, trace_path, seed=0, match_width=False)
        if matched:
            record["E1_width_matched_us"] = matched["E_total_round1_us"]
            record["E1_baseline_rounds"] = matched["baseline_rounds"]
            record["round1_d"] = matched["round1_d"]
            record["gpu_excess_us"] = matched["gpu_excess_us"]
            record["host_cpu_excess_ns"] = matched["host_cpu_excess_ns"]
            record["control_us"] = matched["control_midleg_median_us"]
            record["control_p95_us"] = matched["control_midleg_p95_abs_us"]
        if pooled:
            record["E1_pooled_us"] = pooled["E_total_round1_us"]
    return record


def collect():
    legs = {}
    for arm in ARMS:
        arm_root = os.path.join(PRIV, f"runs-{arm}")
        if not os.path.isdir(arm_root):
            continue
        for leg in sorted(os.listdir(arm_root)):
            directory = os.path.join(arm_root, leg)
            if not os.path.isdir(directory):
                continue
            record = leg_record(arm, leg, directory)
            if record:
                legs.setdefault(leg, {})[arm] = record
    return legs


def contrast(legs, arm_a, arm_b, field):
    """Paired within-leg differences arm_a minus arm_b for one field."""
    pairs = []
    for leg, arms in sorted(legs.items()):
        if arm_a in arms and arm_b in arms:
            va = arms[arm_a].get(field)
            vb = arms[arm_b].get(field)
            if va is None or vb is None:
                continue
            pairs.append({"leg": leg, "a": va, "b": vb, "delta": va - vb})
    if not pairs:
        return None
    deltas = [p["delta"] for p in pairs]
    return {
        "n": len(pairs),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "all_same_sign": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    legs = collect()
    if not legs:
        raise SystemExit("e134_warm_arms: no legs under " + PRIV)

    print("=== per leg ===")
    header = (
        f"{'leg':18s} {'arm':12s} {'prefill_s':>9s} {'block1_s':>9s} "
        f"{'E1_wm_ms':>9s} {'ctrl_ms':>8s} {'nbase':>5s} {'d1':>3s} "
        f"{'spt':>9s} {'warm_ms':>8s} {'refill_ms':>9s} {'match':>6s}"
    )
    print(header)
    for leg, arms in sorted(legs.items()):
        for arm in ARMS:
            record = arms.get(arm)
            if not record:
                continue
            e1 = record.get("E1_width_matched_us")
            ctrl = record.get("control_us")
            print(
                f"{leg:18s} {arm:12s} {record['seed_prefill_s']:9.4f} "
                f"{record['first_block_s']:9.4f} "
                f"{(e1 / 1000.0 if e1 is not None else float('nan')):9.2f} "
                f"{(ctrl / 1000.0 if ctrl is not None else float('nan')):8.2f} "
                f"{str(record.get('E1_baseline_rounds', '')):>5s} "
                f"{str(record.get('round1_d', '')):>3s} "
                f"{record['spt']:9.6f} "
                f"{str(record.get('warm_shapes_ms', '')):>8s} "
                f"{str(record.get('warm_refill_ms', '')):>9s} "
                f"{str(record.get('all_tokens_matched', '')):>6s}"
            )

    fields = [
        ("seed_prefill_s", "seed prefill s"),
        ("first_block_s", "first block s"),
        ("E1_width_matched_us", "round-1 excess us, width matched"),
        ("spt", "seconds per token"),
        ("p50_block_s", "p50 block s"),
    ]
    results = {"legs": legs, "contrasts": {}}
    print("\n=== paired arm contrasts ===")
    for arm_a, arm_b, why in CONTRASTS:
        print(f"\n{arm_a} - {arm_b}   ({why})")
        for field, label in fields:
            value = contrast(legs, arm_a, arm_b, field)
            results["contrasts"].setdefault(f"{arm_a}-{arm_b}", {})[field] = value
            if not value:
                continue
            print(
                f"  {label:34s} n={value['n']} mean={value['mean_delta']:+.6g} "
                f"median={value['median_delta']:+.6g} sd={value['sd_delta']:.6g} "
                f"same_sign={value['all_same_sign']}"
            )

    unmatched = [
        (leg, arm)
        for leg, arms in legs.items()
        for arm, record in arms.items()
        if record.get("all_tokens_matched") != "true"
    ]
    results["exactness_failures"] = unmatched
    print(
        "\nexactness: "
        + (
            "every leg matched the golden trajectory"
            if not unmatched
            else f"FAILURES {unmatched}"
        )
    )

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(results, handle, indent=1, sort_keys=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
