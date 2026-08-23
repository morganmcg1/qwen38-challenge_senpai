#!/usr/bin/env python3
"""Reduce the E147 rung A ABBA legs to one prefill contrast per prompt.

The primary quantity is `seed_prefill_seconds`, which the trusted CLI reports
for the same charged window it later divides into
`parent_measured_seconds_per_token`. Both arms run the identical 512-token
seed, so the prefill work is the same and the contrast prices the kernel
schedule alone.

Within one replicate the arms run base, cand, cand, base, so each arm has mean
position 2.5 and monotone thermal drift cancels to first order. The per-
replicate contrast is therefore formed inside the replicate and only then
averaged, never pooled across replicates first.

Usage:
    research/e147_rungA_report.py --label s1 --runs <dir> --out <json>
"""

import argparse
import json
import pathlib
import statistics


LEG_NUMERIC = (
    "seed_prefill_seconds",
    "prefill_seconds_per_token",
    "parent_measured_seconds_per_token",
    "decode_seconds",
    "round_count",
    "effective_mean_draft_len",
    "accepted_draft_rate",
)


def read_meta(path):
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def collect(runs_dir, label):
    legs = []
    for meta_path in sorted(pathlib.Path(runs_dir).glob("*/*/meta.txt")):
        meta = read_meta(meta_path)
        if meta.get("e147_witness") is None:
            continue
        slot = meta_path.parent.parent.name
        if not slot.startswith(label):
            continue
        report_path = meta_path.parent / "report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        leg = {
            "slot": slot,
            "prompt": meta_path.parent.name,
            "arm": meta["e147_arm_requested"],
            "replicate": int(meta["e147_replicate"]),
            "position": int(meta["e147_position"]),
            "witness": meta["e147_witness"],
            "worker_sha256": meta.get("e147_arm_worker_sha256_got"),
            "all_tokens_matched": report.get("all_tokens_matched"),
            "residual_divergence_count": report.get("residual_divergence_count"),
            "decode_token_count": report.get("decode_token_count"),
            "seed_token_count": report.get("seed_token_count"),
            "entry_c": float(meta["gpu_temp_entry_c"]) if meta.get("gpu_temp_entry_c") else None,
            "exit_c": float(meta["gpu_temp_exit_c"]) if meta.get("gpu_temp_exit_c") else None,
            "timing_valid": meta.get("timing_valid"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "commit": meta.get("e147_session_commit"),
            "base_ref": meta.get("e147_base_ref"),
            "golden_sha256": meta.get("golden_sha256"),
            "chip": meta.get("chip"),
            "host": meta.get("host"),
            "memory_gib": meta.get("memory_gib"),
            "started_utc": meta.get("started_utc"),
        }
        for key in LEG_NUMERIC:
            value = report.get(key)
            leg[key] = float(value) if value is not None else None
        legs.append(leg)
    return legs


def contrast(legs, metric):
    """Mean percentage change of cand against base, formed inside each block.

    One block is one (replicate, prompt) ABBA quartet. Pooling across prompts
    before differencing would let a prompt's absolute level leak into the
    contrast, so each block is reduced first and only the block contrasts are
    averaged.
    """
    blocks = []
    keys = sorted({(leg["replicate"], leg["prompt"]) for leg in legs})
    for key in keys:
        arms = {"base": [], "cand": []}
        for leg in legs:
            if (leg["replicate"], leg["prompt"]) != key or leg["witness"] != "ok":
                continue
            if leg[metric] is None:
                continue
            arms[leg["arm"]].append(leg[metric])
        if len(arms["base"]) != 2 or len(arms["cand"]) != 2:
            continue
        base = statistics.fmean(arms["base"])
        cand = statistics.fmean(arms["cand"])
        blocks.append(100.0 * (cand / base - 1.0))
    if not blocks:
        return None, [], None
    stdev = statistics.stdev(blocks) if len(blocks) > 1 else 0.0
    return statistics.fmean(blocks), blocks, stdev


def summarise(legs):
    out = {}
    ok = [leg for leg in legs if leg["witness"] == "ok"]
    entries = [leg["entry_c"] for leg in ok if leg["entry_c"] is not None]
    out["leg_count"] = len(legs)
    out["witness_mismatches"] = sum(1 for leg in legs if leg["witness"] != "ok")
    out["divergences"] = sum(
        (leg["residual_divergence_count"] or 0) for leg in legs
    )
    out["unmatched_legs"] = sum(
        1 for leg in legs if leg["all_tokens_matched"] is not True
    )
    if entries:
        out["entry_temp_c_min"] = min(entries)
        out["entry_temp_c_max"] = max(entries)
        out["entry_temp_c_spread"] = max(entries) - min(entries)
    for metric, name in (
        ("seed_prefill_seconds", "prefill"),
        ("parent_measured_seconds_per_token", "decode_spt"),
        ("decode_seconds", "decode_seconds"),
    ):
        mean, blocks, stdev = contrast(ok, metric)
        out[f"{name}_gain_pct"] = mean
        out[f"{name}_gain_pct_per_block"] = blocks
        out[f"{name}_gain_pct_stdev"] = stdev
        for arm in ("base", "cand"):
            values = [leg[metric] for leg in ok if leg["arm"] == arm and leg[metric] is not None]
            out[f"{name}_{arm}_mean"] = statistics.fmean(values) if values else None
    return out


def fmt(value, places=4):
    return "n/a" if value is None else f"{value:+.{places}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    legs = collect(args.runs, args.label)
    prompts = sorted({leg["prompt"] for leg in legs})
    payload = {
        "experiment": "e147-nax-seed-prefill-double-buffer",
        "rung": "A",
        "label": args.label,
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_or_ranked_score": False,
        "estimator": "within-replicate ABBA contrast, mean position 2.5 per arm",
        "arm_selector": "whole worker binary; the quantized family is JIT-compiled from the string in the binary",
        "identity": {
            "commit": sorted({leg["commit"] for leg in legs}),
            "base_ref": sorted({leg["base_ref"] for leg in legs}),
            "worker_sha256": sorted({leg["worker_sha256"] for leg in legs}),
            "chip": sorted({leg["chip"] for leg in legs}),
            "host": sorted({leg["host"] for leg in legs}),
            "memory_gib": sorted({leg["memory_gib"] for leg in legs}),
            "golden_sha256": sorted({leg["golden_sha256"] for leg in legs}),
        },
        "legs": legs,
        "prompts": {
            prompt: summarise([leg for leg in legs if leg["prompt"] == prompt])
            for prompt in prompts
        },
        "pooled": summarise(legs),
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"e147 rung A -> {args.out}")
    for prompt in prompts:
        row = payload["prompts"][prompt]
        print(
            f"  {prompt}: prefill {fmt(row['prefill_gain_pct'])} % "
            f"(sd {fmt(row['prefill_gain_pct_stdev'])}), "
            f"decode_spt {fmt(row['decode_spt_gain_pct'])} %, "
            f"divergences {row['divergences']}, "
            f"unmatched {row['unmatched_legs']}, "
            f"witness mismatches {row['witness_mismatches']}"
        )
    pooled = payload["pooled"]
    print(
        f"  pooled: prefill {fmt(pooled['prefill_gain_pct'])} %, "
        f"base {fmt(pooled['prefill_base_mean'], 6)} s, "
        f"cand {fmt(pooled['prefill_cand_mean'], 6)} s"
    )


if __name__ == "__main__":
    main()
