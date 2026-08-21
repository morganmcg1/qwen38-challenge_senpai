#!/usr/bin/env python3
"""Reduce the E110 rung-2 in-situ ABBA session to one publishable document.

The primary statistic is ABSOLUTE candidate MTP seconds per token, because the
ranked numerator is a runner-owned prebuilt serial baseline that no candidate
edit can move. The local serial-to-MTP ratio is reported beside it as a
secondary read: this arm lives in the wide multi-row QMV path, the local serial
leg decodes at width 1 through a different kernel family, and so the arm is
confined to the candidate MTP leg and does not cancel in the local ratio.

Within each replicate the order is base, xv4, xv4, base. Both arms therefore
have mean position 2.5 and any monotone linear drift inside the replicate
cancels exactly to first order. The replicate contrast is the arm effect and
the spread across replicates is the error bar.

    usage: research/e110_rung2_analyse.py [--replicates N] [--tokens N]
                                          [--label L]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

OUT = pathlib.Path("research/out")

# Rule 34 round frame, decode-only M = 5 on the current tree.
ROUND_US = 102_864.0
STREAM_US = 14.4123 / 179.9 * 1e6
RANKED_TRANSFER = 0.95
PROMOTION_BAR_PCT = 0.20
ADVANCE_BAR_PCT = -0.50
RUNG3_BAR_PCT = -0.40
# f4 item 8: advance when the point estimate is negative AND its upper
# confidence bound excludes a regression larger than this.
NO_REGRESSION_BAR_PCT = 0.20

T95_TWO_SIDED = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}

# Rule 35 repeatability of this local harness.
REPEATABILITY_MTP_PCT = 0.33
REPEATABILITY_SERIAL_PCT = 0.18


def read_meta(tag: str) -> dict[str, str]:
    fields = {}
    for line in (OUT / tag / "meta.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def read_score(tag: str) -> dict:
    return json.loads((OUT / tag / "score.json").read_text())["metrics"]


def as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(candidate: float, base: float) -> float:
    return 100.0 * (candidate - base) / base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--label", default="r2")
    ap.add_argument("--exactness", default="research/out/e110/rung2-exactness.json")
    ap.add_argument("--session-split", type=int, default=0,
                    help="last replicate of session 1; 0 means one session")
    args = ap.parse_args(argv)

    order = ["base", "xv4", "xv4", "base"]
    legs, missing = [], []
    for rep in range(1, args.replicates + 1):
        for position, arm in enumerate(order, start=1):
            tag = f"e110{args.label}k{rep}p{position}{arm}"
            if not (OUT / tag / "score.json").is_file():
                missing.append(tag)
                continue
            meta, score = read_meta(tag), read_score(tag)
            legs.append({
                "tag": tag,
                "replicate": rep,
                "position": position,
                "tree": arm,
                "seconds_per_token": score["mtp_seconds_per_token"],
                "serial_seconds_per_token": score["serial_seconds_per_token"],
                "local_ratio": score["mtp_decode_speedup"],
                "accepted_draft_rate": score.get("accepted_draft_rate"),
                "effective_mean_draft_len": score.get("effective_mean_draft_len"),
                "all_tokens_matched": score.get("all_tokens_matched"),
                "rounds": as_float(meta.get("trace_rounds")),
                "gpu_temp_entry_c": as_float(meta.get("gpu_temp_entry_c")),
                "gpu_temp_exit_c": as_float(meta.get("gpu_temp_exit_c")),
                "worker_sha256": meta.get("worker_sha256_pre"),
                "measured_commit_unwound": meta.get("measured_commit_unwound"),
            })
    if missing:
        print(f"e110_rung2_analyse: missing {len(missing)} leg(s): {missing}")
    if not legs:
        return 2

    def arm_mean(rep: int, arm: str, key: str) -> float:
        values = [leg[key] for leg in legs
                  if leg["replicate"] == rep and leg["tree"] == arm]
        return statistics.fmean(values)

    replicates = sorted({leg["replicate"] for leg in legs})
    per_replicate = []
    for rep in replicates:
        row = {"replicate": rep}
        for key, short in (("seconds_per_token", "mtp_spt"),
                           ("serial_seconds_per_token", "serial_spt"),
                           ("local_ratio", "ratio")):
            base, cand = arm_mean(rep, "base", key), arm_mean(rep, "xv4", key)
            row[f"{short}_base"] = base
            row[f"{short}_xv4"] = cand
            row[f"{short}_pct"] = pct(cand, base)
        # Same-arm null: the two xv4 legs sit at positions 2 and 3 and the two
        # base legs at 1 and 4, so the base pair spans the whole replicate and
        # its spread is the session drift this design removes.
        base_legs = sorted((leg for leg in legs
                            if leg["replicate"] == rep and leg["tree"] == "base"),
                           key=lambda leg: leg["position"])
        row["base_pair_drift_pct"] = pct(base_legs[-1]["seconds_per_token"],
                                         base_legs[0]["seconds_per_token"])
        per_replicate.append(row)

    def pooled(short: str) -> dict:
        values = [row[f"{short}_pct"] for row in per_replicate]
        n = len(values)
        mean = statistics.fmean(values)
        spread = statistics.stdev(values) if n > 1 else 0.0
        sem = spread / n ** 0.5 if n > 1 else None
        half = T95_TWO_SIDED.get(n - 1, 1.96) * sem if sem is not None else None
        return {
            "mean_pct": mean,
            "median_pct": statistics.median(values),
            "stdev_pct": spread,
            "sem_pct": sem,
            "t95_half_width_pct": half,
            "ci95_lower_pct": mean - half if half is not None else None,
            "ci95_upper_pct": mean + half if half is not None else None,
            "min_pct": min(values),
            "max_pct": max(values),
            "n_replicates": n,
        }

    mtp, serial, ratio = pooled("mtp_spt"), pooled("serial_spt"), pooled("ratio")
    headline = mtp["mean_pct"]

    # Replicates pool across sessions only if the sessions agree, so report
    # them separately and let that assumption be checked rather than assumed.
    per_session = []
    if args.session_split:
        for index, keep in ((1, False), (2, True)):
            rows = [row for row in per_replicate
                    if (row["replicate"] > args.session_split) == keep]
            if rows:
                per_session.append({
                    "session": index,
                    "replicates": [row["replicate"] for row in rows],
                    "n": len(rows),
                    "mtp_spt_pct_mean": statistics.fmean(
                        row["mtp_spt_pct"] for row in rows),
                })

    entry = [leg["gpu_temp_entry_c"] for leg in legs
             if leg["gpu_temp_entry_c"] is not None]
    exit_c = [leg["gpu_temp_exit_c"] for leg in legs
              if leg["gpu_temp_exit_c"] is not None]

    exact_path = pathlib.Path(args.exactness)
    exact_doc = json.loads(exact_path.read_text()) if exact_path.is_file() else {}
    exactness = [{
        "check": rec["check"],
        "rows": rec.get("rows"),
        "expected_digest": rec.get("expected_digest"),
        "observed_digest": rec.get("observed_digest"),
        "passed": rec["passed"],
    } for rec in exact_doc.get("checks", [])]
    exactness.append({
        "check": "abba_leg_token_match",
        "rows": len(legs),
        "expected_digest": "all_tokens_matched=True on every timed leg",
        "observed_digest": str(sum(1 for leg in legs if leg["all_tokens_matched"])),
        "passed": all(leg["all_tokens_matched"] for leg in legs),
    })

    exact_ok = all(rec["passed"] for rec in exactness)
    doc = {
        "arm": "xv4",
        "experiment": "e110-rung2-insitu",
        "harness": "local",
        "candidate_commit": read_meta(legs[0]["tag"]).get("branch_commit"),
        "base_commit": next(leg["measured_commit_unwound"] for leg in legs
                            if leg["tree"] == "base"),
        "worker_fingerprint": next(leg["worker_sha256"] for leg in legs
                                   if leg["tree"] == "xv4"),
        "token_window": args.tokens,
        "order": "base, xv4, xv4, base per replicate",
        "replicates": len(replicates),
        "reproduction":
            f"research/e110_rung2_exact.sh {args.tokens} && "
            f"research/e110_rung2_abba.sh {len(replicates)} {args.tokens} "
            f"{args.label} && research/e110_rung2_analyse.py "
            f"--replicates {len(replicates)} --tokens {args.tokens}",
        "legs": legs,
        "per_replicate": per_replicate,
        "exactness": exactness,
        "summary": {
            "primary_metric": "candidate_mtp_seconds_per_token_pct_vs_base",
            "mtp_spt_pct_mean": mtp["mean_pct"],
            "mtp_spt_pct_median": mtp["median_pct"],
            "mtp_spt_pct_stdev": mtp["stdev_pct"],
            "mtp_spt_pct_sem": mtp["sem_pct"],
            "mtp_spt_pct_t95_half_width": mtp["t95_half_width_pct"],
            "mtp_spt_pct_ci95_lower": mtp["ci95_lower_pct"],
            "mtp_spt_pct_ci95_upper": mtp["ci95_upper_pct"],
            "mtp_spt_pct_min": mtp["min_pct"],
            "mtp_spt_pct_max": mtp["max_pct"],
            "mtp_spt_base_mean_s": statistics.fmean(
                leg["seconds_per_token"] for leg in legs if leg["tree"] == "base"),
            "mtp_spt_xv4_mean_s": statistics.fmean(
                leg["seconds_per_token"] for leg in legs if leg["tree"] == "xv4"),
            "serial_spt_pct_mean": serial["mean_pct"],
            "local_ratio_pct_mean": ratio["mean_pct"],
            "local_ratio_base_mean": statistics.fmean(
                leg["local_ratio"] for leg in legs if leg["tree"] == "base"),
            "local_ratio_xv4_mean": statistics.fmean(
                leg["local_ratio"] for leg in legs if leg["tree"] == "xv4"),
            "base_pair_drift_pct_mean": statistics.fmean(
                row["base_pair_drift_pct"] for row in per_replicate),
            "round_frame_pct": headline * STREAM_US / ROUND_US,
            "ranked_frame_pct": headline * STREAM_US / ROUND_US * RANKED_TRANSFER,
            "promotion_bar_pct": PROMOTION_BAR_PCT,
            "advance_bar_pct": ADVANCE_BAR_PCT,
            "rung3_bar_pct": RUNG3_BAR_PCT,
            "repeatability_mtp_pct": REPEATABILITY_MTP_PCT,
            "repeatability_serial_pct": REPEATABILITY_SERIAL_PCT,
            "clears_rung3_bar": headline <= RUNG3_BAR_PCT,
            "no_regression_bar_pct": NO_REGRESSION_BAR_PCT,
            "clears_no_regression_rule": bool(
                headline < 0.0
                and mtp["ci95_upper_pct"] is not None
                and mtp["ci95_upper_pct"] < NO_REGRESSION_BAR_PCT),
            "per_session": per_session,
            "exactness_passed": exact_ok,
            "n_legs": len(legs),
            "n_legs_missing": len(missing),
            "gpu_temp_entry_min_c": min(entry) if entry else None,
            "gpu_temp_entry_max_c": max(entry) if entry else None,
            "gpu_temp_entry_spread_c": (max(entry) - min(entry)) if entry else None,
            "gpu_temp_exit_min_c": min(exit_c) if exit_c else None,
            "gpu_temp_exit_max_c": max(exit_c) if exit_c else None,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "timing_valid": False,
        },
    }

    out = OUT / "e110/rung2-insitu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    meta_dir = OUT / "e110-rung2"
    meta_dir.mkdir(parents=True, exist_ok=True)
    first = read_meta(legs[0]["tag"])
    (meta_dir / "meta.txt").write_text("\n".join(
        f"{key}={first.get(key, '')}" for key in (
            "experiment", "harness", "tokens", "local_mode", "sandbox",
            "cool_gate", "cool_gate_passed_real_gate",
            "gate_qualified_for_timing", "official_or_ranked_score",
            "host", "chip", "memory_bytes", "metallib_source_fingerprint",
            "head_dir", "branch_commit")) + "\n")

    print(f"legs {len(legs)} replicates {len(replicates)}")
    for row in per_replicate:
        print(f"  k{row['replicate']}: mtp {row['mtp_spt_pct']:+.3f} % "
              f"serial {row['serial_spt_pct']:+.3f} % "
              f"ratio {row['ratio_pct']:+.3f} % "
              f"base-pair drift {row['base_pair_drift_pct']:+.3f} %")
    for row in per_session:
        print(f"  session {row['session']} k{row['replicates']}: "
              f"mtp {row['mtp_spt_pct_mean']:+.3f} %")
    print(f"POOLED absolute candidate MTP s/token {headline:+.3f} % "
          f"(sd {mtp['stdev_pct']:.3f}, n {mtp['n_replicates']})")
    if mtp["t95_half_width_pct"] is not None:
        print(f"  95 % CI [{mtp['ci95_lower_pct']:+.3f}, "
              f"{mtp['ci95_upper_pct']:+.3f}] % "
              f"(t95 half-width {mtp['t95_half_width_pct']:.3f})")
    print("  no-regression rule "
          + ("CLEARED" if doc["summary"]["clears_no_regression_rule"]
             else "NOT cleared")
          + f" (upper bound must be < {NO_REGRESSION_BAR_PCT:+.2f} %)")
    print(f"  round frame {doc['summary']['round_frame_pct']:+.3f} % "
          f"ranked {doc['summary']['ranked_frame_pct']:+.3f} %")
    print(f"  exactness {'PASS' if exact_ok else 'FAIL'}; "
          f"rung-3 bar {'CLEARED' if headline <= RUNG3_BAR_PCT else 'NOT cleared'}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
