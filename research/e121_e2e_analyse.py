#!/usr/bin/env python3
"""Reduce the E121 rung-3 in-situ ABBA session to one publishable document.

THE PRIMARY STATISTIC IS ABSOLUTE CANDIDATE MTP SECONDS PER TOKEN. The ranked
numerator is a runner-owned prebuilt serial baseline that no candidate edit can
move, so `d ln(ranked baseline serial time) / dx = 0` and any reduction in
candidate seconds per token raises every affected ranked `raw_p`. The local
serial-to-MTP ratio is reported beside it as a secondary read only.

For THIS arm the local ratio also happens to be informative, because the arm is
confined to the wide multi-row QMV path (`ntg.x >= 3`) while the local serial
leg decodes at width 1 through a different kernel family. The two reads are
therefore expected to agree in sign, and a disagreement is itself a finding.

Within each replicate the order is base, share, share, base. Both arms have
mean position 2.5, so monotone linear drift inside the replicate cancels
exactly to first order. The replicate contrast is the arm effect and the spread
across replicates is the error bar.

    usage: research/e121_e2e_analyse.py [--replicates N] [--tokens N]
                                        [--label L]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

OUT = pathlib.Path("research/out")

# E116 rule: measured transfer from a kernel-frame percent to a leg-seconds
# percent on this tree. Rung 2 measured the kernel frame; rung 3 measures the
# leg directly, so this constant is used ONLY to state the pre-registered
# prediction that rung 3 tests.
E116_KERNEL_TO_LEG = 0.607
# Rule 34: local leg percent -> ranked percent.
RANKED_TRANSFER = 0.95

# Pre-registered before the session ran, from the rung-2 cost-weighted shipped
# frame. Rung 2 reported +1.482 %. The shipped dispatcher sends verify width
# M = 2 to `qmv_fast_crossrow_affine4_g64<T, 2>`, a separate function that the
# transplant does not touch, so the probe's NA = 2 cell is not a shipped cell
# and its weight carries a zero effect. That correction is a source fact read
# before this session ran, not a fit to its data.
PREDICTED_KERNEL_PCT = 1.463
PREDICTED_LEG_PCT = -PREDICTED_KERNEL_PCT * E116_KERNEL_TO_LEG
PREDICTION_BAND_PCT = (-1.36, -0.44)

# Rule 59 promotion bar on the ranked frame.
PROMOTION_BAR_RANKED_PCT = 0.20

# Rule 35 repeatability of this local harness.
REPEATABILITY_MTP_PCT = 0.33
REPEATABILITY_SERIAL_PCT = 0.18

T95_TWO_SIDED = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


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


def pooled(values: list[float]) -> dict:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--label", default="r3")
    ap.add_argument("--exactness",
                    default="research/e121-artifacts/row-digest-512.json")
    ap.add_argument("--out", default="research/e121-artifacts/rung3-e2e.json")
    args = ap.parse_args(argv)

    order = ["base", "share", "share", "base"]
    legs, missing = [], []
    for rep in range(1, args.replicates + 1):
        for position, arm in enumerate(order, start=1):
            tag = f"e121{args.label}k{rep}p{position}{arm}"
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
                "gpu_temp_entry_c": as_float(meta.get("gpu_temp_entry_c")),
                "gpu_temp_exit_c": as_float(meta.get("gpu_temp_exit_c")),
                "worker_sha256": meta.get("worker_sha256_pre"),
                "measured_commit_unwound": meta.get("measured_commit_unwound"),
            })
    if missing:
        print(f"e121_e2e_analyse: missing {len(missing)} leg(s): {missing}")
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
                           ("local_ratio", "ratio"),
                           ("effective_mean_draft_len", "draftlen"),
                           ("accepted_draft_rate", "acceptrate")):
            base, cand = arm_mean(rep, "base", key), arm_mean(rep, "share", key)
            row[f"{short}_base"] = base
            row[f"{short}_share"] = cand
            row[f"{short}_pct"] = pct(cand, base)
        base_legs = sorted((leg for leg in legs
                            if leg["replicate"] == rep and leg["tree"] == "base"),
                           key=lambda leg: leg["position"])
        row["base_pair_drift_pct"] = pct(base_legs[-1]["seconds_per_token"],
                                         base_legs[0]["seconds_per_token"])
        per_replicate.append(row)

    mtp = pooled([row["mtp_spt_pct"] for row in per_replicate])
    serial = pooled([row["serial_spt_pct"] for row in per_replicate])
    ratio = pooled([row["ratio_pct"] for row in per_replicate])
    drift = pooled([row["base_pair_drift_pct"] for row in per_replicate])
    headline = mtp["mean_pct"]

    # The schedule must not move. This arm changes only where a chunk sum is
    # computed, so a changed draft length or acceptance rate would mean the
    # measurement compared two different workloads.
    draftlen = pooled([row["draftlen_pct"] for row in per_replicate])
    accept = pooled([row["acceptrate_pct"] for row in per_replicate])
    schedule_invariant = (abs(draftlen["mean_pct"]) < 0.05
                          and abs(accept["mean_pct"]) < 0.05)

    ranked = headline * RANKED_TRANSFER
    ranked_ci = (
        mtp["ci95_lower_pct"] * RANKED_TRANSFER
        if mtp["ci95_lower_pct"] is not None else None,
        mtp["ci95_upper_pct"] * RANKED_TRANSFER
        if mtp["ci95_upper_pct"] is not None else None,
    )

    entry = [leg["gpu_temp_entry_c"] for leg in legs
             if leg["gpu_temp_entry_c"] is not None]
    exit_c = [leg["gpu_temp_exit_c"] for leg in legs
              if leg["gpu_temp_exit_c"] is not None]

    exact_path = pathlib.Path(args.exactness)
    exact_doc = json.loads(exact_path.read_text()) if exact_path.is_file() else {}
    exactness = [{
        "check": f"row_digest_{rec['tag']}",
        "rows": rec.get("rows"),
        "expected": exact_doc.get("pinned_sha256"),
        "observed": rec.get("sha256"),
        "passed": bool(rec.get("matches_pin") and rec.get("row_count_ok")),
    } for rec in exact_doc.get("legs", [])]
    for control in ("value_control", "order_control", "runtime_control"):
        rec = exact_doc.get(control)
        if rec:
            exactness.append({
                "check": control,
                "rows": None,
                "expected": "digest must move",
                "observed": rec.get("kind") or rec.get("tag"),
                "passed": bool(rec.get("digest_moved", rec.get("differs_from_pin"))),
            })
    exactness.append({
        "check": "abba_leg_token_match",
        "rows": len(legs),
        "expected": "all_tokens_matched=True on every timed leg",
        "observed": str(sum(1 for leg in legs if leg["all_tokens_matched"])),
        "passed": all(leg["all_tokens_matched"] for leg in legs),
    })
    exact_ok = bool(exactness) and all(rec["passed"] for rec in exactness)

    prediction_hit = (PREDICTION_BAND_PCT[0] <= headline <= PREDICTION_BAND_PCT[1])

    doc = {
        "arm": "share",
        "experiment": "e121-rung3-insitu",
        "harness": "local",
        "candidate_commit": read_meta(legs[0]["tag"]).get("branch_commit"),
        "base_commit": next((leg["measured_commit_unwound"] for leg in legs
                             if leg["tree"] == "base"), None),
        "worker_fingerprint": next((leg["worker_sha256"] for leg in legs
                                    if leg["tree"] == "share"), None),
        "token_window": args.tokens,
        "order": "base, share, share, base per replicate",
        "replicates": len(replicates),
        "reproduction":
            f"research/e121_exact512.sh && "
            f"research/e121_e2e_abba.sh {len(replicates)} {args.tokens} "
            f"{args.label} && research/e121_e2e_analyse.py "
            f"--replicates {len(replicates)} --tokens {args.tokens}",
        "legs": legs,
        "per_replicate": per_replicate,
        "exactness": exactness,
        "prediction": {
            "registered_before_session": True,
            "source": "rung-2 cost-weighted shipped frame at NA = 4",
            "kernel_pct": PREDICTED_KERNEL_PCT,
            "e116_kernel_to_leg": E116_KERNEL_TO_LEG,
            "predicted_leg_pct": PREDICTED_LEG_PCT,
            "band_pct": list(PREDICTION_BAND_PCT),
            "measured_leg_pct": headline,
            "hit": prediction_hit,
        },
        "summary": {
            "primary_metric": "candidate_mtp_seconds_per_token_pct_vs_base",
            "sign_convention": "negative percent means the candidate is faster",
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
            "mtp_spt_share_mean_s": statistics.fmean(
                leg["seconds_per_token"] for leg in legs if leg["tree"] == "share"),
            "serial_spt_pct_mean": serial["mean_pct"],
            "serial_spt_base_mean_s": statistics.fmean(
                leg["serial_seconds_per_token"] for leg in legs
                if leg["tree"] == "base"),
            "local_ratio_pct_mean": ratio["mean_pct"],
            "local_ratio_base_mean": statistics.fmean(
                leg["local_ratio"] for leg in legs if leg["tree"] == "base"),
            "local_ratio_share_mean": statistics.fmean(
                leg["local_ratio"] for leg in legs if leg["tree"] == "share"),
            "base_pair_drift_pct_mean": drift["mean_pct"],
            "base_pair_drift_pct_max_abs": max(abs(row["base_pair_drift_pct"])
                                               for row in per_replicate),
            "draftlen_pct_mean": draftlen["mean_pct"],
            "acceptrate_pct_mean": accept["mean_pct"],
            "schedule_invariant": schedule_invariant,
            "ranked_frame_pct": ranked,
            "ranked_frame_ci95_lower": ranked_ci[0],
            "ranked_frame_ci95_upper": ranked_ci[1],
            "ranked_transfer": RANKED_TRANSFER,
            "promotion_bar_ranked_pct": PROMOTION_BAR_RANKED_PCT,
            "clears_promotion_bar": bool(
                -ranked >= PROMOTION_BAR_RANKED_PCT
                and ranked_ci[1] is not None
                and -ranked_ci[1] > 0.0),
            "repeatability_mtp_pct": REPEATABILITY_MTP_PCT,
            "repeatability_serial_pct": REPEATABILITY_SERIAL_PCT,
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

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"legs {len(legs)} replicates {len(replicates)}")
    for row in per_replicate:
        print(f"  k{row['replicate']}: mtp {row['mtp_spt_pct']:+.3f} % "
              f"serial {row['serial_spt_pct']:+.3f} % "
              f"ratio {row['ratio_pct']:+.3f} % "
              f"base-pair drift {row['base_pair_drift_pct']:+.3f} %")
    print(f"POOLED absolute candidate MTP s/token {headline:+.3f} % "
          f"(sd {mtp['stdev_pct']:.3f}, n {mtp['n_replicates']})")
    if mtp["t95_half_width_pct"] is not None:
        print(f"  95 % CI [{mtp['ci95_lower_pct']:+.3f}, "
              f"{mtp['ci95_upper_pct']:+.3f}] %")
    print(f"  base {doc['summary']['mtp_spt_base_mean_s']:.6f} s/token -> "
          f"share {doc['summary']['mtp_spt_share_mean_s']:.6f} s/token")
    print(f"  local ratio {doc['summary']['local_ratio_base_mean']:.5f} -> "
          f"{doc['summary']['local_ratio_share_mean']:.5f} "
          f"({ratio['mean_pct']:+.3f} %)")
    print(f"  schedule invariant {schedule_invariant} "
          f"(draftlen {draftlen['mean_pct']:+.4f} %, "
          f"accept {accept['mean_pct']:+.4f} %)")
    print(f"  ranked frame {ranked:+.3f} % "
          f"(bar {PROMOTION_BAR_RANKED_PCT:+.2f} % faster) -> "
          + ("CLEARS" if doc["summary"]["clears_promotion_bar"] else "does NOT clear"))
    print(f"  prediction {PREDICTED_LEG_PCT:+.3f} % band "
          f"[{PREDICTION_BAND_PCT[0]:+.2f}, {PREDICTION_BAND_PCT[1]:+.2f}] -> "
          + ("HIT" if prediction_hit else "MISS"))
    print(f"  exactness {'PASS' if exact_ok else 'FAIL'}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
