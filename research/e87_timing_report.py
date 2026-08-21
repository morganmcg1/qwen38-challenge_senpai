#!/usr/bin/env python3
"""Summarise one E87 rung-2 timed session and test it against the price list.

usage: research/e87_timing_report.py PREFIX [--out FILE]

Reuses the E82 per-leg reader, then adds the three numbers this rung must
decide on:

  * the arm-versus-base delta in ABSOLUTE candidate seconds/token, paired
    within the session so thermal drift cancels to first order;
  * the session null, taken as the first-versus-last leg of the base arm,
    which bounds what the session can resolve; and
  * the predicted delta from the E87 price list, so a measured effect can be
    compared with the byte model that motivated the arm.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e82_headcost_report import OUT, leg  # noqa: E402

BASE_ARM = "declared"

# E87 price list, harness=local, host Mac16,11 48GB.
DECLARED_HEAD_BYTES = 427_738_112
PCT_PER_PCT_HEAD_BYTES = 0.0815  # % candidate s/token per 1% of declared head
ARM_BYTES = {
    "declared": 0,
    "dense": 0,
    "g128": -15_733_760,
    "armc": -98_336_000,
    # Option B at the default 0.15 probe fraction: the 157,337,600 B dense
    # coarse read becomes a 19,667,200 B centroid pass plus 1,844 probed leaves
    # of eight 1,600 B rows.
    "derived": -114_067_200,
    "pinned": None,
}


def predicted_pct(arm: str) -> float | None:
    delta = ARM_BYTES.get(arm)
    if delta is None:
        return None
    return delta / DECLARED_HEAD_BYTES * 100.0 * PCT_PER_PCT_HEAD_BYTES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir() if p.name.startswith(args.prefix + "-"))
    if not tags:
        raise SystemExit(f"no legs under {OUT} with prefix {args.prefix}-")
    legs = [leg(t) for t in tags]
    legs.sort(key=lambda r: r["started"])
    for r in legs:
        meta = (OUT / r["tag"] / "meta.txt").read_text().splitlines()
        for line in meta:
            if line.startswith("e87_arm="):
                r["arm"] = line.partition("=")[2]

    arms: dict[str, list[dict]] = {}
    for r in legs:
        arms.setdefault(r["arm"], []).append(r)
    if BASE_ARM not in arms:
        raise SystemExit(f"session has no {BASE_ARM} legs")

    summary = {}
    for arm, rows in arms.items():
        spt = [r["candidate_mtp_seconds_per_token"] for r in rows]
        summary[arm] = {
            "n": len(rows),
            "spt_mean": st.mean(spt),
            "spt_legs": spt,
            "spt_stdev": st.stdev(spt) if len(spt) > 1 else 0.0,
            "spt_spread_pct": (max(spt) - min(spt)) / st.mean(spt) * 100.0,
            "ratio_mean": st.mean(r["local_ratio"] for r in rows),
            "ratio_legs": [r["local_ratio"] for r in rows],
            "serial_spt_mean": st.mean(r["serial_seconds_per_token"] for r in rows),
            "rounds_legs": [r["rounds"] for r in rows],
            "rows_per_token_legs": [round(r["rows_per_token"], 4) for r in rows],
            "mean_d_legs": [round(r["mean_d"], 4) for r in rows],
            "mean_acc_legs": [round(r["mean_acc"], 4) for r in rows],
            "accepted_draft_rate_legs": [r["accepted_draft_rate"] for r in rows],
            "draft_build_us_per_round": st.mean(r["draft_build_us_per_round"] for r in rows),
            "head_loaded_bytes": rows[0]["head_loaded_bytes"],
            "head_provenance_sha256": sorted({r["head_provenance_sha256"] for r in rows}),
            "all_tokens_matched": all(r["all_tokens_matched"] for r in rows),
            "gpu_temp_entry_c": [r["gpu_temp_entry_c"] for r in rows],
            "gpu_temp_exit_c": [r["gpu_temp_exit_c"] for r in rows],
        }

    base = summary[BASE_ARM]
    base_legs = arms[BASE_ARM]
    session_null_pct = (
        (base_legs[-1]["candidate_mtp_seconds_per_token"]
         - base_legs[0]["candidate_mtp_seconds_per_token"])
        / base["spt_mean"] * 100.0
    )
    entry_spread = max(r["gpu_temp_entry_c"] for r in legs) - min(
        r["gpu_temp_entry_c"] for r in legs)

    for arm, s in summary.items():
        s["spt_delta_pct_vs_base"] = (s["spt_mean"] - base["spt_mean"]) / base["spt_mean"] * 100.0
        s["predicted_pct"] = predicted_pct(arm)
        s["ratio_delta_pct_vs_base"] = (
            (s["ratio_mean"] - base["ratio_mean"]) / base["ratio_mean"] * 100.0)

    print(f"session null (first vs last {BASE_ARM} leg): {session_null_pct:+.3f}%")
    print(f"entry GPU temperature spread across legs: {entry_spread:.1f} C")
    print(f"worker digests: {sorted({r['worker_sha256'][:12] for r in legs})}")
    print()
    print(f"{'arm':<10} {'n':>2} {'s/tok mean':>12} {'Δ% vs base':>11} {'pred Δ%':>9} "
          f"{'stdev%':>7} {'ratio':>7} {'rows/tok':>9} {'head MB':>9} {'matched':>8}")
    for arm in sorted(summary, key=lambda a: summary[a]["spt_mean"]):
        s = summary[arm]
        pred = "  n/a" if s["predicted_pct"] is None else f"{s['predicted_pct']:>+9.3f}"
        print(f"{arm:<10} {s['n']:>2} {s['spt_mean']:>12.6f} "
              f"{s['spt_delta_pct_vs_base']:>+11.3f} {pred} "
              f"{s['spt_stdev'] / s['spt_mean'] * 100.0:>7.3f} "
              f"{s['ratio_mean']:>7.4f} "
              f"{st.mean(s['rows_per_token_legs']):>9.4f} "
              f"{s['head_loaded_bytes'] / 1e6:>9.2f} {str(s['all_tokens_matched']):>8}")

    print("\nper-leg detail (chronological):")
    for r in legs:
        print(f"  {r['tag']:<26} spt={r['candidate_mtp_seconds_per_token']:.6f} "
              f"ratio={r['local_ratio']:.4f} rounds={r['rounds']} "
              f"rows/tok={r['rows_per_token']:.4f} "
              f"acc={r['accepted_draft_rate']:.4f} "
              f"dbuild={r['draft_build_us_per_round']:.0f}us "
              f"T={r['gpu_temp_entry_c']:.1f}->{r['gpu_temp_exit_c']:.1f}C "
              f"matched={r['all_tokens_matched']} "
              f"head={r['head_provenance_sha256'][:12]}/{r['head_loaded_bytes']}")

    doc = {
        "prefix": args.prefix,
        "experiment": "e87-coarse-draft-shortlist-traffic",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "host": "Apple M4 Pro 48GB (not ranked M5)",
        "base_arm": BASE_ARM,
        "session_null_pct": session_null_pct,
        "gpu_temp_entry_spread_c": entry_spread,
        "declared_head_bytes": DECLARED_HEAD_BYTES,
        "pct_per_pct_head_bytes": PCT_PER_PCT_HEAD_BYTES,
        "legs": legs,
        "summary": summary,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
