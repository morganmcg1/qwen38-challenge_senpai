#!/usr/bin/env python3
"""Reduce E111 rung-1 sessions to the pre-registered, round-weighted decision.

    usage: research/e111_analyze.py TAG [TAG ...] [--json OUT]

Each session runs every arm in a palindrome inside every block, so a paired
per-block delta against `a_shipped` cancels monotone drift to first order. The
reported statistic is that paired delta, not the difference of two means.

Advisor feedback f1 replaced the single-width price with a width-weighted one.
The realised verify-width histogram maps each round width onto the shipped
one-group partition, and the resulting NA weights are the share of streaming
time each NA carries. A number measured only at NA=5 prices 3.4 % of the round.

Arm stream widths are bytes read per 64-value group:

    a_shipped  36  32 B of packed 4-bit weight, bf16 scale, bf16 bias
    n_nobias   34  bias load and its accumulation both deleted
    n_nosums   36  bias load kept, the sum accumulation deleted
    d_bias1    35  1-byte code replaces the bias, no reconstruction
    e_bias6    35  1-byte code replaces the bias, exact reconstruction
    b_constw    4  the 32 B weight load deleted, all arithmetic kept
    c_loadonly 36  every load kept, extract and fma deleted
    g_pack32   36  shipped values from one interleaved 32-bit record
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics as st

OUT = pathlib.Path("research/out")

GROUP_BYTES = {
    "a_shipped": 36, "n_nobias": 34, "n_nosums": 36, "d_bias1": 35,
    "e_bias6": 35, "b_constw": 4, "c_loadonly": 36, "g_pack32": 36,
}
ARMS = list(GROUP_BYTES)

# Advisor f1: share of streaming time carried by each one-group width, from
# Edward's realised width histogram, W&B 19kgn6xi.
NA_WEIGHT = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# Pre-registered on PR #113, comment e111-prereg-r1. Advisor f1 applies the
# kill rules to the round-weighted number rather than to a single width.
KILL1_PCT = 1.2
KILL2_PCT = 0.60
ADVANCE_PCT = 0.46
QMV_SHARE_OF_ROUND = 0.877
ROUND_BAR_PCT = 0.20
DRAM_PEAK_GBPS = 273.0


def load(tag: str) -> tuple[dict, dict[int, dict[str, float]]]:
    doc = json.loads((OUT / tag / "arms.json").read_text())
    blocks: dict[int, dict[str, float]] = collections.defaultdict(dict)
    for row in doc["timing"]:
        blocks[row["block"]][row["arm"]] = row["gpu_us"]
    return doc, blocks


def reduce_session(tag: str) -> dict:
    doc, blocks = load(tag)
    present = [a for a in ARMS if any(a in b for b in blocks.values())]
    rows = []
    for arm in present:
        us = [b[arm] for b in blocks.values()]
        rel = [100.0 * (b[arm] - b["a_shipped"]) / b["a_shipped"]
               for b in blocks.values()]
        mean_us = st.mean(us)
        rows.append({
            "arm": arm,
            "group_bytes": GROUP_BYTES[arm],
            "us_mean": mean_us,
            "us_sd": st.stdev(us) if len(us) > 1 else 0.0,
            "blocks": len(us),
            "gbps": doc["groups"] * GROUP_BYTES[arm] / (mean_us * 1e-6) / 1e9,
            "paired_pct": st.mean(rel),
            "paired_sd": st.stdev(rel) if len(rel) > 1 else 0.0,
            "saving_pct": -st.mean(rel),
        })
    by = {r["arm"]: r for r in rows}

    # b_constw deletes 32 of the 36 bytes per group with every instruction
    # kept, so it prices a byte directly. Whatever the byte price cannot buy
    # is not available to any metadata-traffic change.
    byte_price = by["b_constw"]["saving_pct"] / (100.0 * (1 - 4 / 36))

    return {
        "tag": tag,
        "shape": doc["shape"],
        "k": doc["k"], "n": doc["n"], "na": doc["na"],
        "groups": doc["groups"],
        "shipped_stream_bytes": doc["groups"] * 36,
        "shipped_gbps": by["a_shipped"]["gbps"],
        "shipped_pct_of_peak":
            100.0 * by["a_shipped"]["gbps"] / DRAM_PEAK_GBPS,
        "blocks": doc["blocks"], "inner": doc["inner"], "reps": doc["reps"],
        "warm_passes": doc.get("warm_passes"),
        "arms": rows,
        "fidelity": doc["fidelity"],
        "weight_control": doc["weight_control"],
        "code_control": doc["code_control"],
        "pack_control": doc.get("pack_control"),
        "arm_exactness": doc["arm_exactness"],
        "byte_price_pct_per_pct_traffic": byte_price,
        "predicted_onebyte_saving_pct": byte_price * 100.0 / 36,
    }


def weighted(sessions: list[dict], arm: str) -> tuple[float, float]:
    """Round-weighted saving over NA, and the NA weight actually covered."""
    total = num = 0.0
    for s in sessions:
        w = NA_WEIGHT.get(s["na"], 0.0)
        hit = next((r for r in s["arms"] if r["arm"] == arm), None)
        if hit is None:
            continue
        num += w * hit["saving_pct"]
        total += w
    if total == 0:
        return float("nan"), 0.0
    return num / total, total


def report(sessions: list[dict]) -> dict:
    for s in sessions:
        print(f"\n== {s['tag']}  {s['shape']}  NA={s['na']}  "
              f"K={s['k']} N={s['n']} groups={s['groups']:,}  "
              f"{s['shipped_gbps']:.1f} GB/s "
              f"({s['shipped_pct_of_peak']:.1f} % of peak)")
        head = (f"{'arm':<12}{'B/grp':>6}{'us':>9}{'sd':>7}"
                f"{'paired%':>9}{'sd':>7}{'saving%':>9}{'GB/s':>8}")
        print(head)
        print("-" * len(head))
        for r in s["arms"]:
            print(f"{r['arm']:<12}{r['group_bytes']:>6}{r['us_mean']:>9.1f}"
                  f"{r['us_sd']:>7.1f}{r['paired_pct']:>9.2f}"
                  f"{r['paired_sd']:>7.2f}{r['saving_pct']:>9.2f}"
                  f"{r['gbps']:>8.1f}")
        print(f"  byte price {s['byte_price_pct_per_pct_traffic']:.4f} "
              f"% time per % traffic; a 1 B/36 cut predicts "
              f"{s['predicted_onebyte_saving_pct']:.4f} % of the kernel")

    print(f"\n== round-weighted over NA, weights {NA_WEIGHT}")
    print(f"   NA measured: {sorted({s['na'] for s in sessions})}")
    out: dict[str, float] = {}
    print(f"{'arm':<12}{'weighted saving %':>20}{'round %':>10}")
    coverage = 0.0
    for arm in ARMS:
        val, coverage = weighted(sessions, arm)
        if val != val:
            continue
        out[arm] = val
        print(f"{arm:<12}{val:>20.3f}{val * QMV_SHARE_OF_ROUND:>10.3f}")
    if abs(coverage - sum(NA_WEIGHT.values())) > 1e-9:
        print(f"   WARNING: sessions cover {coverage:.3f} of "
              f"{sum(NA_WEIGHT.values()):.3f} NA weight; renormalised")

    k1 = out["n_nobias"] < KILL1_PCT
    k2 = out["d_bias1"] < KILL2_PCT
    adv = out["e_bias6"] >= ADVANCE_PCT
    print(f"\n  KILL1 n_nobias  < {KILL1_PCT} %  -> "
          f"{'FIRES' if k1 else 'does not fire'}  "
          f"(measured {out['n_nobias']:+.3f} %)")
    print(f"  KILL2 d_bias1   < {KILL2_PCT} %  -> "
          f"{'FIRES' if k2 else 'does not fire'}  "
          f"(measured {out['d_bias1']:+.3f} %)")
    print(f"  ADVANCE e_bias6 >= {ADVANCE_PCT} % -> "
          f"{'FIRES' if adv else 'does not fire'}  "
          f"(measured {out['e_bias6']:+.3f} %)")
    print(f"  e_bias6 round   : {out['e_bias6'] * QMV_SHARE_OF_ROUND:+.4f} % "
          f"against a {ROUND_BAR_PCT} % bar")
    return {"weighted": out, "kill1": k1, "kill2": k2, "advance": adv,
            "na_measured": sorted({s["na"] for s in sessions}),
            "na_weight_covered": coverage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    sessions = [reduce_session(t) for t in args.tags]
    verdict = report(sessions)
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"sessions": sessions, "verdict": verdict,
                        "na_weights": NA_WEIGHT}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
