#!/usr/bin/env python3
"""Reduce one or more E111 rung-1 sessions to the pre-registered decision.

    usage: research/e111_analyze.py TAG [TAG ...] [--json OUT]

Each session runs seven arms in a palindrome inside every block, so a paired
per-block delta against `a_shipped` cancels monotone drift to first order. The
reported statistic is that paired delta, not the difference of two means.

Arm stream widths are bytes read per 64-value group:

    a_shipped  36  32 B of packed 4-bit weight, bf16 scale, bf16 bias
    n_nobias   34  bias load and its accumulation both deleted
    n_nosums   36  bias load kept, the sum accumulation deleted
    d_bias1    35  1-byte code replaces the bias, no reconstruction
    e_bias6    35  1-byte code replaces the bias, exact reconstruction
    b_constw    4  the 32 B weight load deleted, all arithmetic kept
    c_loadonly 36  every load kept, extract and fma deleted
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
    "e_bias6": 35, "b_constw": 4, "c_loadonly": 36,
}
ARMS = list(GROUP_BYTES)

# Pre-registered on PR #113, comment e111-prereg-r1.
KILL1_PCT = 1.2      # a_shipped - n_nobias below this kills the mechanism
KILL2_PCT = 0.60     # a_shipped - d_bias1 below this kills the mechanism
ADVANCE_PCT = 0.46   # a_shipped - e_bias6 at or above this advances it
QMV_SHARE_OF_ROUND = 0.877   # QMV pool as a fraction of the M=5 round
ROUND_BAR_PCT = 0.20         # advisor promotion bar, percent of the round


def load(tag: str) -> tuple[dict, dict[int, dict[str, float]]]:
    doc = json.loads((OUT / tag / "arms.json").read_text())
    blocks: dict[int, dict[str, float]] = collections.defaultdict(dict)
    for row in doc["timing"]:
        blocks[row["block"]][row["arm"]] = row["gpu_us"]
    return doc, blocks


def paired(blocks: dict[int, dict[str, float]], arm: str) -> list[float]:
    """Per-block percentage delta of `arm` against `a_shipped`."""
    return [100.0 * (b[arm] - b["a_shipped"]) / b["a_shipped"]
            for b in blocks.values() if arm in b]


def reduce_session(tag: str) -> dict:
    doc, blocks = load(tag)
    stream_bytes = doc["groups"] * GROUP_BYTES["a_shipped"]
    rows = []
    for arm in ARMS:
        us = [b[arm] for b in blocks.values()]
        rel = paired(blocks, arm)
        mean_us = st.mean(us)
        rows.append({
            "arm": arm,
            "group_bytes": GROUP_BYTES[arm],
            "traffic_pct_of_shipped":
                100.0 * GROUP_BYTES[arm] / GROUP_BYTES["a_shipped"],
            "us_mean": mean_us,
            "us_sd": st.stdev(us) if len(us) > 1 else 0.0,
            "blocks": len(us),
            # Bytes this arm actually reads, at this arm's own time.
            "gbps": doc["groups"] * GROUP_BYTES[arm] / (mean_us * 1e-6) / 1e9,
            "paired_pct": st.mean(rel),
            "paired_sd": st.stdev(rel) if len(rel) > 1 else 0.0,
            "saving_pct": -st.mean(rel),
        })
    by = {r["arm"]: r for r in rows}

    # b_constw deletes 32 of the 36 bytes per group with every instruction
    # kept, so it prices a byte directly. Anything the byte price cannot buy
    # is not available to a metadata-traffic change.
    traffic_removed = 100.0 * (1 - GROUP_BYTES["b_constw"] / 36)
    byte_price = by["b_constw"]["saving_pct"] / traffic_removed
    onebyte_traffic = 100.0 / 36

    return {
        "tag": tag,
        "shape": doc["shape"],
        "k": doc["k"], "n": doc["n"], "na": doc["na"],
        "groups": doc["groups"],
        "shipped_stream_bytes": stream_bytes,
        "shipped_gbps": by["a_shipped"]["gbps"],
        "blocks": doc["blocks"], "inner": doc["inner"], "reps": doc["reps"],
        "warm_passes": doc.get("warm_passes"),
        "arms": rows,
        "fidelity": doc["fidelity"],
        "weight_control": doc["weight_control"],
        "code_control": doc["code_control"],
        "arm_exactness": doc["arm_exactness"],
        "byte_price_pct_per_pct_traffic": byte_price,
        "predicted_onebyte_saving_pct": byte_price * onebyte_traffic,
        "kill1_fires": by["n_nobias"]["saving_pct"] < KILL1_PCT,
        "kill2_fires": by["d_bias1"]["saving_pct"] < KILL2_PCT,
        "advance_fires": by["e_bias6"]["saving_pct"] >= ADVANCE_PCT,
        "e_bias6_round_pct": by["e_bias6"]["saving_pct"] * QMV_SHARE_OF_ROUND,
        "clears_round_bar":
            by["e_bias6"]["saving_pct"] * QMV_SHARE_OF_ROUND >= ROUND_BAR_PCT,
    }


def report(sessions: list[dict]) -> None:
    for s in sessions:
        print(f"\n== {s['tag']}  {s['shape']}  "
              f"K={s['k']} N={s['n']} M={s['na']} "
              f"groups={s['groups']:,}  stream={s['shipped_stream_bytes']:,} B"
              f"  {s['shipped_gbps']:.1f} GB/s")
        head = (f"{'arm':<12}{'B/grp':>6}{'us':>9}{'sd':>7}"
                f"{'paired%':>9}{'sd':>7}{'saving%':>9}{'GB/s':>8}")
        print(head)
        print("-" * len(head))
        for r in s["arms"]:
            print(f"{r['arm']:<12}{r['group_bytes']:>6}{r['us_mean']:>9.1f}"
                  f"{r['us_sd']:>7.1f}{r['paired_pct']:>9.2f}"
                  f"{r['paired_sd']:>7.2f}{r['saving_pct']:>9.2f}"
                  f"{r['gbps']:>8.1f}")
        print(f"  byte price      : {s['byte_price_pct_per_pct_traffic']:.4f} "
              f"% time per % traffic (from b_constw)")
        print(f"  predicted 1 B/36: "
              f"{s['predicted_onebyte_saving_pct']:.4f} % of the kernel")
        print(f"  KILL1 n_nobias  < {KILL1_PCT} %  -> "
              f"{'FIRES' if s['kill1_fires'] else 'does not fire'}")
        print(f"  KILL2 d_bias1   < {KILL2_PCT} %  -> "
              f"{'FIRES' if s['kill2_fires'] else 'does not fire'}")
        print(f"  ADVANCE e_bias6 >= {ADVANCE_PCT} % -> "
              f"{'FIRES' if s['advance_fires'] else 'does not fire'}")
        print(f"  e_bias6 round   : {s['e_bias6_round_pct']:+.4f} % "
              f"(bar {ROUND_BAR_PCT} %)")
        f, w, c = s["fidelity"], s["weight_control"], s["code_control"]
        print(f"  fidelity worst_rel {f['worst_rel']:.3e} tol {f['tolerance']}"
              f" pass={f['pass']}")
        print(f"  weight control perturbed {w['perturbed_rel']:.3e} restored "
              f"{w['restored_rel']:.3e} detected={w['detected']}")
        print(f"  code control damaged {c['damaged_differing']} restored "
              f"{c['restored_differing']} detected={c['detected']}")
        for e in s["arm_exactness"]:
            if e["expect_bit_exact"]:
                print(f"  bit-exact {e['arm']}: {e['differing']}/{e['total']} "
                      f"differing, pass={e['pass']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    sessions = [reduce_session(t) for t in args.tags]
    report(sessions)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(sessions, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
