#!/usr/bin/env python3
"""Paired per-round estimator for one E87 rung-2 session.

usage: research/e87_paired.py PREFIX [--base declared] [--out FILE]

Leg totals cannot resolve this experiment. The arm-G session's own null, taken
across its first and last base leg, is -0.408 %, while the effect under test is
a few tenths of a percent. The paired estimator removes that drift because
every leg replays the SAME fixture and produces the same round sequence, so
round `r` of an arm leg and round `r` of a base leg do the same work.

Two things this reports that a leg total cannot:

  * the per-draft cost of the draft chain, `d_submit2_us / d`, which is where
    the coarse shortlist read actually lives. Its arm-versus-base difference is
    the IN-SESSION cost of the readout change, measured in the repo build
    inside a real round, not in a standalone bench.
  * the per-leg host-state stratum. E89 rung 0a showed the local host state is
    an arm-blind global CPU multiplier on the eight host phases, so rounds are
    gated at an absolute 1500 us on their host-phase sum before any pooled
    number is taken.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

# E89 rung 0a: eight host phases, arm-blind, absolute gate at 1500 us.
HOST_PHASES = (
    "d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
    "d_chain_us", "readout_us", "commit_us", "upkeep_us",
)
HOST_GATE_US = 1500.0

FIELD = re.compile(r"(\w+)=([-\d.]+)")

# Head bytes each arm reads per draft, from the build reports. The declared
# head is 427,738,112 B; g128 removes 15,733,760 B of coarse traffic; arm C
# replaces the 157,337,600 B dense coarse read with a 19,667,200 B centroid
# pass plus a 39,334,400 B probed-row pass. Reporting achieved bandwidth beside
# every timed stage keeps a byte model honest about the fixed cost it omits.
HEAD_BYTES = {
    "declared": 427_738_112,
    "dense": 427_738_112,
    "g128": 412_004_352,
    "armc": 329_402_112,
    "armc-damaged": 329_402_112,
}

# Option B derives the index instead of shipping it, so its per-draft read is a
# function of the probe fraction the leg ran with. 12,292 leaves of 8 rows at
# 1,600 B per coarse row: every leg pays the centroid pass, and a probed leaf
# costs 8 rows.
DERIVED_LEAVES = 12_292
DERIVED_ROWS_PER_LEAF = 8
COARSE_ROW_BYTES = 1_600
DENSE_COARSE_BYTES = 157_337_600


def derived_head_bytes(probe_fraction: float) -> int:
    probes = max(1, math.ceil(probe_fraction * DERIVED_LEAVES))
    stage = (DERIVED_LEAVES + probes * DERIVED_ROWS_PER_LEAF) * COARSE_ROW_BYTES
    return HEAD_BYTES["declared"] - DENSE_COARSE_BYTES + stage


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        rec = {k: float(v) for k, v in FIELD.findall(line.split(" arm=")[0])}
        rec["host_us"] = sum(rec[p] for p in HOST_PHASES)
        rec["clean"] = rec["host_us"] < HOST_GATE_US
        if rec["d"] > 0:
            rec["submit2_per_draft_us"] = rec["d_submit2_us"] / rec["d"]
            rec["build_per_draft_us"] = rec["draft_build_us"] / rec["d"]
        out.append(rec)
    return out


def meta_value(tag: str, key: str) -> str | None:
    for line in (OUT / tag / "meta.txt").read_text().splitlines():
        if line.startswith(key + "="):
            return line.partition("=")[2]
    return None


def arm_of(tag: str) -> str:
    value = meta_value(tag, "e87_arm")
    if value is None:
        raise SystemExit(f"{tag}: no e87_arm in meta.txt")
    return value


def head_bytes_of(tag: str, arm: str) -> int | None:
    if arm != "derived":
        return HEAD_BYTES.get(arm)
    raw = meta_value(tag, "e87_probe_fraction")
    if raw is None:
        raise SystemExit(f"{tag}: derived leg with no e87_probe_fraction")
    return derived_head_bytes(float(raw))


def paired(base: list[list[dict]], arm: list[list[dict]], key: str) -> dict | None:
    """Median over rounds of the arm-minus-base difference at that round.

    Each side is first collapsed with a median over its own legs, so one hot
    leg cannot move the estimate.
    """
    n = min(min(len(r) for r in base), min(len(r) for r in arm))
    deltas, ratios = [], []
    for i in range(n):
        b = [legs[i] for legs in base if key in legs[i] and legs[i]["clean"]]
        a = [legs[i] for legs in arm if key in legs[i] and legs[i]["clean"]]
        if not b or not a:
            continue
        bm, am = st.median(r[key] for r in b), st.median(r[key] for r in a)
        deltas.append(am - bm)
        if bm > 0:
            ratios.append(am / bm - 1.0)
    if not deltas:
        return None
    return {
        "rounds_paired": len(deltas),
        "median_delta_us": st.median(deltas),
        "median_pct": st.median(ratios) * 100.0,
        "mean_pct": st.mean(ratios) * 100.0,
        "pct_stdev": st.stdev(ratios) * 100.0 if len(ratios) > 1 else 0.0,
        "sign_test_arm_faster": sum(1 for d in deltas if d < 0),
        "sign_test_n": len(deltas),
        "base_median_us": st.median(
            st.median(legs[i][key] for legs in base if key in legs[i])
            for i in range(n) if all(key in legs[i] for legs in base)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--base", default="declared")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir() if p.name.startswith(args.prefix + "-"))
    if not tags:
        raise SystemExit(f"no legs under {OUT} with prefix {args.prefix}-")

    legs, stratum, arm_bytes = {}, [], {}
    for tag in tags:
        rs = rounds(tag)
        legs.setdefault(arm_of(tag), []).append(rs)
        arm_bytes[arm_of(tag)] = head_bytes_of(tag, arm_of(tag))
        clean = [r["host_us"] for r in rs if r["clean"]]
        dirty = [r["host_us"] for r in rs if not r["clean"]]
        stratum.append({
            "tag": tag,
            "arm": arm_of(tag),
            "rounds": len(rs),
            "clean_rounds": len(clean),
            "dirty_rounds": len(dirty),
            "clean_median_host_us": st.median(clean) if clean else None,
            "dirty_median_host_us": st.median(dirty) if dirty else None,
            "max_host_us": max(r["host_us"] for r in rs),
            "drafts": sum(r["d"] for r in rs),
            "accepted": sum(r["acc"] for r in rs),
        })

    if args.base not in legs:
        raise SystemExit(f"session has no {args.base} legs")

    bandwidth = {}
    for arm, group in legs.items():
        per_draft = [r["submit2_per_draft_us"] for rs in group for r in rs
                     if r["clean"] and "submit2_per_draft_us" in r]
        head_bytes = arm_bytes.get(arm)
        if not per_draft or head_bytes is None:
            continue
        med = st.median(per_draft)
        bandwidth[arm] = {
            "head_bytes_per_draft": head_bytes,
            "submit2_per_draft_median_us": med,
            "clean_drafting_rounds": len(per_draft),
            "achieved_bandwidth_gbs": head_bytes / (med * 1e-6) / 1e9,
        }

    # The fixture is fixed, so an arm that proposes the same tokens must
    # reproduce the same depth sequence. A divergence is the real acceptance
    # effect and it invalidates a strict round-for-round pairing.
    depth = {arm: [tuple(r["d"] for r in rs) for rs in group]
             for arm, group in legs.items()}
    identical = len({seq for seqs in depth.values() for seq in seqs}) == 1

    report = {
        "prefix": args.prefix,
        "harness": "local",
        "estimator": "paired per-round median, E89 host gate",
        "host_gate_us": HOST_GATE_US,
        "host_phases": list(HOST_PHASES),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "depth_sequence_identical_across_arms": identical,
        "per_leg_host_stratum": stratum,
        "achieved_bandwidth": bandwidth,
        "paired": {},
    }

    print(f"host-state stratum (absolute {HOST_GATE_US:.0f} us gate on "
          f"{len(HOST_PHASES)} host phases):")
    print(f"{'leg':<26} {'arm':<10} {'rnds':>5} {'clean':>6} {'dirty':>6} "
          f"{'clean med':>10} {'max':>9}")
    for s in stratum:
        dm = "-" if s["dirty_median_host_us"] is None else f"{s['dirty_median_host_us']:.0f}"
        print(f"{s['tag']:<26} {s['arm']:<10} {s['rounds']:>5} {s['clean_rounds']:>6} "
              f"{s['dirty_rounds']:>6} {s['clean_median_host_us']:>10.0f} "
              f"{s['max_host_us']:>9.0f}   dirty med {dm}")
    print(f"\ndepth sequence identical across arms: {identical}")

    print("\nachieved bandwidth of the per-draft head read (clean rounds):")
    print(f"  {'arm':<14} {'bytes/draft':>13} {'median us':>10} {'GB/s':>8}")
    for arm, b in sorted(bandwidth.items()):
        print(f"  {arm:<14} {b['head_bytes_per_draft']:>13,} "
              f"{b['submit2_per_draft_median_us']:>10.1f} "
              f"{b['achieved_bandwidth_gbs']:>8.1f}")

    keys = ("round_us", "draft_build_us", "d_submit2_us",
            "submit2_per_draft_us", "build_per_draft_us", "verify_build_us")
    for arm, group in legs.items():
        if arm == args.base:
            continue
        report["paired"][arm] = {}
        print(f"\npaired {arm} vs {args.base}:")
        print(f"  {'metric':<24} {'base med us':>12} {'delta us':>10} "
              f"{'delta %':>9} {'faster/n':>10}")
        for key in keys:
            r = paired(legs[args.base], group, key)
            if r is None:
                continue
            report["paired"][arm][key] = r
            print(f"  {key:<24} {r['base_median_us']:>12.1f} "
                  f"{r['median_delta_us']:>+10.1f} {r['median_pct']:>+9.3f} "
                  f"{r['sign_test_arm_faster']:>4}/{r['sign_test_n']:<5}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
