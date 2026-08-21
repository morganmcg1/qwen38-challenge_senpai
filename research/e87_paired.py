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


def score_metric(tag: str, key: str):
    path = OUT / tag / "score.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("metrics", {}).get(key)


def arm_of(tag: str) -> str:
    value = meta_value(tag, "e87_arm")
    if value is None:
        raise SystemExit(f"{tag}: no e87_arm in meta.txt")
    return value


def head_bytes_of(tag: str, arm: str) -> int | None:
    if not arm.startswith("derived"):
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
            "gpu_temp_entry_c": meta_value(tag, "gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta_value(tag, "gpu_temp_exit_c"),
            "sandbox": meta_value(tag, "sandbox"),
            "leg_index": int(meta_value(tag, "e87_leg_index") or -1),
            "mtp_seconds_per_token": score_metric(tag, "mtp_seconds_per_token"),
            "accepted_draft_rate": score_metric(tag, "accepted_draft_rate"),
            "effective_mean_draft_len": score_metric(tag, "effective_mean_draft_len"),
            "all_tokens_matched": score_metric(tag, "all_tokens_matched"),
            "head_provenance_sha256": score_metric(tag, "head_provenance_sha256"),
            "round1_us": rs[0]["round_us"] if rs else None,
            "round2_us": rs[1]["round_us"] if len(rs) > 1 else None,
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

    # A counterbalanced order cancels monotone drift only if each arm occupies
    # the same mean position. Publish the sums so that is checkable rather than
    # asserted, and take the session null from the base arm's own first and
    # last leg.
    by_arm_positions = {}
    for s in stratum:
        by_arm_positions.setdefault(s["arm"], []).append(s["leg_index"])
    position = {arm: {"positions": sorted(p), "position_sum": sum(p),
                      "mean_position": st.mean(p)}
                for arm, p in by_arm_positions.items()}

    base_legs = sorted((s for s in stratum if s["arm"] == args.base),
                       key=lambda s: s["leg_index"])
    null = None
    if len(base_legs) > 1 and base_legs[0]["mtp_seconds_per_token"]:
        first, last = base_legs[0], base_legs[-1]
        null = {
            "first_leg": first["tag"], "last_leg": last["tag"],
            "first_seconds_per_token": first["mtp_seconds_per_token"],
            "last_seconds_per_token": last["mtp_seconds_per_token"],
            "session_null_pct": (last["mtp_seconds_per_token"]
                                 / first["mtp_seconds_per_token"] - 1.0) * 100.0,
        }

    # The advisor reads this before any pooled number: a round only enters the
    # paired estimator when it is clean in the base arm AND in the arm.
    both_clean = {}
    for arm, group in legs.items():
        if arm == args.base:
            continue
        n = min(min(len(r) for r in legs[args.base]), min(len(r) for r in group))
        both = sum(1 for i in range(n)
                   if any(l[i]["clean"] for l in legs[args.base])
                   and any(l[i]["clean"] for l in group))
        both_clean[arm] = {"rounds_compared": n, "clean_in_both_arms": both}

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
        "sandbox": sorted({s["sandbox"] for s in stratum if s["sandbox"]}),
        "abba_position": position,
        "session_null": null,
        "clean_in_both_arms": both_clean,
        "per_leg_host_stratum": stratum,
        "achieved_bandwidth": bandwidth,
        "paired": {},
    }

    print(f"host-state stratum (absolute {HOST_GATE_US:.0f} us gate on "
          f"{len(HOST_PHASES)} host phases):")
    print(f"{'leg':<30} {'arm':<11} {'rnds':>5} {'clean':>6} {'dirty':>6} "
          f"{'clean med':>10} {'max':>9} {'entC':>6} {'exitC':>6} "
          f"{'rnd1':>8} {'rnd2':>8}")
    for s in stratum:
        dm = "-" if s["dirty_median_host_us"] is None else f"{s['dirty_median_host_us']:.0f}"
        print(f"{s['tag']:<30} {s['arm']:<11} {s['rounds']:>5} {s['clean_rounds']:>6} "
              f"{s['dirty_rounds']:>6} {s['clean_median_host_us']:>10.0f} "
              f"{s['max_host_us']:>9.0f} {s['gpu_temp_entry_c'] or '-':>6} "
              f"{s['gpu_temp_exit_c'] or '-':>6} {s['round1_us']:>8.0f} "
              f"{s['round2_us']:>8.0f}   dirty med {dm}")
    print(f"\ndepth sequence identical across arms: {identical}")
    print(f"sandbox: {','.join(report['sandbox']) or 'unrecorded'}")

    print("\nabsolute mtp_seconds_per_token per leg:")
    for s in sorted(stratum, key=lambda s: s["leg_index"]):
        spt = s["mtp_seconds_per_token"]
        print(f"  pos {s['leg_index']:>2} {s['arm']:<11} "
              f"{spt if spt is None else f'{spt:.6f}':>10}  "
              f"entry {s['gpu_temp_entry_c'] or '-':>8} exit {s['gpu_temp_exit_c'] or '-':>8}")

    print("\nABBA position sums (equal sums mean linear drift cancels):")
    for arm, p in sorted(position.items()):
        print(f"  {arm:<11} positions {p['positions']} sum {p['position_sum']} "
              f"mean {p['mean_position']:.2f}")
    if null:
        print(f"\nsession null ({args.base} first vs last leg): "
              f"{null['session_null_pct']:+.3f} %  "
              f"({null['first_seconds_per_token']:.6f} -> "
              f"{null['last_seconds_per_token']:.6f})")
    for arm, c in sorted(both_clean.items()):
        print(f"rounds clean in BOTH {args.base} and {arm}: "
              f"{c['clean_in_both_arms']}/{c['rounds_compared']}")

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
