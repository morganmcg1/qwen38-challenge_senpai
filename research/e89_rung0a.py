#!/usr/bin/env python3
"""E89 rung 0a: is the E86 stuck-leg gate arm-blind?

The E86 gate is absolute: a leg is stuck when more than half of its
post-warmup rounds spend over 1500 us in the eight host phases. E86 then
reported that all four contaminated legs were `default` legs, 4 of 8 against
0 of 22.

If an arm legitimately raises the once-per-round host phases, an absolute
threshold flags that arm preferentially and the association is an artifact of
the instrument. This script measures the per-arm clean-leg host distribution,
builds arm-relative gates, and re-runs the association test under each gate.

usage: research/e89_rung0a.py [--stuck-us 1500] [--ratio 2.0]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
        "readout_us", "commit_us", "upkeep_us"]
PIPE = ["draft_build_us", "d_submit2_us", "verify_build_us", "eval_wall_us"]
SESSIONS = ["e86r0", "e86r1", "e86r2"]
WARMUP = 5


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1))}
        rec.update({k: float(v) for k, v in KV_RE.findall(m.group(4))})
        rec["HOSTSUM"] = sum(rec[k] for k in HOST)
        out.append(rec)
    return out


def iqr(xs: list[float]) -> tuple[float, float]:
    s = sorted(xs)
    q = st.quantiles(s, n=4, method="inclusive")
    return q[0], q[2]


def load_legs() -> list[dict]:
    legs = []
    for session in SESSIONS:
        for p in sorted(OUT.glob(session + "-*")):
            if not (p / "trace.txt").exists():
                continue
            tag = p.name
            arm = tag[len(session) + 1:].rpartition("-")[0]
            rs = rounds(tag)
            post = rs[WARMUP:]
            h = [r["HOSTSUM"] for r in post]
            lo, hi = iqr(h)
            legs.append({
                "tag": tag, "session": session, "arm": arm,
                "rounds": len(rs), "host_med": st.median(h),
                "host_q1": lo, "host_q3": hi,
                "host_series": h,
                "round_med": st.median([r["round_us"] for r in post]),
                "eval_med": st.median([r["eval_wall_us"] for r in post]),
                "vbuild_med": st.median([r["verify_build_us"] for r in post]),
                **{f"{k}_med": st.median([r[k] for r in post]) for k in HOST},
            })
    return legs


def fisher_right_tail(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for the 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def hyp(k: int) -> float:
        return (math.comb(col1, k) * math.comb(n - col1, row1 - k)
                / math.comb(n, row1))

    return sum(hyp(k) for k in range(a, min(row1, col1) + 1))


def report_gate(name: str, legs: list[dict], stuck_key: str) -> None:
    focus = "default"
    a = sum(1 for l in legs if l["arm"] == focus and l[stuck_key])
    b = sum(1 for l in legs if l["arm"] == focus and not l[stuck_key])
    c = sum(1 for l in legs if l["arm"] != focus and l[stuck_key])
    d = sum(1 for l in legs if l["arm"] != focus and not l[stuck_key])
    p = fisher_right_tail(a, b, c, d)
    flagged = [l["tag"] for l in legs if l[stuck_key]]
    print(f"\n### gate: {name}")
    print(f"  stuck legs ({a + c}/{len(legs)}): {', '.join(flagged) or 'none'}")
    print(f"  default stuck {a}/{a + b}   other stuck {c}/{c + d}")
    print(f"  Fisher exact one-sided p = {p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stuck-us", type=float, default=1500.0)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--json-out", default="research/e89-rung0a.json")
    args = ap.parse_args()

    legs = load_legs()

    # Gate A: the shipped E86 absolute gate.
    for l in legs:
        frac = sum(x > args.stuck_us for x in l["host_series"]) / len(l["host_series"])
        l["frac_over_abs"] = frac
        l["stuck_abs"] = frac > 0.5

    print("## per-leg post-warmup host-phase sum (us)")
    hdr = (f"{'leg':<22}{'arm':<10}{'med':>8}{'q1':>8}{'q3':>8}"
           f"{'frac>thr':>10}{'absGate':>9}{'round_med':>11}{'eval_med':>10}")
    print(hdr)
    for l in legs:
        print(f"{l['tag']:<22}{l['arm']:<10}{l['host_med']:>8.0f}{l['host_q1']:>8.0f}"
              f"{l['host_q3']:>8.0f}{l['frac_over_abs']:>10.2f}"
              f"{'STUCK' if l['stuck_abs'] else 'clean':>9}"
              f"{l['round_med']:>11.0f}{l['eval_med']:>10.0f}")

    # Step 1: per-arm distribution restricted to legs the absolute gate calls clean.
    print("\n## step 1: clean-leg host-phase sum by arm (absolute gate defines clean)")
    print(f"{'arm':<10}{'n_clean':>8}{'med':>9}{'q1':>9}{'q3':>9}{'min':>9}{'max':>9}"
          f"{'d_submit1':>11}{'d_chain':>9}{'commit':>9}{'d_head1':>9}")
    arms = sorted({l["arm"] for l in legs})
    clean_by_arm: dict[str, list[dict]] = {}
    for arm in arms:
        cl = [l for l in legs if l["arm"] == arm and not l["stuck_abs"]]
        clean_by_arm[arm] = cl
        if not cl:
            print(f"{arm:<10}{0:>8}   (all legs flagged stuck)")
            continue
        meds = [l["host_med"] for l in cl]
        pooled = [x for l in cl for x in l["host_series"]]
        lo, hi = iqr(pooled)
        print(f"{arm:<10}{len(cl):>8}{st.median(pooled):>9.0f}{lo:>9.0f}{hi:>9.0f}"
              f"{min(meds):>9.0f}{max(meds):>9.0f}"
              f"{st.median([l['d_submit1_us_med'] for l in cl]):>11.0f}"
              f"{st.median([l['d_chain_us_med'] for l in cl]):>9.0f}"
              f"{st.median([l['commit_us_med'] for l in cl]):>9.0f}"
              f"{st.median([l['d_head1_us_med'] for l in cl]):>9.0f}")

    clean_meds = {a: st.median([l["host_med"] for l in c])
                  for a, c in clean_by_arm.items() if c}
    if clean_meds:
        lo_arm = min(clean_meds, key=clean_meds.get)
        hi_arm = max(clean_meds, key=clean_meds.get)
        print(f"\n  clean-leg median spread across arms: "
              f"{lo_arm}={clean_meds[lo_arm]:.0f} us to "
              f"{hi_arm}={clean_meds[hi_arm]:.0f} us "
              f"(ratio {clean_meds[hi_arm] / clean_meds[lo_arm]:.2f}x)")
        print(f"  absolute threshold {args.stuck_us:.0f} us sits at "
              + ", ".join(f"{a}={args.stuck_us / m:.2f}x clean"
                          for a, m in sorted(clean_meds.items())))

    # Step 2: arm-relative gates. Two anchors, because an arm x session cell
    # holds only two legs and its median is not robust when one leg is stuck.
    for l in legs:
        l["ref_arm_all"] = 0.0
        l["ref_arm_session"] = 0.0
    for arm in arms:
        same_arm = [l for l in legs if l["arm"] == arm]
        anchor_all = min(l["host_med"] for l in same_arm)
        for l in same_arm:
            l["ref_arm_all"] = anchor_all
        for session in SESSIONS:
            cell = [l for l in same_arm if l["session"] == session]
            if not cell:
                continue
            anchor = min(l["host_med"] for l in cell)
            for l in cell:
                l["ref_arm_session"] = anchor

    for l in legs:
        l["stuck_rel_arm"] = l["host_med"] > args.ratio * l["ref_arm_all"]
        l["stuck_rel_cell"] = l["host_med"] > args.ratio * l["ref_arm_session"]

    print("\n## step 2: arm-relative classification")
    print(f"{'leg':<22}{'arm':<10}{'med':>8}{'ref_arm':>9}{'x_arm':>7}"
          f"{'ref_cell':>10}{'x_cell':>8}{'abs':>7}{'relArm':>8}{'relCell':>9}")
    for l in legs:
        print(f"{l['tag']:<22}{l['arm']:<10}{l['host_med']:>8.0f}"
              f"{l['ref_arm_all']:>9.0f}{l['host_med'] / l['ref_arm_all']:>7.2f}"
              f"{l['ref_arm_session']:>10.0f}"
              f"{l['host_med'] / l['ref_arm_session']:>8.2f}"
              f"{'S' if l['stuck_abs'] else '.':>7}"
              f"{'S' if l['stuck_rel_arm'] else '.':>8}"
              f"{'S' if l['stuck_rel_cell'] else '.':>9}")

    ratios = sorted(l["host_med"] / l["ref_arm_all"] for l in legs)
    print("\n  sorted within-arm ratios: " + " ".join(f"{r:.2f}" for r in ratios))
    gaps = [(ratios[i + 1] - ratios[i], ratios[i], ratios[i + 1])
            for i in range(len(ratios) - 1)]
    g = max(gaps)
    print(f"  largest gap in the within-arm ratio ladder: "
          f"{g[1]:.2f} -> {g[2]:.2f} (width {g[0]:.2f})")

    # Step 3: the association test under each gate.
    report_gate("A absolute host sum > "
                f"{args.stuck_us:.0f} us in over half the post-warmup rounds",
                legs, "stuck_abs")
    report_gate(f"B within-arm relative, median > {args.ratio}x the fastest leg "
                "of the same arm across all sessions", legs, "stuck_rel_arm")
    report_gate(f"C within-arm-within-session relative, median > {args.ratio}x "
                "the fastest leg of the same arm in the same session",
                legs, "stuck_rel_cell")

    payload = {
        "experiment": "e89-drafting-round-host-state",
        "rung": "0a",
        "harness": "local",
        "gpu_used": False,
        "source_sessions": SESSIONS,
        "warmup_rounds": WARMUP,
        "abs_threshold_us": args.stuck_us,
        "relative_ratio": args.ratio,
        "legs": [{k: v for k, v in l.items() if k != "host_series"} for l in legs],
        "clean_median_by_arm": clean_meds,
    }
    Path(ROOT / args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
