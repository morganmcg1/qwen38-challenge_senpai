#!/usr/bin/env python3
"""E89 rung 0a: is the stuck state a multiplier or an added constant?

The decode ladder replays an identical round sequence in every leg, so a stuck
leg and a clean leg of the same arm and session are pairable round by round.
That makes the shape of the stuck penalty measurable without any GPU time.

  multiplicative  stuck = k * clean      -> the host thread runs slower
  additive        stuck = clean + c      -> the host thread does extra work

The two models predict different residuals against the clean round profile,
and the clean profile is not flat: it carries periodic snapshot and cache
rounds that stand well above its own floor.

usage: research/e89_rung0a_model.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")
HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
        "readout_us", "commit_us", "upkeep_us"]
PIPE = ["draft_build_us", "d_submit2_us", "verify_build_us", "eval_wall_us",
        "round_us"]

PAIRS = [
    ("e86r2-default-4", ["e86r2-default-1", "e86r2-default-2", "e86r2-default-3"]),
    ("e86r1-default-2", ["e86r1-default-1"]),
]


def load(tag: str) -> dict[int, dict]:
    rs = {}
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {k: float(v) for k, v in KV_RE.findall(m.group(4))}
        rec["HOSTSUM"] = sum(rec[k] for k in HOST)
        rec["d"] = int(m.group(2))
        rec["acc"] = int(m.group(3))
        rs[int(m.group(1))] = rec
    return rs


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den


def main() -> None:
    payload = {"experiment": "e89-drafting-round-host-state", "rung": "0a-model",
               "harness": "local", "gpu_used": False, "pairs": []}

    for stuck_tag, clean_tags in PAIRS:
        stuck = load(stuck_tag)
        cleans = [load(t) for t in clean_tags]
        print(f"\n{'=' * 78}\nstuck {stuck_tag}   clean {', '.join(clean_tags)}")

        seq_ok = all(c[r]["d"] == stuck[r]["d"] and c[r]["acc"] == stuck[r]["acc"]
                     for c in cleans for r in stuck)
        print(f"identical round sequence: {seq_ok}")

        # e86r1-default-1 is only clean from round 34, and one round of
        # e86r2-default-4 carries a 51 ms scheduling outlier.
        lo = 40 if stuck_tag == "e86r1-default-2" else 6
        rounds = [r for r in sorted(stuck) if r >= lo
                  and stuck[r]["HOSTSUM"] < 20000]
        print(f"paired rounds used: {len(rounds)} (from round {lo})")

        print(f"\n{'phase':<16}{'clean':>9}{'stuck':>9}{'ratio':>8}{'add_us':>9}"
              f"{'resid_mult':>12}{'resid_add':>11}{'better':>8}")
        rows = {}
        for ph in HOST + ["HOSTSUM"]:
            cl = [st.median([c[r][ph] for c in cleans]) for r in rounds]
            sk = [stuck[r][ph] for r in rounds]
            cm, sm = st.median(cl), st.median(sk)
            k = sm / cm if cm else float("nan")
            add = sm - cm
            rm = st.pstdev([s - k * c for s, c in zip(sk, cl)])
            ra = st.pstdev([s - (c + add) for s, c in zip(sk, cl)])
            better = "mult" if rm < ra else "add"
            rows[ph] = {"clean_med_us": cm, "stuck_med_us": sm, "ratio": k,
                        "add_us": add, "resid_mult": rm, "resid_add": ra,
                        "better": better}
            print(f"{ph:<16}{cm:>9.0f}{sm:>9.0f}{k:>8.2f}{add:>9.0f}"
                  f"{rm:>12.0f}{ra:>11.0f}{better:>8}")

        cl = [st.median([c[r]["HOSTSUM"] for c in cleans]) for r in rounds]
        sk = [stuck[r]["HOSTSUM"] for r in rounds]
        r = pearson(cl, sk)
        mx = st.mean(cl)
        slope = (sum((a - mx) * (b - st.mean(sk)) for a, b in zip(cl, sk))
                 / sum((a - mx) ** 2 for a in cl))
        icpt = st.mean(sk) - slope * mx
        print(f"\nHOSTSUM clean-profile vs stuck-profile pearson r = {r:.3f}")
        print(f"OLS  stuck = {slope:.2f} * clean {icpt:+.0f} us")
        print("  a slope well above 1 with a small intercept means the stuck leg "
              "scales the whole\n  clean host profile, which is the signature of a "
              "slower thread, not of added work.")

        print("\npipeline phases, which are GPU bound and should not move:")
        pipe = {}
        for ph in PIPE:
            cm = st.median([st.median([c[r][ph] for c in cleans]) for r in rounds])
            sm = st.median([stuck[r][ph] for r in rounds])
            pipe[ph] = {"clean_med_us": cm, "stuck_med_us": sm, "ratio": sm / cm}
            print(f"  {ph:<18}clean {cm:>9.0f}  stuck {sm:>9.0f}  "
                  f"ratio {sm / cm:>6.3f}  diff {sm - cm:>+8.0f} us")

        payload["pairs"].append({
            "stuck": stuck_tag, "clean": clean_tags, "n_rounds": len(rounds),
            "identical_round_sequence": seq_ok, "host": rows, "pipe": pipe,
            "hostsum_pearson_r": r, "hostsum_ols_slope": slope,
            "hostsum_ols_intercept_us": icpt,
        })

    # The mid-leg transition in e86r1-default-1.
    leg = load("e86r1-default-1")
    series = [leg[r]["HOSTSUM"] for r in sorted(leg)]
    before = st.median(series[5:33])
    after = st.median(series[34:])
    print(f"\n{'=' * 78}\nmid-leg transition, e86r1-default-1")
    print(f"  rounds 6-33  median host sum {before:.0f} us")
    print(f"  rounds 35-78 median host sum {after:.0f} us")
    print(f"  step ratio {before / after:.2f}x at round 34")
    payload["mid_leg_transition"] = {
        "leg": "e86r1-default-1", "transition_round": 34,
        "before_med_us": before, "after_med_us": after,
        "step_ratio": before / after,
    }

    Path(ROOT / "research/e89-rung0a-model.json").write_text(
        json.dumps(payload, indent=1) + "\n")
    print("\nwrote research/e89-rung0a-model.json")


if __name__ == "__main__":
    main()
