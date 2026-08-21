#!/usr/bin/env python3
"""E86: separate arm effect from session drift and from single-leg anomalies.

usage: research/e86_drift.py PREFIX [--ref default]

A palindromic session cancels MONOTONE drift when the two repeats of each arm
are averaged. It does not cancel a single anomalous leg, and it does not make
`ref_leg1 - ref_leg2` a noise estimate: at the two ends of the palindrome that
difference measures the full drift span, so using it as the session null
rejects real effects.

This script therefore reports three things per leg and per arm:

  1. the per-leg round-loop total and the out-of-loop remainder
     (`spt * 512 - round_total`), which separates a decode-loop effect from
     prefill, warmup and harness time;
  2. the paired per-round median against EACH reference leg separately, so a
     disagreement between them exposes an anomalous reference;
  3. a least-squares fit of round time on arm and leg position, which uses the
     palindrome to estimate the drift slope instead of assuming it away.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")
PHASES = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
          "d_submit2_us", "verify_build_us", "eval_wall_us", "readout_us",
          "commit_us", "upkeep_us", "round_us")


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1)), "d": int(m.group(2)), "acc": int(m.group(3))}
        rec.update({k: float(v) for k, v in KV_RE.findall(m.group(4))})
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ref", default="default")
    args = ap.parse_args()

    doc = json.loads((ROOT / "research" / f"e86-{args.prefix}.json").read_text()) \
        if (ROOT / "research" / f"e86-{args.prefix}.json").exists() else None

    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(args.prefix + "-") and (p / "trace.txt").exists())
    legs = []
    for t in tags:
        meta = dict(line.partition("=")[::2]
                    for line in (OUT / t / "meta.txt").read_text().splitlines() if "=" in line)
        score = json.loads((OUT / t / "score.json").read_text())["metrics"]
        rs = rounds(t)
        legs.append({
            "tag": t, "arm": t[len(args.prefix) + 1:].rpartition("-")[0],
            "started": meta["started"], "rounds": rs,
            "spt": score["mtp_seconds_per_token"],
            "decode_s": score["mtp_seconds_per_token"] * score["decode_tokens"],
            "round_total_s": sum(r["round_us"] for r in rs) / 1e6,
            "temp_in": float(meta["gpu_temp_entry_c"]),
        })
    legs.sort(key=lambda r: r["started"])
    for i, l in enumerate(legs):
        l["pos"] = i
        l["outside_s"] = l["decode_s"] - l["round_total_s"]

    print(f"{'pos':>3} {'leg':<24}{'spt':>10}{'decode s':>10}{'round tot':>11}"
          f"{'outside':>9}{'Tin':>6}")
    for l in legs:
        print(f"{l['pos']:>3} {l['tag']:<24}{l['spt']:>10.6f}{l['decode_s']:>10.4f}"
              f"{l['round_total_s']:>11.4f}{l['outside_s']:>9.4f}{l['temp_in']:>6.1f}")

    out = [l["outside_s"] for l in legs]
    print(f"\nout-of-loop remainder: median={st.median(out):.4f}s "
          f"min={min(out):.4f} max={max(out):.4f} spread={max(out) - min(out):.4f}s")
    print("  a leg whose spt moves but whose round_total does not is an "
          "out-of-loop anomaly, not a ladder effect.")

    refs = [l for l in legs if l["arm"] == args.ref]
    print(f"\npaired per-round median Δ round_us against EACH `{args.ref}` leg")
    print(f"{'arm':<12}" + "".join(f"{'vs ' + r['tag'][-1]:>14}" for r in refs)
          + f"{'disagreement':>14}")
    for arm in sorted({l["arm"] for l in legs}):
        cells = []
        for r in refs:
            d = []
            for la in [l for l in legs if l["arm"] == arm]:
                d += [a["round_us"] - b["round_us"]
                      for a, b in zip(la["rounds"], r["rounds"])]
            cells.append(st.median(d))
        print(f"{arm:<12}" + "".join(f"{c:>+14.0f}" for c in cells)
              + f"{max(cells) - min(cells):>14.0f}")

    # Least squares on arm + linear position, per-round paired to round index.
    arms = sorted({l["arm"] for l in legs})
    idx = {a: i for i, a in enumerate(arms)}
    n = len(legs[0]["rounds"])
    rows, ys = [], []
    for l in legs:
        for k in range(n):
            row = [0.0] * (len(arms) + n + 1)
            row[idx[l["arm"]]] = 1.0
            row[len(arms) + k] = 1.0
            row[-1] = l["pos"] - (len(legs) - 1) / 2.0
            rows.append(row)
            ys.append(l["rounds"][k]["round_us"])
    try:
        import numpy as np
        A = np.array(rows)
        y = np.array(ys)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        base = beta[idx[args.ref]]
        drift = beta[-1]
        print(f"\nleast squares, arm effect with a linear drift term removed")
        print(f"  drift = {drift:+.0f} us per leg position "
              f"({drift * (len(legs) - 1):+.0f} us across the session)")
        print(f"{'arm':<12}{'Δ vs ref us':>14}{'% of round':>12}")
        med_ref = st.median([r["round_us"] for l in refs for r in l["rounds"]])
        for a in arms:
            if a == args.ref:
                continue
            d = beta[idx[a]] - base
            print(f"{a:<12}{d:>+14.0f}{d / med_ref * 100:>+12.3f}")
    except ImportError:
        print("\n(numpy unavailable; skipped the least-squares drift fit)")


if __name__ == "__main__":
    main()
