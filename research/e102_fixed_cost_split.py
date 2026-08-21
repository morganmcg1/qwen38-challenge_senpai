#!/usr/bin/env python3
"""E102 rung 0a: is a ranked per-prompt delta a per-round cost or a one-time cost?

Every ranked leg decodes the same 512 tokens, but the eight prompts take very
different wall times, from about 5.7 s on botany to about 15.5 s on plutarch.
So a FIXED one-time cost, such as a cold Metal pipeline compilation that one
tree pays inside the timed leg and its sibling has already warmed, appears as a
LARGER percentage on the fast, high-width prompts and a smaller percentage on
the slow, low-width ones. That is the same sign and the same ordering as a
genuine per-row cost increase, and the published percentage alone cannot tell
them apart.

Two models are fitted to the same eight per-prompt deltas:

    fixed         delta_seconds[p] = c            (one constant, milliseconds)
    proportional  delta_percent[p] = q            (one constant, percent)

Both have one free parameter, so their residual RMSE in percent space is
directly comparable. The model that wins names the mechanism class.

The E77 flat occupancy tax is subtracted first when a register delta is given,
because that tax IS proportional and belongs to the kernel.

Usage:
    python3 research/e102_fixed_cost_split.py ca9251b8 B --regs 98
    python3 research/e102_fixed_cost_split.py 3ff80e86 A --regs 120
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e102_wide_row_pricing as wrp  # noqa: E402

T = wrp.T
GAMMA = 0.01346
REGISTER_FILE_BYTES = 496 * 1024
BYTES_PER_REGISTER_PER_SIMD = 128


def occupancy_tax(regs: int, base_regs: int = 91) -> float:
    """E77 law, harness=ranked model. Returns a percent, positive = slower."""
    def omega(r):
        s = REGISTER_FILE_BYTES // (BYTES_PER_REGISTER_PER_SIMD * r)
        return (32.0 / s) ** GAMMA
    return 100.0 * (omega(regs) / omega(base_regs) - 1.0)


def fit(rows, tax):
    """rows: [(prompt, M, ctrl_spt, tgt_spt)]. Returns both model fits."""
    obs = []
    for name, m, ctrl, tgt in rows:
        pct = 100.0 * (tgt - ctrl) / ctrl - tax
        leg = T * ctrl                      # control leg seconds
        obs.append((name, m, leg, pct, pct / 100.0 * leg * 1000.0))

    c = st.fmean(o[4] for o in obs)                     # ms
    q = st.fmean(o[3] for o in obs)                     # percent
    rmse_fixed = math.sqrt(st.fmean(
        (o[3] - 100.0 * (c / 1000.0) / o[2]) ** 2 for o in obs))
    rmse_prop = math.sqrt(st.fmean((o[3] - q) ** 2 for o in obs))
    return obs, c, q, rmse_fixed, rmse_prop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("tier")
    ap.add_argument("--regs", type=int, default=91,
                    help="candidate g17s entry-point registers, E102 rung 1")
    args = ap.parse_args()

    scored = wrp.scored_rows()
    fp = wrp.fingerprints()
    row = wrp.pick(scored, args.target)
    tiers = dict(wrp.control_set(scored, row, fp))
    if args.tier not in tiers:
        raise SystemExit(f"tier {args.tier} absent; have {sorted(tiers)}")
    ctrls = tiers[args.tier]

    def spt(r, name):
        return r["_t"][name]["mtp_seconds_per_token_mean"]

    rows = []
    for name, entry in row["_t"].items():
        m = 1.0 + entry["effective_mean_draft_len"]
        rows.append((name, m, st.fmean(spt(r, name) for r in ctrls),
                     spt(row, name)))
    rows.sort(key=lambda r: -r[1])

    tax = occupancy_tax(args.regs)
    obs, c, q, rmse_fixed, rmse_prop = fit(rows, tax)

    print(f"target {args.target}  tier {args.tier}  n_ctrl {len(ctrls)}  "
          f"g17s R={args.regs}")
    print(f"E77 flat occupancy tax subtracted from every prompt: {tax:+.4f} %\n")
    print(f"{'prompt':<10}{'M':>7}{'leg s':>9}{'raw %':>9}{'net %':>9}"
          f"{'implied ms':>12}{'G':>4}")
    for name, m, leg, pct, ms in obs:
        raw = pct + tax
        print(f"{name:<10}{m:>7.3f}{leg:>9.3f}{raw:>+9.4f}{pct:>+9.4f}"
              f"{ms:>+12.1f}{2 if m >= 5 else 1:>4}")

    high = [o for o in obs if o[1] >= 5]
    low = [o for o in obs if o[1] < 5]
    print(f"\nnet high-width mean {st.fmean(o[3] for o in high):+.4f} %  "
          f"(n={len(high)})")
    print(f"net low-width  mean {st.fmean(o[3] for o in low):+.4f} %  "
          f"(n={len(low)})")
    print(f"raw low-width  mean {st.fmean(o[3] + tax for o in low):+.4f} %  "
          f"vs E77 prediction {tax:+.4f} %  "
          f"residual {st.fmean(o[3] for o in low):+.4f} pp")

    print(f"\nfixed one-time model   c = {c:+.1f} ms/leg "
          f"(sd {st.stdev(o[4] for o in obs):.1f})   RMSE {rmse_fixed:.4f} %")
    print(f"proportional model     q = {q:+.4f} %          "
          f"          RMSE {rmse_prop:.4f} %")
    better = "FIXED one-time" if rmse_fixed < rmse_prop else "PROPORTIONAL"
    print(f"better fit: {better}  "
          f"(ratio {max(rmse_fixed, rmse_prop) / min(rmse_fixed, rmse_prop):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
