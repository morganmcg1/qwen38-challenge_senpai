#!/usr/bin/env python3
"""Why the E109 rung-0 protocol stopped at 0.470 % instead of 0.20 %.

The reducer already shows that the per-leg offset carries almost all of the
pair variance. This script asks what that offset actually is, because the
answer decides which redesign is worth building:

  A  the offset is thermal        -> regress entry temperature out
  B  the offset is per-process    -> compare arms inside one leg instead
  C  the offset is unexplained    -> only more blocks help, at 1/sqrt(n)

Read-only over the finished session. Research-only, nothing on the scored path.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import random
import re
import statistics as st

STRATUM = 7  # dominant effective draft length; 56 of 77 rounds


def load(root: pathlib.Path) -> dict[str, dict]:
    legs = {}
    for path in sorted(glob.glob(str(root / "b*-*" / "report.json"))):
        tag = pathlib.Path(path).parent.name
        rep = json.loads(pathlib.Path(path).read_text())
        meta = (pathlib.Path(path).parent / "meta.txt").read_text()

        def field(name: str) -> float | None:
            hit = re.search(rf"^{name}=([0-9.]+)", meta, re.M)
            return float(hit.group(1)) if hit else None

        def text(name: str) -> str:
            hit = re.search(rf"^{name}=(.*)$", meta, re.M)
            return hit.group(1).strip() if hit else ""

        legs[tag] = {
            "t": [x * 1e6 for x in rep["block_request_seconds"]],
            "k": rep["effective_draft_lengths"],
            "entry": field("gpu_temp_entry_c"),
            "exit": field("gpu_temp_exit_c"),
            "started": text("started_utc"),
            "wall": field("leg_wall_seconds"),
            "order": int(field("leg_index") or 0),
            "decode_s": rep.get("decode_seconds"),
            "seed_s": rep.get("seed_prefill_seconds"),
            "spt": rep.get("parent_measured_seconds_per_token"),
            "block": int(tag.split("-")[0][1:]),
            "arm": tag.split("-", 1)[1],
        }
    return legs


def regress(xs: list[float], ys: list[float]) -> tuple[float, float, list[float]]:
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    return slope, r, resid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--bar-pct", type=float, default=0.20)
    ap.add_argument("--leg-seconds", type=float, default=92.3)
    ap.add_argument("--control-us", type=float, default=177431.6)
    args = ap.parse_args()

    legs = load(pathlib.Path(args.session))
    print(f"E109 leg-offset diagnosis -- {args.session}   legs {len(legs)}")

    # Rule 34. Name the frame in absolute microseconds and account for it.
    any_leg = next(iter(legs.values()))
    rounds = len(any_leg["t"])
    tokens_per_round = 512 / rounds
    round_us = st.mean([st.mean(g["t"]) for g in legs.values()])
    covered = sum(any_leg["t"]) / 1e6
    print(f"\n0  frame   rounds/leg {rounds}   tokens/round {tokens_per_round:.3f}"
          f"   mean verify width {st.mean(any_leg['k']) + 1:.3f}")
    print(f"     control round             {round_us:11,.0f} us")
    print(f"     per emitted token         {round_us / tokens_per_round:11,.0f} us")
    print(f"     block_request total       {covered:11.3f} s"
          f"   decode {any_leg['decode_s']:.3f} s"
          f"   seed {any_leg['seed_s']:.3f} s")

    # Hypothesis B. Within one leg, how well can two conditions be separated if
    # they alternate between rounds? Only same-draft-length rounds are
    # comparable, so this is the consecutive difference inside that stratum.
    consec, strat_sd, n_pairs = [], [], []
    for leg in legs.values():
        idx = [i for i, k in enumerate(leg["k"]) if k == STRATUM]
        vals = [leg["t"][i] for i in idx]
        consec.append(st.pstdev([b - a for a, b in zip(vals, vals[1:])]))
        strat_sd.append(st.pstdev(vals))
        n_pairs.append(len(idx) // 2)
    pair_sd, pairs = st.mean(consec), int(st.mean(n_pairs))
    within_sem = pair_sd / math.sqrt(pairs)
    print(f"\nB  within-leg alternation, draft={STRATUM} stratum")
    print(f"     round SD in stratum        {st.mean(strat_sd):8.0f} us")
    print(f"     consecutive-pair diff SD   {pair_sd:8.0f} us  over {pairs} pairs")
    print(f"     one-leg SEM                {within_sem:8.0f} us")

    # Hypothesis A. Is the leg's own level predicted by its entry temperature?
    # The arm effects must come out first: d16 is a real +3,093 us shift and
    # would otherwise be charged to the offset and to temperature.
    have = [g for g in legs.values() if g["entry"] is not None]
    level = {id(g): st.mean([g["t"][i] for i, k in enumerate(g["k"])
                             if k == STRATUM]) for g in have}
    arm_mean = {}
    for arm in {g["arm"] for g in have}:
        arm_mean[arm] = st.mean([level[id(g)] for g in have if g["arm"] == arm])
    xs = [g["entry"] for g in have]
    ys = [level[id(g)] - arm_mean[g["arm"]] for g in have]
    slope, r, resid = regress(xs, ys)
    print(f"\nA  leg offset vs entry temperature   n {len(have)}"
          f"  entry {min(xs):.2f}..{max(xs):.2f} C  (arm means removed)")
    print(f"     leg offset SD              {st.pstdev(ys):8.0f} us")
    print(f"     slope {slope:8.0f} us/C   r {r:+.3f}   r2 {r * r:.3f}")
    print(f"     residual SD after removal  {st.pstdev(resid):8.0f} us")

    # Hypothesis D. Is the leg offset EXCHANGEABLE or is it DRIFT? The two
    # answers point at different redesigns, so the question is worth a
    # zero-GPU test on data that already exists.
    #
    #   exchangeable -> the offset is drawn fresh per leg and is independent of
    #                   when the leg ran. Leg count is then the only lever, and
    #                   shorter legs buy leg count at the same wall clock.
    #   drift        -> the offset is a function of session time. ABBA already
    #                   removes the linear part inside a block, and shorter
    #                   legs shrink the residual curvature as well.
    #
    # Two statistics decide it: the regression of the offset on chronological
    # leg order, and the lag-1 autocorrelation of the offset series in that
    # order. Order comes from `started_utc`, not from the block index, so a
    # retried or reordered leg cannot corrupt it.
    chrono = sorted(have, key=lambda g: g["started"])
    seq = [level[id(g)] - arm_mean[g["arm"]] for g in chrono]
    pos = [float(i) for i in range(len(seq))]
    o_slope, o_r, o_resid = regress(pos, seq)
    print(f"\nD  leg offset vs chronological order   n {len(seq)}")
    print(f"     slope {o_slope:8.1f} us/leg   r {o_r:+.3f}   r2 {o_r * o_r:.3f}"
          f"   total swing {o_slope * (len(seq) - 1):+8.0f} us")
    print(f"     residual SD after order removal {st.pstdev(o_resid):8.0f} us"
          f"   (raw {st.pstdev(seq):.0f} us)")

    def acf1(xs: list[float]) -> float:
        m = st.mean(xs)
        num = sum((a - m) * (b - m) for a, b in zip(xs, xs[1:]))
        den = sum((x - m) ** 2 for x in xs)
        return num / den if den else 0.0

    # Permutation null: if the offsets are exchangeable, every ordering of the
    # same 36 values is equally likely, so the observed lag-1 value has to be
    # judged against the distribution of lag-1 values over reorderings. A fixed
    # seed keeps this reproducible.
    rng = random.Random(20260821)
    obs = acf1(seq)
    obs_resid = acf1(o_resid)
    null = []
    shuffled = list(seq)
    for _ in range(20000):
        rng.shuffle(shuffled)
        null.append(acf1(shuffled))
    p_two = sum(1 for v in null if abs(v) >= abs(obs)) / len(null)
    null_sd = st.pstdev(null)
    print(f"     lag-1 autocorrelation {obs:+.3f}"
          f"   permutation null SD {null_sd:.3f}   two-sided p {p_two:.3f}")
    print(f"     lag-1 after order removal {obs_resid:+.3f}")
    verdict = ("DRIFT" if (p_two < 0.05 or o_r * o_r > 0.10)
               else "EXCHANGEABLE")
    print(f"     verdict  {verdict}")

    # What that verdict is worth. A leg costs a fixed setup plus a per-token
    # decode, so halving the token window does NOT halve the leg. Measure the
    # split from the legs themselves where a second token window is available;
    # otherwise report the 512-token leg cost and let the caller supply the
    # fixed part.
    walls = [g["wall"] for g in have if g["wall"]]
    decodes = [g["decode_s"] for g in have if g["decode_s"]]
    if walls and decodes:
        fixed = st.mean(walls) - st.mean(decodes)
        print(f"\n   leg cost   wall {st.mean(walls):6.1f} s"
              f"   decode {st.mean(decodes):6.1f} s"
              f"   fixed setup {fixed:6.1f} s")
        for frac in (0.5, 0.25):
            wall_h = fixed + st.mean(decodes) * frac
            gain = math.sqrt(st.mean(walls) / wall_h)
            print(f"   at {frac:.0%} tokens: leg {wall_h:5.1f} s"
                  f"   legs per hour x{st.mean(walls) / wall_h:.2f}"
                  f"   resolution x{gain:.2f} better"
                  f"   IF the offset stays per-leg")

    # How many legs each design needs to put the 95 % half-width under the bar.
    control = st.mean([st.mean(g["t"]) for g in legs.values()])
    bar = args.bar_pct / 100.0 * args.control_us
    print(f"\n   control frame {args.control_us:,.0f} us (reducer)"
          f"   unstratified mean {control:,.0f} us"
          f"   bar {args.bar_pct:.2f} % = {bar:.0f} us")
    print("\n   design                          legs   hours   half-width")
    achieved, blocks_built, arms_built = 833.5, 8, 4
    for name, sem1, legs_per, arms in (
        ("B within-leg alternation", within_sem, 1, 1),
        ("C more blocks, as built",
         achieved * math.sqrt(blocks_built) / 1.96, 1, arms_built),
    ):
        n = math.ceil((1.96 * sem1 / bar) ** 2)
        total = n * legs_per * arms
        print(f"   {name:28s} {total:6d} {total * args.leg_seconds / 3600:7.2f}"
              f"   {1.96 * sem1 / math.sqrt(n):8.0f} us"
              f"   ({n} blocks x {arms} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
