#!/usr/bin/env python3
"""E135 rung 2 and rung 3: the per-column launch cost, paired on round index.

RUNG 2. The launch grid is a host-side argument that never enters the Metal
source, so the wide and tight arms are bit-identical and round `i` of a wide
leg is THE SAME ROUND as round `i` of its paired tight leg: same width, same
draft and accept counts, same KV length, same position in the leg. Pairing on
round index therefore leaves exactly one difference in the universe,

    delta_i = round_us_wide[i] - round_us_tight[i] = a * empty_columns(M_i)

with no work term to fit at all and with KV-length growth differenced out. `a`
is the wall-clock cost of launching one threadgroup column that returns at
`qwen_e120_qmv_m:1548` before it loads a weight. One column is 519,040
threadgroups across the 257 routed dispatches of one verify round.

Rule 101 applies to a through-origin fit as much as to a witness, so the script
reports four things beside the point estimate:

  * the same fit with a FREE INTERCEPT. A large intercept with a collapsed
    slope refutes proportionality to empty columns even if the total is
    positive.
  * PER-WIDTH `a` with per-width `n`. A stable `a` is a per-threadgroup cost; a
    rising `a` is something else that scales with width. Under onepass67
    `empty` stops rising at 6 while M keeps rising, so M = 7, 8, 9 are the only
    widths where cost per empty column and cost per unit width separate.
  * a PLANTED POSITIVE CONTROL. A synthetic offset proportional to
    `empty_columns` is added to the wide arm inside the reader and the fit must
    recover it.
  * an UNROUTED NULL. Widths outside 3...9 reach `default: break` and launch no
    routed QMV, so their paired delta must be consistent with zero. This is a
    zero-cost placebo that the data supplies for free.

RUNG 3. The same session crosses the launch grid with the dispatch table, so
the four cell means give `B - A` (the promoted one-pass arm under the wide
grid, which the ranked receipt already prices at about zero) beside `D - C`
(the same arm with the launch confound removed).

  usage: python3 research/e135_per_width.py --label s2
"""

from __future__ import annotations

import argparse
import collections
import math
import re

import numpy as np

from e135_report import fnum, legs

# `(m, ipg)` for every routable width, from `Qwen35CustomQMV.Table.plan`.
PLANS = {
    "shipped": {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3},
    "onepass6": {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 3},
    "onepass67": {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3},
    "onepass678": {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3},
}

# One column of one verify round, summed over the seven routed linear shapes.
THREADGROUPS_PER_COLUMN = 519_040

# F83 ranked frame. The local benchfixture runs at mean width 7.359 with
# P(M>=6) = 0.8718 against the ranked 5.7732 and 0.5861, so a ranked number
# must never be priced from the local width histogram.
RANKED_MEAN_WIDTH = 5.7732
RANKED_MASS_6 = 0.188
RANKED_MASS_7 = 0.211
RANKED_P_GE_6 = 0.5861


def ranked_empty_columns_bounds() -> tuple[float, float]:
    """Bound the F83-weighted empty columns removed per ranked round.

    Under the shipped one-pass table `empty_columns(m)` is exactly `m - 1` for
    `3 <= m <= 7` and saturates at 6 for m = 8 and m = 9, so

        E[empty] = E[M] - 1 - mass(2) - mass(8) - 2*mass(9)

    holds exactly. F83 pins `E[M]`, `mass(6)`, `mass(7)` and `P(M >= 6)`, which
    fixes `mass(8) + mass(9)` but not its split, and leaves `mass(2)` free.
    Both unknowns can only lower the result, so dropping `mass(2)` and putting
    all the `M >= 8` mass at 8 gives a hard upper bound.

    The published ranked distribution does not resolve the 8-versus-9 split, so
    this returns an interval rather than pretending to a point estimate.
    """
    mass_ge_8 = RANKED_P_GE_6 - RANKED_MASS_6 - RANKED_MASS_7
    upper = (RANKED_MEAN_WIDTH - 1) - mass_ge_8
    lower = (RANKED_MEAN_WIDTH - 1) - 2 * mass_ge_8
    return lower, upper


RANKED_EMPTY_LOW, RANKED_EMPTY_HIGH = ranked_empty_columns_bounds()
RANKED_EMPTY_COLUMNS_REMOVED = (RANKED_EMPTY_LOW + RANKED_EMPTY_HIGH) / 2
RANKED_ROUNDS = {
    # prompt: (round_us post-arm from F167.2, rounds, F83 weight)
    "beagle": (52_453.2, 110, 0.4862),
    "medicine": (57_931.4, 90, 0.2508),
    "essays": (57_383.1, 92, 0.1598),
    "botany": (63_791.8, 81, 0.0124),
    "republic": (56_184.3, 93, 0.0100),
    "plutarch": (31_795.0, 487, 0.0000),
    "drama": (38_842.8, 252, 0.0000),
    "travel": (40_250.5, 212, 0.0000),
}

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) ")
FIELD_RE = re.compile(r" (\w+)=(-?[\d.]+)")


def empty_columns(m: int, table: str) -> int:
    """Columns the wide grid launches that return before any weight load."""
    plan = PLANS[table]
    if m not in plan:
        return 0  # unrouted width, `default: break`, no QMV dispatch at all
    ipg = plan[m]
    return m - -(-m // ipg)


def rounds(path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        head = ROUND_RE.match(line)
        if not head:
            continue
        fields = dict(FIELD_RE.findall(line))
        out.append({
            "round": int(head.group(1)),
            "d": int(head.group(2)),
            "m": int(head.group(2)) + 1,
            "acc": int(head.group(3)),
            "round_us": float(fields["round_us"]),
            "eval_wall_us": float(fields.get("eval_wall_us", "nan")),
            "verify_build_us": float(fields.get("verify_build_us", "nan")),
        })
    return out


def wls_through_origin(x: np.ndarray, y: np.ndarray):
    """a = sum(xy)/sum(xx), with the residual standard error of the slope."""
    sxx = float((x * x).sum())
    a = float((x * y).sum()) / sxx
    resid = y - a * x
    dof = max(len(y) - 1, 1)
    se = math.sqrt(float((resid**2).sum()) / dof / sxx)
    return a, se, resid


def ols(design: np.ndarray, y: np.ndarray):
    """Least squares with the standard error of every coefficient."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = float((resid**2).sum()) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return beta, np.sqrt(np.diag(cov)), cov, dof


def build_pairs(rows: list[dict]) -> list[dict]:
    """Adjacent legs of one replicate, one wide and one tight, same table."""
    by_rep = collections.defaultdict(dict)
    for r in rows:
        by_rep[r["rep"]][r["pos"]] = r
    pairs = []
    for rep in sorted(by_rep):
        slots = by_rep[rep]
        for lo in (1, 3, 5, 7):
            a, b = slots.get(lo), slots.get(lo + 1)
            if a is None or b is None:
                continue
            if a["table"] != b["table"] or a["arm"] == b["arm"]:
                continue
            wide, tight = (a, b) if a["arm"] == "wide" else (b, a)
            pairs.append({
                "rep": rep, "slot": lo, "table": a["table"],
                "wide": wide, "tight": tight,
                "order": "wide-first" if a["arm"] == "wide" else "tight-first",
            })
    return pairs


def paired_rounds(pair: dict, drop_first: int = 1):
    """Round-index-aligned deltas, or a hard failure if the arms diverged."""
    w = rounds(pair["wide"]["dir"] / "trace.txt")
    t = rounds(pair["tight"]["dir"] / "trace.txt")
    n = min(len(w), len(t))
    if n == 0:
        return [], "no traced rounds"
    mism = [i for i in range(n)
            if (w[i]["d"], w[i]["acc"]) != (t[i]["d"], t[i]["acc"])]
    if mism:
        return [], (f"round {mism[0] + 1} differs: wide d={w[mism[0]]['d']} "
                    f"acc={w[mism[0]]['acc']} against tight "
                    f"d={t[mism[0]]['d']} acc={t[mism[0]]['acc']}")
    note = "" if len(w) == len(t) else f"leg lengths differ {len(w)}/{len(t)}"
    obs = []
    for i in range(drop_first, n):
        obs.append({
            "i": i + 1, "m": w[i]["m"], "table": pair["table"],
            "x": float(empty_columns(w[i]["m"], pair["table"])),
            "delta": w[i]["round_us"] - t[i]["round_us"],
            "wide_us": w[i]["round_us"], "tight_us": t[i]["round_us"],
            "pair": f"r{pair['rep']}s{pair['slot']}{pair['table'][:2]}",
        })
    return obs, note


def report_fit(obs: list[dict], title: str) -> tuple[float, float]:
    x = np.array([o["x"] for o in obs])
    y = np.array([o["delta"] for o in obs])
    routed = x > 0
    a, se, _ = wls_through_origin(x[routed], y[routed])
    beta, bse, _, dof = ols(
        np.column_stack([np.ones(int(routed.sum())), x[routed]]), y[routed])
    print(f"\n{title}  n = {int(routed.sum())} routed paired rounds")
    print(f"  through the origin   a = {a:9.2f} +- {se:6.2f} us per launched "
          f"empty column   ({a / se:5.1f} sigma)")
    print(f"  free intercept       a = {beta[1]:9.2f} +- {bse[1]:6.2f}, "
          f"c = {beta[0]:9.2f} +- {bse[0]:6.2f} us   dof {dof}")
    print(f"  per empty threadgroup: {a * 1000 / THREADGROUPS_PER_COLUMN:.4f} "
          f"ns  ({THREADGROUPS_PER_COLUMN:,} threadgroups in one column)")
    return a, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s2")
    ap.add_argument("--plant", type=float, default=400.0,
                    help="us per empty column planted into the wide arm")
    args = ap.parse_args()

    rows = legs(args.label)
    if not rows:
        print(f"e135_per_width: no legs for label {args.label}")
        return 1

    print(f"E135 rung 2 and 3, session {args.label}: {len(rows)} legs")
    workers = sorted({r["meta"].get("worker_sha256", "?") for r in rows})
    commits = sorted({r["meta"].get("base_sha", "?") for r in rows})
    print("  worker_sha256 "
          + (f"one build {workers[0][:16]}" if len(workers) == 1 else str(workers)))
    print("  commit        "
          + (f"one commit {commits[0][:16]}" if len(commits) == 1 else str(commits)))
    print("  gate_qualified_for_timing="
          f"{sorted({r['meta'].get('gate_qualified_for_timing') for r in rows})}"
          " cool_gate_passed_real_gate="
          f"{sorted({r['meta'].get('cool_gate_passed_real_gate') for r in rows})}")

    print("\nper leg")
    print(f"{'tag':34s} {'cell':4s} {'table':10s} {'grid':5s} {'idx':>3s} "
          f"{'mtp s/tok':>10s} {'rounds':>6s} {'div':>3s} {'match':>5s} "
          f"{'inC':>5s} {'outC':>5s}")
    for r in rows:
        rr = rounds(r["dir"] / "trace.txt")
        mtp = fnum(r["metrics"].get("mtp_seconds_per_token"))
        print(f"{r['tag']:34s} {r['cell']:4s} {r['table']:10s} {r['arm']:5s} "
              f"{r['idx']:3d} {mtp if mtp is not None else float('nan'):10.6f} "
              f"{len(rr):6d} "
              f"{r['metrics'].get('residual_divergence_count', '?')!s:>3s} "
              f"{r['metrics'].get('all_tokens_matched', '?')!s:>5s} "
              f"{r['meta'].get('gpu_temp_entry_c', '?'):>5s} "
              f"{r['meta'].get('gpu_temp_exit_c', '?'):>5s}")

    # ---- round-index pairing -------------------------------------------
    pairs = build_pairs(rows)
    print(f"\nround-index pairs: {len(pairs)}")
    allobs: list[dict] = []
    for p in pairs:
        obs, note = paired_rounds(p)
        tag = f"rep {p['rep']} slot {p['slot']} {p['table']:10s} {p['order']:11s}"
        if not obs:
            print(f"  {tag} PAIRING FAILED: {note}")
            return 2
        hist = collections.Counter(o["m"] for o in obs)
        print(f"  {tag} {len(obs):4d} rounds after dropping round 1"
              f"   widths {dict(sorted(hist.items()))}"
              + (f"   NOTE {note}" if note else ""))
        allobs.extend(obs)

    if not allobs:
        print("e135_per_width: no paired rounds")
        return 2

    hist = collections.Counter(o["m"] for o in allobs)
    print("\nround-width histogram over all paired rounds: "
          f"{dict(sorted(hist.items()))}")
    print(f"  local mean width {sum(k * v for k, v in hist.items()) / len(allobs):.4f}"
          "   against the ranked F83 mean 5.7732")

    unrouted = [o for o in allobs if o["x"] == 0]
    if unrouted:
        d = np.array([o["delta"] for o in unrouted])
        sem = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"\nunrouted null (widths outside 3...9 launch no routed QMV): "
              f"n = {len(d)}, mean delta {d.mean():+.1f} +- {sem:.1f} us, "
              "which must be consistent with zero")

    a, se = report_fit(allobs, "FIT, both tables pooled")
    for table in sorted({o["table"] for o in allobs}):
        report_fit([o for o in allobs if o["table"] == table],
                   f"FIT, table {table} only")

    print("\nper-width a, with per-width n")
    print(f"  {'M':>2s} {'table':10s} {'empty':>5s} {'n':>5s} "
          f"{'mean delta us':>14s} {'sem':>7s} {'a us/col':>9s} {'sem':>7s}")
    for table in sorted({o["table"] for o in allobs}):
        for m in sorted({o["m"] for o in allobs if o["table"] == table}):
            sub = [o for o in allobs if o["m"] == m and o["table"] == table]
            d = np.array([o["delta"] for o in sub])
            x = empty_columns(m, table)
            sem = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            am = d.mean() / x if x else float("nan")
            ase = sem / x if x else float("nan")
            print(f"  {m:2d} {table:10s} {x:5d} {len(d):5d} {d.mean():14.1f} "
                  f"{sem:7.1f} {am:9.2f} {ase:7.2f}")

    print("\nper-pair a, the between-pair spread as an honest error bar")
    per_pair = []
    for name in sorted({o["pair"] for o in allobs}):
        sub = [o for o in allobs if o["pair"] == name and o["x"] > 0]
        ap_, sep_, _ = wls_through_origin(
            np.array([o["x"] for o in sub]),
            np.array([o["delta"] for o in sub]))
        per_pair.append(ap_)
        print(f"  {name:10s} n {len(sub):4d}   a {ap_:8.2f} +- {sep_:6.2f}")
    if len(per_pair) > 1:
        arr = np.array(per_pair)
        print(f"  between-pair mean {arr.mean():.2f}, sd {arr.std(ddof=1):.2f}, "
              f"sem {arr.std(ddof=1) / math.sqrt(len(arr)):.2f}")

    # ---- Rule 101 on the fit: plant an effect and recover it ------------
    x = np.array([o["x"] for o in allobs if o["x"] > 0])
    y = np.array([o["delta"] + args.plant * o["x"] for o in allobs if o["x"] > 0])
    ap_, sep_, _ = wls_through_origin(x, y)
    want = a + args.plant
    print(f"\nplanted positive control: +{args.plant:.0f} us per empty column "
          "added to the wide arm")
    print(f"  recovered a = {ap_:.2f} +- {sep_:.2f}, expected {want:.2f}, "
          f"error {ap_ - want:+.6f} us "
          f"({'RECOVERED' if abs(ap_ - want) < 1e-6 else 'DID NOT RECOVER'})")

    # ---- consistency with the rung 1 headline ---------------------------
    mean_empty = float(np.mean([o["x"] for o in allobs]))
    wide_leg_us = float(np.mean([o["wide_us"] for o in allobs]))
    print("\nconsistency with rung 1, in the LOCAL frame")
    print(f"  local mean empty columns per round {mean_empty:.4f}")
    print(f"  mean wide round {wide_leg_us:,.0f} us")
    print(f"  predicted local candidate-leg saving "
          f"{100 * a * mean_empty / wide_leg_us:+.4f} % "
          "(rung 1 measured +1.8064 % on absolute mtp seconds per token)")

    # ---- ranked conversion, in the ranked width frame -------------------
    saving = a * RANKED_EMPTY_COLUMNS_REMOVED
    print("\nranked conversion, F83 frame, NOT the local histogram")
    print(f"  E[empty] = E[M] - 1 - mass(2) - mass(8) - 2*mass(9), exact under "
          "the shipped one-pass table")
    print(f"  ranked empty columns removed per round  {RANKED_EMPTY_LOW:.4f} "
          f"to {RANKED_EMPTY_HIGH:.4f}, midpoint {RANKED_EMPTY_COLUMNS_REMOVED:.4f}")
    print(f"  local  empty columns removed per round  {mean_empty:.4f}")
    print(f"  ranked / local column ratio             "
          f"{RANKED_EMPTY_COLUMNS_REMOVED / mean_empty:.4f}  "
          "(the ranked frame removes fewer columns per round)")
    print(f"  ranked saving per round  a * {RANKED_EMPTY_COLUMNS_REMOVED:.4f} "
          f"= {saving:,.1f} us")
    print(f"  {'prompt':10s} {'round_us':>10s} {'weight':>7s} {'saving %':>9s} "
          f"{'low %':>8s} {'high %':>8s}")
    wsum = sum(w for _, _, w in RANKED_ROUNDS.values())
    wpct = upct = wlo = whi = 0.0
    for name, (rus, _r, w) in RANKED_ROUNDS.items():
        pct = 100 * saving / rus
        lo = 100 * a * RANKED_EMPTY_LOW / rus
        hi = 100 * a * RANKED_EMPTY_HIGH / rus
        wpct += w * pct
        wlo += w * lo
        whi += w * hi
        upct += pct / len(RANKED_ROUNDS)
        print(f"  {name:10s} {rus:10,.0f} {w:7.4f} {pct:9.4f} "
              f"{lo:8.4f} {hi:8.4f}")
    print(f"  F83-weighted ranked candidate-leg saving {wpct / wsum:+.4f} % "
          f"(range {wlo / wsum:+.4f} to {whi / wsum:+.4f}; "
          f"weights renormalised, they sum to {wsum:.4f})")
    print(f"  unweighted eight-prompt mean            {upct:+.4f} % "
          "(approximate: it applies the F83-weighted column count to the "
          "three zero-weight prompts, whose width mix is lower)")
    print("  Rule 112: the null sd of the candidate 8-prompt mean difference "
          "between two ranked receipts is 0.067 %, so the 2 sigma "
          "single-receipt bar is 0.133 %.")
    if abs(wpct / wsum) >= 0.5:
        print("  -> at or above 0.5 %, one ranked receipt settles this.")
    elif abs(wpct / wsum) >= 0.133:
        print("  -> between the 2 sigma bar and 0.5 %, one receipt is "
              "suggestive but not decisive.")
    else:
        print("  -> below the 2 sigma single-receipt bar, UNCONFIRMABLE from "
              "one ranked receipt.")

    # ---- rung 3, the 2x2 -------------------------------------------------
    if {r["cell"] for r in rows} >= {"A", "B", "C", "D"}:
        ys, T, G, P = [], [], [], []
        for r in rows:
            y = fnum(r["metrics"].get("mtp_seconds_per_token"))
            if y is None:
                continue
            ys.append(y)
            T.append(1.0 if r["table"] == "onepass67" else -1.0)
            G.append(1.0 if r["arm"] == "tight" else -1.0)
            P.append(float(r["idx"]))
        ys = np.array(ys)
        T, G = np.array(T), np.array(G)
        P = np.array(P) - np.mean(P)
        beta, bse, cov, dof = ols(
            np.column_stack([np.ones(len(ys)), T, G, T * G, P]), ys)
        print("\nrung 3, the 2x2 on absolute candidate seconds per token")
        print(f"  y ~ 1 + table + grid + table:grid + centred leg index, "
              f"dof {dof}")
        for nm, b, s in zip(
                ["mean", "table", "grid", "table:grid", "drift/leg"], beta, bse):
            print(f"    {nm:12s} {b:+.8f} +- {s:.8f}")
        mean = beta[0]

        def contrast(vec, title):
            v = np.array(vec, dtype=float)
            est = float(v @ beta)
            sd = math.sqrt(float(v @ cov @ v))
            print(f"    {title:44s} {est:+.8f} +- {sd:.8f} s/tok "
                  f"({-100 * est / mean:+.4f} +- {100 * sd / abs(mean):.4f} % "
                  "faster)")
            return est, sd

        contrast([0, 2, 0, -2, 0], "B - A  onepass67 - shipped, WIDE grid")
        d_minus_c = contrast([0, 2, 0, 2, 0],
                             "D - C  onepass67 - shipped, TIGHT grid")
        contrast([0, 0, 2, 2, 0], "D - B  tight - wide, onepass67 table")
        contrast([0, 0, 2, -2, 0], "C - A  tight - wide, shipped table")
        print("\n  e135_onepass_gain_under_tight_pct = "
              f"{-100 * d_minus_c[0] / mean:+.4f}")
        print("\n  cell means, drift removed")
        for cell, t, g in (("A", -1, -1), ("B", 1, -1), ("C", -1, 1), ("D", 1, 1)):
            v = np.array([1.0, t, g, t * g, 0.0])
            print(f"    {cell}  table "
                  f"{'onepass67' if t > 0 else 'shipped  '} grid "
                  f"{'tight' if g > 0 else 'wide ':5s}  {float(v @ beta):.6f} s/tok")

    print(f"\ne135_launch_cost_us_per_column = {a:.2f} +- {se:.2f}")
    print("\nThis session is ungated and counterbalanced. It is directional "
          "causal evidence inside itself, not a gate-qualified reading and "
          "not any kind of official score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
