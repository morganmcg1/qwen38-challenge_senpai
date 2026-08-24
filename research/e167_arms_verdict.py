#!/usr/bin/env python3
"""Read one e167_arms_abba.sh session and decide it against a reading table.

The session script records legs and stops. It deliberately computes no
contrast, because the arm-to-arm comparison needs two estimators that answer
different questions and must be reported side by side:

  POOLED    every leg of an arm against every leg of B0, Welch (unequal
            variance, unequal n). Uses all the data and is the interval the
            stop rule was written against. It is NOT protected against a
            monotone thermal or clock drift across the session.

  BLOCKED   `B0 B1 B2 B3 B3 B2 B1 B0` is one palindromic block in which every
            arm appears twice at positions symmetric about the block centre,
            so every arm has the same position sum. Averaging an arm's two
            legs inside a block cancels a monotone drift to first order, and
            differencing against B0's block mean removes the block level.
            With two blocks this gives two independent drift-cancelled
            estimates of each contrast; their spread is the reported
            uncertainty. Fewer degrees of freedom, but the design's actual
            protection.

An arm effect is only credible when both estimators agree in sign and rough
magnitude. If POOLED shows an effect that BLOCKED does not, suspect drift.

scipy is absent on this host, so the Welch tail comes from a regularised
incomplete beta evaluated by the Lentz continued fraction. `--self-test`
checks it against values computed independently.

usage:
  python3 research/e167_arms_verdict.py [--dir research/out/e167/arms]
                                        [--metric decode_only_spt]
                                        [--reference B0] [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys


# --- Student t two-sided tail, no scipy --------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps, itmax = 1e-300, 3e-16, 400
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x out of range: {x}")
    if x in (0.0, 1.0):
        return x
    lbeta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf2(t: float, df: float) -> float:
    """Two-sided survival function of Student t."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return betainc(0.5 * df, 0.5, df / (df + t * t))


def welch(a: list[float], b: list[float]) -> dict:
    """Welch contrast a - b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"delta": statistics.fmean(a) - statistics.fmean(b),
                "t": float("nan"), "df": float("nan"), "p": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    delta = ma - mb
    if se == 0.0:
        return {"delta": delta, "t": float("inf") if delta else 0.0,
                "df": float("nan"), "p": 0.0 if delta else 1.0,
                "ci95_lo": delta, "ci95_hi": delta}
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = delta / se
    # 97.5th t quantile by bisection on the two-sided tail.
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf2(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    tcrit = 0.5 * (lo + hi)
    return {"delta": delta, "t": t, "df": df, "p": t_sf2(t, df),
            "ci95_lo": delta - tcrit * se, "ci95_hi": delta + tcrit * se}


def self_test() -> None:
    # Welch on two samples whose statistics are exact by construction.
    a = [1.0, 2.0, 3.0, 4.0]          # mean 2.5, var 5/3
    b = [2.0, 3.0, 4.0, 5.0]          # mean 3.5, var 5/3
    r = welch(a, b)
    assert abs(r["delta"] + 1.0) < 1e-12, r
    # se = sqrt(2*(5/3)/4) = sqrt(5/6); t = -1/sqrt(5/6)
    assert abs(r["t"] - (-1.0 / math.sqrt(5.0 / 6.0))) < 1e-12, r
    assert abs(r["df"] - 6.0) < 1e-9, r
    # Equal variance and equal n makes Welch identical to Student t(6).
    # P(|t_6| > 1.09544...) = 0.31534...  (regularised incomplete beta)
    assert abs(r["p"] - 0.315336) < 1e-5, r["p"]
    # Known incomplete beta values.
    assert abs(betainc(0.5, 0.5, 0.5) - 0.5) < 1e-12
    assert abs(betainc(2.0, 3.0, 0.5) - 0.6875) < 1e-12
    assert abs(t_sf2(0.0, 5.0) - 1.0) < 1e-12
    assert abs(t_sf2(2.570582, 5.0) - 0.05) < 1e-5, t_sf2(2.570582, 5.0)
    assert abs(t_sf2(1.959964, 1e7) - 0.05) < 1e-4
    print("e167_arms_verdict: self-test ok")


# --- session reading ---------------------------------------------------------
def read_session(path: pathlib.Path) -> dict:
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()
    return fields


def read_legs(path: pathlib.Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def fnum(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="research/out/e167/arms")
    ap.add_argument("--metric", default="decode_only_spt")
    ap.add_argument("--reference", default="B0")
    ap.add_argument("--block", type=int, default=8,
                    help="legs per palindromic block")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = pathlib.Path(args.dir)
    session = read_session(root / "session.txt")
    legs = read_legs(root / "legs.tsv")
    if not legs:
        print("e167_arms_verdict: no legs", file=sys.stderr)
        return 1

    print("=" * 78)
    print("E167 four-arm counter-strip session")
    print("=" * 78)
    for k in ("utc_start", "utc_end", "git_head", "git_dirty", "tokens",
              "depth", "host", "schedule", "reference_arm", "golden_sha256",
              "golden_rows", "wandb_run_id"):
        if k in session:
            print(f"  {k:22s} {session[k]}")

    # Arm identity must be witnessed per leg, not assumed from the label.
    print("\n-- arm witnesses (per leg, from the binary actually run) --")
    seen: dict[str, set] = {}
    for r in legs:
        seen.setdefault(r["arm"], set()).add((r["worker_sha12"], r["witness"]))
    ok_identity = True
    for arm in sorted(seen):
        for sha, wit in sorted(seen[arm]):
            print(f"  {arm:4s} {sha}  {wit}")
        if len(seen[arm]) != 1:
            ok_identity = False
            print(f"  !! {arm} ran more than one binary; its legs are void")

    # Exactness: every leg must match the ONE reference set generated from the
    # reference arm, which makes this a cross-arm bit-exactness test.
    matched = [r["matched"] for r in legs]
    n_true = sum(1 for m in matched if m == "true")
    print(f"\n-- exactness -- all_tokens_matched true in {n_true}/{len(legs)}"
          f" legs (reference arm {session.get('reference_arm', '?')})")
    if n_true != len(legs):
        for r in legs:
            if r["matched"] != "true":
                print(f"  !! leg {r['leg']} arm {r['arm']}"
                      f" matched={r['matched']}")

    # Thermal record. Ungated arms are legal only with this reported.
    entries = [fnum(r, "entry_c") for r in legs]
    entries = [e for e in entries if not math.isnan(e)]
    if entries:
        print(f"\n-- thermal -- entry C: min {min(entries):.1f}"
              f" max {max(entries):.1f} spread {max(entries)-min(entries):.1f}")
        for arm in sorted(seen):
            ea = [fnum(r, "entry_c") for r in legs if r["arm"] == arm]
            ea = [e for e in ea if not math.isnan(e)]
            if ea:
                print(f"  {arm:4s} entry mean {statistics.fmean(ea):7.2f} C"
                      f"  legs {[f'{e:.1f}' for e in ea]}")

    # --- per-arm summary ----------------------------------------------------
    for metric in (args.metric, "spt", "prefill_s", "edl", "accepted",
                   "rounds"):
        vals = {arm: [fnum(r, metric) for r in legs if r["arm"] == arm]
                for arm in sorted(seen)}
        vals = {a: [v for v in xs if not math.isnan(v)]
                for a, xs in vals.items()}
        if not any(vals.values()):
            continue
        star = " (PRIMARY)" if metric == args.metric else ""
        print(f"\n-- {metric}{star} --")
        print(f"  {'arm':5s} {'n':>2s} {'mean':>14s} {'sd':>12s}"
              f" {'cv%':>7s}  legs")
        for arm, xs in vals.items():
            if not xs:
                continue
            m = statistics.fmean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            cv = 100.0 * sd / m if m else float("nan")
            print(f"  {arm:5s} {len(xs):2d} {m:14.8g} {sd:12.6g}"
                  f" {cv:7.3f}  {[round(x, 8) for x in xs]}")

    ref = args.reference
    prim = {arm: [fnum(r, args.metric) for r in legs if r["arm"] == arm]
            for arm in sorted(seen)}
    prim = {a: [v for v in xs if not math.isnan(v)] for a, xs in prim.items()}
    if ref not in prim or not prim[ref]:
        print(f"\ne167_arms_verdict: reference arm {ref} has no legs",
              file=sys.stderr)
        return 1
    base = statistics.fmean(prim[ref])

    print(f"\n-- POOLED Welch contrast on {args.metric}, vs {ref}"
          f" (mean {base:.8g} s/token) --")
    print("  a negative delta means the arm is FASTER than the reference")
    print(f"  {'arm':5s} {'delta_s':>13s} {'delta_%':>9s} {'ci95_% lo':>10s}"
          f" {'hi':>9s} {'t':>8s} {'df':>7s} {'p':>9s}")
    pooled = {}
    for arm, xs in prim.items():
        if arm == ref or not xs:
            continue
        r = welch(xs, prim[ref])
        pooled[arm] = r
        print(f"  {arm:5s} {r['delta']:13.6g} {100*r['delta']/base:9.3f}"
              f" {100*r['ci95_lo']/base:10.3f} {100*r['ci95_hi']/base:9.3f}"
              f" {r['t']:8.3f} {r['df']:7.2f} {r['p']:9.4f}")

    # --- blocked (drift-cancelled) contrast --------------------------------
    nb = args.block
    blocks = [legs[i:i + nb] for i in range(0, len(legs), nb)]
    blocks = [b for b in blocks if len(b) == nb]
    print(f"\n-- BLOCKED contrast on {args.metric}: {len(blocks)} palindromic"
          f" block(s) of {nb} legs, vs {ref} --")
    if not blocks:
        print("  (no complete block)")
    else:
        per_block: dict[str, list[float]] = {}
        for bi, blk in enumerate(blocks, 1):
            bvals = {}
            for arm in sorted(seen):
                xs = [fnum(r, args.metric) for r in blk if r["arm"] == arm]
                xs = [v for v in xs if not math.isnan(v)]
                if xs:
                    bvals[arm] = statistics.fmean(xs)
            if ref not in bvals:
                print(f"  block {bi}: no {ref} leg; skipped")
                continue
            line = f"  block {bi}: {ref}={bvals[ref]:.8g}"
            for arm in sorted(bvals):
                if arm == ref:
                    continue
                d = bvals[arm] - bvals[ref]
                per_block.setdefault(arm, []).append(d)
                line += f"  {arm}{100*d/bvals[ref]:+.3f}%"
            print(line)
        print(f"\n  {'arm':5s} {'mean_delta_%':>13s} {'range_%':>18s}"
              f"  per-block %")
        for arm in sorted(per_block):
            ds = [100 * d / base for d in per_block[arm]]
            rng = (f"[{min(ds):+.3f}, {max(ds):+.3f}]" if len(ds) > 1
                   else "single block")
            print(f"  {arm:5s} {statistics.fmean(ds):13.3f} {rng:>18s}"
                  f"  {[round(d, 3) for d in ds]}")

        # Sign agreement between the two estimators is the credibility test.
        print("\n-- estimator agreement --")
        for arm in sorted(per_block):
            pd = 100 * pooled[arm]["delta"] / base if arm in pooled else float("nan")
            bd = statistics.fmean([100 * d / base for d in per_block[arm]])
            agree = "yes" if (pd > 0) == (bd > 0) else "NO"
            print(f"  {arm:5s} pooled {pd:+7.3f}%  blocked {bd:+7.3f}%"
                  f"  same sign: {agree}")

    if not ok_identity:
        print("\ne167_arms_verdict: ARM IDENTITY FAILED; do not read the"
              " contrasts above", file=sys.stderr)
        return 1
    if n_true != len(legs):
        print("\ne167_arms_verdict: EXACTNESS FAILED; the contrasts above are"
              " not comparable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
