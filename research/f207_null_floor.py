#!/usr/bin/env python3
"""Finding 207 instrument: measure the ranked RUN-LEVEL noise floor at large N.

Campaign Rule 112 claims the null sd of the candidate 8-prompt mean difference
between two ranked receipts is 0.067 %.  That figure came from a handful of
pairs.  Several recent one-line isolations read +/- 1 %, which Rule 112 would
call a 15 sigma effect, and a provably decode-null pair (Finding 197) reads
-158 us per drafting round.  Rule 112 is therefore under test.

The serial leg is the instrument.  Every board row runs the SAME runner-owned
prebuilt serial workspace, so any dispersion in the serial per-prompt vector is
pure run-level noise: host state, thermal state, runner drift, and the Finding
75 measurement mode.  With ~100 scored rows in a day this is a large-N estimate
of the noise floor that must also apply to the candidate leg.

Method
    For each prompt p and row r, take y[r][p] = ln(serial_seconds_per_token).
    Remove the prompt mean.  The residual row effect

        u[r] = mean_p ( y[r][p] - mean_r y[.][p] )

    is the run-level multiplicative state of row r.  sd(u) is the run-level
    noise floor of an 8-prompt mean.  The sd of a DIFFERENCE of two rows is
    sqrt(2) times that.

    The same decomposition is applied to the candidate leg restricted to a
    narrow score band, where the code differences are small, to bound the
    candidate-leg floor from above.

Usage
    python3 research/f207_null_floor.py [--board PATH] [--since ISO8601]
"""
import argparse
import json
import math
import statistics as st

BOARD = "/tmp/yukon-board/full.json"

PROMPT_BY_SHA8 = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ORDER = ["plutarch", "drama", "travel", "beagle", "republic",
         "essays", "medicine", "botany"]
MEDPAIR = ("beagle", "essays")


def load(path, since):
    with open(path) as fh:
        subs = json.load(fh)["submissions"]
    rows = []
    for r in subs:
        pp = (r.get("officialMetrics") or {}).get("per_prompt")
        if not pp or len(pp) != 8:
            continue
        created = r.get("createdAt") or ""
        if since and created < since:
            continue
        legs = {}
        for e in pp:
            nm = PROMPT_BY_SHA8.get(e["prompt_sha256"][:8])
            if nm:
                legs[nm] = e
        if len(legs) != 8:
            continue
        rows.append({
            "id": (r.get("id") or "")[:8],
            "who": r.get("solverUsername") or "?",
            "score": r.get("officialScore"),
            "created": created,
            "legs": legs,
        })
    rows.sort(key=lambda r: r["created"])
    return rows


def row_effects(rows, field, prompts):
    """Return {id: mean log-residual} after removing each prompt's mean."""
    logs = {p: [] for p in prompts}
    for r in rows:
        for p in prompts:
            logs[p].append(math.log(r["legs"][p][field]))
    base = {p: st.mean(logs[p]) for p in prompts}
    out = {}
    for r in rows:
        resid = [math.log(r["legs"][p][field]) - base[p] for p in prompts]
        out[r["id"]] = st.mean(resid)
    return out


def describe(tag, eff, rows):
    vals = [eff[r["id"]] for r in rows]
    pct = [100.0 * v for v in vals]
    sd = st.pstdev(pct)
    print("%-46s n %3d  sd %6.4f %%  range %+7.4f .. %+7.4f  "
          "diff-of-two sd %6.4f %%"
          % (tag, len(pct), sd, min(pct), max(pct), sd * math.sqrt(2.0)))
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=BOARD)
    ap.add_argument("--since", default="2026-08-22T12:00:00")
    args = ap.parse_args()

    rows = load(args.board, args.since)
    print("scored rows with all eight legs since %s : %d" % (args.since, len(rows)))
    print()

    print("=== RUN-LEVEL NOISE FLOOR (log row effect, prompt means removed) ===")
    ser8 = row_effects(rows, "serial_seconds_per_token_mean", ORDER)
    serm = row_effects(rows, "serial_seconds_per_token_mean", MEDPAIR)
    cand8 = row_effects(rows, "mtp_seconds_per_token_mean", ORDER)
    candm = row_effects(rows, "mtp_seconds_per_token_mean", MEDPAIR)
    pre8 = row_effects(rows, "prefill_seconds_per_token", ORDER)

    s8 = describe("serial   8-prompt mean  (IDENTICAL CODE)", ser8, rows)
    describe("serial   medpair mean   (IDENTICAL CODE)", serm, rows)
    describe("prefill  8-prompt mean  (candidate build)", pre8, rows)
    describe("candidate 8-prompt mean (code DIFFERS)", cand8, rows)
    describe("candidate medpair mean  (code DIFFERS)", candm, rows)
    print()
    print("The serial rows are the null: same code, same fixture, same window.")
    print("Campaign Rule 112 claims a candidate 8-prompt diff-of-two sd of")
    print("0.067 %%.  The serial diff-of-two sd above is the floor that the")
    print("candidate leg cannot beat, because both legs share the host state.")
    print()

    print("=== IS THE SERIAL ROW EFFECT MULTI-MODAL? (Finding 75) ===")
    ss = sorted(s8)
    qs = [ss[int(q * (len(ss) - 1))] for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
    print("quantiles %%  min %+7.4f  p10 %+7.4f  p25 %+7.4f  med %+7.4f  "
          "p75 %+7.4f  p90 %+7.4f  max %+7.4f" % tuple(qs))
    gaps = [(ss[i + 1] - ss[i], ss[i], ss[i + 1]) for i in range(len(ss) - 1)]
    gaps.sort(reverse=True)
    print("three largest gaps in the sorted serial row effect:")
    for g, lo, hi in gaps[:3]:
        print("    gap %6.4f pp between %+7.4f and %+7.4f" % (g, lo, hi))
    print()

    print("=== DRIFT: serial row effect against submission time ===")
    n = len(rows)
    for i in range(0, n, max(1, n // 12)):
        r = rows[i]
        print("  %s  %-14s %-8s serial %+7.4f %%   candidate %+7.4f %%"
              % (r["created"][11:19], r["who"], r["id"],
                 100 * ser8[r["id"]], 100 * cand8[r["id"]]))
    print()

    print("=== CANDIDATE-LEG FLOOR FROM PER-PROMPT RESIDUAL SCATTER ===")
    print("For a pair, the per-prompt candidate delta scatter around its own")
    print("mean is the measurement noise plus any real heterogeneity.  A pair")
    print("with no decode mechanism gives the pure floor.")
    print()
    idx = {r["id"]: r for r in rows}
    pairs = [
        ("ed608e64", "115c5c50", "F197 provable decode-null (prefill swizzle)"),
        ("ed608e64", "08b67f12", "probe 0.25 -> 0.15  (confirmed lever)"),
        ("08b67f12", "1760479a", "width-2 launch shrink (crown)"),
        ("ed608e64", "9b6023c1", "probe 0.25 -> 0.12  (FINDING 206)"),
        ("ed608e64", "0b2f0014", "F194 one-pass table on tight base"),
        ("ed608e64", "c466d4d7", "fkiene tight-grid sibling"),
        ("ed608e64", "f4335350", "Carme99 tight-grid sibling"),
        ("ed608e64", "572b2cc4", "our tight grid (different tree)"),
    ]
    print("%-9s %-9s %-42s %8s %8s %8s %8s"
          % ("A", "B", "what", "mean %", "sd %", "se8 %", "medpair"))
    for a, b, what in pairs:
        if a not in idx or b not in idx:
            print("%-9s %-9s %-42s   MISSING" % (a, b, what))
            continue
        d = []
        for p in ORDER:
            va = idx[a]["legs"][p]["mtp_seconds_per_token_mean"]
            vb = idx[b]["legs"][p]["mtp_seconds_per_token_mean"]
            d.append(100.0 * (vb / va - 1.0))
        m = st.mean(d)
        sd = st.pstdev(d)
        mp = st.mean([d[ORDER.index(p)] for p in MEDPAIR])
        print("%-9s %-9s %-42s %+8.4f %8.4f %8.4f %+8.4f"
              % (a, b, what, m, sd, sd / math.sqrt(8.0), mp))
    print()

    print("=== PER-ROW SERIAL STATE, most recent 25 rows ===")
    print("%-8s %-15s %-9s %9s %9s %9s"
          % ("id", "who", "score", "serial %", "cand %", "prefill %"))
    for r in rows[-25:]:
        sc = r["score"]
        print("%-8s %-15s %9s %+9.4f %+9.4f %+9.4f"
              % (r["id"], r["who"],
                 ("%.5f" % sc) if sc is not None else "-",
                 100 * ser8[r["id"]], 100 * cand8[r["id"]], 100 * pre8[r["id"]]))


if __name__ == "__main__":
    main()
