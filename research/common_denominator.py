#!/usr/bin/env python3
"""Rule 100 instrument: rank board rows on a COMMON SERIAL DENOMINATOR.

The ranked score is median_p( serial_p / candidate_p ). The serial leg comes
from a runner-owned prebuilt workspace that no candidate edit can touch, yet its
per-prompt seconds per token still disperses with sd 0.21 % to 0.24 % across
runs (Finding 153). Inside one runner state that lottery contributes sd 0.0967 %
to the published median, which is more variance than the real candidate
differences across the whole frontier cluster.

So the published ordering at the top of the board is NOT a merit ordering.
Before pricing any rival row, re-rank the population with one shared serial
vector so only the candidate legs differ.

Usage
    YUKON_API_TOKEN=... python3 research/common_denominator.py            # fetch
    python3 research/common_denominator.py --board /tmp/yukon-board/full.json
    python3 research/common_denominator.py --anchor 48423d09 --cluster 0.30

Companion instruments
    research/cluster3.py    Rule 98, per-prompt candidate agreement
    research/modetest.py    Finding 76, runner-state mode index
"""
import argparse
import json
import os
import statistics as st
import urllib.request

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"

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
ORDER = ["plutarch", "drama", "travel", "beagle", "republic", "essays", "medicine", "botany"]


def fetch_board(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    token = os.environ["YUKON_API_TOKEN"]
    url = f"https://api.yukon.org/api/benchmarks/{BENCHMARK_ID}/submissions?all=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.load(fh)


def load_rows(board):
    """Keep only rows carrying all eight per-prompt legs."""
    rows = []
    for r in board["submissions"]:
        per_prompt = (r.get("officialMetrics") or {}).get("per_prompt")
        if not per_prompt or len(per_prompt) != 8:
            continue
        legs = {}
        for entry in per_prompt:
            name = PROMPT_BY_SHA8.get(entry["prompt_sha256"][:8])
            if name and entry.get("serial_seconds_per_token_mean"):
                legs[name] = entry
        if len(legs) == 8:
            rows.append((r, legs))
    return rows


def median8(serial, candidate):
    raws = sorted(serial[p] / candidate[p] for p in ORDER)
    return (raws[3] + raws[4]) / 2.0


def vec(legs, key):
    return {p: legs[p][key] for p in ORDER}


def mean_abs_pct_diff(a, b):
    return 100.0 * st.mean(abs(a[p] / b[p] - 1.0) for p in ORDER)


def report_serial_dispersion(rows):
    print("== pinned serial leg dispersion (identical work on every run)")
    print(f"{'prompt':10s} {'n':>4s} {'mean s/tok':>12s} {'sd %':>8s} {'p5-p95 %':>10s} {'min-max %':>10s}")
    for p in ORDER:
        v = sorted(legs[p]["serial_seconds_per_token_mean"] for _, legs in rows)
        mu = st.mean(v)
        lo, hi = v[int(0.05 * len(v))], v[int(0.95 * len(v))]
        print(f"{p:10s} {len(v):4d} {mu:12.6f} {100*st.pstdev(v)/mu:8.3f} "
              f"{100*(hi-lo)/mu:10.3f} {100*(v[-1]-v[0])/mu:10.3f}")
    run_means = [st.mean(legs[p]["serial_seconds_per_token_mean"] for p in ORDER)
                 for _, legs in rows]
    within = []
    for _, legs in rows:
        rm = st.mean(legs[p]["serial_seconds_per_token_mean"] for p in ORDER)
        within += [100 * (legs[p]["serial_seconds_per_token_mean"] - rm) / rm for p in ORDER]
    print(f"  run-level serial mean sd      {100*st.pstdev(run_means)/st.mean(run_means):.4f} %")
    print(f"  within-run per-prompt residual {st.pstdev(within):.4f} %")
    print("  a larger within-run term means the draw is PER PAIR, not per run (Finding 62)")


def report_median_carriers(rows):
    print("\n== which prompts occupy median order statistics 4 and 5")
    count = {}
    for _, legs in rows:
        raws = sorted((legs[p]["serial_seconds_per_token_mean"]
                       / legs[p]["mtp_seconds_per_token_mean"], p) for p in ORDER)
        for _, p in (raws[3], raws[4]):
            count[p] = count.get(p, 0) + 1
    for p, c in sorted(count.items(), key=lambda kv: -kv[1]):
        print(f"  {p:10s} {c:5d} / {len(rows)}  ({100*c/len(rows):5.1f} %)")
    print("  published ~= 0.5*raw_beagle + 0.5*min(medicine, essays, republic, botany)")


def report_cluster(rows, anchor_id, tol):
    by_id = {r["id"][:8]: (r, legs) for r, legs in rows}
    if anchor_id not in by_id:
        print(f"\n!! anchor {anchor_id} has no complete per-prompt evidence")
        return
    anchor_row, anchor_legs = by_id[anchor_id]
    anchor_cand = vec(anchor_legs, "mtp_seconds_per_token_mean")
    anchor_serial = vec(anchor_legs, "serial_seconds_per_token_mean")

    cluster = [(r, legs) for r, legs in rows
               if mean_abs_pct_diff(vec(legs, "mtp_seconds_per_token_mean"), anchor_cand) <= tol]
    scores = [r["officialScore"] for r, _ in cluster]
    print(f"\n== cluster around {anchor_id}, mean |candidate diff| <= {tol:.2f} %   n = {len(scores)}")
    if len(scores) < 3:
        print("   too few rows to decompose")
        return
    mu = st.mean(scores)
    print(f"   published median  mean {mu:.6f}  sd {100*st.pstdev(scores)/mu:.4f} %  "
          f"span {100*(max(scores)-min(scores))/mu:.3f} %")
    serial_only = [median8(vec(legs, "serial_seconds_per_token_mean"), anchor_cand)
                   for _, legs in cluster]
    cand_only = [median8(anchor_serial, vec(legs, "mtp_seconds_per_token_mean"))
                 for _, legs in cluster]
    print(f"   serial-vector-only    sd {100*st.pstdev(serial_only)/st.mean(serial_only):.4f} %"
          "   <- pure numerator lottery")
    print(f"   candidate-vector-only sd {100*st.pstdev(cand_only)/st.mean(cand_only):.4f} %"
          "   <- real tree differences")

    print(f"\n== COMMON-DENOMINATOR LEADERBOARD (serial vector pinned to {anchor_id})")
    board = sorted(
        ((median8(anchor_serial, vec(legs, "mtp_seconds_per_token_mean")),
          r["id"][:8], r.get("solverUsername") or "?", r["officialScore"], r.get("status"))
         for r, legs in rows),
        reverse=True)
    print(f"  {'rank':>4s} {'common-den':>11s} {'published':>10s} {'id':10s} {'solver':16s} status")
    for i, (v, rid, user, score, status) in enumerate(board[:15], 1):
        mark = "  <== OURS" if user == "morganmcg1" else ""
        print(f"  {i:4d} {v:11.6f} {score:10.6f} {rid:10s} {user:16s} {status}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--anchor", default=None,
                    help="row id prefix whose serial vector becomes the common denominator; "
                         "defaults to the highest published score")
    ap.add_argument("--cluster", type=float, default=0.30,
                    help="Rule 98 tolerance in mean absolute candidate percent")
    args = ap.parse_args()

    rows = load_rows(fetch_board(args.board))
    print(f"rows with complete per-prompt evidence: {len(rows)}")
    report_serial_dispersion(rows)
    report_median_carriers(rows)

    anchor = args.anchor
    if anchor is None:
        anchor = max(rows, key=lambda rl: rl[0]["officialScore"])[0]["id"][:8]
        print(f"\nanchor defaulted to the highest published score: {anchor}")
    report_cluster(rows, anchor, args.cluster)


if __name__ == "__main__":
    main()
