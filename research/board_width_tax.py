"""Finding 29 - the within-run width tax.

Screens any proposed mechanism that claims to change the cost of WIDE verify
rounds relative to NARROW ones, using only public Yukon board fields and zero
GPU seconds.

For one scored run, form

    W = mean over the five G=2 prompts of log(mtp_seconds_per_token)
      - mean over the three G=1 prompts of log(mtp_seconds_per_token)

and compare it only inside a cohort of runs whose eight per-prompt
`effective_mean_draft_len` values are bit-identical, so every run in the cohort
offers the same eight verify-width mixes.

Why W is sharp:

  1. It uses only the candidate leg, so the serial-baseline lottery of
     Finding 20 cannot enter it.
  2. Any run-level common shift cancels exactly - the FACT 2 measurement mode,
     thermal state, and a uniform per-kernel register tax all scale both terms.
  3. The FACT 2 mode very nearly cancels as well.  It costs about 0.601 ms per
     DRAFTING round.  On the G=2 prompts that is about +1.0 % of a round; on
     the G=1 prompts it is +1.5 % on drama, +1.44 % on travel and +0.15 % on
     plutarch, mean +1.03 %.  The two terms differ by 0.03 pp.

Measured resolution on the 2026-08-21 board: sd(W) = 0.509 pp over a cohort of
215 runs scoring at or above 3.15, which is about four times finer than the
0.277 % published single-pair floor of Finding 6 once that floor is expressed
on the same quantity.

Usage:
    python3 research/board_width_tax.py <id-prefix> [score-floor]

Refresh /tmp/yukon-board/full.json first; the file is a Yukon board dump whose
rows carry `officialMetrics.per_prompt`.
"""
import json
import math
import statistics
import sys

BOARD = "/tmp/yukon-board/full.json"

SHA2NAME = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
# The dispatch-group split of Finding 12: five scoring prompts run above the
# M = 4 boundary and three below it.  Membership is a property of the prompt,
# not of the run, for every schedule the campaign has seen at or above 3.15.
G2 = ("beagle", "botany", "essays", "medicine", "republic")
G1 = ("drama", "plutarch", "travel")


def load_board(path=BOARD):
    doc = json.load(open(path))
    for key in ("submissions", "rows", "data", "items"):
        if isinstance(doc, dict) and key in doc:
            return [r for r in doc[key] if isinstance(r, dict)]
    return [r for r in doc if isinstance(r, dict)]


def per_prompt(row):
    metrics = row.get("officialMetrics") or {}
    entries = metrics.get("per_prompt") or []
    if len(entries) != 8:
        return None
    out = {}
    for entry in entries:
        name = SHA2NAME.get(entry["prompt_sha256"][:8])
        if name is None:
            return None
        out[name] = entry
    return out if len(out) == 8 else None


def schedule_key(prompts):
    return tuple(round(prompts[n]["effective_mean_draft_len"], 12)
                 for n in sorted(prompts))


def width_tax(prompts):
    hi = statistics.fmean(
        math.log(prompts[n]["mtp_seconds_per_token_mean"]) for n in G2)
    lo = statistics.fmean(
        math.log(prompts[n]["mtp_seconds_per_token_mean"]) for n in G1)
    return hi - lo


def report(target_prefix, floor=3.15):
    rows = [r for r in load_board()
            if per_prompt(r) and r.get("officialScore")]
    target = next(r for r in rows if r["id"].startswith(target_prefix))
    key = schedule_key(per_prompt(target))
    cohort = [r for r in rows
              if schedule_key(per_prompt(r)) == key
              and (r["officialScore"] >= floor or r["id"] == target["id"])]
    if len(cohort) < 8:
        raise SystemExit(
            f"cohort of {len(cohort)} is too small to calibrate; lower the floor")

    values = sorted((width_tax(per_prompt(r)), r) for r in cohort)
    mean = statistics.fmean(v for v, _ in values)
    sd = statistics.stdev(v for v, _ in values)
    wt = width_tax(per_prompt(target))
    rank = sum(1 for v, _ in values if v < wt) + 1

    print(f"target  {target['id'][:8]}  {target.get('solverUsername')}  "
          f"{target['officialScore']:.8f}  {str(target.get('createdAt'))[:19]}")
    print(f"cohort  {len(values)} schedule-matched runs at or above {floor}")
    print(f"W       mean {mean:+.6f}   sd {sd*100:.4f} pp")
    print(f"target  W {wt:+.6f}   z {(wt - mean)/sd:+.3f}   "
          f"rank {rank} of {len(values)}  (1 = cheapest wide rounds)")
    print()

    tgt = per_prompt(target)
    others = [r for r in cohort if r["id"] != target["id"]]
    print("prompt         M      d%       z    cohort sd%")
    shape = {}
    for name in sorted(tgt, key=lambda n: tgt[n]["effective_mean_draft_len"]):
        logs = [math.log(per_prompt(r)[name]["mtp_seconds_per_token_mean"])
                for r in others]
        mu = statistics.fmean(logs)
        psd = statistics.stdev(logs)
        d = math.log(tgt[name]["mtp_seconds_per_token_mean"]) - mu
        shape[name] = d
        print(f"{name:10s} {1.0 + tgt[name]['effective_mean_draft_len']:6.3f} "
              f"{d*100:+7.3f} {d/psd:+7.2f}  {psd*100:8.3f}")

    common = statistics.fmean(shape[n] for n in G1)
    print()
    print(f"G=1 common shift {common*100:+.3f} %   ({', '.join(G1)})")
    print("prompt         M   shape%   = per-prompt d% minus the common shift")
    for name in sorted(G2, key=lambda n: tgt[n]["effective_mean_draft_len"]):
        print(f"{name:10s} {1.0 + tgt[name]['effective_mean_draft_len']:6.3f} "
              f"{(shape[name] - common)*100:+7.3f}")
    hi = statistics.fmean(shape[n] - common for n in G2)
    print(f"{'mean':10s} {'':6s} {hi*100:+7.3f}   "
          f"against a cohort sd of {sd*100:.3f} pp")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    report(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.15)
