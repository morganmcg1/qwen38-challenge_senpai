#!/usr/bin/env python3
"""E147 Rung D: read one ranked receipt and emit research/e147-rungD.json.

harness=ranked throughout. Every number here comes from the public Yukon
receipt for one submission, never from the local M4 Pro session.

The receipt exposes ten per-prompt fields. Three matter here:

  prefill_seconds_per_token   the charged seed-prefill share of the candidate
                              leg, already divided by the 512 counted tokens.
  mtp_seconds_per_token_mean  the whole candidate leg, seed prefill included.
  serial_seconds_per_token_mean  the runner's prebuilt baseline leg.

`mtp_seconds_per_token_mean - prefill_seconds_per_token` is therefore the
decode-only share, and multiplying by 512 restores leg seconds.

Two repricing models appear below and are never mixed:

  model_b  raw' = (S - dP) / (C - dP).  This is the advisor's value table.
  model_a  raw' = S / (C - dP).  This is what `program.md` "Ranked And Local
           Causal Boundaries" implies, because the ranked serial numerator
           comes from a runner-owned prebuilt baseline workspace that
           candidate-editable code cannot reach.

Usage:

    YUKON_API_TOKEN=... python3 research/e147_rungD.py 7226dc9 --fetch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import urllib.request

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = "/tmp/yukon-board/e147-rungD.json"

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
PROMPTS = sorted(PROMPT_NAMES.values())

RANKED_PREFILL_SPT_BAR = 0.001028291
RANKED_BAND_LOW = 0.0010277
RANKED_BAND_HIGH = 0.0010349
BAR_ID8 = "684821ed"
BAR_MEDIAN = 3.71959722580154

# harness=ranked. Advisor value model (model_b), published median against a
# uniform seed-prefill cut, derived from the bar row.
FORECAST_TABLE = {
    0.0: 0.0,
    -1.5: 0.1122,
    -3.0: 0.2251,
    -4.1: 0.3071,
    -5.1: 0.3822,
}

# Feedback 3, D-1. The two schedule columns the branch could ship.
SHIP_DLEN = {
    "beagle": 4.3818, "essays": 5.0870, "medicine": 5.2556, "republic": 4.9892,
    "botany": 6.1481, "drama": 2.2976, "travel": 2.6557, "plutarch": 0.1540,
}
PB6_DLEN = {
    "beagle": 4.2069, "essays": 5.2955, "medicine": 5.2222, "republic": 4.7938,
    "botany": 5.8118, "drama": 2.2948, "travel": 2.6343, "plutarch": 2.6995,
}

# Feedback 3, D-3. F92 per-prompt accepted-draft rates.
ACCEPT_RATE = {
    "beagle": 0.834, "medicine": 0.892, "essays": 0.897, "botany": 0.865,
    "republic": 0.903, "drama": 0.449, "travel": 0.533, "plutarch": 0.333,
}

# Feedback 3, D-2. Campaign constant, microseconds of extra recurrent state
# work per drafting round, sd 54.3 over n=3.
STATE_STEP_US = 879.0
STATE_STEP_SD_US = 54.3
ANCHOR_ID8 = "e003a86d"
DECODE_TOKENS = 512


def fetch() -> list:
    token = os.environ["YUKON_API_TOKEN"]
    req = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        rows = json.loads(resp.read().decode())["submissions"]
    pathlib.Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(CACHE).write_text(json.dumps(rows))
    return rows


def load(do_fetch: bool) -> list:
    if do_fetch or not pathlib.Path(CACHE).is_file():
        return fetch()
    return json.loads(pathlib.Path(CACHE).read_text())


def vec(row: dict) -> dict:
    return {PROMPT_NAMES[e["prompt_sha256"][:8]]: e
            for e in row["officialMetrics"]["per_prompt"]}


def find(rows: list, prefix: str) -> dict:
    hits = [r for r in rows if r["id"].startswith(prefix)]
    if not hits:
        raise SystemExit("no submission starts with %r" % prefix)
    if len(hits) > 1:
        raise SystemExit("ambiguous prefix %r" % prefix)
    return hits[0]


def published_median(values) -> float:
    ordered = sorted(values)
    return (ordered[3] + ordered[4]) / 2.0


def interpolate_forecast(pct: float) -> float:
    xs = sorted(FORECAST_TABLE)
    if pct >= xs[-1]:
        return FORECAST_TABLE[xs[-1]]
    lo, hi = (xs[0], xs[1]) if pct <= xs[0] else (
        max(x for x in xs if x <= pct), min(x for x in xs if x >= pct))
    if lo == hi:
        return FORECAST_TABLE[lo]
    frac = (pct - lo) / (hi - lo)
    return FORECAST_TABLE[lo] + frac * (FORECAST_TABLE[hi] - FORECAST_TABLE[lo])


def counterfactual_median(entries: dict, model: str) -> float:
    """Published median this row would show at the bar's seed prefill."""
    raws = []
    for name in PROMPTS:
        e = entries[name]
        s = e["serial_seconds_per_token_mean"]
        c = e["mtp_seconds_per_token_mean"]
        # dP is negative when the candidate is faster than the bar, so adding
        # it back restores the bar-level prefill.
        d_p = e["prefill_seconds_per_token"] - RANKED_PREFILL_SPT_BAR
        raws.append((s - d_p) / (c - d_p) if model == "model_b" else s / (c - d_p))
    return published_median(raws)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="submission id prefix")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="research/e147-rungD.json")
    args = ap.parse_args()

    rows = load(args.fetch)
    row = find(rows, args.submission)
    metrics = (row.get("officialMetrics") or {})
    if not metrics.get("per_prompt"):
        raise SystemExit("submission %s has no per-prompt receipt yet (status %s)"
                         % (row["id"][:8], row.get("status")))
    ours = vec(row)
    anchor = vec(find(rows, ANCHOR_ID8))

    prefill = {p: ours[p]["prefill_seconds_per_token"] for p in PROMPTS}
    values = [prefill[p] for p in PROMPTS]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    ranked_prefill_pct = 100.0 * (mean / RANKED_PREFILL_SPT_BAR - 1.0)

    raws = {p: ours[p]["raw_ratio_of_means"] for p in PROMPTS}
    median = published_median(raws.values())
    forecast = interpolate_forecast(ranked_prefill_pct)
    realised = {
        model: 100.0 * (median / counterfactual_median(ours, model) - 1.0)
        for model in ("model_a", "model_b")
    }

    # D-1. Which depth-price column did the built worker actually ship?
    dlen = {p: ours[p]["effective_mean_draft_len"] for p in PROMPTS}
    ndr = {p: ours[p]["non_drafting_round_count"] for p in PROMPTS}
    dist_ship = sum(abs(dlen[p] - SHIP_DLEN[p]) for p in PROMPTS)
    dist_pb6 = sum(abs(dlen[p] - PB6_DLEN[p]) for p in PROMPTS)
    matches_pb6 = float(
        abs(dlen["plutarch"] - PB6_DLEN["plutarch"]) < 0.05 and ndr["plutarch"] == 0)

    # D-3. Rounds implied by the accepted-draft rate and the draft length.
    rounds = {p: DECODE_TOKENS / (1.0 + ACCEPT_RATE[p] * dlen[p]) for p in PROMPTS}
    drafting_rounds = {p: rounds[p] - ndr[p] for p in PROMPTS}

    # D-2. Rule 133 decode-only comparison against the only public pb6 anchor.
    decode_ours = {p: (ours[p]["mtp_seconds_per_token_mean"]
                       - prefill[p]) * DECODE_TOKENS for p in PROMPTS}
    decode_anchor = {p: (anchor[p]["mtp_seconds_per_token_mean"]
                         - anchor[p]["prefill_seconds_per_token"]) * DECODE_TOKENS
                     for p in PROMPTS}
    # A prompt only supports the comparison when both rows drafted the same
    # schedule there. Otherwise the decode delta prices the schedule change,
    # not the recurrent state work the constant describes.
    comparable = [p for p in PROMPTS
                  if abs(dlen[p] - anchor[p]["effective_mean_draft_len"]) <= 0.05
                  and drafting_rounds[p] > 0.5]
    step_us, steps, corrected = {}, {}, {}
    for p in PROMPTS:
        delta = decode_ours[p] - decode_anchor[p]
        if drafting_rounds[p] <= 0.5:
            step_us[p] = None
            steps[p] = None
            corrected[p] = decode_ours[p]
            continue
        step_us[p] = abs(delta) * 1e6 / drafting_rounds[p]
        steps[p] = step_us[p] / STATE_STEP_US
        whole = round(steps[p])
        corrected[p] = decode_ours[p] - math.copysign(
            whole * STATE_STEP_US * 1e-6 * drafting_rounds[p], delta)
    pool = comparable or [p for p in PROMPTS if steps[p] is not None]
    usable = [steps[p] for p in pool]
    integer_like = [abs(v - round(v)) <= 0.15 for v in usable]
    all_steps = [steps[p] for p in PROMPTS if steps[p] is not None]

    out = {
        "harness": "ranked",
        "submission_id": row["id"],
        "submission_id8": row["id"][:8],
        "status": row.get("status"),
        "promotion_status": row.get("promotionStatus"),
        "created_at": row.get("createdAt"),
        "score": row.get("officialScore"),
        "improved": row.get("improved"),
        "promoted_source_ref": row.get("promotedSourceRef"),
        "submission_commit_sha": row.get("submissionCommitSha"),
        "bar_id8": BAR_ID8,
        "bar_published_median": BAR_MEDIAN,
        "prompt_order": PROMPTS,
        "prefill_seconds_per_token": values,
        "prefill_seconds_per_token_by_prompt": prefill,
        "raw_ratio_by_prompt": raws,
        "published_median_recomputed": median,
        "realised_median_pct": realised["model_b"],
        "realised_median_pct_model_a": realised["model_a"],
        "forecast_median_pct": forecast,
        "dlen_by_prompt": dlen,
        "non_drafting_round_count_by_prompt": ndr,
        "schedule_column_l1_distance_ship": dist_ship,
        "schedule_column_l1_distance_pb6": dist_pb6,
        "schedule_matches_pb6_column": matches_pb6,
        "implied_rounds_by_prompt": rounds,
        "implied_drafting_rounds_by_prompt": drafting_rounds,
        "decode_only_seconds_by_prompt": decode_ours,
        "decode_only_seconds_anchor_by_prompt": decode_anchor,
        "anchor_id8": ANCHOR_ID8,
        "implied_state_step_us_by_prompt": step_us,
        "implied_state_steps_by_prompt": steps,
        "schedule_comparable_prompts": comparable,
        "implied_state_step_us": statistics.fmean([step_us[p] for p in pool]),
        "implied_state_step_us_median": statistics.median(
            [step_us[p] for p in pool]),
        "implied_state_steps": statistics.fmean(usable),
        "implied_state_steps_median": statistics.median(usable),
        "implied_state_steps_all_prompts": statistics.fmean(all_steps),
        "state_steps_integer": float(all(integer_like)),
        "state_steps_integer_fraction": sum(integer_like) / len(integer_like),
        # The residual is set by the three-decimal accepted-draft rates, so the
        # tolerance is loose enough to absorb that rounding and nothing more.
        "implied_rounds_are_integers": float(all(
            abs(rounds[p] - round(rounds[p])) <= 0.05 for p in PROMPTS)),
        "implied_rounds_max_integer_residual": max(
            abs(rounds[p] - round(rounds[p])) for p in PROMPTS),
        "corrected_decode_seconds_by_prompt": corrected,
        "corrected_decode_seconds_total": sum(corrected.values()),
        "decode_only_seconds_total": sum(decode_ours.values()),
        "decode_only_seconds_total_anchor": sum(decode_anchor.values()),
        "state_step_constant_us": STATE_STEP_US,
        "state_step_constant_sd_us": STATE_STEP_SD_US,
        "prefill_mean": mean,
        "prefill_sd": sd,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("submission %s  status=%s  score=%s" % (out["submission_id8"],
                                                  out["status"], out["score"]))
    print("prefill mean %.9f  sd %.9f  cv %.4f%%" % (mean, sd, 100 * sd / mean))
    print("e147_ranked_prefill_pct %+.4f%%  (bar %.9f)"
          % (ranked_prefill_pct, RANKED_PREFILL_SPT_BAR))
    print("vs band low %+.4f%%  vs band high %+.4f%%"
          % (100 * (mean / RANKED_BAND_LOW - 1), 100 * (mean / RANKED_BAND_HIGH - 1)))
    print("forecast %+.4f pp  realised model_b %+.4f pp  model_a %+.4f pp"
          % (forecast, realised["model_b"], realised["model_a"]))
    print("forecast error %+.4f pp" % (realised["model_b"] - forecast))
    print("schedule: L1 to ship %.4f  to pb6 %.4f  matches_pb6 %.0f"
          % (dist_ship, dist_pb6, matches_pb6))
    print("plutarch dlen %.4f  non-drafting rounds %s"
          % (dlen["plutarch"], ndr["plutarch"]))
    print("comparable prompts %s" % ",".join(comparable) or "(none)")
    print("implied state step %.1f us  steps %.3f (median %.3f)  integer %.0f"
          % (out["implied_state_step_us"], out["implied_state_steps"],
             out["implied_state_steps_median"], out["state_steps_integer"]))
    print("implied rounds are integers %.0f" % out["implied_rounds_are_integers"])
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
