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
import fractions
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

# Feedback 5, Finding 237. `f7d59543` is a zero-delta resample of `eb5eadc`,
# and `eb5eadc7` is the promotedSourceRef of the bar `684821ed`. Same tree,
# zero editable bytes changed, two independent ranked sessions, so the
# per-prompt difference below has a ground truth of exactly zero. That makes
# it a direct measurement of the noise on the statistic this rung reports,
# and it replaces the modelled 0.0634 pp floor from feedback 4.
ZERO_DELTA_PAIR = {
    #            684821ed       f7d59543
    "beagle":   (0.001027957, 0.001027147),
    "botany":   (0.001029482, 0.001030359),
    "drama":    (0.001029822, 0.001028453),
    "essays":   (0.001028957, 0.001029586),
    "medicine": (0.001026861, 0.001030240),
    "plutarch": (0.001028652, 0.001028498),
    "republic": (0.001026973, 0.001027098),
    "travel":   (0.001027623, 0.001028244),
}
RESAMPLE_ID8 = "f7d59543"
RESAMPLE_MEDIAN = 3.69864607884415
PREFILL_NOISE_SE_PCT = 0.0490
PREFILL_NOISE_SD_PCT = 0.1385


def zero_delta_anchors(prefill: dict) -> dict:
    """Feedback 5. Price the measured prefill against both halves of the pair.

    The two anchors differ by a known-zero effect, so the gap between the two
    reported percentages is not an independent replication: for a fixed
    candidate mean C it is exactly 100*C*(B-A)/(A*B), around 0.038 pp. What it
    does buy is an anchor-choice sensitivity bound. The genuinely new test is
    per-prompt: the pair gives a measured per-prompt sd, so a candidate whose
    per-prompt effects scatter far wider than that sd is not explained by
    session noise alone.
    """
    a = {p: v[0] for p, v in ZERO_DELTA_PAIR.items()}
    b = {p: v[1] for p, v in ZERO_DELTA_PAIR.items()}
    mean_a = statistics.fmean(a[p] for p in PROMPTS)
    mean_b = statistics.fmean(b[p] for p in PROMPTS)
    if abs(mean_a - RANKED_PREFILL_SPT_BAR) > 5e-10:
        raise SystemExit(
            "pair column A mean %.12f does not reproduce the bar %.9f"
            % (mean_a, RANKED_PREFILL_SPT_BAR))

    mean = statistics.fmean(prefill[p] for p in PROMPTS)
    pct_a = 100.0 * (mean / mean_a - 1.0)
    pct_b = 100.0 * (mean / mean_b - 1.0)

    per_prompt_a = {p: 100.0 * (prefill[p] / a[p] - 1.0) for p in PROMPTS}
    per_prompt_b = {p: 100.0 * (prefill[p] / b[p] - 1.0) for p in PROMPTS}
    residual = {p: per_prompt_a[p] - pct_a for p in PROMPTS}
    effect_sd = statistics.stdev(per_prompt_a[p] for p in PROMPTS)
    worst = max(PROMPTS, key=lambda p: abs(residual[p]))

    pair_pct = {p: 100.0 * (b[p] / a[p] - 1.0) for p in PROMPTS}
    return {
        "e147_prefill_pct_vs_684821ed": pct_a,
        "e147_prefill_pct_vs_f7d59543": pct_b,
        "e147_prefill_anchor_spread_pp": abs(pct_a - pct_b),
        "e147_prefill_anchor_agreement_ok": float(
            abs(pct_a - pct_b) <= PREFILL_NOISE_SD_PCT),
        "e147_prefill_effect_z_vs_pair_se": pct_a / PREFILL_NOISE_SE_PCT,
        "e147_prefill_effect_sd_pct": effect_sd,
        "e147_prefill_effect_dispersion_ratio": effect_sd / PREFILL_NOISE_SD_PCT,
        "e147_prefill_worst_residual_prompt": worst,
        "e147_prefill_worst_residual_pct": residual[worst],
        "e147_prefill_medicine_is_worst_residual": float(worst == "medicine"),
        "e147_prefill_medicine_residual_pct": residual["medicine"],
        "prefill_pct_by_prompt_vs_684821ed": per_prompt_a,
        "prefill_pct_by_prompt_vs_f7d59543": per_prompt_b,
        "prefill_residual_by_prompt": residual,
        "zero_delta_pair_pct_by_prompt": pair_pct,
        "zero_delta_pair_mean_pct": 100.0 * (mean_b / mean_a - 1.0),
        "prefill_noise_se_pct": PREFILL_NOISE_SE_PCT,
        "prefill_noise_sd_pct": PREFILL_NOISE_SD_PCT,
        "resample_id8": RESAMPLE_ID8,
        "resample_published_median": RESAMPLE_MEDIAN,
    }


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

    anchors = zero_delta_anchors(prefill)

    # Feedback 4, D. Where does each prompt sit against the modern census band?
    mtp = {p: ours[p]["mtp_seconds_per_token_mean"] for p in PROMPTS}
    serial = {p: ours[p]["serial_seconds_per_token_mean"] for p in PROMPTS}
    cluster = {p: ("below" if prefill[p] < RANKED_BAND_LOW
                   else "above" if prefill[p] > RANKED_BAND_HIGH
                   else "in_band") for p in PROMPTS}
    n_below = sum(1 for p in PROMPTS if cluster[p] == "below")

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

    # The receipt also carries the round count exactly, without the accept
    # rates. `effective_mean_draft_len` is drafts over rounds, so the reduced
    # denominator divides the round count. The closed form then selects which
    # multiple, and the two are independent evidence.
    exact_rounds, rounds_error = {}, {}
    for p in PROMPTS:
        q = fractions.Fraction(dlen[p]).limit_denominator(DECODE_TOKENS).denominator
        exact_rounds[p] = max(1, round(rounds[p] / q)) * q
        rounds_error[p] = rounds[p] - exact_rounds[p]
    drafting_rounds = {p: float(exact_rounds[p] - ndr[p]) for p in PROMPTS}

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
        "exact_rounds_by_prompt": exact_rounds,
        "rounds_closed_form_error_by_prompt": rounds_error,
        "rounds_closed_form_max_error": max(abs(v) for v in rounds_error.values()),
        "rounds_closed_form_reproduces_receipt": float(
            max(abs(v) for v in rounds_error.values()) < 0.5),
        "corrected_decode_seconds_by_prompt": corrected,
        "corrected_decode_seconds_total": sum(corrected.values()),
        "decode_only_seconds_total": sum(decode_ours.values()),
        "decode_only_seconds_total_anchor": sum(decode_anchor.values()),
        "state_step_constant_us": STATE_STEP_US,
        "state_step_constant_sd_us": STATE_STEP_SD_US,
        "prefill_mean": mean,
        "prefill_sd": sd,
        "mtp_seconds_per_token_mean_by_prompt": mtp,
        "serial_seconds_per_token_mean_by_prompt": serial,
        "prefill_band_low": RANKED_BAND_LOW,
        "prefill_band_high": RANKED_BAND_HIGH,
        "prefill_band_cluster_by_prompt": cluster,
        "e147_prefill_prompts_below_band": float(n_below),
        "e147_ranked_prefill_pct": ranked_prefill_pct,
    }
    out.update(anchors)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("submission %s  status=%s  score=%s" % (out["submission_id8"],
                                                  out["status"], out["score"]))
    print("prefill mean %.9f  sd %.9f  cv %.4f%%" % (mean, sd, 100 * sd / mean))
    print("e147_ranked_prefill_pct %+.4f%%  (bar %.9f)"
          % (ranked_prefill_pct, RANKED_PREFILL_SPT_BAR))
    print("vs band low %+.4f%%  vs band high %+.4f%%"
          % (100 * (mean / RANKED_BAND_LOW - 1), 100 * (mean / RANKED_BAND_HIGH - 1)))
    print("vs %s %+.4f%%   vs %s %+.4f%%   spread %.4f pp   z %.1f"
          % (BAR_ID8, anchors["e147_prefill_pct_vs_684821ed"],
             RESAMPLE_ID8, anchors["e147_prefill_pct_vs_f7d59543"],
             anchors["e147_prefill_anchor_spread_pp"],
             anchors["e147_prefill_effect_z_vs_pair_se"]))
    print("per-prompt effect sd %.4f%%  pair sd %.4f%%  ratio %.2f"
          % (anchors["e147_prefill_effect_sd_pct"], PREFILL_NOISE_SD_PCT,
             anchors["e147_prefill_effect_dispersion_ratio"]))
    print("worst residual %s %+.4f%%   medicine %+.4f%%"
          % (anchors["e147_prefill_worst_residual_prompt"],
             anchors["e147_prefill_worst_residual_pct"],
             anchors["e147_prefill_medicine_residual_pct"]))
    print("%-9s %-14s %-9s %-9s %-14s" %
          ("prompt", "prefill", "d%_vs_bar", "band", "mtp_spt"))
    for p in PROMPTS:
        print("%-9s %.9f  %+8.4f  %-9s %.9f"
              % (p, prefill[p], anchors["prefill_pct_by_prompt_vs_684821ed"][p],
                 cluster[p], mtp[p]))
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
    print("closed-form rounds reproduce the receipt %.0f, worst error %.3f rounds"
          % (out["rounds_closed_form_reproduces_receipt"],
             out["rounds_closed_form_max_error"]))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
