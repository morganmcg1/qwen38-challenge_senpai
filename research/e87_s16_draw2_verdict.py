#!/usr/bin/env python3
"""Price the promoted draw-2 receipt and test the section-8 per-draft cost model.

Draw 2 (`f04b102e`) differs from arm C alone (`cb8aeefb`) by section 8 only, so
the per-prompt candidate delta between them is a direct ranked measurement of a
fixed-latency per-draft saving.  That is the quantity my local pricing converted
with the 0.236 head-byte transfer factor, and this script checks that conversion
against the board.

Model, fitted over the 8 prompts:

    delta_p = alpha + beta * drafts_per_round_p / round_time_p

`alpha` absorbs a uniform machine-speed offset between the two runs and `beta`
is the saving in microseconds per draft on the ranked host.  `plutarch` carries
almost no drafts, so it pins `alpha` rather than `beta`.

    python3 research/e87_s16_draw2_verdict.py
"""

import json

from e87_s15_level_slope import LOCAL_ACCEPT, per_prompt

CACHE = "/tmp/yukon-board/full.json"
ARM_C = "cb8aeefb"
DRAW2 = "f04b102e"

# Section 8 as measured locally, and the local round it was measured against.
S8_US_PER_DRAFT = 12.84
LOCAL_ROUND_US = 203640.6
LOCAL_DRAFTS_PER_ROUND = 6.358974358974359
BYTE_TRANSFER_FACTOR = 0.236
CHAIN_US_PER_DRAFT = 113.78
FLOOR_SERIALFREE = 0.160


def load():
    return [r for r in json.load(open(CACHE)) if isinstance(r, dict)]


def pick(rows, want):
    hit = [r for r in rows if str(r.get("id", "")).startswith(want)]
    if not hit:
        raise SystemExit("no submission starts with %s" % want)
    return hit[0]


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx
    alpha = my - beta * mx
    resid = [y - alpha - beta * x for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return alpha, beta, (s2 / sxx) ** 0.5


def main():
    rows = load()
    a, b = pick(rows, ARM_C), pick(rows, DRAW2)
    ta, tb = per_prompt(a), per_prompt(b)

    print("draw 2 verdict: %s  %s  published %.10f"
          % (b["id"][:8], b.get("status"), b["officialScore"]))
    # The board reports `accepted`/`rejected`; the Yukon CLI shows the winning
    # accepted row as `promoted`.
    good = sorted((r for r in rows
                   if r.get("status") == "accepted" and r.get("officialScore")),
                  key=lambda r: -r["officialScore"])
    print("  accepted board, top 3 of %d:" % len(good))
    for r in good[:3]:
        mark = "  <- draw 2" if r["id"].startswith(DRAW2) else ""
        print("    %-9s %.10f  %s%s"
              % (r["id"][:8], r["officialScore"], r.get("solverName"), mark))

    free = []
    for r in rows:
        if r.get("officialScore") is None:
            continue
        t = per_prompt(r)
        if len(t) != 8:
            continue
        free.append((r["id"][:8], t))
    board = {}
    for _, t in free:
        for k, e in t.items():
            board.setdefault(k, []).append(e["serial_seconds_per_token_mean"])
    means = {k: sum(v) / len(v) for k, v in board.items()}
    scored = []
    for rid, t in free:
        vals = sorted(means[k] / t[k]["mtp_seconds_per_token_mean"] for k in t)
        scored.append(((vals[3] + vals[4]) / 2.0, rid))
    scored.sort(reverse=True)
    pos = [i for i, s in enumerate(scored) if s[1] == b["id"][:8]]
    print("  serial-free frame: %.8f, rank %d of %d (arm C alone was 3.33339197)"
          % (scored[pos[0]][0], pos[0] + 1, len(scored)))

    print()
    print("  per-draft cost model, delta% = alpha + beta * drafts_per_round / round_time")
    xs, ys, names = [], [], []
    for key in ta:
        ea, eb = ta[key], tb[key]
        d = eb["effective_mean_draft_len"]
        spt_a = ea["mtp_seconds_per_token_mean"]
        spt_b = eb["mtp_seconds_per_token_mean"]
        round_us = spt_b * (1.0 + LOCAL_ACCEPT * d) * 1e6
        xs.append(d / round_us)
        ys.append(100.0 * (spt_b / spt_a - 1.0))
        names.append(key)
    alpha, beta, se = ols(xs, ys)
    # delta% = -100 * beta_us * (d / round_us); the fit returns beta in % per
    # (draft/us), so the microsecond saving is -beta/100.
    us_per_draft = -beta / 100.0
    se_us = se / 100.0
    print("    alpha (uniform machine offset) %+.4f %%" % alpha)
    print("    beta  -> %.2f us per draft   se %.2f   t %.2f"
          % (us_per_draft, se_us, us_per_draft / se_us))
    print("    local isolated-chain census upper bound: %.2f us per draft"
          % S8_US_PER_DRAFT)
    print("    ranked / local ratio: %.2fx" % (us_per_draft / S8_US_PER_DRAFT))

    print()
    print("  what section 8 was priced at, and what it delivered")
    local_pct = 100.0 * S8_US_PER_DRAFT * LOCAL_DRAFTS_PER_ROUND / LOCAL_ROUND_US
    priced = local_pct * BYTE_TRANSFER_FACTOR
    measured = -sum(ys) / len(ys)
    print("    local round gain            %.4f %%" % local_pct)
    print("    priced with the 0.236 byte transfer factor  %+.4f %% published" % priced)
    print("    measured ranked candidate mean              %+.4f %%" % measured)
    print("    measured minus the machine offset alpha     %+.4f %%" % (measured + alpha))
    print("    understatement factor                       %.1fx"
          % ((measured + alpha) / priced))

    print()
    print("  serial draw and the counterfactual at board-mean serial")
    raws = sorted((e["serial_seconds_per_token_mean"] / e["mtp_seconds_per_token_mean"],
                   k) for k, e in tb.items())
    print("    median pair %s %.6f and %s %.6f -> %.8f"
          % (raws[3][1], raws[3][0], raws[4][1], raws[4][0],
             (raws[3][0] + raws[4][0]) / 2.0))
    cf = sorted(means[k] / tb[k]["mtp_seconds_per_token_mean"] for k in tb)
    cf_score = (cf[3] + cf[4]) / 2.0
    print("    published                      %.8f" % b["officialScore"])
    print("    at board-mean serial draws     %.8f  (%+.3f %%)"
          % (cf_score, 100.0 * (cf_score / b["officialScore"] - 1.0)))
    print("    crown it beat                  %.8f" % 3.32794960796967)
    print("    margin taken                   %+.8f  (%+.4f %%)"
          % (b["officialScore"] - 3.32794960796967,
             100.0 * (b["officialScore"] / 3.32794960796967 - 1.0)))

    print()
    print("  forward prediction, no fitted parameter")
    print("  the local census fixes 12.84 us/draft; ranked round times are public,")
    print("  so the ranked gain is predicted outright rather than regressed")
    local_round_pct = 100.0 * S8_US_PER_DRAFT * LOCAL_DRAFTS_PER_ROUND / LOCAL_ROUND_US
    pair = [raws[3][1], raws[4][1]]
    preds = []
    for key in pair:
        e = tb[key]
        d = e["effective_mean_draft_len"]
        round_us = e["mtp_seconds_per_token_mean"] * (1.0 + LOCAL_ACCEPT * d) * 1e6
        pct = 100.0 * S8_US_PER_DRAFT * d / round_us
        preds.append(pct)
        print("    %s  d %.3f  round %8.0f us  ->  predicted %+.4f %%"
              % (key, d, round_us, pct))
    pred = sum(preds) / len(preds)
    got = 100.0 * (cf_score / 3.33339197 - 1.0)
    print("    median-pair prediction  %+.4f %%" % pred)
    print("    measured serial-free    %+.4f %%   (3.33339197 -> %.8f)" % (got, cf_score))
    print("    agreement               %.0f %%" % (100.0 * pred / got))
    print("    local round for the same census was %+.4f %%, so the ranked round"
          % local_round_pct)
    print("    being about %.2fx shorter is the whole of the difference"
          % (LOCAL_ROUND_US / (sum(
              tb[k]["mtp_seconds_per_token_mean"]
              * (1.0 + LOCAL_ACCEPT * tb[k]["effective_mean_draft_len"]) * 1e6
              for k in pair) / 2.0)))

    print()
    print("  consequence for the retired selection chain (f18 section 2)")
    for key in pair:
        e = tb[key]
        d = e["effective_mean_draft_len"]
        round_us = e["mtp_seconds_per_token_mean"] * (1.0 + LOCAL_ACCEPT * d) * 1e6
        print("    %s  chain at %.2f us/draft  ->  %+.4f %%"
              % (key, CHAIN_US_PER_DRAFT, 100.0 * CHAIN_US_PER_DRAFT * d / round_us))
    ratio = CHAIN_US_PER_DRAFT / S8_US_PER_DRAFT
    print("    chain is %.1fx section 8 (%.2f vs %.2f us per draft)"
          % (ratio, CHAIN_US_PER_DRAFT, S8_US_PER_DRAFT))
    print("    retired on a priced value of %+.4f %% against a %.3f %% floor"
          % (local_pct * ratio * BYTE_TRANSFER_FACTOR, FLOOR_SERIALFREE))
    print("    measured-rate value would be %+.4f %%" % ((measured + alpha) * ratio))
    print("    -> above the floor by %.1fx"
          % ((measured + alpha) * ratio / FLOOR_SERIALFREE))


if __name__ == "__main__":
    main()
