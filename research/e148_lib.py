"""E148 board-repricing instrument: six-prompt cohorts and the step corrector.

harness=ranked for every quantity in this file. Every input is a published
per-prompt field of a ranked receipt. Nothing local enters here.

WHAT THIS ADDS TO E146. E146 built the corrector and validated it on one
cohort anchored at the bar. E148 runs it over the whole board, so three things
change.

1. THE COHORT KEY IS SIX PROMPTS, NOT EIGHT. FINDING 234: the eight-prompt
   digit-identical schedule test splits families that differ only on `plutarch`
   and `travel`. Both carry exactly zero median weight, so a difference there
   cannot move the published score and must not split a cohort. Cohorting on
   `beagle, botany, drama, essays, medicine, republic` merges those families.
   Measured on this board it moves the two largest cohorts from 268/204 to
   269/233 rows.

2. THE FIT RUNS ON THE SAME SIX PROMPTS. If a row is admitted to a cohort while
   its `plutarch` schedule differs, then including `plutarch` in the fit mixes a
   round-count change into a per-round cost estimate. The response vector and
   the basis are therefore both restricted to the cohort prompts.

3. THE STEP IS QUANTIZED, NOT DIVIDED OUT. The state adds an integer number of
   879.0 us steps per drafting round, so the correction subtracts
   `round(k / 879.0)` steps and leaves the remainder as the mechanism. This is
   the whole point: dividing k out continuously would zero every per-round
   mechanism, which is exactly the signal being mined. A mechanism worth the
   H148 target of -0.30 % of the total leg is about -155 us per drafting round
   by Rule 134, comfortably inside the +/-439.5 us rounding cell.

THE ANCHOR DOES NOT HAVE TO BE FLAT. `k` is a contrast, so the quantized step
count is the RELATIVE state `s_row - s_anchor`. A negative step count is legal
and means the anchor drew more state than the row. The correction stays valid
either way, which is why no cohort floor has to be trusted to price a row.

FRAMES. Rule 131: isolate in the decode frame, price in the total leg frame.
`k` is fitted on `mtp_seconds_per_token_mean` with the charged seed prefill
removed (F227), because a per-drafting-round cost cannot live in a prefill that
runs no drafting rounds. Every reported effect is then converted to the total
timed leg, which is the frame the ranked score is computed in.
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import e146_lib as L

# FINDING 234. `plutarch` and `travel` carry zero median weight on this board,
# so a schedule difference confined to them cannot move the published score.
COHORT_PROMPTS = ["beagle", "botany", "drama", "essays", "medicine", "republic"]

# FINDING 237, REPRODUCED HERE INDEPENDENTLY. The advisor reports that `drama`
# belongs with `plutarch` and `travel`, not with the paying prompts. The census
# in `median_weight_census` confirms it on this board: measured over all 880
# scored rows the realized median weight is beagle 0.4903, medicine 0.1994,
# essays 0.1500, republic 0.0983, botany 0.0528, travel 0.0045, plutarch
# 0.0034, drama 0.0011. F234 argued that a prompt which cannot move the median
# must not enter the price; the same argument applies to `drama` and F234 got
# that wrong. Pricing therefore runs on WEIGHTED_FIVE.
WEIGHTED_FIVE = ["beagle", "botany", "essays", "medicine", "republic"]
ZERO_WEIGHT_PROMPTS = ["drama", "plutarch", "travel"]
PRICE_PROMPTS = WEIGHTED_FIVE

# These two rows are on the board with status `failed` and carry no
# `officialMetrics`, so they publish no per-prompt receipt and cannot be
# cohorted, fitted, or priced. `load_rows` drops them by construction; the
# assertion keeps that true if the loader ever changes.
EXCLUDED_ROWS = ("3d75f016", "04c79081")

# E146 headline constants, harness=ranked.
STEP_US = 879.0
STEP_SD_US = 54.3

# A row whose k sits this close to a half-step boundary cannot be assigned an
# integer step count from the fit alone.
AMBIGUOUS_STEP_FRACTION = 0.35

# F228: the state does not touch the prefill, so `prefill_seconds_per_token` is
# a state-free channel. The modern band and its noise floor come from E146.
PREFILL_BAND = (0.0010277, 0.0010349)
PREFILL_NOISE_FLOOR_PP = 0.0634

MIN_COHORT = 4


def median_pair(row):
    """The two prompts whose raw ratios are the 4th and 5th of the eight.

    The published score is the mean of those two ratios, so for any change
    small enough to leave the ordering intact these are the ONLY two prompts
    with a nonzero derivative. This is the exact marginal frame of the score.
    Every other prompt has weight zero for that row, not small weight.
    """
    ranked = sorted((row.raw_ratio(p), p) for p in L.PROMPT_ORDER)
    return [ranked[3][1], ranked[4][1]]


# Rule 129: the upper median slot is a minimum over these four, and the tie
# flips between ranked sessions. A gain uniform across all four therefore moves
# whichever one is currently the minimum, so it captures the full upper slot.
UPPER_FOUR = ["essays", "republic", "medicine", "botany"]
LOWER_SLOT = "beagle"


def uniform_four_price(pair_pct, per_prompt_pct):
    """Price a change that is uniform across the four upper-slot candidates.

    `median_pair` prices this row's realized draw: only the two middle prompts
    have a nonzero derivative. That is right for a prompt-specific gain, but it
    understates a gain that moves all four upper candidates together, because
    then the minimum moves with them whichever one it happens to be.
    """
    lower = per_prompt_pct[LOWER_SLOT]
    upper = sum(per_prompt_pct[p] for p in UPPER_FOUR) / len(UPPER_FOUR)
    return 0.5 * lower + 0.5 * upper


def lower_slot_fraction(rows):
    """Fraction of rows whose lower median slot is `beagle`."""
    hits = sum(1 for r in rows if LOWER_SLOT in median_pair(r))
    return hits / float(len(rows) or 1)


def median_weight_census(rows):
    """Realized median weight of each prompt, averaged over `rows`.

    Each row gives 0.5 to each of its two middle prompts. The result is the
    fraction of published median that each prompt actually carries.
    """
    weight = {p: 0.0 for p in L.PROMPT_ORDER}
    for row in rows:
        for p in median_pair(row):
            weight[p] += 0.5
    n = float(len(rows)) or 1.0
    return {p: weight[p] / n for p in L.PROMPT_ORDER}


def cohort_key(row):
    """The six-prompt schedule fingerprint, exact published digits.

    Both `effective_mean_draft_len` and `non_drafting_round_count` enter the
    key. The draft length alone does not pin the schedule: two trees can agree
    on the mean proposal width and still differ on how often they decline to
    draft at all, and the round-count basis depends on both.
    """
    return tuple((repr(row.dlen(p)), repr(row.non_drafting(p)))
                 for p in COHORT_PROMPTS)


def build_cohorts(rows, min_size=MIN_COHORT):
    """Group scored rows by the six-prompt schedule fingerprint."""
    groups = {}
    for row in rows:
        groups.setdefault(cohort_key(row), []).append(row)
    for members in groups.values():
        members.sort(key=lambda r: r.created)
    classified = {k: v for k, v in groups.items() if len(v) >= min_size}
    return groups, classified


def pct_diff(anchor, row, field="decode", prompts=COHORT_PROMPTS):
    return {p: 100.0 * (L.leg(row, p, field) / L.leg(anchor, p, field) - 1.0)
            for p in prompts}


def design(anchor, counts_row, field="decode", prompts=COHORT_PROMPTS):
    """Percent of `anchor`'s leg moved per microsecond of per-drafting-round cost.

    The denominator is the ANCHOR's time. Using the row's own time shrinks the
    basis exactly where the row is slow and inflates the fitted k by that same
    factor, which is the error E146 documents in `design_matrix`.
    """
    out = {}
    for p in prompts:
        rounds = L.round_count(p, counts_row.dlen(p))
        drafting = rounds - counts_row.non_drafting(p)
        out[p] = drafting / (L.DECODE_TOKENS * L.leg(anchor, p, field)) * 1e-6 * 100.0
    return out


def fit_k(anchor, row, field="decode", prompts=COHORT_PROMPTS):
    """Through-origin least squares of per-prompt percent on the round basis."""
    basis = design(anchor, anchor, field, prompts)
    diffs = pct_diff(anchor, row, field, prompts)
    num = sum(basis[p] * diffs[p] for p in prompts)
    den = sum(basis[p] * basis[p] for p in prompts)
    k = num / den if den else float("nan")
    resid = [diffs[p] - k * basis[p] for p in prompts]
    ss_res = sum(v * v for v in resid)
    ss_tot = sum(diffs[p] * diffs[p] for p in prompts)
    return {
        "k_us_per_drafting_round": k,
        "r2_through_origin": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "residual_sd_pp": math.sqrt(ss_res / len(prompts)),
        "mean_pct": sum(diffs.values()) / len(prompts),
        "observed_pct": diffs,
        "residual_pct": {p: diffs[p] - k * basis[p] for p in prompts},
        "field": field,
        "prompts": list(prompts),
    }


def state_pct(anchor, step_us, field="total", prompts=PRICE_PROMPTS):
    """Percent of `anchor`'s leg that `step_us` per drafting round buys."""
    basis = design(anchor, anchor, field, prompts)
    return step_us * sum(basis.values()) / len(prompts)


def matched_prompts(anchor, row):
    """Prompts where anchor and row publish a digit-identical schedule.

    On these prompts the two receipts ran the same number of rounds and made
    the same proposal widths, so only time can differ. That is the exact
    condition under which a per-round cost is identified. Prompts outside this
    set carry a round-count change and must not enter a per-round fit.
    """
    out = []
    for p in L.PROMPT_ORDER:
        if (repr(anchor.dlen(p)) == repr(row.dlen(p))
                and repr(anchor.non_drafting(p)) == repr(row.non_drafting(p))):
            out.append(p)
    return out


def correct(anchor, row, step_us=STEP_US, fit_prompts=COHORT_PROMPTS,
            price_prompts=PRICE_PROMPTS):
    """State-corrected candidate-leg effect of `row` against `anchor`.

    ESTIMATION AND PRICING USE DIFFERENT PROMPT SETS, ON PURPOSE.

    `fit_prompts` estimates the per-drafting-round cost. It should be every
    prompt whose schedule is digit-identical between the two receipts, because
    each such prompt is an independent observation of the same scalar and
    throwing one away only adds variance. `plutarch` in particular carries a
    near-zero drafting-round basis, which makes it the one observation that can
    separate a flat offset from a per-round cost.

    `price_prompts` converts the answer into score-relevant percent. F237: the
    realized median weight of `drama`, `plutarch` and `travel` is below 0.5 %
    each, so pricing over eight, or over the six of F234, charges the answer
    with prompts that cannot pay and imports their noise for nothing.
    """
    decode = fit_k(anchor, row, "decode", fit_prompts)
    total = fit_k(anchor, row, "total", price_prompts)
    k = decode["k_us_per_drafting_round"]
    exact = k / step_us
    steps = int(round(exact))
    distance = abs(exact - steps)
    one_step_total_pct = state_pct(anchor, step_us, "total", price_prompts)
    raw_total = total["mean_pct"]
    corrected = raw_total - steps * one_step_total_pct
    out = {
        "anchor": anchor.id8,
        "row": row.id8,
        "k_us_per_drafting_round": k,
        "k_residual_us": k - steps * step_us,
        "steps_exact": exact,
        "steps": steps,
        "step_distance": distance,
        "ambiguous_step": distance >= AMBIGUOUS_STEP_FRACTION,
        "one_step_total_pct": one_step_total_pct,
        "raw_total_pct": raw_total,
        "raw_decode_pct": fit_k(anchor, row, "decode", price_prompts)["mean_pct"],
        "corrected_total_pct": corrected,
        "residual_sd_pp": decode["residual_sd_pp"],
        "fit_r2": decode["r2_through_origin"],
        "same_cohort": cohort_key(anchor) == cohort_key(row),
        "n_fit_prompts": len(fit_prompts),
        "fit_prompts": list(fit_prompts),
        "price_prompts": list(price_prompts),
    }
    signed = total["observed_pct"]
    out["price_sd_pp"] = sd(list(signed.values())) if len(signed) > 1 else float("nan")
    out["price_se_pp"] = (out["price_sd_pp"] / math.sqrt(len(signed))
                          if len(signed) > 1 else float("nan"))
    pair = median_pair(anchor)
    pair_pct = pct_diff(anchor, row, "total", pair)
    out["median_pair"] = pair
    out["median_pair_pct"] = sum(pair_pct.values()) / len(pair)
    out["median_pair_corrected_pct"] = (
        out["median_pair_pct"] - steps * state_pct(anchor, step_us, "total", pair))
    slots = [LOWER_SLOT] + UPPER_FOUR
    slot_pct = pct_diff(anchor, row, "total", slots)
    slot_basis = design(anchor, anchor, "total", slots)
    slot_state = {p: step_us * slot_basis[p] for p in slots}
    out["uniform_four_pct"] = uniform_four_price(pair_pct, slot_pct)
    out["uniform_four_corrected_pct"] = (
        out["uniform_four_pct"] - steps * uniform_four_price(pair_pct, slot_state))
    if anchor.has_prefill() and row.has_prefill():
        pre = pct_diff(anchor, row, "prefill", price_prompts)
        out["prefill_pct"] = sum(pre.values()) / len(price_prompts)
    return out


def correct_auto(anchor, row, step_us=STEP_US, price_prompts=PRICE_PROMPTS):
    """`correct` with the fit prompt set chosen by schedule identity."""
    fit_set = matched_prompts(anchor, row) or COHORT_PROMPTS
    return correct(anchor, row, step_us, fit_set, price_prompts)


def leg_stats(anchor, row, field, prompts):
    """Mean, sd, se and largest absolute per-prompt percent on one leg."""
    diffs = pct_diff(anchor, row, field, prompts)
    values = list(diffs.values())
    n = len(values)
    dev = sd(values) if n > 1 else float("nan")
    return {
        "field": field, "prompts": list(prompts), "n": n,
        "per_prompt_pct": diffs,
        "mean_pct": sum(values) / n,
        "sd_pp": dev,
        "se_pp": dev / math.sqrt(n) if n > 1 else float("nan"),
        "abs_max_pp": max(abs(v) for v in values),
        "abs_max_prompt": max(diffs, key=lambda p: abs(diffs[p])),
    }


def paired_control_step(anchor, high, low, fit_prompts=None):
    """Read one state step straight out of two declared-null draws.

    `high` and `low` are both declared zero-mechanism draws of the same
    `anchor`, so the mechanism term is zero in each. Whatever separates their
    per-drafting-round costs is therefore state and nothing else. The
    difference of the two fitted k values is one state step in microseconds,
    read without a classifier, without a cohort floor, and without assuming
    STEP_US. It is a direct falsification test of the 879.0 us constant.
    """
    fit_hi = fit_prompts or matched_prompts(anchor, high) or COHORT_PROMPTS
    fit_lo = fit_prompts or matched_prompts(anchor, low) or COHORT_PROMPTS
    k_hi = fit_k(anchor, high, "decode", fit_hi)
    k_lo = fit_k(anchor, low, "decode", fit_lo)
    delta = k_hi["k_us_per_drafting_round"] - k_lo["k_us_per_drafting_round"]
    return {
        "anchor": anchor.id8, "high": high.id8, "low": low.id8,
        "k_high_us": k_hi["k_us_per_drafting_round"],
        "k_low_us": k_lo["k_us_per_drafting_round"],
        "state_step_us": delta,
        "ratio_to_e146_step": delta / STEP_US,
        "residual_sd_high_pp": k_hi["residual_sd_pp"],
        "residual_sd_low_pp": k_lo["residual_sd_pp"],
        "fit_prompts_high": fit_hi, "fit_prompts_low": fit_lo,
    }


def decision_fingerprint(row):
    """The eight published `accepted_pair_count` values, exact digits.

    THIS FINGERPRINT IS VACUOUS AND E146 SHOULD NOT HAVE RELIED ON IT.
    Measured over every prompt slot of all 880 scored rows on this board,
    `accepted_pair_count` takes exactly one value: 1, in 7040 of 7040 slots.
    It is `PAIRS_PER_PROMPT`, the number of thermally gated pairs the runner
    measured, not a count of accepted drafts. E146's claim that "all 424 prompt
    slots match exactly over the 57 declared pure-nuisance pairs" is therefore
    true by construction and carries no information about whether two rows made
    the same drafting decisions. The function is kept so the check is auditable
    and so no later experiment revives the claim.
    """
    return tuple(repr(row.prompts[p]["accepted_pair_count"])
                 for p in L.PROMPT_ORDER)


def same_schedule(anchor, row):
    """Digit-identical schedule on all eight prompts.

    This replaces the vacuous decision fingerprint. `effective_mean_draft_len`
    and `non_drafting_round_count` do vary across rows, so schedule identity is
    a real constraint, and it is the one that makes the round-count basis
    transfer from anchor to row.
    """
    return len(matched_prompts(anchor, row)) == len(L.PROMPT_ORDER)


# ---------------------------------------------------------------------------
# parent attribution
# ---------------------------------------------------------------------------

HEX_TOKEN = re.compile(r"\b([0-9a-f]{7,40})\b")


def note_title_text(row):
    """The row's first Markdown heading, which is where solvers state the claim."""
    for line in (row.note or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
    return ""


def tree_index(rows):
    """`promotedSourceRef` prefix -> the row it names.

    Measured on this board: `promotedSourceRef` is present on the 82 promoted
    rows only, and it never equals another row's `submissionCommitSha`
    (0/81 matches). It is the ref of the row's OWN promoted source tree, which
    is the name later solvers quote when they say which crown they forked. That
    makes it a usable parent key even though the field carries no parent
    pointer of its own.
    """
    index = {}
    for row in rows:
        ref = getattr(row, "source_ref", None)
        if ref:
            index[ref[:7]] = row
    return index


def cited_parents(row, trees, by_id):
    """Rows this row's public note names, newest promoted tree first."""
    note = row.note or ""
    hits = {}
    for token in set(HEX_TOKEN.findall(note)):
        named = trees.get(token[:7])
        if named is not None and named.id8 != row.id8:
            hits.setdefault(named.id8, named)
        named = by_id.get(token[:8])
        if named is not None and named.id8 != row.id8:
            hits.setdefault(named.id8, named)
    out = [r for r in hits.values() if r.created < row.created]
    out.sort(key=lambda r: (r.score or 0.0), reverse=True)
    return out


def resolve_parent(row, trees, by_id, cohorts_of):
    """Pick the anchor for `row`: its own named parent, in its own cohort.

    Preference order, and the reason for each step:
      1. a cited row in the SAME six-prompt cohort - the fit is then clean,
         because only time can differ between the two receipts;
      2. the highest-scoring cited row in any cohort - usable, but the round
         counts move, so the call is marked cross-schedule;
      3. nothing - the row cannot be priced against a parent.
    """
    cited = cited_parents(row, trees, by_id)
    same = [r for r in cited if cohort_key(r) == cohort_key(row)]
    if same:
        return same[0], "cited-same-cohort"
    if cited:
        return cited[0], "cited-cross-schedule"
    return None, "no-parent"


def load_rows(path=None):
    path, rows = L.load(path)
    present = {r.id8 for r in rows} & set(EXCLUDED_ROWS)
    if present:
        raise AssertionError("unscored rows reached the corpus: %s" % sorted(present))
    return path, rows


def mean(values):
    return L.mean(values)


def sd(values, ddof=1):
    return L.sd(values, ddof)


def auc(pos, neg):
    """Mann-Whitney AUC, ties at 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def dump(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    print("wrote %s" % path)
