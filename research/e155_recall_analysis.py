#!/usr/bin/env python3
"""E155 -- draft-head recall audit analysis.

Reads the per-slot JSONL written by the audit instrument
(research/e155-patches/recall-audit.patch) and reports how much of the shipped
proposal head's miss rate is recoverable by an exact readout.

Definitions, all sign conventions stated in words:

  ann            id proposed by the shipped ANN + affine-4 rerank path
  exact_compact  exact argmax over the 98,330 real compact rows
  exact_full     exact argmax over all 248,320 lm_head rows
  target         target verify row argmax, i.e. the truth for that slot

  p_shipped        = P(ann == target)
  p_exact_compact  = P(exact_compact == target)
  p_exact_full     = P(exact_full == target)

  recoverable_index_pp = 100 * (p_exact_compact - p_shipped)
      Positive means an exact compact readout would match the target more
      often than the shipped approximate index does.
  recoverable_vocab_pp = 100 * (p_exact_full - p_exact_compact)
      Positive means the compact vocabulary restriction itself costs recall.
  irreducible_head_error_pp = 100 * (1 - p_exact_full)
      The share no readout change can fix: the head's own hidden row is wrong.

Two slot populations are reported, because they answer different questions.

  marginal     every generated slot. This is the readout-quality question.
  conditional  slots whose shipped prefix matched at every earlier slot. This
               is the chain that feeds E[accepted tokens per round].

usage: research/e155_recall_analysis.py OUT.json AUDIT.jsonl [AUDIT.jsonl ...]
       (audit file name stem before the first '.' is used as the prompt id)
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

# Published percent of raw per acceptance point. harness=ranked.
# F3 supersedes both earlier rows: Edward's within-d contrast measured the
# cost/acceptance coupling directly and it is slightly negative, so the
# co-moving row is dead and the settled figure sits above the old upper bound.
SETTLED_PCT_PER_POINT = 2.6701          # F3, Rule 148 weighted
SETTLED_ABSOLUTE_PER_POINT = 0.0983     # F3, absolute raw units
GAP_TO_CROWN_ABSOLUTE = 0.04632         # 0cf1637e 3.68278758 -> ec24d59 3.72911001
FIXED_DEPTH_PCT_PER_POINT = 2.240       # F1/F2, superseded, kept for comparison
CO_MOVING_PCT_PER_POINT = 1.209         # FINDING 281, retracted by F3

# Verdict thresholds on the recoverable headroom, in acceptance points.
# F1 kept the thresholds fixed in points and moved only the published value
# attached to them, so the repricing in F3 does not move them either.
LARGEST_LEVER_PP = 1.0
BOUNDED_ARM_PP = 0.3

# Ranked per-position conditional acceptance recovered in FINDING 286.
RANKED_ACCEPTANCE = {
    "drama": 0.5280, "travel": 0.6248, "beagle": 0.8973, "republic": 0.9187,
    "essays": 0.9222, "medicine": 0.9299, "botany": 0.9312,
}

# Index geometry read from Qwen35.swift on this base, for the bucket-B price.
HIDDEN = 5_120
COMPACT_PADDED_ROWS = 98_336
FULL_VOCAB_ROWS = 248_320
ROWS_PER_LEAF = 16
PROBE_FRACTION = 0.15
RERANK_CANDIDATES = 32
BYTES_PER_PARAM_2BIT_G64 = 2 / 8 + 2 * 2 / 64      # payload + fp16 scale/bias
BYTES_PER_PARAM_4BIT_G64 = 4 / 8 + 2 * 2 / 64
ROOFLINE_BYTES_PER_SECOND = 567e9                  # FINDING 279, ranked host
RANKED_ROUND_FIXED_US = 25_409.0                   # FINDING 281 intercept
RANKED_ROUND_PER_TOKEN_US = 4_291.0                # retained only for round size


def load(path: str) -> list[dict]:
    """Per-slot rows only. `event` lines are the instrument's own install and
    per-round diagnostics and carry no measurement."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "event" not in row:
                rows.append(row)
    return rows


def rate(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def slot_profile(rows: list[dict], key: str) -> dict[int, tuple[int, int]]:
    """Conditional per-slot counts: (hits, trials) restricted to slots whose
    shipped prefix matched everywhere earlier in the same round."""
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[row["round"]].append(row)
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for _, slots in by_round.items():
        for row in sorted(slots, key=lambda r: r["slot"]):
            counts[row["slot"]][1] += 1
            if row[key] == row["target"]:
                counts[row["slot"]][0] += 1
            if row["ann"] != row["target"]:
                break
    return {slot: (hit, trial) for slot, (hit, trial) in counts.items()}


def chain_rows(rows: list[dict]) -> list[dict]:
    """Slots the shipped chain actually reached: every earlier slot in the same
    round matched, so this slot's outcome enters E[accepted per round]."""
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[row["round"]].append(row)
    alive = []
    for _, slots in by_round.items():
        for row in sorted(slots, key=lambda r: r["slot"]):
            alive.append(row)
            if row["ann"] != row["target"]:
                break
    return alive


def marginal_slot_profile(rows: list[dict], key: str) -> dict[int, tuple[int, int]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["slot"]][1] += 1
        if row[key] == row["target"]:
            counts[row["slot"]][0] += 1
    return {slot: (hit, trial) for slot, (hit, trial) in counts.items()}


def expected_accepted(profile: dict[int, float], depths: list[int]) -> float:
    """E[accepted drafts per round] from a conditional per-slot profile and the
    observed distribution of offered depth."""
    total = 0.0
    for depth in depths:
        survive = 1.0
        for slot in range(depth):
            survive *= profile.get(slot, 0.0)
            total += survive
    return total / len(depths) if depths else float("nan")


def derived_exchange_rate(
    profile: dict[int, float], depths: list[int], bump: float = 0.01
) -> dict[str, float]:
    """Own fixed-depth exchange rate at the measured profile and depth mix.

    Fixed depth means the round's row count M = d + 1 does not move, so the
    round cost is held constant and the published ratio moves with tokens per
    round only.
    """
    base_tokens = 1.0 + expected_accepted(profile, depths)
    bumped = {slot: min(1.0, value + bump) for slot, value in profile.items()}
    bumped_tokens = 1.0 + expected_accepted(bumped, depths)
    return {
        "tokens_per_round_base": base_tokens,
        "tokens_per_round_bumped": bumped_tokens,
        "published_pct_per_acceptance_point": 100.0
        * (bumped_tokens - base_tokens)
        / base_tokens,
    }


def three_bucket_split(rows: list[dict]) -> dict:
    """F3's three buckets, as an exhaustive partition of every audited slot.

    hit          the shipped path already proposes the target.
    A selection  the exact compact readout proposes the target and the shipped
                 ANN plus rerank stage drops it. A selection defect inside the
                 editable surface: no new weights, no wider candidate set.
    B candidate  the target is outside the 98,330 compact ids the head can
                 reach, and the exact full-vocabulary readout proposes it. A
                 larger candidate set catches it, at a byte cost priced below.
    C irreducible  the exact full readout also disagrees with the target. No
                 readout change fixes it; only a different head can.
    lucky        the shipped path proposes the target and the exact compact
                 readout does not. Repairing the index LOSES these slots, so
                 the net index headroom is A minus lucky, not A.
    """
    hit = a_sel = b_cand = c_irr = lucky = 0
    for row in rows:
        if row["ann"] == row["target"]:
            hit += 1
            if row["exact_compact"] != row["target"]:
                lucky += 1
        elif row["exact_compact"] == row["target"]:
            a_sel += 1
        elif row["exact_full"] == row["target"]:
            b_cand += 1
        else:
            c_irr += 1
    total = max(len(rows), 1)
    return {
        "slots": len(rows),
        "hit_shipped": hit,
        "bucket_a_selection_defect": a_sel,
        "bucket_b_candidate_set_trim": b_cand,
        "bucket_c_irreducible_head_disagreement": c_irr,
        "lucky_shipped_win_exact_miss": lucky,
        "bucket_a_pp": 100.0 * a_sel / total,
        "bucket_b_pp": 100.0 * b_cand / total,
        "bucket_c_pp": 100.0 * c_irr / total,
        "lucky_pp": 100.0 * lucky / total,
        "bucket_a_net_of_lucky_pp": 100.0 * (a_sel - lucky) / total,
        "partition_is_exhaustive":
            hit + a_sel + b_cand + c_irr == len(rows),
    }


def head_margin_structure(rows: list[dict]) -> dict:
    """How far the proposal head is from the target when it disagrees.

    margin = full_max - target_logit, both read from the same head logit
    vector. Zero means the head's own exact argmax IS the target. A small
    positive margin is a near tie the head almost got; a large one is real
    disagreement that no readout or calibration change recovers.
    """
    margins = sorted(
        row["full_max"] - row["target_logit"]
        for row in rows
        if row["exact_full"] != row["target"]
    )
    if not margins:
        return {"disagreeing_slots": 0}

    def quantile(q: float) -> float:
        return margins[min(len(margins) - 1, int(q * len(margins)))]

    total = len(margins)
    return {
        "disagreeing_slots": total,
        # An exact tie means a different tie-break would emit the target. It is
        # not a lever on its own: the tie-break is a coin flip unless it is
        # correlated with the target ordering, so its expected value is half
        # this count minus the ties the shipped order already wins.
        "head_logit_exact_ties": sum(1 for m in margins if m == 0.0),
        "margin_min": margins[0],
        "margin_p25": quantile(0.25),
        "margin_median": quantile(0.50),
        "margin_p75": quantile(0.75),
        "margin_max": margins[-1],
        "margin_mean": sum(margins) / total,
        "fraction_within_0p25": sum(1 for m in margins if m <= 0.25) / total,
        "fraction_within_0p50": sum(1 for m in margins if m <= 0.50) / total,
        "fraction_within_1p00": sum(1 for m in margins if m <= 1.00) / total,
        "fraction_within_2p00": sum(1 for m in margins if m <= 2.00) / total,
    }


def bucket_b_price(mean_offered_depth: float, tokens_per_round: float) -> dict:
    """First-order roofline price of untrimming the draft vocabulary.

    The compact head is a 2-bit affine group-64 gather. The shipped index reads
    one 2-bit centroid per leaf in the coarse pass, then the 2-bit rows of the
    probed leaves, then reranks 32 rows against the exact 4-bit head. Growing
    `compactDraftPrefixCount` to the full vocabulary scales the first two terms
    and leaves the third alone. harness=ranked: the roofline and the round
    model both describe the ranked host.
    """

    def bytes_for(rows: int) -> dict:
        leaves = rows // ROWS_PER_LEAF
        probed = math.ceil(PROBE_FRACTION * leaves)
        coarse = leaves * HIDDEN * BYTES_PER_PARAM_2BIT_G64
        refine = probed * ROWS_PER_LEAF * HIDDEN * BYTES_PER_PARAM_2BIT_G64
        rerank = RERANK_CANDIDATES * HIDDEN * BYTES_PER_PARAM_4BIT_G64
        return {
            "rows": rows, "leaves": leaves, "probed_leaves": probed,
            "refined_rows": probed * ROWS_PER_LEAF,
            "coarse_bytes": coarse, "refine_bytes": refine,
            "rerank_bytes": rerank,
            "total_bytes_per_draft_step": coarse + refine + rerank,
        }

    shipped = bytes_for(COMPACT_PADDED_ROWS)
    untrimmed = bytes_for(FULL_VOCAB_ROWS)
    delta_bytes = (
        untrimmed["total_bytes_per_draft_step"]
        - shipped["total_bytes_per_draft_step"]
    )
    delta_us_per_step = 1e6 * delta_bytes / ROOFLINE_BYTES_PER_SECOND
    delta_us_per_round = delta_us_per_step * mean_offered_depth
    round_us = (
        RANKED_ROUND_FIXED_US + RANKED_ROUND_PER_TOKEN_US * tokens_per_round
    )
    return {
        "harness": "ranked",
        "shipped_geometry": shipped,
        "untrimmed_geometry": untrimmed,
        "delta_bytes_per_draft_step": delta_bytes,
        "delta_us_per_draft_step": delta_us_per_step,
        "mean_offered_depth": mean_offered_depth,
        "delta_us_per_round": delta_us_per_round,
        "modelled_round_us": round_us,
        "cost_pct_published": 100.0 * delta_us_per_round / round_us,
    }


def analyse(prompt: str, rows: list[dict]) -> dict:
    total = len(rows)
    shipped = sum(1 for r in rows if r["ann"] == r["target"])
    compact = sum(1 for r in rows if r["exact_compact"] == r["target"])
    full = sum(1 for r in rows if r["exact_full"] == r["target"])
    masked = sum(1 for r in rows if r["masked_compact"] == r["target"])
    index_miss = sum(1 for r in rows if r["ann"] != r["exact_compact"])
    masked_moved = sum(1 for r in rows if r["masked_compact"] != r["exact_compact"])
    outside = sum(
        1
        for r in rows
        if not (r["ann"] < 98_304 or 248_044 <= r["ann"] < 248_070)
    )
    # A shipped miss against the exact compact winner is only a real index
    # failure when the two scores actually differ. Equal scores are a tie the
    # rerank kernel is free to break either way.
    exact_ties = sum(
        1
        for r in rows
        if r["ann"] != r["exact_compact"]
        and abs(r["compact_max"] - r["ann_logit"]) <= 0.0
    )

    # Permutation positive control: pair each slot's shipped id with the NEXT
    # slot's target. A real comparison must collapse toward chance.
    ordered = sorted(rows, key=lambda r: (r["round"], r["slot"]))
    shifted = sum(
        1
        for a, b in zip(ordered, ordered[1:])
        if a["ann"] == b["target"]
    )
    shifted_total = max(len(ordered) - 1, 1)

    p_shipped = rate(shipped, total)
    p_compact = rate(compact, total)
    p_full = rate(full, total)

    depths = []
    seen_rounds = set()
    for row in ordered:
        if row["round"] not in seen_rounds:
            seen_rounds.add(row["round"])
            depths.append(row["d"])

    cond = {
        key: slot_profile(rows, key)
        for key in ("ann", "exact_compact", "exact_full")
    }
    marg = {
        key: marginal_slot_profile(rows, key)
        for key in ("ann", "exact_compact", "exact_full")
    }
    cond_rate = {
        key: {slot: rate(h, t) for slot, (h, t) in value.items()}
        for key, value in cond.items()
    }

    shipped_rate_profile = cond_rate["ann"]
    exchange = derived_exchange_rate(shipped_rate_profile, depths)

    # Same three rates over the conditional population only. This is the
    # population whose per-slot probability enters E[accepted per round], so
    # it is the one the acceptance-point exchange rate is defined on.
    cond_pooled = {}
    for key in ("ann", "exact_compact", "exact_full"):
        hits = sum(hit for hit, _ in cond[key].values())
        trials = sum(trial for _, trial in cond[key].values())
        cond_pooled[key] = (hits, trials)

    # First-order counterfactual: replace the shipped per-slot conditional
    # acceptance with the exact-compact one, holding depth and round cost.
    compact_profile = cond_rate["exact_compact"]
    full_profile = cond_rate["exact_full"]
    base_tokens = 1.0 + expected_accepted(shipped_rate_profile, depths)
    compact_tokens = 1.0 + expected_accepted(compact_profile, depths)
    full_tokens = 1.0 + expected_accepted(full_profile, depths)

    # Greedy 512-token continuations degenerate into repeated phrases, which
    # makes both the retrieval index and the head look easier than they would
    # on diverse text. Publish the concentration so the recall numbers are read
    # with that limit in view.
    target_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        target_counts[row["target"]] += 1
    top_share = max(target_counts.values()) / total if total else float("nan")

    return {
        "prompt": prompt,
        "slots_total": total,
        "rounds": len(seen_rounds),
        "distinct_target_tokens": len(target_counts),
        "distinct_proposed_tokens": len({row["ann"] for row in rows}),
        "most_common_target_share": top_share,
        "mean_offered_depth": sum(depths) / len(depths) if depths else float("nan"),
        "counts": {
            "shipped_hits": shipped,
            "exact_compact_hits": compact,
            "exact_full_hits": full,
            "masked_compact_hits": masked,
            "index_miss": index_miss,
            "index_miss_exact_ties": exact_ties,
            "masked_control_moved": masked_moved,
            "ann_outside_compact_set": outside,
            "permutation_shifted_hits": shifted,
            "permutation_shifted_trials": shifted_total,
        },
        "three_bucket_split_marginal": three_bucket_split(rows),
        "three_bucket_split_conditional": three_bucket_split(chain_rows(rows)),
        "head_margin_when_head_disagrees": head_margin_structure(rows),
        "e155_p_shipped": p_shipped,
        "e155_p_exact_compact": p_compact,
        "e155_p_exact_full": p_full,
        "e155_index_miss_rate": rate(index_miss, total),
        "e155_recoverable_index_pp": 100.0 * (p_compact - p_shipped),
        "e155_recoverable_vocab_pp": 100.0 * (p_full - p_compact),
        "e155_irreducible_head_error_pp": 100.0 * (1.0 - p_full),
        "conditional_population": {
            "slots": cond_pooled["ann"][1],
            "p_shipped": rate(*cond_pooled["ann"]),
            "p_exact_compact": rate(*cond_pooled["exact_compact"]),
            "p_exact_full": rate(*cond_pooled["exact_full"]),
            "recoverable_index_pp": 100.0
            * (rate(*cond_pooled["exact_compact"]) - rate(*cond_pooled["ann"])),
            "recoverable_vocab_pp": 100.0
            * (
                rate(*cond_pooled["exact_full"])
                - rate(*cond_pooled["exact_compact"])
            ),
        },
        "permutation_control": {
            "p_shipped": p_shipped,
            "p_shipped_shifted_by_one_slot": rate(shifted, shifted_total),
            "magnitude_pp": 100.0 * (p_shipped - rate(shifted, shifted_total)),
        },
        "masked_winner_control": {
            "p_exact_compact": p_compact,
            "p_masked_compact": rate(masked, total),
            "magnitude_pp": 100.0 * (p_compact - rate(masked, total)),
            "moved_fraction": rate(masked_moved, total),
        },
        "conditional_slot_table": {
            str(slot + 1): {
                "trials": cond["ann"][slot][1],
                "p_shipped": cond_rate["ann"].get(slot),
                "p_exact_compact": cond_rate["exact_compact"].get(slot),
                "p_exact_full": cond_rate["exact_full"].get(slot),
                "recoverable_index_pp": 100.0
                * (
                    cond_rate["exact_compact"].get(slot, 0.0)
                    - cond_rate["ann"].get(slot, 0.0)
                ),
                "recoverable_vocab_pp": 100.0
                * (
                    cond_rate["exact_full"].get(slot, 0.0)
                    - cond_rate["exact_compact"].get(slot, 0.0)
                ),
            }
            for slot in sorted(cond["ann"])
        },
        "marginal_slot_table": {
            str(slot + 1): {
                "trials": marg["ann"][slot][1],
                "p_shipped": rate(*marg["ann"][slot]),
                "p_exact_compact": rate(*marg["exact_compact"][slot]),
                "p_exact_full": rate(*marg["exact_full"][slot]),
            }
            for slot in sorted(marg["ann"])
        },
        "derived_exchange_rate": exchange,
        "counterfactual_tokens_per_round": {
            "shipped": base_tokens,
            "exact_compact": compact_tokens,
            "exact_full": full_tokens,
            "index_published_pct": 100.0 * (compact_tokens - base_tokens) / base_tokens,
            "vocab_published_pct": 100.0
            * (full_tokens - compact_tokens)
            / base_tokens,
        },
    }


def pool(per_prompt: list[dict], rows: list[dict]) -> dict:
    pooled = analyse("pooled", rows)
    pooled["prompts"] = [entry["prompt"] for entry in per_prompt]
    return pooled


def verdict(recoverable_pp: float) -> str:
    if recoverable_pp >= LARGEST_LEVER_PP:
        return "largest_single_lever"
    if recoverable_pp >= BOUNDED_ARM_PP:
        return "one_bounded_r2_arm"
    return "close_the_index_axis"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out_path = sys.argv[1]
    per_prompt = []
    everything: list[dict] = []
    for path in sys.argv[2:]:
        prompt = os.path.basename(path).split(".")[0]
        rows = load(path)
        if not rows:
            print(f"e155: {path} is empty", file=sys.stderr)
            return 1
        per_prompt.append(analyse(prompt, rows))
        everything.extend(rows)

    # Pooling across prompts must not merge round numbering, so re-key rounds.
    offset = 0
    rekeyed: list[dict] = []
    for entry, path in zip(per_prompt, sys.argv[2:]):
        rows = load(path)
        top = max(r["round"] for r in rows)
        for row in rows:
            copy = dict(row)
            copy["round"] = row["round"] + offset
            rekeyed.append(copy)
        offset += top + 1
    pooled = pool(per_prompt, rekeyed)

    index_pp = pooled["e155_recoverable_index_pp"]
    vocab_pp = pooled["e155_recoverable_vocab_pp"]
    cond_index_pp = pooled["conditional_population"]["recoverable_index_pp"]
    cond_vocab_pp = pooled["conditional_population"]["recoverable_vocab_pp"]
    chain = pooled["counterfactual_tokens_per_round"]
    derived = pooled["derived_exchange_rate"]["published_pct_per_acceptance_point"]
    cond_bucket = pooled["three_bucket_split_conditional"]
    marg_bucket = pooled["three_bucket_split_marginal"]
    report = {
        "harness": "local",
        "per_prompt": per_prompt,
        "pooled": pooled,
        "e155_recoverable_acceptance_points": {
            "definition": "one acceptance point is one percentage point of "
                          "the per-position conditional acceptance probability "
                          "p, which is the quantity F3 prices at "
                          "+2.6701 % of raw",
            "population": "conditional, the chain that feeds "
                          "E[accepted tokens per round]",
            "bucket_a_selection_defect_points": cond_bucket[
                "bucket_a_net_of_lucky_pp"],
            "bucket_b_candidate_set_trim_points": cond_bucket["bucket_b_pp"],
            "bucket_c_irreducible_points": cond_bucket["bucket_c_pp"],
            "total_recoverable_points": cond_bucket["bucket_a_net_of_lucky_pp"]
            + cond_bucket["bucket_b_pp"],
            "marginal_population_cross_check": {
                "bucket_a_points": marg_bucket["bucket_a_net_of_lucky_pp"],
                "bucket_b_points": marg_bucket["bucket_b_pp"],
                "bucket_c_points": marg_bucket["bucket_c_pp"],
            },
        },
        "conversion": {
            "harness": "ranked",
            "settled_pct_per_acceptance_point": SETTLED_PCT_PER_POINT,
            "settled_absolute_per_acceptance_point":
                SETTLED_ABSOLUTE_PER_POINT,
            "gap_to_crown_absolute": GAP_TO_CROWN_ABSOLUTE,
            "bucket_a_pct_published": cond_bucket["bucket_a_net_of_lucky_pp"]
            * SETTLED_PCT_PER_POINT,
            "bucket_b_pct_published": cond_bucket["bucket_b_pp"]
            * SETTLED_PCT_PER_POINT,
            "bucket_a_absolute": cond_bucket["bucket_a_net_of_lucky_pp"]
            * SETTLED_ABSOLUTE_PER_POINT,
            "bucket_b_absolute": cond_bucket["bucket_b_pp"]
            * SETTLED_ABSOLUTE_PER_POINT,
            "total_recoverable_pct_published": (
                cond_bucket["bucket_a_net_of_lucky_pp"]
                + cond_bucket["bucket_b_pp"]
            ) * SETTLED_PCT_PER_POINT,
            "total_recoverable_as_fraction_of_gap": (
                cond_bucket["bucket_a_net_of_lucky_pp"]
                + cond_bucket["bucket_b_pp"]
            ) * SETTLED_ABSOLUTE_PER_POINT / GAP_TO_CROWN_ABSOLUTE,
            "e155_published_pct_per_acceptance_point": derived,
            "e155_published_pct_per_acceptance_point_source": "derived here "
            "from the measured pooled depth and per-slot acceptance profile, "
            "holding depth and round cost fixed",
            "reproduces_advisor_fixed_depth_rate": abs(
                derived - FIXED_DEPTH_PCT_PER_POINT
            ) <= 0.1 * FIXED_DEPTH_PCT_PER_POINT,
            "relative_gap_vs_advisor_fixed_depth": (
                derived - FIXED_DEPTH_PCT_PER_POINT
            ) / FIXED_DEPTH_PCT_PER_POINT,
            "fixed_depth_pct_per_acceptance_point": FIXED_DEPTH_PCT_PER_POINT,
            "co_moving_pct_per_acceptance_point": CO_MOVING_PCT_PER_POINT,
            "e155_recoverable_index_pct_published": index_pp
            * FIXED_DEPTH_PCT_PER_POINT,
            "e155_recoverable_vocab_pct_published": vocab_pp
            * FIXED_DEPTH_PCT_PER_POINT,
            "e155_recoverable_index_pct_published_co_moving": index_pp
            * CO_MOVING_PCT_PER_POINT,
            "e155_recoverable_vocab_pct_published_co_moving": vocab_pp
            * CO_MOVING_PCT_PER_POINT,
            # Robustness check 1: the same conversion on the conditional
            # acceptance population, which is the population the exchange rate
            # is defined on.
            "index_pct_published_conditional_population": cond_index_pp
            * FIXED_DEPTH_PCT_PER_POINT,
            "vocab_pct_published_conditional_population": cond_vocab_pp
            * FIXED_DEPTH_PCT_PER_POINT,
            # Robustness check 2: no linearization at all. Rebuild the accept
            # chain from the exact per-slot profile and take the token ratio
            # directly, holding depth and round cost.
            "index_pct_published_chain_recomputed": chain[
                "index_published_pct"],
            "vocab_pct_published_chain_recomputed": chain[
                "vocab_published_pct"],
        },
        "bucket_b_untrim_price": bucket_b_price(
            pooled["mean_offered_depth"],
            pooled["derived_exchange_rate"]["tokens_per_round_base"]),
        "ranked_acceptance_reference": {
            "harness": "ranked",
            "source": "FINDING 286, per-position conditional acceptance "
                      "recovered from the ranked receipts",
            "rates": RANKED_ACCEPTANCE,
            "local_conditional_measured": {
                entry["prompt"]: entry["conditional_population"]["p_shipped"]
                for entry in per_prompt
            },
        },
        "e155_axis_verdict": verdict(cond_bucket["bucket_a_net_of_lucky_pp"]),
        "e155_vocab_axis_verdict": verdict(cond_bucket["bucket_b_pp"]),
        "e155_warm_verdict": "closed_by_inheritance",
    }
    net = (
        report["conversion"]["bucket_b_pct_published"]
        - report["bucket_b_untrim_price"]["cost_pct_published"]
    )
    report["bucket_b_untrim_price"]["net_pct_published"] = net
    report["bucket_b_untrim_price"]["worth_building"] = net > 0.0
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["conversion"], indent=2))
    print(json.dumps(report["e155_recoverable_acceptance_points"], indent=2))
    print("index verdict:", report["e155_axis_verdict"])
    print("vocab verdict:", report["e155_vocab_axis_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
