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
import os
import sys
from collections import defaultdict

# Fixed-depth published-percent per acceptance point, from the advisor's
# corrected exchange rate. harness=ranked.
FIXED_DEPTH_PCT_PER_POINT = 2.240
CO_MOVING_PCT_PER_POINT = 1.209

# Verdict thresholds on the recoverable index headroom, in published percent.
LARGEST_LEVER_PCT = 1.0
BOUNDED_ARM_PCT = 0.3


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


def verdict(index_pct_published: float) -> str:
    if index_pct_published >= LARGEST_LEVER_PCT:
        return "largest_single_lever"
    if index_pct_published >= BOUNDED_ARM_PCT:
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
    report = {
        "harness": "local",
        "per_prompt": per_prompt,
        "pooled": pooled,
        "conversion": {
            "harness": "ranked",
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
        "e155_axis_verdict": verdict(index_pp * FIXED_DEPTH_PCT_PER_POINT),
        "e155_warm_verdict": "closed_by_inheritance",
    }
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["conversion"], indent=2))
    print("verdict:", report["e155_axis_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
