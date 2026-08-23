#!/usr/bin/env python3
"""E143 R0: the first-divergence channel census.

Only the FIRST divergent position of a round matters. Every draft after it is
discarded, so the whole per-step acceptance deficit is carried by one row per
divergent round. This script finds that row in the E142 capture and splits it
into named channels.

WHAT THE CAPTURE HOLDS, AND WHY NO GPU REPLAY IS NEEDED FOR STAGE A.
`verify/<seed>.json` carries a `row_ledger` with one entry per verify row:

    kind        "draft" for a checked draft row, "targetTail" for the bonus row
    token       the token the row CHECKED, i.e. the head's proposal `d`
    top2_tokens the TARGET's exact top two ids at that row; `top2_tokens[0]`
                is the target's argmax `t*`
    top2_logits the target's exact top two scores
    accepted    `token == top2_tokens[0]` AND no earlier draft in the round
                diverged

So the first divergence of a round is the first `kind == "draft"` row with
`accepted == false`, and both `t*` and `d` are already recorded there.

CHANNELS. The shipped proposal path (Qwen35.swift
`draftTokenIDWithDeclaredRerank`) is:

  1. a COMPACT draft vocabulary of `compactDraftRealCount = 98,330` rows:
     tokenizer ids `[0, 98,304)` plus the control block `[248,044, 248,070)`;
  2. a 2-bit affine COARSE readout over those rows;
  3. a top-`draftRerankCandidateCount` (= 32) window from the coarse scores;
  4. an EXACT affine-4 rerank of the window against the target's own lm_head
     rows, applied to the HEAD's hidden state.

  C-a  `t*` is not a compact row at all. Structurally unproposable. Decided
       here by id alone.
  C-b  `t*` is a compact row whose coarse score fell outside the top-32
       window. Reachable: widen or improve the screen.
  C-c  `t*` was inside the window and the rerank still ordered `d` first.
  C-d  the head's own hidden state prefers `d`.

  Stage A decides C-a exactly and bounds the rest. C-b, C-c and C-d need the
  head's hidden row, which stage A does not have.

  NOTE ON C-c. Step 4 is an EXACT affine-4 readout of the target's lm_head
  rows. If the head's exact full-compact readout ranks `t*` first, and `t*` is
  in the window, then the rerank cannot order `d` first. C-c is therefore
  identically zero for the shipped head, and the only reachable in-vocabulary
  channel is C-b. This is a property of the shipped arithmetic, not an
  assumption, and stage C tests it directly.

STRATIFICATION (Rule 76). The regime variable is per-step acceptance `p`. The
captured public goldens do not sit at the ranked prompts' `p`, so every channel
number is reported inside a stratum with that stratum's own measured `p`, and
the reweighting to the ranked beagle and essays regimes is labelled `derived`.

The model-free per-step estimator used everywhere here is

    p = accepted_draft_total / (accepted_draft_total + divergent_round_count)

Each round is a truncated Bernoulli run: every accepted draft is one success,
and a divergent round contributes exactly one genuine failure. Rows after the
first divergence are off-trajectory and are NOT trials.

Usage:
  research/e143_r0.py --stage a
  research/e143_r0.py --stage b      # adds the offline target-side rank of `d`
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HIDDEN = 5120
VOCAB = 248_320
GROUP = 64

# Qwen35.swift:5432-5437. The compact draft head keeps the low `PREFIX` ids and
# appends the official control block; everything between them is unproposable.
COMPACT_PREFIX_COUNT = 98_304
COMPACT_CONTROL_START = 248_044
COMPACT_CONTROL_END = 248_070
COMPACT_REAL_COUNT = (
    COMPACT_PREFIX_COUNT + COMPACT_CONTROL_END - COMPACT_CONTROL_START)  # 98,330
RERANK_CANDIDATES = 32

# E142 measured the offline-to-device arithmetic gap on these same rows.
ARITHMETIC_GAP = 0.107

# Campaign constants used only for pricing, quoted from the assignment.
MISS_TO_SCORE_PCT = 203.0
CARRIER_WEIGHT = {"beagle": 0.478, "essays": 0.522}
RANKED_P = {"beagle": 0.9341, "essays": 0.9647}

SEED_CARRIER = {
    "beagle_a": "beagle",
    "beagle_f": "beagle",
    "essays_montaigne": "essays",
}

CACHE = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows"


def compact_rank(token: int) -> int:
    """Row of `token` in the compact draft vocabulary, or -1 when absent."""
    if 0 <= token < COMPACT_PREFIX_COUNT:
        return token
    if COMPACT_CONTROL_START <= token < COMPACT_CONTROL_END:
        return COMPACT_PREFIX_COUNT + token - COMPACT_CONTROL_START
    return -1


def load_seed(seed: str) -> dict:
    return json.loads((CACHE / "verify" / f"{seed}.json").read_text())


def first_divergences(seed: str, payload: dict) -> tuple[list[dict], dict]:
    """One record per divergent round, plus that seed's run-level counters."""
    rounds: dict[int, list[dict]] = defaultdict(list)
    for row in payload["row_ledger"]:
        rounds[row["round"]].append(row)

    records: list[dict] = []
    n_rounds = 0
    n_drafting_rounds = 0
    accepted = 0
    for index in sorted(rounds):
        rows = rounds[index]
        drafts = sorted(
            (r for r in rows if r["kind"] == "draft"), key=lambda r: r["draft_index"])
        n_rounds += 1
        if not drafts:
            continue
        n_drafting_rounds += 1
        diverged = next((r for r in drafts if not r["accepted"]), None)
        if diverged is None:
            accepted += len(drafts)
            continue
        j = diverged["draft_index"]
        accepted += j
        t_star = diverged["top2_tokens"][0]
        d = diverged["token"]
        if diverged["reference_token"] != t_star:
            raise SystemExit(
                f"e143-r0: {seed} round {index}: golden token "
                f"{diverged['reference_token']} != target argmax {t_star}")
        if d == t_star:
            raise SystemExit(
                f"e143-r0: {seed} round {index}: first non-accepted row matches "
                "its target argmax")
        s1, s2 = diverged["top2_logits"]
        records.append({
            "seed": seed,
            "carrier": SEED_CARRIER.get(seed, "other"),
            "round": index,
            "width": len(drafts),
            "first_divergence_index": j,
            "t_star": t_star,
            "d": d,
            "s1": s1,
            "s2": s2,
            "margin_top2": s1 - s2,
            "d_is_target_rank2": d == diverged["top2_tokens"][1],
            "t_star_compact_row": compact_rank(t_star),
            "d_compact_row": compact_rank(d),
            # Two structural tests. `strict` is the code's real boundary
            # (Qwen35.swift:5432). `assignment` is the boundary the assignment
            # names, which uses the compact ROW COUNT 98,330 as if it were the
            # tokenizer-id prefix boundary 98,304.
            "channel_a_strict": compact_rank(t_star) < 0,
            "channel_a_assignment": (
                t_star >= COMPACT_REAL_COUNT
                and not (COMPACT_CONTROL_START <= t_star < COMPACT_CONTROL_END)),
        })

    counters = {
        "seed": seed,
        "carrier": SEED_CARRIER.get(seed, "other"),
        "round_count": n_rounds,
        "drafting_round_count": n_drafting_rounds,
        "divergent_round_count": len(records),
        "accepted_draft_total": accepted,
        "reported_accepted_draft_total": payload["accepted_draft_total"],
        "reported_rejected_draft_total": payload["rejected_draft_total"],
        "reported_accepted_draft_rate": payload["accepted_draft_rate"],
        "effective_mean_draft_len": payload["effective_mean_draft_len"],
        "decode_token_count": payload["decode_token_count"],
        "all_tokens_matched": payload["all_tokens_matched"],
        "parity_all_ok": payload["parity_all_ok"],
        "residual_divergence_count": payload["residual_divergence_count"],
        "head_sha256": payload["head_provenance"]["sha256"],
    }
    if accepted != payload["accepted_draft_total"]:
        raise SystemExit(
            f"e143-r0: {seed}: reconstructed accepted total {accepted} != "
            f"reported {payload['accepted_draft_total']}")
    counters["per_step_p"] = (
        accepted / (accepted + len(records)) if accepted + len(records) else float("nan"))
    counters["per_step_miss"] = 1.0 - counters["per_step_p"]
    return records, counters


def wilson(successes: int, trials: int, z: float = 1.0) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def channel_table(records: list[dict], key: str = "channel_a_strict") -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    ca = sum(1 for r in records if r[key])
    lo, hi = wilson(ca, n)
    return {
        "n": n,
        "channel_a_count": ca,
        "channel_a_share": ca / n,
        "channel_a_share_ci68": [lo, hi],
        "not_channel_a_share": 1.0 - ca / n,
        "d_is_target_rank2_share": sum(r["d_is_target_rank2"] for r in records) / n,
        "margin_top2_median": float(np.median([r["margin_top2"] for r in records])),
        "margin_top2_p10": float(np.percentile([r["margin_top2"] for r in records], 10)),
        "margin_top2_p90": float(np.percentile([r["margin_top2"] for r in records], 90)),
        "margin_top2_within_arithmetic_gap_share": (
            sum(r["margin_top2"] < ARITHMETIC_GAP for r in records) / n),
        "mean_first_divergence_index": float(
            np.mean([r["first_divergence_index"] for r in records])),
        "mean_width": float(np.mean([r["width"] for r in records])),
    }


def width_strata(records: list[dict], counters: list[dict]) -> list[dict]:
    """Channel table inside realised-width strata, each with its own `p`.

    The adaptive depth policy widens a round when recent rounds accepted, so
    realised width is a leakage-free per-round proxy for the local regime. The
    stratum's own `p` is recomputed from that stratum's trials only.
    """
    per_round: dict[tuple[str, int], dict] = {}
    for c in counters:
        payload = load_seed(c["seed"])
        rounds: dict[int, list[dict]] = defaultdict(list)
        for row in payload["row_ledger"]:
            rounds[row["round"]].append(row)
        for index, rows in rounds.items():
            drafts = [r for r in rows if r["kind"] == "draft"]
            if not drafts:
                continue
            diverged = next((r for r in sorted(
                drafts, key=lambda r: r["draft_index"]) if not r["accepted"]), None)
            per_round[(c["seed"], index)] = {
                "width": len(drafts),
                "accepted": len(drafts) if diverged is None
                            else diverged["draft_index"],
                "divergent": diverged is not None,
            }
    by_record = {(r["seed"], r["round"]): r for r in records}

    out = []
    for width in sorted({v["width"] for v in per_round.values()}):
        cell = [(k, v) for k, v in per_round.items() if v["width"] == width]
        accepted = sum(v["accepted"] for _, v in cell)
        div = sum(v["divergent"] for _, v in cell)
        rows = [by_record[k] for k, v in cell if v["divergent"]]
        table = channel_table(rows)
        table.update({
            "width": width,
            "rounds": len(cell),
            "accepted_draft_total": accepted,
            "divergent_round_count": div,
            "per_step_p": accepted / (accepted + div) if accepted + div else float("nan"),
        })
        out.append(table)
    return out


def weighted_line(xs, ys, ws):
    """Weighted least squares `y = a + b x` with the slope's standard error."""
    xs, ys, ws = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ws, float)
    keep = np.isfinite(xs) & np.isfinite(ys) & (ws > 0)
    xs, ys, ws = xs[keep], ys[keep], ws[keep]
    if xs.size < 3 or np.ptp(xs) == 0:
        return None
    sw = ws.sum()
    xbar = (ws * xs).sum() / sw
    ybar = (ws * ys).sum() / sw
    sxx = (ws * (xs - xbar) ** 2).sum()
    b = (ws * (xs - xbar) * (ys - ybar)).sum() / sxx
    a = ybar - b * xbar
    resid = ys - (a + b * xs)
    dof = xs.size - 2
    s2 = (ws * resid ** 2).sum() / dof if dof > 0 else float("nan")
    se_b = math.sqrt(s2 / sxx) if sxx > 0 and np.isfinite(s2) and s2 >= 0 else float("nan")
    return {"intercept": float(a), "slope": float(b), "slope_stderr": float(se_b),
            "points": int(xs.size), "x_min": float(xs.min()), "x_max": float(xs.max())}


def price_rate(rate: float, ci: tuple[float, float], carriers=("beagle", "essays")) -> dict:
    """Ranked percent of the published median for a per-trial miss-rate rate.

    A channel's PER-TRIAL RATE is the quantity that transfers between regimes;
    its SHARE OF DIVERGENCES is not, because the share's denominator is the
    regime's own miss rate. Closing a channel raises per-step acceptance by
    exactly its per-trial rate, so `delta_p = rate` and the ranked value is
    `MISS_TO_SCORE_PCT * rate * carrier weight`, summed over the two carriers
    that Rule 116 gives non-zero weight.
    """
    out = {"rate": rate, "rate_ci68": list(ci), "delta_p": rate}
    total = total_lo = total_hi = 0.0
    for carrier in carriers:
        w = CARRIER_WEIGHT[carrier]
        out[f"ranked_pct_{carrier}"] = MISS_TO_SCORE_PCT * rate * w
        out[f"implied_share_of_ranked_misses_{carrier}"] = rate / (1.0 - RANKED_P[carrier])
        total += MISS_TO_SCORE_PCT * rate * w
        total_lo += MISS_TO_SCORE_PCT * ci[0] * w
        total_hi += MISS_TO_SCORE_PCT * ci[1] * w
    out["ranked_pct"] = total
    out["ranked_pct_ci68"] = [total_lo, total_hi]
    return out


def emitted_token_census(seeds: list[str]) -> dict:
    """Unproposable-token rate over the whole emitted trajectory.

    Cross-check against the census Alphonse reports as a share OF TOKENS.

    THE TRAP THIS AVOIDS. A round emits its accepted drafts and then its
    `targetTail` token. In a DIVERGENT round the `targetTail` row is not a new
    position at all: it re-reports the divergence row's readout, so its
    `reference_token` is the same `t*` the divergence row already carries.
    Counting both rows doubles every unproposable token that caused a miss and
    inflates the emitted total by exactly the divergence count. This function
    counts the accepted drafts and the tail row only, and asserts that the
    result is the declared decode length.
    """
    total = 0
    unproposable = 0
    unproposable_ids: list[int] = []
    drafted_positions = 0
    drafted_unproposable = 0
    tail_repeats_divergence = 0
    divergent_rounds = 0
    for seed in seeds:
        payload = load_seed(seed)
        rounds: dict[int, list[dict]] = defaultdict(list)
        for row in payload["row_ledger"]:
            rounds[row["round"]].append(row)
        emitted = 0
        for rows in rounds.values():
            drafts = sorted((r for r in rows if r["kind"] == "draft"),
                            key=lambda r: r["draft_index"])
            tails = [r for r in rows if r["kind"] == "targetTail"]
            if len(tails) != 1:
                raise SystemExit(f"e143-r0: {seed}: {len(tails)} tail rows in a round")
            diverged = next((r for r in drafts if not r["accepted"]), None)
            accepted = len(drafts) if diverged is None else diverged["draft_index"]
            if diverged is not None:
                divergent_rounds += 1
                if tails[0]["reference_token"] == diverged["reference_token"]:
                    tail_repeats_divergence += 1
            emitted += accepted + 1
            # Draft trials: the accepted drafts plus the one genuine failure.
            drafted_positions += accepted + (1 if diverged is not None else 0)
            if diverged is not None and compact_rank(diverged["reference_token"]) < 0:
                drafted_unproposable += 1
            for r in drafts[:accepted] + tails:
                total += 1
                if compact_rank(r["reference_token"]) < 0:
                    unproposable += 1
                    unproposable_ids.append(r["reference_token"])
        if emitted != payload["decode_token_count"]:
            raise SystemExit(
                f"e143-r0: {seed}: reconstructed {emitted} emitted tokens != "
                f"declared {payload['decode_token_count']}")
    lo, hi = wilson(unproposable, total)
    dlo, dhi = wilson(drafted_unproposable, drafted_positions)
    return {
        "emitted_token_total": total,
        "emitted_unproposable": unproposable,
        "emitted_unproposable_ids": sorted(set(unproposable_ids)),
        "emitted_unproposable_share": unproposable / total,
        "emitted_unproposable_share_ci68": [lo, hi],
        "draft_trial_total": drafted_positions,
        "draft_trial_unproposable": drafted_unproposable,
        "draft_trial_unproposable_rate": drafted_unproposable / drafted_positions,
        "draft_trial_unproposable_rate_ci68": [dlo, dhi],
        "divergent_rounds": divergent_rounds,
        "tail_row_repeats_divergence_row": tail_repeats_divergence,
        "note": ("the draft-trial rate is the causal one: an unproposable token "
                 "at a draft position forces a miss, while a bonus-row token in "
                 "a fully accepted round costs nothing because no draft was made "
                 "for it"),
    }


def stage_a(seeds: list[str]) -> dict:
    all_records: list[dict] = []
    counters: list[dict] = []
    for seed in seeds:
        payload = load_seed(seed)
        records, c = first_divergences(seed, payload)
        all_records.extend(records)
        counters.append(c)
        print(f"e143-r0: {seed:<18} rounds={c['round_count']:>4} "
              f"div={c['divergent_round_count']:>4} accepted={c['accepted_draft_total']:>4} "
              f"p={c['per_step_p']:.4f} acc_rate={c['reported_accepted_draft_rate']:.4f}",
              flush=True)

    by_carrier = {}
    for carrier in ("beagle", "essays", "other"):
        rows = [r for r in all_records if r["carrier"] == carrier]
        cs = [c for c in counters if c["carrier"] == carrier]
        if not cs:
            continue
        accepted = sum(c["accepted_draft_total"] for c in cs)
        div = sum(c["divergent_round_count"] for c in cs)
        table = channel_table(rows)
        table.update({
            "seeds": [c["seed"] for c in cs],
            "accepted_draft_total": accepted,
            "divergent_round_count": div,
            "per_step_p": accepted / (accepted + div),
            "per_step_miss": div / (accepted + div),
        })
        by_carrier[carrier] = table

    strata = width_strata(all_records, counters)
    fit_a = weighted_line(
        [s["per_step_p"] for s in strata],
        [s.get("channel_a_share", float("nan")) for s in strata],
        [s.get("n", 0) for s in strata])
    fit_tie = weighted_line(
        [s["per_step_p"] for s in strata],
        [s.get("margin_top2_within_arithmetic_gap_share", float("nan")) for s in strata],
        [s.get("n", 0) for s in strata])

    pooled = channel_table(all_records)
    pooled_assignment = channel_table(all_records, key="channel_a_assignment")

    # Per-trial rates. These are the transferable quantities: `rate` is the
    # probability that a draft trial misses through this channel, and closing
    # the channel raises per-step `p` by exactly that rate.
    trials = sum(c["accepted_draft_total"] + c["divergent_round_count"] for c in counters)
    ca_events = sum(1 for r in all_records if r["channel_a_strict"])
    rate_by_stratum = [
        {"per_step_p": s["per_step_p"],
         "trials": s["accepted_draft_total"] + s["divergent_round_count"],
         "channel_a_events": s.get("channel_a_count", 0),
         "channel_a_rate": (s.get("channel_a_count", 0)
                            / (s["accepted_draft_total"] + s["divergent_round_count"])),
         "miss_rate": s["divergent_round_count"]
                      / (s["accepted_draft_total"] + s["divergent_round_count"])}
        for s in strata]
    fit_ca_rate = weighted_line(
        [s["per_step_p"] for s in rate_by_stratum],
        [s["channel_a_rate"] for s in rate_by_stratum],
        [s["trials"] for s in rate_by_stratum])

    channel_a = price_rate(ca_events / trials, wilson(ca_events, trials))
    channel_a.update({
        "events": ca_events, "trials": trials, "label": "measured",
        "boundary": "t* not in [0,98304) and not in [248044,248070)",
    })

    # The ceiling for everything that is NOT structurally unproposable: the
    # most that C-b + C-c + C-d can be worth before the head replay splits it.
    inv_events = len(all_records) - ca_events
    ceiling = price_rate(inv_events / trials, wilson(inv_events, trials))
    ceiling.update({"events": inv_events, "trials": trials, "label": "measured"})

    # The same ceiling carried to the RANKED regimes, where the miss rate is
    # known from ranked evidence rather than measured here. This is the number
    # the fork threshold is compared against, and it is `derived`.
    ranked_ceiling = {}
    for carrier in ("beagle", "essays"):
        if carrier not in by_carrier:
            continue
        in_vocab_share = by_carrier[carrier]["not_channel_a_share"]
        ranked_miss = 1.0 - RANKED_P[carrier]
        ranked_ceiling[carrier] = {
            "measured_at_p": by_carrier[carrier]["per_step_p"],
            "ranked_p": RANKED_P[carrier],
            "ranked_miss": ranked_miss,
            "in_vocabulary_share_of_divergences": in_vocab_share,
            "delta_p_ceiling": ranked_miss * in_vocab_share,
            "ranked_pct_ceiling": (MISS_TO_SCORE_PCT * ranked_miss * in_vocab_share
                                   * CARRIER_WEIGHT[carrier]),
            "label": "derived",
        }
    ranked_ceiling["ranked_pct_ceiling_both_carriers"] = sum(
        v["ranked_pct_ceiling"] for v in ranked_ceiling.values() if isinstance(v, dict))

    total_gap = {
        "beagle_full_gap_ranked_pct": (
            MISS_TO_SCORE_PCT * (RANKED_P["essays"] - RANKED_P["beagle"])
            * CARRIER_WEIGHT["beagle"]),
        "note": "the whole beagle-to-essays per-step gap, for scale",
    }

    return {
        "stage": "a",
        "harness": "local",
        "seed_counters": counters,
        "pooled_never_quote_as_ranked": pooled,
        "pooled_assignment_boundary_98330": pooled_assignment,
        "by_carrier": by_carrier,
        "width_strata": strata,
        "rate_by_stratum": rate_by_stratum,
        "channel_a_share_vs_p_fit": fit_a,
        "channel_a_rate_vs_p_fit": fit_ca_rate,
        "near_tie_share_vs_p_fit": fit_tie,
        "channel_a": channel_a,
        "emitted_token_census": emitted_token_census(seeds),
        "in_vocabulary_ceiling": ceiling,
        "in_vocabulary_ceiling_at_ranked_p": ranked_ceiling,
        "scale": total_gap,
        "records": all_records,
    }


def stage_b(state: dict, chunk: int) -> dict:
    """The target's exact logit and rank of the proposed token `d`.

    One streamed pass over the affine-4 lm_head against the captured hidden
    rows of the first-divergence positions only. This says how wrong the head
    was in the TARGET's geometry, which bounds how near-miss the divergences
    are without needing the head's own hidden state.
    """
    import mlx.core as mx

    records = state["records"]
    seeds = sorted({r["seed"] for r in records})
    want: dict[str, dict[int, int]] = defaultdict(dict)
    for i, r in enumerate(records):
        want[r["seed"]][r["round"]] = i

    hidden = np.zeros((len(records), HIDDEN), dtype=np.float32)
    dump = CACHE / "hidden"
    for seed in seeds:
        shards = sorted(dump.glob(f"{seed}.pid*.meta.i32"))
        if not shards:
            raise SystemExit(f"e143-r0: no dump shard for seed {seed}")
        for meta_path in shards:
            stem = str(meta_path)[: -len(".meta.i32")]
            meta = np.fromfile(meta_path, dtype=np.int32)
            tok = np.fromfile(stem + ".tok.i32", dtype=np.int32)
            t2i = np.fromfile(stem + ".top2.i32", dtype=np.int32).reshape(-1, 2)
            t2v = np.fromfile(stem + ".top2.f32", dtype=np.float32).reshape(-1, 2)
            xs = np.fromfile(stem + ".x.f32", dtype=np.float32).reshape(-1, HIDDEN)
            offset = 0
            for index, width in enumerate(meta.tolist()):
                slot = want[seed].get(index)
                if slot is not None:
                    rec = records[slot]
                    row = offset + rec["first_divergence_index"]
                    # The dump and the ledger must agree on this row or the
                    # hidden state does not belong to the divergence.
                    if int(t2i[row, 0]) != rec["t_star"]:
                        raise SystemExit(
                            f"e143-r0: {seed} round {index}: dump argmax "
                            f"{int(t2i[row, 0])} != ledger t* {rec['t_star']}")
                    if int(tok[row + 1]) != rec["d"]:
                        raise SystemExit(
                            f"e143-r0: {seed} round {index}: dump input token "
                            f"{int(tok[row + 1])} != ledger proposal {rec['d']}")
                    rec["dump_s1"] = float(t2v[row, 0])
                    rec["dump_s2"] = float(t2v[row, 1])
                    hidden[slot] = xs[row]
                offset += width
    print(f"e143-r0: stage b over {len(records)} first-divergence rows", flush=True)

    blob = mx.load("weights/model-00003-of-00003.safetensors")
    wq = blob["language_model.lm_head.weight"]
    scales = blob["language_model.lm_head.scales"]
    biases = blob["language_model.lm_head.biases"]

    H = mx.array(hidden)
    n = len(records)
    d_ids = np.array([r["d"] for r in records], dtype=np.int32)
    t_ids = np.array([r["t_star"] for r in records], dtype=np.int32)

    def exact_at(ids: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(ids, return_inverse=True)
        take = mx.array(unique.astype(np.int32))
        deq = mx.dequantize(wq[take], scales[take], biases[take],
                            group_size=GROUP, bits=4).astype(mx.float32)
        return np.array(H @ deq.T)[np.arange(n), inverse]

    logit_d = exact_at(d_ids).astype(np.float64)
    logit_t = exact_at(t_ids).astype(np.float64)

    greater_d = mx.zeros((n,), dtype=mx.int32)
    off_top1 = mx.full((n,), -float("inf"))
    off_top1_id = mx.zeros((n,), dtype=mx.int32)
    ld = mx.array(logit_d.astype(np.float32))
    started = time.time()
    for start in range(0, VOCAB, chunk):
        deq = mx.dequantize(wq[start:start + chunk], scales[start:start + chunk],
                            biases[start:start + chunk],
                            group_size=GROUP, bits=4).astype(mx.float32)
        exact = H @ deq.T
        del deq
        greater_d = greater_d + (exact > ld[:, None]).sum(axis=1).astype(mx.int32)
        best = exact.max(axis=1)
        off_top1_id = mx.where(
            best > off_top1, (mx.argmax(exact, axis=1) + start).astype(mx.int32),
            off_top1_id)
        off_top1 = mx.maximum(off_top1, best)
        mx.eval(greater_d, off_top1, off_top1_id)
        del exact
        print(f"  stage-b {start:>7}..{start + chunk:<7} {time.time() - started:6.1f}s",
              flush=True)

    rank_d = np.array(greater_d).astype(int) + 1
    off_id = np.array(off_top1_id)
    off_val = np.array(off_top1).astype(np.float64)
    device_s1 = np.array([r["s1"] for r in records], dtype=np.float64)

    agree = off_id == t_ids
    gap_t_minus_d = logit_t - logit_d
    unresolved = gap_t_minus_d < ARITHMETIC_GAP

    for i, r in enumerate(records):
        r["target_rank_of_d"] = int(rank_d[i])
        r["target_logit_d"] = float(logit_d[i])
        r["target_logit_t_star"] = float(logit_t[i])
        r["target_gap_t_minus_d"] = float(gap_t_minus_d[i])
        r["offline_argmax_matches_device"] = bool(agree[i])
        r["unresolved"] = bool(unresolved[i])

    fidelity = {
        "offline_argmax_matches_device": float(agree.mean()),
        "s1_abs_delta_max": float(np.max(np.abs(off_val - device_s1))),
        "s1_abs_delta_median": float(np.median(np.abs(off_val - device_s1))),
        "arithmetic_gap_used": ARITHMETIC_GAP,
    }
    out = {"fidelity": fidelity, "rows": n}
    for carrier in ("beagle", "essays", "other"):
        sel = [i for i, r in enumerate(records) if r["carrier"] == carrier]
        if not sel:
            continue
        rk = rank_d[sel]
        out[carrier] = {
            "n": len(sel),
            "target_rank_of_d_median": float(np.median(rk)),
            "target_rank_of_d_p90": float(np.percentile(rk, 90)),
            "target_rank_of_d_is_2_share": float(np.mean(rk == 2)),
            "target_rank_of_d_le_4_share": float(np.mean(rk <= 4)),
            "target_rank_of_d_le_32_share": float(np.mean(rk <= 32)),
            "target_rank_of_d_gt_1000_share": float(np.mean(rk > 1000)),
            "target_gap_t_minus_d_median": float(np.median(gap_t_minus_d[sel])),
            "unresolved_fraction": float(np.mean(unresolved[sel])),
        }
    out["unresolved_fraction_all"] = float(unresolved.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="a", choices=["a", "b"])
    parser.add_argument("--seeds", default="")
    parser.add_argument("--chunk", type=int, default=15520)
    parser.add_argument("--out", default="research/e143-r0.json")
    args = parser.parse_args()

    seeds = [s for s in args.seeds.split(",") if s] or sorted(
        p.stem for p in (CACHE / "verify").glob("*.json"))

    state = stage_a(seeds)
    if args.stage == "b":
        state["stage"] = "ab"
        state["stage_b"] = stage_b(state, args.chunk)

    Path(args.out).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in state.items() if k != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"e143-r0: wrote {args.out} ({len(state['records'])} divergences)")


if __name__ == "__main__":
    main()
