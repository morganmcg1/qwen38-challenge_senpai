#!/usr/bin/env python3
"""E34 r2: autopsy of the r1 cost model against the 15 declared-head ranked rows.

The advisor asked one question: feed the five shallower rows' beagle
`effective_mean_draft_len` into the r1 model, and if it predicts gains where
reality measured losses, name the missing term.

It does predict gains. There are four missing terms, and the largest one is
not in the cost half of the model at all.

  MT1  TOKEN CREDIT.  `effective_mean_draft_len` counts drafts PROPOSED, not
       accepted.  The r1 model credited `1 + n` emitted tokens per round, which
       is identically equivalent to asserting a 100.00 % draft acceptance rate
       on every prompt.  Ranked reality on our own row is 33-90 %.

  MT2  PASS COUNT IS SHAPE-DEPENDENT.  `weight_passes(M)` returns one number
       per width.  The live kernel picks between three regimes on
       `out_vec_size`, and Qwen 3.8 has projections in two of them.

  MT3  BUILD ASYMMETRY.  The ranked ratio divides the PINNED SERIAL build by
       OUR candidate build; the local ratio divides our candidate by itself.
       The r1 model predicted a ranked ratio with a local ratio's structure.

  MT4  GEOMETRY.  The local ladder's absolutes were measured under the
       low-memory command-buffer profile, where the buffer-flush boundary and
       the weight-pass boundary are perfectly confounded at M = 6.

MT1 and MT3 are arithmetic, not opinion, and either one alone is enough to
withdraw the r1 prediction.

The decisive evidence is MT5, which needs no GPU time: the ranked telemetry
publishes `effective_mean_draft_len` as an exact rational and a fixed 512-token
window, so `512 = rounds + accepted` pins the round count by integer
enumeration.  That reconstructs the ranked per-round width ladder, the
per-prompt acceptance rate, and the pinned-serial/candidate build ratio
directly from published numbers, and it reproduces our own official score to
six decimal places.

    python3 research/e34r2_model_autopsy.py            # full report
    python3 research/e34r2_model_autopsy.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Measured inputs. Every number below is quoted from a named source.
# ---------------------------------------------------------------------------

# thorfinn E33, absolute per-width round cost of the BASE arm, post-E27,
# quoted in the advisor's 2026-08-19T03:47Z feedback, section 3.
E33_LADDER_MS = {
    1: 58.676, 2: 63.212, 3: 72.507, 4: 82.774, 5: 96.163,
    6: 128.843, 7: 138.694, 8: 149.490, 9: 164.443,
}

# Board top 0cd0a6b4 (ofou) and our row ca9251b8, per-prompt.
# n = effective_mean_draft_len (drafts PROPOSED per round -- see prove_semantics).
# Ascending by ratio on the reference frontier row; central pair is ranks 4-5.
RANKED = {
    #            n_top     R_top      n_ours    R_ours    non_drafting
    "plutarch": (0.1540, 1.25600, 0.1540, 1.25280, 449),
    "drama":    (2.2976, 1.92310, 2.2976, 1.91668, 0),
    "travel":   (2.6557, 2.18950, 2.6557, 2.17980, 0),
    "beagle":   (4.5327, 3.14330, 4.5327, 3.12015, 0),
    "medicine": (4.7677, 3.35530, 4.7677, 3.34486, 0),
    "republic": (5.2697, 3.41440, 5.2697, 3.39402, 0),
    "essays":   (5.4253, 3.39070, 5.4253, 3.36612, 0),
    "botany":   (5.7765, 3.44910, 5.7765, 3.42536, 0),
}
CENTRAL = ("beagle", "medicine")
BOARD_TOP = 3.24929398547457
DECODE_TOKENS = 512
OURS_OFFICIAL_SCORE = 3.23250848263467

# Full per-prompt telemetry for our own ranked row ca9251b8 (morganmcg1).
# The cache under .mlxfast-private/ is gitignored, so the fields the
# reconstruction needs are transcribed here and re-verified against the cache by
# `--verify-telemetry` when it is present.
# name: (n as an exact fraction p/q, raw_p, non_drafting, mtp_ms/token, serial_ms/token)
OURS_PER_PROMPT = {
    "plutarch": ((75, 487),  1.25280, 449, 30.3063, 37.9678),
    "drama":    ((193, 84),  1.91668,   0, 19.7767, 37.9057),
    "travel":   ((563, 212), 2.17980,   0, 17.3887, 37.9038),
    "beagle":   ((485, 107), 3.12015,   0, 12.1740, 37.9849),
    "medicine": ((472, 99),  3.34486,   0, 11.3686, 38.0265),
    "republic": ((469, 89),  3.39402,   0, 11.1838, 37.9579),
    "essays":   ((472, 87),  3.36612,   0, 11.2571, 37.8929),
    "botany":   ((491, 85),  3.42536,   0, 11.0805, 37.9546),
}

# Submission c91581eb (scarletbright), the near-zero-drafting row. Every prompt
# reconstructs uniquely to 512 rounds and ZERO accepted drafts, which makes its
# per-round cost a direct measurement of a candidate build at width 1.
# name: (n as an exact fraction, raw_p, non_drafting, mtp_ms/token, serial_ms/token)
NEAR_ZERO_ROW = {
    "id": "c91581eb",
    "solver": "scarletbright",
    "prompts": {
        "p0": ((21, 512), 1.21028, 502, 31.4806, 38.1003),
        "p1": ((27, 512), 1.20585, 502, 31.5063, 37.9921),
        "p2": ((3, 64),   1.20785, 502, 31.4704, 38.0115),
        "p3": ((19, 512), 1.21018, 502, 31.4673, 38.0811),
        "p4": ((23, 512), 1.20836, 502, 31.4857, 38.0462),
        "p5": ((9, 256),  1.21155, 502, 31.4382, 38.0888),
        "p6": ((25, 512), 1.20620, 502, 31.5001, 37.9954),
        "p7": ((9, 256),  1.21150, 502, 31.4194, 38.0647),
    },
}

# Pinned-serial reproducibility across all 88 declared-head submissions and all
# 8 prompts, from the same cache. The serial leg is pinned AND prompt-independent.
PINNED_SERIAL_MS = {
    "mean_over_8_prompts": 37.9908,
    "spread_across_prompts_ms": 0.0231,   # 37.9805 .. 38.0036
    "within_prompt_cv_pct_range": (0.1991, 0.2766),
    "rows_per_prompt": 88,
}

# The advisor's declared-head direction test (r2 request). beagle only.
# (submission id, beagle n, beagle raw_p).  Ours is the reference row.
#
# Two provenance corrections against the telemetry cache, both re-derived by
# `--verify-telemetry`:
#   - a874233e was in the advisor's table but is still `validating` with null
#     officialMetrics. It has no beagle measurement at all and is dropped.
#   - baa75efa, 26d0e934 and a1326b4b are scored declared-head rows that the
#     table omitted. They are added, which widens the near-zero end of the
#     sweep from one row to three.
DECLARED_HEAD_BEAGLE = [
    ("c91581eb", 0.0410, 1.21028),
    ("baa75efa", 0.0410, 1.20367),   # added by cross-check
    ("26d0e934", 0.0509, 1.20544),   # added by cross-check
    ("77d5f0f7", 1.0000, 2.00142),
    ("505caf3d", 3.5352, 2.66656),
    ("6a57c528", 4.2793, 3.11026),
    ("0d47d685", 4.3393, 3.07650),
    ("581cc3a9", 4.3839, 3.06026),
    ("a1326b4b", 4.3839, 3.04472),   # added by cross-check
    ("9100a4e7", 4.3964, 3.01183),
    ("6f37594f", 4.4537, 3.07681),
    ("ca9251b8", 4.5327, 3.12015),   # ours
    ("0691113e", 4.6019, 3.00622),
    ("96a84b1b", 4.6296, 3.06201),
    ("74ec0ec7", 4.7358, 3.06265),
]
DROPPED_PHANTOM_ROW = {
    "id": "a874233e", "solver": "jonathan308", "status": "validating",
    "quoted_in_advisor_table": {"beagle_n": 4.583, "beagle_raw_p": 3.06986},
    "reason": ("officialMetrics is null and the submission has no beagle "
               "per-prompt entry, so the quoted pair is not a measurement."),
}
OURS_BEAGLE_N = 4.5327
OURS_BEAGLE_R = 3.12015
# The five rows that are shallower than ours: the direct test of the r1 direction.
SHALLOWER = [r for r in DECLARED_HEAD_BEAGLE if 4.0 < r[1] < OURS_BEAGLE_N]

# askeladd's exact bracket on the M>=6 round share, from published telemetry.
ASKELADD_BRACKET = {"beagle": (0.1332, 0.9065), "medicine": (0.1919, 0.9535)}
E34_R1_SIM_SHARE = {"beagle": 0.538, "medicine": 0.593}

# Advisor retraction 2026-08-19T03:47Z section 2.
SIGMA_SCORE_PCT = 0.0923
ENGINEERABLE_GAP_PCT = 0.561


def ladder(width: float) -> float:
    """Linear interpolation of the E33 absolute ladder at a fractional width."""
    if width <= 1:
        return E33_LADDER_MS[1]
    if width >= 9:
        return E33_LADDER_MS[9]
    lo = int(math.floor(width))
    frac = width - lo
    return E33_LADDER_MS[lo] + frac * (E33_LADDER_MS[lo + 1] - E33_LADDER_MS[lo])


# ---------------------------------------------------------------------------
# MT1: what does effective_mean_draft_len actually count?
# ---------------------------------------------------------------------------

def prove_semantics() -> dict:
    """Settle proposed-vs-accepted from the trusted harness and exact integers.

    Two independent proofs:

    1. SOURCE. `Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:363-374`
       documents `effectiveDraftLengths` as "EFFECTIVE per-round draft counts
       ... what the candidate actually PROPOSED, not what the parent offered",
       and `effectiveMeanDraftLength` is their plain mean.

    2. ARITHMETIC, on our own local tapes, with no free parameters:
           sum(effective_draft_lengths) == accepted_draft_total + rejected_draft_total
           declared_rows_total          == round_count + sum(effective_draft_lengths)
           emitted_token_total          == round_count + accepted_draft_total
       The first identity is only possible if the array counts PROPOSALS.
    """
    tapes = []
    for path in sorted(glob.glob(str(REPO / ".mlxfast-private/e25/runs-*/*/reports/04-mtp-timed.json"))):
        doc = json.load(open(path))
        if doc.get("is_serial_control"):
            continue
        lengths = doc["effective_draft_lengths"]
        proposed = sum(lengths)
        rounds = doc["round_count"]
        accepted = doc["accepted_draft_total"]
        rejected = doc["rejected_draft_total"]
        tapes.append({
            "tape": pathlib.Path(path).parts[-3],
            "round_count": rounds,
            "effective_mean_draft_len": doc["effective_mean_draft_len"],
            "proposed_total": proposed,
            "accepted_total": accepted,
            "rejected_total": rejected,
            # proposals == accepted + rejected  <=>  the array counts proposals
            "identity_proposed_eq_acc_plus_rej": proposed == accepted + rejected,
            "identity_rows_eq_rounds_plus_proposed": doc["declared_rows_total"] == rounds + proposed,
            "emitted_minus_rounds_plus_accepted": doc["emitted_token_total"] - (rounds + accepted),
            "true_acceptance_rate": accepted / proposed if proposed else float("nan"),
        })

    # Plutarch impossibility: if n were the ACCEPTED count, the ranked row's own
    # published non_drafting_round_count could not fit inside the token window.
    n, non_drafting = RANKED["plutarch"][0], RANKED["plutarch"][4]
    rounds_if_accepted = DECODE_TOKENS / (1.0 + n)
    plutarch = {
        "n": n,
        "non_drafting_round_count": non_drafting,
        "rounds_implied_if_n_were_accepted": rounds_if_accepted,
        "drafting_rounds_implied": rounds_if_accepted - non_drafting,
        "impossible": rounds_if_accepted < non_drafting,
    }

    return {
        "source": "Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift:363-374",
        "verdict": "effective_mean_draft_len counts drafts PROPOSED per round, over all rounds",
        "tapes": tapes,
        "all_identities_hold": all(
            t["identity_proposed_eq_acc_plus_rej"] and t["identity_rows_eq_rounds_plus_proposed"]
            for t in tapes),
        "measured_acceptance_rate_range": [
            min(t["true_acceptance_rate"] for t in tapes),
            max(t["true_acceptance_rate"] for t in tapes),
        ] if tapes else None,
        "plutarch_impossibility": plutarch,
        # The r1 credit of (1+n) tokens per round implies rounds = 512/(1+n) and
        # therefore accepted = 512 - rounds = 512*n/(1+n) = n * rounds, i.e. an
        # acceptance rate of exactly 1 for EVERY n. Not approximately: exactly.
        "r1_implied_acceptance_rate": {
            name: (DECODE_TOKENS - DECODE_TOKENS / (1 + v[0])) / (v[0] * DECODE_TOKENS / (1 + v[0]))
            for name, v in RANKED.items()
        },
    }


# ---------------------------------------------------------------------------
# MT3: the ranked ratio is not the local ratio
# ---------------------------------------------------------------------------

def transfer_falsification() -> dict:
    """Show the r1 local->ranked transfer implies more accepted than proposed.

    Round accounting, for one prompt:

        raw_p = serial_s_per_tok / mtp_s_per_tok
        mtp_s_per_tok = mean_round_cost / (1 + a)          a = ACCEPTED per round

    The r1 model set serial_s_per_tok = T(1) from the same build, giving

        1 + a = raw_p * T(Mbar) / T(1),     Mbar = 1 + n

    `a` is bounded above by `n`, because a round cannot accept more drafts than
    it proposed. If the implied `a` exceeds `n`, the transfer is falsified.

    The resolution is `k`, the ratio by which the PINNED SERIAL build is slower
    per token at width 1 than our candidate build. program.md defines the ranked
    numerator as the pinned serial build, while both LOCAL legs use the
    candidate build. Requiring a <= n on every prompt gives a lower bound on k.
    """
    rows, k_bounds = [], []
    for name, (n_top, r_top, n_ours, r_ours, _nd) in RANKED.items():
        width = 1.0 + n_ours
        shape = ladder(width) / ladder(1.0)
        implied_a = r_ours * shape - 1.0
        # a <= n  =>  raw_p * shape / k - 1 <= n  =>  k >= raw_p*shape/(1+n)
        k_min = r_ours * shape / (1.0 + n_ours)
        k_bounds.append(k_min)
        rows.append({
            "prompt": name,
            "n_proposed": n_ours,
            "mean_width": width,
            "raw_p": r_ours,
            "ladder_shape_T(M)/T(1)": shape,
            "implied_accepted_per_round": implied_a,
            "exceeds_proposed": implied_a > n_ours,
            "excess_over_proposed": implied_a - n_ours,
            "k_min_required": k_min,
        })
    return {
        "rows": rows,
        "prompts_violating": sum(1 for r in rows if r["exceeds_proposed"]),
        "k_lower_bound": max(k_bounds),
        "k_lower_bound_pct": (max(k_bounds) - 1.0) * 100.0,
        "note": ("A single k >= max(k_min) reconciles every prompt, so the "
                 "violation is systematic and not a Jensen artifact."),
    }


# ---------------------------------------------------------------------------
# The r1 model, replayed on the five shallower rows
# ---------------------------------------------------------------------------

def r1_prediction(n: float) -> float:
    """r1's predicted beagle raw_p at proposed depth `n`, anchored on our row.

    This is the r1 model exactly as it was: round cost from the width ladder,
    tokens per round credited as `1 + n`, and the ratio taken against a
    same-build width-1 leg. Anchoring on our own measured row removes any
    absolute-scale question and isolates the model's SHAPE, which is what the
    advisor's test is about.
    """
    anchor = (1.0 + OURS_BEAGLE_N) / ladder(1.0 + OURS_BEAGLE_N)
    return OURS_BEAGLE_R * ((1.0 + n) / ladder(1.0 + n)) / anchor


def corrected_prediction(n: float, accept_rate: float, k: float) -> float:
    """Same ladder, but with MT1 and MT3 repaired.

    MT1: credit `1 + a` tokens, where `a` is the ACCEPTED count implied by the
         per-slot acceptance profile, not the proposed count `n`.
    MT3: divide by a pinned serial leg that is `k` times slower at width 1.
    """
    a = accept_rate * n
    return k * ladder(1.0) * (1.0 + a) / ladder(1.0 + n)


def replay_shallower() -> dict:
    """The advisor's requested test, plus the corrected model's answer."""
    # Calibrate the corrected model on our own row: choose the acceptance rate
    # that reproduces our measured beagle raw_p at the measured k lower bound.
    trans = transfer_falsification()
    k = trans["k_lower_bound"]
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if corrected_prediction(OURS_BEAGLE_N, mid, k) < OURS_BEAGLE_R:
            lo = mid
        else:
            hi = mid
    accept_rate = 0.5 * (lo + hi)

    rows = []
    for sid, n, measured in DECLARED_HEAD_BEAGLE:
        r1 = r1_prediction(n)
        corrected = corrected_prediction(n, accept_rate, k)
        rows.append({
            "row": sid,
            "beagle_n_proposed": n,
            "measured_raw_p": measured,
            "measured_vs_ours_pct": (measured / OURS_BEAGLE_R - 1.0) * 100.0,
            "r1_predicted_raw_p": r1,
            "r1_predicted_vs_ours_pct": (r1 / OURS_BEAGLE_R - 1.0) * 100.0,
            "r1_sign_correct": (r1 > OURS_BEAGLE_R) == (measured > OURS_BEAGLE_R),
            "corrected_predicted_raw_p": corrected,
            "corrected_vs_ours_pct": (corrected / OURS_BEAGLE_R - 1.0) * 100.0,
            "corrected_sign_correct": (corrected > OURS_BEAGLE_R) == (measured > OURS_BEAGLE_R),
            "is_shallower_than_ours": 4.0 < n < OURS_BEAGLE_N,
        })

    others = [r for r in rows if r["row"] != "ca9251b8"]
    shallow = [r for r in rows if r["is_shallower_than_ours"]]
    return {
        "calibrated_acceptance_rate": accept_rate,
        "calibrated_k": k,
        "rows": rows,
        "r1_sign_correct_count": sum(1 for r in others if r["r1_sign_correct"]),
        "corrected_sign_correct_count": sum(1 for r in others if r["corrected_sign_correct"]),
        "n_compared": len(others),
        "r1_sign_correct_on_shallower": sum(1 for r in shallow if r["r1_sign_correct"]),
        "corrected_sign_correct_on_shallower": sum(1 for r in shallow if r["corrected_sign_correct"]),
        "n_shallower": len(shallow),
    }


# ---------------------------------------------------------------------------
# Where the residual must live, by elimination
# ---------------------------------------------------------------------------

def elimination(accept_rate: float) -> dict:
    """Bound every acceptance-side rescue of the M=6 round, then bound the cost.

    Compare a round that proposes 5 drafts (M=6) against one that proposes 4
    (M=5), with `q` the expected acceptance of the fifth row:

        s/tok(M=6) = T(6) / (1 + a4 + q)      s/tok(M=5) = T(5) / (1 + a4)

    M=6 wins exactly when

        q  >  (1 + a4) * (T(6)/T(5) - 1)

    `q` is a probability, so `q <= 1`. If the right-hand side exceeds 1, then NO
    acceptance-side term can make the wide round win - not a higher average
    acceptance, not the streak path's conditional selection, not a perfect
    draft head. The residual is then forced into the cost term, and the same
    inequality inverts into an upper bound on the ranked width step:

        T(6)/T(5)  <  1 + 1/(1 + a4)
    """
    ranked_rates = reconstruct_rounds()["acceptance_rate_by_prompt"]
    measured_ranked_step = ranked_vs_local_shape()["ranked_step_T6_over_T5"]
    t5, t6 = E33_LADDER_MS[5], E33_LADDER_MS[6]
    measured_step = t6 / t5
    rows = []
    for rate in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, accept_rate, 0.90, 1.00):
        a4 = rate * 4.0
        q_required = (1.0 + a4) * (measured_step - 1.0)
        rows.append({
            "assumed_accept_rate": rate,
            "a4_accepted_from_4_proposals": a4,
            "q_required_for_M6_to_win": q_required,
            "q_is_possible": q_required <= 1.0,
            "max_ranked_step_T6_over_T5": 1.0 + 1.0 / (1.0 + a4),
        })
    a4 = accept_rate * 4.0
    bound = 1.0 + 1.0 / (1.0 + a4)
    # q_required <= 1  <=>  rate <= (1/(step-1) - 1)/4. Above that threshold no
    # probability can rescue the wide round; below it, acceptance can still matter.
    rate_threshold = (1.0 / (measured_step - 1.0) - 1.0) / 4.0
    return {
        "local_ladder_T5_ms": t5,
        "local_ladder_T6_ms": t6,
        "local_step_T6_over_T5": measured_step,
        "sensitivity": rows,
        "acceptance_rate_threshold": rate_threshold,
        "threshold_meaning": (
            "For a per-draft acceptance rate at or above %.4f the required marginal "
            "acceptance exceeds 1, so NO acceptance-side term - higher average "
            "acceptance, streak-path selection, or a perfect head - can make the "
            "M=6 round beat the M=5 round on the local ladder. Below it, the "
            "acceptance side is still in play." % rate_threshold),
        "local_fixture_acceptance_rate_range": [0.276, 0.359],
        "local_measured_rate_reaches_threshold": 0.359 >= rate_threshold,
        "ranked_measured_rates": ranked_rates,
        "ranked_prompts_clearing_threshold": sorted(
            p for p, v in ranked_rates.items() if v >= rate_threshold),
        "central_prompts_clear_threshold": all(
            ranked_rates[p] >= rate_threshold for p in CENTRAL),
        "at_calibrated_rate": {
            "accept_rate": accept_rate,
            "q_required_for_M6_to_win": (1.0 + a4) * (measured_step - 1.0),
            "q_possible": (1.0 + a4) * (measured_step - 1.0) <= 1.0,
            "max_ranked_step_T6_over_T5": bound,
            "local_step_exceeds_bound_by_pct": (measured_step / bound - 1.0) * 100.0,
        },
        "verdict": (
            "CONFIRMED. The threshold is %.4f. Reconstructing our own ranked row "
            "exactly (see reconstruct_rounds) measures per-prompt acceptance of "
            "%.4f-%.4f, clearing the threshold on %d of 8 prompts INCLUDING BOTH "
            "central prompts. On the prompts that decide the median, no "
            "acceptance-side term can rescue the wide round, so the residual is "
            "forced into the cost term and the ranked width step must be below "
            "%.5f. Directly measuring that step from the same reconstruction gives "
            "%.5f, which satisfies the bound."
            % (rate_threshold, min(ranked_rates.values()), max(ranked_rates.values()),
               sum(1 for v in ranked_rates.values() if v >= rate_threshold),
               bound, measured_ranked_step)),
    }


# ---------------------------------------------------------------------------
# MT5: the ranked width ladder, reconstructed exactly from published telemetry
# ---------------------------------------------------------------------------

def _feasible_round_counts(num: int, den: int, non_drafting: int, mtp_ms: float):
    """Every round count consistent with the published integers.

    The parent counts a fixed 512-token window and every round commits exactly
    one primary token, so

        512 = R + accepted_total          accepted_total <= proposed_total = n*R

    `n` is published as an exact rational p/q, so proposed_total = n*R is an
    integer only when R is a multiple of q. That leaves a handful of candidates.
    """
    out = []
    for r in range(den, DECODE_TOKENS + 1, den):
        accepted = DECODE_TOKENS - r
        proposed = num * r // den
        if accepted < 0 or accepted > proposed or r < non_drafting:
            continue
        out.append({
            "rounds": r,
            "proposed_total": proposed,
            "accepted_total": accepted,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "per_round_ms": mtp_ms * DECODE_TOKENS / r,
        })
    return out


def reconstruct_rounds() -> dict:
    """Pin R, the accepted count, and the per-round cost for every ranked prompt.

    Two of the eight prompts reconstruct uniquely with no assumption at all:
    plutarch (its 449 non-drafting rounds force R = 487) and travel. The only
    extra input needed to pin the other six is that per-round cost is
    non-decreasing in mean width, applied as a two-sided bracket between those
    anchors. Nothing here leaves our own submission, so no cross-build or
    cross-machine comparison is involved.
    """
    # The near-zero row first: it is unique with no filtering at all.
    near_zero = []
    for key, (frac, raw, ndr, mtp, ser) in sorted(NEAR_ZERO_ROW["prompts"].items()):
        cands = _feasible_round_counts(frac[0], frac[1], ndr, mtp)
        near_zero.append({
            "prompt": key, "raw_p": raw, "candidates": len(cands),
            "serial_ms": ser, **cands[0],
        })
    k_direct = [r["serial_ms"] / r["per_round_ms"] for r in near_zero]

    rows = []
    for name, (frac, raw, ndr, mtp, ser) in OURS_PER_PROMPT.items():
        cands = _feasible_round_counts(frac[0], frac[1], ndr, mtp)
        rows.append({
            "prompt": name, "raw_p": raw, "mean_width": 1.0 + frac[0] / frac[1],
            "non_drafting": ndr, "mtp_ms_per_token": mtp, "serial_ms_per_token": ser,
            "candidates": cands, "unique_before_filter": len(cands) == 1,
        })
    rows.sort(key=lambda r: r["mean_width"])

    # Enumerate every globally monotone selection: one candidate per prompt whose
    # per-round cost never decreases as mean width grows. If exactly one chain
    # survives, the reconstruction is determined.
    chains = [[]]
    for r in rows:
        nxt = []
        for chain in chains:
            prev = chain[-1]["per_round_ms"] if chain else 0.0
            for c in r["candidates"]:
                if c["per_round_ms"] >= prev - 1e-9:
                    nxt.append(chain + [c])
        chains = nxt
    resolved = []
    for r, c in zip(rows, chains[0] if chains else [x["candidates"][0] for x in rows]):
        survivors = {id(x) for chain in chains for x in chain}
        resolved.append({
            "prompt": r["prompt"], "mean_width": r["mean_width"], "raw_p": r["raw_p"],
            "unique_before_filter": r["unique_before_filter"],
            "candidates_before_filter": len(r["candidates"]),
            "candidates_after_filter": sum(1 for x in r["candidates"] if id(x) in survivors),
            **c,
        })

    # The score is the median of the eight raw ratios; reproducing the published
    # official score to 6 dp validates the whole extraction.
    ordered = sorted(r["raw_p"] for r in resolved)
    median = 0.5 * (ordered[3] + ordered[4])
    central = [r["prompt"] for r in sorted(resolved, key=lambda r: r["raw_p"])][3:5]

    # Least squares per_round = A + B*mean_width over the eight ranked points.
    xs = [r["mean_width"] for r in resolved]
    ys = [r["per_round_ms"] for r in resolved]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)

    return {
        "near_zero_row": {
            "id": NEAR_ZERO_ROW["id"], "solver": NEAR_ZERO_ROW["solver"],
            "rows": near_zero,
            "all_unique": all(r["candidates"] == 1 for r in near_zero),
            "all_zero_accepted": all(r["accepted_total"] == 0 for r in near_zero),
            "candidate_width1_ms_range": [min(r["per_round_ms"] for r in near_zero),
                                          max(r["per_round_ms"] for r in near_zero)],
            "k_measured_range": [min(k_direct), max(k_direct)],
            "k_measured_mean": sum(k_direct) / len(k_direct),
            "meaning": (
                "This submission proposes 18-27 drafts across 512 rounds and has "
                "ZERO accepted, on all eight prompts. Its candidate leg is therefore "
                "serial decoding through the candidate build, and it still scores "
                "1.206-1.212. Any model whose denominator is the candidate's own "
                "width-1 time must score exactly 1.000 here. MT3 is measured, not "
                "inferred."),
        },
        "unique_anchor_prompts": [r["prompt"] for r in resolved if r["unique_before_filter"]],
        "monotone_chains_surviving": len(chains),
        "rows": resolved,
        "median_reproduced": median,
        "official_score": OURS_OFFICIAL_SCORE,
        "median_matches_official": abs(median - OURS_OFFICIAL_SCORE) < 1e-5,
        "central_prompts_reconstructed": central,
        "fit": {"intercept_ms": a, "slope_ms_per_width": b,
                "r_squared": 1.0 - ss_res / ss_tot},
        "acceptance_rate_by_prompt": {r["prompt"]: r["acceptance_rate"] for r in resolved},
        "caveat": (
            "Ranked mean width is the MEAN of an adaptive policy's width "
            "distribution, not a fixed width, so this is a ladder in mean width. "
            "It is directly comparable with the fixed-width local ladder only "
            "under near-linearity, which is itself part of what is being tested. "
            "The claim that survives without that assumption is the absence of a "
            "30 ms cliff between prompts whose mean widths straddle 6."),
    }


def ranked_vs_local_shape() -> dict:
    """Compare the measured ranked ladder with the local one, anchored at plutarch."""
    rec = reconstruct_rounds()
    anchor = next(r for r in rec["rows"] if r["prompt"] == "plutarch")
    ax, ay = anchor["mean_width"], anchor["per_round_ms"]
    la = ladder(ax)
    rows = []
    for r in rec["rows"]:
        if r["prompt"] == "plutarch":
            continue
        ranked_shape = r["per_round_ms"] / ay
        local_shape = ladder(r["mean_width"]) / la
        rows.append({
            "prompt": r["prompt"], "mean_width": r["mean_width"],
            "ranked_per_round_ms": r["per_round_ms"],
            "ranked_shape": ranked_shape,
            "local_shape": local_shape,
            "local_overstates_pct": (local_shape / ranked_shape - 1.0) * 100.0,
        })
    narrow = [r for r in rows if r["mean_width"] < 4.0]
    wide = [r for r in rows if r["mean_width"] >= 5.0]
    b = rec["fit"]["slope_ms_per_width"]
    a = rec["fit"]["intercept_ms"]
    t5, t6 = a + 5 * b, a + 6 * b
    local_step = E33_LADDER_MS[6] / E33_LADDER_MS[5]
    return {
        "anchor": {"prompt": "plutarch", "mean_width": ax, "ranked_ms": ay, "local_ms": la},
        "rows": rows,
        "mean_overstatement_narrow_pct": sum(r["local_overstates_pct"] for r in narrow) / len(narrow),
        "mean_overstatement_wide_pct": sum(r["local_overstates_pct"] for r in wide) / len(wide),
        "ranked_T5_ms": t5,
        "ranked_T6_ms": t6,
        "ranked_step_T6_over_T5": t6 / t5,
        "local_step_T6_over_T5": local_step,
        "local_overstates_step_pct": (local_step / (t6 / t5) - 1.0) * 100.0,
    }


# ---------------------------------------------------------------------------
# MT4: the ladder's absolutes were measured under the wrong command-buffer geometry
# ---------------------------------------------------------------------------

# RuntimeStartupMemoryPolicy, via QwenRuntimeMTPWorker.swift:487
# `guard policy.isLowMemory else { return }`, reported by the advisor
# 2026-08-19T03:59Z. The 48 GiB local box takes the low-memory branch.
GEOMETRY = {
    "local_48gib":  {"max_mb_per_buffer": 128, "max_ops_per_buffer": 64,
                     "cache_limit_gib": 6, "clear_cache_after_warmup": True,
                     "wired_residency": False},
    "ranked_128gib": {"max_mb_per_buffer": 512, "max_ops_per_buffer": 50,
                      "cache_limit_gib": None, "clear_cache_after_warmup": False,
                      "wired_residency": True},
}
# Advisor's traffic census: shipped M=6 mlp.down moves 100.3 MB in 0.4752 ms.
MLP_DOWN_M6_MB = 100.3


def geometry_confound(accept_rate: float) -> dict:
    """The local ladder aliases the weight-pass step with a buffer-flush step.

    At M=6 the wide branch goes to two weight passes, so `mlp.down` moves about
    100.3 MB per call. The local box force-sets `MLX_MAX_MB_PER_BUFFER = 128`,
    so at M=6 roughly ONE such call fits per command buffer, while at M=5 - one
    pass, about half the traffic - two fit. The command-buffer flush boundary
    therefore falls at exactly the same width as the pass-count boundary.

    Under local geometry the two are perfectly confounded and no fit on that
    ladder can separate them. The ranked box runs a 4x larger byte budget, so
    at ranked geometry the flush boundary moves and the two separate.

    This is not a repair of the model. It is the reason its absolutes cannot be
    transferred, and it yields a prediction that can be checked before anyone
    spends a ranked slot.
    """
    local_budget = GEOMETRY["local_48gib"]["max_mb_per_buffer"]
    ranked_budget = GEOMETRY["ranked_128gib"]["max_mb_per_buffer"]
    m5_mb = MLP_DOWN_M6_MB / 2.0          # one weight pass instead of two
    elim = elimination(accept_rate)
    bound = elim["at_calibrated_rate"]["max_ranked_step_T6_over_T5"]
    local_step = elim["local_step_T6_over_T5"]
    return {
        "geometry": GEOMETRY,
        "mlp_down_mb_per_call": {"M5_estimated": m5_mb, "M6_measured": MLP_DOWN_M6_MB},
        "calls_per_command_buffer_local": {
            "M5": math.floor(local_budget / m5_mb),
            "M6": math.floor(local_budget / MLP_DOWN_M6_MB),
        },
        "calls_per_command_buffer_ranked": {
            "M5": math.floor(ranked_budget / m5_mb),
            "M6": math.floor(ranked_budget / MLP_DOWN_M6_MB),
        },
        "confound": ("Under local geometry the buffer-flush boundary and the "
                     "weight-pass boundary both fall at M=6, so the step-vs-smooth "
                     "fit cannot attribute the step to either one."),
        # Two independent routes to the same conclusion.
        "elimination_bound_on_ranked_step": bound,
        "local_measured_step": local_step,
        "local_step_exceeds_bound_by_pct": (local_step / bound - 1.0) * 100.0,
        "prediction": {
            "P1": ("Re-running the width ladder at ranked command-buffer geometry "
                   "(DARKBLOOM_STARTUP_MEMORY_PROFILE=full MLX_MAX_MB_PER_BUFFER=512 "
                   "MLX_MAX_OPS_PER_BUFFER=50) COMPRESSES the M=5->6 step: "
                   "T(6)/T(5) falls below the locally measured %.4f." % local_step),
            "P1_target": ("Quantitatively, the ranked rows require T(6)/T(5) < %.4f, "
                          "so the predicted compression is at least %.2f%%."
                          % (bound, (local_step / bound - 1.0) * 100.0)),
            "P2": ("The widths that change most are M>=6. Widths M=1..5 sit under "
                   "the 128 MB budget on both geometries and should move materially "
                   "less than the M>=6 widths."),
            "falsifier": ("If T(6)/T(5) stays at or above 1.30 under ranked geometry, "
                          "command-buffer geometry is NOT the explanation and the "
                          "residual belongs to threadgroup occupancy or elsewhere."),
        },
        "caution": ("The ~1900-threadgroup occupancy knee was located under local "
                    "geometry. Command-buffer packing and occupancy are both "
                    "dispatch-side, so the knee's location is geometry-dependent and "
                    "is not measured at ranked geometry. It is not a machine constant."),
    }


# ---------------------------------------------------------------------------
# The withdrawn primary metric, recomputed under the corrected model
# ---------------------------------------------------------------------------

def central_pair_under_cap(cap: int, accept_rate: float, k: float) -> dict:
    """Central-pair score with a hard proposed-depth cap of `cap`.

    A hard cap truncates the proposed depth, so mean proposed depth becomes
    min(n, cap) as an upper bound on what the policy can still offer. This is
    deliberately the MOST favourable reading for the cap arm: it assumes the
    policy keeps proposing right up to the cap on every round.
    """
    per_prompt = {}
    for name, (_nt, _rt, n, _r, _nd) in RANKED.items():
        capped_n = min(n, float(cap))
        per_prompt[name] = corrected_prediction(capped_n, accept_rate, k)
    ordered = sorted(per_prompt.values())
    central = 0.5 * (ordered[3] + ordered[4])
    ranked_names = [nm for nm, _ in sorted(per_prompt.items(), key=lambda kv: kv[1])]
    return {
        "cap": cap,
        "per_prompt": per_prompt,
        "central_pair": central,
        "central_prompts": ranked_names[3:5],
    }


def primary_metric() -> dict:
    rep = replay_shallower()
    rate, k = rep["calibrated_acceptance_rate"], rep["calibrated_k"]
    shipped = central_pair_under_cap(8, rate, k)
    cap4 = central_pair_under_cap(4, rate, k)
    # Scale so the shipped arm reproduces the board top, then read the cap arm.
    scale = BOARD_TOP / shipped["central_pair"]
    candidate = cap4["central_pair"] * scale
    observed_central = list(CENTRAL)
    central_ok = sorted(shipped["central_prompts"]) == sorted(observed_central)
    return {
        "name": "e34/predicted_ranked_central_pair_at_best_cap",
        "baseline": BOARD_TOP,
        "shipped_arm_central_pair": shipped["central_pair"],
        "cap4_arm_central_pair": cap4["central_pair"],
        "candidate_rescaled": candidate,
        "delta": candidate - BOARD_TOP,
        "delta_pct": (candidate / BOARD_TOP - 1.0) * 100.0,
        "r1_value": 3.77855631847542,
        "shipped_central_prompts": shipped["central_prompts"],
        "cap4_central_prompts": cap4["central_prompts"],
        "observed_central_prompts": observed_central,
        "central_prompts_correct": central_ok,
        "sigma_score_pct": SIGMA_SCORE_PCT,
        "detection_threshold_2sigma_pct": 2 * SIGMA_SCORE_PCT,
        "status": "WITHDRAWN",
        "withdrawal_reason": (
            "Withdrawn as a decision input. The r1 value %.6f came from a model that "
            "gets 0 of 5 signs right on the shallower declared-head rows. Repairing "
            "MT1 and MT3 fixes magnitude and transfer but leaves the sign record "
            "unchanged at 0 of 5, and the repaired model also mis-orders the central "
            "pair (predicts %s, ranked reality is %s). A median of eight numbers is "
            "decided by exactly the ordering the model gets wrong, so no value from "
            "this family - r1's or the repaired one - should size a ranked decision. "
            "The repaired number %.6f is reported only so the withdrawal is auditable."
            % (3.77855631847542, ", ".join(shipped["central_prompts"]),
               ", ".join(observed_central), candidate)),
    }


def report() -> dict:
    rep = replay_shallower()
    rate = rep["calibrated_acceptance_rate"]
    return {
        "semantics": prove_semantics(),
        "transfer": transfer_falsification(),
        "replay": rep,
        "elimination": elimination(rate),
        "ranked_reconstruction": reconstruct_rounds(),
        "ranked_vs_local_shape": ranked_vs_local_shape(),
        "geometry": geometry_confound(rate),
        "primary_metric": primary_metric(),
        "dropped_phantom_row": DROPPED_PHANTOM_ROW,
        "added_rows_from_cross_check": ["baa75efa", "26d0e934", "a1326b4b"],
        "askeladd_bracket": {
            "bracket": ASKELADD_BRACKET,
            "e34_r1_simulation": E34_R1_SIM_SHARE,
            "inside_bracket": {
                p: ASKELADD_BRACKET[p][0] <= E34_R1_SIM_SHARE[p] <= ASKELADD_BRACKET[p][1]
                for p in ASKELADD_BRACKET
            },
        },
    }


TELEMETRY_CACHE = REPO / ".mlxfast-private" / "ranked-telemetry.json"
DECLARED_HEAD_SHA = "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71"


def verify_telemetry() -> int:
    """Re-derive every embedded telemetry constant from the gitignored cache."""
    if not TELEMETRY_CACHE.exists():
        print("telemetry cache absent at %s; embedded constants not re-verified"
              % TELEMETRY_CACHE)
        return 0
    subs = json.loads(TELEMETRY_CACHE.read_text())["submissions"]
    by_id = {str(s["id"])[:8]: s for s in subs}
    failures = []

    ours = by_id["ca9251b8"]
    if abs(ours["officialScore"] - OURS_OFFICIAL_SCORE) > 1e-12:
        failures.append("official score constant does not match the cache")
    live = sorted((p["effective_mean_draft_len"], p["raw_ratio_of_means"],
                   p["non_drafting_round_count"],
                   round(1000 * p["mtp_seconds_per_token_mean"], 4),
                   round(1000 * p["serial_seconds_per_token_mean"], 4))
                  for p in ours["officialMetrics"]["per_prompt"])
    mine = sorted((f[0] / f[1], raw, ndr, mtp, ser)
                  for f, raw, ndr, mtp, ser in OURS_PER_PROMPT.values())
    for got, want in zip(live, mine):
        if abs(got[0] - want[0]) > 1e-9 or abs(got[1] - want[1]) > 5e-6:
            failures.append(f"OURS_PER_PROMPT row mismatch: cache {got} vs embedded {want}")
        if got[2] != want[2] or abs(got[3] - want[3]) > 1e-3 or abs(got[4] - want[4]) > 1e-3:
            failures.append(f"OURS_PER_PROMPT timing mismatch: cache {got} vs embedded {want}")

    nz = by_id[NEAR_ZERO_ROW["id"]]
    live_nz = sorted(p["effective_mean_draft_len"] for p in nz["officialMetrics"]["per_prompt"])
    mine_nz = sorted(f[0] / f[1] for f, *_ in NEAR_ZERO_ROW["prompts"].values())
    if any(abs(a - b) > 1e-9 for a, b in zip(live_nz, mine_nz)):
        failures.append("NEAR_ZERO_ROW draft lengths do not match the cache")

    phantom = by_id[DROPPED_PHANTOM_ROW["id"]]
    if phantom.get("officialMetrics") is not None:
        failures.append("the dropped row now has officialMetrics and should be reinstated")

    beagle_sha = next(p["prompt_sha256"] for p in ours["officialMetrics"]["per_prompt"]
                      if abs(p["effective_mean_draft_len"] - 485 / 107) < 1e-9)
    for sid, n, raw in DECLARED_HEAD_BEAGLE:
        sub = by_id.get(sid)
        row = next((p for p in (sub.get("officialMetrics") or {}).get("per_prompt", [])
                    if p["prompt_sha256"] == beagle_sha), None) if sub else None
        if row is None:
            failures.append(f"{sid} has no beagle row in the cache")
            continue
        if row["head_provenance_sha256"] != DECLARED_HEAD_SHA:
            failures.append(f"{sid} is not on the declared head")
        if abs(row["effective_mean_draft_len"] - n) > 5e-5 or abs(row["raw_ratio_of_means"] - raw) > 5e-6:
            failures.append("%s beagle mismatch: cache n=%.6f raw=%.6f vs embedded %.4f/%.5f"
                            % (sid, row["effective_mean_draft_len"],
                               row["raw_ratio_of_means"], n, raw))

    print("telemetry verification failures: %d" % len(failures))
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


def self_test() -> int:
    failures = []
    sem = prove_semantics()
    if not sem["tapes"]:
        failures.append("no local tapes found")
    if not sem["all_identities_hold"]:
        failures.append("proposed/accepted integer identities did not close")
    for name, rate in sem["r1_implied_acceptance_rate"].items():
        if abs(rate - 1.0) > 1e-9:
            failures.append(f"r1 implied acceptance rate for {name} is {rate}, expected exactly 1")
    if not sem["plutarch_impossibility"]["impossible"]:
        failures.append("plutarch impossibility test did not fire")
    lo, hi = sem["measured_acceptance_rate_range"]
    if not (0.0 < lo <= hi < 0.6):
        failures.append(f"measured acceptance rate range {lo}-{hi} outside expectation")

    trans = transfer_falsification()
    # The violation is systematic but not universal: the two narrowest prompts
    # have too little drafting for the bound to bite. Both central prompts must.
    if trans["prompts_violating"] < 6:
        failures.append(f"transfer violation on only {trans['prompts_violating']}/{len(RANKED)} prompts")
    for name in CENTRAL:
        row = next(r for r in trans["rows"] if r["prompt"] == name)
        if not row["exceeds_proposed"]:
            failures.append(f"transfer bound does not bite on central prompt {name}")
    if trans["k_lower_bound"] <= 1.0:
        failures.append("k lower bound did not exceed 1")

    rep = replay_shallower()
    elim = elimination(rep["calibrated_acceptance_rate"])
    # The elimination is conditional, not universal: it must hold above the
    # threshold and must NOT be claimed below it.
    thr = elim["acceptance_rate_threshold"]
    for row in elim["sensitivity"]:
        rate = row["assumed_accept_rate"]
        if rate >= thr and row["q_is_possible"]:
            failures.append(
                "rate %.4f is above the threshold %.4f but q=%.3f is still possible"
                % (rate, thr, row["q_required_for_M6_to_win"]))
        if rate < thr and not row["q_is_possible"]:
            failures.append(
                "rate %.4f is below the threshold %.4f but q=%.3f was called impossible"
                % (rate, thr, row["q_required_for_M6_to_win"]))
    if not 0.0 < thr < 1.0:
        failures.append(f"acceptance threshold {thr} is not a usable probability")
    if elim["local_measured_rate_reaches_threshold"]:
        failures.append(
            "local measured acceptance tops out at 0.359, which must sit BELOW the "
            f"threshold {thr:.4f}; the elimination must not be claimed unconditionally")
    if not elim["at_calibrated_rate"]["q_possible"] is False:
        failures.append("elimination does not fire at the calibrated acceptance rate")
    geo = geometry_confound(rep["calibrated_acceptance_rate"])
    if geo["calls_per_command_buffer_local"]["M6"] >= geo["calls_per_command_buffer_local"]["M5"]:
        failures.append("local buffer-flush boundary does not separate M=5 from M=6")
    if geo["local_step_exceeds_bound_by_pct"] <= 0:
        failures.append("local step does not exceed the ranked bound; prediction P1 has no content")
    if abs(r1_prediction(OURS_BEAGLE_N) - OURS_BEAGLE_R) > 1e-9:
        failures.append("r1 replay does not reproduce our own anchor row")
    if abs(corrected_prediction(OURS_BEAGLE_N, rep["calibrated_acceptance_rate"],
                                rep["calibrated_k"]) - OURS_BEAGLE_R) > 1e-6:
        failures.append("corrected model does not reproduce our own anchor row")
    if not 0.0 < rep["calibrated_acceptance_rate"] <= 1.0:
        failures.append("calibrated acceptance rate outside (0, 1]")

    doc = report()
    for name, ok in doc["askeladd_bracket"]["inside_bracket"].items():
        if not ok:
            failures.append(f"E34 r1 simulated share for {name} outside askeladd bracket")

    rec = doc["ranked_reconstruction"]
    if rec["monotone_chains_surviving"] != 1:
        failures.append("%d monotone reconstructions survive; the ladder is not determined"
                        % rec["monotone_chains_surviving"])
    nz = rec["near_zero_row"]
    if not nz["all_unique"]:
        failures.append("near-zero-draft row does not reconstruct uniquely")
    if not nz["all_zero_accepted"]:
        failures.append("near-zero-draft row does not have zero accepted drafts")
    if nz["k_measured_range"][0] <= 1.15:
        failures.append(f"measured k range {nz['k_measured_range']} does not clear 1.15")
    # The reconstruction is only trustworthy if it reproduces the published score.
    if not rec["median_matches_official"]:
        failures.append("reconstructed median %.8f does not match the official score %.8f"
                        % (rec["median_reproduced"], rec["official_score"]))
    if rec["central_prompts_reconstructed"] != list(CENTRAL):
        failures.append("reconstructed central prompts %s != %s"
                        % (rec["central_prompts_reconstructed"], list(CENTRAL)))
    for r in rec["rows"]:
        if r["candidates_after_filter"] != 1:
            failures.append("prompt %s left %d candidates after filtering"
                            % (r["prompt"], r["candidates_after_filter"]))
        if not 0.0 <= r["acceptance_rate"] <= 1.0:
            failures.append(f"prompt {r['prompt']} acceptance rate out of range")
    # MT1 again, now on ranked data: proposed must strictly exceed accepted.
    for r in rec["rows"]:
        if r["accepted_total"] > r["proposed_total"]:
            failures.append(f"prompt {r['prompt']} accepted more than it proposed")

    thr2 = doc["elimination"]["acceptance_rate_threshold"]
    fired = [p for p, v in rec["acceptance_rate_by_prompt"].items() if v >= thr2]
    if not set(CENTRAL) <= set(fired):
        failures.append(
            "the measured ranked acceptance rate does not clear the elimination "
            "threshold on both central prompts, so the elimination stays conditional")

    sh = doc["ranked_vs_local_shape"]
    if sh["local_overstates_step_pct"] <= 0:
        failures.append("measured ranked step is not flatter than the local step; P1 refuted")
    if sh["mean_overstatement_wide_pct"] <= sh["mean_overstatement_narrow_pct"]:
        failures.append("overstatement is not concentrated at wide widths; P2 refuted")

    pm = doc["primary_metric"]
    if pm["status"] != "WITHDRAWN":
        failures.append("primary metric is not marked withdrawn")
    # The third independent failure: the repaired model still picks the wrong
    # pair of prompts to sit at the median.
    if pm["central_prompts_correct"]:
        failures.append(
            "repaired model now orders the central pair correctly; the withdrawal "
            "rationale needs rewriting rather than asserting")
    if rep["corrected_sign_correct_on_shallower"] != 0:
        failures.append(
            "repaired model scores %d/%d on the shallower rows; the headline "
            "'repairs do not fix the sign' no longer holds"
            % (rep["corrected_sign_correct_on_shallower"], rep["n_shallower"]))

    print("self-test failures: %d" % len(failures))
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verify-telemetry", action="store_true")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()
    if args.verify_telemetry:
        return verify_telemetry()
    if args.self_test:
        return self_test() or verify_telemetry()

    doc = report()
    if args.json:
        args.json.write_text(json.dumps(doc, indent=2, sort_keys=True))

    sem = doc["semantics"]
    print("=" * 78)
    print("MT1  effective_mean_draft_len counts PROPOSED drafts, not accepted")
    print("=" * 78)
    print("source: %s" % sem["source"])
    print("%-30s %6s %8s %9s %9s %9s" % ("tape", "rounds", "mean_n", "proposed", "accepted", "accept%"))
    for t in sem["tapes"]:
        print("%-30s %6d %8.3f %9d %9d %8.1f%%" % (
            t["tape"], t["round_count"], t["effective_mean_draft_len"],
            t["proposed_total"], t["accepted_total"], 100 * t["true_acceptance_rate"]))
    print("exact integer identities hold on all tapes: %s" % sem["all_identities_hold"])
    pl = sem["plutarch_impossibility"]
    print("\nplutarch: if n were the ACCEPTED count, rounds = 512/(1+%.4f) = %.2f,"
          % (pl["n"], pl["rounds_implied_if_n_were_accepted"]))
    print("          but the row publishes %d non-drafting rounds -> %.2f drafting rounds. IMPOSSIBLE."
          % (pl["non_drafting_round_count"], pl["drafting_rounds_implied"]))
    print("\nr1 credited (1+n) tokens/round. Implied acceptance rate, per prompt:")
    print("   " + "  ".join("%s=%.6f" % (k, v) for k, v in sem["r1_implied_acceptance_rate"].items()))
    _rk = doc["ranked_reconstruction"]["acceptance_rate_by_prompt"]
    print("   -> exactly 1.000000 everywhere, by construction.")
    print("      local fixture tapes  %.1f%%-%.1f%%" % (
        100 * sem["measured_acceptance_rate_range"][0],
        100 * sem["measured_acceptance_rate_range"][1]))
    print("      RANKED, our own row  %.1f%%-%.1f%%  (reconstructed exactly, see MT5)" % (
        100 * min(_rk.values()), 100 * max(_rk.values())))
    print("      -> the local fixture's acceptance does not transfer either: it sits in")
    print("         the shallow regime, near ranked plutarch (%.1f%%), not near the"
          % (100 * _rk["plutarch"]))
    print("         prompts that decide the median (%s).\n"
          % ", ".join("%s %.1f%%" % (p, 100 * _rk[p]) for p in CENTRAL))
    ph = DROPPED_PHANTOM_ROW
    print("provenance: dropped %s (%s, status=%s) from the advisor's table -- %s"
          % (ph["id"], ph["solver"], ph["status"], ph["reason"]))
    print("            added baa75efa, 26d0e934, a1326b4b: scored declared-head rows")
    print("            the table omitted. All 15 rows re-verified with --verify-telemetry.")

    tr = doc["transfer"]
    print("\n" + "=" * 78)
    print("MT3  the local->ranked transfer implies more accepted than proposed")
    print("=" * 78)
    print("%-10s %8s %10s %10s %12s %10s" % ("prompt", "n", "raw_p", "T(M)/T(1)", "implied a", "a>n?"))
    for r in tr["rows"]:
        print("%-10s %8.4f %10.5f %10.5f %12.4f %10s" % (
            r["prompt"], r["n_proposed"], r["raw_p"], r["ladder_shape_T(M)/T(1)"],
            r["implied_accepted_per_round"], "YES" if r["exceeds_proposed"] else "no"))
    print("violating prompts: %d/%d" % (tr["prompts_violating"], len(tr["rows"])))
    print("pinned serial build must be >= %.4fx slower at width 1 (+%.2f%%)"
          % (tr["k_lower_bound"], tr["k_lower_bound_pct"]))

    rep = doc["replay"]
    print("\n" + "=" * 78)
    print("THE ADVISOR'S TEST  r1 vs corrected, on the declared-head beagle rows")
    print("=" * 78)
    print("calibrated acceptance rate %.4f, k %.4f" % (rep["calibrated_acceptance_rate"], rep["calibrated_k"]))
    print("%-10s %8s %10s %9s %11s %9s %11s %9s" % (
        "row", "n", "measured", "vs ours", "r1 pred", "r1 vs", "corrected", "corr vs"))
    for r in rep["rows"]:
        mark = " <-- ours" if r["row"] == "ca9251b8" else (" *" if r["is_shallower_than_ours"] else "")
        print("%-10s %8.4f %10.5f %+8.2f%% %11.5f %+8.2f%% %11.5f %+8.2f%%%s" % (
            r["row"], r["beagle_n_proposed"], r["measured_raw_p"], r["measured_vs_ours_pct"],
            r["r1_predicted_raw_p"], r["r1_predicted_vs_ours_pct"],
            r["corrected_predicted_raw_p"], r["corrected_vs_ours_pct"], mark))
    print("\nsign agreement with measurement (excluding our anchor row):")
    print("  r1 model        %d/%d   (on the 5 shallower rows: %d/%d)" % (
        rep["r1_sign_correct_count"], rep["n_compared"],
        rep["r1_sign_correct_on_shallower"], rep["n_shallower"]))
    print("  corrected model %d/%d   (on the 5 shallower rows: %d/%d)" % (
        rep["corrected_sign_correct_count"], rep["n_compared"],
        rep["corrected_sign_correct_on_shallower"], rep["n_shallower"]))

    el = doc["elimination"]
    print("\n" + "=" * 78)
    print("ELIMINATION  measured ranked acceptance >= %.4f forces the residual into cost"
          % el["acceptance_rate_threshold"])
    print("=" * 78)
    print("local ladder T(5)=%.3f ms  T(6)=%.3f ms  step=%.5f"
          % (el["local_ladder_T5_ms"], el["local_ladder_T6_ms"], el["local_step_T6_over_T5"]))
    print("THRESHOLD  %.4f   <- above this, no probability rescues the M=6 round"
          % el["acceptance_rate_threshold"])
    print("LOCAL FIXTURE acceptance %.3f-%.3f sits below the threshold: %s"
          % (el["local_fixture_acceptance_rate_range"][0],
             el["local_fixture_acceptance_rate_range"][1],
             "no" if el["local_measured_rate_reaches_threshold"] else "yes"))
    print("%-14s %10s %14s %12s %16s" % (
        "accept rate", "a4", "q required", "q<=1 ?", "max ranked step"))
    for r in el["sensitivity"]:
        print("%-14.4f %10.4f %14.4f %12s %16.5f" % (
            r["assumed_accept_rate"], r["a4_accepted_from_4_proposals"],
            r["q_required_for_M6_to_win"],
            "possible" if r["q_is_possible"] else "IMPOSSIBLE",
            r["max_ranked_step_T6_over_T5"]))
    at = el["at_calibrated_rate"]
    print("=> the ranked step must be < %.5f, but the local ladder measures %.5f"
          % (at["max_ranked_step_T6_over_T5"], el["local_step_T6_over_T5"]))
    print("=> the local ladder overstates the M=5->6 step by at least %.2f%%"
          % at["local_step_exceeds_bound_by_pct"])
    print("VERDICT: %s" % el["verdict"])

    rec = doc["ranked_reconstruction"]
    nz = rec["near_zero_row"]
    print("\n" + "=" * 78)
    print("MT5  the RANKED ladder, reconstructed exactly from published telemetry")
    print("=" * 78)
    print("near-zero-draft row %s (%s): unique on all 8 prompts=%s, zero accepted=%s"
          % (nz["id"], nz["solver"], nz["all_unique"], nz["all_zero_accepted"]))
    print("  candidate build at width 1: %.3f-%.3f ms/round   pinned serial: %.4f ms/token"
          % (nz["candidate_width1_ms_range"][0], nz["candidate_width1_ms_range"][1],
             PINNED_SERIAL_MS["mean_over_8_prompts"]))
    print("  => k MEASURED DIRECTLY = %.4f  (range %.4f-%.4f over 8 prompts)"
          % (nz["k_measured_mean"], nz["k_measured_range"][0], nz["k_measured_range"][1]))
    print("  %s" % nz["meaning"])
    print("\npinned serial leg over %d declared-head rows x 8 prompts: %.4f ms/token,"
          % (PINNED_SERIAL_MS["rows_per_prompt"], PINNED_SERIAL_MS["mean_over_8_prompts"]))
    print("  spread across prompts %.4f ms, within-prompt CV %.4f%%-%.4f%% -> the serial"
          % (PINNED_SERIAL_MS["spread_across_prompts_ms"],
             *PINNED_SERIAL_MS["within_prompt_cv_pct_range"]))
    print("  leg is both pinned and essentially prompt-independent.")
    print("\nour row ca9251b8, exact reconstruction:")
    print("%-10s %7s %9s %6s %9s %9s %8s %11s %6s" % (
        "prompt", "mean_M", "raw_p", "rounds", "proposed", "accepted", "accept%",
        "per_round_ms", "cands"))
    for r in rec["rows"]:
        print("%-10s %7.4f %9.5f %6d %9d %9d %7.2f%% %11.3f %3d->%d" % (
            r["prompt"], r["mean_width"], r["raw_p"], r["rounds"], r["proposed_total"],
            r["accepted_total"], 100 * r["acceptance_rate"], r["per_round_ms"],
            r["candidates_before_filter"], r["candidates_after_filter"]))
    print("unique with no assumption: %s;  globally monotone reconstructions surviving: %d"
          % (rec["unique_anchor_prompts"], rec["monotone_chains_surviving"]))
    print("median of the 8 raw ratios %.8f vs published official score %.8f -> match %s"
          % (rec["median_reproduced"], rec["official_score"], rec["median_matches_official"]))
    print("central prompts reconstructed %s" % rec["central_prompts_reconstructed"])
    print("linear fit per_round = %.3f + %.3f*M ms, R^2 = %.4f"
          % (rec["fit"]["intercept_ms"], rec["fit"]["slope_ms_per_width"],
             rec["fit"]["r_squared"]))
    print("CAVEAT: %s" % rec["caveat"])

    sh = doc["ranked_vs_local_shape"]
    print("\n" + "=" * 78)
    print("P1 VERDICT  measured ranked shape vs the local ladder")
    print("=" * 78)
    print("anchored at %s (mean width %.4f): ranked %.3f ms, local %.3f ms"
          % (sh["anchor"]["prompt"], sh["anchor"]["mean_width"],
             sh["anchor"]["ranked_ms"], sh["anchor"]["local_ms"]))
    print("%-10s %8s %14s %13s %13s %14s" % (
        "prompt", "mean_M", "ranked_ms", "ranked_shape", "local_shape", "local over %"))
    for r in sh["rows"]:
        print("%-10s %8.4f %14.3f %13.4f %13.4f %+13.2f%%" % (
            r["prompt"], r["mean_width"], r["ranked_per_round_ms"],
            r["ranked_shape"], r["local_shape"], r["local_overstates_pct"]))
    print("mean overstatement at mean width < 4 : %+.2f%%" % sh["mean_overstatement_narrow_pct"])
    print("mean overstatement at mean width >= 5: %+.2f%%" % sh["mean_overstatement_wide_pct"])
    print("ranked T(5)=%.3f T(6)=%.3f -> step %.5f;  local step %.5f;  local overstates by %.2f%%"
          % (sh["ranked_T5_ms"], sh["ranked_T6_ms"], sh["ranked_step_T6_over_T5"],
             sh["local_step_T6_over_T5"], sh["local_overstates_step_pct"]))
    el2 = doc["elimination"]
    fired = [p for p, v in rec["acceptance_rate_by_prompt"].items()
             if v >= el2["acceptance_rate_threshold"]]
    print("prompts whose MEASURED ranked acceptance clears the %.4f elimination threshold: %d/8 %s"
          % (el2["acceptance_rate_threshold"], len(fired), sorted(fired)))

    geo = doc["geometry"]
    print("\n" + "=" * 78)
    print("MT4  the ladder's absolutes were measured under the wrong geometry")
    print("=" * 78)
    for box, cfg in geo["geometry"].items():
        print("  %-14s %s" % (box, cfg))
    print("mlp.down per call: M=5 ~%.1f MB, M=6 %.1f MB (measured)"
          % (geo["mlp_down_mb_per_call"]["M5_estimated"], geo["mlp_down_mb_per_call"]["M6_measured"]))
    print("calls per command buffer  local(128MB): M5=%d M6=%d   ranked(512MB): M5=%d M6=%d"
          % (geo["calls_per_command_buffer_local"]["M5"], geo["calls_per_command_buffer_local"]["M6"],
             geo["calls_per_command_buffer_ranked"]["M5"], geo["calls_per_command_buffer_ranked"]["M6"]))
    print("CONFOUND: %s" % geo["confound"])
    print("\nFALSIFIABLE PREDICTION (registered before measurement):")
    for key in ("P1", "P1_target", "P2", "falsifier"):
        print("  %-10s %s" % (key, geo["prediction"][key]))
    print("\nCAUTION: %s" % geo["caution"])

    pm = doc["primary_metric"]
    print("\n" + "=" * 78)
    print("PRIMARY METRIC  %s  -- %s" % (pm["name"], pm["status"]))
    print("=" * 78)
    print("baseline (board top)      %.10f" % pm["baseline"])
    print("r1 value (WITHDRAWN)      %.10f  (%+.1f%%)" % (pm["r1_value"], (pm["r1_value"] / pm["baseline"] - 1) * 100))
    print("corrected model           %.10f  (%+.4f%%)" % (pm["candidate_rescaled"], pm["delta_pct"]))
    print("detection threshold 2s    %.4f%%" % pm["detection_threshold_2sigma_pct"])
    print("central prompts predicted %s" % pm["shipped_central_prompts"])
    print("central prompts OBSERVED  %s   -> model correct: %s"
          % (pm["observed_central_prompts"], pm["central_prompts_correct"]))
    print("central prompts at cap 4  %s" % pm["cap4_central_prompts"])
    print("\n%s" % pm["withdrawal_reason"])

    ab = doc["askeladd_bracket"]
    print("\naskeladd bracket check: %s" % ab["inside_bracket"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
