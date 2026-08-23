#!/usr/bin/env python3
"""E151 rung R1 section 4: which M values reach `affine_qmm_t_nax`.

harness=offline. Nothing here is a timing measurement.

The advisor asks one question: does the reference row `5cdc9c17` gate
(`M >= 128 && M % 128 == 0`) change anything on our tree? To answer it the
model has to reproduce the HOST dispatch chain, because `affine_qmm_t_nax` is
not reached from every quantized matmul:

  QuantizedMatmul::eval_gpu   quantized.cpp:1415  vector_limit = get_qmv_batch_limit(K, N)
                              quantized.cpp:1418  M < vector_limit  -> dispatch_qmv, NOT nax
                              quantized.cpp:1420  transpose && B==1 -> qmm_splitk
  qmm_splitk                  quantized.cpp:790   split_k = max(1, 512 / (ceil(N/32)*ceil(M/32)))
                              quantized.cpp:803   k_align = max(group_size, 32)
                              quantized.cpp:804   split_k = min(split_k, K / k_align)
                              quantized.cpp:807   shrink while K % (split_k*k_align) != 0
                              quantized.cpp:809   split_k <= 1 -> qmm, else the splitk kernel
  qmm                         quantized.cpp:697   is_nax_available() && transpose && K%64==0
                                                  -> qmm_nax -> affine_qmm_t_nax

`get_qmv_batch_limit` returns the same value on `applegpu_g16s` and
`applegpu_g17s`: both take the `else` arm at quantized.cpp:105 (arch_gen is
neither 13 nor 14) and the same `default` case (arch_size 's'). Only
`is_nax_available()` differs between the two hosts, and that flag decides
whether `qmm` enters the NAX family, not which (M, N, K) triples arrive at
`qmm`. So this dispatch model is host-independent and is not an M5 inference.

Two producers of M exist in a scored leg:

  1. The target seed prefill. `Qwen36MTPBlockSession.begin` (:694) feeds the
     whole seed in one call, `MLXArray(seedTokens).reshaped([1, count])`, so
     the target sees exactly one M and it equals the seed length, 512.
  2. The proposal head's committed-history flush. `Qwen36MTPBlockSession`
     :1558-1577 concatenates the lazily primed seed rows, the backlog and the
     current row into ONE head forward, so the head sees M = 1 + backlog, plus
     511 primed seed rows on the first drafting round only. This runs during
     DECODE, not during the seed pass.

Nothing else produces M >= 10 in the leg: the verify block is
`[primary] + drafts`, at most 1 + 8 = 9 rows, which is below every
`vector_limit` these shapes take.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAFETY_CASE = ROOT / "research/e151-r0-safety-case.json"
DEFAULT_TRACE = ROOT / "research/out/e151x512cand/trace.txt"

SEED_TOKENS = 512
MAX_DRAFT_DEPTH = 8

# Target seed-prefill shapes. Same source as the R0.1 table; `evaluated` marks
# the GEMMs the scored seed pass actually evaluates. `lm_head` is false because
# `begin` (:701-707) never evaluates the full-seed projection: it is a dead
# lazy graph and only the last row is projected, at M = 1.
TARGET_SHAPES = [
    ("gdn.in_proj", 5120, 16480, 48, True),
    ("fa.qkv", 5120, 14336, 16, True),
    ("mlp.gate_up", 5120, 34816, 64, True),
    ("lm_head", 5120, 248320, 1, False),
    ("gdn.out_proj", 6144, 5120, 48, True),
    ("fa.o_proj", 6144, 5120, 16, True),
    ("mlp.down", 17408, 5120, 64, True),
]

# Proposal-head shapes, read from the pinned artifact's safetensors header.
# Every one is affine group-64 4-bit (scale columns == K/64) and transposed,
# and every K is a multiple of 64, so all of them are NAX-eligible.
HEAD_SHAPES = [
    ("head.fc", 10240, 5120, 1),
    ("head.q_proj", 5120, 12288, 1),
    ("head.k_proj", 5120, 1024, 1),
    ("head.v_proj", 5120, 1024, 1),
    ("head.o_proj", 6144, 5120, 1),
    ("head.mlp.gate_proj", 5120, 17408, 1),
    ("head.mlp.up_proj", 5120, 17408, 1),
    ("head.mlp.down_proj", 17408, 5120, 1),
]

GROUP_SIZE = 64


def qmv_batch_limit(K: int, N: int) -> int:
    """quantized.cpp:84. The `else` arm, `default` case: g16s and g17s agree."""
    if K <= 2048 and N <= 2048:
        return 18
    if K <= 4096 and N <= 4096:
        return 12
    return 10


def split_k_for(M: int, N: int, K: int) -> int:
    """quantized.cpp:790-808."""
    n_tiles = (N + 31) // 32
    m_tiles = (M + 31) // 32
    split_k = max(1, 512 // (n_tiles * m_tiles))
    k_align = max(GROUP_SIZE, 32)
    split_k = min(split_k, K // k_align)
    while split_k > 1 and K % (split_k * k_align) != 0:
        split_k -= 1
    return split_k


def reaches_nax(M: int, N: int, K: int) -> tuple[bool, str]:
    """Does this (M, N, K) reach `affine_qmm_t_nax` on the ranked host?"""
    if M < qmv_batch_limit(K, N):
        return False, "dispatch_qmv"
    if split_k_for(M, N, K) > 1:
        return False, "qmm_t_splitk"
    if K % 64 != 0:
        return False, "qmm non-nax (K % 64 != 0)"
    return True, "affine_qmm_t_nax"


def flop(M: int, K: int, N: int, layers: int) -> int:
    return 2 * M * K * N * layers


def reconstruct_head_flush_M(trace_path: pathlib.Path) -> dict:
    """Replay the head-history flush schedule from a real round trace.

    `Qwen36MTPBlockSession` grows `headHistoryBacklog*` by one row on every
    non-drafting round (:1501) and by `acceptedCount` rows after a drafting
    round (:1793-1810), and drains the whole backlog into the next drafting
    round's single head forward (:1571-1577). The first drafting round also
    carries `seedTokens.count - 1` primed seed rows (:1558-1566). So the M the
    head presents is fully determined by the (d, acc) sequence.
    """
    text = trace_path.read_text(errors="replace")
    # `mtp-anchor:` mirrors every `mtp-trace:` round line with the same
    # round/d/acc fields, so an unanchored regex double counts the leg.
    rounds = re.findall(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)",
                        text, re.MULTILINE)
    if not rounds:
        raise SystemExit(f"no `mtp-trace: round=` records in {trace_path}")

    # A wrapper pass restarts the round counter at 1. Take the first pass.
    first_pass = []
    for rnd, d, acc in rounds:
        rnd = int(rnd)
        if first_pass and rnd <= first_pass[-1][0]:
            break
        first_pass.append((rnd, int(d), int(acc)))

    backlog = 0
    primed = False
    priming_M = None
    post_prime = []
    for _rnd, d, acc in first_pass:
        if d == 0:
            backlog += 1
            continue
        m = backlog + 1
        if not primed:
            priming_M = m + SEED_TOKENS - 1
            primed = True
        else:
            post_prime.append(m)
        backlog = acc
    return {
        "trace": str(trace_path.relative_to(ROOT)),
        "rounds_in_first_pass": len(first_pass),
        "drafting_rounds": sum(1 for _r, d, _a in first_pass if d > 0),
        "non_drafting_rounds": sum(1 for _r, d, _a in first_pass if d == 0),
        "priming_flush_M": priming_M,
        "post_prime_flush_M_values": post_prime,
    }


def model_low_draft_head_flush_M(rounds: int, draft_every: int, accepted: int) -> dict:
    """A plutarch-like schedule: rare drafting rounds, large backlogs.

    This is a MODEL, not a measurement. The ranked receipts report plutarch at
    449 non-drafting rounds and `effective_mean_draft_len` 0.156, which is the
    only published regime that can produce a head flush between the two
    interesting boundaries: 10, the qmv limit below which the flush never
    reaches a NAX kernel at all, and 128, the reference row's predicate.
    """
    backlog = 0
    primed = False
    priming_M = None
    post_prime = []
    for r in range(1, rounds + 1):
        if r % draft_every != 0:
            backlog += 1
            continue
        m = backlog + 1
        if not primed:
            priming_M = m + SEED_TOKENS - 1
            primed = True
        else:
            post_prime.append(m)
        backlog = accepted
    return {"priming_flush_M": priming_M, "post_prime_flush_M_values": post_prime}


def head_flush_nax_flop(M: int) -> tuple[int, list[str]]:
    """FLOP the head flush at M rows sends to `affine_qmm_t_nax`."""
    total = 0
    hit = []
    for name, K, N, layers in HEAD_SHAPES:
        if reaches_nax(M, N, K)[0]:
            total += flop(M, K, N, layers)
            hit.append(name)
    return total, hit


def low_draft_sensitivity() -> dict:
    """Sweep the plutarch-consistent schedules.

    Constraints from the receipt: 449 non-drafting rounds and
    `effective_mean_draft_len` = drafts / rounds = 0.156 over a 512-token
    window. With `k` drafting rounds the leg has R = 449 + k rounds, emits
    512 = R + accepted tokens, and drafts 0.156 * R times. `k` between 20 and
    40 spans every acceptance rate that closes those three equations.
    """
    arms = []
    for k in (20, 30, 40):
        rounds = 449 + k
        drafts = 0.156 * rounds
        accepted_total = 512 - rounds
        draft_every = max(2, round(rounds / k))
        accepted_per = max(0, round(accepted_total / k))
        modelled = model_low_draft_head_flush_M(rounds, draft_every, accepted_per)
        post_prime = modelled["post_prime_flush_M_values"]
        all_flushes = [modelled["priming_flush_M"]] + post_prime
        reaching = [m for m in all_flushes if head_flush_nax_flop(m)[0] > 0]
        arms.append({
            "drafting_rounds": k,
            "total_rounds": rounds,
            "drafts": round(drafts, 1),
            "accepted_total": accepted_total,
            "draft_every": draft_every,
            "accepted_per_drafting_round": accepted_per,
            "priming_flush_M": modelled["priming_flush_M"],
            "post_prime_flush_M_values": sorted(set(post_prime)),
            "M_reaching_nax": sorted(set(reaching)),
            "nax_flop": sum(head_flush_nax_flop(m)[0] for m in reaching),
            "M_where_reference_predicate_would_not_fire": sorted(
                {m for m in reaching if not (m >= 128 and m % 128 == 0)}),
            "any_M_at_or_above_128": any(m >= 128 for m in reaching),
        })
    return {
        "note": "MODEL, not a measurement. Calibrated to the published "
                "plutarch schedule: 449 non-drafting rounds, "
                "effective_mean_draft_len 0.156.",
        "arms": arms,
        "worst_case_nax_flop": max(a["nax_flop"] for a in arms),
    }


def classify(M: int, shapes: list[tuple[str, int, int, int]]) -> dict:
    rows = []
    for name, K, N, layers in shapes:
        ok, route = reaches_nax(M, N, K)
        rows.append({
            "projection": name,
            "M": M,
            "K": K,
            "N": N,
            "layers": layers,
            "route": route,
            "reaches_affine_qmm_t_nax": ok,
            "flop": flop(M, K, N, layers) if ok else 0,
        })
    return {"rows": rows, "nax_flop": sum(r["flop"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(DEFAULT_TRACE))
    ap.add_argument("--json", default=str(ROOT / "research/e151-r1-m-histogram.json"))
    args = ap.parse_args()

    # ---- 1. the scored seed pass -------------------------------------------
    seed_shapes = [(n, k, nn, l) for n, k, nn, l, ev in TARGET_SHAPES if ev]
    seed = classify(SEED_TOKENS, seed_shapes)
    seed_skipped = [
        {"projection": n, "K": k, "N": nn, "layers": l,
         "reason": "begin(): the full-seed projection is a dead lazy graph; "
                   "only the last row is projected, at M = 1"}
        for n, k, nn, l, ev in TARGET_SHAPES if not ev
    ]
    seed_all_nax = all(r["reaches_affine_qmm_t_nax"] for r in seed["rows"])

    # ---- 2. the head-history flush -----------------------------------------
    trace = reconstruct_head_flush_M(pathlib.Path(args.trace))
    head_prime_M = SEED_TOKENS
    head_prime = classify(head_prime_M, HEAD_SHAPES)

    # Everything the head flushes after priming, on the real local schedule.
    post_prime = trace["post_prime_flush_M_values"]
    trace["post_prime_max_M"] = max(post_prime) if post_prime else 0
    trace["post_prime_reaching_nax"] = sorted({
        m for m in post_prime if head_flush_nax_flop(m)[0] > 0
    })

    sensitivity = low_draft_sensitivity()
    worst = max(sensitivity["arms"], key=lambda a: a["nax_flop"])

    # ---- 3. mass ------------------------------------------------------------
    # Two legs, not one. A heavy-drafting prompt and a plutarch-like prompt run
    # different schedules, so their NAX mass is not additive. Each leg gets its
    # own denominator.
    seed_flop = seed["nax_flop"]
    head_prime_flop = head_prime["nax_flop"]
    modelled_flop = worst["nax_flop"]

    observed_total = seed_flop + head_prime_flop
    modelled_total = seed_flop + modelled_flop
    histogram = [
        {"leg": "observed local heavy-drafting schedule (beagle/essays-like)",
         "M": SEED_TOKENS, "producer": "target seed prefill",
         "channel": "prefill", "occurrences": 1,
         "nax_flop": seed_flop,
         "share_of_leg_nax_flop": seed_flop / observed_total,
         "M_mod_128": SEED_TOKENS % 128,
         "reference_predicate_fires": True},
        {"leg": "observed local heavy-drafting schedule (beagle/essays-like)",
         "M": head_prime_M, "producer": "proposal-head history priming",
         "channel": "decode", "occurrences": 1,
         "nax_flop": head_prime_flop,
         "share_of_leg_nax_flop": head_prime_flop / observed_total,
         "M_mod_128": head_prime_M % 128,
         "reference_predicate_fires": True},
        {"leg": "observed local heavy-drafting schedule (beagle/essays-like)",
         "M": trace["post_prime_reaching_nax"] or "none",
         "producer": "proposal-head backlog flush",
         "channel": "decode",
         "occurrences": len(trace["post_prime_reaching_nax"]),
         "nax_flop": 0,
         "share_of_leg_nax_flop": 0.0,
         "M_mod_128": [],
         "reference_predicate_fires": None},
        {"leg": "modelled plutarch-like low-drafting schedule",
         "M": SEED_TOKENS, "producer": "target seed prefill",
         "channel": "prefill", "occurrences": 1,
         "nax_flop": seed_flop,
         "share_of_leg_nax_flop": seed_flop / modelled_total,
         "M_mod_128": SEED_TOKENS % 128,
         "reference_predicate_fires": True},
        {"leg": "modelled plutarch-like low-drafting schedule",
         "M": worst["M_reaching_nax"] or "none",
         "producer": "proposal-head priming and backlog flushes",
         "channel": "decode",
         "occurrences": len(worst["M_reaching_nax"]),
         "nax_flop": modelled_flop,
         "share_of_leg_nax_flop": modelled_flop / modelled_total,
         "M_mod_128": sorted({m % 128 for m in worst["M_reaching_nax"]}),
         "reference_predicate_fires": False},
    ]

    distinct_scored_M = sorted({SEED_TOKENS})
    all_observed_M = (
        distinct_scored_M
        + [trace["priming_flush_M"]]
        + trace["post_prime_flush_M_values"]
        + [a["priming_flush_M"] for a in sensitivity["arms"]]
        + [m for a in sensitivity["arms"] for m in a["post_prime_flush_M_values"]]
    )
    m511_present = 511 in all_observed_M

    verdict = {
        "e151_r1_scored_seed_M_values": distinct_scored_M,
        "e151_r1_scored_seed_M_all_multiple_of_128": all(
            m % 128 == 0 for m in distinct_scored_M),
        "e151_r1_seed_pass_all_reach_nax": seed_all_nax,
        "e151_r1_M511_occurs_anywhere": m511_present,
        "e151_r1_prefill_share_of_nax_flop_observed_leg": seed_flop / observed_total,
        "e151_r1_prefill_share_of_nax_flop_worst_modelled_leg":
            seed_flop / modelled_total,
        "e151_r1_predicate_is_noop_on_prefill_channel": True,
        "e151_r1_predicate_affects_decode_flop_share_worst_modelled_leg":
            modelled_flop / modelled_total,
    }

    decision = {
        "registered_before_any_rerun": True,
        "choice": "DO NOT add the M >= 128 && M % 128 == 0 predicate",
        "reasons": [
            "The r2 request set the closing criterion itself: `If every "
            "scored M is a multiple of 128 after all, say so in one line and "
            "section 4 closes.` Every scored M IS a multiple of 128. The "
            "scored seed pass presents exactly ONE M, 512 = 4 * 128, on every "
            "evaluated projection, because `begin` feeds the whole seed in a "
            "single call. The reference predicate is therefore TRUE on the "
            "entire prefill channel and both arms retile identically there.",
            "M = 511 never occurs anywhere in a scored leg. The 511 rows in "
            "the accepted R0.1 table are the hypothetical ragged tail the "
            "assignment asked me to enumerate, not a shape the leg presents. "
            "That row's `evaluated_in_scored_seed_prefill` field was "
            "mislabelled: it describes the projection, not the (M, "
            "projection) pair. This corrects the section 4 premise that both "
            "shapes are scored.",
            "Because the predicate is a provable no-op on the prefill "
            "channel, it cannot change the -4.9721 % claim in either "
            "direction. Adding it buys no reproduction fidelity where the "
            "measured effect lives.",
            "Adding it would break the byte-identity that section 4 lists as "
            "Required. A predicate edits `quantized_nax.h` and its twin, so "
            "the submitted surface would no longer be `.h 50cf7876` / "
            "`.cpp e7c55209`, and the tree the advisor merges would again be "
            "a tree no gate chain has seen. Under a six-hour shelf life that "
            "is the higher-variance option, not the lower one.",
            "A runtime predicate is not free here. The arm is a compile-time "
            "template selection, so gating it at runtime means instantiating "
            "BOTH tile shapes inside one kernel and branching between them. "
            "That grows the kernel and can move register allocation and "
            "instruction-cache pressure on a kernel this fleet cannot "
            "execute. Campaign rule from E147: never price a kernel change "
            "from an AIR delta. The cost is unmeasurable offline while the "
            "prefill benefit is provably zero.",
        ],
        "residual_exposure_i_accept": {
            "where": "proposal-head committed-history flush, DECODE channel "
                     "only, on low-drafting prompts",
            "detail": "On a plutarch-like schedule the head priming flush "
                      "carries M in 523..534. Those M are >= 128 but not "
                      "multiples of 128, so the reference row falls back to "
                      "the host 64x64 tile there and our arm fires. This is "
                      "the RULE 158 shape and I am naming it rather than "
                      "hiding it.",
            "bound": "at most 2.6880 % of that leg's NAX FLOP in the worst "
                     "modelled schedule, on prompts whose Rule 148 weight is "
                     "0.0000",
            "on_median_deciding_prompts": "zero. The observed local 512-token "
                                          "trace drafts on all 82 rounds, so "
                                          "the priming flush is M = 512 (the "
                                          "predicate fires) and every later "
                                          "flush is M <= 8, below the qmv "
                                          "limit of 10, so it never reaches a "
                                          "NAX kernel at all.",
            "correctness": "not a fidelity risk in either arm. R0.2 proves "
                           "exact tile coverage at these M with three failing "
                           "controls, and R0.3 proves the K accumulation "
                           "order is unchanged, so the retile is bit-identical "
                           "arithmetic.",
        },
        "what_would_reverse_it": [
            "A receipt whose decode channel reads outside +/-0.15 pp with the "
            "prefill channel moving as predicted. That would localise the cost "
            "to the head flush and the predicate becomes the repair.",
            "Evidence that a median-deciding prompt runs long non-drafting "
            "streaks, which would push its head flushes above M = 10.",
            "A change to `begin` that splits the seed into chunks. That would "
            "put more than one M on the prefill channel and reopen the "
            "question.",
        ],
    }

    payload = {
        "experiment": "E151",
        "rung": "R1 section 4",
        "harness": "offline",
        "dispatch_model": {
            "qmv_batch_limit_for_scored_shapes": {
                f"{name} K={K} N={N}": qmv_batch_limit(K, N)
                for name, K, N, _l, _e in TARGET_SHAPES
            },
            "host_independent": True,
            "why": "get_qmv_batch_limit takes the same branch on g16s and "
                   "g17s; only is_nax_available() differs, and that decides "
                   "the family qmm enters, not which (M,N,K) reach qmm.",
        },
        "seed_pass": {"M": SEED_TOKENS, **seed, "not_evaluated": seed_skipped},
        "head_priming": {"M": head_prime_M, **head_prime},
        "head_flush_from_real_trace": trace,
        "head_flush_low_draft_model": sensitivity,
        "e151_r1_scored_m_histogram": histogram,
        "verdict": verdict,
        "registered_gating_decision": decision,
    }

    out = pathlib.Path(args.json)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("=== e151_r1_scored_m_histogram (harness=offline) ===")
    for row in histogram:
        print(f"  M={row['M']!s:<48} {row['channel']:<8} "
              f"share={row['share_of_leg_nax_flop']:.6f} "
              f"ref_predicate_fires={row['reference_predicate_fires']}")
    print()
    for k, v in verdict.items():
        print(f"  {k} = {v}")
    print()
    print(f"  local trace: {trace['drafting_rounds']} drafting rounds, "
          f"{trace['non_drafting_rounds']} non-drafting, "
          f"post-prime max head-flush M = {trace['post_prime_max_M']}, "
          f"post-prime flushes reaching nax = {trace['post_prime_reaching_nax']}")
    for arm in sensitivity["arms"]:
        print(f"  low-draft model k={arm['drafting_rounds']:>3}: "
              f"prime M {arm['priming_flush_M']}, backlog M {arm['post_prime_flush_M_values']} -> "
              f"reaching nax {arm['M_reaching_nax']}, "
              f"any M >= 128: {arm['any_M_at_or_above_128']}")
    print()
    print(f"  DECISION: {decision['choice']}")
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
