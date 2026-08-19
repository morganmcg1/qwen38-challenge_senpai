#!/usr/bin/env python3
"""E57 rung 0: which SDPA routes does the scored decode actually reach?

Consumes the `e57-sdpa:` lines that `attentionWithCacheUpdate` writes under
MLX_E57_SDPA_TRACE=1 and reports, per leg of one local run:

  * the joint histogram of (qL, kL bucket) over the outer decode calls, where
    the buckets are the trusted dispatcher's own 2-pass boundary kL < 1024 and
    kL >= 1024;
  * the number of calls with qL >= 6 AND kL >= 1024, which is the only cell in
    which the threadgroup limit that motivated the chunk can bind;
  * the route each SDPA dispatch takes, derived from the quoted conditions in
    mlx/backend/metal/scaled_dot_product_attention.cpp rather than measured, and
    the requested threads per threadgroup on the 2-pass route.

Route derivation, all line numbers from that file at this base:

    :685   vector mode iff q.shape(2) <= 8, else full self attention (steel)
    :746   2-pass iff ((devc == 'd' || devc == 's') && kL >= 1024)
                    || (kv_heads < q_heads && kL >= 4096)
    :484   2-pass group_dims(32, gqa_factor, qL) = 32 * gqa_factor * qL threads
    :358   1-pass group_dims(1024, 1, 1), independent of qL and gqa_factor

The legs are separated by process id, because the wrapper spawns one worker per
leg and they append to one shared trace file in wall-clock order.

usage:
    research/e57_route_histogram.py research/out/e57-armA/trace.txt \
        [--devc s] [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

CALL = re.compile(
    r"^e57-sdpa: pid=(\d+) qL=(\d+) kL=(\d+) b=(\d+) causal=(\d) "
    r"arm=(\w+) chunk=(\d) reason=(\w+)"
)

Q_HEADS = 24
KV_HEADS = 4
GQA_FACTOR = Q_HEADS // KV_HEADS
MAX_THREADS_PER_THREADGROUP = 1024
TWO_PASS_KL = 1024
TWO_PASS_GQA_KL = 4096
VECTOR_QL_LIMIT = 8
MAX_VECTOR_QL_GQA = 32
FULL_ATTENTION_LAYERS = 16


def route(q_len: int, k_len: int, devc: str) -> tuple[str, int]:
    """Return (route name, requested threads per threadgroup).

    MEASURED CORRECTION (rung 1). `ScaledDotProductAttention::use_fallback`
    (:634-637) admits the fused vector primitive only when

        qL <= 8 && qL <= kL && head_dim in {64, 96, 128, 256}
                 && qL * gqa_factor <= 32

    and admits the fused full primitive (:631) only when

        qL > 8 && head_dim in {64, 80, 128}.

    Our head dimension is 256, so the full primitive is NEVER available at any
    width, and `qL * 6 <= 32` caps the vector primitive at `qL <= 5`. Every call
    at `qL >= 6` therefore leaves the fused primitive entirely for MLX's
    composed fallback: a materialised causal mask, two `steel_gemm_fused`
    dispatches and `block_softmax_precise`. The dispatcher condition at :685 is
    reached only after `use_fallback` returns false, which is why the earlier
    reading of it was wrong. Confirmed by the dispatch counter at qL 6..9 and by
    the unsplit qL=6 call at kL=1030, which returns instead of raising
    `Maximum threads per threadgroup`.
    """
    if q_len * GQA_FACTOR > MAX_VECTOR_QL_GQA or q_len > VECTOR_QL_LIMIT:
        # Composed fallback: mask build + steel_gemm_fused nt/nn +
        # block_softmax_precise, threadgroup (32, 2, 2), width independent.
        return "composed_fallback", 32 * 2 * 2
    two_pass = (devc in ("d", "s") and k_len >= TWO_PASS_KL) or (
        KV_HEADS < Q_HEADS and k_len >= TWO_PASS_GQA_KL
    )
    if two_pass:
        return "vector_2pass", 32 * GQA_FACTOR * q_len
    return "vector_1pass", MAX_THREADS_PER_THREADGROUP


def boundary(leg: list[dict]) -> dict:
    """The 2-pass `blocks` split the advisor asked for.

    `blocks` is function constant 26 and belongs to the 2-pass route only. On
    `devc == 's'` it is 64, promoted to 128 when `N > 1024 && n_simds > 4`. With
    `n_simds = gqa_factor * qL >= 6` always, the promotion reduces to
    `kL > 1024` STRICTLY. So `kL == 1024` and `kL >= 1025` select DIFFERENT
    pipeline objects, and a warm that lands on exactly 1024 does not warm the
    pipeline a longer window uses.
    """
    hot = [call for call in leg if call["kL"] >= TWO_PASS_KL]
    at_boundary = [call for call in hot if call["kL"] == TWO_PASS_KL]
    above = [call for call in hot if call["kL"] > TWO_PASS_KL]
    first_index = next(
        (index for index, call in enumerate(leg) if call["kL"] >= TWO_PASS_KL), None)
    per_width: dict[str, dict[str, int]] = {}
    for call in hot:
        entry = per_width.setdefault(
            f"qL={call['qL']}", {"kl_eq_1024": 0, "kl_ge_1025": 0})
        entry["kl_eq_1024" if call["kL"] == TWO_PASS_KL else "kl_ge_1025"] += 1
    return {
        "kl_boundary": {
            "calls_kl_eq_1024": len(at_boundary),
            "calls_kl_ge_1025": len(above),
            "first_kl_at_or_above_1024":
                leg[first_index]["kL"] if first_index is not None else None,
            "first_ql_at_or_above_1024":
                leg[first_index]["qL"] if first_index is not None else None,
            # One round issues one call per full-attention layer, so the layer
            # count converts the remaining calls into remaining rounds.
            "calls_from_first_boundary_to_end":
                len(leg) - first_index if first_index is not None else 0,
            "rounds_from_first_boundary_to_end":
                (len(leg) - first_index) / FULL_ATTENTION_LAYERS
                if first_index is not None else 0,
            "per_width": per_width,
            "blocks_64_calls": len(at_boundary),
            "blocks_128_calls": len(above),
        }
    }


def segments(call: dict) -> list[tuple[int, int, str]]:
    """The SDPA dispatches one attentionWithCacheUpdate call issues."""
    q_len, k_len = call["qL"], call["kL"]
    if not call["chunk"]:
        return [(q_len, k_len, "whole")]
    split = 5
    return [(split, k_len - (q_len - split), "chunkA"), (q_len - split, k_len, "chunkB")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--devc", default="s", help="architecture letter of the host")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    calls: list[dict] = []
    for line in args.trace.read_text(errors="replace").splitlines():
        match = CALL.match(line)
        if not match:
            continue
        calls.append(
            {
                "pid": int(match.group(1)),
                "qL": int(match.group(2)),
                "kL": int(match.group(3)),
                "b": int(match.group(4)),
                "causal": match.group(5) == "1",
                "arm": match.group(6),
                "chunk": match.group(7) == "1",
                "reason": match.group(8),
            }
        )
    if not calls:
        print(f"e57_route_histogram: no e57-sdpa lines in {args.trace}", file=sys.stderr)
        return 1

    order: list[int] = []
    for call in calls:
        if call["pid"] not in order:
            order.append(call["pid"])

    record: dict = {
        "trace": str(args.trace),
        "devc": args.devc,
        "gqa_factor": GQA_FACTOR,
        "total_calls": len(calls),
        "arms_seen": sorted({call["arm"] for call in calls}),
        "legs": [],
    }

    for index, pid in enumerate(order):
        leg = [call for call in calls if call["pid"] == pid]
        joint: dict[str, int] = collections.Counter()
        widths: dict[int, int] = collections.Counter()
        reasons: dict[str, int] = collections.Counter()
        routes: dict[str, int] = collections.Counter()
        illegal: dict[str, int] = collections.Counter()
        pairs: set[tuple[int, int]] = set()
        wide_hot = 0
        dispatches = 0
        for call in leg:
            bucket = "kL<1024" if call["kL"] < TWO_PASS_KL else "kL>=1024"
            joint[f"qL={call['qL']} {bucket}"] += 1
            widths[call["qL"]] += 1
            reasons[call["reason"]] += 1
            pairs.add((call["qL"], call["kL"]))
            if call["qL"] >= 6 and call["kL"] >= TWO_PASS_KL:
                wide_hot += 1
            for q_len, k_len, _tag in segments(call):
                name, threads = route(q_len, k_len, args.devc)
                routes[name] += 1
                dispatches += 1
                if threads > MAX_THREADS_PER_THREADGROUP:
                    illegal[f"{name} qL={q_len} threads={threads}"] += 1

        record["legs"].append(
            {
                "leg_index": index,
                "pid": pid,
                "calls": len(leg),
                "sdpa_dispatches_derived": dispatches,
                "width_histogram": dict(sorted(widths.items())),
                "joint_histogram": dict(sorted(joint.items())),
                "chunk_reason_histogram": dict(reasons),
                "route_histogram": dict(routes),
                "illegal_threadgroup_requests": dict(illegal),
                "calls_ql_ge6_and_kl_ge1024": wide_hot,
                **boundary(leg),
                "distinct_ql_kl_pairs": len(pairs),
                "kl_range": [min(c["kL"] for c in leg), max(c["kL"] for c in leg)],
                "ql_kl_pairs_ql_ge6": sorted(
                    {(q, k) for q, k in pairs if q >= 6}
                ),
            }
        )

    totals = {
        "calls_ql_ge6_and_kl_ge1024": sum(
            leg["calls_ql_ge6_and_kl_ge1024"] for leg in record["legs"]
        ),
        "illegal_threadgroup_requests": sum(
            sum(leg["illegal_threadgroup_requests"].values()) for leg in record["legs"]
        ),
    }
    record["totals"] = totals

    print(json.dumps(record, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
