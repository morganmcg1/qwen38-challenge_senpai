"""E147 F8 follow-up: answer the four percent, with no GPU.

F8 asks two questions about a +4.40 % candidate-leg regression on a change that
was supposed to touch only the 512-token seed prefill, which is 8.45 % of the
leg. A +2.18 % prefill regression explains +0.18 %. The other +4.2 % has no
declared mechanism.

This script settles the parts that are decidable from source and from the
public board.

  1. THE FLOAT32 NAX THREADGROUP HYPOTHESIS. Rung E-0 reported that a naive
     doubling of the float32 transposed NAX weight tile would need 34,816 B
     against a 32,768 B budget. Rung B never shipped that naive doubling: the
     allocation goes through NAXWsStagingPlan, which drops back to a single
     tile whenever the doubled pair would not fit, and carries a static_assert
     for the case the offline compiler does not check. This enumerates every
     instantiated transposed NAX shape and reports the plan's decision and byte
     count against the base, so "a pipeline creation failed on the ranked
     runner" is either demonstrated or removed.

  2. IS THE RANKED PAIR TREE-MATCHED? F8 compares e003a86d with 7226dc9a and
     calls the pair schedule-matched and head-matched, which it is. That does
     not make it tree-matched. If the two rows were built from different
     campaign commits, the +4.40 % contains every intervening merge and not
     only rungs A and B. This prints the source reference of every row of ours
     so the comparison can be repaired or confirmed.

  usage: research/e147_f8_followup.py [--fetch] [--json out.json]
"""
import argparse
import json
import os
import re
import statistics
import urllib.request

BENCHMARK = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
BOARD = (
    f"https://api.yukon.org/api/benchmarks/{BENCHMARK}/submissions?all=true"
)
CACHE = "research/out/e147-board.json"

NAX_METAL = (
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.metal"
)
NAX_H = (
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
)

MAX_TGP_BYTES = 32768
# Every instantiate_quantized_aligned_batched(affine_qmm_t_nax, ...) in the
# .metal uses 64, 64, 64, 2, 2. The element type set comes from
# instantiate_quantized_types.
NAX_TRANSPOSED_TILE = (64, 64, 64)  # BM, BK, BN
ELEMENT_TYPES = {"float": 4, "float16_t": 2, "bfloat16_t": 2}

OUR_ROWS = ("7226dc9a", "e003a86d", "1db9d63e", "572b2cc4")


def staging_plan(bn, bk, elem_bytes):
    """Replay NAXWsStagingPlan<T, BN, BK> exactly."""
    bk_padded = bk + 16 // elem_bytes
    tile = bn * bk_padded
    pipelined = 2 * tile * elem_bytes <= MAX_TGP_BYTES
    elements = 2 * tile if pipelined else tile
    return {
        "bk_padded": bk_padded,
        "tile_elements": tile,
        "single_bytes": tile * elem_bytes,
        "pipelined": pipelined,
        "elements": elements,
        "bytes": elements * elem_bytes,
        "fits": elements * elem_bytes <= MAX_TGP_BYTES,
    }


def threadgroup_table():
    bm, bk, bn = NAX_TRANSPOSED_TILE
    rows = []
    for name, size in ELEMENT_TYPES.items():
        plan = staging_plan(bn, bk, size)
        rows.append(
            {
                "element_type": name,
                "element_bytes": size,
                "BM": bm,
                "BK": bk,
                "BN": bn,
                "base_bytes": plan["single_bytes"],
                "candidate_bytes": plan["bytes"],
                "doubled": plan["pipelined"],
                "fits_budget": plan["fits"],
                "naive_double_bytes": 2 * plan["single_bytes"],
                "naive_double_would_fit": 2 * plan["single_bytes"] <= MAX_TGP_BYTES,
            }
        )
    return rows


def source_checks():
    nax_h = open(NAX_H, encoding="utf-8").read()
    metal = open(NAX_METAL, encoding="utf-8").read()
    shift_dst_defs = len(re.findall(r"void shift_dst\(const int delta\)", nax_h))
    plan_sites = len(re.findall(r"NAXWsStagingPlan<T, BN, BK>::elements", nax_h))
    raw_double = len(re.findall(r"threadgroup T Ws\[2 \* BN \* BK_padded\]", nax_h))
    return {
        "shift_dst_definitions_in_quantized_nax_h": shift_dst_defs,
        "shift_dst_call_sites": len(re.findall(r"loader_w\.shift_dst\(", nax_h)),
        "staging_plan_allocation_sites": plan_sites,
        "unguarded_doubled_allocations": raw_double,
        "static_assert_present": "must fit in threadgroup memory" in nax_h,
        "transposed_instantiations_are_64_64_64": bool(
            re.search(
                r"instantiate_quantized_aligned_batched\(affine_qmm_t_nax, type, "
                r"group_size, bits, 64, 64, 64, 2, 2",
                metal,
            )
        ),
    }


def fetch_board():
    token = os.environ.get("YUKON_API_TOKEN")
    if not token:
        raise SystemExit("YUKON_API_TOKEN is required with --fetch")
    req = urllib.request.Request(BOARD, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        data = json.load(fh)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as out:
        json.dump(data, out)
    return data


def load_board(fetch):
    if fetch or not os.path.exists(CACHE):
        return fetch_board()
    return json.load(open(CACHE, encoding="utf-8"))


def rows_of(board):
    if isinstance(board, dict):
        for key in ("submissions", "data", "items", "results"):
            if isinstance(board.get(key), list):
                return board[key]
    return board if isinstance(board, list) else []


def prefill_values(row):
    blob = json.dumps(row)
    return [float(v) for v in re.findall(r'"prefill_seconds_per_token":\s*([0-9.eE+-]+)', blob)]


def source_ref(row):
    for key in (
        "sourceReference",
        "source_reference",
        "sourceRef",
        "commit",
        "commitSha",
        "baseSha",
    ):
        if row.get(key):
            return str(row[key])
    blob = json.dumps(row)
    m = re.search(r'"(?:sourceReference|source_reference|commit[A-Za-z]*)":\s*"([0-9a-f]{7,40})"', blob)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    report = {
        "threadgroup_table": threadgroup_table(),
        "source_checks": source_checks(),
    }

    tg = report["threadgroup_table"]
    report["e147_max_threadgroup_bytes"] = max(r["candidate_bytes"] for r in tg)
    report["every_instantiated_shape_fits"] = all(r["fits_budget"] for r in tg)
    report["float32_nax_pipeline_disposition"] = (
        "single-buffered; NAXWsStagingPlan refuses the doubling at "
        f"{[r for r in tg if r['element_type'] == 'float'][0]['naive_double_bytes']} B "
        "and keeps the base allocation, so no pipeline creation can fail"
    )

    print("=== instantiated affine_qmm_t_nax shapes (BM=BK=BN=64, WM=WN=2)")
    print(
        f"{'type':<12}{'base B':>9}{'cand B':>9}{'doubled':>9}"
        f"{'fits':>7}{'naive x2 B':>12}{'x2 fits':>9}"
    )
    for r in tg:
        print(
            f"{r['element_type']:<12}{r['base_bytes']:>9}{r['candidate_bytes']:>9}"
            f"{str(r['doubled']):>9}{str(r['fits_budget']):>7}"
            f"{r['naive_double_bytes']:>12}{str(r['naive_double_would_fit']):>9}"
        )
    print()
    for k, v in report["source_checks"].items():
        print(f"{k:<45} {v}")
    print()

    board = load_board(args.fetch)
    ours = []
    for row in rows_of(board):
        rid = str(row.get("id", ""))
        short = rid[:8]
        if short not in OUR_ROWS:
            continue
        pref = prefill_values(row)
        ours.append(
            {
                "id": short,
                "status": row.get("status"),
                "score": row.get("officialScore") or row.get("score"),
                "source_ref": source_ref(row),
                "created": row.get("createdAt"),
                "prefill_n": len(pref),
                "prefill_mean": statistics.fmean(pref) if pref else None,
                "prefill_cv_pct": (
                    100 * statistics.stdev(pref) / statistics.fmean(pref)
                    if len(pref) > 1
                    else None
                ),
            }
        )
    ours.sort(key=lambda r: r["created"] or "")
    report["our_rows"] = ours

    print("=== our rows, in creation order")
    print(f"{'id':<10}{'status':<12}{'score':>14}{'source_ref':<44}{'pref mean':>13}{'CV%':>8}")
    for r in ours:
        mean = "" if r["prefill_mean"] is None else f"{r['prefill_mean']:.9f}"
        cv = "" if r["prefill_cv_pct"] is None else f"{r['prefill_cv_pct']:.4f}"
        print(
            f"{r['id']:<10}{str(r['status']):<12}{str(r['score']):>14}"
            f"{str(r['source_ref']):<44}{mean:>13}{cv:>8}"
        )

    refs = {r["source_ref"] for r in ours if r["source_ref"]}
    report["our_rows_share_one_source_ref"] = len(refs) <= 1
    print()
    print(f"distinct source refs across our rows: {sorted(refs)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
