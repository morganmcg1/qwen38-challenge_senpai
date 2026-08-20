#!/usr/bin/env python3
"""E78 rung 1: is a width-dependent inner-group count bit-exact?

E66 rung 2 proved 12 of 12 that changing `IPG` alone is bit-identical: group
partitions are unordered and `simd_sum` reduces along K WITHIN a row, never
across rows, so moving a row between groups cannot reorder its scalar chain.
E78 changes WHICH partition a given shape gets, not the arithmetic inside it, so
the same argument applies. This rung tests it instead of assuming it.

Arms compared, all against one pinned golden:

  a_ship          the base table
  c_hybrid24928   the cutoff at 24928
  d_hybrid8192    the cutoff at 8192
  c_perturb       positive control: arm C with input rows 3 and 4 swapped in
                  every NA=5 accumulator group. Its ledger MUST differ.

The generic ledger machinery is imported from the E55/E66 instruments rather
than copied, so a fix there reaches this experiment too.

  python3 research/e78_exactness.py --out research/e78-artifacts/exactness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e55_exactness import compare_ledger  # noqa: E402
from e66_exactness import (  # noqa: E402
    closure, negative_controls, read_meta, rows_touching_width, width_census,
)

# The dispatch set each arm must be PROVEN by grep to embed in the worker.
EXPECTED_DISPATCH = {
    "a_ship": {"m5": "5", "m6": "6", "m9": "5", "cutoff": "none", "perturb": "0"},
    "c_hybrid24928": {"m5": "3,5", "m6": "3,6", "m9": "3,5",
                      "cutoff": "24928", "perturb": "0"},
    "d_hybrid8192": {"m5": "3,5", "m6": "3,6", "m9": "3,5",
                     "cutoff": "8192", "perturb": "0"},
    "c_perturb": {"m5": "3,5", "m6": "3,6", "m9": "3,5",
                  "cutoff": "24928", "perturb": "1"},
}


def provenance(metas: dict, names: list[str]) -> dict:
    mlib = "metallib_sha256[.build-worker/release/mlx.metallib]"
    shared = ("golden_sha256", "head_dir", "tokens", "depth")
    workers = {n: metas[n].get("worker_sha256") for n in names}
    dispatch = {
        n: {"m5": metas[n].get("e78_binary_assert_m5_ipgs"),
            "m6": metas[n].get("e78_binary_assert_m6_ipgs"),
            "m9": metas[n].get("e78_binary_assert_m9_ipgs"),
            "cutoff": metas[n].get("e78_binary_assert_cutoff"),
            "perturb": metas[n].get("e78_binary_assert_lane_perturb_copies")}
        for n in names
    }
    out = {
        "arms": names,
        "worker_sha256": workers,
        "all_worker_digests_distinct": len(set(workers.values())) == len(names),
        "git_head": {n: metas[n].get("git_head") for n in names},
        "metallib_source_fingerprint": {
            n: metas[n].get("metallib_source_fingerprint") for n in names},
        "metallib_sha256": {n: metas[n].get(mlib) for n in names},
        # The dispatch set each worker was PROVEN by grep to embed, and to be
        # the only one it embeds. This, not any digest, is the content witness.
        "binary_assert_dispatch": dispatch,
        "dispatch_matches_expected": {
            n: dispatch[n] == EXPECTED_DISPATCH[n] for n in names},
        "worker_text_sha256": {
            n: metas[n].get("e78_binary_assert_worker_text_sha256")
            for n in names},
        "worker_cstring_sha256": {
            n: metas[n].get("e78_binary_assert_worker_cstring_sha256")
            for n in names},
        "worker_text_sha256_used_as_gate": False,
        "fields_that_must_match": {
            k: len({metas[n].get(k) for n in names}) == 1 for k in shared},
        "verify_exit": {n: metas[n].get("verify_exit") for n in names},
    }
    out["arms_provably_distinct_binaries"] = (
        out["all_worker_digests_distinct"]
        and all(out["dispatch_matches_expected"].values())
        and all(out["fields_that_must_match"].values())
        and all(v == "0" for v in out["verify_exit"].values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", default=".mlxfast-private/e78/ledgers")
    ap.add_argument("--base", default="a_ship")
    ap.add_argument("--arms", default="c_hybrid24928,d_hybrid8192")
    ap.add_argument("--positive-control", default="c_perturb")
    ap.add_argument("--out", default="research/e78-artifacts/exactness.json")
    args = ap.parse_args()

    root = pathlib.Path(args.ledgers)
    candidates = [a.strip() for a in args.arms.split(",") if a.strip()]
    timed = [args.base] + candidates

    control_path = root / ("%s.json" % args.positive_control)
    control_meta = read_meta(control_path)
    # The control can fire two ways: a ledger that differs from its arm, or a
    # trusted-parent rejection that writes no ledger at all. The second is the
    # stronger outcome and leaves a 0-byte file.
    control_rejected = (control_path.exists()
                        and control_path.stat().st_size == 0
                        and control_meta.get("verify_exit") not in (None, "0"))
    has_control = control_path.exists() and control_path.stat().st_size > 0
    names = timed + ([args.positive_control] if has_control else [])

    paths = {n: root / ("%s.json" % n) for n in names}
    for n, p in paths.items():
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit("e78_exactness: missing or empty ledger %s" % p)
    payload = {n: json.loads(p.read_text()) for n, p in paths.items()}
    metas = {n: read_meta(p) for n, p in paths.items()}
    digests = {n: hashlib.sha256(p.read_bytes()).hexdigest()
               for n, p in paths.items()}

    against_base = {n: compare_ledger(payload[args.base], payload[n])
                    for n in candidates}

    pos = None
    if control_rejected:
        pos = {
            "arm": args.positive_control,
            "fired": True,
            "mode": "rejected_by_trusted_parent",
            "differs_from_candidate": True,
            "verify_exit": control_meta.get("verify_exit"),
            "lane_perturb_copies":
                control_meta.get("e78_binary_assert_lane_perturb_copies"),
            "worker_sha256": control_meta.get("worker_sha256"),
        }
    elif has_control:
        ref = candidates[0]
        cmp_pos = compare_ledger(payload[ref], payload[args.positive_control])
        pos = {
            "arm": args.positive_control,
            "compared_against": ref,
            "fired": not cmp_pos["identical"],
            "mode": "ledger_differs",
            "rows_at_widths_5_and_9": rows_touching_width(
                payload[ref]["row_ledger"], {5, 9}),
            "comparison": cmp_pos,
        }

    result = {
        "harness": "local",
        "base_arm": args.base,
        "candidate_arms": candidates,
        "ledger_sha256": digests,
        "provenance": provenance(metas, names),
        "closure": {n: closure(payload[n]) for n in names},
        "against_base": against_base,
        "width_census": {n: width_census(payload[n]["row_ledger"])
                         for n in timed},
        "negative_controls": negative_controls(payload[candidates[0]]),
        "positive_control": pos,
    }
    result["passed"] = bool(
        all(result["closure"][n]["closes"] for n in timed)
        and all(against_base[n]["identical"] for n in candidates)
        and result["negative_controls"]["all_fired"]
        and pos is not None and pos["fired"]
        and result["provenance"]["arms_provably_distinct_binaries"])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "passed": result["passed"],
        "identical_to_base": {n: against_base[n]["identical"]
                              for n in candidates},
        "closes": {n: result["closure"][n]["closes"] for n in timed},
        "positive_control_fired": pos["fired"] if pos else None,
        "negative_controls_all_fired":
            result["negative_controls"]["all_fired"],
        "binaries_provably_distinct":
            result["provenance"]["arms_provably_distinct_binaries"],
    }, indent=2))
    print("e78_exactness: wrote %s" % out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
