#!/usr/bin/env python3
"""E66 rung 2: is the composed `t55` + `t6` QMV surface bit-exact?

Reuses E61's instrument, which is PATH C of E55's three row populations: it is at
once ON-PATH and LOGIT-LEVEL, because `mtp-verify --golden` runs the candidate
session with `retainLedger: true`, so every row the wide multi-row dispatch
actually evaluated carries its own `top2_tokens` and `top2_logits`.

E66 compares three timed arms plus one positive control:

  a  `<T,5,3>` `<T,6,3>`   pre-t6 shipped table
  b  `<T,5,3>` `<T,6,6>`   the merged base
  c  `<T,5,5>` `<T,6,6>`   the candidate
  c_perturb                arm C with input rows 3 and 4 swapped in every NA=5
                           accumulator group. The ledger MUST differ from `c`.

E61's negative controls perturbed the real ledger in memory. They are kept, and
`c_perturb` adds the missing half: a control that is compiled, dispatched and
measured end to end, so the instrument is proven to see a real change inside the
NA=5 helper body that `t55` introduces.

  python3 research/e66_exactness.py --out research/e66-exactness.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e55_exactness import compare_ledger  # noqa: E402

EOS_TOKEN_ID = 248044
DECODE_TOKENS = 512

# The dispatch each arm must be PROVEN by grep to embed in the worker binary.
EXPECTED_DISPATCH = {
    "a": {"m5": "3", "m6": "3", "m9": "5", "bound": "5", "perturb": "0"},
    "b": {"m5": "3", "m6": "6", "m9": "5", "bound": "6", "perturb": "0"},
    "c": {"m5": "5", "m6": "6", "m9": "5", "bound": "6", "perturb": "0"},
    "c_perturb": {"m5": "5", "m6": "6", "m9": "5", "bound": "6", "perturb": "1"},
}


def git_tree(sha: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "%s^{tree}" % sha],
            capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def read_meta(path: pathlib.Path) -> dict:
    meta = path.with_name(path.stem + "-meta.txt")
    out = {}
    if meta.exists():
        for line in meta.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def provenance(metas: dict, names: list[str]) -> dict:
    mlib = "metallib_sha256[.build-worker/release/mlx.metallib]"
    shared = ("golden_sha256", "head_dir", "tokens", "depth")
    workers = {n: metas[n].get("worker_sha256") for n in names}
    trees = {n: git_tree(metas[n].get("git_head", "")) for n in names}

    dispatch = {n: {"m5": metas[n].get("e66_binary_assert_m5_na"),
                    "m6": metas[n].get("e66_binary_assert_m6_na"),
                    "m9": metas[n].get("e66_binary_assert_m9_na"),
                    "bound": metas[n].get("e66_binary_assert_wide_bound"),
                    "perturb": metas[n].get(
                        "e66_binary_assert_lane_perturb_copies")}
                for n in names}

    out = {
        "arms": names,
        "worker_sha256": workers,
        "all_worker_digests_distinct": len(set(workers.values())) == len(names),
        "git_head": {n: metas[n].get("git_head") for n in names},
        "git_tree": trees,
        "all_git_trees_distinct": len(set(trees.values())) == len(names),
        "metallib_source_fingerprint": {
            n: metas[n].get("metallib_source_fingerprint") for n in names},
        "metallib_sha256": {n: metas[n].get(mlib) for n in names},
        "all_metallib_digests_distinct": len(
            {metas[n].get(mlib) for n in names}) == len(names),
        # The dispatch each worker was PROVEN by grep to embed, and to be the
        # only one it embeds. This, not any digest, is the content witness.
        "binary_assert_dispatch": dispatch,
        "dispatch_matches_expected": {
            n: dispatch[n] == EXPECTED_DISPATCH[n] for n in names},
        # __TEXT,__text alone is not a content witness (ledger 202(I)). Both
        # sections are recorded so a reader can see which one moved.
        "worker_text_sha256": {
            n: metas[n].get("e66_binary_assert_worker_text_sha256") for n in names},
        "worker_cstring_sha256": {
            n: metas[n].get("e66_binary_assert_worker_cstring_sha256") for n in names},
        "worker_text_sha256_used_as_gate": False,
        "fields_that_must_match": {
            k: len({metas[n].get(k) for n in names}) == 1 for k in shared},
        "verify_exit": {n: metas[n].get("verify_exit") for n in names},
    }
    out["arms_provably_distinct_binaries"] = (
        out["all_worker_digests_distinct"]
        and out["all_git_trees_distinct"]
        and out["all_metallib_digests_distinct"]
        and all(out["dispatch_matches_expected"].values())
        and all(out["fields_that_must_match"].values())
        and all(v == "0" for v in out["verify_exit"].values()))
    return out


def width_census(ledger: list[dict]) -> dict:
    rounds = sorted({r["round"] for r in ledger})
    widths = collections.Counter()
    rows_by_width = collections.Counter()
    for rnd in rounds:
        rows = [r for r in ledger if r["round"] == rnd]
        m = 1 + sum(1 for r in rows if r["kind"] == "draft")
        widths[m] += 1
        rows_by_width[m] += len(rows)
    total = len(rounds)
    return {
        "round_count": total,
        "rounds_by_width": dict(sorted(widths.items())),
        "rows_by_width": dict(sorted(rows_by_width.items())),
        "round_share_by_width": {m: c / total for m, c in sorted(widths.items())},
        "widths_exercised": sorted(widths),
        "widths_not_exercised": [m for m in range(2, 10) if m not in widths],
    }


def closure(payload: dict) -> dict:
    led = payload["row_ledger"]
    drafts = sum(1 for r in led if r["kind"] == "draft")
    toks = [r["token"] for r in led]
    eos = [i for i, t in enumerate(toks) if t == EOS_TOKEN_ID]
    return {
        "rows": len(led),
        "declared_rows_total": payload["declared_rows_total"],
        "reference_checked_row_total": payload["reference_checked_row_total"],
        "round_count": payload["round_count"],
        "rows_equal_declared": len(led) == payload["declared_rows_total"],
        "declared_equals_rounds_plus_drafts": (
            payload["round_count"] + drafts == payload["declared_rows_total"]),
        "every_row_reference_checked": (
            payload["reference_checked_row_total"] == payload["declared_rows_total"]),
        "rejected_rows_reference_checked": payload["rejected_rows_reference_checked"],
        "accepted_draft_total": payload["accepted_draft_total"],
        "rejected_draft_total": payload["rejected_draft_total"],
        "all_tokens_matched": payload["all_tokens_matched"],
        "parity_all_ok": payload["parity_all_ok"],
        "residual_divergence_count": payload["residual_divergence_count"],
        "emitted_token_total": payload["emitted_token_total"],
        "decode_token_count": payload["decode_token_count"],
        "window_closed_at_512": payload["decode_token_count"] == DECODE_TOKENS,
        "eos_row_indices": eos,
        "eos_present_in_ledger": bool(eos),
        "rows_after_first_eos": (len(led) - eos[0] - 1) if eos else 0,
        "reference_checked_by": sorted({r["reference_checked_by"] for r in led}),
        "kinds": sorted({r["kind"] for r in led}),
        "closes": (
            len(led) == payload["declared_rows_total"]
            and payload["round_count"] + drafts == payload["declared_rows_total"]
            and payload["reference_checked_row_total"] == payload["declared_rows_total"]
            and payload["all_tokens_matched"] is True
            and payload["parity_all_ok"] is True
            and payload["residual_divergence_count"] == 0
            and payload["decode_token_count"] == DECODE_TOKENS),
    }


def negative_controls(real: dict) -> dict:
    """Perturb the REAL ledger in memory; every case must be caught."""
    cases = {}

    def perturb(fn):
        b = copy.deepcopy(real)
        fn(b["row_ledger"])
        return compare_ledger(real, b)

    r = perturb(lambda L: L[300].__setitem__(
        "top2_logits", [math.nextafter(L[300]["top2_logits"][0], math.inf),
                        L[300]["top2_logits"][1]]))
    cases["top2_logit_one_ulp_at_row_300"] = (
        r["field_mismatch_counts"]["top2_logits"] == 1
        and r["max_abs_ulp_top2_logits"] == 1 and not r["identical"])

    r = perturb(lambda L: L[100].__setitem__("token", L[100]["token"] + 1))
    cases["token_flipped_at_row_100"] = (
        r["field_mismatch_counts"]["token"] == 1 and not r["identical"])

    r = perturb(lambda L: L[200].__setitem__("accepted", not L[200]["accepted"]))
    cases["acceptance_flipped_at_row_200"] = (
        r["field_mismatch_counts"]["accepted"] == 1 and not r["identical"])

    r = perturb(lambda L: L[400].__setitem__(
        "top2_tokens", [L[400]["top2_tokens"][1], L[400]["top2_tokens"][0]]))
    cases["top2_token_order_swapped_at_row_400"] = (
        r["field_mismatch_counts"]["top2_tokens"] == 1 and not r["identical"])

    r = perturb(lambda L: L[50].__setitem__(
        "reference_checked_by", "verify_block_replay"))
    cases["reference_source_changed_at_row_50"] = (
        r["field_mismatch_counts"]["reference_checked_by"] == 1 and not r["identical"])

    r = perturb(lambda L: L.pop())
    cases["truncated_ledger"] = not r["identical"]

    cases["empty_ledger_is_not_a_pass"] = not compare_ledger({}, {})["identical"]
    return {"cases": cases, "all_fired": all(cases.values())}


def rows_touching_width(ledger: list[dict], widths: set[int]) -> int:
    rounds = collections.defaultdict(list)
    for row in ledger:
        rounds[row["round"]].append(row)
    n = 0
    for rows in rounds.values():
        m = 1 + sum(1 for r in rows if r["kind"] == "draft")
        if m in widths:
            n += len(rows)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", default=".mlxfast-private/e66/ledgers")
    ap.add_argument("--arms", default="a,b,c")
    ap.add_argument("--positive-control", default="c_perturb")
    ap.add_argument("--out", default="research/e66-exactness.json")
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",")]
    if len(names) != 3:
        raise SystemExit("--arms needs exactly three names")
    a, b, c = names
    root = pathlib.Path(args.ledgers)

    control_path = root / ("%s.json" % args.positive_control)
    # The control can fire two ways. It can produce a ledger that differs from
    # arm C, or the trusted parent can reject the run outright and write no
    # ledger at all. The second is the stronger outcome and leaves a 0-byte file.
    control_meta = read_meta(control_path)
    control_rejected = (control_path.exists()
                        and control_path.stat().st_size == 0
                        and control_meta.get("verify_exit") not in (None, "0"))
    has_control = control_path.exists() and control_path.stat().st_size > 0
    all_names = names + ([args.positive_control] if has_control else [])

    paths = {n: root / ("%s.json" % n) for n in all_names}
    payload = {n: json.loads(p.read_text()) for n, p in paths.items()}
    metas = {n: read_meta(p) for n, p in paths.items()}
    digests = {n: hashlib.sha256(p.read_bytes()).hexdigest()
               for n, p in paths.items()}

    prov = provenance(metas, all_names)
    c_vs_b = compare_ledger(payload[b], payload[c])
    c_vs_a = compare_ledger(payload[a], payload[c])
    b_vs_a = compare_ledger(payload[a], payload[b])
    neg = negative_controls(payload[c])
    clo = {n: closure(payload[n]) for n in all_names}
    cen = width_census(payload[c]["row_ledger"])

    timed_digests = {n: digests[n] for n in names}
    pos = None
    if control_rejected:
        pos = {
            "arm": args.positive_control,
            "fired": True,
            "mode": "rejected_by_trusted_parent",
            "differs_from_candidate": True,
            "verify_exit": control_meta.get("verify_exit"),
            "lane_perturb_copies":
                control_meta.get("e66_binary_assert_lane_perturb_copies"),
            "worker_sha256": control_meta.get("worker_sha256"),
            "metallib_source_fingerprint":
                control_meta.get("metallib_source_fingerprint"),
            "note": ("The trusted parent raised a contract violation and wrote "
                     "no ledger, so the perturbed kernel could not even produce "
                     "a candidate row set. NA=5 groups occur at M=5 (the t55 "
                     "cell) and as the first group of the M=9 cell, so this "
                     "control proves the ledger reads the NA=5 helper body, "
                     "not that it isolates M=5."),
        }
    elif has_control:
        pos_cmp = compare_ledger(payload[c], payload[args.positive_control])
        touched = rows_touching_width(payload[c]["row_ledger"], {5, 9})
        pos = {
            "arm": args.positive_control,
            "differs_from_candidate": not pos_cmp["identical"],
            "whole_file_sha256_differs":
                digests[args.positive_control] != digests[c],
            "field_mismatch_counts": pos_cmp["field_mismatch_counts"],
            "max_abs_ulp_top2_logits": pos_cmp["max_abs_ulp_top2_logits"],
            "rows_in_na5_bearing_rounds": touched,
            "note": ("NA=5 groups occur at M=5 (the t55 cell) and as the first "
                     "group of the M=9 cell, so this control proves the ledger "
                     "reads the NA=5 helper body, not that it isolates M=5."),
        }

    report = {
        "experiment": "qwen38-r1-e66-composition-certification",
        "rung": 2,
        "path": "PATH C only: on-path and logit-level (mtp-verify --golden row_ledger)",
        "tokens": DECODE_TOKENS,
        "depth": 8,
        "whole_file_sha256": digests,
        "all_three_ledgers_byte_identical": len(set(timed_digests.values())) == 1,
        "provenance": prov,
        "candidate_vs_merged_base_c_vs_b": c_vs_b,
        "candidate_vs_pre_t6_c_vs_a": c_vs_a,
        "merged_base_vs_pre_t6_b_vs_a": b_vs_a,
        "row_ledger_closure": clo,
        "width_census_candidate": cen,
        "negative_controls": neg,
        "positive_control": pos,
        "coverage_limits": [
            "Widths covered by this trajectory: %s. Widths never scheduled: %s."
            % (cen["widths_exercised"], cen["widths_not_exercised"]),
            "One public fixture and one 512-token trajectory. This is exactness "
            "evidence for the rows this trajectory produced, not a proof over "
            "all inputs.",
            "The M=5 cell carries %d of %d rounds here and the M=6 cell %d."
            % (cen["rounds_by_width"].get(5, 0), cen["round_count"],
               cen["rounds_by_width"].get(6, 0)),
        ],
    }

    gates = {
        "all_three_ledgers_byte_identical": report["all_three_ledgers_byte_identical"],
        "candidate_identical_to_merged_base": c_vs_b["identical"],
        "candidate_identical_to_pre_t6": c_vs_a["identical"],
        "max_abs_ulp_top2_logits_is_zero_c_vs_b":
            c_vs_b["max_abs_ulp_top2_logits"] == 0,
        "max_abs_ulp_top2_logits_is_zero_c_vs_a":
            c_vs_a["max_abs_ulp_top2_logits"] == 0,
        "negative_controls_all_fired": neg["all_fired"],
        "arms_provably_distinct_binaries": prov["arms_provably_distinct_binaries"],
        "every_row_ledger_closes": all(clo[n]["closes"] for n in names),
        "m5_actually_exercised": cen["rounds_by_width"].get(5, 0) > 0,
        "m6_actually_exercised": cen["rounds_by_width"].get(6, 0) > 0,
        "eos_inside_window_and_512_tokens_emitted": all(
            clo[n]["eos_present_in_ledger"] and clo[n]["window_closed_at_512"]
            for n in names),
    }
    if pos is not None:
        gates["positive_control_differs"] = pos["differs_from_candidate"]
    report["gates"] = gates
    report["verdict"] = "EXACT" if all(gates.values()) else "FAILED"

    pathlib.Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    for k, v in gates.items():
        print("%-46s %s" % (k, "PASS" if v else "FAIL"))
    print("\nledger sha256:")
    for n in all_names:
        print("  %-10s %s" % (n, digests[n]))
    print("\nrounds by width: %s" % cen["rounds_by_width"])
    print("round share by width: %s"
          % {m: round(s, 6) for m, s in cen["round_share_by_width"].items()})
    print("rows: %d declared %d reference-checked %d  accepted %d rejected %d"
          % (clo[c]["rows"], clo[c]["declared_rows_total"],
             clo[c]["reference_checked_row_total"],
             clo[c]["accepted_draft_total"], clo[c]["rejected_draft_total"]))
    print("EOS %d at ledger row(s) %s; rows after the first EOS: %d; "
          "decode_token_count %d"
          % (EOS_TOKEN_ID, clo[c]["eos_row_indices"], clo[c]["rows_after_first_eos"],
             clo[c]["decode_token_count"]))
    print("verdict: %s -> %s" % (report["verdict"], args.out))
    return 0 if report["verdict"] == "EXACT" else 7


if __name__ == "__main__":
    raise SystemExit(main())
