#!/usr/bin/env python3
"""E61 exactness gate: does the single-weight-stream M=6 cell change any row?

E55's instrument defines three row populations (see its docstring). E61 ran only
PATH C, which is the strongest of the three and the only one that is at once
ON-PATH and LOGIT-LEVEL: `mtp-verify --golden` runs the candidate session with
`retainLedger: true`, so every row the wide multi-row dispatch actually
evaluated carries its own `top2_tokens` and `top2_logits`.

PATH A and PATH B are absent here on purpose. They were not skipped to save
time; PATH C strictly dominates them for this gate, and a ledger leg is untimed,
so three ledger legs cost less than one timed ABBA session.

Reused verbatim from E55: `compare_ledger` and `ulp_gap`. E61 adds only the
provenance gate, the window/EOS closure check, the per-width attribution, and
negative controls driven by the REAL 567-row ledger instead of a synthetic pair.

  python3 research/e61_exactness.py --out research/e61-exactness.json
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

from e55_exactness import compare_ledger, ulp_gap  # noqa: E402

EOS_TOKEN_ID = 248044
DECODE_TOKENS = 512

# The arm patch driver commits each arm, so an arm's tree is addressable. base
# and base2 are the same arm and must therefore carry the SAME tree; m6 must
# carry a different one. This is a source-level witness that does not depend on
# any build artefact being reproducible.
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


def provenance(metas: dict, names: tuple[str, str, str]) -> dict:
    control, candidate, drift = names
    mc, mk, md = metas[control], metas[candidate], metas[drift]
    shared = ("golden_sha256", "head_dir", "tokens", "depth")
    mlib = "metallib_sha256[.build-worker/release/mlx.metallib]"

    workers = {n: metas[n].get("worker_sha256") for n in names}
    trees = {n: git_tree(metas[n].get("git_head", "")) for n in names}

    out = {
        "arms": list(names),
        "worker_sha256": workers,
        "all_worker_digests_distinct": len(set(workers.values())) == 3,
        "git_head": {n: metas[n].get("git_head") for n in names},
        "git_tree": trees,
        # Source-level identity, independent of build reproducibility.
        "control_and_drift_share_a_tree": (
            trees[control] is not None and trees[control] == trees[drift]),
        "candidate_tree_differs": (
            trees[candidate] is not None and trees[candidate] != trees[control]),
        "metallib_source_fingerprint": {
            n: metas[n].get("metallib_source_fingerprint") for n in names},
        "metallib_sha256": {n: metas[n].get(mlib) for n in names},
        "candidate_metallib_differs": mk.get(mlib) != mc.get(mlib),
        "control_and_drift_metallib_match": md.get(mlib) == mc.get(mlib),
        # The dispatch each worker was PROVEN by grep to embed, and to be the
        # only one it embeds. This, not any digest, is the content witness.
        "binary_assert_m6_na": {
            n: metas[n].get("e61_binary_assert_m6_na") for n in names},
        "binary_assert_m9_na": {
            n: metas[n].get("e61_binary_assert_m9_na") for n in names},
        "binary_assert_wide_bound": {
            n: metas[n].get("e61_binary_assert_wide_bound") for n in names},
        "fields_that_must_match": {
            k: mc.get(k) == mk.get(k) == md.get(k) for k in shared},
        "verify_exit": {n: metas[n].get("verify_exit") for n in names},
    }

    # __TEXT,__text is recorded but is NOT a gate: base and base2 have the same
    # git tree yet different __text digests, so the section is not reproducible
    # across relinks on this toolchain and cannot witness source identity in
    # either direction.
    texts = {n: metas[n].get("e61_binary_assert_worker_text_sha256") for n in names}
    out["worker_text_sha256"] = texts
    out["worker_text_sha256_is_reproducible"] = (
        texts[control] == texts[drift] if out["control_and_drift_share_a_tree"] else None)
    out["worker_text_sha256_used_as_gate"] = False

    out["candidate_dispatch_is_live"] = (
        out["binary_assert_m6_na"][control] == "3"
        and out["binary_assert_m6_na"][candidate] == "6"
        and out["binary_assert_m6_na"][drift] == "3"
        and all(out["binary_assert_m9_na"][n] == "5" for n in names))
    out["arms_provably_distinct_binaries"] = (
        out["all_worker_digests_distinct"]
        and out["candidate_dispatch_is_live"]
        and out["candidate_metallib_differs"]
        and out["control_and_drift_metallib_match"]
        and out["control_and_drift_share_a_tree"]
        and out["candidate_tree_differs"]
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
    return {
        "round_count": len(rounds),
        "rounds_by_width": dict(sorted(widths.items())),
        "rows_by_width": dict(sorted(rows_by_width.items())),
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
    """Perturb the REAL 567-row ledger; every case must be caught."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", default=".mlxfast-private/e61/ledgers")
    ap.add_argument("--arms", default="base,m6,base2",
                    help="control,candidate,drift-control")
    ap.add_argument("--out", default="research/e61-exactness.json")
    args = ap.parse_args()

    names = tuple(a.strip() for a in args.arms.split(","))
    if len(names) != 3:
        raise SystemExit("--arms needs exactly three names")
    control, candidate, drift = names

    root = pathlib.Path(args.ledgers)
    paths = {n: root / ("%s.json" % n) for n in names}
    payload = {n: json.loads(p.read_text()) for n, p in paths.items()}
    metas = {n: read_meta(p) for n, p in paths.items()}
    digests = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}

    prov = provenance(metas, names)
    cand = compare_ledger(payload[control], payload[candidate])
    null = compare_ledger(payload[control], payload[drift])
    neg = negative_controls(payload[control])
    clo = {n: closure(payload[n]) for n in names}
    cen = width_census(payload[candidate]["row_ledger"])

    report = {
        "experiment": "qwen38-r1-e61-single-weight-stream-qmv-m6",
        "path": "PATH C only: on-path and logit-level (mtp-verify --golden row_ledger)",
        "paths_a_b_absent": (
            "PATH A is off-path (M=1 reference rows) and PATH B is argmax-level; "
            "PATH C strictly dominates both for this gate"),
        "tokens": DECODE_TOKENS,
        "depth": 8,
        "whole_file_sha256": digests,
        "all_three_ledgers_byte_identical": len(set(digests.values())) == 1,
        "provenance": prov,
        "candidate_vs_control": cand,
        "null_control_vs_drift_control": null,
        "row_ledger_closure": clo,
        "width_census_candidate": cen,
        "negative_controls": neg,
        "coverage_limits": [
            "M=3 was never scheduled in this trajectory, so no row exercises the "
            "M=3 cell; widths covered are %s." % cen["widths_exercised"],
            "One public fixture and one 512-token trajectory. This is exactness "
            "evidence for the rows this trajectory produced, not a proof over "
            "all inputs.",
            "The M=6 cell is the only cell E61 changes; it carries %d of %d "
            "rounds here." % (cen["rounds_by_width"].get(6, 0), cen["round_count"]),
        ],
    }

    gates = {
        "all_three_ledgers_byte_identical": report["all_three_ledgers_byte_identical"],
        "candidate_identical_to_control": cand["identical"],
        "max_abs_ulp_top2_logits_is_zero": cand["max_abs_ulp_top2_logits"] == 0,
        "drift_control_identical": null["identical"],
        "negative_controls_all_fired": neg["all_fired"],
        "arms_provably_distinct_binaries": prov["arms_provably_distinct_binaries"],
        "candidate_dispatch_is_live": prov["candidate_dispatch_is_live"],
        "every_row_ledger_closes": all(c["closes"] for c in clo.values()),
        "m6_actually_exercised": cen["rounds_by_width"].get(6, 0) > 0,
    }
    report["gates"] = gates
    report["verdict"] = "EXACT" if all(gates.values()) else "FAILED"

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for k, v in gates.items():
        print("%-42s %s" % (k, "PASS" if v else "FAIL"))
    print("\nwidths (rounds): %s" % cen["rounds_by_width"])
    print("rows: %d  accepted: %d  rejected: %d"
          % (clo[candidate]["rows"], clo[candidate]["accepted_draft_total"],
             clo[candidate]["rejected_draft_total"]))
    print("verdict: %s -> %s" % (report["verdict"], args.out))
    return 0 if report["verdict"] == "EXACT" else 7


if __name__ == "__main__":
    raise SystemExit(main())
