#!/usr/bin/env python3
"""E55 exactness: does the NA=5 dispatch at M=9 change any target row?

The advisor's hard stop is a BITWISE delta at any dispatched width M <= 9, not at
M=9 alone: zeroing lane 4 of the NA=5 accumulator moved lane 1 by 8 ulp, so the
NA=5 instantiation can re-schedule arithmetic inside helpers that every width
inlines. An instrument for that gate must therefore observe rows that the WIDE
multi-row dispatch actually produced.

The run tree holds two different row populations, and only the second one is
produced by the code this experiment changed:

  PATH A  `02-mtp-verify-output.json` -- the reference golden. Stage 02 is
          `mtp-verify --generate`, and `Qwen36MTPReferenceSession.step` feeds ONE
          token per forward (`[1, 1]`). These 513 rows are M=1. They do NOT
          exercise `case 9: qmv_fast_crossrow_affine4_g64_m<T, 9, NA, true>`.
          Comparing them is still worth doing -- it proves the candidate did not
          perturb the sequential path and that every arm was judged against a
          byte-identical reference -- but on its own it is VACUOUS for the gate.

  PATH B  `04-mtp-timed.json` -- the candidate's own speculative decode, widths
          M = draft_len + 1 over {2, 4, 5, 6, 7, 8, 9}. The per-row top-2 is not
          retained by `mtp-timed` (`retainLedger: false`), so the cross-arm
          reading here is ARGMAX-level and indirect: acceptance per round is
          decided by comparing the wide dispatch's argmax against the golden, so
          if any of the 567 wide rows had changed argmax, the accepted count for
          that round would move and the per-round draft-length trajectory would
          diverge from that round on. An identical elementwise trajectory plus
          identical accepted/rejected totals is a strong argmax-level statement
          about every wide row.

  PATH C  optional, and the only DIRECT bitwise reading of the wide rows.
          `mtp-verify --golden` (no `--generate`) runs the candidate with
          `retainLedger: true` and emits `row_ledger` carrying each wide row's
          `top2_tokens` and `top2_logits`. The benchmark pipeline never invokes
          that mode, so those files exist only if they were produced on purpose.
          Pass `--ledger-base` / `--ledger-candidate` to compare them.

Reported honestly: A is logit-level but off-path, B is on-path but argmax-level,
C is on-path and logit-level. Only A and B are available from a timed run.

  python3 research/e55_exactness.py --out research/e55-exactness.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import pathlib
import struct

RUNS = pathlib.Path(".mlxfast-private/e55/runs")
ARMS = ("base", "m9two", "base2")
LEGS = ("leg-1", "leg-2")
ROW_FIELDS = ("sequential_argmax", "top1_logit", "top2_logits", "top2_tokens")
LEDGER_FIELDS = ("top2_tokens", "top2_logits", "token", "reference_token",
                 "reference_margin", "reference_checked_by", "accepted", "kind",
                 "round", "draft_index")
EOS_TOKEN_ID = 248044


def width_of_draft_len(draft_len: int) -> int:
    """A round of `d` drafts is checked in one forward of the primary plus `d`."""
    return draft_len + 1


def ulp_gap(a: float, b: float) -> int:
    """Signed float64 ulp distance, so a mismatch is reported with a magnitude."""
    if a == b:
        return 0
    ia, ib = (struct.unpack("<q", struct.pack("<d", x))[0] for x in (a, b))
    if ia < 0:
        ia = -9223372036854775808 - ia
    if ib < 0:
        ib = -9223372036854775808 - ib
    return ib - ia


def read(arm: str, leg: str, stem: str) -> dict:
    return json.loads((RUNS / arm / "reports" / leg / stem).read_text())


# --- PATH A: the M=1 reference golden ---------------------------------------


def compare_golden(a: dict, b: dict) -> dict:
    out = {
        "row_population": "reference golden, M=1 sequential forward",
        "covers_changed_dispatch": False,
        "emitted_token_count": [len(a["emitted_tokens"]), len(b["emitted_tokens"])],
        "row_count": [len(a["rows"]), len(b["rows"])],
        "seed_token_count": [len(a["seed_tokens"]), len(b["seed_tokens"])],
        "reference_seed_token": [a["reference_seed_token"], b["reference_seed_token"]],
    }
    out["shape_matches"] = (
        len(a["emitted_tokens"]) == len(b["emitted_tokens"])
        and len(a["rows"]) == len(b["rows"])
        and len(a["seed_tokens"]) == len(b["seed_tokens"]))
    if not out["shape_matches"]:
        out["identical"] = False
        return out

    out["seed_mismatch_positions"] = [
        i for i, (x, y) in enumerate(zip(a["seed_tokens"], b["seed_tokens"])) if x != y]
    out["emitted_mismatch_positions"] = [
        i for i, (x, y) in enumerate(zip(a["emitted_tokens"], b["emitted_tokens"]))
        if x != y]

    mism = {f: [] for f in ROW_FIELDS}
    max_ulp = {"top1_logit": 0, "top2_logits": 0}
    for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"])):
        for f in ROW_FIELDS:
            va, vb = ra[f], rb[f]
            if va == vb:
                continue
            rec = {"position": i, "a": va, "b": vb}
            if f == "top1_logit":
                g = ulp_gap(va, vb)
                rec["ulp"] = g
                max_ulp[f] = max(max_ulp[f], abs(g))
            elif f == "top2_logits":
                gaps = [ulp_gap(x, y) for x, y in zip(va, vb)]
                rec["ulp"] = gaps
                max_ulp[f] = max([max_ulp[f]] + [abs(g) for g in gaps])
            mism[f].append(rec)

    out["row_field_mismatch_counts"] = {f: len(v) for f, v in mism.items()}
    out["row_field_first_mismatches"] = {f: v[:5] for f, v in mism.items()}
    out["max_abs_ulp"] = max_ulp
    out["rows_compared"] = len(a["rows"])
    out["identical"] = (
        not out["seed_mismatch_positions"]
        and not out["emitted_mismatch_positions"]
        and all(not v for v in mism.values()))
    return out


# --- PATH B: the wide speculative dispatch ----------------------------------

WIDE_SCALARS = ("all_tokens_matched", "parity_all_ok", "residual_divergence_count",
                "accepted_draft_total", "rejected_draft_total", "declared_rows_total",
                "reference_checked_row_total", "rejected_rows_reference_checked",
                "round_count", "emitted_token_total",
                "verify_block_replayed_round_count", "effective_mean_draft_len",
                "effective_max_draft_len", "target_cache_offset_final",
                "accepted_draft_rate")


def compare_wide(a: dict, b: dict) -> dict:
    da, db = a["effective_draft_lengths"], b["effective_draft_lengths"]
    out = {
        "row_population": "candidate speculative decode, wide multi-row dispatch",
        "covers_changed_dispatch": True,
        "reading": "argmax-level (mtp-timed does not retain per-row top-2)",
        "round_count": [len(da), len(db)],
        "trajectory_length_matches": len(da) == len(db),
        "draft_len_mismatch_rounds": [
            i for i, (x, y) in enumerate(zip(da, db)) if x != y],
        "widths_exercised": dict(sorted(
            collections.Counter(width_of_draft_len(d) for d in da).items())),
        "scalar_mismatches": {
            k: [a[k], b[k]] for k in WIDE_SCALARS if a[k] != b[k]},
        "declared_rows": a["declared_rows_total"],
        "row_ledger_closes": (
            a["round_count"] + sum(da) == a["declared_rows_total"]
            and a["reference_checked_row_total"] == a["declared_rows_total"]),
    }
    out["identical"] = (
        out["trajectory_length_matches"]
        and not out["draft_len_mismatch_rounds"]
        and not out["scalar_mismatches"]
        and out["row_ledger_closes"])
    return out


# --- PATH C: direct bitwise reading of the wide rows ------------------------


def compare_ledger(a: dict, b: dict) -> dict:
    ra, rb = a.get("row_ledger", []), b.get("row_ledger", [])
    out = {
        "row_population": "candidate wide-dispatch rows, mtp-verify row_ledger",
        "covers_changed_dispatch": True,
        "reading": "logit-level bitwise",
        "row_count": [len(ra), len(rb)],
        "shape_matches": len(ra) == len(rb) and bool(ra),
    }
    if not out["shape_matches"]:
        out["identical"] = False
        return out
    mism = {f: [] for f in LEDGER_FIELDS}
    max_ulp = 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        for f in LEDGER_FIELDS:
            va, vb = x.get(f), y.get(f)
            if va == vb:
                continue
            rec = {"row_index": i, "round": x.get("round"), "a": va, "b": vb}
            if f == "top2_logits":
                gaps = [ulp_gap(p, q) for p, q in zip(va, vb)]
                rec["ulp"] = gaps
                max_ulp = max([max_ulp] + [abs(g) for g in gaps])
            elif f == "reference_margin":
                rec["ulp"] = ulp_gap(va, vb)
            mism[f].append(rec)
    out["field_mismatch_counts"] = {f: len(v) for f, v in mism.items()}
    out["field_first_mismatches"] = {f: v[:5] for f, v in mism.items()}
    out["max_abs_ulp_top2_logits"] = max_ulp
    out["widths_in_ledger"] = dict(sorted(collections.Counter(
        width_of_draft_len(
            sum(1 for r in ra if r.get("round") == rnd and r.get("kind") == "draft"))
        for rnd in sorted({r.get("round") for r in ra})).items()))
    out["identical"] = all(not v for v in mism.values())
    return out


def ledger_provenance(base_path: str, cand_path: str) -> dict:
    """Prove the two PATH C ledgers came from two differently built binaries.

    Without this, an identical-ledger result is also consistent with having
    compared one arm against itself.
    """
    def meta(p: str) -> dict:
        f = pathlib.Path(p).with_name(pathlib.Path(p).stem + "-meta.txt")
        if not f.exists():
            return {}
        out = {}
        for line in f.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    mb, mc = meta(base_path), meta(cand_path)
    if not mb or not mc:
        return {"meta_present": False, "arms_provably_distinct_binaries": False}
    differ = ("worker_sha256", "metallib_sha256[.build-worker/release/mlx.metallib]",
              "e55_binary_assert_m9_na", "e55_binary_assert_wide_bound")
    shared = ("golden_sha256", "head_dir", "tokens", "depth")
    return {
        "meta_present": True,
        "base": {k: mb.get(k) for k in differ + shared + ("arm", "git_head")},
        "candidate": {k: mc.get(k) for k in differ + shared + ("arm", "git_head")},
        "fields_that_must_differ": {k: mb.get(k) != mc.get(k) for k in differ},
        "fields_that_must_match": {k: mb.get(k) == mc.get(k) for k in shared},
        "base_dispatches_m9_na": mb.get("e55_binary_assert_m9_na"),
        "candidate_dispatches_m9_na": mc.get("e55_binary_assert_m9_na"),
        "arms_provably_distinct_binaries": (
            mb.get("worker_sha256") != mc.get("worker_sha256")
            and mb.get("e55_binary_assert_m9_na") == "3"
            and mc.get("e55_binary_assert_m9_na") == "5"
            and all(mb.get(k) == mc.get(k) for k in shared)),
    }


def eos_report(payload: dict) -> dict:
    em = payload["emitted_tokens"]
    hits = [i for i, t in enumerate(em) if t == EOS_TOKEN_ID]
    return {
        "eos_token_id": EOS_TOKEN_ID,
        "eos_positions": hits,
        "eos_present": bool(hits),
        "tokens_after_first_eos": (len(em) - 1 - hits[0]) if hits else None,
        "emitted_token_count": len(em),
        "window_closed": len(em) == 513,
    }


def negative_control() -> dict:
    """One perturbation per instrument; each must be reported exactly once."""
    g = read("base", "leg-1", "02-mtp-verify-output.json")
    w = read("base", "leg-1", "04-mtp-timed.json")
    cases = {}

    b = copy.deepcopy(g)
    b["emitted_tokens"][256] += 1
    cases["A_emitted_token_flipped_at_256"] = (
        compare_golden(g, b)["emitted_mismatch_positions"] == [256])

    b = copy.deepcopy(g)
    b["rows"][100]["sequential_argmax"] += 1
    r = compare_golden(g, b)
    cases["A_argmax_flipped_at_100"] = (
        r["row_field_mismatch_counts"]["sequential_argmax"] == 1 and not r["identical"])

    b = copy.deepcopy(g)
    v = b["rows"][200]["top1_logit"]
    b["rows"][200]["top1_logit"] = math.nextafter(v, math.inf)
    r = compare_golden(g, b)
    cases["A_top1_logit_one_ulp_at_200"] = (
        r["row_field_mismatch_counts"]["top1_logit"] == 1
        and r["max_abs_ulp"]["top1_logit"] == 1 and not r["identical"])

    b = copy.deepcopy(g)
    v = b["rows"][300]["top2_logits"][1]
    b["rows"][300]["top2_logits"][1] = math.nextafter(v, -math.inf)
    r = compare_golden(g, b)
    cases["A_top2_logit_one_ulp_at_300"] = (
        r["row_field_mismatch_counts"]["top2_logits"] == 1
        and r["max_abs_ulp"]["top2_logits"] == 1 and not r["identical"])

    b = copy.deepcopy(g)
    b["rows"] = b["rows"][:-1]
    cases["A_truncated_row_ledger"] = not compare_golden(g, b)["shape_matches"]

    b = copy.deepcopy(w)
    b["effective_draft_lengths"][40] -= 1
    r = compare_wide(w, b)
    cases["B_draft_len_changed_at_round_40"] = (
        r["draft_len_mismatch_rounds"] == [40] and not r["identical"])

    b = copy.deepcopy(w)
    b["accepted_draft_total"] -= 1
    r = compare_wide(w, b)
    cases["B_accepted_total_changed"] = (
        "accepted_draft_total" in r["scalar_mismatches"] and not r["identical"])

    b = copy.deepcopy(w)
    b["all_tokens_matched"] = False
    r = compare_wide(w, b)
    cases["B_all_tokens_matched_cleared"] = (
        "all_tokens_matched" in r["scalar_mismatches"] and not r["identical"])

    # PATH C's comparator is exercised on synthetic ledgers so it is proven able
    # to fail even in runs where no ledger was produced.
    la = {"row_ledger": [
        {"top2_tokens": [1, 2], "top2_logits": [1.5, 0.5], "token": 1,
         "reference_token": 1, "reference_margin": 1.0,
         "reference_checked_by": "serial_golden", "accepted": True,
         "kind": "draft", "round": 0, "draft_index": 0}]}
    lb = copy.deepcopy(la)
    lb["row_ledger"][0]["top2_logits"][0] = math.nextafter(1.5, math.inf)
    r = compare_ledger(la, lb)
    cases["C_ledger_top2_logit_one_ulp"] = (
        r["field_mismatch_counts"]["top2_logits"] == 1
        and r["max_abs_ulp_top2_logits"] == 1 and not r["identical"])
    lb = copy.deepcopy(la)
    lb["row_ledger"][0]["accepted"] = False
    r = compare_ledger(la, lb)
    cases["C_ledger_acceptance_flipped"] = (
        r["field_mismatch_counts"]["accepted"] == 1 and not r["identical"])
    lb = copy.deepcopy(la)
    lb["row_ledger"][0]["reference_margin"] = math.nextafter(1.0, math.inf)
    r = compare_ledger(la, lb)
    cases["C_ledger_reference_margin_one_ulp"] = (
        r["field_mismatch_counts"]["reference_margin"] == 1 and not r["identical"])
    lb = copy.deepcopy(la)
    lb["row_ledger"][0]["reference_checked_by"] = "verify_block_replay"
    r = compare_ledger(la, lb)
    cases["C_ledger_reference_source_changed"] = (
        r["field_mismatch_counts"]["reference_checked_by"] == 1 and not r["identical"])
    cases["C_ledger_absent_is_not_a_pass"] = not compare_ledger({}, {})["identical"]

    return {"cases": cases, "all_fired": all(cases.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e55-exactness.json")
    ap.add_argument("--ledger-base")
    ap.add_argument("--ledger-candidate")
    args = ap.parse_args()

    golden = {(a, l): read(a, l, "02-mtp-verify-output.json")
              for a in ARMS for l in LEGS}
    wide = {(a, l): read(a, l, "04-mtp-timed.json") for a in ARMS for l in LEGS}

    path_a, path_b = {}, {}
    for leg in LEGS:
        path_a["candidate_vs_base_%s" % leg] = compare_golden(
            golden[("base", leg)], golden[("m9two", leg)])
        path_a["null_base_vs_base2_%s" % leg] = compare_golden(
            golden[("base", leg)], golden[("base2", leg)])
        path_b["candidate_vs_base_%s" % leg] = compare_wide(
            wide[("base", leg)], wide[("m9two", leg)])
        path_b["null_base_vs_base2_%s" % leg] = compare_wide(
            wide[("base", leg)], wide[("base2", leg)])
    path_a["candidate_leg1_vs_leg2"] = compare_golden(
        golden[("m9two", "leg-1")], golden[("m9two", "leg-2")])

    path_c = None
    if args.ledger_base and args.ledger_candidate:
        path_c = compare_ledger(
            json.loads(pathlib.Path(args.ledger_base).read_text()),
            json.loads(pathlib.Path(args.ledger_candidate).read_text()))
        path_c["provenance"] = ledger_provenance(
            args.ledger_base, args.ledger_candidate)

    neg = negative_control()
    if path_c is not None:
        # The failure this guards against is real: comparing one arm against
        # itself also yields an all-zero ledger diff.
        neg["cases"]["C_self_comparison_is_not_distinct_arms"] = not (
            ledger_provenance(args.ledger_candidate, args.ledger_candidate)
            ["arms_provably_distinct_binaries"])
        neg["all_fired"] = all(neg["cases"].values())

    payload = {
        "arms": list(ARMS),
        "legs": list(LEGS),
        "decode_tokens": 512,
        "path_a_reference_golden_m1": path_a,
        "path_b_wide_dispatch": path_b,
        "path_c_wide_row_ledger": path_c,
        "eos": {"%s/%s" % (a, l): eos_report(p) for (a, l), p in golden.items()},
        "correctness_gate": {
            "%s/%s" % (a, l): {
                k: read(a, l, "01-correctness.json")[k]
                for k in ("passed", "golden_hash", "checked_steps", "case_count",
                          "first_failing_step", "first_failing_case")}
            for a in ARMS for l in LEGS},
        "negative_control": neg,
    }

    a_cand = all(v["identical"] for k, v in path_a.items()
                 if k.startswith("candidate_vs_base"))
    a_null = all(v["identical"] for k, v in path_a.items()
                 if k.startswith("null_base_vs_base2"))
    b_cand = all(v["identical"] for k, v in path_b.items()
                 if k.startswith("candidate_vs_base"))
    b_null = all(v["identical"] for k, v in path_b.items()
                 if k.startswith("null_base_vs_base2"))
    ghash = {v["golden_hash"] for v in payload["correctness_gate"].values()}

    payload["verdicts"] = {
        "path_a_m1_golden_bitwise_identical": a_cand,
        "path_a_null_identical": a_null,
        "path_b_wide_argmax_trajectory_identical": b_cand,
        "path_b_null_identical": b_null,
        "path_c_wide_rows_bitwise_identical": (
            path_c["identical"] if path_c else None),
        "path_c_arms_provably_distinct_binaries": (
            path_c["provenance"]["arms_provably_distinct_binaries"]
            if path_c else None),
        "golden_hash_shared_across_all_arms": len(ghash) == 1,
        "all_correctness_gates_passed": all(
            v["passed"] for v in payload["correctness_gate"].values()),
        "negative_controls_all_fired": neg["all_fired"],
    }
    payload["wide_rows_covered_argmax_level"] = wide[("base", "leg-1")][
        "declared_rows_total"]
    payload["widths_exercised"] = path_b["candidate_vs_base_leg-1"]["widths_exercised"]
    payload["hard_stop_tripped"] = not (a_cand and b_cand) or (
        path_c is not None and not path_c["identical"])
    # A matched pair of ledgers only carries cross-arm meaning when the two arms
    # provably ran different binaries, so provenance gates the claim itself.
    payload["direct_bitwise_wide_evidence_present"] = bool(
        path_c is not None
        and path_c["provenance"]["arms_provably_distinct_binaries"])
    payload["verdict_ok"] = (
        a_cand and a_null and b_cand and b_null
        and payload["verdicts"]["golden_hash_shared_across_all_arms"]
        and payload["verdicts"]["all_correctness_gates_passed"]
        and neg["all_fired"]
        and (path_c is None
             or (path_c["identical"]
                 and path_c["provenance"]["arms_provably_distinct_binaries"])))

    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E55 exactness, 512 decode tokens")
    print()
    print("PATH A  reference golden, M=1 -- logit-level, DOES NOT cover the "
          "changed dispatch")
    for k, v in path_a.items():
        print("  %-30s identical=%-6s rows=%s %s"
              % (k, v["identical"], v.get("rows_compared"),
                 v.get("row_field_mismatch_counts", "SHAPE MISMATCH")))
    print()
    print("PATH B  wide speculative dispatch -- covers the changed dispatch, "
          "argmax-level")
    print("  widths exercised (M = drafts + 1): %s" % payload["widths_exercised"])
    for k, v in path_b.items():
        print("  %-30s identical=%-6s rows=%s mismatch_rounds=%s scalars=%s"
              % (k, v["identical"], v["declared_rows"],
                 v["draft_len_mismatch_rounds"], v["scalar_mismatches"] or "{}"))
    print()
    print("PATH C  wide row ledger -- logit-level on the changed dispatch")
    if path_c is None:
        print("  ABSENT. `mtp-verify --golden` was not run, so no direct bitwise")
        print("  reading of the wide rows exists. PATH B is argmax-level only.")
    else:
        print("  identical=%s rows=%s max_ulp=%s widths=%s %s"
              % (path_c["identical"], path_c["row_count"],
                 path_c.get("max_abs_ulp_top2_logits"),
                 path_c.get("widths_in_ledger"),
                 path_c.get("field_mismatch_counts")))
        pv = path_c["provenance"]
        print("  arms provably distinct binaries : %s"
              % pv["arms_provably_distinct_binaries"])
        print("    base      m9_na=%s worker=%s"
              % (pv.get("base_dispatches_m9_na"),
                 (pv.get("base") or {}).get("worker_sha256", "")[:12]))
        print("    candidate m9_na=%s worker=%s"
              % (pv.get("candidate_dispatches_m9_na"),
                 (pv.get("candidate") or {}).get("worker_sha256", "")[:12]))
        print("    must differ: %s" % pv.get("fields_that_must_differ"))
        print("    must match : %s" % pv.get("fields_that_must_match"))
    print()
    print("negative control (each must be reported):")
    for k, v in neg["cases"].items():
        print("  %-36s %s" % (k, "FIRED" if v else "DID NOT FIRE"))
    print("  all_fired = %s" % neg["all_fired"])
    print()
    print("EOS / window closure:")
    for k, v in payload["eos"].items():
        print("  %-14s emitted=%d eos_at=%s after_eos=%s closed=%s"
              % (k, v["emitted_token_count"], v["eos_positions"],
                 v["tokens_after_first_eos"], v["window_closed"]))
    print()
    print("correctness gate:")
    for k, v in payload["correctness_gate"].items():
        print("  %-14s passed=%s steps=%s golden=%s"
              % (k, v["passed"], v["checked_steps"], v["golden_hash"][:12]))
    print()
    for k, v in payload["verdicts"].items():
        print("  %-42s %s" % (k, v))
    print()
    print("wide rows covered at argmax level : %d"
          % payload["wide_rows_covered_argmax_level"])
    print("direct bitwise wide evidence      : %s"
          % payload["direct_bitwise_wide_evidence_present"])
    print("HARD STOP tripped                 : %s" % payload["hard_stop_tripped"])
    print("VERDICT                           : %s"
          % ("OK" if payload["verdict_ok"] else "FAIL"))
    return 0 if payload["verdict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
