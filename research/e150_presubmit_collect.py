#!/usr/bin/env python3
"""E150 R4 -- collect the pre-submit evidence into one artifact.

Reads the two 512-token exactness arms, checks the row ledger closes on every
leg, extracts the CAMPAIGN RULE 114 arm witness from each trace, and builds the
drafted-depth histogram the offline replay predicted.

CAMPAIGN RULE 79. Every leg here carries the per-round phase trace, so no leg
is timing valid. This collector reads `d=`, `acc=` and the arm witness from the
trace and never reads `round_us`. The depth histogram is a schedule observation,
not a price.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from e150_lib import write_artifact  # noqa: E402

EOS_IDS = {248044, 248046}
GOLDEN_DIR = ".mlxfast-private/e128/goldens"

# What the offline replay said this rule would do, recorded before the Swift
# rule ran so the comparison below is a prediction and not a fit.
REPLAY_PREDICTION = {
    "r05_shipped_linearised_noclamp_depth": 4.2444,
    "r05_shipped_linearised_noclamp_pct": 1.1178,
    "bracket_decide_measured_bill_measured_depth": 4.2042,
    "bracket_decide_measured_bill_measured_pct": 0.9699,
    "bracket_frac_rounds_inadmissible": 0.1485,
    "r1_primary_depth": 3.7926,
}

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
ARM_RE = re.compile(r"\barm=(\S+)")
RULE_RE = re.compile(r"\brule=(\S+)")
LAM_RE = re.compile(r"\blam=([0-9.]+)")
CAP_RE = re.compile(r"\bcap=(\d+)")


def read_meta(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        if "=" in line:
            k, _, v = line.rstrip("\n").partition("=")
            out[k] = v
    return out


def read_trace(path):
    """Depth histogram, accept histogram and the arm witness from one trace."""
    depths = collections.Counter()
    accepts = collections.Counter()
    witness = {"arm": None, "rule": None, "lam": None, "cap": None}
    seen = set()
    if not os.path.exists(path):
        return depths, accepts, witness, 0
    rounds = 0
    for line in open(path):
        m = ROUND_RE.match(line)
        if not m:
            continue
        rounds += 1
        depths[int(m.group(2))] += 1
        accepts[int(m.group(3))] += 1
        for key, rx in (("arm", ARM_RE), ("rule", RULE_RE),
                        ("lam", LAM_RE), ("cap", CAP_RE)):
            hit = rx.search(line)
            if hit:
                seen.add((key, hit.group(1)))
                witness[key] = hit.group(1)
    # A single trace must witness ONE arm. More than one value means the leg
    # changed rule mid-run, which would invalidate every reading taken from it.
    witness["witness_is_unique"] = len(
        {k for k, _ in seen}) == len({(k, v) for k, v in seen})
    return depths, accepts, witness, rounds


def ledger_closes(rep):
    """The eight row-accounting identities the trusted parent also checks."""
    checks = {
        "declared_equals_reference_checked":
            rep["declared_rows_total"] == rep["reference_checked_row_total"],
        "declared_equals_accepted_plus_rejected_plus_primaries":
            rep["declared_rows_total"] == (rep["accepted_draft_total"]
                                           + rep["rejected_draft_total"]
                                           + rep["round_count"]),
        "target_tail_equals_round_count":
            rep["target_tail_total"] == rep["round_count"],
        "emitted_equals_decode_tokens":
            rep["emitted_token_total"] == rep["decode_token_count"],
        "cache_offset_equals_seed_plus_decode":
            rep["target_cache_offset_final"] == (rep["seed_token_count"]
                                                 + rep["decode_token_count"]),
        "rejected_rows_all_reference_checked":
            rep["rejected_rows_reference_checked"] == rep[
                "rejected_draft_total"],
        "parity_all_ok": bool(rep["parity_all_ok"]),
        "no_rejected_tail_logit_leak":
            rep["max_rejected_tail_logit_delta"] == 0,
    }
    return checks, all(checks.values())


def golden_eos(prompt_id):
    path = os.path.join(GOLDEN_DIR, f"{prompt_id}-rows-513.json")
    if not os.path.exists(path):
        return None
    emitted = json.load(open(path)).get("emitted_tokens") or []
    hits = [i for i, t in enumerate(emitted) if t in EOS_IDS]
    if not hits:
        return {"eos_present": False, "tokens_after_first_eos": 0}
    return {"eos_present": True, "first_eos_index": hits[0],
            "eos_count": len(hits),
            "tokens_after_first_eos": 512 - hits[0] - 1}


def collect_arm(runs_dir, expected_rule):
    legs = {}
    for prompt_id in sorted(os.listdir(runs_dir)):
        leg = os.path.join(runs_dir, prompt_id)
        rep_path = os.path.join(leg, "report.json")
        if not os.path.isdir(leg) or not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path))
        meta = read_meta(os.path.join(leg, "meta.txt"))
        depths, accepts, witness, traced = read_trace(
            os.path.join(leg, "trace.txt"))
        checks, closed = ledger_closes(rep)
        total = sum(depths.values()) or 1
        legs[prompt_id] = {
            "all_tokens_matched": bool(rep["all_tokens_matched"]),
            "residual_divergence_count": rep["residual_divergence_count"],
            "row_ledger_checks": checks,
            "row_ledger_closes": closed,
            "round_count": rep["round_count"],
            "non_drafting_round_count": rep["non_drafting_round_count"],
            "accepted_draft_total": rep["accepted_draft_total"],
            "rejected_draft_total": rep["rejected_draft_total"],
            "accepted_draft_rate": rep["accepted_draft_rate"],
            "effective_mean_draft_len": rep["effective_mean_draft_len"],
            "effective_max_draft_len": rep["effective_max_draft_len"],
            "verify_block_replayed_round_count":
                rep["verify_block_replayed_round_count"],
            "declared_rows_total": rep["declared_rows_total"],
            "trace_rounds": traced,
            "trace_rounds_match_report": traced == rep["round_count"],
            "arm_witness": witness,
            "rule_witness_matches_request":
                witness.get("rule") == expected_rule,
            "drafted_depth_histogram": {str(k): v
                                        for k, v in sorted(depths.items())},
            "accepted_histogram": {str(k): v
                                   for k, v in sorted(accepts.items())},
            "mean_drafted_depth": sum(k * v for k, v in depths.items()) / total,
            # Verify width is drafted depth plus the pending primary row.
            "frac_rounds_width_ge_6":
                sum(v for k, v in depths.items() if k + 1 >= 6) / total,
            "frac_rounds_inadmissible_rule138":
                sum(v for k, v in depths.items() if k + 1 > 5) / total,
            "golden_eos": golden_eos(prompt_id),
            "base_sha": meta.get("base_sha"),
            "golden_sha256": meta.get("golden_sha256"),
            "worker_sha256": meta.get("worker_sha256"),
            "timing_valid": meta.get("timing_valid"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "exit": meta.get("exit"),
        }
    return legs


def arm_summary(legs):
    if not legs:
        return {}
    rounds = sum(v["round_count"] for v in legs.values())
    agg = collections.Counter()
    for v in legs.values():
        for k, n in v["drafted_depth_histogram"].items():
            agg[int(k)] += n
    total = sum(agg.values()) or 1
    return {
        "legs": len(legs),
        "all_matched": all(v["all_tokens_matched"] for v in legs.values()),
        "all_ledgers_close": all(v["row_ledger_closes"] for v in legs.values()),
        "all_rule_witnesses_correct":
            all(v["rule_witness_matches_request"] for v in legs.values()),
        "total_divergences":
            sum(v["residual_divergence_count"] for v in legs.values()),
        "total_rounds": rounds,
        "total_declared_rows":
            sum(v["declared_rows_total"] for v in legs.values()),
        "pooled_drafted_depth_histogram":
            {str(k): agg[k] for k in sorted(agg)},
        "pooled_mean_drafted_depth":
            sum(k * v for k, v in agg.items()) / total,
        "pooled_frac_rounds_width_ge_6":
            sum(v for k, v in agg.items() if k + 1 >= 6) / total,
        "pooled_frac_rounds_inadmissible_rule138":
            sum(v for k, v in agg.items() if k + 1 > 5) / total,
        "pooled_non_drafting_round_count":
            sum(v["non_drafting_round_count"] for v in legs.values()),
        "pooled_accepted_draft_total":
            sum(v["accepted_draft_total"] for v in legs.values()),
        "pooled_rejected_draft_total":
            sum(v["rejected_draft_total"] for v in legs.values()),
        "max_effective_draft_len":
            max(v["effective_max_draft_len"] for v in legs.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--linearised", required=True)
    ap.add_argument("--local-submit", required=True)
    ap.add_argument("--chain-rc", type=int, default=0)
    args = ap.parse_args()

    shipped = collect_arm(args.shipped, "shipped")
    linear = collect_arm(args.linearised, "linearised")

    submit = {}
    if os.path.exists(args.local_submit):
        raw = json.load(open(args.local_submit))
        metrics = raw.get("metrics", {})
        submit = {
            "passed": raw.get("passed"),
            "decode_tokens": metrics.get("decode_tokens"),
            "all_tokens_matched": metrics.get("all_tokens_matched"),
            "residual_divergence_count":
                metrics.get("residual_divergence_count"),
            "public_drift_tripwire_passed":
                metrics.get("public_drift_tripwire_passed"),
            "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
            "round_count": metrics.get("round_count"),
        }

    ship_sum = arm_summary(shipped)
    lin_sum = arm_summary(linear)

    eos_legs = {k: v["golden_eos"] for k, v in linear.items()
                if v["golden_eos"] and v["golden_eos"]["eos_present"]}
    post_eos_exact = all(linear[k]["all_tokens_matched"] for k in eos_legs)

    payload = {
        "experiment": "e150-per-round-discrimination",
        "rung": "R4-presubmit",
        "harness": "local",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "rule_79_note": (
            "Every leg carries the per-round phase trace and is not timing "
            "valid. No depth-price or schedule-policy timing contrast is "
            "published from this chain."),
        "chain_rc": args.chain_rc,
        "prediction_from_offline_replay": REPLAY_PREDICTION,
        "shipped_arm": {"legs": shipped, "summary": ship_sum},
        "linearised_arm": {"legs": linear, "summary": lin_sum},
        "local_submit_512": submit,
        "post_eos": {
            "legs_with_eos_in_window": sorted(eos_legs),
            "detail": eos_legs,
            "all_matched": post_eos_exact,
        },
        "metrics": {
            "e150_r4_exactness_all_matched":
                bool(lin_sum.get("all_matched")) and bool(
                    ship_sum.get("all_matched")),
            "e150_r4_linearised_all_matched": bool(lin_sum.get("all_matched")),
            "e150_r4_shipped_control_all_matched":
                bool(ship_sum.get("all_matched")),
            "e150_r4_row_ledger_closes": bool(
                lin_sum.get("all_ledgers_close")) and bool(
                    ship_sum.get("all_ledgers_close")),
            "e150_r4_rule_witness_correct":
                bool(lin_sum.get("all_rule_witnesses_correct")) and bool(
                    ship_sum.get("all_rule_witnesses_correct")),
            "e150_r4_post_eos_continuation_exact": bool(post_eos_exact)
                and bool(eos_legs),
            "e150_r4_observed_mean_drafted_depth":
                lin_sum.get("pooled_mean_drafted_depth"),
            "e150_r4_shipped_mean_drafted_depth":
                ship_sum.get("pooled_mean_drafted_depth"),
            "e150_r4_observed_frac_inadmissible":
                lin_sum.get("pooled_frac_rounds_inadmissible_rule138"),
            "e150_r4_depth_vs_replay_prediction":
                (lin_sum.get("pooled_mean_drafted_depth") or 0.0)
                - REPLAY_PREDICTION["bracket_decide_measured_bill_measured_depth"],
            "e150_r4_local_submit_passed": bool(submit.get("passed")),
        },
    }

    write_artifact("r4_presubmit.json", payload)
    print(json.dumps(payload["metrics"], indent=2))
    print("post_eos:", json.dumps(payload["post_eos"], indent=2))
    print("shipped summary:", json.dumps(ship_sum, indent=2))
    print("linearised summary:", json.dumps(lin_sum, indent=2))
    return 0 if payload["metrics"]["e150_r4_exactness_all_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
