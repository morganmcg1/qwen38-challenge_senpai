#!/usr/bin/env python3
"""E25 r2 timed headline: prefill-free decode gain of the PRICE arm over BASE.

Reads the trusted parent's own journals from each matched ABBA leg and reports
``e25/mtp_true_decode_gain_pct_median_of_8``.

The primary quantity per leg is decode-only seconds per token, taken as
``decode_seconds / decode_token_count`` from ``reports/04-mtp-timed.json``.
That excludes seed prefill, which is timed separately in ``seed_prefill_seconds``.

Every leg is gated on fidelity, row accounting, arm binary identity and head
identity before it may contribute to the headline. The depth-0 serial control
that each ``--local-iterate`` leg measures for itself is used as a host-drift
control: both arms share a byte-identical serial path, so a material
between-arm serial gap means the host moved rather than the candidate.
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ALL_PROMPTS = [
    "english",
    "narrative",
    "technical",
    "dramatic",
    "travel",
    "philosophy",
    "natural_history",
    "medicine",
]

RUNS_ROOT = Path(".mlxfast-private/e25/runs")
BINS_ROOT = Path(".mlxfast-private/e25/bins")

EXPECTED_DECODE_TOKENS = 512
EXPECTED_CLI_SHA = "c9bfcaf9c58d5b5bd31466f4bab8c90a5d693bf8f0afd2818840deef0fd060b7"
EXPECTED_HEAD_SAFETENSORS_SHA = (
    "d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1"
)
EXPECTED_HEAD_TREE_BYTES = 427746170
EXPECTED_HEAD_DIR_SUFFIX = "mtp-head-declared-q2q4-run"
COOL_GATE_DISCLOSURES = [
    "cool_gate_passed_real_gate",
    "gate_qualified_for_timing",
    "cool_gate_temp_c",
    "cool_gate_bypass_reason",
]

R1_HEADLINE_GAIN_PCT = 3.8346226261260976
MODELLED_ARM_D_GAIN_PCT = 0.322
LIVE_RANKED_BAR = 3.2341518328631


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def parse_meta(path):
    """meta.txt is append-only and repeats some keys; keep every value."""
    meta = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        meta.setdefault(key.strip(), []).append(value.strip())
    return meta


def meta_last(meta, key):
    values = meta.get(key)
    return values[-1] if values else None


def arm_identity(arm):
    digest_path = BINS_ROOT / arm / "sha256.txt"
    digests = {}
    for line in digest_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            digests[parts[1]] = parts[0]
    return {
        "arm": arm,
        "worker_sha256": digests.get("mlxfast-runtime-worker"),
        "cli_sha256": digests.get("mlxfast-swift"),
        "source_sha256": digests.get("source.swift"),
        "source_blob": (BINS_ROOT / arm / "source-blob.txt").read_text().strip(),
    }


def ms_per_token(report):
    return 1000.0 * report["decode_seconds"] / report["decode_token_count"]


CLIFF_DEPTH = 4


def depth_profile(mtp):
    """Per-round cost by chosen draft depth, on the trusted parent's clock.

    ``effective_draft_lengths[i]`` is the number of draft rows round ``i``
    proposed, so pairing it with ``block_request_seconds[i]`` prices each
    depth without any trace instrumentation. Depth >= CLIFF_DEPTH is the
    first depth whose verify width M = depth+1 crosses the measured
    two-pass weight-stream cliff in ``affine_qmv_fast``.
    """
    depths = mtp["effective_draft_lengths"]
    walls = mtp["block_request_seconds"]
    by_depth = collections.defaultdict(list)
    for depth, wall in zip(depths, walls):
        by_depth[depth].append(1000.0 * wall)
    below = [x for d, v in by_depth.items() if d < CLIFF_DEPTH for x in v]
    above = [x for d, v in by_depth.items() if d >= CLIFF_DEPTH for x in v]
    return {
        "histogram": {str(d): len(v) for d, v in sorted(by_depth.items())},
        "mean_ms_by_depth": {
            str(d): statistics.mean(v) for d, v in sorted(by_depth.items())
        },
        "rounds": len(depths),
        "rounds_above_cliff": len(above),
        "rounds_above_cliff_share": len(above) / len(depths),
        "mean_ms_below_cliff": statistics.mean(below) if below else None,
        "mean_ms_above_cliff": statistics.mean(above) if above else None,
        "cliff_excess_ms_per_round": (
            statistics.mean(above) - statistics.mean(below)
            if above and below else None
        ),
        "cliff_excess_ms_total": (
            len(above) * (statistics.mean(above) - statistics.mean(below))
            if above and below else 0.0
        ),
    }


def collect_leg(prompt, arm, identity, runs_root):
    run_dir = runs_root / f"{prompt}-{arm}"
    mtp = load_json(run_dir / "reports" / "04-mtp-timed.json")
    serial = load_json(run_dir / "reports" / "03-mtp-timed.json")
    score = load_json(run_dir / "score.json")
    correctness = load_json(run_dir / "reports" / "01-correctness.json")
    meta = parse_meta(run_dir / "meta.txt")

    head = mtp.get("head_provenance") or {}
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(f"{prompt}/{arm}: {message}")

    check(mtp["all_tokens_matched"] is True, "mtp all_tokens_matched is not True")
    check(mtp["parity_all_ok"] is True, "mtp parity_all_ok is not True")
    check(mtp["residual_divergence_count"] == 0, "mtp residual divergence")
    check(
        mtp["decode_token_count"] == EXPECTED_DECODE_TOKENS,
        f"mtp decode_token_count={mtp['decode_token_count']}",
    )
    check(
        mtp["declared_rows_total"] == mtp["reference_checked_row_total"],
        "mtp row ledger not closed",
    )
    check(
        mtp["declared_rows_total"]
        == mtp["target_tail_total"] + mtp["accepted_draft_total"] + mtp["rejected_draft_total"],
        "mtp declared rows do not decompose into primary + accepted + rejected",
    )
    check(
        mtp["target_tail_total"] == mtp["round_count"],
        "mtp primary target rows do not match the round count",
    )
    check(
        mtp["rejected_rows_reference_checked"] == mtp["rejected_draft_total"],
        "mtp rejected rows were not all reference-checked",
    )
    check(
        mtp["emitted_token_total"] == mtp["decode_token_count"],
        "mtp emitted tokens do not match the decode window",
    )
    check(
        mtp["max_rejected_tail_logit_delta"] == 0,
        "mtp rejected tail logit delta is not 0",
    )
    check(mtp["uses_native_mtp_head"] is True, "mtp leg did not draft")
    check(head.get("bytes") == EXPECTED_HEAD_TREE_BYTES, f"head bytes={head.get('bytes')}")

    check(serial["is_serial_control"] is True, "serial leg is not the serial control")
    check(serial["mtp_depth"] == 0, f"serial mtp_depth={serial['mtp_depth']}")
    check(serial["all_tokens_matched"] is True, "serial all_tokens_matched is not True")
    check(
        serial["decode_token_count"] == EXPECTED_DECODE_TOKENS,
        f"serial decode_token_count={serial['decode_token_count']}",
    )

    check(correctness["passed"] is True, "01-correctness did not pass")
    check(not correctness.get("error"), f"01-correctness error={correctness.get('error')}")

    check(score["passed"] is True, "score.json passed is not True")
    check(
        score["metrics"]["all_tokens_matched"] is True,
        "score.json metrics.all_tokens_matched is not True",
    )
    check(
        score["metrics"]["decode_tokens"] == EXPECTED_DECODE_TOKENS,
        f"score.json decode_tokens={score['metrics']['decode_tokens']}",
    )

    check(meta_last(meta, "exit") == "0", f"meta exit={meta_last(meta, 'exit')}")
    check(meta_last(meta, "dirty") == "0", f"meta dirty={meta_last(meta, 'dirty')}")
    check(
        meta_last(meta, "worker_sha256") == identity["worker_sha256"],
        "meta worker_sha256 does not match the installed arm binary",
    )
    check(
        meta_last(meta, "cli_sha256") == EXPECTED_CLI_SHA,
        "meta cli_sha256 does not match the shared trusted driver",
    )
    check(
        meta_last(meta, "source_sha256") == identity["source_sha256"],
        "meta source_sha256 does not match the installed arm source",
    )
    check(
        meta_last(meta, "head_safetensors_sha256") == EXPECTED_HEAD_SAFETENSORS_SHA,
        "meta head_safetensors_sha256 is not the declared q2-q4 head",
    )
    check(
        (meta_last(meta, "head_dir") or "").endswith(EXPECTED_HEAD_DIR_SUFFIX),
        f"meta head_dir={meta_last(meta, 'head_dir')}",
    )
    for key in COOL_GATE_DISCLOSURES:
        check(meta.get(key) is not None, f"meta is missing disclosure {key}")

    reported_score = score["score"]
    derived_score = (
        serial["parent_measured_seconds_per_token"]
        / mtp["parent_measured_seconds_per_token"]
    )
    check(
        abs(reported_score - derived_score) < 1e-6,
        f"score.json score {reported_score} != derived {derived_score}",
    )
    for label, report in (("mtp", mtp), ("serial", serial)):
        parent = 1000.0 * report["parent_measured_seconds_per_token"]
        check(
            abs(parent - ms_per_token(report)) < 1e-6,
            f"{label} parent clock disagrees with decode_seconds/decode_token_count",
        )

    return {
        "run_dir": str(run_dir),
        "decode_ms_per_token": ms_per_token(mtp),
        "serial_decode_ms_per_token": ms_per_token(serial),
        "local_ratio": reported_score,
        "seed_prefill_seconds": mtp["seed_prefill_seconds"],
        "prefill_seconds_per_token": mtp["prefill_seconds_per_token"],
        "counters": {
            "round_count": mtp["round_count"],
            "effective_mean_draft_len": mtp["effective_mean_draft_len"],
            "effective_max_draft_len": mtp["effective_max_draft_len"],
            "accepted_draft_rate": mtp["accepted_draft_rate"],
            "accepted_draft_total": mtp["accepted_draft_total"],
            "rejected_draft_total": mtp["rejected_draft_total"],
            "declared_rows_total": mtp["declared_rows_total"],
            "non_drafting_round_count": mtp["non_drafting_round_count"],
            "verify_block_replayed_round_count": mtp["verify_block_replayed_round_count"],
            "p50_block_request_seconds": mtp["p50_block_request_seconds"],
            "max_block_request_seconds_after_first": mtp[
                "max_block_request_seconds_after_first"
            ],
            "first_block_seconds": mtp["first_block_seconds"],
        },
        "head": {
            "sha256": head.get("sha256"),
            "bytes": head.get("bytes"),
            "file_count": head.get("file_count"),
            "origin": head.get("origin"),
            "source": head.get("source"),
        },
        "depth_profile": depth_profile(mtp),
        "decode_seconds": mtp["decode_seconds"],
        "started": meta_last(meta, "started"),
        "finished": meta_last(meta, "finished"),
        "thermal_before": meta_last(meta, "thermal_before"),
        "failures": failures,
    }


def pct_gain(base, candidate):
    return 100.0 * (base - candidate) / base


def cliff_attribution(base, cand):
    """How much of the measured saving is just not crossing the cost cliff.

    The candidate is a hard DEEP_CAP=3, so every base round at depth >=
    CLIFF_DEPTH is a round the candidate cannot run. Pricing those rounds
    at the base leg's own measured excess over its sub-cliff rounds gives
    the share of the total decode saving that the cap alone explains; the
    remainder is the price curve reshaping mass below the cliff.
    """
    saved_ms = 1000.0 * (base["decode_seconds"] - cand["decode_seconds"])
    excess_ms = base["depth_profile"]["cliff_excess_ms_total"]
    return {
        "total_saved_ms": saved_ms,
        "base_rounds_above_cliff": base["depth_profile"]["rounds_above_cliff"],
        "base_rounds_above_cliff_share":
            base["depth_profile"]["rounds_above_cliff_share"],
        "candidate_rounds_above_cliff":
            cand["depth_profile"]["rounds_above_cliff"],
        "base_cliff_excess_ms_per_round":
            base["depth_profile"]["cliff_excess_ms_per_round"],
        "cliff_excess_ms_total": excess_ms,
        "share_of_saving_from_cliff_avoidance": (
            excess_ms / saved_ms if saved_ms > 0 else None
        ),
        "round_count_delta": (
            cand["counters"]["round_count"] - base["counters"]["round_count"]
        ),
        "accepted_draft_delta": (
            cand["counters"]["accepted_draft_total"]
            - base["counters"]["accepted_draft_total"]
        ),
        "rejected_draft_delta": (
            cand["counters"]["rejected_draft_total"]
            - base["counters"]["rejected_draft_total"]
        ),
    }


def pooled_cliff_attribution(per_prompt):
    rows = [row["cliff_attribution"] for row in per_prompt.values()]
    if not rows:
        return None
    saved = sum(r["total_saved_ms"] for r in rows)
    excess = sum(r["cliff_excess_ms_total"] for r in rows)
    return {
        "total_saved_ms": saved,
        "cliff_excess_ms_total": excess,
        "share_of_saving_from_cliff_avoidance": (
            excess / saved if saved > 0 else None
        ),
        "base_rounds_above_cliff": sum(
            r["base_rounds_above_cliff"] for r in rows),
        "candidate_rounds_above_cliff": sum(
            r["candidate_rounds_above_cliff"] for r in rows),
        "base_rounds": sum(
            row["base"]["depth_profile"]["rounds"]
            for row in per_prompt.values()),
        "round_count_delta": sum(r["round_count_delta"] for r in rows),
        "accepted_draft_delta": sum(r["accepted_draft_delta"] for r in rows),
        "rejected_draft_delta": sum(r["rejected_draft_delta"] for r in rows),
        "per_prompt_share": {
            p: row["cliff_attribution"]["share_of_saving_from_cliff_avoidance"]
            for p, row in per_prompt.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=",".join(ALL_PROMPTS))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--base-arm", default="BASE")
    parser.add_argument("--cand-arm", default="PRICE")
    parser.add_argument("--out")
    parser.add_argument("--text", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="report on the prompts whose legs exist instead of failing",
    )
    args = parser.parse_args()

    prompts = [p for p in args.prompts.split(",") if p]
    runs_root = Path(args.runs_root)
    identities = {
        args.base_arm: arm_identity(args.base_arm),
        args.cand_arm: arm_identity(args.cand_arm),
    }
    if identities[args.base_arm]["worker_sha256"] == identities[args.cand_arm]["worker_sha256"]:
        sys.exit("base and candidate arms share one worker binary; nothing to compare")

    per_prompt = {}
    failures = []
    missing = []
    for prompt in prompts:
        legs = {}
        for arm in (args.base_arm, args.cand_arm):
            run_dir = runs_root / f"{prompt}-{arm}"
            if not (run_dir / "reports" / "04-mtp-timed.json").exists():
                missing.append(str(run_dir))
                continue
            legs[arm] = collect_leg(prompt, arm, identities[arm], runs_root)
            failures.extend(legs[arm]["failures"])
        if len(legs) != 2:
            continue
        base = legs[args.base_arm]
        cand = legs[args.cand_arm]
        per_prompt[prompt] = {
            "base": base,
            "candidate": cand,
            "gain_pct": pct_gain(base["decode_ms_per_token"], cand["decode_ms_per_token"]),
            "serial_delta_pct": pct_gain(
                base["serial_decode_ms_per_token"], cand["serial_decode_ms_per_token"]
            ),
            "local_ratio_delta": cand["local_ratio"] - base["local_ratio"],
            "mean_draft_len_delta": (
                cand["counters"]["effective_mean_draft_len"]
                - base["counters"]["effective_mean_draft_len"]
            ),
            "max_draft_len": {
                "base": base["counters"]["effective_max_draft_len"],
                "candidate": cand["counters"]["effective_max_draft_len"],
            },
            "cliff_attribution": cliff_attribution(base, cand),
        }

    if missing and not args.allow_partial:
        sys.exit("missing legs (pass --allow-partial to proceed):\n  " + "\n  ".join(missing))

    gains = {p: row["gain_pct"] for p, row in per_prompt.items()}
    serial_deltas = [abs(row["serial_delta_pct"]) for row in per_prompt.values()]
    report = {
        "prompts_requested": prompts,
        "prompts_measured": sorted(per_prompt),
        "missing_legs": missing,
        "base_arm": args.base_arm,
        "candidate_arm": args.cand_arm,
        "arm_identity": identities,
        "gates": {"all_pass": not failures, "failures": failures},
        "headline": {
            "metric": "e25/mtp_true_decode_gain_pct_median_of_8",
            "definition": "median over prompts of (BASE-PRICE)/BASE*100 on decode_seconds/decode_token_count",
            "n_prompts": len(gains),
            "median_gain_pct": statistics.median(gains.values()) if gains else None,
            "mean_gain_pct": statistics.fmean(gains.values()) if gains else None,
            "min_gain_pct": min(gains.values()) if gains else None,
            "max_gain_pct": max(gains.values()) if gains else None,
            "prompts_improved": sum(1 for g in gains.values() if g > 0),
            "per_prompt_gain_pct": gains,
        },
        "host_drift_control": {
            "note": "both arms run a byte-identical depth-0 serial leg; a large gap means host drift",
            "serial_delta_pct": {p: row["serial_delta_pct"] for p, row in per_prompt.items()},
            "median_abs_serial_delta_pct": (
                statistics.median(serial_deltas) if serial_deltas else None
            ),
            "max_abs_serial_delta_pct": max(serial_deltas) if serial_deltas else None,
        },
        "reference_points": {
            "r1_headline_gain_pct": R1_HEADLINE_GAIN_PCT,
            "r2_modelled_arm_d_gain_pct": MODELLED_ARM_D_GAIN_PCT,
            "live_ranked_bar": LIVE_RANKED_BAR,
        },
        "cliff_attribution_pooled": pooled_cliff_attribution(per_prompt),
        "per_prompt": per_prompt,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.text or not args.out:
        head = report["headline"]
        print(f"arms: {args.base_arm} -> {args.cand_arm}")
        print(f"gates: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        for failure in failures:
            print(f"  ! {failure}")
        print(f"{'prompt':<16}{'base ms/tok':>12}{'cand ms/tok':>12}{'gain %':>9}"
              f"{'serial d%':>11}{'base d̄':>8}{'cand d̄':>8}")
        for prompt in sorted(per_prompt):
            row = per_prompt[prompt]
            print(
                f"{prompt:<16}{row['base']['decode_ms_per_token']:>12.3f}"
                f"{row['candidate']['decode_ms_per_token']:>12.3f}"
                f"{row['gain_pct']:>9.3f}{row['serial_delta_pct']:>11.3f}"
                f"{row['base']['counters']['effective_mean_draft_len']:>8.3f}"
                f"{row['candidate']['counters']['effective_mean_draft_len']:>8.3f}"
            )
        print(
            f"median gain% over {head['n_prompts']} prompt(s): {head['median_gain_pct']}"
            if head["median_gain_pct"] is not None
            else "no complete pairs"
        )
        if head["median_gain_pct"] is not None:
            print(f"  mean {head['mean_gain_pct']:.4f}  min {head['min_gain_pct']:.4f}"
                  f"  max {head['max_gain_pct']:.4f}  improved {head['prompts_improved']}/{head['n_prompts']}")
            print(f"  r1 headline {R1_HEADLINE_GAIN_PCT:.4f}  modelled arm D {MODELLED_ARM_D_GAIN_PCT:.4f}")
            print(f"  host drift control: max |serial delta| "
                  f"{report['host_drift_control']['max_abs_serial_delta_pct']:.3f}%")
            pooled = report["cliff_attribution_pooled"]
            print(f"  cliff attribution: {pooled['base_rounds_above_cliff']}"
                  f"/{pooled['base_rounds']} base rounds at depth "
                  f">= {CLIFF_DEPTH} -> {pooled['candidate_rounds_above_cliff']};"
                  f" {100.0 * pooled['share_of_saving_from_cliff_avoidance']:.1f}%"
                  f" of {pooled['total_saved_ms']:.0f} ms saved")
            print(f"  accepted drafts {pooled['accepted_draft_delta']:+d}"
                  f"  rejected drafts {pooled['rejected_draft_delta']:+d}"
                  f"  rounds {pooled['round_count_delta']:+d}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
