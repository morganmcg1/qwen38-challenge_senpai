#!/usr/bin/env python3
"""qwen38-r1-e26: publish the stop-token continuation-defect legs to W&B.

usage:
  research/e26_log_wandb.py [--root DIR] [--group G] [--notes ...] [--name N]

Reads the per-leg reports written by research/e26-legs.sh under
  <root>/base-020c6b5/          control legs, unchanged base behaviour
  <root>/base-020c6b5-boundary/ control legs bisecting the abort boundary
  <root>/cand-07345f4/          candidate legs
and the matching `exit-codes.txt` (TOKENS DEPTH EXITCODE SECONDS) in each.

A leg that aborted leaves an EMPTY report file, so the exit-code ledger is the
authority on which legs ran and which died. Both are logged: an aborted leg is
published with `ran=False` and no counters rather than silently dropped, because
the aborts ARE the result -- the control cannot reach the 512-token window that
senpai/program.md requires.

The four pre-registered falsifiers are logged as config (the prediction) and as
summary booleans (the outcome) so the comparison is visible without leaving the
run. The primary metric is the largest local decode window that still matches
the organizer golden exactly, which is 302 at the control and 512 at the
candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

BASE_SHA = "0ac14570a0c26b803c8f84594a307e92402d98cc"
CONTROL_SHA = "020c6b5"
CANDIDATE_SHA = "07345f4"

ARMS = {
    "control": ["base-020c6b5", "base-020c6b5-boundary"],
    "candidate": ["cand-07345f4"],
}

# Pre-registered on PR #30 (comment e26-prereg-r1) before any timed leg. Each
# entry is the observation that would have KILLED the experiment.
FALSIFIERS = {
    "f1_candidate_512_matches_golden": (
        "the candidate's 512-token legs match the organizer golden exactly "
        "(all_tokens_matched, zero residual divergence, every rejected row "
        "reference-checked)"
    ),
    "f2_short_windows_are_bit_identical": (
        "on windows that never reach a stop token (128, 256) the candidate and "
        "control agree on every counter INCLUDING the element-wise draft-length "
        "schedule, i.e. the change is inert where the ranked legs live"
    ),
    "f3_control_aborts_at_the_stop_token": (
        "the control's abort boundary sits exactly at the golden's stop-token "
        "position and not somewhere else, at depth 2 AND depth 0"
    ),
    "f4_row_ledger_closes": (
        "accepted + rejected + tails == declared == reference-checked, tails == "
        "round count, and the target cache offset equals seed + emitted on every "
        "leg that completed"
    ),
}

# Counter keys that must be identical between arms on a stop-token-free window.
INERTNESS_KEYS = [
    "round_count",
    "accepted_draft_total",
    "rejected_draft_total",
    "emitted_token_total",
    "declared_rows_total",
    "reference_checked_row_total",
    "rejected_rows_reference_checked",
    "verify_block_replayed_round_count",
    "target_cache_offset_final",
    "target_tail_total",
    "non_drafting_round_count",
    "accepted_draft_rate",
    "effective_mean_draft_len",
    "effective_max_draft_len",
    "all_tokens_matched",
]

CORRECTNESS = [
    "all_tokens_matched",
    "parity_all_ok",
    "residual_divergence_count",
    "max_rejected_tail_logit_delta",
    "declared_rows_total",
    "reference_checked_row_total",
    "rejected_rows_reference_checked",
    "emitted_token_total",
    "decode_token_count",
    "seed_token_count",
    "target_cache_offset_final",
    "round_count",
    "target_tail_total",
    "accepted_draft_total",
    "rejected_draft_total",
    "non_drafting_round_count",
    "verify_block_replayed_round_count",
]

TIMING = [
    "parent_measured_seconds_per_token",
    "decode_seconds",
    "seed_prefill_seconds",
    "prefill_seconds_per_token",
    "accepted_draft_rate",
    "effective_mean_draft_len",
    "effective_max_draft_len",
    "block_request_seconds",
    "first_block_seconds",
    "max_block_request_seconds",
    "p50_block_request_seconds",
    "max_block_request_seconds_after_first",
    "p50_block_request_seconds_after_first",
]

PROVENANCE = [
    "head_provenance",
    "mtp_head_attached",
    "mtp_head_tensor_count",
    "uses_native_mtp_head",
    "uses_pinned_mtp_head",
]


def read_exit_codes(path):
    legs = {}
    if not os.path.exists(path):
        return legs
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 4:
                continue
            tokens, depth, code, seconds = (int(p) for p in parts)
            legs[(tokens, depth)] = {"exit_code": code, "wall_seconds": seconds}
    return legs


def read_report(directory, tokens, depth):
    path = os.path.join(directory, f"leg-t{tokens}-d{depth}.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as handle:
        return json.load(handle)


def collect(root):
    """Return one record per (arm, tokens, depth), aborted legs included."""
    legs = []
    for arm, directories in ARMS.items():
        for directory in directories:
            full = os.path.join(root, directory)
            for (tokens, depth), ledger in sorted(read_exit_codes(
                    os.path.join(full, "exit-codes.txt")).items()):
                report = read_report(full, tokens, depth)
                legs.append({
                    "arm": arm,
                    "tokens": tokens,
                    "depth": depth,
                    "source_dir": directory,
                    "ran": report is not None,
                    **ledger,
                    "report": report or {},
                })
    return legs


def draft_schedule_digest(report):
    lengths = report.get("effective_draft_lengths")
    if not lengths:
        return None, None
    blob = ",".join(str(int(x)) for x in lengths).encode()
    return hashlib.sha256(blob).hexdigest()[:16], len(lengths)


def leg_key(leg):
    return f"{leg['arm']}/t{leg['tokens']}_d{leg['depth']}"


def evaluate_falsifiers(legs):
    """Recompute each falsifier from the logged legs, never from prose."""
    by_key = {(leg["arm"], leg["tokens"], leg["depth"]): leg for leg in legs}

    def report(arm, tokens, depth):
        leg = by_key.get((arm, tokens, depth))
        return leg["report"] if leg and leg["ran"] else None

    outcome = {}

    # F1: exact match on both candidate 512 legs.
    f1 = True
    for depth in (0, 2):
        rep = report("candidate", 512, depth)
        f1 = f1 and rep is not None and (
            rep.get("all_tokens_matched") is True
            and rep.get("parity_all_ok") is True
            and rep.get("residual_divergence_count") == 0
            and rep.get("first_divergence_index") is None
            and rep.get("max_rejected_tail_logit_delta") == 0
            and rep.get("emitted_token_total") == 512
            and rep.get("rejected_rows_reference_checked")
            == rep.get("rejected_draft_total")
        )
    outcome["f1_candidate_512_matches_golden"] = f1

    # F2: inert on stop-token-free windows, counters AND draft schedule.
    f2 = True
    for tokens in (128, 256):
        control = report("control", tokens, 2)
        candidate = report("candidate", tokens, 2)
        if control is None or candidate is None:
            f2 = False
            continue
        f2 = f2 and all(
            control.get(key) == candidate.get(key) for key in INERTNESS_KEYS)
        f2 = f2 and (draft_schedule_digest(control)
                     == draft_schedule_digest(candidate))
    outcome["f2_short_windows_are_bit_identical"] = f2

    # F3: the control's boundary is the stop-token position, at both depths.
    f3 = True
    for depth in (0, 2):
        passing = by_key.get(("control", 302, depth))
        failing = by_key.get(("control", 303, depth))
        f3 = f3 and passing is not None and failing is not None
        f3 = f3 and passing["exit_code"] == 0 and failing["exit_code"] != 0
        f3 = f3 and passing["ran"] and not failing["ran"]
    outcome["f3_control_aborts_at_the_stop_token"] = f3

    # F4: row-ledger closure on every leg that completed.
    f4 = True
    for leg in legs:
        if not leg["ran"]:
            continue
        rep = leg["report"]
        fields = (rep.get("accepted_draft_total"),
                  rep.get("rejected_draft_total"),
                  rep.get("target_tail_total"),
                  rep.get("round_count"),
                  rep.get("declared_rows_total"),
                  rep.get("reference_checked_row_total"),
                  rep.get("emitted_token_total"),
                  rep.get("seed_token_count"),
                  rep.get("target_cache_offset_final"))
        if None in fields:
            f4 = False
            break
        (accepted, rejected, tails, rounds, declared, checked, emitted, seed,
         offset) = fields
        f4 = f4 and accepted + rejected + tails == declared
        f4 = f4 and declared == checked
        f4 = f4 and tails == rounds
        f4 = f4 and offset == seed + emitted
    outcome["f4_row_ledger_closes"] = f4
    return outcome


def largest_matching_window(legs, arm):
    best = 0
    for leg in legs:
        if leg["arm"] != arm or not leg["ran"]:
            continue
        if leg["report"].get("all_tokens_matched") is True:
            best = max(best, leg["report"].get("emitted_token_total", 0))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.path.join(os.path.expanduser("~"), "e26-stop-token"))
    parser.add_argument("--group", default="qwen38-r1-e26")
    parser.add_argument("--name", default="e26-stop-token-continuation-defect")
    parser.add_argument("--notes", default=(
        "E26: the base session commits a stop token and then refuses every "
        "later round with .notBegun, capping local decode at 302 tokens. "
        "Deleting the two early-return sites restores the full 512-token "
        "window with an exact golden match and is inert below the boundary."))
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and check every leg without opening a run")
    args = parser.parse_args()

    legs = collect(args.root)
    if not legs:
        raise SystemExit(f"no leg ledgers under {args.root}")

    falsifiers = evaluate_falsifiers(legs)
    baseline = largest_matching_window(legs, "control")
    candidate = largest_matching_window(legs, "candidate")

    headline = next(
        leg for leg in legs
        if leg["arm"] == "candidate" and leg["tokens"] == 512
        and leg["depth"] == 2)

    config = {
        "experiment": "qwen38-r1-e26-stop-token-continuation-defect",
        "assignment_pr": 30,
        "revision": "r1",
        "base_sha": BASE_SHA,
        "control_sha": CONTROL_SHA,
        "candidate_sha": CANDIDATE_SHA,
        "control_arm": "unchanged base behaviour, truncates at the stop token",
        "candidate_arm": "both early-return sites deleted, accept-loop cap kept",
        "fixture": "correctness_prompts/public_longcopy_gate_english_512_1024",
        "golden_rows": 513,
        "golden_stop_token": 248044,
        "golden_stop_token_index": 300,
        "emitted_stop_token_index": 301,
        "stop_token_set": [248044, 248046],
        "seed_tokens": 512,
        "host": "M4 Pro applegpu_g16s, NAX off (ranked host is M5)",
        "gpu_temperature_recorded": False,
        "speed_claim": False,
        "prereg_falsifiers": FALSIFIERS,
        "primary_metric": "max_local_decode_tokens_with_all_tokens_matched",
        "primary_metric_direction": "maximize",
        **{f"head/{key}": headline["report"].get(key) for key in PROVENANCE},
    }

    if args.dry_run:
        run = None
    else:
        run = wandb.init(entity=ENTITY, project=PROJECT, group=args.group,
                         name=args.name, notes=args.notes, config=config,
                         job_type="correctness",
                         tags=["e26", "stop-token", "fixed-window", "r1"])

    columns = ["arm", "tokens", "depth", "ran", "exit_code", "wall_seconds",
               "draft_schedule_sha256_16"] + CORRECTNESS + TIMING
    table = None if args.dry_run else wandb.Table(columns=columns)
    summary = {}

    for leg in legs:
        rep = leg["report"]
        digest, _ = draft_schedule_digest(rep)
        if table is not None:
            table.add_data(leg["arm"], leg["tokens"], leg["depth"], leg["ran"],
                           leg["exit_code"], leg["wall_seconds"], digest,
                           *[rep.get(key) for key in CORRECTNESS + TIMING])
        prefix = leg_key(leg)
        summary[f"{prefix}/ran"] = leg["ran"]
        summary[f"{prefix}/exit_code"] = leg["exit_code"]
        summary[f"{prefix}/wall_seconds"] = leg["wall_seconds"]
        if digest is not None:
            summary[f"{prefix}/draft_schedule_sha256_16"] = digest
        for key in CORRECTNESS + TIMING:
            value = rep.get(key)
            # Per-round arrays live in the table; the summary stays scalar.
            if value is not None and not isinstance(value, list):
                summary[f"{prefix}/{key}"] = value

    for name, survived in falsifiers.items():
        summary[f"falsifier/{name}_survived"] = survived
    summary["falsifier/all_survived"] = all(falsifiers.values())

    summary["primary/baseline"] = baseline
    summary["primary/candidate"] = candidate
    summary["primary/delta"] = candidate - baseline
    summary["primary/name"] = "max_local_decode_tokens_with_all_tokens_matched"
    summary["primary/direction"] = "maximize"

    # Directional only: no thermal record was taken, so these ratios explain
    # the legs, they do not claim a score.
    for arm, tokens in (("candidate", 512), ("control", 302)):
        serial = next((leg for leg in legs if leg["arm"] == arm
                       and leg["tokens"] == tokens and leg["depth"] == 0
                       and leg["ran"]), None)
        mtp = next((leg for leg in legs if leg["arm"] == arm
                    and leg["tokens"] == tokens and leg["depth"] == 2
                    and leg["ran"]), None)
        if serial and mtp:
            summary[f"directional/{arm}_t{tokens}_local_ratio"] = (
                serial["report"]["parent_measured_seconds_per_token"]
                / mtp["report"]["parent_measured_seconds_per_token"])

    print(f"legs_logged={len(legs)} ran={sum(1 for l in legs if l['ran'])}")
    print(f"falsifiers={falsifiers}")
    print(f"primary baseline={baseline} candidate={candidate}")
    if run is None:
        for key in sorted(summary):
            print(f"  {key} = {summary[key]}")
        return
    run.log({"legs": table})
    run.summary.update(summary)
    print(f"run_id={run.id}")
    print(f"run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
