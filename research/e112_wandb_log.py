#!/usr/bin/env python3
"""Publish every E112 session to W&B, including the zero-GPU analysis rungs.

    usage: research/e112_wandb_log.py [--only RUN]

  `e112-rung1-q1-abba`
      The one GPU session. Nine ABBA legs test whether deleting the kL=1025
      128-block SDPA compile warm changes candidate MTP seconds per token.

  `e112-f1-replicate-floor`
      Zero GPU. What IS a board replicate pair, and what does the corrected
      canonicaliser do to the measured floor? Carries the class x gap x solver
      tables, the stratified variance-ratio tests, the byte-identical
      provenance grep, and the string-aware against naive stripper check.

  `e112-f1-mechanism-survivors`
      Zero GPU. Which single-path board mechanisms still clear 2 sigma once
      they are priced against the corrected floor instead of the old 0.0431 %?

The ABBA legs ran with `MLXFAST_LOCAL_COOL_GATE=0` under the counterbalanced
exception. Every leg logs `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false`
verbatim. No number here is a score.

Board numbers come from ONE frozen snapshot, because `yukon` refreshes its
cache in place and a live file cannot support a reproducible count. The
snapshot digest and row count are in the config of both analysis runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board_prompt_instrument import (  # noqa: E402
    ANCHOR, BOARD_JSON, CONSERVATIVE, MODE_DRAFT_SHIFT, RESOLUTION,
    SOURCE_SUFFIXES, ZERO, canon_digest, changed_blobs, code_identity, collect,
    f_two_sided_p, load_rows, per_run_sd, replicate_pairs)
from e112_contrast import neighbour_contrast  # noqa: E402
from e112_survivors import mine, normal_two_sided, price  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e112-board-mined-target-path-edits"
ARTIFACTS = pathlib.Path("research/e112-artifacts")
ABBA = ARTIFACTS / "rung1-abba.json"

BASE_SHA = "b129f202fc25413015463da559777aaa59534065"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BAR_LOCAL_PCT = 0.20  # the assignment promotion bar
MIN_SIGMA = 2.0
MAX_LINES = 40


def identity() -> dict:
    return {
        "experiment": "e112-board-mined-target-path-edits",
        "harness": "local",
        "base_sha": BASE_SHA,
        "board_anchor": ANCHOR,
        "host": HOST,
        "gpu_cores": 20,
        "chip": "Apple M4 Pro",
        "device_class": "AGXG16SDevice",
        "ranked_runner_chip": "M5",
        "promotion_bar_local_pct": BAR_LOCAL_PCT,
    }


def gate_flags(kind: str) -> dict:
    return {
        "leg_kind": kind,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }


def board_identity() -> dict:
    raw = pathlib.Path(BOARD_JSON).read_bytes()
    rows = load_rows()
    return {
        "board_snapshot_path": BOARD_JSON,
        "board_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "board_snapshot_bytes": len(raw),
        "board_rows": len(rows),
        "board_latest_created_at": max(r.get("createdAt") or "" for r in rows),
    }


def log_abba() -> str:
    report = json.loads(ABBA.read_text())
    legs = report["legs"]
    temps = [leg["entry_c"] for leg in legs]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="abba-timing", name="e112-rung1-q1-abba",
        config={
            "rung": "1",
            "question": (
                "does deleting the kL=1025 128-block SDPA compile warm change "
                "candidate MTP seconds per token at the scored 512-token cell"
            ),
            "mechanism": (
                "Qwen36MTPBlockSession.swift:555-581 warms an SDPA shape at "
                "kL=1025 that the 512-token trace never reaches; the arm "
                "switch MLX_E112_SKIP_1025_WARM=1 removes it"
            ),
            "design": "ABBA, off on off on off on off on off, one session",
            "arms": sorted({leg["arm"] for leg in legs}),
            "legs": len(legs),
            "decode_tokens": legs[0]["decode_tokens"],
            "session_commit": legs[0]["session_commit"],
            "worker_sha256": legs[0]["worker_sha256"],
            "all_tokens_matched": all(leg["all_tokens_matched"] for leg in legs),
            "entry_temp_min_c": min(temps),
            "entry_temp_max_c": max(temps),
            "entry_temp_spread_c": max(temps) - min(temps),
            "job_id": "2247e9b7-dfea-4be9-95dd-43d878591ea6",
            "leg_command": "research/e112_abba.sh",
            **identity(),
            **gate_flags("kL=1025 warm ABBA, GPU timing"),
        },
        reinit=True,
    )

    leg_table = wandb.Table(columns=[
        "tag", "position", "arm", "decode_tokens", "mtp_seconds_per_token",
        "serial_seconds_per_token", "speedup", "mtp_median_round_us",
        "mtp_rounds", "mtp_max_key_len", "mean_draft_len",
        "accepted_draft_rate", "all_tokens_matched", "gpu_temp_entry_c",
        "gpu_temp_exit_c", "crossing_round", "crossing_round_us",
        "worker_sha256",
    ])
    for leg in legs:
        crossing = (leg["mtp_crossings"] or [{}])[0]
        leg_table.add_data(
            leg["tag"], leg["position"], leg["arm"], leg["decode_tokens"],
            leg["mtp_s_per_tok"], leg["serial_s_per_tok"], leg["speedup"],
            leg["mtp_median_round_us"], leg["mtp_rounds"],
            leg["mtp_max_key_len"], leg["mean_draft_len"],
            leg["accepted_draft_rate"], leg["all_tokens_matched"],
            leg["entry_c"], leg["exit_c"], crossing.get("round"),
            crossing.get("round_us"), leg["worker_sha256"])
    run.log({"rung1/legs": leg_table})

    arm_table = wandb.Table(columns=[
        "metric", "off_mean", "off_sd", "off_n", "on_mean", "on_sd", "on_n",
        "delta", "delta_pct",
    ])
    for name in ("mtp_s_per_tok", "serial_s_per_tok", "speedup",
                 "mtp_median_round_us", "entry_c"):
        block = report[name]
        arm_table.add_data(
            name, block["off"]["mean"], block["off"]["sd"], block["off"]["n"],
            block["on"]["mean"], block["on"]["sd"], block["on"]["n"],
            block["delta"], block["delta_pct"])
    run.log({"rung1/arm_means": arm_table})

    contrast_table = wandb.Table(columns=[
        "quantity", "mean_pct", "sd_pct", "se_pct", "t", "n", "values_pct",
    ])
    contrasts = {}
    for label, key in (
        ("absolute candidate MTP s/token", "mtp_s_per_tok"),
        ("local serial-to-MTP ratio", "speedup"),
        ("median round us", "mtp_median_round_us"),
    ):
        values = neighbour_contrast(legs, key)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        se = sd / math.sqrt(len(values))
        contrast_table.add_data(label, mean, sd, se, mean / se, len(values),
                                json.dumps([round(v, 4) for v in values]))
        contrasts[label] = (mean, se)
    run.log({"rung1/neighbour_contrast": contrast_table})

    off = [leg["mtp_s_per_tok"] for leg in legs if leg["arm"] == "off"]
    on = [leg["mtp_s_per_tok"] for leg in legs if leg["arm"] == "on"]
    sd_off = 100.0 * statistics.stdev(off) / statistics.fmean(off)
    sd_on = 100.0 * statistics.stdev(on) / statistics.fmean(on)
    pooled = math.sqrt(((len(off) - 1) * sd_off ** 2
                        + (len(on) - 1) * sd_on ** 2)
                       / (len(off) + len(on) - 2))
    mean_pct, se_pct = contrasts["absolute candidate MTP s/token"]
    leg_us = 15_800_830.0
    crossing_delta_us = 439.0
    run.summary.update({
        "rung1/contrast_abs_mtp_pct": mean_pct,
        "rung1/contrast_abs_mtp_se_pct": se_pct,
        "rung1/contrast_abs_mtp_t": mean_pct / se_pct,
        "rung1/per_leg_sd_off_pct": sd_off,
        "rung1/per_leg_sd_on_pct": sd_on,
        "rung1/per_leg_sd_pooled_pct": pooled,
        "rung1/arm_mean_diff_se_pct": pooled * math.sqrt(1 / len(off)
                                                         + 1 / len(on)),
        "rung1/leg_wall_us": leg_us,
        "rung1/crossing_round_arm_delta_us": crossing_delta_us,
        "rung1/crossing_round_share_of_leg_pct": (100.0 * crossing_delta_us
                                                  / leg_us),
        "rung1/bar_local_pct": BAR_LOCAL_PCT,
        "rung1/clears_bar": False,
        "rung1/verdict": "not useful",
    })

    print(f"e112-rung1-q1-abba  {run.id}  {run.url}")
    url = run.url
    run.finish()
    return url


def _floor_cells(sub):
    same = [p for p in sub if p["same_mode"]]
    return (len(sub), len(same),
            per_run_sd([p["target"] for p in same]),
            per_run_sd([p["draft"] for p in same]),
            per_run_sd([p["target"] for p in sub]),
            per_run_sd([p["draft"] for p in sub]))


def log_floor() -> str:
    rows = load_rows()
    recs = collect(rows)
    groups, pairs = replicate_pairs(recs)
    byte_pairs = [p for p in pairs if p["bytes_equal"]]
    comment_pairs = [p for p in pairs if not p["bytes_equal"]]
    classes = [("byte-identical", byte_pairs),
               ("comment-only diff", comment_pairs),
               ("all code-identical", pairs)]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="board-analysis", name="e112-f1-replicate-floor",
        config={
            "rung": "f1",
            "question": (
                "what is a board replicate pair, and what does a "
                "string-literal-aware canonicaliser do to the measured "
                "TARGET and DRAFT floor"
            ),
            "replicate_rule": "comment-insensitive code identity, string-aware",
            "usable_rows": len(recs),
            "identity_schedule_groups": len(groups),
            "replicated_groups": sum(1 for m in groups.values() if len(m) > 1),
            "pairs": len(pairs),
            "mode_draft_shift_pct": MODE_DRAFT_SHIFT,
            "prior_adopted_target_per_run_pct": 0.1196,
            "prior_adopted_target_per_pair_pct": 0.1691,
            **board_identity(),
            **identity(),
            **gate_flags("board analysis, zero GPU"),
        },
        reinit=True,
    )

    cols = ["group", "pairs", "same_mode_pairs", "target_same_mode_pct",
            "draft_same_mode_pct", "target_all_pct", "draft_all_pct"]

    class_table = wandb.Table(columns=cols)
    for label, sub in classes:
        class_table.add_data(label, *_floor_cells(sub))
    run.log({"floor/replicate_class": class_table})

    gap_table = wandb.Table(columns=cols)
    for label, sub in classes:
        for gap_label, keep in (("< 3 h", lambda p: p["gap"] < 3),
                                (">= 3 h", lambda p: p["gap"] >= 3)):
            gap_table.add_data(f"{label}, {gap_label}",
                               *_floor_cells([p for p in sub if keep(p)]))
    run.log({"floor/class_by_time_gap": gap_table})

    solver_table = wandb.Table(columns=cols)
    for label, sub in classes:
        for s_label, keep in (("same solver", lambda p: p["same_solver"]),
                              ("diff solver", lambda p: not p["same_solver"])):
            solver_table.add_data(f"{label}, {s_label}",
                                  *_floor_cells([p for p in sub if keep(p)]))
    run.log({"floor/class_by_solver": solver_table})

    def same_target(sub):
        return [p["target"] for p in sub if p["same_mode"]]

    tests = [
        ("gap < 3 h", "comment-only vs byte-identical",
         [p for p in comment_pairs if p["gap"] < 3],
         [p for p in byte_pairs if p["gap"] < 3]),
        ("gap >= 3 h", "comment-only vs byte-identical",
         [p for p in comment_pairs if p["gap"] >= 3],
         [p for p in byte_pairs if p["gap"] >= 3]),
        ("byte-identical", "< 3 h vs >= 3 h",
         [p for p in byte_pairs if p["gap"] < 3],
         [p for p in byte_pairs if p["gap"] >= 3]),
        ("comment-only", "< 3 h vs >= 3 h",
         [p for p in comment_pairs if p["gap"] < 3],
         [p for p in comment_pairs if p["gap"] >= 3]),
        ("nothing", "comment-only vs byte-identical",
         comment_pairs, byte_pairs),
        ("nothing", "< 3 h vs >= 3 h",
         [p for p in pairs if p["gap"] < 3],
         [p for p in pairs if p["gap"] >= 3]),
    ]
    f_table = wandb.Table(columns=["held_fixed", "contrast", "n1", "n2",
                                   "F", "p_two_sided"])
    for held, contrast, sub1, sub2 in tests:
        v1, v2 = same_target(sub1), same_target(sub2)
        if not v1 or not v2:
            f_table.add_data(held, contrast, len(v1), len(v2), None, None)
            continue
        var1 = per_run_sd(v1, 1) ** 2
        var2 = per_run_sd(v2, 1) ** 2
        f_table.add_data(held, contrast, len(v1), len(v2), var1 / var2,
                         f_two_sided_p(var1, len(v1), var2, len(v2)))
    run.log({"floor/stratified_f_tests": f_table})

    floor_table = wandb.Table(columns=["probe", "basis", "per_run_pct",
                                       "per_pair_pct", "in_use_pct"])
    floor_table.add_data("TARGET", "conservative, same mode",
                         CONSERVATIVE["target_per_run"],
                         CONSERVATIVE["target_per_pair"],
                         CONSERVATIVE["target_per_run"])
    floor_table.add_data("TARGET", "point estimate, same mode",
                         RESOLUTION["target_same_mode"],
                         RESOLUTION["target_same_mode"] * math.sqrt(2),
                         RESOLUTION["target_same_mode"])
    floor_table.add_data("DRAFT", "conservative, same mode",
                         CONSERVATIVE["draft_per_run"],
                         CONSERVATIVE["draft_per_pair"],
                         CONSERVATIVE["draft_per_run"])
    floor_table.add_data("DRAFT", "point estimate, same mode",
                         RESOLUTION["draft_same_mode"],
                         RESOLUTION["draft_same_mode"] * math.sqrt(2),
                         RESOLUTION["draft_same_mode"])
    run.log({"floor/adopted": floor_table})

    prov = wandb.Table(columns=[
        "target_pct", "draft_pct", "gap_hours", "same_solver",
        "a_id8", "a_solver", "a_commit_sha", "a_promotion_status", "a_status",
        "b_id8", "b_solver", "b_commit_sha", "b_promotion_status", "b_status",
        "same_commit_sha",
    ])
    same_solver = both_sha = same_sha = mixed_outcome = 0
    for pair in sorted(byte_pairs, key=lambda p: p["a"]["date"]):
        a, b = pair["a"]["row"], pair["b"]["row"]
        sha_a = a.get("submissionCommitSha")
        sha_b = b.get("submissionCommitSha")
        if sha_a and sha_b:
            both_sha += 1
            same_sha += int(sha_a == sha_b)
            verdict = "yes" if sha_a == sha_b else "no"
        else:
            verdict = "unknown"
        same_solver += int(pair["same_solver"])
        mixed_outcome += int(a.get("promotionStatus")
                             != b.get("promotionStatus"))
        prov.add_data(
            pair["target"], pair["draft"], pair["gap"], pair["same_solver"],
            pair["a"]["id8"], pair["a"]["solver"], sha_a,
            a.get("promotionStatus"), a.get("status"),
            pair["b"]["id8"], pair["b"]["solver"], sha_b,
            b.get("promotionStatus"), b.get("status"), verdict)
    run.log({"floor/byte_identical_provenance": prov})

    blob_diff = blob_total = 0
    for rec in recs:
        for path, (_, new_oid) in (changed_blobs(rec["ref"]) or {}).items():
            if new_oid == ZERO or not path.endswith(SOURCE_SUFFIXES):
                continue
            blob_total += 1
            if canon_digest(new_oid, path, True) != canon_digest(new_oid, path,
                                                                 False):
                blob_diff += 1
    differing = sum(1 for rec in recs
                    if code_identity(rec["ref"], True)
                    != code_identity(rec["ref"], False))
    naive_groups = {}
    for rec in recs:
        key = code_identity(rec["ref"], False)
        if key is not None:
            naive_groups.setdefault((key, rec["sig"]), []).append(rec)
    naive_n = sum(len(m) * (len(m) - 1) // 2 for m in naive_groups.values())
    run.summary.update({
        "floor/target_conservative_per_run_pct": CONSERVATIVE["target_per_run"],
        "floor/target_conservative_per_pair_pct": CONSERVATIVE["target_per_pair"],
        "floor/draft_conservative_per_run_pct": CONSERVATIVE["draft_per_run"],
        "floor/draft_conservative_per_pair_pct": CONSERVATIVE["draft_per_pair"],
        "floor/target_point_estimate_per_run_pct":
            RESOLUTION["target_same_mode"],
        "floor/draft_point_estimate_per_run_pct": RESOLUTION["draft_same_mode"],
        "floor/byte_identical_pairs": len(byte_pairs),
        "floor/byte_pairs_same_solver": same_solver,
        "floor/byte_pairs_with_both_commit_shas": both_sha,
        "floor/byte_pairs_with_same_commit_sha": same_sha,
        "floor/byte_pairs_disagreeing_on_promotion": mixed_outcome,
        "canon/blobs_inspected": blob_total,
        "canon/blobs_where_strippers_differ": blob_diff,
        "canon/identities_where_strippers_differ": differing,
        "canon/pairs_string_aware_rule": len(pairs),
        "canon/pairs_naive_regex_rule": naive_n,
        "canon/false_replicates_removed": naive_n - len(pairs),
        "canon/verdict": ("string-aware and naive disagree; the naive regex "
                          "merged trees that JIT-compile different Metal "
                          "source"),
    })

    print(f"e112-f1-replicate-floor  {run.id}  {run.url}")
    url = run.url
    run.finish()
    return url


def log_survivors() -> str:
    per_run = CONSERVATIVE["target_per_run"]
    per_pair = per_run * math.sqrt(2)
    recs = collect(load_rows())
    groups = mine(recs)
    rows = price(groups, per_run)
    npairs = sum(len(v) for v in groups.values())

    clear = [r for r in rows if abs(r["sigma"]) >= MIN_SIGMA]
    expected = len(rows) * normal_two_sided(MIN_SIGMA)
    single = [r for r in rows if r["size"] <= MAX_LINES]
    replicated = [r for r in single if r["n"] > 1]
    passing = [r for r in replicated if abs(r["sigma"]) >= MIN_SIGMA]
    survivors = [r for r in passing if not r["het"]]
    rejected = [r for r in passing if r["het"]]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="board-analysis", name="e112-f1-mechanism-survivors",
        config={
            "rung": "f1",
            "question": (
                "which single-path board mechanisms still clear 2 sigma once "
                "they are priced against the corrected floor instead of the "
                "0.0431 % byte-identical spread"
            ),
            "floor_basis": "conservative, widest replicate class",
            "per_run_floor_pct": per_run,
            "per_pair_floor_pct": per_pair,
            "min_sigma": MIN_SIGMA,
            "max_changed_canonical_lines": MAX_LINES,
            "scored_submissions": len(recs),
            "single_path_mechanisms": len(rows),
            "schedule_matched_same_mode_pairs": npairs,
            **board_identity(),
            **identity(),
            **gate_flags("board analysis, zero GPU"),
        },
        reinit=True,
    )

    table = wandb.Table(columns=[
        "class", "sigma", "pooled_pct", "pairs", "lo_runs", "hi_runs",
        "changed_canonical_lines", "between_pair_sd_pct",
        "between_pair_sd_over_floor", "homogeneous", "in_our_tree", "path",
        "edit", "pair_effects",
    ])

    def add(rows_in, label):
        for r in rows_in:
            table.add_data(
                label, r["sigma"], r["pooled"], r["n"], r["nlo"], r["nhi"],
                r["size"], r["spread"], r["spread"] / per_pair if r["n"] > 1
                else 0.0, not r["het"], r["ours"], r["path"], r["label"],
                json.dumps({f"{p['lo']['id8']}->{p['hi']['id8']}":
                            round(p["target"], 4) for p in r["pairs"]}))

    add(survivors, "defensible survivor")
    add(rejected, "rejected for heterogeneity")
    add([r for r in single
         if abs(r["sigma"]) >= MIN_SIGMA and r["n"] == 1][:10],
        "single pair, not defensible")
    add([r for r in replicated if abs(r["sigma"]) < MIN_SIGMA][:12],
        "replicated but below 2 sigma")
    run.log({"survivors/mechanisms": table})

    bonf = math.sqrt(2) * _inv_erfc(0.05 / len(rows))
    run.summary.update({
        "survivors/mechanisms_tested": len(rows),
        "survivors/clear_2_sigma": len(clear),
        "survivors/expected_by_chance": expected,
        "survivors/excess_over_null_x": len(clear) / expected,
        "survivors/bonferroni_sigma": bonf,
        "survivors/clearing_bonferroni": sum(1 for r in rows
                                             if abs(r["sigma"]) >= bonf),
        "survivors/small_edit_mechanisms": len(single),
        "survivors/small_edit_replicated": len(replicated),
        "survivors/defensible_survivors": len(survivors),
        "survivors/rejected_for_heterogeneity": len(rejected),
        "survivors/single_pair_2_sigma_threshold_pct": MIN_SIGMA * per_pair,
        "survivors/two_pair_2_sigma_threshold_pct": (MIN_SIGMA * per_pair
                                                     / math.sqrt(2)),
        "survivors/strongest_sigma": survivors[0]["sigma"] if survivors else None,
        "survivors/strongest_pooled_pct": (survivors[0]["pooled"]
                                           if survivors else None),
        "survivors/strongest_path": survivors[0]["path"] if survivors else None,
    })

    print(f"e112-f1-mechanism-survivors  {run.id}  {run.url}")
    url = run.url
    run.finish()
    return url


def _inv_erfc(y, lo=0.0, hi=10.0, iters=200):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


RUNS = {
    "e112-rung1-q1-abba": log_abba,
    "e112-f1-replicate-floor": log_floor,
    "e112-f1-mechanism-survivors": log_survivors,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args(argv)
    wanted = [args.only] if args.only else list(RUNS)
    for name in wanted:
        RUNS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
