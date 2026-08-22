#!/usr/bin/env python3
"""Publish the E121 cross-simdgroup activation chunk-sum share study to W&B.

    usage: research/e121_wandb_log.py [--only RUN ...]

THE QUESTION. Inside the wide multi-row QMV body both simdgroups of a
threadgroup walk the SAME activation range and each computes the SAME per-lane
chunk sum `sums[m]`. One of the two copies is redundant work. The arm splits
the `m` range between the simdgroups, exchanges the halves through 512 bytes of
threadgroup memory, and gates the whole construct off at NA = 5 with
`if constexpr (NA <= 4)`.

WHY THE GATE. Ungated, NA = 5 spills on the local `applegpu_g16s` and the cell
collapses by 30 %. The gate also matters on the ranked `applegpu_g17s`, where
it recovers entry-point occupancy: 120 registers with no gate against 102 with
it, which is 33 against 38 simdgroups per core.

FOUR RUNS:

  `e121-rung0-census`
      Static instruments for every arm on both GPU generations: AIR load,
      barrier and threadgroup-byte counts after -O2, plus registers, spill and
      machine-text digests. It also carries the gate-exactness census: at
      NA = 5 every gated arm must hash to the same machine text as `a_base`.
  `e121-rung2-isolated`
      The isolated timing sweep over five scored shapes and NA = 2..5, the
      three instrument validity gates, the cost-weighted shipped frame, and the
      four pre-registered predictions with their scores.
  `e121-rung3-insitu`
      The shipped transplant: full 512-token exactness against the pinned row
      digest, and matched ABBA absolute candidate MTP seconds per token against
      a fresh unchanged base in the same session.
  `e121-rung3-presubmit`
      The pre-submit chain on the exact tree that would be submitted.

Every timed leg here ran with no thermal gate, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim except on the `--local-submit` leg, which runs the real gate. No
number here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e121-cross-simdgroup-activation-sum-sharing"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
ART = pathlib.Path("research/e121-artifacts")

BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 122
ENTRY_CELL = ("qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true> "
              "k-block chunk-sum accumulation, reached through "
              "affine_qmv_fast<bfloat16_t, 64, 4, false>")
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
BASE_ARM = "a_base"

# The realised verify-width histogram of the fixture, from the advisor.
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}
# E116 measured kernel-percent -> leg-seconds-percent transfer on this tree.
E116_KERNEL_TO_LEG = 0.607
# Rule 34 local-leg -> ranked transfer.
RANKED_TRANSFER = 0.95
# Rule 59 submission bar, round-weighted kernel frame.
SUBMISSION_BAR_PCT = 1.0

ARM_NOTE = {
    "a_base": "shipped wide body, unmodified",
    "a_scaffold": "byte-identical control: the arm harness with no source "
                  "change, so its reading is the instrument null",
    "x_split_pred": "UNGATED split predicate: m < NA/2 owns the low half, the "
                    "other simdgroup owns the high half",
    "x_min_ask": "UNGATED, Askeladd's ownership form ported onto the xv4 base",
    "g_min_ask": "x_min_ask with the if constexpr (NA <= 4) gate",
    "g_split_pred_pp": "g_split_pred with a ping-pong buffer in place of the "
                       "second barrier",
    "g_split_pred": "the shipped arm: split predicate plus the "
                    "if constexpr (NA <= 4) gate",
}


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            meta[key] = value
    return meta


def start(name: str, job_type: str, question: str, rung: int, config: dict,
          meta: dict | None = None):
    meta = meta if meta is not None else read_meta(ART / "rung2-meta.txt")
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "rung": rung, "question": question,
            "entry_cell": ENTRY_CELL, "host": HOST,
            "hostname": meta.get("host"), "chip": meta.get("chip"),
            "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "metal_toolchain": meta.get("metal_toolchain"),
            "git_head": meta.get("git_head"),
            "base_sha": BASE_SHA, "advisor_branch": ADVISOR_BRANCH,
            "pr_number": PR_NUMBER,
            "local_arch": LOCAL_ARCH, "ranked_arch": RANKED_ARCH,
            "round_weights": {str(k): v for k, v in ROUND_WEIGHTS.items()},
            "e116_kernel_to_leg": E116_KERNEL_TO_LEG,
            "ranked_transfer_factor": RANKED_TRANSFER,
            "submission_bar_pct": SUBMISSION_BAR_PCT,
            **config, **gate_flags(),
        },
        reinit=True,
    )


def attach(run, *paths: pathlib.Path) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="analysis")
    for path in present:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def log_census() -> None:
    path = ART / "rung0-census.json"
    if not path.exists():
        print(f"[wandb] no {path}")
        return
    census = json.loads(path.read_text())
    widths = [int(w) for w in census["widths"]]
    arms = list(census["arms"])

    run = start(
        "e121-rung0-census", "census",
        "Does the split predicate change the static instruments, and is the "
        "if constexpr gate exact at NA = 5 by machine text?",
        0, {"arms": arms, "widths": widths, "arm_notes": ARM_NOTE,
            "instrument": "post -O2 AIR counts plus metal-tt registers, spill "
                          "and machine-text digest on both architectures"})

    air = wandb.Table(columns=[
        "arm", "cell", "device_loads", "threadgroup_loads",
        "threadgroup_stores", "threadgroup_bytes", "barriers", "fadd",
        "air_lines", "note"])
    for arm in arms:
        for cell, rec in census["arms"][arm]["air"].items():
            air.add_data(arm, cell, rec["device_loads"],
                         rec["threadgroup_loads"], rec["threadgroup_stores"],
                         rec["threadgroup_bytes"], rec.get("barriers"),
                         rec.get("fadd"), rec["air_lines"],
                         ARM_NOTE.get(arm, ""))
    run.log({"air_census": air})

    text = wandb.Table(columns=[
        "arm", "arch", "cell", "registers", "spill_bytes", "text_bytes",
        "text_sha8"])
    for arm in arms:
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for cell, rec in census["arms"][arm].get(arch, {}).items():
                text.add_data(arm, arch, cell, rec.get("registers"),
                              rec.get("spill_bytes"), rec.get("text_bytes"),
                              rec.get("text_sha8"))
    run.log({"machine_text_census": text})

    # The gate claim. At NA = 5 every gated arm must be byte-identical machine
    # text to the base, and every ungated arm must differ. A census that could
    # not separate them would not be evidence.
    gate = wandb.Table(columns=[
        "arm", "arch", "gated", "text_sha8_na5", "base_text_sha8_na5",
        "identical_to_base", "expected_identical", "as_expected"])
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        base = census["arms"][BASE_ARM].get(arch, {}).get("5", {})
        for arm in arms:
            if arm == BASE_ARM:
                continue
            rec = census["arms"][arm].get(arch, {}).get("5")
            if rec is None:
                continue
            gated = arm.startswith("g_") or arm == "a_scaffold"
            same = rec.get("text_sha8") == base.get("text_sha8")
            gate.add_data(arm, arch, gated, rec.get("text_sha8"),
                          base.get("text_sha8"), same, gated, same == gated)
    run.log({"gate_exactness_na5": gate})

    summary: dict[str, object] = {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        tag = arch.replace("applegpu_", "")
        for arm in arms:
            per_arch = census["arms"][arm].get(arch, {})
            for cell, rec in per_arch.items():
                summary[f"registers/{arm}_{tag}_{cell}"] = rec.get("registers")
                summary[f"spill_bytes/{arm}_{tag}_{cell}"] = rec.get("spill_bytes")
    run.summary.update(summary)
    attach(run, path)
    run.finish()


def log_isolated() -> None:
    path = ART / "rung2-summary.json"
    if not path.exists():
        print(f"[wandb] no {path}")
        return
    doc = json.loads(path.read_text())
    rate_path = ART / "rung2-rate.json"
    rate = json.loads(rate_path.read_text()) if rate_path.exists() else {}
    arms = list(doc["arms"])

    run = start(
        "e121-rung2-isolated", "timing",
        "Does sharing the redundant chunk sum across the two simdgroups lower "
        "the wide QMV time at the realised verify-width operating point?",
        2, {"arms": arms, "arm_notes": ARM_NOTE,
            "shapes": doc.get("shapes"), "widths": doc.get("widths"),
            "warmup_blocks_discarded": doc.get("warmup_blocks_discarded"),
            "cost_share_na4": doc.get("cost_share", {}).get("4"),
            "device": rate.get("device"),
            "architecture": rate.get("architecture"),
            "pairs": rate.get("pairs"), "order": rate.get("order")})

    ladder = wandb.Table(columns=[
        "arm", "na", "pooled_pct", "cost_weighted_pct", "note"])
    for arm in arms:
        rec = doc["arms"][arm]
        for width in sorted(rec["per_width_pct"], key=int):
            ladder.add_data(arm, int(width), rec["per_width_pct"][width],
                            rec["cost_weighted_per_width_pct"].get(width),
                            ARM_NOTE.get(arm, ""))
    run.log({"per_width_ladder": ladder})

    frames = wandb.Table(columns=[
        "arm", "round_weighted_pct", "round_weighted_ex_na5_pct",
        "cost_weighted_round_pct", "predicted_leg_pct", "predicted_ranked_pct",
        "ci95_lo_ex_na5", "ci95_hi_ex_na5", "clears_submission_bar"])
    for arm in arms:
        rec = doc["arms"][arm]
        ci = rec.get("ci95_ex_na5") or [None, None]
        frames.add_data(arm, rec.get("round_weighted_pct"),
                        rec.get("round_weighted_ex_na5_pct"),
                        rec.get("cost_weighted_round_pct"),
                        rec.get("predicted_leg_pct"),
                        rec.get("predicted_ranked_pct"), ci[0], ci[1],
                        rec.get("clears_submission_bar"))
    run.log({"score_frames": frames})

    validity = wandb.Table(columns=["gate", "observed", "limit", "passed"])
    for name, rec in sorted(doc.get("validity", {}).items()):
        if isinstance(rec, dict):
            validity.add_data(name, rec.get("observed"), rec.get("limit"),
                              rec.get("passed"))
    run.log({"validity_gates": validity})

    pre = wandb.Table(columns=["prediction", "band", "measured", "hit"])
    for name, rec in sorted(doc.get("preregistered", {}).items()):
        if isinstance(rec, dict):
            pre.add_data(name, str(rec.get("band")), rec.get("measured"),
                         rec.get("hit"))
    run.log({"preregistered_predictions": pre})

    if rate.get("measurements"):
        fidelity = wandb.Table(columns=[
            "shape", "na", "arm", "exact_required", "differing", "total",
            "bit_identical"])
        for row in rate["measurements"]:
            if row.get("kind") != "fidelity":
                continue
            for rec in row["arms"]:
                fidelity.add_data(row["shape"], row["m"], rec["arm"],
                                  rec["exact_required"], rec["differing"],
                                  rec["total"], rec["bit_identical"])
        run.log({"fidelity": fidelity})

    summary: dict[str, object] = {
        "exactness_failures": doc.get("exactness_failures"),
        "positive_controls": doc.get("positive_controls"),
        "thermal": doc.get("thermal"),
    }
    for arm in arms:
        rec = doc["arms"][arm]
        summary[f"cost_weighted_round_pct/{arm}"] = rec.get(
            "cost_weighted_round_pct")
        summary[f"round_weighted_pct/{arm}"] = rec.get("round_weighted_pct")
        summary[f"predicted_ranked_pct/{arm}"] = rec.get("predicted_ranked_pct")
    run.summary.update(summary)
    attach(run, path, rate_path, ART / "rung2-meta.txt")
    run.finish()


def log_insitu() -> None:
    path = ART / "rung3-e2e.json"
    if not path.exists():
        print(f"[wandb] no {path}; run research/e121_e2e_analyse.py first")
        return
    doc = json.loads(path.read_text())

    run = start(
        "e121-rung3-insitu", "timing",
        "Does the gated chunk-sum share lower absolute candidate MTP seconds "
        "per token in the real worker, with full-window exactness intact?",
        3, {"arm": doc["arm"], "candidate_commit": doc["candidate_commit"],
            "base_commit": doc["base_commit"],
            "worker_fingerprint": doc.get("worker_fingerprint"),
            "token_window": doc.get("token_window"), "order": doc.get("order"),
            "replicates": doc.get("replicates"),
            "reproduction": doc.get("reproduction"),
            "prediction": doc.get("prediction")})

    legs = wandb.Table(columns=[
        "replicate", "position", "tree", "tag", "seconds_per_token",
        "serial_seconds_per_token", "local_ratio", "effective_mean_draft_len",
        "accepted_draft_rate", "all_tokens_matched", "gpu_temp_entry_c",
        "gpu_temp_exit_c", "worker_sha256"])
    for leg in doc["legs"]:
        legs.add_data(leg["replicate"], leg["position"], leg["tree"],
                      leg.get("tag"), leg["seconds_per_token"],
                      leg.get("serial_seconds_per_token"),
                      leg.get("local_ratio"),
                      leg.get("effective_mean_draft_len"),
                      leg.get("accepted_draft_rate"),
                      leg.get("all_tokens_matched"),
                      leg.get("gpu_temp_entry_c"), leg.get("gpu_temp_exit_c"),
                      leg.get("worker_sha256"))
    run.log({"abba_legs": legs})

    per_replicate = wandb.Table(columns=[
        "replicate", "mtp_spt_base", "mtp_spt_share", "mtp_spt_pct",
        "serial_spt_pct", "ratio_base", "ratio_share", "ratio_pct",
        "draftlen_pct", "acceptrate_pct", "base_pair_drift_pct"])
    for rec in doc.get("per_replicate", []):
        per_replicate.add_data(
            rec["replicate"], rec["mtp_spt_base"], rec["mtp_spt_share"],
            rec["mtp_spt_pct"], rec["serial_spt_pct"], rec["ratio_base"],
            rec["ratio_share"], rec["ratio_pct"], rec["draftlen_pct"],
            rec["acceptrate_pct"], rec["base_pair_drift_pct"])
    run.log({"per_replicate": per_replicate})

    exact = wandb.Table(columns=["check", "rows", "expected", "observed",
                                 "passed"])
    for rec in doc.get("exactness", []):
        exact.add_data(rec["check"], rec.get("rows"), rec.get("expected"),
                       rec.get("observed"), rec["passed"])
    run.log({"exactness": exact})

    run.summary.update(doc["summary"])
    run.summary.update({f"prediction/{k}": v
                        for k, v in doc.get("prediction", {}).items()})
    attach(run, path, ART / "row-digest-512.json", ART / "e121-share.patch")
    run.finish()


def log_presubmit() -> None:
    path = ART / "rung3-presubmit.json"
    if not path.exists():
        print(f"[wandb] no {path}; run research/e121_presubmit.sh first")
        return
    doc = json.loads(path.read_text())
    submit = doc.get("local_submit", {})
    metrics = submit.get("metrics", {})

    run = start(
        "e121-rung3-presubmit", "validation",
        "Does the gated chunk-sum share pass every pre-submit gate on the "
        "exact tree that would be submitted?",
        3, {"arm": doc["arm"], "candidate_commit": doc["candidate_commit"],
            "base_sha": doc.get("base_sha"),
            "submitted_paths": doc.get("submitted_paths"),
            "worker_sha256": doc.get("worker_sha256"),
            "local_mode": "--local-submit",
            "token_window": metrics.get("decode_tokens")})
    # The --local-submit leg is the only one here that ran the real gate.
    if submit:
        run.config.update({"cool_gate_passed_real_gate": True,
                           "gate_qualified_for_timing": True,
                           "timing_valid": False}, allow_val_change=True)

    gates = wandb.Table(columns=["step", "command", "exit", "passed",
                                 "observation"])
    for step in doc.get("steps", []):
        gates.add_data(step["step"], step["command"], step["exit"],
                       step["passed"], step["observation"])
    run.log({"gates": gates})

    run.summary.update({
        "chain_passed": doc.get("passed"),
        "local_submit_passed": submit.get("passed"),
        "cool_gate_passed_real_gate": bool(submit),
        "gate_qualified_for_timing": bool(submit),
        "timing_valid": False,
        "official_or_ranked_score": False,
        **{f"local_submit/{k}": v for k, v in metrics.items()},
    })
    attach(run, path)
    run.finish()


RUNS = {"census": log_census, "isolated": log_isolated, "insitu": log_insitu,
        "presubmit": log_presubmit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS), nargs="*")
    args = ap.parse_args()
    for name in (args.only or list(RUNS)):
        print(f"[wandb] {name}")
        RUNS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
