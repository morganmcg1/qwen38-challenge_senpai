#!/usr/bin/env python3
"""Research-only: publish the E16 prefill-ladder arms and the prefill floor to W&B.

One run per ladder arm carries both timed phases of that arm (`serial/` = the
depth-0 control, `mtp/` = the depth-8 native-MTP leg) together with the
`begin` build/eval split parsed from the run log, the cool-gate temperature the
phase entered at, the exactness ledger and the binary digests that produced it.
One extra analysis run carries the cross-arm table and the Q2/Q3 prefill floor.

usage:
  research/e16_wandb.py --group G --log RUNLOG [--floor research/floor-e16.json]
                        [--summary-name NAME] [--out research/e16-wandb-runs.json]

Arms are discovered from the `e12: ##### arm TAG:LADDER` markers in the log, so
the published set is exactly what the run executed.
"""

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

ARM_RE = re.compile(r"^e12: ##### arm ([^:]+):(.*?) (\S+)$")
HEAD_RE = re.compile(r"^e12: HEAD (\S+)$")
WORKER_SHA_RE = re.compile(r"^e12: worker sha256 (\S+)$")
CLI_SHA_RE = re.compile(r"^e12: cli sha256 (\S+)$")
TAGLINE_RE = re.compile(r"^e12: tag=(\S+) tokens=(\d+) trace=(\S+) depth=(\S+)$")
LADDER_RE = re.compile(r"^e12: prefill_ladder=(.*)$")
TEMP_BEFORE_RE = re.compile(r"^e12: gpu_temp_before=(\S+) (\S+)$")
TEMP_AFTER_RE = re.compile(r"^e12: gpu_temp_after=(\S+)$")
GATE_RE = re.compile(
    r"^benchmark\.sh: GPU cool-down gate passed \(current ([\d.]+)C, "
    r"target <=([\d.]+)C, waited (\d+)s\)$"
)
SPEC_RE = re.compile(
    r"mtp-trace: prefill_ladder spec=(\S*) rungs=(\d+) at=(\S*)\s*$"
)
BEGIN_RE = re.compile(
    r"mtp-trace: begin seed=(\d+) build_us=(\d+) eval_wall_us=(\d+)"
)

# Phase order inside one --local-iterate arm: the MTP reference pass runs first
# and emits no `begin` line, so the traced/gated phases line up like this.
GATE_PHASE = {1: "serial", 2: "mtp"}
TRACE_PHASE = {0: "serial", 1: "mtp"}

PHASE_FILE = {"serial": "03-mtp-timed.json", "mtp": "04-mtp-timed.json"}

PHASE_METRICS = [
    "seed_prefill_seconds", "decode_seconds", "decode_token_count",
    "prefill_seconds_per_token", "parent_measured_seconds_per_token",
    "first_block_seconds", "p50_block_request_seconds",
    "p50_block_request_seconds_after_first", "max_block_request_seconds",
    "max_block_request_seconds_after_first", "round_count",
    "non_drafting_round_count", "accepted_draft_total", "rejected_draft_total",
    "accepted_draft_rate", "effective_mean_draft_len",
    "effective_max_draft_len", "requested_draft_depth", "mtp_depth",
    "max_draft_depth_bound", "all_tokens_matched",
    "residual_divergence_count", "declared_rows_total", "emitted_token_total",
    "reference_checked_row_total", "rejected_rows_reference_checked",
    "target_cache_offset_final", "parity_all_ok",
    "max_rejected_tail_logit_delta", "seed_token_count", "is_serial_control",
    "serial_control_depth", "uses_native_mtp_head", "uses_pinned_mtp_head",
    "mtp_head_attached", "mtp_head_tensor_count",
    "verify_block_replayed_round_count",
]

SCORE_METRICS = [
    "mtp_decode_speedup", "serial_seconds_per_token", "mtp_seconds_per_token",
    "accepted_draft_rate", "effective_mean_draft_len", "all_tokens_matched",
    "residual_divergence_count", "public_drift_tripwire_passed",
    "uses_pinned_mtp_head", "rankable", "ranked_decode_speedup_floor",
    "decode_tokens", "mtp_depth",
]

TABLE_COLUMNS = [
    "arm", "ladder_spec", "rungs", "phase", "build_us", "eval_wall_us",
    "seed_prefill_seconds", "decode_seconds", "gate_temp_c", "gate_waited_s",
    "all_tokens_matched", "residual_divergence_count", "declared_rows_total",
    "emitted_token_total", "accepted_draft_rate", "mtp_decode_speedup",
]


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config():
    swift = sh("swift", "--version")
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_cores": sh("sysctl", "-n", "hw.ncpu"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "host_os_build": sh("sysctl", "-n", "kern.osversion"),
        "swift_version": swift.splitlines()[0] if swift else "",
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain")),
    }


def parse_log(path):
    """Split the runner log into one record per `##### arm` block."""
    arms, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = ARM_RE.match(line)
        if m:
            cur = {"tag": m.group(1), "ladder_argv": m.group(2),
                   "started_at": m.group(3), "gates": [], "specs": [],
                   "begins": []}
            arms.append(cur)
            continue
        if cur is None:
            continue
        for regex, key in ((HEAD_RE, "head"), (WORKER_SHA_RE, "worker_sha256"),
                           (CLI_SHA_RE, "cli_sha256"),
                           (LADDER_RE, "ladder_env"),
                           (TEMP_AFTER_RE, "gpu_temp_after")):
            m = regex.match(line)
            if m:
                cur[key] = m.group(1)
        m = TAGLINE_RE.match(line)
        if m:
            cur["decode_tokens_requested"] = int(m.group(2))
            cur["trace"] = m.group(3)
            cur["depth_env"] = m.group(4)
        m = TEMP_BEFORE_RE.match(line)
        if m:
            cur["gpu_temp_before"] = m.group(1)
            cur["arm_start_utc"] = m.group(2)
        m = GATE_RE.match(line)
        if m:
            cur["gates"].append({"temp_c": float(m.group(1)),
                                 "target_c": float(m.group(2)),
                                 "waited_s": int(m.group(3))})
        m = SPEC_RE.search(line)
        if m:
            cur["specs"].append({"spec": m.group(1), "rungs": int(m.group(2)),
                                 "at": m.group(3)})
        m = BEGIN_RE.search(line)
        if m:
            cur["begins"].append({"seed": int(m.group(1)),
                                  "build_us": int(m.group(2)),
                                  "eval_wall_us": int(m.group(3))})
    return arms


def phase_view(arm, index, phase):
    """Trace/gate evidence the given timed phase of one arm entered with."""
    view = {}
    if index < len(arm["begins"]):
        view["build_us"] = arm["begins"][index]["build_us"]
        view["eval_wall_us"] = arm["begins"][index]["eval_wall_us"]
        view["begin_seed_tokens"] = arm["begins"][index]["seed"]
    if index < len(arm["specs"]):
        view["trace_ladder_spec"] = arm["specs"][index]["spec"]
        view["trace_ladder_rungs"] = arm["specs"][index]["rungs"]
        view["trace_ladder_at"] = arm["specs"][index]["at"]
    gate = next((g for i, g in enumerate(arm["gates"])
                 if GATE_PHASE.get(i) == phase), None)
    if gate:
        view["gate_temp_c"] = gate["temp_c"]
        view["gate_waited_s"] = gate["waited_s"]
    return view


def log_arm(arm, group, table, out_runs):
    tag = arm["tag"]
    capture = Path(f"research/capture-e12-{tag}")
    score_path = Path(f"research/score-e12-{tag}.json")
    if not capture.is_dir():
        print(f"{tag}: missing {capture}, skipped", file=sys.stderr)
        return
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics") or {}

    config = dict(host_config())
    config.update({
        "experiment": "qwen38-r1-e16-prefill-ladder-adjudication",
        "arm": tag,
        "ladder_argv": arm["ladder_argv"],
        "ladder_env": arm.get("ladder_env"),
        "local_mode": "local-iterate",
        "decode_tokens_requested": arm.get("decode_tokens_requested"),
        "worker_trace_enabled": arm.get("trace"),
        "run_head_sha": arm.get("head"),
        "worker_sha256": arm.get("worker_sha256"),
        "cli_sha256": arm.get("cli_sha256"),
        "gpu_temp_before_arm_c": arm.get("gpu_temp_before"),
        "gpu_temp_after_arm_c": arm.get("gpu_temp_after"),
        "arm_start_utc": arm.get("arm_start_utc"),
    })
    if arm["specs"]:
        config["ladder_rungs"] = arm["specs"][0]["rungs"]
        config["ladder_rung_positions"] = arm["specs"][0]["at"]

    run = wandb.init(entity=ENTITY, project=PROJECT, name=f"e16-{tag}",
                     group=group, job_type="local-iterate", config=config,
                     tags=["qwen38-r1-e16", "prefill-ladder", tag])
    summary = {f"score/{k}": metrics[k] for k in SCORE_METRICS if k in metrics}
    summary["score/directional_score"] = score.get("score")
    summary["score/passed"] = score.get("passed")

    for index, phase in TRACE_PHASE.items():
        payload_path = capture / PHASE_FILE[phase]
        if not payload_path.exists():
            print(f"{tag}: missing {payload_path}", file=sys.stderr)
            continue
        payload = json.loads(payload_path.read_text())
        view = phase_view(arm, index, phase)
        for key, value in view.items():
            summary[f"{phase}/{key}"] = value
        for key in PHASE_METRICS:
            if key in payload:
                summary[f"{phase}/{key}"] = payload[key]
        provenance = payload.get("head_provenance") or {}
        for key in ("sha256", "bytes", "file_count", "origin", "source"):
            if key in provenance:
                summary[f"{phase}/head_provenance_{key}"] = provenance[key]
        if "build_us" in view:
            reconciled = (view["build_us"] + view["eval_wall_us"]) / 1e6
            summary[f"{phase}/begin_interval_seconds"] = reconciled
            summary[f"{phase}/begin_reconciliation_error_seconds"] = (
                reconciled - payload["seed_prefill_seconds"])
        table.add_data(
            tag, view.get("trace_ladder_spec"), view.get("trace_ladder_rungs"),
            phase, view.get("build_us"), view.get("eval_wall_us"),
            payload.get("seed_prefill_seconds"), payload.get("decode_seconds"),
            view.get("gate_temp_c"), view.get("gate_waited_s"),
            payload.get("all_tokens_matched"),
            payload.get("residual_divergence_count"),
            payload.get("declared_rows_total"),
            payload.get("emitted_token_total"),
            payload.get("accepted_draft_rate"),
            metrics.get("mtp_decode_speedup"),
        )

    run.summary.update(summary)
    out_runs.append({"arm": tag, "id": run.id, "url": run.url,
                     "name": f"e16-{tag}"})
    print(f"{tag}: {run.url}")
    run.finish()


def log_summary(args, arms, table, out_runs):
    floor = json.loads(Path(args.floor).read_text()) if args.floor else None
    config = dict(host_config())
    config.update({
        "experiment": "qwen38-r1-e16-prefill-ladder-adjudication",
        "arms": [a["tag"] for a in arms],
        "arm_ladders": {a["tag"]: a["ladder_argv"] for a in arms},
    })
    if floor:
        config["floor_host"] = floor["host"]
        config["floor_seed_tokens"] = floor["seed_tokens"]

    run = wandb.init(entity=ENTITY, project=PROJECT,
                     name=args.summary_name, group=args.group,
                     job_type="analysis", config=config,
                     tags=["qwen38-r1-e16", "prefill-floor", "analysis"])
    summary = {}
    if floor:
        for key, value in floor.items():
            if isinstance(value, (int, float)):
                summary[f"floor/{key}"] = value
        for key, value in (floor.get("budget") or {}).items():
            if isinstance(value, (int, float)):
                summary[f"budget/{key}"] = value
        for key, value in (floor.get("pipelined_prefill") or {}).items():
            if isinstance(value, (int, float)):
                summary[f"pipelined/{key}"] = value
        columns = sorted({k for c in floor["components"] for k in c})
        comp_table = wandb.Table(columns=columns)
        for component in floor["components"]:
            comp_table.add_data(*[component.get(c) for c in columns])
        run.log({"floor/components": comp_table})
        for component in floor["components"]:
            name = component["component"]
            for key in ("total_seconds", "tflops_achieved",
                        "chained_tflops_achieved", "chain_scaling"):
                if component.get(key) is not None:
                    summary[f"component/{name}/{key}"] = component[key]

    run.log({"arms/timed_phases": table})
    run.summary.update(summary)
    out_runs.append({"arm": "summary", "id": run.id, "url": run.url,
                     "name": args.summary_name})
    print(f"summary: {run.url}")
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--log", required=True, action="append",
                    help="runner log to parse; repeat for multi-job sweeps")
    ap.add_argument("--floor")
    ap.add_argument("--summary-name", default="e16-prefill-floor-and-arms")
    ap.add_argument("--out", default="research/e16-wandb-runs.json")
    args = ap.parse_args()

    arms = [arm for path in args.log for arm in parse_log(path)]
    if not arms:
        sys.exit("no `e12: ##### arm` markers found in the supplied logs")

    table = wandb.Table(columns=TABLE_COLUMNS)
    out_runs = []
    for arm in arms:
        log_arm(arm, args.group, table, out_runs)
    log_summary(args, arms, table, out_runs)

    Path(args.out).write_text(json.dumps(out_runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
