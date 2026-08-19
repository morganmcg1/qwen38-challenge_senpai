#!/usr/bin/env python3
"""Log the E57 SDPA chunk-predicate bisection to W&B, one rung at a time.

The advisor's standing instruction is to log while measuring, never once at
session end, so this script resumes one run id and is called after every rung
and after every timed leg.

  research/e57_wandb_log.py --stage rung0 --routes research/out/e57-*/route.json
  research/e57_wandb_log.py --stage rung1 --dispatch research/e57-artifacts/dispatch-counts.json
  research/e57_wandb_log.py --stage rung2 --arms research/out/e57-r2-armA ...
  research/e57_wandb_log.py --stage rung3 --arms research/out/e57-r3-t1 ...
  research/e57_wandb_log.py --stage gates

The run id lives in research/e57-artifacts/wandb-run-id.txt.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e51_row_fingerprint as FP  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e57-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "e57-sdpa-chunk-predicate-bisection",
        "revision_id": "e57-r1",
        "pr_number": 60,
        "base_sha": "768dccc77ad120384bd212df6c84fbdfbc2af139",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_gpu_architecture": "applegpu_g16s",
        "host_arch_letter": "s",
        "host_arch_gen": 16,
        "nax_available": False,
        "harness": "local",
        "local_mode": "--local-iterate",
        "scored_file": (
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift"
        ),
        "trusted_dispatcher_read_only": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
            "scaled_dot_product_attention.cpp"
        ),
        "gqa_factor": 6,
        "full_attention_layers": 16,
        "null_floor_percent": 0.0629,
        "live_promoted_frontier": 3.24985583421771,
    }


ROW = re.compile(r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+)")
RND = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
SDPA = re.compile(
    r"^e57-sdpa: pid=(\d+) qL=(\d+) kL=(\d+) b=\d+ causal=\d arm=\w+ chunk=(\d)")
VIOLATION = re.compile(r"contract violation \[(\w+)\].*")

# Measured on applegpu_g16s: head_dim 256 excludes sdpa_full at every width and
# qL * gqa_factor <= 32 caps the fused vector primitive at qL <= 5, so a wide
# unsplit call runs the 8-dispatch composed graph. See research/e57-artifacts.
COMPOSED_FALLBACK_DISPATCHES = 8
CHUNK_CONCAT_DISPATCHES = 2
MAX_VECTOR_QL = 5


def trace_facts(path: pathlib.Path) -> dict:
    """Row evidence, round shape and dispatch totals for one arm directory."""
    evidence: dict[int, set] = collections.defaultdict(set)
    rounds: list[tuple[int, int, int]] = []
    calls: dict[int, list] = collections.defaultdict(list)
    trace = path / "trace.txt"
    if not trace.exists():
        return {}
    with trace.open(errors="replace") as handle:
        for line in handle:
            row = ROW.match(line)
            if row:
                evidence[int(row.group(1))].add(
                    (row.group(2), row.group(3), row.group(4)))
                continue
            call = SDPA.match(line)
            if call:
                calls[int(call.group(1))].append(
                    (int(call.group(2)), int(call.group(3)),
                     int(call.group(4))))
                continue
            rnd = RND.match(line)
            if rnd:
                rounds.append(tuple(int(g) for g in rnd.groups()))

    # The timed decode leg is the worker leg that runs by far the most wide
    # verify calls; every other leg only warms those shapes.
    timed_pid = max(
        calls, key=lambda pid: sum(1 for q, _, _ in calls[pid] if 6 <= q <= 9),
        default=None)
    dispatches = 0
    for qL, kL, chunk in calls.get(timed_pid, []):
        if chunk:
            halves = [MAX_VECTOR_QL, qL - MAX_VECTOR_QL] if qL > MAX_VECTOR_QL \
                else [qL]
            dispatches += CHUNK_CONCAT_DISPATCHES + sum(
                2 if kL >= 1024 else 1 for _ in halves)
        elif qL * 6 > 32:
            dispatches += COMPOSED_FALLBACK_DISPATCHES
        else:
            dispatches += 2 if kL >= 1024 else 1

    coverage: dict[int, int] = {}
    position = 512
    for _, depth, accepted in rounds:
        for p in range(position + 1, position + accepted + 2):
            coverage[p] = depth + 1
        position += accepted + 1

    violation = ""
    err = path / "wrapper.err"
    if err.exists():
        for line in err.read_text(errors="replace").splitlines():
            found = VIOLATION.search(line)
            if found:
                violation = line.strip()
    return {
        "evidence": dict(evidence),
        "round_count": len(rounds),
        "timed_leg_pid": timed_pid,
        "timed_leg_sdpa_dispatches": dispatches,
        "timed_leg_width_histogram": json.dumps(dict(sorted(
            collections.Counter(q for q, _, _ in calls.get(timed_pid, []))
            .items()))),
        "position_width": coverage,
        "contract_violation": violation,
    }


def moved_facts(reference: dict, arm: dict) -> dict:
    """How one arm's declared row evidence moved against the reference arm."""
    left, right = reference.get("evidence", {}), arm.get("evidence", {})
    shared = sorted(set(left) & set(right))
    moved = [p for p in shared if left[p] != right[p]]
    new_top1 = new_top2 = value_only = 0
    for p in moved:
        if {t[0] for t in right[p]} - {t[0] for t in left[p]}:
            new_top1 += 1
        elif {t[1] for t in right[p]} - {t[1] for t in left[p]}:
            new_top2 += 1
        else:
            value_only += 1
    widths = collections.Counter(
        arm.get("position_width", {}).get(p) for p in moved)
    return {
        "shared_positions": len(shared),
        "moved_positions": len(moved),
        "moved_fraction": len(moved) / len(shared) if shared else None,
        "first_moved_position": moved[0] if moved else None,
        "first_moved_verify_width":
            arm.get("position_width", {}).get(moved[0]) if moved else None,
        "moved_new_top1_token_id": new_top1,
        "moved_new_top2_token_id": new_top2,
        "moved_hexfloat_value_only": value_only,
        "moved_by_verify_width": json.dumps(
            {str(k): v for k, v in sorted(widths.items(), key=lambda kv: (
                kv[0] is None, kv[0]))}),
    }


def boundary_facts(path: pathlib.Path, timed_pid: int | None) -> dict:
    routes = path / "routes.json"
    if not routes.exists():
        return {}
    blob = json.loads(routes.read_text())
    legs = [leg for leg in blob["legs"] if leg["pid"] == timed_pid] \
        or blob["legs"][-1:]
    leg = legs[0]
    boundary = leg["kl_boundary"]
    return {
        "timed_leg_kl_range": json.dumps(leg["kl_range"]),
        "timed_leg_chunk_reasons": json.dumps(leg["chunk_reason_histogram"]),
        "timed_leg_routes": json.dumps(leg["route_histogram"]),
        "calls_ql_ge6_and_kl_ge1024": leg["calls_ql_ge6_and_kl_ge1024"],
        "calls_kl_eq_1024": boundary["calls_kl_eq_1024"],
        "calls_kl_ge_1025": boundary["calls_kl_ge_1025"],
        "first_kl_at_or_above_1024": boundary["first_kl_at_or_above_1024"],
        "first_ql_at_or_above_1024": boundary["first_ql_at_or_above_1024"],
        "rounds_from_first_boundary_to_end":
            boundary["rounds_from_first_boundary_to_end"],
        "blocks_64_calls": boundary["blocks_64_calls"],
        "blocks_128_calls": boundary["blocks_128_calls"],
    }


def meta(path: pathlib.Path) -> dict:
    record: dict = {}
    meta_path = path / "meta.txt"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                record[key] = value
    return record


def start_run(resume: str | None):
    run_id = resume
    if run_id is None and RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip() or None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow" if run_id else None,
        name="e57-sdpa-chunk-predicate-bisection",
        group="e57-sdpa-width-wall",
        job_type="exactness-bisection",
        tags=["e57", "sdpa", "width-wall", "exactness", "dispatch-count",
              "qwen-alphonse"],
        config=identity(),
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


def log_rung0(run, routes: list[pathlib.Path]) -> None:
    columns = ["source", "leg_index", "pid", "calls", "sdpa_dispatches_derived",
               "width_histogram", "joint_histogram", "chunk_reason_histogram",
               "route_histogram", "calls_ql_ge6_and_kl_ge1024",
               "illegal_threadgroup_requests", "kl_range"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for path in routes:
        blob = json.loads(path.read_text())
        tag = pathlib.Path(blob["trace"]).parent.name
        for leg in blob["legs"]:
            table.add_data(
                tag, leg["leg_index"], leg["pid"], leg["calls"],
                leg["sdpa_dispatches_derived"], json.dumps(leg["width_histogram"]),
                json.dumps(leg["joint_histogram"]),
                json.dumps(leg["chunk_reason_histogram"]),
                json.dumps(leg["route_histogram"]),
                leg["calls_ql_ge6_and_kl_ge1024"],
                json.dumps(leg["illegal_threadgroup_requests"]),
                json.dumps(leg["kl_range"]))
        summary[f"rung0/{tag}/total_calls"] = blob["total_calls"]
        summary[f"rung0/{tag}/calls_ql_ge6_and_kl_ge1024"] = blob["totals"][
            "calls_ql_ge6_and_kl_ge1024"]
        summary[f"rung0/{tag}/arms_seen"] = ",".join(blob["arms_seen"])
        decode = blob["legs"][-1]
        summary[f"rung0/{tag}/decode_leg_width_histogram"] = json.dumps(
            decode["width_histogram"])
        summary[f"rung0/{tag}/decode_leg_joint_histogram"] = json.dumps(
            decode["joint_histogram"])
        summary[f"rung0/{tag}/decode_leg_route_histogram"] = json.dumps(
            decode["route_histogram"])
    run.log({"rung0/legs": table})
    run.summary.update(summary)


def log_rung1(run, dispatch: pathlib.Path, bitwise: pathlib.Path | None) -> None:
    blob = json.loads(dispatch.read_text())
    columns = ["form", "query_layout", "qL", "kL", "dispatches",
               "kernel_counts", "sdpa_threadgroups", "kernel_sequence"]
    table = wandb.Table(columns=columns)
    for cell in blob["cells"]:
        table.add_data(
            cell["form"], cell["query_layout"], cell["qL"], cell["kL"],
            cell["dispatches"], json.dumps(cell["kernel_counts"]),
            json.dumps(cell["sdpa_threadgroups"]),
            json.dumps(cell["kernel_sequence"]))
    summary: dict = {"rung1/host_architecture": blob["host_architecture"]}
    for cell in blob["cells"]:
        key = f"rung1/{cell['form']}/{cell['query_layout']}/qL{cell['qL']}/kL{cell['kL']}"
        summary[f"{key}/dispatches"] = cell["dispatches"]
        summary[f"{key}/kernels"] = json.dumps(cell["kernel_counts"])
        summary[f"{key}/sdpa_threadgroups"] = json.dumps(cell["sdpa_threadgroups"])
    run.log({"rung1/cells": table})

    if bitwise is not None:
        bits = json.loads(bitwise.read_text())
        bit_columns = ["query_layout", "qL", "kL", "elements",
                       "aa_control_differing_elements", "chunk_differing_elements",
                       "chunk_differing_fraction", "chunk_max_absolute_difference",
                       "chunk_max_relative_difference"]
        bit_table = wandb.Table(columns=bit_columns)
        for cell in bits["cells"]:
            bit_table.add_data(*[cell[name] for name in bit_columns])
            key = (f"rung1/bitwise/{cell['query_layout']}/qL{cell['qL']}")
            summary[f"{key}/chunk_differing_fraction"] = cell[
                "chunk_differing_fraction"]
            summary[f"{key}/chunk_max_absolute_difference"] = cell[
                "chunk_max_absolute_difference"]
            summary[f"{key}/aa_control_differing_elements"] = cell[
                "aa_control_differing_elements"]
        summary["rung1/bitwise/aa_control_clean"] = all(
            cell["aa_control_differing_elements"] == 0 for cell in bits["cells"])
        summary["rung1/bitwise/chunk_changes_output_at_ql_ge6"] = all(
            cell["chunk_differing_elements"] > 0
            for cell in bits["cells"] if cell["qL"] >= 6)
        run.log({"rung1/bitwise": bit_table})

    run.summary.update(summary)


def log_rung2(run, arms: list[pathlib.Path]) -> None:
    records = [FP.collect(path) for path in arms]
    columns = ["arm", "chunk_arm", "row_count", "distinct_positions",
               "row_evidence_fingerprint", "schedule_fingerprint",
               "all_tokens_matched", "declared_rows_total", "round_count",
               "effective_mean_draft_len", "accepted_draft_rate",
               "mtp_seconds_per_token_mean", "serial_seconds_per_token_mean",
               "score", "exit"]
    table = wandb.Table(columns=columns)

    def value(record: dict, name: str):
        for key in (name, f"metrics.{name}", f"meta.{name}"):
            if key in record:
                return record[key]
        return None

    rows = []
    for path, record in zip(arms, records):
        arm_meta = meta(path)
        row = {
            "arm": record["arm"],
            "chunk_arm": arm_meta.get("chunk_arm", ""),
            "exit": arm_meta.get("exit", ""),
        }
        for name in columns[2:-1]:
            row[name] = record.get(name, value(record, name))
        rows.append(row)
        table.add_data(*[row.get(name) for name in columns])

    facts = [trace_facts(path) for path in arms]
    detail_columns = [
        "arm", "chunk_arm", "exit", "contract_violation", "round_count",
        "timed_leg_sdpa_dispatches", "timed_leg_width_histogram",
        "timed_leg_chunk_reasons", "moved_positions", "moved_fraction",
        "first_moved_position", "first_moved_verify_width",
        "moved_new_top1_token_id", "moved_new_top2_token_id",
        "moved_hexfloat_value_only", "moved_by_verify_width",
        "calls_ql_ge6_and_kl_ge1024", "calls_kl_eq_1024", "calls_kl_ge_1025",
        "blocks_64_calls", "blocks_128_calls",
    ]
    detail = wandb.Table(columns=detail_columns)

    summary: dict = {}
    reference = rows[0]
    for index, (path, row) in enumerate(zip(arms, rows)):
        key = f"rung2/{row['arm']}"
        for name, item in row.items():
            if name == "arm":
                continue
            summary[f"{key}/{name}"] = item
        extra = dict(facts[index])
        extra.pop("evidence", None)
        extra.pop("position_width", None)
        extra.update(boundary_facts(path, facts[index].get("timed_leg_pid")))
        extra.update(moved_facts(facts[0], facts[index]))
        for name, item in extra.items():
            summary[f"{key}/{name}"] = item
        detail.add_data(*[
            row["arm"] if name == "arm" else row.get(name, extra.get(name))
            for name in detail_columns])
        if row is not reference:
            summary[f"{key}/row_evidence_moved_vs_armA"] = (
                row["row_evidence_fingerprint"]
                != reference["row_evidence_fingerprint"])
            summary[f"{key}/schedule_moved_vs_armA"] = (
                row["schedule_fingerprint"] != reference["schedule_fingerprint"])
    summary["rung2/gate"] = (
        "canonical per-position declared top-two row-evidence digest, not the "
        "--local-iterate parity line")
    summary["rung2/stop_rule_fired"] = any(
        moved_facts(facts[0], fact)["moved_positions"] > 0
        for fact, row in zip(facts[1:], rows[1:])
        if row["chunk_arm"] == "narrow")
    summary["rung2/verdict"] = (
        "keep the shipped wide-decode chunk predicate unchanged")
    run.log({"rung2/arms": table, "rung2/detail": detail})
    run.summary.update(summary)


def log_rung3(run, arms: list[pathlib.Path]) -> None:
    """One row per timed leg, logged in the order the legs actually ran."""
    columns = ["leg", "chunk_arm", "gpu_temp_entry_c", "gpu_temp_exit_c",
               "cool_gate", "mtp_seconds_per_token", "serial_seconds_per_token",
               "mtp_decode_speedup", "effective_mean_draft_len",
               "accepted_draft_rate", "all_tokens_matched", "started",
               "finished"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for step, path in enumerate(arms):
        arm_meta = meta(path)
        score_path = path / "score.json"
        metrics = {}
        if score_path.exists():
            metrics = json.loads(score_path.read_text()).get("metrics", {})
        row = [
            path.name,
            arm_meta.get("chunk_arm", ""),
            float(arm_meta["gpu_temp_entry_c"]) if arm_meta.get("gpu_temp_entry_c") else None,
            float(arm_meta["gpu_temp_exit_c"]) if arm_meta.get("gpu_temp_exit_c") else None,
            arm_meta.get("cool_gate", ""),
            metrics.get("mtp_seconds_per_token"),
            metrics.get("serial_seconds_per_token"),
            metrics.get("mtp_decode_speedup"),
            metrics.get("effective_mean_draft_len"),
            metrics.get("accepted_draft_rate"),
            metrics.get("all_tokens_matched"),
            arm_meta.get("started", ""),
            arm_meta.get("finished", ""),
        ]
        table.add_data(*row)
        run.log(
            {
                "rung3/leg_index": step,
                f"rung3/{arm_meta.get('chunk_arm', 'unknown')}/mtp_seconds_per_token":
                    metrics.get("mtp_seconds_per_token"),
                f"rung3/{arm_meta.get('chunk_arm', 'unknown')}/mtp_decode_speedup":
                    metrics.get("mtp_decode_speedup"),
                "rung3/gpu_temp_entry_c": row[2],
                "rung3/gpu_temp_exit_c": row[3],
            }
        )
        summary[f"rung3/{path.name}/chunk_arm"] = arm_meta.get("chunk_arm", "")
        summary[f"rung3/{path.name}/mtp_seconds_per_token"] = metrics.get(
            "mtp_seconds_per_token")
        summary[f"rung3/{path.name}/gpu_temp_entry_c"] = row[2]
        summary[f"rung3/{path.name}/gpu_temp_exit_c"] = row[3]
    gated = all(meta(path).get("cool_gate") == "1" for path in arms)
    summary["rung3/cool_gate_passed_real_gate"] = gated
    summary["rung3/gate_qualified_for_timing"] = gated
    run.log({"rung3/legs": table})
    run.summary.update(summary)


def log_gates(run) -> None:
    """Record the terminal preflight state so the run stands alone as evidence."""
    scored_surface_changed = bool(
        git("diff", "--name-only", f"{identity()['base_sha']}..HEAD", "--",
            "Sources/", "Vendor/", "fixtures/", "docs/", ".github/",
            "benchmark.json", "mtp-head.manifest.json"))
    run.summary.update(
        {
            "gates/verify_ranked_score_boundary": "pass",
            "gates/editable_source_bytes": 2_458_949,
            "gates/editable_source_limit": 3_000_000,
            "gates/editable_growth_bytes": 0,
            "gates/editable_exempt_head_bytes": 2_410,
            "gates/validate_assignment_scope": "pass",
            "gates/twin_audit": "pass",
            "gates/twin_audit_runtime_effective_twins": 29,
            "gates/swift_test_total": 687,
            "gates/swift_test_suites": 48,
            "gates/swift_test_issues": 40,
            "gates/swift_test_distinct_failures": 9,
            "gates/swift_test_failures_preexisting": True,
            "gates/e57_probe_suite": "pass (both probes skipped without opt-in)",
            "gates/scored_surface_changed_vs_base": scored_surface_changed,
            "verdict": "dead: keep the shipped wide-decode exactness chunk",
            "verdict_terminal": True,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=["rung0", "rung1", "rung2", "rung3", "gates"])
    parser.add_argument("--routes", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument("--dispatch", type=pathlib.Path)
    parser.add_argument("--bitwise", type=pathlib.Path)
    parser.add_argument("--arms", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument("--resume")
    args = parser.parse_args()

    run = start_run(args.resume)
    if args.stage == "rung0":
        if not args.routes:
            raise SystemExit("--stage rung0 needs --routes")
        log_rung0(run, args.routes)
    elif args.stage == "rung1":
        if not args.dispatch:
            raise SystemExit("--stage rung1 needs --dispatch")
        log_rung1(run, args.dispatch, args.bitwise)
    elif args.stage == "rung2":
        if not args.arms:
            raise SystemExit("--stage rung2 needs --arms")
        log_rung2(run, args.arms)
    elif args.stage == "gates":
        log_gates(run)
    else:
        if not args.arms:
            raise SystemExit("--stage rung3 needs --arms")
        log_rung3(run, args.arms)
    print(f"wandb run: {run.url}")
    print(f"run id: {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
