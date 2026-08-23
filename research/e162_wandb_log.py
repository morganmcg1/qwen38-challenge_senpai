#!/usr/bin/env python3
"""E162: publish the gated ABBA session for the prefill affine qmm double buffer.

Everything here is `harness=local`. Not one number in this run is an official
or ranked score, and none may be compared with a receipt.

Four record groups, one run:

  identity     The full experiment identity tuple: base, candidate, host,
               token window, head, reference source, and the per-arm worker
               sha256 and `qmm_t_pipelined_k_loop` witness count. An arm whose
               witness disagrees with its label is void.
  legs         One row per timed leg, in schedule order, with entry and exit
               GPU temperature so thermal drift stays auditable.
  contrast     Base-versus-candidate means and the E162 price
               `prefill_delta * 0.104 + decode_delta * 0.896`, plus the
               within-arm spread that sets the noise floor.
  fidelity     Bit-exactness against base-generated reference rows, and the
               three counters that must not move: effective_mean_draft_len,
               accepted_draft_total, round_count.

Usage:
  python3 research/e162_wandb_log.py --run-name e162-r0-abba-512
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys

import wandb

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "research" / "out" / "e162"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# LOCAL model only. The share of the local scored leg that prefill occupies, and
# its complement, set by the advisor in E162 feedback 3.
LOCAL_PREFILL_WEIGHT = 0.104
LOCAL_DECODE_WEIGHT = 0.896

# RANKED model only, from advisor FINDING 336 in E162 feedback 5. The ranked
# numerator is a pinned prebuilt serial workspace, so a candidate prefill
# improvement carries straight through with no local-to-ranked correction.
RANKED_PREFILL_WEIGHT = 0.1007
# Published gain the campaign has to clear to take the crown, same source.
CROWN_GAP_PCT = 0.5735


def local_priced_pct(prefill_pct: float, decode_pct: float) -> float:
    """harness=local. Blended local leg change. Never use for official pricing."""
    return prefill_pct * LOCAL_PREFILL_WEIGHT + decode_pct * LOCAL_DECODE_WEIGHT


def ranked_published_gain_pct(prefill_pct: float) -> float:
    """harness=ranked. Published-score gain from a candidate prefill change."""
    return -prefill_pct * RANKED_PREFILL_WEIGHT


def read_kv(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def read_legs(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO).decode().strip()


def pct(base, cand):
    """Signed percent change of candidate against base. Negative is faster."""
    if base in (None, 0) or cand is None:
        return None
    return (cand - base) / base * 100.0


def summarise(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    record = {"n": len(clean), "mean": statistics.fmean(clean), "min": min(clean), "max": max(clean)}
    if len(clean) > 1:
        record["stdev"] = statistics.stdev(clean)
        # The spread of the arm against itself is the only noise estimate in
        # this session that costs nothing extra, so it is what the stop rule
        # is checked against.
        record["rel_stdev_pct"] = statistics.stdev(clean) / statistics.fmean(clean) * 100.0
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--session-dir", default=str(OUT / "abba"))
    parser.add_argument("--control-dir", default=str(OUT / "positive-control"))
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session_dir)
    session = read_kv(session_dir / "session.txt")
    legs = read_legs(session_dir / "legs.tsv")
    control = read_kv(pathlib.Path(args.control_dir) / "meta.txt")

    if not legs:
        raise SystemExit("no legs recorded; nothing to publish")

    # THE ARM WITNESS GATE. A leg whose binary does not carry the witness its
    # label claims is not evidence about that arm, so refuse to publish rather
    # than average it in.
    for leg in legs:
        witness = int(leg["witness"])
        if leg["arm"] == "P" and witness != 0:
            raise SystemExit(f"leg {leg['leg']} labelled P but witness={witness}")
        if leg["arm"] == "C" and witness < 1:
            raise SystemExit(f"leg {leg['leg']} labelled C but witness={witness}")

    def column(arm: str, field: str) -> list[float]:
        return [as_float(leg[field]) for leg in legs if leg["arm"] == arm]

    fields = ["spt", "decode_s", "prefill_s", "decode_only_spt"]
    stats = {arm: {f: summarise(column(arm, f)) for f in fields} for arm in ("P", "C")}

    prefill_pct = pct(stats["P"]["prefill_s"].get("mean"), stats["C"]["prefill_s"].get("mean"))
    decode_pct = pct(stats["P"]["decode_only_spt"].get("mean"),
                     stats["C"]["decode_only_spt"].get("mean"))
    spt_pct = pct(stats["P"]["spt"].get("mean"), stats["C"]["spt"].get("mean"))

    priced = None
    ranked_gain = None
    if prefill_pct is not None and decode_pct is not None:
        priced = local_priced_pct(prefill_pct, decode_pct)
    if prefill_pct is not None:
        ranked_gain = ranked_published_gain_pct(prefill_pct)

    analysis = {}
    analysis_path = session_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
    tests = read_kv(OUT / "tests" / "meta.txt")

    config = {
        "experiment": "e162",
        "assignment_id": "e162-prefill-affine-qmm-pipelining",
        "revision_id": "r0",
        "student": "qwen-alphonse",
        "harness": "local",
        "official_score_produced": False,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "candidate_sha": git("rev-parse", "HEAD"),
        "base_sha": session.get("git_head"),
        "worktree_dirty_files": session.get("git_dirty"),
        "mechanism": "double-buffered Ws tile in the affine qmm_t k-loop",
        "arm_a_files": [
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
        ],
        "arm_b_files": [
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
        ],
        "measured_arm": "A (non-NAX quantized.h)",
        # The dispatch reading that sets what this session can and cannot
        # price. quantized.cpp:697 sends prefill to qmm_nax whenever NAX is
        # available, so arm A is not on the ranked M5 path at all.
        "arm_a_reaches_ranked_m5": False,
        "arm_b_reaches_ranked_m5": True,
        "nax_available_on_measurement_host": False,
        "gpu_architecture": "applegpu_g16s (gen 16)",
        "tokens": as_float(session.get("tokens")),
        "depth": as_float(session.get("depth")),
        "schedule": session.get("schedule"),
        "host": session.get("host"),
        "mem_bytes": as_float(session.get("mem_bytes")),
        "head_dir": session.get("head_dir"),
        "golden_sha256": session.get("golden_sha256"),
        "golden_rows": as_float(session.get("golden_rows")),
        "golden_source_arm": "P (base). Reused by both arms, so all_tokens_matched is a cross-arm test.",
        "worker_P_sha256": session.get("worker_P_sha256"),
        "worker_P_witness": session.get("worker_P_witness"),
        "worker_C_sha256": session.get("worker_C_sha256"),
        "worker_C_witness": session.get("worker_C_witness"),
        "metallib_P_sha256": session.get("metallib_P_sha256"),
        "metallib_C_sha256": session.get("metallib_C_sha256"),
        "trusted_cli_sha256": session.get("swift_bin_sha256"),
        "trusted_cli_witness": session.get("swift_bin_witness"),
        "cool_gate": "real 40C gate via ./benchmark.sh --local-cool-gate-only",
        "local_prefill_weight": LOCAL_PREFILL_WEIGHT,
        "local_decode_weight": LOCAL_DECODE_WEIGHT,
        "ranked_prefill_weight": RANKED_PREFILL_WEIGHT,
        "crown_gap_pct": CROWN_GAP_PCT,
        # B.1: the double buffer fits at the shipped BN=64 for every 16-bit
        # instantiation, so the NAX arm needs no retile. float would need 34816
        # bytes against the 32768 byte limit and keeps a single half.
        "b1_ws_bytes_single_half_bfloat16": 9216,
        "b1_ws_bytes_double_half_bfloat16": 18432,
        "b1_ws_bytes_double_half_float": 34816,
        "b1_threadgroup_limit_bytes": 32768,
        "b1_retile_required": False,
        # B.2: no decode-width call can reach the code arm B changes. Reaching
        # it needs transpose and M >= vector_limit, vector_limit is at least 10
        # on gen 17, and trusted Constants.swift caps decode width at 9.
        "b2_decode_sites_reaching_qmm_t_nax": 0,
        "b2_min_vector_limit_gen17": 10,
        "b2_max_decode_rows": 9,
        "b2_census_calls": 2863,
        "b2_census_rows_in_2_to_9": 0,
        # B.3: the transformation moves only the destination pointer, so the
        # accumulation order over src, scales and biases is unchanged.
        "b3_reduction_order_changed": False,
        "b3_single_half_branch_byte_identical": True,
        # The proposal head this session ran is dense, so its 512 row priming
        # forward never enters a quantized matmul. A quantized head would put a
        # 512 row call inside decode and break the B.2 proof.
        "mtp_head_is_quantized": False,
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, name=args.run_name,
                     job_type="local-abba", config=config)

    table = wandb.Table(columns=list(legs[0].keys()))
    for leg in legs:
        table.add_data(*[leg[c] for c in legs[0].keys()])
    run.log({"legs": table})

    summary = {
        "e162_prefill_pct": prefill_pct,
        "e162_decode_pct": decode_pct,
        "e162_parent_spt_pct": spt_pct,
        "e162_local_priced_pct": priced,
        "e162_ranked_published_gain_pct": ranked_gain,
        "e162_crown_gap_pct": CROWN_GAP_PCT,
        "e162_prefill_pct_needed_for_crown": -CROWN_GAP_PCT / RANKED_PREFILL_WEIGHT,
        "leg_count": len(legs),
        "leg_count_P": len([l for l in legs if l["arm"] == "P"]),
        "leg_count_C": len([l for l in legs if l["arm"] == "C"]),
        "all_legs_matched": all(l["matched"] == "true" for l in legs),
        "edl_values": sorted({l["edl"] for l in legs}),
        "accepted_values": sorted({l["accepted"] for l in legs}),
        "round_count_values": sorted({l["rounds"] for l in legs}),
        "counters_unchanged": (
            len({l["edl"] for l in legs}) == 1
            and len({l["accepted"] for l in legs}) == 1
            and len({l["rounds"] for l in legs}) == 1
        ),
        "positive_control_passed_field": control.get("passed"),
        "positive_control_proves_gate_can_fail": control.get("passed") == "false",
        "positive_control_metallib_sha256": control.get("metallib_sha256"),
    }
    for arm, label in (("P", "base"), ("C", "cand")):
        for field in fields:
            for stat, value in stats[arm][field].items():
                summary[f"{label}_{field}_{stat}"] = value

    for field, block in analysis.get("fields", {}).items():
        summary[f"{field}_exact_permutation_p"] = block.get("exact_permutation_p")
        summary[f"{field}_arms_fully_separated"] = block.get("arms_fully_separated")
        summary[f"{field}_base_rel_sd_pct"] = block.get("base_rel_sd_pct")
        summary[f"{field}_cand_rel_sd_pct"] = block.get("cand_rel_sd_pct")

    if tests:
        summary["swift_tests_head"] = tests.get("head")
        summary["swift_tests_worktree_dirty_files"] = as_float(tests.get("dirty"))
        for run_label in ("plain", "runtime"):
            summary[f"swift_{run_label}_tests_started"] = as_float(
                tests.get(f"{run_label}_tests_started"))
            summary[f"swift_{run_label}_issue_lines"] = as_float(
                tests.get(f"{run_label}_issue_lines"))
            summary[f"swift_{run_label}_metallib_aborts"] = as_float(
                tests.get(f"{run_label}_metallib_aborts"))
        # senpai/known-test-failures.md records 9 organizer failures carrying 40
        # issues, plus 1 campaign-added E130 failure carrying 1 issue.
        summary["swift_known_failing_tests"] = 10
        summary["swift_known_issue_lines"] = 41
        summary["swift_new_failures_introduced"] = (
            as_float(tests.get("plain_issue_lines")) == 41.0
            and as_float(tests.get("runtime_issue_lines")) == 41.0
        ) is False

    entry = [as_float(l["entry_c"]) for l in legs]
    exit_ = [as_float(l["exit_c"]) for l in legs]
    if all(v is not None for v in entry):
        summary["entry_temp_spread_c"] = max(entry) - min(entry)
        summary["entry_temp_mean_c"] = statistics.fmean(entry)
    if all(v is not None for v in exit_):
        summary["exit_temp_mean_c"] = statistics.fmean(exit_)

    run.summary.update({k: v for k, v in summary.items() if v is not None})
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)},
                     indent=2, default=str))
    print(f"run_url={run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
