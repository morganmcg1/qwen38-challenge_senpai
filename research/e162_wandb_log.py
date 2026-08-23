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

# The share of the scored leg that prefill occupies, and its complement. The
# advisor set these in E162 feedback 3 after four out of four prefill receipts
# on the board took a decode regression.
PREFILL_WEIGHT = 0.104
DECODE_WEIGHT = 0.896


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
    if prefill_pct is not None and decode_pct is not None:
        priced = prefill_pct * PREFILL_WEIGHT + decode_pct * DECODE_WEIGHT

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
        "prefill_weight": PREFILL_WEIGHT,
        "decode_weight": DECODE_WEIGHT,
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
        "e162_priced_pct": priced,
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
