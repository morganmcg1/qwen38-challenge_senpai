#!/usr/bin/env python3
"""E151 R1: log every required artifact and the full identity tuple to W&B.

HARNESS LABELLING. `program.md` invalidates an unlabelled score model. This
experiment carries three harnesses that must never be mixed:

  harness=offline  static source analysis and offline compilation. R0.1-R0.8
                   and the R1 compile gate. Zero GPU seconds, no timing.
  harness=local    anything measured on this M4 Pro. This host reports
                   `is_nax_available() == false`, so the retiled NAX kernel
                   NEVER EXECUTES here. Local exactness and `--local-submit`
                   prove only that the non-NAX path is unbroken and that the
                   candidate is submission ready. They are not, and cannot be,
                   an effect estimate for the arm.
  harness=ranked   published receipt values and the published-median repricing
                   derived from them. Both ranked legs pay the same absolute
                   seed prefill, and the ranked serial numerator comes from a
                   runner-owned prebuilt workspace, so no `psi_serial` term
                   exists or may be subtracted.

The prediction logged here was pre-registered on PR #151 before any timing
existed and is not revised after the fact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
TRACK_ID = "qwen3.8-27b-mtp-v1"

BASE_SHA = "de8ce44c7bc133c3c6c079957240782664afd287"
GROWTH_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"

# harness=ranked. Pre-registered on PR #151 before any timing existed.
PREDICTION = {
    "e151_predicted_ranked_prefill_pct": -3.5,
    "e151_predicted_ranked_prefill_pct_low": -5.5,
    "e151_predicted_ranked_prefill_pct_high": -2.0,
    "e151_minimum_useful_effect_pct": -0.5,
    "e151_null_band_pct": 0.15,
    "e151_predicted_published_median_gain_pct": 0.356,
    "e151_remaining_candidate_leg_gap_pct": 0.4006,
}

# harness=ranked. Board state this candidate is priced against.
RANKED_ANCHORS = {
    "board_bar_id8": "684821ed",
    "board_bar_published_median": 3.71959723,
    "crown_fair_median": 3.70683223,
    "our_best_id8": "0cf1637e",
    "our_best_published_median": 3.68278758,
    "our_fair_median": 3.69204161,
    "uniform_speedup_to_publish_above_bar_pct": 0.9896,
}

BUILD_AFFECTING_PREFIXES = (
    "Package.swift",
    "Sources/",
    "Vendor/",
    "mtp-head.manifest.json",
    "mtp-head/",
)

TRUTHY = {"true": 1.0, "false": 0.0, "ok": 1.0, "OK": 1.0,
          "PASS": 1.0, "FAIL": 0.0, "1": 1.0, "0": 0.0}


def load_json(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def submitted_surface_check() -> tuple[dict, dict]:
    """Prove the ranked host would build the worker this host built.

    Yukon packages only `editablePaths`, so the ranked build is the organizer's
    tree at `upstream/main` with our copies of those paths substituted. A
    build-affecting path that diverges from upstream but is NOT submitted
    exists only here. For E151 the deciding files are the three that choose
    between the JIT string and `mlx.metallib`: if any of them drifted, the
    retile arm proven present in the local JIT string would not be the source
    form the ranked host compiles.
    """
    track = json.loads(pathlib.Path("benchmark.json").read_text())
    if track["trackId"] != TRACK_ID:
        raise SystemExit(f"benchmark.json is track {track['trackId']}, not {TRACK_ID}")
    editable = track["editablePaths"] + track.get("optionalEditablePaths", [])

    diverged = git("diff", "--name-only", "upstream/main", "HEAD").split()
    submitted, local_only = [], []
    for path in diverged:
        if not path.startswith(BUILD_AFFECTING_PREFIXES):
            continue
        covered = any(
            path == e or path.startswith(e.rstrip("/") + "/") for e in editable
        )
        (submitted if covered else local_only).append(path)

    source_form_files = [
        "Vendor/mlx-swift/Package.swift",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/nojit_kernels.cpp",
    ]
    drifted = sorted(set(source_form_files) & set(local_only))

    metrics = {
        "e151_submitted_surface_self_contained": float(not local_only),
        "e151_build_decision_files_match_upstream": float(not drifted),
        "e151_submitted_changed_file_count": float(len(submitted)),
    }
    config = {
        "submitted_changed_files": sorted(submitted),
        "local_only_build_affecting_files": sorted(local_only),
        "source_form_files_checked": source_form_files,
        "source_form_files_drifted": drifted,
    }
    return metrics, config


def r0_metrics(doc: dict) -> tuple[dict, dict]:
    v = doc["verdict"]
    tgp = doc["r0_5_threadgroup"]
    traffic = doc["r0_6_traffic_model"]
    roof = doc["r0_7_roofline"]
    cov = doc["r0_2_coverage"]
    metrics = {
        "e151_r0_pass": float(v["e151_r0_pass"]),
        "e151_source_facts_all_true": float(v["e151_source_facts_all_true"]),
        "e151_coverage_exact": float(v["e151_coverage_exact"]),
        "e151_coverage_failing_controls_caught": float(
            v["e151_coverage_failing_controls_caught"]
        ),
        "e151_k_order_preserved": float(v["e151_k_order_preserved"]),
        "e151_rule145_arm_on_legal": float(v["e151_rule145_arm_on_legal"]),
        "e151_tgp_fits": float(v["e151_tgp_fits"]),
        "e151_r1_touches_decode_qmv_library": float(
            v["e151_r1_touches_decode_qmv_library"]
        ),
        "e151_tgp_bytes_64x64": float(tgp["e151_tgp_bytes_64x64"]),
        "e151_tgp_bytes_128x32": float(tgp["e151_tgp_bytes_128x32"]),
        "e151_tgp_bytes_delta": float(
            tgp["e151_tgp_bytes_128x32"] - tgp["e151_tgp_bytes_64x64"]
        ),
        "e151_predicted_loader_traffic_reduction": float(
            traffic["e151_predicted_loader_traffic_reduction"]
        ),
        "e151_weight_bytes_single_pass": float(traffic["weight_bytes_single_pass"]),
        "e151_weight_bytes_arm_off": float(traffic["weight_bytes_arm_off"]),
        "e151_weight_bytes_arm_on": float(traffic["weight_bytes_arm_on"]),
        "e151_prefill_tflops_achieved": float(roof["e151_prefill_tflops_achieved"]),
        "e151_prefill_flop": float(roof["prefill_flop"]),
        "e151_implied_weight_read_bw_arm_off_GBs": float(
            roof["implied_weight_read_bandwidth_arm_off_GBs"]
        ),
        "e151_implied_weight_read_bw_arm_on_GBs": float(
            roof["implied_weight_read_bandwidth_arm_on_GBs"]
        ),
        "e151_decode_demonstrated_read_bw_lower_bound_GBs": float(
            roof["decode_demonstrated_read_bandwidth_lower_bound_GBs"]
        ),
        "e151_prefill_bw_fraction_of_decode_demonstrated": float(
            roof["prefill_bandwidth_as_fraction_of_decode_demonstrated"]
        ),
        "e151_arithmetic_intensity_arm_off": float(
            roof["arithmetic_intensity_arm_off_flop_per_byte"]
        ),
        "e151_arithmetic_intensity_arm_on": float(
            roof["arithmetic_intensity_arm_on_flop_per_byte"]
        ),
        "e151_arm_off_legal_everywhere": float(v["e151_arm_off_legal_everywhere"]),
        "e151_scored_cell_arm_engages": float(v["e151_scored_cell_arm_engages"]),
        "e151_aot_cells_arm_disarmed": float(v["e151_aot_cells_arm_disarmed"]),
    }
    config = {
        "e151_retile_shape_table": doc["e151_retile_shape_table"],
        "r0_source_facts": doc["source_facts"],
        "r0_host_launch_geometry": doc["host_launch_geometry"],
        "r0_arm_geometry": doc["arm_geometry"],
        "r0_2_coverage": cov,
        "r0_3_k_order": doc["r0_3_k_order"],
        "r0_4_rule145": doc["r0_4_rule145"],
        "r0_5_threadgroup": doc["r0_5_threadgroup"],
        "r0_6_traffic_model": traffic,
        "r0_7_roofline": roof,
        "r0_8_rule153_jit_library_partition": doc[
            "r0_8_rule153_jit_library_partition"
        ],
        "r0_9_loader_legality": doc["r0_9_loader_legality"],
    }
    return metrics, config


def compile_gate_metrics(doc: dict) -> tuple[dict, dict]:
    metrics = {}
    for key, value in doc.items():
        if not key.startswith("e151_"):
            continue
        if isinstance(value, bool):
            metrics[key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics, {"r1_compile_gate": doc}


def gate_chain_metrics(doc: dict) -> tuple[dict, dict]:
    metrics = {}
    for key, value in doc.get("metrics", {}).items():
        metrics[key] = float(value)
    return metrics, {"r1_gate_chain": doc}


def attribution_metrics(doc: dict) -> tuple[dict, dict]:
    """Three 512-token traced legs decide whether the arm moved local output.

    The E121 pin is a historical constant. The decisive comparison is the
    matched same-host arm-off base, plus a second independent arm-on rebuild
    that tests determinism.
    """
    legs = doc["legs"]
    digests = {tag: leg["sha256"] for tag, leg in legs.items()}
    metrics = {
        "e151_arm_is_local_output_neutral": float(
            doc["e151_arm_is_local_output_neutral"]
        ),
        "e151_leg_is_deterministic": float(doc["e151_leg_is_deterministic"]),
        "e151_base_also_misses_pin": float(doc["e151_base_also_misses_pin"]),
        "e151_attribution_complete": float(doc["e151_attribution_complete"]),
        "e151_attribution_distinct_digests": float(len(set(digests.values()))),
        "e151_attribution_leg_count": float(len(legs)),
        "e151_attribution_rows": float(legs["e151x512cand"]["rows"]),
        "e151_attribution_trace_rounds": float(
            legs["e151x512cand"]["trace_rounds"]
        ),
    }
    return metrics, {"r1_arm_attribution": doc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", default="research/e151-r0-safety-case.json")
    ap.add_argument("--compile-gate", default="research/e151-r1-compile-gate.json")
    ap.add_argument("--gates", default="research/e151-r1-gate-chain.json")
    ap.add_argument(
        "--attribution", default="research/e151-arm-attribution.json"
    )
    ap.add_argument("--name", default="e151-r1-nax-seed-prefill-retile-on")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    config: dict = {
        "experiment": "E151",
        "rung": "R1",
        "pr": 151,
        "commit": git("rev-parse", "HEAD"),
        "base_sha": BASE_SHA,
        "assignment_scope_base": BASE_SHA,
        "growth_enforced_base": GROWTH_BASE_SHA,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_clean": git("status", "--porcelain") == "",
        "mechanism": (
            "default the (128, 32) grid-stride retile of the affine NAX "
            "transposed GEMM threadgroup tile ON. Host launch geometry stays "
            "(BM, BN, BK) = (64, 64, 64) with (WM, WN) = (2, 2); only the "
            "compute tile is remapped, so the threadgroup count and the "
            "threadgroup memory allocation are unchanged and the number of "
            "M-tile passes over the weight matrix halves from 8 to 4 at M=512"
        ),
        "arm_flag": "kE147NaxRetileOn",
        "arm_default_before": False,
        "arm_default_after": True,
        "reassociation": "none; BK, SK, TK and the k/kk1/kk loop nest unchanged",
        "changed_candidate_paths": [
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
        ],
        "live_source_form": (
            "Metal JIT string compiled into the runtime worker. "
            "get_qmm_nax_kernel concatenates metal::quantized_nax(), which is "
            "mlx-generated/quantized_nax.cpp; nojit_kernels.cpp is excluded "
            "from the package"
        ),
        "rule153_partition": (
            "get_quantized_kernel (decode QMV) concatenates utils, gemm, "
            "quantized_utils, quantized|fp_quantized. get_qmm_nax_kernel "
            "(prefill NAX GEMM) concatenates utils, gemm_nax, quantized_utils, "
            "quantized_nax|fp_quantized_nax. The changed paths back "
            "metal::quantized_nax() only"
        ),
        "host": "Mac16,11 Apple M4 Pro 48 GiB applegpu_g16s",
        "nax_available_locally": False,
        "local_evidence_scope": (
            "the retiled NAX kernel never executes on this host, so every "
            "local result here is a submission-readiness and non-regression "
            "check, not an effect estimate for the arm"
        ),
        "official_or_ranked_score": False,
        "prediction_pre_registered": True,
        "prediction_pre_registered_at": (
            "https://github.com/morganmcg1/qwen38-challenge_senpai/pull/151"
            "#issuecomment-5385805022"
        ),
        "ranked_value_model": (
            "harness=ranked: both legs pay the same absolute seed prefill, so "
            "raw' = (S - dP)/(C - dP). No psi_serial term exists or may be "
            "subtracted, because the ranked serial numerator comes from a "
            "runner-owned prebuilt baseline workspace"
        ),
        "ranked_anchors": RANKED_ANCHORS,
        "correction_1_lm_head": (
            "Qwen36MTPBlockSession.begin() builds the full-seed lm_head "
            "projection at line 688 but never evaluates it; it is a dead lazy "
            "graph. Only applyLMHead(pendingHidden) at M=1 runs. Scored "
            "prefill FLOP is therefore 24.935 TFLOP, not ~26.2, and achieved "
            "throughput is 47.37 TFLOP/s, not 49.8"
        ),
        "finding_metallib_loader_incompatibility": (
            "the first arm-on build of mlx.metallib FAILED. "
            "QuantizedBlockLoader binds BROWS to the tile's BN, so halving BN "
            "from 64 to 32 halves n_reads and breaks the group_size == 32 "
            "specialisation's (BCOLS_PACKED / n_reads) == n_groups assert. "
            "quantized_nax.metal instantiates group sizes 128, 64 and 32 ahead "
            "of time, so all six group-32 cells stopped the build. E147 "
            "shipped the arm off, so the retiled path was never instantiated "
            "and the defect was unreachable. R1 disarms the retile where the "
            "loader cannot support it; the scored group-64 4-bit cell keeps "
            "its 28027-byte AIR digest, so the guard is free on the ranked path"
        ),
        "correction_2_shift_dst": (
            "quantized_nax.h contains no shift_dst, no Ws_tile and no double "
            "buffer. shift_dst exists only on the fp path, at "
            "fp_quantized_nax.h:248 with call sites :366, :380, :1034, :1049. "
            "R2 must ADD both the loader helper and the double buffer to the "
            "affine path"
        ),
        "finding_row_digest_pin_is_stale": (
            "the 512-token exactness gate missed the E121 pin "
            "719d82b8 over 1025 rows. research/e151_arm_attribution.sh ran "
            "three traced legs on this host: the arm-off base de8ce44c, the "
            "arm-on candidate 58b415d4 and a second independent arm-on "
            "rebuild at 9dda0cc1. All three emit d070b397 over 1024 rows and "
            "82 trace rounds, from three different worker binaries. The arm "
            "is therefore local output neutral and the leg is deterministic. "
            "The pin was recorded on E89 base f18400c4 over 78 trace rounds, "
            "so the drift belongs to the intervening scheduler commits, not "
            "to R1. Re-pinning is a campaign decision: the constant is shared "
            "with E101, E110, E116, E121 and E129"
        ),
        "exactness_verdict_basis": (
            "matched same-host arm-off base control, not the historical pin"
        ),
    }

    metrics: dict = dict(PREDICTION)
    metrics["e151_prediction_pre_registered"] = 1.0

    surface_m, surface_c = submitted_surface_check()
    metrics.update(surface_m)
    config.update(surface_c)

    r0 = load_json(args.r0)
    if r0 is None:
        raise SystemExit(f"missing required artifact {args.r0}")
    m, c = r0_metrics(r0)
    metrics.update(m)
    config.update(c)

    cg = load_json(args.compile_gate)
    if cg is None:
        raise SystemExit(f"missing required artifact {args.compile_gate}")
    m, c = compile_gate_metrics(cg)
    metrics.update(m)
    config.update(c)

    attrib = load_json(args.attribution)
    if attrib is None:
        raise SystemExit(f"missing required artifact {args.attribution}")
    m, c = attribution_metrics(attrib)
    metrics.update(m)
    config.update(c)

    gates = load_json(args.gates)
    if gates is None:
        raise SystemExit(f"missing required artifact {args.gates}")
    m, c = gate_chain_metrics(gates)
    metrics.update(m)
    config.update(c)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        config=config,
        mode="offline" if args.offline else "online",
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    print(f"e151 wandb run {run.id} {run.url}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    run.finish()


if __name__ == "__main__":
    main()
