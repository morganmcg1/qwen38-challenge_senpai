#!/usr/bin/env python3
"""E87: stream the coarse-shortlist decision to W&B.

One run holds every fact another agent needs to reproduce or overturn it:

  build      the arm-G head build report, including the byte delta the price
             list is charged against.
  validate   the rung-0 positive control. The offline exact argmax must equal
             the proposal the runtime returned, the shipped g64 shortlist must
             miss zero times, and a deliberately damaged scorer must miss often.
  screen     the rung-1 `m` tables for arm G and every arm-C cell, with the
             paired discordance against the shipped shortlist per work and per
             domain, the centroid and stage-2 byte columns, and the predicted
             score change on the WORST domain.
  timed      the rung-2 ungated ABABA session. Every leg carries
             cool_gate_passed_real_gate=false and gate_qualified_for_timing
             =false verbatim. It is directional causal evidence inside one
             session, never a score.
  paired     the same session priced round-for-round, with the host-state
             stratum and the achieved bandwidth per arm.
  headline   the same session converted to the published score. The score is
             (raw_beagle + raw_essays) / 2, so each scored prompt gets its own
             price at its own mean draft count.
  liveness   the damaged-index positive control.
  derivation option B's load-time index: whether the table the runtime derives
             from the declared head equals the table the screen priced, and
             whether the derivation lands outside the timed window.
  submit-gate the rung-3 --local-submit legs, one directory each.

usage:
  research/e87_wandb.py --name e87-coarse-shortlist \
      [--build research/e87-build-e87-coarse-g128.json] \
      [--validate research/e87-validate.json] \
      [--screen research/e87-screen.json] \
      [--timed research/e87-timing.json] \
      [--paired research/e87-paired.json] \
      [--derivation research/e87-derivation.json] \
      [--submit-gate research/out/e87s-declared]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e87-coarse-draft-shortlist-traffic"


def cell(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def table(columns, rows):
    t = wandb.Table(columns=list(columns))
    for row in rows:
        t.add_data(*[cell(row.get(c)) for c in columns])
    return t


def log_build(run, path: Path) -> None:
    report = json.loads(path.read_text())
    tag = report["tag"]
    run.log({f"build/{tag}/raw":
             table(["json"], [{"json": json.dumps(report, indent=2)}])})
    run.log({f"build/{tag}/{k}": v for k, v in report.items()
             if isinstance(v, (int, float))})


def log_validate(run, path: Path) -> None:
    report = json.loads(path.read_text())
    run.log({
        "validate/samples": report["samples"],
        "validate/proposal_match": report["proposal_match"],
        "validate/m_shipped_g64": report["m_shipped_g64"]["p"],
        "validate/m_shipped_g64_hi": report["m_shipped_g64"]["hi"],
        "validate/m_damaged_control": report["m_damaged_control"]["p"],
        "validate/raw": table(["json"], [{"json": json.dumps(report, indent=2)}]),
    })


def log_screen(run, path: Path) -> None:
    doc = json.loads(path.read_text())
    cells = doc["cells"]
    columns = ["arm", "n", "misses", "m", "m_lo", "m_hi", "worse_than_shipped",
               "better_than_shipped", "net_miss_vs_shipped", "net_miss_lo",
               "net_miss_hi", "net_miss_worst_domain", "net_miss_worst_work",
               "centroid_bytes", "stage2_bytes", "read_bytes", "removed_bytes",
               "head_pct", "score_gain_pct", "breakeven_m", "predicted_score_pct",
               "predicted_worst_pct", "predicted_worst_domain_pct",
               "by_domain", "by_work"]
    run.log({"screen/cells": table(columns, cells)})
    run.log({"screen/samples": doc["samples"]})

    # One row per (cell, group) so the worst domain is queryable, not buried.
    group_rows = []
    for c in cells:
        for kind in ("by_domain", "by_work"):
            for label, v in c.get(kind, {}).items():
                group_rows.append({
                    "arm": c["arm"], "group_kind": kind[3:], "group": label,
                    "n": v["n"], "m": v["m"], "net_miss_vs_shipped": v["net"],
                    "score_gain_pct": c["score_gain_pct"],
                    "predicted_pct_on_group":
                        c["score_gain_pct"] - 206.6 * v["net"],
                })
    run.log({"screen/by_group": table(sorted({k for r in group_rows for k in r}),
                                      group_rows)})

    best = max(cells, key=lambda c: c["predicted_worst_domain_pct"])
    run.log({
        "screen/best_arm": best["arm"],
        "screen/best_predicted_worst_domain_pct": best["predicted_worst_domain_pct"],
        "screen/best_predicted_score_pct": best["predicted_score_pct"],
    })
    for c in cells:
        if c["arm"] in ("shipped-g64", "armG-g128"):
            run.log({
                f"screen/{c['arm']}/m": c["m"],
                f"screen/{c['arm']}/net_miss_vs_shipped": c["net_miss_vs_shipped"],
                f"screen/{c['arm']}/net_miss_worst_domain": c["net_miss_worst_domain"],
                f"screen/{c['arm']}/predicted_score_pct": c["predicted_score_pct"],
                f"screen/{c['arm']}/predicted_worst_domain_pct":
                    c["predicted_worst_domain_pct"],
            })


def log_timed(run, path: Path) -> None:
    doc = json.loads(path.read_text())
    prefix = doc["prefix"]
    leg_cols = ["tag", "arm", "rep", "started", "candidate_mtp_seconds_per_token",
                "serial_seconds_per_token", "local_ratio", "rounds",
                "rows_per_token", "mean_d", "mean_acc", "accepted_draft_rate",
                "draft_build_us_per_round", "verify_build_us_per_round",
                "round_us_total", "all_tokens_matched",
                "residual_divergence_count", "head_provenance_sha256",
                "head_loaded_bytes", "gpu_temp_entry_c", "gpu_temp_exit_c",
                "cool_gate_passed_real_gate", "gate_qualified_for_timing",
                "base_sha", "worker_sha256"]
    run.log({f"timed/{prefix}/legs": table(leg_cols, doc["legs"])})
    run.log({
        f"timed/{prefix}/session_null_pct": doc["session_null_pct"],
        f"timed/{prefix}/gpu_temp_entry_spread_c": doc["gpu_temp_entry_spread_c"],
    })
    arm_rows = []
    for arm, s in doc["summary"].items():
        arm_rows.append({"arm": arm, **{k: cell(v) for k, v in s.items()}})
        run.log({
            f"timed/{prefix}/{arm}/candidate_seconds_per_token": s["spt_mean"],
            f"timed/{prefix}/{arm}/spt_delta_pct_vs_base": s["spt_delta_pct_vs_base"],
            f"timed/{prefix}/{arm}/local_ratio": s["ratio_mean"],
            f"timed/{prefix}/{arm}/draft_build_us_per_round": s["draft_build_us_per_round"],
            f"timed/{prefix}/{arm}/head_loaded_bytes": s["head_loaded_bytes"],
        })
        if s.get("predicted_pct") is not None:
            run.log({f"timed/{prefix}/{arm}/predicted_pct": s["predicted_pct"]})
    run.log({f"timed/{prefix}/by_arm": table(sorted({k for r in arm_rows for k in r}), arm_rows)})


def log_decision(run, path: Path) -> None:
    doc = json.loads(path.read_text())
    ranked = doc["ranked"]
    columns = ["arm", "m", "net", "worst_domain_net", "worst_work_net",
               "byte_model_gain_pct", "measured_gain_pct",
               "byte_model_worst_pct", "measured_worst_pct", "by_domain"]
    run.log({"decision/ranked": table(columns, ranked)})
    best = ranked[0]
    run.log({
        "decision/samples": doc["samples"],
        "decision/best_arm": best["arm"],
        "decision/best_m": best["m"],
        "decision/best_worst_domain_net": best["worst_domain_net"],
        "decision/best_byte_model_worst_pct": best["byte_model_worst_pct"],
        "decision/best_measured_worst_pct": best["measured_worst_pct"],
    })


def log_paired(run, path: Path) -> None:
    """One rung-2 session priced round-for-round instead of leg-for-leg."""
    doc = json.loads(path.read_text())
    prefix = doc["prefix"]
    run.log({
        f"paired/{prefix}/stratum": table(
            ["tag", "arm", "leg_index", "sandbox", "rounds", "clean_rounds",
             "dirty_rounds", "clean_median_host_us", "dirty_median_host_us",
             "max_host_us", "drafts", "accepted", "mtp_seconds_per_token",
             "gpu_temp_entry_c", "gpu_temp_exit_c", "all_tokens_matched",
             "effective_mean_draft_len", "accepted_draft_rate",
             "head_provenance_sha256", "round1_us", "round2_us"],
            doc["per_leg_host_stratum"]),
        f"paired/{prefix}/host_gate_us": doc["host_gate_us"],
        f"paired/{prefix}/depth_sequence_identical_across_arms":
            doc["depth_sequence_identical_across_arms"],
    })
    bw_rows = [{"arm": a, **b} for a, b in doc.get("achieved_bandwidth", {}).items()]
    if bw_rows:
        run.log({f"paired/{prefix}/achieved_bandwidth":
                 table(sorted({k for r in bw_rows for k in r}), bw_rows)})
        for r in bw_rows:
            run.log({f"paired/{prefix}/{r['arm']}/achieved_bandwidth_gbs":
                     r["achieved_bandwidth_gbs"]})
    rows = []
    for arm, stages in doc["paired"].items():
        for stage, r in stages.items():
            rows.append({"arm": arm, "stage": stage, **r})
            run.log({f"paired/{prefix}/{arm}/{stage}_delta_pct": r["median_pct"],
                     f"paired/{prefix}/{arm}/{stage}_delta_us": r["median_delta_us"]})
    run.log({f"paired/{prefix}/stages": table(sorted({k for r in rows for k in r}),
                                              rows)})


def log_derivation(run, path: Path) -> None:
    """Option B: is the load-time index canonical, deterministic, and untimed."""
    doc = json.loads(path.read_text())
    det, place = doc["determinism"], doc["placement"]
    run.log({
        "derivation/matches_canonical": det["matches_canonical"],
        "derivation/canonical_order_fnv1a64": det["canonical_order_fnv1a64"],
        "derivation/runtime_order_fnv1a64":
            ",".join(det["runtime_order_fnv1a64_values"]),
        "derivation/processes": det["processes"],
        "derivation/dumps_identical_across_processes":
            det["dumps_identical_across_processes"],
        "derivation/build_seconds_mean": place["build_seconds_mean"],
        "derivation/build_seconds_min": place["build_seconds_min"],
        "derivation/build_seconds_max": place["build_seconds_max"],
        "derivation/worst_round1_excess_over_round2_us":
            place["worst_round1_excess_over_round2_us"],
        "derivation/worst_excess_as_fraction_of_build":
            place["worst_excess_as_fraction_of_build"],
        "derivation/build_is_outside_timed_window":
            place["build_is_outside_timed_window"],
        "derivation/builds_once_per_process": place["builds_once_per_process"],
        "derivation/builds": table(
            ["leg", "pid", "leaves", "rows_per_leaf", "probes", "probe_fraction",
             "iterations", "centroid_bits", "order_fnv1a64", "build_seconds",
             "dump_bytes", "dump_sha256"],
            doc["builds"]),
        "derivation/legs": table(
            ["leg", "arm", "probe_fraction", "worker_processes_that_built",
             "one_build_per_process", "rounds", "round1_us", "round2_us",
             "round1_excess_over_round2_us", "mtp_seconds_per_token",
             "all_tokens_matched", "residual_divergence_count",
             "head_provenance_sha256", "effective_mean_draft_len",
             "accepted_draft_rate", "gpu_temp_entry_c", "gpu_temp_exit_c"],
            doc["legs"]),
        "derivation/raw": table(["json"], [{"json": json.dumps(doc, indent=2)}]),
    })


def log_submit_gate(run, legs) -> None:
    """The rung-3 --local-submit legs, read straight from their leg trees."""
    rows = []
    for leg in sorted(Path(p) for p in legs):
        score_path = leg / "score.json"
        if not score_path.is_file():
            raise SystemExit(f"no score.json under {leg}")
        meta = dict(
            line.split("=", 1)
            for line in (leg / "meta.txt").read_text().splitlines()
            if "=" in line)
        m = json.loads(score_path.read_text())["metrics"]
        rows.append({
            "tag": meta.get("tag", leg.name),
            "arm": meta.get("e87_arm"),
            # The probe fraction is a compile-time constant, so the arm label
            # alone no longer identifies the binary. Carry the value the driver
            # read back from the model source, plus the build it names.
            "probe_fraction": meta.get("e87_probe_fraction"),
            "derived_index": meta.get("e87_derived_index"),
            "worker_sha256": meta.get("worker_sha256"),
            "base_sha": meta.get("base_sha"),
            "sandbox": meta.get("sandbox"),
            "mode": m["mode"],
            "decode_tokens": m["decode_tokens"],
            "golden": meta.get("golden"),
            "public_drift_tripwire_passed": m["public_drift_tripwire_passed"],
            "all_tokens_matched": m["all_tokens_matched"],
            "residual_divergence_count": m["residual_divergence_count"],
            "head_provenance_sha256": m["head_provenance_sha256"],
            "mtp_seconds_per_token": m["mtp_seconds_per_token"],
            "serial_seconds_per_token": m["serial_seconds_per_token"],
            "mtp_decode_speedup": m["mtp_decode_speedup"],
            "effective_mean_draft_len": m["effective_mean_draft_len"],
            "accepted_draft_rate": m["accepted_draft_rate"],
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "exit": meta.get("exit"),
        })
    run.log({"submit_gate/legs": table(sorted({k for r in rows for k in r}), rows)})
    for r in rows:
        run.log({
            f"submit_gate/{r['arm']}/mtp_seconds_per_token": r["mtp_seconds_per_token"],
            f"submit_gate/{r['arm']}/all_tokens_matched": int(r["all_tokens_matched"]),
            f"submit_gate/{r['arm']}/public_drift_tripwire_passed":
                int(r["public_drift_tripwire_passed"]),
        })


def log_headline(run, path: Path) -> None:
    """The score-relevant conversion of one balanced timing session.

    The published score is (raw_beagle + raw_essays) / 2, so the two scored
    prompts get their own rows and the mean7-style fixture number stays a
    mechanism diagnostic.
    """
    doc = json.loads(path.read_text())
    prefix = doc["prefix"]

    arm_rows = [{"arm": a, **{k: cell(v) for k, v in s.items()}}
                for a, s in doc["arms"].items()]
    run.log({f"headline/{prefix}/arms":
             table(sorted({k for r in arm_rows for k in r}), arm_rows)})

    score_rows, prompt_rows = [], []
    for arm, s in doc["score_model"].items():
        price = s.get("scored_prompt_price", {})
        score_rows.append({"arm": arm,
                           **{k: cell(v) for k, v in s.items()
                              if k != "scored_prompt_price"},
                           "published_score_gain_pct":
                               price.get("published_score_gain_pct"),
                           "scored_prompt_spread_pp":
                               price.get("scored_prompt_spread_pp"),
                           "spread_within_one_stderr":
                               price.get("spread_within_one_stderr")})
        for prompt, p in price.get("prompts", {}).items():
            prompt_rows.append({"arm": arm, "prompt": prompt, **p})
            if p["in_published_score"]:
                run.log({f"headline/{prefix}/{arm}/{prompt}_raw_p_gain_pct":
                         p["ranked_raw_p_gain_pct"]})
        run.log({
            f"headline/{prefix}/{arm}/measured_leg_total_gain_pct":
                s["measured_leg_total_gain_pct"],
            f"headline/{prefix}/{arm}/leg_total_gain_stderr_pct":
                s["leg_total_gain_stderr_pct"],
            f"headline/{prefix}/{arm}/measured_round_only_gain_pct":
                s["measured_round_only_gain_pct"],
            f"headline/{prefix}/{arm}/ranked_raw_p_gain_pct_at_fixture_depth":
                s["ranked_raw_p_gain_pct"],
            f"headline/{prefix}/{arm}/published_score_gain_pct":
                price.get("published_score_gain_pct"),
            f"headline/{prefix}/{arm}/nonround_seconds_delta":
                s["nonround_seconds_delta"],
            f"headline/{prefix}/{arm}/round_seconds_delta":
                s["round_seconds_delta"],
        })
    run.log({
        f"headline/{prefix}/score_model":
            table(sorted({k for r in score_rows for k in r}), score_rows),
        f"headline/{prefix}/scored_prompts":
            table(sorted({k for r in prompt_rows for k in r}), prompt_rows),
        f"headline/{prefix}/session_null_pct": doc["session_null"]["session_null_pct"],
        f"headline/{prefix}/depth_sequence_identical_across_arms":
            doc["depth_sequence_identical_across_arms"],
        f"headline/{prefix}/raw": table(["json"], [{"json": json.dumps(doc, indent=2)}]),
    })


def log_abba(run, path: Path) -> None:
    """The probe-fraction rider: p = 0.15 against the submitted p = 0.25.

    The two arms need two binaries, so the design counterbalances leg order and
    reads the head-free depth-0 serial leg of each pair as a drift control.
    """
    doc = json.loads(path.read_text())
    scalars = {f"abba/{k}": v for k, v in doc.items()
               if isinstance(v, (int, float, bool))}
    run.log(scalars)
    leg_columns = sorted({k for leg in doc["legs"] for k in leg})
    run.log({
        "abba/legs": table(leg_columns, doc["legs"]),
        "abba/raw": table(["json"], [{"json": json.dumps(doc, indent=2)}]),
    })


def log_coarse_split(run, path: Path) -> None:
    """Section 4: is the affine-2 coarse readout limited by bytes or unpacking?

    Every row is one command buffer that carried exactly one kernel, so the
    buffer interval is that kernel's GPU time and the achieved rate needs no
    fit. `stage_set` names which coarse pass the leg measured.
    """
    doc = json.loads(path.read_text())
    rows = []
    kernels = []
    scalars = {}
    for census, leg in doc["legs"].items():
        tag = Path(census).parent.name
        run.config.update({f"coarse_split/{tag}/{k}": v
                           for k, v in leg["meta"].items()},
                          allow_val_change=True)
        for key in ("phase_us_per_round", "phase_us_per_draft",
                    "phase_dispatches_per_draft"):
            scalars[f"coarse_split/{tag}/{key}"] = leg[key]
        for entry in leg["roster"]:
            kernels.append({"leg": tag, **entry})
        for name, stage in leg["stages"].items():
            source = stage["isolated"] or stage["whole_buffer"]
            rows.append({
                "leg": tag,
                "stage": name,
                "shape": stage["shape"],
                "isolated": stage["isolated"] is not None,
                "us_per_draft": source["measured_us"],
                "moved_mb": source["moved_mb"],
                "achieved_gb_s": source["achieved_gb_s"],
                "fraction_of_dram_ceiling": source["fraction_of_dram_ceiling"],
                "memory_us_at_ceiling": source["memory_us_at_ceiling"],
                "non_memory_us": source["non_memory_us"],
                "ps_per_weight": source["ps_per_weight"],
                "whole_buffer_us_per_draft": stage["whole_buffer"]["measured_us"],
                "whole_buffer_dispatches": stage["whole_buffer"]["dispatches_per_buffer"],
            })
            for key in ("achieved_gb_s", "non_memory_us", "ps_per_weight"):
                scalars[f"coarse_split/{tag}/{name}/{key}"] = source[key]
        combined = leg.get("combined")
        if combined:
            for key, value in combined.items():
                scalars[f"coarse_split/{tag}/combined/{key}"] = value
    run.log(scalars)
    run.log({
        "coarse_split/stages": table(sorted(rows[0]), rows),
        "coarse_split/kernels": table(sorted(kernels[0]), kernels),
        "coarse_split/raw": table(["json"], [{"json": json.dumps(doc, indent=2)}]),
    })


def log_liveness(run, path: Path) -> None:
    """The damaged-index control that proves the cluster path is on the clock."""
    doc = json.loads(path.read_text())
    run.log({"liveness/arms": table(
        ["arm", "head_dir", "effective_mean_draft_len", "accepted_draft_rate",
         "all_tokens_matched", "mtp_seconds_per_token",
         "serial_seconds_per_token"], doc["arms"])})
    run.log({"liveness/passed": doc["passed"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--validate")
    ap.add_argument("--screen")
    ap.add_argument("--decision")
    ap.add_argument("--liveness")
    ap.add_argument("--coarse-split", action="append", default=[],
                    help="one e87-coarse-split JSON per stage set")
    ap.add_argument("--derivation")
    ap.add_argument("--abba",
                    help="report written by research/e87_abba_probe.py")
    ap.add_argument("--timed", action="append", default=[])
    ap.add_argument("--paired", action="append", default=[])
    ap.add_argument("--headline", action="append", default=[],
                    help="report written by research/e87_r2t_headline.py")
    ap.add_argument("--submit-gate", action="append", default=[],
                    help="leg directory written by research/e87_submit_gate.sh")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name, notes=args.notes,
        job_type="e87", tags=["e87", "qwen38-mtp", "harness:local"],
        config={
            "experiment": EXPERIMENT,
            "student": "qwen-thorfinn",
            "pr": 89,
            "harness": "local",
            "host": "Apple M4 Pro 48GB (not the ranked M5)",
            "declared_head_tensor_bytes": 427_738_112,
            "coarse_stage_bytes": 157_337_600,
            "bytes_to_score_pct": 0.0815,
            "miss_to_score_pct": 206.6,
            "official_or_ranked_score": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
        },
    )
    for flag, fn in (("validate", log_validate), ("screen", log_screen),
                     ("decision", log_decision), ("liveness", log_liveness),
                     ("derivation", log_derivation), ("abba", log_abba)):
        path = getattr(args, flag)
        if path:
            fn(run, Path(path))
            print(f"logged {flag} from {path}")
    for flag, fn in (("build", log_build), ("timed", log_timed),
                     ("coarse_split", log_coarse_split),
                     ("paired", log_paired), ("headline", log_headline)):
        for path in getattr(args, flag):
            fn(run, Path(path))
            print(f"logged {flag} from {path}")
    if args.submit_gate:
        log_submit_gate(run, args.submit_gate)
        print(f"logged submit_gate from {args.submit_gate}")
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
