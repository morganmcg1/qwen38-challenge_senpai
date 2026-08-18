#!/usr/bin/env python3
"""E21: publish the shipped-default depth histogram and the declination analysis to W&B.

One run per traced probe prompt, so every counter in the report is reachable from
a run URL, plus one analysis run carrying the pooled deliverable, the separability
table, the oracle headroom and the uniform-threshold sweep.

`research/e17_wandb.py` cannot be reused: E17 logged timed arm pairs, whereas E21
produced counter-only traced probes and an offline replay. Every key here is
prefixed `e21_` so nothing collides with an earlier experiment's series.

usage:
  research/e21_wandb.py --report /tmp/e21_fit8.json
  research/e21_wandb.py --report /tmp/e21_fit8.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
GROUP = "qwen38-r1-e21-depth-preserving-row-declination"
BASE_SHA = "c0f7e370921a14f348fa1872f2176b1b43028752"
ARM = "S18I"

# Counter-only probes: this host idles above the compile-time 40C gate, so the
# cool gate was bypassed and no leg here is a gate-qualified timing claim. The
# leg-anchored cost is a within-session ratio, where thermal state is common to
# both legs.
GATE_QUALIFIED = False


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config() -> dict:
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_cores": sh("sysctl", "-n", "hw.ncpu"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_base_sha": BASE_SHA,
        "git_dirty_files": len(sh("git", "status", "--porcelain").splitlines()),
    }


def read_meta(run: Path) -> dict:
    out = {}
    for line in (run / "meta.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def thermals(meta: dict) -> dict:
    out = {}
    for when in ("before", "after"):
        for field in (meta.get(f"thermal_{when}") or "").split():
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            try:
                out[f"e21_thermal_{when}_{key}"] = float(value.rstrip("CW"))
            except ValueError:
                pass
    return out


def leg(run: Path, name: str) -> dict:
    return json.loads((run / "reports" / name).read_text())


def histogram_metrics(prefix: str, summary: dict) -> dict:
    """Flatten the two histograms so each bucket is its own queryable key."""
    out = {}
    total = sum(summary["depth_histogram"].values()) or 1
    for depth, n in summary["depth_histogram"].items():
        out[f"{prefix}_depth_{depth}_rounds"] = n
        out[f"{prefix}_depth_{depth}_share"] = n / total
    for width, n in summary["verify_width_histogram"].items():
        out[f"{prefix}_verify_width_{width}_rounds"] = n
        out[f"{prefix}_verify_width_{width}_share"] = n / total
    return out


def finite(value):
    """W&B summaries must not carry NaN; an undefined rate is absent, not zero.

    A predicate that never fires has no precision, which is not the same fact as
    a precision of zero, so it is dropped rather than coerced.
    """
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def best_firing(report: dict) -> dict:
    """Best-ranked rule that actually declines something.

    `ranked[0]` is routinely the degenerate no-op rule, which is the headline
    finding but carries no precision or recall worth plotting. Scanning the
    full candidate list keeps this independent of the fitter's `--top`.
    """
    firing = [r for r in report["all_candidates"] if r.get("fired")]
    if not firing:
        return {}
    return max(firing, key=lambda r: (r["worst_prompt_gain_pct"], r["gain_pct"]))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def thermal_robustness(rows: list[dict]) -> dict:
    """Does the leg-anchored marginal track the host's thermal state?

    The probes were taken with the cool gate bypassed, so entry temperature
    varies by ~20C across the pool. If `h` tracked that, the whole cost model
    would be an artefact of run order rather than a property of the prompt.
    """
    temps = [r["entry_c"] for r in rows]
    d0s = [r["d0_us"] for r in rows]
    hs = [r["h"] for r in rows]
    return {
        "e21_thermal_entry_c_min": min(temps),
        "e21_thermal_entry_c_max": max(temps),
        "e21_thermal_entry_c_span": max(temps) - min(temps),
        "e21_thermal_exit_c_max": max(r["exit_c"] for r in rows),
        "e21_thermal_d0_round_us_min": min(d0s),
        "e21_thermal_d0_round_us_max": max(d0s),
        "e21_thermal_d0_round_us_span_pct": 100.0 * (max(d0s) - min(d0s)) / min(d0s),
        "e21_thermal_pearson_entry_c_vs_h": pearson(temps, hs),
        "e21_thermal_pearson_entry_c_vs_d0_round_us": pearson(temps, d0s),
    }


def probe_payload(prompt: str, run: Path, report: dict) -> tuple[dict, dict, list]:
    key = run.name
    meta = read_meta(run)
    serial, mtp = leg(run, "03-mtp-timed.json"), leg(run, "04-mtp-timed.json")
    correctness = leg(run, "01-correctness.json")
    summary = report["per_prompt_summary"][key]
    anchor = report["leg_anchored_cost_model"][key]
    regression = report["measured_cost_model_per_prompt"][key]
    auc = report["separability_auc_per_prompt"][key]
    nonparam = report["nonparametric_oracle_per_prompt"][key]

    config = dict(host_config())
    config.update({
        "experiment": GROUP,
        "assignment_revision": "r1",
        "pr_number": 25,
        "prompt": prompt,
        "arm": ARM,
        "arm_role": "shipped-default-verbatim",
        "decode_tokens": mtp["decode_token_count"],
        "seed_tokens": mtp["seed_token_count"],
        "local_mode": "local-iterate",
        "traced": True,
        "mtp_depth": mtp["mtp_depth"],
        "max_draft_depth_bound": mtp["max_draft_depth_bound"],
        "requested_draft_depth": mtp["requested_draft_depth"],
        "serial_control_depth": serial["mtp_depth"],
        "shipped_head_step_cost_ratio": report["replay_fidelity"]["shipped_h"],
        "worker_sha256": meta.get("worker_sha256"),
        "source_sha256": meta.get("source_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "head_sha256": mtp["head_provenance"]["sha256"],
        "head_origin": mtp["head_provenance"]["origin"],
        "uses_pinned_mtp_head": mtp["uses_pinned_mtp_head"],
        "uses_native_mtp_head": mtp["uses_native_mtp_head"],
        "golden": meta.get("golden"),
        "worktree_dirty_at_run": int(meta.get("dirty", "0")),
        "mlx_qwen_env": meta.get("mlx_qwen_env", ""),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        # carried verbatim: these probes are counters, not timing claims
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate") == "true",
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing") == "true",
        "cool_gate_bypass_reason": meta.get("cool_gate_bypass_reason", ""),
    })

    metrics = {
        # shipped-default schedule shape: the advisor's required deliverable
        "e21_round_count": summary["round_count"],
        "e21_proposed_row_count": summary["proposed_drafts"],
        "e21_accepted_drafts": summary["accepted_drafts"],
        "e21_accepted_draft_rate": mtp["accepted_draft_rate"],
        "e21_effective_mean_draft_len": summary["effective_mean_draft_len"],
        "e21_effective_max_draft_len": summary["effective_max_draft_len"],
        "e21_zero_accept_rounds": summary["zero_accept_rounds"],
        "e21_verify_block_replayed_round_count":
            mtp["verify_block_replayed_round_count"],
        "e21_non_drafting_round_count": mtp["non_drafting_round_count"],
        # leg-anchored marginal draft cost: the retraction-grade estimator
        "e21_h_measured": anchor["h_measured"],
        "e21_d0_round_us_measured": anchor["d0_round_us_measured"],
        "e21_draft_us_marginal": anchor["draft_us_marginal"],
        "e21_serial_decode_seconds": anchor["serial_decode_seconds"],
        "e21_mtp_decode_seconds": anchor["mtp_decode_seconds"],
        # superseded within-arm regression, logged so the retraction is auditable
        "e21_regression_draft_cost_units": regression["draft_cost_units_measured"],
        "e21_regression_round_us_intercept": regression["round_us_intercept"],
        "e21_regression_draft_us_slope": regression["draft_us_slope"],
        "e21_regression_r_squared": regression["r_squared"],
        # oracle headroom priced at this prompt's own leg anchor
        "e21_oracle_gain_pct": nonparam["gain_pct"],
        "e21_oracle_drafts_saved": nonparam["drafts_saved"],
        "e21_oracle_tokens_lost": nonparam["tokens_lost"],
        "e21_round_timer_coverage_of_decode":
            nonparam["round_timer_coverage_of_decode"],
        # correctness gates
        "e21_all_tokens_matched": mtp["all_tokens_matched"],
        "e21_parity_all_ok": mtp["parity_all_ok"],
        "e21_residual_divergence_count": mtp["residual_divergence_count"],
        "e21_declared_rows_total": mtp["declared_rows_total"],
        "e21_reference_checked_row_total": mtp["reference_checked_row_total"],
        "e21_rejected_rows_reference_checked":
            mtp["rejected_rows_reference_checked"],
        "e21_rejected_draft_total": mtp["rejected_draft_total"],
        "e21_correctness_passed": correctness.get("passed"),
        "e21_correctness_checked_steps": correctness.get("checked_steps"),
    }
    metrics.update(histogram_metrics("e21", summary))
    metrics.update({f"e21_auc_{f}": v for f, v in auc.items()})
    metrics.update(thermals(meta))
    return config, metrics, mtp["block_request_seconds"]


def headline_payload(report: dict, thermal_rows: list[dict]) -> tuple[dict, dict]:
    pooled = report["pooled_summary"]
    auc = report["separability_auc"]
    ranked = report["ranked"]
    top = ranked[0] if ranked else {}
    firing = best_firing(report)
    nonparam = report["nonparametric_oracle_per_prompt"]
    dispatch = report["e23_dispatch_accounting"]

    config = dict(host_config())
    config.update({
        "experiment": GROUP,
        "assignment_revision": "r1",
        "pr_number": 25,
        "arm": ARM,
        "prompt_count": len(report["prompts"]),
        "prompts": ",".join(sorted(
            k.removeprefix("probe-").removesuffix(f"-{ARM}")
            for k in report["prompts"])),
        "decode_tokens": 512,
        "local_mode": "local-iterate",
        "shipped_head_step_cost_ratio": report["replay_fidelity"]["shipped_h"],
        "sweep_scored_at_draft_cost": report["h_sweep_scored_at_x"],
        "gate_qualified_for_timing": GATE_QUALIFIED,
        "declination_arm_run": False,
        "declination_arm_skipped_reason":
            "best legal predicate is within noise of declining nothing",
    })

    auc_values = list(auc.values())
    metrics = {
        "e21_round_count": pooled["round_count"],
        "e21_proposed_row_count": pooled["proposed_drafts"],
        "e21_accepted_drafts": pooled["accepted_drafts"],
        "e21_effective_mean_draft_len": pooled["effective_mean_draft_len"],
        "e21_effective_max_draft_len": pooled["effective_max_draft_len"],
        "e21_zero_accept_rounds": pooled["zero_accept_rounds"],
        # the central retraction: shipped 0.18 vs measured marginal cost
        "e21_h_measured_pooled": report["h_measured_pooled"],
        "e21_h_measured_min": report["h_measured_range"][0],
        "e21_h_measured_max": report["h_measured_range"][1],
        "e21_h_shipped_minus_measured":
            report["replay_fidelity"]["shipped_h"] - report["h_measured_pooled"],
        # superseded pooled regression estimate, kept so the retraction is auditable
        "e21_regression_draft_cost_units":
            report["measured_cost_model"]["draft_cost_units_measured"],
        "e21_regression_r_squared": report["measured_cost_model"]["r_squared"],
        # headroom that exists ...
        "e21_oracle_gain_pct": report["oracle"]["gain_pct"],
        "e21_oracle_drafts_saved": report["oracle"]["drafts_saved"],
        "e21_oracle_tokens_lost": report["oracle"]["tokens_lost"],
        "e21_oracle_gain_pct_worst_prompt":
            min(v["gain_pct"] for v in nonparam.values()),
        "e21_oracle_gain_pct_best_prompt":
            max(v["gain_pct"] for v in nonparam.values()),
        # ... versus the signal that does not
        "e21_auc_best": max(auc_values),
        "e21_auc_best_feature": max(auc, key=auc.get),
        "e21_auc_worst": min(auc_values),
        "e21_auc_feature_count": len(auc),
        # top-ranked rule overall: routinely the degenerate "decline nothing"
        "e21_top_predicate": top.get("predicate"),
        "e21_top_predicate_threshold": top.get("threshold"),
        "e21_top_predicate_fires": top.get("fired"),
        "e21_top_predicate_gain_pct": top.get("gain_pct"),
        "e21_top_predicate_worst_prompt_gain_pct":
            top.get("worst_prompt_gain_pct"),
        # best rule that actually declines something
        "e21_best_firing_predicate": firing.get("predicate"),
        "e21_best_firing_threshold": firing.get("threshold"),
        "e21_best_firing_fires": firing.get("fired"),
        "e21_best_firing_precision_pct": finite(firing.get("precision_pct")),
        "e21_best_firing_drafts_saved": firing.get("drafts_saved"),
        "e21_best_firing_tokens_lost": firing.get("tokens_lost"),
        "e21_best_firing_gain_pct": firing.get("gain_pct"),
        "e21_best_firing_worst_prompt_gain_pct":
            firing.get("worst_prompt_gain_pct"),
        # offline instrument fidelity
        "e21_replay_mismatches": report["replay_fidelity"]["mismatches"],
        "e21_replay_rounds": report["replay_fidelity"]["rounds"],
        "e21_replay_exact": report["replay_fidelity"]["exact"],
        # E23 relay: does declining a round cross the verify-forward dispatch cliff?
        "e21_mechanism_changes_S": dispatch["changes_S"],
        "e21_mechanism_changes_nConfirmed": dispatch["changes_nConfirmed"],
        "e21_mechanism_changes_mask": dispatch["changes_mask"],
        "e21_dispatch_per_round_shipped": dispatch["shipped"]["dispatches_per_round"],
        "e21_gdn_dispatch_per_round_shipped":
            dispatch["shipped"]["gdn_dispatches_per_round"],
        "e21_rounds_at_dispatch_max_M2": dispatch["rounds_at_dispatch_maximum_M2"],
        "e21_rounds_one_row_above_cliff_M3":
            dispatch["rounds_one_row_above_cliff_M3"],
        "e21_oracle_dispatch_delta_pct": dispatch["oracle_dispatch_delta_pct"],
    }
    metrics.update(histogram_metrics("e21", pooled))
    metrics.update({f"e21_auc_{f}": v for f, v in auc.items()})
    metrics.update(thermal_robustness(thermal_rows))
    return config, metrics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path,
                        default=Path(".mlxfast-private/e21/runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    if not args.dry_run:
        import wandb

    urls = {}
    thermal_rows = []
    for key in sorted(report["per_prompt_summary"]):
        prompt = key.removeprefix("probe-").removesuffix(f"-{ARM}")
        run_dir = args.runs_root / key
        config, metrics, series = probe_payload(prompt, run_dir, report)
        thermal_rows.append({
            "prompt": prompt,
            "entry_c": metrics["e21_thermal_before_gpu_temp"],
            "exit_c": metrics["e21_thermal_after_gpu_temp"],
            "d0_us": metrics["e21_d0_round_us_measured"],
            "h": metrics["e21_h_measured"],
        })
        name = f"e21-probe-{prompt}"
        if args.dry_run:
            print(f"--- {name}\n  config={json.dumps(config, sort_keys=True)}"
                  f"\n  metrics={json.dumps(metrics, sort_keys=True)}")
            continue
        run = wandb.init(entity=ENTITY, project=PROJECT, name=name, group=GROUP,
                         job_type="traced-probe", config=config, reinit=True,
                         tags=["qwen38-r1-e21", "shipped-default-histogram",
                               ARM, prompt])
        for i, seconds in enumerate(series):
            run.log({"e21_block_request_seconds": seconds}, step=i)
        run.summary.update(metrics)
        urls[name] = run.url
        print(f"{name}: {run.url}")
        run.finish()

    config, metrics = headline_payload(report, thermal_rows)
    if args.dry_run:
        print(f"--- e21-headline\n  config={json.dumps(config, sort_keys=True)}"
              f"\n  metrics={json.dumps(metrics, sort_keys=True)}")
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, name="e21-headline",
                     group=GROUP, job_type="analysis", config=config, reinit=True,
                     tags=["qwen38-r1-e21", "analysis", "negative-result"])
    # The uniform-threshold sweep is the load-bearing curve: it shows that any
    # effectively-uniform declination collapses the histogram it was meant to keep.
    for entry in sorted(report["h_sweep"].values(), key=lambda e: e["pooled"]["h"]):
        p = entry["pooled"]
        run.log({
            "e21_sweep_h": p["h"],
            "e21_sweep_proposed_rows": p["proposed_rows"],
            "e21_sweep_mean_draft_len": p["mean_depth"],
            "e21_sweep_max_draft_len": p["max_depth"],
            "e21_sweep_accepted": p["accepted"],
            "e21_sweep_accepted_rate_pct": p["accepted_rate_pct"],
            "e21_sweep_cost_per_token": p["cost_per_token"],
            "e21_sweep_gain_pct": p["gain_pct"],
            "e21_sweep_worst_prompt_gain_pct": entry["worst_prompt_gain_pct"],
        })
    run.summary.update(metrics)
    for table in ("ranked", "leg_anchored_cost_model", "separability_auc_per_prompt",
                  "nonparametric_oracle_per_prompt", "per_prompt_summary", "h_sweep",
                  "e23_dispatch_accounting"):
        run.summary[f"e21_{table}_json"] = json.dumps(
            finite(report[table]), sort_keys=True, allow_nan=False)
    print(f"e21-headline: {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
