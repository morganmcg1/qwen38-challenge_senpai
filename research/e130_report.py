#!/usr/bin/env python3
"""E130 final report: fold every rung into one W&B record.

The rungs are measured by separate tools and land in separate artifacts. This
script reads them, recomputes the two headline metrics from the artifacts
rather than from prose, and publishes one run that another agent can audit
without re-running any GPU work.

Rule 89: the primary metric is DERIVED from a measured register count on a
ranked-architecture offline compile. It is not a measured occupancy and it is
not a measured time.

  python3 research/e130_report.py --out research/e130-artifacts/final.json \
      [--swift-test LOG] [--wandb]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

ART = pathlib.Path("research/e130-artifacts")
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
ARM = "prune_na5_pair"
# STREAM triad ceilings measured on each part, GB/s.
CEILING = {LOCAL_ARCH: 227.0, RANKED_ARCH: 542.8}
LABELS = {
    "harness": "local",
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
}


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def bandwidth_rows(files: list[str]) -> list[dict]:
    """Achieved read bandwidth of the unmodified arm, per (shape, width)."""
    cells: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for name in files:
        for row in load(name)["measurements"]:
            if row["kind"] != "timing" or row["block"] == 0:
                continue
            cells.setdefault((row["shape"], row["m"]), []).append(
                (row["read_bytes"], row["seconds"]["base"]))
    out = []
    for (shape, m) in sorted(cells):
        samples = cells[(shape, m)]
        gbs = samples[0][0] / statistics.fmean(s for _, s in samples) / 1e9
        out.append({"shape": shape, "m": m, "gb_per_s": gbs,
                    "pct_of_local_ceiling": 100.0 * gbs / CEILING[LOCAL_ARCH],
                    "pct_of_ranked_ceiling": 100.0 * gbs / CEILING[RANKED_ARCH]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    rung3 = load("rung3-shipped-arm.json")
    wide = load("rung2-analysis.json")
    narrow = load("rung2-m1-analysis.json")
    leg = load("rung0b.json")
    swift = load("swift-test.json")
    gates = load("gates.json")

    residency = rung3["residency_vs_base"][ARM]
    bw = bandwidth_rows(["rung2-m1-timing.json", "rung2-ladder-timing.json"])
    # Seed prefill is outside the scored decode window, so the phase shares
    # below are normalised over the decode phases only.
    decode = {k: v for k, v in leg["gpu_time"]["by_phase_total_ns"].items()
              if k != "seed_prefill"}
    decode_ns = sum(decode.values())

    report: dict = dict(LABELS)
    report.update({
        "experiment": "E130",
        "arm": ARM,
        "device": wide["device"],
        "architecture": LOCAL_ARCH,
        "ranked_architecture": RANKED_ARCH,
        "metric_kind": "derived_from_measured_register_count",

        # Primary metric, rule 89.
        "e130_ranked_g17s_f47_weighted_residency_pct":
            residency[RANKED_ARCH]["weighted_pct"],
        "e130_ranked_g17s_min_cell_residency_pct":
            residency[RANKED_ARCH]["min_cell_pct"],
        "e130_local_g16s_weighted_residency_pct":
            residency[LOCAL_ARCH]["weighted_pct"],
        "e130_ranked_g17s_entry_registers_before":
            rung3["audit"]["per_arch"][RANKED_ARCH]["entry_registers"],
        "e130_ranked_g17s_entry_simdgroups_before":
            residency[RANKED_ARCH]["entry_sg_before"],
        "e130_ranked_g17s_entry_simdgroups_after":
            residency[RANKED_ARCH]["entry_sg_after"],

        # Secondary metrics.
        "e130_residency_time_coefficient_pct_per_pct":
            wide["coefficient_pct_per_pct"],
        "e130_residency_time_coefficient_2sigma_low":
            wide["coefficient_2sigma_low"],
        "e130_residency_time_coefficient_2sigma_high":
            wide["coefficient_2sigma_high"],
        "e130_residency_time_coefficient_m1m2_pct_per_pct":
            narrow["coefficient_pct_per_pct"],
        "e130_residency_time_coefficient_m1m2_2sigma_high":
            narrow["coefficient_2sigma_high"],
        "e130_bit_exact_cells_differing":
            wide["ladder_bit_exact_cells_differing"]
            + wide["m5_route_cells_differing"]
            + narrow["ladder_bit_exact_cells_differing"],
        "e130_bit_exact_cells_checked":
            wide["ladder_bit_exact_cells_checked"]
            + wide["m5_route_cells_checked"]
            + narrow["ladder_bit_exact_cells_checked"],
        "e130_positive_control_fired":
            bool(wide["positive_control_fired"]
                 and narrow["positive_control_fired"]),

        # Source cost and spill safety.
        "e130_source_bytes_delta":
            rung3["variants"][ARM]["source_bytes"]
            - rung3["variants"]["base"]["source_bytes"],
        "e130_spill_free_local":
            rung3["audit"]["per_arch"][LOCAL_ARCH]["spill_free"],
        "e130_spill_free_ranked":
            rung3["audit"]["per_arch"][RANKED_ARCH]["spill_free"],
        "e130_entry_equals_max_live_body_ranked":
            rung3["audit"]["per_arch"][RANKED_ARCH]
            ["entry_equals_max_live_body"],

        # Rung 0b leg census: who actually reaches the entry point.
        "e130_entry_dispatches_total": sum(leg["counts"]["by_ntg"].values()),
        "e130_entry_dispatches_m1": leg["counts"]["by_ntg"].get("1", 0),
        "e130_entry_dispatches_m5": leg["counts"]["by_ntg"].get("5", 0),
        "e130_ntg_x_equals_5_reachable":
            leg["question_3_ntg_x_equals_5"]["reachable"],
        "e130_target_verify_gpu_share_pct": 100.0 * decode["target_verify"]
            / decode_ns,
        "e130_draft_head_gpu_share_pct": 100.0 * decode["draft_head"]
            / decode_ns,

        "bandwidth": bw,
        "response_curve_m369": wide["response_curve"],
        "response_curve_m12": narrow["response_curve"],
        "coefficient_by_width_m369": wide["coefficient_by_width"],
        "coefficient_by_width_m12": narrow["coefficient_by_width"],
        "residency_vs_base": rung3["residency_vs_base"],
        "e130_swift_test_issues": swift["issues"],
        "e130_swift_test_matches_pre_existing_floor":
            swift["matches_pre_existing_floor"],
        "swift_test": swift,
        "e130_twin_audit_ok": gates["twin_audit_ok"],
        "e130_editable_budget_ok": gates["editable_budget_ok"],
        "e130_assignment_scope_ok": gates["assignment_scope_ok"],
        "base_sha": gates["base_sha"],
        "gates": gates,
    })

    saturated = [r for r in bw if r["pct_of_local_ceiling"] >= 90.0]
    report["e130_local_cells_at_or_above_90pct_of_stream"] = len(saturated)
    report["e130_local_cells_measured"] = len(bw)
    report["e130_m1_mean_pct_of_local_ceiling"] = statistics.fmean(
        r["pct_of_local_ceiling"] for r in bw if r["m"] == 1)
    report["e130_m1_mean_pct_of_ranked_ceiling"] = statistics.fmean(
        r["pct_of_ranked_ceiling"] for r in bw if r["m"] == 1)

    args.out.write_text(json.dumps(report, indent=1) + "\n")

    print("E130 final report")
    print("  primary  ranked g17s weighted residency  %+.2f %% (derived)"
          % report["e130_ranked_g17s_f47_weighted_residency_pct"])
    print("  control  local g16s weighted residency   %+.2f %%"
          % report["e130_local_g16s_weighted_residency_pct"])
    print("  secondary coefficient M in 3,6,9  %+.4f %%/%% 2s [%+.4f, %+.4f]"
          % (report["e130_residency_time_coefficient_pct_per_pct"],
             report["e130_residency_time_coefficient_2sigma_low"],
             report["e130_residency_time_coefficient_2sigma_high"]))
    print("  secondary coefficient M in 1,2    %+.4f %%/%% 2s high %+.4f"
          % (report["e130_residency_time_coefficient_m1m2_pct_per_pct"],
             report["e130_residency_time_coefficient_m1m2_2sigma_high"]))
    print("  exactness %d differing over %d cells, positive control %s"
          % (report["e130_bit_exact_cells_differing"],
             report["e130_bit_exact_cells_checked"],
             report["e130_positive_control_fired"]))
    print("  saturation %d of %d local cells at or above 90 %% of STREAM; "
          "M=1 mean %.1f %% local vs %.1f %% ranked"
          % (report["e130_local_cells_at_or_above_90pct_of_stream"],
             report["e130_local_cells_measured"],
             report["e130_m1_mean_pct_of_local_ceiling"],
             report["e130_m1_mean_pct_of_ranked_ceiling"]))
    print("  wrote %s" % args.out)

    if args.wandb:
        import wandb
        run = wandb.init(project="qwen38-mlx-challenge-senpai",
                         entity="wandb-applied-ai-team",
                         id="e130final", name="e130-entry-point-occupancy-tax",
                         resume="allow",
                         config={k: report[k] for k in
                                 ("experiment", "arm", "device",
                                  "architecture", "ranked_architecture",
                                  "metric_kind", "harness",
                                  "cool_gate_passed_real_gate",
                                  "gate_qualified_for_timing",
                                  "official_or_ranked_score")})
        for row in report["response_curve_m369"]:
            run.log({"m369/residency_pct": row["residency_pct"],
                     "m369/registers": row["registers"],
                     "m369/simdgroups": row["simdgroups"],
                     "m369/gain_pct": row["gain_pct_mean"],
                     "m369/gain_sem_pct": row["gain_pct_sem"]})
        for row in report["response_curve_m12"]:
            run.log({"m12/residency_pct": row["residency_pct"],
                     "m12/registers": row["registers"],
                     "m12/simdgroups": row["simdgroups"],
                     "m12/gain_pct": row["gain_pct_mean"],
                     "m12/gain_sem_pct": row["gain_pct_sem"]})
        run.log({"bandwidth": wandb.Table(
            columns=["shape", "m", "gb_per_s", "pct_of_local_ceiling",
                     "pct_of_ranked_ceiling"],
            data=[[r["shape"], r["m"], r["gb_per_s"],
                   r["pct_of_local_ceiling"], r["pct_of_ranked_ceiling"]]
                  for r in bw])})
        run.log({"dispatch_census": wandb.Table(
            columns=["phase", "m", "n", "dispatches", "projection"],
            data=[[key.split("|")[0], int(key.split("|")[1][2:]),
                   int(key.split("|")[2][2:]), count,
                   leg["by_ntg_and_n"].get(
                       "|".join(key.split("|")[1:]), {}).get(
                           "projection", "")]
                  for key, count in
                  sorted(leg["counts"]["by_phase_ntg_n"].items())])})
        run.log({"register_census": wandb.Table(
            columns=["arm", "architecture", "cell", "registers",
                     "spill_bytes", "resident_simdgroups_derived"],
            data=[[arm, arch, cell, body["registers"], body["spill_bytes"],
                   body["resident_simdgroups_derived"]]
                  for arm, variant in rung3["variants"].items()
                  for arch, cells in variant["cells"].items()
                  for cell, body in cells.items()])})
        run.summary.update({k: v for k, v in report.items()
                            if isinstance(v, (int, float, bool, str))
                            or v is None})
        art = wandb.Artifact("e130-final", type="analysis")
        for path in sorted(ART.glob("*.json")):
            art.add_file(str(path))
        run.log_artifact(art)
        run.finish()
        print("  logged to W&B run e130final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
