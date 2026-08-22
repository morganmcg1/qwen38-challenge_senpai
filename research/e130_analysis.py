#!/usr/bin/env python3
"""E130 rung 2: fit the occupancy response function from the ladder session.

The ladder arms differ ONLY in the register count of the shared entry point.
Every one of them executes the same instructions at M = 3, 6 and 9, because
those widths reach `qmv_fast_crossrow_affine4_g64_wide<T, 3>` and nothing else,
and the register count is set by a body behind an unreachable switch case. So
the contrast between two ladder rungs has a register delta and a zero executed
instruction delta, which is the clean case under rule 86.

`base` is NOT a ladder rung. It carries the shipped wide<4> and wide<5> bodies,
so it holds the same 32 simdgroups as `l94` with a different ISA text size. It
is the pruning control, not an occupancy point.

Rule 85: no coefficient is pooled across widths without a width stratum. Every
fit below is computed per (shape, width) first.

  python3 research/e130_analysis.py --timing /tmp/e130-rung2.json \
      --census research/e130-artifacts/rung2-session-census.json \
      --out research/e130-artifacts/rung2-response.json [--wandb]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
LADDER = ["l94", "l91", "l88", "l86", "l84", "l82"]
REFERENCE = "l94"
# Only these widths execute wide<3> alone, so only these are occupancy-clean.
CLEAN_WIDTHS = (3, 6, 9)
M5_ARMS = ["base", "prune_na5_ipg3", "prune_na5_pair"]
M5_PASSES = {"base": 1, "prune_na5_ipg3": 2, "prune_na5_pair": 3}


def mean_sem(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else float("nan"), float("nan"))
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(
        len(values))


def slope_through_origin(xs: list[float], ys: list[float]) -> float:
    """Least squares slope of y on x with the intercept pinned at the origin.

    y is a percentage gain against the reference rung, so it is exactly zero at
    x = 0 by construction. Fitting a free intercept would spend a degree of
    freedom re-estimating a quantity the design already fixes.
    """
    sxx = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares with an intercept. Returns slope, intercept, r2."""
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def load(timing: pathlib.Path, census: pathlib.Path) -> tuple[dict, dict]:
    t = json.loads(timing.read_text())
    c = json.loads(census.read_text())
    return t, c


def arm_registers(census: dict, arm: str, arch: str) -> dict:
    cell = census["variants"][arm]["cells"][arch]["entry"]
    return {"registers": cell["registers"],
            "spill_bytes": cell["spill_bytes"],
            "simdgroups": cell["resident_simdgroups_derived"],
            "isa_text_bytes": cell["text_bytes"]}


def collect(timing: dict) -> dict:
    """Arm seconds per (shape, width), with block 0 discarded."""
    out: dict = {}
    for row in timing["measurements"]:
        if row["kind"] != "timing" or row["block"] == 0:
            continue
        key = (row["shape"], row["m"])
        for arm, sec in row["seconds"].items():
            out.setdefault(key, {}).setdefault(arm, []).append(sec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=pathlib.Path, required=True)
    ap.add_argument("--census", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--clean-widths", default="3,6,9",
                    help="widths whose executed instructions are identical "
                         "across every ladder rung")
    ap.add_argument("--run-id", default="e130rung2")
    ap.add_argument("--run-name", default="e130-rung2-occupancy-response")
    ap.add_argument("--artifact-name", default="e130-rung2")
    args = ap.parse_args()

    global CLEAN_WIDTHS
    CLEAN_WIDTHS = tuple(int(w) for w in args.clean_widths.split(","))

    timing, census = load(args.timing, args.census)
    cells = collect(timing)

    regs = {arm: {arch: arm_registers(census, arm, arch)
                  for arch in (LOCAL_ARCH, RANKED_ARCH)}
            for arm in census["variants"]}

    report: dict = {
        "harness": "local",
        "timing_valid": True,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "device": timing["device"],
        "architecture": timing["architecture"],
        "ranked_architecture": RANKED_ARCH,
        "base_sha": census.get("base_sha"),
        "arms": regs,
        "reference_rung": REFERENCE,
        "clean_widths": list(CLEAN_WIDTHS),
    }

    # --- exactness ------------------------------------------------------------
    exact, controls = [], []
    for row in timing["measurements"]:
        if row["kind"] == "fidelity":
            for a in row["arms"]:
                exact.append({"shape": row["shape"], "m": row["m"],
                              "arm": a["arm"], "differing": a["differing"],
                              "total": a["total"], "max_ulp": a["max_ulp"]})
        elif row["kind"] == "positive_control":
            controls.append({"shape": row["shape"], "m": row["m"],
                             "arm": row["arm"],
                             "activation_hit":
                                 row["activation_perturbed_differing"],
                             "metadata_hit":
                                 row["metadata_perturbed_differing"],
                             "restored_differing": row["restored_differing"]})
    report["exactness"] = exact
    report["positive_controls"] = controls
    ladder_clean = [e for e in exact
                    if e["arm"] in LADDER and e["m"] in CLEAN_WIDTHS]
    report["ladder_bit_exact_cells_differing"] = sum(
        e["differing"] for e in ladder_clean)
    report["ladder_bit_exact_cells_checked"] = len(ladder_clean)
    m5_exact = [e for e in exact if e["arm"] in M5_ARMS and e["m"] == 5]
    report["m5_route_cells_differing"] = sum(e["differing"] for e in m5_exact)
    report["m5_route_cells_checked"] = len(m5_exact)
    report["positive_control_fired"] = all(
        c["activation_hit"] > 0 and c["metadata_hit"] > 0 for c in controls)

    # --- thermal --------------------------------------------------------------
    thermal = [r for r in timing["measurements"] if r["kind"] == "thermal"]
    entries = [r["gpu_temp_entry_c"] for r in thermal]
    report["thermal"] = {
        "blocks": thermal,
        "entry_c_min": min(entries) if entries else None,
        "entry_c_max": max(entries) if entries else None,
        "entry_c_spread": (max(entries) - min(entries)) if entries else None,
    }

    # --- the occupancy response function --------------------------------------
    strata: list = []
    per_width: dict = {}
    for (shape, m), arms in sorted(cells.items()):
        if m not in CLEAN_WIDTHS or REFERENCE not in arms:
            continue
        ref_mean, ref_sem = mean_sem(arms[REFERENCE])
        xs, ys, rows = [], [], []
        for arm in LADDER:
            if arm not in arms:
                continue
            mu, sem = mean_sem(arms[arm])
            sg = regs[arm][LOCAL_ARCH]["simdgroups"]
            sg_ref = regs[REFERENCE][LOCAL_ARCH]["simdgroups"]
            x = 100.0 * (sg / sg_ref - 1.0)
            y = 100.0 * (ref_mean - mu) / ref_mean
            # Rule 86: both rungs of every published cell carry their registers.
            rows.append({
                "arm": arm, "seconds_mean": mu, "seconds_sem": sem,
                "blocks": len(arms[arm]),
                "registers": regs[arm][LOCAL_ARCH]["registers"],
                "reference_registers": regs[REFERENCE][LOCAL_ARCH]["registers"],
                "register_delta": (regs[arm][LOCAL_ARCH]["registers"]
                                   - regs[REFERENCE][LOCAL_ARCH]["registers"]),
                "executed_instruction_delta": 0,
                "isa_text_delta": (regs[arm][LOCAL_ARCH]["isa_text_bytes"]
                                   - regs[REFERENCE][LOCAL_ARCH]
                                   ["isa_text_bytes"]),
                "simdgroups": sg, "residency_pct": x, "gain_pct": y,
                "gain_sem_pct": 100.0 * math.hypot(sem, ref_sem) / ref_mean,
            })
            xs.append(x)
            ys.append(y)
        if len(xs) < 3:
            continue
        c = slope_through_origin(xs, ys)
        inv = [1.0 / r["simdgroups"] for r in rows]
        sg_lin = [float(r["simdgroups"]) for r in rows]
        secs = [r["seconds_mean"] for r in rows]
        _, _, r2_inv = linfit(inv, secs)
        _, _, r2_sg = linfit(sg_lin, secs)
        stratum = {"shape": shape, "m": m, "coefficient_pct_per_pct": c,
                   "r2_time_vs_inv_sg": r2_inv, "r2_time_vs_sg": r2_sg,
                   "rungs": rows}
        strata.append(stratum)
        per_width.setdefault(m, []).append(c)
    report["strata"] = strata

    # Rule 85: a width fixed effect, not a pooled correlation.
    report["coefficient_by_width"] = {
        str(m): {"mean": statistics.fmean(v),
                 "sem": (statistics.stdev(v) / math.sqrt(len(v))
                         if len(v) > 1 else float("nan")),
                 "n_strata": len(v)}
        for m, v in sorted(per_width.items())}
    all_c = [s["coefficient_pct_per_pct"] for s in strata]
    if all_c:
        mu, sem = mean_sem(all_c)
        report["coefficient_pct_per_pct"] = mu
        report["coefficient_sem"] = sem
        report["coefficient_2sigma_low"] = mu - 2.0 * sem
        report["coefficient_2sigma_high"] = mu + 2.0 * sem
        report["coefficient_n_strata"] = len(all_c)

    # Shape of the response: the mean gain at each rung, averaged over strata.
    by_rung: dict = {}
    for s in strata:
        for r in s["rungs"]:
            by_rung.setdefault(r["arm"], []).append(r["gain_pct"])
    report["response_curve"] = [
        {"arm": a,
         "registers": regs[a][LOCAL_ARCH]["registers"],
         "simdgroups": regs[a][LOCAL_ARCH]["simdgroups"],
         "residency_pct": 100.0 * (regs[a][LOCAL_ARCH]["simdgroups"]
                                   / regs[REFERENCE][LOCAL_ARCH]["simdgroups"]
                                   - 1.0),
         "gain_pct_mean": statistics.fmean(by_rung[a]),
         "gain_pct_sem": (statistics.stdev(by_rung[a])
                          / math.sqrt(len(by_rung[a]))
                          if len(by_rung[a]) > 1 else float("nan")),
         "marginal_pct_per_sg": None}
        for a in LADDER if a in by_rung]
    for i in range(1, len(report["response_curve"])):
        lo, hi = report["response_curve"][i - 1], report["response_curve"][i]
        d_sg = hi["simdgroups"] - lo["simdgroups"]
        if d_sg:
            hi["marginal_pct_per_sg"] = (hi["gain_pct_mean"]
                                         - lo["gain_pct_mean"]) / d_sg

    # --- the pruning control: base against l94, same registers, same sg -------
    control = []
    for (shape, m), arms in sorted(cells.items()):
        if m not in CLEAN_WIDTHS or "base" not in arms or REFERENCE not in arms:
            continue
        b, b_sem = mean_sem(arms["base"])
        l, l_sem = mean_sem(arms[REFERENCE])
        control.append({
            "shape": shape, "m": m, "base_s": b, "l94_s": l,
            "delta_pct": 100.0 * (b - l) / b,
            "delta_sem_pct": 100.0 * math.hypot(b_sem, l_sem) / b,
            "register_delta": (regs[REFERENCE][LOCAL_ARCH]["registers"]
                               - regs["base"][LOCAL_ARCH]["registers"]),
            "isa_text_delta": (regs[REFERENCE][LOCAL_ARCH]["isa_text_bytes"]
                               - regs["base"][LOCAL_ARCH]["isa_text_bytes"]),
        })
    report["pruning_control"] = control
    if control:
        mu, sem = mean_sem([c["delta_pct"] for c in control])
        report["pruning_control_delta_pct"] = mu
        report["pruning_control_delta_sem_pct"] = sem

    # --- the isolated M = 5 cell ----------------------------------------------
    m5 = []
    for (shape, m), arms in sorted(cells.items()):
        if m != 5 or "base" not in arms:
            continue
        b, b_sem = mean_sem(arms["base"])
        for arm in M5_ARMS[1:]:
            if arm not in arms:
                continue
            mu, sem = mean_sem(arms[arm])
            m5.append({
                "shape": shape, "arm": arm,
                "weight_passes": M5_PASSES[arm],
                "base_s": b, "arm_s": mu,
                "delta_pct": 100.0 * (b - mu) / b,
                "delta_sem_pct": 100.0 * math.hypot(sem, b_sem) / b,
                "register_delta": (regs[arm][LOCAL_ARCH]["registers"]
                                   - regs["base"][LOCAL_ARCH]["registers"]),
                "simdgroup_delta": (regs[arm][LOCAL_ARCH]["simdgroups"]
                                    - regs["base"][LOCAL_ARCH]["simdgroups"]),
            })
    report["m5_cell"] = m5
    for arm in M5_ARMS[1:]:
        vals = [r["delta_pct"] for r in m5 if r["arm"] == arm]
        if vals:
            mu, sem = mean_sem(vals)
            report["m5_cell_%s_pct" % arm] = mu
            report["m5_cell_%s_sem_pct" % arm] = sem

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    # --- console --------------------------------------------------------------
    print("device %s  architecture %s  ranked %s"
          % (report["device"], report["architecture"], RANKED_ARCH))
    print("entry temperature %.1f to %.1f C, spread %.1f C"
          % (report["thermal"]["entry_c_min"], report["thermal"]["entry_c_max"],
             report["thermal"]["entry_c_spread"]))
    print("cool_gate_passed_real_gate=false gate_qualified_for_timing=false")
    print("\nladder bit exactness at M in %s: %d differing over %d cells"
          % (CLEAN_WIDTHS, report["ladder_bit_exact_cells_differing"],
             report["ladder_bit_exact_cells_checked"]))
    print("M=5 route exactness: %d differing over %d cells"
          % (report["m5_route_cells_differing"],
             report["m5_route_cells_checked"]))
    print("positive control fired for every arm: %s"
          % report["positive_control_fired"])

    print("\nresponse curve, mean over %d strata, reference %s"
          % (len(strata), REFERENCE))
    print("%-6s %5s %4s %9s %10s %8s %10s"
          % ("arm", "regs", "sg", "res %", "gain %", "sem", "d%/dsg"))
    for r in report["response_curve"]:
        print("%-6s %5d %4d %+9.2f %+10.3f %8.3f %10s"
              % (r["arm"], r["registers"], r["simdgroups"], r["residency_pct"],
                 r["gain_pct_mean"], r["gain_pct_sem"],
                 "-" if r["marginal_pct_per_sg"] is None
                 else "%+.4f" % r["marginal_pct_per_sg"]))

    if all_c:
        print("\ncoefficient %.4f %% per %% residency, 2 sigma [%.4f, %.4f], "
              "n=%d strata"
              % (report["coefficient_pct_per_pct"],
                 report["coefficient_2sigma_low"],
                 report["coefficient_2sigma_high"], len(all_c)))
    print("by width (rule 85, no pooled correlation):")
    for m, v in report["coefficient_by_width"].items():
        print("  M=%s  c=%.4f  sem=%.4f  n=%d"
              % (m, v["mean"], v["sem"], v["n_strata"]))
    print("mean r2  time vs 1/sg %.4f   time vs sg %.4f"
          % (statistics.fmean([s["r2_time_vs_inv_sg"] for s in strata]),
             statistics.fmean([s["r2_time_vs_sg"] for s in strata])))

    if control:
        print("\npruning control base vs l94, both 94 regs and 32 sg: "
              "%+.3f %% (sem %.3f), ISA text delta %+d bytes"
              % (report["pruning_control_delta_pct"],
                 report["pruning_control_delta_sem_pct"],
                 control[0]["isa_text_delta"]))

    if m5:
        print("\nisolated M=5 cell, occupancy held at 94 regs and 32 sg")
        for arm in M5_ARMS[1:]:
            if "m5_cell_%s_pct" % arm in report:
                print("  %-16s %d weight passes  %+.3f %% (sem %.3f)"
                      % (arm, M5_PASSES[arm], report["m5_cell_%s_pct" % arm],
                         report["m5_cell_%s_sem_pct" % arm]))

    print("\nwrote %s" % args.out)

    if args.wandb:
        import wandb
        run = wandb.init(project="qwen38-mlx-challenge-senpai",
                         entity="wandb-applied-ai-team",
                         id=args.run_id, name=args.run_name,
                         resume="allow", config={
                             "experiment": "E130",
                             "base_sha": report["base_sha"],
                             "harness": "local",
                             "architecture": report["architecture"],
                             "ranked_architecture": RANKED_ARCH,
                             "reference_rung": REFERENCE,
                             "clean_widths": list(CLEAN_WIDTHS),
                             "cool_gate_passed_real_gate": False,
                             "gate_qualified_for_timing": False,
                             "official_or_ranked_score": False,
                         })
        for r in report["response_curve"]:
            run.log({"ladder/registers": r["registers"],
                     "ladder/simdgroups": r["simdgroups"],
                     "ladder/residency_pct": r["residency_pct"],
                     "ladder/gain_pct": r["gain_pct_mean"],
                     "ladder/gain_sem_pct": r["gain_pct_sem"]})
        run.summary.update({
            k: v for k, v in report.items()
            if isinstance(v, (int, float, bool, str)) or v is None})
        art = wandb.Artifact(args.artifact_name, type="analysis")
        art.add_file(str(args.out))
        art.add_file(str(args.census))
        run.log_artifact(art)
        run.finish()
        print("logged to W&B run %s" % args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
