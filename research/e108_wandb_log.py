#!/usr/bin/env python3
"""Publish the E108 instruction-fetch experiment to W&B.

    usage: research/e108_wandb_log.py [--only RUN]

Four runs:

  `e108-instrument-validation`
      Rung 0 step 1. The static census of `a_base`, `H_prune_narrow` and
      `I_prune_all_dead` on both GPU generations, the comparison against E102
      rung 1, and the pre-E100 reconstruction that explains the drift.
  `e108-exactness`
      Rung 0 step 2. The bit-exactness matrix over seven output widths and
      M = 1..8, the mis-prune arm that the generic fallback absorbs, and the
      no-fallback control that must fire.
  `e108-rung0-timing`
      Rung 0 step 3. The counterbalanced timing of the three arms plus two
      comment-only null arms, and the verdict against the pre-registered rule.
  `e108-extent`
      Rung 0 step 4. Where each live case sits inside the compiled entry point.

Every leg ran with no thermal gate, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim. No number here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e108-wide-qmv-instruction-fetch"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
OUT = pathlib.Path("research/out")

BASE_SHA = "05d88b8fc0976ea3ff17c42f13890c1b8c7f0297"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
BASE_ARM = "a_base"
WARMUP_BLOCKS = 1
CELL_LIMIT_PCT = 1.0
POOLED_FLOOR_PCT = 0.2

# E102 rung 1, on the pre-E100 tree it censused.
E102 = {
    "a_base": {LOCAL_ARCH: (94, 121072, "f54263e7"),
               RANKED_ARCH: (91, 126984, "846d5999")},
    "h_prunenarrow": {LOCAL_ARCH: (94, 75948, "329766ff"),
                      RANKED_ARCH: (91, 79190, "53e07912")},
    "i_pruneall": {LOCAL_ARCH: (94, 70352, "39c27e83"),
                   RANKED_ARCH: (91, 73380, "f178d05e")},
}
PRE_E100 = {"a_base": "v_a_pre_e100", "h_prunenarrow": "v_h_pre_e100",
            "i_pruneall": "v_i_pre_e100"}


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


def start(name: str, job_type: str, question: str, step: int, config: dict,
          meta_dir: str = "e108-rung0"):
    meta = read_meta(OUT / meta_dir / "meta.txt")
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "rung": 0, "step": step,
            "question": question, "entry_cell": ENTRY_CELL,
            "host": HOST, "hostname": meta.get("host"),
            "chip": meta.get("chip"), "toolchain": meta.get("toolchain"),
            "metal_toolchain": meta.get("metal_toolchain"),
            "git_head": meta.get("git_head"),
            "base_sha": BASE_SHA, "advisor_branch": ADVISOR_BRANCH,
            "shipped_source_touched": False,
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
        artifact.add_file(str(path))
    run.log_artifact(artifact)


def log_instrument_validation() -> None:
    now = json.loads((OUT / "e108/instrument-validation.json").read_text())
    old = json.loads((OUT / "e108/e102base-census.json").read_text())
    run = start(
        "e108-instrument-validation", "census",
        "Are the timed arms the E102 arms, and if the text figures moved, why?",
        1, {"arms": list(now["arms"]), "architectures": now["architectures"]},
        meta_dir="e108-exact")

    table = wandb.Table(columns=[
        "arm", "arch", "registers", "spill_bytes", "text_bytes", "text_sha8",
        "e102_registers", "e102_text_bytes", "e102_text_sha8",
        "text_delta_bytes", "text_delta_pct", "gate_within_1pct"])
    summary: dict[str, object] = {}
    for arm, rec in now["arms"].items():
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            reg, spill = rec[arch]["registers"], rec[arch]["spill_bytes"]
            text, sha = rec[arch]["text_bytes"], rec[arch]["text_sha8"]
            e_reg, e_text, e_sha = E102[arm][arch]
            delta = text - e_text
            pct = 100.0 * delta / e_text
            table.add_data(arm, arch, reg, spill, text, sha, e_reg, e_text,
                           e_sha, delta, pct, abs(pct) <= 1.0)
            tag = f"{arm}_{arch.replace('applegpu_', '')}"
            summary[f"text_bytes/{tag}"] = text
            summary[f"registers/{tag}"] = reg
            summary[f"spill_bytes/{tag}"] = spill
            summary[f"text_delta_pct_vs_e102/{tag}"] = pct
    run.log({"instrument_validation": table})

    # The pre-E100 reconstruction: one instantiation reversed, nothing else.
    recon = wandb.Table(columns=[
        "arm", "arch", "registers", "spill_bytes", "text_bytes", "text_sha8",
        "e102_text_bytes", "e102_text_sha8", "reproduces_e102"])
    exact = True
    for arm, name in PRE_E100.items():
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            rec = old["arms"][name][arch]
            e_reg, e_text, e_sha = E102[arm][arch]
            ok = rec["text_bytes"] == e_text and rec["text_sha8"] == e_sha
            exact = exact and ok
            recon.add_data(name, arch, rec["registers"], rec["spill_bytes"],
                           rec["text_bytes"], rec["text_sha8"], e_text, e_sha,
                           ok)
    run.log({"pre_e100_reconstruction": recon})

    for arch in (LOCAL_ARCH, RANKED_ARCH):
        tag = arch.replace("applegpu_", "")
        a = now["arms"]["a_base"][arch]["text_bytes"]
        i = now["arms"]["i_pruneall"][arch]["text_bytes"]
        h = now["arms"]["h_prunenarrow"][arch]["text_bytes"]
        summary[f"dead_text_bytes_removed/{tag}"] = a - i
        summary[f"dead_text_share_pct/{tag}"] = 100.0 * (a - i) / a
        summary[f"narrow_text_bytes_removed/{tag}"] = a - h
        e_a, e_i = E102["a_base"][arch][1], E102["i_pruneall"][arch][1]
        summary[f"dead_text_bytes_removed_e102/{tag}"] = e_a - e_i
    summary["pre_e100_reproduces_e102_exactly"] = exact
    summary["gate_fired"] = True
    summary["gate_cause"] = "base moved: the E100 collapse, M=5 IPG 3 -> 5"
    run.summary.update(summary)
    attach(run, OUT / "e108/instrument-validation.json",
           OUT / "e108/e102base-census.json")
    run.finish()


def fidelity_rows(doc: dict) -> list[dict]:
    return [row for row in doc["measurements"] if row["kind"] == "fidelity"]


def log_exactness() -> None:
    exact_doc = json.loads((OUT / "e108-exact/probe.json").read_text())
    ctl_doc = json.loads((OUT / "e108-control/probe.json").read_text())
    run = start(
        "e108-exactness", "fidelity",
        "Is pruning the dead cases a no-op on every scored shape and width, "
        "and can the comparison detect a mis-prune?",
        2, {"arms": exact_doc["arms"], "control_arms": ctl_doc["arms"],
            "widths": "1..8", "shapes": 7},
        meta_dir="e108-exact")

    table = wandb.Table(columns=["source", "shape", "m", "arm",
                                 "exact_required", "differing", "total",
                                 "bit_identical"])
    checks = failures = 0
    for label, doc in (("prune", exact_doc), ("control", ctl_doc)):
        for row in fidelity_rows(doc):
            for arm in row["arms"]:
                table.add_data(label, row["shape"], row["m"], arm["arm"],
                               arm["exact_required"], arm["differing"],
                               arm["total"], arm["bit_identical"])
                if label == "prune" and arm["exact_required"]:
                    checks += 1
                    failures += 0 if arm["bit_identical"] else 1
    run.log({"bit_exactness": table})

    controls = wandb.Table(columns=["kind", "shape", "m", "arm", "differing",
                                    "total", "fired"])
    perturb_fired = 0
    for doc in (exact_doc, ctl_doc):
        for row in doc["measurements"]:
            if row["kind"] != "positive_control":
                continue
            controls.add_data("activation_perturbation", row["shape"],
                              row["m"], row["arm"], row["differing"],
                              row["total"], row["detected"])
            perturb_fired += 1 if row["detected"] else 0
    nofallback = [(row["shape"], row["m"], arm["differing"], arm["total"])
                  for row in fidelity_rows(ctl_doc) for arm in row["arms"]
                  if arm["arm"] == "p_nofallback5"]
    fired_cells = [c for c in nofallback if c[2]]
    for shape, m, differing, total in nofallback:
        controls.add_data("no_fallback_misprune", shape, m, "p_nofallback5",
                          differing, total, bool(differing))
    misprune = [(row["shape"], row["m"], arm["differing"])
                for row in fidelity_rows(exact_doc) for arm in row["arms"]
                if arm["arm"] == "p_misprune5"]
    run.log({"positive_controls": controls})

    run.summary.update({
        "exactness_checks": checks,
        "exactness_failures": failures,
        "activation_control_cells_fired": perturb_fired,
        "no_fallback_control_cells_fired": len(fired_cells),
        "no_fallback_control_widths_fired": sorted({c[1] for c in fired_cells}),
        "no_fallback_control_elements_differing": sum(c[2] for c in fired_cells),
        "misprune_generic_fallback_cells": len(misprune),
        "misprune_generic_fallback_differing": sum(d for _, _, d in misprune),
        "generic_fallback_is_bit_exact": all(d == 0 for _, _, d in misprune),
    })
    attach(run, OUT / "e108-exact/probe.json", OUT / "e108-control/probe.json",
           OUT / "e108-exact/meta.txt", OUT / "e108-control/meta.txt")
    run.finish()


def timing_cells(doc: dict) -> dict:
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        cell = cells.setdefault((row["shape"], row["m"]),
                                {"bytes": row["read_bytes"], "seconds": {},
                                 "temp": row["gpu_temp_entry_c"]})
        for arm, sec in row["seconds"].items():
            cell["seconds"].setdefault(arm, []).append(sec)
    return cells


def log_timing() -> None:
    doc = json.loads((OUT / "e108-rung0/probe.json").read_text())
    census = json.loads((OUT / "e108/instrument-validation.json").read_text())
    meta = read_meta(OUT / "e108-rung0/meta.txt")
    arms = list(doc["arms"])
    others = [a for a in arms if a != BASE_ARM]
    cells = timing_cells(doc)

    run = start(
        "e108-rung0-timing", "timing",
        "Does text the measured width never executes cost time?",
        3, {"arms": arms, "blocks_per_cell": doc["pairs"],
            "warmup_blocks_dropped": WARMUP_BLOCKS, "order": doc["order"],
            "cells": len(cells),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c")},
        meta_dir="e108-rung0")

    table = wandb.Table(columns=["shape", "m", "base_us", "arm", "median_pct",
                                 "min_pct", "max_pct", "blocks",
                                 "text_bytes_g17s", "text_bytes_g16s"])
    pooled: dict[str, list[float]] = {a: [] for a in others}
    for (shape, m), cell in sorted(cells.items()):
        base = cell["seconds"][BASE_ARM]
        for arm in others:
            values = [100.0 * (v - b) / b
                      for v, b in zip(cell["seconds"][arm], base)]
            med = statistics.median(values)
            pooled[arm].append(med)
            source = arm[:-5] if arm.endswith("_null") else arm
            table.add_data(
                shape, m, 1e6 * statistics.median(base), arm, med,
                min(values), max(values), len(values),
                census["arms"].get(source, {}).get(RANKED_ARCH, {})
                .get("text_bytes"),
                census["arms"].get(source, {}).get(LOCAL_ARCH, {})
                .get("text_bytes"))
    run.log({"per_cell_deltas": table})

    summary: dict[str, object] = {}
    for arm in others:
        values = sorted(pooled[arm])
        med = statistics.median(values)
        over = [v for v in values if abs(v) > CELL_LIMIT_PCT]
        summary[f"pooled_median_pct/{arm}"] = med
        summary[f"pooled_mean_pct/{arm}"] = statistics.fmean(values)
        summary[f"pooled_min_pct/{arm}"] = values[0]
        summary[f"pooled_max_pct/{arm}"] = values[-1]
        summary[f"cells_over_1pct/{arm}"] = len(over)
        summary[f"inside_instrument_floor/{arm}"] = abs(med) <= POOLED_FLOOR_PCT

    verdict = json.loads((OUT / "e108/rung0-verdict.json").read_text())
    for arm in others:
        lo, hi = verdict["verdict"][arm]["pooled_median_ci95_pct"]
        summary[f"pooled_median_ci95_lo_pct/{arm}"] = lo
        summary[f"pooled_median_ci95_hi_pct/{arm}"] = hi

    target = "i_pruneall"
    med = statistics.median(pooled[target])
    over = [v for v in pooled[target] if abs(v) > CELL_LIMIT_PCT]
    negative = abs(med) <= POOLED_FLOOR_PCT and not over
    # A `_null` arm is the same machine code in a different palindrome slot, so
    # its pooled median is what the instrument reads when the true effect is
    # zero. A real arm has to clear that, not just the pre-registered floor.
    zero_point = verdict["instrument_zero_point_pct"]
    summary.update({
        "stop_rule_pooled_floor_pct": POOLED_FLOOR_PCT,
        "stop_rule_cell_limit_pct": CELL_LIMIT_PCT,
        "stop_rule_arm": target,
        "stop_rule_negative": negative,
        "hypothesis_supported": "H0_null" if negative else "H1_or_H2",
        "instrument_zero_point_pct": zero_point,
        "effect_below_zero_point": abs(med) <= zero_point,
        "gpu_temp_entry_spread_c": (
            max(c["temp"] for c in cells.values())
            - min(c["temp"] for c in cells.values())),
    })
    run.summary.update(summary)
    attach(run, OUT / "e108-rung0/probe.json", OUT / "e108-rung0/meta.txt",
           OUT / "e108/rung0-verdict.json")
    run.finish()


def log_extent() -> None:
    doc = json.loads((OUT / "e108/extent-census.json").read_text())
    ref = doc["reference_arm"]
    run = start(
        "e108-extent", "census",
        "Where does each live case sit inside the compiled entry point?",
        4, {"reference_arm": ref, "arms": list(doc["arms"])},
        meta_dir="e108-exact")

    table = wandb.Table(columns=["arm", "arch", "text_bytes", "registers",
                                 "spill_bytes", "case_extent_bytes",
                                 "offset_from_entry_bytes",
                                 "bytes_after_case"])
    for name, rec in doc["arms"].items():
        if name == ref:
            continue
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            edit = rec["vs_reference"][arch]
            total = doc["arms"][ref][arch]["text_bytes"]
            table.add_data(
                name, arch, rec[arch]["text_bytes"], rec[arch]["registers"],
                rec[arch]["spill_bytes"],
                edit["ref_changed_bytes"], edit["common_prefix_bytes"],
                total - edit["ref_changed_to"])
    run.log({"executed_path_extent": table})

    summary: dict[str, object] = {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        short = arch.replace("applegpu_", "")
        total = doc["arms"][ref][arch]["text_bytes"]
        live = sum(total - rec[arch]["text_bytes"]
                   for name, rec in doc["arms"].items() if name != ref)
        summary[f"reference_text_bytes/{short}"] = total
        summary[f"live_case_extent_bytes/{short}"] = live
        summary[f"live_case_share_pct/{short}"] = 100.0 * live / total
        # Metal allocates registers once for the whole entry point, so any case
        # whose removal lowers the count is setting the ceiling for every width.
        base_reg = doc["arms"][ref][arch]["registers"]
        summary[f"reference_registers/{short}"] = base_reg
        for name, rec in doc["arms"].items():
            if name == ref or rec[arch]["registers"] == base_reg:
                continue
            summary[f"register_ceiling_case/{short}"] = name
            summary[f"register_ceiling_drop/{short}"] = (
                base_reg - rec[arch]["registers"])
    run.summary.update(summary)
    attach(run, OUT / "e108/extent-census.json")
    run.finish()


RUNS = {
    "instrument": log_instrument_validation,
    "exactness": log_exactness,
    "timing": log_timing,
    "extent": log_extent,
}


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
