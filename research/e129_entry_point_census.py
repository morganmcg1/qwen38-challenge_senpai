#!/usr/bin/env python3
"""E129 rung 1 — does templating the Route B QMV entry point on `M` pay?

    usage: research/e129_entry_point_census.py [--out PATH] [--keep DIR]
                                               [--wandb]

THE QUESTION. The shipped wide-QMV entry point is one pipeline holding a
`switch (qmv_m)` over the seven routed widths. A switch entry point's register
count is the MAXIMUM over its inlined branches, so the widest branch sets the
residency of every width. E120's g17s census read the M=5 branch at 102
registers and every other branch at 90 or 94, so M=5 alone costs the whole
dispatch four resident simdgroups at widths that never run it.

`MLXFast.metalKernel` accepts template arguments and MLX hashes them into the
pipeline name (`metal_kernel.cpp:289-338`), so templating the entry point on
`M` would compile and cache one specialization per width instead of one switch.
This census prices that change before any GPU time is spent on it.

WHAT IS MEASURED. Registers, spill bytes, machine-text bytes and resident
simdgroups for three variants of each of the two shipped QMV pipelines, on both
`applegpu_g16s` (this host) and `applegpu_g17s` (the ranked runner):

    switch      the shipped entry point, one pipeline for all widths
    templated   `M` and `IPG` as template parameters, one pipeline per width
    body        the inlined `qwen_e120_qmv_wide<NA>` alone, per accumulator
                width, which separates entry-point cost from body cost

Rule 56 requires the g17s census at the entry point as well as in the body, so
all three appear here.

HOW IT IS MEASURED. `xcrun metal-tt` runs the real AGX backend for a named
architecture on any Mac, wrapped by `research/agx_crossarch.py`. Zero GPU
seconds. The Metal source is lifted out of `Qwen35.swift` and the MLX signature
generation is reproduced by `research/e120_g17s_census.py`, which this module
imports rather than copies, so the switch rows here must reproduce the E120
switch rows exactly. That reproduction is the instrument's own control.

WHAT IT CANNOT SHOW. A census is a cost observation, never correctness
evidence and never a timing result. Residency is `budget // registers` with the
budgets FITTED in E123, so read the ratio between two variants, not the
absolute count.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402
import e120_g17s_census as e120  # noqa: E402

ARCHS = e120.ARCHS
SIMDGROUP_BUDGET = e120.SIMDGROUP_BUDGET
WIDTH_CASES = e120.WIDTH_CASES
# `Qwen35CustomQMV.widths` is 3...9; M=1 and M=2 reach other MLX kernels and
# are not part of this entry point at all.
ROUTED_WIDTHS = tuple(m for m, _ in WIDTH_CASES)
# Accumulator widths the body is instantiated at, from `qwen_e120_qmv_m`:
# IPG for a full group, and `max(TAIL, 2)` for the tail group.
BODY_NA = (2, 3, 4, 5)

# `Qwen35CustomQMV.minimumTableWidth`. M=3 runs the no-table replica pipeline,
# M>=4 runs the chunk-sum pipeline.
MINIMUM_TABLE_WIDTH = 4

# Ranked mean verify width and F83 median weight per prompt, from the advisor's
# E129 assignment. The weights are the median-sensitivity weights, so they do
# not sum to one; the census normalises them.
RANKED_WIDTH_MIX = {
    "beagle": (5.382, 0.4862),
    "medicine": (6.256, 0.2508),
    "essays": (6.087, 0.1598),
    "botany": (7.148, 0.0124),
    "republic": (5.989, 0.0100),
}
# The local fixture for contrast: mean width 7.359, 76.9 % of rounds at M=8.
LOCAL_MEAN_WIDTH = 7.359

# The advisor's rung-1 gate: below this ranked-weighted resident-simdgroup
# gain, close the axis and report the census as the result.
GATE_FRACTION = 0.05

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e129-entry-point-templating"


def templated_body(table: bool) -> str:
    """The entry point with `M` and `IPG` as compile-time parameters.

    Same statements as `qwen35E120QMVSource`, minus the width read and the
    switch. `qmv_stride` folds because `M` is constant, and `x_shape` is still
    named so MLX still binds the shape buffer.
    """
    sums = "xsums" if table else "qmv_null_sums"
    flag = "USE_TABLE" if table else "false"
    null_decl = "" if table else (
        "\n    const device float* qmv_null_sums = nullptr;"
    )
    return """    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    constexpr int qmv_stride = M <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_gx = int(qmv_tid.x);%s
    qwen_e120_qmv_m<M, IPG, %s>(
        w, scales, biases, x, %s, y,
        qmv_k, qmv_n, qmv_stride,
        qmv_gx, qmv_out_row, qmv_lid);""" % (null_decl, flag, sums)


def body_only(table: bool) -> str:
    """One inlined `qwen_e120_qmv_wide<NA>` with no width dispatch above it."""
    sums = "xsums" if table else "qmv_null_sums"
    flag = "USE_TABLE" if table else "false"
    null_decl = "" if table else (
        "\n    const device float* qmv_null_sums = nullptr;"
    )
    return """    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = 8;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_first_m = int(qmv_tid.x) * NA;%s
    qwen_e120_qmv_wide<NA, %s>(
        w, scales, biases, x, %s, y,
        qmv_k, qmv_n, qmv_stride,
        qmv_first_m, qmv_out_row, qmv_lid);""" % (null_decl, flag, sums)


def arm_source(header: str, table: bool) -> str:
    """One library holding every variant of one shipped pipeline."""
    base = ("qwen35_custom_affine4_g64_qmv_wide_sums_v1" if table
            else "qwen35_custom_affine4_g64_qmv_wide_v1")
    inputs = e120.QMV_INPUTS + ([("xsums", "float")] if table else [])
    use_table = [("bool", "USE_TABLE", "true")] if table else []
    parts = [e120.PRELUDE, header, ""]

    parts.append(e120.generate(base, inputs, e120.QMV_OUTPUTS,
                               e120.qmv_body(table), use_table or None))
    for m, ipg in WIDTH_CASES:
        parts.append(e120.generate(
            "%s_tmpl_m%d" % (base, m), inputs, e120.QMV_OUTPUTS,
            templated_body(table),
            [("int", "M", str(m)), ("int", "IPG", str(ipg))] + use_table))
    for na in BODY_NA:
        parts.append(e120.generate(
            "%s_body_na%d" % (base, na), inputs, e120.QMV_OUTPUTS,
            body_only(table),
            [("int", "NA", str(na))] + use_table))
    return "\n".join(parts) + "\n"


def classify(kernel: str) -> tuple[str, int | None]:
    """`(variant, width)` for a censused kernel name."""
    if "_tmpl_m" in kernel:
        return "templated", int(kernel.rsplit("_tmpl_m", 1)[1])
    if "_body_na" in kernel:
        return "body", int(kernel.rsplit("_body_na", 1)[1])
    return "switch", None


def rows(census: dict) -> list[dict]:
    out = []
    for arm, per_arch in census.items():
        for arch, kernels in per_arch.items():
            for kernel, cell in kernels.items():
                variant, index = classify(kernel)
                out.append({
                    "arm": arm,
                    "arch": arch.replace("applegpu_", ""),
                    "variant": variant,
                    "m": index if variant == "templated" else None,
                    "na": index if variant == "body" else None,
                    "registers": cell["registers"],
                    "spill_bytes": cell["spill_bytes"],
                    "text_bytes": cell["text_bytes"],
                    "resident_simdgroups": cell["resident_simdgroups"],
                })
    return out


def shipped_arm(m: int) -> str:
    """The pipeline the shipped gate runs at width `m`."""
    return "sumtable" if m >= MINIMUM_TABLE_WIDTH else "replica_no_table"


def residency(table: list[dict], arm: str, arch: str,
              variant: str, m: int | None) -> int:
    for row in table:
        if (row["arm"] == arm and row["arch"] == arch
                and row["variant"] == variant and row["m"] == m):
            return row["resident_simdgroups"]
    raise SystemExit("no %s %s %s M=%s row" % (arm, arch, variant, m))


def interpolate(table: list[dict], arch: str, width: float) -> tuple[float, float]:
    """Switch and templated residency at a FRACTIONAL mean verify width.

    A prompt's mean width is a mix of integer widths, and the census only has
    integer rows, so the value between two integers is INTERPOLATED and is
    marked as such wherever it is reported. The mix that produced the mean is
    not measured, so this is a linear stand-in for it, not the histogram.
    """
    low = max(min(ROUTED_WIDTHS), min(int(math.floor(width)), max(ROUTED_WIDTHS)))
    high = min(max(ROUTED_WIDTHS), low + 1)
    frac = 0.0 if high == low else width - low
    switch = residency(table, shipped_arm(low), arch, "switch", None)
    switch_high = residency(table, shipped_arm(high), arch, "switch", None)
    tmpl_low = residency(table, shipped_arm(low), arch, "templated", low)
    tmpl_high = residency(table, shipped_arm(high), arch, "templated", high)
    return (switch * (1 - frac) + switch_high * frac,
            tmpl_low * (1 - frac) + tmpl_high * frac)


def bracket(table: list[dict], arch: str, rounding) -> float:
    """Ranked-weighted gain when every prompt runs at one integer width.

    The per-prompt histogram is not measured, only its mean, and the residency
    step between M=5 and M=6 is the largest in the table. Sending every prompt
    to the floor of its mean is the adverse corner and to the ceiling is the
    favourable one, so the two bound the linear interpolation.
    """
    weighted_switch = weighted_tmpl = 0.0
    for width, weight in RANKED_WIDTH_MIX.values():
        m = max(min(ROUTED_WIDTHS), min(int(rounding(width)), max(ROUTED_WIDTHS)))
        weighted_switch += weight * residency(
            table, shipped_arm(m), arch, "switch", None)
        weighted_tmpl += weight * residency(
            table, shipped_arm(m), arch, "templated", m)
    return weighted_tmpl / weighted_switch - 1.0


def gate(table: list[dict]) -> dict:
    """Ranked-weighted resident-simdgroup gain, and the rung-1 gate verdict."""
    verdict = {}
    for arch in ("g16s", "g17s"):
        per_prompt = {}
        total_weight = sum(w for _, w in RANKED_WIDTH_MIX.values())
        weighted_switch = weighted_tmpl = 0.0
        for prompt, (width, weight) in RANKED_WIDTH_MIX.items():
            switch, tmpl = interpolate(table, arch, width)
            per_prompt[prompt] = {
                "mean_verify_width": width,
                "median_weight": weight,
                "switch_resident_simdgroups": switch,
                "templated_resident_simdgroups": tmpl,
                "gain_fraction": tmpl / switch - 1.0,
                "interpolated": abs(width - round(width)) > 1e-9,
            }
            weighted_switch += weight * switch
            weighted_tmpl += weight * tmpl
        ranked_gain = weighted_tmpl / weighted_switch - 1.0
        local_switch, local_tmpl = interpolate(table, arch, LOCAL_MEAN_WIDTH)
        adverse = bracket(table, arch, math.floor)
        favourable = bracket(table, arch, math.ceil)
        verdict[arch] = {
            "per_prompt": per_prompt,
            "weight_sum": total_weight,
            "ranked_weighted_switch": weighted_switch / total_weight,
            "ranked_weighted_templated": weighted_tmpl / total_weight,
            "ranked_weighted_gain_fraction": ranked_gain,
            "adverse_all_floor_gain_fraction": adverse,
            "favourable_all_ceiling_gain_fraction": favourable,
            "local_fixture_gain_fraction": local_tmpl / local_switch - 1.0,
            "gate_fraction": GATE_FRACTION,
            "passes_gate": ranked_gain >= GATE_FRACTION,
            "passes_gate_on_adverse_corner": adverse >= GATE_FRACTION,
        }
    return verdict


def text_summary(table: list[dict]) -> dict:
    """Per-pipeline machine-text bytes, which is the second claimed channel."""
    out = {}
    for arch in ("g16s", "g17s"):
        for arm in ("replica_no_table", "sumtable"):
            switch = [r for r in table if r["arm"] == arm and r["arch"] == arch
                      and r["variant"] == "switch"]
            tmpl = [r for r in table if r["arm"] == arm and r["arch"] == arch
                    and r["variant"] == "templated"]
            if not switch or not tmpl:
                continue
            out["%s/%s" % (arch, arm)] = {
                "switch_text_bytes": switch[0]["text_bytes"],
                "templated_text_bytes_min": min(r["text_bytes"] for r in tmpl),
                "templated_text_bytes_max": max(r["text_bytes"] for r in tmpl),
                "templated_text_bytes_total": sum(r["text_bytes"] for r in tmpl),
            }
    return out


def log_wandb(result: dict) -> str | None:
    import wandb

    table = result["rows"]
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type="census",
        name="e129-rung1-entry-point-census",
        config={
            "experiment": GROUP,
            "rung": 1,
            "question": ("does templating the Route B QMV entry point on M "
                         "remove the M=5 register maximum from every other "
                         "width, and is the residency change worth anything "
                         "at the ranked widths?"),
            "harness": "local",
            "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
            "timing_valid": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "simdgroup_budget": SIMDGROUP_BUDGET,
            "ranked_width_mix": {k: {"width": w, "weight": g}
                                 for k, (w, g) in RANKED_WIDTH_MIX.items()},
            "gate_fraction": GATE_FRACTION,
            "base_sha": result["base_sha"],
            "git_head": result["git_head"],
            "pr_number": 128,
        })
    columns = ["arm", "arch", "variant", "m", "na", "registers",
               "spill_bytes", "text_bytes", "resident_simdgroups"]
    run.log({"rung1/census": wandb.Table(
        columns=columns,
        data=[[row[c] for c in columns] for row in table])})
    summary = {}
    for arch, cell in result["gate"].items():
        for field in ("ranked_weighted_gain_fraction",
                      "adverse_all_floor_gain_fraction",
                      "favourable_all_ceiling_gain_fraction",
                      "local_fixture_gain_fraction",
                      "passes_gate", "passes_gate_on_adverse_corner"):
            summary["rung1/%s/%s" % (arch, field)] = cell[field]
    run.summary.update(summary)
    run.summary.update({"rung1/text": result["text"]})
    url = run.url
    run.finish()
    return url


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(
                            "research/out/e129-entry-point-census.json"))
    parser.add_argument("--keep", type=pathlib.Path,
                        help="write the reproduced Metal sources here")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    header = e120.swift_literal("qwen35E120QMVHeader")
    arms = {
        "replica_no_table": arm_source(header, table=False),
        "sumtable": arm_source(header, table=True),
    }

    census: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for tag, source in arms.items():
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / ("%s.metal" % tag)).write_text(source)
            census[tag] = e120.census(source, tag, workdir)

    table = rows(census)
    result = {
        "harness": "local",
        "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "git_head": git("rev-parse", "HEAD"),
        "base_sha": git("rev-parse", "HEAD"),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "routed_widths": list(ROUTED_WIDTHS),
        "minimum_table_width": MINIMUM_TABLE_WIDTH,
        "ranked_width_mix": {k: {"mean_verify_width": w, "median_weight": g}
                             for k, (w, g) in RANKED_WIDTH_MIX.items()},
        "rows": table,
        "gate": gate(table),
        "text": text_summary(table),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("%-17s %-5s %-10s %4s %5s %6s %7s %s"
          % ("arm", "arch", "variant", "M", "NA", "regs", "spill", "text B   resident"))
    for row in sorted(table, key=lambda r: (r["arch"], r["arm"], r["variant"],
                                            r["m"] or 0, r["na"] or 0)):
        print("%-17s %-5s %-10s %4s %5s %6s %5s %8s   %s"
              % (row["arm"], row["arch"], row["variant"],
                 row["m"] if row["m"] else "-",
                 row["na"] if row["na"] else "-",
                 row["registers"], row["spill_bytes"], row["text_bytes"],
                 row["resident_simdgroups"]))

    for arch, cell in result["gate"].items():
        print("\n%s ranked-weighted resident simdgroups: switch %.2f -> "
              "templated %.2f, gain %+.2f %%   gate %.0f %%   %s"
              % (arch, cell["ranked_weighted_switch"],
                 cell["ranked_weighted_templated"],
                 100 * cell["ranked_weighted_gain_fraction"],
                 100 * GATE_FRACTION,
                 "PASS" if cell["passes_gate"] else "STOP"))
        for prompt, per in cell["per_prompt"].items():
            print("    %-9s width %.3f  weight %.4f  %.2f -> %.2f  %+.2f %%%s"
                  % (prompt, per["mean_verify_width"], per["median_weight"],
                     per["switch_resident_simdgroups"],
                     per["templated_resident_simdgroups"],
                     100 * per["gain_fraction"],
                     "  (interpolated)" if per["interpolated"] else ""))
        print("    local fixture width %.3f: %+.2f %%"
              % (LOCAL_MEAN_WIDTH, 100 * cell["local_fixture_gain_fraction"]))
        print("    bracket over the unmeasured width histogram: adverse "
              "%+.2f %%, favourable %+.2f %%"
              % (100 * cell["adverse_all_floor_gain_fraction"],
                 100 * cell["favourable_all_ceiling_gain_fraction"]))

    if args.wandb:
        url = log_wandb(result)
        result["wandb_url"] = url
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("\nW&B %s" % url)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
