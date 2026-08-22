#!/usr/bin/env python3
"""E129 — price three-tier templating of the Route B QMV entry point.

    usage: research/e129_entry_point_census.py [--out PATH] [--keep DIR]
                                               [--wandb]

THE QUESTION. The shipped wide-QMV entry point is one pipeline holding a
`switch (qmv_m)` over the seven routed widths. A switch entry point's register
count is the MAXIMUM over its inlined branches, so the widest branch sets the
residency of every width. Each width's own maximum is set by the largest
accumulator the branch instantiates, which is its `IPG`, so the seven widths
collapse into three register tiers and not seven:

    tier 3   M = 3, 6, 9   largest body wide<3>
    tier 4   M = 4, 7, 8   largest body wide<4>
    tier 5   M = 5         largest body wide<5>

Crossed with `USE_TABLE` that is six pipelines. `Qwen35CustomQMV.tablePays`
sends M=3 to the no-table arm and M>=4 to the chunk-sum arm, so only four of
the six are ever instantiated at run time, against two today.

The change is bit-exact by construction: every width keeps the same emitted
body, and only the register union above it changes. That makes it the clean
probe of the campaign constant `c`, the fraction of ranked round time that
moves with QMV residency.

WHAT IS MEASURED. Registers, spill bytes, machine-text bytes and resident
simdgroups for the shared switch and for each tier pipeline, on both
`applegpu_g16s` (this host) and `applegpu_g17s` (the ranked runner).

HOW IT IS MEASURED. `xcrun metal-tt` runs the real AGX backend for a named
architecture on any Mac, wrapped by `research/agx_crossarch.py`. Zero GPU
seconds. The Metal source is lifted out of `Qwen35.swift` and the MLX signature
generation is reproduced by `research/e120_g17s_census.py`, which this module
imports rather than copies, so the switch rows here must reproduce the E120
switch rows exactly. That reproduction is the instrument's own control.

HOW IT IS WEIGHTED. The ranked width histogram is not published. The board
publishes each prompt's mean verify width and its width-1 count, which
identifies a set and not a point (E114). This census therefore weights with the
per-prompt MAXIMUM-ENTROPY histogram on that mean, computed by
`research/e114_width_recovery.maxent`, and not with the local fixture's width
mix. The local mix is reported beside it as a contrast, never as the ranked
prediction.

WHAT IT CANNOT SHOW. A census is a cost observation, never correctness
evidence and never a timing result. Residency is `budget // registers` with the
budgets FITTED in E123, so read the ratio between two variants, not the
absolute count.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e114_width_recovery as e114  # noqa: E402
import e120_g17s_census as e120  # noqa: E402

ARCHS = e120.ARCHS
SIMDGROUP_BUDGET = e120.SIMDGROUP_BUDGET
# `Qwen35CustomQMV.shippedPlan` and `Qwen35CustomQMV.onePass67Plan`. Keep these
# in step with the Swift literals; `--table` selects which one is censused.
PLANS = {
    "shipped": ((3, 3, 4), (4, 4, 4), (5, 5, 4), (6, 3, 4), (7, 4, 4), (8, 4, 4), (9, 3, 4)),
    "onepass67": ((3, 3, 4), (4, 4, 4), (5, 5, 4), (6, 6, 4), (7, 7, 4), (8, 4, 4), (9, 3, 4)),
}

WIDTH_CASES = ()
ROUTED_WIDTHS = ()
# `Qwen35CustomQMV.tier(m:)` is the case's IPG, and `Qwen35CustomQMV.tiers` is
# the sorted set of those values.
TIER_OF: dict[int, int] = {}
TIERS = ()

# `Qwen35CustomQMV.minimumTableWidth`. M=3 runs the no-table replica pipeline,
# M>=4 runs the chunk-sum pipeline.
MINIMUM_TABLE_WIDTH = 4
ARMS = ("replica_no_table", "sumtable")

# The ranked verify-width cap is 8 (`segmentedVerifyDepthCap = 7`), so M=9 is
# unreachable on the board even though the entry point can serve it.
RANKED_WIDTHS = ()


def set_plan(name: str) -> None:
    """Rebind the plan-derived globals to one of `PLANS`."""
    global WIDTH_CASES, ROUTED_WIDTHS, TIER_OF, TIERS, RANKED_WIDTHS
    WIDTH_CASES = PLANS[name]
    ROUTED_WIDTHS = tuple(m for m, _, _ in WIDTH_CASES)
    TIER_OF = {m: ipg for m, ipg, _ in WIDTH_CASES}
    TIERS = tuple(sorted(set(TIER_OF.values())))
    RANKED_WIDTHS = tuple(m for m in ROUTED_WIDTHS if m <= 8)


set_plan("shipped")

# F83 ranked weights: mean verify width and median-sensitivity weight per
# scored prompt. The weights do not sum to one; this module normalises them.
RANKED_WIDTH_MIX = {
    "beagle": (5.382, 0.4862),
    "medicine": (6.256, 0.2508),
    "essays": (6.087, 0.1598),
    "botany": (7.148, 0.0124),
    "republic": (5.989, 0.0100),
}

# Realised verification widths over the 312 rounds of the rung 5e local
# session, for contrast only. M=2 is below `Qwen35CustomQMV.widths`, so it
# never reaches either pipeline.
LOCAL_HISTOGRAM = {2: 4, 4: 16, 5: 20, 6: 20, 7: 12, 8: 240}

# The rung gate: below this ranked-weighted resident-simdgroup gain, close the
# axis and report the census as the result.
GATE_FRACTION = 0.05

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e129-entry-point-templating"


def tier_cases(tier: int) -> tuple:
    """The switch cases `qwen35E120QMVSource(table:tier:)` emits for `tier`."""
    return tuple(c for c in WIDTH_CASES if c[1] == tier)


def arm_source(header: str, table: bool) -> str:
    """One library holding the shared switch plus every tier pipeline."""
    base = ("qwen35_custom_affine4_g64_qmv_wide_sums_v2" if table
            else "qwen35_custom_affine4_g64_qmv_wide_v2")
    inputs = e120.QMV_INPUTS + ([("xsums", "float")] if table else [])
    use_table = [("bool", "USE_TABLE", "true")] if table else None
    parts = [e120.PRELUDE, header, ""]
    parts.append(e120.generate(base, inputs, e120.QMV_OUTPUTS,
                               e120.qmv_body(table), use_table))
    for tier in TIERS:
        parts.append(e120.generate(
            "%s_tier%d" % (base, tier), inputs, e120.QMV_OUTPUTS,
            e120.qmv_body(table, tier_cases(tier)), use_table))
    return "\n".join(parts) + "\n"


def xsums_source() -> str:
    """`qwen35CustomAffine4XSumsKernel`, the chunk-sum fill, lifted verbatim."""
    text = e120.QWEN35.read_text()
    start = text.index('name: "qwen35_custom_affine4_g64_xsums_v1"')
    start = text.index('source: """', start)
    start = text.index("\n", start) + 1
    end = text.index('        """', start)
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in text[start:end].splitlines())
    return e120.PRELUDE + e120.generate(
        "qwen35_custom_affine4_g64_xsums_v1",
        [("x", "bfloat16_t")], [("xsums", "float")], body)


def classify(kernel: str) -> tuple[str, int | None]:
    """`(variant, tier)` for a censused kernel name."""
    if "_tier" in kernel:
        return "tier", int(kernel.rsplit("_tier", 1)[1])
    if kernel.endswith("xsums_v1"):
        return "xsums", None
    return "switch", None


def rows(census: dict) -> list[dict]:
    out = []
    for arm, per_arch in census.items():
        for arch, kernels in per_arch.items():
            for kernel, cell in kernels.items():
                variant, tier = classify(kernel)
                out.append({
                    "arm": arm,
                    "arch": arch.replace("applegpu_", ""),
                    "variant": variant,
                    "tier": tier,
                    "widths": ([m for m, _, _ in tier_cases(tier)]
                               if tier else list(ROUTED_WIDTHS)),
                    "registers": cell["registers"],
                    "spill_bytes": cell["spill_bytes"],
                    "text_bytes": cell["text_bytes"],
                    "resident_simdgroups": cell["resident_simdgroups"],
                })
    return out


def shipped_arm(m: int) -> str:
    """The pipeline the shipped gate runs at width `m`."""
    return "sumtable" if m >= MINIMUM_TABLE_WIDTH else "replica_no_table"


def cell(table: list[dict], arm: str, arch: str, variant: str,
         tier: int | None) -> dict:
    for row in table:
        if (row["arm"] == arm and row["arch"] == arch
                and row["variant"] == variant and row["tier"] == tier):
            return row
    raise SystemExit("no %s %s %s tier=%s row" % (arm, arch, variant, tier))


def residency(table: list[dict], arch: str, m: int, tiered: bool) -> int:
    """Resident simdgroups the QMV dispatch reaches at verify width `m`."""
    arm = shipped_arm(m)
    if tiered:
        return cell(table, arm, arch, "tier", TIER_OF[m])["resident_simdgroups"]
    return cell(table, arm, arch, "switch", None)["resident_simdgroups"]


def ranked_histograms() -> dict[str, dict[int, float]]:
    """Per-prompt max-entropy width histogram, renormalised onto the widths
    that actually reach this entry point.

    `e114.maxent` puts mass on M = 2..8. M=2 is below
    `Qwen35CustomQMV.widths`, so it reaches a different MLX kernel and carries
    no weight here. Dropping it and renormalising raises the reported mean
    slightly; both means are recorded so the shift is visible.
    """
    out = {}
    for prompt, (width, _weight) in RANKED_WIDTH_MIX.items():
        full = e114.maxent(width)
        routed = {m: p for m, p in full.items() if m in RANKED_WIDTHS}
        total = sum(routed.values())
        out[prompt] = {m: p / total for m, p in sorted(routed.items())}
    return out


def weighted(table: list[dict], arch: str, hist: dict[int, float],
             tiered: bool) -> float:
    return sum(p * residency(table, arch, m, tiered) for m, p in hist.items())


def gate(table: list[dict], hists: dict[str, dict[int, float]]) -> dict:
    """Ranked-weighted resident-simdgroup gain, and the gate verdict."""
    verdict = {}
    local = {m: n for m, n in LOCAL_HISTOGRAM.items() if m in ROUTED_WIDTHS}
    local_rounds = sum(local.values())
    local_hist = {m: n / local_rounds for m, n in local.items()}
    for arch in ("g16s", "g17s"):
        per_prompt = {}
        total_weight = sum(w for _, w in RANKED_WIDTH_MIX.values())
        shared_sum = tiered_sum = 0.0
        for prompt, (width, weight) in RANKED_WIDTH_MIX.items():
            hist = hists[prompt]
            shared = weighted(table, arch, hist, False)
            tiered = weighted(table, arch, hist, True)
            per_prompt[prompt] = {
                "board_mean_verify_width": width,
                "routed_mean_verify_width": sum(m * p for m, p in hist.items()),
                "median_weight": weight,
                "maxent_histogram": hist,
                "shared_resident_simdgroups": shared,
                "tiered_resident_simdgroups": tiered,
                "gain_fraction": tiered / shared - 1.0,
            }
            shared_sum += weight * shared
            tiered_sum += weight * tiered
        ranked_gain = tiered_sum / shared_sum - 1.0
        local_shared = weighted(table, arch, local_hist, False)
        local_tiered = weighted(table, arch, local_hist, True)
        verdict[arch] = {
            "per_prompt": per_prompt,
            "weight_sum": total_weight,
            "ranked_weighted_shared": shared_sum / total_weight,
            "ranked_weighted_tiered": tiered_sum / total_weight,
            "ranked_weighted_gain_fraction": ranked_gain,
            "local_fixture_shared": local_shared,
            "local_fixture_tiered": local_tiered,
            "local_fixture_gain_fraction": local_tiered / local_shared - 1.0,
            "local_histogram": local_hist,
            "gate_fraction": GATE_FRACTION,
            "passes_gate": ranked_gain >= GATE_FRACTION,
        }
    return verdict


def per_width(table: list[dict]) -> dict:
    """Residency and register count at each routed width, both designs."""
    out = {}
    for arch in ("g16s", "g17s"):
        rows_out = {}
        for m in ROUTED_WIDTHS:
            arm = shipped_arm(m)
            shared = cell(table, arm, arch, "switch", None)
            tier = cell(table, arm, arch, "tier", TIER_OF[m])
            rows_out[m] = {
                "arm": arm,
                "tier": TIER_OF[m],
                "shared_registers": shared["registers"],
                "shared_spill_bytes": shared["spill_bytes"],
                "shared_resident_simdgroups": shared["resident_simdgroups"],
                "tiered_registers": tier["registers"],
                "tiered_spill_bytes": tier["spill_bytes"],
                "tiered_resident_simdgroups": tier["resident_simdgroups"],
                "gain_fraction": (tier["resident_simdgroups"]
                                  / shared["resident_simdgroups"] - 1.0),
            }
        out[arch] = rows_out
    return out


def pipelines(table: list[dict]) -> dict:
    """Pipeline count and total machine text, the warmup-risk channel.

    `tablePays` fixes which arm each width uses, so the tiered design does not
    instantiate all six pipelines: the no-table arm only ever serves M=3.
    """
    instantiated_shared = []
    instantiated_tiered = []
    for m in ROUTED_WIDTHS:
        arm = shipped_arm(m)
        if (arm, None) not in instantiated_shared:
            instantiated_shared.append((arm, None))
        key = (arm, TIER_OF[m])
        if key not in instantiated_tiered:
            instantiated_tiered.append(key)
    out = {
        "shared_pipelines": [{"arm": a, "tier": t}
                             for a, t in instantiated_shared],
        "tiered_pipelines": [{"arm": a, "tier": t}
                             for a, t in instantiated_tiered],
        "shared_pipeline_count": len(instantiated_shared),
        "tiered_pipeline_count": len(instantiated_tiered),
        "extra_compilations": (len(instantiated_tiered)
                               - len(instantiated_shared)),
    }
    for arch in ("g16s", "g17s"):
        out["%s_shared_text_bytes" % arch] = sum(
            cell(table, a, arch, "switch", t)["text_bytes"]
            for a, t in instantiated_shared)
        out["%s_tiered_text_bytes" % arch] = sum(
            cell(table, a, arch, "tier", t)["text_bytes"]
            for a, t in instantiated_tiered)
    return out


def log_wandb(result: dict) -> str | None:
    import wandb

    table = result["rows"]
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type="census",
        name="e129-tier-entry-point-census",
        config={
            "experiment": GROUP,
            "question": ("does splitting the Route B QMV entry point into one "
                         "pipeline per accumulator tier raise resident "
                         "simdgroups at the ranked widths?"),
            "harness": "local",
            "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
            "timing_valid": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "simdgroup_budget": SIMDGROUP_BUDGET,
            "width_plan": [list(c) for c in WIDTH_CASES],
            "tiers": list(TIERS),
            "ranked_width_mix": {k: {"width": w, "weight": g}
                                 for k, (w, g) in RANKED_WIDTH_MIX.items()},
            "weighting": "e114 per-prompt max-entropy histogram",
            "gate_fraction": GATE_FRACTION,
            "base_sha": result["base_sha"],
            "git_head": result["git_head"],
            "pr_number": 128,
        })
    columns = ["arm", "arch", "variant", "tier", "registers",
               "spill_bytes", "text_bytes", "resident_simdgroups"]
    run.log({"census/rows": wandb.Table(
        columns=columns,
        data=[[row[c] for c in columns] for row in table])})
    summary = {}
    for arch, verdict in result["gate"].items():
        for field in ("ranked_weighted_shared", "ranked_weighted_tiered",
                      "ranked_weighted_gain_fraction",
                      "local_fixture_gain_fraction", "passes_gate"):
            summary["census/%s/%s" % (arch, field)] = verdict[field]
    for key, value in result["pipelines"].items():
        if isinstance(value, (int, float)):
            summary["census/pipelines/%s" % key] = value
    run.summary.update(summary)
    run.summary.update({"census/per_width": result["per_width"],
                        "census/pipelines": result["pipelines"]})
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
    parser.add_argument("--table", choices=sorted(PLANS), default="shipped")
    args = parser.parse_args()
    set_plan(args.table)

    header = e120.swift_literal("qwen35E120QMVHeader")
    arms = {
        "replica_no_table": arm_source(header, table=False),
        "sumtable": arm_source(header, table=True),
        "xsums_fill": xsums_source(),
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
    hists = ranked_histograms()
    result = {
        "harness": "local",
        "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "git_head": git("rev-parse", "HEAD"),
        "base_sha": git("rev-parse", "HEAD"),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "width_plan": [list(c) for c in WIDTH_CASES],
        "tier_of_width": TIER_OF,
        "tiers": list(TIERS),
        "routed_widths": list(ROUTED_WIDTHS),
        "ranked_widths": list(RANKED_WIDTHS),
        "minimum_table_width": MINIMUM_TABLE_WIDTH,
        "ranked_width_mix": {k: {"board_mean_verify_width": w,
                                 "median_weight": g}
                             for k, (w, g) in RANKED_WIDTH_MIX.items()},
        "maxent_histograms": hists,
        "rows": table,
        "per_width": per_width(table),
        "pipelines": pipelines(table),
        "gate": gate(table, hists),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("%-17s %-5s %-7s %5s %6s %6s %8s %s"
          % ("arm", "arch", "variant", "tier", "regs", "spill", "text B",
             "resident"))
    for row in sorted(table, key=lambda r: (r["arch"], r["arm"], r["variant"],
                                            r["tier"] or 0)):
        print("%-17s %-5s %-7s %5s %6s %6s %8s %s"
              % (row["arm"], row["arch"], row["variant"],
                 row["tier"] if row["tier"] else "-",
                 row["registers"], row["spill_bytes"], row["text_bytes"],
                 row["resident_simdgroups"]))

    for arch in ("g16s", "g17s"):
        print("\n%s per width, resident simdgroups (DERIVED, budget // regs)"
              % arch)
        print("   M  arm               tier   shared            tiered"
              "            gain")
        for m, row in result["per_width"][arch].items():
            print("  %2d  %-17s %3d  %3d regs/%3d B %3d  %3d regs/%3d B %3d"
                  "  %+7.2f %%"
                  % (m, row["arm"], row["tier"],
                     row["shared_registers"], row["shared_spill_bytes"],
                     row["shared_resident_simdgroups"],
                     row["tiered_registers"], row["tiered_spill_bytes"],
                     row["tiered_resident_simdgroups"],
                     100 * row["gain_fraction"]))

    for arch, verdict in result["gate"].items():
        print("\n%s ranked-weighted resident simdgroups: shared %.3f -> "
              "tiered %.3f, gain %+.2f %%   gate %.0f %%   %s"
              % (arch, verdict["ranked_weighted_shared"],
                 verdict["ranked_weighted_tiered"],
                 100 * verdict["ranked_weighted_gain_fraction"],
                 100 * GATE_FRACTION,
                 "PASS" if verdict["passes_gate"] else "STOP"))
        for prompt, per in verdict["per_prompt"].items():
            print("    %-9s board width %.3f  routed %.3f  weight %.4f  "
                  "%.3f -> %.3f  %+.2f %%"
                  % (prompt, per["board_mean_verify_width"],
                     per["routed_mean_verify_width"], per["median_weight"],
                     per["shared_resident_simdgroups"],
                     per["tiered_resident_simdgroups"],
                     100 * per["gain_fraction"]))
        print("    local fixture histogram: %.3f -> %.3f  %+.2f %%"
              % (verdict["local_fixture_shared"],
                 verdict["local_fixture_tiered"],
                 100 * verdict["local_fixture_gain_fraction"]))

    pipes = result["pipelines"]
    print("\nwarmup channel: %d instantiated pipelines -> %d, %+d compilations"
          % (pipes["shared_pipeline_count"], pipes["tiered_pipeline_count"],
             pipes["extra_compilations"]))
    for arch in ("g16s", "g17s"):
        print("  %s machine text %d B -> %d B"
              % (arch, pipes["%s_shared_text_bytes" % arch],
                 pipes["%s_tiered_text_bytes" % arch]))

    if args.wandb:
        url = log_wandb(result)
        result["wandb_url"] = url
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("\nW&B %s" % url)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
