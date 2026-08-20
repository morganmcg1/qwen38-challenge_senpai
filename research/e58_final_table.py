#!/usr/bin/env python3
"""Render the E58 rung-1 deliverable: assignment kernel group x verify width M,
priced against the local leg and both ranked median-pair legs.

Counts come from the census. Prices come from the tax session and the storm, and
are carried as a range because the three methods measure different things.
"""
from __future__ import annotations

import argparse
import json

# harness=ranked, from the reconstructed receipt quoted in the assignment.
RANKED = {
    "beagle": {"leg_ms": 6233.1, "rounds": 107, "ms_per_round": 53.33, "dilution": 0.91552},
    "medicine": {"leg_ms": 5820.7, "rounds": 99, "ms_per_round": 53.48, "dilution": 0.90953},
}
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PERCENT = 0.283

# ns per dispatch. Each entry names what it actually measures.
PRICES = {
    "in_situ_pipelined": (77.2, "marginal end-to-end cost of one more dispatch on the real candidate leg, submission overlapped with GPU work"),
    "census_host_total": (428.2, "total host encode+commit time actually spent per dispatch, whether or not it is on the critical path"),
    "storm_serialised": (940.0, "floor: trivial dispatch with no GPU work to hide behind, host fully exposed"),
}

GROUP_LABEL = {
    "1_quant_matvec": "1 quantized matvec (affine_qmv_fast*)",
    "2_qmm_splitk": "2 quantized matmul / split-k (qmm*)",
    "3_sdpa": "3 SDPA fused + composed fallback members",
    "4_gdn": "4 GDN recurrence + scan JIT",
    "5_norm_rope": "5 normalisation + RoPE",
    "6_elementwise": "6 elementwise / copy / concat / pad / arange / select / compare",
    "7_top2_readout": "7 top-two partial+finalize + vocabulary readout",
}
GROUP_ORDER = list(GROUP_LABEL)


def leg_by_role(report: dict, role: str) -> dict:
    for leg in report["legs"]:
        if leg.get("leg") == role:
            return leg
    raise SystemExit(f"no leg with role {role!r}; saw {[l.get('leg') for l in report['legs']]}")


def width_table(leg: dict) -> tuple[list[str], dict[str, dict[str, float]]]:
    """group -> width -> dispatches per round at that width."""
    gbw = leg["group_by_width"]
    widths = sorted(gbw.keys(), key=lambda w: int(w))
    table: dict[str, dict[str, float]] = {}
    for w in widths:
        rounds = leg["widths"][w]["rounds"]
        for group, count in gbw[w].items():
            table.setdefault(group, {})[w] = count / rounds
    return widths, table


def render(report_path: str, local_ms_per_round: float, head_group_note: str) -> str:
    report = json.load(open(report_path))
    out: list[str] = []

    for role, label in (("serial(depth0)", "serial depth-0 control"), ("candidate(mtp)", "candidate MTP")):
        try:
            leg = leg_by_role(report, role)
        except SystemExit:
            continue
        widths, table = width_table(leg)
        per_round = leg["dispatches_per_round_mean"]
        out.append(f"\n### {label} (harness=local)\n")
        out.append(
            f"rounds={leg['rounds']}  dispatches/round={per_round:.2f}  "
            f"command buffers={leg['commits_in_rounds']}  "
            f"dispatches/buffer={leg['dispatches_per_commit']:.2f}"
        )
        header = ["kernel group"] + [f"M={w}" for w in widths] + [
            "per round",
            "ms/round @77.2ns",
            "ms/round @940ns",
            "% local leg",
            "% ranked beagle",
            "% ranked medicine",
        ]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for group in sorted(table, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99):
            row = table[group]
            total = leg["group_totals"].get(group, 0.0) / leg["rounds"]
            lo_ms = total * PRICES["in_situ_pipelined"][0] / 1e6
            hi_ms = total * PRICES["storm_serialised"][0] / 1e6
            cells = [GROUP_LABEL.get(group, group)] + [f"{row.get(w, 0.0):.1f}" for w in widths] + [
                f"{total:.2f}",
                f"{lo_ms:.4f}",
                f"{hi_ms:.4f}",
                f"{100 * lo_ms / local_ms_per_round:.4f}-{100 * hi_ms / local_ms_per_round:.4f}",
                f"{100 * lo_ms / RANKED['beagle']['ms_per_round']:.4f}-{100 * hi_ms / RANKED['beagle']['ms_per_round']:.4f}",
                f"{100 * lo_ms / RANKED['medicine']['ms_per_round']:.4f}-{100 * hi_ms / RANKED['medicine']['ms_per_round']:.4f}",
            ]
            out.append("| " + " | ".join(cells) + " |")
        lo_ms = per_round * PRICES["in_situ_pipelined"][0] / 1e6
        hi_ms = per_round * PRICES["storm_serialised"][0] / 1e6
        out.append(
            "| **total** | "
            + " | ".join(f"{sum(table[g].get(w, 0.0) for g in table):.1f}" for w in widths)
            + f" | {per_round:.2f} | {lo_ms:.4f} | {hi_ms:.4f} | "
            + f"{100 * lo_ms / local_ms_per_round:.4f}-{100 * hi_ms / local_ms_per_round:.4f} | "
            + f"{100 * lo_ms / RANKED['beagle']['ms_per_round']:.4f}-{100 * hi_ms / RANKED['beagle']['ms_per_round']:.4f} | "
            + f"{100 * lo_ms / RANKED['medicine']['ms_per_round']:.4f}-{100 * hi_ms / RANKED['medicine']['ms_per_round']:.4f} |"
        )
        phases = leg.get("phase_totals") or {}
        if phases:
            out.append("\ngroup 8, proposal head separated from target, by phase:\n")
            out.append("| phase | dispatches/round | share of round | commits/round | host encode+commit ms/round |")
            out.append("|---|---:|---:|---:|---:|")
            for phase, entry in sorted(phases.items()):
                pr = entry["dispatches"] / leg["rounds"]
                cr = entry["commits"] / leg["rounds"]
                host_ms = (entry["dispatch_ns"] + entry["commit_ns"] - entry["clock_bias_ns"]) / 1e6 / leg["rounds"]
                out.append(
                    f"| {phase} | {pr:.2f} | {100 * pr / per_round:.1f} % | {cr:.2f} | {host_ms:.3f} |"
                )

        ht = leg.get("host_timing") or {}
        if ht.get("encode_ns_per_dispatch"):
            out.append(
                f"\nhost timing: encode={ht['encode_ms_per_round']:.3f} ms/round, "
                f"commit={ht['commit_ms_per_round']:.3f} ms/round, "
                f"wall={ht['wall_ms_per_round']:.2f} ms/round (census-on), "
                f"host share={ht['host_share_of_round_percent']:.2f} %"
            )
        out.append(f"\n{head_group_note}")

    out.append("\n### price range, and why the methods disagree\n")
    out.append("| method | ns/dispatch | what it measures |")
    out.append("|---|---:|---|")
    for name, (ns, what) in PRICES.items():
        out.append(f"| {name} | {ns:.1f} | {what} |")
    out.append(
        f"\nranked MDE at 2 sd = {RANKED_MDE_PERCENT} % of score; "
        f"dilution beagle x{RANKED['beagle']['dilution']}, medicine x{RANKED['medicine']['dilution']}, "
        "median pair x0.9125."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--local-ms-per-round", type=float, required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    print(render(args.report, args.local_ms_per_round, args.note))


if __name__ == "__main__":
    main()
