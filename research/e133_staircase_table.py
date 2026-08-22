"""F25 item (4): price W1 / W2 / A1 in the register-staircase currency.

FINDING 169 says the wide widths are occupancy-limited, so a deleted
instruction is worth nothing unless it crosses a residency step.  Residency
follows the exact floor law of Rule 89:

    resident simdgroups = floor(REGISTER_BUDGET / registers_per_thread)

This tool reads the compile-only census and reports, per form and per cell,
the register count, the residency it buys, and whether it crosses the step
above the shipped form.  It answers the question in the currency FINDING 169
defines, not in deleted instructions.

Usage:  python3 research/e133_staircase_table.py [CENSUS_JSON]
"""

import json
import sys

BUDGET = {"applegpu_g17s": 3968, "applegpu_g16s": 3072}
CEILING = {"applegpu_g17s": 126, "applegpu_g16s": 96}
FORMS = ["shipped", "w1", "w2", "a1", "w1_a1", "w1_w2_a1"]
CELLS = [("cell_na5_rps4", 5), ("cell_na6_rps4", 6), ("cell_na7_rps4", 7)]

# F83 ranked mass at the widths the one-pass table routes to these cells.
RANKED_MASS = {5: 0.1730, 6: 0.1880, 7: 0.2110}
QMV_SHARE = 0.8735


def residency(regs, arch):
    return BUDGET[arch] // regs


def regs_for_next_step(regs, arch):
    """Fewest registers that buys one more resident simdgroup."""
    target = residency(regs, arch) + 1
    return BUDGET[arch] // target


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "research/e133-artifacts/f23-composed-census.json"
    )
    census = json.load(open(path))
    variants = census["variants"]

    out = {
        "schema_version": 1,
        "tool": "research/e133_staircase_table.py",
        "harness": "compile_only",
        "gpu_used": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "occupancy_rule": "Rule 89 floor law",
        "source_census": path,
        "base_sha": census.get("base_sha"),
        "toolchain": census.get("toolchain"),
        "register_budget": BUDGET,
        "register_ceiling": CEILING,
        "arches": {},
    }

    for arch in ("applegpu_g17s", "applegpu_g16s"):
        rows = []
        print("\n===== %s   budget %d   ceiling %d =====" % (
            arch, BUDGET[arch], CEILING[arch]))
        print("%-4s %-10s %6s %6s %6s %5s %5s %8s %6s  %s" % (
            "NA", "form", "text", "dtext", "AIR", "regs", "sg",
            "need", "spill", "crosses?"))
        for cell_key, na in CELLS:
            base = variants["shipped"]["cells"][cell_key][arch]
            base_regs = base["registers"]
            base_sg = residency(base_regs, arch)
            need = regs_for_next_step(base_regs, arch)
            base_text = base["text_bytes"]
            for form in FORMS:
                c = variants[form]["cells"][cell_key]
                a = c[arch]
                regs = a["registers"]
                sg = residency(regs, arch)
                crosses = sg > base_sg
                row = {
                    "na": na,
                    "form": form,
                    "air_total": c["air"]["total"],
                    "text_bytes": a["text_bytes"],
                    "delta_text_bytes": a["text_bytes"] - base_text,
                    "registers": regs,
                    "delta_registers": regs - base_regs,
                    "resident_simdgroups": sg,
                    "delta_resident_simdgroups": sg - base_sg,
                    "registers_needed_for_next_step": need,
                    "reaches_next_step": crosses,
                    "spill_bytes": a["spill_bytes"],
                    "clamped_at_ceiling": regs >= CEILING[arch],
                }
                rows.append(row)
                print("%-4d %-10s %6d %+6d %6d %5d %5d %8d %6d  %s" % (
                    na, form, a["text_bytes"], row["delta_text_bytes"],
                    c["air"]["total"], regs, sg, need, a["spill_bytes"],
                    "YES +%d sg" % (sg - base_sg) if crosses else "no"))
            print()

        # Value of every form that does cross, priced by ranked mass.
        gain = 0.0
        for r in rows:
            if r["form"] != "shipped" and r["reaches_next_step"]:
                frac = r["delta_resident_simdgroups"] / (
                    r["resident_simdgroups"] - r["delta_resident_simdgroups"])
                gain += frac * RANKED_MASS[r["na"]] * QMV_SHARE
        out["arches"][arch] = {
            "rows": rows,
            "any_form_crosses": any(
                r["reaches_next_step"] for r in rows if r["form"] != "shipped"),
            "ranked_weighted_occupancy_gain_pct": 100.0 * gain,
        }
        print("  any form crosses a step: %s   ranked-weighted occupancy gain "
              "%.4f %%" % (out["arches"][arch]["any_form_crosses"],
                           100.0 * gain))

    g17 = out["arches"]["applegpu_g17s"]
    g16 = out["arches"]["applegpu_g16s"]
    out["verdict"] = {
        "ranked_any_form_crosses": g17["any_form_crosses"],
        "local_any_form_crosses": g16["any_form_crosses"],
        "transfer_hazard": (
            g16["any_form_crosses"] and not g17["any_form_crosses"]),
    }
    print("\nVERDICT %s" % json.dumps(out["verdict"]))

    dest = "research/e133-artifacts/f25-staircase-table.json"
    json.dump(out, open(dest, "w"), indent=2)
    print("wrote %s" % dest)


if __name__ == "__main__":
    main()
