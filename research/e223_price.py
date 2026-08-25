#!/usr/bin/env python3
"""E223 stage 0.4: desk price of every activation-access variant.

Inputs are all measured, none are assumed:

  * `research/e223-artifacts/e223-census-*.json` gives the real AGX
    machine-instruction slope per column per k-block for the shipped kernel and
    for each variant, on `applegpu_g16s` (this host) and `applegpu_g17s` (the
    ranked arch), plus registers and spill bytes at every staged NA.
  * `research/e221-artifacts/e221-rows-report.json` gives E221's measured
    per-cell NA cost slope in ms/round, the measured price of extra issued
    activation bytes in us per issued MB, and `lane_k_blocks_per_dispatch`.

The price of a variant is then

    relief = (removed instructions / base instructions) * NA-proportional cost
    cost   = extra issued activation bytes * measured us per issued MB
    net    = relief - cost

`relief` uses the op-proportional reading, which is the most generous reading
available: it assumes every removed instruction removes its full share of the
measured NA cost. The opposite reading, in which the kernel is limited by
floating-point issue alone, gives a relief of exactly zero for every variant
here, because no variant removes a floating-point operation. The desk decision
uses the generous reading so that a variant only dies when even its best case
falls under the bar.

harness=local; measurement=static_compile plus E221 replay; no GPU seconds and
no score.
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e223-artifacts"
E221 = HERE / "e221-artifacts/e221-rows-report.json"

# Stage bars from the assignment, in ms/round pooled.
STOP_BAR = 0.3
ADVANCE_BAR = 1.0

# FINDING 573, pooled NA-relief ceiling in ms/round.
NA_RELIEF_CEILING = 23.066

# Widths the scored template instantiations actually reach.
SCORED_NA = (2, 3, 4, 5, 6)

# The NA 2..5 text_bytes fit only measures dynamic instructions per column
# while the backend keeps ONE unroll decision across the whole range. When a
# variant makes the backend roll the k-block body, static code size stops
# tracking dynamic instruction count and the slope is meaningless. Spill code
# inside the fitted range contaminates it the same way.
REGIME_MIN_R2 = 0.995

# RULE 407: `applegpu_g17s` is the ranked arch and decides legality, so it also
# decides pricing. `applegpu_g16s` is reported for local-timing fidelity only.
PRICING_ARCH = "g17s"

# Cells whose QMV dispatches carry more than one draft column. `mlp.down` runs
# one column per dispatch, so it has no NA slope to relieve and never enters
# the pooled number.
POOLED_CELLS = ("gdn.in_proj", "mlp.gate_up")

# Extra activation bytes each variant makes the kernel issue, per lane per
# k-block, as a multiple of NA. The shipped kernel issues 32*NA (16 bfloat16
# values per column) plus 4*NA of chunk sums.
EXTRA_ACTIVATION_BYTES_PER_NA = {
    "base": 0,
    "hoistbase": 0,
    "wideload": 0,
    "narrowload": 0,
    # Both float32 paths read 16 four-byte values per column instead of 16
    # two-byte values.
    "xtf32": 32,
    "xf32flat": 32,
}

# A variant that needs a rebuilt activation slab also needs a fill pass. The
# bracket is FINDING 451's measured standalone `xsumsTable` fill cost, and the
# slab is eight times larger than that table, so the bracket is a floor.
SLAB_FILL_US_FLOOR = {"xtf32": 3.8, "xf32flat": 3.8}

CANDIDATES = ("hoistbase", "wideload", "xtf32", "xf32flat")
CONTROLS = ("narrowload",)


def load_census(variant: str) -> dict:
    path = ARTIFACTS / f"e223-census-{variant}.json"
    if not path.exists():
        raise SystemExit(f"e223_price: missing census {path}")
    return json.loads(path.read_text())


def regime_slope(census: dict, arch: str) -> dict:
    """Machine instructions per column per k-block, NA 2..5 unrolled regime."""
    key = f"agx/{arch}/tbl/text_bytes_by_regime"
    fits = census["fits"]
    if key not in fits:
        key = f"agx/{arch}/table/text_bytes_by_regime"
    return fits[key]


def scored_pressure(census: dict, arch: str) -> dict:
    agx = census["agx"][arch]
    return {
        na: {
            "registers": agx[f"na{na}_tbl"]["registers"],
            "spill_bytes": agx[f"na{na}_tbl"]["spill_bytes"],
            "text_bytes": agx[f"na{na}_tbl"]["text_bytes"],
        }
        for na in SCORED_NA
    }


def main() -> int:
    base = load_census("base")
    lane_k_blocks = base["lane_k_blocks_per_dispatch"]
    na_slope = base["measured_na_slope"]
    byte_price = json.loads(E221.read_text())["issued_byte_conversion"]

    variants = {v: load_census(v) for v in ("base",) + CANDIDATES + CONTROLS}

    report: dict = {
        "experiment": "e223-percolumn-inner-loop",
        "stage": "0.4 desk pricing",
        "harness": "local",
        "measurement": "static_compile + e221_replay",
        "gpu_seconds": 0,
        "official_or_ranked_score": None,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "host_chip": base["host_chip"],
        "local_arch": base["local_arch"],
        "ranked_arch": base["ranked_arch"],
        "source_sha8": base["source_sha8"],
        "na_relief_ceiling_ms_per_round": NA_RELIEF_CEILING,
        "stop_bar_ms_per_round": STOP_BAR,
        "advance_bar_ms_per_round": ADVANCE_BAR,
        "per_column_instructions": {},
        "witness_resolution": {},
        "pricing": {},
        "verdict": {},
    }

    # --- per-column machine-instruction slope, both arches -------------------
    for name, census in variants.items():
        row = {}
        for arch in ("g16s", "g17s"):
            fit = regime_slope(census, arch)
            low = fit["na2_5"]
            row[arch] = {
                "insn_per_na": round(low["instructions_per_na"], 2),
                "bytes_per_na": round(low["slope"], 1),
                "r_squared": round(low["r_squared"], 5),
            }
            row[arch]["scored_pressure"] = scored_pressure(census, arch)
        report["per_column_instructions"][name] = row

    # The narrow-load control bounds how small an effect the witness resolves.
    for arch in ("g16s", "g17s"):
        b = report["per_column_instructions"]["base"][arch]["insn_per_na"]
        c = report["per_column_instructions"]["narrowload"][arch][
            "insn_per_na"]
        report["witness_resolution"][arch] = {
            "base_insn_per_na": b,
            "narrowload_insn_per_na": c,
            "control_delta_insn_per_na": round(c - b, 2),
            "note": (
                "narrowload issues four scalar 2-byte loads where the shipped "
                "kernel issues one 8-byte vector load. The delta is the "
                "witness noise floor for a source-level access-path change."),
        }
        report["witness_resolution"][arch]["identical_text_sha8_at_na"] = [
            na for na in SCORED_NA
            if (variants["base"]["agx"][arch][f"na{na}_tbl"]["text_sha8"]
                == variants["narrowload"]["agx"][arch][f"na{na}_tbl"][
                    "text_sha8"])
        ]

    # --- regime validity -----------------------------------------------------
    for name, census in variants.items():
        row = report["per_column_instructions"][name]
        for arch in ("g16s", "g17s"):
            fit = row[arch]
            spills = [fit["scored_pressure"][na]["spill_bytes"]
                      for na in (2, 3, 4, 5)]
            valid = fit["r_squared"] >= REGIME_MIN_R2 and max(spills) == 0
            fit["regime_valid"] = valid
            fit["spill_bytes_na2_5"] = spills
            if not valid:
                fit["invalid_reason"] = (
                    "backend changed the k-block unroll decision inside "
                    "NA 2..5 (r_squared %.3f), so static code size no longer "
                    "tracks dynamic instructions per column"
                    % fit["r_squared"] if fit["r_squared"] < REGIME_MIN_R2
                    else "spill code inside NA 2..5 contaminates the fit: %s"
                    % spills)
        # NA=2 is the one width where every variant stays in the shipped
        # fully-unrolled zero-spill regime, so it bounds a regime-changed
        # variant even when its slope is unusable.
        row["same_regime_na2_text_delta_bytes"] = {
            arch: (row[arch]["scored_pressure"][2]["text_bytes"]
                   - report["per_column_instructions"]["base"][arch][
                       "scored_pressure"][2]["text_bytes"])
            for arch in ("g16s", "g17s")
        }

    # --- price every candidate ----------------------------------------------
    for name in CANDIDATES:
        base_fit = report["per_column_instructions"]["base"][PRICING_ARCH]
        var_fit = report["per_column_instructions"][name][PRICING_ARCH]
        base_insn = base_fit["insn_per_na"]
        var_insn = var_fit["insn_per_na"]
        if var_fit["regime_valid"]:
            removed = base_insn - var_insn
        else:
            # No measurable per-column reduction. The NA=2 same-regime delta
            # bounds it, and the bound is inside the witness floor.
            removed = 0.0
        frac = removed / base_insn
        extra_per_na = EXTRA_ACTIVATION_BYTES_PER_NA[name]

        arms = {}
        for key, slope in na_slope.items():
            cell = slope["cell"]
            inv = slope["invocations_per_round"]
            price = byte_price[key]
            geom = {r["na"]: r["us_per_issued_mb"]
                    for r in price["geometry_route_rows4_to_rows2"]}
            for na in (4, 5):
                if na not in geom:
                    continue
                relief = frac * slope["rows4_ms_per_round_per_na"] * na
                extra_mb = (extra_per_na * na * lane_k_blocks[cell]) / 2 ** 20
                cost_us = extra_mb * geom[na]
                fill = SLAB_FILL_US_FLOOR.get(name, 0.0)
                cost = (cost_us + fill) * inv / 1000.0
                arms[f"{key}/na{na}"] = {
                    "cell": cell,
                    "na": na,
                    "invocations_per_round": inv,
                    "relief_ms_per_round": round(relief, 4),
                    "extra_issued_mb_per_invocation": round(extra_mb, 2),
                    "us_per_issued_mb": round(geom[na], 5),
                    "slab_fill_us_floor_per_invocation": fill,
                    "cost_ms_per_round": round(cost, 4),
                    "net_ms_per_round": round(relief - cost, 4),
                    "pooled_cell": cell in POOLED_CELLS,
                    # The width route bundles extra activation bytes with
                    # extra arithmetic, so it is the pessimistic price for the
                    # same extra bytes. The geometry route above is the
                    # optimistic one and drives the verdict.
                    "us_per_issued_mb_width_route": round(
                        price["width_route_na4_to_na5_at_rows4"][
                            "us_per_issued_mb"], 5),
                    "net_ms_per_round_width_price": round(
                        relief - (extra_mb * price[
                            "width_route_na4_to_na5_at_rows4"][
                                "us_per_issued_mb"] + fill) * inv / 1000.0, 4),
                }

        pooled = {}
        for groups in ("g1", "g2"):
            for thermal in ("cold", "hot"):
                for na in (4, 5):
                    tot = 0.0
                    parts = {}
                    ok = True
                    for cell in POOLED_CELLS:
                        k = f"{cell}/{groups}/{thermal}/na{na}"
                        if k not in arms:
                            ok = False
                            break
                        parts[cell] = arms[k]["net_ms_per_round"]
                        tot += arms[k]["net_ms_per_round"]
                    if ok:
                        pooled[f"{groups}/{thermal}/na{na}"] = {
                            "net_ms_per_round": round(tot, 4),
                            "by_cell": parts,
                        }

        best = max(pooled.values(), key=lambda v: v["net_ms_per_round"])
        worst = min(pooled.values(), key=lambda v: v["net_ms_per_round"])
        report["pricing"][name] = {
            "pricing_arch": PRICING_ARCH,
            "regime_valid": var_fit["regime_valid"],
            "insn_per_na_base_g17s": base_insn,
            "insn_per_na_variant_g17s": var_insn,
            "removed_insn_per_column_per_kblock": round(removed, 2),
            "removed_fraction": round(frac, 4),
            "extra_activation_bytes_per_lane_per_kblock_per_na": extra_per_na,
            "arms": arms,
            "pooled": pooled,
            "pooled_best_ms_per_round": best["net_ms_per_round"],
            "pooled_worst_ms_per_round": worst["net_ms_per_round"],
        }

    # --- verdicts -----------------------------------------------------------
    for name in CANDIDATES:
        p = report["pricing"][name]
        best = p["pooled_best_ms_per_round"]
        g17 = report["per_column_instructions"][name]["g17s"][
            "scored_pressure"]
        base_g17 = report["per_column_instructions"]["base"]["g17s"][
            "scored_pressure"]
        new_spill = {
            na: [base_g17[na]["spill_bytes"], g17[na]["spill_bytes"]]
            for na in SCORED_NA
            if g17[na]["spill_bytes"] > base_g17[na]["spill_bytes"]
        }
        if best >= ADVANCE_BAR:
            state = "advance_to_stage_1"
        elif best >= STOP_BAR:
            state = "stage_1_screen_only"
        else:
            state = "dead_at_desk"
        report["verdict"][name] = {
            "state": state,
            "pooled_best_ms_per_round": best,
            "new_spill_at_scored_na_g17s": new_spill,
            "share_of_na_relief_ceiling": round(
                best / NA_RELIEF_CEILING, 4),
        }

    report["verdict"]["stage_0_outcome"] = (
        "advance" if any(
            report["verdict"][n]["state"] != "dead_at_desk"
            for n in CANDIDATES) else "stop_and_report")

    out = ARTIFACTS / "e223-desk-pricing.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    # --- console summary ----------------------------------------------------
    print("E223 stage 0 desk pricing  (harness=local, static_compile, "
          "0 GPU s)")
    print()
    print(f"{'variant':11s} {'g17s insn/col':>14s} {'regime':>8s} "
          f"{'d_insn':>7s} {'frac':>7s}  {'pooled net ms/round':>21s}  state")
    for name in ("base",) + CONTROLS + CANDIDATES:
        pc = report["per_column_instructions"][name]
        b = pc[PRICING_ARCH]["insn_per_na"]
        reg = "ok" if pc[PRICING_ARCH]["regime_valid"] else "CHANGED"
        if name in report["pricing"]:
            p = report["pricing"][name]
            v = report["verdict"][name]
            rng = (f"{p['pooled_worst_ms_per_round']:+.3f} .. "
                   f"{p['pooled_best_ms_per_round']:+.3f}")
            print(f"{name:11s} {b:14.1f} {reg:>8s} "
                  f"{p['removed_insn_per_column_per_kblock']:+7.1f} "
                  f"{p['removed_fraction']:+7.3f}  {rng:>21s}  {v['state']}")
        else:
            tag = "control" if name in CONTROLS else "shipped"
            print(f"{name:11s} {b:14.1f} {reg:>8s} {'':>7s} {'':>7s}  "
                  f"{'':>21s}  {tag}")
    print()
    for name in CANDIDATES:
        pc = report["per_column_instructions"][name]
        if not pc[PRICING_ARCH]["regime_valid"]:
            d = pc["same_regime_na2_text_delta_bytes"]
            print(f"  {name}: slope unusable ({pc[PRICING_ARCH]['r_squared']:.3f} "
                  f"R2, spills {pc[PRICING_ARCH]['spill_bytes_na2_5']}). "
                  f"NA=2 same-regime text delta g16s {d['g16s']:+d} B, "
                  f"g17s {d['g17s']:+d} B "
                  f"(= {d['g17s'] / 9.2565:+.1f} instructions for the whole "
                  f"kernel).")
    print()
    for arch in ("g16s", "g17s"):
        w = report["witness_resolution"][arch]
        print(f"  witness floor {arch}: control delta "
              f"{w['control_delta_insn_per_na']:+.2f} insn/column "
              f"({100 * abs(w['control_delta_insn_per_na']) / w['base_insn_per_na']:.1f}% "
              f"of base); identical machine code at NA "
              f"{w['identical_text_sha8_at_na']}")
    print()
    print(f"  stage 0 outcome: {report['verdict']['stage_0_outcome']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
