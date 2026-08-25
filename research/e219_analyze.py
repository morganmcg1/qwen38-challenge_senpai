#!/usr/bin/env python3
"""Fit the E219 QMV pass-anatomy census and run its reconciliation gate.

    usage: research/e219_analyze.py SESSION_DIR [SESSION_DIR ...]
                                    [--out research/e219-artifacts/pass-anatomy.json]

Input is one JSON per phase, as written by `Tests/MLXFastTests/
E219QMVPassAnatomyTests.swift` through `research/e219_session.sh`.

Three stages:

1.  Chain slope. Every timed unit was measured at chain lengths C in {1,2,4,8},
    each C repeated once per block. `microseconds = intercept + slope * C`
    regressed over all blocks gives `slope`, the PIPELINED cost of one unit, and
    `intercept`, the `eval()` barrier the unit does not pay in decode. Block
    bootstrap gives the interval.

2.  Composition fit. The pooled design regresses those slopes on

        cost = a * dispatches
             + b * stream_bytes            (packed weights + bf16 scale/bias)
             + c * activation_read_bytes
             + d * output_bytes
             + e * threadgroups

    A single cell shape cannot identify five terms: at fixed (k, n) only NA
    moves, and `stream_bytes` and `threadgroups` are collinear. The k ladder
    breaks stream-vs-threadgroup collinearity, the n ladder breaks
    output-vs-threadgroup, the NA sweep breaks activation-vs-stream, and the
    dispatch split identifies `a` at conserved bytes. The fit is therefore
    POOLED, and per-cell evidence is reported as residuals plus the directly
    measured per-pass cost of each cell.

3.  Reconciliation gate, predeclared in the assignment:
      (a) standalone per-dispatch cost x in-situ dispatch counts must reproduce
          the FINDING 536 m=9 cell totals within +/-15% per cell;
      (b) the standalone G=1 -> G=2 pass ratio must land in the FINDING 559
          in-situ band [1.80, 1.91].

harness=local-microbench. Every number here prices ONE dispatch. None is a
whole-leg or ranked number (RULE 79).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess

import numpy as np

MIB = 1048576.0

# FINDING 536 cell rank, ms per m=9 round, as published.
FINDING_536_CELLS = {
    "mlp.gate_up": 17.580,
    "mlp.down": 9.129,
    "gdn.in_proj": 8.652,
    "fa.qkv": 3.037,
    "gdn.out_proj": 2.205,
    "lm_head": 1.943,
    "fa.o_proj": 0.755,
}
FINDING_536_HEADLINE_MS = 40.051
# FINDING 543: the directly measured, paired price of removing that same third
# pass at m=9. It is 1.93x smaller than the FINDING 536 headline, so the gate is
# reported against BOTH references rather than silently against one.
FINDING_543_THIRD_PASS_MS = 20.774
FINDING_543_CI_MS = (20.396, 21.152)
# FINDING 559 second-pass band: T(G=2) / T(G=1).
FINDING_559_RATIO_BAND = (1.80, 1.91)
FINDING_559_F = {2: 67.724, 3: 69.545, 4: 76.202, 5: 89.468}
# FINDING 487 per-round wide-QMV invocations.
INVOCATIONS = {
    "mlp.gate_up": 64, "mlp.down": 64, "gdn.in_proj": 48,
    "gdn.out_proj": 48, "fa.qkv": 16, "fa.o_proj": 16, "lm_head": 1,
}
# All seven fused cells, (k, n, invocations). Used for the byte-share model and
# for pricing candidate mechanisms over the whole round.
FUSED_CELLS = {
    "mlp.gate_up": (5120, 34816, 64),
    "mlp.down": (17408, 5120, 64),
    "gdn.in_proj": (5120, 16480, 48),
    "gdn.out_proj": (6144, 5120, 48),
    "fa.qkv": (5120, 14336, 16),
    "fa.o_proj": (6144, 5120, 16),
    "lm_head": (5120, 248320, 1),
}
# FINDING 564 pooled census (primary) and the public leg census (sensitivity).
POOLED_WIDTH_SHARE = {
    2: 0.00471, 3: 0.09576, 4: 0.20094, 5: 0.26060,
    6: 0.00785, 7: 0.01099, 8: 0.01570, 9: 0.40345,
}
PUBLIC_WIDTH_SHARE = {
    2: 1 / 73, 5: 12 / 73, 6: 2 / 73, 7: 2 / 73, 8: 1 / 73, 9: 55 / 73,
}
# Shipped staged width plan: m -> (IPG, G).
STAGED_PLAN = {
    2: (2, 1), 3: (3, 1), 4: (4, 1), 5: (5, 1),
    6: (3, 2), 7: (4, 2), 8: (4, 2), 9: (5, 2),
}
GATE_TOLERANCE = 0.15
BOOTSTRAP = 4000
RNG = np.random.default_rng(0xE219)

TERMS = [
    ("a_dispatch_us", "dispatches"),
    ("b_stream_us_per_mib", "stream_mib"),
    ("c_activation_us_per_mib", "activation_mib"),
    ("d_output_us_per_mib", "output_mib"),
    ("e_threadgroup_us_per_1k", "threadgroups_k"),
]


def stream_bytes(k: int, n: int) -> int:
    """affine 4-bit group-64: 32 B packed + 4 B bf16 scale/bias per 64."""
    return n * k * 9 // 16


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def load(dirs: list[pathlib.Path]) -> dict:
    phases = {}
    for d in dirs:
        for path in sorted(d.glob("*.json")):
            blob = json.loads(path.read_text())
            if blob.get("probe") != "e219-qmv-pass-anatomy":
                continue
            phases[blob["phase"]] = blob
    if not phases:
        raise SystemExit("no e219 phase reports found in %s" % dirs)
    return phases


# ---------------------------------------------------------------- chain slope


def chain_slopes(blob: dict) -> dict:
    """Per-unit pipelined cost from the chain-length regression."""
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for row in blob.get("samples", []):
        by_label[row["label"]].append(row)

    out = {}
    for label, rows in by_label.items():
        blocks = sorted({r["block"] for r in rows})
        per_block = {b: [r for r in rows if r["block"] == b] for b in blocks}

        chains = sorted({r["chain"] for r in rows})

        def robust_total(selected: list[int], chain: int) -> float:
            """Lower-trimmed mean over blocks.

            Interference (residency faults, co-tenant dispatch, throttle) only
            ever adds time to a microbenchmark, so the achievable cost sits at
            the bottom of the block distribution. A plain mean or median lets a
            single contaminated block dominate the chain regression.
            """
            vals = sorted(
                r["microseconds"] for b in selected for r in per_block[b]
                if r["chain"] == chain)
            if not vals:
                return float("nan")
            keep = max(1, (len(vals) + 1) // 2)
            return float(np.mean(vals[:keep]))

        def fit(selected: list[int]) -> tuple[float, float]:
            x = np.array(chains, float)
            y = np.array([robust_total(selected, c) for c in chains], float)
            ok = np.isfinite(y)
            if ok.sum() < 2:
                return float("nan"), float("nan")
            design = np.column_stack([np.ones_like(x[ok]), x[ok]])
            coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            return float(coef[1]), float(coef[0])

        slope, intercept = fit(blocks)
        draws = []
        for _ in range(BOOTSTRAP):
            picked = RNG.choice(blocks, size=len(blocks), replace=True).tolist()
            s, _i = fit(picked)
            if np.isfinite(s):
                draws.append(s)
        draws_a = np.array(draws) if draws else np.array([slope])
        fields = {k: v for k, v in rows[0].items()
                  if k not in {"block", "ascending", "position", "chain",
                               "reps", "microseconds",
                               "microseconds_per_unit", "label"}}
        # Per-unit cost at each chain length, for the residual audit.
        by_chain = {c: robust_total(blocks, c) / c for c in chains}
        out[label] = {
            "label": label,
            "slope_us": slope,
            "intercept_us": intercept,
            "slope_ci95": [float(np.percentile(draws_a, 2.5)),
                           float(np.percentile(draws_a, 97.5))],
            "slope_sd": float(np.std(draws_a, ddof=1)) if len(draws_a) > 1
            else float("nan"),
            "blocks": len(blocks),
            "robust_us_per_unit_by_chain": by_chain,
            "fields": fields,
        }
    return out


# --------------------------------------------------------------- design matrix


def predictors(fields: dict) -> dict:
    g = int(fields.get("groups", 1))
    d = int(fields.get("dispatches", 1))
    return {
        "dispatches": float(d),
        "stream_mib": fields["stream_bytes"] * g / MIB,
        "activation_mib": fields["activation_read_bytes"] * g / MIB,
        "output_mib": fields["output_bytes"] * g / MIB,
        "threadgroups_k": fields["threadgroups_per_pass"] * g / 1000.0,
    }


def composition_fit(units: list[dict]) -> dict:
    rows, response, weights, labels = [], [], [], []
    for unit in units:
        f = unit["fields"]
        if not f.get("cold", True):
            continue
        p = predictors(f)
        rows.append([p[col] for _name, col in TERMS])
        response.append(unit["slope_us"])
        sd = unit["slope_sd"]
        weights.append(1.0 / max(sd, 1e-6) ** 2 if np.isfinite(sd) else 1.0)
        labels.append(unit["label"])

    design = np.array(rows, float)
    y = np.array(response, float)
    w = np.array(weights, float)
    rank = int(np.linalg.matrix_rank(design))

    def wls(mask: np.ndarray) -> np.ndarray:
        sw = np.sqrt(w[mask])
        coef, *_ = np.linalg.lstsq(
            design[mask] * sw[:, None], y[mask] * sw, rcond=None)
        return coef

    full = np.ones(len(y), bool)
    coef = wls(full)
    draws = []
    for _ in range(BOOTSTRAP // 4):
        idx = RNG.choice(len(y), size=len(y), replace=True)
        mask = np.zeros(len(y), bool)
        for i in idx:
            mask[i] = True
        try:
            draws.append(wls(mask))
        except np.linalg.LinAlgError:
            continue
    draws_a = np.array(draws)

    predicted = design @ coef
    resid = y - predicted
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    ss_res = float(np.sum(w * resid ** 2))

    coefficients = {}
    for i, (name, col) in enumerate(TERMS):
        lo, hi = (float(np.percentile(draws_a[:, i], 2.5)),
                  float(np.percentile(draws_a[:, i], 97.5))) \
            if len(draws_a) else (float("nan"), float("nan"))
        entry = {
            "value": float(coef[i]), "ci95": [lo, hi], "column": col,
            "resolved_away_from_zero": bool(np.isfinite(lo) and lo * hi > 0),
        }
        if col.endswith("_mib"):
            entry["implied_gib_per_s"] = (
                1e6 / coef[i] / MIB * 1.024 ** 0 if coef[i] > 0 else None)
            entry["implied_gb_per_s"] = (
                MIB / (coef[i] * 1e-6) / 1e9 if coef[i] > 0 else None)
        coefficients[name] = entry

    return {
        "coefficients": coefficients,
        "design_rank": rank,
        "design_columns": len(TERMS),
        "identifiable": rank == len(TERMS),
        "n_units": int(len(y)),
        "weighted_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "residuals": [
            {"label": labels[i], "measured_us": float(y[i]),
             "predicted_us": float(predicted[i]),
             "residual_us": float(resid[i]),
             "residual_pct": float(100.0 * resid[i] / y[i]) if y[i] else None}
            for i in range(len(y))
        ],
    }


# ------------------------------------------------------------------- the gate


def reconciliation(units: dict) -> dict:
    """Gate (a): per-pass cost x invocations vs FINDING 536 / 543.

    The FINDING 536 cell numbers price ONE additional weight pass, so the
    standalone analogue is the MARGINAL cost of the second pass, not the whole
    dispatch. Both readings are reported.
    """
    by_cell: dict[str, dict] = collections.defaultdict(dict)
    for unit in units.values():
        f = unit["fields"]
        if not f.get("cold", True) or int(f.get("dispatches", 1)) != 1:
            continue
        cell = f.get("cell")
        if cell not in INVOCATIONS:
            continue
        by_cell[cell][(int(f["na"]), int(f["groups"]))] = unit

    # Byte-share allocation of a whole-round pass price, shared by both refs.
    total_stream = sum(
        stream_bytes(k, n) * inv for k, n, inv in FUSED_CELLS.values())
    share = {
        name: stream_bytes(k, n) * inv / total_stream
        for name, (k, n, inv) in FUSED_CELLS.items()
    }

    cells = {}
    for cell, arms in sorted(by_cell.items()):
        inv = INVOCATIONS[cell]
        entry = {"invocations_per_round": inv,
                 "byte_share_of_one_pass": share.get(cell)}
        for na in sorted({na for na, _g in arms}):
            g1 = arms.get((na, 1))
            g2 = arms.get((na, 2))
            g3 = arms.get((na, 3))
            arm = {"g1_us": g1["slope_us"] if g1 else None,
                   "g2_us": g2["slope_us"] if g2 else None,
                   "g3_us": g3["slope_us"] if g3 else None}
            if g1 and g2:
                arm["ratio_g2_over_g1"] = g2["slope_us"] / g1["slope_us"]
                arm["marginal_pass_us"] = g2["slope_us"] - g1["slope_us"]
                arm["marginal_pass_ms_per_round"] = (
                    (g2["slope_us"] - g1["slope_us"]) * inv / 1000.0)
                arm["whole_dispatch_ms_per_round"] = (
                    g1["slope_us"] * inv / 1000.0)
                arm["in_finding_559_band"] = bool(
                    FINDING_559_RATIO_BAND[0]
                    <= arm["ratio_g2_over_g1"]
                    <= FINDING_559_RATIO_BAND[1])
            if g2 and g3:
                arm["ratio_g3_over_g2"] = g3["slope_us"] / g2["slope_us"]
                arm["third_pass_us"] = g3["slope_us"] - g2["slope_us"]
                arm["third_pass_ms_per_round"] = (
                    (g3["slope_us"] - g2["slope_us"]) * inv / 1000.0)
            entry["na%d" % na] = arm
        cells[cell] = entry

    # Gate (a) is scored at the scored m=9 geometry, NA = 5.
    checks = []
    for cell, entry in cells.items():
        arm = entry.get("na5", {})
        published = FINDING_536_CELLS.get(cell)
        if published is None or arm.get("marginal_pass_ms_per_round") is None:
            continue
        measured = arm["marginal_pass_ms_per_round"]
        alt = FINDING_543_THIRD_PASS_MS * entry["byte_share_of_one_pass"]
        checks.append({
            "cell": cell,
            "measured_marginal_pass_ms_per_round": measured,
            "finding_536_published_ms": published,
            "ratio_vs_536": measured / published,
            "within_15pct_of_536": abs(measured / published - 1.0)
            <= GATE_TOLERANCE,
            "finding_543_byte_share_ms": alt,
            "ratio_vs_543": measured / alt if alt else None,
            "within_15pct_of_543": abs(measured / alt - 1.0)
            <= GATE_TOLERANCE if alt else None,
        })

    ratios = [
        arm["ratio_g2_over_g1"]
        for entry in cells.values() for key, arm in entry.items()
        if key.startswith("na") and isinstance(arm, dict)
        and arm.get("ratio_g2_over_g1") is not None
    ]
    return {
        "cells": cells,
        "gate_a_cell_checks": checks,
        "gate_a_pass_vs_536": bool(checks) and all(
            c["within_15pct_of_536"] for c in checks),
        "gate_a_pass_vs_543": bool(checks) and all(
            bool(c["within_15pct_of_543"]) for c in checks),
        "gate_b_ratios": ratios,
        "gate_b_band": list(FINDING_559_RATIO_BAND),
        "gate_b_pass": bool(ratios) and all(
            FINDING_559_RATIO_BAND[0] <= r <= FINDING_559_RATIO_BAND[1]
            for r in ratios),
        "reference_conflict": {
            "note":
                "FINDING 536 prices the m=9 third pass at %.3f ms/round while "
                "FINDING 543 measured removing that same pass at %.3f ms/round "
                "(CI %s). The two references disagree by %.2fx, so a single "
                "+/-15%% gate cannot be satisfied against both."
                % (FINDING_536_HEADLINE_MS, FINDING_543_THIRD_PASS_MS,
                   FINDING_543_CI_MS,
                   FINDING_536_HEADLINE_MS / FINDING_543_THIRD_PASS_MS),
            "finding_536_headline_ms": FINDING_536_HEADLINE_MS,
            "finding_536_cell_sum_ms": sum(FINDING_536_CELLS.values()),
            "finding_543_measured_ms": FINDING_543_THIRD_PASS_MS,
        },
    }


# --------------------------------------------------------- hot vs cold, bytes


def cache_share(units: dict) -> list[dict]:
    pairs = []
    index = {}
    for unit in units.values():
        f = unit["fields"]
        if "cell" not in f or "na" not in f:
            continue
        key = (f["cell"], f["na"], f["groups"], bool(f.get("cold", True)))
        index[key] = unit
    for (cell, na, g, cold), unit in sorted(index.items()):
        if not cold:
            continue
        hot = index.get((cell, na, g, False))
        if hot is None:
            continue
        pairs.append({
            "cell": cell, "na": na, "groups": g,
            "cold_us": unit["slope_us"], "hot_us": hot["slope_us"],
            "hot_minus_cold_pct":
                100.0 * (hot["slope_us"] - unit["slope_us"]) / unit["slope_us"],
            "cold_implied_gb_per_s":
                unit["fields"]["stream_bytes"] * g / (unit["slope_us"] * 1e-6)
                / 1e9,
            "hot_implied_gb_per_s":
                hot["fields"]["stream_bytes"] * g / (hot["slope_us"] * 1e-6)
                / 1e9,
            "replicas": unit["fields"].get("replicas"),
            "replica_set_bytes": unit["fields"].get("replica_set_bytes"),
        })
    return pairs


def bytes_accounting(units: dict) -> list[dict]:
    seen = {}
    for unit in units.values():
        f = unit["fields"]
        if "cell" not in f or not f.get("cold", True):
            continue
        if int(f.get("dispatches", 1)) != 1:
            continue
        key = (f["cell"], f["na"], f["groups"])
        seen[key] = (unit, f)
    table = []
    for (cell, na, g), (unit, f) in sorted(seen.items()):
        stream = f["stream_bytes"] * g
        table.append({
            "cell": cell, "na": na, "groups": g,
            "k": f["k"], "n": f["n"],
            "slope_us": unit["slope_us"],
            "weight_bytes": f["weight_bytes"] * g,
            "scale_bias_bytes": f["scale_bias_bytes"] * g,
            "scale_bias_share_of_stream":
                f["scale_bias_bytes"] / (f["stream_bytes"] or 1),
            "activation_unique_bytes": f["activation_unique_bytes"] * g,
            "activation_read_bytes": f["activation_read_bytes"] * g,
            "output_bytes": f["output_bytes"] * g,
            "xsums_read_bytes": f["xsums_read_bytes"] * g,
            "threadgroups": f["threadgroups_per_pass"] * g,
            "implied_stream_gb_per_s": stream / (unit["slope_us"] * 1e-6) / 1e9,
            "implied_all_traffic_gb_per_s":
                (stream + f["activation_read_bytes"] * g
                 + f["output_bytes"] * g + f["xsums_read_bytes"] * g)
                / (unit["slope_us"] * 1e-6) / 1e9,
        })
    return table


# --------------------------------------------------------- mechanism pricing


def round_pass_cost(coef: dict, na: int, groups: int) -> float:
    """Modelled ms per decode round for every fused cell at (NA, G)."""
    total_us = 0.0
    for _name, (k, n, inv) in FUSED_CELLS.items():
        fields = {
            "groups": groups, "dispatches": 1,
            "stream_bytes": stream_bytes(k, n),
            "activation_read_bytes": (n // 8) * 2 * na * k * 2,
            "output_bytes": na * n * 2,
            "threadgroups_per_pass": n // 8,
        }
        p = predictors(fields)
        total_us += inv * sum(
            coef[name]["value"] * p[col] for name, col in TERMS)
    return total_us / 1000.0


def price_mechanisms(coef: dict, recon: dict) -> list[dict]:
    """Price candidate mechanisms under the fitted coefficients.

    Every entry states which fitted term it spends and what it assumes. These
    are CEILINGS from the census, not measured mechanism results.
    """
    def modelled_round(na: int, groups: int) -> float:
        return round_pass_cost(coef, na, groups)

    def pooled(fn) -> float:
        """Round-weighted ms/round over the FINDING 564 pooled census."""
        total = 0.0
        for m, w in POOLED_WIDTH_SHARE.items():
            ipg, g = STAGED_PLAN[m]
            total += w * fn(ipg, g)
        return total

    def public(fn) -> float:
        total = 0.0
        norm = sum(PUBLIC_WIDTH_SHARE.values())
        for m, w in PUBLIC_WIDTH_SHARE.items():
            ipg, g = STAGED_PLAN[m]
            total += (w / norm) * fn(ipg, g)
        return total

    a = coef["a_dispatch_us"]["value"]
    b = coef["b_stream_us_per_mib"]["value"]
    d_out = coef["d_output_us_per_mib"]["value"]
    e_tg = coef["e_threadgroup_us_per_1k"]["value"]

    out = []

    # 1. Remove the bf16 scale/bias interleave cost from the stream: pack the
    #    group metadata so it rides in the same cache lines as its nibbles.
    #    Ceiling = the whole scale/bias share of the b term.
    def metadata(ipg: int, g: int) -> float:
        us = 0.0
        for _n, (k, n, inv) in FUSED_CELLS.items():
            meta_mib = n * (k // 64) * 4 * g / MIB
            us += inv * b * meta_mib
        return us / 1000.0
    out.append({
        "mechanism": "transform-side scale/bias interleave (b term)",
        "term": "b_stream_us_per_mib",
        "assumption":
            "the bf16 scale and bias stream is 1/9 of device bytes; a layout "
            "that co-locates each group's 4 metadata bytes with its 32 packed "
            "bytes removes at most that share of the fitted stream cost",
        "pooled_ms_per_round": pooled(metadata),
        "public_ms_per_round": public(metadata),
    })

    # 2. QMV dispatch fusion: cells that share one activation read are issued as
    #    one dispatch. Ceiling = the a term on the removed dispatches. The
    #    fusable pairs in one round are (gdn.in_proj + fa.qkv share x) and the
    #    two o_proj cells; conservatively price removing one dispatch per layer.
    def fusion(ipg: int, g: int) -> float:
        removed = 64  # one dispatch per transformer layer
        return removed * a / 1000.0
    out.append({
        "mechanism": "QMV dispatch fusion (a term)",
        "term": "a_dispatch_us",
        "assumption":
            "one dispatch per layer is removed by fusing cells that already "
            "share an activation read; no byte term changes",
        "pooled_ms_per_round": pooled(fusion),
        "public_ms_per_round": public(fusion),
    })

    # 3. Output staging: write bf16 results through a wider per-simdgroup store
    #    instead of one lane-0 scalar per (row, column). Ceiling = the d term.
    def output(ipg: int, g: int) -> float:
        us = 0.0
        for _n, (k, n, inv) in FUSED_CELLS.items():
            us += inv * d_out * (ipg * n * 2 * g) / MIB
        return us / 1000.0
    out.append({
        "mechanism": "output store coalescing (d term)",
        "term": "d_output_us_per_mib",
        "assumption":
            "the fitted output cost is removable in full by a coalesced store; "
            "an upper bound, since some of it is unavoidable write traffic",
        "pooled_ms_per_round": pooled(output),
        "public_ms_per_round": public(output),
    })

    # 4. Geometry: 8 output rows per threadgroup is a compile-time choice.
    #    Doubling rows per threadgroup halves the threadgroup count at the same
    #    bytes. Ceiling = half the e term.
    def geometry(ipg: int, g: int) -> float:
        us = 0.0
        for _n, (k, n, inv) in FUSED_CELLS.items():
            us += inv * e_tg * ((n // 8) * g / 1000.0) * 0.5
        return us / 1000.0
    out.append({
        "mechanism": "wider output-row slice per threadgroup (e term)",
        "term": "e_threadgroup_us_per_1k",
        "assumption":
            "rows_per_simd 4 -> 8 halves the threadgroup count at conserved "
            "bytes; the FINDING 549 register grid must be re-proved before "
            "this is implementable",
        "pooled_ms_per_round": pooled(geometry),
        "public_ms_per_round": public(geometry),
    })

    # 5. Remove the second pass entirely (single-pass at NA = 9/10). This is the
    #    already-measured E208 axis, priced here as a model cross-check.
    def single_pass(ipg: int, g: int) -> float:
        if g < 2:
            return 0.0
        return modelled_round(ipg, g) - modelled_round(ipg * g, 1)
    out.append({
        "mechanism": "single-pass at the G=2 widths (model cross-check)",
        "term": "all",
        "assumption":
            "extrapolates the fitted model to NA > 5, which FINDING 549 marks "
            "register-illegal at rows=4. Reported ONLY as a cross-check "
            "against the measured FINDING 543 value, never as a candidate.",
        "pooled_ms_per_round": pooled(single_pass),
        "public_ms_per_round": public(single_pass),
    })

    for entry in out:
        entry["clears_1ms_bar_pooled"] = entry["pooled_ms_per_round"] >= 1.0
    out.sort(key=lambda e: -e["pooled_ms_per_round"])
    return out


# -------------------------------------------------------------------- driver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=pathlib.Path("research/e219-artifacts/pass-anatomy.json"))
    args = parser.parse_args()

    phases = load(args.sessions)
    units: dict[str, dict] = {}
    phase_meta = {}
    for name, blob in phases.items():
        phase_meta[name] = {
            "deliverable": blob.get("deliverable"),
            "blocks": blob.get("blocks"),
            "chains": blob.get("chains"),
            "gpu_temperature_c": blob.get("gpu_temperature_c"),
            "replica_rings": blob.get("replica_rings"),
            "harness": blob.get("harness"),
            "cool_gate_passed_real_gate":
                blob.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": blob.get("gate_qualified_for_timing"),
        }
        if name == "sanity":
            phase_meta[name]["comparisons"] = blob.get("comparisons")
            continue
        for label, unit in chain_slopes(blob).items():
            unit["phase"] = name
            units["%s/%s" % (name, label)] = unit

    cold_units = [u for u in units.values()
                  if u["fields"].get("cold", True)]
    fit = composition_fit(cold_units) if cold_units else {}
    recon = reconciliation(units) if units else {}

    report = {
        "probe": "e219-qmv-pass-anatomy",
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "identity": {
            "base_sha": git("rev-parse", "HEAD"),
            "campaign_base_sha": "66af0086288cda7f676935a512d8c9c1e4e5e17f",
            "host": "aws-mac ec2 Apple M4 Pro, 48 GiB",
            "sessions": [str(p) for p in args.sessions],
            "reference_source":
                "self-generated random affine-4/group-64 cells at scored "
                "shapes; no model checkpoint is loaded",
        },
        "phases": phase_meta,
        "units": sorted(units.values(), key=lambda u: u["label"]),
        "composition_fit": fit,
        "reconciliation": recon,
        "hot_vs_cold": cache_share(units),
        "bytes_accounting": bytes_accounting(units),
    }
    if fit.get("coefficients"):
        report["priced_shortlist"] = price_mechanisms(fit["coefficients"], recon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("wrote %s" % args.out)

    if fit:
        print("\ncomposition fit (rank %s/%s, weighted R2 %.4f, n=%s):"
              % (fit["design_rank"], fit["design_columns"],
                 fit["weighted_r2"], fit["n_units"]))
        for name, entry in fit["coefficients"].items():
            print("  %-28s %12.6g  CI95 [%.6g, %.6g]%s"
                  % (name, entry["value"], entry["ci95"][0], entry["ci95"][1],
                     "" if entry["resolved_away_from_zero"] else "  (spans 0)"))
    if recon:
        print("\nreconciliation gate:")
        print("  gate A vs FINDING 536: %s" % recon["gate_a_pass_vs_536"])
        print("  gate A vs FINDING 543: %s" % recon["gate_a_pass_vs_543"])
        print("  gate B (G2/G1 in %s): %s"
              % (recon["gate_b_band"], recon["gate_b_pass"]))
        for check in recon["gate_a_cell_checks"]:
            print("    %-14s measured %8.3f ms  536 %7.3f (%.2fx)  "
                  "543-share %7.3f (%.2fx)"
                  % (check["cell"],
                     check["measured_marginal_pass_ms_per_round"],
                     check["finding_536_published_ms"], check["ratio_vs_536"],
                     check["finding_543_byte_share_ms"],
                     check["ratio_vs_543"] or float("nan")))
    if report.get("priced_shortlist"):
        print("\npriced shortlist (pooled census, ms/round):")
        for entry in report["priced_shortlist"]:
            print("  %-52s %8.3f  public %8.3f  %s"
                  % (entry["mechanism"], entry["pooled_ms_per_round"],
                     entry["public_ms_per_round"],
                     "CLEARS" if entry["clears_1ms_bar_pooled"] else "below"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
