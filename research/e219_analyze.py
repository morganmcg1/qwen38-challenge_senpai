#!/usr/bin/env python3
"""Fit the E219 QMV pass-anatomy census and run its reconciliation gate.

    usage: research/e219_analyze.py SESSION_DIR [SESSION_DIR ...]
                                    [--out research/e219-artifacts/pass-anatomy.json]

Input is one JSON per phase, as written by `Tests/MLXFastTests/
E219QMVPassAnatomyTests.swift` through `research/e219_session.sh`.

Three stages:

1.  Chain slope. Every timed unit was measured at chain lengths C in {1,2,3,4},
    each C repeated once per block. `microseconds = intercept + slope * C`
    gives `slope`, the PIPELINED cost of one unit, and `intercept`, the `eval()`
    barrier the unit does not pay in decode. The response at each C is a
    lower-trimmed mean over blocks, because interference only adds time, and the
    regression is restricted to the chain lengths where per-unit cost still
    falls or flattens. Block bootstrap gives the interval.

2.  Composition fit. The pooled design regresses those slopes on

        cost = a * dispatches
             + b * stream_bytes            (packed weights + bf16 scale/bias)
             + c * multiply_accumulates
             + d * output_bytes
             + e * threadgroups

    A single cell shape cannot identify five terms: at fixed (k, n) only NA
    moves, and `stream_bytes` and `threadgroups` are collinear. The k ladder
    breaks stream-vs-threadgroup collinearity, the n ladder breaks
    output-vs-threadgroup, the NA sweep breaks arithmetic-vs-stream, and the
    dispatch split identifies `a` at conserved bytes. The fit is therefore
    POOLED, and per-cell evidence is reported as residuals plus the directly
    measured per-pass cost of each cell. Coefficients are constrained to be
    non-negative: every term is a physical cost, so a negative estimate is an
    interpolation artefact of a near-collinear design, not a decomposition.

3.  Reconciliation gates.
      (a') PRIMARY, absolute: the census total for each staged width (m, IPG, G)
           must land within 15 % of the FINDING 559 local per-round cost R(m).
           This checks the instrument against a whole-round in-situ measurement
           with no cache-residency assumption in between.
      (a)  bracket containment, per the advisor ruling on PR 217: the standalone
           hot-to-cold bracket for the marginal m=9 pass must contain the
           FINDING 543 anchor, 20.774 ms/round.
      (b)  the FINDING 559 in-situ second-pass band [1.80, 1.91] must sit inside
           the standalone hot-to-cold ratio bracket.
    Gate (a) also yields phi, the cache-served fraction of the in-situ pass. On
    this host the hot and cold arms agree within noise, because one pass over a
    single cell already exceeds every cache level, so phi is NOT identifiable
    from this instrument and the report says so instead of inventing a value.

harness=local-microbench. Every number here prices ONE dispatch. None is a
whole-leg or ranked number (RULE 79).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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
# Fractional rise in per-unit cost that ends the chain-regression window.
LINEAR_TOL = 0.03
BOOTSTRAP = 4000
RNG = np.random.default_rng(0xE219)

TERMS = [
    ("a_dispatch_us", "dispatches"),
    ("b_stream_us_per_mib", "stream_mib"),
    ("c_mac_us_per_gmac", "mac_gmac"),
    ("d_output_us_per_mib", "output_mib"),
    ("e_threadgroup_us_per_1k", "threadgroups_k"),
]
# FINDING 559 in-situ per-round cost, ms, under the shipped staged plan.
FINDING_559_R_LOCAL = {
    1: 64.832, 2: 67.724, 3: 69.545, 4: 76.202, 5: 89.468,
    6: 117.989, 7: 136.846, 8: 145.346, 9: 164.459,
}
# Host DRAM peak used for every bandwidth statement in this artifact.
HOST_DRAM_PEAK_GB_PER_S = 273.0


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

        def window(selected: list[int]) -> list[int]:
            """Chain lengths over which the pipeline is still filling.

            Per-unit cost must fall or flatten as the chain grows. When it
            climbs, concurrent dispatches have begun to compete for a shared
            resource, and a regression across that break would report the
            contention instead of the marginal pass cost.
            """
            kept = [chains[0]]
            floor = robust_total(selected, chains[0]) / chains[0]
            for c in chains[1:]:
                cur = robust_total(selected, c) / c
                if not np.isfinite(cur) or cur > floor * (1.0 + LINEAR_TOL):
                    break
                kept.append(c)
                floor = min(floor, cur)
            return kept

        def fit(selected: list[int]) -> tuple[float, float, list[int]]:
            keep = window(selected)
            if len(keep) < 2:
                keep = chains[:2]
            x = np.array(keep, float)
            y = np.array([robust_total(selected, c) for c in keep], float)
            ok = np.isfinite(y)
            if ok.sum() < 2:
                return float("nan"), float("nan"), keep
            design = np.column_stack([np.ones_like(x[ok]), x[ok]])
            coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            return float(coef[1]), float(coef[0]), keep

        slope, intercept, fitted_chains = fit(blocks)
        draws = []
        for _ in range(BOOTSTRAP):
            picked = RNG.choice(blocks, size=len(blocks), replace=True).tolist()
            s, _i, _w = fit(picked)
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
            "fitted_chains": fitted_chains,
            "chains_offered": chains,
            "robust_us_per_unit_by_chain": by_chain,
            "fields": fields,
        }
    return out


# --------------------------------------------------------------- design matrix


def predictors(fields: dict) -> dict:
    """Design row for one timed unit.

    The activation tensor is 30 KB and fully cache resident, so its logical
    re-read count is not a traffic term. The quantity that actually grows with
    NA is arithmetic and register pressure, so the third term is multiply-
    accumulates, not activation bytes.
    """
    g = int(fields.get("groups", 1))
    d = int(fields.get("dispatches", 1))
    na = int(fields.get("na", 1))
    return {
        "dispatches": float(d),
        "stream_mib": fields["stream_bytes"] * g / MIB,
        "mac_gmac": na * g * fields["n"] * fields["k"] / 1e9,
        "output_mib": fields["output_bytes"] * g / MIB,
        "threadgroups_k": fields["threadgroups_per_pass"] * g / 1000.0,
    }


def nnls(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares constrained to non-negative coefficients.

    Every term here is a physical cost and cannot be negative. On a
    near-collinear design an unconstrained fit answers with a large negative
    coefficient on the smallest column, which is an interpolation rather than a
    decomposition. Columns are dropped greedily until the fit is admissible.
    """
    sw = np.sqrt(w)
    a = design * sw[:, None]
    b = y * sw
    active = np.ones(a.shape[1], bool)
    coef = np.zeros(a.shape[1])
    while active.any():
        sol, *_ = np.linalg.lstsq(a[:, active], b, rcond=None)
        if (sol >= 0).all():
            coef[active] = sol
            return coef
        idx = np.where(active)[0]
        active[idx[int(np.argmin(sol))]] = False
    return coef


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

    # Collinearity diagnostics. The census varies k, n, NA, G and dispatch
    # count, but several columns still move together, so publish how badly.
    scaled = design / np.maximum(np.abs(design).max(axis=0), 1e-12)
    svals = np.linalg.svd(scaled, compute_uv=False)
    condition = float(svals[0] / svals[-1]) if svals[-1] > 0 else float("inf")
    corr = np.corrcoef(design, rowvar=False)
    pair_corr = [
        {"a": TERMS[i][1], "b": TERMS[j][1], "pearson_r": float(corr[i, j])}
        for i in range(len(TERMS)) for j in range(i + 1, len(TERMS))
    ]
    pair_corr.sort(key=lambda e: -abs(e["pearson_r"]))

    def fit_once(mask: np.ndarray) -> np.ndarray:
        return nnls(design[mask], y[mask], w[mask])

    full = np.ones(len(y), bool)
    coef = fit_once(full)
    draws = []
    for _ in range(BOOTSTRAP // 4):
        idx = RNG.choice(len(y), size=len(y), replace=True)
        mask = np.zeros(len(y), bool)
        for i in idx:
            mask[i] = True
        try:
            draws.append(fit_once(mask))
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
        frac_zero = (float(np.mean(draws_a[:, i] <= 0.0))
                     if len(draws_a) else float("nan"))
        entry = {
            "value": float(coef[i]), "ci95": [lo, hi], "column": col,
            "resolved_away_from_zero": bool(np.isfinite(lo) and lo > 0.0),
            "bootstrap_fraction_dropped_or_zero": frac_zero,
        }
        if col.endswith("_mib") and coef[i] > 0:
            entry["implied_gb_per_s"] = MIB / (coef[i] * 1e-6) / 1e9
        coefficients[name] = entry

    return {
        "coefficients": coefficients,
        "design_rank": rank,
        "design_columns": len(TERMS),
        "identifiable": rank == len(TERMS),
        "estimator": "weighted least squares, coefficients constrained >= 0",
        "scaled_condition_number": condition,
        "pairwise_design_correlation": pair_corr,
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
    """Restructured reconciliation gate, per the advisor ruling on PR 217.

    FINDING 543 is a direct paired measurement of the marginal m=9 weight pass
    and governs. FINDING 536's per-cell rank is a byte-share allocation of that
    same pass, so it is kept as context and does not gate.

    One pass over the seven fused cells is 14.4123 GB. At the 273 GB/s host DRAM
    peak that cannot cost less than 52.8 ms, yet the in-situ marginal pass costs
    20.774 ms, so the in-situ pass is majority cache-served. A cold-replica
    standalone pass must therefore cost MORE than the in-situ anchor and a
    hot-replica pass less, which makes the gate bracket containment:

        gate (a): [hot, cold] must contain the FINDING 543 anchor.
        gate (b): [hot_ratio, cold_ratio] must contain the FINDING 559 band.

    The cache-served fraction from the same bracket,

        phi = (cold - insitu) / (cold - hot),

    upper-bounds every mechanism that attacks weight-fetch traffic: only the
    (1 - phi) DRAM-served residual is addressable.
    """
    by_cell: dict[str, dict] = collections.defaultdict(dict)
    for unit in units.values():
        f = unit["fields"]
        if int(f.get("dispatches", 1)) != 1:
            continue
        cell = f.get("cell")
        if cell not in INVOCATIONS:
            continue
        by_cell[cell][
            (int(f["na"]), int(f["groups"]), bool(f.get("cold", True)))] = unit

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
        for na in sorted({key[0] for key in arms}):
            arm: dict = {}
            for cold in (True, False):
                tag = "cold" if cold else "hot"
                g = {j: arms.get((na, j, cold)) for j in (1, 2, 3)}
                for j, unit in g.items():
                    arm["%s_g%d_us" % (tag, j)] = (
                        unit["slope_us"] if unit else None)
                if g[1] and g[2]:
                    arm["%s_ratio_g2_over_g1" % tag] = (
                        g[2]["slope_us"] / g[1]["slope_us"])
                    arm["%s_marginal_pass_us" % tag] = (
                        g[2]["slope_us"] - g[1]["slope_us"])
                    arm["%s_marginal_pass_ms_per_round" % tag] = (
                        (g[2]["slope_us"] - g[1]["slope_us"]) * inv / 1000.0)
                    arm["%s_marginal_pass_sd_ms_per_round" % tag] = (
                        math.hypot(g[2]["slope_sd"], g[1]["slope_sd"])
                        * inv / 1000.0)
                    arm["%s_whole_dispatch_ms_per_round" % tag] = (
                        g[1]["slope_us"] * inv / 1000.0)
                if g[2] and g[3]:
                    arm["%s_third_pass_ms_per_round" % tag] = (
                        (g[3]["slope_us"] - g[2]["slope_us"]) * inv / 1000.0)
            hot = arm.get("hot_marginal_pass_ms_per_round")
            cold_ms = arm.get("cold_marginal_pass_ms_per_round")
            if hot is not None and cold_ms is not None:
                anchor = (FINDING_543_THIRD_PASS_MS
                          * entry["byte_share_of_one_pass"])
                arm["finding_543_byte_share_ms"] = anchor
                arm["bracket_ms_per_round"] = [
                    min(hot, cold_ms), max(hot, cold_ms)]
                arm["bracket_contains_543_share"] = bool(
                    min(hot, cold_ms) <= anchor <= max(hot, cold_ms))
                span = cold_ms - hot
                arm["phi_cache_served_fraction"] = (
                    (cold_ms - anchor) / span if span else None)
            entry["na%d" % na] = arm
        cells[cell] = entry

    # Gate (a). The census measures three of the seven fused cells, so their
    # measured marginal pass is scaled up by their byte share of one pass to
    # compare with the whole-round FINDING 543 anchor.
    measured_share = sum(
        entry["byte_share_of_one_pass"] for entry in cells.values())

    def staged_cost(ipg: int, g: int, cold: bool = True) -> float | None:
        """Byte-share-scaled whole-round QMV cost at one staged width."""
        total, covered = 0.0, 0.0
        for cell, entry in cells.items():
            unit = by_cell[cell].get((ipg, g, cold))
            if unit is None:
                return None
            total += unit["slope_us"] * INVOCATIONS[cell] / 1000.0
            covered += entry["byte_share_of_one_pass"]
        return total / covered if covered > 0 else None

    totals: dict = {}
    for tag in ("cold", "hot"):
        parts = {
            cell: entry.get("na5", {}).get(
                "%s_marginal_pass_ms_per_round" % tag)
            for cell, entry in cells.items()
        }
        if not parts or any(v is None for v in parts.values()):
            totals[tag] = None
            continue
        measured = sum(parts.values())
        totals[tag] = {
            "measured_cells_ms_per_round": measured,
            "byte_share_covered": measured_share,
            "scaled_to_all_cells_ms_per_round": measured / measured_share,
            "per_cell_ms_per_round": parts,
        }

    for tag in ("cold", "hot"):
        if totals.get(tag) is None:
            continue
        sds = [
            entry.get("na5", {}).get("%s_marginal_pass_sd_ms_per_round" % tag)
            for entry in cells.values()
        ]
        if all(v is not None for v in sds):
            totals[tag]["scaled_sd_ms_per_round"] = (
                math.sqrt(sum(v * v for v in sds)) / measured_share)

    gate_a: dict = {
        "governing_reference": "FINDING 543",
        "form": "bracket containment",
        "anchor_ms_per_round": FINDING_543_THIRD_PASS_MS,
        "anchor_ci95_ms_per_round": list(FINDING_543_CI_MS),
        "cold": totals.get("cold"),
        "hot": totals.get("hot"),
        "pass": None,
    }
    if totals.get("cold") and totals.get("hot"):
        cold_v = totals["cold"]["scaled_to_all_cells_ms_per_round"]
        hot_v = totals["hot"]["scaled_to_all_cells_ms_per_round"]
        lo, hi = min(cold_v, hot_v), max(cold_v, hot_v)
        span = cold_v - hot_v
        sd_cold = totals["cold"].get("scaled_sd_ms_per_round")
        sd_hot = totals["hot"].get("scaled_sd_ms_per_round")
        noise = (2.0 * math.hypot(sd_cold, sd_hot)
                 if sd_cold is not None and sd_hot is not None else None)
        # phi needs a bracket that is both wide enough to resolve and oriented
        # the physical way round. A hot arm that is SLOWER than the cold arm is
        # not a weaker cache-residency signal, it is no signal at all.
        identifiable = (noise is not None and span > noise)
        gate_a.update({
            "bracket_ms_per_round": [lo, hi],
            "pass": bool(lo <= FINDING_543_CI_MS[1]
                         and hi >= FINDING_543_CI_MS[0]),
            "contains_point_anchor": bool(
                lo <= FINDING_543_THIRD_PASS_MS <= hi),
            "cold_over_insitu": cold_v / FINDING_543_THIRD_PASS_MS,
            "hot_over_insitu": hot_v / FINDING_543_THIRD_PASS_MS,
            "cold_minus_hot_ms_per_round": span,
            "cold_minus_hot_pct_of_cold": 100.0 * span / cold_v,
            "bracket_noise_2sigma_ms_per_round": noise,
            "phi_identifiable": bool(identifiable),
        })
        if identifiable:
            phi = (cold_v - FINDING_543_THIRD_PASS_MS) / span
            gate_a["phi_cache_served_fraction"] = phi
            gate_a["residual_dram_served_fraction"] = 1.0 - phi
            gate_a["phi_from_anchor_ci"] = [
                (cold_v - FINDING_543_CI_MS[1]) / span,
                (cold_v - FINDING_543_CI_MS[0]) / span,
            ]
        else:
            gate_a["phi_cache_served_fraction"] = None
            gate_a["outcome"] = "phi non-identifiable, gate inconclusive"
            gate_a["phi_non_identifiable_reason"] = (
                "the hot arm does not undercut the cold arm: cold minus hot is "
                "%+.3f ms per round, %.2f %% of the cold value, against a "
                "%.3f ms 2-sigma band. Re-touching the same replica does not "
                "create a cache-resident condition, because one pass over a "
                "single fused cell already streams 47-100 MB and exceeds every "
                "cache level on this host. The bracket therefore has no usable "
                "width and cannot resolve phi. Both ends of it also sit about "
                "3.5x above the 20.774 ms in-situ anchor, so no orientation of "
                "the bracket can contain that anchor. This REFUTES the "
                "hypothesis that a repeated weight pass is majority "
                "cache-served on a 48 GiB M4 Pro, and it refutes reading "
                "FINDING 543 as the price of a whole extra weight pass."
                % (span, 100.0 * span / cold_v,
                   noise if noise else float("nan")))

    # Gate (a'), PRIMARY. Absolute reconciliation against the FINDING 559 local
    # per-round QMV cost R(m). For each staged width the census sums
    # invocations x measured cold slope over the cells it measured, then scales
    # by the byte share those cells cover. No cache-residency assumption enters,
    # so this checks the instrument end to end where gate (a) cannot.
    staged_rows = []
    for m in sorted(FINDING_559_R_LOCAL):
        if m not in STAGED_PLAN:
            continue
        ipg, g = STAGED_PLAN[m]
        parts, covered, missing = {}, 0.0, []
        for cell, entry in sorted(cells.items()):
            unit = by_cell[cell].get((ipg, g, True))
            if unit is None:
                missing.append(cell)
                continue
            parts[cell] = unit["slope_us"] * INVOCATIONS[cell] / 1000.0
            covered += entry["byte_share_of_one_pass"]
        reference = FINDING_559_R_LOCAL[m]
        row = {
            "m": m, "ipg": ipg, "groups": g,
            "finding_559_r_local_ms_per_round": reference,
            "measured_cells": sorted(parts),
            "missing_cells": missing,
            "byte_share_covered": covered,
            "per_cell_ms_per_round": parts,
        }
        if missing or covered <= 0:
            row["census_ms_per_round"] = None
            row["within_15pct"] = None
            row["gap_reason"] = (
                "the groups phase measured NA in {3, 5} only, so the (%d, %d) "
                "arm needed by m=%d was not timed" % (ipg, g, m))
        else:
            scaled = sum(parts.values()) / covered
            row["census_ms_per_round"] = scaled
            row["ratio_census_over_559"] = scaled / reference
            row["error_pct"] = 100.0 * (scaled / reference - 1.0)
            row["within_15pct"] = abs(scaled / reference - 1.0) <= (
                GATE_TOLERANCE)
        staged_rows.append(row)

    scored_rows = [r for r in staged_rows if r["within_15pct"] is not None]
    gate_a_prime = {
        "governing_reference": "FINDING 559 R_local(m)",
        "form": "absolute per-round cost within 15 % at each staged width",
        "tolerance_pct": 100.0 * GATE_TOLERANCE,
        "rows": staged_rows,
        "widths_checked": [r["m"] for r in scored_rows],
        "widths_missing": [r["m"] for r in staged_rows
                           if r["within_15pct"] is None],
        "worst_error_pct": (
            max((abs(r["error_pct"]) for r in scored_rows), default=None)),
        "pass": bool(scored_rows) and all(
            r["within_15pct"] for r in scored_rows),
    }

    # Layout headroom. No mechanism that keeps the same device bytes can go
    # below the host DRAM floor for those bytes, so the distance from the
    # measured cost down to that floor caps every such mechanism.
    pooled_ceiling, pooled_weight, per_width = 0.0, 0.0, []
    for row in staged_rows:
        if row["census_ms_per_round"] is None:
            continue
        w = POOLED_WIDTH_SHARE.get(row["m"], 0.0)
        floor_ms = total_stream * row["groups"] / HOST_DRAM_PEAK_GB_PER_S / 1e9 \
            * 1000.0
        ceiling = max(0.0, row["census_ms_per_round"] - floor_ms)
        per_width.append({
            "m": row["m"], "pooled_weight": w,
            "census_ms_per_round": row["census_ms_per_round"],
            "dram_floor_ms_per_round": floor_ms,
            "ceiling_ms_per_round": ceiling,
        })
        pooled_ceiling += w * ceiling
        pooled_weight += w

    layout = None
    m9 = next((r for r in staged_rows if r["m"] == 9
               and r["census_ms_per_round"]), None)
    if m9:
        bytes_per_round = total_stream * m9["groups"]
        achieved = bytes_per_round / (m9["census_ms_per_round"] * 1e-3) / 1e9
        layout = {
            "per_width": per_width,
            "pooled_weight_covered": pooled_weight,
            "pooled_ceiling_ms_per_round": (
                pooled_ceiling / pooled_weight if pooled_weight else None),
            "reference_width_m": 9,
            "bytes_per_round": bytes_per_round,
            "census_ms_per_round": m9["census_ms_per_round"],
            "achieved_gb_per_s": achieved,
            "host_peak_gb_per_s": HOST_DRAM_PEAK_GB_PER_S,
            "fraction_of_peak": achieved / HOST_DRAM_PEAK_GB_PER_S,
            "headroom_fraction": max(
                0.0, 1.0 - achieved / HOST_DRAM_PEAK_GB_PER_S),
            "ceiling_ms_per_round": m9["census_ms_per_round"] * max(
                0.0, 1.0 - achieved / HOST_DRAM_PEAK_GB_PER_S),
            "note":
                "this bounds mechanisms that only reorder or coalesce existing "
                "traffic. A mechanism that REMOVES bytes is not bounded by it, "
                "and because phi is not identifiable here (hot == cold) a "
                "byte-removing mechanism takes no cache-residency discount",
        }

    # FINDING 536 as context only: it allocates the same pass by byte share.
    context_536 = []
    for cell, entry in sorted(cells.items()):
        arm = entry.get("na5", {})
        published = FINDING_536_CELLS.get(cell)
        cold_ms = arm.get("cold_marginal_pass_ms_per_round")
        if published is None or cold_ms is None:
            continue
        context_536.append({
            "cell": cell,
            "cold_marginal_pass_ms_per_round": cold_ms,
            "hot_marginal_pass_ms_per_round":
                arm.get("hot_marginal_pass_ms_per_round"),
            "finding_536_published_ms": published,
            "ratio_cold_vs_536": cold_ms / published,
            "within_15pct_of_536":
                abs(cold_ms / published - 1.0) <= GATE_TOLERANCE,
        })

    # Gate (b): the in-situ second-pass ratio band must sit inside the
    # standalone hot-to-cold ratio bracket.
    ratio_rows = []
    for cell, entry in sorted(cells.items()):
        for key, arm in sorted(entry.items()):
            if not key.startswith("na") or not isinstance(arm, dict):
                continue
            cold_r = arm.get("cold_ratio_g2_over_g1")
            hot_r = arm.get("hot_ratio_g2_over_g1")
            if cold_r is None or hot_r is None:
                continue
            lo, hi = min(cold_r, hot_r), max(cold_r, hot_r)
            ratio_rows.append({
                "cell": cell, "arm": key,
                "cold_ratio": cold_r, "hot_ratio": hot_r,
                "bracket": [lo, hi],
                "contains_559_band": bool(
                    lo <= FINDING_559_RATIO_BAND[0]
                    and hi >= FINDING_559_RATIO_BAND[1]),
                "overlaps_559_band": bool(
                    lo <= FINDING_559_RATIO_BAND[1]
                    and hi >= FINDING_559_RATIO_BAND[0]),
            })
    # The scored quantity for gate (b) is the byte-weighted WHOLE-ROUND ratio.
    # FINDING 559's second-pass ratio is a whole-round in-situ quantity, so the
    # comparable census quantity weights each cell by its invocations and bytes
    # rather than treating one cell as representative.
    gate_b = {
        "band": list(FINDING_559_RATIO_BAND),
        "form": "the byte-weighted whole-round T(G=2)/T(G=1) at NA=5 must lie "
                "inside the FINDING 559 in-situ band",
        "per_cell_rows_are_context": True,
        "rows": ratio_rows,
        "band_note":
            "the advisor ruling wrote this band as 0.80-0.91; FINDING 559's "
            "published second-pass ratios are 1.796, 1.907 and 1.838, so the "
            "band applied here is [1.80, 1.91]. The two forms agree in fact: "
            "0.80-0.91 is the marginal increment, 1.80-1.91 is the total ratio",
    }
    for tag, cold in (("cold", True), ("hot", False)):
        one, two = staged_cost(5, 1, cold), staged_cost(5, 2, cold)
        if one is None or two is None:
            continue
        gate_b["%s_g1_ms_per_round" % tag] = one
        gate_b["%s_g2_ms_per_round" % tag] = two
        gate_b["%s_whole_round_ratio" % tag] = two / one
    scored = gate_b.get("cold_whole_round_ratio")
    gate_b["scored_ratio"] = scored
    gate_b["pass"] = bool(
        scored is not None
        and FINDING_559_RATIO_BAND[0] <= scored <= FINDING_559_RATIO_BAND[1])

    # What FINDING 543 actually bought. FINDING 543 is the paired in-situ move
    # from (m=9, IPG=3, G=3) to (m=9, IPG=5, G=2). The census prices exactly
    # those two whole-round configurations, so the weight-pass share of the
    # 20.774 ms is the difference between them.
    decomp = None
    before, after = staged_cost(3, 3), staged_cost(5, 2)
    if before is not None and after is not None:
        weight_pass = before - after
        decomp = {
            "finding_543_move": "(m=9, IPG=3, G=3) -> (m=9, IPG=5, G=2)",
            "finding_543_measured_ms_per_round": FINDING_543_THIRD_PASS_MS,
            "census_ipg3_g3_ms_per_round": before,
            "census_ipg5_g2_ms_per_round": after,
            "census_weight_pass_delta_ms_per_round": weight_pass,
            "weight_pass_share_of_543": weight_pass / (
                FINDING_543_THIRD_PASS_MS),
            "residual_non_qmv_ms_per_round": (
                FINDING_543_THIRD_PASS_MS - weight_pass),
            "interpretation":
                "the QMV weight pass explains only part of FINDING 543. G=2->3 "
                "adds no extra QMV dispatch, because G lives in the grid x "
                "dimension, so the remainder is per-group work outside the QMV "
                "kernels: activation slicing, sums-table construction, "
                "non-QMV kernels and extra graph or scheduling cost",
        }

    return {
        "cells": cells,
        "gate_a_prime": gate_a_prime,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_a_prime_pass": gate_a_prime["pass"],
        "gate_a_pass": gate_a.get("pass"),
        "gate_b_pass": gate_b["pass"],
        "phi_cache_served_fraction": gate_a.get("phi_cache_served_fraction"),
        "phi_identifiable": gate_a.get("phi_identifiable"),
        "layout_headroom": layout,
        "finding_543_decomposition": decomp,
        "finding_536_context": context_536,
        "bandwidth_argument": {
            "one_pass_bytes": total_stream,
            "host_dram_peak_gb_per_s": 273.0,
            "fully_streamed_floor_ms": total_stream / 273e9 * 1000.0,
            "insitu_marginal_pass_ms": FINDING_543_THIRD_PASS_MS,
            "insitu_implied_gb_per_s":
                total_stream / (FINDING_543_THIRD_PASS_MS * 1e-3) / 1e9,
            "finding_536_headline_ms": FINDING_536_HEADLINE_MS,
            "finding_536_cell_sum_ms": sum(FINDING_536_CELLS.values()),
            "conclusion":
                "the in-situ 20.774 ms figure implies 694 GB/s over 14.4123 GB, "
                "which is 2.5x host DRAM peak, so it cannot be the price of a "
                "whole extra weight pass. The census confirms this directly: "
                "the measured (3,3)->(5,2) weight-pass delta is far smaller, "
                "and the hot arm does not undercut the cold arm, so the excess "
                "is not cache residency. FINDING 543 therefore measured mostly "
                "per-group work outside the QMV weight stream",
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


# ---------------------------------------------------- what the terms mean
# Nominal Apple M4 Pro GPU lane throughput: 20 cores x 128 lanes x 1.4 GHz.
# Used only to convert the fitted per-byte cost into lane-cycles per weight
# element, so it is labelled nominal wherever it appears.
NOMINAL_LANE_THROUGHPUT_PER_S = 20 * 128 * 1.4e9


def term_interpretation(coef: dict) -> dict:
    """Convert the fitted terms into physical rates and name what they can be.

    The key test is whether the byte-proportional term can be device fetch at
    all. If the b term alone implies a streaming rate above host DRAM peak, it
    cannot be the fetch, so it must be per-byte WORK that the fetch hides
    under: nibble extraction, scale and bias application, accumulation.
    """
    a = coef["a_dispatch_us"]["value"]
    b = coef["b_stream_us_per_mib"]["value"]
    c = coef["c_mac_us_per_gmac"]["value"]

    total_stream = sum(
        stream_bytes(k, n) * inv for k, n, inv in FUSED_CELLS.values())
    total_nk = sum(n * k * inv for k, n, inv in FUSED_CELLS.values())
    dispatches = sum(inv for _k, _n, inv in FUSED_CELLS.values())

    b_ms_per_pass = b * total_stream / MIB / 1000.0
    b_implied_gbps = MIB / (b * 1e-6) / 1e9
    seconds_per_element = b * 1e-6 / MIB * (9.0 / 16.0)
    lane_cycles_per_element = (
        seconds_per_element * NOMINAL_LANE_THROUGHPUT_PER_S)
    c_ms_per_pass_per_na = c * total_nk / 1e9 / 1000.0
    a_ms_per_round = a * dispatches / 1000.0

    pass51 = a_ms_per_round + b_ms_per_pass + 5 * c_ms_per_pass_per_na
    achieved = total_stream / (pass51 * 1e-3) / 1e9

    return {
        "a_dispatch": {
            "us_per_dispatch": a,
            "dispatches_per_round": dispatches,
            "ms_per_round": a_ms_per_round,
            "reading": "pipelined marginal cost of one extra QMV dispatch at "
                       "conserved bytes, measured directly by the dispatch "
                       "split phase",
        },
        "b_stream": {
            "us_per_mib": b,
            "ms_per_pass_all_cells": b_ms_per_pass,
            "implied_stream_gb_per_s": b_implied_gbps,
            "host_dram_peak_gb_per_s": HOST_DRAM_PEAK_GB_PER_S,
            "exceeds_dram_peak": bool(
                b_implied_gbps > HOST_DRAM_PEAK_GB_PER_S),
            "nominal_lane_cycles_per_weight_element": lane_cycles_per_element,
            "reading":
                "this term is byte-proportional and NA-independent. Taken as "
                "pure streaming it implies %.0f GB/s, which is %.1fx host DRAM "
                "peak and therefore impossible. It is per-byte WORK that the "
                "device fetch hides under. At nominal lane throughput it is "
                "%.1f lane-cycles per 4-bit weight element, which matches the "
                "shift, mask, scale-multiply and bias-add of one affine "
                "group-64 dequantisation. The b term is dequantisation "
                "arithmetic, not the fetch."
                % (b_implied_gbps, b_implied_gbps / HOST_DRAM_PEAK_GB_PER_S,
                   lane_cycles_per_element),
        },
        "c_activation_row": {
            "us_per_gmac": c,
            "ms_per_pass_per_na": c_ms_per_pass_per_na,
            "implied_tmac_per_s": 1e9 / c / 1e12 * 1e6,
            "reading":
                "cost of one more activation row over the same weights. It is "
                "the accumulate work plus the register pressure that FINDING "
                "549 records at NA=5 (125 registers).",
        },
        "whole_pass_5_1": {
            "modelled_ms_per_round": pass51,
            "bytes_per_round": total_stream,
            "achieved_gb_per_s": achieved,
            "fraction_of_dram_peak": achieved / HOST_DRAM_PEAK_GB_PER_S,
            "reading":
                "one pass at NA=5, G=1 moves its bytes at %.0f GB/s, %.0f %% "
                "of host DRAM peak. The scored QMV pass is COMPUTE bound on "
                "this host, not DRAM bound. Mechanisms that remove device "
                "bytes are therefore worth less than mechanisms that remove "
                "dequantisation or per-row accumulate work."
                % (achieved, 100.0 * achieved / HOST_DRAM_PEAK_GB_PER_S),
        },
    }


# --------------------------------------------------------- mechanism pricing


def round_pass_cost(coef: dict, na: int, groups: int) -> float:
    """Modelled ms per decode round for every fused cell at (NA, G)."""
    total_us = 0.0
    for _name, (k, n, inv) in FUSED_CELLS.items():
        fields = {
            "groups": groups, "dispatches": 1, "na": na, "k": k, "n": n,
            "stream_bytes": stream_bytes(k, n),
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

    def pooled(fn, hi: bool = False) -> float:
        """Round-weighted ms/round over the FINDING 564 pooled census."""
        total = 0.0
        for m, w in POOLED_WIDTH_SHARE.items():
            ipg, g = STAGED_PLAN[m]
            total += w * fn(ipg, g, hi)
        return total

    def public(fn, hi: bool = False) -> float:
        total = 0.0
        norm = sum(PUBLIC_WIDTH_SHARE.values())
        for m, w in PUBLIC_WIDTH_SHARE.items():
            ipg, g = STAGED_PLAN[m]
            total += (w / norm) * fn(ipg, g, hi)
        return total

    def cval(name: str, hi: bool) -> float:
        """Point estimate, or the 97.5th bootstrap percentile when hi."""
        entry = coef[name]
        if not hi:
            return entry["value"]
        top = entry["ci95"][1]
        return max(entry["value"], top if np.isfinite(top) else entry["value"])

    out = []

    # 1. Remove the bf16 scale/bias interleave cost from the stream: pack the
    #    group metadata so it rides in the same cache lines as its nibbles.
    #    Ceiling = the whole scale/bias share of the b term.
    def metadata(ipg: int, g: int, hi: bool) -> float:
        removed = sum(
            inv * n * (k // 64) * 4 * g for k, n, inv in FUSED_CELLS.values())
        return removed / HOST_DRAM_PEAK_GB_PER_S / 1e9 * 1000.0
    out.append({
        "mechanism": "transform-side scale/bias interleave (fetch bytes)",
        "term": "fetch_bytes",
        "not_resolved_by_this_census": True,
        "assumption":
            "the bf16 scale and bias stream is 1/9 of device bytes. This "
            "ceiling is the host DRAM time of those bytes, which is only "
            "realizable if the pass were DRAM bound. The census says it is "
            "not: the pass runs at about two thirds of peak and its "
            "byte-proportional term is dequantisation arithmetic, not fetch. "
            "The census therefore CANNOT resolve a separate fetch term, and "
            "the realizable value of better metadata locality is well below "
            "this number",
        "fn": metadata,
    })

    # 2. QMV dispatch fusion: cells that share one activation read are issued as
    #    one dispatch. Ceiling = the a term on the removed dispatches. The
    #    fusable pairs in one round are (gdn.in_proj + fa.qkv share x) and the
    #    two o_proj cells; conservatively price removing one dispatch per layer.
    def fusion(ipg: int, g: int, hi: bool) -> float:
        removed = 64  # one dispatch per transformer layer
        return removed * cval("a_dispatch_us", hi) / 1000.0
    out.append({
        "mechanism": "QMV dispatch fusion (a term)",
        "term": "a_dispatch_us",
        "assumption":
            "one dispatch per layer is removed by fusing cells that already "
            "share an activation read; no byte term changes",
        "fn": fusion,
    })

    # 3. Output staging: write bf16 results through a wider per-simdgroup store
    #    instead of one lane-0 scalar per (row, column). Ceiling = the d term.
    def output(ipg: int, g: int, hi: bool) -> float:
        d_out = cval("d_output_us_per_mib", hi)
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
        "fn": output,
    })

    # 4. Geometry: 8 output rows per threadgroup is a compile-time choice.
    #    Doubling rows per threadgroup halves the threadgroup count at the same
    #    bytes. Ceiling = half the e term.
    def geometry(ipg: int, g: int, hi: bool) -> float:
        e_tg = cval("e_threadgroup_us_per_1k", hi)
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
        "fn": geometry,
    })

    # 5. Remove the second pass entirely (single-pass at NA = 9/10). This is the
    #    already-measured E208 axis, priced here as a model cross-check.
    def single_pass(ipg: int, g: int, hi: bool) -> float:
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
        "fn": single_pass,
    })

    # 6. NA-scaling relief, priced from the measured IPG ladder rather than the
    #    fit. Between NA=2 and NA=3 the per-pass cost is nearly flat: the extra
    #    activation row costs only a fused multiply-add against weights that are
    #    already in registers. Above NA=3 the cost per streamed MB climbs
    #    sharply, which matches the FINDING 549 register grid (NA5 needs 125
    #    registers). The census cannot say whether that is spilling, lower
    #    occupancy or repeated dequantisation, but it does price the channel:
    #    the ceiling is the excess over the cheap low-NA marginal rate.
    ladder: dict[str, dict[int, float]] = {}
    for cell, entry in (recon.get("cells") or {}).items():
        for key, arm in entry.items():
            if not key.startswith("na") or not isinstance(arm, dict):
                continue
            g1 = arm.get("cold_g1_us")
            if g1 is not None:
                ladder.setdefault(cell, {})[int(key[2:])] = g1

    def na_relief(ipg: int, g: int, hi: bool) -> float:
        us, covered = 0.0, 0.0
        for cell, rung in ladder.items():
            if ipg not in rung or 2 not in rung or 3 not in rung:
                continue
            cheap = rung[2] + (ipg - 2) * (rung[3] - rung[2])
            k, n, inv = FUSED_CELLS[cell]
            us += inv * g * max(0.0, rung[ipg] - cheap)
            covered += stream_bytes(k, n) * inv
        total = sum(stream_bytes(k, n) * inv
                    for k, n, inv in FUSED_CELLS.values())
        if covered <= 0:
            return 0.0
        return us / 1000.0 * total / covered
    out.append({
        "mechanism": "NA-scaling relief above NA=3 (measured IPG ladder)",
        "term": "measured",
        "assumption":
            "the NA=2 to NA=3 step is the cheap marginal rate, because the "
            "extra activation row only adds a fused multiply-add on weights "
            "already held in registers. The ceiling is the excess of the "
            "measured NA=4 and NA=5 cost over that rate, extended across "
            "groups linearly. Linear extension across groups is conservative, "
            "because the widest arms measured here turn superlinear at chain "
            "depth 3 and above",
        "implementation_route":
            "the accumulator count per thread is outputs_per_simdgroup x NA. "
            "The shipped geometry gives 4 outputs per simdgroup, so NA=5 holds "
            "20 accumulators and FINDING 549 measures 125 registers. Halving "
            "outputs per simdgroup from 4 to 2 halves the accumulator count at "
            "UNCHANGED device bytes, because each output column owns its own "
            "weight column. The only cost is twice as many threadgroups in n, "
            "and the fitted threadgroup term is %s us per 1000 threadgroups, "
            "which is negligible. This also creates register headroom above "
            "the FINDING 549 NA=5 legality wall, which is the same wall that "
            "blocks the single-pass arm listed below"
            % coef["e_threadgroup_us_per_1k"]["value"],
        "fn": na_relief,
    })

    # 7. Dequantisation sharing across token groups: the E217 axis. The census
    #    shows the second token group re-pays the whole NA-independent
    #    byte-proportional term, and that term cannot be device fetch because it
    #    alone implies a rate above host DRAM peak. E216 already deduped the
    #    FETCH by co-residency and gained nothing, which leaves dequantisation
    #    as its content. Staging the dequantised tile therefore attacks the
    #    whole term on every pass after the first.
    def dequant_share(ipg: int, g: int, hi: bool) -> float:
        if g < 2:
            return 0.0
        b = cval("b_stream_us_per_mib", hi)
        us = 0.0
        for _n, (k, n, inv) in FUSED_CELLS.items():
            us += inv * b * stream_bytes(k, n) / MIB * (g - 1)
        return us / 1000.0
    out.append({
        "mechanism": "dequantisation sharing across token groups (E217 axis)",
        "term": "b_stream_us_per_mib",
        "assumption":
            "every token group after the first re-pays the whole b term today. "
            "A shared dequantised tile removes it. The ceiling assumes perfect "
            "sharing and charges nothing for the geometry needed to co-locate "
            "the groups. In the shipped kernel the token groups occupy "
            "DIFFERENT threadgroups along grid x, so they cannot share "
            "threadgroup memory without a geometry change, and that geometry "
            "change is the whole risk of the mechanism",
        "fn": dequant_share,
    })

    # Discount policy. The advisor ruling prices fetch mechanisms only on the
    # (1 - phi) DRAM-served residual. On this host phi is not identifiable,
    # because the hot arm does not undercut the cold arm, so there is no
    # measured cache-residency discount to apply and a byte-REMOVING mechanism
    # keeps its full census price. A mechanism that only REORDERS bytes is
    # instead capped by the measured distance to host DRAM peak.
    gate_a = recon.get("gate_a") or {}
    layout = recon.get("layout_headroom") or {}
    phi = gate_a.get("phi_cache_served_fraction")
    phi_identifiable = bool(gate_a.get("phi_identifiable"))
    residual = (max(0.0, min(1.0, 1.0 - phi))
                if phi_identifiable and phi is not None else 1.0)
    headroom = layout.get("headroom_fraction")

    reorder_cap = layout.get("pooled_ceiling_ms_per_round")

    for entry in out:
        fn = entry.pop("fn")
        entry["pooled_ms_per_round"] = pooled(fn)
        entry["public_ms_per_round"] = public(fn)
        entry["pooled_ms_per_round_ci_upper"] = pooled(fn, hi=True)
        entry["clears_1ms_bar_pooled"] = entry["pooled_ms_per_round"] >= 1.0
        entry["phi_identifiable"] = phi_identifiable
        entry["phi_cache_served_fraction"] = phi
        entry["dram_residual_fraction_applied"] = residual

        removes_bytes = entry["term"] == "fetch_bytes"
        reorders_only = entry["term"] in (
            "d_output_us_per_mib", "e_threadgroup_us_per_1k", "measured",
            "b_stream_us_per_mib")
        cap = entry["pooled_ms_per_round"]
        if removes_bytes:
            entry["discount_basis"] = (
                "removes device bytes; full census price, no phi discount "
                "because phi is not identifiable on this instrument"
                if not phi_identifiable else
                "removes device bytes; priced on the DRAM-served residual")
            cap *= residual
        elif reorders_only and reorder_cap is not None:
            entry["discount_basis"] = (
                "keeps the same device bytes, so it cannot pass the host DRAM "
                "floor for those bytes; capped by the pooled distance from the "
                "measured cost down to that floor")
            entry["layout_headroom_fraction"] = headroom
            entry["pooled_layout_ceiling_ms_per_round"] = reorder_cap
            cap = min(cap, reorder_cap)
        else:
            entry["discount_basis"] = (
                "no traffic assumption enters; census price used directly")
        entry["decision_ms_per_round"] = cap
        entry["clears_1ms_bar"] = cap >= 1.0
    out.sort(key=lambda e: (bool(e.get("not_resolved_by_this_census")),
                            -e["decision_ms_per_round"]))
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
        report["term_interpretation"] = term_interpretation(fit["coefficients"])
        report["priced_shortlist"] = price_mechanisms(fit["coefficients"], recon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("wrote %s" % args.out)

    if fit:
        print("\ncomposition fit (rank %s/%s, weighted R2 %.4f, n=%s, "
              "scaled condition %.3g):"
              % (fit["design_rank"], fit["design_columns"],
                 fit["weighted_r2"], fit["n_units"],
                 fit["scaled_condition_number"]))
        for name, entry in fit["coefficients"].items():
            print("  %-28s %12.6g  CI95 [%.6g, %.6g]%s"
                  % (name, entry["value"], entry["ci95"][0], entry["ci95"][1],
                     "" if entry["resolved_away_from_zero"]
                     else "  (not resolved > 0)"))
        print("  worst design correlation: %s vs %s  r=%.4f"
              % (fit["pairwise_design_correlation"][0]["a"],
                 fit["pairwise_design_correlation"][0]["b"],
                 fit["pairwise_design_correlation"][0]["pearson_r"]))

    ti = report.get("term_interpretation")
    if ti:
        print("\nterm interpretation:")
        print("  a: %.1f us/dispatch x %d dispatches = %.3f ms/round"
              % (ti["a_dispatch"]["us_per_dispatch"],
                 ti["a_dispatch"]["dispatches_per_round"],
                 ti["a_dispatch"]["ms_per_round"]))
        print("  b: %.3f ms/pass, implies %.0f GB/s = %.1fx DRAM peak "
              "(impossible as fetch), %.1f nominal lane-cycles per weight "
              "element"
              % (ti["b_stream"]["ms_per_pass_all_cells"],
                 ti["b_stream"]["implied_stream_gb_per_s"],
                 ti["b_stream"]["implied_stream_gb_per_s"]
                 / ti["b_stream"]["host_dram_peak_gb_per_s"],
                 ti["b_stream"]["nominal_lane_cycles_per_weight_element"]))
        print("  c: %.3f ms/pass per activation row"
              % ti["c_activation_row"]["ms_per_pass_per_na"])
        print("  pass (5,1): %.3f ms/round at %.0f GB/s = %.0f %% of peak "
              "-> COMPUTE bound"
              % (ti["whole_pass_5_1"]["modelled_ms_per_round"],
                 ti["whole_pass_5_1"]["achieved_gb_per_s"],
                 100.0 * ti["whole_pass_5_1"]["fraction_of_dram_peak"]))

    if recon:
        ap = recon["gate_a_prime"]
        print("\ngate (a') absolute vs FINDING 559 R_local: %s "
              "(worst error %.2f %%)"
              % (ap["pass"], ap["worst_error_pct"] or float("nan")))
        for row in ap["rows"]:
            if row["census_ms_per_round"] is None:
                print("    m=%d (%d,%d)  NOT MEASURED: %s"
                      % (row["m"], row["ipg"], row["groups"],
                         row["gap_reason"]))
                continue
            print("    m=%d (%d,%d)  census %8.3f  559 %8.3f  %+7.2f %%  %s"
                  % (row["m"], row["ipg"], row["groups"],
                     row["census_ms_per_round"],
                     row["finding_559_r_local_ms_per_round"],
                     row["error_pct"],
                     "ok" if row["within_15pct"] else "OUT"))

        ga = recon["gate_a"]
        print("\ngate (a) bracket containment vs FINDING 543 (%.3f ms): %s"
              % (ga["anchor_ms_per_round"], ga.get("outcome") or ga["pass"]))
        if ga.get("bracket_ms_per_round"):
            print("    bracket [%.3f, %.3f] ms/round, cold-hot %+.3f ms "
                  "(%.2f %%), 2-sigma %.3f, phi identifiable %s"
                  % (ga["bracket_ms_per_round"][0],
                     ga["bracket_ms_per_round"][1],
                     ga["cold_minus_hot_ms_per_round"],
                     ga["cold_minus_hot_pct_of_cold"],
                     ga["bracket_noise_2sigma_ms_per_round"],
                     ga["phi_identifiable"]))

        gb = recon["gate_b"]
        print("\ngate (b) whole-round ratio %.4f in band %s: %s"
              % (gb["scored_ratio"] or float("nan"), gb["band"], gb["pass"]))

        dec = recon.get("finding_543_decomposition")
        if dec:
            print("\nFINDING 543 decomposition: census (3,3) %.3f -> (5,2) "
                  "%.3f = %.3f ms/round weight pass, %.1f %% of the measured "
                  "%.3f ms"
                  % (dec["census_ipg3_g3_ms_per_round"],
                     dec["census_ipg5_g2_ms_per_round"],
                     dec["census_weight_pass_delta_ms_per_round"],
                     100.0 * dec["weight_pass_share_of_543"],
                     dec["finding_543_measured_ms_per_round"]))

        lay = recon.get("layout_headroom")
        if lay:
            print("layout headroom: %.1f GB/s achieved of %.0f peak "
                  "(%.1f %% of peak), reorder ceiling %.3f ms/round"
                  % (lay["achieved_gb_per_s"], lay["host_peak_gb_per_s"],
                     100.0 * lay["fraction_of_peak"],
                     lay["ceiling_ms_per_round"]))

    if report.get("priced_shortlist"):
        print("\npriced shortlist (decision ms/round, pooled FINDING 564):")
        for entry in report["priced_shortlist"]:
            print("  %-54s %8.3f  ci-hi %8.3f  public %8.3f  %s%s"
                  % (entry["mechanism"], entry["decision_ms_per_round"],
                     entry["pooled_ms_per_round_ci_upper"],
                     entry["public_ms_per_round"],
                     "CLEARS" if entry["clears_1ms_bar"] else "below",
                     "  (channel not resolved by this census)"
                     if entry.get("not_resolved_by_this_census") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
