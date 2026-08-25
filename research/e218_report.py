#!/usr/bin/env python3
"""E218: the non-QMV width-tax census — who charges for each extra draft row?

    usage: research/e218_report.py SESSION [SESSION ...]
                                   [--e186 research/out/e186/<tag>/widths.json]
                                   [--out research/e218-artifacts/width-tax-census.json]

harness=local, NOT gate-qualified. Every leg runs with the per-round phase
trace on and the cool gate off, so no number here is a gate-qualified timing
claim and none of them is a candidate speed claim.

Two leg kinds are read:

  * `wN` legs (band sync OFF) carry the round-cost law R_local(m) and its exact
    phase identity: round = draft_build + verify_build + eval_wall + readout +
    commit + upkeep. Adjacent-width differences of those phases are the TRUE
    per-round slopes.
  * `bN` legs (band sync ON) drain the verify forward at every mixer and MLP
    boundary, so `band_*_us` are per-layer-family device times. Those legs are
    ATTRIBUTION ONLY: the extra drains inflate the round, so a band step is
    read as a SHARE of the band total, never as a round cost. The paired
    w/b legs at the same width bound that perturbation directly.

The census combines them: the in-situ band shares distribute the TRUE verify
step measured by the w legs. Within a band, the isolated E186 per-cell curves
(qmv cells, gdn recurrence, sdpa, fused mlp, whole gdn/fa layers) split the
family into cells under an explicitly reported isolated->in-situ transfer
factor.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e212_report import (  # noqa: E402
    BYTES_PER_ELEMENT,
    CELLS,
    NOISE_FLOOR_MS,
    boot_ci,
    cell_groups,
    cell_variant,
    parse_meta,
    parse_rounds,
    weight_bytes,
)

ROOT = pathlib.Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out" / "e218"

# Phases that sum EXACTLY to round_us in a traced round.
ROUND_PHASES = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
DRAFT_SUBPHASES = [
    "d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
    "d_chain_us", "d_submit2_us",
]
BANDS = [
    "band_pre_us",
    "band_gdn_mixer_us",
    "band_gdn_mlp_us",
    "band_fa_mixer_us",
    "band_fa_mlp_us",
]
# How many layer instances fold into each band per verify forward.
BAND_LAYERS = {
    "band_pre_us": 1,
    "band_gdn_mixer_us": 48,
    "band_gdn_mlp_us": 48,
    "band_fa_mixer_us": 16,
    "band_fa_mlp_us": 16,
}
# The verify forward's GPU work in an untraced leg lands in these two phases:
# the host builds the graph (verify_build) and then blocks on it (eval_wall).
FORWARD_PHASES = ["verify_build_us", "eval_wall_us"]

ALL_PHASES = ROUND_PHASES + DRAFT_SUBPHASES + BANDS + ["round_us"]


def read_leg(leg_dir: pathlib.Path) -> dict | None:
    meta = parse_meta(leg_dir / "meta.txt")
    if not meta or "e218_arm" not in meta:
        return None
    arm = meta["e218_arm"]
    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})
    rounds = parse_rounds(leg_dir / "trace.txt")

    if arm[0] in "wb":
        depth = int(arm[1:])
        kept = [r for r in rounds if int(r["d"]) == depth]
        width = depth + 1
    else:
        kept = rounds
        width = None

    record = {
        "leg": leg_dir.name,
        "session": meta.get("e218_session"),
        "arm": arm,
        "width_m": width,
        "band_sync": meta.get("band_sync") == "1",
        "leg_index": int(meta.get("e218_leg", 0)),
        "rounds_traced": len(rounds),
        "rounds_at_width": len(kept),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c") or None,
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c") or None,
        "worker_sha256": meta.get("worker_sha256"),
        "worker_digest_stable": meta.get("worker_digest_stable"),
        "cli_digest_stable": meta.get("cli_digest_stable"),
        "base_sha": meta.get("base_sha"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
        "phases_ms": {},
        "phase_ci_ms": {},
    }
    for phase in ALL_PHASES:
        values = [r[phase] / 1000.0 for r in kept if phase in r]
        if not values:
            continue
        record["phases_ms"][phase] = {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "n": len(values),
        }
        lo, hi = boot_ci(values)
        record["phase_ci_ms"][phase] = [lo, hi]
    if width is None and rounds:
        histogram: dict[int, int] = {}
        for r in rounds:
            histogram[int(r["d"]) + 1] = histogram.get(int(r["d"]) + 1, 0) + 1
        record["served_width_histogram"] = {
            str(k): v for k, v in sorted(histogram.items())}
        record["served_rounds"] = len(rounds)
    return record


def width_table(legs: list[dict], band_sync: bool) -> dict:
    """Per width: the mean of the two counterbalanced leg medians, per phase."""
    buckets: dict[int, list[dict]] = {}
    for leg in legs:
        if leg["width_m"] is None or leg["band_sync"] != band_sync:
            continue
        buckets.setdefault(leg["width_m"], []).append(leg)

    table: dict[str, dict] = {}
    for m, bucket in sorted(buckets.items()):
        entry: dict = {
            "legs": [leg["leg"] for leg in bucket],
            "n_legs": len(bucket),
            "rounds_at_width": sum(leg["rounds_at_width"] for leg in bucket),
            "phase_ms": {},
        }
        for phase in ALL_PHASES:
            values = [leg["phases_ms"][phase]["median"] for leg in bucket
                      if phase in leg["phases_ms"]]
            if not values:
                continue
            spread = max(values) - min(values) if len(values) > 1 else 0.0
            entry["phase_ms"][phase] = {
                "value": statistics.fmean(values),
                "legs": values,
                "leg_spread": spread,
                "over_noise_floor": spread > NOISE_FLOOR_MS,
            }
        if band_sync:
            band_sum = sum(entry["phase_ms"][b]["value"] for b in BANDS
                           if b in entry["phase_ms"])
            entry["band_sum_ms"] = band_sum
            entry["band_plus_eval_ms"] = band_sum + entry["phase_ms"][
                "eval_wall_us"]["value"]
            entry["band_unaccounted_ms"] = (
                entry["phase_ms"]["round_us"]["value"] - band_sum)
            entry["band_per_layer_ms"] = {
                b: entry["phase_ms"][b]["value"] / BAND_LAYERS[b]
                for b in BANDS if b in entry["phase_ms"]}
        table[str(m)] = entry
    return table


def steps(table: dict, phases: list[str]) -> list[dict]:
    widths = sorted(int(k) for k in table)
    out = []
    for lo, hi in zip(widths, widths[1:]):
        if hi != lo + 1:
            continue
        a, b = table[str(lo)], table[str(hi)]
        row = {"boundary": f"{lo}->{hi}", "m_lo": lo, "m_hi": hi, "step_ms": {}}
        for phase in phases:
            if phase in a["phase_ms"] and phase in b["phase_ms"]:
                row["step_ms"][phase] = (
                    b["phase_ms"][phase]["value"] - a["phase_ms"][phase]["value"])
        out.append(row)
    return out


def linear_fit(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {
        "slope_per_row": slope,
        "intercept": intercept,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "rmse": math.sqrt(ss_res / n),
        "max_abs_residual": max(abs(r) for r in resid),
        "residuals": resid,
        "n": n,
    }


def step_fit(xs: list[float], ys: list[float], knot: float) -> dict:
    """Linear in m plus one step at `knot` (indicator m >= knot)."""
    n = len(xs)
    if n < 3:
        return {}
    ind = [1.0 if x >= knot else 0.0 for x in xs]
    # Normal equations for y = a + b*x + c*ind.
    cols = [[1.0] * n, xs, ind]
    ata = [[sum(ci[k] * cj[k] for k in range(n)) for cj in cols] for ci in cols]
    atb = [sum(ci[k] * ys[k] for k in range(n)) for ci in cols]
    # 3x3 solve by Gaussian elimination.
    mat = [row[:] + [atb[i]] for i, row in enumerate(ata)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(mat[r][i]))
        if abs(mat[piv][i]) < 1e-12:
            return {}
        mat[i], mat[piv] = mat[piv], mat[i]
        for r in range(3):
            if r == i:
                continue
            f = mat[r][i] / mat[i][i]
            for c in range(i, 4):
                mat[r][c] -= f * mat[i][c]
    coef = [mat[i][3] / mat[i][i] for i in range(3)]
    pred = [coef[0] + coef[1] * x + coef[2] * d for x, d in zip(xs, ind)]
    resid = [y - p for y, p in zip(ys, pred)]
    ss_res = sum(r * r for r in resid)
    my = statistics.fmean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {
        "knot": knot,
        "intercept": coef[0],
        "slope_per_row": coef[1],
        "step_ms": coef[2],
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "rmse": math.sqrt(ss_res / n),
    }


def shape_report(table: dict, phase: str, per_layer: bool = False) -> dict:
    widths = sorted(int(k) for k in table)
    xs, ys = [], []
    for m in widths:
        entry = table[str(m)]
        if phase not in entry["phase_ms"]:
            continue
        value = entry["phase_ms"][phase]["value"]
        if per_layer:
            value /= BAND_LAYERS.get(phase, 1)
        xs.append(float(m))
        ys.append(value)
    if len(xs) < 3:
        return {}
    report = {"widths": xs, "values_ms": ys, "linear": linear_fit(xs, ys)}
    best = None
    for knot in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        if knot <= min(xs) or knot > max(xs):
            continue
        fit = step_fit(xs, ys, knot)
        if fit and (best is None or fit["rmse"] < best["rmse"]):
            best = fit
    if best:
        report["best_step_fit"] = best
        report["step_improves_rmse_x"] = (
            report["linear"]["rmse"] / best["rmse"] if best["rmse"] else None)
    return report


def perturbation(w_table: dict, b_table: dict) -> dict:
    """How much the band drains inflate the round, width by width."""
    out = {}
    for key in sorted(set(w_table) & set(b_table), key=int):
        w, b = w_table[key], b_table[key]
        w_round = w["phase_ms"]["round_us"]["value"]
        b_round = b["phase_ms"]["round_us"]["value"]
        w_fwd = sum(w["phase_ms"][p]["value"] for p in FORWARD_PHASES)
        b_fwd = sum(b["phase_ms"][p]["value"] for p in FORWARD_PHASES)
        out[key] = {
            "w_round_ms": w_round,
            "b_round_ms": b_round,
            "round_inflation_ms": b_round - w_round,
            "round_inflation_x": b_round / w_round if w_round else None,
            "w_forward_ms": w_fwd,
            "b_forward_ms": b_fwd,
            "forward_inflation_ms": b_fwd - w_fwd,
            "forward_inflation_x": b_fwd / w_fwd if w_fwd else None,
        }
    return out


def family_census(w_table: dict, b_table: dict) -> dict:
    """Distribute the TRUE per-round slope over the in-situ families.

    The w legs give the exact phase identity of the round. The b legs give the
    share of the verify forward each layer family owns, and those shares
    distribute the w legs' forward step. `lm_head_readout` is what the verify
    forward pays outside every band: the final norm, the lm_head projection and
    the top-2 readout graph, which no `Qwen35BandTimer` lap covers.
    """
    families = {}
    widths = sorted(set(int(k) for k in w_table) & set(int(k) for k in b_table))
    for m in widths:
        b = b_table[str(m)]
        w = w_table[str(m)]
        bands = {name: b["phase_ms"][name]["value"] for name in BANDS
                 if name in b["phase_ms"]}
        # Every band drains inside the graph build, so the band-arm forward is
        # verify_build + eval_wall and whatever it holds beyond the band sum is
        # the final norm, the lm_head projection and the top-2 readout graph,
        # which no `Qwen35BandTimer` lap covers. The serial body (m=1) reports
        # the whole round as eval_wall with verify_build=0, so this residual
        # form is the only definition that reads both round shapes.
        forward_b = sum(b["phase_ms"][p]["value"] for p in FORWARD_PHASES)
        post = forward_b - sum(bands.values())
        shares = {name: value / forward_b for name, value in bands.items()}
        shares["lm_head_readout"] = post / forward_b
        w_forward = sum(w["phase_ms"][p]["value"] for p in FORWARD_PHASES)
        families[str(m)] = {
            "band_ms": bands,
            "post_band_ms": post,
            "band_forward_ms": forward_b,
            "shares": shares,
            "w_forward_ms": w_forward,
            "scaled_family_ms": {k: v * w_forward for k, v in shares.items()},
            "draft_head_ms": w["phase_ms"]["draft_build_us"]["value"],
            "round_ms": w["phase_ms"]["round_us"]["value"],
            "other_round_ms": (
                w["phase_ms"]["round_us"]["value"]
                - w_forward
                - w["phase_ms"]["draft_build_us"]["value"]),
        }
    return families


def slope_census(families: dict, w_table: dict) -> list[dict]:
    """dR/dm per family at each adjacent-width boundary, plus the residual."""
    out = []
    widths = sorted(int(k) for k in families)
    for lo, hi in zip(widths, widths[1:]):
        if hi != lo + 1:
            continue
        a, b = families[str(lo)], families[str(hi)]
        true_step = (w_table[str(hi)]["phase_ms"]["round_us"]["value"]
                     - w_table[str(lo)]["phase_ms"]["round_us"]["value"])
        row = {
            "boundary": f"{lo}->{hi}",
            "true_round_step_ms": true_step,
            "true_forward_step_ms": b["w_forward_ms"] - a["w_forward_ms"],
            "draft_head_step_ms": b["draft_head_ms"] - a["draft_head_ms"],
            "other_round_step_ms": b["other_round_ms"] - a["other_round_ms"],
            "family_step_ms": {},
            "family_share_of_forward_step": {},
        }
        forward_step = row["true_forward_step_ms"]
        for name in list(BANDS) + ["lm_head_readout"]:
            if name not in a["scaled_family_ms"] or name not in b["scaled_family_ms"]:
                continue
            delta = b["scaled_family_ms"][name] - a["scaled_family_ms"][name]
            row["family_step_ms"][name] = delta
            row["family_share_of_forward_step"][name] = (
                delta / forward_step if forward_step else None)
        attributed = (sum(row["family_step_ms"].values())
                      + row["draft_head_step_ms"] + row["other_round_step_ms"])
        row["attributed_ms"] = attributed
        row["closure_residual_ms"] = true_step - attributed
        row["closure_residual_over_2ms"] = abs(true_step - attributed) > 2.0
        # Weight-pass bookkeeping: which cells change pass count at this step.
        row["cells_changing_passes"] = [
            name for name, _, _, _ in CELLS
            if cell_groups(hi, name) != cell_groups(lo, name)]
        row["cell_plan_lo"] = {name: [cell_variant(lo, name), cell_groups(lo, name)]
                               for name, _, _, _ in CELLS}
        row["cell_plan_hi"] = {name: [cell_variant(hi, name), cell_groups(hi, name)]
                               for name, _, _, _ in CELLS}
        row["nominal_weight_gb_step"] = (
            weight_bytes(hi) - weight_bytes(lo)) / 1e9
        out.append(row)
    return out


# --- E186 isolated cell curves ------------------------------------------------

def load_e186(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    samples = payload["samples"]
    curves: dict[str, dict[int, float]] = {}
    spreads: dict[str, dict[int, list[float]]] = {}
    for s in samples:
        key = f"{s['family']}:{s['cell']}"
        curves.setdefault(key, {})
        spreads.setdefault(key, {}).setdefault(s["m"], []).append(
            s["microseconds"])
    for key, by_m in spreads.items():
        for m, values in by_m.items():
            curves[key][m] = statistics.median(values) / 1000.0  # ms
    return {
        "eval_floor_us": payload.get("eval_floor_microseconds"),
        "blocks": payload.get("blocks"),
        "qmv_arm": payload.get("qmv_arm"),
        "curves_ms": {k: {str(m): v for m, v in sorted(by_m.items())}
                      for k, by_m in curves.items()},
        "raw_spread_us": {k: {str(m): [min(v), max(v), len(v)]
                              for m, v in sorted(by_m.items())}
                          for k, by_m in spreads.items()},
    }


def cell_decomposition(e186: dict) -> dict:
    """Split each isolated layer curve into its cells and its remainder."""
    c = {k: {int(m): v for m, v in by_m.items()}
         for k, by_m in e186["curves_ms"].items()}

    def get(key: str) -> dict[int, float]:
        return c.get(key, {})

    widths = sorted(set(get("gdn_layer:nConfirmed=1")) or set())
    out: dict[str, dict] = {}

    gdn_layer = get("gdn_layer:nConfirmed=1")
    gdn_in = get("qmv_inpath:gdn.in_proj")
    gdn_out = get("qmv_inpath:gdn.out_proj")
    gdn_rec = get("gdn_recurrence:T=m")
    out["gdn_layer"] = {
        str(m): {
            "layer_ms": gdn_layer.get(m),
            "in_proj_ms": gdn_in.get(m),
            "out_proj_ms": gdn_out.get(m),
            "recurrence_ms": gdn_rec.get(m),
            "remainder_ms": (gdn_layer.get(m, float("nan"))
                             - gdn_in.get(m, float("nan"))
                             - gdn_out.get(m, float("nan"))
                             - gdn_rec.get(m, float("nan"))),
        }
        for m in widths
    }

    fa_keys = [k for k in c if k.startswith("fa_layer:")]
    sdpa_keys = [k for k in c if k.startswith("sdpa:")]
    fa_qkv = get("qmv_inpath:fa.qkv")
    fa_o = get("qmv_inpath:fa.o_proj")
    out["fa_layer"] = {}
    for fk in fa_keys:
        kv = fk.split("kv=")[-1]
        sk = f"sdpa:kv={kv}"
        out["fa_layer"][kv] = {
            str(m): {
                "layer_ms": c[fk].get(m),
                "qkv_ms": fa_qkv.get(m),
                "o_proj_ms": fa_o.get(m),
                "sdpa_ms": c.get(sk, {}).get(m),
                "remainder_ms": (c[fk].get(m, float("nan"))
                                 - fa_qkv.get(m, float("nan"))
                                 - fa_o.get(m, float("nan"))
                                 - c.get(sk, {}).get(m, float("nan"))),
            }
            for m in sorted(c[fk])
        }

    mlp = get("mlp:fused")
    gate_up = get("qmv_inpath:mlp.gate_up")
    down = get("qmv_inpath:mlp.down")
    out["mlp"] = {
        str(m): {
            "mlp_ms": mlp.get(m),
            "gate_up_ms": gate_up.get(m),
            "down_ms": down.get(m),
            "remainder_ms": (mlp.get(m, float("nan"))
                             - gate_up.get(m, float("nan"))
                             - down.get(m, float("nan"))),
        }
        for m in sorted(mlp)
    }
    out["lm_head"] = {str(m): v for m, v in sorted(get("qmv_inpath:lm_head").items())}

    # Per-row structure of the isolated recurrence: slope vs fixed launch cost.
    if gdn_rec:
        xs = [float(m) for m in sorted(gdn_rec)]
        ys = [gdn_rec[int(x)] for x in xs]
        out["recurrence_fit"] = {
            "linear": linear_fit(xs, ys),
            "value_at_m1_ms": gdn_rec.get(1),
        }
        best = None
        for knot in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            fit = step_fit(xs, ys, knot)
            if fit and (best is None or fit["rmse"] < best["rmse"]):
                best = fit
        out["recurrence_fit"]["best_step_fit"] = best
    return out


# Which isolated WHOLE-LAYER curve each in-situ band runs, and how many times
# per forward. 64 decoder layers: 48 gated-DeltaNet and 16 full attention, each
# with its own fused MLP. The layer curve is the unit of reconciliation because
# a per-cell sum charges the fixed per-`eval` submit floor once for every cell,
# while the layer runs all of its cells inside one graph.
BAND_LAYER_CURVES = {
    "band_gdn_mixer_us": ("gdn_layer:nConfirmed=1", 48),
    "band_gdn_mlp_us": ("mlp:fused", 48),
    "band_fa_mixer_us": ("fa_layer:kv={kv}", 16),
    "band_fa_mlp_us": ("mlp:fused", 16),
    "lm_head_readout": ("qmv_inpath:lm_head", 1),
}
# The cells inside each isolated layer curve, split by whether they stream
# quantized weights through the QMV path.
LAYER_CELLS = {
    "gdn_layer:nConfirmed=1": (
        ["qmv_inpath:gdn.in_proj", "qmv_inpath:gdn.out_proj"],
        ["gdn_recurrence:T=m"]),
    "fa_layer:kv={kv}": (
        ["qmv_inpath:fa.qkv", "qmv_inpath:fa.o_proj"], ["sdpa:kv={kv}"]),
    "mlp:fused": (["qmv_inpath:mlp.gate_up", "qmv_inpath:mlp.down"], []),
    "qmv_inpath:lm_head": (["qmv_inpath:lm_head"], []),
}


def insitu_reconciliation(e186: dict, b_table: dict, w_table: dict,
                          kv: str = "1024") -> dict:
    """Absolute-level closure: isolated layers scaled to the whole forward.

    RULE 405 v2 wants absolute levels, not a hot/cold bracket. Each band's
    modeled level is its isolated whole-layer curve times the layer count. The
    in-situ level comes from the band arm at the same width, and the forward
    total is checked against the untraced w leg, which is the only unperturbed
    instrument here.

    Inside a layer, the isolated per-cell times are corrected by one additive
    per-call submit floor `phi_call = (cell_sum - layer_total) / n_cells`, so
    the corrected cells sum exactly to the measured layer. When the cells
    already sum to less than the layer, the shortfall is in-layer work this
    census does not measure separately (norms, convolution, gating, activation).
    """
    curves = {k: {int(m): v for m, v in by_m.items()}
              for k, by_m in e186["curves_ms"].items()}

    def cell(key: str, m: int) -> float | None:
        return curves.get(key.format(kv=kv), {}).get(m)

    rows = []
    for key in sorted(b_table, key=int):
        m = int(key)
        band_rows = {}
        modeled_total = 0.0
        for band, (layer_key, count) in BAND_LAYER_CURVES.items():
            layer = cell(layer_key, m)
            qmv_keys, non_qmv_keys = LAYER_CELLS[layer_key]
            qmv_raw = [cell(k, m) for k in qmv_keys]
            non_raw = [cell(k, m) for k in non_qmv_keys]
            named = [k.format(kv=kv) for k in qmv_keys + non_qmv_keys]
            values = qmv_raw + non_raw
            if layer is None or any(v is None for v in values):
                continue
            cell_sum = sum(values)
            n_cells = len(values)
            phi = max((cell_sum - layer) / n_cells, 0.0) if n_cells else 0.0
            corrected = [max(v - phi, 0.0) for v in values]
            in_layer_other = layer - sum(corrected)
            qmv = sum(corrected[:len(qmv_keys)]) * count
            non_qmv = sum(corrected[len(qmv_keys):]) * count
            insitu = b_table[key]["phase_ms"].get(band, {}).get("value")
            if insitu is None and band == "lm_head_readout":
                insitu = post_band_ms(b_table[key])
            modeled = layer * count
            modeled_total += modeled
            band_rows[band] = {
                "layer_curve": layer_key.format(kv=kv),
                "layer_count": count,
                "isolated_layer_ms": layer,
                "phi_call_ms": phi,
                "modeled_ms": modeled,
                "modeled_qmv_ms": qmv,
                "modeled_non_qmv_ms": non_qmv,
                "modeled_in_layer_other_ms": in_layer_other * count,
                "corrected_cells_ms": dict(zip(named, corrected)),
                "insitu_ms": insitu,
                "insitu_minus_modeled_ms": (
                    None if insitu is None else insitu - modeled),
                "modeled_over_insitu": (
                    None if not insitu else modeled / insitu),
            }
        # The band arm is the only in-situ instrument that reports band levels,
        # so it is the comparator for the modeled total. The unperturbed w leg
        # is reported beside it: their ratio is the band-drain perturbation
        # already measured in `perturbation`, not a modeling error.
        band_forward = (b_table[key]["phase_ms"]["verify_build_us"]["value"]
                        + b_table[key]["phase_ms"]["eval_wall_us"]["value"])
        w_forward = (w_table[key]["phase_ms"]["verify_build_us"]["value"]
                     + w_table[key]["phase_ms"]["eval_wall_us"]["value"]) \
            if key in w_table else None
        rows.append({
            "m": m,
            "bands": band_rows,
            "modeled_forward_ms": modeled_total,
            "insitu_forward_band_arm_ms": band_forward,
            "modeled_over_insitu_band_arm": modeled_total / band_forward,
            "within_15pct": abs(modeled_total / band_forward - 1.0) <= 0.15,
            "unperturbed_w_forward_ms": w_forward,
            "band_arm_over_w_forward": (
                None if not w_forward else band_forward / w_forward),
        })
    return {
        "kv_used": kv,
        "unit": "isolated whole-layer curve x layer count",
        "eval_floor_ms": (e186.get("eval_floor_us") or 0.0) / 1000.0,
        "per_width": rows,
    }


def slope_closure(reconciliation: dict, w_table: dict,
                  tolerance_ms: float = 2.0) -> list[dict]:
    """Do the ISOLATED family slopes sum to the TRUE R_local(m) step?

    This is the independent closure test. `slope_census` distributes the true
    step by in-situ band share, so it closes by construction. Here the modeled
    step comes only from the isolated layer curves and the layer counts, and it
    is compared with the unperturbed w-leg forward step. A residual above
    `tolerance_ms` at any boundary is itself a finding: it names a boundary
    where in-situ behaviour differs from the sum of its isolated parts.
    """
    rows = reconciliation["per_width"]
    out = []
    for lo, hi in zip(rows, rows[1:]):
        key_lo, key_hi = str(lo["m"]), str(hi["m"])
        if key_lo not in w_table or key_hi not in w_table:
            continue
        modeled = hi["modeled_forward_ms"] - lo["modeled_forward_ms"]

        def forward(key: str) -> float:
            phases = w_table[key]["phase_ms"]
            return (phases["verify_build_us"]["value"]
                    + phases["eval_wall_us"]["value"])

        true_step = forward(key_hi) - forward(key_lo)
        per_band = {}
        for band in hi["bands"]:
            if band in lo["bands"]:
                per_band[band] = (hi["bands"][band]["modeled_ms"]
                                  - lo["bands"][band]["modeled_ms"])
        residual = modeled - true_step
        out.append({
            "boundary": f"{lo['m']}->{hi['m']}",
            "modeled_forward_step_ms": modeled,
            "true_w_leg_forward_step_ms": true_step,
            "residual_ms": residual,
            "residual_over_tolerance": abs(residual) > tolerance_ms,
            "modeled_band_step_ms": per_band,
        })
    return out


def post_band_ms(entry: dict) -> float | None:
    """lm_head + final norm + top-two: the forward total minus every band."""
    phases = entry["phase_ms"]
    if "verify_build_us" not in phases:
        return None
    forward = phases["verify_build_us"]["value"] + phases["eval_wall_us"]["value"]
    return forward - sum(phases[b]["value"] for b in BANDS if b in phases)


def finding571_allocation(reconciliation: dict, e186: dict,
                          total_ms: float, weight_pass_ms: float,
                          m: int = 9, kv: str = "1024") -> dict:
    """Allocate FINDING 571's non-QMV residual across census families.

    FINDING 543 measured +20.774 ms/round at m=9 for the third weight pass the
    staged (9,5) entry removes; FINDING 570 prices only `weight_pass_ms` of it
    as weight-stream work. The residual is charged here against two disjoint
    inventories at fixed m=9:

      * PASS-PLAN SENSITIVE work, which is inside the QMV kernels. Its cost
        depends on the group plan at the same width.
      * PASS-PLAN INVARIANT work: the GDN recurrence, SDPA, and the in-layer
        norm, convolution, gating, and activation remainder. Every one of those
        is a function of `m` and the KV length only. None of them is reached a
        different number of times when the group plan changes at fixed `m`.

    Whatever a census family can be charged must come from the first inventory.
    The invariant inventory is reported as the falsifiable bound: if any part of
    the residual really sat in a non-QMV family, that family's cost would have
    to depend on the group plan, which nothing in this census shows.
    """
    row = next((r for r in reconciliation["per_width"] if r["m"] == m), None)
    if row is None:
        return {}
    residual = total_ms - weight_pass_ms
    sensitive, invariant, allocation = {}, {}, {}
    for band, entry in row["bands"].items():
        sensitive[band] = entry["modeled_qmv_ms"]
        inv = entry["modeled_non_qmv_ms"] + entry["modeled_in_layer_other_ms"]
        if entry["insitu_minus_modeled_ms"] is not None:
            inv += entry["insitu_minus_modeled_ms"]
        invariant[band] = inv
        allocation[band] = 0.0
    sensitive_total = sum(sensitive.values())
    qmv_share = {b: (residual * v / sensitive_total if sensitive_total else None)
                 for b, v in sensitive.items()}
    return {
        "shape": f"m={m}, G-boundary, kv={kv}, harness=local",
        "finding_543_total_ms_per_round": total_ms,
        "finding_570_weight_pass_ms_per_round": weight_pass_ms,
        "non_qmv_residual_ms_per_round": residual,
        "pass_plan_sensitive_ms": sensitive,
        "pass_plan_sensitive_total_ms": sensitive_total,
        "pass_plan_invariant_ms": invariant,
        "pass_plan_invariant_total_ms": sum(invariant.values()),
        "allocation_to_non_qmv_families_ms": allocation,
        "allocated_to_non_qmv_total_ms": 0.0,
        "uncovered_by_non_qmv_families_ms": residual,
        "residual_placed_in": "QMV kernel internals at fixed width",
        "if_inside_qmv_by_band_ms": qmv_share,
        "basis": (
            "the invariant inventory is a function of m and kv only, so it "
            "cannot change when the group plan changes at fixed m; the band "
            "split of the residual under the QMV hypothesis is proportional, "
            "NOT causal"),
        "pass_count_contrast": pass_count_contrast(e186),
    }


def pass_count_contrast(e186: dict) -> dict:
    """What one extra weight pass costs, read across the 6->7 plan change.

    At m=6 the shipped plan runs every decode cell except `mlp.down` in the
    single-pass IPG=6 variant, so those six cells pay ONE weight pass. At m=7
    they return to the staged IPG=4 variant and pay TWO. The 6->7 isolated step
    therefore contains one extra weight pass plus one extra row, which upper
    bounds the pass alone. The nominal extra bytes are the second stream of the
    same weight tile.
    """
    curves = {k: {int(m): v for m, v in by_m.items()}
              for k, by_m in e186["curves_ms"].items()}
    rows = []
    for name, k, n, count in CELLS:
        curve = curves.get(f"qmv_inpath:{name}", {})
        if 6 not in curve or 7 not in curve:
            continue
        step = curve[7] - curve[6]
        rows.append({
            "cell": name,
            "count_per_forward": count,
            "groups_m6": cell_groups(6, name),
            "groups_m7": cell_groups(7, name),
            "adds_a_pass": cell_groups(7, name) > cell_groups(6, name),
            "isolated_step_ms": step,
            "round_scaled_step_ms": step * count,
            "extra_pass_weight_gb": k * n * BYTES_PER_ELEMENT / 1e9,
        })
    added = [r for r in rows if r["adds_a_pass"]]
    return {
        "cells": rows,
        "round_scaled_step_ms_cells_adding_a_pass": sum(
            r["round_scaled_step_ms"] for r in added),
        "extra_pass_weight_gb_cells_adding_a_pass": sum(
            r["extra_pass_weight_gb"] * r["count_per_forward"] for r in added),
    }


def cap7_truncated(rounds_by_width: dict) -> dict:
    """Model a cap-7 schedule from a cap-8 round census.

    At `segmentedVerifyDepthCap = 7` the widest legal round serves 8 rows, so
    m=9 has ZERO exposure. A round that served 9 rows at cap 8 serves 8 at cap
    7 and commits one token fewer, so the same generated text needs 9/8 as many
    such rounds. MODEL, not a measurement: the acceptance pattern inside the
    truncated round is assumed unchanged.
    """
    rounds = {int(m): float(v) for m, v in rounds_by_width.items()}
    moved = rounds.pop(9, 0.0) * 9.0 / 8.0
    rounds[8] = rounds.get(8, 0.0) + moved
    total = sum(rounds.values())
    return {str(m): v / total for m, v in sorted(rounds.items()) if total}


def cap7_median_lens(census: dict, targets: tuple[float, float] = (5.38, 6.09),
                     ) -> dict:
    """Model the cap-7 central order-statistic pair (FINDING 582).

    The published median is the mean of the two central raw ratios. The advisor
    reports the cap-7 central pair at mean served widths near `targets`. This
    census holds no hidden prompt, so each target is reproduced as the mixture
    of two locally measured cap-7-truncated prompt shapes whose means bracket
    it, and the lens is the equal average of the two mixtures. MODEL of an
    advisor-supplied central pair using local shape, not a measurement.
    """
    shapes = {}
    for name, prompt in census["prompts"].items():
        share = cap7_truncated(prompt["rounds_by_served_width"])
        mean = sum(int(m) * w for m, w in share.items())
        shapes[name] = (mean, share)
    lens: dict[str, float] = {}
    for target in targets:
        below = max((s for s in shapes.values() if s[0] <= target),
                    key=lambda s: s[0], default=None)
        above = min((s for s in shapes.values() if s[0] >= target),
                    key=lambda s: s[0], default=None)
        if below is None or above is None:
            continue
        if above[0] == below[0]:
            weight = 1.0
        else:
            weight = (above[0] - target) / (above[0] - below[0])
        for m in set(below[1]) | set(above[1]):
            lens[m] = lens.get(m, 0.0) + 0.5 * (
                weight * below[1].get(m, 0.0)
                + (1.0 - weight) * above[1].get(m, 0.0))
    return dict(sorted(lens.items(), key=lambda kv: int(kv[0])))


def census_pricing(families: dict, w_table: dict, census_path: pathlib.Path) -> dict:
    """Census-weighted ceiling of each family's width tax.

    The ceiling of family F is the round-weighted cost F carries ABOVE its
    width-1 level: sum over served widths of share(m) * (F(m) - F(1)). It is
    the local screening value of a mechanism that removed F's whole width tax
    and left the m=1 work in place. NOT-A-PRICE for ranked value (RULE 79).
    """
    census = json.loads(census_path.read_text())
    shares = {
        "pooled": census["pooled_all_prompts"]["share_by_served_width"],
        "public": census["prompts"]["public"]["share_by_served_width"],
        "cap7_pooled": cap7_truncated(
            census["pooled_all_prompts"]["rounds_by_served_width"]),
        "cap7_median_lens": cap7_median_lens(census),
    }
    anchor = "1"
    if anchor not in families:
        return {"error": "no m=1 anchor leg in this session"}

    names = list(BANDS) + ["lm_head_readout"]
    out: dict = {"census_source": str(census_path), "shares": shares,
                 "families": {}, "round_total": {}}
    for label, share in shares.items():
        total = 0.0
        for m, weight in share.items():
            if m in w_table and anchor in w_table:
                total += float(weight) * (
                    w_table[m]["phase_ms"]["round_us"]["value"]
                    - w_table[anchor]["phase_ms"]["round_us"]["value"])
        out["round_total"][label] = total
    for name in names:
        base = families[anchor]["scaled_family_ms"].get(name)
        if base is None:
            continue
        entry = {"m1_ms": base, "weighted_tax_ms": {}}
        for label, share in shares.items():
            total = 0.0
            for m, weight in share.items():
                if m in families and name in families[m]["scaled_family_ms"]:
                    total += float(weight) * (
                        families[m]["scaled_family_ms"][name] - base)
            entry["weighted_tax_ms"][label] = total
        out["families"][name] = entry
    for extra, key in [("draft_head", "draft_head_ms"), ("other_round", "other_round_ms")]:
        base = families[anchor][key]
        entry = {"m1_ms": base, "weighted_tax_ms": {}}
        for label, share in shares.items():
            total = 0.0
            for m, weight in share.items():
                if m in families:
                    total += float(weight) * (families[m][key] - base)
            entry["weighted_tax_ms"][label] = total
        out["families"][extra] = entry
    return out


def priced_shortlist(pricing: dict, e186: dict, kv: str = "1024") -> dict:
    """The three mechanisms this census nominates, priced under every weighting.

    Each entry is the round-weighted cost the mechanism carries ABOVE the m=1
    level, so it is the local screening ceiling of removing that whole width
    tax. NOT-A-PRICE for ranked value (RULE 79): no ranked leg is measured here,
    and the local serial-to-MTP ratio is not used at all.
    """
    fam = pricing["families"]
    labels = list(pricing["round_total"])
    shares = pricing["shares"]
    curves = {k: {int(m): v for m, v in by_m.items()}
              for k, by_m in e186["curves_ms"].items()}
    sdpa = curves.get(f"sdpa:kv={kv}", {})
    split_step = (sdpa.get(6, 0.0) - sdpa.get(5, 0.0)) * 16

    def combine(names: list[str]) -> dict:
        return {label: sum(fam[n]["weighted_tax_ms"][label] for n in names)
                for label in labels}

    items = [
        {
            "rank": 1,
            "mechanism": "fused MLP QMV width tax, all 64 layers",
            "families": ["band_gdn_mlp_us", "band_fa_mlp_us"],
            "cells": ["mlp.gate_up", "mlp.down"],
            "shape": "slope in m, f(IPG) convexity, one extra pass for "
                     "mlp.down only at the G boundary",
            "weighted_tax_ms": combine(["band_gdn_mlp_us", "band_fa_mlp_us"]),
        },
        {
            "rank": 2,
            "mechanism": "GDN mixer projection QMV width tax, 48 layers",
            "families": ["band_gdn_mixer_us"],
            "cells": ["gdn.in_proj", "gdn.out_proj"],
            "shape": "slope in m; the recurrence itself is nearly flat",
            "weighted_tax_ms": combine(["band_gdn_mixer_us"]),
        },
        {
            "rank": 3,
            "mechanism": f"SDPA two-call split at width>=6 (kv={kv}), 16 layers",
            "families": ["band_fa_mixer_us"],
            "cells": ["sdpa"],
            "shape": "pure STEP at m=6 and flat above it; no weight traffic",
            "isolated_step_ms_per_round": split_step,
            "weighted_tax_ms": {
                label: split_step * sum(
                    w for m, w in shares[label].items() if int(m) >= 6)
                for label in labels},
            "exposure_share_m_ge_6": {
                label: sum(w for m, w in shares[label].items() if int(m) >= 6)
                for label in labels},
        },
    ]
    return {
        "not_a_price_for_ranked_value": True,
        "rule": "RULE 79: local kernel-cost contrast at fixed schedule",
        "weightings": {label: {
            "mean_served_width": sum(int(m) * w for m, w in shares[label].items()),
            "share_m_ge_6": sum(w for m, w in shares[label].items()
                                if int(m) >= 6),
        } for label in labels},
        "items": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--e186", default=None)
    ap.add_argument("--census",
                    default="research/e215-artifacts/width-census-stepq.json")
    ap.add_argument("--out", default="research/e218-artifacts/width-tax-census.json")
    ap.add_argument("--kv", default="1024",
                    help="isolated attention KV length used for reconciliation")
    ap.add_argument("--f543-ms", type=float, default=20.774,
                    help="FINDING 543 paired m=9 third-weight-pass value")
    ap.add_argument("--f570-weight-pass-ms", type=float, default=5.253,
                    help="FINDING 570 weight-pass share of that value")
    args = ap.parse_args()

    legs: list[dict] = []
    for session in args.sessions:
        root = OUT_ROOT / session
        for leg_dir in sorted(root.iterdir()):
            if not leg_dir.is_dir():
                continue
            record = read_leg(leg_dir)
            if record:
                record["session_tag"] = session
                legs.append(record)
    if not legs:
        print("e218: no legs found", file=sys.stderr)
        return 2

    w_table = width_table(legs, band_sync=False)
    b_table = width_table(legs, band_sync=True)

    report: dict = {
        "experiment": "e218-width-tax-census",
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_or_ranked_score": False,
        "sessions": args.sessions,
        "noise_floor_ms": NOISE_FLOOR_MS,
        "identity": {
            "base_sha": sorted({leg["base_sha"] for leg in legs if leg["base_sha"]}),
            "worker_sha256": sorted({leg["worker_sha256"] for leg in legs
                                     if leg["worker_sha256"]}),
            "head_provenance_sha256": sorted(
                {leg["head_provenance_sha256"] for leg in legs
                 if leg.get("head_provenance_sha256")}),
            "host": sorted({leg["host"] for leg in legs if leg["host"]}),
            "chip": sorted({leg["chip"] for leg in legs if leg["chip"]}),
            "all_legs_exact": all(leg["all_tokens_matched"] is True for leg in legs),
            "residual_divergence_total": sum(
                leg["residual_divergence_count"] or 0 for leg in legs),
            "digest_stable_all_legs": all(
                leg["worker_digest_stable"] == "true" for leg in legs),
        },
        "legs": legs,
        "width_table_untraced": w_table,
        "width_table_band_arm": b_table,
        "round_steps_untraced": steps(w_table, ROUND_PHASES + ["round_us"]),
        "band_steps": steps(b_table, BANDS + ["round_us", "eval_wall_us"]),
        "perturbation": perturbation(w_table, b_table),
    }
    report["family_census"] = family_census(w_table, b_table)
    report["slope_census"] = slope_census(report["family_census"], w_table)
    report["shapes"] = {
        "round_us": shape_report(w_table, "round_us"),
        **{f"{b}_per_layer": shape_report(b_table, b, per_layer=True)
           for b in BANDS},
    }

    census_path = pathlib.Path(args.census)
    if census_path.exists():
        report["census_pricing"] = census_pricing(
            report["family_census"], w_table, census_path)

    if args.e186:
        e186 = load_e186(pathlib.Path(args.e186))
        report["e186"] = e186
        report["e186_decomposition"] = cell_decomposition(e186)
        recon = insitu_reconciliation(e186, b_table, w_table, kv=args.kv)
        report["insitu_reconciliation"] = recon
        report["slope_closure"] = slope_closure(recon, w_table)
        report["finding571_allocation"] = finding571_allocation(
            recon, e186, total_ms=args.f543_ms,
            weight_pass_ms=args.f570_weight_pass_ms, kv=args.kv)
        if "census_pricing" in report:
            report["priced_shortlist"] = priced_shortlist(
                report["census_pricing"], e186, kv=args.kv)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=False))
    print(f"e218: wrote {out_path}")

    print("\nR_local(m) — untraced pinned-width legs (harness=local, ungated)")
    print(f"{'m':>3} {'round ms':>10} {'draft':>8} {'vbuild':>8} {'eval':>8} "
          f"{'other':>7} {'spread':>7} {'rounds':>7}")
    for key in sorted(w_table, key=int):
        e = w_table[key]["phase_ms"]
        other = (e["round_us"]["value"] - e["draft_build_us"]["value"]
                 - e["verify_build_us"]["value"] - e["eval_wall_us"]["value"])
        print(f"{key:>3} {e['round_us']['value']:10.3f} "
              f"{e['draft_build_us']['value']:8.3f} "
              f"{e['verify_build_us']['value']:8.3f} "
              f"{e['eval_wall_us']['value']:8.3f} {other:7.3f} "
              f"{e['round_us']['leg_spread']:7.3f} "
              f"{w_table[key]['rounds_at_width']:7d}")

    if b_table:
        print("\nband arm — per-round device time by layer family (ATTRIBUTION ONLY)")
        header = f"{'m':>3} " + " ".join(f"{b.replace('band_', '').replace('_us', ''):>10}"
                                         for b in BANDS) + f" {'post':>8} {'round':>9}"
        print(header)
        for key in sorted(b_table, key=int):
            e = b_table[key]["phase_ms"]
            row = f"{key:>3} " + " ".join(
                f"{e[b]['value']:10.3f}" if b in e else f"{'-':>10}" for b in BANDS)
            print(row + f" {e['eval_wall_us']['value']:8.3f}"
                        f" {e['round_us']['value']:9.3f}")

    if report["slope_census"]:
        print("\nfamily dR/dm census (true round step distributed by in-situ band share)")
        for row in report["slope_census"]:
            parts = " ".join(
                f"{k.replace('band_', '').replace('_us', '')}={v:6.2f}"
                for k, v in row["family_step_ms"].items())
            print(f"  {row['boundary']:>5} step {row['true_round_step_ms']:7.3f} "
                  f"| {parts} head={row['draft_head_step_ms']:5.2f} "
                  f"other={row['other_round_step_ms']:5.2f} "
                  f"| residual {row['closure_residual_ms']:6.3f}")

    recon = report.get("insitu_reconciliation")
    if recon:
        print(f"\nabsolute-level reconciliation ({recon['unit']}, "
              f"kv={recon['kv_used']})")
        print(f"{'m':>3} {'modeled':>9} {'band arm':>9} {'ratio':>7} {'<=15%':>6} "
              f"{'w leg':>8} {'b/w':>6}   per-band modeled/in-situ")
        for row in recon["per_width"]:
            bands = " ".join(
                f"{b.replace('band_', '').replace('_us', '')}="
                f"{(e['modeled_over_insitu'] or float('nan')):.3f}"
                for b, e in row["bands"].items())
            print(f"{row['m']:>3} {row['modeled_forward_ms']:9.3f} "
                  f"{row['insitu_forward_band_arm_ms']:9.3f} "
                  f"{row['modeled_over_insitu_band_arm']:7.3f} "
                  f"{str(row['within_15pct']):>6} "
                  f"{(row['unperturbed_w_forward_ms'] or float('nan')):8.3f} "
                  f"{(row['band_arm_over_w_forward'] or float('nan')):6.3f}   {bands}")

    closure = report.get("slope_closure")
    if closure:
        print("\nindependent slope closure: isolated layer model vs the true "
              "w-leg forward step (tolerance 2 ms/round)")
        print(f"  {'boundary':>8} {'modeled':>9} {'true':>9} {'residual':>9} "
              f"{'over':>5}   per-band modeled step")
        for row in closure:
            bands = " ".join(
                f"{b.replace('band_', '').replace('_us', '')}={v:6.2f}"
                for b, v in row["modeled_band_step_ms"].items())
            print(f"  {row['boundary']:>8} {row['modeled_forward_step_ms']:9.3f} "
                  f"{row['true_w_leg_forward_step_ms']:9.3f} "
                  f"{row['residual_ms']:9.3f} "
                  f"{str(row['residual_over_tolerance']):>5}   {bands}")

    alloc = report.get("finding571_allocation")
    if alloc:
        print(f"\nFINDING 571 allocation at {alloc['shape']}: "
              f"{alloc['non_qmv_residual_ms_per_round']:.3f} ms/round "
              f"non-QMV residual of {alloc['finding_543_total_ms_per_round']:.3f}")
        print(f"  {'family':>14} {'pass-sensitive':>15} {'pass-invariant':>15} "
              f"{'allocated':>10} {'if in QMV':>10}")
        for band in alloc["pass_plan_sensitive_ms"]:
            print(f"  {band.replace('band_', '').replace('_us', ''):>14} "
                  f"{alloc['pass_plan_sensitive_ms'][band]:15.3f} "
                  f"{alloc['pass_plan_invariant_ms'][band]:15.3f} "
                  f"{alloc['allocation_to_non_qmv_families_ms'][band]:10.3f} "
                  f"{(alloc['if_inside_qmv_by_band_ms'][band] or float('nan')):10.3f}")
        print(f"  sensitive total {alloc['pass_plan_sensitive_total_ms']:.3f} ms, "
              f"invariant total {alloc['pass_plan_invariant_total_ms']:.3f} ms, "
              f"non-QMV families cover "
              f"{alloc['allocated_to_non_qmv_total_ms']:.3f} ms, uncovered "
              f"{alloc['uncovered_by_non_qmv_families_ms']:.3f} ms")
        pc = alloc["pass_count_contrast"]
        print("\n  6->7 plan contrast: one extra weight pass plus one row")
        for r in pc["cells"]:
            print(f"    {r['cell']:>14} G {r['groups_m6']}->{r['groups_m7']} "
                  f"x{r['count_per_forward']:>3}  isolated step "
                  f"{r['isolated_step_ms']:7.4f} ms  round-scaled "
                  f"{r['round_scaled_step_ms']:7.3f} ms  extra pass "
                  f"{r['extra_pass_weight_gb']:.4f} GB")
        print(f"    cells adding a pass: "
              f"{pc['round_scaled_step_ms_cells_adding_a_pass']:.3f} ms/round "
              f"for {pc['extra_pass_weight_gb_cells_adding_a_pass']:.3f} GB "
              f"of extra nominal weight traffic")

    pricing = report.get("census_pricing")
    if pricing:
        print("\ncensus-weighted width tax above the m=1 level (NOT A PRICE for "
              "ranked value, RULE 79)")
        keys = list(next(iter(pricing["families"].values()))["weighted_tax_ms"])
        print(f"  {'family':>16} {'m=1 ms':>9} "
              + " ".join(f"{k:>12}" for k in keys))
        for fam, entry in pricing["families"].items():
            print(f"  {fam.replace('band_', '').replace('_us', ''):>16} "
                  f"{entry['m1_ms']:9.3f} "
                  + " ".join(f"{entry['weighted_tax_ms'][k]:12.3f}" for k in keys))
        print(f"  {'ROUND TOTAL':>16} {'':>9} "
              + " ".join(f"{pricing['round_total'][k]:12.3f}" for k in keys))
        print(f"  {'mean served m':>16} {'':>9} "
              + " ".join(
                  f"{sum(int(m) * w for m, w in pricing['shares'][k].items()):12.3f}"
                  for k in keys))

    shortlist = report.get("priced_shortlist")
    if shortlist:
        print("\npriced shortlist (NOT A PRICE for ranked value, RULE 79)")
        for item in shortlist["items"]:
            print(f"  {item['rank']}. {item['mechanism']}")
            print(f"     {item['shape']}")
            print("     " + "  ".join(
                f"{label}={value:.3f} ms/round"
                for label, value in item["weighted_tax_ms"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
