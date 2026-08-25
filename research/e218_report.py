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
        # In a band-sync leg every band drains, so the forward's device time is
        # the band sum plus whatever the final blocking eval still holds.
        post = b["phase_ms"]["eval_wall_us"]["value"]
        forward_b = sum(bands.values()) + post
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--e186", default=None)
    ap.add_argument("--census",
                    default="research/e215-artifacts/width-census-stepq.json")
    ap.add_argument("--out", default="research/e218-artifacts/width-tax-census.json")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
