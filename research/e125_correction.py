#!/usr/bin/env python3
"""E125 Stage 2 - the isolated-to-in-situ correction as a function of the
achieved bandwidth and the distance to the roofline.

Stage 1 moved the memory regime of one compiled kernel at a fixed shape, a
fixed width and a fixed launched grid volume. This file turns that frame axis
into a correction that converts an isolated per-instruction price into the
price the same instruction pays in situ.

Three separate things are kept apart on purpose:

  W  the width term. A ranked-percent curve that is convex in the realised
     wide-QMV width is read at the mean width instead of averaged over the
     realised width histogram. Stage 0 measured W = 1.33 [1.00, 1.76] from
     published numbers. E125 does not re-measure W.
  F  the frame term. The same instruction is priced in a different memory
     regime. E125 measures F.
  C  the composite, C = W x F.

Every number here is local, ungated evidence from one M4 Pro. None of it is a
ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ART = Path("research/e125-artifacts")
PEAK_BANDWIDTH_GB_S = 273.0

# Mechanism classes that Stage 1 prices. `deletion` is the class the shipped
# candidate work belongs to: an add tree or an exchange is removed.
CLASSES = ("ld", "alu", "deletion")

# Published isolated-to-in-situ pairs. Both predate this file.
#   value = shipped isolated ranked %, measured in-situ ranked %, and the
#   achieved GB/s of the isolated cell the shipped number was read from.
PUBLISHED_PAIRS = {
    "alphonse_threadgroup_exchange": {
        "isolated_pct": 0.890,
        "in_situ_pct": 0.436,
        "in_situ_sd": 0.093,
        "mechanism_class": "deletion",
        "isolated_cell": "fa_qkv_k5120_n14336|m4",
        "note": "threadgroup exchange removed from the wide-QMV inner loop",
    },
    "rule58_launched_volume": {
        "isolated_pct": 22.75,
        "in_situ_pct": 13.12,
        "in_situ_sd": None,
        "mechanism_class": "deletion",
        "isolated_cell": None,
        "note": "launched-volume / grouping class, cell not published",
    },
}

# Stage 0 registered inputs, restated so this file fails loudly if the Stage 0
# artifact moves under it.
W_CENTRAL = 1.33
W_BAND = (1.00, 1.76)

# Excluded from every smooth fit: at width 5 the wide-QMV kernel falls off an
# occupancy cliff and the injected-instruction price jumps by more than an
# order of magnitude. A cliff is not a roofline.
CLIFF_PCT = 40.0


def load(name: str) -> dict:
    p = ART / name
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# A. why the isolated grid cannot supply its own transfer law
# --------------------------------------------------------------------------

def _slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def confound_audit(scan: dict) -> dict:
    """The E123 grid correlates price with achieved GB/s only across widths.

    Inside a fixed width the sign is not stable, so the pooled slope is a
    between-width effect wearing a bandwidth label. That is the reason the
    isolated grid cannot be asked for its own frame correction, and the reason
    Stage 1 had to move the regime at a fixed width.
    """
    cells = scan["cells"]
    arms = sorted({a for c in cells.values() for a in c["pct"]})
    out = {}
    for arm in arms:
        pooled_x, pooled_y = [], []
        by_width: dict[int, tuple[list, list]] = {}
        for key, cell in cells.items():
            if arm not in cell["pct"]:
                continue
            m = int(key.split("|m")[1])
            x, y = cell["gbs"], cell["pct"][arm]
            pooled_x.append(x)
            pooled_y.append(y)
            bx, by = by_width.setdefault(m, ([], []))
            bx.append(x)
            by.append(y)
        pooled = _slope(pooled_x, pooled_y)
        within = {m: _slope(bx, by) for m, (bx, by) in sorted(by_width.items())}
        signs = {m: (0 if s is None else (1 if s > 0 else -1))
                 for m, s in within.items()}
        nonzero = [s for s in signs.values() if s]
        out[arm] = {
            "pooled_slope_pct_per_gbs": pooled,
            "within_width_slope_pct_per_gbs": within,
            "within_width_sign": signs,
            "sign_stable_within_width": len(set(nonzero)) <= 1 if nonzero
                                        else None,
            "pooled_sign_matches_all_within":
                pooled is not None and bool(nonzero)
                and all(s == (1 if pooled > 0 else -1) for s in nonzero),
        }
    n_arms = len(out)
    n_stable = sum(1 for v in out.values() if v["sign_stable_within_width"])
    return {
        "source": scan["source"],
        "by_arm": out,
        "n_arms": n_arms,
        "n_arms_sign_stable_within_width": n_stable,
        "identified": n_stable == n_arms,
        "note":
            "The pooled achieved-GB/s slope in the isolated grid is confounded "
            "with the width axis. Where the within-width sign is not stable, "
            "the pooled slope is a between-width effect and cannot be read as "
            "a bandwidth law.",
    }


# --------------------------------------------------------------------------
# B. the exposure law fitted on the frame axis
# --------------------------------------------------------------------------

def exposure(mu: float, p: float) -> float:
    """Share of an added instruction's own time that reaches the wall clock.

    A power-blend roofline, t = (t_mem^p + t_alu^p)^(1/p), gives
    dt/dt_alu = (t_alu/t)^(p-1) = (1 - mu^p)^((p-1)/p) with mu = t_mem/t.
    p = 1 is a fully additive machine that hides nothing. Large p is a hard
    max that hides everything until the compute side overtakes memory.
    """
    mu = min(max(mu, 0.0), 0.999999)
    inner = 1.0 - mu ** p
    if inner <= 0.0:
        return 0.0
    return inner ** ((p - 1.0) / p)


def frame_points(law: dict, klass: str) -> list[dict]:
    """One point per (shape, m, frame) for one mechanism class."""
    pts = []
    for pt in law["frame_law"].get(klass, {}).get("points", []):
        val = pt.get("value_us_per_k_block")
        if val is None or not math.isfinite(val):
            continue
        pts.append(pt)
    return pts


def mu_of(cell: dict) -> float | None:
    """Memory-bound fraction.

    Achieved bandwidth alone is not enough under contention: a consumer thread
    lowers the achieved GB/s of the probe while making the probe *more* memory
    bound, so the two frames move the achieved rate the same way and the
    exposure opposite ways. The consumer frame is therefore priced against the
    share of the bus the consumer left, which is what `phi_eff` already is.
    """
    if cell["frame"] == "consumer":
        return cell.get("phi_eff")
    return cell.get("phi_arm")


def fit_exposure(law: dict, klass: str, drop_frames: tuple = (),
                 drop_shapes: tuple = ()) -> dict:
    """Shared exposure exponent p, free amplitude per (shape, m).

    The amplitude carries the unhidden per-instruction price, which depends on
    shape and width. Only the *shape* of the phi dependence is shared, so the
    fit cannot borrow strength from an amplitude difference.
    """
    cells = {(c["shape"], c["m"], c["frame"]): c for c in law["cells"]}
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for pt in frame_points(law, klass):
        if pt["frame"] in drop_frames or pt["shape"] in drop_shapes:
            continue
        cell = cells.get((pt["shape"], pt["m"], pt["frame"]))
        if cell is None:
            continue
        mu = mu_of(cell)
        val = pt["value_us_per_k_block"]
        if mu is None or val is None or val <= 0:
            continue
        if abs(pt.get("pct") or 0.0) > CLIFF_PCT:
            continue
        groups.setdefault((pt["shape"], pt["m"]), []).append((mu, val))

    usable = {g: v for g, v in groups.items()
              if len({round(mu, 4) for mu, _ in v}) >= 3}
    if len(usable) < 2:
        return {"identified": False, "n_groups": len(usable),
                "note": "not measured: fewer than two groups carry three "
                        "distinct memory regimes for this class"}

    def sse(p: float) -> float:
        total = 0.0
        for pts in usable.values():
            logs = []
            for mu, val in pts:
                e = exposure(mu, p)
                if e <= 0:
                    return float("inf")
                logs.append(math.log(val) - math.log(e))
            amp = statistics.fmean(logs)
            total += sum((v - amp) ** 2 for v in logs)
        return total

    grid = [1.0 + 0.05 * i for i in range(0, 60)]
    grid += [4.0 + 0.5 * i for i in range(1, 57)]
    scored = [(sse(p), p) for p in grid]
    scored = [(s, p) for s, p in scored if math.isfinite(s)]
    if not scored:
        return {"identified": False, "n_groups": len(usable),
                "note": "not measured: no finite exposure fit"}
    best_sse, best_p = min(scored)

    n = sum(len(v) for v in usable.values())
    dof = max(n - len(usable) - 1, 1)
    # profile band: every p whose sse stays inside an F-like 1-sigma bowl
    thresh = best_sse * (1.0 + 1.0 / dof)
    inside = [p for s, p in scored if s <= thresh]

    flat = sse(1.0)
    amps = {}
    for g, pts in usable.items():
        amps["|".join(map(str, g))] = math.exp(statistics.fmean(
            [math.log(v) - math.log(exposure(mu, best_p)) for mu, v in pts]))

    return {
        "identified": True,
        "p": best_p,
        "p_band": [min(inside), max(inside)] if inside else [best_p, best_p],
        "sse": best_sse,
        "sse_flat_law": flat,
        "beats_flat_law": best_sse < flat,
        "variance_explained_vs_flat":
            None if flat <= 0 else 1.0 - best_sse / flat,
        "n_points": n,
        "n_groups": len(usable),
        "groups": sorted("|".join(map(str, g)) for g in usable),
        "unhidden_price_us_per_k_block": amps,
        "mu_range": [min(mu for v in usable.values() for mu, _ in v),
                     max(mu for v in usable.values() for mu, _ in v)],
    }


def hold_out(law: dict, klass: str) -> dict:
    """Leave one frame out, then leave one shape out."""
    frames = sorted({pt["frame"] for pt in frame_points(law, klass)})
    shapes = sorted({pt["shape"] for pt in frame_points(law, klass)})
    full = fit_exposure(law, klass)
    if not full.get("identified"):
        return {"identified": False, "note": full.get("note")}
    rows = []
    for f in frames:
        sub = fit_exposure(law, klass, drop_frames=(f,))
        if sub.get("identified"):
            rows.append({"held_out": f, "kind": "frame", "p": sub["p"],
                         "p_shift": sub["p"] - full["p"]})
    for s in shapes:
        sub = fit_exposure(law, klass, drop_shapes=(s,))
        if sub.get("identified"):
            rows.append({"held_out": s, "kind": "shape", "p": sub["p"],
                         "p_shift": sub["p"] - full["p"]})
    shifts = [abs(r["p_shift"]) for r in rows]
    return {
        "identified": True,
        "full_p": full["p"],
        "rows": rows,
        "max_abs_p_shift": max(shifts) if shifts else None,
        "stable": bool(shifts) and max(shifts) <= 0.5 * full["p"],
    }


# --------------------------------------------------------------------------
# C. the empirical correction table, the headline deliverable
# --------------------------------------------------------------------------

def correction_table(law: dict) -> dict:
    """Correction factor against the achieved bandwidth of the isolated cell.

    For every mechanism class and every (shape, m) group the base frame is the
    isolated reference. Each other frame gives one measured ratio
    price(reference) / price(frame). The table reports that ratio against the
    achieved GB/s and the roofline distance of the reference cell.
    """
    cells = {(c["shape"], c["m"], c["frame"]): c for c in law["cells"]}
    out = {}
    for klass in CLASSES:
        rows = []
        for pt in frame_points(law, klass):
            if pt["frame"] == "base":
                continue
            ref = cells.get((pt["shape"], pt["m"], "base"))
            cur = cells.get((pt["shape"], pt["m"], pt["frame"]))
            if ref is None or cur is None:
                continue
            base_pt = next((q for q in frame_points(law, klass)
                            if q["shape"] == pt["shape"] and q["m"] == pt["m"]
                            and q["frame"] == "base"), None)
            if base_pt is None:
                continue
            a = base_pt["value_us_per_k_block"]
            b = pt["value_us_per_k_block"]
            if not a or not b:
                continue
            rows.append({
                "shape": pt["shape"], "m": pt["m"], "frame": pt["frame"],
                "isolated_gb_s": ref["achieved_gb_s"],
                "isolated_roofline_distance":
                    1.0 - ref["achieved_gb_s"] / PEAK_BANDWIDTH_GB_S,
                "frame_gb_s": cur["achieved_gb_s"],
                "frame_roofline_distance":
                    1.0 - cur["achieved_gb_s"] / PEAK_BANDWIDTH_GB_S,
                "frame_mu": mu_of(cur),
                "isolated_price_us_per_k_block": a,
                "frame_price_us_per_k_block": b,
                "transfer_factor": a / b,
            })
        factors = [r["transfer_factor"] for r in rows]
        out[klass] = {
            "rows": rows,
            "n_independent_points": len(rows),
            "measured": len(rows) >= 2,
            "median_transfer_factor":
                statistics.median(factors) if factors else None,
            "transfer_factor_range":
                [min(factors), max(factors)] if factors else None,
            "note": None if len(rows) >= 2 else
                    "not measured: fewer than two independent frame points",
        }
    return out


# --------------------------------------------------------------------------
# D. calibrate the in-situ regime against the published transfer pairs
# --------------------------------------------------------------------------

def invert_published(fit: dict, scan: dict) -> dict:
    """Solve each published pair for the in-situ memory-bound fraction.

    One pair calibrates the in-situ point, the other is held out. If the two
    invert to the same regime the law carries; if they do not, the frame term
    is not a single scalar and the correction must stay a band.
    """
    if not fit.get("identified"):
        return {"identified": False,
                "note": "not measured: the exposure law is not identified"}
    p = fit["p"]
    out = {}
    for name, pair in PUBLISHED_PAIRS.items():
        cell = scan["cells"].get(pair["isolated_cell"] or "")
        if cell is None:
            out[name] = {"identified": False,
                         "note": "not measured: the isolated cell of this "
                                 "pair was never published"}
            continue
        mu_iso = cell["roofline_frac"]
        # observed total transfer already carries W, so divide it out first
        f_obs = (pair["isolated_pct"] / pair["in_situ_pct"]) / W_CENTRAL
        target = exposure(mu_iso, p) / f_obs
        lo, hi = 0.0, 0.999999
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if exposure(mid, p) > target:
                lo = mid
            else:
                hi = mid
        mu_situ = 0.5 * (lo + hi)
        out[name] = {
            "identified": True,
            "isolated_cell": pair["isolated_cell"],
            "isolated_gb_s": cell["gbs"],
            "isolated_mu": mu_iso,
            "observed_total_transfer": pair["isolated_pct"] / pair["in_situ_pct"],
            "frame_term_after_removing_W": f_obs,
            "implied_in_situ_mu": mu_situ,
            "implied_in_situ_gb_s": mu_situ * PEAK_BANDWIDTH_GB_S,
        }
    solved = [v["implied_in_situ_mu"] for v in out.values()
              if v.get("identified")]
    return {
        "identified": bool(solved),
        "pairs": out,
        "n_solved": len(solved),
        "consistent": (len(solved) >= 2
                       and max(solved) - min(solved) <= 0.10),
        "in_situ_mu_point": statistics.fmean(solved) if solved else None,
        "in_situ_mu_band": [min(solved), max(solved)] if solved else None,
        "note": "One published pair is enough to calibrate the in-situ point. "
                "A second pair is a held-out check, not extra precision.",
    }


# --------------------------------------------------------------------------
# E. apply the correction
# --------------------------------------------------------------------------

def apply_to_route_b(pre: dict, table: dict, fit: dict, inv: dict) -> dict:
    isolated = pre["isolated_ranked_recomputed_on_7x7"]

    measured = table.get("deletion", {})
    f_points = []
    if measured.get("measured"):
        f_points.append(("frame_axis_median",
                         measured["median_transfer_factor"]))
    if inv.get("identified") and inv.get("in_situ_mu_point") is not None \
            and fit.get("identified"):
        mu_situ = inv["in_situ_mu_point"]
        pair = inv["pairs"].get("alphonse_threadgroup_exchange", {})
        mu_iso = pair.get("isolated_mu")
        if mu_iso is not None:
            f_points.append(("published_inversion",
                             exposure(mu_iso, fit["p"])
                             / exposure(mu_situ, fit["p"])))

    if not f_points:
        reg = pre["predictions"]["primary"]
        return {
            "identified": False,
            "note": "not measured: no frame term is identified, so the Stage 0 "
                    "registered band stands unchanged",
            "sources": {},
            "F_point": reg["F"], "F_band": [1.00, 2.041284403669725],
            "W_point": W_CENTRAL, "W_band": list(W_BAND),
            "C_point": reg["C"], "C_band": list(pre["envelope"]),
            "isolated_ranked_pct": isolated,
            "point": reg["ranked_pct"],
            "low": pre["envelope"][0], "high": pre["envelope"][1],
            "registered_point": reg["ranked_pct"],
            "parity_line_pct": pre["decision_lines"]["parity"],
            "mode_proof_line_pct": pre["decision_lines"]["mode_proof"],
        }

    vals = [v for _, v in f_points]
    f_point = math.exp(statistics.fmean([math.log(v) for v in vals]))
    f_lo, f_hi = min(vals), max(vals)
    if f_lo == f_hi:
        f_lo, f_hi = f_lo / 1.2, f_hi * 1.2

    c_point = W_CENTRAL * f_point
    c_lo = W_BAND[0] * f_lo
    c_hi = W_BAND[1] * f_hi
    return {
        "identified": True,
        "sources": dict(f_points),
        "F_point": f_point, "F_band": [f_lo, f_hi],
        "W_point": W_CENTRAL, "W_band": list(W_BAND),
        "C_point": c_point, "C_band": [c_lo, c_hi],
        "isolated_ranked_pct": isolated,
        "point": isolated / c_point,
        "low": isolated / c_hi,
        "high": isolated / c_lo,
        "registered_point": pre["predictions"]["primary"]["ranked_pct"],
        "parity_line_pct": pre["decision_lines"]["parity"],
        "mode_proof_line_pct": pre["decision_lines"]["mode_proof"],
    }


def back_check(route_b: dict) -> dict:
    pair = PUBLISHED_PAIRS["alphonse_threadgroup_exchange"]
    c = route_b["C_point"]
    pred = pair["isolated_pct"] / c
    sd = pair["in_situ_sd"]
    return {
        "identified": bool(route_b.get("identified")),
        "composite_source": ("E125 measured frame term"
                             if route_b.get("identified")
                             else "Stage 0 registered composite, unchanged"),
        "predicted_pct": pred,
        "measured_pct": pair["in_situ_pct"],
        "sd": sd,
        "sd_away": abs(pred - pair["in_situ_pct"]) / sd if sd else None,
        "note": "The composite C is applied to alphonse's shipped isolated "
                "number. The alphonse pair also calibrates the in-situ point, "
                "so this is a consistency check of the composite, not an "
                "independent test.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default=str(ART / "frame-law.json"))
    ap.add_argument("--out", default=str(ART / "correction.json"))
    args = ap.parse_args()

    law = json.loads(Path(args.law).read_text())
    scan = load("e123-bandwidth-scan.json")
    pre = load("routeb-prediction.json")

    audit = confound_audit(scan)
    table = correction_table(law)
    fits = {k: fit_exposure(law, k) for k in CLASSES}
    holds = {k: hold_out(law, k) for k in CLASSES}
    best = fits.get("deletion") if fits.get("deletion", {}).get("identified") \
        else next((f for f in fits.values() if f.get("identified")), {})
    inv = invert_published(best, scan)
    route_b = apply_to_route_b(pre, table, best, inv)
    bc = back_check(route_b)

    class_rows = []
    for klass in CLASSES:
        t = table[klass]
        class_rows.append({
            "mechanism_class": klass,
            "factor": t["median_transfer_factor"],
            "low": t["transfer_factor_range"][0] if t["transfer_factor_range"]
                   else None,
            "high": t["transfer_factor_range"][1] if t["transfer_factor_range"]
                    else None,
            "n_independent_points": t["n_independent_points"],
            "measured": t["measured"],
            "evidence": ("Stage 1 frame axis, base frame against every other "
                         "frame at a fixed shape and a fixed width")
                        if t["measured"] else t["note"],
        })

    out = {
        "experiment": "e125",
        "stage": 2,
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "device": law.get("device"),
        "architecture": law.get("architecture"),
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "session_valid": law.get("gates", {}).get("session_valid"),
        "form": "predicted in-situ = isolated / (W x F); "
                "F = exposure(mu_isolated) / exposure(mu_in_situ); "
                "exposure(mu) = (1 - mu^p)^((p-1)/p); "
                "mu = memory-bound time fraction, taken as the achieved "
                "bandwidth over the bus share the kernel actually had",
        "isolated_grid_confound": audit,
        "exposure_fits": fits,
        "hold_out": holds,
        "fitted_points": best.get("n_points"),
        "held_out_points": (holds.get("deletion", {}) or {}).get("rows"),
        "correction_table": table,
        "class_table": class_rows,
        "published_pair_inversion": inv,
        "route_b": route_b,
        "alphonse_back_check": bc,
        "e124": {
            "low": 0.0, "high": 0.0, "factor": 1.0,
            "note": "no correction. Rule 69: E124 was measured in situ, so "
                    "the frame term is already paid and applying it again "
                    "would double count.",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")

    print(f"\nisolated-grid confound: "
          f"{audit['n_arms_sign_stable_within_width']}/{audit['n_arms']} arms "
          f"keep one sign inside a fixed width "
          f"(identified={audit['identified']})")
    for row in class_rows:
        if row["measured"]:
            print(f"  {row['mechanism_class']:9s} F={row['factor']:.3f} "
                  f"[{row['low']:.3f}, {row['high']:.3f}]  "
                  f"n={row['n_independent_points']}")
        else:
            print(f"  {row['mechanism_class']:9s} {row['evidence']}")
    if route_b.get("identified"):
        print(f"\nroute B corrected ranked: {route_b['point']:+.3f} % "
              f"[{route_b['low']:+.3f}, {route_b['high']:+.3f}]  "
              f"(registered {route_b['registered_point']:+.3f} %)")
        print(f"  parity {route_b['parity_line_pct']:+.2f} %  "
              f"mode proof {route_b['mode_proof_line_pct']:+.2f} %")
    if bc.get("identified"):
        print(f"back check: predicted {bc['predicted_pct']:.3f} % vs measured "
              f"{bc['measured_pct']:.3f} % "
              f"({bc['sd_away']:.2f} sd)" if bc.get("sd_away") is not None
              else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
