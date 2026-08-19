#!/usr/bin/env python3
"""E42: divide an injected bit-exact regression by its own kernel cost.

    research/e42_analyze.py --arms base p2L1 p2L2 p6L1 p6L2 m1L1 base2 [--wandb]

Reads, per arm:
  .mlxfast-private/e42/runs/<arm>/score-*.json          leg scalars
  .mlxfast-private/e42/runs/<arm>/reports/leg-*/0{3,4}-mtp-timed.json
  .mlxfast-private/qmv-curve/e42-<arm>/summary.json     per-width kernel cost

and reports:

  x(M)      per-width kernel slowdown vs base, from the cost curve.
  psi_eff   (dT/T) / x_bar on the MTP leg, x_bar QMV-time-weighted over the
            arm's treated widths using the arm's own round-width histogram.
            THE PRIMARY ANSWER.
  phi_local psi_eff(p6) / psi_eff(p2).
  alpha     dT_measured / dQ_predicted. psi_eff = alpha * Q_treated/T, so this
            separates the marginal share (what an optimisation is actually worth)
            from the occupancy share (what the kernel timer says it holds).
  linearity psi_eff(L1) vs psi_eff(L2) per arm. Disagreement means no single psi.

Every leg here ran with MLXFAST_LOCAL_COOL_GATE=0. Nothing printed is
gate-qualified or an official score.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / ".mlxfast-private/e42/runs"
CURVES = ROOT / ".mlxfast-private/qmv-curve"

# Width gate per arm, as spliced by research/e42_perturb.py.
TREATED = {
    "p2": set(range(2, 10)),
    "p6": set(range(6, 10)),
    "m1": {1},
}
# The draft-side compact vocabulary readout is bits=2 and M=1 only, so it is
# outside every arm's affine4 gate and outside the verify mix.
DRAFT_SIDE_SHAPE = "head.compact_draft_vocab"
# Cross-build reproduction tolerance for an untreated calibration cell.
UNTREATED_DRIFT_TOL = 0.02


def arm_family(arm: str) -> str:
    for fam in TREATED:
        if arm.startswith(fam):
            return fam
    return "base"


def arm_level(arm: str) -> int:
    tail = arm.split("L")[-1]
    return int(tail) if tail.isdigit() else 0


def load_legs(arm: str) -> list[dict]:
    """One entry per --local-iterate invocation: serial control + MTP leg."""
    d = RUNS / arm
    legs = []
    for score_path in sorted(d.glob("score-*.json")):
        idx = score_path.stem.split("-")[-1]
        reports = d / "reports" / f"leg-{idx}"
        timed = sorted(reports.glob("0*-mtp-timed.json"))
        serial = mtp = None
        for path in timed:
            payload = json.loads(path.read_text())
            if payload.get("is_serial_control"):
                serial = payload
            else:
                mtp = payload
        if serial is None or mtp is None:
            raise SystemExit(f"e42_analyze: {arm} leg {idx} missing a timed report")
        legs.append(
            {
                "index": idx,
                "score": json.loads(score_path.read_text()),
                "serial": serial,
                "mtp": mtp,
            }
        )
    if not legs:
        raise SystemExit(f"e42_analyze: no legs under {d}")
    return legs


def load_curve(arm: str) -> dict | None:
    path = CURVES / f"e42-{arm}" / "vendored.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    # shape name -> {"calls_per_verify": n, "widths": {m: seconds_per_call}}
    out = {}
    for shape in payload["shapes"]:
        widths = {}
        for row in shape["rows"]:
            widths[int(row["m"])] = row
        out[shape["name"]] = {
            "calls_per_verify": shape["calls_per_verify"],
            "k": shape["k"],
            "n": shape["n"],
            "rows": widths,
        }
    return out


def verify_cost_per_round(
    curve: dict, m: int, exclude: frozenset[str] = frozenset()
) -> float:
    """Predicted QMV seconds for one target verify forward at width m.

    Isolated single-op dispatch cost from a --shapes-only curve, which is why
    the absolute figure feeds only alpha and never psi_eff.
    """
    total = 0.0
    for name, shape in curve.items():
        if name == DRAFT_SIDE_SHAPE or not shape["calls_per_verify"]:
            continue
        if name in exclude:
            continue
        row = shape["rows"].get(m)
        if row is None:
            raise SystemExit(f"e42_analyze: curve has no width {m} for {name}")
        total += shape["calls_per_verify"] * row["seconds_per_call"]
    return total


def widths_of(mtp: dict) -> list[int]:
    """Dispatched verify width per round: 1 primary + the round's drafts."""
    return [d + 1 for d in mtp["effective_draft_lengths"]]


def histogram(widths: list[int]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for m in widths:
        hist[m] = hist.get(m, 0) + 1
    return hist


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def leg_seconds(legs: list[dict], key: str) -> list[float]:
    return [leg[key]["decode_seconds"] for leg in legs]


def summarise_arm(arm: str, legs: list[dict], curve: dict | None) -> dict:
    mtp_secs = leg_seconds(legs, "mtp")
    ser_secs = leg_seconds(legs, "serial")
    widths = widths_of(legs[0]["mtp"])
    rec = {
        "arm": arm,
        "family": arm_family(arm),
        "level": arm_level(arm),
        "legs": len(legs),
        "mtp_decode_seconds": mean(mtp_secs),
        "mtp_decode_seconds_all": mtp_secs,
        "serial_decode_seconds": mean(ser_secs),
        "serial_decode_seconds_all": ser_secs,
        "mtp_seconds_per_token": mean(
            leg["score"]["metrics"]["mtp_seconds_per_token"] for leg in legs
        ),
        "serial_seconds_per_token": mean(
            leg["score"]["metrics"]["serial_seconds_per_token"] for leg in legs
        ),
        "raw_p": mean(leg["score"]["metrics"]["mtp_decode_speedup"] for leg in legs),
        "accepted_draft_rate": mean(
            leg["score"]["metrics"]["accepted_draft_rate"] for leg in legs
        ),
        "effective_mean_draft_len": legs[0]["mtp"]["effective_mean_draft_len"],
        "all_tokens_matched": all(
            leg["score"]["metrics"]["all_tokens_matched"] for leg in legs
        ),
        "round_count": legs[0]["mtp"]["round_count"],
        "declared_rows_total": legs[0]["mtp"]["declared_rows_total"],
        "width_histogram": histogram(widths),
        "widths": widths,
        "seed_prefill_seconds": legs[0]["mtp"]["seed_prefill_seconds"],
        "p50_block_request_seconds": mean(
            leg["mtp"]["p50_block_request_seconds_after_first"] for leg in legs
        ),
        "has_curve": curve is not None,
    }
    if len(mtp_secs) > 1:
        rec["mtp_decode_seconds_sd_pct"] = (
            100.0 * statistics.stdev(mtp_secs) / rec["mtp_decode_seconds"]
        )
    # M = d+1 must close against the parent's own row ledger.
    rec["row_ledger_closes"] = sum(widths) == legs[0]["mtp"]["declared_rows_total"]
    return rec


def per_width_slowdown(
    base_curve: dict, arm_curve: dict, exclude: frozenset[str] = frozenset()
) -> dict[int, dict]:
    """x(M) per dispatched width, aggregated over the verify shape mix."""
    out = {}
    ms = sorted(
        set(base_curve["mlp.down"]["rows"]) & set(arm_curve["mlp.down"]["rows"])
    )
    for m in ms:
        b = verify_cost_per_round(base_curve, m, exclude)
        a = verify_cost_per_round(arm_curve, m, exclude)
        per_shape = {}
        for name in base_curve:
            if name == DRAFT_SIDE_SHAPE or not base_curve[name]["calls_per_verify"]:
                continue
            bs = base_curve[name]["rows"][m]["seconds_per_call"]
            as_ = arm_curve[name]["rows"][m]["seconds_per_call"]
            per_shape[name] = as_ / bs - 1.0
        out[m] = {
            "base_verify_seconds": b,
            "arm_verify_seconds": a,
            "x": a / b - 1.0,
            "x_by_shape": per_shape,
            "in_kernel_path_base": base_curve["mlp.down"]["rows"][m]["in_kernel_path"],
            "in_kernel_path_arm": arm_curve["mlp.down"]["rows"][m]["in_kernel_path"],
        }
    return out


def unstable_shapes(x_by_m: dict[int, dict], ref_width: int) -> list[str]:
    """Shapes whose untreated calibration cell moved more than the tolerance.

    An untreated width must reproduce across builds. A shape that fails there
    is measuring session noise, not the injected regression, so it is dropped
    from the robust denominator rather than silently averaged in.
    """
    cell = x_by_m.get(ref_width)
    if cell is None:
        return []
    return sorted(
        n for n, x in cell["x_by_shape"].items() if abs(x) > UNTREATED_DRIFT_TOL
    )


def analyse(base: dict, arm: dict, base_curve: dict, arm_curve: dict) -> dict:
    """psi_eff, alpha and the occupancy share for one perturbation arm."""
    fam = arm["family"]
    treated = TREATED[fam]
    hist = arm["width_histogram"]
    x_by_m = per_width_slowdown(base_curve, arm_curve)

    def weighted(cells: dict[int, dict]) -> tuple[float, list[float], dict]:
        # Weight x(M) by each width's share of BASE treated QMV cost, so the
        # denominator is not itself inflated by the perturbation.
        weights, xs = [], []
        acc = {"q_treated": 0.0, "q_total": 0.0, "dq": 0.0}
        for m, count in sorted(hist.items()):
            cell = cells[m]
            acc["q_total"] += count * cell["base_verify_seconds"]
            if m in treated:
                w = count * cell["base_verify_seconds"]
                weights.append(w)
                xs.append(cell["x"])
                acc["q_treated"] += w
                acc["dq"] += count * (
                    cell["arm_verify_seconds"] - cell["base_verify_seconds"]
                )
        if not weights:
            raise SystemExit(f"e42_analyze: {arm['arm']} treated no dispatched width")
        return sum(w * x for w, x in zip(weights, xs)) / sum(weights), xs, acc

    x_bar, xs, acc = weighted(x_by_m)
    q_treated, q_total, dq = acc["q_treated"], acc["q_total"], acc["dq"]

    t_base = base["mtp_decode_seconds"]
    t_arm = arm["mtp_decode_seconds"]
    dt_frac = t_arm / t_base - 1.0
    psi_eff = dt_frac / x_bar

    # An untreated dispatched width is the cross-build calibration cell: it must
    # read x ~ 0. Whatever it does read is the honest error bar on x_bar, so psi
    # is reported as an interval over three denominators, not one point.
    ref_width = min(set(x_by_m) - treated, default=None)
    x_ref = x_by_m[ref_width]["x"] if ref_width is not None else 0.0
    x_bar_dc = (1.0 + x_bar) / (1.0 + x_ref) - 1.0
    dropped = unstable_shapes(x_by_m, ref_width) if ref_width is not None else []
    if dropped:
        x_by_m_stable = per_width_slowdown(base_curve, arm_curve, frozenset(dropped))
        x_bar_stable, _, _ = weighted(x_by_m_stable)
        x_ref_stable = x_by_m_stable[ref_width]["x"]
    else:
        x_bar_stable, x_ref_stable = x_bar, x_ref
    psi_variants = {
        "as_measured": psi_eff,
        "drift_corrected": dt_frac / x_bar_dc,
        "stable_shapes_only": dt_frac / x_bar_stable,
    }

    # The serial leg is depth 0: pure M=1, so p2/p6 must leave it alone and m1
    # must slow it down. Both directions are checks, not free parameters.
    serial_frac = arm["serial_decode_seconds"] / base["serial_decode_seconds"] - 1.0

    return {
        "arm": arm["arm"],
        "family": fam,
        "level": arm["level"],
        "treated_widths_dispatched": sorted(set(hist) & treated),
        "untreated_widths_dispatched": sorted(set(hist) - treated),
        "x_by_width": {m: x_by_m[m]["x"] for m in sorted(hist)},
        "x_by_width_all": {m: x_by_m[m]["x"] for m in sorted(x_by_m)},
        "x_bar_treated": x_bar,
        "x_spread_treated": (max(xs) - min(xs)) if len(xs) > 1 else 0.0,
        "mtp_delta_frac": dt_frac,
        "psi_eff": psi_eff,
        # Drift accounting on the untreated calibration cell.
        "calibration_width_untreated": ref_width,
        "x_untreated_calibration": x_ref,
        "x_untreated_calibration_by_shape": (
            x_by_m[ref_width]["x_by_shape"] if ref_width is not None else {}
        ),
        "unstable_shapes_dropped": dropped,
        "x_bar_treated_drift_corrected": x_bar_dc,
        "x_bar_treated_stable_shapes": x_bar_stable,
        "x_untreated_calibration_stable_shapes": x_ref_stable,
        "psi_eff_variants": psi_variants,
        "psi_eff_low": min(psi_variants.values()),
        "psi_eff_high": max(psi_variants.values()),
        "serial_delta_frac": serial_frac,
        "raw_p_base": base["raw_p"],
        "raw_p_arm": arm["raw_p"],
        "raw_p_delta": arm["raw_p"] - base["raw_p"],
        # Occupancy vs marginal: predicted absolute QMV time from the isolated
        # curve, against what the leg actually paid.
        "q_treated_predicted_seconds": q_treated,
        "q_total_predicted_seconds": q_total,
        "occupancy_share_treated": q_treated / t_base,
        "occupancy_share_total_qmv": q_total / t_base,
        "dq_predicted_seconds": dq,
        "dt_measured_seconds": t_arm - t_base,
        "mtp_decode_seconds_base": t_base,
        "mtp_decode_seconds_arm": t_arm,
        "alpha_absorption": (t_arm - t_base) / dq if dq else float("nan"),
        "trajectory_identical_to_base": arm["widths"] == base["widths"],
        "all_tokens_matched": arm["all_tokens_matched"],
    }


def slope_psi(base: dict, results: list[dict]) -> dict | None:
    """psi from the slope of leg time against measured kernel slowdown.

    Points are (x_bar, T) plus the untreated origin (0, T_base). The slope is
    the treated QMV seconds the leg actually pays in situ, so psi = slope/T_base
    needs no absolute cost prediction at all, and the spread of the pairwise
    slopes is an empirical error bar instead of an assumption. It is also the
    strongest form of the linearity test: a single psi exists iff the slopes
    agree.
    """
    if len(results) < 2:
        return None
    t0 = base["mtp_decode_seconds"]
    pts = [(0.0, t0)] + sorted(
        (r["x_bar_treated"], r["mtp_decode_seconds_arm"]) for r in results
    )
    slopes = []
    for (x1, t1), (x2, t2) in itertools.combinations(pts, 2):
        slopes.append(
            {"x_from": x1, "x_to": x2, "q_treated_seconds": (t2 - t1) / (x2 - x1)}
        )
    qs = [s["q_treated_seconds"] for s in slopes]
    q_mean = sum(qs) / len(qs)
    rounds = base["round_count"]
    return {
        "points_x_bar_and_leg_seconds": pts,
        "pairwise_slopes": slopes,
        "q_treated_seconds_from_slope": q_mean,
        "slope_spread_pct": 100.0 * (max(qs) - min(qs)) / q_mean,
        "psi_from_slope": q_mean / t0,
        "non_qmv_seconds": t0 - q_mean,
        "non_qmv_ms_per_round": 1000.0 * (t0 - q_mean) / rounds,
        "q_treated_seconds_from_curve": results[0]["q_treated_predicted_seconds"],
        "curve_vs_slope_pct": 100.0
        * (q_mean / results[0]["q_treated_predicted_seconds"] - 1.0),
    }


def mde(sd_pct: float, n: int, design: str = "two_sample") -> dict:
    """Delegate to research/e39_mde.py, unmodified, so nulls stay auditable."""
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/e39_mde.py"),
            "--mde",
            "--sd",
            f"{sd_pct}",
            "--n",
            str(n),
            "--design",
            design,
        ],
        capture_output=True,
        text=True,
    )
    return {"stdout": proc.stdout.strip(), "returncode": proc.returncode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="research/e42-artifacts/analysis.json")
    args = ap.parse_args()

    arms, curves = {}, {}
    for arm in args.arms:
        curves[arm] = load_curve(arm)
        arms[arm] = summarise_arm(arm, load_legs(arm), curves[arm])

    base = arms[args.base]
    base_curve = curves[args.base]
    if base_curve is None:
        raise SystemExit("e42_analyze: the base arm needs a cost curve")

    print("=== arms (MLXFAST_LOCAL_COOL_GATE=0; not gate-qualified) ===")
    hdr = (
        f"{'arm':>7} {'legs':>4} {'mtp_s':>8} {'ser_s':>8} {'raw_p':>7} "
        f"{'acc':>6} {'mean_d':>7} {'rounds':>6} {'ledger':>6} {'exact':>5}"
    )
    print(hdr)
    for arm, rec in arms.items():
        print(
            f"{arm:>7} {rec['legs']:>4} {rec['mtp_decode_seconds']:>8.3f} "
            f"{rec['serial_decode_seconds']:>8.3f} {rec['raw_p']:>7.4f} "
            f"{rec['accepted_draft_rate']:>6.4f} {rec['effective_mean_draft_len']:>7.4f} "
            f"{rec['round_count']:>6} {str(rec['row_ledger_closes']):>6} "
            f"{str(rec['all_tokens_matched']):>5}"
        )

    print("\n=== round width histogram (M = drafts + 1) ===")
    for arm, rec in arms.items():
        h = " ".join(f"M{m}:{c}" for m, c in sorted(rec["width_histogram"].items()))
        same = "same-as-base" if rec["widths"] == base["widths"] else "*** MOVED ***"
        print(f"{arm:>7} {h}   [{same}]")

    results = []
    print("\n=== psi / phi ===")
    for arm, rec in arms.items():
        if rec["family"] == "base" or curves[arm] is None:
            continue
        results.append(analyse(base, rec, base_curve, curves[arm]))

    # An arm without its own curve inherits x(M) from the same-level p2 curve:
    # a treated width's kernel is the same template instantiation in both.
    for arm, rec in arms.items():
        if rec["family"] == "base" or curves[arm] is not None:
            continue
        donor = f"p2L{rec['level']}"
        if curves.get(donor) is None:
            print(f"{arm:>7} no curve and no level-matched donor ({donor}); skipped")
            continue
        res = analyse(base, rec, base_curve, curves[donor])
        res["x_donor_curve"] = donor
        results.append(res)

    for res in results:
        print(
            f"{res['arm']:>7} L{res['level']} treated={res['treated_widths_dispatched']} "
            f"x_bar={res['x_bar_treated']:+.4f} (spread {res['x_spread_treated']:.4f})"
            + (f" [x from {res['x_donor_curve']}]" if "x_donor_curve" in res else "")
        )
        print(
            f"        MTP leg {res['mtp_delta_frac']*100:+.3f} %   "
            f"serial leg {res['serial_delta_frac']*100:+.3f} %   "
            f"raw_p {res['raw_p_delta']:+.4f}"
        )
        print(
            f"        psi_eff = {res['psi_eff']:.4f}    "
            f"occupancy(treated) = {res['occupancy_share_treated']:.4f}    "
            f"alpha = {res['alpha_absorption']:.4f}"
        )
        print(
            f"        calibration width M={res['calibration_width_untreated']} "
            f"reads x={res['x_untreated_calibration']:+.4f} "
            f"(stable-shape subset {res['x_untreated_calibration_stable_shapes']:+.4f}, "
            f"dropped {res['unstable_shapes_dropped'] or 'none'})"
        )
        print(
            "        psi_eff interval "
            f"[{res['psi_eff_low']:.4f}, {res['psi_eff_high']:.4f}]  "
            + "  ".join(f"{k}={v:.4f}" for k, v in res["psi_eff_variants"].items())
        )

    by_fam: dict[str, list[dict]] = {}
    for res in results:
        by_fam.setdefault(res["family"], []).append(res)

    print("\n=== linearity (a single psi requires psi_eff(L1) == psi_eff(L2)) ===")
    linearity = {}
    for fam, group in sorted(by_fam.items()):
        group.sort(key=lambda r: r["level"])
        vals = {r["level"]: r["psi_eff"] for r in group}
        line = "  ".join(f"L{lv}: psi_eff={v:.4f}" for lv, v in sorted(vals.items()))
        ratio = None
        if 1 in vals and 2 in vals and vals[1]:
            ratio = vals[2] / vals[1]
            line += f"   ratio L2/L1 = {ratio:.4f}"
        linearity[fam] = {"psi_eff_by_level": vals, "ratio_l2_over_l1": ratio}
        print(f"{fam:>7} {line}")

    print("\n=== psi from the ladder slope (no absolute cost prediction) ===")
    slopes = {}
    for fam, group in sorted(by_fam.items()):
        sl = slope_psi(base, group)
        if sl is None:
            print(f"{fam:>7} needs >=2 magnitudes for a slope")
            continue
        slopes[fam] = sl
        print(
            f"{fam:>7} Q_treated={sl['q_treated_seconds_from_slope']:.4f} s "
            f"(pairwise spread {sl['slope_spread_pct']:.3f} %)  "
            f"psi_from_slope={sl['psi_from_slope']:.4f}"
        )
        print(
            f"        non-QMV intercept {sl['non_qmv_seconds']:.4f} s = "
            f"{sl['non_qmv_ms_per_round']:.2f} ms/round;  "
            f"isolated curve predicted {sl['q_treated_seconds_from_curve']:.4f} s "
            f"({sl['curve_vs_slope_pct']:+.3f} % vs slope)"
        )

    phi_local = None
    p2 = {r["level"]: r for r in by_fam.get("p2", [])}
    p6 = {r["level"]: r for r in by_fam.get("p6", [])}
    if p2 and p6:
        print("\n=== phi_local = psi_eff(p6) / psi_eff(p2), per level ===")
        phi_local = {}
        for lv in sorted(set(p2) & set(p6)):
            phi_local[lv] = p6[lv]["psi_eff"] / p2[lv]["psi_eff"]
            print(f"     L{lv} phi_local = {phi_local[lv]:.4f}")

    # Session drift envelope from the base bracket. Every psi_eff is a ratio to
    # `base`, so a bracket wider than the smallest effect would void the session.
    drift = {}
    bracket = [a for a in arms if arm_family(a) == "base"]
    if len(bracket) > 1:
        first, last = arms[bracket[0]], arms[bracket[-1]]
        drift = {
            "arms": bracket,
            "mtp_drift_pct": 100.0
            * (last["mtp_decode_seconds"] / first["mtp_decode_seconds"] - 1.0),
            "serial_drift_pct": 100.0
            * (last["serial_decode_seconds"] / first["serial_decode_seconds"] - 1.0),
            "raw_p_drift": last["raw_p"] - first["raw_p"],
            "trajectory_identical": first["widths"] == last["widths"],
        }
        print("\n=== base bracket drift envelope ===")
        print(
            f"{bracket[0]} -> {bracket[-1]}   MTP {drift['mtp_drift_pct']:+.4f} %   "
            f"serial {drift['serial_drift_pct']:+.4f} %   "
            f"raw_p {drift['raw_p_drift']:+.4f}   "
            f"trajectory identical={drift['trajectory_identical']}"
        )

    sds = [
        r["mtp_decode_seconds_sd_pct"]
        for r in arms.values()
        if "mtp_decode_seconds_sd_pct" in r
    ]
    power = {}
    if sds:
        sd = max(sds)
        power = {"worst_within_arm_sd_pct": sd, "mde": mde(sd, 2)}
        print(f"\n=== power (research/e39_mde.py, unmodified) ===")
        print(f"worst within-arm sd = {sd:.4f} %")
        print(power["mde"]["stdout"])

    payload = {
        "arms": arms,
        "results": results,
        "linearity": linearity,
        "slope_psi": slopes,
        "phi_local": phi_local,
        "power": power,
        "drift": drift,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")

    if args.wandb:
        log_wandb(payload)
    return 0


def log_wandb(payload: dict) -> None:
    import wandb

    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="qwen38-mlx-challenge-senpai",
        name="e42-psi-phi-injected-regression",
        job_type="analysis",
        config={
            "experiment": "e42-psi-phi-by-injected-regression",
            "base_sha": "04ad6bf11437c269df85a47e91faa769c74fe6da",
            "host": "local-m4-pro",
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "fixture": "correctness_prompts/public_longcopy_gate_english_512_256.json",
            "decode_tokens": 512,
            "offered_depth": 8,
        },
    )
    arm_cols = [
        "arm",
        "family",
        "level",
        "legs",
        "mtp_decode_seconds",
        "serial_decode_seconds",
        "mtp_seconds_per_token",
        "serial_seconds_per_token",
        "raw_p",
        "accepted_draft_rate",
        "effective_mean_draft_len",
        "round_count",
        "declared_rows_total",
        "row_ledger_closes",
        "all_tokens_matched",
        "seed_prefill_seconds",
        "p50_block_request_seconds",
    ]
    arms_table = wandb.Table(columns=arm_cols)
    for rec in payload["arms"].values():
        arms_table.add_data(*[rec.get(c) for c in arm_cols])

    res_cols = [
        "arm",
        "family",
        "level",
        "x_bar_treated",
        "x_spread_treated",
        "mtp_delta_frac",
        "serial_delta_frac",
        "psi_eff",
        "psi_eff_low",
        "psi_eff_high",
        "calibration_width_untreated",
        "x_untreated_calibration",
        "x_bar_treated_drift_corrected",
        "x_bar_treated_stable_shapes",
        "occupancy_share_treated",
        "occupancy_share_total_qmv",
        "alpha_absorption",
        "dq_predicted_seconds",
        "dt_measured_seconds",
        "raw_p_delta",
        "trajectory_identical_to_base",
        "all_tokens_matched",
    ]
    res_table = wandb.Table(columns=res_cols)
    for res in payload["results"]:
        res_table.add_data(*[res.get(c) for c in res_cols])

    width_table = wandb.Table(columns=["arm", "m", "x", "count"])
    for res in payload["results"]:
        hist = payload["arms"][res["arm"]]["width_histogram"]
        for m, x in res["x_by_width"].items():
            width_table.add_data(res["arm"], int(m), x, hist.get(str(m), hist.get(m, 0)))

    run.log({"arms": arms_table, "psi_phi": res_table, "x_by_width": width_table})
    summary = {
        "linearity": payload["linearity"],
        "phi_local": payload["phi_local"],
        "slope_psi": payload["slope_psi"],
    }
    for fam, sl in payload["slope_psi"].items():
        summary[f"psi_from_slope/{fam}"] = sl["psi_from_slope"]
        summary[f"slope_spread_pct/{fam}"] = sl["slope_spread_pct"]
        summary[f"non_qmv_ms_per_round/{fam}"] = sl["non_qmv_ms_per_round"]
    for res in payload["results"]:
        summary[f"psi_eff/{res['arm']}"] = res["psi_eff"]
        summary[f"psi_eff_low/{res['arm']}"] = res["psi_eff_low"]
        summary[f"psi_eff_high/{res['arm']}"] = res["psi_eff_high"]
        summary[f"alpha/{res['arm']}"] = res["alpha_absorption"]
        summary[f"x_bar/{res['arm']}"] = res["x_bar_treated"]
        summary[f"x_untreated_calibration/{res['arm']}"] = res[
            "x_untreated_calibration"
        ]
    summary["power"] = payload["power"]
    run.summary.update(summary)
    print(f"wandb_run_url={run.url}")
    print(f"wandb_run_id={run.id}")
    run.finish()


if __name__ == "__main__":
    sys.exit(main())
