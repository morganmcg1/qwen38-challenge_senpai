#!/usr/bin/env python3
"""E48: does a UNIFORM QMV cost change move the score up or down?

    research/e48_analyze.py --arms base ulo uhi base2 [--wandb]

The score identity is `raw_p = serial / mtp`, so a change that scales QMV cost
by (1+x) everywhere scales raw_p by (1 + psi_serial*x)/(1 + psi_mtp_TOTAL*x).
Ledger 173(A) claims psi_mtp_TOTAL < psi_serial, i.e. a uniform QMV *speedup*
LOWERS the score. E48 injects the cost instead of modelling it.

Two things make this harder than E42:

1. The pass-loop injection is NOT uniform in x. It covers ~90 % of a crossrow
   kernel and ~43 % of `qmv_fast_impl`, so one level everywhere over-doses the
   candidate leg 2.1x. The arms therefore carry independent levels per family
   and the realised dose ratio x1/xX is measured, not assumed.
2. The candidate leg is NOT free of width-1 QMV. `psi_mtp_w1` is small but
   non-zero, so `psi_mtp_TOTAL = psi_mtp_w1 + psi_mtp_X` and the two dosed arms
   are used to identify both terms rather than inheriting either.

Every leg here ran with MLXFAST_LOCAL_COOL_GATE=0. Nothing printed is
gate-qualified, ranked-equivalent, or an official score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e42_analyze import (  # noqa: E402
    DRAFT_SIDE_SHAPE,
    histogram,
    load_legs,
    mean,
    summarise_arm,
    verify_cost_per_round,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / ".mlxfast-private/e48/runs"
CURVES = ROOT / ".mlxfast-private/qmv-curve"

# (crossrow_level, m1_level) as spliced by research/e48_perturb.py. The levels
# are deliberately non-proportional between ulo and uhi: proportional doses make
# the two-unknown MTP-leg system singular.
DOSES = {"base": (0, 0), "base2": (0, 0), "ulo": (1, 2), "uhi": (2, 3)}
CROSSROW_WIDTHS = frozenset(range(2, 10))
# Cross-build reproduction tolerance for a shape's per-call cost.
REPRO_TOL = 0.02
nan = float("nan")


def load_meta(arm: str) -> dict:
    """Provenance and thermal record written by research/e42-run.sh."""
    meta: dict[str, str] = {}
    path = RUNS / arm / "meta.txt"
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        meta[key] = value

    def gpu_c(raw: str | None) -> float:
        if not raw:
            return nan
        for field in raw.split():
            if field.startswith("gpu_temp="):
                return float(field[len("gpu_temp=") : -1])
        return nan

    legs = sorted(k for k in meta if k.endswith("_thermal_after") and k.startswith("leg"))
    return {
        "head_sha": meta["head_sha"],
        "twin_digests": meta["twin_digests"],
        "metallib_fingerprint": meta.get("metallib_fingerprint"),
        "started": meta.get("started"),
        "entry_gpu_temp_c": gpu_c(meta.get("thermal_before")),
        "exit_gpu_temp_c": gpu_c(meta.get("thermal_after")),
        "per_leg_exit_gpu_temp_c": [gpu_c(meta[k]) for k in legs],
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
    }


def load_curve(arm: str) -> dict | None:
    path = CURVES / f"e48-{arm}" / "vendored.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    out = {}
    for shape in payload["shapes"]:
        out[shape["name"]] = {
            "calls_per_verify": shape["calls_per_verify"],
            "k": shape["k"],
            "n": shape["n"],
            "rows": {int(row["m"]): row for row in shape["rows"]},
        }
    return out


def per_width_slowdown(base_curve, arm_curve, exclude=frozenset()) -> dict[int, dict]:
    ms = sorted(set(base_curve["mlp.down"]["rows"]) & set(arm_curve["mlp.down"]["rows"]))
    out = {}
    for m in ms:
        b = verify_cost_per_round(base_curve, m, exclude)
        a = verify_cost_per_round(arm_curve, m, exclude)
        per_shape = {}
        for name, shape in base_curve.items():
            if name == DRAFT_SIDE_SHAPE or not shape["calls_per_verify"] or name in exclude:
                continue
            per_shape[name] = (
                arm_curve[name]["rows"][m]["seconds_per_call"]
                / shape["rows"][m]["seconds_per_call"]
                - 1.0
            )
        out[m] = {
            "base_verify_seconds": b,
            "arm_verify_seconds": a,
            "x": a / b - 1.0,
            "x_by_shape": per_shape,
        }
    return out


def costed_shapes(curve: dict, exclude: frozenset[str] = frozenset()) -> list[str]:
    return [
        n
        for n, s in curve.items()
        if n != DRAFT_SIDE_SHAPE and s["calls_per_verify"] and n not in exclude
    ]


def irreproducible_shapes(base_curve: dict, repeat_curve: dict) -> list[str]:
    """Shapes that fail to reproduce between two builds of the SAME source.

    E42 had to infer this from an untreated width inside a dosed arm. E48 treats
    every width, so the base bracket is the only clean cross-build control -- and
    it is the stronger one, because it exercises every width and every shape.
    """
    x_by_m = per_width_slowdown(base_curve, repeat_curve)
    bad = set()
    for cell in x_by_m.values():
        bad |= {n for n, x in cell["x_by_shape"].items() if abs(x) > REPRO_TOL}
    return sorted(bad)


def crossrow_xbar(x_by_m: dict[int, dict], hist: dict[int, int]) -> float:
    """x over widths 2..9, weighted by each width's share of BASE verify QMV cost."""
    weights, xs = [], []
    for m, count in sorted(hist.items()):
        if m not in CROSSROW_WIDTHS or m not in x_by_m:
            continue
        weights.append(count * x_by_m[m]["base_verify_seconds"])
        xs.append(x_by_m[m]["x"])
    if not weights:
        return nan
    return sum(w * x for w, x in zip(weights, xs)) / sum(weights)


def coverage_gap(base_curve: dict, hist: dict[int, int], mtp_seconds: float) -> dict:
    """Candidate-leg QMV that NO arm treats, so psi_mtp comes out a lower bound.

    Two families escape both injections:

    * the 2-bit compact draft readout. `quantized.h:1908` dispatches
      `qmv_fast_singlerow_affine2_g64` at `ntg.x == 1 && out_vec_size == 98336`
      and RETURNS, upstream of both crossrow tiers and of the `qmv_fast_impl`
      fall-through the m1 dose patches. It is candidate-leg-only by construction.
    * GDN `in_proj_b` / `in_proj_a`, n=48 (`Qwen35GatedDelta.swift:254-255`).
      `out_vec_size < 1024` sends them to `qmv_fast_impl` at every width, so the
      crossrow dose misses them and the m1 dose only catches width 1. The curve
      models all four GDN input projections as one fused n=16480 shape
      (`QwenQMVCostCurveTests.swift:396`), so their bandwidth is already inside
      the treated total; only ~0.6 % of that shape's width is a/b columns.

    The serial leg has neither: at depth 0 every backbone shape falls through to
    `qmv_fast_impl`, which the m1 dose treats in full (E42 measured its
    alpha at 0.9733). So the asymmetry runs one way -- psi_mtp is understated,
    psi_serial is not -- and the uniform coefficient psi_mtp - psi_serial is a
    LOWER bound. A measured positive coefficient is therefore conclusive; a
    measured negative one has to clear this correction first.
    """
    treated = sum(count * verify_cost_per_round(base_curve, m) for m, count in hist.items())
    draft = base_curve.get(DRAFT_SIDE_SHAPE)
    draft_steps = sum(count * (m - 1) for m, count in hist.items())
    untreated = draft_steps * draft["rows"][1]["seconds_per_call"] if draft else 0.0
    return {
        "treated_verify_qmv_seconds": treated,
        "untreated_draft_readout_seconds": untreated,
        "draft_steps": draft_steps,
        "untreated_share_of_candidate_qmv": untreated / (treated + untreated),
        "psi_mtp_additive_correction": untreated / mtp_seconds,
        "note": "isolated single-op dispatch cost, so the correction is indicative, not exact",
        "direction": "psi_mtp measured is a LOWER bound; uniform coefficient is a LOWER bound",
    }


def solve_two_arms(arms: dict[str, dict]) -> dict:
    """Identify psi_mtp_w1 and psi_mtp_X from two non-proportional dosed arms."""
    lo, hi = arms["ulo"], arms["uhi"]
    a11, a12, b1 = lo["x1"], lo["xbar_X"], lo["mtp_frac"]
    a21, a22, b2 = hi["x1"], hi["xbar_X"], hi["mtp_frac"]
    det = a11 * a22 - a12 * a21
    out = {"determinant": det, "dose_proportionality": (a11 / a21) / (a12 / a22)}
    if abs(det) < 1e-9:
        out["identified"] = False
        return out
    out["identified"] = True
    out["psi_mtp_w1"] = (b1 * a22 - b2 * a12) / det
    out["psi_mtp_X"] = (a11 * b2 - a21 * b1) / det
    out["psi_mtp_total"] = out["psi_mtp_w1"] + out["psi_mtp_X"]
    # The serial leg is depth 0, so it is pure width-1 QMV: two arms, one unknown.
    out["psi_serial_per_arm"] = {k: arms[k]["serial_frac"] / arms[k]["x1"] for k in ("ulo", "uhi")}
    out["psi_serial"] = mean(out["psi_serial_per_arm"].values())
    out["psi_serial_form_residual_pct"] = 100.0 * (
        out["psi_serial_per_arm"]["uhi"] / out["psi_serial_per_arm"]["ulo"] - 1.0
    )
    out["uniform_coefficient"] = out["psi_mtp_total"] - out["psi_serial"]
    out["uniform_sign"] = "negative" if out["uniform_coefficient"] < 0 else "positive"
    out["uniform_raw_p_ratio"] = {
        f"x={x}": (1 + out["psi_serial"] * x) / (1 + out["psi_mtp_total"] * x)
        for x in (0.5, 0.9, 1.0, 1.8)
    }
    return out


def one_sided_bound(arm: dict) -> dict:
    """Sign conclusion that survives without knowing the dose ratio exactly.

    raw_p rises iff serial_frac > mtp_frac. Under-dosing width 1 (x1 <= xX) can
    only make raw_p look WORSE than a truly uniform change would, so raw_p up at
    x1 <= xX proves the uniform sign is negative. Over-dosing gives the mirror
    proof for a positive sign.
    """
    ratio = arm["x1"] / arm["xbar_X"] if arm["xbar_X"] else nan
    up = arm["serial_frac"] > arm["mtp_frac"]
    verdict = "inconclusive"
    if ratio <= 1.0 and up:
        verdict = "uniform sign NEGATIVE (proved: under-dosed width 1 and raw_p still rose)"
    elif ratio >= 1.0 and not up:
        verdict = "uniform sign POSITIVE (proved: over-dosed width 1 and raw_p still fell)"
    return {
        "realised_dose_ratio_x1_over_xX": ratio,
        "raw_p_direction": "up" if up else "down",
        "one_sided_verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "ulo", "uhi", "base2"])
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    arms, curves, metas = {}, {}, {}
    for arm in args.arms:
        arms[arm] = summarise_arm(arm, load_legs_for(arm), None)
        curves[arm] = load_curve(arm)
        metas[arm] = load_meta(arm)

    base, base_curve = arms["base"], curves["base"]
    exclude = frozenset()
    entry_temps = [m["entry_gpu_temp_c"] for m in metas.values()]
    payload: dict = {
        "base_sha": "fb0a09d3912477d94ed631bdb90fd04172d7b4cf",
        "head_sha": git_head(),
        "doses": {k: DOSES.get(k) for k in args.arms},
        "arm_order": args.arms,
        "counterbalancing": (
            "bracketed A B B' A: the two zero-dose arms bookend the dosed arms in one "
            "session, so monotone drift shows up as the base/base2 spread reported below"
        ),
        "provenance": metas,
        "entry_gpu_temp_spread_c": max(entry_temps) - min(entry_temps),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }

    if "base2" in arms:
        b2 = arms["base2"]
        payload["null_arm"] = {
            "what": "byte-identical zero-dose repeat of base, separate build and session slot",
            "serial_frac": b2["serial_decode_seconds"] / base["serial_decode_seconds"] - 1.0,
            "mtp_frac": b2["mtp_decode_seconds"] / base["mtp_decode_seconds"] - 1.0,
            "raw_p_base": base["raw_p"],
            "raw_p_repeat": b2["raw_p"],
            "raw_p_ratio": b2["raw_p"] / base["raw_p"],
        }
        if curves.get("base2"):
            exclude = frozenset(irreproducible_shapes(base_curve, curves["base2"]))
            payload["null_arm"]["irreproducible_shapes"] = sorted(exclude)
            payload["null_arm"]["curve_repro_tolerance"] = REPRO_TOL
            if not costed_shapes(base_curve, exclude):
                payload["null_arm"]["stable_shapes_variant"] = (
                    "unavailable: every costed shape failed cross-build reproduction, "
                    "so there is no stable subset to reprice against"
                )
                exclude = frozenset()

    dosed = {}
    for arm in ("ulo", "uhi"):
        if arm not in arms or not curves.get(arm):
            continue
        rec = arms[arm]
        x_all = per_width_slowdown(base_curve, curves[arm])
        x_stable = per_width_slowdown(base_curve, curves[arm], exclude) if exclude else x_all
        hist = rec["width_histogram"]
        entry = {
            "crossrow_level": DOSES[arm][0],
            "m1_level": DOSES[arm][1],
            "serial_frac": rec["serial_decode_seconds"] / base["serial_decode_seconds"] - 1.0,
            "mtp_frac": rec["mtp_decode_seconds"] / base["mtp_decode_seconds"] - 1.0,
            "raw_p": rec["raw_p"],
            "raw_p_ratio": rec["raw_p"] / base["raw_p"],
            "x1": x_all[1]["x"],
            "xbar_X": crossrow_xbar(x_all, hist),
            "x1_stable": x_stable[1]["x"],
            "xbar_X_stable": crossrow_xbar(x_stable, hist),
            "x_by_width": {m: c["x"] for m, c in sorted(x_all.items())},
            "all_tokens_matched": rec["all_tokens_matched"],
            "row_ledger_closes": rec["row_ledger_closes"],
        }
        entry.update(one_sided_bound(entry))
        dosed[arm] = entry
    payload["dosed_arms"] = dosed

    gap = coverage_gap(base_curve, base["width_histogram"], base["mtp_decode_seconds"])
    payload["coverage_gap"] = gap

    if len(dosed) == 2:
        payload["identified"] = solve_two_arms(dosed)
        stable = {k: dict(v, x1=v["x1_stable"], xbar_X=v["xbar_X_stable"]) for k, v in dosed.items()}
        payload["identified_stable_shapes"] = solve_two_arms(stable)
        for key in ("identified", "identified_stable_shapes"):
            ident = payload[key]
            if not ident.get("identified"):
                continue
            corrected = ident["psi_mtp_total"] + gap["psi_mtp_additive_correction"]
            coeff = corrected - ident["psi_serial"]
            ident["psi_mtp_total_gap_corrected"] = corrected
            ident["uniform_coefficient_gap_corrected"] = coeff
            ident["uniform_sign_gap_corrected"] = "negative" if coeff < 0 else "positive"
            ident["sign_robust_to_coverage_gap"] = (
                ident["uniform_coefficient"] > 0 or coeff < 0
            )

    payload["arms"] = {k: strip(v) for k, v in arms.items()}
    payload["width_histogram_spread"] = width_spread(args.arms)
    print(json.dumps(payload, indent=2, default=str))
    if args.wandb:
        log_wandb(payload)
    return 0


def load_legs_for(arm: str) -> list[dict]:
    import e42_analyze

    e42_analyze.RUNS = RUNS
    return load_legs(arm)


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not isinstance(v, (list, dict)) or k.endswith("histogram")}


def width_spread(arm_names: list[str]) -> dict:
    """Is the per-round width histogram a random variable across repeats?"""
    draws = []
    for arm in arm_names:
        for leg in load_legs_for(arm):
            widths = [d + 1 for d in leg["mtp"]["effective_draft_lengths"]]
            draws.append(
                {
                    "arm": arm,
                    "leg": leg["index"],
                    "rounds": len(widths),
                    "mean_m": mean(widths),
                    "histogram": histogram(widths),
                }
            )
    means = [d["mean_m"] for d in draws]
    hists = [json.dumps(d["histogram"], sort_keys=True) for d in draws]
    return {
        "draws": draws,
        "n_draws": len(draws),
        "identical_across_all_draws": len(set(hists)) == 1,
        "mean_m_range": [min(means), max(means)] if means else [],
        "mean_m_sd_pct": (100.0 * statistics.stdev(means) / mean(means)) if len(means) > 1 else 0.0,
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def log_wandb(payload: dict) -> None:
    import wandb

    ident = payload.get("identified", {})
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="qwen38-mlx-challenge-senpai",
        name="e48-uniform-qmv-sign",
        job_type="analysis",
        tags=["e48", "qwen3.8-27b-mtp-v1", "injected-regression", "ungated-local"],
        config={
            "base_sha": payload["base_sha"],
            "head_sha": payload["head_sha"],
            "doses": payload["doses"],
            "host": "local-m4-pro",
            "fixture": "correctness_prompts/public_longcopy_gate_english_512_256.json",
            "decode_tokens": 512,
            "offered_depth": 8,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        },
    )
    summary = {
        "psi_serial": ident.get("psi_serial"),
        "psi_mtp_w1": ident.get("psi_mtp_w1"),
        "psi_mtp_X": ident.get("psi_mtp_X"),
        "psi_mtp_total": ident.get("psi_mtp_total"),
        "uniform_coefficient": ident.get("uniform_coefficient"),
        "uniform_sign": ident.get("uniform_sign"),
        "uniform_coefficient_gap_corrected": ident.get("uniform_coefficient_gap_corrected"),
        "uniform_sign_gap_corrected": ident.get("uniform_sign_gap_corrected"),
        "sign_robust_to_coverage_gap": ident.get("sign_robust_to_coverage_gap"),
        "psi_serial_form_residual_pct": ident.get("psi_serial_form_residual_pct"),
        "identified": ident.get("identified"),
        "untreated_share_of_candidate_qmv": payload["coverage_gap"][
            "untreated_share_of_candidate_qmv"
        ],
    }
    for arm, rec in payload.get("dosed_arms", {}).items():
        for key in (
            "serial_frac",
            "mtp_frac",
            "raw_p",
            "raw_p_ratio",
            "x1",
            "xbar_X",
            "realised_dose_ratio_x1_over_xX",
            "all_tokens_matched",
        ):
            summary[f"{arm}/{key}"] = rec.get(key)
        summary[f"{arm}/one_sided_verdict"] = rec.get("one_sided_verdict")
    for key, value in (payload.get("null_arm") or {}).items():
        if not isinstance(value, (list, dict)):
            summary[f"null/{key}"] = value
    summary["width_histogram_identical_across_draws"] = payload["width_histogram_spread"][
        "identical_across_all_draws"
    ]
    summary["width_mean_m_sd_pct"] = payload["width_histogram_spread"]["mean_m_sd_pct"]
    summary["entry_gpu_temp_spread_c"] = payload["entry_gpu_temp_spread_c"]
    for arm, meta in payload["provenance"].items():
        summary[f"{arm}/entry_gpu_temp_c"] = meta["entry_gpu_temp_c"]
        summary[f"{arm}/exit_gpu_temp_c"] = meta["exit_gpu_temp_c"]
        summary[f"{arm}/head_sha"] = meta["head_sha"]
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    arm_table = wandb.Table(
        columns=[
            "arm",
            "crossrow_level",
            "m1_level",
            "serial_frac",
            "mtp_frac",
            "raw_p",
            "raw_p_ratio",
            "x1",
            "xbar_X",
            "dose_ratio",
            "verdict",
        ]
    )
    for arm, rec in payload.get("dosed_arms", {}).items():
        arm_table.add_data(
            arm,
            rec["crossrow_level"],
            rec["m1_level"],
            rec["serial_frac"],
            rec["mtp_frac"],
            rec["raw_p"],
            rec["raw_p_ratio"],
            rec["x1"],
            rec["xbar_X"],
            rec["realised_dose_ratio_x1_over_xX"],
            rec["one_sided_verdict"],
        )
    width_table = wandb.Table(columns=["arm", "m", "x"])
    for arm, rec in payload.get("dosed_arms", {}).items():
        for m, x in rec["x_by_width"].items():
            width_table.add_data(arm, int(m), x)
    hist_table = wandb.Table(columns=["arm", "leg", "rounds", "mean_m", "histogram"])
    for draw in payload["width_histogram_spread"]["draws"]:
        hist_table.add_data(
            draw["arm"], draw["leg"], draw["rounds"], draw["mean_m"], json.dumps(draw["histogram"])
        )
    run.log({"arms": arm_table, "per_width_x": width_table, "width_histograms": hist_table})
    run.finish()
    print(f"wandb_run_url={run.url}")
    print(f"wandb_run_id={run.id}")


if __name__ == "__main__":
    sys.exit(main())
