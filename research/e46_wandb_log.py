#!/usr/bin/env python3
"""Log E46 -- weight streams vs group width at fixed M -- to W&B.

E46 asks what E41's 20.291 ms/stream coefficient is actually a coefficient OF.
E41 fitted it on a table with a single stream boundary at 5->6, where the stream
count and the widest group width move together, so the name was not identified.
E46 separates them twice at fixed M on the shipped NA<=4 table, and re-measures
the width curve on a table whose boundaries sit at 4->5 and 8->9 instead.

The durable record is the pre-registration, the compile-only register census
that gated the GPU time, the dispatch readback of each build, the ABBA curve
with its per-width replication floor, the per-shape deltas, the thermal record,
and the independent prior replication.

  python3 research/e46_wandb_log.py research/e46-artifacts/e46-metrics.json
  python3 research/e46_wandb_log.py research/e46-artifacts/e46-metrics.json \
      --resume <run_id>
"""

from __future__ import annotations

import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e46_prereg as P  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
CENSUS = pathlib.Path("research/e46-reg-census.json")


def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text())
    resume_id = _flag("--resume")
    ident = d["identity"]
    census = json.loads(CENSUS.read_text()) if CENSUS.exists() else None

    def temp(key, field):
        return _f(ident.get(key, {}).get(field))

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=resume_id,
        resume="must" if resume_id else None,
        name="e46-stream-vs-groupwidth-fixed-m",
        job_type="cost-curve",
        tags=["e46", "crossrow-qmv", "weight-streams", "group-width", "cost-curve",
              "prereg", "abba", "e41-followup", "mechanism-identification",
              "replication"],
        config={
            "assignment_id": "qwen38-r1-e46-stream-vs-groupwidth-fixed-m",
            "revision_id": "r1",
            "student": "qwen-thorfinn",
            "pr_number": 51,
            "base_sha": P.BASE_SHA,
            "e41_base_sha": P.E41_BASE_SHA,
            "host": ident.get("base1", {}).get("host"),
            "reps": ident.get("base1", {}).get("reps"),
            "inner": ident.get("base1", {}).get("inner"),
            "widths": d["widths"],
            "run_order": d["order"],
            "build_heads": {k: v.get("head") for k, v in ident.items()},
            "build_dirty": {k: v.get("dirty") for k, v in ident.items()},
            # The design: both builds occupy mean sweep position 2.5, so linear
            # session drift cancels in the arithmetic mean of each pair.
            "counterbalancing": "ABBA within one session (base, arm, arm, base)",
            "sweep_matched_across_arms": True,
            "arm_edit": P.RUN_CONFIG["arm_edit"],
            "controls": P.RUN_CONFIG["controls"],
            "metric": P.RUN_CONFIG["metric"],
            "shipped_ipg_table": P.TABLES["01f69e1"]["ipg"],
            "shipped_stream_vector": P.TABLES["01f69e1"]["streams"],
            "shipped_boundaries": P.TABLES["01f69e1"]["boundaries"],
            "e41_ipg_table": P.TABLES["04ad6bf"]["ipg"],
            "e41_stream_vector": P.TABLES["04ad6bf"]["streams"],
            "e41_boundaries": P.TABLES["04ad6bf"]["boundaries"],
            "e41_a_per_row": P.E41_A,
            "e41_b_per_stream": P.E41_B,
            "e41_c_intercept": P.E41_C,
            "prereg_mde_rule": P.MDE_RULE,
            "prereg_sign_test": P.SIGN_TEST,
            "prereg_B_band_strict": list(P.B_BAND_STRICT),
            "prereg_B_band_lenient": list(P.B_BAND_LENIENT),
            "prereg_stop_rules": P.STOP_RULES,
            "not_shipped": P.NOT_SHIPPED,
            "register_ceiling": (census or {}).get("ceiling"),
            "register_census_all_arms_equal": (census or {}).get("all_arms_equal"),
            "register_census_any_exceeds": (census or {}).get("any_exceeds_ceiling"),
            # Thermal contract, recorded honestly: this host's real 40C gate is
            # unreachable (~43.4C floor), so every number here is directional
            # causal evidence inside one counterbalanced session, never a
            # gate-qualified or ranked score.
            "cool_gate": ident.get("base1", {}).get("cool_gate_vendored"),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "gpu_entry_c": {k: temp(k, "gpu_temp_c_before_vendored") for k in ident},
            "gpu_exit_c": {k: temp(k, "gpu_temp_c_after_vendored") for k in ident},
            "gpu_entry_c_spread": d.get("entry_temp_c", {}).get("spread"),
        },
    )

    tables = {}

    curve = wandb.Table(columns=["M", "ipg_base", "ipg_arm", "streams_base",
                                 "streams_arm", "base_ms", "arm_ms", "delta_ms",
                                 "delta_pct", "mde_ms", "exceeds_floor", "role"])
    disp = d["dispatch"]
    for m in d["widths"]:
        k = str(m)
        if k not in d.get("T_arm_ms", {}):
            continue
        base_ms = d["T_base_all_widths_ms"][k]
        arm_ms = d["T_arm_ms"][k]
        dl = d["delta_ms"][k]
        mde = d["mde_ms"][k]
        cell = disp.get(k, {}).get("base1", {})
        acell = disp.get(k, {}).get("arm1", {})
        role = ("contrast A" if m == 6 else "contrast B" if m == 8
                else "untreated control")
        curve.add_data(m, cell.get("want_ipg"), acell.get("want_ipg"),
                       cell.get("want_streams"), acell.get("want_streams"),
                       base_ms, arm_ms, dl, dl / base_ms * 100 if base_ms else None,
                       mde, bool(mde == mde and abs(dl) > mde), role)
    tables["e46/curve_by_width"] = curve

    d1 = wandb.Table(columns=["step", "measured_d1_ms", "prereg_H_streams_ms",
                              "replicate_floor_ms", "is_stream_boundary",
                              "is_argmax"])
    pred = dict(zip(P.D1_LABELS, P.PRED_D1))
    for lab in P.D1_LABELS:
        if lab not in d["d1_ms"]:
            continue
        d1.add_data(lab, d["d1_ms"][lab], pred[lab],
                    d["d1_replicate_floor_ms"].get(lab),
                    lab in ("4->5", "8->9"), lab == d["argmax_d1"])
    tables["e46/first_differences"] = d1

    con = wandb.Table(columns=["contrast", "M", "ipg_base", "ipg_arm",
                               "streams_base", "streams_arm", "width_base",
                               "width_arm", "delta_ms", "mde_ms", "verdict"])
    con.add_data("A", 6, 3, 4, 2, 2, "3+3", "4+2",
                 d["contrast"]["A"]["delta_ms"], d["contrast"]["A"]["mde_ms"],
                 "null" if d["contrast"]["A"]["null"] else "real effect")
    con.add_data("B", 8, 4, 3, 2, 3, "4+4", "3+3+2",
                 d["contrast"]["B"]["delta_ms"], d["contrast"]["B"]["mde_ms"],
                 "real effect" if d["contrast"]["B"]["real"] else "below floor")
    tables["e46/contrasts"] = con

    hyp = wandb.Table(columns=["hypothesis", "claim", "step2_prediction",
                               "contrast_A_prediction", "contrast_B_prediction",
                               "survives"])
    for name, h in P.HYPOTHESES.items():
        hyp.add_data(name, h["claim"], h["step2"], h["contrast_A"], h["contrast_B"],
                     name == d["verdict"]["surviving"])
    tables["e46/hypotheses"] = hyp

    shape = wandb.Table(columns=["contrast", "M", "shape", "delta_us_per_call",
                                 "delta_pct"])
    for name, s in d.get("sign_test", {}).items():
        for sh, us in s["per_shape_us"].items():
            shape.add_data(name, s["m"], sh, us, s["per_shape_pct"][sh])
    tables["e46/per_shape_deltas"] = shape

    jit = wandb.Table(columns=["M"] + [f"{k}_mean_over_min_pct"
                                       for k in sorted(ident)])
    for m, v in sorted(d["jit_spread_pct"].items(), key=lambda kv: int(kv[0])):
        jit.add_data(int(m), *[v.get(k) for k in sorted(ident)])
    tables["e46/jit_leak_check"] = jit

    if census:
        # The gate that let this experiment reach the GPU: every arm's register
        # max is the max over the seven per-width cases, not a per-NA cell.
        reg = wandb.Table(columns=["arm", "contrast", "treated_m", "edits",
                                   "kernel_wide_reg_max", "entry_point_reg_max",
                                   "argmax_width", "exceeds_ceiling",
                                   "max_total_threads_per_threadgroup"]
                          + [f"regs_M{m}" for m in P.WIDTHS])
        for z in census["cells"]:
            if z.get("status") != "ok":
                continue
            occ = z.get("occupancy", {}).get("functions", {})
            mtt = next(iter(occ.values()), {}).get(
                "max_total_threads_per_threadgroup")
            edits = "; ".join(f"{a} -> {b}" for a, b in z["edits"]) or "none"
            reg.add_data(z["name"], z["contrast"], z["treated_m"],
                         edits, z["kernel_wide_reg_max"],
                         z["entry_point_reg_max"], z["argmax_width"],
                         z["exceeds_ceiling"], mtt,
                         *[z["width_cells"][str(m)]["peak_live_regs"]
                           for m in P.WIDTHS])
        tables["e46/register_census"] = reg

    prior = d.get("prior_replication")
    if prior:
        pr = wandb.Table(columns=["M", "prior_delta_ms", "prior_pct",
                                  "e46_delta_ms", "e46_pct", "role"])
        for m, v in sorted(prior["delta"].items(), key=lambda kv: int(kv[0])):
            here = d["delta_ms"].get(str(m))
            base_ms = d["T_base_ms"].get(str(m))
            role = ("contrast A" if int(m) == 6 else "contrast B" if int(m) == 8
                    else "prior-only cell" if int(m) == 4 else "control")
            pr.add_data(int(m), v["delta_ms"], v["pct"], here,
                        here / base_ms * 100 if here and base_ms else None, role)
        tables["e46/prior_replication"] = pr

    run.log(tables)

    su = run.summary
    su["argmax_d1"] = d["argmax_d1"]
    su["argmax_d1_is_stream_boundary"] = d["step2"]["supports_H_streams"]
    su["surviving_hypothesis"] = d["verdict"]["surviving"]
    su["mechanism_name"] = d["verdict"]["mechanism_name"]
    su["delta_A_ms"] = d["contrast"]["A"]["delta_ms"]
    su["mde_A_ms"] = d["contrast"]["A"]["mde_ms"]
    su["contrast_A_is_null"] = d["contrast"]["A"]["null"]
    su["delta_B_ms"] = d["contrast"]["B"]["delta_ms"]
    su["mde_B_ms"] = d["contrast"]["B"]["mde_ms"]
    su["contrast_B_in_strict_band"] = d["contrast"]["B"]["in_strict_band"]
    su["contrast_B_in_lenient_band"] = d["contrast"]["B"]["in_lenient_band"]
    su["refit_b_per_stream_ms"] = d["refit_shipped_streams"]["b_per_stream"]
    su["refit_a_per_row_ms"] = d["refit_shipped_streams"]["a_per_row"]
    su["refit_c_intercept_ms"] = d["refit_shipped_streams"]["c_intercept"]
    su["refit_max_abs_resid_ms"] = d["refit_shipped_streams"]["max_abs_resid_ms"]
    su["refit_m6_indicator_max_abs_resid_ms"] = \
        d["refit_m6_indicator"]["max_abs_resid_ms"]
    su["dispatch_readback_ok"] = d["dispatch_readback_ok"]
    su["controls_exceeding_floor"] = d["controls_exceeding_floor"]
    su["worst_control_delta_pct"] = d["control_worst"]["pct"]
    for name, s in d.get("sign_test", {}).items():
        su[f"sign_test_{name}_n_positive"] = s["n_positive"]
        su[f"sign_test_{name}_p"] = s["p_two_sided"]
    su["cool_gate_passed_real_gate"] = False
    su["gate_qualified_for_timing"] = False

    print(f"logged E46 to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
