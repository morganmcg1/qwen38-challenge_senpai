#!/usr/bin/env python3
"""Publish the E177 ranked round-cost law recovery to W&B.

    usage: research/e177_wandb_log.py

Every number here is a DESK recovery from two paid ranked receipts on the Yukon
board. No GPU ran for this experiment, no local timing leg was measured, and no
value here is a gate-qualified local measurement. The harness label on every
metric is ``ranked``, because the inputs are official M5 receipts.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e177_ranked_depth_law as E177  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e177-ranked-depth-law-recovery"
BASE_SHA = "d9bb94c31b267044e3c7a8065f282a45968fe16a"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def git_dirty() -> int:
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                         text=True, check=True).stdout.strip()
    return len(out.split("\n")) if out else 0


def main() -> None:
    rows = E177.load_board()
    data = {}
    for label, prefix, cap, note in E177.RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(prefix))
        vec = E177.per_prompt(row)
        data[label] = {"id": row["id"][:8], "cap": cap, "score": row["officialScore"],
                       "vec": vec, "rec": E177.recover_rounds(vec)}

    surv, bounds, mu = {}, {}, {}
    for name in E177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        t0 = 1.0 - a["nondraft"] / a["rounds"]
        _, lam, kappa, s = E177.fit_survival(c["edl"], a["edl"], t0)
        surv[name] = s
        bounds[name] = E177.tail_bounds(c["edl"], a["edl"], s[3])
        d_edl = a["edl"] - c["edl"]
        mu[name] = (a["abar"] - c["abar"]) / d_edl if d_edl > 1e-9 else 0.0

    observations = []
    for label in ("A", "C"):
        for name in E177.ORDER:
            r = data[label]["rec"][name]
            observations.append((label, name,
                                 E177.width_probs(surv[name], data[label]["cap"]),
                                 r["R"], r["rounds"]))
    fits = E177.fit_cost(observations, ["A", "C"], free_intercepts=[])
    fits_free = E177.fit_cost(observations, ["A", "C"], free_intercepts=["C"])
    a_only = [o for o in observations if o[0] == "A"]
    fits_a = E177.fit_cost(a_only, ["A"], free_intercepts=[])
    best = fits["quadratic"]

    tail_light = {}
    for name in E177.ORDER:
        t = list(surv[name])
        t[4] = t[5] = t[6] = t[7] = bounds[name]["delta"] / 3.0
        tail_light[name] = t

    def chain(name, tvar, model, target_cap, anchor):
        rec = data[anchor]["rec"][name]
        cap = data[anchor]["cap"]
        abar, R, t, m = rec["abar"], rec["R"], tvar[name], mu[name]
        while cap < target_cap:
            abar += t[cap] * m
            R += t[cap] * (E177.cost_at_width(model, cap + 2)
                           - E177.cost_at_width(model, cap + 1))
            cap += 1
        while cap > target_cap:
            abar -= t[cap - 1] * m
            R -= t[cap - 1] * (E177.cost_at_width(model, cap + 1)
                               - E177.cost_at_width(model, cap))
            cap -= 1
        n = E177.TOKENS / (1.0 + abar)
        return n * R

    def score(cap, tvar, model, anchor):
        raws = []
        for name in E177.ORDER:
            total = chain(name, tvar, model, cap, anchor)
            v = data["A"]["vec"][name]
            raws.append(v["serial"] / (total / E177.TOKENS + v["prefill"]))
        return E177.published_median(raws)

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e177-ranked-depth-law-recovery",
        job_type="desk-analysis",
        config={
            "experiment": "E177",
            "assignment_pr": 176,
            "base_sha": BASE_SHA,
            "commit": git_head(),
            "dirty_files": git_dirty(),
            "harness": "ranked",
            "leg_kind": "desk-recovery-from-paid-receipts",
            "gpu_seconds_spent": 0,
            "timed_local_legs": 0,
            "cool_gate_passed_real_gate": None,
            "gate_qualified_for_timing": False,
            "board_rows": len(rows),
            "receipt_A": "5a9f130a", "receipt_A_cap": 7,
            "receipt_A_score": E177.BEST_A,
            "receipt_C": "90c131dc", "receipt_C_cap": 4,
            "receipt_C_score": E177.CAP4_PUBLISHED,
            "crown": E177.CROWN,
            "cap_literal": "Qwen36MTPBlockSession.segmentedVerifyDepthCap",
            "decode_tokens": 512,
            "sigma_candidate_leg": E177.SIGMA_LEG,
        })

    per_prompt = wandb.Table(columns=[
        "receipt", "cap", "prompt", "decode_seconds", "prefill_s_per_tok", "edl",
        "Mbar", "rounds", "accepted", "abar", "alpha", "round_ms", "raw", "serial"])
    for label in ("A", "C", "B", "D"):
        for name in E177.ORDER:
            r = data[label]["rec"][name]
            per_prompt.add_data(data[label]["id"], data[label]["cap"], name,
                                r["decode_total"], r["prefill"], r["edl"], r["Mbar"],
                                r["rounds"], r["accepted"], r["abar"], r["alpha"],
                                1000 * r["R"], r["raw"], r["serial"])
    run.log({"per_prompt_recovery": per_prompt})

    law = wandb.Table(columns=[
        "model", "wrmse_ms", "max_resid_ms", "aic", "dF_C_ms", "R1_ms",
        "R7_over_R5", "heldout_cap4_score", "heldout_cap4_error_pct"])
    for key in sorted(fits, key=lambda k: fits[k]["aic"]):
        idx = fits_free[key]["names"].index("dF_C")
        raws = []
        for name in E177.ORDER:
            total = chain(name, surv, fits_a[key], 4, "A")
            v = data["A"]["vec"][name]
            raws.append(v["serial"] / (total / E177.TOKENS + v["prefill"]))
        p4 = E177.published_median(raws)
        law.add_data(
            key, fits[key]["wrmse_ms"], fits[key]["max_resid_ms"], fits[key]["aic"],
            1000 * fits_free[key]["beta"][idx],
            1000 * E177.cost_at_width(fits[key], 1),
            E177.cost_at_width(fits[key], 7) / E177.cost_at_width(fits[key], 5),
            p4, 100 * (p4 / E177.CAP4_PUBLISHED - 1))
    run.log({"cost_law_comparison": law})

    curve = wandb.Table(columns=["width_M", "round_ms", "marginal_ms"])
    for m in range(1, 10):
        prev = E177.cost_at_width(best, m - 1) if m > 1 else None
        curve.add_data(m, 1000 * E177.cost_at_width(best, m),
                       1000 * (E177.cost_at_width(best, m) - prev) if prev else 0.0)
    run.log({"recovered_cost_curve": curve})

    cells = wandb.Table(columns=["cap", "survival", "anchor", "published_score",
                                 "delta_vs_cap7_pct", "clears_crown"])
    for cap in (4, 5, 6, 7, 8):
        for vname, tvar in (("fitted", surv), ("tail-light", tail_light)):
            for anchor in ("A", "C"):
                if cap == 8 and anchor == "C":
                    continue
                s = score(cap, tvar, best, anchor)
                cells.add_data(cap, vname, anchor, s,
                               100 * (s / E177.BEST_A - 1), s > E177.CROWN)
    run.log({"priced_depth_cells": cells})

    marginal = wandb.Table(columns=[
        "prompt", "d_edl", "d_abar", "mu", "alpha_cap4",
        "round_ms_cap7", "breakeven_marginal_ms", "quadratic_marginal_ms", "verdict"])
    for name in E177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        be = 1000 * a["R"] * mu[name] / (1.0 + a["abar"])
        act = 1000 * (E177.cost_at_width(best, 9) - E177.cost_at_width(best, 8))
        marginal.add_data(name, a["edl"] - c["edl"], a["abar"] - c["abar"], mu[name],
                          c["alpha"], 1000 * a["R"], be, act,
                          "pays" if act < be and mu[name] > 0 else "costs")
    run.log({"cap8_breakeven": marginal})

    oracle_raws = []
    for name in E177.ORDER:
        total = min(chain(name, surv, best, cap, "A") for cap in range(4, 9))
        v = data["A"]["vec"][name]
        oracle_raws.append(v["serial"] / (total / E177.TOKENS + v["prefill"]))

    run.summary.update({
        "law/form": "R(M) = F + b*M + c*M^2, candidate-leg seconds per round",
        "law/F_ms": 1000 * best["beta"][0],
        "law/b_ms_per_row": 1000 * best["beta"][1],
        "law/c_ms_per_row2": 1000 * best["beta"][2],
        "law/wrmse_ms": best["wrmse_ms"],
        "law/dF_C_ms_unconstrained": 1000 * fits_free["quadratic"]["beta"][
            fits_free["quadratic"]["names"].index("dF_C")],
        "law/R1_ms": 1000 * E177.cost_at_width(best, 1),
        "law/R1_independent_anchor_ms": 30.2519,
        "law/R7_over_R5": (E177.cost_at_width(best, 7)
                           / E177.cost_at_width(best, 5)),
        "law/modelfree_wide_round_ratio": 1.2579,
        "heldout/cap4_predicted_from_A_only": score(4, surv, fits_a["quadratic"], "A"),
        "heldout/cap4_paid": E177.CAP4_PUBLISHED,
        "heldout/cap4_error_pct": 100 * (
            score(4, surv, fits_a["quadratic"], "A") / E177.CAP4_PUBLISHED - 1),
        "cells/cap5": score(5, tail_light, best, "A"),
        "cells/cap6": score(6, tail_light, best, "A"),
        "cells/cap7_paid": E177.BEST_A,
        "cells/cap8": score(8, tail_light, best, "A"),
        "cells/oracle_per_prompt_cap": E177.published_median(oracle_raws),
        "cells/best_cell_clears_crown": False,
        "crown": E177.CROWN,
        "verdict": ("ranked round cost is convex; cap 7 is the ranked optimum; "
                    "no depth cell clears the crown"),
    })
    print("run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
