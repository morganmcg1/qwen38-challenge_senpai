#!/usr/bin/env python3
"""Log the E171 r2 factor decomposition to W&B.

Desk analysis only: it reads the cached Yukon board payload and the derived
numbers from ``research/e171_r2_decompose.py``. No GPU leg is measured here.
"""

import statistics as st

import wandb

from e171_r2_decompose import (DRAFTING, RECEIPTS, delta_pct, edl, load, mean7,
                               pick, serial_null, vec)

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
PAIRS = [("D_vs_A", "A", "D", "Q + E165 + P"),
         ("C_vs_A", "A", "C", "I + P"),
         ("B_vs_C", "C", "B", "Q"),
         ("B_vs_A", "A", "B", "Q + I + P"),
         ("D_vs_B", "B", "D", "E165 - I")]


def main():
    scored = load()
    rows = {k: pick(scored, v) for k, v in RECEIPTS.items()}
    floor = st.stdev(serial_null(scored)) * (2 ** 0.5)
    computed = {n: delta_pct(rows[b], rows[c]) for n, b, c, _ in PAIRS}
    e = edl(rows["A"])

    run = wandb.init(
        entity="wandb-applied-ai-team", project="qwen38-mlx-challenge-senpai",
        name="e171-r2-factor-decomposition",
        job_type="desk-analysis",
        tags=["e171", "r2", "harness=ranked", "decomposition", "no-gpu"],
        config={
            "harness": "ranked",
            "assignment": "e171-compose-and-fire-the-crown-attempt",
            "revision": "r2",
            "frozen_sha": "4ba44f8251960eaf79ec4f05135485442438f5ef",
            "receipts": RECEIPTS,
            "null_floor_pct_1sd": floor,
            "receipt_published_sd_pct": 0.271,
            "crown_score": 3.72911001,
        })

    per_prompt = wandb.Table(
        columns=["prompt", "edl", "A_mtp", "B_mtp", "C_mtp", "D_mtp"]
        + [p[0] + "_pct" for p in PAIRS])
    for n in sorted(e, key=lambda k: e[k]):
        per_prompt.add_data(
            n, e[n],
            *[vec(rows[k])[n]["mtp_seconds_per_token_mean"] for k in "ABCD"],
            *[computed[p[0]][n] for p in PAIRS])

    xs = [e[n] for n in DRAFTING]
    factors = wandb.Table(
        columns=["pair", "meaning", "mean7_pct", "sd_pct", "plutarch_pct",
                 "sigma_vs_null", "edl_corr", "edl_slope_pct_per_edl"])
    scalars = {}
    for name, _, _, meaning in PAIRS:
        d = computed[name]
        ys = [d[n] for n in DRAFTING]
        m = mean7(d)
        factors.add_data(name, meaning, m, st.stdev(ys), d["plutarch"],
                         m / floor, st.correlation(xs, ys),
                         st.linear_regression(xs, ys).slope)
        scalars["mean7_pct/" + name] = m

    DA, BC, DB = (scalars["mean7_pct/D_vs_A"], scalars["mean7_pct/B_vs_C"],
                  scalars["mean7_pct/D_vs_B"])
    scalars.update({
        "factor/Q_pct": BC,
        "factor/E165_lower_bound_pct": DB,
        "factor/E165_upper_bound_pct": DA - BC,
        "factor/E165_sigma_lower": DB / floor,
        "factor/tau_lower": -(DA - BC) / 1.150,
        "factor/tau_upper": -DB / 1.150,
        "null_floor_pct_1sd": floor,
        "official/D_score": rows["D"]["officialScore"],
        "official/A_score": rows["A"]["officialScore"],
    })

    run.log({"per_prompt": per_prompt, "factors": factors, **scalars})
    run.summary.update(scalars)
    print("logged", run.url)
    run.finish()


if __name__ == "__main__":
    main()
