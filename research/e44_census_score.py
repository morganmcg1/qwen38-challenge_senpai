"""Convert E44 r2 per-(M, shape) speedups into score deltas under a width census.

Reports every cell separately so a future census can be re-applied without
re-running the timing session. Optionally appends the tables to the W&B run.
"""

import argparse
import json

PSI_MTP = 0.6736

SPEEDUP_PCT = {
    ("attn_out", 7): 11.389,
    ("mlp_down", 7): 4.596,
    ("attn_out", 8): 17.050,
    ("mlp_down", 8): 12.649,
}
CI_PCT = {
    ("attn_out", 7): (11.284, 11.495),
    ("mlp_down", 7): (4.506, 4.685),
    ("attn_out", 8): (16.876, 17.225),
    ("mlp_down", 8): (12.478, 12.820),
}
BASE_US = {
    ("attn_out", 7): 154.03,
    ("mlp_down", 7): 486.07,
    ("attn_out", 8): 165.96,
    ("mlp_down", 8): 523.81,
}

# askeladd E42 corpus census, tree 04ad6bf11437c269df85a47e91faa769c74fe6da.
CENSUS_COUNTS = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
CENSUS_COST_SHARE = {2: 0.0054, 4: 0.0350, 5: 0.0507, 6: 0.2519,
                     7: 0.0471, 8: 0.0755, 9: 0.5345}
BOARD_FLOOR_PCT = 0.7678
CROWN_GAP_PCT = 0.5193

TOUCHED = (7, 8)
SHAPES = ("attn_out", "mlp_down")


def touched_share():
    return sum(CENSUS_COST_SHARE[m] for m in TOUCHED)


def cell_rows():
    rows = []
    for shape in SHAPES:
        for m in TOUCHED:
            sp = SPEEDUP_PCT[(shape, m)]
            lo, hi = CI_PCT[(shape, m)]
            rows.append({
                "shape": shape,
                "M": m,
                "speedup_pct": sp,
                "ci_lo_pct": lo,
                "ci_hi_pct": hi,
                "dscore_per_1pp_of_f_pct": PSI_MTP * sp * 0.01,
                "dscore_at_census_f_shape_pure_pct":
                    PSI_MTP * sp * CENSUS_COST_SHARE[m],
            })
    return sorted(rows, key=lambda r: (r["M"], r["shape"]))


def effective_speedup(width_speedup):
    f = touched_share()
    return sum(width_speedup(m) * CENSUS_COST_SHARE[m] for m in TOUCHED) / f


def cost_proportional(m):
    a, d = BASE_US[("attn_out", m)], BASE_US[("mlp_down", m)]
    total = a + d
    return (SPEEDUP_PCT[("attn_out", m)] * a
            + SPEEDUP_PCT[("mlp_down", m)] * d) / total


def mixture_rows(f_override=None):
    f = touched_share() if f_override is None else f_override
    mixes = [
        ("all mlp_down", lambda m: SPEEDUP_PCT[("mlp_down", m)]),
        ("cost-proportional", cost_proportional),
        ("equal weight per cell",
         lambda m: (SPEEDUP_PCT[("attn_out", m)]
                    + SPEEDUP_PCT[("mlp_down", m)]) / 2),
        ("all attn_out", lambda m: SPEEDUP_PCT[("attn_out", m)]),
    ]
    rows = []
    for name, fn in mixes:
        eff = effective_speedup(fn)
        ds = PSI_MTP * eff * f
        rows.append({
            "shape_mixture": name,
            "f_touched": f,
            "effective_speedup_pct": eff,
            "dscore_pct": ds,
            "clears_board_floor": ds >= BOARD_FLOOR_PCT,
            "clears_crown_gap": ds >= CROWN_GAP_PCT,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-run", help="existing run id to append tables to")
    ap.add_argument("--wandb-project",
                    default="wandb-applied-ai-team/qwen38-mlx-challenge-senpai")
    ap.add_argument("--half-f", type=float, default=0.06,
                    help="reduced f for the scoring-prompt sensitivity row")
    args = ap.parse_args()

    f = touched_share()
    cells = cell_rows()
    census_mix = mixture_rows()
    half_mix = mixture_rows(args.half_f)
    pooled = sum(SPEEDUP_PCT.values()) / len(SPEEDUP_PCT)
    pooled_dscore = PSI_MTP * pooled * f

    payload = {
        "psi_mtp": PSI_MTP,
        "census_tree": "04ad6bf11437c269df85a47e91faa769c74fe6da",
        "census_counts": CENSUS_COUNTS,
        "census_cost_share": CENSUS_COST_SHARE,
        "f_touched_7_8": f,
        "board_floor_pct": BOARD_FLOOR_PCT,
        "crown_gap_pct": CROWN_GAP_PCT,
        "cells": cells,
        "census_mixtures": census_mix,
        "half_f_mixtures": half_mix,
        "pooled_speedup_pct": pooled,
        "pooled_then_weighted_dscore_pct": pooled_dscore,
        "weighted_then_summed_equal_mix_dscore_pct":
            next(r["dscore_pct"] for r in census_mix
                 if r["shape_mixture"] == "equal weight per cell"),
    }
    payload["pooling_bias_pp"] = (
        payload["weighted_then_summed_equal_mix_dscore_pct"] - pooled_dscore)
    print(json.dumps(payload, indent=2, sort_keys=False))

    if not args.wandb_run:
        return
    import wandb

    entity, project = args.wandb_project.split("/", 1)
    run = wandb.init(entity=entity, project=project, id=args.wandb_run,
                     resume="must")
    run.log({
        "e44_score_cells": wandb.Table(
            columns=list(cells[0].keys()),
            data=[list(r.values()) for r in cells]),
        "e44_census_mixtures": wandb.Table(
            columns=list(census_mix[0].keys()),
            data=[list(r.values()) for r in census_mix + half_mix]),
    })
    run.summary.update({
        f"score/{k}": v for k, v in payload.items()
        if isinstance(v, (int, float))
    })
    run.summary["score/census_json"] = json.dumps(payload)
    run.finish()


if __name__ == "__main__":
    main()
