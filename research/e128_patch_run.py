"""Backfill the anchor-credibility fields onto an already-published run.

`e128_wandb_log.py` writes these fields now, but run `mys5l3kq` was published
before they existed. Patching the existing run keeps one run per result instead
of creating a duplicate.

    python3 research/e128_patch_run.py mys5l3kq
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

ART = Path("research/e128-artifacts")
PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
EXCLUDED = {"oracle", "ship", "price0.18"}


def main() -> int:
    run_id = sys.argv[1]
    base = json.loads((ART / "rung2-ours-pricing.json").read_text())
    band = json.loads((ART / "rung2-ours-r-band.json").read_text())
    sweep = json.loads((ART / "rung2-curve-sweep.json").read_text())

    primary = base["transfer"]
    alt = band["scenarios"]["assumed"]
    depth_err_primary = max(
        abs(e["simulated_depth_ship"] - e["target_depth_f92"])
        for e in primary.values())
    depth_err_alt = max(
        abs(alt["simulated_depth_ship"][p] - e["target_depth_f92"])
        for p, e in primary.items())

    best = {}
    for key, entry in sweep["curves"].items():
        imp = {a: v for a, v in entry["median_gain_pct_vs_ship"].items()
               if a not in EXCLUDED}
        best[key] = max(imp.values())
    positive = sorted(k for k, v in best.items() if v > 0)
    well = {k: v for k, v in best.items() if k != "predicted"}

    note = (
        "The R band uses the accept-rate anchor, which misses the published "
        "F92 depth by up to %.3f tokens against %.3f for the depth anchor "
        "used by the headline. Its positive arms price counterfactuals "
        "against a shipped baseline that the published depths rule out."
        % (depth_err_alt, depth_err_primary))

    run = wandb.Api().run("%s/%s" % (PROJECT, run_id))
    run.summary.update({
        "transfer_depth_anchor_max_abs_depth_error": depth_err_primary,
        "transfer_accept_anchor_max_abs_depth_error": depth_err_alt,
        "transfer_headline_anchor": "depth",
        "r_band_positive_arms_are_credible": False,
        "r_band_credibility_note": note,
        "curve_sweep_positive_curves": positive,
        "curve_sweep_best_implementable_pct_well_fitted_max":
            max(well.values()),
        "curve_sweep_any_well_fitted_curve_positive":
            any(v > 0 for v in well.values()),
    })
    run.update()
    print("depth anchor max depth error %.3f, accept anchor %.3f"
          % (depth_err_primary, depth_err_alt))
    print("positive curves %s, well-fitted best %.4f"
          % (positive, max(well.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
