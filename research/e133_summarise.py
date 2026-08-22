#!/usr/bin/env python3
"""Compact reader for the E133 screen JSON, for the report and for W&B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RANK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
STRATA = ("beagle", "min_carriers", "zero_weight", "essays_bacon",
          "essays_bacon_holdout")


def show_cell(cell: dict, survival: bool = False) -> None:
    print(f"  arm                {cell['arm']}")
    print(f"  bytes/row          {cell['bytes_per_row']}  proj {cell['proj_bytes']}"
          f"  survivors {cell['survivors']}  probe {cell['probe_fraction']}"
          f"  cross_fit {cell['cross_fit']}")
    print(f"  stage bytes        {cell['arm_stage_bytes']} vs shipped "
          f"{cell['shipped_stage_bytes']}  removed {cell['removed_bytes']} "
          f"({100.0 * cell['removed_step_fraction']:.2f}% of step)")
    print(f"  net worst gating   {cell['net_miss_worst_gating']:.4e}   "
          f"hi {cell['net_miss_worst_gating_hi']:.4e}")
    print(f"  m_absolute worst   {cell['m_absolute_worst_gating']:.4e}   "
          f"hi {cell['m_absolute_worst_gating_hi']:.4e}")
    print(f"  m_incremental      {cell['m_incremental_worst_gating']:.4e}")
    print(f"  acceptance loss    F1.5 {cell['acceptance_loss_worst_gating']:.4e}"
          f"  pooled {cell['acceptance_loss_pooled_worst_gating']:.4e}"
          f"  live substitutions {cell['substitutions_live_gating']}")
    print(f"  recall worst       {cell['recall_worst_gating']:.5f}")
    print(f"  gain% 7/9 share    {cell['pct_head_share_7']:.3f} / "
          f"{cell['pct_head_share_9']:.3f}   byte-rate {cell['pct_byte_rate']:.3f}")
    print(f"  pred% F1.5 loss    {cell['predicted_pct_gating']:.3f}")
    print(f"  pred% pooled p     {cell['predicted_pct_pooled']:.3f}")
    print(f"  pred% raw mInc     {cell['predicted_pct_raw_miss']:.3f}")
    print(f"  T0 {cell['passes_t0']}   T0b {cell['passes_t0b']}")
    for s in STRATA:
        v = cell["by_stratum"].get(s)
        if not v:
            continue
        print(f"    {s:14s} n={v['n']:5d} net={v['net_miss']:.4e} "
              f"abs={v['m_absolute']:.4e} "
              f"CI[{v['m_absolute_lo']:.3e},{v['m_absolute_hi']:.3e}] "
              f"rec={v['recall']:.5f} probe={v['probe_hit_rate']:.5f} "
              f"surv={v['survivor_hit_rate']:.5f} "
              f"loss={v['acceptance_loss']:.4e} subs_live={v['substitutions_live']}")
    if not survival:
        return
    for s in ("beagle", "min_carriers"):
        v = cell["by_stratum"].get(s)
        if not v:
            continue
        cur, n = v["survival_curve"], v["n"]
        print(f"  survival {s}: fraction whose affine-2 top-1 needs width >= N")
        print("    " + "  ".join(f"{g}:{cur[str(g)] / n:.5f}" for g in RANK_GRID
                                 if str(g) in cur))
        print(f"    tail fit: {v['tail_fit_at_survivors']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/e133-screen.json")
    ap.add_argument("--arms", default="")
    ap.add_argument("--survival", action="store_true")
    args = ap.parse_args()
    out = json.loads(Path(args.json).read_text())

    print(f"samples {out['samples']}  base {out['base_sha']}  "
          f"wall {out['wall_seconds']:.1f}s")
    print(f"p_head_step_accuracy {out['p_head_step_accuracy']:.6f}  "
          f"p_row_accepted {out['p_row_accepted']:.6f}  "
          f"offline_shipped_chain_reproduces_runtime "
          f"{out['offline_shipped_chain_reproduces_runtime']:.6f}")

    sh, px = out["shipped"], out["shipped_structural_proxy"]
    print("\n=== shipped baseline (E121 derived index) true miss vs affine-4 argmax")
    for s in STRATA:
        v = sh["by_stratum"][s]
        print(f"  {s:14s} n={v['n']:5d} miss={v['m_absolute']:.4e} "
              f"CI[{v['m_absolute_lo']:.3e},{v['m_absolute_hi']:.3e}] "
              f"({v['misses_absolute']} of {v['n']})  "
              f"head_step_acc={v['p_head_step_accuracy']:.5f}")
    print(f"  worst gating   {sh['m_absolute_worst_gating']:.6e}")
    print(f"  structural proxy worst gating {px['m_absolute_worst_gating']:.6e}"
          f"  underestimate "
          f"{sh['m_absolute_worst_gating'] / max(px['m_absolute_worst_gating'], 1e-12):.1f}x")

    print("\n=== query basis energy (cross-fit)")
    for pair, keeps in out["query_basis"]["energy_kept"].items():
        print(f"  {pair:36s} "
              + " ".join(f"k{r}={e:.4f}" for r, e in keeps.items()))

    for stage, blk in out["by_stage_a"].items():
        print(f"\n=== stage_a={stage} ({blk['label']})")
        print(f"  cells {blk['cells']}  T0 {blk['cells_passing_t0']}  "
              f"T0b {blk['cells_passing_t0b']}  both {blk['cells_passing_both']}")
        if blk["best_cell"]:
            show_cell(blk["best_cell"], survival=args.survival)

    ctrl = [c for c in out["cells"] if c["family"] == "exact"]
    bad = [c for c in ctrl if c["m_incremental_worst_gating"] != 0.0
           or c["recall_worst_gating"] != 1.0
           or c["net_miss_worst_gating"] != 0.0]
    print(f"\n=== exact0 control: {len(ctrl)} cells, {len(bad)} with nonzero "
          f"incremental miss, nonzero net, or recall<1")
    for c in ctrl[:3]:
        print(f"  {c['arm']:36s} mInc={c['m_incremental_worst_gating']:.3e} "
              f"net={c['net_miss_worst_gating']:.3e} "
              f"rec={c['recall_worst_gating']:.6f} "
              f"abs={c['m_absolute_worst_gating']:.6e}")

    for arm in [a for a in args.arms.split(",") if a]:
        cell = next((c for c in out["cells"] if c["arm"] == arm), None)
        if cell is None:
            print(f"\n!! no such arm {arm}")
            continue
        print(f"\n=== {arm}")
        show_cell(cell, survival=args.survival)


if __name__ == "__main__":
    main()
