#!/usr/bin/env python3
"""E136 F4 section 5: price a probe-fraction grid on both cost models.

    usage: research/e136_probe_grid.py [--screen research/e136-probe-grid.json]
                                       [--json OUT.json]

The rival crown `02742bf0` moved `qwen35DerivedClusterProbeFraction` from 0.25
to 0.15 and nothing else. Nobody has fitted the curve, so this prices the whole
grid rather than the rival's single point.

TWO COST MODELS, and that is the point of the exercise.

Under the shipped chain a probed row costs `AFFINE2_ROW_BYTES = 1,600 B`.
Under C1 a probed row costs `rank + 8 = 264 B`. Lowering `p` therefore buys
about 6x fewer bytes under C1 than under the shipped chain, while the
acceptance it costs is the same object. The argmax `p` must move UP under C1.

This module owns only the COST side, which is exact arithmetic on the byte
model and needs no GPU. The ACCEPTANCE side comes from the E133 corpus replay:

    research/e133_screen.py screen --families exact0,lowrank256 \\
        --widths 4096 --probes 0.10,0.125,0.15,0.175,0.20,0.25,0.35 \\
        --stage-a sketch --out research/e136-probe-grid.json

Pass that file with `--screen` to merge the measured columns in. Never run the
replay while a timed leg is on the GPU: it is an MLX program on the same
device and the same run lock.
"""

from __future__ import annotations

import argparse
import json
import math

# Restated from research/e133_screen.py rather than imported, because that
# module pulls in MLX and the checkpoint loader at import time and this script
# must be safe to run beside a timed leg. `--self-check` asserts they agree.
LEAVES = 12_292
ROWS_PER_LEAF = 8
AFFINE2_ROW_BYTES = 1_600
AFFINE4_ROW_BYTES = 2_880
SHORTLIST = 32
HIDDEN = 5_120
MU_BYTES = HIDDEN * 2
STEP_BYTES = 323.59e6
HEAD_SHARE_LO = 0.07

# The C1 cell as built: rank-256 int8 codes, one fp32 scale and one fp32
# offset per row, a bf16 basis read once per draft step, 4,096 survivors
# rescored with the exact affine-2 row.
C1_RANK = 256
C1_ROW_BYTES = C1_RANK + 8
C1_PROJ_BYTES = C1_RANK * HIDDEN * 2
C1_SURVIVORS = 4_096

SHIPPED_P = 0.25
C1_P = 0.35
# E139 F2 section 2 extends the ladder below 0.10. Recall was still exactly
# 1.000000 at the old bottom of the grid, so the measured argmax was set by
# where the sampling stopped, not by where the mechanism breaks.
GRID = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def clusters(p: float) -> int:
    return max(1, math.ceil(p * LEAVES))


def shipped_stage_bytes(p: float) -> int:
    return (LEAVES * AFFINE2_ROW_BYTES
            + clusters(p) * ROWS_PER_LEAF * AFFINE2_ROW_BYTES
            + SHORTLIST * AFFINE4_ROW_BYTES)


def c1_stage_bytes(p: float) -> int:
    return (LEAVES * C1_ROW_BYTES
            + clusters(p) * ROWS_PER_LEAF * C1_ROW_BYTES
            + C1_SURVIVORS * AFFINE2_ROW_BYTES
            + C1_PROJ_BYTES + MU_BYTES
            + SHORTLIST * AFFINE4_ROW_BYTES)


def gross_pct(removed_bytes: float) -> float:
    """The 0.02167 %/MB line, stated as the 7 % head share of the step budget.

    100 * 0.07 / 323.59 MB = 0.021632 %/MB, which is the advisor's coefficient.
    """
    return 100.0 * HEAD_SHARE_LO * removed_bytes / STEP_BYTES


def rows_of(model: str):
    if model == "shipped":
        bytes_at, anchor, row_bytes = shipped_stage_bytes, SHIPPED_P, AFFINE2_ROW_BYTES
    else:
        bytes_at, anchor, row_bytes = c1_stage_bytes, C1_P, C1_ROW_BYTES
    base = bytes_at(anchor)
    out = []
    for p in GRID:
        step = bytes_at(p)
        removed = base - step
        out.append({
            "cost_model": model,
            "p": p,
            "probes": clusters(p),
            "coarse_rows": clusters(p) * ROWS_PER_LEAF,
            "row_bytes": row_bytes,
            "stage_bytes": step,
            "anchor_p": anchor,
            "removed_vs_anchor_bytes": removed,
            "gross_pct_vs_anchor": gross_pct(removed),
        })
    return out


def marginal(rows):
    """Bytes bought per 0.01 of `p`, which is the exchange rate the argmax
    depends on. It is constant in `p` up to the ceiling, so one number per
    cost model is the whole cost side of the curve."""
    row_bytes = rows[0]["row_bytes"]
    return 0.01 * LEAVES * ROWS_PER_LEAF * row_bytes


# RETIRED by E139 F2 / FINDING 192. Do not reinstate without a new receipt.
#
# This factor was `0.0992 / 0.3403 = 0.2915`. The numerator was read off the
# RAW ranked ratio, which is `serial / candidate`. The pinned serial leg is
# hardware the candidate cannot touch and its residual standard deviation is
# 0.4788 % against the candidate leg's 0.0498 %, so reading a candidate-side
# mechanism off the raw ratio imports 9.6 times the noise for nothing
# (Rule 112). Re-read off the CANDIDATE leg, the same p=0.25 -> 0.15 edit has
# two independent ranked receipts on two different bases:
#
#   b6cb0fea -> 02742bf0   wide-grid base    +0.3836 %
#   ed608e64 -> 08b67f12   tight-grid base   +0.2786 %
#   pooled                                   +0.3311 %  +/- 0.0948 (2 sigma)
#
# against this file's unscaled byte model of +0.3403 %, a ratio of 0.973. The
# byte model was never biased, so there is nothing to scale. E139 then
# measured the acceptance term directly on the zero-noise channel and found
# it exactly zero at p=0.15 and p=0.10 on two fixtures in different
# acceptance regimes, which is the same conclusion from the other side.
#
# The `*_scaled` columns are kept at unity rather than deleted so that older
# E136 artefacts still load and so the retraction stays visible here.
BOARD_SCALE = 1.0

# `e133_screen` names the ARM `exact0-N4096-p0.25` but the FAMILY `exact`, and
# an unrecognised family here silently anchors the shipped ladder on C1's
# probe fraction, so the names are asserted rather than compared inline.
SHIPPED_FAMILY = "exact"
C1_FAMILY = "lowrank"

NULL_FLOOR = "research/e136-null-floor.json"

# `e133_screen.summarize` emits these names. Restating them here as a mapping
# rather than reading them inline keeps one place to fix if the screen renames
# a column, and makes a silent all-None merge impossible: `merge_screen`
# asserts every source key is present on the first cell it sees.
SCREEN_FIELDS = {
    "recall_wg": "recall_worst_gating",
    # E139: the probe-stage margin. Recall can hold at 1.000 while this is
    # already falling, so it is the column that shows the knee approaching.
    "probe_hit_wg": "probe_hit_rate_worst_gating",
    "survivor_hit_wg": "survivor_hit_rate_worst_gating",
    "acc_loss_wg": "acceptance_loss_worst_gating",
    "acc_loss_pooled_wg": "acceptance_loss_pooled_worst_gating",
    "net_miss_wg": "net_miss_worst_gating",
    "m_absolute_wg": "m_absolute_worst_gating",
    "gross_pct": "pct_head_share_7",
    "predicted_pct_gating": "predicted_pct_gating",
    "predicted_pct_absolute": "predicted_pct_absolute",
    "bytes_per_row": "bytes_per_row",
    "stage_bytes_screen": "removed_bytes",
    "n_gating": "n_gating",
    "passes_t0": "passes_t0",
    "passes_t0b": "passes_t0b",
}


# Written before the E133 replay ran, so the replay can falsify it. The cost
# side is exact arithmetic and needs no GPU; only the acceptance side is
# measured, so the prediction is a genuine risk.
PREDICTION = {
    "written_before_screen_ran": True,
    "5a_shipped_argmax": "the shipped ladder keeps falling below p=0.15, "
                         "because gross rises linearly at 0.0340 %/0.01p "
                         "while recall is already 1.000 at the probe stage; "
                         "the argmax is set by where acc_loss finally turns "
                         "up, not by where gross stops paying",
    "5b_c1_argmax": "under C1 the WHOLE probe lever from p=0.35 to p=0.10 is "
                    "worth +0.1404 % gross, which is below the pooled 2-sigma "
                    "clustered null floor of 0.1462 % that the acceptance "
                    "side is measured against. So there is no argmax worth "
                    "acting on under C1: the C1 sketch already removed the "
                    "bytes the probe fraction was buying. C1 and the "
                    "probe-fraction rider are substitutes, not complements.",
    "5b_direction_check": "the advisor predicted the C1 argmax moves UP. The "
                          "cost side agrees in direction, and adds that the "
                          "move is too small to measure or to ship.",
    "risk": "C1 orders leaves with the sketch, not with the affine-2 centroid "
            "readout. If the sketch ordering is worse, recall at a given p "
            "falls faster under C1 and the acceptance loss of lowering p is "
            "LARGER under C1 than under the shipped chain. That would "
            "strengthen this conclusion, not weaken it. The falsifier is the "
            "opposite: recall_wg under lowrank256 at low p must not be "
            "materially BETTER than under exact0.",
}


def null_floors(path=NULL_FLOOR):
    payload = json.loads(open(path, encoding="utf-8").read())
    out = {}
    for arm, block in payload["statistics"].items():
        for scope, stats in block["pooled"].items():
            out[f"{arm}:{scope}"] = {
                "n": stats["n"],
                "two_sigma_null_pct": stats["two_sigma_band_null_ranked_pct"],
                "two_sigma_clustered_pct":
                    stats["two_sigma_band_clustered_ranked_pct"],
                "floor_pure_gain_pct": stats["floor_pure_gain_ranked_pct"],
            }
    return out


def merge_screen(path):
    """Build one probe ladder per family from an E133 screen run.

    The screen prices every cell against the shipped chain, so
    `predicted_pct_gating` already carries a constant offset per family. The
    probe-fraction question is differential in `p` at fixed family, so the
    anchor is subtracted and the argmax is read off the difference.
    """
    payload = json.loads(open(path, encoding="utf-8").read())
    cells = payload["cells"]
    assert cells, f"{path} has no cells"
    missing = [k for k in SCREEN_FIELDS.values() if k not in cells[0]]
    assert not missing, f"screen cells are missing {missing}"

    ladders = {}
    for cell in cells:
        family = cell["family"]
        row = {"family": family, "arm": cell["arm"],
               "p": float(cell["probe_fraction"]),
               "probes": clusters(float(cell["probe_fraction"])),
               "coarse_rows": clusters(float(cell["probe_fraction"]))
               * ROWS_PER_LEAF}
        for short, key in SCREEN_FIELDS.items():
            row[short] = cell.get(key)
        prior = ladders.setdefault(family, {}).get(row["p"])
        # `--widths` may leave several cells at one `p`; keep the best, which
        # is the choice a real selection would make at that probe fraction.
        if prior is None or (row["predicted_pct_absolute"] or -1e9) > (
                prior["predicted_pct_absolute"] or -1e9):
            ladders[family][row["p"]] = row

    unknown = set(ladders) - {SHIPPED_FAMILY, C1_FAMILY}
    assert not unknown, f"unrecognised screen family: {sorted(unknown)}"
    for family, by_p in ladders.items():
        shipped = family == SHIPPED_FAMILY
        anchor = SHIPPED_P if shipped else C1_P
        if anchor not in by_p:
            anchor = max(by_p)
        base = by_p[anchor]
        scale = BOARD_SCALE if shipped else 1.0
        for row in by_p.values():
            row["anchor_p"] = anchor
            row["board_scale_applied"] = scale
            row["d_gross_pct"] = row["gross_pct"] - base["gross_pct"]
            row["d_gross_pct_scaled"] = row["d_gross_pct"] * scale
            row["d_acc_loss_pp"] = 100.0 * (
                row["acc_loss_wg"] - base["acc_loss_wg"])
            row["d_net_pct"] = (row["predicted_pct_gating"]
                                - base["predicted_pct_gating"])
            # Rule 107 nets the scaled gross against the UNSCALED acceptance
            # loss: the loss is measured on the corpus, not predicted by the
            # byte model, so it must not inherit the byte model's error.
            row["d_net_pct_scaled"] = (row["d_gross_pct_scaled"]
                                       + row["d_net_pct"] - row["d_gross_pct"])
    return payload.get("samples"), ladders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-check", action="store_true",
                    help="import e133_screen and assert the restated "
                         "constants and byte model agree (needs MLX)")
    args = ap.parse_args()

    if args.self_check:
        import sys
        sys.path.insert(0, "research")
        import e133_screen as S
        assert S.LEAVES == LEAVES, (S.LEAVES, LEAVES)
        assert S.AFFINE2_ROW_BYTES == AFFINE2_ROW_BYTES
        assert S.AFFINE4_ROW_BYTES == AFFINE4_ROW_BYTES
        assert S.SHORTLIST == SHORTLIST
        assert S.MU_BYTES == MU_BYTES
        assert S.STEP_BYTES == STEP_BYTES
        assert S.HEAD_SHARE_LO == HEAD_SHARE_LO
        for p in GRID:
            assert S.shipped_stage_bytes(p) == shipped_stage_bytes(p), p
            assert S.arm_stage_bytes(C1_ROW_BYTES, C1_PROJ_BYTES, C1_SURVIVORS,
                                     p, "sketch") == c1_stage_bytes(p), p
            assert abs(S.price(1 << 20)["pct_head_share_7"]
                       - gross_pct(1 << 20)) < 1e-12
        print("self-check: e133_screen agrees on every constant and both "
              "byte models at every grid point")

    report = {"grid": GRID, "rows": [], "harness": "local",
              "official_or_ranked_score": False, "timing_valid": False,
              "preregistered_prediction": PREDICTION}
    for model in ("shipped", "c1"):
        rows = rows_of(model)
        report["rows"].extend(rows)
        report["marginal_bytes_per_0.01_p_%s" % model] = marginal(rows)
        report["marginal_pct_per_0.01_p_%s" % model] = gross_pct(marginal(rows))
        print(f"\n-- cost model {model} "
              f"({rows[0]['row_bytes']} B per probed row, "
              f"anchor p={rows[0]['anchor_p']}) --")
        print(f"{'p':>7}{'probes':>9}{'coarse rows':>13}{'stage MB':>11}"
              f"{'removed MB':>12}{'gross %':>10}")
        for row in rows:
            print(f"{row['p']:>7.3f}{row['probes']:>9}{row['coarse_rows']:>13}"
                  f"{row['stage_bytes'] / 1e6:>11.3f}"
                  f"{row['removed_vs_anchor_bytes'] / 1e6:>12.3f}"
                  f"{row['gross_pct_vs_anchor']:>10.4f}")
        print(f"marginal: {marginal(rows) / 1e6:.4f} MB and "
              f"{gross_pct(marginal(rows)):+.4f} % per 0.01 of p")

    report["null_floors"] = null_floors()
    print("\n-- pooled null floors, ranked % (E136 owed number) --")
    for name, floor in report["null_floors"].items():
        print(f"{name:34s} n={floor['n']:>6}  2s_null={floor['two_sigma_null_pct']:+.4f}"
              f"  2s_clustered={floor['two_sigma_clustered_pct']:+.4f}"
              f"  pure_gain={floor['floor_pure_gain_pct']:+.4f}")

    if args.screen:
        samples, ladders = merge_screen(args.screen)
        report["screen_samples"] = samples
        report["screen_file"] = args.screen
        report["ladders"] = {f: [by_p[p] for p in sorted(by_p)]
                             for f, by_p in ladders.items()}
        report["argmax"] = {}
        floor = report["null_floors"]["perfect_readout:corpus"]
        for family, rows in report["ladders"].items():
            best = max(rows, key=lambda r: r["d_net_pct"])
            report["argmax"][family] = {
                "p": best["p"], "anchor_p": best["anchor_p"],
                "d_net_pct": best["d_net_pct"],
                "d_net_pct_scaled": best["d_net_pct_scaled"],
                "beats_pooled_2sigma_clustered":
                    best["d_net_pct"] > floor["two_sigma_clustered_pct"],
                "beats_pooled_2sigma_clustered_scaled":
                    best["d_net_pct_scaled"] > floor["two_sigma_clustered_pct"],
            }
            print(f"\n-- measured ladder, family {family}, "
                  f"{rows[0]['bytes_per_row']} B per probed row, "
                  f"anchor p={rows[0]['anchor_p']}, n_gating={rows[0]['n_gating']} --")
            print(f"{'p':>7}{'probes':>8}{'rows':>8}{'recall':>10}"
                  f"{'probeHit':>10}{'mAbs':>11}{'accLoss':>11}"
                  f"{'dAcc pp':>10}{'dGross%':>10}{'dNet%':>9}"
                  f"{'t0':>4}{'t0b':>5}")
            for row in rows:
                print(f"{row['p']:>7.3f}{row['probes']:>8}{row['coarse_rows']:>8}"
                      f"{row['recall_wg']:>10.6f}{row['probe_hit_wg']:>10.6f}"
                      f"{row['m_absolute_wg']:>11.3e}"
                      f"{row['acc_loss_wg']:>11.3e}"
                      f"{row['d_acc_loss_pp']:>10.4f}{row['d_gross_pct']:>10.4f}"
                      f"{row['d_net_pct']:>9.4f}"
                      f"{str(row['passes_t0'])[0]:>4}{str(row['passes_t0b'])[0]:>5}")
            arg = report["argmax"][family]
            print(f"argmax p={arg['p']:g}  dNet={arg['d_net_pct']:+.4f} %  "
                  f"dNetScaled={arg['d_net_pct_scaled']:+.4f} %  "
                  f"vs pooled 2s clustered floor "
                  f"{floor['two_sigma_clustered_pct']:.4f} %: "
                  f"{'CLEARS' if arg['beats_pooled_2sigma_clustered'] else 'inside floor'}"
                  f" / scaled "
                  f"{'CLEARS' if arg['beats_pooled_2sigma_clustered_scaled'] else 'inside floor'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
