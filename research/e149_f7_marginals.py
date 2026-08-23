#!/usr/bin/env python3
"""E149 F7: bound the high-width marginals instead of only ranking two curves.

F7 asks for `e149_width8_admissible_ranked` as a BOUNDED INTERVAL and for
`e149_rungD_marginal_7_to_8_us` WITH AN ERROR BAR. The arm D splice only ranked
two named curves against each other. This scans the marginal continuously.

Method. Take the low-width block (widths 1..6, or 1..7 for the 7->8 scan) from
the configuration the splice already selected for that mass model, hold it
fixed, and step the marginal of interest over a grid. At each grid point refit
the single free scalar level exactly as `e149_rungD_robust.level_fit` does, so
no new modelling assumption enters. That gives chi2 as a function of the
marginal.

Error bars under a MISSPECIFIED model. Both hypotheses are rejected outright:
rms 4.17 % and 7.92 % against an empirical reproducibility floor of 1.1365 %.
A raw delta-chi2 = 1 interval on such a fit is meaningless and far too narrow.
So the per-row sigmas are inflated by sqrt(chi2_min / dof) before the interval
is read, which is the standard convention for a fit whose chi2/dof exceeds one.
The interval is then honest about the scatter the data actually show, and it is
reported as CONDITIONAL on the low-width block being right.

Admissibility. `research/e145_r7.py:68-98`, reproduced in
`e149_rungD_curve.admissible_set`: width w is admissible exactly when C_w / w
is a new strict running minimum. With widths 1..7 fixed that becomes a single
inequality on C_8, and therefore a single threshold on the 7->8 marginal.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "e149-rungD.json"
ROBUST = HERE / "e149-rungD-robust.json"
OUT = HERE / "e149-f7-marginals.json"

# The two named reference curves, for the record.
H_NULL = {1: 31173.2, 2: 34619.3, 3: 38065.4, 4: 41511.4, 5: 44957.5,
          6: 61198.8, 7: 62824.9, 8: 70315.4}
H_ALT = {2: 33482.9, 3: 35748.7, 4: 39549.6, 5: 45308.7, 6: 59158.9,
         7: 71454.6, 8: 73329.7}

GRID_LO, GRID_HI, GRID_N = -4000.0, 26000.0, 3001


def level_fit(mass, y, sigma, base):
    pred = mass @ base
    w = 1.0 / sigma ** 2
    scale = float(np.sum(w * pred * y) / np.sum(w * pred * pred))
    resid = scale * pred - y
    return (float(np.sum((resid / sigma) ** 2)),
            float(np.sqrt(np.mean((resid / y) ** 2)) * 100.0),
            scale)


def curve_vector(block: dict, overrides: dict) -> np.ndarray:
    c = {int(k): float(v) for k, v in block.items()}
    c.update(overrides)
    return np.array([c[w] for w in range(1, 9)])


def admissible_widths(vec: np.ndarray) -> list[int]:
    out, best = [], math.inf
    for i, cost in enumerate(vec):
        width = i + 1
        per_token = cost / width
        if per_token < best:
            out.append(width)
            best = per_token
    return out


def width8_threshold(vec: np.ndarray) -> float:
    """Largest C_8 for which width 8 is still a new strict running minimum."""
    best = min(vec[i] / (i + 1) for i in range(7))
    return best * 8.0


def profile(mass, y, sigma, block, fixed_widths, moving_width, anchor_width,
            tail_offsets):
    """chi2 against the marginal `anchor_width -> moving_width`.

    `tail_offsets` pins every width above `moving_width` at a fixed distance
    from it, so the scan moves one marginal and drags the tail rigidly.
    """
    grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
    chi2s, rmss = [], []
    anchor = float(block[str(anchor_width)])
    for delta in grid:
        over = {moving_width: anchor + delta}
        for w, off in tail_offsets.items():
            over[w] = anchor + delta + off
        vec = curve_vector(block, over)
        c, r, _ = level_fit(mass, y, sigma, vec)
        chi2s.append(c)
        rmss.append(r)
    chi2s = np.array(chi2s)
    rmss = np.array(rmss)
    j = int(np.argmin(chi2s))
    dof = max(1, len(y) - 2)          # one level, one marginal
    inflation = math.sqrt(chi2s[j] / dof)
    scaled = chi2s / inflation ** 2

    def band(threshold):
        ok = np.where(scaled <= scaled[j] + threshold)[0]
        return float(grid[ok[0]]), float(grid[ok[-1]])

    lo1, hi1 = band(1.0)
    lo2, hi2 = band(4.0)
    # A per-round cost curve cannot fall as the verify width rises: the wider
    # round evaluates a superset of the rows. Report the interval clipped to
    # that physical bound beside the free one, and say when the bound binds.
    monotone = {
        "best_marginal_us": max(0.0, float(grid[j])),
        "ci68_us": [max(0.0, lo1), max(0.0, hi1)],
        "ci95_us": [max(0.0, lo2), max(0.0, hi2)],
        "bound_binds": bool(lo2 < 0.0),
    }
    return {
        "best_marginal_us": float(grid[j]),
        "best_chi2": float(chi2s[j]),
        "best_rms_pct": float(rmss[j]),
        "dof": dof,
        "sigma_inflation": inflation,
        "ci68_us": [lo1, hi1],
        "ci95_us": [lo2, hi2],
        "grid_us": [GRID_LO, GRID_HI],
        "hit_grid_edge": bool(lo2 <= GRID_LO + 1e-6 or hi2 >= GRID_HI - 1e-6),
        "monotone_constrained": monotone,
        "_grid": grid,
        "_scaled": scaled,
        "_ref_chi2": scaled[j],
    }


def main() -> int:
    report = json.loads(SRC.read_text())
    robust = json.loads(ROBUST.read_text())

    out = {
        "harness": "ranked",
        "purpose": "F7: bounded interval on the high-width marginals",
        "reference_marginals_us": {
            "h_null_replayed_5_to_6": H_NULL[6] - H_NULL[5],
            "h_null_replayed_6_to_7": H_NULL[7] - H_NULL[6],
            "h_null_replayed_7_to_8": H_NULL[8] - H_NULL[7],
            "h_alt_measured_over_k_5_to_6": H_ALT[6] - H_ALT[5],
            "h_alt_measured_over_k_6_to_7": H_ALT[7] - H_ALT[6],
            "h_alt_measured_over_k_7_to_8": H_ALT[8] - H_ALT[7],
        },
        "f7_label_correction": (
            "F7 asks whether the 7->8 marginal is nearer 1,876 us (labelled "
            "replayed) or 12,295 us (labelled measured). Both labels belong to "
            "the MEASURED curve and only one of them is a 7->8 value. "
            "H-alt (E145 measured local / k) has 6->7 = 12,295.7 and "
            "7->8 = 1,875.1. H-null (replayed ranked) has 6->7 = 1,626.1 and "
            "7->8 = 7,490.5. The two curves SWAP the two marginals, which is "
            "F6's 7.56x and 0.25x. The well-posed 7->8 question is therefore "
            "7,490.5 replayed against 1,875.1 measured."
        ),
        "models": {},
    }

    for model in ("tilt", "replay"):
        entry = report["results"][model]
        mass = np.array(entry["mass"])
        y = np.array(entry["y_us_per_round"])
        sigma = np.array(entry["sigma_us_per_round"])

        winner = robust["models"][model]["high_width_block_best"]
        block = robust["models"][model]["high_width_block"][winner]["curve_us"]

        # 7->8: widths 1..7 pinned at the winning block, nothing above 8.
        p78 = profile(mass, y, sigma, block, None, 8, 7, {})
        # 6->7: widths 1..6 pinned, and 8 dragged rigidly behind 7 at the
        # winning block's own 7->8 distance.
        tail = float(block["8"]) - float(block["7"])
        p67 = profile(mass, y, sigma, block, None, 7, 6, {8: tail})
        # 5->6: widths 1..5 pinned, 7 and 8 dragged rigidly behind 6.
        off7 = float(block["7"]) - float(block["6"])
        off8 = float(block["8"]) - float(block["6"])
        p56 = profile(mass, y, sigma, block, None, 6, 5, {7: off7, 8: off8})

        base_vec = curve_vector(block, {})
        thr = width8_threshold(base_vec)
        thr_marginal = thr - float(block["7"])

        # Where in the 7->8 profile does width 8 stop being admissible?
        grid = p78["_grid"]
        scaled = p78["_scaled"]
        ref = p78["_ref_chi2"]
        adm = np.array([
            8 in admissible_widths(curve_vector(block, {8: float(block["7"]) + d}))
            for d in grid])
        in95 = scaled <= ref + 4.0
        frac_admissible_in_ci95 = float(np.mean(adm[in95])) if in95.any() else 0.0

        for p in (p78, p67, p56):
            for k in ("_grid", "_scaled", "_ref_chi2"):
                p.pop(k)

        out["models"][model] = {
            "splice_winner_block": winner,
            "low_width_block_us": {k: float(v) for k, v in block.items()},
            "e149_rungD_marginal_7_to_8_us": p78,
            "e149_rungD_marginal_6_to_7_us": p67,
            "e149_rungD_marginal_5_to_6_us": p56,
            "width8_admissibility": {
                "rule": "C_8 / 8 must be a new strict running minimum",
                "max_admissible_C8_us": float(thr),
                "max_admissible_7_to_8_marginal_us": float(thr_marginal),
                "binding_competitor_width": int(
                    1 + int(np.argmin([base_vec[i] / (i + 1) for i in range(7)]))),
                "replayed_7_to_8_admissible":
                    bool((H_NULL[8] - H_NULL[7]) < thr_marginal),
                "measured_7_to_8_admissible":
                    bool((H_ALT[8] - H_ALT[7]) < thr_marginal),
                "fraction_of_ci95_that_is_admissible": frac_admissible_in_ci95,
                "e149_width8_admissible_ranked_interval":
                    "admissible" if frac_admissible_in_ci95 >= 0.95
                    else ("inadmissible" if frac_admissible_in_ci95 <= 0.05
                          else "unresolved"),
            },
        }

    a = out["models"]["tilt"]["e149_rungD_marginal_7_to_8_us"]
    b = out["models"]["replay"]["e149_rungD_marginal_7_to_8_us"]
    out["e149_rungD_marginal_7_to_8_us"] = a["best_marginal_us"]
    out["e149_rungD_marginal_7_to_8_us_ci95"] = a["ci95_us"]
    out["e149_rungD_marginal_7_to_8_us_replay_model"] = b["best_marginal_us"]
    out["e149_rungD_marginal_7_to_8_us_replay_model_ci95"] = b["ci95_us"]
    out["e149_width8_admissible_ranked"] = (
        out["models"]["tilt"]["width8_admissibility"][
            "e149_width8_admissible_ranked_interval"])
    out["caveat"] = (
        "Conditional on the low-width block selected by the arm D splice. The "
        "underlying fit is misspecified (rms 4.17 % against a 1.1365 % "
        "empirical floor), so sigmas are inflated to chi2/dof = 1 before the "
        "interval is read. This is a bounded interval, not a calibrated "
        "frequentist one, and it does not repair the stop rule that fired on "
        "the free-eight inversion."
    )

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}\n")
    print("reference marginals, us/round")
    for k, v in out["reference_marginals_us"].items():
        print(f"  {k:38s} {v:10.1f}")
    print()
    for model, rec in out["models"].items():
        print(f"=== mass model {model}   block {rec['splice_winner_block']} ===")
        for name in ("5_to_6", "6_to_7", "7_to_8"):
            p = rec[f"e149_rungD_marginal_{name}_us"]
            m = p["monotone_constrained"]
            edge = "  MONOTONE BOUND BINDS" if m["bound_binds"] else ""
            print(f"  {name}  best {m['best_marginal_us']:9.1f}  "
                  f"68 % [{m['ci68_us'][0]:8.1f}, {m['ci68_us'][1]:8.1f}]  "
                  f"95 % [{m['ci95_us'][0]:8.1f}, {m['ci95_us'][1]:8.1f}]  "
                  f"rms {p['best_rms_pct']:.3f} %  infl {p['sigma_inflation']:.1f}{edge}")
        w = rec["width8_admissibility"]
        print(f"  width 8 admissible while 7->8 < {w['max_admissible_7_to_8_marginal_us']:.1f} us "
              f"(binding competitor width {w['binding_competitor_width']})")
        print(f"    replayed 7->8 admissible: {w['replayed_7_to_8_admissible']}   "
              f"measured 7->8 admissible: {w['measured_7_to_8_admissible']}")
        print(f"    fraction of the 95 % interval that keeps width 8 admissible: "
              f"{w['fraction_of_ci95_that_is_admissible']:.3f}  -> "
              f"{w['e149_width8_admissible_ranked_interval']}")
        print()
    print(f"e149_width8_admissible_ranked = {out['e149_width8_admissible_ranked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
