"""E134 item 2 -- invert one ranked receipt pair for a per-width cost change.

`harness=ranked`. Zero GPU. The only input is the advisor's F7 section 3 table,
which reconstructs microseconds per round for the promoted pair

    623e77af (one-pass QMV table {6:6, 7:7})  against  its own pre-arm base

as `mtp_seconds_per_token_mean * 512 * 1e6 / R`. The schedule is bit-exact
across the pair, so `R` is pinned and the mean round-time change equals the
candidate seconds-per-token change exactly.

This is a measurement, not a prediction. Advisor error 126 was to price the
boundary family on a curve predicted from an instruction census. The receipt
refutes that curve, so the price has to be re-selected from the measured one.

Model, over the eight ranked prompts:

    delta_us(p) = uniform * f(p) + d6 * mass(p, 6) + d7 * mass(p, 7)

`mass(p, w)` is the share of the prompt's rounds whose verify dispatches `w`
rows, taken from the shipped arm of the same replayer that produced every
E134 number. Drama, travel and plutarch carry almost no mass at widths 6 and 7
and therefore identify the uniform term; the five weighted prompts identify the
two widths. Three forms of `f` are fitted, because the receipt cannot say on
its own whether the templating tax is per round, per drafting round, or
proportional to round time.

Usage:
  python3 e134_item2_refit.py --json e134-artifacts/item2-measured-curve.json
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e128_price import MAX_DEPTH, RANKED_PROMPTS  # noqa: E402
from e134_rung2 import build_legs, prompt_panel  # noqa: E402
from e134_rung3 import OUR_CURVE  # noqa: E402

# Advisor F7 section 3, verbatim. `delta_us` is after minus before, so a
# negative value is a faster round. `R` is the pinned round count.
RECEIPT = "623e77af"
RECEIPT_NOTE = ("F7 section 3, promoted 623e77af with the one-pass QMV table "
                "{6:6, 7:7} live, against its own pre-arm base; round counts "
                "pinned by the F119 R vector")
MEASURED = {
    "plutarch": {"mbar": 1.154, "R": 487, "before": 31813.9, "after": 31795.0,
                 "delta_us": -18.9, "delta_pct": -0.0593, "f83": 0.0000},
    "drama":    {"mbar": 3.298, "R": 252, "before": 38708.9, "after": 38842.8,
                 "delta_us": +133.8, "delta_pct": +0.3457, "f83": 0.0000},
    "travel":   {"mbar": 3.656, "R": 212, "before": 40103.7, "after": 40250.5,
                 "delta_us": +146.8, "delta_pct": +0.3660, "f83": 0.0000},
    "beagle":   {"mbar": 5.382, "R": 110, "before": 52531.6, "after": 52453.2,
                 "delta_us": -78.4, "delta_pct": -0.1492, "f83": 0.4862},
    "republic": {"mbar": 5.989, "R": 93, "before": 56266.4, "after": 56184.3,
                 "delta_us": -82.1, "delta_pct": -0.1459, "f83": 0.0100},
    "essays":   {"mbar": 6.087, "R": 92, "before": 57570.0, "after": 57383.1,
                 "delta_us": -186.9, "delta_pct": -0.3246, "f83": 0.1598},
    "medicine": {"mbar": 6.256, "R": 90, "before": 58113.2, "after": 57931.4,
                 "delta_us": -181.8, "delta_pct": -0.3129, "f83": 0.2508},
    "botany":   {"mbar": 7.148, "R": 81, "before": 63732.9, "after": 63791.8,
                 "delta_us": +58.9, "delta_pct": +0.0924, "f83": 0.0124},
}


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting. Square systems only."""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular normal equations at column %d" % col)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def least_squares(design, observed, weights):
    """Weighted normal equations, returned with residuals and fit quality."""
    k = len(design[0])
    ata = [[sum(weights[i] * design[i][a] * design[i][b]
                for i in range(len(observed)))
            for b in range(k)] for a in range(k)]
    atb = [sum(weights[i] * design[i][a] * observed[i]
               for i in range(len(observed))) for a in range(k)]
    beta = solve(ata, atb)
    fitted = [sum(beta[a] * row[a] for a in range(k)) for row in design]
    residual = [observed[i] - fitted[i] for i in range(len(observed))]
    mean = sum(observed) / len(observed)
    ss_res = sum(weights[i] * residual[i] ** 2 for i in range(len(observed)))
    ss_tot = sum(weights[i] * (observed[i] - mean) ** 2
                 for i in range(len(observed)))
    return {
        "beta": beta,
        "fitted": fitted,
        "residual": residual,
        "rms_residual_us": (sum(r * r for r in residual)
                            / len(residual)) ** 0.5,
        "max_abs_residual_us": max(abs(r) for r in residual),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def width_masses(args):
    """Share of rounds at each verify width, per ranked prompt, shipped arm.

    Averaged over the same seeds rung 3 uses, so the masses describe the same
    replayed population the price arms are scored on.
    """
    legs, gate = build_legs(args.accept, args.runs)
    totals = {}
    for offset in range(args.seeds):
        panel = prompt_panel(legs, args.windows, args.fit_windows,
                             args.seed + offset)
        for prompt, entry in panel.items():
            counts = entry["ship"]["depth_counts"]
            bucket = totals.setdefault(prompt, [0.0] * len(counts))
            for depth, count in enumerate(counts):
                bucket[depth] += count
    masses = {}
    for prompt, counts in totals.items():
        rounds = sum(counts)
        # A round that drafts `depth` tokens verifies `depth + 1` rows.
        masses[prompt] = {
            "rounds": rounds,
            "by_width": {depth + 1: counts[depth] / rounds
                         for depth in range(len(counts))
                         if counts[depth]},
            "drafting_share": sum(counts[1:]) / rounds,
        }
    return masses, gate


def fit_all(masses):
    prompts = [p for p in MEASURED if p in masses]
    observed = [MEASURED[p]["delta_us"] for p in prompts]
    m6 = [masses[p]["by_width"].get(6, 0.0) for p in prompts]
    m7 = [masses[p]["by_width"].get(7, 0.0) for p in prompts]
    forms = {
        "per_round": [1.0 for _ in prompts],
        "per_drafting_round": [masses[p]["drafting_share"] for p in prompts],
        "proportional": [MEASURED[p]["before"] / 1000.0 for p in prompts],
    }
    weights = [1.0] * len(prompts)
    out = {}
    for name, column in forms.items():
        design = [[column[i], m6[i], m7[i]] for i in range(len(prompts))]
        fit = least_squares(design, observed, weights)
        fit["prompts"] = prompts
        fit["form"] = name
        fit["column"] = column
        out[name] = fit
    return prompts, observed, m6, m7, out


def base_round_us(rows):
    intercept, slope = (OUR_CURVE["lo"] if rows < OUR_CURVE["breakpoint"]
                        else OUR_CURVE["hi"])
    return intercept + slope * rows


def measured_curve(fit, form):
    """OUR_CURVE plus the inverted corrections, as an installable curve."""
    uniform, d6, d7 = fit["beta"]
    curve = {
        "name": "e134_item2_measured_%s" % form,
        "breakpoint": OUR_CURVE["breakpoint"],
        "lo": list(OUR_CURVE["lo"]),
        "hi": list(OUR_CURVE["hi"]),
        "per_width": {},
        "provenance": {
            "receipt": RECEIPT,
            "note": RECEIPT_NOTE,
            "form": form,
            "uniform_us": uniform,
            "delta_round_us_6": d6,
            "delta_round_us_7": d7,
            "rms_residual_us": fit["rms_residual_us"],
            "base_curve": "e134_rung3.OUR_CURVE slopeonly_b6",
        },
    }
    if form == "per_round":
        curve["uniform"] = uniform
    elif form == "per_drafting_round":
        # A width-1 round drafts nothing, so the tax lands on widths >= 2.
        for width in range(2, MAX_DEPTH + 2):
            curve["per_width"][width] = uniform
    else:
        # `uniform` is microseconds per 1000 microseconds of pre-arm round
        # time, so the additive equivalent is width dependent.
        for width in range(1, MAX_DEPTH + 2):
            curve["per_width"][width] = uniform * base_round_us(width) / 1000.0
    curve["per_width"][6] = curve["per_width"].get(6, 0.0) + d6
    curve["per_width"][7] = curve["per_width"].get(7, 0.0) + d7
    return curve


def report_curve(curve, label):
    saved = e128_price.CURVE
    e128_price.CURVE = curve
    rows = list(range(1, MAX_DEPTH + 2))
    values = [e128_price.ranked_round_us(m) for m in rows]
    print("\n## %s" % label)
    print("rows      " + " ".join("%9d" % m for m in rows))
    print("round us  " + " ".join("%9.1f" % v for v in values))
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    shallow = steps[0]
    print("step us   " + " " * 9 + " ".join("%9.1f" % s for s in steps))
    print("ratio     " + " " * 9 + " ".join(
        "%9.3f" % (s / shallow if shallow else float("nan")) for s in steps))
    boundary = max(range(len(steps)), key=lambda i: steps[i])
    print("largest step is boundary %d (M %d -> %d), ratio %.3f"
          % (boundary, boundary + 1, boundary + 2,
             steps[boundary] / shallow if shallow else float("nan")))
    e128_price.CURVE = saved
    return {"rows": rows, "round_us": values, "steps": steps,
            "ratios": [s / shallow if shallow else None for s in steps],
            "argmax_boundary": boundary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e134-artifacts/item2-measured-curve.json")
    args = ap.parse_args()

    masses, gate = width_masses(args)
    print("## attachment gate")
    print("legs %d ; rounds attached %d ; accept mismatches %d ; "
          "margin mismatches %d ; unmatched %d" % (
              gate["legs"], gate["attached"], gate["accept_mismatch"],
              gate["margin_mismatch"], gate["unmatched"]))

    print("\n## replayed shipped width masses, %d seeds x %d windows"
          % (args.seeds, args.windows))
    print("%-10s %9s %8s %8s %8s %8s %8s %8s" % (
        "prompt", "rounds", "draft%", "m(5)", "m(6)", "m(7)", "m(8)", "m(9)"))
    for prompt in MEASURED:
        entry = masses.get(prompt)
        if entry is None:
            print("%-10s   MISSING" % prompt)
            continue
        by = entry["by_width"]
        print("%-10s %9.0f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            prompt, entry["rounds"], entry["drafting_share"],
            by.get(5, 0.0), by.get(6, 0.0), by.get(7, 0.0),
            by.get(8, 0.0), by.get(9, 0.0)))

    prompts, observed, m6, m7, fits = fit_all(masses)
    print("\n## inversion, three forms of the uniform templating term")
    print("%-20s %12s %12s %12s %10s %10s" % (
        "form", "uniform", "d round(6)", "d round(7)", "rms res", "r2"))
    for name, fit in fits.items():
        print("%-20s %12.2f %12.2f %12.2f %10.2f %10.4f" % (
            name, fit["beta"][0], fit["beta"][1], fit["beta"][2],
            fit["rms_residual_us"], fit["r2"]))

    best = min(fits.values(), key=lambda f: f["rms_residual_us"])
    print("\n## residuals, best form %s" % best["form"])
    print("%-10s %8s %8s %10s %10s %10s" % (
        "prompt", "m(6)", "m(7)", "observed", "fitted", "residual"))
    for index, prompt in enumerate(prompts):
        print("%-10s %8.4f %8.4f %+10.1f %+10.1f %+10.1f" % (
            prompt, m6[index], m7[index], observed[index],
            best["fitted"][index], best["residual"][index]))

    results = {
        "receipt": RECEIPT,
        "receipt_note": RECEIPT_NOTE,
        "harness": "ranked",
        "measured_table": MEASURED,
        "width_masses": masses,
        "attachment_gate": gate,
        "fits": {name: {k: v for k, v in fit.items() if k != "column"}
                 for name, fit in fits.items()},
        "best_form": best["form"],
        "curves": {},
    }
    for name, fit in fits.items():
        curve = measured_curve(fit, name)
        shape = report_curve(curve, "measured curve, %s form" % name)
        results["curves"][name] = {"curve": curve, "shape": shape}
    pre = report_curve(dict(OUR_CURVE, name="ours_pre_arm"),
                       "pre-arm curve, for reference")
    results["curves"]["pre_arm"] = {"curve": dict(OUR_CURVE), "shape": pre}

    path = args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
