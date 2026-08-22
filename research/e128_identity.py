#!/usr/bin/env python3
"""E128 rung 0 revised -- close the round-count identity and pin ranked R.

Advisor finding F1: the ranked round count `R` is not published, so
`accept_rate = (512 - R) / (R * eff)` has two unknowns and one equation, and
every quantity fitted through it inherits the assumption. This module answers
that with local measurement instead of assumption. Three sections:

  1. IDENTITY. Every local leg reports `round_count`, `accepted_draft_total`,
     `emitted_token_total`, `declared_rows_total` and `effective_mean_draft_len`
     directly, so the accounting can be checked rather than assumed.

  2. CALIBRATION. At termination the shipped walk holds
     `expected = sum_{k<d} prod_{j<=k} p_j`, which is identically its own
     estimate of the accepted-draft count for that round. Comparing it with
     the round's realised accepted count is a direct calibration test of the
     estimator, and a calibrated estimator pins `R` through the identity.

  3. MANIFOLD. `eff` and `E[accepted]` are not free of each other. Both are
     produced by the same shipped scheduler from the same one-parameter family
     of prompt difficulty, so the locally measured `(eff, E[accepted])` curve
     predicts `E[accepted]` from a published `eff`, and therefore predicts `R`.
     Leave-one-out residuals give the band. This is the measurement that
     replaces the assumed round-count vector.

`harness=local` for sections 1 and 2. Section 3 predicts a ranked quantity from
local measurement and is labelled `harness=ranked-prediction`; it is a model
output, never a timing measurement.

  usage: research/e128_identity.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from e128_replay import read_meta, read_rounds

WINDOW = 512

# `research/rankedcurve.py`'s hardcoded round-count vector, and the published
# `effective_mean_draft_len` it goes with. The assumed column is exactly what
# F1 says is unmeasured; this file tests it.
RANKED = {
    "beagle":   {"eff": 4.3818, "assumed_R": 110, "floor_R": 96, "nondraft": 0},
    "medicine": {"eff": 5.2556, "assumed_R": 90, "floor_R": 82, "nondraft": 0},
    "essays":   {"eff": 5.0870, "assumed_R": 92, "floor_R": 85, "nondraft": 0},
    "botany":   {"eff": 6.1481, "assumed_R": 81, "floor_R": 72, "nondraft": 0},
    "republic": {"eff": 4.9892, "assumed_R": 93, "floor_R": 86, "nondraft": 0},
    "drama":    {"eff": 2.2976, "assumed_R": 252, "floor_R": 156, "nondraft": 0},
    "travel":   {"eff": 2.6557, "assumed_R": 212, "floor_R": 141, "nondraft": 0},
    "plutarch": {"eff": 0.1540, "assumed_R": 487, "floor_R": 458,
                 "nondraft": 449},
}

MAX_DEPTH_CAP = 7  # segmentedVerifyDepthCap


def non_drafting_bound(eff: float, nondraft: int) -> list | None:
    """Bound `R` from `non_drafting_round_count`, with no manifold needed.

    `eff` is the mean declared draft count over ALL rounds, so the total draft
    count is `eff * R`. A drafting round declares between one and seven drafts,
    and there are `R - nondraft` of them, which brackets `R` from both sides.
    That `eff = 0.154 < 1` while 449 rounds declared no drafts also proves the
    published `eff` is averaged over all rounds and not over drafting rounds.
    """
    if nondraft <= 0:
        return None
    low = nondraft * MAX_DEPTH_CAP / (MAX_DEPTH_CAP - eff)
    high = nondraft / (1.0 - eff) if eff < 1.0 else float(WINDOW)
    return [math.ceil(low), min(WINDOW, math.floor(high))]


def leg_identity(run_dir: Path) -> dict | None:
    report = run_dir / "report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    meta = read_meta(run_dir)
    rounds = data["round_count"]
    accepted = data["accepted_draft_total"]
    rejected = data["rejected_draft_total"]
    emitted = data["emitted_token_total"]
    eff = data["effective_mean_draft_len"]
    rate = data["accepted_draft_rate"]
    drafted = accepted + rejected
    return {
        "leg": run_dir.name,
        "forced_depth": meta.get("forced_depth", "none"),
        "base_sha": meta.get("base_sha"),
        "rounds": rounds,
        "accepted": accepted,
        "rejected": rejected,
        "drafted": drafted,
        "emitted": emitted,
        "declared_rows": data["declared_rows_total"],
        "non_drafting_rounds": data["non_drafting_round_count"],
        # R + A must equal the window, plus at most one token that the parent
        # asked for and then threw away: its tail rule floors the offer at one
        # draft even on the last token, so a final accepted tail draft is
        # counted as accepted and never emitted.
        "rounds_plus_accepted": rounds + accepted,
        "window_residual": rounds + accepted - WINDOW,
        # eff is the mean over ALL rounds of the declared draft count.
        "eff_reported": eff,
        "eff_recomputed": drafted / rounds,
        "eff_residual": drafted / rounds - eff,
        "accept_rate_reported": rate,
        "accept_rate_recomputed": accepted / drafted if drafted else 0.0,
        "accept_rate_residual": (accepted / drafted if drafted else 0.0) - rate,
        "declared_rows_residual": data["declared_rows_total"] - rounds - drafted,
        # The identity the advisor needs: R implied by the accounting alone.
        "R_implied": WINDOW / (1.0 + accepted / rounds),
        "R_counted": rounds,
        "mean_accepted": accepted / rounds,
    }


def parse_sched(text: str) -> list[dict]:
    """`sched=depth:p/reach/threshold;` as recorded by `snapshotScheduleSignal`."""
    steps = []
    for chunk in text.split(";"):
        if not chunk:
            continue
        index, _, rest = chunk.partition(":")
        parts = rest.split("/")
        if len(parts) != 3:
            continue
        steps.append({"depth": int(index), "p": float(parts[0]),
                      "reach": float(parts[1]),
                      "threshold": float(parts[2])})
    return steps


def leg_calibration(run_dir: Path) -> dict | None:
    """`expected` at termination against the round's realised accepted count.

    `expected` is read from the recorded walk rather than recomputed, so this
    measures the shipped estimator and not a paraphrase of it.

    A forced leg still records the shipped walk, and its realised accepted
    count is uncensored, so the estimator can be scored against
    `min(capability, shipped depth)` on rounds the shipped policy would have
    cut short. That is the same comparison as on a shipped leg, with the
    selection removed.
    """
    rounds = read_rounds(run_dir)
    pairs = []
    for record in rounds:
        steps = parse_sched(record["sched"])
        if not steps:
            continue
        depth = 0
        expected = 0.0
        for step in steps:
            if not step["reach"] > step["threshold"]:
                break
            expected += step["reach"]
            depth += 1
        pairs.append((expected, float(min(record["accepted"], depth)), depth))
    if not pairs:
        return None
    exp = [p[0] for p in pairs]
    act = [float(p[1]) for p in pairs]
    n = len(pairs)
    mean_e, mean_a = sum(exp) / n, sum(act) / n
    cov = sum((e - mean_e) * (a - mean_a) for e, a in zip(exp, act))
    var = sum((e - mean_e) ** 2 for e in exp)
    bins = {}
    for e, a, _ in pairs:
        key = min(int(e), 7)
        bins.setdefault(key, []).append(a)
    return {
        "leg": run_dir.name,
        "rounds": n,
        "mean_expected": mean_e,
        "mean_accepted": mean_a,
        "bias": mean_e - mean_a,
        "bias_pct": 100.0 * (mean_e - mean_a) / mean_a if mean_a else float("nan"),
        "slope": cov / var if var > 0 else float("nan"),
        "pearson_r": (
            cov / math.sqrt(var * sum((a - mean_a) ** 2 for a in act))
            if var > 0 and sum((a - mean_a) ** 2 for a in act) > 0
            else float("nan")),
        "bins": {str(k): {"rounds": len(v), "mean_accepted": sum(v) / len(v)}
                 for k, v in sorted(bins.items())},
    }


def interpolate(points: list[tuple], x: float) -> float:
    """Piecewise linear in `eff`, with flat extrapolation outside the range."""
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return 0.5 * (y0 + y1)
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def manifold(identities: list[dict]) -> dict:
    """Predict ranked `R` from published `eff` through the local scheduler curve.

    The shipped scheduler maps a prompt's acceptance profile to BOTH its mean
    draft depth and its mean accepted count, so the two are tied together by
    the code rather than free. Every point here comes from a shipped-policy
    512-token leg driven by the same estimator that produced the ranked legs.
    """
    points = sorted(
        (row["eff_reported"], row["mean_accepted"]) for row in identities)
    lo_eff, hi_eff = points[0][0], points[-1][0]
    order = {row["eff_reported"]: row["leg"] for row in identities}
    # Leave-one-out: predict each leg from the other legs only, so the band is
    # an honest out-of-sample spread and not a fit residual. Dropping an
    # endpoint forces the remaining curve to extrapolate, which is not the
    # question being asked, so the band comes from the interior legs.
    loo = []
    for index, (x, y) in enumerate(points):
        rest = points[:index] + points[index + 1:]
        if len(rest) < 2:
            continue
        predicted = interpolate(rest, x)
        loo.append({
            "leg": order.get(x, "?"), "eff": x,
            "interior": 0 < index < len(points) - 1,
            "measured_mean_accepted": y, "loo_predicted": predicted,
            "residual": predicted - y,
            "R_loo": WINDOW / (1.0 + predicted),
        })
    interior = [row["residual"] for row in loo if row["interior"]]
    residuals = [row["residual"] for row in loo]
    spread = statistics.pstdev(interior) if len(interior) > 1 else 0.0
    worst = max(abs(r) for r in interior) if interior else 0.0

    predictions = {}
    for prompt, spec in RANKED.items():
        eff = spec["eff"]
        centre = interpolate(points, eff)
        inside = lo_eff <= eff <= hi_eff
        band = [max(centre - worst, 0.0), centre + worst]
        if not inside:
            # The interpolator clamps outside the measured range, so its answer
            # there is an artefact. Report nothing rather than a clamp.
            centre = float("nan")
            band = [float("nan"), float("nan")]
        predictions[prompt] = {
            "eff": eff,
            "inside_local_eff_range": inside,
            "R_bound_from_non_drafting_rounds":
                non_drafting_bound(eff, spec["nondraft"]),
            "predicted_mean_accepted": centre,
            "predicted_mean_accepted_band": band,
            "R_predicted": WINDOW / (1.0 + centre),
            "R_band": [WINDOW / (1.0 + band[1]), WINDOW / (1.0 + band[0])],
            "R_assumed": spec["assumed_R"],
            "R_floor": spec["floor_R"],
            "R_ratio_predicted_over_assumed":
                (WINDOW / (1.0 + centre)) / spec["assumed_R"],
            "accept_rate_predicted": centre / eff if eff > 0 else float("nan"),
            "accept_rate_at_assumed_R":
                (WINDOW - spec["assumed_R"]) / (spec["assumed_R"] * eff)
                if eff > 0 else float("nan"),
        }
    return {
        "harness": "ranked-prediction",
        "local_points": [{"leg": row["leg"], "eff": row["eff_reported"],
                          "mean_accepted": row["mean_accepted"],
                          "R": row["rounds"]} for row in identities],
        "leave_one_out": loo,
        "loo_residual_pstdev_interior": spread,
        "loo_residual_max_abs_interior": worst,
        "loo_residual_pstdev_all": (
            statistics.pstdev(residuals) if len(residuals) > 1 else 0.0),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    identities, calibrations = [], []
    for run_dir in args.run_dirs:
        row = leg_identity(run_dir)
        if row is None:
            print("skip %s: no report.json" % run_dir, file=sys.stderr)
            continue
        identities.append(row)
        cal = leg_calibration(run_dir)
        if cal:
            calibrations.append(cal)

    print("=== 1. identity: R + A = window, and eff / rate recomputed ===")
    print("%-18s %5s %6s %6s %8s %9s %11s %11s" % (
        "leg", "R", "A", "R+A", "emitted", "rows err", "eff err", "rate err"))
    for row in identities:
        print("%-18s %5d %6d %6d %8d %9d %11.2e %11.2e" % (
            row["leg"], row["rounds"], row["accepted"],
            row["rounds_plus_accepted"], row["emitted"],
            row["declared_rows_residual"], row["eff_residual"],
            row["accept_rate_residual"]))
    residual = {row["window_residual"] for row in identities}
    print("window residual R + A - %d over %d legs: %s" % (
        WINDOW, len(identities), sorted(residual)))

    shipped = [row for row in identities if row["forced_depth"] == "none"]
    if calibrations:
        print("\n=== 2. calibration: the walk's own `expected` vs realised ===")
        print("%-18s %6s %10s %10s %8s %8s %7s %7s" % (
            "leg", "rounds", "expected", "accepted", "bias", "bias %",
            "slope", "r"))
        for row in calibrations:
            print("%-18s %6d %10.4f %10.4f %8.4f %8.2f %7.3f %7.3f" % (
                row["leg"], row["rounds"], row["mean_expected"],
                row["mean_accepted"], row["bias"], row["bias_pct"],
                row["slope"], row["pearson_r"]))

    result = {"harness": "local", "window": WINDOW,
              "identity": identities, "calibration": calibrations}
    if len(shipped) >= 3:
        model = manifold(shipped)
        result["manifold"] = model
        print("\n=== 3. manifold: ranked R predicted from published eff ===")
        print("leave-one-out residual in mean accepted: pstdev %.4f, "
              "max |err| %.4f over %d shipped legs" % (
                  model["loo_residual_pstdev_interior"],
                  model["loo_residual_max_abs_interior"],
                  len(model["leave_one_out"])))
        print("%-10s %7s %9s %7s %7s %7s %9s %9s" % (
            "prompt", "eff", "E[acc]~", "R pred", "R low", "R high",
            "R assumed", "pred/asm"))
        for prompt, row in model["predictions"].items():
            note = "" if row["inside_local_eff_range"] else \
                "  OUT OF RANGE, bound %s" % (
                    row["R_bound_from_non_drafting_rounds"],)
            print("%-10s %7.4f %9.4f %7.1f %7.1f %7.1f %9d %9.4f%s" % (
                prompt, row["eff"], row["predicted_mean_accepted"],
                row["R_predicted"], row["R_band"][0], row["R_band"][1],
                row["R_assumed"], row["R_ratio_predicted_over_assumed"], note))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
