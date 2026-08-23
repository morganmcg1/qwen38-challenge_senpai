#!/usr/bin/env python3
"""E145 R2: the measured width cost curve, against the replayed one.

`harness=local`. Zero GPU. Reads `research/e145-artifacts/legs.json`.

WHAT THIS ANSWERS

1. What one round actually costs at each realised verify width, in absolute
   microseconds, read off a live 512-token decode with the depth pinned.
2. Whether the measured curve has the SAME SHAPE as the replayed curve the
   campaign prices depth from. Levels cannot be compared: this is an M4 Pro and
   the replayed curve is a ranked M5 curve. So one scalar level transfer is
   fitted across the measured widths and the residual per width is the shape
   disagreement. That residual is what R3's 5 percent trigger reads.
3. How big the 5 to 6 step really is, absolutely and as a fraction of the
   width-5 round.
4. FINDING 217. Whether the measured curve, applied to a +0.0953 shift in
   beagle's realised draft length, predicts the +0.2391 percent candidate-leg
   slowdown that ranked row `09b452f3` measured against `1760479a`.

Usage:
  python3 research/e145_curve.py --json research/e145-artifacts/curve.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

import numpy

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEGS_JSON = ROOT / "research/e145-artifacts/legs.json"

# Every fit below prices the same object the R1 closure test validated: the
# mean over all of a leg's blocks. Switching to the per-leg median would drop
# the warm first round and break comparability with that check.
BASIS = "us_mean_from_blocks"

# The curve every depth-price decision descends from. Ranked scale, rebuilt by
# inverting ranked receipts, never read off a decode.
REPLAYED_RANKED_US = {
    1: 31173.0, 2: 34619.0, 3: 38065.0, 4: 41511.0, 5: 44958.0,
    6: 61199.0, 7: 62825.0, 8: 70315.0, 9: 75639.0,
}

# FINDING 217, ranked row `09b452f3` against `1760479a`, beagle only. A head
# swap with the shipped scheduler left alone.
F217_DRAFTLEN_A = 4.3818
F217_DRAFTLEN_B = 4.4771
F217_CAND_PCT = 0.2391

R3_TRIGGER_PCT = 5.0


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def measured_curve(legs: list[dict]) -> dict:
    """Per width, the live per-round cost, pooled over the palindrome pair.

    The per-leg median of the tail is the headline, because it is the statistic
    the first-round warm miss cannot move. The pooled median over both legs'
    rounds is reported beside it, and the two agreeing is evidence that the
    palindrome positions did not drift apart.
    """
    timed = [leg for leg in legs
             if leg["slot"].startswith("r2-") and leg["pin"] != "none"]
    by_pin: dict[int, list[dict]] = {}
    for leg in timed:
        by_pin.setdefault(int(leg["pin"]), []).append(leg)

    out: dict[int, dict] = {}
    for pin, pin_legs in sorted(by_pin.items()):
        width = pin + 1
        per_leg_median = [leg["block_us_median"] for leg in pin_legs]
        pooled: list[float] = []
        for leg in pin_legs:
            blocks = leg["blocks"]
            pooled.extend(1e6 * b for b in (blocks[1:] or blocks))
        hist = pin_legs[0]["width_hist"]
        out[width] = {
            "pin": pin,
            "legs": len(pin_legs),
            "positions": [leg["position"] for leg in pin_legs],
            "us": statistics.fmean(per_leg_median),
            "us_per_leg": per_leg_median,
            "us_spread_pct": (pct(max(per_leg_median), min(per_leg_median))
                              if len(per_leg_median) > 1 else 0.0),
            "us_pooled_median": statistics.median(pooled),
            "us_pooled_mean": statistics.fmean(pooled),
            # The basis the R1 closure test was validated on: every block of
            # the leg, including the warm first round, divided by the round
            # count. Kept beside the median so a reader can see the warm miss.
            "us_mean_from_blocks": statistics.fmean(
                [leg["round_us_from_blocks"] for leg in pin_legs]),
            "rounds_pooled": len(pooled),
            "round_counts": sorted({leg["round_count"] for leg in pin_legs}),
            "realised_width_mass": hist.get(str(width), 0.0),
            "pin_leak_fraction": 1.0 - hist.get(str(width), 0.0),
            "accepted_draft_rate": sorted({round(leg["accepted_draft_rate"], 6)
                                           for leg in pin_legs}),
            "mean_draft_len": sorted({round(leg["mean_draft_len"], 6)
                                      for leg in pin_legs}),
            # The repair term. A pinned leg still rolls back and replays after
            # a rejected draft, and that cost is inside these round times.
            "replayed_round_count": sorted({leg["replayed_round_count"]
                                            for leg in pin_legs}),
            "replayed_round_share": statistics.fmean(
                [leg["replayed_round_count"] / leg["round_count"]
                 for leg in pin_legs]),
            "spt": statistics.fmean([leg["spt"] for leg in pin_legs]),
            "spt_blocks_only": statistics.fmean(
                [leg["spt_blocks_only"] for leg in pin_legs]),
            "entry_temps": [leg["gate_entry_temp_c"] for leg in pin_legs],
            "exit_temps": [leg["leg_exit_temp_c"] for leg in pin_legs],
            "all_tokens_matched": all(leg["all_tokens_matched"]
                                      for leg in pin_legs),
        }
    return out


def fit_repair_model(measured: dict[int, dict],
                     shipped_replay_share: float) -> dict:
    """Separate the width cost from the repair cost the pin forces with it.

    Pinning the width forces a deep draft even where the evidence is weak, so
    the accepted rate falls and rollback-and-replay becomes more frequent as
    the ladder rises. The pinned curve therefore carries a repair term that the
    shipped scheduler does not pay at the same rate. The model is

        blocks_us(M) = c0 + c1*M + c2*replays_per_round + c3*[M >= 6]

    fitted over the pinned legs, then re-evaluated at the shipped scheduler's
    own replay rate so the reported curve prices width alone.

    THIS FIT IS NOT IDENTIFIED AND THE FUNCTION SAYS SO. The pin forces the two
    regressors to move together: a deeper pinned draft is rejected more often,
    so the replay rate is a near-deterministic function of the pinned width.
    Between-width variation cannot separate `c1` from `c2`, and least squares
    on seven widths returns whatever split the noise prefers. The collinearity
    diagnostic is reported and the coefficients are withheld.

    The identified bound lives in `research/e145_r1.py`. A shipped leg runs a
    mixture whose replay rate differs from the mass-weighted pinned reference,
    so its closure residual bounds the per-replay cost directly.
    """
    rows, target, widths, shares = [], [], [], []
    for w, e in sorted(measured.items()):
        rows.append([1.0, float(w), e["replayed_round_share"],
                     1.0 if w >= 6 else 0.0])
        target.append(e[BASIS])
        widths.append(float(w))
        shares.append(e["replayed_round_share"])
    a = numpy.array(rows, dtype=float)
    y = numpy.array(target, dtype=float)
    if a.shape[0] <= a.shape[1]:
        return {"fitted": False, "reason":
                f"{a.shape[0]} widths cannot identify {a.shape[1]} terms"}
    collinearity = float(numpy.corrcoef(widths, shares)[0, 1])
    if abs(collinearity) > 0.9:
        return {
            "fitted": False,
            "reason": "the pin makes the replay rate a function of the width",
            "width_replay_correlation": collinearity,
            "widths": widths,
            "replay_share_per_width": shares,
            "shipped_replay_share": shipped_replay_share,
        }
    coef, *_ = numpy.linalg.lstsq(a, y, rcond=None)
    fitted = a @ coef
    corrected = {}
    for i, (w, e) in enumerate(sorted(measured.items())):
        delta = coef[2] * (shipped_replay_share - e["replayed_round_share"])
        corrected[w] = {
            "raw_us": e[BASIS],
            "fit_us": float(fitted[i]),
            "fit_residual_pct": pct(float(fitted[i]), e[BASIS]),
            "repair_adjustment_us": float(delta),
            "at_shipped_repair_us": e[BASIS] + float(delta),
            "repair_adjustment_pct": 100.0 * float(delta) / e[BASIS],
        }
    return {
        "fitted": True,
        "shipped_replay_share": shipped_replay_share,
        "intercept_us": float(coef[0]),
        "per_width_us": float(coef[1]),
        "per_replay_us": float(coef[2]),
        "cliff_us": float(coef[3]),
        "max_abs_fit_residual_pct": max(
            abs(v["fit_residual_pct"]) for v in corrected.values()),
        "max_abs_repair_adjustment_pct": max(
            abs(v["repair_adjustment_pct"]) for v in corrected.values()),
        "per_width": {str(w): v for w, v in corrected.items()},
    }


def identify_prefill(measured: dict[int, dict]) -> dict:
    """Solve for the ranked seed prefill hidden inside the replayed curve.

    R0b established that the ranked `mtp_seconds_per_token_mean` is
    seed-inclusive, so a round cost derived from it carries `P/R` and, because
    `R = 512/tokens_per_round`, a term linear in tokens per round. The local
    blocks-only curve carries no prefill at all. Two curves over the same
    widths therefore identify both unknowns in

        replayed(w) = work_scale * local(w) + P_ranked * tokens_per_round(w)/512

    `tokens_per_round(w)` comes from the pinned legs themselves, so no accept
    rate is assumed. A `P_ranked` near zero would refute R0b's reading; a value
    near the local seed prefill times the level factor supports it.

    The estimate is checked before it is published. The replayed curve is
    CONSTRAINED LINEAR over w = 1..5 by the parametric form it was fitted with,
    so over most of the ladder it carries no shape that could identify a second
    term, and the two regressors are collinear besides. A returned `P_ranked`
    outside a physically possible range means the design, not the hypothesis,
    has failed, and the R0b level-factor estimate remains the only one.
    """
    plausible_s = (0.0, 30.0)
    rows, target, widths = [], [], []
    for w, e in sorted(measured.items()):
        if w not in REPLAYED_RANKED_US:
            continue
        acc = statistics.fmean(e["accepted_draft_rate"])
        dlen = statistics.fmean(e["mean_draft_len"])
        tpr = 1.0 + acc * dlen
        rows.append([e[BASIS], tpr / 512.0])
        target.append(float(REPLAYED_RANKED_US[w]))
        widths.append((w, tpr))
    a = numpy.array(rows, dtype=float)
    y = numpy.array(target, dtype=float)
    if a.shape[0] <= a.shape[1]:
        return {"fitted": False, "reason": f"{a.shape[0]} widths"}
    coef, *_ = numpy.linalg.lstsq(a, y, rcond=None)
    work_scale, p_ranked = float(coef[0]), float(coef[1])
    scaled = a / numpy.linalg.norm(a, axis=0)
    condition = float(numpy.linalg.cond(scaled))
    if not plausible_s[0] <= p_ranked <= plausible_s[1]:
        return {
            "fitted": False,
            "reason": "the two-curve design does not identify a prefill term",
            "unconstrained_p_ranked_seconds": p_ranked,
            "unconstrained_work_scale": work_scale,
            "plausible_range_s": list(plausible_s),
            "scale_free_condition_number": condition,
            "regressor_correlation": float(numpy.corrcoef(
                a[:, 0], a[:, 1])[0, 1]),
        }
    pred = a @ coef
    per_width = {}
    for i, (w, tpr) in enumerate(widths):
        prefill_us = p_ranked * tpr / 512.0
        per_width[str(w)] = {
            "tokens_per_round": tpr,
            "local_us": measured[w][BASIS],
            "replayed_us": REPLAYED_RANKED_US[w],
            "predicted_replayed_us": float(pred[i]),
            "residual_pct": pct(float(pred[i]), REPLAYED_RANKED_US[w]),
            "prefill_component_us": prefill_us,
            "prefill_share_of_round": prefill_us / REPLAYED_RANKED_US[w],
        }
    return {
        "fitted": True,
        "work_scale_local_to_ranked": work_scale,
        "p_ranked_seconds": p_ranked,
        "max_abs_residual_pct": max(abs(v["residual_pct"])
                                    for v in per_width.values()),
        "per_width": per_width,
    }


def fit_level_transfer(measured: dict[int, dict]) -> dict:
    """One scalar `k` with `measured[w] ~= k * replayed[w]`, in log space.

    A single scalar is the only honest level conversion available: the measured
    curve is an M4 Pro curve and the replayed curve is a ranked M5 curve. Any
    per-width freedom would absorb exactly the shape disagreement this exists
    to expose. Fitting in log space makes the fit scale free, so a wide width
    cannot dominate a narrow one.
    """
    widths = [w for w in sorted(measured) if w in REPLAYED_RANKED_US]
    logs = [math.log(measured[w]["us"] / REPLAYED_RANKED_US[w])
            for w in widths]
    k = math.exp(statistics.fmean(logs))
    residual = {w: pct(measured[w]["us"], k * REPLAYED_RANKED_US[w])
                for w in widths}
    worst = max(residual, key=lambda w: abs(residual[w]))
    return {
        "k": k,
        "widths": widths,
        "per_width_ratio": {w: measured[w]["us"] / REPLAYED_RANKED_US[w]
                            for w in widths},
        "residual_pct": residual,
        "worst_width": worst,
        "worst_residual_pct": residual[worst],
        "max_abs_residual_pct": abs(residual[worst]),
        "r3_triggered": abs(residual[worst]) > R3_TRIGGER_PCT,
        "r3_trigger_pct": R3_TRIGGER_PCT,
    }


def steps(measured: dict[int, dict]) -> dict:
    ordered = sorted(measured)
    out = {}
    for index in range(1, len(ordered)):
        low, high = ordered[index - 1], ordered[index]
        m = measured[high]["us"] - measured[low]["us"]
        r = REPLAYED_RANKED_US[high] - REPLAYED_RANKED_US[low]
        out[f"{low}->{high}"] = {
            "measured_us": m,
            "replayed_ranked_us": r,
            "measured_over_replayed": m / r if r else None,
            "measured_as_share_of_low_round": m / measured[low]["us"],
            "replayed_as_share_of_low_round": r / REPLAYED_RANKED_US[low],
        }
    return out


def cliff_summary(measured: dict[int, dict], step_table: dict) -> dict:
    """The 5 to 6 step, which is the number the whole apparatus rests on."""
    key = "5->6"
    if key not in step_table:
        return {"available": False}
    step = step_table[key]
    others = [v["measured_us"] for k, v in step_table.items() if k != key]
    typical = statistics.fmean(others) if others else float("nan")
    return {
        "available": True,
        "measured_us": step["measured_us"],
        "replayed_ranked_us": step["replayed_ranked_us"],
        "measured_share_of_width5_round": step["measured_as_share_of_low_round"],
        "replayed_share_of_width5_round": step["replayed_as_share_of_low_round"],
        "mean_other_measured_step_us": typical,
        # A cliff is a step that is bigger than the ordinary steps around it.
        # This is the number that decides whether a cliff exists at all.
        "cliff_excess_over_typical_step": (step["measured_us"] / typical
                                           if typical else None),
        "replayed_cliff_excess": (
            step["replayed_ranked_us"]
            / statistics.fmean([v["replayed_ranked_us"]
                                for k, v in step_table.items() if k != key])
            if len(step_table) > 1 else None),
    }


def cost_of(hist: dict[int, float], curve: dict[int, float]) -> float:
    total = sum(hist.values())
    return sum(mass * curve[w] for w, mass in hist.items()) / total


def shift_hist(hist: dict[int, float], delta: float,
               max_width: int) -> dict[int, float]:
    """Move `delta` of mean width upward, one width at a time.

    A "+0.0953 of mean draft length" does not name a unique histogram. This
    moves a fraction `theta` of the mass at every width up to the next width,
    with the same `theta` everywhere, which is the shift an across-the-board
    acceptance improvement produces under a threshold walk: every round becomes
    `theta` more likely to clear its next threshold. `theta` equals `delta`
    exactly, because moving `theta` of every unit of mass up by one width
    raises the mean by `theta`. Mass already at `max_width` cannot move, so the
    realised shift is reported and checked, not assumed.
    """
    out: dict[int, float] = {w: 0.0 for w in hist}
    for w, mass in hist.items():
        if w >= max_width:
            out[w] = out.get(w, 0.0) + mass
            continue
        out[w] = out.get(w, 0.0) + mass * (1.0 - delta)
        out[w + 1] = out.get(w + 1, 0.0) + mass * delta
    return out


def mean_width(hist: dict[int, float]) -> float:
    total = sum(hist.values())
    return sum(w * m for w, m in hist.items()) / total


def geometric_accepted(depth: float, p: float) -> float:
    """Expected accepted drafts for `depth` drafted tokens at per-step `p`."""
    if p >= 1.0:
        return depth
    return p * (1.0 - p ** depth) / (1.0 - p)


def f217_validation(measured: dict[int, dict], legs: list[dict],
                    transfer: dict) -> dict:
    """Does the measured curve reproduce the ranked beagle slowdown?

    The test is only decidable if the token side is included. A wider round is
    always slower, so a cost-only prediction predicts a slowdown for any
    positive width shift and can never fail. The candidate leg's seconds per
    token is

        s/tok = E[round cost] / (1 + accepted per round)

    so the prediction needs both terms. Two scenarios bound the token side:

      cost_only   the extra drafted token is never accepted. Tokens per round
                  do not move. This is the LARGEST slowdown the curve can
                  predict and it is not a realistic head improvement.
      same_p      the extra drafted token is accepted at the leg's own
                  observed per-step acceptance. This is the natural reading of
                  a head that got better.

    A measured ranked value inside the two is a pass.
    """
    ship = [leg for leg in legs
            if leg["slot"].startswith("r1-") and leg["fixture"] == "beagle_a"
            and leg["arm"] == "ship"]
    if not ship or not measured:
        return {"available": False,
                "reason": "needs the R1 beagle_a ship arm and the R2 curve"}

    leg = ship[0]
    hist = {int(w): m for w, m in leg["width_hist"].items()}
    max_width = max(measured)
    curve_measured = {w: measured[w]["us"] for w in measured}
    k = transfer["k"]
    curve_replayed_local = {w: k * REPLAYED_RANKED_US[w] for w in measured}

    # Widths the pin did not reach cannot be priced, so drop that mass and
    # report how much was dropped rather than inventing a cost for it.
    priced = {w: m for w, m in hist.items() if w in curve_measured}
    unpriced_mass = 1.0 - sum(priced.values()) / sum(hist.values())

    delta = F217_DRAFTLEN_B - F217_DRAFTLEN_A
    shifted = shift_hist(priced, delta, max_width)
    realised_shift = mean_width(shifted) - mean_width(priced)

    p = leg["accepted_draft_rate"]
    depth_a = mean_width(priced) - 1.0
    depth_b = mean_width(shifted) - 1.0
    acc_a = geometric_accepted(depth_a, p)
    acc_b = geometric_accepted(depth_b, p)

    out = {
        "available": True,
        "source_leg": leg["slot"],
        "local_beagle_draftlen_ship": leg["mean_draft_len"],
        "ranked_beagle_draftlen_a": F217_DRAFTLEN_A,
        "ranked_beagle_draftlen_b": F217_DRAFTLEN_B,
        "requested_shift": delta,
        "realised_shift": realised_shift,
        "unpriced_width_mass": unpriced_mass,
        "hist_before": priced,
        "hist_after": shifted,
        "per_step_acceptance": p,
        "accepted_per_round_before": acc_a,
        "accepted_per_round_after": acc_b,
        "ranked_measured_cand_pct": F217_CAND_PCT,
    }

    for name, curve in (("measured", curve_measured),
                        ("replayed_rescaled", curve_replayed_local)):
        cost_a = cost_of(priced, curve)
        cost_b = cost_of(shifted, curve)
        cost_only = pct(cost_b, cost_a)
        same_p = pct(cost_b / (1.0 + acc_b), cost_a / (1.0 + acc_a))
        low, high = sorted((cost_only, same_p))
        out[name] = {
            "round_cost_before_us": cost_a,
            "round_cost_after_us": cost_b,
            "cost_only_pct": cost_only,
            "same_p_pct": same_p,
            "bracket_low_pct": low,
            "bracket_high_pct": high,
            "brackets_ranked_measurement": low <= F217_CAND_PCT <= high,
            "sign_matches_ranked": (cost_only > 0) == (F217_CAND_PCT > 0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/e145-artifacts/curve.json")
    args = ap.parse_args()

    legs = json.loads(LEGS_JSON.read_text())["legs"]
    measured = measured_curve(legs)
    if not measured:
        print("e145_curve: no R2 timed legs yet")
        return 1

    transfer = fit_level_transfer(measured)
    step_table = steps(measured)
    shipped = [leg for leg in legs
               if leg["slot"].startswith("r1-") and leg["arm"] == "ship"
               and leg["fixture"] == "beagle_a"]
    shipped_share = (statistics.fmean(
        [l["replayed_round_count"] / l["round_count"] for l in shipped])
        if shipped else 0.0)
    repair = fit_repair_model(measured, shipped_share)
    prefill = identify_prefill(measured)
    blob = {
        "harness": "local",
        "measured_widths": sorted(measured),
        "measured": {str(w): v for w, v in measured.items()},
        "replayed_ranked_us": REPLAYED_RANKED_US,
        "level_transfer": {
            "k": transfer["k"],
            "per_width_ratio": {str(w): v for w, v
                                in transfer["per_width_ratio"].items()},
            "residual_pct": {str(w): v for w, v
                             in transfer["residual_pct"].items()},
            "worst_width": transfer["worst_width"],
            "worst_residual_pct": transfer["worst_residual_pct"],
            "max_abs_residual_pct": transfer["max_abs_residual_pct"],
            "r3_triggered": transfer["r3_triggered"],
            "r3_trigger_pct": transfer["r3_trigger_pct"],
        },
        "steps": step_table,
        "cliff": cliff_summary(measured, step_table),
        "repair_model": repair,
        "ranked_prefill": prefill,
        "f217": f217_validation(measured, legs, transfer),
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")

    print(f"e145_curve: {len(measured)} widths -> {out}")
    print("\nmeasured live per-round cost, microseconds")
    print(f"  {'M':>2} {'measured':>10} {'pooled':>10} {'spread%':>8}"
          f" {'replayed':>9} {'ratio':>7} {'resid%':>8} {'leak':>7}"
          f" {'rounds':>7}")
    for w in sorted(measured):
        e = measured[w]
        print(f"  {w:>2} {e['us']:>10.1f} {e['us_pooled_median']:>10.1f}"
              f" {e['us_spread_pct']:>8.3f}"
              f" {REPLAYED_RANKED_US[w]:>9.0f}"
              f" {transfer['per_width_ratio'][w]:>7.4f}"
              f" {transfer['residual_pct'][w]:>+8.2f}"
              f" {e['pin_leak_fraction']:>7.4f}"
              f" {str(e['round_counts']):>7}")
    print(f"\nlevel transfer k = {transfer['k']:.4f}"
          f"   worst shape residual {transfer['worst_residual_pct']:+.2f} %"
          f" at width {transfer['worst_width']}"
          f"   R3 triggered: {transfer['r3_triggered']}")

    print("\nsteps")
    for name, s in step_table.items():
        print(f"  {name:>6}: measured {s['measured_us']:>+9.1f} us"
              f"  replayed {s['replayed_ranked_us']:>+9.1f} us"
              f"  ratio {s['measured_over_replayed']:>7.4f}"
              f"  as share of lower round"
              f" {s['measured_as_share_of_low_round']:>7.4f}"
              f" against {s['replayed_as_share_of_low_round']:>7.4f}")

    cliff = blob["cliff"]
    if cliff.get("available"):
        print(f"\nthe 5 to 6 step")
        print(f"  measured   {cliff['measured_us']:.1f} us"
              f"   {100 * cliff['measured_share_of_width5_round']:.2f} %"
              f" of the width-5 round")
        print(f"  replayed   {cliff['replayed_ranked_us']:.1f} us"
              f"   {100 * cliff['replayed_share_of_width5_round']:.2f} %"
              f" of the width-5 round")
        print(f"  measured step / mean other measured step"
              f" {cliff['cliff_excess_over_typical_step']:.3f}"
              f"   replayed {cliff['replayed_cliff_excess']:.3f}")

    if repair.get("fitted"):
        print(f"\nrepair model, evaluated at the shipped beagle_a replay rate"
              f" {repair['shipped_replay_share']:.4f} per round")
        print(f"  intercept {repair['intercept_us']:>10.1f} us"
              f"   per width {repair['per_width_us']:>9.1f} us"
              f"   per replay {repair['per_replay_us']:>10.1f} us"
              f"   cliff {repair['cliff_us']:>9.1f} us")
        for w in sorted(measured):
            v = repair["per_width"][str(w)]
            print(f"  M={w}  raw {v['raw_us']:>9.1f}"
                  f"   at shipped repair {v['at_shipped_repair_us']:>9.1f}"
                  f"   adjustment {v['repair_adjustment_pct']:>+7.3f} %"
                  f"   fit residual {v['fit_residual_pct']:>+7.3f} %")
        print(f"  worst repair adjustment"
              f" {repair['max_abs_repair_adjustment_pct']:.3f} %"
              f"   worst fit residual"
              f" {repair['max_abs_fit_residual_pct']:.3f} %")
    else:
        print(f"\nrepair model not fitted: {repair.get('reason')}")

    if prefill.get("fitted"):
        print(f"\nranked seed prefill identified from the two curves")
        print(f"  local work scale {prefill['work_scale_local_to_ranked']:.4f}"
              f"   P_ranked {prefill['p_ranked_seconds']:.4f} s"
              f"   worst residual {prefill['max_abs_residual_pct']:+.2f} %")
        for w in sorted(measured):
            v = prefill["per_width"].get(str(w))
            if not v:
                continue
            print(f"  M={w}  tok/round {v['tokens_per_round']:.4f}"
                  f"   prefill {v['prefill_component_us']:>8.1f} us"
                  f"   {100 * v['prefill_share_of_round']:>5.2f} % of round"
                  f"   residual {v['residual_pct']:>+7.3f} %")
    else:
        print(f"\nranked prefill not identified: {prefill.get('reason')}")

    f217 = blob["f217"]
    if f217.get("available"):
        print("\nFINDING 217 validation, beagle +0.0953 draft length")
        print(f"  ranked measured candidate delta"
              f" {f217['ranked_measured_cand_pct']:+.4f} %")
        for name in ("measured", "replayed_rescaled"):
            r = f217[name]
            print(f"  {name:>18}: cost-only {r['cost_only_pct']:+.4f} %"
                  f"   same-p {r['same_p_pct']:+.4f} %"
                  f"   brackets ranked: {r['brackets_ranked_measurement']}")
        print(f"  unpriced width mass {f217['unpriced_width_mass']:.4f}"
              f"   realised shift {f217['realised_shift']:.4f}")
    else:
        print(f"\nFINDING 217 validation unavailable: {f217.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
