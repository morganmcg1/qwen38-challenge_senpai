#!/usr/bin/env python3
"""E205 stage B: the acceptance-sensitivity curve for reading a cap-8 receipt.

    usage: python3 research/e205_sensitivity.py [--stage-a JSON] [--json OUT]

harness=ranked for every score in this file. The local stage-A measurement
enters only as a per-round premium in milliseconds, transferred with the same
6-row ratio E197 uses for its 9-row scenario.

WHAT THIS ANSWERS. E197 predicts a cap-8 receipt at 3.81997883 under a flat
9-row continuation and 3.67941423 under the transferred local step. A receipt
that lands between those two values is unreadable without a second axis. The
second axis is ACCEPTANCE: the E197 predictions hold the paid receipts'
measured acceptance fixed, and a hidden-prompt acceptance shortfall moves the
predicted score in the same region. This file prices both axes on one grid so
an intermediate receipt value can be attributed instead of shrugged at.

THE MODEL, WITH EVERY INPUT LABELLED (RULE 386).

  MEASURED (ranked receipts, E197 / FINDING 514):
    t_p[d]   per-prompt survival of the greedy stopping depth D*, so
             P(D = d) = t[d-1] - t[d] under D = min(D*, cap).
    S_p[k]   per-prompt probability that the first k drafted rows are ALL
             accepted, i.e. S_k = P(K >= k) for the would-be accepted count K.
             abar = sum_k t[k-1] * S_k = E[min(K, D)] reproduces the paid
             receipts at caps 4, 5 and 7.
    R(m)     ranked round cost at m verified rows, smooth-step family, fitted
             to 24 weighted constraints from three paid receipts.
    serial_p, prefill_p  per-prompt ranked legs from receipt A.

  MEASURED (local, this experiment): R_rej(kappa), the round-endpoint premium
    of a rejection round over a matched full-acceptance round of the same
    depth, indexed by the accepted count kappa (the repair replays kappa + 1
    target rows).

  INFERRED: the 9-row rejection premium (no cap-8 tree exists locally and
    RULE 79 forbids building one for a timing contrast), and the local-to-
    ranked transfer of the premium.

  THE ACCEPTANCE KNOB. S_k(lam) = S_k ** lam, applied to every prompt. lam = 1
  is the measured board. lam > 1 lowers acceptance. The axis is reported as
  the implied per-draft acceptance a_p(cap) = S_cap ** (1/cap), the geometric
  rate that reproduces the same full-acceptance probability, so
  "tokens per drafting round = 1 + sum_{k=1..cap} a^k" is the closed-form
  reading of the same object.

  THE PREMIUM ENTERS AS A DELTA. Every paid receipt already contains the cost
  of its own rejection rounds, so the model adds only
  E[premium](cap, lam) - E[premium](7, 1). At the anchor the correction is
  identically zero and the curve reproduces receipt A exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e197_refit as R  # noqa: E402
import e177_ranked_depth_law as e177  # noqa: E402

ORDER = R.ORDER
TOKENS = R.TOKENS
BEST_A = R.BEST_A          # 3.70784519415395, paid cap-7 receipt
CROWN = R.CROWN            # 3.7291100105909
SIGMA_PUBLISHED = R.SIGMA_PUBLISHED
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e205-artifacts"

# E197 cap-8 desk predictions, reproduced by this file at lam = 1.
FLAT_CAP8 = 3.81997883
STEP_CAP8 = 3.67941423

# Public-fixture open-loop fixed-depth-7 acceptance census, research/
# e203-artifacts/stage0a-p7.json (benchfixture): mean accepted drafts/round.
PUBLIC_FIXTURE_ACC_MEAN = 6.0417
PUBLIC_FIXTURE_CAP = 7


# ------------------------------------------------------------------ helpers

def geometric_edl(a, cap):
    """Tokens accepted per drafting round under iid per-draft acceptance."""
    if a >= 1.0:
        return float(cap)
    return a * (1.0 - a ** cap) / (1.0 - a)


def invert_geometric(target, cap):
    """Per-draft acceptance a with sum_{k=1..cap} a^k == target."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if geometric_edl(mid, cap) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scaled_S(S, lam, deep=1.0, deep_from=8):
    """S_k ** lam for every position, then a separate multiplier on the cells
    no cap<=7 receipt can observe.

    ``lam`` moves acceptance everywhere, including the cells the paid receipts
    measure. ``deep`` moves ONLY position ``deep_from`` and beyond, which is
    the 8th drafted row: it leaves every cap-7 prediction exactly unchanged and
    moves only the cap-8 prediction. That is the sharper instrument for
    reading a cap-8 receipt, because the 8th-position acceptance is the one
    quantity the paid board extrapolates rather than measures.
    """
    out = []
    for k in range(len(S)):
        if k == 0:
            out.append(1.0)
            continue
        value = min(1.0, S[k] ** lam)
        if k >= deep_from:
            value *= deep
        out.append(value)
    return out


def monotone(S):
    """Non-increasing copy of S, for the rejection-event distribution only.

    E197's measured S is mildly non-nested -- a cap change perturbs the token
    trajectory, so S_6 > S_5 on four prompts -- and the score arithmetic keeps
    those cells exactly as fitted. A distribution over the accepted count
    needs P(K = kappa) = S_kappa - S_{kappa+1} >= 0, so the premium term uses
    this clamped copy. The clamp touches only the premium, never abar or R.
    """
    out = list(S)
    for k in range(1, len(out)):
        out[k] = min(out[k], out[k - 1])
    return out


def depth_probs(t, cap):
    """P(D = d) for d = 0..cap under D = min(D*, cap)."""
    return e177.width_probs(t, cap)


def abar_of(t, S, cap):
    return sum(S[k] * t[k - 1] for k in range(1, cap + 1))


# ------------------------------------------------------- rejection premium

def premium_model(stage_a):
    """R_rej(kappa) in ms, from the stage-A paired endpoint measurement.

    ``kappa`` is the accepted count of the rejecting round; the repair replays
    kappa + 1 target rows, so this is the physically motivated index. The
    fallback is the FINDING 494 commit-phase step, which is a SITE number and
    therefore an upper bound at the endpoint.
    """
    if stage_a is None:
        return {"kind": "finding494-site-upper-bound", "const_ms": 1.107,
                "slope_ms_per_row": 0.0, "source": "FINDING 494 site step",
                "inferred": True}
    points = stage_a["premium_by_replayed_rows"]
    xs = [p["replayed_rows"] for p in points]
    ys = [p["endpoint_premium_ms"] for p in points]
    ws = [p["weight"] for p in points]
    if len(xs) >= 2 and max(xs) > min(xs):
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        return {"kind": "measured-linear-in-replayed-rows",
                "const_ms": my - slope * mx, "slope_ms_per_row": slope,
                "source": stage_a.get("source", "e205 stage A"),
                "inferred": False, "points": points}
    return {"kind": "measured-constant", "const_ms": ys[0],
            "slope_ms_per_row": 0.0,
            "source": stage_a.get("source", "e205 stage A"),
            "inferred": False, "points": points}


def premium_ms(model, kappa, transfer):
    value = model["const_ms"] + model["slope_ms_per_row"] * (kappa + 1)
    return transfer * max(0.0, value)


def expected_premium_ms(t, S, cap, model, transfer):
    """E[premium] per round: P(D = d) x P(K = kappa) over rejecting rounds."""
    probs = depth_probs(t, cap)
    Sm = monotone(S)
    total = 0.0
    for d in range(1, cap + 1):
        p_d = probs[d]
        if p_d <= 0.0:
            continue
        for kappa in range(0, d):
            p_k = Sm[kappa] - Sm[kappa + 1] if kappa + 1 < len(Sm) else Sm[kappa]
            if p_k <= 0.0:
                continue
            total += p_d * p_k * premium_ms(model, kappa, transfer)
    return total


# ------------------------------------------------------------------- score

def predict_median(data, tvar, acc, fit, cap, lam, model, transfer,
                   deep=1.0, premium=True, anchor="A"):
    """Published median at (cap, lam, deep) with the rejection-premium delta.

    Levels stay measured, exactly as E197's chain does: the paid receipt's own
    accepted count and round cost carry the tree, and the model supplies only
    the CHANGE produced by the cap, the acceptance knobs and the premium. At
    (cap 7, lam 1, deep 1) every delta is identically zero, so the curve
    reproduces receipt A to the last digit.
    """
    raws = []
    for name in ORDER:
        t = tvar[name]
        S0 = scaled_S(acc[name]["S"], 1.0)
        S = scaled_S(acc[name]["S"], lam, deep)
        rec = data[anchor]["rec"][name]
        abar = max(0.0, rec["abar"] + abar_of(t, S, cap) - abar_of(t, S0, 7))
        probs = depth_probs(t, cap)
        probs_a = depth_probs(t, 7)
        R_round = sum(q * R.R_of(fit, d + 1) for d, q in enumerate(probs))
        R_anchor = sum(q * R.R_of(fit, d + 1) for d, q in enumerate(probs_a))
        delta_prem = 0.0
        if premium:
            delta_prem = (
                expected_premium_ms(t, S, cap, model, transfer)
                - expected_premium_ms(t, S0, 7, model, transfer)) / 1000.0
        R_pred = rec["R"] + (R_round - R_anchor) + delta_prem
        n_rounds = TOKENS / (1.0 + abar)
        raws.append(R.raw_of(data, name, n_rounds * R_pred, anchor))
    return e177.published_median(raws)


def board_acceptance(data, tvar, acc, lam, cap, deep=1.0):
    """Acceptance summary of the board at one setting of the knobs.

    ``abar_rel`` is the mean relative change in accepted drafts per round,
    which is the quantity a receipt's per-prompt `edl` and round count expose.
    ``a_drafting`` is the mean per-draft acceptance over the seven drafting
    prompts; plutarch drafts on 8% of rounds and would otherwise dominate a
    plain board mean.
    """
    rel, per_draft = [], []
    for name in ORDER:
        t = tvar[name]
        S0 = scaled_S(acc[name]["S"], 1.0)
        S = scaled_S(acc[name]["S"], lam, deep)
        base = abar_of(t, S0, 7)
        now = abar_of(t, S, cap)
        rel.append(now / base - 1.0 if base > 1e-9 else 0.0)
        if name != "plutarch":
            s_cap = S[min(cap, len(S) - 1)]
            per_draft.append(s_cap ** (1.0 / cap) if s_cap > 0 else 0.0)
    return {"abar_rel": sum(rel) / len(rel),
            "a_drafting": sum(per_draft) / len(per_draft)}


# ------------------------------------------------------------------ report

def edl_model_check(data, tvar, acc):
    """Does the iid geometric edl model reproduce the paid caps?

    a_p is calibrated on receipt A (cap 7) drafting rounds; the check predicts
    the cap-4 and cap-5 receipts with the same a_p and the measured depth
    mixture.
    """
    rows = []
    for name in ORDER:
        t, S = tvar[name], acc[name]["S"]
        drafting = t[0]
        abar7 = data["A"]["rec"][name]["abar"]
        a = invert_geometric(abar7 / drafting if drafting > 1e-9 else 0.0, 7)
        row = {"prompt": name, "drafting_fraction": drafting,
               "a_per_draft": a, "abar7_obs": abar7,
               "abar7_pred": drafting * geometric_edl(a, 7)}
        for label, cap in (("C", 4), ("E", 5)):
            obs = data[label]["rec"][name]["abar"]
            pred = drafting * geometric_edl(a, cap)
            row[f"abar{cap}_obs"] = obs
            row[f"abar{cap}_pred"] = pred
            row[f"abar{cap}_err_pct"] = 100 * (pred / obs - 1) if obs else 0.0
        rows.append(row)
    return rows


def curve_lam(data, tvar, acc, laws, model, transfer, lams):
    """Score vs a GLOBAL acceptance move, at cap 7 and cap 8."""
    out = []
    for lam in lams:
        entry = {"lam": lam}
        entry.update({f"board_{k}": v for k, v in
                      board_acceptance(data, tvar, acc, lam, 7).items()})
        for cap in (7, 8):
            for label, fit in laws.items():
                entry[f"cap{cap}_{label}"] = predict_median(
                    data, tvar, acc, fit, cap, lam, model, transfer)
        out.append(entry)
    return out


def curve_deep(data, tvar, acc, laws, model, transfer, deeps):
    """Score vs the 8th-position acceptance only.

    Every cap-7 prediction is invariant along this axis by construction, so
    this axis isolates the one acceptance cell a cap-8 receipt can refute.
    """
    out = []
    for deep in deeps:
        entry = {"deep": deep}
        entry.update({f"board_{k}": v for k, v in
                      board_acceptance(data, tvar, acc, 1.0, 8, deep).items()})
        for label, fit in laws.items():
            with_prem = predict_median(
                data, tvar, acc, fit, 8, 1.0, model, transfer, deep=deep)
            no_prem = predict_median(
                data, tvar, acc, fit, 8, 1.0, model, transfer, deep=deep,
                premium=False)
            entry[f"cap8_{label}"] = with_prem
            entry[f"cap8_{label}_no_premium"] = no_prem
            entry[f"cap8_{label}_premium_pct"] = 100 * (with_prem / no_prem - 1)
        out.append(entry)
    return out


def invert(rows, key, axis, value):
    """Axis setting whose predicted score is closest to an observed value."""
    best = min(rows, key=lambda r: abs(r[key] - value))
    inside = (min(r[key] for r in rows) - 1e-9 <= value
              <= max(r[key] for r in rows) + 1e-9)
    return {"axis": axis, "setting": best[axis], "predicted": best[key],
            "board_abar_rel_pct": 100 * best["board_abar_rel"],
            "in_range": inside}


def reading_table(lam_rows, deep_rows, laws, observed_values, anchor):
    band = SIGMA_PUBLISHED
    table = []
    for value in observed_values:
        lo, hi = value * (1 - 2 * band), value * (1 + 2 * band)
        entry = {"observed": value, "band_2sigma": [lo, hi], "laws": {}}
        for label in laws:
            key = f"cap8_{label}"
            at_measured = anchor[key]
            entry["laws"][label] = {
                "predicted_at_measured_acceptance": at_measured,
                "consistent_at_measured_acceptance": lo <= at_measured <= hi,
                "deep": invert(deep_rows, key, "deep", value),
                "lam": invert(lam_rows, key, "lam", value),
            }
        flat_ok = entry["laws"]["flat"]["consistent_at_measured_acceptance"]
        step_ok = entry["laws"]["step"]["consistent_at_measured_acceptance"]
        if flat_ok and step_ok:
            reading = "both laws (the receipt cannot separate them)"
        elif flat_ok:
            reading = "FLAT 9-row continuation, acceptance as paid"
        elif step_ok:
            reading = "STEP 9-row cell, acceptance as paid"
        elif value > entry["laws"]["flat"]["predicted_at_measured_acceptance"]:
            reading = "above flat: deep acceptance BETTER than the paid board"
        elif value < entry["laws"]["step"]["predicted_at_measured_acceptance"]:
            reading = ("below step: a 9-row cost above the transferred step, "
                       "or a deeper acceptance shortfall")
        else:
            reading = ("between the laws: read the implied 8th-position "
                       "acceptance in the deep column")
        entry["reading"] = reading
        table.append(entry)
    return table


def nearest(rows, axis, value):
    return min(rows, key=lambda r: abs(r[axis] - value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", dest="stage_a")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--transfer", type=float, default=None,
                        help="local->ranked premium transfer ratio; default "
                             "is the E197 6-row ratio")
    args = parser.parse_args()

    data = R.build_receipts()
    tvar, _ = R.survival(data)
    acc = R.acceptance(data, tvar)
    fits = R.fit_all(R.observations(data, tvar))
    flat = fits["smooth-step"]
    step, ratio = R.step9_scenario(flat)
    laws = {"flat": flat, "step": step}
    transfer = args.transfer if args.transfer is not None else ratio

    stage_a = None
    if args.stage_a:
        stage_a = json.loads(pathlib.Path(args.stage_a).read_text())
    model = premium_model(stage_a)

    lams = [round(0.70 + 0.01 * i, 4) for i in range(0, 81)]
    deeps = [round(0.00 + 0.01 * i, 4) for i in range(0, 141)]
    lam_rows = curve_lam(data, tvar, acc, laws, model, transfer, lams)
    deep_rows = curve_deep(data, tvar, acc, laws, model, transfer, deeps)
    anchor = nearest(lam_rows, "lam", 1.0)

    print("=" * 100)
    print("E205 STAGE B  acceptance-sensitivity curve (harness=ranked)")
    print("=" * 100)
    print(f"premium model: {model['kind']}  const {model['const_ms']:.3f} ms"
          f"  slope {model['slope_ms_per_row']:+.3f} ms per replayed row"
          f"  inferred={model['inferred']}  source={model['source']}")
    print(f"local->ranked premium transfer ratio {transfer:.5f}"
          f"  (E197 6-row ratio {ratio:.5f})")

    print("\nANCHOR AND REPRODUCTION CHECKS")
    cap7 = predict_median(data, tvar, acc, flat, 7, 1.0, model, transfer)
    print(f"   cap7 at measured acceptance {cap7:.14f}"
          f"   receipt A {BEST_A:.14f}   delta {cap7 - BEST_A:+.2e}")
    for label, fit, target in (("flat", flat, FLAT_CAP8),
                               ("step", step, STEP_CAP8)):
        without = predict_median(data, tvar, acc, fit, 8, 1.0, model,
                                 transfer, premium=False)
        with_prem = anchor[f"cap8_{label}"]
        print(f"   cap8 {label}: premium off {without:.8f} (E197 "
              f"{target:.8f}, delta {without - target:+.2e})   "
              f"premium on {with_prem:.8f} "
              f"({100 * (with_prem / without - 1):+.3f}%)")

    print("\nEDL MODEL CHECK  iid geometric edl(a, cap) = sum_{k<=cap} a^k,")
    print("   a calibrated on receipt A (cap 7), predicting the paid cap-4 "
          "and cap-5 receipts")
    print(f"{'prompt':10s} {'draft%':>7s} {'a':>7s} {'abar4 obs':>10s} "
          f"{'pred':>8s} {'err%':>7s} {'abar5 obs':>10s} {'pred':>8s} "
          f"{'err%':>7s}")
    checks = edl_model_check(data, tvar, acc)
    for row in checks:
        print(f"{row['prompt']:10s} {100 * row['drafting_fraction']:7.2f} "
              f"{row['a_per_draft']:7.4f} {row['abar4_obs']:10.4f} "
              f"{row['abar4_pred']:8.4f} {row['abar4_err_pct']:+7.2f} "
              f"{row['abar5_obs']:10.4f} {row['abar5_pred']:8.4f} "
              f"{row['abar5_err_pct']:+7.2f}")
    a_public = invert_geometric(PUBLIC_FIXTURE_ACC_MEAN, PUBLIC_FIXTURE_CAP)
    print(f"   public fixture, open-loop depth 7, acc_mean "
          f"{PUBLIC_FIXTURE_ACC_MEAN}: a = {a_public:.4f}  "
          f"(ranked drafting prompts "
          f"{min(r['a_per_draft'] for r in checks if r['prompt'] != 'plutarch'):.4f}"
          f" .. "
          f"{max(r['a_per_draft'] for r in checks):.4f})")

    print("\nCURVE 1  global acceptance knob (every position moves)")
    print(f"{'lam':>6s} {'abar%':>8s} {'a_draft':>8s} {'cap7':>11s} "
          f"{'cap8 flat':>11s} {'cap8 step':>11s} {'cap8-cap7 flat':>15s}")
    for row in lam_rows:
        if round(row["lam"] * 100) % 5:
            continue
        print(f"{row['lam']:6.2f} {100 * row['board_abar_rel']:+8.2f} "
              f"{row['board_a_drafting']:8.4f} {row['cap7_flat']:11.6f} "
              f"{row['cap8_flat']:11.6f} {row['cap8_step']:11.6f} "
              f"{100 * (row['cap8_flat'] / row['cap7_flat'] - 1):+14.2f}%")

    print("\nCURVE 2  8th-position acceptance only (cap 7 invariant at "
          f"{BEST_A:.6f})")
    print(f"{'deep':>6s} {'S8 x':>7s} {'abar%':>8s} {'cap8 flat':>11s} "
          f"{'cap8 step':>11s} {'vs receipt A flat':>18s} {'step':>9s} "
          f"{'premium%':>9s}")
    for row in deep_rows:
        if round(row["deep"] * 100) % 10:
            continue
        print(f"{row['deep']:6.2f} {row['deep']:7.2f} "
              f"{100 * row['board_abar_rel']:+8.2f} "
              f"{row['cap8_flat']:11.6f} {row['cap8_step']:11.6f} "
              f"{100 * (row['cap8_flat'] / BEST_A - 1):+17.2f}% "
              f"{100 * (row['cap8_step'] / BEST_A - 1):+8.2f}% "
              f"{row['cap8_flat_premium_pct']:+9.4f}")
    print("   premium% is the share of the cap-8 prediction that the "
          "rejection premium itself moves.")
    print("   At deep = 1 it is exactly zero: E197 pads the unobserved 8th "
          "acceptance cell to the 7th,")
    print("   so the padded profile makes the 8th drafted row a certain "
          "accept and adds no rejection event.")
    print("   The premium can only bite where the 8th row actually rejects, "
          "that is deep < 1.")

    candidates = [3.60, 3.65, STEP_CAP8, 3.68, 3.70, BEST_A, 3.72, CROWN,
                  3.75, 3.78, 3.80, FLAT_CAP8, 3.84]
    table = reading_table(lam_rows, deep_rows, laws, candidates, anchor)
    print("\nREADING TABLE for the E199 cap-8 receipt "
          f"(receipt channel 2sigma = {200 * SIGMA_PUBLISHED:.2f}%)")
    print(f"{'observed':>9s} {'vs A':>7s}  {'flat deep':>9s} {'flat lam':>8s}"
          f"  {'step deep':>9s} {'step lam':>8s}  reading")
    for entry in table:
        flat_e = entry["laws"]["flat"]
        step_e = entry["laws"]["step"]
        print(f"{entry['observed']:9.5f} "
              f"{100 * (entry['observed'] / BEST_A - 1):+6.2f}%  "
              f"{flat_e['deep']['setting']:9.2f} "
              f"{flat_e['lam']['setting']:8.2f}  "
              f"{step_e['deep']['setting']:9.2f} "
              f"{step_e['lam']['setting']:8.2f}  {entry['reading']}")
    print("   deep = multiplier on the 8th-position acceptance; a setting at "
          "the grid edge (0.00 or 1.40) means the axis cannot reach the value")
    print("   lam  = exponent on every position's acceptance; 1.00 is the "
          "paid board")

    payload = {
        "harness": "ranked",
        "anchor": {
            "receipt_A": BEST_A, "crown": CROWN,
            "cap7_at_measured_acceptance": cap7,
            "cap8_flat_at_measured_acceptance": anchor["cap8_flat"],
            "cap8_step_at_measured_acceptance": anchor["cap8_step"],
            "e197_flat": FLAT_CAP8, "e197_step": STEP_CAP8,
        },
        "premium_model": model,
        "transfer_ratio": transfer,
        "edl_model_check": checks,
        "public_fixture_a": a_public,
        "curve_lam": lam_rows,
        "curve_deep": deep_rows,
        "reading_table": table,
    }
    if args.json_path:
        path = pathlib.Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


