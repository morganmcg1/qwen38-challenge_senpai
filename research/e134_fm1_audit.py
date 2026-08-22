"""E134 FM1 -- do the rounds `pb6` suppresses accept at the population rate?

`harness=local traces, ranked price curve`. Zero GPU.

The replayer that scores `pb6` at `+2.47 %` draws realised capability from a
fitted population survival curve, independently of round-start state
(FINDING 164). `pb6` does not add information: it raises `marginal[4]` so the
shipped walk stops at depth 4 in the rounds whose ESTIMATED reach is marginal
at that boundary. Rung 1 measured the estimator to be anti-informative exactly
there, `AUC 0.0361` at depth 4 on `medicine_hist`. If the suppressed rounds are
in fact MORE capable than the population, the replayer systematically overpays
`pb6` and the headline must be discounted.

The archived shipped traces carry the joint the replayer throws away: the
round-start `ema` and `margin` that drive the decision, and the realised `acc`
that followed it. So the question is answered by reading, not by modelling.

  1. Replay the shipped walk over the RECORDED round-start state, once under
     the ship price and once under the pb6 price.
  2. Take the flipped rounds, where pb6 stops shallower than ship.
  3. Read each flipped round's RECORDED `acc`.
  4. Price the counterfactual with the E134 item 2 measured ranked curve.

The decisive output is two conditional probabilities side by side:

  P(acc >= k | flipped by pb6)   from the recorded traces
  P(acc >= k | offered >= k)     the population rate the replayer used

`k = d_pb6 + 1` is the first draft `pb6` declines. Both the shipped-leg
population, which is selected by the same estimator, and the forced-leg
population, which is not, are reported, because the sampler is fitted on
forced legs while the flip happens under shipped policy.

Two fidelity gates run before any number is reported, and both can fail:

  - the replayed ship depth must equal the recorded `d=` on every round, so
    the walk really is the shipped walk;
  - the same flip detector at tier 1.0 must find zero flips, because tier 1.0
    IS the shipped uniform price.

Usage:
  python3 e134_fm1_audit.py --json e134-artifacts/fm1-flipped-round-audit.json
"""

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e134_rung1 import parse_trace  # noqa: E402
from e134_rung2 import build_legs, prompt_panel, simulate, walk  # noqa: E402
from e134_rung3 import CURVES, OUR_CURVE, boundary_price  # noqa: E402

SHIP_TIER = 1.0
PB6_CLIFF = 4


def load_legs(directory):
    legs = {}
    if not directory.is_dir():
        return legs
    for leg in sorted(p for p in directory.iterdir() if p.is_dir()):
        trace = leg / "trace.txt"
        if not trace.is_file():
            continue
        rounds, gate = parse_trace(trace)
        legs[leg.name] = {"rounds": rounds, "gate": gate}
    return legs


def replay(legs, tier, cliff):
    """Ship and candidate depth for every recorded round, plus the flip set.

    The FINAL round of a leg is dropped. The parent stops at a fixed 512-token
    window, so that round's draft count is clamped by the tokens still owed,
    not by the walk, and no price arm can change it. Every dropped round is
    reported with its token arithmetic rather than silently skipped.
    """
    ship = boundary_price(SHIP_TIER, cliff)[:2]
    cand = boundary_price(tier, cliff)[:2]
    out = {"rounds": 0, "sched_fidelity_bad": 0, "flipped": [],
           "deepened": 0, "records": [], "budget_truncated": []}
    for name, leg in legs.items():
        rounds = leg["rounds"]
        emitted = 0
        for index, record in enumerate(rounds):
            offer = record["cap"]
            d_ship = walk(record["ema"], record["margin"], offer, price=ship)
            d_cand = walk(record["ema"], record["margin"], offer, price=cand)
            row = {"leg": name, "round": record["round"],
                   "d_recorded": record["depth"], "d_ship": d_ship,
                   "d_cand": d_cand, "acc": record["acc"], "offer": offer,
                   "tokens_before": emitted}
            emitted += record["acc"] + 1
            if index == len(rounds) - 1:
                out["budget_truncated"].append(row)
                continue
            out["rounds"] += 1
            if d_ship != record["depth"]:
                out["sched_fidelity_bad"] += 1
            out["records"].append(row)
            if d_cand > d_ship:
                out["deepened"] += 1
            elif d_cand < d_ship:
                out["flipped"].append(row)
    return out


def survival(records, k, offered_field="d_ship"):
    """P(acc >= k) among rounds that actually drafted at least k tokens.

    A round that drafted fewer than `k` cannot report whether draft `k` would
    have been accepted, so it is excluded rather than counted as a failure.
    """
    eligible = [r for r in records if r[offered_field] >= k]
    if not eligible:
        return None
    hits = sum(1 for r in eligible if r["acc"] >= k)
    return {"n": len(eligible), "hits": hits, "rate": hits / len(eligible)}


def conditional_survival(records, k, offered_field="d_ship"):
    """P(acc >= k | acc >= k - 1), the step the walk actually gambles on."""
    eligible = [r for r in records
                if r[offered_field] >= k and r["acc"] >= k - 1]
    if not eligible:
        return None
    hits = sum(1 for r in eligible if r["acc"] >= k)
    return {"n": len(eligible), "hits": hits, "rate": hits / len(eligible)}


def rate(rows, k):
    if not rows:
        return None
    hits = sum(1 for r in rows if r["acc"] >= k)
    return {"n": len(rows), "hits": hits, "rate": hits / len(rows)}


def two_proportion_z(a, b):
    """Pooled two-proportion z for the flipped-versus-kept contrast."""
    if not a or not b or not a["n"] or not b["n"]:
        return None
    pooled = (a["hits"] + b["hits"]) / (a["n"] + b["n"])
    var = pooled * (1.0 - pooled) * (1.0 / a["n"] + 1.0 / b["n"])
    if var <= 0.0:
        return None
    return (a["rate"] - b["rate"]) / var ** 0.5


def price_leg(records, round_us):
    """First-order counterfactual cost and token count, trajectory held fixed.

    Suppressing a draft changes the tokens committed and therefore the state of
    every later round, which no archived trace can show. This prices only the
    rounds as recorded, which is the same first-order assumption the replayer
    makes, with the realised `acc` substituted for a drawn one.
    """
    us_ship = tokens_ship = us_cand = tokens_cand = 0.0
    for row in records:
        us_ship += round_us(row["d_ship"] + 1)
        tokens_ship += min(row["acc"], row["d_ship"]) + 1
        us_cand += round_us(row["d_cand"] + 1)
        tokens_cand += min(row["acc"], row["d_cand"]) + 1
    spt_ship = us_ship / tokens_ship
    spt_cand = us_cand / tokens_cand
    return {
        "us_ship": us_ship, "tokens_ship": tokens_ship, "spt_ship": spt_ship,
        "us_cand": us_cand, "tokens_cand": tokens_cand, "spt_cand": spt_cand,
        "pct": 100.0 * (spt_ship - spt_cand) / spt_cand,
    }


def curve_round_us(curve):
    def round_us(rows):
        saved = e128_price.CURVE
        e128_price.CURVE = curve
        try:
            return e128_price.ranked_round_us(rows)
        finally:
            e128_price.CURVE = saved
    return round_us


def replayer_booked_rate(args, tier, cliff):
    """P(capability > cliff | the walk asked at `cliff`), inside the replayer.

    `simulate` already counts this as `boundary_needed / boundary_asked`. The
    drawn capability does not enter the depth-4 test, which reads only the
    simulated `ema` and the recorded margin, so this rate is what the replayer
    books whenever it suppresses a draft at that boundary. Running it under
    both prices is the check on that independence claim: if the two columns
    agree, the decision really is uninformative about the draw.
    """
    try:
        legs, _ = build_legs(args.accept, args.replay_runs)
        panel = prompt_panel(legs, args.windows, args.fit_windows, args.seed)
    except (OSError, ValueError, KeyError):
        return {}
    ship = boundary_price(SHIP_TIER, cliff)[:2]
    cand = boundary_price(tier, cliff)[:2]
    out = {}
    for prompt, entry in panel.items():
        row = {}
        for label, price in (("ship", ship), ("pb6", cand)):
            run = simulate(None, entry["factory"](entry["p_target"]),
                           args.windows, price=price)
            asked = run["boundary_asked"][cliff]
            needed = run["boundary_needed"][cliff]
            row[label + "_asked"] = asked
            row[label + "_rate"] = needed / asked if asked else float("nan")
        out[prompt] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-shipped")
    ap.add_argument("--forced", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--replay-runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--tier", type=float, default=1.45)
    ap.add_argument("--cliff", type=int, default=PB6_CLIFF)
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e134-artifacts/fm1-flipped-round-audit.json")
    args = ap.parse_args()

    shipped = load_legs(args.shipped)
    forced = load_legs(args.forced)
    if not shipped:
        raise SystemExit("e134_fm1_audit: no shipped legs under %s"
                         % args.shipped)

    run = replay(shipped, args.tier, args.cliff)
    control = replay(shipped, SHIP_TIER, args.cliff)

    print("## fidelity gates")
    print("shipped legs %d ; rounds scored %d ; final rounds dropped %d"
          % (len(shipped), run["rounds"], len(run["budget_truncated"])))
    for row in run["budget_truncated"]:
        print("  dropped %-18s round %3d  recorded d=%d walk d=%d  "
              "tokens before it %d of 512" % (
                  row["leg"], row["round"], row["d_recorded"], row["d_ship"],
                  row["tokens_before"]))
    print("replayed ship depth != recorded d= on %d scored rounds"
          % run["sched_fidelity_bad"])
    print("tier 1.0 control flips %d rounds, deepens %d  (must be 0 and 0)"
          % (len(control["flipped"]), control["deepened"]))
    for name, leg in sorted(shipped.items()):
        gate = leg["gate"]
        print("  %-18s rounds %4d  row_count_bad %2d  margin_bad %2d  "
              "sched_max_abs_err %.3e" % (
                  name, gate["rounds"], gate["row_count_bad"],
                  gate["margin_identity_bad"], gate["sched_max_abs_error"]))
    gates_ok = (run["sched_fidelity_bad"] == 0
                and not control["flipped"] and control["deepened"] == 0)
    print("gates %s" % ("PASS" if gates_ok else "FAIL"))

    flipped = run["flipped"]
    print("\n## what pb6 tier %.2f does to the recorded rounds" % args.tier)
    print("rounds %d ; flipped shallower %d (%.2f %%) ; deepened %d"
          % (run["rounds"], len(flipped),
             100.0 * len(flipped) / run["rounds"], run["deepened"]))
    by_k = {}
    for row in flipped:
        by_k.setdefault(row["d_cand"] + 1, []).append(row)

    forced_rows = [{"d_ship": r["depth"], "acc": r["acc"]}
                   for leg in forced.values() for r in leg["rounds"]]

    print("\n## THE DECISIVE COMPARISON, by first declined draft k")
    print("`kept` are the rounds ship also took to depth >= k that pb6 did "
          "NOT suppress.")
    print("%3s %6s %9s %6s %9s %8s %9s %9s %9s" % (
        "k", "n_flip", "P(flip)", "n_kept", "P(kept)", "z",
        "P(sh pop)", "P(forced)", "P(f|k-1)"))
    decisive = {}
    for k in sorted(by_k):
        rows = by_k[k]
        eligible = [r for r in run["records"] if r["d_ship"] >= k]
        kept = [r for r in eligible if r["d_cand"] >= k]
        r_flip, r_kept = rate(rows, k), rate(kept, k)
        p_ship = survival(run["records"], k)
        p_forced = survival(forced_rows, k) if forced_rows else None
        entry = {
            "n_flipped": len(rows), "hits": r_flip["hits"],
            "p_flipped": r_flip["rate"],
            "kept": r_kept,
            "z_flipped_minus_kept": two_proportion_z(r_flip, r_kept),
            "population_shipped": p_ship,
            "population_forced": p_forced,
            "excess_vs_shipped": r_flip["rate"] - p_ship["rate"],
            "excess_vs_kept": (r_flip["rate"] - r_kept["rate"]
                               if r_kept else None),
            "excess_vs_forced": (r_flip["rate"] - p_forced["rate"]
                                 if p_forced else None),
            "conditional_flipped": conditional_survival(rows, k, "d_ship"),
            "conditional_population": conditional_survival(
                run["records"], k),
            "conditional_forced": (conditional_survival(forced_rows, k)
                                   if forced_rows else None),
        }
        decisive[k] = entry
        cond_f = entry["conditional_forced"]
        z = entry["z_flipped_minus_kept"]
        print("%3d %6d %9.4f %6d %9s %8s %9.4f %9s %9s" % (
            k, len(rows), r_flip["rate"], r_kept["n"] if r_kept else 0,
            ("%.4f" % r_kept["rate"]) if r_kept else "n/a",
            ("%+.2f" % z) if z is not None else "n/a",
            p_ship["rate"],
            ("%.4f" % p_forced["rate"]) if p_forced else "n/a",
            ("%.4f" % cond_f["rate"]) if cond_f else "n/a"))

    pooled_hits = sum(e["hits"] for e in decisive.values())
    pooled_n = sum(e["n_flipped"] for e in decisive.values())
    pooled_flip = pooled_hits / pooled_n if pooled_n else float("nan")
    print("pooled  P(acc >= k | flipped) = %.4f  (%d of %d)"
          % (pooled_flip, pooled_hits, pooled_n))

    print("\n## the same comparison per leg, because rung 1 found the "
          "estimator anti-informative on medicine_hist alone")
    print("%-18s %6s %9s %6s %9s %9s" % (
        "leg", "n_flip", "P(flip)", "n_kept", "P(kept)", "excess"))
    per_leg = {}
    for name in sorted(shipped):
        rows = [r for r in flipped if r["leg"] == name]
        if not rows:
            continue
        k = rows[0]["d_cand"] + 1
        kept = [r for r in run["records"]
                if r["leg"] == name and r["d_ship"] >= k and r["d_cand"] >= k]
        r_flip, r_kept = rate(rows, k), rate(kept, k)
        per_leg[name] = {"k": k, "flipped": r_flip, "kept": r_kept}
        print("%-18s %6d %9.4f %6d %9s %9s" % (
            name, r_flip["n"], r_flip["rate"], r_kept["n"] if r_kept else 0,
            ("%.4f" % r_kept["rate"]) if r_kept else "n/a",
            ("%+.4f" % (r_flip["rate"] - r_kept["rate"])) if r_kept
            else "n/a"))

    print("\n## realised first-order price, trajectory held fixed")
    print("local round-width population x ranked round-cost curve, so the "
          "MAGNITUDE is not a ranked prediction; the sign and the "
          "decomposition are what this measures.")
    curves = {name: curve for name, curve in CURVES.items()
              if name.startswith("measured")}
    curves["pre_arm"] = OUR_CURVE
    # Suppression and deepening are separated because `boundary_price` holds
    # the total: raising `marginal[4]` LOWERS every other entry, so pb6 also
    # takes some rounds deeper. A headline that nets them hides half the arm.
    only_flip = [dict(r, d_cand=min(r["d_cand"], r["d_ship"]))
                 for r in run["records"]]
    only_deep = [dict(r, d_cand=max(r["d_cand"], r["d_ship"]))
                 for r in run["records"]]
    priced = {}
    print("%-34s %10s %10s %10s %9s %9s %9s" % (
        "curve", "spt_ship", "spt_pb6", "tokens_d",
        "pct_both", "pct_supp", "pct_deep"))
    for name, curve in sorted(curves.items()):
        round_us = curve_round_us(curve)
        both = price_leg(run["records"], round_us)
        supp = price_leg(only_flip, round_us)
        deep = price_leg(only_deep, round_us)
        priced[name] = {"both": both, "suppression_only": supp,
                        "deepening_only": deep}
        print("%-34s %10.1f %10.1f %+10.0f %+9.4f %+9.4f %+9.4f" % (
            name, both["spt_ship"], both["spt_cand"],
            both["tokens_cand"] - both["tokens_ship"],
            both["pct"], supp["pct"], deep["pct"]))

    booked = replayer_booked_rate(args, args.tier, args.cliff)
    if booked:
        print("\n## what the REPLAYER books at the same boundary")
        print("In the replayer the drawn capability is independent of the "
              "depth-4 decision, so P(cap > 4 | asked at 4) IS the loss rate "
              "it books for a suppressed round. Equality of the ship and pb6 "
              "columns is the check that the independence really holds.")
        print("%-10s %10s %10s %12s %12s" % (
            "prompt", "ship", "pb6", "asked_ship", "asked_pb6"))
        for prompt in sorted(booked):
            row = booked[prompt]
            print("%-10s %10.4f %10.4f %12d %12d" % (
                prompt, row["ship_rate"], row["pb6_rate"],
                row["ship_asked"], row["pb6_asked"]))
        drift = max(abs(row["pb6_rate"] - row["ship_rate"])
                    for row in booked.values())
        print("max |pb6 - ship| across prompts %.4f" % drift)

        mean_booked = statistics.fmean(
            row["pb6_rate"] for row in booked.values())
        asked = sum(row["pb6_asked"] for row in booked.values())
        asked_booked = sum(row["pb6_rate"] * row["pb6_asked"]
                           for row in booked.values()) / asked

        # Only three panel prompts are drawn from the legs audited above, so
        # this is the one weighting that compares the same text under both
        # estimators. `benchfixture` is the public fixture and is in no panel.
        covered = {p: e128_price.RANKED_PROMPTS[p]["weight"]
                   for p in ("beagle", "medicine", "essays") if p in booked}
        held = sum(covered.values())
        like = sum(booked[p]["pb6_rate"] * w
                   for p, w in covered.items()) / held if held else float("nan")
        eligible = (decisive[5]["population_shipped"]["rate"]
                    if 5 in decisive else float("nan"))
        print("booked loss rate: equal-weight %.4f  asked-weight %.4f  "
              "like-for-like %.4f (%s, %.4f of ranked weight)"
              % (mean_booked, asked_booked, like,
                 "+".join(sorted(covered)), held))
        print("recorded: flipped %.4f  eligible %.4f" % (pooled_flip, eligible))
        print("the replayer books MORE loss than the flipped rounds realise "
              "under every weighting, so its pb6 price is a lower bound"
              if min(mean_booked, asked_booked, like) > pooled_flip else
              "WARNING: some weighting books LESS loss than realised; the "
              "replayer may overpay pb6 and the headline needs a discount")
        booked["_summary"] = {
            "equal_weight": mean_booked, "asked_weight": asked_booked,
            "like_for_like": like, "like_for_like_prompts": sorted(covered),
            "like_for_like_ranked_weight": held,
            "max_ship_pb6_drift": drift,
            "recorded_flipped": pooled_flip, "recorded_eligible": eligible,
        }

    payload = {
        "replayer_booked": booked,
        "harness": "local traces, ranked price curve",
        "tier": args.tier, "cliff": args.cliff,
        "gates_pass": gates_ok,
        "sched_fidelity_bad": run["sched_fidelity_bad"],
        "control_flips": len(control["flipped"]),
        "control_deepened": control["deepened"],
        "legs": {name: leg["gate"] for name, leg in shipped.items()},
        "rounds": run["rounds"],
        "n_flipped": len(flipped),
        "n_deepened": run["deepened"],
        "decisive": {str(k): v for k, v in decisive.items()},
        "per_leg": per_leg,
        "pooled_p_flipped": pooled_flip,
        "priced": priced,
        "flipped_rounds": flipped,
        "budget_truncated_rounds": run["budget_truncated"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("\nwrote %s" % args.json)
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
