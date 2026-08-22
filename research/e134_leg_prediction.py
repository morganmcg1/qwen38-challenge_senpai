"""E134 item 5 -- does the item 2 cliff curve PREDICT the measured leg deltas?

`harness=local`. Zero GPU: reads two archived 512-token leg sets.

Every `pb6` number reported so far is a replay or a re-price of recorded
rounds. `runs-pb6` is different: the shipped policy was rebuilt with
`depthPriceArm = .pb6` and the four legs were re-run end to end at 512 tokens.
So the realised round-width histogram and the parent-measured seconds per
token are both observed, on the same host, for both arms.

That makes an out-of-sample test available which no replay can give. Take only
the observed width histograms, price them under competing cost models, and ask
which model reproduces the observed per-leg time deltas.

  rows-only   cost proportional to declared rows. This is the model the
              round accounting implies if width is free beyond the row count.
  flat        the two-segment fit with the E134 item 2 corrections removed.
  measured*   the item 2 curves, which carry the boundary-4 step.

`medicine_hist` is the discriminating leg. Its declared rows RISE under `pb6`
(689 -> 692) while its measured time FALLS 3.8 %. The rows-only model must
therefore predict the wrong sign there, and a model that prices the width
cliff should predict the right one. If instead the cliff curves also miss, the
item 2 curve does not describe this host and the pb6 case rests on the replay
alone.

Two honest limits, stated before the numbers:

  - The arms are NOT thermally matched. `runs-shipped` was collected in an
    earlier session, `runs-pb6` just now, and neither is ABBA-counterbalanced.
    The measured deltas therefore carry unknown thermal drift, and the sign
    per leg is worth more than the magnitude.
  - The cost curves are fitted to RANKED M5 receipts and this host is an M4
    Pro. Absolute microseconds do not transfer. Only the RELATIVE per-leg
    delta is compared, and even that assumes the two hosts share the shape of
    the width-cost curve, which is the assumption under test.

Usage:
  python3 e134_leg_prediction.py --json e134-artifacts/leg-prediction.json
"""

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e134_rung3 import CURVES  # noqa: E402

LEGS = ("beagle_a", "benchfixture", "essays_montaigne", "medicine_hist")


ROUND_US = re.compile(r"round_us=([0-9.]+)")


def load(root, arm, leg):
    report = json.loads((root / arm / leg / "report.json").read_text())
    widths = collections.Counter(d + 1 for d in report["effective_draft_lengths"])
    meta = (root / arm / leg / "meta.txt").read_text()
    trace = (root / arm / leg / "trace.txt").read_text()
    in_round = sum(float(m) for m in ROUND_US.findall(trace))
    return {
        "widths": widths,
        "timing_valid": "timing_valid=false" not in meta,
        "phase_trace": "phase_trace=1" in meta,
        "in_round_us": in_round,
        "seed_us": report["seed_prefill_seconds"] * 1e6,
        "rounds": report["round_count"],
        "tokens": report["decode_token_count"],
        "rows": report["declared_rows_total"],
        "spt_us": report["parent_measured_seconds_per_token"] * 1e6,
        "decode_spt_us": (report["decode_seconds"] / report["decode_token_count"]
                          * 1e6),
        "matched": report["all_tokens_matched"],
        "divergence": report["residual_divergence_count"],
        "accepted": report["accepted_draft_total"],
        "rejected": report["rejected_draft_total"],
    }


def priced(widths, tokens, curve):
    """Predicted seconds per token from the width histogram alone."""
    saved = e128_price.CURVE
    e128_price.CURVE = curve
    try:
        total = sum(e128_price.ranked_round_us(rows) * n
                    for rows, n in widths.items())
    finally:
        e128_price.CURVE = saved
    return total / tokens


def rows_only(widths, tokens):
    return sum(rows * n for rows, n in widths.items()) / tokens


def flatten(curve):
    """The same two-segment fit with the item 2 width corrections removed."""
    out = dict(curve)
    out.pop("uniform", None)
    out.pop("per_width", None)
    return out


def pct(ship, cand):
    return (ship - cand) / ship * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128")
    ap.add_argument("--ship", default="runs-shipped")
    ap.add_argument("--cand", default="runs-pb6")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e134-artifacts/leg-prediction.json")
    args = ap.parse_args()

    arms = {}
    for leg in LEGS:
        try:
            arms[leg] = {"ship": load(args.root, args.ship, leg),
                         "pb6": load(args.root, args.cand, leg)}
        except (OSError, KeyError) as exc:
            raise SystemExit("e134_leg_prediction: %s: %s" % (leg, exc))

    print("## fidelity, both arms, 512 tokens")
    bad = 0
    for leg, arm in arms.items():
        for name, run in arm.items():
            ok = run["matched"] and run["divergence"] == 0
            bad += 0 if ok else 1
            print("  %-18s %-5s matched=%s divergence=%d tokens=%d"
                  % (leg, name, run["matched"], run["divergence"],
                     run["tokens"]))
    if bad:
        raise SystemExit("e134_leg_prediction: %d leg(s) failed exactness" % bad)
    print("all legs exact")

    # Every `e128_session.sh` leg sets `MLX_QWEN_MTP_TRACE=1` and writes two
    # lines inside every round, so `meta.txt` records `timing_valid=false`.
    # That label must travel with any time reported here. The trace cost is
    # NOT common mode: `pb6` changes the round count, so a per-round overhead
    # biases the arm comparison. The bias is bounded rather than assumed.
    #
    # `round_us` spans `tRound0` to `tTailDone` and both `traceWrite` calls
    # happen after `tTailDone`, so the whole trace cost sits in
    # `decode_seconds - sum(round_us) - seed_prefill_seconds`. Regressing that
    # residual on round count gives the largest per-round overhead the data
    # will support.
    traced = [leg for leg, arm in arms.items()
              for run in arm.values() if not run["timing_valid"]]
    print("\n## timing_valid=false on %d of %d legs; the trace cost is bounded"
          % (len(traced), 2 * len(LEGS)))
    print("%-18s %-5s %6s %10s %10s %10s"
          % ("leg", "arm", "rounds", "seed_us", "in_round", "residual"))
    points = []
    for leg, arm in arms.items():
        for name, run in arm.items():
            residual = (run["spt_us"] * run["tokens"] - run["in_round_us"]
                        - run["seed_us"])
            points.append((run["rounds"], residual))
            print("%-18s %-5s %6d %10.0f %10.0f %10.0f"
                  % (leg, name, run["rounds"], run["seed_us"],
                     run["in_round_us"], residual))

    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    slope = (sum((x - mx) * (y - my) for x, y in points)
             / sum((x - mx) ** 2 for x, _ in points))
    spread = max(abs(arms[leg]["ship"]["rounds"] - arms[leg]["pb6"]["rounds"])
                 for leg in LEGS)
    worst_leg = max(LEGS, key=lambda leg: abs(arms[leg]["ship"]["rounds"]
                                              - arms[leg]["pb6"]["rounds"]))
    worst_us = arms[worst_leg]["ship"]["spt_us"] * arms[worst_leg]["ship"]["tokens"]
    bias_pp = abs(slope) * spread / worst_us * 100.0
    print("per-round residual slope %.1f us/round" % slope)
    print("largest round-count difference %d rounds, on %s"
          % (spread, worst_leg))
    print("=> worst-case trace bias %.4f pp, and it penalises the arm with "
          "MORE rounds" % bias_pp)
    print("seed prefill is %.1f %% of the timed leg and is arm-invariant, so "
          "it dilutes every round-policy gain"
          % (statistics.fmean(run["seed_us"] / (run["spt_us"] * run["tokens"])
                              for arm in arms.values()
                              for run in arm.values()) * 100.0))

    print("\n## observed, per leg")
    print("%-18s %7s %7s %7s %7s %9s %9s %8s"
          % ("leg", "rnd_sh", "rnd_pb", "row_sh", "row_pb", "spt_sh", "spt_pb",
             "pct"))
    observed = {}
    for leg, arm in arms.items():
        s, c = arm["ship"], arm["pb6"]
        observed[leg] = pct(s["spt_us"], c["spt_us"])
        print("%-18s %7d %7d %7d %7d %9.1f %9.1f %+8.4f"
              % (leg, s["rounds"], c["rounds"], s["rows"], c["rows"],
                 s["spt_us"], c["spt_us"], observed[leg]))

    # The item 2 refit must have ADDED corrections to the pre-arm two-segment
    # fit without moving it. Strip the corrections and the pre-arm curve must
    # come back, width by width. A silent change to the base fit would make
    # every item 2 comparison against pre-arm meaningless, so it is asserted
    # rather than assumed, and `flat` is then the one pre-arm row in the table.
    flat = flatten(CURVES["measured"])
    drift = max(abs(priced({r: 1}, 1, flat) - priced({r: 1}, 1, CURVES["ours"]))
                for r in range(2, 10))
    if drift > 1e-6:
        raise SystemExit("e134_leg_prediction: item 2 moved the base fit by "
                         "%.6f us; `flat` is not the pre-arm curve" % drift)
    print("\nitem 2 left the pre-arm two-segment fit unmoved "
          "(max width drift %.2e us), so `flat` IS the pre-arm model" % drift)

    models = {"rows_only": None, "flat_pre_arm": flat}
    for name in ("measured", "measured_per_round", "measured_proportional"):
        if name in CURVES:
            models[name] = CURVES[name]

    print("\n## predicted percent improvement from the WIDTH HISTOGRAM ALONE")
    print("a model is only useful here if it reproduces the SIGN of every leg")
    header = "%-22s" % "model"
    for leg in LEGS:
        header += " %10s" % leg[:10]
    print(header + " %8s" % "signs")
    rows_out = {"observed": observed, "models": {}}
    for name, curve in models.items():
        line = "%-22s" % name
        signs = 0
        preds = {}
        for leg in LEGS:
            s, c = arms[leg]["ship"], arms[leg]["pb6"]
            if curve is None:
                p = pct(rows_only(s["widths"], s["tokens"]),
                        rows_only(c["widths"], c["tokens"]))
            else:
                p = pct(priced(s["widths"], s["tokens"], curve),
                        priced(c["widths"], c["tokens"], curve))
            preds[leg] = p
            signs += 1 if (p > 0) == (observed[leg] > 0) else 0
            line += " %+10.4f" % p
        rows_out["models"][name] = preds
        print(line + " %6d/%d" % (signs, len(LEGS)))

    line = "%-22s" % "OBSERVED"
    for leg in LEGS:
        line += " %+10.4f" % observed[leg]
    print(line + " %8s" % "-")

    weights = {leg: e128_price.RANKED_PROMPTS[p]["weight"]
               for leg, p in (("beagle_a", "beagle"),
                              ("essays_montaigne", "essays"),
                              ("medicine_hist", "medicine"))}
    held = sum(weights.values())
    obs_w = sum(observed[k] * w for k, w in weights.items()) / held
    print("\nranked-weighted over the three legs that map to ranked prompts "
          "(%.4f of ranked weight, benchfixture excluded):" % held)
    print("  OBSERVED %+.4f %%" % obs_w)
    for name, preds in rows_out["models"].items():
        print("  %-22s %+.4f %%"
              % (name, sum(preds[k] * w for k, w in weights.items()) / held))
    rows_out["ranked_weighted_observed"] = obs_w
    rows_out["ranked_weight_held"] = held

    rows_out["harness"] = "local"
    rows_out["thermally_matched"] = False
    rows_out["abba_counterbalanced"] = False
    rows_out["note"] = ("arms collected in separate sessions; per-leg SIGN is "
                        "the evidence, magnitude carries thermal drift")
    rows_out["legs"] = {leg: {name: {k: v for k, v in run.items()
                                     if k != "widths"}
                              for name, run in arm.items()}
                        for leg, arm in arms.items()}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows_out, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
