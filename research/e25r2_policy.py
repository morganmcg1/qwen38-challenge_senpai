#!/usr/bin/env python3
"""E25 r2 deliverable (d): exact offline replay of any row-price rule.

`costModelDepth` reads only three recorded quantities before it proposes
anything: the pending primary's target top-2 margin (`m=`), the per-position
acceptance EMAs (`ema=`), and the width cap (`cap=`). The forced-depth
instrument records all three on every round, so the depth a DIFFERENT price
vector would have chosen is reconstructible exactly, with no unevaluable
rounds and no second GPU run.

The engine is validated against ground truth before it is used: the forced arm
also stamps `shipped=`, the depth the shipped BASE walk computed from that same
recorded state, so a faithful replay must reproduce every one of those marks.

What replay CANNOT do is invent acceptance outcomes at depths the leg never
drafted. Token counts therefore come from the round's own recorded belief
(1 + expected), and `calibration` reports how well that belief predicted the
accepted counts actually observed at the forced depth.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e21_trace import parse_trace  # noqa: E402

MAX_DEPTH = 8
WARMUP_ROUNDS = 8
# The trace snapshots `widthCap`, but the walk runs against
# min(offeredDepth, maxDepth, widthCap). `offeredDepth < MAX_DEPTH` requires
# fewer than MAX_DEPTH tokens left in the fixed window, and every round emits
# at least one token, so only the final MAX_DEPTH rounds can be offer-bound
# rather than price-bound. Those rounds are not replayable from the record.
TAIL_ROUNDS = MAX_DEPTH
SHIPPED_H = 0.18
# E25 r1 arm D, as committed: max(base price, measured T-step ratio).
R1_MEASURED_STEP = [0.0, 0.095904, 0.152261, 0.442442]


def base_coeffs(h: float = SHIPPED_H) -> list[float]:
    return [h / (1.0 + d * h) for d in range(MAX_DEPTH)]


def arm_d_coeffs(measured: list[float] | None = None) -> list[float]:
    measured = R1_MEASURED_STEP if measured is None else measured
    base = base_coeffs()
    return [max(base[d], measured[d]) if d < len(measured) else base[d]
            for d in range(MAX_DEPTH)]


def walk(coeffs: list[float], margin: float, ema: list[float], cap: int):
    """Faithful port of `costModelDepth`'s extension walk.

    Returns (depth, expected). `1 + expected` is the round's own predicted
    token yield at the depth it selected.
    """
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = margin == margin  # NaN-safe
    while depth < cap:
        p = ema[depth]
        if have_margin:
            if depth == 0:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
            elif depth == 1:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
        reach *= p
        if not reach > coeffs[depth] * (1.0 + expected):
            break
        expected += reach
        depth += 1
    return depth, expected


def expected_at(margin: float, ema: list[float], depth: int) -> float:
    """`expected` accumulated by forcing the walk to exactly `depth` rows."""
    return walk([-1.0] * MAX_DEPTH, margin, ema, depth)[1]


def load_rounds(run_dirs: list[Path]) -> list[dict]:
    rounds = []
    for d in run_dirs:
        sessions = parse_trace(d / "trace.txt")
        if len(sessions) != 1:
            raise SystemExit(f"{d}: expected one traced session, got {len(sessions)}")
        for r in sessions[0][WARMUP_ROUNDS:-TAIL_ROUNDS]:
            if "ema" not in r or len(r["ema"]) < MAX_DEPTH:
                continue
            if "shipped_depth" not in r:
                continue
            r["prompt"] = d.name
            rounds.append(r)
    return rounds


def observed_depths(run_dirs: list[Path]) -> dict:
    """Runtime depth histogram of an unforced arm, straight off its own trace.

    Independent of the offline replay: whatever a price vector does in theory,
    this is the depth the shipped binary actually selected. A price whose
    `max_depth_observed` sits below the width cap is behaving as a hard cap.
    """
    hist, accepted, widths, rounds = {}, 0, {}, 0
    for d in run_dirs:
        sessions = parse_trace(d / "trace.txt")
        if len(sessions) != 1:
            raise SystemExit(f"{d}: expected one traced session, got {len(sessions)}")
        for r in sessions[0][WARMUP_ROUNDS:-TAIL_ROUNDS]:
            rounds += 1
            hist[r["d"]] = hist.get(r["d"], 0) + 1
            widths[r["cap"]] = widths.get(r["cap"], 0) + 1
            accepted += r["acc"]
    if not rounds:
        raise SystemExit("observed_depths: no analysable rounds")
    return {
        "rounds": rounds,
        "depth_histogram": dict(sorted(hist.items())),
        "width_cap_histogram": dict(sorted(widths.items())),
        "max_depth_observed": max(hist),
        "rounds_at_depth_ge_4": sum(n for d, n in hist.items() if d >= 4),
        "mean_depth": round(sum(d * n for d, n in hist.items()) / rounds, 4),
        "mean_accepted": round(accepted / rounds, 4),
        "tokens_per_round": round(1.0 + accepted / rounds, 4),
    }


def validate(rounds: list[dict]) -> dict:
    """Acid test: replay the shipped walk and demand every mark reproduces."""
    coeffs, ok, bad = base_coeffs(), 0, []
    for r in rounds:
        d, _ = walk(coeffs, r.get("m", float("nan")), r["ema"], r["cap"])
        if d == r["shipped_depth"]:
            ok += 1
        elif len(bad) < 5:
            bad.append({"round": r["round"], "prompt": r["prompt"],
                        "replayed": d, "recorded": r["shipped_depth"]})
    return {"rounds": len(rounds), "reproduced": ok,
            "exact": ok == len(rounds), "mismatches": bad}


def calibration(rounds: list[dict]) -> dict:
    """Does the round's own belief predict the tokens it actually accepted?"""
    by_depth, pred_all, act_all = {}, [], []
    for r in rounds:
        f = r.get("forced_depth")
        if f is None or "acc" not in r:
            continue
        pred = 1.0 + expected_at(r.get("m", float("nan")), r["ema"], f)
        act = 1.0 + r["acc"]
        by_depth.setdefault(f, []).append((pred, act))
        pred_all.append(pred)
        act_all.append(act)
    out = {}
    for d, pairs in sorted(by_depth.items()):
        p = [x for x, _ in pairs]
        a = [y for _, y in pairs]
        out[d] = {
            "n": len(pairs),
            "predicted_tokens": round(statistics.fmean(p), 4),
            "actual_tokens": round(statistics.fmean(a), 4),
            "bias": round(statistics.fmean(p) - statistics.fmean(a), 4),
            "ratio": round(statistics.fmean(p) / statistics.fmean(a), 4),
        }
    return {
        "by_forced_depth": out,
        "pooled_predicted": round(statistics.fmean(pred_all), 4),
        "pooled_actual": round(statistics.fmean(act_all), 4),
        "pooled_ratio": round(statistics.fmean(pred_all) / statistics.fmean(act_all), 4),
        "n": len(pred_all),
    }


def score(rounds: list[dict], coeffs: list[float], t_ms: dict[int, float],
          shrink: float = 1.0, cap: int = MAX_DEPTH) -> dict:
    """Rate of a price vector: predicted tokens per modelled millisecond.

    `shrink` rescales the belief's surplus over 1 token by the calibration
    ratio so an optimistic belief cannot buy depth for free.
    """
    tokens, ms, hist, capped = 0.0, 0.0, {}, 0
    for r in rounds:
        limit = min(r["cap"], cap)
        d, exp = walk(coeffs, r.get("m", float("nan")), r["ema"], limit)
        if d == limit and limit < r["cap"]:
            capped += 1
        tokens += 1.0 + exp * shrink
        ms += t_ms[d]
        hist[d] = hist.get(d, 0) + 1
    return {
        "tokens_per_ms": tokens / ms,
        "ms_per_token": ms / tokens,
        "mean_depth": sum(d * n for d, n in hist.items()) / len(rounds),
        "depth_histogram": dict(sorted(hist.items())),
        "rounds_hitting_cap": capped,
        "depth_cap": cap,
        "coeffs": [round(c, 6) for c in coeffs],
    }


def optimise(rounds: list[dict], t_ms: dict[int, float], shrink: float,
             rounds_of_search: int = 6) -> list[float]:
    """Coordinate descent on the price vector, maximising predicted rate."""
    best = base_coeffs()
    best_v = score(rounds, best, t_ms, shrink)["tokens_per_ms"]
    grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
            0.22, 0.26, 0.30, 0.40, 0.55, 0.80, 1.20, 2.0]
    for _ in range(rounds_of_search):
        improved = False
        for d in range(MAX_DEPTH):
            for g in grid:
                trial = list(best)
                trial[d] = g
                v = score(rounds, trial, t_ms, shrink)["tokens_per_ms"]
                if v > best_v * (1.0 + 1e-9):
                    best, best_v, improved = trial, v, True
        if not improved:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", default="research/e25r2-pool.json",
                    help="e25r2_refit.py report supplying the measured T(d)")
    ap.add_argument("--runs-root", default=".mlxfast-private/e25/runs")
    ap.add_argument("--arm", default="FORCE")
    ap.add_argument("--prompts", default="english,narrative,technical")
    ap.add_argument("--observed-arm", default="PRICE",
                    help="unforced arm whose real runtime depths are reported")
    ap.add_argument("--observed-prompts", default="english")
    ap.add_argument("--out")
    args = ap.parse_args()

    refit = json.load(open(args.refit))
    t_ms = {int(d): v["mean_ms"]
            for d, v in refit["time_parent_clock"]["round_ms"].items()}
    missing = [d for d in range(MAX_DEPTH) if d not in t_ms]
    if missing:
        raise SystemExit(f"refit lacks measured T(d) for depths {missing}")
    # The forced cycle spans depths 0..7; the shipped cap also admits depth 8
    # (M=9, a THIRD weight-stream pass at IPG 3), so a rule with cheap deep
    # rows can select it. Extrapolate that one point from the fitted pass model
    # rather than dropping the rounds, and keep it labelled as modelled.
    pcm = refit["pass_count_model"]
    t_ms[MAX_DEPTH] = (pcm["intercept_ms"] + pcm["per_row_ms"] * MAX_DEPTH
                       + pcm["per_weight_pass_ms"] * 3)

    root = Path(args.runs_root)
    dirs = [root / f"probe-{p}-{args.arm}" for p in args.prompts.split(",")]
    rounds = load_rounds(dirs)

    val = validate(rounds)
    cal = calibration(rounds)
    shrink = 1.0 / cal["pooled_ratio"]
    obs = observed_depths([root / f"probe-{p}-{args.observed_arm}"
                           for p in args.observed_prompts.split(",")])

    arms = {
        "base_shipped_h0.18": base_coeffs(),
        "arm_d_r1_measured": arm_d_coeffs(),
        "arm_d_refit_measured": arm_d_coeffs(
            [0.0] + [refit["admissibility_parent"][str(d)]["measured_c"]
                     for d in range(3)]),
        "free_deep_rows": base_coeffs()[:2] + [0.0] * (MAX_DEPTH - 2),
    }
    best = optimise(rounds, t_ms, shrink)
    arms["coordinate_optimum"] = best

    scored = {k: score(rounds, c, t_ms, shrink) for k, c in arms.items()}
    # Arm D is behaviourally a hard DEEP_CAP; price the cap on its own so its
    # cost is separated from the d<3 price changes it also carries.
    for c in (3, 4, 5):
        scored[f"base_shipped_deep_cap_{c}"] = score(
            rounds, base_coeffs(), t_ms, shrink, cap=c)
    ref = scored["base_shipped_h0.18"]["tokens_per_ms"]
    for v in scored.values():
        v["vs_base_pct"] = round(100.0 * (v["tokens_per_ms"] / ref - 1.0), 4)

    report = {
        "prompts": args.prompts.split(","),
        "arm": args.arm,
        "rounds": len(rounds),
        "warmup_rounds_dropped_per_prompt": WARMUP_ROUNDS,
        "tail_rounds_dropped_per_prompt": TAIL_ROUNDS,
        "replay_validation": val,
        "calibration": cal,
        "belief_shrink_applied": round(shrink, 6),
        "measured_T_ms": {d: round(t_ms[d], 4) for d in sorted(t_ms) if d < MAX_DEPTH},
        "modelled_T8_ms": round(t_ms[MAX_DEPTH], 4),
        "arms": scored,
        "observed_runtime_depths": obs,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"rounds={len(rounds)}  replay_exact={val['exact']} "
          f"({val['reproduced']}/{val['rounds']})")
    if val["mismatches"]:
        print("  mismatches:", val["mismatches"])
    print(f"belief calibration: predicted {cal['pooled_predicted']} vs actual "
          f"{cal['pooled_actual']} tokens/round (ratio {cal['pooled_ratio']}), "
          f"shrink={shrink:.4f}")
    print("\n  forced d |    n | predicted | actual | ratio")
    for d, v in cal["by_forced_depth"].items():
        print(f"  {d:8d} | {v['n']:4d} | {v['predicted_tokens']:9.4f} | "
              f"{v['actual_tokens']:6.4f} | {v['ratio']:.4f}")
    print("\n  arm                    | ms/token | mean d | vs base | histogram")
    for k, v in sorted(scored.items(), key=lambda kv: kv[1]["ms_per_token"]):
        print(f"  {k:22s} | {v['ms_per_token']:8.4f} | {v['mean_depth']:6.3f} | "
              f"{v['vs_base_pct']:+7.3f}% | {v['depth_histogram']}")
    print("\n  coordinate optimum coeffs:", [round(c, 4) for c in best])
    print(f"\nOBSERVED RUNTIME DEPTHS  arm={args.observed_arm} "
          f"prompts={args.observed_prompts} rounds={obs['rounds']}")
    print(f"  depth_histogram={obs['depth_histogram']} "
          f"max_depth_observed={obs['max_depth_observed']} "
          f"rounds_at_depth_ge_4={obs['rounds_at_depth_ge_4']}")
    print(f"  width_cap_histogram={obs['width_cap_histogram']} "
          f"mean_depth={obs['mean_depth']} "
          f"tokens_per_round={obs['tokens_per_round']}")


if __name__ == "__main__":
    main()
