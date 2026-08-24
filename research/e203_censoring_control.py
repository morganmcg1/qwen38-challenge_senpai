#!/usr/bin/env python3
"""E203 positive control: the shipped rule manufactures its own structure.

WHY THIS FILE EXISTS. Read on the SHIPPED adaptive leg, the accepted-length
sequence looks strongly predictable: lag-1 autocorrelation +0.117 against a
+-0.04 permutation band, and the live EMA forecast correlates +0.25 with the
realised accepted count. Read on the fixed-depth-7 leg of the SAME prompts,
both statistics collapse to +0.001 and +0.009. An experiment that priced
within-prompt lookahead from the shipped trace alone would have read the first
pair as a live signal.

THE MECHANISM. On the shipped leg the accepted count is censored at the depth
the rule chose, and that depth rises with the EMA of recent accepted counts. A
run of good rounds therefore raises the depth, which RAISES THE CEILING on the
next observation, and a run of bad rounds lowers it. The correlation is
created by the rule's own feedback, not by the prompt.

THE CONTROL. Feed the shipped closed loop a sequence that is exchangeable BY
CONSTRUCTION -- a permutation of the fixed-depth-7 run lengths of the same
prompt, so no serial structure can survive -- and measure the same statistics
on the simulated leg. Whatever the loop produces from an exchangeable input is
manufactured. The round's own top-two margin travels with its run length
through the permutation, so the only thing destroyed is the ORDER.

WHAT IS EXACT. `cost_model_depth` and `record_accept_outcome` are the replayed
shipped functions from `e128_replay`, which reproduces the live `sched=` string
byte for byte (E128). Acceptance is a prefix rule, so an input run length g
observed at depth d yields min(g, d): this is the same exactness argument the
E200 replay uses for a DOWN shift, and here every simulated depth is at or
below 7, the depth the input was measured at.

LIMIT. Run lengths of exactly 7 are censored in the input too, so the simulated
leg slightly under-states long runs. The affected share is reported per prompt;
it is under 3 % for every prose prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_replay as R  # noqa: E402
import e203_stage0a as S  # noqa: E402
import e203_traces as T  # noqa: E402

FIXED = 7
TOKENS = 512
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e203-artifacts"


def simulate(runs: np.ndarray, margins: np.ndarray, offered: int = 8,
             carry: bool = False):
    """One closed-loop decode leg driven by a given run-length sequence.

    `carry` switches on the token-level accounting. A run length is a property
    of the TOKEN STREAM -- how many consecutive tokens the head gets right from
    the current position -- not of the round. When the rule drafts fewer tokens
    than the run holds, the round is censored and the REST OF THAT RUN is still
    there for the next round. Without carry, a censored remainder is thrown
    away and the next round draws a fresh run, which understates the serial
    correlation the rule creates for itself. The two legs of this leg-level
    experiment are decoding the same 512 tokens, so carry is the faithful
    accounting and the discard mode is the conservative one.
    """
    ema = list(R.EMA_PRIOR)
    emitted, i = 0, 0
    pending = 0
    depths, accs, forecasts = [], [], []
    n = runs.size
    while emitted < TOKENS and i < n:
        offer = max(1, min(offered, R.MAX_DEPTH, TOKENS - emitted - 1))
        depth, _, _ = R.cost_model_depth(ema, float(margins[i]),
                                         offered_depth=offer)
        forecasts.append(S.ema_expected_accept(ema, depth))
        if carry and pending > 0:
            run = pending
        else:
            run = int(runs[i])
            i += 1
        accepted = int(min(run, depth))
        pending = run - accepted if (carry and accepted == depth) else 0
        depths.append(depth)
        accs.append(accepted)
        ema = R.record_accept_outcome(ema, accepted, depth)
        emitted += 1 + accepted
    return (np.array(depths, dtype=float), np.array(accs, dtype=float),
            np.array(forecasts, dtype=float))


def stats(depths, accs, forecasts):
    rho1 = float(S.acf(accs, 1))
    f = forecasts - forecasts.mean()
    a = accs - accs.mean()
    den = math.sqrt(float((f * f).sum() * (a * a).sum()))
    return {"rho1": rho1,
            "tracker_corr": float((f * a).sum() / den) if den > 0 else math.nan,
            "mean_depth": float(depths.mean()),
            "mean_acc": float(accs.mean()),
            "censored": float(np.mean(accs >= depths)),
            "rounds": int(accs.size)}


def observed(arm: str):
    out = {}
    for name, leg in T.corpus(arm).items():
        rounds = leg["rounds"]
        depths = np.array([r["depth"] for r in rounds], dtype=float)
        accs = np.array([r["accepted"] for r in rounds], dtype=float)
        forecasts = np.array(
            [S.ema_expected_accept(r["ema"], r["depth"]) for r in rounds],
            dtype=float)
        out[name] = stats(depths, accs, forecasts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2031)
    ap.add_argument("--json", default=str(ARTIFACTS / "censoring-control.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    p7 = T.corpus("p7")
    obs = observed("adapt")
    out = {"draws": args.draws, "seed": args.seed, "prompts": {}}

    print("=" * 78)
    print("E203 POSITIVE CONTROL  does the shipped loop manufacture structure?")
    print("=" * 78)
    print("  input: fixed-depth-7 run lengths of the SAME prompt, permuted, so")
    print("         the input carries no serial structure at all")
    print("  loop:  replayed costModelDepth + recordAcceptOutcome (E128)")
    print("  draws: %d permutations per prompt" % args.draws)
    print()
    print("  rho1: lag-1 autocorrelation of accepted length.  carry: the run")
    print("  remainder survives a censored round, which is what the token")
    print("  stream does; discard: it does not, the conservative reading.")
    print("  %-16s %6s | %8s %9s %9s | %8s %9s %9s"
          % ("prompt", "trunc", "rho1_obs", "sim_disc", "sim_carry",
             "trk_obs", "sim_disc", "sim_carry"))

    keys = ("rho1_obs", "rho1_disc", "rho1_carry",
            "trk_obs", "trk_disc", "trk_carry")
    pooled = {k: [] for k in keys}
    pooled["w"] = []
    for name, leg in p7.items():
        rounds = [r for r in leg["rounds"] if r["depth"] == FIXED]
        runs = np.array([r["accepted"] for r in rounds], dtype=float)
        margins = np.array([r.get("m", 1.0) for r in rounds], dtype=float)
        trunc = float(np.mean(runs >= FIXED))
        modes = {}
        for tag, carry in (("disc", False), ("carry", True)):
            rows = []
            for _ in range(args.draws):
                idx = rng.permutation(runs.size)
                rows.append(stats(*simulate(runs[idx], margins[idx],
                                            carry=carry)))
            modes[tag] = {
                "rho1_mean": float(np.nanmean([r["rho1"] for r in rows])),
                "rho1_p2.5": float(np.nanpercentile(
                    [r["rho1"] for r in rows], 2.5)),
                "rho1_p97.5": float(np.nanpercentile(
                    [r["rho1"] for r in rows], 97.5)),
                "tracker_mean": float(np.nanmean(
                    [r["tracker_corr"] for r in rows])),
                "tracker_p97.5": float(np.nanpercentile(
                    [r["tracker_corr"] for r in rows], 97.5)),
                "mean_depth": float(np.mean([r["mean_depth"] for r in rows])),
                "mean_acc": float(np.mean([r["mean_acc"] for r in rows])),
                "censored": float(np.mean([r["censored"] for r in rows])),
            }
        o = obs[name]
        out["prompts"][name] = {"input_truncated_fraction": trunc,
                                "observed_adapt": o, "sim": modes}
        print("  %-16s %6.3f | %8.4f %9.4f %9.4f | %8.4f %9.4f %9.4f"
              % (name, trunc, o["rho1"], modes["disc"]["rho1_mean"],
                 modes["carry"]["rho1_mean"], o["tracker_corr"],
                 modes["disc"]["tracker_mean"],
                 modes["carry"]["tracker_mean"]))
        pooled["w"].append(float(o["rounds"]))
        pooled["rho1_obs"].append(o["rho1"])
        pooled["rho1_disc"].append(modes["disc"]["rho1_mean"])
        pooled["rho1_carry"].append(modes["carry"]["rho1_mean"])
        pooled["trk_obs"].append(o["tracker_corr"])
        pooled["trk_disc"].append(modes["disc"]["tracker_mean"])
        pooled["trk_carry"].append(modes["carry"]["tracker_mean"])

    w = np.array(pooled["w"])
    agg = {k: float(np.dot(w, pooled[k]) / w.sum()) for k in keys}
    out["pooled"] = agg
    print()
    print("  POOLED (weighted by rounds).  The simulated input is EXCHANGEABLE")
    print("  by construction, so anything the simulated columns show is")
    print("  manufactured by the rule's own depth feedback and censoring.")
    print("   %-14s %10s %10s %10s" % ("statistic", "shipped", "sim discard",
                                       "sim carry"))
    print("   %-14s %10.4f %10.4f %10.4f"
          % ("lag-1 acf", agg["rho1_obs"], agg["rho1_disc"],
             agg["rho1_carry"]))
    print("   %-14s %10.4f %10.4f %10.4f"
          % ("tracker corr", agg["trk_obs"], agg["trk_disc"],
             agg["trk_carry"]))
    print()
    print("  FIDELITY of the simulated leg against the real adaptive leg")
    print("   %-16s %9s %9s | %8s %8s | %8s %8s"
          % ("prompt", "depth_obs", "depth_sim", "acc_obs", "acc_sim",
             "cens_obs", "cens_sim"))
    for name, row in out["prompts"].items():
        o = row["observed_adapt"]
        s = row["sim"]["carry"]
        print("   %-16s %9.3f %9.3f | %8.3f %8.3f | %8.3f %8.3f"
              % (name, o["mean_depth"], s["mean_depth"], o["mean_acc"],
                 s["mean_acc"], o["censored"], s["censored"]))
    print()

    ARTIFACTS.mkdir(exist_ok=True)
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
    print("  wrote %s" % args.json)


if __name__ == "__main__":
    main()
