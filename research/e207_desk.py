#!/usr/bin/env python3
"""E207 stage 0 desk: what does the frozen-prior schedule do, and what is it
worth?

    usage: python3 research/e207_desk.py [--json OUT]

THE ARM. `DARKBLOOM_E207_EMA_ARM=frozen` removes the EMA UPDATE step. The
price rule, the prior shape, the 0.95 optimism cap and the `pendingTop2`
margin clamps are unchanged, so the frozen schedule is the shipped walk driven
by the seed prior `0.85 * 0.98^d` for every round of the request.

TWO HARNESSES, KEPT APART.

  harness=local   the E168 `p7` corpus. Every round of those legs drafts a
                  pinned depth 7, so the accepted run length is observed
                  OPEN-LOOP: it is not censored by a schedule decision that
                  the replayed policy would have made differently. Nine local
                  prompts, 512 decode tokens each. The local round-cost table
                  (`research/out/e168/round_cost.json`) prices a round at
                  verified width m. That table was measured on the E168 tree,
                  so every local millisecond here is DATED and is a prediction
                  of the SIGN and rough size of the stage-1 contrast, not a
                  substitute for it.

  harness=ranked  the FINDING 520 survival-pinned latent-q instrument, taken
                  from `e201_online_cap` / `e203_stage0b` without modification,
                  and its published-median assembly.

THE REPLAY IS OPEN-LOOP AND EXCHANGEABLE. A policy that drafts fewer tokens
per round needs more rounds to emit 512 tokens, so round r of the corpus is
not round r of the counterfactual leg. E203 stage 0a measured the accepted
length sequence to be exchangeable within a prompt (pooled rho1 = +0.0014,
null sigma 0.0239), so a per-round (margin, run) pair may be treated as an
exchangeable draw and the arm value read as a ratio of means over the pool.
The permutation spread over that pool is reported beside every local number.

CENSORING. A pinned round that accepts all 7 drafts observes only `k >= 7`.
At cap 7 no policy drafts deeper than 7, so `accepted = min(k, 7)` is exact
and the censoring is harmless. At cap 8 the live arm is UNDER-credited,
because a run that would have reached 8 is truncated at 7; the cap-8 numbers
are therefore conservative FOR the frozen arm.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e203_traces as T  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e207-artifacts"

# Shipped constants, Qwen36MTPBlockSession.swift.
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
PRIOR = [0.85 * 0.98 ** d for d in range(MAX_DEPTH)]
ALPHA = 0.15
OPTIMISM_CAP = 0.95
CLAMP_T = {0: 2.0, 1: 3.0}

PRICE_MARGINAL = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
PRICE_CUMULATIVE = [1.0 + d * HEAD_STEP_COST_RATIO for d in range(MAX_DEPTH + 1)]


# ------------------------------------------------------------ shipped walk

def depth_walk(ema, margin, cap):
    """Port of `costModelDepth`, including the depth 0/1 margin clamps."""
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if depth in CLAMP_T and margin is not None and not math.isnan(margin):
            conf = 1.0 / (1.0 + math.exp(-margin / CLAMP_T[depth]))
            p = min(p, conf)
        reach *= p
        threshold = PRICE_MARGINAL[depth] * (1.0 + expected) \
            / PRICE_CUMULATIVE[depth]
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def update_ema(ema, accepted, drafted):
    """Port of `recordAcceptOutcome` (the step the frozen arm removes).

    The stop-token early-exit branch is not modelled: the corpus does not name
    the stop token of a round, and a committed stop inside the accepted prefix
    is rare on 512-token prose legs.
    """
    for i in range(min(accepted, len(ema))):
        ema[i] += ALPHA * (1.0 - ema[i])
    if accepted < drafted and accepted < len(ema):
        ema[accepted] += ALPHA * (0.0 - ema[accepted])
    elif accepted == drafted and drafted > 0 and accepted < len(ema):
        if ema[accepted] < OPTIMISM_CAP:
            ema[accepted] += ALPHA * (OPTIMISM_CAP - ema[accepted])


# -------------------------------------------------- walk positive control

def walk_control():
    """Reproduce the depth every recorded round actually chose.

    Each `adapt` and `p7` round prints the EMA vector, the margin and the cap
    it used, so the port above must return the depth the round took. Rounds
    whose offered depth was narrowed by the token budget at the tail print a
    `cap` that already carries the clamp, so no round needs special handling.
    A pinned `p7` round replaces the schedule with a fixed depth, so only its
    `sched=` walk is comparable, not its `d=`.
    """
    checked = mismatched = 0
    examples = []
    for run_dir in sorted((T.E168 / "adapt").iterdir()):
        if not (run_dir / "trace.txt").exists():
            continue
        for record in T.read_leg(run_dir):
            if "ema" not in record or "cap" not in record:
                continue
            margin = record.get("m")
            margin = None if isinstance(margin, str) else margin
            got = depth_walk(list(record["ema"]), margin, int(record["cap"]))
            checked += 1
            if got != record["depth"]:
                mismatched += 1
                if len(examples) < 5:
                    examples.append({"leg": run_dir.name,
                                     "round": record["round"],
                                     "recorded": record["depth"], "port": got,
                                     "margin": margin, "cap": record["cap"]})
    return {"rounds_checked": checked, "mismatches": mismatched,
            "examples": examples}


# ------------------------------------------------------------ local corpus

def corpus(prompt):
    """Open-loop (margin, run, censored) pairs from one pinned p7 leg."""
    pairs = []
    for record in T.read_leg(T.E168 / "p7" / prompt):
        if record["depth"] != 7:
            continue                      # tail-narrowed or non-pinned round
        margin = record.get("m")
        margin = None if isinstance(margin, str) else margin
        pairs.append({"m": margin, "k": record["accepted"],
                      "censored": record["accepted"] == 7})
    return pairs


def local_cost_table():
    """Round cost in ms at verified width m, from the E168 legs.

    `round_cost.json` starts at m = 2, and the E168 legs never ran a serial
    phase, so no round of that corpus verifies a single row. m = 1 is INFERRED
    by continuing the table's own low-width slope, `2 * R(2) - R(3)`. Only the
    live arm ever reaches depth 0, and only on the few rounds where the margin
    clamp collapses the first position, so this inferred cell carries almost no
    weight; it is marked inferred in the artifact.
    """
    table = json.loads(
        (T.E168.parent.parent / "out" / "e168" / "round_cost.json").read_text()
    )["table"]
    cost = {row["m"]: row["median_ms"] for row in table}
    cost[1] = 2.0 * cost[2] - cost[3]
    return cost


COST_MS = local_cost_table()

# Seed prologue of one local leg, measured on the CURRENT tree by the E207
# 64-token exactness screen (`mtp-trace: begin ... wall_us=3922437`). Both arms
# pay it identically, so it only dilutes a decode-side effect at the leg
# endpoint, which is where stage 1 reads its statistic.
SEED_MS = 3922.437
LEG_TOKENS = 512

# The same screen measured the current tree at the two widths the frozen arm
# uses, so the dated table can be checked where the frozen arm actually lives.
FRESH_SCREEN_MS = {4: 75.91, 5: 89.50}


def round_ms(depth):
    return COST_MS[depth + 1]


def leg_ms_per_token(result, tokens=LEG_TOKENS):
    """Leg-endpoint seconds per token, the stage-1 statistic (RULE 394)."""
    rounds = tokens / result["tokens_per_round"]
    return (SEED_MS + rounds * result["ms_per_round"]) / tokens


# ------------------------------------------------------------- policy arms

def run_policy(pairs, arm, cap, order=None):
    """One pass of one policy over one prompt's exchangeable round pool."""
    ema = list(PRIOR)
    idx = order if order is not None else range(len(pairs))
    depths, accepted_counts, cost = [], [], 0.0
    for i in idx:
        pair = pairs[i]
        if arm == "live":
            depth = depth_walk(ema, pair["m"], cap)
        elif arm == "frozen":
            depth = depth_walk(PRIOR, pair["m"], cap)
        elif arm == "frozen_nomargin":
            depth = depth_walk(PRIOR, None, cap)
        elif arm == "live_nomargin":
            depth = depth_walk(ema, None, cap)
        elif isinstance(arm, int):
            depth = min(arm, cap)
        else:
            raise ValueError(arm)
        accepted = min(pair["k"], depth)
        depths.append(depth)
        accepted_counts.append(accepted)
        cost += round_ms(depth)
        if arm in ("live", "live_nomargin"):
            update_ema(ema, accepted, depth)
    rounds = len(depths)
    tokens = rounds + sum(accepted_counts)
    return {"rounds": rounds, "mean_depth": sum(depths) / rounds,
            "mean_accepted": sum(accepted_counts) / rounds,
            "tokens_per_round": tokens / rounds,
            "ms_per_token": cost / tokens,
            "ms_per_round": cost / rounds,
            "depths": depths, "final_ema": ema}


def permutation_spread(pairs, arm, cap, draws=200, seed=207):
    """Spread of the arm value over random orders of the exchangeable pool.

    Only the live arm depends on the order at all; the frozen and constant
    arms are order-invariant and must return a zero spread, which is the
    instrument's own null control.
    """
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    values = []
    for _ in range(draws):
        rng.shuffle(order)
        values.append(run_policy(pairs, arm, cap, list(order))["ms_per_token"])
    return {"mean": statistics.fmean(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0}


# ------------------------------------------------------------------ ranked

def ranked_projection(frozen_depth_mix):
    """Published median of the frozen policy on the ranked instrument.

    Imports the FINDING 520 instrument unchanged. Two treatments:

      const_d      the frozen walk with no margin binding: a single depth for
                   every round of every prompt.
      margin_mix   the depth DISTRIBUTION the frozen walk takes on the local
                   corpus, applied to the ranked rounds independently of the
                   round's latent q. The independence is an assumption, and it
                   is the conservative direction for a rule whose only
                   round-level input is the margin.
    """
    import e177_ranked_depth_law as e177
    import e197_refit as E
    import e200_desk_price as D
    import e201_online_cap as O
    import e203_stage0b as S

    inst = O.build_instrument()
    laws, _ratio9 = O.cost_laws(inst["fit"])
    grid = O.cap_grid(inst, laws)
    law = laws["smooth"]

    out = {"harness": "ranked", "receipt_A": E.BEST_A, "crown": E.CROWN}
    const = {name: {d: S.assemble(inst, law, name, S.items_const(inst, name, d))
                    for d in range(S.MAXD + 1)} for name in S.ORDER}
    out["const_table"] = {
        d: e177.published_median([const[n][d]["raw"] for n in S.ORDER])
        for d in range(S.MAXD + 1)}
    out["ship_cap7"] = e177.published_median(
        [grid["smooth"][n][7]["raw"] for n in S.ORDER])
    out["ship_cap8"] = e177.published_median(
        [grid["smooth"][n][8]["raw"] for n in S.ORDER])

    mix_raw = {}
    for name in S.ORDER:
        items = []
        for depth, weight in frozen_depth_mix.items():
            if weight <= 0.0:
                continue
            for w, _q, p, _d in S.nodes(inst, name):
                items.append((w * weight, depth, D.geom(p, depth)))
        mix_raw[name] = S.assemble(inst, law, name, items)["raw"]
    out["frozen_margin_mix"] = e177.published_median(
        [mix_raw[n] for n in S.ORDER])
    out["frozen_margin_mix_per_prompt"] = mix_raw
    out["frozen_depth_mix"] = {str(k): v for k, v in frozen_depth_mix.items()}
    return out


# ------------------------------------------------------------------- report

def pct(x, base):
    return 100.0 * (x / base - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=7)
    ap.add_argument("--json", default=str(ARTIFACTS / "stage0-desk.json"))
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    print("=" * 78)
    print("E207 STAGE 0 DESK  EMA SUBTRACTION  (frozen prior vs live update)")
    print("=" * 78)

    control = walk_control()
    print("\nWALK PORT POSITIVE CONTROL (recorded depth vs replayed depth)")
    print("  rounds checked %d, mismatches %d"
          % (control["rounds_checked"], control["mismatches"]))
    for ex in control["examples"]:
        print("   %s round %d recorded %d port %d margin %s cap %s"
              % (ex["leg"], ex["round"], ex["recorded"], ex["port"],
                 ex["margin"], ex["cap"]))

    print("\nFROZEN WALK, CLOSED FORM (no margin binding)")
    for cap in (7, 8):
        print("  cap %d -> depth %d" % (cap, depth_walk(PRIOR, None, cap)))
    print("  frozen depth by margin, cap %d:" % args.cap)
    row = []
    for margin in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        row.append("m=%.0f:d=%d"
                   % (margin, depth_walk(PRIOR, margin, args.cap)))
    print("   " + "  ".join(row))

    prompts = sorted(p.name for p in (T.E168 / "p7").iterdir()
                     if (p / "trace.txt").exists())
    # The local cost table starts at m = 2, so a non-drafting round (depth 0,
    # m = 1) has no measured price on this tree and the const arms start at 1.
    arms = ["live", "frozen", "frozen_nomargin", "live_nomargin"] \
        + list(range(1, 8))
    local = {}
    print("\nLOCAL OPEN-LOOP REPLAY, cap %d  (harness=local, DATED cost table)"
          % args.cap)
    print("  %-16s %6s %7s %7s %9s %9s  %s"
          % ("prompt", "rounds", "d_live", "d_froz", "ms/tok_L",
             "ms/tok_F", "frozen-live"))
    for prompt in prompts:
        pairs = corpus(prompt)
        if not pairs:
            continue
        result = {}
        for arm in arms:
            key = arm if isinstance(arm, str) else "const%d" % arm
            value = run_policy(pairs, arm, args.cap)
            value.pop("depths")
            value.pop("final_ema")
            result[key] = value
        result["live_permutation"] = permutation_spread(pairs, "live",
                                                        args.cap)
        result["frozen_permutation"] = permutation_spread(pairs, "frozen",
                                                          args.cap)
        result["n_pairs"] = len(pairs)
        result["censored_fraction"] = sum(
            1 for p in pairs if p["censored"]) / len(pairs)
        local[prompt] = result
        print("  %-16s %6d %7.3f %7.3f %9.3f %9.3f  %+7.2f%%"
              % (prompt, result["live"]["rounds"],
                 result["live"]["mean_depth"], result["frozen"]["mean_depth"],
                 result["live"]["ms_per_token"],
                 result["frozen"]["ms_per_token"],
                 pct(result["frozen"]["ms_per_token"],
                     result["live"]["ms_per_token"])))

    print("\nCHANNEL DECOMPOSITION, ms/token, cap %d (harness=local)"
          % args.cap)
    print("  %-16s %9s %9s %9s %9s"
          % ("prompt", "live", "live-noM", "frozen", "froz-noM"))
    for prompt, result in local.items():
        print("  %-16s %9.3f %9.3f %9.3f %9.3f"
              % (prompt, result["live"]["ms_per_token"],
                 result["live_nomargin"]["ms_per_token"],
                 result["frozen"]["ms_per_token"],
                 result["frozen_nomargin"]["ms_per_token"]))

    print("\nDATED COST TABLE vs THE FRESH E207 SCREEN (current tree)")
    for m, fresh in sorted(FRESH_SCREEN_MS.items()):
        print("  m=%d dated %.2f ms, screen %.2f ms (%+0.2f%%)"
              % (m, COST_MS[m], fresh, pct(fresh, COST_MS[m])))
    print("  m=6,7,8 have no current-tree measurement: the dated table holds")
    print("  10, 6 and 120 rounds there and predates E192/E193/E195.")

    bench = local.get("benchfixture")
    prereg = {}
    if bench:
        live_leg = leg_ms_per_token(bench["live"])
        frozen_leg = leg_ms_per_token(bench["frozen"])
        print("\n  STAGE-1 FIXTURE (benchfixture is the --local-iterate prompt)")
        print("   live   depth %.3f accepted %.3f ms/round %.2f decode "
              "ms/token %.3f"
              % (bench["live"]["mean_depth"], bench["live"]["mean_accepted"],
                 bench["live"]["ms_per_round"], bench["live"]["ms_per_token"]))
        print("   frozen depth %.3f accepted %.3f ms/round %.2f decode "
              "ms/token %.3f"
              % (bench["frozen"]["mean_depth"],
                 bench["frozen"]["mean_accepted"],
                 bench["frozen"]["ms_per_round"],
                 bench["frozen"]["ms_per_token"]))
        print("   decode-only delta            %+0.2f%%"
              % pct(bench["frozen"]["ms_per_token"],
                    bench["live"]["ms_per_token"]))
        print("   leg endpoint at %d tokens    live %.3f frozen %.3f ms/token"
              % (LEG_TOKENS, live_leg, frozen_leg))
        print("   PREDICTED STAGE-1 DELTA      %+0.2f%% (seed prologue %.0f ms"
              " included in both arms)" % (pct(frozen_leg, live_leg), SEED_MS))
        print("   permutation sd: live %.4f frozen %.4f decode ms/token"
              % (bench["live_permutation"]["sd"],
                 bench["frozen_permutation"]["sd"]))

        # The live arm spends most of its benchfixture rounds at widths the
        # current tree has never been measured at, so the sign of the local
        # prediction is a statement about those cells. Scale them until the
        # two arms tie.
        pairs = corpus("benchfixture")
        lo, hi = 0.3, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            saved = {m: COST_MS[m] for m in (6, 7, 8)}
            for m in (6, 7, 8):
                COST_MS[m] = saved[m] * mid
            value = leg_ms_per_token(run_policy(pairs, "live", args.cap))
            for m in (6, 7, 8):
                COST_MS[m] = saved[m]
            if value > frozen_leg:
                hi = mid
            else:
                lo = mid
        tie = 0.5 * (lo + hi)
        print("   SENSITIVITY: the two arms tie when the m=6,7,8 cells of the"
              " dated table are scaled by %.3f" % tie)

        # E195 measured the merged QMV width plan at m=6 (118.73 vs 126.04
        # ms/round, -5.8%) and found m=7 inside its own floor, so the wide
        # cells of the dated table are the ones most likely to be stale. The
        # tie point above is inside that range, so the local sign is scanned
        # rather than asserted.
        widths = {}
        for depth in run_policy(pairs, "live", args.cap)["depths"]:
            widths[depth + 1] = widths.get(depth + 1, 0) + 1
        print("   live width histogram on benchfixture: "
              + " ".join("m=%d:%d" % (m, n) for m, n in sorted(widths.items())))
        scan = {}
        for scale in (1.00, 0.97, 0.955, 0.94, 0.90):
            saved = {m: COST_MS[m] for m in (6, 7, 8)}
            for m in (6, 7, 8):
                COST_MS[m] = saved[m] * scale
            value = leg_ms_per_token(run_policy(pairs, "live", args.cap))
            for m in (6, 7, 8):
                COST_MS[m] = saved[m]
            scan["%.3f" % scale] = pct(frozen_leg, value)
            print("   wide-cell scale %.3f -> predicted stage-1 delta %+0.2f%%"
                  % (scale, pct(frozen_leg, value)))
        prereg = {"benchfixture_live_leg_ms_per_token": live_leg,
                  "benchfixture_frozen_leg_ms_per_token": frozen_leg,
                  "predicted_stage1_delta_pct": pct(frozen_leg, live_leg),
                  "wide_cell_tie_scale": tie,
                  "wide_cell_scan_delta_pct": scan,
                  "live_width_histogram": widths,
                  "sign": "frozen faster" if frozen_leg < live_leg
                          else "frozen slower",
                  "sign_robust_to_wide_cell_staleness": tie < 0.90}

    # The frozen depth distribution the ranked projection needs. It is a
    # property of the margin sequence only, so it is pooled over the prompts.
    mix = {}
    total = 0
    for prompt in prompts:
        for pair in corpus(prompt):
            depth = depth_walk(PRIOR, pair["m"], args.cap)
            mix[depth] = mix.get(depth, 0) + 1
            total += 1
    mix = {d: n / total for d, n in sorted(mix.items())}
    print("\nFROZEN DEPTH DISTRIBUTION over the pooled local margins")
    print("  " + "  ".join("d=%d:%.3f" % (d, w) for d, w in mix.items()))

    print("\nRANKED PROJECTION (harness=ranked)")
    ranked = ranked_projection(mix)
    const_at = ranked["const_table"][depth_walk(PRIOR, None, args.cap)]
    print("  receipt A (paid cap 7)          %.6f" % ranked["receipt_A"])
    print("  instrument shipped cap 7        %.6f" % ranked["ship_cap7"])
    print("  instrument shipped cap 8        %.6f" % ranked["ship_cap8"])
    print("  constant depth %d (frozen, no margin) %.6f  (%+0.2f%% vs A)"
          % (depth_walk(PRIOR, None, args.cap), const_at,
             pct(const_at, ranked["receipt_A"])))
    print("  frozen with margin mix          %.6f  (%+0.2f%% vs A, "
          "%+0.2f%% vs cap 8)"
          % (ranked["frozen_margin_mix"],
             pct(ranked["frozen_margin_mix"], ranked["receipt_A"]),
             pct(ranked["frozen_margin_mix"], ranked["ship_cap8"])))
    print("  best constant depth             %.6f at d=%d"
          % (max(ranked["const_table"].values()),
             max(ranked["const_table"], key=ranked["const_table"].get)))

    hi = const_at
    lo = ranked["frozen_margin_mix"]
    print("\nPRE-REGISTERED RANKED BRACKET (harness=ranked)")
    print("  upper end %.6f (%+0.2f%% vs A): the margin never binds on a"
          % (hi, pct(hi, ranked["receipt_A"])))
    print("            hidden prompt, so the frozen rule is constant depth %d."
          % depth_walk(PRIOR, None, args.cap))
    print("  lower end %.6f (%+0.2f%% vs A): hidden margins are distributed"
          % (lo, pct(lo, ranked["receipt_A"])))
    print("            like the pooled local margins.")
    print("  both ends are NEGATIVE, so the pre-registered ranked sign is:")
    print("  FROZEN LOSES on the published median.")
    ranked["bracket_pct_vs_A"] = [pct(lo, ranked["receipt_A"]),
                                  pct(hi, ranked["receipt_A"])]
    ranked["bracket_pct_vs_cap8"] = [pct(lo, ranked["ship_cap8"]),
                                     pct(hi, ranked["ship_cap8"])]
    ranked["predicted_sign"] = "frozen slower on the published median"

    payload = {
        "prereg_local_stage1": prereg,
        "harness_local": {"cap": args.cap, "cost_table_ms": COST_MS,
                          "cost_table_m1_inferred": True,
                          "cost_table_source": "research/out/e168/"
                                               "round_cost.json (E168 tree, "
                                               "DATED)",
                          "prompts": local},
        "harness_ranked": ranked,
        "walk_control": control,
        "frozen_depth_closed_form": {str(cap): depth_walk(PRIOR, None, cap)
                                     for cap in (7, 8)},
        "frozen_depth_mix_local_margins": {str(d): w for d, w in mix.items()},
        "prior": PRIOR,
    }
    pathlib.Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
