#!/usr/bin/env python3
"""E79 -- price the proposal head.

Four questions, one module:

  chainfit  What per-position acceptance chain reproduces the eight ranked
            aggregates of ledger 207(A)? Two estimators: a closed-form
            mean-width inversion and a simulation of the SHIPPED scheduler,
            which is over-identified because it must reproduce both the
            proposed width and the accepted count.
  reprice   What does the schedule do, and what does decode cost, when the
            head step is free or half price? Uses the E68/216 crown-table
            ladder plus a head term.
  census    Per-position acceptance and converged EMAs from a phase trace.
  score     Ranked score under a counterfactual acceptance chain, and the
            round-level and score-level value of a cheaper head step.

Everything here is research-only. Nothing in this file is on the submitted
surface.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- constants

MAX_DEPTH = 8
SDPA_WIDTH_WALL_DEPTH_CAP = 5
SEGMENTED_VERIFY_DEPTH_CAP = 8
SEGMENTED_STREAK_GATE = 2
ACCEPT_EMA_ALPHA = 0.15
SHIPPED_EMA_SEED = [0.85 * 0.98 ** i for i in range(MAX_DEPTH)]
SHIPPED_H = 0.18

# E68 rung-1 isolated single-group QMV cost, ms, refit in ledger 216.
LADDER_S = {1: 60.372, 2: 65.377, 3: 72.128, 4: 82.163, 5: 95.568,
            6: 122.876, 7: 149.368, 8: 183.642, 9: 451.747}
LADDER_L = 15.191
# Our shipped dispatch table, ledger 216(C): one group up to width 8, [5,4] at 9.
OUR_PARTITION = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7],
                 8: [8], 9: [5, 4]}

# Ledger 207(A) ranked receipt integers for `ca9251b8`, plus 211(A) raw ratios.
RANKED = [
    # prompt, rounds, proposed, accepted, round_ms, raw
    ("plutarch", 487, 75, 25, 30.781, 1.2528),
    ("drama", 252, 579, 260, 38.094, 1.9163),
    ("travel", 212, 563, 300, 39.511, 2.1795),
    ("beagle", 107, 485, 405, 53.338, 3.1201),
    ("medicine", 99, 472, 413, 53.476, 3.3446),
    ("republic", 89, 469, 423, 58.421, 3.3930),
    ("essays", 87, 472, 425, 60.196, 3.3666),
    ("botany", 85, 491, 427, 60.548, 3.4253),
]
SEED_PROLOGUE_MS = 525.963  # charged inside the timed leg, outside the rounds

PUBLIC_GOLDEN = "correctness_prompts/public_longcopy_gate_english_512_1024.json"
# Qwen35TextModel compact draft vocabulary: rows 0 ..< 98,304 plus the 26
# official text/control tokens 248,044 ..< 248,070, padded to 98,336.
COMPACT_DRAFT_ROWS = 98_336
CONTROL_START = 248_044
CONTROL_END = 248_070

# Weight bytes ONE head step reads, from the safetensors headers of the two
# provisioned heads. Decode at batch 1 is memory-bandwidth-bound, so these
# bytes are the head step's cost driver.
#   declared  hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1  427,738,112 B
#     stage (i)   transformer block, affine-4 g64 + bf16 qkv islands
#     stage (ii)  draft_lm_head, affine-2 g64 over 98,336 rows
#     stage (iii) top-32 over 98,330 coarse logits
#     stage (iv)  32 gathered affine-4 rows of the compact target lm_head
COARSE_READOUT_BYTES = 157_337_600          # stage (ii)
DECLARED_HEAD_BLOCK_BYTES = 270_400_512     # stage (i)
DECLARED_RERANK_BYTES = 32 * (640 * 4 + 80 * 2 + 80 * 2)  # stage (iv)
DECLARED_TOP32_BYTES = 2 * 98_330 * 4                     # stage (iii)
DECLARED_HEAD_STEP_BYTES = (DECLARED_HEAD_BLOCK_BYTES + COARSE_READOUT_BYTES
                            + DECLARED_TOP32_BYTES + DECLARED_RERANK_BYTES)
# pinned  EigenLabs/Qwen3.8-27B-MTP-bf16: no draft_lm_head, so the proposal
# reads the affine-4 g64 compact slice of the target lm_head instead.
PINNED_HEAD_BLOCK_BYTES = 849_398_784
PINNED_SELECT_BYTES = 98_336 * 640 * 4 + 2 * 98_336 * 80 * 2
PINNED_HEAD_STEP_BYTES = (PINNED_HEAD_BLOCK_BYTES + PINNED_SELECT_BYTES
                          + DECLARED_TOP32_BYTES)
# floor(M) = 30.402 + (M-1) * 8.42/8, ledger 211(A).
RANKED_DEPTH0_ROUND_MS = 30.402
RANKED_HEAD_STEP_MS = 8.42 / 8

# `Qwen36MTPBlockSession.swift:919-930`: two official ranked runs that differ
# only in `headStepCostRatio`, with per-prompt mean DRAFTS per round for five
# wide prompts and one hard prompt. Paired in the order the comment writes
# them. Two settings per prompt is what makes the acceptance chain
# identifiable, because one setting alone cannot separate acceptance from
# price.
H_SWEEP_H = (0.18, 0.32)
H_SWEEP_WIDE = [(4.35, 3.36), (4.89, 4.01), (5.78, 4.53), (5.33, 4.03),
                (5.04, 4.76)]
H_SWEEP_HARD = (0.17, 0.06)
# Arm 3, submission `2da69933`: the E68 `pbfit` depth-price vector as executed.
PBFIT_MARGINAL = [0.12014, 0.13337, 0.15825, 0.18378, 0.28911, 0.19918,
                  0.16198, 0.19419]
ARM3 = {"published_median": 3.21126, "baseline_published_median": 3.23251,
        "fourth_sorted": 3.08697, "fifth_sorted": 3.33554,
        "min_raw": 1.95855, "candidate_spt_delta_pct": -11.26}


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"slope": slope, "intercept": intercept,
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def ranked_round_cost_model():
    """Least-squares `round_ms = fixed + slope * drafts` over the eight ranked
    prompts of ledger 207(A). The slope is what one more DRAFT really costs on
    the ranked M5, and `slope / fixed` is the total marginal ratio the greedy
    rate rule in `costModelDepth` actually needs."""
    xs = [prop / R for _, R, prop, _, _, _ in RANKED]
    ys = [ms for *_, ms, _ in RANKED]
    fit = fit_line(xs, ys)
    fit["marginal_ratio"] = fit["slope"] / fit["intercept"]
    return fit


def ladder_cost(width: int) -> float:
    groups = OUR_PARTITION[width]
    return sum(LADDER_S[g] for g in groups) - LADDER_L * (len(groups) - 1)


def round_cost_ms(width: int, head_step_ms: float) -> float:
    """Modelled local round cost at verify width `width`."""
    return ladder_cost(width) + head_step_ms * (width - 1)


# ------------------------------------------------------- shipped scheduler

class Scheduler:
    """The shipped cost-model schedule, transcribed from
    `Qwen36MTPBlockSession.costModelDepth` and `recordAcceptOutcome`."""

    def __init__(self, marginal=None, ema_seed=None, alpha=ACCEPT_EMA_ALPHA,
                 streak_gate=SEGMENTED_STREAK_GATE,
                 narrow_cap=SDPA_WIDTH_WALL_DEPTH_CAP,
                 wide_cap=SEGMENTED_VERIFY_DEPTH_CAP):
        self.marginal = list(marginal if marginal is not None
                             else [SHIPPED_H] * MAX_DEPTH)
        self.cumulative = [1.0]
        for m in self.marginal:
            self.cumulative.append(self.cumulative[-1] + m)
        self.ema = list(ema_seed if ema_seed is not None else SHIPPED_EMA_SEED)
        self.alpha = alpha
        self.streak_gate = streak_gate
        self.narrow_cap = narrow_cap
        self.wide_cap = wide_cap
        self.streak = 0

    def depth(self, offered=MAX_DEPTH, margin=None):
        width_cap = (self.wide_cap if self.streak >= self.streak_gate
                     else self.narrow_cap)
        cap = min(min(offered, MAX_DEPTH), width_cap)
        if cap <= 0:
            return 0
        reach, expected, depth = 1.0, 0.0, 0
        while depth < cap:
            p = self.ema[depth]
            if margin is not None and depth == 0:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
            elif margin is not None and depth == 1:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
            reach *= p
            threshold = (self.marginal[depth] * (1.0 + expected)
                         / self.cumulative[depth])
            if not reach > threshold:
                break
            expected += reach
            depth += 1
        return depth

    def record(self, accepted, drafted):
        a = self.alpha
        for i in range(min(accepted, MAX_DEPTH)):
            self.ema[i] += a * (1.0 - self.ema[i])
        if accepted < drafted and accepted < MAX_DEPTH:
            self.ema[accepted] += a * (0.0 - self.ema[accepted])
        elif accepted == drafted and drafted > 0 and accepted < MAX_DEPTH:
            if self.ema[accepted] < 0.95:
                self.ema[accepted] += a * (0.95 - self.ema[accepted])
        self.streak = self.streak + 1 if accepted == drafted else 0


def simulate(p_vec, marginal=None, tokens=512, seed=0, offered=MAX_DEPTH,
             margins=None, head_step_ms=0.0, reps=1):
    """Run the shipped schedule against a Bernoulli chain with acceptance
    `p_vec[i]` at draft position i. Returns aggregate observables."""
    rng = random.Random(seed)
    out = []
    for rep in range(reps):
        sched = Scheduler(marginal=marginal)
        emitted = rounds = proposed = accepted_total = 0
        widths, cost_ms = Counter(), 0.0
        while emitted < tokens:
            margin = rng.choice(margins) if margins else None
            d = sched.depth(offered=offered, margin=margin)
            k = 0
            while k < d and rng.random() < p_vec[k]:
                k += 1
            sched.record(k, d)
            rounds += 1
            proposed += d
            accepted_total += k
            widths[d + 1] += 1
            cost_ms += round_cost_ms(d + 1, head_step_ms)
            emitted += 1 + k
        out.append({
            "rounds": rounds,
            "proposed": proposed,
            "accepted": accepted_total,
            "M": 1 + proposed / rounds,
            "tokens_per_round": 1 + accepted_total / rounds,
            "cost_ms": cost_ms,
            "ms_per_token": cost_ms / emitted,
            "emitted": emitted,
            "widths": widths,
            "final_ema": list(sched.ema),
        })
    if reps == 1:
        return out[0]
    agg = {k: st.mean(o[k] for o in out)
           for k in ("rounds", "proposed", "accepted", "M", "tokens_per_round",
                     "cost_ms", "ms_per_token", "emitted")}
    widths = Counter()
    for o in out:
        widths.update(o["widths"])
    agg["widths"] = Counter({w: n / reps for w, n in widths.items()})
    agg["final_ema"] = [st.mean(o["final_ema"][i] for i in range(MAX_DEPTH))
                        for i in range(MAX_DEPTH)]
    agg["ms_per_token_sd"] = st.pstdev([o["ms_per_token"] for o in out])
    agg["M_sd"] = st.pstdev([o["M"] for o in out])
    return agg


# ------------------------------------------------------------- chain fits

def expected_accepted(mean_width, p_vec):
    """Accepted drafts per round for a linear chain at a fractional width."""
    total, reach, whole = 0.0, 1.0, int(mean_width)
    for i in range(whole):
        reach *= p_vec[i]
        total += reach
    frac = mean_width - whole
    if frac > 0 and whole < len(p_vec):
        total += frac * reach * p_vec[whole]
    return total


def solve_constant_p(mean_width, mean_accepted):
    lo, hi = 1e-6, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_accepted(mean_width, [mid] * (MAX_DEPTH + 1)) < mean_accepted:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_geometric_p1(mean_width, mean_accepted, decay):
    lo, hi = 1e-6, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        vec = [min(1.0, mid * decay ** i) for i in range(MAX_DEPTH + 1)]
        if expected_accepted(mean_width, vec) < mean_accepted:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_geometric_by_simulation(target_M, target_tpr, reps=8, seed=11,
                                  margins=None, decays=None):
    """Two-parameter chain (p1, decay) matched to BOTH ranked moments.

    Eight aggregates cannot resolve eight positions, but each prompt supplies
    two independent numbers -- proposed width and accepted count -- so a
    two-parameter family is exactly identified per prompt. The schedule itself
    supplies the second equation: a decaying chain stops the EMA walk earlier
    at the same accepted count.
    """
    decays = decays if decays is not None else [
        0.80 + 0.01 * i for i in range(21)]
    best, feasible = None, []
    for decay in decays:
        lo, hi = 0.30, 0.9999
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            vec = [min(0.9999, mid * decay ** i) for i in range(MAX_DEPTH)]
            sim = simulate(vec, reps=reps, seed=seed, margins=margins)
            if sim["tokens_per_round"] < target_tpr:
                lo = mid
            else:
                hi = mid
        p1 = 0.5 * (lo + hi)
        vec = [min(0.9999, p1 * decay ** i) for i in range(MAX_DEPTH)]
        sim = simulate(vec, reps=reps, seed=seed, margins=margins)
        # A decay is INFEASIBLE when even p1 -> 1 cannot reach the receipt's
        # accepted count. That bound is the identifying content of the fit.
        if abs(sim["tokens_per_round"] - target_tpr) / target_tpr > 0.01:
            continue
        feasible.append(decay)
        err = abs(sim["M"] - target_M) / target_M
        if best is None or err < best[0]:
            best = (err, p1, decay, sim, vec)
    if best is None:
        return None
    return best + (min(feasible), max(feasible))


def solve_p_by_simulation(target_M, target_tpr, reps=12, seed=11,
                          margins=None):
    """Find the constant p whose SIMULATED shipped schedule matches the
    receipt's tokens-per-round. Reports the width residual, which is the
    over-identifying test the closed form cannot make."""
    lo, hi = 0.30, 0.999
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        sim = simulate([mid] * MAX_DEPTH, reps=reps, seed=seed, margins=margins)
        if sim["tokens_per_round"] < target_tpr:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    sim = simulate([p] * MAX_DEPTH, reps=reps, seed=seed, margins=margins)
    return p, sim


# ------------------------------------------------------------ trace census

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
FIELD_RE = re.compile(r"(\w+)_us=(\d+)")
SIGNAL_RE = re.compile(
    r"arm=(\w+) m=([-\d.naif]+) streak=(\d+) cap=(\d+) ema=([0-9.,]+)")


def parse_trace(path):
    """Every traced round, split into legs. A leg restarts at round=0."""
    legs, leg = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = ROUND_RE.search(line)
        if not m:
            continue
        idx, d, acc = (int(x) for x in m.groups())
        rec = {"idx": idx, "d": d, "acc": acc}
        rec.update({k: int(v) for k, v in FIELD_RE.findall(line)})
        s = SIGNAL_RE.search(line)
        if s:
            rec["arm"] = s.group(1)
            rec["margin"] = float(s.group(2)) if s.group(2)[0] != "n" else None
            rec["streak"] = int(s.group(3))
            rec["cap"] = int(s.group(4))
            rec["ema"] = [float(x) for x in s.group(5).split(",")]
        if idx == 0 and leg:
            legs.append(leg)
            leg = []
        leg.append(rec)
    if leg:
        legs.append(leg)
    return legs


PHASES = ("round", "draft_build", "d_pre", "d_flush", "d_head1", "d_submit1",
          "d_chain", "d_submit2", "verify_build", "eval_wall", "readout",
          "commit", "upkeep")


def slope_vs_draft(rounds, phase):
    """Least-squares `phase_ms = fixed + slope * draftCount` over the leg.

    The slope is the marginal cost of one more DRAFT at the widths the
    schedule actually visited; the intercept is the width-independent part.
    Reported with R^2 so a phase that does not scale with width is visible
    as such instead of being read as a per-draft price.
    """
    pts = [(r["d"], r[phase] / 1000.0) for r in rounds if phase in r]
    xs = {x for x, _ in pts}
    if len(pts) < 3 or len(xs) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    fixed = my - slope * mx
    sst = sum((y - my) ** 2 for _, y in pts)
    sse = sum((y - fixed - slope * x) ** 2 for x, y in pts)
    return {"slope_ms_per_draft": slope, "fixed_ms": fixed,
            "r2": 1.0 - sse / sst if sst > 0 else float("nan"), "n": n}


def position_table(rounds):
    """Per-position (reached, accepted, p, Wilson 95% interval).

    Position i is REACHED when a draft existed there (d > i) and every earlier
    draft was accepted (acc >= i); it SUCCEEDS when acc > i. Rounds past the
    first rejection never reach the position and contribute nothing, which is
    why `accepted/proposed` is a different and wrong estimator.
    """
    rows = []
    for i in range(MAX_DEPTH):
        reached = sum(1 for r in rounds if r["d"] > i and r["acc"] >= i)
        ok = sum(1 for r in rounds if r["d"] > i and r["acc"] > i)
        if not reached:
            continue
        p = ok / reached
        z = 1.959964
        den = 1 + z * z / reached
        centre = (p + z * z / (2 * reached)) / den
        half = z * math.sqrt(p * (1 - p) / reached
                             + z * z / (4 * reached * reached)) / den
        rows.append({"position": i + 1, "reached": reached, "accepted": ok,
                     "p": p, "lo": max(0.0, centre - half),
                     "hi": min(1.0, centre + half)})
    return rows


# ------------------------------------------------------------ ranked score

def ranked_leg_ms(rounds, round_ms):
    return rounds * round_ms + SEED_PROLOGUE_MS


def ranked_score_table(new_round_ms=None, new_tokens_per_round=None):
    """Published median under a counterfactual per-prompt round cost and
    tokens-per-round. `raw` scales as candidate seconds-per-token falls."""
    raws = []
    rows = []
    for name, R, prop, acc, ms, raw in RANKED:
        tpr = 1 + acc / R
        base_leg = ranked_leg_ms(R, ms)
        tpr2 = new_tokens_per_round(name, tpr) if new_tokens_per_round else tpr
        ms2 = new_round_ms(name, ms) if new_round_ms else ms
        R2 = 512.0 / tpr2
        leg2 = ranked_leg_ms(R2, ms2)
        raw2 = raw * base_leg / leg2
        rows.append((name, tpr, tpr2, ms, ms2, raw, raw2))
        raws.append(raw2)
    raws.sort()
    median = 0.5 * (raws[3] + raws[4])
    return rows, median


def baseline_median():
    _, m = ranked_score_table()
    return m


# ------------------------------------------------------------------- CLIs

def cmd_chainfit(args):
    report = {"model": "ledger-207 chain fit", "prompts": []}
    print("== closed-form chain inversion on the ledger 207(A) aggregates ==")
    print("  %-10s %6s %8s %8s %8s %8s %8s %8s" %
          ("prompt", "R", "M", "tok/rd", "const p", "geo p1", "geo p8",
           "shipped"))
    shipped_pred = {}
    for name, R, prop, acc, ms, raw in RANKED:
        mean_width = prop / R
        mean_acc = acc / R
        p_const = solve_constant_p(mean_width, mean_acc)
        p1 = solve_geometric_p1(mean_width, mean_acc, args.decay)
        p8 = p1 * args.decay ** 7
        shipped = expected_accepted(mean_width, SHIPPED_EMA_SEED + [0.0])
        shipped_pred[name] = shipped
        print("  %-10s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" %
              (name, R, 1 + mean_width, 1 + mean_acc, p_const, p1, p8,
               1 + shipped))
        report["prompts"].append({
            "prompt": name, "rounds": R, "M": 1 + mean_width,
            "tokens_per_round": 1 + mean_acc, "p_constant": p_const,
            "geometric_p1": p1, "geometric_p8": p8, "geometric_decay":
            args.decay, "shipped_prior_tokens_per_round": 1 + shipped})
    print("  `shipped` is the tokens/round the SHIPPED EMA SEED predicts at the")
    print("  same proposed width. The gap is the size of the prior's error.")

    print()
    print("== over-identified fit: simulate the shipped schedule ==")
    print("  matched on tokens/round; the M residual is the free test")
    print("  %-10s %8s %8s %8s %8s %9s" %
          ("prompt", "p_sim", "M_obs", "M_sim", "M_resid", "tpr_sim"))
    for name, R, prop, acc, ms, raw in RANKED:
        tpr = 1 + acc / R
        p, sim = solve_p_by_simulation(1 + prop / R, tpr, reps=args.reps)
        resid = 100 * (sim["M"] - (1 + prop / R)) / (1 + prop / R)
        print("  %-10s %8.4f %8.4f %8.4f %8.2f%% %9.4f" %
              (name, p, 1 + prop / R, sim["M"], resid,
               sim["tokens_per_round"]))
        for entry in report["prompts"]:
            if entry["prompt"] == name:
                entry.update(p_simulated=p, M_simulated=sim["M"],
                             M_residual_pct=resid,
                             tokens_per_round_simulated=sim["tokens_per_round"])

    print()
    print("== two-parameter fit: (p1, decay) matched to BOTH ranked moments ==")
    print("  a decay is FEASIBLE only when some p1 <= 1 reproduces the")
    print("  receipt's accepted count to 1%; the fastest feasible decay is")
    print("  the hard bound the eight aggregates place on the curve's shape")
    print("  %-10s %8s %8s %8s %8s %9s %14s" %
          ("prompt", "p1", "decay", "p8", "M_sim", "M_resid", "feasible decay"))
    for name, R, prop, acc, ms, raw in RANKED:
        found = solve_geometric_by_simulation(
            1 + prop / R, 1 + acc / R, reps=args.reps)
        if found is None:
            print("  %-10s   no (p1, decay) in the search grid reproduces the "
                  "receipt" % name)
            continue
        err, p1, decay, sim, vec, dmin, dmax = found
        print("  %-10s %8.4f %8.4f %8.4f %8.4f %8.2f%% %6.2f - %5.2f" %
              (name, p1, decay, vec[7], sim["M"], 100 * err, dmin, dmax))
        for entry in report["prompts"]:
            if entry["prompt"] == name:
                entry.update(fit2_p1=p1, fit2_decay=decay, fit2_vector=vec,
                             fit2_M=sim["M"], fit2_M_err_pct=100 * err,
                             fit2_tokens_per_round=sim["tokens_per_round"],
                             fit2_feasible_decay_min=dmin,
                             fit2_feasible_decay_max=dmax)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


def cmd_reprice(args):
    p_vec = [args.p] * MAX_DEPTH if args.p_vector is None else args.p_vector
    head = args.head_step_ms
    print("== modelled local round cost, ms ==")
    print("  %5s %10s %12s %12s %12s" %
          ("M", "ladder", f"+head {head:.3f}", "+head/2", "+head 0"))
    for w in range(1, 10):
        print("  %5d %10.3f %12.3f %12.3f %12.3f" %
              (w, ladder_cost(w), round_cost_ms(w, head),
               round_cost_ms(w, head / 2), round_cost_ms(w, 0.0)))

    arms = []
    for label, hcost in (("head as measured", head), ("head at half", head / 2),
                         ("head free", 0.0)):
        v = round_cost_ms(1, hcost)
        true_marginal = [(round_cost_ms(d + 2, hcost) - round_cost_ms(d + 1, hcost))
                         / v for d in range(MAX_DEPTH)]
        arms.append((f"{label}, shipped flat price", hcost, None))
        arms.append((f"{label}, true-cost price", hcost, true_marginal))

    print()
    print("== schedule and decode cost under each head price, p = %s ==" %
          (f"{args.p:.4f}" if args.p_vector is None else "measured vector"))
    print("  %-38s %8s %10s %12s %10s" %
          ("arm", "M", "tok/rd", "ms/token", "vs ship"))
    baseline = None
    rows = []
    for label, hcost, marginal in arms:
        sim = simulate(p_vec, marginal=marginal, reps=args.reps,
                       head_step_ms=hcost, seed=7)
        if baseline is None:
            baseline = sim["ms_per_token"]
        delta = 100 * (sim["ms_per_token"] - baseline) / baseline
        print("  %-38s %8.4f %10.4f %12.4f %9.2f%%" %
              (label, sim["M"], sim["tokens_per_round"], sim["ms_per_token"],
               delta))
        rows.append({"arm": label, "head_step_ms": hcost,
                     "marginal_price": marginal, "M": sim["M"],
                     "tokens_per_round": sim["tokens_per_round"],
                     "ms_per_token": sim["ms_per_token"],
                     "delta_pct_vs_ship": delta,
                     "widths": {str(k): v for k, v in
                                sorted(sim["widths"].items())}})
    print()
    print("== width histogram, share of rounds ==")
    widths = sorted({int(w) for r in rows for w in r["widths"]})
    print("  %-38s " % "arm" + " ".join("%7s" % f"M={w}" for w in widths))
    for r in rows:
        total = sum(r["widths"].values())
        print("  %-38s " % r["arm"] + " ".join(
            "%6.1f%%" % (100 * r["widths"].get(str(w), 0) / total)
            for w in widths))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"p": args.p, "p_vector": p_vec, "head_step_ms": head,
             "arms": rows}, indent=2))
    return rows


def cmd_census(args):
    legs = parse_trace(args.trace)
    drafting = [leg for leg in legs if any(r["d"] for r in leg)]
    print("== %s: %d leg(s), %d drafting leg(s) ==" %
          (args.trace, len(legs), len(drafting)))
    report = {"trace": str(args.trace), "legs": []}
    for n, leg in enumerate(legs):
        body = leg[args.warmup:]
        if not body:
            continue
        emitted = sum(1 + r["acc"] for r in body)
        total_all = sum(1 + r["acc"] for r in leg)
        kind = "drafting" if any(r["d"] for r in leg) else "serial"
        print("\n-- leg %d (%s): %d rounds, %d emitted tokens "
              "(%d after dropping %d warmup)" %
              (n, kind, len(leg), total_all, emitted, args.warmup))
        widths = Counter(r["d"] + 1 for r in body)
        accs = Counter(r["acc"] for r in body)
        mean_d = sum(r["d"] for r in body) / len(body)
        mean_a = sum(r["acc"] for r in body) / len(body)
        print("   M histogram: " + ", ".join(
            "M=%d:%d" % (w, c) for w, c in sorted(widths.items())))
        print("   acc histogram: " + ", ".join(
            "%d:%d" % (a, c) for a, c in sorted(accs.items())))
        print("   M = %.4f   accepted/round = %.4f   tokens/round = %.4f" %
              (1 + mean_d, mean_a, 1 + mean_a))
        table = position_table(body)
        for row in table:
            print("   pos %d: reached %5d accepted %5d  p = %.4f "
                  "[%.4f, %.4f]" % (row["position"], row["reached"],
                                    row["accepted"], row["p"], row["lo"],
                                    row["hi"]))
        entry = {"leg": n, "kind": kind, "rounds": len(leg),
                 "rounds_scored": len(body), "emitted_all_rounds": total_all,
                 "M": 1 + mean_d, "tokens_per_round": 1 + mean_a,
                 "width_histogram": {str(k): v for k, v in sorted(widths.items())},
                 "accept_histogram": {str(k): v for k, v in sorted(accs.items())},
                 "positions": table}
        if body[-1].get("ema"):
            final = body[-1]["ema"]
            entry["final_ema"] = final
            entry["shipped_ema_seed"] = SHIPPED_EMA_SEED
            print("   converged EMA: " + ", ".join("%.4f" % v for v in final))
            print("   shipped seed:  " + ", ".join(
                "%.4f" % v for v in SHIPPED_EMA_SEED))
        margins = [r["margin"] for r in body
                   if r.get("margin") is not None and math.isfinite(r["margin"])]
        if margins:
            entry["margin_median"] = st.median(margins)
            entry["margin_mean"] = st.mean(margins)
            print("   top-2 margin: median %.4f mean %.4f n=%d" %
                  (st.median(margins), st.mean(margins), len(margins)))
        # A round that rejects has at least one position where the head did
        # not reproduce the target. Contrast the target's own top-2 margin in
        # those rounds with the fully accepted rounds: a SMALL margin at the
        # rejecting rounds means the head is losing near-ties the target
        # itself barely resolves, which bounds what retraining can recover.
        rej = [r["margin"] for r in body if r["d"] and r["acc"] < r["d"]
               and r.get("margin") is not None
               and math.isfinite(r["margin"])]
        full = [r["margin"] for r in body if r["d"] and r["acc"] == r["d"]
                and r.get("margin") is not None
                and math.isfinite(r["margin"])]
        if rej and full:
            entry["margin_rejecting_rounds"] = {
                "n": len(rej), "median": st.median(rej), "mean": st.mean(rej)}
            entry["margin_full_accept_rounds"] = {
                "n": len(full), "median": st.median(full),
                "mean": st.mean(full)}
            print("   top-2 margin, rounds WITH a rejection: median %.4f "
                  "mean %.4f n=%d" % (st.median(rej), st.mean(rej), len(rej)))
            print("   top-2 margin, fully accepted rounds:   median %.4f "
                  "mean %.4f n=%d"
                  % (st.median(full), st.mean(full), len(full)))
        # In-situ round cost by verify width.
        by_width = defaultdict(list)
        for r in body:
            if "round" in r:
                by_width[r["d"] + 1].append(r["round"] / 1000.0)
        if by_width:
            entry["round_ms_by_width"] = {
                str(w): {"n": len(v), "median": st.median(v),
                         "mean": st.mean(v)}
                for w, v in sorted(by_width.items())}
            print("   in-situ round ms by width: " + ", ".join(
                "M=%d:%.3f(n=%d)" % (w, st.median(v), len(v))
                for w, v in sorted(by_width.items())))
        for phase in PHASES:
            vals = [r[phase] / 1000.0 for r in body if phase in r]
            if not vals:
                continue
            entry.setdefault("phase_ms", {})[phase] = {
                "median": st.median(vals), "mean": st.mean(vals)}
            per_width = defaultdict(list)
            for r in body:
                if phase in r:
                    per_width[r["d"] + 1].append(r[phase] / 1000.0)
            entry.setdefault("phase_ms_by_width", {})[phase] = {
                str(w): {"n": len(v), "median": st.median(v)}
                for w, v in sorted(per_width.items())}
            fit = slope_vs_draft(body, phase)
            if fit:
                entry.setdefault("phase_fit", {})[phase] = fit
        if entry.get("phase_fit"):
            print("   per-draft slope (ms/draft) / fixed (ms) / R^2:")
            for phase, f in entry["phase_fit"].items():
                print("     %-14s %8.4f  %8.4f  %.3f" %
                      (phase, f["slope_ms_per_draft"], f["fixed_ms"],
                       f["r2"]))
        report["legs"].append(entry)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


def vocabulary_coverage(cuts, fixture=PUBLIC_GOLDEN, tokens=512):
    """Share of decoded tokens a compact draft vocabulary of `cut` rows can
    still propose. A token the vocabulary drops is a GUARANTEED rejection at
    that draft position, whatever the head's quality.

    Two estimators bracket the design space.

    `id_prefix` keeps IDs 0 ..< cut plus the 26 official control tokens, which
    is what truncating the shipped vocabulary does. Qwen's IDs are roughly
    BPE-merge ordered, so this is close to keeping the most frequent rows.

    `corpus` additionally keeps every dropped ID that a selection corpus
    already showed, and is scored on a DISJOINT held-out half, so it estimates
    what re-selecting the rows by empirical frequency would recover. The
    selection half is the first `tokens` decoded tokens plus the prompt; the
    held-out half is the next `tokens` decoded tokens.
    """
    case = json.load(Path(fixture).open())["cases"][0]
    expected = case["expected_tokens"]
    decoded = expected[:tokens]
    holdout = expected[tokens:2 * tokens]
    selection = set(decoded) | set(case["prompt_tokens"])
    controls = set(range(CONTROL_START, CONTROL_END))
    out = {}
    for cut in cuts:
        kept = sum(1 for t in decoded if t < cut or t in controls)
        row = {"id_prefix": kept / len(decoded)}
        if holdout:
            # Budget neutral: every corpus row displaces one ID-ordered row.
            corpus = {t for t in selection if t not in controls}
            prefix = max(0, cut - len(corpus))
            chosen = corpus | set(range(prefix))
            hit = sum(1 for t in holdout if t in chosen or t in controls)
            row["corpus"] = hit / len(holdout)
            row["id_prefix_holdout"] = sum(
                1 for t in holdout if t < cut or t in controls) / len(holdout)
        out[cut] = row
    return out, len(decoded)


def head_fraction(d, head_fixed, head_slope, round_fixed, round_slope):
    """Head-chain share of one round that drafts `d` tokens."""
    return ((head_fixed + head_slope * d)
            / (round_fixed + round_slope * d))


def cmd_price(args):
    """Rung 3: price a better head (3a) and a smaller draft vocabulary (3b).

    Costs transfer as a FRACTION of the round, not as absolute milliseconds:
    the local host and the ranked M5 differ in absolute speed, but the head
    step and the target verify are both memory-bandwidth-bound on the same
    weights, so their ratio transfers far better than either time alone.
    """
    print("== measured local cost model (ms) ==")
    print("  round      = %.4f + %.4f * d" % (args.round_fixed,
                                              args.round_slope))
    print("  head chain = %.4f + %.4f * d" % (args.head_fixed,
                                              args.head_slope))
    print("  true headStepCostRatio h = head_slope / round(d=0) = %.4f "
          "(shipped %.2f)" % (args.head_slope / args.round_fixed, SHIPPED_H))
    print("  total marginal ratio     = round_slope / round(d=0) = %.4f"
          % (args.round_slope / args.round_fixed))

    print("\n== per-prompt ranked working point (ledger 207A) ==")
    work = {}
    for name, R, prop, acc, ms, raw in RANKED:
        d = prop / R
        p = solve_constant_p(d, acc / R)
        f = head_fraction(d, args.head_fixed, args.head_slope,
                          args.round_fixed, args.round_slope)
        work[name] = (d, p, f)
        print("  %-9s drafts/round %5.3f  p %6.4f  head share of round "
              "%5.2f%%" % (name, d, p, 100 * f))

    base = baseline_median()
    print("\n  reconstructed baseline published median = %.5f" % base)

    def evaluate(label, head_scale, p_scale):
        def new_ms(name, ms):
            d, _, f = work[name]
            return ms * (1.0 - f * (1.0 - head_scale))

        def new_tpr(name, tpr):
            d, p, _ = work[name]
            q = min(1.0, p * p_scale)
            return 1.0 + expected_accepted(d, [q] * (MAX_DEPTH + 1))

        rows, median = ranked_score_table(new_ms, new_tpr)
        print("  %-34s median %.5f  (%+.2f%%)"
              % (label, median, 100 * (median - base) / base))
        return {"arm": label, "head_scale": head_scale, "p_scale": p_scale,
                "median": median, "delta_pct": 100 * (median - base) / base,
                "per_prompt": [{"prompt": r[0], "tokens_per_round": r[2],
                                "round_ms": r[4], "raw": r[6]} for r in rows]}

    def evaluate_floor(label, floor, head_scale=1.0):
        """Lift every prompt's acceptance to at least `floor`."""
        def new_ms(name, ms):
            return ms * (1.0 - work[name][2] * (1.0 - head_scale))

        def new_tpr(name, tpr):
            d, p, _ = work[name]
            q = max(p, floor)
            return 1.0 + expected_accepted(d, [q] * (MAX_DEPTH + 1))

        _, median = ranked_score_table(new_ms, new_tpr)
        print("  %-34s median %.5f  (%+.2f%%)"
              % (label, median, 100 * (median - base) / base))
        return {"arm": label, "head_scale": head_scale,
                "acceptance_floor": floor, "median": median,
                "delta_pct": 100 * (median - base) / base}

    def evaluate_shape(label, ratios):
        """Apply a measured per-position acceptance SHAPE to each prompt.

        Each prompt keeps its ledger-derived scalar `p`, and position i gets
        `p * ratios[i]`, where `ratios` is normalised to position 1. This is
        the only way to price a change that touches some positions and not
        others, because the ranked aggregates alone cannot resolve the shape.
        """
        def new_tpr(name, tpr):
            d, p, _ = work[name]
            vec = [min(1.0, p * r) for r in ratios] + [min(1.0, p * ratios[-1])]
            return 1.0 + expected_accepted(d, vec)

        _, median = ranked_score_table(None, new_tpr)
        print("  %-34s median %.5f  (%+.2f%%)"
              % (label, median, 100 * (median - base) / base))
        return {"arm": label, "position_ratios": list(ratios),
                "median": median, "delta_pct": 100 * (median - base) / base}

    print("\n== 3a: better proposals at unchanged head cost ==")
    arms = [evaluate("p x %.3f" % s, 1.0, s)
            for s in (1.00, 1.005, 1.01, 1.02, 1.03, 1.05)]
    arms += [evaluate_floor("p_i -> %.3f every position" % f, f)
             for f in (0.97, 0.98, 0.99, 0.995, 1.0)]

    if args.shape:
        ratios = [v / args.shape[0] for v in args.shape]
        deep = ratios[:MAX_DEPTH - 3] + [1.0] * 3
        print("  measured local position shape, normalised to position 1:")
        print("    " + ", ".join("%.4f" % r for r in ratios))
        arms.append(evaluate_shape("measured position shape", ratios))
        arms.append(evaluate_shape(
            "deepest 3 restored to position 1", deep))

    print("\n== head cost alone, acceptance unchanged ==")
    arms += [evaluate("head cost x %.2f" % s, s, 1.0)
             for s in (0.75, 0.50, 0.25, 0.00)]

    cuts = [98304, 90112, 81920, 73728, 65536, 57344, 49152, 40960, 32768,
            24576, 16384, 8192]
    cov, n_dec = vocabulary_coverage(cuts, tokens=args.coverage_tokens)
    print("\n== 3b: smaller compact draft vocabulary "
          "(coverage from %d decoded tokens of %s) =="
          % (n_dec, Path(PUBLIC_GOLDEN).name))
    print("  stage (ii) coarse readout is %d B at %d rows; the head chain is "
          "%d B, so the readout is %.1f%% of the head step"
          % (COARSE_READOUT_BYTES, COMPACT_DRAFT_ROWS,
             DECLARED_HEAD_STEP_BYTES,
             100 * COARSE_READOUT_BYTES / DECLARED_HEAD_STEP_BYTES))
    vocab = []
    print("  %8s %10s %12s %12s %10s" %
          ("rows", "head x", "break-even", "measured", "median"))
    for cut in cuts:
        rows = cut + (CONTROL_END - CONTROL_START)
        readout = COARSE_READOUT_BYTES * rows / COMPACT_DRAFT_ROWS
        scale = ((DECLARED_HEAD_STEP_BYTES - COARSE_READOUT_BYTES + readout)
                 / DECLARED_HEAD_STEP_BYTES)
        lo, hi = 0.5, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, m = ranked_score_table(
                lambda n, ms: ms * (1.0 - work[n][2] * (1.0 - scale)),
                lambda n, tpr: 1.0 + expected_accepted(
                    work[n][0],
                    [min(1.0, work[n][1] * mid)] * (MAX_DEPTH + 1)))
            if m < base:
                lo = mid
            else:
                hi = mid
        breakeven = 0.5 * (lo + hi)
        c = cov[cut]["id_prefix"]
        _, m = ranked_score_table(
            lambda n, ms: ms * (1.0 - work[n][2] * (1.0 - scale)),
            lambda n, tpr: 1.0 + expected_accepted(
                work[n][0],
                [min(1.0, work[n][1] * c)] * (MAX_DEPTH + 1)))
        print("  %8d %10.4f %12.4f %12.4f %10.5f  (%+.2f%%)%s" %
              (rows, scale, breakeven, c, m, 100 * (m - base) / base,
               "  PAYS" if c >= breakeven else ""))
        vocab.append({"arm": "vocab %d" % rows, "vocabulary_rows": rows,
                      "head_scale": scale, "p_scale": c,
                      "coverage_id_prefix": c,
                      "coverage_corpus_degenerate": cov[cut].get("corpus"),
                      "breakeven_coverage": breakeven, "median": m,
                      "delta_pct": 100 * (m - base) / base})
    print("  NOTE: the `corpus` re-selection estimator is DEGENERATE on this"
          " fixture.\n        `longcopy-gate-english-512` copies 95 percent of"
          " its prompt, so its\n        held-out half holds 126 distinct tokens"
          " and exactly 1 unseen ID. It\n        cannot bound a"
          " frequency-selected vocabulary and is recorded, not used.")
    arms += vocab

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"baseline_median": base,
             "cost_model": {"round_fixed_ms": args.round_fixed,
                            "round_slope_ms": args.round_slope,
                            "head_fixed_ms": args.head_fixed,
                            "head_slope_ms": args.head_slope,
                            "measured_h": args.head_slope / args.round_fixed,
                            "shipped_h": SHIPPED_H},
             "working_point": {k: {"drafts_per_round": v[0], "p": v[1],
                                   "head_share_of_round": v[2]}
                               for k, v in work.items()},
             "head_step_bytes": {
                 "declared_total": DECLARED_HEAD_STEP_BYTES,
                 "declared_stage_i_block": DECLARED_HEAD_BLOCK_BYTES,
                 "declared_stage_ii_coarse_readout": COARSE_READOUT_BYTES,
                 "declared_stage_iii_top32": DECLARED_TOP32_BYTES,
                 "declared_stage_iv_rerank": DECLARED_RERANK_BYTES,
                 "pinned_total": PINNED_HEAD_STEP_BYTES,
                 "pinned_stage_i_block": PINNED_HEAD_BLOCK_BYTES,
                 "pinned_stage_ii_compact_select": PINNED_SELECT_BYTES},
             "coverage": {str(k): v for k, v in cov.items()},
             "arms": arms}, indent=2))
    return arms


def fit_p_to_drafts(target_drafts, marginal, reps, margins=None, seed=11):
    """Constant-`p` chain that makes the SHIPPED schedule propose
    `target_drafts` drafts per round at the given depth price. Common random
    numbers make the simulated mean a deterministic function of `p`, so plain
    bisection is valid."""
    def f(p):
        s = simulate([p] * (MAX_DEPTH + 1), marginal=marginal, seed=seed,
                     margins=margins, reps=reps)
        return s["M"] - 1.0

    lo, hi = 1e-4, 1.0 - 1e-6
    if f(hi) < target_drafts:
        return hi, f(hi)
    if f(lo) > target_drafts:
        return lo, f(lo)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if f(mid) < target_drafts:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return p, f(p)


def geometric_chain(p1, decay):
    return [min(1.0, p1 * decay ** i) for i in range(MAX_DEPTH + 1)]


def fit_p1_and_decay(target_drafts, target_accepted, marginal, reps,
                     margins=None, seed=11):
    """Two-parameter per-prompt fit against the TWO ranked observables.

    A constant-`p` chain is over-identified here: the `p` that reproduces the
    proposed width leaves accepted drafts far too low, and the `p` that
    reproduces accepted drafts proposes far too deep. A geometric chain adds
    the one degree of freedom that can separate them, because the schedule
    stops on the DEEP positions while the accepted count is earned by the
    SHALLOW ones. Decay near 1 lets the walk run deep, so `p1` must fall to
    hit the target width and accepted drafts fall with it; strong decay stops
    the walk early and lets `p1` stay high. Accepted is therefore monotone in
    the decay and the pair is identified.
    """
    def run(p1, decay):
        return simulate(geometric_chain(p1, decay), marginal=marginal,
                        seed=seed, margins=margins, reps=reps)

    def p1_for(decay):
        lo, hi = 1e-4, 1.0 - 1e-6
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if run(mid, decay)["M"] - 1.0 < target_drafts:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lo_d, hi_d = 0.40, 1.0
    if run(p1_for(lo_d), lo_d)["accepted"] / run(
            p1_for(lo_d), lo_d)["rounds"] < target_accepted:
        decay = lo_d
        p1 = p1_for(decay)
        return p1, decay, run(p1, decay), False
    for _ in range(22):
        mid = 0.5 * (lo_d + hi_d)
        r = run(p1_for(mid), mid)
        if r["accepted"] / r["rounds"] > target_accepted:
            lo_d = mid
        else:
            hi_d = mid
    decay = 0.5 * (lo_d + hi_d)
    p1 = p1_for(decay)
    return p1, decay, run(p1, decay), True


def cmd_calibrate(args):
    """The advisor's calibration set, used as an OUT-OF-SAMPLE test.

    Every prompt's acceptance chain is fitted on the `h = 0.18` column alone,
    which is an exact one-parameter fit and therefore carries no information
    about the second column. The same chain then predicts the `h = 0.32`
    column with no free parameter left. A model that survives that is allowed
    to predict arm 3; a model that does not is reported as failed.
    """
    margins = None
    if args.margins:
        body = parse_trace(args.margins)[0][2:]
        margins = [r["margin"] for r in body
                   if r.get("margin") is not None
                   and math.isfinite(r["margin"])]
        print("margin clamp ON: %d empirical top-2 margins, median %.3f"
              % (len(margins), st.median(margins)))
    else:
        print("margin clamp OFF: the schedule's depth-0 and depth-1 top-2 "
              "clamp is disabled")

    rr = ranked_round_cost_model()
    print("\n== ranked round cost from the ledger 207(A) eight prompts ==")
    print("  round_ms = %.3f + %.3f * drafts   (R^2 %.4f)"
          % (rr["intercept"], rr["slope"], rr["r2"]))
    print("  ranked TOTAL marginal ratio = %.4f, against the shipped "
          "headStepCostRatio %.2f" % (rr["marginal_ratio"], SHIPPED_H))

    lo, hi = H_SWEEP_H
    m_lo = [lo] * MAX_DEPTH
    m_hi = [hi] * MAX_DEPTH
    print("\n== fit each prompt on h = %.2f, then PREDICT h = %.2f ==" %
          (lo, hi))
    print("  %-8s %8s %9s %10s %10s %9s" %
          ("prompt", "p fitted", "obs@0.18", "pred@0.32", "obs@0.32", "error"))
    rows = []
    data = [("wide%d" % (i + 1), a, b)
            for i, (a, b) in enumerate(H_SWEEP_WIDE)]
    data.append(("hard", H_SWEEP_HARD[0], H_SWEEP_HARD[1]))
    for name, obs_lo, obs_hi in data:
        p, got = fit_p_to_drafts(obs_lo, m_lo, args.reps, margins)
        pred = simulate([p] * (MAX_DEPTH + 1), marginal=m_hi, seed=11,
                        margins=margins, reps=args.reps)["M"] - 1.0
        err = pred - obs_hi
        print("  %-8s %8.4f %9.3f %10.3f %10.3f %+9.3f"
              % (name, p, got, pred, obs_hi, err))
        rows.append({"prompt": name, "p_fitted": p, "fit_drafts_h_lo": got,
                     "observed_drafts_h_lo": obs_lo,
                     "predicted_drafts_h_hi": pred,
                     "observed_drafts_h_hi": obs_hi, "error": err})
    errs = [r["error"] for r in rows]
    rel = [r["error"] / r["observed_drafts_h_hi"] for r in rows]
    print("  out-of-sample: mean error %+.3f drafts, mean |relative| %.1f%%, "
          "max |relative| %.1f%%"
          % (st.mean(errs), 100 * st.mean(abs(x) for x in rel),
             100 * max(abs(x) for x in rel)))
    print("  sign test: %d of %d predicted the observed DIRECTION of change"
          % (sum(1 for r in rows
                 if (r["predicted_drafts_h_hi"] - r["observed_drafts_h_lo"])
                 * (r["observed_drafts_h_hi"] - r["observed_drafts_h_lo"]) > 0),
             len(rows)))

    print("\n== ledger 207(A): two-parameter geometric chain fitted on BOTH "
          "ranked observables ==")
    print("  %-9s %8s %8s %9s %9s %10s %10s %6s" %
          ("prompt", "p1", "decay", "drafts", "fit", "accepted", "fit", "ok"))
    arm3 = []
    for name, R, prop, acc, ms, raw in RANKED:
        t_d, t_a = prop / R, acc / R
        p1, dk, r, ok = fit_p1_and_decay(t_d, t_a, m_lo, args.reps, margins)
        print("  %-9s %8.4f %8.4f %9.3f %9.3f %10.3f %10.3f %6s"
              % (name, p1, dk, t_d, r["M"] - 1.0, t_a,
                 r["accepted"] / r["rounds"], ok))
        arm3.append({"prompt": name, "p1_fitted": p1, "decay_fitted": dk,
                     "identified": ok, "shipped_drafts": t_d,
                     "fit_drafts": r["M"] - 1.0, "shipped_accepted": t_a,
                     "fit_accepted": r["accepted"] / r["rounds"],
                     "chain": geometric_chain(p1, dk)[:MAX_DEPTH]})
    if not all(r["identified"] for r in arm3):
        print("  NOT IDENTIFIED on at least one prompt: even the strongest "
              "decay the fit allows\n  cannot reach the observed accepted "
              "count at the observed width. The\n  two-parameter geometric "
              "chain is REJECTED on those prompts and the arm-3\n  prediction "
              "below is reported as a failed model, not as a forecast.")

    print("\n== arm 3: the pbfit depth-price vector, predicted not fitted ==")
    for row, (name, R, prop, acc, ms, raw) in zip(arm3, RANKED):
        pred = simulate(geometric_chain(row["p1_fitted"],
                                        row["decay_fitted"]),
                        marginal=PBFIT_MARGINAL, seed=11, margins=margins,
                        reps=args.reps)
        row["pbfit_drafts"] = pred["M"] - 1.0
        row["pbfit_tokens_per_round"] = pred["tokens_per_round"]
        print("  %-9s drafts %5.3f -> %5.3f   tokens/round %5.3f -> %5.3f"
              % (name, row["shipped_drafts"], row["pbfit_drafts"],
                 1 + acc / R, pred["tokens_per_round"]))

    by = {r["prompt"]: r for r in arm3}
    base = baseline_median()
    rows2, median = ranked_score_table(
        lambda n, ms: rr["intercept"] + rr["slope"] * by[n]["pbfit_drafts"],
        lambda n, tpr: by[n]["pbfit_tokens_per_round"])
    raws = sorted(r[6] for r in rows2)
    print("\n  predicted published median %.5f (%+.2f%%), observed arm 3 "
          "%.5f (%+.2f%%)"
          % (median, 100 * (median - base) / base, ARM3["published_median"],
             100 * (ARM3["published_median"]
                    - ARM3["baseline_published_median"])
             / ARM3["baseline_published_median"]))
    print("  predicted 4th/5th sorted raw %.4f / %.4f, observed %.4f / %.4f"
          % (raws[3], raws[4], ARM3["fourth_sorted"], ARM3["fifth_sorted"]))
    print("  predicted minimum raw %.4f, observed %.4f"
          % (raws[0], ARM3["min_raw"]))
    report = {"margin_clamp": bool(margins), "reps": args.reps,
              "ranked_round_cost": rr, "h_sweep": rows, "arm3": arm3,
              "arm3_predicted_median": median,
              "arm3_predicted_delta_pct": 100 * (median - base) / base,
              "arm3_predicted_sorted_raw": raws,
              "arm3_observed": ARM3,
              "baseline_median": base}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("chainfit")
    a.add_argument("--decay", type=float, default=0.98)
    a.add_argument("--reps", type=int, default=8)
    a.add_argument("--out")
    a.set_defaults(func=cmd_chainfit)

    b = sub.add_parser("reprice")
    b.add_argument("--p", type=float, default=0.88)
    b.add_argument("--p-vector", type=float, nargs=MAX_DEPTH, default=None)
    b.add_argument("--head-step-ms", type=float, default=2.590)
    b.add_argument("--reps", type=int, default=64)
    b.add_argument("--out")
    b.set_defaults(func=cmd_reprice)

    c = sub.add_parser("census")
    c.add_argument("trace")
    c.add_argument("--warmup", type=int, default=2)
    c.add_argument("--out")
    c.set_defaults(func=cmd_census)

    d = sub.add_parser("price")
    d.add_argument("--round-fixed", type=float, required=True)
    d.add_argument("--round-slope", type=float, required=True)
    d.add_argument("--head-fixed", type=float, required=True)
    d.add_argument("--head-slope", type=float, required=True)
    d.add_argument("--shape", type=float, nargs=MAX_DEPTH, default=None,
                   help="measured per-position acceptance, position 1 first")
    d.add_argument("--coverage-tokens", type=int, default=512)
    d.add_argument("--out")
    d.set_defaults(func=cmd_price)

    e = sub.add_parser("calibrate")
    e.add_argument("--reps", type=int, default=24)
    e.add_argument("--margins", help="trace whose top-2 margins feed the "
                                     "schedule's depth-0/1 clamp")
    e.add_argument("--out")
    e.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
