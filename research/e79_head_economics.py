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
# floor(M) = 30.402 + (M-1) * 8.42/8, ledger 211(A).
RANKED_DEPTH0_ROUND_MS = 30.402
RANKED_HEAD_STEP_MS = 8.42 / 8


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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
