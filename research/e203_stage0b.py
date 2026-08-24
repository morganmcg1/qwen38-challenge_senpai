#!/usr/bin/env python3
"""E203 stage 0b: the hard ceiling on every WITHIN-prompt depth rule.

harness=ranked for every number this file prints.

WHAT A CEILING HAS TO BOUND. A within-prompt rule chooses round r's depth from
what it has observed inside the current prompt. Two rules bracket the whole
family:

  const(d)    the SAME depth for every round of a prompt. This rule uses no
              within-prompt information at all, so it is the floor the family
              must beat. Chosen per prompt it is also the strongest rule the
              exchangeable world of E201 admits: under exchangeability the
              best causal decision cannot depend on the round.
  oracle_q    the depth that is optimal for the round's own latent acceptance
              rate, chosen with full knowledge of that rate. No implementable
              rule can beat it, because the rate is the only thing about the
              round that a depth decision can use.

Between them sits the arm the campaign already pays for: the SHIPPED greedy
price at its best static cap, 3.829386 at cap 8 (FINDING 523). So

  oracle_q - const          = all within-prompt information is worth this
  shipped(cap 8) - const    = what the shipped EMA tracker already collects
  oracle_q - shipped(cap 8) = the HEADROOM E203 would have to build a rule for

The assignment also names a looser bound, and this file prices it too:

  oracle_a    the depth that is optimal for the round's REALISED accepted
              run length. It knows the coin flips, not only the rate, so it is
              above oracle_q and above every implementable rule. It is
              reported because the assignment asks for it, and it is not used
              as the stop gate, because no rule can reach it even with perfect
              prediction.

THE EVIDENCE-ANCHORED CEILING. A headroom is only reachable to the extent that
the history predicts the round. Stage 0a measures that directly and bounds the
predictable fraction of the accepted-length variance at psi. The ceiling at
information psi is priced by giving the rule PERFECT knowledge of the rate on a
psi fraction of rounds and nothing beyond the prompt constant on the rest.
Value is concave in information, so full knowledge on a psi fraction dominates
partial knowledge everywhere at the same variance reduction: this mixture is an
upper bound on any rule whose predictive power is psi.

THE INSTRUMENT. FINDING 520 / RULE 79: the only admissible depth-price
instrument is the survival-pinned latent-q desk replay, validated out of sample
against the paid cap-4 and cap-5 receipts inside the 0.689 % receipt channel.
This file imports it from `e201_online_cap` unchanged and adds only the arms
above, so the validation gate is the same gate, not a restatement of it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e197_refit as E  # noqa: E402
import e200_desk_price as D  # noqa: E402
import e201_online_cap as O  # noqa: E402

ORDER = E.ORDER
TOKENS = E.TOKENS
MAXD = 8                    # Qwen36MTPLimits.maxDepth, the parent's own limit
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e203-artifacts"


# ------------------------------------------------------------------- nodes

def nodes(inst, name):
    """(weight, latent q, per-position rate, shipped depth) for one prompt."""
    c = inst["cal"][name]
    gamma = c["gamma"]
    return [(w, q, q ** gamma, d)
            for q, w, d in zip(c["q"], c["w"], c["depth_ship"])]


def assemble(inst, law, name, items):
    """Turn per-round (weight, depth, accepted) items into a priced prompt.

    The receipt anchor is kept exactly as `e200_desk_price.price_arm2` keeps
    it: the paid mean round cost plus the fitted difference between the depth
    this arm takes and the depth the shipped rule took on the same round.
    """
    rec = inst["data"]["A"]["rec"][name]
    abar = sum(w * a for w, _, a in items)
    ship = sum(w * E.R_of(law, d + 1) for w, _, _, d in nodes(inst, name))
    cost = sum(w * E.R_of(law, d + 1) for w, d, _ in items)
    R = rec["R"] + cost - ship
    total = TOKENS / (1.0 + abar) * R
    return {"abar": abar, "R": R, "edl": sum(w * d for w, d, _ in items),
            "total": total, "raw": E.raw_of(inst["data"], name, total)}


# -------------------------------------------------------------------- arms

def items_const(inst, name, depth):
    """Every round of the prompt drafts the same depth."""
    return [(w, depth, D.geom(p, depth)) for w, _, p, _ in nodes(inst, name)]


def items_oracle_q(inst, law, name, lam):
    """Each round drafts the depth that is optimal for its own latent rate."""
    out = []
    for w, _, p, _ in nodes(inst, name):
        best = min(range(MAXD + 1),
                   key=lambda d: E.R_of(law, d + 1) - lam * D.geom(p, d))
        out.append((w, best, D.geom(p, best)))
    return out


def items_oracle_a(inst, law, name, lam):
    """Each round drafts the depth that is optimal for its REALISED run.

    The realised accepted run length k is geometric at the round's own rate.
    Drafting past k buys no token, so the search stops at k.
    """
    out = []
    for w, _, p, _ in nodes(inst, name):
        tail = 1.0
        for k in range(MAXD + 1):
            mass = tail * (1.0 - p) if k < MAXD else tail
            tail *= p
            if mass <= 0.0:
                continue
            best = min(range(k + 1),
                       key=lambda d: E.R_of(law, d + 1) - lam * d)
            out.append((w * mass, best, float(best)))
    return out


def optimise(inst, law, name, builder, iters=80):
    """Dinkelbach fixed point for a ratio objective.

    The prompt's decode seconds are a RATIO of two round-level sums, so a
    per-round rule cannot be optimised by minimising per-round cost per token
    independently. `lam` is the current seconds-per-token value of the arm, and
    each round minimises R(d + 1) - lam * accepted at that value. The iteration
    is monotone and stops when the value stops moving.
    """
    state = assemble(inst, law, name, items_const(inst, name, MAXD))
    lam = state["total"] / TOKENS
    for _ in range(iters):
        state = assemble(inst, law, name, builder(inst, law, name, lam))
        nxt = state["total"] / TOKENS
        if abs(nxt - lam) < 1e-15:
            break
        lam = nxt
    return state


def mix_value(base, oracle, psi):
    """Seconds per token of a prompt whose rounds are psi oracle, rest base.

    Rounds add, so seconds and emitted tokens both mix linearly; the ratio does
    not.
    """
    R = psi * oracle["R"] + (1.0 - psi) * base["R"]
    tok = psi * (1.0 + oracle["abar"]) + (1.0 - psi) * (1.0 + base["abar"])
    return {"R": R, "abar": tok - 1.0, "total": TOKENS / tok * R}


def optimise_mix(inst, law, name, base, psi, iters=80):
    """The psi-informed ceiling, with the oracle rounds priced at the MIXTURE.

    The oracle rounds of a mixture must be optimised against the mixture's own
    seconds per token, not against the value a pure-oracle prompt would reach.
    Optimising them at the pure-oracle rate understates the arm, and an
    understated ceiling is not a ceiling.
    """
    state = mix_value(base, optimise(inst, law, name, items_oracle_q), psi)
    lam = state["total"] / TOKENS
    for _ in range(iters):
        oracle = assemble(inst, law, name, items_oracle_q(inst, law, name, lam))
        state = mix_value(base, oracle, psi)
        nxt = state["total"] / TOKENS
        if abs(nxt - lam) < 1e-15:
            break
        lam = nxt
    return state


# ------------------------------------------------------------------ scoring

def median_of(raws):
    return e177.published_median([raws[n] for n in ORDER])


def pct(x, base):
    return 100.0 * (x / base - 1.0)


# ------------------------------------------------------------------ pricing

def price(inst, grid, laws, psis):
    out = {}
    for key, law in laws.items():
        arms, per_prompt = {}, {}
        const = {name: {d: assemble(inst, law, name, items_const(inst, name, d))
                        for d in range(MAXD + 1)} for name in ORDER}
        oq = {name: optimise(inst, law, name, items_oracle_q)
              for name in ORDER}
        oa = {name: optimise(inst, law, name, items_oracle_a)
              for name in ORDER}

        # constant depth, one value for the whole board and per prompt
        best_global = max(
            range(MAXD + 1),
            key=lambda d: median_of({n: const[n][d]["raw"] for n in ORDER}))
        arms["const_global"] = {
            "depth": best_global,
            "median": median_of({n: const[n][best_global]["raw"]
                                 for n in ORDER}),
            "table": {d: median_of({n: const[n][d]["raw"] for n in ORDER})
                      for d in range(MAXD + 1)},
        }
        best_pp = {n: max(range(MAXD + 1), key=lambda d: const[n][d]["raw"])
                   for n in ORDER}
        arms["const_per_prompt"] = {
            "depths": best_pp,
            "median": median_of({n: const[n][best_pp[n]]["raw"]
                                 for n in ORDER}),
        }
        arms["oracle_q"] = {"median": median_of({n: oq[n]["raw"]
                                                 for n in ORDER})}
        arms["oracle_a"] = {"median": median_of({n: oa[n]["raw"]
                                                 for n in ORDER})}

        # The evidence-anchored ceilings. Two bases, because they answer two
        # different questions:
        #
        #  const  the per-prompt constant, i.e. what is ALL within-prompt
        #         information worth from a standing start;
        #  ship   the shipped greedy price at cap 8, i.e. what is the EXTRA
        #         information worth to a rule bolted onto the incumbent. This
        #         is the E203 question, and it is the generous reading: stage
        #         0a measures the TOTAL predictability of the acceptance
        #         history, and the shipped EMA state is itself a function of
        #         that history, so the increment is at most psi.
        bases = {"const": {n: const[n][best_pp[n]] for n in ORDER},
                 "ship": {n: grid[key][n][8] for n in ORDER}}
        arms["psi"] = {}
        for tag, base in bases.items():
            arms["psi"][tag] = {}
            for psi in psis:
                raws = {}
                for n in ORDER:
                    m = optimise_mix(inst, law, n, base[n], psi)
                    raws[n] = E.raw_of(inst["data"], n, m["total"])
                arms["psi"][tag]["%.6f" % psi] = median_of(raws)

        # How much predictive power would the assignment's stop rule need?
        # Inverting the ceiling curve turns the +0.2 % and +0.5 % thresholds
        # into a requirement on the evidence, which is what makes the closure
        # falsifiable rather than merely negative.
        c8_raw = {n: grid[key][n][8]["raw"] for n in ORDER}
        c8_med = median_of(c8_raw)
        arms["break_even"] = {}
        for target in (0.002, 0.005):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                psi = 0.5 * (lo + hi)
                raws = {n: E.raw_of(
                    inst["data"], n,
                    optimise_mix(inst, law, n, bases["ship"][n], psi)["total"])
                    for n in ORDER}
                if median_of(raws) < c8_med * (1.0 + target):
                    lo = psi
                else:
                    hi = psi
            arms["break_even"]["%.3f" % target] = 0.5 * (lo + hi)

        for n in ORDER:
            per_prompt[n] = {
                "const_best_depth": best_pp[n],
                "const_raw": const[n][best_pp[n]]["raw"],
                "const_edl": float(best_pp[n]),
                "oracle_q_raw": oq[n]["raw"], "oracle_q_edl": oq[n]["edl"],
                "oracle_a_raw": oa[n]["raw"], "oracle_a_edl": oa[n]["edl"],
                "const_table": {d: const[n][d]["raw"]
                                for d in range(MAXD + 1)},
            }
        out[key] = {"arms": arms, "per_prompt": per_prompt}
    return out


# ------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0a", default=str(ARTIFACTS / "stage0a-p7.json"))
    ap.add_argument("--json", default=str(ARTIFACTS / "stage0b-ceiling.json"))
    args = ap.parse_args()

    budget = json.loads(pathlib.Path(args.stage0a).read_text())["structure_budget"]
    psi_lag1 = budget["predictable_variance_fraction_upper95"]
    psi_any = budget["predictable_variance_fraction_any_lag_upper95"]
    psis = sorted({0.0, psi_lag1, psi_any, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0})

    inst = O.build_instrument()
    laws, ratio9 = O.cost_laws(inst["fit"])
    grid = O.cap_grid(inst, laws)
    A = E.BEST_A
    res = price(inst, grid, laws, psis)

    print("=" * 78)
    print("E203 STAGE 0b  WITHIN-PROMPT CEILING  (harness=ranked)")
    print("=" * 78)
    print("  instrument: FINDING 520 survival-pinned latent-q, imported from")
    print("              e201_online_cap without modification")
    print("  ranked law: smooth-step, AICc %.1f, wRMSE %.3f ms"
          % (inst["fit"]["aicc"], inst["fit"]["wrmse_ms"]))
    print("  9-row cell: smooth continuation vs E186 step transferred at %.3f"
          % ratio9)
    print("  receipt A %.8f (cap 7), crown %.8f" % (A, E.CROWN))
    print()

    m7 = median_of({n: grid["smooth"][n][7]["raw"] for n in ORDER})
    m4 = e177.published_median(
        [E.raw_of(inst["data"], n, grid["smooth"][n][4]["total"], "C")
         for n in ORDER])
    m5 = e177.published_median(
        [E.raw_of(inst["data"], n, grid["smooth"][n][5]["total"], "E")
         for n in ORDER])
    print("  INSTRUMENT VALIDATION GATE (same bar as FINDING 520)")
    print("   cap 7 in-sample     %.8f vs paid %.8f  (%+0.4f%%)"
          % (m7, A, pct(m7, A)))
    print("   cap 4 out-of-sample %.6f vs paid %.6f  (%+0.2f%%)"
          % (m4, E.CAP4, pct(m4, E.CAP4)))
    print("   cap 5 out-of-sample %.6f vs paid %.6f  (%+0.2f%%)"
          % (m5, E.CAP5, pct(m5, E.CAP5)))
    print("   receipt channel 1-sigma %.3f%%" % (100 * E.SIGMA_PUBLISHED))
    gate = (abs(pct(m7, A)) < 1e-6
            and abs(pct(m4, E.CAP4)) < 100 * E.SIGMA_PUBLISHED
            and abs(pct(m5, E.CAP5)) < 100 * E.SIGMA_PUBLISHED)
    print("   gate: %s" % ("PASS" if gate else "FAIL"))
    print()

    static = {c: median_of({n: grid["smooth"][n][c]["raw"] for n in ORDER})
              for c in O.CAPS}
    cap8 = static[8]
    print("  COMPARATOR (FINDING 523): best static global cap")
    for c in O.CAPS:
        print("   cap %d  %12.6f  %+7.2f%% vs A" % (c, static[c],
                                                    pct(static[c], A)))
    print("   best static cap 8 = %.6f (%+0.2f%% vs A). Every E203 number is"
          % (cap8, pct(cap8, A)))
    print("   read against THIS, not against receipt A.")
    print()

    out = {"harness": "ranked", "receipt_A": A, "static_cap8": cap8,
           "psi_lag1": psi_lag1, "psi_any_lag": psi_any,
           "gate": {"cap7": m7, "cap4": m4, "cap5": m5, "pass": gate},
           "static_global_cap": static, "laws": {}}

    for key in ("smooth", "step9"):
        arms = res[key]["arms"]
        c8 = median_of({n: grid[key][n][8]["raw"] for n in ORDER})
        print("  WITHIN-PROMPT INFORMATION LADDER  (9-row cell: %s)" % key)
        print("   %-34s %12s %9s %9s"
              % ("arm", "median", "vs A", "vs cap8"))
        rows = [
            ("best constant depth, global d=%d" % arms["const_global"]["depth"],
             arms["const_global"]["median"]),
            ("best constant depth, per prompt", arms["const_per_prompt"]["median"]),
            ("SHIPPED greedy price, cap 8", c8),
            ("oracle_q  (knows the round's rate)", arms["oracle_q"]["median"]),
            ("oracle_a  (knows the round's run)", arms["oracle_a"]["median"]),
        ]
        for label, value in rows:
            print("   %-34s %12.6f %+8.2f%% %+8.2f%%"
                  % (label, value, pct(value, A), pct(value, c8)))
        print("   per-prompt constant depths: "
              + " ".join("%s=%d" % (n[:4], d)
                         for n, d in arms["const_per_prompt"]["depths"].items()))
        print()
        print("   DECOMPOSITION of the within-prompt information, vs the")
        print("   per-prompt constant that uses none of it")
        base = arms["const_per_prompt"]["median"]
        print("     all of it (oracle_q)                 %+7.2f%%"
              % pct(arms["oracle_q"]["median"], base))
        print("     collected by the shipped EMA tracker %+7.2f%%"
              % pct(c8, base))
        print("     HEADROOM left for any E203 rule      %+7.2f%%"
              % pct(arms["oracle_q"]["median"], c8))
        print("     looser bound that knows the run      %+7.2f%%"
              % pct(arms["oracle_a"]["median"], c8))
        print()
        print("   CEILING AT MEASURED INFORMATION psi (stage 0a). psi is the")
        print("   95%% upper bound on the fraction of accepted-length variance")
        print("   the history predicts; the rule is GIVEN the round's exact")
        print("   rate on that fraction of rounds and the incumbent's decision")
        print("   on the rest. THE `ship` COLUMN IS THE E203 ANSWER: it prices")
        print("   a rule bolted onto the shipped tracker, which is what the")
        print("   assignment would have to build.")
        print("   %-14s %12s %9s | %12s %9s"
              % ("psi", "on const", "vs cap8", "on ship", "vs cap8"))
        for psi_key in sorted(arms["psi"]["ship"], key=float):
            psi = float(psi_key)
            tag = ""
            if abs(psi - psi_lag1) < 1e-12:
                tag = "  <- stage 0a lag-1 bound"
            elif abs(psi - psi_any) < 1e-12:
                tag = "  <- stage 0a best-of-16-lags bound"
            vc = arms["psi"]["const"][psi_key]
            vs = arms["psi"]["ship"][psi_key]
            print("   %-14s %12.6f %+8.2f%% | %12.6f %+8.3f%%%s"
                  % ("%.4f %%" % (100 * psi), vc, pct(vc, c8), vs,
                     pct(vs, c8), tag))
        be = arms["break_even"]
        print("   REQUIRED EVIDENCE. To clear the assignment's thresholds the")
        print("   acceptance history would have to predict this share of the")
        print("   round-to-round variance:")
        print("     +0.2%% (hold for advisor)  psi >= %.2f%%"
              % (100 * be["0.002"]))
        print("     +0.5%% (proceed to stage 1) psi >= %.2f%%"
              % (100 * be["0.005"]))
        print("   measured 95%% upper bound: %.4f%% (lag 1), %.4f%% (best lag)"
              % (100 * psi_lag1, 100 * psi_any))
        print()
        out["laws"][key] = {
            "cap8": c8, "const_global": arms["const_global"],
            "const_per_prompt": arms["const_per_prompt"],
            "oracle_q": arms["oracle_q"]["median"],
            "oracle_a": arms["oracle_a"]["median"],
            "psi": arms["psi"], "break_even": arms["break_even"],
            "headroom_pct_vs_cap8": pct(arms["oracle_q"]["median"], c8),
            "headroom_a_pct_vs_cap8": pct(arms["oracle_a"]["median"], c8),
            "ceiling_at_psi_lag1_pct_vs_cap8":
                pct(arms["psi"]["ship"]["%.6f" % psi_lag1], c8),
            "ceiling_at_psi_any_pct_vs_cap8":
                pct(arms["psi"]["ship"]["%.6f" % psi_any], c8),
            "per_prompt": res[key]["per_prompt"],
        }

    ARTIFACTS.mkdir(exist_ok=True)
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
    print("  wrote %s" % args.json)


if __name__ == "__main__":
    main()
