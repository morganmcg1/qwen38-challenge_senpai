#!/usr/bin/env python3
"""E25 r2 rule-space analysis: is a measured row price a price, or a wall?

Zero GPU. Three things:

  1. ADMISSIBILITY THEOREM. For the greedy marginal rule the campaign ships,
     row `d+1` is admissible for SOME acceptance vector iff `c_d < 1/(d+1)`.
     This settles PR #29 r2 objection 1 for all `p`, not just the taped ones,
     and it explains why nobody has ever seen this wall: the shipped scalar
     form `h/(1 + d*h)` provably cannot produce one for any `h < 1`.

  2. THE GREEDY RULE IS LOCAL ASCENT. Maximising the round's token rate
     `(1 + expected(D)) / T(D)` over `D` is the same objective; the shipped
     rule is its FIRST-DIFFERENCE test, so it stops at the first decline. On a
     non-convex `T` curve -- a cliff followed by a cheap shelf -- local ascent
     cannot see past the cliff. The global argmax over the SAME measured curve
     prices every depth and bans none. That is arm G.

  3. BREAK-EVEN. Given `T(0..5)`, what `T(6)`, `T(7)` would make depth 6 or 7
     beat depth 3 at perfect acceptance? A single number to measure against.

usage: research/e25r2_rule.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHIPPED_H = 0.18

# r1's tape means, d = 1..5 (ms), old base 0d2eef9. STALE by construction: the
# organizer frontier changed the draft readout kernel, the shortlist and the
# residual norm. Kept only as the r1 reference the refit is compared against.
R1_ROUND_MS = {1: 72.2749, 2: 79.2064, 3: 91.2664, 4: 131.6465, 5: 143.1506}
R1_STEP_RATIO = {0: 0.0, 1: 0.095904, 2: 0.152261, 3: 0.442442}


def shipped_coefficient(depth: int, h: float = SHIPPED_H) -> float:
    return h / (1.0 + depth * h)


# --------------------------------------------------------------------------
# 1. admissibility
# --------------------------------------------------------------------------
def admissible_ceiling(depth: int) -> float:
    """Largest row-price coefficient at `depth` that still admits row depth+1.

    The greedy rule adds row `d+1` iff `reach_d > c_d * (1 + expected_d)`, with
    `reach_d = prod_{i<=d} p_i` and `expected_d = sum_{k<d} prod_{i<=k} p_i`.
    Every `p_i <= 1`, so `reach_k >= reach_d` for `k < d`, hence

        expected_d >= d * reach_d.

    Writing `x = reach_d in [0, 1]`, firing requires

        x > c_d * (1 + d*x)   <=>   x * (1 - c_d*d) > c_d.

    If `c_d*d >= 1` the left side is <= 0 and the row is refused outright.
    Otherwise it needs `x > c_d / (1 - c_d*d)`, which some `x <= 1` satisfies
    iff `c_d / (1 - c_d*d) < 1`, i.e. `c_d*(d+1) < 1`.

    Both bounds are attained at `p_i = 1` for all i (`reach = 1`,
    `expected = d`), so the condition is necessary AND sufficient:

        row d+1 is admissible for some acceptance vector  <=>  c_d < 1/(d+1).
    """
    return 1.0 / (depth + 1)


def fires(coeff, p: list[float]) -> int:
    """Depth the greedy rule reaches under coefficients `coeff` and probs `p`."""
    reach, expected, depth = 1.0, 0.0, 0
    while depth < len(p):
        reach *= p[depth]
        if not reach > coeff(depth) * (1.0 + expected):
            break
        expected += reach
        depth += 1
    return depth


def verify_theorem(trials: int = 400_000, seed: int = 20260818) -> dict:
    """Numerically corroborate the algebra on the two forms in play.

    Monotone non-increasing `p` vectors (the shipped EMA update keeps accepted
    positions above rejected ones, so this is the realistic family) plus the
    unconstrained i.i.d. family, which is the strictly harder test.
    """
    rng = random.Random(seed)
    price_d = {d: max(shipped_coefficient(d), R1_STEP_RATIO.get(d, 0.0))
               for d in range(8)}

    def armd(d: int) -> float:
        return price_d[d]

    out: dict[str, object] = {
        "admissible_ceiling": {d: admissible_ceiling(d) for d in range(8)},
        "shipped_coefficient": {d: shipped_coefficient(d) for d in range(8)},
        "arm_d_coefficient": price_d,
        "arm_d_admissible": {d: price_d[d] < admissible_ceiling(d)
                             for d in range(8)},
        "shipped_admissible": {d: shipped_coefficient(d) < admissible_ceiling(d)
                               for d in range(8)},
    }

    # `h/(1+dh) < 1/(d+1)` reduces to `h < 1`, so the shipped form is
    # wall-free for every legal h. Checked over the whole h grid the campaign
    # has ever run, at every depth.
    out["shipped_form_wall_free"] = all(
        shipped_coefficient(d, h) < admissible_ceiling(d)
        for d in range(8)
        for h in (0.01, 0.14, 0.15, 0.18, 0.32, 0.5, 0.9, 0.99))

    for name, monotone in (("monotone", True), ("iid", False)):
        best = -1e9
        hist = {d: 0 for d in range(9)}
        deep = 0
        for _ in range(trials):
            if monotone:
                p = sorted((rng.random() for _ in range(8)), reverse=True)
            else:
                p = [rng.random() for _ in range(8)]
            d = fires(armd, p)
            hist[d] += 1
            if d >= 4:
                deep += 1
            # slack of the depth-3 test, whose sign is the theorem's claim
            reach, expected = 1.0, 0.0
            for i in range(3):
                reach *= p[i]
                expected += reach
            reach *= p[3]
            best = max(best, reach - armd(3) * (1.0 + expected))
        out[f"{name}_depth_histogram"] = hist
        out[f"{name}_depth_ge_4"] = deep
        out[f"{name}_best_depth3_slack"] = best

    # The corner the theorem says is the maximiser.
    reach, expected = 1.0, 0.0
    for i in range(3):
        reach *= 1.0
        expected += reach
    out["corner_p1_depth3_slack"] = 1.0 - armd(3) * (1.0 + expected)
    out["corner_p1_depth3_threshold"] = armd(3) * (1.0 + expected)
    out["arm_d_is_hard_cap_at_depth_3"] = out["corner_p1_depth3_slack"] < 0
    return out


# --------------------------------------------------------------------------
# 2. rate maximisation: greedy is local ascent, argmax is global
# --------------------------------------------------------------------------
def expected_accepted(p: list[float], depth: int) -> float:
    """Expected accepted draft tokens when drafting `depth` rows."""
    reach, total = 1.0, 0.0
    for i in range(depth):
        reach *= p[i]
        total += reach
    return total


def rate_argmax_depth(p: list[float], round_ms: dict[int, float],
                      cap: int) -> tuple[int, dict[int, float]]:
    """Depth maximising `(1 + expected(D)) / T(D)`, and the whole rate curve.

    Identical objective to the shipped rule. The shipped rule tests the sign of
    the first difference and stops at the first decline; this evaluates every
    reachable depth, so a cheap shelf beyond a cliff is reachable.
    """
    rates: dict[int, float] = {}
    for d in range(0, cap + 1):
        t = round_ms.get(d)
        if t is None:
            continue
        rates[d] = (1.0 + expected_accepted(p, d)) / t
    if not rates:
        return 0, rates
    best = max(rates, key=lambda d: (rates[d], -d))
    return best, rates


def break_even_deep(round_ms: dict[int, float], depths=(6, 7, 8)) -> dict:
    """At perfect acceptance, what T(D) makes depth D beat the best of 0..5?

    `rate(D) = (1+D)/T(D)` at `p = 1`, so `T(D) < (1+D)/rate(best)`.
    """
    known = {d: t for d, t in round_ms.items() if d <= 5}
    if not known:
        return {}
    rate = {d: (1.0 + d) / t for d, t in known.items()}
    best_d = max(rate, key=lambda d: rate[d])
    out = {
        "p1_rate_tokens_per_ms": rate,
        "p1_best_depth": best_d,
        "p1_best_rate": rate[best_d],
    }
    for d in depths:
        out[f"T{d}_break_even_ms"] = (1.0 + d) / rate[best_d]
    return out


# --------------------------------------------------------------------------
# 3. report
# --------------------------------------------------------------------------
MEASURED_LOCAL_P = [0.6926, 0.5840, 0.5077, 0.4190, 0.3860, 0.6875, 0.4000, 0.4000]
RANKED_MEAN_DRAFTS = [4.35, 4.89, 5.78, 5.33, 5.04]


def pool_acceptance_gap(h: float = SHIPPED_H) -> dict:
    """Acceptance a pool needs before the shipped rule drafts as deep as ranked.

    The ranked h-sweep comment records mean draft lengths of 4.35-5.78 under
    the shipped scalar rule at h=0.18. Inverting that rule's own threshold
    walk under a uniform-p assumption bounds how much easier the hidden pool
    is to draft on than this local fixture, whose per-position acceptance is
    measured directly from the forced-depth pool.
    """
    def coeff(d: int) -> float:
        return shipped_coefficient(d, h)

    implied = {}
    for target in range(1, 9):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if fires(coeff, [mid] * 8) >= target:
                hi = mid
            else:
                lo = mid
        implied[target] = hi if fires(coeff, [hi] * 8) >= target else None
    ranked_mean = sum(RANKED_MEAN_DRAFTS) / len(RANKED_MEAN_DRAFTS)
    ranked_depth = round(ranked_mean)
    return {
        "h": h,
        "implied_uniform_p_for_depth": implied,
        "local_measured_p": MEASURED_LOCAL_P,
        "local_depth_under_shipped_rule": fires(coeff, MEASURED_LOCAL_P),
        "ranked_mean_draft_lengths": RANKED_MEAN_DRAFTS,
        "ranked_mean_draft_len": ranked_mean,
        "ranked_implied_uniform_p": implied.get(ranked_depth),
        "local_p_head": MEASURED_LOCAL_P[0],
        "acceptance_gap": (
            implied[ranked_depth] - MEASURED_LOCAL_P[0]
            if implied.get(ranked_depth) is not None else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    thm = verify_theorem()
    be = break_even_deep(R1_ROUND_MS)

    print("=== 1. admissibility theorem: row d+1 admissible iff c_d < 1/(d+1) ===")
    print(f"{'d':>2} {'ceiling':>9} {'shipped':>9} {'ship ok':>8} "
          f"{'arm D':>9} {'D ok':>6}")
    for d in range(8):
        print(f"{d:>2} {admissible_ceiling(d):>9.6f} "
              f"{thm['shipped_coefficient'][d]:>9.6f} "
              f"{str(thm['shipped_admissible'][d]):>8} "
              f"{thm['arm_d_coefficient'][d]:>9.6f} "
              f"{str(thm['arm_d_admissible'][d]):>6}")
    print(f"shipped form wall-free for every h tried: "
          f"{thm['shipped_form_wall_free']}")
    print(f"arm D is a HARD CAP at depth 3: {thm['arm_d_is_hard_cap_at_depth_3']}"
          f"  (corner p=1 slack {thm['corner_p1_depth3_slack']:+.6f}, "
          f"threshold {thm['corner_p1_depth3_threshold']:.6f} > 1)")
    for fam in ("monotone", "iid"):
        print(f"  {fam:>8}: depth>=4 fires {thm[f'{fam}_depth_ge_4']} / 400000, "
              f"best depth-3 slack {thm[f'{fam}_best_depth3_slack']:+.6f}")

    print()
    print("=== 2/3. rate curve at p=1 on r1's STALE table, and break-even ===")
    for d, r in sorted(be["p1_rate_tokens_per_ms"].items()):
        print(f"  d={d} T={R1_ROUND_MS[d]:8.3f} ms  rate={r:.6f} tok/ms"
              + ("   <- best of 0..5" if d == be["p1_best_depth"] else ""))
    for d in (6, 7, 8):
        print(f"  depth {d} beats it iff T({d}) < {be[f'T{d}_break_even_ms']:.3f} ms")

    gap = pool_acceptance_gap()
    print()
    print("=== 4. pool acceptance gap: why a local depth cap need not transfer ===")
    print(f"  local measured p per position: "
          + " ".join(f"{v:.3f}" for v in MEASURED_LOCAL_P[:5]))
    print(f"  shipped rule on the local pool reaches depth "
          f"{gap['local_depth_under_shipped_rule']}")
    for target, p in sorted(gap["implied_uniform_p_for_depth"].items()):
        print(f"  depth {target} needs uniform p >= "
              + (f"{p:.4f}" if p is not None else "unreachable"))
    print(f"  ranked mean draft len {gap['ranked_mean_draft_len']:.2f} implies "
          f"uniform p ~ {gap['ranked_implied_uniform_p']:.4f}, "
          f"a gap of {gap['acceptance_gap']:+.4f} over local p0")

    if args.json:
        args.json.write_text(json.dumps(
            {"theorem": thm, "break_even": be, "pool_acceptance_gap": gap},
            indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
