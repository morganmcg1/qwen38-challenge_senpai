"""E157 R0: the marginal-depth rule priced on the measured round-cost law.

Everything here is arithmetic on measured quantities. Two inputs are measured
and nothing is fitted twice:

  * the E154 R2 round-cost law, `clean_round_us = a + b * rows`, rows = d + 1,
    refit here on whatever session directory the caller passes;
  * the per-position acceptance curve, `P(accept draft i | prefix accepted)`,
    counted from the (d, acc) pairs the same session recorded.

The shipped scheduler's threshold is
`price.marginal[d] * (1 + expected) / price.cumulative[d]`, which is the
advisor's `T(d) * (b + rollback) / C(d)` after dividing numerator and
denominator by `C(0) = a + b`. So the measured price enters the shipped walk
as

    marginal[d]   = (b + rollback) / (a + b)
    cumulative[d] = 1 + d * b / (a + b)

harness=local unless a row is explicitly tagged harness=ranked. Never an
official or ranked score.

    python3 research/e157_marginal_rule.py [SESSION_DIR ...]
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
import sys

MAX_DEPTH = 8
SHIPPED_H = 0.18
# E154 R2 accounting, harness=local, base 30d3bcce, M4 Pro 48 GiB.
E154_FIXED_TERM_US = 23421.071843078098
E154_ROW_PRICE_US = 15708.498145888343
# E154 within-d acceptance contrast at d = 7, n = 120: an accepted token makes
# the round 262.87 +- 29.44 us faster, so a rejected draft pays that much more.
E154_ROLLBACK_US = 262.87154073660724
# FINDING 286, harness=ranked: local round time / ranked round time.
LOCAL_OVER_RANKED = 2.776911336360769

# FINDING 286, harness=ranked: acceptance and live drafted width recovered
# from the ranked receipts, plus Rule 148 median weights.
RANKED_PROMPTS = [
    # name, p, live d, rule-148 weight
    ("beagle", 0.8973, 5, 0.500),
    ("republic", 0.9187, 6, 0.000),
    ("essays", 0.9222, 6, 0.447),
    ("medicine", 0.9299, 6, 0.030),
    ("botany", 0.9312, 7, 0.023),
    ("travel", 0.6248, 4, 0.000),
    ("drama", 0.5280, 4, 0.000),
]


def ols(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(residual / (n - 2) / sxx)
    return slope, intercept, se, sxy / math.sqrt(sxx * syy)


def load_rounds(paths: list[str]) -> list[dict]:
    rounds = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            for line in open(path):
                fields = dict(re.findall(r"(\w+)=([-\d.]+)", line))
                if fields and "d" in fields and "acc" in fields:
                    rounds.append({k: float(v) for k, v in fields.items()})
    return rounds


def acceptance_by_position(rounds: list[dict]) -> dict:
    """P(accept draft i | drafts 0..<i accepted and the round drafted i)."""
    out = {}
    for i in range(1, MAX_DEPTH + 1):
        at_risk = [r for r in rounds if r["d"] >= i and r["acc"] >= i - 1]
        if not at_risk:
            continue
        hit = sum(1 for r in at_risk if r["acc"] >= i)
        p = hit / len(at_risk)
        # Wilson 95 % interval, because deep positions are thin.
        n = len(at_risk)
        z = 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        out[i] = {
            "n_at_risk": n,
            "n_accepted": hit,
            "p": p,
            "wilson_low": centre - half,
            "wilson_high": centre + half,
        }
    return out


def fit_row_law(rounds: list[dict], time_key: str = "wall_us") -> dict:
    usable = [r for r in rounds if time_key in r]
    if len(usable) < 8:
        return {}
    rows = [r["d"] + 1.0 for r in usable]
    times = [r[time_key] for r in usable]
    slope, intercept, se, r = ols(rows, times)
    return {
        "n": len(usable),
        "fixed_term_us": intercept,
        "row_price_us": slope,
        "se_us": se,
        "r": r,
    }


def walk(p_at, cap: int, marginal, cumulative) -> tuple[int, list]:
    """The shipped extension walk, driven at its acceptance inputs."""
    reach, expected, depth = 1.0, 0.0, 0
    trace = []
    for _ in range(cap):
        reach *= p_at(depth)
        threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
        pays = reach > threshold
        trace.append(
            {"row": depth + 1, "reach": reach, "threshold": threshold, "pays": pays}
        )
        if not pays:
            break
        expected += reach
        depth += 1
    return depth, trace


def uniform_price(h: float) -> tuple[list, list]:
    marginal = [h] * MAX_DEPTH
    cumulative = [1.0 + i * h for i in range(MAX_DEPTH + 1)]
    return marginal, cumulative


def measured_price(a: float, b: float, rollback: float) -> tuple[list, list]:
    marginal = [(b + rollback) / (a + b)] * MAX_DEPTH
    cumulative = [1.0 + i * b / (a + b) for i in range(MAX_DEPTH + 1)]
    return marginal, cumulative


def expected_tokens(p: float, depth: int) -> float:
    return 1.0 + sum(p ** k for k in range(1, depth + 1))


def leg_ratio(p: float, depth: int, a: float, b: float, rollback: float) -> float:
    """Tokens per unit round cost at a fixed drafted width."""
    cost = a + b * (depth + 1) + (rollback if depth > 0 else 0.0) * (1.0 - p ** depth)
    return expected_tokens(p, depth) / cost


def main() -> None:
    session_globs = sys.argv[1:] or [
        ".mlxfast-private/e154/runs-r2a/*zero/rounds.txt"
    ]
    rounds = load_rounds(session_globs)
    curve = acceptance_by_position(rounds)
    refit = fit_row_law(rounds)

    a = refit.get("fixed_term_us", E154_FIXED_TERM_US)
    b = refit.get("row_price_us", E154_ROW_PRICE_US)
    rollback = E154_ROLLBACK_US

    ship_marg, ship_cum = uniform_price(SHIPPED_H)
    meas_marg, meas_cum = measured_price(a, b, rollback)

    result = {
        "experiment": "e157-r0-marginal-rule",
        "harness": "local",
        "official_or_ranked_score": False,
        "sessions": session_globs,
        "n_rounds": len(rounds),
        "e157_row_price_us": {
            "source": "E154 R2 accounting" if not refit else "refit on this session",
            "fixed_term_a_us": a,
            "row_price_b_us": b,
            "rollback_us_per_rejected_draft": rollback,
            "refit": refit,
            "h_marginal_measured": meas_marg[0],
            "h_cumulative_measured": meas_cum[1] - 1.0,
            "h_shipped": SHIPPED_H,
            "underprice_factor": meas_marg[0] / SHIPPED_H,
        },
        "e157_measured_acceptance_by_position": curve,
    }

    # Host invariance of the break-even: scale every price by the ranked/local
    # factor and check the decision is unchanged.
    ranked_marg, ranked_cum = measured_price(
        a / LOCAL_OVER_RANKED, b / LOCAL_OVER_RANKED, rollback / LOCAL_OVER_RANKED
    )
    invariance = max(
        abs(x - y) for x, y in zip(meas_marg + meas_cum, ranked_marg + ranked_cum)
    )
    result["e157_marginal_rule_breakeven"] = {
        "host_invariance_max_abs_price_difference": invariance,
        "note": "prices are ratios, so the local/ranked scale cancels exactly",
        "by_prompt": [],
    }

    for name, p, live_d, weight in RANKED_PROMPTS:
        d_meas, trace_meas = walk(lambda _i: p, 7, meas_marg, meas_cum)
        d_ship, _ = walk(lambda _i: p, 7, ship_marg, ship_cum)
        gain = (
            leg_ratio(p, d_meas, a, b, rollback) / leg_ratio(p, live_d, a, b, rollback)
            - 1.0
        )
        result["e157_marginal_rule_breakeven"]["by_prompt"].append(
            {
                "prompt": name,
                "harness": "ranked",
                "p_recovered": p,
                "live_d_recovered": live_d,
                "d_star_measured_price": d_meas,
                "d_star_shipped_price_true_p": d_ship,
                "rule148_weight": weight,
                "raw_gain_at_d_star_vs_live_pct": 100.0 * gain,
                "walk": trace_meas,
            }
        )

    weighted = sum(
        row["rule148_weight"] * row["raw_gain_at_d_star_vs_live_pct"]
        for row in result["e157_marginal_rule_breakeven"]["by_prompt"]
    ) / sum(row["rule148_weight"] for row in result["e157_marginal_rule_breakeven"]["by_prompt"])
    result["e157_marginal_rule_breakeven"]["rule148_weighted_raw_gain_pct"] = weighted

    # What the measured LOCAL acceptance curve does under both prices. This is
    # the local instrument's power check: if the local fixture sits above the
    # break-even at every position, a local timing leg cannot see the mechanism.
    if curve:
        def p_at(i: int) -> float:
            key = i + 1
            if key in curve:
                return curve[key]["p"]
            return curve[max(curve)]["p"]

        d_meas_local, trace_local = walk(p_at, 7, meas_marg, meas_cum)
        d_ship_local, _ = walk(p_at, 7, ship_marg, ship_cum)
        result["e157_local_fixture_power"] = {
            "harness": "local",
            "d_star_measured_price": d_meas_local,
            "d_star_shipped_price": d_ship_local,
            "walk_measured_price": trace_local,
            "mechanism_visible_locally": d_meas_local != d_ship_local,
        }

    # F1's request: the rule must track p rather than pin a constant.
    tracks = []
    for p in (0.8973, 0.9273, 0.5280, 0.6248, 0.9600):
        d_meas, _ = walk(lambda _i: p, 7, meas_marg, meas_cum)
        tracks.append({"p": p, "d_star_measured_price": d_meas})
    result["e157_rule_tracks_p"] = tracks

    out_dir = "research/e157-artifacts"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "e157_marginal_rule.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(f"rounds={len(rounds)} a={a:.1f} us b={b:.1f} us rollback={rollback:.1f} us")
    print(
        f"h_marginal_measured={meas_marg[0]:.4f} vs shipped {SHIPPED_H}"
        f"  ({meas_marg[0] / SHIPPED_H:.2f}x)"
    )
    print("position acceptance (local):")
    for i, row in sorted(curve.items()):
        print(
            f"  {i}  n={row['n_at_risk']:4d}  p={row['p']:.4f}"
            f"  [{row['wilson_low']:.4f}, {row['wilson_high']:.4f}]"
        )
    print("break-even, harness=ranked p:")
    for row in result["e157_marginal_rule_breakeven"]["by_prompt"]:
        print(
            f"  {row['prompt']:9s} p={row['p_recovered']:.4f} live={row['live_d_recovered']}"
            f" d*(measured)={row['d_star_measured_price']}"
            f" d*(shipped price, true p)={row['d_star_shipped_price_true_p']}"
            f" gain={row['raw_gain_at_d_star_vs_live_pct']:+.3f} %"
        )
    print(f"rule 148 weighted raw gain = {weighted:+.4f} %")
    if "e157_local_fixture_power" in result:
        power = result["e157_local_fixture_power"]
        print(
            "local fixture: d*(measured price) = "
            f"{power['d_star_measured_price']}, d*(shipped) = "
            f"{power['d_star_shipped_price']}, "
            f"mechanism visible locally = {power['mechanism_visible_locally']}"
        )
    print("rule tracks p:", [(t["p"], t["d_star_measured_price"]) for t in tracks])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
