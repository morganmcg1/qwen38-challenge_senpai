#!/usr/bin/env python3
"""E136 interim: the error bar on FINDING 170's null and its detection floor.

E133 reported that a PERFECT draft readout has a measured realised acceptance
value of zero or below on all four strata. Two closed levers rest on that null,
and it was published without an error bar. This module supplies one.

THE STATISTIC IS A PAIRED SIGN TEST, NOT A RATE. On every live missed row the
shipped chain emitted one token and the true affine-4 argmax would have emitted
a different one, so exactly one of three things happens:

    d = +1  the true argmax matches the target and the shipped token does not
    d = -1  the shipped token matches the target and the true argmax does not
    d =  0  neither matches

`perfect_readout_acceptance_gain` is `sum(d) / n` over all `n` sampled rows in
the stratum, so `b = #(d=+1)` and `c = #(d=-1)` are the only random quantities.
Concordant rows contribute exactly zero and add no variance.

Three floors are reported, because they answer three different questions.

  se_null       The standard error of the estimator under the null, using the
                measured discordant background `D = b + c`: `sqrt(D)/n`. A
                two-sided 2 sigma band is `+/- 2 sqrt(D)/n`.
  floor_pure    The smallest TRUE effect the protocol could have resolved at
                2 sigma, for an alternative that converts rejected rows into
                accepted rows at rate `g` on top of the measured background.
                Solving `g = 2 sqrt((D/n + g)/n)` gives `g = (2 + 2 sqrt(1+D))/n`
                exactly. At `D = 0` it reduces to `4/n`, the familiar
                four-expected-events rule for a Poisson counting measurement.
  sign_test     The conditional exact statement. Given `D` discordant pairs,
                `b ~ Binomial(D, 1/2)` under the null, so no split of fewer
                than six discordant pairs can reach a two-sided p <= 0.0455.

The i.i.d. multinomial standard error is reported beside a seed-clustered one.
Draft rows inside one trajectory are not independent, so the clustered figure is
the honest one wherever the two disagree.

Usage:
    python3 research/e133_screen.py attrib --batch 32 --per-seed \\
        --out research/e136-attrib-perseed.json
    python3 research/e136_null_floor.py --out research/e136-null-floor.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "research" / "e136-attrib-perseed.json"
DEFAULT_OUT = ROOT / "research" / "e136-null-floor.json"

# Finding 69 after F1.7: one unit of per-row acceptance delta is worth this
# many percent of the ranked candidate leg.
MISS_TO_SCORE_PCT = 203.0

# The two statistics that campaign rule 107 prices on. Each is (gain counter,
# loss counter, denominator label).
STATISTICS = {
    "perfect_readout": (
        "base_miss_live_true_is_target",
        "base_miss_live_shipped_is_target",
        "base_miss_live",
    ),
    "sketch_arm_incremental": (
        "sk_net_miss_live_arm_is_target",
        "sk_net_miss_live_shipped_is_target",
        "sk_net_miss_live",
    ),
}

GATING = ("beagle", "min_carriers")
REPORT_STRATA = ("beagle", "min_carriers", "zero_weight", "essays_bacon")


def sign_test_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for `b` successes in `b + c` fair trials."""
    d = b + c
    if d == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(d, i) for i in range(k + 1)) / (2.0 ** d)
    return min(1.0, 2.0 * tail)


def min_resolvable_split(d: int, alpha: float = 0.0455) -> int | None:
    """Smallest `|b - c|` at `D = d` that reaches two-sided `p <= alpha`."""
    for c in range(d // 2, -1, -1):
        if sign_test_p(d - c, c) <= alpha:
            return d - 2 * c
    return None


def summarise(n: int, b: int, c: int) -> dict:
    d = b + c
    delta = (b - c) / n
    var_iid = (d - (b - c) ** 2 / n) / n ** 2 if n else float("nan")
    se_iid = math.sqrt(max(var_iid, 0.0))
    se_null = math.sqrt(d) / n
    floor_pure = (2.0 + 2.0 * math.sqrt(1.0 + d)) / n
    split = min_resolvable_split(d)
    return {
        "n": n, "gain_events_b": b, "loss_events_c": c, "discordant_D": d,
        "acceptance_delta": delta,
        "acceptance_delta_pp": 100.0 * delta,
        "ranked_pct": MISS_TO_SCORE_PCT * delta,
        "se_iid": se_iid,
        "se_null": se_null,
        "two_sigma_band_null": 2.0 * se_null,
        "two_sigma_band_null_ranked_pct": MISS_TO_SCORE_PCT * 2.0 * se_null,
        "floor_pure_gain": floor_pure,
        "floor_pure_gain_pp": 100.0 * floor_pure,
        "floor_pure_gain_ranked_pct": MISS_TO_SCORE_PCT * floor_pure,
        "sign_test_p": sign_test_p(b, c),
        "sign_test_min_resolvable_abs_split": split,
        "sign_test_resolvable_at_this_D": split is not None,
        "null_survives_2sigma": abs(delta) <= 2.0 * se_null,
    }


def structural_bounds(src: dict, counters: dict, sizes: dict) -> dict:
    """The two closed levers do not need the sampling floor at all.

    A lever can only change acceptance on a row whose EMITTED TOKEN changes.
    That count is observed, not estimated, so it bounds the attainable gain
    above any statistical question. Where the count is zero the null is
    structural: the intervention is the identity map on this corpus.
    """
    out: dict = {"k_ladder": {}, "probe_at_0_25": {}}
    for k, block in src["k_curve"].items():
        rows = {}
        for s in GATING:
            r = block["by_stratum"].get(s)
            if r is None:
                continue
            n = sizes[s]
            rows[s] = {
                "n": n,
                "live_rows_whose_emitted_token_changes": r["swapped_live"],
                "max_attainable_gain": r["swapped_live"] / n,
                "max_attainable_ranked_pct":
                    MISS_TO_SCORE_PCT * r["swapped_live"] / n,
                "observed_gain": r["acceptance_gain_realised"],
                "observed_ranked_pct":
                    MISS_TO_SCORE_PCT * r["acceptance_gain_realised"],
                "structural_zero": r["swapped_live"] == 0,
            }
        out["k_ladder"][k] = rows
    # Raising the probe fraction can only recover a row the shipped chain fails
    # to probe. Zero such rows were seen, so the rule of three bounds the rate.
    for s in REPORT_STRATA:
        if s not in sizes:
            continue
        n = sizes[s]
        unprobed = n - counters.get(s, {}).get("probe_hit_a2", 0)
        bound = unprobed / n if unprobed else 3.0 / n
        out["probe_at_0_25"][s] = {
            "n": n,
            "argmax_rows_outside_the_probed_set": unprobed,
            "rule_of_three_95pct_upper_bound_on_the_rate": bound,
            "max_attainable_ranked_pct_from_raising_p":
                MISS_TO_SCORE_PCT * bound,
            "bound_is_rule_of_three": unprobed == 0,
        }
    n = sum(sizes[s] for s in ("beagle", "min_carriers", "zero_weight"))
    unprobed = sum(n_s - counters.get(s, {}).get("probe_hit_a2", 0)
                   for s, n_s in ((s, sizes[s]) for s in
                                  ("beagle", "min_carriers", "zero_weight")))
    out["probe_at_0_25"]["pool:corpus"] = {
        "n": n,
        "argmax_rows_outside_the_probed_set": unprobed,
        "rule_of_three_95pct_upper_bound_on_the_rate":
            unprobed / n if unprobed else 3.0 / n,
        "max_attainable_ranked_pct_from_raising_p":
            MISS_TO_SCORE_PCT * (unprobed / n if unprobed else 3.0 / n),
        "bound_is_rule_of_three": unprobed == 0,
    }
    return out


def cluster_se(units: list[tuple[int, int, int]]) -> dict:
    """Seed-clustered standard error of `sum(d) / sum(n)`.

    `units` is a list of `(n_j, b_j, c_j)` per seed. The estimator is a ratio of
    sums, so the influence function of seed `j` is `(b_j - c_j) - delta * n_j`.
    """
    n = sum(u[0] for u in units)
    if not n:
        return {"clusters": 0}
    delta = sum(u[1] - u[2] for u in units) / n
    g = len(units)
    ss = sum(((u[1] - u[2]) - delta * u[0]) ** 2 for u in units)
    # A single cluster leaves no between-cluster residual, so the estimate is
    # degenerate rather than small. Report it as absent.
    se = (math.sqrt(g / (g - 1) * ss) / n) if g > 1 else None
    return {
        "clusters": g,
        "clusters_carrying_a_discordant_event":
            sum(1 for u in units if (u[1] + u[2]) > 0),
        "clusters_carrying_a_net_effect":
            sum(1 for u in units if (u[1] - u[2]) != 0),
        "se_clustered": se,
        "two_sigma_band_clustered": 2.0 * se if se is not None else None,
        "two_sigma_band_clustered_ranked_pct":
            MISS_TO_SCORE_PCT * 2.0 * se if se is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = json.loads(Path(args.attrib).read_text())
    raw = src["raw_counters"]
    counters, sizes, seed_of = raw["c"], raw["n"], raw["seed_of"]
    assert src["miss_to_score_pct"] == MISS_TO_SCORE_PCT, src["miss_to_score_pct"]

    out = {
        "source": args.attrib,
        "base_sha": src["base_sha"],
        "samples": src["samples"],
        "miss_to_score_pct": MISS_TO_SCORE_PCT,
        "harness": "offline",
        "timing_valid": False,
        "official_or_ranked_score": False,
        "statistics": {},
    }

    for stat, (gain_key, loss_key, live_key) in STATISTICS.items():
        rows = {}
        for s in REPORT_STRATA:
            if s not in sizes:
                continue
            cs = counters.get(s, {})
            row = summarise(sizes[s], cs.get(gain_key, 0), cs.get(loss_key, 0))
            row["live_missed_rows"] = cs.get(live_key, 0)
            row["gating"] = s in GATING
            units = [
                (sizes[seed], counters.get(seed, {}).get(gain_key, 0),
                 counters.get(seed, {}).get(loss_key, 0))
                for seed, stratum in seed_of.items()
                if stratum == s or seed == s
            ]
            row.update(cluster_se(units))
            row["per_seed"] = {
                seed: {
                    "n": sizes[seed],
                    "b": counters.get(seed, {}).get(gain_key, 0),
                    "c": counters.get(seed, {}).get(loss_key, 0),
                }
                for seed, stratum in sorted(seed_of.items())
                if (stratum == s or seed == s)
                and (counters.get(seed, {}).get(gain_key, 0)
                     or counters.get(seed, {}).get(loss_key, 0))
            }
            rows[s] = row

        # Pooled over the two gating strata and over the whole corpus. The
        # corpus pool double counts nothing: `essays_bacon` is a seed inside
        # `min_carriers`, so it is excluded from the pool and reported beside it.
        pools = {}
        for label, members in (("gating", GATING),
                               ("corpus", ("beagle", "min_carriers",
                                           "zero_weight"))):
            n = sum(sizes[s] for s in members if s in sizes)
            b = sum(counters.get(s, {}).get(gain_key, 0) for s in members)
            c = sum(counters.get(s, {}).get(loss_key, 0) for s in members)
            pool = summarise(n, b, c)
            units = [
                (sizes[seed], counters.get(seed, {}).get(gain_key, 0),
                 counters.get(seed, {}).get(loss_key, 0))
                for seed, stratum in seed_of.items() if stratum in members
            ]
            pool.update(cluster_se(units))
            pools[label] = pool
        out["statistics"][stat] = {"by_stratum": rows, "pooled": pools}

    out["structural_bounds"] = structural_bounds(src, counters, sizes)
    Path(args.out).write_text(json.dumps(out, indent=2))

    for stat, block in out["statistics"].items():
        print(f"\n=== {stat}: measured delta against its own detection floor")
        print(f"{'stratum':16s}{'n':>7s}{'b':>4s}{'c':>4s}"
              f"{'delta':>12s}{'seNull':>11s}{'seClust':>11s}"
              f"{'2sigma':>11s}{'floorPure':>11s}{'floor%':>9s}{'null ok':>9s}")
        for s, r in block["by_stratum"].items():
            print(f"{s:16s}{r['n']:7d}{r['gain_events_b']:4d}"
                  f"{r['loss_events_c']:4d}{r['acceptance_delta']:12.4e}"
                  f"{r['se_null']:11.4e}{(r["se_clustered"] if r["se_clustered"] is not None else float("nan")):11.4e}"
                  f"{r['two_sigma_band_null']:11.4e}"
                  f"{r['floor_pure_gain']:11.4e}"
                  f"{r['floor_pure_gain_ranked_pct']:9.3f}"
                  f"{'yes' if r['null_survives_2sigma'] else 'NO':>9s}")
        for label, r in block["pooled"].items():
            print(f"{'pool:' + label:16s}{r['n']:7d}{r['gain_events_b']:4d}"
                  f"{r['loss_events_c']:4d}{r['acceptance_delta']:12.4e}"
                  f"{r['se_null']:11.4e}{r.get('se_clustered', 0.0):11.4e}"
                  f"{r['two_sigma_band_null']:11.4e}"
                  f"{r['floor_pure_gain']:11.4e}"
                  f"{r['floor_pure_gain_ranked_pct']:9.3f}"
                  f"{'yes' if r['null_survives_2sigma'] else 'NO':>9s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
