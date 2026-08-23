"""E144 — the Finding 219 six-step pricing chain, driven by an INFERRED acceptance delta.

harness=local. Every number this file produces is INFERRED. R-C was not run, so
there is no measured per-prompt acceptance delta to feed the chain. Instead the
chain is driven by the acceptance recovery that `e144_price.py` infers from the
measured R-B relL2 improvement factor, and it is evaluated under two explicitly
named depth-response regimes so the advisor can read a bound rather than a point.

    regime NEUTRAL   the scheduler does not react. Draft length and the verify
                     width histogram hold, so the round cost holds and every
                     recovered acceptance point turns into tokens per round.
                     This is the best case and therefore an UPPER BOUND.

    regime SHIPPED   the scheduler reacts through its own depth ladder. A higher
                     per-round acceptance estimate lifts the realised draft
                     length, which lifts the verify row count, which is charged
                     against edward's per-width curve. This is Rule 125.

Inputs, all from advisor feedback F2 / Finding 219 on PR #145:

  * the ranked anchor table for candidate `1760479a`;
  * the shipped depth thresholds at flat p;
  * the per-width round cost curve, which F2 labels PROVISIONAL because it is
    replayed rather than measured.

The value step reuses the advisor's published `f215_lottery_price` pool and
helpers so the Rule 126 expectation is computed exactly as the advisor computes
it.
"""

import argparse
import json
import os
import random
import statistics

import f215_lottery_price as f215

HERE = os.path.dirname(os.path.abspath(__file__))

# Finding 219, comment 5383899804. Ranked anchor `1760479a`.
# prompt -> (candidate seconds per token, effective mean draft length, round count)
ANCHOR = {
    "beagle": (0.0106863, 4.3818, 110.00),
    "botany": (0.0096534, 6.1481, 81.04),
    "drama": (0.0178217, 2.2976, 252.01),
    "essays": (0.0098320, 5.0870, 92.04),
    "medicine": (0.0097197, 5.2556, 90.01),
    "plutarch": (0.0300828, 0.1557, 486.76),
    "republic": (0.0097098, 4.9892, 93.00),
    "travel": (0.0156178, 2.6479, 212.33),
}

DECODE_TOKENS = 512

# Shipped depth rule at flat p: d >= k requires p above THRESHOLDS[k - 1].
THRESHOLDS = [0.1800, 0.4742, 0.6497, 0.7527, 0.8168, 0.8591, 0.8884]

# Edward's rebuilt per-verify-row round cost, rows 1..9, microseconds.
# F2: replayed, not measured. Every derived number is provisional.
WIDTH_CURVE_US = [
    31173.2,
    34619.3,
    38065.4,
    41511.4,
    44957.5,
    61198.8,
    62824.9,
    70315.4,
    75638.9,
]

BAR = 3.71959722580154
ANCHOR_ID = "1760479a"


def accept_rate(prompt):
    """`a` from the round-count identity R = 512 / (1 + a*d)."""
    _, draft_len, rounds = ANCHOR[prompt]
    return (DECODE_TOKENS / rounds - 1.0) / draft_len


def chain_accept(step_p, width):
    """Accepted drafts per round for i.i.d. per-step acceptance over `width` drafts.

    Fractional width interpolates the geometric sum, which is what an
    effective *mean* draft length is.
    """
    if step_p >= 1.0:
        return width
    return step_p * (1.0 - step_p**width) / (1.0 - step_p)


def implied_step_p(prompt):
    """Invert the chain model for the per-step acceptance behind (a, d)."""
    _, draft_len, _ = ANCHOR[prompt]
    target = accept_rate(prompt) * draft_len
    low, high = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (low + high)
        if chain_accept(mid, draft_len) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def ladder_spacing(draft_len):
    """Width of the depth rung the realised draft length currently straddles."""
    rung = int(draft_len)
    if rung <= 0:
        return THRESHOLDS[0]
    if rung >= len(THRESHOLDS):
        return THRESHOLDS[-1] - THRESHOLDS[-2]
    return THRESHOLDS[rung] - THRESHOLDS[rung - 1]


def round_cost_us(rows):
    """Linear interpolation of the provisional per-width curve at fractional rows."""
    rows = max(1.0, min(float(len(WIDTH_CURVE_US)), rows))
    low = int(rows)
    if low >= len(WIDTH_CURVE_US):
        return WIDTH_CURVE_US[-1]
    frac = rows - low
    base = WIDTH_CURVE_US[low - 1]
    return base + frac * (WIDTH_CURVE_US[low] - base)


def price_shape(gains, bar, samples=40000, seed=20260823):
    """Rule 126: expectation of the published median over the serial lottery."""
    data = f215.load()
    order = data["order"]
    pool = [row["serial"] for row in data["serial_pool"]]
    cand0 = data["anchor_candidate_vectors"][ANCHOR_ID]
    cand1 = [c / (1.0 + gains[name] / 100.0) for c, name in zip(cand0, order)]

    random.seed(seed)
    deltas, levels = [], []
    owner_before, owner_after = {}, {}
    for _ in range(samples):
        serial = random.choice(pool)
        m0 = f215.median_pair(cand0, serial)
        m1 = f215.median_pair(cand1, serial)
        deltas.append(100.0 * (m1 / m0 - 1.0))
        levels.append(m1)
        before = f215.upper_owner(cand0, serial, order)
        after = f215.upper_owner(cand1, serial, order)
        owner_before[before] = owner_before.get(before, 0) + 1
        owner_after[after] = owner_after.get(after, 0) + 1
    deltas.sort()
    levels.sort()

    def census(counts):
        return {k: v / float(samples) for k, v in sorted(counts.items(), key=lambda t: -t[1])}

    return {
        "e144_expected_median_pct": statistics.fmean(deltas),
        "median_pct_sd": statistics.pstdev(deltas),
        "e144_median_p05_pct": deltas[int(0.05 * samples)],
        "median_pct_p95": deltas[int(0.95 * samples)],
        "expected_published_median": statistics.fmean(levels),
        "published_median_p05": levels[int(0.05 * samples)],
        "published_median_p95": levels[int(0.95 * samples)],
        "e144_p_beat_bar": sum(1 for m in levels if m > bar) / float(samples),
        "bar": bar,
        "e144_upper_slot_owner_before": census(owner_before),
        "e144_upper_slot_owner_after": census(owner_after),
    }


def run_regime(delta_pt, regime, bar):
    """The six F219 steps for one inferred per-step acceptance recovery."""
    delta_p = delta_pt / 100.0
    prompts, gains = {}, {}
    for name in ANCHOR:
        seconds, draft_len, rounds = ANCHOR[name]
        accept = accept_rate(name)
        step_p = implied_step_p(name)
        tokens_per_round = 1.0 + accept * draft_len

        # step 1: the new per-step acceptance
        new_step_p = min(1.0 - 1e-9, step_p + delta_p)

        # step 2: the depth response
        if regime == "neutral":
            new_draft_len = draft_len
        else:
            new_draft_len = draft_len + delta_p / ladder_spacing(draft_len)

        # step 3: the new round count
        new_accept = chain_accept(new_step_p, new_draft_len) / new_draft_len
        new_tokens_per_round = 1.0 + new_accept * new_draft_len
        new_rounds = DECODE_TOKENS / new_tokens_per_round

        # step 4: the new round cost, from the provisional per-width curve
        cost = round_cost_us(1.0 + draft_len)
        new_cost = round_cost_us(1.0 + new_draft_len)
        # anchor the curve on the F219 absolute round cost so only its slope is used
        measured_cost = seconds * 1e6 * tokens_per_round
        new_measured_cost = measured_cost * (new_cost / cost)

        # step 5: the new candidate seconds per token
        new_seconds = new_measured_cost / new_tokens_per_round / 1e6
        gain_pct = 100.0 * (seconds / new_seconds - 1.0)

        prompts[name] = {
            "accept_rate": accept,
            "implied_step_p": step_p,
            "e144_new_step_p": new_step_p,
            "draft_len": draft_len,
            "e144_new_draft_len": new_draft_len,
            "e144_new_accept_rate": new_accept,
            "rounds": rounds,
            "e144_new_rounds": new_rounds,
            "round_cost_us": measured_cost,
            "e144_new_round_cost_us": new_measured_cost,
            "round_cost_pct": 100.0 * (new_measured_cost / measured_cost - 1.0),
            "seconds_per_token": seconds,
            "e144_new_seconds_per_token": new_seconds,
            "candidate_leg_gain_pct": gain_pct,
        }
        gains[name] = gain_pct

    # step 6: the score, as a Rule 126 expectation
    value = price_shape(gains, bar)

    # Rule 121 read against the anchor row's own realised serial draw, for the
    # deterministic raw' table the advisor asked for. Rule 126 says the
    # expectation above is the answer; this row is the illustration.
    data = f215.load()
    order = data["order"]
    realised = None
    for row in data["serial_pool"]:
        if row["id8"] == ANCHOR_ID:
            realised = row["serial"]
    raw_before, raw_after = {}, {}
    if realised is not None:
        for serial, name in zip(realised, order):
            raw_before[name] = serial / prompts[name]["seconds_per_token"]
            raw_after[name] = serial / prompts[name]["e144_new_seconds_per_token"]

    return {
        "regime": regime,
        "regime_note": {
            "neutral": "scheduler does not react; draft length and width histogram hold. UPPER BOUND.",
            "shipped": "scheduler reacts through its own depth ladder; extra rows charged "
            "against the PROVISIONAL per-width curve. Rule 125.",
        }[regime],
        "inferred_acceptance_delta_pt": delta_pt,
        "prompts": prompts,
        "e144_new_raw_ratio": raw_after,
        "raw_ratio_before": raw_before,
        "candidate_leg_gain_pct": gains,
        "gain_shape_is_uniform": max(gains.values()) - min(gains.values()) < 1e-9,
        "value": value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta-pt",
        default="0.084,0.141,0.160",
        help="inferred per-step acceptance recovery in points, comma separated",
    )
    parser.add_argument("--bar", type=float, default=BAR)
    parser.add_argument("--out", default="e144-f219.json")
    arguments = parser.parse_args()

    deltas = [float(x) for x in arguments.delta_pt.split(",") if x.strip()]
    report = {
        "harness": "local",
        "status": "INFERRED — R-C was not run; the chain is driven by the price model",
        "anchor": ANCHOR_ID,
        "bar": arguments.bar,
        "width_curve_us": WIDTH_CURVE_US,
        "width_curve_caveat": "PROVISIONAL — replayed, not measured (advisor F2)",
        "depth_thresholds": THRESHOLDS,
        "implied_step_p": {name: implied_step_p(name) for name in sorted(ANCHOR)},
        "accept_rate": {name: accept_rate(name) for name in sorted(ANCHOR)},
        "arms": [],
    }
    for delta in deltas:
        for regime in ("neutral", "shipped"):
            report["arms"].append(run_regime(delta, regime, arguments.bar))

    with open(os.path.join(HERE, arguments.out), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("implied per-step acceptance behind the ranked anchor")
    for name in sorted(ANCHOR):
        print(
            "  %-9s a %.5f  d %.4f  p %.5f"
            % (name, accept_rate(name), ANCHOR[name][1], implied_step_p(name))
        )
    print()
    for arm in report["arms"]:
        value = arm["value"]
        print(
            "delta %+.3f pt  regime %-8s  expected median %+.4f %%  p05 %+.4f %%  "
            "published %.5f  P(beat bar) %.1f %%"
            % (
                arm["inferred_acceptance_delta_pt"],
                arm["regime"],
                value["e144_expected_median_pct"],
                value["e144_median_p05_pct"],
                value["expected_published_median"],
                100.0 * value["e144_p_beat_bar"],
            )
        )
        for name in sorted(arm["prompts"]):
            entry = arm["prompts"][name]
            print(
                "    %-9s d %.4f -> %.4f   round us %9.1f -> %9.1f (%+.3f %%)   "
                "cand leg %+.4f %%"
                % (
                    name,
                    entry["draft_len"],
                    entry["e144_new_draft_len"],
                    entry["round_cost_us"],
                    entry["e144_new_round_cost_us"],
                    entry["round_cost_pct"],
                    entry["candidate_leg_gain_pct"],
                )
            )
        print()


if __name__ == "__main__":
    main()
