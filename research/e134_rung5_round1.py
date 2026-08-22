"""E134 rung 5 (F6) -- round-1 excess, attributed to a statement not a section.

Zero GPU. Reads the archived E128 traces and reports, per leg:

  E_total(r) = round_us[r] - median(round_us[2..N])           for r = 1, 2, 3
  the same first-round-minus-median excess for all 11 segments
  a mid-leg control that must return approximately zero

The eleven segments come from the emitter in Qwen36MTPBlockSession.swift, which
splits draft_build into six statements so a cold cost names the statement that
pays it. host_thread_cpu_ns is carried alongside as the host/GPU discriminator
the F6 E65 question turns on.

Usage:
  python3 e134_rung5_round1.py --json e134-artifacts/rung5-round1-excess.json
"""

import argparse
import json
import os
import random
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRIV = os.path.join(ROOT, ".mlxfast-private", "e128")

# The six draft_build statements, the verify window, and the tail. round_us is
# the total and is handled separately so the split can be checked against it.
SEGMENTS = [
    "draft_build_us",
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
# draft_build_us is the sum of the six d_* statements, so it is reported but
# excluded from any additive attribution to avoid double counting.
ADDITIVE = [
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
# The E65 discriminator. Qwen36MTPBlockSession.swift:729-738 measures
# verify_build_us as ~97 % GPU wait under the shipped async ladder, and
# :742-746 records head GPU execute moving between d_submit2_us and the verify
# window with overlap. Those three counters are therefore the GPU pipeline;
# everything else is host graph construction and host tail work.
GPU_SEGMENTS = ["d_submit2_us", "verify_build_us", "eval_wall_us"]
HOST_SEGMENTS = [
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]

W83 = {
    "beagle": 0.4862,
    "medicine": 0.2508,
    "essays": 0.1598,
    "botany": 0.0124,
    "republic": 0.0100,
    "plutarch": 0.0,
    "drama": 0.0,
    "travel": 0.0,
}

# Candidate seconds per token per ranked prompt, from the F6 pricing table.
CAND_SPT = {
    "beagle": 0.01138928,
    "medicine": 0.01037385,
    "essays": 0.01041310,
    "botany": 0.01024127,
    "republic": 0.01035745,
}
DECODE_TOKENS = 512

LEG_PROMPT = {
    "beagle_a": "beagle",
    "beagle_b": "beagle",
    "botany_andrews": "botany",
    "drama_dollhouse": "drama",
    "essays_bacon": "essays",
    "essays_montaigne": "essays",
    "medicine_hippoc": "medicine",
    "medicine_hist": "medicine",
    "plutarch_lives": "plutarch",
    "republic_jowett": "republic",
    "travel_eothen": "travel",
    "benchfixture": None,
}

FIELD_RE = re.compile(r"(\w+)=(-?\d+)")


def parse_trace(path):
    """Return the per-round field dicts of one leg, ordered by round."""
    rounds = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("mtp-trace: round="):
                continue
            body = line[len("mtp-trace: ") :]
            fields = {}
            for key, value in FIELD_RE.findall(body):
                if key in ("round", "d", "acc", "host_thread_cpu_ns") or key.endswith(
                    "_us"
                ):
                    fields[key] = int(value)
            if "round_us" in fields:
                rounds.append(fields)
    rounds.sort(key=lambda item: item["round"])
    return rounds


def excess(rounds, key, target_round, baseline_rounds):
    """round[target] minus the median of the baseline population, in us."""
    values = [r[key] for r in baseline_rounds if key in r]
    if not values:
        return None
    hit = next((r for r in rounds if r["round"] == target_round), None)
    if hit is None or key not in hit:
        return None
    return hit[key] - statistics.median(values)


def analyse_leg(name, path, seed, match_width=False):
    rounds = parse_trace(path)
    if len(rounds) < 8:
        return None
    # The baseline population is every round after the first, matching the F6
    # definition. Under match_width it is narrowed to rounds that dispatch the
    # same width as round 1, which is the only valid baseline once the policy
    # varies the width.
    tail = [r for r in rounds if r["round"] >= 2]
    if match_width:
        tail = [r for r in tail if r["d"] == rounds[0]["d"]]
        if len(tail) < 3:
            return None
    out = {
        "leg": name,
        "prompt": LEG_PROMPT.get(name),
        "rounds": len(rounds),
        "baseline_rounds": len(tail),
        "width_matched_baseline": match_width,
        "round1_d": rounds[0]["d"],
        "round1_acc": rounds[0]["acc"],
        "median_round_us": statistics.median([r["round_us"] for r in tail]),
    }

    out["E_total_round1_us"] = excess(rounds, "round_us", 1, tail)
    # The decay probe needs a baseline that does not contain the rounds being
    # probed. On a short width-matched tail, rounds 2 and 3 are a large share of
    # the median and would hide their own excess.
    decay_tail = [r for r in tail if r["round"] >= 4]
    out["decay_baseline_rounds"] = len(decay_tail)
    for target in (1, 2, 3):
        out[f"E_decay_round{target}_us"] = (
            excess(rounds, "round_us", target, decay_tail)
            if len(decay_tail) >= 3
            else None
        )

    out["segments"] = {}
    for key in SEGMENTS:
        out["segments"][key] = {
            "round1_us": rounds[0].get(key),
            "median_tail_us": statistics.median(
                [r[key] for r in tail if key in r]
            )
            if any(key in r for r in tail)
            else None,
            "excess_us": excess(rounds, key, 1, tail),
        }
    out["additive_excess_us"] = sum(
        out["segments"][k]["excess_us"] or 0.0 for k in ADDITIVE
    )
    out["gpu_excess_us"] = sum(
        out["segments"][k]["excess_us"] or 0.0 for k in GPU_SEGMENTS
    )
    out["host_excess_us"] = sum(
        out["segments"][k]["excess_us"] or 0.0 for k in HOST_SEGMENTS
    )

    # Host CPU nanoseconds carried beside the wall clock. A round whose wall
    # time rises while this stays flat waited on the device rather than
    # building a graph, which is the E65 discriminator.
    out["host_cpu_round1_ns"] = rounds[0].get("host_thread_cpu_ns")
    out["host_cpu_median_tail_ns"] = statistics.median(
        [r["host_thread_cpu_ns"] for r in tail if "host_thread_cpu_ns" in r]
    )
    out["host_cpu_excess_ns"] = excess(rounds, "host_thread_cpu_ns", 1, tail)

    # Matched control with a demonstrated failing polarity: the same statistic
    # on mid-leg rounds must return approximately zero. Drawn over many rounds
    # so the control reports a distribution rather than one lucky draw.
    rng = random.Random(seed)
    mid = [r for r in rounds if 0.25 * len(rounds) <= r["round"] <= 0.75 * len(rounds)]
    if match_width:
        mid = [r for r in mid if r["d"] == rounds[0]["d"]] or mid
    draws = [
        excess(rounds, "round_us", rng.choice(mid)["round"], tail) for _ in range(200)
    ]
    out["control_midleg_mean_us"] = statistics.mean(draws)
    out["control_midleg_median_us"] = statistics.median(draws)
    out["control_midleg_p95_abs_us"] = sorted(abs(d) for d in draws)[int(0.95 * 200)]
    return out


def width_matched(name, path):
    """Shipped legs vary width, so also price round 1 against its own width."""
    rounds = parse_trace(path)
    if len(rounds) < 8:
        return None
    width = rounds[0]["d"]
    same = [r for r in rounds if r["round"] >= 2 and r["d"] == width]
    if len(same) < 3:
        return {"width": width, "n_matched": len(same), "excess_us": None}
    return {
        "width": width,
        "n_matched": len(same),
        "excess_us": rounds[0]["round_us"]
        - statistics.median([r["round_us"] for r in same]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    report = {"harness": "local", "arms": {}}
    # Forced legs hold d = 7 for every round, so the pooled tail is already
    # width matched. Shipped legs vary the width, so they are analysed against
    # a width-matched tail as well and the two are compared.
    for arm, root, match in (
        ("forced", "runs-forced", False),
        ("shipped", "runs-shipped", False),
        ("shipped_width_matched", "runs-shipped", True),
    ):
        base = os.path.join(PRIV, root)
        legs = []
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name, "trace.txt")
            if not os.path.exists(path):
                continue
            row = analyse_leg(name, path, args.seed, match_width=match)
            if row is None:
                continue
            row["width_matched"] = width_matched(name, path)
            legs.append(row)
        report["arms"][arm] = legs

    # F83-weighted round-1 excess and its ranked upper bound, ranked prompts
    # only. Weights are renormalised over the prompts that have an archived leg.
    for arm, legs in report["arms"].items():
        per_prompt = {}
        for row in legs:
            prompt = row["prompt"]
            if prompt is None or W83.get(prompt, 0.0) <= 0.0:
                continue
            per_prompt.setdefault(prompt, []).append(row["E_total_round1_us"])
        summary = {}
        total_w = sum(W83[p] for p in per_prompt)
        weighted_us = 0.0
        weighted_pct = 0.0
        for prompt, values in sorted(per_prompt.items()):
            mean_us = statistics.mean(values)
            leg_s = CAND_SPT[prompt] * DECODE_TOKENS
            pct = 100.0 * (mean_us / 1e6) / leg_s
            summary[prompt] = {
                "n_legs": len(values),
                "mean_excess_us": mean_us,
                "leg_seconds": leg_s,
                "ranked_pct_upper_bound": pct,
            }
            weighted_us += W83[prompt] / total_w * mean_us
            weighted_pct += W83[prompt] / total_w * pct
        report[f"{arm}_f83"] = {
            "per_prompt": summary,
            "weighted_excess_us": weighted_us,
            "weighted_excess_ms": weighted_us / 1000.0,
            "weighted_ranked_pct_upper_bound": weighted_pct,
        }

    print("=" * 78)
    print("E134 rung 5 (F6) -- round-1 excess. harness=local, M4 Pro, zero GPU.")
    print("=" * 78)
    for arm, legs in report["arms"].items():
        print(f"\n### {arm} legs: E_total by round, and the mid-leg control (us)\n")
        print(
            f"{'leg':<18}{'N':>5}{'nbase':>7}{'d1':>4}{'med_round':>11}"
            f"{'E@1':>10}{'dec@1':>9}{'dec@2':>9}{'dec@3':>9}"
            f"{'ctrl':>8}{'ctrl_p95':>10}"
        )
        for row in legs:
            cells = "".join(
                f"{row[f'E_decay_round{t}_us']:>9.0f}"
                if row[f"E_decay_round{t}_us"] is not None
                else f"{'n/a':>9}"
                for t in (1, 2, 3)
            )
            print(
                f"{row['leg']:<18}{row['rounds']:>5}{row['baseline_rounds']:>7}"
                f"{row['round1_d']:>4}{row['median_round_us']:>11.0f}"
                f"{row['E_total_round1_us']:>10.0f}" + cells
                + f"{row['control_midleg_median_us']:>8.0f}"
                f"{row['control_midleg_p95_abs_us']:>10.0f}"
            )

    keys = ADDITIVE
    for arm in ("forced", "shipped", "shipped_width_matched"):
        print(f"\n### Segment attribution of the round-1 excess (us), {arm}\n")
        print(f"{'leg':<18}" + "".join(f"{k.replace('_us',''):>14}" for k in keys))
        for row in report["arms"][arm]:
            print(
                f"{row['leg']:<18}"
                + "".join(f"{row['segments'][k]['excess_us']:>14.0f}" for k in keys)
            )
        print(f"{'MEAN':<18}" + "".join(
            f"{statistics.mean([r['segments'][k]['excess_us'] for r in report['arms'][arm]]):>14.0f}"
            for k in keys
        ))

    print("\n### The E65 discriminator: GPU pipeline against host build (us)\n")
    print(f"{'leg':<18}{'arm':>22}{'GPU':>10}{'host':>10}{'E_total':>10}{'GPU share':>11}")
    for arm in ("forced", "shipped", "shipped_width_matched"):
        for row in report["arms"][arm]:
            total = row["E_total_round1_us"]
            share = row["gpu_excess_us"] / total if total else float("nan")
            print(
                f"{row['leg']:<18}{arm:>22}{row['gpu_excess_us']:>10.0f}"
                f"{row['host_excess_us']:>10.0f}{total:>10.0f}{share:>11.3f}"
            )

    print("\n### Host thread CPU, round 1 against the tail median (ms)\n")
    print(f"{'leg':<18}{'arm':>9}{'cpu_r1':>10}{'cpu_med':>10}{'cpu_excess':>12}{'wall_excess':>13}")
    for arm, legs in report["arms"].items():
        for row in legs:
            print(
                f"{row['leg']:<18}{arm:>9}"
                f"{row['host_cpu_round1_ns']/1e6:>10.2f}"
                f"{row['host_cpu_median_tail_ns']/1e6:>10.2f}"
                f"{row['host_cpu_excess_ns']/1e6:>12.2f}"
                f"{row['E_total_round1_us']/1e3:>13.2f}"
            )

    print("\n### Width-matched round-1 excess (shipped legs vary width)\n")
    for arm, legs in report["arms"].items():
        for row in legs:
            wm = row["width_matched"]
            if wm and wm["excess_us"] is not None:
                print(
                    f"{row['leg']:<18}{arm:>9} d={wm['width']} "
                    f"n_matched={wm['n_matched']:>4} excess={wm['excess_us']:>9.0f} us"
                )

    for arm in ("forced", "shipped", "shipped_width_matched"):
        block = report[f"{arm}_f83"]
        print(f"\n### F83-weighted round-1 excess, {arm} legs\n")
        print(f"{'prompt':<12}{'legs':>6}{'mean_us':>12}{'leg_s':>9}{'ranked_pct':>13}")
        for prompt, row in block["per_prompt"].items():
            print(
                f"{prompt:<12}{row['n_legs']:>6}{row['mean_excess_us']:>12.0f}"
                f"{row['leg_seconds']:>9.3f}{row['ranked_pct_upper_bound']:>13.4f}"
            )
        print(
            f"\nF83-weighted E_total = {block['weighted_excess_ms']:.2f} ms"
            f"   ranked upper bound = {block['weighted_ranked_pct_upper_bound']:.4f} %"
        )

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
