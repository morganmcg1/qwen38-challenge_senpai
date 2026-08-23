#!/usr/bin/env python3
"""E150 R3 phase 2. Price the host round trip a sequential stop rule must pay.

The contrast is absolute microseconds per round between the control and each
arm, paired inside an ABBA block so monotone thermal drift cancels to first
order. Round count is identical across arms by construction (phase 1 proves
it), so microseconds per round and microseconds per token carry the same
information here and both are reported.

TWO ESTIMATES OF ONE SCALAR. `firstOnly` pays one round trip per round and
`perStep` pays d, so

    cost_per_readback = delta(firstOnly)
    cost_per_readback = delta(perStep) / mean_d

are independent estimates of the same quantity from the same session. Their
ratio is a consistency check the experiment gets for free: if the two disagree
badly, the arm is measuring something other than the round trip, for example
lost overlap between the head chain and the device rather than sync latency.
That distinction matters, because lost overlap is a cost a smarter sequential
design could recover and pure latency is not.

THE DECISION. R2 priced the break-even host readback at 775.0 us/round at the
PAD operating point, and the pre-registered stop rule is 100 us/round. The
report states which side of both the measurement lands on. It does not restate
the R2 gain, because Rule 79 forbids publishing a schedule-policy contrast from
a local timing leg and the R2 number is an offline replay, not a leg.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

BREAK_EVEN_US_PER_ROUND = 775.0
STOP_RULE_US_PER_ROUND = 100.0
CONTROL = "off"


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def leg_metrics(leg: pathlib.Path) -> dict | None:
    report = leg / "report.json"
    meta = read_meta(leg / "meta.txt")
    if not report.exists():
        return None
    r = json.loads(report.read_text())
    rounds = r.get("round_count")
    secs = r.get("decode_seconds")
    if not rounds or not secs:
        return None
    return {
        "us_per_round": secs * 1e6 / rounds,
        "s_per_token": r.get("parent_measured_seconds_per_token"),
        "round_count": rounds,
        "decode_seconds": secs,
        "edl": r.get("effective_mean_draft_len"),
        "accept": r.get("accepted_draft_rate"),
        "matched": meta.get("all_tokens_matched"),
        "timing_valid": meta.get("timing_valid"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "arm_requested": meta.get("e150_r3_arm_requested"),
        "position": meta.get("e150_r3_position"),
        "worker_sha256": meta.get("worker_sha256"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--witness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument(
        "--prompts", default="beagle_a,essays_montaigne")
    args = ap.parse_args()

    root = pathlib.Path(args.runs)
    cands = [c for c in args.candidates.split(",") if c]
    prompts = [p for p in args.prompts.split(",") if p]

    witness = {}
    wpath = pathlib.Path(args.witness)
    if wpath.exists():
        witness = json.loads(wpath.read_text())

    out: dict = {
        "e150_r3_harness": "local",
        "e150_r3_gpu_used": True,
        "e150_r3_cool_gate_passed_real_gate": False,
        "e150_r3_gate_qualified_for_timing": False,
        "e150_r3_break_even_us_per_round": BREAK_EVEN_US_PER_ROUND,
        "e150_r3_stop_rule_us_per_round": STOP_RULE_US_PER_ROUND,
        "e150_r3_witness_ok": witness.get("e150_r3_witness_ok"),
        "e150_r3_schedule_identical": witness.get(
            "e150_r3_schedule_identical"),
        "blocks": [],
        "problems": [],
    }

    mean_d = None
    if witness.get("arms", {}).get(CONTROL):
        a = witness["arms"][CONTROL]
        mean_d = a["d_total"] / max(1, a["rounds"])
    out["e150_r3_mean_drafts_per_round"] = mean_d

    per_cand: dict[str, list[float]] = {c: [] for c in cands}
    per_cand_tok: dict[str, list[float]] = {c: [] for c in cands}
    temps: list[float] = []

    for rep in range(args.first, args.first + args.replicates):
        for cand in cands:
            for prompt in prompts:
                arms = [CONTROL, cand, cand, CONTROL]
                legs = []
                for position, arm in enumerate(arms, start=1):
                    slot = f"{args.label}k{rep}{cand}p{position}{arm}"
                    m = leg_metrics(root / slot / prompt)
                    if m is None:
                        out["problems"].append(f"missing leg {slot}/{prompt}")
                    else:
                        m["slot"] = slot
                        m["arm"] = arm
                        if m["matched"] != "true":
                            out["problems"].append(
                                f"{slot}/{prompt} all_tokens_matched="
                                f"{m['matched']}")
                        if m["timing_valid"] != "true":
                            out["problems"].append(
                                f"{slot}/{prompt} timing_valid="
                                f"{m['timing_valid']}")
                        if m["gpu_temp_entry_c"]:
                            temps.append(float(m["gpu_temp_entry_c"]))
                    legs.append(m)
                if any(m is None for m in legs):
                    continue

                # ABBA pairing: mean of the two control legs against the mean
                # of the two arm legs. Both arms then sit at mean position 2.5.
                ctrl = statistics.fmean(
                    [legs[0]["us_per_round"], legs[3]["us_per_round"]])
                armv = statistics.fmean(
                    [legs[1]["us_per_round"], legs[2]["us_per_round"]])
                ctrl_t = statistics.fmean(
                    [legs[0]["s_per_token"], legs[3]["s_per_token"]])
                arm_t = statistics.fmean(
                    [legs[1]["s_per_token"], legs[2]["s_per_token"]])
                rc = {m["round_count"] for m in legs}
                if len(rc) != 1:
                    out["problems"].append(
                        f"rep{rep} {cand} {prompt}: round_count differs across"
                        f" arms {sorted(rc)}; the arms did not run the same"
                        " schedule")

                delta = armv - ctrl
                per_cand[cand].append(delta)
                per_cand_tok[cand].append((arm_t - ctrl_t) * 1e6)
                out["blocks"].append({
                    "replicate": rep,
                    "candidate": cand,
                    "prompt": prompt,
                    "round_count": sorted(rc),
                    "control_us_per_round": ctrl,
                    "arm_us_per_round": armv,
                    "delta_us_per_round": delta,
                    "delta_us_per_token": (arm_t - ctrl_t) * 1e6,
                    "within_arm_spread_us": abs(
                        legs[1]["us_per_round"] - legs[2]["us_per_round"]),
                    "within_control_spread_us": abs(
                        legs[0]["us_per_round"] - legs[3]["us_per_round"]),
                    "legs": [
                        {k: m[k] for k in (
                            "slot", "arm", "us_per_round", "round_count",
                            "edl", "gpu_temp_entry_c", "gpu_temp_exit_c")}
                        for m in legs
                    ],
                })

    summary = {}
    for cand in cands:
        vals = per_cand[cand]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        se = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
        summary[cand] = {
            "blocks": len(vals),
            "delta_us_per_round_mean": mean,
            "delta_us_per_round_sd": sd,
            "delta_us_per_round_se": se,
            "delta_us_per_round_values": vals,
            "delta_us_per_token_mean": statistics.fmean(per_cand_tok[cand]),
        }
    out["e150_r3_summary"] = summary

    fo = summary.get("firstOnly", {}).get("delta_us_per_round_mean")
    ps = summary.get("perStep", {}).get("delta_us_per_round_mean")
    out["e150_r3_cost_per_readback_from_firstonly_us"] = fo
    out["e150_r3_cost_per_readback_from_perstep_us"] = (
        ps / mean_d if ps is not None and mean_d else None)
    if fo and ps and mean_d:
        out["e150_r3_estimate_ratio"] = (ps / mean_d) / fo
        out["e150_r3_estimates_agree"] = 0.5 <= out["e150_r3_estimate_ratio"] <= 2.0

    # The decision uses the cheapest design that can implement a stop rule,
    # which is firstOnly. perStep bounds the most expensive one.
    decisive = fo
    out["e150_r3_decision_cost_us_per_round"] = decisive
    if decisive is not None:
        out["e150_r3_below_stop_rule"] = decisive < STOP_RULE_US_PER_ROUND
        out["e150_r3_below_break_even"] = decisive < BREAK_EVEN_US_PER_ROUND
        out["e150_r3_headroom_multiple"] = (
            BREAK_EVEN_US_PER_ROUND / decisive if decisive > 0 else None)
        if decisive >= BREAK_EVEN_US_PER_ROUND:
            out["e150_r3_verdict"] = (
                "the round trip costs more than the sequential rule can earn;"
                " the L5/L6 axis is closed on this host")
        elif decisive >= STOP_RULE_US_PER_ROUND:
            out["e150_r3_verdict"] = (
                "the round trip is affordable but not free; a sequential rule"
                " must clear the measured cost, not the modelled gain")
        else:
            out["e150_r3_verdict"] = (
                "the round trip is below the pre-registered stop rule; cost is"
                " not what blocks the sequential axis")

    if temps:
        out["e150_r3_entry_temp_c_min"] = min(temps)
        out["e150_r3_entry_temp_c_max"] = max(temps)
        out["e150_r3_entry_temp_c_spread"] = max(temps) - min(temps)

    if not out["e150_r3_witness_ok"]:
        out["problems"].append(
            "the phase 1 witness did not pass; RULE 79 clearance is not"
            " established and this contrast must not be published")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("-- e150 r3 readback cost --")
    print(f"  mean drafts per round {mean_d}")
    for cand, s in summary.items():
        print(
            f"  {cand:10s} blocks {s['blocks']}"
            f"  delta {s['delta_us_per_round_mean']:+9.1f} us/round"
            f"  sd {s['delta_us_per_round_sd']:8.1f}"
            f"  ({s['delta_us_per_token_mean']:+8.1f} us/token)")
    print(f"  cost per readback, from firstOnly  {fo}")
    print(
        "  cost per readback, from perStep    "
        f"{out['e150_r3_cost_per_readback_from_perstep_us']}")
    print(f"  estimate ratio {out.get('e150_r3_estimate_ratio')}")
    print(f"  break-even {BREAK_EVEN_US_PER_ROUND}  stop rule"
          f" {STOP_RULE_US_PER_ROUND}")
    print(f"  VERDICT {out.get('e150_r3_verdict')}")
    for p in out["problems"]:
        print("  PROBLEM", p)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
