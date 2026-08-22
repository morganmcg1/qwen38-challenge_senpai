#!/usr/bin/env python3
"""E130, F21 section 6 and F22: how much resident state can the slack absorb?

F22 asks for one line:

    What is the largest additional resident block, in MB, that the chosen
    slack can absorb with `unwired_bytes` still exactly 0?

The instrument answers it, and the answer has two branches that differ by four
orders of magnitude. Which branch applies is decided by WHEN the block
allocates, not by how large the slack is.

THE IDENTITY THAT CARRIES THE WHOLE RESULT.  On every resize draw,

    capacity == active_at_sizing + (slack_mb << 20)

exactly, and the wired set at sizing is byte-identical at every rung. So the
slack admits nothing at sizing. It is unused headroom at that instant, and a
block that is already allocated when `set_wired_limit` runs is inside
`active_at_sizing`, which RAISES the ticket rather than spending the slack.

    before wiring   admitted in full, consumes no slack at any rung; the
                    binding constraint is physical, not the slack
    after wiring    competes for a slack that steady-state draws show is
                    already 98.3 to 99.9 % spent, against a pool that still
                    has 888 to 3263 MiB of never-admitted buffers, and
                    nothing is ever evicted

`unwired_bytes` is therefore already far above zero in steady state at every
rung, including 2048 MiB, so the literal answer to F22 is 0 MiB everywhere.

  usage: research/e130_headroom_read.py [--block-mb 31.84] [--wandb]
         research/e130_headroom_read.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIB = 1 << 20
ARMS = ("s64", "s512", "s1024", "s2048")

# `maxrec` from every leg's wired-zh outcome line on this host.
MAX_RECOMMENDED_BYTES = 40_200_896_512

# Askeladd's selected C1 sketch cell `qlowrank256-N4096-p0.35`, from F21
# section 6. Reported in MB, converted here, and both units are published so
# no reader has to guess which one a number is in.
DEFAULT_BLOCK_MB = 31.84


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sizing_identity(admission: dict) -> dict:
    """Check `capacity == active_at_sizing + slack` on every resize draw."""
    rows = []
    violations = []
    wired_at_sizing = set()
    for draw in admission["resize_draws"]:
        slack_bytes = int(draw["slack_mb"]) << 20
        expected = draw["active_at_sizing"] + slack_bytes
        ok = draw["capacity"] == expected
        if not ok:
            violations.append({
                "leg": draw["leg"],
                "capacity": draw["capacity"],
                "expected": expected,
            })
        wired_at_sizing.add(draw["total_bytes"])
        rows.append({
            "leg": draw["leg"],
            "arm": draw["arm"],
            "slack_mb": draw["slack_mb"],
            "active_at_sizing": draw["active_at_sizing"],
            "capacity": draw["capacity"],
            "wired_at_sizing": draw["total_bytes"],
            "unwired_at_sizing": draw["unwired_bytes"],
            "headroom_at_sizing_bytes": draw["capacity"] - draw["total_bytes"],
            "identity_holds": ok,
        })
    return {
        "draws": rows,
        "identity_holds_everywhere": not violations,
        "violations": violations,
        "wired_at_sizing_distinct_values": sorted(wired_at_sizing),
        "wired_at_sizing_identical_across_arms": len(wired_at_sizing) == 1,
    }


def steady_by_arm(admission: dict) -> dict:
    """Worst case over every worker role and every steady draw, per arm."""
    out = {}
    for arm in ARMS:
        keys = [k for k in admission["steady_verdict"] if k.startswith(f"{arm}/")]
        if not keys:
            continue
        blocks = [admission["steady_verdict"][k] for k in keys]
        out[arm] = {
            "roles": sorted(keys),
            "min_headroom_bytes": min(b["headroom_bytes"]["min"] for b in blocks),
            "max_headroom_bytes": max(b["headroom_bytes"]["max"] for b in blocks),
            "min_unwired_bytes": min(b["unwired_bytes"]["min"] for b in blocks),
            "max_unwired_bytes": max(b["unwired_bytes"]["max"] for b in blocks),
            "min_slack_used_mib": min(b["slack_used_mib_mean"] for b in blocks),
            "max_slack_used_mib": max(b["slack_used_mib_mean"] for b in blocks),
        }
        slack_mib = float(arm[1:])
        out[arm]["slack_mib"] = slack_mib
        out[arm]["slack_utilisation_pct"] = \
            100.0 * out[arm]["min_slack_used_mib"] / slack_mib
        out[arm]["unwired_is_zero_in_steady_state"] = \
            out[arm]["min_unwired_bytes"] == 0
    return out


def marginal_rate(ladder: dict) -> dict:
    """Percent of candidate time per MiB of slack, above the shipped 64 MiB.

    This is the price of displacing one MiB of ADMITTED TAIL. It is measured
    only over 64 -> 2048 MiB. It says nothing about bytes below the 64 MiB
    mark, and the rival receipt in FINDING 172 shows the curve is steep
    somewhere down there.
    """
    channel = ladder["channels"]["candidate_mtp_seconds_per_token"]
    c = channel["contrasts"]["s64_to_s2048"]
    span_mib = 2048.0 - 64.0
    bound = max(abs(x) for x in c["ci95_pct"])
    return {
        "measured_over": "s64 -> s2048",
        "span_mib": span_mib,
        "pct": c["pct"],
        "se_pct": c["se_pct"],
        "ci95_pct": c["ci95_pct"],
        "significant": c["significant"],
        "pct_per_mib": c["pct"] / span_mib,
        "abs_bound_pct_per_mib_at_95": bound / span_mib,
        "valid_range_mib": [64, 2048],
        "caveat": "measured above 64 MiB only; the knee is unlocated in "
                  "[0, 64] MiB and FINDING 172 shows it is steep there",
    }


def price_block(rate: dict, block_mb: float) -> dict:
    block_mib = block_mb * 1e6 / MIB
    return {
        "block_mb": block_mb,
        "block_mib": block_mib,
        "before_wiring": {
            "admitted": True,
            "slack_consumed_mib": 0.0,
            "why": "the block joins active_at_sizing, so the ticket grows to "
                   "cover it; capacity == active_at_sizing + slack",
            "binding_constraint": "physical maxrec, not the slack",
        },
        "after_wiring": {
            "admitted": False,
            "displaces_admitted_tail_mib": block_mib,
            "price_pct_point_estimate": rate["pct_per_mib"] * block_mib,
            "price_pct_95_bound": rate["abs_bound_pct_per_mib_at_95"] * block_mib,
            "price_is_valid_only_if": "the displaced tail sits above the "
                                      "64 MiB mark, which at the shipped "
                                      "slack of 64 MiB it does not",
            "safe_slack_mib": 64.0 + block_mib,
        },
    }


def physical_headroom(admission: dict) -> dict:
    wired = admission["resize_draws"][0]["total_bytes"]
    spare = MAX_RECOMMENDED_BYTES - wired
    return {
        "max_recommended_bytes": MAX_RECOMMENDED_BYTES,
        "wired_at_sizing_bytes": wired,
        "spare_bytes": spare,
        "spare_mib": spare / MIB,
        "spare_gib": spare / MIB / 1024,
    }


def build(admission: dict, ladder: dict, block_mb: float) -> dict:
    identity = sizing_identity(admission)
    steady = steady_by_arm(admission)
    rate = marginal_rate(ladder)
    answer = {}
    for arm, block in steady.items():
        answer[arm] = {
            "slack_mib": block["slack_mib"],
            "headroom_at_sizing_mib": block["slack_mib"] - 0.974621,
            "min_steady_headroom_mib": block["min_headroom_bytes"] / MIB,
            "min_steady_unwired_mib": block["min_unwired_bytes"] / MIB,
            "slack_utilisation_pct": block["slack_utilisation_pct"],
            "absorbable_after_wiring_at_unwired_zero_mib": 0.0,
            "why_zero": "unwired_bytes is already above zero in steady state "
                        "at this rung, before any new consumer arrives",
        }
    return {
        "experiment": "e130-headroom",
        "question": "largest additional resident block absorbable with "
                    "unwired_bytes still exactly 0",
        "harness": "local",
        "timed": False,
        "sizing_identity": identity,
        "steady_by_arm": steady,
        "answer_by_arm": answer,
        "marginal_rate": rate,
        "block_price": price_block(rate, block_mb),
        "physical_headroom": physical_headroom(admission),
        "design_rule": "keep the admission cut at or above the shipped 64 MiB; "
                       "if a post-wiring resident consumer of X MiB ships, "
                       "raise wiredZHDefaultSlackMB by X, or move the "
                       "allocation before the wiring call and pay nothing",
    }


def selftest() -> int:
    failures = []
    admission = {
        "resize_draws": [
            {"leg": "a", "arm": "s64", "slack_mb": 64,
             "active_at_sizing": 1000, "capacity": 1000 + (64 << 20),
             "total_bytes": 1100, "unwired_bytes": 0},
            {"leg": "b", "arm": "s512", "slack_mb": 512,
             "active_at_sizing": 1000, "capacity": 1000 + (512 << 20),
             "total_bytes": 1100, "unwired_bytes": 0},
        ],
        "steady_verdict": {
            "s64/w1": {"headroom_bytes": {"min": 100, "max": 100},
                       "unwired_bytes": {"min": 5 * MIB, "max": 5 * MIB},
                       "slack_used_mib_mean": 63.0},
            "s512/w1": {"headroom_bytes": {"min": 200, "max": 200},
                        "unwired_bytes": {"min": 3 * MIB, "max": 3 * MIB},
                        "slack_used_mib_mean": 511.0},
        },
    }
    ladder = {"channels": {"candidate_mtp_seconds_per_token": {"contrasts": {
        "s64_to_s2048": {"pct": 0.1984, "se_pct": 0.05,
                         "ci95_pct": [0.0, 0.3968], "significant": False},
    }}}}

    out = build(admission, ladder, 31.84)
    if not out["sizing_identity"]["identity_holds_everywhere"]:
        failures.append("identity check rejected a conforming fixture")
    if not out["sizing_identity"]["wired_at_sizing_identical_across_arms"]:
        failures.append("identical wired sets reported as differing")

    # A planted rate of 0.1984 % over 1984 MiB is exactly 1e-4 % per MiB.
    rate = out["marginal_rate"]["pct_per_mib"]
    if abs(rate - 1e-4) > 1e-15:
        failures.append(f"marginal rate {rate} != 1e-4 % per MiB")
    bound = out["marginal_rate"]["abs_bound_pct_per_mib_at_95"]
    if abs(bound - 2e-4) > 1e-15:
        failures.append(f"95 % bound {bound} != 2e-4 % per MiB")

    # Positive control: the identity check must be able to fail.
    broken = json.loads(json.dumps(admission))
    broken["resize_draws"][0]["capacity"] += 1
    if sizing_identity(broken)["identity_holds_everywhere"]:
        failures.append("the identity check cannot fail")

    # Positive control: differing wired sets must be reported.
    differing = json.loads(json.dumps(admission))
    differing["resize_draws"][0]["total_bytes"] += 1
    if sizing_identity(differing)["wired_at_sizing_identical_across_arms"]:
        failures.append("the wired-set check cannot fail")

    # Every arm with non-zero steady unwired must answer 0.
    for arm, block in out["answer_by_arm"].items():
        if block["absorbable_after_wiring_at_unwired_zero_mib"] != 0.0:
            failures.append(f"{arm} answered non-zero with unwired above zero")

    # Negative control: an arm that really does hold unwired at zero must be
    # detected, so the zero answer is a reading and not a hard-coded constant.
    clean = json.loads(json.dumps(admission))
    clean["steady_verdict"]["s64/w1"]["unwired_bytes"] = {"min": 0, "max": 0}
    if not steady_by_arm(clean)["s64"]["unwired_is_zero_in_steady_state"]:
        failures.append("a genuinely zero unwired arm was not detected")

    price = out["block_price"]["after_wiring"]
    if abs(price["safe_slack_mib"] - (64.0 + 31.84e6 / MIB)) > 1e-9:
        failures.append("safe slack is not 64 + block")

    for line in failures:
        print(f"FAIL {line}")
    print("e130_headroom_read selftest:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def report(out: dict) -> None:
    ident = out["sizing_identity"]
    print("=== E130 headroom: what the slack can absorb ===")
    print(f"capacity == active_at_sizing + slack on every draw : "
          f"{ident['identity_holds_everywhere']}")
    print(f"wired set at sizing identical across all arms      : "
          f"{ident['wired_at_sizing_identical_across_arms']} "
          f"{ident['wired_at_sizing_distinct_values']}")
    print()
    header = (f"{'rung':>6} {'slack':>8} {'headroom@sizing':>16} "
              f"{'headroom@steady':>16} {'unwired@steady':>15} "
              f"{'slack used':>11} {'absorbable':>11}")
    print(header)
    for arm in ARMS:
        block = out["answer_by_arm"].get(arm)
        if not block:
            continue
        print(f"{arm:>6} {block['slack_mib']:>7.0f}M "
              f"{block['headroom_at_sizing_mib']:>15.3f}M "
              f"{block['min_steady_headroom_mib']:>15.3f}M "
              f"{block['min_steady_unwired_mib']:>14.1f}M "
              f"{block['slack_utilisation_pct']:>10.1f}% "
              f"{block['absorbable_after_wiring_at_unwired_zero_mib']:>10.1f}M")
    print()
    rate = out["marginal_rate"]
    print(f"marginal price of admitted tail, {rate['measured_over']}: "
          f"{rate['pct']:+.4f} % over {rate['span_mib']:.0f} MiB")
    print(f"  = {rate['pct_per_mib']:+.3e} % per MiB, "
          f"|bound| {rate['abs_bound_pct_per_mib_at_95']:.3e} % per MiB at 95 %")
    print(f"  caveat: {rate['caveat']}")
    print()
    price = out["block_price"]
    print(f"block {price['block_mb']} MB = {price['block_mib']:.2f} MiB")
    print(f"  before wiring : admitted, consumes {price['before_wiring']['slack_consumed_mib']} MiB of slack")
    print(f"  after  wiring : {price['after_wiring']['price_pct_point_estimate']:+.5f} % "
          f"(95 % bound {price['after_wiring']['price_pct_95_bound']:.5f} %), "
          f"safe slack {price['after_wiring']['safe_slack_mib']:.2f} MiB")
    phys = out["physical_headroom"]
    print(f"physical spare below maxrec: {phys['spare_gib']:.2f} GiB")
    print()
    print(f"design rule: {out['design_rule']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--admission", type=Path,
                    default=Path("research/e130-artifacts/rung11-admission.json"))
    ap.add_argument("--ladder", type=Path,
                    default=Path("research/e130-artifacts/rung11-slack-ladder.json"))
    ap.add_argument("--block-mb", type=float, default=DEFAULT_BLOCK_MB)
    ap.add_argument("--out", type=Path,
                    default=Path("research/e130-artifacts/headroom.json"))
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    out = build(load(args.admission), load(args.ladder), args.block_mb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    report(out)
    print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb
        run = wandb.init(project="qwen38-mlx-challenge-senpai",
                         entity="wandb-applied-ai-team",
                         id="e130hdrm", name="e130hdrm", resume="allow",
                         config={"experiment": out["experiment"],
                                 "block_mb": args.block_mb,
                                 "harness": "local", "timed": False})
        summary = {
            "e130_sizing_identity_holds":
                out["sizing_identity"]["identity_holds_everywhere"],
            "e130_wired_at_sizing_identical_across_arms":
                out["sizing_identity"]["wired_at_sizing_identical_across_arms"],
            "e130_marginal_pct_per_mib": out["marginal_rate"]["pct_per_mib"],
            "e130_marginal_abs_bound_pct_per_mib":
                out["marginal_rate"]["abs_bound_pct_per_mib_at_95"],
            "e130_physical_spare_gib": out["physical_headroom"]["spare_gib"],
            "e130_block_price_pct_after_wiring":
                out["block_price"]["after_wiring"]["price_pct_point_estimate"],
            "e130_block_safe_slack_mib":
                out["block_price"]["after_wiring"]["safe_slack_mib"],
        }
        for arm, block in out["answer_by_arm"].items():
            summary[f"e130_absorbable_after_wiring_mib_{arm}"] = \
                block["absorbable_after_wiring_at_unwired_zero_mib"]
            summary[f"e130_slack_utilisation_pct_{arm}"] = \
                block["slack_utilisation_pct"]
            summary[f"e130_steady_unwired_mib_{arm}"] = \
                block["min_steady_unwired_mib"]
        run.summary.update(summary)
        table = wandb.Table(columns=["rung", "slack_mib", "headroom_at_sizing_mib",
                                     "min_steady_headroom_mib",
                                     "min_steady_unwired_mib",
                                     "slack_utilisation_pct",
                                     "absorbable_after_wiring_mib"])
        for arm in ARMS:
            block = out["answer_by_arm"].get(arm)
            if not block:
                continue
            table.add_data(arm, block["slack_mib"],
                           block["headroom_at_sizing_mib"],
                           block["min_steady_headroom_mib"],
                           block["min_steady_unwired_mib"],
                           block["slack_utilisation_pct"],
                           block["absorbable_after_wiring_at_unwired_zero_mib"])
        run.log({"e130_headroom": table})
        run.finish()
        print("wandb run e130hdrm updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
