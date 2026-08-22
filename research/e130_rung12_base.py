#!/usr/bin/env python3
"""E130 rung 12, F21 section 4: read the fresh base level on the 35d8cf58 tree.

Two outputs, in this order:

  1. LEVEL. The absolute candidate seconds per token of an unchanged
     `35d8cf58` tree on this host, with the one-pass QMV table `{6:6, 7:7}`
     live, under the identity tuple a candidate would be measured with. This
     is the number F21 section 4 asks for, and it replaces every absolute
     figure this experiment measured at `770a3ff2`.

  2. CONTRAST. `none -> s512` on that tree. Rung 11 measured the same contrast
     at `cbf87ee8` as `+0.0179 % +/- 0.0531`. F21 section 4 notes that a
     clamped kernel changes the resident scratch footprint, so the arm could
     in principle behave differently on the new base.

The fit machinery is imported from the rung 11 ladder rather than copied, so
the two rungs cannot drift apart. Only the arm set and the leg order change.

  usage: research/e130_rung12_base.py --prefix e130-r12base [--wandb]
         research/e130_rung12_base.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e130_rung11_ladder as ladder

ARMS = ("none", "s512")
ORDER = [
    "none", "s512", "s512", "none",
    "none", "s512", "s512", "none",
    "none", "s512", "s512", "none",
]

# Leg 00 absorbs the cold start. Rung 11 leg 1 entered at 37.67 C while legs
# 2-13 entered at 58.55 to 60.94 C, and that cold leg was an arm leg, which
# biased its arm fast. This one is outside ORDER and outside ARMS, so no code
# path can pull it into the fit.
WARMUP_ARM = "warmup"
WARMUP_INDEX = 0

# `read_leg` resolves every arm through the ladder's own table, so the two arms
# this rung adds must exist there before ANY read, not only inside the fit. 64
# is the shipped `wiredZHDefaultSlackMB`, which is the slack these legs would
# have requested had the guard let them reach the request at all.
ladder.SLACK_MB.setdefault("none", 64)
ladder.SLACK_MB.setdefault(WARMUP_ARM, 64)

# Prior absolute numbers, each carried with the facts that make it comparable
# or not. Rule: never compare across a differing field without saying so.
PRIOR = {
    "rung11_s64_arm_mean_traced": {
        "seconds_per_token": 0.031965139248491106,
        "base_sha": "cbf87ee8",
        "mode": "local-iterate",
        "decode_tokens": 512,
        "traced": True,
        "one_pass_table_live": False,
        "comparable_to_rung12": False,
        "why": "traced, and measured before the one-pass QMV table reached "
               "this host",
    },
    "rung11_leg13_untraced_s512": {
        "seconds_per_token": 0.03195577,
        "base_sha": "cbf87ee8",
        "mode": "local-iterate",
        "decode_tokens": 512,
        "traced": False,
        "one_pass_table_live": False,
        "comparable_to_rung12": False,
        "why": "untraced and 512 tokens, but still pre-one-pass-table, so it "
               "differs from rung 12 in the kernel that decode dispatches",
    },
    "archive_5846b986_local_submit": {
        "seconds_per_token": 0.032103583915159106,
        "base_sha": "770a3ff2",
        "mode": "local-submit",
        "decode_tokens": 512,
        "traced": False,
        "one_pass_table_live": False,
        "comparable_to_rung12": False,
        "why": "a different local mode with real thermal gates; recorded for "
               "provenance only",
    },
}

# The trace tax rung 11 measured, so a reader can convert a traced level to an
# untraced one and see how weak that conversion is.
TRACE_TAX_PCT = 0.0410
TRACE_TAX_SE_PCT = 0.0835


def arm_level(model: dict, arm: str) -> dict:
    """Absolute level of one arm, with the standard error of that level."""
    present = model["_present"]
    if arm not in present:
        return {"arm": arm, "usable": False}
    k = len(model["columns"])
    index = present.index(arm)
    variance = model["_sigma2"] * model["_inv"][index][index]
    se = variance ** 0.5
    mean = model["arm_means_adjusted"][arm]
    t_crit = ladder.T_CRIT_95.get(model["df"], 1.96)
    return {
        "arm": arm,
        "usable": True,
        "seconds_per_token": mean,
        "se_abs": se,
        "ci95_abs": [mean - t_crit * se, mean + t_crit * se],
        "se_pct_of_level": 100.0 * se / mean if mean else None,
        "df": model["df"],
        "t_crit_95": t_crit,
    }


def read_session(root: Path, prefix: str) -> tuple[list[dict], dict]:
    legs = [ladder.read_leg(root, prefix, i + 1, arm)
            for i, arm in enumerate(ORDER)]
    warmup = ladder.read_leg(root, prefix, WARMUP_INDEX, WARMUP_ARM)
    return legs, warmup


def exactness(legs: list[dict], warmup: dict) -> dict:
    every = legs + [warmup]
    matched = [leg["tag"] for leg in every if leg.get("all_tokens_matched") is True]
    unmatched = [leg["tag"] for leg in every
                 if leg.get("all_tokens_matched") is not True]
    windows = sorted({leg.get("decode_tokens") for leg in every
                      if leg.get("decode_tokens") is not None})
    return {
        "legs_matched": matched,
        "legs_not_matched": unmatched,
        "all_legs_matched": not unmatched,
        "decode_token_windows": windows,
        "one_window_only": len(windows) == 1,
    }


def provenance(legs: list[dict], warmup: dict) -> dict:
    every = legs + [warmup]
    bases = sorted({leg.get("base_sha") for leg in every if leg.get("base_sha")})
    workers = sorted({leg.get("worker_sha256") for leg in every
                      if leg.get("worker_sha256")})
    traced = sorted({leg.get("leg_trace") for leg in every})
    slacks = {leg["tag"]: leg.get("wired_slack_mb") for leg in every}
    return {
        "base_sha_values": bases,
        "one_base_served_every_leg": len(bases) == 1,
        "worker_sha256_values": workers,
        "one_binary_served_every_leg": len(workers) == 1,
        "leg_trace_values": traced,
        "every_leg_untraced": traced == ["0"],
        "wired_slack_mb_by_leg": slacks,
    }


def build(legs: list[dict], warmup: dict) -> dict:
    saved_arms, saved_order, saved_slack = ladder.ARMS, ladder.ORDER, ladder.SLACK_MB
    ladder.ARMS = ARMS
    ladder.ORDER = ORDER
    ladder.SLACK_MB = dict(saved_slack)
    ladder.SLACK_MB.update({"none": 64, WARMUP_ARM: 64})
    try:
        model = ladder.fit(legs, "mtp_seconds_per_token")
        serial = ladder.fit(legs, "serial_seconds_per_token")
        out = {
            "experiment": "e130-rung12-fresh-base",
            "question": "what is the candidate leg time on an unchanged "
                        "35d8cf58 tree, and does the wired-slack arm still "
                        "read null there",
            "harness": "local",
            "headline_channel": "candidate_mtp_seconds_per_token",
            "order": ORDER,
            "legs": legs,
            "warmup_leg_discarded": warmup,
            "prior_absolute_numbers": PRIOR,
            "trace_tax_pct_from_rung11": TRACE_TAX_PCT,
            "trace_tax_se_pct_from_rung11": TRACE_TAX_SE_PCT,
            "exactness": exactness(legs, warmup),
            "provenance": provenance(legs, warmup),
            "thermal": ladder.thermal(legs + [warmup]),
            "safety": ladder.safety(legs, model),
            "channels": {},
        }
        for name, fitted in (("candidate_mtp_seconds_per_token", model),
                             ("serial_seconds_per_token", serial)):
            if fitted is None:
                out["channels"][name] = {"usable": False}
                continue
            block = {key: value for key, value in fitted.items()
                     if not key.startswith("_")}
            block["levels"] = {arm: arm_level(fitted, arm) for arm in ARMS}
            block["contrasts"] = {
                "none_to_s512": ladder.contrast(fitted, "none", "s512"),
            }
            out["channels"][name] = block
        if model is not None:
            level = arm_level(model, "none")
            out["fresh_base_seconds_per_token"] = level.get("seconds_per_token")
            out["fresh_base_level"] = level
        return out
    finally:
        ladder.ARMS, ladder.ORDER, ladder.SLACK_MB = saved_arms, saved_order, saved_slack


def selftest() -> int:
    """Plant a known arm effect and a known session trend, then recover both."""
    saved_arms, saved_order, saved_slack = ladder.ARMS, ladder.ORDER, ladder.SLACK_MB
    ladder.ARMS = ARMS
    ladder.ORDER = ORDER
    ladder.SLACK_MB = dict(saved_slack)
    ladder.SLACK_MB.update({"none": 64, WARMUP_ARM: 64})
    failures = []
    try:
        base = 0.032
        effect = -0.0032 * base  # exactly -0.32 %
        slope = 1.1e-7
        centre = 6.5
        legs = []
        for i, arm in enumerate(ORDER):
            index = i + 1
            value = base + (effect if arm == "s512" else 0.0)
            value += slope * (index - centre)
            legs.append({
                "tag": f"self-{index:02d}-{arm}", "index": index, "arm": arm,
                "mtp_seconds_per_token": value,
                "wired_clamped_count": 0, "wired_apply_failures": 0,
                "wired_residency_active": "true", "swapped": False,
                "all_tokens_matched": True, "decode_tokens": 512,
            })
        model = ladder.fit(legs, "mtp_seconds_per_token")
        if model is None:
            failures.append("fit returned None on a complete design")
        else:
            if model["df"] != 9:
                failures.append(f"df {model['df']} != 9")
            if abs(model["slope_per_leg"] - slope) > 1e-15:
                failures.append("planted trend not recovered")
            got = ladder.contrast(model, "none", "s512")["pct"]
            if abs(got - (-0.32)) > 1e-9:
                failures.append(f"planted arm effect {got} != -0.32 %")
            level = arm_level(model, "none")
            if abs(level["seconds_per_token"] - base) > 1e-15:
                failures.append("planted base level not recovered")
            if level["se_abs"] > 1e-12:
                failures.append("noiseless design reported a non-zero level se")

        # Negative control: a pure trend with no arm effect must read zero.
        flat = []
        for i, arm in enumerate(ORDER):
            index = i + 1
            flat.append({
                "tag": f"flat-{index:02d}-{arm}", "index": index, "arm": arm,
                "mtp_seconds_per_token": base + slope * (index - centre),
                "wired_clamped_count": 0, "wired_apply_failures": 0,
                "wired_residency_active": "true", "swapped": False,
                "all_tokens_matched": True, "decode_tokens": 512,
            })
        flat_model = ladder.fit(flat, "mtp_seconds_per_token")
        flat_pct = ladder.contrast(flat_model, "none", "s512")["pct"]
        if abs(flat_pct) > 1e-9:
            failures.append(f"pure trend leaked {flat_pct} % into the contrast")

        # The order must be exactly balanced against a linear trend, otherwise
        # the negative control above passes for the wrong reason.
        for arm in ARMS:
            mean_index = sum(i + 1 for i, a in enumerate(ORDER) if a == arm)
            count = sum(1 for a in ORDER if a == arm)
            if abs(mean_index / count - 6.5) > 1e-12:
                failures.append(f"arm {arm} is not trend balanced")

        # Positive control: the warmup leg must never enter the fit.
        if WARMUP_ARM in ARMS or WARMUP_INDEX in range(1, len(ORDER) + 1):
            failures.append("the warmup leg can reach the fit")
    finally:
        ladder.ARMS, ladder.ORDER, ladder.SLACK_MB = saved_arms, saved_order, saved_slack

    for line in failures:
        print(f"FAIL {line}")
    print("e130_rung12_base selftest:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


def report(out: dict) -> None:
    print("=== E130 rung 12: fresh base on 35d8cf58 ===")
    prov = out["provenance"]
    print(f"base_sha        {prov['base_sha_values']}  one_base={prov['one_base_served_every_leg']}")
    print(f"worker_sha256   {prov['worker_sha256_values']}  one_binary={prov['one_binary_served_every_leg']}")
    print(f"untraced legs   {prov['every_leg_untraced']}")
    ex = out["exactness"]
    print(f"exactness       all_matched={ex['all_legs_matched']}  windows={ex['decode_token_windows']}")
    th = out["thermal"]
    print(f"thermal         entry spread {th['entry_temp_spread_c']:.2f} C  "
          f"exit spread {th['exit_temp_spread_c']:.2f} C  "
          f"gate_qualified_for_timing={th['gate_qualified_for_timing']}")
    print(f"safety          all_clear={out['safety']['all_clear']}")
    print()

    channel = out["channels"].get("candidate_mtp_seconds_per_token", {})
    if not channel or channel.get("usable") is False:
        print("candidate channel unusable")
        return
    print(f"residual sd     {channel['residual_sd']:.6e} = "
          f"{channel['residual_sd_pct']:.4f} %   df {channel['df']}")
    print()
    print("LEVEL")
    for arm in ARMS:
        lvl = channel["levels"][arm]
        if not lvl.get("usable"):
            continue
        print(f"  {arm:<6} {lvl['seconds_per_token']:.8f} s/token   "
              f"se {lvl['se_abs']:.3e} ({lvl['se_pct_of_level']:.4f} %)   "
              f"CI [{lvl['ci95_abs'][0]:.8f}, {lvl['ci95_abs'][1]:.8f}]")
    print()
    print("CONTRAST")
    c = channel["contrasts"]["none_to_s512"]
    if c.get("usable"):
        print(f"  none -> s512  {c['pct']:+.4f} %  se {c['se_pct']:.4f}  "
              f"CI [{c['ci95_pct'][0]:+.4f}, {c['ci95_pct'][1]:+.4f}]  "
              f"{'SIG' if c['significant'] else 'ns'}")
    print()
    print("PRIOR ABSOLUTE NUMBERS, none of them comparable")
    for name, block in out["prior_absolute_numbers"].items():
        print(f"  {name:<34} {block['seconds_per_token']:.8f}  "
              f"base {block['base_sha']}  {block['why']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prefix", default="e130-r12base")
    ap.add_argument("--root", type=Path, default=Path("research/out"))
    ap.add_argument("--out", type=Path,
                    default=Path("research/e130-artifacts/rung12-fresh-base.json"))
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    legs, warmup = read_session(args.root, args.prefix)
    out = build(legs, warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    report(out)
    print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb
        run = wandb.init(project="qwen38-mlx-challenge-senpai",
                         entity="wandb-applied-ai-team",
                         id="e130r12", name="e130r12", resume="allow",
                         config={
                             "experiment": out["experiment"],
                             "order": ORDER,
                             "decode_tokens": 512,
                             "traced": False,
                             "harness": "local",
                         })
        channel = out["channels"].get("candidate_mtp_seconds_per_token", {})
        summary = {
            "e130_rung12_fresh_base_seconds_per_token":
                out.get("fresh_base_seconds_per_token"),
            "e130_rung12_all_legs_matched": out["exactness"]["all_legs_matched"],
            "e130_rung12_safety_all_clear": out["safety"]["all_clear"],
            "e130_rung12_entry_temp_spread_c": out["thermal"]["entry_temp_spread_c"],
        }
        if channel and channel.get("usable") is not False:
            summary["e130_rung12_residual_sd_pct"] = channel["residual_sd_pct"]
            summary["e130_rung12_df"] = channel["df"]
            level = channel["levels"]["none"]
            if level.get("usable"):
                summary["e130_rung12_base_level_se_pct"] = level["se_pct_of_level"]
            c = channel["contrasts"]["none_to_s512"]
            if c.get("usable"):
                summary["e130_rung12_none_to_s512_pct"] = c["pct"]
                summary["e130_rung12_none_to_s512_se_pct"] = c["se_pct"]
                summary["e130_rung12_none_to_s512_significant"] = c["significant"]
        run.summary.update(summary)
        table = wandb.Table(columns=["tag", "index", "arm", "slack_mb",
                                     "mtp_seconds_per_token",
                                     "serial_seconds_per_token",
                                     "gpu_temp_entry_c", "gpu_temp_exit_c",
                                     "all_tokens_matched", "base_sha"])
        for leg in legs + [warmup]:
            table.add_data(leg["tag"], leg["index"], leg["arm"],
                           leg.get("slack_mb"),
                           leg.get("mtp_seconds_per_token"),
                           leg.get("serial_seconds_per_token"),
                           leg.get("gpu_temp_entry_c"),
                           leg.get("gpu_temp_exit_c"),
                           leg.get("all_tokens_matched"),
                           leg.get("base_sha"))
        run.log({"e130_rung12_legs": table})
        run.finish()
        print("wandb run e130r12 updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
