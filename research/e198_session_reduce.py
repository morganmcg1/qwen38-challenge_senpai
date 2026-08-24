#!/usr/bin/env python3
"""Reduce one E198 ABBA session to the terminal Stage 4 decision.

harness=local. Every number here is a within-session relative measurement on
one M4 Pro host. It is not a ranked score and it is not comparable with
sandboxed history: the session lifts the runtime-worker sandbox in every leg so
the RULE 391(b) arm witness can be written at all.

Two statistics are reported.

  ratio       serial_seconds_per_token / mtp_seconds_per_token, i.e. the
              score.json `mtp_decode_speedup`. PRIMARY. The fused kernel is
              reachable only from the `6 <= qL <= 9` causal branch in
              `attentionWithCacheUpdate`. Serial decode runs qL == 1 and
              prefill runs qL == tokens, so neither arm can change the serial
              leg. The serial leg is therefore an untouched within-leg control
              that divides out slow thermal drift.

  absolute    mtp_seconds_per_token. SECONDARY, reported because program.md
              requires matched absolute candidate MTP time whenever a local
              ratio is used.

Both are converted to ms/round with

    ms_per_round = seconds_per_token * effective_mean_draft_len * 1000

because the census, the MUE, and the advisor's stop rule are all stated per
round rather than per token.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

# Advisor's terminal Stage 4 rule: promote when recovery - 2 sigma >= this.
PROMOTION_MS_PER_ROUND = 0.5


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k] = v
    return meta


def load_leg(leg_dir: pathlib.Path) -> dict | None:
    meta_path = leg_dir / "meta.txt"
    score_path = leg_dir / "score.json"
    if not meta_path.exists() or not score_path.exists():
        return None
    meta = read_meta(meta_path)
    m = json.load(score_path.open())["metrics"]
    edl = m["effective_mean_draft_len"]
    mtp = m["mtp_seconds_per_token"]
    serial = m["serial_seconds_per_token"]
    return {
        "leg": int(meta.get("e198_leg", 0)),
        "arm": meta.get("e198_arm", "?"),
        "rows": meta.get("e198_rows", "?"),
        "exit": meta.get("exit"),
        "arm_witness": meta.get("arm_witness", "MISSING"),
        "worker_sha256_before": meta.get("worker_sha256_before"),
        "worker_digest_stable": meta.get("worker_digest_stable"),
        "worker_sandbox": meta.get("worker_sandbox"),
        "head_provenance_sha256": meta.get("head_provenance_sha256"),
        "gpu_temp_entry": meta.get("gpu_temp_entry"),
        "gpu_temp_exit": meta.get("gpu_temp_exit"),
        "decode_tokens": m["decode_tokens"],
        "mtp_depth": m["mtp_depth"],
        "all_tokens_matched": m["all_tokens_matched"],
        "residual_divergence_count": m["residual_divergence_count"],
        "effective_mean_draft_len": edl,
        "accepted_draft_rate": m["accepted_draft_rate"],
        "serial_seconds_per_token": serial,
        "mtp_seconds_per_token": mtp,
        "mtp_decode_speedup": m["mtp_decode_speedup"],
        "mtp_ms_per_round": mtp * edl * 1000.0,
        "serial_ms_per_round_equivalent": serial * edl * 1000.0,
    }


def witness_verdict(legs: list[dict]) -> dict:
    """RULE 391(b). A contrast without a positive arm witness is VOID."""
    problems = []
    fused_served, off_served = [], []
    for leg in legs:
        w = leg["arm_witness"]
        if w == "MISSING":
            problems.append(f"leg {leg['leg']} ({leg['arm']}): witness file missing")
        elif leg["arm"] == "FUSED":
            if not w.startswith("served:") or w == "served:none":
                problems.append(
                    f"leg {leg['leg']} FUSED: no served rows, witness={w!r}")
            else:
                fused_served.append(w)
        elif leg["arm"] == "OFF":
            if w != "served:none":
                problems.append(
                    f"leg {leg['leg']} OFF: fused kernel served rows, witness={w!r}")
            else:
                off_served.append(w)
    return {
        "rule": "391b",
        "positive_arm_witness_present": not problems,
        "problems": problems,
        "fused_witnesses": fused_served,
        "off_witnesses": off_served,
    }


def integrity_verdict(legs: list[dict]) -> dict:
    problems = []
    digests = {leg["worker_sha256_before"] for leg in legs}
    if len(digests) != 1:
        problems.append(f"worker digest moved across legs: {sorted(digests)}")
    heads = {leg["head_provenance_sha256"] for leg in legs}
    if len(heads) != 1:
        problems.append(f"head provenance moved across legs: {sorted(heads)}")
    for leg in legs:
        if leg["exit"] != "0":
            problems.append(f"leg {leg['leg']} exited {leg['exit']}")
        if leg["worker_digest_stable"] != "true":
            problems.append(f"leg {leg['leg']} worker digest moved during the leg")
        if not leg["all_tokens_matched"]:
            problems.append(f"leg {leg['leg']} token mismatch")
        if leg["residual_divergence_count"] != 0:
            problems.append(
                f"leg {leg['leg']} residual_divergence_count="
                f"{leg['residual_divergence_count']}")
    return {
        "clean": not problems,
        "problems": problems,
        "worker_sha256": sorted(digests)[0] if len(digests) == 1 else None,
        "head_provenance_sha256": sorted(heads)[0] if len(heads) == 1 else None,
    }


def contrast(off: list[float], fused: list[float], lower_is_better: bool) -> dict:
    """Difference of two independent arm means, with a 2-sigma interval.

    sigma is the standard error of the DIFFERENCE, propagated from each arm's
    sample standard deviation. With four legs per arm this is a wide interval;
    the reported `min_detectable_effect_2sigma` says so explicitly instead of
    letting a null be read as an absence of effect.
    """
    n_off, n_fused = len(off), len(fused)
    mean_off, mean_fused = statistics.fmean(off), statistics.fmean(fused)
    sd_off = statistics.stdev(off) if n_off > 1 else float("nan")
    sd_fused = statistics.stdev(fused) if n_fused > 1 else float("nan")
    se = math.sqrt(sd_off**2 / n_off + sd_fused**2 / n_fused)
    delta = (mean_off - mean_fused) if lower_is_better else (mean_fused - mean_off)
    return {
        "n_off": n_off,
        "n_fused": n_fused,
        "mean_off": mean_off,
        "mean_fused": mean_fused,
        "stdev_off": sd_off,
        "stdev_fused": sd_fused,
        "standard_error_of_difference": se,
        "improvement": delta,
        "improvement_minus_2sigma": delta - 2 * se,
        "improvement_plus_2sigma": delta + 2 * se,
        "relative_improvement": delta / mean_off if mean_off else float("nan"),
        "min_detectable_effect_2sigma": 2 * se,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="session directory written by e198_session.sh")
    parser.add_argument("--out", default="research/e198-stage4.json")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    session = pathlib.Path(args.session)
    legs = [x for x in (load_leg(d) for d in sorted(session.iterdir()) if d.is_dir()) if x]
    legs.sort(key=lambda x: x["leg"])
    if not legs:
        print(f"e198_session_reduce: no complete legs in {session}", file=sys.stderr)
        return 2

    witness = witness_verdict(legs)
    integrity = integrity_verdict(legs)

    off = [x for x in legs if x["arm"] == "OFF"]
    fused = [x for x in legs if x["arm"] == "FUSED"]

    ratio = contrast([x["mtp_decode_speedup"] for x in off],
                     [x["mtp_decode_speedup"] for x in fused],
                     lower_is_better=False)
    absolute = contrast([x["mtp_ms_per_round"] for x in off],
                        [x["mtp_ms_per_round"] for x in fused],
                        lower_is_better=True)

    # Express the ratio effect in ms/round so it meets the stop rule in the
    # stop rule's own unit: hold the serial leg fixed at its session mean and
    # ask what MTP round time each arm's speedup implies.
    serial_mean = statistics.fmean(x["serial_seconds_per_token"] for x in legs)
    edl_mean = statistics.fmean(x["effective_mean_draft_len"] for x in legs)
    ratio_ms_per_round = {
        arm: serial_mean / spd * edl_mean * 1000.0
        for arm, spd in (("off", ratio["mean_off"]), ("fused", ratio["mean_fused"]))
    }
    ratio_implied_improvement = ratio_ms_per_round["off"] - ratio_ms_per_round["fused"]

    decisive = absolute
    verdict = "VOID"
    if witness["positive_arm_witness_present"] and integrity["clean"]:
        verdict = ("PROMOTE"
                   if decisive["improvement_minus_2sigma"] >= PROMOTION_MS_PER_ROUND
                   else "NEGATIVE")

    # COLD-LEG SENSITIVITY. In every session so far leg 1 enters far cooler
    # than legs 2..8, which all sit within about 1 C of each other. ABBA
    # counterbalancing cancels a monotone drift, not a single cold outlier, and
    # the abba1 null control showed that outlier alone can manufacture an
    # apparent 0.18 ms/round arm effect. Recomputing without leg 1 unbalances
    # the design, so this is a robustness check on the sign, never the headline.
    warm = [x for x in legs if x["leg"] != 1]
    warm_off = [x for x in warm if x["arm"] == "OFF"]
    warm_fused = [x for x in warm if x["arm"] == "FUSED"]
    sensitivity = None
    if len(warm_off) > 1 and len(warm_fused) > 1:
        sensitivity = contrast([x["mtp_ms_per_round"] for x in warm_off],
                               [x["mtp_ms_per_round"] for x in warm_fused],
                               lower_is_better=True)
        sensitivity["excluded_leg"] = 1
        sensitivity["balanced_design"] = len(warm_off) == len(warm_fused)
        sensitivity["sign_agrees_with_headline"] = (
            (sensitivity["improvement"] >= 0) == (absolute["improvement"] >= 0))

    # STEADY-STATE SENSITIVITY. The session's second half is balanced 2v2 under
    # this ABBA order, and by then per-leg time has stopped drifting. Early legs
    # carry a warm-up transient that the arm labels do not share evenly, so this
    # is the cleaner estimate of the arm effect even though it uses fewer legs.
    half = len(legs) // 2
    late = legs[half:]
    late_off = [x for x in late if x["arm"] == "OFF"]
    late_fused = [x for x in late if x["arm"] == "FUSED"]
    steady = None
    if len(late_off) > 1 and len(late_fused) > 1:
        steady = contrast([x["mtp_ms_per_round"] for x in late_off],
                          [x["mtp_ms_per_round"] for x in late_fused],
                          lower_is_better=True)
        steady["legs_used"] = [x["leg"] for x in late]
        steady["balanced_design"] = len(late_off) == len(late_fused)
        steady["excludes_promotion_bar"] = (
            steady["improvement_plus_2sigma"] < PROMOTION_MS_PER_ROUND)

    report = {
        "harness": "local",
        "probe": "e198-route1-row-amortized-kernel",
        "session": str(session),
        "host_note": "M4 Pro; ranked runner is M5. Directional local evidence only.",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "worker_sandbox": legs[0]["worker_sandbox"],
        "comparable_with_sandboxed_history": False,
        "promotion_threshold_ms_per_round": PROMOTION_MS_PER_ROUND,
        "verdict": verdict,
        "rule_391b": witness,
        "integrity": integrity,
        "primary_metric_ratio_mtp_decode_speedup": ratio,
        "ratio_implied_ms_per_round": ratio_ms_per_round,
        "ratio_implied_improvement_ms_per_round": ratio_implied_improvement,
        "decisive_metric_absolute_mtp_ms_per_round": absolute,
        "cold_leg_sensitivity_absolute_mtp_ms_per_round": sensitivity,
        "steady_state_sensitivity_absolute_mtp_ms_per_round": steady,
        "session_serial_seconds_per_token_mean": serial_mean,
        "session_effective_mean_draft_len_mean": edl_mean,
        "legs": legs,
    }

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"session   {session}")
    print(f"verdict   {verdict}")
    print(f"witness   ok={witness['positive_arm_witness_present']} "
          f"problems={witness['problems']}")
    print(f"integrity ok={integrity['clean']} problems={integrity['problems']}")
    print()
    print(f"{'leg':>3} {'arm':<6} {'ms/round':>9} {'speedup':>8} {'entryC':>7} "
          f"{'exitC':>7}  witness")
    for x in legs:
        print(f"{x['leg']:>3} {x['arm']:<6} {x['mtp_ms_per_round']:>9.3f} "
              f"{x['mtp_decode_speedup']:>8.4f} {float(x['gpu_temp_entry']):>7.2f} "
              f"{float(x['gpu_temp_exit']):>7.2f}  {x['arm_witness']}")
    print()
    print("absolute mtp ms/round (decisive, lower is better)")
    print(f"  OFF   {absolute['mean_off']:.3f} +/- {absolute['stdev_off']:.3f}")
    print(f"  FUSED {absolute['mean_fused']:.3f} +/- {absolute['stdev_fused']:.3f}")
    print(f"  improvement        {absolute['improvement']:+.3f} ms/round "
          f"({absolute['relative_improvement']*100:+.3f}%)")
    print(f"  improvement - 2sig {absolute['improvement_minus_2sigma']:+.3f} ms/round "
          f"(bar {PROMOTION_MS_PER_ROUND})")
    print(f"  min detectable 2sig {absolute['min_detectable_effect_2sigma']:.3f} ms/round")
    if sensitivity:
        print(f"  drop cold leg 1   {sensitivity['improvement']:+.3f} ms/round "
              f"(2sig {sensitivity['min_detectable_effect_2sigma']:.3f}, "
              f"balanced={sensitivity['balanced_design']}, "
              f"sign_agrees={sensitivity['sign_agrees_with_headline']})")
    if steady:
        print(f"  steady state legs {steady['legs_used']}  "
              f"{steady['improvement']:+.3f} ms/round "
              f"(2sig {steady['min_detectable_effect_2sigma']:.3f}, "
              f"excludes 0.5 bar={steady['excludes_promotion_bar']})")
    print()
    print("ratio mtp_decode_speedup (within-leg serial control, higher is better)")
    print(f"  OFF   {ratio['mean_off']:.4f} +/- {ratio['stdev_off']:.4f}")
    print(f"  FUSED {ratio['mean_fused']:.4f} +/- {ratio['stdev_fused']:.4f}")
    print(f"  improvement        {ratio['improvement']:+.4f} "
          f"({ratio['relative_improvement']*100:+.3f}%)")
    print(f"  implied            {ratio_implied_improvement:+.3f} ms/round")
    print()
    print(f"wrote {out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project="qwen38-mlx-challenge-senpai",
            entity="wandb-applied-ai-team",
            name="e198-route1-row-amortized-kernel-stage4",
            job_type="stage4-abba",
            config={
                "probe": report["probe"],
                "harness": "local",
                "host": "m4-pro",
                "arms": "OFF,FUSED",
                "order": legs[0].get("e198_session_order"),
                "decode_tokens": legs[0]["decode_tokens"],
                "mtp_depth": legs[0]["mtp_depth"],
                "rows": "6,7,8,9",
                "worker_sha256": integrity["worker_sha256"],
                "head_provenance_sha256": integrity["head_provenance_sha256"],
                "worker_sandbox": report["worker_sandbox"],
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
                "comparable_with_sandboxed_history": False,
                "promotion_threshold_ms_per_round": PROMOTION_MS_PER_ROUND,
            },
        )
        for x in legs:
            run.log({f"leg/{k}": v for k, v in x.items()
                     if isinstance(v, (int, float, bool))}, step=x["leg"])
        run.summary.update({
            "verdict": verdict,
            "improvement_ms_per_round": absolute["improvement"],
            "improvement_minus_2sigma_ms_per_round":
                absolute["improvement_minus_2sigma"],
            "min_detectable_effect_2sigma_ms_per_round":
                absolute["min_detectable_effect_2sigma"],
            "relative_improvement": absolute["relative_improvement"],
            "ratio_improvement": ratio["improvement"],
            "ratio_implied_improvement_ms_per_round": ratio_implied_improvement,
            "rule_391b_witness_ok": witness["positive_arm_witness_present"],
            "integrity_clean": integrity["clean"],
        })
        print(f"wandb {run.url}")
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
