#!/usr/bin/env python3
"""E160: publish the fused SwiGLU producer measurement to W&B.

One run, five record groups, every number read from the legs' own artifacts.

  legs          the twelve gated 512-token legs, with the accept ledger each
                one reconstructed from the parent's own ratios, the entry and
                exit GPU temperature, and both gate flags verbatim.
  contrasts     the three arm contrasts, per replicate and pooled, in per cent
                of the timed leg, in microseconds per decode round, and in
                microseconds per removed chunk-sum fill.
  census        the (sg_cand, xs_fill, xs_hit) tuple each arm produced in its
                own trace, which is what proves the arm reached the worker.
  depth         the realized proposed-depth histogram of the traced leg.
  headline      the stop-rule verdict and the labelled ranked projection.

harness=local throughout. No number here is an official or ranked score.

    python3 research/e160_wandb_log.py --run-name e160-r0-swiglu-producer-fusion
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import re
import statistics
import subprocess

import wandb

sys_path = pathlib.Path(__file__).resolve().parent
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ARMS = ("off", "replica", "fuse")
REMOVED_FILLS_PER_ROUND = 64
PREFILL_S = 4.0042  # measured by Edward on this host and fixture, borrowed
MINIMUM_USEFUL_PCT = 0.06
K_TRANSFER = 1.13  # advisor F4, per-round arm shape
M_HARNESS_MIX = 1.232  # advisor F6, decode-only arm flat in draft depth


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sys_path.parent,
        capture_output=True, text=True, check=True).stdout.strip()


def meta(path: pathlib.Path) -> dict:
    return dict(
        line.strip().split("=", 1) for line in path.open() if "=" in line)


def ledger(edl: float, acc: float, tokens: int) -> tuple[int, int, int]:
    for rounds in range(1, tokens + 1):
        proposed = edl * rounds
        if abs(proposed - round(proposed)) > 1e-6:
            continue
        proposed = round(proposed)
        accepted = acc * proposed
        if abs(accepted - round(accepted)) > 1e-6:
            continue
        return rounds, proposed, round(accepted)
    raise SystemExit(f"no small-denominator ledger fits edl={edl} acc={acc}")


def load_legs(label: str) -> list[dict]:
    legs = []
    for score_path in sorted(glob.glob(f"research/out/e160{label}k*/score.json")):
        m = json.load(open(score_path))["metrics"]
        d = meta(pathlib.Path(score_path).with_name("meta.txt"))
        rounds, proposed, accepted = ledger(
            m["effective_mean_draft_len"], m["accepted_draft_rate"],
            m["decode_tokens"])
        legs.append({
            "tag": d["tag"], "arm": d["e160_arm"], "rep": int(d["e160_rep"]),
            "position": int(d["e160_position"]),
            "mtp": m["mtp_seconds_per_token"],
            "serial": m["serial_seconds_per_token"],
            "edl": m["effective_mean_draft_len"],
            "acc": m["accepted_draft_rate"],
            "matched": m["all_tokens_matched"],
            "tokens": m["decode_tokens"],
            "rounds": rounds, "proposed": proposed, "accepted": accepted,
            "entry_c": float(d["gpu_temp_entry_c"]),
            "exit_c": float(d["gpu_temp_exit_c"]),
            "real_gate": d["cool_gate_passed_real_gate"],
            "qualified": d["gate_qualified_for_timing"],
            "worker_sha256": d["worker_sha256"],
            "post_run_worker_sha256": d["post_run_worker_sha256"],
            "metallib_fingerprint": d["metallib_source_fingerprint"],
            "commit": d["e160_session_commit"],
        })
    if not legs:
        raise SystemExit(f"no legs under research/out/e160{label}k*")
    return legs


def arm_means(legs: list[dict]) -> dict[str, float]:
    return {a: statistics.fmean(d["mtp"] for d in legs if d["arm"] == a)
            for a in ARMS if any(d["arm"] == a for d in legs)}


def two_se(legs: list[dict]) -> tuple[float, int]:
    """Pooled within-arm 2se of a difference of two arm means, per cent."""
    residuals, dof = [], 0
    for arm in ARMS:
        vals = [d["mtp"] for d in legs if d["arm"] == arm]
        if len(vals) < 2:
            continue
        mu = statistics.fmean(vals)
        residuals += [v - mu for v in vals]
        dof += len(vals) - 1
    var = sum(r * r for r in residuals) / dof
    n = len(legs) / len(ARMS)
    grand = statistics.fmean(d["mtp"] for d in legs)
    return 2 * (2 * var / n) ** 0.5 / grand * 100, dof


CONTRASTS = (
    ("fuse", "off", "shippable"),
    ("replica", "off", "kernel swap"),
    ("fuse", "replica", "fill removal"),
)


def contrast_rows(legs: list[dict], scope: str) -> list[list]:
    mean = arm_means(legs)
    band, dof = two_se(legs)
    rounds = legs[0]["rounds"]
    tokens = legs[0]["tokens"]
    rows = []
    for better, base, name in CONTRASTS:
        saved_s = (mean[base] - mean[better]) * tokens
        pct_leg = saved_s / (mean[base] * tokens) * 100
        decode_base = mean[base] * tokens - PREFILL_S
        us_round = saved_s / rounds * 1e6
        rows.append([
            scope, f"{better} - {base}", name, pct_leg,
            saved_s / decode_base * 100, us_round,
            us_round / REMOVED_FILLS_PER_ROUND, band, dof,
            pct_leg - band, pct_leg + band,
        ])
    return rows


def census_rows(label: str) -> list[list]:
    pattern = re.compile(
        r"sg_cand=(\d+).*?xs_fill=(\d+)\s+xs_hit=(\d+)"
    )
    rows = []
    for check in sorted(glob.glob(f"research/out/e160{label}*/e160-arm-check.txt")):
        text = pathlib.Path(check).read_text()
        want = re.search(r"want (\w+)\s+\(sg_cand, xs_fill, xs_hit\) = \(([^)]*)\)", text)
        seen = re.search(r"observed\s+\[\(([^)]*)\)\]", text)
        paying = re.search(r"paying rounds\s+(\d+) of (\d+)", text)
        verdict = "OK" if text.rstrip().endswith("OK") else "FAILED"
        if not (want and seen and paying):
            continue
        rows.append([
            pathlib.Path(check).parent.name, want.group(1),
            f"({want.group(2)})", f"({seen.group(1)})",
            int(paying.group(1)), int(paying.group(2)), verdict,
        ])
    return rows


def depth_rows(trace: pathlib.Path) -> list[list]:
    counts = collections.Counter(
        int(m) for m in re.findall(r"mtp-trace: round=\d+ d=(\d+)", trace.read_text()))
    total = sum(counts.values())
    return [[d, counts.get(d, 0), counts.get(d, 0) / total] for d in range(9)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="fuse")
    ap.add_argument("--run-name", default="e160-r0-swiglu-producer-fusion")
    args = ap.parse_args()

    legs = load_legs(args.label)
    reps = sorted({d["rep"] for d in legs})
    pooled_mean = arm_means(legs)
    band, dof = two_se(legs)
    rounds = legs[0]["rounds"]
    shippable = [r for r in contrast_rows(legs, "pooled")
                 if r[2] == "shippable"][0]
    kernel_swap = [r for r in contrast_rows(legs, "pooled")
                   if r[2] == "kernel swap"][0]
    fill_removal = [r for r in contrast_rows(legs, "pooled")
                    if r[2] == "fill removal"][0]

    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=args.run_name,
        job_type="measurement",
        tags=["e160", "r0", "swiglu-producer-fusion", "chunk-sum", "harness=local",
              "gated", "null-result"],
        config={
            "experiment": "e160-mlp-down-producer-fusion",
            "rung": "r0",
            "pr": 160,
            "base_sha": "d0a1501e",
            "session_commit": legs[0]["commit"],
            "worker_sha256": legs[0]["worker_sha256"],
            "metallib_source_fingerprint": legs[0]["metallib_fingerprint"],
            "harness": "local",
            "local_mode": "--local-iterate",
            "decode_tokens": legs[0]["tokens"],
            "rounds_per_leg": rounds,
            "arms": list(ARMS),
            "arm_selector_env": "MLX_E160_SWIGLU_ARM",
            "design": "ABBA palindrome off replica fuse fuse replica off",
            "replicates": len(reps),
            "legs": len(legs),
            "cool_gate": "real 40 C gate on every timed leg",
            "removed_fills_per_round": REMOVED_FILLS_PER_ROUND,
            "prefill_seconds_borrowed": PREFILL_S,
            "minimum_useful_effect_pct": MINIMUM_USEFUL_PCT,
            "k_transfer": K_TRANSFER,
            "m_harness_mix": M_HARNESS_MIX,
            "host": "ip-10-231-2-95.ec2.internal",
            "chip": "Apple M4 Pro",
            "official_or_ranked_score": False,
        },
    )

    run.log({"legs": wandb.Table(
        columns=["tag", "arm", "rep", "position", "mtp_seconds_per_token",
                 "serial_seconds_per_token", "rounds", "proposed", "accepted",
                 "effective_mean_draft_len", "accepted_draft_rate",
                 "all_tokens_matched", "gpu_temp_entry_c", "gpu_temp_exit_c",
                 "cool_gate_passed_real_gate", "gate_qualified_for_timing",
                 "worker_sha256", "post_run_worker_sha256"],
        data=[[d["tag"], d["arm"], d["rep"], d["position"], d["mtp"], d["serial"],
               d["rounds"], d["proposed"], d["accepted"], d["edl"], d["acc"],
               d["matched"], d["entry_c"], d["exit_c"], d["real_gate"],
               d["qualified"], d["worker_sha256"], d["post_run_worker_sha256"]]
              for d in legs])})

    rows = []
    for rep in reps:
        rows += contrast_rows([d for d in legs if d["rep"] == rep], f"k{rep}")
    rows += contrast_rows(legs, "pooled")
    run.log({"contrasts": wandb.Table(
        columns=["scope", "contrast", "isolates", "pct_of_leg", "pct_of_decode",
                 "us_per_round", "us_per_removed_fill", "two_se_pct", "dof",
                 "ci_low_pct", "ci_high_pct"],
        data=rows)})

    run.log({"arm_means": wandb.Table(
        columns=["arm", "mtp_seconds_per_token", "leg_seconds",
                 "decode_seconds", "R_leg_ms", "R_decode_ms"],
        data=[[a, mu, mu * legs[0]["tokens"],
               mu * legs[0]["tokens"] - PREFILL_S,
               mu * legs[0]["tokens"] / rounds * 1e3,
               (mu * legs[0]["tokens"] - PREFILL_S) / rounds * 1e3]
              for a, mu in pooled_mean.items()])})

    census = census_rows(args.label)
    if census:
        run.log({"census": wandb.Table(
            columns=["leg", "arm", "expected_tuple", "observed_tuple",
                     "paying_rounds", "transitions", "verdict"], data=census)})

    trace = pathlib.Path(f"research/out/e160{args.label}warm1/trace.txt")
    if trace.exists():
        run.log({"proposed_depth_histogram": wandb.Table(
            columns=["d", "rounds", "fraction"], data=depth_rows(trace))})

    verdict = ("PROMOTE" if shippable[3] >= MINIMUM_USEFUL_PCT and shippable[9] > 0
               else "STOP")
    run.summary.update({
        "pooled_fuse_minus_off_pct": shippable[3],
        "pooled_fuse_minus_off_us_per_round": shippable[5],
        "pooled_us_per_removed_fill": shippable[6],
        "pooled_replica_minus_off_pct": kernel_swap[3],
        "pooled_replica_minus_off_ci": [kernel_swap[9], kernel_swap[10]],
        "pooled_fill_removal_pct": fill_removal[3],
        "pooled_fill_removal_us_per_removed_fill": fill_removal[6],
        "two_se_pct": band,
        "dof": dof,
        "ci_low_pct": shippable[9],
        "ci_high_pct": shippable[10],
        "minimum_useful_effect_pct": MINIMUM_USEFUL_PCT,
        "stop_rule_verdict": verdict,
        "projected_published_pct_at_k_m": shippable[3] * K_TRANSFER * M_HARNESS_MIX,
        "projected_published_pct_upper_ci": shippable[10] * K_TRANSFER * M_HARNESS_MIX,
        "accept_ledger_identical": len({(d["rounds"], d["proposed"], d["accepted"])
                                        for d in legs}) == 1,
        "all_tokens_matched_every_leg": all(d["matched"] for d in legs),
        "gate_qualified_every_leg": all(d["qualified"] == "true" for d in legs),
        "harness": "local",
        "official_or_ranked_score": False,
    })
    print(run.url, run.id, sep="\n")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
