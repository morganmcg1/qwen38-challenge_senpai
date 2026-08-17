#!/usr/bin/env python3
"""Research-only: publish the E15 Phase 2 matched control/candidate A/B to W&B.

usage:
  research/log_phase2_wandb.py .mlxfast-private/draft-bits/e15-r1-p2 \
      --log <run_job log> [--arms 4,3] [--group G] [--notes ...]

`prefix` is the tag prefix; arm directories are `<prefix>-b<bits>`. Three
independent records are published so no claim in the report needs the local
filesystem to be reachable:

  * per-arm amdahl/leg scalars and the exactness verdict for both legs,
  * the requant provenance line for every worker instance (proves the 3-bit
    materialisation is one-time and lands in the untimed warm),
  * the paired per-round trace decomposition, which is the only place the
    per-round term is separable from the acceptance term.
"""

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

import wandb

from draft_bits_phase2 import FIELDS, parse

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

LEG_KEYS = ("all_tokens_matched", "emitted_token_total", "round_count",
            "accepted_draft_rate", "effective_mean_draft_len",
            "tokens_per_round", "declared_rows_total", "decode_seconds",
            "parent_measured_seconds_per_token", "steady_seconds_per_token",
            "residual_seconds", "residual_divergence_count",
            "block_seconds_total", "first_block_seconds", "mtp_depth")

AMDAHL_KEYS = ("measured_local_score", "decode_only_speedup", "prefill_seconds",
               "prefill_fraction_of_mtp_leg", "ranked_window_modelled_score",
               "ranked_window_steady_seconds_per_token_mtp",
               "ranked_window_steady_seconds_per_token_serial",
               "ranked_window_leg_seconds_mtp",
               "ranked_window_leg_seconds_serial",
               "serial_decode_seconds_per_token_excl_prefill")


def parse_identity(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        for field in line.split(": ", 1)[-1].split():
            if "=" in field:
                k, v = field.split("=", 1)
                out[k] = v
    return out


def peak_rss(path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "maximum resident set size" in line:
            return int(line.split()[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=Path)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--arms", default="4,3")
    ap.add_argument("--group", default="qwen38-r1-e15-draft-readout-3bit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    arms = args.arms.split(",")
    control, candidate = arms[0], arms[-1]
    docs, ident = {}, {}
    for bits in arms:
        d = args.prefix.parent / f"{args.prefix.name}-b{bits}"
        docs[bits] = json.loads((d / "amdahl.json").read_text())
        ident[bits] = parse_identity(d / "identity.txt")
        ident[bits]["peak_rss_bytes"] = peak_rss(d / "rusage.txt")

    prov = docs[control]["provenance"]
    config = {
        "experiment": "e15-phase2-draft-readout-3bit",
        "phase": 2,
        "arms": arms,
        "control_bits": int(control),
        "candidate_bits": int(candidate),
        "base_sha": prov["base_sha"],
        "head_sha": prov["head_sha"],
        "host_chip": prov["host_chip"],
        "host_os": prov["host_os"],
        "host_memsize_bytes": int(prov["host_memsize_bytes"]),
        "mode": prov["mode"],
        "decode_tokens": docs[control]["mtp_leg"]["emitted_token_total"],
        "seed_token_count": docs[control]["mtp_leg"]["seed_token_count"],
        "cool_gate": ident[control].get("cool_gate"),
        "gate_qualified_for_timing": False,
        "git_head_dirty": ident[control].get("dirty"),
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, group=args.group,
                     job_type="phase2-exactness", config=config,
                     notes=args.notes or "E15 Phase 2 matched A/B: exactness, "
                     "requant provenance, paired per-round decomposition")

    legs = wandb.Table(columns=["bits", "leg", "all_tokens_matched",
                                "emitted_token_total", "seconds_per_token"])
    summary = {}
    for bits in arms:
        doc = docs[bits]
        for leg in ("mtp_leg", "serial_leg"):
            node = doc[leg]
            legs.add_data(int(bits), leg.replace("_leg", ""),
                          bool(node["all_tokens_matched"]),
                          node["emitted_token_total"],
                          node["parent_measured_seconds_per_token"])
        for k in LEG_KEYS:
            if k in doc["mtp_leg"]:
                summary[f"b{bits}/mtp/{k}"] = doc["mtp_leg"][k]
        for k in AMDAHL_KEYS:
            summary[f"b{bits}/{k}"] = doc["amdahl"][k]
        summary[f"b{bits}/peak_rss_bytes"] = ident[bits]["peak_rss_bytes"]

    blocks = parse(args.log)
    reqt = wandb.Table(columns=["arm_bits", "instance", "bits", "source_bits",
                                "requant_ms", "round_count"])
    for bits in arms:
        for i, b in enumerate([x for x in blocks if x["arm"] == bits], 1):
            reqt.add_data(int(bits), i, int(b["bits"]), int(b["source_bits"]),
                          b["requant_ms"], len(b["rounds"]))
    requant = [b["requant_ms"] for b in blocks
               if b["arm"] == candidate and b["requant_ms"] > 0]
    summary["requant_ms_mean"] = st.mean(requant) if requant else 0.0
    summary["requant_ms_max"] = max(requant) if requant else 0.0

    timed = {b: next(x for x in blocks if x["arm"] == b and len(x["rounds"]) > 1)
             for b in arms}
    ctl, cnd = timed[control]["rounds"], timed[candidate]["rounds"]
    assert [r["depth"] for r in ctl] == [r["depth"] for r in cnd], \
        "depth schedules diverge; the acceptance term is not zero"

    rounds = wandb.Table(columns=["round", "depth"] +
                         [f"{p}_{f}" for p in ("ctl", "cnd", "delta")
                          for f in FIELDS])
    for a, b in zip(ctl, cnd):
        rounds.add_data(a["round"], a["depth"],
                        *[a[f] for f in FIELDS], *[b[f] for f in FIELDS],
                        *[a[f] - b[f] for f in FIELDS])
    for f in FIELDS:
        d = [a[f] - b[f] for a, b in zip(ctl, cnd)]
        summary[f"delta/{f}_median_us"] = st.median(d)
        summary[f"delta/{f}_mean_us"] = st.mean(d)

    steady = sum(a["round_us"] - b["round_us"]
                 for a, b in zip(ctl[1:], cnd[1:]))
    readouts = sum(r["depth"] for r in ctl[1:])
    summary["steady_round_us_delta_total"] = steady
    summary["steady_readout_count"] = readouts
    summary["implied_us_saved_per_readout"] = steady / readouts

    mtp = {b: docs[b]["mtp_leg"]["parent_measured_seconds_per_token"]
           for b in arms}
    ser = {b: docs[b]["serial_leg"]["parent_measured_seconds_per_token"]
           for b in arms}
    summary["headline/mtp_seconds_per_token_pct"] = \
        100.0 * (mtp[candidate] - mtp[control]) / mtp[control]
    summary["headline/serial_drift_pct"] = \
        100.0 * (ser[candidate] - ser[control]) / ser[control]
    summary["headline/steady_mtp_seconds_per_token_pct"] = 100.0 * (
        docs[candidate]["mtp_leg"]["steady_seconds_per_token"]
        - docs[control]["mtp_leg"]["steady_seconds_per_token"]
    ) / docs[control]["mtp_leg"]["steady_seconds_per_token"]
    summary["headline/local_score_pct"] = 100.0 * (
        docs[candidate]["amdahl"]["measured_local_score"]
        - docs[control]["amdahl"]["measured_local_score"]
    ) / docs[control]["amdahl"]["measured_local_score"]
    summary["headline/ranked_modelled_score_fixed_serial"] = (
        docs[control]["amdahl"]["ranked_window_leg_seconds_serial"]
        / docs[candidate]["amdahl"]["ranked_window_leg_seconds_mtp"])
    summary["headline/acceptance_term_pct"] = 0.0

    run.log({"legs": legs, "requant_provenance": reqt, "rounds": rounds})
    run.summary.update(summary)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
