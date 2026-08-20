#!/usr/bin/env python3
"""Publish one W&B run per E80 census leg.

The census legs are measured by `e80_census_session.sh`, which writes a
`census.jsonl` of command-buffer intervals and a `meta.txt` of provenance. This
script reads those files after the fact and publishes the derived per-kernel and
per-family GPU-time census to W&B, one run per leg.

Every run is labelled with the leg's real gate status, taken from `meta.txt`
`cool_gate`, so a gated leg and an ungated ABBA leg can never be mistaken for
each other or pooled by accident.

    usage: research/e80_wandb_log.py research/out/e80-census-w6-isolated ...
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import wandb

import e80_blocks as B
import e80_census_report as R

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def families(att, rounds):
    fam = collections.Counter()
    disp = collections.Counter()
    for key, v in att.items():
        f = B.family_of_owner(key)
        fam[f] += v["gpu_ns"] / rounds / 1e6
        disp[f] += v["disp"] / rounds
    return fam, disp


def totals(leg, phase, width, rules):
    att, rounds = B.attribute(leg, phase, width, rules)
    ms = sum(v["gpu_ns"] for v in att.values()) / rounds / 1e6
    return att, rounds, ms


def leg_payload(directory: pathlib.Path, min_rounds: int) -> dict:
    leg = R.Leg([directory / "census.jsonl"])
    rules = B.learn_axis_rules(leg)
    out = {"phases": {}, "widths": {}}

    lo_att, lo_rounds, lo_tot = totals(leg, "target_forward", 1, rules)
    if lo_att:
        fam, disp = families(lo_att, lo_rounds)
        out["serial"] = {
            "rounds": lo_rounds, "gpu_ms_per_round": lo_tot,
            "families": dict(fam), "dispatches": dict(disp),
            "kernels": B.dump(lo_att, lo_rounds),
        }

    for width in B.verify_widths(leg, min_rounds):
        hi_att, hi_rounds, hi_tot = totals(leg, "target_verify", width, rules)
        hd_att, hd_rounds, hd_tot = totals(leg, "draft_head", width, rules)
        round_att = B.merge_acc(hi_att, hd_att)
        fam_v, disp_v = families(hi_att, hi_rounds)
        fam_r, disp_r = families(round_att, hi_rounds)
        out["widths"][str(width)] = {
            "verify_rounds": hi_rounds,
            "verify_gpu_ms_per_round": hi_tot,
            "draft_head_gpu_ms_per_round": hd_tot,
            "whole_round_gpu_ms_per_round": hi_tot + hd_tot,
            "tax_vs_serial_ms": hi_tot - lo_tot if lo_rounds else None,
            "verify_families": dict(fam_v),
            "verify_dispatches": dict(disp_v),
            "whole_round_families": dict(fam_r),
            "whole_round_dispatches": dict(disp_r),
            "verify_kernels": B.dump(hi_att, hi_rounds),
            "draft_head_kernels": B.dump(hd_att, hd_rounds) if hd_rounds
            else [],
        }

    out["phases"] = {k: dict(v) for k, v in leg.by_width_phase.items()}
    out["health"] = dict(leg.health)
    out["host"] = {
        f"w{w}": {
            "rounds": n,
            "wall_ms_per_round": leg.round_wall_ns[w] / n / 1e6,
            "commits_per_round": leg.round_commits[w] / n,
            "dispatches_per_round": leg.round_dispatches[w] / n,
            "waits_per_round": leg.round_waits[w] / n,
            "wait_ms_per_round": leg.round_wait_ns[w] / n / 1e6,
        }
        for w, n in leg.round_total.items() if n
    }
    return out


def publish(directory: pathlib.Path, group: str, min_rounds: int) -> str:
    meta = read_meta(directory / "meta.txt")
    gated = meta.get("cool_gate") == "1" and meta.get("exit") == "0"
    payload = leg_payload(directory, min_rounds)
    entity, project = PROJECT.split("/", 1)
    run = wandb.init(
        entity=entity, project=project, name=directory.name, group=group,
        job_type="census", reinit=True,
        config={
            "experiment": "e80-per-kernel-gpu-time-census",
            "harness": "local",
            "attribution_method": "dominant-dispatch over measured "
                                  "command-buffer intervals; no fit",
            "cool_gate_passed_real_gate": gated,
            "gate_qualified_for_timing": gated,
            "official_or_ranked_score": False,
            "host": "apple-m4-pro-applegpu_g16s-20core-48gib",
            **meta,
        },
    )
    summary = {}
    if "serial" in payload:
        summary["serial/gpu_ms_per_round"] = payload["serial"]["gpu_ms_per_round"]
        summary["serial/rounds"] = payload["serial"]["rounds"]
        for fam, ms in payload["serial"]["families"].items():
            summary[f"serial/family_ms/{fam}"] = ms
    for width, w in payload["widths"].items():
        summary[f"w{width}/verify_gpu_ms_per_round"] = \
            w["verify_gpu_ms_per_round"]
        summary[f"w{width}/draft_head_gpu_ms_per_round"] = \
            w["draft_head_gpu_ms_per_round"]
        summary[f"w{width}/whole_round_gpu_ms_per_round"] = \
            w["whole_round_gpu_ms_per_round"]
        summary[f"w{width}/verify_rounds"] = w["verify_rounds"]
        if w["tax_vs_serial_ms"] is not None:
            summary[f"w{width}/width_tax_ms"] = w["tax_vs_serial_ms"]
        for fam, ms in w["verify_families"].items():
            summary[f"w{width}/verify_family_ms/{fam}"] = ms
        for fam, ms in w["whole_round_families"].items():
            summary[f"w{width}/round_family_ms/{fam}"] = ms
            summary[f"w{width}/round_family_share/{fam}"] = \
                ms / w["whole_round_gpu_ms_per_round"]
    for phase, acc in payload["phases"].items():
        summary[f"phase/{phase}/gpu_ms"] = acc["gpu_ns"] / 1e6
        summary[f"phase/{phase}/buffers"] = acc["buffers"]
        summary[f"phase/{phase}/dispatches"] = acc["dispatches"]
    for width, h in payload["host"].items():
        for key, value in h.items():
            summary[f"host/{width}/{key}"] = value
        gpu = summary.get(f"{width}/whole_round_gpu_ms_per_round")
        if gpu is None and width == "w1":
            gpu = summary.get("serial/gpu_ms_per_round")
        if gpu is not None:
            summary[f"host/{width}/closure_gap_ms_per_round"] = \
                h["wall_ms_per_round"] - gpu
    for key, value in payload["health"].items():
        summary[f"health/{key}"] = value
    run.summary.update(summary)

    rows = []
    for width, w in payload["widths"].items():
        for phase, key in (("target_verify", "verify_kernels"),
                           ("draft_head", "draft_head_kernels")):
            for k in w[key]:
                rows.append([int(width), phase, k["owner"], k["family"],
                             k["dispatches_per_round"], k["gpu_ms_per_round"],
                             k["mean_buffer_ns"]])
    if "serial" in payload:
        for k in payload["serial"]["kernels"]:
            rows.append([1, "target_forward", k["owner"], k["family"],
                         k["dispatches_per_round"], k["gpu_ms_per_round"],
                         k["mean_buffer_ns"]])
    run.log({"census": wandb.Table(
        columns=["width", "phase", "owning_dispatch", "family",
                 "dispatches_per_round", "gpu_ms_per_round", "mean_buffer_ns"],
        data=rows)})

    artifact = wandb.Artifact(f"e80-census-{directory.name}", type="census")
    with artifact.new_file("census.json", mode="w") as handle:
        json.dump(payload, handle, indent=1)
    with artifact.new_file("meta.txt", mode="w") as handle:
        handle.write((directory / "meta.txt").read_text())
    run.log_artifact(artifact)

    url, rid = run.url, run.id
    run.finish()
    print(f"E80_WANDB {directory.name} {rid} {url}", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--group", default="e80-per-kernel-gpu-time-census")
    ap.add_argument("--min-rounds", type=int, default=20)
    args = ap.parse_args()
    for path in args.legs:
        directory = pathlib.Path(path)
        if not (directory / "census.jsonl").exists():
            print(f"skip {directory.name}: no census.jsonl", flush=True)
            continue
        publish(directory, args.group, args.min_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
