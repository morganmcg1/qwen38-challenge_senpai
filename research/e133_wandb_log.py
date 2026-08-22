#!/usr/bin/env python3
"""Publish the E133 offline screen to W&B.

    usage: research/e133_wandb_log.py --rung 1|2|3 [--dry]

E133 is an offline screen. It replays a captured hidden-state corpus through
numpy/MLX arrays and never times a decode leg, so no run here is a timing
measurement, a gated measurement, or a score. Every run therefore logs
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e133-sketch-first-draft-readout-offline-screen"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "197e0550ab46842b639a4ff4fe3f4889ca3b01ec"

# T0 kills a cell whose worst-domain net miss exceeds this. T0b kills a cell
# that loses the exact affine-2 top-1 more often than 0.3 % of the time.
T0_NET_MISS = 3.0e-3
T0B_RECALL = 0.997

RUNGS = {
    "1": {
        "run_name": "e133-rung1-corpus-capture",
        "file": "research/e133-corpus.json",
        "question":
            "how many decode-position hidden states did the instrumented "
            "worker capture over the committed E124 corpus, and does every "
            "seed keep MTP parity with its serial reference",
        "command":
            "research/e133_job.sh research/e133_capture.sh all "
            "--steps 512 --depth 8",
    },
    "2": {
        "run_name": "e133-rung2-screen-validation",
        "file": "research/e133-validate.json",
        "question":
            "does the offline model of the shipped chain agree with the "
            "runtime proposal token, and can a deliberately damaged 8-bit "
            "SimHash make the miss column fail",
        "command": "python3 research/e133_screen.py validate",
    },
    "3": {
        "run_name": "e133-rung3-sketch-cell-sweep",
        "file": "research/e133-screen.json",
        "question":
            "does any compact sketch keep the shipped draft argmax while it "
            "removes enough of the 59.09 MB two-pass readout to pay for "
            "itself on the ranked score",
        "command": "python3 research/e133_screen.py screen",
    },
}

CELL_COLUMNS = [
    "arm", "family", "size", "stage_a", "bytes_per_row", "proj_bytes",
    "survivors", "probe_fraction", "n", "misses", "m", "m_lo", "m_hi",
    "worse_than_shipped", "better_than_shipped",
    "net_miss", "net_miss_lo", "net_miss_hi",
    "net_miss_worst_domain", "net_miss_low_acceptance",
    "recall_affine2_top1", "probe_hit_rate", "survivor_hit_rate",
    "arm_stage_bytes", "shipped_stage_bytes", "removed_bytes",
    "removed_step_fraction", "pct_byte_rate", "pct_head_share_7",
    "pct_head_share_9", "predicted_pct", "predicted_pct_worst_domain",
    "predicted_pct_low_acceptance", "passes_t0", "passes_t0b",
]

DOMAIN_COLUMNS = [
    "arm", "domain", "n", "m", "net", "net_lo", "net_hi", "discordant",
    "recall",
]

SEED_COLUMNS = [
    "seed", "domain", "samples", "shards", "accepted_draft_rate",
    "effective_mean_draft_len", "round_count", "parity_all_ok",
    "all_tokens_matched",
]


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def gates(cell: dict) -> tuple[bool, bool]:
    return (cell["net_miss_worst_domain"] <= T0_NET_MISS,
            cell["recall_affine2_top1"] >= T0B_RECALL)


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def corpus_summary(payload: dict) -> tuple[dict, dict]:
    seeds = payload["seeds"]
    parity = [s for s in seeds if s.get("parity_all_ok") is not True]
    summary = {
        "corpus_samples": payload["samples"],
        "corpus_seeds": len(seeds),
        "corpus_seeds_without_parity": len(parity),
        "corpus_meets_6000_floor": payload["samples"] >= 6000,
    }
    flatten("by_domain", payload["by_domain"], summary)
    return summary, {"corpus_seeds": table(SEED_COLUMNS, seeds)}


def validate_summary(payload: dict) -> tuple[dict, dict]:
    summary: dict = {"validate_samples": payload["samples"]}
    for section in ("proposal_match", "proposal_mismatch",
                    "m_shipped_live_chain", "m_damaged_simhash8_control"):
        flatten(section, payload.get(section), summary)
    # The control only proves anything if it actually fails.
    summary["control_can_fail"] = (
        payload["m_damaged_simhash8_control"]["p"]
        > 10 * max(payload["m_shipped_live_chain"]["p"], 1e-6))
    return summary, {}


def screen_summary(payload: dict) -> tuple[dict, dict]:
    cells = []
    domains = []
    for cell in payload["cells"]:
        t0, t0b = gates(cell)
        row = dict(cell)
        row["passes_t0"] = t0
        row["passes_t0b"] = t0b
        cells.append(row)
        for domain, stats in cell["by_domain"].items():
            domains.append({"arm": cell["arm"], "domain": domain, **stats})

    survivors = [c for c in cells if c["passes_t0"] and c["passes_t0b"]]
    best = max(survivors, key=lambda c: c["predicted_pct_worst_domain"],
               default=None)
    # The primary metric is what a compliant cell is worth on the ranked
    # score. No compliant cell means the mechanism is worth exactly zero.
    summary = {
        "screen_samples": payload["samples"],
        "screen_cells": len(cells),
        "cells_passing_t0": sum(1 for c in cells if c["passes_t0"]),
        "cells_passing_t0b": sum(1 for c in cells if c["passes_t0b"]),
        "cells_passing_both": len(survivors),
        "e133_best_cell_ranked_pct":
            best["predicted_pct_worst_domain"] if best else 0.0,
        "e133_worst_domain_net_miss_rate":
            min((c["net_miss_worst_domain"] for c in cells), default=float("nan")),
        "best_cell_arm": best["arm"] if best else None,
        "t0_threshold": T0_NET_MISS,
        "t0b_threshold": T0B_RECALL,
        "miss_to_score_pct": 203.0,
    }
    if best:
        for key in ("family", "size", "stage_a", "survivors", "probe_fraction",
                    "bytes_per_row", "removed_bytes", "pct_byte_rate",
                    "pct_head_share_7", "pct_head_share_9", "net_miss",
                    "net_miss_worst_domain", "net_miss_low_acceptance",
                    "recall_affine2_top1"):
            summary["best_cell/%s" % key] = best[key]
    flatten("shipped", {k: v for k, v in payload["shipped"].items()
                        if k != "by_domain"}, summary)
    return summary, {"screen_cells": table(CELL_COLUMNS, cells),
                     "screen_by_domain": table(DOMAIN_COLUMNS, domains)}


BUILDERS = {"1": corpus_summary, "2": validate_summary, "3": screen_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    payload = json.loads(pathlib.Path(spec["file"]).read_text())
    summary, tables = BUILDERS[args.rung](payload)
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "offline",
        "timing_valid": False,
        "host": HOST,
        "base_sha": BASE_SHA,
        "rung": args.rung,
        "question": spec["question"],
        "command": spec["command"],
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type="offline-screen",
                     config={"experiment": "E133", "rung": args.rung,
                             "base_sha": BASE_SHA, "host": HOST,
                             "corpus": "e124-corpus-manifest.json",
                             "command": spec["command"],
                             "question": spec["question"]})
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e133-rung%s" % args.rung, type="offline-screen")
    artifact.add_file(spec["file"])
    run.log_artifact(artifact)
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
