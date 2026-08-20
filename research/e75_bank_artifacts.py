#!/usr/bin/env python3
"""Bank the E75 raw per-leg evidence into research/e75-artifacts/.

Rung B legs carry the whole-table isolated QMV width curve; rung D legs carry
the end-to-end 2x2. Both are reduced to the fields a reviewer needs to redo the
arithmetic, so the bundle stays small enough to live in Git while the bulky
run directories stay under .mlxfast-private/.
"""

import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRIV = ROOT / ".mlxfast-private"
OUT = ROOT / "research" / "e75-artifacts"

RB_LEGS = [
    ("e75-rB-a1", "ours"), ("e75-rB-a2", "crown"),
    ("e75-rB-a3", "crown"), ("e75-rB-a4", "ours"),
    ("e75-rB-a5", "ours"), ("e75-rB-a6", "crown"),
    ("e75-rB-a7", "crown"), ("e75-rB-a8", "ours"),
]
WANDB = {
    "e75-rB-a1": "dlk0dil9", "e75-rB-a2": "iiyrtg8k",
    "e75-rB-a3": "fdo47a74", "e75-rB-a4": "w24k9vc7",
    "e75-rB-a5": "hflbcpf1", "e75-rB-a6": "0qbqlmv0",
    "e75-rB-a7": "94o5jfus", "e75-rB-a8": "41zmikig",
}
RD_LEGS = [
    ("e75-rD-warmup", "ours-ship", True), ("e75-rD-d1", "ours-ship", False),
    ("e75-rD-d2", "ours-pbfit", False), ("e75-rD-d3", "crown-ship", False),
    ("e75-rD-d4", "crown-pbfit", False), ("e75-rD-d5", "crown-pbfit", False),
    ("e75-rD-d6", "crown-ship", False), ("e75-rD-d7", "ours-pbfit", False),
    ("e75-rD-d8", "ours-ship", False),
]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def table_cost_ms(summary):
    """Whole-table isolated QMV cost per verify forward, by width.

    Same field `e75_rungB_analyze.py` reduces, so the banked artifact and the
    published curve cannot drift apart.
    """
    return {
        row["verify_width"]: round(row["gemm_seconds"] * 1e3, 6)
        for row in summary["round_cost_model"]["rows"]
    }


def bank_rung_b():
    legs = []
    for tag, arm in RB_LEGS:
        leg = load(PRIV / "e75-legs" / f"{tag}-leg.json")
        summary = load(PRIV / "qmv-curve" / tag / "summary.json")
        gate = load(PRIV / "e75-legs" / f"{tag}-gpu-gate.json")
        patch = leg["arm_patch"]
        legs.append({
            "tag": tag,
            "arm": arm,
            "wandb_run_id": WANDB[tag],
            "wandb_url":
                "https://wandb.ai/wandb-applied-ai-team/"
                f"qwen38-mlx-challenge-senpai/runs/{WANDB[tag]}",
            "cell": patch["cell"],
            "na_max": patch["na_max"],
            "dispatch": patch["dispatch"],
            "crown_bytes_verified": patch["crown_bytes_verified"],
            "sources_as_measured": leg["sources_as_measured"],
            "binary_probe": leg["binary_probe"],
            "gpu_gate": gate,
            "branch_commit": leg["branch_commit"],
            "measured_commit_unwound": leg["measured_commit_unwound"],
            "base_sha": summary["base_sha"],
            "host": summary["host"],
            "gpu_architecture": summary["device"]["architecture"],
            # Known defect: model-law constant, not build state. See results.
            "crossrow_na_max_reported": summary["crossrow_na_max"],
            "table_cost_ms_by_width": table_cost_ms(summary),
        })

    by_arm = {}
    for leg in legs:
        by_arm.setdefault(leg["arm"], []).append(leg["table_cost_ms_by_width"])
    curve = {}
    for arm, tables in by_arm.items():
        widths = sorted(set().union(*[set(t) for t in tables]))
        curve[arm] = {
            str(w): {
                "median_ms": round(
                    statistics.median([t[w] for t in tables if w in t]), 4),
                "n_legs": len([t for t in tables if w in t]),
                "legs_ms": [round(t[w], 4) for t in tables if w in t],
            }
            for w in widths
        }

    doc = {
        "experiment": "E75 rung B",
        "harness": "local",
        "not_a_ranked_score": True,
        "design": "balanced mirrored palindrome, 4 legs per arm",
        "instrument":
            "E68 rung-1, unchanged: --widths 1,2,3,4,5,6,7,8,9,10 "
            "--reps 21 --inner 10 --skip-stock",
        "upstream_crown_digests": {
            "quantized.h":
                "75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e0"
                "10817b1e5249f281",
            "quantized.cpp":
                "350de46828265271e504c93d009a3b3e8b05c83047666be7"
                "fc0de51ded29b6bb",
        },
        "arm_median_table_cost_ms": curve,
        "legs": legs,
    }
    (OUT / "e75-rungB-legs.json").write_text(json.dumps(doc, indent=1) + "\n")
    return len(legs)


def bank_rung_d():
    legs = []
    for tag, cell, discarded in RD_LEGS:
        run = PRIV / "e75-e2e" / "runs" / tag
        out = load(run / "reports" / "02-mtp-verify-output.json")
        timed = load(run / "reports" / "04-mtp-timed.json")
        meta = dict(
            line.strip().split("=", 1)
            for line in (run / "meta.txt").read_text().splitlines()
            if "=" in line)
        stream = ",".join(str(int(t)) for t in out["emitted_tokens"])
        widths = [d + 1 for d in timed["effective_draft_lengths"]]
        round_ms = {}
        for width, seconds in zip(widths, timed["block_request_seconds"]):
            round_ms.setdefault(width, []).append(seconds * 1e3)
        legs.append({
            "tag": tag,
            "cell": cell,
            "discarded": discarded,
            "score_metrics": load(run / "score.json")["metrics"],
            "meta": meta,
            "emitted_token_count": len(out["emitted_tokens"]),
            "stream_sha256": hashlib.sha256(stream.encode()).hexdigest(),
            "verify_width_histogram": {
                str(w): widths.count(w) for w in sorted(set(widths))},
            "round_ms_median_by_width": {
                str(w): round(statistics.median(v), 4)
                for w, v in sorted(round_ms.items())},
        })
    doc = {
        "experiment": "E75 rung D",
        "harness": "local",
        "not_a_ranked_score": True,
        "status":
            "The session finished at 14:57Z, after the advisor cancelled the "
            "rung at 14:48Z. The cost was already spent, so the evidence is "
            "reported as an appendix and no further rung-D work was done.",
        "design":
            "2x2 {kernel table} x {depth price}, mirrored palindrome, "
            "one discarded warmup leg, four prebuilt cells",
        "legs": legs,
    }
    (OUT / "e75-rungD-legs.json").write_text(json.dumps(doc, indent=1) + "\n")
    return len(legs)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"rung B legs banked: {bank_rung_b()}")
    print(f"rung D legs banked: {bank_rung_d()}")
