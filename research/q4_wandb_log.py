#!/usr/bin/env python3
"""Log the E14 Q4 dispatch-frequency screen to W&B as one analysis run.

    research/q4_wandb_log.py --out-dir research/out/e14-trace \
        --hist .mlxfast-private/ipg-arms/e14-trace-hist-w0.json \
        --frequency .mlxfast-private/ipg-arms/e14-q4-frequency.json \
        --base-sha SHA --host HOST

`ipg_wandb_log.py` carries the isolated-kernel arm comparison. This run carries
the complementary end-to-end evidence that the comparison cannot express: which
verify widths the live scheduler actually dispatches, and therefore how much a
cheaper `M=5` could ever be worth. The window is 256 decode tokens rather than
the ranked 512, so this is a directional policy screen; frequencies transfer,
absolute timings do not.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

# Shipped dispatch table, verified at quantized.h:1809. M=2 uses the crossrow
# kernel rather than the `_m` wide branch, so it has no IPG/pass entry.
DISPATCH = {3: (3, 1, 3), 4: (4, 1, 4), 5: (3, 2, 3), 6: (3, 2, 3),
            7: (4, 2, 4), 8: (4, 2, 4), 9: (3, 3, 3)}


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_meta(out_dir: pathlib.Path) -> dict:
    meta = {}
    for line in (out_dir / "meta.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--frequency", required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--host", default="unknown")
    ap.add_argument("--run-name", default="e14-q4-dispatch-frequency")
    ap.add_argument("--group", default="qwen38-r1-e14-ipg-weight-passes")
    ap.add_argument("--job-id", default=None)
    args = ap.parse_args()

    import wandb

    out_dir = pathlib.Path(args.out_dir)
    meta = read_meta(out_dir)
    score = json.loads((out_dir / "score.json").read_text())
    metrics = score["metrics"]
    hist_doc = json.loads(pathlib.Path(args.hist).read_text())
    leg = next(iter(hist_doc.values()))
    freq = json.loads(pathlib.Path(args.frequency).read_text())

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=args.run_name,
        job_type="analysis",
        group=args.group,
        config={
            "experiment": "qwen38-r1-e14-ipg-weight-passes",
            "question": "Q4-dispatch-frequency",
            "base_sha": args.base_sha,
            "candidate_sha": git_sha(),
            "host": args.host,
            "job_id": args.job_id,
            "decode_tokens": metrics["decode_tokens"],
            "ranked_decode_tokens": 512,
            "ranked_equivalent": False,
            "screen_kind": "directional-policy-screen",
            "mtp_depth_cap": metrics["mtp_depth"],
            "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
            "twins_dirty": meta.get("dirty"),
            "head_provenance_sha256": metrics["head_provenance_sha256"],
            "fixture": "public_longcopy_gate_english_512_256",
        },
    )

    rounds = leg["rounds"]
    width_rows = wandb.Table(
        columns=["offered_depth", "verify_width_m", "rounds", "round_share",
                 "ipg", "weight_passes", "na"]
    )
    for d_str, n in sorted(leg["hist"].items(), key=lambda kv: int(kv[0])):
        d = int(d_str)
        m = d + 1
        ipg, passes, na = DISPATCH.get(m, (None, None, None))
        width_rows.add_data(d, m, n, n / rounds, ipg, passes, na)

    policy_rows = wandb.Table(
        columns=["policy", "rounds", "depth4_rounds", "round_fraction",
                 "time_fraction", "best_case_end_to_end_pct"]
    )
    for name, p in freq["frequency"].items():
        policy_rows.add_data(name, p["rounds"], p["depth4_rounds"],
                             p["round_fraction"], p["time_fraction"],
                             p.get("end_to_end_pct", 0.0))

    multi_pass = sum(n for d, n in leg["hist"].items()
                     if DISPATCH.get(int(d) + 1, (0, 0, 0))[1] and
                     DISPATCH[int(d) + 1][1] > 1)
    three_pass = sum(n for d, n in leg["hist"].items()
                     if DISPATCH.get(int(d) + 1, (0, 0, 0))[1] == 3)
    measured = freq["frequency"]["measured e14-trace"]

    run.summary.update({
        "q4/rounds": rounds,
        "q4/committed_tokens": leg["committed_tokens"],
        "q4/implied_d0_rounds": leg["implied_d0_rounds"],
        "q4/mean_offered_depth": leg["mean_offered_depth"],
        "q4/mean_accepted": leg["mean_accepted"],
        "q4/m5_round_share": measured["round_fraction"],
        "q4/m5_time_share": measured["time_fraction"],
        "q4/m5_best_case_end_to_end_pct": measured["end_to_end_pct"],
        "q4/multi_pass_round_share": multi_pass / rounds,
        "q4/three_pass_m9_round_share": three_pass / rounds,
        "e2e/mtp_decode_speedup": metrics["mtp_decode_speedup"],
        "e2e/serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "e2e/mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "e2e/accepted_draft_rate": metrics["accepted_draft_rate"],
        "e2e/effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "e2e/all_tokens_matched": metrics["all_tokens_matched"],
        "e2e/residual_divergence_count": metrics["residual_divergence_count"],
    })
    run.log({"q4/dispatch_occupancy": width_rows, "q4/policies": policy_rows})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
