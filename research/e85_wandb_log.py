#!/usr/bin/env python3
"""Publish the E85 evidence to W&B: one run per timed leg plus one summary run.

    usage: research/e85_wandb_log.py SESSION_DIR [--stats STATS_JSON]
                                     [--census CENSUS_JSON] [--buffers N]

Every leg run carries its own arm, absolute times, draft counters, entry and
exit GPU temperature, and the verbatim gate flags. A leg from this session is
ungated ABBA evidence, so `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged as `false` on every run and never
omitted or relabelled.

The summary run carries the drift-free arm contrasts, their conversion to
microseconds per draft token and per eliminated intermediate buffer, and the
verdict against the assignment's stop rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e85-materialised-intermediate-elimination"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

ARM_DESCRIPTION = {
    "base": "eager embed + eager take/quantizedMM rerank (unchanged base)",
    "a": "fused quantized-embedding dual-RMS-norm concat only",
    "b": "gather_qmm rerank only",
    "ab": "both eliminations",
}


def session_kind(legs: list[dict]) -> str:
    """`abba` legs carry an `arm` column; buffer-tax legs carry a `tax` level."""
    return "tax" if "tax" in legs[0] else "abba"


def leg_arm(row: dict) -> str:
    return row["arm"] if "arm" in row else f"k{row['tax']}"


def leg_config(row: dict) -> dict:
    if "arm" in row:
        return {
            "arm": row["arm"],
            "arm_description": ARM_DESCRIPTION.get(row["arm"], row["arm"]),
            "MLX_E85_FUSED_EMBED": int(row["fused_embed"]),
            "MLX_E85_GATHER_QMM": int(row["gather_qmm"]),
        }
    tax = int(row["tax"])
    return {
        "arm": f"k{tax}",
        "arm_description": f"{tax} added materialised intermediates per draft",
        "MLX_E85_FUSED_EMBED": 1,
        "MLX_E85_GATHER_QMM": 1,
        "MLX_E85_BUFFER_TAX": tax,
    }


def flatten(prefix: str, value, out: dict) -> None:
    """Flatten nested report dicts into scalar W&B summary keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, (int, float, bool, str)):
        out[prefix] = value
    elif isinstance(value, list):
        out[prefix] = json.dumps(value)


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            meta[key.strip()] = value.strip()
    return meta


def read_legs(path: pathlib.Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def drafts_per_token(mean_draft_len: float, accepted_rate: float) -> float:
    return mean_draft_len / (1.0 + mean_draft_len * accepted_rate)


def base_config(meta: dict, kind: str = "abba") -> dict:
    return {
        "experiment": "e85-materialised-intermediate-elimination",
        "assignment_pr": 87,
        "assignment_id": "qwen38-r1-e85-materialised-intermediate-elimination",
        "revision_id": "r1",
        "student": "qwen-edward",
        "host": HOST,
        "harness": "local",
        "local_mode": "--local-iterate",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "design": ("palindromic dose-response inside one session"
                   if kind == "tax" else "ABBA counterbalanced inside one session"),
        **meta,
    }


def log_legs(session: pathlib.Path, meta: dict, legs: list[dict]) -> list[str]:
    urls = []
    kind = session_kind(legs)
    for row in legs:
        arm = leg_arm(row)
        mtp = as_float(row["mtp_s_per_tok"])
        serial = as_float(row["serial_s_per_tok"])
        draft_len = as_float(row["mean_draft_len"])
        accepted = as_float(row["accepted_rate"])
        dpt = drafts_per_token(draft_len, accepted)

        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP, reinit=True,
            name=f"{session.name}-leg{int(row['leg']):02d}-{arm}",
            job_type=f"e85-{kind}-leg",
            config={
                **base_config(meta, kind),
                **leg_config(row),
                "leg_index": int(row["leg"]),
            },
        )
        run.log({
            "mtp_seconds_per_token": mtp,
            "serial_seconds_per_token": serial,
            "mtp_decode_speedup": as_float(row["ratio"]),
            "effective_mean_draft_len": draft_len,
            "accepted_draft_rate": accepted,
            "drafts_per_emitted_token": dpt,
            "all_tokens_matched": row["matched"] == "True",
            "gpu_temp_entry_c": as_float(row["temp_in"]),
            "gpu_temp_exit_c": as_float(row["temp_out"]),
            "leg_wall_seconds": int(row["seconds"]),
        })
        run.summary.update({
            "arm": arm,
            "mtp_seconds_per_token": mtp,
            "serial_seconds_per_token": serial,
            "all_tokens_matched": row["matched"] == "True",
        })
        urls.append(f"{run.id}\t{run.url}")
        run.finish()
    return urls


def log_summary(session: pathlib.Path, meta: dict, legs: list[dict],
                stats: dict | None, census: dict | None,
                buffers: int) -> str:
    kind = session_kind(legs)
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, reinit=True,
        name=f"{session.name}-summary", job_type=f"e85-{kind}-summary",
        config={
            **base_config(meta, kind),
            "arms": sorted({leg_arm(row) for row in legs}),
            "legs": len(legs),
            "buffers_removed_per_draft": buffers,
            "stop_rule": "<5 us/buffer terminal negative; 5-10 report and stop; "
                         ">=10 law holds",
            "claimed_law_us_per_buffer": [13, 16],
        },
    )

    payload: dict = {}
    for arm in sorted({leg_arm(row) for row in legs}):
        sub = [row for row in legs if leg_arm(row) == arm]
        payload[f"{arm}/mtp_seconds_per_token_mean"] = statistics.fmean(
            as_float(r["mtp_s_per_tok"]) for r in sub)
        payload[f"{arm}/serial_seconds_per_token_mean"] = statistics.fmean(
            as_float(r["serial_s_per_tok"]) for r in sub)
        payload[f"{arm}/mtp_decode_speedup_mean"] = statistics.fmean(
            as_float(r["ratio"]) for r in sub)
        payload[f"{arm}/effective_mean_draft_len_mean"] = statistics.fmean(
            as_float(r["mean_draft_len"]) for r in sub)
        payload[f"{arm}/accepted_draft_rate_mean"] = statistics.fmean(
            as_float(r["accepted_rate"]) for r in sub)
        payload[f"{arm}/legs"] = len(sub)
        if len(sub) > 1:
            payload[f"{arm}/mtp_seconds_per_token_sd"] = statistics.stdev(
                [as_float(r["mtp_s_per_tok"]) for r in sub])

    if stats:
        flatten("contrast", stats, payload)
    if census:
        payload["census/answer"] = json.dumps(census)

    run.summary.update(payload)
    table = wandb.Table(columns=list(legs[0].keys()),
                        data=[list(row.values()) for row in legs])
    run.log({"legs": table})
    url = f"{run.id}\t{run.url}"
    run.finish()
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--census", default=None)
    ap.add_argument("--buffers", type=int, default=6,
                    help="net materialised intermediates removed per draft token")
    args = ap.parse_args()

    session = pathlib.Path(args.session)
    meta = read_meta(session / "session.txt")
    legs = read_legs(session / "legs.tsv")
    stats = json.loads(pathlib.Path(args.stats).read_text()) if args.stats else None
    census = json.loads(pathlib.Path(args.census).read_text()) if args.census else None

    urls = log_legs(session, meta, legs)
    urls.append(log_summary(session, meta, legs, stats, census, args.buffers))

    out = session / "wandb-runs.tsv"
    out.write_text("run_id\turl\n" + "\n".join(urls) + "\n")
    print("\n".join(urls))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
