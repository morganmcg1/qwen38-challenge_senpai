#!/usr/bin/env python3
"""E53 Part 2a: the parity wall is a survivorship filter, and the row census.

Decisive join: if `officialScore` absent AND `officialMetrics` present is
EXACTLY 0, then "every parity_ok I can see is true" is a tautology -- the
parity gate runs before scoring and a parity failure emits no metrics at all.
Item 102's parity invariance is then survivorship, not evidence of safety.

Also emits the per-prompt telemetry table Part 1 fits, and the failure
taxonomy by workflow step.

No `.get()` defaults on board fields anywhere: a missing field must raise.

Input: a live board pull (see BOARD env var / default path). Content identity
comes from E50's `/tmp/tree_ids.json` when present (git tree shas of the
submitted snapshots); rows without a content id are reported, never silently
merged.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re

BOARD = pathlib.Path(os.environ.get("E53_BOARD", "/tmp/e53/board.json"))
TREE_IDS = pathlib.Path(os.environ.get("E53_TREE_IDS", "/tmp/tree_ids.json"))
OUT = pathlib.Path(__file__).resolve().parent / "e53-board-facts.json"

# Prompt sha256 prefixes -> names (E42 census, `research/e42_width_census.py`).
NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
HEALTHY_SCORE = 2.0
STEP_RE = re.compile(r'concluded failure at step "([^"]+)"')


def load_rows() -> list[dict]:
    with BOARD.open() as handle:
        payload = json.load(handle)
    return payload["submissions"]


def prompt_name(sha: str) -> str:
    key = sha[:8]
    if key not in NAMES:
        raise KeyError(f"unknown prompt sha256 prefix {key}")
    return NAMES[key]


def content_ids() -> dict[str, str]:
    """submission id -> build-relevant content signature, from E50."""
    if not TREE_IDS.exists():
        return {}
    with TREE_IDS.open() as handle:
        blob = json.load(handle)
    if "build" not in blob:
        raise KeyError("tree_ids.json has no 'build' signature map")
    return blob["build"]


def failure_step(row: dict) -> str | None:
    reason = row["rejectionReason"]
    if reason is None:
        return None
    hit = STEP_RE.search(reason)
    if hit is not None:
        return hit.group(1)
    return reason


def main() -> None:
    rows = load_rows()
    build_id = content_ids()

    have_score = [r for r in rows if r["officialScore"] is not None]
    have_metrics = [r for r in rows if r["officialMetrics"]]
    join = {
        "rows_total": len(rows),
        "score_present_metrics_present": sum(
            1 for r in rows if r["officialScore"] is not None and r["officialMetrics"]),
        "score_absent_metrics_present": sum(
            1 for r in rows if r["officialScore"] is None and r["officialMetrics"]),
        "score_present_metrics_absent": sum(
            1 for r in rows if r["officialScore"] is not None and not r["officialMetrics"]),
        "score_absent_metrics_absent": sum(
            1 for r in rows if r["officialScore"] is None and not r["officialMetrics"]),
    }

    parity_all = collections.Counter()
    per_prompt_parity = collections.Counter()
    per_prompt_rows = 0
    for row in have_metrics:
        metrics = row["officialMetrics"]
        parity_all[metrics["parity_all_ok"]] += 1
        for entry in metrics["per_prompt"]:
            per_prompt_rows += 1
            per_prompt_parity[entry["parity_ok"]] += 1

    healthy = [r for r in have_metrics if r["officialScore"] >= HEALTHY_SCORE]

    # Content dedupe: keep the earliest submission of each build signature.
    seen: dict[str, str] = {}
    unique_healthy, unknown_identity = [], []
    for row in sorted(healthy, key=lambda r: r["createdAt"]):
        sig = build_id.get(row["id"])
        if sig is None:
            unknown_identity.append(row["id"])
            unique_healthy.append(row)
            continue
        if sig in seen:
            continue
        seen[sig] = row["id"]
        unique_healthy.append(row)

    taxonomy = collections.Counter(failure_step(r) for r in rows)

    # Per-prompt telemetry for the healthy, content-deduped rows.
    telemetry: dict[str, list[dict]] = {name: [] for name in NAMES.values()}
    rank_identity = collections.Counter()
    for row in unique_healthy:
        metrics = row["officialMetrics"]
        ratios = []
        for entry in metrics["per_prompt"]:
            name = prompt_name(entry["prompt_sha256"])
            telemetry[name].append({
                "submission": row["id"],
                "solver": row["solverUsername"],
                "created": row["createdAt"],
                "score": row["officialScore"],
                "commit": metrics["commit"],
                "head": entry["head_provenance_sha256"],
                "mean_draft_len": entry["effective_mean_draft_len"],
                "non_drafting_rounds": entry["non_drafting_round_count"],
                "raw_ratio": entry["raw_ratio_of_means"],
                "mtp_spt": entry["mtp_seconds_per_token_mean"],
                "serial_spt": entry["serial_seconds_per_token_mean"],
                "parity_ok": entry["parity_ok"],
            })
            ratios.append((entry["raw_ratio_of_means"], name))
        ratios.sort()
        # rank 4 and 5 (1-based) are the order statistics the median averages.
        rank_identity[(ratios[3][1], ratios[4][1])] += 1

    non_drafting = {
        name: collections.Counter(e["non_drafting_rounds"] for e in entries)
        for name, entries in telemetry.items()
    }

    summary = {
        "board_rows": len(rows),
        "rows_with_score": len(have_score),
        "rows_with_metrics": len(have_metrics),
        "healthy_rows": len(healthy),
        "healthy_content_unique": len(unique_healthy),
        "healthy_without_content_id": len(unknown_identity),
        "survivorship_join": join,
        "parity_all_ok": dict(parity_all),
        "per_prompt_rows": per_prompt_rows,
        "per_prompt_parity_ok": dict(per_prompt_parity),
        "failure_taxonomy": dict(taxonomy.most_common()),
        "non_drafting_round_modes": {
            name: counter.most_common(4) for name, counter in non_drafting.items()
        },
        "non_drafting_nonzero_rows": {
            name: sum(v for k, v in counter.items() if k != 0)
            for name, counter in non_drafting.items()
        },
        "central_pair_identity": {
            f"{a}+{b}": n for (a, b), n in rank_identity.most_common()
        },
    }

    with OUT.open("w") as handle:
        json.dump({"summary": summary, "telemetry": telemetry}, handle, indent=1)

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
