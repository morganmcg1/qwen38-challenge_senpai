"""E157 R0: attribute the `75a21a4` shortfall to the base or to the scheduler.

Read-only analysis of the public Yukon receipts. It reads the payload that
`research/board_per_prompt.py fetch` writes to /tmp/yukon-board/full.json and
never calls a Yukon mutation.

THE SCHEDULE IS PUBLISHED, NOT INFERRED. The trusted driver builds
`effective_mean_draft_len` from `rounds.map { $0.draftTokens.count }`
(`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift:295`), so the
receipt field is the mean DRAFTED width per round, not the accepted count.
Every receipt therefore carries the realised per-prompt draft schedule
directly, at full precision, with no fitted parameter and no scale factor.

Sign convention in words: a negative percentage means the first receipt is
LOWER (worse for a score, faster for a time) than the reference receipt named
in the same row. Every number here is harness=ranked unless labelled otherwise.

    python3 research/e157_e150_attribution.py [CANDIDATE_ID8 REFERENCE_ID8]
"""
from __future__ import annotations

import json
import os
import statistics
import sys

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def load_rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["submissions"]


def receipt(rows: list[dict], prefix: str) -> dict:
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{prefix}: {len(hits)} matches, need exactly one")
    return hits[0]


def per_prompt(row: dict) -> dict:
    out = {}
    for entry in row["officialMetrics"]["per_prompt"]:
        name = PROMPT_NAMES.get(entry["prompt_sha256"][:8], entry["prompt_sha256"][:8])
        out[name] = entry
    return out


def median8(values: list[float]) -> float:
    ordered = sorted(values)
    return (ordered[3] + ordered[4]) / 2.0


def attribution_verdict(max_abs_edl_delta: float, table: list[dict]) -> str:
    """Case A (base handicap) against Case B (scheduler under-delivery).

    Case B requires the realised schedule to differ from the schedule the
    reference receipt ran. The receipt publishes that schedule directly, so
    the test is an exact comparison and not an inference.
    """
    if max_abs_edl_delta > 0.0:
        slower_with_same_schedule = [
            row
            for row in table
            if row["edl_delta"] == 0.0 and row["cand_spt_pct_vs_ref"] > 0.0
        ]
        return "both" if slower_with_same_schedule else "scheduler_dominated"
    if all(row["cand_spt_pct_vs_ref"] > 0.0 for row in table):
        return "base_dominated"
    return "unresolvable_from_held_evidence"


def main() -> None:
    cand_id = sys.argv[1] if len(sys.argv) > 1 else "75a21a4"
    ref_id = sys.argv[2] if len(sys.argv) > 2 else "0cf1637e"
    rows = load_rows()
    cand, ref = receipt(rows, cand_id), receipt(rows, ref_id)
    cp, rp = per_prompt(cand), per_prompt(ref)

    names = sorted(cp)
    table = []
    for name in names:
        c, r = cp[name], rp[name]
        table.append(
            {
                "prompt": name,
                "harness": "ranked",
                "edl_candidate": c["effective_mean_draft_len"],
                "edl_reference": r["effective_mean_draft_len"],
                "edl_delta": c["effective_mean_draft_len"]
                - r["effective_mean_draft_len"],
                "non_drafting_candidate": c["non_drafting_round_count"],
                "non_drafting_reference": r["non_drafting_round_count"],
                "cand_spt": c["mtp_seconds_per_token_mean"],
                "ref_spt": r["mtp_seconds_per_token_mean"],
                "cand_spt_pct_vs_ref": 100.0
                * (c["mtp_seconds_per_token_mean"] / r["mtp_seconds_per_token_mean"] - 1.0),
                "serial_candidate": c["serial_seconds_per_token_mean"],
                "serial_reference": r["serial_seconds_per_token_mean"],
                "serial_pct_vs_ref": 100.0
                * (
                    c["serial_seconds_per_token_mean"]
                    / r["serial_seconds_per_token_mean"]
                    - 1.0
                ),
                "raw_candidate": c["raw_ratio_of_means"],
                "raw_reference": r["raw_ratio_of_means"],
                "raw_pct_vs_ref": 100.0
                * (c["raw_ratio_of_means"] / r["raw_ratio_of_means"] - 1.0),
            }
        )

    # Published medians, reproduced from the per-prompt rows.
    med_cand = median8([row["raw_candidate"] for row in table])
    med_ref = median8([row["raw_reference"] for row in table])

    # Serial-free counterfactual: give the candidate the reference run's serial
    # draw on every prompt, so only the candidate leg differs.
    serialfree = median8(
        [row["serial_reference"] / row["cand_spt"] for row in table]
    )

    # Schedule-free counterfactual is not needed when the schedules are equal;
    # report the exact equality test instead.
    identical = [row for row in table if row["edl_delta"] == 0.0]
    max_abs_edl_delta = max(abs(row["edl_delta"]) for row in table)

    slot_cand = sorted(table, key=lambda row: row["raw_candidate"])[3:5]
    slot_ref = sorted(table, key=lambda row: row["raw_reference"])[3:5]

    result = {
        "experiment": "e157-r0-e150-shortfall-attribution",
        "harness": "ranked",
        "official_or_ranked_score": True,
        "source": "Yukon list endpoint officialMetrics.per_prompt, read only",
        "candidate": {
            "id8": cand["id"][:8],
            "official_score": cand["officialScore"],
            "commit": cand["submissionCommitSha"],
            "created_utc": cand["createdAt"],
            "status": cand["status"],
        },
        "reference": {
            "id8": ref["id"][:8],
            "official_score": ref["officialScore"],
            "commit": ref["submissionCommitSha"],
            "created_utc": ref["createdAt"],
            "status": ref["status"],
        },
        "median_reproduction": {
            "candidate_from_per_prompt": med_cand,
            "candidate_published": cand["officialScore"],
            "reference_from_per_prompt": med_ref,
            "reference_published": ref["officialScore"],
        },
        "median_pair_candidate": [row["prompt"] for row in slot_cand],
        "median_pair_reference": [row["prompt"] for row in slot_ref],
        "e157_e150_recovered_schedule": {
            row["prompt"]: row["edl_candidate"] for row in table
        },
        "e157_e150_intended_schedule": {
            row["prompt"]: row["edl_reference"] for row in table
        },
        "e157_schedule_transfer_faithful": max_abs_edl_delta == 0.0,
        "e157_e150_shortfall_attribution": attribution_verdict(
            max_abs_edl_delta, table
        ),
        "schedule_identical_prompt_count": len(identical),
        "max_abs_edl_delta": max_abs_edl_delta,
        "per_prompt": table,
        "attribution": {
            "published_delta_pct": 100.0 * (med_cand / med_ref - 1.0),
            "serial_free_delta_pct": 100.0 * (serialfree / med_ref - 1.0),
            "serial_draw_share_pct": 100.0 * (med_cand / med_ref - serialfree / med_ref),
            "candidate_leg_mean_pct": statistics.fmean(
                [row["cand_spt_pct_vs_ref"] for row in table]
            ),
            "serial_draw_mean_pct": statistics.fmean(
                [row["serial_pct_vs_ref"] for row in table]
            ),
        },
    }

    out_dir = "research/e157-artifacts"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"e157_attribution_{cand['id'][:8]}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(
        f"candidate {cand['id'][:8]} {cand['officialScore']:.8f}  "
        f"reference {ref['id'][:8]} {ref['officialScore']:.8f}"
    )
    print(
        f"{'prompt':9s} {'edl cand':>9s} {'edl ref':>9s} {'d edl':>7s} "
        f"{'cand s/tok %':>12s} {'serial %':>9s} {'raw %':>8s}"
    )
    for row in table:
        print(
            f"{row['prompt']:9s} {row['edl_candidate']:9.6f} {row['edl_reference']:9.6f} "
            f"{row['edl_delta']:+7.4f} {row['cand_spt_pct_vs_ref']:+12.4f} "
            f"{row['serial_pct_vs_ref']:+9.4f} {row['raw_pct_vs_ref']:+8.4f}"
        )
    print(
        f"median pair candidate {result['median_pair_candidate']} "
        f"reference {result['median_pair_reference']}"
    )
    print(json.dumps(result["attribution"], indent=1))
    print(f"schedule identical on {len(identical)}/8 prompts, "
          f"max |d edl| = {max_abs_edl_delta}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
