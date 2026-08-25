#!/usr/bin/env python3
"""Research-only (qwen38-r1-e215): reduce the E215 legs to the three artifacts.

usage: research/e215_collect.py [--out research/e215-artifacts]

Reads `research/out/e215-<prompt>-<arm>/` and writes:

  exactness.json          Q1: per leg, the fixed-window verdict -- matched,
                          residual divergences, stop-token positions inside the
                          window, tokens after the first stop token, row-ledger
                          closure and the final target cache offset.
  census.json             Q2: per prompt and arm, rounds / effective mean draft
                          length / accepted rate, and the stepq-minus-ship delta.
  width-census-stepq.json Deliverable 4: rounds by SERVED VERIFY WIDTH for the
                          composed base, per prompt, with the leg identity tuple.

Seconds are NOT read from these legs (RULE 79): the legs are ungated, traced
and unsandboxed, so they carry no price.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "research/out"
STOP_TOKENS = {248044, 248046}
ARMS = ("stepq", "ship")


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def load_reports(leg_dir: pathlib.Path) -> tuple[dict, dict, dict]:
    """Return (reference rows, serial control report, MTP decode report)."""
    reference, serial, mtp = None, None, None
    for path in sorted((leg_dir / "reports").glob("*.json")):
        if path.stat().st_size == 0:
            continue
        doc = json.loads(path.read_text())
        if "rows" in doc and "emitted_tokens" in doc:
            reference = doc
        elif doc.get("verb") == "mtp-timed":
            if doc.get("is_serial_control"):
                serial = doc
            else:
                mtp = doc
    if reference is None or mtp is None:
        raise SystemExit(f"{leg_dir}: missing reference rows or MTP report")
    return reference, serial, mtp


def leg_summary(prompt: str, arm: str) -> dict:
    leg_dir = OUT_ROOT / f"e215-{prompt}-{arm}"
    meta = read_meta(leg_dir / "meta.txt")
    reference, serial, mtp = load_reports(leg_dir)

    window = mtp["decode_token_count"]
    emitted = reference["emitted_tokens"][:window]
    stops = [i for i, token in enumerate(emitted) if token in STOP_TOKENS]
    drafts = mtp["effective_draft_lengths"]
    ledger_parts = (
        mtp["accepted_draft_total"] + mtp["rejected_draft_total"] + mtp["round_count"]
    )

    return {
        "prompt": prompt,
        "arm": arm,
        "harness": "local",
        "fixture": meta["fixture"],
        "tokens": window,
        "printed_price_arm": meta["printed_price_arm"],
        "arm_certified": meta["printed_price_arm"] == arm,
        "worker_sha256": meta["worker_sha256"],
        "post_run_worker_sha256": meta["post_run_worker_sha256"],
        "worker_unchanged_during_leg": meta["worker_sha256"]
        == meta["post_run_worker_sha256"],
        "branch_sha": meta["branch_sha"],
        "base_sha": meta["base_sha"],
        "host": meta["host"],
        "chip": meta["chip"],
        "memory_bytes": int(meta["memory_bytes"]),
        "head_dir_sha256": meta["head_dir_sha256"],
        "metallib_sha256": meta["metallib_sha256"],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "seconds_per_token_is_a_price": False,
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "started": meta["started"],
        "finished": meta["finished"],
        # Q1 -- fixed-window exactness through and past the stop token.
        "all_tokens_matched": mtp["all_tokens_matched"],
        "residual_divergence_count": mtp["residual_divergence_count"],
        "parity_all_ok": mtp["parity_all_ok"],
        "emitted_token_total": mtp["emitted_token_total"],
        "seed_token_count": mtp["seed_token_count"],
        "target_cache_offset_final": mtp["target_cache_offset_final"],
        "declared_rows_total": mtp["declared_rows_total"],
        "reference_checked_row_total": mtp["reference_checked_row_total"],
        "rejected_rows_reference_checked": mtp["rejected_rows_reference_checked"],
        "row_ledger_parts": ledger_parts,
        "row_ledger_closed": ledger_parts == mtp["declared_rows_total"]
        == mtp["reference_checked_row_total"],
        "stop_token_count_in_window": len(stops),
        "first_stop_token_index": stops[0] if stops else None,
        "first_stop_token_id": emitted[stops[0]] if stops else None,
        "tokens_after_first_stop_token": (window - 1 - stops[0]) if stops else 0,
        "stop_token_positions": stops,
        "serial_control_all_tokens_matched": (
            serial["all_tokens_matched"] if serial else None
        ),
        # Q2 -- schedule census.
        "round_count": mtp["round_count"],
        "effective_mean_draft_len": mtp["effective_mean_draft_len"],
        "effective_max_draft_len": mtp["effective_max_draft_len"],
        "accepted_draft_rate": mtp["accepted_draft_rate"],
        "accepted_draft_total": mtp["accepted_draft_total"],
        "rejected_draft_total": mtp["rejected_draft_total"],
        "non_drafting_round_count": mtp["non_drafting_round_count"],
        "verify_block_replayed_round_count": mtp["verify_block_replayed_round_count"],
        "served_verify_widths": [d + 1 for d in drafts],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "research/e215-artifacts"))
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    for path in sorted(OUT_ROOT.glob("e215-*-stepq")):
        prompts.append(path.name[len("e215-") : -len("-stepq")])
    if not prompts:
        raise SystemExit("no E215 legs found under research/out")

    legs = {}
    for prompt in prompts:
        for arm in ARMS:
            legs[(prompt, arm)] = leg_summary(prompt, arm)

    exactness = {
        "harness": "local",
        "experiment": "e215-schedule-evidence",
        "legs": [
            {k: v for k, v in legs[key].items() if k != "served_verify_widths"}
            for key in sorted(legs)
        ],
    }
    exactness["all_legs_matched"] = all(
        leg["all_tokens_matched"]
        and leg["residual_divergence_count"] == 0
        and leg["row_ledger_closed"]
        and leg["arm_certified"]
        and leg["worker_unchanged_during_leg"]
        for leg in exactness["legs"]
    )

    census = {"harness": "local", "seconds_are_a_price": False, "prompts": {}}
    for prompt in prompts:
        stepq, ship = legs[(prompt, "stepq")], legs[(prompt, "ship")]
        census["prompts"][prompt] = {
            "fixture": stepq["fixture"],
            "stepq": {
                "rounds": stepq["round_count"],
                "effective_mean_draft_len": stepq["effective_mean_draft_len"],
                "accepted_draft_rate": stepq["accepted_draft_rate"],
                "accepted_draft_total": stepq["accepted_draft_total"],
                "rejected_draft_total": stepq["rejected_draft_total"],
                "declared_rows_total": stepq["declared_rows_total"],
            },
            "ship": {
                "rounds": ship["round_count"],
                "effective_mean_draft_len": ship["effective_mean_draft_len"],
                "accepted_draft_rate": ship["accepted_draft_rate"],
                "accepted_draft_total": ship["accepted_draft_total"],
                "rejected_draft_total": ship["rejected_draft_total"],
                "declared_rows_total": ship["declared_rows_total"],
            },
            "delta": {
                "rounds_abs": stepq["round_count"] - ship["round_count"],
                "rounds_pct": 100.0
                * (stepq["round_count"] - ship["round_count"])
                / ship["round_count"],
                "edl_pct": 100.0
                * (
                    stepq["effective_mean_draft_len"]
                    - ship["effective_mean_draft_len"]
                )
                / ship["effective_mean_draft_len"],
                "accepted_rate_pp": 100.0
                * (stepq["accepted_draft_rate"] - ship["accepted_draft_rate"]),
                "rows_abs": stepq["declared_rows_total"] - ship["declared_rows_total"],
            },
        }
    directions = [
        entry["delta"]["rounds_abs"] <= 0 and entry["delta"]["edl_pct"] >= 0
        for entry in census["prompts"].values()
    ]
    census["direction_holds_on_every_prompt"] = all(directions)
    census["prompts_with_direction"] = sum(directions)
    census["prompt_count"] = len(directions)

    width_census = {
        "harness": "local",
        "arm": "stepq",
        "note": (
            "Rounds by served verify width on the composed base "
            "(cap-8 + staged (9,5) + stepq). Served verify width is "
            "effective_draft_lengths[i] + 1: the drafted rows plus the bonus "
            "row the round always evaluates. Untimed legs, RULE 79: no "
            "seconds in this artifact."
        ),
        "identity": {
            "base_sha": legs[(prompts[0], "stepq")]["base_sha"],
            "branch_sha": legs[(prompts[0], "stepq")]["branch_sha"],
            "session_blob": "48d190f991aeff4c32412c762764bf83bf541deb",
            "worker_sha256": legs[(prompts[0], "stepq")]["worker_sha256"],
            "metallib_sha256": legs[(prompts[0], "stepq")]["metallib_sha256"],
            "head_dir_sha256": legs[(prompts[0], "stepq")]["head_dir_sha256"],
            "host": legs[(prompts[0], "stepq")]["host"],
            "chip": legs[(prompts[0], "stepq")]["chip"],
            "memory_bytes": legs[(prompts[0], "stepq")]["memory_bytes"],
            "local_mode": "--local-iterate",
            "tokens": legs[(prompts[0], "stepq")]["tokens"],
            "depth_offered": 8,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
        },
        "prompts": {},
    }
    pooled: collections.Counter = collections.Counter()
    for prompt in prompts:
        widths = collections.Counter(legs[(prompt, "stepq")]["served_verify_widths"])
        pooled.update(widths)
        total = sum(widths.values())
        width_census["prompts"][prompt] = {
            "fixture": legs[(prompt, "stepq")]["fixture"],
            "rounds": total,
            "rounds_by_served_width": {str(w): widths[w] for w in sorted(widths)},
            "share_by_served_width": {
                str(w): widths[w] / total for w in sorted(widths)
            },
            "rows_by_served_width": {str(w): w * widths[w] for w in sorted(widths)},
        }
    pooled_total = sum(pooled.values())
    width_census["pooled_all_prompts"] = {
        "rounds": pooled_total,
        "rounds_by_served_width": {str(w): pooled[w] for w in sorted(pooled)},
        "share_by_served_width": {
            str(w): pooled[w] / pooled_total for w in sorted(pooled)
        },
        "rows_by_served_width": {str(w): w * pooled[w] for w in sorted(pooled)},
    }

    for name, payload in (
        ("exactness.json", exactness),
        ("census.json", census),
        ("width-census-stepq.json", width_census),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_dir / name}")

    print(f"all_legs_matched={exactness['all_legs_matched']}")
    for prompt, entry in census["prompts"].items():
        print(
            f"{prompt}: rounds {entry['ship']['rounds']} -> {entry['stepq']['rounds']} "
            f"({entry['delta']['rounds_abs']:+d}), EDL {entry['delta']['edl_pct']:+.3f}%, "
            f"acc {entry['delta']['accepted_rate_pp']:+.3f} pp"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
