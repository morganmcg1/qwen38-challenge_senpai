#!/usr/bin/env python3
"""E134 item 5: collect the pb6 pre-submit evidence into one artifact.

This reads the archived exactness legs, checks the row ledger closes on every
leg, checks the provenance tuple is single-valued across legs, and records the
shipped depth histogram next to the `ship` archive so the arm's mechanism is
visible. It runs no GPU work and decides nothing about the effect size, which
Rule 79 reserves for a ranked receipt.

    python3 research/e134_item5_presubmit.py \
        --pb6 .mlxfast-private/e128/runs-pb6 \
        --ship .mlxfast-private/e128/runs-shipped \
        --json research/e134-artifacts/item5-presubmit.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e134_rung1 import MAX_DEPTH, parse_trace  # noqa: E402

# `meta.txt` fields that must hold exactly one value across every leg, so the
# archive cannot silently mix two builds or two heads.
SINGLE_VALUED = ("worker_sha256", "cli_sha256", "head_provenance_sha256",
                 "head_manifest_tree_sha256", "base_sha", "chip", "host",
                 "harness", "timing_valid", "gate_qualified_for_timing",
                 "official_or_ranked_score", "dirty_candidate_paths")


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def histogram(rounds: list[dict], key: str) -> dict:
    counts = {}
    for record in rounds:
        counts[record[key]] = counts.get(record[key], 0) + 1
    return {str(d): counts.get(d, 0) for d in range(MAX_DEPTH + 1)}


def collect(runs: pathlib.Path) -> dict:
    legs = {}
    for child in sorted(runs.iterdir()):
        meta_path, report_path = child / "meta.txt", child / "report.json"
        if not (meta_path.is_file() and report_path.is_file()):
            continue
        meta = read_meta(meta_path)
        report = json.loads(report_path.read_text())
        rounds, gate = parse_trace(child / "trace.txt")
        declared = report["declared_rows_total"]
        checked = report["reference_checked_row_total"]
        legs[child.name] = {
            "fixture": child.name,
            "tokens": int(meta["tokens"]),
            "base_sha": meta["base_sha"],
            "all_tokens_matched": report["all_tokens_matched"],
            "residual_divergence_count": report["residual_divergence_count"],
            "parity_all_ok": report["parity_all_ok"],
            "rounds": report["round_count"],
            "declared_rows_total": declared,
            "reference_checked_row_total": checked,
            "target_tail_total": report["target_tail_total"],
            "emitted_token_total": report["emitted_token_total"],
            "decode_token_count": report["decode_token_count"],
            "target_cache_offset_final": report["target_cache_offset_final"],
            "max_rejected_tail_logit_delta":
                report["max_rejected_tail_logit_delta"],
            "rejected_rows_reference_checked":
                report["rejected_rows_reference_checked"],
            "accepted_draft_total": report["accepted_draft_total"],
            "rejected_draft_total": report["rejected_draft_total"],
            "accepted_draft_rate": report["accepted_draft_rate"],
            "effective_mean_draft_len": report["effective_mean_draft_len"],
            "effective_max_draft_len": report["effective_max_draft_len"],
            "non_drafting_round_count": report["non_drafting_round_count"],
            "uses_pinned_mtp_head": report["uses_pinned_mtp_head"],
            "mtp_head_attached": report["mtp_head_attached"],
            "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
            "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
            "row_count_bad": gate["row_count_bad"],
            "margin_identity_bad": gate["margin_identity_bad"],
            "sched_max_abs_error": gate["sched_max_abs_error"],
            "mean_offered_depth": float(np.mean([r["depth"]
                                                 for r in rounds])),
            "mean_accepted": float(np.mean([r["acc"] for r in rounds])),
            "offered_depth_histogram": histogram(rounds, "depth"),
            "meta": {k: meta.get(k) for k in SINGLE_VALUED},
        }
    return legs


def provenance(legs: dict) -> dict:
    out = {}
    for field in SINGLE_VALUED:
        values = sorted({leg["meta"].get(field) for leg in legs.values()})
        out[field] = {"values": values, "single_valued": len(values) == 1}
    return out


def deep_share(leg: dict) -> float:
    hist = leg["offered_depth_histogram"]
    total = sum(hist.values())
    deep = sum(v for k, v in hist.items() if int(k) >= 5)
    return deep / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb6", required=True)
    parser.add_argument("--ship", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()

    pb6 = collect(pathlib.Path(args.pb6))
    ship = collect(pathlib.Path(args.ship))
    prov = provenance(pb6)

    print(f"{'fixture':<20}{'tok':>5}{'match':>7}{'div':>5}{'parity':>8}"
          f"{'rounds':>8}{'declared':>10}{'checked':>9}{'tail':>7}"
          f"{'rowbad':>8}")
    for leg in pb6.values():
        print(f"{leg['fixture']:<20}{leg['tokens']:>5}"
              f"{str(leg['all_tokens_matched']):>7}"
              f"{leg['residual_divergence_count']:>5}"
              f"{str(leg['parity_all_ok']):>8}{leg['rounds']:>8}"
              f"{leg['declared_rows_total']:>10}"
              f"{leg['reference_checked_row_total']:>9}"
              f"{leg['target_tail_total']:>7}{leg['row_count_bad']:>8}")

    print(f"\n{'fixture':<20}{'emitted':>9}{'decode':>8}{'cacheend':>10}"
          f"{'rejtail':>10}{'maxlogitd':>12}")
    for leg in pb6.values():
        print(f"{leg['fixture']:<20}{leg['emitted_token_total']:>9}"
              f"{leg['decode_token_count']:>8}"
              f"{leg['target_cache_offset_final']:>10}"
              f"{leg['rejected_rows_reference_checked']:>10}"
              f"{leg['max_rejected_tail_logit_delta']:>12.3e}")

    print("\nprovenance, one value per field across every pb6 leg")
    for field, blob in prov.items():
        mark = "OK " if blob["single_valued"] else "BAD"
        shown = blob["values"][0] if blob["single_valued"] \
            else ",".join(str(v) for v in blob["values"])
        print(f"  {mark} {field:<32}{shown}")

    print(f"\nmechanism: pb6 against the ship archive on shared fixtures")
    print(f"{'fixture':<20}{'d pb6':>8}{'d ship':>9}{'delta':>9}"
          f"{'acc pb6':>10}{'acc ship':>10}{'R pb6':>8}{'R ship':>9}"
          f"{'deep pb6':>10}{'deep ship':>11}")
    shared = [f for f in pb6 if f in ship]
    for fixture in shared:
        a, b = pb6[fixture], ship[fixture]
        print(f"{fixture:<20}{a['mean_offered_depth']:>8.3f}"
              f"{b['mean_offered_depth']:>9.3f}"
              f"{a['mean_offered_depth'] - b['mean_offered_depth']:>+9.3f}"
              f"{a['mean_accepted']:>10.3f}{b['mean_accepted']:>10.3f}"
              f"{a['rounds']:>8}{b['rounds']:>9}"
              f"{deep_share(a):>10.3f}{deep_share(b):>11.3f}")

    print(f"\n{'fixture':<20}{'arm':<6}" +
          "".join(f"{d:>7}" for d in range(MAX_DEPTH + 1)))
    for fixture in shared:
        for arm, blob in (("pb6", pb6), ("ship", ship)):
            hist = blob[fixture]["offered_depth_histogram"]
            print(f"{fixture:<20}{arm:<6}" +
                  "".join(f"{hist[str(d)]:>7}" for d in range(MAX_DEPTH + 1)))

    matched = all(leg["all_tokens_matched"] for leg in pb6.values())
    diverged = sum(leg["residual_divergence_count"] for leg in pb6.values())
    rowbad = sum(leg["row_count_bad"] for leg in pb6.values())
    ledger_closed = all(leg["reference_checked_row_total"]
                        >= leg["declared_rows_total"]
                        for leg in pb6.values())
    summary = {
        "exactness_legs": len(pb6),
        "exactness_tokens": sorted({leg["tokens"] for leg in pb6.values()}),
        "all_tokens_matched_every_leg": matched,
        "residual_divergence_total": diverged,
        "trace_row_ledger_bad_rounds": rowbad,
        "declared_rows_all_reference_checked": ledger_closed,
        "parity_all_ok_every_leg": all(leg["parity_all_ok"]
                                       for leg in pb6.values()),
        "pinned_head_every_leg": all(leg["uses_pinned_mtp_head"]
                                     for leg in pb6.values()),
        "provenance_single_valued": all(b["single_valued"]
                                        for b in prov.values()),
        "post_eos_continuation_exercised": False,
        "post_eos_note": "No local fixture emits an EOS or other special id "
                         "inside its 512-token window, so these legs cannot "
                         "exercise post-EOS continuation. pb6 changes only "
                         "the depth price constant and touches no "
                         "continuation code.",
        "mean_offered_depth_delta_vs_ship": {
            fixture: pb6[fixture]["mean_offered_depth"]
            - ship[fixture]["mean_offered_depth"] for fixture in shared},
        "deep_round_share_delta_vs_ship": {
            fixture: deep_share(pb6[fixture]) - deep_share(ship[fixture])
            for fixture in shared},
    }
    print("\nsummary")
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"  {key:<42}{value}")

    blob = {"harness": "local", "arm": "pb6", "summary": summary,
            "provenance": prov, "legs": list(pb6.values()),
            "ship_reference": {f: ship[f] for f in shared}}
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
