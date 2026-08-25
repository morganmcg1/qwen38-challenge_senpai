#!/usr/bin/env python3
"""Collect the E214 confirmation artifact from the measured run directories.

    python3 research/e214_collect_confirm.py

Reads three legs that already exist on disk and writes
`research/e214-artifacts/confirm.json` for `research/e214_wandb.py`:

  candidate   the GATED 512-token --local-submit confirmation, which carries
              exactness, the row ledger and the schedule census;
  base        the same-host BASE census leg, which is ungated and traced and
              therefore carries schedule counts only;
  delta       schedule movement between the two.

RULE 79. Seconds are recorded because they are part of the run record, and
every arm carries `secondsPerTokenIsAPrice=false`. A schedule change cannot be
priced from a local ratio; only the ranked receipt prices it.

FINDING 545's base row total is quoted as an INFERRED comparison, because the
same-host base leg is a `--local-iterate` census leg and does not emit the
CLI row report.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "e214-artifacts" / "confirm.json"
FINDING_545_BASE_ROWS = 597


def read(path: pathlib.Path):
    return json.loads(path.read_text())


def meta(run_dir: pathlib.Path) -> dict:
    fields = {}
    for line in (run_dir / "meta.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def candidate_leg() -> dict:
    run = ROOT / "research" / "out" / "e214-confirm"
    metrics = read(run / "score.json")["metrics"]
    timed = read(run / "reports" / "04-mtp-timed.json")
    info = meta(run)
    declared = timed["declared_rows_total"]
    checked = timed["reference_checked_row_total"]
    parts = (timed["accepted_draft_total"] + timed["rejected_draft_total"]
             + timed["target_tail_total"])
    return {
        "mode": metrics["mode"],
        "tokens": metrics["decode_tokens"],
        "allTokensMatched": metrics["all_tokens_matched"],
        "residualDivergenceCount": metrics["residual_divergence_count"],
        "roundCount": timed["round_count"],
        "effectiveMeanDraftLen": metrics["effective_mean_draft_len"],
        "effectiveMaxDraftLen": timed["effective_max_draft_len"],
        "acceptedDraftRate": metrics["accepted_draft_rate"],
        "acceptedDraftTotal": timed["accepted_draft_total"],
        "rejectedDraftTotal": timed["rejected_draft_total"],
        "targetTailTotal": timed["target_tail_total"],
        "nonDraftingRoundCount": timed["non_drafting_round_count"],
        "declaredRowsTotal": declared,
        "referenceCheckedRowTotal": checked,
        "rowLedgerClosed": declared == checked == parts,
        "rowLedgerParts": parts,
        "maxRejectedTailLogitDelta": timed["max_rejected_tail_logit_delta"],
        "parityAllOk": timed["parity_all_ok"],
        "emittedTokenTotal": timed["emitted_token_total"],
        "seedTokenCount": timed["seed_token_count"],
        "targetCacheOffsetFinal": timed["target_cache_offset_final"],
        "verifyBlockReplayedRoundCount": timed["verify_block_replayed_round_count"],
        "mtpSecondsPerToken": metrics["mtp_seconds_per_token"],
        "serialSecondsPerToken": metrics["serial_seconds_per_token"],
        "localRatio": metrics["mtp_decode_speedup"],
        "prefillSecondsPerToken": timed["prefill_seconds_per_token"],
        "firstBlockSeconds": timed["first_block_seconds"],
        "p50BlockRequestSeconds": timed["p50_block_request_seconds"],
        "decodeSeconds": timed["decode_seconds"],
        "headProvenanceSha256": metrics["head_provenance_sha256"],
        "candidateSha": info["candidate_sha"],
        "workerSha256": info["worker_sha256"],
        "dirtyCandidatePaths": int(info["dirty_candidate_paths"]),
        "host": info["host"],
        "chip": info["chip"],
        "gpuTempEntryPreGateC": float(info["gpu_temp_entry_c"]),
        "gpuTempExitC": float(info["gpu_temp_exit_c"]),
        "coolGatePassedRealGate": True,
        "gateQualifiedForTiming": True,
        "secondsPerTokenIsAPrice": False,
        "officialOrRankedScore": False,
    }


def base_leg() -> dict:
    run = ROOT / "research" / "out" / "e214-base-census"
    metrics = read(run / "score.json")["metrics"]
    info = meta(run)
    trace = (run / "trace.txt").read_text().splitlines()
    rounds = sum(1 for line in trace if line.startswith("mtp-anchor:"))
    return {
        "mode": metrics["mode"],
        "tokens": metrics["decode_tokens"],
        "allTokensMatched": metrics["all_tokens_matched"],
        "residualDivergenceCount": metrics["residual_divergence_count"],
        "roundCount": rounds,
        "effectiveMeanDraftLen": metrics["effective_mean_draft_len"],
        "acceptedDraftRate": metrics["accepted_draft_rate"],
        "mtpSecondsPerToken": metrics["mtp_seconds_per_token"],
        "serialSecondsPerToken": metrics["serial_seconds_per_token"],
        "localRatio": metrics["mtp_decode_speedup"],
        "headProvenanceSha256": metrics["head_provenance_sha256"],
        "workerSha256": info["worker_sha256"],
        "surfaceEqualsBase": info["arm_surface_equals_base"] == "true",
        "printedPriceArm": info["printed_price_arm"],
        "host": info["host"],
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "secondsPerTokenIsAPrice": False,
        "rowsTotalFinding545Inferred": FINDING_545_BASE_ROWS,
    }


def main() -> int:
    cand = candidate_leg()
    base = base_leg()
    delta = {
        "roundsAbs": cand["roundCount"] - base["roundCount"],
        "roundsPct": 100.0 * (cand["roundCount"] / base["roundCount"] - 1.0),
        "edlPct": 100.0 * (cand["effectiveMeanDraftLen"]
                           / base["effectiveMeanDraftLen"] - 1.0),
        "acceptedRatePp": 100.0 * (cand["acceptedDraftRate"]
                                   - base["acceptedDraftRate"]),
        "rowsAbsVsFinding545Inferred":
            cand["declaredRowsTotal"] - FINDING_545_BASE_ROWS,
        "mtpSecondsPerTokenPctNotAPrice":
            100.0 * (cand["mtpSecondsPerToken"]
                     / base["mtpSecondsPerToken"] - 1.0),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"harness": "local", "candidate": cand, "base": base, "delta": delta},
        indent=1, sort_keys=True) + "\n")
    print("wrote %s" % OUT)
    for key, value in delta.items():
        print("  %-34s %s" % (key, value))
    print("  row ledger closed                  %s (%d declared, %d checked)"
          % (cand["rowLedgerClosed"], cand["declaredRowsTotal"],
             cand["referenceCheckedRowTotal"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
