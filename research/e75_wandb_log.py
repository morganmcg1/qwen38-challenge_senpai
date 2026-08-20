#!/usr/bin/env python3
"""Log one E75 leg to W&B, immediately after it is measured.

One W&B run per leg, all in one group, so a session that dies on leg 5 still
leaves legs 1-4 on the board.

    research/e75_wandb_log.py --leg .mlxfast-private/e75-e2e/runs/TAG

Everything comes from artifacts the leg runner already wrote: `meta.txt` for
the identity tuple, `score.json` for the metrics, `arm.json` for the resolved
price vector, and the capture reports for the emitted stream, the row ledger
and the realised width histogram.

E75 adds three fields the E68 logger did not carry, because E75 is a 2x2 over
the kernel dispatch table as well as the depth price: `kernel_table`, the
emitted-stream digest, and the row-ledger closure counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import wandb

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e68_wandb_log import (  # noqa: E402
    ENTITY,
    PROJECT,
    git,
    read_meta,
    verify_width_histogram,
)

ASSIGNMENT = {
    "assignment_id": "qwen38-r1-e75-bank-pbfit-and-price-it-on-the-crown-table",
    "revision_id": "r1",
    "pr_number": 78,
    "student": "qwen-thorfinn",
    "host_chip": "Apple M4 Pro",
    "harness": "local",
    "local_mode": "--local-iterate",
    "scored_files": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
}

GROUP = "e75-bank-pbfit-and-price-it-on-the-crown-table"


def emitted_stream(leg_dir):
    """Digest of the emitted token stream, and its length.

    Two legs that emit the same tokens must produce the same digest; that is
    the exactness evidence a schedule arm owes, because a schedule change is
    only legal if it changes timing and nothing else.
    """
    path = leg_dir / "reports/02-mtp-verify-output.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    tokens = payload.get("emitted_tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    body = ",".join(str(int(t)) for t in tokens)
    return {
        "emitted_token_count": len(tokens),
        "emitted_stream_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }


def timed_report(leg_dir):
    path = leg_dir / "reports/04-mtp-timed.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--rung", default="D")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble config and metrics, print them, log nothing")
    args = ap.parse_args()

    leg_dir = pathlib.Path(args.leg).resolve()
    meta = read_meta(leg_dir / "meta.txt")
    arm = json.loads((leg_dir / "arm.json").read_text())

    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})
    timed = timed_report(leg_dir)

    config = dict(ASSIGNMENT)
    config.update({
        "rung": args.rung,
        "leg_tag": meta.get("tag"),
        "arm": meta.get("arm"),
        "depth_price_arm": meta.get("arm"),
        "kernel_table": meta.get("kernel_table", "ours"),
        "cell": "%s-%s" % (meta.get("kernel_table", "ours"), meta.get("arm")),
        "arm_role": arm.get("role"),
    })
    config.update({"meta_" + k: v for k, v in meta.items()})
    config["marginal"] = arm.get("marginal")
    config["marginal_total"] = arm.get("marginal_total")
    config["measured_raw"] = arm.get("measured_raw")
    config["head_sha"] = git("rev-parse", "HEAD")
    config["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")

    for name, block in (arm.get("predicted_depth") or {}).items():
        config["predicted_%s_p" % name] = block["p"]
        config["predicted_%s_depth_default_cap" % name] = \
            block["default_cap"]["depth"]
        config["predicted_%s_depth_streak_cap" % name] = \
            block["streak_cap"]["depth"]

    log = {
        # The primary. A depth-price change is confined to the candidate MTP
        # leg, so absolute candidate time is the causal quantity; the local
        # ratio is reported beside it, never instead of it. A kernel-table
        # change is NOT confined to that leg, which is why the 2x2 reads the
        # absolute number first.
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "decode_tokens": metrics.get("decode_tokens"),
        "score": score.get("score"),
        "passed": score.get("passed"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "stale_metallib_warnings": meta.get("stale_metallib_warnings"),
        "leg_status": meta.get("status"),
        # Row ledger and exactness.
        "parity_all_ok": timed.get("parity_all_ok"),
        "declared_rows_total": timed.get("declared_rows_total"),
        "reference_checked_row_total": timed.get("reference_checked_row_total"),
        "rejected_rows_reference_checked":
            timed.get("rejected_rows_reference_checked"),
        "target_tail_total": timed.get("target_tail_total"),
        "round_count": timed.get("round_count"),
        "non_drafting_round_count": timed.get("non_drafting_round_count"),
        "verify_block_replayed_round_count":
            timed.get("verify_block_replayed_round_count"),
        "accepted_draft_total": timed.get("accepted_draft_total"),
        "rejected_draft_total": timed.get("rejected_draft_total"),
        "p50_block_request_seconds": timed.get("p50_block_request_seconds"),
        "max_block_request_seconds_after_first":
            timed.get("max_block_request_seconds_after_first"),
        "decode_seconds": timed.get("decode_seconds"),
        "seed_prefill_seconds": timed.get("seed_prefill_seconds"),
    }
    stream = emitted_stream(leg_dir)
    if stream:
        log.update(stream)
        config.update(stream)

    widths = verify_width_histogram(leg_dir)
    if widths:
        log["realised_mean_verify_width"] = widths["mean_verify_width"]
        log["realised_rounds"] = widths["rounds"]
        for width, fraction in widths["fraction"].items():
            log["realised_verify_width_fraction_%s" % width] = fraction
        for width, count in widths["histogram"].items():
            log["realised_verify_width_count_%s" % width] = count
        config["realised_verify_width_histogram"] = widths["histogram"]

    if args.dry_run:
        json.dump({"config": config,
                   "log": {k: v for k, v in log.items() if v is not None}},
                  sys.stdout, indent=2, default=str)
        print()
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=args.group,
                     job_type="e75-rung%s-leg" % args.rung,
                     name=meta.get("tag"), config=config, reinit=True)
    run.log({k: v for k, v in log.items() if v is not None})
    run.finish()
    print("e75_wandb_log: logged %s as %s (%s)"
          % (meta.get("tag"), run.id, run.url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
