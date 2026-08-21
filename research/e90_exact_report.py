#!/usr/bin/env python3
"""E90 rung 0a verdict: composed-tree exactness over one untimed gate pass.

Reads the reference rows, the `mtp-verify --golden` report and the positive
control, and prints one verdict plus the fields the assignment requires. Exit
status is 0 only when every required condition holds.
"""

import argparse
import json
import sys

STOP_TOKENS = {248044}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--gate-exit", type=int, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-exit", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--meta")
    parser.add_argument("--output")
    args = parser.parse_args()

    golden = load(args.golden)
    emitted = golden["emitted_tokens"]
    eos_indices = [i for i, t in enumerate(emitted[: args.tokens]) if t in STOP_TOKENS]

    try:
        report = load(args.ledger)
    except Exception as exc:  # noqa: BLE001
        print("e90 rung 0a: FAIL, unreadable gate report (%s)" % exc)
        return 1

    ledger = report.get("row_ledger", [])
    checks = {}
    checks["gate_exit_zero"] = args.gate_exit == 0
    checks["all_tokens_matched"] = report.get("all_tokens_matched") is True
    checks["parity_all_ok"] = report.get("parity_all_ok") is True
    checks["residual_zero"] = report.get("residual_divergence_count") == 0
    checks["emitted_equals_window"] = report.get("emitted_token_total") == args.tokens
    declared = report.get("declared_rows_total")
    checked = report.get("reference_checked_row_total")
    checks["ledger_equals_declared"] = len(ledger) == declared
    checks["checked_equals_declared"] = checked == declared
    checks["post_eos_continuation"] = bool(eos_indices) and eos_indices[0] < args.tokens - 1

    # The control must be rejected. A non-zero exit or an explicit mismatch both
    # count; a gate that accepts a corrupted golden is vacuous.
    control_rejected = args.control_exit != 0
    control_report = None
    if not control_rejected:
        try:
            control_report = load(args.control)
            control_rejected = control_report.get("all_tokens_matched") is False
        except Exception:  # noqa: BLE001
            control_rejected = True
    checks["positive_control_rejects"] = control_rejected

    summary = {
        "experiment": "e90",
        "rung": "0a",
        "tokens": args.tokens,
        "all_tokens_matched": report.get("all_tokens_matched"),
        "parity_all_ok": report.get("parity_all_ok"),
        "residual_divergence_count": report.get("residual_divergence_count"),
        "emitted_token_total": report.get("emitted_token_total"),
        "declared_rows_total": declared,
        "reference_checked_row_total": checked,
        "row_ledger_rows": len(ledger),
        "rejected_rows_reference_checked": report.get("rejected_rows_reference_checked"),
        "round_count": report.get("round_count"),
        "accepted_draft_total": report.get("accepted_draft_total"),
        "rejected_draft_total": report.get("rejected_draft_total"),
        "effective_mean_draft_len": report.get("effective_mean_draft_len"),
        "accepted_draft_rate": report.get("accepted_draft_rate"),
        "target_cache_offset_final": report.get("target_cache_offset_final"),
        "max_rejected_tail_logit_delta": report.get("max_rejected_tail_logit_delta"),
        "verify_block_replayed_round_count": report.get(
            "verify_block_replayed_round_count"
        ),
        "head_provenance_sha256": report.get("head_provenance_sha256"),
        "golden_rows": len(golden.get("rows", [])),
        "golden_reference_self_consistent": golden.get("reference_self_consistent"),
        "first_eos_index_in_window": eos_indices[0] if eos_indices else None,
        "tokens_after_first_eos": (
            args.tokens - eos_indices[0] - 1 if eos_indices else 0
        ),
        "gate_exit": args.gate_exit,
        "control_exit": args.control_exit,
        "control_all_tokens_matched": (
            control_report.get("all_tokens_matched") if control_report else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if summary["round_count"]:
        summary["rows_per_token"] = declared / args.tokens if declared else None

    if args.meta:
        meta = {}
        with open(args.meta) as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    meta[key] = value
        summary["meta"] = meta

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("e90 rung 0a: FAIL: " + ", ".join(failed))
        return 1
    print("e90 rung 0a: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
