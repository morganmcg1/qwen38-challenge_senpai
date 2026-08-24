#!/usr/bin/env python3
"""E199 gate reader: prove that width-9 verify rounds actually ran.

Reads the captured CLI reports of one ``--local-submit`` session and reports
the per-round draft/row histogram of the TIMED MTP leg, the RULE 179 ledger
identity, the RULE 389/390 head fields and the exactness fields.

``effective_draft_lengths`` is the trusted parent's own journal of what the
candidate proposed on each round; rows verified on a round are ``drafts + 1``.
The gate PASSES only when at least one round verified 9 rows, because a run
with no 9-row round has not exercised the risk the cap-8 edit introduces.
"""
import collections
import json
import pathlib
import sys

FAIL = 1


def load_timed_leg(reports: pathlib.Path) -> dict:
    timed = []
    for path in sorted(reports.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        if doc.get("verb") == "mtp-timed" and not doc.get("is_serial_control", True):
            timed.append((path, doc))
    if len(timed) != 1:
        raise SystemExit(
            f"e199_histogram: expected exactly 1 timed MTP leg report, found {len(timed)}"
            f" in {reports}: {[p.name for p, _ in timed]}"
        )
    return timed[0][1]


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    report = load_timed_leg(out / "reports")
    score = json.loads((out / "score.json").read_text())
    metrics = score["metrics"]

    lengths = report["effective_draft_lengths"]
    hist = collections.Counter(lengths)
    rounds9 = hist.get(8, 0)

    rounds = report["round_count"]
    accepted = report["accepted_draft_total"]
    rejected = report["rejected_draft_total"]
    declared = report["declared_rows_total"]
    checked = report["reference_checked_row_total"]
    ledger_ok = (rounds + accepted + rejected == declared == checked)

    print("=== E199 width-9 gate (harness=local) ===")
    print(f"rounds={rounds} emitted_tokens={report['emitted_token_total']}")
    print("drafts -> rows : rounds")
    for drafts in sorted(hist):
        print(f"  {drafts:>2}   -> {drafts + 1:>2}   : {hist[drafts]}")
    print(f"effective_max_draft_len={report['effective_max_draft_len']} "
          f"(max rows={report['effective_max_draft_len'] + 1})")
    print(f"effective_mean_draft_len={report['effective_mean_draft_len']:.6f}")
    print(f"non_drafting_round_count={report['non_drafting_round_count']}")
    print(f"nine_row_round_count={rounds9}")
    print()
    print("--- RULE 179 ledger ---")
    print(f"rounds+accepted+rejected = {rounds}+{accepted}+{rejected} = "
          f"{rounds + accepted + rejected}")
    print(f"declared_rows_total={declared} reference_checked_row_total={checked}")
    print(f"rejected_rows_reference_checked={report['rejected_rows_reference_checked']}")
    print(f"ledger_identity_ok={ledger_ok}")
    print()
    print("--- exactness ---")
    print(f"all_tokens_matched={report['all_tokens_matched']}")
    print(f"residual_divergence_count={report['residual_divergence_count']}")
    print(f"parity_all_ok={report['parity_all_ok']}")
    print(f"max_rejected_tail_logit_delta={report['max_rejected_tail_logit_delta']}")
    print(f"verify_block_replayed_round_count={report['verify_block_replayed_round_count']}")
    print(f"decode_tokens={metrics['decode_tokens']} mode={metrics['mode']}")
    print(f"public_drift_tripwire_passed={metrics['public_drift_tripwire_passed']}")
    print()
    print("--- RULE 389/390 head identity ---")
    print(f"head_class=declared")
    print(f"head_provenance_sha256={metrics['head_provenance_sha256']}")
    print(f"head_provenance={json.dumps(report['head_provenance'], sort_keys=True)}")
    print()
    print("--- timing record (harness=local, NOT the decision) ---")
    print(f"serial_seconds_per_token={metrics['serial_seconds_per_token']:.7f}")
    print(f"mtp_seconds_per_token={metrics['mtp_seconds_per_token']:.7f}")
    print(f"local_ratio={metrics['mtp_decode_speedup']:.6f}")
    print(f"accepted_draft_rate={metrics['accepted_draft_rate']:.6f}")
    print(f"seed_prefill_seconds={report['seed_prefill_seconds']:.6f}")
    print(f"p50_block_request_seconds={report['p50_block_request_seconds']:.6f}")
    print(f"max_block_request_seconds_after_first="
          f"{report['max_block_request_seconds_after_first']:.6f}")
    print()

    failures = []
    if not metrics["all_tokens_matched"]:
        failures.append("all_tokens_matched=false")
    if report["residual_divergence_count"] != 0:
        failures.append("residual_divergence_count != 0")
    if not ledger_ok:
        failures.append("row-ledger identity does not close")
    if metrics["decode_tokens"] != 512:
        failures.append(f"decode_tokens={metrics['decode_tokens']} != 512")
    if rounds9 == 0:
        failures.append("no 9-row round occurred: the gate did NOT exercise width 9")
    if report["effective_max_draft_len"] != 8:
        failures.append(
            f"effective_max_draft_len={report['effective_max_draft_len']} != 8:"
            " the cap-8 edit is not live in the measured worker")
    # RULE 389: candidate evidence must run the head the candidate declares.
    # dadbfb80 is the declared-run digest of the tracked mtp-head.manifest.json.
    if not metrics["head_provenance_sha256"].startswith("dadbfb80"):
        failures.append(
            f"head_provenance_sha256={metrics['head_provenance_sha256']} is not the"
            " declared-run head (dadbfb80...): HARNESS DEFECT 44 exposure")

    if failures:
        print("E199 GATE FAILED:")
        for item in failures:
            print(f"  - {item}")
        return FAIL
    print("E199 GATE PASSED: width-9 rounds ran, tokens exact, ledger closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
