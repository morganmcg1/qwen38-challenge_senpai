#!/usr/bin/env python3
"""E110 rung-2 exactness report over two traced in-situ legs.

Digests the ordered ``mtp-row:`` evidence of a base leg and an ``xv4`` leg,
compares them with each other and with the digests the merged base recorded,
and reads contract closure out of each leg's ``score.json``.

The negative control perturbs one character of one base row line and
re-digests. It must produce a DIFFERENT digest, otherwise the comparison above
is vacuous.

    usage: research/e110_rung2_exact_report.py --base-tag TAG --cand-tag TAG
                                               --tokens N --output PATH
"""

import argparse
import hashlib
import json
import pathlib
import sys

# research/e101-results.md, recorded by the merged base this experiment starts
# from. Keyed by the row count the window produces, because the digest is over
# the whole ordered stream.
PINNED_DIGESTS = {
    64: "c556822abdd850b6fefadd0ebb26dce0750c55eb0362235b6054752bb7afeb3a",
    1025: "719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e",
}

OUT_ROOT = pathlib.Path("research/out")


def row_lines(tag):
    path = OUT_ROOT / tag / "trace.txt"
    return [line.strip() for line in path.read_text().splitlines()
            if line.startswith("mtp-row:")]


def digest(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def score(tag):
    path = OUT_ROOT / tag / "score.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def meta(tag):
    path = OUT_ROOT / tag / "meta.txt"
    fields = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--cand-tag", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    base_rows = row_lines(args.base_tag)
    cand_rows = row_lines(args.cand_tag)
    base_digest = digest(base_rows)
    cand_digest = digest(cand_rows)

    control_rows = list(base_rows)
    if control_rows:
        # A one-character perturbation of the last hex value: the smallest
        # edit the digest must still catch.
        control_rows[len(control_rows) // 2] += "0"
    control_digest = digest(control_rows)

    first_mismatch = next(
        (i for i, (a, b) in enumerate(zip(base_rows, cand_rows)) if a != b), -1)

    checks = []

    checks.append({
        "check": "arm_relative_row_evidence",
        "rows": len(base_rows),
        "expected_digest": base_digest,
        "observed_digest": cand_digest,
        "first_mismatch_row": first_mismatch,
        "passed": bool(base_rows)
                  and len(base_rows) == len(cand_rows)
                  and base_digest == cand_digest,
    })

    pinned = PINNED_DIGESTS.get(len(base_rows))
    checks.append({
        "check": "campaign_pinned_row_evidence",
        "rows": len(base_rows),
        "expected_digest": pinned,
        "observed_digest": cand_digest,
        "passed": pinned is not None and pinned == cand_digest,
    })

    checks.append({
        "check": "negative_control_digest_discriminates",
        "rows": len(control_rows),
        "expected_digest": f"anything but {base_digest}",
        "observed_digest": control_digest,
        "passed": bool(base_rows) and control_digest != base_digest,
    })

    for tag, arm in ((args.base_tag, "base"), (args.cand_tag, "xv4")):
        s = score(tag).get("metrics", {})
        checks.append({
            "check": f"contract_closure_{arm}",
            "rows": len(row_lines(tag)),
            "expected_digest": "all_tokens_matched=True residual_divergence_count=0",
            "observed_digest": (
                f"all_tokens_matched={s.get('all_tokens_matched')} "
                f"residual_divergence_count={s.get('residual_divergence_count')} "
                f"decode_tokens={s.get('decode_tokens')}"),
            "passed": bool(s.get("all_tokens_matched"))
                      and s.get("residual_divergence_count") == 0
                      and s.get("decode_tokens") == args.tokens,
        })

    base_meta, cand_meta = meta(args.base_tag), meta(args.cand_tag)
    report = {
        "experiment": "e110-rung2-exact",
        "token_window": args.tokens,
        "harness": "local",
        "base_tag": args.base_tag,
        "cand_tag": args.cand_tag,
        "base_commit": base_meta.get("measured_commit_unwound"),
        "candidate_commit": cand_meta.get("branch_commit"),
        "base_worker_sha256": base_meta.get("worker_sha256_pre"),
        "candidate_worker_sha256": cand_meta.get("worker_sha256_pre"),
        "host": cand_meta.get("host"),
        "chip": cand_meta.get("chip"),
        "head_dir": cand_meta.get("head_dir"),
        "cool_gate_passed_real_gate": cand_meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": cand_meta.get("gate_qualified_for_timing"),
        "official_or_ranked_score": False,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'} {c['check']}: "
              f"rows={c['rows']} observed={c['observed_digest'][:24]}")
    print(f"e110_rung2_exact_report: {'PASS' if report['passed'] else 'FAIL'}"
          f" -> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
