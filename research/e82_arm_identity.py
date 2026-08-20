#!/usr/bin/env python3
"""Compare the raw draft streams of two or more screened heads.

The acceptance report in `e82_accept.py` aggregates. Aggregates can agree by
coincidence, so this script asks the sharper question: did two heads propose
the *same token* at the *same position* on every round of every seed?

That matters for one specific pair. `master-bf16` is the pinned trunk plus the
declared head's two-bit `draft_lm_head`. `pinned` ships no `draft_lm_head`, so
the runtime derives a four-bit compact readout instead. If the two arms agree
row for row, the two-bit shortlist plus four-bit rerank reproduces the derived
four-bit argmax exactly, and the declared readout is a pure traffic saving
rather than a quality trade.

Usage:
  python3 research/e82_arm_identity.py --reference pinned
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e82/screen"))


def draft_stream(payload: dict) -> dict[tuple[int, int], dict]:
    """Map every drafted row to `(emitted_base, depth)`.

    Keying by `round` would be wrong. Each arm commits a different number of
    tokens per round, so round 5 of one arm sits at a different place in the
    text than round 5 of another. `emitted_base` is the position the round
    starts from, which is the same question for every arm.

    Every row is kept, including rows the harness checked by
    `verify_block_replay`. This function compares what the head proposed, not
    whether the proposal was right, so a row on a rejected tail is still
    evidence about the head.
    """
    by_round: dict[int, list] = {}
    for row in payload["row_ledger"]:
        if row["kind"] == "draft":
            by_round.setdefault(row["round"], []).append(row)

    stream, base = {}, 0
    for index in sorted(by_round):
        rows = by_round[index]
        for row in rows:
            stream[(base, row["draft_index"] + 1)] = {
                "token": row["token"],
                "accepted": bool(row["accepted"]),
                "checked_by": row["reference_checked_by"],
            }
        base += sum(bool(r["accepted"]) for r in rows) + 1
    return stream


def emitted_stream(payload: dict) -> list[int]:
    """Rebuild the committed token sequence in emission order."""
    out = []
    for row in sorted(payload["row_ledger"],
                      key=lambda r: (r["round"], r["row_index"])):
        if row["kind"] == "targetTail" or row["accepted"]:
            out.append(row["token"])
    return out


def compare(ref: dict, cand: dict) -> dict:
    ref_rows, cand_rows = draft_stream(ref), draft_stream(cand)
    shared = ref_rows.keys() & cand_rows.keys()
    differing = [k for k in shared if ref_rows[k]["token"] != cand_rows[k]["token"]]
    # A row is only a fair comparison when both arms reached it along the
    # golden trajectory. Once either arm has been rejected inside the round,
    # its deeper rows continue its own wrong tokens, so the two heads are
    # answering different questions and a token difference proves nothing.
    same_context = [k for k in shared
                    if ref_rows[k]["checked_by"] == "serial_golden"
                    and cand_rows[k]["checked_by"] == "serial_golden"]
    context_differing = [k for k in same_context
                         if ref_rows[k]["token"] != cand_rows[k]["token"]]
    ref_emit, cand_emit = emitted_stream(ref), emitted_stream(cand)
    return {
        "reference_rows": len(ref_rows),
        "candidate_rows": len(cand_rows),
        "shared_rows": len(shared),
        "reference_only_rows": len(ref_rows.keys() - cand_rows.keys()),
        "candidate_only_rows": len(cand_rows.keys() - ref_rows.keys()),
        "differing_tokens_on_shared_rows": len(differing),
        "same_context_rows": len(same_context),
        "differing_tokens_on_same_context_rows": len(context_differing),
        "identical_draft_stream":
            len(shared) == len(ref_rows) == len(cand_rows) and not differing,
        "emitted_prefix_match": ref_emit[:len(cand_emit)] == cand_emit[:len(ref_emit)],
        "reference_emitted": len(ref_emit),
        "candidate_emitted": len(cand_emit),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(CACHE / "verify"))
    ap.add_argument("--reference", default="pinned")
    ap.add_argument("--out", default="research/e82-arm-identity.json")
    args = ap.parse_args()

    root = Path(args.root)
    ref_dir = root / args.reference
    arms = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name != args.reference)
    seeds = sorted(p.stem for p in ref_dir.glob("*.json"))

    report = {"reference": args.reference, "seeds": seeds, "arms": {}}
    print(f"reference arm: {args.reference}   seeds: {len(seeds)}\n")
    print("arm".ljust(13) + "seeds  identical   golden rows  differing  "
          "unpaired  emitted match")
    for arm in arms:
        per_seed, totals = {}, {
            "shared_rows": 0, "differing_tokens_on_shared_rows": 0,
            "same_context_rows": 0, "differing_tokens_on_same_context_rows": 0,
            "reference_only_rows": 0, "candidate_only_rows": 0,
        }
        identical, emitted_ok = 0, 0
        for seed in seeds:
            cand_path = root / arm / f"{seed}.json"
            if not cand_path.exists():
                continue
            result = compare(
                json.loads((ref_dir / f"{seed}.json").read_text()),
                json.loads(cand_path.read_text()))
            per_seed[seed] = result
            for key in totals:
                totals[key] += result[key]
            identical += result["identical_draft_stream"]
            emitted_ok += result["emitted_prefix_match"]
        report["arms"][arm] = {"totals": totals, "seeds_identical": identical,
                               "seeds_compared": len(per_seed),
                               "seeds_emitted_match": emitted_ok,
                               "per_seed": per_seed}
        print(f"{arm.ljust(13)}{len(per_seed):5d}"
              f"{identical:11d}{totals['same_context_rows']:14,}"
              f"{totals['differing_tokens_on_same_context_rows']:11,}"
              f"{totals['reference_only_rows'] + totals['candidate_only_rows']:10,}"
              f"{emitted_ok:15d}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
