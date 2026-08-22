#!/usr/bin/env python3
"""E116 cleanup: witness that a census row now names its dispatch entry point.

    usage: research/e116_entry_witness.py TAG [--json OUT]

THE DEFECT. `swizzleDispatch(cls, name)` in `E58DispatchCensus.swift` installs
the same replacement block on `dispatchThreadgroups:threadsPerThreadgroup:` and
on `dispatchThreads:threadsPerThreadgroup:`. It captures the selector so it can
call the original, but it never passes the name to `DispatchLedger.dispatch`.
Both selectors carry an `MTLSize` called `grid`; one counts THREADS and the
other counts THREADGROUPS. A census row of `grid=1x4352x1 tg=32x2x1` was
therefore ambiguous by a factor of 64, and every census this campaign has run
carries the ambiguity.

THE FIX appends `entry=threads` or `entry=groups` to the shape key. It lives in
`research/e116-artifacts/instruments.patch`, not in `Sources/`, because the
whole census is research instrumentation and is deleted from the scored
surface.

This reducer reads a census leg produced with the patch applied and reports the
dispatch counts by entry point, so the fix is evidence rather than an assertion.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--json")
    args = ap.parse_args()

    src = pathlib.Path("research/out") / args.tag / "census.jsonl"
    by_entry: collections.Counter[str] = collections.Counter()
    qmv: collections.Counter[str] = collections.Counter()
    examples: dict[str, str] = {}
    untagged = 0
    for line in src.open():
        record = json.loads(line)
        for _, phase in (record.get("phases") or {}).items():
            for shape, count in (phase.get("shapes") or {}).items():
                token = shape.rsplit(" ", 1)[-1]
                if not token.startswith("entry="):
                    untagged += count
                    continue
                entry = token[len("entry="):]
                by_entry[entry] += count
                examples.setdefault(entry, shape)
                if "affine_qmv_fast" in shape:
                    qmv[entry] += count

    total = sum(by_entry.values())
    if not total:
        raise SystemExit(f"{src}: no shape carries an entry token; the leg was "
                         f"run by a worker built without the fix")

    out = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": "cleanup",
        "witness_tag": args.tag,
        "defect": "swizzleDispatch captured the selector for the original "
                  "call but never gave it to DispatchLedger.dispatch, so a "
                  "census row could not say whether grid counted threads or "
                  "threadgroups",
        "site": "Sources/MLXFastModel/E58DispatchCensus.swift, "
                "swizzleDispatch and DispatchLedger.dispatch",
        "fix_lives_in": "research/e116-artifacts/instruments.patch",
        "witness_is_not_a_timing_leg": True,
        "dispatches_by_entry": dict(by_entry),
        "dispatches_total": total,
        "dispatches_without_an_entry_token": untagged,
        "groups_share_of_all_dispatches": by_entry["groups"] / total,
        "affine_qmv_fast_dispatches_by_entry": dict(qmv),
        "every_wide_qmv_dispatch_is_threadgroups": qmv.get("threads", 0) == 0,
        "examples": examples,
        "consequence":
            "affine_qmv_fast grid=1x4352x1 tg=32x2x1 launches 4,352 "
            "THREADGROUPS of 64 threads, which is 278,528 threads, not 4,352 "
            "threads. mlp.gate_up has N=34,816 rows, so 34,816/4,352 is 8 rows "
            "per threadgroup and 4 rows per 32-lane SIMD group, which is the "
            "affine_qmv_fast layout. The threads reading is not merely wrong, "
            "it is not realisable: 4,352 threads cannot fill 68 threadgroups "
            "of 64.",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }

    print(f"E116 cleanup -- dispatch entry-point witness   harness=local")
    print(f"  leg {args.tag}, {total:,} tagged dispatches, "
          f"{untagged:,} untagged")
    for entry, count in sorted(by_entry.items()):
        print(f"    entry={entry:<8} {count:>8,}  "
              f"{count / total:6.2%}   e.g. {examples[entry]}")
    print(f"  affine_qmv_fast by entry: {dict(qmv)}")
    print(f"  every wide-QMV dispatch is threadgroups: "
          f"{out['every_wide_qmv_dispatch_is_threadgroups']}")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
