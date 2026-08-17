#!/usr/bin/env python3
"""Compare two draft-bits sweep runs arm-by-arm.

Used to prove a base change did not move the compact draft readout's kernel
path, dispatch family, or cost. Reports per-arm drift plus the 4->3 contrast
at each measured row count.

    research/compare_bits_sweeps.py OLD_TAG NEW_TAG [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT_DEFAULT = ".mlxfast-private/draft-bits-sweep"
IDENTITY_FIELDS = ("kernel", "in_kernel_path", "uses_crossrow_kernel", "weight_bytes")


def load(root: pathlib.Path, tag: str) -> dict:
    path = root / tag / "bits.json"
    if not path.is_file():
        sys.exit(f"missing sweep artifact: {path}")
    return json.loads(path.read_text())


def arm(arms: list[dict], m: int, bits: int) -> dict | None:
    for row in arms:
        if row.get("m") == m and row.get("bits") == bits:
            return row
    return None


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_tag")
    ap.add_argument("new_tag")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    old, new = load(root, args.old_tag), load(root, args.new_tag)
    old_arms, new_arms = old["arms"], new["arms"]

    keys = sorted({(r["m"], r["bits"]) for r in new_arms if "m" in r})

    print(f"# sweep drift: {args.old_tag} -> {args.new_tag}\n")
    print(f"{'arm':<16}{'old s/call':>14}{'new s/call':>14}{'drift':>9}  identity")
    identical = True
    for m, bits in keys:
        a, b = arm(old_arms, m, bits), arm(new_arms, m, bits)
        if a is None or b is None:
            print(f"{'m=%d bits=%d' % (m, bits):<16}{'--':>14}{'--':>14}{'--':>9}  ARM MISSING")
            identical = False
            continue
        same = all(a[f] == b[f] for f in IDENTITY_FIELDS)
        identical &= same
        print(
            f"{'m=%d bits=%d' % (m, bits):<16}"
            f"{a['seconds_per_call']:>14.7f}{b['seconds_per_call']:>14.7f}"
            f"{pct(b['seconds_per_call'], a['seconds_per_call']):>8.2f}%"
            f"  {'same' if same else 'CHANGED'} {b['in_kernel_path']}"
            f" crossrow={b['uses_crossrow_kernel']}"
        )

    print(f"\nkernel identity preserved across every shared arm: {identical}")

    for tag, arms in ((args.old_tag, old_arms), (args.new_tag, new_arms)):
        print(f"\n## {tag}")
        for m in sorted({k[0] for k in keys}):
            four, three = arm(arms, m, 4), arm(arms, m, 3)
            if four is None or three is None:
                continue
            delta = three["seconds_per_call"] - four["seconds_per_call"]
            print(
                f"  M={m} 4->3: {pct(three['seconds_per_call'], four['seconds_per_call']):+7.3f}%"
                f"  delta={delta * 1e6:+8.2f} us"
                f"  bytes={pct(three['weight_bytes'], four['weight_bytes']):+6.2f}%"
                f"  bw={pct(three['achieved_gb_per_second'], four['achieved_gb_per_second']):+6.2f}%"
                f"  crossrow 4bit={four['uses_crossrow_kernel']} 3bit={three['uses_crossrow_kernel']}"
            )
        # Row-count amortisation: how much a second row costs each width.
        for bits in (4, 3):
            one, two = arm(arms, 1, bits), arm(arms, 2, bits)
            if one is None or two is None:
                continue
            print(
                f"  {bits}-bit M=2/M=1 cost ratio: {two['seconds_per_call'] / one['seconds_per_call']:.3f}"
                f"  ({two['in_kernel_path']})"
            )

    dev = new["device"]
    print(
        f"\ndevice: {dev['architecture']} class={dev['architecture_class']}"
        f" gen={dev['architecture_gen']} nax_available={dev['nax_available']}"
    )
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
