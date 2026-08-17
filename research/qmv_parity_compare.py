#!/usr/bin/env python3
"""Compare per-arm quantized_matmul output digests produced by run-qmv-parity.sh.

The first file is the reference arm; every later file is reported as the set of
(shape, width) cells whose digest differs from it.
"""
import json
import pathlib
import sys

NA_SET = {3: "{3}", 4: "{4}", 5: "{3,2}", 6: "{3}", 7: "{4,3}", 8: "{4}", 9: "{3}"}


def load(path):
    payload = json.loads(pathlib.Path(path).read_text())
    return {(e["shape"], e["m"]): e["digest"] for e in payload["entries"]}


def main(paths):
    ref_path, *rest = paths
    ref = load(ref_path)
    ref_name = pathlib.Path(ref_path).stem
    print(f"reference arm: {ref_name} ({len(ref)} cells)")

    for path in rest:
        arm = load(path)
        name = pathlib.Path(path).stem
        shared = sorted(set(ref) & set(arm))
        differing = [k for k in shared if ref[k] != arm[k]]
        widths = sorted({m for _, m in differing})
        print(f"\n=== {name} vs {ref_name} ===")
        print(f"cells compared : {len(shared)}")
        print(f"cells differing: {len(differing)}")
        print(f"widths differing: {widths}")
        print(f"verdict: {'BIT-IDENTICAL' if not differing else 'DIVERGES'}")
        if differing:
            by_width = {}
            for shape, m in differing:
                by_width.setdefault(m, []).append(shape)
            print(f"{'M':>3}  {'NA set':>8}  {'shapes differing':>16}")
            for m in widths:
                print(f"{m:>3}  {NA_SET.get(m, '-'):>8}  {len(by_width[m]):>16}")


if __name__ == "__main__":
    main(sys.argv[1:])
