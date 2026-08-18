#!/usr/bin/env python3
"""Compare per-arm quantized_matmul output digests produced by run-qmv-parity.sh.

The first file is the reference arm; every later file is reported as the set of
(shape, bits, width) cells whose digest differs from it.
"""
import json
import pathlib
import sys

NA_SET = {3: "{3}", 4: "{4}", 5: "{3,2}", 6: "{3}", 7: "{4,3}", 8: "{4}", 9: "{3}"}


def load(path):
    payload = json.loads(pathlib.Path(path).read_text())
    return {(e["shape"], e.get("bits", 4), e["m"]): e["digest"] for e in payload["entries"]}


def main(paths):
    ref_path, *rest = paths
    ref = load(ref_path)
    ref_name = pathlib.Path(ref_path).stem
    print(f"reference arm: {ref_name} ({len(ref)} cells, bits={sorted({b for _, b, _ in ref})})")

    for path in rest:
        arm = load(path)
        name = pathlib.Path(path).stem
        shared = sorted(set(ref) & set(arm))
        differing = [k for k in shared if ref[k] != arm[k]]
        print(f"\n=== {name} vs {ref_name} ===")
        print(f"cells compared : {len(shared)}")
        for b in sorted({b for _, b, _ in shared}):
            compared = sum(1 for _, bb, _ in shared if bb == b)
            diff = sum(1 for _, bb, _ in differing if bb == b)
            print(f"  bits={b}: {compared} compared, {diff} differing")
        print(f"cells differing: {len(differing)}")
        print(f"cells differing by (bits, M): {sorted({(b, m) for _, b, m in differing})}")
        print(f"verdict: {'BIT-IDENTICAL' if not differing else 'DIVERGES'}")
        if differing:
            by_cell = {}
            for shape, b, m in differing:
                by_cell.setdefault((b, m), []).append(shape)
            print(f"{'bits':>5}  {'M':>3}  {'NA set':>8}  {'shapes differing':>16}")
            for b, m in sorted(by_cell):
                print(f"{b:>5}  {m:>3}  {NA_SET.get(m, '-'):>8}  {len(by_cell[(b, m)]):>16}")


if __name__ == "__main__":
    main(sys.argv[1:])
