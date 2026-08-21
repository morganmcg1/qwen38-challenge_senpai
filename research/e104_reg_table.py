#!/usr/bin/env python3
"""Print the per-arm, per-width register and spill table from an E104 census."""
import argparse
import json

ARCHES = ("applegpu_g16s", "applegpu_g17s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census_json")
    args = ap.parse_args()
    doc = json.load(open(args.census_json))
    rows = doc["arms"]
    arms = [r["arm"] for r in rows]
    nas = sorted({int(k) for r in rows for a in ARCHES for k in r.get(a, {})})
    for arch in ARCHES:
        print(f"\n=== {arch}: registers / spill bytes ===")
        print(f"{'NA':>3}" + "".join(f"{a:>16}" for a in arms))
        for na in nas:
            cells = []
            for r in rows:
                st = r.get(arch, {}).get(str(na))
                cells.append(
                    f"{'-':>16}" if st is None
                    else f"{st['registers']:>11}/{st['spill_bytes']:<4}"
                )
            print(f"{na:>3}" + "".join(cells))
        print(f"\n=== {arch}: ISA text bytes (vs a_base) ===")
        print(f"{'NA':>3}" + "".join(f"{a:>16}" for a in arms))
        for na in nas:
            base = rows[0].get(arch, {}).get(str(na), {}).get("text_bytes")
            cells = []
            for r in rows:
                st = r.get(arch, {}).get(str(na))
                if st is None:
                    cells.append(f"{'-':>16}")
                else:
                    d = 100.0 * (st["text_bytes"] / base - 1.0)
                    cells.append(f"{st['text_bytes']:>9}{d:>+6.1f}%")
            print(f"{na:>3}" + "".join(cells))


if __name__ == "__main__":
    main()
