"""Cross the chunk lever with every row block, from the parity artifacts.

The chunk arms allocate cleanly and still return wrong answers on device.
This prints the arm x width table used to locate the discriminator.
"""

import collections
import glob
import json

ARM_ORDER = ["mc4", "mc3", "mc2", "rps2mc4", "rps2mc2", "rps1mc4", "rps1mc2"]
WIDTHS = (3, 4, 5, 6)


def load():
    res = collections.defaultdict(dict)
    for path in sorted(glob.glob("research/e76-artifacts/parity-na*.json")):
        doc = json.load(open(path))
        na = doc["na"]
        for shape in doc["shapes"]:
            elements = na * shape["n"]
            for arm, differing in shape["parity_differing_vs_plain"].items():
                frac = differing / elements if elements else 0.0
                cell = res[arm].setdefault(na, {"max": 0.0, "failed": 0, "total": 0})
                cell["max"] = max(cell["max"], frac)
                cell["total"] += 1
                if differing:
                    cell["failed"] += 1
    return res


def main():
    res = load()
    head = f"{'arm':<10}" + "".join(f"{'NA=' + str(n):>24}" for n in WIDTHS)
    print(head)
    for arm in ARM_ORDER:
        line = f"{arm:<10}"
        for n in WIDTHS:
            cell = res.get(arm, {}).get(n)
            if cell is None:
                line += f"{'-':>24}"
            elif cell["failed"] == 0:
                line += f"{'PASS 0/' + str(cell['total']):>24}"
            else:
                txt = f"FAIL {cell['max'] * 100:.2f}% {cell['failed']}/{cell['total']}"
                line += f"{txt:>24}"
        print(line)

    failing = sorted(
        (a, n) for a in res for n in res[a] if res[a][n]["failed"] > 0
    )
    print()
    print(f"arms parsed: {len(res)}  widths: {sorted({n for a in res for n in res[a]})}")
    print(f"failing arm-width pairs ({len(failing)}): {failing}")


if __name__ == "__main__":
    main()
