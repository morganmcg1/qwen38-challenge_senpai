#!/usr/bin/env python3
"""E163: diff the width-plan arms of `E163IPGPlanExactnessTests`.

Each arm writes one JSON of per-cell output digests. This reads two or more
arm files and decides three separate questions:

  1. Did each arm match MLX's own `quantized_matmul` bit for bit?
  2. Do the arms produce identical bits at every scored shape and width,
     including the widths no arm touches?
  3. Did every positive control reject, in every arm?

A run where two files carry the same plan is refused: two identical arms would
agree trivially and the comparison would prove nothing.

  usage: research/e163_exactness_compare.py ARM.json ARM.json [ARM.json ...]
"""
import json
import sys


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    payload["_path"] = path
    return payload


def plan_signature(payload):
    return tuple((entry["m"], entry["inputs_per_group"]) for entry in payload["plan"])


def plan_text(payload):
    return " ".join(f"{e['m']}:{e['inputs_per_group']}" for e in payload["plan"])


def key(cell):
    return (cell["shape"], cell["m"])


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip())
        return 2
    arms = [load(path) for path in argv[1:]]

    failures = []
    signatures = {}
    for arm in arms:
        signature = plan_signature(arm)
        if signature in signatures:
            print(f"FAIL: {arm['_path']} carries the same plan as "
                  f"{signatures[signature]}, so this is one arm twice")
            return 1
        signatures[signature] = arm["_path"]

    for arm in arms:
        for cell in arm["cells"]:
            if not cell["matches_incumbent_bitwise"]:
                failures.append(
                    f"{arm['arm']} {cell['shape']} M={cell['m']}: "
                    f"{cell['mismatched_elements']} outputs differ from the "
                    f"incumbent, max_abs_delta {cell['max_abs_delta_vs_incumbent']}")
            control = cell.get("positive_control")
            if control and not control["rejects"]:
                failures.append(
                    f"{arm['arm']} {cell['shape']} M={cell['m']}: the positive "
                    "control passed, so the comparison cannot fail")
            if control and control["changed_rows"] != [cell["m"] - 1]:
                failures.append(
                    f"{arm['arm']} {cell['shape']} M={cell['m']}: one ulp on row "
                    f"{cell['m'] - 1} moved rows {control['changed_rows']}")

    indexed = [{key(cell): cell for cell in arm["cells"]} for arm in arms]
    covered = set(indexed[0])
    for cells in indexed[1:]:
        if set(cells) != covered:
            failures.append("the arms cover different cells")
            covered &= set(cells)

    names = [arm["arm"] for arm in arms]
    print(f"{'shape':<34} {'M':>2}  " + "  ".join(f"{n:>10}" for n in names)
          + "   bits equal")
    print("-" * (40 + 12 * len(names) + 13))
    for cell_key in sorted(covered):
        cells = [index[cell_key] for index in indexed]
        digests = {cell["candidate_digest"] for cell in cells}
        equal = len(digests) == 1
        if not equal:
            failures.append(
                f"{cell_key[0]} M={cell_key[1]}: the arms disagree bitwise")
        groups = "  ".join(
            f"{cell['active_groups']}x{cell['accumulator_lanes_na']:<8}"
            for cell in cells)
        print(f"{cell_key[0]:<34} {cell_key[1]:>2}  {groups}   {equal}")

    print()
    for arm in arms:
        print(f"{arm['arm']:<10} plan {plan_text(arm)}")
    controls = sum(1 for arm in arms for cell in arm["cells"]
                   if "positive_control" in cell)
    rejecting = sum(1 for arm in arms for cell in arm["cells"]
                    if cell.get("positive_control", {}).get("rejects"))
    print(f"positive controls: {rejecting} of {controls} rejecting, at widths "
          f"{sorted(set(arms[0]['touched_widths']))}")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("\nE163 EXACTNESS OK: every arm matches the incumbent and every other arm")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
