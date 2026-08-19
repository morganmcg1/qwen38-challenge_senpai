#!/usr/bin/env python3
"""Read the pinned-verify-width cost curve out of the sweep legs.

Each leg pins the parent's offered draft ceiling, and the unmodified base walk
wants depth 8 at this fixture's acceptance, so every round in a leg runs at one
verify width. The palindrome order gives two legs per width at positions
symmetric about the session midpoint, so linear drift cancels per point.

The staircase hypothesis predicts a kink in the marginal between width 4 and
width 5, where the second weight stream is charged. A smooth marginal refutes
it.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ["w3a", "w4a", "w5a", "w6a", "w6b", "w5b", "w4b", "w3b"]
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def read_meta(path: pathlib.Path) -> dict:
    return {k.strip(): v.strip() for k, v in
            (line.split("=", 1) for line in
             path.read_text(encoding="utf-8", errors="replace").splitlines()
             if "=" in line)}


def decode_leg(path: pathlib.Path) -> list[tuple[int, int]]:
    legs, current, last = [], [], -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROUND_RE.search(line)
        if not match:
            continue
        index, depth, accepted = (int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)))
        if index <= last and current:
            legs.append(current)
            current = []
        last = index
        current.append((depth, accepted))
    if current:
        legs.append(current)
    drafting = [leg for leg in legs if any(d > 0 for d, _ in leg)]
    return max(drafting or legs, key=len)


def main() -> None:
    rows = []
    for tag in TAGS:
        out = ROOT / "research" / "out" / tag
        metrics = json.loads((out / "score.json").read_text())["metrics"]
        meta = read_meta(out / "meta.txt")
        leg = decode_leg(out / "trace.txt")
        widths = Counter(depth + 1 for depth, _ in leg)
        rows.append({
            "tag": tag,
            "offer": int(meta["offered_depth"]),
            "mtp": metrics["mtp_seconds_per_token"],
            "serial": metrics["serial_seconds_per_token"],
            "draft": metrics["effective_mean_draft_len"],
            "accept": metrics["accepted_draft_rate"],
            "matched": metrics["all_tokens_matched"],
            "rounds": len(leg),
            "mean_width": sum(d + 1 for d, _ in leg) / len(leg),
            "modal_width": widths.most_common(1)[0],
            "entry_c": float(meta["entry_gpu_temp_c"]),
            "gate": meta["cool_gate_passed_real_gate"],
        })

    print("tag  W   mtp s/tok     serial s/tok  draft  accept  rounds "
          "meanW  modal(W,n)  matched  gate   entryC")
    for r in rows:
        print(f"{r['tag']:<4} {r['offer'] + 1}  {r['mtp']:.8f}  "
              f"{r['serial']:.8f}  {r['draft']:.3f}  {r['accept']:.4f}  "
              f"{r['rounds']:>4}  {r['mean_width']:.3f}  "
              f"{str(r['modal_width']):<11} {r['matched']}     "
              f"{r['gate']}  {r['entry_c']:.1f}")

    by_width: dict[int, list[dict]] = {}
    for r in rows:
        by_width.setdefault(r["offer"] + 1, []).append(r)

    print("\nPinned verify width -> candidate seconds per token")
    print("(mean of the palindrome pair; spread is the pair's own disagreement)")
    print("  W   mean s/token   pair spread   marginal vs W-1      "
          "per-emitted-token change")
    previous = None
    curve = {}
    for width in sorted(by_width):
        values = [r["mtp"] for r in by_width[width]]
        mean = sum(values) / len(values)
        spread = abs(values[0] - values[1]) / mean * 100.0
        curve[width] = mean
        line = f"  {width}   {mean:.8f}   {spread:>6.4f} %"
        if previous:
            delta = mean - previous[1]
            line += (f"    {delta:+.8f}"
                     f"   {delta / previous[1] * 100:+7.3f} %")
        print(line)
        previous = (width, mean)

    print("\nSecond difference of the cost curve. The staircase predicts one")
    print("large positive value at the 4->5 step and small values elsewhere.")
    widths = sorted(curve)
    for i in range(1, len(widths) - 1):
        low, mid, high = widths[i - 1], widths[i], widths[i + 1]
        second = curve[high] - 2 * curve[mid] + curve[low]
        print(f"  d2 at W={mid}: {second:+.8f}"
              f"  ({second / curve[mid] * 100:+.4f} % of the level)")

    print("\nSerial leg across the sweep (all legs ran the same base binary,")
    print("so this is the harness's own drift, not a treatment):")
    serials = [r["serial"] for r in rows]
    mean_serial = sum(serials) / len(serials)
    print(f"  min {min(serials):.8f}  max {max(serials):.8f}"
          f"  spread {(max(serials) - min(serials)) / mean_serial * 100:.4f} %")

    matched = {r["matched"] for r in rows}
    print(f"\nall_tokens_matched across the sweep: {matched}")
    gates = {r["gate"] for r in rows}
    print(f"cool_gate_passed_real_gate across the sweep: {gates}")

    (ROOT / "research" / "e56-width-sweep.json").write_text(
        json.dumps({"legs": rows, "curve": curve}, indent=1, sort_keys=True))
    print("\nwritten: research/e56-width-sweep.json")


if __name__ == "__main__":
    main()
