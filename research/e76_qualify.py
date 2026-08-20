"""E76: list every arm that clears the 91-register g17s bar with parity proven.

An arm qualifies only when it allocates at or below the bar, spills nothing and
returns zero differing elements on every priced shape at that width.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e76_report  # noqa: E402  (needs the research directory on the path)

BAR = 91
ARTIFACTS = pathlib.Path("research/e76-artifacts")


def per_round(shapes: dict[str, float]) -> float | None:
    if not shapes:
        return None
    return sum(e76_report.DISPATCHES_PER_ROUND[shape] * seconds
               for shape, seconds in shapes.items())


def main():
    rows = json.loads((ARTIFACTS / "rung1-table.json").read_text())
    checked = e76_report.parity()
    timed = e76_report.timings()
    for na in (5, 6):
        base = per_round(timed.get((na, "plain"), {}))
        print(f"\n=== NA={na}: arms at or below {BAR} g17s registers ===")
        print(f"{'arm':<16}{'g17s':>6}{'spl':>5}{'g16s':>6}{'spl':>5}"
              f"{'parity':>20}{'s/round':>11}{'vs plain':>10}")
        sel = sorted((r for r in rows if r["na"] == na),
                     key=lambda r: r["g17s_registers"])
        for r in sel:
            if r["g17s_registers"] > BAR:
                continue
            check = checked.get((na, r["arm"]))
            if r["arm"] == "plain":
                par = "reference"
            elif check is None:
                par = "NOT CHECKED"
            elif check["differing"] == 0:
                par = f"clean {len(check['shapes'])} shapes"
            else:
                par = f"DIFFERS {check['differing']}"
            sec = per_round(timed.get((na, r["arm"]), {}))
            pct = (sec / base - 1.0) * 100 if (sec and base) else None
            print(f"{r['arm']:<16}{r['g17s_registers']:>6}"
                  f"{r['g17s_spill_bytes']:>5}{r['g16s_registers']:>6}"
                  f"{r['g16s_spill_bytes']:>5}{par:>20}"
                  f"{(f'{sec:.6f}' if sec else '-'):>11}"
                  f"{(f'{pct:+.2f}%' if pct is not None else '-'):>10}")


if __name__ == "__main__":
    main()
