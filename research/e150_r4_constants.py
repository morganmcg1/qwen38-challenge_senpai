#!/usr/bin/env python3
"""Emit the two constants the E150 R4 Swift schedule needs, at full precision.

`harness=local`. Zero GPU. This is a transcription aid, not a measurement:
it prints the ranked per-width round cost curve and the derived depth price
so the Swift table can be checked against the Python replay byte for byte.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import ranked_round_us  # noqa: E402
from e128_replay import MAX_DEPTH  # noqa: E402
from e150_lib import build_env  # noqa: E402


def main() -> int:
    env = build_env(windows=1, seeds=1)
    print("## e128_price.ranked_round_us, the REPLAYED curve, microseconds")
    for width in range(1, MAX_DEPTH + 2):
        print("    %.17g,  // verify width %d"
              % (ranked_round_us(width), width))

    per_width = env.measured["per_width"]
    print("\n## env.measured per_width, the MEASURED curve, microseconds")
    print("    extrapolated widths %s"
          % env.measured["provenance"].get("extrapolated_widths"))
    for width in sorted(per_width):
        print("    %.17g,  // verify width %d" % (per_width[width], width))

    marginal, cumulative = env.measured_price
    print("\n## env.measured_price marginal")
    for value in marginal:
        print("    %.17g," % value)
    print("\n## env.measured_price cumulative")
    for value in cumulative:
        print("    %.17g," % value)

    print("\n## derived: marginal[d] == (C(d+2) - C(d+1)) / C(1)")
    unit = per_width[1]
    worst = 0.0
    for depth in range(MAX_DEPTH):
        derived = (per_width[depth + 2] - per_width[depth + 1]) / unit
        worst = max(worst, abs(derived - marginal[depth]))
    print("    worst |derived - marginal| %.3e" % worst)
    print("    so Swift can carry the microsecond curve and derive the price")

    print("\n## env.replayed_price marginal, the bracket's other corner")
    for value in env.replayed_price[0]:
        print("    %.17g," % value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
