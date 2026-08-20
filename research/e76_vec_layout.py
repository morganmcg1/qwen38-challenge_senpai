#!/usr/bin/env python3
"""Read the storage layout of `metal::vec<float, N>` for N = 2..6.

The g17s register count of the one-group `_wide` cell steps 83, 90, 91, 98, 111
across NA = 2..6, so +7, +1, +7, +13 for a live-float demand that rises by
exactly 4 at every step. A per-float allocator cannot produce that. The first
suspect is the vector type: the body holds thirteen `vec<float, NA>` values at
the peak, so if the type is padded to a power-of-two lane count then the real
storage demand steps only where the padded width steps.

`sizeof` and `alignof` are front-end facts, so no GPU and no backend is needed.
The undefined-template diagnostic prints the value, which avoids guessing a
constant for a `static_assert`.

  python3 research/e76_vec_layout.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import pathlib

WIDTHS = range(2, 7)

SOURCE = (
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "template <int N> struct Probe;\nvoid probe() {\n"
    + "".join(f"  Probe<sizeof(vec<float,{n}>)> s{n};\n"
              f"  Probe<alignof(vec<float,{n}>)> a{n};\n" for n in WIDTHS)
    + "}\n"
)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "vec_layout.metal"
        src.write_text(SOURCE)
        proc = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(src), "-o", "/dev/null"],
            capture_output=True, text=True)

    # The value is on the diagnostic line and the probe name is on the source
    # line under it, so the two must be read as one pair.
    pairs = re.findall(
        r"undefined template 'Probe<(\d+)>'\n.*?\b(sizeof|alignof)"
        r"\(vec<float,([2-6])>\)",
        proc.stderr)
    found = {("s" if kind == "sizeof" else "a", int(n)): int(value)
             for value, kind, n in pairs}

    missing = [(k, n) for k in "sa" for n in WIDTHS if (k, n) not in found]
    if missing:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"could not read the layout of {missing}")

    print(f"{'N':>3}{'sizeof':>8}{'alignof':>9}{'float slots':>13}{'padding':>9}")
    for n in WIDTHS:
        slots = found[("s", n)] // 4
        print(f"{n:>3}{found[('s', n)]:>8}{found[('a', n)]:>9}{slots:>13}"
              f"{slots - n:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
