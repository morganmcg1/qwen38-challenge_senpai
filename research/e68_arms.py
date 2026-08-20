#!/usr/bin/env python3
"""E68 arm definitions: the shipped table's measured curve, and the NA >= 7 closure.

Rung 1 has two deliverables and one arm each.

`shipped` measures the CURRENT table. Every entry in the advisor's marginal
cost table is modelled, not measured: `C(M) = C1(NA_0) + 0.80 * sum_{g>0}
C1(NA_g)` over the E61 single-stream ladder. The whole E68 brief rests on the
claim that the 4->5 and 5->6 steps inverted, so rung 1 measures them.

`t789` closes the dispatch table. The advisor rejected `<T,7,7>`, `<T,8,8>` and
`<T,9,9>` on an extrapolation: the ladder stops at NA=7 (147.21 ms), and the
NA=8 and NA=9 points (172.21, 199.21) exist nowhere in the repository. This arm
routes all three cases to their lone-NA form at once, so one leg measures the
three table-level contrasts the rejection needs. It is the table-level form on
purpose: the decision is `<T,7,7>` against shipped `{4,3}`, not an isolated
cell against another isolated cell.

The `iso_*` arms are census only. They exist so the register census can price
each NA in isolation and so a `t789` register reading can be attributed to one
cell rather than to the shared `[[kernel]]` allocation.

E68 changes no scored file. Every edit is applied to the readable header AND
its runtime-effective generated twin, and the leg runner unwinds both on every
exit path.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e54_arms import (  # noqa: E402
    HEADER,
    SOURCES,
    only_case,
    swap_ipg,
)
from e59_arms import routing_table  # noqa: E402

REPO = HERE.parent

# The live assert, read from the current header rather than assumed. E54's
# `NA_ASSERT` constants are stale: `t6` moved the shipped bound to 6.
NA_ASSERT_LIVE = ('static_assert(NA >= 2 && NA <= 6, '
                  '"wide multi-row QMV supports NA in [2, 6]");')
NA_ASSERT_9 = ('static_assert(NA >= 2 && NA <= 9, '
               '"wide multi-row QMV supports NA in [2, 9]");')


def relax_na_assert_9(text: str) -> str:
    """Admit NA in [7, 9] for probe arms only. Never shipped."""
    if text.count(NA_ASSERT_LIVE) != 1:
        raise SystemExit("e68_arms: live NA assert anchor not unique")
    return text.replace(NA_ASSERT_LIVE, NA_ASSERT_9)


_STEPS = {
    "relax9": lambda t, **kw: relax_na_assert_9(t),
    "only_case": lambda t, m=7, **kw: only_case(t, m),
    "swap_ipg": lambda t, m=7, ipg=7, **kw: swap_ipg(t, m, ipg),
}


def _iso(m: int) -> dict:
    return {
        "family": "census_probe",
        "doc": "only case %d in the wide tier, <T,%d,%d>: the LONE NA=%d cell"
               % (m, m, m, m),
        "cell": "<T,%d,%d>" % (m, m),
        "never_time": True,
        "steps": [("relax9", {}), ("only_case", {"m": m}),
                  ("swap_ipg", {"m": m, "ipg": m})],
    }


ARMS: dict[str, dict] = {
    "shipped": {
        "family": "control",
        "doc": "the tip, unmodified",
        "cell": None,
        "steps": [],
    },
    "t789": {
        "family": "candidate",
        "doc": "real table, cases 7/8/9 -> <T,7,7>/<T,8,8>/<T,9,9>: the "
               "one-group merge at every width the advisor rejected",
        "cell": "<T,7,7>+<T,8,8>+<T,9,9>",
        "steps": [("relax9", {}),
                  ("swap_ipg", {"m": 7, "ipg": 7}),
                  ("swap_ipg", {"m": 8, "ipg": 8}),
                  ("swap_ipg", {"m": 9, "ipg": 9})],
    },
    "iso_m7_ipg7": _iso(7),
    "iso_m8_ipg8": _iso(8),
    "iso_m9_ipg9": _iso(9),
}


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e68_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="apply one E68 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    ap.add_argument("--dry-run", action="store_true",
                    help="apply in memory only and report the digests")
    args = ap.parse_args()

    digests = {}
    routing = None
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        if path == HEADER:
            routing = routing_table(text)
        if not args.dry_run:
            path.write_text(text)
        digests[str(path.relative_to(REPO))] = hashlib.sha256(
            text.encode()).hexdigest()
    spec = ARMS[args.arm]
    payload = {"arm": args.arm, "doc": spec["doc"], "family": spec["family"],
               "cell": spec["cell"], "dry_run": bool(args.dry_run),
               "routing": {str(k): v for k, v in sorted(routing.items())},
               "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
