#!/usr/bin/env python3
"""E75 arm definitions: our dispatch table against the crown's.

Two arms, one contrast.

`ours` is the tip unmodified: `NA <= 6`, and `<T,5,5>`, `<T,6,6>`, `<T,9,5>`.
`crown` is Layr-Labs upstream main at bfab0de, which is this campaign's own
base before three unpromoted Senpai commits: b757237 (M=5), aa8ce50 (M=6) and
2267a84 (M=9). It sets `NA <= 4` and dispatches `<T,5,3>`, `<T,6,3>`,
`<T,9,3>`.

The frozen host launches M x-groups per verify width. The group at `tid.x`
claims rows [tid.x*IPG, tid.x*IPG + IPG) and any group past the end returns
without touching weights, so a width-M round streams the whole weight tile
ceil(M / IPG) times. Our three cells each buy one fewer weight stream and pay
the NA >= 5 register cliff for it. The arm-2 receipt says that trade loses at
rank by 0.298 % on the scoring prompts.

Why this module does not reuse `e54_arms.SHIPPED_IPG`
-----------------------------------------------------
That constant says M=5 dispatches IPG 3. The live source says 5. It went stale
when b757237 landed, and `swap_ipg` builds its search anchor from it, so every
M=5 arm derived from it would fail at the substitution step. This module reads
the dispatch map out of the header instead, so it cannot go stale in the same
way.

Why the digest assertion is the real check
------------------------------------------
Reproducing eight lines by substitution is not the same as reproducing the
crown's bytes. `crown` therefore asserts that both patched files hash to the
exact upstream sha256 before any leg is allowed to build. Anything less would
let a whitespace or ordering difference reach a timed measurement while still
looking like the crown table in a diff.

E75 changes no scored file. Every edit is applied to the readable header AND
its runtime-effective generated twin, and the leg runner unwinds both on every
exit path.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e54_arms import HEADER, SOURCES, TWIN  # noqa: E402
from e59_arms import routing_table  # noqa: E402

REPO = HERE.parent

# Layr-Labs/qwen-3.8-mtp-challenge main at bfab0de, which the advisor confirmed
# is the crown submission 9ad17378's exact promotedSourceRef.
CROWN_SHA256 = {
    HEADER: "75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e010817b1e5249f281",
    TWIN: "350de46828265271e504c93d009a3b3e8b05c83047666be7fc0de51ded29b6bb",
}
CROWN_DISPATCH = {5: 3, 6: 3, 9: 3}
CROWN_NA_MAX = 4

NA_ASSERT = ('static_assert(NA >= 2 && NA <= %d, '
             '"wide multi-row QMV supports NA in [2, %d]");')
DISPATCH = re.compile(
    r"qmv_fast_crossrow_affine4_g64_m<T, (?P<m>\d+), (?P<ipg>\d+), true>")


def live_dispatch(text: str) -> dict[int, int]:
    """The dispatch map as the source actually holds it, not as E54 recorded it."""
    out: dict[int, int] = {}
    for hit in DISPATCH.finditer(text):
        m, ipg = int(hit.group("m")), int(hit.group("ipg"))
        if out.setdefault(m, ipg) != ipg:
            raise SystemExit(
                "e75_arms: M=%d dispatches both IPG %d and %d; the anchor is "
                "not unique and a swap would be ambiguous" % (m, out[m], ipg))
    if not out:
        raise SystemExit("e75_arms: found no crossrow dispatch at all")
    return out


def live_na_max(text: str) -> int:
    for bound in range(2, 10):
        if text.count(NA_ASSERT % (bound, bound)) == 1:
            return bound
    raise SystemExit("e75_arms: cannot read the live NA assert bound")


def set_na_max(text: str, bound: int) -> str:
    live = live_na_max(text)
    if live == bound:
        return text
    return text.replace(NA_ASSERT % (live, live), NA_ASSERT % (bound, bound))


def swap_ipg(text: str, m: int, ipg: int) -> str:
    """Repoint `case m:` at `<T, m, ipg>`, reading the old IPG from the source."""
    live = live_dispatch(text)
    if m not in live:
        raise SystemExit("e75_arms: M=%d is not in the live dispatch map" % m)
    if live[m] == ipg:
        return text
    old = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, live[m])
    new = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, ipg)
    if text.count(old) != 1:
        raise SystemExit("e75_arms: M=%d dispatch anchor not unique" % m)
    return text.replace(old, new)


def apply_crown(text: str) -> str:
    for m, ipg in sorted(CROWN_DISPATCH.items()):
        text = swap_ipg(text, m, ipg)
    return set_na_max(text, CROWN_NA_MAX)


ARMS: dict[str, dict] = {
    "ours": {
        "family": "control",
        "doc": "the tip, unmodified: NA <= 6, <T,5,5> <T,6,6> <T,9,5>",
        "cell": "<T,5,5>+<T,6,6>+<T,9,5>",
        "expect_sha256": None,
        "steps": [],
    },
    "crown": {
        "family": "candidate",
        "doc": "Layr-Labs upstream main bfab0de, the crown's literal bytes: "
               "NA <= 4, <T,5,3> <T,6,3> <T,9,3>",
        "cell": "<T,5,3>+<T,6,3>+<T,9,3>",
        "expect_sha256": CROWN_SHA256,
        "steps": [("crown", {})],
    },
}

_STEPS = {"crown": lambda t, **kw: apply_crown(t)}


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e75_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="apply one E75 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    ap.add_argument("--dry-run", action="store_true",
                    help="apply in memory only and report the digests")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    digests, routing = {}, None
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        digest = hashlib.sha256(text.encode()).hexdigest()
        expect = (spec["expect_sha256"] or {}).get(path)
        if expect and digest != expect:
            raise SystemExit(
                "e75_arms: arm %s produced %s for %s but the crown's bytes "
                "hash to %s; refusing to time a table that is not the crown's"
                % (args.arm, digest, path.relative_to(REPO), expect))
        if path == HEADER:
            routing = routing_table(text)
            dispatch = live_dispatch(text)
            na_max = live_na_max(text)
        if not args.dry_run:
            path.write_text(text)
        digests[str(path.relative_to(REPO))] = digest

    payload = {"arm": args.arm, "doc": spec["doc"], "family": spec["family"],
               "cell": spec["cell"], "dry_run": bool(args.dry_run),
               "dispatch": {str(k): v for k, v in sorted(dispatch.items())},
               "na_max": na_max,
               "crown_bytes_verified": bool(spec["expect_sha256"]),
               "routing": {str(k): v for k, v in sorted(routing.items())},
               "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
