#!/usr/bin/env python3
"""Negative control for `research/twin_audit.py`'s comment-only waiver machinery.

A waiver in a drift audit is dangerous by construction: it is exactly the
mechanism by which a real edit could be silently absorbed. This script is the
falsification test for that risk, and it is written so that it keeps its teeth
whether or not any waiver is currently allowlisted.

PART A -- THE LIVE TABLE.
`KNOWN_COMMENT_DIVERGENCES` is currently EMPTY. The single row it used to carry
(`quantized` / `mlx/backend/metal/kernels/quantized.h`) was inherited from
promoted organizer frontier 79683c633b13c63aa23f112756a9c6b5173705b0 and went
dead when frontier sync c8dceb9 plus campaign regeneration 08fb76a made the two
comment blocks byte-identical. An earlier revision of THIS script detected that
and said so; the row was then deleted, because a waiver whose digests point at a
body that no longer exists still keeps its (stem, header) key waivable. Part A
pins the empty state: it fails if a row reappears without this file being
updated to justify it, and it fails if the real quantized.h section is divergent
again.

PART B -- THE MACHINERY.
The fail-closed logic must stay correct for the day a future frontier sync
legitimately needs a waiver. Part B therefore builds a SYNTHETIC divergent pair
out of the real checked-in section, installs a synthetic allowlist row in
memory, and asserts the waiver is fail-closed on every mutation class that could
matter:

  1. any change to a non-comment line,
  2. any change to the checked-in comment block,
  3. any change to the regenerated comment block,
  4. deletion of a comment line,
  5. application to any other (stem, header) pair.

`comment_only_waiver` is exercised as a pure function and the synthetic row is
removed again before exit, so this test cannot perturb the working tree, the
allowlist on disk, or the built kernels.

Run after any edit to `twin_audit.py`, to either QMV twin, or to the allowlist:

    python3 research/twin_waiver_negative_control.py

Exit 0 means the live table is honest AND the waiver cannot hide a code change.
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

import twin_audit as ta  # noqa: E402

STEM = "quantized"
HEADER = "mlx/backend/metal/kernels/quantized.h"

failures = []


def check(label, note, want_waived):
    got_waived = note is not None
    ok = got_waived == want_waived
    print(
        f"{'PASS' if ok else 'FAIL'}  {label}: "
        f"waived={got_waived} (expected {want_waived})"
    )
    if not ok:
        failures.append(label)


def assert_true(label, condition):
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


with tempfile.TemporaryDirectory(prefix="twin-negctl-") as scratch:
    checked, regenerated = ta.regenerate(
        STEM, ta.GEN_DIR / f"{STEM}.cpp", pathlib.Path(scratch)
    )

checked_sections = dict(checked["sections"])
regenerated_sections = dict(regenerated["sections"])

print("PART A -- the live allowlist")

assert_true(
    "KNOWN_COMMENT_DIVERGENCES is empty",
    ta.KNOWN_COMMENT_DIVERGENCES == {},
)

assert_true(
    f"{HEADER} is present in both the twin and the regenerated source",
    HEADER in checked_sections and HEADER in regenerated_sections,
)

real_checked = list(checked_sections.get(HEADER, []))
real_regenerated = list(regenerated_sections.get(HEADER, []))

assert_true(
    f"{HEADER} has NO divergence at all (no waiver is needed)",
    real_checked == real_regenerated,
)

check(
    "real section pair is not waivable with an empty table",
    ta.comment_only_waiver(STEM, HEADER, real_checked, real_regenerated),
    False,
)

print()
print("PART B -- the fail-closed machinery, exercised on a synthetic waiver")

# Build a synthetic comment-only divergence out of the REAL checked-in section:
# same code lines, one comment line textually different. This is exactly the
# shape of the divergence the allowlist exists for.
comment_idx = next(
    (i for i, line in enumerate(real_checked) if line.strip().startswith("//")),
    None,
)
if comment_idx is None:
    print("FAIL  the real section has no comment line to build a synthetic pair from")
    failures.append("synthetic pair construction")
    cbody = rbody = []
else:
    cbody = list(real_checked)
    rbody = list(real_checked)
    rbody[comment_idx] = rbody[comment_idx] + " (synthetic frontier variant)"
    assert_true(
        "synthetic pair differs in comment text only",
        cbody != rbody and ta.code_lines(cbody) == ta.code_lines(rbody),
    )

    ta.KNOWN_COMMENT_DIVERGENCES[(STEM, HEADER)] = {
        "checked_in_sha256": ta.body_digest(cbody),
        "regenerated_sha256": ta.body_digest(rbody),
        "reason": "synthetic; installed in memory by the negative control only",
    }

try:
    if cbody:
        # 0. Baseline: the synthetic pair must be waived, or the rest proves nothing.
        note = ta.comment_only_waiver(STEM, HEADER, cbody, rbody)
        check("baseline synthetic pair", note, True)
        if note is not None and "comment-only divergence" not in note:
            failures.append("baseline note text")

        # 1. A non-comment line changes: the semantic guard must fire.
        code_idx = next(
            i
            for i, line in enumerate(cbody)
            if "DIRECT_NIBBLES" in line and not line.strip().startswith("//")
        )
        mutated = list(cbody)
        mutated[code_idx] = mutated[code_idx].replace(
            "DIRECT_NIBBLES", "DIRECT_NIBBLES_X"
        )
        check(
            f"code line mutated (checked-in line {code_idx})",
            ta.comment_only_waiver(STEM, HEADER, mutated, rbody),
            False,
        )

        # 2. A checked-in comment line changes: the sha256 pin must fire.
        mutated = list(cbody)
        mutated[comment_idx] = mutated[comment_idx] + " tampered"
        check(
            f"comment line mutated (checked-in line {comment_idx})",
            ta.comment_only_waiver(STEM, HEADER, mutated, rbody),
            False,
        )

        # 3. A regenerated comment line changes: the other sha256 pin must fire.
        mutated_regen = list(rbody)
        mutated_regen[comment_idx] = mutated_regen[comment_idx] + " tampered"
        check(
            f"comment line mutated (regenerated line {comment_idx})",
            ta.comment_only_waiver(STEM, HEADER, cbody, mutated_regen),
            False,
        )

        # 4. A deleted comment line changes length without changing any code line.
        mutated = [line for i, line in enumerate(cbody) if i != comment_idx]
        check(
            "comment line deleted (checked-in)",
            ta.comment_only_waiver(STEM, HEADER, mutated, rbody),
            False,
        )

        # 5. The waiver is keyed on one exact (stem, header) pair and nothing else.
        check(
            "wrong header key",
            ta.comment_only_waiver(
                STEM, "mlx/backend/metal/kernels/gemm.h", cbody, rbody
            ),
            False,
        )
        check(
            "wrong stem key",
            ta.comment_only_waiver("gemm", HEADER, cbody, rbody),
            False,
        )
        check(
            "checked-in compared against itself",
            ta.comment_only_waiver(STEM, HEADER, cbody, cbody),
            False,
        )
finally:
    ta.KNOWN_COMMENT_DIVERGENCES.pop((STEM, HEADER), None)

assert_true(
    "the synthetic row was removed again",
    ta.KNOWN_COMMENT_DIVERGENCES == {},
)

print()
if failures:
    print(f"NEGATIVE CONTROL FAILED: {len(failures)} case(s): {failures}")
    sys.exit(1)
print(
    "NEGATIVE CONTROL PASSED: the allowlist is empty and unnecessary, and the "
    "waiver machinery cannot hide a code or comment change."
)
