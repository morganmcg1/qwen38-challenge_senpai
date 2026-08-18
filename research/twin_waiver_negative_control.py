#!/usr/bin/env python3
"""Negative control for `research/twin_audit.py`'s comment-only waiver machinery.

A waiver in a drift audit is dangerous by construction: it is exactly the
mechanism by which a real edit could be silently absorbed. This script is the
falsification test for that risk, and it is written so that it keeps its teeth
whether or not any waiver is currently allowlisted.

PART A -- THE LIVE TABLE.
`KNOWN_COMMENT_DIVERGENCES` currently carries EXACTLY ONE row,
(`quantized` / `mlx/backend/metal/kernels/quantized.h`), inherited -- not
authored -- from promoted organizer frontier
036fd9ca2a2cac3b51c62a63237bd5d28c024487 when advisor merge a6eed9f adopted that
frontier's `quantized.h` / `mlx-generated/quantized.cpp` pair with `--theirs`.
(The same key was waived once before, for frontier
79683c633b13c63aa23f112756a9c6b5173705b0; it went dead at frontier sync c8dceb9
plus campaign regeneration 08fb76a, an earlier revision of THIS script detected
that, and the row was deleted -- because a waiver whose digests point at a body
that no longer exists still keeps its (stem, header) key waivable.)

Part A pins the live state and, crucially, attacks the LIVE row rather than only
a synthetic one:

  A1. the table has exactly the one expected key, and no other,
  A2. the real quantized.h section really IS divergent (a live waiver that
      matches nothing is a silent hole and must be deleted instead),
  A3. the divergence really is comment-only -- every non-comment line matches,
  A4. the two pinned digests really are the digests of the two live bodies, so
      the row cannot be stale,
  A5. the real pair is waived,
  A6. THE TEETH: mutating ONE REAL CODE LINE inside the REAL checked-in section
      makes the LIVE row refuse to waive. This is the property that matters --
      it is proved against the row that actually ships, not against a synthetic
      stand-in,
  A7. the same for a one-character edit to either real comment block.

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

# A1. Exactly one row, and it is the one this file documents.
assert_true(
    "KNOWN_COMMENT_DIVERGENCES has exactly one row",
    len(ta.KNOWN_COMMENT_DIVERGENCES) == 1,
)
assert_true(
    f"the single row is keyed ({STEM!r}, {HEADER!r})",
    set(ta.KNOWN_COMMENT_DIVERGENCES) == {(STEM, HEADER)},
)

LIVE_ROW = dict(ta.KNOWN_COMMENT_DIVERGENCES.get((STEM, HEADER), {}))

assert_true(
    "the live row records where the divergence was inherited from",
    LIVE_ROW.get("inherited_from") == "036fd9ca2a2cac3b51c62a63237bd5d28c024487",
)

assert_true(
    f"{HEADER} is present in both the twin and the regenerated source",
    HEADER in checked_sections and HEADER in regenerated_sections,
)

real_checked = list(checked_sections.get(HEADER, []))
real_regenerated = list(regenerated_sections.get(HEADER, []))

# A2. A live waiver that matches nothing is a silent hole: it must be deleted,
# not left in place. So the divergence it claims to cover must actually exist.
assert_true(
    f"{HEADER} IS divergent, so the live waiver covers something real",
    real_checked != real_regenerated,
)

# A3. Structural: the divergence is comment-only.
assert_true(
    "the live divergence is comment-only (all non-comment lines match)",
    ta.code_lines(real_checked) == ta.code_lines(real_regenerated),
)

# A4. The pinned digests are the digests of the LIVE bodies, so the row is not
# stale. If a frontier sync moves either body, these two fail first and loudest.
assert_true(
    "pinned checked_in_sha256 == digest of the live checked-in body",
    LIVE_ROW.get("checked_in_sha256") == ta.body_digest(real_checked),
)
assert_true(
    "pinned regenerated_sha256 == digest of the live regenerated body",
    LIVE_ROW.get("regenerated_sha256") == ta.body_digest(real_regenerated),
)

# A5. The waiver fires on the real pair.
live_note = ta.comment_only_waiver(STEM, HEADER, real_checked, real_regenerated)
check("real section pair IS waived by the live row", live_note, True)
if live_note is not None:
    assert_true(
        "the live note names the header and reports the non-comment line count",
        HEADER in live_note
        and f"{len(ta.code_lines(real_checked))} non-comment" in live_note,
    )

# A6. THE TEETH. Attack the LIVE row directly: change one real CODE line inside
# the real checked-in body and the live waiver must refuse. Every code line is
# tried, not just one, so no single line is privileged.
live_code_indices = [
    i for i, line in enumerate(real_checked) if not line.strip().startswith("//")
]
assert_true(
    "the live section has code lines to mutate",
    len(live_code_indices) > 0,
)
live_code_leaks = []
for i in live_code_indices:
    mutated = list(real_checked)
    mutated[i] = mutated[i] + " /*X*/"
    if ta.comment_only_waiver(STEM, HEADER, mutated, real_regenerated) is not None:
        live_code_leaks.append(i)
assert_true(
    f"the live row refuses to waive ANY of {len(live_code_indices)} "
    "single-code-line mutations",
    not live_code_leaks,
)
if live_code_leaks:
    print(f"      leaked at checked-in line indices: {live_code_leaks[:10]}")

# A7. A one-character edit to either real comment block must also refuse.
live_comment_idx = next(
    (i for i, line in enumerate(real_checked) if line.strip().startswith("//")),
    None,
)
assert_true("the live section has a comment line", live_comment_idx is not None)
if live_comment_idx is not None:
    mutated = list(real_checked)
    mutated[live_comment_idx] = mutated[live_comment_idx] + " x"
    check(
        "live row refuses a checked-in comment edit",
        ta.comment_only_waiver(STEM, HEADER, mutated, real_regenerated),
        False,
    )
    mutated_regen = list(real_regenerated)
    mutated_regen[0] = mutated_regen[0] + " x"
    check(
        "live row refuses a regenerated-body edit",
        ta.comment_only_waiver(STEM, HEADER, real_checked, mutated_regen),
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

    # Temporarily SHADOW the live row with a synthetic one keyed identically, so
    # Part B's assertions are about the machinery and not about the live digests.
    # The finally block below restores the live row byte-for-byte.
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
    if LIVE_ROW:
        ta.KNOWN_COMMENT_DIVERGENCES[(STEM, HEADER)] = dict(LIVE_ROW)
    else:
        ta.KNOWN_COMMENT_DIVERGENCES.pop((STEM, HEADER), None)

assert_true(
    "the synthetic row was replaced by the live row again",
    ta.KNOWN_COMMENT_DIVERGENCES.get((STEM, HEADER)) == LIVE_ROW,
)
assert_true(
    "the live table still has exactly one row after Part B",
    set(ta.KNOWN_COMMENT_DIVERGENCES) == {(STEM, HEADER)},
)
assert_true(
    "the live row still waives the real pair after Part B",
    ta.comment_only_waiver(STEM, HEADER, real_checked, real_regenerated) is not None,
)

print()
if failures:
    print(f"NEGATIVE CONTROL FAILED: {len(failures)} case(s): {failures}")
    sys.exit(1)
print(
    "NEGATIVE CONTROL PASSED: the one live allowlist row covers a real "
    "comment-only divergence, its digests are current, it refuses to waive any "
    "single-code-line mutation of the real section, and the waiver machinery "
    "cannot hide a code or comment change."
)
