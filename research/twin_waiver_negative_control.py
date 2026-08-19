#!/usr/bin/env python3
"""Negative control for `research/twin_audit.py`'s comment-only waiver machinery.

A waiver in a drift audit is dangerous by construction: it is exactly the
mechanism by which a real edit could be silently absorbed. This script is the
falsification test for that risk, and it is written so that it keeps its teeth
whether or not any waiver is currently allowlisted.

PART A -- THE LIVE TABLE, WHICH CARRIES ONE ROW AGAIN.
`KNOWN_COMMENT_DIVERGENCES` has been emptied twice, both times because a
frontier sync moved one of the two pinned bodies and left the row waiving
nothing:

  * The key (`quantized` / `mlx/backend/metal/kernels/quantized.h`) was first
    waived for promoted frontier 79683c633b13c63aa23f112756a9c6b5173705b0. It
    went dead at frontier sync c8dceb9 plus campaign regeneration 08fb76a, an
    earlier revision of THIS script detected that, and the row was deleted.
  * The same key came back for promoted frontier
    036fd9ca2a2cac3b51c62a63237bd5d28c024487 when advisor merge a6eed9f adopted
    that frontier's `quantized.h` / `mlx-generated/quantized.cpp` pair with
    `--theirs`. It went dead at promoted frontier
    86fb1f020fc1fddc7e55aceac4761e5054b71dd6, which rewrote the whole `case 8`
    comment (retracting its own register-cliff rationale and moving the M=8
    wide-crossrow dispatch from 3+3+2 to 4+4 lanes), and at this branch taking
    campaign main's canonically regenerated twin blob d75b4a2f from 76b961f
    instead of upstream's unfaithful 72013491. THIS script detected it again --
    five checks red -- and the row was deleted again.

A dead row is a SILENT HOLE, which is the whole reason those five checks exist:
a waiver whose digests point at a body that no longer exists still keeps its
(stem, header) key waivable, so a later divergence in that exact section would
only have to reproduce two digests to be waived without a human revisiting the
table. Deleting beats re-pinning; regenerating the twin canonically beats both,
because it removes the fact instead of recording it.

The row came back a THIRD time at organizer 474c750 / advisor e468efd, and this
time it was re-pinned rather than fixed. The reasoning is recorded at the table
itself in twin_audit.py; the short version is that both twin paths are held
byte-identical to the frontier by the E27 revert, so editing either one to
remove the divergence would cost more than recording it does.

Part A therefore no longer pins EMPTINESS, which was never the real invariant.
The real invariant is that THE TABLE AND THE TREE AGREE -- a row exists exactly
when a divergence exists, and it pins the bodies actually present:

  A1. THE COVERAGE GUARD: every row is one this control actually snapshots. A
      row for some other (stem, header) makes this script stop covering the
      table, so it reds rather than passing on partial coverage,
  A2. the waived section is present in both the twin and the regenerated
      source, so A3 is a real comparison and not a vacuous one,
  A3. THE CLOSURE, BOTH WAYS. With a live row: the section really does diverge
      (so the row is not DEAD), both pinned digests match the LIVE bodies, the
      divergence is comment-only, and the row names its provenance. With no
      row: the section is byte-for-byte identical, so there is nothing to
      waive. Either way a future sync that changes the situation reds this,
  A4. the real pair is waived EXACTLY when a live row covers it -- never by a
      stale key, and never inertly (a row that fails to waive its own section
      is a row nobody has revisited),
  A5. THE TEETH: no single mutated CODE line in the real section becomes
      waivable. Against an empty table this was trivially true; against the
      live row it is now a real exercise of the code-lines guard, run over all
      3005 non-comment lines.

Part B below is what proves the machinery still has teeth, and it does so
against the REAL checked-in section body rather than a fabricated one.

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

print("PART A -- the live allowlist (table and tree must AGREE)")

# A1. THE COVERAGE GUARD. This control can only speak about the one section it
# snapshots. If a row ever appears for a different (stem, header) the control
# silently stops covering the table, so refuse instead: the operator must extend
# this script before that row is trusted.
assert_true(
    "every allowlist row is one this control actually checks",
    set(ta.KNOWN_COMMENT_DIVERGENCES) <= {(STEM, HEADER)},
)
if not set(ta.KNOWN_COMMENT_DIVERGENCES) <= {(STEM, HEADER)}:
    print(
        "      uncovered rows: "
        f"{sorted(set(ta.KNOWN_COMMENT_DIVERGENCES) - {(STEM, HEADER)})}"
    )

LIVE_ROW = dict(ta.KNOWN_COMMENT_DIVERGENCES.get((STEM, HEADER), {}))

# A2. The formerly-waived section still exists on both sides, so A3 compares
# something real instead of passing vacuously on a missing header.
assert_true(
    f"{HEADER} is present in both the twin and the regenerated source",
    HEADER in checked_sections and HEADER in regenerated_sections,
)

real_checked = list(checked_sections.get(HEADER, []))
real_regenerated = list(regenerated_sections.get(HEADER, []))

# A3. THE CLOSURE, IN BOTH DIRECTIONS. The table and the tree must agree about
# whether this section diverges. Two ways to be wrong, and both are hazards:
#
#   * a row with NO divergence is a DEAD ROW -- the silent hole this file's
#     docstring is about. Its (stem, header) key stays waivable, so a later
#     divergence needs only to reproduce two digests to pass unexamined.
#   * a divergence with NO row is an unrecorded drift, which is what the audit
#     exists to catch in the first place.
#
# When a row IS live, its two pinned digests must match the live bodies exactly
# -- that is what makes the row describe THIS divergence and not a remembered
# one -- and the divergence must be comment-only.
divergent = real_checked != real_regenerated
if LIVE_ROW:
    assert_true(
        f"{HEADER} really does diverge, so the live row is not a DEAD row",
        divergent,
    )
    assert_true(
        "the row's checked_in_sha256 pins the LIVE twin body",
        ta.body_digest(real_checked) == LIVE_ROW["checked_in_sha256"],
    )
    assert_true(
        "the row's regenerated_sha256 pins the LIVE regenerated body",
        ta.body_digest(real_regenerated) == LIVE_ROW["regenerated_sha256"],
    )
    assert_true(
        "the waived divergence is COMMENT-ONLY (every code line matches)",
        ta.code_lines(real_checked) == ta.code_lines(real_regenerated),
    )
    assert_true(
        "the row names where it came from and who adopted it",
        bool(LIVE_ROW.get("inherited_from")) and bool(LIVE_ROW.get("adopted_by")),
    )
else:
    assert_true(
        f"{HEADER} is byte-for-byte IDENTICAL, so there is nothing to waive",
        not divergent,
    )
if divergent:
    print(
        f"      {len(real_checked)} checked-in vs {len(real_regenerated)} "
        f"regenerated line(s); non-comment lines "
        f"{'match' if ta.code_lines(real_checked) == ta.code_lines(real_regenerated) else 'DIFFER'}"
    )

# A4. The real pair is waived EXACTLY when a live row covers it. With no row,
# nothing may waive it -- not even a stale key left behind by a deleted row.
# With a row, it must actually be waived, or the row is inert and the audit is
# red for a reason nobody recorded.
check(
    "real section pair is waived iff a live row covers it",
    ta.comment_only_waiver(STEM, HEADER, real_checked, real_regenerated) is not None,
    bool(LIVE_ROW),
)

# A5. THE TEETH AGAINST A STALE KEY. Mutating any single real CODE line must not
# become waivable. With an empty table every case is trivially refused; the
# check is kept so it retains its meaning on the day a row legitimately returns,
# and so a re-added row can never be merged without passing it.
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
    f"nothing waives ANY of {len(live_code_indices)} single-code-line "
    "mutations of the real section",
    not live_code_leaks,
)
if live_code_leaks:
    print(f"      leaked at checked-in line indices: {live_code_leaks[:10]}")

# A6. A one-character comment edit on either side must also stay unwaived.
live_comment_idx = next(
    (i for i, line in enumerate(real_checked) if line.strip().startswith("//")),
    None,
)
assert_true("the live section has a comment line", live_comment_idx is not None)
if live_comment_idx is not None:
    mutated = list(real_checked)
    mutated[live_comment_idx] = mutated[live_comment_idx] + " x"
    check(
        "a checked-in comment edit is not waived",
        ta.comment_only_waiver(STEM, HEADER, mutated, real_regenerated),
        False,
    )
    mutated_regen = list(real_regenerated)
    mutated_regen[0] = mutated_regen[0] + " x"
    check(
        "a regenerated-body edit is not waived",
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

# PART C -- Part B must leave no residue. The synthetic row existed only inside
# the `try` above; if it survived, every later audit run in this process would
# be silently waiving a real section.
assert_true(
    "the synthetic row was removed and the live state was restored exactly",
    ta.KNOWN_COMMENT_DIVERGENCES.get((STEM, HEADER)) == (LIVE_ROW or None),
)
assert_true(
    "the live table has its original row count after Part B",
    len(ta.KNOWN_COMMENT_DIVERGENCES) == (1 if LIVE_ROW else 0),
)
assert_true(
    "the real pair's waiver status is unchanged by Part B",
    (ta.comment_only_waiver(STEM, HEADER, real_checked, real_regenerated) is not None)
    == bool(LIVE_ROW),
)

print()
print("PART D -- DEAD-WAIVER DETECTION (added 2026-08-19, ledger 175)")
print(
    "  A waiver whose digests point at a body that no longer exists is a silent\n"
    "  hole: the audit prints a PASS with the row still in the table, and the\n"
    "  NEXT real divergence in that section gets compared against a stale pin.\n"
    "  twin_audit.py said so in a comment for weeks without checking it. The\n"
    "  concrete trigger is regeneration -- alphonse's E44 r2 regenerated the twin\n"
    "  and passed 29/29 with ZERO waivers used while the row was still present.\n"
    "  These four cases prove the new check fires, does not misfire, and leaves\n"
    "  no residue."
)

_saved_exercised = set(ta.WAIVERS_EXERCISED)
try:
    # D1 POSITIVE CONTROL. Nothing exercised, the waived stem audited clean =>
    # the row is dead and must be reported. If this case ever passes silently
    # the whole check is decoration.
    ta.WAIVERS_EXERCISED.clear()
    dead = ta.dead_waivers([STEM])
    assert_true(
        "D1 dead waiver on a clean audit of the waived stem IS reported",
        dead == [(STEM, HEADER)],
    )

    # D2 The check must not fire once the waiver has actually done its job.
    ta.WAIVERS_EXERCISED.add((STEM, HEADER))
    assert_true(
        "D2 exercised waiver is NOT reported dead",
        ta.dead_waivers([STEM]) == [],
    )

    # D3 A stem that was never audited is EXEMPT. `twin_audit.py unary` must not
    # fail because a quantized waiver did not fire -- absence of evidence is not
    # a dead waiver, and a check that conflated them would make every
    # single-stem invocation red.
    ta.WAIVERS_EXERCISED.clear()
    assert_true(
        "D3 waived stem not audited => exempt, not reported",
        ta.dead_waivers(["unary"]) == [],
    )

    # D4 A synthetic SECOND row on the same stem must also be caught, so the
    # check is keyed on (stem, header) and not merely on "this stem has some
    # waiver somewhere".
    synthetic = (STEM, "mlx/backend/metal/kernels/does-not-exist.h")
    ta.KNOWN_COMMENT_DIVERGENCES[synthetic] = {
        "checked_in_sha256": "0" * 64,
        "regenerated_sha256": "1" * 64,
        "inherited_from": "synthetic",
        "adopted_by": "synthetic",
        "note": "synthetic row, removed below",
    }
    ta.WAIVERS_EXERCISED.add((STEM, HEADER))
    assert_true(
        "D4 second row on an exercised stem is still caught individually",
        ta.dead_waivers([STEM]) == [synthetic],
    )
    del ta.KNOWN_COMMENT_DIVERGENCES[synthetic]
    assert_true(
        "D4 synthetic row removed without residue",
        synthetic not in ta.KNOWN_COMMENT_DIVERGENCES,
    )
finally:
    ta.WAIVERS_EXERCISED.clear()
    ta.WAIVERS_EXERCISED.update(_saved_exercised)

assert_true(
    "D5 module state restored exactly",
    ta.WAIVERS_EXERCISED == _saved_exercised,
)

print()
if failures:
    print(f"NEGATIVE CONTROL FAILED: {len(failures)} case(s): {failures}")
    sys.exit(1)
if LIVE_ROW:
    print(
        "NEGATIVE CONTROL PASSED: the allowlist's single row is LIVE, not dead --"
        " both pinned digests match the bodies actually in the tree, the"
        " divergence they waive is comment-only across 3005 identical code"
        " lines, and the row names its organizer and advisor provenance."
        " Nothing waives any single-code-line mutation of that section, and the"
        " machinery -- exercised on a synthetic row and then removed without"
        " residue -- still cannot hide a code or comment change."
    )
else:
    print(
        "NEGATIVE CONTROL PASSED: the allowlist is empty, the section its last "
        "row used to waive is byte-for-byte identical on both sides (so there is "
        "nothing left to waive), nothing waives the real pair or any "
        "single-code-line mutation of it, and the waiver machinery -- exercised "
        "on a synthetic row and then removed without residue -- still cannot "
        "hide a code or comment change."
    )
