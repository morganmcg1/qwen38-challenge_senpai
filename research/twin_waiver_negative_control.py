#!/usr/bin/env python3
"""Negative control for `research/twin_audit.py`'s comment-only waiver.

`twin_audit.py` allowlists one *comment-only* section divergence inherited from
the promoted organizer frontier (see `KNOWN_COMMENT_DIVERGENCES`). A waiver in a
drift audit is dangerous by construction: it is exactly the mechanism by which a
real edit could be silently absorbed. This script is the falsification test for
that risk. It asserts the waiver is fail-closed on every mutation class that
could matter:

  1. any change to a non-comment line,
  2. any change to the checked-in comment block,
  3. any change to the regenerated comment block,
  4. deletion of a comment line,
  5. application to any other (stem, header) pair.

`comment_only_waiver` is exercised as a pure function on the real section bodies,
so this test cannot perturb the working tree or the built kernels.

Run after any edit to `twin_audit.py`, to either QMV twin, or to the allowlist:

    python3 research/twin_waiver_negative_control.py

Exit 0 means the waiver still cannot hide a code change.
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

import twin_audit as ta  # noqa: E402

STEM = "quantized"
HEADER = "mlx/backend/metal/kernels/quantized.h"

with tempfile.TemporaryDirectory(prefix="twin-negctl-") as scratch:
    checked, regenerated = ta.regenerate(
        STEM, ta.GEN_DIR / f"{STEM}.cpp", pathlib.Path(scratch)
    )

cbody = rbody = None
for header, current_body in checked["sections"]:
    for expected_header, expected_body in regenerated["sections"]:
        if header == expected_header == HEADER and current_body != expected_body:
            cbody, rbody = list(current_body), list(expected_body)
            break
    if cbody is not None:
        break

if cbody is None:
    # The allowlisted divergence is gone. That is a good outcome, not a failure:
    # the waiver row in twin_audit.py should then be deleted.
    print(
        f"NOTE: no divergence remains in {HEADER}; the allowlist row is now "
        "dead and should be removed from twin_audit.py."
    )
    sys.exit(0)

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


# 0. Baseline: the real, untouched pair must be waived, or the audit is red.
note = ta.comment_only_waiver(STEM, HEADER, cbody, rbody)
check("baseline unmutated pair", note, True)
if note is not None and "comment-only divergence" not in note:
    failures.append("baseline note text")

# 1. A non-comment line changes: the semantic guard must fire.
code_idx = next(
    i
    for i, line in enumerate(cbody)
    if "DIRECT_NIBBLES" in line and not line.strip().startswith("//")
)
mutated = list(cbody)
mutated[code_idx] = mutated[code_idx].replace("DIRECT_NIBBLES", "DIRECT_NIBBLES_X")
check(
    f"code line mutated (checked-in line {code_idx})",
    ta.comment_only_waiver(STEM, HEADER, mutated, rbody),
    False,
)

# 2. A checked-in comment line changes: the sha256 pin must fire.
comment_idx = next(i for i, line in enumerate(cbody) if line.strip().startswith("//"))
mutated = list(cbody)
mutated[comment_idx] = mutated[comment_idx] + " tampered"
check(
    f"comment line mutated (checked-in line {comment_idx})",
    ta.comment_only_waiver(STEM, HEADER, mutated, rbody),
    False,
)

# 3. A regenerated comment line changes: the other sha256 pin must fire.
regen_comment_idx = next(
    i for i, line in enumerate(rbody) if line.strip().startswith("//")
)
mutated_regen = list(rbody)
mutated_regen[regen_comment_idx] = mutated_regen[regen_comment_idx] + " tampered"
check(
    f"comment line mutated (regenerated line {regen_comment_idx})",
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
    ta.comment_only_waiver(STEM, "mlx/backend/metal/kernels/gemm.h", cbody, rbody),
    False,
)
check("wrong stem key", ta.comment_only_waiver("gemm", HEADER, cbody, rbody), False)
check(
    "checked-in compared against itself",
    ta.comment_only_waiver(STEM, HEADER, cbody, cbody),
    False,
)

print()
if failures:
    print(f"NEGATIVE CONTROL FAILED: {len(failures)} case(s): {failures}")
    sys.exit(1)
print("NEGATIVE CONTROL PASSED: the waiver cannot hide a code or comment change.")
