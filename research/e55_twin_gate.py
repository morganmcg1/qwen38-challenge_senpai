#!/usr/bin/env python3
"""Research-only (qwen38-r1-e55): arm-independent twin gate.

WHY THIS EXISTS -- AND WHY IT IS NOT A WEAKENED `twin_audit.py`
--------------------------------------------------------------
`research/twin_audit.py` is the campaign gate and stays the authority. It waives
the one known comment-only divergence in the `quantized` section (the case-8
comment block, where the readable header argues for 3+3+2 and the twin correctly
describes the 4+4 the code actually contains) against TWO PINNED WHOLE-BODY
sha256 digests.

Pinning the WHOLE BODY means any code edit anywhere in that ~3000-line section
de-pins the waiver, so the audit reports STALE. That is correct fail-closed
behaviour, but it makes the audit unusable as a per-arm gate for an experiment
that edits this section, which is what most of this campaign does:

  * the E55 `m9two` arm edits case 9, so its bodies are not the pinned pair;
  * the E55 `base` and `base2` arms restore the pinned bodies, so THEY pass;
  * one pinned pair therefore cannot cover an A/B experiment on this section.

This gate asserts the same property, pinned to the DIVERGENCE instead of to the
whole body, so it holds for every arm:

  1. every non-comment line of the two bodies is byte-identical -- the same
     structural guard `twin_audit.comment_only_waiver` applies, and it is what
     actually proves the runtime-effective JIT string and the readable header
     compile to the same kernel;
  2. every DIFFERING line is a whole-line comment;
  3. the differing comment lines on each side hash to a PINNED digest, so the
     only tolerated divergence is the known case-8 block. A new or edited
     comment divergence anywhere in the section fails.

It is strictly narrower than the audit in what it tolerates (one pinned comment
block, not one pinned pair of ~3000-line bodies) and it cannot mask a code
change, because condition 1 is the audit's own guard. It does NOT replace
`twin_audit.py` in the promotion chain.

    python3 research/e55_twin_gate.py [stem ...]      # default: quantized
"""
import difflib
import hashlib
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import twin_audit as ta  # noqa: E402

# The known case-8 comment divergence, pinned by the digest of the differing
# comment lines on each side. Derived with --print-digests on campaign base
# a35bb006, where research/twin_audit.py independently WAIVES the same section.
PINNED_COMMENT_DIVERGENCE = {
    ("quantized", "mlx/backend/metal/kernels/quantized.h"): {
        "checked_in_comment_sha256": (
            "ada984911293a195244438605b3f2eb5a8bbcb6923d924ad011de2ae12664212"
        ),
        "regenerated_comment_sha256": (
            "5f747f1300c96a8773dfe2856bc067c2875e96752a106dbd70d67d2e0fc365c0"
        ),
        "note": (
            "case 8: the twin carries the correct 3-line 4+4 description; the "
            "readable header carries a 17-line argument for 3+3+2 that E46 "
            "measured at +18.72 % SLOWER and that neither file's code does. "
            "Waived by research/twin_audit.py against whole-body digests; "
            "pinned here against the differing comment lines so the gate "
            "survives an unrelated code edit in the same section."
        ),
    }
}


def comment_digest(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def differing_lines(current_body, expected_body):
    """The lines each side owns exclusively, in order."""
    matcher = difflib.SequenceMatcher(None, current_body, expected_body, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            left.extend(current_body[i1:i2])
        if tag in ("replace", "insert"):
            right.extend(expected_body[j1:j2])
    return left, right


def audit_stem(stem, print_digests=False):
    failures = []
    with tempfile.TemporaryDirectory(prefix="e55-twin-gate-") as directory:
        current_path = ta.GEN_DIR / f"{stem}.cpp"
        if not current_path.is_file():
            return [f"{stem}: no generated twin at {current_path}"]
        checked, regenerated = ta.regenerate(stem, current_path, pathlib.Path(directory))

    if [p for p, _ in checked["sections"]] != [p for p, _ in regenerated["sections"]]:
        return [f"{stem}: vendored include graph/order changed"]
    for key in ("wrapper", "root", "prologue", "system_sections"):
        if checked[key] != regenerated[key]:
            return [f"{stem}: {key} changed between twin and regenerated source"]

    for (header, current_body), (_, expected_body) in zip(
        checked["sections"], regenerated["sections"]
    ):
        if current_body == expected_body:
            print(f"    {stem}/{header}: byte-identical")
            continue

        # Condition 1 -- the audit's own structural guard.
        if ta.code_lines(current_body) != ta.code_lines(expected_body):
            diff = list(
                difflib.unified_diff(
                    ta.code_lines(current_body), ta.code_lines(expected_body),
                    fromfile=f"checked-in:{header}", tofile=f"regenerated:{header}",
                    lineterm="", n=1,
                )
            )[:12]
            failures.append(
                f"{stem}: NON-COMMENT drift in {header}\n" + "\n".join(diff)
            )
            continue

        left, right = differing_lines(current_body, expected_body)
        # Condition 2 -- everything that differs must be a whole-line comment.
        stray = [ln for ln in left + right if ln.strip() and not ln.strip().startswith("//")]
        if stray:
            failures.append(
                f"{stem}: divergence in {header} includes non-comment lines: {stray[:4]}"
            )
            continue

        left_digest, right_digest = comment_digest(left), comment_digest(right)
        if print_digests:
            print(f"    {stem}/{header}: checked_in_comment_sha256={left_digest}")
            print(f"    {stem}/{header}: regenerated_comment_sha256={right_digest}")

        # Condition 3 -- and it must be the ONE known block.
        pin = PINNED_COMMENT_DIVERGENCE.get((stem, header))
        if pin is None:
            failures.append(f"{stem}: UNPINNED comment divergence in {header}")
            continue
        if (left_digest != pin["checked_in_comment_sha256"]
                or right_digest != pin["regenerated_comment_sha256"]):
            failures.append(
                f"{stem}: comment divergence in {header} does not match the pinned "
                f"block (got {left_digest[:12]}/{right_digest[:12]}, expected "
                f"{pin['checked_in_comment_sha256'][:12]}/"
                f"{pin['regenerated_comment_sha256'][:12]})"
            )
            continue
        print(
            f"    {stem}/{header}: PINNED comment-only divergence "
            f"({len(left)} vs {len(right)} comment line(s), "
            f"{len(ta.code_lines(current_body))} non-comment line(s) identical)"
        )
    return failures


def main():
    args = [a for a in sys.argv[1:] if a != "--print-digests"]
    print_digests = "--print-digests" in sys.argv[1:]
    stems = args or ["quantized"]
    failures = []
    for stem in stems:
        print(f"E55 twin gate: {stem}")
        failures.extend(audit_stem(stem, print_digests))
    if failures:
        for failure in failures:
            print(f"STALE {failure}", file=sys.stderr)
        print(f"E55 TWIN GATE FAILED: {len(failures)} finding(s)", file=sys.stderr)
        return 1
    print(f"E55 TWIN GATE OK: {len(stems)} runtime-effective twin(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
