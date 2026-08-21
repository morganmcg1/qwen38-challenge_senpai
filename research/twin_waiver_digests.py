#!/usr/bin/env python3
"""Derive the exact allowlist row `research/twin_audit.py` would need for every
comment-only divergence currently present in a generated Metal twin.

WHY THIS EXISTS
---------------
`research/twin_audit.py` compares each vendored Metal header against the section
of the same name embedded in the checked-in `mlx-generated/<stem>.cpp` twin,
which is the runtime-effective JIT source. When the two differ only in comment
text the audit can waive the divergence, but ONLY against two pinned SHA-256
digests plus a structural guard that every non-comment line matches. Those
digests are computed over the section body exactly as the audit compares it
(`"\\n".join(body) + "\\n"`), so they cannot be eyeballed or reproduced with
`shasum` on a file -- they must come from the audit's own parser.

Twice now a pinned row has gone DEAD because a frontier sync moved one of the
two bodies (c8dceb9 + 08fb76a the first time, 86fb1f02 the second). A dead row
is a silent hole: it keeps its (stem, header) key waivable, so a later
divergence in that exact section only has to reproduce two digests to be waived
without anyone revisiting the table. This script makes re-deriving -- or
confirming the absence of -- those digests a single command, so the table is
never left stale out of friction.

USAGE
-----
    python3 research/twin_waiver_digests.py                # every stem
    python3 research/twin_waiver_digests.py quantized      # one stem
    python3 research/twin_waiver_digests.py quantized gemm # several

Exit status is 0 whether or not divergences are found -- this is a derivation
tool, not a gate. `research/twin_audit.py` is the gate and
`research/twin_waiver_negative_control.py` is its falsification test.

READ THE OUTPUT LIKE THIS
-------------------------
* "no divergent sections" is the GOOD state and means the allowlist should be
  EMPTY for that stem. Do not add a row to record a divergence that is gone.
* A section reported as CODE-DIVERGENT is a real drift and must NEVER be
  waived: regenerate the twin or fix the header.
* A section reported as COMMENT-ONLY prints a paste-ready row. Prefer
  regenerating the twin canonically (which removes the fact) over waiving it
  (which merely records it); waive only when byte-agreement with a promoted
  frontier is worth more than the cleanliness, and say so in the row's note.
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

import twin_audit as ta  # noqa: E402


def stems():
    return sorted(path.stem for path in ta.GEN_DIR.glob("*.cpp"))


def describe(stem):
    """Print every divergent section of one stem, with a paste-ready row."""
    current = ta.GEN_DIR / f"{stem}.cpp"
    if not current.exists():
        print(f"{stem}: NO SUCH GENERATED TWIN at {current}")
        return 0

    with tempfile.TemporaryDirectory(prefix="waiver-digests-") as directory:
        checked, regenerated = ta.regenerate(stem, current, pathlib.Path(directory))

    checked_sections = dict(checked["sections"])
    regenerated_sections = dict(regenerated["sections"])

    headers = [h for h in checked_sections if h in regenerated_sections]
    only_checked = sorted(set(checked_sections) - set(regenerated_sections))
    only_regenerated = sorted(set(regenerated_sections) - set(checked_sections))

    divergent = [
        h for h in headers if checked_sections[h] != regenerated_sections[h]
    ]

    print(f"=== {stem}")
    print(
        f"    sections: {len(checked_sections)} checked-in, "
        f"{len(regenerated_sections)} regenerated, {len(headers)} in both"
    )
    for header in only_checked:
        print(f"    ONLY IN CHECKED-IN TWIN : {header}")
    for header in only_regenerated:
        print(f"    ONLY IN REGENERATED     : {header}")

    if not divergent:
        print("    no divergent sections -- the allowlist should be EMPTY here")
        return 0

    for header in sorted(divergent):
        checked_body = checked_sections[header]
        regenerated_body = regenerated_sections[header]
        checked_code = ta.code_lines(checked_body)
        regenerated_code = ta.code_lines(regenerated_body)
        comment_only = checked_code == regenerated_code
        kind = "COMMENT-ONLY" if comment_only else "CODE-DIVERGENT"
        print()
        print(f"    {kind}: {header}")
        print(
            f"      lines      : {len(checked_body)} checked-in vs "
            f"{len(regenerated_body)} regenerated"
        )
        print(
            f"      non-comment: {len(checked_code)} checked-in vs "
            f"{len(regenerated_code)} regenerated"
        )
        if not comment_only:
            print(
                "      *** REAL DRIFT -- DO NOT WAIVE. Regenerate the twin or "
                "fix the header. ***"
            )
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(checked_code, regenerated_code))
                    if a != b
                ),
                min(len(checked_code), len(regenerated_code)),
            )
            print(f"      first differing non-comment index: {first}")
            for label, body in (
                ("checked-in ", checked_code),
                ("regenerated", regenerated_code),
            ):
                if first < len(body):
                    print(f"        {label}: {body[first]!r}")
            continue
        print("      paste-ready row for KNOWN_COMMENT_DIVERGENCES:")
        print(f"        ({stem!r}, {header!r}): {{")
        # The gate reads COMMENT digests under these exact key names
        # (twin_audit.comment_only_waiver). Printing body digests, or the
        # short key names, produced a row that raised KeyError in the gate.
        print(
            '            "checked_in_comment_sha256": (\n'
            f'                "{ta.comment_digest(checked_body)}"\n'
            "            ),"
        )
        print(
            '            "regenerated_comment_sha256": (\n'
            f'                "{ta.comment_digest(regenerated_body)}"\n'
            "            ),"
        )
        print('            "inherited_from": "<organizer commit>",')
        print('            "adopted_by": "<advisor merge commit>",')
        print(
            '            "note": (\n'
            f'                "{len(checked_body)} checked-in vs '
            f'{len(regenerated_body)} regenerated lines, "\n'
            f'                "{len(checked_code)} non-comment lines identical '
            'on both sides."\n'
            "            ),"
        )
        print("        },")
    return len(divergent)


def main(argv):
    requested = argv[1:] or stems()
    total = 0
    for stem in requested:
        total += describe(stem)
        print()
    print(
        f"{total} divergent section(s) across {len(requested)} stem(s). "
        "An empty allowlist is the strongest state twin_audit.py can be in."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
