#!/usr/bin/env python3
"""Regenerate MLX Metal JIT twins and fail on any vendored-source drift.

The checked-in ``mlx-generated/*.cpp`` files are the runtime-effective Metal
source. MLX's own generator resolves the recursive quoted-include graph, keeps
system/angle includes, and embeds each vendored header in dependency order.
This audit runs that generator in a temporary directory and compares every
relative (vendored) section. Toolchain-owned absolute sections are ignored
because their paths vary by installed Metal toolchain.

A section may carry an allowlisted *comment-only* divergence (see
``KNOWN_COMMENT_DIVERGENCES``). Such a waiver is fail-closed three ways: both
section bodies are sha256-pinned, and every non-comment line must still be
byte-identical. Any code edit, and any edit to either comment block, re-reds the
audit. Waivers are printed rather than silently skipped.

Usage:
    research/twin_audit.py                 # every editable generated twin
    research/twin_audit.py gemm_nax ...    # selected generated stems
"""

import difflib
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent.parent
CMLX = ROOT / "Vendor/mlx-swift/Source/Cmlx"
GEN_DIR = CMLX / "mlx-generated"
MLX_ROOT = CMLX / "mlx"
GENERATOR = MLX_ROOT / "mlx/backend/metal/make_compiled_preamble.sh"
KERNEL_PREFIX = "mlx/backend/metal/kernels/"
EDITABLE_PREFIX = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/"

BANNER = re.compile(r'^// Contents from "(.+)"$')
ROOT_HEADER = re.compile(r"^// Auto generated source for (.+)$")
RULE = re.compile(r"^/{40,}$")

# Sections whose checked-in twin and readable header differ in COMMENT TEXT ONLY.
#
# Keyed by (generated stem, vendored section header). ``checked_in_sha256`` is the
# digest of the section body extracted from the checked-in mlx-generated/*.cpp
# twin; ``regenerated_sha256`` is the digest of the same section as regenerated
# from the readable vendored header. Both are pinned, so editing either comment
# block re-reds the audit and forces this table to be revisited deliberately.
# HISTORY OF THIS TABLE, so that the current row is read as a decision and not
# as a convenience.
#
# The key ("quantized", "mlx/backend/metal/kernels/quantized.h") was first
# waived while promoted organizer frontier
# 79683c633b13c63aa23f112756a9c6b5173705b0 shipped the long M=8 register-cliff
# rationale in the readable header and a short pointer comment in the
# runtime-effective twin. Frontier sync c8dceb9 (organizer
# d1530a409848b82a0a1890141c1483875d1e0173) plus the campaign regeneration
# 08fb76a removed that divergence, the audit went clean at 29/29, and the row
# was DELETED rather than left in place -- because a waiver whose digests point
# at a body that no longer exists still keeps its (stem, header) key waivable,
# and a future sync reintroducing ANY comment divergence in that exact section
# would then only have to reproduce two digests to be waived silently.
#
# The row came BACK, with new digests, for a divergence we did not author.
#
# Advisor-branch merge commit a6eed9f ("Sync advisor branch to promoted frontier
# 036fd9c") resolved Vendor/.../kernels/quantized.h and
# mlx-generated/quantized.cpp with --theirs, adopting the promoted frontier
# 036fd9ca2a2cac3b51c62a63237bd5d28c024487 (submission
# b1e2591b-13f2-4b17-baf1-2956ca9242df, ranked 3.19088426880882) byte-for-byte.
# That frontier's own twin pair carried the divergence back again: the readable
# header kept a 13-line prose paragraph in the wide-crossrow ``case 8`` block
# (the "M = 8 ... register cliff, not work scaling" rationale, the
# row-independence exactness argument, a receipts line, and the streak-gate
# synergy note) while the JIT twin kept a 3-line pointer comment back to the
# header. That row pinned checked_in 6a3ec412... / regenerated 0031c13e... and
# was waived on this branch from a6eed9f through advisor HEAD a3a351db.
#
# AND IT IS DEAD AGAIN -- FOR THE SECOND TIME, BY THE SAME MECHANISM.
#
# Advisor merge commit "Sync advisor branch to promoted frontier 86fb1f02"
# adopted promoted organizer frontier
# 86fb1f020fc1fddc7e55aceac4761e5054b71dd6 (submission
# 3a995c2b-3c42-48e8-b982-f36a8abda0e7, ranked 3.23222998733732). Two things
# changed at once in that sync:
#
#   (a) The frontier REWROTE the whole ``case 8`` comment. It retracted its own
#       register-cliff rationale and changed the wide-crossrow M=8 dispatch from
#       qmv_fast_crossrow_affine4_g64_m<T, 8, 3, true> to <T, 8, 4, true>, on
#       the argument that direct-nibble affine-4 lowered inner-loop register
#       pressure so two four-row groups stream each weight tile twice instead of
#       the three streams paid by 3+3+2. So both pinned digests moved: the
#       header body is no longer 6a3ec412..., the regenerated body is no longer
#       0031c13e....
#
#   (b) The frontier's own mlx-generated/quantized.cpp (blob 72013491) still
#       carried an ABBREVIATED comment while its quantized.h (blob 57e8ec84)
#       carried the full one -- i.e. upstream's checked-in twin is NOT a
#       faithful regeneration of its own header, exactly the defect this audit
#       exists to catch. Campaign main had already regenerated it canonically at
#       76b961f ("Regenerate quantized Metal twin canonically", blob d75b4a2f),
#       so this branch takes campaign main's repaired blob with
#       ``git checkout 50a5be6e -- Vendor/.../mlx-generated/quantized.cpp``
#       rather than inheriting upstream's unfaithful twin.
#
# With (b) applied the quantized section is byte-for-byte identical between the
# checked-in twin and the regenerated header. There is NO divergence left to
# waive, so the row is DELETED rather than re-pinned with fresh digests --
# exactly as it was deleted the first time it went dead at c8dceb9 + 08fb76a.
# The reason is the same and it is worth restating, because a dead waiver is a
# SILENT HOLE: a waiver whose digests point at a body that no longer exists
# still keeps its (stem, header) key waivable, so a future sync reintroducing
# ANY comment divergence in that exact section would only have to reproduce two
# digests to be waived without a human ever revisiting this table.
#
# The table is therefore EMPTY. An empty allowlist is the strongest state this
# audit can be in: every divergence, in every section, reds immediately.
# Regenerating the twin canonically (b) is strictly preferable to waiving,
# because it removes the fact instead of recording it; prefer that route again
# if a future frontier ships another unfaithful twin.
#
# When a waiver IS legitimately needed again, add exactly one row of the shape
#
#     ("<generated stem>", "<vendored section header>"): {
#         "checked_in_sha256": "<64 hex>",   # digest of the twin's section body
#         "regenerated_sha256": "<64 hex>",  # digest of the regenerated body
#         "inherited_from": "<organizer commit>",
#         "adopted_by": "<advisor merge commit>",
#         "note": "<what the two comment blocks say and the line counts>",
#     }
#
# and re-derive both digests with research/twin_waiver_digests.py against the
# live tree. The waiver is fail-closed on three independent conditions (see
# comment_only_waiver below): both pinned digests plus a structural guard that
# every non-comment line matches.
#
# research/twin_waiver_negative_control.py asserts the exact shape of this
# table -- that it carries no DEAD row, that each live row still describes a
# real divergence, and that the waiver machinery cannot mask a code change.
#
# AND IT IS BACK, FOR THE THIRD TIME, BY THE SAME MECHANISM. 2026-08-19.
#
# Organizer commit 474c750 (Accept submission
# 942e5ab2-1c46-4c50-b7c3-eaf948878ed0) ships blobs 12e2f73d (header) and
# 2429e888 (twin) that are once again mutually unfaithful, and advisor commit
# e468efd ("rebase the shipped surface onto the live frontier; drop E27")
# adopted both verbatim. Current upstream/main 0c90733d carries the same two
# blobs, so this is the LIVE frontier state, not a stale corner.
#
# WHY THIS ROW EXISTS INSTEAD OF A FIX, WHICH REVERSES THE PREFERENCE STATED
# ABOVE. The note above says canonical regeneration is "strictly preferable to
# waiving, because it removes the fact instead of recording it". That was
# written before these two paths acquired a second, stronger constraint, and it
# is now the wrong call for this section:
#
#   The E27 revert deliberately restored BOTH files to frontier byte-identity,
#   and that decision is worth 0.3321 % of score. research/scored-surface-gate.sh
#   marks them FRONTIER-TAKEN and ASSERTS the byte-identity as part of the ack.
#   Any edit -- including a comment-only one, in either file -- converts them
#   from "our overlay writes back exactly what is already there" into "our
#   overlay REPLACES the tip's copy of the hottest file in the competition",
#   which is the precise mechanism by which a submission silently reverts
#   organizer-accepted work. It also costs JIT bytes: mlx-generated/quantized.cpp
#   is Metal source compiled at runtime, and JIT cost is inside the timed window
#   on this benchmark, so comment lines here are not provably free.
#
#   So the cheapest correct action is to record the divergence, not remove it.
#   Prefer regeneration again for any section where byte-identity to the
#   frontier is NOT a live scored decision.
#
# ⚠️  THE DIVERGENCE IS NOT COSMETIC AND THE READABLE HEADER IS THE WRONG ONE.
#
#   quantized.h  case 8: 17 lines arguing for qmv_fast_crossrow_affine4_g64_m
#                        <T, 8, 3, true> (3+3+2) on a register-cliff theory,
#                        with an exactness argument, cross-row profiling
#                        numbers, two submission receipts and a synergy claim.
#   quantized.cpp case 8: 3 lines, correct, describing the 4+4 two-stream
#                        dispatch that BOTH files actually contain.
#
#   The header's argument is FALSE. Making the code match it is measured at
#   +18.72 % SLOWER (E46, thorfinn: 4-leg ABBA, pre-registered, 8/8 shapes,
#   sign p=0.0078, range +14.83..+21.34 %), corroborated at +19.02 % by the E27
#   probe 7b5183d on a different base -- agreement to 0.30 pp. The companion
#   contrast isolates the mechanism: widening the GROUP while holding streams
#   at 2 (<T,6,3> -> <T,6,4>) is null at +0.263 ms, under its own replicate
#   floor, so the cost is STREAM COUNT and not group width. Registers cannot
#   carry the story either: kernel-wide max is 108 in both arms, attained at
#   M=7 regardless.
#
#   Because the code cannot be defended by a comment we are choosing not to
#   write, it is defended by senpai/campaign-invariants.txt, which carries four
#   entries -- `present` on <T, 8, 4, true> and `absent` on <T, 8, 3, true>, in
#   both twins. Those were verified by CONSTRUCTION, not by reading: the exact
#   hazardous edit was applied and asserted to have landed. The case that
#   justifies them is the two-sided edit, which THIS audit cannot see, because
#   this audit asks whether the twins agree and not whether they are right.
KNOWN_COMMENT_DIVERGENCES = {
    ("quantized", "mlx/backend/metal/kernels/quantized.h"): {
        "checked_in_sha256": (
            "d2dc1f7d4938524a500c355b51a9fb631cb3500efffc73e8ca87fd6e2b627992"
        ),
        "regenerated_sha256": (
            "1995814a7f1b3f8859e0d14bfa61a694c8dcfb237465fb7598adf9bbb6abab49"
        ),
        "inherited_from": "474c750 (Accept submission 942e5ab2-1c46-4c50-b7c3-eaf948878ed0)",
        "adopted_by": "e468efd (rebase the shipped surface onto the live frontier; drop E27)",
        "note": (
            "3087 checked-in vs 3097 regenerated lines, 3005 non-comment lines "
            "identical on both sides. Header case 8 carries a 17-line argument "
            "for 3+3+2 that is measured +18.72 % SLOWER (E46) and is NOT what "
            "either file's code does; the twin's 3-line 4+4 comment is the "
            "correct one. Waived rather than fixed because both paths are held "
            "byte-identical to the frontier by the E27 revert. Code guarded by "
            "senpai/campaign-invariants.txt."
        ),
    },
}


class AuditError(RuntimeError):
    pass


def body_digest(body):
    """Digest a section body exactly as the audit compares it."""
    return hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest()


def code_lines(body):
    """Drop whole-line comments, keeping everything the Metal compiler sees."""
    return [line for line in body if not line.strip().startswith("//")]


def comment_only_waiver(stem, header, current_body, expected_body):
    """Return a printable note when this exact divergence is an allowed waiver.

    Fail-closed on three independent conditions: the pinned digest of the
    checked-in body, the pinned digest of the regenerated body, and a structural
    guard that every non-comment line still matches. If any of them fails the
    waiver does not apply and the caller reports real drift.
    """
    entry = KNOWN_COMMENT_DIVERGENCES.get((stem, header))
    if entry is None:
        return None
    if body_digest(current_body) != entry["checked_in_sha256"]:
        return None
    if body_digest(expected_body) != entry["regenerated_sha256"]:
        return None
    if code_lines(current_body) != code_lines(expected_body):
        return None
    return (
        f"WAIVED {stem}: comment-only divergence in {header} "
        f"({len(current_body)} checked-in vs {len(expected_body)} regenerated "
        f"line(s), {len(code_lines(current_body))} non-comment line(s) identical)"
    )


def embedded_block(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if 'R"preamble(' in line]
    ends = [index for index, line in enumerate(lines) if ')preamble";' in line]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise AuditError(f"{path.name}: expected exactly one compiled preamble")
    block = lines[starts[0] + 1 : ends[0]]
    wrapper = lines[: starts[0] + 1] + ["<COMPILED-PREAMBLE>"] + lines[ends[0] :]
    return block, wrapper


def snapshot(path):
    block, wrapper = embedded_block(path)
    roots = [match.group(1) for line in block if (match := ROOT_HEADER.match(line))]
    marks = [
        (index, match.group(1))
        for index, line in enumerate(block)
        if (match := BANNER.match(line))
    ]
    if len(roots) != 1 or not marks:
        raise AuditError(f"{path.name}: missing root metadata or embedded sections")

    sections = []
    system_sections = []
    for number, (index, header) in enumerate(marks):
        end = marks[number + 1][0] if number + 1 < len(marks) else len(block)
        body = block[index + 1 : end]
        while body and RULE.match(body[0]):
            body.pop(0)
        while body and RULE.match(body[-1]):
            body.pop()
        if header.startswith("/"):
            normalized_body = [
                re.sub(r'^(#line\s+\d+\s+").*(")$', r'\1<SYSTEM>\2', line)
                for line in body
            ]
            system_sections.append((pathlib.PurePosixPath(header).name, normalized_body))
            continue
        source = MLX_ROOT / header
        if not source.is_file():
            raise AuditError(f"{path.name}: unresolved vendored header {header}")
        sections.append((header, body))

    if not sections:
        raise AuditError(f"{path.name}: no vendored sections were checked")
    return {
        "root": roots[0],
        "wrapper": wrapper,
        "prologue": block[: marks[0][0]],
        "sections": sections,
        "system_sections": system_sections,
    }


def source_name(root_header):
    if not root_header.startswith(KERNEL_PREFIX) or not root_header.endswith(".h"):
        raise AuditError(f"unsupported generated root {root_header}")
    return root_header[len(KERNEL_PREFIX) : -2]


def regenerate(stem, current, temporary):
    current_snapshot = snapshot(current)
    source = source_name(current_snapshot["root"])
    if pathlib.PurePosixPath(source).name != stem:
        raise AuditError(
            f"{stem}: generated filename does not match root {current_snapshot['root']}"
        )

    output_dir = temporary / stem
    module_cache = temporary / "module-cache"
    output_dir.mkdir()
    command = [
        "bash",
        str(GENERATOR),
        str(output_dir),
        "cc",
        str(MLX_ROOT),
        source,
        f"-fmodules-cache-path={module_cache}",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise AuditError(f"{stem}: MLX generator failed\n{detail}")

    generated = output_dir / f"{stem}.cpp"
    if not generated.is_file():
        raise AuditError(f"{stem}: MLX generator produced no {generated.name}")
    return current_snapshot, snapshot(generated)


def first_difference(current, expected, stem, notes=None):
    if current["wrapper"] != expected["wrapper"]:
        return "generated C++ wrapper changed"
    if current["root"] != expected["root"]:
        return f"root changed: {current['root']} != {expected['root']}"
    if current["prologue"] != expected["prologue"]:
        return "generated preamble metadata changed"
    if current["system_sections"] != expected["system_sections"]:
        return "toolchain section count, order, or embedded body changed"

    current_paths = [path for path, _ in current["sections"]]
    expected_paths = [path for path, _ in expected["sections"]]
    if current_paths != expected_paths:
        return "vendored include graph/order changed: " + " | ".join(
            difflib.unified_diff(current_paths, expected_paths, lineterm="", n=1)
        )

    for (header, current_body), (_, expected_body) in zip(
        current["sections"], expected["sections"]
    ):
        if current_body == expected_body:
            continue
        waiver = comment_only_waiver(stem, header, current_body, expected_body)
        if waiver is not None:
            if notes is not None:
                notes.append(waiver)
            continue
        sample = list(
            difflib.unified_diff(
                current_body,
                expected_body,
                fromfile=f"checked-in:{header}",
                tofile=f"regenerated:{header}",
                lineterm="",
                n=1,
            )
        )[:12]
        return f"section drift in {header}\n" + "\n".join(sample)
    return None


def default_stems():
    manifest = json.loads((ROOT / "benchmark.json").read_text(encoding="utf-8"))
    stems = []
    for path in manifest["editablePaths"]:
        if path.startswith(EDITABLE_PREFIX) and path.endswith(".cpp"):
            stems.append(pathlib.PurePosixPath(path).stem)
    if not stems:
        raise AuditError("benchmark.json exposes no generated Metal twins")
    return sorted(set(stems))


def main():
    stems = sys.argv[1:] or default_stems()
    failures = []
    waivers = []
    with tempfile.TemporaryDirectory(prefix="qwen38-twin-audit-") as directory:
        temporary = pathlib.Path(directory)
        for stem in stems:
            current = GEN_DIR / f"{stem}.cpp"
            if not current.is_file():
                failures.append(f"{stem}: no generated twin at {current.relative_to(ROOT)}")
                continue
            notes = []
            try:
                checked, regenerated = regenerate(stem, current, temporary)
                difference = first_difference(checked, regenerated, stem, notes)
            except (AuditError, OSError) as error:
                failures.append(str(error))
                continue
            if difference:
                failures.append(f"{stem}: {difference}")
            else:
                for note in notes:
                    print(note)
                waivers.extend(notes)
                print(
                    f"OK {stem}: {len(checked['sections'])} vendored section(s), "
                    f"{len(checked['system_sections'])} normalized toolchain section(s)"
                    + (f", {len(notes)} allowlisted waiver(s)" if notes else "")
                )

    if failures:
        for failure in failures:
            print(f"STALE {failure}", file=sys.stderr)
        print(
            f"TWIN AUDIT FAILED: {len(failures)}/{len(stems)} twin(s)",
            file=sys.stderr,
        )
        return 1
    summary = f"TWIN AUDIT OK: {len(stems)} runtime-effective twin(s)"
    if waivers:
        summary += (
            f", {len(waivers)} allowlisted comment-only waiver(s) "
            "(non-comment lines byte-identical, both bodies sha256-pinned)"
        )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
