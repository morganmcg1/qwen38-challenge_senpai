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
# The row is BACK, with new digests, for a divergence we did not author.
#
# Advisor-branch merge commit a6eed9f ("Sync advisor branch to promoted frontier
# 036fd9c") resolved Vendor/.../kernels/quantized.h and
# mlx-generated/quantized.cpp with --theirs, adopting the promoted frontier
# 036fd9ca2a2cac3b51c62a63237bd5d28c024487 (submission
# b1e2591b-13f2-4b17-baf1-2956ca9242df, ranked 3.19088426880882) byte-for-byte.
# That frontier's own twin pair carries the divergence back again: the readable
# header keeps a 13-line prose paragraph in the wide-crossrow ``case 8`` block
# (the "M = 8 ... register cliff, not work scaling", the row-independence
# exactness argument, a receipts line, and the streak-gate synergy note) while
# the JIT twin keeps a 3-line pointer comment back to the header. Our previous
# side of the merge had mirrored the two blocks; taking theirs reintroduced
# their state.
#
# We record it here instead of re-editing the vendored kernel for two reasons.
# (1) The divergence is COMMENT-ONLY and therefore semantically inert: the JIT
#     twin is the runtime-effective source, its non-comment body is identical to
#     the regenerated header body line for line (2928 == 2928 lines), and a
#     comment-only twin edit has been measured on this campaign at +-0.07% of
#     MTP true decode -- i.e. inside reproducibility noise.
# (2) Re-editing a vendored file we just adopted verbatim would put our branch
#     back out of byte-agreement with the promoted frontier for zero measurable
#     benefit, and every subsequent frontier sync would re-conflict on it. The
#     campaign-owned audit tool is the right place to carry an inherited fact.
#
# The waiver is still fail-closed on three independent conditions (see
# comment_only_waiver below): both pinned digests plus a structural guard that
# every non-comment line matches. So if the frontier -- or we -- ever change a
# single line of CODE in this section, or change either comment block, the audit
# reds immediately and this table must be revisited deliberately.
#
# research/twin_waiver_negative_control.py asserts the exact shape of this
# table and independently re-derives that the waiver cannot mask a code change.
KNOWN_COMMENT_DIVERGENCES = {
    ("quantized", "mlx/backend/metal/kernels/quantized.h"): {
        "checked_in_sha256": (
            "6a3ec412c7b4ae1cdd32765fc04dfc3cee663fea31e5f401fb4ac5331e7d1ea0"
        ),
        "regenerated_sha256": (
            "0031c13e7f8bf0368f537f18d063e0d09fc571ffdb2466a26365868793f32f78"
        ),
        "inherited_from": "036fd9ca2a2cac3b51c62a63237bd5d28c024487",
        "adopted_by": "a6eed9f7ab068dc21017189b2a88a75a7a95c45b",
        "note": (
            "case 8 register-cliff rationale: 13-line paragraph in the readable "
            "header, 3-line pointer in the runtime-effective JIT twin. "
            "2992 checked-in vs 3002 regenerated lines, 2928 non-comment lines "
            "identical on both sides."
        ),
    }
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
