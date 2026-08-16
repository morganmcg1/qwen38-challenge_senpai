#!/usr/bin/env python3
"""Regenerate MLX Metal JIT twins and fail on any vendored-source drift.

The checked-in ``mlx-generated/*.cpp`` files are the runtime-effective Metal
source. MLX's own generator resolves the recursive quoted-include graph, keeps
system/angle includes, and embeds each vendored header in dependency order.
This audit runs that generator in a temporary directory and compares every
relative (vendored) section. Toolchain-owned absolute sections are ignored
because their paths vary by installed Metal toolchain.

Usage:
    research/twin_audit.py                 # every editable generated twin
    research/twin_audit.py gemm_nax ...    # selected generated stems
"""

import difflib
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


class AuditError(RuntimeError):
    pass


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


def first_difference(current, expected):
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
    with tempfile.TemporaryDirectory(prefix="qwen38-twin-audit-") as directory:
        temporary = pathlib.Path(directory)
        for stem in stems:
            current = GEN_DIR / f"{stem}.cpp"
            if not current.is_file():
                failures.append(f"{stem}: no generated twin at {current.relative_to(ROOT)}")
                continue
            try:
                checked, regenerated = regenerate(stem, current, temporary)
                difference = first_difference(checked, regenerated)
            except (AuditError, OSError) as error:
                failures.append(str(error))
                continue
            if difference:
                failures.append(f"{stem}: {difference}")
            else:
                print(
                    f"OK {stem}: {len(checked['sections'])} vendored section(s), "
                    f"{len(checked['system_sections'])} normalized toolchain section(s)"
                )

    if failures:
        for failure in failures:
            print(f"STALE {failure}", file=sys.stderr)
        print(
            f"TWIN AUDIT FAILED: {len(failures)}/{len(stems)} twin(s)",
            file=sys.stderr,
        )
        return 1
    print(f"TWIN AUDIT OK: {len(stems)} runtime-effective twin(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
