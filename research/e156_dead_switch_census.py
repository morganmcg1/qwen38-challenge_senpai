#!/usr/bin/env python3
"""RULE 170 census: find every environment switch that is dropped at the
runtime-worker boundary.

`sanitizedRuntimeWorkerEnvironment` forwards only six prefixes and eleven exact
keys to the worker process. Any environment read performed by worker-linked code
under a name outside that allowlist always returns the shipped default, so an
A/B selected by that name ran the same arm twice.

The census classifies each source file as worker-linked or parent-only, then
reports every environment read in worker-linked code whose name cannot cross the
boundary.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "research" / "e156-dead-switch-census.json"

WORKER_ROOTS = (
    "Sources/MLXFastCore/",
    "Sources/MLXFastModel/",
    "Sources/MLXFastRuntimeWorkerSupport/",
    "Sources/MLXFastRuntimeWorkerCLI/",
    "Vendor/mlx-swift-lm/",
    "Vendor/mlx-swift/",
)

# Parent-only: these never link into mlxfast-runtime-worker.
PARENT_ROOTS = (
    "Sources/MLXFastCLI/",
    "Sources/MLXFastHarness/",
    "Sources/MLXFastTrustedHarness/",
    "Sources/MLXFastTransform/",
)

# Package manifests and SwiftPM plugins run in the build process, never in the
# worker process, so their environment reads cannot form a dead timed arm.
BUILD_TIME_RE = re.compile(r"(^|/)(Package\.swift$|Plugins/|Tests/)")

# A name can only form a dead A/B arm if worker-linked code reads it AND the
# worker actually reaches that read. These reads are worker-linked by file but
# adjudicated by inspecting their call sites.
ADJUDICATED: dict[tuple[str, int], dict[str, str]] = {
    ("Sources/MLXFastCore/ByteLimitParsing.swift", 30): {
        "verdict": "unreachable_from_worker",
        "evidence": "transformedWeightsByteLimit is called only from "
                    "MLXFastHarness/MLXFastTrustedHarness BenchmarkSupport.swift "
                    "and QwenRuntimePreflight.swift, both parent-only targets; "
                    "MLXFastRuntimeWorkerCLI and MLXFastModel never reference it.",
    },
}

ALLOWED_PREFIXES = ("DARKBLOOM_", "DYLD_", "LC_", "METAL_", "MLX_", "MTL_")
ALLOWED_EXACT = {
    "HF_HUB_OFFLINE", "HOME", "LANG", "LOGNAME", "PATH", "SHELL", "TERM",
    "TMPDIR", "TRANSFORMERS_OFFLINE", "USER", "__CF_USER_TEXT_ENCODING",
}

# environment["NAME"] , environment[ "NAME" ] , getenv("NAME")
READ_RE = re.compile(
    r'(?:environment\s*\[\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\]'
    r'|getenv\s*\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\))'
)


def crosses_boundary(name: str) -> bool:
    return name in ALLOWED_EXACT or name.startswith(ALLOWED_PREFIXES)


def classify(path: str) -> str:
    if BUILD_TIME_RE.search(path):
        return "build_time"
    if path.startswith(WORKER_ROOTS):
        return "worker"
    if path.startswith(PARENT_ROOTS):
        return "parent"
    return "other"


def tracked_swift_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.swift"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def strip_line_comment(line: str) -> str:
    """Drop a trailing // comment so a documented-but-unused name is not counted."""
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


def main() -> int:
    dead: list[dict] = []
    unreachable: list[dict] = []
    live_worker: list[dict] = []
    counts = {"parent": 0, "build_time": 0}
    scanned = 0

    for rel in tracked_swift_files():
        side = classify(rel)
        if side == "other":
            continue
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        scanned += 1
        for lineno, raw in enumerate(text.splitlines(), 1):
            code = strip_line_comment(raw)
            for match in READ_RE.finditer(code):
                name = match.group(1) or match.group(2)
                if side != "worker":
                    counts[side] += 1
                    continue
                record = {
                    "name": name,
                    "file": rel,
                    "line": lineno,
                    "source": raw.strip(),
                }
                if crosses_boundary(name):
                    live_worker.append(record)
                elif (rel, lineno) in ADJUDICATED:
                    record.update(ADJUDICATED[(rel, lineno)])
                    unreachable.append(record)
                else:
                    dead.append(record)

    for bucket in (dead, unreachable, live_worker):
        bucket.sort(key=lambda r: (r["name"], r["file"], r["line"]))
    dead_names = sorted({r["name"] for r in dead})

    result = {
        "rule": "RULE 170",
        "harness": "offline",
        "swift_files_scanned": scanned,
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "allowed_exact_keys": sorted(ALLOWED_EXACT),
        "worker_roots": list(WORKER_ROOTS),
        "parent_only_roots": list(PARENT_ROOTS),
        "parent_side_reads_ignored": counts["parent"],
        "build_time_reads_ignored": counts["build_time"],
        "e156_dead_switch_count": len(dead),
        "e156_dead_switch_names": dead_names,
        "dead_switches": dead,
        "unreachable_from_worker_count": len(unreachable),
        "unreachable_from_worker": unreachable,
        "live_worker_switch_count": len(live_worker),
        "live_worker_switches": live_worker,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"swift files scanned:        {scanned}")
    print(f"parent-side reads ignored:  {counts['parent']}")
    print(f"build-time reads ignored:   {counts['build_time']}")
    print()
    print(f"DEAD switches reachable in the worker: {len(dead)} reads, "
          f"{len(dead_names)} distinct names")
    for r in dead:
        print(f"  {r['name']:38s} {r['file']}:{r['line']}")
    print()
    print(f"worker-linked but unreachable: {len(unreachable)}")
    for r in unreachable:
        print(f"  {r['name']:38s} {r['file']}:{r['line']}  [{r['verdict']}]")
    print()
    print(f"LIVE worker switches (cross the boundary): {len(live_worker)}")
    print()
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
