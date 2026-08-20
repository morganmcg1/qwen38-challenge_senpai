#!/usr/bin/env python3
"""Rung 0: check every row of the advisor's ranked allocator timeline table
against the source of the merged base, and print the corrections.

Free, no GPU. Every check names the exact file and line it read, so the answer
can be reproduced by opening that line.

  research/e62_rung0.py --out research/e62-artifacts/e62-rung0.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

POLICY = pathlib.Path("Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift")
SESSION = pathlib.Path("Sources/MLXFastModel/Qwen36MTPBlockSession.swift")
WORKER_MTP = pathlib.Path("Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift")
WORKER = pathlib.Path("Sources/MLXFastHarness/QwenRuntimeWorker.swift")


def lines(path: pathlib.Path) -> list[str]:
    return path.read_text().splitlines()


def find(path: pathlib.Path, needle: str) -> list[int]:
    return [i + 1 for i, line in enumerate(lines(path)) if needle in line]


def at(path: pathlib.Path, number: int) -> str:
    return lines(path)[number - 1].strip()


def check(name: str, claim: str, observed, expected, note: str = "") -> dict:
    return {
        "row": name,
        "advisor_claim": claim,
        "observed": observed,
        "expected": expected,
        "verdict": "confirmed" if observed == expected else "CORRECTION",
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    editable = set(json.loads(pathlib.Path("benchmark.json").read_text())["editablePaths"])

    checks = []

    checks.append(check(
        "worker startup calls resolve",
        "applyQwenMTPStartupMemoryProfile() at QwenRuntimeMTPWorker.swift:491",
        find(WORKER_MTP, "private func applyQwenMTPStartupMemoryProfile"),
        [491],
        "called unconditionally at " + str(find(WORKER_MTP, "    applyQwenMTPStartupMemoryProfile()")),
    ))

    checks.append(check(
        "96 GiB gate in installQwenMTPFullProfileCommandBufferDefaults",
        "RuntimeStartupMemoryPolicy.swift:66 returns on a sub-96 GiB host",
        find(POLICY, "guard physicalMemoryBytes >= (UInt64(96) << 30)"),
        [66],
    ))

    checks.append(check(
        "editable setenv of the ranked command-buffer geometry",
        "setenv MLX_MAX_MB_PER_BUFFER=512 / MLX_MAX_OPS_PER_BUFFER=50, overwrite=1, at :75-76",
        [
            (number, at(POLICY, number))
            for number in find(POLICY, 'setenv("MLX_MAX_')
        ],
        [
            (75, 'setenv("MLX_MAX_MB_PER_BUFFER", "512", 1)'),
            (76, 'setenv("MLX_MAX_OPS_PER_BUFFER", "50", 1)'),
        ],
    ))

    checks.append(check(
        "isLowMemory guard in the MTP worker",
        "guard policy.isLowMemory else { return } at QwenRuntimeMTPWorker.swift:498",
        find(WORKER_MTP, "guard policy.isLowMemory else { return }"),
        [498],
        "on the ranked 128 GiB host resolve() returns isLowMemory=false, so this "
        "returns; on this 48 GiB host it returns ONLY under "
        "DARKBLOOM_STARTUP_MEMORY_PROFILE=full. Under the default auto profile "
        "48 GiB < 64 GiB makes isLowMemory true, the guard PASSES, and the "
        "following setenv pair overwrites any parent export with 128/64.",
    ))

    checks.append(check(
        "Qwen35RuntimeWeightCache is never constructed on the MTP worker path",
        "never constructed on this path",
        [
            path
            for path in subprocess.run(
                ["grep", "-rln", "Qwen35RuntimeWeightCache(",
                 "Sources/MLXFastHarness", "Sources/MLXFastTrustedHarness",
                 "Sources/MLXFastModel"],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            if "MTP" in path
        ],
        [],
    ))

    checks.append(check(
        "wired residency 96 GiB gate",
        "Qwen36MTPBlockSession.swift:226 returns on a sub-96 GiB host",
        find(SESSION, "guard ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30)"),
        [225],
        "the predicate is on :225 and its `else { return }` continuation is on "
        ":226, so the advisor's :226 names the return, not the predicate",
    ))

    checks.append(check(
        "wired residency is reached from warmAllDepths",
        "called from warmAllDepths at :287",
        find(SESSION, "Self.wireResidentWeightsIfEnabled()"),
        [287],
    ))

    checks.append(check(
        "wiredZHDefaultFraction / wiredZHDefaultSlackMB",
        "1.0 and 64 at :213-214",
        [
            (number, at(SESSION, number))
            for number in find(SESSION, "private static let wiredZHDefault")
        ],
        [
            (213, "private static let wiredZHDefaultFraction = 1.0"),
            (214, "private static let wiredZHDefaultSlackMB = 64"),
        ],
    ))

    checks.append(check(
        "trusted phase-start allocator reset",
        "QwenRuntimeWorker.swift:176-192 sets cacheLimit 6 GiB, clears, asserts zero",
        {
            "definition": find(WORKER, "static func resetRuntimeWorkerAllocatorForPhaseStart"),
            "cache_limit_constant": [
                at(WORKER, number)
                for number in find(WORKER, "trustedRuntimeWorkerPhaseStartCacheLimitBytes =")
            ],
            "clear_inside_the_reset": [
                number for number in find(WORKER, "Memory.clearCache()")
                if 176 <= number <= 192
            ],
            "assert": find(WORKER, "guard remainingCacheBytes == 0 else {"),
        },
        {
            "definition": [176],
            "cache_limit_constant": ["static let trustedRuntimeWorkerPhaseStartCacheLimitBytes = 6 << 30"],
            "clear_inside_the_reset": [178],
            "assert": [187],
        },
    ))

    checks.append(check(
        "cacheLimitBytes = 32 << 30 is dead on the MTP path",
        "apply() is never called on this path",
        subprocess.run(
            ["grep", "-rn", r"\.apply()", "Sources/"],
            capture_output=True, text=True, check=False,
        ).stdout.split(),
        ["Sources/MLXFastModel/LagunaRuntimeWeights.swift:368:", "policy.apply()"],
    ))

    checks.append(check(
        "clearAllocatorCacheAfterWarmup is dead on the MTP path",
        "read only in LagunaRuntimeWeights.swift:395 and the two DFlash workers",
        sorted(
            line.split(":")[0]
            for line in subprocess.run(
                ["grep", "-rn", "clearAllocatorCacheAfterWarmup ==", "Sources/"],
                capture_output=True, text=True, check=False,
            ).stdout.splitlines()
        ),
        sorted([
            "Sources/MLXFastModel/LagunaRuntimeWeights.swift",
            "Sources/MLXFastHarness/QwenRuntimeDFlashWorker.swift",
            "Sources/MLXFastTrustedHarness/QwenRuntimeDFlashWorker.swift",
        ]),
    ))

    checks.append(check(
        "trusted code invites an in-window cache-limit change",
        "QwenRuntimeWorker.swift:168-173 says editable code may change it",
        find(WORKER, "editable code may change `Memory.cacheLimit` again inside"),
        [169],
    ))

    checks.append(check(
        "submission surface of the two knob sites",
        "RuntimeStartupMemoryPolicy.swift and Qwen36MTPBlockSession.swift are submitted; "
        "QwenRuntimeWorker.swift and QwenRuntimeMTPWorker.swift are trusted",
        {
            "Sources/MLXFastModel": "Sources/MLXFastModel" in editable,
            "Sources/MLXFastHarness": "Sources/MLXFastHarness" in editable,
            "Sources/MLXFastTrustedHarness": "Sources/MLXFastTrustedHarness" in editable,
        },
        {
            "Sources/MLXFastModel": True,
            "Sources/MLXFastHarness": False,
            "Sources/MLXFastTrustedHarness": False,
        },
    ))

    payload = {
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "checks": checks,
        "confirmed": sum(1 for c in checks if c["verdict"] == "confirmed"),
        "corrections": sum(1 for c in checks if c["verdict"] == "CORRECTION"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    for entry in checks:
        print(f"[{entry['verdict']:10}] {entry['row']}")
        if entry["verdict"] == "CORRECTION":
            print(f"             observed: {entry['observed']}")
            print(f"             expected: {entry['expected']}")
    print(f"confirmed={payload['confirmed']} corrections={payload['corrections']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
