#!/usr/bin/env python3
"""E37 r2 addendum: what the startup memory policy actually does on the
QWEN-MTP worker path, local (48 GiB) vs ranked (128 GiB).

Why this exists
---------------
The r2 addendum asserts that the two boxes run different allocator regimes and
that my warm-coverage negative was therefore measured under the more punishing
one. That is a claim about which branch a `guard` takes, so it is checkable
from source, and it is checkable in a way that can FAIL. This file is the
check, not a restatement.

It does three things:

  1. Extracts the gates and constants structurally (brace-matched function
     bodies, not "the string appears somewhere in the file") from the shipped
     policy, both MTP worker copies, the MTP block session, and the vendored
     MLX allocator/device.
  2. Derives the resolved regime for a 48 GiB, a 128 GiB and a 72 GiB host by
     simulating the extracted logic, so the table in the report is generated
     rather than typed.
  3. Runs mutation negative controls: for each check, a targeted edit of the
     source text that SHOULD break it, with the requirement that the check
     actually flips to FAIL. A gate that has never failed is not a gate --
     that is the same discipline the r2 request demanded of my proxies, and
     the reason the traced-vs-untraced control in this experiment was rebuilt
     after it turned out to be comparing a run against a copy of itself.

Zero GPU, zero model load, no timing. Read-only on the tree.

  python3 research/e37_alloc_regime.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

REPO = pathlib.Path(__file__).resolve().parent.parent

POLICY = "Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift"
SESSION = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
MTP_TRUSTED = "Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift"
MTP_HARNESS = "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift"
LAGUNA = "Sources/MLXFastModel/LagunaRuntimeWeights.swift"
DFLASH_TRUSTED = "Sources/MLXFastTrustedHarness/QwenRuntimeDFlashWorker.swift"
DFLASH_HARNESS = "Sources/MLXFastHarness/QwenRuntimeDFlashWorker.swift"
MLX_ALLOC = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp"
MLX_DEVICE_H = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.h"
MLX_DEVICE_CPP = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp"
MLX_METAL_DIR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal"

GIB = 1 << 30

Tree = dict[str, str]


def load_tree() -> Tree:
    paths = [
        POLICY, SESSION, MTP_TRUSTED, MTP_HARNESS, LAGUNA,
        DFLASH_TRUSTED, DFLASH_HARNESS, MLX_ALLOC, MLX_DEVICE_H, MLX_DEVICE_CPP,
    ]
    tree: Tree = {}
    for rel in paths:
        p = REPO / rel
        if not p.is_file():
            raise SystemExit("missing source file: %s" % rel)
        tree[rel] = p.read_text(encoding="utf-8")
    # Every other file in the vendored metal backend, for the scope check on
    # the command-buffer knobs.
    for p in sorted((REPO / MLX_METAL_DIR).rglob("*")):
        if p.is_file() and p.suffix in {".cpp", ".h"}:
            rel = str(p.relative_to(REPO))
            tree.setdefault(rel, p.read_text(encoding="utf-8"))
    return tree


def body_after(text: str, signature: str, open_char: str = "{") -> str:
    """Brace-matched body of the declaration whose signature line contains
    `signature`. Structural, so a check cannot be satisfied by a coincidental
    match elsewhere in the file."""
    i = text.find(signature)
    if i < 0:
        raise LookupError(signature)
    j = text.find(open_char, i)
    if j < 0:
        raise LookupError(signature + " (no body)")
    close = {"{": "}", "(": ")"}[open_char]
    depth, k = 0, j
    while k < len(text):
        if text[k] == open_char:
            depth += 1
        elif text[k] == close:
            depth -= 1
            if depth == 0:
                return text[j : k + 1]
        k += 1
    raise LookupError(signature + " (unbalanced)")


def line_of(text: str, needle: str) -> int:
    i = text.find(needle)
    return -1 if i < 0 else text.count("\n", 0, i) + 1


@dataclass
class Check:
    cid: str
    what: str
    fn: Callable[[Tree], str]
    evidence: str = ""


@dataclass
class Result:
    cid: str
    what: str
    ok: bool
    detail: str
    evidence: str = ""


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def c1_low_threshold(tree: Tree) -> str:
    src = tree[POLICY]
    if "fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30" not in src:
        raise AssertionError("full-profile minimum is not 64 GiB")
    resolve = body_after(src, "public static func resolve(")
    if "lowMemory = physicalMemoryBytes < fullProfileMinimumPhysicalMemoryBytes" \
            not in resolve:
        raise AssertionError("auto branch does not select low profile by <64 GiB")
    return "low profile iff physicalMemory < 64 GiB (auto branch)"


def c2_installer_gate(tree: Tree) -> str:
    body = body_after(
        tree[POLICY], "private static func installQwenMTPFullProfileCommandBufferDefaults("
    )
    if "guard physicalMemoryBytes >= (UInt64(96) << 30) else { return }" not in body:
        raise AssertionError("command-buffer installer is not gated at >=96 GiB")
    if 'setenv("MLX_MAX_MB_PER_BUFFER", "512", 0)' not in body:
        raise AssertionError("installer does not default MB_PER_BUFFER to 512")
    if 'setenv("MLX_MAX_OPS_PER_BUFFER", "50", 0)' not in body:
        raise AssertionError("installer does not default OPS_PER_BUFFER to 50")
    return "installer gated >=96 GiB; installs 512/50 with overwrite=0 (default, not forced)"


def c3_installer_runs_before_branch(tree: Tree) -> str:
    resolve = body_after(tree[POLICY], "public static func resolve(")
    call = resolve.find("installQwenMTPFullProfileCommandBufferDefaults(")
    branch = resolve.find("let lowMemory: Bool")
    if call < 0:
        raise AssertionError("resolve() does not call the command-buffer installer")
    if not 0 <= call < branch:
        raise AssertionError("installer does not run before the profile branch")
    return "installer runs inside resolve() on BOTH boxes, before the low/full branch"


def c4_low_constants(tree: Tree) -> str:
    resolve = body_after(tree[POLICY], "public static func resolve(")
    low = body_after(resolve, "if lowMemory {")
    want = [
        "isLowMemory: true,", "cacheLimitBytes: 6 << 30,",
        "maxMegabytesPerCommandBuffer: 128,", "maxOperationsPerCommandBuffer: 64,",
        "clearAllocatorCacheAfterWarmup: true,", "environmentOverrides: [:]",
    ]
    for w in want:
        if w not in low:
            raise AssertionError("low profile missing %r" % w)
    return "low: cache 6 GiB, 128 MB/buf, 64 ops/buf, clear-after-warmup flag TRUE, no env overrides"


def c5_full_constants(tree: Tree) -> str:
    resolve = body_after(tree[POLICY], "public static func resolve(")
    tail = resolve[resolve.rindex("return RuntimeStartupMemoryPolicy(") :]
    want = [
        "isLowMemory: false,", "cacheLimitBytes: 32 << 30,",
        "maxMegabytesPerCommandBuffer: 320,", "maxOperationsPerCommandBuffer: 128,",
        "clearAllocatorCacheAfterWarmup: false,",
    ]
    for w in want:
        if w not in tail:
            raise AssertionError("full profile missing %r" % w)
    return "full: cache 32 GiB, 320 MB/buf, 128 ops/buf, clear flag FALSE"


def c6_mtp_worker_forces(tree: Tree) -> str:
    for rel in (MTP_TRUSTED, MTP_HARNESS):
        body = body_after(tree[rel], "private func applyQwenMTPStartupMemoryProfile()")
        if "guard policy.isLowMemory else { return }" not in body:
            raise AssertionError("%s: no isLowMemory guard" % rel)
        for name, field_ in (
            ("MLX_MAX_MB_PER_BUFFER", "maxMegabytesPerCommandBuffer"),
            ("MLX_MAX_OPS_PER_BUFFER", "maxOperationsPerCommandBuffer"),
        ):
            forced = 'setenv("%s", String(policy.%s), 1)' % (name, field_)
            if forced not in body:
                raise AssertionError("%s: %s is not force-set (overwrite=1)" % (rel, name))
        if "Memory.cacheLimit = policy.cacheLimitBytes" not in body:
            raise AssertionError("%s: cache limit not applied" % rel)
        guard = body.index("guard policy.isLowMemory else { return }")
        if body.index("Memory.cacheLimit = policy.cacheLimitBytes") < guard:
            raise AssertionError("%s: cache limit applied before the guard" % rel)
    return ("both MTP worker copies: low-memory-only, then FORCE 128/64 (overwrite=1) "
            "and set cacheLimit; ranked returns at the guard")


def c7_mtp_worker_ignores_clear_flag(tree: Tree) -> str:
    for rel in (MTP_TRUSTED, MTP_HARNESS):
        if "clearAllocatorCacheAfterWarmup" in tree[rel]:
            raise AssertionError("%s references clearAllocatorCacheAfterWarmup" % rel)
        if "policy.apply()" in tree[rel]:
            raise AssertionError("%s calls policy.apply()" % rel)
    return ("neither MTP worker reads clearAllocatorCacheAfterWarmup nor calls apply(): "
            "the flag is INERT on this path")


def c8_clear_flag_consumers(tree: Tree) -> str:
    consumers = set()
    for rel, src in tree.items():
        if not rel.startswith("Sources/") or rel == POLICY:
            continue
        if "clearAllocatorCacheAfterWarmup" in src:
            consumers.add(rel)
    expected = {LAGUNA, DFLASH_TRUSTED, DFLASH_HARNESS}
    if consumers != expected:
        raise AssertionError("clear-flag consumers are %s, expected %s"
                             % (sorted(consumers), sorted(expected)))
    return "the clear-after-warmup flag is honoured only by Laguna + the two DFlash workers"


def c9_postwarm_clear_is_ranked_only(tree: Tree) -> str:
    src = tree[SESSION]
    warm = body_after(src, "public func warmAllDepths(maxDepth: Int) throws")
    if "Self.wireResidentWeightsIfEnabled()" not in warm:
        raise AssertionError("warmAllDepths does not call wireResidentWeightsIfEnabled")
    wire = body_after(src, "private static func wireResidentWeightsIfEnabled()")
    gate = 'guard ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30)'
    if gate not in wire:
        raise AssertionError("wired-residency path is not gated at >=96 GiB")
    if 'guard environment["DARKBLOOM_QWEN_MTP_WIRED_ZH"] != "0" else { return }' not in wire:
        raise AssertionError("wired-residency kill switch missing")
    if "Memory.clearCache()" not in wire:
        raise AssertionError("wired-residency path does not clear the buffer cache")
    if wire.index(gate) > wire.index("Memory.clearCache()"):
        raise AssertionError("clearCache runs before the 96 GiB gate")
    if src.count("Memory.clearCache()") != 1:
        raise AssertionError("Qwen36MTPBlockSession has %d clearCache calls, expected 1"
                             % src.count("Memory.clearCache()"))
    return ("the ONLY post-warm clearCache on the MTP session path sits behind the same "
            ">=96 GiB gate as wired residency: it runs on RANKED, not locally")


def c10_mlx_default_cache_limit(tree: Tree) -> str:
    src = tree[MLX_ALLOC]
    if "block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);" not in src:
        raise AssertionError("MLX block limit formula changed")
    if "max_pool_size_ = block_limit_;" not in src:
        raise AssertionError("MLX default cache limit is no longer block_limit_")
    return "MLX default cache limit = min(1.5 x maxRecommendedWorkingSet, 0.95 x memsize)"


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def c11_clear_cache_is_buffers_only(tree: Tree) -> str:
    body = body_after(tree[MLX_ALLOC], "void MetalAllocator::clear_cache()")
    if "buffer_cache_.clear()" not in body:
        raise AssertionError("clear_cache no longer clears the buffer cache")
    # Comments are stripped first: the check is about what the function DOES,
    # so a comment mentioning kernels must not be able to trip it, and a
    # mutation that only adds a comment must not be able to "kill" it either.
    code = strip_c_comments(body).lower()
    for forbidden in ("kernel", "pipeline", "library"):
        if forbidden in code:
            raise AssertionError("clear_cache touches %s state" % forbidden)
    if "library_kernels_" not in tree[MLX_DEVICE_H]:
        raise AssertionError("PSO cache library_kernels_ not found on Device")
    if "library_kernels_" in tree[MLX_ALLOC]:
        raise AssertionError("allocator can reach the PSO cache")
    return ("clear_cache() frees BUFFERS only; compiled pipelines live in "
            "Device::library_kernels_ and no allocator knob can evict them")


def c12_cmdbuf_knobs_are_device_scope(tree: Tree) -> str:
    hits = sorted(
        rel for rel, src in tree.items()
        if rel.startswith(MLX_METAL_DIR)
        and ("max_ops_per_buffer_" in src or "max_mb_per_buffer_" in src)
    )
    expected = sorted([MLX_DEVICE_CPP, MLX_DEVICE_H])
    if hits != expected:
        raise AssertionError("command-buffer knobs reachable from %s, expected %s"
                             % (hits, expected))
    dev = tree[MLX_DEVICE_CPP]
    if "max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);" not in dev:
        raise AssertionError("ops-per-buffer env override missing")
    if "max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);" not in dev:
        raise AssertionError("mb-per-buffer env override missing")
    return ("MLX_MAX_*_PER_BUFFER are Device-scope commit thresholds read once at Device "
            "construction; they appear in no kernel-selection site")


def c13_full_profile_is_a_noop_everywhere(tree: Tree) -> str:
    """Every site that applies the policy is guarded by isLowMemory, so the
    full-profile constants (320 MB / 128 ops / 32 GiB) are applied on NO host.
    This is documented intent, not drift -- the DFlash worker says so in
    source -- but it means the ranked box does not run the numbers a reader of
    the policy struct would assume it runs."""
    sites = {
        LAGUNA: ("if policy.isLowMemory {", "policy.apply()"),
        DFLASH_TRUSTED: ("if resolvedStartupMemoryPolicy.isLowMemory {",
                         'setenv(\n                "MLX_MAX_MB_PER_BUFFER",'),
        DFLASH_HARNESS: ("if resolvedStartupMemoryPolicy.isLowMemory {",
                         'setenv(\n                "MLX_MAX_MB_PER_BUFFER",'),
        MTP_TRUSTED: ("guard policy.isLowMemory else { return }",
                      'setenv("MLX_MAX_MB_PER_BUFFER"'),
        MTP_HARNESS: ("guard policy.isLowMemory else { return }",
                      'setenv("MLX_MAX_MB_PER_BUFFER"'),
    }
    for rel, (guard, application) in sites.items():
        src = tree[rel]
        if guard not in src:
            raise AssertionError("%s: low-memory guard missing" % rel)
        if application not in src:
            raise AssertionError("%s: expected application site missing" % rel)
        if src.index(guard) > src.index(application):
            raise AssertionError("%s: policy applied before its low-memory guard" % rel)
    if "profile stays a deliberate no-op" not in tree[DFLASH_TRUSTED]:
        raise AssertionError("the shipped rationale for the full-profile no-op is gone")
    return ("all 5 application sites are low-memory-guarded: the 320/128/32 GiB full-profile "
            "constants are applied on NO host (documented intent)")


CHECKS = [
    Check("C1", "low profile is selected by physicalMemory < 64 GiB", c1_low_threshold,
          "%s:%s" % (POLICY, "fullProfileMinimumPhysicalMemoryBytes")),
    Check("C2", "the 512/50 command-buffer defaults are gated at >=96 GiB", c2_installer_gate,
          POLICY),
    Check("C3", "the installer runs inside resolve() on both boxes", c3_installer_runs_before_branch,
          POLICY),
    Check("C4", "low-profile constants", c4_low_constants, POLICY),
    Check("C5", "full-profile constants (never applied on the MTP path)", c5_full_constants, POLICY),
    Check("C6", "MTP worker forces 128/64 + 6 GiB only when low-memory", c6_mtp_worker_forces,
          "%s / %s" % (MTP_TRUSTED, MTP_HARNESS)),
    Check("C7", "MTP worker ignores clearAllocatorCacheAfterWarmup and apply()",
          c7_mtp_worker_ignores_clear_flag, "%s / %s" % (MTP_TRUSTED, MTP_HARNESS)),
    Check("C8", "clear-after-warmup consumers are Laguna + DFlash only", c8_clear_flag_consumers,
          "Sources/**"),
    Check("C9", "the only post-warm clearCache on the MTP path is >=96 GiB gated",
          c9_postwarm_clear_is_ranked_only, SESSION),
    Check("C10", "MLX's default cache limit formula", c10_mlx_default_cache_limit, MLX_ALLOC),
    Check("C11", "clear_cache() cannot evict a compiled pipeline", c11_clear_cache_is_buffers_only,
          "%s / %s" % (MLX_ALLOC, MLX_DEVICE_H)),
    Check("C12", "command-buffer knobs are Device-scope only", c12_cmdbuf_knobs_are_device_scope,
          MLX_METAL_DIR),
    Check("C13", "the full profile is applied on no host (documented no-op)",
          c13_full_profile_is_a_noop_everywhere,
          "%s / %s / %s / %s / %s" % (LAGUNA, DFLASH_TRUSTED, DFLASH_HARNESS,
                                      MTP_TRUSTED, MTP_HARNESS)),
]


def run_checks(tree: Tree) -> list[Result]:
    out = []
    for chk in CHECKS:
        try:
            detail = chk.fn(tree)
            out.append(Result(chk.cid, chk.what, True, detail, chk.evidence))
        except (AssertionError, LookupError, KeyError, ValueError) as exc:
            out.append(Result(chk.cid, chk.what, False, "%s: %s" % (type(exc).__name__, exc),
                              chk.evidence))
    return out


# --------------------------------------------------------------------------
# derived regime, by simulating the extracted logic
# --------------------------------------------------------------------------

@dataclass
class Regime:
    host: str
    gib: int
    low_profile: bool
    mb_per_buffer: str
    ops_per_buffer: str
    cache_limit: str
    postwarm_clear_cache: bool
    wired_residency: bool
    notes: list[str] = field(default_factory=list)


def derive(host: str, gib: int) -> Regime:
    """Reproduce, in Python, the branch structure the checks just pinned."""
    installer = gib >= 96                      # C2/C3
    low = gib < 64                             # C1
    notes: list[str] = []

    if low:                                    # C6: worker forces, overwrite=1
        mb, ops = "128 (forced)", "64 (forced)"
        cache = "6 GiB (forced)"
    elif installer:                            # C2: defaults, overwrite=0
        mb, ops = "512 (default)", "50 (default)"
        cache = "MLX default = min(1.5 x maxRec, 0.95 x memsize)"
        notes.append("guard policy.isLowMemory returns before any cacheLimit assignment, "
                     "so the FULL-profile 320/128/32 GiB constants are never applied here")
    else:                                      # 64 GiB <= mem < 96 GiB
        mb, ops = "MLX built-in", "MLX built-in"
        cache = "MLX default = min(1.5 x maxRec, 0.95 x memsize)"
        notes.append("neither gate fires in the 64-96 GiB band: this host gets no campaign "
                     "command-buffer geometry at all")

    if installer:
        notes.append("wired residency ON and one Memory.clearCache() after warm, both from "
                     "wireResidentWeightsIfEnabled (C9)")
    else:
        notes.append("wired residency OFF and NO post-warm clearCache: warm-phase buffers stay "
                     "in the (capped) pool")

    return Regime(host, gib, low, mb, ops, cache, installer, installer, notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/results/e37/r2-alloc-regime.json")
    args = ap.parse_args()

    tree = load_tree()
    results = run_checks(tree)

    print("=" * 78)
    print("E37 r2 addendum - startup memory regime on the QWEN-MTP worker path")
    print("=" * 78)
    print("\n-- source checks (positive pass) " + "-" * 44)
    for r in results:
        print("  [%s] %-4s %s" % ("PASS" if r.ok else "FAIL", r.cid, r.what))
        print("         %s" % r.detail)
    positive_ok = all(r.ok for r in results)

    # ---- derived table -----------------------------------------------------
    regimes = [derive("local (this host)", 48), derive("ranked", 128), derive("unbanded", 72)]
    print("\n-- derived regime " + "-" * 58)
    hdr = "%-22s %-16s %-16s %s" % ("", "local 48 GiB", "ranked 128 GiB", "64-96 GiB band")
    print(hdr)
    rows = [
        ("profile", lambda x: "low" if x.low_profile else "full(unapplied)"),
        ("MLX_MAX_MB_PER_BUFFER", lambda x: x.mb_per_buffer),
        ("MLX_MAX_OPS_PER_BUFFER", lambda x: x.ops_per_buffer),
        ("Memory.cacheLimit", lambda x: x.cache_limit.split(" =")[0]),
        ("clearCache after warm", lambda x: "YES" if x.postwarm_clear_cache else "no"),
        ("wired residency", lambda x: "ON" if x.wired_residency else "OFF"),
    ]
    for label, fn in rows:
        print("%-22s %-16s %-16s %s" % (label, fn(regimes[0]), fn(regimes[1]), fn(regimes[2])))

    # ---- negative controls -------------------------------------------------
    print("\n-- negative controls (each mutation MUST break its check) " + "-" * 19)
    mutations = [
        ("M1", "C2", POLICY, "guard physicalMemoryBytes >= (UInt64(96) << 30)",
         "guard physicalMemoryBytes >= (UInt64(32) << 30)"),
        ("M2", "C1", POLICY, "fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30",
         "fullProfileMinimumPhysicalMemoryBytes = UInt64(128) << 30"),
        ("M3", "C6", MTP_TRUSTED, "guard policy.isLowMemory else { return }", ""),
        ("M4", "C7", MTP_TRUSTED, "Memory.cacheLimit = policy.cacheLimitBytes",
         "if policy.clearAllocatorCacheAfterWarmup { Memory.clearCache() }\n"
         "    Memory.cacheLimit = policy.cacheLimitBytes"),
        ("M5", "C6", MTP_TRUSTED,
         'setenv("MLX_MAX_MB_PER_BUFFER", String(policy.maxMegabytesPerCommandBuffer), 1)',
         'setenv("MLX_MAX_MB_PER_BUFFER", String(policy.maxMegabytesPerCommandBuffer), 0)'),
        ("M6", "C9", SESSION, "        try warmAllDepthShapes(maxDepth: maxDepth)",
         "        try warmAllDepthShapes(maxDepth: maxDepth)\n        Memory.clearCache()"),
        ("M7", "C10", MLX_ALLOC, "block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);",
         "block_limit_ = std::min(0.5 * max_rec_size, 0.95 * memsize);"),
        ("M8", "C12", MLX_ALLOC, "void MetalAllocator::clear_cache() {",
         "void MetalAllocator::clear_cache() {\n  (void)max_ops_per_buffer_;"),
        ("M9", "C4", POLICY, "maxMegabytesPerCommandBuffer: 128,",
         "maxMegabytesPerCommandBuffer: 256,"),
        ("M10", "C3", POLICY, "        installQwenMTPFullProfileCommandBufferDefaults(\n"
         "            physicalMemoryBytes: physicalMemoryBytes,\n"
         "            requestedProfile: requestedProfile\n        )\n", ""),
        ("M11", "C8", MTP_HARNESS, "Memory.cacheLimit = policy.cacheLimitBytes",
         "_ = policy.clearAllocatorCacheAfterWarmup\n"
         "    Memory.cacheLimit = policy.cacheLimitBytes"),
        ("M12", "C11", MLX_ALLOC,
         "void MetalAllocator::clear_cache() {\n  std::unique_lock lk(mutex_);",
         "void MetalAllocator::clear_cache() {\n  std::unique_lock lk(mutex_);\n"
         "  library_kernels_.clear();"),
        ("M13", "C13", LAGUNA, "            if policy.isLowMemory {\n                policy.apply()",
         "            if true {\n                policy.apply()"),
    ]

    control_rows, controls_ok = [], True
    baseline = {r.cid: r.ok for r in results}
    for mid, target, rel, old, new in mutations:
        mutated = dict(tree)
        src = mutated[rel]
        occurrences = src.count(old)
        if occurrences == 0:
            control_rows.append({"id": mid, "target": target, "file": rel, "ok": False,
                                 "why": "mutation anchor not found -- control is VACUOUS"})
            controls_ok = False
            print("  [VACUOUS] %-4s anchor missing in %s" % (mid, rel))
            continue
        if occurrences > 1:
            # An anchor that matches twice edits the FIRST site, which may not
            # be the one under test -- the mutation then "passes" while
            # proving nothing. This bit me on M12 (the clear_cache anchor also
            # matches inside malloc), and it is the same vacuous-control
            # failure mode as comparing a run against a copy of itself.
            control_rows.append({"id": mid, "target": target, "file": rel, "ok": False,
                                 "why": "anchor matches %d sites -- control is AMBIGUOUS"
                                        % occurrences})
            controls_ok = False
            print("  [AMBIGUOUS] %-4s anchor matches %d sites in %s"
                  % (mid, occurrences, rel))
            continue
        mutated[rel] = src.replace(old, new, 1)
        if mutated[rel] == src:
            control_rows.append({"id": mid, "target": target, "file": rel, "ok": False,
                                 "why": "mutation was a no-op -- control is VACUOUS"})
            controls_ok = False
            print("  [VACUOUS] %-4s mutation changed nothing in %s" % (mid, rel))
            continue
        after = {r.cid: r.ok for r in run_checks(mutated)}
        broke = baseline[target] and not after[target]
        controls_ok &= broke
        control_rows.append({"id": mid, "target": target, "file": rel, "ok": broke,
                             "why": "target check flipped to FAIL" if broke
                                    else "target check still PASSES under mutation"})
        print("  [%s] %-4s %s -> %s" % ("OK" if broke else "DEAD", mid, target,
                                        "check fails as required" if broke
                                        else "CHECK DID NOT FAIL"))

    verdict = positive_ok and controls_ok
    print("\n-- verdict " + "-" * 65)
    print("  positive checks : %s (%d/%d)"
          % ("PASS" if positive_ok else "FAIL",
             sum(1 for r in results if r.ok), len(results)))
    print("  negative controls: %s (%d/%d mutations killed their check)"
          % ("PASS" if controls_ok else "FAIL",
             sum(1 for c in control_rows if c["ok"]), len(control_rows)))
    print("  gate            : %s" % ("PASS" if verdict else "FAIL"))

    payload = {
        "schema": "e37-r2-alloc-regime/1",
        "repo_relative_sources": {
            "policy": POLICY, "session": SESSION,
            "mtp_worker_trusted": MTP_TRUSTED, "mtp_worker_harness": MTP_HARNESS,
            "mlx_allocator": MLX_ALLOC, "mlx_device_h": MLX_DEVICE_H,
        },
        "checks": [
            {"id": r.cid, "what": r.what, "pass": r.ok, "detail": r.detail,
             "evidence": r.evidence} for r in results
        ],
        "negative_controls": control_rows,
        "regime": [
            {"host": r.host, "gib": r.gib, "low_profile": r.low_profile,
             "mlx_max_mb_per_buffer": r.mb_per_buffer,
             "mlx_max_ops_per_buffer": r.ops_per_buffer,
             "memory_cache_limit": r.cache_limit,
             "postwarm_clear_cache": r.postwarm_clear_cache,
             "wired_residency": r.wired_residency,
             "notes": r.notes} for r in regimes
        ],
        "key_lines": {
            "installer_gate": "%s:%d" % (POLICY, line_of(
                tree[POLICY], "guard physicalMemoryBytes >= (UInt64(96) << 30)")),
            "low_threshold": "%s:%d" % (POLICY, line_of(
                tree[POLICY], "fullProfileMinimumPhysicalMemoryBytes = UInt64(64)")),
            "mtp_guard_trusted": "%s:%d" % (MTP_TRUSTED, line_of(
                tree[MTP_TRUSTED], "guard policy.isLowMemory else { return }")),
            "mtp_guard_harness": "%s:%d" % (MTP_HARNESS, line_of(
                tree[MTP_HARNESS], "guard policy.isLowMemory else { return }")),
            "wired_gate": "%s:%d" % (SESSION, line_of(
                tree[SESSION],
                "guard ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30)")),
            "session_clear_cache": "%s:%d" % (SESSION, line_of(
                tree[SESSION], "Memory.clearCache()")),
            "mlx_cache_limit": "%s:%d" % (MLX_ALLOC, line_of(
                tree[MLX_ALLOC], "max_pool_size_ = block_limit_;")),
        },
        "positive_pass": positive_ok,
        "controls_pass": controls_ok,
        "gate_pass": verdict,
    }
    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("  wrote %s" % args.json)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
