#!/usr/bin/env python3
"""E50 support. Content-addressed tree identity for board submissions.

Ledger 149 trap: two submissions of the same tree are ONE tree and N replicates.
Deduping by submission id (or even by commit sha) is not enough -- a commit sha
depends on parent/author/timestamp, so byte-identical trees can carry different
commit shas. `upstream/submissions/<submission-id>` refs are fetched locally for
most board rows, so real content identity is available.

Emits /tmp/tree_ids.json with, per submission id:
  full  : git tree sha of the whole submitted snapshot
  build : sha256 over the `git ls-tree` lines of the paths that can affect the
          compiled candidate binary (everything else is inert for timing)

Two batched `git cat-file --batch-check` calls, no per-ref subprocess.
"""
import hashlib
import json
import subprocess
import sys

BUILD_PATHS = ["Sources", "Vendor", "Package.swift", "Package.resolved", "tools",
               "mtp-head.manifest.json", "mtp-head"]

V = "Vendor/mlx-swift/Source/Cmlx"
# The quantized matvec family. Ledger 173 puts psi_serial = 0.8525 of the SERIAL
# leg in here, so if the serial leg executes candidate code at all, edits to
# these files are the ones that must move it.
QMV = [f"{V}/mlx-generated/quantized.cpp",
       f"{V}/mlx-generated/quantized_nax.cpp",
       f"{V}/mlx-generated/quantized_utils.cpp",
       f"{V}/mlx-generated/metal/quantized.h",
       f"{V}/mlx-generated/metal/quantized_nax.h",
       f"{V}/mlx-generated/metal/quantized_utils.h",
       f"{V}/mlx/mlx/backend/metal/quantized.cpp",
       f"{V}/mlx/mlx/backend/metal/kernels/quantized.h",
       f"{V}/mlx/mlx/backend/metal/kernels/quantized_nax.h",
       f"{V}/mlx/mlx/backend/metal/kernels/quantized.metal"]
# Target-model code the depth-0 serial leg must execute (width 1), MTP excluded.
TARGET = [f"Sources/MLXFastModel/Qwen35{n}.swift" for n in
          ["Attention", "Block", "Cache", "Config", "FastEngine",
           "FastPathReadiness", "GatedDelta", "LinearWeight", "MLP", "Model",
           "Ops", "RoPE", "RuntimeWeights", "Weights"]] + [
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MoE.swift"]
# Reachable only from the candidate (drafting) leg.
MTP_ONLY = [f"Sources/MLXFastModel/Qwen36MTP{n}.swift" for n in
            ["BlockSession", "HeadAttachment", "ReferenceSession", "Target"]] + [
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift",
    "mtp-head.manifest.json", "mtp-head"]

PATH_GROUPS = {"qmv": QMV, "target": TARGET, "serial_shared": QMV + TARGET,
               "mtp_only": MTP_ONLY}
ALL_PATHS = sorted({p for g in PATH_GROUPS.values() for p in g})

ROWS = "/tmp/rows_live.json"
OUT = "/tmp/tree_ids.json"


def batch_check(specs):
    """Resolve many rev specs in one git call. Returns {spec: oid or None}."""
    p = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(rest)"],
                       input="".join(f"{s} {s}\n" for s in specs),
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"git cat-file failed: {p.stderr.strip()}")
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ("missing", "ambiguous"):
            out[parts[0]] = None
        elif len(parts) >= 2:
            out[parts[1]] = parts[0]
    return out


def main():
    rows = json.load(open(ROWS))
    ids = [r["id"] for r in rows
           if isinstance((r.get("officialMetrics") or {}).get("per_prompt"), list)
           and r.get("officialScore") is not None]
    if not ids:
        sys.exit("no metric-bearing rows selected")

    full_raw = batch_check([f"upstream/submissions/{s}^{{tree}}" for s in ids])
    full = {}
    for s in ids:
        oid = full_raw.get(f"upstream/submissions/{s}^{{tree}}")
        if oid:
            full[s] = oid

    wanted = sorted(set(BUILD_PATHS) | set(ALL_PATHS))
    path_raw = batch_check([f"upstream/submissions/{s}:{p}"
                            for s in full for p in wanted])

    def sig(s, paths):
        return hashlib.sha256("\n".join(
            f"{p}={path_raw.get(f'upstream/submissions/{s}:{p}')}"
            for p in paths).encode()).hexdigest()[:16]

    build = {s: sig(s, BUILD_PATHS) for s in full}
    groups = {g: {s: sig(s, paths) for s in full}
              for g, paths in PATH_GROUPS.items()}

    json.dump({"full": full, "build": build, "groups": groups,
               "missing": [s for s in ids if s not in full],
               "build_paths": BUILD_PATHS,
               "path_groups": PATH_GROUPS}, open(OUT, "w"))
    print(f"metric rows {len(ids)} | snapshots resolved {len(full)} | "
          f"missing {len(ids) - len(full)}")
    print(f"distinct full trees  {len(set(full.values()))}")
    print(f"distinct build trees {len(set(build.values()))}")
    for g in PATH_GROUPS:
        print(f"distinct {g:<14s} variants {len(set(groups[g].values()))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
