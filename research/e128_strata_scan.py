"""E128 F7 item 3, step 1: survey the QMV width-dispatch variation across the
456 board submission trees.

Reads /tmp/tree_ids.json, walks each submission tree with `git cat-file`, and
fingerprints every place a per-M QMV dispatch could live.
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

REPO = "/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-edward/workspace/target"

PATHS = {
    "qwen35": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
    "qh": "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "qcpp": "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
    "session": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
    "qdispatch": "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp",
}

CASES_RE = re.compile(r"let\s+cases\s*=\s*\[([^\]]*)\]", re.S)
PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
BLT_RE = re.compile(r"\bB\s*<\s*(\d+)\b")
QMVFAST_RE = re.compile(r"qmv_fast")
QMVQUAD_RE = re.compile(r"qmv_quad")


def cat(tree, path):
    p = subprocess.run(
        ["git", "cat-file", "-p", f"{tree}:{path}"], cwd=REPO, capture_output=True
    )
    if p.returncode != 0:
        return None
    return p.stdout.decode("utf-8", "replace")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main():
    ids = json.load(open("/tmp/tree_ids.json"))["full"]
    digests = defaultdict(Counter)
    feats = Counter()
    rec = {}
    for i, (sid, tree) in enumerate(sorted(ids.items())):
        entry = {"tree": tree}
        for key, path in PATHS.items():
            text = cat(tree, path)
            if text is None:
                entry[key] = None
                continue
            entry[key] = sha(text)
            digests[key][entry[key]] += 1
            if key == "qwen35":
                entry["qwen35_lines"] = text.count("\n") + 1
                entry["qwen35_qmv"] = text.lower().count("qmv")
                m = CASES_RE.search(text)
                entry["cases"] = (
                    {int(a): int(b) for a, b in PAIR_RE.findall(m.group(1))} if m else None
                )
            if key == "qh":
                entry["qh_B_lt"] = sorted({int(x) for x in BLT_RE.findall(text)})
                entry["qh_fast"] = len(QMVFAST_RE.findall(text))
                entry["qh_quad"] = len(QMVQUAD_RE.findall(text))
            if key == "qcpp":
                entry["qcpp_B_lt"] = sorted({int(x) for x in BLT_RE.findall(text)})
                entry["qcpp_lines"] = text.count("\n") + 1
        rec[sid] = entry
        feats[(entry.get("qwen35_lines"), tuple(entry.get("qh_B_lt") or ()))] += 1
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(ids)}", file=sys.stderr)

    for key in PATHS:
        print(
            f"{key}: {len(digests[key])} distinct contents over "
            f"{sum(digests[key].values())} trees"
        )
        for d, n in digests[key].most_common(8):
            print(f"    {n:4d}  {d}")
    print()
    print("(qwen35 line count, quantized.h `B < k` thresholds) histogram:")
    for k, n in feats.most_common(20):
        print(f"  {n:4d}  {k}")

    json.dump(rec, open("/tmp/e128_strata_scan.json", "w"))
    print("\nwrote /tmp/e128_strata_scan.json")


if __name__ == "__main__":
    main()
