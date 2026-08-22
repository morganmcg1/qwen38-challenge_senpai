"""Which QMV dispatch do the 164 reference-schedule board rows use?

The reference-schedule rows are the only ones whose round count R is validated,
so the board-fitted cost curve rests on them. This prints the dispatch form
found in each such row's tree.
"""

import json
import re
import subprocess
from collections import Counter

from e128_rounds import load_rows, per_prompt

REPO = "/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-edward/workspace/target"
QH = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
REF_BEAGLE = 4.381818181818182

CROSSROW_M_RE = re.compile(r"qmv_fast_crossrow_affine4_g64_m<\s*T\s*,\s*(\d+)\s*,\s*(\d+)\s*>")
CROSSROW_RE = re.compile(r"qmv_fast_crossrow_affine4_g64<\s*T\s*,\s*(\d+)\s*>")
ANY_CROSSROW_RE = re.compile(r"crossrow[A-Za-z0-9_]*<[^>]*>")


def cat(tree, path):
    p = subprocess.run(
        ["git", "cat-file", "-p", f"{tree}:{path}"], cwd=REPO, capture_output=True
    )
    return None if p.returncode != 0 else p.stdout.decode("utf-8", "replace")


def main():
    trees = json.load(open("/tmp/tree_ids.json"))["full"]
    rows = load_rows()
    forms = Counter()
    scores = []
    examples = {}
    nref = 0
    for r in rows:
        e = per_prompt(r)
        if len(e) != 8:
            continue
        if abs(e["beagle"]["effective_mean_draft_len"] - REF_BEAGLE) > 1e-9:
            continue
        nref += 1
        tree = trees.get(r["id"])
        if tree is None:
            forms["no-local-tree"] += 1
            continue
        text = cat(tree, QH)
        if text is None:
            forms["no-quantized.h"] += 1
            continue
        mtbl = CROSSROW_M_RE.findall(text)
        plain = sorted({int(a) for a in CROSSROW_RE.findall(text)})
        if mtbl:
            key = "m-table:" + ",".join(f"{a}:{b}" for a, b in mtbl[:7])
        elif plain:
            key = "plain-crossrow M in " + str(plain)
        else:
            hits = sorted({m.group(0)[:48] for m in ANY_CROSSROW_RE.finditer(text)})
            key = "other:" + (hits[0] if hits else "no-crossrow")
        forms[key] += 1
        scores.append(r.get("officialScore"))
        examples.setdefault(key, (r["id"], tree, r.get("officialScore")))

    print(f"reference-schedule rows: {nref}")
    for k, n in forms.most_common():
        ex = examples.get(k)
        print(f"  {n:4d}  {k}")
        if ex:
            print(f"          example {ex[0][:8]} score {ex[2]}")


if __name__ == "__main__":
    main()
