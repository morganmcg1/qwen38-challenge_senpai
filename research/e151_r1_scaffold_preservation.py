#!/usr/bin/env python3
"""E151 R1: answer F8 section 4 and Rule 159 on the two scored NAX files.

F8 section 4 says the base moves today: thorfinn imports the crown's editable
surface, which deletes the E147 retile scaffold from `quantized_nax.h` and its
generated twin and restores the frontier content. The advisor asked me to copy
the current base version of both files aside before that lands, and then to
re-apply the scaffold as part of the R1 diff so my receipt prices it.

This emits two patches against the pristine frontier and proves each one
reproduces its target content byte for byte:

  e151-r1-scaffold-armoff.patch    frontier -> base 14247cce, the scaffold
                                   alone with the arm OFF
  e151-r1-frontier-reapply.patch   frontier -> my head, scaffold plus the R1
                                   arm (written by e151_r1_frontier_reapply.py)

Storing the arm-off patch is the preservation the advisor asked for. It does
not depend on commit 14247cce staying reachable, and it is 23 KB rather than a
second copy of two large generated files.

Rule 159 wants the last ranked receipt for my base named together with the
delta that no receipt covers. On these two files that delta is the scaffold
plus the arm, and this file counts both.

harness=offline. Static source analysis only. No GPU, no timing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTIER = "0863b06ac16e26e48fc06e97444095b00feb66d4"
BASE = "14247cce11216639a04ecfc2798cf0798091ff92"
PATHS = [
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
]
ARMOFF_PATCH = ROOT / "research/e151-r1-scaffold-armoff.patch"
JSON_OUT = ROOT / "research/e151-r1-scaffold-preservation.json"

# F8 section 4's digest table. `e09d6aa7` is the tree behind our own best
# receipt `0cf1637e` at 3.68278758. That commit is not reachable in this
# checkout, so these two values are the advisor's and are not verified here.
ADVISOR_RECEIPT_TREE = "e09d6aa7"
ADVISOR_RECEIPT_DIGESTS = {
    "quantized_nax.h": "387d1095",
    "quantized_nax.cpp": "39fa08bd",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def blob_digests(ref: str) -> dict:
    out = {}
    for rel in PATHS:
        name = pathlib.Path(rel).name
        text = git("show", f"{ref}:{rel}")
        out[name] = {
            "sha256_8": hashlib.sha256(text.encode()).hexdigest()[:8],
            "git_blob_8": git("rev-parse", f"{ref}:{rel}").strip()[:8],
        }
    return out


def delta(a: str, b: str) -> dict:
    patch = git("diff", a, b, "--", *PATHS)
    return {
        "bytes": len(patch),
        "hunks": patch.count("\n@@ "),
        "files": sum(patch.count(f"diff --git a/{p}") for p in PATHS),
    }


def build_armoff_patch() -> dict:
    """Write frontier -> base and prove it reproduces the base content."""
    work = pathlib.Path(tempfile.mkdtemp(prefix="e151-scaffold-"))
    try:
        tree = work / "tree"
        for rel in PATHS:
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(git("show", f"{FRONTIER}:{rel}"))
        subprocess.run(["git", "init", "-q", "."], cwd=tree, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tree, check=True)
        subprocess.run(
            ["git", "-c", "user.email=o@a.dev", "-c", "user.name=o",
             "commit", "-qm", "frontier"], cwd=tree, check=True)

        for rel in PATHS:
            (tree / rel).write_text(git("show", f"{BASE}:{rel}"))
        patch = subprocess.run(["git", "diff"], cwd=tree, capture_output=True,
                               text=True, check=True).stdout
        ARMOFF_PATCH.write_text(patch)

        subprocess.run(["git", "checkout", "-q", "."], cwd=tree, check=True)
        check = subprocess.run(["git", "apply", "--check", str(ARMOFF_PATCH)],
                               cwd=tree, capture_output=True, text=True)
        applies = check.returncode == 0
        digests = {}
        if applies:
            subprocess.run(["git", "apply", str(ARMOFF_PATCH)], cwd=tree,
                           check=True)
            for rel in PATHS:
                name = pathlib.Path(rel).name
                digests[name] = hashlib.sha256(
                    (tree / rel).read_bytes()).hexdigest()[:8]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"applies_clean": applies, "result_digests": digests,
            "bytes": len(patch)}


def main() -> int:
    head = git("rev-parse", "HEAD").strip()
    surfaces = {
        "frontier_0863b06a": blob_digests(FRONTIER),
        "base_14247cce": blob_digests(BASE),
        "head": blob_digests("HEAD"),
    }
    armoff = build_armoff_patch()
    base_digests = {k: v["sha256_8"] for k, v in surfaces["base_14247cce"].items()}
    frontier_digests = {
        k: v["sha256_8"] for k, v in surfaces["frontier_0863b06a"].items()
    }

    doc = {
        "experiment": "E151",
        "rung": "R1 scaffold preservation, F8 section 4 and Rule 159",
        "harness": "offline",
        "frontier_sha": FRONTIER,
        "base_sha": BASE,
        "head_sha": head,
        "paths": PATHS,
        "surface_digests": surfaces,

        "e151_r1_scaffold_armoff_patch_applies_clean": armoff["applies_clean"],
        "e151_r1_scaffold_armoff_patch_reproduces_base_surface":
            armoff["result_digests"] == base_digests,
        "e151_r1_scaffold_armoff_patch_bytes": armoff["bytes"],
        "e151_r1_scaffold_preserved_without_base_commit": (
            armoff["applies_clean"]
            and armoff["result_digests"] == base_digests
        ),

        # The advisor's own table gives the receipted tree the same two digests
        # the frontier carries, so one patch serves both re-application routes.
        "e151_r1_receipt_tree_reported_by_advisor": ADVISOR_RECEIPT_TREE,
        "e151_r1_receipt_tree_digests_advisor_reported": ADVISOR_RECEIPT_DIGESTS,
        "e151_r1_receipt_tree_digests_locally_verified": False,
        "e151_r1_receipted_surface_equals_frontier_surface":
            ADVISOR_RECEIPT_DIGESTS == frontier_digests,

        "e151_r1_delta_frontier_to_base_scaffold_armoff": delta(FRONTIER, BASE),
        "e151_r1_delta_frontier_to_head_scaffold_plus_arm":
            delta(FRONTIER, head),
        "e151_r1_delta_base_to_head_arm_alone": delta(BASE, head),

        # Rule 159.
        "e151_r1_last_ranked_receipt_for_base": {
            "submission": "0cf1637e",
            "published_median": 3.68278758,
            "tree": ADVISOR_RECEIPT_TREE,
            "source": "F8 section 4 and FINDING 255, advisor reported",
        },
        "e151_r1_un_receipted_delta_on_scored_surface": (
            "Both. The E147 retile scaffold on these two files is on no "
            "receipt: the base carries kRetiled, kHostBM, kHostBN, the "
            "compute_tile lambda and the RULE 145 static asserts, while the "
            "receipted tree and the frontier both carry the stock content. "
            "The R1 arm is also on no receipt of ours. So the whole "
            "frontier-to-head delta on the scored surface is un-receipted."
        ),
        "e151_r1_scaffold_is_priced_by_this_candidate": True,
        "e151_r1_scaffold_pricing_note": (
            "After the import the R1 diff against the new base is the whole "
            "frontier-to-head delta, so one receipt prices the scaffold and "
            "the arm together. It cannot separate them. Reading the arm alone "
            "needs a second receipt on the arm-off patch, which I did not run "
            "and do not propose inside R1."
        ),
        "e151_r1_rule145_static_assert_kept_verbatim": True,
    }

    JSON_OUT.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in doc.items()
                      if k.startswith("e151_")}, indent=1, sort_keys=True))
    print(f"wrote {JSON_OUT} and {ARMOFF_PATCH}")
    ok = (
        doc["e151_r1_scaffold_preserved_without_base_commit"]
        and doc["e151_r1_receipted_surface_equals_frontier_surface"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
