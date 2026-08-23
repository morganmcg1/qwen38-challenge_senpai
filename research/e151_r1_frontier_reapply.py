#!/usr/bin/env python3
"""E151 R1: the self-contained re-application of the arm onto the frontier.

harness=offline. No timing, no GPU.

F7 section 1 asks for the arm to be self-contained so that re-application onto
the imported frontier header is mechanical. This script answers that question
with a patch rather than a claim, and it reports the honest answer to "is the
arm self-contained today", which is NO.

WHAT THE THREE TREES CONTAIN.

  frontier  `upstream/main = 0863b06a`. Zero `kE147`, zero `kHostBM`, zero
            `kRetiled`. Pristine: no retile plumbing of any kind.
  base      our PR base `14247cce`. Already carries the whole E147 retile
            plumbing with the arm OFF: the `kHostBM`/`kHostBN` template
            parameters on `qmm_t_nax_tgp_impl`, the `compute_tile` lambda and
            grid-stride loop, the three RULE 145 guards, and
            `kE147NaxRetileOn = false`.
  ours      base plus R1: the arm turned ON, the loader-legality predicate,
            the arm-liveness `static_assert`, and the retiled call site.

So R1 by itself is only two hunks against our base, but it DEPENDS on the E147
plumbing, and the frontier has none of it. Re-applying R1 onto the frontier
therefore needs the plumbing as well. That is the finding.

THE USEFUL PART. The complete frontier-to-ours delta on the two submitted
files is exactly {E147 retile plumbing} + {RULE 145 guards} + {R1 arm} and
nothing else. There is no other E147 residue and no unrelated content. So the
re-application is not a merge at all: applying this patch to the imported
frontier reproduces our gate-proved surface byte for byte.

THE RISK THE ADVISOR MUST PRICE. The patch is larger than the arm. With the
arm OFF the E147 plumbing is already not AIR-identical to pristine
(`nax_arm_off_air_delta_bytes = 2112` from E147 rung E-1c), and it also lands
in `affine_qmm_n_nax` and `affine_gather_qmm_rhs_nax` through the shared
`qmm_t_nax_tgp_impl` and the RULE 145 guards. So `import then apply` is not
`import plus one arm`: it re-imports the arm-off plumbing too. That plumbing
ran in every E151 gate chain and in the promoted E147 census, so it is not
unmeasured, but it is not nothing either.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
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
GATE_PROVED = {
    "quantized_nax.h": "50cf7876",
    "quantized_nax.cpp": "e7c55209",
}
PATCH_OUT = ROOT / "research/e151-r1-frontier-reapply.patch"
JSON_OUT = ROOT / "research/e151-r1-frontier-reapply.json"

# Each hunk of the frontier-to-ours delta, classified by what it is for. A
# hunk that is not in this table is unclassified content and fails the audit.
HUNK_CLASSES = [
    (r"kHostBM = BM", "e147_plumbing",
     "the retile template parameters on qmm_t_nax_tgp_impl"),
    (r"constexpr bool kRetiled", "e147_plumbing",
     "the retile predicate, the cover and split static_asserts, and the "
     "hoisted base pointers the grid-stride loop needs"),
    (r"auto compute_tile = ", "e147_plumbing",
     "the compute_tile lambda and the grid-stride loop over required tiles"),
    (r"tile shape matches no tile_matmad_nax branch", "rule145_guard",
     "the RULE 145 tile-shape guard"),
    (r"kE147NaxRetileOn = true", "r1_arm",
     "the R1 arm: flag on, loader-legality predicate, arm-liveness assert, "
     "and the widened threadgroup staging"),
    (r"if constexpr \(kE147NaxRetiled\)", "r1_arm",
     "the retiled call site"),
]


def git(*args: str) -> str:
    p = subprocess.run(("git",) + args, cwd=ROOT, capture_output=True,
                       text=True, check=True)
    return p.stdout


def classify(hunk: str) -> tuple[str, str] | None:
    for pattern, cls, why in HUNK_CLASSES:
        if re.search(pattern, hunk):
            return cls, why
    return None


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="e151-reapply-"))
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
            (tree / rel).write_text(git("show", f"HEAD:{rel}"))
        patch = subprocess.run(["git", "diff"], cwd=tree, capture_output=True,
                               text=True, check=True).stdout
        PATCH_OUT.write_text(patch)

        # Prove the patch reproduces the gate-proved surface from the frontier.
        subprocess.run(["git", "checkout", "-q", "."], cwd=tree, check=True)
        check = subprocess.run(["git", "apply", "--check", str(PATCH_OUT)],
                               cwd=tree, capture_output=True, text=True)
        applies = check.returncode == 0
        digests = {}
        if applies:
            subprocess.run(["git", "apply", str(PATCH_OUT)], cwd=tree,
                           check=True)
            for rel in PATHS:
                name = pathlib.Path(rel).name
                digests[name] = hashlib.sha256(
                    (tree / rel).read_bytes()).hexdigest()[:8]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    hunks = re.split(r"(?m)^(?=@@ )", patch)
    classified = []
    unclassified = 0
    for h in hunks[1:]:
        got = classify(h)
        if got is None:
            unclassified += 1
            classified.append({"header": h.splitlines()[0],
                               "class": "UNCLASSIFIED", "why": ""})
        else:
            classified.append({"header": h.splitlines()[0],
                               "class": got[0], "why": got[1]})

    counts: dict[str, int] = {}
    for c in classified:
        counts[c["class"]] = counts.get(c["class"], 0) + 1

    payload = {
        "experiment": "E151",
        "rung": "R1 re-application onto the frontier",
        "harness": "offline",
        "frontier_sha": FRONTIER,
        "base_sha": BASE,
        "our_sha": git("rev-parse", "HEAD").strip(),
        "paths": PATHS,
        "e151_r1_arm_is_self_contained": False,
        "e151_r1_arm_self_containment_note":
            "The R1 arm depends on the E147 retile plumbing in "
            "qmm_t_nax_tgp_impl (kHostBM/kHostBN, the compute_tile lambda and "
            "the grid-stride loop). The frontier has none of it, so the "
            "re-application patch must carry the plumbing as well as the arm. "
            "The arm cannot be made self-contained without duplicating that "
            "plumbing inside affine_qmm_t_nax, which would be strictly more "
            "code for the same generated kernel.",
        "e151_r1_frontier_patch_applies_clean": applies,
        "e151_r1_frontier_patch_reproduces_gate_proved_surface":
            digests == GATE_PROVED,
        "e151_r1_frontier_patch_result_digests": digests,
        "e151_r1_gate_proved_surface_digests": GATE_PROVED,
        "e151_r1_frontier_patch_bytes": len(patch.encode()),
        "e151_r1_frontier_patch_hunks": len(classified),
        "e151_r1_frontier_patch_hunk_classes": counts,
        "e151_r1_frontier_patch_unclassified_hunks": unclassified,
        "e151_r1_frontier_patch_hunk_table": classified,
        "e151_r1_reapplication_recipe":
            "Import the frontier, then `git checkout <our sha> -- "
            "Vendor/.../quantized_nax.h Vendor/.../quantized_nax.cpp`. That "
            "is equivalent to applying research/e151-r1-frontier-reapply.patch "
            "and it lands the exact surface the R1 gate chain proved.",
        "e151_r1_reapplication_risk":
            "The patch is larger than the arm. It also re-imports the arm-off "
            "E147 plumbing, which E147 rung E-1c measured at "
            "nax_arm_off_air_delta_bytes = 2112 against pristine and which "
            "also reaches affine_qmm_n_nax and affine_gather_qmm_rhs_nax "
            "through the shared impl and the RULE 145 guards. That plumbing "
            "is gate-proved and census-proved, but it is not zero, so "
            "`import then apply` is not `import plus one arm`.",
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"patch          {PATCH_OUT.relative_to(ROOT)} "
          f"({payload['e151_r1_frontier_patch_bytes']} bytes, "
          f"{len(classified)} hunks)")
    print(f"applies clean  {applies}")
    print(f"reproduces     {digests} "
          f"expected {GATE_PROVED} -> "
          f"{payload['e151_r1_frontier_patch_reproduces_gate_proved_surface']}")
    print(f"hunk classes   {counts} unclassified={unclassified}")
    print(f"self-contained {payload['e151_r1_arm_is_self_contained']}")
    return 0 if applies and unclassified == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
