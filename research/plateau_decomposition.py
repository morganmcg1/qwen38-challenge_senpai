#!/usr/bin/env python3
"""Why has our official score been pinned near 3.2325?

CAUTION, READ THIS FIRST. The obvious version of this analysis is a tautology and
I nearly wrote it up as a triumph. If you define

    crown_gain   = main/base - 1
    overlay_cost = ours/base - 1

then main*(1-crown_gain)*(1+overlay_cost) reproduces `ours` to second order for
ANY three numbers whatsoever. The 0.004-sigma "agreement" I first computed is
algebra, not evidence, and proves nothing about any mechanism.

What is load-bearing here is NOT arithmetic. It is four git facts, each checkable
with one command. The scores only put a size on something the object store
already establishes on its own.

Run: research/plateau_decomposition.py
"""

MAIN = 3.24929398547457   # upstream/main = 0c90733 (ofou), "Accept submission 0cd0a6b4..."
BASE = 3.24326223889754   # 5068eb8d tree == companygardener; the base we submitted onto
OURS = 3.23250848263467   # our best official submission (2b0c36a0)
FKIENE = 3.24417896624589  # 1cb1f43a, the tree the organizer overlaid the crown ONTO
# 🔴 CORRECTED, ledger 166. This script compares SEPARATE BOARD SUBMISSIONS
# (main vs base vs ours), which is a BETWEEN-SUBMISSION question. It previously
# divided by 0.0978 %, a WITHIN-RUN replicate sd -- too small by 7.9x, so every
# "sigma_score" it printed was inflated by that factor. The retired value is
# kept below only so the inflation is visible in the output.
from noise_floors import SCORE_BETWEEN_SUBMISSION, SCORE_WITHIN_RUN

SIGMA_PCT = SCORE_BETWEEN_SUBMISSION.pct          # 0.7678 %, 17 whole-tree sets
SIGMA_PCT_RETIRED = SCORE_WITHIN_RUN.pct          # within-run; DO NOT divide by


def pct(x):
    return f"{x * 100:+.4f} %"


def main():
    crown = MAIN / BASE - 1.0
    ovl = OURS / BASE - 1.0

    print("=== the two measured differences (each ONE board measurement) ===")
    print(f"  crown hunks over our submit base : {pct(crown)}"
          f"   ({abs(crown) * 100 / SIGMA_PCT:.2f} sigma_score)")
    print(f"  our overlay over the same base   : {pct(ovl)}"
          f"   ({abs(ovl) * 100 / SIGMA_PCT:.2f} sigma_score)")
    print(f"  gap from us to live main         : {pct(MAIN / OURS - 1)}")
    print()
    print("  These two differences EXHAUST the gap by construction:")
    print(f"    (main/base) * (base/ours) = {(MAIN / BASE) * (BASE / OURS):.12f}")
    print(f"     main/ours                = {MAIN / OURS:.12f}")
    print("  That identity is not a finding, it is division. The finding is WHICH")
    print("  lines each factor corresponds to, and that we still hold one factor")
    print("  in reverted form in our own working tree.")
    print()

    print("=== the only number here with an independent second route ===")
    print("  Our overlay's cost is not only a score difference. The per-prompt legs")
    print("  say the same thing by themselves: MTP wide +0.3098 % (n=5, 5/5 slower,")
    print("  t=+3.69 on 4 df), MTP narrow +0.0157 % (n=3). Two routes, same sign,")
    print("  same size.")
    print("  The crown's +0.186 % has ONE route and sits at "
          f"{abs(crown) * 100 / SIGMA_PCT:.2f} sigma. Suggestive, not settled.")
    print()

    print("=== git fact 1: what a submission from HEAD does to live main ===")
    print("  Not a prediction. `git diff upstream/main HEAD` over editablePaths:")
    print("    Qwen35RuntimeWeights.swift        MLX_MAX_MB_PER_BUFFER 512 -> 128")
    print("    RuntimeStartupMemoryPolicy.swift  setenv overwrite 1 -> 0")
    print("    RuntimeStartupMemoryPolicy.swift  full profile 512/50 -> 320/128")
    print("  All three are the crown in reverted form, in files our submission")
    print("  packages: Sources/MLXFastModel is a DIRECTORY entry in editablePaths.")
    print()

    print("=== git fact 2: 'submit does a merge' is refuted, not assumed ===")
    print(f"  fkiene 1cb1f43a scored {FKIENE:.11f} ({pct(FKIENE / BASE - 1)} over base)")
    print("  by adding a 19-line verify-concat JIT warm to Qwen36MTPBlockSession.swift.")
    print("  ofou branched from 5068eb8d, which PREDATES fkiene, and never opened that")
    print("  file: `git diff 5068eb8d ef42e043` is 2 files, +11/-5, memory policy only.")
    print("  Yet `git diff 1cb1f43a 0c90733` DELETES all 19 of fkiene's lines.")
    print("  A three-way merge keeps a hunk the author never touched. This did not.")
    print("  => submit replaces whole files. Every stale file we package reverts")
    print("     main's copy of it, including regions we have never read.")
    print()

    print("=== git fact 3: the deletion was collateral, not a mechanism ===")
    print("  Rival notes on the board credit ofou with deliberately deleting the")
    print("  concat warm and reason forward from that ('the crown just deleted it,")
    print("  do not restore it'). The diff of ofou's tree against ofou's own base")
    print("  contains no such deletion. They are reading an artifact of the overlay")
    print("  system as an authored decision. We made the same class of error in")
    print("  ledger 160 when we credited ourselves with a frontier advance.")
    print()

    print("=== git fact 4: our own overlay already did this once ===")
    print("  Our scored overlay deleted 17 lines of quantized.h that we never wrote,")
    print("  including a 12-line frontier comment. Same mechanism, already observed,")
    print("  and we filed it as a curiosity instead of a recurring cost.")
    print()

    print("=== consequences, ordered by cost to us ===")
    print(f"  1. Rebase onto main, ship nothing of ours : ~{MAIN:.5f}")
    print(f"  2. Rebase and keep our overlay            : ~{MAIN * (1 + ovl):.5f}")
    print(f"  3. Submit HEAD as it stands               : ~{OURS:.5f}  (again)")
    print("  (1) and (2) are model predictions carrying about one sigma_score of")
    print("  slack each. (3) is not a prediction; it is what we have already done.")
    print()
    print("  The three hunks are 11 added and 5 deleted lines. Nothing in this")
    print("  decomposition requires K-tiling, a register ceiling, psi, or rho.")
    print()
    print("  Standing order: research/frontier-revert-gate.sh must pass before any")
    print("  submission, and it fails closed on exactly the reverts listed above.")


if __name__ == "__main__":
    main()
