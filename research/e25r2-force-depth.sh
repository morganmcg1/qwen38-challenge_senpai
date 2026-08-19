#!/usr/bin/env bash
# Research-only (qwen38-r1-e25-per-row-draft-price, r2): turn the base blob's
# adaptive schedule into a DETERMINISTIC INTERLEAVED DEPTH CYCLE, so one 512-
# token leg measures the round-time curve T(d) and the per-position acceptance
# p_d at EVERY depth 0..7 with a real sample size.
#
# WHY A CYCLE AND NOT ONE LEG PER DEPTH.  r1 fitted its step ratios from the
# depths the adaptive rule happened to choose, which gave n=167 at d=4 and n=9
# at d=5 -- and those samples are not exchangeable, because the rule picks deep
# rounds exactly where the head is hot and the KV cache is short.  Depth was
# confounded with position and with prompt difficulty.  Running one FIXED depth
# per leg would fix the sample size but not the confound: the legs would then
# differ in thermal state and in cache-length trajectory.  Cycling depth
# round-by-round inside a single leg makes position, temperature, prompt and
# cache length COMMON MODE BY CONSTRUCTION -- every depth sees the same
# positions, in the same leg, interleaved 8 rounds apart.
#
# The cycle is the identity permutation over 0..7, so depth d is taken on
# rounds d, d+8, d+16, ...  A 512-token window at a mean of ~3 accepted tokens
# per round is ~170 rounds, hence ~21 rounds per depth per prompt and ~170 per
# depth over the 8-prompt set.  That is the n the refit needs.
#
# WHAT STAYS HONEST.  The forced depth replaces only the RETURN of
# `costModelDepth`; the price walk still runs and still traces, so each round
# records both the depth actually taken and the depth the shipped rule WOULD
# have taken.  Rows are declared from the depth actually proposed, the target
# still verifies every row, and every emitted token is still checked against the
# serial trajectory by the trusted parent.  This is an instrument, never a
# candidate: it is not in `editablePaths` and it is never submitted.
#
# usage: research/e25r2-force-depth.sh <src-file>
set -euo pipefail

src="${1:?usage: research/e25r2-force-depth.sh <src-file>}"

python3 - "${src}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()

# 1. The width cap must not clip the forced depth. The base opens width 6..9
#    only on a full-accept streak; the instrument needs every width on every
#    round, and the source records widths 6..8 as measured bit-exact.
old_cap = """        let widthCap = fullAcceptStreak >= Self.segmentedStreakGate
            ? Self.segmentedVerifyDepthCap
            : Self.sdpaWidthWallDepthCap"""
new_cap = """        // INSTRUMENT: the streak gate is removed so the forced depth is never
        // clipped. Widths 6..8 are the ones the base records as measured
        // bit-exact; the cycle below stops at depth 7 for that reason.
        let widthCap = Self.segmentedVerifyDepthCap"""
assert text.count(old_cap) == 1, "width cap anchor not unique"
text = text.replace(old_cap, new_cap)

# 2. Force the return, but only AFTER the walk has run and traced. `depth` is
#    the shipped rule's counterfactual choice; it is recorded in the trace so
#    the same leg measures the policy it replaces.
old_ret = """            expected += reach
            depth += 1
        }
        return depth
    }"""
new_ret = """            expected += reach
            depth += 1
        }
        // INSTRUMENT: `depth` is now the counterfactual SHIPPED choice, already
        // written into `scheduleTrace` by the walk above. The round takes the
        // cycle's depth instead, so T(d) and p_d are measured at every depth
        // with position, prompt, cache length and temperature common-mode.
        let forced = Self.forcedDepthCycle[
            (roundCount - 1) % Self.forcedDepthCycle.count]
        if Self.traceRounds {
            scheduleTrace += String(
                format: "shipped=%d;forced=%d;", depth, forced)
        }
        return Swift.min(forced, cap)
    }

    /// INSTRUMENT ONLY. Identity permutation over the widths the base records
    /// as measured bit-exact (depths 0..7 = widths 1..8); depth 8 (width 9) is
    /// excluded because it was never verified exact.
    private static let forcedDepthCycle: [Int] = [0, 1, 2, 3, 4, 5, 6, 7]"""
assert text.count(old_ret) == 1, "return anchor not unique"
text = text.replace(old_ret, new_ret)

open(path, "w").write(text)
print("e25r2-force-depth: patched", path)
PY

# Fail loudly rather than silently timing the base: the instrument is only
# useful if all three of its marks are present in the file the compiler reads.
for mark in \
  'private static let forcedDepthCycle: [Int] = [0, 1, 2, 3, 4, 5, 6, 7]' \
  'let widthCap = Self.segmentedVerifyDepthCap' \
  'return Swift.min(forced, cap)'
do
  grep -qF -- "${mark}" "${src}" || {
    echo "e25r2-force-depth: mark missing after patch: ${mark}" >&2
    exit 2; }
done
grep -qF -- 'fullAcceptStreak >= Self.segmentedStreakGate' "${src}" && {
  echo "e25r2-force-depth: streak gate still live in the width cap" >&2
  exit 2; }
echo "e25r2-force-depth: ok"
