#!/usr/bin/env bash
# E129 -- the 512-token row-evidence gate for per-width entry-point templating
# and for the one-pass width table.
#
#   usage: research/e129_exact512.sh [--pin SHA256] [--legs LIST]
#
# WHAT IS CLAIMED. Neither arm may move one bit of the target model's row
# evidence.
#
#   templating   `MLX_E120_QMV_ENTRY` picks how many Metal entry points carry
#                the width switch. Both layouts emit the same case bodies from
#                the same generator, so only the compiler's register ceiling
#                moves.
#   one pass     `MLX_E120_QMV_TABLE` picks `IPG` and `RPS` per width. `IPG` is
#                how many INPUT rows one threadgroup accumulates and `RPS` is
#                how many OUTPUT rows one simdgroup accumulates. Neither
#                changes the k range an output row walks, nor the order it
#                walks it in, so every dot product is summed in the same order
#                by a different thread.
#
# Both claims are bit exactness, so the check is over `mtp-row:` lines whose
# top-two values are hex float literals. An argmax match would pass on a
# candidate that moved a low bit, and the trusted parent checks the values.
#
# WHY 512 AND NOT 64. The seed is 512 tokens, so only a 512-token window walks
# the key length past 1024 and exercises the boundary the row ledger closes
# over. A 64-token leg never reaches it. The window also has to run past EOS,
# which a short leg can miss entirely.
#
# THE CONTROLS. `research/e116_row_digest_check.py` runs a value control and an
# order control on the rows themselves. The runtime control has to change the
# decode window, because no compliant candidate knob may move this digest: a
# `mtp-row:` line is emitted once per emitted token position for each of the
# wrapper's two passes, so a 512-token leg carries 1025 rows and a 128-token
# leg carries 257. The 128-token leg's digest MUST NOT match the pin.
#
# WARMUP COVERAGE. Every leg writes `MLX_E120_QMV_PIPELINE_LOG`. Templating
# turns 2 instantiated pipelines into 4, and the one-pass table into 7, so a
# leg that never instantiates a pipeline would pay a JIT compile inside the
# timed window. The log records the dispatch ordinal at which each pipeline and
# each width was first seen, which is what proves warmup reached them.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Pinned by E121 rung 3, several campaign bases before this experiment existed.
# Re-pin only from a clean build of the recorded BASE_SHA, never from a
# candidate leg.
pin=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e
legs="base,tier,onepass,neg"

while (($#)); do
  case "$1" in
    --pin) pin="$2"; shift 2 ;;
    --legs) legs="$2"; shift 2 ;;
    *) echo "e129_exact512.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e129_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

out=research/e129-artifacts
mkdir -p "${out}"
failures=0
declare -a exact_tags=()

run_leg() {
  local tag="$1" tokens="$2" entry="$3" table="$4"
  echo
  echo "=== ${tag}: entry=${entry} table=${table} tokens=${tokens} ==="
  (
    export MLX_E120_QMV_ENTRY="${entry}"
    export MLX_E120_QMV_TABLE="${table}"
    export MLX_E120_QMV_PIPELINE_LOG="${PWD}/research/out/${tag}-pipelines.json"
    rm -f "${MLX_E120_QMV_PIPELINE_LOG}"
    research/e79_trace_leg.sh "${tag}" "${tokens}"
  ) || { echo "e129_exact512: leg ${tag} failed" >&2; failures=$((failures + 1)); }
}

case ",${legs}," in *,base,*)
  run_leg e129x512base 512 shared_switch shipped
  exact_tags+=(e129x512base) ;;
esac
case ",${legs}," in *,tier,*)
  run_leg e129x512tier 512 tiered_switch shipped
  exact_tags+=(e129x512tier) ;;
esac
case ",${legs}," in *,onepass,*)
  run_leg e129x512onep 512 tiered_switch onepass
  exact_tags+=(e129x512onep) ;;
esac
case ",${legs}," in *,neg,*)
  run_leg e129x128neg 128 tiered_switch shipped ;;
esac

echo
neg=()
[[ ",${legs}," == *,neg,* ]] && neg=(--negative-control e129x128neg)
python3 research/e116_row_digest_check.py "${exact_tags[@]}" \
  --pin "${pin}" \
  --expect-rows 1025 \
  "${neg[@]}" \
  --json "${out}/row-digest-512.json" \
  || failures=$((failures + 1))

echo
echo "--- wrapper verdicts ---"
for tag in "${exact_tags[@]}" e129x128neg; do
  [[ -s "research/out/${tag}/wrapper.out" ]] || continue
  echo "${tag}: $(grep -oE 'all_tokens_matched[": ]*[a-z]+' \
    "research/out/${tag}/wrapper.out" 2>/dev/null | tail -1)"
done

echo
echo "--- pipeline coverage ---"
python3 research/e129_pipeline_coverage.py \
  --json "${out}/pipeline-coverage.json" \
  research/out/*-pipelines.json \
  || failures=$((failures + 1))

echo
echo "e129_exact512: failures=${failures}"
exit $((failures > 0))
