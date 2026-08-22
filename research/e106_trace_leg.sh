#!/usr/bin/env bash
# E106 rung 0b: one census leg with the per-dispatch trace on.
#
#   usage: research/e106_trace_leg.sh TAG DRAFTS TOKENS
#
# `research/e96_census_leg.sh` pins one dispatch per command buffer, which
# makes each buffer interval one dispatch's exclusive GPU time. That is still
# aggregated per kernel signature, and the three N=5120 projections share one
# signature: `gdn.out_proj` (K=6144), `fa.o_proj` (K=6144) and `mlp.down`
# (K=17408) differ only in the k-loop trip count, which no grid dimension
# carries. K is the variable that separates the E106 hypotheses, so this leg
# adds `MLX_E58_DISPATCH_TRACE=1` to record the encode ordinal beside every
# single-dispatch buffer. The ordinal names the layer and therefore the tensor.
#
# A census leg is NEVER a timing leg. run_job takes an argv list with no
# environment field, so this wrapper exists only to set that one variable.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e106_trace_leg.sh TAG DRAFTS TOKENS}"
drafts="${2:?usage: e106_trace_leg.sh TAG DRAFTS TOKENS}"
tokens="${3:?usage: e106_trace_leg.sh TAG DRAFTS TOKENS}"

export MLX_E58_DISPATCH_TRACE=1
research/e96_census_leg.sh "${tag}" "${drafts}" "${tokens}" 0
status=$?

echo "dispatch_trace=1" >> "research/out/${tag}/meta.txt"
exit "${status}"
