#!/usr/bin/env bash
# E124: run ONE leg of ./benchmark-qwen-mtp.sh under one island arm.
#
#   usage: research/e124_leg.sh TAG ARM TOKENS [e79_trace_leg.sh flags...]
#
#   ARM  all | none | q | kv, passed through DARKBLOOM_QWEN_MTP_ISLAND_ARM.
#
# The DARKBLOOM_ prefix is required, not stylistic.
# `sanitizedRuntimeWorkerEnvironment` forwards only DARKBLOOM_, DYLD_, LC_,
# METAL_, MLX_ and MTL_ into the runtime worker. An MLXFAST_-spelled selector
# never arrives, and every arm then runs the shipped default in silence.
#
# Every arm is the SAME worker binary under one environment variable, so no leg
# of this experiment needs a rebuild. Harness defect 25 attaches a thermal
# confound to the ARM whenever an arm change triggers a rebuild, because the
# position behind the rebuild gets a free cooling gap that the other position
# does not. This experiment removes that confound by construction: the gap in
# front of every leg is the same gap.
#
# The leg body is research/e79_trace_leg.sh, unchanged. This wrapper only sets
# the arm, asserts that the worker actually selected it, and records the arm in
# meta.txt so no leg can be attributed to the wrong arm after the fact.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e124_leg.sh TAG ARM TOKENS [flags...]}"
arm="${2:?usage: e124_leg.sh TAG ARM TOKENS [flags...]}"
tokens="${3:?usage: e124_leg.sh TAG ARM TOKENS [flags...]}"
shift 3

case "${arm}" in
  all|none|q|kv) ;;
  *) echo "e124_leg.sh: unknown arm '${arm}'" >&2; exit 2 ;;
esac

export DARKBLOOM_QWEN_MTP_ISLAND_ARM="${arm}"
research/e79_trace_leg.sh "${tag}" "${tokens}" "$@"
status=$?

out="research/out/${tag}"
witness="$(grep -h '^qwen-mtp-island-arm: ' "${out}/wrapper.err" 2>/dev/null \
  | sort -u | tr '\n' ';')"
{
  echo "experiment=e124-noislands-acceptance-exchange"
  echo "e124_arm=${arm}"
  echo "DARKBLOOM_QWEN_MTP_ISLAND_ARM=${arm}"
  echo "e124_arm_witness=${witness:-<absent>}"
} >> "${out}/meta.txt"

# A leg with no witness line ran a worker that does not carry the selector, so
# every arm in that session is silently `all`. That is the one failure mode
# that would make the whole experiment a null for the wrong reason.
if [[ "${status}" -eq 0 && -z "${witness}" ]]; then
  echo "e124_leg.sh: ${tag} exited 0 but printed no island-arm witness;" \
       "the worker predates the selector. Rebuild before timing." >&2
  exit 3
fi
if [[ "${status}" -eq 0 && "${witness}" != "qwen-mtp-island-arm: ${arm} "* ]]; then
  echo "e124_leg.sh: ${tag} requested arm ${arm} but the worker reported" \
       "'${witness}'" >&2
  exit 4
fi

exit "${status}"
