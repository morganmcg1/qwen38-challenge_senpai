#!/usr/bin/env bash
# research/e42-parity.sh GROUP ARM=SHA [ARM=SHA ...]
#
# Runs the 192-cell cross-build parity rig for one group of arms into a
# group-private output directory, because run-qmv-parity.sh wipes its output
# directory on entry and a second group would otherwise destroy the first
# group's digests before they could be archived.
#
# The comparator never exits non-zero, so the verdict line is the result: grep
# for `verdict: BIT-IDENTICAL`.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

group="${1:?usage: e42-parity.sh GROUP ARM=SHA [ARM=SHA ...]}"
shift
[[ $# -ge 2 ]] || { echo "e42-parity.sh: need a reference arm and at least one comparison arm" >&2; exit 1; }

out_dir="${repo_root}/.mlxfast-private/e42/parity/${group}"
export MLXFAST_QMV_PARITY_DIR="${out_dir}"
mkdir -p .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache"

echo "e42-parity: group=${group} arms=$*"
research/run-qmv-parity.sh "$@" 2>&1 | tee "${repo_root}/.mlxfast-private/e42/parity-${group}.log"

# The rig reports every arm against the first; a single differing cell is a hard
# stop for the whole experiment, so make that machine-checkable here rather than
# leaving it to a human reading a wall of output.
verdicts="$(grep -c '^verdict: BIT-IDENTICAL' "${repo_root}/.mlxfast-private/e42/parity-${group}.log" || true)"
diverges="$(grep -c '^verdict: DIVERGES' "${repo_root}/.mlxfast-private/e42/parity-${group}.log" || true)"
echo "e42-parity: group=${group} bit_identical=${verdicts} diverges=${diverges}"
[[ "${diverges}" == "0" ]] || { echo "e42-parity: HARD STOP, group ${group} has a differing cell" >&2; exit 1; }
