#!/usr/bin/env bash
# E158 R1 -- one recall-audit pass per HEAD over the same prompts.
#
#   usage: research/e158_r1_session.sh declared|pinned PROMPT_ID [PROMPT_ID ...]
#
# `declared` is the head `mtp-head.manifest.json` names and the ranked
# candidate leg drafts with. `pinned` is the organizer's bf16 head, which is
# also what an unmodified local harness loads.
#
# NOT A TIMING SESSION. The audit adds a full-vocabulary projection and two
# reductions per proposal slot, and the two arms read different weight volumes
# per draft step by construction. Every leg records `timing_valid=false`.
#
# Needs research/e155-patches/recall-audit.patch applied and the worker rebuilt
# with senpai/rebuild-and-assert-worker.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:-}"
shift || true
(($#)) || {
  echo "usage: research/e158_r1_session.sh declared|pinned PROMPT_ID [...]" >&2
  exit 2
}

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
case "${arm}" in
  declared) head_dir="${cache}/mtp-head-declared-run" ;;
  pinned) head_dir="${cache}/mtp-head" ;;
  *)
    echo "e158_r1_session: arm must be 'declared' or 'pinned', got '${arm}'" >&2
    exit 2
    ;;
esac

export E128_HEAD_DIR="${head_dir}"
export E155_AUDIT_DIR="${E158_AUDIT_DIR:-.mlxfast-private/e158/${arm}}"
export E128_RUNS_DIR="${E158_RUNS_DIR:-runs-e158-${arm}}"
export E128_TOKENS="${E128_TOKENS:-512}"
export E128_DEPTH="${E128_DEPTH:-8}"

echo "e158_r1_session: arm=${arm} head_dir=${head_dir}"
echo "e158_r1_session: tokens=${E128_TOKENS} depth=${E128_DEPTH}"
exec research/e155_audit_session.sh "$@"
