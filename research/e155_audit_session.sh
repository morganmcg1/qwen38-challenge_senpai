#!/usr/bin/env bash
# E155 -- run the audit-ON legs, one prompt per worker process.
#
#   usage: research/e155_audit_session.sh PROMPT_ID [PROMPT_ID ...]
#
# The audit writes one JSONL line per proposal slot, so each leg needs its own
# output path. `research/e128_session.sh` takes several prompts in one call and
# would share a single path between them, which is why this wrapper drives it
# once per prompt instead.
#
# NOT A TIMING SESSION. The audit adds a full-vocabulary lm_head projection and
# two reductions per proposal slot AFTER the round has committed. Those extra
# dispatches sit inside the measured block, so every audit-ON leg records
# `timing_valid=false` and no timing number may be read from it. The audit
# cannot change what the round emitted: it runs after the accept walk, writes
# to a file, and returns nothing to the session.
#
# Needs research/e155-patches/recall-audit.patch applied and the worker
# rebuilt with senpai/rebuild-and-assert-worker.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($#)) || { echo "usage: research/e155_audit_session.sh PROMPT_ID [...]" >&2; exit 2; }

audit_dir="${E155_AUDIT_DIR:-.mlxfast-private/e155}"
mkdir -p "${audit_dir}"
status=0

for id in "$@"; do
  path="${PWD}/${audit_dir}/${id}.jsonl"
  rm -f "${path}"
  echo "=== e155: audit-on leg ${id} -> ${path} ==="
  env MLX_E155_RECALL_AUDIT=1 \
      MLX_E155_RECALL_AUDIT_PATH="${path}" \
      E128_FORCE=1 \
      E128_TOKENS="${E128_TOKENS:-512}" \
      E128_DEPTH="${E128_DEPTH:-8}" \
      E128_RUNS_DIR="${E128_RUNS_DIR:-runs-e155-on}" \
      research/e128_session.sh "${id}" || status=1
  if [[ -s "${path}" ]]; then
    echo "e155: ${id}: $(wc -l < "${path}" | tr -d ' ') audited slots"
  else
    echo "e155: ${id}: NO AUDIT ROWS" >&2
    status=1
  fi
done

exit "${status}"
