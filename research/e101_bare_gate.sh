#!/usr/bin/env bash
# E101 compiled-default gate.
#
#   usage: research/e101_bare_gate.sh
#
# The ranked worker exports no environment, so a mechanism that only runs
# under an exported variable scores zero. This gate proves three things about
# `MLX_E101_ROW_TOP32` on the built worker:
#
#   1. UNSET REACHES THE FUSED KERNELS. Two legs run with the variable absent
#      from the environment. Each round trace must read `sel_env=unset` with
#      `sel_fused` rising and `sel_argpart` at zero.
#   2. THE LEDGER CLOSES ON THOSE LEGS. 32 decode tokens first as a smoke, then
#      512, with all_tokens_matched true and residual_divergence_count zero.
#   3. MALFORMED VALUES FAIL CLOSED. One leg exports a value that is neither 0
#      nor 1 and must exit non-zero rather than resolve to either arm.
#
# The gate refuses to run if the variable is already exported, because that
# would make case 1 vacuous.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "${MLX_E101_ROW_TOP32:-}" ]]; then
  echo "e101_bare_gate: MLX_E101_ROW_TOP32 is exported; unset it first" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "e101_bare_gate: worktree is dirty" >&2
  exit 1
fi

failures=0
witness() {
  local tag="$1"
  local line
  line="$(grep -o 'sel_env=[^ ]* sel_fused=[0-9]* sel_argpart=[0-9]*' \
    "research/out/${tag}/trace.txt" | tail -1)"
  echo "${tag} witness: ${line:-<none>}"
  case "${line}" in
    "sel_env=unset sel_fused="*" sel_argpart=0")
      [[ "${line}" == *"sel_fused=0 "* ]] && {
        echo "e101_bare_gate: ${tag} never entered the fused path" >&2
        return 1
      }
      return 0
      ;;
    *)
      echo "e101_bare_gate: ${tag} witness is wrong" >&2
      return 1
      ;;
  esac
}

for tokens in 32 512; do
  tag="e101bare${tokens}"
  echo "=== ${tag}: nothing exported, ${tokens} decode tokens ==="
  research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  ((status == 0)) || { echo "e101_bare_gate: ${tag} exited ${status}" >&2
                       failures=$((failures + 1)); continue; }
  witness "${tag}" || failures=$((failures + 1))
  python3 - "${tag}" <<'PY' || failures=$((failures + 1))
import json, re, sys
tag = sys.argv[1]
m = json.load(open(f"research/out/{tag}/score.json"))["metrics"]
rows = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(-?\d+)", line))
        for line in open(f"research/out/{tag}/trace.txt")
        if line.startswith("mtp-trace: round=")]
n = len(rows)
drafts = sum(r["d"] for r in rows)
acc = sum(r["acc"] for r in rows)
ok = (m["all_tokens_matched"] and m["residual_divergence_count"] == 0
      and acc + n == m["decode_tokens"] + 1
      and abs(acc / drafts - m["accepted_draft_rate"]) < 1e-12
      and abs(drafts / n - m["effective_mean_draft_len"]) < 1e-9)
print(f"{tag} ledger: rounds {n} target_rows {drafts + n} emitted {acc + n} "
      f"drafts {drafts} accepted {acc} rejected {drafts - acc} "
      f"matched {m['all_tokens_matched']} "
      f"divergence {m['residual_divergence_count']} "
      f"closes {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
done

echo "=== e101malformed: MLX_E101_ROW_TOP32=maybe must fail closed ==="
MLX_E101_ROW_TOP32=maybe research/e79_trace_leg.sh e101malformed 32
status=$?
if ((status == 0)); then
  echo "e101_bare_gate: a malformed value did NOT fail closed" >&2
  failures=$((failures + 1))
else
  echo "e101malformed: exited ${status} as required"
fi

echo "e101_bare_gate: ${failures} failures"
exit "${failures}"
