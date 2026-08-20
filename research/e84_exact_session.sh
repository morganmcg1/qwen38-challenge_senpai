#!/usr/bin/env bash
# The E84 exactness stage, end to end.
#
#   research/e84_exact_session.sh [--tokens N]
#
# 1. One base leg at N tokens. Its `mtp-verify --generate` pass writes the
#    reference rows, so the golden every arm is judged against comes from code
#    that carries neither mechanism. The leg runs `--hot`: this host's GPU
#    asymptotes near 40.5 C and never reaches the 40 C gate, and the leg exists
#    only to produce reference rows, so no number it reports is used as timing.
# 2. One untimed 512-token `mtp-verify --golden` pass per arm against that one
#    golden, which reports exact tokens, post-EOS continuation and row-ledger
#    closure.
# 3. One positive control: the tip arm against a golden with one reference
#    token changed. Four passing arms prove nothing unless the gate that
#    passed them is able to fail.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tokens=512
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e84_exact_session: unknown argument $1" >&2; exit 2 ;;
  esac
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E84_BASE_SHA="${E84_BASE_SHA:-07c75a708c2347021d3148d7bc87b246ba2aec73}"
export E84_ROOT="${E84_ROOT:-${repo_root}/.mlxfast-private/e84}"
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

golden="${E84_ROOT}/runs/exact-base-${tokens}/reports/02-mtp-verify-output.json"

if [[ ! -s "${golden}" ]]; then
  echo "e84_exact_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) golden leg (base) ===" >&2
  research/e84_run_leg.sh base "exact-base-${tokens}" --tokens "${tokens}" --hot || {
    echo "e84_exact_session: the base golden leg failed" >&2
    exit 1
  }
fi
[[ -s "${golden}" ]] || {
  echo "e84_exact_session: no golden at ${golden}" >&2; exit 1; }

export E84_GOLDEN="${golden}"
failed=0
for arm in base a b ab; do
  echo "e84_exact_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) ledger ${arm} ===" >&2
  research/e84_ledger_run.sh "${arm}" --tokens "${tokens}" || {
    failed=$((failed + 1))
    echo "e84_exact_session: ledger ${arm} failed; continuing" >&2
  }
done

# Positive control. Without it, four passing arms are equally consistent with
# "the mechanisms are exact" and "the gate cannot fail". One reference token is
# changed in the middle of the window and the tip arm is judged against that
# corrupted golden; the control passes only when the gate rejects it.
mutated="${E84_ROOT}/golden-mutated-row256.json"
python3 - "${golden}" "${mutated}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
g = json.load(open(src))
i = 256
old = g["rows"][i]["sequential_argmax"]
new = old + 1
g["rows"][i]["sequential_argmax"] = new
g["rows"][i]["top2_tokens"][0] = new
g["emitted_tokens"][i] = new
json.dump(g, open(dst, "w"))
print("e84_exact_session: mutated golden row %d token %d -> %d" % (i, old, new))
PY

echo "e84_exact_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) positive control ===" >&2
E84_GOLDEN="${mutated}" \
E84_LEDGER_LABEL="ab-mutated-golden" \
E84_EXPECT_MISMATCH=1 \
  research/e84_ledger_run.sh ab --tokens "${tokens}" || {
  failed=$((failed + 1))
  echo "e84_exact_session: the positive control failed; the gate may be vacuous" >&2
}

echo "e84_exact_session: done, ${failed} failed arm(s)" >&2
exit 0
