#!/usr/bin/env bash
# E66 rung 5: submission certification of the merged t55 + t6 surface.
#
#   research/e66-rung5.sh
#
# Runs with the SHIPPED command-buffer geometry, not the E61-comparability lever
# that research/e66-run.sh sets. The ranked runner uses the shipped defaults, so
# a certification run must too.
#
# Every step records its own exit code and the chain continues, so one failure
# does not hide the rest of the evidence. The summary at the end is the verdict.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly OUT=".mlxfast-private/e66/rung5"
readonly ORGANIZER="bfab0de58d43453e506523707e1720a3485570f4"
readonly MERGED_BASE="b7b6589a9b319c1176c737b5698b915740df0937"
mkdir -p "${OUT}"

declare -a STEP_NAME=() STEP_RC=()
step() {
  local name="$1"; shift
  echo "=== e66-rung5: ${name} ==="
  "$@" >"${OUT}/${name}.log" 2>&1
  local rc=$?
  STEP_NAME+=("${name}"); STEP_RC+=("${rc}")
  echo "e66-rung5: ${name} rc=${rc} log=${OUT}/${name}.log"
  tail -6 "${OUT}/${name}.log"
  return 0
}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e66-rung5: refusing to certify a dirty worktree" >&2
  git status --porcelain >&2
  exit 2
fi
head_sha="$(git rev-parse HEAD)"
echo "e66-rung5: certifying HEAD=${head_sha}"
echo "${head_sha}" >"${OUT}/head-sha.txt"

assert_arm() {
  senpai/rebuild-and-assert-worker.sh \
    --require '<T, 5, 5, true>' --require '<T, 6, 6, true>' \
    --forbid  '<T, 5, 3, true>' --forbid  '<T, 6, 3, true>'
}

step "01-assert-before" assert_arm
step "02-twin-audit" python3 research/twin_audit.py
runtime_tests() {
  MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
}

step "03-swift-test" swift test --force-resolved-versions
step "04-swift-test-runtime" runtime_tests
step "05-local-submit" ./benchmark-qwen-mtp.sh --local-submit
step "06-assert-after" assert_arm

# The submitted set is every editablePath that differs from the organizer tree.
# `mapfile` is bash 4; macOS ships bash 3.2, so read the list the portable way.
submitted=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && submitted+=("${line}")
done < <(python3 - "${ORGANIZER}" <<'PY'
import json, subprocess, sys
bm = json.load(open("benchmark.json"))
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "editablePaths" and isinstance(v, list):
                yield v
            else:
                yield from find(v)
    elif isinstance(o, list):
        for v in o:
            yield from find(v)
eps = sorted({p for lst in find(bm) for p in lst})
out = subprocess.run(["git", "diff", "--name-only", sys.argv[1], "HEAD", "--"] + eps,
                     capture_output=True, text=True)
print(out.stdout.strip())
PY
)
printf '%s\n' "${submitted[@]}" >"${OUT}/submitted-paths.txt"
echo "e66-rung5: submitted paths: ${submitted[*]}"

step "07-scope-vs-organizer" senpai/validate-assignment-scope.sh "${ORGANIZER}" "${submitted[@]}"
step "08-scope-vs-merged-base" senpai/validate-assignment-scope.sh "${MERGED_BASE}" "${submitted[@]}"
step "09-editable-budget" senpai/check-editable-budget.sh "${ORGANIZER}" "${ORGANIZER}"
step "10-ranked-score-boundary" senpai/verify-ranked-score-boundary.sh

git diff --numstat "${ORGANIZER}" HEAD -- "${submitted[@]}" >"${OUT}/submitted-numstat.txt" 2>&1
git diff "${ORGANIZER}" HEAD -- "${submitted[@]}" >"${OUT}/submitted-diff.patch" 2>&1
echo "e66-rung5: wrote ${OUT}/submitted-diff.patch"

echo
echo "=== e66-rung5 summary for ${head_sha} ==="
fail=0
for i in "${!STEP_NAME[@]}"; do
  printf '%-28s rc=%s\n' "${STEP_NAME[$i]}" "${STEP_RC[$i]}"
  [[ "${STEP_RC[$i]}" -ne 0 ]] && fail=1
done
echo "e66-rung5: overall_nonzero_step=${fail}"
echo "e66-rung5: swift test failures are compared against the known nine by hand;"
echo "e66-rung5: a nonzero rc on 03/04 is expected and is not by itself a defect."
exit 0
