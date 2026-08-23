#!/usr/bin/env bash
# E162 correctness evidence: the full Swift suite, then the opt-in MLX runtime
# suite that actually executes kernels. Arm A is the correctness harness for the
# double-buffer transformation, so this is the gate that has to be green before
# arm B ships blind to the ranked M5.
#
# Cmlx searches for mlx.metallib next to the RUNNING executable, so the xctest
# bundle needs its own copy. Without it the suite aborts part way through with
# "Failed to load the default metallib". Build the tests, publish the metallib
# into every build root, then run. Recipe from research/e70_run_arch_probe.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/out/e162/tests"
rm -rf "${out}"
mkdir -p "${out}"

echo "head=$(git rev-parse HEAD)" > "${out}/meta.txt"
echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')" >> "${out}/meta.txt"

echo "=== 1/3 build tests + publish mlx.metallib to every build root ==="
if ! { swift build --build-tests --force-resolved-versions \
  && tools/build-mlx-metallib.sh --all-build-roots ; } > "${out}/build.log" 2>&1
then
  echo "e162_tests: BUILD FAILED, log=${out}/build.log"
  echo "build_failed=1" >> "${out}/meta.txt"
  tail -40 "${out}/build.log"
  exit 1
fi
echo "build_failed=0" >> "${out}/meta.txt"
echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)" \
  >> "${out}/meta.txt"

echo "=== 2/3 swift test ==="
swift test --force-resolved-versions > "${out}/plain.log" 2>&1
plain=$?
echo "plain_exit=${plain}" >> "${out}/meta.txt"

echo "=== 3/3 swift test with MLXFAST_RUN_MLX_RUNTIME_TESTS=1 ==="
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
  > "${out}/runtime.log" 2>&1
runtime=$?
echo "runtime_exit=${runtime}" >> "${out}/meta.txt"

for log in plain runtime; do
  f="${out}/${log}.log"
  echo "${log}_tests_started=$(grep -c '◇ Test .* started' "${f}")" >> "${out}/meta.txt"
  echo "${log}_issue_lines=$(grep -c 'recorded an issue' "${f}")" >> "${out}/meta.txt"
  echo "${log}_metallib_aborts=$(grep -c 'Failed to load the default metallib' "${f}")" \
    >> "${out}/meta.txt"
  grep -oE '✘ Test [A-Za-z0-9_]+\(\)' "${f}" | sort | uniq -c | sort -rn \
    > "${out}/${log}.failures.txt"
  grep -E "^Test run with [0-9]+ tests" "${f}" | tail -2 >> "${out}/meta.txt"
done

echo "=== meta ==="
cat "${out}/meta.txt"
echo
echo "=== plain failures ==="
cat "${out}/plain.failures.txt"
echo "=== runtime failures ==="
cat "${out}/runtime.failures.txt"
