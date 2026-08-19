#!/usr/bin/env bash
# Research-only (qwen38-r1-e42-psi-phi-by-injected-regression): drive one E42 arm.
#
#   research/e42-run.sh TAG [--curve] [--legs N]
#
# TAG names the arm (base, p2L1, p2L2, p6L1, p6L2, m1L1, base2). The twins must
# already hold that arm's source and be COMMITTED, so every timed leg records
# dirty=0 and the W&B/parity record can name an exact commit.
#
# Timing here is NOT gate-qualified: MLXFAST_LOCAL_COOL_GATE=0 is required on
# this ~43 C-idle host. meta.txt records that verbatim. Arms are compared only
# with the base brackets measured in the same session, and the effects under
# test (5-100 %) are far above the 0.311 % drift envelope.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

root="${E42_ROOT:-${repo_root}/.mlxfast-private/e42}"
tokens="${E42_TOKENS:-512}"
fixture="${E42_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
head_dir="${E42_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
curve_widths="${E42_CURVE_WIDTHS:-1,2,3,4,5,6,7,8,9}"
curve_reps="${E42_CURVE_REPS:-21}"
curve_inner="${E42_CURVE_INNER:-10}"

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

tag="${1:?usage: research/e42-run.sh TAG [--curve] [--legs N]}"; shift
want_curve=0
legs=2
while (($#)); do
  case "$1" in
    --curve) want_curve=1; shift ;;
    --legs) legs="${2:?--legs needs a count}"; shift 2 ;;
    *) echo "e42-run: unknown argument $1" >&2; exit 2 ;;
  esac
done

[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e42-run: declared head run tree missing at ${head_dir}" >&2; exit 2; }
[[ -s "${fixture}" ]] || { echo "e42-run: missing fixture ${fixture}" >&2; exit 2; }

dirty="$(git status --porcelain -- \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp | wc -l | tr -d ' ')"
if ((dirty)); then
  echo "e42-run: twins are uncommitted; commit the arm first so timed runs are reproducible" >&2
  exit 2
fi
python3 research/twin_audit.py quantized || exit 2

# The kernel ships as a source string inside a C++ translation unit, so a Metal
# syntax or template-arity error survives `swift build` and only surfaces as a
# JIT failure minutes into the run. quantized.metal carries the
# instantiate_quantized_* macros, so compiling it exercises every width case the
# JIT will build, for ~24s.
if ! xcrun -sdk macosx metal -std=metal3.1 -O2 -c \
    Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal \
    -I Vendor/mlx-swift/Source/Cmlx/mlx -o /dev/null 2>"${TMPDIR:-/tmp}/e42-metalcc.err"; then
  echo "e42-run: quantized.metal does not compile at this arm; not spending a run" >&2
  grep 'error:' "${TMPDIR:-/tmp}/e42-metalcc.err" | head -5 >&2
  exit 2
fi

out="${root}/runs/${tag}"
rm -rf "${out}"; mkdir -p "${out}/reports"

{
  echo "tag=${tag}"
  echo "head_sha=$(git rev-parse HEAD)"
  echo "base_sha=04ad6bf11437c269df85a47e91faa769c74fe6da"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "twin_digests=$(shasum -a 256 \
    Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
    Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp \
    | awk '{printf "%s ", $1}')"
  echo "fixture=${fixture}"
  echo "fixture_sha256=$(shasum -a 256 "${fixture}" | cut -d' ' -f1)"
  echo "tokens=${tokens}"
  echo "offered_depth=${MLXFAST_QWEN_MTP_DEPTH:-8}"
  echo "head_dir=${head_dir}"
  echo "head_safetensors_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "legs=${legs}"
  echo "curve=${want_curve}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "thermal_before=$(sample_thermal)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

status=0
if ((want_curve)); then
  echo "=== e42-run: cost curve ${tag} ==="
  # Rebuilds both build roots itself, which is also what makes the legs below
  # run against this arm's metallib.
  research/run-qmv-curve.sh "e42-${tag}" \
    --widths "${curve_widths}" --shapes-only \
    --reps "${curve_reps}" --inner "${curve_inner}" --skip-stock
  rc=$?
  echo "curve_exit=${rc}" >> "${out}/meta.txt"
  ((rc == 0)) || { echo "e42-run: curve exited ${rc}" >&2; status=1; }
fi

if ((status == 0)); then
  # `benchmark-qwen-mtp.sh --local-iterate` returns before benchmark.sh:1832's
  # build, and the kernel under test lives in mlx-generated/quantized.cpp -- a
  # C++ translation unit, not the metallib -- so the wrapper's metallib refresh
  # would leave both binaries holding the previous arm's kernel. Reuse
  # benchmark.sh's own two-root recipe: the scored binary is the .build-worker
  # twin, which a plain `swift build` does not touch.
  mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift || status=1
  CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker || status=1
  # Freshness, not existence: a silently-skipped rebuild would time the previous
  # arm and read as a null.
  for product in .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker; do
    stale="$(find Package.swift Package.resolved Sources Vendor -newer "${product}" -print -quit 2>/dev/null || true)"
    [[ -z "${stale}" ]] || {
      echo "e42-run: ${product} is older than ${stale}; refusing to time a stale binary" >&2
      status=1
    }
  done
fi

if ((status == 0)); then
  export MLXFAST_LOCAL_COOL_GATE=0
  export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
  export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${fixture}"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
  export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
  for ((i = 1; i <= legs; i++)); do
    export MLXFAST_SCORE_PATH="${out}/score-${i}.json"
    export MLXFAST_CAPTURE_DIR="${out}/reports/leg-${i}"
    mkdir -p "${MLXFAST_CAPTURE_DIR}"
    echo "=== e42-run: ${tag} leg pair ${i}/${legs} (${tokens} tokens) ==="
    ./benchmark-qwen-mtp.sh --local-iterate
    rc=$?
    {
      echo "leg${i}_exit=${rc}"
      echo "leg${i}_thermal_after=$(sample_thermal)"
      echo "leg${i}_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "${out}/meta.txt"
    ((rc == 0)) || { echo "e42-run: ${tag} leg ${i} exited ${rc}" >&2; status=1; break; }
  done
fi

{
  echo "cli_sha256=$(shasum -a 256 "${repo_root}/.build/release/mlxfast-swift" 2>/dev/null | cut -d' ' -f1)"
  echo "worker_sha256=$(shasum -a 256 "${repo_root}/.build-worker/release/mlxfast-runtime-worker" 2>/dev/null | cut -d' ' -f1)"
  echo "metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "thermal_after=$(sample_thermal)"
  echo "status=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"
exit "${status}"
