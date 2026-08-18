#!/usr/bin/env bash
# Research-only (qwen38-r1-e20): run one or more --local-iterate arms from
# prebuilt, hash-pinned binary pairs, with per-family attribution captured.
#
#   research/e20-run.sh LABEL=BUILD:MODE:DEPTH [...]
#
#     BUILD  INSTR | BASE          (directory under .mlxfast-private/e20/bins)
#     MODE   0 | 1 | 2             (MLX_QWEN_ATTRIB)
#     DEPTH  offered draft depth   (--mtp-depth via MLXFAST_QWEN_MTP_DEPTH)
#
# e.g. research/e20-run.sh A1=INSTR:1:8 B1=INSTR:2:8 B2=INSTR:2:8 A2=INSTR:1:8
#      research/e20-run.sh N1=INSTR:0:8 N0=BASE:0:8
#
# The label names the MEASUREMENT and the triple names the BUILD+CONFIG, so a
# repeated arm is expressed without pretending it is a different build.
#
# Every gate benchmark-qwen-mtp.sh owns (drift tripwire, orphan scan, run lock,
# report seals) runs unmodified. The 40C cool gate is the one exception and it
# is disabled deliberately: idle GPU on this host sits at ~42.9C, above
# COOL_GATE_TEMP_C=40, so the real gate can never be satisfied. Entry and exit
# temperatures are sampled per arm instead, and the resulting numbers are
# reported with cool_gate_passed_real_gate=false / gate_qualified_for_timing=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repo_root="${PWD}"
bins_root="${E20_BINS_ROOT:-${repo_root}/.mlxfast-private/e20/bins}"
runs_root="${E20_RUNS_ROOT:-${repo_root}/.mlxfast-private/e20/runs}"
head_dir="${E20_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head}"
tokens="${E20_TOKENS:-256}"

# Leaked overrides would silently answer a different question under the same
# label, and the attribution names are exactly the ones under test.
for v in $(env | sed -n 's/^\(MLX_QWEN_[A-Z_]*\)=.*/\1/p'); do unset "${v}"; done

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W all_power=\(.all_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

# Hashing 850 MB per arm would add a minute of wall clock to a thermally
# sensitive session, so the weight digest is computed once and cached beside
# the run tree; the cheap files are hashed every time.
head_provenance() {
  local cache="${runs_root}/.head-weights-sha256"
  local w="${head_dir}/model.safetensors"
  if [[ ! -s "${cache}" && -f "${w}" ]]; then
    mkdir -p "${runs_root}"
    shasum -a 256 "${w}" | cut -d' ' -f1 > "${cache}"
  fi
  printf 'weights=%s config=%s index=%s' \
    "$(cat "${cache}" 2>/dev/null || echo absent)" \
    "$(shasum -a 256 "${head_dir}/config.json" 2>/dev/null | cut -d' ' -f1)" \
    "$(shasum -a 256 "${head_dir}/model.safetensors.index.json" 2>/dev/null | cut -d' ' -f1)"
}

status=0
for spec in "$@"; do
  label="${spec%%=*}"
  cfg="${spec#*=}"
  build="${cfg%%:*}"; rest="${cfg#*:}"
  mode="${rest%%:*}"; depth="${rest#*:}"
  src="${bins_root}/${build}"
  out="${runs_root}/${label}"

  if [[ ! -f "${src}/sha256.txt" ]]; then
    echo "e20-run: no built binaries for build ${build} (${src})" >&2
    status=1; break
  fi

  rm -rf "${out}"; mkdir -p "${out}/reports"

  install -m 755 "${src}/mlxfast-swift" "${repo_root}/.build/release/mlxfast-swift"
  install -m 755 "${src}/mlxfast-runtime-worker" \
    "${repo_root}/.build-worker/release/mlxfast-runtime-worker"

  installed_cli="$(shasum -a 256 "${repo_root}/.build/release/mlxfast-swift" | cut -d' ' -f1)"
  installed_worker="$(shasum -a 256 "${repo_root}/.build-worker/release/mlxfast-runtime-worker" | cut -d' ' -f1)"
  want_cli="$(awk '$2=="mlxfast-swift"{print $1}' "${src}/sha256.txt")"
  want_worker="$(awk '$2=="mlxfast-runtime-worker"{print $1}' "${src}/sha256.txt")"
  if [[ "${installed_cli}" != "${want_cli}" || "${installed_worker}" != "${want_worker}" ]]; then
    echo "e20-run: ${label}: installed hashes do not match build ${build}" >&2
    status=1; break
  fi

  export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_QWEN_MTP_DEPTH="${depth}"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_SWIFT_BIN="${repo_root}/research/e20-cli.sh"
  export MLXFAST_MACMON_BIN="${macmon_bin}"
  export MLXFAST_LOCAL_COOL_GATE=0

  if [[ "${mode}" == "0" ]]; then
    unset MLX_QWEN_ATTRIB MLXFAST_NO_SANDBOX
  else
    export MLX_QWEN_ATTRIB="${mode}"
    # The worker sandbox denies file-write*, and mtp-timed swallows worker
    # stderr, so the file sink needs the documented local relaxation. Applied
    # to every attributing arm, so it is never a difference between arms.
    export MLXFAST_NO_SANDBOX=1
  fi

  {
    echo "label=${label}"
    echo "build=${build}"
    echo "attrib_mode=${mode}"
    echo "offered_depth=${depth}"
    echo "tokens=${tokens}"
    echo "head_dir=${head_dir}"
    echo "head_provenance_sha256=$(head_provenance)"
    echo "head_bytes=$(find "${head_dir}" -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')"
    echo "head_dtype=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("torch_dtype") or c.get("dtype") or "unset")' "${head_dir}/config.json" 2>/dev/null || echo unreadable)"
    echo "cli_sha256=${installed_cli}"
    echo "worker_sha256=${installed_worker}"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "no_sandbox=${MLXFAST_NO_SANDBOX:-0}"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "mlx_qwen_env=$(env | sed -n 's/^\(MLX_QWEN_[A-Z_]*\)=.*/\1/p' | sort | tr '\n' ',')"
    echo "thermal_before=$(sample_thermal)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  echo "=== e20-run: ${label} (build ${build} mode ${mode} depth ${depth}) ==="
  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?
  {
    echo "exit=${rc}"
    echo "thermal_after=$(sample_thermal)"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"
  echo "--- ${label} meta ---"; cat "${out}/meta.txt"
  if ((rc != 0)); then
    echo "e20-run: ${label}: benchmark exited ${rc}" >&2
    status=1; break
  fi
done
exit "${status}"
