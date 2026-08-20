#!/usr/bin/env bash
# E69 rung 1: time the isolated `qmv_fast_crossrow_affine4_g64_wide` arms on the
# GPU, one NA per process, each behind the real 40C cool gate.
#
# The rung-0 AIR census says the kernel issues 16 weight loads, 8 scale/bias
# loads and 4*NA x loads per lane per k-block, every one of them a 2-byte scalar,
# so 80% of the load instructions at NA=6 belong to x. Rung 1 asks whether that
# instruction count is what the GPU is actually spending time on.
#
# One session per NA: regenerate the arms from the shipped kernel, certify them
# from AIR, emit the JIT-style source, build the harness, pass the real 40C cool
# gate, then time every arm in palindrome order inside one process while a W&B
# streamer records each leg as it lands.
#
#   research/e69_rung1.sh [--na "2 3 4 5 6"] [--reps N] [--shape NAME]
#                         [--arms a,b,c] [--target-bytes X] [--skip-gate]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

na_list="2 3 4 5 6"
reps="21"
target_bytes="24e9"
artifacts="research/e69-artifacts"
log="${artifacts}/rung1.log"
build="/tmp/e69-build"
shape=""
arms=""
tag=""
skip_gate=""
skip_census=""
skip_wandb=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --na) na_list="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --arms) arms="$2"; shift 2 ;;
    --target-bytes) target_bytes="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    # Suffix for the per-NA artifact and W&B run, so a replication session
    # cannot overwrite the session it is meant to check.
    --tag) tag="$2"; shift 2 ;;
    --skip-gate) skip_gate="1"; shift ;;
    --skip-census) skip_census="1"; shift ;;
    --skip-wandb) skip_wandb="1"; shift ;;
    *) echo "e69_rung1: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "${artifacts}" "${build}"

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

head_sha="$(git rev-parse HEAD)"
{
  echo "e69_rung1 head=${head_sha} dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e69_rung1 host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e69_rung1 na_list=${na_list} reps=${reps} target_bytes=${target_bytes}"
  echo "e69_rung1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

python3 research/e69_wide_gen.py --check | tee -a "${log}" || exit 1
if [[ -z "${skip_census}" ]]; then
  python3 research/e69_air_census.py --na ${na_list} \
    --out "${artifacts}/rung0-air.json" | tee -a "${log}" || exit 1
fi
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e69_cell_ab" research/e69_cell_ab.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings\? generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e69_cell_ab" ]] || {
  echo "e69_rung1: harness build failed" | tee -a "${log}"; exit 1; }

overall=0
for na in ${na_list}; do
  out="${artifacts}/rung1-na${na}${tag}.json"
  src="${build}/arms_na${na}.metal"
  python3 research/e69_emit_arms.py --na "${na}" --out "${src}" \
    | tee -a "${log}" || exit 1

  gate_line="gate_qualified_for_timing=false"
  if [[ -z "${skip_gate}" ]]; then
    echo "e69_rung1 na=${na} cool_gate before=$(sample_thermal)" | tee -a "${log}"
    if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
      gate_line="cool_gate_passed_real_gate=true gate_qualified_for_timing=true"
    else
      gate_line="cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
    fi
  else
    gate_line="cool_gate=skipped_by_request cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
  fi
  echo "e69_rung1 na=${na} ${gate_line}" | tee -a "${log}"
  echo "e69_rung1 na=${na} entry_thermal $(sample_thermal)" | tee -a "${log}"

  cfg="${build}/config_na${na}.json"
  python3 - "${cfg}" "${na}" "${reps}" "${head_sha}" "${gate_line}" "${src}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
cfg, na, reps, head, gate, src = sys.argv[1:7]
record = {
    "experiment": "e69",
    "rung": 1,
    "na": int(na),
    "reps": int(reps),
    "head_sha": head,
    "host_chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True).stdout.strip(),
    "source_sha256": hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest(),
    "arms_header_sha256": hashlib.sha256(
        pathlib.Path("research/generated/e69_wide_arms.h").read_bytes()).hexdigest(),
    "grid": "dispatchThreadgroups(1, N/8, 1) x threadsPerThreadgroup(32, 2, 1)",
    "host_grid_editable": False,
}
for token in gate.split():
    if "=" in token:
        key, value = token.split("=", 1)
        record[key] = {"true": True, "false": False}.get(value, value)
pathlib.Path(cfg).write_text(json.dumps(record, indent=2))
PY

  shape_args=()
  [[ -n "${shape}" ]] && shape_args=(--shape "${shape}")
  [[ -n "${arms}" ]] && shape_args+=(--arms "${arms}")

  if [[ -n "${skip_wandb}" ]]; then
    MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e69_cell_ab" \
      --source "${src}" --na "${na}" --reps "${reps}" --warmup-reps 1 \
      --target-bytes "${target_bytes}" \
      ${shape_args[@]+"${shape_args[@]}"} --out "${out}" 2>&1 | tee -a "${log}"
    status="${PIPESTATUS[0]}"
  else
    MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e69_cell_ab" \
      --source "${src}" --na "${na}" --reps "${reps}" --warmup-reps 1 \
      --target-bytes "${target_bytes}" \
      ${shape_args[@]+"${shape_args[@]}"} --out "${out}" \
      2> >(tee -a "${log}" >&2) \
      | python3 research/e69_wandb_stream.py --name "e69-rung1-na${na}${tag}" \
          --config "${cfg}" 2> >(tee -a "${log}" >&2) \
      | tee -a "${log}" >/dev/null
    status="${PIPESTATUS[0]}"
  fi

  echo "e69_rung1 na=${na} exit_thermal $(sample_thermal) status=${status}" \
    | tee -a "${log}"
  [[ "${status}" -ne 0 ]] && overall="${status}"
done

echo "e69_rung1 finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${overall}" \
  | tee -a "${log}"
exit "${overall}"
