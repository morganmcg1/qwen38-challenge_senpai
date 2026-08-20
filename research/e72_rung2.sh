#!/usr/bin/env bash
# E72 rung 2: time the unspilled `qmv_fast_crossrow_affine4_g64_wide` arms on
# the GPU, one NA per process, each behind the real 40C cool gate.
#
# The rung-2 AIR census says the shipped body grows a second alloca,
# `[4 x <6 x float>]`, at NA = 6 and only at NA = 6, and that
# `#pragma clang loop unroll(full)` on the constant-trip loops removes it:
# peak live registers fall 182 -> 146 (`tailfull`) or 182 -> 133 (`allfull`).
# NA = 6 carries 34.6 % of ranked verify-width time, so this asks whether the
# static spill is worth anything on the GPU, and whether removing it lets E69's
# `xvec` arm recover at NA = 6 the -3.56 % it produced at NA = 5.
#
# One session per NA: regenerate the arms from the shipped kernel, certify them
# from AIR, emit the JIT-style source, build the shared session harness, pass
# the real 40C cool gate, then time every arm in palindrome order inside one
# process while a W&B streamer records each leg as it lands.
#
#   research/e72_rung2.sh [--na "5 6"] [--reps N] [--shape NAME]
#                         [--arms a,b,c] [--target-bytes X] [--skip-gate]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

na_list="5 6"
reps="21"
target_bytes="24e9"
artifacts="research/e72-artifacts"
log="${artifacts}/rung2.log"
build="/tmp/e72-build"
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
    *) echo "e72_rung2: unknown argument $1" >&2; exit 2 ;;
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
  echo "e72_rung2 head=${head_sha} dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e72_rung2 host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e72_rung2 na_list=${na_list} reps=${reps} target_bytes=${target_bytes}"
  echo "e72_rung2 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

python3 research/e72_wide_gen.py --check | tee -a "${log}" || exit 1
if [[ -z "${skip_census}" ]]; then
  # Tagged per session: the canonical NA=2..6 census at rung2-air.json is
  # written by an explicit full run, so a narrow session cannot silently
  # replace it with its own subset.
  python3 research/e72_air_census.py --na ${na_list} \
    --out "${artifacts}/rung2-air-session${tag}.json" | tee -a "${log}" || exit 1
fi
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e72_cell_ab" research/e69_cell_ab.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings\? generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e72_cell_ab" ]] || {
  echo "e72_rung2: harness build failed" | tee -a "${log}"; exit 1; }

overall=0
for na in ${na_list}; do
  out="${artifacts}/rung2-na${na}${tag}.json"
  src="${build}/arms_na${na}.metal"
  python3 research/e72_emit_arms.py --na "${na}" --out "${src}" \
    | tee -a "${log}" || exit 1

  gate_line="gate_qualified_for_timing=false"
  if [[ -z "${skip_gate}" ]]; then
    echo "e72_rung2 na=${na} cool_gate before=$(sample_thermal)" | tee -a "${log}"
    if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
      gate_line="cool_gate_passed_real_gate=true gate_qualified_for_timing=true"
    else
      gate_line="cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
    fi
  else
    gate_line="cool_gate=skipped_by_request cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
  fi
  echo "e72_rung2 na=${na} ${gate_line}" | tee -a "${log}"
  echo "e72_rung2 na=${na} entry_thermal $(sample_thermal)" | tee -a "${log}"

  cfg="${build}/config_na${na}.json"
  python3 - "${cfg}" "${na}" "${reps}" "${head_sha}" "${gate_line}" "${src}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
cfg, na, reps, head, gate, src = sys.argv[1:7]
record = {
    "experiment": "e72",
    "rung": 2,
    "na": int(na),
    "reps": int(reps),
    "head_sha": head,
    "host_chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True).stdout.strip(),
    "source_sha256": hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest(),
    "arms_header_sha256": hashlib.sha256(
        pathlib.Path("research/generated/e72_wide_arms.h").read_bytes()).hexdigest(),
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
  arm_args=(--arms "${arms:-plain,tailfull,mfull,rfull,allfull,split,shift,xvec,tailfullxvec}")

  if [[ -n "${skip_wandb}" ]]; then
    MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e72_cell_ab" \
      --prefix e72_cell_ --source "${src}" --na "${na}" --reps "${reps}" \
      --warmup-reps 1 --target-bytes "${target_bytes}" \
      "${arm_args[@]}" ${shape_args[@]+"${shape_args[@]}"} --out "${out}" \
      2>&1 | tee -a "${log}"
    status="${PIPESTATUS[0]}"
  else
    MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e72_cell_ab" \
      --prefix e72_cell_ --source "${src}" --na "${na}" --reps "${reps}" \
      --warmup-reps 1 --target-bytes "${target_bytes}" \
      "${arm_args[@]}" ${shape_args[@]+"${shape_args[@]}"} --out "${out}" \
      2> >(tee -a "${log}" >&2) \
      | python3 research/e69_wandb_stream.py --name "e72-rung2-na${na}${tag}" \
          --config "${cfg}" 2> >(tee -a "${log}" >&2) \
      | tee -a "${log}" >/dev/null
    status="${PIPESTATUS[0]}"
  fi

  echo "e72_rung2 na=${na} exit_thermal $(sample_thermal) status=${status}" \
    | tee -a "${log}"
  [[ "${status}" -ne 0 ]] && overall="${status}"
done

echo "e72_rung2 finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${overall}" \
  | tee -a "${log}"
exit "${overall}"
