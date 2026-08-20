#!/usr/bin/env bash
# E73 rung 1: time every legal `(M, IPG)` group partition on the GPU, all 19
# arms in ONE process and ONE thermal session behind the real 40C cool gate.
#
# One session, not one per M: the two-term model is fit across partitions AND
# across widths, so a cross-M contrast must not also be a cross-session
# contrast. The palindrome order gives every arm the same mean leg position
# inside that one session.
#
#   research/e73_rung1.sh [--reps N] [--shape NAME] [--arms a,b,c]
#                         [--target-bytes X] [--tag SUFFIX] [--skip-gate]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

reps="9"
target_bytes="12e9"
artifacts="research/e73-artifacts"
log="${artifacts}/rung1.log"
build="/tmp/e73-build"
shape=""
arms=""
tag=""
skip_gate=""
skip_wandb=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --reps) reps="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --arms) arms="$2"; shift 2 ;;
    --target-bytes) target_bytes="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --skip-gate) skip_gate="1"; shift ;;
    --skip-wandb) skip_wandb="1"; shift ;;
    *) echo "e73_rung1: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "${artifacts}" "${build}"

macmon_bin="${MLXFAST_MACMON_BIN:-}"
if [[ -z "${macmon_bin}" ]]; then
  for macmon_cand in "${HOME}/bin/macmon" "$(command -v macmon 2>/dev/null || true)"; do
    if [[ -n "${macmon_cand}" && -x "${macmon_cand}" ]]; then
      macmon_bin="${macmon_cand}"
      break
    fi
  done
fi
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

if [[ -z "${arms}" ]]; then
  arms="$(python3 -c "import sys; sys.path.insert(0, 'research');
from e73_pairs import pairs, name
print(','.join(name(m, i) for m, i in pairs()))")"
fi

head_sha="$(git rev-parse HEAD)"
{
  echo "e73_rung1 head=${head_sha} dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e73_rung1 host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e73_rung1 arms=${arms}"
  echo "e73_rung1 reps=${reps} target_bytes=${target_bytes} shape=${shape:-all}"
  echo "e73_rung1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

src="${build}/arms${tag}.metal"
python3 research/e73_emit_arms.py --out "${src}" | tee -a "${log}" || exit 1
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e73_cell_ab" research/e73_cell_ab.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings\? generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e73_cell_ab" ]] || {
  echo "e73_rung1: harness build failed" | tee -a "${log}"; exit 1; }

gate_line="cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
if [[ -z "${skip_gate}" ]]; then
  echo "e73_rung1 cool_gate before=$(sample_thermal)" | tee -a "${log}"
  if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
    gate_line="cool_gate_passed_real_gate=true gate_qualified_for_timing=true"
  fi
else
  gate_line="cool_gate=skipped_by_request cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
fi
echo "e73_rung1 ${gate_line}" | tee -a "${log}"
echo "e73_rung1 entry_thermal $(sample_thermal)" | tee -a "${log}"

out="${artifacts}/rung1${tag}.json"
cfg="${build}/config${tag}.json"
python3 - "${cfg}" "${reps}" "${head_sha}" "${src}" "${arms}" "${gate_line}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
cfg, reps, head, src, arms, gate = sys.argv[1:7]
record = {
    "experiment": "e73",
    "rung": 1,
    "harness": "local",
    "reps": int(reps),
    "head_sha": head,
    "arms": arms,
    "host_chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True).stdout.strip(),
    "source_sha256": hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest(),
    "grid": "dispatchThreadgroups(M, N/8, 1) x threadsPerThreadgroup(32, 2, 1)",
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

if [[ -n "${skip_wandb}" ]]; then
  MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e73_cell_ab" \
    --source "${src}" --arms "${arms}" --reps "${reps}" --warmup-reps 1 \
    --target-bytes "${target_bytes}" \
    ${shape_args[@]+"${shape_args[@]}"} --out "${out}" 2>&1 | tee -a "${log}"
  status="${PIPESTATUS[0]}"
else
  MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e73_cell_ab" \
    --source "${src}" --arms "${arms}" --reps "${reps}" --warmup-reps 1 \
    --target-bytes "${target_bytes}" \
    ${shape_args[@]+"${shape_args[@]}"} --out "${out}" \
    2> >(tee -a "${log}" >&2) \
    | python3 research/e73_wandb_stream.py --name "e73-rung1${tag}" \
        --config "${cfg}" 2> >(tee -a "${log}" >&2) \
    | tee -a "${log}" >/dev/null
  status="${PIPESTATUS[0]}"
fi

echo "e73_rung1 exit_thermal $(sample_thermal) status=${status}" | tee -a "${log}"
echo "e73_rung1 finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${log}"
exit "${status}"
