#!/usr/bin/env bash
# E77 rung 1: measure GPU time against per-thread register count at FIXED
# traffic, fixed group count and fixed grid, in ONE process and ONE thermal
# session behind the real 40C cool gate.
#
# One session, not one per pressure level: the sweep reads a ratio between
# pressure levels, so a cross-pressure contrast must not also be a
# cross-session contrast. The palindrome order gives every arm the same mean
# leg position inside that one session.
#
#   research/e77_rung1.sh [--reps N] [--shape NAME] [--arms a,b,c]
#                         [--target-bytes X] [--tag SUFFIX] [--skip-gate]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

reps="11"
target_bytes="12e9"
artifacts="research/e77-artifacts"
log="${artifacts}/rung1.log"
build="/tmp/e77-build"
regs="${artifacts}/rung0-regs.json"
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
    *) echo "e77_rung1: unknown argument $1" >&2; exit 2 ;;
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
  echo "e77_rung1 head=${head_sha} dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e77_rung1 host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e77_rung1 reps=${reps} target_bytes=${target_bytes} shape=${shape:-all}"
  echo "e77_rung1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

# The emitter writes the arm list to stdout and its digest to stderr. The
# digest must match the rung-0 census, or the measured points carry register
# counts that were read from different code.
src="${build}/sweep${tag}.metal"
emitted="$(python3 research/e77_emit_sweep.py --out "${src}" \
  2> >(tee -a "${log}" >&2))" || exit 1
[[ -n "${arms}" ]] || arms="${emitted}"
echo "e77_rung1 arms=${arms}" | tee -a "${log}"

python3 - "${src}" "${regs}" <<'PY' | tee -a "${log}"
import hashlib, json, pathlib, sys
src, regs = sys.argv[1:3]
have = hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest()
want = json.loads(pathlib.Path(regs).read_text())["sweep_source_sha256"]
print(f"e77_rung1 source_sha256={have} rung0_source_sha256={want} "
      f"census_matches_source={str(have == want).lower()}")
PY

clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e77_sweep" research/e77_sweep.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings\? generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e77_sweep" ]] || {
  echo "e77_rung1: harness build failed" | tee -a "${log}"; exit 1; }

gate_line="cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
if [[ -z "${skip_gate}" ]]; then
  echo "e77_rung1 cool_gate before=$(sample_thermal)" | tee -a "${log}"
  if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
    gate_line="cool_gate_passed_real_gate=true gate_qualified_for_timing=true"
  fi
else
  gate_line="cool_gate=skipped_by_request cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
fi
echo "e77_rung1 ${gate_line}" | tee -a "${log}"
echo "e77_rung1 entry_thermal $(sample_thermal)" | tee -a "${log}"

out="${artifacts}/rung1${tag}.json"
cfg="${build}/config${tag}.json"
python3 - "${cfg}" "${reps}" "${head_sha}" "${src}" "${arms}" "${gate_line}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
cfg, reps, head, src, arms, gate = sys.argv[1:7]
record = {
    "experiment": "e77",
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
  MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e77_sweep" \
    --source "${src}" --arms "${arms}" --reps "${reps}" --warmup-reps 1 \
    --target-bytes "${target_bytes}" \
    ${shape_args[@]+"${shape_args[@]}"} --out "${out}" 2>&1 | tee -a "${log}"
  status="${PIPESTATUS[0]}"
else
  MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e77_sweep" \
    --source "${src}" --arms "${arms}" --reps "${reps}" --warmup-reps 1 \
    --target-bytes "${target_bytes}" \
    ${shape_args[@]+"${shape_args[@]}"} --out "${out}" \
    2> >(tee -a "${log}" >&2) \
    | python3 research/e77_wandb_stream.py --name "e77-rung1${tag}" \
        --config "${cfg}" --regs "${regs}" 2> >(tee -a "${log}" >&2) \
    | tee -a "${log}" >/dev/null
  status="${PIPESTATUS[0]}"
fi

echo "e77_rung1 exit_thermal $(sample_thermal) status=${status}" | tee -a "${log}"

# The sweep binary knows nothing about the cool gate, so fold the driver's
# provenance and gate record into the artifact the analysis reads.
if [[ "${status}" -eq 0 ]]; then
  python3 - "${out}" "${cfg}" <<'PY'
import json, pathlib, sys
out, cfg = (pathlib.Path(p) for p in sys.argv[1:3])
record = json.loads(out.read_text())
record.update({k: v for k, v in json.loads(cfg.read_text()).items()
               if k not in record})
out.write_text(json.dumps(record, indent=2))
PY
  cp "${log}" "${artifacts}/rung1${tag}.log"
fi

echo "e77_rung1 finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${log}"
exit "${status}"
