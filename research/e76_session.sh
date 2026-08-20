#!/usr/bin/env bash
# E76: run the arms from research/e76_wide_gen.py on the GPU, one NA per process.
#
# Two modes, because the two questions have different requirements:
#
#   parity  Bit-identity. Every arm is compared against `plain` on all seven
#           scored shapes. This is correctness, not timing, so it takes no cool
#           gate and no W&B run. The session harness holds at most 12 arms, so
#           the arm set is run in batches that each carry `plain` as reference.
#
#   timed   Cost. The named arms are timed in palindrome order inside one
#           process, behind the real 40 C cool gate, with entry and exit GPU
#           temperature recorded and each leg streamed to W&B as it lands.
#
#   research/e76_session.sh --mode parity
#   research/e76_session.sh --mode timed --arms plain,rps2lazy --reps 21
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

mode="parity"
na_list="5 6"
reps="21"
target_bytes="24e9"
artifacts="research/e76-artifacts"
log=""
build="/tmp/e76-build"
arms=""
tag=""
skip_gate=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --na) na_list="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --arms) arms="$2"; shift 2 ;;
    --target-bytes) target_bytes="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    --skip-gate) skip_gate="1"; shift ;;
    *) echo "e76_session: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "${log}" ]] || log="${artifacts}/${mode}.log"
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
  echo "e76_session mode=${mode} head=${head_sha} dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e76_session host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e76_session na_list=${na_list} reps=${reps} target_bytes=${target_bytes}"
  echo "e76_session started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

python3 research/e76_wide_gen.py --check | tee -a "${log}" || exit 1

clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e76_cell_ab" research/e69_cell_ab.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings\? generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e76_cell_ab" ]] || {
  echo "e76_session: harness build failed" | tee -a "${log}"; exit 1; }

# The session harness holds at most 12 arms, so the parity mode splits the arm
# set into batches that each carry `plain` in position 0 as the reference.
all_arms="$(python3 -c "
import sys; sys.path.insert(0, 'research')
from e76_wide_gen import ARMS
print(','.join(a for a, _, _, _ in ARMS))")"

overall=0
for na in ${na_list}; do
  src="${build}/arms_na${na}.metal"
  python3 research/e76_emit_arms.py --na "${na}" --out "${src}" \
    | tee -a "${log}" || exit 1

  if [[ "${mode}" == "parity" ]]; then
    batch=0
    python3 - "${all_arms}" <<'PY' > "${build}/batches.txt"
import sys
arms = sys.argv[1].split(",")
rest = [a for a in arms if a != "plain"]
for start in range(0, len(rest), 11):
    print(",".join(["plain", *rest[start:start + 11]]))
PY
    while read -r batch_arms; do
      out="${artifacts}/parity-na${na}-b${batch}${tag}.json"
      echo "e76_session na=${na} parity batch=${batch} arms=${batch_arms}" \
        | tee -a "${log}"
      MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e76_cell_ab" \
        --prefix e76_cell_ --source "${src}" --na "${na}" --reps 1 \
        --warmup-reps 0 --target-bytes 1e8 --arms "${batch_arms}" \
        --out "${out}" 2>&1 | tee -a "${log}"
      status="${PIPESTATUS[0]}"
      [[ "${status}" -ne 0 ]] && overall="${status}"
      batch=$((batch + 1))
    done < "${build}/batches.txt"
    continue
  fi

  out="${artifacts}/timed-na${na}${tag}.json"
  gate_line="gate_qualified_for_timing=false"
  if [[ -z "${skip_gate}" ]]; then
    echo "e76_session na=${na} cool_gate before=$(sample_thermal)" | tee -a "${log}"
    if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
      gate_line="cool_gate_passed_real_gate=true gate_qualified_for_timing=true"
    else
      gate_line="cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
    fi
  else
    gate_line="cool_gate=skipped_by_request cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
  fi
  echo "e76_session na=${na} ${gate_line}" | tee -a "${log}"
  echo "e76_session na=${na} entry_thermal $(sample_thermal)" | tee -a "${log}"

  cfg="${build}/config_na${na}.json"
  python3 - "${cfg}" "${na}" "${reps}" "${head_sha}" "${gate_line}" "${src}" \
      "${arms:-${all_arms}}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
cfg, na, reps, head, gate, src, arms = sys.argv[1:8]
record = {
    "experiment": "e76",
    "rung": 2,
    "na": int(na),
    "reps": int(reps),
    "arms": arms,
    "head_sha": head,
    "host_chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True).stdout.strip(),
    "source_sha256": hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest(),
    "arms_header_sha256": hashlib.sha256(
        pathlib.Path("research/generated/e76_wide_arms.h").read_bytes()).hexdigest(),
    "grid": "dispatchThreadgroups(1, N/8, 1) x threadsPerThreadgroup(32, 2, 1)",
    "host_grid_editable": False,
}
for token in gate.split():
    if "=" in token:
        key, value = token.split("=", 1)
        record[key] = {"true": True, "false": False}.get(value, value)
pathlib.Path(cfg).write_text(json.dumps(record, indent=2))
PY

  MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e76_cell_ab" \
    --prefix e76_cell_ --source "${src}" --na "${na}" --reps "${reps}" \
    --warmup-reps 1 --target-bytes "${target_bytes}" \
    --arms "${arms:-${all_arms}}" --out "${out}" \
    2> >(tee -a "${log}" >&2) \
    | python3 research/e69_wandb_stream.py --name "e76-rung2-na${na}${tag}" \
        --config "${cfg}" 2> >(tee -a "${log}" >&2) \
    | tee -a "${log}" >/dev/null
  status="${PIPESTATUS[0]}"

  echo "e76_session na=${na} exit_thermal $(sample_thermal) status=${status}" \
    | tee -a "${log}"
  [[ "${status}" -ne 0 ]] && overall="${status}"
done

echo "e76_session finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${overall}" \
  | tee -a "${log}"
exit "${overall}"
