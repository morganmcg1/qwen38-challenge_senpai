#!/usr/bin/env bash
# E98 rung 1b -- price the indexed metadata read inside the SCORED cross-row
# kernel, and record the identity tuple around it.
#
#   usage: research/e98_lut_probe.sh TAG [extra e98_qmv_ab args...]
#
# Rung 1a could only reach `qmv_fast_impl` at M = 1 because the cross-row
# family is gated on `group_size == 64`. This probe compiles three variants of
# the runtime-effective JIT string and alternates them inside one session, so
# the scored cross-row cells are measured directly.
#
# The comparison is within-session and counterbalanced, so it runs under no
# thermal gate and reports no score. Entry and exit GPU temperature are
# recorded because a drifting session only supports a relative claim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e98_lut_probe.sh TAG [args...]}"
shift
out_dir="research/out/${tag}"
arms_dir="/tmp/e98-arms"
bin="/tmp/e98_qmv_ab"
mkdir -p "${out_dir}"

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

python3 research/e98_variant_sources.py --outdir "${arms_dir}" \
  2>&1 | tee "${out_dir}/arms.log" || exit 1

clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${bin}" research/e98_qmv_ab.m 2>&1 | grep -v 'setFastMathEnabled\|deprecated' \
  | tee "${out_dir}/build.log"

# No GPU: what the indexed read costs in registers and AIR at the entry point
# MLX really compiles.
python3 research/e98_arm_regs.py --dir "${arms_dir}" \
  --out "${out_dir}/regs.json" 2>&1 | tee "${out_dir}/regs.log"

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${bin}" --dir "${arms_dir}" --out "${PWD}/${out_dir}/lut.json" "$@" \
  2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e98-transform-owned-weight-metadata-index"
  echo "rung=1b-lut-emulation"
  echo "harness=local"
  echo "args=$*"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "metal_toolchain=$(xcrun -sdk macosx metal --version 2>&1 | head -1)"
  for a in a b c; do
    echo "arm_${a}_sha256=$(shasum -a 256 "${arms_dir}/arm_${a}.metal" | cut -d' ' -f1)"
  done
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "timing_valid=false"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
exit "${status}"
