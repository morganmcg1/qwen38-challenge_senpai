#!/usr/bin/env bash
# E110 rung 1 -- loop-nest order and activation staging for one wide x-group,
# on the isolated per-width entry points, with the rule 36b roofline pair.
#
#   usage: research/e110_probe.sh TAG [extra probe args...]
#
# The comparison is within-session and counterbalanced, so it runs under no
# thermal gate and reports no score. Entry and exit GPU temperature are recorded
# per timed cell because a drifting session only supports a relative claim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e110_probe.sh TAG [args...]}"
shift
out_dir="research/out/${tag}"
arms_dir="/tmp/e110-arms"
bin="/tmp/e110_rate_probe"
# The harness admits eight arms per session and runs its positive control on the
# last one, so any override must end with an arm that is exact against a_base.
arms="${E110_ARMS:-a_base,l_loadonly:diag,b_constw:diag,b_barrier,xs_stage,mo_stage,mo_swap}"
mkdir -p "${out_dir}"

macmon_bin=""
for candidate in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                 /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    macmon_bin="${candidate}"
    break
  fi
done

gpu_temp() {
  [[ -n "${macmon_bin}" ]] || { echo ""; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
}

rm -rf "${arms_dir}"
python3 research/e110_arms.py --emit "${arms_dir}" 2>&1 \
  | tee "${out_dir}/arms.log" || exit 1

# E104's harness, unchanged: the arm set, the entry-point name and the width
# list are all arguments, so E110 reuses the instrument that produced the
# numbers it has to compare against.
clang -fobjc-arc -O2 -Wno-format-nonliteral -framework Metal \
  -framework Foundation -o "${bin}" research/e104_rate_probe.m 2>&1 \
  | grep -v 'setFastMathEnabled\|deprecated' | tee "${out_dir}/build.log"

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

macmon_arg=()
[[ -n "${macmon_bin}" ]] && macmon_arg=(--macmon "${macmon_bin}")

"${bin}" --dir "${arms_dir}" --out "${PWD}/${out_dir}/rate.json" \
  --fn 'e110_iso_na%d' --arms "${arms}" --widths 2,3,4,5 \
  "${macmon_arg[@]}" "$@" 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e110-one-group-wide-qmv-activation-reread"
  echo "harness=local"
  echo "arms=${arms}"
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
  for f in "${arms_dir}"/arm_*.metal; do
    base="$(basename "${f}")"
    base="${base#arm_}"
    echo "arm_${base%.metal}_sha256=$(shasum -a 256 "${f}" | cut -d' ' -f1)"
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
