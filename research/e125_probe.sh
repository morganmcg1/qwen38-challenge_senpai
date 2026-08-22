#!/usr/bin/env bash
# E125 -- one counterbalanced session for the frame axis.
#
#   usage: research/e125_probe.sh TAG [extra probe args...]
#
# The instrument is research/e125_frame_probe.m, which is E118's probe plus a
# frame axis and nothing else. The arms come from research/e125_arms.py, which
# emits E123's arm set from twins pinned at E125_ARM_REV, so an E125 frame
# effect is measured on exactly the arms E123 priced and the two sessions are
# comparable arm for arm.
#
# The comparison is within-session and counterbalanced, so it runs under no
# thermal gate and reports no score. Entry temperature is recorded per
# frame-block, not per cell, because the consumer frame heats the GPU by
# construction and a per-cell number would hide that.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e125_probe.sh TAG [args...]}"
shift
out_dir="research/out/${tag}"
arms_dir="/tmp/e125-arms"
bin="/tmp/e125_frame_probe"
arm_rev="${E125_ARM_REV:-5d97175c~1}"
arms="$(python3 research/e125_arms.py --arm-list)"
[[ -n "${arms}" ]] || { echo "e125_probe: empty arm list" >&2; exit 1; }
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
python3 research/e125_arms.py --emit "${arms_dir}" --rev "${arm_rev}" 2>&1 \
  | tee "${out_dir}/arms.log" || exit 1

clang -fobjc-arc -O2 -Wno-format-nonliteral -framework Metal \
  -framework Foundation -o "${bin}" research/e125_frame_probe.m 2>&1 \
  | grep -v 'setFastMathEnabled\|deprecated\|^ *|\|^ *\^' \
  | tee "${out_dir}/build.log"

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

macmon_arg=()
[[ -n "${macmon_bin}" ]] && macmon_arg=(--macmon "${macmon_bin}")

"${bin}" --dir "${arms_dir}" --out "${PWD}/${out_dir}/rate.json" \
  --fn 'e118_iso_na%d' --arms "${arms}" \
  "${macmon_arg[@]}" "$@" 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e125-isolated-to-in-situ-transfer-law"
  echo "harness=local"
  echo "instrument=research/e125_frame_probe.m (E118 probe plus the frame axis)"
  echo "arms=${arms}"
  echo "arm_twin_rev=${arm_rev}"
  echo "arm_twin_sha=$(git rev-parse "${arm_rev}")"
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
  echo "official_or_ranked_score=false"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
exit "${status}"
