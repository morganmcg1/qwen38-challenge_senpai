#!/usr/bin/env bash
# E104 -- measure rate(NA) for one wide x-group and find what binds it.
#
#   usage: E104_SET=rate|ladder research/e104_probe.sh TAG [extra probe args...]
#
# The comparison is within-session and counterbalanced, so it runs under no
# thermal gate and reports no score. Entry and exit GPU temperature are recorded
# per timed cell because a drifting session only supports a relative claim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e104_probe.sh TAG [args...]}"
shift
set_name="${E104_SET:-rate}"
out_dir="research/out/${tag}"
arms_dir="/tmp/e104-arms-${set_name}"
bin="/tmp/e104_rate_probe"
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
python3 research/e104_variant_sources.py --outdir "${arms_dir}" \
  --set "${set_name}" 2>&1 | tee "${out_dir}/arms.log" || exit 1

arm_names=()
for f in "${arms_dir}"/arm_*.metal; do
  base="$(basename "${f}")"
  base="${base#arm_}"
  arm_names+=("${base%.metal}")
done

clang -fobjc-arc -O2 -Wno-format-nonliteral -framework Metal \
  -framework Foundation -o "${bin}" research/e104_rate_probe.m 2>&1 \
  | grep -v 'setFastMathEnabled\|deprecated' | tee "${out_dir}/build.log"

# No GPU: the source-level census that prices the arms before they are timed.
arm_list="$(IFS=,; echo "${arm_names[*]}")"
if [[ "${set_name}" == "rate" ]]; then
  python3 research/e104_arm_census.py --dir "${arms_dir}" \
    --out "${out_dir}/census.json" 2>&1 | tee "${out_dir}/census.log"
else
  python3 research/e104_ladder_census.py --dir "${arms_dir}" \
    --arms "${arm_list}" \
    --out "${out_dir}/census.json" 2>&1 | tee "${out_dir}/census.log"
fi
if [[ "${set_name}" == "arith" ]]; then
  # The register census cannot see the FP instruction count, and that count is
  # the whole axis these arms move, so read it from the AIR as well.
  python3 research/e104_arm_census.py --dir "${arms_dir}" --arms "${arm_list}" \
    --na 2,3,4,5,6 --out "${out_dir}/air-census.json" 2>&1 \
    | tee "${out_dir}/air-census.log"
fi

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

macmon_arg=()
[[ -n "${macmon_bin}" ]] && macmon_arg=(--macmon "${macmon_bin}")

"${bin}" --dir "${arms_dir}" --out "${PWD}/${out_dir}/rate.json" \
  "${macmon_arg[@]}" "$@" 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e104-why-a-wide-x-group-streams-slowly"
  echo "harness=local"
  echo "arm_set=${set_name}"
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
  for a in "${arm_names[@]}"; do
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
