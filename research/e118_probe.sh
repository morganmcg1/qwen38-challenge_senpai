#!/usr/bin/env bash
# E118 -- the metadata-load instruction screen for the wide affine-4 QMV.
#
#   usage: research/e118_probe.sh TAG [extra probe args...]
#
# The comparison is within-session and counterbalanced, so it runs under no
# thermal gate and reports no score. Entry and exit GPU temperature are recorded
# per timed cell because a drifting session only supports a relative claim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e118_probe.sh TAG [args...]}"
shift
out_dir="research/out/${tag}"
arms_dir="/tmp/e118-arms"
bin="/tmp/e118_qmv_probe"
arms="a_base,q_scaffold,s_bcast,s_bcast_all,s_bcast_scale,p_split_meta,g_pack32,s_bcast_pack32,p_prefetch_w,n_nosums:diag,l_loadonly:diag,n_nobias:diag,d_bias1:diag,e_bias6,z_ballast"
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
python3 research/e118_arms.py --emit "${arms_dir}" 2>&1 \
  | tee "${out_dir}/arms.log" || exit 1

clang -fobjc-arc -O2 -Wno-format-nonliteral -framework Metal \
  -framework Foundation -o "${bin}" research/e118_qmv_probe.m 2>&1 \
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
  echo "experiment=e118-wide-qmv-inner-loop-load-instruction-screen"
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
