#!/usr/bin/env bash
# E147 rung E-1b: does the retiled qmm_t reproduce the pinned golden rows?
#
#   usage: research/e147_rungE1b.sh
#
# WHAT THIS RUNG DECIDES. Rung E-1a proved the grid-stride tile re-derivation
# correct by arithmetic over every scored shape. Arithmetic cannot see the
# things a real build decides for itself: whether BlockMMA, BlockLoader and
# QuantizedBlockLoader are legal at (BM, BN) = (64, 16), whether the retiled
# threadgroup allocation is sound, and whether the reordered accumulation
# reproduces the shipped tokens bit for bit. This rung answers all three at
# once against reference rows the retile cannot have influenced.
#
# WHY THE GOLDEN FILES ARE THE RIGHT REFERENCE, AND --local-submit IS NOT.
# `--local-submit` generates its reference rows from the candidate build, so
# both of its legs run the retiled kernel and a systematic numeric change
# cancels out of the comparison. It cannot falsify a kernel edit. The E128
# goldens were written on 2026-08-23 at 00:00Z and 00:10Z from the unmodified
# kernel path, before the first E147 kernel commit 01ed58d5 at 05:22Z, so they
# are a genuine cross-build reference. `research/e128_session.sh` re-validates
# `reference_self_consistent` and the row count before it uses them and records
# `golden_sha256` in every leg's meta.txt.
#
# HOW THE ARM IS SELECTED. The runtime-effective source of the `quantized`
# family is the Metal JIT string compiled INTO the worker binary, so an arm is
# a whole worker binary. The single line `constexpr bool kE147RetileOn` is the
# whole arm switch; both polarities are asserted on real builds, in both
# directions, before any leg runs (Rule 136).
#
# RULE 101 POSITIVE CONTROL. A green exactness result is worthless unless the
# same harness can go red on the same code path. Phase 2 builds a third worker
# whose only difference from the retile arm is that the tile decode rotates the
# column index by one host-independent step. That permutes the output columns
# of every retiled GEMM while keeping every pointer in range, so it is a defect
# the golden comparison must catch and cannot fault the GPU.
#
# THERMAL MODE. Nothing here is timed. Exactness does not depend on the cool
# gate, and no number this rung produces is a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prompts=(beagle_a essays_montaigne)
control_prompt=beagle_a
tokens=512
depth=8

bin_root=".mlxfast-private/e147/bin"
runs_parent=".mlxfast-private/e147/runs"
goldens_dir="${E147_GOLDENS_DIR:-.mlxfast-private/e128/goldens}"
worker=".build-worker/release/mlxfast-runtime-worker"
out="research/e147-rungE1b.txt"

header="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

ARM_OFF='constexpr bool kE147RetileOn = false;'
ARM_ON='constexpr bool kE147RetileOn = true;'
TILE_OK='compute_tile((t / tiles_x) * BM, (t % tiles_x) * BN);'
TILE_ROT='compute_tile((t / tiles_x) * BM, ((t + 1) % tiles_x) * BN);'

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e147_rungE1b: scored surface is dirty; refusing to build arms" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

restore_tree() { git checkout "${session_commit}" -- "${header}" "${twin}"; }
sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# Every edit below must hit exactly one site in each of the two files. A drifted
# tree therefore stops the rung instead of building a half-applied arm.
edit_both() {
  local old="$1" new="$2" f n
  for f in "${header}" "${twin}"; do
    n="$(grep -F -c -- "${old}" "${f}")"
    if [[ "${n}" != "1" ]]; then
      echo "e147_rungE1b: expected 1 site of '${old}' in ${f}, found ${n}" >&2
      return 1
    fi
    python3 - "${f}" "${old}" "${new}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace(sys.argv[2], sys.argv[3]))
PY
  done
}

mkdir -p "${bin_root}"
: > "${out}"
say() { echo "$*"; echo "$*" >> "${out}"; }

say "e147_rungE1b_session_commit=${session_commit}"
say "e147_rungE1b_goldens_dir=${goldens_dir}"
say "e147_rungE1b_tokens=${tokens}"

echo "=== e147_rungE1b phase 0a: arm OFF worker (submitted default) ==="
senpai/rebuild-and-assert-worker.sh \
  --require "${ARM_OFF}" --forbid "${ARM_ON}" || {
  echo "e147_rungE1b: the OFF polarity assertion failed" >&2; exit 3; }
cp -c "${worker}" "${bin_root}/worker.noretile" 2>/dev/null \
  || cp "${worker}" "${bin_root}/worker.noretile"
off_sha="$(sha_of "${bin_root}/worker.noretile")"
say "e147_rungE1b_worker_off_sha256=${off_sha}"

trap 'restore_tree' EXIT

echo "=== e147_rungE1b phase 0b: arm ON worker (64x16 retile) ==="
edit_both "${ARM_OFF}" "${ARM_ON}" || exit 1
python3 research/twin_audit.py || {
  echo "e147_rungE1b: twin audit failed after the ON edit" >&2; exit 3; }
senpai/rebuild-and-assert-worker.sh \
  --require "${ARM_ON}" --forbid "${ARM_OFF}" || {
  echo "e147_rungE1b: the ON polarity assertion failed" >&2; exit 3; }
cp -c "${worker}" "${bin_root}/worker.retile" 2>/dev/null \
  || cp "${worker}" "${bin_root}/worker.retile"
on_sha="$(sha_of "${bin_root}/worker.retile")"
say "e147_rungE1b_worker_on_sha256=${on_sha}"
if [[ "${off_sha}" == "${on_sha}" ]]; then
  say "e147_rungE1b_arm_contrast=EMPTY"
  echo "e147_rungE1b: both polarities produced the same binary" >&2
  exit 3
fi
say "e147_rungE1b_arm_contrast=ok"

echo "=== e147_rungE1b phase 1: exactness legs on the retile arm ==="
cp -c "${bin_root}/worker.retile" "${worker}" 2>/dev/null \
  || cp "${bin_root}/worker.retile" "${worker}"
chmod +x "${worker}"

matched_all=1
for id in "${prompts[@]}"; do
  slot="rungE1b-retile"
  legdir="${runs_parent}/${slot}/${id}"
  rm -rf "${legdir}"
  echo "=== e147_rungE1b leg: prompt=${id} arm=retile ==="
  E128_FORCE=1 E128_NO_TRACE=1 E128_TOKENS="${tokens}" E128_DEPTH="${depth}" \
  E128_ROOT=".mlxfast-private/e147" E128_GOLDENS_DIR="${goldens_dir}" \
  E128_RUNS_DIR="runs/${slot}" \
    research/e128_session.sh "${id}"
  status=$?
  meta="${legdir}/meta.txt"
  got="$(sed -n 's/^worker_sha256=//p' "${meta}" 2>/dev/null)"
  m="$(sed -n 's/^all_tokens_matched=//p' "${meta}" 2>/dev/null)"
  r="$(sed -n 's/^residual_divergence_count=//p' "${meta}" 2>/dev/null)"
  g="$(sed -n 's/^golden_sha256=//p' "${meta}" 2>/dev/null)"
  say "e147_rungE1b_${id}_exit=${status}"
  say "e147_rungE1b_${id}_worker_sha256=${got:-none}"
  say "e147_rungE1b_${id}_worker_is_retile_arm=$([[ "${got}" == "${on_sha}" ]] \
       && echo true || echo false)"
  say "e147_rungE1b_${id}_golden_sha256=${g:-none}"
  say "e147_rungE1b_${id}_all_tokens_matched=${m:-none}"
  say "e147_rungE1b_${id}_residual_divergence_count=${r:-none}"
  if [[ "${status}" != "0" || "${m}" != "true" || "${r}" != "0" \
        || "${got}" != "${on_sha}" ]]; then
    matched_all=0
  fi
done
say "e147_rungE1b_retile_exact_local=$([[ ${matched_all} == 1 ]] \
     && echo true || echo false)"

echo "=== e147_rungE1b phase 2: Rule 101 positive control (column rotation) ==="
edit_both "${TILE_OK}" "${TILE_ROT}" || exit 1
senpai/rebuild-and-assert-worker.sh \
  --require "${TILE_ROT}" --forbid "${TILE_OK}" --require "${ARM_ON}" || {
  echo "e147_rungE1b: the control polarity assertion failed" >&2; exit 3; }
ctl_sha="$(sha_of "${worker}")"
say "e147_rungE1b_control_worker_sha256=${ctl_sha}"
say "e147_rungE1b_control_differs_from_retile=$([[ "${ctl_sha}" != "${on_sha}" ]] \
     && echo true || echo false)"

slot="rungE1b-control"
legdir="${runs_parent}/${slot}/${control_prompt}"
rm -rf "${legdir}"
E128_FORCE=1 E128_NO_TRACE=1 E128_TOKENS="${tokens}" E128_DEPTH="${depth}" \
E128_ROOT=".mlxfast-private/e147" E128_GOLDENS_DIR="${goldens_dir}" \
E128_RUNS_DIR="runs/${slot}" \
  research/e128_session.sh "${control_prompt}"
ctl_status=$?
meta="${legdir}/meta.txt"
ctl_m="$(sed -n 's/^all_tokens_matched=//p' "${meta}" 2>/dev/null)"
ctl_r="$(sed -n 's/^residual_divergence_count=//p' "${meta}" 2>/dev/null)"
ctl_diag="$(grep -m1 -i 'contract violation\|mismatch' \
  "${legdir}/stderr.log" 2>/dev/null | tail -c 300)"
say "e147_rungE1b_control_exit=${ctl_status}"
say "e147_rungE1b_control_all_tokens_matched=${ctl_m:-none}"
say "e147_rungE1b_control_residual_divergence_count=${ctl_r:-none}"
say "e147_rungE1b_control_diagnostic=${ctl_diag:-none}"
caught=0
if [[ "${ctl_status}" != "0" || "${ctl_m}" == "false" \
      || ( -n "${ctl_r}" && "${ctl_r}" != "0" ) ]]; then
  caught=1
fi
say "e147_rungE1b_positive_control_caught=${caught}"

echo "=== e147_rungE1b phase 3: restore the submitted default ==="
restore_tree
trap - EXIT
dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
say "e147_rungE1b_tree_restored=$([[ -z "${dirty}" ]] && echo true || echo false)"
python3 research/twin_audit.py > /dev/null \
  && say "e147_rungE1b_twin_audit_after=ok" \
  || say "e147_rungE1b_twin_audit_after=FAIL"
cp -c "${bin_root}/worker.noretile" "${worker}" 2>/dev/null \
  || cp "${bin_root}/worker.noretile" "${worker}"
chmod +x "${worker}"
say "e147_rungE1b_worker_left_installed=$(sha_of "${worker}")"

verdict=FAIL
if [[ ${matched_all} == 1 && ${caught} == 1 ]]; then verdict=PASS; fi
say "e147_rungE1b_verdict=${verdict}"
[[ "${verdict}" == "PASS" ]]
