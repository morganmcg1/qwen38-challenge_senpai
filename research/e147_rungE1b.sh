#!/usr/bin/env bash
# E147 rung E-1b: does the retiled qmm_t reproduce the pinned golden rows?
#
#   usage: research/e147_rungE1b.sh build   # build the three arms, then restore
#          research/e147_rungE1b.sh legs    # run the golden legs on those arms
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
# WHY BUILD AND LEGS ARE SEPARATE COMMANDS. Selecting an arm means editing the
# checked-in header and its generated twin, so the working tree is dirty from
# the moment an arm is applied until it is restored. `build` applies all three
# arms back to back, copies each binary out, and restores the tree before it
# returns, so the dirty window is bounded by the compiler and never spans a
# measurement. `legs` then runs entirely on cached binaries with a clean tree.
#
# RULE 101 POSITIVE CONTROL. A green exactness result is worthless unless the
# same harness can go red on the same code path. `build` produces a third
# worker whose only difference from the retile arm is a defective tile decode.
#
# The first attempt at this control rotated the column index by one step. That
# control did not fire, and the reason is arithmetic, not dispatch: the lambda
# `compute_tile(row, col)` reads its operands at `(row, col)` and stores its
# result at `(row, col)`, so ANY bijection from the flattened tile index onto
# the tile set leaves the output identical. Rotating the column inside a row
# block is exactly such a bijection, so it only changed which threadgroup owned
# which tile. Every consistent permutation of the tile assignment is a no-op
# here, so the control must instead break coverage. The shipped control maps
# two source tiles onto one column block, which computes the lower half of the
# column blocks twice and never writes the upper half, while keeping every
# pointer in range. `research/e147-rungE1b-rotation-control.txt` keeps the
# evidence from the defective control.
#
# THERMAL MODE. Nothing here is timed. Exactness does not depend on the cool
# gate, and no number this rung produces is a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:-}"
case "${mode}" in
  build | legs) ;;
  *)
    echo "usage: research/e147_rungE1b.sh {build|legs}" >&2
    exit 2
    ;;
esac

prompts=(beagle_a essays_montaigne)
control_prompt=beagle_a
tokens=512
depth=8

bin_root=".mlxfast-private/e147/bin"
runs_parent=".mlxfast-private/e147/runs"
goldens_dir="${E147_GOLDENS_DIR:-.mlxfast-private/e128/goldens}"
worker=".build-worker/release/mlxfast-runtime-worker"
arms="${bin_root}/arms.txt"
out="research/e147-rungE1b.txt"

header="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

ARM_OFF='constexpr bool kE147RetileOn = false;'
ARM_ON='constexpr bool kE147RetileOn = true;'
TILE_OK='compute_tile((t / tiles_x) * BM, (t % tiles_x) * BN);'
TILE_BAD='compute_tile((t / tiles_x) * BM, ((t % tiles_x) / 2) * BN);'

scored_dirty() {
  git status --porcelain -- Sources Vendor Package.swift Package.resolved \
    mtp-head.manifest.json
}

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }
install_worker() {
  cp -c "$1" "${worker}" 2>/dev/null || cp "$1" "${worker}"
  chmod +x "${worker}"
}

mkdir -p "${bin_root}"

# ---------------------------------------------------------------- build mode

if [[ "${mode}" == "build" ]]; then
  dirty="$(scored_dirty)"
  if [[ -n "${dirty}" ]]; then
    echo "e147_rungE1b: scored surface is dirty; refusing to build arms" >&2
    echo "${dirty}" >&2
    exit 1
  fi
  session_commit="$(git rev-parse HEAD)"
  restore_tree() { git checkout "${session_commit}" -- "${header}" "${twin}"; }
  trap 'restore_tree' EXIT

  # Every edit below must hit exactly one site in each of the two files. A
  # drifted tree therefore stops the rung instead of building a half-applied
  # arm.
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

  echo "=== e147_rungE1b build 0a: arm OFF worker (submitted default) ==="
  senpai/rebuild-and-assert-worker.sh \
    --require "${ARM_OFF}" --forbid "${ARM_ON}" || {
    echo "e147_rungE1b: the OFF polarity assertion failed" >&2
    exit 3
  }
  cp -c "${worker}" "${bin_root}/worker.noretile" 2>/dev/null \
    || cp "${worker}" "${bin_root}/worker.noretile"
  off_sha="$(sha_of "${bin_root}/worker.noretile")"

  echo "=== e147_rungE1b build 0b: arm ON worker (64x16 retile) ==="
  edit_both "${ARM_OFF}" "${ARM_ON}" || exit 1
  python3 research/twin_audit.py || {
    echo "e147_rungE1b: twin audit failed after the ON edit" >&2
    exit 3
  }
  senpai/rebuild-and-assert-worker.sh \
    --require "${ARM_ON}" --forbid "${ARM_OFF}" || {
    echo "e147_rungE1b: the ON polarity assertion failed" >&2
    exit 3
  }
  cp -c "${worker}" "${bin_root}/worker.retile" 2>/dev/null \
    || cp "${worker}" "${bin_root}/worker.retile"
  on_sha="$(sha_of "${bin_root}/worker.retile")"

  echo "=== e147_rungE1b build 0c: Rule 101 control (halved columns) ==="
  edit_both "${TILE_OK}" "${TILE_BAD}" || exit 1
  senpai/rebuild-and-assert-worker.sh \
    --require "${TILE_BAD}" --forbid "${TILE_OK}" --require "${ARM_ON}" || {
    echo "e147_rungE1b: the control polarity assertion failed" >&2
    exit 3
  }
  cp -c "${worker}" "${bin_root}/worker.control" 2>/dev/null \
    || cp "${worker}" "${bin_root}/worker.control"
  ctl_sha="$(sha_of "${bin_root}/worker.control")"

  echo "=== e147_rungE1b build 0d: restore the submitted default ==="
  restore_tree
  trap - EXIT
  dirty="$(scored_dirty)"
  tree_restored=$([[ -z "${dirty}" ]] && echo true || echo false)
  audit=$(python3 research/twin_audit.py > /dev/null && echo ok || echo FAIL)
  install_worker "${bin_root}/worker.noretile"

  {
    echo "session_commit=${session_commit}"
    echo "off_sha=${off_sha}"
    echo "on_sha=${on_sha}"
    echo "ctl_sha=${ctl_sha}"
    echo "tree_restored=${tree_restored}"
    echo "twin_audit_after=${audit}"
  } > "${arms}"
  cat "${arms}"

  ok=1
  [[ "${off_sha}" != "${on_sha}" ]] || ok=0
  [[ "${ctl_sha}" != "${on_sha}" ]] || ok=0
  [[ "${tree_restored}" == "true" ]] || ok=0
  [[ "${audit}" == "ok" ]] || ok=0
  if [[ ${ok} != 1 ]]; then
    echo "e147_rungE1b: build phase did not produce three distinct clean arms" >&2
    exit 3
  fi
  echo "e147_rungE1b: build phase ok"
  exit 0
fi

# ----------------------------------------------------------------- legs mode

if [[ ! -f "${arms}" ]]; then
  echo "e147_rungE1b: ${arms} is missing; run 'build' first" >&2
  exit 2
fi
dirty="$(scored_dirty)"
if [[ -n "${dirty}" ]]; then
  echo "e147_rungE1b: scored surface is dirty; refusing to run legs" >&2
  echo "${dirty}" >&2
  exit 1
fi

session_commit="$(sed -n 's/^session_commit=//p' "${arms}")"
off_sha="$(sed -n 's/^off_sha=//p' "${arms}")"
on_sha="$(sed -n 's/^on_sha=//p' "${arms}")"
ctl_sha="$(sed -n 's/^ctl_sha=//p' "${arms}")"

: > "${out}"
say() {
  echo "$*"
  echo "$*" >> "${out}"
}

say "e147_rungE1b_session_commit=${session_commit}"
say "e147_rungE1b_head_at_legs=$(git rev-parse HEAD)"
say "e147_rungE1b_goldens_dir=${goldens_dir}"
say "e147_rungE1b_tokens=${tokens}"
say "e147_rungE1b_worker_off_sha256=${off_sha}"
say "e147_rungE1b_worker_on_sha256=${on_sha}"
say "e147_rungE1b_control_worker_sha256=${ctl_sha}"
say "e147_rungE1b_tree_restored=$(sed -n 's/^tree_restored=//p' "${arms}")"
say "e147_rungE1b_twin_audit_after=$(sed -n 's/^twin_audit_after=//p' "${arms}")"
say "e147_rungE1b_arm_contrast=$([[ "${off_sha}" != "${on_sha}" ]] \
     && echo ok || echo EMPTY)"
say "e147_rungE1b_control_differs_from_retile=$([[ "${ctl_sha}" != "${on_sha}" ]] \
     && echo true || echo false)"

echo "=== e147_rungE1b phase 1: exactness legs on the retile arm ==="
install_worker "${bin_root}/worker.retile"

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

echo "=== e147_rungE1b phase 2: Rule 101 positive control leg ==="
install_worker "${bin_root}/worker.control"
slot="rungE1b-control"
legdir="${runs_parent}/${slot}/${control_prompt}"
rm -rf "${legdir}"
E128_FORCE=1 E128_NO_TRACE=1 E128_TOKENS="${tokens}" E128_DEPTH="${depth}" \
  E128_ROOT=".mlxfast-private/e147" E128_GOLDENS_DIR="${goldens_dir}" \
  E128_RUNS_DIR="runs/${slot}" \
  research/e128_session.sh "${control_prompt}"
ctl_status=$?
meta="${legdir}/meta.txt"
ctl_got="$(sed -n 's/^worker_sha256=//p' "${meta}" 2>/dev/null)"
ctl_m="$(sed -n 's/^all_tokens_matched=//p' "${meta}" 2>/dev/null)"
ctl_r="$(sed -n 's/^residual_divergence_count=//p' "${meta}" 2>/dev/null)"
ctl_diag="$(grep -m1 -i 'contract violation\|mismatch' \
  "${legdir}/stderr.log" 2>/dev/null | tail -c 300)"
say "e147_rungE1b_control_exit=${ctl_status}"
say "e147_rungE1b_control_worker_is_control_arm=$([[ "${ctl_got}" == "${ctl_sha}" ]] \
     && echo true || echo false)"
say "e147_rungE1b_control_all_tokens_matched=${ctl_m:-none}"
say "e147_rungE1b_control_residual_divergence_count=${ctl_r:-none}"
say "e147_rungE1b_control_diagnostic=${ctl_diag:-none}"
caught=0
if [[ "${ctl_got}" == "${ctl_sha}" ]] \
  && { [[ "${ctl_status}" != "0" ]] || [[ "${ctl_m}" == "false" ]] \
    || { [[ -n "${ctl_r}" ]] && [[ "${ctl_r}" != "0" ]]; }; }; then
  caught=1
fi
say "e147_rungE1b_positive_control_caught=${caught}"

echo "=== e147_rungE1b phase 3: reinstall the submitted default ==="
install_worker "${bin_root}/worker.noretile"
say "e147_rungE1b_worker_left_installed=$(sha_of "${worker}")"
say "e147_rungE1b_scored_surface_clean_at_exit=$([[ -z "$(scored_dirty)" ]] \
     && echo true || echo false)"

verdict=FAIL
if [[ ${matched_all} == 1 && ${caught} == 1 ]]; then verdict=PASS; fi
say "e147_rungE1b_verdict=${verdict}"
[[ "${verdict}" == "PASS" ]]
