#!/usr/bin/env bash
# E147 rung A: does software-pipelining the affine qmm_t k-loop cut the charged
# 512-token seed prefill on hardware this team owns?
#
#   usage: research/e147_rungA_abba.sh [REPLICATES] [LABEL] [FIRST]
#
# A = base       the k-loop at E147_BASE_REF: two barriers, one staging tile
# B = candidate  the pipelined k-loop at HEAD: one barrier, two staging tiles
#
# WHY THIS RUNG EXISTS. The ranked seed prefill runs `affine_qmm_t_nax`, which
# needs `is_nax_available()`. This host is Mac16,11 / Apple M4 Pro /
# applegpu_g16s, where that predicate is false, so the ranked kernel never
# executes here and cannot be timed or checked locally. The non-NAX
# `affine_qmm_t` IS the local 512-token seed-prefill path, and it carries the
# same single-buffer k-loop shape. Rung A therefore buys two things the ranked
# port cannot buy for itself: a bit-exactness proof of the transformation, and
# a first measurement of whether overlapping the staging phase pays at all.
#
# HOW THE ARM IS SELECTED, and why it is not an environment variable. The
# runtime-effective source of the `quantized` family is the Metal JIT string
# compiled INTO the worker binary: `nojit_kernels.cpp` is excluded from the
# package, so `get_quantized_kernel` concatenates `metal::quantized()` from
# `mlx-generated/quantized.cpp` at run time. An arm is therefore a whole worker
# binary. Phase 0 builds both binaries once, asserts the pipelined needle is
# present in one and absent in the other, and every later leg only copies a
# saved binary into place. No leg rebuilds, and no leg edits the tree.
#
# ARM WITNESS (Rule 114). Each leg's own `meta.txt` records the sha256 of the
# worker that ran it. That digest is compared against the phase-0 digest of the
# requested arm, and a mismatch discards the leg. This is a witness taken from
# the run's own trace, not from the variable this script exported.
#
# THE REFERENCE IS ARM-INDEPENDENT. `mlxfast-swift` carries no quantized JIT
# string, so the CLI resolves the family from `.build/release/mlx.metallib`,
# which this script never republishes. The golden rows are produced by the
# unmodified kernel path and both arms are checked against them.
#
# THERMAL MODE. Order inside a replicate is A B B A per prompt, so both arms
# have mean position 2.5 and monotone drift cancels to first order. The real
# 40 C gate is not taken; `research/e128_session.sh` writes
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
# verbatim and records entry and exit GPU temperature for every leg. These legs
# are directional causal evidence inside one counterbalanced session and are
# never a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
label="${2:-s1}"
first="${3:-1}"

tokens=512
depth=8
prompts=(beagle_a essays_montaigne)
base_ref="${E147_BASE_REF:-bcc11dc6527ea4fa32be22c15b99a3a95695bfaf}"
bin_root=".mlxfast-private/e147/bin"
runs_parent=".mlxfast-private/e147/runs"

# Reference rows carried over from the E128 corpus session. They were written
# on 2026-08-23 at 00:00Z and 00:10Z, before the first E147 kernel commit
# 01ed58d5 (05:22Z), so their provenance is the BASE kernel and neither arm can
# have influenced them. `research/e128_session.sh` re-validates
# `reference_self_consistent` and the row count before it uses them, and it
# records `golden_sha256` in every leg's `meta.txt`.
goldens_dir="${E147_GOLDENS_DIR:-.mlxfast-private/e128/goldens}"

# The pipelined schedule's own line, 51 bytes, so it clears HARNESS DEFECT 38.
#
# RULE 101, WITH A REAL FAILURE. The first needle here was
# `loader_w.shift_dst(cur ? Ws_tile : -Ws_tile);`, and the base build refused it
# at 2026-08-23T05:35:55Z: `FAIL forbid ... found 2 copies, expected 0`. The
# reference implementation `fp_quantized_nax.h` ships that exact line, so it is
# in every worker and could never have separated the arms. The needle below
# reads the double-buffered mma consumption, which only the affine non-NAX
# `qmm_t_impl` written for this experiment contains:
#
#   Vendor/.../kernels/quantized.h        4
#   Vendor/.../mlx-generated/quantized.cpp 4
#   everything else under Vendor/          0
NEEDLE='mma_op.mma(Xs + cur * Xs_tile, Ws + cur * Ws_tile);'

kernel_paths=(
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp
)

worker=".build-worker/release/mlxfast-runtime-worker"

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e147_rungA: scored surface is dirty; refusing to build arms over" \
       "uncommitted work" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

restore_tree() {
  git checkout "${session_commit}" -- "${kernel_paths[@]}" 2>/dev/null
}

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

mkdir -p "${bin_root}"

echo "=== e147_rungA phase 0a: candidate worker (HEAD ${session_commit}) ==="
senpai/rebuild-and-assert-worker.sh --require "${NEEDLE}" || {
  echo "e147_rungA: the candidate worker does not carry the pipelined loop" >&2
  exit 3; }
cp -c "${worker}" "${bin_root}/worker.cand" 2>/dev/null \
  || cp "${worker}" "${bin_root}/worker.cand"
cand_sha="$(sha_of "${bin_root}/worker.cand")"

echo "=== e147_rungA phase 0b: base worker (${base_ref}) ==="
git checkout "${base_ref}" -- "${kernel_paths[@]}" || exit 1
trap 'restore_tree' EXIT
senpai/rebuild-and-assert-worker.sh --forbid "${NEEDLE}" \
  --require 'threadgroup_barrier(mem_flags::mem_threadgroup);' || {
  echo "e147_rungA: the base worker assertion failed" >&2
  restore_tree
  exit 3; }
cp -c "${worker}" "${bin_root}/worker.base" 2>/dev/null \
  || cp "${worker}" "${bin_root}/worker.base"
base_sha_bin="$(sha_of "${bin_root}/worker.base")"

restore_tree
trap - EXIT
dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e147_rungA: tree did not restore cleanly after the base build" >&2
  echo "${dirty}" >&2
  exit 1
fi

if [[ "${cand_sha}" == "${base_sha_bin}" ]]; then
  echo "e147_rungA: both arms produced the same binary; the contrast is empty" >&2
  exit 3
fi

echo "e147_rungA: session_commit=${session_commit}"
echo "e147_rungA: base_ref=${base_ref}"
echo "e147_rungA: worker_cand_sha256=${cand_sha}"
echo "e147_rungA: worker_base_sha256=${base_sha_bin}"

install_arm() {
  case "$1" in
    base) cp -c "${bin_root}/worker.base" "${worker}" 2>/dev/null \
            || cp "${bin_root}/worker.base" "${worker}" ;;
    cand) cp -c "${bin_root}/worker.cand" "${worker}" 2>/dev/null \
            || cp "${bin_root}/worker.cand" "${worker}" ;;
  esac
  chmod +x "${worker}"
}

want_sha_for() {
  [[ "$1" == "cand" ]] && echo "${cand_sha}" || echo "${base_sha_bin}"
}

failures=0
discarded=0

# RULE 137 WARMUP LEG. E130 rung 12 measured entry-temperature spread falling
# from 23.27 C to 1.757 C across 13 legs once one leg ran before the timed set,
# and gated per-leg noise of 0.052 % against 0.091 % ungated. One discarded leg
# is cheap next to a session whose first replicate reads a cold GPU.
#
# The warmup runs the SAME arm that precedes position 1 in steady state. From
# replicate 2 onward, position 1 (base) is preceded by the previous replicate's
# position 4, which is also base. Warming with base therefore makes replicate 1
# structurally identical to every later replicate instead of merely warmer.
#
# Its slot name does not start with ${label}, so research/e147_rungA_report.py
# cannot collect it: the leg is thermal conditioning and never evidence.
warmup_arm="${E147_WARMUP_ARM:-base}"
if [[ "${E147_WARMUP:-1}" == "1" ]]; then
  warmup_slot="warmup-${label}k${first}"
  warmup_id="${prompts[0]}"
  echo "=== e147_rungA ${warmup_slot}: prompt=${warmup_id}" \
       "arm=${warmup_arm} DISCARDED thermal conditioning ==="
  install_arm "${warmup_arm}"

  E128_FORCE=1 \
  E128_NO_TRACE=1 \
  E128_TOKENS="${tokens}" \
  E128_DEPTH="${depth}" \
  E128_ROOT=".mlxfast-private/e147" \
  E128_GOLDENS_DIR="${goldens_dir}" \
  E128_RUNS_DIR="runs/${warmup_slot}" \
    research/e128_session.sh "${warmup_id}"
  warmup_status=$?

  warmup_meta="${runs_parent}/${warmup_slot}/${warmup_id}/meta.txt"
  {
    echo "e147_warmup=1"
    echo "e147_arm_requested=${warmup_arm}"
    echo "e147_session_commit=${session_commit}"
  } >> "${warmup_meta}"
  echo "e147_rungA: warmup exited ${warmup_status};" \
       "$(sed -n 's/^gpu_temp_entry_c=/entry_c=/p;s/^gpu_temp_exit_c=/exit_c=/p' \
          "${warmup_meta}" 2>/dev/null | tr '\n' ' ')"

  # A failed warmup leaves the GPU cold, which is the condition this block
  # exists to remove. Do not silently time a cold session.
  if ((warmup_status != 0)); then
    echo "e147_rungA: warmup leg failed; refusing to time a cold session" >&2
    exit 4
  fi
else
  echo "e147_rungA: warmup leg skipped by E147_WARMUP=0 (Rule 137 not met)"
fi

for ((rep = first; rep < first + replicates; rep++)); do
  for id in "${prompts[@]}"; do
    position=0
    for arm in base cand cand base; do
      position=$((position + 1))
      slot="${label}k${rep}p${position}${arm}"
      out="${runs_parent}/${slot}/${id}"
      want_sha="$(want_sha_for "${arm}")"

      echo "=== e147_rungA ${slot}: prompt=${id} arm=${arm}" \
           "replicate=${rep} position=${position} ==="
      install_arm "${arm}"

      E128_FORCE=1 \
      E128_NO_TRACE=1 \
      E128_TOKENS="${tokens}" \
      E128_DEPTH="${depth}" \
      E128_ROOT=".mlxfast-private/e147" \
      E128_GOLDENS_DIR="${goldens_dir}" \
      E128_RUNS_DIR="runs/${slot}" \
        research/e128_session.sh "${id}"
      status=$?

      got_sha="$(sed -n 's/^worker_sha256=//p' "${out}/meta.txt" 2>/dev/null)"
      witness="ok"
      if [[ "${got_sha}" != "${want_sha}" ]]; then
        witness="MISMATCH"
        discarded=$((discarded + 1))
        echo "e147_rungA: ${slot} ${id} requested ${arm} but ran worker" \
             "${got_sha:-<none>}, wanted ${want_sha}" >&2
      fi
      {
        echo "e147_arm_requested=${arm}"
        echo "e147_arm_worker_sha256_want=${want_sha}"
        echo "e147_arm_worker_sha256_got=${got_sha:-none}"
        echo "e147_witness=${witness}"
        echo "e147_replicate=${rep}"
        echo "e147_position=${position}"
        echo "e147_session_commit=${session_commit}"
        echo "e147_base_ref=${base_ref}"
      } >> "${out}/meta.txt"

      if ((status != 0)); then
        echo "e147_rungA: ${slot} ${id} exited ${status}" >&2
        failures=$((failures + 1))
      fi
    done
  done
done

# Leave the candidate binary in place so a later gate chain does not time the
# base by accident.
install_arm cand

echo "e147_rungA: ${failures} failed legs, ${discarded} witness mismatches"
python3 research/e147_rungA_report.py --label "${label}" \
  --runs "${runs_parent}" --out research/e147-rungA.json
exit $(( failures > 0 ))
