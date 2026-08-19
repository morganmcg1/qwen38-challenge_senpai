#!/usr/bin/env bash
# Mutation negative controls for benchmark-qwen-mtp.sh's metallib-freshness guard.
#
# WHY THIS EXISTS
#
# benchmark-qwen-mtp.sh reuses three definitions out of benchmark.sh by awk
# extraction -- RUNTIME_WORKER_BIN, MLX_METALLIB, and metallib_rebuild_required()
# -- so that the two scripts cannot drift apart about what "the metallib is
# stale" means. Reuse was the right call. The extraction, as originally written,
# failed silently and in the one direction that hurts:
#
#   an awk pattern that no longer matches emits nothing
#     -> `eval ""` succeeds
#     -> metallib_rebuild_required is undefined
#     -> `if metallib_rebuild_required` exits 127
#     -> `set -e` exempts an `if` condition, so nothing aborts
#     -> the branch reads FALSE
#     -> the rebuild is skipped without a word
#
# and skipping that rebuild is exactly the failure the block was written to
# prevent: a kernel edit gets timed against the PREVIOUS mlx.metallib, so a real
# decode change reads as noise and a correctness-breaking one never executes.
# Every local kernel measurement in this campaign depends on that rebuild firing.
#
# A guard against a silent failure must not itself be able to fail silently, and
# a guard written by the person who wrote the silent failure inherits their blind
# spot. So the guard is not trusted on inspection. It is mutated and made to fail.
#
# WHAT THIS CHECKS
#
# The guard block is extracted VERBATIM from benchmark-qwen-mtp.sh between its
# METALLIB-GUARD-BEGIN / METALLIB-GUARD-END markers, so these controls exercise
# the shipped code and not a copy of it. Each case builds a sandbox holding a
# possibly-mutated benchmark.sh, runs the real guard in it, and asserts an exit
# code. Case 0 additionally asserts that the guard is NECESSARY by running the
# pre-fix form of the block and showing that it passes the same mutation without
# a word -- a control on the absence of the guard, not just on its presence.
#
# Usage: research/metallib-guard-controls.sh
# Exit 0 iff every control behaves as designed.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

runner="benchmark-qwen-mtp.sh"
subject="benchmark.sh"

for f in "${runner}" "${subject}"; do
  if [[ ! -f "${f}" ]]; then
    echo "metallib-guard-controls: missing ${f}" >&2
    exit 1
  fi
done

# --- extract the shipped guard, fail closed on a marker rename ----------------
guard_body="$(awk '/# METALLIB-GUARD-BEGIN/{flag=1; next} /# METALLIB-GUARD-END/{flag=0} flag' "${runner}")"
if [[ -z "${guard_body}" ]]; then
  echo "metallib-guard-controls: could not extract the guard from ${runner};" >&2
  echo "metallib-guard-controls: the METALLIB-GUARD-BEGIN/END markers have moved -- refusing to test a guard I cannot find" >&2
  exit 1
fi
for required in metallib_reuse_definitions 'declare -F metallib_rebuild_required' RUNTIME_WORKER_BIN MLX_METALLIB; do
  if [[ "${guard_body}" != *"${required}"* ]]; then
    echo "metallib-guard-controls: extracted guard does not mention ${required};" >&2
    echo "metallib-guard-controls: the extraction is wrong or the guard has been gutted" >&2
    exit 1
  fi
done

# The pre-fix form, reconstructed exactly: bare eval, no checks. Used once, as a
# control proving the failure mode was real and silent.
prefix_body='eval "$(
  awk '"'"'/^RUNTIME_WORKER_BIN=/'"'"' benchmark.sh
  awk '"'"'/^MLX_METALLIB=/'"'"' benchmark.sh
  awk '"'"'/^metallib_rebuild_required\(\) \{/,/^\}/'"'"' benchmark.sh
)"'

sandbox_root="$(mktemp -d "${TMPDIR:-/tmp}/metallib-guard-controls.XXXXXX")"
trap 'rm -rf "${sandbox_root}"' EXIT

pass_count=0
fail_count=0

# build_sandbox <name> <mutation>
#   mutation: none | rename_fn | indent_worker_bin | indent_metallib | no_metallib_file
# Leaves the sandbox path on stdout.
build_sandbox() {
  local name="$1" mutation="$2"
  local dir="${sandbox_root}/${name}"
  mkdir -p "${dir}/.build-worker/release"
  mkdir -p "${dir}/Vendor/mlx-swift/Source/Cmlx/mlx"
  mkdir -p "${dir}/Vendor/mlx-swift/Source/Cmlx/mlx-generated"

  # Only the three reused definitions matter, and taking just them keeps the
  # sandbox small and the mutation unambiguous.
  {
    awk '/^RUNTIME_WORKER_BIN=/' "${repo_root}/${subject}"
    awk '/^MLX_METALLIB=/' "${repo_root}/${subject}"
    awk '/^metallib_rebuild_required\(\) \{/,/^\}/' "${repo_root}/${subject}"
  } > "${dir}/${subject}"

  case "${mutation}" in
    none) ;;
    rename_fn)
      # A plausible refactor: the function is renamed, everything else intact.
      sed -i '' 's/^metallib_rebuild_required() {/metallib_needs_rebuild() {/' "${dir}/${subject}"
      ;;
    indent_worker_bin)
      # A plausible reformat: the assignment moves inside a block and gains
      # leading whitespace, so /^RUNTIME_WORKER_BIN=/ stops matching.
      sed -i '' 's/^RUNTIME_WORKER_BIN=/  RUNTIME_WORKER_BIN=/' "${dir}/${subject}"
      ;;
    indent_metallib)
      sed -i '' 's/^MLX_METALLIB=/  MLX_METALLIB=/' "${dir}/${subject}"
      ;;
    no_metallib_file) ;;
    *)
      echo "metallib-guard-controls: unknown mutation ${mutation}" >&2
      exit 1
      ;;
  esac

  if [[ "${mutation}" != "no_metallib_file" ]]; then
    : > "${dir}/.build-worker/release/mlx.metallib"
  fi
  printf '%s\n' "${dir}"
}

# run_guard <dir> <body> -> exit code, output on stderr captured to a file
run_guard() {
  local dir="$1" body="$2" out="$3"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' "${body}"
    # The consequence under test: does the rebuild branch fire, and is its
    # decision reported at all? "SKIPPED-SILENTLY" is the outcome the guard
    # exists to make impossible.
    printf '%s\n' 'if metallib_rebuild_required; then echo "DECISION=rebuild"; else echo "DECISION=skip"; fi'
  } > "${dir}/guard.sh"
  ( cd "${dir}" && bash guard.sh ) > "${out}" 2>&1
  return $?
}

check() {
  local label="$1" expected="$2" actual="$3" detail="$4"
  if [[ "${expected}" == "${actual}" ]]; then
    printf '  PASS  %-58s %s\n' "${label}" "${detail}"
    pass_count=$((pass_count + 1))
  else
    printf '  FAIL  %-58s expected %s, got %s  %s\n' "${label}" "${expected}" "${actual}" "${detail}"
    fail_count=$((fail_count + 1))
  fi
}

echo "metallib-freshness guard: mutation negative controls"
echo "  subject : ${runner} (guard extracted between its own markers)"
echo "  reused  : ${subject} RUNTIME_WORKER_BIN, MLX_METALLIB, metallib_rebuild_required()"
echo

# --- CONTROL 0: the guard is NECESSARY ---------------------------------------
# Pre-fix block, function renamed away. If this exits 0 and reports a decision
# of "skip", the original code silently declined to rebuild -- which is the whole
# reason the guard was added. A PASS here is a demonstration of the old bug.
dir="$(build_sandbox prefix_rename_fn rename_fn)"
out="${dir}/out.txt"
run_guard "${dir}" "${prefix_body}" "${out}"
rc=$?
decision="$(grep -o 'DECISION=[a-z]*' "${out}" | tail -1 || true)"
check "control 0a  PRE-FIX code, function renamed: exits 0" "0" "${rc}" "(the old silent pass)"
check "control 0b  PRE-FIX code, function renamed: decides skip" "DECISION=skip" "${decision}" "(rebuild silently declined)"

# --- CONTROL 1: unmutated benchmark.sh must pass -----------------------------
dir="$(build_sandbox guard_clean none)"
out="${dir}/out.txt"
run_guard "${dir}" "${guard_body}" "${out}"
rc=$?
check "control 1   guard, unmutated benchmark.sh: exits 0" "0" "${rc}" "(no false alarm)"

# --- CONTROL 2: each mutation must be refused, FOR THE RIGHT REASON ----------
# Exit 1 alone is too weak an assertion: a syntax error in the guard itself would
# also exit 1 and would look like a pass. So each case must additionally emit the
# guard's own diagnostic, and must NOT reach a rebuild decision at all. (One
# mutation also trips bash's own `set -u` message from inside the eval, which
# prints first; that is fine, but it must not be the only thing printed.)
for mutation in rename_fn indent_worker_bin indent_metallib no_metallib_file; do
  dir="$(build_sandbox "guard_${mutation}" "${mutation}")"
  out="${dir}/out.txt"
  run_guard "${dir}" "${guard_body}" "${out}"
  rc=$?
  own_msg="absent"
  if grep -q '^benchmark-qwen-mtp\.sh: ' "${out}"; then own_msg="present"; fi
  reached="no"
  if grep -q 'DECISION=' "${out}"; then reached="yes"; fi
  diag="$(grep -m1 '^benchmark-qwen-mtp\.sh: ' "${out}" | cut -c1-64)"
  check "control 2a  refuses mutation ${mutation}: exit 1" "1" "${rc}" ""
  check "control 2b  refuses mutation ${mutation}: own diagnostic" "present" "${own_msg}" "${diag}"
  check "control 2c  refuses mutation ${mutation}: no decision reached" "no" "${reached}" ""
done

# --- CONTROL 3: the guard is not over-strict under an explicit override ------
# With MLXFAST_MLX_METALLIB set, the caller owns the artifact and the reused
# function returns 1 by design without ever reading MLX_METALLIB, so the path
# arms are correctly not load-bearing. This must pass even with a lost arm.
dir="$(build_sandbox guard_override_lost_arm indent_worker_bin)"
out="${dir}/out.txt"
export MLXFAST_MLX_METALLIB="${dir}/caller-owned.metallib"
run_guard "${dir}" "${guard_body}" "${out}"
rc=$?
decision="$(grep -o 'DECISION=[a-z]*' "${out}" | tail -1 || true)"
unset MLXFAST_MLX_METALLIB
check "control 3a  override set, lost path arm: exits 0" "0" "${rc}" "(caller owns the artifact)"
check "control 3b  override set: reused function declines rebuild" "DECISION=skip" "${decision}" "(by design in benchmark.sh)"

# --- CONTROL 4: the reused function still WORKS, not merely parses -----------
# Syntactic reuse is not semantic reuse. Drive the real staleness decision both
# ways by touch ordering.
dir="$(build_sandbox guard_semantics none)"
out="${dir}/out.txt"
: > "${dir}/Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
# metallib newer than sources -> fresh
touch "${dir}/Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
sleep 1
touch "${dir}/.build-worker/release/mlx.metallib"
run_guard "${dir}" "${guard_body}" "${out}"
rc=$?
decision="$(grep -o 'DECISION=[a-z]*' "${out}" | tail -1 || true)"
check "control 4a  metallib newer than kernel source: skip" "DECISION=skip" "${decision}" "rc=${rc}"
# kernel source newer than metallib -> stale
sleep 1
touch "${dir}/Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
run_guard "${dir}" "${guard_body}" "${out}"
rc=$?
decision="$(grep -o 'DECISION=[a-z]*' "${out}" | tail -1 || true)"
check "control 4b  kernel source newer than metallib: rebuild" "DECISION=rebuild" "${decision}" "rc=${rc}"

echo
echo "  ${pass_count} passed, ${fail_count} failed"
if (( fail_count > 0 )); then
  echo "metallib-freshness guard controls: FAIL"
  exit 1
fi
echo "metallib-freshness guard controls: PASS"
