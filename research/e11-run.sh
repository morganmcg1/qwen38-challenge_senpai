#!/usr/bin/env bash
# Research-only (qwen38-r1-e11-depth-lever-showdown): run one or more 512-token
# --local-iterate arms from prebuilt, hash-pinned binary pairs.
#
#   research/e11-run.sh LABEL=BINARM [LABEL=BINARM ...]
#
# e.g. research/e11-run.sh C1=C C2=C   (the noise-floor pair: same binary twice)
#      research/e11-run.sh H=H K=K
#
# The label names the MEASUREMENT, the binary names the BUILD, so a repeat of
# the control is expressed without pretending it is a different build. Every
# gate benchmark-qwen-mtp.sh owns (drift tripwire, orphan scan, run lock, 40C
# cool gate, report seals) runs unmodified.
#
# Hashes are re-verified at install time and recorded per label, because the
# whole experiment collapses if two arms that must differ ran the same bytes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repo_root="${PWD}"
# Overridable so a later experiment can drive the same runner over its own arm
# set without its labels colliding with E11's.
bins_root="${E11_BINS_ROOT:-${repo_root}/.mlxfast-private/e11/bins}"
runs_root="${E11_RUNS_ROOT:-${repo_root}/.mlxfast-private/e11/runs}"
head_dir="${E11_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared}"
tokens="${E11_TOKENS:-512}"
# Defaulted rather than passed in, because run_job launches an argv with no
# environment of its own: an arm that silently fell back to the copy fixture
# would answer a different question with the same label.
golden="${E11_GOLDEN:-.mlxfast-private/e11/goldens/e11_prose_512_${tokens}.json}"

# all_tokens_matched compares only as many tokens as the golden carries, so a
# short golden reports an exact match over a prefix of the decode window.
golden_steps="$(python3 -c 'import json,sys; print(min(len(c["expected_tokens"]) for c in json.load(open(sys.argv[1]))["cases"]))' "${golden}" 2>/dev/null || echo 0)"
if ((golden_steps < tokens)); then
  echo "e11-run: golden ${golden} covers ${golden_steps} of ${tokens} decode tokens" >&2
  exit 1
fi

# Two passes, and the difference matters.
#
# TIMED (E11_TRACE unset, the default): every MLX_QWEN_MTP_* name is cleared,
# including the trace. The h-curve arms have to prove the DEFAULT flipped, so a
# leaked override would make H unfalsifiable; and since PR #2 the trace gate
# also buys per-round file I/O inside the timed round, which a headline number
# must not carry.
#
# FINGERPRINT (E11_TRACE=1): the same binary re-run with the phase trace on to
# recover the depth histogram, which score.json does not report. Never a source
# of headline timing. The loop is closed by checking the fingerprint pass's
# histogram mean against the timed pass's effective_mean_draft_len.
for v in $(env | sed -n 's/^\(MLX_QWEN_MTP_[A-Z_]*\)=.*/\1/p'); do unset "${v}"; done

# Per-arm thermal witness. benchmark.sh already gates entry at <=40C, but it
# gates only ITS OWN runs: foreign GPU load on this host has been seen to hold
# 65-83C / 16-31W while an arm was resident, which is exactly the confound that
# makes two arms incomparable. Sampling the same reader benchmark.sh uses,
# before and after each arm, turns that into recorded evidence instead of a
# guess.
macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W all_power=\(.all_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

status=0
for spec in "$@"; do
  label="${spec%%=*}"
  arm="${spec#*=}"
  src="${bins_root}/${arm}"
  out="${runs_root}/${label}"

  if [[ ! -f "${src}/sha256.txt" ]]; then
    echo "e11-run: no built binaries for arm ${arm} (${src})" >&2
    status=1; break
  fi

  rm -rf "${out}"; mkdir -p "${out}/reports"

  install -m 755 "${src}/mlxfast-swift" "${repo_root}/.build/release/mlxfast-swift"
  install -m 755 "${src}/mlxfast-runtime-worker" \
    "${repo_root}/.build-worker/release/mlxfast-runtime-worker"

  installed_cli="$(shasum -a 256 "${repo_root}/.build/release/mlxfast-swift" | cut -d' ' -f1)"
  installed_worker="$(shasum -a 256 "${repo_root}/.build-worker/release/mlxfast-runtime-worker" | cut -d' ' -f1)"
  want_cli="$(awk '$2=="mlxfast-swift"{print $1}' "${src}/sha256.txt")"
  want_worker="$(awk '$2=="mlxfast-runtime-worker"{print $1}' "${src}/sha256.txt")"
  if [[ "${installed_cli}" != "${want_cli}" || "${installed_worker}" != "${want_worker}" ]]; then
    echo "e11-run: ${label}: installed hashes do not match arm ${arm}" >&2
    status=1; break
  fi

  export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
  export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${golden}"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
  if [[ -n "${E11_TRACE:-}" ]]; then
    # The worker sandbox denies file-write*, and the parent swallows worker
    # stderr, so the phase trace needs the documented local relaxation.
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${out}/trace.txt"
    export MLXFAST_NO_SANDBOX=1
  else
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH MLXFAST_NO_SANDBOX
  fi

  {
    echo "label=${label}"
    echo "arm=${arm}"
    echo "tokens=${tokens}"
    echo "head_dir=${head_dir}"
    echo "cli_sha256=${installed_cli}"
    echo "worker_sha256=${installed_worker}"
    echo "source_sha256=$(awk '$2=="source.swift"{print $1}' "${src}/sha256.txt")"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "pass=${E11_TRACE:+fingerprint}${E11_TRACE:-timed}"
    # Verbatim list, not a count of two known names: the H arms claim to need
    # NO research variable at all, and only the full list can show that.
    echo "mlx_qwen_env=$(env | sed -n 's/^\(MLX_QWEN_MTP_[A-Z_]*\)=.*/\1/p' \
      | sort | tr '\n' ',')"
    echo "golden=${MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE:-<default>}"
    echo "thermal_before=$(sample_thermal)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  echo "=== e11-run: ${label} (build ${arm}) ==="
  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?
  {
    echo "exit=${rc}"
    echo "thermal_after=$(sample_thermal)"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"
  if ((rc != 0)); then
    echo "e11-run: ${label}: benchmark exited ${rc}" >&2
    status=1; break
  fi
done
exit "${status}"
