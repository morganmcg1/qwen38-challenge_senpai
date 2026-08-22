#!/usr/bin/env bash
# E142 rung 0: capture the post-norm hidden rows that reach the SCORED verify
# readout, with the exact top-2 evidence the dense readout produced for them.
#
# The plans and the 512-token goldens this replay needs already exist under
# ${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/e133/screen, written by
# research/e133_capture.sh at the same corpus and the same 512-token window, so
# this script runs the capture stage only. Regenerating the goldens would cost
# a serial width-1 walk per seed and would produce the same rows.
#
# `--golden` replays the recorded serial trajectory with the declared head
# drafting under the shipped adaptive depth policy bounded by --mtp-depth, so
# the realised verify widths are the widths the scored path actually produces.
#
# Usage:
#   research/e142_capture.sh [--steps N] [--depth D] [--only A,B] [--limit N]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
src="${cache}/e133/screen"
out="${cache}/e142/verifyrows"
head_dir="${cache}/mtp-head-declared-run"
cli=.build/release/mlxfast-swift

steps=512
depth=8
only=""
limit=0
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --only)  only="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    *) echo "e142-capture: unknown flag '$1'" >&2; exit 2 ;;
  esac
done

dump_dir="${out}/hidden"
mkdir -p "${dump_dir}" "${out}/verify"

echo "e142-capture: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e142-capture: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e142-capture: head   $(git rev-parse HEAD)"
echo "e142-capture: HOME   ${HOME}"

seed_names() {
  if [[ -n "${only}" ]]; then
    tr ',' '\n' <<<"${only}"
  else
    ls "${src}/reference" | sed -n "s/_${steps}\.json$//p"
  fi
}

# `--golden` hands the worker a seatbelt profile carrying a blanket
# `(deny file-write*)`, so the instrument's FileHandle open fails and the dump
# is silently empty. The capture stage is untimed research, so drop the profile
# here and check the dump instead, exactly as E133 did.
export MLXFAST_NO_SANDBOX=1

count=0
for name in $(seed_names); do
  ((limit)) && ((count >= limit)) && break
  golden="${src}/reference/${name}_${steps}.json"
  [[ -s "${golden}" ]] || { echo "e142-capture: missing golden ${name}"; continue; }
  dest="${out}/verify/${name}.json"
  if [[ -s "${dest}" ]] && compgen -G "${dump_dir}/${name}.pid*.meta.i32" >/dev/null; then
    echo "=== skip capture ${name} ==="; count=$((count + 1)); continue
  fi
  echo "=== capture ${name} (depth ${depth}, ${steps} tokens) ==="
  # Shards are named by PID, so a retry after a partial run would leave orphan
  # shards from the old PIDs and inflate the sample count.
  rm -f "${dump_dir}/${name}".pid*
  start=$(date +%s)
  log="${out}/verify/${name}.leg.log"
  if MLX_E142_VERIFY_DUMP="${dump_dir}/${name}" ${cli} mtp-verify \
    --golden "${golden}" \
    --mtp-head "${head_dir}" \
    --mtp-depth "${depth}" \
    --tokens "${steps}" \
    --output "${dest}" >"${log}" 2>&1
  then
    rm -f "${log}"
    jq -r '"e142-capture: parity=\(.parity_all_ok) matched=\(.all_tokens_matched)"
           + " rounds=\(.round_count) accept=\(.accepted_draft_rate)"
           + " meandraft=\(.effective_mean_draft_len)"
           + " head=\(.head_provenance.sha256[0:12])"' "${dest}"
    meta_bytes=$(cat "${dump_dir}/${name}".pid*.meta.i32 2>/dev/null | wc -c | tr -d ' ')
    x_bytes=$(cat "${dump_dir}/${name}".pid*.x.f32 2>/dev/null | wc -c | tr -d ' ')
    tok_bytes=$(cat "${dump_dir}/${name}".pid*.tok.i32 2>/dev/null | wc -c | tr -d ' ')
    if ((meta_bytes == 0)) || ((x_bytes != tok_bytes / 4 * 5120 * 4)); then
      echo "e142-capture: ${name} DUMP UNUSABLE (meta=${meta_bytes}" \
           "tok=${tok_bytes} x=${x_bytes})" >&2
      rm -f "${dest}" "${dump_dir}/${name}".pid*
      continue
    fi
    echo "e142-capture: ${name} dumped $((meta_bytes / 4)) rounds," \
         "$((tok_bytes / 4)) rows in $(( $(date +%s) - start ))s"
    count=$((count + 1))
  else
    mv "${log}" "${out}/verify/${name}.failed.log"
    rm -f "${dest}" "${dump_dir}/${name}".pid*
    echo "e142-capture: ${name} FAILED" >&2
    tail -5 "${out}/verify/${name}.failed.log" >&2
  fi
done
echo "e142-capture: done (${count} seeds)"
