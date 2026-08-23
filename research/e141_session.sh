#!/usr/bin/env bash
# E141 rung 3: does widening the compact draft vocabulary raise acceptance?
#
# Three stages, the same contract research/e133_capture.sh uses.
#
#   plans      tokenize each seed into the `{seed_tokens, emitted}` plan the
#              MTP verbs take. No GPU work.
#   reference  mtp-verify --generate walks the SERIAL width-1 frame. The head
#              is loaded but never drafts, so the rows are the target's own
#              greedy chain and are the SAME golden for every arm.
#   verify     mtp-verify --golden replays those rows with the declared head
#              drafting, once per arm. Untimed, so the row ledger costs
#              nothing that matters.
#
# ARMS. One binary, one session. `MLX_E141_DRAFT_PREFIX` selects the compact
# draft prefix; unset is the shipped 98,304 arm bit for bit. Acceptance is a
# deterministic function of (arm, golden, depth), so the resolution floor of
# this contrast is exactly zero and a replicate is a reproducibility control
# rather than a noise estimate. The `repeat` stage runs that control.
#
# WITNESS (Rule 114). The arm is read out of the run's own row ledger, never
# out of the variable the leg was asked with: a draft row carrying a token id
# in [98_304, 248_044) is impossible in the shipped arm and proves the widened
# table was live. research/e141_rung3_analysis.py checks it.
#
# Usage:
#   research/e141_session.sh plans
#   research/e141_session.sh reference [--steps N] [--depth D]
#   research/e141_session.sh verify    [--steps N] [--depth D] [--arms A,B]
#   research/e141_session.sh all       [--steps N] [--depth D] [--arms A,B]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
out="${cache}/e141"
head_dir="${cache}/mtp-head-declared-run"
cli=.build/release/mlxfast-swift
manifest=research/e133-corpus-manifest.json

# The two median carriers of the published score (Rule 116: beagle 0.478,
# essays 0.522, all six others exactly 0.000).
seeds="beagle_a essays_montaigne"

stage="${1:-}"; shift || true
steps=512
depth=8
arms="shipped,full"
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --arms)  arms="$2"; shift 2 ;;
    --seeds) seeds="${2//,/ }"; shift 2 ;;
    *) echo "e141-session: unknown flag '$1'" >&2; exit 2 ;;
  esac
done
ref_dir="${out}/reference"
verify_dir="${out}/verify"

# An arm is `<prefix>` or `<prefix>@<probes>`. `shipped` exports nothing at
# all, so its leg is the compiled default. `full` widens the table and keeps
# the declared probe fraction, which is what rung 2 timed. `armA*` widens the
# table and pins the absolute probe count, so the row pass keeps the shipped
# byte cost and only the centroid pass grows.
arm_spec() {
  case "$1" in
    shipped)    echo "" ;;
    full)       echo "248320" ;;
    armA)       echo "248320@3073" ;;   # p25 shipped probe count, this base
    armA1844)   echo "248320@1844" ;;   # p15 shipped probe count, thorfinn
    *)          echo "$1" ;;
  esac
}

arm_prefix() { local s; s="$(arm_spec "$1")"; echo "${s%%@*}"; }
arm_probes() {
  local s
  s="$(arm_spec "$1")"
  [[ "${s}" == *@* ]] && echo "${s##*@}" || echo ""
}

echo "e141-session: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e141-session: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e141-session: head   $(git rev-parse HEAD)"
echo "e141-session: steps ${steps} depth ${depth} arms ${arms}"

case "${stage}" in
plans)
  mkdir -p "${out}/plans"
  python3 - "${out}/plans" "${manifest}" ${seeds} <<'PY'
import hashlib, json, sys
from pathlib import Path
from tokenizers import Tokenizer

plans_dir, manifest = Path(sys.argv[1]), sys.argv[2]
want = set(sys.argv[3:])
tok = Tokenizer.from_file("weights/tokenizer.json")
blob = json.load(open(manifest))
for seed in blob["seeds"]:
    if seed["id"] not in want:
        continue
    text = Path(seed["path"]).read_bytes()
    got = hashlib.sha256(text).hexdigest()
    if got != seed["sha256"]:
        raise SystemExit(f"{seed['id']}: sha256 {got} != manifest {seed['sha256']}")
    ids = tok.encode(text.decode(), add_special_tokens=False).ids
    if len(ids) != blob["seed_tokens"]:
        raise SystemExit(f"{seed['id']}: {len(ids)} tokens != {blob['seed_tokens']}")
    (plans_dir / f"{seed['id']}.json").write_text(
        json.dumps({"seed_tokens": ids, "emitted": []}))
    print(f"  {seed['id']:20s} {seed['domain']:10s} {len(ids)} tokens sha256 ok")
print("plans written")
PY
  ;;
reference)
  mkdir -p "${ref_dir}"
  for name in ${seeds}; do
    dest="${ref_dir}/${name}_${steps}.json"
    if [[ -s "${dest}" ]]; then echo "=== skip reference ${name} ==="; continue; fi
    echo "=== reference ${name} ($(( steps + 1 )) rows) ==="
    start=$(date +%s)
    ${cli} mtp-verify \
      --mtp-head "${head_dir}" \
      --emitted "${out}/plans/${name}.json" \
      --generate "$(( steps + 1 ))" \
      --mtp-depth "${depth}" \
      --output "${dest}" \
      --plan-output "${ref_dir}/${name}_${steps}.plan.json" \
      || { echo "e141-session: reference ${name} FAILED" >&2; rm -f "${dest}"; continue; }
    jq -e --argjson tokens "${steps}" '
        . as $g | ($g.rows | length) as $rows
        | $g.reference_self_consistent == true
          and ($g.emitted_tokens | length) == $rows
          and $rows >= ($tokens + 1)
          and ([range(0; $rows)
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
      ' "${dest}" >/dev/null \
      || { echo "e141-session: ${name} reference unusable" >&2; rm -f "${dest}"; continue; }
    echo "e141-session: ${name} reference in $(( $(date +%s) - start ))s"
  done
  ;;
verify|repeat)
  mkdir -p "${verify_dir}"
  suffix=""
  [[ "${stage}" == "repeat" ]] && suffix="_r2"
  for arm in ${arms//,/ }; do
    prefix="$(arm_prefix "${arm}")"
    nprobes="$(arm_probes "${arm}")"
    tag="${arm//@/p}"
    for name in ${seeds}; do
      golden="${ref_dir}/${name}_${steps}.json"
      [[ -s "${golden}" ]] || { echo "e141-session: missing golden ${name}"; continue; }
      dest="${verify_dir}/${name}_${tag}_${steps}${suffix}.json"
      if [[ -s "${dest}" ]]; then echo "=== skip verify ${name} ${arm} ==="; continue; fi
      echo "=== verify ${name} arm=${arm} prefix=${prefix:-unset}" \
           "probes=${nprobes:-declared} depth=${depth} ==="
      start=$(date +%s)
      log="${verify_dir}/${name}_${tag}${suffix}.leg.log"
      # The model runs in a sandboxed worker whose stderr is drained and shown
      # only on failure, so the arm witness needs its own file.
      witness="${verify_dir}/${name}_${tag}_${steps}${suffix}.arm.txt"
      rm -f "${witness}"
      if env ${prefix:+MLX_E141_DRAFT_PREFIX=${prefix}} \
        ${nprobes:+MLX_E141_PROBES=${nprobes}} \
        MLX_E141_WITNESS_FILE="${witness}" ${cli} mtp-verify \
        --golden "${golden}" \
        --mtp-head "${head_dir}" \
        --mtp-depth "${depth}" \
        --tokens "${steps}" \
        --output "${dest}" >"${log}" 2>&1
      then
        mv "${log}" "${verify_dir}/${name}_${tag}${suffix}.ok.log"
        if [[ -s "${witness}" ]]; then
          sort -u "${witness}"
        else
          echo "e141-session: NO ARM WITNESS for ${name} ${arm}" >&2
        fi
        jq -r '"e141-session: parity=\(.parity_all_ok) matched=\(.all_tokens_matched)"
               + " rounds=\(.round_count) accept=\(.accepted_draft_rate)"
               + " acc=\(.accepted_draft_total) rej=\(.rejected_draft_total)"
               + " tail=\(.target_tail_total) div=\(.residual_divergence_count)"
               + " head=\(.head_provenance.sha256[0:12])"' "${dest}"
        echo "e141-session: ${name} ${arm} in $(( $(date +%s) - start ))s"
      else
        mv "${log}" "${verify_dir}/${name}_${arm}${suffix}.failed.log"
        rm -f "${dest}"
        echo "e141-session: ${name} ${arm} FAILED" >&2
        tail -20 "${verify_dir}/${name}_${arm}${suffix}.failed.log" >&2
      fi
    done
  done
  ;;
all)
  flags=(--steps "${steps}" --depth "${depth}" --arms "${arms}" --seeds "${seeds// /,}")
  "$0" plans --seeds "${seeds// /,}" || exit $?
  "$0" reference "${flags[@]}" || exit $?
  "$0" verify "${flags[@]}" || exit $?
  ;;
*)
  echo "usage: research/e141_session.sh {plans|reference|verify|repeat|all} [flags]" >&2
  exit 2
  ;;
esac
