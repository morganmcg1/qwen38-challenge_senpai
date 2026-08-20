#!/usr/bin/env bash
# E82 rung 0: the untimed acceptance screen.
#
# Three stages, following the same reference contract benchmark-qwen-mtp.sh
# uses, because the reference trajectory must not depend on the head:
#
#   plans      tokenize each seed to the `{seed_tokens, emitted}` plan the MTP
#              verbs take. The CLI accepts token ids only. This stage runs no
#              GPU work; its tokenization was checked byte-identical to the
#              trusted CLI's own on seed candle-0.
#   reference  mtp-verify --generate walks the SERIAL width-1 frame and writes
#              the reference rows. The head is loaded but never drafts, so the
#              rows are the target's own greedy chain and every arm is later
#              scored against the same token stream.
#   verify     mtp-verify --golden replays those rows under each head at the
#              requested depth. It is untimed, so no thermal gate applies and
#              the numbers are acceptance evidence only -- never a speedup.
#
# Usage:
#   research/e82_screen.sh plans
#   research/e82_screen.sh reference [--steps N] [--head ARM] [--tag T] [--only S]
#   research/e82_screen.sh verify    [--steps N] [--depth D] [--arms A,..] [--only S]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
seeds_dir="${cache}/e82/corpus/seeds"
out="${cache}/e82/screen"
cli=.build/release/mlxfast-swift

# Arm -> head run tree. Every arm carries the byte-identical affine-2 draft
# readout except `pinned`, which is the organizer default and is included so
# this screen ties back to the E79 timed numbers.
declare -a ARM_ORDER=(declared soup-q4 qat-q4 master-bf16 kamciosz pinned)
arm_head() {
  case "$1" in
    declared)    echo "${cache}/mtp-head-declared-run" ;;
    soup-q4)     echo "${cache}/e82/built/e82-soup-q4-run" ;;
    qat-q4)      echo "${cache}/e82/built/e82-qat-q4-run" ;;
    master-bf16) echo "${cache}/e82/built/e82-master-bf16-run" ;;
    kamciosz)    echo "${cache}/e82/built/e82-kamciosz-run" ;;
    pinned)      echo "${cache}/mtp-head" ;;
    # rung 6: the best group-wise affine-4 g64 estimator we can build.
    best)        echo "${cache}/e82/built/e82-master-best-run" ;;
    # island allocation. Both trunks are byte-identical to `declared`, so the
    # only difference from it is which BF16 correction rows ship.
    noislands)   echo "${cache}/e82/built/e82-raw-noislands-run" ;;
    qonly)       echo "${cache}/e82/built/e82-raw-qonly-run" ;;
    *) echo "e82-screen: unknown arm '$1'" >&2; return 1 ;;
  esac
}

stage="${1:-}"; shift || true
steps=512
depth=8
only=""
ref_head=declared
tag=""
arms="$(IFS=,; echo "${ARM_ORDER[*]}")"
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --only)  only="$2"; shift 2 ;;
    --arms)  arms="$2"; shift 2 ;;
    --head)  ref_head="$2"; shift 2 ;;
    --tag)   tag="$2"; shift 2 ;;
    *) echo "e82-screen: unknown flag '$1'" >&2; exit 2 ;;
  esac
done
ref_dir="${out}/reference${tag:+-${tag}}"

seed_names() {
  python3 - "$only" <<'PY'
import json, sys
only = sys.argv[1]
keep = set(filter(None, only.split(","))) if only else None
for s in json.load(open("research/e82-corpus-manifest.json"))["seeds"]:
    if keep is None or s["name"] in keep:
        print(s["name"])
PY
}

echo "e82-screen: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e82-screen: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e82-screen: head   $(git rev-parse HEAD)"

case "${stage}" in
plans)
  mkdir -p "${out}/plans"
  python3 - "${seeds_dir}" "${out}/plans" <<'PY'
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

seeds_dir, plans_dir = Path(sys.argv[1]), Path(sys.argv[2])
tok = Tokenizer.from_file("weights/tokenizer.json")
for seed in json.load(open("research/e82-corpus-manifest.json"))["seeds"]:
    ids = tok.encode((seeds_dir / f"e82-{seed['name']}.txt").read_text(), add_special_tokens=False).ids
    dest = plans_dir / f"{seed['name']}.json"
    dest.write_text(json.dumps({"seed_tokens": ids, "emitted": []}))
    print(f"{seed['name']:28s} {len(ids):4d} seed tokens -> {dest}")
PY
  ;;
reference)
  mkdir -p "${ref_dir}"
  head_dir="$(arm_head "${ref_head}")" || exit 2
  for name in $(seed_names); do
    dest="${ref_dir}/${name}_${steps}.json"
    if [[ -s "${dest}" ]]; then echo "=== skip reference ${name} ==="; continue; fi
    echo "=== reference ${name} ($(( steps + 1 )) rows, head ${ref_head}) ==="
    start=$(date +%s)
    ${cli} mtp-verify \
      --mtp-head "${head_dir}" \
      --emitted "${out}/plans/${name}.json" \
      --generate "$(( steps + 1 ))" \
      --mtp-depth "${depth}" \
      --output "${dest}" \
      --plan-output "${ref_dir}/${name}_${steps}.plan.json" \
      || { echo "e82-screen: reference ${name} FAILED" >&2; exit 1; }
    # The wrapper's own usability preflight: self-consistency alone does not
    # prove the recorded rows agree with the chain they were generated against.
    jq -e --argjson tokens "${steps}" '
        . as $g | ($g.rows | length) as $rows
        | $g.reference_self_consistent == true
          and ($g.emitted_tokens | length) == $rows
          and $rows >= ($tokens + 1)
          and ([range(0; $rows)
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
      ' "${dest}" >/dev/null || { echo "e82-screen: ${name} reference unusable" >&2; exit 1; }
    echo "e82-screen: ${name} reference in $(( $(date +%s) - start ))s"
  done
  ;;
verify)
  mkdir -p "${out}/verify"
  IFS=, read -r -a arm_list <<<"${arms}"
  # A leg can abort on a contract violation the arm did not cause -- a near-tie
  # seed argmax makes the block-session prefill disagree with the reference
  # prefill, for instance. Record such a leg and keep going: losing the other 65
  # legs of a 2-hour sweep to one bad seed destroys more evidence than it saves.
  declare -a failed=()
  for arm in "${arm_list[@]}"; do
    head_dir="$(arm_head "${arm}")" || exit 2
    mkdir -p "${out}/verify/${arm}"
    for name in $(seed_names); do
      golden="${ref_dir}/${name}_${steps}.json"
      [[ -s "${golden}" ]] || { echo "e82-screen: missing ${golden}" >&2; exit 1; }
      dest="${out}/verify/${arm}/${name}.json"
      if [[ -s "${dest}" ]]; then echo "=== skip ${arm}/${name} ==="; continue; fi
      echo "=== verify ${arm} / ${name} (depth ${depth}, ${steps} tokens) ==="
      start=$(date +%s)
      log="${out}/verify/${arm}/${name}.leg.log"
      if ${cli} mtp-verify \
        --golden "${golden}" \
        --mtp-head "${head_dir}" \
        --mtp-depth "${depth}" \
        --tokens "${steps}" \
        --output "${dest}" >"${log}" 2>&1
      then
        rm -f "${log}" "${out}/verify/${arm}/${name}.failed.log"
        jq -r '"e82-screen: parity=\(.parity_all_ok) matched=\(.all_tokens_matched)"
               + " rounds=\(.round_count) accept=\(.accepted_draft_rate)"
               + " meandraft=\(.effective_mean_draft_len) rows=\(.reference_checked_row_total)"
               + " head=\(.head_provenance.sha256[0:12])"' "${dest}"
      else
        mv "${log}" "${out}/verify/${arm}/${name}.failed.log"
        rm -f "${dest}"
        failed+=("${arm}/${name}")
        echo "e82-screen: ${arm}/${name} FAILED:" >&2
        tail -2 "${out}/verify/${arm}/${name}.failed.log" >&2
      fi
      echo "e82-screen: ${arm}/${name} in $(( $(date +%s) - start ))s"
    done
  done
  printf '%s\n' "${failed[@]:-}" | jq -Rn '[inputs | select(length > 0)]' \
    >"${out}/verify/failures.json"
  if ((${#failed[@]})); then
    echo "e82-screen: ${#failed[@]} leg(s) failed: ${failed[*]}" >&2
  fi
  ;;
*)
  echo "usage: research/e82_screen.sh {plans|reference|verify} [flags]" >&2
  exit 2
  ;;
esac
echo "e82-screen: ${stage} done"
