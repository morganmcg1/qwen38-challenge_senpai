#!/usr/bin/env bash
# E87 rung 0: capture real MTP-head hidden states for the offline shortlist screen.
#
# Same two-stage contract as research/e82_screen.sh, with one addition: the
# `capture` stage sets MLX_E87_HIDDEN_DUMP so the research instrument in
# Qwen35.swift streams every `x` that reaches the draft readout, plus the
# proposal token the runtime actually returned for it.
#
#   plans      tokenize each seed into the `{seed_tokens, emitted}` plan the
#              MTP verbs take. No GPU work.
#   reference  mtp-verify --generate walks the SERIAL width-1 frame. The head
#              is loaded but never drafts, so the rows are the target's own
#              greedy chain. No instrument is needed here.
#   capture    mtp-verify --golden replays those rows with the declared head
#              drafting at depth 8. Untimed, so the instrument's host readback
#              costs nothing that matters.
#
# Usage:
#   research/e87_capture.sh plans
#   research/e87_capture.sh reference [--steps N] [--only A,B] [--limit N]
#   research/e87_capture.sh capture   [--steps N] [--depth D] [--only A,B] [--limit N]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
seeds_dir="${cache}/e87/corpus/seeds"
out="${cache}/e87/screen"
head_dir="${cache}/mtp-head-declared-run"
cli=.build/release/mlxfast-swift
manifest=research/e87-corpus-manifest.json

stage="${1:-}"; shift || true
steps=512
depth=8
only=""
limit=0
rebuild=0
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --only)  only="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    *) echo "e87-capture: unknown flag '$1'" >&2; exit 2 ;;
  esac
done
((rebuild)) && { research/e87_rebuild.sh || exit $?; }
ref_dir="${out}/reference"
dump_dir="${out}/hidden"

seed_names() {
  python3 - "$only" "$limit" "$manifest" <<'PY'
import json, sys
only, limit, manifest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
keep = set(filter(None, only.split(","))) if only else None
names = [s["name"] for s in json.load(open(manifest))["seeds"]
         if keep is None or s["name"] in keep]
# Interleave the works so any prefix of the list stays domain-balanced.
by_work: dict[str, list[str]] = {}
for n in names:
    by_work.setdefault(n.rsplit("-", 1)[0], []).append(n)
order, i = [], 0
while len(order) < len(names):
    for work in by_work:
        if i < len(by_work[work]):
            order.append(by_work[work][i])
    i += 1
print("\n".join(order[:limit] if limit else order))
PY
}

echo "e87-capture: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e87-capture: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e87-capture: head   $(git rev-parse HEAD)"

case "${stage}" in
plans)
  mkdir -p "${out}/plans"
  python3 - "${seeds_dir}" "${out}/plans" "${manifest}" <<'PY'
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

seeds_dir, plans_dir, manifest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tok = Tokenizer.from_file("weights/tokenizer.json")
for seed in json.load(open(manifest))["seeds"]:
    ids = tok.encode((seeds_dir / f"e82-{seed['name']}.txt").read_text(),
                     add_special_tokens=False).ids
    (plans_dir / f"{seed['name']}.json").write_text(
        json.dumps({"seed_tokens": ids, "emitted": []}))
print("plans written")
PY
  ;;
reference)
  mkdir -p "${ref_dir}"
  for name in $(seed_names); do
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
      || { echo "e87-capture: reference ${name} FAILED" >&2; rm -f "${dest}"; continue; }
    jq -e --argjson tokens "${steps}" '
        . as $g | ($g.rows | length) as $rows
        | $g.reference_self_consistent == true
          and ($g.emitted_tokens | length) == $rows
          and $rows >= ($tokens + 1)
          and ([range(0; $rows)
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
      ' "${dest}" >/dev/null || { echo "e87-capture: ${name} reference unusable" >&2; rm -f "${dest}"; continue; }
    echo "e87-capture: ${name} reference in $(( $(date +%s) - start ))s"
  done
  ;;
capture)
  mkdir -p "${dump_dir}" "${out}/verify"
  # `--golden` makes the CLI hand the worker a seatbelt profile that denies the
  # golden file AND carries a blanket `(deny file-write*)`, so the instrument's
  # FileHandle open fails and the dump is silently empty. The capture stage is
  # untimed research, and research/e79_trace_leg.sh already uses this same knob
  # for local legs, so drop the profile here and check the dump instead.
  export MLXFAST_NO_SANDBOX=1
  for name in $(seed_names); do
    golden="${ref_dir}/${name}_${steps}.json"
    [[ -s "${golden}" ]] || { echo "e87-capture: missing golden ${name}"; continue; }
    dest="${out}/verify/${name}.json"
    if [[ -s "${dest}" ]] && compgen -G "${dump_dir}/${name}.pid*.tok.i32" >/dev/null; then
      echo "=== skip capture ${name} ==="; continue
    fi
    echo "=== capture ${name} (depth ${depth}, ${steps} tokens) ==="
    start=$(date +%s)
    log="${out}/verify/${name}.leg.log"
    if MLX_E87_HIDDEN_DUMP="${dump_dir}/${name}" ${cli} mtp-verify \
      --golden "${golden}" \
      --mtp-head "${head_dir}" \
      --mtp-depth "${depth}" \
      --tokens "${steps}" \
      --output "${dest}" >"${log}" 2>&1
    then
      rm -f "${log}"
      jq -r '"e87-capture: parity=\(.parity_all_ok) matched=\(.all_tokens_matched)"
             + " rounds=\(.round_count) accept=\(.accepted_draft_rate)"
             + " meandraft=\(.effective_mean_draft_len)"
             + " head=\(.head_provenance.sha256[0:12])"' "${dest}"
      # A silently empty dump is the failure mode this stage exists to avoid,
      # so treat it exactly like a failed leg rather than reporting success.
      # The instrument shards by process, so sum the shards.
      tok_bytes=$(cat "${dump_dir}/${name}".pid*.tok.i32 2>/dev/null | wc -c | tr -d ' ')
      x_bytes=$(cat "${dump_dir}/${name}".pid*.x.f32 2>/dev/null | wc -c | tr -d ' ')
      shards=$(ls "${dump_dir}/${name}".pid*.tok.i32 2>/dev/null | wc -l | tr -d ' ')
      if ((tok_bytes == 0)) || ((x_bytes != tok_bytes / 4 * 5120 * 4)); then
        echo "e87-capture: ${name} DUMP UNUSABLE (tok=${tok_bytes} x=${x_bytes})" >&2
        rm -f "${dest}" "${dump_dir}/${name}".pid*
        continue
      fi
      echo "e87-capture: ${name} dumped $((tok_bytes / 4)) samples" \
           "in ${shards} shard(s) in $(( $(date +%s) - start ))s"
    else
      mv "${log}" "${out}/verify/${name}.failed.log"
      rm -f "${dest}" "${dump_dir}/${name}".pid*
      echo "e87-capture: ${name} FAILED" >&2
      tail -3 "${out}/verify/${name}.failed.log" >&2
    fi
  done
  ;;
*)
  echo "usage: research/e87_capture.sh {plans|reference|capture} [flags]" >&2
  exit 2
  ;;
esac
echo "e87-capture: ${stage} done"
