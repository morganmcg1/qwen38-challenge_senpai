#!/usr/bin/env bash
# E133 rung 1: capture real MTP-head hidden states for the C1 shortlist screen.
#
# Same three-stage contract as research/e87_capture.sh, with one change that is
# scientific rather than cosmetic: the seed corpus is the E124 median-regime
# corpus, not the E87 Gutenberg corpus.
#
# WHY THE CORPUS CHANGED. The E87 corpus lives under student-qwen-thorfinn's
# home and every seed path in research/e87-corpus-manifest.json points there,
# so neither its hidden states nor its seed TEXTS exist on this machine.
# Rebuilding it would need a Gutenberg re-download, and it would still contain
# no beagle-domain seed. Finding 153 says beagle occupies a ranked median order
# statistic in 97.9 % of runs and carries the lowest per-step acceptance, so
# beagle is the deciding regime for any shortlist approximation. The E124
# corpus is committed text in this repository, is sha256-pinned, and covers
# every ranked domain family including two beagle seeds.
#
#   plans      tokenize each seed into the `{seed_tokens, emitted}` plan the
#              MTP verbs take. No GPU work.
#   reference  mtp-verify --generate walks the SERIAL width-1 frame. The head
#              is loaded but never drafts, so the rows are the target's own
#              greedy chain.
#   capture    mtp-verify --golden replays those rows with the declared head
#              drafting at depth 8, with MLX_E87_HIDDEN_DUMP set so the
#              instrument streams every `x` that reaches the draft readout.
#              Untimed, so the host readback costs nothing that matters.
#
# Usage:
#   research/e133_capture.sh plans
#   research/e133_capture.sh reference [--steps N] [--only A,B] [--limit N]
#   research/e133_capture.sh capture   [--steps N] [--depth D] [--only A,B] [--limit N]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
out="${cache}/e133/screen"
head_dir="${cache}/mtp-head-declared-run"
cli=.build/release/mlxfast-swift
# The E133 manifest carries the E124 windows unchanged plus the reweighting
# windows F1.3 asks for, and each stage skips a seed that already has output,
# so pointing here reuses the first capture instead of repeating it.
manifest=research/e133-corpus-manifest.json

stage="${1:-}"; shift || true
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
    *) echo "e133-capture: unknown flag '$1'" >&2; exit 2 ;;
  esac
done
ref_dir="${out}/reference"
dump_dir="${out}/hidden"

seed_names() {
  python3 - "$only" "$limit" "$manifest" <<'PY'
import json, sys
only, limit, manifest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
keep = set(filter(None, only.split(","))) if only else None
seeds = [s for s in json.load(open(manifest))["seeds"]
         if keep is None or s["id"] in keep]
# beagle first, then one seed per domain per pass, so ANY prefix of the list
# covers the deciding low-acceptance regime and stays domain-balanced.
by_domain: dict[str, list[str]] = {}
for s in seeds:
    by_domain.setdefault(s["domain"], []).append(s["id"])
order, i = [], 0
domains = ["beagle"] + [d for d in by_domain if d != "beagle"]
while len(order) < len(seeds):
    for d in domains:
        if d in by_domain and i < len(by_domain[d]):
            order.append(by_domain[d][i])
    i += 1
print("\n".join(order[:limit] if limit else order))
PY
}

echo "e133-capture: cli    $(shasum -a 256 ${cli} | cut -d' ' -f1)"
echo "e133-capture: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
echo "e133-capture: head   $(git rev-parse HEAD)"
echo "e133-capture: HOME   ${HOME}"

case "${stage}" in
plans)
  mkdir -p "${out}/plans"
  python3 - "${out}/plans" "${manifest}" <<'PY'
import hashlib, json, sys
from pathlib import Path
from tokenizers import Tokenizer

plans_dir, manifest = Path(sys.argv[1]), sys.argv[2]
tok = Tokenizer.from_file("weights/tokenizer.json")
blob = json.load(open(manifest))
for seed in blob["seeds"]:
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
      || { echo "e133-capture: reference ${name} FAILED" >&2; rm -f "${dest}"; continue; }
    jq -e --argjson tokens "${steps}" '
        . as $g | ($g.rows | length) as $rows
        | $g.reference_self_consistent == true
          and ($g.emitted_tokens | length) == $rows
          and $rows >= ($tokens + 1)
          and ([range(0; $rows)
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
      ' "${dest}" >/dev/null || { echo "e133-capture: ${name} reference unusable" >&2; rm -f "${dest}"; continue; }
    echo "e133-capture: ${name} reference in $(( $(date +%s) - start ))s"
  done
  ;;
capture)
  mkdir -p "${dump_dir}" "${out}/verify"
  # `--golden` hands the worker a seatbelt profile carrying a blanket
  # `(deny file-write*)`, so the instrument's FileHandle open fails and the
  # dump is silently empty. The capture stage is untimed research, so drop the
  # profile here and check the dump instead, exactly as E87 did.
  export MLXFAST_NO_SANDBOX=1
  for name in $(seed_names); do
    golden="${ref_dir}/${name}_${steps}.json"
    [[ -s "${golden}" ]] || { echo "e133-capture: missing golden ${name}"; continue; }
    dest="${out}/verify/${name}.json"
    if [[ -s "${dest}" ]] && compgen -G "${dump_dir}/${name}.pid*.tok.i32" >/dev/null; then
      echo "=== skip capture ${name} ==="; continue
    fi
    echo "=== capture ${name} (depth ${depth}, ${steps} tokens) ==="
    # Shards are named by PID, so a retry after a partial run would leave
    # orphan shards from the old PIDs and inflate the sample count.
    rm -f "${dump_dir}/${name}".pid*
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
      jq -r '"e133-capture: parity=\(.parity_all_ok) matched=\(.all_tokens_matched)"
             + " rounds=\(.round_count) accept=\(.accepted_draft_rate)"
             + " meandraft=\(.effective_mean_draft_len)"
             + " head=\(.head_provenance.sha256[0:12])"' "${dest}"
      tok_bytes=$(cat "${dump_dir}/${name}".pid*.tok.i32 2>/dev/null | wc -c | tr -d ' ')
      x_bytes=$(cat "${dump_dir}/${name}".pid*.x.f32 2>/dev/null | wc -c | tr -d ' ')
      shards=$(ls "${dump_dir}/${name}".pid*.tok.i32 2>/dev/null | wc -l | tr -d ' ')
      if ((tok_bytes == 0)) || ((x_bytes != tok_bytes / 4 * 5120 * 4)); then
        echo "e133-capture: ${name} DUMP UNUSABLE (tok=${tok_bytes} x=${x_bytes})" >&2
        rm -f "${dest}" "${dump_dir}/${name}".pid*
        continue
      fi
      echo "e133-capture: ${name} dumped $((tok_bytes / 4)) samples" \
           "in ${shards} shard(s) in $(( $(date +%s) - start ))s"
    else
      mv "${log}" "${out}/verify/${name}.failed.log"
      rm -f "${dest}" "${dump_dir}/${name}".pid*
      echo "e133-capture: ${name} FAILED" >&2
      tail -5 "${out}/verify/${name}.failed.log" >&2
    fi
  done
  ;;
all)
  # One supervised process for the whole rung, so the model is loaded and the
  # GPU is held by exactly one job rather than by a chain of launches.
  flags=(--steps "${steps}" --depth "${depth}")
  [[ -n "${only}" ]] && flags+=(--only "${only}")
  ((limit)) && flags+=(--limit "${limit}")
  "$0" plans || exit $?
  "$0" reference "${flags[@]}" || exit $?
  "$0" capture "${flags[@]}" || exit $?
  ;;
*)
  echo "usage: research/e133_capture.sh {plans|reference|capture|all} [flags]" >&2
  exit 2
  ;;
esac
echo "e133-capture: ${stage} done"
