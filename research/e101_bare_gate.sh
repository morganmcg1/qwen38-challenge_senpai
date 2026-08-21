#!/usr/bin/env bash
# E101 compiled-default gate.
#
#   usage: research/e101_bare_gate.sh
#
# The ranked worker exports no environment, so a mechanism that only runs
# under an exported variable scores zero. This gate proves four things about
# `MLX_E101_ROW_TOP32` on the built worker:
#
#   1. UNSET REACHES THE FUSED KERNELS. Legs run with the variable absent from
#      the environment. Each must read `sel_env=unset` with `sel_fused` rising
#      and `sel_argpart` at zero. The paired control leg exports 0 and must
#      read the mirror image, which proves the witness can discriminate.
#   2. THE COMPILED DEFAULT IS BEHAVIOURALLY IDENTICAL TO THE LEGACY PATH. The
#      row ledger recovered from the trace must agree tuple for tuple between
#      the bare fused leg and the legacy control leg at the same token count.
#      This is an arm-relative claim, so it does not depend on any assumption
#      about how the parent's fixed decode window cuts the final round.
#   3. THE LEDGER AGREES WITH `score.json`, which is an independent record:
#      exact `effective_mean_draft_len` and `accepted_draft_rate`, plus
#      `all_tokens_matched` and zero residual divergence.
#   4. MALFORMED VALUES FAIL CLOSED. One leg exports a value that is neither 0
#      nor 1 and must exit non-zero rather than resolve to either arm.
#
# `emitted - decode_tokens` is REPORTED but not asserted against a constant.
# It is 0 when the window ends on a round boundary and 1 when the final round's
# committed primary falls outside the counted window. An earlier revision of
# this gate hard-coded 1, which held at 512 tokens and failed at 32 for that
# reason alone, with every fidelity field clean.
#
# The gate refuses to run if the variable is already exported, because that
# would make case 1 vacuous.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "${MLX_E101_ROW_TOP32:-}" ]]; then
  echo "e101_bare_gate: MLX_E101_ROW_TOP32 is exported; unset it first" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "e101_bare_gate: worktree is dirty" >&2
  exit 1
fi

failures=0

# $1 tag, $2 expected witness arm: "fused" or "legacy"
witness() {
  local tag="$1" want="$2" line
  line="$(grep -o 'sel_env=[^ ]* sel_fused=[0-9]* sel_argpart=[0-9]*' \
    "research/out/${tag}/trace.txt" | tail -1)"
  echo "${tag} witness: ${line:-<none>}"
  local env fused argpart
  env="${line#sel_env=}"; env="${env%% *}"
  fused="${line#*sel_fused=}"; fused="${fused%% *}"
  argpart="${line##*sel_argpart=}"
  case "${want}" in
    fused)
      [[ "${env}" == "unset" && "${fused}" -gt 0 && "${argpart}" -eq 0 ]] \
        && return 0 ;;
    legacy)
      [[ "${env}" == "0" && "${fused}" -eq 0 && "${argpart}" -gt 0 ]] \
        && return 0 ;;
  esac
  echo "e101_bare_gate: ${tag} witness is wrong; wanted ${want}" >&2
  return 1
}

# $1 tag -> writes research/out/$tag/ledger.json, asserts score.json agreement
ledger() {
  python3 - "$1" <<'PY'
import json, re, sys
tag = sys.argv[1]
m = json.load(open(f"research/out/{tag}/score.json"))["metrics"]
rows = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(-?\d+)", line))
        for line in open(f"research/out/{tag}/trace.txt")
        if line.startswith("mtp-trace: round=")]
n = len(rows)
drafts = sum(r["d"] for r in rows)
acc = sum(r["acc"] for r in rows)
led = {"rounds": n, "target_rows": drafts + n, "emitted": acc + n,
       "drafts": drafts, "accepted": acc, "rejected": drafts - acc,
       "per_round": [[r["d"], r["acc"]] for r in rows]}
json.dump(led, open(f"research/out/{tag}/ledger.json", "w"))
ok = (m["all_tokens_matched"] and m["residual_divergence_count"] == 0
      and abs(acc / drafts - m["accepted_draft_rate"]) < 1e-12
      and abs(drafts / n - m["effective_mean_draft_len"]) < 1e-9)
print(f"{tag} ledger: rounds {n} target_rows {drafts + n} emitted {acc + n} "
      f"drafts {drafts} accepted {acc} rejected {drafts - acc} "
      f"emitted-decode {acc + n - m['decode_tokens']} "
      f"matched {m['all_tokens_matched']} "
      f"divergence {m['residual_divergence_count']} "
      f"score.json {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

for tokens in 32 512; do
  bare="e101bare${tokens}"
  ctl="e101ctl${tokens}"

  echo "=== ${bare}: nothing exported, ${tokens} decode tokens ==="
  research/e79_trace_leg.sh "${bare}" "${tokens}"
  status=$?
  ((status == 0)) || { echo "e101_bare_gate: ${bare} exited ${status}" >&2
                       failures=$((failures + 1)); continue; }
  witness "${bare}" fused || failures=$((failures + 1))
  ledger "${bare}" || failures=$((failures + 1))

  echo "=== ${ctl}: MLX_E101_ROW_TOP32=0 control, ${tokens} decode tokens ==="
  MLX_E101_ROW_TOP32=0 research/e79_trace_leg.sh "${ctl}" "${tokens}"
  status=$?
  ((status == 0)) || { echo "e101_bare_gate: ${ctl} exited ${status}" >&2
                       failures=$((failures + 1)); continue; }
  witness "${ctl}" legacy || failures=$((failures + 1))
  ledger "${ctl}" || failures=$((failures + 1))

  python3 - "${bare}" "${ctl}" <<'PY' || failures=$((failures + 1))
import json, sys
a, b = (json.load(open(f"research/out/{t}/ledger.json")) for t in sys.argv[1:3])
same = a == b
print(f"{sys.argv[1]} vs {sys.argv[2]} ledger identity: "
      f"{'IDENTICAL' if same else 'DIFFERENT'} "
      f"({a['rounds']} rounds, {a['drafts']} drafts, {a['accepted']} accepted)")
if not same:
    for i, (x, y) in enumerate(zip(a["per_round"], b["per_round"])):
        if x != y:
            print(f"  first differing round {i}: fused {x} legacy {y}")
            break
sys.exit(0 if same else 1)
PY
done

echo "=== e101malformed: MLX_E101_ROW_TOP32=maybe must fail closed ==="
MLX_E101_ROW_TOP32=maybe research/e79_trace_leg.sh e101malformed 32
status=$?
if ((status == 0)); then
  echo "e101_bare_gate: a malformed value did NOT fail closed" >&2
  failures=$((failures + 1))
else
  echo "e101malformed: exited ${status} as required"
fi

echo "e101_bare_gate: ${failures} failures"
exit "${failures}"
