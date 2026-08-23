#!/usr/bin/env bash
# E152 F4 step 6 -- the 512-token exactness gate for the imported promoted
# frontier surface.
#
#   usage: research/e152_exact512.sh
#
# WHAT IS CLAIMED. The candidate tree now carries the promoted frontier's
# `Qwen35.swift`, `Qwen36MTPBlockSession.swift`, `quantized_nax.h` and its
# generated twin byte for byte. The claim is that this tree still produces the
# serial token stream and the serial row evidence exactly, on this host, with
# the head that `mtp-head.manifest.json` declares.
#
# WHY 512 AND NOT 64. The seed is 512 tokens, so only a 512-token decode window
# walks the key length past 1024 and exercises the boundary that the row ledger
# closes over. A 64-token or 128-token leg never reaches it. The window also
# has to continue past EOS, which a short leg can miss.
#
# WHY A ROW DIGEST AND NOT AN ARGMAX MATCH. The trusted parent checks the exact
# top-two target values, not just the winner, so the gate is over `mtp-row:`
# lines whose top-two values are hex float literals.
#
# THE PIN. `719d82b8...` is the campaign's 1025-row digest, first pinned by
# E121 rung 3 and reproduced by E101, E116, E121 and E129 on later bases. No
# compliant candidate change may move it, so it is the correct pin for an
# import that claims to change no emitted value.
#
# THE CONTROL. `research/e116_row_digest_check.py` runs its own value and order
# controls. The runtime control has to change the decode window, because no
# compliant candidate knob can move this digest: a `mtp-row:` line is emitted
# once per emitted token position for each of the wrapper's two passes, so a
# 512-token leg carries 1025 rows and a 128-token leg carries 257. The
# 128-token leg's digest MUST NOT match the pin.
#
# THERMAL STATE. The 512-token leg keeps the real 40 C gate, so it is
# gate-qualified. The 128-token control is a correctness control and not a
# timing arm, so it runs without the gate and is labelled as such in its own
# `meta.txt`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PIN=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e152_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

out=research/e152-artifacts
mkdir -p "${out}"
failures=0

echo "=== e152x512cand: imported frontier surface, tokens=512, real 40 C gate ==="
research/e79_trace_leg.sh e152x512cand 512 --cool-gate \
  || { echo "e152_exact512: 512-token leg failed" >&2; failures=$((failures + 1)); }

echo
echo "=== e152x128neg: tokens=128 (runtime negative control, no gate) ==="
research/e79_trace_leg.sh e152x128neg 128 \
  || { echo "e152_exact512: 128-token control leg failed" >&2
       failures=$((failures + 1)); }

echo
python3 research/e116_row_digest_check.py e152x512cand \
  --pin "${PIN}" \
  --expect-rows 1025 \
  --negative-control e152x128neg \
  --json "${out}/row-digest-512.json" \
  || failures=$((failures + 1))

echo
echo "--- wrapper verdicts ---"
for tag in e152x512cand e152x128neg; do
  echo "${tag}: $(grep -oE 'all_tokens_matched[": ]*[a-z]+' \
    "research/out/${tag}/wrapper.out" 2>/dev/null | tail -1)"
  python3 - "${tag}" <<'PY'
import json, sys
tag = sys.argv[1]
try:
    d = json.load(open(f"research/out/{tag}/score.json"))
except OSError:
    print(f"{tag}: no score.json")
    raise SystemExit(0)
m = d.get("metrics", {})
print(
    f"{tag}: passed={d.get('passed')} decode_tokens={m.get('decode_tokens')} "
    f"all_tokens_matched={m.get('all_tokens_matched')} "
    f"residual_divergence_count={m.get('residual_divergence_count')} "
    f"uses_pinned_mtp_head={m.get('uses_pinned_mtp_head')} "
    f"head_provenance_sha256={m.get('head_provenance_sha256')} "
    f"serial_s_per_token={m.get('serial_seconds_per_token')} "
    f"mtp_s_per_token={m.get('mtp_seconds_per_token')} "
    f"ratio={m.get('mtp_decode_speedup')} "
    f"mean_draft_len={m.get('effective_mean_draft_len')} "
    f"accepted_draft_rate={m.get('accepted_draft_rate')}"
)
PY
done

echo
echo "--- thermal and identity ---"
for tag in e152x512cand e152x128neg; do
  grep -E '^(gpu_temp_entry_c|gpu_temp_exit_c|cool_gate_passed_real_gate|gate_qualified_for_timing|worker_sha256|base_sha|dirty_candidate_paths|trace_rounds)=' \
    "research/out/${tag}/meta.txt" 2>/dev/null | sed "s/^/${tag} /"
done

echo
echo "e152_exact512: failures=${failures}"
exit "${failures}"
