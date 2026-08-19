#!/bin/bash
# E33 diagnostic: per-round verify widths + cross-arm golden exactness.
#
# Not a timed headline. Runs the trusted CLI's mtp-verify / mtp-timed verbs
# directly because benchmark-qwen-mtp.sh rm -rf's the run dir that carries
# `effective_draft_lengths`, `decode_seconds` and `seed_prefill_seconds`.
#
#   e33_diag.sh golden        -> generate reference rows from THIS build
#   e33_diag.sh timed TAG     -> time this build against /tmp/e33_diag/golden-base.json
set -u -o pipefail

MODE="${1:?usage: e33_diag.sh golden|timed [TAG]}"
TAG="${2:-}"
OUT=/tmp/e33_diag
mkdir -p "$OUT"

eval "$(./setup-qwen-mtp.sh --print-paths)"
SWIFT_BIN=.build/release/mlxfast-swift
FIXTURE=correctness_prompts/public_longcopy_gate_english_512_256.json
TOKENS=64
DEPTH=8
GOLDEN="$OUT/golden-base.json"

tools/build-mlx-metallib.sh --all-build-roots > "$OUT/metallib-$MODE$TAG.log" 2>&1 || exit 90

HEAD_SHA="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
FP="$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
echo "e33-diag: mode=$MODE tag=$TAG head=$HEAD_SHA dirty=$DIRTY metallib=$FP" >&2

if [[ "$MODE" == "golden" ]]; then
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "$FIXTURE" > "$OUT/plan.json" || exit 80
  "$SWIFT_BIN" mtp-verify \
    --weights weights \
    --mtp-head "$MLXFAST_QWEN_MTP_HEAD_DIR" \
    --emitted "$OUT/plan.json" \
    --generate "$(( TOKENS + 1 ))" \
    --mtp-depth "$DEPTH" \
    --output "$GOLDEN" \
    --plan-output "$OUT/generated-plan.json" 2> "$OUT/golden.log" || exit 81
  echo "e33-diag: golden written head=$HEAD_SHA metallib=$FP" >&2
  jq '{rows: (.rows|length), self_consistent: .reference_self_consistent}' "$GOLDEN" >&2
  echo "$HEAD_SHA $FP" > "$OUT/golden.provenance"
  exit 0
fi

[[ -f "$GOLDEN" ]] || { echo "e33-diag: missing $GOLDEN" >&2; exit 82; }
# Phase trace ON: this arm is a DIAGNOSTIC, not a timing arm. It buys the
# per-round draft width (`d=`) and a hexfloat dump of every declared top-2
# value, which is what makes the cross-arm row comparison bit-exact.
rm -f "$OUT/trace-$TAG.txt"
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="$OUT/trace-$TAG.txt"
"$SWIFT_BIN" mtp-timed \
  --weights weights \
  --mtp-head "$MLXFAST_QWEN_MTP_HEAD_DIR" \
  --golden "$GOLDEN" \
  --tokens "$TOKENS" \
  --mtp-depth "$DEPTH" > "$OUT/timed-$TAG.json" 2> "$OUT/timed-$TAG.log" || exit 83
echo "$HEAD_SHA $DIRTY $FP" > "$OUT/timed-$TAG.provenance"
jq '{all_tokens_matched, parity_all_ok, residual_divergence_count,
     round_count, effective_mean_draft_len, effective_max_draft_len,
     effective_draft_lengths, decode_seconds, seed_prefill_seconds,
     parent_measured_seconds_per_token, accepted_draft_total,
     rejected_draft_total, declared_rows_total,
     reference_checked_row_total}' "$OUT/timed-$TAG.json" >&2
