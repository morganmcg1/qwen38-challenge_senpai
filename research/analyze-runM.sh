#!/usr/bin/env bash
# Run the full analysis suite for Run M (gate 0, cap 8, 512 decode tokens).
set -uo pipefail

T=research/trace-runM-gate0-cap8-512.log
S=research/score-runM-gate0-cap8-512.json
C=research/capture-runM-gate0-cap8-512/04-mtp-timed.json

echo "=== provenance ==="
grep -E "^run-gate-arm" "$T" | head -20
echo
echo "=== score ==="
python3 -c "import json;d=json.load(open('$S'));print('score',d['score'],'passed',d['passed'],'track',d.get('track_id'))"
echo
echo "=== depth histogram ==="
grep -oE "mtp-trace: round=[0-9]+ d=[0-9]+" "$T" | sed -E 's/.* d=//' | sort -n | uniq -c
echo
echo "=== cap histogram ==="
grep -oE "cap=[0-9]+" "$T" | sort | uniq -c
echo
echo "=== row gate ==="
python3 research/mtp_row_gate.py --trace "$T" --score "$S" \
  --label runM-gate0-cap8-512 \
  --notes "Part B variant 3: segmentedStreakGate=0 (always deep), segmentedVerifyDepthCap=8, 512 decode tokens, M4 Pro local-iterate" \
  --config '{"segmentedVerifyDepthCap":8,"segmentedStreakGate":0,"sdpaWidthWallDepthCap":4,"decode_tokens":512,"mode":"--local-iterate"}' \
  --wandb --out research/analysis-runM.json
echo
echo "=== gate counterfactual ==="
python3 research/gate_counterfactual.py --trace "$T" --shallow-cap 4 --out research/gate-counterfactual-runM.json
echo
echo "=== regime split ==="
python3 research/regime_split.py --trace "$T" --label runM --out research/regime-runM.json
echo
echo "=== kl boundary ==="
python3 research/kl_boundary.py --trace "$T" --seed 512 --boundary 1024 --out research/kl-boundary-runM.json
echo
echo "=== fb7 head rebase ==="
python3 research/fb7_head_rebase.py --trace Mtrace="$T" --parent M="$C" \
  --parent-depths-from Mtrace --bandwidth 227e9 --out research/fb7-runM.json
echo
echo "=== fb11 stream cost (all traces) ==="
python3 research/fb11_stream_cost.py --out research/fb11-stream-cost.json
echo
echo "=== serial overhead ==="
python3 research/serial_overhead.py
echo
echo "=== arm summary ==="
python3 research/arm_summary.py --out research/arm-summary.json
