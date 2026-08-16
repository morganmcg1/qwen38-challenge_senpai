#!/usr/bin/env bash
# Run the full analysis suite for Run N (gate 1, cap 8, 512 decode tokens; repeat of Run L).
set -uo pipefail

T=research/trace-runN-gate1-cap8-512-confirm.log
S=research/score-runN-gate1-cap8-512-confirm.json
C=research/capture-runN-gate1-cap8-512-confirm/04-mtp-timed.json

echo "=== provenance ==="
grep -aE "^run-gate-arm" "$T" | head -20
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
  --label runN-gate1-cap8-512-confirm \
  --notes "Part B confirmation repeat of Run L: segmentedStreakGate=1, segmentedVerifyDepthCap=8, 512 decode tokens, M4 Pro local-iterate" \
  --config '{"segmentedVerifyDepthCap":8,"segmentedStreakGate":1,"sdpaWidthWallDepthCap":4,"decode_tokens":512,"mode":"--local-iterate","repeat_of":"runL"}' \
  --wandb --out research/analysis-runN.json
echo
echo "=== gate counterfactual ==="
python3 research/gate_counterfactual.py --trace "$T" --shallow-cap 4 --out research/gate-counterfactual-runN.json
echo
echo "=== regime split ==="
python3 research/regime_split.py --trace "$T" --label runN --out research/regime-runN.json
echo
echo "=== kl boundary ==="
python3 research/kl_boundary.py --trace "$T" --seed 512 --boundary 1024 --out research/kl-boundary-runN.json
echo
echo "=== fb7 head rebase ==="
python3 research/fb7_head_rebase.py --trace Ntrace="$T" --parent N="$C" \
  --parent-depths-from Ntrace --bandwidth 227e9 --out research/fb7-runN.json
echo
echo "=== fb11 stream cost (all traces) ==="
python3 research/fb11_stream_cost.py --out research/fb11-stream-cost.json
echo
echo "=== serial overhead ==="
python3 research/serial_overhead.py
echo
echo "=== arm summary ==="
python3 research/arm_summary.py --out research/arm-summary.json
