#!/usr/bin/env bash
# Run the full analysis suite for Run O (r3 candidate arm: cap 7, gate 3, 512 decode tokens).
# Control arm for this comparison is Run I (cap 8, gate 3, 512 decode tokens).
set -uo pipefail

T=research/trace-runO-cap7-gate3-512.log
S=research/score-runO-cap7-gate3-512.json
C=research/capture-runO-cap7-gate3-512/04-mtp-timed.json

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
  --label runO-cap7-gate3-512 \
  --notes "r3 candidate arm: segmentedVerifyDepthCap=7, segmentedStreakGate=3 (shipped), 512 decode tokens, M4 Pro local-iterate. Control is Run I (cap 8, gate 3)." \
  --config '{"segmentedVerifyDepthCap":7,"segmentedStreakGate":3,"sdpaWidthWallDepthCap":4,"decode_tokens":512,"mode":"--local-iterate","control_arm":"runI"}' \
  --wandb --out research/analysis-runO.json
echo
echo "=== gate counterfactual ==="
python3 research/gate_counterfactual.py --trace "$T" --shallow-cap 4 --out research/gate-counterfactual-runO.json
echo
echo "=== regime split ==="
python3 research/regime_split.py --trace "$T" --label runO --out research/regime-runO.json
echo
echo "=== kl boundary ==="
python3 research/kl_boundary.py --trace "$T" --seed 512 --boundary 1024 --out research/kl-boundary-runO.json
echo
echo "=== fb7 head rebase ==="
python3 research/fb7_head_rebase.py --trace Otrace="$T" --parent O="$C" \
  --parent-depths-from Otrace --bandwidth 227e9 --out research/fb7-runO.json
echo
echo "=== fb11 stream cost (all traces) ==="
python3 research/fb11_stream_cost.py --out research/fb11-stream-cost.json
echo
echo "=== serial overhead ==="
python3 research/serial_overhead.py
echo
echo "=== arm summary ==="
python3 research/arm_summary.py --out research/arm-summary.json
