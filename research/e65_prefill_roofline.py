#!/usr/bin/env python3
"""Is the seed prefill limited by the GEMM kernels or by the overheads?

E65 measured begin() at about 4.0 s, which is 23 % of a candidate leg, and
showed it costs the same in the serial and the MTP session. That makes it the
largest unoptimised block in the leg, but it only pays to attack the kernels if
prefill runs well below the device roofline. This computes the achieved rate
from the pinned tensor inventory so the answer is arithmetic, not opinion.

Every shape comes from Qwen35Weights.swift and the expect() block in
Qwen35Config.swift, so the inventory is checkable against the enforcing source.
"""

VOCAB = 248_320
HIDDEN = 5_120
INTERMEDIATE = 17_408
LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
LIN_KEY_HEADS = 16
LIN_VALUE_HEADS = 48
LIN_KEY_HEAD_DIM = 128
LIN_VALUE_HEAD_DIM = 128

SEED_TOKENS = 512
# begin(): build interval plus eval_wall, six measurements over three legs,
# build 2.945-2.976 s and eval 1.047-1.049 s, identical in both sessions.
BEGIN_SECONDS = 4.0
# edward's measured bf16 peak on a sibling Apple host.
PEAK_TFLOPS = 7.51

fa_layers = LAYERS // FULL_ATTENTION_INTERVAL
gdn_layers = LAYERS - fa_layers

lin_key = LIN_KEY_HEADS * LIN_KEY_HEAD_DIM
lin_value = LIN_VALUE_HEADS * LIN_VALUE_HEAD_DIM
lin_conv = 2 * lin_key + lin_value
# numAttentionHeads * headDim * 2: query and output gate are one fused tensor.
full_query = HEADS * HEAD_DIM * 2
full_kv = KV_HEADS * HEAD_DIM
full_out = HEADS * HEAD_DIM

mlp = 3 * HIDDEN * INTERMEDIATE
fa = (full_query + 2 * full_kv) * HIDDEN + HIDDEN * full_out
gdn = (lin_conv + lin_value + 2 * LIN_VALUE_HEADS) * HIDDEN + HIDDEN * lin_value

matmul_params = LAYERS * mlp + fa_layers * fa + gdn_layers * gdn
embed = VOCAB * HIDDEN
total_params = matmul_params + 2 * embed  # untied lm_head

# Projections dominate. Attention scores are causal, so QK^T and AV each cost
# about half of the dense 2*qL*kL*heads*head_dim.
proj_flops = 2 * matmul_params * SEED_TOKENS
attn_flops = fa_layers * 2 * (SEED_TOKENS ** 2) * HEADS * HEAD_DIM
lm_head_flops = 2 * embed  # logits for the final row only
total_flops = proj_flops + attn_flops + lm_head_flops

achieved = total_flops / BEGIN_SECONDS / 1e12

print(f"layers                  {fa_layers} full attention + {gdn_layers} gated deltanet")
print(f"matmul parameters       {matmul_params / 1e9:.2f} B")
print(f"embed + lm_head         {2 * embed / 1e9:.2f} B  (gather, no FLOPs on input)")
print(f"total parameters        {total_params / 1e9:.2f} B")
print()
print(f"projection FLOPs        {proj_flops / 1e12:.2f} T")
print(f"attention score FLOPs   {attn_flops / 1e12:.3f} T")
print(f"lm_head FLOPs           {lm_head_flops / 1e12:.4f} T")
print(f"prefill total           {total_flops / 1e12:.2f} TFLOP for {SEED_TOKENS} tokens")
print()
print(f"measured begin()        {BEGIN_SECONDS:.2f} s")
print(f"achieved                {achieved:.2f} TFLOP/s")
print(f"fraction of bf16 peak   {100 * achieved / PEAK_TFLOPS:.1f} % of {PEAK_TFLOPS} TFLOP/s")
print()
ideal = total_flops / (PEAK_TFLOPS * 1e12)
print(f"prefill at 100 % of peak would take {ideal:.2f} s, so the absolute")
print(f"ceiling on any prefill work is {BEGIN_SECONDS - ideal:.2f} s.")
print("Caveat: the backbone is affine 4-bit group-64, not bf16, and the peak")
print("was measured on a sibling host, so treat this as an order-of-magnitude")
print("roofline rather than an exact efficiency.")
