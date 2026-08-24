#!/usr/bin/env python3
"""E169 step 1: exact static FLOP/byte budget of one marginal verified row.

Reads real tensor shapes from the transformed checkpoint headers, groups them by
the executed module (GDN layer, full-attention layer, MLP, lm_head), and prices
one extra verified row. No GPU work, no model load.

harness=static (architecture arithmetic only; not a timing measurement)
"""

from __future__ import annotations

import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

WEIGHTS = Path(sys.argv[1] if len(sys.argv) > 1 else "weights")


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def collect() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for shard in sorted(WEIGHTS.glob("model-*.safetensors")):
        for name, meta in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            shapes[name] = meta["shape"]
    return shapes


def classify(name: str) -> str:
    if ".linear_attn." in name:
        return "gdn"
    if ".self_attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "embed"
    return "norm"


def main() -> None:
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    shapes = collect()

    # Logical (unpacked) parameter count of every quantized or dense matmul
    # weight, keyed by module family. A 4-bit weight is stored packed in uint32
    # words, so the packed `.weight` shape is [out, in/8]; `.scales` is
    # [out, in/group] and recovers the true `in`.
    group = cfg["quantization"]["group_size"]
    macs: dict[str, int] = defaultdict(int)  # multiply-accumulates per row
    wbytes: dict[str, float] = defaultdict(float)  # stored bytes of the weights
    per_layer: dict[tuple[str, int], int] = defaultdict(int)

    for name, shape in shapes.items():
        fam = classify(name)
        layer = -1
        if ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
        if name.endswith(".scales"):
            out, ngroups = shape
            n_in = ngroups * group
            macs[fam] += out * n_in
            per_layer[(fam, layer)] += out * n_in
            # 4-bit packed weight + fp16 scale + fp16 bias per group
            wbytes[fam] += out * n_in / 2 + 2 * 2 * out * ngroups
        elif name.endswith(".weight") and len(shape) == 2 and ".conv1d" not in name:
            quantized = name[: -len("weight")] + "scales" in shapes
            if not quantized:
                out, n_in = shape
                macs[fam] += out * n_in
                per_layer[(fam, layer)] += out * n_in
                wbytes[fam] += out * n_in * 2

    layer_types = cfg["layer_types"]
    n_gdn = sum(1 for t in layer_types if t == "linear_attention")
    n_attn = sum(1 for t in layer_types if t == "full_attention")

    # Recurrence and attention arithmetic that is not a weight matmul.
    dk = cfg["linear_key_head_dim"]
    dv = cfg["linear_value_head_dim"]
    hv = cfg["linear_num_value_heads"]
    state_elems = hv * dk * dv
    # delta rule per token: S <- S*g + k^T (v - S k) ~ 3 MAC per state element
    gdn_recurrence_macs = 3 * state_elems * n_gdn
    gdn_state_bytes = state_elems * 4 * n_gdn  # float32 state, read + written once

    kv_len = 512 + 64  # representative decode position inside the scored window
    q_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    attn_score_macs = 2 * q_heads * head_dim * kv_len * n_attn
    kv_bytes_per_row = 2 * cfg["num_key_value_heads"] * head_dim * 2 * kv_len * n_attn

    rows = {
        "gdn_proj": macs["gdn"],
        "gdn_recurrence": gdn_recurrence_macs,
        "attn_proj": macs["attn"],
        "attn_scores": attn_score_macs,
        "mlp": macs["mlp"],
        "lm_head": macs["lm_head"],
    }
    total = sum(rows.values())

    print(f"# E169 static budget  harness=static  weights={WEIGHTS}")
    print(f"layers: {n_gdn} gated-delta + {n_attn} full-attention")
    print(f"hidden={cfg['hidden_size']} inter={cfg['intermediate_size']} "
          f"vocab={cfg['vocab_size']} quant={cfg['quantization']}")
    print()
    print("per marginal verified row (2 FLOP per MAC):")
    print(f"{'component':<18}{'GFLOP/row':>12}{'share':>9}")
    for k, v in rows.items():
        print(f"{k:<18}{2*v/1e9:>12.3f}{100*v/total:>8.2f}%")
    print(f"{'TOTAL':<18}{2*total/1e9:>12.3f}{100.0:>8.2f}%")
    print()

    gflop = 2 * total / 1e9
    for host, ms in (("local g16s M4 Pro", 7.002), ("ranked M5", 5.3350)):
        print(f"implied throughput {host:<20} {ms:.4f} ms/row -> "
              f"{gflop/ms:.3f} TFLOP/s")
    print()

    # Per-round weight traffic (shared by all rows of a round) for contrast.
    stored = sum(wbytes.values())
    print("stored weight bytes by family (per round, shared across rows):")
    for k in sorted(wbytes, key=lambda k: -wbytes[k]):
        print(f"  {k:<12}{wbytes[k]/1e9:>8.3f} GB{100*wbytes[k]/stored:>8.2f}%")
    print(f"  {'TOTAL':<12}{stored/1e9:>8.3f} GB")
    print()
    print(f"gdn float32 recurrent state traffic per forward: "
          f"{2*gdn_state_bytes/1e6:.1f} MB (read+write, once per call, "
          f"independent of row count)")
    print(f"kv cache read per attention pass at kv_len={kv_len}: "
          f"{kv_bytes_per_row/1e6:.1f} MB (shared across rows)")
    print()
    per_l_gdn = [v for (f, l), v in per_layer.items() if f == "gdn"]
    per_l_mlp = [v for (f, l), v in per_layer.items() if f == "mlp"]
    per_l_attn = [v for (f, l), v in per_layer.items() if f == "attn"]
    print(f"per-layer MAC (uniform check): gdn {min(per_l_gdn)/1e6:.1f}M "
          f"attn {min(per_l_attn)/1e6:.1f}M mlp {min(per_l_mlp)/1e6:.1f}M "
          f"(spread gdn {max(per_l_gdn)-min(per_l_gdn)}, "
          f"mlp {max(per_l_mlp)-min(per_l_mlp)})")

    out = {
        "harness": "static",
        "component_gflop_per_row": {k: 2 * v / 1e9 for k, v in rows.items()},
        "total_gflop_per_row": gflop,
        "implied_tflops": {"local_7.002ms": gflop / 7.002,
                           "ranked_5.3350ms": gflop / 5.3350},
        "stored_weight_bytes": wbytes,
        "gdn_state_bytes_per_forward": 2 * gdn_state_bytes,
        "kv_bytes_per_attention_pass": kv_bytes_per_row,
    }
    Path("research/out/e169").mkdir(parents=True, exist_ok=True)
    Path("research/out/e169/flop_budget.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
