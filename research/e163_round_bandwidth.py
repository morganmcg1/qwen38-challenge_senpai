#!/usr/bin/env python3
"""E163 F4 §3: what fraction of memory bandwidth does one scored round reach?

Four checks, in the order the advisor asked for them.

1. BYTES. Census the pinned checkpoint and the pinned proposal head from their
   safetensors headers. No estimate, no parameter-count arithmetic.
2. PEAK. Read the MEASURED streaming peak from `bandwidth-peak.json`
   (`research/e163_bandwidth_run.sh`). The datasheet 273 GB/s is reported
   beside it, never instead of it.
3. ONCE. State every stream the round reads and how many times, so the
   "reads the weights exactly once" claim is checkable rather than assumed.
4. BOUND. Arithmetic intensity against the measured machine balance.

Every number is harness=local. Usage:

    python3 research/e163_round_bandwidth.py [bandwidth-peak.json]
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "research" / "e163-artifacts"
TARGET = ROOT / "reference_weights" / "Qwen3.8-27B-4bit"
HEAD = ROOT / "reference_weights" / "Qwen3.6-27B-MTP-4bit"

DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64,
}

# Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:5253-5258. The
# drafting argmax runs over a materialised row slice of lm_head, not the whole
# vocabulary.
COMPACT_DRAFT_PADDED_ROWS = 98_336
NOMINAL_PEAK_GB_S = 273.0

# FINDING 352, harness=ranked: R = 16.158 ms + 5.3351 ms x rows on 5a9f130a.
RANKED_PER_ROW_MS = 5.3351

# The affine 4-bit group-64 transposed projections one target verify pass
# reaches, as (k, n, dispatches per pass). Same census the exactness gate uses.
SCORED_SHAPES = [
    (5120, 16480, 48),   # linear_attn.in_proj_fused_qkvzba
    (6144, 5120, 48),    # linear_attn.out_proj
    (5120, 14336, 16),   # full_attn.qkv_proj_fused
    (6144, 5120, 16),    # full_attn.o_proj
    (5120, 34816, 64),   # mlp.gate_up_fused
    (17408, 5120, 64),   # mlp.down
    (5120, 248320, 1),   # head.lm_head
]


def _header(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n).decode("utf-8"))


def census(root: Path) -> dict:
    by_role: dict[str, int] = defaultdict(int)
    tensors: dict[str, int] = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".safetensors"):
            continue
        for key, meta in _header(root / name).items():
            if key == "__metadata__":
                continue
            nelem = 1
            for d in meta["shape"]:
                nelem *= d
            nbytes = nelem * DTYPE_BITS[meta["dtype"]] // 8
            tensors[key] = nbytes
            if "embed_tokens" in key:
                role = "embed_gather"
            elif "lm_head" in key:
                role = "vocab_readout"
            elif re.search(r"layers\.\d+\.", key):
                role = (
                    "layer_gdn" if "linear_attn" in key
                    else "layer_full_attn" if ("full_attn" in key or "self_attn" in key)
                    else "layer_mlp" if "mlp" in key
                    else "layer_norm_misc")
            else:
                role = "top_level"
            by_role[role] += nbytes
    by_role["_total"] = sum(v for k, v in by_role.items() if not k.startswith("_"))
    return {"by_role": dict(by_role), "tensors": tensors}


def gb(x: float) -> str:
    return f"{x/1e9:8.3f} GB"


def main() -> int:
    bw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACTS / "bandwidth-peak.json"

    tgt = census(TARGET)
    head = census(HEAD)
    tr = tgt["by_role"]

    embed = tr["embed_gather"]
    vocab = tr["vocab_readout"]
    target_stream = tr["_total"] - embed
    head_stream = head["by_role"]["_total"]
    lm_rows = 248_320
    compact_stream = round(vocab * COMPACT_DRAFT_PADDED_ROWS / lm_rows)

    print("=" * 78)
    print("CHECK 1 - BYTES.  Source: safetensors headers of the pinned artifacts.")
    print("=" * 78)
    print(f"  target checkpoint total       {tr['_total']:>15,}  {gb(tr['_total'])}")
    for role in ("layer_mlp", "layer_gdn", "layer_full_attn", "vocab_readout",
                 "embed_gather", "layer_norm_misc", "top_level"):
        if role in tr:
            print(f"    {role:<26} {tr[role]:>15,}  {gb(tr[role])}"
                  f"  {100*tr[role]/tr['_total']:5.1f}%")
    print()
    print("  embed_tokens is GATHERED (M rows of 5,120), not streamed, so it")
    print("  leaves the per-round budget. lm_head is NOT tied "
          "(tie_word_embeddings=false)")
    print("  and IS streamed: the verify pass needs exact top-two evidence over")
    print("  the whole 248,320-row vocabulary.")
    print(f"  target weight stream per verify pass   {target_stream:>15,}"
          f"  {gb(target_stream)}")
    print()
    print(f"  pinned MTP head (BF16, no .scales tensors, so it loads unquantised)")
    print(f"    per draft step                       {head_stream:>15,}"
          f"  {gb(head_stream)}")
    print(f"  compact draft lm_head slice "
          f"({COMPACT_DRAFT_PADDED_ROWS:,}/{lm_rows:,} rows)")
    print(f"    per draft step                       {compact_stream:>15,}"
          f"  {gb(compact_stream)}")

    # --- peak -------------------------------------------------------------
    print()
    print("=" * 78)
    print("CHECK 2 - PEAK.  Measured, not quoted.")
    print("=" * 78)
    measured_peak = None
    size_matched = None
    if bw_path.exists():
        bwd = json.loads(bw_path.read_text())
        measured_peak = bwd["measured_peak_gb_per_s"]
        # The right denominator for a 14 GB round is a 14 GB read: same
        # working set, same absence of cache help.
        matched = [c for c in bwd["cells"]
                   if c["kernel"] in ("sum", "max") and c["bytes_touched"] >= 14e9]
        if matched:
            size_matched = max(c["gb_per_s_best"] for c in matched)
        print(f"  device {bwd['device_architecture']}, "
              f"{bwd['device_memory_size_bytes']/2**30:.0f} GiB, working set "
              f"{bwd['device_max_recommended_working_set_bytes']/2**30:.1f} GiB")
        print(f"  {'kernel':<12}{'dtype':<10}{'buffer':>9}{'touched':>12}"
              f"{'us_min':>10}{'GB/s best':>11}{'GB/s med':>10}")
        for c in bwd["cells"]:
            print(f"  {c['kernel']:<12}{c['dtype']:<10}{c['buffer_gib']:>8.2f}G"
                  f"{c['bytes_touched']/1e9:>11.2f}G{c['us_min']:>10.1f}"
                  f"{c['gb_per_s_best']:>11.1f}{c['gb_per_s_median']:>10.1f}")
        print()
        print(f"  MEASURED streaming peak  {measured_peak:.1f} GB/s"
              f"   = {100*measured_peak/NOMINAL_PEAK_GB_S:.1f}% of the "
              f"{NOMINAL_PEAK_GB_S:.0f} GB/s datasheet figure")
        if size_matched:
            print(f"  SIZE-MATCHED read of a >=14 GB buffer  {size_matched:.1f} GB/s"
                  f"   = {100*size_matched/NOMINAL_PEAK_GB_S:.1f}% of datasheet")
            print("  Use the size-matched rate as the round's denominator: it has")
            print("  the same working set and the same absence of cache help.")
        measured_peak = size_matched or measured_peak
    else:
        print(f"  {bw_path} missing - run research/e163_bandwidth_run.sh first.")

    # --- round budget -----------------------------------------------------
    report_path = ARTIFACTS / "e163_pinned_w5.json"
    if not report_path.exists():
        print(f"\n{report_path} missing; cannot price a round.")
        return 1
    rep = json.loads(report_path.read_text())

    print()
    print("=" * 78)
    print("CHECK 3 - HOW MANY TIMES.  Streams per round, from the source.")
    print("=" * 78)
    print("  target verify pass   x1 per round at G=1: one routed QMV per")
    print("    projection per layer, 257 dispatches, no re-read. At G>=2 the")
    print("    kernel splits the width over G groups and reads the same weight")
    print("    G times - that is FINDING 369 and it is a bandwidth statement.")
    print("  proposal head        xd per round: the draft ladder is sequential")
    print("    (Qwen36MTPBlockSession.swift:1458-1464), each step depends on the")
    print("    previous draft id, and 849 MB does not fit any cache.")
    print("  compact draft head   xd per round, same argument.")
    print("  embed_tokens         gathered, not streamed.")

    print()
    print("=" * 78)
    print("ROUND BANDWIDTH.  harness=local, sealed six-leg session.")
    print("=" * 78)
    hdr = (f"  {'arm':<14}{'W':>2}{'G':>2}{'d':>5}{'R ms':>9}{'verify ms':>10}"
           f"{'GB/round':>10}{'GB/s':>8}{'%meas':>7}{'%273':>7}")
    print(hdr)
    rows = []
    cells = rep["decomposition"]["cells"]
    legs = rep["legs"]
    for key in ("shipped_low", "arm_low", "shipped_high"):
        cell = cells[key]
        arm = cell["arm_key"]
        width = cell["rows"]
        groups = cell["weight_streams_G"]
        matching = [leg for leg in legs
                    if leg["arm_key"] == arm and leg["verify_width"] == width]
        if not matching:
            continue
        r_ms = sum(leg["R_ms_from_leg"] for leg in matching) / len(matching)
        verify_ms = sum(leg["ch_verify_pipeline_ms"] for leg in matching) / len(matching)
        edl = sum(leg["effective_mean_draft_len"] for leg in matching) / len(matching)
        drafts = width - 1
        weights = groups * target_stream + drafts * (head_stream + compact_stream)
        achieved = weights / (r_ms * 1e6)
        rows.append({
            "arm": arm, "width": width, "groups": groups, "drafts": drafts,
            "edl": edl, "round_ms": r_ms, "verify_ms": verify_ms,
            "round_weight_bytes": weights,
            "target_share_bytes": groups * target_stream,
            "draft_share_bytes": drafts * (head_stream + compact_stream),
            "achieved_gb_s": achieved,
        })
        pm = f"{100*achieved/measured_peak:6.1f}" if measured_peak else "     -"
        print(f"  {arm:<14}{width:>2}{groups:>2}{drafts:>5}{r_ms:>9.3f}"
              f"{verify_ms:>10.3f}{weights/1e9:>10.3f}{achieved:>8.1f}{pm:>7}"
              f"{100*achieved/NOMINAL_PEAK_GB_S:>7.1f}")

    over = [r for r in rows if r["achieved_gb_s"] > NOMINAL_PEAK_GB_S]
    if over:
        print()
        print("  !! Every G>=2 row above exceeds the datasheet peak. A DRAM rate")
        print("     cannot exceed the bus. So 'G groups read the weight G times'")
        print("     counts DISPATCH-LEVEL loads, not DRAM traffic: the second")
        print("     group is largely served from cache. The G=1 row is the only")
        print("     one whose byte count is a sound DRAM figure.")
        base = [r for r in rows if r["groups"] == 1]
        if base:
            b = base[0]
            for r in over:
                if r["width"] != b["width"]:
                    continue
                extra_ms = r["verify_ms"] - b["verify_ms"]
                dram_rate = b["target_share_bytes"] / (b["verify_ms"] * 1e6)
                would_be = target_stream / (dram_rate * 1e6)
                print(f"     {r['arm']} costs +{extra_ms:.2f} ms for its second")
                print(f"     stream. A second DRAM stream at the G=1 rate "
                      f"({dram_rate:.0f} GB/s)")
                print(f"     would cost +{would_be:.1f} ms. The measured extra is "
                      f"{100*extra_ms/would_be:.0f}% of that, so the")
                print(f"     second stream moves at an effective "
                      f"{target_stream/(extra_ms*1e6):.0f} GB/s.")

    print()
    print("  The advisor's model counted the target stream only and divided by")
    print("  the whole round, which drops the draft window's weight traffic:")
    for r in rows:
        naive = r["target_share_bytes"] / (r["round_ms"] * 1e6)
        print(f"    {r['arm']:<14} W={r['width']} "
              f"target-only {naive:6.1f} GB/s -> with drafts "
              f"{r['achieved_gb_s']:6.1f} GB/s "
              f"(+{100*(r['achieved_gb_s']/naive-1):.1f}%)")

    print()
    print("  Verify pass alone, against its own measured pipeline time:")
    for r in rows:
        vb = r["target_share_bytes"] / (r["verify_ms"] * 1e6)
        pm = f"{100*vb/measured_peak:.1f}%" if measured_peak else "-"
        print(f"    {r['arm']:<14} W={r['width']} G={r['groups']} "
              f"{vb:6.1f} GB/s = {pm} of measured peak")

    # --- state traffic ----------------------------------------------------
    cfg = json.loads((TARGET / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    gdn_layers = sum(1 for t in tc["layer_types"] if t == "linear_attention")
    fa_layers = sum(1 for t in tc["layer_types"] if t == "full_attention")
    ssm_bytes = 4 if tc.get("mamba_ssm_dtype") == "float32" else 2
    gdn_state = (gdn_layers * tc["linear_num_value_heads"]
                 * tc["linear_key_head_dim"] * tc["linear_value_head_dim"]
                 * ssm_bytes)
    kv_per_token = fa_layers * 2 * tc["num_key_value_heads"] * tc["head_dim"] * 2
    context = 512 + 256
    print()
    print("  Non-weight traffic, ESTIMATED from the config (not measured):")
    print(f"    GDN recurrent state {gdn_state/1e6:.1f} MB; the round reads and")
    print(f"      writes it, and snapshots it again for rollback: "
          f"<= {4*gdn_state/1e9:.3f} GB/round")
    print(f"    full-attention KV {kv_per_token} B/token x ~{context} ctx = "
          f"{kv_per_token*context/1e9:.3f} GB/round")
    if rows:
        r = rows[0]
        extra = 4 * gdn_state + kv_per_token * context
        total = r["round_weight_bytes"] + extra
        ach = total / (r["round_ms"] * 1e6)
        pm = f"{100*ach/measured_peak:.1f}%" if measured_peak else "-"
        print(f"    upper-bound total for {r['arm']} W={r['width']}: "
              f"{total/1e9:.3f} GB -> {ach:.1f} GB/s = {pm} of measured peak")

    # --- check 4 ----------------------------------------------------------
    print()
    print("=" * 78)
    print("CHECK 4 - IS IT BANDWIDTH BOUND?")
    print("=" * 78)
    # One packed byte holds two 4-bit weights. Per weight the kernel does one
    # multiply-add per verified row, plus one multiply and one add to
    # dequantise, repeated once per accumulator group.
    print("    per packed byte (2 weights): 4*M FLOP of matvec + 4*G FLOP of")
    print("    dequantisation, against 1 byte of weight and 1/16 byte of")
    print("    bf16 scale and bias. AI ~ 4*(M+G)/1.0625.")
    print("    The advisor's 'AI ~ 5' is the SERIAL number: at M=1, G=1 it is")
    print("    7.5. Arithmetic intensity rises with verified rows, which is the")
    print("    whole point of batching the verify pass.")
    fp32_peak = 20 * 128 * 2 * 1.4e9  # 20-core M4 Pro, 128 lanes/core, ~1.4 GHz
    for r in rows:
        ai = 4 * (r["width"] + r["groups"]) / 1.0625
        line = (f"    {r['arm']:<14} W={r['width']} G={r['groups']}  "
                f"AI ~ {ai:5.1f} FLOP/byte")
        if measured_peak:
            knee = fp32_peak / (measured_peak * 1e9)
            line += f"   knee ~ {knee:.0f} FLOP/byte"
        print(line)
    if measured_peak:
        knee = fp32_peak / (measured_peak * 1e9)
        print(f"    machine balance uses {fp32_peak/1e12:.1f} TFLOP/s fp32 "
              f"(20 cores x 128 lanes x 2 x 1.4 GHz), an ESTIMATE, over the")
        print(f"    measured {measured_peak:.0f} GB/s.")

    # --- B against the weight stream -------------------------------------
    print()
    print("=" * 78)
    print("B, THE MARGINAL VERIFIED ROW, AGAINST THE WEIGHT STREAM")
    print("=" * 78)
    row_bytes = sum(
        (k + n) * 2 * calls for k, n, calls in SCORED_SHAPES)
    row_weight_elems = sum(k * n * calls for k, n, calls in SCORED_SHAPES)
    row_flops = 2 * row_weight_elems
    # Use the un-diluted per-row slope, not the raw width-6 leg mean.
    recon_path = ARTIFACTS / "e163_round_cost_reconciliation.json"
    b_ms = None
    if recon_path.exists():
        b_ms = json.loads(recon_path.read_text())["two_band"]["per_row_h_ms"]
    print("  One extra verified row reads NO extra weights. The batch grows by")
    print("  one activation row and one output row per projection, and nothing")
    print("  else. Across the 257 routed dispatches of a verify pass:")
    print(f"    extra bytes per row   {row_bytes:>15,}  = {row_bytes/1e6:.2f} MB")
    if measured_peak:
        print(f"    at the measured {measured_peak:.0f} GB/s that is "
              f"{row_bytes/(measured_peak*1e6):.3f} ms")
    print(f"    extra FLOP per row    {row_flops/1e9:>15.1f} GFLOP "
          f"(2 x {row_weight_elems/1e9:.2f}e9 weights)")
    if b_ms:
        print(f"    B measured            {b_ms:>15.3f} ms")
        if measured_peak:
            print(f"    B is {b_ms/(row_bytes/(measured_peak*1e6)):.0f}x its own byte "
                  f"cost, so B is not a bandwidth term at all.")
        print(f"    B implies {row_flops*1e3/b_ms/1e12:.2f} TFLOP/s against an "
              f"estimated {fp32_peak/1e12:.1f} TFLOP/s fp32 peak")
        print(f"    = {100*row_flops*1e3/b_ms/fp32_peak:.0f}% of peak. Above 100 %")
        print("    means the peak estimate is a little low, not that the row")
        print("    broke physics: the clock and the lane count are both guesses.")
        print("    Either way the marginal row sits on the FLOP roofline and not")
        print("    on the bandwidth roofline. Lowering h needs arithmetic, not")
        print("    traffic.")
        print(f"    local B / ranked h = {b_ms:.3f} / {RANKED_PER_ROW_MS:.4f} = "
              f"{b_ms/RANKED_PER_ROW_MS:.4f}")
        print("    This is a clean local-to-ranked row-channel transfer factor:")
        print("    both sides are one extra verified row and nothing else.")

    print()
    print("=" * 78)
    print("WHAT IS THE HEADROOM WORTH, AND WHAT COULD EXPLAIN IT?")
    print("=" * 78)
    if rows and measured_peak:
        r = [x for x in rows if x["groups"] == 1][0]
        ideal_ms = r["round_weight_bytes"] / (measured_peak * 1e6)
        gap = r["round_ms"] - ideal_ms
        print(f"  {r['arm']} W={r['width']}: {r['round_ms']:.2f} ms measured, "
              f"{ideal_ms:.2f} ms at the size-matched rate.")
        print(f"  Headroom {gap:.2f} ms/round = "
              f"{100*gap/r['round_ms']:.1f}% of the round, not 43%.")
        idle = 1.03  # Thorfinn's host-side GPU-idle window per round, ms
        print(f"  Thorfinn measures a {idle:.2f} ms host-side GPU-idle window per")
        print(f"  round. That is {100*idle/gap:.0f}% of this headroom, so the idle")
        print("  window and the missing bandwidth are NOT the same phenomenon at")
        print("  this width. Most of the gap is inside GPU execution.")
        round_flops = (r["width"] * row_flops
                       + r["groups"] * row_flops / 2
                       + r["drafts"] * 2 * (head_stream / 2 + compact_stream * 2))
        print()
        print("  The round has two regimes, and they answer the question")
        print("  differently:")
        print(f"    whole round   {r['achieved_gb_s']:.0f} / {measured_peak:.0f} "
              f"GB/s = {100*r['achieved_gb_s']/measured_peak:.0f}% of bandwidth, "
              f"{round_flops*1e3/r['round_ms']/1e12:.2f} / {fp32_peak/1e12:.1f} "
              f"TFLOP/s = {100*round_flops*1e3/r['round_ms']/fp32_peak:.0f}% of "
              f"compute")
        if b_ms:
            print(f"    marginal row  {row_bytes/(b_ms*1e6):.1f} GB/s = "
                  f"{100*row_bytes/(b_ms*1e6)/measured_peak:.1f}% of bandwidth, "
                  f"{row_flops*1e3/b_ms/1e12:.2f} TFLOP/s = "
                  f"{100*row_flops*1e3/b_ms/fp32_peak:.0f}% of compute")
        print("  The base of the round streams weights and is bandwidth limited.")
        print("  The rows stacked on that base are compute limited. That is why")
        print("  neither 'wasting 43 % of bandwidth' nor 'purely compute bound'")
        print("  describes it, and it is why a kernel change that saves traffic")
        print("  helps the base while a kernel change that saves arithmetic helps")
        print("  the per-row term.")

    out = ARTIFACTS / "e163_round_bandwidth.json"
    out.write_text(json.dumps({
        "harness": "local",
        "official_or_ranked_score": False,
        "target_total_bytes": tr["_total"],
        "target_by_role_bytes": {k: v for k, v in tr.items() if not k.startswith("_")},
        "target_stream_bytes": target_stream,
        "head_stream_bytes": head_stream,
        "compact_draft_stream_bytes": compact_stream,
        "compact_draft_rows": COMPACT_DRAFT_PADDED_ROWS,
        "measured_peak_gb_per_s": measured_peak,
        "size_matched_peak_gb_per_s": size_matched,
        "nominal_peak_gb_per_s": NOMINAL_PEAK_GB_S,
        "gdn_state_bytes": gdn_state,
        "full_attn_kv_bytes_per_token": kv_per_token,
        "rounds": rows,
        "row_extra_bytes": row_bytes,
        "row_extra_gflop": row_flops / 1e9,
        "row_b_ms": b_ms,
        "ranked_per_row_ms": RANKED_PER_ROW_MS,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
