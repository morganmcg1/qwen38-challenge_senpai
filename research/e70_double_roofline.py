#!/usr/bin/env python3
"""Double roofline: is the scored verify kernel bandwidth-bound or compute-bound?"""

# --- measured local peaks (edward E63 rung 1, own host, committed artifact) ---
PEAK_BW = 226035064068.14008          # B/s
PEAK_FLOPS = 7505967353490.118        # FLOP/s bf16 GEMM

# --- Apple official spec bandwidth ---
SPEC_M4PRO = 273e9      # 20-core M4 Pro
SPEC_M5MAX = 614e9      # M5 Max 40-core GPU / 128 GiB  (Apple newsroom 2026-03-03)

# --- master traffic fact (ledger 199(A)) ---
W = 14_412_349_440       # bytes: all quantized linears minus embed_tokens
PARAMS = 26.9e9          # backbone params
FLOP_PER_ROW = 2 * PARAMS

# --- E1 whole-model per-round times at local width M (ms) ---
LOCAL_ROUND_MS = {1: 65.009, 2: 70.482, 3: 75.519, 4: 91.288,
                  5: 115.691, 6: 134.668, 7: 154.169, 8: 172.827, 9: 198.237}

# --- E61 / E64 measured achieved bandwidth of the QMV cell alone, per NA ---
CELL_BW_E61 = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946, 6: 117.8, 7: 97.9}
CELL_BW_E64 = {2: 218.7, 3: 216.6, 4: 185.4, 5: 155.1, 6: 120.5, 7: 95.2}

# --- ranked receipt (ca9251b8, beagle) ---
R_MTP_SPT = 0.0121740429
R_SER_SPT = 0.0379848885
R_PRE_SPT = 0.001027271
R_M = 5.5327
DECODE = 512
SEED = 512

# Exact receipt round accounting, not a model. mean_draft_len * R must be a whole
# number of proposals and R + accepted == 512, which pins beagle at R = 107.
# The verify width M = 1 + drafts PROPOSED; tokens per round = 1 + drafts
# ACCEPTED. Beagle accepts 405 of 485, so tokens per round is 4.785, not M.
R_ROUNDS = 107
R_PROPOSED = 485
R_ACCEPTED = 405
assert R_ROUNDS + R_ACCEPTED == DECODE
assert abs((R_M - 1.0) - R_PROPOSED / R_ROUNDS) < 1e-4
R_M = 1.0 + R_PROPOSED / R_ROUNDS

print("=" * 78)
print("A.  RANKED HOST ROUND TIMES (beagle, receipt ca9251b8, prefill removed)")
print("=" * 78)
K = SEED * R_PRE_SPT
ser_leg = DECODE * R_SER_SPT
mtp_leg = DECODE * R_MTP_SPT
ser_round = (ser_leg - K) / DECODE
n_rounds = R_ROUNDS
mtp_round = (mtp_leg - K) / n_rounds
print(f"  seed prefill K                = {K*1000:8.3f} ms")
print(f"  serial leg                    = {ser_leg:8.4f} s   round = {ser_round*1000:7.3f} ms")
print(f"  mtp    leg                    = {mtp_leg:8.4f} s   round = {mtp_round*1000:7.3f} ms  (M={R_M})")

print()
print("=" * 78)
print("B.  ACHIEVED BANDWIDTH AND COMPUTE, WHOLE ROUND")
print("=" * 78)
hdr = f"  {'host/round':<28}{'GB/s':>9}{'%specBW':>9}{'TFLOP/s':>10}{'%compute':>10}"
print(hdr)


def line(name, secs, rows, spec, peak_flops):
    bw = W / secs
    fl = FLOP_PER_ROW * rows / secs
    print(f"  {name:<28}{bw/1e9:9.1f}{100*bw/spec:9.1f}{fl/1e12:10.2f}"
          f"{100*fl/peak_flops:10.1f}")


# local: measured peaks are what the host can actually reach
print("  -- local M4 Pro (peaks: %.1f GB/s measured / %.2f TFLOP/s measured) --"
      % (PEAK_BW / 1e9, PEAK_FLOPS / 1e12))
for m in (1, 4, 5, 6, 9):
    line(f"local depth-0 round" if m == 1 else f"local M={m} round",
         LOCAL_ROUND_MS[m] / 1000.0, m, PEAK_BW, PEAK_FLOPS)
# interpolate local at ranked M
lm = LOCAL_ROUND_MS[5] + (R_M - 5) * (LOCAL_ROUND_MS[6] - LOCAL_ROUND_MS[5])
line(f"local M={R_M} (interp)", lm / 1000.0, R_M, PEAK_BW, PEAK_FLOPS)

# Ledger 206(I): the receipt carries two builds. `ser_round` is the runner's
# pinned BASELINE build; `cand_depth0` is our CANDIDATE build at depth 0, model-
# fit from plutarch by research/prompt_round_reconstruction.py. Only
# candidate:candidate ratios are transfer ratios.
CAND_DEPTH0_MS = 30.402

print("  -- ranked M5 Max (spec 614 GB/s; measured 3rd-party ~53 TFLOP/s bf16) --")
line("ranked baseline serial round", ser_round, 1.0, SPEC_M5MAX, 53e12)
line("ranked candidate depth-0", CAND_DEPTH0_MS / 1000.0, 1.0, SPEC_M5MAX, 53e12)
line("ranked candidate beagle round", mtp_round, R_M, SPEC_M5MAX, 53e12)

print()
print("=" * 78)
print("C.  TRANSFER RATIOS local/ranked  (candidate build on both sides only)")
print("=" * 78)
r_depth0 = LOCAL_ROUND_MS[1] / CAND_DEPTH0_MS
r_width = lm / 1000 / mtp_round
print(f"  depth-0 round       {r_depth0:6.3f} x   (candidate:candidate)")
print(f"  M={R_M} verify round  {r_width:6.3f} x   (candidate:candidate)")
print(f"  spec bandwidth      {SPEC_M5MAX/SPEC_M4PRO:6.3f} x")
local_pen = lm / LOCAL_ROUND_MS[1]
ranked_pen = mtp_round * 1000 / CAND_DEPTH0_MS
print(f"  width penalty M=1 -> M={R_M}:  local {local_pen:5.3f} x   "
      f"ranked {ranked_pen:5.3f} x")
print(f"  local width curve is {local_pen/ranked_pen:.3f} x as steep as ranked")
print(f"  => R is width dependent: R(1) = {r_depth0:.4f}, "
      f"R({R_M}) = {r_width:.4f}")
print()
print("  -- ledger 206(M): banked non-speculative advantage over the pinned base --")
print(f"  baseline serial round / candidate depth-0 = "
      f"{(ser_round * 1000) / CAND_DEPTH0_MS:6.4f} x")

print()
print("=" * 78)
print("D.  THE QMV CELL ALONE: compute rate per NA (E1 standalone stream costs)")
print("=" * 78)
S = {1: 65.009, 2: 64.40, 3: 72.17, 4: 82.24, 5: 95.48, 6: 122.34, 7: 147.21}
print(f"  {'NA':>3}{'ms':>9}{'rel FLOP rate':>16}{'E61 GB/s':>10}{'%peak':>8}"
      f"{'TFLOP/s':>10}{'%peak':>8}")
best = None
for na in sorted(S):
    rate = na / S[na]
    bw = CELL_BW_E61.get(na)
    fl = FLOP_PER_ROW * na / (S[na] / 1000.0)
    s = f"  {na:>3}{S[na]:9.2f}{rate:16.5f}"
    if bw:
        s += f"{bw:10.1f}{100*bw*1e9/PEAK_BW:8.1f}"
    else:
        s += " " * 18
    s += f"{fl/1e12:10.2f}{100*fl/PEAK_FLOPS:8.1f}"
    print(s)
    if best is None or rate > best[1]:
        best = (na, rate)
print(f"  --> compute throughput PEAKS at NA={best[0]} and declines after")
