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

print("=" * 78)
print("A.  RANKED HOST ROUND TIMES (beagle, receipt ca9251b8, prefill removed)")
print("=" * 78)
K = SEED * R_PRE_SPT
ser_leg = DECODE * R_SER_SPT
mtp_leg = DECODE * R_MTP_SPT
ser_round = (ser_leg - K) / DECODE
n_rounds = DECODE / R_M
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

print("  -- ranked M5 Max (spec 614 GB/s; measured 3rd-party ~53 TFLOP/s bf16) --")
line("ranked serial round", ser_round, 1.0, SPEC_M5MAX, 53e12)
line("ranked beagle round", mtp_round, R_M, SPEC_M5MAX, 53e12)

print()
print("=" * 78)
print("C.  TRANSFER RATIOS local/ranked")
print("=" * 78)
print(f"  serial round        {LOCAL_ROUND_MS[1]/1000/ser_round:6.3f} x")
print(f"  M={R_M} verify round  {lm/1000/mtp_round:6.3f} x")
print(f"  spec bandwidth      {SPEC_M5MAX/SPEC_M4PRO:6.3f} x")
print(f"  width curve is flatter at rank by "
      f"{(lm/1000/mtp_round)/(LOCAL_ROUND_MS[1]/1000/ser_round):.3f} x")

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
