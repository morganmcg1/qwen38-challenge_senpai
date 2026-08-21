#!/usr/bin/env python3
"""Finding 36 candidate: is every streaming family at DRAM peak once a fixed
per-dispatch cost is removed?

Inputs are the E96 round census (research/finding22_reprice.py numbers) and the
model shape table.  The census ran at the G=2 operating point, so every weight
tensor is streamed twice per round.
"""

# name, K, N, layers, census us/round, census GB/s
FAM = [
    ("lm_head",       5120, 248320,  1,  5269.31, 271.9),
    ("mlp.gate_up",   5120,  34816, 64, 48381.86, 265.8),
    ("gdn.in_proj",   5120,  16480, 48, 17675.04, 258.4),
    ("fa.qkv",        5120,  14336, 16,  5163.37, 256.5),
]
# the anomaly family, two tensors sharing one census line
OUTDOWN = [
    ("gdn.out_proj+fa.o_proj", 6144,  5120, 64),
    ("mlp.down",              17408,  5120, 64),
]
OUTDOWN_US = 36559.21
OUTDOWN_GBS = 238.1

G = 2                 # weight streams per round at M>=5
BPE = 0.5625          # affine 4-bit group-64 bytes per element
DRAM_PEAK = 273.0     # GB/s, M4 Pro measured peak


def gb(K, N, layers):
    return K * N * BPE * layers * G / 1e9


print("=" * 78)
print("STEP 1  do the census GB/s figures reconcile with the shape table?")
print("=" * 78)
print(f"{'family':<26}{'GB/round':>10}{'implied GB/s':>14}{'census GB/s':>13}{'err %':>8}")
rows = []
for name, K, N, layers, us, gbs in FAM:
    b = gb(K, N, layers)
    implied = b / (us * 1e-6)
    print(f"{name:<26}{b:>10.4f}{implied:>14.1f}{gbs:>13.1f}{100*(implied/gbs-1):>8.2f}")
    rows.append((name, K, N, layers, us, b))

b_od = sum(gb(K, N, L) for _, K, N, L in OUTDOWN)
implied_od = b_od / (OUTDOWN_US * 1e-6)
print(f"{'out_proj + mlp.down':<26}{b_od:>10.4f}{implied_od:>14.1f}{OUTDOWN_GBS:>13.1f}"
      f"{100*(implied_od/OUTDOWN_GBS-1):>8.2f}")

print()
print("=" * 78)
print("STEP 2  fit  dispatch_us = F + bytes_GB * Sl   on the four clean families")
print("=" * 78)
pts = []
for name, K, N, layers, us, b in rows:
    ndisp = layers * G
    pts.append((name, ndisp, b / ndisp, us / ndisp))

print(f"{'family':<26}{'disp/rnd':>9}{'GB/disp':>10}{'us/disp':>10}{'GB/s':>9}")
for name, nd, bpd, upd in pts:
    print(f"{name:<26}{nd:>9d}{bpd:>10.5f}{upd:>10.2f}{bpd/(upd*1e-6):>9.1f}")

n = len(pts)
sx = sum(p[2] for p in pts)
sy = sum(p[3] for p in pts)
sxx = sum(p[2] * p[2] for p in pts)
sxy = sum(p[2] * p[3] for p in pts)
sl = (n * sxy - sx * sy) / (n * sxx - sx * sx)
F = (sy - sl * sx) / n
print()
print(f"  slope        {sl:10.1f} us per GB   ->  {1e6/sl:7.1f} GB/s "
      f"({100*(1e6/sl)/DRAM_PEAK:.1f} % of DRAM peak)")
print(f"  intercept F  {F:10.2f} us per dispatch")
print()
print(f"{'family':<26}{'us/disp':>10}{'predicted':>11}{'resid us':>10}{'resid %':>9}")
ss_res = ss_tot = 0.0
ybar = sy / n
for name, nd, bpd, upd in pts:
    pred = F + sl * bpd
    ss_res += (upd - pred) ** 2
    ss_tot += (upd - ybar) ** 2
    print(f"{name:<26}{upd:>10.2f}{pred:>11.2f}{upd-pred:>10.2f}{100*(upd/pred-1):>9.3f}")
print(f"\n  R^2 = {1 - ss_res/ss_tot:.8f}")

print()
print("=" * 78)
print("STEP 3  price the fixed cost over the whole round")
print("=" * 78)
tot_disp = sum(p[1] for p in pts) + sum(L * G for _, _, _, L in OUTDOWN)
LOCAL_ROUND = 127533.3   # E96 phase-table clone, us
print(f"  streaming dispatches per round          {tot_disp:8d}")
print(f"  fixed cost                              {tot_disp*F:8.0f} us")
print(f"  as a share of the {LOCAL_ROUND:.0f} us local round   "
      f"{100*tot_disp*F/LOCAL_ROUND:8.3f} %")

print()
print("=" * 78)
print("STEP 4  the out_proj + mlp.down anomaly")
print("=" * 78)
pred_od = 0.0
for name, K, N, L in OUTDOWN:
    nd = L * G
    b = gb(K, N, L)
    p = nd * F + b * sl
    pred_od += p
    print(f"  {name:<26} {nd:>4d} disp  {b:7.4f} GB  predicted {p:9.1f} us "
          f"({1e3*b/nd:6.2f} MB/disp)")
print(f"  predicted total                {pred_od:9.1f} us")
print(f"  measured  total                {OUTDOWN_US:9.1f} us")
exc = OUTDOWN_US - pred_od
nd_od = sum(L * G for _, _, _, L in OUTDOWN)
print(f"  EXCESS                         {exc:9.1f} us  = {100*exc/LOCAL_ROUND:.3f} % "
      f"of the local round  = {exc/nd_od:.2f} us per dispatch")
print(f"  excess as a fraction of the family {100*exc/OUTDOWN_US:.2f} %")

print()
print("=" * 78)
print("STEP 5  what the census 'rate deficits' really are")
print("=" * 78)
print(f"{'family':<26}{'census %peak':>13}{'peak after F':>14}")
for name, nd, bpd, upd in pts:
    census_pct = 100 * (bpd / (upd * 1e-6)) / DRAM_PEAK
    marg = 100 * (bpd / ((upd - F) * 1e-6)) / DRAM_PEAK
    print(f"{name:<26}{census_pct:>13.1f}{marg:>14.1f}")
bpd_od = b_od / nd_od
upd_od = OUTDOWN_US / nd_od
print(f"{'out_proj + mlp.down':<26}"
      f"{100*(bpd_od/(upd_od*1e-6))/DRAM_PEAK:>13.1f}"
      f"{100*(bpd_od/((upd_od-F)*1e-6))/DRAM_PEAK:>14.1f}")

print()
print("=" * 78)
print("STEP 6  does the law explain the M=1 round?  (E92 measured 64445.4 us)")
print("=" * 78)
disp_g1 = sum(L for _, _, _, L, _, _ in FAM) + sum(L for _, _, _, L in OUTDOWN)
bytes_g1 = sum(gb(K, N, L) for _, K, N, L, _, _ in FAM) / G \
    + sum(gb(K, N, L) for _, K, N, L in OUTDOWN) / G
pred_g1 = disp_g1 * F + bytes_g1 * sl
print(f"  G=1 streaming dispatches       {disp_g1:8d}")
print(f"  G=1 streaming bytes            {bytes_g1:8.4f} GB")
print(f"  predicted streaming time       {pred_g1:8.0f} us")
print(f"  E96 non-streaming subtotal     {5074.05:8.0f} us")
print(f"  predicted round                {pred_g1+5074.05:8.0f} us")
print(f"  E92 measured M=1 round         {64445.4:8.0f} us")
print(f"  unexplained                    {64445.4-pred_g1-5074.05:8.0f} us "
      f"({100*(64445.4-pred_g1-5074.05)/64445.4:.1f} %)")

print()
print("  NA=5 one-group control (alphonse E104 measured 103404 us, same 257 disp):")
print(f"    law predicts {pred_g1+5074.05:8.0f} us, measured 103404 us -> "
      f"{100*(103404/(pred_g1+5074.05)-1):.1f} % above the law")
print("    => the rate(NA) penalty is NOT the fixed cost. E104 stays a separate axis.")
