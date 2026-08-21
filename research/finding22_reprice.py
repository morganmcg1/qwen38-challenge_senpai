"""Finding 22 reprice: apply the two-class transfer law to the E96 census.

The law, in one line:

    ranked delta_us / local delta_us  =  (local achieved rate) / (ranked achieved rate)

For DRAM-bound work both rates are the machine's streaming bandwidth, so the
ratio is 249.55 / 542.8 = 0.460 and the PERCENTAGE is preserved.  For
latency-bound work neither rate scales with DRAM bandwidth, so the ratio is
about 1.0 and the PERCENTAGE is amplified by local_round / ranked_round.

Sanity check the law must pass: a DRAM-bound saving must keep its percentage,
because the item and the round it divides into scale by the same factor.
"""

LOCAL_ROUND_M5 = 126103.0   # Finding 2, local M=5 round busy us
E96_ROUND = 127533.0        # E96 census round us, same operating point
RANKED_ROUND_M5 = 53108.0   # Finding 12, ranked M=5 round us
DRAM_PEAK_M4PRO = 273.0     # GB/s
LOCAL_STREAM_RATE = 249.55  # GB/s, E96 census measured
RANKED_STREAM_RATE = 542.8  # GB/s, 28.8247 GB / 53108 us

AMP = E96_ROUND / RANKED_ROUND_M5           # latency-class percentage gain
STREAM_F = LOCAL_STREAM_RATE / RANKED_STREAM_RATE

# family, us/round isolated, achieved GB/s (None = not bandwidth bound)
FAMILIES = [
    ('MLP gate_up',                 48381.86, 265.8),
    ('out_proj + down_proj',        36559.21, 238.1),
    ('GDN in_proj',                 17675.04, 258.4),
    ('lm_head',                      5269.31, 271.9),
    ('attn fused QKV + gate',        5163.37, 256.5),
    ('GDN recurrent step',           1421.13, 212.5),
    ('SDPA over FA history',         1267.00, None),
    ('fused residual + RMSNorm',      771.54,  27.0),
    ('GDN prework',                   543.39,  32.6),
    ('q/k norm + RoPE',               149.85, None),
    ('KV cache write',                 89.10, None),
    ('MTP top-2 partial + finalize',    56.13, None),
]

# dose-ladder measurements, free of the isolation bias
DOSE = [
    ('GDN recurrent step',       861.0, 'stream'),
    ('fused residual + RMSNorm', 298.0, 'latency'),
]

PUB_FLOOR = 0.277   # 2-sigma single-pair, published frame
SF_FLOOR = 0.160    # 2-sigma single-pair, serial-free frame

BW_BOUND_CUT = 0.60  # achieved / peak above this = DRAM bound


def cls(gbs):
    if gbs is None:
        return 'latency?'
    return 'stream' if gbs / DRAM_PEAK_M4PRO >= BW_BOUND_CUT else 'latency'


print(f'local round  {E96_ROUND:9.1f} us      ranked round {RANKED_ROUND_M5:9.1f} us')
print(f'amplification for latency-class work  x {AMP:.3f}')
print(f'absolute-us factor for stream-class work x {STREAM_F:.3f}'
      f'   -> percentage x {STREAM_F * AMP:.3f}  (must be ~1.0)')
print()

hdr = (f'{"family":<30}{"us/rnd":>10}{"GB/s":>8}{"%peak":>7}'
       f'{"class":>10}{"local %":>10}{"ranked %":>10}')
print(hdr)
print('-' * len(hdr))

lat_iso = 0.0
str_iso = 0.0
for name, us, gbs in FAMILIES:
    c = cls(gbs)
    lp = 100.0 * us / E96_ROUND
    rp = lp * (AMP if c.startswith('latency') else STREAM_F * AMP)
    pk = f'{100 * gbs / DRAM_PEAK_M4PRO:6.1f}' if gbs else '     -'
    g = f'{gbs:8.1f}' if gbs else '       -'
    print(f'{name:<30}{us:10.2f}{g}{pk}{c:>10}{lp:9.3f}%{rp:9.3f}%')
    if c.startswith('latency'):
        lat_iso += us
    else:
        str_iso += us

print('-' * len(hdr))
print(f'{"STREAM subtotal":<30}{str_iso:10.2f}{"":8}{"":7}{"":10}'
      f'{100 * str_iso / E96_ROUND:9.3f}%{100 * str_iso / E96_ROUND * STREAM_F * AMP:9.3f}%')
print(f'{"LATENCY subtotal":<30}{lat_iso:10.2f}{"":8}{"":7}{"":10}'
      f'{100 * lat_iso / E96_ROUND:9.3f}%{100 * lat_iso / E96_ROUND * AMP:9.3f}%')
print()

print('LATENCY pool after the measured isolation discount (1.65x to 2.59x):')
for d in (1.0, 1.65, 2.59):
    ins = lat_iso / d
    print(f'  discount {d:4.2f}x  ->  in situ {ins:8.2f} us/round'
          f'   local {100 * ins / E96_ROUND:6.3f} %'
          f'   ranked {100 * ins / RANKED_ROUND_M5:6.3f} %')
print()

print('Dose-ladder items, the campaign default cost instrument:')
print(f'{"item":<30}{"us/rnd":>10}{"class":>10}{"local %":>10}'
      f'{"ranked %":>10}   verdict vs the 0.277 % published floor')
for name, us, c in DOSE:
    lp = 100.0 * us / E96_ROUND
    rp = lp * (AMP if c == 'latency' else STREAM_F * AMP)
    v = 'ALIVE' if rp >= PUB_FLOOR else 'below floor'
    was = 'ALIVE' if lp >= PUB_FLOOR else 'below floor'
    print(f'{name:<30}{us:10.1f}{c:>10}{lp:9.3f}%{rp:9.3f}%   '
          f'was {was:12s} now {v}')
print()

print('The closure threshold in LOCAL percentage terms:')
print(f'  a STREAM-class item is dead below       {SF_FLOOR:.3f} % local')
print(f'  a LATENCY-class item is dead below      {SF_FLOOR / AMP:.3f} % local'
      f'   ({PUB_FLOOR / AMP:.3f} % on the published floor)')
print(f'  items closed between those two bounds were closed on the wrong test.')
print()

print('Retired: Finding 13 fixed/launch transfer 0.670.')
print('  If a fixed-class local cost of 65,674 us transferred at 0.98 it would')
print(f'  need {65674 * 0.98:.0f} us of a {55870:.0f} us ranked round. Impossible.')
print('  So Finding 13 "fixed" is not latency work; it is streaming work that')
print('  the marginal-per-row model failed to attribute. Finding 21 supersedes')
print('  the split. Keep only the MEASURED head factor 0.236 and acceptance 1.0.')
