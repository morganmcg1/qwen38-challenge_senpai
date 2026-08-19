"""E33: per-rows_per_simd register laws, refit from askeladd's published E32 grid.

The within-r laws are exact on every spill-free coverage-preserving cell. The
cross-r slope law is not affine and cannot be fit by a two-parameter form on the
three r values that tile the frozen four rows.
"""
import json

GRID = 'research/e32-rps-grid.json'
ADVISOR_SLOPE = (8.00, 3.30)  # slope(r) = a + b*r, from the E36 feedback


def fit(pts):
    m = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    b = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    a = (sy - b * sx) / m
    return a, b, max(abs(y - (a + b * x)) for x, y in pts)


cells = json.load(open(GRID))['cells']
by_r = {}
for c in cells:
    if c['arm'] == 'coverage_preserving' and c['status'] == 'ok':
        by_r.setdefault(c['r'], {})[c['na']] = (c['peak_live_regs'], c['acc_spill'])

slopes = []
for r in sorted(by_r):
    row = by_r[r]
    nas = sorted(row)
    print('r=%d  ' % r + ' '.join('NA%d:%d%s' % (n, row[n][0], '*' if row[n][1] else '')
                                  for n in nas))
    clean = [n for n in nas if not row[n][1]]
    a, b, res = fit([(n, row[n][0]) for n in clean])
    slopes.append((r, b))
    print('   spill-free fit over NA%s: regs = %.3f + %.3f*NA   maxres = %.2f'
          % (clean, a, b, res))
    print('   advisor slope(r) = %.2f + %.2f*r = %.2f   (error %+.2f per NA)'
          % (ADVISOR_SLOPE[0], ADVISOR_SLOPE[1],
             ADVISOR_SLOPE[0] + ADVISOR_SLOPE[1] * r,
             ADVISOR_SLOPE[0] + ADVISOR_SLOPE[1] * r - b))

a, b, res = fit(slopes)
print()
print('cross-r: measured slopes %s' % ['r=%d:%.0f' % s for s in slopes])
print('  best affine-in-r fit : slope(r) = %.3f + %.3f*r   maxres = %.2f' % (a, b, res))
print('  advisor              : slope(r) = %.2f + %.2f*r   maxres = %.2f'
      % (ADVISOR_SLOPE[0], ADVISOR_SLOPE[1],
         max(abs(s - (ADVISOR_SLOPE[0] + ADVISOR_SLOPE[1] * r)) for r, s in slopes)))
print('  => slope is concave in r; use the exact per-r table, do not interpolate.')

print()
print('E33 production range, r=2:')
for n in (6, 7, 8, 9):
    v, sp = by_r[2][n]
    print('  NA=%d regs=%d spill=%s' % (n, v, sp))
print('  first differences:', [by_r[2][n + 1][0] - by_r[2][n][0] for n in (6, 7, 8)])
print()
print('* = accumulator spill (excluded from the fit; the r=4 law breaking at NA=6 IS')
print('  the spill signature: it predicts 146 and measures 144)')
