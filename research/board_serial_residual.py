"""Is the serial leg correlated with the candidate leg inside one ranked run?

Thorfinn asked the right first question in PR #89 comment 47: before hunting
where a published-frame residual lives, test whether one exists at all. His
test compared two variances, `D = var(published) - var(serialfree)`, and got
`t = +1.08` on 39 byte-identical replicate pairs. That test is weak because it
throws away the pairing between the two statistics.

This script uses the pairing. For one run, with `S` the two prompts the median
selects,

    published   ~ mean over S of  serial_p / mtp_p
    serialfree  ~ mean over S of  sbar_p   / mtp_p        sbar = board-mean serial

so their log difference

    u = ln(published) - ln(serialfree)

contains no candidate-leg term at all. It is a pure serial-draw statistic. On a
replicate pair the three deltas satisfy, exactly,

    var(d_pub) = var(d_sf) + var(d_u) + 2 cov(d_sf, d_u)

Under independent legs the covariance is zero and `D` is just `var(d_u)`. A
run-level factor that speeds or slows both legs together -- the `g != 1` case
thorfinn flagged -- shows up as `cov(d_sf, d_u) > 0`. Testing that covariance
directly is far more powerful than differencing two variances, because the two
statistics share every candidate-side fluctuation.

Inputs, both produced outside this repository:

    /tmp/yukon-board/full.json
    /tmp/yukon-board/treedigest.json

Usage:  python3 research/board_serial_residual.py
"""
import json
import math

BOARD = '/tmp/yukon-board/full.json'
DIGEST = '/tmp/yukon-board/treedigest.json'

# A discrete run-level mode adds about 0.601 ms per drafting round to the
# candidate leg only (FACT 2). It is ~1.2 % on the score, far outside the
# 0.16 % serial-free noise, so a cut on |d_sf| separates the two cleanly.
MODE_CUT_PCT = 0.60


def load_rows():
    payload = json.load(open(BOARD))
    rows = payload
    for k in ('submissions', 'rows', 'data', 'items'):
        if isinstance(rows, dict) and k in rows:
            rows = rows[k]
            break
    return [r for r in rows if isinstance(r, dict)]


def perprompt(r):
    om = r.get('officialMetrics') or {}
    pp = (om.get('per_prompt') if isinstance(om, dict) else None) or []
    out = {}
    for e in pp:
        if isinstance(e, dict) and e.get('prompt_sha256'):
            out[e['prompt_sha256'][:8]] = e
    return out


def median8(vals):
    s = sorted(vals)
    return 0.5 * (s[3] + s[4])


def mean(xs):
    return sum(xs) / len(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def main():
    rows = load_rows()
    dig = json.load(open(DIGEST))

    recs = []
    for r in rows:
        sid = r.get('id')
        if sid not in dig or not isinstance(r.get('officialScore'), (int, float)):
            continue
        pp = perprompt(r)
        if len(pp) != 8:
            continue
        try:
            ser = {k: v['serial_seconds_per_token_mean'] for k, v in pp.items()}
            mtp = {k: v['mtp_seconds_per_token_mean'] for k, v in pp.items()}
            dl = {k: v['effective_mean_draft_len'] for k, v in pp.items()}
        except KeyError:
            continue
        if any(not x for x in ser.values()) or any(not x for x in mtp.values()):
            continue
        recs.append(dict(id=sid, dig=dig[sid], ser=ser, mtp=mtp, dl=dl,
                         score=r['officialScore'], user=r.get('solverUsername'),
                         created=r.get('createdAt')))

    # board-mean serial per prompt, over every scored run
    keys = sorted(recs[0]['ser'])
    sbar = {k: mean([x['ser'][k] for x in recs]) for k in keys}

    for x in recs:
        x['pub'] = median8([x['ser'][k] / x['mtp'][k] for k in keys])
        x['sf'] = median8([sbar[k] / x['mtp'][k] for k in keys])
        x['u'] = math.log(x['pub']) - math.log(x['sf'])

    err = max(abs(x['pub'] - x['score']) for x in recs)
    print(f'scored runs with a tree digest: {len(recs)}')
    print(f'median-of-8 reproduces officialScore to {err:.3g}')

    # replicate pairs: identical scored subtree and identical draft schedule
    pairs = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            if a['dig'] != b['dig']:
                continue
            if any(abs(a['dl'][k] - b['dl'][k]) > 1e-3 for k in keys):
                continue
            pairs.append((a, b))

    print(f'byte-identical replicate pairs: {len(pairs)}')
    print()

    def report(label, sub):
        if len(sub) < 4:
            print(f'{label}: only {len(sub)} pairs, skipped')
            return
        dpub = [100 * (math.log(a['pub']) - math.log(b['pub'])) for a, b in sub]
        dsf = [100 * (math.log(a['sf']) - math.log(b['sf'])) for a, b in sub]
        du = [100 * (a['u'] - b['u']) for a, b in sub]

        n = len(sub)
        vpub, vsf, vu = var(dpub), var(dsf), var(du)
        # per-run sds: a pair difference has twice the per-run variance
        print(f'--- {label}  n = {n} pairs ---')
        print(f'  per-run sd  published  {math.sqrt(vpub / 2):7.4f} %')
        print(f'  per-run sd  serialfree {math.sqrt(vsf / 2):7.4f} %')
        print(f'  per-run sd  u (serial) {math.sqrt(vu / 2):7.4f} %')
        D = vpub - vsf
        print(f'  D = var(d_pub) - var(d_sf)   {D:9.6f}')
        print(f'  var(d_u), the g=1 prediction {vu:9.6f}')
        print(f'  excess D - var(d_u)          {D - vu:9.6f}'
              f'   = 2 * cov(d_sf, d_u)')

        # The direct paired test. Pair order is arbitrary, but flipping a pair
        # negates dsf and du together, so an UNCENTERED correlation is
        # order-invariant. Never orient by dsf: that replaces dsf with |dsf|,
        # whose variance is sigma^2 (1 - 2/pi), and inflates r by 1.659x.
        sxy = sum(x * y for x, y in zip(dsf, du))
        sxx = sum(x * x for x in dsf)
        syy = sum(y * y for y in du)
        r = sxy / math.sqrt(sxx * syy)
        se_r = (1.0 - r * r) / math.sqrt(n)
        t = r / se_r
        print(f'  corr(d_sf, d_u)  r = {r:+.4f}   se {se_r:.4f}   t = {t:+.2f}')
        det = 2.0 * se_r
        print(f'  detectable |r| at 2 sigma with n = {n}:  {det:.3f}')
        if abs(t) < 2.0:
            print('  VERDICT: no run-level coupling between the serial and')
            print('           candidate legs is detected. g = 1 is not refuted.')
        else:
            print('  VERDICT: the legs are coupled. g != 1.')
        print()

    report('all replicate pairs', pairs)
    same = [(a, b) for a, b in pairs
            if abs(100 * (math.log(a['sf']) - math.log(b['sf']))) < MODE_CUT_PCT]
    report(f'mode-matched pairs, |d_sf| < {MODE_CUT_PCT} %', same)

    print('Empirical floors, no model, from the mode-matched pairs:')
    if len(same) >= 4:
        dpub = [abs(100 * (math.log(a['pub']) - math.log(b['pub']))) for a, b in same]
        dsf = [abs(100 * (math.log(a['sf']) - math.log(b['sf']))) for a, b in same]
        dpub.sort()
        dsf.sort()
        mid = len(dpub) // 2
        print(f'  median |pair gap| published  {dpub[mid]:.4f} %'
              f'   max {dpub[-1]:.4f} %')
        print(f'  median |pair gap| serialfree {dsf[mid]:.4f} %'
              f'   max {dsf[-1]:.4f} %')
        # Campaign convention: the floor is the sd of ONE pair difference,
        # which is sqrt(2) x the per-run sd. Adopted values are published
        # 0.277 % and serial-free 0.160 % (thorfinn, E87 s13).
        print(f'  pair-difference sd, published  '
              f'{math.sqrt(var([100 * (math.log(a["pub"]) - math.log(b["pub"])) for a, b in same])):.3f} %'
              f'   (adopted campaign floor 0.277 %)')
        print(f'  pair-difference sd, serialfree '
              f'{math.sqrt(var([100 * (math.log(a["sf"]) - math.log(b["sf"])) for a, b in same])):.3f} %'
              f'   (adopted campaign floor 0.160 %)')


if __name__ == '__main__':
    main()
