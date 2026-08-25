"""E219 desk: re-derive the stepq free optimum with the measured width curve.

harness=ranked for every printed number (FINDING 520 instrument; no GPU).

Corrections vs the E211 vintage forward laws, from FINDING 556/559 (E212
census, current tree) compared within-shape against the E186 receipt-vintage
curve:
  - m=6 cell: current tree is 9.152 ms/round cheaper locally (E195
    selective-m6; corroborated by the -9.15/+9.38 step pair). Transferred at
    the matched-vintage 6-row ratio. INFERRED (cross-vintage within-shape).
  - m=9 cell: measured post-E208 residual 19.113 ms replaces the constructed
    residual 19.053 ms (vintage dR9 - FINDING 543). Confirmation, not a shift.
  - m=7, m=8: unchanged within noise (+0.23 / +2.09 ms, no causal mechanism).
Baseline policy = the SHIPPED E214 stepq table, not the old uniform rule.
"""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, 'research')
import e211_step_price as M
import e197_refit as E
import e201_online_cap as O

RUN_LOO = len(sys.argv) > 1 and sys.argv[1] == 'loo'

rng = np.random.default_rng(219)
inst = O.build_instrument()
laws, cell = M.build_laws(inst['fit'])
gate = M.reproduction_gate(inst, laws)
if not gate['pass']:
    raise SystemExit('reproduction gate FAILED')
print('reproduction gate: PASS')

fit = inst['fit']
ratio = cell['transfer_ratio']
M6_LOCAL_DELTA_MS = 9.152
M9_RESIDUAL_MEAS_MS = 19.113
m6corr = ratio * M6_LOCAL_DELTA_MS
print('transfer ratio %.5f  m6 ranked correction -%.3f ms  m9 residual %.3f (was %.3f)'
      % (ratio, m6corr, M9_RESIDUAL_MEAS_MS, cell['residual_local_dr9_ms']))

base8 = [1000.0 * E.R_of(fit, m) for m in range(1, 9)]
sm = list(base8) + [1000.0 * E.R_of(fit, 9)]
sm[5] -= m6corr
se = list(base8)
se[5] -= m6corr
se = se + [base8[7] + ratio * M9_RESIDUAL_MEAS_MS]
laws_meas = {'smooth': E.table_law(sm), 'step_e208': E.table_law(se)}

FWD = ('smooth', 'step_e208')
tab_vint = {k: M.prompt_tables(inst, laws[k]) for k in FWD}
tab_meas = {k: M.prompt_tables(inst, laws_meas[k]) for k in FWD}

E214 = M.cuts_of([480, 619, 2103, 2604, 3315, 3315, 3315, 3315])
UNIF = M.ship_cuts(8)

def med(tables, cuts):
    r = M.evaluate(tables, cuts)
    return M.median_of([r[n] for n in M.ORDER])

print()
print('E214 stepq table vs uniform ship, per law:')
for tag, tabs in (('vintage', tab_vint), ('measured', tab_meas)):
    for k in FWD:
        a, b = med(tabs[k], E214), med(tabs[k], UNIF)
        print('  %-9s %-10s stepq %8.6f  uniform %8.6f  delta %+0.3f%%'
              % (tag, k, a, b, M.pct(a, b)))

ship_meas = {k: med(tab_meas[k], E214) for k in FWD}
floors_meas = {k: M.evaluate(tab_meas[k], E214) for k in FWD}

def optimise(pool):
    best_cuts, best_score = None, -1e18
    seeds = list(M.starts(rng)) + [list(E214)]
    for seed in seeds:
        cuts, score = M.ascend_minimax(tab_meas, ship_meas, pool, seed,
                                       floors_meas)
        if score > best_score:
            best_cuts, best_score = cuts, score
    return best_cuts

cuts_new = optimise(M.ORDER)
price = M.price_of_cuts(cuts_new)
agree, mism = M.verify_price(cuts_new, price)

print()
print('guarded forward minimax under MEASURED laws (baseline & floors = E214 table):')
print('  E214 cuts:', E214[1:M.MAXD + 1])
print('  new  cuts:', cuts_new[1:M.MAXD + 1])
print('  new thresholds:', [round(c / M.QGRID, 4) for c in cuts_new[1:M.MAXD + 1]])
print('  price_marginal:', [round(p, 5) for p in price['marginal']])
print('  greedy agreement: %.6f (%d mismatches)' % (agree, mism))
for k in FWD:
    raws = M.evaluate(tab_meas[k], cuts_new)
    v = M.median_of([raws[n] for n in M.ORDER])
    pp = {n: M.pct(raws[n], floors_meas[k][n]) for n in M.ORDER}
    print('  law %-10s median %8.6f  delta vs E214 %+0.4f%%  worst prompt %+0.4f%% (%s)'
          % (k, v, M.pct(v, ship_meas[k]), min(pp.values()),
             min(pp, key=lambda n: pp[n])))

if RUN_LOO:
    print()
    print('LOO-honest (fold optimum evaluated on held-out prompt):')
    honest = {k: {} for k in FWD}
    for held in M.ORDER:
        pool = [n for n in M.ORDER if n != held]
        fold = optimise(pool)
        for k in FWD:
            honest[k][held] = M.evaluate(tab_meas[k], fold)[held]
    for k in FWD:
        v = M.median_of([honest[k][n] for n in M.ORDER])
        print('  law %-10s LOO median %8.6f  delta vs E214 %+0.4f%%'
              % (k, v, M.pct(v, ship_meas[k])))

out = {'harness': 'ranked', 'ratio': ratio, 'm6corr_ms': m6corr,
       'e214_cuts': E214, 'new_cuts': cuts_new,
       'price_marginal': price['marginal']}
pathlib.Path('/tmp/e219_desk_out.json').write_text(json.dumps(out, indent=1, default=float))
print('\nwrote /tmp/e219_desk_out.json')
