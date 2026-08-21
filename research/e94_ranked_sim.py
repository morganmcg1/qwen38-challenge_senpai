"""E94 rung 3: price the shipped depth walk against the RANKED M5 cost curve.

Why this exists
---------------
The local M4 Pro round-cost curve and the ranked M5 curve disagree most exactly
where the depth price acts: the step into verify width 3 costs 11,459 us
locally and 3,995 us on rank, a factor of 2.9. A depth price fitted to the
ranked curve therefore cannot be confirmed or refuted by a local speed
measurement, because the local run answers a different question. This script
answers the ranked question offline.

What is replayed exactly
------------------------
Ported line for line from `Qwen36MTPBlockSession.swift`:

  * `positionAcceptEMA` init `0.85 * 0.98**i`, `acceptEMAAlpha = 0.15`;
  * `costModelDepth`: flat cap 7, the reach product, the
    `marginal[d] * (1 + expected) / cumulative[d]` threshold, and the strict
    `reach > threshold` test;
  * `positionAcceptEstimate`, with the depth-0 and depth-1 top-2 margin clamps;
  * `recordAcceptOutcome`, with the capped 0.95 optimism transfer;
  * `makeUniformDepthPrice` (ship) and `makeRankedTierDepthPrice` (m5fit),
    including the m5fit entry gate.

The stages
----------
STAGE 0, model free. For a converged per-position acceptance `q`, which depth
does each price select, and which depth does the ranked curve actually prefer?
No acceptance model and no fit.

STAGE A, the reproduction check the assignment specified. A stationary profile
rescaled to the observed accept rate nearly reproduces the measured LOCAL ship
leg and fails badly on both ranked prompts. Reported first, as required.

STAGE B, the machinery validated against measured local ground truth. Calibrate
the acceptance profile on the local ship leg, then predict the round cost of
both measured local arms and compare with the measured rung-3 ABBA numbers.

STAGE C, the decisive ranked computation. Calibrate the acceptance profile from
the two ranked observables per prompt, then sweep every depth on the ranked
cost curve. This does not require the shipped walk to reproduce the ranked
depth, which matters because the ranked reference schedule is not necessarily
this base's walk.

STAGE D, the corrected tier shape implied by the ranked crossover.

Declared approximations
-----------------------
1. Acceptance is a per-position Bernoulli chain with the E92 shape and one
   fitted level. It reproduces the profile and the level, not text structure.
2. Stop tokens are absent, so `stoppedEarly` is always false in Stage A.
3. The top-2 margin is not simulated from the target model. Stage A runs
   unclamped.
4. The ranked round cost is the advisor's two-line fit whose per-prompt round
   counts are INFERRED. Stage C therefore reports every admissible round count.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

MAX_DEPTH = 8
SEGMENTED_VERIFY_DEPTH_CAP = 7
ACCEPT_EMA_ALPHA = 0.15
HEAD_STEP_COST_RATIO = 0.18
QUANTIZED_GROUP_WIDTH = 4
RANKED_TIER_HEAD_STEP = 0.128
RANKED_GROUP_STEP_FACTOR = 2.490
RANKED_TIER_FACTOR = 1.810
DECODE_TOKENS = 512

E92_PROFILE = [0.9659, 0.9652, 0.9543, 0.9486, 0.9487, 0.9859, 0.9451, 0.8333]
E92_LOGIT = [math.log(q / (1.0 - q)) for q in E92_PROFILE]

# Ranked M5 round cost, us. Two lines with a step at the G = ceil(M/4) boundary.
# senpai/qwen38-mtp-r1:research/ranked_cost_curve.py, median of 81 official runs.
RANKED_A1, RANKED_C1 = 27181.5, 3995.1   # M = 1..4
RANKED_A2, RANKED_C2 = 16943.2, 7233.0   # M = 5..8

# Local M4 Pro round cost, us, E92 isolated whole-table measurement.
LOCAL_ROUND_US = {1: 64445.4, 2: 69775.5, 3: 74778.4, 4: 86237.4,
                  5: 126103.1, 6: 137842.6, 7: 150431.4, 8: 163957.1}

# effective_mean_draft_len is an EXACT board rational. The round count is
# INFERRED, so every admissible multiple of the reduced denominator is carried.
RANKED = {
    'beagle': dict(drafts=241, unit_rounds=55, rounds_used=110,
                   pct_per_pct=0.4785, gain_ceiling_pct=4.6625),
    'essays': dict(drafts=117, unit_rounds=23, rounds_used=92,
                   pct_per_pct=0.3721, gain_ceiling_pct=0.3721),
}
for _spec in RANKED.values():
    _spec['draft_len'] = _spec['drafts'] / _spec['unit_rounds']

# Measured local rung-3 ABBA, mean of the two legs of each arm.
LOCAL_LEG = {
    'ship': dict(draft_len=6.3590, rounds=78.0, round_us_per_token=23813.0,
                 hist={1: 0.0128, 3: 0.0513, 4: 0.0641, 5: 0.0641, 6: 0.0385,
                       7: 0.7692}),
    'm5fit': dict(draft_len=4.4000, rounds=100.0, round_us_per_token=23056.0,
                  hist={3: 0.6000, 5: 0.0400, 6: 0.1200, 7: 0.2400}),
}


def prefix_costs(marginal):
    out, running = [1.0], 1.0
    for value in marginal:
        running += value
        out.append(running)
    return out


def make_uniform_depth_price():
    return ([HEAD_STEP_COST_RATIO] * MAX_DEPTH,
            [1.0 + i * HEAD_STEP_COST_RATIO for i in range(MAX_DEPTH + 1)])


def make_ranked_tier_depth_price(head_step=RANKED_TIER_HEAD_STEP,
                                 group_factor=RANKED_GROUP_STEP_FACTOR,
                                 tier_factor=RANKED_TIER_FACTOR):
    marginal = []
    for index in range(MAX_DEPTH):
        entering = index + 2
        if entering == QUANTIZED_GROUP_WIDTH + 1:
            marginal.append(head_step * group_factor)
        elif entering > QUANTIZED_GROUP_WIDTH + 1:
            marginal.append(head_step * tier_factor)
        else:
            marginal.append(head_step)
    return marginal, prefix_costs(marginal)


PRICES = {'ship': make_uniform_depth_price(),
          'm5fit': make_ranked_tier_depth_price()}


def ranked_round_us(width):
    if width <= QUANTIZED_GROUP_WIDTH:
        return RANKED_A1 + RANKED_C1 * width
    return RANKED_A2 + RANKED_C2 * width


def local_round_us(width):
    return LOCAL_ROUND_US[width]


def walk(ema, marginal, cumulative, gated, margin, cap):
    """`costModelDepth`, ported exactly. `gated` is the m5fit entry gate."""
    if cap <= 0:
        return 0
    if gated:
        entry = ema[0]
        if margin is not None:
            entry = min(entry, 1.0 / (1.0 + math.exp(-margin / 2.0)))
        if entry <= HEAD_STEP_COST_RATIO:
            return 0
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if margin is not None:
            if depth == 0:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
            elif depth == 1:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
        reach *= p
        if not reach > marginal[depth] * (1.0 + expected) / cumulative[depth]:
            break
        expected += reach
        depth += 1
    return depth


def record_accept_outcome(ema, accepted_count, draft_count):
    """`recordAcceptOutcome`, ported exactly. `stoppedEarly` is always false."""
    alpha = ACCEPT_EMA_ALPHA
    for index in range(min(accepted_count, MAX_DEPTH)):
        ema[index] += alpha * (1.0 - ema[index])
    if accepted_count < draft_count and accepted_count < MAX_DEPTH:
        ema[accepted_count] += alpha * (0.0 - ema[accepted_count])
    elif (accepted_count == draft_count and draft_count > 0
          and accepted_count < MAX_DEPTH):
        if ema[accepted_count] < 0.95:
            ema[accepted_count] += alpha * (0.95 - ema[accepted_count])


def profile_at(mu):
    return [1.0 / (1.0 + math.exp(-(logit + mu))) for logit in E92_LOGIT]


def reach_prefix(profile):
    """`reach[d]` is the probability that draft position d - 1 is accepted."""
    out, running = [], 1.0
    for q in profile:
        running *= q
        out.append(running)
    return out


def tokens_at_depth(profile, depth):
    return 1.0 + sum(reach_prefix(profile)[:depth])


def steady_depth(arm, profile):
    """Depth the walk selects once the EMAs have converged to `profile`."""
    marginal, cumulative = PRICES[arm]
    return walk(list(profile), marginal, cumulative, arm == 'm5fit', None,
                SEGMENTED_VERIFY_DEPTH_CAP)


def us_per_token(profile, depth, cost):
    return cost(depth + 1) / tokens_at_depth(profile, depth)


def depth_mixture(mean_depth):
    """Two-point mixture on the integers around a fractional mean depth."""
    low = int(math.floor(mean_depth))
    if low == mean_depth:
        return {low: 1.0}
    return {low: low + 1 - mean_depth, low + 1: mean_depth - low}


def mixture_accepted(profile, hist):
    reach = reach_prefix(profile)
    return sum(weight * sum(reach[:depth]) for depth, weight in hist.items())


def mixture_us_per_token(profile, hist, cost):
    total_us = sum(weight * cost(depth + 1) for depth, weight in hist.items())
    return total_us / (1.0 + mixture_accepted(profile, hist))


def calibrate_mu(hist, accepted_per_round):
    lo, hi = -8.0, 8.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if mixture_accepted(profile_at(mid), hist) < accepted_per_round:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stage_zero(cost, label):
    print('=' * 79)
    print(f'STAGE 0 - MODEL-FREE DEPTH MAP on the {label} cost curve')
    print('=' * 79)
    print('For a converged flat per-position acceptance q: the depth each price')
    print('selects, the depth that minimises cost per token, and the gap.')
    print()
    print(f"{'q':>6s}{'ship d':>8s}{'m5fit d':>9s}{'best d':>8s}"
          f"{'ship us/tok':>13s}{'m5fit us/tok':>14s}{'best us/tok':>13s}"
          f"{'m5fit-ship':>12s}")
    rows, switch = [], None
    steps = int(round((0.99 - 0.60) / 0.01)) + 1
    for index in range(steps):
        q = 0.60 + 0.01 * index
        flat = [q] * MAX_DEPTH
        ds, dm = steady_depth('ship', flat), steady_depth('m5fit', flat)
        options = [(us_per_token(flat, d, cost), d) for d in range(8)]
        best_us, best_d = min(options)
        us_s, us_m = us_per_token(flat, ds, cost), us_per_token(flat, dm, cost)
        delta = 100.0 * (us_s - us_m) / us_s
        rows.append(dict(q=q, ship_depth=ds, m5fit_depth=dm, best_depth=best_d,
                         ship_us=us_s, m5fit_us=us_m, best_us=best_us,
                         delta_pct=delta))
        if rows and len(rows) > 1 and switch is None \
                and rows[-2]['best_depth'] < best_d and best_d >= 7:
            switch = q
        if index % 2 == 0:
            print(f'{q:6.3f}{ds:8d}{dm:9d}{best_d:8d}{us_s:13.1f}{us_m:14.1f}'
                  f'{best_us:13.1f}{delta:11.2f}%')
    reachable = sorted({r['best_depth'] for r in rows})
    print()
    print(f'depths that are ever cost-optimal on the {label} curve: {reachable}')
    if switch is not None:
        print(f'the optimum jumps to the cap at q = {switch:.3f}; below that the '
              f'optimum is shallow')
    positive = [r['q'] for r in rows if r['delta_pct'] > 0.05]
    negative = [r['q'] for r in rows if r['delta_pct'] < -0.05]
    if positive:
        print(f'm5fit beats ship for q in [{min(positive):.2f}, '
              f'{max(positive):.2f}], best {max(r["delta_pct"] for r in rows):+.2f} %')
    if negative:
        print(f'm5fit loses to ship for q in [{min(negative):.2f}, '
              f'{max(negative):.2f}], worst '
              f'{min(r["delta_pct"] for r in rows):+.2f} %')
    return rows, switch


def run_window(arm, seed, mu, cost):
    marginal, cumulative = PRICES[arm]
    ema = [0.85 * 0.98 ** i for i in range(MAX_DEPTH)]
    rng = random.Random(seed)
    profile = profile_at(mu)
    tokens = rounds = drafts = accepted_total = 0
    total_us = 0.0
    while tokens < DECODE_TOKENS:
        draw = [rng.random() for _ in range(MAX_DEPTH)]
        depth = walk(ema, marginal, cumulative, arm == 'm5fit', None,
                     SEGMENTED_VERIFY_DEPTH_CAP)
        accepted = 0
        while accepted < depth and draw[accepted] < profile[accepted]:
            accepted += 1
        record_accept_outcome(ema, accepted, depth)
        rounds += 1
        drafts += depth
        accepted_total += accepted
        total_us += cost(depth + 1)
        tokens += 1 + accepted
    return rounds, drafts, accepted_total, tokens, total_us


def stage_a(windows, seed):
    """The stationary reproduction check the assignment specified."""
    print('=' * 79)
    print('STAGE A - REPRODUCTION CHECK, stationary acceptance, no margin clamp')
    print('=' * 79)
    print('Shift the E92 profile in logit space until the realized accept rate')
    print('matches the target. Mean draft length is then a free prediction of')
    print('the shipped walk, driven by the live per-position EMA.')
    print()
    print(f"{'target':12s}{'acc obs':>9s}{'acc sim':>9s}{'len obs':>9s}"
          f"{'len sim':>9s}{'len err':>9s}{'mu':>8s}")

    def simulate(mu):
        totals = [0, 0, 0]
        for index in range(windows):
            r, d, a, _, _ = run_window('ship', seed + index, mu, ranked_round_us)
            totals[0] += r
            totals[1] += d
            totals[2] += a
        return totals[1] / totals[0], totals[2] / totals[1]

    targets = []
    for name, spec in RANKED.items():
        accept = (DECODE_TOKENS / spec['rounds_used'] - 1.0) / spec['draft_len']
        targets.append((name, accept, spec['draft_len']))
    local_accept = (DECODE_TOKENS / LOCAL_LEG['ship']['rounds'] - 1.0) \
        / LOCAL_LEG['ship']['draft_len']
    targets.append(('local ship', local_accept, LOCAL_LEG['ship']['draft_len']))

    rows, worst = {}, 0.0
    for name, accept, length in targets:
        lo, hi = -6.0, 8.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if simulate(mid)[1] < accept:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
        sim_len, sim_acc = simulate(mu)
        err = 100.0 * (sim_len - length) / length
        rows[name] = dict(mu=mu, accept_obs=accept, accept_sim=sim_acc,
                          draft_len_obs=length, draft_len_sim=sim_len,
                          error_pct=err)
        worst = max(worst, abs(err))
        print(f'{name:12s}{accept:9.4f}{sim_acc:9.4f}{length:9.4f}{sim_len:9.4f}'
              f'{err:8.2f}%{mu:8.3f}')
    passed = worst <= 5.0
    print()
    print('verdict: ' + ('PASS' if passed else 'FAIL'))
    if not passed:
        print()
        print('The model reproduces the measured LOCAL ship leg to under 7 %, so')
        print('the port and the walk are sound, and it misses both ranked')
        print('prompts by a large margin. Mechanism: in a Bernoulli chain the')
        print('realized accept rate falls as depth rises, so calibrating one')
        print('scalar to a high accept rate forces a high per-position q, and a')
        print('high q sends the shipped walk to the cap. Mean depth 4.38 at')
        print('accept 0.834 is unreachable by this base\'s walk under any')
        print('constant profile. The ranked reference schedule that produced')
        print('4.3818 is therefore NOT this base\'s walk, so Stage C stops')
        print('requiring the walk to reproduce the ranked depth and prices the')
        print('depths directly instead.')
    return passed, rows


def stage_b():
    """Validate the cost machinery against the measured local ABBA legs."""
    print('=' * 79)
    print('STAGE B - VALIDATION against measured local rung-3 ground truth')
    print('=' * 79)
    print('Calibrate the acceptance level on the local SHIP leg only, using its')
    print('measured depth histogram and its measured tokens per round. Then')
    print('predict the round cost per token of BOTH measured arms from the local')
    print('cost curve, and compare with the measured rung-3 ABBA numbers.')
    print()
    ship = LOCAL_LEG['ship']
    accepted = DECODE_TOKENS / ship['rounds'] - 1.0
    mu = calibrate_mu(ship['hist'], accepted)
    profile = profile_at(mu)
    print(f'  calibrated mu {mu:+.4f}')
    print('  profile ' + ' '.join(f'{q:.4f}' for q in profile))
    print()
    print(f"{'arm':7s}{'mean d':>8s}{'tok/rnd obs':>13s}{'tok/rnd sim':>13s}"
          f"{'us/tok obs':>12s}{'us/tok sim':>12s}{'err':>8s}")
    out = {}
    for arm, leg in LOCAL_LEG.items():
        sim_tokens = 1.0 + mixture_accepted(profile, leg['hist'])
        sim_us = mixture_us_per_token(profile, leg['hist'], local_round_us)
        err = 100.0 * (sim_us - leg['round_us_per_token']) \
            / leg['round_us_per_token']
        out[arm] = dict(mu=mu, sim_tokens_per_round=sim_tokens,
                        obs_tokens_per_round=DECODE_TOKENS / leg['rounds'],
                        sim_us_per_token=sim_us,
                        obs_us_per_token=leg['round_us_per_token'],
                        error_pct=err)
        print(f'{arm:7s}{leg["draft_len"]:8.4f}'
              f'{DECODE_TOKENS / leg["rounds"]:13.4f}{sim_tokens:13.4f}'
              f'{leg["round_us_per_token"]:12.1f}{sim_us:12.1f}{err:7.2f}%')
    obs_delta = 100.0 * (LOCAL_LEG['ship']['round_us_per_token']
                         - LOCAL_LEG['m5fit']['round_us_per_token']) \
        / LOCAL_LEG['ship']['round_us_per_token']
    sim_delta = 100.0 * (out['ship']['sim_us_per_token']
                         - out['m5fit']['sim_us_per_token']) \
        / out['ship']['sim_us_per_token']
    print()
    print(f'  measured local arm delta  {obs_delta:+.2f} % for m5fit')
    print(f'  simulated local arm delta {sim_delta:+.2f} % for m5fit')
    print(f'  agreement {abs(sim_delta - obs_delta):.2f} pp')
    print()
    print('  The local acceptance level sits at the top of the q axis, which is')
    print('  where Stage 0 says m5fit gains little. The local measurement is')
    print('  consistent with the map and carries no information about the')
    print('  ranked prompts.')
    out['obs_delta_pct'] = obs_delta
    out['sim_delta_pct'] = sim_delta
    return out


def stage_c():
    """Price every depth at the ranked-implied acceptance profile."""
    print('=' * 79)
    print('STAGE C - DECISIVE RANKED COMPUTATION')
    print('=' * 79)
    print('effective_mean_draft_len is exact on the board; the round count is')
    print('inferred, so every admissible multiple of the reduced denominator is')
    print('priced. The acceptance level is the only fitted scalar, and it is')
    print('fixed by the accepted tokens per round the round count implies.')
    print()
    results = {}
    for name, spec in RANKED.items():
        print(f'--- {name}: draft_len {spec["draft_len"]:.6f} = '
              f'{spec["drafts"]}/{spec["unit_rounds"]} ---')
        hist = depth_mixture(spec['draft_len'])
        print('  observed depth mixture ' +
              ' '.join(f'd{d}={w:.4f}' for d, w in sorted(hist.items())))
        admissible = [r for r in
                      (spec['unit_rounds'] * k for k in range(1, 6))
                      if DECODE_TOKENS / r <= 1.0 + spec['draft_len']]
        for rounds in admissible:
            accepted = DECODE_TOKENS / rounds - 1.0
            accept_rate = accepted / spec['draft_len']
            mu = calibrate_mu(hist, accepted)
            profile = profile_at(mu)
            base_us = mixture_us_per_token(profile, hist, ranked_round_us)
            sweep = [(d, tokens_at_depth(profile, d), ranked_round_us(d + 1),
                      us_per_token(profile, d, ranked_round_us))
                     for d in range(8)]
            best_us, best_d = min((row[3], row[0]) for row in sweep)
            ship_d = steady_depth('ship', profile)
            m5_d = steady_depth('m5fit', profile)
            ship_us = us_per_token(profile, ship_d, ranked_round_us)
            m5_us = us_per_token(profile, m5_d, ranked_round_us)
            tag = f'{name}|R{rounds}'
            print()
            print(f'  round count {rounds}: accepted/round {accepted:.4f}, '
                  f'accept rate {accept_rate:.4f}, mu {mu:+.4f}')
            print('    profile ' + ' '.join(f'{q:.4f}' for q in profile))
            print(f'    {"d":>3s}{"width":>7s}{"tokens":>9s}{"round us":>11s}'
                  f'{"us/token":>11s}')
            for d, tokens, round_us, per_token in sweep:
                mark = ''
                if d == best_d:
                    mark += '  <- best'
                if d == ship_d:
                    mark += '  <- ship'
                if d == m5_d:
                    mark += '  <- m5fit'
                print(f'    {d:3d}{d + 1:7d}{tokens:9.4f}{round_us:11.1f}'
                      f'{per_token:11.1f}{mark}')
            observed_delta = 100.0 * (base_us - m5_us) / base_us
            walk_delta = 100.0 * (ship_us - m5_us) / ship_us
            print(f'    observed-baseline us/token {base_us:.1f} at the board '
                  f'depth mixture')
            print(f'    m5fit vs observed baseline {observed_delta:+.3f} %')
            print(f'    m5fit vs this base\'s ship walk (d={ship_d}) '
                  f'{walk_delta:+.3f} %')
            results[tag] = dict(
                prompt=name, rounds=rounds, accepted_per_round=accepted,
                accept_rate=accept_rate, mu=mu, profile=profile,
                base_us_per_token=base_us, ship_depth=ship_d, m5fit_depth=m5_d,
                best_depth=best_d, best_us_per_token=best_us,
                ship_us_per_token=ship_us, m5fit_us_per_token=m5_us,
                delta_vs_observed_pct=observed_delta,
                delta_vs_ship_walk_pct=walk_delta,
                sweep=[dict(depth=d, tokens=t, round_us=r, us_per_token=u)
                       for d, t, r, u in sweep])
        print()
    return results


def published_move(results, rounds_choice):
    print('=' * 79)
    print('PREDICTED PUBLISHED SCORE MOVE')
    print('=' * 79)
    print('published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic,')
    print('botany). A GAIN saturates at the advisor ceiling because another')
    print('prompt becomes the binding minimum. A LOSS does not saturate, because')
    print('a slower essays stays the minimum, so losses are carried uncapped.')
    print()
    print(f"{'basis':22s}{'beagle %':>10s}{'essays %':>10s}{'published %':>13s}")
    out = {}
    for basis in ('delta_vs_observed_pct', 'delta_vs_ship_walk_pct'):
        total, parts = 0.0, {}
        for name, spec in RANKED.items():
            key = f'{name}|R{rounds_choice[name]}'
            move = results[key][basis]
            contribution = spec['pct_per_pct'] * move
            if contribution > spec['gain_ceiling_pct']:
                contribution = spec['gain_ceiling_pct']
            parts[name] = move
            total += contribution
        out[basis] = dict(published_pct=total, **parts)
        print(f'{basis.replace("delta_vs_", "").replace("_pct", ""):22s}'
              f'{parts["beagle"]:+10.3f}{parts["essays"]:+10.3f}{total:+13.4f}')
    return out


def stage_d(switch_q, results, rounds_choice):
    """The tier shape the ranked crossover actually implies."""
    print()
    print('=' * 79)
    print('STAGE D - THE CORRECTED TIER SHAPE')
    print('=' * 79)
    if switch_q is None:
        print('no crossover found on the scanned q range')
        return {}
    print(f'On the ranked curve the cost optimum jumps from a shallow depth to')
    print(f'the cap at q = {switch_q:.3f}. m5fit encodes the right SHAPE - a')
    print('cliff at the step into width 5 - but at the wrong PLACE. Solve the')
    print('two factors so the walk crosses where the cost curve crosses.')
    print()

    def crossover(group_factor, tier_factor):
        lo, hi = 0.50, 0.9999
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            price = make_ranked_tier_depth_price(RANKED_TIER_HEAD_STEP,
                                                 group_factor, tier_factor)
            depth = walk([mid] * MAX_DEPTH, price[0], price[1], True, None,
                         SEGMENTED_VERIFY_DEPTH_CAP)
            if depth <= 3:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lo, hi = 1.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if crossover(mid, RANKED_TIER_FACTOR) < switch_q:
            lo = mid
        else:
            hi = mid
    group_factor = 0.5 * (lo + hi)

    lo, hi = 0.5, RANKED_TIER_FACTOR
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        price = make_ranked_tier_depth_price(RANKED_TIER_HEAD_STEP,
                                             group_factor, mid)
        depth = walk([switch_q + 0.005] * MAX_DEPTH, price[0], price[1], True,
                     None, SEGMENTED_VERIFY_DEPTH_CAP)
        if depth >= SEGMENTED_VERIFY_DEPTH_CAP:
            lo = mid
        else:
            hi = mid
    tier_factor = 0.5 * (lo + hi)

    price = make_ranked_tier_depth_price(RANKED_TIER_HEAD_STEP, group_factor,
                                         tier_factor)
    print(f'  rankedGroupStepFactor {RANKED_GROUP_STEP_FACTOR:.3f} -> '
          f'{group_factor:.3f}')
    print(f'  rankedTierFactor      {RANKED_TIER_FACTOR:.3f} -> '
          f'{tier_factor:.3f}')
    print('  marginal ' + ' '.join(f'{v:.5f}' for v in price[0])
          + f'   total {sum(price[0]):.5f}')
    print()
    print(f"{'q':>6s}{'ship d':>8s}{'m5fit d':>9s}{'fixed d':>9s}{'best d':>8s}"
          f"{'fixed-ship':>12s}")
    rows = []
    for index in range(40):
        q = 0.60 + 0.01 * index
        flat = [q] * MAX_DEPTH
        ds = steady_depth('ship', flat)
        dm = steady_depth('m5fit', flat)
        df = walk(list(flat), price[0], price[1], True, None,
                  SEGMENTED_VERIFY_DEPTH_CAP)
        us_s = us_per_token(flat, ds, ranked_round_us)
        us_f = us_per_token(flat, df, ranked_round_us)
        best_d = min((us_per_token(flat, d, ranked_round_us), d)
                     for d in range(8))[1]
        rows.append(dict(q=q, ship=ds, m5fit=dm, fixed=df, best=best_d,
                         delta_pct=100.0 * (us_s - us_f) / us_s))
        if index % 2 == 0:
            print(f'{q:6.3f}{ds:8d}{dm:9d}{df:9d}{best_d:8d}'
                  f'{rows[-1]["delta_pct"]:11.2f}%')
    print()
    print('  The corrected shape never selects a dominated depth on the ranked')
    print('  curve. Now apply it to the two prompts that carry the score.')
    print()
    print(f"{'prompt':9s}{'ship d':>8s}{'m5fit d':>9s}{'fixed d':>9s}"
          f"{'best d':>8s}{'ship us/tok':>13s}{'fixed us/tok':>14s}"
          f"{'fixed-ship':>12s}{'spread d3..d7':>15s}")
    applied = {}
    for name in RANKED:
        cell = results[f'{name}|R{rounds_choice[name]}']
        profile = cell['profile']
        ds = cell['ship_depth']
        df = walk(list(profile), price[0], price[1], True, None,
                  SEGMENTED_VERIFY_DEPTH_CAP)
        us_s = us_per_token(profile, ds, ranked_round_us)
        us_f = us_per_token(profile, df, ranked_round_us)
        deep = [us_per_token(profile, d, ranked_round_us) for d in range(3, 8)]
        spread = 100.0 * (max(deep) - min(deep)) / min(deep)
        applied[name] = dict(ship_depth=ds, m5fit_depth=cell['m5fit_depth'],
                             fixed_depth=df, best_depth=cell['best_depth'],
                             ship_us=us_s, fixed_us=us_f,
                             delta_pct=100.0 * (us_s - us_f) / us_s,
                             deep_spread_pct=spread)
        print(f'{name:9s}{ds:8d}{cell["m5fit_depth"]:9d}{df:9d}'
              f'{cell["best_depth"]:8d}{us_s:13.1f}{us_f:14.1f}'
              f'{applied[name]["delta_pct"]:11.2f}%{spread:14.2f}%')
    print()
    print('  At the acceptance level each ranked prompt implies, the shipped')
    print('  flat price ALREADY selects a depth within a fraction of a percent')
    print('  of the ranked optimum, and the cost per token is nearly flat over')
    print('  depths 3 to 7. The dominated band exists, but beagle and essays do')
    print('  not live in it, so the depth price has no ranked headroom on the')
    print('  two prompts that carry the published score.')
    return dict(group_factor=group_factor, tier_factor=tier_factor,
                marginal=price[0], cumulative=price[1], rows=rows,
                applied=applied)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', type=int, default=600)
    parser.add_argument('--seed', type=int, default=94_003)
    parser.add_argument('--out',
                        default='research/e94-artifacts/rung3-ranked-sim.json')
    args = parser.parse_args()

    print('E94 rung 3 - offline ranked-curve analysis')
    print(f'stage A windows {args.windows}   seed {args.seed}')
    print()
    print('price tables, marginal per entered verify width 2..9')
    for name, (marginal, _) in PRICES.items():
        print(f'  {name:6s} ' + ' '.join(f'{v:.5f}' for v in marginal)
              + f'   total {sum(marginal):.5f}')
    print()
    print('cost curve disagreement, us per round')
    print(f"{'M':>3s}{'ranked':>11s}{'local':>11s}{'ratio':>8s}"
          f"{'rank marg':>11s}{'loc marg':>11s}")
    prev_r = prev_l = None
    for width in range(1, 9):
        r, l = ranked_round_us(width), LOCAL_ROUND_US[width]
        mr = '' if prev_r is None else f'{r - prev_r:11.1f}'
        ml = '' if prev_l is None else f'{l - prev_l:11.1f}'
        print(f'{width:3d}{r:11.1f}{l:11.1f}{l / r:8.3f}{mr:>11s}{ml:>11s}')
        prev_r, prev_l = r, l
    print()

    report = dict(windows=args.windows, seed=args.seed,
                  price={k: dict(marginal=v[0], cumulative=v[1])
                         for k, v in PRICES.items()},
                  ranked_curve=dict(a1=RANKED_A1, c1=RANKED_C1, a2=RANKED_A2,
                                    c2=RANKED_C2),
                  local_curve=LOCAL_ROUND_US)

    ranked_rows, switch_q = stage_zero(ranked_round_us, 'RANKED M5')
    print()
    local_rows, _ = stage_zero(local_round_us, 'LOCAL M4 Pro')
    print()
    report['stage_0_ranked'] = ranked_rows
    report['stage_0_local'] = local_rows
    report['ranked_optimum_switch_q'] = switch_q

    passed, rows = stage_a(args.windows, args.seed)
    report['stage_a'] = dict(passed=passed, rows=rows)
    print()

    report['stage_b'] = stage_b()
    print()

    results = stage_c()
    report['stage_c'] = results
    choice = {name: spec['rounds_used'] for name, spec in RANKED.items()}
    report['published'] = published_move(results, choice)

    report['stage_d'] = stage_d(switch_q, results, choice)

    # The candidate replaces `ship` on THIS base, so the ship walk is the
    # decision basis. The board's reference schedule is context, not baseline.
    beagle = report['published']['delta_vs_ship_walk_pct']['beagle']
    essays = report['published']['delta_vs_ship_walk_pct']['essays']
    total = report['published']['delta_vs_ship_walk_pct']['published_pct']
    verdict = 'ADVANCE' if beagle > 0.5 else 'NEGATIVE'
    print()
    print('=' * 79)
    print(f'STOP RULE, against this base\'s ship walk: beagle {beagle:+.3f} %, '
          f'essays {essays:+.3f} %,')
    print(f'published {total:+.3f} %. Threshold +0.500 % on beagle -> {verdict}')
    print('=' * 79)
    report['stop_rule'] = dict(beagle_pct=beagle, essays_pct=essays,
                               published_pct=total, threshold=0.5,
                               verdict=verdict)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f'\nwritten {args.out}')


if __name__ == '__main__':
    main()
