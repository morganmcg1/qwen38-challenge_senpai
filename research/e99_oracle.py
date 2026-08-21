#!/usr/bin/env python3
"""E99: how much of the round-level cost is allocation, not depth?

The question
------------
The shipped walk picks one depth per round. Three counterfactuals bound what a
better allocator could buy, all priced on the RANKED M5 cost curve:

  actual         what the shipped walk achieved on the recorded rounds;
  oracle         a policy that knows each round's realised acceptance prefix
                 before it chooses the depth;
  best fixed     the best single constant depth over the same rounds.

`actual - oracle` is the whole allocation prize and no implementable policy can
exceed it. `best fixed - actual` is what the adaptive walk already earns over a
constant policy. Rung 4 then fits the best policy that reads ONLY pre-round
state, which is the honest estimate of the reachable part.

Where the rounds come from
--------------------------
`MLX_QWEN_MTP_TRACE=1` already records, per round, the chosen depth, the
accepted prefix, the full pre-round `positionAcceptEMA` vector, the pending
primary's top-2 margin, the full-accept streak, the width cap and the round
index. No scored-surface change is needed to answer this question, so this
script replays recorded legs and spends no GPU time.

Censoring
---------
A round with `acc == d` is CENSORED: a deeper draft might have accepted more,
and the oracle bound computed from it is biased downward. Every counterfactual
is therefore reported three ways:

  observed   a* = a_r exactly. Conservative lower bound on the oracle.
  exclude    censored rounds dropped from both numerator and denominator.
  impute     the unseen tail is drawn from the E92 per-position acceptance
             profile, so a censored round carries a distribution over a*.

The oracle
----------
Minimising `sum_r C(d_r + 1) / sum_r tokens_r` is a ratio, not a sum, so the
per-round choice is not independent. Dinkelbach's method solves it exactly:
at multiplier `lam` each round (each imputation branch of each round) takes the
depth that minimises `C(d + 1) - lam * tokens(d)`, then `lam` is updated to the
achieved ratio and the sweep repeats to a fixed point.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

MAX_DEPTH = 8
SEGMENTED_VERIFY_DEPTH_CAP = 7

# Ranked M5 round cost, us. research/ranked_cost_curve.py, median of the
# official runs on the reference schedule. Two lines with a step at the
# G = ceil(M / 4) group boundary.
RANKED_A1, RANKED_C1 = 27181.5, 3995.1   # M = 1..4, G = 1
RANKED_A2, RANKED_C2 = 16943.2, 7233.0   # M = 5..8, G = 2

# Local M4 Pro round cost, us. E92 isolated whole-table measurement.
LOCAL_ROUND_US = {1: 64445.4, 2: 69775.5, 3: 74778.4, 4: 86237.4,
                  5: 126103.1, 6: 137842.6, 7: 150431.4, 8: 163957.1}

# E92 per-position acceptance, used only by the `impute` treatment.
E92_PROFILE = [0.9659, 0.9652, 0.9543, 0.9486, 0.9487, 0.9859, 0.9451, 0.8333]

# Finding 16 sensitivities, from the assignment brief.
BEAGLE_PCT_PER_PCT, BEAGLE_CEILING_PCT = 0.4801, 4.51
ESSAYS_PCT_PER_PCT, ESSAYS_SATURATION_PCT = 0.5199, 0.53

# Ranked mean verify width per scoring prompt, from the assignment brief.
RANKED_WIDTH = {'beagle': 5.382, 'republic': 5.989, 'essays': 6.087,
                'medicine': 6.256, 'botany': 7.148, 'travel': 3.656,
                'drama': 3.298, 'plutarch': 1.154}

TRACE_RE = re.compile(r'^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) ')
FIELD_RE = re.compile(r'(\w+)=([-+0-9.eE]+|nan)')


def ranked_round_us(width: int) -> float:
    if not 1 <= width <= 8:
        raise ValueError(f'ranked curve is defined for M = 1..8, got {width}')
    if width <= 4:
        return RANKED_A1 + RANKED_C1 * width
    return RANKED_A2 + RANKED_C2 * width


def local_round_us(width: int) -> float:
    return LOCAL_ROUND_US[width]


CURVES = {'ranked': ranked_round_us, 'local': local_round_us}


class Round:
    """One recorded round, with only pre-round state as features."""

    __slots__ = ('leg', 'index', 'depth', 'accepted', 'cap', 'round_us',
                 'ema', 'margin', 'streak', 'features')

    def __init__(self, leg, index, depth, accepted, cap, round_us, ema,
                 margin, streak):
        self.leg = leg
        self.index = index
        self.depth = depth
        self.accepted = accepted
        self.cap = cap
        self.round_us = round_us
        self.ema = ema
        self.margin = margin
        self.streak = streak
        self.features = {}

    @property
    def censored(self) -> bool:
        return self.accepted == self.depth


def parse_trace(path: Path, leg: str, offered_cap: int) -> list[Round]:
    rounds = []
    for line in path.read_text().splitlines():
        head = TRACE_RE.match(line)
        if head is None:
            continue
        index, depth, accepted = (int(head.group(i)) for i in (1, 2, 3))
        fields = dict(FIELD_RE.findall(line.split(' arm=')[-1]))
        ema_text = line.split(' ema=')[1].split(' ')[0]
        ema = [float(v) for v in ema_text.split(',')]
        margin = float(fields.get('m', 'nan'))
        streak = int(float(fields.get('streak', 0)))
        round_us = float(re.search(r' round_us=(\d+) ', line).group(1))
        cap = min(offered_cap, SEGMENTED_VERIFY_DEPTH_CAP, MAX_DEPTH)
        rounds.append(Round(leg, index, depth, accepted, cap, round_us,
                            ema, margin, streak))
    return rounds


def load_leg(tag: str, out_root: Path) -> tuple[dict, list[Round]]:
    directory = out_root / tag
    meta = dict(
        line.split('=', 1)
        for line in (directory / 'meta.txt').read_text().splitlines()
        if '=' in line)
    score = json.loads((directory / 'score.json').read_text())['metrics']
    offered = int(score['mtp_depth'])
    rounds = parse_trace(directory / 'trace.txt', tag, offered)
    return dict(meta=meta, score=score, offered_cap=offered), rounds


def attach_features(rounds: list[Round]) -> None:
    """Only state the schedule could legally read BEFORE it proposes."""
    previous = None
    for record in rounds:
        feature = {f'ema{i}': value for i, value in enumerate(record.ema)}
        reach = 1.0
        for i, value in enumerate(record.ema):
            reach *= value
            feature[f'reach{i + 1}'] = reach
        feature['margin'] = record.margin
        feature['streak'] = float(record.streak)
        feature['round_idx'] = float(record.index)
        feature['cap'] = float(record.cap)
        feature['d_ship'] = float(record.depth)
        if previous is None:
            feature['prev_acc'] = -1.0
            feature['prev_depth'] = -1.0
            feature['prev_full'] = -1.0
        else:
            feature['prev_acc'] = float(previous.accepted)
            feature['prev_depth'] = float(previous.depth)
            feature['prev_full'] = 1.0 if previous.censored else 0.0
        record.features = feature
        previous = record


def branches(record: Round, treatment: str) -> list[tuple[float, int]]:
    """Distribution over the round's true acceptance prefix a*."""
    if not record.censored or treatment == 'observed':
        return [(1.0, record.accepted)]
    if treatment == 'exclude':
        return []
    if treatment != 'impute':
        raise ValueError(treatment)
    out, survive = [], 1.0
    limit = record.cap - record.depth
    for step in range(limit):
        position = record.depth + step
        q = E92_PROFILE[position] if position < len(E92_PROFILE) else \
            E92_PROFILE[-1]
        out.append((survive * (1.0 - q), record.depth + step))
        survive *= q
    out.append((survive, record.cap))
    return [(p, a) for p, a in out if p > 0.0]


def token_row(record: Round, treatment: str) -> list[float]:
    """Expected committed tokens for every legal depth on this round."""
    parts = branches(record, treatment)
    return [sum(p * (min(depth, a) + 1) for p, a in parts)
            for depth in range(record.cap + 1)]


def usable(rounds: list[Round], treatment: str) -> list[Round]:
    if treatment == 'exclude':
        return [r for r in rounds if not r.censored]
    return list(rounds)


def actual(rounds: list[Round], curve) -> dict:
    kept = rounds
    cost = sum(curve(r.depth + 1) for r in kept)
    tokens = sum(r.accepted + 1 for r in kept)
    return dict(us_per_token=cost / tokens, rounds=len(kept), tokens=tokens,
                round_us=cost / len(kept),
                mean_depth=statistics.fmean([r.depth for r in kept]),
                mean_tokens=tokens / len(kept))


def fixed_depth(rounds: list[Round], curve, treatment: str) -> dict:
    kept = usable(rounds, treatment)
    table = []
    for depth in range(0, min(r.cap for r in kept) + 1):
        cost = sum(curve(depth + 1) for _ in kept)
        tokens = sum(token_row(r, treatment)[depth] for r in kept)
        table.append(dict(depth=depth, us_per_token=cost / tokens,
                          tokens_per_round=tokens / len(kept)))
    best = min(table, key=lambda row: row['us_per_token'])
    return dict(best=best, sweep=table, rounds=len(kept))


def dinkelbach(rounds: list[Round], curve, treatment: str, candidates,
               seed: float) -> dict:
    """Exact ratio minimisation over a per-round-branch choice set."""
    kept = usable(rounds, treatment)
    lam = seed
    chosen = {}
    for _ in range(64):
        cost = tokens = 0.0
        depth_mass = {}
        for record in kept:
            options = candidates(record)
            for probability, accept in branches(record, treatment):
                best = None
                for depth in options:
                    got = min(depth, accept) + 1
                    value = curve(depth + 1) - lam * got
                    if best is None or value < best[0]:
                        best = (value, depth, got)
                _, depth, got = best
                cost += probability * curve(depth + 1)
                tokens += probability * got
                depth_mass[depth] = depth_mass.get(depth, 0.0) + probability
        new_lam = cost / tokens
        chosen = depth_mass
        if abs(new_lam - lam) < 1e-9:
            lam = new_lam
            break
        lam = new_lam
    total = sum(chosen.values())
    return dict(us_per_token=lam, rounds=len(kept),
                depth_share={d: v / total for d, v in sorted(chosen.items())},
                mean_depth=sum(d * v for d, v in chosen.items()) / total)


def oracle(rounds, curve, treatment, seed):
    return dinkelbach(rounds, curve, treatment,
                      lambda r: range(r.cap + 1), seed)


def one_bit_g(rounds, curve, treatment, seed, mode):
    """The binary G decision alone: pay one weight stream, or the shipped one.

    `truncate` is the implementable one-bit form: keep the shipped depth, or
    clamp this round into the G = 1 band. `band` additionally lets the G = 1
    arm pick its own best depth inside the band, which needs the same oracle
    foresight and is reported only to show the size of that extra freedom.
    """
    if mode == 'truncate':
        def candidates(record):
            return sorted({min(3, record.depth), record.depth})
    elif mode == 'band':
        def candidates(record):
            return sorted(set(range(0, min(3, record.cap) + 1))
                          | {record.depth})
    else:
        raise ValueError(mode)
    return dinkelbach(rounds, curve, treatment, candidates, seed)


# ---------------------------------------------------------------- rung 4 fit

FEATURES = ('ema0', 'ema1', 'ema2', 'ema3', 'ema4', 'ema5', 'ema6', 'ema7',
            'reach1', 'reach3', 'reach4', 'reach7', 'margin', 'streak',
            'round_idx', 'prev_acc', 'prev_depth', 'prev_full', 'd_ship')


def feature_value(record: Round, name: str) -> float:
    value = record.features[name]
    if isinstance(value, float) and math.isnan(value):
        return 1e9
    return value


def thresholds(rounds: list[Round], name: str, limit: int = 24) -> list[float]:
    values = sorted({feature_value(r, name) for r in rounds})
    if len(values) < 2:
        return []
    cuts = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    if len(cuts) <= limit:
        return cuts
    step = len(cuts) / limit
    return [cuts[int(i * step)] for i in range(limit)]


# Both clamp classes hold the shipped depth or move the round into the G = 1
# band at depth 3. Only these classes have a matching random control.
CLAMP_CLASSES = ('one_bit_g', 'margin_gate')


def action_set(name: str):
    """Every action is a function of PRE-ROUND state only, so every policy
    built from them is implementable inside `costModelDepth`."""
    if name == 'absolute':
        actions = [(f'd{d}', (lambda d: lambda r: min(d, r.cap))(d))
                   for d in range(SEGMENTED_VERIFY_DEPTH_CAP + 1)]
        actions.append(('ship', lambda r: r.depth))
        return actions
    if name in CLAMP_CLASSES:
        return [('ship', lambda r: r.depth),
                ('g1', lambda r: min(3, r.depth))]
    raise ValueError(name)


# `margin_gate` is the smallest implementable form of the fitted policy: one
# threshold on the pending primary's top-2 margin, one clamp into the G = 1
# band. It is fitted with a single split so the reported threshold is a
# constant a follow-up experiment can put straight into `costModelDepth`.
CLASS_FEATURES = {'margin_gate': ('margin',)}
CLASS_TREE_DEPTH = {'margin_gate': 1}


def reward_table(rounds, curve, treatment, lam, actions):
    """reward[r][a] = lam * expected tokens - round cost, to be MAXIMISED."""
    table = []
    for record in rounds:
        tokens = token_row(record, treatment)
        table.append([lam * tokens[act(record)] - curve(act(record) + 1)
                      for _, act in actions])
    return table


def best_leaf(indices, table):
    """Best single action for a set of rounds, and its total reward."""
    best = None
    for slot in range(len(table[indices[0]])):
        total = sum(table[i][slot] for i in indices)
        if best is None or total > best[0]:
            best = (total, slot)
    return best


def fit_tree(rounds, table, indices, depth_left, actions, min_leaf=12,
             features=FEATURES):
    """Greedy cost-sensitive tree. Leaves hold one action each."""
    total, slot = best_leaf(indices, table)
    node = dict(kind='leaf', action=actions[slot][0], slot=slot,
                reward=total, count=len(indices))
    if depth_left == 0 or len(indices) < 2 * min_leaf:
        return node
    subset = [rounds[i] for i in indices]
    best = None
    for name in features:
        for cut in thresholds(subset, name):
            left, right = [], []
            for i in indices:
                (left if feature_value(rounds[i], name) <= cut
                 else right).append(i)
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            gain = best_leaf(left, table)[0] + best_leaf(right, table)[0]
            if best is None or gain > best[0]:
                best = (gain, name, cut, left, right)
    if best is None or best[0] <= total + 1e-9:
        return node
    _, name, cut, left, right = best
    return dict(
        kind='split', feature=name, threshold=cut, count=len(indices),
        left=fit_tree(rounds, table, left, depth_left - 1, actions, min_leaf,
                      features),
        right=fit_tree(rounds, table, right, depth_left - 1, actions,
                       min_leaf, features))


def apply_tree(node, record: Round, actions) -> int:
    while node['kind'] == 'split':
        node = node['left'] \
            if feature_value(record, node['feature']) <= node['threshold'] \
            else node['right']
    return min(actions[node['slot']][1](record), record.cap)


def evaluate_policy(rounds, curve, treatment, depth_of) -> dict:
    kept = usable(rounds, treatment)
    cost = tokens = 0.0
    depths = []
    for record in kept:
        depth = depth_of(record)
        row = token_row(record, treatment)
        cost += curve(depth + 1)
        tokens += row[depth]
        depths.append(depth)
    return dict(us_per_token=cost / tokens, rounds=len(kept),
                mean_depth=statistics.fmean(depths))


def fit_policy(train, test, curve, treatment, max_depth, seed,
               class_name='absolute') -> dict:
    """Fit on `train`, report held-out `test`. Ratio objective, Dinkelbach."""
    actions = action_set(class_name)
    features = CLASS_FEATURES.get(class_name, FEATURES)
    max_depth = CLASS_TREE_DEPTH.get(class_name, max_depth)
    kept = usable(train, treatment)
    lam = seed
    tree = None
    for _ in range(32):
        table = reward_table(kept, curve, treatment, lam, actions)
        tree = fit_tree(kept, table, list(range(len(kept))), max_depth,
                        actions, features=features)
        train_score = evaluate_policy(
            kept, curve, treatment, lambda r: apply_tree(tree, r, actions))
        if abs(train_score['us_per_token'] - lam) < 1e-9:
            lam = train_score['us_per_token']
            break
        lam = train_score['us_per_token']
    held = evaluate_policy(test, curve, treatment,
                           lambda r: apply_tree(tree, r, actions))
    return dict(tree=tree, policy_class=class_name, train_us_per_token=lam,
                held_out=held)


def describe_tree(node, prefix='') -> list[str]:
    if node['kind'] == 'leaf':
        return [f'{prefix}{node["action"]}  (n={node["count"]})']
    out = [f'{prefix}if {node["feature"]} <= {node["threshold"]:.6g}:']
    out += describe_tree(node['left'], prefix + '    ')
    out += [f'{prefix}else:']
    out += describe_tree(node['right'], prefix + '    ')
    return out


def random_clamp_control(rounds, curve, treatment, count, draws=200,
                         seed=20260821) -> dict:
    """Clamp the same NUMBER of rounds, chosen at random.

    A gate that clamps 24 % of rounds also lowers the mean depth, and a lower
    mean depth alone can move us/token. This control holds the clamp RATE and
    destroys only the information in the gate, so whatever the fitted policy
    earns above this control is allocation and not level.
    """
    generator = random.Random(seed)
    kept = usable(rounds, treatment)
    reference = actual(kept, curve)['us_per_token']
    gains = []
    for _ in range(draws):
        clamped = set(generator.sample(range(len(kept)), count))
        cost = tokens = 0.0
        for index, record in enumerate(kept):
            depth = min(3, record.depth) if index in clamped else record.depth
            cost += curve(depth + 1)
            tokens += token_row(record, treatment)[depth]
        gains.append(100.0 * (reference - cost / tokens) / reference)
    gains.sort()
    return dict(mean_gain_pct=statistics.fmean(gains),
                p95_gain_pct=gains[int(0.95 * (len(gains) - 1))],
                max_gain_pct=gains[-1], draws=draws, clamped=count,
                rounds=len(kept))


def clamp_count(rounds, treatment, depth_of) -> int:
    return sum(1 for r in usable(rounds, treatment) if depth_of(r) < r.depth)


def pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def feature_ranking(rounds) -> list[dict]:
    """Does pre-round state predict what the round is about to do?

    Two targets. `a_r` is the realised accepted prefix, which is what an
    allocator needs. `shallow` marks a round that accepts fewer than three
    drafts, which is the binary G decision's target.
    """
    accepted = [float(r.accepted) for r in rounds]
    shallow = [1.0 if r.accepted < 3 else 0.0 for r in rounds]
    out = []
    for name in FEATURES:
        values = [feature_value(r, name) for r in rounds]
        out.append(dict(feature=name,
                        r_accepted=pearson(values, accepted),
                        r_shallow=pearson(values, shallow)))
    out.sort(key=lambda row: -abs(row['r_accepted'] if not
                                  math.isnan(row['r_accepted']) else 0.0))
    return out


# ------------------------------------------------------------------ reporting

def published_move(gain_pct: float) -> dict:
    """Finding 16: published = 0.5 raw_beagle + 0.5 min(other four)."""
    beagle = min(BEAGLE_PCT_PER_PCT * gain_pct, BEAGLE_CEILING_PCT)
    essays = min(ESSAYS_PCT_PER_PCT * gain_pct,
                 ESSAYS_PCT_PER_PCT * ESSAYS_SATURATION_PCT)
    return dict(applied_gain_pct=gain_pct, beagle_pct=beagle,
                essays_pct=essays, published_pct=beagle + essays)


def nearest_prompt(width: float) -> str:
    return min(RANKED_WIDTH, key=lambda k: abs(RANKED_WIDTH[k] - width))


def boundary_curve(step: float):
    """The ranked curve with the G boundary resized to `step`.

    The ranked round counts behind the fitted curve are inferred, so the size
    of the M = 4 -> M = 5 jump carries real uncertainty. Everything the one-bit
    G policy earns comes from that jump, so the conclusion has to be reported
    against a range of jump sizes. The G = 1 line and the G = 2 slope are held;
    only the G = 2 intercept moves, so `C(5) = C(4) * (1 + step)`.
    """
    c4 = RANKED_A1 + RANKED_C1 * 4
    intercept = c4 * (1.0 + step) - RANKED_C2 * 5

    def curve(width: int) -> float:
        if width <= 4:
            return RANKED_A1 + RANKED_C1 * width
        return intercept + RANKED_C2 * width

    return curve


def analyse_leg(tag, info, rounds, curve_name) -> dict:
    curve = CURVES[curve_name]
    base = actual(rounds, curve)
    out = dict(leg=tag, curve=curve_name, offered_cap=info['offered_cap'],
               actual=base,
               measured_local_round_us=statistics.fmean(
                   [r.round_us for r in rounds]),
               censored_share=statistics.fmean(
                   [1.0 if r.censored else 0.0 for r in rounds]),
               score={k: info['score'][k] for k in (
                   'effective_mean_draft_len', 'accepted_draft_rate',
                   'mtp_seconds_per_token', 'all_tokens_matched')},
               treatments={})
    for treatment in ('observed', 'exclude', 'impute'):
        kept = usable(rounds, treatment)
        if not kept:
            continue
        reference = actual(kept, curve)
        oracle_result = oracle(rounds, curve, treatment,
                               reference['us_per_token'])
        fixed = fixed_depth(rounds, curve, treatment)
        bit_truncate = one_bit_g(rounds, curve, treatment,
                                 reference['us_per_token'], 'truncate')
        bit_band = one_bit_g(rounds, curve, treatment,
                             reference['us_per_token'], 'band')
        gap = 100.0 * (reference['us_per_token'] - oracle_result['us_per_token']) \
            / reference['us_per_token']
        out['treatments'][treatment] = dict(
            actual=reference, oracle=oracle_result, best_fixed=fixed['best'],
            fixed_sweep=fixed['sweep'],
            one_bit_g_truncate=bit_truncate, one_bit_g_band=bit_band,
            oracle_gap_pct=gap,
            fixed_gap_pct=100.0 * (fixed['best']['us_per_token']
                                   - reference['us_per_token'])
            / reference['us_per_token'],
            one_bit_g_gap_pct=100.0 * (reference['us_per_token']
                                       - bit_truncate['us_per_token'])
            / reference['us_per_token'],
            one_bit_share_of_oracle=(
                (reference['us_per_token'] - bit_truncate['us_per_token'])
                / (reference['us_per_token'] - oracle_result['us_per_token'])
                if reference['us_per_token'] > oracle_result['us_per_token']
                else float('nan')))
    return out


def splits(rounds, legs) -> list[tuple[str, list, list]]:
    """Three held-out designs, from the most optimistic to the strictest.

    `parity` alternates rounds. Neighbouring rounds share EMA state and local
    text, so parity leaks and its held-out number is an upper estimate.
    `time` trains on the first half of every leg and tests on the second half,
    which removes that leak but changes the operating point, because the EMAs
    are still warming in the first half. `leg-out` trains on every other leg
    and tests on one whole leg at a different offered cap, which is the only
    design here that tests transfer.
    """
    out = [('parity even->odd',
            [r for i, r in enumerate(rounds) if i % 2 == 0],
            [r for i, r in enumerate(rounds) if i % 2 == 1]),
           ('parity odd->even',
            [r for i, r in enumerate(rounds) if i % 2 == 1],
            [r for i, r in enumerate(rounds) if i % 2 == 0])]
    first, second = [], []
    for tag in legs:
        leg_rounds = [r for r in rounds if r.leg == tag]
        half = len(leg_rounds) // 2
        first.extend(leg_rounds[:half])
        second.extend(leg_rounds[half:])
    out.append(('time first->second', first, second))
    for tag in legs:
        train = [r for r in rounds if r.leg != tag]
        test = [r for r in rounds if r.leg == tag]
        if train and test:
            out.append((f'leg-out {tag}', train, test))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legs', nargs='+', required=True)
    parser.add_argument('--headline', required=True,
                        help='leg tag whose mean verify width sits in the '
                             'ranked 5.4-6.1 band')
    parser.add_argument('--out-root', default='research/out')
    parser.add_argument('--out', default='')
    parser.add_argument('--tree-depth', type=int, default=2)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    legs, all_rounds = {}, []
    for tag in args.legs:
        info, rounds = load_leg(tag, out_root)
        attach_features(rounds)
        legs[tag] = dict(info=info, rounds=rounds)
        all_rounds.extend(rounds)

    report = dict(legs={}, headline=args.headline)

    print('=' * 79)
    print('RECORDED LEGS')
    print('=' * 79)
    print(f'{"leg":12s}{"cap":>5s}{"rounds":>8s}{"width":>8s}'
          f'{"accept":>9s}{"censored":>10s}{"round_us":>11s}{"model_us":>11s}')
    for tag, item in legs.items():
        rounds = item['rounds']
        info = item['info']
        width = info['score']['effective_mean_draft_len'] + 1.0
        measured = statistics.fmean([r.round_us for r in rounds])
        model = statistics.fmean([local_round_us(r.depth + 1) for r in rounds])
        print(f'{tag:12s}{info["offered_cap"]:5d}{len(rounds):8d}{width:8.3f}'
              f'{info["score"]["accepted_draft_rate"]:9.4f}'
              f'{statistics.fmean([1.0 if r.censored else 0.0 for r in rounds]):10.3f}'
              f'{measured:11.1f}{model:11.1f}')

    for curve_name in ('ranked', 'local'):
        print()
        print('=' * 79)
        print(f'COUNTERFACTUALS ON THE {curve_name.upper()} CURVE')
        print('=' * 79)
        print(f'{"leg":11s}{"treat":9s}{"actual":>10s}{"oracle":>10s}'
              f'{"gap %":>8s}{"fixed":>10s}{"fixed %":>9s}'
              f'{"1bitG":>10s}{"1bitG %":>9s}{"share":>7s}')
        for tag, item in legs.items():
            result = analyse_leg(tag, item['info'], item['rounds'], curve_name)
            report['legs'].setdefault(tag, {})[curve_name] = result
            for treatment, values in result['treatments'].items():
                print(f'{tag:11s}{treatment:9s}'
                      f'{values["actual"]["us_per_token"]:10.1f}'
                      f'{values["oracle"]["us_per_token"]:10.1f}'
                      f'{values["oracle_gap_pct"]:8.2f}'
                      f'{values["best_fixed"]["us_per_token"]:10.1f}'
                      f'{values["fixed_gap_pct"]:9.2f}'
                      f'{values["one_bit_g_truncate"]["us_per_token"]:10.1f}'
                      f'{values["one_bit_g_gap_pct"]:9.2f}'
                      f'{values["one_bit_share_of_oracle"]:7.2f}')

    print()
    print('=' * 79)
    print('IS THE ROUND PREDICTABLE AT ALL? PRE-ROUND STATE VERSUS OUTCOME')
    print('=' * 79)
    ranking = feature_ranking(all_rounds)
    report['feature_ranking'] = ranking
    print(f'{"feature":12s}{"r(accepted)":>13s}{"r(accepts<3)":>14s}')
    for row in ranking:
        print(f'{row["feature"]:12s}{row["r_accepted"]:13.3f}'
              f'{row["r_shallow"]:14.3f}')

    print()
    print('=' * 79)
    print('RUNG 4 - BEST POLICY ON PRE-ROUND STATE ONLY, HELD OUT')
    print('=' * 79)
    print(f'{"class":12s}{"treat":9s}{"design":20s}{"actual":>10s}'
          f'{"fitted":>10s}{"gain %":>9s}{"oracle %":>9s}{"recov %":>9s}')
    fits = {}
    designs = splits(all_rounds, list(legs))
    for class_name in ('absolute', 'one_bit_g', 'margin_gate'):
        for treatment in ('observed', 'impute'):
            folds = []
            for name, train, test in designs:
                seed = actual(usable(train, treatment),
                              ranked_round_us)['us_per_token']
                fitted = fit_policy(train, test, ranked_round_us, treatment,
                                    args.tree_depth, seed, class_name)
                reference = actual(usable(test, treatment), ranked_round_us)
                oracle_test = oracle(test, ranked_round_us, treatment,
                                     reference['us_per_token'])
                gain = 100.0 * (reference['us_per_token']
                                - fitted['held_out']['us_per_token']) \
                    / reference['us_per_token']
                oracle_gap = 100.0 * (reference['us_per_token']
                                      - oracle_test['us_per_token']) \
                    / reference['us_per_token']
                share = gain / oracle_gap if oracle_gap > 0 else float('nan')
                control = None
                if class_name in CLAMP_CLASSES:
                    actions = action_set(class_name)
                    changed = clamp_count(
                        test, treatment,
                        lambda r: apply_tree(fitted['tree'], r, actions))
                    control = random_clamp_control(
                        test, ranked_round_us, treatment, changed)
                    control['beaten_by_fit'] = gain > control['p95_gain_pct']
                folds.append(dict(
                    fold=name, treatment=treatment, policy_class=class_name,
                    held_out_us_per_token=fitted['held_out']['us_per_token'],
                    actual_us_per_token=reference['us_per_token'],
                    oracle_us_per_token=oracle_test['us_per_token'],
                    train_us_per_token=fitted['train_us_per_token'],
                    gain_pct=gain, oracle_gap_pct=oracle_gap,
                    recovered_share=share, random_control=control,
                    tree_text=describe_tree(fitted['tree'])))
                print(f'{class_name:12s}{treatment:9s}{name:20s}'
                      f'{reference["us_per_token"]:10.1f}'
                      f'{fitted["held_out"]["us_per_token"]:10.1f}'
                      f'{gain:+9.2f}{oracle_gap:9.2f}{100 * share:9.1f}')
                if control is not None and treatment == 'observed':
                    print(f'        random clamp control: '
                          f'{control["clamped"]}/{control["rounds"]} rounds, '
                          f'mean {control["mean_gain_pct"]:+.2f} % '
                          f'p95 {control["p95_gain_pct"]:+.2f} % '
                          f'max {control["max_gain_pct"]:+.2f} % '
                          f'-> fit beats control: {control["beaten_by_fit"]}')
                if class_name in CLAMP_CLASSES and treatment == 'observed':
                    for line in describe_tree(fitted['tree'], '        '):
                        print(line)
            fits[f'{class_name}|{treatment}'] = folds
    report['rung4'] = fits

    headline = report['legs'][args.headline]['ranked']
    print()
    print('=' * 79)
    print('HEADLINE - RANKED CURVE AT THE RANKED OPERATING POINT')
    print('=' * 79)
    width = headline['score']['effective_mean_draft_len'] + 1.0
    print(f'leg {args.headline}, mean verify width {width:.3f}, nearest '
          f'ranked prompt {nearest_prompt(width)} '
          f'({RANKED_WIDTH[nearest_prompt(width)]:.3f})')
    signs = set()
    for treatment, values in headline['treatments'].items():
        gap = values['oracle_gap_pct']
        signs.add(gap > 0)
        print(f'  {treatment:9s} oracle gap {gap:6.2f} %   '
              f'one-bit G {values["one_bit_g_gap_pct"]:6.2f} %   '
              f'best fixed {values["fixed_gap_pct"]:+6.2f} %')
    print(f'  censoring treatments agree in sign: {len(signs) == 1}')
    worst = min(v['oracle_gap_pct'] for v in headline['treatments'].values())
    best = max(v['oracle_gap_pct'] for v in headline['treatments'].values())
    print(f'  oracle gap range {worst:.2f} % .. {best:.2f} %')
    print(f'  published move if the FULL oracle gap were realised on every '
          f'prompt: {published_move(best)["published_pct"]:+.4f} %')
    report['headline_summary'] = dict(
        leg=args.headline, width=width, nearest_prompt=nearest_prompt(width),
        oracle_gap_pct_min=worst, oracle_gap_pct_max=best,
        treatments_agree_in_sign=len(signs) == 1,
        published_move_at_oracle=published_move(best))

    print()
    print('=' * 79)
    print('SENSITIVITY TO THE SIZE OF THE G BOUNDARY')
    print('=' * 79)
    print('The ranked curve fit puts the M=4 -> M=5 jump at +23.0 %. The local')
    print('curve puts it at +46.2 %. Every one-bit G gain comes from that jump,')
    print('so the headline leg is repriced across the range.')
    print()
    print(f'{"boundary step":>14s}{"actual":>10s}{"oracle":>10s}'
          f'{"oracle %":>10s}{"1bitG":>10s}{"1bitG %":>9s}')
    sensitivity = []
    headline_rounds = legs[args.headline]['rounds']
    for step in (0.10, 0.15, 0.230, 0.30, 0.462):
        curve = boundary_curve(step)
        reference = actual(headline_rounds, curve)
        oracle_result = oracle(headline_rounds, curve, 'observed',
                               reference['us_per_token'])
        bit = one_bit_g(headline_rounds, curve, 'observed',
                        reference['us_per_token'], 'truncate')
        row = dict(
            boundary_step_pct=100.0 * step,
            actual_us_per_token=reference['us_per_token'],
            oracle_us_per_token=oracle_result['us_per_token'],
            oracle_gap_pct=100.0 * (reference['us_per_token']
                                    - oracle_result['us_per_token'])
            / reference['us_per_token'],
            one_bit_g_us_per_token=bit['us_per_token'],
            one_bit_g_gap_pct=100.0 * (reference['us_per_token']
                                       - bit['us_per_token'])
            / reference['us_per_token'])
        sensitivity.append(row)
        print(f'{100 * step:13.1f}%{row["actual_us_per_token"]:10.1f}'
              f'{row["oracle_us_per_token"]:10.1f}{row["oracle_gap_pct"]:10.2f}'
              f'{row["one_bit_g_us_per_token"]:10.1f}'
              f'{row["one_bit_g_gap_pct"]:9.2f}')
    report['boundary_sensitivity'] = sensitivity

    print()
    print('=' * 79)
    print('WIDTH SURROGATE FOR THE SCORING PROMPTS')
    print('=' * 79)
    print('One local fixture cannot produce eight prompts. The cap sweep moves')
    print('the local mean verify width across the band the scoring prompts')
    print('occupy, so each leg is the nearest available surrogate for the')
    print('prompt at its width. This is a width match, not a prompt match.')
    print()
    print(f'{"leg":11s}{"width":>8s}{"prompt":>10s}{"prompt w":>10s}'
          f'{"oracle %":>10s}{"1bitG %":>9s}{"fixed %":>9s}')
    surrogates = []
    for tag in args.legs:
        result = report['legs'][tag]['ranked']
        leg_width = result['score']['effective_mean_draft_len'] + 1.0
        values = result['treatments']['observed']
        prompt = nearest_prompt(leg_width)
        surrogates.append(dict(
            leg=tag, width=leg_width, prompt=prompt,
            prompt_width=RANKED_WIDTH[prompt],
            oracle_gap_pct=values['oracle_gap_pct'],
            one_bit_g_gap_pct=values['one_bit_g_gap_pct'],
            fixed_gap_pct=values['fixed_gap_pct']))
        print(f'{tag:11s}{leg_width:8.3f}{prompt:>10s}'
              f'{RANKED_WIDTH[prompt]:10.3f}{values["oracle_gap_pct"]:10.2f}'
              f'{values["one_bit_g_gap_pct"]:9.2f}'
              f'{values["fixed_gap_pct"]:+9.2f}')
    report['width_surrogate'] = surrogates

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print()
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
