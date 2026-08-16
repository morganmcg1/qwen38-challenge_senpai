"""Depth-policy arithmetic for the Qwen 3.8 MTP track.

Compares the shipped `costModelDepth` hill-climb against the same objective
evaluated as a global argmax, under the measured per-draft cost curve.

Cost curve: PR #5 isolated qmv measurements, cross-validated against Edward's
in-situ h(d).  h_meas[j-1] is the marginal cost of the j-th draft -- i.e. the
step that widens the verify from M=j to M=j+1 -- in units of a width-1 verify.

Acceptance vectors mirror `positionAcceptEMA` in
Sources/MLXFastModel/Qwen36MTPBlockSession.swift:497.
"""

MAX_DEPTH = 8

h_meas = [0.0862, 0.0795, 0.2446, 0.3774, 0.2939, 0.3020, 0.2890, 0.3929]
h_flat = [0.20] * MAX_DEPTH

# The shipped seed prior: 0.85 * 0.98**i  (optimistic, gently decaying).
p_seed = [0.85 * (0.98 ** i) for i in range(MAX_DEPTH)]
# Converged easy-prose: the 0.95 optimism cap binds at every position.
p_capped = [0.95] * MAX_DEPTH
# Flat-q idealisations, for comparison against the ms/token table.
p_q1000 = [1.000] * MAX_DEPTH
p_q0976 = [0.976] * MAX_DEPTH
p_q0940 = [0.940] * MAX_DEPTH
# The real-prose production conditionals the code comment says were tried.
p_prose = [0.92, 0.70, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]


def greedy(h, p, cap=MAX_DEPTH):
    """Exact transcription of the shipped costModelDepth loop."""
    reach, expected, cum, depth = 1.0, 0.0, 0.0, 0
    while depth < cap:
        reach *= p[depth]
        thr = h[depth] * (1.0 + expected) / (1.0 + cum)
        if not (reach > thr):
            break
        expected += reach
        cum += h[depth]
        depth += 1
    return depth


def argmax(h, p, cap=MAX_DEPTH):
    """Global argmax of tokens-per-verify-unit over the same candidate range."""
    reach, expected, cum = 1.0, 0.0, 0.0
    best, best_ratio = 0, 1.0
    for depth in range(cap):
        reach *= p[depth]
        cum += h[depth]
        expected += reach
        ratio = (1.0 + expected) / (1.0 + cum)
        if ratio > best_ratio:
            best_ratio, best = ratio, depth + 1
    return best, best_ratio


def ratio_at(h, p, depth):
    reach, expected, cum = 1.0, 0.0, 0.0
    for d in range(depth):
        reach *= p[d]
        cum += h[d]
        expected += reach
    return (1.0 + expected) / (1.0 + cum)


REGIMES = [
    ("seed prior 0.85*0.98^i", p_seed),
    ("converged, optimism cap 0.95", p_capped),
    ("flat q=1.000", p_q1000),
    ("flat q=0.976", p_q0976),
    ("flat q=0.940", p_q0940),
    ("real-prose 0.92/0.70/0.50", p_prose),
]

print("policy selection by acceptance regime (cap = 8)")
print(f"{'regime':<30}{'shipped':>9}{'meas-greedy':>13}{'meas-argmax':>13}{'argmax gain':>13}")
for name, p in REGIMES:
    s = greedy(h_flat, p)
    g = greedy(h_meas, p)
    a, ar = argmax(h_meas, p)
    gain = ar / ratio_at(h_meas, p, s) - 1.0
    print(f"{name:<30}{s:>9}{g:>13}{a:>13}{gain:>+12.1%}")

print()
print("same, with the streak gate closed (cap = 4, sdpaWidthWallDepthCap)")
print(f"{'regime':<30}{'shipped':>9}{'meas-greedy':>13}{'meas-argmax':>13}")
for name, p in REGIMES:
    print(f"{name:<30}{greedy(h_flat, p, 4):>9}{greedy(h_meas, p, 4):>13}{argmax(h_meas, p, 4)[0]:>13}")

print()
print("tokens per verify-unit, by regime and depth (measured cost curve)")
print("regime".ljust(30) + "".join(f"{d:>8}" for d in range(MAX_DEPTH + 1)))
for name, p in REGIMES:
    print(name.ljust(30) + "".join(f"{ratio_at(h_meas, p, d):>8.3f}" for d in range(MAX_DEPTH + 1)))


# ---------------------------------------------------------------------------
# Closed-loop simulation.
#
# The static tables above are misleading: positionAcceptEMA is a RATCHET.
# recordAcceptOutcome only updates positions the round actually reached, so
# the policy's own depth choice determines which EMAs get evidence.  A fully
# accepted round additionally transfers optimism (capped 0.95) to the single
# position just past the round.  The honest question is what the closed loop
# settles on, not what a frozen acceptance vector implies.
# ---------------------------------------------------------------------------

ALPHA = 0.15
STREAK_GATE = 3
DEEP_CAP = 8
WALL_CAP = 4


def record(ema, accepted, drafted):
    for i in range(min(accepted, len(ema))):
        ema[i] += ALPHA * (1.0 - ema[i])
    if accepted < drafted and accepted < len(ema):
        ema[accepted] += ALPHA * (0.0 - ema[accepted])
    elif accepted == drafted and drafted > 0 and accepted < len(ema):
        if ema[accepted] < 0.95:
            ema[accepted] += ALPHA * (0.95 - ema[accepted])


def simulate(h, chooser, truth, rounds=400, seed=12345):
    """Run the closed loop; `truth[i]` is the real P(accept i | 0..i-1 accepted)."""
    rng = seed
    ema = [0.85 * (0.98 ** i) for i in range(MAX_DEPTH)]
    streak = 0
    hist = [0] * (MAX_DEPTH + 1)
    tokens = 0.0
    cost = 0.0
    for _ in range(rounds):
        cap = DEEP_CAP if streak >= STREAK_GATE else WALL_CAP
        d = chooser(h, ema, cap)
        hist[d] += 1
        accepted = 0
        for i in range(d):
            rng = (1103515245 * rng + 12345) % (1 << 31)
            if (rng / (1 << 31)) < truth[i]:
                accepted += 1
            else:
                break
        record(ema, accepted, d)
        streak = streak + 1 if (d > 0 and accepted == d) else 0
        tokens += accepted + 1
        cost += 1.0 + sum(h[:d])
    return hist, tokens / cost, ema


def choose_shipped(h, ema, cap):
    return greedy(h_flat, ema, cap)


def choose_curve_greedy(h, ema, cap):
    return greedy(h_meas, ema, cap)


def choose_curve_argmax(h, ema, cap):
    return argmax(h_meas, ema, cap)[0]


TRUTHS = [
    ("easy prose  (0.98 flat)", [0.98] * MAX_DEPTH),
    ("mid prose   (0.93 flat)", [0.93] * MAX_DEPTH),
    ("decaying    (0.97^(i+1))", [0.97 ** (i + 1) for i in range(MAX_DEPTH)]),
    ("hard prose  (0.85 flat)", [0.85] * MAX_DEPTH),
]

print()
print("CLOSED-LOOP simulation, 400 rounds, realised tokens per verify-unit")
print(f"{'ground truth':<26}{'shipped':>22}{'curve+greedy':>22}{'curve+argmax':>22}")
for tname, truth in TRUTHS:
    row = ""
    base = None
    for chooser in (choose_shipped, choose_curve_greedy, choose_curve_argmax):
        hist, tpu, _ = simulate(h_meas, chooser, truth)
        mode = max(range(MAX_DEPTH + 1), key=lambda d: hist[d])
        if base is None:
            base = tpu
            row += f"{tpu:>13.3f} (d~{mode})"
        else:
            row += f"{tpu:>9.3f} {tpu/base-1:>+6.1%} (d~{mode})"
    print(f"{tname:<26}{row}")
