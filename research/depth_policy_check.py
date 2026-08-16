"""Depth-policy arithmetic for the Qwen 3.8 MTP track.

Compares the shipped `costModelDepth` hill-climb against the same objective
evaluated as a global argmax, under the measured per-draft cost curve.

!!! THE COST CURVE BELOW IS ASSUMED, NOT MEASURED. !!!

`h_assumed` was recorded in this campaign's notes as "Edward's in-situ h(d)".
It is not.  PR #1 has never reported: zero student commits, zero student
comments, and `git log --all -S 0.0862` returns only the advisor's own research
commits.  It is additionally falsified by its own endpoint --

    sum(h_assumed)      = 2.0655
    implies C(8) = 67.0 * 3.0655 = 205.4 ms
    measured C(8)                = 161.0 ms   (PR #3 parent-clock algebra)
    error                        = +27.6%

-- so it overstates total depth cost by 1.47x (local head) or 1.91x (ranked).
A real in-situ marginal curve must satisfy sum(h) ~= 1.403 local / 1.082
ranked.  `endpoint_error()` below checks any candidate vector against that.

EVERY DEPTH AND PERCENTAGE THIS SCRIPT PRINTS IS THEREFORE PROVISIONAL.  The
structural conclusions are not: that `costModelDepth` is a strict hill-climb,
that it is only correct when h is flat, and that `positionAcceptEMA` is a
ratchet, all follow from the Swift source and are independent of h.

To use real data, replace `h_assumed` and rerun.  h[j-1] is the marginal cost
of the j-th draft -- the step that widens the verify from M=j to M=j+1 -- in
units of a width-1 verify.

Acceptance vectors mirror `positionAcceptEMA` in
Sources/MLXFastModel/Qwen36MTPBlockSession.swift:497.
"""

MAX_DEPTH = 8

# C(0) and C(8) are the two *measured* endpoints (PR #3 parent-clock algebra).
C0_MS = 67.0
C8_MS_LOCAL = 161.0
HEAD_REBASE_MS = 2.689  # local bf16 head - ranked 4-bit head, per draft step

h_assumed = [0.0862, 0.0795, 0.2446, 0.3774, 0.2939, 0.3020, 0.2890, 0.3929]


def endpoint_error(h):
    """Fraction by which a candidate curve misses the measured C(8)."""
    return C0_MS * (1.0 + sum(h)) / C8_MS_LOCAL - 1.0


def required_sum_h(ranked=False):
    c8 = C8_MS_LOCAL - (HEAD_REBASE_MS * MAX_DEPTH if ranked else 0.0)
    return c8 / C0_MS - 1.0


h_meas = h_assumed
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


def simulate(h, chooser, truth, rounds=400, seed=12345, wall_cap=WALL_CAP):
    """Run the closed loop; `truth[i]` is the real P(accept i | 0..i-1 accepted)."""
    rng = seed
    ema = [0.85 * (0.98 ** i) for i in range(MAX_DEPTH)]
    streak = 0
    hist = [0] * (MAX_DEPTH + 1)
    tokens = 0.0
    cost = 0.0
    for _ in range(rounds):
        cap = DEEP_CAP if streak >= STREAK_GATE else wall_cap
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
print("  [PROVISIONAL: h_assumed is not measured and misses the measured C(8)")
print(f"   by {endpoint_error(h_assumed):+.1%}. See the endpoint sensitivity below.]")
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

# ---------------------------------------------------------------------------
# ENDPOINT SENSITIVITY.  The two endpoints C(0)=67.0 and C(8)=161.0 are the only
# measured facts about the depth-cost curve; the shape between them is the open
# question PR #1 exists to answer.  Every candidate below is rescaled to satisfy
# the measured endpoint exactly, so they differ ONLY in shape.  The spread of
# this table is the honest uncertainty on the depth line.
# ---------------------------------------------------------------------------


def rescale(h, target):
    s = target / sum(h)
    return [x * s for x in h]


TARGET = required_sum_h()  # 1.403 against the measured local C(8)

# Null: no shape at all.  If true, the shipped flat 0.20 is nearly right and
# there is no prize here.
h_null = rescale([1.0] * MAX_DEPTH, TARGET)
# The retracted vector's shape, corrected to the measured endpoint.
h_shape = rescale(h_assumed, TARGET)
# Thorfinn's PR #5 isolated law: a per-step head floor, verify ~free to M~=3,
# then a linear ramp, plus stream-boundary bumps at M=5 (d=4) and M=9 (d=8).
h_ramp = rescale(
    [0.06, 0.06, 0.20, 0.30, 0.24, 0.24, 0.24, 0.34], TARGET
)
# Pessimistic alternative: cost concentrated early, flat late (a shape nobody
# has argued for, included so the table is not stacked in my favour).
h_front = rescale([0.34, 0.30, 0.24, 0.20, 0.16, 0.14, 0.12, 0.10], TARGET)

CANDIDATES = [
    ("null: flat, sum=1.403", h_null),
    ("retracted shape, rescaled", h_shape),
    ("PR #5 ramp+boundaries", h_ramp),
    ("front-loaded (adversarial)", h_front),
    ("h_assumed AS RECORDED (bad endpoint)", h_assumed),
]

print()
print("ENDPOINT SENSITIVITY: same closed loop, curves rescaled to the measured C(8)")
print("gain of curve+greedy over shipped; the spread is the real uncertainty")
print("candidate curve".ljust(38) + "".join(f"{t.split('(')[0].strip():>12}" for t, _ in TRUTHS))
for cname, hc in CANDIDATES:
    h_meas = hc  # choosers read this global
    row = ""
    for _, truth in TRUTHS:
        _, tpu_ship, _ = simulate(hc, choose_shipped, truth)
        hist, tpu_new, _ = simulate(hc, choose_curve_greedy, truth)
        mode = max(range(MAX_DEPTH + 1), key=lambda d: hist[d])
        row += f"{tpu_new/tpu_ship-1:>+8.1%} d{mode}"
    print(f"{cname:<38}{row}")
h_meas = h_assumed


# ---------------------------------------------------------------------------
# CURVE-INDEPENDENT INTERVENTIONS
#
# Everything above needs a curve *shape* nobody has measured.  The two knobs
# below need only the two endpoints we really measured, C(0)=67.0 and
# C(8)=161.0, so they can be evaluated today and are each a one-line change.
#
#   1. The shipped scalar h=0.20 is simply too big.  The measured endpoints
#      pin the average marginal step at (C(8)-C(0))/8/C(0):
#         local  : (161.0-67.0)/8/67.0 = 0.1754   -> shipped overprices 1.14x
#         ranked : (139.5-67.0)/8/67.0 = 0.1353   -> shipped overprices 1.48x
#      The ranked leg is the one that scores.  Overpricing depth makes the
#      hill-climb stop early, so this biases the shipped policy *shallow*.
#      Crucially the loop's correctness proof needs h flat -- and this fix
#      keeps it flat.  It is a constant edit, not a new algorithm.
#
#   2. The width wall caps depth at 4 whenever fullAcceptStreak < 3.
#      Removing it is independent of the cost curve entirely.
# ---------------------------------------------------------------------------

H_LOCAL = (C8_MS_LOCAL - C0_MS) / MAX_DEPTH / C0_MS
H_RANKED = (C8_MS_LOCAL - HEAD_REBASE_MS * MAX_DEPTH - C0_MS) / MAX_DEPTH / C0_MS

h_local = [H_LOCAL] * MAX_DEPTH
h_ranked = [H_RANKED] * MAX_DEPTH


def choose_flat(hv):
    return lambda h, ema, cap: greedy(hv, ema, cap)


# An arm is a *belief* about cost plus a cap.  The realised cost is charged
# from `h_true`, which is identical across arms -- otherwise we would be
# rewarding an arm for believing depth is cheap rather than for acting well.
ARMS = [
    ("shipped (h=0.20, wall=4)", h_flat, WALL_CAP),
    ("h retuned local  (0.175)", h_local, WALL_CAP),
    ("h retuned RANKED (0.135)", h_ranked, WALL_CAP),
    ("wall off only (h=0.20)", h_flat, DEEP_CAP),
    ("both: h=0.135 + wall off", h_ranked, DEEP_CAP),
]

# Two honest truths, both hitting the measured ranked endpoint sum=1.0819:
# a flat null shape, and the PR #5 ramp (the most physically motivated shape).
TRUE_COSTS = [
    ("truth = flat ranked", rescale([1.0] * MAX_DEPTH, required_sum_h(ranked=True))),
    ("truth = PR#5 ramp, ranked", rescale(
        [0.06, 0.06, 0.20, 0.30, 0.24, 0.24, 0.24, 0.34],
        required_sum_h(ranked=True))),
]

for cost_name, h_true in TRUE_COSTS:
    print()
    print(f"CURVE-INDEPENDENT ARMS -- {cost_name} (only measured endpoints used)")
    print("realised tokens per verify-unit (gain vs shipped), realised depth mode")
    print("arm".ljust(28)
          + "".join(f"{t.split('(')[0].strip():>16}" for t, _ in TRUTHS))
    base = {}
    for aname, belief, wcap in ARMS:
        row = ""
        for tname, truth in TRUTHS:
            hist, tpu, _ = simulate(
                h_true, choose_flat(belief), truth, wall_cap=wcap)
            mode = max(range(MAX_DEPTH + 1), key=lambda d: hist[d])
            if aname.startswith("shipped"):
                base[tname] = tpu
                row += f"{tpu:>10.3f} d{mode}  "
            else:
                row += f"{tpu / base[tname] - 1:>+10.1%} d{mode}  "
        print(f"{aname:<28}{row}")

print()
print("How often does the width wall actually bind? (rounds with streak < 3)")
for tname, truth in TRUTHS:
    ema = [0.85 * (0.98 ** i) for i in range(MAX_DEPTH)]
    rng, streak, walled, deep = 12345, 0, 0, 0
    for _ in range(400):
        cap = DEEP_CAP if streak >= STREAK_GATE else WALL_CAP
        if cap == WALL_CAP:
            walled += 1
        else:
            deep += 1
        d = greedy(h_flat, ema, cap)
        accepted = 0
        for i in range(d):
            rng = (1103515245 * rng + 12345) % (1 << 31)
            if (rng / (1 << 31)) < truth[i]:
                accepted += 1
            else:
                break
        record(ema, accepted, d)
        streak = streak + 1 if (d > 0 and accepted == d) else 0
    print(f"  {tname:<26} wall binds {walled/4.0:5.1f}% of rounds")

