"""E127 design: the shipped depth price against the LIVE dispatch table.

Shipped rule (Qwen36MTPBlockSession.costModelDepth):

    threshold[d] = marginal[d] * (1 + expected) / cumulative[d]
    extend iff reach > threshold

`marginal[d]` prices the step into verify width d+2 as a fraction of the
width-1 verify forward V, and is defined as `h + (C(d+2) - C(d+1)) / V` with
`h = headStepCostRatio`.  Every arm is rescaled so the total stays at
`maxDepth * h = 1.44`, so an arm can change the SHAPE but never the LEVEL.

Two independent staleness claims are tested here.

SHAPE.  `measuredRawDepthPrice` was fitted on E68's table, whose step into
verify width 5 cost 13.405 ms and whose step into width 6 cost 27.308 ms
against V = 60.300 ms.  Edward's E92 production curve reproduced that table:
its marginal round cost exploded at width 5 (40.2 ms) and width 9 (39.7 ms).
E100 then replaced `<T,5,3,true>` with `<T,5,5,true>`, which collapses width 5
to a single group.  F52's shipped-frame curve shows the width-5 cliff is gone
and the surviving cliffs are at widths 4, 6 and 9.

LEVEL.  F13 measured the proposal head at 7.70 % of the local round and 1.82 %
of the ranked round.  Converted to a per-step fraction of V those give h about
0.026 local and about 0.0075 ranked, against the shipped 0.18.
"""

MAXD = 8
CAP = 7               # segmentedVerifyDepthCap
H_SHIP = 0.18

# F52 shipped-frame mlp.gate_up microseconds per matvec at verify width M.
GATE_UP = {1: 432.26, 2: 434.08, 3: 457.13, 4: 592.09, 5: 664.87,
           6: 891.86, 7: 962.53, 8: 1026.43, 9: 1280.11}
ROUND_M5 = 127533.0
STREAM_SHARE = 0.89757

stream_m5 = ROUND_M5 * STREAM_SHARE
fixed = ROUND_M5 - stream_m5
Cabs = {m: fixed + stream_m5 * GATE_UP[m] / GATE_UP[5] for m in GATE_UP}
V = Cabs[1]
C = [Cabs[m] / V for m in range(1, 10)]        # C[i] = relative cost at width i+1

# Edward E92 production round, the table E68/pbfit was fitted on.
E92 = [64497.2, 69812.4, 74784.4, 86233.6, 126447.4, 137780.7,
       151393.7, 164314.6, 204051.1]
C92 = [x / E92[0] for x in E92]

E68_RAW = [0.26300121724709807, 0.29195567495854047, 0.34642143034825884,
           0.40231023217247086, 0.63287276451077956, 0.43601634825870655,
           0.35457813598673293, 0.42510483416251998]

print("relative whole-round verify cost C(width)/V")
print("  width      %s" % " ".join("%8d" % w for w in range(1, 10)))
print("  live (F52) %s" % " ".join("%8.4f" % x for x in C))
print("  E92 table  %s" % " ".join("%8.4f" % x for x in C92))
print()
print("marginal verify step (C(w) - C(w-1)) / V")
print("  into width %s" % " ".join("%8d" % w for w in range(2, 10)))
print("  live (F52) %s" % " ".join("%8.4f" % (C[i + 1] - C[i])
                                   for i in range(8)))
print("  E92 table  %s" % " ".join("%8.4f" % (C92[i + 1] - C92[i])
                                   for i in range(8)))
print("  E68 raw-h  %s" % " ".join("%8.4f" % (x - H_SHIP) for x in E68_RAW))
print()


def rescale(raw, level):
    s = level / sum(raw)
    return [x * s for x in raw]


def prefix(marg):
    out, run = [1.0], 1.0
    for v in marg:
        run += v
        out.append(run)
    return out


def greedy(marg, p_at):
    cum = prefix(marg)
    reach, expected, depth = 1.0, 0.0, 0
    while depth < CAP:
        reach *= p_at(depth)
        thr = marg[depth] * (1.0 + expected) / cum[depth]
        if not reach > thr:
            break
        expected += reach
        depth += 1
    return depth


def true_cost(d, h):
    """True relative round cost at depth d: verify at width d+1 plus d head steps."""
    return C[d] + d * h


def yields(p_at):
    out, reach, tot = [1.0], 1.0, 1.0
    for d in range(MAXD):
        reach *= p_at(d)
        tot += reach
        out.append(tot)
    return out


def best_depth(p_at, h):
    Y = yields(p_at)
    vals = [true_cost(d, h) / Y[d] for d in range(CAP + 1)]
    return vals.index(min(vals)), vals


live_raw = [H_SHIP + (C[d + 1] - C[d]) for d in range(MAXD)]
ARMS = {
    "ship uniform 0.18": [H_SHIP] * MAXD,
    "pbfit E68 stale": rescale(E68_RAW, MAXD * H_SHIP),
    "refit live, level 1.44": rescale(live_raw, MAXD * H_SHIP),
    "refit live, true level": [0.026 + (C[d + 1] - C[d]) for d in range(MAXD)],
}
print("candidate marginal vectors")
for name, m in ARMS.items():
    print("  %-24s %s   total %.3f" % (
        name, " ".join("%7.4f" % x for x in m), sum(m)))
print()

print("chosen depth and realised cost per emitted token on the LIVE curve")
print("h used for the true cost: 0.026 (F13 local).  cap %d" % CAP)
print()
print("%-6s %-28s %-28s %-28s %-28s %s" % (
    "p", "ship uniform 0.18", "pbfit E68 stale",
    "refit live level 1.44", "refit live true level", "ORACLE"))
for pc in range(70, 100, 3):
    p = pc / 100.0

    def p_at(d, p=p):
        return p
    cells = []
    Y = yields(p_at)
    for name in ARMS:
        d = greedy(ARMS[name], p_at)
        cells.append("d=%d  cost/tok %.4f" % (d, true_cost(d, 0.026) / Y[d]))
    bd, vals = best_depth(p_at, 0.026)
    print("%-6.2f %-28s %-28s %-28s %-28s d=%d  %.4f" % (
        p, cells[0], cells[1], cells[2], cells[3], bd, vals[bd]))
print()

print("gap of each arm to the oracle, per cent of candidate time")
print("%-6s %10s %10s %10s %10s" % (
    "p", "ship", "pbfit", "refit1.44", "refitTrue"))
for pc in range(70, 100, 3):
    p = pc / 100.0

    def p_at(d, p=p):
        return p
    Y = yields(p_at)
    bd, vals = best_depth(p_at, 0.026)
    row = []
    for name in ARMS:
        d = greedy(ARMS[name], p_at)
        row.append(100.0 * (true_cost(d, 0.026) / Y[d] / vals[bd] - 1.0))
    print("%-6.2f %10.2f %10.2f %10.2f %10.2f" % (p, *row))
print()

print("same, with a decaying profile p_d = p0 * 0.94^d")
print("%-6s %10s %10s %10s %10s   %s" % (
    "p0", "ship", "pbfit", "refit1.44", "refitTrue", "oracle d / arms d"))
for pc in range(75, 100, 3):
    p0 = pc / 100.0

    def p_at(d, p0=p0):
        return p0 * (0.94 ** d)
    Y = yields(p_at)
    bd, vals = best_depth(p_at, 0.026)
    row, ds = [], []
    for name in ARMS:
        d = greedy(ARMS[name], p_at)
        ds.append(d)
        row.append(100.0 * (true_cost(d, 0.026) / Y[d] / vals[bd] - 1.0))
    print("%-6.2f %10.2f %10.2f %10.2f %10.2f   %d / %s" % (
        p0, *row, bd, ",".join(str(x) for x in ds)))
