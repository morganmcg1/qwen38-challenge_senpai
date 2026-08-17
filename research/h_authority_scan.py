#!/usr/bin/env python3
"""Where does headStepCostRatio still have authority once the width caps bind?

Replicates the SHIPPED scalar walk in Qwen36MTPBlockSession.costModelDepth:

    threshold = h * (1 + expected) / (1 + depth*h)
    guard reach > threshold else break
    expected += reach; depth += 1

The depth-0 confidence damping is omitted (it needs live top-2 margins and it
only ever LOWERS p, i.e. it can only make the walk shallower -- so the depths
printed here are upper bounds on the cost model's demand).
"""


def walk(q, h, cap):
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= q
        thr = h * (1.0 + expected) / (1.0 + depth * h)
        if not (reach > thr):
            break
        expected += reach
        depth += 1
    return depth


QS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.9190, 0.95, 0.99, 1.00]
BIG = 16  # cap released, to expose raw cost-model demand

print("raw cost-model demand (cap released to %d)" % BIG)
print(" q      h=0.20  h=0.18   delta")
rows = [(q, walk(q, 0.20, BIG), walk(q, 0.18, BIG)) for q in QS]
print("\n".join(
    " %.4f    %2d      %2d     %+d" % (q, a, b, b - a) for q, a, b in rows))

print()
print("effective depth AFTER the shipped ternary clamp")
print("cold branch = sdpaWidthWallDepthCap, hot branch = segmentedVerifyDepthCap")
print()
for label, cold, hot in (
    ("ours      (cold 4, hot 7)", 4, 7),
    ("frontier  (cold 5, hot 8)", 5, 8),
    ("composed  (cold 5, hot 7)", 5, 7),
):
    line20 = [min(walk(q, 0.20, BIG), hot) for q in QS]
    line18 = [min(walk(q, 0.18, BIG), hot) for q in QS]
    diff = sum(1 for a, b in zip(line20, line18) if a != b)
    cold20 = [min(walk(q, 0.20, BIG), cold) for q in QS]
    cold18 = [min(walk(q, 0.18, BIG), cold) for q in QS]
    cdiff = sum(1 for a, b in zip(cold20, cold18) if a != b)
    print("%s" % label)
    print("   hot  branch: h=0.20 %s" % line20)
    print("   hot  branch: h=0.18 %s" % line18)
    print("   -> q values where h still changes the HOT depth : %d/%d" % (diff, len(QS)))
    print("   cold branch: h=0.20 %s" % cold20)
    print("   cold branch: h=0.18 %s" % cold18)
    print("   -> q values where h still changes the COLD depth: %d/%d" % (cdiff, len(QS)))
    print()

# The measured operating point from Alphonse r5 cap-7 winner arm.
Q_OBS = 0.9189765458422174
print("at the MEASURED cap-7 acceptance q = %.16f" % Q_OBS)
for h in (0.2562, 0.20, 0.18):
    print("   h=%.4f -> raw demand %d  | clamped cold(4) %d  cold(5) %d  hot(7) %d  hot(8) %d"
          % (h, walk(Q_OBS, h, BIG),
             min(walk(Q_OBS, h, BIG), 4), min(walk(Q_OBS, h, BIG), 5),
             min(walk(Q_OBS, h, BIG), 7), min(walk(Q_OBS, h, BIG), 8)))
