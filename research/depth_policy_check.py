h_meas = [0.0862, 0.0795, 0.2446, 0.3774, 0.2939, 0.3020, 0.2890, 0.3929]  # cost of j-th draft, j=1..8
h_flat = [0.20] * 8


def greedy(h, q, cap=8):
    """Exact transcription of the shipped costModelDepth hill-climb."""
    reach, expected, cum, depth = 1.0, 0.0, 0.0, 0
    while depth < cap:
        reach *= q
        thr = h[depth] * (1.0 + expected) / (1.0 + cum)
        if not (reach > thr):
            break
        expected += reach
        cum += h[depth]
        depth += 1
    return depth


def argmax(h, q, cap=8):
    """Global argmax of tokens-per-verify-unit over the same range."""
    reach, expected, cum = 1.0, 0.0, 0.0
    best, bestr = 0, 1.0
    for depth in range(cap):
        reach *= q
        cum += h[depth]
        expected += reach
        r = (1.0 + expected) / (1.0 + cum)
        if r > bestr:
            bestr, best = r, depth + 1
    return best, bestr


for qname, q in [("q=1.000", 1.0), ("q=0.976", 0.976), ("q=0.940", 0.94), ("q=0.90", 0.90)]:
    g_flat = greedy(h_flat, q)
    g_meas = greedy(h_meas, q)
    a_meas, ar = argmax(h_meas, q)
    print(f"{qname}: shipped-flat-greedy={g_flat}  measured-greedy={g_meas}  measured-ARGMAX={a_meas} (ratio {ar:.4f})")

print()
print("tokens per verify-unit at reach=1 (q=1), measured curve:")
cum = 0.0
for d in range(9):
    if d > 0:
        cum += h_meas[d - 1]
    print(f"  depth={d}  M={d+1}  cost={1+cum:.4f}  tokens={1+d}  ratio={(1+d)/(1+cum):.4f}")
