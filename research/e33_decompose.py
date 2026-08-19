#!/usr/bin/env python3
"""Decompose the M=6 result into weight-pass saving vs row-blocking cost.

Within-stream increments C_round(M) - C_round(M-1) at constant weight-stream
count measure the marginal cost of one more activation row. The base's 5->6
increment additionally pays a second weight pass; the candidate's 5->6
increment additionally pays row blocking. Subtracting the within-stream
baseline from each isolates the two effects.
"""
import json
import statistics as st

B = ".mlxfast-private/qmv-curve/e33-base-r1/vendored.json"
C = ".mlxfast-private/qmv-curve/e33-cand-r1/vendored.json"


def cround(path):
    v = json.load(open(path))
    o = {}
    for sh in v["shapes"]:
        for r in sh["rows"]:
            o[r["m"]] = o.get(r["m"], 0.0) + sh["calls_per_verify"] * r["seconds_per_call"]
    return o


cb, cc = cround(B), cround(C)
sb = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
sc = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}

print("| step | base dC (ms) | base | cand dC (ms) | cand |")
print("|---|---:|---|---:|---|")
for m in range(2, 10):
    tb = "same stream" if sb[m] == sb[m - 1] else "**+1 weight pass**"
    tc = "same stream" if sc[m] == sc[m - 1] else "**+1 weight pass**"
    print(f"| {m-1}->{m} | {(cb[m]-cb[m-1])*1e3:.3f} | {tb} | {(cc[m]-cc[m-1])*1e3:.3f} | {tc} |")

ws = [(cb[m] - cb[m - 1]) * 1e3 for m in (4, 5, 8, 9)]
med = st.median(ws)
print(f"\nbase within-stream increments (3->4, 4->5, 7->8, 8->9): "
      f"{[round(x, 2) for x in ws]}  median {med:.2f} ms")
b56 = (cb[6] - cb[5]) * 1e3
c56 = (cc[6] - cc[5]) * 1e3
print(f"base 5->6 (1 -> 2 passes)        : {b56:.3f} ms  =>  second weight pass ~ {b56-med:+.2f} ms")
print(f"cand 5->6 (1 pass, row-blocked)  : {c56:.3f} ms  =>  row-blocking cost   ~ {c56-med:+.2f} ms")
print(f"\nnet at M=6: {(cc[6]-cb[6])*1e3:+.3f} ms  ({cc[6]/cb[6]:.4f}x)")
print(f"the saving is real but the blocking cost is {(c56-med)/(b56-med):.2f}x as large")

# per-shape ordering evidence
v = json.load(open(B))
w = json.load(open(C))
pb = {sh["name"]: sh for sh in v["shapes"]}
pw = {sh["name"]: sh for sh in w["shapes"]}
print("\n| shape | n | k | weight MB | ratio at M=6 |")
print("|---|---:|---:|---:|---:|")
rows = []
for n, sh in pb.items():
    a = [r for r in sh["rows"] if r["m"] == 6][0]["seconds_per_call"]
    b = [r for r in pw[n]["rows"] if r["m"] == 6][0]["seconds_per_call"]
    rows.append((sh["n"], sh["k"], n, sh["weight_bytes"] / 1e6, b / a))
for n_, k_, name, mb, ratio in sorted(rows):
    print(f"| {name} | {n_} | {k_} | {mb:.1f} | {ratio:.4f} |")
