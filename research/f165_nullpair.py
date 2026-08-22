#!/usr/bin/env python3
"""F165: the exact-replicate null pair 48423d09 -> bed5081a (zero source delta)."""
import json
import statistics as st

D = json.load(open("/tmp/yukon-board/full.json"))["submissions"]
NAME = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
        "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
        "ea82dcb5": "republic", "3b10cb4d": "travel"}


def row(p):
    hits = [s for s in D if str(s.get("id", "")).startswith(p)]
    assert len(hits) == 1, (p, len(hits))
    return hits[0]


def pp(r):
    return {NAME[str(x["prompt_sha256"])[:8]]: x
            for x in r["officialMetrics"]["per_prompt"]
            if str(x["prompt_sha256"])[:8] in NAME}


A, B = row("48423d09"), row("bed5081a")
pa, pb = pp(A), pp(B)
order = ["beagle", "medicine", "essays", "botany", "republic", "plutarch", "drama", "travel"]
cand, ser, raw_a, raw_b = [], [], [], []
print(f"{'prompt':10s} {'cand d%':>9s} {'serial d%':>10s} {'raw d%':>9s}")
for n in order:
    ca, cb = pa[n]["mtp_seconds_per_token_mean"], pb[n]["mtp_seconds_per_token_mean"]
    sa, sb = pa[n]["serial_seconds_per_token_mean"], pb[n]["serial_seconds_per_token_mean"]
    dc, ds = (cb / ca - 1) * 100, (sb / sa - 1) * 100
    ra, rb = sa / ca, sb / cb
    cand.append(dc)
    ser.append(ds)
    raw_a.append(ra)
    raw_b.append(rb)
    print(f"{n:10s} {dc:+9.4f} {ds:+10.4f} {(rb / ra - 1) * 100:+9.4f}")


def med8(v):
    s = sorted(v)
    return (s[3] + s[4]) / 2


print()
print(f"candidate 8-prompt mean delta  {st.mean(cand):+.4f} %   "
      f"sd {st.stdev(cand):.4f}   se of mean {st.stdev(cand) / 8 ** .5:.4f}")
print(f"serial    8-prompt mean delta  {st.mean(ser):+.4f} %   "
      f"sd {st.stdev(ser):.4f}   se of mean {st.stdev(ser) / 8 ** .5:.4f}")
print(f"published median  {med8(raw_a):.8f} -> {med8(raw_b):.8f}  "
      f"= {(med8(raw_b) / med8(raw_a) - 1) * 100:+.4f} %")
print(f"officialScore     {A['officialScore']:.8f} -> {B['officialScore']:.8f}")
sa_ = sorted(range(8), key=lambda i: raw_a[i])
sb_ = sorted(range(8), key=lambda i: raw_b[i])
print(f"median carriers A {order[sa_[3]]}, {order[sa_[4]]}   "
      f"B {order[sb_[3]]}, {order[sb_[4]]}")
