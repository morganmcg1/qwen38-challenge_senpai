#!/usr/bin/env python3
"""Which prompt occupies the fourth order statistic, and why.

beagle is the fourth smallest ratio on essentially every competitive row, so the
published median is `(beagle + min(essays, medicine, republic, botany)) / 2`.
The second term is a minimum over four near-tied noisy values, so it is biased
low and it is dominated by whichever of the four draws worst. This script
measures how often that holds, how wide the four-prompt cluster is, and how much
of the frontier's lead comes from cluster tightness rather than speed.

    RECEIPT_CACHE=/tmp/board_live_t3.json python3 research/index4_lottery.py
"""
import json
import os
import statistics as st

CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/board_live_t3.json")
NAME = {"c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
        "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
        "a2ea8b60": "essays", "192fb621": "botany"}
FAST = ("essays", "medicine", "republic", "botany")


def profile(row):
    out = {}
    for entry in (row.get("officialMetrics") or {}).get("per_prompt", []):
        out[NAME[entry["prompt_sha256"][:8]]] = entry
    return out if len(out) == 8 else None


def main():
    payload = json.load(open(CACHE))
    rows = payload["submissions"] if isinstance(payload, dict) else payload
    scored = []
    for row in rows:
        prof = profile(row)
        score = row.get("officialScore")
        if prof is None or score in (None, "None"):
            continue
        scored.append((float(score), row, prof))
    scored.sort(key=lambda t: -t[0])

    print(f"rows with a full 8-prompt profile and a score: {len(scored)}")
    print()
    print("=== who occupies index 3 and index 4, top 20 rows by score ===")
    print("rank score      id        solver           idx3      idx4      "
          "fast spread%")
    hits = 0
    for i, (score, row, prof) in enumerate(scored[:20]):
        pairs = sorted(((e["raw_ratio_of_means"], n) for n, e in prof.items()))
        fast = [prof[n]["raw_ratio_of_means"] for n in FAST]
        spread = (max(fast) / min(fast) - 1) * 100
        if pairs[3][1] == "beagle":
            hits += 1
        print(f"{i+1:>4} {score:.6f}  {row['id'][:8]}  "
              f"{str(row.get('solverUsername'))[:15]:<15}  "
              f"{pairs[3][1]:<8}  {pairs[4][1]:<8}  {spread:6.3f}")

    top = [t for t in scored if t[0] >= 3.4]
    beagle3 = sum(1 for _, _, p in top
                  if sorted(((e["raw_ratio_of_means"], n)
                             for n, e in p.items()))[3][1] == "beagle")
    fast4 = sum(1 for _, _, p in top
                if sorted(((e["raw_ratio_of_means"], n)
                           for n, e in p.items()))[4][1] in FAST)
    print()
    print(f"rows scoring at least 3.4: {len(top)}")
    print(f"  beagle sits at index 3 on {beagle3} of them "
          f"({beagle3/len(top)*100:.1f}%)")
    print(f"  one of the four fast prompts sits at index 4 on {fast4} "
          f"({fast4/len(top)*100:.1f}%)")
    counts = {}
    for _, _, p in top:
        pairs = sorted(((e["raw_ratio_of_means"], n) for n, e in p.items()))
        counts[pairs[4][1]] = counts.get(pairs[4][1], 0) + 1
    print("  index 4 occupancy:", dict(sorted(counts.items(),
                                              key=lambda kv: -kv[1])))

    print()
    print("=== cluster width of the four fast prompts, rows scoring >= 3.4 ===")
    spreads = []
    for _, _, p in top:
        fast = [p[n]["raw_ratio_of_means"] for n in FAST]
        spreads.append((max(fast) / min(fast) - 1) * 100)
    print(f"  n {len(spreads)}  min {min(spreads):.3f}%  "
          f"median {st.median(spreads):.3f}%  max {max(spreads):.3f}%")
    print("  the median is set by the MINIMUM of these four, so a wide cluster "
          "is a direct loss")

    print()
    print("=== us against the frontier, decomposed ===")
    ours = next(p for s, r, p in scored if r["id"].startswith("5a9f130a"))
    them = next(p for s, r, p in scored if r["id"].startswith("ec24d591"))
    print("prompt     our raw   their raw  d_raw%   d_cand%   d_serial%")
    for n in ("beagle",) + FAST:
        o, t = ours[n], them[n]
        d_raw = (t["raw_ratio_of_means"] / o["raw_ratio_of_means"] - 1) * 100
        d_c = (t["mtp_seconds_per_token_mean"]
               / o["mtp_seconds_per_token_mean"] - 1) * 100
        d_s = (t["serial_seconds_per_token_mean"]
               / o["serial_seconds_per_token_mean"] - 1) * 100
        print(f"{n:<10} {o['raw_ratio_of_means']:.5f}   "
              f"{t['raw_ratio_of_means']:.5f}  {d_raw:+7.3f}  {d_c:+7.3f}  "
              f"{d_s:+8.3f}")
    for label, prof in (("ours", ours), ("frontier", them)):
        fast = {n: prof[n]["raw_ratio_of_means"] for n in FAST}
        lo = min(fast, key=fast.get)
        print(f"  {label:<9} index4 = {lo} at {fast[lo]:.5f}, "
              f"four-prompt mean {st.mean(fast.values()):.5f}, "
              f"min is {(1 - fast[lo]/st.mean(fast.values()))*100:.3f}% "
              f"below that mean")

    o_fast = {n: ours[n]["raw_ratio_of_means"] for n in FAST}
    t_fast = {n: them[n]["raw_ratio_of_means"] for n in FAST}
    print(f"  frontier lead on the four-prompt MEAN: "
          f"{(st.mean(t_fast.values())/st.mean(o_fast.values())-1)*100:+.3f}%")
    print(f"  frontier lead on the four-prompt MIN:  "
          f"{(min(t_fast.values())/min(o_fast.values())-1)*100:+.3f}%")
    print("  the difference between those two lines is cluster luck, not speed")

    print()
    print("=== what our own row scores if essays merely ties our medicine ===")
    vals = sorted(e["raw_ratio_of_means"] for e in ours.values())
    base = 0.5 * (vals[3] + vals[4])
    lifted = sorted([e["raw_ratio_of_means"] for n, e in ours.items()
                     if n != "essays"] + [ours["medicine"]["raw_ratio_of_means"]])
    med = 0.5 * (lifted[3] + lifted[4])
    print(f"  {base:.6f} -> {med:.6f}   {(med/base-1)*100:+.4f}%   "
          f"frontier is 3.7291100105909")


if __name__ == "__main__":
    main()
