#!/usr/bin/env python3
"""Ranked test of E34's central claim: is sdpaWidthWallDepthCap 5 -> 4 worth +16.3%?

E34 predicts the central pair rises 3.24929 -> 3.7786 (+16.3%) from moving one
constant, with an honest interval [3.5498, 4.0073] that excludes the status quo.
On a board whose top 20 rows span 0.5%, a +16% lever from a single integer should
be impossible to miss: 48 notes mention sdpaWidthWallDepthCap.

So: find every row whose note discusses the width cap, extract what it did, and
read the score. This is the same method that killed E36 (ledger 120).
"""
import json
import re
from collections import Counter

rows = json.load(open("/tmp/rows_live.json"))
scored = [r for r in rows if isinstance(r.get("officialScore"), (int, float))]
allrows = rows
print(f"rows total {len(allrows)}  scored {len(scored)}")

TERMS = ["sdpaWidthWallDepthCap", "widthCap", "width wall", "segmentedVerifyDepthCap",
         "segmentedStreakGate", "fullAcceptStreak", "depth cap", "depthCap"]
hits = Counter()
for r in allrows:
    n = r.get("note") or ""
    for t in TERMS:
        if t.lower() in n.lower():
            hits[t] += 1
print("\nnote mentions:")
for t in TERMS:
    print(f"  {t:<26} {hits[t]}")

# Find rows that state a NUMERIC cap value
pat = re.compile(
    r"(sdpaWidthWallDepthCap|segmentedVerifyDepthCap|segmentedStreakGate)"
    r"[^0-9\n]{0,40}?(\d)", re.I)
found = []
for r in allrows:
    n = r.get("note") or ""
    ms = pat.findall(n)
    if ms:
        found.append((r, ms))
print(f"\nrows naming a numeric value for one of those constants: {len(found)}")

print()
print("=" * 112)
print("ROWS THAT NAME A CAP VALUE (sorted by score desc)")


def key(t):
    s = t[0].get("officialScore")
    return -s if isinstance(s, (int, float)) else 1e9


found.sort(key=key)
for r, ms in found[:45]:
    s = r.get("officialScore")
    ss = f"{s:.6f}" if isinstance(s, (int, float)) else f"{r.get('status')}"
    vals = ", ".join(f"{a}={b}" for a, b in ms[:6])
    print(f"  {ss:>10}  {r['status']:<11} {r['id'][:8]}  {r.get('solverUsername','?'):<16} {vals}")

print()
print("=" * 112)
print("EXPLICIT 'cap = 4' / 'cap to 4' / '5 -> 4' EVIDENCE, with note excerpts")
pat4 = re.compile(
    r"(sdpaWidthWallDepthCap\s*(?:=|to|:|->)\s*4"
    r"|width\s*(?:wall\s*)?cap\s*(?:=|to|:|->)\s*4"
    r"|cap\s*(?:from\s*)?5\s*(?:->|to|→)\s*4"
    r"|sdpaWidthWallDepthCap\s*5\s*(?:->|to|→)\s*4)", re.I)
n4 = 0
for r in allrows:
    n = r.get("note") or ""
    m = pat4.search(n)
    if not m:
        continue
    n4 += 1
    s = r.get("officialScore")
    ss = f"{s:.8f}" if isinstance(s, (int, float)) else "unscored"
    print(f"\n  --- {r['id'][:8]}  score {ss}  status {r['status']}"
          f"  user {r.get('solverUsername','?')}")
    i = max(0, m.start() - 420)
    print("      ..." + n[i:m.end() + 620].replace("\n", "\n      ") + "...")
print(f"\n  rows explicitly moving the cap to 4: {n4}")

print()
print("=" * 112)
print("Any note mentioning M=5 single pass / M=6 two passes / the pass boundary")
pat5 = re.compile(r"(M\s*=\s*5[^.\n]{0,60}(single|one)\s*pass"
                  r"|(single|one)\s*pass[^.\n]{0,60}M\s*=\s*5"
                  r"|M\s*=\s*6[^.\n]{0,60}(two|2)\s*pass)", re.I)
c = 0
for r in allrows:
    n = r.get("note") or ""
    m = pat5.search(n)
    if not m:
        continue
    c += 1
    s = r.get("officialScore")
    ss = f"{s:.8f}" if isinstance(s, (int, float)) else "unscored"
    if c <= 8:
        print(f"\n  --- {r['id'][:8]}  score {ss}  {r['status']}  {r.get('solverUsername','?')}")
        i = max(0, m.start() - 260)
        print("      ..." + n[i:m.end() + 380].replace("\n", "\n      ") + "...")
print(f"\n  rows discussing the pass boundary: {c}")
