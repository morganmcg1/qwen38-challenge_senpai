#!/usr/bin/env python3
"""Is the cap=4 evidence confounded by head artifact / pre-E27 vintage?

Every row I can find that ran sdpaWidthWallDepthCap=4 scores 2.68-2.93, while
cap=5 rows reach 3.23. E34 argues that comparison is void because all cap=4 rows
predate E27's move of the single-pass boundary from M=4 to M=5.

Test the confounder directly: which HEAD did each cap=4 row run? Only rows on the
declared head 559b24eb... are comparable to ours at all.
"""
import json
import re
from collections import Counter

rows = json.load(open("/tmp/rows_live.json"))

CAP4 = ["12b1c699", "f03469a9", "8df8616f", "40864c53", "adf015f1",
        "180edbaa", "81fe4a28", "fd14a471", "55fa8d31"]
CAP5_TOP = ["efe01dcf", "3ba27d91", "a1326b4b", "ca9251b8", "0cd0a6b4",
            "b0994092", "11863aa9", "de7981ae"]


def head_of(r):
    om = r.get("officialMetrics") or {}
    pp = om.get("per_prompt") or []
    hs = {p.get("head_provenance_sha256") for p in pp}
    hs.discard(None)
    if len(hs) == 1:
        return next(iter(hs))
    return f"<{len(hs)} distinct>" if hs else None


# what head does the top of the board run?
scored = [r for r in rows if isinstance(r.get("officialScore"), (int, float))]
scored.sort(key=lambda r: -r["officialScore"])
print("head artifact of the top 6 rows:")
for r in scored[:6]:
    h = head_of(r)
    print(f"  {r['id'][:8]}  {r['officialScore']:.6f}  head {str(h)[:24]}")

byhead = Counter()
best = {}
for r in scored:
    h = head_of(r)
    byhead[h] += 1
    if h not in best or r["officialScore"] > best[h]:
        best[h] = r["officialScore"]
print(f"\ndistinct heads among {len(scored)} scored rows: {len(byhead)}")
for h, c in byhead.most_common(8):
    print(f"  {str(h)[:24]:<26} n={c:4d}  best={best[h]:.6f}")

print()
print("=" * 104)
print("CAP=4 ROWS: head, date, score")
print(f"  {'id':<10} {'score':>11} {'created':<20} {'head':<26} user")
for pre in CAP4:
    m = [r for r in rows if r["id"].startswith(pre)]
    if not m:
        print(f"  {pre}  NOT FOUND")
        continue
    r = m[0]
    s = r.get("officialScore")
    ss = f"{s:.6f}" if isinstance(s, (int, float)) else "unscored"
    print(f"  {pre:<10} {ss:>11} {r['createdAt'][:19]:<20} "
          f"{str(head_of(r))[:24]:<26} {r.get('solverUsername','?')}")

print()
print("CAP=5 / TOP ROWS for comparison")
print(f"  {'id':<10} {'score':>11} {'created':<20} {'head':<26} user")
for pre in CAP5_TOP:
    m = [r for r in rows if r["id"].startswith(pre)]
    if not m:
        continue
    r = m[0]
    s = r.get("officialScore")
    ss = f"{s:.6f}" if isinstance(s, (int, float)) else "unscored"
    print(f"  {pre:<10} {ss:>11} {r['createdAt'][:19]:<20} "
          f"{str(head_of(r))[:24]:<26} {r.get('solverUsername','?')}")

print()
print("=" * 104)
print("VERDICT INPUTS")
declared = head_of([r for r in rows if r["id"].startswith("ca9251b8")][0])
print(f"  our declared head = {declared}")
cap4heads = []
for pre in CAP4:
    m = [r for r in rows if r["id"].startswith(pre)]
    if m:
        cap4heads.append((pre, head_of(m[0])))
same = [p for p, h in cap4heads if h == declared]
print(f"  cap=4 rows on the SAME head as us: {same if same else 'NONE'}")
print(f"  => the cap=4 vs cap=5 score comparison is "
      f"{'A VALID TEST' if same else 'CONFOUNDED BY HEAD ARTIFACT and cannot test E34'}")

print()
print("=" * 104)
print("mpjunior92 55fa8d31 correctness claim -- verify widths >=6 drift?")
r = [x for x in rows if x["id"].startswith("55fa8d31")][0]
n = r.get("note") or ""
i = n.lower().find("drift")
print("  note length", len(n), " score", r.get("officialScore"), " status", r["status"])
if i > 0:
    print("  ..." + n[max(0, i - 900):i + 900] + "...")
