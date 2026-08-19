#!/usr/bin/env python3
"""Refresh the live promoted frontier and diff it against senpai/frontier-state.json.

`program.md` requires this comparison immediately before every official
submission: if the submission ID, source reference, or score differs from the
recorded campaign state, the organizer and promoted frontier must be synced,
the candidate replayed on the new base, and the measurement repeated.

Exits 0 when the frontier is unchanged and 3 when it has moved, so it can gate
a submission chain.

Schema note: the list endpoint returns {"submissions": [...]} and the fields
that matter are `officialScore`, `promotionStatus`, `promotedSourceRef` and
`solverUsername`. `officialMetrics` is null on failed rows, so it must be
guarded. Paging (`?page=N&limit=100`) returns 404; use `?all=true`.
"""
import json
import subprocess
import os
import sys

BENCH = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
URL = f"https://api.yukon.org/api/benchmarks/{BENCH}/submissions?all=true"

tok = os.environ["YUKON_API_TOKEN"]
raw = subprocess.run(
    ["curl", "-s", "-H", f"Authorization: Bearer {tok}", URL],
    capture_output=True, text=True, check=True).stdout
rows = json.loads(raw)["submissions"]
print(f"total submissions: {len(rows)}")


def sc(r):
    v = r.get("officialScore")
    return v if isinstance(v, (int, float)) else -1.0


promoted = [r for r in rows if r.get("promotionStatus") == "promoted"]
promoted.sort(key=sc, reverse=True)
print(f"promoted rows: {len(promoted)}")

print("\n--- top 6 promoted (the live frontier) ---")
for r in promoted[:6]:
    print(f"  {r['id'][:8]}  {sc(r):.14f}  {r['solverUsername']:<14}"
          f"  {str(r.get('promotedSourceRef'))[:12]}"
          f"  {str(r.get('promotionFinishedAt'))[:19]}")

frontier = promoted[0] if promoted else None

mine = [r for r in rows if r.get("solverUsername") == "morganmcg1"]
print(f"\n--- our submissions ({len(mine)}) ---")
for r in sorted(mine, key=sc, reverse=True):
    print(f"  {r['id'][:8]}  {sc(r):>18.14f}  {r.get('status'):<10}"
          f"  promo={str(r.get('promotionStatus')):<10}"
          f"  {str((r.get('officialMetrics') or {}).get('commit'))[:12]}")
    if r.get("rejectionReason"):
        print(f"      rejection: {str(r['rejectionReason'])[:110]}")

# Compare with recorded campaign state.
with open("senpai/frontier-state.json") as fh:
    state = json.load(fh)
rec = state.get("promotedSubmission", {})
print("\n--- recorded vs live ---")
print(f"  recorded id     {str(rec.get('id'))[:8]}   score {rec.get('score')}")
print(f"  recorded ref    {str(rec.get('sourceRef'))[:12]}")
if frontier:
    print(f"  live     id     {frontier['id'][:8]}   score {sc(frontier)}")
    print(f"  live     ref    {str(frontier.get('promotedSourceRef'))[:12]}")
    same = (str(rec.get("id")) == frontier["id"]
            and abs(float(rec.get("score", -1)) - sc(frontier)) < 1e-12
            and str(rec.get("sourceRef")) == str(frontier.get("promotedSourceRef")))
    print(f"\n  FRONTIER {'UNCHANGED' if same else 'MOVED -- resync required'}")
    if not same:
        sys.exit(3)
