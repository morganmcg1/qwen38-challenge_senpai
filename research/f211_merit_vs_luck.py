#!/usr/bin/env python3
"""FINDING 211c -- is the crown a better candidate, or a luckier serial draw?

The serial leg is identical code in every row and no candidate edit can move
it.  Score each recent row on two separate axes:

  candidate axis   medpair candidate seconds per token, relative to the
                   window mean.  LOWER is genuinely better work.
  serial axis      medpair serial seconds per token, relative to the window
                   mean.  HIGHER is pure luck, because a slower baseline
                   inflates every raw ratio.

If the leader's advantage sits on the serial axis, the board ordering is a
lottery result and not a merit ordering.
"""
import json
import statistics

BOARD = "/tmp/yukon-board/full.json"
SINCE = "2026-08-22T18:00:00Z"
PROMPTS = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
MEDPAIR = ("beagle", "essays")

with open(BOARD) as fh:
    rows = json.load(fh)["submissions"]


def vec(row):
    pp = (row.get("officialMetrics") or {}).get("per_prompt")
    if not pp or len(pp) != 8:
        return None
    out = {}
    for e in pp:
        n = PROMPTS.get(e["prompt_sha256"][:8])
        if n is None or e.get("mtp_seconds_per_token_mean") is None:
            return None
        out[n] = e
    return out if len(out) == 8 else None


sel = []
for r in rows:
    v = vec(r)
    if v is None or (r.get("createdAt") or "") < SINCE:
        continue
    if v["beagle"]["effective_mean_draft_len"] < 3.0 or not r.get("officialScore"):
        continue
    sel.append((r, v))
sel.sort(key=lambda t: -t[0]["officialScore"])

cand_mp = {
    r["id"]: statistics.fmean([v[n]["mtp_seconds_per_token_mean"] for n in MEDPAIR])
    for r, v in sel
}
ser_mp = {
    r["id"]: statistics.fmean([v[n]["serial_seconds_per_token_mean"] for n in MEDPAIR])
    for r, v in sel
}
cbar = statistics.fmean(cand_mp.values())
sbar = statistics.fmean(ser_mp.values())

print(f"window since {SINCE}, rows {len(sel)}")
print(f"window mean medpair candidate s/tok  {cbar:.6f}")
print(f"window mean medpair serial    s/tok  {sbar:.6f}")
print()
hdr = (
    f"{'id':<10}{'solver':<15}{'score':>12}{'st':<5}"
    f"{'cand%':>9}{'serial%':>9}{'rank c':>8}{'rank s':>8}"
)
print(hdr)
print("-" * len(hdr))

# rank 1 = best on that axis
by_c = sorted(cand_mp, key=lambda k: cand_mp[k])
by_s = sorted(ser_mp, key=lambda k: -ser_mp[k])
rc = {k: i + 1 for i, k in enumerate(by_c)}
rs = {k: i + 1 for i, k in enumerate(by_s)}

for r, v in sel[:18]:
    i = r["id"]
    print(
        f"{i[:8]:<10}{(r.get('solverUsername') or '?')[:14]:<15}"
        f"{r['officialScore']:>12.6f}{(r.get('status') or '')[:4]:<5}"
        f"{100.0*(cand_mp[i]/cbar-1.0):>9.4f}"
        f"{100.0*(ser_mp[i]/sbar-1.0):>9.4f}{rc[i]:>8}{rs[i]:>8}"
    )

print()
print("cand% : negative is genuinely faster candidate work")
print("serial%: positive is a slower baseline, which is pure luck")
print()
top = sel[0][0]["id"]
print(
    f"leader {top[:8]}  candidate rank {rc[top]} of {len(sel)},"
    f"  serial-luck rank {rs[top]} of {len(sel)}"
)
