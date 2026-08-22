"""Diagnose receipt cf9a9eda against our own fast receipts and the crown."""
import json
import math
import sys

BOARD = "/tmp/yukon-board/full.json"
SHA = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
       "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
       "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]
W83 = {"beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
       "botany": 0.0124, "republic": 0.0100, "plutarch": 0.0,
       "drama": 0.0, "travel": 0.0}

rows = json.load(open(BOARD))["submissions"]
by = {}
for r in rows:
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp:
        continue
    d = {}
    for e in pp:
        nm = SHA.get((e.get("prompt_sha256") or "")[:8])
        if nm:
            d[nm] = e
    if len(d) == 8:
        by[r["id"][:8]] = {"p": d, "score": r.get("officialScore"),
                           "solver": r.get("solverName") or "?",
                           "at": (r.get("resolvedAt") or "")[:19]}

TARGET = "cf9a9eda"
REFS = ["44559d02", "b8b8b860", "bc070b7b", "f04b102e", "7bef7d4c"]

t = by[TARGET]
print("TARGET %s  published %.8f  %s" % (TARGET, t["score"], t["at"]))
print()
print("per-prompt candidate ms/token, and the ratio published by the runner")
print("prompt     cand_ms   serial_ms  raw_ratio   depth   parity")
for nm in ORDER:
    e = t["p"][nm]
    print("  %-9s %8.4f  %9.4f   %8.5f  %6.3f   %s"
          % (nm, e["mtp_seconds_per_token_mean"] * 1000,
             e["serial_seconds_per_token_mean"] * 1000,
             e["raw_ratio_of_means"], e["effective_mean_draft_len"],
             e.get("parity_ok")))
print()

print("=" * 78)
print("candidate ms/token vs each reference, %% change (positive = SLOWER)")
print()
hdr = "  prompt      w83   " + "".join("%11s" % r for r in REFS)
print(hdr)
for nm in ORDER:
    a = t["p"][nm]["mtp_seconds_per_token_mean"]
    cells = []
    for R in REFS:
        b = by[R]["p"][nm]["mtp_seconds_per_token_mean"]
        cells.append("%+10.3f%%" % (100.0 * (a / b - 1.0)))
    print("  %-9s %.4f  %s" % (nm, W83[nm], "".join(cells)))
print()
print("  %-9s %6s  %s" % ("SERIAL leg", "", ""))
for nm in ORDER:
    a = t["p"][nm]["serial_seconds_per_token_mean"]
    cells = []
    for R in REFS:
        b = by[R]["p"][nm]["serial_seconds_per_token_mean"]
        cells.append("%+10.3f%%" % (100.0 * (a / b - 1.0)))
    print("  %-9s        %s" % (nm, "".join(cells)))
print()

print("=" * 78)
print("realised draft depth, target vs references")
print()
print("  prompt     " + "".join("%11s" % r for r in [TARGET] + REFS))
for nm in ORDER:
    cells = ["%11.3f" % t["p"][nm]["effective_mean_draft_len"]]
    for R in REFS:
        cells.append("%11.3f" % by[R]["p"][nm]["effective_mean_draft_len"])
    print("  %-9s %s" % (nm, "".join(cells)))
print()

print("=" * 78)
print("other published per-prompt fields, target vs 44559d02")
print()
keys = [k for k in sorted(t["p"]["beagle"]) if k not in
        ("prompt_sha256", "head_provenance_sha256")]
for k in keys:
    a = t["p"]["beagle"].get(k)
    b = by["44559d02"]["p"]["beagle"].get(k)
    print("  beagle  %-38s  %-22r  %-22r" % (k, a, b))
print()
print("  head_provenance_sha256")
print("    target    %s" % t["p"]["beagle"].get("head_provenance_sha256"))
for R in REFS:
    print("    %-9s %s" % (R, by[R]["p"]["beagle"].get(
        "head_provenance_sha256")))
