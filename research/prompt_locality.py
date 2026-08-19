#!/usr/bin/env python3
"""Do we hold any of the eight RANKED prompts locally?

The API exposes prompt_sha256 per prompt. Hash every plausible local prompt/prose
file (raw bytes, and a few normalisations) and compare against the eight known
prefixes. A hit means we can trace the exact ranked prompt locally.
"""
import glob
import hashlib
import json
import os

WANT = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# full sha256 values, from the live board row
rows = json.load(open("/tmp/rows_live.json"))
r = [x for x in rows if x["id"].startswith("ca9251b8")][0]
full = {}
for p in r["officialMetrics"]["per_prompt"]:
    full[p["prompt_sha256"]] = WANT[p["prompt_sha256"][:8]]
print("ranked prompt sha256 (full):")
for h, n in sorted(full.items(), key=lambda t: t[1]):
    print(f"  {n:<9} {h}")

pats = ["research/*.txt", "research/*.json", "fixtures/*.json", "fixtures/*.txt",
        "research/**/*.txt", "correctness_prompts/**/*", "*.txt"]
files = []
for p in pats:
    files += glob.glob(p, recursive=True)
files = sorted({f for f in files if os.path.isfile(f) and os.path.getsize(f) < 4_000_000})
print(f"\nscanning {len(files)} local files")

variants = {
    "raw": lambda b: b,
    "strip": lambda b: b.strip(),
    "strip+nl": lambda b: b.strip() + b"\n",
    "no_trailing_nl": lambda b: b.rstrip(b"\n"),
}

hits = []
for f in files:
    try:
        b = open(f, "rb").read()
    except Exception:
        continue
    for vn, fn in variants.items():
        h = hashlib.sha256(fn(b)).hexdigest()
        if h in full:
            hits.append((f, vn, full[h]))
        elif h[:8] in WANT:
            hits.append((f, vn, "PREFIX-ONLY " + WANT[h[:8]]))

# also try JSON string fields inside json files
for f in [x for x in files if x.endswith(".json")]:
    try:
        d = json.load(open(f))
    except Exception:
        continue

    def walk(o, path=""):
        if isinstance(o, str):
            for vn, fn in variants.items():
                h = hashlib.sha256(fn(o.encode())).hexdigest()
                if h in full:
                    hits.append((f + "::" + path, vn, full[h]))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o[:400]):
                walk(v, path + f"[{i}]")

    walk(d)

print("\nMATCHES:")
if hits:
    for f, vn, n in hits:
        print(f"  {n:<24} {vn:<16} {f}")
else:
    print("  NONE - no ranked prompt text is present locally in any form tried.")

print("\nlocal prose-seed candidates (for proxy tracing):")
for f in files:
    bn = os.path.basename(f)
    if "prose" in bn or "prompt" in bn or "512" in bn:
        print(f"  {os.path.getsize(f):>9d}  {f}")
