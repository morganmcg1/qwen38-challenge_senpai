#!/usr/bin/env python3
"""Is our fast serial leg session noise, box assignment, or tree-attributable?

Discriminators available per prompt:
  noop_reference_decode_speedup  -- organizer control, should be ~0.994 and
                                    tree-INDEPENDENT if it is a serial/serial ratio
  serial_seconds_per_token_mean  -- the scored numerator
Envelope may also name the runner box.
"""
import json
import os
import statistics as st
import urllib.request

NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
PIN = 0.037994794617407023


def get(sub_id):
    tok = os.environ.get("YUKON_API_TOKEN", "")
    req = urllib.request.Request("https://api.yukon.org/api/submissions/" + sub_id,
                                 headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d.get("submission", d)


rows = json.load(open("/tmp/rows_live.json"))
scored = [r for r in rows if isinstance(r.get("officialScore"), (int, float))]
scored.sort(key=lambda r: -r["officialScore"])
ours_row = [r for r in rows if r["id"].startswith("ca9251b8")][0]

# top 8 scored rows + ours
picks = [(r["id"], r["officialScore"]) for r in scored[:8]]
picks.append((ours_row["id"], ours_row["officialScore"]))

print("=" * 108)
print("ENVELOPE KEYS on one row (looking for a box / runner identifier)")
one = get(scored[0]["id"])
for k in sorted(one.keys()):
    v = one[k]
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = str(v)
        print(f"  {k:<34} {s[:64]}")
    else:
        print(f"  {k:<34} <{type(v).__name__} len={len(v)}>")
print()
om = one.get("officialMetrics", {})
print("officialMetrics top-level keys:")
for k in sorted(om.keys()):
    v = om[k]
    print(f"  {k:<34} {str(v)[:70] if not isinstance(v, (list, dict)) else '<'+type(v).__name__+'>'}")
print()
print("per_prompt[0] keys:", sorted(om["per_prompt"][0].keys()))

print()
print("=" * 108)
print(f"{'id':<10} {'score':>13} {'serial_mean':>13} {'vs_pin_%':>9} "
      f"{'serial_sd_%':>11} {'noop_mean':>11} {'noop_sd_%':>10} {'cand_mean':>12}")
tab = {}
for sid, sc in picks:
    s = get(sid)
    pp = {NAMES[p["prompt_sha256"][:8]]: p for p in s["officialMetrics"]["per_prompt"]}
    ser = [pp[n]["serial_seconds_per_token_mean"] for n in ORDER]
    noop = [pp[n]["noop_reference_decode_speedup"] for n in ORDER]
    cand = [pp[n]["mtp_seconds_per_token_mean"] for n in ORDER]
    m = st.mean(ser)
    tab[sid[:8]] = (pp, m, st.mean(noop))
    print(f"{sid[:8]:<10} {sc:>13.8f} {m:>13.8f} {100*(m/PIN-1):>+8.3f}% "
          f"{100*st.stdev(ser)/m:>10.3f}% {st.mean(noop):>11.6f} "
          f"{100*st.stdev(noop)/st.mean(noop):>9.3f}% {st.mean(cand):>12.8f}")

print()
print("=" * 108)
print("PER-PROMPT noop_reference_decode_speedup  (organizer control leg)")
hdr = f"{'id':<10}" + "".join(f"{n[:8]:>10}" for n in ORDER)
print(hdr)
for sid, _ in picks:
    pp = tab[sid[:8]][0]
    print(f"{sid[:8]:<10}" + "".join(f"{pp[n]['noop_reference_decode_speedup']:>10.5f}" for n in ORDER))

print()
print("PER-PROMPT serial_seconds_per_token_mean")
print(hdr)
for sid, _ in picks:
    pp = tab[sid[:8]][0]
    print(f"{sid[:8]:<10}" + "".join(f"{pp[n]['serial_seconds_per_token_mean']:>10.6f}" for n in ORDER))

print()
print("=" * 108)
print("CORRELATION: does a slow serial reading come with a proportionally slow noop?")
print("If noop == serialA/serialB of the same session it should be ~flat; if noop is")
print("serial/base_reference it should TRACK serial.")
xs, ys = [], []
for sid, _ in picks:
    pp = tab[sid[:8]][0]
    for n in ORDER:
        xs.append(pp[n]["serial_seconds_per_token_mean"])
        ys.append(pp[n]["noop_reference_decode_speedup"])
mx, my = st.mean(xs), st.mean(ys)
num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
print(f"  n={len(xs)}  pearson r = {num/den:+.4f}")
print(f"  serial CV = {100*st.stdev(xs)/mx:.3f}%   noop CV = {100*st.stdev(ys)/my:.3f}%")
