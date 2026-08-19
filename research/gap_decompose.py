#!/usr/bin/env python3
"""Decompose the score gap to #1 into (a) candidate decode speed and (b) serial-reading luck.

raw_p = mean(serial_s_per_tok) / mean(candidate_s_per_tok)

A rival can lead either because their candidate is faster (engineerable) or
because their *serial* leg happened to read slow (pure noise, not engineerable).
Item 117(c): the organizers state all serial readings are interchangeable at
~0.03800 s/tok because the depth-0 leg does prompt-independent work.
"""
import json
import os
import urllib.request

NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
PINNED_SERIAL = 0.037994794617407023  # fixtures: serial_calibration


def get(sub_id):
    tok = os.environ.get("YUKON_API_TOKEN", "")
    url = "https://api.yukon.org/api/submissions/" + sub_id
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d.get("submission", d)


def bysha(sub):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in sub["officialMetrics"]["per_prompt"]}


def median_even(vals):
    v = sorted(vals)
    return (v[3] + v[4]) / 2.0


rows = json.load(open("/tmp/rows_live.json"))
scored = [r for r in rows if isinstance(r.get("officialScore"), (int, float))]
scored.sort(key=lambda r: -r["officialScore"])

OURS = "ca9251b8"
ids = [scored[0]["id"], scored[1]["id"], scored[3]["id"]]
subs = {}
for i in ids:
    subs[i[:8]] = bysha(get(i))
subs[OURS] = bysha(get([r for r in rows if r["id"].startswith(OURS)][0]["id"]))

order = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]

print("=" * 100)
print("IDENTITY CHECK: score == mean of 4th and 5th order statistics of raw_ratio_of_means")
for k in [OURS] + [i[:8] for i in ids]:
    pp = subs[k]
    raws = [pp[n]["raw_ratio_of_means"] for n in order]
    ident = median_even(raws)
    sc = [r["officialScore"] for r in rows if r["id"].startswith(k)][0]
    v = sorted(raws)
    which = [n for n in order if pp[n]["raw_ratio_of_means"] in (v[3], v[4])]
    print(f"  {k}  reported={sc!r}  identity={ident!r}  delta={sc-ident:+.3e}  central_pair={which}")

print()
print("=" * 100)
print("PER-PROMPT DECOMPOSITION vs each rival (positive delta_pct = rival better than us)")
ours = subs[OURS]
for k in [i[:8] for i in ids]:
    pp = subs[k]
    print(f"\n--- rival {k} ---")
    print(f"{'prompt':<10} {'our_serial':>12} {'riv_serial':>12} {'serial_%':>9} "
          f"{'our_cand':>11} {'riv_cand':>11} {'cand_%':>9} {'raw_%':>9}")
    tot_s = tot_c = 0.0
    for n in order:
        os_ = ours[n]["serial_seconds_per_token_mean"]
        rs = pp[n]["serial_seconds_per_token_mean"]
        oc = ours[n]["mtp_seconds_per_token_mean"]
        rc = pp[n]["mtp_seconds_per_token_mean"]
        # rival raw advantage = (rs/rc) / (os/oc) - 1
        s_adv = rs / os_ - 1.0          # rival's serial read slower -> free raw gain
        c_adv = oc / rc - 1.0            # rival's candidate faster -> earned raw gain
        raw_adv = pp[n]["raw_ratio_of_means"] / ours[n]["raw_ratio_of_means"] - 1.0
        tot_s += s_adv
        tot_c += c_adv
        print(f"{n:<10} {os_:>12.8f} {rs:>12.8f} {100*s_adv:>+8.3f}% "
              f"{oc:>11.8f} {rc:>11.8f} {100*c_adv:>+8.3f}% {100*raw_adv:>+8.3f}%")
    print(f"{'MEAN':<10} {'':>12} {'':>12} {100*tot_s/8:>+8.3f}% "
          f"{'':>11} {'':>11} {100*tot_c/8:>+8.3f}%")

print()
print("=" * 100)
print("COUNTERFACTUAL: our score if our candidate were unchanged but every serial")
print("reading equalled the fixture's pinned calibration %.18f" % PINNED_SERIAL)
for k in [OURS] + [i[:8] for i in ids]:
    pp = subs[k]
    real = median_even([pp[n]["raw_ratio_of_means"] for n in order])
    norm = median_even([PINNED_SERIAL / pp[n]["mtp_seconds_per_token_mean"] for n in order])
    print(f"  {k}  as_scored={real:.8f}   serial_normalised={norm:.8f}   luck={100*(real/norm-1):+.4f}%")

print()
print("=" * 100)
print("HOW MUCH BEAGLE DO WE NEED? (all other prompts held fixed, as-scored)")
raws = {n: ours[n]["raw_ratio_of_means"] for n in order}
for k in [i[:8] for i in ids]:
    target = [r["officialScore"] for r in rows if r["id"].startswith(k)][0]
    lo, hi = 1.0, 1.20
    for _ in range(200):
        mid = (lo + hi) / 2
        trial = dict(raws)
        trial["beagle"] = raws["beagle"] * mid
        if median_even(list(trial.values())) < target:
            lo = mid
        else:
            hi = mid
    ok = median_even([raws["beagle"] * hi if n == "beagle" else raws[n] for n in order]) >= target - 1e-12
    print(f"  to reach {k} ({target:.8f}): beagle x{hi:.6f} = {100*(hi-1):+.3f}%  reachable={ok}")
