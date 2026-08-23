#!/usr/bin/env python3
"""Median sensitivity matrix for the ranked eight-prompt score.

The published score is the median of eight per-prompt ratios, not their mean.
Every arm in this campaign has been priced with a mean, which is correct only
when the arm's effect is uniform across prompts. It is not uniform for prefill
work (fixed cost, so the share falls as the round count rises) and it is not
uniform for depth-dependent work (duty cycle rises with the accepted draft
length).

This script decomposes one receipt's candidate leg into prefill and decode,
then reports the exact derivative of the published median with respect to
three independent levers, including the order-statistic switch points where
the median-setting pair changes.

    python3 research/median_sensitivity.py <receipt_id_prefix>
"""
import json
import os
import sys
import urllib.request

BENCHMARK = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/yukon_board.json")
NAME = {
    "c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
    "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
    "a2ea8b60": "essays", "192fb621": "botany",
}
TOKENS = 512
# FINDING 330: exact ranked round counts, recovered from the accept ledger.
ROUNDS = {"plutarch": 488, "drama": 252, "travel": 213, "beagle": 110,
          "republic": 93, "essays": 92, "medicine": 90, "botany": 81}


def load():
    if os.path.exists(CACHE):
        with open(CACHE) as handle:
            return json.load(handle)
    token = os.environ["YUKON_API_TOKEN"]
    url = (f"https://api.yukon.org/api/benchmarks/{BENCHMARK}/submissions"
           "?all=true")
    request = urllib.request.Request(url,
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    with open(CACHE, "w") as handle:
        json.dump(payload, handle)
    return payload


def median8(values):
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4]), ordered


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "5a9f130a"
    payload = load()
    rows = payload["submissions"] if isinstance(payload, dict) else payload
    row = next(r for r in rows if r["id"].startswith(prefix))

    legs = []
    for entry in row["officialMetrics"]["per_prompt"]:
        sha = entry["prompt_sha256"][:8]
        name = NAME.get(sha, sha)
        serial = entry["serial_seconds_per_token_mean"]
        cand = entry["mtp_seconds_per_token_mean"]
        legs.append({
            "name": name,
            "serial": serial,
            "cand": cand,
            "raw": entry["raw_ratio_of_means"],
            "rounds": ROUNDS[name],
            "edl": entry["effective_mean_draft_len"],
            "prefill": entry["prefill_seconds_per_token"] * TOKENS,
            "leg": cand * TOKENS,
        })

    print(f"receipt {row['id'][:8]}  solver {row.get('solverUsername')}  "
          f"official {row.get('officialScore')}")
    print()
    print("prompt      serial_spt   cand_spt    raw_p     leg_s   "
          "rounds  prefill%  R_ms")
    for leg in sorted(legs, key=lambda x: x["raw"]):
        rounds = leg["rounds"]
        pre_share = leg["prefill"] / leg["leg"]
        r_ms = (leg["leg"] - leg["prefill"]) / rounds * 1000.0
        print(f"{leg['name']:<10}  {leg['serial']:.7f}  {leg['cand']:.7f}  "
              f"{leg['raw']:.5f}  {leg['leg']:6.3f}  "
              f"{rounds:>6}  {pre_share*100:7.3f}  {r_ms:7.3f}")

    base_median, ordered = median8([leg["raw"] for leg in legs])
    names_by_raw = sorted(legs, key=lambda x: x["raw"])
    print()
    print(f"median = {base_median:.14g}   set by "
          f"{names_by_raw[3]['name']} ({names_by_raw[3]['raw']:.5f}) and "
          f"{names_by_raw[4]['name']} ({names_by_raw[4]['raw']:.5f})")

    def median_after(scale_fn):
        vals = []
        for leg in legs:
            new_leg = scale_fn(leg)
            vals.append(leg["serial"] * TOKENS / new_leg)
        return median8(vals)

    def uniform(frac):
        return lambda leg: leg["leg"] * (1.0 - frac)

    def prefill_only(frac):
        return lambda leg: leg["leg"] - leg["prefill"] * frac

    def decode_only(frac):
        return lambda leg: leg["prefill"] + (leg["leg"] - leg["prefill"]) * (1.0 - frac)

    def depth_weighted(frac):
        """A saving whose duty cycle rises with the accepted draft length.

        Normalised so the round-count-weighted mean saving equals `frac`, which
        makes it directly comparable with `decode_only` at the same `frac`.
        """
        total = sum(x["rounds"] * x["edl"] for x in legs)
        weight = sum(x["rounds"] for x in legs) / total

        def apply(leg):
            share = frac * leg["edl"] * weight
            return leg["prefill"] + (leg["leg"] - leg["prefill"]) * (1.0 - share)

        return apply

    print()
    print("=== lever sensitivity: published median gain per 1% resource saved ===")
    print("lever            1%      5%     10%     20%     40%   "
          " (published % gain)")
    for label, fn in (("uniform-leg", uniform),
                      ("prefill-only", prefill_only),
                      ("decode-only", decode_only),
                      ("depth-weighted", depth_weighted)):
        cells = []
        for frac in (0.01, 0.05, 0.10, 0.20, 0.40):
            med, _ = median_after(fn(frac))
            cells.append((med / base_median - 1.0) * 100.0)
        print(f"{label:<14} " + "  ".join(f"{c:6.3f}" for c in cells))

    print()
    print("=== per-1%-of-lever multiplier (published % per 1% of that lever) ===")
    for label, fn in (("uniform-leg", uniform),
                      ("prefill-only", prefill_only),
                      ("decode-only", decode_only),
                      ("depth-weighted", depth_weighted)):
        med, _ = median_after(fn(0.01))
        print(f"{label:<14} {(med / base_median - 1.0) * 100.0:.5f} "
              f"published % per 1% of lever")

    print()
    print("=== order-statistic stability: which pair sets the median ===")
    for label, fn in (("prefill-only", prefill_only),
                      ("decode-only", decode_only),
                      ("depth-weighted", depth_weighted)):
        print(f"  {label}:")
        for frac in (0.0, 0.10, 0.25, 0.50, 0.75, 1.00):
            vals = []
            for leg in legs:
                vals.append((leg["serial"] * TOKENS / fn(frac)(leg), leg["name"]))
            vals.sort()
            med = 0.5 * (vals[3][0] + vals[4][0])
            print(f"    {frac*100:5.0f}% saved -> median {med:.6f} "
                  f"set by {vals[3][1]} + {vals[4][1]}")

    print()
    print("=== marginal published weight of each prompt ===")
    print("  d(median)/median per 1% of that prompt's own raw ratio")
    for leg in names_by_raw:
        if leg is names_by_raw[3] or leg is names_by_raw[4]:
            w = 0.5 * leg["raw"] / base_median
        else:
            w = 0.0
        print(f"  {leg['name']:<10} raw {leg['raw']:.5f}  edl {leg['edl']:.4f}  "
              f"weight {w:.5f}")

    print()
    print("=== order-statistic headroom for a single-prompt arm ===")
    print("  how far one prompt can improve alone before its gain stops counting")
    upper = names_by_raw[5]["raw"]
    e = names_by_raw[4]
    b = names_by_raw[3]
    lower = names_by_raw[2]["raw"]
    cap_e = upper / e["raw"] - 1.0
    med_cap_e = 0.5 * (b["raw"] + upper)
    print(f"  {e['name']:<10} may gain {cap_e*100:6.3f}% before {names_by_raw[5]['name']} "
          f"takes the slot; median caps at {med_cap_e:.6f} "
          f"({(med_cap_e/base_median-1)*100:+.4f}%)")
    print(f"  {b['name']:<10} never saturates: passing {e['name']} only swaps the "
          f"two summed slots")
    print(f"  a gain on any other prompt is worth exactly 0 until it crosses "
          f"{b['name']} ({b['raw']:.5f}) from below "
          f"(nearest is {names_by_raw[2]['name']} at {lower:.5f}, "
          f"{(b['raw']/lower-1)*100:.3f}% away)")

    print()
    print("=== gap to frontier 3.7291100105909 ===")
    frontier = 3.7291100105909
    need = frontier / base_median - 1.0
    print(f"  need +{need*100:.4f}% published")
    for label, fn in (("uniform-leg", uniform),
                      ("prefill-only", prefill_only),
                      ("decode-only", decode_only),
                      ("depth-weighted", depth_weighted)):
        lo, hi = 0.0, 0.95
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            med, _ = median_after(fn(mid))
            if med < frontier:
                lo = mid
            else:
                hi = mid
        print(f"  {label:<14} requires {hi*100:7.3f}% of that lever removed")


if __name__ == "__main__":
    main()
