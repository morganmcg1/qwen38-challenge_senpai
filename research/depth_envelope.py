#!/usr/bin/env python3
"""What does the board's own data say about the optimal draft depth?

The schedule is a live control: on the organizer head `559b24eb` the board
shows beagle proposal depths from 0.04 to 5.94 proposals per round. If deeper
drafting paid at the current row price, the lower envelope of candidate time
against depth would still be falling at the deep end. This script draws that
envelope for the two prompts that set the published median.

    RECEIPT_CACHE=/tmp/board_live_t3.json python3 research/depth_envelope.py
"""
import json
import os
import statistics as st

CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/board_live_t3.json")
NAME = {"c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
        "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
        "a2ea8b60": "essays", "192fb621": "botany"}
HEAD = "559b24eb"
FOCUS = ("beagle", "essays", "medicine", "republic", "botany")


def main():
    payload = json.load(open(CACHE))
    rows = payload["submissions"] if isinstance(payload, dict) else payload

    obs = {name: [] for name in NAME.values()}
    for row in rows:
        metrics = row.get("officialMetrics") or {}
        for entry in metrics.get("per_prompt", []):
            if (entry.get("head_provenance_sha256") or "")[:8] != HEAD:
                continue
            if entry.get("non_drafting_round_count"):
                continue
            obs[NAME[entry["prompt_sha256"][:8]]].append({
                "edl": entry["effective_mean_draft_len"],
                "mtp": entry["mtp_seconds_per_token_mean"],
                "id": row["id"][:8],
                "user": row.get("solverUsername"),
                "score": row.get("officialScore"),
            })

    for name in FOCUS:
        pts = sorted(obs[name], key=lambda p: p["edl"])
        if not pts:
            continue
        print(f"=== {name}: head {HEAD}, fully drafting rows only, n = {len(pts)}")
        print("  edl bin   n     best mtp     median mtp   best row  solver           score")
        bins = {}
        for p in pts:
            bins.setdefault(round(p["edl"] * 4) / 4, []).append(p)
        best_overall = None
        for key in sorted(bins):
            group = bins[key]
            best = min(group, key=lambda p: p["mtp"])
            med = st.median([p["mtp"] for p in group])
            if best_overall is None or best["mtp"] < best_overall["mtp"]:
                best_overall = dict(best, bin=key)
            sc = best.get("score")
            sc = f"{float(sc):.5f}" if sc not in (None, "None") else "n/a"
            print(f"  {key:7.2f}  {len(group):>4}  {best['mtp']:.8f}  {med:.8f}  "
                  f"{best['id']}  {str(best['user'])[:15]:<15}  {sc}")
        print(f"  minimum at edl bin {best_overall['bin']:.2f}: "
              f"{best_overall['mtp']:.8f} by {best_overall['id']} "
              f"({best_overall['user']})")
        deep = [p for p in pts if p["edl"] >= best_overall["bin"] + 0.5]
        if deep:
            bd = min(deep, key=lambda p: p["mtp"])
            print(f"  deepest half a proposal above the minimum: n = {len(deep)}, "
                  f"best {bd['mtp']:.8f} at edl {bd['edl']:.4f} "
                  f"({bd['id']}, {bd['user']}), "
                  f"{(bd['mtp'] / best_overall['mtp'] - 1) * 100:+.3f}% vs the minimum")
        else:
            print("  no row on the board ever drafted half a proposal deeper")
        print()


if __name__ == "__main__":
    main()
