#!/usr/bin/env python3
"""Recover the exact ranked accept ledger from the published per-prompt fields.

`effective_mean_draft_len` is proposals per round. Multiplying it by the round
count from FINDING 330 returns an exact integer for all eight prompts, which
confirms both the round table and the reading of the field. Accepted tokens
then follow from the 512-token window: every round emits one primary token plus
its accepted drafts, so accepted = 512 - rounds.

    python3 research/accept_ledger_exact.py [receipt_prefix]
"""
import json
import os
import sys

CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/yukon-board/full.json")
NAME = {"c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
        "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
        "a2ea8b60": "essays", "192fb621": "botany"}
ROUNDS = {"plutarch": 488, "drama": 252, "travel": 213, "beagle": 110,
          "republic": 93, "essays": 92, "medicine": 90, "botany": 81}
TOKENS = 512
PREFILL = 0.5274


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "5a9f130a"
    payload = json.load(open(CACHE))
    rows = payload["submissions"] if isinstance(payload, dict) else payload
    row = next(r for r in rows if r["id"].startswith(prefix))

    print(f"receipt {row['id'][:8]}  official {row.get('officialScore')}")
    print()
    print("prompt     rounds  edl      proposed  int?   accepted  accept%   "
          "M_mean  R_ms    ms/row")
    table = []
    for entry in row["officialMetrics"]["per_prompt"]:
        name = NAME[entry["prompt_sha256"][:8]]
        rounds = ROUNDS[name]
        edl = entry["effective_mean_draft_len"]
        proposed = edl * rounds
        accepted = TOKENS - rounds
        drafting = rounds - entry["non_drafting_round_count"]
        leg = entry["mtp_seconds_per_token_mean"] * TOKENS
        r_ms = (leg - PREFILL) / rounds * 1000.0
        table.append({
            "name": name, "rounds": rounds, "edl": edl, "proposed": proposed,
            "accepted": accepted, "rate": accepted / proposed,
            "m": 1.0 + proposed / drafting, "r_ms": r_ms,
            "raw": entry["raw_ratio_of_means"],
        })
    table.sort(key=lambda x: x["m"])
    prev = None
    for t in table:
        exact = abs(t["proposed"] - round(t["proposed"])) < 1e-6
        slope = ""
        if prev is not None and t["m"] > prev["m"] + 1e-9:
            slope = f"{(t['r_ms'] - prev['r_ms']) / (t['m'] - prev['m']):7.2f}"
        print(f"{t['name']:<9} {t['rounds']:>6}  {t['edl']:7.4f}  "
              f"{t['proposed']:8.3f}  {str(exact):<5}  {t['accepted']:>7}  "
              f"{t['rate']*100:6.2f}   {t['m']:6.3f}  {t['r_ms']:6.2f}  {slope}")
        prev = t

    print()
    print("M_mean is the mean verified row count on a DRAFTING round, "
          "1 + proposed/drafting_rounds.")
    print("ms/row is the marginal ranked round cost per verified row against "
          "the next prompt down.")
    print()
    tot_prop = sum(t["proposed"] for t in table)
    tot_acc = sum(t["accepted"] for t in table)
    tot_rounds = sum(t["rounds"] for t in table)
    print(f"campaign totals: rounds {tot_rounds}  proposed {tot_prop:.0f}  "
          f"accepted {tot_acc}  overall acceptance {tot_acc/tot_prop*100:.2f}%")

    print()
    print("=== value of one acceptance point on each prompt ===")
    print("  raising acceptance by 1pt at fixed proposal depth")
    base = sorted(t["raw"] for t in table)
    med0 = 0.5 * (base[3] + base[4])
    for t in table:
        acc_per_round = t["accepted"] / t["rounds"]
        new_acc = (t["rate"] + 0.01) * t["proposed"]
        new_rounds = TOKENS / (1.0 + new_acc / t["rounds"] * t["rounds"] / t["rounds"])
        new_rounds = TOKENS / (1.0 + new_acc / t["rounds"])
        old_leg = PREFILL + t["rounds"] * t["r_ms"] / 1000.0
        new_leg = PREFILL + new_rounds * t["r_ms"] / 1000.0
        new_raw = t["raw"] * old_leg / new_leg
        vals = sorted([x["raw"] for x in table if x is not t] + [new_raw])
        med = 0.5 * (vals[3] + vals[4])
        print(f"  {t['name']:<9} accept {t['rate']*100:6.2f} -> "
              f"{(t['rate']+0.01)*100:6.2f}  raw {t['raw']:.4f} -> {new_raw:.4f}  "
              f"median {(med/med0-1)*100:+7.4f}%")


if __name__ == "__main__":
    main()
