#!/usr/bin/env python3
"""E102 helper: does the validated round reconstruction reproduce the crown table?

``research/board_same_schedule.py`` hard-codes eight crown round counts and
never says where they came from. ``research/prompt_round_reconstruction.py``
derives round counts from public receipt fields under constraints C1-C4 and
passes its own 7-check self-test. This script feeds the crown receipt
``8819b108`` into that reconstruction and compares, which is the positive
control the E102 pricing needs before it converts seconds per token into
microseconds per round.
"""

import json
import sys

sys.path.insert(0, "research")
import prompt_round_reconstruction as prr  # noqa: E402

PROMPTS = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
           "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
           "ea82dcb5": "republic", "3b10cb4d": "travel"}
CROWN = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
         "republic": 93, "essays": 92, "medicine": 90, "botany": 81}


def rows_for(row):
    out = {}
    for e in row["officialMetrics"]["per_prompt"]:
        out[PROMPTS[e["prompt_sha256"][:8]]] = {
            "mean_draft_len": e["effective_mean_draft_len"],
            "non_drafting_rounds": e["non_drafting_round_count"],
            "mtp_spt": e["mtp_seconds_per_token_mean"],
            "serial_spt": e["serial_seconds_per_token_mean"],
            "raw_ratio": e["raw_ratio_of_means"],
        }
    return out


def main():
    board = json.load(open("/tmp/yukon-board/full.json"))
    ident = sys.argv[1] if len(sys.argv) > 1 else "8819b108"
    hit = [r for r in board if r["id"].startswith(ident)][0]
    res = prr.reconstruct(rows_for(hit))
    print("receipt %s  %s  score %.5f"
          % (ident, hit.get("solverUsername"), hit["officialScore"]))
    print("calibration:", json.dumps(res["calibration"], indent=None))
    bad = 0
    print("  %-9s %6s %6s %8s %10s %10s" % ("prompt", "R", "crown", "unique",
                                            "resid", "feasible"))
    for name, p in res["prompts"].items():
        ok = CROWN.get(name) == p["rounds"]
        bad += 0 if ok else 1
        print("  %-9s %6d %6s %8s %+10.4f %10s%s"
              % (name, p["rounds"], CROWN.get(name, "-"),
                 p["unique_under_c1_c4"], p["rel_residual"],
                 len(p["feasible_round_counts"]), "" if ok else "  MISMATCH"))
    print("crown agreement: %d/8" % (8 - bad))


if __name__ == "__main__":
    main()
