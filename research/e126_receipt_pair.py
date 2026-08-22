#!/usr/bin/env python3
"""Compare two ranked receipts prompt by prompt, without any mode weighting.

    usage: research/e126_receipt_pair.py NEW_PREFIX OLD_PREFIX [--out PATH]

The published median is the mean of the fourth and fifth of eight sorted raw
ratios, so it discards six of the eight prompts and mixes in a fresh serial
draw. The F76 mode index keeps all eight but weights them, and that weighting
is confounded with any mechanism that acts only where drafting happens.

This tool does neither. It reports the per-prompt change in candidate seconds
per token, which is the only quantity a candidate edit can move, next to the
per-prompt change in serial seconds per token, which a candidate edit provably
cannot move. The serial column is the null: it shows what this pair of runs
would have differed by with no change of tree at all.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st

from e126_modeindex import PROMPT_NAMES, WEIGHTS

BOARD = pathlib.Path("/tmp/yukon-board/full.json")


def receipt(prefix: str) -> tuple[dict, dict]:
    rows = json.loads(BOARD.read_text())
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d rows" % (prefix, len(hits)))
    row = hits[0]
    per_prompt = {}
    for entry in (row.get("officialMetrics") or {}).get("per_prompt") or []:
        name = PROMPT_NAMES.get(str(entry.get("prompt_sha256", ""))[:8])
        if name:
            per_prompt[name] = entry
    return row, per_prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    new_row, new = receipt(args.new)
    old_row, old = receipt(args.old)

    print("harness=ranked. %s (%s, %.8f) against %s (%s, %.8f)"
          % (new_row["id"][:8], new_row.get("solverUsername"),
             float(new_row["officialScore"]), old_row["id"][:8],
             old_row.get("solverUsername"), float(old_row["officialScore"])))
    print("positive percent means the newer run was SLOWER\n")
    print("  %-9s %8s %9s %9s %9s" % ("prompt", "draftlen", "cand %",
                                      "serial %", "raw %"))

    order = sorted(new, key=lambda p: new[p]["effective_mean_draft_len"])
    cand, serial, drafting = [], [], []
    for p in order:
        c = 100.0 * (new[p]["mtp_seconds_per_token_mean"]
                     / old[p]["mtp_seconds_per_token_mean"] - 1.0)
        s = 100.0 * (new[p]["serial_seconds_per_token_mean"]
                     / old[p]["serial_seconds_per_token_mean"] - 1.0)
        r = 100.0 * (new[p]["raw_ratio_of_means"]
                     / old[p]["raw_ratio_of_means"] - 1.0)
        d = new[p]["effective_mean_draft_len"]
        print("  %-9s %8.3f %+9.3f %+9.3f %+9.3f" % (p, d, c, s, r))
        cand.append(c)
        serial.append(s)
        if d >= 1.0:
            drafting.append(c)

    print("\n  candidate leg, all eight      mean %+.3f %%, sd %.3f"
          % (st.fmean(cand), st.pstdev(cand)))
    print("  candidate leg, seven drafting mean %+.3f %%, sd %.3f"
          % (st.fmean(drafting), st.pstdev(drafting)))
    print("  serial leg, all eight         mean %+.3f %%, sd %.3f  <- the null"
          % (st.fmean(serial), st.pstdev(serial)))

    idx = sum(w * 100.0 * math.log(new[p]["mtp_seconds_per_token_mean"]
                                   / old[p]["mtp_seconds_per_token_mean"])
              for p, w in WEIGHTS.items())
    print("\n  the F76 index moves %+.4f units on this pair." % idx)
    print("  a mechanism confined to the seven drafting prompts moves it")
    print("  %+.4f units per 1 %% of speed, so %.0f %% of a 1.000 unit flip."
          % (-sum(w for p, w in WEIGHTS.items() if p != "plutarch"),
             100.0 * abs(sum(w for p, w in WEIGHTS.items() if p != "plutarch"))))

    if args.out:
        args.out.write_text(json.dumps({
            "harness": "ranked",
            "new": new_row["id"], "old": old_row["id"],
            "new_score": float(new_row["officialScore"]),
            "old_score": float(old_row["officialScore"]),
            "candidate_pct": dict(zip(order, cand)),
            "serial_pct": dict(zip(order, serial)),
            "candidate_drafting_mean_pct": st.fmean(drafting),
            "serial_mean_pct": st.fmean(serial),
            "f76_index_delta": idx,
        }, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
