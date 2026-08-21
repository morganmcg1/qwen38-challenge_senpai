#!/usr/bin/env python3
"""Separate the environmental and mechanistic parts of a published-score gap.

``board_per_prompt.py serialfree`` normalises every candidate by the per-prompt
board-mean serial time.  A submission whose published score sits above its
serial-free score met its own serial draws that were slower than the board mean;
a submission whose published score sits below met its own serial draws that were
faster.  That difference is environmental: it does not reflect the candidate's
own code.  This script prints the per-prompt own-serial excess so the
environmental part can be separated from a measured candidate regression.

Usage:
    python3 research/e87_s13_serial_control.py <uuid-prefix> [...]
"""
import statistics as st
import sys

from board_per_prompt import load, vec, serial_means

ORDER = ["botany", "medicine", "republic", "essays", "beagle", "travel",
         "drama", "plutarch"]


def main(prefixes):
    scored = load()
    means = serial_means(scored)
    print("%d scored submissions" % len(scored))
    print()
    print("board-mean serial seconds/token")
    for name in ORDER:
        print("  %-9s %.8f" % (name, means[name]))
    print()

    for want in prefixes:
        hit = [r for r in scored if r["id"].startswith(want)]
        if not hit:
            print("%s: not in the scored set" % want)
            continue
        row = hit[0]
        v = vec(row)
        print("=== %s  published %.8f  status %s"
              % (row["id"][:8], row["officialScore"], row.get("status")))
        print("%-9s %12s %12s %11s %12s %11s"
              % ("prompt", "own serial", "board mean", "own excess%",
                 "own mtp", "raw ratio"))
        excess = []
        for name in ORDER:
            e = v[name]
            own = e["serial_seconds_per_token_mean"]
            bm = means[name]
            pct = 100.0 * (own / bm - 1.0)
            excess.append(pct)
            print("%-9s %12.8f %12.8f %+11.3f %12.8f %11.6f"
                  % (name, own, bm, pct, e["mtp_seconds_per_token_mean"],
                     e["raw_ratio_of_means"]))
        print("  mean own-serial excess   %+.3f %%" % st.fmean(excess))
        print("  median own-serial excess %+.3f %%" % st.median(excess))
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["cb8aeefb", "84b9ef7b", "8819b108"])
