"""E146: reconcile the headline with Ledger 305 finding by finding.

harness=ranked. The advisor published F226-REVISED with a 903.0 us step and a
decode-only frame. This script prints the same quantity at both steps and in
both frames so the two records can be compared without re-deriving anything.
"""

import json
import os
import sys

import e146_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-ledger305.json")
ANCHOR = "572b2cc4"
ROW = "1db9d63e"
ADVISOR_DECODE_PCT = -0.5319
STEPS = [
    ("advisor F226-REVISED", 903.0),
    ("E146 zero-delta only", 879.0038085693578),
    ("F152/F172 wired residency", 930.9),
]


def main():
    _, rows = L.load()
    by_id = {r.id8: r for r in rows}
    anchor, row = by_id[ANCHOR], by_id[ROW]

    raw_decode = L.mean(L.pct_diff(anchor, row, field="decode").values())
    raw_total = L.mean(L.pct_diff(anchor, row, field="total").values())

    print("%-28s %12s %12s" % ("step source", "decode %", "total leg %"))
    print("%-28s %12.4f %12.4f" % ("raw, no correction", raw_decode, raw_total))
    table = []
    for name, step in STEPS:
        decode = raw_decode - L.state_pct(anchor, row, step, field="decode")
        total = raw_total - L.state_pct(anchor, row, step, field="total")
        table.append({"step_source": name, "step_us": step,
                      "corrected_decode_pct": decode,
                      "corrected_total_pct": total})
        print("%-28s %12.4f %12.4f" % (name, decode, total))
    print("\nadvisor Ledger 305 all-eight decode-only state-corrected: %.4f"
          % ADVISOR_DECODE_PCT)
    replay = table[0]["corrected_decode_pct"]
    print("E146 replay of the same step, decode frame:                %.4f" % replay)
    print("agreement:                                                 %.4f pp"
          % (replay - ADVISOR_DECODE_PCT))
    print("\nThe advisor's decode figure and the E146 decode figure differ only")
    print("by the step. The total-leg figure is the one the score prices,")
    print("because the published candidate mean already contains the seed")
    print("prefill (F227) and the state does not touch the prefill (F228).")

    lo = min(e["corrected_total_pct"] for e in table)
    hi = max(e["corrected_total_pct"] for e in table)
    with open(OUT, "w") as handle:
        json.dump({
            "harness": "ranked",
            "anchor": ANCHOR,
            "row": ROW,
            "raw_decode_pct": raw_decode,
            "raw_total_pct": raw_total,
            "steps": table,
            "advisor_decode_pct": ADVISOR_DECODE_PCT,
            "e146_replay_of_advisor_step_decode_pct": replay,
            "e146_advisor_agreement_pp": replay - ADVISOR_DECODE_PCT,
            "e146_headline_step_sensitivity_total_low_pct": lo,
            "e146_headline_step_sensitivity_total_high_pct": hi,
        }, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
