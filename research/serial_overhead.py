#!/usr/bin/env python3
"""Compare the fixed (non-block) overhead of the serial control and MTP legs."""
from __future__ import annotations

import glob
import json

RUNS = {
    "I": "research/capture-runI",
    "J": "research/capture-runJ-gate2-512",
    "K": "research/capture-runK-gate2-cap8-512",
    "L": "research/capture-runL-gate1-cap8-512",
    "M": "research/capture-runM-gate0-cap8-512",
    "N": "research/capture-runN-gate1-cap8-512-confirm",
}

rows = []
for lab, dr in RUNS.items():
    for path in sorted(glob.glob(dr + "/0*.json")):
        try:
            d = json.load(open(path))
        except json.JSONDecodeError:
            continue
        if "decode_seconds" not in d:
            continue
        bl = d.get("block_request_seconds") or []
        tot = sum(bl)
        rows.append(
            {
                "run": lab,
                "phase": path.rsplit("/", 1)[-1],
                "serial_control": bool(d.get("is_serial_control")),
                "depth": d.get("mtp_depth"),
                "tokens": d.get("decode_token_count") or d.get("emitted_token_total"),
                "rounds": len(bl),
                "decode_s": d["decode_seconds"],
                "blocks_s": tot,
                "gap_s": d["decode_seconds"] - tot,
                "s_per_token": d.get("parent_measured_seconds_per_token"),
            }
        )

hdr = (
    "%-4s %-20s %-7s %-6s %-7s %-7s %10s %10s %9s %12s"
    % ("run", "phase", "serial", "depth", "tokens", "rounds", "decode_s", "blocks_s", "gap_s", "s/token")
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        "%-4s %-20s %-7s %-6s %-7s %-7s %10.4f %10.4f %9.4f %12.8f"
        % (
            r["run"],
            r["phase"],
            r["serial_control"],
            r["depth"],
            r["tokens"],
            r["rounds"],
            r["decode_s"],
            r["blocks_s"],
            r["gap_s"],
            r["s_per_token"] or 0.0,
        )
    )

print()
ser = [r for r in rows if r["serial_control"]]
mtp = [r for r in rows if not r["serial_control"] and r["phase"].startswith("04")]
if ser and mtp:
    gs = sum(r["gap_s"] for r in ser) / len(ser)
    gm = sum(r["gap_s"] for r in mtp) / len(mtp)
    bs = sum(r["blocks_s"] for r in ser) / len(ser)
    bm = sum(r["blocks_s"] for r in mtp) / len(mtp)
    print("mean serial gap = %.4f s, mean MTP gap = %.4f s" % (gs, gm))
    print("mean serial blocks = %.4f s, mean MTP blocks = %.4f s" % (bs, bm))
    print("ratio as measured        = %.4f" % ((bs + gs) / (bm + gm)))
    print("ratio with gap removed   = %.4f" % (bs / bm))

json.dump(rows, open("research/serial-overhead.json", "w"), indent=2)
print("\nwrote research/serial-overhead.json")
