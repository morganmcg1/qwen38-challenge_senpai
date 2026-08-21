#!/usr/bin/env python3
"""Seed-prefill host and GPU cost by host state, for an E85 traced session.

    usage: research/e85_seed_phase.py SESSION_DIR STATE_JSON

Each leg writes two `mtp-trace: begin` lines: one for the reference/serial
phase and one for the timed MTP phase. Both carry the 512-token seed
`build_us` (host graph build) and `eval_wall_us` (GPU wall). Prefill is
GPU-bound, so an invariant seed `eval_wall_us` across the two host states is
independent evidence that the state does not touch GPU compute.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

BEGIN = re.compile(r"mtp-trace: begin (.*)")
FIELD = re.compile(r"(\w+)=(-?[\d.]+)")


def main() -> None:
    root = Path(sys.argv[1])
    state = {l["leg"]: l["state"]
             for l in json.loads(Path(sys.argv[2]).read_text())["legs"]}

    rows = []
    for path in sorted(root.glob("leg*/rounds.txt")):
        leg = int(path.parent.name[3:5])
        begins = [{k: float(v) for k, v in FIELD.findall(m.group(1))}
                  for m in BEGIN.finditer(path.read_text(errors="replace"))]
        rows.append({"leg": leg, "state": state[leg], "begins": begins})

    print(f"{'leg':>3s} {'state':<5s} "
          + " ".join(f"{'seg' + str(i) + '_' + k:>14s}"
                     for i in (1, 2) for k in ("build_us", "eval_wall_us")))
    for r in rows:
        print(f"{r['leg']:3d} {r['state']:<5s} "
              + " ".join(f"{b[k]:14.0f}" for b in r["begins"]
                         for k in ("build_us", "eval_wall_us")))

    print()
    contrast = {}
    for index, name in ((0, "serial_reference"), (1, "timed_mtp")):
        for key in ("build_us", "eval_wall_us"):
            slow = [r["begins"][index][key] for r in rows if r["state"] == "slow"]
            fast = [r["begins"][index][key] for r in rows if r["state"] == "fast"]
            s, f = statistics.fmean(slow), statistics.fmean(fast)
            contrast[f"{name}_{key}"] = {"slow": s, "fast": f,
                                         "pct": 100.0 * (s - f) / f}
            print(f"  seed {name:<17s} {key:<13s} slow {s:11.0f}  "
                  f"fast {f:11.0f}  {100.0 * (s - f) / f:+7.3f} %")

    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(
            json.dumps({"contrast": contrast, "legs": rows}, indent=2) + "\n")
        print(f"\nwrote {sys.argv[3]}")


if __name__ == "__main__":
    main()
