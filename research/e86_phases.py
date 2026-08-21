#!/usr/bin/env python3
"""E86: per-leg and per-arm medians of every traced round phase.

usage: research/e86_phases.py PREFIX [PREFIX ...] [--arm NAME]

Two questions need this table.

  1. The E86 position confound. The verify pipeline
     (`verify_build_us + eval_wall_us`) is GPU bound and barely moves with leg
     position, but the HOST phases outside it (`d_pre`, `d_flush`, `d_head1`,
     `d_submit1`, `d_chain`, `readout`, `commit`, `upkeep`) cost about
     760-820 us/round in an interior leg and several times that in a leg at
     position 0 or at the last position. Printing the host sum per leg makes
     that contamination visible before any arm claim is made.

  2. The head-path host share. `d_submit2_us` covers the second head submit.
     With MLX_QWEN_MTP_TRACE_SYNC_HEAD=1 the head chain is drained inside that
     window, so the counter holds host encode PLUS head GPU execute. Without
     it the head chain overlaps the verify window, so the counter is close to
     pure host encode. The difference between the two modes bounds how much of
     the head path is host work, which is what decides whether the verify-path
     law ("host-side removals cannot pay") also applies to the head path.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
        "readout_us", "commit_us", "upkeep_us"]
PIPE = ["draft_build_us", "d_submit2_us", "verify_build_us", "eval_wall_us"]


def read_meta(path: Path) -> dict:
    return dict(line.partition("=")[::2] for line in path.read_text().splitlines()
                if "=" in line)


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if m:
            out.append({k: float(v) for k, v in KV_RE.findall(m.group(4))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--arm")
    args = ap.parse_args()

    cols = PIPE + ["HOSTSUM"] + HOST + ["round_us"]
    head = f"{'leg':<26}{'sync':>5}"
    for c in cols:
        head += f"{c.replace('_us', ''):>14}"
    print(head)

    per_arm: dict[tuple[str, str], list[dict]] = {}
    for prefix in args.prefixes:
        tags = sorted(p.name for p in OUT.iterdir()
                      if p.name.startswith(prefix + "-") and (p / "trace.txt").exists())
        for t in tags:
            arm = t[len(prefix) + 1:].rpartition("-")[0]
            if args.arm and arm != args.arm:
                continue
            meta = read_meta(OUT / t / "meta.txt")
            sync = meta.get("sync_head", "?")
            rs = rounds(t)
            for r in rs:
                r["HOSTSUM"] = sum(r[k] for k in HOST)
            per_arm.setdefault((prefix + ":" + arm, sync), []).extend(rs)
            line = f"{t:<26}{sync:>5}"
            for c in cols:
                line += f"{st.median([r[c] for r in rs]):>14.0f}"
            print(line)

    print()
    for (key, sync), rs in sorted(per_arm.items()):
        line = f"{key:<26}{sync:>5}"
        for c in cols:
            line += f"{st.median([r[c] for r in rs]):>14.0f}"
        print(line)


if __name__ == "__main__":
    main()
