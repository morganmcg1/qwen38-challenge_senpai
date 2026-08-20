#!/usr/bin/env python3
"""Query one E65 census JSON: slowest rounds, a named round, or a cell.

usage:
  research/e65_round_query.py CENSUS.json [--session N] [--top N]
                              [--round R ...] [--cell NAME] [--tail N]
"""
import argparse
import json


COLS = ("round", "M", "cell", "kL_verify", "repaired", "rejected",
        "round_us", "draft_build", "verify_build", "eval_wall", "readout",
        "commit", "upkeep")


def show(rows, title):
    print(f"--- {title} ({len(rows)}) ---")
    print("  rnd  M  cell          kL_ver  round_ms  dbuild  vbuild  evwall "
          " rdout  commit  upkeep")
    for r in rows:
        print(f"  {r['round']:4d} {r['M']:2d} {r['cell']:12s} "
              f"{r['kL_verify']:6d} {r['round_us'] / 1000:9.3f} "
              f"{r['draft_build'] / 1000:7.3f} {r['verify_build'] / 1000:7.3f} "
              f"{r['eval_wall'] / 1000:7.3f} {r['readout'] / 1000:6.3f} "
              f"{r['commit'] / 1000:7.3f} {r['upkeep'] / 1000:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--session", type=int, default=None)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--round", type=int, action="append", default=[])
    ap.add_argument("--cell", default=None)
    ap.add_argument("--tail", type=int, default=0)
    args = ap.parse_args()

    report = json.load(open(args.census))
    for entry in report["sessions"]:
        if args.session is not None and entry["session_index"] != args.session:
            continue
        rows = entry["round_table"]
        print(f"=== session {entry['session_index']} "
              f"rounds={entry['rounds']} leg={entry['leg_round_us'] / 1e6:.3f}s")
        if args.top:
            show(sorted(rows, key=lambda r: -r["round_us"])[: args.top],
                 f"slowest {args.top}")
        if args.cell:
            members = [r for r in rows if r["cell"] == args.cell]
            show(sorted(members, key=lambda r: -r["round_us"]),
                 f"cell {args.cell}")
        if args.round:
            show([r for r in rows if r["round"] in args.round], "named rounds")
        if args.tail:
            show(rows[-args.tail:], f"last {args.tail}")


if __name__ == "__main__":
    main()
