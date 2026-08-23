#!/usr/bin/env python3
"""Read which probe-selection arm a leg actually dispatched.

`qwen35E87SelectEnabled` parses one environment variable, so an unset or
misspelled selector and a selector that arrived and was ignored both run the
incumbent chain and are indistinguishable from outside. The dispatch itself
records its own arm through `Qwen35CustomQMV.notePipeline`, so this reads the
answer off the run instead of off the environment.

    usage: e135_e87_select_check.py PIPELINES_JSON --want {select,incumbent}

Exit 0 when the leg dispatched the wanted arm and only that arm.
"""
import argparse
import json
import sys

SELECT = "e87_probe_select"
INCUMBENT = ("e87_probe_sort_compaction", "e87_probe_merge_sort")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipelines")
    ap.add_argument("--want", required=True, choices=["select", "incumbent"])
    args = ap.parse_args()

    log = json.load(open(args.pipelines))
    keys = log.get("by_key", {})
    select = keys.get(SELECT, 0)
    incumbent = {name: keys.get(name, 0) for name in INCUMBENT}
    total_incumbent = sum(incumbent.values())

    print(f"  probe_arm {log.get('probe_arm')}  leaves {log.get('probe_leaves')}"
          f"  probes {log.get('probe_count')}")
    print(f"  {SELECT:<28} {select}")
    for name, count in incumbent.items():
        print(f"  {name:<28} {count}")

    if args.want == "select":
        ok = select > 0 and total_incumbent == 0
    else:
        ok = total_incumbent > 0 and select == 0

    print(f"  want {args.want}: {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  a leg must dispatch exactly one arm, and it must be the one "
              "the session asked for", file=sys.stderr)
    return 0 if ok else 1


sys.exit(main())
