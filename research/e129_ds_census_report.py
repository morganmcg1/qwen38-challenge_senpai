#!/usr/bin/env python3
"""E129 F17 — read the paired entry-point censuses and price D_S.

    usage: research/e129_ds_census_report.py DIR [--json PATH]

`DIR` holds pairs written by `research/e129_entry_point_census.py`:
`census-<table>-nods.json` is the QMV template at the pre-D_S revision and
`census-<table>-ds.json` is the worktree template. Both are compiled by the
same instrument for the same plan, so every difference between a pair is the
code motion and nothing else.

WHAT IS REPORTED. Per tier, on both architectures, in the `sumtable` arm that
production routes at M >= 4: registers, spill bytes and DERIVED resident
simdgroups. Then the ranked-weighted residency gate for each table against the
shipped switch, on the D_S body.

WHAT IT CANNOT SHOW. Registers, spill and text bytes are read out of the
translated binary and are measurements. Resident simdgroups are
`budget // registers`, a model output. Nothing here is a timing result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ARMS = ("sumtable", "replica_no_table")
ARCHS = ("g17s", "g16s")


def load(path: pathlib.Path) -> tuple[dict, dict]:
    doc = json.loads(path.read_text())
    index = {(r["arm"], r["arch"], r["variant"], r["tier"]): r
             for r in doc["rows"]}
    return doc, index


def tier_table(before: dict, after: dict, tiers: list[int]) -> list[dict]:
    out = []
    for arch in ARCHS:
        for arm in ARMS:
            for tier in tiers:
                key = (arm, arch, "tier", tier)
                if key not in before or key not in after:
                    continue
                a, b = before[key], after[key]
                out.append({
                    "arch": arch,
                    "arm": arm,
                    "tier": tier,
                    "registers_before": a["registers"],
                    "spill_before": a["spill_bytes"],
                    "simdgroups_before": a["resident_simdgroups"],
                    "registers_after": b["registers"],
                    "spill_after": b["spill_bytes"],
                    "simdgroups_after": b["resident_simdgroups"],
                    "text_before": a["text_bytes"],
                    "text_after": b["text_bytes"],
                })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    tables = sorted({p.name[len("census-"):-len("-nods.json")]
                     for p in args.directory.glob("census-*-nods.json")})
    if not tables:
        raise SystemExit("no census-*-nods.json in %s" % args.directory)

    result: dict = {"tables": {}}
    for name in tables:
        pre, before = load(args.directory / ("census-%s-nods.json" % name))
        post, after = load(args.directory / ("census-%s-ds.json" % name))
        if pre["width_plan"] != post["width_plan"]:
            raise SystemExit("%s: the pair does not share a width plan" % name)
        if pre["header_sha256"] == post["header_sha256"]:
            raise SystemExit(
                "%s: both censuses used the same QMV template, so the pair "
                "cannot show the code motion" % name)
        result["tables"][name] = {
            "width_plan": post["width_plan"],
            "switch": {arch: {arm: {
                "registers_before": before[(arm, arch, "switch", None)]["registers"],
                "spill_before": before[(arm, arch, "switch", None)]["spill_bytes"],
                "simdgroups_before": before[(arm, arch, "switch", None)]["resident_simdgroups"],
                "registers_after": after[(arm, arch, "switch", None)]["registers"],
                "spill_after": after[(arm, arch, "switch", None)]["spill_bytes"],
                "simdgroups_after": after[(arm, arch, "switch", None)]["resident_simdgroups"],
            } for arm in ARMS} for arch in ARCHS},
            "header_before": pre["header_sha256"],
            "header_before_rev": pre["header_rev"],
            "header_after": post["header_sha256"],
            "header_after_rev": post["header_rev"],
            "tiers": tier_table(before, after, post["tiers"]),
            "gate_ds": {arch: {
                k: post["gate"][arch][k] for k in (
                    "ranked_weighted_shared", "ranked_weighted_tiered",
                    "ranked_weighted_gain_fraction", "passes_gate")}
                for arch in ("g16s", "g17s")},
            "gate_nods": {arch: {
                k: pre["gate"][arch][k] for k in (
                    "ranked_weighted_shared", "ranked_weighted_tiered",
                    "ranked_weighted_gain_fraction", "passes_gate")}
                for arch in ("g16s", "g17s")},
            "per_prompt_ds": {
                arch: {p: v["gain_fraction"]
                       for p, v in post["gate"][arch]["per_prompt"].items()}
                for arch in ("g16s", "g17s")},
            "pipelines": post["pipelines"]["tiered_pipeline_count"],
        }

    widest = max(tables, key=lambda n: len(result["tables"][n]["tiers"]))
    print("PER-TIER REGISTER CENSUS, plan %s, rps from the plan" % widest)
    print("  %-5s %-17s %4s | %8s %6s %4s | %8s %6s %4s | %s"
          % ("arch", "arm", "tier", "regs", "spill", "sg",
             "regs D_S", "spill", "sg", "d sg"))
    for row in result["tables"][widest]["tiers"]:
        print("  %-5s %-17s %4d | %8d %5dB %4d | %8d %5dB %4d | %+d"
              % (row["arch"], row["arm"], row["tier"],
                 row["registers_before"], row["spill_before"],
                 row["simdgroups_before"],
                 row["registers_after"], row["spill_after"],
                 row["simdgroups_after"],
                 row["simdgroups_after"] - row["simdgroups_before"]))

    print("\nSHIPPED SWITCH ENTRY POINT, the incumbent both designs replace")
    sw = result["tables"][widest]["switch"]
    for arch in ARCHS:
        for arm in ARMS:
            c = sw[arch][arm]
            print("  %-5s %-17s %3d regs /%3d B  %3d sg  ->  %3d regs /%3d B "
                  " %3d sg" % (arch, arm, c["registers_before"],
                               c["spill_before"], c["simdgroups_before"],
                               c["registers_after"], c["spill_after"],
                               c["simdgroups_after"]))

    print("\nRANKED-WEIGHTED RESIDENCY, vs the SAME-BODY shipped switch")
    print("  %-12s %-5s %19s %19s"
          % ("table", "arch", "pre-D_S body", "D_S body"))
    for name in tables:
        for arch in ("g17s", "g16s"):
            a = result["tables"][name]["gate_nods"][arch]
            b = result["tables"][name]["gate_ds"][arch]
            print("  %-12s %-5s %6.3f -> %6.3f %+6.2f %% %6.3f -> %6.3f "
                  "%+6.2f %%"
                  % (name, arch,
                     a["ranked_weighted_shared"], a["ranked_weighted_tiered"],
                     100 * a["ranked_weighted_gain_fraction"],
                     b["ranked_weighted_shared"], b["ranked_weighted_tiered"],
                     100 * b["ranked_weighted_gain_fraction"]))

    print("\nAGAINST THE CAMPAIGN INCUMBENT: pre-D_S body, shipped switch")
    for arch in ("g17s", "g16s"):
        incumbent = result["tables"][widest]["gate_nods"][arch][
            "ranked_weighted_shared"]
        print("  %-5s incumbent %6.3f resident simdgroups" % (arch, incumbent))
        for name in tables:
            for body, key in (("pre-D_S", "gate_nods"), ("D_S", "gate_ds")):
                value = result["tables"][name][key][arch][
                    "ranked_weighted_tiered"]
                print("      %-12s %-8s %6.3f  %+6.2f %%"
                      % (name, body, value, 100 * (value / incumbent - 1.0)))

    print("\nPER-PROMPT g17s RESIDENCY GAIN, D_S body")
    for name in tables:
        per = result["tables"][name]["per_prompt_ds"]["g17s"]
        print("  %-12s %s" % (name, "  ".join(
            "%s %+.2f %%" % (p, 100 * v) for p, v in per.items())))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True)
                             + "\n")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
