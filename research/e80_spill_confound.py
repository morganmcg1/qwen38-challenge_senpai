#!/usr/bin/env python3
"""E80 rung 0a. Cross the measured E76 parity verdict with the `applegpu_g16s`
spill column.

`research/e76-results.md` refutation 2 read spill from the `applegpu_g17s`
column only, and concluded "it is not spill". The g16s column -- the
architecture the parity run actually executed on -- separates the two classes
perfectly, so that refutation does not hold. This script rebuilds the
contingency table from the primary artifacts:

  verdicts  research/e76-artifacts/parity-na{3,4,5,6}-b{0,1,2}.json
  spill     research/e76-artifacts/rung1.json

An arm-width pair FAILS when any priced shape reports a non-zero
`parity_differing_vs_plain` element count.
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "research/e76-artifacts"
OUT = ROOT / "research/e80-artifacts/rung0a-spill-parity-cross.json"


def load_verdicts():
    """Return {(arm, na): worst differing-element count over every priced shape}."""
    worst = collections.defaultdict(float)
    for path in sorted(ART.glob("parity-na*-b*.json")):
        doc = json.loads(path.read_text())
        na = doc["na"]
        for shape in doc["shapes"]:
            for arm, frac in shape["parity_differing_vs_plain"].items():
                worst[(arm, na)] = max(worst[(arm, na)], float(frac))
    return dict(worst)


def main() -> int:
    census = json.loads((ART / "rung1.json").read_text())["census"]
    g16s, g17s = census["applegpu_g16s"], census["applegpu_g17s"]
    verdicts = load_verdicts()

    rows = []
    for (arm, na), frac in sorted(verdicts.items()):
        key = f"e76_{arm}_na{na}"
        if key not in g16s:
            print(f"WARNING: no census entry for {key}", file=sys.stderr)
            continue
        rows.append({
            "arm": arm, "na": na,
            "differing_elements": frac,
            "fails": frac > 0.0,
            "g16s_registers": g16s[key]["registers"],
            "g16s_spill_bytes": g16s[key]["spill_bytes"],
            "g17s_registers": g17s[key]["registers"],
            "g17s_spill_bytes": g17s[key]["spill_bytes"],
        })

    fails = [r for r in rows if r["fails"]]
    passes = [r for r in rows if not r["fails"]]
    min_fail_g16s = min(r["g16s_spill_bytes"] for r in fails)
    max_pass_g16s = max(r["g16s_spill_bytes"] for r in passes)
    min_fail_g17s = min(r["g17s_spill_bytes"] for r in fails)
    max_pass_g17s = max(r["g17s_spill_bytes"] for r in passes)

    print(f"parity-tested pairs = {len(rows)}  failing = {len(fails)}  "
          f"passing = {len(passes)}")
    print()
    print("| arm | NA | differing elements | g16s regs / spill | g17s regs / spill |")
    print("|---|---:|---:|---:|---:|")
    for r in sorted(fails, key=lambda r: (r["arm"], r["na"])):
        print(f"| `{r['arm']}` | {r['na']} | {r['differing_elements']:,.0f} "
              f"| {r['g16s_registers']} / **{r['g16s_spill_bytes']}** "
              f"| {r['g17s_registers']} / {r['g17s_spill_bytes']} |")

    print()
    print(f"g16s: min spill over FAILING = {min_fail_g16s} B, "
          f"max spill over PASSING = {max_pass_g16s} B -> "
          f"{'CLEAN SEPARATION' if min_fail_g16s > max_pass_g16s else 'overlap'}")
    print(f"g17s: min spill over FAILING = {min_fail_g17s} B, "
          f"max spill over PASSING = {max_pass_g17s} B -> "
          f"{'clean separation' if min_fail_g17s > max_pass_g17s else 'OVERLAP'}")
    print()
    print("highest-spill PASSING pairs on g16s:")
    for r in sorted(passes, key=lambda r: -r["g16s_spill_bytes"])[:5]:
        print(f"  {r['arm']}_na{r['na']}: {r['g16s_registers']} regs / "
              f"{r['g16s_spill_bytes']} B")

    shipped = {n: {"g16s": g16s[f"shipped_na{n}"], "g17s": g17s[f"shipped_na{n}"]}
               for n in (2, 3, 4, 5, 6)}
    s6 = shipped[6]
    print()
    print(f"shipped_na6 -- the <T,6,6> instantiation the candidate ships -- "
          f"g16s {s6['g16s']['registers']} / {s6['g16s']['spill_bytes']} B, "
          f"g17s {s6['g17s']['registers']} / {s6['g17s']['spill_bytes']} B")
    print(f"headroom from shipped g16s spill to the observed hazard threshold = "
          f"{min_fail_g16s / max(s6['g16s']['spill_bytes'], 1):.1f}x")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source_verdicts": sorted(p.name for p in ART.glob("parity-na*-b*.json")),
        "source_census": "research/e76-artifacts/rung1.json",
        "pairs_total": len(rows),
        "pairs_failing": len(fails),
        "pairs_passing": len(passes),
        "g16s_min_spill_failing_bytes": min_fail_g16s,
        "g16s_max_spill_passing_bytes": max_pass_g16s,
        "g16s_separation_clean": min_fail_g16s > max_pass_g16s,
        "g17s_min_spill_failing_bytes": min_fail_g17s,
        "g17s_max_spill_passing_bytes": max_pass_g17s,
        "g17s_separation_clean": min_fail_g17s > max_pass_g17s,
        "shipped": {f"na{n}": {
            "g16s_registers": v["g16s"]["registers"],
            "g16s_spill_bytes": v["g16s"]["spill_bytes"],
            "g17s_registers": v["g17s"]["registers"],
            "g17s_spill_bytes": v["g17s"]["spill_bytes"],
        } for n, v in shipped.items()},
        "pairs": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
