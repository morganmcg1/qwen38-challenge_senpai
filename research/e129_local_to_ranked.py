#!/usr/bin/env python3
"""Translate an ABCCBA local table delta into a ranked prediction.

The local fixture and the ranked mix put their verification widths in almost
opposite places. Over the 312 rounds of the rung 5e local session the routed
mass is 78 % at M=8 and 10 % across M=6 and M=7 together. The ranked mix is
27 % at M=8 and 38 % across M=6 and M=7. So the same table change reads
differently on the two harnesses, by a factor that this script computes rather
than assumes.

    python3 research/e129_local_to_ranked.py [--delta ARM=PCT ...]

Two weightings are reported for every pair, because round mass alone ignores
that a wide round costs more than a narrow one:

  rounds  the share of routed rounds at the widths where the two tables give
          different plans.
  passes  the same share, weighted by the shipped plan's reads of the weight
          matrix at that width, which is the first-order cost term.

The transfer factor is ranked mass over local mass. A measured local delta
times that factor is the ranked prediction. This is a first-order transfer and
it says so: it assumes the per-width saving is the same fraction of per-width
QMV time on both harnesses, which is the assumption a receipt then tests.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e129_entry_point_census as census  # noqa: E402

RANKED_WIDTHS = tuple(range(3, 9))
PAIRS = (
    ("shipped", "onepass67"),
    ("onepass67", "onepass678"),
    ("shipped", "onepass678"),
    ("shipped", "onepass6"),
)


def plan_ipg(name: str) -> dict[int, int]:
    return {m: ipg for m, ipg, _ in census.PLANS[name]}


def passes(name: str) -> dict[int, int]:
    return {m: -(-m // ipg) for m, ipg in plan_ipg(name).items()}


def differing_widths(a: str, b: str) -> tuple[int, ...]:
    pa, pb = plan_ipg(a), plan_ipg(b)
    return tuple(m for m in RANKED_WIDTHS if pa[m] != pb[m])


def local_histogram() -> dict[int, float]:
    routed = {m: n for m, n in census.LOCAL_HISTOGRAM.items()
              if m in RANKED_WIDTHS}
    total = sum(routed.values())
    return {m: routed.get(m, 0) / total for m in RANKED_WIDTHS}


def ranked_histogram() -> dict[int, float]:
    hists = census.ranked_histograms()
    weights = {p: w for p, (_, w) in census.RANKED_WIDTH_MIX.items()}
    total = sum(weights.values())
    out = {m: 0.0 for m in RANKED_WIDTHS}
    for prompt, hist in hists.items():
        w = weights[prompt] / total
        for m in RANKED_WIDTHS:
            out[m] += w * hist[m]
    return out


def mass(hist: dict[int, float], widths: tuple[int, ...],
         weight: dict[int, int] | None) -> float:
    if weight is None:
        num = sum(hist[m] for m in widths)
        den = sum(hist.values())
    else:
        num = sum(hist[m] * weight[m] for m in widths)
        den = sum(hist[m] * weight[m] for m in RANKED_WIDTHS)
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", action="append", default=[],
                    help="a measured local change, `onepass67=-3.2`, in per "
                         "cent of candidate MTP time, negative is faster")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/out/e129-local-to-ranked.json"))
    args = ap.parse_args()

    loc, rank = local_histogram(), ranked_histogram()
    base_passes = passes("shipped")

    print("routed width mass, M=3..8")
    print(f"{'harness':10s} " + " ".join(f"{'M=%d' % m:>7s}" for m in RANKED_WIDTHS))
    print(f"{'local':10s} " + " ".join(f"{loc[m]:7.3f}" for m in RANKED_WIDTHS))
    print(f"{'ranked':10s} " + " ".join(f"{rank[m]:7.3f}" for m in RANKED_WIDTHS))
    print()

    records = []
    print("transfer factor, ranked mass over local mass")
    print(f"{'pair':26s} {'widths':10s} {'local r':>8s} {'rank r':>8s} "
          f"{'x rounds':>9s} {'local p':>8s} {'rank p':>8s} {'x passes':>9s}")
    for a, b in PAIRS:
        w = differing_widths(a, b)
        lr, rr = mass(loc, w, None), mass(rank, w, None)
        lp, rp = mass(loc, w, base_passes), mass(rank, w, base_passes)
        fr = rr / lr if lr else float("nan")
        fp = rp / lp if lp else float("nan")
        label = f"{a} -> {b}"
        print(f"{label:26s} {str(list(w)):10s} {lr:8.3f} {rr:8.3f} {fr:9.3f} "
              f"{lp:8.3f} {rp:8.3f} {fp:9.3f}")
        records.append({"from": a, "to": b, "widths": list(w),
                        "local_mass_rounds": lr, "ranked_mass_rounds": rr,
                        "transfer_rounds": fr,
                        "local_mass_passes": lp, "ranked_mass_passes": rp,
                        "transfer_passes": fp})

    measured = {}
    for spec in args.delta:
        name, _, value = spec.partition("=")
        measured[name.strip()] = float(value)

    if measured:
        print()
        print("measured local deltas translated, negative is faster")
        print(f"{'pair':26s} {'local %':>9s} {'ranked, rounds':>15s} "
              f"{'ranked, passes':>15s}")
        for rec in records:
            key = rec["to"]
            if key not in measured:
                continue
            d = measured[key]
            rec["measured_local_pct"] = d
            rec["ranked_pct_rounds"] = d * rec["transfer_rounds"]
            rec["ranked_pct_passes"] = d * rec["transfer_passes"]
            label = f"{rec['from']} -> {rec['to']}"
            print(f"{label:26s} {d:9.3f} {rec['ranked_pct_rounds']:15.3f} "
                  f"{rec['ranked_pct_passes']:15.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "harness": "local-to-ranked transfer, first order",
        "local_histogram": loc,
        "ranked_histogram": rank,
        "pairs": records,
    }, indent=1) + "\n")
    print()
    print("wrote %s" % args.out)
    print("First-order transfer. It assumes the per-width saving is the same "
          "fraction of per-width QMV time on both harnesses. A receipt tests "
          "that assumption; this script does not establish it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
