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


def untruncated_histogram() -> dict[int, float]:
    """The same F83 mixture before the routed renormalisation.

    `ranked_histogram` divides out the M=2 mass because M=2 is below
    `Qwen35CustomQMV.widths` and never reaches either pipeline. Keeping that
    mass changes every width share by one common factor, so it is a
    denominator choice, not a different belief about the runner.
    """
    weights = {p: w for p, (_, w) in census.RANKED_WIDTH_MIX.items()}
    total = sum(weights.values())
    out: dict[int, float] = {}
    for prompt, (width, _w) in census.RANKED_WIDTH_MIX.items():
        share = weights[prompt] / total
        for m, p in census.e114.maxent(width).items():
            out[m] = out.get(m, 0.0) + share * p
    return out


# Edward's F83-weighted ranked frame, quoted by the advisor in F22 section 3.
# Only these four numbers were stated. mass(8) below is the stated tail minus
# the two stated masses; nothing else is reconstructed. The mass under M=6 was
# not stated, so this frame can price a pair only when the differing widths lie
# inside {6, 7, 8}, and its denominator still hides an unstated mass(2).
F83_STATED = {6: 0.188, 7: 0.211}
F83_TAIL_GE_6 = 0.5861
F83_MEAN_M = 5.7732


def f83_frame() -> dict[int, float]:
    return dict(F83_STATED) | {8: F83_TAIL_GE_6 - sum(F83_STATED.values())}


def f83_mass(widths: tuple[int, ...], mass_2: float) -> float:
    """Share of routed rounds at `widths` under the stated F83 frame.

    `mass_2` is the unstated mass the frame puts at M=2. Passing 0 treats the
    stated numbers as already routed; passing the maxent value renormalises
    them the way `ranked_histogram` renormalises ours. The two ends bracket the
    frame rather than pretending one of them is known.
    """
    frame = f83_frame()
    return sum(frame[m] for m in widths) / (1.0 - mass_2)


def mass(hist: dict[int, float], widths: tuple[int, ...],
         weight: dict[int, int] | None) -> float:
    if weight is None:
        num = sum(hist[m] for m in widths)
        den = sum(hist.values())
    else:
        num = sum(hist[m] * weight[m] for m in widths)
        den = sum(hist[m] * weight[m] for m in RANKED_WIDTHS)
    return num / den


def frame_comparison() -> list[dict]:
    """Price every pair under each candidate ranked frame.

    The three frames agree on the mean verification width to four significant
    figures and disagree on the shape of the top of the distribution. A table
    change is priced by the shape, so agreement on the mean is not agreement on
    the answer.
    """
    loc = local_histogram()
    routed, untrunc = ranked_histogram(), untruncated_histogram()
    mass_2 = untrunc[2]
    rows = []
    print("transfer factor by ranked frame, rounds weighting")
    print(f"  {'pair':26s} {'widths':10s} {'maxent unt':>11s} "
          f"{'maxent rtd':>11s} {'F83 as-is':>10s} {'F83 renorm':>11s}")
    for a, b in PAIRS:
        w = differing_widths(a, b)
        lr = mass(loc, w, None)
        cells = {
            "maxent_untruncated": mass(untrunc, w, None) / lr,
            "maxent_routed": mass(routed, w, None) / lr,
            "f83_as_stated": f83_mass(w, 0.0) / lr,
            "f83_renormalised": f83_mass(w, mass_2) / lr,
        }
        print(f"  {a + ' -> ' + b:26s} {str(list(w)):10s} "
              + " ".join(f"{cells[k]:>11.3f}" for k in
                         ("maxent_untruncated", "maxent_routed",
                          "f83_as_stated", "f83_renormalised")))
        rows.append({"from": a, "to": b, "widths": list(w),
                     "local_mass_rounds": lr, "factors": cells,
                     "spread_pct": 100.0 * (max(cells.values())
                                            / min(cells.values()) - 1.0)})
    print(f"  unstated F83 mass at M=2 bracketed by 0 and {mass_2:.4f}")
    worst = max(rows, key=lambda r: r["spread_pct"])
    print(f"  widest frame disagreement {worst['spread_pct']:.1f} % on "
          f"{worst['from']} -> {worst['to']}")
    return rows


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

    print()
    frames = frame_comparison()

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
        "untruncated_histogram": untruncated_histogram(),
        "f83_stated_frame": {"mass": f83_frame(), "tail_ge_6": F83_TAIL_GE_6,
                             "mean_m": F83_MEAN_M},
        "pairs": records,
        "frame_comparison": frames,
    }, indent=1) + "\n")
    print()
    print("wrote %s" % args.out)
    print("First-order transfer. It assumes the per-width saving is the same "
          "fraction of per-width QMV time on both harnesses. A receipt tests "
          "that assumption; this script does not establish it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
