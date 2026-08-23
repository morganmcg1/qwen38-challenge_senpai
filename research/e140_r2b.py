"""E140 F7 item R2, part b: the reorder re-flag for the remaining cells.

`e140_r1r2.py` re-flags the E140 2x2, the E134 tier grid and the E140 cliff
perturbation. F8 asks for every cell E140 has ever reported. The two artifacts
left are the post-tight curve arms in `posttight.json` and the oracle-state
arms in `oracle-state.json`. Neither stored a per-prompt ratio, so the rank
vector cannot be recovered from the artifact and the cells must be replayed.

The replay is confined to the headline form `per_drafting_round`, which is the
form every reported post-tight and oracle conclusion used, and it reuses the
committed width-8 tier fits rather than re-fitting them. Each cell carries a
self-check: the replayed median must reproduce the committed median, otherwise
the rank vector does not belong to the number it is flagging.

Rank vectors are computed per seed. A cell reports its modal vector and a
stability flag, so a vector that only holds on some seeds cannot be read as a
property of the mechanism.

harness=local. Zero GPU.

Usage:
  python3 e140_r2b.py --json e140-artifacts/r2b.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct  # noqa: E402
from e140_cells import (  # noqa: E402
    install_pb68, load_masses, lopo_curves, run_cell as cells_run_cell,
    transfer_cache,
)
from e140_lookahead import SHIPPED_TIER, load_curves  # noqa: E402
from e140_oracle_state import CELLS as ORACLE_CELLS  # noqa: E402
from e140_oracle_state import run_cell as oracle_run_cell  # noqa: E402
from e140_posttight import ARMS as POSTTIGHT_ARMS  # noqa: E402
from e140_posttight import VARIANTS  # noqa: E402
from e140_r1r2 import binding_gap, rank_row, rank_vector, ship_raws  # noqa: E402

FORM = "per_drafting_round"
TOLERANCE = 0.02

# Rule 123: beagle holds rank 3 and the upper slot is the minimum of the four
# heavy prompts, so the bottom three carry no weight at any magnitude.
ZERO_WEIGHT = ("plutarch", "drama", "travel")
PLUTARCH_UNLOCK_RATIO = 0.95


def replay(runner, cache, seeds, curve, lopo, receipt, windows, cell,
           ship_order) -> dict:
    """One cell: per-seed medians, per-seed rank vectors, modal vector."""
    in_sample, held_out, vectors, ratio_acc = [], [], [], {}
    for seed in seeds:
        ratios = {p: runner(cache, seed, p, cell, curve, windows)
                  for p in RANKED_PROMPTS}
        in_sample.append(median_pct(receipt, ratios))
        row = rank_row(receipt, ratios, ship_order)
        vectors.append(tuple(row["rank_vector"]))
        for prompt, entry in ratios.items():
            ratio_acc.setdefault(prompt, []).append(entry["ratio"])
        if lopo is not None:
            held_out.append(median_pct(receipt, {
                p: runner(cache, seed, p, cell, lopo[p], windows)
                for p in RANKED_PROMPTS}))
    modal, hits = collections.Counter(vectors).most_common(1)[0]
    mean_ratio = {p: statistics.fmean(v) for p, v in ratio_acc.items()}
    gain = {p: 1.0 - r for p, r in mean_ratio.items()}
    total = sum(gain.values())
    return {
        "in_sample": statistics.fmean(in_sample),
        "curve_lopo": statistics.fmean(held_out) if held_out else None,
        "rank_vector": list(modal),
        "median_pair": [modal[3], modal[4]],
        "reordered": list(modal) != ship_order,
        "pair_changed": set(modal[3:5]) != set(ship_order[3:5]),
        "rank_stable": hits == len(seeds),
        "rank_agreement": hits / len(seeds),
        "per_prompt_ratio": mean_ratio,
        "beagle_cost_pct": (mean_ratio["beagle"] - 1.0) * 100.0,
        "zero_weight_gain_share": (
            sum(gain[p] for p in ZERO_WEIGHT) / total if total else 0.0),
        "plutarch_unlock": mean_ratio["plutarch"] < PLUTARCH_UNLOCK_RATIO,
    }


def check(name, replayed, committed, notes) -> bool:
    if committed is None:
        notes.append("%s: no committed median to check against" % name)
        return False
    ok = abs(replayed - committed) <= TOLERANCE
    if not ok:
        notes.append("%s: replay %+.4f but artifact %+.4f"
                     % (name, replayed, committed))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--artifacts", type=pathlib.Path,
                    default=HERE / "e140-artifacts")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/r2b.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 F7 R2b  zero GPU")
    curves, _ = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)
    curve = curves[FORM]
    lopo = lopo_curves(masses, FORM) if FORM in FORMS else None

    order_ship = rank_vector(ship_raws(receipt))
    print("ship rank vector: %s" % " ".join(order_ship))
    print("median pair     : %s/%s" % (order_ship[3], order_ship[4]))
    print("Rule 121 binding gap: %.2f %%" % binding_gap(ship_raws(receipt)))

    notes, out = [], {}

    posttight = json.loads((args.artifacts / "posttight.json").read_text())
    tier8 = posttight.get("tier8_fits", {})
    print("\n## the post-tight curve arms, form %s" % FORM)
    print("%-28s %10s %10s %10s %-22s %8s %7s %s"
          % ("curve|arm", "replay", "artifact", "curve-lopo",
             "rank flag / pair", "beagle%", "0wt", "unlock"))
    post = {}
    for name, make in VARIANTS.items():
        variant = make(curves[FORM])
        variant_lopo = ({p: make(c) for p, c in lopo.items()}
                        if lopo is not None else None)
        install_pb68(SHIPPED_TIER,
                     tier8.get(name, tier8.get("base", {}))["best_tier8"])
        for arm in POSTTIGHT_ARMS:
            row = replay(cells_run_cell, cache, seeds, variant, variant_lopo,
                         receipt, args.windows, arm, order_ship)
            key = "%s|%s" % (name, arm)
            stored = posttight["arms"].get("%s|%s|%s" % (FORM, name, arm))
            committed = stored["in_sample"][0] if stored else None
            row["artifact_in_sample"] = committed
            row["reproduces_artifact"] = check(key, row["in_sample"],
                                               committed, notes)
            post[key] = row
            print("%-28s %+10.4f %+10s %+10.4f %-22s %+8.3f %6.1f%% %s%s"
                  % (key, row["in_sample"],
                     "n/a" if committed is None else "%.4f" % committed,
                     row["curve_lopo"] or 0.0,
                     "%s %s/%s" % ("YES" if row["reordered"] else " no",
                                   row["median_pair"][0],
                                   row["median_pair"][1]),
                     row["beagle_cost_pct"],
                     100.0 * row["zero_weight_gain_share"],
                     "UNLOCK" if row["plutarch_unlock"] else "  -   ",
                     "" if row["rank_stable"] else "  UNSTABLE"))
    out["posttight"] = post

    oracle = json.loads((args.artifacts / "oracle-state.json").read_text())
    print("\n## the oracle-state arms, form %s" % FORM)
    print("%-28s %10s %10s %10s %-22s %8s %7s %s"
          % ("cell", "replay", "artifact", "curve-lopo", "rank flag / pair",
             "beagle%", "0wt", "unlock"))
    orac = {}
    for cell in ORACLE_CELLS:
        row = replay(oracle_run_cell, cache, seeds, curve, lopo, receipt,
                     args.windows, cell, order_ship)
        stored = oracle["cells"].get(cell)
        committed = stored["in_sample_mean"] if stored else None
        row["artifact_in_sample"] = committed
        row["reproduces_artifact"] = check(cell, row["in_sample"],
                                           committed, notes)
        orac[cell] = row
        print("%-28s %+10.4f %+10s %+10.4f %-22s %+8.3f %6.1f%% %s%s"
              % (cell, row["in_sample"],
                 "n/a" if committed is None else "%.4f" % committed,
                 row["curve_lopo"] or 0.0,
                 "%s %s/%s" % ("YES" if row["reordered"] else " no",
                               row["median_pair"][0], row["median_pair"][1]),
                 row["beagle_cost_pct"],
                 100.0 * row["zero_weight_gain_share"],
                 "UNLOCK" if row["plutarch_unlock"] else "  -   ",
                 "" if row["rank_stable"] else "  UNSTABLE"))
    out["oracle"] = orac

    rows = list(post.values()) + list(orac.values())
    lower = collections.Counter(r["median_pair"][0] for r in rows)
    upper = collections.Counter(r["median_pair"][1] for r in rows)
    plutarch_unlock = {k: r for k, r in
                       list(post.items()) + list(orac.items())
                       if r["plutarch_unlock"]}
    print("\n## flags")
    print("cells re-flagged        %d" % len(rows))
    print("reordered               %d" % sum(1 for r in rows if r["reordered"]))
    print("median pair changed     %d"
          % sum(1 for r in rows if r["pair_changed"]))
    print("rank vector unstable    %d"
          % sum(1 for r in rows if not r["rank_stable"]))
    print("reproduce the artifact  %d of %d"
          % (sum(1 for r in rows if r["reproduces_artifact"]), len(rows)))
    print("lower slot              %s" % dict(lower))
    print("upper slot              %s" % dict(upper))
    print("plutarch unlock cells   %d %s"
          % (len(plutarch_unlock), sorted(plutarch_unlock)))
    for note in notes:
        print("  note: %s" % note)

    out["summary"] = {
        "cells": len(rows),
        "reordered": sum(1 for r in rows if r["reordered"]),
        "pair_changed": sum(1 for r in rows if r["pair_changed"]),
        "rank_unstable": sum(1 for r in rows if not r["rank_stable"]),
        "reproduces_artifact": sum(1 for r in rows
                                   if r["reproduces_artifact"]),
        "lower_slot": dict(lower),
        "upper_slot": dict(upper),
        "plutarch_unlock_cells": sorted(plutarch_unlock),
        "notes": notes,
    }
    out["ship_rank_vector"] = order_ship
    out["binding_gap_pct"] = binding_gap(ship_raws(receipt))
    out["form"] = FORM
    out["receipt"] = args.receipt
    out["windows"] = args.windows
    out["seeds"] = seeds
    out["harness"] = "local"
    out["gpu_used"] = False
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
