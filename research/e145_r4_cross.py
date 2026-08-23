#!/usr/bin/env python3
"""E145 R4 cross-evaluation: tune the price plane on one curve, pay on the other.

R4 fits and scores on a single cost model. That answers "what is the best cell
under this curve", but not the question E145 exists to settle: the shipped
price plane was tuned against a replayed curve, so what did believing that
curve actually cost, now that the curve is measured?

The two R4 runs sweep the same (h, tier) grid with the same seeds, so no extra
fitting is needed. Take the argmax cell of the replayed grid, look that same
cell up in the measured grid, and compare it with the measured grid's own
argmax. The difference is the price of the wrong curve, in percentage points
of the R4 objective.

The regret is reported in the measured frame only. The replayed frame cannot
price a cell it mis-costs, so a "regret" computed there would just be the
fitting error read backwards.
"""
from __future__ import annotations

import json
import pathlib
import sys


def cell_label(key: str) -> str:
    h, tier = key.split("|")
    return "h=%.2f tier=%.2f" % (float(h), float(tier))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: e145_r4_cross.py R4_MEASURED_JSON R4_REPLAYED_JSON")
        return 2
    measured = json.loads(pathlib.Path(argv[1]).read_text())
    replayed = json.loads(pathlib.Path(argv[2]).read_text())

    if measured["cost_model"] == replayed["cost_model"]:
        print("e145_r4_cross: both blobs use cost model %s; nothing to cross"
              % measured["cost_model"])
        return 1
    for field in ("h_grid", "tier_grid", "seeds", "windows",
                  "width1_anchor_used", "receipt"):
        if measured[field] != replayed[field]:
            print("e145_r4_cross: %s differs between the two runs (%r vs %r);"
                  " the cells are not comparable"
                  % (field, measured[field], replayed[field]))
            return 1

    mgrid = measured["grid"]
    rgrid = replayed["grid"]
    missing = sorted(set(rgrid) - set(mgrid))
    if missing:
        print("e145_r4_cross: %d cells are absent from the measured grid"
              % len(missing))
        return 1

    m_best = measured["best_cell"]
    r_best = replayed["best_cell"]
    shipped = measured["shipped_cell"]
    uniform = measured["uniform_cell"]

    print("\n## the price of the wrong curve, scored in the measured frame")
    print("  every objective below comes from the measured cost model, so the"
          " rows differ only in which curve chose the cell")
    rows = [
        ("uniform, no boundary at all", uniform),
        ("shipped cell", shipped),
        ("cell chosen by the replayed curve", r_best),
        ("cell chosen by the measured curve", m_best),
    ]
    for name, key in rows:
        cell = mgrid[key]
        print("  %-34s %-18s objective %+7.3f  spread %5.3f  worst slot %s"
              % (name, cell_label(key), cell["objective_mean"],
                 cell["upper_slot_spread_mean"],
                 cell["worst_upper_slot_prompt"]))

    regret = (mgrid[m_best]["objective_mean"]
              - mgrid[r_best]["objective_mean"])
    over_shipped = (mgrid[m_best]["objective_mean"]
                    - mgrid[shipped]["objective_mean"])
    print("\n  regret of tuning on the replayed curve: %+0.4f pp" % regret)
    print("  gain of the measured argmax over the shipped cell: %+0.4f pp"
          % over_shipped)
    print("  the replayed curve picked %s; the measured curve picks %s; %s"
          % (cell_label(r_best), cell_label(m_best),
             "same cell" if r_best == m_best else "different cells"))

    print("\n  what the replayed curve believed it was buying: %+0.4f pp"
          % (rgrid[r_best]["objective_mean"]
             - rgrid[replayed["shipped_cell"]]["objective_mean"]))
    print("  what that cell is really worth against shipped: %+0.4f pp"
          % (mgrid[r_best]["objective_mean"] - mgrid[shipped]["objective_mean"]))

    blob = {
        "harness": "local",
        "gpu_used": False,
        "measured_best_cell": m_best,
        "replayed_best_cell": r_best,
        "shipped_cell": shipped,
        "same_cell": r_best == m_best,
        "regret_pp": regret,
        "measured_best_over_shipped_pp": over_shipped,
        "replayed_believed_gain_pp":
            (rgrid[r_best]["objective_mean"]
             - rgrid[replayed["shipped_cell"]]["objective_mean"]),
        "replayed_cell_true_gain_pp":
            (mgrid[r_best]["objective_mean"]
             - mgrid[shipped]["objective_mean"]),
        "measured_frame": {
            name: {"cell": key,
                   "objective": mgrid[key]["objective_mean"],
                   "upper_slot_spread": mgrid[key]["upper_slot_spread_mean"],
                   "worst_upper_slot_prompt":
                       mgrid[key]["worst_upper_slot_prompt"]}
            for name, key in rows
        },
    }
    out = pathlib.Path(argv[1]).with_name("r4-cross.json")
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
