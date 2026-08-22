#!/usr/bin/env python3
"""E134 item 4: the shipped-policy population next to the forced-depth one.

Rung 1 refuted the information hypothesis on a *forced* depth-7 population.
The advisor's item 4 asks whether that refutation survives the population the
shipped policy actually produces, because the shipped policy selects which
rounds ever reach a deep boundary. This reports, per fixture and per boundary,
the observation count, the accept rate, and the fit-free AUC of each candidate
input under both populations. It fits nothing, so no fold or null machinery is
needed and a thin deep boundary degrades to "too few observations" instead of
crashing a pooled fit.

    python3 research/e134_item4_population.py \
        --shipped .mlxfast-private/e128/runs-shipped \
        --forced  .mlxfast-private/e128/runs-forced \
        --json research/e134-artifacts/item4-shipped-population.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e128_signals import auc  # noqa: E402
from e134_rung1 import (  # noqa: E402
    FIXTURE_PROMPT,
    MAX_DEPTH,
    SHORT,
    features_at,
    parse_trace,
)

# The shipped inputs plus the rung 1 candidates that carried any signal.
REPORTED = ("margin", "ema_d", "reach_shipped",
            "km_reach", "prev_margin_min", "prev_margin_at_d")
MIN_OBS = 20  # below this an AUC is reported but flagged as unusable


def load(runs: pathlib.Path) -> dict:
    out = {}
    for child in sorted(runs.iterdir()):
        trace = child / "trace.txt"
        if not trace.is_file():
            continue
        rounds, gate = parse_trace(trace)
        if not rounds:
            continue
        out[child.name] = {"rounds": rounds, "gate": gate}
    return out


def histogram(rounds: list[dict], key: str) -> dict:
    counts = {}
    for record in rounds:
        counts[record[key]] = counts.get(record[key], 0) + 1
    return {str(k): counts[k] for k in sorted(counts)}


def boundary_table(rounds: list[dict]) -> list[dict]:
    table = []
    for depth in range(MAX_DEPTH):
        cols = features_at(rounds, depth)
        labels = cols["label"]
        row = {"depth": depth, "obs": int(labels.size),
               "accept_rate": float(labels.mean()) if labels.size else None,
               "usable": bool(labels.size >= MIN_OBS
                              and 0 < labels.sum() < labels.size),
               "auc": {}}
        for name in REPORTED:
            if not row["usable"]:
                row["auc"][SHORT.get(name, name)] = None
                continue
            value, lo, hi, npos, nneg = auc(cols[name], labels)
            row["auc"][SHORT.get(name, name)] = {
                "value": float(value), "lo": float(lo), "hi": float(hi),
                "npos": int(npos), "nneg": int(nneg)}
        table.append(row)
    return table


def fixture_record(name: str, blob: dict) -> dict:
    rounds, gate = blob["rounds"], blob["gate"]
    return {
        "fixture": name,
        "prompt": FIXTURE_PROMPT.get(name),
        "rounds": len(rounds),
        "mean_offered_depth": float(np.mean([r["depth"] for r in rounds])),
        "mean_accepted": float(np.mean([r["acc"] for r in rounds])),
        "offered_depth_histogram": histogram(rounds, "depth"),
        "accepted_histogram": histogram(rounds, "acc"),
        "gate": {
            "rounds": gate["rounds"],
            "row_count_ok": gate["row_count_ok"],
            "row_count_bad": gate["row_count_bad"],
            "margin_identity_ok": gate["margin_identity_ok"],
            "margin_identity_bad": gate["margin_identity_bad"],
            "sched_checked": gate["sched_checked"],
            "sched_max_abs_error": gate["sched_max_abs_error"],
        },
        "boundaries": boundary_table(rounds),
    }


def print_population(title: str, records: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'fixture':<20}{'prompt':<10}{'rounds':>8}{'mean d':>9}"
          f"{'mean acc':>10}{'rows bad':>10}{'sched err':>12}")
    for rec in records:
        print(f"{rec['fixture']:<20}{str(rec['prompt']):<10}"
              f"{rec['rounds']:>8}{rec['mean_offered_depth']:>9.3f}"
              f"{rec['mean_accepted']:>10.3f}{rec['gate']['row_count_bad']:>10}"
              f"{rec['gate']['sched_max_abs_error']:>12.2e}")


def print_histograms(records: list[dict]) -> None:
    print(f"\n{'fixture':<20}{'key':<10}" +
          "".join(f"{d:>8}" for d in range(MAX_DEPTH + 1)))
    for rec in records:
        for key, label in (("offered_depth_histogram", "offered"),
                           ("accepted_histogram", "accepted")):
            cells = "".join(f"{rec[key].get(str(d), 0):>8}"
                            for d in range(MAX_DEPTH + 1))
            print(f"{rec['fixture']:<20}{label:<10}{cells}")


def print_boundaries(title: str, rec: dict) -> None:
    print(f"\n{title}  ({rec['fixture']}, {rec['rounds']} rounds)")
    names = [SHORT.get(n, n) for n in REPORTED]
    print(f"{'d':>3}{'obs':>7}{'accept':>9}" + "".join(f"{n:>10}"
                                                       for n in names))
    for row in rec["boundaries"]:
        acc = "     -" if row["accept_rate"] is None \
            else f"{row['accept_rate']:>9.4f}"
        cells = ""
        for name in names:
            cell = row["auc"][name]
            cells += "         -" if cell is None else f"{cell['value']:>10.4f}"
        print(f"{row['depth']:>3}{row['obs']:>7}{acc}{cells}")


def print_paired(shipped: dict, forced: dict, fixtures: list[str]) -> None:
    names = [SHORT.get(n, n) for n in REPORTED]
    for fixture in fixtures:
        if fixture not in shipped or fixture not in forced:
            continue
        print(f"\nshipped minus forced, {fixture}")
        print(f"{'d':>3}{'obs s':>8}{'obs f':>8}{'acc s':>8}{'acc f':>8}"
              + "".join(f"{n:>10}" for n in names))
        srows = {r["depth"]: r for r in shipped[fixture]["boundaries"]}
        frows = {r["depth"]: r for r in forced[fixture]["boundaries"]}
        for depth in range(MAX_DEPTH):
            s, f = srows[depth], frows[depth]
            sacc = "       -" if s["accept_rate"] is None \
                else f"{s['accept_rate']:>8.3f}"
            facc = "       -" if f["accept_rate"] is None \
                else f"{f['accept_rate']:>8.3f}"
            cells = ""
            for name in names:
                a, b = s["auc"][name], f["auc"][name]
                cells += "         -" if (a is None or b is None) \
                    else f"{a['value'] - b['value']:>+10.4f}"
            print(f"{depth:>3}{s['obs']:>8}{f['obs']:>8}{sacc}{facc}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shipped", required=True)
    parser.add_argument("--forced", required=True)
    parser.add_argument("--focus", nargs="*",
                        default=["medicine_hist", "essays_montaigne"])
    parser.add_argument("--json")
    args = parser.parse_args()

    shipped_raw = load(pathlib.Path(args.shipped))
    forced_raw = load(pathlib.Path(args.forced))
    shipped = {k: fixture_record(k, v) for k, v in shipped_raw.items()}
    forced = {k: fixture_record(k, v) for k, v in forced_raw.items()}

    print_population("SHIPPED-POLICY POPULATION", list(shipped.values()))
    print_histograms(list(shipped.values()))
    for fixture in args.focus:
        if fixture in shipped:
            print_boundaries("shipped policy", shipped[fixture])
        if fixture in forced:
            print_boundaries("forced depth 7", forced[fixture])
    print_paired(shipped, forced, args.focus)

    blob = {"shipped_dir": args.shipped, "forced_dir": args.forced,
            "focus": args.focus, "min_obs": MIN_OBS,
            "reported_inputs": list(REPORTED),
            "shipped": shipped, "forced": {k: forced[k] for k in args.focus
                                           if k in forced}}
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
