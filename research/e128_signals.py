#!/usr/bin/env python3
"""E128-F4 item 4 - rank the candidate discrimination signals by measured AUC.

harness=local instrument, ranked target. Zero GPU: this reads the archived
rung-1 forced-depth traces, which already record every scalar the schedule may
legally read before it proposes anything.

The shipped schedule discriminates on ONE signal, the pending primary's target
top-2 margin, through a strictly downward `min(p, conf)` override at depths 0
and 1. E122 measured that signal's pooled AUC at 0.5109. The forced-depth arm
removes the scheduler's own selection, so every position outcome it records is
uncensored with respect to the estimator, and the free signals in the same
trace line can be scored against it on equal terms.

Free signals, already in the trace, needing no new instrumentation:

    margin      the shipped signal
    ema_j       the per-position acceptance EMA at the position being scored
    reach_j     the product of the EMAs up to that position, the shipped
                `reach` without the override
    streak      `fullAcceptStreak`, the accepted prefix length history
    prev_acc    the previous round's accepted count
    round_idx   position within the decode window

AUC is computed per position and per fixture and never pooled, because the
advisor's Rule 76 amendment makes the per-step acceptance `p` the regime
variable and pooling across regimes destroyed the E122 signal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# Fixtures grouped by which ranked prompts they stand for. The band is the
# advisor's amended Rule 76 window, per-step p in [0.934, 0.966].
BAND_FIXTURES = ("benchfixture", "medicine_hippoc")
LOWER_FIXTURES = ("beagle_a", "beagle_b", "republic_jowett", "drama_dollhouse")

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?"
    r"arm=(\S+) m=(\S+) streak=(\d+) cap=(\d+) ema=([0-9.,]+)")


def parse_trace(path: Path) -> list[dict]:
    rounds = []
    for line in path.read_text().splitlines():
        hit = ROUND_RE.search(line)
        if hit is None:
            continue
        rounds.append({
            "round": int(hit.group(1)),
            "depth": int(hit.group(2)),
            "acc": int(hit.group(3)),
            "arm": hit.group(4),
            "margin": float(hit.group(5)),
            "streak": int(hit.group(6)),
            "cap": int(hit.group(7)),
            "ema": [float(v) for v in hit.group(8).split(",")],
        })
    return rounds


def auc(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """Mann-Whitney AUC with a normal-approximation 95 % interval.

    Ties are given half credit, which is the correct handling for the margin:
    it is quantized at 2^-4 and reuses values often.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan"), len(pos), len(neg)
    diff = pos[:, None] - neg[None, :]
    wins = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    value = wins / (len(pos) * len(neg))
    # Hanley-McNeil standard error.
    q1 = value / (2.0 - value)
    q2 = 2.0 * value * value / (1.0 + value)
    var = (value * (1.0 - value)
           + (len(pos) - 1) * (q1 - value * value)
           + (len(neg) - 1) * (q2 - value * value)) / (len(pos) * len(neg))
    se = math.sqrt(max(var, 0.0))
    return value, value - 1.96 * se, value + 1.96 * se, len(pos), len(neg)


def signals_at(rounds: list[dict], position: int) -> dict:
    """Round-start signals and the outcome at `position`, on reached rounds."""
    rows = {"margin": [], "ema_j": [], "reach_j": [], "streak": [],
            "prev_acc": [], "round_idx": [], "label": []}
    for index, record in enumerate(rounds):
        if record["depth"] <= position:
            continue  # the round never proposed that position
        if record["acc"] < position:
            continue  # the chain was already rejected before it
        ema = record["ema"]
        rows["margin"].append(record["margin"])
        rows["ema_j"].append(ema[position] if position < len(ema) else ema[-1])
        rows["reach_j"].append(float(np.prod(ema[:position + 1])))
        rows["streak"].append(float(record["streak"]))
        rows["prev_acc"].append(float(rounds[index - 1]["acc"])
                                if index > 0 else 0.0)
        rows["round_idx"].append(float(record["round"]))
        rows["label"].append(1 if record["acc"] > position else 0)
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def fixture_table(rounds: list[dict], positions: range) -> list:
    out = []
    for position in positions:
        data = signals_at(rounds, position)
        labels = data.pop("label")
        if len(labels) < 20 or labels.min() == labels.max():
            continue
        row = {"position": position, "n": int(len(labels)),
               "p": float(labels.mean()), "signals": {}}
        for name, scores in data.items():
            if np.allclose(scores, scores[0]):
                continue
            value, low, high, n_pos, n_neg = auc(scores, labels)
            row["signals"][name] = {"auc": value, "low": low, "high": high,
                                    "n_pos": n_pos, "n_neg": n_neg}
        out.append(row)
    return out


def pooled_over_positions(table: list, name: str) -> float:
    """Sample-size weighted mean AUC over positions within one fixture."""
    num = den = 0.0
    for row in table:
        got = row["signals"].get(name)
        if got is None or math.isnan(got["auc"]):
            continue
        weight = got["n_pos"] * got["n_neg"]
        num += weight * got["auc"]
        den += weight
    return num / den if den else float("nan")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path,
                        default=here.parent / ".mlxfast-private/e128/"
                                              "runs-forced")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    names = ("margin", "ema_j", "reach_j", "streak", "prev_acc", "round_idx")
    per_fixture, summary = {}, {}
    for directory in sorted(args.runs.iterdir()):
        trace = directory / "trace.txt"
        if not trace.is_file():
            continue
        rounds = parse_trace(trace)
        if not rounds:
            continue
        table = fixture_table(rounds, range(0, 7))
        rejects = sum(row["signals"]["margin"]["n_neg"] for row in table
                      if "margin" in row["signals"])
        per_fixture[directory.name] = {"rounds": len(rounds), "table": table,
                                       "rejects_scored": rejects}
        summary[directory.name] = {
            name: pooled_over_positions(table, name) for name in names}

    print("harness=local instrument  E128-F4 item 4 - discrimination signals")
    print("forced depth 7, uncensored by the estimator, never pooled across "
          "fixtures")
    print("`rej` is the total number of rejects scored; it caps the precision "
          "of every AUC on that row\n")
    print("%-18s %6s %4s %8s %8s %8s %8s %8s %8s" % (
        ("fixture", "rounds", "rej") + names))
    for fixture in sorted(summary):
        print("%-18s %6d %4d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            (fixture, per_fixture[fixture]["rounds"],
             per_fixture[fixture]["rejects_scored"])
            + tuple(summary[fixture][n] for n in names)))

    def stratum(members, label):
        row = {}
        for name in names:
            values = [summary[f][name] for f in members if f in summary]
            row[name] = float(np.mean(values)) if values else float("nan")
        print("%-18s %6s %4s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            (label, "-", "-") + tuple(row[n] for n in names)))
        return row

    print()
    band = stratum(BAND_FIXTURES, "BAND p>=0.93")
    lower = stratum(LOWER_FIXTURES, "LOWER proxy")
    allf = stratum(tuple(summary), "ALL fixtures")

    print("\n## per-position detail, band fixtures only")
    for fixture in BAND_FIXTURES:
        if fixture not in per_fixture:
            continue
        print("\n%s" % fixture)
        print("%-4s %6s %7s %8s %8s %8s %8s" % (
            "pos", "n", "p", "margin", "ema_j", "reach_j", "streak"))
        for row in per_fixture[fixture]["table"]:
            got = row["signals"]
            print("%-4d %6d %7.4f %8.4f %8.4f %8.4f %8.4f" % (
                row["position"], row["n"], row["p"],
                got.get("margin", {}).get("auc", float("nan")),
                got.get("ema_j", {}).get("auc", float("nan")),
                got.get("reach_j", {}).get("auc", float("nan")),
                got.get("streak", {}).get("auc", float("nan"))))

    if args.json:
        args.json.write_text(json.dumps({
            "harness": "local instrument",
            "per_fixture": per_fixture,
            "summary": summary,
            "strata": {"band": band, "lower": lower, "all": allf},
        }, indent=2) + "\n")
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
