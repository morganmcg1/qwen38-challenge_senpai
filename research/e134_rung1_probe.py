#!/usr/bin/env python3
"""Probe the depth-4 inversion found by e134_rung1.py.

An AUC of 0.036 from a held-out logistic fit can mean a genuine inverted
signal or a degenerate fit. Raw single-column AUCs need no fit at all, so
they separate the two explanations.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e128_signals import auc  # noqa: E402
from e134_rung1 import (CANDIDATES, SHIPPED, features_at, is_usable,  # noqa
                        parse_trace, pooled_folds, weighted_mean)
from e128_ourcurve import F83_WEIGHT  # noqa: E402
from e134_rung1 import FIXTURE_PROMPT, PROMPT_FIXTURES  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parent.parent / (
    ".mlxfast-private/e128/runs-forced")


def main() -> int:
    traces = {}
    for directory in sorted(RUNS.iterdir()):
        path = directory / "trace.txt"
        if path.is_file():
            rounds, _ = parse_trace(path)
            if rounds:
                traces[directory.name] = rounds

    weight = {}
    for fixture in traces:
        prompt = FIXTURE_PROMPT.get(fixture)
        if prompt is not None:
            weight[fixture] = F83_WEIGHT[prompt] / len(PROMPT_FIXTURES[prompt])

    print("harness=local instrument  E134 rung 1 probe  zero GPU")
    print("raw per-fixture AUC of each shipped input, NO fitting at all\n")
    print("%-18s %2s %5s %6s %8s %8s %8s" % (
        "fixture", "d", "n", "rate", "margin", "ema_d", "reach"))
    for depth in (3, 4):
        for fixture in sorted(traces):
            if fixture not in weight:
                continue
            cols = features_at(traces[fixture], depth)
            labels = cols["label"]
            if len(labels) < 30 or labels.min() == labels.max():
                continue
            print("%-18s %2d %5d %6.3f %8.4f %8.4f %8.4f" % (
                fixture, depth, len(labels), labels.mean(),
                auc(cols["margin"], labels)[0],
                auc(cols["ema_d"], labels)[0],
                auc(cols["reach_shipped"], labels)[0]))
        print()

    print("## medicine_hist at depth 4, every column, raw AUC")
    cols = features_at(traces["medicine_hist"], 4)
    labels = cols["label"]
    print("  n %d  positives %d  rate %.4f"
          % (len(labels), int(labels.sum()), labels.mean()))
    for name in list(SHIPPED) + list(CANDIDATES):
        column = cols[name]
        if column.std() < 1e-12:
            print("  %-20s constant at %.6f" % (name, column[0]))
            continue
        value, lo, hi, npos, nneg = auc(column, labels)
        print("  %-20s %8.4f  [%.4f %.4f]" % (name, value, lo, hi))

    print("\n## is the shipped margin monotone in the outcome here?")
    order = np.argsort(cols["margin"])
    ranked = labels[order]
    quintile = np.array_split(ranked, 5)
    print("  margin quintile, low to high, then accept rate")
    for index, part in enumerate(quintile):
        print("    q%d  n %3d  rate %.4f" % (index + 1, len(part),
                                             part.mean()))

    print("\n## the same for depth 3, where the estimator works")
    cols3 = features_at(traces["medicine_hist"], 3)
    order3 = np.argsort(cols3["margin"])
    ranked3 = cols3["label"][order3]
    for index, part in enumerate(np.array_split(ranked3, 5)):
        print("    q%d  n %3d  rate %.4f" % (index + 1, len(part),
                                             part.mean()))

    shipped_dir = RUNS.parent / "runs-shipped"
    if not shipped_dir.is_dir():
        return 0
    print("\n## which rounds does the SHIPPED scheduler actually ask at d=4?")
    print("   Rung 1 measured forced-depth-7 traces, where every round is")
    print("   asked. The shipped scheduler chooses its own depth first, so it")
    print("   only reaches a deep boundary in rounds it already likes. Only")
    print("   these fixtures have an archived shipped leg.")
    print("%-18s %7s %9s %9s %9s %9s" % (
        "fixture", "rounds", "d>4", "reach d4", "P(acc>4)", "margin auc"))
    for directory in sorted(shipped_dir.iterdir()):
        path = directory / "trace.txt"
        if not path.is_file():
            continue
        rounds, _ = parse_trace(path)
        if not rounds:
            continue
        deep = [r for r in rounds if r["depth"] > 4]
        asked = [r for r in deep if r["acc"] >= 4]
        if len(asked) < 10:
            print("%-18s %7d %9d %9d %9s %9s" % (
                directory.name, len(rounds), len(deep), len(asked), "-", "-"))
            continue
        labels = np.array([1.0 if r["acc"] > 4 else 0.0 for r in asked])
        scores = np.array([r["margin"] for r in asked])
        value = ("-" if labels.min() == labels.max()
                 else "%.4f" % auc(scores, labels)[0])
        print("%-18s %7d %9d %9d %9.3f %9s" % (
            directory.name, len(rounds), len(deep), len(asked),
            labels.mean(), value))
    print("\n   `reach d4` is the population rung 1 scored. If it is a small")
    print("   share of decoding, then a perfect depth-4 rule moves few rounds")
    print("   however good its AUC is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
