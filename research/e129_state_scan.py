#!/usr/bin/env python3
"""Separate ranked host state from candidate code in recent receipts.

    python3 research/e129_state_scan.py [--since 2026-08-22T04:00]

The F76 mode index is a weighted sum over the CANDIDATE leg, so it cannot tell a
host-state shift from a code change that reshapes the per-prompt profile. The
serial leg can. ``program.md`` fixes the ranked causal boundary as

    d ln(ranked baseline serial time) / dx = 0   for every candidate edit x

so the unweighted 8-prompt mean of ``serial_seconds_per_token_mean`` moves only
with the host. It is the classifier; the candidate-leg mean is the measurement.

harness=ranked.
"""

from __future__ import annotations

import argparse
import statistics

from e129_prereg import BOARD, NAMES, ORDER, leg_means, load, mode_index


def serial_means(row: dict) -> dict[str, float]:
    return {
        NAMES[e["prompt_sha256"][:8]]: e["serial_seconds_per_token_mean"]
        for e in row["officialMetrics"]["per_prompt"]
    }


def unweighted(d: dict[str, float]) -> float:
    return sum(d[n] for n in ORDER) / len(ORDER)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-22T04:00")
    args = ap.parse_args()

    rows = [r for r in load().values() if r["createdAt"] >= args.since]
    rows.sort(key=lambda r: r["createdAt"])

    print("harness=ranked  board %s  %d receipts since %s"
          % (BOARD, len(rows), args.since))
    print()
    print("%-20s %-9s %-14s %9s %10s %10s %9s"
          % ("createdAt", "receipt", "solver", "published", "candidate", "serial", "F76"))
    cand, ser = [], []
    for r in rows:
        c = unweighted(leg_means(r))
        s = unweighted(serial_means(r))
        cand.append(c)
        ser.append(s)
        print("%-20s %-9s %-14s %9.6f %10.6f %10.6f %9.4f"
              % (r["createdAt"][:19], r["id"][:8], (r.get("solverUsername") or "")[:14],
                 r.get("officialScore") or float("nan"), c, s, mode_index(leg_means(r))))

    print()
    print("dispersion over the window, as a percent of the mean")
    for label, xs in (("candidate leg", cand), ("serial leg", ser)):
        m = statistics.mean(xs)
        print("  %-14s mean %.6f  sd %.4f %%  range %.4f %%"
              % (label, m, statistics.stdev(xs) / m * 100,
                 (max(xs) - min(xs)) / m * 100))
    print()
    print("reading: the serial leg cannot respond to candidate code, so its")
    print("dispersion bounds the host-state contribution. Any candidate-leg")
    print("dispersion above that bound is code.")

    print()
    print("same-solver repeats, consecutive gaps in the candidate leg")
    by_solver: dict[str, list] = {}
    for r in rows:
        by_solver.setdefault(r.get("solverUsername") or "?", []).append(r)
    for who, rs in sorted(by_solver.items()):
        if len(rs) < 2:
            continue
        gaps = []
        for a, b in zip(rs, rs[1:]):
            ca, cb = unweighted(leg_means(a)), unweighted(leg_means(b))
            gaps.append((b["createdAt"][11:19], (cb - ca) / ca * 100))
        print("  %-14s %s" % (who, "  ".join("%s %+.3f%%" % g for g in gaps)))
    print()
    print("a gap near the serial-leg bound is a repeat of the same tree; a large")
    print("gap is a code change. Repeats measure the host, not the solver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
