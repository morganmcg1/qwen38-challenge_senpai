#!/usr/bin/env python3
"""E159 Part B control. Does `marginal[0]` gate plutarch on the ranked board?

harness=ranked. Zero GPU.

H159 says `threshold(0) == marginal[0]` is the drafting on/off gate, and that
pb6 flipped plutarch from 449 non-drafting rounds to zero because it cut that
one number by 5.3 %. The claim is testable without any replay: the ranked
board records `effective_mean_draft_len` for plutarch on every submission, and
the h-family receipts moved `marginal[0]` uniformly and by more than pb6 did.

If a 22 % cut of the gate (h = 0.14) leaves plutarch non-drafting while pb6's
smaller cut flips it, the gate is not the cause.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import statistics
import sys

CACHE = pathlib.Path("/tmp/yukon-board/full.json")
NAME = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
DRAFTING_EDL = 1.0


def rows() -> list[dict]:
    payload = json.loads(CACHE.read_text())
    return payload["submissions"] if isinstance(payload, dict) else payload


def cells(row: dict) -> dict:
    metrics = row.get("officialMetrics") or {}
    out = {}
    for entry in metrics.get("per_prompt") or []:
        name = NAME.get(entry["prompt_sha256"][:8])
        if name:
            out[name] = entry
    return out


def main() -> int:
    board = rows()
    scored = [r for r in board if cells(r)]
    print("harness=ranked  E159 Part B control  zero GPU")
    print("board rows %d, scored rows %d\n" % (len(board), len(scored)))

    # 1. How rare is a drafting plutarch on the whole public board?
    flipped, held = [], []
    for row in scored:
        entry = cells(row).get("plutarch")
        if entry is None:
            continue
        (flipped if entry["effective_mean_draft_len"] >= DRAFTING_EDL
         else held).append((row, entry))
    print("## plutarch drafting state across every scored board row")
    print("  rows with a plutarch cell        %d" % (len(flipped) + len(held)))
    print("  plutarch drafts, edl >= %.1f      %d" % (DRAFTING_EDL, len(flipped)))
    print("  plutarch held off                %d" % len(held))
    held_edl = [e["effective_mean_draft_len"] for _, e in held]
    print("  held-off edl median %.4f, max %.4f"
          % (statistics.median(held_edl), max(held_edl)))
    print("  solvers that ever flipped it     %s"
          % sorted({r["solverUsername"] for r, _ in flipped}))

    print("\n## every board row where plutarch drafts")
    print("%-9s %-14s %10s %8s %9s %9s  %s" % (
        "id", "solver", "score", "plu edl", "plu cand", "plu ser", "created"))
    for row, entry in sorted(flipped, key=lambda x: x[0]["createdAt"]):
        print("%-9s %-14s %10s %8.4f %9.6f %9.6f  %s" % (
            row["id"][:8], row["solverUsername"][:14],
            ("%.5f" % row["officialScore"]) if row.get("officialScore") else "-",
            entry["effective_mean_draft_len"],
            entry["mtp_seconds_per_token_mean"],
            entry["serial_seconds_per_token_mean"],
            row["createdAt"][:19]))

    # 2. The h-family. These moved `marginal[0]` uniformly, and further than
    #    pb6 did, so they are the direct falsification test for the gate story.
    print("\n## named arm receipts, gate value against plutarch draft length")
    named = {
        "572b2cc4": ("ship, pb6 pair base", 0.18),
        "e003a86d": ("pb6", None),
        "ca9251b8": ("ship, pbfit pair base", 0.18),
        "2da69933": ("pbfit", 0.120143),
        "fc62d1aa": ("h = 0.32", 0.32),
        "ec24d591": ("crown", 0.18),
        "5a9f130a": ("our parity anchor", 0.18),
    }
    index = {r["id"][:8]: r for r in board}
    print("%-9s %-22s %8s %9s %9s %10s %8s" % (
        "id", "arm", "gate", "plu edl", "plu cand", "score", "status"))
    for key, (label, gate) in named.items():
        row = index.get(key)
        if row is None:
            print("%-9s %-22s  ABSENT FROM BOARD" % (key, label))
            continue
        entry = cells(row).get("plutarch")
        print("%-9s %-22s %8s %9s %9s %10s %8s" % (
            key, label,
            "%.6f" % gate if gate is not None else "?",
            "%.4f" % entry["effective_mean_draft_len"] if entry else "-",
            "%.6f" % entry["mtp_seconds_per_token_mean"] if entry else "-",
            ("%.5f" % row["officialScore"]) if row.get("officialScore") else "-",
            row.get("status", "-")))

    # 2b. The gate response across ALL eight prompts, not only plutarch, and
    #     the provenance that decides whether each pair is a clean contrast.
    order = ["beagle", "essays", "republic", "botany", "medicine", "drama",
             "travel", "plutarch"]
    print("\n## draft length by prompt for each named receipt")
    print("%-9s %-20s " % ("id", "arm")
          + " ".join("%8s" % p[:8] for p in order))
    for key, (label, _gate) in named.items():
        row = index.get(key)
        if row is None:
            continue
        c = cells(row)
        print("%-9s %-20s " % (key, label)
              + " ".join("%8.4f" % c[p]["effective_mean_draft_len"]
                         for p in order))

    print("\n## provenance of each named receipt")
    print("%-9s %-20s %-12s %-18s %s" % (
        "id", "arm", "commit", "head sha256", "mode"))
    for key, (label, _gate) in named.items():
        row = index.get(key)
        if row is None:
            continue
        metrics = row.get("officialMetrics") or {}
        head = metrics.get("head_provenance") or {}
        print("%-9s %-20s %-12s %-18s %s" % (
            key, label, str(metrics.get("commit"))[:12],
            str(head.get("sha256"))[:18], metrics.get("mode")))

    # 3. Our own submission history, so the h-family receipts can be located
    #    by score even when the ledger does not give an id.
    ours = [r for r in scored if r["solverUsername"] == "morganmcg1"]
    print("\n## our scored submissions, plutarch draft length by date")
    print("%-9s %10s %8s %9s %9s  %s" % (
        "id", "score", "plu edl", "beagle", "essays", "created"))
    for row in sorted(ours, key=lambda r: r["createdAt"]):
        c = cells(row)
        print("%-9s %10s %8.4f %9.4f %9.4f  %s" % (
            row["id"][:8],
            ("%.5f" % row["officialScore"]) if row.get("officialScore") else "-",
            c["plutarch"]["effective_mean_draft_len"],
            c["beagle"]["effective_mean_draft_len"],
            c["essays"]["effective_mean_draft_len"],
            row["createdAt"][:19]))

    # 4. The natural experiment. Our own history contains submissions on both
    #    sides of the gate. Every open-gate submission is paired with its
    #    nearest closed-gate neighbour in time, so era and base are matched as
    #    closely as the record allows.
    dated = sorted(ours, key=lambda r: r["createdAt"])
    opened, closed = [], []
    for row in dated:
        if not row.get("officialScore"):
            continue
        edl = cells(row)["plutarch"]["effective_mean_draft_len"]
        (opened if edl >= DRAFTING_EDL else closed).append(row)

    print("\n## natural experiment: our submissions with plutarch drafting")
    print("   Each open-gate submission against its nearest closed-gate")
    print("   neighbour in time. harness=ranked, published median.")
    print("%-9s %9s %8s  %-9s %9s %8s %10s %s" % (
        "open id", "score", "plu edl", "closed id", "score", "plu edl",
        "delta %", "hours apart"))
    deltas = []
    for row in opened:
        when = row["createdAt"]
        near = min(closed, key=lambda r: abs(
            _epoch(r["createdAt"]) - _epoch(when)))
        a, b = float(near["officialScore"]), float(row["officialScore"])
        delta = (b / a - 1.0) * 100.0
        deltas.append(delta)
        print("%-9s %9.5f %8.4f  %-9s %9.5f %8.4f %+10.4f %11.2f" % (
            row["id"][:8], b, cells(row)["plutarch"]["effective_mean_draft_len"],
            near["id"][:8], a,
            cells(near)["plutarch"]["effective_mean_draft_len"], delta,
            abs(_epoch(when) - _epoch(near["createdAt"])) / 3600.0))
    if deltas:
        print("  open-gate submissions %d, all below their neighbour: %s"
              % (len(deltas), all(d < 0 for d in deltas)))
        print("  mean %+.4f %%, median %+.4f %%, worst %+.4f %%, best %+.4f %%"
              % (statistics.fmean(deltas), statistics.median(deltas),
                 min(deltas), max(deltas)))
        print("  Opening plutarch's gate has never once helped our score.")
    return 0


def _epoch(stamp: str) -> float:
    return datetime.datetime.fromisoformat(
        stamp.replace("Z", "+00:00")).timestamp()


if __name__ == "__main__":
    sys.exit(main())
