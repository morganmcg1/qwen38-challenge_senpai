#!/usr/bin/env python3
"""E140 F2: CAMPAIGN RULE 116, answered on the receipt and on every arm.

`harness=local instrument`. Zero GPU.

The advisor asks two questions and imposes one rule.

Question 1, how `e134_replayed_ranked_median_pct` was computed. The answer is
in `e134_rung2.median_pct`, which this file re-implements independently and
then compares against, so the answer is a measurement rather than a claim. No
F83 weight enters the score path in either implementation.

Question 2, the per-prompt replayed deltas for `pb6` on the candidate leg, and
whether essays crosses republic. Both are read out of `cells.json`.

FINDING 196 is checked directly on the live board rather than accepted: for
every row, the sorted fourth and fifth prompts are identified and the
median-pair identity is compared with the published official score.

Usage:
  python3 e140_rule116.py --cells e140-artifacts/cells.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import PROMPT_NAMES, load_board_receipt, median_of  # noqa: E402
from e134_rung2 import median_pct  # noqa: E402


def raw_ratios(receipt, ratios=None) -> dict:
    """`serial / candidate` per prompt, with an arm's candidate-time ratio."""
    out = {}
    for prompt, entry in receipt["per_prompt"].items():
        scale = 1.0 if ratios is None else ratios.get(prompt, 1.0)
        out[prompt] = entry["serial"] / (entry["candidate"] * scale)
    return out


def median_pair(raws) -> tuple[float, str, str]:
    """The published median, and the two prompts that are its only inputs."""
    order = sorted(raws, key=raws.get)
    fourth, fifth = order[3], order[4]
    return 0.5 * (raws[fourth] + raws[fifth]), fourth, fifth


def headroom(raws) -> dict:
    """How far the fourth and fifth prompts are from their next crossing.

    The fifth prompt stops paying once it passes the sixth, and the fourth
    stops paying half its rate once it passes the fifth. Both are the caps a
    weighted sum cannot represent, which is why Rule 116 exists.
    """
    order = sorted(raws, key=raws.get)
    fourth, fifth, sixth = order[3], order[4], order[5]
    return {
        "fourth": fourth, "fifth": fifth, "sixth": sixth,
        "fifth_to_sixth_pct": (raws[sixth] / raws[fifth] - 1.0) * 100.0,
        "fourth_to_fifth_pct": (raws[fifth] / raws[fourth] - 1.0) * 100.0,
        "fourth_raised_to_fifth_pct": (
            (0.5 * (raws[fifth] + raws[fifth])
             / (0.5 * (raws[fourth] + raws[fifth]))) - 1.0) * 100.0,
    }


def board_identity(path: pathlib.Path) -> list[dict]:
    """FINDING 196, checked against every board row that publishes prompts."""
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows["submissions"]
    out = []
    for row in rows:
        metrics = row.get("officialMetrics") or {}
        entries = metrics.get("per_prompt") or []
        if len(entries) != 8:
            continue
        raws = {}
        for entry in entries:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8],
                                    entry["prompt_sha256"][:8])
            raws[name] = entry["raw_ratio_of_means"]
        value, fourth, fifth = median_pair(raws)
        out.append({
            "id": row["id"][:8], "solver": row.get("solverUsername"),
            "status": row.get("status"), "official": row.get("officialScore"),
            "identity": value, "fourth": fourth, "fifth": fifth,
            "error": (value - row["officialScore"])
            if row.get("officialScore") else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    # `d3c491b5` is the E128 replay base and the continuity point for the
    # published `+2.4683`. `623e77af` is our promoted tree and `08b67f12` is
    # the live crown. The three do not share a fourth and fifth prompt, and
    # that is the whole point of reporting all three.
    ap.add_argument("--receipts", default="d3c491b5,623e77af,08b67f12")
    ap.add_argument("--cells", type=pathlib.Path,
                    default=HERE / "e140-artifacts/cells.json")
    ap.add_argument("--form", default="per_drafting_round")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/rule116.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 F2 Rule 116  zero GPU")

    print("\n## FINDING 196, checked on every board row with eight prompts")
    print("%-10s %-14s %-11s %12s %12s %10s  %s"
          % ("id", "solver", "status", "official", "median pair", "error",
             "4th / 5th"))
    rows = board_identity(args.board)
    for row in rows:
        print("%-10s %-14s %-11s %12.8f %12.8f %10.2e  %s / %s"
              % (row["id"], row["solver"], row["status"],
                 row["official"] or float("nan"), row["identity"],
                 abs(row["error"]) if row["error"] is not None
                 else float("nan"), row["fourth"], row["fifth"]))
    worst = max((abs(r["error"]) for r in rows if r["error"] is not None),
                default=float("nan"))
    print("largest absolute identity error across %d rows  %.3e"
          % (len(rows), worst))

    blob = (json.loads(args.cells.read_text())["cells"]
            if args.cells.exists() else {})
    arms = {cell.split("|")[1]: entry["per_prompt_ratio"]
            for cell, entry in sorted(blob.items())
            if cell.split("|")[0] == args.form}
    if not arms:
        print("\n%s holds no arm for form %s; arm sections skipped"
              % (args.cells, args.form))

    payload = {"harness": "local instrument", "form": args.form,
               "finding196_rows": rows, "finding196_worst_error": worst,
               "receipts": {}}
    for name in args.receipts.split(","):
        receipt = load_board_receipt(args.board, name)
        base = raw_ratios(receipt)
        base_median, fourth, fifth = median_pair(base)
        names = sorted(base, key=base.get)
        room = headroom(base)
        print("\n## receipt %s (%s, official %.8f)"
              % (receipt["id"][:8], receipt["solver"], receipt["score"]))
        print("%-10s %12s %12s %14s" % ("prompt", "serial s/tok",
                                        "cand s/tok", "raw ratio"))
        for prompt in names:
            entry = receipt["per_prompt"][prompt]
            print("%-10s %12.6f %12.6f %14.6f"
                  % (prompt, entry["serial"], entry["candidate"],
                     base[prompt]))
        print("median pair %s + %s -> %.8f   published %.8f"
              % (fourth, fifth, base_median, receipt["score"]))
        print("headroom: %s to %s %+.4f %%   %s to %s %+.4f %%   raising %s "
              "alone to %s's level is worth %+.4f %%"
              % (room["fifth"], room["sixth"], room["fifth_to_sixth_pct"],
                 room["fourth"], room["fifth"], room["fourth_to_fifth_pct"],
                 room["fourth"], room["fifth"],
                 room["fourth_raised_to_fifth_pct"]))
        if not arms:
            payload["receipts"][name] = {
                "score": receipt["score"], "raw_ratios": base,
                "median": base_median, "headroom": room}
            continue

        # The marginal weights of FINDING 196. They are exact for an
        # infinitesimal order-preserving move and for nothing else, which is
        # what the `error pp` column measures.
        weights = {fourth: 0.5 * base[fourth] / base_median,
                   fifth: 0.5 * base[fifth] / base_median}
        print("\n   per-prompt raw gain %%, then the median three ways")
        print("   %-12s %s | %9s %9s %8s | %-18s %10s"
              % ("cell", " ".join("%9s" % n for n in names), "sorted",
                 "weighted", "err pp", "pair after", "5th->6th"))
        per_arm = {}
        for cell, ratios in arms.items():
            raws = raw_ratios(receipt, ratios)
            value, new_fourth, new_fifth = median_pair(raws)
            gains = {p: (raws[p] / base[p] - 1.0) * 100.0 for p in names}
            direct = (value / base_median - 1.0) * 100.0
            weighted = sum(w * gains[p] for p, w in weights.items())
            cross = median_pct(receipt,
                               {p: {"ratio": r} for p, r in ratios.items()})
            after = headroom(raws)
            per_arm[cell] = {
                "median_pct_sorted": direct, "median_pct_harness": cross,
                "median_pct_f196_weighted": weighted,
                "f196_error_pp": weighted - direct,
                "per_prompt_raw_gain_pct": gains,
                "per_prompt_candidate_ratio": ratios,
                "fourth": new_fourth, "fifth": new_fifth,
                "reordered": (new_fourth != fourth or new_fifth != fifth),
                "headroom": after,
            }
            print("   %-12s %s | %+9.4f %+9.4f %+8.4f | %-18s %+10.4f"
                  % (cell, " ".join("%+9.4f" % gains[n] for n in names),
                     direct, weighted, weighted - direct,
                     "%s + %s" % (new_fourth, new_fifth),
                     after["fifth_to_sixth_pct"]))

        gap = max(abs(v["median_pct_sorted"] - v["median_pct_harness"])
                  for v in per_arm.values())
        print("   Rule 116 cross-check, independent sort against the "
              "harness: largest gap %.3e pp" % gap)
        if gap > 1e-9:
            raise SystemExit("the harness and an independent sort disagree")
        payload["receipts"][name] = {
            "score": receipt["score"], "raw_ratios": base,
            "median": base_median, "headroom": room,
            "f196_weights": weights, "arms": per_arm}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
