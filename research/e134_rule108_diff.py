#!/usr/bin/env python3
"""E134 item 5 -- the Rule 108 deletion diff for the pb6 archive.

    usage: research/e134_rule108_diff.py [--board PATH] [--json PATH]

Run from the repository ROOT. Zero GPU.

A Yukon archive replaces every required path in `editablePaths`, so the
submitted tree DELETES every mechanism the live frontier holds that our tree
does not. Rule 108 requires that deletion set to be enumerated and priced from
our own instrument before the archive goes out.

The frontier lineage and our lineage forked. This script reads the live board
cache, reconstructs both chains from the promoted rows, and prints the
mechanisms on each side with the price our own measurements give them. It
does NOT read rival source: `yukon reset` is forbidden in this maintained
checkout, so every rival mechanism here is identified from its own public note
and priced only from evidence this campaign produced.
"""

from __future__ import annotations

import argparse
import json
import pathlib

BOARD = "/tmp/yukon-board/full.json"

OUR_CROWN = "623e77af"
OUR_SOURCE = "60d5b34a"

# Mechanisms the live frontier tree holds and our tree does not. Every price
# is from this campaign's own instrument, never from a rival's published
# median (Rule 62, Rule 63).
FRONTIER_ONLY = [
    {
        "mechanism": "qL{2,3} later-window SDPA warm",
        "credited": "noskillcoding 48423d09, restored by nagaral 0b8602e1",
        "our_file": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "our_price_pct": 0.0,
        "our_price_source":
            "FINDING 177 measured it as a null on our own ranked pair. "
            "FINDING 178 explains the null at source: the MLX SDPA pipeline "
            "cache key is kname + mask flag + qt|qnt + c|nc + sinks|nosinks, "
            "and qL never reaches any of those fields. E134 rung 6 confirms "
            "it from the other side: the warm query and both scored chunk "
            "expressions resolve to _qnt at every width 1 to 9, so the "
            "shipped [1, 5, 4] warm already compiles every specialisation "
            "[1, 2, 3, 4, 5] would.",
        "confidence": "high, three independent lines agree",
    },
    {
        "mechanism": "E87 probe-select",
        "credited": "francip, restored by nagaral 0b8602e1",
        "our_file": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
        "our_price_pct": 0.064,
        "our_price_source":
            "advisor F7 section 2 rival-mechanism audit, priced F83-weighted "
            "from our own instrument. Real but tiny. The rival's own note "
            "claims +0.72 to +0.88 percent from published medians, which "
            "Rule 62 and Rule 63 forbid us from using as a price.",
        "confidence": "medium, our instrument and the rival receipt disagree "
                      "by an order of magnitude",
    },
    {
        "mechanism": "E020 replay-prefetch",
        "credited": "Amal-David, restored by nagaral 0b8602e1",
        "our_file": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "our_price_pct": -0.4767,
        "our_price_source":
            "E134 item 0 measured this mechanism directly as the W-PREFETCH "
            "arm: +0.4767 percent +- 0.0963 SLOWER on absolute candidate "
            "seconds per token, worse on 4 of 4 pairs, and +15.41 percent on "
            "width-matched round-1 excess. Local M4 Pro, ABBA-counterbalanced, "
            "ungated. The rival holds a +0.033 percent ranked increment, so "
            "this is a host disagreement, not a refutation of their receipt.",
        "confidence": "medium, our host measures a regression where the "
                      "ranked M5 measured a small gain",
    },
    {
        "mechanism": "normed-verify warm (callWithHiddenAndNormed)",
        "credited": "newjordan 1d7876fd, restored by nagaral 0b8602e1",
        "our_file": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "our_price_pct": -0.0293,
        "our_price_source":
            "E134 item 0 measured this mechanism directly as the W-NORM arm: "
            "+0.0293 percent +- 0.0513 slower on absolute candidate seconds "
            "per token and +24.87 percent +- 3.54 WORSE on width-matched "
            "round-1 excess, worse on 4 of 4 pairs on both metrics. Rule 110 "
            "predicts the null: the warm normed call changes no pipeline "
            "cache key.",
        "confidence": "high on round 1, low on the end-to-end mean, which is "
                      "inside its own spread",
    },
]

# Mechanisms our tree holds and the live frontier tree does not.
OURS_ONLY = [
    {
        "mechanism": "one-pass QMV width specialisation, table {6:6, 7:7}",
        "credited": "senpai, promoted as 623e77af on source 60d5b34a",
        "our_price_pct": 0.1206,
        "our_price_source":
            "our own promoted pair: cf79f7df 3.51661724 to 623e77af "
            "3.52085227 on the published median.",
    },
    {
        "mechanism": "pb6 pass-boundary depth price, width 6 at tier 1.45",
        "credited": "senpai E134, not yet submitted",
        "our_price_pct": 2.4683,
        "our_price_source":
            "E134 item 2, replayed ranked median percent, leave-one-prompt-out "
            "held out on the measured post-arm curve. Rule 79: no ranked "
            "receipt yet.",
    },
]


def load_board(path: str) -> list[dict]:
    data = json.loads(pathlib.Path(path).read_text())
    return data["submissions"] if isinstance(data, dict) else data


def promoted_chain(rows: list[dict]) -> list[dict]:
    chain = [r for r in rows
             if r.get("promotionStatus") == "promoted" and r.get(
                 "officialScore")]
    chain.sort(key=lambda r: r["officialScore"], reverse=True)
    return chain


def short(value) -> str:
    return str(value)[:8] if value else "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=BOARD)
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = load_board(args.board)
    chain = promoted_chain(rows)
    frontier = chain[0]
    ours = next((r for r in chain if r["id"].startswith(OUR_CROWN)), None)

    print("live promoted rows, highest score first")
    print("%-10s %-14s %12s  %-24s %s"
          % ("id", "solver", "score", "promoted", "source"))
    for row in chain[:6]:
        print("%-10s %-14s %12.8f  %-24s %s"
              % (short(row["id"]), row["solverUsername"],
                 row["officialScore"], row.get("promotionFinishedAt"),
                 short(row.get("promotedSourceRef"))))

    print()
    print("frontier tree   %s  %s  %.8f"
          % (short(frontier.get("promotedSourceRef")),
             frontier["solverUsername"], frontier["officialScore"]))
    if ours is not None:
        print("our crown       %s  %s  %.8f"
              % (short(ours.get("promotedSourceRef")),
                 ours["solverUsername"], ours["officialScore"]))
        print("gap to frontier %+0.8f  (%+0.4f percent)"
              % (frontier["officialScore"] - ours["officialScore"],
                 100.0 * (frontier["officialScore"] / ours["officialScore"]
                          - 1.0)))

    print()
    print("RULE 108 -- mechanisms a pb6 archive would DELETE")
    print("`held value` is what the mechanism is worth WHILE PRESENT, so a")
    print("negative number means deleting it GAINS that much.")
    print("%-46s %12s  %s" % ("mechanism", "held value", "confidence"))
    held = 0.0
    for entry in FRONTIER_ONLY:
        held += entry["our_price_pct"]
        print("%-46s %+11.4f%%  %s"
              % (entry["mechanism"], entry["our_price_pct"],
                 entry["confidence"].split(",")[0]))
    print("%-46s %+11.4f%%" % ("held value of the deleted set", held))
    print("%-46s %+11.4f%%" % ("effect of deleting them", -held))

    print()
    print("mechanisms a pb6 archive would ADD to the frontier tree")
    added = 0.0
    for entry in OURS_ONLY:
        added += entry["our_price_pct"]
        print("%-46s %+11.4f%%" % (entry["mechanism"], entry["our_price_pct"]))
    print("%-46s %+11.4f%%" % ("effect of adding them", added))

    # Conservative reading. Two of the four deletions are priced from one
    # ungated local M4 Pro session that disagrees with a ranked M5 receipt.
    # Credit no gain from shedding them and keep every loss.
    conservative = sum(min(0.0, -e["our_price_pct"]) for e in FRONTIER_ONLY)
    print()
    print("%-46s %+11.4f%%" % ("net, our instrument", added - held))
    print("%-46s %+11.4f%%"
          % ("net, conservative deletion accounting", added + conservative))

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "harness": "ranked-model",
            "board_snapshot": args.board,
            "frontier": {
                "id": frontier["id"], "solver": frontier["solverUsername"],
                "score": frontier["officialScore"],
                "source_ref": frontier.get("promotedSourceRef"),
                "promoted_at": frontier.get("promotionFinishedAt"),
            },
            "our_crown": None if ours is None else {
                "id": ours["id"], "score": ours["officialScore"],
                "source_ref": ours.get("promotedSourceRef"),
            },
            "deletes": FRONTIER_ONLY,
            "adds": OURS_ONLY,
            "held_value_of_deleted_set_pct": held,
            "effect_of_deleting_them_pct": -held,
            "effect_of_adding_them_pct": added,
            "net_pct": added - held,
            "net_conservative_pct": added + conservative,
            "source_diff_available": False,
            "source_diff_note":
                "yukon reset is forbidden in this maintained checkout, so no "
                "rival source tree was read. Every rival mechanism is named "
                "from its own public note and priced only from this "
                "campaign's own measurements.",
        }, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
