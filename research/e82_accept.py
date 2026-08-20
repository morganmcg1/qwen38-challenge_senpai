#!/usr/bin/env python3
"""E82 rung 0: read the acceptance screen and apply the stop rule.

Every arm replays the SAME serial reference rows, so every arm emits the same
512 tokens and the committed prefix at any emitted position is identical
across arms. That is what makes a paired comparison possible at all.

The unit of comparison
----------------------
A CELL is `(seed, round_base, depth)`. `round_base` is the emitted index the
round started from, and `depth` is the 1-based draft position inside it. A cell
asks one well-posed question:

  the head resumed at committed position `round_base`, proposed `depth`
  tokens autoregressively, and every earlier proposal in this round matched
  the truth -- is proposal `depth` also correct?

Two arms that both produce a cell at the same `(seed, round_base, depth)` faced
an identical true prefix and an identical number of self-proposed steps, so the
cell is genuinely paired. Round INDEX is not a valid key: acceptance differs
between arms, so their round boundaries drift apart after the first
disagreement.

Acceptance is a prefix property -- the round takes the longest correct prefix --
so p_d is a SURVIVAL curve, P(accepted_count >= d | the round drafted d). The
per-step conditional rate is p_d / p_(d-1).

The schedule is adaptive, and that confounds unpaired p_d
-----------------------------------------------------
`Qwen36MTPBlockSession.costModelDepth` chooses each round's depth from its own
online per-position acceptance EMA, the pending tail margin and a full-accept
streak gate, capped at 5 rows and opened to 8 after two clean rounds. A better
head drafts deeper, so it samples deep positions on rounds a worse head never
reaches, and an unpaired p_5 compares two different round populations.

So the report carries three things the schedule cannot confound:
  * `rounds_per_512`, a paired per-seed count with no reach conditioning. The
    token stream is fixed, so fewer rounds means more accepted drafts per round;
  * `rows_per_token`, the verify work those rounds cost, because a deeper round
    is a wider verify;
  * the paired McNemar table over cells both arms produced.
Unpaired p_d is still printed with its reach count beside it, so the confound
stays visible instead of being adjusted away.

Difficulty
----------
Difficulty is a property of the POSITION being predicted, taken from the
reference rows themselves: the target's top1-top2 logit margin at emitted
position `round_base + depth`. It comes from the golden, so it is identical for
every arm and no arm's own behaviour can move its own tercile assignment.

The hardest tercile is the beagle analogue. Beagle is the 4th order statistic of
the eight ranked prompts in every session above score 3.15, so it and its
neighbour set the published median; a head that wins on easy prose and loses on
hard prose moves the mean and not the median.

Usage:
  python3 research/e82_accept.py --report research/e82-accept.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e82/screen"))
DEPTHS = range(1, 9)
RULE_DEPTHS = (3, 4, 5, 6)
Z = 1.959963984540054  # 95 %


def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"accepted": 0, "reached": 0, "p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return {"accepted": k, "reached": n, "p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def cells_of(payload: dict, seed: str) -> tuple[dict, dict]:
    """Split the row ledger into paired cells and per-round facts.

    `round_base` is reconstructed by walking the rounds: a round emits its
    accepted drafts plus one committed tail token, so the next round starts
    `accepted + 1` positions later. That mirrors `emittedBaseIndex` in
    `QwenRuntimeMTPDriver`, which is not itself exported in the payload.
    """
    by_round: dict[int, dict] = {}
    for row in payload["row_ledger"]:
        entry = by_round.setdefault(row["round"], {"drafts": {}, "accepted_count": 0})
        if row["kind"] == "draft":
            depth = row["draft_index"] + 1
            entry["drafts"][depth] = bool(row["accepted"])
            entry["accepted_count"] += bool(row["accepted"])

    cells, rounds = {}, {}
    base = 0
    for index in sorted(by_round):
        entry = by_round[index]
        rounds[(seed, index)] = {"base": base, **entry}
        for depth, accepted in entry["drafts"].items():
            cells[(seed, base, depth)] = accepted
        base += entry["accepted_count"] + 1
    return cells, rounds


def difficulty(seed: str, steps: int) -> dict[int, float]:
    """Per-emitted-position top1-top2 margin, straight from the reference rows."""
    rows = json.loads((CACHE / "reference" / f"{seed}_{steps}.json").read_text())["rows"]
    out = {}
    for position, row in enumerate(rows):
        logits = row["top2_logits"]
        out[position] = logits[0] - logits[1] if len(logits) >= 2 else float("inf")
    return out


def profile(cells: dict, keep: set) -> dict:
    per_depth = {}
    for depth in DEPTHS:
        chosen = [v for (s, b, d), v in cells.items() if d == depth and (s, b, d) in keep]
        per_depth[depth] = wilson(sum(chosen), len(chosen))
    pooled = [v for (s, b, d), v in cells.items() if d in RULE_DEPTHS and (s, b, d) in keep]
    return {"per_depth": per_depth, "pooled_3_6": wilson(sum(pooled), len(pooled))}


def work(payload: dict, rounds: dict, seeds: list[str], steps: int) -> dict:
    per_seed_rounds = {s: payload[s]["round_count"] for s in seeds}
    drafted = {s: 0 for s in seeds}
    accepted = {s: 0 for s in seeds}
    for (seed, _), r in rounds.items():
        drafted[seed] += len(r["drafts"])
        accepted[seed] += r["accepted_count"]
    total_rounds = sum(per_seed_rounds.values())
    return {
        "rounds": per_seed_rounds,
        "drafted_rows": drafted,
        "accepted_rows": accepted,
        "mean_rounds_per_512": total_rounds / len(seeds),
        "mean_accepted_per_round": sum(accepted.values()) / total_rounds,
        "mean_drafted_per_round": sum(drafted.values()) / total_rounds,
        # A round evaluates depth + 1 rows: the drafts plus the committed tail.
        "rows_per_token": (sum(drafted.values()) + total_rounds) / (steps * len(seeds)),
    }


def mcnemar(ref: dict, cand: dict, keep: set) -> dict:
    b01 = b10 = both = neither = 0
    for key in keep:
        x, y = ref.get(key), cand.get(key)
        if x is None or y is None:
            continue
        if x and y:
            both += 1
        elif x and not y:
            b10 += 1
        elif y and not x:
            b01 += 1
        else:
            neither += 1
    n = b01 + b10
    chi2 = (abs(b10 - b01) - 1) ** 2 / n if n else 0.0
    return {
        "candidate_only": b01,
        "reference_only": b10,
        "both": both,
        "neither": neither,
        "discordant": n,
        "paired_cells": b01 + b10 + both + neither,
        "chi2": chi2,
        "significant_95": chi2 > 3.841459,
    }


def selftest() -> None:
    """Check the round-base reconstruction, which the pairing depends on."""
    def draft(r, i, a):
        return {"round": r, "kind": "draft", "draft_index": i, "accepted": a}

    ledger = [
        draft(0, 0, True), draft(0, 1, True), draft(0, 2, False), {"round": 0, "kind": "targetTail"},
        draft(1, 0, False), {"round": 1, "kind": "targetTail"},
        draft(2, 0, True), draft(2, 1, True), draft(2, 2, True), draft(2, 3, True),
        {"round": 2, "kind": "targetTail"},
    ]
    cells, rounds = cells_of({"row_ledger": ledger}, "s")
    # Round 0 accepts 2 and commits 1, so round 1 starts at 3; round 1 accepts
    # 0 and commits 1, so round 2 starts at 4.
    assert [r["base"] for _, r in sorted(rounds.items())] == [0, 3, 4]
    assert sorted(cells) == [
        ("s", 0, 1), ("s", 0, 2), ("s", 0, 3), ("s", 3, 1),
        ("s", 4, 1), ("s", 4, 2), ("s", 4, 3), ("s", 4, 4),
    ]
    assert [cells[k] for k in sorted(cells)] == [True, True, False, False, True, True, True, True]
    print("selftest: round-base reconstruction and cell keys OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reference", default="declared")
    ap.add_argument("--candidates", default="soup-q4,qat-q4,master-bf16,kamciosz,pinned")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--report", default="research/e82-accept.json")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    manifest = json.loads(Path("research/e82-corpus-manifest.json").read_text())
    all_seeds = [s["name"] for s in manifest["seeds"]]
    arms = [args.reference] + [a for a in args.candidates.split(",") if a]

    payloads = {}
    for arm in arms:
        found = {}
        for seed in all_seeds:
            path = CACHE / "verify" / arm / f"{seed}.json"
            if path.exists():
                found[seed] = json.loads(path.read_text())
        if found:
            payloads[arm] = found
    if args.reference not in payloads:
        raise SystemExit(f"no payload for the reference arm '{args.reference}'")
    seeds = sorted(set.intersection(*(set(v) for v in payloads.values())))
    if not seeds:
        raise SystemExit("no seed has a payload for every arm")

    margins = {s: difficulty(s, args.steps) for s in seeds}
    cells, rounds = {}, {}
    for arm, found in payloads.items():
        cells[arm], rounds[arm] = {}, {}
        for seed in seeds:
            c, r = cells_of(found[seed], seed)
            cells[arm].update(c)
            rounds[arm].update(r)

    # Tercile cuts come from the reference rows over every predicted position,
    # so they describe the corpus and not any arm's round population.
    pool = sorted(m for s in seeds for p, m in margins[s].items() if 1 <= p <= args.steps)
    lo_cut, hi_cut = pool[len(pool) // 3], pool[2 * len(pool) // 3]

    def cell_margin(key):
        seed, base, depth = key
        return margins[seed].get(base + depth)

    every = set().union(*(set(c) for c in cells.values()))
    splits = {
        "pooled": {k for k in every if cell_margin(k) is not None},
        "easiest": {k for k in every if (m := cell_margin(k)) is not None and m >= hi_cut},
        "hardest": {k for k in every if (m := cell_margin(k)) is not None and m <= lo_cut},
    }

    report = {
        "seeds": seeds,
        "steps": args.steps,
        "reference_arm": args.reference,
        "rule_depths": list(RULE_DEPTHS),
        "difficulty": {
            "signal": "reference top1-top2 logit margin at the predicted emitted position",
            "tercile_cuts": {"hardest_max": lo_cut, "easiest_min": hi_cut},
            "positions": len(pool),
            # Margin is strongly seed-correlated on this corpus, so the seed mix
            # of each tercile says how much of the split is genre and how much is
            # position-level difficulty.
            "tercile_seed_mix": {
                name: dict(
                    Counter(
                        s
                        for s in seeds
                        for p, m in margins[s].items()
                        if 1 <= p <= args.steps and pick(m)
                    )
                )
                for name, pick in (
                    ("hardest", lambda m: m <= lo_cut),
                    ("easiest", lambda m: m >= hi_cut),
                )
            },
            "per_seed_median_margin": {
                s: statistics.median([m for p, m in margins[s].items() if 1 <= p <= args.steps])
                for s in seeds
            },
        },
        "arms": {},
    }

    for arm in payloads:
        # A head that silently failed to load would still produce a payload, so
        # the provenance digest is recorded per arm rather than assumed.
        prov = [payloads[arm][s].get("head_provenance", {}) for s in seeds]
        entry = {
            "head_provenance_sha256": sorted({p.get("sha256", "?") for p in prov}),
            "head_bytes": sorted({p.get("bytes") for p in prov}),
            "head_file_count": sorted({p.get("file_count") for p in prov}),
            "parity_all_ok": all(payloads[arm][s].get("parity_all_ok") for s in seeds),
            "accepted_draft_rate": {s: payloads[arm][s].get("accepted_draft_rate") for s in seeds},
            "work": work(payloads[arm], rounds[arm], seeds, args.steps),
            "splits": {name: profile(cells[arm], ks) for name, ks in splits.items()},
        }
        report["arms"][arm] = entry

    base_entry = report["arms"][args.reference]
    for arm in payloads:
        if arm == args.reference:
            continue
        entry = report["arms"][arm]
        entry["paired_vs_reference"] = {
            name: mcnemar(cells[args.reference], cells[arm], {k for k in ks if k[2] in RULE_DEPTHS})
            for name, ks in splits.items()
        }
        pooled_delta = 100 * (
            entry["splits"]["pooled"]["pooled_3_6"]["p"] - base_entry["splits"]["pooled"]["pooled_3_6"]["p"]
        )
        hard = {
            d: 100
            * (
                entry["splits"]["hardest"]["per_depth"][d]["p"]
                - base_entry["splits"]["hardest"]["per_depth"][d]["p"]
            )
            for d in RULE_DEPTHS
        }
        entry["stop_rule"] = {
            "pooled_delta_points": pooled_delta,
            "hardest_delta_points_by_depth": hard,
            "pooled_gate_passed": pooled_delta >= 1.0,
            "hardest_gate_passed": all(v >= 0.0 for v in hard.values() if v == v),
            "advance": pooled_delta >= 1.0 and all(v >= 0.0 for v in hard.values() if v == v),
        }
        entry["rounds_delta_pct"] = 100 * (
            entry["work"]["mean_rounds_per_512"] / base_entry["work"]["mean_rounds_per_512"] - 1
        )

    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    render(report)
    print(f"\nwrote {args.report}")


def render(report: dict) -> None:
    ref = report["reference_arm"]
    cuts = report["difficulty"]["tercile_cuts"]
    print(f"seeds: {len(report['seeds'])}  reference arm: {ref}  window: {report['steps']} tokens")
    print(
        f"difficulty over {report['difficulty']['positions']} predicted positions:"
        f" hardest margin <= {cuts['hardest_max']:.4f}, easiest >= {cuts['easiest_min']:.4f}"
    )

    print("\n=== work per 512 emitted tokens (paired, no reach conditioning) ===")
    print("arm            rounds/512  vs ref   acc/round  drafted/round  rows/token  parity  head sha256")
    for arm, entry in report["arms"].items():
        w = entry["work"]
        delta = entry.get("rounds_delta_pct", 0.0)
        digest = entry["head_provenance_sha256"][0][:12] if entry["head_provenance_sha256"] else "?"
        print(
            f"{arm.ljust(13)} {w['mean_rounds_per_512']:9.2f} {delta:+7.2f}%"
            f" {w['mean_accepted_per_round']:10.3f} {w['mean_drafted_per_round']:14.3f}"
            f" {w['rows_per_token']:11.3f}  {str(entry['parity_all_ok']):5s}   {digest}"
        )

    for split in ("pooled", "easiest", "hardest"):
        print(f"\n=== {split}: acceptance survival % by depth (unpaired), reach in brackets ===")
        print("arm".ljust(13) + "".join(f"     d{d}        " for d in DEPTHS))
        for arm, entry in report["arms"].items():
            cells = []
            for d in DEPTHS:
                c = entry["splits"][split]["per_depth"][d]
                cells.append(f" {100*c['p']:5.1f} [{c['reached']:5d}]" if c["reached"] else "      -      ")
            print(arm.ljust(13) + "".join(cells))
        print("  pooled d3-d6:")
        for arm, entry in report["arms"].items():
            c = entry["splits"][split]["pooled_3_6"]
            if c["reached"]:
                print(
                    f"    {arm.ljust(13)} {100*c['p']:6.2f} % "
                    f"[{100*c['lo']:.2f}, {100*c['hi']:.2f}] n={c['reached']}"
                )

    print(f"\n=== stop rule vs {ref}: pooled >= +1.0 pt AND no hardest-tercile loss at d3-d6 ===")
    for arm, entry in report["arms"].items():
        rule = entry.get("stop_rule")
        if not rule:
            continue
        hard = " ".join(f"d{d}{v:+.2f}" for d, v in rule["hardest_delta_points_by_depth"].items())
        print(
            f"  {arm.ljust(13)} pooled {rule['pooled_delta_points']:+.2f} pt  hardest [{hard}]"
            f"  -> {'ADVANCE' if rule['advance'] else 'STOP'}"
        )
        for split in ("pooled", "hardest"):
            mc = entry["paired_vs_reference"][split]
            print(
                f"      paired {split:8s} d3-6 on {mc['paired_cells']} shared cells:"
                f" candidate-only {mc['candidate_only']}, reference-only {mc['reference_only']},"
                f" chi2 {mc['chi2']:.2f} {'significant' if mc['significant_95'] else 'not significant'}"
            )


if __name__ == "__main__":
    main()
