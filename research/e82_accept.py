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
        if row["kind"] != "draft":
            continue
        entry = by_round.setdefault(row["round"], {"drafts": {}, "accepted_count": 0})
        entry["drafts"][row["draft_index"] + 1] = row
        entry["accepted_count"] += bool(row["accepted"])

    cells, rounds = {}, {}
    base = 0
    for index in sorted(by_round):
        entry = by_round[index]
        rounds[(seed, index)] = {
            "base": base,
            "accepted_count": entry["accepted_count"],
            "drafted": len(entry["drafts"]),
        }
        for depth, row in entry["drafts"].items():
            # ONLY the golden-trajectory rows are comparable. Once a draft is
            # rejected, every deeper draft in that round continues the
            # candidate's own wrong tokens, so the harness checks it with
            # `verify_block_replay` against a replayed reference instead of a
            # golden row. Those rows describe a trajectory that is unique to the
            # arm, so pairing them across arms would compare different questions.
            if row["reference_checked_by"] != "serial_golden":
                continue
            cells[(seed, base, depth)] = {
                "accepted": bool(row["accepted"]),
                "reference_margin": row["reference_margin"],
                "reference_token": row["reference_token"],
            }
        base += entry["accepted_count"] + 1
    return cells, rounds


def difficulty(seed: str, steps: int) -> dict[int, float]:
    """Per-golden-row top1-top2 margin, straight from the reference rows.

    Row `j` predicts emission index `j + 1`; emission index 0 is the seed argmax
    and carries no row (`QwenRuntimeMTPDriver.swift:43-50`). A draft at depth `d`
    in a round based at `base` therefore lands on golden row `base + d - 1`.
    """
    rows = json.loads((CACHE / "reference" / f"{seed}_{steps}.json").read_text())["rows"]
    out = {}
    for position, row in enumerate(rows):
        logits = row["top2_logits"]
        out[position] = logits[0] - logits[1] if len(logits) >= 2 else float("inf")
    return out


def validate_pairing(cells: dict, margins: dict, steps: int) -> dict:
    """Check the reconstructed base against the golden the runtime itself used.

    Every ledger draft row carries the `reference_margin` and `reference_token`
    the trusted driver read out of the golden. If `base + depth - 1` is the right
    golden row, those two fields must reproduce that row exactly. This is a
    live check on real data, so it is far stronger than the synthetic self-test.
    """
    goldens = {}
    checked = mismatched = 0
    examples = []
    for (seed, base, depth), fact in cells.items():
        if seed not in goldens:
            goldens[seed] = json.loads(
                (CACHE / "reference" / f"{seed}_{steps}.json").read_text()
            )["rows"]
        rows = goldens[seed]
        index = base + depth - 1
        checked += 1
        if index >= len(rows):
            mismatched += 1
            examples.append({"seed": seed, "row": index, "why": "row index past the golden"})
            continue
        row = rows[index]
        want = row["top2_logits"][0] - row["top2_logits"][1]
        if row["sequential_argmax"] != fact["reference_token"] or abs(
            want - fact["reference_margin"]
        ) > 1e-6:
            mismatched += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "seed": seed,
                        "row": index,
                        "golden_token": row["sequential_argmax"],
                        "ledger_token": fact["reference_token"],
                        "golden_margin": want,
                        "ledger_margin": fact["reference_margin"],
                    }
                )
    return {"cells_checked": checked, "mismatched": mismatched, "examples": examples}


def is_tie(fact: dict) -> bool:
    """The golden's own argmax is arbitrary when its top two logits are equal.

    A head that proposes the other tied token is then marked wrong by a coin
    flip, so such a cell is neither an accept nor a reject and is dropped from
    every rate. Its reach is still reported, so the exclusion stays visible.
    """
    return fact["reference_margin"] <= 0.0


def profile(cells: dict, keep: set) -> dict:
    per_depth, ties = {}, {}
    for depth in DEPTHS:
        at = [v for k, v in cells.items() if k[2] == depth and k in keep]
        scored = [v["accepted"] for v in at if not is_tie(v)]
        per_depth[depth] = wilson(sum(scored), len(scored))
        ties[depth] = sum(1 for v in at if is_tie(v))
    at = [v for k, v in cells.items() if k[2] in RULE_DEPTHS and k in keep]
    scored = [v["accepted"] for v in at if not is_tie(v)]
    tied = [v for v in at if is_tie(v)]
    return {
        "per_depth": per_depth,
        "pooled_3_6": wilson(sum(scored), len(scored)),
        "ties_by_depth": ties,
        "ties_3_6": len(tied),
        "tie_fraction_3_6": len(tied) / len(at) if at else 0.0,
        # Reported because a tie the candidate happens to match is scored as an
        # accept by the harness even though the reference token was arbitrary.
        "tied_cells_the_arm_matched_3_6": sum(1 for v in tied if v["accepted"]),
    }


def work(payload: dict, rounds: dict, seeds: list[str], steps: int) -> dict:
    per_seed_rounds = {s: payload[s]["round_count"] for s in seeds}
    drafted = {s: 0 for s in seeds}
    accepted = {s: 0 for s in seeds}
    for (seed, _), r in rounds.items():
        drafted[seed] += r["drafted"]
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
        rx, ry = ref.get(key), cand.get(key)
        if rx is None or ry is None or is_tie(rx):
            continue
        x, y = rx["accepted"], ry["accepted"]
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
    def draft(r, i, a, checked="serial_golden"):
        return {
            "round": r, "kind": "draft", "draft_index": i, "accepted": a,
            "reference_checked_by": checked, "reference_margin": 0.0, "reference_token": 0,
        }

    # Round 1 drafts twice, is rejected at depth 1, and so has its depth-2 row
    # checked by replay instead of the golden. That row must not become a cell.
    ledger = [
        draft(0, 0, True), draft(0, 1, True), draft(0, 2, False), {"round": 0, "kind": "targetTail"},
        draft(1, 0, False), draft(1, 1, False, "verify_block_replay"),
        {"round": 1, "kind": "targetTail"},
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
    assert [cells[k]["accepted"] for k in sorted(cells)] == [
        True, True, False, False, True, True, True, True
    ]
    print("selftest: round-base reconstruction and cell keys OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reference", default="declared")
    ap.add_argument("--candidates", default="soup-q4,qat-q4,master-bf16,kamciosz,pinned")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--report", default="research/e82-accept.json")
    ap.add_argument("--exclude-latched", action="store_true")
    ap.add_argument("--latch-threshold", type=float, default=0.05)
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

    # `costModelDepth` returns 0 whenever `positionAcceptEMA[0]` falls under the
    # marginal price, and a round that drafts nothing produces no acceptance
    # observation to raise that EMA again. The schedule therefore has a
    # near-absorbing non-drafting state that a run enters on early bad luck.
    # It is bimodal in practice -- a seed is at 0.000 or above 0.5 -- and it is
    # head-independent, so it is a confound and not an arm effect. Cells are
    # already conditioned on a round having drafted, but a latched arm drafts
    # only where it stayed confident, which biases its conditional acceptance
    # upward. Report the latch and offer the unlatched subset as the clean read.
    latch = {
        arm: {s: payloads[arm][s]["non_drafting_round_count"] / payloads[arm][s]["round_count"]
              for s in seeds}
        for arm in payloads
    }
    latched_seeds = sorted(
        s for s in seeds if max(latch[a][s] for a in payloads) > args.latch_threshold
    )
    if args.exclude_latched:
        seeds = [s for s in seeds if s not in latched_seeds]
        if not seeds:
            raise SystemExit("every seed latched on some arm")

    margins = {s: difficulty(s, args.steps) for s in seeds}
    cells, rounds = {}, {}
    for arm, found in payloads.items():
        cells[arm], rounds[arm] = {}, {}
        for seed in seeds:
            c, r = cells_of(found[seed], seed)
            cells[arm].update(c)
            rounds[arm].update(r)

    # The reconstructed base is what every paired comparison rests on, so check
    # it against the golden the trusted driver itself read, for every arm.
    pairing = {a: validate_pairing(cells[a], margins, args.steps) for a in cells}
    bad = {a: v for a, v in pairing.items() if v["mismatched"]}
    if bad:
        raise SystemExit(f"round-base reconstruction disagrees with the golden: {bad}")

    # Tercile cuts come from the reference rows over every predicted position,
    # so they describe the corpus and not any arm's round population.
    pool = sorted(m for s in seeds for m in margins[s].values())
    lo_cut, hi_cut = pool[len(pool) // 3], pool[2 * len(pool) // 3]

    def cell_margin(key):
        seed, base, depth = key
        return margins[seed].get(base + depth - 1)

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
                name: dict(Counter(s for s in seeds for m in margins[s].values() if pick(m)))
                for name, pick in (
                    ("hardest", lambda m: m <= lo_cut),
                    ("easiest", lambda m: m >= hi_cut),
                )
            },
            "per_seed_median_margin": {
                s: statistics.median(list(margins[s].values())) for s in seeds
            },
            "tied_positions": sum(1 for m in pool if m <= 0.0),
            "tied_position_fraction": sum(1 for m in pool if m <= 0.0) / len(pool),
            "near_tie_fraction_under_0p5": sum(1 for m in pool if m < 0.5) / len(pool),
        },
        "pairing_validation": pairing,
        "schedule_latch": {
            "non_drafting_round_fraction": latch,
            "threshold": args.latch_threshold,
            "latched_seeds": latched_seeds,
            "excluded": bool(args.exclude_latched),
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
    diff = report["difficulty"]
    print(
        f"difficulty over {diff['positions']} predicted positions:"
        f" hardest margin <= {cuts['hardest_max']:.4f}, easiest >= {cuts['easiest_min']:.4f}"
    )
    print(
        f"exact ties (margin 0, reference argmax arbitrary): {diff['tied_positions']}"
        f" = {100 * diff['tied_position_fraction']:.2f} % of positions;"
        f" margin < 0.5: {100 * diff['near_tie_fraction_under_0p5']:.2f} %"
    )

    latch = report["schedule_latch"]
    frac = latch["non_drafting_round_fraction"]
    arms = list(report["arms"])
    print(
        f"\n=== schedule latch: non-drafting round fraction"
        f" (latched > {latch['threshold']:.2f},"
        f" {'excluded' if latch['excluded'] else 'kept'}) ==="
    )
    print("seed".ljust(26) + "".join(a.rjust(14) for a in arms))
    for seed in sorted(frac[arms[0]]):
        row = "".join(f"{frac[a][seed]:14.3f}" for a in arms)
        print(seed.ljust(26) + row)
    print(f"latched seeds: {latch['latched_seeds'] or 'none'}")

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
        print("  pooled d3-d6 (ties excluded):")
        for arm, entry in report["arms"].items():
            s = entry["splits"][split]
            c = s["pooled_3_6"]
            if c["reached"]:
                print(
                    f"    {arm.ljust(13)} {100*c['p']:6.2f} % "
                    f"[{100*c['lo']:.2f}, {100*c['hi']:.2f}] n={c['reached']}"
                    f"  ties dropped {s['ties_3_6']}"
                    f" ({100*s['tie_fraction_3_6']:.2f} %)"
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
