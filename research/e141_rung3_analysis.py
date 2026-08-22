#!/usr/bin/env python3
"""E141 rung 3: read acceptance and the arm witness out of the row ledger.

Every number here comes from the leg's own `mtp-verify` JSON, never from the
variable the leg was launched with (Rule 114). Three quantities matter.

**The witness.** A draft row whose proposed token id lies in
`[98_304, 248_044)` is impossible in the shipped arm, because that arm's
compact table has no row for it. Its presence in a leg's ledger proves the
widened table was live in that process.

**The realised gain.** `accepted_draft_total / round_count` is the mean
accepted draft length. The published-score channel is the mean accepted tokens
per round, which is that plus the one committed tail token per round.

**The truncation census.** For the SHIPPED arm alone, count the rounds whose
accepted prefix ended because the target wanted a token the shipped table
cannot propose. That is the rung 0 census measured on the live decode path
rather than on raw corpus text, so the two should agree.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SHIPPED_PREFIX = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070
VOCAB = 248_320

# Rule 116: the published median rides on these two prompts alone.
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}


def proposable_by_shipped(token: int) -> bool:
    return token < SHIPPED_PREFIX or CONTROL_START <= token < CONTROL_END


def summarise(path: Path) -> dict:
    blob = json.loads(path.read_text())
    ledger = blob.get("row_ledger") or []
    rounds: dict[int, list[dict]] = {}
    for row in ledger:
        rounds.setdefault(row["round"], []).append(row)

    drafts = [r for r in ledger if r["kind"] == "draft"]
    tails = [r for r in ledger if r["kind"] == "targetTail"]
    widened_rows = [
        r for r in drafts if not proposable_by_shipped(r["token"])
    ]

    # Rounds whose accepted prefix stopped at a token the shipped table cannot
    # propose. The reference token of the FIRST rejected draft row of a round
    # is what the target actually wanted there.
    truncating = Counter()
    truncated_rounds = 0
    scored_rounds = 0
    for _, rows in sorted(rounds.items()):
        ordered = sorted(
            (r for r in rows if r["kind"] == "draft"),
            key=lambda r: r.get("draft_index", 0),
        )
        if not ordered:
            continue
        scored_rounds += 1
        rejected = [r for r in ordered if not r["accepted"]]
        if not rejected:
            continue
        wanted = rejected[0]["reference_token"]
        if not proposable_by_shipped(wanted):
            truncated_rounds += 1
            truncating[wanted] += 1

    round_count = blob.get("round_count") or len(rounds)
    accepted = blob.get("accepted_draft_total", 0)
    emitted = blob.get("emitted_token_total") or len(tails)
    declared_rows = blob.get("declared_rows_total") or len(ledger)
    return {
        "path": str(path),
        "round_count": round_count,
        "tokens": emitted,
        "all_tokens_matched": blob.get("all_tokens_matched"),
        "parity_all_ok": blob.get("parity_all_ok"),
        "residual_divergence_count": blob.get("residual_divergence_count"),
        "declared_rows_total": blob.get("declared_rows_total"),
        "reference_checked_row_total": blob.get("reference_checked_row_total"),
        "head_sha256": (blob.get("head_provenance") or {}).get("sha256", "")[:12],
        "accepted_draft_total": accepted,
        "rejected_draft_total": blob.get("rejected_draft_total"),
        "target_tail_total": blob.get("target_tail_total"),
        "accepted_draft_rate": blob.get("accepted_draft_rate"),
        "mean_accepted_drafts_per_round": accepted / round_count if round_count else 0.0,
        # Rounds are the unit of cost, so this is the acceptance channel that
        # reaches the score. Taken from the emitted count rather than
        # accepted + rounds, because the window truncates the final round.
        "mean_tokens_per_round": emitted / round_count if round_count else 0.0,
        # Target rows actually evaluated per emitted token. A second, coarser
        # cost view: the verify forward is batched, so rows move round cost
        # less than round count does, but both must point the same way.
        "declared_rows_per_token": declared_rows / emitted if emitted else 0.0,
        # Rule 114 witness.
        "widened_draft_rows": len(widened_rows),
        "widened_draft_rows_accepted": sum(1 for r in widened_rows if r["accepted"]),
        "widened_distinct_tokens": len({r["token"] for r in widened_rows}),
        "arm_witnessed": "widened" if widened_rows else "shipped-compatible",
        # Truncation census on the live decode path.
        "rounds_with_draft_rows": scored_rounds,
        "rounds_truncated_by_unproposable_token": truncated_rounds,
        "truncation_share_of_rounds_pct": 100.0 * truncated_rounds / scored_rounds
        if scored_rounds
        else 0.0,
        "top_truncating_tokens": truncating.most_common(12),
    }


def attribute(arms: dict) -> dict:
    """Split the acceptance gain into its two inseparable causes.

    Widening does two things that cannot be separated by construction. It lets
    the head propose a high id at all, and it repartitions the cluster index
    over 248,320 rows instead of 98,336, which also changes the shortlist for
    ids the shipped table already held. Only the first is the hypothesis.

    The ledger settles the split directly: an accepted draft row carrying a
    high id is the hypothesis firing, and the rest of the gain is the
    repartition. Neither is a confound for the DECISION, because no
    implementation can widen without repartitioning, but the explanation
    should say which one paid.
    """
    out: dict = {"seeds": {}}
    for seed in MEDPAIR:
        shipped = arms.get("shipped", {}).get("seeds", {}).get(seed)
        full = arms.get("full", {}).get("seeds", {}).get(seed)
        if not shipped or not full:
            continue
        gain = full["accepted_draft_total"] - shipped["accepted_draft_total"]
        direct = full["widened_draft_rows_accepted"]
        out["seeds"][seed] = {
            "accepted_shipped": shipped["accepted_draft_total"],
            "accepted_full": full["accepted_draft_total"],
            "accepted_gain": gain,
            "accepted_gain_from_high_ids": direct,
            "accepted_gain_from_repartition": gain - direct,
            "shipped_rounds_truncated_by_unproposable": shipped[
                "rounds_truncated_by_unproposable_token"
            ],
        }
    if set(out["seeds"]) == set(MEDPAIR):
        for key in (
            "accepted_gain",
            "accepted_gain_from_high_ids",
            "accepted_gain_from_repartition",
        ):
            out[f"medpair_{key}"] = sum(
                MEDPAIR[s] * out["seeds"][s][key] for s in MEDPAIR
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify-dir",
        default=str(
            Path.home()
            / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e141/verify"
        ),
    )
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--arms", default="shipped,full")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="research/e141-rung3.json")
    args = ap.parse_args()

    root = Path(args.verify_dir)
    arms = args.arms.split(",")
    report: dict = {"steps": args.steps, "arms": {}}

    for arm in arms:
        per_seed = {}
        for seed in MEDPAIR:
            path = root / f"{seed}_{arm}_{args.steps}{args.suffix}.json"
            if not path.exists():
                print(f"  missing {path}")
                continue
            per_seed[seed] = summarise(path)
        if not per_seed:
            continue
        report["arms"][arm] = {"seeds": per_seed}
        if set(per_seed) == set(MEDPAIR):
            report["arms"][arm]["medpair"] = {
                key: sum(MEDPAIR[s] * per_seed[s][key] for s in MEDPAIR)
                for key in (
                    "mean_accepted_drafts_per_round",
                    "mean_tokens_per_round",
                    "declared_rows_per_token",
                    "accepted_draft_rate",
                    "truncation_share_of_rounds_pct",
                )
            }

    if "shipped" in report["arms"] and "full" in report["arms"]:
        base = report["arms"]["shipped"].get("medpair")
        cand = report["arms"]["full"].get("medpair")
        if base and cand:
            base_tpr = base["mean_tokens_per_round"]
            cand_tpr = cand["mean_tokens_per_round"]
            report["delta"] = {
                "accept_rate_delta_pp": 100.0
                * (cand["accepted_draft_rate"] - base["accepted_draft_rate"]),
                "mean_tokens_per_round_shipped": base_tpr,
                "mean_tokens_per_round_full": cand_tpr,
                # Rounds are the unit of cost; more tokens per round means
                # fewer rounds for the same 512-token window.
                "rounds_saved_pct": 100.0 * (1.0 - base_tpr / cand_tpr)
                if cand_tpr
                else 0.0,
                "declared_rows_per_token_shipped": base["declared_rows_per_token"],
                "declared_rows_per_token_full": cand["declared_rows_per_token"],
                "target_rows_saved_pct": 100.0
                * (1.0 - cand["declared_rows_per_token"]
                   / base["declared_rows_per_token"])
                if base["declared_rows_per_token"]
                else 0.0,
            }
        report["attribution"] = attribute(report["arms"])

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for arm, blob in report["arms"].items():
        print(f"\n== arm {arm} ==")
        for seed, s in blob["seeds"].items():
            print(
                f"  {seed:18s} rounds={s['round_count']:4d} "
                f"acc={s['accepted_draft_total']:5d} rej={s['rejected_draft_total']:5d} "
                f"rate={s['accepted_draft_rate']:.4f} "
                f"tok/round={s['mean_tokens_per_round']:.4f}"
            )
            print(
                f"  {'':18s} witness={s['arm_witnessed']} "
                f"widened_rows={s['widened_draft_rows']} "
                f"(accepted {s['widened_draft_rows_accepted']}, "
                f"distinct {s['widened_distinct_tokens']})"
            )
            print(
                f"  {'':18s} truncated_by_unproposable="
                f"{s['rounds_truncated_by_unproposable_token']}"
                f"/{s['rounds_with_draft_rows']} "
                f"({s['truncation_share_of_rounds_pct']:.3f} %)  "
                f"matched={s['all_tokens_matched']} "
                f"parity={s['parity_all_ok']} "
                f"div={s['residual_divergence_count']}"
            )
        if "medpair" in blob:
            m = blob["medpair"]
            print(
                f"  medpair            rate={m['accepted_draft_rate']:.4f} "
                f"tok/round={m['mean_tokens_per_round']:.4f} "
                f"trunc={m['truncation_share_of_rounds_pct']:.3f} %"
            )

    attribution = report.get("attribution", {})
    if attribution.get("seeds"):
        print("\n== where the extra accepted drafts came from ==")
        for seed, a in attribution["seeds"].items():
            print(
                f"  {seed:18s} accepted {a['accepted_shipped']} -> "
                f"{a['accepted_full']} ({a['accepted_gain']:+d}); "
                f"high ids {a['accepted_gain_from_high_ids']:+d}, "
                f"repartition {a['accepted_gain_from_repartition']:+d}"
            )
        if "medpair_accepted_gain" in attribution:
            print(
                f"  medpair            gain "
                f"{attribution['medpair_accepted_gain']:+.2f} = high ids "
                f"{attribution['medpair_accepted_gain_from_high_ids']:+.2f} + "
                f"repartition "
                f"{attribution['medpair_accepted_gain_from_repartition']:+.2f}"
            )

    if "delta" in report:
        d = report["delta"]
        print(
            f"\ne141_full_vocab_accept_delta_pp = {d['accept_rate_delta_pp']:.4f}"
        )
        print(
            f"tokens/round {d['mean_tokens_per_round_shipped']:.4f} -> "
            f"{d['mean_tokens_per_round_full']:.4f}  "
            f"rounds saved {d['rounds_saved_pct']:.4f} %"
        )
        print(
            f"target rows/token {d['declared_rows_per_token_shipped']:.4f} -> "
            f"{d['declared_rows_per_token_full']:.4f}  "
            f"rows saved {d['target_rows_saved_pct']:.4f} %"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
