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

# F2 field `e141_uniform_round_cost_pct`: the arm's byte cost as a percent of
# the round, applied to all eight prompts. Rung 3 cannot measure this; it comes
# from the byte table in research/e141-arm-geometry.json priced at the 28.91 us
# per MB this session's 16-leg ABBA block measured directly.
ROUND_COST_PCT = {
    "full": 2.014,
    "armA": 0.671,
    "armA1844": 0.671,
    "armB20": 0.004,
    "armB16": 0.115,
    "leaf16": 0.000,
}


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
    rounds_with_reject = 0
    # F4: a miss at draft step 1 costs the whole round, a miss at step 5 has
    # already banked four tokens. `draft_index` is 0-based, so step 1 is index
    # 0. Split the truncation census by that index, because a variant that
    # widens the probe at step 1 only recovers the index-0 rows and nothing
    # deeper.
    first_reject_position = Counter()
    unproposable_position = Counter()
    # F6 Rule 125: an acceptance gain is spent by the scheduler, not banked. A
    # round submits its drafts plus one committed tail token to the target in
    # one verify call, so that row count IS the verify width the width-cost
    # curve is indexed by. E82 lost 2.7 to 3.4 percent of seconds per token by
    # converting acceptance into depth, because the curve steps 36 percent into
    # width 6. Count the widths so the same failure cannot hide here.
    width_histogram = Counter()
    for _, rows in sorted(rounds.items()):
        width_histogram[sum(1 for r in rows if r["kind"] == "draft") + 1] += 1
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
        rounds_with_reject += 1
        position = rejected[0].get("draft_index", 0)
        first_reject_position[position] += 1
        wanted = rejected[0]["reference_token"]
        if not proposable_by_shipped(wanted):
            truncated_rounds += 1
            truncating[wanted] += 1
            unproposable_position[position] += 1
    position1 = unproposable_position.get(0, 0)

    # Recall at fixed probed rows, split by which table can reach the answer.
    # A draft row is accepted exactly when the head proposed the token the
    # target wanted, so the accept rate IS the head's recall. Arm A holds the
    # probed row count fixed across arms, which makes this the clean way to
    # ask whether widening the haystack costs recall on the core domain.
    core = [r for r in drafts if proposable_by_shipped(r["reference_token"])]
    widened_domain = [
        r for r in drafts if not proposable_by_shipped(r["reference_token"])
    ]

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
        # Recall at fixed probed rows, the F1 headline scientific quantity.
        "draft_rows_total": len(drafts),
        "draft_rows_core_domain": len(core),
        "draft_rows_widened_domain": len(widened_domain),
        "recall_pct_all": 100.0 * sum(1 for r in drafts if r["accepted"]) / len(drafts)
        if drafts
        else 0.0,
        "recall_pct_core_domain": 100.0 * sum(1 for r in core if r["accepted"]) / len(core)
        if core
        else 0.0,
        "recall_pct_widened_domain": 100.0
        * sum(1 for r in widened_domain if r["accepted"])
        / len(widened_domain)
        if widened_domain
        else None,
        # Rule 114 witness.
        "widened_draft_rows": len(widened_rows),
        "widened_draft_rows_accepted": sum(1 for r in widened_rows if r["accepted"]),
        "widened_distinct_tokens": len({r["token"] for r in widened_rows}),
        "arm_witnessed": "widened" if widened_rows else "shipped-compatible",
        # Truncation census on the live decode path.
        "rounds_with_draft_rows": scored_rounds,
        "rounds_with_rejected_draft": rounds_with_reject,
        "rounds_truncated_by_unproposable_token": truncated_rounds,
        "truncation_share_of_rounds_pct": 100.0 * truncated_rounds / scored_rounds
        if scored_rounds
        else 0.0,
        "top_truncating_tokens": truncating.most_common(12),
        # F4 column: how much of the truncation census a step-1-only widened
        # probe could reach. draft_index is 0-based, so step 1 is index 0.
        "first_reject_position_histogram": dict(sorted(first_reject_position.items())),
        "unproposable_by_position": dict(sorted(unproposable_position.items())),
        # F6 Rule 125 columns.
        "verify_width_histogram": dict(sorted(width_histogram.items())),
        "verify_width_mass_ge6_pct": 100.0
        * sum(n for w, n in width_histogram.items() if w >= 6)
        / round_count
        if round_count
        else 0.0,
        "verify_width_mean": sum(w * n for w, n in width_histogram.items())
        / round_count
        if round_count
        else 0.0,
        "verify_rows_per_round": declared_rows / round_count if round_count else 0.0,
        "rounds_truncated_at_position1": position1,
        "rounds_truncated_beyond_position1": truncated_rounds - position1,
        "position1_share_of_unproposable_pct": 100.0 * position1 / truncated_rounds
        if truncated_rounds
        else None,
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

    # F2: every non-shipped arm gets the same contrast, and the two halves of
    # the prize are reported separately so the advisor can compose them under
    # Rule 121 instead of blending them here.
    if "shipped" in report["arms"]:
        ships = report["arms"]["shipped"]["seeds"]
        report["contrasts"] = {}
        for name, blob in report["arms"].items():
            if name == "shipped":
                continue
            cands = blob["seeds"]
            if set(ships) != set(MEDPAIR) or set(cands) != set(MEDPAIR):
                continue
            recovered = {
                seed: 100.0
                * (1.0 - cands[seed]["round_count"] / ships[seed]["round_count"])
                for seed in MEDPAIR
            }
            # F4 column. Round boundaries shift between arms, so no ledger pair
            # can name which recovered round was which. Apportion the measured
            # recovery by the shipped arm's own position census instead, which
            # is the population a step-1-only widened probe would attack. This
            # is DERIVED, not measured, and it assumes a widened probe recovers
            # position-1 and deeper truncations at the same rate.
            position1 = {}
            for seed in MEDPAIR:
                share = ships[seed]["position1_share_of_unproposable_pct"]
                if share is None:
                    position1[seed] = {"share_pct": None, "recovered_pct": None}
                    continue
                position1[seed] = {
                    "share_pct": share,
                    "shipped_truncations": ships[seed][
                        "rounds_truncated_by_unproposable_token"
                    ],
                    "shipped_truncations_at_position1": ships[seed][
                        "rounds_truncated_at_position1"
                    ],
                    "recovered_pct": recovered[seed] * share / 100.0,
                    "recovered_pct_beyond_position1": recovered[seed]
                    * (100.0 - share)
                    / 100.0,
                }
            report["contrasts"][name] = {
                "e141_recovered_pct_beagle_raw": recovered["beagle_a"],
                "e141_recovered_pct_essays_raw": recovered["essays_montaigne"],
                "e141_uniform_round_cost_pct": ROUND_COST_PCT.get(name),
                "e141_recovered_pct_position1_derived": position1,
                "round_count_shipped": {s: ships[s]["round_count"] for s in MEDPAIR},
                "round_count_candidate": {s: cands[s]["round_count"] for s in MEDPAIR},
                "recall_pct_all": {
                    s: {
                        "shipped": ships[s]["recall_pct_all"],
                        "candidate": cands[s]["recall_pct_all"],
                        "delta_pp": cands[s]["recall_pct_all"]
                        - ships[s]["recall_pct_all"],
                    }
                    for s in MEDPAIR
                },
                "recall_pct_core_domain": {
                    s: {
                        "shipped": ships[s]["recall_pct_core_domain"],
                        "candidate": cands[s]["recall_pct_core_domain"],
                        "delta_pp": cands[s]["recall_pct_core_domain"]
                        - ships[s]["recall_pct_core_domain"],
                    }
                    for s in MEDPAIR
                },
                "recall_pct_widened_domain": {
                    s: {
                        "shipped": ships[s]["recall_pct_widened_domain"],
                        "candidate": cands[s]["recall_pct_widened_domain"],
                    }
                    for s in MEDPAIR
                },
                # F6 Rule 125. The flag the advisor asked for is a MOVE of mass
                # into the expensive widths, so it compares against shipped
                # rather than against an absolute level.
                "verify_width": {
                    s: {
                        "histogram_shipped": ships[s]["verify_width_histogram"],
                        "histogram_candidate": cands[s]["verify_width_histogram"],
                        "mass_ge6_pct_shipped": ships[s]["verify_width_mass_ge6_pct"],
                        "mass_ge6_pct_candidate": cands[s]["verify_width_mass_ge6_pct"],
                        "mass_ge6_delta_pp": cands[s]["verify_width_mass_ge6_pct"]
                        - ships[s]["verify_width_mass_ge6_pct"],
                        "mean_shipped": ships[s]["verify_width_mean"],
                        "mean_candidate": cands[s]["verify_width_mean"],
                        "rows_per_round_shipped": ships[s]["verify_rows_per_round"],
                        "rows_per_round_candidate": cands[s]["verify_rows_per_round"],
                        "rows_per_round_delta_pct": 100.0
                        * (
                            cands[s]["verify_rows_per_round"]
                            / ships[s]["verify_rows_per_round"]
                            - 1.0
                        ),
                    }
                    for s in MEDPAIR
                },
                "verify_width_mass_moved_to_6plus": any(
                    cands[s]["verify_width_mass_ge6_pct"]
                    > ships[s]["verify_width_mass_ge6_pct"]
                    for s in MEDPAIR
                ),
                "f209_gain_argument": "beagle={:.4f},essays={:.4f}".format(
                    recovered["beagle_a"], recovered["essays_montaigne"]
                ),
            }

    if "shipped" in report["arms"] and "full" in report["arms"]:
        base = report["arms"]["shipped"].get("medpair")
        cand = report["arms"]["full"].get("medpair")
        ships = report["arms"]["shipped"]["seeds"]
        fulls = report["arms"]["full"]["seeds"]
        if base and cand and set(ships) == set(MEDPAIR) == set(fulls):
            # The published score is a weighted combination of PER-PROMPT
            # relative changes, because each prompt contributes its own
            # raw_p = baseline_spt_p / candidate_spt_p. Weighting the levels
            # and then taking one ratio is a different, wrong quantity: it
            # lets a prompt with more rounds per token borrow gain from a
            # prompt that had none.
            per_prompt = {
                seed: 100.0
                * (1.0 - fulls[seed]["round_count"] / ships[seed]["round_count"])
                for seed in MEDPAIR
            }
            report["delta"] = {
                "accept_rate_delta_pp": 100.0
                * (cand["accepted_draft_rate"] - base["accepted_draft_rate"]),
                "mean_tokens_per_round_shipped": base["mean_tokens_per_round"],
                "mean_tokens_per_round_full": cand["mean_tokens_per_round"],
                "rounds_saved_pct_per_prompt": per_prompt,
                "rounds_saved_pct": sum(
                    MEDPAIR[s] * per_prompt[s] for s in MEDPAIR
                ),
                "round_count_shipped": {s: ships[s]["round_count"] for s in MEDPAIR},
                "round_count_full": {s: fulls[s]["round_count"] for s in MEDPAIR},
                "declared_rows_per_token_shipped": base["declared_rows_per_token"],
                "declared_rows_per_token_full": cand["declared_rows_per_token"],
                "target_rows_saved_pct": sum(
                    MEDPAIR[s]
                    * 100.0
                    * (
                        1.0
                        - fulls[s]["declared_rows_per_token"]
                        / ships[s]["declared_rows_per_token"]
                    )
                    for s in MEDPAIR
                ),
            }
        report["attribution"] = attribute(report["arms"])

    # Rule 101 positive control for the arm selector itself. essays produces
    # zero widened rows, so the widened-row witness cannot prove the selector
    # reached that prompt's worker. An arm that changes the proposable set must
    # instead change the ledger. If it does not, the arm never took effect and
    # the shipped-equals-full null is a plumbing artefact rather than a result.
    #
    # `leaf16` is excluded on purpose. It is the NULL control: it repartitions
    # the shipped vocabulary and must reproduce shipped exactly, so requiring it
    # to change would invert its meaning. Its own liveness is witnessed by the
    # leaf override in the run's arm record, not by a ledger change.
    NULL_CONTROL_ARMS = {"leaf16"}
    narrow = [
        a
        for a in report["arms"]
        if a not in ("shipped", "full") and a not in NULL_CONTROL_ARMS
    ]
    if narrow:
        control: dict = {"arms": narrow, "null_control_arms": sorted(
            set(report["arms"]) & NULL_CONTROL_ARMS
        ), "seeds": {}}
        for arm in narrow:
            for seed, blob in report["arms"][arm]["seeds"].items():
                ship = report["arms"]["shipped"]["seeds"][seed]
                control["seeds"][f"{seed}@{arm}"] = {
                    "round_count": [ship["round_count"], blob["round_count"]],
                    "accepted": [
                        ship["accepted_draft_total"],
                        blob["accepted_draft_total"],
                    ],
                    "rejected": [
                        ship["rejected_draft_total"],
                        blob["rejected_draft_total"],
                    ],
                    "ledger_changed": (
                        ship["round_count"] != blob["round_count"]
                        or ship["accepted_draft_total"]
                        != blob["accepted_draft_total"]
                    ),
                    "still_exact": bool(
                        blob["all_tokens_matched"]
                        and blob["residual_divergence_count"] == 0
                    ),
                }
        # Liveness is a property of the arm, not of every seed. A seed whose
        # decode is insensitive to the arm (essays does not respond to probe
        # fraction at all) produces an identical ledger without implying the
        # override was dropped, so require one changed seed per arm.
        per_arm = {
            arm: any(
                control["seeds"][f"{seed}@{arm}"]["ledger_changed"]
                for seed in report["arms"][arm]["seeds"]
            )
            for arm in narrow
        }
        control["selector_proven_live_per_arm"] = per_arm
        control["selector_proven_live"] = all(per_arm.values())
        control["exact_at_narrow_prefix"] = all(
            v["still_exact"] for v in control["seeds"].values()
        )
        report["selector_positive_control"] = control

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
            share = s["position1_share_of_unproposable_pct"]
            print(
                f"  {'':18s} first_reject_pos={s['first_reject_position_histogram']} "
                f"unproposable_pos={s['unproposable_by_position']} "
                f"pos1={s['rounds_truncated_at_position1']}"
                f"/{s['rounds_truncated_by_unproposable_token']} "
                + (f"({share:.1f} %)" if share is not None else "(n/a)")
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
            f"{d['mean_tokens_per_round_full']:.4f}"
        )
        for seed, pct in d["rounds_saved_pct_per_prompt"].items():
            print(
                f"  {seed:18s} rounds {d['round_count_shipped'][seed]:4d} -> "
                f"{d['round_count_full'][seed]:4d}  saved {pct:+.4f} %"
            )
        print(
            f"  {'medpair':18s} rounds saved {d['rounds_saved_pct']:.4f} % "
            f"(weight of per-prompt relative changes)"
        )
        print(
            f"target rows/token {d['declared_rows_per_token_shipped']:.4f} -> "
            f"{d['declared_rows_per_token_full']:.4f}  "
            f"rows saved {d['target_rows_saved_pct']:.4f} %"
        )
    if "selector_positive_control" in report:
        c = report["selector_positive_control"]
        print("\n== selector positive control (narrow prefix must hurt) ==")
        for key, v in c["seeds"].items():
            print(
                f"  {key:28s} rounds {v['round_count'][0]} -> "
                f"{v['round_count'][1]}  accepted {v['accepted'][0]} -> "
                f"{v['accepted'][1]}  changed={v['ledger_changed']} "
                f"exact={v['still_exact']}"
            )
        print(
            f"  selector_proven_live={c['selector_proven_live']}  "
            f"exact_at_narrow_prefix={c['exact_at_narrow_prefix']}"
        )

    for name, c in (report.get("contrasts") or {}).items():
        print(f"\n== F2 contrast, arm {name} against shipped ==")
        print(
            f"  e141_recovered_pct_beagle_raw  {c['e141_recovered_pct_beagle_raw']:+.4f}"
            f"   rounds {c['round_count_shipped']['beagle_a']} -> "
            f"{c['round_count_candidate']['beagle_a']}"
        )
        print(
            f"  e141_recovered_pct_essays_raw  {c['e141_recovered_pct_essays_raw']:+.4f}"
            f"   rounds {c['round_count_shipped']['essays_montaigne']} -> "
            f"{c['round_count_candidate']['essays_montaigne']}"
        )
        cost = c["e141_uniform_round_cost_pct"]
        print(
            "  e141_uniform_round_cost_pct    "
            + (f"{cost:+.4f}" if cost is not None else "unpriced")
        )
        print("  recall at fixed probed rows, percentage points:")
        for seed in MEDPAIR:
            a = c["recall_pct_all"][seed]
            k = c["recall_pct_core_domain"][seed]
            w = c["recall_pct_widened_domain"][seed]
            wc = w["candidate"]
            print(
                f"    {seed:18s} all {a['shipped']:.3f} -> {a['candidate']:.3f}"
                f" ({a['delta_pp']:+.3f})   core {k['shipped']:.3f} -> "
                f"{k['candidate']:.3f} ({k['delta_pp']:+.3f})   widened domain "
                + (f"{wc:.3f}" if wc is not None else "no rows")
            )
        print(f"  f209 argument: --gain {c['f209_gain_argument']}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
