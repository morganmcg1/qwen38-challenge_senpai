#!/usr/bin/env python3
"""F7 section 6 — resolve the 2.05x C-a census disagreement on one token stream.

E141's corpus census counts unproposable ids over the tokenised corpus.
E143's live census counts unproposable target tokens over real draft trials
on the scored trajectory. They disagree by 2.05x (0.4832 % against 0.2658 %).

Both are computable from a single 512-token verify row ledger, so this
settles which denominator the campaign should price against. Zero GPU.

Bases reported, all on the same stream:

  corpus_basis      unproposable share of the 512 emitted tokens
  trial_basis       unproposable share of `kind == draft` rows, the E143
                    denominator
  trial_basis_dedup the same with `targetTail` rows excluded explicitly, to
                    show the double-count term E143 named
  binding_basis     rounds whose accepted prefix is truncated by an
                    unproposable target token, over total rounds. This is the
                    only basis that maps 1:1 onto a recoverable round.

`binding_basis` is the number that prices the mechanism, because a round is
the unit the scheduler and the score both see.
"""

import argparse
import json
from pathlib import Path

VERIFY = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e141/verify"
PREFIX_COUNT = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}


def proposable(t: int) -> bool:
    return t < PREFIX_COUNT or CONTROL_START <= t < CONTROL_END


def analyse(path: Path) -> dict:
    blob = json.loads(path.read_text())
    ledger = blob["row_ledger"]
    rounds: dict[int, list[dict]] = {}
    for row in ledger:
        rounds.setdefault(row["round"], []).append(row)

    draft_rows = [r for r in ledger if r["kind"] == "draft"]
    tail_rows = [r for r in ledger if r["kind"] == "targetTail"]

    emitted: list[int] = []
    emitted_with_trial: list[int] = []
    emitted_no_trial: list[int] = []
    binding = 0
    binding_at_index: dict[int, int] = {}
    first_div_index: dict[int, int] = {}
    trials_that_could_never_match = 0
    discarded_rows = 0

    for rnd in sorted(rounds):
        rows = sorted(
            [r for r in rounds[rnd] if r["kind"] == "draft"],
            key=lambda r: r["draft_index"],
        )
        tail = [r for r in rounds[rnd] if r["kind"] == "targetTail"]
        # The parent commits the accepted draft prefix and then the tail row's
        # reference token, so the emitted stream is prefix + tail.
        for r in rows:
            if r["accepted"]:
                emitted.append(r["reference_token"])
                emitted_with_trial.append(r["reference_token"])
        rejected = [r for r in rows if not r["accepted"]]
        for r in tail:
            emitted.append(r["reference_token"])
            # A round that rejected a draft emits the correction at that
            # position, so that emitted token did get a draft trial. A round
            # that accepted every draft emits a free bonus token that no
            # draft ever attempted.
            if rejected:
                emitted_with_trial.append(r["reference_token"])
            else:
                emitted_no_trial.append(r["reference_token"])

        for r in rows:
            if not proposable(r["reference_token"]):
                trials_that_could_never_match += 1

        if rejected:
            first = min(rejected, key=lambda r: r["draft_index"])
            # Rows after the first divergence are speculative work the parent
            # discards. They inflate a per-row denominator without ever
            # corresponding to an emitted token.
            discarded_rows += sum(
                1 for r in rows if r["draft_index"] > first["draft_index"]
            )
            first_div_index[first["draft_index"]] = (
                first_div_index.get(first["draft_index"], 0) + 1
            )
            if not proposable(first["reference_token"]):
                binding += 1
                binding_at_index[first["draft_index"]] = (
                    binding_at_index.get(first["draft_index"], 0) + 1
                )

    total_rounds = len(rounds)
    emitted_unprop = sum(1 for t in emitted if not proposable(t))
    trial_unprop = sum(1 for t in emitted_with_trial if not proposable(t))
    first_divergences = sum(first_div_index.values())
    return {
        "path": path.name,
        "rounds": total_rounds,
        "emitted_tokens": len(emitted),
        "draft_rows": len(draft_rows),
        "tail_rows": len(tail_rows),
        "corpus_basis_events": emitted_unprop,
        "corpus_basis_pct": 100.0 * emitted_unprop / len(emitted),
        "row_basis_events": trials_that_could_never_match,
        "row_basis_pct": 100.0 * trials_that_could_never_match / len(draft_rows),
        "row_basis_denominator": len(draft_rows),
        "discarded_post_divergence_rows": discarded_rows,
        "tail_double_count_ratio": (len(draft_rows) + len(tail_rows))
        / len(draft_rows),
        "trial_basis_events": trial_unprop,
        "trial_basis_pct": 100.0 * trial_unprop / len(emitted_with_trial),
        "trial_basis_denominator": len(emitted_with_trial),
        "emitted_no_trial": len(emitted_no_trial),
        "emitted_no_trial_pct": 100.0 * len(emitted_no_trial) / len(emitted),
        "binding_basis_events": binding,
        "binding_basis_pct": 100.0 * binding / total_rounds,
        "binding_share_of_first_divergences_pct": (
            100.0 * binding / first_divergences if first_divergences else None
        ),
        "first_divergence_count": first_divergences,
        "binding_at_draft_index": dict(sorted(binding_at_index.items())),
        "first_divergence_index_histogram": dict(sorted(first_div_index.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="shipped")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--out", default="research/e141-ca-resolver.json")
    args = ap.parse_args()

    report: dict = {"arm": args.arm, "steps": args.steps, "seeds": {}}
    for seed in MEDPAIR:
        path = VERIFY / f"{seed}_{args.arm}_{args.steps}.json"
        if not path.exists():
            continue
        report["seeds"][seed] = analyse(path)

    if set(report["seeds"]) == set(MEDPAIR):
        for basis in ("corpus_basis_pct", "trial_basis_pct", "binding_basis_pct"):
            report[f"medpair_{basis}"] = sum(
                MEDPAIR[s] * report["seeds"][s][basis] for s in MEDPAIR
            )

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for seed, blob in report["seeds"].items():
        print(f"\n== {seed} arm={args.arm} ==")
        print(
            f"  rounds {blob['rounds']}  emitted {blob['emitted_tokens']}  "
            f"draft rows {blob['draft_rows']}  tail rows {blob['tail_rows']}"
        )
        print(
            f"  corpus  basis {blob['corpus_basis_events']:3d} / "
            f"{blob['emitted_tokens']:4d} emitted = "
            f"{blob['corpus_basis_pct']:.4f} %"
        )
        print(
            f"  row     basis {blob['row_basis_events']:3d} / "
            f"{blob['row_basis_denominator']:4d} draft rows = "
            f"{blob['row_basis_pct']:.4f} %  "
            f"({blob['discarded_post_divergence_rows']} rows discarded past "
            f"first divergence, tail double-count ratio "
            f"{blob['tail_double_count_ratio']:.4f})"
        )
        print(
            f"  trial   basis {blob['trial_basis_events']:3d} / "
            f"{blob['trial_basis_denominator']:4d} emitted-with-trial = "
            f"{blob['trial_basis_pct']:.4f} %  "
            f"({blob['emitted_no_trial']} emitted with no draft trial = "
            f"{blob['emitted_no_trial_pct']:.2f} %)"
        )
        share = blob["binding_share_of_first_divergences_pct"]
        share_text = f"{share:.4f} %" if share is not None else "n/a"
        print(
            f"  binding basis {blob['binding_basis_events']:3d} / "
            f"{blob['rounds']:4d} rounds = {blob['binding_basis_pct']:.4f} %"
            f"   = {share_text} of the "
            f"{blob['first_divergence_count']} first divergences"
        )
        print(f"  binding at draft index {blob['binding_at_draft_index']}")
        print(
            f"  first divergence index histogram "
            f"{blob['first_divergence_index_histogram']}"
        )
    for key in list(report):
        if key.startswith("medpair_"):
            print(f"{key} = {report[key]:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
