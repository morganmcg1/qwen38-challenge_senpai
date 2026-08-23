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


def token_fates(path: Path) -> list[dict]:
    """Per emitted-token fate, indexed by position in the 512-token stream.

    Every arm emits the identical token stream, so position is the only key
    that lets arms be compared directly. Round boundaries are not comparable
    because the arms disagree about them.

    fate is one of:
      accepted   the draft matched, the round continued
      corrected  the draft was the round's first divergence, target overruled
      bonus      the round accepted every draft, this token cost no trial
    """
    blob = json.loads(path.read_text())
    rounds: dict[int, list[dict]] = {}
    for row in blob["row_ledger"]:
        rounds.setdefault(row["round"], []).append(row)

    out: list[dict] = []
    for rnd in sorted(rounds):
        drafts = sorted(
            [r for r in rounds[rnd] if r["kind"] == "draft"],
            key=lambda r: r["draft_index"],
        )
        rejected = [r for r in drafts if not r["accepted"]]
        first = min(rejected, key=lambda r: r["draft_index"]) if rejected else None
        for r in drafts:
            if r["accepted"]:
                out.append(
                    {
                        "token": r["reference_token"],
                        "fate": "accepted",
                        "round": rnd,
                        "draft_index": r["draft_index"],
                    }
                )
        for r in [x for x in rounds[rnd] if x["kind"] == "targetTail"]:
            out.append(
                {
                    "token": r["reference_token"],
                    "fate": "corrected" if first is not None else "bonus",
                    "round": rnd,
                    "draft_index": first["draft_index"] if first else None,
                }
            )
    return out


def cross_arm(seed: str, arms: list[str], steps: int) -> dict | None:
    """How much of the binding channel does each arm actually recover?

    A binding event is an emitted token that the shipped compact vocabulary
    cannot propose, so shipped must spend a correction on it. Making a token
    proposable is necessary but not sufficient: the draft head must also rank
    it top-1 inside the probed rows. This measures the sufficient part.
    """
    fates: dict[str, list[dict]] = {}
    for arm in arms:
        path = VERIFY / f"{seed}_{arm}_{steps}.json"
        if path.exists():
            fates[arm] = token_fates(path)
    if "shipped" not in fates:
        return None

    lengths = {a: len(f) for a, f in fates.items()}
    streams_agree = len({tuple(x["token"] for x in f) for f in fates.values()}) == 1

    binding = [
        i
        for i, x in enumerate(fates["shipped"])
        if x["fate"] == "corrected" and not proposable(x["token"])
    ]
    rows = []
    for i in binding:
        row = {
            "position": i,
            "token": fates["shipped"][i]["token"],
            "fate_by_arm": {
                a: f[i]["fate"] for a, f in fates.items() if i < len(f)
            },
        }
        rows.append(row)
    recovered = {
        a: sum(
            1
            for r in rows
            if r["fate_by_arm"].get(a) in ("accepted", "bonus")
        )
        for a in fates
    }
    return {
        "emitted_lengths": lengths,
        "token_streams_agree": streams_agree,
        "binding_positions": rows,
        "binding_event_count": len(rows),
        "binding_events_recovered": recovered,
        "binding_recovery_pct": {
            a: (100.0 * v / len(rows) if rows else None)
            for a, v in recovered.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="shipped")
    ap.add_argument("--arms", default="", help="cross-arm binding recovery")
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

    arms = [a for a in args.arms.split(",") if a]
    if arms:
        report["cross_arm"] = {}
        for seed in MEDPAIR:
            blob = cross_arm(seed, arms, args.steps)
            if blob:
                report["cross_arm"][seed] = blob

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
    for seed, blob in report.get("cross_arm", {}).items():
        print(f"\n== cross-arm binding recovery, {seed} ==")
        print(
            f"  emitted lengths {blob['emitted_lengths']}  "
            f"token streams agree {blob['token_streams_agree']}"
        )
        if not blob["binding_positions"]:
            print("  no binding events on this seed")
            continue
        arm_names = list(blob["binding_positions"][0]["fate_by_arm"])
        header = "  position   token  " + "  ".join(f"{a:>9s}" for a in arm_names)
        print(header)
        for row in blob["binding_positions"]:
            fates = "  ".join(
                f"{row['fate_by_arm'].get(a, '-'):>9s}" for a in arm_names
            )
            print(f"  {row['position']:8d} {row['token']:7d}  {fates}")
        print(
            f"  recovered of {blob['binding_event_count']}: "
            + "  ".join(
                f"{a} {blob['binding_events_recovered'][a]}" for a in arm_names
            )
        )
    for key in list(report):
        if key.startswith("medpair_"):
            print(f"{key} = {report[key]:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
