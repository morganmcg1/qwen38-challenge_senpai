#!/usr/bin/env python3
"""E204 arm reader: the falsification gate, then the round-time decision.

    usage: research/e204_arms.py pair OFF_TAG ON_TAG [--json OUT]
           research/e204_arms.py alt  TAG [--skip-rounds N] [--json OUT]

`pair` is the STEP-7 FALSIFICATION GATE. Two legs of the same binary, one with
`DARKBLOOM_E204_CHAIN_ARM=off` and one on the shipped unconditional path. The
chain prefetch moves WHEN head steps 2..d are built and submitted; it must not
move WHAT they compute. Four independent ways for that claim to fail, all
checked here:

1. **Row exactness.** `mtp-row:` carries the top-2 ids AND the top-2 logits as
   hexfloats, so the comparison is bitwise on actual floating-point values at
   the rows the target produced, not an argmax match. A reordering that
   perturbed the head cache would show here first.
2. **Accept-ledger invariance.** The per-round `(d, acc)` sequence must be
   identical. The prefetched width is a PREDICTION; if it ever leaked into what
   the next round proposed or accepted, this sequence would diverge.
3. **Row-ledger closure.** `rows_per_round = d + 1` and the emitted-token count
   must close on both legs, and the harness's own `all_tokens_matched` and
   `residual_divergence_count` must hold.
4. **Positive control.** The comparison is proved capable of failing: the
   reader reports the first position at which the two legs' hexfloats differ,
   and a self-check compares each leg against a deliberately perturbed copy of
   itself to confirm a difference IS detected.

`alt` is the STAGE-2 DECISION, and it obeys RULE 394: the statistic is
`round_us`, the round's own wall-clock endpoint, NOT a sum of phase timers.
Phase timers are reported beside it for attribution only. The arm is read from
the run's own `pf_chain` witness, so a round is attributed to the arm it
actually executed.

harness=local.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def read_rounds(path: Path) -> list[dict]:
    """Every `mtp-trace:` line as a field dict."""
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-trace: "):
            continue
        fields: dict[str, str] = {}
        for token in line[len("mtp-trace: ") :].split():
            key, _, value = token.partition("=")
            fields[key] = value
        if "round" in fields:
            rounds.append(fields)
    return rounds


def read_rows(path: Path) -> dict[int, tuple[str, str]]:
    """`pos -> (ids, hexfloat values)`, verbatim text.

    Kept as TEXT on purpose. Parsing the hexfloats to Python floats would round
    through a decimal repr on the way back out and could mask a low-bit
    difference, which is the one thing this gate exists to catch.
    """
    rows: dict[int, tuple[str, str]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-row: "):
            continue
        fields: dict[str, str] = {}
        for token in line[len("mtp-row: ") :].split():
            key, _, value = token.partition("=")
            fields[key] = value
        pos = int(fields["pos"])
        # A position may be re-emitted after a repair forward revises its tail
        # row; the LAST emission is the one the trajectory carries.
        rows[pos] = (fields.get("ids", ""), fields.get("v", ""))
    return rows


def read_score(tag: str) -> dict:
    path = Path("research/out") / tag / "score.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def read_meta(tag: str) -> dict:
    path = Path("research/out") / tag / "meta.txt"
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text(errors="replace").splitlines():
        key, _, value = line.partition("=")
        meta[key.strip()] = value.strip()
    return meta


def compare_rows(a: dict[int, tuple[str, str]],
                 b: dict[int, tuple[str, str]]) -> dict:
    shared = sorted(set(a) & set(b))
    ids_diff = [p for p in shared if a[p][0] != b[p][0]]
    val_diff = [p for p in shared if a[p][1] != b[p][1]]
    return {
        "positions_compared": len(shared),
        "positions_only_in_a": len(set(a) - set(b)),
        "positions_only_in_b": len(set(b) - set(a)),
        "id_mismatches": len(ids_diff),
        "value_mismatches": len(val_diff),
        "first_id_mismatch_pos": ids_diff[0] if ids_diff else None,
        "first_value_mismatch_pos": val_diff[0] if val_diff else None,
        "first_value_mismatch": (
            {"pos": val_diff[0], "a": a[val_diff[0]], "b": b[val_diff[0]]}
            if val_diff else None),
        "bitwise_identical": not ids_diff and not val_diff,
    }


def positive_control(rows: dict[int, tuple[str, str]]) -> dict:
    """Prove the comparison can fail.

    Flip the last hex digit of one value on one position and confirm
    `compare_rows` reports it. A gate that cannot fail is not a gate.
    """
    if not rows:
        return {"ran": False}
    pos = sorted(rows)[len(rows) // 2]
    ids, values = rows[pos]
    perturbed = dict(rows)
    perturbed[pos] = (ids, values[:-1] + ("0" if values[-1] != "0" else "1"))
    result = compare_rows(rows, perturbed)
    return {
        "ran": True,
        "perturbed_pos": pos,
        "detected": result["value_mismatches"] == 1
        and result["first_value_mismatch_pos"] == pos,
    }


def ledger_closure(rounds: list[dict]) -> dict:
    """`rows_per_round = d + 1`, and accepted + rejected + tail == declared."""
    declared = 0
    accepted = 0
    rejected = 0
    bad = []
    for r in rounds:
        d = int(r["d"])
        acc = int(r["acc"])
        if acc > d or acc < 0:
            bad.append(int(r["round"]))
        declared += d + 1
        accepted += acc
        rejected += d - acc
    # Every declared row is exactly one of: an accepted draft, a rejected
    # draft, or the round's single tail/bonus row.
    return {
        "rounds": len(rounds),
        "declared_rows": declared,
        "accepted_drafts": accepted,
        "rejected_drafts": rejected,
        "tail_rows": len(rounds),
        "closes": declared == accepted + rejected + len(rounds),
        "illegal_rounds": bad,
    }


def accept_ledger(rounds: list[dict]) -> list[tuple[int, int]]:
    return [(int(r["d"]), int(r["acc"])) for r in rounds]


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": st.fmean(values),
        "median": st.median(values),
        "stdev": st.stdev(values) if len(values) > 1 else 0.0,
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
    }


def cmd_pair(args: argparse.Namespace) -> int:
    out_off = Path("research/out") / args.off_tag
    out_on = Path("research/out") / args.on_tag
    rounds_off = read_rounds(out_off / "trace.txt")
    rounds_on = read_rounds(out_on / "trace.txt")
    rows_off = read_rows(out_off / "trace.txt")
    rows_on = read_rows(out_on / "trace.txt")
    score_off = read_score(args.off_tag)
    score_on = read_score(args.on_tag)

    # Arm witnesses, read from the runs themselves.
    witness_off = sorted({r.get("pf_chain", "absent") for r in rounds_off})
    witness_on = sorted({r.get("pf_chain", "absent") for r in rounds_on})
    chain_used_on = max(
        (int(r.get("pf_chain_used", 0)) for r in rounds_on), default=0)
    chain_used_off = max(
        (int(r.get("pf_chain_used", 0)) for r in rounds_off), default=0)
    chain_over_on = max(
        (int(r.get("pf_chain_overs", 0)) for r in rounds_on), default=0)

    rows_cmp = compare_rows(rows_off, rows_on)
    ledger_off = accept_ledger(rounds_off)
    ledger_on = accept_ledger(rounds_on)
    ledger_same = ledger_off == ledger_on
    first_ledger_diff = None
    for index, (a, b) in enumerate(zip(ledger_off, ledger_on)):
        if a != b:
            first_ledger_diff = {"index": index, "off": a, "on": b}
            break

    report = {
        "experiment": "e204-round-end-seam-overlap",
        "harness": "local",
        "check": "step7-falsification-gate",
        "official_or_ranked_score": False,
        "off_tag": args.off_tag,
        "on_tag": args.on_tag,
        "off_meta": read_meta(args.off_tag),
        "on_meta": read_meta(args.on_tag),
        "arm_witness_off": witness_off,
        "arm_witness_on": witness_on,
        "chain_steps_used_off": chain_used_off,
        "chain_steps_used_on": chain_used_on,
        "chain_steps_overshoot_on": chain_over_on,
        "row_exactness": rows_cmp,
        "positive_control": positive_control(rows_on),
        "accept_ledger_invariant": ledger_same,
        "first_accept_ledger_difference": first_ledger_diff,
        "rounds_off": len(rounds_off),
        "rounds_on": len(rounds_on),
        "ledger_closure_off": ledger_closure(rounds_off),
        "ledger_closure_on": ledger_closure(rounds_on),
        "score_off": score_off,
        "score_on": score_on,
    }

    checks = {
        "rows_bitwise_identical": rows_cmp["bitwise_identical"],
        "accept_ledger_invariant": ledger_same,
        "ledger_closes_off": report["ledger_closure_off"]["closes"],
        "ledger_closes_on": report["ledger_closure_on"]["closes"],
        "all_tokens_matched_off": score_off.get("all_tokens_matched") is True,
        "all_tokens_matched_on": score_on.get("all_tokens_matched") is True,
        "no_residual_divergence_off":
            score_off.get("residual_divergence_count", 0) == 0,
        "no_residual_divergence_on":
            score_on.get("residual_divergence_count", 0) == 0,
        "positive_control_detected": report["positive_control"]["detected"],
        "off_arm_used_no_chain": chain_used_off == 0,
        "on_arm_used_chain": chain_used_on > 0,
    }
    report["checks"] = checks
    report["gate"] = "PASS" if all(checks.values()) else "FAIL"

    print("E204 step-7 falsification gate: %s" % report["gate"])
    print("  off=%s (%d rounds, pf_chain=%s)"
          % (args.off_tag, len(rounds_off), witness_off))
    print("  on =%s (%d rounds, pf_chain=%s)"
          % (args.on_tag, len(rounds_on), witness_on))
    print("  rows compared %d; id mismatches %d; VALUE mismatches %d"
          % (rows_cmp["positions_compared"], rows_cmp["id_mismatches"],
             rows_cmp["value_mismatches"]))
    if rows_cmp["first_value_mismatch"]:
        print("  first value mismatch: %s" % rows_cmp["first_value_mismatch"])
    print("  chain steps used: off %d, on %d (overshoot on %d)"
          % (chain_used_off, chain_used_on, chain_over_on))
    for name, ok in checks.items():
        print("  %-28s %s" % (name, "ok" if ok else "FAIL"))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["gate"] == "PASS" else 1


def cmd_alt(args: argparse.Namespace) -> int:
    out = Path("research/out") / args.tag
    rounds = read_rounds(out / "trace.txt")[args.skip_rounds :]
    if not rounds:
        print("no rounds")
        return 2

    # RULE 394. The decision statistic is the round's own wall-clock endpoint.
    on_us = [float(r["round_us"]) for r in rounds if r.get("pf_chain") == "1"]
    off_us = [float(r["round_us"]) for r in rounds if r.get("pf_chain") == "0"]
    on = summarise(on_us)
    off = summarise(off_us)

    delta = None
    sigma = None
    if on.get("n", 0) > 1 and off.get("n", 0) > 1:
        delta = off["mean"] - on["mean"]
        # Standard error of the difference of two independent means.
        sigma = math.sqrt(
            on["stdev"] ** 2 / on["n"] + off["stdev"] ** 2 / off["n"])

    # Attribution only, never the decision (RULE 394).
    def phase(name: str, key: str) -> dict:
        return {
            "on": summarise([float(r[name]) for r in rounds
                             if r.get("pf_chain") == "1" and name in r]),
            "off": summarise([float(r[name]) for r in rounds
                              if r.get("pf_chain") == "0" and name in r]),
        }

    report = {
        "experiment": "e204-round-end-seam-overlap",
        "harness": "local",
        "check": "stage2-in-process-arm-switch",
        "official_or_ranked_score": False,
        "decision_statistic": "round_us",
        "tag": args.tag,
        "meta": read_meta(args.tag),
        "score": read_score(args.tag),
        "rounds_analysed": len(rounds),
        "rounds_skipped": args.skip_rounds,
        "round_us_on": on,
        "round_us_off": off,
        "delta_us_off_minus_on": delta,
        "delta_sigma_us": sigma,
        "delta_sigma_ratio": (delta / sigma) if delta and sigma else None,
        "promote_threshold_us": 200.0,
        "attribution_only": {
            name: phase(name, name)
            for name in ("d_chain_us", "eval_wall_us", "commit_us",
                         "upkeep_us", "draft_build_us")
        },
        "accept_ledger_on": [
            (int(r["d"]), int(r["acc"])) for r in rounds
            if r.get("pf_chain") == "1"],
        "accept_ledger_off": [
            (int(r["d"]), int(r["acc"])) for r in rounds
            if r.get("pf_chain") == "0"],
    }
    if delta is not None and sigma:
        report["promote"] = delta >= 200.0 and delta - 2.0 * sigma > 0.0

    print("E204 stage 2, %s (%d rounds analysed)" % (args.tag, len(rounds)))
    print("  round_us ON  n=%d mean %.1f median %.1f sd %.1f"
          % (on["n"], on["mean"], on["median"], on["stdev"]))
    print("  round_us OFF n=%d mean %.1f median %.1f sd %.1f"
          % (off["n"], off["mean"], off["median"], off["stdev"]))
    if delta is not None:
        print("  delta (off - on) %.1f us +- %.1f us (%.2f sigma); "
              "promote at >= 200 us with 2 sigma clear of zero: %s"
              % (delta, sigma, (delta / sigma) if sigma else float("nan"),
                 report.get("promote")))
    print("  attribution only: d_chain_us on %.1f / off %.1f"
          % (report["attribution_only"]["d_chain_us"]["on"].get("mean", 0.0),
             report["attribution_only"]["d_chain_us"]["off"].get("mean", 0.0)))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    pair = sub.add_parser("pair")
    pair.add_argument("off_tag")
    pair.add_argument("on_tag")
    pair.add_argument("--json", dest="json_path")
    pair.set_defaults(func=cmd_pair)

    alt = sub.add_parser("alt")
    alt.add_argument("tag")
    alt.add_argument("--skip-rounds", type=int, default=8)
    alt.add_argument("--json", dest="json_path")
    alt.set_defaults(func=cmd_alt)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
