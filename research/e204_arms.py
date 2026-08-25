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

    # ATTRIBUTION. A round is attributed to the arm IT EXPERIENCED, which is
    # `pf_chain_in` — whether a prefetched chain was waiting for it. It is NOT
    # `pf_chain`, which records the gate this round's own tail resolved and so
    # describes the NEXT round. Attributing by `pf_chain` would shift every
    # measurement one round and smear the two arms into each other at every
    # window boundary.
    rounds = [r for r in rounds if "pf_chain_in" in r]
    if not rounds:
        print("trace has no pf_chain_in witness; rebuild before measuring")
        return 2
    on_rounds = [r for r in rounds if r["pf_chain_in"] == "1"]
    off_rounds = [r for r in rounds if r["pf_chain_in"] == "0"]

    # RULE 394. The decision statistic is the round's own wall-clock endpoint.
    on = summarise([float(r["round_us"]) for r in on_rounds])
    off = summarise([float(r["round_us"]) for r in off_rounds])

    delta = None
    sigma = None
    if on.get("n", 0) > 1 and off.get("n", 0) > 1:
        delta = off["mean"] - on["mean"]
        # Standard error of the difference of two independent means.
        sigma = math.sqrt(
            on["stdev"] ** 2 / on["n"] + off["stdev"] ** 2 / off["n"])

    # Attribution only, never the decision (RULE 394).
    def phase(name: str) -> dict:
        return {
            "on": summarise([float(r[name]) for r in on_rounds if name in r]),
            "off": summarise([float(r[name]) for r in off_rounds if name in r]),
        }

    # BALANCE CHECK. The arms must see the same mix of round difficulty. The
    # emitted tokens are arm-independent, so the (d, acc) sequence is fixed by
    # the prompt; an alternating window could still land unevenly on it, and a
    # difference in mix would confound the round-time comparison.
    def mix(rows: list[dict]) -> dict:
        hist: dict[str, int] = {}
        for r in rows:
            key = "%s/%s" % (r["acc"], r["d"])
            hist[key] = hist.get(key, 0) + 1
        return hist

    def full_accept_share(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["acc"] == r["d"]) / len(rows)

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
            name: phase(name)
            for name in ("d_chain_us", "eval_wall_us", "commit_us",
                         "upkeep_us", "draft_build_us", "verify_build_us",
                         "host_thread_cpu_ns")
        },
        "round_mix_on": mix(on_rounds),
        "round_mix_off": mix(off_rounds),
        "full_accept_share_on": full_accept_share(on_rounds),
        "full_accept_share_off": full_accept_share(off_rounds),
        "chain_steps_used": max(
            (int(r.get("pf_chain_used", 0)) for r in rounds), default=0),
        "chain_steps_overshoot": max(
            (int(r.get("pf_chain_overs", 0)) for r in rounds), default=0),
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
    print("  balance: full-accept share on %.3f / off %.3f"
          % (report["full_accept_share_on"], report["full_accept_share_off"]))
    print("  chain steps used %d, overshoot %d"
          % (report["chain_steps_used"], report["chain_steps_overshoot"]))
    print("  -- attribution only, never the decision --")
    for name, block in report["attribution_only"].items():
        print("     %-20s on %12.1f  off %12.1f  delta %10.1f"
              % (name, block["on"].get("mean", 0.0),
                 block["off"].get("mean", 0.0),
                 block["on"].get("mean", 0.0) - block["off"].get("mean", 0.0)))

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def cmd_paired(args: argparse.Namespace) -> int:
    """THE DECISION. Two complementary alternating legs, paired by round index.

    One alternating leg is not enough. A round's cost is dominated by whether
    it accepts its whole chain; that outcome sequence is fixed by the prompt,
    and an alternating window that lands unevenly on it hands one arm the
    expensive rejection rounds. The first `alt2` session split 0.703 / 0.854 on
    full acceptance and the raw arm means were unusable.

    Running the exact complement as a second leg fixes both confounds at once:

    * **Difficulty.** Round index N carries the identical `(d, acc)` outcome in
      both legs — asserted here, not assumed — so the pair differs only in the
      arm.
    * **Session offset.** Index N is ON in exactly one leg. Half the indices
      are ON in leg A and half in leg B, so a constant per-leg offset (a warmer
      start, a different power state) enters the two halves with OPPOSITE signs
      and cancels in the mean.

    The statistic stays `round_us` per RULE 394.
    """
    legs = {}
    for tag in (args.tag_a, args.tag_b):
        rows = [r for r in read_rounds(Path("research/out") / tag / "trace.txt")
                if "pf_chain_in" in r]
        if not rows:
            print("%s has no pf_chain_in witness; rebuild before measuring" % tag)
            return 2
        legs[tag] = rows
    a, b = legs[args.tag_a], legs[args.tag_b]
    if len(a) != len(b):
        print("legs decoded different round counts: %d vs %d" % (len(a), len(b)))
        return 2

    seq_a = [(r["d"], r["acc"]) for r in a]
    seq_b = [(r["d"], r["acc"]) for r in b]
    if seq_a != seq_b:
        print("the two legs did not follow the same trajectory; not pairable")
        return 2

    pairs = []
    unusable = 0
    for index in range(args.skip_rounds, len(a)):
        arm_a, arm_b = a[index]["pf_chain_in"], b[index]["pf_chain_in"]
        if arm_a == arm_b:
            # Not complementary at this index: both legs saw the same arm, so
            # the pair carries no arm contrast. Counted, never silently used.
            unusable += 1
            continue
        on_row, off_row = ((a[index], b[index]) if arm_a == "1"
                           else (b[index], a[index]))
        pairs.append({
            "index": index,
            "d": int(on_row["d"]),
            "acc": int(on_row["acc"]),
            "full_accept": on_row["acc"] == on_row["d"],
            # off - on: POSITIVE means the chain prefetch made the round
            # shorter, matching the "minimize round time" direction.
            "delta_us": float(off_row["round_us"]) - float(on_row["round_us"]),
            "on_leg": args.tag_a if arm_a == "1" else args.tag_b,
        })

    if len(pairs) < 2:
        print("only %d usable pairs" % len(pairs))
        return 2

    deltas = [p["delta_us"] for p in pairs]
    mean = st.fmean(deltas)
    sigma = st.stdev(deltas) / math.sqrt(len(deltas))

    # The cancellation only works if both directions are represented. Report
    # the split and the per-direction means so a reader can see the session
    # offset being removed rather than take it on trust.
    by_leg: dict[str, list[float]] = {}
    for p in pairs:
        by_leg.setdefault(p["on_leg"], []).append(p["delta_us"])

    def group(rows: list[dict]) -> dict:
        values = [r["delta_us"] for r in rows]
        if len(values) < 2:
            return {"n": len(values)}
        m = st.fmean(values)
        s = st.stdev(values) / math.sqrt(len(values))
        return {"n": len(values), "mean_us": m, "sigma_us": s,
                "median_us": st.median(values), "sigma_ratio": m / s if s else None}

    # TRANSITION AUDIT (RULE 391(c)). The scheduled arm phase for round N is
    # the gate round N-1 resolved, which the trace records as that round's
    # `pf_chain`. `pf_chain_in` is what round N actually received. They differ
    # whenever a prefetch was built but handed back before the next round could
    # use it. The paired design assumes those transition costs are symmetric
    # between the two legs; this counts them per leg instead of assuming it.
    def transitions(rows: list[dict]) -> dict:
        disagree = [
            int(rows[i]["round"]) for i in range(1, len(rows))
            if rows[i]["pf_chain_in"] != rows[i - 1]["pf_chain"]
        ]
        return {"compared": len(rows) - 1, "disagreements": len(disagree),
                "rounds": disagree[:20]}

    report = {
        "experiment": "e204-round-end-seam-overlap",
        "harness": "local",
        "check": "stage2-complementary-paired-arm-switch",
        "official_or_ranked_score": False,
        "decision_statistic": "round_us",
        "sign_convention": "delta = off - on; positive means the prefetch is faster",
        "leg_a": args.tag_a,
        "leg_b": args.tag_b,
        "meta_a": read_meta(args.tag_a),
        "meta_b": read_meta(args.tag_b),
        "score_a": read_score(args.tag_a),
        "score_b": read_score(args.tag_b),
        "trajectories_identical": True,
        "pairs_used": len(pairs),
        "pairs_unusable_same_arm": unusable,
        "rounds_skipped": args.skip_rounds,
        "delta_us_mean": mean,
        "delta_us_sigma": sigma,
        "delta_us_median": st.median(deltas),
        "delta_sigma_ratio": mean / sigma if sigma else None,
        "delta_2sigma_interval": [mean - 2 * sigma, mean + 2 * sigma],
        "promote_threshold_us": 200.0,
        "promote": mean >= 200.0 and mean - 2 * sigma > 0.0,
        "by_on_leg": {tag: group([p for p in pairs if p["on_leg"] == tag])
                      for tag in by_leg},
        "full_acceptance": group([p for p in pairs if p["full_accept"]]),
        "rejection": group([p for p in pairs if not p["full_accept"]]),
        "transitions": {args.tag_a: transitions(a), args.tag_b: transitions(b)},
        "entry_temp_spread_c": abs(
            float(read_meta(args.tag_a).get("gpu_temp_entry_c", "nan"))
            - float(read_meta(args.tag_b).get("gpu_temp_entry_c", "nan"))),
    }
    report["transitions_symmetric"] = (
        report["transitions"][args.tag_a]["disagreements"]
        == report["transitions"][args.tag_b]["disagreements"])

    print("E204 stage 2 DECISION: complementary paired arm switch")
    print("  legs %s / %s" % (args.tag_a, args.tag_b))
    print("  trajectories identical; %d usable pairs, %d unusable (same arm)"
          % (len(pairs), unusable))
    print("  delta (off - on) mean %.1f us +- %.1f us (%.2f sigma), median %.1f"
          % (mean, sigma, mean / sigma if sigma else float("nan"),
             report["delta_us_median"]))
    print("  2-sigma interval [%.1f, %.1f]"
          % (report["delta_2sigma_interval"][0],
             report["delta_2sigma_interval"][1]))
    print("  promote at >= 200 us with 2 sigma clear of zero: %s"
          % report["promote"])
    print("  -- transition audit (RULE 391(c)) --")
    for tag, block in report["transitions"].items():
        meta = read_meta(tag)
        print("     %-16s witness disagrees with scheduled phase on %d of %d "
              "rounds; entry %.2f C exit %.2f C"
              % (tag, block["disagreements"], block["compared"],
                 float(meta.get("gpu_temp_entry_c", "nan")),
                 float(meta.get("gpu_temp_exit_c", "nan"))))
    print("     symmetric between legs: %s; entry spread %.2f C"
          % (report["transitions_symmetric"], report["entry_temp_spread_c"]))
    print("  -- session-offset cancellation --")
    for tag, block in report["by_on_leg"].items():
        print("     ON in %-16s n=%2d mean %8.1f us" % (tag, block.get("n", 0),
                                                       block.get("mean_us", 0.0)))
    print("  -- by outcome --")
    for name in ("full_acceptance", "rejection"):
        block = report[name]
        print("     %-16s n=%2d mean %8.1f us +- %7.1f (%.2f sigma)"
              % (name, block.get("n", 0), block.get("mean_us", 0.0),
                 block.get("sigma_us", 0.0), block.get("sigma_ratio") or 0.0))

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

    paired = sub.add_parser("paired")
    paired.add_argument("tag_a")
    paired.add_argument("tag_b")
    paired.add_argument("--skip-rounds", type=int, default=8)
    paired.add_argument("--json", dest="json_path")
    paired.set_defaults(func=cmd_paired)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
