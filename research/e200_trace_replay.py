#!/usr/bin/env python3
"""E200 step 2: exact offline replay of REAL per-round state under each price.

This removes the latent-q proxy of `research/e200_desk_price.py`. Every round
here is a real round of a real leg, with the exact `ema=`, `m=`, `cap=` and
`offer` the shipped scheduler saw, and the exact `d=` and `acc=` it produced.

RULE 79 carve-out. This is an offline replay against a measured cost table, not
a timing contrast between two legs. Two measured tables are used and every
printed figure names which one it came from:

  * `local`  -- median `round_us` by verify width m = 1 + d, taken from the
                SAME ship leg whose policy state is being replayed. Widths the
                leg never visited are filled by an affine map fitted to the
                ranked law on the widths it did visit, and are reported.
  * `ranked` -- the merged E197 smooth-step law R(1..9) in ms, fitted on paid
                ranked receipts.

WHAT IS EXACT AND WHAT IS NOT.

  * The policy shift is exact. `costModelDepth` is a pure function of the
    round's snapshotted state, and `research/e128_replay.py` reproduces the
    shipped `sched=` string byte for byte, so a re-run with a different price
    table is the true one-step decision of that round.
  * The acceptance under a DOWN shift is exact. The proposal head builds draft
    positions 1..d' the same way whether the round asks for d' or for d > d'
    drafts, and acceptance is a prefix rule, so acc' = min(acc, d').
  * An UP shift is not exact, because the extra draft tokens were never
    verified. Those rounds are reported separately and bracketed by
    acc' = acc (pessimistic) and acc' = d' (optimistic).
  * The EMA feedback loop is NOT closed here. A changed depth changes
    `recordAcceptOutcome`, which changes every later round's state. That is
    what the live `MLX_E200_DEPTH_PRICE` leg measures; this file freezes the
    shipped state on purpose so the price is the only thing that moves.

  usage: research/e200_trace_replay.py SHIP_LEG_DIR [--live LIVE_LEG_DIR]
                                       [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_replay as R128  # noqa: E402
import e200_desk_price as DESK  # noqa: E402

ROUND_US_RE = __import__("re").compile(r"\bround_us=(\d+)\b")
CAP = DESK.CAP
MAXD = DESK.MAXD
# E197 merged smooth-step ranked round-cost law, R(m) in ms for m = 1..9.
RANKED_R_MS = [
    30.260724998120555, 31.666388233791732, 33.278091533039610,
    35.678159248309770, 40.435096101831750, 48.360761661689416,
    55.231766557393320, 58.796862540567346, 60.759154322285190,
]
SHIP_LEVEL = CAP * DESK.SHIP_H  # 1.26, the level the shipped price spends


def read_rounds_with_cost(run_dir: pathlib.Path) -> list[dict]:
    """`e128_replay.read_rounds` plus the round's own wall time."""
    rounds = R128.read_rounds(run_dir)
    costs = []
    for line in (run_dir / "trace.txt").read_text(errors="replace").splitlines():
        if not R128.ROUND_RE.match(line):
            continue
        if not R128.SIGNAL_RE.search(line):
            continue
        hit = ROUND_US_RE.search(line)
        costs.append(int(hit.group(1)) if hit else None)
    for record, cost in zip(rounds, costs):
        record["round_us"] = cost
    return rounds


def attach_offers(rounds: list[dict], total_tokens: int,
                  offered_depth: int = MAXD) -> None:
    """`QwenRuntimeMTPDriver` narrows the offer at the tail of the window."""
    emitted = 0
    for record in rounds:
        record["offer"] = max(
            1, min(offered_depth, MAXD, total_tokens - emitted - 1))
        emitted += min(1 + record["accepted"], total_tokens - emitted)


def local_cost_table(rounds: list[dict], min_samples: int = 3) -> dict:
    """Measured R(m) in ms pooled over legs, ranked-law filled where unseen.

    Round 1 of every leg is dropped: it pays the cold verify build, which is a
    warmup cost of the leg and not a cost of the width.
    """
    by_width: dict[int, list[float]] = {}
    for record in rounds:
        if record["round_us"] is None or record["round"] == 1:
            continue
        by_width.setdefault(1 + record["depth"], []).append(
            record["round_us"] / 1000.0)
    seen = {m: statistics.median(v) for m, v in by_width.items()
            if len(v) >= min_samples}
    if len(seen) < 2:
        return {"table": None, "seen": {}, "filled": list(range(1, MAXD + 2))}
    xs = [RANKED_R_MS[m - 1] for m in sorted(seen)]
    ys = [seen[m] for m in sorted(seen)]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
             if var > 0 else 1.0)
    intercept = my - slope * mx
    table, filled = [], []
    for m in range(1, MAXD + 2):
        if m in seen:
            table.append(seen[m])
        else:
            table.append(slope * RANKED_R_MS[m - 1] + intercept)
            filled.append(m)
    return {
        "table": table,
        "seen": {m: seen[m] for m in sorted(seen)},
        "samples": {m: len(v) for m, v in sorted(by_width.items())},
        "filled": filled,
        "fill_slope": slope,
        "fill_intercept": intercept,
    }


def _sched_fields(sched: str) -> list[float]:
    """Every printed number of a `d:p/reach/threshold;` walk, in order."""
    out = []
    for step in sched.split(";"):
        if not step:
            continue
        _, _, values = step.partition(":")
        out.extend(float(v) for v in values.split("/"))
    return out


def witness_positive_control(rounds: list[dict], price: dict) -> int:
    """Prove the witness comparison can fail.

    Replay the same rounds under a DIFFERENT price and report how far the
    printed fields move. A witness that cannot fail proves nothing.
    """
    worst = 0
    for record in rounds:
        _, sched, _ = R128.cost_model_depth(
            record["ema"], record["margin"],
            offered_depth=record["offer"], width_cap=record["cap"],
            marginal=price["marginal"], cumulative=price["cumulative"])
        shipped_fields = _sched_fields(record["sched"])
        other_fields = _sched_fields(sched)
        pairs = zip(shipped_fields, other_fields)
        worst = max(worst, max((abs(round((a - b) * 1e6)) for a, b in pairs),
                               default=0))
    return worst


def closed_form_witness(rounds: list[dict]) -> dict:
    """RULE 391(b) arm witness: the printed thresholds ARE the uniform price.

    Every `sched=` step prints `d:p/reach/threshold`. Under
    `makeUniformDepthPrice` the threshold has the closed form
    `0.18 * (1 + expected) / (1 + 0.18 d)`, where `expected` is the running sum
    of `reach` over the accepted steps of the same walk. A byte-identical
    rebuild of the whole walk string therefore proves the shipped binary is
    running that price and nothing else.

    The trace prints each field as `%.6f`, so a value that lands on a rounding
    boundary can differ in its last printed digit while the underlying double
    is the same. `field_max_ulp` is that distance in units of the last printed
    place. It must stay at or below 1 for the witness to hold; a real price
    change moves these fields by thousands of units.
    """
    checked = 0
    steps = 0
    exact = 0
    depth_hits = 0
    max_ulp = 0
    mismatches = []
    for record in rounds:
        depth, sched, _ = R128.cost_model_depth(
            record["ema"], record["margin"],
            offered_depth=record["offer"], width_cap=record["cap"])
        checked += 1
        steps += record["sched"].count(";")
        exact += sched == record["sched"]
        depth_hits += depth == record["depth"]
        shipped_fields = _sched_fields(record["sched"])
        replay_fields = _sched_fields(sched)
        if len(shipped_fields) != len(replay_fields):
            mismatches.append({
                "round": record["round"], "kind": "shape",
                "shipped_sched": record["sched"], "replayed_sched": sched})
            continue
        ulp = max((abs(round((a - b) * 1e6))
                   for a, b in zip(shipped_fields, replay_fields)), default=0)
        max_ulp = max(max_ulp, ulp)
        if ulp > 1 or depth != record["depth"]:
            mismatches.append({
                "round": record["round"], "kind": "value", "ulp": ulp,
                "shipped_sched": record["sched"], "replayed_sched": sched,
                "shipped_depth": record["depth"], "replayed_depth": depth})
    return {
        "rounds_checked": checked,
        "steps_checked": steps,
        "sched_byte_identical": exact / checked if checked else 0.0,
        "depth_agreement": depth_hits / checked if checked else 0.0,
        "field_max_ulp": max_ulp,
        "sched_agreement": (checked - len(mismatches)) / checked if checked else 0.0,
        "mismatches": mismatches[:8],
    }


def replay_price(rounds: list[dict], price: dict, cost_ms: list[float]) -> dict:
    """One-step policy shift of every real round under `price`."""
    ship_ms = 0.0
    ship_tokens = 0
    cand_ms = 0.0
    cand_tok_pess = 0
    cand_tok_opt = 0
    up = 0
    down = 0
    same = 0
    hist_ship = [0] * (MAXD + 1)
    hist_cand = [0] * (MAXD + 1)
    for record in rounds:
        d = record["depth"]
        acc = record["accepted"]
        ship_ms += cost_ms[d]
        ship_tokens += 1 + acc
        hist_ship[d] += 1
        d2, _, _ = R128.cost_model_depth(
            record["ema"], record["margin"],
            offered_depth=record["offer"], width_cap=record["cap"],
            marginal=price["marginal"], cumulative=price["cumulative"])
        hist_cand[d2] += 1
        cand_ms += cost_ms[d2]
        if d2 < d:
            down += 1
            acc2 = min(acc, d2)
            cand_tok_pess += 1 + acc2
            cand_tok_opt += 1 + acc2
        elif d2 > d:
            up += 1
            cand_tok_pess += 1 + acc
            cand_tok_opt += 1 + d2
        else:
            same += 1
            cand_tok_pess += 1 + acc
            cand_tok_opt += 1 + acc
    ship_mspt = ship_ms / ship_tokens
    out = {
        "rounds": len(rounds),
        "ship_ms_per_token": ship_mspt,
        "cand_ms_per_token_pessimistic": cand_ms / cand_tok_pess,
        "cand_ms_per_token_optimistic": cand_ms / cand_tok_opt,
        "shift_down": down, "shift_same": same, "shift_up": up,
        "ship_depth_hist": hist_ship,
        "cand_depth_hist": hist_cand,
        "ship_edl": sum(i * n for i, n in enumerate(hist_ship)) / len(rounds),
        "cand_edl": sum(i * n for i, n in enumerate(hist_cand)) / len(rounds),
        "exact_token_accounting": up == 0,
    }
    for key in ("pessimistic", "optimistic"):
        out["speedup_" + key] = ship_mspt / out["cand_ms_per_token_" + key]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ship_dir", type=pathlib.Path)
    parser.add_argument("--live", type=pathlib.Path)
    parser.add_argument("--cost-leg", type=pathlib.Path, action="append",
                        default=[],
                        help="extra leg pooled into the measured local R(m); "
                             "the shallow arm covers the widths the shipped "
                             "arm never visits")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    rounds = read_rounds_with_cost(args.ship_dir)
    if not rounds:
        print("no traced rounds in %s" % args.ship_dir, file=sys.stderr)
        return 2
    attach_offers(rounds, args.tokens)
    meta = R128.read_meta(args.ship_dir)

    witness = closed_form_witness(rounds)
    cost_rounds = list(rounds)
    for extra in args.cost_leg:
        cost_rounds.extend(read_rounds_with_cost(extra))
    local = local_cost_table(cost_rounds)
    local["pooled_legs"] = [str(args.ship_dir)] + [str(p) for p in args.cost_leg]

    print("E200 step 2 -- exact per-round replay")
    print("leg          %s" % args.ship_dir)
    print("arm          %s" % meta.get("e200_arm", "?"))
    print("worker       %s" % meta.get("worker_sha256_before", "?")[:16])
    print("head_class   %s" % meta.get("head_class", "?"))
    print("rounds       %d" % len(rounds))
    print()
    print("RULE 391(b) arm witness -- shipped sched= string vs uniform price")
    print("  rounds checked        %d" % witness["rounds_checked"])
    print("  walk steps            %d" % witness["steps_checked"])
    print("  depth agreement       %.6f" % witness["depth_agreement"])
    print("  sched byte identical  %.6f" % witness["sched_byte_identical"])
    print("  sched agreement <=1ulp %.6f" % witness["sched_agreement"])
    control = witness_positive_control(
        rounds, DESK.table_price(RANKED_R_MS, level=SHIP_LEVEL))
    witness["positive_control_max_ulp"] = control
    print("  max field distance    %d units in the last printed place"
          % witness["field_max_ulp"])
    print("  positive control      %d units under the rstep price "
          "(the comparison can fail)" % control)
    for bad in witness["mismatches"]:
        print("  MISMATCH round %d: %s != %s"
              % (bad["round"], bad["replayed_sched"], bad["shipped_sched"]))
    print()
    print("measured local cost table (ms by verify width m = 1 + d)")
    print("  samples %s" % local.get("samples"))
    print("  filled  %s" % local.get("filled"))
    if local["table"]:
        print("  R_local %s" % ["%.2f" % v for v in local["table"]])
    print("  R_ranked %s" % ["%.2f" % v for v in RANKED_R_MS])
    print()

    arms = {
        "ship_uniform_0.18": DESK.ship_price(),
        "rstep_level_held": DESK.table_price(RANKED_R_MS, level=SHIP_LEVEL),
        "rstepnat_measured_level": DESK.table_price(RANKED_R_MS, level=None),
    }
    for w in (0.1, 0.2, 0.4, 0.6, 0.8):
        arms["blend_w%.1f" % w] = DESK.blend_price(
            RANKED_R_MS, w, SHIP_LEVEL)

    tables = {"ranked": RANKED_R_MS}
    if local["table"]:
        tables["local"] = local["table"]

    results = {}
    for table_name, cost_ms in tables.items():
        print("cost table = %s   (harness=%s price, local per-round state)"
              % (table_name, table_name))
        print("  %-26s %9s %9s %9s %9s %6s %6s"
              % ("arm", "ms/token", "speedup", "edl", "d_hist", "down", "up"))
        for arm, price in arms.items():
            got = replay_price(rounds, price, cost_ms)
            results["%s/%s" % (table_name, arm)] = got
            print("  %-26s %9.4f %9.5f %9.3f %9s %6d %6d"
                  % (arm, got["cand_ms_per_token_pessimistic"],
                     got["speedup_pessimistic"], got["cand_edl"],
                     "".join(str(min(9, n)) for n in got["cand_depth_hist"]),
                     got["shift_down"], got["shift_up"]))
        print()

    live_report = None
    if args.live:
        live_rounds = read_rounds_with_cost(args.live)
        attach_offers(live_rounds, args.tokens)
        live_meta = R128.read_meta(args.live)
        hist = [0] * (MAXD + 1)
        for record in live_rounds:
            hist[record["depth"]] += 1
        live_report = {
            "run_dir": str(args.live),
            "arm": live_meta.get("e200_arm"),
            "rounds": len(live_rounds),
            "depth_hist": hist,
            "edl": (sum(i * n for i, n in enumerate(hist)) / len(live_rounds)
                    if live_rounds else 0.0),
            "accepted_mean": (statistics.fmean(
                [r["accepted"] for r in live_rounds]) if live_rounds else 0.0),
        }
        key = ("ranked/rstep_level_held" if live_meta.get("e200_arm") == "rstep"
               else "ranked/rstepnat_measured_level")
        predicted = results.get(key)
        print("live arm %s: rounds=%d edl=%.3f hist=%s"
              % (live_report["arm"], live_report["rounds"],
                 live_report["edl"], live_report["depth_hist"]))
        if predicted:
            print("one-step prediction for the same arm: edl=%.3f hist=%s"
                  % (predicted["cand_edl"], predicted["cand_depth_hist"]))
            print("EMA-loop contribution to edl: %+.3f"
                  % (live_report["edl"] - predicted["cand_edl"]))

    payload = {
        "harness": "ranked-cost-law over local per-round policy state",
        "ship_leg": str(args.ship_dir),
        "ship_meta": meta,
        "witness": witness,
        "local_cost_table": local,
        "ranked_cost_table": RANKED_R_MS,
        "prices": {k: v for k, v in arms.items()},
        "results": results,
        "live": live_report,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
