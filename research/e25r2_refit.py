#!/usr/bin/env python3
"""Research-only (qwen38-r1-e25-per-row-draft-price, r2).

Refit the round-time curve T(d) and the per-position acceptance p_d on the
CURRENT base (d7619a7) and the DECLARED q2-q4-rerank head, from the forced-depth
tape produced by the FORCE arm, then score every depth policy that tape can
evaluate.

WHY THIS TAPE AND NOT r1's.  r1 fitted T(d) from the depths the adaptive rule
happened to choose: n = 193/995/583/167/9 at d = 1..5, no d = 0 round at all,
and the deep rounds were exactly the rounds where the head was hot.  The FORCE
arm cycles depth 0..7 round by round inside one leg, so position, prompt, cache
length and temperature are common mode by construction and every depth gets a
real sample.

WHY DEEPENING IS NOW EVALUABLE OFFLINE.  r1 could only truncate: the tape held
no verdict for a row the shipped rule never proposed.  Every FORCE round
proposes its cycle depth and the target verifies every row, so a round forced to
depth D carries the accept/reject verdict for rows 1..D.  Any policy that picks
D' <= D on that round has a known outcome, accepted = min(acc, D').  The replay
is still a counterfactual over a FIXED input tape -- the EMA trajectory that fed
the recorded `p` sequence was driven by the forced depths, not by the policy
under test -- but it no longer has to refuse credit for going deeper.

WHY round_us IS NOT THE FIT TARGET.  `upkeep_us` spans the trace's own per-row
top-2 dump (`Self.traceRow` over 0 ... acceptedCount, between tCommitDone and
tTailDone), so it scales with accepted rows and therefore with depth.  T(d) is
fitted from the phases before it and the round_us variant is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e21_trace import parse_trace  # noqa: E402

RUNS_ROOT = Path(".mlxfast-private/e25/runs")
PROMPTS = ("english", "narrative", "technical", "dramatic", "travel",
           "philosophy", "natural_history", "medicine")

SHIPPED_H = 0.18
# r1's fit, kept only so the r2 refit can be reported as a delta against it.
R1_ROW_STEP_RATIO = (0.0, 0.095904, 0.152261, 0.442442)

# Weight-stream passes per verify forward, from the affine-4 g64 crossrow
# dispatch table (mlx quantized.h). M = depth + 1 rows; IPG is the template's
# rows-per-group.  Tier A (out_vec_size >= 4096) covers every scored width.
IPG_BY_M = {1: None, 2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def weight_passes(depth: int) -> int:
    m = depth + 1
    ipg = IPG_BY_M.get(m)
    return 1 if ipg is None else math.ceil(m / ipg)


# --------------------------------------------------------------------------
# tape
# --------------------------------------------------------------------------
def load_force_tape(arm: str = "FORCE", prompts=PROMPTS,
                    runs_root: Path = RUNS_ROOT):
    """Pooled forced-depth rounds plus each prompt's two leg reports."""
    rounds: list[dict] = []
    legs: dict[str, dict] = {}
    missing: list[str] = []
    for prompt in prompts:
        run = runs_root / f"probe-{prompt}-{arm}"
        trace = run / "trace.txt"
        if not trace.exists():
            missing.append(str(run))
            continue
        sessions = parse_trace(trace)
        if len(sessions) != 1:
            raise SystemExit(
                f"{run}: expected one traced session, got {len(sessions)}")
        for r in sessions[0]:
            r["prompt"] = prompt
            rounds.append(r)
        serial = json.loads((run / "reports/03-mtp-timed.json").read_text())
        mtp = json.loads((run / "reports/04-mtp-timed.json").read_text())
        if not serial.get("is_serial_control") or mtp.get("is_serial_control"):
            raise SystemExit(f"{run}: leg reports are not (serial, mtp)")
        legs[prompt] = {"serial": serial, "mtp": mtp}
    if missing:
        raise SystemExit("e25r2_refit: no tape at " + ", ".join(missing))
    return rounds, legs


def fidelity(legs: dict) -> dict:
    """Every gate the trusted parent reports, per prompt, plus the head used."""
    out = {}
    for prompt, pair in legs.items():
        mtp = pair["mtp"]
        out[prompt] = {
            k: mtp.get(k) for k in (
                "all_tokens_matched", "residual_divergence_count",
                "parity_all_ok", "uses_pinned_mtp_head",
                "declared_head_digest", "decode_tokens", "round_count")
        }
    return out


def instrument_ok(rounds: list[dict]) -> dict:
    """The forced cycle must actually have driven the taken depth."""
    marked = [r for r in rounds if "forced_depth" in r]
    agree = sum(1 for r in marked if r["d"] == r["forced_depth"])
    return {
        "rounds": len(rounds),
        "rounds_carrying_forced_mark": len(marked),
        "taken_depth_equals_forced": agree,
        "forced_depth_histogram": dict(sorted(Counter(
            r["forced_depth"] for r in marked).items())),
        "taken_depth_histogram": dict(sorted(Counter(
            r["d"] for r in rounds).items())),
        "shipped_counterfactual_histogram": dict(sorted(Counter(
            r["shipped_depth"] for r in marked).items())),
    }


# --------------------------------------------------------------------------
# T(d)
# --------------------------------------------------------------------------
def core_us(r: dict) -> float:
    """Round cost with the trace's depth-scaling row dump excluded."""
    return r["round_us"] - r["upkeep_us"]


def time_table(rounds: list[dict], timer=core_us) -> dict:
    by: dict[int, list[float]] = defaultdict(list)
    for r in rounds:
        by[r["d"]].append(timer(r) / 1000.0)
    table = {}
    for d, v in sorted(by.items()):
        table[d] = {
            "n": len(v),
            "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "stdev_ms": statistics.stdev(v) if len(v) > 1 else 0.0,
            "sem_ms": (statistics.stdev(v) / math.sqrt(len(v))
                       if len(v) > 1 else 0.0),
            "weight_passes": weight_passes(d),
        }
    step = {}
    for d in sorted(table):
        if d + 1 in table:
            t0, t1 = table[d]["mean_ms"], table[d + 1]["mean_ms"]
            # The shipped rule's own definition: price of adding row d+1 as a
            # fraction of the round it extends (Qwen36MTPBlockSession :571).
            step[d] = (t1 - t0) / t0
    return {"round_ms": table, "step_ratio": step}


def admissibility(step: dict) -> dict:
    """A price c_d can fire at all only if c_d < 1/(d+1) (see e25r2_rule.py)."""
    out = {}
    for d, c in sorted(step.items()):
        ceiling = 1.0 / (d + 1)
        out[d] = {
            "measured_c": c,
            "ceiling": ceiling,
            "admissible": c < ceiling,
            "shipped_scalar_c": SHIPPED_H / (1.0 + d * SHIPPED_H),
            "r1_c": (R1_ROW_STEP_RATIO[d]
                     if d < len(R1_ROW_STEP_RATIO) else None),
        }
    return out


def position_control(rounds: list[dict], timer=core_us, quartiles: int = 4):
    """Replicate the depth curve inside token-offset quartiles."""
    seq = defaultdict(list)
    for r in sorted(rounds, key=lambda r: (r["prompt"], r["round"])):
        seq[r["prompt"]].append(r)
    for prompt, rs in seq.items():
        offset = 0
        for r in rs:
            r["offset"] = offset
            offset += 1 + r["acc"]
        for r in rs:
            r["quartile"] = min(quartiles - 1,
                                int(quartiles * r["offset"] / max(1, offset)))
    out = {}
    for q in range(quartiles):
        sub = [r for r in rounds if r.get("quartile") == q]
        out[f"q{q}"] = time_table(sub, timer)["round_ms"]
    xs = [r["offset"] for r in rounds]
    ys = [timer(r) / 1000.0 for r in rounds]
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return {"by_quartile": out, "ols_ms_per_1000_tokens": 1000.0 * slope,
            "n": n}


def pass_model(table: dict) -> dict:
    """Least squares T(d) = a + b*d + c*passes(d) over the measured means."""
    rows = [(d, v["mean_ms"], weight_passes(d)) for d, v in table.items()]
    if len(rows) < 3:
        return {"fitted": False}
    cols = [[1.0, float(d), float(p)] for d, _, p in rows]
    y = [t for _, t, _ in rows]
    k = 3
    a = [[sum(cols[i][r] * cols[i][c] for i in range(len(rows)))
          for c in range(k)] + [sum(cols[i][r] * y[i] for i in range(len(rows)))]
         for r in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        if abs(a[c][c]) < 1e-12:
            return {"fitted": False}
        for r in range(k):
            if r == c:
                continue
            f = a[r][c] / a[c][c]
            for j in range(c, k + 1):
                a[r][j] -= f * a[c][j]
    beta = [a[i][k] / a[i][i] for i in range(k)]
    pred = {d: beta[0] + beta[1] * d + beta[2] * p for d, _, p in rows}
    resid = [y[i] - pred[rows[i][0]] for i in range(len(rows))]
    sst = sum((t - statistics.mean(y)) ** 2 for t in y)
    sse = sum(e * e for e in resid)
    return {"fitted": True, "intercept_ms": beta[0], "per_row_ms": beta[1],
            "per_weight_pass_ms": beta[2],
            "r_squared": 1.0 - sse / sst if sst else 1.0,
            "predicted_ms": pred,
            "max_abs_residual_ms": max(abs(e) for e in resid)}


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------
def accept_table(rounds: list[dict]) -> dict:
    """Conditional accept probability of row i given rows < i were accepted."""
    prop = Counter()
    acc = Counter()
    for r in rounds:
        for i in range(r["d"]):
            if r["acc"] < i:
                break
            prop[i] += 1
            if r["acc"] > i:
                acc[i] += 1
    out = {}
    for i in sorted(prop):
        p = acc[i] / prop[i]
        out[i] = {"reached": prop[i], "accepted": acc[i], "p": p,
                  "sem": math.sqrt(max(p * (1 - p), 0.0) / prop[i])}
    return out


def expected_accepted(p_by_pos: dict, depth: int) -> float:
    reach, expected = 1.0, 0.0
    for i in range(depth):
        reach *= p_by_pos.get(i, {"p": 0.0})["p"]
        expected += reach
    return expected


def rate_table(table: dict, p_by_pos: dict) -> dict:
    """Emitted tokens per ms at each fixed depth, under the measured curve."""
    out = {}
    for d in sorted(table):
        e = expected_accepted(p_by_pos, d)
        t = table[d]["mean_ms"]
        out[d] = {"expected_accepted": e, "tokens": 1.0 + e, "T_ms": t,
                  "tokens_per_ms": (1.0 + e) / t,
                  "ms_per_token": t / (1.0 + e)}
    best = max(out, key=lambda d: out[d]["tokens_per_ms"])
    greedy = 0
    for d in sorted(table):
        if d + 1 not in table:
            break
        reach = 1.0
        for i in range(d + 1):
            reach *= p_by_pos.get(i, {"p": 0.0})["p"]
        c = (table[d + 1]["mean_ms"] - table[d]["mean_ms"]) / table[d]["mean_ms"]
        if reach > c * (1.0 + expected_accepted(p_by_pos, d)):
            greedy = d + 1
        else:
            break
    return {"by_depth": out, "global_argmax_depth": best,
            "greedy_first_local_max_depth": greedy,
            "greedy_leaves_on_table_pct":
                100.0 * (out[best]["tokens_per_ms"] / out[greedy]["tokens_per_ms"] - 1.0)}


# --------------------------------------------------------------------------
# policy replay
# --------------------------------------------------------------------------
def replay(rounds: list[dict], choose, table: dict) -> dict:
    """Score a depth policy on the forced tape.

    `choose(r)` returns the depth the policy would take. Only depths at or
    below the depth the round actually proposed are evaluable, because only
    those rows carry a target verdict; anything deeper is counted and the round
    is charged at its forced depth instead of being credited.
    """
    tokens = 0.0
    ms = 0.0
    rows = 0
    unevaluable = 0
    depths = []
    for r in rounds:
        d = choose(r)
        if d > r["d"]:
            unevaluable += 1
            d = r["d"]
        depths.append(d)
        tokens += 1 + min(r["acc"], d)
        rows += d
        ms += table[d]["mean_ms"] if d in table else float("nan")
    return {"rounds": len(rounds), "emitted_tokens": tokens,
            "draft_rows": rows, "modelled_ms": ms,
            "ms_per_token": ms / tokens,
            "mean_depth": statistics.mean(depths),
            "depth_histogram": dict(sorted(Counter(depths).items())),
            "rounds_not_evaluable": unevaluable}


def global_argmax_policy(table: dict, p_by_pos: dict, cap: int):
    """Arm G: pick the depth that maximises (1 + expected(D)) / T(D)."""
    best, best_rate = 0, -1.0
    for d in sorted(table):
        if d > cap:
            break
        rate = (1.0 + expected_accepted(p_by_pos, d)) / table[d]["mean_ms"]
        if rate > best_rate:
            best, best_rate = d, rate
    return lambda r: best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="FORCE")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    prompts = tuple(p for p in args.prompts.split(",") if p)
    rounds, legs = load_force_tape(args.arm, prompts)

    core = time_table(rounds, core_us)
    raw = time_table(rounds, lambda r: r["round_us"])
    p_by_pos = accept_table(rounds)
    rates = rate_table(core["round_ms"], p_by_pos)
    cap = max(core["round_ms"])

    report = {
        "arm": args.arm,
        "prompts": list(prompts),
        "fidelity": fidelity(legs),
        "instrument": instrument_ok(rounds),
        "time_core": core,
        "time_round_us": raw,
        "upkeep_bias_ms": {
            d: raw["round_ms"][d]["mean_ms"] - core["round_ms"][d]["mean_ms"]
            for d in core["round_ms"]},
        "admissibility_core": admissibility(core["step_ratio"]),
        "admissibility_round_us": admissibility(raw["step_ratio"]),
        "position_control": position_control(rounds, core_us),
        "pass_count_model": pass_model(core["round_ms"]),
        "acceptance": p_by_pos,
        "rate": rates,
        "replay": {
            "shipped_counterfactual": replay(
                rounds, lambda r: r.get("shipped_depth", r["d"]),
                core["round_ms"]),
            "forced_as_run": replay(rounds, lambda r: r["d"],
                                    core["round_ms"]),
            "arm_g_global_argmax": replay(
                rounds, global_argmax_policy(core["round_ms"], p_by_pos, cap),
                core["round_ms"]),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
