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

WHY THE PARENT'S CLOCK IS THE PRIMARY TIMER.  The trusted parent already
publishes `block_request_seconds` and `effective_draft_lengths`, one entry per
round, both computed from its own journal.  Pairing them measures T(d) on the
clock that actually produces the score, needs nothing from the editable side,
and -- unlike the trace, which only emits a line when a draft was proposed --
covers d = 0.  T(0) was simply unmeasurable in r1.  The trace's `round_us` and
`round_us - upkeep_us` are reported beside it as agreement checks; `upkeep_us`
spans the trace's own per-row top-2 dump (`Self.traceRow` over
0 ... acceptedCount) so it scales with accepted rows and therefore with depth.

WARM-UP.  A depth's first dispatch pays kernel specialisation once: on english
the first d = 5 round cost 339.9 ms against a 143.3 ms median.  The first full
forced cycle (8 rounds) is dropped per prompt so that every depth's first-touch
round is excluded by the same rule, and medians are reported beside means.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
WARMUP_ROUNDS = 8
# Near the end of the fixed 512-token window the parent offers fewer draft rows
# than the width cap, so the last 8 rounds are offer-bound rather than
# forced-depth samples. Dropping them keeps this tape identical to the one
# `e25r2_policy.py` replays.
TAIL_ROUNDS = 8


def load_force_tape(arm: str = "FORCE", prompts=PROMPTS,
                    runs_root: Path = RUNS_ROOT):
    """Pooled forced-depth rounds plus each prompt's two leg reports.

    One record per round of the MTP leg, keyed on the parent's own journal:
    `d` and `parent_ms` come from `effective_draft_lengths` and
    `block_request_seconds`, so a d = 0 round exists even though the trace
    emits no line for it.  Trace fields are merged in where they exist.
    """
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
        serial = json.loads((run / "reports/03-mtp-timed.json").read_text())
        mtp = json.loads((run / "reports/04-mtp-timed.json").read_text())
        if not serial.get("is_serial_control") or mtp.get("is_serial_control"):
            raise SystemExit(f"{run}: leg reports are not (serial, mtp)")
        legs[prompt] = {"serial": serial, "mtp": mtp,
                        "trace_rounds": sessions[0]}

        depths = mtp["effective_draft_lengths"]
        secs = mtp["block_request_seconds"]
        if not len(depths) == len(secs) == mtp["round_count"]:
            raise SystemExit(f"{run}: parent per-round arrays disagree")
        by_index = {r["round"] - 1: r for r in sessions[0]}
        for i, (d, s) in enumerate(zip(depths, secs)):
            traced = by_index.get(i)
            if traced is not None and traced["d"] != d:
                raise SystemExit(
                    f"{run}: round {i} trace depth {traced['d']} != parent {d}")
            if traced is None and d != 0:
                raise SystemExit(f"{run}: round {i} depth {d} is untraced")
            rec = dict(traced) if traced else {"round": i + 1, "d": d,
                                               "acc": 0}
            rec.update(prompt=prompt, index=i, parent_ms=s * 1000.0,
                       warm=WARMUP_ROUNDS <= i < len(depths) - TAIL_ROUNDS,
                       traced=traced is not None)
            rounds.append(rec)
    if missing:
        raise SystemExit("e25r2_refit: no tape at " + ", ".join(missing))
    return rounds, legs


def warm(rounds: list[dict]) -> list[dict]:
    return [r for r in rounds if r["warm"]]


def parent_ms(r: dict) -> float:
    """Primary timer: the trusted parent's own per-round wall clock, in us."""
    return r["parent_ms"] * 1000.0


def ledger_crosscheck(legs: dict, rounds: list[dict]) -> dict:
    """The editable trace against the parent's independent row accounting."""
    out = {}
    for prompt, pair in legs.items():
        mtp = pair["mtp"]
        traced = [r for r in rounds if r["prompt"] == prompt and r["traced"]]
        out[prompt] = {
            "parent_accepted_draft_total": mtp["accepted_draft_total"],
            "trace_accepted_sum": sum(r["acc"] for r in traced),
            "parent_drafted_rows":
                mtp["accepted_draft_total"] + mtp["rejected_draft_total"],
            "trace_proposed_sum": sum(r["d"] for r in traced),
            "parent_declared_rows_total": mtp["declared_rows_total"],
            "parent_reference_checked_row_total":
                mtp["reference_checked_row_total"],
            "primary_plus_draft_rows":
                mtp["round_count"] + mtp["accepted_draft_total"]
                + mtp["rejected_draft_total"],
            "max_rejected_tail_logit_delta":
                mtp["max_rejected_tail_logit_delta"],
            "verify_block_replayed_round_count":
                mtp["verify_block_replayed_round_count"],
        }
        c = out[prompt]
        c["accepted_agrees"] = (
            c["parent_accepted_draft_total"] == c["trace_accepted_sum"])
        c["proposed_agrees"] = (
            c["parent_drafted_rows"] == c["trace_proposed_sum"])
        c["rows_closed"] = (
            c["parent_declared_rows_total"]
            == c["parent_reference_checked_row_total"]
            == c["primary_plus_draft_rows"])
    return out


def fidelity(legs: dict) -> dict:
    """Every gate the trusted parent reports, per prompt, plus the head used.

    `uses_pinned_mtp_head` is *not* "the organizer head": main.swift :1962 sets
    it from `report.usesNativeMTPHead`, i.e. whether the leg drafted at all.
    The head identity lives in `head_provenance`.
    """
    out = {}
    for prompt, pair in legs.items():
        mtp, serial = pair["mtp"], pair["serial"]
        prov = mtp.get("head_provenance") or {}
        out[prompt] = {
            k: mtp.get(k) for k in (
                "all_tokens_matched", "residual_divergence_count",
                "parity_all_ok", "uses_pinned_mtp_head",
                "decode_token_count", "emitted_token_total", "round_count")
        }
        out[prompt].update(
            head_origin=prov.get("origin"), head_source=prov.get("source"),
            head_bytes=prov.get("bytes"), head_sha256=prov.get("sha256"),
            head_file_count=prov.get("file_count"),
            serial_all_tokens_matched=serial.get("all_tokens_matched"),
            serial_uses_pinned_mtp_head=serial.get("uses_pinned_mtp_head"),
            serial_seconds_per_token=serial.get(
                "parent_measured_seconds_per_token"),
            mtp_seconds_per_token=mtp.get(
                "parent_measured_seconds_per_token"))
        sp = out[prompt]["serial_seconds_per_token"]
        mp = out[prompt]["mtp_seconds_per_token"]
        out[prompt]["round_robin_speedup"] = sp / mp if sp and mp else None
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
    step, step_median = {}, {}
    for d in sorted(table):
        if d + 1 in table:
            # The shipped rule's own definition: price of adding row d+1 as a
            # fraction of the round it extends (Qwen36MTPBlockSession :571).
            t0, t1 = table[d]["mean_ms"], table[d + 1]["mean_ms"]
            step[d] = (t1 - t0) / t0
            m0, m1 = table[d]["median_ms"], table[d + 1]["median_ms"]
            step_median[d] = (m1 - m0) / m0
    return {"round_ms": table, "step_ratio": step,
            "step_ratio_median": step_median}


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


def empirical_rate_table(rounds: list[dict], timer, boots: int = 4000,
                         seed: int = 20260818) -> dict:
    """Realised tokens per ms at each forced depth, straight off the tape.

    Needs no acceptance model: within a forced round-robin each depth sees a
    fair sample of positions, so `mean(1 + acc) / mean(T)` is an unbiased
    estimate of what a *constant* depth-d policy would deliver.  This table is
    therefore both the price-curve evidence and the constant-depth policy
    comparison, and it is the arm-G decision statistic in its weakest form.

    Uncertainty is dominated by the granted-token count, not by T, so the
    per-stratum sem and the bootstrap over rounds are what decide the argmax.
    """
    tok = defaultdict(list)
    ms = defaultdict(list)
    for r in rounds:
        tok[r["d"]].append(1.0 + r["acc"])
        ms[r["d"]].append(timer(r) / 1000.0)
    out = {}
    for d in sorted(tok):
        mt, mm = statistics.mean(tok[d]), statistics.mean(ms[d])
        med = statistics.median(ms[d])
        n = len(tok[d])
        s_tok = statistics.stdev(tok[d]) / math.sqrt(n) if n > 1 else 0.0
        s_ms = statistics.stdev(ms[d]) / math.sqrt(n) if n > 1 else 0.0
        rate = mt / mm
        out[d] = {"n": n, "mean_tokens": mt, "sem_tokens": s_tok,
                  "mean_T_ms": mm, "sem_T_ms": s_ms, "median_T_ms": med,
                  "tokens_per_ms": rate,
                  "sem_tokens_per_ms": rate * math.sqrt(
                      (s_tok / mt) ** 2 + (s_ms / mm) ** 2),
                  "tokens_per_ms_median_T": mt / med,
                  "ms_per_token": mm / mt}

    rng = random.Random(seed)
    depths = sorted(tok)
    wins = Counter()
    for _ in range(boots):
        best_d, best_r = None, -1.0
        for d in depths:
            n = len(tok[d])
            idx = [rng.randrange(n) for _ in range(n)]
            rt = sum(tok[d][i] for i in idx) / sum(ms[d][i] for i in idx)
            if rt > best_r:
                best_d, best_r = d, rt
        wins[best_d] += 1
    best = max(out, key=lambda d: out[d]["tokens_per_ms"])
    best_med = max(out, key=lambda d: out[d]["tokens_per_ms_median_T"])
    return {"by_depth": out, "argmax_depth": best,
            "argmax_depth_median_T": best_med,
            "bootstrap_draws": boots,
            "bootstrap_argmax_share": {
                d: wins[d] / boots for d in depths if wins[d]},
            "gain_over_depth3_pct": (
                100.0 * (out[best]["tokens_per_ms"]
                         / out[3]["tokens_per_ms"] - 1.0)
                if 3 in out else None),
            "rate_vs_argmax_pct": {
                d: 100.0 * (out[d]["tokens_per_ms"]
                            / out[best]["tokens_per_ms"] - 1.0)
                for d in depths}}


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


def render(r: dict) -> str:
    L = []
    add = L.append
    add(f"arm={r['arm']} prompts={','.join(r['prompts'])} "
        f"rounds {r['rounds_total']} -> {r['rounds_analysed']} "
        f"(dropped first {r['warmup_rounds_dropped_per_prompt']} + last "
        f"{r['tail_rounds_dropped_per_prompt']}/prompt)")

    add("\nFIDELITY + HEAD")
    for p, f in sorted(r["fidelity"].items()):
        add(f"  {p:<16} matched={f['all_tokens_matched']} "
            f"parity={f['parity_all_ok']} div={f['residual_divergence_count']} "
            f"tok={f['decode_token_count']}/{f['emitted_token_total']} "
            f"speedup={f['round_robin_speedup']:.4f}")
        add(f"                   head={f['head_origin']} "
            f"bytes={f['head_bytes']} files={f['head_file_count']}")
    add("\nPARENT LEDGER CROSSCHECK")
    for p, c in sorted(r["ledger_crosscheck"].items()):
        add(f"  {p:<16} accepted {c['trace_accepted_sum']}=="
            f"{c['parent_accepted_draft_total']} {c['accepted_agrees']}  "
            f"proposed {c['trace_proposed_sum']}=={c['parent_drafted_rows']} "
            f"{c['proposed_agrees']}  rows_closed={c['rows_closed']} "
            f"tail_delta={c['max_rejected_tail_logit_delta']} "
            f"replayed={c['verify_block_replayed_round_count']}")

    i = r["instrument"]
    add(f"\nINSTRUMENT  marked={i['rounds_carrying_forced_mark']} "
        f"taken==forced={i['taken_depth_equals_forced']}  "
        f"forced_hist={i['forced_depth_histogram']}")
    add(f"            taken_hist={i['taken_depth_histogram']}")
    add(f"            shipped_counterfactual={i['shipped_counterfactual_histogram']}")

    t = r["time_parent_clock"]["round_ms"]
    ag = r["timer_agreement_ms"]
    add("\nT(d) ON THE PARENT'S OWN CLOCK  (ms; trace timers as check)")
    add(f"  {'d':>2} {'n':>4} {'mean':>8} {'sem':>6} {'median':>8} {'pass':>4}"
        f" {'core-par':>9} {'raw-par':>8}")
    for d, v in sorted(t.items(), key=lambda kv: int(kv[0])):
        a = ag.get(d, {})
        c = f"{a['core_minus_parent']:>9.3f}" if a else f"{'-':>9}"
        w = f"{a['round_us_minus_parent']:>8.3f}" if a else f"{'-':>8}"
        add(f"  {d:>2} {v['n']:>4} {v['mean_ms']:>8.3f} {v['sem_ms']:>6.3f} "
            f"{v['median_ms']:>8.3f} {v['weight_passes']:>4} {c} {w}")

    for label, key in (("mean", "admissibility_parent"),
                       ("median", "admissibility_parent_median")):
        add(f"\nPRICE ADMISSIBILITY ({label} T; a price can fire only if "
            f"c_d < 1/(d+1))")
        for d, v in sorted(r[key].items(), key=lambda kv: int(kv[0])):
            flag = "ok " if v["admissible"] else "WALL"
            add(f"  c_{d} = {v['measured_c']:.6f}  ceiling {v['ceiling']:.6f}"
                f"  {flag}  shipped_scalar={v['shipped_scalar_c']:.5f}"
                + (f"  r1={v['r1_c']}" if v["r1_c"] is not None else ""))

    e = r["rate_empirical"]
    add("\nREALISED RATE PER FORCED DEPTH  == constant-depth policy comparison")
    add(f"  {'d':>2} {'n':>4} {'tok/round':>9} {'+-':>6} {'T_mean':>8}"
        f" {'tok/ms':>9} {'+-':>8} {'ms/tok':>7} {'vs best':>8}")
    for d, v in sorted(e["by_depth"].items(), key=lambda kv: int(kv[0])):
        add(f"  {d:>2} {v['n']:>4} {v['mean_tokens']:>9.4f} "
            f"{v['sem_tokens']:>6.4f} {v['mean_T_ms']:>8.3f} "
            f"{v['tokens_per_ms']:>9.6f} {v['sem_tokens_per_ms']:>8.6f} "
            f"{v['ms_per_token']:>7.3f} "
            f"{e['rate_vs_argmax_pct'][d]:>+8.2f}%")
    add(f"  argmax={e['argmax_depth']} argmax(medianT)="
        f"{e['argmax_depth_median_T']} "
        f"gain_over_depth3={e['gain_over_depth3_pct']:+.3f}%")
    add(f"  bootstrap argmax share ({e['bootstrap_draws']} draws): "
        f"{ {k: round(v, 3) for k, v in e['bootstrap_argmax_share'].items()} }")
    add(f"  per-prompt argmax: {r['per_prompt_argmax']}")

    m = r["rate_modelled"]
    add(f"\nMODELLED RATE  argmax={m['global_argmax_depth']} "
        f"greedy_first_local_max={m['greedy_first_local_max_depth']} "
        f"greedy_leaves_on_table={m['greedy_leaves_on_table_pct']:+.3f}%")

    add("\nCONDITIONAL ACCEPTANCE p_i (row i | rows < i accepted)")
    for i_, v in sorted(r["acceptance"].items(), key=lambda kv: int(kv[0])):
        add(f"  i={i_} reached={v['reached']:>4} accepted={v['accepted']:>4} "
            f"p={v['p']:.4f} +-{v['sem']:.4f}")

    pm = r["pass_count_model"]
    if pm.get("fitted"):
        add(f"\nPASS MODEL T = a + b*d + c*ceil((d+1)/IPG):  "
            f"a={pm['intercept_ms']:.3f} b={pm['per_row_ms']:.3f} "
            f"c={pm['per_weight_pass_ms']:.3f} R2={pm['r_squared']:.5f} "
            f"max|resid|={pm['max_abs_residual_ms']:.3f} ms")

    pc = r["position_control"]
    add(f"\nPOSITION CONTROL  OLS drift = "
        f"{pc['ols_ms_per_1000_tokens']:+.3f} ms per 1000 tokens of offset")
    add(f"  {'d':>2}" + "".join(f" {q:>10}" for q in sorted(pc["by_quartile"])))
    depths = sorted({int(d) for q in pc["by_quartile"].values() for d in q})
    for d in depths:
        cells = []
        for q in sorted(pc["by_quartile"]):
            v = pc["by_quartile"][q].get(str(d)) or pc["by_quartile"][q].get(d)
            cells.append(f" {v['mean_ms']:>10.3f}" if v else f" {'-':>10}")
        add(f"  {d:>2}" + "".join(cells))

    add("\nPOLICY REPLAY ON THE FORCED TAPE (modelled ms from the T table)")
    for k, v in r["replay"].items():
        add(f"  {k:<24} ms/token={v['ms_per_token']:.4f} "
            f"mean_depth={v['mean_depth']:.3f} tokens={v['emitted_tokens']:.0f} "
            f"unevaluable={v['rounds_not_evaluable']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="FORCE")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT,
                    help="tape root: <root>/probe-<prompt>-<arm>")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--text", action="store_true",
                    help="human-readable summary instead of JSON")
    args = ap.parse_args()

    prompts = tuple(p for p in args.prompts.split(",") if p)
    all_rounds, legs = load_force_tape(args.arm, prompts, args.runs_root)
    rounds = warm(all_rounds)

    # The parent's own per-round clock is the primary timer; the two trace
    # timers are independent checks that must reproduce it.
    parent = time_table(rounds, parent_ms)
    # The trace emits no line for a d = 0 round, so the trace timers can only
    # be compared where a line exists.
    traced = [r for r in rounds if r["traced"]]
    core = time_table(traced, core_us)
    raw = time_table(traced, lambda r: r["round_us"])
    table = parent["round_ms"]
    p_by_pos = accept_table(rounds)
    cap = max(table)

    report = {
        "arm": args.arm,
        "prompts": list(prompts),
        "warmup_rounds_dropped_per_prompt": WARMUP_ROUNDS,
        "tail_rounds_dropped_per_prompt": TAIL_ROUNDS,
        "rounds_total": len(all_rounds),
        "rounds_analysed": len(rounds),
        "fidelity": fidelity(legs),
        "ledger_crosscheck": ledger_crosscheck(legs, all_rounds),
        "instrument": instrument_ok(rounds),
        "time_parent_clock": parent,
        "time_core": core,
        "time_round_us": raw,
        "timer_agreement_ms": {
            d: {"core_minus_parent":
                    core["round_ms"][d]["mean_ms"] - v["mean_ms"],
                "round_us_minus_parent":
                    raw["round_ms"][d]["mean_ms"] - v["mean_ms"],
                "upkeep_bias": raw["round_ms"][d]["mean_ms"]
                               - core["round_ms"][d]["mean_ms"]}
            for d, v in table.items() if d in core["round_ms"]},
        "admissibility_parent": admissibility(parent["step_ratio"]),
        "admissibility_parent_median": admissibility(parent["step_ratio_median"]),
        "admissibility_core": admissibility(core["step_ratio"]),
        "position_control": position_control(rounds, parent_ms),
        "pass_count_model": pass_model(table),
        "acceptance": p_by_pos,
        "rate_empirical": empirical_rate_table(rounds, parent_ms),
        "rate_modelled": rate_table(table, p_by_pos),
        "replay": {
            "shipped_counterfactual": replay(
                rounds, lambda r: r.get("shipped_depth", r["d"]), table),
            "forced_as_run": replay(rounds, lambda r: r["d"], table),
            "arm_g_global_argmax": replay(
                rounds, global_argmax_policy(table, p_by_pos, cap), table),
        },
        "per_prompt_argmax": {
            p: empirical_rate_table(
                [r for r in rounds if r["prompt"] == p], parent_ms)["argmax_depth"]
            for p in prompts},
        # Pooling hides whether one prompt carries the curve, so every prompt
        # also gets its own T(d) and price vector.
        "per_prompt_curve": {
            p: time_table([r for r in rounds if r["prompt"] == p], parent_ms)
            for p in prompts},
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(render(report) if args.text else text)


if __name__ == "__main__":
    main()
