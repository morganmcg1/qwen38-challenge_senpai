#!/usr/bin/env python3
"""E122 rung 0 -- does the target's own top-2 margin predict draft acceptance?

Reads the per-round phase traces written by research/e122_rung0_session.sh and
answers three questions per draft position:

  1. AUC of the margin against CONDITIONAL acceptance at that position.
  2. How many rounds are behind that AUC, and how many rounds never reached the
     position at all (the policy's own selection).
  3. Whether one margin threshold can serve every prompt, or whether the
     threshold has to be scale-free.

The margin is `m=` in the trace line. `snapshotScheduleSignal` records it
BEFORE the round proposes anything, so it is the target top-1 minus top-2 of
the pending primary -- the row the previous round accepted last. Nothing here
uses information the scored policy could not read at the same instant.

Position i is 1-indexed over draft slots. It is OBSERVED when the round drafted
at least i tokens and accepted at least i-1 of them; it is ACCEPTED when the
round accepted at least i.

  usage: research/e122_auc.py RUN_DIR [RUN_DIR ...] [--json OUT] [--boot N]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) ")
FIELD_RE = re.compile(r"\bm=([-\d.naif]+) streak=(\d+) cap=(\d+) ema=([\d.,]+)")
MAX_POSITION = 8


def read_meta(run_dir: Path) -> dict:
    meta = {}
    path = run_dir / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    return meta


def read_rounds(run_dir: Path) -> list[dict]:
    rounds = []
    path = run_dir / "trace.txt"
    for line in path.read_text(errors="replace").splitlines():
        head = ROUND_RE.match(line)
        if not head:
            continue
        fields = FIELD_RE.search(line)
        if not fields:
            # A leg whose schedule never ran the trace snapshot (depth-0
            # control) carries no margin and cannot enter the fit.
            continue
        margin = float(fields.group(1))
        rounds.append({
            "round": int(head.group(1)),
            "depth": int(head.group(2)),
            "accepted": int(head.group(3)),
            "margin": margin,
            "streak": int(fields.group(2)),
            "cap": int(fields.group(3)),
        })
    return rounds


def auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """Mann-Whitney AUC with half credit for ties."""
    if not scores_pos or not scores_neg:
        return None
    merged = sorted([(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg])
    ranks = {}
    i = 0
    rank_sum_pos = 0.0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if merged[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def stratified_auc(groups: list[list[tuple[float, int]]]) -> tuple[float | None, int, int]:
    """AUC over prompt strata: discordant pairs are counted WITHIN a prompt.

    Pooling the raw margin across prompts lets a between-prompt difference in
    margin scale masquerade as discrimination. The shipped policy decides
    inside one request, so the within-prompt comparison is the one that
    matches the decision it would make.
    """
    u_total = 0.0
    pair_total = 0
    n_pos_total = 0
    n_neg_total = 0
    for pairs in groups:
        pos = [m for m, y in pairs if y == 1]
        neg = [m for m, y in pairs if y == 0]
        if not pos or not neg:
            continue
        value = auc(pos, neg)
        if value is None:
            continue
        u_total += value * len(pos) * len(neg)
        pair_total += len(pos) * len(neg)
        n_pos_total += len(pos)
        n_neg_total += len(neg)
    if pair_total == 0:
        return None, n_pos_total, n_neg_total
    return u_total / pair_total, n_pos_total, n_neg_total


def bootstrap_stratified(groups: list[list[tuple[float, int]]], reps: int,
                         rng: random.Random):
    """Cluster bootstrap: resample rounds inside each prompt, keep the strata."""
    if reps <= 0 or not groups:
        return None, None
    values = []
    for _ in range(reps):
        resampled = []
        for pairs in groups:
            if not pairs:
                continue
            n = len(pairs)
            resampled.append([pairs[rng.randrange(n)] for _ in range(n)])
        value, _, _ = stratified_auc(resampled)
        if value is not None:
            values.append(value)
    if len(values) < reps // 4:
        return None, None
    values.sort()
    return (values[int(0.025 * len(values))],
            values[min(len(values) - 1, int(0.975 * len(values)))])


def bootstrap_auc(pairs: list[tuple[float, int]], reps: int, rng: random.Random):
    """Percentile CI, resampling ROUNDS (the independent unit of the trace)."""
    if not pairs or reps <= 0:
        return None, None
    values = []
    n = len(pairs)
    for _ in range(reps):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        pos = [m for m, y in sample if y == 1]
        neg = [m for m, y in sample if y == 0]
        value = auc(pos, neg)
        if value is not None:
            values.append(value)
    if len(values) < reps // 4:
        return None, None
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return lo, hi


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation of the margin against the round's accepted count.

    Uses every round instead of only the rare rejections, so it is far less
    noisy than a per-position AUC on this data. It is only interpretable on a
    FORCED-DEPTH arm: under the shipped policy the accepted count is capped by
    a depth the margin already chose, which manufactures the correlation.
    """
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def spearman_pooled(rounds: list[dict], prompts: list[str]) -> float | None:
    """Sample-weighted mean of the within-prompt rank correlations."""
    total, weight = 0.0, 0
    for prompt in prompts:
        subset = [r for r in rounds if r["prompt"] == prompt]
        value = spearman([r["margin"] for r in subset],
                         [float(r["accepted"]) for r in subset])
        if value is None:
            continue
        total += value * len(subset)
        weight += len(subset)
    return total / weight if weight else None


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = q * (len(sorted_values) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[int(index)]
    return sorted_values[low] * (high - index) + sorted_values[high] * (index - low)


def position_table(rounds: list[dict], key: str = "margin") -> dict:
    table = {}
    for position in range(1, MAX_POSITION + 1):
        observed = [r for r in rounds
                    if r["depth"] >= position and r["accepted"] >= position - 1]
        not_reached = [r for r in rounds if r["depth"] < position]
        pairs = [(r[key], 1 if r["accepted"] >= position else 0) for r in observed]
        pos = [m for m, y in pairs if y == 1]
        neg = [m for m, y in pairs if y == 0]
        table[position] = {
            "observed": len(observed),
            "accepted": len(pos),
            "rejected": len(neg),
            "not_drafted": len(not_reached),
            "accept_rate": (len(pos) / len(observed)) if observed else None,
            "auc": auc(pos, neg),
            "pairs": pairs,
            "margin_mean_accept": statistics.fmean(pos) if pos else None,
            "margin_mean_reject": statistics.fmean(neg) if neg else None,
        }
    return table


def concordance_counts(pairs: list[tuple[float, int]]) -> tuple[int, int, int]:
    """Concordant, discordant and tied pair counts for margin against accept.

    A pair takes one accepted and one rejected draft. It is concordant when the
    accepted draft carried the larger margin.
    """
    accepted = sorted(m for m, y in pairs if y == 1)
    rejected = [m for m, y in pairs if y == 0]
    concordant = discordant = tied = 0
    for margin in rejected:
        low = bisect.bisect_left(accepted, margin)
        high = bisect.bisect_right(accepted, margin)
        concordant += len(accepted) - high
        discordant += low
        tied += high - low
    return concordant, discordant, tied


def pooled_concordance(cells: list[list[tuple[float, int]]]) -> dict:
    """THE PRIMARY GATE STATISTIC.

    Counts are aggregated across every `(prompt, position)` cell BEFORE the
    ratio is formed, so a cell contributes in proportion to the pairs it
    actually carries. Averaging per-cell ratios instead would give a cell with
    three rejections the same weight as one with three hundred.

    A pair never crosses a cell boundary, so neither a between-prompt
    difference in margin scale nor a between-position difference in acceptance
    rate can enter the statistic.
    """
    total_c = total_d = total_t = 0
    informative = 0
    for pairs in cells:
        c, d, t = concordance_counts(pairs)
        if c + d + t == 0:
            continue
        informative += 1
        total_c += c
        total_d += d
        total_t += t
    total = total_c + total_d + total_t
    return {
        "concordant": total_c,
        "discordant": total_d,
        "tied": total_t,
        "pairs": total,
        "cells": informative,
        "auc": ((total_c + 0.5 * total_t) / total) if total else None,
        "somers_d": ((total_c - total_d) / total) if total else None,
    }


def gate_cells(rounds_by_prompt: dict, positions: range) -> list[list[tuple[float, int]]]:
    cells = []
    for rounds in rounds_by_prompt.values():
        table = position_table(rounds)
        for position in positions:
            if position in table:
                cells.append(table[position]["pairs"])
    return cells


def bootstrap_pooled_concordance(rounds_by_prompt: dict, positions: range,
                                 reps: int, rng: random.Random):
    """Cluster bootstrap over ROUNDS within prompt.

    One round supplies a correlated observation at several positions, so the
    round is the resampling unit. Resampling the position observations
    independently would understate the interval.
    """
    if reps <= 0:
        return None, None
    values = []
    for _ in range(reps):
        resampled = {}
        for prompt, rounds in rounds_by_prompt.items():
            n = len(rounds)
            if not n:
                continue
            resampled[prompt] = [rounds[rng.randrange(n)] for _ in range(n)]
        block = pooled_concordance(gate_cells(resampled, positions))
        if block["auc"] is not None:
            values.append(block["auc"])
    if len(values) < reps // 4:
        return None, None
    values.sort()
    return (values[int(0.025 * len(values))],
            values[min(len(values) - 1, int(0.975 * len(values)))])


def somers_d_runlength(rounds: list[dict]) -> float | None:
    """Somers' D of the accept run length given the margin, within one prompt.

    This is the quantity that is algebraically comparable with `2 * AUC - 1`.
    Pairs tied on the margin are excluded; pairs tied on the run length sit in
    the denominator, which is what makes it a `D(Y|X)` rather than a gamma.
    """
    data = [(r["margin"], r["accepted"]) for r in rounds]
    concordant = discordant = tied_y = 0
    for i in range(len(data)):
        xi, yi = data[i]
        for j in range(i + 1, len(data)):
            xj, yj = data[j]
            if xi == xj:
                continue
            if yi == yj:
                tied_y += 1
            elif (xi < xj) == (yi < yj):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant + tied_y
    return ((concordant - discordant) / total) if total else None


def weighted_mean(items: list[tuple[float | None, int]]) -> float | None:
    total, weight = 0.0, 0
    for value, count in items:
        if value is None:
            continue
        total += value * count
        weight += count
    return total / weight if weight else None


def verdict_for(value: float | None) -> str:
    """The pre-registered thresholds, applied without discretion."""
    if value is None:
        return "no data"
    if value <= 0.55:
        return "kill: at or below 0.55"
    if value >= 0.65:
        return "licensed: at or above 0.65"
    return "ask the advisor: between 0.55 and 0.65"


def margin_bins(pairs: list[tuple[float, int]], bins: int = 5) -> list[dict]:
    if len(pairs) < bins * 2:
        return []
    ordered = sorted(pairs, key=lambda p: p[0])
    size = len(ordered) / bins
    out = []
    for b in range(bins):
        lo = int(round(b * size))
        hi = int(round((b + 1) * size))
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        out.append({
            "bin": b + 1,
            "n": len(chunk),
            "margin_lo": chunk[0][0],
            "margin_hi": chunk[-1][0],
            "accept_rate": statistics.fmean([y for _, y in chunk]),
        })
    return out


def quantisation_summary(margins: list[float], below: float = 2.0) -> dict:
    """How many distinct values the margin actually takes, and the coarsest
    power-of-two grid that reproduces every observed value exactly.

    The trace prints six decimals, so a coarse grid here cannot be a printing
    artifact: it is the resolution the decision variable arrives with.
    """
    distinct = sorted(set(margins))
    small = [m for m in distinct if m < below]
    gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
    grid = None
    for k in range(0, 13):
        step = 2.0 ** (-k)
        if all(abs(m / step - round(m / step)) < 1e-6 for m in distinct):
            grid = step
            break
    return {
        "n": len(margins),
        "distinct": len(distinct),
        "distinct_below": len(small),
        "n_below": sum(1 for m in margins if m < below),
        "below": below,
        "min_gap": min(gaps) if gaps else None,
        "grid": grid,
        "smallest_values": distinct[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--gate-low", type=int, default=2)
    parser.add_argument("--gate-high", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    per_prompt = {}
    pooled: list[dict] = []
    for run_dir in args.run_dirs:
        meta = read_meta(run_dir)
        prompt = meta.get("prompt_id", run_dir.name)
        rounds = read_rounds(run_dir)
        if not rounds:
            print(f"e122_auc: {run_dir} has no traced rounds with a margin",
                  file=sys.stderr)
            continue
        margins = sorted(r["margin"] for r in rounds)
        depth_hist = {}
        for r in rounds:
            depth_hist[r["depth"]] = depth_hist.get(r["depth"], 0) + 1
        # Within-prompt standardisation, the scale-free candidate: a threshold
        # expressed in units of the prompt's own margin spread instead of
        # nats. The policy cannot compute this offline value, so it is a
        # DIAGNOSTIC of how much of the signal is scale, not a shippable map.
        mean = statistics.fmean(margins)
        sd = statistics.pstdev(margins) or 1.0
        for r in rounds:
            r["prompt"] = prompt
            r["margin_z"] = (r["margin"] - mean) / sd
        per_prompt[prompt] = {
            "run_dir": str(run_dir),
            "rounds": len(rounds),
            "meta": {k: meta.get(k) for k in (
                "base_sha", "worker_sha256", "golden_sha256", "head_dir",
                "tokens", "offered_depth", "prompt_file", "prompt_sha256",
                "gpu_temp_entry_c", "gpu_temp_exit_c", "trace_rounds",
                "leg_kind", "forced_depth")},
            "depth_hist": dict(sorted(depth_hist.items())),
            "mean_depth": statistics.fmean([r["depth"] for r in rounds]),
            "mean_accepted": statistics.fmean([r["accepted"] for r in rounds]),
            "spearman_margin_accepted": spearman(
                [r["margin"] for r in rounds],
                [float(r["accepted"]) for r in rounds]),
            "accept_rate_of_drafted": (
                sum(r["accepted"] for r in rounds)
                / max(1, sum(r["depth"] for r in rounds))),
            "margin": {
                "mean": mean,
                "sd": sd,
                "min": margins[0],
                "p10": quantile(margins, 0.10),
                "p25": quantile(margins, 0.25),
                "p50": quantile(margins, 0.50),
                "p75": quantile(margins, 0.75),
                "p90": quantile(margins, 0.90),
                "max": margins[-1],
            },
            "margin_quantisation": quantisation_summary(margins),
            "positions": {},
        }
        table = position_table(rounds)
        for position, cell in table.items():
            lo, hi = bootstrap_auc(cell["pairs"], args.boot, rng)
            per_prompt[prompt]["positions"][position] = {
                k: v for k, v in cell.items() if k != "pairs"}
            per_prompt[prompt]["positions"][position]["auc_ci"] = [lo, hi]
            per_prompt[prompt]["positions"][position]["bins"] = margin_bins(
                cell["pairs"])
        pooled.extend(rounds)

    result = {
        "experiment": "e122-target-margin-conditioned-draft-depth",
        "rung": 0,
        "harness": "local",
        "timing_valid": False,
        "official_or_ranked_score": False,
        "prompts": per_prompt,
    }

    if pooled:
        pooled_raw = position_table(pooled, key="margin")
        pooled_z = position_table(pooled, key="margin_z")
        by_prompt_tables = {
            prompt: position_table([r for r in pooled if r["prompt"] == prompt])
            for prompt in per_prompt}
        rounds_by_prompt = {
            prompt: [r for r in pooled if r["prompt"] == prompt]
            for prompt in per_prompt}
        positions = range(args.gate_low, args.gate_high + 1)
        gate = pooled_concordance(gate_cells(rounds_by_prompt, positions))
        gate_lo, gate_hi = bootstrap_pooled_concordance(
            rounds_by_prompt, positions, args.boot, rng)
        gate["ci"] = [gate_lo, gate_hi]
        gate["positions"] = [positions.start, positions.stop - 1]
        gate["somers_d_runlength_within_prompt"] = weighted_mean(
            [(somers_d_runlength(rs), len(rs))
             for rs in rounds_by_prompt.values()])
        gate["verdict"] = verdict_for(gate["auc"])
        gate["per_prompt"] = {}
        for prompt, rounds in rounds_by_prompt.items():
            block = pooled_concordance(gate_cells({prompt: rounds}, positions))
            block["somers_d_runlength"] = somers_d_runlength(rounds)
            block["spearman_margin_accepted"] = (
                per_prompt[prompt]["spearman_margin_accepted"])
            block["verdict"] = verdict_for(block["auc"])
            gate["per_prompt"][prompt] = block
        result["gate"] = gate

        result["pooled"] = {
            "rounds": len(pooled),
            "spearman_margin_accepted": spearman_pooled(
                pooled, list(per_prompt)),
            "positions": {},
        }
        for position in pooled_raw:
            lo, hi = bootstrap_auc(pooled_raw[position]["pairs"], args.boot, rng)
            zlo, zhi = bootstrap_auc(pooled_z[position]["pairs"], args.boot, rng)
            groups = [table[position]["pairs"] for table in by_prompt_tables.values()
                      if position in table]
            strat, n_pos, n_neg = stratified_auc(groups)
            slo, shi = bootstrap_stratified(groups, args.boot, rng)
            informative = sum(
                1 for pairs in groups
                if any(y == 1 for _, y in pairs) and any(y == 0 for _, y in pairs))
            result["pooled"]["positions"][position] = {
                "observed": pooled_raw[position]["observed"],
                "accepted": pooled_raw[position]["accepted"],
                "rejected": pooled_raw[position]["rejected"],
                "not_drafted": pooled_raw[position]["not_drafted"],
                "accept_rate": pooled_raw[position]["accept_rate"],
                "auc_raw_margin": pooled_raw[position]["auc"],
                "auc_raw_ci": [lo, hi],
                "auc_within_prompt_z": pooled_z[position]["auc"],
                "auc_z_ci": [zlo, zhi],
                "auc_stratified": strat,
                "auc_stratified_ci": [slo, shi],
                "stratified_accepted": n_pos,
                "stratified_rejected": n_neg,
                "informative_prompts": informative,
                "bins": margin_bins(pooled_raw[position]["pairs"]),
            }

    text = []
    text.append("E122 rung 0 -- target top-2 margin against draft acceptance")
    text.append("harness=local  timing_valid=false  official_or_ranked_score=false")
    if "gate" in result:
        g = result["gate"]
        lo, hi = g["ci"]
        text.append("")
        text.append(f"## PRIMARY GATE -- prompt-stratified concordance, "
                    f"positions {g['positions'][0]} to {g['positions'][1]}")
        text.append(f"   pooled AUC {g['auc']:.4f} "
                    f"[{(lo if lo is not None else float('nan')):.4f}, "
                    f"{(hi if hi is not None else float('nan')):.4f}]   "
                    f"pairs {g['pairs']}  cells {g['cells']}")
        text.append(f"   concordant {g['concordant']}  discordant {g['discordant']}"
                    f"  tied {g['tied']}")
        sd_gate = g["somers_d"]
        sd_run = g["somers_d_runlength_within_prompt"]
        text.append(f"   Somers D from the gate pairs      {sd_gate:+.4f}"
                    f"   (= 2 x AUC - 1)")
        text.append(f"   Somers D of margin vs run length  "
                    f"{(sd_run if sd_run is not None else float('nan')):+.4f}"
                    f"   (the agreement check)")
        if sd_gate is not None and sd_run is not None:
            delta = abs(sd_gate - sd_run)
            text.append(f"   agreement |delta D| {delta:.4f}  "
                        f"{'OK' if delta <= 0.10 else 'DISAGREE, report it'}")
        text.append(f"   VERDICT  {g['verdict']}")
        text.append("")
        text.append("   per prompt (natural_history can overrule the pool)")
        text.append(f"   {'prompt':<18}{'pairs':>7}{'AUC':>9}{'D_gate':>9}"
                    f"{'D_runlen':>10}{'spearman':>10}  verdict")
        for prompt in sorted(g["per_prompt"]):
            b = g["per_prompt"][prompt]
            text.append(
                f"   {prompt:<18}{b['pairs']:>7}"
                f"{(b['auc'] if b['auc'] is not None else float('nan')):>9.4f}"
                f"{(b['somers_d'] if b['somers_d'] is not None else float('nan')):>+9.4f}"
                f"{(b['somers_d_runlength'] if b['somers_d_runlength'] is not None else float('nan')):>+10.4f}"
                f"{(b['spearman_margin_accepted'] if b['spearman_margin_accepted'] is not None else float('nan')):>+10.4f}"
                f"  {b['verdict']}")
    for prompt, block in per_prompt.items():
        text.append("")
        text.append(f"## {prompt}  ({block['rounds']} rounds, "
                    f"mean depth {block['mean_depth']:.4f}, "
                    f"accepted/drafted {block['accept_rate_of_drafted']:.4f})")
        m = block["margin"]
        text.append(f"   margin mean {m['mean']:.4f} sd {m['sd']:.4f} "
                    f"p10 {m['p10']:.4f} p50 {m['p50']:.4f} p90 {m['p90']:.4f}")
        q = block["margin_quantisation"]
        text.append(
            f"   margin resolution: {q['distinct']} distinct in {q['n']} rounds, "
            f"{q['distinct_below']} distinct in the {q['n_below']} rounds "
            f"below {q['below']:.1f}, min gap "
            f"{(q['min_gap'] if q['min_gap'] is not None else float('nan')):.6f}, "
            f"exact grid {q['grid']}")
        text.append("   smallest margin values "
                    + ", ".join(f"{v:g}" for v in q["smallest_values"]))
        text.append(f"   depth histogram {block['depth_hist']}")
        rho = block["spearman_margin_accepted"]
        text.append(f"   spearman(margin, accepted) "
                    f"{(rho if rho is not None else float('nan')):.4f}  "
                    f"forced_depth={block['meta'].get('forced_depth')}")
        text.append("   pos   n   acc_rate      AUC   [95% CI]        not_drafted")
        for position in range(1, MAX_POSITION + 1):
            cell = block["positions"][position]
            a = cell["auc"]
            ci = cell["auc_ci"]
            text.append(
                f"   {position:>3} {cell['observed']:>4} "
                f"{(cell['accept_rate'] if cell['accept_rate'] is not None else float('nan')):>9.4f} "
                f"{(a if a is not None else float('nan')):>8.4f} "
                f"[{(ci[0] if ci[0] is not None else float('nan')):.4f}, "
                f"{(ci[1] if ci[1] is not None else float('nan')):.4f}] "
                f"{cell['not_drafted']:>8}")
    if pooled:
        text.append("")
        text.append(f"## pooled over prompts  ({result['pooled']['rounds']} rounds)")
        prho = result["pooled"]["spearman_margin_accepted"]
        text.append("   within-prompt spearman(margin, accepted) "
                    f"{(prho if prho is not None else float('nan')):.4f}")
        text.append("   AUC(raw)   one nat-valued threshold across every prompt")
        text.append("   AUC(strat) discordant pairs counted only inside one prompt")
        text.append("   AUC(z)     within-prompt standardised margin, diagnostic only")
        text.append("   pos    n  rej  acc_rate   AUC(raw)   [95% CI]         "
                    "AUC(strat)   [95% CI]         AUC(z)  prompts")
        for position in range(1, MAX_POSITION + 1):
            cell = result["pooled"]["positions"][position]
            a = cell["auc_raw_margin"]
            ci = cell["auc_raw_ci"]
            s = cell["auc_stratified"]
            sci = cell["auc_stratified_ci"]
            z = cell["auc_within_prompt_z"]
            text.append(
                f"   {position:>3} {cell['observed']:>5} {cell['rejected']:>4} "
                f"{(cell['accept_rate'] or float('nan')):>9.4f} "
                f"{(a if a is not None else float('nan')):>9.4f} "
                f"[{(ci[0] if ci[0] is not None else float('nan')):.4f}, "
                f"{(ci[1] if ci[1] is not None else float('nan')):.4f}] "
                f"{(s if s is not None else float('nan')):>11.4f} "
                f"[{(sci[0] if sci[0] is not None else float('nan')):.4f}, "
                f"{(sci[1] if sci[1] is not None else float('nan')):.4f}] "
                f"{(z if z is not None else float('nan')):>8.4f} "
                f"{cell['informative_prompts']:>6}")
    report = "\n".join(text)
    print(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        args.json.with_suffix(".txt").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
