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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=122)
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
        result["pooled"] = {"rounds": len(pooled), "positions": {}}
        for position in pooled_raw:
            lo, hi = bootstrap_auc(pooled_raw[position]["pairs"], args.boot, rng)
            zlo, zhi = bootstrap_auc(pooled_z[position]["pairs"], args.boot, rng)
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
                "bins": margin_bins(pooled_raw[position]["pairs"]),
            }

    text = []
    text.append("E122 rung 0 -- target top-2 margin against draft acceptance")
    text.append("harness=local  timing_valid=false  official_or_ranked_score=false")
    for prompt, block in per_prompt.items():
        text.append("")
        text.append(f"## {prompt}  ({block['rounds']} rounds, "
                    f"mean depth {block['mean_depth']:.4f}, "
                    f"accepted/drafted {block['accept_rate_of_drafted']:.4f})")
        m = block["margin"]
        text.append(f"   margin mean {m['mean']:.4f} sd {m['sd']:.4f} "
                    f"p10 {m['p10']:.4f} p50 {m['p50']:.4f} p90 {m['p90']:.4f}")
        text.append(f"   depth histogram {block['depth_hist']}")
        text.append("   pos   n   acc_rate      AUC   [95% CI]        not_drafted")
        for position in range(1, 6):
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
        text.append("## pooled over prompts")
        text.append("   pos    n   acc_rate   AUC(raw)   [95% CI]        AUC(within-prompt z)")
        for position in range(1, 6):
            cell = result["pooled"]["positions"][position]
            a = cell["auc_raw_margin"]
            ci = cell["auc_raw_ci"]
            z = cell["auc_within_prompt_z"]
            text.append(
                f"   {position:>3} {cell['observed']:>5} "
                f"{(cell['accept_rate'] or float('nan')):>9.4f} "
                f"{(a if a is not None else float('nan')):>9.4f} "
                f"[{(ci[0] if ci[0] is not None else float('nan')):.4f}, "
                f"{(ci[1] if ci[1] is not None else float('nan')):.4f}] "
                f"{(z if z is not None else float('nan')):>12.4f}")
    report = "\n".join(text)
    print(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        args.json.with_suffix(".txt").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
