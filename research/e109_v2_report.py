#!/usr/bin/env python3
"""E109 rung 0 v2: resolve a dose from ONE leg by alternating it per round.

    usage: research/e109_v2_report.py LEG_DIR [LEG_DIR ...] [--json OUT]

Rung 0 v1 contrasted whole legs and failed its bar at a 833.5 us half-width.
The diagnosis was a per-leg offset with SD 697 us that carried 97.9 percent of
the pair variance and was not thermal: r squared against entry temperature was
0.004. A whole-leg contrast pays that offset in full, and no affordable number
of legs removes it -- reaching the bar needed 180 legs and 4.6 hours.

v2 moves the contrast inside the leg. `MLX_E105_DOSE_ALTERNATE=1` applies the
dose on every second decode round, so the estimate is a mean of differences
between NEIGHBOURING rounds of one leg. Whatever the per-leg offset is, both
members of a pair carry it, and it cancels in the difference.

The dose is numerically inert, so the token stream and the per-round verify
widths are identical to a dose-free leg. Pairs are therefore formed only
between neighbouring rounds of EQUAL width: round time depends strongly on
verify width, and pairing across widths would import that variance.

THE NULL LEG IS NOT OPTIONAL. Applying an alternating estimator to a series
that has its own period-2 structure would manufacture an effect. Every session
runs a dose-free leg through the identical estimator with the identical
hypothetical assignment; its interval must cover zero.

Round 0 is dropped, as in v1: it carries first-round warm effects.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

# Student t, two-sided 95 %, by degrees of freedom. Falls back to the normal
# quantile above the table, where the difference is under 1 %.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if df < k:
            return _T95[k]
    return 1.960


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            out[key] = value
    return out


def dose_accounting(stderr_log: pathlib.Path) -> dict[str, str] | None:
    """The worker prints its own alignment evidence at exit."""
    if not stderr_log.exists():
        return None
    for line in reversed(stderr_log.read_text().splitlines()):
        if line.startswith("e105_dose_accounting"):
            fields = {}
            for token in line.split()[1:]:
                key, _, value = token.partition("=")
                fields[key] = value
            return fields
    return None


def pair_rounds(us: list[float], widths: list[int],
                dosed: list[bool]) -> list[dict]:
    """Greedy non-overlapping neighbouring pairs at equal width.

    Left to right so every pair is adjacent in time, which is what makes the
    per-leg offset and any slow drift cancel.
    """
    pairs = []
    i = 1  # round 0 dropped
    while i + 1 < len(us):
        if widths[i] == widths[i + 1] and dosed[i] != dosed[i + 1]:
            hi, lo = (i, i + 1) if dosed[i] else (i + 1, i)
            pairs.append({
                "index_dosed": hi,
                "index_undosed": lo,
                "width": widths[i],
                "us_dosed": us[hi],
                "us_undosed": us[lo],
                "difference_us": us[hi] - us[lo],
            })
            i += 2
        else:
            i += 1
    return pairs


def analyse(leg_dir: pathlib.Path) -> dict:
    report = json.loads((leg_dir / "report.json").read_text())
    meta = read_meta(leg_dir / "meta.txt")
    accounting = dose_accounting(leg_dir / "stderr.log")

    us = [s * 1e6 for s in report["block_request_seconds"]]
    widths = list(report["effective_draft_lengths"])
    rounds = len(us)
    if len(widths) != rounds:
        raise SystemExit(
            f"{leg_dir}: {rounds} round times but {len(widths)} widths")

    # The dose lands on every second QUALIFYING forward. Round i therefore
    # carries it when i is odd, and only if there was exactly one qualifying
    # forward per round. That is checked, not assumed.
    dosed = [i % 2 == 1 for i in range(rounds)]
    alignment = None
    if accounting:
        forwards = int(accounting["qualifying_forwards"])
        alignment = {
            "qualifying_forwards": forwards,
            "dosed_forwards": int(accounting["dosed_forwards"]),
            "round_count": report["round_count"],
            "one_forward_per_round": forwards == report["round_count"],
            "alternate": accounting["alternate"] == "true",
            "dose": int(accounting["dose"]),
            "shape": accounting["shape"],
        }

    pairs = pair_rounds(us, widths, dosed)
    diffs = [p["difference_us"] for p in pairs]
    n = len(diffs)
    mean = statistics.fmean(diffs) if n else float("nan")
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    half = t95(n - 1) * sem if n > 1 else float("nan")
    control_round_us = statistics.fmean(us[1:]) if rounds > 1 else float("nan")

    by_width: dict[int, int] = {}
    for p in pairs:
        by_width[p["width"]] = by_width.get(p["width"], 0) + 1

    return {
        "leg": leg_dir.name,
        "harness": "local",
        "arm_label": meta.get("arm_label"),
        "arm_env": meta.get("arm_env", ""),
        "git_head": meta.get("git_head"),
        "git_dirty_build": meta.get("git_dirty_build"),
        "worker_sha256": meta.get("worker_sha256"),
        "golden_sha256": meta.get("golden_sha256"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "leg_wall_seconds": meta.get("leg_wall_seconds"),
        "all_tokens_matched": report["all_tokens_matched"],
        "round_count": report["round_count"],
        "dose_alignment": alignment,
        "rounds_used": rounds - 1,
        "pairs": n,
        "pairs_by_width": dict(sorted(by_width.items())),
        "e109_control_round_us": control_round_us,
        "paired_difference_mean_us": mean,
        "paired_difference_sd_us": sd,
        "paired_difference_sem_us": sem,
        "half_width_us": half,
        "half_width_percent": 100.0 * half / control_round_us if n > 1 else None,
        "effect_percent": 100.0 * mean / control_round_us if n else None,
        "pair_detail": pairs,
    }


BAR_US = 355.0
BAR_PERCENT = 0.20


def render(results: list[dict]) -> str:
    lines = [
        "E109 rung 0 v2 -- within-leg alternating dose   harness=local",
        "  ungated: cool_gate_passed_real_gate=false,"
        " gate_qualified_for_timing=false, official_or_ranked_score=false",
        "",
        f"{'leg':<12} {'arm':<8} {'matched':>8} {'rounds':>7} {'pairs':>6}"
        f" {'round us':>10} {'effect us':>10} {'+-95% us':>9}"
        f" {'+-95% pct':>10} {'entry C':>8}",
    ]
    for r in results:
        lines.append(
            f"{r['leg']:<12} {str(r['arm_label']):<8}"
            f" {str(r['all_tokens_matched']):>8} {r['round_count']:>7}"
            f" {r['pairs']:>6} {r['e109_control_round_us']:>10.0f}"
            f" {r['paired_difference_mean_us']:>+10.1f}"
            f" {r['half_width_us']:>9.1f}"
            f" {r['half_width_percent']:>10.3f}"
            f" {str(r['gpu_temp_entry_c']):>8}")
    lines.append("")
    for r in results:
        a = r["dose_alignment"]
        if a is None:
            lines.append(f"  {r['leg']}: no dose accounting (dose-free leg)")
            continue
        lines.append(
            f"  {r['leg']}: qualifying_forwards={a['qualifying_forwards']}"
            f" round_count={a['round_count']}"
            f" one_forward_per_round={a['one_forward_per_round']}"
            f" dosed_forwards={a['dosed_forwards']}"
            f" dose={a['dose']} shape={a['shape']}")
    lines.append("")
    for r in results:
        lo = r["paired_difference_mean_us"] - r["half_width_us"]
        hi = r["paired_difference_mean_us"] + r["half_width_us"]
        verdict = "covers zero" if lo <= 0 <= hi else "excludes zero"
        lines.append(
            f"  {r['leg']} ({r['arm_label']}): 95 % CI"
            f" [{lo:+.1f}, {hi:+.1f}] us -> {verdict};"
            f" pairs by width {r['pairs_by_width']}")
    lines.append("")
    best = min((r for r in results if r["pairs"] > 1),
               key=lambda r: r["half_width_us"], default=None)
    if best:
        lines.append(
            f"  best single-leg resolution {best['half_width_us']:.1f} us ="
            f" {best['half_width_percent']:.3f} % against a bar of"
            f" {BAR_US:.0f} us = {BAR_PERCENT:.2f} %"
            f" -> {'PASS' if best['half_width_us'] <= BAR_US else 'FAIL'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legs", nargs="+", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    results = [analyse(leg) for leg in args.legs]
    print(render(results))
    if args.json:
        args.json.write_text(json.dumps(
            {"protocol": "e109-rung0-v2-within-leg-alternating-dose",
             "harness": "local",
             "bar_us": BAR_US,
             "bar_percent": BAR_PERCENT,
             "legs": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
