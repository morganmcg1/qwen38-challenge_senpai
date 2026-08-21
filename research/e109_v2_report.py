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


def dose_accounting(witness: pathlib.Path) -> dict | None:
    """Read the worker's own record of the dose schedule, one line per forward.

    Returns None when the leg ran without `MLX_E105_DOSE_WITNESS`, which is the
    normal case for a timing leg: the witness is recorded once on a separate
    leg, so no timing leg pays for the file writes.
    """
    if not witness.exists():
        return None
    forwards: list[tuple[int, int]] = []
    fields: dict[str, str] = {}
    for line in witness.read_text().splitlines():
        if not line.startswith("e105_dose_forward"):
            continue
        fields = {}
        for token in line.split()[1:]:
            key, _, value = token.partition("=")
            fields[key] = value
        forwards.append((int(fields["forward"]), int(fields["dosed"])))
    if not forwards:
        return None
    # The estimator assumes round i carries the dose exactly when i is odd.
    # That holds only if the dose lands on every second qualifying forward,
    # which is checked here against the recorded sequence rather than assumed.
    alternation_exact = all(
        dosed == (1 if index % 2 == 0 else 0) for index, dosed in forwards)
    return {
        "qualifying_forwards": len(forwards),
        "dosed_forwards": sum(dosed for _, dosed in forwards),
        "alternation_exact": alternation_exact,
        "alternate": fields.get("alternate") == "true",
        "dose": int(fields.get("dose", 0)),
        "shape": fields.get("shape", ""),
    }


def pair_rounds(us: list[float], widths: list[int],
                dosed: list[bool]) -> list[dict]:
    """Greedy non-overlapping neighbouring pairs at equal width.

    Left to right so every pair is adjacent in time, which is what makes the
    per-leg offset cancel.
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


def triple_rounds(us: list[float], widths: list[int],
                  dosed: list[bool]) -> list[dict]:
    """Greedy non-overlapping equal-width triples, drift cancelled.

    The dose alternates DUDU..., so every adjacent pair is ordered the same
    way: dosed round first, undosed round second. A within-leg drift of r us
    per round therefore biases EVERY pair difference by the same -r, and the
    pair estimator cannot see it. A leg heats from entry to exit, so r is not
    assumed to be zero.

    A three-round contrast at one width removes it. For a DUD run the estimate
    is (x0 + x2)/2 - x1, and for UDU it is x1 - (x0 + x2)/2. Under a linear
    drift the outer mean sits exactly at the centre round's time, so the drift
    term cancels and only the dose survives. Per triple the variance is
    1.5 sigma^2 against 2 sigma^2 for a pair, so the cost is the third round,
    not precision.
    """
    triples = []
    i = 1  # round 0 dropped
    while i + 2 < len(us):
        equal_width = widths[i] == widths[i + 1] == widths[i + 2]
        alternating = dosed[i] != dosed[i + 1] and dosed[i + 1] != dosed[i + 2]
        if equal_width and alternating:
            outer_mean = (us[i] + us[i + 2]) / 2.0
            sign = 1.0 if dosed[i] else -1.0
            triples.append({
                "index_first": i,
                "width": widths[i],
                "pattern": "DUD" if dosed[i] else "UDU",
                "estimate_us": sign * (outer_mean - us[i + 1]),
            })
            i += 3
        else:
            i += 1
    return triples


def within_leg_drift_us_per_round(us: list[float],
                                  widths: list[int]) -> float | None:
    """Least-squares slope of round time on round index, within each width.

    Reported so a reader can see how large the bias on the pair estimator is
    without having to trust that it was removed.
    """
    numerator = 0.0
    denominator = 0.0
    by_width: dict[int, list[tuple[int, float]]] = {}
    for i in range(1, len(us)):
        by_width.setdefault(widths[i], []).append((i, us[i]))
    for samples in by_width.values():
        if len(samples) < 3:
            continue
        mean_i = statistics.fmean(s[0] for s in samples)
        mean_t = statistics.fmean(s[1] for s in samples)
        for index, value in samples:
            numerator += (index - mean_i) * (value - mean_t)
            denominator += (index - mean_i) ** 2
    if denominator == 0.0:
        return None
    return numerator / denominator


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values) if n else float("nan")
    sd = statistics.stdev(values) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    half = t95(n - 1) * sem if n > 1 else float("nan")
    return {"n": n, "mean_us": mean, "sd_us": sd, "sem_us": sem,
            "half_width_us": half}


def analyse(leg_dir: pathlib.Path) -> dict:
    report = json.loads((leg_dir / "report.json").read_text())
    meta = read_meta(leg_dir / "meta.txt")
    accounting = dose_accounting(leg_dir / "dose-witness.txt")

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
        # The witness counts EVERY qualifying forward in the process, and
        # warm-up runs many before the timed window opens: a 512-token leg
        # records 401 forwards against 77 rounds. So the timed rounds are the
        # last `round_count` forwards, preceded by `warmup_forwards`.
        #
        # Timed round i is then forward W + i + 1, which is dosed when
        # W + i + 1 is even. The estimator assumes round i is dosed when i is
        # odd, i.e. when i + 1 is even, so the two agree exactly when W is
        # even. An odd W would invert every pair and flip the sign of the
        # result, which is why this is checked rather than assumed.
        forwards = accounting["qualifying_forwards"]
        warmup = forwards - report["round_count"]
        alignment = {
            **accounting,
            "round_count": report["round_count"],
            "warmup_forwards": warmup,
            "parity_matches_estimator": warmup >= 0 and warmup % 2 == 0,
        }

    pairs = pair_rounds(us, widths, dosed)
    pair_stats = summarise([p["difference_us"] for p in pairs])
    triples = triple_rounds(us, widths, dosed)
    triple_stats = summarise([t["estimate_us"] for t in triples])
    drift = within_leg_drift_us_per_round(us, widths)
    n = pair_stats["n"]
    mean = pair_stats["mean_us"]
    half = pair_stats["half_width_us"]
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
        # What the leg was ASKED to do, read back from the recorded arm
        # environment. `dose_alignment` is separate: it is the worker's own
        # account of what it then did, and it is absent unless worker stderr
        # reaches the parent.
        "dose_requested": "MLX_E105_DOSE_ALTERNATE=1" in meta.get("arm_env", ""),
        "dose_alignment": alignment,
        "rounds_used": rounds - 1,
        "pairs": n,
        "pairs_by_width": dict(sorted(by_width.items())),
        "e109_control_round_us": control_round_us,
        "within_leg_drift_us_per_round": drift,
        "paired_difference_mean_us": mean,
        "paired_difference_sd_us": pair_stats["sd_us"],
        "paired_difference_sem_us": pair_stats["sem_us"],
        "half_width_us": half,
        "half_width_percent": 100.0 * half / control_round_us if n > 1 else None,
        "effect_percent": 100.0 * mean / control_round_us if n else None,
        "drift_cancelled_triples": triple_stats["n"],
        "drift_cancelled_mean_us": triple_stats["mean_us"],
        "drift_cancelled_half_width_us": triple_stats["half_width_us"],
        "pair_detail": pairs,
        "triple_detail": triples,
    }


BAR_US = 355.0
BAR_PERCENT = 0.20


def session_summary(results: list[dict]) -> dict:
    """Contrast the dosed legs against the null legs, not against zero.

    A null leg is not guaranteed to read zero. If decode rounds carry any
    period-2 structure of their own -- alternating buffer reuse, an internal
    two-phase schedule -- the alternating estimator reports it whether or not a
    dose is applied. The dose is then the DIFFERENCE between the dosed legs and
    the null legs, and quoting a dosed leg against zero would credit the dose
    with that offset.

    The interval combines the two arms as independent samples of leg-level
    estimates, so it prices between-leg scatter rather than assuming the
    within-leg interval is the whole story.
    """
    dosed = [r for r in results if r["dose_requested"]]
    null = [r for r in results if not r["dose_requested"]]

    def arm(legs: list[dict], key: str) -> dict:
        values = [leg[key] for leg in legs]
        n = len(values)
        mean = statistics.fmean(values) if n else float("nan")
        sd = statistics.stdev(values) if n > 1 else float("nan")
        return {"legs": n, "mean_us": mean, "sd_us": sd,
                "sem_us": sd / math.sqrt(n) if n > 1 else float("nan")}

    out = {}
    for key, label in (("paired_difference_mean_us", "pair"),
                       ("drift_cancelled_mean_us", "triple")):
        d, z = arm(dosed, key), arm(null, key)
        delta = d["mean_us"] - z["mean_us"]
        variance = 0.0
        df = 0
        for side in (d, z):
            if side["legs"] > 1 and not math.isnan(side["sem_us"]):
                variance += side["sem_us"] ** 2
                df += side["legs"] - 1
        half = t95(df) * math.sqrt(variance) if df > 0 else float("nan")
        out[label] = {
            "dosed": d, "null": z,
            "dose_minus_null_us": delta,
            "half_width_us": half,
            "excludes_zero": bool(df > 0 and abs(delta) > half),
        }

    halves = sorted(r["half_width_us"] for r in dosed if r["pairs"] > 1)
    control = statistics.fmean(r["e109_control_round_us"] for r in results)
    median_half = statistics.median(halves) if halves else float("nan")
    out["resolution"] = {
        "dosed_legs": len(halves),
        "single_leg_half_width_us_median": median_half,
        "single_leg_half_width_us_min": halves[0] if halves else None,
        "single_leg_half_width_us_max": halves[-1] if halves else None,
        "single_leg_half_width_percent": 100.0 * median_half / control,
        "control_round_us": control,
        "clears_bar": bool(median_half <= BAR_US),
        "round_alignment_verified": any(
            r["dose_alignment"]
            and r["dose_alignment"]["one_forward_per_round"]
            and r["dose_alignment"]["alternation_exact"]
            and r["dose_alignment"]["alternate"]
            for r in dosed),
    }

    # The deliverable is a recipe, not a single number: how many legs buy how
    # much resolution, and what that costs in wall clock. Pooling k legs of
    # independent within-leg pairs divides the half-width by sqrt(k).
    wall = [float(r["leg_wall_seconds"]) for r in results
            if r.get("leg_wall_seconds")]
    seconds_per_leg = statistics.fmean(wall) if wall else float("nan")
    if not math.isnan(median_half):
        out["cost_curve"] = [
            {"legs": k,
             "half_width_us": median_half / math.sqrt(k),
             "half_width_percent": 100.0 * median_half / math.sqrt(k) / control,
             "minutes": k * seconds_per_leg / 60.0,
             "clears_bar": bool(median_half / math.sqrt(k) <= BAR_US)}
            for k in (1, 2, 4, 8)
        ]
        needed = max(1, math.ceil((median_half / BAR_US) ** 2))
        out["legs_to_reach_bar"] = needed
        out["minutes_to_reach_bar"] = needed * seconds_per_leg / 60.0
        out["seconds_per_leg"] = seconds_per_leg
    return out


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
            state = ("DOSE REQUESTED but UNVERIFIED: no worker accounting"
                     " reached the parent; rerun with"
                     " MLX_DFLASH_TRACE_CACHE_SEAM=1"
                     if r["dose_requested"] else "dose-free null leg")
            lines.append(f"  {r['leg']}: {state}")
            continue
        lines.append(
            f"  {r['leg']}: qualifying_forwards={a['qualifying_forwards']}"
            f" round_count={a['round_count']}"
            f" one_forward_per_round={a['one_forward_per_round']}"
            f" alternation_exact={a['alternation_exact']}"
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
    for r in results:
        drift = r["within_leg_drift_us_per_round"]
        drift_text = "n/a" if drift is None else f"{drift:+.1f} us/round"
        lines.append(
            f"  {r['leg']}: within-leg drift {drift_text};"
            f" drift-cancelled triples n={r['drift_cancelled_triples']}"
            f" estimate {r['drift_cancelled_mean_us']:+.1f}"
            f" +-{r['drift_cancelled_half_width_us']:.1f} us")

    summary = session_summary(results)
    resolution = summary["resolution"]
    lines.append("")
    # The claim is about what ONE leg resolves, so the headline is the median
    # over the dosed legs. Taking the minimum would report the luck of the best
    # leg as though it were the method's resolution.
    if resolution["dosed_legs"]:
        lines.append(
            f"  single-leg resolution over {resolution['dosed_legs']} dosed"
            f" legs: median {resolution['single_leg_half_width_us_median']:.1f}"
            f" us, range {resolution['single_leg_half_width_us_min']:.1f}-"
            f"{resolution['single_leg_half_width_us_max']:.1f} us")
        lines.append(
            f"  median = {resolution['single_leg_half_width_percent']:.3f} %"
            f" of a {resolution['control_round_us']:.0f} us round,"
            f" against a bar of {BAR_US:.0f} us = {BAR_PERCENT:.2f} %"
            f" -> {'PASS' if resolution['clears_bar'] else 'FAIL'}")

    lines.append("")
    lines.append("  recovered dose, dosed legs minus null legs:")
    for label in ("pair", "triple"):
        block = summary[label]
        lines.append(
            f"    {label:<7} dosed {block['dosed']['mean_us']:+.1f}"
            f" (n={block['dosed']['legs']})"
            f"  null {block['null']['mean_us']:+.1f}"
            f" (n={block['null']['legs']})"
            f"  -> {block['dose_minus_null_us']:+.1f}"
            f" +-{block['half_width_us']:.1f} us"
            f" {'excludes' if block['excludes_zero'] else 'covers'} zero")

    if "cost_curve" in summary:
        lines.append("")
        lines.append(
            f"  cost of one decision at {summary['seconds_per_leg']:.0f} s/leg:")
        for point in summary["cost_curve"]:
            lines.append(
                f"    {point['legs']:>2} leg(s)"
                f"  +-{point['half_width_us']:>6.1f} us"
                f" = {point['half_width_percent']:.3f} %"
                f"  {point['minutes']:>5.1f} min"
                f"  {'clears' if point['clears_bar'] else 'short of'} the bar")
        lines.append(
            f"  -> {summary['legs_to_reach_bar']} leg(s) ="
            f" {summary['minutes_to_reach_bar']:.1f} min to resolve"
            f" {BAR_PERCENT:.2f} %")
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
             "session": session_summary(results),
             "legs": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
