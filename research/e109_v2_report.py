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
    forwards: list[dict] = []
    fields: dict[str, str] = {}
    for line in witness.read_text().splitlines():
        if not line.startswith("e105_dose_forward"):
            continue
        fields = {}
        for token in line.split()[1:]:
            key, _, value = token.partition("=")
            fields[key] = value
        forwards.append({"forward": int(fields["forward"]),
                         "dosed": int(fields["dosed"]),
                         "width": int(fields["width"])})
    if not forwards:
        return None
    alternation_exact = all(
        f["dosed"] == (1 if i % 2 == 1 else 0)
        for i, f in enumerate(forwards))
    return {
        "qualifying_forwards": len(forwards),
        "dosed_forwards": sum(f["dosed"] for f in forwards),
        "alternation_exact": alternation_exact,
        "alternate": fields.get("alternate") == "true",
        "dose": int(fields.get("dose", 0)),
        "shape": fields.get("shape", ""),
        "sequence": forwards,
    }


def forward_stream_diagnosis(leg_dir: pathlib.Path) -> dict:
    """What the qualifying-forward stream actually looks like inside one leg.

    The alternating estimator needs to know which TIMED ROUNDS carry the dose.
    The dose alternates over qualifying FORWARDS, so the two agree only if the
    boundary-fused chain sees exactly one forward per round. This reads the
    witness and checks that, using the row width as the fingerprint: a timed
    round verifies one row per proposed draft plus the primary token, so its
    width must be `effective_draft_lengths[i] + 1`.
    """
    report = json.loads((leg_dir / "report.json").read_text())
    accounting = dose_accounting(leg_dir / "dose-witness.txt")
    if accounting is None:
        raise SystemExit(f"{leg_dir}: no dose-witness.txt")
    sequence = accounting["sequence"]
    widths = [f["width"] for f in sequence]
    rounds = report["round_count"]
    expected = [k + 1 for k in report["effective_draft_lengths"]]

    def histogram(values: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    # `begin()` warms every legal shape before the timed window opens, so the
    # leg starts with an ascending 1..9 sweep. Everything after it is either a
    # timed round or work the estimator must not attribute to a round.
    sweep = 0
    while sweep + 1 < len(widths) and widths[sweep + 1] == widths[sweep] + 1:
        sweep += 1
    warmup_sweep = widths[:sweep + 1]

    return {
        "leg": leg_dir.name,
        "tokens": report["decode_token_count"],
        "round_count": rounds,
        "qualifying_forwards": len(sequence),
        "forwards_per_round": len(sequence) / rounds if rounds else float("nan"),
        "warmup_ascending_sweep": warmup_sweep,
        "width_histogram_observed": histogram(widths),
        "width_histogram_expected_for_rounds": histogram(expected),
        # A verify carries at least two rows whenever any draft was proposed,
        # so width-1 forwards can never be timed rounds in a drafting leg.
        "width_one_forwards": sum(1 for w in widths if w == 1),
        "wide_forwards_after_warmup": sum(
            1 for w in widths[sweep + 1:] if w >= 5),
        "verify_block_replayed_round_count": report[
            "verify_block_replayed_round_count"],
        "rejected_draft_total": report["rejected_draft_total"],
        "alternation_exact": accounting["alternation_exact"],
        "one_forward_per_round": len(sequence) - (sweep + 1) == rounds,
        "tail_width_fingerprint_matched": widths[-rounds:] == expected,
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

    # Which rounds carry the dose. Without a witness this is the parity
    # assumption "round i is dosed iff i is odd", which needs exactly one
    # qualifying forward per round and an even number of warm-up forwards.
    # Neither is safe: `begin()` warms every legal shape before the timed
    # window opens, so the leg starts with a run of non-round forwards.
    dosed = [i % 2 == 1 for i in range(rounds)]
    alignment = None
    if accounting:
        # With a witness the mapping is read off the ROW WIDTH instead. A
        # timed round verifies one row per proposed draft plus the primary
        # token, so round i has width `effective_draft_lengths[i] + 1`. The
        # timed rounds are the last `round_count` qualifying forwards, and
        # requiring that whole width vector to match the parent's own record
        # identifies them by fingerprint rather than by parity. The dose flags
        # of those forwards then replace the assumption entirely.
        sequence = accounting["sequence"]
        expected = [k + 1 for k in widths]
        tail = sequence[-rounds:] if len(sequence) >= rounds else []
        observed = [f["width"] for f in tail]
        matched = bool(tail) and observed == expected
        warmup = len(sequence) - rounds
        witness_dosed = [f["dosed"] == 1 for f in tail]
        if matched:
            dosed = witness_dosed
        alignment = {
            **{k: v for k, v in accounting.items() if k != "sequence"},
            "round_count": report["round_count"],
            "warmup_forwards": warmup,
            "warmup_widths": [f["width"] for f in sequence[:warmup]],
            "timed_widths_observed": observed,
            "timed_widths_expected": expected,
            "width_fingerprint_matched": matched,
            "parity_assumption_agrees": (
                matched and witness_dosed == [i % 2 == 1 for i in range(rounds)]
            ),
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


def synthetic_injection(leg_dirs: list[pathlib.Path], delta_us: float) -> dict:
    """Recover a known effect of exactly `delta_us` from real null-leg data.

    The GPU dose probe can only prove that the estimator sees SOMETHING; it
    cannot prove the estimator returns the right number, because the true size
    of the injected cost is not known independently. This check does. It takes
    the measured round times of a leg that carried NO dose, adds `delta_us` to
    the rounds the estimator will call dosed, and runs the same estimator.
    The answer must come back at `delta_us` and the interval must cover it.

    Everything except the injection is real measured data, so the round-to-
    round noise, the width sequence and the within-leg drift are the ones the
    protocol actually faces.
    """
    per_leg = []
    for leg_dir in leg_dirs:
        report = json.loads((leg_dir / "report.json").read_text())
        widths = list(report["effective_draft_lengths"])
        rounds = len(report["block_request_seconds"])
        dosed = [i % 2 == 1 for i in range(rounds)]
        us = [s * 1e6 + (delta_us if d else 0.0)
              for s, d in zip(report["block_request_seconds"], dosed)]
        pairs = summarise(
            [p["difference_us"] for p in pair_rounds(us, widths, dosed)])
        triples = summarise(
            [t["estimate_us"] for t in triple_rounds(us, widths, dosed)])
        per_leg.append({
            "leg": leg_dir.name,
            "pair_mean_us": pairs["mean_us"],
            "pair_half_width_us": pairs["half_width_us"],
            "pair_covers_delta": bool(
                abs(pairs["mean_us"] - delta_us) <= pairs["half_width_us"]),
            "triple_mean_us": triples["mean_us"],
            "triple_half_width_us": triples["half_width_us"],
            "triple_covers_delta": bool(
                abs(triples["mean_us"] - delta_us) <= triples["half_width_us"]),
        })
    return {
        "injected_us": delta_us,
        "legs": per_leg,
        "pair_bias_us": statistics.fmean(
            leg["pair_mean_us"] - delta_us for leg in per_leg),
        "triple_bias_us": statistics.fmean(
            leg["triple_mean_us"] - delta_us for leg in per_leg),
        "all_legs_cover_delta": all(
            leg["pair_covers_delta"] and leg["triple_covers_delta"]
            for leg in per_leg),
    }


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
        "round_alignment_verified": all(
            r["dose_alignment"]
            and r["dose_alignment"]["width_fingerprint_matched"]
            and r["dose_alignment"]["alternation_exact"]
            and r["dose_alignment"]["alternate"]
            for r in dosed) if dosed else False,
    }
    # The frame, named. A percent without the round it divides is not a
    # reportable number, so every consumer of this protocol gets the absolute
    # microseconds per round under the v2 key as well as per arm.
    out["e109_v2_control_round_us"] = control
    out["arms"] = {}
    for label in sorted({r["arm_label"] for r in results if r["arm_label"]}):
        legs = [r for r in results if r["arm_label"] == label]
        rounds = [r["e109_control_round_us"] for r in legs]
        out["arms"][label] = {
            "legs": len(legs),
            "e109_v2_control_round_us": statistics.fmean(rounds),
            "control_round_us_sd": (
                statistics.stdev(rounds) if len(rounds) > 1 else float("nan")),
            "dose_requested": legs[0]["dose_requested"],
            "half_width_us_median": statistics.median(
                [r["half_width_us"] for r in legs if r["pairs"] > 1]
                or [float("nan")]),
        }

    # The deliverable is a recipe, not a single number: how many legs buy how
    # much resolution, and what that costs in wall clock. Pooling k legs of
    # independent within-leg pairs divides the half-width by sqrt(k).
    wall = [float(r["leg_wall_seconds"]) for r in results
            if r.get("leg_wall_seconds")]
    seconds_per_leg = statistics.fmean(wall) if wall else float("nan")
    # Resolution and DETECTION are different questions. A half-width under the
    # bar only says the interval is narrow enough; whether a bar-sized arm
    # comes back with an interval that excludes zero also depends on the leg's
    # own offset, which the null legs measure directly. Power is the normal
    # approximation P(|estimate| > half-width) at a true effect of one bar.
    null_scatter = statistics.stdev(
        [r["paired_difference_mean_us"] for r in null]) if len(null) > 1 else (
            float("nan"))
    effect = BAR_PERCENT / 100.0 * control

    def power(k: int) -> float:
        if math.isnan(null_scatter) or null_scatter <= 0:
            return float("nan")
        z = (effect - median_half / math.sqrt(k)) / (
            null_scatter / math.sqrt(k))
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if not math.isnan(median_half):
        out["detection"] = {
            "effect_us": effect,
            "null_leg_estimate_scatter_us": null_scatter,
        }
        out["cost_curve"] = [
            {"legs": k,
             "half_width_us": median_half / math.sqrt(k),
             "half_width_percent": 100.0 * median_half / math.sqrt(k) / control,
             "minutes_per_arm": k * seconds_per_leg / 60.0,
             # A decision needs the null arm as well, so it costs 2k legs.
             "decision_minutes": 2 * k * seconds_per_leg / 60.0,
             "clears_bar": bool(median_half / math.sqrt(k) <= BAR_US),
             "power_at_one_bar": power(k)}
            for k in (1, 2, 4, 8)
        ]
        needed = max(1, math.ceil((median_half / BAR_US) ** 2))
        out["legs_to_reach_bar"] = needed
        out["minutes_to_reach_bar"] = needed * seconds_per_leg / 60.0
        out["seconds_per_leg"] = seconds_per_leg
    return out


def render_forward_stream(diagnoses: list[dict]) -> list[str]:
    lines = ["", "  FORWARD STREAM (diagnostic legs, witness enabled):"]
    for d in diagnoses:
        lines.append(
            f"    {d['leg']}: tokens={d['tokens']} rounds={d['round_count']}"
            f" qualifying_forwards={d['qualifying_forwards']}"
            f" ({d['forwards_per_round']:.2f} per round)"
            f" alternation_exact={d['alternation_exact']}")
        lines.append(
            f"      warmup ascending sweep {d['warmup_ascending_sweep']}"
            f"   width histogram {d['width_histogram_observed']}")
        lines.append(
            f"      widths the ROUNDS need"
            f" {d['width_histogram_expected_for_rounds']}"
            f"   width-1 forwards {d['width_one_forwards']}"
            f"   wide (>=5) after warmup {d['wide_forwards_after_warmup']}"
            f"   replayed verify blocks"
            f" {d['verify_block_replayed_round_count']}")
        lines.append(
            f"      one_forward_per_round={d['one_forward_per_round']}"
            f"   tail_width_fingerprint_matched="
            f"{d['tail_width_fingerprint_matched']}")
    return lines


def render_injection(check: dict) -> list[str]:
    lines = ["",
             f"  SYNTHETIC INJECTION on null legs, known effect"
             f" {check['injected_us']:.1f} us:"]
    for leg in check["legs"]:
        lines.append(
            f"    {leg['leg']}: pair {leg['pair_mean_us']:+.1f}"
            f" +-{leg['pair_half_width_us']:.1f} us"
            f" {'covers' if leg['pair_covers_delta'] else 'MISSES'};"
            f" triple {leg['triple_mean_us']:+.1f}"
            f" +-{leg['triple_half_width_us']:.1f} us"
            f" {'covers' if leg['triple_covers_delta'] else 'MISSES'}")
    lines.append(
        f"    bias: pair {check['pair_bias_us']:+.1f} us,"
        f" triple {check['triple_bias_us']:+.1f} us;"
        f" all legs cover the injected value:"
        f" {check['all_legs_cover_delta']}")
    return lines


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
            state = ("DOSE REQUESTED but UNVERIFIED: this leg ran without"
                     " MLX_E105_DOSE_WITNESS, so the round mapping is the"
                     " parity assumption"
                     if r["dose_requested"] else "dose-free null leg")
            lines.append(f"  {r['leg']}: {state}")
            continue
        lines.append(
            f"  {r['leg']}: qualifying_forwards={a['qualifying_forwards']}"
            f" round_count={a['round_count']}"
            f" warmup_forwards={a['warmup_forwards']}"
            f" warmup_widths={a['warmup_widths']}")
        lines.append(
            f"    width_fingerprint_matched={a['width_fingerprint_matched']}"
            f" alternation_exact={a['alternation_exact']}"
            f" parity_assumption_agrees={a['parity_assumption_agrees']}"
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
            f"  cost at {summary['seconds_per_leg']:.0f} s/leg;"
            f" a decision needs the null arm too, so it costs twice:")
        for point in summary["cost_curve"]:
            lines.append(
                f"    {point['legs']:>2} leg(s)/arm"
                f"  +-{point['half_width_us']:>6.1f} us"
                f" = {point['half_width_percent']:.3f} %"
                f"  {point['minutes_per_arm']:>5.1f} min/arm"
                f"  {point['decision_minutes']:>5.1f} min/decision"
                f"  {'clears' if point['clears_bar'] else 'short of'} the bar"
                f"  power at one bar {point['power_at_one_bar']:.2f}")
        lines.append(
            f"  -> {summary['legs_to_reach_bar']} leg(s)/arm ="
            f" {summary['minutes_to_reach_bar']:.1f} min/arm"
            f" ({2 * summary['minutes_to_reach_bar']:.1f} min/decision)"
            f" to resolve {BAR_PERCENT:.2f} %")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legs", nargs="+", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument(
        "--witness-leg", action="append", default=[], type=pathlib.Path,
        help="diagnostic leg that ran with MLX_E105_DOSE_WITNESS")
    parser.add_argument(
        "--inject-us", type=float,
        help="known effect to recover from the null legs;"
             " defaults to the bar as a fraction of the measured round")
    args = parser.parse_args()

    results = [analyse(leg) for leg in args.legs]
    text = render(results)
    diagnoses = [forward_stream_diagnosis(leg) for leg in args.witness_leg]
    if diagnoses:
        text += "\n" + "\n".join(render_forward_stream(diagnoses))
    null_legs = [leg for leg, r in zip(args.legs, results)
                 if not r["dose_requested"]]
    inject_us = args.inject_us
    if inject_us is None:
        inject_us = BAR_PERCENT / 100.0 * statistics.fmean(
            r["e109_control_round_us"] for r in results)
    injection = (synthetic_injection(null_legs, inject_us)
                 if null_legs else None)
    if injection:
        text += "\n" + "\n".join(render_injection(injection))
    print(text)
    if args.json:
        args.json.write_text(json.dumps(
            {"protocol": "e109-rung0-v2-within-leg-alternating-dose",
             "harness": "local",
             "bar_us": BAR_US,
             "bar_percent": BAR_PERCENT,
             "session": session_summary(results),
             "forward_stream_diagnosis": diagnoses,
             "synthetic_injection": injection,
             "legs": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
