#!/usr/bin/env python3
"""E154 R2 delay-injection boundedness analysis.

Reads the per-arm run directories written by `research/e154_r2_session.sh`
and answers one question: when the host stalls for `delta` microseconds
inside a scored round, how much of that stall does the round absorb?

  absorbed fully  (slope 0) -> the round was waiting on the GPU, so the
                               host had at least `delta` of slack
  absorbed not at all (slope 1) -> the round was waiting on the host

The same probe runs on the device side with a dependent matmul chain. The
two slopes together separate GPU-bound from dispatch-bound.

harness=local. Never an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

RECORD_PREFIX = "e154-arm: "


def parse_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        row = {}
        for field in line[len(RECORD_PREFIX):].split():
            if "=" not in field:
                continue
            key, _, value = field.partition("=")
            try:
                row[key] = float(value) if "." in value or "e" in value.lower() \
                    else int(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def parse_meta(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    meta = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            meta[key.strip()] = value.strip()
    return meta


def load_arm(run_dir: pathlib.Path) -> dict | None:
    meta = parse_meta(run_dir / "meta.txt")
    if not meta:
        return None
    report = {}
    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}
    return {
        "dir": run_dir.name,
        "arm": meta.get("arm", "?"),
        "position": int(meta.get("position", 0)),
        "meta": meta,
        "report": report,
        "rounds": parse_records(run_dir / "rounds.txt"),
    }


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Slope, intercept, and standard error of the slope."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(residual / (n - 2) / sxx) if n > 2 else float("nan")
    return slope, intercept, se


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = p / 100.0 * (len(ordered) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": pct(5),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "sd": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def level_curve(rounds: list[dict], level_key: str) -> dict:
    """Mean wall clock per injection level, differenced against level zero.

    Levels cycle by drafting-round index, so every level is interleaved
    across the leg and monotone thermal drift is shared.

    Everything that decides the knee runs on the modal draft count alone.
    Pooling across `d` is what would ruin this measurement: `d` moves the
    round by tens of milliseconds, so the all-`d` spread is an order of
    magnitude larger than the effect and would hide any real knee.
    """
    by_level: dict[float, list[dict]] = defaultdict(list)
    for row in rounds:
        by_level[float(row.get(level_key, 0))].append(row)
    if 0.0 not in by_level:
        return {"error": f"no zero level present for {level_key}"}

    modal_d = Counter(r.get("d") for r in rounds).most_common(1)[0][0]
    out = {"level_key": level_key, "modal_d": modal_d, "levels": []}
    base_all = statistics.fmean([r["wall_us"] for r in by_level[0.0]])
    base_modal = [r["wall_us"] for r in by_level[0.0] if r.get("d") == modal_d]
    if len(base_modal) < 2:
        return {"error": f"level zero has {len(base_modal)} modal-d rounds"}
    base_mean = statistics.fmean(base_modal)
    base_sem = statistics.stdev(base_modal) / math.sqrt(len(base_modal))

    for level in sorted(by_level):
        rows = by_level[level]
        walls = [r["wall_us"] for r in rows]
        modal = [r["wall_us"] for r in rows if r.get("d") == modal_d]
        entry = {
            "level": level,
            "n": len(rows),
            "mean_wall_us": statistics.fmean(walls),
            "sd_wall_us": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "delta_wall_us": statistics.fmean(walls) - base_all,
            "mean_thread_cpu_us": statistics.fmean(
                [r["thread_cpu_us"] for r in rows]),
            "mean_process_cpu_us": statistics.fmean(
                [r["process_cpu_us"] for r in rows]),
            "n_modal_d": len(modal),
        }
        if len(modal) >= 2:
            mean = statistics.fmean(modal)
            sem = statistics.stdev(modal) / math.sqrt(len(modal))
            delta = mean - base_mean
            threshold = 3.0 * math.sqrt(sem ** 2 + base_sem ** 2)
            entry.update({
                "mean_wall_us_modal_d": mean,
                "sem_wall_us_modal_d": sem,
                "delta_wall_us_modal_d": delta,
                "delta_3sigma_threshold_us": threshold,
                "delta_is_significant": abs(delta) > threshold,
                # Absorption can only fail upward. A significantly NEGATIVE
                # delta still means the round swallowed the stall, so the
                # knee search must use the one-sided test.
                "delta_is_significant_increase": delta > threshold,
                # 1.0 means the round swallowed the whole stall, 0.0 means it
                # passed straight through to the wall clock.
                "absorbed_fraction": (
                    1.0 - delta / level if level > 0 else None),
            })
        out["levels"].append(entry)

    usable = [e for e in out["levels"] if "delta_wall_us_modal_d" in e]
    xs = [e["level"] for e in usable]
    ys = [e["delta_wall_us_modal_d"] for e in usable]
    slope, intercept, se = ols(xs, ys)
    out["absorption_slope"] = slope
    out["absorption_slope_se"] = se
    out["absorption_intercept_us"] = intercept

    rows_modal = [r for r in rounds if r.get("d") == modal_d]
    s2, _, se2 = ols(
        [float(r.get(level_key, 0)) for r in rows_modal],
        [r["wall_us"] for r in rows_modal])
    out["per_round_slope_modal_d"] = s2
    out["per_round_slope_modal_d_se"] = se2
    out["per_round_n_modal_d"] = len(rows_modal)

    # The knee is the largest level the round still absorbs, and the first
    # level above it brackets the true value. Report both ends: with a
    # geometric ladder the bracket is wide and quoting one number would
    # overstate the resolution.
    knee = 0.0
    knee_upper = None
    for entry in usable:
        if entry["level"] == 0.0:
            continue
        if not entry["delta_is_significant_increase"]:
            knee = entry["level"]
        else:
            knee_upper = entry["level"]
            break
    out["slack_us_per_round"] = knee
    out["knee_bracket_us"] = [knee, knee_upper]
    out["zero_level_sd_us"] = (
        statistics.stdev(base_modal) if len(base_modal) > 1 else 0.0)
    out["zero_level_sem_us"] = base_sem

    # Slope inside the absorbing region only. This is the number that
    # separates "absorbed" from "leaks a little"; the pooled slope is
    # dominated by the levels above the knee.
    below = [e for e in usable if 0 < e["level"] <= knee] if knee else []
    if len(below) >= 2:
        s3, _, se3 = ols([0.0] + [e["level"] for e in below],
                         [0.0] + [e["delta_wall_us_modal_d"] for e in below])
        out["slope_below_knee_us_per_us"] = s3
        out["slope_below_knee_se"] = se3
    elif below:
        e = below[0]
        out["slope_below_knee_us_per_us"] = (
            e["delta_wall_us_modal_d"] / e["level"])
        out["slope_below_knee_se"] = e["delta_3sigma_threshold_us"] / 3.0 \
            / e["level"]
    above = [e for e in usable if e["level"] > knee and e["level"] > 0]
    if len(above) >= 2:
        s4, _, se4 = ols([e["level"] for e in above],
                         [e["delta_wall_us_modal_d"] for e in above])
        out["slope_above_knee_us_per_us"] = s4
        out["slope_above_knee_se"] = se4

    # The CPU ladder is already in microseconds. The GPU ladder is in chain
    # steps, so it needs its own measured price before the two slacks can be
    # compared. The above-knee slope IS that price: once the device has no
    # idle left, each extra step costs its full serial time.
    if level_key != "gpu_chain_steps":
        out.update(hinge_fit(usable))
    else:
        price = out.get("slope_above_knee_us_per_us")
        out["us_per_chain_step"] = price
        if price:
            out["slack_us_per_round"] = out["slack_us_per_round"] * price
            out["knee_bracket_us"] = [
                (b * price if b is not None else None)
                for b in out["knee_bracket_us"]]
            # The hinge is a statement about time, so it has to see the
            # ladder in time. Fitting it on raw step counts would compare
            # a step index with a microsecond delta.
            out.update(hinge_fit([
                dict(e, level=e["level"] * price) for e in usable]))
        # Marginal cost between consecutive levels. A step that costs less
        # than the asymptotic price was partly hidden behind device work the
        # round was already doing, which is the only place GPU idle can show.
        marginal = []
        for previous, entry in zip(usable, usable[1:]):
            span = entry["level"] - previous["level"]
            if span <= 0:
                continue
            marginal.append({
                "from_steps": previous["level"],
                "to_steps": entry["level"],
                "us_per_step": (entry["delta_wall_us_modal_d"]
                                - previous["delta_wall_us_modal_d"]) / span,
            })
        out["marginal_us_per_chain_step"] = marginal
    return out


# FINDING 281 regresses ranked round time on tokens per round and reads the
# intercept as one weight stream. Our zero arms vary both the draft count and
# the accepted count round to round, so we can fit the same law locally and,
# more usefully, ask which regressor it should have used. Emitted tokens and
# verified rows are different quantities: a round that drafts 7 and accepts 2
# costs the same device work as one that drafts 7 and accepts 7, but emits
# five fewer tokens. Whichever regressor explains more variance is the one the
# cost law belongs to, and that decides whether acceptance is nearly free.
FINDING_281 = {"intercept_us": 25409.0, "slope_us_per_token": 4291.0,
               "r": 0.9933, "harness": "ranked"}


def cost_law(rounds: list[dict]) -> dict:
    """Fit round time against emitted tokens and against verified rows."""
    if len(rounds) < 8:
        return {"error": f"only {len(rounds)} zero-arm rounds"}
    walls = [r["wall_us"] for r in rounds]
    out = {"n": len(rounds), "finding_281": dict(FINDING_281)}
    for name, xs in (
            ("emitted_tokens", [r["acc"] + 1 for r in rounds]),
            ("verified_rows", [r["d"] + 1 for r in rounds])):
        slope, intercept, se = ols([float(x) for x in xs], walls)
        out[name] = {
            "intercept_us": intercept,
            "slope_us_per_unit": slope,
            "slope_se": se,
            "r": pearson([float(x) for x in xs], walls),
            "mean_x": statistics.fmean(xs),
        }
    # Both regressors are fitted; the campaign should price against whichever
    # one actually explains the round.
    better = max(("emitted_tokens", "verified_rows"),
                 key=lambda k: abs(out[k]["r"]))
    out["dominant_regressor"] = better
    out["intercept_ratio_local_over_ranked"] = (
        out[better]["intercept_us"] / FINDING_281["intercept_us"])
    out["slope_ratio_local_over_ranked"] = (
        out[better]["slope_us_per_unit"] / FINDING_281["slope_us_per_token"])
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def hinge_fit(levels: list[dict]) -> dict:
    """Fit `delta = max(0, level - S)` and return S.

    A geometric ladder brackets the knee only to within its own spacing,
    which here is a factor of four. The hinge model uses the levels ABOVE
    the knee to locate it: if the round absorbs `S` and passes the rest
    through at unit slope, then every such level reports `S = level - delta`
    independently. Their agreement is the test of the model, so the
    per-level estimates are published next to the fit rather than hidden
    behind it.
    """
    points = [(e["level"], e["delta_wall_us_modal_d"]) for e in levels]
    if len(points) < 3:
        return {}

    per_level = [
        {"level": lvl, "implied_slack_us": lvl - delta}
        for lvl, delta in points
        if lvl > 0 and delta > 0]

    top = max(lvl for lvl, _ in points)
    best_s, best_loss = 0.0, float("inf")
    steps = 2000
    for i in range(steps + 1):
        candidate = top * i / steps
        loss = sum((delta - max(0.0, lvl - candidate)) ** 2
                   for lvl, delta in points)
        if loss < best_loss:
            best_s, best_loss = candidate, loss

    residuals = [delta - max(0.0, lvl - best_s) for lvl, delta in points]
    out = {
        "hinge_slack_us": best_s,
        "hinge_rms_residual_us": math.sqrt(best_loss / len(points)),
        "hinge_max_abs_residual_us": max(abs(r) for r in residuals),
        "hinge_per_level_implied_slack": per_level,
    }
    if len(per_level) > 1:
        implied = [p["implied_slack_us"] for p in per_level]
        out["hinge_implied_slack_spread_us"] = max(implied) - min(implied)
        out["hinge_implied_slack_median_us"] = statistics.median(implied)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.runs_dir)
    arms = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if args.label and not child.name.startswith(args.label):
            continue
        arm = load_arm(child)
        if arm:
            arms.append(arm)
    if not arms:
        print(f"no arms under {root}", file=sys.stderr)
        return 1

    result = {
        "experiment": "e154-r2-round-boundedness",
        "harness": "local",
        "official_or_ranked_score": False,
        "runs_dir": str(root),
        "arms": [],
    }

    all_zero_walls: list[float] = []
    all_zero_rounds: list[dict] = []
    for arm in arms:
        meta = arm["meta"]
        entry = {
            "dir": arm["dir"],
            "arm": arm["arm"],
            "position": arm["position"],
            "arm_env": meta.get("arm_env"),
            "tokens": meta.get("tokens"),
            "all_tokens_matched": meta.get("all_tokens_matched"),
            "residual_divergence_count": meta.get("residual_divergence_count"),
            "round_count": meta.get("round_count"),
            "seconds_per_token": meta.get("seconds_per_token"),
            "effective_mean_draft_len": meta.get("effective_mean_draft_len"),
            "accepted_draft_rate": meta.get("accepted_draft_rate"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "cool_gate_passed_real_gate": meta.get(
                "cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "instrument_records": len(arm["rounds"]),
        }
        rounds = arm["rounds"]
        if rounds:
            entry["wall_clock_us"] = describe([r["wall_us"] for r in rounds])
            entry["thread_cpu_us"] = describe(
                [r["thread_cpu_us"] for r in rounds])
            entry["process_cpu_us"] = describe(
                [r["process_cpu_us"] for r in rounds])
            entry["draft_count_histogram"] = dict(
                sorted(Counter(r.get("d") for r in rounds).items()))
            entry["accepted_histogram"] = dict(
                sorted(Counter(r.get("acc") for r in rounds).items()))
            injected = {float(r.get("cpu_injected_us", 0)) for r in rounds}
            chains = {int(r.get("gpu_chain_steps", 0)) for r in rounds}
            if injected - {0.0}:
                entry["cpu_curve"] = level_curve(rounds, "cpu_injected_us")
            if chains - {0}:
                entry["gpu_curve"] = level_curve(rounds, "gpu_chain_steps")
            if arm["arm"] == "zero":
                all_zero_walls.extend(r["wall_us"] for r in rounds)
                all_zero_rounds.extend(rounds)
        report = arm["report"]
        for key in ("first_block_seconds", "seed_prefill_seconds",
                    "p50_block_request_seconds_after_first",
                    "max_block_request_seconds_after_first"):
            if key in report:
                entry[key] = report[key]
        result["arms"].append(entry)

    result["e154_absolute_round_wall_clock_us"] = describe(all_zero_walls)
    result["e154_host_syncs_per_round"] = 1
    result["e154_local_cost_law"] = cost_law(all_zero_rounds)

    # F4 asks for the knee and the sub-knee slope by name. `cpu_eval` is the
    # site the FINDING 281 prediction is actually about: it burns while the
    # whole round's device work is outstanding, so its knee is the host slack
    # against the round's GPU busy time. `cpu_pre` burns while only the head
    # chain is outstanding and answers a narrower question.
    cpu_slack: dict[str, list[float]] = defaultdict(list)
    gpu_slack: list[float] = []
    knee: dict[str, list[float]] = defaultdict(list)
    bracket: dict[str, list[list]] = defaultdict(list)
    slope_below: dict[str, list[float]] = defaultdict(list)
    for arm in result["arms"]:
        curve = arm.get("cpu_curve")
        if curve and "error" not in curve:
            site = "preeval" if arm["arm"] == "cpu_eval" else "preverify"
            knee[site].append(curve["slack_us_per_round"])
            bracket[site].append(curve["knee_bracket_us"])
            if "slope_below_knee_us_per_us" in curve:
                slope_below[site].append(curve["slope_below_knee_us_per_us"])
            if "hinge_slack_us" in curve:
                cpu_slack[site].append(curve["hinge_slack_us"])
        curve = arm.get("gpu_curve")
        if curve and "error" not in curve and curve.get("hinge_slack_us"):
            gpu_slack.append(curve["hinge_slack_us"])

    # Replicates are kept side by side. A knee that does not survive its own
    # replicate is a property of the detector, not of the round.
    result["e154_fixed_term_absorption_knee_us"] = {
        site: statistics.fmean(v) for site, v in knee.items()}
    result["e154_fixed_term_absorption_knee_replicates_us"] = dict(knee)
    result["e154_knee_bracket_us"] = {
        site: [min(b[0] for b in v),
               (max(b[1] for b in v) if all(b[1] is not None for b in v)
                else None)]
        for site, v in bracket.items()}
    result["e154_delay_slope_below_knee_us_per_us"] = {
        site: statistics.fmean(v) for site, v in slope_below.items()}
    result["e154_delay_slope_below_knee_replicates"] = dict(slope_below)

    # The two headline numbers, both in microseconds per round so they can be
    # compared directly. Replicate arms are averaged; their spread is
    # published so the average is not read as more precise than it is.
    result["e154_cpu_slack_us_per_round"] = {
        site: statistics.fmean(values) for site, values in cpu_slack.items()}
    result["e154_cpu_slack_replicate_spread_us"] = {
        site: (max(values) - min(values))
        for site, values in cpu_slack.items() if len(values) > 1}
    if gpu_slack:
        result["e154_gpu_slack_us_per_round"] = statistics.fmean(gpu_slack)
        if len(gpu_slack) > 1:
            result["e154_gpu_slack_replicate_spread_us"] = (
                max(gpu_slack) - min(gpu_slack))
    preeval = cpu_slack.get("preeval")
    if preeval and gpu_slack:
        result["e154_cpu_over_gpu_slack_ratio"] = (
            statistics.fmean(preeval) / statistics.fmean(gpu_slack))
        # The verdict must survive the worst pairing the data allows: the
        # smallest CPU slack measured at either site against the largest GPU
        # slack the nonparametric bracket permits.
        gpu_upper = max(a["gpu_curve"]["knee_bracket_us"][1]
                        for a in result["arms"] if a.get("gpu_curve"))
        cpu_lower = min(v for values in cpu_slack.values() for v in values)
        result["e154_cpu_over_gpu_slack_ratio_bounds"] = {
            "adversarial_min": cpu_lower / gpu_upper,
            "adversarial_cpu_slack_us": cpu_lower,
            "adversarial_gpu_slack_us": gpu_upper,
            "preeval_vs_gpu_bracket_upper": min(preeval) / gpu_upper,
        }

    # Token neutrality: the instrument must not change what is generated.
    matched = {a["all_tokens_matched"] for a in result["arms"]}
    rounds_seen = {a["round_count"] for a in result["arms"]}
    edl = {a["effective_mean_draft_len"] for a in result["arms"]}
    result["e154_instrument_is_token_neutral"] = {
        "all_tokens_matched": sorted(matched),
        "round_count": sorted(rounds_seen),
        "effective_mean_draft_len": sorted(edl),
        "verdict": (
            matched == {"true"} and len(rounds_seen) == 1 and len(edl) == 1),
    }

    # Zero-arm drift check. The session is void if the control moves more
    # than the effect it is supposed to measure.
    zero_arms = [a for a in result["arms"] if a["arm"] == "zero"
                 and a.get("wall_clock_us")]
    if len(zero_arms) > 1:
        means = [a["wall_clock_us"]["mean"] for a in zero_arms]
        result["zero_control_drift_us"] = max(means) - min(means)
        result["zero_control_means_us"] = means

    temps = [float(a["gpu_temp_entry_c"]) for a in result["arms"]
             if a.get("gpu_temp_entry_c")]
    if temps:
        result["entry_temperature_spread_c"] = max(temps) - min(temps)
        result["entry_temperatures_c"] = temps

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
