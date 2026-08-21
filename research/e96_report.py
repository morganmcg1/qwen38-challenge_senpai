#!/usr/bin/env python3
"""Summarise E96 legs: per-round trace statistics beside the score metrics.

    usage: research/e96_report.py [--fit-json PATH] [--bucket D,ACC] TAG [TAG ...]

Each leg contributes its rounds keyed by (d, acc), because the repair path a
round takes depends on both. A comparison between two arms is only valid
inside one such bucket.

`--fit-json` writes the analysis payload the W&B publisher consumes. It fits
round cost against the repeat count over the `rep` arms. Those arms emit one
shared token stream, so they populate one shared bucket and the slope is the
marginal cost of one recurrent step.
"""
import argparse
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
FIELDS = (
    "round_us",
    "verify_build_us",
    "eval_wall_us",
    "draft_build_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
)


def parse_meta(tag):
    meta = {}
    path = OUT / tag / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    return meta


def parse_rounds(tag):
    rounds = []
    path = OUT / tag / "trace.txt"
    if not path.exists():
        return rounds
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        record = {}
        for token in line[len("mtp-trace: "):].split():
            key, _, value = token.partition("=")
            try:
                record[key] = int(value)
            except ValueError:
                continue
        rounds.append(record)
    return rounds


def score(tag):
    """Leg metrics from whichever producer ran.

    A wrapper leg leaves the composed `score.json`. A direct-CLI leg leaves the
    trusted `mtp-timed` report itself, so the few comparable fields are lifted
    out of it under the same names. An ablated arm may leave no report at all
    when its post-window audit throws; the trace still carries the rounds.
    """
    path = OUT / tag / "score.json"
    if path.exists():
        return json.loads(path.read_text()).get("metrics", {})
    path = OUT / tag / "mtp-decode.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    metrics = {
        "mode": "direct-mtp-timed",
        "decode_tokens": report.get("decode_token_count"),
        "mtp_depth": report.get("mtp_depth"),
        "all_tokens_matched": report.get("all_tokens_matched"),
        "mtp_seconds_per_token": report.get("parent_measured_seconds_per_token"),
        "accepted_draft_rate": report.get("accepted_draft_rate"),
        "effective_mean_draft_len": report.get("effective_mean_draft_len"),
        "residual_divergence_count": report.get("residual_divergence_count"),
    }
    serial = OUT / tag / "serial-control.json"
    if serial.exists():
        try:
            metrics["serial_seconds_per_token"] = json.loads(
                serial.read_text()
            ).get("parent_measured_seconds_per_token")
        except json.JSONDecodeError:
            pass
    return metrics


def summarise(tag, warmup=1):
    meta = parse_meta(tag)
    rounds = parse_rounds(tag)
    kept = [r for r in rounds if r.get("round", 0) > warmup]
    buckets = {}
    for record in kept:
        key = (record.get("d"), record.get("acc"))
        buckets.setdefault(key, []).append(record)
    return {
        "tag": tag,
        "step_mode": meta.get("step_mode"),
        "repeat": meta.get("repeat", "1"),
        "tg_y": meta.get("tg_y"),
        "force_drafts": meta.get("force_drafts"),
        "tokens": meta.get("tokens"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "exit": meta.get("exit", meta.get("timed_exit")),
        "leg_path": meta.get("leg_path", "wrapper"),
        "rounds_total": len(rounds),
        "rounds_kept": len(kept),
        "score": score(tag),
        "buckets": buckets,
    }


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def least_squares(points):
    """Ordinary least squares slope, intercept and R^2 for (x, y) pairs."""
    n = len(points)
    mean_x = statistics.mean(x for x, _ in points)
    mean_y = statistics.mean(y for _, y in points)
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residual = sum((y - slope * x - intercept) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": 1.0 - residual / total if total else float("nan"),
        "n": n,
    }


def fit_payload(legs, bucket, modelled_step_us=8112.6):
    """Dose-response analysis over the token-exact repeat arms.

    The slope of round cost against the repeat count is the marginal cost of
    one recurrent step. `off` is reported beside it as the removal bracket, and
    it is labelled unmatched because it emits different tokens and so lands at
    different acceptance counts.
    """
    ladder = []
    points = []
    control = None
    for leg in legs:
        records = leg["buckets"].get(bucket, [])
        values = [r["round_us"] for r in records if "round_us" in r]
        if not values:
            continue
        mean = statistics.mean(values)
        ladder.append([
            leg["tag"], leg["step_mode"], int(leg["repeat"]), len(values),
            mean, statistics.stdev(values) if len(values) > 1 else 0.0,
            leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            bool(leg["score"].get("all_tokens_matched")),
        ])
        if leg["step_mode"] == "rep":
            points.append((int(leg["repeat"]), mean))
        if leg["step_mode"] == "clone":
            control = mean if control is None else (control + mean) / 2

    fit = least_squares(points) if len(points) > 2 else None

    # The removal arm never shares a bucket with the control, so its rounds are
    # pooled over every acceptance count at the control's draft width. The
    # measured acceptance insensitivity at fixed width is what makes that
    # pooling legible, and it is reported as a bracket, never as the headline.
    removal = []
    for leg in legs:
        if leg["step_mode"] != "off":
            continue
        for (d, _acc), records in leg["buckets"].items():
            if d != bucket[0]:
                continue
            removal.extend(r["round_us"] for r in records if "round_us" in r)

    metrics = {
        "bucket_d": bucket[0],
        "bucket_acc": bucket[1],
        "control_round_us": control,
        "modelled_step_us_per_round": modelled_step_us,
    }
    if fit:
        metrics.update({
            "measured_step_us_per_round": fit["slope"],
            "fit_intercept_us": fit["intercept"],
            "fit_r_squared": fit["r_squared"],
            "fit_points": fit["n"],
        })
        if control:
            metrics["measured_step_round_share_pct"] = (
                100.0 * fit["slope"] / control
            )
        metrics["model_overstatement_factor"] = modelled_step_us / fit["slope"]
    if removal and control:
        metrics.update({
            "removal_round_us_unmatched": statistics.mean(removal),
            "removal_rounds_n": len(removal),
            "control_minus_removal_us_unmatched": control
            - statistics.mean(removal),
        })
    return {
        "metrics": metrics,
        "tables": {
            "arm_ladder": {
                "columns": [
                    "tag", "step_mode", "repeat", "rounds", "round_us_mean",
                    "round_us_sd", "gpu_temp_entry_c", "gpu_temp_exit_c",
                    "tokens_matched_own_reference",
                ],
                "rows": ladder,
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--fit-json")
    parser.add_argument("--bucket", default="4,4")
    args = parser.parse_args()
    tags = args.tags
    bucket = tuple(int(part) for part in args.bucket.split(","))
    legs = [summarise(tag) for tag in tags]
    for leg in legs:
        metrics = leg["score"]
        print(
            f"{leg['tag']:<22} step={leg['step_mode']:<7} R={leg['repeat']:<3}"
            f" y={leg['tg_y']:<3}"
            f" d={leg['force_drafts']:<3} rounds={leg['rounds_kept']:<4}"
            f" exit={leg['exit']}"
            f" mtp_spt={metrics.get('mtp_seconds_per_token')}"
            f" serial_spt={metrics.get('serial_seconds_per_token')}"
            f" matched={metrics.get('all_tokens_matched')}"
            f" acc_rate={metrics.get('accepted_draft_rate')}"
            f" temp={leg['gpu_temp_entry_c']}->{leg['gpu_temp_exit_c']}"
        )
        for key in sorted(leg["buckets"], key=lambda k: (k[0] or 0, k[1] or 0)):
            records = leg["buckets"][key]
            row = [f"  d={key[0]} acc={key[1]} n={len(records)}"]
            for field in FIELDS:
                summary = stats([r[field] for r in records if field in r])
                if summary:
                    row.append(f"{field}={summary['mean']:.0f}")
            print(" ".join(row))
    print()
    print("== arm aggregate over identical (d, acc) buckets ==")
    by_arm = {}
    for leg in legs:
        arm = f"{leg['step_mode']}R{leg['repeat']}/y{leg['tg_y']}"
        for key, records in leg["buckets"].items():
            by_arm.setdefault(arm, {}).setdefault(key, []).extend(records)
    shared = None
    for arm, buckets in by_arm.items():
        keys = set(buckets)
        shared = keys if shared is None else (shared & keys)
    for key in sorted(shared or [], key=lambda k: (k[0] or 0, k[1] or 0)):
        print(f"-- bucket d={key[0]} acc={key[1]}")
        reference = None
        for arm in by_arm:
            records = by_arm[arm][key]
            summary = {
                field: stats([r[field] for r in records if field in r])
                for field in FIELDS
            }
            mean = summary["round_us"]["mean"]
            if reference is None:
                reference = mean
            delta = mean - reference
            print(
                f"   {arm:<12} n={summary['round_us']['n']:<4}"
                f" round_us={mean:9.1f}"
                f" sd={summary['round_us']['stdev']:7.1f}"
                f" delta={delta:9.1f} ({100 * delta / reference:+.2f}%)"
                f" verify_build={summary['verify_build_us']['mean']:9.1f}"
                f" eval_wall={summary['eval_wall_us']['mean']:9.1f}"
            )

    payload = fit_payload(legs, bucket)
    print()
    print(f"== repeat dose-response in bucket d={bucket[0]} acc={bucket[1]} ==")
    for key, value in payload["metrics"].items():
        print(f"   {key:<38} {value}")
    if args.fit_json:
        Path(args.fit_json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"   wrote {args.fit_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
