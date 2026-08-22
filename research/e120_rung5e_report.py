#!/usr/bin/env python3
"""Report E120 rung 5e from one research/e120_rung5e_session.sh output tree.

The headline is ABSOLUTE candidate seconds per token on the native-MTP leg,
`parent_measured_seconds_per_token`, the trusted parent's own clock. The local
serial-to-MTP ratio is reported as a control only: a wide-QMV change speeds
both local legs and partly cancels there.

Also produced, because the advisor asked for each by name in E120 F6:

  * the realised verify-width histogram, taken from the timed legs themselves
    rather than from a prior experiment (harness defect 20);
  * the isolated-to-in-situ transfer ratio, predicted leg effect from the rung
    5d grid divided by the measured leg effect;
  * the entry-temperature spread beside the effect;
  * a histogram-weighted ranked estimate with an interval.

    usage: research/e120_rung5e_report.py OUT_DIR [--gate-price PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

# E116 primary, `e116_wide_qmv_pct_to_leg_pct_transfer`, 12-leg OLS at
# R-squared 0.9972. W&B 7ex6rk98, 7juaip0i, sxypaucl.
WIDE_QMV_TO_LEG = 0.6070
WIDE_QMV_TO_LEG_CI = (0.5843, 0.6297)
# Campaign leg-to-ranked factor.
LEG_TO_RANKED = 0.95
# The shipped gate is `tablePays(m:) = m >= 4`, which is the gate-price table's
# `m4_and_volume_100k` column: the volume term never binds at any width the
# gate can see, so the two price identically.
SHIPPED_GATE = "m4_and_volume_100k"


def read_json(path: pathlib.Path):
    with path.open() as handle:
        return json.load(handle)


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def summarise_arm(reports: list[dict]) -> dict:
    spt = [r["parent_measured_seconds_per_token"] for r in reports]
    return {
        "legs": len(spt),
        "seconds_per_token": spt,
        "mean": statistics.fmean(spt),
        "median": statistics.median(spt),
        "sd": statistics.stdev(spt) if len(spt) > 1 else 0.0,
        "min": min(spt),
        "max": max(spt),
    }


def width_histogram(reports: list[dict]) -> dict[int, int]:
    """Verify width M per round. M is the primary token plus its drafts."""
    histogram: dict[int, int] = {}
    for report in reports:
        for drafts in report.get("effective_draft_lengths", []):
            width = int(drafts) + 1
            histogram[width] = histogram.get(width, 0) + 1
    return dict(sorted(histogram.items()))


def predict_leg_pct(histogram: dict[int, int], gate_price: dict) -> dict:
    """Time-weighted wide-QMV percentage over the realised width histogram.

    Widths the rung 5d grid did not measure carry no predicted saving and no
    predicted base, so they are reported separately rather than silently
    treated as zero-gain rounds.
    """
    saved_us = 0.0
    base_us = 0.0
    unmodelled = {}
    for width, rounds in histogram.items():
        cell = gate_price.get(str(width))
        if cell is None:
            unmodelled[width] = rounds
            continue
        saved_us += rounds * cell["gates"][SHIPPED_GATE]["net_us"]
        base_us += rounds * cell["base_us"]
    wide_qmv_pct = pct(saved_us, base_us)
    return {
        "wide_qmv_pct": wide_qmv_pct,
        "leg_pct": wide_qmv_pct * WIDE_QMV_TO_LEG,
        "leg_pct_interval": [
            wide_qmv_pct * WIDE_QMV_TO_LEG_CI[0],
            wide_qmv_pct * WIDE_QMV_TO_LEG_CI[1],
        ],
        "ranked_pct": wide_qmv_pct * WIDE_QMV_TO_LEG * LEG_TO_RANKED,
        "rounds_modelled": sum(
            n for w, n in histogram.items() if str(w) in gate_price
        ),
        "rounds_unmodelled": unmodelled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument(
        "--gate-price",
        type=pathlib.Path,
        default=pathlib.Path("research/out/e120-gate-price.json"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    meta = read_meta(out_dir / "meta.txt")
    legs = [
        json.loads(line)
        for line in (out_dir / "legs.jsonl").read_text().splitlines()
        if line.strip()
    ]

    by_arm: dict[str, list[dict]] = {}
    entry_temps: dict[str, list[float]] = {}
    exit_temps: dict[str, list[float]] = {}
    failures = []
    fidelity = []
    for leg in legs:
        if leg["status"] != 0:
            failures.append(leg["label"])
            continue
        report = read_json(pathlib.Path(leg["report"]))
        by_arm.setdefault(leg["arm"], []).append(report)
        if leg["gpu_temp_entry_c"] is not None:
            entry_temps.setdefault(leg["arm"], []).append(leg["gpu_temp_entry_c"])
        if leg["gpu_temp_exit_c"] is not None:
            exit_temps.setdefault(leg["arm"], []).append(leg["gpu_temp_exit_c"])
        fidelity.append(
            {
                "label": leg["label"],
                "arm": leg["arm"],
                "all_tokens_matched": report.get("all_tokens_matched"),
                "residual_divergence_count": report.get("residual_divergence_count"),
                "decode_token_count": report.get("decode_token_count"),
                "seconds_per_token": report["parent_measured_seconds_per_token"],
                "effective_mean_draft_len": report.get("effective_mean_draft_len"),
                "accepted_draft_rate": report.get("accepted_draft_rate"),
            }
        )

    arms = {arm: summarise_arm(reports) for arm, reports in by_arm.items()}
    exactness_ok = all(
        f["all_tokens_matched"] is True and f["residual_divergence_count"] == 0
        for f in fidelity
    )

    result: dict = {
        "out_dir": str(out_dir),
        "meta": meta,
        "harness": "local",
        "instrument": "mtp-timed",
        "headline_metric": "absolute candidate seconds per token, native-MTP leg",
        "failed_legs": failures,
        "arms": arms,
        "fidelity": fidelity,
        "exactness_ok": exactness_ok,
        "gpu_temp_entry_c": {
            arm: {
                "mean": statistics.fmean(v),
                "min": min(v),
                "max": max(v),
                "values": v,
            }
            for arm, v in entry_temps.items()
        },
        "gpu_temp_exit_c": {
            arm: {"mean": statistics.fmean(v), "values": v}
            for arm, v in exit_temps.items()
        },
    }

    # The pre-registered routing-gate null control: one serial leg per arm.
    # Serial decode is M=1 and `routable` requires m in 3...9, so the arm can
    # never reach it. The two serial legs must agree within 0.5 %.
    serial_by_arm = {}
    for serial_path in sorted(out_dir.glob("serial-control.*.json")):
        arm = serial_path.name[len("serial-control.") : -len(".json")]
        serial = read_json(serial_path)
        serial_by_arm[arm] = {
            "seconds_per_token": serial["parent_measured_seconds_per_token"],
            "all_tokens_matched": serial.get("all_tokens_matched"),
            "decode_token_count": serial.get("decode_token_count"),
        }
    if serial_by_arm:
        spts = [v["seconds_per_token"] for v in serial_by_arm.values()]
        spread = pct(max(spts) - min(spts), statistics.fmean(spts))
        result["serial_control"] = {
            "by_arm": serial_by_arm,
            "mean_seconds_per_token": statistics.fmean(spts),
            "spread_pct": spread,
            "arm_independent": spread <= 0.5,
            "note": (
                "pre-registered null control: the arm cannot reach serial "
                "decode, which runs at M=1 below the routing gate's m>=3"
            ),
        }

    # Every timed leg must have run the same worker bytes.
    worker_hashes = {
        h
        for leg in legs
        for h in (leg.get("worker_sha256_before"), leg.get("worker_sha256_after"))
        if h
    }
    result["worker_assertion"] = {
        "distinct_sha256_across_legs": sorted(worker_hashes),
        "one_binary_for_every_leg": len(worker_hashes) == 1,
        "per_leg_unchanged": all(
            leg.get("worker_unchanged", True) for leg in legs
        ),
    }

    if "off" in arms and "sumtable" in arms:
        base = arms["off"]["median"]
        cand = arms["sumtable"]["median"]
        measured_leg_pct = pct(base - cand, base)
        result["effect"] = {
            "baseline_arm": "off",
            "candidate_arm": "sumtable",
            "baseline_seconds_per_token": base,
            "candidate_seconds_per_token": cand,
            "delta_seconds_per_token": cand - base,
            "measured_leg_pct": measured_leg_pct,
            "ranked_pct": measured_leg_pct * LEG_TO_RANKED,
            "entry_temp_spread_c": (
                statistics.fmean(entry_temps["sumtable"])
                - statistics.fmean(entry_temps["off"])
                if entry_temps.get("sumtable") and entry_temps.get("off")
                else None
            ),
        }
        if serial_by_arm:
            # Each arm's own serial leg is its own denominator, which is what
            # `benchmark-qwen-mtp.sh` would have reported for that arm.
            result["effect"]["local_ratio_control"] = {
                arm: serial_by_arm[arm]["seconds_per_token"] / arms[arm]["median"]
                for arm in ("off", "sumtable")
                if arm in serial_by_arm
            }
            result["effect"]["local_ratio_control"]["note"] = (
                "local serial-to-MTP ratio, control only; a wide-QMV change "
                "speeds both local legs and partly cancels here"
            )
        # CAMPAIGN RULE 39 sizing, from this session's own dispersion.
        pooled = [
            v for arm in ("off", "sumtable") for v in arms[arm]["seconds_per_token"]
        ]
        pooled_mean = statistics.fmean(pooled)
        within = [
            v - arms[arm]["mean"]
            for arm in ("off", "sumtable")
            for v in arms[arm]["seconds_per_token"]
        ]
        within_sd = statistics.stdev(within) if len(within) > 2 else 0.0
        legs_per_arm = min(arms["off"]["legs"], arms["sumtable"]["legs"])
        result["dispersion"] = {
            "pooled_mean_seconds_per_token": pooled_mean,
            "within_arm_sd_seconds_per_token": within_sd,
            "within_arm_cv_pct": pct(within_sd, pooled_mean),
            "legs_per_arm": legs_per_arm,
            "two_sigma_resolvable_pct": (
                2.0 * pct(within_sd, pooled_mean) * math.sqrt(2.0 / legs_per_arm)
                if legs_per_arm
                else None
            ),
        }

    all_reports = [r for reports in by_arm.values() for r in reports]
    histogram = width_histogram(by_arm.get("sumtable") or all_reports)
    total_rounds = sum(histogram.values())
    result["realised_width_histogram"] = {
        "source": "timed legs of this session, effective_draft_lengths + 1",
        "counts": histogram,
        "fraction": (
            {w: n / total_rounds for w, n in histogram.items()} if total_rounds else {}
        ),
        "mean_width": (
            sum(w * n for w, n in histogram.items()) / total_rounds
            if total_rounds
            else None
        ),
        "total_rounds": total_rounds,
    }

    if args.gate_price.exists() and histogram:
        gate_price = read_json(args.gate_price)
        predicted = predict_leg_pct(histogram, gate_price)
        result["predicted_from_rung5d"] = predicted
        if "effect" in result:
            measured = result["effect"]["measured_leg_pct"]
            result["isolated_to_in_situ_transfer_ratio"] = (
                predicted["leg_pct"] / measured if measured else None
            )
            result["effect"]["ranked_pct_histogram_weighted_prediction"] = predicted[
                "ranked_pct"
            ]

    out_path = out_dir / "rung5e_report.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"legs                {[l['label'] for l in legs]}")
    print(f"failed legs         {failures or 'none'}")
    for arm, summary in sorted(arms.items()):
        print(
            f"arm {arm:<10} n={summary['legs']} "
            f"median={summary['median']:.6f} s/tok "
            f"mean={summary['mean']:.6f} sd={summary['sd']:.6f}"
        )
    if "serial_control" in result:
        serial = result["serial_control"]
        detail = " ".join(
            f"{arm}={v['seconds_per_token']:.6f}"
            for arm, v in sorted(serial["by_arm"].items())
        )
        print(
            f"serial control      {detail} s/tok  spread "
            f"{serial['spread_pct']:.3f} % "
            f"(arm_independent={serial['arm_independent']})"
        )
    worker = result["worker_assertion"]
    print(
        f"worker assertion    one_binary={worker['one_binary_for_every_leg']} "
        f"per_leg_unchanged={worker['per_leg_unchanged']} "
        f"{worker['distinct_sha256_across_legs']}"
    )
    if "effect" in result:
        effect = result["effect"]
        print(
            f"HEADLINE            absolute candidate MTP s/tok "
            f"{effect['candidate_seconds_per_token']:.6f} vs "
            f"{effect['baseline_seconds_per_token']:.6f} = "
            f"{effect['measured_leg_pct']:+.3f} % leg, "
            f"{effect['ranked_pct']:+.3f} % ranked"
        )
        print(f"entry temp spread   {effect['entry_temp_spread_c']} C (sumtable - off)")
    if "dispersion" in result:
        print(
            f"within-arm CV       {result['dispersion']['within_arm_cv_pct']:.3f} % "
            f"=> 2 sigma resolvable "
            f"{result['dispersion']['two_sigma_resolvable_pct']:.3f} %"
        )
    print(f"width histogram     {result['realised_width_histogram']['counts']}")
    print(f"mean width          {result['realised_width_histogram']['mean_width']}")
    if "predicted_from_rung5d" in result:
        predicted = result["predicted_from_rung5d"]
        print(
            f"predicted leg       {predicted['leg_pct']:+.3f} % "
            f"(wide-QMV {predicted['wide_qmv_pct']:+.3f} %)"
        )
        ratio = result.get("isolated_to_in_situ_transfer_ratio")
        if ratio is not None:
            print(f"transfer ratio      {ratio:.3f} (predicted / measured)")
    print(f"exactness ok        {exactness_ok}")
    print(f"wrote               {out_path}")
    # Everything is printed and written before the exit code is decided, so a
    # failing session still yields its evidence.
    return (
        0
        if exactness_ok
        and not failures
        and worker["one_binary_for_every_leg"]
        and worker["per_leg_unchanged"]
        and result.get("serial_control", {}).get("arm_independent", True)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
