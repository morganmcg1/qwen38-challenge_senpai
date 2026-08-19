#!/usr/bin/env python3
"""Research-only (qwen38-r1-e30): adjudicate the pre-registered E30 predictions.

E29 attributed the per-round host tail with a six-way trace and left two things
open: an 8.0-8.5 % "all rounds at M=8" full-accept bound, and the question of
whether `verify_build_us` is host graph construction or host time spent blocked
on GPU backpressure. E27 then changed ONLY the M=5 and M=9 crossrow QMV dispatch
(`<T,9,3>` -> `<T,9,5>`, three weight streams -> two), leaving M=8 at
`<T,8,4>`. That is a one-sided natural experiment: the M=9 rounds get a ~21 ms
GPU-side saving, the M=8 rounds in the SAME arm get nothing, so M=8 is an
interleaved internal control for session-level drift.

This script does not re-measure. It consumes the unmodified `e29_analyze.py`
trace parse plus the trusted parent's own per-round `block_request_seconds`, and
reports both sides of every number so a trace artefact cannot masquerade as a
parent-visible effect. Predictions and thresholds are those committed in
`research/results/e30-preregistration.md` before the first E30 run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e29_analyze import SEGMENTS, Round, parse_trace  # noqa: E402

# Registered in research/results/e30-preregistration.md at 3ec7cbe.
PREREG = {
    "p1_advisor_threshold_pct": 2.0,
    "p1_student_point_pct": 2.58,
    "p1_student_range_pct": [1.8, 3.6],
    "p1_baseline_pct": 8.25,
    "p2_advisor_threshold_ms": -5.0,
    "p2_student_point_ms": -10.5,
    "p2_student_range_ms": [-16.0, -4.0],
    "p2_baseline_vbuild_m9_ms": 101.720,
    "p2_predicted_round_m9_ms": 185.28,
    "control_m8_round_ms": 167.675,
    "control_m8_tolerance_pct": 3.0,
    "control_serial_ms_per_token": 66.8745,
    "control_serial_tolerance_pct": 2.0,
    "control_histogram": {"2": 1, "6": 9, "7": 3, "8": 5, "9": 15},
    "control_accepted_draft_total": 222,
}
TREATED_WIDTH = 9
CONTROL_WIDTH = 8


def mean_sd(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def ols_slope(values: list[float]) -> float | None:
    """Least-squares slope of value against its position in the arm.

    Thermal or power drift inside one ungated arm shows up here; a width whose
    cost is flat in sequence position is not being carried by drift.
    """
    n = len(values)
    if n < 3:
        return None
    xs = list(range(n))
    mx = statistics.fmean(xs)
    my = statistics.fmean(values)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def load_meta(run_dir: Path) -> dict[str, str]:
    meta = run_dir / "meta.txt"
    if not meta.is_file():
        return {}
    return dict(
        line.split("=", 1) for line in meta.read_text().splitlines()
        if "=" in line)


def load_reports(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("03-mtp-timed.json", "04-mtp-timed.json"):
        path = run_dir / "reports" / name
        if path.is_file():
            out[name] = json.loads(path.read_text())
    return out


def legs(reports: dict[str, dict]) -> dict[str, object]:
    """Serial and MTP legs as TWO numbers on true decode.

    `decode_seconds` is prefill-inclusive (E29), so the denominator is the sum
    of the parent's own per-round `block_request_seconds`, which equals
    `decode_seconds - seed_prefill_seconds` to within a millisecond.
    """
    out: dict[str, object] = {}
    for name, label in (("03-mtp-timed.json", "serial"),
                        ("04-mtp-timed.json", "mtp")):
        doc = reports.get(name)
        if not doc:
            continue
        blocks = doc.get("block_request_seconds") or []
        tokens = doc.get("decode_token_count")
        true_decode_s = sum(blocks)
        out[label] = {
            "report": name,
            "rounds": len(blocks),
            "decode_tokens": tokens,
            "true_decode_s": true_decode_s,
            "true_decode_ms_per_token": (
                1e3 * true_decode_s / tokens if tokens else None),
            "ms_per_round": 1e3 * true_decode_s / len(blocks) if blocks else None,
            "decode_seconds_prefill_inclusive": doc.get("decode_seconds"),
            "seed_prefill_seconds": doc.get("seed_prefill_seconds"),
            "prefill_share_of_decode_seconds_pct": (
                100.0 * doc["seed_prefill_seconds"] / doc["decode_seconds"]
                if doc.get("decode_seconds") else None),
            "all_tokens_matched": doc.get("all_tokens_matched"),
            "residual_divergence_count": doc.get("residual_divergence_count"),
            "parity_all_ok": doc.get("parity_all_ok"),
            "declared_rows_total": doc.get("declared_rows_total"),
            "reference_checked_row_total": doc.get("reference_checked_row_total"),
            "accepted_draft_total": doc.get("accepted_draft_total"),
            "rejected_draft_total": doc.get("rejected_draft_total"),
            "accepted_draft_rate": doc.get("accepted_draft_rate"),
            "effective_mean_draft_len": doc.get("effective_mean_draft_len"),
        }
    serial = out.get("serial", {}).get("true_decode_ms_per_token")
    mtp = out.get("mtp", {}).get("true_decode_ms_per_token")
    out["true_decode_serial_over_mtp"] = serial / mtp if serial and mtp else None
    return out


def parent_per_width(reports: dict[str, dict],
                     skip_warmup: int) -> dict[str, object] | None:
    """Per-width round cost from the TRUSTED PARENT's own timing.

    `effective_draft_lengths[i] + 1` is the width of round i and
    `block_request_seconds[i]` is the parent-observed cost of that same round,
    so this reproduces the trace's per-width table without using the trace.
    """
    doc = reports.get("04-mtp-timed.json")
    if not doc:
        return None
    depths = doc.get("effective_draft_lengths") or []
    blocks = doc.get("block_request_seconds") or []
    if not depths or len(depths) != len(blocks):
        return None
    rows = list(zip(depths[skip_warmup:], blocks[skip_warmup:]))
    by_width: dict[int, list[float]] = {}
    for depth, seconds in rows:
        by_width.setdefault(depth + 1, []).append(1e3 * seconds)
    per_width = {
        str(m): {
            **mean_sd(v),
            "full_accept_ms_per_token": statistics.fmean(v) / m,
            "sequence_slope_ms_per_round": ols_slope(v),
        }
        for m, v in sorted(by_width.items())
    }
    return {
        "skip_warmup": skip_warmup,
        "rounds": len(rows),
        "histogram": {str(m): len(v) for m, v in sorted(by_width.items())},
        "round_total_ms": sum(ms for _, v in by_width.items() for ms in v),
        "full_accept_tokens": sum(m * len(v) for m, v in by_width.items()),
        "per_width": per_width,
        **bounds(
            {m: statistics.fmean(v) for m, v in by_width.items()},
            {m: len(v) for m, v in by_width.items()}),
    }


def bounds(round_mean_ms: dict[int, float],
           counts: dict[int, int]) -> dict[str, object]:
    """The all-at-M=8 bound and the best-width bound, kept distinct.

    `bound_M8` is always evaluated at M=8 even once M=8 stops being the
    cheapest width, because that is the pre-registered primary metric. It goes
    negative exactly when forcing every round to M=8 would COST time.
    """
    total = sum(round_mean_ms[m] * counts[m] for m in counts)
    tokens = sum(m * counts[m] for m in counts)
    per_token = {m: round_mean_ms[m] / m for m in counts}
    best = min(per_token, key=lambda m: per_token[m])
    out: dict[str, object] = {
        "steady_round_total_ms": total,
        "full_accept_tokens": tokens,
        "realised_ms_per_token": total / tokens if tokens else None,
        "best_width": best,
        "best_ms_per_token": per_token[best],
        "best_width_headroom_pct": (
            100.0 * (total - tokens * per_token[best]) / total if total else None),
        "per_width_ms_per_token": {str(m): per_token[m] for m in sorted(per_token)},
    }
    if CONTROL_WIDTH in per_token:
        all_at_m8 = tokens * per_token[CONTROL_WIDTH]
        out["all_at_m8_total_ms"] = all_at_m8
        out["all_at_m8_upper_bound_pct"] = (
            100.0 * (total - all_at_m8) / total if total else None)
    if CONTROL_WIDTH in round_mean_ms and TREATED_WIDTH in round_mean_ms:
        step = round_mean_ms[TREATED_WIDTH] / round_mean_ms[CONTROL_WIDTH]
        out["m8_to_m9_round_step"] = step
        out["m8_to_m9_row_step"] = TREATED_WIDTH / CONTROL_WIDTH
        out["m9_per_row_vs_m8_pct"] = 100.0 * (
            per_token[TREATED_WIDTH] / per_token[CONTROL_WIDTH] - 1.0)
    return out


def trace_arm(run_dir: Path, skip_warmup: int) -> dict[str, object]:
    rounds = parse_trace(run_dir / "rounds.trace")
    last = max(r.session for r in rounds)
    timed = [r for r in rounds if r.session == last]
    steady = timed[max(skip_warmup, 0):]
    by_width: dict[int, list[Round]] = {}
    for r in steady:
        by_width.setdefault(r.width, []).append(r)

    per_width: dict[str, object] = {}
    for m, rs in sorted(by_width.items()):
        round_ms = [r.round_us / 1e3 for r in rs]
        entry: dict[str, object] = {
            "rounds": len(rs),
            "round_ms": mean_sd(round_ms),
            "round_sequence_slope_ms_per_round": ols_slope(round_ms),
            "full_accept_ms_per_token": statistics.fmean(round_ms) / m,
            "host_tail_ms": mean_sd([r.host_tail_us / 1e3 for r in rs]),
        }
        for seg in SEGMENTS:
            entry[seg] = mean_sd([r.seg[seg] / 1e3 for r in rs])
        per_width[str(m)] = entry

    round_mean = {m: statistics.fmean([r.round_us / 1e3 for r in rs])
                  for m, rs in by_width.items()}
    counts = {m: len(rs) for m, rs in by_width.items()}
    unaccounted = sum(
        r.round_us - sum(r.seg.values()) for r in timed)
    return {
        "timed_rounds": len(timed),
        "steady_rounds": len(steady),
        "skip_warmup": skip_warmup,
        "histogram": {str(m): counts[m] for m in sorted(counts)},
        "accepted_draft_total": sum(r.accepted for r in timed),
        "unaccounted_us": unaccounted,
        "per_width": per_width,
        **bounds(round_mean, counts),
    }


def verdict(name: str, value: float | None, threshold: float,
            direction: str, point: float,
            interval: list[float]) -> dict[str, object]:
    if value is None:
        return {"prediction": name, "status": "unmeasured"}
    passed = value < threshold if direction == "below" else value > threshold
    return {
        "prediction": name,
        "measured": value,
        "threshold": threshold,
        "direction": direction,
        "threshold_met": passed,
        "student_point": point,
        "student_point_error": value - point,
        "student_interval": interval,
        "inside_student_interval": interval[0] <= value <= interval[1],
    }


def adjudicate(base: dict, cand: dict) -> dict[str, object]:
    """Pre-registered verdicts, plus the M=8 / serial / histogram controls."""
    def w(arm: dict, m: int, key: str) -> float | None:
        entry = arm["trace"]["per_width"].get(str(m))
        if not entry:
            return None
        return entry[key]["mean"] if isinstance(entry[key], dict) else entry[key]

    vb_base = w(base, TREATED_WIDTH, "verify_build_us")
    vb_cand = w(cand, TREATED_WIDTH, "verify_build_us")
    d_vbuild = vb_cand - vb_base if None not in (vb_base, vb_cand) else None

    r9_base = w(base, TREATED_WIDTH, "round_ms")
    r9_cand = w(cand, TREATED_WIDTH, "round_ms")
    d_round9 = r9_cand - r9_base if None not in (r9_base, r9_cand) else None

    r8_base = w(base, CONTROL_WIDTH, "round_ms")
    r8_cand = w(cand, CONTROL_WIDTH, "round_ms")

    ev9_base = w(base, TREATED_WIDTH, "eval_wall_us")
    ev9_cand = w(cand, TREATED_WIDTH, "eval_wall_us")

    sd9 = (cand["trace"]["per_width"].get(str(TREATED_WIDTH), {})
           .get("verify_build_us", {}).get("sd"))
    flat_band = 2.0 * sd9 if sd9 else None

    serial_base = base["legs"]["serial"]["true_decode_ms_per_token"]
    serial_cand = cand["legs"]["serial"]["true_decode_ms_per_token"]
    mtp_base = base["legs"]["mtp"]["true_decode_ms_per_token"]
    mtp_cand = cand["legs"]["mtp"]["true_decode_ms_per_token"]

    bound_cand = cand["trace"]["all_at_m8_upper_bound_pct"]
    parent_bound = (cand["parent"] or {}).get("all_at_m8_upper_bound_pct")

    out: dict[str, object] = {
        "p1_bound_M8": {
            **verdict("bound_M8 < 2.0", bound_cand,
                      PREREG["p1_advisor_threshold_pct"], "below",
                      PREREG["p1_student_point_pct"],
                      PREREG["p1_student_range_pct"]),
            "baseline_pct": PREREG["p1_baseline_pct"],
            "e29_measured_pct": base["trace"]["all_at_m8_upper_bound_pct"],
            "parent_side_pct": parent_bound,
            "best_width_now": cand["trace"]["best_width"],
            "best_width_headroom_pct": cand["trace"]["best_width_headroom_pct"],
            "m8_to_m9_round_step_e29": base["trace"].get("m8_to_m9_round_step"),
            "m8_to_m9_round_step_e30": cand["trace"].get("m8_to_m9_round_step"),
            "m9_per_row_vs_m8_pct_e29": base["trace"].get("m9_per_row_vs_m8_pct"),
            "m9_per_row_vs_m8_pct_e30": cand["trace"].get("m9_per_row_vs_m8_pct"),
        },
        "p2_vbuild_m9": {
            **verdict("delta vbuild(M=9) <= -5 ms", d_vbuild,
                      PREREG["p2_advisor_threshold_ms"], "below",
                      PREREG["p2_student_point_ms"],
                      PREREG["p2_student_range_ms"]),
            "e29_ms": vb_base,
            "e30_ms": vb_cand,
            "e30_sd_ms": sd9,
            "flat_band_2sd_ms": flat_band,
            "is_flat": abs(d_vbuild) < flat_band
            if None not in (d_vbuild, flat_band) else None,
            "delta_eval_wall_m9_ms": (
                ev9_cand - ev9_base if None not in (ev9_base, ev9_cand) else None),
            "delta_round_m9_ms": d_round9,
            "share_of_round_delta_in_vbuild_pct": (
                100.0 * d_vbuild / d_round9
                if None not in (d_vbuild, d_round9) and d_round9 else None),
        },
        "controls": {
            "m8_round_ms_e29": r8_base,
            "m8_round_ms_e30": r8_cand,
            "m8_delta_pct": (100.0 * (r8_cand / r8_base - 1.0)
                             if None not in (r8_base, r8_cand) else None),
            "m8_within_tolerance": (
                abs(100.0 * (r8_cand / r8_base - 1.0))
                <= PREREG["control_m8_tolerance_pct"]
                if None not in (r8_base, r8_cand) else None),
            "serial_ms_per_token_e29": serial_base,
            "serial_ms_per_token_e30": serial_cand,
            "serial_delta_pct": 100.0 * (serial_cand / serial_base - 1.0),
            "serial_within_tolerance": (
                abs(100.0 * (serial_cand / serial_base - 1.0))
                <= PREREG["control_serial_tolerance_pct"]),
            "mtp_ms_per_token_e29": mtp_base,
            "mtp_ms_per_token_e30": mtp_cand,
            "mtp_delta_pct": 100.0 * (mtp_cand / mtp_base - 1.0),
            "histogram_e29": base["trace"]["histogram"],
            "histogram_e30": cand["trace"]["histogram"],
            "histogram_unchanged":
                cand["trace"]["histogram"] == base["trace"]["histogram"],
            "accepted_draft_total_e29": base["trace"]["accepted_draft_total"],
            "accepted_draft_total_e30": cand["trace"]["accepted_draft_total"],
            "exactness_e30": {
                "all_tokens_matched": cand["legs"]["mtp"]["all_tokens_matched"],
                "residual_divergence_count":
                    cand["legs"]["mtp"]["residual_divergence_count"],
                "parity_all_ok": cand["legs"]["mtp"]["parity_all_ok"],
                "declared_rows_total": cand["legs"]["mtp"]["declared_rows_total"],
                "reference_checked_row_total":
                    cand["legs"]["mtp"]["reference_checked_row_total"],
            },
        },
    }
    return out


def arm(run_dir: Path, skip_warmup: int) -> dict[str, object]:
    reports = load_reports(run_dir)
    meta = load_meta(run_dir)
    return {
        "run_dir": str(run_dir),
        "meta": meta,
        "legs": legs(reports),
        "trace": trace_arm(run_dir, skip_warmup),
        "parent": parent_per_width(reports, skip_warmup),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True,
                    help="E29 D0 run dir (pre-NA=5 tree)")
    ap.add_argument("--candidate", type=Path, action="append", required=True,
                    help="E30 run dir; first is the primary comparison")
    ap.add_argument("--skip-warmup", type=int, default=2)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    doc: dict[str, object] = {
        "prereg": PREREG,
        "baseline": arm(args.baseline, args.skip_warmup),
        "candidates": {c.name: arm(c, args.skip_warmup) for c in args.candidate},
    }
    primary = doc["candidates"][args.candidate[0].name]
    doc["adjudication"] = adjudicate(doc["baseline"], primary)

    if len(args.candidate) > 1:
        repeats = list(doc["candidates"].values())
        doc["repeat_dispersion"] = {
            "arms": [a["run_dir"] for a in repeats],
            "serial_ms_per_token": mean_sd(
                [a["legs"]["serial"]["true_decode_ms_per_token"] for a in repeats]),
            "mtp_ms_per_token": mean_sd(
                [a["legs"]["mtp"]["true_decode_ms_per_token"] for a in repeats]),
            "per_width_round_mean_ms": {
                str(m): mean_sd([
                    a["trace"]["per_width"][str(m)]["round_ms"]["mean"]
                    for a in repeats if str(m) in a["trace"]["per_width"]])
                for m in (2, 6, 7, 8, 9)
            },
            "per_width_vbuild_mean_ms": {
                str(m): mean_sd([
                    a["trace"]["per_width"][str(m)]["verify_build_us"]["mean"]
                    for a in repeats if str(m) in a["trace"]["per_width"]])
                for m in (2, 6, 7, 8, 9)
            },
            "bound_M8_pct": mean_sd(
                [a["trace"]["all_at_m8_upper_bound_pct"] for a in repeats]),
        }

    if args.json_out:
        args.json_out.write_text(json.dumps(doc, indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")

    adj = doc["adjudication"]
    print("\n=== legs (true decode, prefill excluded) ===")
    for label, a in [("E29 D0", doc["baseline"])] + [
            (n, v) for n, v in doc["candidates"].items()]:
        lg = a["legs"]
        print(f"  {label:8s} serial {lg['serial']['true_decode_ms_per_token']:8.4f}"
              f" ms/tok   MTP {lg['mtp']['true_decode_ms_per_token']:8.4f} ms/tok"
              f"  ({lg['mtp']['ms_per_round']:8.4f} ms/round)"
              f"  ratio {lg['true_decode_serial_over_mtp']:.4f}")

    print("\n=== per width (trace, steady state) ===")
    for label, a in [("E29 D0", doc["baseline"])] + [
            (n, v) for n, v in doc["candidates"].items()]:
        print(f"  -- {label}: histogram {a['trace']['histogram']}"
              f" bound_M8 {a['trace']['all_at_m8_upper_bound_pct']:.4f}%"
              f" best M={a['trace']['best_width']}")
        for m, e in a["trace"]["per_width"].items():
            print(f"     M={m:>2s} n={e['rounds']:<3d}"
                  f" round {e['round_ms']['mean']:9.3f}"
                  f" (sd {e['round_ms']['sd'] or float('nan'):7.3f})"
                  f" vbuild {e['verify_build_us']['mean']:8.3f}"
                  f" eval {e['eval_wall_us']['mean']:9.3f}"
                  f" dbuild {e['draft_build_us']['mean']:7.3f}"
                  f" ms/tok {e['full_accept_ms_per_token']:7.3f}")

    print("\n=== adjudication ===")
    print(json.dumps(adj, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
