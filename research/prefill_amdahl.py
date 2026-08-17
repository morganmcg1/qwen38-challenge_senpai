#!/usr/bin/env python3
"""Decompose a local Qwen-MTP benchmark run into seed-prefill and decode work.

The trusted parent starts its clock immediately before `beginMTPDecode` and
stops it after the last round (QwenRuntimeMTPDriver.swift:93-194), and it also
reports every per-round latency it measured itself. So for one timed leg

    decode_seconds = P + sum(block_request_seconds) + N * c

where P is the charged seed prefill (`begin`), N is the round count and c is the
parent-side per-round bookkeeping that falls outside its own round stopwatch.
The serial control and the MTP leg share the same seed and the same `begin`, but
run very different round counts, so the pair solves for both P and c.

Since main.swift:2015-2027 the CLI also reports `seed_prefill_seconds` directly,
measured around the same `beginMTPDecode` call the clock already contains. The
`direct` block uses that number instead of inferring it, and cross-checks it
against the residual decomposition above.

Usage:
    research/prefill_amdahl.py CAPTURE_DIR [--wandb] [--tag NAME] ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

RANKED_WINDOW = 512

# Highest promoted ranked median at the time of e12 (frontier-state.json).
RANKED_SPEEDUP = 2.94661597308114


def sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def load_timed_reports(capture_dir: Path) -> dict[str, dict]:
    """Return {'serial': report, 'mtp': report} from a capture directory."""
    legs: dict[str, dict] = {}
    for path in sorted(capture_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "decode_seconds" not in payload:
            continue
        if "block_request_seconds" not in payload:
            continue
        payload["_source_file"] = path.name
        leg = "serial" if payload.get("mtp_depth", 0) == 0 else "mtp"
        legs[leg] = payload
    return legs


def leg_stats(report: dict) -> dict:
    blocks = [float(x) for x in report["block_request_seconds"]]
    decode_seconds = float(report["decode_seconds"])
    emitted = int(report["emitted_token_total"])
    rounds = len(blocks)
    first = blocks[0]
    rest = blocks[1:]
    tokens_per_round = emitted / rounds
    steady_tokens = emitted - tokens_per_round
    return {
        "source_file": report["_source_file"],
        "mtp_depth": int(report.get("mtp_depth", 0)),
        "decode_seconds": decode_seconds,
        "block_seconds_total": sum(blocks),
        "residual_seconds": decode_seconds - sum(blocks),
        "round_count": rounds,
        "emitted_token_total": emitted,
        "seed_token_count": int(report.get("seed_token_count", 0)),
        # Absent keys mean the report came from a binary older than the
        # instrumentation commit, not a zero-cost prefill.
        "has_direct_prefill_fields": "seed_prefill_seconds" in report,
        "seed_prefill_seconds": float(report.get("seed_prefill_seconds", float("nan"))),
        "prefill_seconds_per_seed_token": float(
            report.get("prefill_seconds_per_token", float("nan"))
        ),
        "first_block_seconds": first,
        "mean_block_after_first": statistics.fmean(rest) if rest else float("nan"),
        "p50_block_after_first": statistics.median(rest) if rest else float("nan"),
        "max_block_after_first": max(rest) if rest else float("nan"),
        "tokens_per_round": tokens_per_round,
        "steady_seconds_per_token": (sum(rest) / steady_tokens) if rest else float("nan"),
        "parent_measured_seconds_per_token": float(report["parent_measured_seconds_per_token"]),
        "accepted_draft_rate": float(report.get("accepted_draft_rate", 0.0)),
        "effective_mean_draft_len": float(report.get("effective_mean_draft_len", 0.0) or 0.0),
        "residual_divergence_count": int(report.get("residual_divergence_count", 0)),
        "all_tokens_matched": bool(report.get("all_tokens_matched", False)),
        "declared_rows_total": int(report.get("declared_rows_total", 0)),
        # Stall guardrail: fixtures/qwen3_8_27b_mtp_track.json rejects a run whose
        # max block latency exceeds 4x its p50. Report the parent's own fields and
        # the ratio, plus the ratio the recommended "exclude the first block" fix
        # would produce.
        "guardrail_max_block_seconds": float(report["max_block_request_seconds"]),
        "guardrail_p50_block_seconds": float(report["p50_block_request_seconds"]),
        "guardrail_max_over_p50": float(report["max_block_request_seconds"])
        / float(report["p50_block_request_seconds"]),
        "guardrail_max_over_p50_excluding_first": float(
            report["max_block_request_seconds_after_first"]
        )
        / float(report["p50_block_request_seconds_after_first"]),
        "guardrail_margin_to_4x": 4.0
        - float(report["max_block_request_seconds"]) / float(report["p50_block_request_seconds"]),
        "guardrail_max_is_first_block": max(blocks) == first,
    }


def direct_charge(serial: dict, mtp: dict, ranked_r: float, inferred: dict) -> dict:
    """Score the seed-prefill charge from the CLI's own `seed_prefill_seconds`.

    With decode_seconds = P + D per leg and p = P_mtp / D_mtp, the ideal
    prefill-free ratio and the measured one are related exactly by

        r_ideal - r = (P_mtp * r_ideal - P_serial) / (P_mtp + D_mtp)

    which collapses to p * (r - 1) when both legs are charged the same P. That
    collapsed form is what makes the charge transferable to the ranked window:
    it needs only p and the ranked ratio, not the ranked leg times.
    """
    for label, leg in (("serial", serial), ("mtp", mtp)):
        if not leg["has_direct_prefill_fields"]:
            raise SystemExit(
                f"prefill_amdahl: {label} report has no seed_prefill_seconds; "
                "rebuild the CLI at or after the instrumentation commit"
            )

    p_s = serial["seed_prefill_seconds"]
    p_m = mtp["seed_prefill_seconds"]
    d_s = serial["decode_seconds"] - p_s
    d_m = mtp["decode_seconds"] - p_m
    p = p_m / d_m

    r = serial["decode_seconds"] / mtp["decode_seconds"]
    r_ideal = d_s / d_m
    identity_rhs = p * (r - 1.0)
    exact_gap = (p_m * r_ideal - p_s) / (p_m + d_m)

    mean_p = 0.5 * (p_s + p_m)
    sym_pct = 100.0 * abs(p_s - p_m) / mean_p

    # Ranked-window p. The seed is 512 tokens in both cases, so P transfers, but
    # the decode work must be restated at 512 emitted tokens using each leg's
    # steady per-token cost.
    ranked_d_m = mtp["steady_seconds_per_token"] * RANKED_WINDOW
    ranked_d_s = serial["steady_seconds_per_token"] * RANKED_WINDOW
    p_ranked_local_host = p_m / ranked_d_m

    out = {
        "prefill_seconds_serial": p_s,
        "prefill_seconds_mtp": p_m,
        "prefill_seconds_per_seed_token_serial": serial["prefill_seconds_per_seed_token"],
        "prefill_seconds_per_seed_token_mtp": mtp["prefill_seconds_per_seed_token"],
        "seed_token_count_serial": serial["seed_token_count"],
        "seed_token_count_mtp": mtp["seed_token_count"],
        "decode_work_serial_seconds": d_s,
        "decode_work_mtp_seconds": d_m,
        "p_prefill_over_decode_work_mtp": p,
        "p_prefill_over_decode_work_serial": p_s / d_s,
        "prefill_fraction_of_mtp_leg": p_m / mtp["decode_seconds"],
        "prefill_fraction_of_serial_leg": p_s / serial["decode_seconds"],
        # Prediction 3 gate: the two legs share `begin`, so a spread beyond a few
        # percent means the charge is not the shared quantity the model assumes.
        "prefill_symmetry_abs_diff_seconds": abs(p_s - p_m),
        "prefill_symmetry_pct_of_mean": sym_pct,
        "prefill_symmetry_within_2pct": sym_pct <= 2.0,
        "prefill_symmetry_exceeds_5pct": sym_pct > 5.0,
        # Local window, measured.
        "local_window_tokens": mtp["emitted_token_total"],
        "local_r_from_decode_seconds": r,
        "local_r_from_parent_spt": serial["parent_measured_seconds_per_token"]
        / mtp["parent_measured_seconds_per_token"],
        "local_r_ideal_prefill_free": r_ideal,
        "local_leverage_measured": r_ideal - r,
        "local_leverage_from_identity": identity_rhs,
        "local_leverage_exact_asymmetric": exact_gap,
        "identity_abs_error": abs((r_ideal - r) - identity_rhs),
        "identity_rel_error": abs((r_ideal - r) - identity_rhs) / max(r_ideal - r, 1e-12),
        # Ranked window at the promoted ratio. p*(r-1) is the score the campaign
        # would recover if `begin` cost nothing and nothing else changed.
        "ranked_r_promoted": ranked_r,
        "ranked_window_tokens": RANKED_WINDOW,
        "ranked_p_using_local_steady_rates": p_ranked_local_host,
        "ranked_leverage_at_local_window_p": p * (ranked_r - 1.0),
        "ranked_leverage_at_ranked_window_p": p_ranked_local_host * (ranked_r - 1.0),
        "ranked_r_ideal_at_ranked_window_p": ranked_r
        + p_ranked_local_host * (ranked_r - 1.0),
        "ranked_window_decode_work_serial_seconds": ranked_d_s,
        "ranked_window_decode_work_mtp_seconds": ranked_d_m,
        "ranked_window_local_host_r": (p_s + ranked_d_s) / (p_m + ranked_d_m),
        # How much the local ratio understates the ranked prize, given that the
        # same fixed P sits on a much shorter local decode window.
        "local_to_ranked_leverage_factor": (p * (ranked_r - 1.0))
        / max(r_ideal - r, 1e-12),
        # Direct vs inferred. `prefill_seconds` from decompose() is an upper
        # bound that also absorbs any per-emitted-token parent cost, so
        # inferred >= direct is the expected sign.
        "inferred_prefill_seconds": inferred["prefill_seconds"],
        "inferred_minus_direct_mtp_seconds": inferred["prefill_seconds"] - p_m,
        "inferred_over_direct_mtp_ratio": inferred["prefill_seconds"] / p_m,
        "inferred_agrees_within_10pct": abs(inferred["prefill_seconds"] - p_m) / p_m <= 0.10,
    }
    for cut in (0.20, 0.50, 1.00):
        out[f"ranked_leverage_if_prefill_cut_{int(cut * 100)}pct"] = (
            cut * p_ranked_local_host * (ranked_r - 1.0)
        )
    return out


def window_model(short: dict, long: dict) -> dict:
    """Solve leg_seconds(N) = F + d*N for one arm across two decode windows.

    F is whatever the leg pays regardless of window length -- the seed prefill
    plus any other fixed cost -- so comparing F with the directly measured
    seed_prefill_seconds tests whether `begin` explains the fixed term.
    """
    n_a, n_b = short["emitted_token_total"], long["emitted_token_total"]
    if n_a == n_b:
        raise SystemExit("prefill_amdahl: window model needs two different decode windows")
    t_a, t_b = short["decode_seconds"], long["decode_seconds"]
    d = (t_b - t_a) / (n_b - n_a)
    f = t_a - d * n_a
    return {
        "short_window_tokens": n_a,
        "long_window_tokens": n_b,
        "solved_fixed_seconds": f,
        "solved_seconds_per_token": d,
        "direct_prefill_seconds_long_window": long["seed_prefill_seconds"],
        "fixed_minus_direct_prefill_seconds": f - long["seed_prefill_seconds"],
        "fixed_over_direct_prefill_ratio": f / long["seed_prefill_seconds"],
    }


def decompose(serial: dict, mtp: dict) -> dict:
    """Solve residual_leg = P + N_leg * c for the shared prefill P."""
    d_rounds = serial["round_count"] - mtp["round_count"]
    if d_rounds == 0:
        raise SystemExit("prefill_amdahl: both legs ran the same round count; P is not separable")
    per_round_overhead = (serial["residual_seconds"] - mtp["residual_seconds"]) / d_rounds
    prefill = mtp["residual_seconds"] - mtp["round_count"] * per_round_overhead

    d_s = serial["decode_seconds"] - prefill
    d_c = mtp["decode_seconds"] - prefill
    score = (prefill + d_s) / (prefill + d_c)

    # d/dP of (P + D_s) / (P + D_c)
    dscore_dp = (d_c - d_s) / (prefill + d_c) ** 2

    out = {
        "prefill_seconds": prefill,
        "parent_per_round_overhead_seconds": per_round_overhead,
        "prefill_estimate_from_serial_residual": serial["residual_seconds"],
        "prefill_estimate_from_mtp_residual": mtp["residual_seconds"],
        "prefill_residual_spread_seconds": serial["residual_seconds"] - mtp["residual_seconds"],
        "decode_work_serial_seconds": d_s,
        "decode_work_mtp_seconds": d_c,
        "prefill_fraction_of_serial_leg": prefill / serial["decode_seconds"],
        "prefill_fraction_of_mtp_leg": prefill / mtp["decode_seconds"],
        "modelled_local_score": score,
        "measured_local_score": serial["parent_measured_seconds_per_token"]
        / mtp["parent_measured_seconds_per_token"],
        "dscore_dprefill_per_second": dscore_dp,
        "score_points_per_100ms_prefill_removed": -dscore_dp * 0.100,
        "score_if_prefill_were_free": d_s / d_c,
        "headroom_to_prefill_free": d_s / d_c - score,
        # Attribution band. Allowing a third per-emitted-token parent cost k, the
        # legs give R_leg = P + N_leg*c + T*k. The subtraction that fixes c is
        # independent of k because both legs emit the same T, so `prefill_seconds`
        # is exactly P + T*k: an upper bound on P, tight to whatever the parent
        # spends per token on its token-equality check and array append.
        "prefill_is_upper_bound_because_of_per_token_overhead": True,
        "emitted_tokens_both_legs": serial["emitted_token_total"],
        "prefill_upper_bound_slack_if_k_is_50us": 50e-6 * serial["emitted_token_total"],
        # Decode-only speedup: what the local ratio would read if the shared
        # prefill were removed from both legs. The gap to `measured_local_score`
        # is exactly how much fixed prefill dilutes the local number.
        "decode_only_speedup": d_s / d_c,
        "serial_decode_seconds_per_token_excl_prefill": d_s / serial["emitted_token_total"],
        "mtp_decode_seconds_per_token_excl_prefill": d_c / mtp["emitted_token_total"],
    }

    # Same model re-evaluated at the ranked 512-token decode window. Prefill is
    # identical (the ranked seed is also 512 tokens); only the decode work grows.
    for label, leg, stats in (("serial", "serial", serial), ("mtp", "mtp", mtp)):
        w = stats["emitted_token_total"]
        steady = stats["steady_seconds_per_token"]
        fixed = stats["decode_seconds"] - steady * w
        out[f"ranked_window_fixed_seconds_{label}"] = fixed
        out[f"ranked_window_steady_seconds_per_token_{label}"] = steady
        out[f"ranked_window_leg_seconds_{label}"] = fixed + steady * RANKED_WINDOW

    ranked_s = out["ranked_window_leg_seconds_serial"]
    ranked_c = out["ranked_window_leg_seconds_mtp"]
    ranked_ds = ranked_s - prefill
    ranked_dc = ranked_c - prefill
    out["ranked_window_modelled_score"] = ranked_s / ranked_c
    out["ranked_window_prefill_fraction_of_mtp_leg"] = prefill / ranked_c
    out["ranked_window_prefill_fraction_of_serial_leg"] = prefill / ranked_s
    out["ranked_window_dscore_dprefill_per_second"] = (ranked_dc - ranked_ds) / ranked_c**2
    out["ranked_window_score_points_per_100ms_prefill_removed"] = (
        -out["ranked_window_dscore_dprefill_per_second"] * 0.100
    )
    out["ranked_window_score_if_prefill_were_free"] = ranked_ds / ranked_dc
    for cut in (0.20, 0.30, 0.50, 1.00):
        p2 = prefill * (1.0 - cut)
        out[f"ranked_window_score_if_prefill_cut_{int(cut * 100)}pct"] = (
            (p2 + ranked_ds) / (p2 + ranked_dc)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--tag", default="part-a-baseline")
    ap.add_argument("--mode", default="--local-iterate")
    ap.add_argument("--base-sha", default="")
    ap.add_argument("--head-sha", default="")
    ap.add_argument("--score-json", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--ranked-speedup", type=float, default=RANKED_SPEEDUP)
    ap.add_argument(
        "--short-window-capture-dir",
        type=Path,
        default=None,
        help="second capture dir at a different decode window, for the fixed/variable solve",
    )
    args = ap.parse_args()

    legs = load_timed_reports(args.capture_dir)
    missing = {"serial", "mtp"} - set(legs)
    if missing:
        raise SystemExit(f"prefill_amdahl: no timed report captured for {sorted(missing)}")

    serial = leg_stats(legs["serial"])
    mtp = leg_stats(legs["mtp"])
    model = decompose(serial, mtp)
    direct = direct_charge(serial, mtp, args.ranked_speedup, model)

    windows: dict[str, dict] = {}
    if args.short_window_capture_dir:
        short_legs = load_timed_reports(args.short_window_capture_dir)
        short_missing = {"serial", "mtp"} - set(short_legs)
        if short_missing:
            raise SystemExit(
                f"prefill_amdahl: short-window capture is missing {sorted(short_missing)}"
            )
        for name, long_stats in (("serial", serial), ("mtp", mtp)):
            windows[name] = window_model(leg_stats(short_legs[name]), long_stats)

    provenance = {
        "tag": args.tag,
        "mode": args.mode,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "host_chip": sysctl("machdep.cpu.brand_string"),
        "host_memsize_bytes": sysctl("hw.memsize"),
        "host_os": f"{platform.mac_ver()[0]} ({sysctl('kern.osversion')})",
        "ranked_window_tokens": RANKED_WINDOW,
        "notes": args.notes,
    }
    if args.score_json and args.score_json.exists():
        provenance["score_json"] = json.loads(args.score_json.read_text())

    payload = {
        "provenance": provenance,
        "serial_leg": serial,
        "mtp_leg": mtp,
        "amdahl": model,
        "direct": direct,
    }
    if windows:
        payload["window_model"] = windows
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=f"prefill-amdahl-{args.tag}",
            job_type="measurement",
            group=os.environ.get(
                "WANDB_RUN_GROUP", "qwen38-r1-e3-seed-prefill-amdahl"
            ),
            config={**provenance, "serial_leg": serial, "mtp_leg": mtp},
        )
        flat = {f"serial/{k}": v for k, v in serial.items() if isinstance(v, (int, float))}
        flat |= {f"mtp/{k}": v for k, v in mtp.items() if isinstance(v, (int, float))}
        flat |= {f"amdahl/{k}": v for k, v in model.items() if isinstance(v, (int, float))}
        flat |= {f"direct/{k}": v for k, v in direct.items() if isinstance(v, (int, float))}
        for name, block in windows.items():
            flat |= {
                f"window_model/{name}/{k}": v
                for k, v in block.items()
                if isinstance(v, (int, float))
            }
        run.log(flat)
        run.summary.update(flat)
        for leg_name, stats in (("serial", legs["serial"]), ("mtp", legs["mtp"])):
            table = wandb.Table(columns=["round_index", "block_request_seconds"])
            for i, v in enumerate(stats["block_request_seconds"]):
                table.add_data(i, float(v))
            run.log({f"{leg_name}/block_request_seconds": table})
        print(f"WANDB_RUN_URL {run.url}", file=sys.stderr)
        print(f"WANDB_RUN_ID {run.id}", file=sys.stderr)
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
