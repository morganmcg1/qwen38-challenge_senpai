#!/usr/bin/env python3
"""Seal one E163 pinned-width session into a report and an artifact.

The session runs six gated legs in the palindrome

    shipped@dA   arm@dA   shipped@dB   shipped@dB   arm@dA   shipped@dA

so all three leg kinds sit at mean position 3.5 and a monotone drift in leg
index cancels to first order in every pairwise contrast.

Three things come out of it:

1. THE ARM.  shipped@dA against arm@dA at one pinned verify width.
2. THE BOUNDARY.  shipped@dB against shipped@dA is R(width+1) - R(width) inside
   one thermal session, which no ranked receipt can show because no ranked
   prompt sits between those row counts.
3. THE NOISE CHANNEL.  Every leg also times a true serial control at depth 0.
   At M = 1 the width switch reaches `default: break` and launches no routed
   QMV, so no arm can move the serial leg.  The spread of the six serial
   numbers is therefore a measured floor for this session, not a quoted one.

`R` is milliseconds per decode round with the seed prefill removed.  It is
computed twice per leg, once from the leg total and once from the sum of the
traced per-round costs, and the two must agree.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "out"
ARTIFACTS = ROOT / "research" / "e163-artifacts"

TIMED = re.compile(
    r"^mtp-timed: tokens=(?P<tokens>\d+) depth=(?P<depth>\d+) "
    r"rounds=(?P<rounds>\d+) accepted_draft_rate=(?P<adr>[\d.]+) "
    r"all_tokens_matched=(?P<matched>\w+) "
    r"reference_checked_rows=(?P<checked>\d+)/(?P<total>\d+) "
    r"seconds_per_token=(?P<spt>[\d.]+)"
)
BEGIN = re.compile(r"^mtp-trace: begin seed=(?P<seed>\d+) .*?\bwall_us=(?P<wall>\d+)")
ROUND = re.compile(r"^mtp-trace: round=(?P<round>\d+) d=(?P<d>\d+) acc=(?P<acc>\d+) ")
FIELD = re.compile(r"(\w+)=(-?[\d.]+)")


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def read_timed(path: pathlib.Path) -> list[dict]:
    phases = []
    for line in path.read_text(errors="replace").splitlines():
        hit = TIMED.match(line)
        if hit:
            phases.append(hit.groupdict())
    return phases


def read_trace(path: pathlib.Path) -> dict:
    """Per-round costs of the MTP leg, plus the seed prefill that precedes it.

    A leg emits one `begin` for the serial control's prefill and one for the
    MTP leg's prefill.  Only the MTP leg emits round lines, so the begin that
    matters is the last one seen before the first round line.
    """
    if not path.exists():
        return {}
    prefill_us = None
    last_begin = None
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        begin = BEGIN.match(line)
        if begin:
            last_begin = int(begin.group("wall"))
            continue
        head = ROUND.match(line)
        if head:
            if prefill_us is None:
                prefill_us = last_begin
            fields = {k: float(v) for k, v in FIELD.findall(line)}
            fields["width"] = float(head.group("d")) + 1.0
            rounds.append(fields)
    if not rounds:
        return {}
    widths: dict[int, int] = {}
    for row in rounds:
        widths[int(row["width"])] = widths.get(int(row["width"]), 0) + 1
    return {
        "prefill_seconds": None if prefill_us is None else prefill_us / 1e6,
        "rounds": len(rounds),
        "width_histogram": dict(sorted(widths.items())),
        "pinned_width_share": max(widths.values()) / len(rounds),
        "round_us_mean": statistics.fmean(r["round_us"] for r in rounds),
        "round_us_sd": (
            statistics.stdev([r["round_us"] for r in rounds]) if len(rounds) > 1 else 0.0
        ),
        "eval_wall_us_mean": statistics.fmean(r["eval_wall_us"] for r in rounds),
        "verify_build_us_mean": statistics.fmean(r["verify_build_us"] for r in rounds),
        "draft_build_us_mean": statistics.fmean(r["draft_build_us"] for r in rounds),
        "accepted_mean": statistics.fmean(r["acc"] for r in rounds),
    }


def load_leg(path: pathlib.Path) -> dict:
    meta = read_meta(path / "meta.txt")
    score = json.loads((path / "score.json").read_text())["metrics"]
    phases = read_timed(path / "wrapper.err")
    serial = next(p for p in phases if p["depth"] == "0")
    mtp = next(p for p in phases if p["depth"] != "0")
    trace = read_trace(path / "trace.txt")

    tokens = int(mtp["tokens"])
    rounds = int(mtp["rounds"])
    spt = float(mtp["spt"])
    leg_seconds = tokens * spt
    prefill = trace.get("prefill_seconds")
    leg = {
        "tag": meta["tag"],
        "plan": meta["e163_plan"],
        "position": int(meta["e163_position"]),
        "pinned_depth": int(meta["e163_pinned_depth"]),
        "verify_width": int(meta["e163_verify_width"]),
        "arm_key": f"{meta['e163_plan']}@d{meta['e163_pinned_depth']}",
        "decode_tokens": tokens,
        "rounds": rounds,
        "mtp_seconds_per_token": spt,
        "serial_seconds_per_token": float(serial["spt"]),
        "mtp_decode_speedup": score["mtp_decode_speedup"],
        "effective_mean_draft_len": score["effective_mean_draft_len"],
        "accepted_draft_rate": score["accepted_draft_rate"],
        "all_tokens_matched": bool(score["all_tokens_matched"]),
        "residual_divergence_count": score["residual_divergence_count"],
        "reference_checked_rows": f"{mtp['checked']}/{mtp['total']}",
        "head_provenance_sha256": score["head_provenance_sha256"],
        "leg_seconds": leg_seconds,
        "seed_prefill_seconds": prefill,
        "R_ms_from_leg": (
            None if prefill is None else (leg_seconds - prefill) / rounds * 1e3
        ),
        "R_ms_from_trace": (
            None if not trace else trace["round_us_mean"] / 1e3 * trace["rounds"] / rounds
        ),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "worker_sha256": meta["worker_sha256"],
        "post_run_worker_sha256": meta["post_run_worker_sha256"],
        "base_sha": meta["base_sha"],
        "dirty_candidate_paths": int(meta["dirty_candidate_paths"]),
        "sandbox": meta["sandbox"],
        "trace": meta["trace"],
        "exit": int(meta["exit"]),
        "session_commit": meta.get("e163_session_commit"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
    }
    leg.update({f"trace_{k}": v for k, v in trace.items()})
    return leg


def scored_surface_differs(first: str, second: str) -> list[str]:
    """Files on the SUBMITTED surface that differ between two commits.

    A leg records `git rev-parse HEAD` when it starts, so a research-only or
    test-only commit landing mid-session gives the legs different commits while
    the timed bytes are identical. What binds a timing contrast is the built
    worker digest, which the caller already asserts equal across every leg.
    This decides whether a commit difference is cosmetic or real.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", first, second, "--", "Sources", "Vendor", "Package.swift"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [f"<git diff {first}..{second} failed: {result.stderr.strip()}>"]
    return [line for line in result.stdout.splitlines() if line]


def contrast(name: str, base: list[dict], arm: list[dict], key: str) -> dict:
    """Per-cent by which `arm` is FASTER than `base` on `key`, with a range.

    Two replicates per side, so the interval is the sum of the half-ranges.  It
    is a range, not a standard error, and it is reported as one.
    """
    b = [leg[key] for leg in base]
    a = [leg[key] for leg in arm]
    b_mean, a_mean = statistics.fmean(b), statistics.fmean(a)
    b_half, a_half = (max(b) - min(b)) / 2, (max(a) - min(a)) / 2
    gain = (b_mean - a_mean) / b_mean * 100.0
    half = (b_half + a_half) / b_mean * 100.0
    return {
        "contrast": name,
        "metric": key,
        "base_arm": base[0]["arm_key"],
        "base_values": b,
        "base_mean": b_mean,
        "test_arm": arm[0]["arm_key"],
        "test_values": a,
        "test_mean": a_mean,
        "gain_percent": gain,
        "half_range_percent": half,
        "separated": gain - half > 0.0,
        "gain_minus_half_range_percent": gain - half,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--min-useful-percent", type=float, default=0.06)
    ap.add_argument("--noise-floor-percent", type=float, default=0.039)
    args = ap.parse_args()

    legs = sorted(
        (
            load_leg(p)
            for p in OUT.glob(f"e163{args.label}p[0-9]*")
            if (p / "score.json").exists()
        ),
        key=lambda leg: leg["position"],
    )
    if not legs:
        print(f"e163_pinned_report: no sealed legs for label {args.label}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for leg in legs:
        if leg["exit"] != 0:
            problems.append(f"{leg['tag']}: exited {leg['exit']}")
        if not leg["all_tokens_matched"]:
            problems.append(f"{leg['tag']}: all_tokens_matched=false")
        if leg["residual_divergence_count"]:
            problems.append(f"{leg['tag']}: residual_divergence_count nonzero")
        if leg["gate_qualified_for_timing"] != "true":
            problems.append(f"{leg['tag']}: not gate qualified")
        if leg["dirty_candidate_paths"]:
            problems.append(f"{leg['tag']}: scored surface dirty")
        if leg["worker_sha256"] != leg["post_run_worker_sha256"]:
            problems.append(f"{leg['tag']}: worker digest moved during the leg")
        if leg["worker_sha256"] != legs[0]["worker_sha256"]:
            problems.append(f"{leg['tag']}: a different worker than leg 1")
        r_leg, r_trace = leg["R_ms_from_leg"], leg["R_ms_from_trace"]
        if r_leg is not None and r_trace is not None:
            if abs(r_leg - r_trace) / r_leg > 0.05:
                problems.append(
                    f"{leg['tag']}: R disagrees between the leg total "
                    f"({r_leg:.2f} ms) and the traced rounds ({r_trace:.2f} ms)"
                )

    # Every leg must have been driven by one session script and one worker.
    session_commits = {leg["session_commit"] for leg in legs}
    if len(session_commits) > 1:
        problems.append(f"legs came from different sessions {sorted(session_commits)}")

    # A leg records HEAD when it starts, so a research-only commit landing
    # mid-session gives the legs different commits over identical timed bytes.
    # Cosmetic differences are recorded; a moved scored surface is a problem.
    leg_commits = sorted({leg["base_sha"] for leg in legs})
    commit_note = None
    if len(leg_commits) > 1:
        moved: list[str] = []
        for other in leg_commits[1:]:
            moved.extend(scored_surface_differs(leg_commits[0], other))
        if moved:
            problems.append(
                f"the scored surface moved between leg commits {leg_commits}: "
                f"{sorted(set(moved))}"
            )
        else:
            commit_note = (
                f"legs recorded {len(leg_commits)} commits {leg_commits}, and no file "
                "under Sources, Vendor or Package.swift differs between them. The "
                "worker digest is identical on every leg, so the timed bytes did not "
                "move."
            )

    # The accept ledger must be identical inside one pinned depth, or the
    # arithmetic moved and the timing contrast is void.
    by_depth: dict[int, list[dict]] = {}
    for leg in legs:
        by_depth.setdefault(leg["pinned_depth"], []).append(leg)
    for depth, group in by_depth.items():
        for field in (
            "rounds",
            "effective_mean_draft_len",
            "accepted_draft_rate",
            "trace_width_histogram",
            "reference_checked_rows",
        ):
            seen = {json.dumps(leg.get(field), sort_keys=True) for leg in group}
            if len(seen) > 1:
                problems.append(
                    f"pinned depth {depth}: {field} differs across arms {sorted(seen)}"
                )

    by_arm: dict[str, list[dict]] = {}
    for leg in legs:
        by_arm.setdefault(leg["arm_key"], []).append(leg)

    depths = sorted(by_depth)
    low = depths[0]
    variant = next((leg["plan"] for leg in by_depth[low] if leg["plan"] != "shipped"), None)

    contrasts = []
    if variant is not None:
        base = by_arm[f"shipped@d{low}"]
        test = by_arm[f"{variant}@d{low}"]
        contrasts.append(contrast("arm", base, test, "mtp_seconds_per_token"))
        if all(leg["R_ms_from_leg"] is not None for leg in base + test):
            contrasts.append(contrast("arm", base, test, "R_ms_from_leg"))
        # Adjacent-position pairs, which cancel a linear drift exactly.
        paired = []
        for first_pos, second_pos in ((1, 2), (5, 6)):
            found = [leg for leg in legs if leg["position"] in (first_pos, second_pos)]
            if len(found) != 2 or {leg["plan"] for leg in found} != {"shipped", variant}:
                continue
            ship = next(leg for leg in found if leg["plan"] == "shipped")
            arm = next(leg for leg in found if leg["plan"] == variant)
            paired.append(
                (ship["mtp_seconds_per_token"] - arm["mtp_seconds_per_token"])
                / ship["mtp_seconds_per_token"]
                * 100.0
            )
        if paired:
            half = (max(paired) - min(paired)) / 2
            contrasts.append(
                {
                    "contrast": "arm-adjacent-pairs",
                    "metric": "mtp_seconds_per_token",
                    "gain_percent_each": paired,
                    "gain_percent": statistics.fmean(paired),
                    "half_range_percent": half,
                    "gain_minus_half_range_percent": statistics.fmean(paired) - half,
                }
            )

    serials = [leg["serial_seconds_per_token"] for leg in legs]
    serial_mean = statistics.fmean(serials)
    noise = {
        "channel": "true serial control, depth 0, no routed QMV launched",
        "values": serials,
        "mean": serial_mean,
        "sd_percent": (
            statistics.stdev(serials) / serial_mean * 100.0 if len(serials) > 1 else 0.0
        ),
        "half_range_percent": (max(serials) - min(serials)) / 2 / serial_mean * 100.0,
        "predeclared_floor_percent": args.noise_floor_percent,
    }

    boundary = None
    if len(depths) == 2:
        lo = by_arm[f"shipped@d{depths[0]}"]
        hi = by_arm[f"shipped@d{depths[1]}"]
        if all(leg["R_ms_from_leg"] is not None for leg in lo + hi):
            r_lo = statistics.fmean(leg["R_ms_from_leg"] for leg in lo)
            r_hi = statistics.fmean(leg["R_ms_from_leg"] for leg in hi)
            w_lo, w_hi = lo[0]["verify_width"], hi[0]["verify_width"]
            ev_lo = statistics.fmean(leg["trace_eval_wall_us_mean"] for leg in lo) / 1e3
            ev_hi = statistics.fmean(leg["trace_eval_wall_us_mean"] for leg in hi) / 1e3
            vb_lo = statistics.fmean(leg["trace_verify_build_us_mean"] for leg in lo) / 1e3
            vb_hi = statistics.fmean(leg["trace_verify_build_us_mean"] for leg in hi) / 1e3
            boundary = {
                "contrast": f"R({w_hi}) - R({w_lo}) on the shipped plan, one session",
                "width_low": w_lo,
                "width_high": w_hi,
                "R_ms_low": r_lo,
                "R_ms_high": r_hi,
                "delta_R_ms_per_row": r_hi - r_lo,
                "R_ms_low_range": [
                    min(leg["R_ms_from_leg"] for leg in lo),
                    max(leg["R_ms_from_leg"] for leg in lo),
                ],
                "R_ms_high_range": [
                    min(leg["R_ms_from_leg"] for leg in hi),
                    max(leg["R_ms_from_leg"] for leg in hi),
                ],
                "gpu_eval_ms_low": ev_lo,
                "gpu_eval_ms_high": ev_hi,
                "delta_gpu_eval_ms_per_row": ev_hi - ev_lo,
                "cpu_verify_build_ms_low": vb_lo,
                "cpu_verify_build_ms_high": vb_hi,
                "delta_cpu_verify_build_ms_per_row": vb_hi - vb_lo,
                "gpu_share_of_marginal_row": (
                    (ev_hi - ev_lo) / (r_hi - r_lo) if r_hi != r_lo else None
                ),
            }

    shares = [
        {
            "arm_key": key,
            "verify_width": group[0]["verify_width"],
            "R_ms": statistics.fmean(leg["R_ms_from_leg"] for leg in group),
            "gpu_eval_share_of_round": statistics.fmean(
                leg["trace_eval_wall_us_mean"] / leg["trace_round_us_mean"] for leg in group
            ),
            "cpu_verify_build_share_of_round": statistics.fmean(
                leg["trace_verify_build_us_mean"] / leg["trace_round_us_mean"]
                for leg in group
            ),
            "draft_build_share_of_round": statistics.fmean(
                leg["trace_draft_build_us_mean"] / leg["trace_round_us_mean"]
                for leg in group
            ),
        }
        for key, group in sorted(by_arm.items())
        if all(leg.get("trace_round_us_mean") for leg in group)
    ]

    report = {
        "label": args.label,
        "harness": "local",
        "official_or_ranked_score": False,
        "identity": {
            "base_sha": legs[0]["base_sha"],
            "leg_commits": leg_commits,
            "leg_commit_note": commit_note,
            "session_commit": legs[0]["session_commit"],
            "worker_sha256": legs[0]["worker_sha256"],
            "host": legs[0]["host"],
            "chip": legs[0]["chip"],
            "metallib_source_fingerprint": legs[0]["metallib_source_fingerprint"],
            "head_provenance_sha256": legs[0]["head_provenance_sha256"],
            "decode_tokens": legs[0]["decode_tokens"],
            "local_mode": "--local-iterate",
            "sandbox": legs[0]["sandbox"],
        },
        "order": [leg["arm_key"] for leg in legs],
        "legs": legs,
        "contrasts": contrasts,
        "boundary": boundary,
        "round_cost_shares": shares,
        "noise_channel": noise,
        "min_useful_percent": args.min_useful_percent,
        "problems": problems,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"e163_pinned_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"=== E163 pinned session {args.label} (harness=local) ===")
    print(f"artifact {path.relative_to(ROOT)}")
    if commit_note:
        print(f"note: {commit_note}")
    print(
        f"{'pos':>3} {'arm':>16} {'W':>2} {'mtp s/tok':>11} {'R ms':>8} "
        f"{'rounds':>6} {'edl':>7} {'Tin':>6} {'Tout':>6}"
    )
    for leg in legs:
        r = leg["R_ms_from_leg"]
        print(
            f"{leg['position']:>3} {leg['arm_key']:>16} {leg['verify_width']:>2} "
            f"{leg['mtp_seconds_per_token']:>11.7f} "
            f"{(math.nan if r is None else r):>8.2f} {leg['rounds']:>6} "
            f"{leg['effective_mean_draft_len']:>7.4f} "
            f"{leg['gpu_temp_entry_c']:>6.2f} {leg['gpu_temp_exit_c']:>6.2f}"
        )
    for item in contrasts:
        print(
            f"contrast {item['contrast']:>20} {item['metric']:>18}: "
            f"{item['gain_percent']:+.4f} % +/- {item['half_range_percent']:.4f} "
            f"(gain - half range {item['gain_minus_half_range_percent']:+.4f} %)"
        )
    print(
        f"noise channel serial spt: sd {noise['sd_percent']:.4f} %, "
        f"half range {noise['half_range_percent']:.4f} %, "
        f"predeclared floor {noise['predeclared_floor_percent']:.4f} %"
    )
    if boundary:
        print(
            f"boundary {boundary['contrast']}: "
            f"{boundary['delta_R_ms_per_row']:+.3f} ms/row, of which GPU eval "
            f"{boundary['delta_gpu_eval_ms_per_row']:+.3f} ms and CPU verify build "
            f"{boundary['delta_cpu_verify_build_ms_per_row']:+.3f} ms"
        )
    for share in shares:
        print(
            f"round shares {share['arm_key']:>16} W={share['verify_width']}: "
            f"R {share['R_ms']:.2f} ms, GPU eval {share['gpu_eval_share_of_round']:.3f}, "
            f"CPU verify build {share['cpu_verify_build_share_of_round']:.3f}, "
            f"draft build {share['draft_build_share_of_round']:.3f}"
        )
    if problems:
        print("PROBLEMS:")
        for line in problems:
            print(f"  {line}")
        return 1
    print("E163 pinned session gates OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
