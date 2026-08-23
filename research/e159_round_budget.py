#!/usr/bin/env python3
"""E159 R1 -- the decode round's cost law and its budget.

Reads the legs of one pinned-depth timing session and, optionally, one
sync-head traced session, and reports:

  R(D)      per-round seconds at a pinned PROPOSED draft count D
  s, h      intercept and slope of R(D) = s + h*D
  rho       8h/s, the dimensionless headline the advisor pre-registered
  budget    the four shares of the round at the adaptive operating point
  a, q      mean ACCEPTED and mean PROPOSED drafts per round, never conflated

Arithmetic, all of it exact rather than estimated (QwenRuntimeMTPDriver.swift
:274, :295, :336):

    rounds = tokens - acceptedDraftTotal
    R      = decodeSeconds / rounds
    a      = acceptedDraftTotal / rounds
    q      = effective_mean_draft_len          (PROPOSED, not accepted)
    mtp    = R / (1 + a)

harness=local. No leg here is gate-qualified and no number here is a ranked
score.

Usage:
  python3 research/e159_round_budget.py research/out/e159-timing \\
      --trace research/out/e159-trace \\
      --json research/e159-artifacts/e159_round_budget.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import statistics as st

TRIM = 0.10
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
FIELD_RE = re.compile(r"(\w+)_us=(\d+)")


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def trimmed_mean(values: list[float], frac: float = TRIM) -> float:
    ordered = sorted(values)
    drop = int(len(ordered) * frac)
    kept = ordered[drop: len(ordered) - drop] or ordered
    return st.fmean(kept)


def pinned_depth(arm_env: str) -> int | None:
    match = re.search(r"MLX_E159_FIXED_DRAFT_DEPTH=(\d+)", arm_env)
    return int(match.group(1)) if match else None


def load_legs(session: pathlib.Path) -> list[dict]:
    legs = []
    for leg_dir in sorted(session.glob("b*-*")):
        report_path = leg_dir / "report.json"
        meta_path = leg_dir / "meta.txt"
        if not (report_path.exists() and meta_path.exists()):
            continue
        report = json.loads(report_path.read_text())
        meta = read_meta(meta_path)
        tokens = report["decode_token_count"]
        accepted = report["accepted_draft_total"]
        rounds = report["round_count"]
        decode_seconds = report["decode_seconds"]
        # `decode_seconds` is the parent's whole decode loop and INCLUDES the
        # one-off seed prefill. That prefill is a per-LEG constant of about 4 s,
        # so dividing it by a round count that itself falls with depth adds a
        # spurious depth slope of P*(1+a)/tokens to R. Subtract it to get the
        # round cost law; keep the raw form because the assignment names it.
        prefill_seconds = report["seed_prefill_seconds"]
        per_round = [v * 1e6 for v in report["block_request_seconds"]]
        depth = pinned_depth(meta.get("arm_env", ""))
        q = report["effective_mean_draft_len"]
        a = accepted / rounds
        legs.append({
            "dir": leg_dir.name,
            "arm": meta["arm_label"],
            "block": int(meta.get("block", -1)),
            "pinned_depth": depth,
            "tokens": tokens,
            "rounds": rounds,
            "rounds_identity": tokens - accepted,
            "accepted_draft_total": accepted,
            "non_drafting_round_count": report["non_drafting_round_count"],
            "decode_seconds": decode_seconds,
            "seed_prefill_seconds": prefill_seconds,
            "R_seconds": decode_seconds / rounds,
            "R_decode_seconds": (decode_seconds - prefill_seconds) / rounds,
            "prefill_per_round_seconds": prefill_seconds / rounds,
            "round_us_trimmed": trimmed_mean(per_round[1:]),
            "first_round_us": per_round[0],
            "a_accepted_per_round": a,
            "q_proposed_per_round": q,
            "alpha_accept_fraction": accepted / (q * rounds) if q else 0.0,
            "mtp_seconds_per_token": report[
                "parent_measured_seconds_per_token"],
            "R_over_1_plus_a": decode_seconds / rounds / (1 + a),
            "rejected_per_round": q - a,
            "all_tokens_matched": report["all_tokens_matched"],
            "residual_divergence_count": report["residual_divergence_count"],
            "draft_length_histogram": dict(sorted(
                collections.Counter(
                    report["effective_draft_lengths"]).items())),
            "head_provenance_sha256": report.get(
                "head_provenance", {}).get("sha256", ""),
            "uses_pinned_mtp_head": report.get("uses_pinned_mtp_head"),
            "entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
            "exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
            "wall_seconds": float(meta.get("leg_wall_seconds", "nan")),
            "worker_sha256": meta.get("worker_sha256", ""),
            "git_head": meta.get("git_head", ""),
        })
    return legs


def ols(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares with textbook standard errors."""
    n = len(xs)
    if n < 3:
        raise SystemExit(f"ols: need at least 3 points, got {n}")
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    sigma2 = sum(r * r for r in resid) / dof
    se_slope = math.sqrt(sigma2 / sxx)
    se_intercept = math.sqrt(sigma2 * (1.0 / n + mx * mx / sxx))
    sst = sum((y - my) ** 2 for y in ys)
    return {
        "n": n,
        "intercept": intercept,
        "slope": slope,
        "se_intercept": se_intercept,
        "se_slope": se_slope,
        "residuals": resid,
        "residual_sd": math.sqrt(sigma2),
        "r_squared": 1.0 - sum(r * r for r in resid) / sst if sst else 1.0,
    }


def quadratic_term(xs: list[float], ys: list[float]) -> dict:
    """Fit y = c0 + c1 x + c2 x^2 and report c2 with its standard error."""
    n = len(xs)
    design = [[1.0, x, x * x] for x in xs]
    ata = [[sum(row[i] * row[j] for row in design) for j in range(3)]
           for i in range(3)]
    atb = [sum(row[i] * y for row, y in zip(design, ys)) for i in range(3)]
    aug = [ata[i] + [atb[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-18:
            return {"c2": float("nan"), "se_c2": float("nan"),
                    "t_c2": float("nan")}
        for r in range(3):
            if r == col:
                continue
            factor = aug[r][col] / aug[col][col]
            for c in range(col, 4):
                aug[r][c] -= factor * aug[col][c]
    coef = [aug[i][3] / aug[i][i] for i in range(3)]
    resid = [y - (coef[0] + coef[1] * x + coef[2] * x * x)
             for x, y in zip(xs, ys)]
    dof = n - 3
    if dof <= 0:
        return {"c2": coef[2], "se_c2": float("nan"), "t_c2": float("nan")}
    sigma2 = sum(r * r for r in resid) / dof
    # (A'A)^-1 [2][2] through the same elimination, on an identity column.
    inv = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    work = [row[:] for row in ata]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(work[r][col]))
        work[col], work[pivot] = work[pivot], work[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        inv[col] = [v / scale for v in inv[col]]
        for r in range(3):
            if r == col:
                continue
            factor = work[r][col]
            work[r] = [v - factor * w for v, w in zip(work[r], work[col])]
            inv[r] = [v - factor * w for v, w in zip(inv[r], inv[col])]
    se_c2 = math.sqrt(sigma2 * inv[2][2])
    return {"c2": coef[2], "se_c2": se_c2,
            "t_c2": coef[2] / se_c2 if se_c2 else float("nan"),
            "residual_sd": math.sqrt(sigma2)}


def implied_uniform_p(a: float, depth: int) -> float:
    """The single per-position acceptance a geometric chain would need.

    Under independent per-position acceptance p, the mean accepted prefix of a
    D-draft chain is sum_{k=1..D} p^k. Inverting that for p turns the measured
    `a` into one comparable number per depth, so a p that FALLS with depth says
    the later positions are genuinely harder rather than merely rarer.
    """
    if depth <= 0 or a <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = sum(mid ** k for k in range(1, depth + 1))
        if total < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def parse_trace(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        head = ROUND_RE.search(line)
        if not head:
            continue
        fields = {f"{k}_us": int(v) for k, v in FIELD_RE.findall(line)}
        fields["round"] = int(head.group(1))
        fields["d"] = int(head.group(2))
        fields["acc"] = int(head.group(3))
        rows.append(fields)
    return rows


def trace_summary(session: pathlib.Path) -> list[dict]:
    out = []
    for leg_dir in sorted(session.glob("b*-*")):
        trace_path = leg_dir / "trace.txt"
        meta_path = leg_dir / "meta.txt"
        if not (trace_path.exists() and meta_path.exists()):
            continue
        meta = read_meta(meta_path)
        rows = parse_trace(trace_path)[1:]
        if not rows:
            continue
        depth = pinned_depth(meta.get("arm_env", ""))

        def mean(key: str) -> float:
            return trimmed_mean([float(r.get(key, 0)) for r in rows])

        head_us = mean("draft_build_us") - mean("d_pre_us")
        out.append({
            "dir": leg_dir.name,
            "arm": meta["arm_label"],
            "block": int(meta.get("block", -1)),
            "pinned_depth": depth,
            "rounds": len(rows),
            "round_us": mean("round_us"),
            "draft_build_us": mean("draft_build_us"),
            "d_pre_us": mean("d_pre_us"),
            "d_flush_us": mean("d_flush_us"),
            "d_head1_us": mean("d_head1_us"),
            "d_submit1_us": mean("d_submit1_us"),
            "d_chain_us": mean("d_chain_us"),
            "d_submit2_us": mean("d_submit2_us"),
            "head_segment_us": head_us,
            "verify_build_us": mean("verify_build_us"),
            "eval_wall_us": mean("eval_wall_us"),
            "readout_us": mean("readout_us"),
            "commit_us": mean("commit_us"),
            "upkeep_us": mean("upkeep_us"),
            "session_tail_us": mean("readout_us") + mean("commit_us")
            + mean("upkeep_us") + mean("d_pre_us"),
        })
    return out


def build_depth_table(pinned: list[dict], s_fixed: float,
                      h_slope: float) -> list[dict]:
    by_depth: dict[int, list[dict]] = {}
    for leg in pinned:
        by_depth.setdefault(leg["pinned_depth"], []).append(leg)
    table = []
    for depth in sorted(by_depth):
        group = by_depth[depth]
        values = [leg["R_decode_seconds"] for leg in group]
        a_mean = st.fmean([leg["a_accepted_per_round"] for leg in group])
        table.append({
            "D": depth,
            "legs": len(group),
            "R_seconds_mean": st.fmean([leg["R_seconds"] for leg in group]),
            "R_decode_seconds_mean": st.fmean(values),
            "R_decode_seconds_spread": max(values) - min(values),
            "prefill_per_round_seconds": st.fmean(
                [leg["prefill_per_round_seconds"] for leg in group]),
            "rounds": [leg["rounds"] for leg in group],
            "accepted_draft_total": [leg["accepted_draft_total"]
                                     for leg in group],
            "a_accepted_per_round": a_mean,
            "q_proposed_per_round": st.fmean(
                [leg["q_proposed_per_round"] for leg in group]),
            "alpha_accept_fraction": st.fmean(
                [leg["alpha_accept_fraction"] for leg in group]),
            "non_drafting_round_count": [leg["non_drafting_round_count"]
                                         for leg in group],
            "mtp_seconds_per_token": st.fmean(
                [leg["mtp_seconds_per_token"] for leg in group]),
            "realized_acceptance_a_over_D": a_mean / depth if depth else 0.0,
            "implied_uniform_p": implied_uniform_p(a_mean, depth),
            "rejected_per_round": st.fmean(
                [leg["rejected_per_round"] for leg in group]),
            "fit_R_seconds": s_fixed + h_slope * depth,
            "residual_seconds": st.fmean(values) - (s_fixed + h_slope * depth),
            "entry_c": [leg["entry_c"] for leg in group],
            "exit_c": [leg["exit_c"] for leg in group],
        })
    return table


def build_segments(table: list[dict]) -> list[dict]:
    """The marginal price of one more PROPOSED draft between sweep points.

    Measured rather than assumed constant. If these disagree, `h` is not a
    number and `rho = 8h/s` inherits whichever segment the fit weighted most.
    """
    segments = []
    for lo, hi in zip(table, table[1:]):
        span = hi["D"] - lo["D"]
        segments.append({
            "from_D": lo["D"],
            "to_D": hi["D"],
            "delta_R_seconds":
                hi["R_decode_seconds_mean"] - lo["R_decode_seconds_mean"],
            "marginal_seconds_per_draft":
                (hi["R_decode_seconds_mean"]
                 - lo["R_decode_seconds_mean"]) / span,
            "delta_rejected_per_round":
                (hi["q_proposed_per_round"] - hi["a_accepted_per_round"])
                - (lo["q_proposed_per_round"] - lo["a_accepted_per_round"]),
        })
    return segments


def build_width_wall(table: list[dict], segments: list[dict],
                     adaptive: list[dict]) -> dict:
    """Locate the largest width step and price it where the schedule runs.

    Verify width is `1 + D`: the committed primary token plus the proposed
    drafts. The step reported against a width is the price of entering it.
    """
    steps = {seg["to_D"]: seg["marginal_seconds_per_draft"]
             for seg in segments}
    if not steps:
        return {}
    reachable = {depth: value for depth, value in steps.items() if depth <= 7}
    wall_depth = max(reachable, key=lambda d: reachable[d])
    neighbours = [value for depth, value in reachable.items()
                  if abs(depth - wall_depth) == 1]

    histogram: collections.Counter = collections.Counter()
    for leg in adaptive:
        for length, count in leg["draft_length_histogram"].items():
            histogram[int(length)] += count
    rounds = sum(histogram.values())
    at_or_above = sum(count for depth, count in histogram.items()
                      if depth >= wall_depth)

    curve = {row["D"]: row["R_decode_seconds_mean"] for row in table}
    reach_cap = curve.get(7)
    counterfactual = (st.fmean(neighbours) if neighbours else None)
    return {
        "wall_depth_D": wall_depth,
        "wall_verify_width": wall_depth + 1,
        "wall_step_seconds": steps[wall_depth],
        "neighbour_step_seconds": counterfactual,
        "wall_excess_seconds": (steps[wall_depth] - counterfactual
                                if counterfactual is not None else None),
        "shipped_width_cap": 8,
        "cost_of_reaching_depth_7_seconds": (
            reach_cap - curve[0] if reach_cap is not None else None),
        "wall_share_of_reaching_depth_7": (
            steps[wall_depth] / (reach_cap - curve[0])
            if reach_cap is not None else None),
        "adaptive_round_count": rounds,
        "adaptive_draft_length_histogram": dict(sorted(histogram.items())),
        "adaptive_rounds_at_or_above_wall": at_or_above,
        "adaptive_share_at_or_above_wall": (at_or_above / rounds
                                            if rounds else None),
        "steps_by_verify_width": {
            str(depth + 1): value for depth, value in sorted(steps.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=pathlib.Path)
    parser.add_argument("--trace", type=pathlib.Path, default=None)
    parser.add_argument("--extra", type=pathlib.Path, action="append",
                        default=[])
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--include-block-zero", action="store_true")
    args = parser.parse_args()

    legs = load_legs(args.session)
    if not legs:
        raise SystemExit(f"no legs under {args.session}")
    estimate = [leg for leg in legs
                if args.include_block_zero or leg["block"] > 0]
    pinned = [leg for leg in estimate if leg["pinned_depth"] is not None]
    adaptive = [leg for leg in estimate if leg["pinned_depth"] is None]

    depths = [float(leg["pinned_depth"]) for leg in pinned]
    # The assignment's literal estimator, reported because it was named.
    fit_prescribed = ols(depths, [leg["R_seconds"] for leg in pinned])
    # The round cost law, with the per-leg seed prefill removed.
    fit = ols(depths, [leg["R_decode_seconds"] for leg in pinned])
    curve = quadratic_term(depths,
                           [leg["R_decode_seconds"] for leg in pinned])
    curve_prescribed = quadratic_term(depths,
                                      [leg["R_seconds"] for leg in pinned])
    # Same law, but on the parent's own per-round clock with round 0 dropped
    # and both tails trimmed. If the two intercepts disagree, the disagreement
    # is post-prefill warmup and OS stalls, not the depth law.
    fit_trimmed = ols([float(leg["pinned_depth"]) for leg in pinned],
                      [leg["round_us_trimmed"] * 1e-6 for leg in pinned])
    s_fixed, h_slope = fit["intercept"], fit["slope"]
    rho = 8.0 * h_slope / s_fixed

    depth_table = build_depth_table(pinned, s_fixed, h_slope)
    segments = build_segments(depth_table)

    # The kink session fills D in {3,5,6,7}. Folding it in localizes where the
    # marginal price actually steps instead of averaging over a 2->4 or 4->8
    # gap that could hide a wall at one width.
    dense_legs = list(pinned)
    for extra in args.extra:
        dense_legs.extend(
            leg for leg in load_legs(extra)
            if leg["pinned_depth"] is not None
            and (args.include_block_zero or leg["block"] > 0))
    dense_table = build_depth_table(dense_legs, s_fixed, h_slope)
    dense_segments = build_segments(dense_table)
    dense_le7 = [leg for leg in dense_legs if leg["pinned_depth"] <= 7]
    fit_dense = ols([float(leg["pinned_depth"]) for leg in dense_legs],
                    [leg["R_decode_seconds"] for leg in dense_legs])
    fit_dense_le7 = ols([float(leg["pinned_depth"]) for leg in dense_le7],
                        [leg["R_decode_seconds"] for leg in dense_le7])

    # `s` pinned to the measured D=0 round instead of extrapolated, and 8h read
    # as the measured chord from D=0 to D=8. This uses no linearity assumption,
    # so it survives the convexity that the quadratic term reports.
    r0 = next(r["R_decode_seconds_mean"] for r in depth_table if r["D"] == 0)
    r8 = next(r["R_decode_seconds_mean"] for r in depth_table if r["D"] == 8)
    chord = {
        "s_seconds": r0,
        "eight_h_seconds": r8 - r0,
        "h_seconds": (r8 - r0) / 8.0,
        "rho": (r8 - r0) / r0,
        "definition": "s = measured R(0); 8h = measured R(8) - R(0)",
    }

    # `rho = 8h/s` keeps its pre-registered factor 8. Only the estimate of `h`
    # changes between these rows. The D<=7 rows matter because the shipped
    # schedule caps at `segmentedVerifyDepthCap = 7`, so D=8 is a depth the
    # scored policy never proposes.
    def chord_rho(depth: int) -> dict | None:
        row = next((r for r in dense_table if r["D"] == depth), None)
        if row is None:
            return None
        h = (row["R_decode_seconds_mean"] - r0) / depth
        return {"h_seconds": h, "rho": 8.0 * h / r0, "top_D": depth}

    rho_variants = {
        "ols_prescribed_depths_0_8": rho,
        "ols_dense_0_8": 8.0 * fit_dense["slope"] / fit_dense["intercept"],
        "ols_dense_0_7_shipped_cap":
            8.0 * fit_dense_le7["slope"] / fit_dense_le7["intercept"],
        "chord_0_8": chord_rho(8),
        "chord_0_7_shipped_cap": chord_rho(7),
        "trimmed_round_clock": 8.0 * fit_trimmed["slope"]
        / fit_trimmed["intercept"],
    }

    traces = trace_summary(args.trace) if args.trace else []
    head_fit = None
    if len({t["pinned_depth"] for t in traces}) >= 3:
        head_fit = ols([float(t["pinned_depth"]) for t in traces],
                       [t["head_segment_us"] * 1e-6 for t in traces])

    adapt = None
    if adaptive:
        q_adapt = st.fmean([leg["q_proposed_per_round"] for leg in adaptive])
        r_adapt = st.fmean([leg["R_decode_seconds"] for leg in adaptive])
        head_share = None
        verify_share = None
        if head_fit:
            head_cost = head_fit["slope"] * q_adapt
            head_share = head_cost / r_adapt
            verify_share = (h_slope * q_adapt - head_cost) / r_adapt
        # The four shares the assignment asks for, anchored on measurements
        # rather than on the linear fit: the batch-1 round is the measured
        # `R(0)`, the head chain is the traced head fit at the adaptive `q`,
        # the session tail is traced directly, and the extra verify rows are
        # what remains. These sum to 1 by construction, which the OLS split
        # cannot do while the linear model is rejected.
        trace_budget = None
        if head_fit and traces:
            head_cost = (head_fit["intercept"]
                         + head_fit["slope"] * q_adapt)
            tail = st.fmean(t["session_tail_us"] * 1e-6 for t in traces)
            target_batch1 = next(
                row["R_decode_seconds_mean"] for row in depth_table
                if row["D"] == 0)
            extra_rows = r_adapt - target_batch1 - head_cost - tail
            trace_budget = {
                "target_batch_1_seconds": target_batch1,
                "target_batch_1_share": target_batch1 / r_adapt,
                "extra_verify_rows_seconds": extra_rows,
                "extra_verify_rows_share": extra_rows / r_adapt,
                "proposal_head_seconds": head_cost,
                "proposal_head_share": head_cost / r_adapt,
                "session_overhead_seconds": tail,
                "session_overhead_share": tail / r_adapt,
            }

        adapt = {
            "legs": len(adaptive),
            "trace_anchored_budget": trace_budget,
            "q_proposed_per_round": q_adapt,
            "a_accepted_per_round": st.fmean(
                [leg["a_accepted_per_round"] for leg in adaptive]),
            "alpha_accept_fraction": st.fmean(
                [leg["alpha_accept_fraction"] for leg in adaptive]),
            "R_decode_seconds": r_adapt,
            "R_seconds_prescribed": st.fmean(
                [leg["R_seconds"] for leg in adaptive]),
            "prefill_per_round_seconds": st.fmean(
                [leg["prefill_per_round_seconds"] for leg in adaptive]),
            "mtp_seconds_per_token": st.fmean(
                [leg["mtp_seconds_per_token"] for leg in adaptive]),
            "non_drafting_round_count": [leg["non_drafting_round_count"]
                                         for leg in adaptive],
            "fixed_share": s_fixed / r_adapt,
            "drafting_share": (h_slope * q_adapt) / r_adapt,
            "head_share": head_share,
            "marginal_verify_share": verify_share,
            "R0_over_Radapt": s_fixed / r_adapt,
        }

    out = {
        "experiment": "e159-r1-round-budget",
        "harness": "local",
        "official_or_ranked_score": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "session": str(args.session),
        "trace_session": str(args.trace) if args.trace else None,
        "legs": legs,
        "round_time_definition":
            "R_decode = (decode_seconds - seed_prefill_seconds) / rounds",
        "fit_prescribed_includes_prefill": {
            k: v for k, v in fit_prescribed.items() if k != "residuals"},
        "rho_prescribed_includes_prefill":
            8.0 * fit_prescribed["slope"] / fit_prescribed["intercept"],
        "quadratic_term_prescribed": curve_prescribed,
        "fit": {k: v for k, v in fit.items() if k != "residuals"},
        "fit_residuals_seconds": fit["residuals"],
        "fit_trimmed_round_clock": {k: v for k, v in fit_trimmed.items()
                                    if k != "residuals"},
        "rho_trimmed_round_clock": 8.0 * fit_trimmed["slope"]
        / fit_trimmed["intercept"],
        "quadratic_term": curve,
        "linear_model_rejected": abs(curve["t_c2"]) >= 2.0,
        "segment_marginals": segments,
        "dense_sessions": [str(p) for p in args.extra],
        "dense_depth_table": dense_table,
        "dense_segment_marginals": dense_segments,
        "width_wall": build_width_wall(dense_table, dense_segments, adaptive),
        "fit_dense_0_8": {k: v for k, v in fit_dense.items()
                          if k != "residuals"},
        "fit_dense_0_7_shipped_cap": {k: v for k, v in fit_dense_le7.items()
                                      if k != "residuals"},
        "rho_variants": rho_variants,
        "chord_estimate": chord,
        "s_fixed_seconds": s_fixed,
        "h_slope_seconds": h_slope,
        "rho_8h_over_s": rho,
        "rho_se": 8.0 * math.sqrt(
            (fit["se_slope"] / s_fixed) ** 2
            + (h_slope * fit["se_intercept"] / s_fixed ** 2) ** 2),
        "depth_table": depth_table,
        "trace_legs": traces,
        "head_fit": ({k: v for k, v in head_fit.items() if k != "residuals"}
                     if head_fit else None),
        "h_head_seconds": head_fit["slope"] if head_fit else None,
        "t1_marginal_verify_seconds": (h_slope - head_fit["slope"]
                                       if head_fit else None),
        "rho_head_only": (8.0 * head_fit["slope"] / s_fixed
                          if head_fit else None),
        "adaptive_operating_point": adapt,
        "all_tokens_matched": all(leg["all_tokens_matched"] for leg in legs),
        "head_provenance_sha256": sorted(
            {leg["head_provenance_sha256"] for leg in legs}),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")

    print(f"legs {len(legs)}  estimate {len(estimate)}  "
          f"matched {out['all_tokens_matched']}")
    print(f"{'D':>3} {'legs':>4} {'R ms':>9} {'spread ms':>9} {'pre ms':>8} "
          f"{'rounds':>12} "
          f"{'a':>7} {'q':>6} {'alpha':>7} {'p_impl':>7} {'nondraft':>10} {'resid ms':>9}")
    for row in depth_table:
        print(f"{row['D']:>3} {row['legs']:>4} "
              f"{row['R_decode_seconds_mean'] * 1e3:>9.4f} "
              f"{row['R_decode_seconds_spread'] * 1e3:>9.4f} "
              f"{row['prefill_per_round_seconds'] * 1e3:>8.3f} "
              f"{str(row['rounds']):>12} "
              f"{row['a_accepted_per_round']:>7.4f} "
              f"{row['q_proposed_per_round']:>6.3f} "
              f"{row['alpha_accept_fraction']:>7.4f} "
              f"{row['implied_uniform_p']:>7.4f} "
              f"{str(row['non_drafting_round_count']):>10} "
              f"{row['residual_seconds'] * 1e3:>9.4f}")
    print()
    print(f"PRESCRIBED R = decode/rounds, carries the seed prefill: "
          f"s {fit_prescribed['intercept'] * 1e3:.4f} ms  "
          f"h {fit_prescribed['slope'] * 1e3:.4f} ms  "
          f"rho {out['rho_prescribed_includes_prefill']:.4f}  "
          f"c2 {curve_prescribed['t_c2']:.2f} sigma")
    print("--- seed prefill removed below; this is the round cost law ---")
    print(f"s (fixed per round)   {s_fixed * 1e3:.4f} ms "
          f"+- {fit['se_intercept'] * 1e3:.4f}")
    print(f"h (per proposed draft){h_slope * 1e3:.4f} ms "
          f"+- {fit['se_slope'] * 1e3:.4f}")
    print(f"rho = 8h/s            {rho:.4f} +- {out['rho_se']:.4f}")
    print(f"R^2 {fit['r_squared']:.6f}  residual sd "
          f"{fit['residual_sd'] * 1e3:.4f} ms  "
          f"quadratic c2 {curve['c2'] * 1e3:.5f} ms "
          f"({curve['t_c2']:.2f} sigma)")
    print(f"trimmed round clock: s {fit_trimmed['intercept'] * 1e3:.4f} ms  "
          f"h {fit_trimmed['slope'] * 1e3:.4f} ms  "
          f"rho {out['rho_trimmed_round_clock']:.4f}")
    print(f"chord: s = R(0) {chord['s_seconds'] * 1e3:.4f} ms  "
          f"h {chord['h_seconds'] * 1e3:.4f} ms  rho {chord['rho']:.4f}")
    print("segment marginals ms/draft: " + "  ".join(
        f"{seg['from_D']}->{seg['to_D']} "
        f"{seg['marginal_seconds_per_draft'] * 1e3:.3f}"
        for seg in segments))
    if args.extra:
        print()
        print("dense sweep with the kink session folded in")
        print(f"{'D':>3} {'legs':>4} {'R ms':>9} {'spread ms':>9} "
              f"{'a':>7} {'q':>6} {'alpha':>7} {'step ms':>8}")
        step = {seg["to_D"]: seg["marginal_seconds_per_draft"]
                for seg in dense_segments}
        for row in dense_table:
            marginal = step.get(row["D"])
            print(f"{row['D']:>3} {row['legs']:>4} "
                  f"{row['R_decode_seconds_mean'] * 1e3:>9.4f} "
                  f"{row['R_decode_seconds_spread'] * 1e3:>9.4f} "
                  f"{row['a_accepted_per_round']:>7.4f} "
                  f"{row['q_proposed_per_round']:>6.3f} "
                  f"{row['alpha_accept_fraction']:>7.4f} "
                  + (f"{marginal * 1e3:>8.3f}" if marginal is not None
                     else f"{'-':>8}"))
        print("rho = 8h/s by how h is estimated:")
        for key, value in rho_variants.items():
            if isinstance(value, dict):
                print(f"  {key:<26} {value['rho']:.4f} "
                      f"(h {value['h_seconds'] * 1e3:.4f} ms)")
            else:
                print(f"  {key:<26} {value:.4f}")
        wall = out["width_wall"]
        if wall:
            print(f"width wall: entering verify width "
                  f"{wall['wall_verify_width']} costs "
                  f"{wall['wall_step_seconds'] * 1e3:.3f} ms against "
                  f"{wall['neighbour_step_seconds'] * 1e3:.3f} ms for its "
                  f"neighbours, "
                  f"{wall['wall_share_of_reaching_depth_7'] * 100:.1f} % of "
                  f"the whole cost of reaching depth 7")
            print(f"  the shipped schedule runs "
                  f"{wall['adaptive_rounds_at_or_above_wall']} of "
                  f"{wall['adaptive_round_count']} rounds at or above it "
                  f"({wall['adaptive_share_at_or_above_wall'] * 100:.1f} %)")
    if head_fit:
        print(f"h_head (sync-head trace) {head_fit['slope'] * 1e3:.4f} ms "
              f"+- {head_fit['se_slope'] * 1e3:.4f}  "
              f"t1 = {out['t1_marginal_verify_seconds'] * 1e3:.4f} ms")
    if adapt:
        print(f"adaptive q {adapt['q_proposed_per_round']:.4f}  "
              f"a {adapt['a_accepted_per_round']:.4f}  "
              f"R {adapt['R_decode_seconds'] * 1e3:.4f} ms  "
              f"fixed {adapt['fixed_share'] * 100:.1f} %  "
              f"drafting {adapt['drafting_share'] * 100:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
