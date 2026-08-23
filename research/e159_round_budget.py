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
            "R_seconds": decode_seconds / rounds,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=pathlib.Path)
    parser.add_argument("--trace", type=pathlib.Path, default=None)
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

    fit = ols([float(leg["pinned_depth"]) for leg in pinned],
              [leg["R_seconds"] for leg in pinned])
    curve = quadratic_term([float(leg["pinned_depth"]) for leg in pinned],
                           [leg["R_seconds"] for leg in pinned])
    # Same law, but on the parent's own per-round clock with round 0 dropped
    # and both tails trimmed. If the two intercepts disagree, the disagreement
    # is post-prefill warmup and OS stalls, not the depth law.
    fit_trimmed = ols([float(leg["pinned_depth"]) for leg in pinned],
                      [leg["round_us_trimmed"] * 1e-6 for leg in pinned])
    s_fixed, h_slope = fit["intercept"], fit["slope"]
    rho = 8.0 * h_slope / s_fixed

    by_depth: dict[int, list[dict]] = {}
    for leg in pinned:
        by_depth.setdefault(leg["pinned_depth"], []).append(leg)
    depth_table = []
    for depth in sorted(by_depth):
        group = by_depth[depth]
        values = [leg["R_seconds"] for leg in group]
        depth_table.append({
            "D": depth,
            "legs": len(group),
            "R_seconds_mean": st.fmean(values),
            "R_seconds_spread": max(values) - min(values),
            "rounds": [leg["rounds"] for leg in group],
            "accepted_draft_total": [leg["accepted_draft_total"]
                                     for leg in group],
            "a_accepted_per_round": st.fmean(
                [leg["a_accepted_per_round"] for leg in group]),
            "q_proposed_per_round": st.fmean(
                [leg["q_proposed_per_round"] for leg in group]),
            "alpha_accept_fraction": st.fmean(
                [leg["alpha_accept_fraction"] for leg in group]),
            "non_drafting_round_count": [leg["non_drafting_round_count"]
                                         for leg in group],
            "mtp_seconds_per_token": st.fmean(
                [leg["mtp_seconds_per_token"] for leg in group]),
            "realized_acceptance_a_over_D": (
                st.fmean([leg["a_accepted_per_round"] for leg in group])
                / depth if depth else 0.0),
            "implied_uniform_p": implied_uniform_p(
                st.fmean([leg["a_accepted_per_round"] for leg in group]),
                depth),
            "rejected_per_round": st.fmean(
                [leg["rejected_per_round"] for leg in group]),
            "fit_R_seconds": s_fixed + h_slope * depth,
            "residual_seconds": st.fmean(values) - (s_fixed + h_slope * depth),
            "entry_c": [leg["entry_c"] for leg in group],
            "exit_c": [leg["exit_c"] for leg in group],
        })

    # The marginal price of one more PROPOSED draft, measured between adjacent
    # sweep points instead of assumed constant. If these disagree, `h` is not a
    # number and `rho = 8h/s` inherits whichever segment the fit happened to
    # weight most.
    segments = []
    for lo, hi in zip(depth_table, depth_table[1:]):
        span = hi["D"] - lo["D"]
        segments.append({
            "from_D": lo["D"],
            "to_D": hi["D"],
            "delta_R_seconds": hi["R_seconds_mean"] - lo["R_seconds_mean"],
            "marginal_seconds_per_draft":
                (hi["R_seconds_mean"] - lo["R_seconds_mean"]) / span,
            "delta_rejected_per_round":
                (hi["q_proposed_per_round"] - hi["a_accepted_per_round"])
                - (lo["q_proposed_per_round"] - lo["a_accepted_per_round"]),
        })

    # `s` pinned to the measured D=0 round instead of extrapolated, and 8h read
    # as the measured chord from D=0 to D=8. This uses no linearity assumption,
    # so it survives the convexity that the quadratic term reports.
    r0 = next(r["R_seconds_mean"] for r in depth_table if r["D"] == 0)
    r8 = next(r["R_seconds_mean"] for r in depth_table if r["D"] == 8)
    chord = {
        "s_seconds": r0,
        "eight_h_seconds": r8 - r0,
        "h_seconds": (r8 - r0) / 8.0,
        "rho": (r8 - r0) / r0,
        "definition": "s = measured R(0); 8h = measured R(8) - R(0)",
    }

    traces = trace_summary(args.trace) if args.trace else []
    head_fit = None
    if len({t["pinned_depth"] for t in traces}) >= 3:
        head_fit = ols([float(t["pinned_depth"]) for t in traces],
                       [t["head_segment_us"] * 1e-6 for t in traces])

    adapt = None
    if adaptive:
        q_adapt = st.fmean([leg["q_proposed_per_round"] for leg in adaptive])
        r_adapt = st.fmean([leg["R_seconds"] for leg in adaptive])
        head_share = None
        verify_share = None
        if head_fit:
            head_cost = head_fit["slope"] * q_adapt
            head_share = head_cost / r_adapt
            verify_share = (h_slope * q_adapt - head_cost) / r_adapt
        adapt = {
            "legs": len(adaptive),
            "q_proposed_per_round": q_adapt,
            "a_accepted_per_round": st.fmean(
                [leg["a_accepted_per_round"] for leg in adaptive]),
            "alpha_accept_fraction": st.fmean(
                [leg["alpha_accept_fraction"] for leg in adaptive]),
            "R_seconds": r_adapt,
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
        "fit": {k: v for k, v in fit.items() if k != "residuals"},
        "fit_residuals_seconds": fit["residuals"],
        "fit_trimmed_round_clock": {k: v for k, v in fit_trimmed.items()
                                    if k != "residuals"},
        "rho_trimmed_round_clock": 8.0 * fit_trimmed["slope"]
        / fit_trimmed["intercept"],
        "quadratic_term": curve,
        "linear_model_rejected": abs(curve["t_c2"]) >= 2.0,
        "segment_marginals": segments,
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
    print(f"{'D':>3} {'legs':>4} {'R ms':>9} {'spread ms':>9} {'rounds':>12} "
          f"{'a':>7} {'q':>6} {'alpha':>7} {'p_impl':>7} {'nondraft':>10} {'resid ms':>9}")
    for row in depth_table:
        print(f"{row['D']:>3} {row['legs']:>4} "
              f"{row['R_seconds_mean'] * 1e3:>9.4f} "
              f"{row['R_seconds_spread'] * 1e3:>9.4f} "
              f"{str(row['rounds']):>12} "
              f"{row['a_accepted_per_round']:>7.4f} "
              f"{row['q_proposed_per_round']:>6.3f} "
              f"{row['alpha_accept_fraction']:>7.4f} "
              f"{row['implied_uniform_p']:>7.4f} "
              f"{str(row['non_drafting_round_count']):>10} "
              f"{row['residual_seconds'] * 1e3:>9.4f}")
    print()
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
    if head_fit:
        print(f"h_head (sync-head trace) {head_fit['slope'] * 1e3:.4f} ms "
              f"+- {head_fit['se_slope'] * 1e3:.4f}  "
              f"t1 = {out['t1_marginal_verify_seconds'] * 1e3:.4f} ms")
    if adapt:
        print(f"adaptive q {adapt['q_proposed_per_round']:.4f}  "
              f"a {adapt['a_accepted_per_round']:.4f}  "
              f"R {adapt['R_seconds'] * 1e3:.4f} ms  "
              f"fixed {adapt['fixed_share'] * 100:.1f} %  "
              f"drafting {adapt['drafting_share'] * 100:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
