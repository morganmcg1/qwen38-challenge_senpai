#!/usr/bin/env python3
"""E59 rung 4 stage A: score the whole-table M=5 cell palindrome.

Rung 3 measured the same mechanism in ISOLATED builds, where `t55` also carried
a +38 register entry-point dose. This stage measures it in the shipped table,
where the census says `t55` moves the entry point by +2 and `m5_rbx` by 0, so
the number transfers to a real leg.

The advisor's kill rule for `t55` lives here, not on the leg: this host's width
histogram is not the ranked one, so a small leg effect cannot refute a
mechanism that the ranked mixture weights four times more heavily.

  python3 research/e59_rung4_cells.py --out research/e59-artifacts/e59-rung4-cells.json

Exit code 0 means the M=5 cell gain cleared the kill rule for at least one
route. Exit code 2 means both routes fell under it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e46_analyze import load  # noqa: E402
from e49_analyze import contrast  # noqa: E402
from e54_bandwidth import bandwidth_rows  # noqa: E402
from e59_ols import arm_contrast, fit_arm_position  # noqa: E402
import e54_analyze  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
e54_analyze.LEGS = REPO / ".mlxfast-private/e59-legs"

TAG_PREFIX = "e59-r4c-"
TREATED_WIDTH = 5

# The advisor's unified cost model. Cell cost is the sum over working groups of
# 1 / bw(NA_g), with bw from E54 for NA <= 5 and from askeladd's E61 rung 1 for
# NA >= 6. At M=5 the shipped table runs [3, 2] and `t55` runs [5].
MODEL_PREDICTED_PCT = -30.09
# Rung 3 measured the same contrast in isolated builds.
RUNG3_ISOLATED_PCT = -20.010

# `t55` is killed if the whole-table M=5 cell gain is under this, after the
# untreated widths are used as the instrument bar.
KILL_RULE_PCT = -2.0


def legs_for_stage() -> list[dict]:
    return [
        leg
        for leg in e54_analyze.leg_records()
        if leg["tag"].startswith(TAG_PREFIX) and leg["status"] == "ok"
    ]


def by_arm(legs: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for leg in legs:
        arms.setdefault(leg["arm"], []).append(leg)
    return arms


def cell_bandwidth(legs: list[dict]) -> dict[int, dict]:
    """Achieved rate per width, weighted by calls per verify.

    Two numbers per width. `modelled_gbps` charges every working group for a
    full weight pass, which is what the cost model assumes; above the measured
    stream peak it is proof that the later groups hit cache instead of DRAM.
    `single_pass_gbps` charges the weight matrix once, so it compares forms that
    stream it a different number of times on the same footing.
    """
    per_width: dict[int, list[dict]] = {}
    for leg in legs:
        data, _ = load(leg["tag"])
        for row in bandwidth_rows(data):
            if row["bandwidth_reliable"]:
                per_width.setdefault(row["m"], []).append(row)

    out: dict[int, dict] = {}
    for m, rows in sorted(per_width.items()):
        secs = sum(r["calls_per_verify"] * r["seconds_per_call"] for r in rows)
        modelled = sum(r["calls_per_verify"] * r["aggregate_bytes"] for r in rows)
        single = sum(r["calls_per_verify"] * r["weight_bytes"] for r in rows)
        lone = [g for r in rows for g in r["per_group_gbps"] if r["lone_group"]]
        out[m] = {
            "group_na": sorted({tuple(r["group_na"]) for r in rows}),
            "working_groups": sorted({r["working_groups"] for r in rows}),
            "kernels": sorted({r["kernel"] for r in rows}),
            "seconds_per_verify": secs,
            "modelled_gbps": modelled / secs / 1e9,
            "single_pass_gbps": single / secs / 1e9,
            "lone_group_gbps_mean": (sum(lone) / len(lone)) if lone else None,
            "samples": len(rows),
        }
    return out


def cell_regression(legs: list[dict], width: int, base_arm: str = "shipped") -> dict:
    """`t_ms(width) ~ arm + leg_position`, the campaign-standard estimator.

    A palindrome makes arm and linear position orthogonal, so these arm
    contrasts reproduce the plain means. The fit is still worth reporting: the
    position coefficient prices the session drift the palindrome cancelled, and
    the residual sd is a second, independent read of the replicate noise.
    """
    observations = [
        {"arm": leg["arm"], "position": position, "value": leg["t_ms"][width],
         "label": leg["tag"]}
        for position, leg in enumerate(legs, start=1)
        if width in leg["t_ms"]
    ]
    if not observations:
        return {"fitted": False, "reason": f"no leg reports width {width}"}
    fit = fit_arm_position(observations, base_arm)
    if not fit.get("fitted"):
        return fit
    arms = sorted({obs["arm"] for obs in observations})
    fit["contrasts"] = {
        f"{left}_vs_{right}": arm_contrast(fit, left, right)
        for left in arms for right in arms if left != right
    }
    return fit


def route_report(arms: dict, treated_arm: str, control_arm: str) -> dict | None:
    if treated_arm not in arms or control_arm not in arms:
        return None
    widths = sorted(set(arms[control_arm][0]["t_ms"]) & set(arms[treated_arm][0]["t_ms"]))
    rows = contrast(arms[treated_arm], arms[control_arm], widths)
    if TREATED_WIDTH not in rows:
        return None

    controls = {m: r["delta_pct"] for m, r in rows.items() if m != TREATED_WIDTH}
    control_bar = max((abs(v) for v in controls.values()), default=0.0)
    treated = rows[TREATED_WIDTH]
    spread = (
        100.0 * treated["mde_ms"] / treated["control_ms"]
        if treated["mde_ms"] is not None and treated["control_ms"]
        else None
    )
    bar = max([control_bar] + ([spread] if spread is not None else []))
    delta = treated["delta_pct"]
    return {
        "treated_arm": treated_arm,
        "control_arm": control_arm,
        "m5_delta_pct": delta,
        "m5_delta_ms": treated["delta_ms"],
        "m5_control_ms": treated["control_ms"],
        "m5_treated_ms": treated["treated_ms"],
        "m5_replicate_spread_pct": spread,
        "untreated_width_pct": controls,
        "worst_untreated_width_pct": control_bar,
        "decision_bar_pct": bar,
        "clears_bar": bool(abs(delta) > bar),
        "clears_kill_rule": bool(delta <= KILL_RULE_PCT and abs(delta) > bar),
        "per_width": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-rung4-cells.json")
    args = ap.parse_args()

    legs = legs_for_stage()
    if not legs:
        raise SystemExit(f"no readable legs tagged {TAG_PREFIX}*")
    arms = by_arm(legs)

    routes = {
        "t55_vs_shipped": route_report(arms, "t55", "shipped"),
        "rbx_vs_shipped": route_report(arms, "m5_rbx", "shipped"),
        "t55_vs_rbx": route_report(arms, "t55", "m5_rbx"),
    }

    bandwidth = {arm: cell_bandwidth(arm_legs) for arm, arm_legs in arms.items()}
    widths = sorted(legs[0]["t_ms"])
    regressions = {m: cell_regression(legs, m) for m in widths}

    t55 = routes["t55_vs_shipped"]
    realisation = None
    if t55:
        realisation = t55["m5_delta_pct"] / MODEL_PREDICTED_PCT

    artifact = {
        "experiment": "E59",
        "stage": "rung4-cells",
        "tag_prefix": TAG_PREFIX,
        "order": [leg["tag"] for leg in legs],
        "arms_measured": {arm: [leg["tag"] for leg in v] for arm, v in arms.items()},
        "model_predicted_m5_pct": MODEL_PREDICTED_PCT,
        "rung3_isolated_m5_pct": RUNG3_ISOLATED_PCT,
        "kill_rule_pct": KILL_RULE_PCT,
        "routes": routes,
        "bandwidth": bandwidth,
        "regression_t_ms_by_arm_and_position": regressions,
        "t55_model_realisation": realisation,
        "legs": legs,
    }

    survivors = [
        name
        for name, r in routes.items()
        if name != "t55_vs_rbx" and r and r["clears_kill_rule"]
    ]
    artifact["routes_clearing_kill_rule"] = survivors
    artifact["all_passed"] = bool(survivors)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    for name, r in routes.items():
        if not r:
            print(f"{name:18s} MISSING")
            continue
        print(
            f"{name:18s} M=5 {r['m5_delta_pct']:+7.3f} % "
            f"bar {r['decision_bar_pct']:5.3f} % "
            f"worst-untreated {r['worst_untreated_width_pct']:5.3f} % "
            f"{'CLEARS' if r['clears_bar'] else 'INSIDE BAR'}"
        )
    for arm in sorted(bandwidth):
        b = bandwidth[arm].get(TREATED_WIDTH)
        if b:
            print(
                f"bw {arm:10s} M=5 groups {b['group_na']} "
                f"single-pass {b['single_pass_gbps']:6.1f} GB/s "
                f"modelled {b['modelled_gbps']:6.1f} GB/s"
            )
    fit = regressions.get(TREATED_WIDTH, {})
    if fit.get("fitted"):
        print(
            f"\nregression  t_ms(M={TREATED_WIDTH}) ~ arm + leg_position "
            f"(n={fit['n']}, residual dof={fit['residual_dof']}, "
            f"residual sd={fit['residual_sd_pct_of_base']:.4f} % of base)"
        )
        print(f"  drift {fit['position_drift_pct_of_base_per_leg']:+.4f} % per leg")
        for name, term in fit["terms"].items():
            if not name.startswith("arm["):
                continue
            t = term["t"]
            print(
                f"  {name:18s} {term['estimate_pct_of_base']:+8.4f} % of base"
                + (f"  t={t:+.2f}" if t is not None else "")
            )
        gap = fit["contrasts"].get("m5_rbx_vs_t55")
        if gap and gap["available"]:
            print(
                f"  m5_rbx - t55      {gap['estimate_pct_of_base']:+8.4f} % "
                f"+/- {gap['std_error_pct_of_base']:.4f} %  t={gap['t']:+.2f}"
            )
    if realisation is not None:
        print(f"\nt55 realisation vs model {MODEL_PREDICTED_PCT} % = {realisation:.3f}")
    print(f"routes clearing the kill rule: {survivors or 'none'} -> {out}")
    return 0 if survivors else 2


if __name__ == "__main__":
    sys.exit(main())
