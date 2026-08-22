#!/usr/bin/env python3
"""E130 rung 11, F16: read the wired-slack ladder.

Four arms, one binary, differing only by
``DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB``::

    s64    the rung 10a anchor, measured there at -0.1968 % against s512
    s512   the currently shipped value
    s1024  untimed before this session
    s2048  untimed before this session, and the bound-C hard ceiling

MODEL. ``y ~ arm + leg_index``. Rung 10a's palindrome made the trend exactly
orthogonal to every arm contrast, so the normal equations separated. The rung
11 order is a palindrome over the first eight legs plus one pre-registered
random permutation over the last four, which buys degrees of freedom at the
cost of that orthogonality. So this reader solves the normal equations
properly and takes every contrast standard error from the inverted
information matrix rather than assuming balance.

THE HEADLINE IS ABSOLUTE CANDIDATE SECONDS PER TOKEN. Finding 160: the local
ratio adds 0.4788 % of serial noise to a 0.0498 % measurement, a 9.6x variance
penalty, on an estimand the ranked score does not use. The serial channel and
the ratio are fitted too, but only as reported controls.

PRE-REGISTERED. The linear-in-admitted-bytes model and the saturating null
were written into PR 130 interim 17 before any leg ran, together with the
decision rule and the seed of the random permutation.

Usage
-----
    python3 research/e130_rung11_ladder.py --prefix e130-r11 \
        --out research/e130-artifacts/rung11-slack-ladder.json --wandb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ORDER = [
    "s64", "s512", "s1024", "s2048",
    "s2048", "s1024", "s512", "s64",
    "s2048", "s512", "s64", "s1024",
]
ARMS = ("s64", "s512", "s1024", "s2048")
SLACK_MB = {"s64": 64, "s512": 512, "s1024": 1024, "s2048": 2048}

# Two-sided 95 % Student t. Only the degrees of freedom this design can produce.
T_CRIT_95 = {
    1: 12.7062, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
}

# F16 section 3. Percent change in candidate seconds per token, predicted by
# the linear-in-admitted-bytes model. The saturating null predicts 0.0 for all.
PREREGISTERED_LINEAR = {
    ("s512", "s1024"): -0.225,
    ("s1024", "s2048"): -0.450,
    ("s512", "s2048"): -0.675,
    ("s64", "s2048"): -0.872,
}


def read_meta(path: Path) -> dict:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def swap_fields(blob: str) -> dict:
    return {k: int(v) for k, v in re.findall(r"([a-z_]+)=([0-9]+)", blob or "")}


def read_leg(root: Path, prefix: str, index: int, arm: str) -> dict:
    tag = f"{prefix}-{index:02d}-{arm}"
    out = root / tag
    meta = read_meta(out / "meta.txt")
    leg = {
        "tag": tag,
        "index": index,
        "arm": arm,
        "slack_mb": SLACK_MB[arm],
        "exit": meta.get("exit"),
        "wired_residency_active": meta.get("wired_residency_active"),
        "wired_outcome_line": meta.get("wired_outcome_line"),
        "wired_clamped_count": int(meta.get("wired_clamped_count", "0") or 0),
        "wired_apply_failures": int(meta.get("wired_apply_failures", "0") or 0),
        "wired_slack_mb": meta.get("wired_slack_mb"),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "worker_sha256": meta.get("worker_sha256"),
        "base_sha": meta.get("base_sha"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
    }
    entry = swap_fields(meta.get("swap_entry", ""))
    exit_ = swap_fields(meta.get("swap_exit", ""))
    leg["swap_delta"] = {
        key: exit_.get(key, 0) - entry.get(key, 0)
        for key in sorted(set(entry) | set(exit_))
    }
    leg["swapped"] = leg["swap_delta"].get("swapouts", 0) > 0

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        leg["mtp_seconds_per_token"] = metrics.get("mtp_seconds_per_token")
        leg["serial_seconds_per_token"] = metrics.get("serial_seconds_per_token")
        leg["mtp_decode_speedup"] = metrics.get("mtp_decode_speedup")
        leg["all_tokens_matched"] = metrics.get("all_tokens_matched")
        leg["decode_tokens"] = metrics.get("decode_tokens")
    return leg


def invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan with partial pivoting. The design is 5x5, so this is cheap."""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise SystemExit(f"singular design at column {col}: an arm has no legs")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def fit(legs: list[dict], field: str) -> dict | None:
    """Least squares for ``y = mean(arm) + slope * (leg_index - centre)``."""
    used = [leg for leg in legs if leg.get(field) is not None]
    present = [arm for arm in ARMS if any(leg["arm"] == arm for leg in used)]
    if len(present) < 2:
        return None

    centre = sum(leg["index"] for leg in used) / len(used)
    columns = list(present) + ["trend"]

    def design_row(leg: dict) -> list[float]:
        row = [1.0 if leg["arm"] == arm else 0.0 for arm in present]
        row.append(leg["index"] - centre)
        return row

    rows = [design_row(leg) for leg in used]
    values = [float(leg[field]) for leg in used]
    k = len(columns)
    df = len(used) - k
    if df < 1:
        return None

    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * y for r, y in zip(rows, values)) for i in range(k)]
    inv = invert(xtx)
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]

    fitted = [sum(b * v for b, v in zip(beta, r)) for r in rows]
    residuals = [y - f for y, f in zip(values, fitted)]
    sse = sum(r * r for r in residuals)
    sigma2 = sse / df
    sigma = sigma2 ** 0.5
    grand = sum(values) / len(values)

    # How much of the arm-means-only residual the single trend column removes.
    means = {
        arm: sum(y for y, leg in zip(values, used) if leg["arm"] == arm)
        / sum(1 for leg in used if leg["arm"] == arm)
        for arm in present
    }
    ss_arm_only = sum((y - means[leg["arm"]]) ** 2 for y, leg in zip(values, used))

    return {
        "n": len(used),
        "df": df,
        "columns": columns,
        "arm_means_adjusted": {arm: beta[i] for i, arm in enumerate(present)},
        "arm_means_raw": means,
        "slope_per_leg": beta[-1],
        "slope_pct_per_leg": 100.0 * beta[-1] / grand,
        "residual_sd": sigma,
        "residual_sd_pct": 100.0 * sigma / grand,
        "trend_share_of_arm_only_residual":
            1.0 - sse / ss_arm_only if ss_arm_only > 0 else float("nan"),
        "grand_mean": grand,
        "residuals": {leg["tag"]: r for leg, r in zip(used, residuals)},
        "_inv": inv,
        "_sigma2": sigma2,
        "_present": present,
    }


def contrast(model: dict, lo: str, hi: str) -> dict:
    """Signed effect of moving from ``lo`` to ``hi``, in percent of ``lo``.

    The standard error comes from the inverted information matrix, so it stays
    correct even though the last four legs break the palindrome's balance.
    """
    present = model["_present"]
    if lo not in present or hi not in present:
        return {"from": lo, "to": hi, "usable": False}
    k = len(model["columns"])
    vector = [0.0] * k
    vector[present.index(hi)] = 1.0
    vector[present.index(lo)] = -1.0
    inv = model["_inv"]
    variance = model["_sigma2"] * sum(
        vector[i] * inv[i][j] * vector[j] for i in range(k) for j in range(k)
    )
    se = variance ** 0.5
    base = model["arm_means_adjusted"][lo]
    delta = model["arm_means_adjusted"][hi] - base
    pct = 100.0 * delta / base
    se_pct = 100.0 * se / base
    t_crit = T_CRIT_95.get(model["df"], 1.96)
    half = t_crit * se_pct
    return {
        "from": lo,
        "to": hi,
        "usable": True,
        "delta": delta,
        "pct": pct,
        "se_pct": se_pct,
        "t": pct / se_pct if se_pct else float("nan"),
        "t_crit_95": t_crit,
        "ci95_pct": [pct - half, pct + half],
        "significant": abs(pct) > half,
    }


def decision(pct: float | None) -> str:
    """The F16 decision rule on the s512 -> s2048 contrast, applied verbatim."""
    if pct is None:
        return "undecided: the s512 to s2048 contrast is missing"
    if pct < -0.35:
        return ("ship the ladder argmax; the linear-in-admitted-bytes model is "
                "confirmed")
    if pct < -0.05:
        return "ship the ladder argmax; the response is sub-linear"
    return ("the linear model is refuted; 512 ships unchanged and the ceiling "
            "is now measured rather than guessed")


def safety(legs: list[dict], model: dict | None) -> dict:
    """F16 section 5. Every one of these must stay clean or the ladder stops."""
    clamped = [leg["tag"] for leg in legs if leg["wired_clamped_count"]]
    failures = [leg["tag"] for leg in legs if leg["wired_apply_failures"]]
    inactive = [leg["tag"] for leg in legs
                if leg.get("wired_residency_active") != "true"]
    swapped = [leg["tag"] for leg in legs if leg.get("swapped")]

    # "any leg slower than its s1024 counterpart by more than 2 sigma"
    regressions = []
    if model:
        s1024 = model["arm_means_adjusted"].get("s1024")
        sigma = model["residual_sd"]
        if s1024 and sigma:
            for leg in legs:
                if leg["arm"] != "s2048" or leg.get("mtp_seconds_per_token") is None:
                    continue
                excess = leg["mtp_seconds_per_token"] - s1024
                if excess > 2.0 * sigma:
                    regressions.append({
                        "tag": leg["tag"],
                        "excess_seconds_per_token": excess,
                        "excess_sigma": excess / sigma,
                    })
    return {
        "clamp_bound_on_any_leg": bool(clamped),
        "clamped_legs": clamped,
        "wired_apply_failed_on_any_leg": bool(failures),
        "apply_failure_legs": failures,
        "wiring_inactive_legs": inactive,
        "swapped_legs": swapped,
        "s2048_legs_slower_than_s1024_by_2sigma": regressions,
        "all_clear": not (clamped or failures or inactive or swapped
                          or regressions),
    }


def thermal(legs: list[dict]) -> dict:
    entries = [leg["gpu_temp_entry_c"] for leg in legs
               if leg.get("gpu_temp_entry_c") is not None]
    exits = [leg["gpu_temp_exit_c"] for leg in legs
             if leg.get("gpu_temp_exit_c") is not None]
    return {
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "entry_temp_c": entries,
        "entry_temp_spread_c": max(entries) - min(entries) if entries else None,
        "exit_temp_c": exits,
        "exit_temp_spread_c": max(exits) - min(exits) if exits else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e130-r11")
    ap.add_argument("--root", type=Path, default=Path("research/out"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    legs = [read_leg(args.root, args.prefix, i + 1, arm)
            for i, arm in enumerate(ORDER)]

    channels = {
        "candidate_mtp_seconds_per_token": "mtp_seconds_per_token",
        "serial_seconds_per_token": "serial_seconds_per_token",
        "local_ratio_mtp_decode_speedup": "mtp_decode_speedup",
    }
    models = {name: fit(legs, field) for name, field in channels.items()}
    primary = models["candidate_mtp_seconds_per_token"]

    pairs = [("s64", "s512"), ("s512", "s1024"), ("s1024", "s2048"),
             ("s512", "s2048"), ("s64", "s2048"), ("s64", "s1024")]

    report = {
        "experiment": "e130-rung11-slack-ladder",
        "harness": "local",
        "model": "arm means + one linear session trend",
        "headline_channel": "candidate_mtp_seconds_per_token",
        "headline_note": (
            "absolute candidate seconds per token; the local ratio is a "
            "control only (finding 160)"
        ),
        "order": ORDER,
        "permutation_seed_material": "e130-rung11-slack-ladder-F16",
        "permutation_seed": 14383609076371482244,
        "legs": legs,
        "thermal": thermal(legs),
        "one_binary_served_every_arm":
            len({leg["worker_sha256"] for leg in legs
                 if leg.get("worker_sha256")}) == 1,
        "exactness_all_legs":
            all(leg.get("all_tokens_matched") is True for leg in legs
                if "all_tokens_matched" in leg),
        "channels": {},
    }
    report["safety"] = safety(legs, primary)

    for name, model in models.items():
        if model is None:
            report["channels"][name] = {"usable": False}
            continue
        block = {k: v for k, v in model.items() if not k.startswith("_")}
        block["contrasts"] = {f"{lo}_to_{hi}": contrast(model, lo, hi)
                              for lo, hi in pairs}
        report["channels"][name] = block

    if primary:
        table = []
        for (lo, hi), predicted in PREREGISTERED_LINEAR.items():
            got = report["channels"]["candidate_mtp_seconds_per_token"][
                "contrasts"].get(f"{lo}_to_{hi}", {})
            table.append({
                "contrast": f"{lo}_to_{hi}",
                "linear_model_predicts_pct": predicted,
                "saturating_null_predicts_pct": 0.0,
                "measured_pct": got.get("pct"),
                "ci95_pct": got.get("ci95_pct"),
                "linear_prediction_inside_ci":
                    (got.get("ci95_pct") is not None
                     and got["ci95_pct"][0] <= predicted <= got["ci95_pct"][1]),
                "null_inside_ci":
                    (got.get("ci95_pct") is not None
                     and got["ci95_pct"][0] <= 0.0 <= got["ci95_pct"][1]),
            })
        report["preregistered_test"] = table

        means = primary["arm_means_adjusted"]
        argmax = min(means, key=means.get)
        headline = report["channels"]["candidate_mtp_seconds_per_token"][
            "contrasts"].get("s512_to_s2048", {})
        report["ladder_argmax_arm"] = argmax
        report["ladder_argmax_slack_mb"] = SLACK_MB[argmax]
        report["s512_to_s2048_pct"] = headline.get("pct")
        report["decision"] = decision(headline.get("pct"))

    print(f"=== e130 rung 11 slack ladder: {args.prefix} ===")
    print(f"  legs with a score  "
          f"{sum(1 for leg in legs if leg.get('mtp_seconds_per_token'))}/12")
    print(f"  one binary         {report['one_binary_served_every_arm']}")
    print(f"  exact all legs     {report['exactness_all_legs']}")
    t = report["thermal"]
    print(f"  entry temp spread  {t['entry_temp_spread_c']} C "
          f"(cool_gate_passed_real_gate=false, gate_qualified_for_timing=false)")
    s = report["safety"]
    print(f"  SAFETY all_clear   {s['all_clear']}")
    for key in ("clamped_legs", "apply_failure_legs", "wiring_inactive_legs",
                "swapped_legs"):
        if s[key]:
            print(f"    {key}: {s[key]}")
    if s["s2048_legs_slower_than_s1024_by_2sigma"]:
        print(f"    s2048 regressions: "
              f"{s['s2048_legs_slower_than_s1024_by_2sigma']}")

    for name, block in report["channels"].items():
        if not block.get("df"):
            print(f"\n=== {name}: not usable ===")
            continue
        print(f"\n=== {name} ===")
        print(f"  df {block['df']}  resid sd {block['residual_sd']:.6e}"
              f" = {block['residual_sd_pct']:.4f} %"
              f"  slope {block['slope_pct_per_leg']:+.4f} %/leg"
              f"  (trend removes "
              f"{100 * block['trend_share_of_arm_only_residual']:.2f} %)")
        for arm in ARMS:
            if arm in block["arm_means_adjusted"]:
                print(f"  mean {arm:>6}  {block['arm_means_adjusted'][arm]:.8f}")
        for key, c in block["contrasts"].items():
            if not c.get("usable"):
                continue
            print(f"  {key:<16} {c['pct']:+.4f} %  t={c['t']:+.2f}"
                  f"  95% CI [{c['ci95_pct'][0]:+.4f}, {c['ci95_pct'][1]:+.4f}]"
                  f"  {'SIG' if c['significant'] else 'ns'}")

    if "preregistered_test" in report:
        print("\n=== pre-registered test (F16 section 3) ===")
        for row in report["preregistered_test"]:
            measured = row["measured_pct"]
            got = f"{measured:+.4f}" if measured is not None else "  n/a "
            print(f"  {row['contrast']:<16} linear {row['linear_model_predicts_pct']:+.3f}"
                  f"  null 0.000  measured {got}"
                  f"  linear_in_ci={row['linear_prediction_inside_ci']}"
                  f"  null_in_ci={row['null_inside_ci']}")
        print(f"\n  argmax   {report['ladder_argmax_arm']} "
              f"({report['ladder_argmax_slack_mb']} MiB)")
        print(f"  DECISION {report['decision']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130r11",
            name="e130-rung11-slack-ladder",
            resume="allow",
            config={
                "experiment": "e130-rung11-slack-ladder",
                "harness": "local",
                "arms": list(ARMS),
                "order": ORDER,
                "permutation_seed": 14383609076371482244,
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
            },
        )
        flat = {
            "e130_rung11_argmax_slack_mb": report.get("ladder_argmax_slack_mb"),
            "e130_rung11_s512_to_s2048_pct": report.get("s512_to_s2048_pct"),
            "e130_rung11_safety_all_clear": report["safety"]["all_clear"],
            "e130_rung11_entry_temp_spread_c":
                report["thermal"]["entry_temp_spread_c"],
            "e130_rung11_one_binary": report["one_binary_served_every_arm"],
            "e130_rung11_exact_all_legs": report["exactness_all_legs"],
        }
        if primary:
            block = report["channels"]["candidate_mtp_seconds_per_token"]
            flat["e130_rung11_residual_sd_pct"] = block["residual_sd_pct"]
            flat["e130_rung11_df"] = block["df"]
            for arm, mean in block["arm_means_adjusted"].items():
                flat[f"e130_rung11_mean_{arm}_seconds_per_token"] = mean
            for key, c in block["contrasts"].items():
                if c.get("usable"):
                    flat[f"e130_rung11_{key}_pct"] = c["pct"]
                    flat[f"e130_rung11_{key}_t"] = c["t"]
        run.log(flat)
        run.summary.update(flat)
        if args.out:
            artifact = wandb.Artifact("e130-rung11-slack-ladder", type="analysis")
            artifact.add_file(str(args.out))
            run.log_artifact(artifact)
        run.finish()
        print("logged to W&B run e130r11")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
