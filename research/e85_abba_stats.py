#!/usr/bin/env python3
"""Read one E85 ABBA session and decide the advisor's per-buffer stop rule.

    usage: research/e85_abba_stats.py SESSION_DIR [--buffers N] [--json OUT]

The session file `legs.tsv` holds one row per timed leg in run order. Arms
alternate A B B A inside every repeat, so a linear thermal drift cancels
inside each block of four legs.

Two independent estimators are reported:

  block   the ABBA contrast mean(B) - mean(A) inside each block of four legs.
          One value per repeat. Exactly cancels a linear drift.
  ols     an ordinary least squares fit of the leg value on a leg-order term
          and an arm indicator, over every leg at once. Uses more degrees of
          freedom and reports the same drift-free contrast.

`serial_seconds_per_token` times the depth-0 serial leg. Neither E85 arm can
reach that code, so its measured contrast is a direct read of the session
noise floor and of any drift the design failed to remove.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145,
       16: 2.120, 18: 2.101, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021}


def t95(dof: int) -> float:
    if dof < 1:
        return math.nan
    keys = sorted(T95)
    for key in keys:
        if dof <= key:
            return T95[key]
    return 1.96


def read_legs(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in (
            "mtp_s_per_tok",
            "serial_s_per_tok",
            "ratio",
            "mean_draft_len",
            "accepted_rate",
            "temp_in",
            "temp_out",
        ):
            row[key] = float(row[key]) if row[key] not in ("", "None") else math.nan
        row["leg"] = int(row["leg"])
        row["seconds"] = int(row["seconds"])
    return rows


def block_contrasts(rows: list[dict], arm_a: str, field: str) -> list[float]:
    """One drift-cancelling contrast for every complete A B B A block."""
    out = []
    for start in range(0, len(rows) - 3, 4):
        block = rows[start : start + 4]
        a = [r[field] for r in block if r["arm"] == arm_a]
        b = [r[field] for r in block if r["arm"] != arm_a]
        if len(a) != 2 or len(b) != 2:
            continue
        out.append(statistics.fmean(b) - statistics.fmean(a))
    return out


def ols_contrast(rows: list[dict], arm_a: str, field: str,
                 covariate: str | None = None) -> dict:
    """Fit value ~ 1 + leg_order [+ covariate] + is_arm_b; return the arm effect.

    The arm indicator is always the last column, so its coefficient is the
    drift-free contrast whether or not a covariate is present.
    """
    n = len(rows)

    def design(i: int, r: dict) -> tuple[float, ...]:
        head = [1.0, float(i)]
        if covariate is not None:
            head.append(r[covariate])
        head.append(0.0 if r["arm"] == arm_a else 1.0)
        return tuple(head)

    x = [design(i, r) for i, r in enumerate(rows)]
    y = [r[field] for r in rows]
    k = len(x[0])
    arm_col = k - 1
    xtx = [[sum(x[i][a] * x[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xty = [sum(x[i][a] * y[i] for i in range(n)) for a in range(k)]

    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(k):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
    beta = [aug[i][k] for i in range(k)]

    resid = [y[i] - sum(beta[a] * x[i][a] for a in range(k)) for i in range(n)]
    dof = n - k
    sigma2 = sum(r * r for r in resid) / dof

    # standard error of beta[2] needs the (2,2) entry of inv(X'X)
    inv = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    work = [row[:] for row in xtx]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(work[r][col]))
        work[col], work[pivot] = work[pivot], work[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        div = work[col][col]
        work[col] = [v / div for v in work[col]]
        inv[col] = [v / div for v in inv[col]]
        for r in range(k):
            if r == col:
                continue
            factor = work[r][col]
            work[r] = [v - factor * w for v, w in zip(work[r], work[col])]
            inv[r] = [v - factor * w for v, w in zip(inv[r], inv[col])]

    se = math.sqrt(sigma2 * inv[arm_col][arm_col])
    effect = beta[arm_col]
    return {
        "effect": effect,
        "stderr": se,
        "t": effect / se if se > 0 else math.nan,
        "dof": dof,
        "ci95_lo": effect - t95(dof) * se,
        "ci95_hi": effect + t95(dof) * se,
        "drift_per_leg": beta[1],
        "resid_sd": math.sqrt(sigma2),
        "covariate": covariate,
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    sem = sd / math.sqrt(len(values)) if len(values) > 1 else math.nan
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "t": mean / sem if sem and sem > 0 else math.nan,
        "values": values,
    }


def drafts_per_token(row: dict) -> float:
    """Proposal-head passes per emitted token.

    A round proposes `mean_draft_len` drafts and emits one committed primary
    token plus `mean_draft_len * accepted_rate` accepted drafts.
    """
    return row["mean_draft_len"] / (1.0 + row["mean_draft_len"] * row["accepted_rate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--buffers", type=int, default=7,
                    help="materialised intermediates removed per draft token")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.session)
    rows = read_legs(root / "legs.tsv")
    if not rows:
        raise SystemExit(f"{root}/legs.tsv has no legs")

    arm_a = rows[0]["arm"]
    arm_b = next(r["arm"] for r in rows if r["arm"] != arm_a)

    report: dict = {
        "session": str(root),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "legs": len(rows),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "buffers_removed_per_draft": args.buffers,
    }

    for arm in (arm_a, arm_b):
        sub = [r for r in rows if r["arm"] == arm]
        report[f"arm_{arm}"] = {
            "legs": len(sub),
            "mtp_s_per_tok_mean": statistics.fmean(r["mtp_s_per_tok"] for r in sub),
            "mtp_s_per_tok_sd": statistics.stdev([r["mtp_s_per_tok"] for r in sub])
            if len(sub) > 1 else math.nan,
            "serial_s_per_tok_mean": statistics.fmean(r["serial_s_per_tok"] for r in sub),
            "ratio_mean": statistics.fmean(r["ratio"] for r in sub),
            "mean_draft_len_mean": statistics.fmean(r["mean_draft_len"] for r in sub),
            "accepted_rate_mean": statistics.fmean(r["accepted_rate"] for r in sub),
            "drafts_per_token_mean": statistics.fmean(drafts_per_token(r) for r in sub),
            "temp_in_min": min(r["temp_in"] for r in sub),
            "temp_in_max": max(r["temp_in"] for r in sub),
            "temp_out_max": max(r["temp_out"] for r in sub),
            "all_matched": all(r["matched"] == "True" for r in sub),
        }

    report["temp_in_spread_c"] = (
        max(r["temp_in"] for r in rows) - min(r["temp_in"] for r in rows)
    )

    for field in ("mtp_s_per_tok", "serial_s_per_tok", "ratio", "mean_draft_len",
                  "accepted_rate"):
        report[f"block_{field}"] = summarize(block_contrasts(rows, arm_a, field))
        report[f"ols_{field}"] = ols_contrast(rows, arm_a, field)

    # neither arm can reach the depth-0 serial path, so the serial leg times an
    # unchanged code path in the same process and absorbs common-mode host noise
    report["cov_mtp_s_per_tok"] = ols_contrast(
        rows, arm_a, "mtp_s_per_tok", covariate="serial_s_per_tok")

    dpt = statistics.fmean(drafts_per_token(r) for r in rows)
    base_mtp = report[f"arm_{arm_a}"]["mtp_s_per_tok_mean"]
    report["drafts_per_token"] = dpt

    def convert(prefix: str, eff: float, lo: float, hi: float) -> None:
        for name, value in (("", eff), ("_ci95_lo", lo), ("_ci95_hi", hi)):
            report[f"{prefix}_us_per_token{name}"] = value * 1e6
            report[f"{prefix}_us_per_draft{name}"] = value * 1e6 / dpt
            report[f"{prefix}_us_per_buffer{name}"] = value * 1e6 / dpt / args.buffers
        report[f"{prefix}_pct_of_mtp"] = 100.0 * eff / base_mtp

    blk = report["block_mtp_s_per_tok"]
    blk_t = t95(blk["n"] - 1) if blk["n"] > 1 else math.nan
    convert("block_effect", blk["mean"],
            blk["mean"] - blk_t * blk["sem"], blk["mean"] + blk_t * blk["sem"])
    for tag in ("ols", "cov"):
        model = report[f"{tag}_mtp_s_per_tok"]
        convert(f"{tag}_effect", model["effect"], model["ci95_lo"], model["ci95_hi"])

    # the control channel: an unchanged code path measured in the same legs
    serial_ctl = report["ols_serial_s_per_tok"]
    report["control_serial_effect_pct"] = (
        100.0 * serial_ctl["effect"]
        / report[f"arm_{arm_a}"]["serial_s_per_tok_mean"]
    )
    report["control_serial_effect_t"] = serial_ctl["t"]

    # a saving is a negative time delta; report it as a positive number
    point = -report["cov_effect_us_per_buffer"]
    best = -report["cov_effect_us_per_buffer_ci95_lo"]  # most favourable bound
    report["saving_us_per_buffer"] = point
    report["saving_us_per_buffer_ci95"] = [
        -report["cov_effect_us_per_buffer_ci95_hi"], best]

    if point >= 10.0:
        verdict = "law-holds (>= 10 us/buffer): harvest and take to pre-submit"
    elif point >= 5.0:
        verdict = "partial (5-10 us/buffer): report and stop"
    elif best < 10.0:
        verdict = ("terminal negative (< 5 us/buffer, and the 95% upper bound "
                   "excludes the 10 us/buffer law)")
    else:
        verdict = ("underpowered: the point estimate is < 5 us/buffer but the "
                   "95% upper bound still admits the 10 us/buffer law; "
                   "add repeats")
    report["verdict"] = verdict

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
