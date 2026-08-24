#!/usr/bin/env python3
"""E159 F10: separate the two width-6 effects in the local dense sweep.

Advisor F10 asks for one refit of the r1 dense fixed-depth sweep:

    R(M) = a + b*G(M) + c*M + s*[M >= 6]

`G(M) = activeInputGroups(M)` is the QMV input-group stream count from
`Qwen35.swift:1568` and `:1715`. The indicator is the wide-decode exactness
split at `AttentionUtils.swift:103`, which fires at M = 6..9 alike because
`qL * gqa > 32` with `gqa = 6`.

F10's identification argument is that `G` steps twice, at 6 and at 9, while
the indicator steps once, so the two are separable. This script checks that
claim quantitatively before reporting the fit, because

    G = 1 + [M >= 6] + [M == 9]

exactly on the integer width grid. `b` is therefore identified only through
the single M = 9 observation, and the script reports the resulting leverage.

harness=local. No GPU. Nothing here is a ranked score.

  python3 research/e159_width6_decomposition.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "e159-artifacts" / "e159_width6_decomposition.json"

# Decode round cost R_dec, milliseconds, from the r1 dense fixed proposed-depth
# sweep. M = D + 1 is the verify width. M4 Pro, 512-token window, every leg
# all_tokens_matched = true.
R_DEC_MS = {
    1: 65.813,
    2: 69.484,
    3: 71.561,
    4: 78.323,
    5: 91.634,
    6: 127.084,
    7: 138.065,
    8: 145.655,
    9: 185.427,
}

# activeInputGroups, Qwen35.swift:1715, over the shipped table at :1568.
G = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

# The wide-decode exactness split, AttentionUtils.swift:103. gqa = 24/4 = 6, so
# the fused vector path is left when qL*gqa > 32, i.e. qL >= 6.
def split_fires(m: int) -> int:
    return 1 if m >= 6 else 0


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    return {
        "beta": [float(v) for v in beta],
        "se": [float(v) for v in se],
        "t": [float(b / s) if s > 0 else float("nan") for b, s in zip(beta, se)],
        "resid": [float(v) for v in resid],
        "resid_sd": float(np.sqrt(sigma2)) if dof > 0 else None,
        "dof": int(dof),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "condition_number": float(np.linalg.cond(X)),
    }


def leverage(X: np.ndarray) -> list[float]:
    hat = X @ np.linalg.pinv(X.T @ X) @ X.T
    return [float(v) for v in np.diag(hat)]


def fit_set(widths: list[int]) -> dict:
    y = np.array([R_DEC_MS[m] for m in widths], dtype=float)
    ones = np.ones(len(widths))
    g = np.array([G[m] for m in widths], dtype=float)
    m = np.array(widths, dtype=float)
    ind = np.array([split_fires(w) for w in widths], dtype=float)

    x3 = np.column_stack([ones, g, m])
    x4 = np.column_stack([ones, g, m, ind])
    # The world in which the split owns the step and the stream owns nothing.
    x_split_only = np.column_stack([ones, m, ind])

    f3 = ols(x3, y)
    f4 = ols(x4, y)
    fs = ols(x_split_only, y)

    return {
        "widths": widths,
        "R_dec_ms": [R_DEC_MS[w] for w in widths],
        "G": [G[w] for w in widths],
        "split_indicator": [split_fires(w) for w in widths],
        "fit3_a_b_c": {"names": ["a", "b_stream", "c_row"], **f3},
        "fit4_a_b_c_s": {"names": ["a", "b_stream", "c_row", "s_split"], **f4},
        "fit_split_only_a_c_s": {"names": ["a", "c_row", "s_split"], **fs},
        "fit4_leverage": dict(zip(map(str, widths), leverage(x4))),
    }


def collinearity_note(widths: list[int]) -> dict:
    """G = 1 + [M>=6] + [M==9] exactly. Show it, and show what it implies."""
    ones = np.ones(len(widths))
    ind6 = np.array([split_fires(w) for w in widths], dtype=float)
    ind9 = np.array([1.0 if w == 9 else 0.0 for w in widths])
    g = np.array([G[w] for w in widths], dtype=float)
    reconstructed = ones + ind6 + ind9
    return {
        "identity": "G(M) == 1 + [M>=6] + [M==9] on the integer width grid",
        "exact": bool(np.allclose(g, reconstructed)),
        "max_abs_error": float(np.max(np.abs(g - reconstructed))),
        "consequence": (
            "In the four-parameter fit the intercept absorbs the 1 and the "
            "indicator absorbs [M>=6], so b_stream is identified only by the "
            "single M=9 observation. Its standard error is a one-point "
            "standard error dressed as a regression coefficient."
        ),
        "n_widths_with_G_above_2": int((g > 2).sum()),
    }


def drop_nine(widths: list[int]) -> dict:
    """What the fit says with the one identifying point removed."""
    kept = [w for w in widths if w != 9]
    y = np.array([R_DEC_MS[m] for m in kept], dtype=float)
    ones = np.ones(len(kept))
    g = np.array([G[m] for m in kept], dtype=float)
    m = np.array(kept, dtype=float)
    ind = np.array([split_fires(w) for w in kept], dtype=float)
    same = bool(np.allclose(g, 1.0 + ind))
    out = {
        "widths": kept,
        "G_equals_one_plus_indicator": same,
        "note": (
            "With M=9 removed, G and the split indicator are the same "
            "regressor up to the intercept, so b_stream and s_split are not "
            "separately identified at all. Only their sum is."
        ),
    }
    x = np.column_stack([ones, m, ind])
    out["fit_a_c_combined_step"] = {"names": ["a", "c_row", "b_plus_s"], **ols(x, y)}
    return out


def three_world_verdict(fit4: dict, coll: dict) -> dict:
    a, b, c, s = fit4["beta"]
    se_a, se_b, se_c, se_s = fit4["se"]
    return {
        "b_stream_ms": b,
        "b_stream_se": se_b,
        "s_split_ms": s,
        "s_split_se": se_s,
        "share_of_the_width6_step_from_split": (
            s / (b + s) if (b + s) != 0 else None
        ),
        "world": (
            "both_material"
            if abs(b) > 2 * se_b and abs(s) > 2 * se_s
            else "s_large_b_small"
            if abs(s) > 2 * se_s
            else "b_large_s_small"
            if abs(b) > 2 * se_b
            else "neither_resolved"
        ),
        "identification_caveat": coll["consequence"],
    }


def main() -> None:
    full = fit_set([2, 3, 4, 5, 6, 7, 8, 9])
    with_serial = fit_set([1, 2, 3, 4, 5, 6, 7, 8, 9])
    coll = collinearity_note([2, 3, 4, 5, 6, 7, 8, 9])

    report = {
        "experiment": "e159-f10-width6-decomposition",
        "harness": "local",
        "host": "M4 Pro",
        "official_or_ranked_score": False,
        "rule79_not_evidence": True,
        "gpu_seconds": 0,
        "law": "R_dec_ms = a + b*G(M) + c*M + s*[M >= 6]",
        "law_source": "advisor F10; three-parameter form from E95QmvWidthProbeTests.swift",
        "source_of_G": "Qwen35.swift:1568 table, :1715 activeInputGroups",
        "source_of_indicator": "AttentionUtils.swift:103, qL*gqa > 32 with gqa = 6",
        "collinearity": coll,
        "widths_2_to_9": full,
        "widths_1_to_9_sensitivity": with_serial,
        "drop_the_identifying_point": drop_nine([2, 3, 4, 5, 6, 7, 8, 9]),
        "three_world_verdict": three_world_verdict(
            full["fit4_a_b_c_s"], coll
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")

    f3 = full["fit3_a_b_c"]
    f4 = full["fit4_a_b_c_s"]
    fs = full["fit_split_only_a_c_s"]
    print("collinearity: G == 1 + [M>=6] + [M==9] exact:", coll["exact"])
    print()
    for label, fit in (("3-param a,b,c", f3), ("4-param a,b,c,s", f4), ("no-stream a,c,s", fs)):
        parts = " ".join(
            f"{n}={v:+.3f}+-{e:.3f}"
            for n, v, e in zip(fit["names"], fit["beta"], fit["se"])
        )
        print(f"{label:18s} {parts}  resid_sd={fit['resid_sd']:.3f} dof={fit['dof']} R2={fit['r2']:.5f} cond={fit['condition_number']:.1f}")
    print()
    print("per-width residuals, ms")
    print("  M    3-param    4-param   no-stream   leverage(4p)")
    for i, m in enumerate(full["widths"]):
        print(
            f"  {m}  {f3['resid'][i]:+9.3f}  {f4['resid'][i]:+9.3f}  {fs['resid'][i]:+9.3f}"
            f"   {full['fit4_leverage'][str(m)]:.3f}"
        )
    print()
    print("verdict:", json.dumps(report["three_world_verdict"], indent=1))
    print()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
