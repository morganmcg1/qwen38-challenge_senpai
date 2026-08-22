#!/usr/bin/env python3
"""E128-F5 - a two-channel mode and mechanism estimator for board receipts.

harness=ranked. Zero GPU. Board rows only.

Why the one-channel F76 index is not enough
-------------------------------------------

F76 reads one linear functional of the eight per-prompt log times:

    index = sum_p w_p * 100 * ln(t_p),      sum_p w_p = 0

Because the weights sum to zero, a uniform multiplicative speedup cancels and
the index answers "which scheduler mode is this receipt in?". That holds only
while mode is the ONLY thing that moves the per-prompt profile non-uniformly.
A kernel or runtime mechanism whose gain depends on realised draft width is
also non-uniform, so it moves the same scalar. One number cannot then tell a
mode flip from a mechanism step: the channels are confounded.

The estimator
-------------

Work in the profile space. For a receipt r let

    x_r[p] = 100 * ln(t_r[p])            (eight numbers)
    profile_r = x_r - mean_p(x_r)        (uniform speed removed)

`w . profile_r == w . x_r` exactly, because `sum(w) = 0`, so nothing about
F76 is lost by working in profile space.

Two directions are estimated from the board by one multivariate regression
over the rows that carry complete per-prompt evidence:

    profile_r = mu + a_r * u_mode + b_r * u_mech + e_r

    u_mode  regression coefficient on a mode indicator
    u_mech  regression coefficient on ln(official score)

`u_mode` is the way the profile moves when the scheduler mode flips.
`u_mech` is the way the profile moves as the campaign frontier advances,
which is the accumulated non-uniform footprint of the kernel and runtime
mechanisms. The mode indicator is taken from F76 itself but only on the
low-score epoch, where the mechanism footprint is small, so the label is not
contaminated by the thing it must stay independent of.

Both directions are then rescaled to keep the published F76 calibration:

    w . u_mode = 1.0000     one unit of channel a is one F76 mode flip
    w . u_mech = 1.0000     one unit of channel b is the mechanism step that
                            would move F76 by the same amount as a mode flip

With that scaling the decomposition of the published index is exactly

    index_r = w . mu + a_r + b_r + (w . e_r)

so the confounding is read directly: the mode call is safe only when |b_r| is
small next to |a_r|.

Reading a receipt is then a two-regressor least squares fit of its own
profile onto (u_mode, u_mech), reported with the residual that neither
channel explains.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
ORDER = ["plutarch", "drama", "travel", "beagle",
         "medicine", "republic", "essays", "botany"]
W = np.array([-0.3852, +0.0215, +0.4945, +0.2068,
              -0.1480, -0.0917, -0.0041, -0.0939])
THRESHOLD = -12.9

ANCHORS = ["d3c491b5", "cf79f7df", "c63eaa21", "1986338b",
           "bc070b7b", "44559d02", "ec778a91", "b8b8b860", "f04b102e"]


def load_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw["submissions"]
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        pp = (r.get("officialMetrics") or {}).get("per_prompt")
        score = r.get("officialScore")
        if not pp or len(pp) != 8 or not score:
            continue
        times = {}
        for e in pp:
            name = PROMPT_NAMES.get(str(e.get("prompt_sha256"))[:8])
            t = e.get("mtp_seconds_per_token_mean")
            if name and t:
                times[name] = float(t)
        if len(times) != 8:
            continue
        x = np.array([100.0 * math.log(times[p]) for p in ORDER])
        out.append({
            "id": str(r["id"]), "short": str(r["id"])[:8],
            "solver": str(r.get("solverUsername") or "")[:16],
            "score": float(score), "status": str(r.get("status") or ""),
            "when": str(r.get("resolvedAt") or r.get("createdAt") or ""),
            "x": x, "profile": x - x.mean(), "index": float(W @ x),
        })
    out.sort(key=lambda r: r["when"])
    return out


def fit_channels(rows: list[dict], centre_score: float) -> dict:
    """Estimate u_mode and u_mech by one multivariate regression.

    `rows` must already be restricted to the epoch in which the published
    F76 fast/slow calibration applies. `centre_score` fixes the origin of the
    mechanism channel, so `b = 0` means "the mechanism footprint of a receipt
    scoring exactly `centre_score`".
    """
    # Mode label from F76 at the published threshold.
    for r in rows:
        r["mode_fast"] = 1.0 if r["index"] < THRESHOLD else 0.0

    lns = np.array([math.log(r["score"] / centre_score) for r in rows])
    mode = np.array([r["mode_fast"] for r in rows])
    design = np.column_stack([np.ones(len(rows)), mode, lns])
    profiles = np.array([r["profile"] for r in rows])
    coef, *_ = np.linalg.lstsq(design, profiles, rcond=None)
    mu, u_mode, u_mech = coef[0], coef[1], coef[2]

    # Put both channels in published index units, so that `a` and `b` are
    # each exactly the number of index units their channel contributes. F76
    # makes the fast side the more negative index, so a fast receipt lands
    # near a = -1 and a slow receipt near a = 0.
    scale_mode = float(W @ u_mode)
    scale_mech = float(W @ u_mech)
    u_mode = u_mode / scale_mode
    u_mech = u_mech / scale_mech

    basis = np.column_stack([u_mode, u_mech])
    return {
        "mu": mu, "u_mode": u_mode, "u_mech": u_mech, "basis": basis,
        "raw_scale_mode": scale_mode, "raw_scale_mech": scale_mech,
        "w_u_mode": float(W @ u_mode), "w_u_mech": float(W @ u_mech),
        "centre_score": centre_score,
        "n_rows": len(rows),
        "n_fast": int(mode.sum()), "n_slow": int((1 - mode).sum()),
        "cos": float(u_mode @ u_mech
                     / (np.linalg.norm(u_mode) * np.linalg.norm(u_mech))),
        "cond": float(np.linalg.cond(basis)),
    }


def read_receipt(fit: dict, row: dict) -> dict:
    y = row["profile"] - fit["mu"]
    coef, *_ = np.linalg.lstsq(fit["basis"], y, rcond=None)
    resid = y - fit["basis"] @ coef
    a, b = float(coef[0]), float(coef[1])
    contrib_a = a * fit["w_u_mode"]
    contrib_b = b * fit["w_u_mech"]
    return {
        "a_mode": a, "b_mech": b,
        "index_from_mode": contrib_a, "index_from_mech": contrib_b,
        "resid_norm": float(np.linalg.norm(resid)),
        "w_resid": float(W @ resid),
        "index": row["index"],
        "index_check": (float(W @ fit["mu"]) + contrib_a + contrib_b
                        + float(W @ resid)),
        "mode_call": "fast" if a < -0.5 else ("slow" if a > -0.5 else "edge"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--min-score", type=float, default=3.25)
    parser.add_argument("--centre-score", type=float, default=3.49065044)
    parser.add_argument("--split-from", default="d3c491b5")
    parser.add_argument("--split-to", default="cf79f7df")
    parser.add_argument("--predict-base", default="d3c491b5")
    parser.add_argument("--predict-uniform-pct", type=float, default=0.0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("anchors", nargs="*", default=ANCHORS)
    args = parser.parse_args()

    every = load_rows(args.board)
    rows = [r for r in every if r["score"] >= args.min_score]
    fit = fit_channels(rows, args.centre_score)

    print("harness=ranked  E128-F5 two-channel mode and mechanism estimator")
    print("board rows with complete per-prompt evidence: %d" % len(every))
    print("fitting epoch score >= %.2f: %d rows (%d fast, %d slow by F76)"
          % (args.min_score, fit["n_rows"], fit["n_fast"], fit["n_slow"]))
    print("mechanism channel centred at score %.8f" % fit["centre_score"])
    print("one mode flip measured at %+.4f index units"
          % fit["raw_scale_mode"])
    print("one e-fold of score measured at %+.4f index units"
          % fit["raw_scale_mech"])
    print("cos(u_mode, u_mech) = %+.4f   cond(basis) = %.2f"
          % (fit["cos"], fit["cond"]))
    print()
    print("%-10s " % "prompt" + " ".join("%9s" % p for p in ORDER))
    print("%-10s " % "w" + " ".join("%+9.4f" % v for v in W))
    print("%-10s " % "mu" + " ".join("%+9.3f" % v for v in fit["mu"]))
    print("%-10s " % "u_mode" + " ".join("%+9.3f" % v for v in fit["u_mode"]))
    print("%-10s " % "u_mech" + " ".join("%+9.3f" % v for v in fit["u_mech"]))
    print()

    print("%-10s %-16s %11s %9s %9s %9s %9s %8s  %s" % (
        "id", "solver", "score", "F76", "a_mode", "b_mech", "resid",
        "w.resid", "call"))
    out_rows = {}
    by_short = {r["short"]: r for r in every}
    for short in args.anchors:
        row = by_short.get(short)
        if row is None:
            print("%-10s  NOT ON THE BOARD WITH COMPLETE PER-PROMPT ROWS"
                  % short)
            continue
        got = read_receipt(fit, row)
        out_rows[short] = {k: v for k, v in got.items()}
        out_rows[short].update(
            {"solver": row["solver"], "score": row["score"],
             "status": row["status"], "when": row["when"]})
        print("%-10s %-16s %11.8f %9.4f %+9.4f %+9.4f %9.4f %+8.4f  %s" % (
            short, row["solver"], row["score"], got["index"],
            got["a_mode"], got["b_mech"], got["resid_norm"],
            got["w_resid"], got["mode_call"]))

    # Per-prompt residual: what neither channel explains, prompt by prompt.
    print("\nper-prompt residual (index-unit log time, neither channel)")
    print("%-10s " % "id" + " ".join("%9s" % p for p in ORDER) + "     norm")
    resid_rows = {}
    for short in args.anchors:
        row = by_short.get(short)
        if row is None:
            continue
        y = row["profile"] - fit["mu"]
        coef, *_ = np.linalg.lstsq(fit["basis"], y, rcond=None)
        e = y - fit["basis"] @ coef
        resid_rows[short] = [float(v) for v in e]
        print("%-10s " % short + " ".join("%+9.3f" % v for v in e)
              + " %8.4f" % np.linalg.norm(e))

    # Exact decomposition of the moves the advisor named.
    print("\nindex-move decomposition, channel by channel")
    print("%-24s %9s %9s %9s %9s %9s" % (
        "from -> to", "d index", "d mode", "d mech", "d resid", "check"))
    pairs = [("d3c491b5", "cf79f7df"), ("d3c491b5", "c63eaa21"),
             ("d3c491b5", "1986338b"), ("44559d02", "d3c491b5"),
             ("bc070b7b", "d3c491b5")]
    moves = {}
    for lo, hi in pairs:
        if lo not in by_short or hi not in by_short:
            continue
        g0 = read_receipt(fit, by_short[lo])
        g1 = read_receipt(fit, by_short[hi])
        d_index = g1["index"] - g0["index"]
        d_mode = g1["a_mode"] - g0["a_mode"]
        d_mech = g1["b_mech"] - g0["b_mech"]
        d_res = g1["w_resid"] - g0["w_resid"]
        moves["%s->%s" % (lo, hi)] = {
            "d_index": d_index, "d_mode": d_mode,
            "d_mech": d_mech, "d_resid": d_res,
            "mode_share_pct": 100.0 * d_mode / d_index if d_index else None,
            "mech_share_pct": 100.0 * d_mech / d_index if d_index else None,
        }
        print("%-24s %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f" % (
            "%s -> %s" % (lo, hi), d_index, d_mode, d_mech, d_res,
            d_mode + d_mech + d_res))

    # Why a per-prompt reading of the index misleads. Each term
    # `w_p * 100 * ln(t_p)` still carries the uniform log-speed `s`, which
    # only cancels AFTER the sum over prompts. Attributing the move to the
    # prompt with the largest term therefore charges a uniform speedup to
    # whichever prompt happens to hold the largest positive weight.
    print("\nper-prompt attribution, %s -> %s (index units)"
          % (args.split_from, args.split_to))
    split = {}
    if args.split_from in by_short and args.split_to in by_short:
        lo, hi = by_short[args.split_from], by_short[args.split_to]
        dx = hi["x"] - lo["x"]
        ds = float(dx.mean())
        g0, g1 = read_receipt(fit, lo), read_receipt(fit, hi)
        da, db = g1["a_mode"] - g0["a_mode"], g1["b_mech"] - g0["b_mech"]
        t_uni = W * ds
        t_mode = W * da * fit["u_mode"]
        t_mech = W * db * fit["u_mech"]
        t_res = W * (dx - dx.mean()
                     - da * fit["u_mode"] - db * fit["u_mech"])
        print("uniform log-time change over all eight prompts: %+.4f "
              "(%.3f %% faster)" % (ds, -ds))
        print("%-10s %9s %9s %9s %9s %9s"
              % ("prompt", "d term", "uniform", "mode", "mech", "resid"))
        for i, p in enumerate(ORDER):
            print("%-10s %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f"
                  % (p, W[i] * dx[i], t_uni[i], t_mode[i], t_mech[i],
                     t_res[i]))
        print("%-10s %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f"
              % ("SUM", float(W @ dx), t_uni.sum(), t_mode.sum(),
                 t_mech.sum(), t_res.sum()))
        split = {
            "from": args.split_from, "to": args.split_to,
            "uniform_log_time_change": ds,
            "per_prompt": {p: {"d_term": float(W[i] * dx[i]),
                               "uniform": float(t_uni[i]),
                               "mode": float(t_mode[i]),
                               "mech": float(t_mech[i]),
                               "resid": float(t_res[i])}
                           for i, p in enumerate(ORDER)},
        }

    # How much of the epoch index spread each channel explains.
    A = np.array([read_receipt(fit, r)["a_mode"] for r in rows])
    B = np.array([read_receipt(fit, r)["b_mech"] for r in rows])
    E = np.array([read_receipt(fit, r)["w_resid"] for r in rows])
    idx = np.array([r["index"] for r in rows])
    print()
    print("over the %d epoch rows: sd(F76) %.4f = sd(mode) %.4f "
          "+ sd(mech) %.4f + sd(resid) %.4f"
          % (len(rows), idx.std(ddof=1), A.std(ddof=1), B.std(ddof=1),
             E.std(ddof=1)))
    print("corr(a_mode, b_mech) = %+.4f   "
          "share of var(F76): mode %.1f %%  mech %.1f %%  resid %.1f %%"
          % (float(np.corrcoef(A, B)[0, 1]),
             100 * np.cov(A, idx)[0, 1] / idx.var(ddof=1),
             100 * np.cov(B, idx)[0, 1] / idx.var(ddof=1),
             100 * np.cov(E, idx)[0, 1] / idx.var(ddof=1)))
    fast = [r for r in rows if r["mode_fast"] > 0.5]
    slow = [r for r in rows if r["mode_fast"] < 0.5]
    clusters = {}
    for name, sub in (("fast", fast), ("slow", slow)):
        if not sub:
            continue
        sa = np.array([read_receipt(fit, r)["a_mode"] for r in sub])
        sb = np.array([read_receipt(fit, r)["b_mech"] for r in sub])
        se = np.array([read_receipt(fit, r)["resid_norm"] for r in sub])
        clusters[name] = {
            "n": len(sub), "a_mean": float(sa.mean()),
            "a_sd": float(sa.std(ddof=1)), "b_mean": float(sb.mean()),
            "b_sd": float(sb.std(ddof=1)),
            "resid_mean": float(se.mean()), "resid_sd": float(se.std(ddof=1)),
        }
        print("  %-4s n=%3d  a_mode %+.4f +/- %.4f   b_mech %+.4f +/- %.4f"
              "   |resid| %.4f +/- %.4f"
              % (name, len(sub), sa.mean(), sa.std(ddof=1),
                 sb.mean(), sb.std(ddof=1), se.mean(), se.std(ddof=1)))

    # Held-out temporal validation: fit the basis on the older half and ask
    # whether it still spans the newer half. A basis that only memorised the
    # training rows loses its advantage over the plain mean out of sample.
    order = sorted(rows, key=lambda r: r["when"])
    cut = len(order) // 2
    train, test = order[:cut], order[cut:]
    heldout = {}
    if len(train) >= 30 and len(test) >= 30:
        fit2 = fit_channels([dict(r) for r in train], fit["centre_score"])
        basis1 = fit2["u_mode"].reshape(-1, 1)

        def norms(sub, basis):
            out = []
            for r in sub:
                y = r["profile"] - fit2["mu"]
                if basis is None:
                    out.append(np.linalg.norm(y))
                    continue
                coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
                out.append(np.linalg.norm(y - basis @ coef))
            return float(np.mean(out))

        heldout = {
            "n_train": len(train), "n_test": len(test),
            "one_flip_train": fit2["raw_scale_mode"],
            "train_resid_0ch": norms(train, None),
            "train_resid_1ch": norms(train, basis1),
            "train_resid_2ch": norms(train, fit2["basis"]),
            "test_resid_0ch": norms(test, None),
            "test_resid_1ch": norms(test, basis1),
            "test_resid_2ch": norms(test, fit2["basis"]),
        }
        print("\nheld-out temporal check: basis fitted on the older %d rows, "
              "applied to the newer %d" % (len(train), len(test)))
        print("  mean |profile - model|   %8s %8s %8s"
              % ("mu only", "+mode", "+mech"))
        print("  in sample  (%3d rows)    %8.4f %8.4f %8.4f"
              % (len(train), heldout["train_resid_0ch"],
                 heldout["train_resid_1ch"], heldout["train_resid_2ch"]))
        print("  held out   (%3d rows)    %8.4f %8.4f %8.4f"
              % (len(test), heldout["test_resid_0ch"],
                 heldout["test_resid_1ch"], heldout["test_resid_2ch"]))
        print("  one mode flip re-measured on the training half: %+.4f"
              % fit2["raw_scale_mode"])

    # Forward prediction for a receipt that is not on the board yet.
    forecast = {}
    if args.predict_base:
        base = by_short.get(args.predict_base)
        if base is None:
            print("\npredict base %s is not on the board" % args.predict_base)
        else:
            g = read_receipt(fit, base)
            d_mech = args.predict_uniform_pct  # exactly zero effect on F76
            same_mode_sd, run_noise = 0.116, 0.0817
            band = math.sqrt(same_mode_sd ** 2 + run_noise ** 2)
            forecast = {
                "base": args.predict_base, "base_index": g["index"],
                "base_a_mode": g["a_mode"], "base_b_mech": g["b_mech"],
                "assumed_uniform_gain_pct": d_mech,
                "predicted_index": g["index"],
                "band_1sd": band, "band_2sd": 2 * band,
            }
            print("\nforward prediction for a receipt built on %s"
                  % args.predict_base)
            print("  a width-independent gain leaves F76 unchanged because "
                  "sum(w) = 0")
            print("  predicted index %.4f  1 sd +/- %.4f  2 sd +/- %.4f"
                  % (g["index"], band, 2 * band))
            print("  predicted a_mode %+.4f  predicted b_mech %+.4f"
                  % (g["a_mode"], g["b_mech"]))

    if args.json:
        payload = {
            "harness": "ranked",
            "weights": {p: float(v) for p, v in zip(ORDER, W)},
            "threshold": THRESHOLD,
            "order": ORDER,
            "fit": {
                "n_rows": fit["n_rows"],
                "n_fast": fit["n_fast"], "n_slow": fit["n_slow"],
                "min_score": args.min_score,
                "centre_score": fit["centre_score"],
                "one_flip_index_units": fit["raw_scale_mode"],
                "one_efold_index_units": fit["raw_scale_mech"],
                "mu": [float(v) for v in fit["mu"]],
                "u_mode": [float(v) for v in fit["u_mode"]],
                "u_mech": [float(v) for v in fit["u_mech"]],
                "cos": fit["cos"], "cond": fit["cond"],
            },
            "receipts": out_rows,
            "per_prompt_residual": resid_rows,
            "index_moves": moves,
            "per_prompt_attribution": split,
            "clusters": clusters,
            "heldout": heldout,
            "forecast": forecast,
            "spread": {
                "sd_f76": float(idx.std(ddof=1)),
                "sd_a_mode": float(A.std(ddof=1)),
                "sd_b_mech": float(B.std(ddof=1)),
                "sd_w_resid": float(E.std(ddof=1)),
                "corr_a_b": float(np.corrcoef(A, B)[0, 1]),
            },
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
