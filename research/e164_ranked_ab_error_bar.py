#!/usr/bin/env python3
"""E164 - the run-to-run error bar for a two-receipt ranked A/B.

harness=ranked throughout. Every number here comes from public Yukon board
receipts for track qwen3.8-27b-mtp-v1. No local timing is used or mixed in.

The instrument is the pinned serial leg. `senpai/verify-ranked-score-boundary.sh`
enforces d ln(ranked baseline serial time)/dx = 0 for every candidate edit x, so
`serial_seconds_per_token_mean` is one fixed program re-run once per receipt.
Its dispersion measures ranked run-to-run variability with no model.

Questions
  Q1  dispersion and intraclass correlation of the pinned serial leg
  Q2  does `raw_ratio_of_means` cancel the run-level component?
  Q3  re-price FINDING 259 / FINDING 348 with the correct denominator
  Q4  reconcile FINDING 259 against FINDING 279

Usage
  python3 research/e164_ranked_ab_error_bar.py [--board PATH] [--out DIR] [--wandb]

Board refresh (writes ~17 MB outside the checkout):
  YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from fractions import Fraction

import numpy as np

LIVE_BOARD = "/tmp/yukon-board/full.json"
FROZEN_BOARD = os.path.join(os.path.dirname(__file__), "board-per-prompt-2026-08-23.json")

# Harness constants that must be identical for receipts to be comparable replicates.
INVARIANTS = (
    "decode_tokens",
    "pairs_per_prompt",
    "prompt_count",
    "mode",
    "aggregation",
    "median_rule",
    "scoring_normalized",
    "score_anchor",
    "qwen_mtp_weights_hash",
)

CHANNELS = ("serial", "mtp", "raw", "prefill")


# ---------------------------------------------------------------- data loading


class Board:
    """Receipt x prompt matrices for one homogeneous ranked harness."""

    def __init__(self, receipts):
        self.r = receipts
        self.n = len(receipts)
        self.prompts = [p["prompt_sha256"][:8] for p in receipts[0]["officialMetrics"]["per_prompt"]]
        self.ids = [x["id"][:8] for x in receipts]
        self.created = [x["createdAt"] for x in receipts]
        self.score = np.array([x["officialScore"] for x in receipts])

        def mat(key):
            out = np.full((self.n, 8), np.nan)
            for i, x in enumerate(receipts):
                for j, p in enumerate(x["officialMetrics"]["per_prompt"]):
                    v = p.get(key)
                    if v is not None:
                        out[i, j] = v
            return out

        self.serial = mat("serial_seconds_per_token_mean")
        self.mtp = mat("mtp_seconds_per_token_mean")
        self.raw = mat("raw_ratio_of_means")
        self.prefill = mat("prefill_seconds_per_token")
        self.edl = mat("effective_mean_draft_len")
        self.nondraft = mat("non_drafting_round_count")

    def chan(self, name):
        return getattr(self, name)

    def index(self, id8):
        return self.ids.index(id8)


def load_board(path):
    """Return (Board, provenance dict). Accepts the live payload or the frozen export."""
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        rows = payload.get("submissions") or payload.get("rows") or []
    else:
        rows = payload
    if rows and "officialMetrics" not in rows[0]:
        raise SystemExit(
            f"{path} is the frozen export, which drops officialMetrics and prefill.\n"
            "Refresh the live board: YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch"
        )
    scored = [
        r
        for r in rows
        if (r.get("officialMetrics") or {}).get("per_prompt") and r.get("officialScore") is not None
    ]
    scored.sort(key=lambda r: r["createdAt"])
    prov = {
        "board_path": path,
        "rows_total": len(rows),
        "rows_scored_with_per_prompt": len(scored),
        "created_first": scored[0]["createdAt"],
        "created_last": scored[-1]["createdAt"],
    }
    return Board(scored), prov


def check_invariants(board):
    """Every receipt must share the harness constants, or it is not a replicate."""
    out = {}
    for key in INVARIANTS:
        vals = collections.Counter(json.dumps(x["officialMetrics"].get(key)) for x in board.r)
        if len(vals) != 1:
            raise SystemExit(f"harness invariant {key} is not constant: {dict(vals)}")
        out[key] = json.loads(next(iter(vals)))
    for i, x in enumerate(board.r):
        got = [p["prompt_sha256"][:8] for p in x["officialMetrics"]["per_prompt"]]
        if got != board.prompts:
            raise SystemExit(f"receipt {board.ids[i]} has a different prompt set or order")
    out["prompt_sha256_8"] = board.prompts
    return out


# ------------------------------------------------------------------ statistics


def log_dev(mat):
    """Per-prompt centred log deviations, in percent. NaN rows are dropped by caller."""
    lg = np.log(mat)
    return 100.0 * (lg - lg.mean(axis=0, keepdims=True))


def icc_oneway(dev):
    """One-way random-effects ICC on a receipts x prompts deviation matrix.

    dev[i,j] = mu + a_i + e_ij. Returns the variance split and the design effect
    that a run-level effect must pay instead of the naive sqrt(k).
    """
    n, k = dev.shape
    row = dev.mean(axis=1)
    grand = dev.mean()
    msb = k * ((row - grand) ** 2).sum() / (n - 1)
    msw = ((dev - row[:, None]) ** 2).sum() / (n * (k - 1))
    var_run = max((msb - msw) / k, 0.0)
    icc = var_run / (var_run + msw) if (var_run + msw) > 0 else 0.0
    return {
        "n_receipts": int(n),
        "k_prompts": int(k),
        "sd_total_pct": float(math.sqrt(var_run + msw)),
        "sd_within_receipt_pct": float(math.sqrt(msw)),
        "sd_run_level_pct": float(math.sqrt(var_run)),
        "icc": float(icc),
        "sd_of_run_mean_measured_pct": float(row.std(ddof=1)),
        "sd_of_run_mean_if_independent_pct": float(math.sqrt(msw / k)),
        "design_effect": float(1.0 + (k - 1) * icc),
        "se_inflation_factor": float(math.sqrt(1.0 + (k - 1) * icc)),
    }


def bootstrap_icc(dev, n_boot=2000, seed=164):
    rng = np.random.default_rng(seed)
    n = dev.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = icc_oneway(dev[rng.integers(0, n, n)])["icc"]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pair_delta(board, channel, i, j):
    """Percent change from receipt i to receipt j. Positive = j is larger."""
    a = board.chan(channel)[i]
    b = board.chan(channel)[j]
    ok = np.isfinite(a) & np.isfinite(b)
    d = 100.0 * (b[ok] / a[ok] - 1.0)
    return d, ok


def pair_stats(d):
    k = len(d)
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    se = sd / math.sqrt(k)
    return {
        "k": k,
        "mean_pct": mean,
        "sd_pct": sd,
        "se_naive_pct": se,
        "z_naive": mean / se if se else float("nan"),
        "n_positive": int((d > 0).sum()),
    }


def sign_test_p(n_pos, k):
    """Two-sided exact binomial p at p0 = 0.5."""
    n_pos = max(n_pos, k - n_pos)
    tail = sum(math.comb(k, i) for i in range(n_pos, k + 1))
    return min(1.0, 2.0 * tail / (2**k))


# -------------------------------------------------------------------------- Q1


def q1_serial_dispersion(board):
    S = board.serial
    dev = log_dev(S)
    res = {"harness": "ranked", "channel": "serial_seconds_per_token_mean"}

    res["per_prompt"] = []
    for j, p in enumerate(board.prompts):
        c = S[:, j]
        res["per_prompt"].append(
            {
                "prompt": p,
                "mean_s_per_token": float(c.mean()),
                "sd_s_per_token": float(c.std(ddof=1)),
                "rel_sd_pct": float(100 * c.std(ddof=1) / c.mean()),
                "p2_5": float(np.percentile(c, 2.5)),
                "p97_5": float(np.percentile(c, 97.5)),
                "min": float(c.min()),
                "max": float(c.max()),
            }
        )

    pooled = S.ravel()
    res["pooled"] = {
        "n_cells": int(pooled.size),
        "mean_s_per_token": float(pooled.mean()),
        "sd_s_per_token": float(pooled.std(ddof=1)),
        "rel_sd_pct": float(100 * pooled.std(ddof=1) / pooled.mean()),
        "p2_5": float(np.percentile(pooled, 2.5)),
        "p97_5": float(np.percentile(pooled, 97.5)),
    }

    res["icc"] = icc_oneway(dev)
    lo, hi = bootstrap_icc(dev)
    res["icc"]["icc_ci95"] = [lo, hi]

    corr = np.corrcoef(dev.T)
    off = corr[~np.eye(8, dtype=bool)]
    res["cross_prompt_correlation"] = {
        "matrix": [[float(v) for v in row] for row in corr],
        "mean_off_diagonal": float(off.mean()),
        "min_off_diagonal": float(off.min()),
        "max_off_diagonal": float(off.max()),
        "n_off_diagonal_pairs": int(off.size // 2),
        "fraction_positive": float((off > 0).mean()),
    }

    # Drift versus jitter. Day means isolate the slow component.
    day = [c[:10] for c in board.created]
    res["by_day"] = []
    within, between = [], []
    for d in sorted(set(day)):
        idx = [i for i, x in enumerate(day) if x == d]
        if len(idx) < 8:
            continue
        blk = S[idx]
        res["by_day"].append(
            {
                "day": d,
                "n_receipts": len(idx),
                "median_s_per_token": float(np.median(blk)),
                "mean_s_per_token": float(blk.mean()),
                "within_day_rel_sd_pct": float(100 * blk.std(ddof=1) / blk.mean()),
            }
        )
        within.append(100 * blk.std(ddof=1) / blk.mean())
        between.append(blk.mean())
    res["drift"] = {
        "n_days": len(within),
        "mean_within_day_rel_sd_pct": float(np.mean(within)),
        "between_day_rel_sd_of_means_pct": float(100 * np.std(between, ddof=1) / np.mean(between)),
        "day_median_min": float(min(d["median_s_per_token"] for d in res["by_day"])),
        "day_median_max": float(max(d["median_s_per_token"] for d in res["by_day"])),
    }

    # Narrow window: 6-hour buckets remove any slow term almost entirely.
    bucket = [c[:13] for c in board.created]
    hb = collections.defaultdict(list)
    for i, b in enumerate(bucket):
        hb[b].append(i)
    tight = [i for v in hb.values() if len(v) >= 8 for i in v]
    if tight:
        blk = S[tight]
        dev_t = log_dev(blk)
        res["narrow_window"] = {
            "bucket": "hour",
            "n_receipts": len(tight),
            "rel_sd_pct": float(100 * blk.std(ddof=1) / blk.mean()),
            "icc": icc_oneway(dev_t),
        }

    ppt = res["pooled"]["mean_s_per_token"]
    res["finding_329_check"] = {
        "claim": "baseline serial daily median 0.03799 +/- 0.0001, flat 2026-08-14..08-23",
        "pooled_mean_s_per_token": ppt,
        "daily_median_range": [res["drift"]["day_median_min"], res["drift"]["day_median_max"]],
        "pooled_sd_s_per_token": res["pooled"]["sd_s_per_token"],
        "verdict": "confirmed" if abs(ppt - 0.03799) < 1e-5 else "corrected",
    }
    return res


# -------------------------------------------------------------------------- Q2


def behaviour_fingerprint(receipt):
    """Digit-identical drafting behaviour on all eight prompts.

    Matching head provenance, proposed-depth rational, accepted-pair count and
    non-drafting round count means the two candidates walked the same tree. It
    does NOT mean the submitted source is identical: FINDING 259's own pair is a
    fingerprint match with a real Metal change, so this set is an upper bound on
    a null, never a clean null.
    """
    return tuple(
        (
            p["head_provenance_sha256"][:16],
            p["accepted_pair_count"],
            repr(p["effective_mean_draft_len"]),
            p["non_drafting_round_count"],
        )
        for p in receipt["officialMetrics"]["per_prompt"]
    )


def q2_raw_versus_candidate(board, q1, seed=164):
    res = {"harness": "ranked", "icc_inflation_factor": q1["icc"]["se_inflation_factor"]}

    # The advisor proposed same-commit / same-source-ref replicates. Test that first.
    for key in ("submissionCommitSha", "promotedSourceRef"):
        g = collections.defaultdict(list)
        for i, x in enumerate(board.r):
            if x.get(key):
                g[x[key]].append(i)
        dup = {k: v for k, v in g.items() if len(v) > 1}
        res[f"replicates_by_{key}"] = {
            "n_groups": len(dup),
            "n_receipts": sum(len(v) for v in dup.values()),
            "n_pairs": sum(len(v) * (len(v) - 1) // 2 for v in dup.values()),
        }

    groups = collections.defaultdict(list)
    for i, x in enumerate(board.r):
        groups[behaviour_fingerprint(x)].append(i)
    fp_groups = [v for v in groups.values() if len(v) > 1]
    res["replicates_by_behaviour_fingerprint"] = {
        "n_groups": len(fp_groups),
        "n_receipts": sum(len(v) for v in fp_groups),
        "n_pairs": sum(len(v) * (len(v) - 1) // 2 for v in fp_groups),
        "largest_groups": sorted((len(v) for v in fp_groups), reverse=True)[:8],
    }

    # raw is exactly serial / mtp, so the variance algebra below is exact.
    ok = np.isfinite(board.raw) & np.isfinite(board.mtp) & np.isfinite(board.serial)
    ident = np.abs(board.raw[ok] - board.serial[ok] / board.mtp[ok]) / board.raw[ok]
    res["raw_identity"] = {
        "expression": "raw_ratio_of_means == serial_seconds_per_token_mean / mtp_seconds_per_token_mean",
        "max_relative_deviation": float(ident.max()),
    }

    # --- run-level dispersion per channel, on fingerprint replicates ---
    # Within a group, remove the group x prompt mean so that only run-to-run
    # movement of the mean-of-eight survives. A real mechanism difference inside
    # the group inflates the candidate and raw numbers, so they are upper bounds.
    per_channel = {}
    for ch in CHANNELS:
        mat = board.chan(ch)
        run_means, within = [], []
        for g in fp_groups:
            sub = mat[g]
            keep = np.isfinite(sub).all(axis=1)
            sub = sub[keep]
            if sub.shape[0] < 2:
                continue
            lg = 100.0 * np.log(sub)
            lg = lg - lg.mean(axis=0, keepdims=True)  # group x prompt centring
            run_means.extend(lg.mean(axis=1))
            within.extend((lg - lg.mean(axis=1, keepdims=True)).ravel())
        run_means = np.array(run_means)
        within = np.array(within)
        # Group centring removes 1/m of the between-receipt variance; the small
        # bias is conservative for an upper bound, so it is reported unadjusted.
        sd_run_mean = float(run_means.std(ddof=1))
        per_channel[ch] = {
            "n_receipts_in_groups": int(run_means.size),
            "sd_of_run_mean_of_8_pct": sd_run_mean,
            "sd_within_receipt_pct": float(within.std(ddof=1)),
            "se_single_receipt_pair_pct": sd_run_mean * math.sqrt(2.0),
        }
    res["fingerprint_replicate_dispersion"] = per_channel

    # --- exact serial-leg null: the only assumption-free number here ---
    S = board.serial
    dev = log_dev(S)
    sd_run_mean_serial = float(dev.mean(axis=1).std(ddof=1))
    res["serial_true_null"] = {
        "note": "the pinned serial build is byte-identical on every receipt, so this is a true null",
        "sd_of_run_mean_of_8_pct": sd_run_mean_serial,
        "se_single_receipt_pair_pct": sd_run_mean_serial * math.sqrt(2.0),
    }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, board.n, size=(200000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    d = 100.0 * (S[idx[:, 1]] / S[idx[:, 0]] - 1.0)
    m = d.mean(axis=1)
    z = m / (d.std(axis=1, ddof=1) / math.sqrt(8))
    n_pos = (d > 0).sum(axis=1)
    res["serial_true_null"]["empirical_pair_draws"] = {
        "n_pairs": int(len(m)),
        "sd_of_mean8_pct": float(m.std(ddof=1)),
        "mean_within_pair_sd_pct": float(d.std(axis=1, ddof=1).mean()),
        "abs_z_naive_p50": float(np.percentile(np.abs(z), 50)),
        "abs_z_naive_p95": float(np.percentile(np.abs(z), 95)),
        "abs_z_naive_p97_5": float(np.percentile(np.abs(z), 97.5)),
        "abs_z_naive_p99": float(np.percentile(np.abs(z), 99)),
        "abs_z_naive_max": float(np.abs(z).max()),
        "false_positive_rate_at_z_1_96": float((np.abs(z) > 1.96).mean()),
        "false_positive_rate_at_z_3": float((np.abs(z) > 3).mean()),
        "false_positive_rate_at_z_4_16": float((np.abs(z) > 4.16).mean()),
        # The sign test is the usual fallback when the mean looks fragile. It is
        # also anticonservative here, because the eight prompts share a run-level
        # shift, so "all eight moved the same way" is far commoner than 2/2^8.
        "sign_test_fpr_at_p_0_0078_nominal": float(((n_pos == 0) | (n_pos == 8)).mean()),
        "sign_test_fpr_at_p_0_07_nominal": float((n_pos <= 1).mean() + (n_pos >= 7).mean()),
        "sign_test_nominal_p_0_0078": 2.0 / 256.0,
        "sign_test_nominal_p_0_07": 18.0 / 256.0,
        "max_abs_mean8_pct": float(np.abs(m).max()),
        "p_abs_mean8_above_1_0583": float((np.abs(m) > 1.0583).mean()),
    }

    # --- cross-leg coupling: does a fast run make BOTH legs fast? ---
    # Within fingerprint groups, group x prompt centring leaves run movement plus
    # whatever real mechanism differences remain inside the group.
    s_run, m_run, s_cell, m_cell = [], [], [], []
    for g in fp_groups:
        sub_s, sub_m = board.serial[g], board.mtp[g]
        keep = np.isfinite(sub_s).all(axis=1) & np.isfinite(sub_m).all(axis=1)
        sub_s, sub_m = sub_s[keep], sub_m[keep]
        if sub_s.shape[0] < 2:
            continue
        ls = 100.0 * np.log(sub_s)
        lm = 100.0 * np.log(sub_m)
        ls = ls - ls.mean(axis=0, keepdims=True)
        lm = lm - lm.mean(axis=0, keepdims=True)
        s_run.extend(ls.mean(axis=1))
        m_run.extend(lm.mean(axis=1))
        s_cell.extend(ls.ravel())
        m_cell.extend(lm.ravel())
    s_run, m_run = np.array(s_run), np.array(m_run)
    s_cell, m_cell = np.array(s_cell), np.array(m_cell)
    r_run = float(np.corrcoef(s_run, m_run)[0, 1])

    # The correlation is attenuated by mechanism variance in the candidate leg,
    # so it cannot be read as "the legs are decoupled". The regression slope of
    # mtp on serial is not attenuated, because the serial leg is pinned and
    # therefore carries no mechanism signal to correlate with.
    beta = float(np.cov(s_run, m_run, ddof=1)[0, 1] / s_run.var(ddof=1))
    rng_b = np.random.default_rng(1964)
    boot = np.empty(4000)
    for b in range(4000):
        k = rng_b.integers(0, s_run.size, s_run.size)
        boot[b] = np.cov(s_run[k], m_run[k], ddof=1)[0, 1] / s_run[k].var(ddof=1)
    res["cross_leg_coupling"] = {
        "n_receipts": int(s_run.size),
        "corr_run_mean_serial_vs_mtp": r_run,
        "corr_per_cell_serial_vs_mtp": float(np.corrcoef(s_cell, m_cell)[0, 1]),
        "corr_is_attenuated_by_mechanism_variance": True,
        "regression_slope_mtp_on_serial": beta,
        "regression_slope_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "sd_run_mean_serial_pct": float(s_run.std(ddof=1)),
        "sd_run_mean_mtp_pct": float(m_run.std(ddof=1)),
        "verdict": (
            "the board cannot resolve the cross-leg coupling slope: the candidate leg's "
            "mechanism variance is 30x the serial leg's total variance"
        ),
    }

    # var(delta ln raw) = var(delta ln serial) + var(delta ln mtp) - 2 cov
    vs = float(s_run.var(ddof=1))
    vm = float(m_run.var(ddof=1))
    cov = float(np.cov(s_run, m_run, ddof=1)[0, 1])
    var_raw_pred = vs + vm - 2 * cov
    res["variance_identity"] = {
        "note": "exact in logs because raw == serial / mtp; measured on mechanism-contaminated data",
        "var_serial_run_mean": vs,
        "var_mtp_run_mean": vm,
        "cov_serial_mtp_run_mean": cov,
        "predicted_sd_raw_run_mean_pct": float(math.sqrt(max(var_raw_pred, 0.0))),
        "measured_sd_raw_run_mean_pct": per_channel["raw"]["sd_of_run_mean_of_8_pct"],
        "cov_needed_for_raw_to_beat_mtp": vs / 2.0,
        "raw_tighter_than_mtp": bool(var_raw_pred < vm),
        "raw_advantage_pct_of_mtp_sd": float(
            100.0 * (1.0 - math.sqrt(max(var_raw_pred, 0.0)) / math.sqrt(vm))
        ),
    }

    # Same solver plus same fingerprint is the closest the board gets to a rerun.
    # Its lower quartile bounds the candidate-leg noise from above without the
    # outcome-selection bias of a tightness cut on the metric being estimated.
    same_solver = []
    for g in fp_groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                i, j = g[a], g[b]
                if board.r[i]["solverUsername"] == board.r[j]["solverUsername"]:
                    same_solver.append((i, j))
    res["same_solver_same_fingerprint"] = {"n_pairs": len(same_solver)}
    for ch in ("serial", "mtp", "raw"):
        mat = board.chan(ch)
        m8 = np.array([100.0 * (mat[j] / mat[i] - 1.0) for i, j in same_solver]).mean(axis=1)
        res["same_solver_same_fingerprint"][ch] = {
            "sd_of_mean8_pair_delta_pct": float(m8.std(ddof=1)),
            "median_abs_mean8_pct": float(np.median(np.abs(m8))),
            "p25_abs_mean8_pct": float(np.percentile(np.abs(m8), 25)),
            "p10_abs_mean8_pct": float(np.percentile(np.abs(m8), 10)),
        }

    res["deliverable_se_single_receipt_pair_pct"] = {
        "serial_leg_mean_of_8_exact": res["serial_true_null"]["se_single_receipt_pair_pct"],
        "candidate_leg_mean_of_8_upper_bound": per_channel["mtp"]["se_single_receipt_pair_pct"],
        "raw_mean_of_8_upper_bound": per_channel["raw"]["se_single_receipt_pair_pct"],
        "note": (
            "the candidate and raw figures are upper bounds only: no two board receipts share "
            "source, so every candidate-leg set carries real mechanism differences"
        ),
    }
    res["prior_art_reconciliation"] = {
        "FINDING_20_icc": 0.131,
        "FINDING_20_sd_run_level_pct": 0.0821,
        "FINDING_20_sd_within_pct": 0.2120,
        "ledger_L4030_common_mode_sd_pct": 0.0978,
        "FINDING_207_serial_mean8_diff_of_two_pct": 0.1517,
        "FINDING_211_candidate_mean8_diff_of_two_pct": 0.0731,
        "FINDING_211_serial_mean8_diff_of_two_pct": 0.1695,
        "comment": (
            "FINDING 20 and ledger item 4030 already measured a run-level serial component. "
            "This experiment reproduces both at larger n and does not overturn them."
        ),
    }
    return res


# -------------------------------------------------------------------------- Q3


def published_median(raw_row):
    """The track median rule: mean of the two central order statistics of eight."""
    s = np.sort(raw_row)
    return float((s[3] + s[4]) / 2.0)


def q3_reprice(board, q2, q4, base_id="684821ed", arm_id="c47b45be"):
    i, j = board.index(base_id), board.index(arm_id)
    res = {
        "harness": "ranked",
        "pair": f"{base_id} -> {arm_id}",
        "base": {"id8": base_id, "score": float(board.score[i]), "created": board.created[i]},
        "arm": {"id8": arm_id, "score": float(board.score[j]), "created": board.created[j]},
    }

    res["per_prompt"] = []
    for p in range(8):
        res["per_prompt"].append(
            {
                "prompt": board.prompts[p],
                "cand_delta_pct": float(100 * (board.mtp[j, p] / board.mtp[i, p] - 1)),
                "serial_delta_pct": float(100 * (board.serial[j, p] / board.serial[i, p] - 1)),
                "raw_delta_pct": float(100 * (board.raw[j, p] / board.raw[i, p] - 1)),
                "edl_base": float(board.edl[i, p]),
                "edl_arm": float(board.edl[j, p]),
                "edl_identical": bool(board.edl[i, p] == board.edl[j, p]),
                "nondraft_identical": bool(board.nondraft[i, p] == board.nondraft[j, p]),
            }
        )
    res["single_mechanism_check"] = {
        "edl_digit_identical_all_8": all(x["edl_identical"] for x in res["per_prompt"]),
        "nondraft_identical_all_8": all(x["nondraft_identical"] for x in res["per_prompt"]),
    }

    res["legs"] = {}
    for ch, label in (("mtp", "candidate"), ("serial", "serial"), ("raw", "raw")):
        d, _ = pair_delta(board, ch, i, j)
        st = pair_stats(d)
        st["sign_test_p"] = sign_test_p(st["n_positive"], st["k"])
        res["legs"][label] = st

    # Four candidate denominators for the same +1.0583 %, weakest evidence last.
    infl = q2["icc_inflation_factor"]
    cand = res["legs"]["candidate"]["mean_pct"]
    serial_null = q2["serial_true_null"]["se_single_receipt_pair_pct"]
    rep_sd = q4["replicate_analysis"]["sd_of_one_estimate_pct"]
    res["denominators"] = {
        "A_naive_between_prompt": {
            "se_pct": res["legs"]["candidate"]["se_naive_pct"],
            "z": cand / res["legs"]["candidate"]["se_naive_pct"],
            "basis": "sd of the 8 per-prompt deltas / sqrt(8), as FINDING 259 quoted it",
            "valid": False,
            "why": "treats 8 prompts as independent replicates of a run-level effect",
        },
        "B_naive_times_icc_inflation": {
            "se_pct": res["legs"]["candidate"]["se_naive_pct"] * infl,
            "z": cand / (res["legs"]["candidate"]["se_naive_pct"] * infl),
            "basis": f"denominator A inflated by sqrt(1 + 7 * ICC) = {infl:.3f}",
            "valid": True,
            "why": "corrects A for within-run correlation measured on the pinned serial leg",
        },
        "C_serial_true_null": {
            "se_pct": serial_null,
            "z": cand / serial_null,
            "basis": "measured sd of the mean-of-8 pair delta on the byte-identical serial leg",
            "valid": True,
            "why": "exact, but assumes the candidate leg is no noisier than the serial leg",
        },
        "D_same_arm_replicates": {
            "se_pct": rep_sd,
            "z": cand / rep_sd,
            "basis": "spread of the two independent ranked receipts of this same mechanism",
            "valid": True,
            "why": "the only denominator measured on the candidate leg for THIS arm",
            "dof": q4["replicate_analysis"]["dof"],
        },
    }

    pub_i, pub_j = published_median(board.raw[i]), published_median(board.raw[j])
    res["published"] = {
        "base_median_recomputed": pub_i,
        "arm_median_recomputed": pub_j,
        "base_officialScore": float(board.score[i]),
        "arm_officialScore": float(board.score[j]),
        "median_rule_reproduces_base": bool(abs(pub_i - board.score[i]) < 1e-9),
        "median_rule_reproduces_arm": bool(abs(pub_j - board.score[j]) < 1e-9),
        "published_delta_pct": float(100 * (pub_j / pub_i - 1)),
    }

    # The advisor's defence: the serial leg fell, so the run was fast, so a slow
    # candidate cannot be run-level drift. That argument needs the two legs to be
    # coupled. Price it with the measured coupling.
    beta = q2["cross_leg_coupling"]["regression_slope_mtp_on_serial"]
    blo, bhi = q2["cross_leg_coupling"]["regression_slope_ci95"]
    serial_pair_delta = res["legs"]["serial"]["mean_pct"]
    res["opposite_sign_argument"] = {
        "claim": (
            "the serial leg moved -0.2449 %, so the run was fast; a fast run cannot "
            "make the candidate slow, therefore drift cannot explain the effect"
        ),
        "requires": "a positive cross-leg coupling slope beta",
        "beta": beta,
        "beta_ci95": [blo, bhi],
        "observed_serial_pair_delta_pct": serial_pair_delta,
        "implied_candidate_drift_pct": beta * serial_pair_delta,
        "implied_candidate_drift_range_pct": sorted([blo * serial_pair_delta, bhi * serial_pair_delta]),
        "verdict": (
            "directionally supported but not quantitatively usable: beta = "
            f"{beta:.2f} with CI [{blo:.2f}, {bhi:.2f}] spans zero, so the board cannot "
            "confirm that a fast serial leg implies a fast candidate leg"
        ),
    }
    return res


# -------------------------------------------------------------------------- Q4


def chi2_sigma_ci(s, dof):
    """95 % CI for a standard deviation from a chi-square with `dof` degrees."""
    try:
        from scipy.stats import chi2
    except ImportError:  # small-dof fallback keeps the script dependency-light
        table = {1: (0.000982, 5.0239), 2: (0.0506, 7.3778), 3: (0.2158, 9.3484)}
        if dof not in table:
            return [float("nan"), float("nan")]
        lo_q, hi_q = table[dof]
    else:
        lo_q, hi_q = float(chi2.ppf(0.025, dof)), float(chi2.ppf(0.975, dof))
    return [float(s * math.sqrt(dof / hi_q)), float(s * math.sqrt(dof / lo_q))]


def student_t_sf(t, dof):
    """Two-sided p from Student's t. Closed forms cover the small dof used here."""
    t = abs(t)
    try:
        from scipy.stats import t as tdist
    except ImportError:
        if dof == 1:  # Cauchy
            return float(1.0 - 2.0 * math.atan(t) / math.pi)
        if dof == 2:
            return float(1.0 - t / math.sqrt(2.0 + t * t))
        return float(math.erfc(t / math.sqrt(2.0)))  # normal approximation
    return float(2 * tdist.sf(t, dof))


def q4_onepass6_replicates(board, base_id="684821ed", arm_ids=("c47b45be", "24fb4012")):
    """FINDING 259 and FINDING 279 measure the same arm. Treat them as replicates.

    Both arms declare the same parent (`eb5eadc` / `684821ed`) and the same two
    integer literals: the generated Metal width plan `(6, 3) -> (6, 6)` and the
    launch witness `activeInputGroups case 6: inputsPerGroup = 6`. Their tree
    behaviour is digit-identical to the base on all eight prompts, so the board
    treats them as one mechanism measured twice.
    """
    res = {"harness": "ranked", "base": base_id, "arms": list(arm_ids)}
    i = board.index(base_id)
    res["base_detail"] = {
        "id8": base_id,
        "solver": board.r[i]["solverUsername"],
        "score": float(board.score[i]),
        "created": board.created[i],
    }

    arms = []
    for a in arm_ids:
        j = board.index(a)
        row = {
            "id8": a,
            "solver": board.r[j]["solverUsername"],
            "score": float(board.score[j]),
            "created": board.created[j],
            "commit": board.r[j].get("submissionCommitSha"),
            "edl_digit_identical_to_base": bool((board.edl[i] == board.edl[j]).all()),
            "nondraft_identical_to_base": bool((board.nondraft[i] == board.nondraft[j]).all()),
            "head_identical_to_base": bool(
                all(
                    p["head_provenance_sha256"] == q["head_provenance_sha256"]
                    for p, q in zip(
                        board.r[i]["officialMetrics"]["per_prompt"],
                        board.r[j]["officialMetrics"]["per_prompt"],
                    )
                )
            ),
        }
        for ch, label in (("mtp", "candidate"), ("serial", "serial"), ("raw", "raw")):
            d, _ = pair_delta(board, ch, i, j)
            row[label] = pair_stats(d)
        row["published_delta_pct"] = float(
            100 * (published_median(board.raw[j]) / published_median(board.raw[i]) - 1)
        )
        arms.append(row)
    res["arm_detail"] = arms

    # The replicate spread IS the error bar for a single-receipt-pair ranked A/B.
    est = np.array([a["candidate"]["mean_pct"] for a in arms])
    m = len(est)
    s = float(est.std(ddof=1))
    mean = float(est.mean())
    se_mean = s / math.sqrt(m)
    t = mean / se_mean if se_mean else float("nan")
    res["replicate_analysis"] = {
        "n_replicates": m,
        "candidate_leg_estimates_pct": [float(v) for v in est],
        "spread_pct": float(est.max() - est.min()),
        "pooled_estimate_pct": mean,
        "sd_of_one_estimate_pct": s,
        "sd_ci95": chi2_sigma_ci(s, m - 1),
        "se_of_pooled_pct": se_mean,
        "t_statistic": t,
        "dof": m - 1,
        "p_two_sided": student_t_sf(t, m - 1),
        "quoted_se_finding_259": 0.2542,
        "quoted_se_finding_279": 0.0799,
        "understatement_vs_finding_259": float(s / 0.2542),
        "understatement_vs_finding_279": float(s / 0.0799),
    }
    for ch in ("serial", "raw"):
        v = np.array([a[ch]["mean_pct"] for a in arms])
        res["replicate_analysis"][f"{ch}_estimates_pct"] = [float(x) for x in v]
        res["replicate_analysis"][f"{ch}_sd_of_one_estimate_pct"] = float(v.std(ddof=1))

    # Provenance: confirm the -0.0390 % / 0.0799 digits belong to the candidate
    # leg of base -> 24fb4012 and to nothing else on the board.
    res["provenance_search"] = {"target_mean_pct": -0.0390, "target_se_pct": 0.0799}
    wide = []
    for ch in CHANNELS:
        mat = board.chan(ch)
        for j in range(board.n):
            if j == i or not np.isfinite(mat[j]).all() or not np.isfinite(mat[i]).all():
                continue
            st = pair_stats(100.0 * (mat[j] / mat[i] - 1.0))
            if abs(st["mean_pct"] + 0.0390) < 0.0015 and abs(st["se_naive_pct"] - 0.0799) < 0.0015:
                wide.append({"pair": f"{base_id} -> {board.ids[j]}", "channel": ch, **st})
    res["provenance_search"]["matches_from_base"] = wide
    res["provenance_search"]["n_matches"] = len(wide)
    res["provenance_search"]["verdict"] = (
        "FINDING 279's figure is a real candidate-leg measurement, not a mislabelled prefill number"
        if any(w["channel"] == "mtp" for w in wide)
        else "no candidate-leg source found for the quoted digits"
    )
    return res


# ------------------------------------------------------------- round recovery


def recover_rounds(edl, decode_tokens=512):
    """Smallest feasible round count N from the exact rational edl = P / N.

    Constraints: N + A = decode_tokens and A <= P = edl * N.
    Returns (N, candidates_considered, ambiguous).
    """
    fr = Fraction(edl).limit_denominator(10**6)
    den = fr.denominator
    feasible = []
    n = den
    while n <= decode_tokens:
        P = fr * n
        if P.denominator == 1:
            A = decode_tokens - n
            if 0 <= A <= int(P):
                feasible.append(n)
        n += den
    if not feasible:
        return None, [], True
    # Ambiguous only if a second feasible N implies a plausible acceptance rate.
    return feasible[0], feasible, len(feasible) > 1


# ------------------------------------------------------------------- reporting


def campaign_rule(q1, q2, q4):
    """The required deliverable: a rule usable without re-deriving anything here.

    Everything a future reader needs is a channel name, a count of independent
    ranked receipts behind the arm, and the observed effect size.
    """
    tn = q2["serial_true_null"]
    e = tn["empirical_pair_draws"]
    ra = q4["replicate_analysis"]
    se_serial = tn["se_single_receipt_pair_pct"]
    se_cand = ra["sd_of_one_estimate_pct"]
    se_raw = ra["raw_sd_of_one_estimate_pct"]
    return {
        "id": "PROPOSED RULE E164",
        "title": "Significance of a board-deconvolved ranked A/B",
        "scope": (
            "any claim of the form 'arm X moves the ranked serial leg, candidate leg or raw "
            "by D %' that is obtained by differencing officialMetrics.per_prompt cells of two "
            "Yukon receipts. harness=ranked only."
        ),
        "denominator_ladder": [
            {
                "rung": 1,
                "when": (
                    "two or more independent ranked receipts exist for the SAME arm against "
                    "the SAME base"
                ),
                "use": (
                    "the sample sd of those single-receipt-pair estimates, with Student-t on "
                    "n-1 dof"
                ),
                "why": "this is the only denominator that prices the whole receipt-level error",
            },
            {
                "rung": 2,
                "when": "only one ranked receipt exists for the arm",
                "use": "the channel floor below, as a LOWER bound on the standard error",
                "floor_se_single_receipt_pair_mean_of_8_pct": {
                    "serial": se_serial,
                    "candidate": se_cand,
                    "raw": se_raw,
                },
                "why": (
                    "the serial floor is exact: the pinned serial build is byte-identical on "
                    "every receipt, so 199798 board pairs are a true null. The candidate and "
                    "raw floors come from the single same-arm replicate pair on the board "
                    "(dof 1), so they are the weakest supported values, not point estimates"
                ),
            },
            {
                "rung": 3,
                "when": "never",
                "use": "the between-prompt sd of one receipt pair divided by sqrt(8)",
                "why": (
                    "it prices the prompt-to-prompt spread of one draw, not the error of the "
                    "effect. The eight prompts are positively correlated on every off-diagonal"
                ),
            },
        ],
        "critical_values_for_the_naive_between_prompt_z": {
            "note": (
                "if a naive between-prompt z must be quoted, calibrate it against the measured "
                "true null instead of the normal table"
            ),
            "nominal_1_96_true_false_positive_rate": e["false_positive_rate_at_z_1_96"],
            "nominal_3_00_true_false_positive_rate": e["false_positive_rate_at_z_3"],
            "true_5_pct_critical_value": e["abs_z_naive_p95"],
            "true_2_5_pct_critical_value": e["abs_z_naive_p97_5"],
            "true_1_pct_critical_value": e["abs_z_naive_p99"],
        },
        "sign_test_is_also_anticonservative": {
            "all_8_same_sign_nominal_p": e["sign_test_nominal_p_0_0078"],
            "all_8_same_sign_measured_rate": e["sign_test_fpr_at_p_0_0078_nominal"],
            "at_least_7_of_8_nominal_p": e["sign_test_nominal_p_0_07"],
            "at_least_7_of_8_measured_rate": e["sign_test_fpr_at_p_0_07_nominal"],
            "cause": "the eight prompts share a run-level shift",
        },
        "why_the_prompts_are_correlated": {
            "icc": q1["icc"]["icc"],
            "icc_ci95": q1["icc"]["icc_ci95"],
            "mean_off_diagonal_corr": q1["cross_prompt_correlation"]["mean_off_diagonal"],
            "fraction_positive_off_diagonals": q1["cross_prompt_correlation"][
                "fraction_positive"
            ],
            "design_effect": q1["icc"]["design_effect"],
            "se_inflation_factor": q1["icc"]["se_inflation_factor"],
        },
        "minimum_detectable_effect_pct": {
            "serial_one_receipt_pair": 1.96 * se_serial,
            "candidate_one_receipt_pair": 1.96 * se_cand,
            "raw_one_receipt_pair": 1.96 * se_raw,
            "note": (
                "a single ranked receipt pair cannot resolve a candidate-leg effect smaller "
                "than about this size. Report anything smaller as a bound, not a measurement"
            ),
        },
        "falsification_stays_valid": (
            "a wide error bar still refutes a far larger prediction. An observed candidate-leg "
            "effect of 1.06 % with a 0.78 % standard error refutes a predicted 28.4 % at more "
            "than 30 sigma. State 'the effect is below X'; do not state 'the effect is exactly X'"
        ),
        "reporting_requirement": [
            "name the channel: serial, candidate or raw",
            "name the number of independent ranked receipts behind the arm",
            "name which rung of the ladder produced the denominator",
            "quote a one-receipt candidate-leg effect as a bound, never as a point estimate",
        ],
        "worked_example_finding_259": {
            "claim": "+1.0583 % on the candidate leg, se 0.2542, z 4.16",
            "rung_used_by_the_claim": 3,
            "correct_rung": 1,
            "correct_se_pct": se_cand,
            "correct_t": ra["t_statistic"],
            "correct_p_two_sided": ra["p_two_sided"],
            "verdict": "not resolved. The two replicates of this arm differ by "
            f"{ra['spread_pct']:.4f} pp",
        },
    }


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(prov, inv, q1, q2, q3, q4, rule):
    L = []
    a = L.append
    a("=" * 78)
    a("E164 - run-to-run error bar for a two-receipt ranked A/B    harness=ranked")
    a("=" * 78)
    a(f"board            {prov['board_path']}")
    a(f"receipts scored  {prov['rows_scored_with_per_prompt']} of {prov['rows_total']}")
    a(f"window           {prov['created_first']} .. {prov['created_last']}")
    a(f"harness          decode_tokens={inv['decode_tokens']} pairs_per_prompt={inv['pairs_per_prompt']} "
      f"prompts={inv['prompt_count']}")
    a(f"weights hash     {inv['qwen_mtp_weights_hash'][:16]}...  (identical on every receipt)")

    a("")
    a("-" * 78)
    a("Q1  dispersion of the pinned serial leg")
    a("-" * 78)
    a("prompt    mean s/tok     rel sd %   p2.5        p97.5")
    for p in q1["per_prompt"]:
        a(f"{p['prompt']}  {p['mean_s_per_token']:.7f}  {p['rel_sd_pct']:8.4f}   "
          f"{p['p2_5']:.7f}  {p['p97_5']:.7f}")
    pl = q1["pooled"]
    a(f"pooled    {pl['mean_s_per_token']:.7f}  {pl['rel_sd_pct']:8.4f}   "
      f"{pl['p2_5']:.7f}  {pl['p97_5']:.7f}   n={pl['n_cells']} cells")
    ic = q1["icc"]
    a("")
    a(f"  sd total (per prompt cell)            {fmt(ic['sd_total_pct'])} %")
    a(f"  sd within receipt (prompt jitter)     {fmt(ic['sd_within_receipt_pct'])} %")
    a(f"  sd run level (common shift)           {fmt(ic['sd_run_level_pct'])} %")
    a(f"  INTRACLASS CORRELATION                {fmt(ic['icc'])}   CI95 "
      f"[{fmt(ic['icc_ci95'][0])}, {fmt(ic['icc_ci95'][1])}]")
    a(f"  sd of run mean, measured              {fmt(ic['sd_of_run_mean_measured_pct'])} %")
    a(f"  sd of run mean, if prompts iid        {fmt(ic['sd_of_run_mean_if_independent_pct'])} %")
    a(f"  design effect / SE inflation          {fmt(ic['design_effect'],3)} / "
      f"{fmt(ic['se_inflation_factor'],3)}x")
    cc = q1["cross_prompt_correlation"]
    a(f"  cross-prompt corr: mean off-diag {fmt(cc['mean_off_diagonal'])}, range "
      f"[{fmt(cc['min_off_diagonal'])}, {fmt(cc['max_off_diagonal'])}], "
      f"{cc['fraction_positive']*100:.0f} % of {cc['n_off_diagonal_pairs']} pairs positive")
    dr = q1["drift"]
    a(f"  drift: within-day rel sd {fmt(dr['mean_within_day_rel_sd_pct'])} % vs between-day "
      f"{fmt(dr['between_day_rel_sd_of_means_pct'])} % over {dr['n_days']} days")
    f329 = q1["finding_329_check"]
    a(f"  FINDING 329: daily median range [{f329['daily_median_range'][0]:.6f}, "
      f"{f329['daily_median_range'][1]:.6f}] -> {f329['verdict'].upper()}")

    a("")
    a("-" * 78)
    a("Q2  does raw cancel the run-level component?")
    a("-" * 78)
    for key in ("submissionCommitSha", "promotedSourceRef"):
        g = q2[f"replicates_by_{key}"]
        a(f"  replicates by {key:20s} groups={g['n_groups']} pairs={g['n_pairs']}")
    fg = q2["replicates_by_behaviour_fingerprint"]
    a(f"  replicates by behaviour fingerprint    groups={fg['n_groups']} "
      f"receipts={fg['n_receipts']} pairs={fg['n_pairs']}")
    a(f"  raw == serial/mtp to {q2['raw_identity']['max_relative_deviation']:.2e} relative")
    a("")
    a("  channel   sd(run mean of 8) %   sd within receipt %   SE of a 1-pair A/B %")
    for ch in CHANNELS:
        d = q2["fingerprint_replicate_dispersion"][ch]
        a(f"  {ch:9s} {d['sd_of_run_mean_of_8_pct']:18.4f}   {d['sd_within_receipt_pct']:18.4f}   "
          f"{d['se_single_receipt_pair_pct']:18.4f}")
    tn = q2["serial_true_null"]
    a(f"  serial TRUE NULL  sd(run mean) {fmt(tn['sd_of_run_mean_of_8_pct'])} %  "
      f"SE of a 1-pair A/B {fmt(tn['se_single_receipt_pair_pct'])} %")
    e = tn["empirical_pair_draws"]
    a(f"  naive |z| on {e['n_pairs']} TRUE NULL pairs: p50 {fmt(e['abs_z_naive_p50'],2)}  "
      f"p95 {fmt(e['abs_z_naive_p95'],2)}  p99 {fmt(e['abs_z_naive_p99'],2)}  "
      f"max {fmt(e['abs_z_naive_max'],2)}")
    a(f"  false positive rate of naive z: |z|>1.96 {e['false_positive_rate_at_z_1_96']*100:.1f} %   "
      f"|z|>3 {e['false_positive_rate_at_z_3']*100:.1f} %   "
      f"|z|>4.16 {e['false_positive_rate_at_z_4_16']*100:.2f} %")
    cl = q2["cross_leg_coupling"]
    a(f"  cross-leg corr(run-mean serial, run-mean mtp) = {fmt(cl['corr_run_mean_serial_vs_mtp'])} "
      f"(ATTENUATED by mechanism variance, do not read as decoupling)")
    a(f"  cross-leg regression slope beta = {fmt(cl['regression_slope_mtp_on_serial'],3)} "
      f"CI95 [{fmt(cl['regression_slope_ci95'][0],3)}, {fmt(cl['regression_slope_ci95'][1],3)}]  "
      f"n={cl['n_receipts']}  -> unresolved, CI spans zero")
    vi = q2["variance_identity"]
    a(f"  var identity: cov {vi['cov_serial_mtp_run_mean']:.6f} vs cov needed "
      f"{vi['cov_needed_for_raw_to_beat_mtp']:.6f} -> raw tighter than mtp: "
      f"{vi['raw_tighter_than_mtp']} (by only {vi['raw_advantage_pct_of_mtp_sd']:.2f} % of the mtp sd)")
    ss = q2["same_solver_same_fingerprint"]
    a(f"  same solver + same fingerprint, {ss['n_pairs']} pairs:")
    for ch in ("serial", "mtp", "raw"):
        d = ss[ch]
        a(f"     {ch:7s} sd(mean8) {d['sd_of_mean8_pair_delta_pct']:8.4f} %   "
          f"median |mean8| {d['median_abs_mean8_pct']:7.4f} %   p25 {d['p25_abs_mean8_pct']:7.4f} %   "
          f"p10 {d['p10_abs_mean8_pct']:7.4f} %")
    dl = q2["deliverable_se_single_receipt_pair_pct"]
    a("  DELIVERABLE SE of a single-receipt-pair ranked A/B (mean of 8):")
    a(f"     serial leg        {fmt(dl['serial_leg_mean_of_8_exact'])} %   (exact, true null)")
    a(f"     candidate leg    <{fmt(dl['candidate_leg_mean_of_8_upper_bound'])} %   (vacuous upper bound)")
    a(f"     raw              <{fmt(dl['raw_mean_of_8_upper_bound'])} %   (vacuous upper bound)")
    pa = q2["prior_art_reconciliation"]
    a(f"  prior art: FINDING 20 ICC {pa['FINDING_20_icc']} / run-level sd "
      f"{pa['FINDING_20_sd_run_level_pct']} %, ledger 4030 common-mode sd "
      f"{pa['ledger_L4030_common_mode_sd_pct']} %, FINDING 207 serial pair "
      f"{pa['FINDING_207_serial_mean8_diff_of_two_pct']} %")

    a("")
    a("-" * 78)
    a(f"Q3  re-price FINDING 259 / 348   {q3['pair']}")
    a("-" * 78)
    a(f"  single mechanism: edl digit-identical on 8/8 = "
      f"{q3['single_mechanism_check']['edl_digit_identical_all_8']}, "
      f"non-drafting identical = {q3['single_mechanism_check']['nondraft_identical_all_8']}")
    a("  prompt      cand d%    serial d%     raw d%")
    for p in q3["per_prompt"]:
        a(f"  {p['prompt']}  {p['cand_delta_pct']:+8.4f}   {p['serial_delta_pct']:+8.4f}   "
          f"{p['raw_delta_pct']:+8.4f}")
    a("")
    a("  leg         mean %     sd %    naive se   naive z    sign p")
    for name in ("candidate", "serial", "raw"):
        s = q3["legs"][name]
        a(f"  {name:10s} {s['mean_pct']:+8.4f} {s['sd_pct']:8.4f} {s['se_naive_pct']:9.4f} "
          f"{s['z_naive']:+9.2f} {s['sign_test_p']:9.4f}")
    pb = q3["published"]
    a(f"  published median delta {pb['published_delta_pct']:+.4f} %  "
      f"(median rule reproduces both receipts: {pb['median_rule_reproduces_base']} / "
      f"{pb['median_rule_reproduces_arm']})")
    a("")
    a("  THE SAME +1.0583 % UNDER FOUR DENOMINATORS")
    a("  key                          se %       z      valid   basis")
    for k in ("A_naive_between_prompt", "B_naive_times_icc_inflation", "C_serial_true_null",
              "D_same_arm_replicates"):
        d = q3["denominators"][k]
        a(f"  {k:28s} {d['se_pct']:7.4f} {d['z']:+7.2f}   {str(d['valid']):5s}   {d['basis'][:44]}")
    oa = q3["opposite_sign_argument"]
    a(f"  opposite-sign defence: beta = {fmt(oa['beta'],2)} CI [{fmt(oa['beta_ci95'][0],2)}, "
      f"{fmt(oa['beta_ci95'][1],2)}] -> implied candidate drift "
      f"{fmt(oa['implied_candidate_drift_pct'])} % (range "
      f"{fmt(oa['implied_candidate_drift_range_pct'][0])} to "
      f"{fmt(oa['implied_candidate_drift_range_pct'][1])} %)")

    a("")
    a("-" * 78)
    a("Q4  FINDING 259 and FINDING 279 are two replicates of ONE arm")
    a("-" * 78)
    bd = q4["base_detail"]
    a(f"  base {bd['id8']} {bd['solver']} {bd['score']:.8f}  {bd['created']}")
    a("  arm        solver          score        cand mean %   serial mean %   published %   tree==base")
    for arm in q4["arm_detail"]:
        same = arm["edl_digit_identical_to_base"] and arm["nondraft_identical_to_base"] and arm[
            "head_identical_to_base"
        ]
        a(f"  {arm['id8']}   {arm['solver']:14s} {arm['score']:.8f}  "
          f"{arm['candidate']['mean_pct']:+10.4f}   {arm['serial']['mean_pct']:+11.4f}   "
          f"{arm['published_delta_pct']:+10.4f}   {same}")
    ra = q4["replicate_analysis"]
    a("")
    a(f"  replicate spread on the candidate leg   {ra['spread_pct']:.4f} pp")
    a(f"  pooled effect                          {ra['pooled_estimate_pct']:+.4f} %")
    a(f"  sd of ONE single-pair estimate         {ra['sd_of_one_estimate_pct']:.4f} %   "
      f"CI95 [{ra['sd_ci95'][0]:.4f}, {ra['sd_ci95'][1]:.4f}]  dof={ra['dof']}")
    a(f"  t = {ra['t_statistic']:+.3f} on {ra['dof']} dof, two-sided p = {ra['p_two_sided']:.4f}")
    a(f"  FINDING 259 quoted se {ra['quoted_se_finding_259']:.4f} -> understated "
      f"{ra['understatement_vs_finding_259']:.1f}x")
    a(f"  FINDING 279 quoted se {ra['quoted_se_finding_279']:.4f} -> understated "
      f"{ra['understatement_vs_finding_279']:.1f}x")
    ps = q4["provenance_search"]
    a(f"  provenance of -0.0390 % +/- 0.0799: {ps['n_matches']} match(es) from the base")
    for m in ps["matches_from_base"]:
        a(f"     {m['pair']}  channel={m['channel']:8s} mean {m['mean_pct']:+.4f} % "
          f"sd {m['sd_pct']:.4f} se {m['se_naive_pct']:.4f}")
    a(f"  -> {ps['verdict']}")

    a("")
    a("=" * 78)
    a(f"{rule['id']}  {rule['title']}")
    a("=" * 78)
    a(f"  scope: {rule['scope']}")
    a("")
    a("  denominator ladder, use the first rung that applies:")
    for r in rule["denominator_ladder"]:
        a(f"    rung {r['rung']}  when {r['when']}")
        a(f"             use  {r['use']}")
        fl = r.get("floor_se_single_receipt_pair_mean_of_8_pct")
        if fl:
            a(f"             floor se, mean of 8, one receipt pair:  serial {fl['serial']:.4f} %   "
              f"candidate {fl['candidate']:.4f} %   raw {fl['raw']:.4f} %")
        a(f"             why  {r['why']}")
    cv = rule["critical_values_for_the_naive_between_prompt_z"]
    a("")
    a("  if a naive between-prompt z is unavoidable, use the measured critical values:")
    a(f"    nominal |z|>1.96 really fires {cv['nominal_1_96_true_false_positive_rate']*100:5.1f} % "
      f"of the time under the true null")
    a(f"    nominal |z|>3.00 really fires {cv['nominal_3_00_true_false_positive_rate']*100:5.1f} % "
      f"of the time under the true null")
    a(f"    true 5 % critical value {cv['true_5_pct_critical_value']:.2f}   "
      f"2.5 % {cv['true_2_5_pct_critical_value']:.2f}   1 % {cv['true_1_pct_critical_value']:.2f}")
    st = rule["sign_test_is_also_anticonservative"]
    a(f"    sign test, all 8 same sign: nominal p {st['all_8_same_sign_nominal_p']:.4f}, "
      f"measured rate {st['all_8_same_sign_measured_rate']:.4f}")
    a(f"    sign test, 7 or 8 of 8:     nominal p {st['at_least_7_of_8_nominal_p']:.4f}, "
      f"measured rate {st['at_least_7_of_8_measured_rate']:.4f}")
    mde = rule["minimum_detectable_effect_pct"]
    a("")
    a(f"  minimum detectable effect from ONE receipt pair at 95 %:  serial "
      f"{mde['serial_one_receipt_pair']:.3f} %   candidate {mde['candidate_one_receipt_pair']:.3f} %"
      f"   raw {mde['raw_one_receipt_pair']:.3f} %")
    a(f"  {rule['falsification_stays_valid']}")
    a("  every board-derived ranked A/B must report:")
    for line in rule["reporting_requirement"]:
        a(f"    - {line}")
    we = rule["worked_example_finding_259"]
    a(f"  worked example: FINDING 259 claimed {we['claim']} on rung {we['rung_used_by_the_claim']}. "
      f"On rung {we['correct_rung']} it is se {we['correct_se_pct']:.4f} %, t {we['correct_t']:+.3f}, "
      f"p {we['correct_p_two_sided']:.3f}. {we['verdict']}")
    return "\n".join(L)


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=LIVE_BOARD if os.path.exists(LIVE_BOARD) else FROZEN_BOARD)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "e164-artifacts"))
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    board, prov = load_board(args.board)
    inv = check_invariants(board)
    q1 = q1_serial_dispersion(board)
    q2 = q2_raw_versus_candidate(board, q1)
    q4 = q4_onepass6_replicates(board)
    q3 = q3_reprice(board, q2, q4)
    rule = campaign_rule(q1, q2, q4)

    # Q4 supersedes the mechanism-contaminated fingerprint bound carried by Q2.
    floors = rule["denominator_ladder"][1]["floor_se_single_receipt_pair_mean_of_8_pct"]
    deliv = q2["deliverable_se_single_receipt_pair_pct"]
    deliv["candidate_leg_mean_of_8_same_arm_replicate"] = floors["candidate"]
    deliv["raw_mean_of_8_same_arm_replicate"] = floors["raw"]

    os.makedirs(args.out, exist_ok=True)
    result = {
        "experiment": "e164-ranked-ab-error-bar",
        "harness": "ranked",
        "provenance": prov,
        "harness_invariants": inv,
        "q1_serial_dispersion": q1,
        "q2_raw_versus_candidate": q2,
        "q3_reprice_finding_259": q3,
        "q4_onepass6_replicates": q4,
        "proposed_campaign_rule": rule,
    }
    with open(os.path.join(args.out, "e164-results.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    text = report(prov, inv, q1, q2, q3, q4, rule)
    with open(os.path.join(args.out, "e164-report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)

    if args.wandb:
        log_wandb(board, result)


def log_followups(wandb, run, out):
    """Fold the E1/E2/E3 and red-team artifacts into the same W&B run."""

    def read(name):
        path = os.path.join(out, name)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    e1 = read("e164-e1-median-lever.json")
    if e1:
        run.summary.update(
            {
                "e1/baseline_published": e1["baseline_published"],
                "e1/published_per_ms": e1["marginal"]["analytic_published_per_ms_at_zero"],
                "e1/any_reachable_crossing": e1["any_reachable_crossing"],
                "e1/smallest_positive_crossing_ms": e1["smallest_positive_crossing_delta_ms"],
                "e1/max_evaluable_delta_ms": e1["max_evaluable_delta_ms"],
                "e1/full_prefill_removal_gain": e1["extended_sweep"][-1]["delta_published"],
                "e1/ms_to_close_frontier_gap": e1["gap_accounting"]["uniform_ms_needed_to_close"],
                "e1/prefill_share_of_gap_pct": e1["gap_accounting"][
                    "prefill_share_of_published_gap_pct"
                ],
            }
        )
        t = wandb.Table(
            columns=["delta_ms", "published", "delta_published", "delta_pct", "per_ms", "central"]
        )
        for s in e1["sweep"]:
            t.add_data(
                s["delta_ms"],
                s["published"],
                s["delta_published"],
                s["delta_published_pct"],
                s["published_per_ms"],
                ", ".join(s["central_prompts"]),
            )
        wandb.log({"e1/uniform_lever_sweep": t})

        d = wandb.Table(columns=["prompt", "raw_per_ms"])
        for k, v in e1["marginal"]["per_prompt_raw_per_ms"].items():
            d.add_data(k, v)
        wandb.log({"e1/per_prompt_marginal": d})

    e23 = read("e164-e2e3.json")
    if e23:
        run.summary.update(
            {
                "e2/n_consistent_low_solvers": len(e23["e2_consistent_low_solvers"]),
                "e2/n_always_below_pack_solvers": len(e23["e2_always_below_pack_solvers"]),
                "e2/solver_median_residual_sd_ms": e23[
                    "e2_solver_median_residual_spread"
                ]["sd"],
                "e2/anova_F": e23["e2_anova"]["F"],
                "e3/top12_spread": e23["e3c"]["top12_spread"],
                "e3/first_minus_second": e23["e3c"]["first_minus_second"],
                "e3/run_level_neutral_sd_units": e23["e3c"][
                    "run_level_neutral_sd_units_at_beta_point"
                ],
                "e3/selection_slope_best": e23["e3b_slope_best"]["slope"],
                "e3/selection_slope_median": e23["e3b_slope_median"]["slope"],
            }
        )
        for key, title in (
            ("e3_neutral_all", "e3/neutral_all"),
            ("e3_neutral_excluding_anomaly", "e3/neutral_excluding_anomaly"),
            ("e3_neutral_prefill_corrected", "e3/neutral_prefill_corrected"),
        ):
            col = "neutral_prefill_corrected" if "corrected" in key else "neutral"
            t = wandb.Table(columns=["rank", "solver", "receipt", "day", col, "official", "prefill_ms"])
            for i, x in enumerate(e23[key], start=1):
                t.add_data(i, x["solver"], str(x["id"])[:8], x["day"], x[col], x["official"], x["prefill_ms"])
            wandb.log({title: t})

        t = wandb.Table(columns=["rank", "solver", "top2_mean", "n"])
        for i, x in enumerate(e23["e3_neutral_top2_mean"], start=1):
            t.add_data(i, x["solver"], x["top2_mean"], x["n"])
        wandb.log({"e3/neutral_top2_mean": t})

        t = wandb.Table(columns=["solver", "n", "median", "min", "max", "median_day_residual"])
        for x in e23["e2_solvers"]:
            t.add_data(x["solver"], x["n"], x["median"], x["min"], x["max"], x["median_day_residual"])
        wandb.log({"e2/per_solver_prefill": t})

    rt = read("e164-redteam.json")
    if rt:
        a1 = rt["a1"]["paired_difference_regression"]
        a4 = rt["a1"]["a4_is_the_affine_form_forced"]
        pref = rt["a1"]["a1_prefill_attribution"]
        run.summary.update(
            {
                "a1/delta_s_ms_per_round": a1["delta_s_ms_per_round"],
                "a1/delta_s_se": a1["se_delta_s"],
                "a1/delta_s_t": a1["t_delta_s"],
                "a1/delta_s_ci95_lo": a1["delta_s_ci95"][0],
                "a1/delta_s_ci95_hi": a1["delta_s_ci95"][1],
                "a1/delta_h_ms_per_row": a1["delta_h_ms_per_row"],
                "a1/delta_h_t": a1["t_delta_h"],
                "a1/survives_at_2se": a1["survives_at_2se"],
                "a1/sigma_ms": a1["sigma_ms"],
                "a1/prefill_share_of_delta_s": pref[
                    "share_of_delta_s_explained_by_the_prefill_probe"
                ],
                "a4/sse_two_param_affine": a4["sse_two_param_affine_in_rows"],
                "a4/sse_constant_leg_gap": a4["sse_one_param_constant_leg_gap"],
                "a4/D_rel_sd_pct": a4["D_rel_sd_pct"],
                "a4/corr_D_vs_rows": a4["corr_D_vs_rows"],
            }
        )


def log_wandb(board, result):
    import wandb

    q1 = result["q1_serial_dispersion"]
    q2 = result["q2_raw_versus_candidate"]
    q3 = result["q3_reprice_finding_259"]
    q4 = result["q4_onepass6_replicates"]
    rule = result["proposed_campaign_rule"]
    run = wandb.init(
        project="qwen38-mlx-challenge-senpai",
        entity="wandb-applied-ai-team",
        name="e164-ranked-ab-error-bar",
        job_type="analysis",
        config={
            "experiment": "e164-ranked-ab-error-bar",
            "harness": "ranked",
            "gpu_used": False,
            "board_receipts_scored": result["provenance"]["rows_scored_with_per_prompt"],
            "board_window": [result["provenance"]["created_first"], result["provenance"]["created_last"]],
            **{f"invariant_{k}": v for k, v in result["harness_invariants"].items() if k != "prompt_sha256_8"},
        },
    )

    ic = q1["icc"]
    ra = q4["replicate_analysis"]
    floors = rule["denominator_ladder"][1]["floor_se_single_receipt_pair_mean_of_8_pct"]
    wandb.summary.update(
        {
            "q1/serial_pooled_mean_s_per_token": q1["pooled"]["mean_s_per_token"],
            "q1/serial_pooled_rel_sd_pct": q1["pooled"]["rel_sd_pct"],
            "q1/sd_within_receipt_pct": ic["sd_within_receipt_pct"],
            "q1/sd_run_level_pct": ic["sd_run_level_pct"],
            "q1/icc": ic["icc"],
            "q1/icc_ci95_lo": ic["icc_ci95"][0],
            "q1/icc_ci95_hi": ic["icc_ci95"][1],
            "q1/se_inflation_factor": ic["se_inflation_factor"],
            "q1/design_effect": ic["design_effect"],
            "q1/mean_off_diagonal_corr": q1["cross_prompt_correlation"]["mean_off_diagonal"],
            "q1/within_day_rel_sd_pct": q1["drift"]["mean_within_day_rel_sd_pct"],
            "q1/between_day_rel_sd_pct": q1["drift"]["between_day_rel_sd_of_means_pct"],
            "q2/se_pair_serial_pct": q2["deliverable_se_single_receipt_pair_pct"]["serial_leg_mean_of_8_exact"],
            "q2/se_pair_candidate_upper_pct": q2["deliverable_se_single_receipt_pair_pct"][
                "candidate_leg_mean_of_8_upper_bound"
            ],
            "q2/se_pair_raw_upper_pct": q2["deliverable_se_single_receipt_pair_pct"]["raw_mean_of_8_upper_bound"],
            "q2/corr_serial_mtp_run_mean": q2["cross_leg_coupling"]["corr_run_mean_serial_vs_mtp"],
            "q2/raw_tighter_than_mtp": q2["variance_identity"]["raw_tighter_than_mtp"],
            "q2/null_abs_z_p95": q2["serial_true_null"]["empirical_pair_draws"]["abs_z_naive_p95"],
            "q2/null_fpr_at_z_1_96": q2["serial_true_null"]["empirical_pair_draws"][
                "false_positive_rate_at_z_1_96"
            ],
            "q3/candidate_mean_pct": q3["legs"]["candidate"]["mean_pct"],
            "q3/candidate_z_naive": q3["legs"]["candidate"]["z_naive"],
            "q3/candidate_sign_test_p": q3["legs"]["candidate"]["sign_test_p"],
            "q3/published_delta_pct": q3["published"]["published_delta_pct"],
            "q3/z_denominator_A_naive": q3["denominators"]["A_naive_between_prompt"]["z"],
            "q3/z_denominator_B_icc": q3["denominators"]["B_naive_times_icc_inflation"]["z"],
            "q3/z_denominator_C_serial_null": q3["denominators"]["C_serial_true_null"]["z"],
            "q3/z_denominator_D_replicates": q3["denominators"]["D_same_arm_replicates"]["z"],
            "q3/se_denominator_D_pct": q3["denominators"]["D_same_arm_replicates"]["se_pct"],
            "q4/n_replicates": ra["n_replicates"],
            "q4/spread_pct": ra["spread_pct"],
            "q4/pooled_estimate_pct": ra["pooled_estimate_pct"],
            "q4/sd_of_one_estimate_pct": ra["sd_of_one_estimate_pct"],
            "q4/sd_ci95_lo": ra["sd_ci95"][0],
            "q4/sd_ci95_hi": ra["sd_ci95"][1],
            "q4/t_statistic": ra["t_statistic"],
            "q4/p_two_sided": ra["p_two_sided"],
            "q4/understatement_vs_finding_259": ra["understatement_vs_finding_259"],
            "q4/understatement_vs_finding_279": ra["understatement_vs_finding_279"],
            "rule/floor_se_serial_pct": floors["serial"],
            "rule/floor_se_candidate_pct": floors["candidate"],
            "rule/floor_se_raw_pct": floors["raw"],
            "rule/naive_z_true_5pct_critical_value": rule[
                "critical_values_for_the_naive_between_prompt_z"
            ]["true_5_pct_critical_value"],
            "rule/mde_candidate_one_receipt_pair_pct": rule["minimum_detectable_effect_pct"][
                "candidate_one_receipt_pair"
            ],
        }
    )

    t = wandb.Table(columns=["prompt", "mean_s_per_token", "rel_sd_pct", "p2_5", "p97_5"])
    for p in q1["per_prompt"]:
        t.add_data(p["prompt"], p["mean_s_per_token"], p["rel_sd_pct"], p["p2_5"], p["p97_5"])
    wandb.log({"q1/serial_per_prompt": t})

    cm = wandb.Table(columns=["prompt"] + board.prompts)
    for name, row in zip(board.prompts, q1["cross_prompt_correlation"]["matrix"]):
        cm.add_data(name, *row)
    wandb.log({"q1/cross_prompt_correlation": cm})

    ch = wandb.Table(columns=["channel", "sd_run_mean_of_8_pct", "sd_within_receipt_pct", "se_pair_pct"])
    for name in CHANNELS:
        d = q2["fingerprint_replicate_dispersion"][name]
        ch.add_data(
            name, d["sd_of_run_mean_of_8_pct"], d["sd_within_receipt_pct"], d["se_single_receipt_pair_pct"]
        )
    wandb.log({"q2/channel_dispersion": ch})

    pp = wandb.Table(columns=["prompt", "cand_delta_pct", "serial_delta_pct", "raw_delta_pct"])
    for p in q3["per_prompt"]:
        pp.add_data(p["prompt"], p["cand_delta_pct"], p["serial_delta_pct"], p["raw_delta_pct"])
    wandb.log({"q3/finding_259_per_prompt": pp})

    arms = wandb.Table(
        columns=[
            "arm",
            "solver",
            "score",
            "cand_mean8_pct",
            "serial_mean8_pct",
            "raw_mean8_pct",
            "published_delta_pct",
            "tree_identical_to_base",
        ]
    )
    for arm in q4["arm_detail"]:
        arms.add_data(
            arm["id8"],
            arm["solver"],
            arm["score"],
            arm["candidate"]["mean_pct"],
            arm["serial"]["mean_pct"],
            arm["raw"]["mean_pct"],
            arm["published_delta_pct"],
            bool(
                arm["edl_digit_identical_to_base"]
                and arm["nondraft_identical_to_base"]
                and arm["head_identical_to_base"]
            ),
        )
    wandb.log({"q4/onepass6_replicate_arms": arms})

    lad = wandb.Table(columns=["rung", "when", "use", "why"])
    for r in rule["denominator_ladder"]:
        lad.add_data(r["rung"], r["when"], r["use"], r["why"])
    wandb.log({"rule/denominator_ladder": lad})

    dev = log_dev(board.serial)
    hist = wandb.Table(columns=["run_mean_dev_pct"])
    for v in dev.mean(axis=1):
        hist.add_data(float(v))
    wandb.log({"q1/serial_run_mean_deviation": hist})

    out = os.path.join(os.path.dirname(__file__), "e164-artifacts")
    log_followups(wandb, run, out)

    art = wandb.Artifact("e164-ranked-ab-error-bar", type="analysis")
    for fname in sorted(os.listdir(out)):
        art.add_file(os.path.join(out, fname))
    run.log_artifact(art)

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
