#!/usr/bin/env python3
"""E125 Stage 2 - the isolated-to-in-situ correction, as a class-by-regime
table rather than a scalar.

Stage 0 registered a single correction factor F = 1.43, band [1.00, 2.04]. Two
in-situ anchors now bracket unity from opposite sides:

  E121 rung 2 -> rung 3   cross-simdgroup chunk-sum sharing   factor 2.04
  E120 rung 5e            hoisted activation sum table        factor 0.763

No scalar fits both, and no constant offset fits both either, so the Stage 0
model cannot be repaired by re-fitting F. This file replaces it with a table
whose rows are mechanism classes and whose columns are memory regimes.

The table is fitted only on the Stage 1 frame axis. The two anchors are never
fitting data. They are validation targets, and the reported test is whether one
regime column reproduces both of them once the class axis is allowed to differ.

Three quantities are kept apart on purpose, because Stage 0 conflated them:

  W  the width aggregation term. It is a ranked-only term. Locally it is 1.000
     exactly: E120 rung 5e weighted the isolated grid by the realised local
     width histogram, which IS E[f(M)], so there is no Jensen gap left to
     correct. Applying a local W would double-count.
  F  the frame term. The same mechanism priced in a different memory regime.
     This file measures F and finds it is not a scalar.
  The leg share. Never corrected. E116 measured alpha x beta = 1.000, band
     [0.963, 1.038], and that null is this file's control: the table multiplies
     per-cell effects only and never touches a share.

Every number here is local, ungated evidence from one M4 Pro. None of it is a
ranked score.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ART = Path("research/e125-artifacts")
PEAK_BANDWIDTH_GB_S = 273.0
BOOTSTRAP_DRAWS = 4000

# Memory regimes, defined by what the frame does to the memory system rather
# than by an achieved rate. Stage 1 falsified the achieved-rate labelling: at
# mlp_gate_up M=4 the k1024 frame and the consumer frame reach a similar
# achieved GB/s and move the price in opposite directions, so one function of
# achieved bandwidth cannot label both. What the frame DOES is unambiguous.
REGIMES = {
    "reference": {
        "frames": ("base",),
        "what_the_frame_does": "long resident weight stream, no competitor",
    },
    "resident": {
        "frames": ("cycle",),
        "what_the_frame_does":
            "cycles several weight buffers so the stream cannot stay cached",
    },
    "latency_bound": {
        "frames": ("k1024", "k2048", "k4096"),
        "what_the_frame_does":
            "shortens the weight stream, so the fixed prologue and issue "
            "latency carry more of the kernel and sustained bandwidth carries "
            "less",
    },
    "bandwidth_contended": {
        "frames": ("consumer",),
        "what_the_frame_does":
            "runs a competing bandwidth consumer, so the kernel's share of "
            "the bus falls while its own demand is unchanged",
    },
}
REFERENCE_REGIME = "reference"

# Mechanism classes, in the order the report prints them. `sync` is not a
# ladder name: it is the union of the three threadgroup ladders, because the
# E121 mechanism is an exchange plus the barriers that make it legal, not any
# one of them alone.
LADDER_CLASSES = ("ld", "alu", "deletion", "tg_scaffold", "tgld", "bar")
SYNC_MEMBERS = ("tg_scaffold", "tgld", "bar")

# The two published in-situ anchors. Both predate this file and neither is
# fitting data. `factor` is isolated prediction divided by in-situ measurement,
# so above one means the in-situ context hid the mechanism and below one means
# it charged more than the isolated cell predicted.
ANCHORS = {
    "e120_rung5e_sumtable": {
        "student": "thorfinn",
        "wandb_run": "zkcfcaxr",
        "mechanism": "hoisted activation sum table",
        "effect_on_kernel": "deletes per-round arithmetic, adds one table read",
        "mechanism_class": "deletion",
        # (gain half, cost half). The rung is not class pure: the deleted
        # arithmetic is the gain and the table read it is replaced by is a
        # load, so a deletion-only row cannot price it.
        "component_classes": ("deletion", "ld"),
        "isolated_prediction_pct": 3.240,
        "in_situ_measurement_pct": 4.249,
        "frame": "local",
        "frame_note":
            "already local: the 3.240 % prediction was weighted by the "
            "realised local width histogram, so the width term in it is "
            "E[f(M)] and W = 1.000 exactly. No frame conversion is applied.",
        "factor": 0.763,
        # The in-situ leg medians are 0.030754 and 0.029447 s/token at CV
        # 0.080 %, so the measured delta is tight and the measurement side of
        # this anchor contributes about +-0.003 to the factor.
        "measurement_interval": (0.760, 0.766),
        "prediction_side_quantified": False,
        "prediction_side_note":
            "the 3.240 % prediction multiplies the isolated grid by the "
            "realised width histogram and then by the E116 leg share 0.6070. "
            "F95 applies: that share was not published with the width it was "
            "measured at, and reading it at width 7.359 instead moves the "
            "factor to 1.000. This file validates against the point estimate "
            "and reports the alternative reading beside it.",
        "alternative_reading_factor": 1.000,
    },
    "e121_rung2_to_rung3_share": {
        "student": "alphonse",
        "wandb_run": "qmr3mgl8",
        "mechanism": "cross-simdgroup chunk-sum sharing",
        "effect_on_kernel":
            "adds a threadgroup exchange and two barriers per k-block",
        "mechanism_class": "sync",
        # Sharing the chunk sums deletes the redundant per-simdgroup sums;
        # the exchange and the barriers are what pays for that deletion.
        "component_classes": ("deletion", "sync"),
        "isolated_prediction_pct": 0.890,
        "in_situ_measurement_pct": 0.436,
        "in_situ_sd_pct": 0.093,
        "frame": "ranked",
        "frame_note":
            "the published 2.041 carries the ranked width term: 2.041 / 1.33 "
            "= 1.535, and 1.33 is exactly W_RANKED_CENTRAL. This table is a "
            "local table, so the anchor is converted to the local frame "
            "before it is used. Placing a ranked-frame anchor on a "
            "local-frame table would charge the frame axis for a width term.",
        "published_factor": 2.041,
        "published_interval": (1.686, 2.588),
        "factor": 1.535,
        "measurement_interval": (1.268, 1.946),
        "isolated_cell": "fa_qkv_k5120_n14336|m4",
        "prediction_side_quantified": False,
        "prediction_side_note":
            "1.535 is the published 2.041 with the ranked width term removed. "
            "The NA re-weighting in E121 run 5zms9ntd moves it further to "
            "1.250. All three readings are reported; the local-frame 1.535 is "
            "the one this table is tested against.",
        "alternative_reading_factor": 1.250,
    },
}
# hi/lo of a factor band. A band wider than this spans more than a factor of
# two either side of its own centre, so "the anchor lies inside it" is not
# evidence that the cell reproduces the anchor. Such a cell is reported as
# consistent but non-resolving and does not satisfy the kill rule.
RESOLVING_BAND_RATIO = 4.0

# Stage 0 answered "no correction" for edward's E124. Stage 2 does not change
# that answer, and the reason is now stronger rather than weaker.
E124_STATEMENT = {
    "correction_applied": False,
    "ranked_pct_unchanged": [0.41, 0.45],
    "reason":
        "the ledger prices `noislands` through the Finding 85 class "
        "coefficients 0.24 for draft-path head bytes and 0.95 for draft-path "
        "dispatch deletions. Those are already measured isolated-to-in-situ "
        "transfers. Every row of this table is a per-cell instruction-price "
        "transfer for the same step, so multiplying the two would count the "
        "step twice. Rule 69 forbids mixing the classes as well.",
    "where_the_uncertainty_is":
        "the ledger marks the 0.304 % dispatch residual [INFERRED]. That "
        "component, not the frame, carries E124's risk, and nothing measured "
        "in Stage 1 or Stage 2 speaks to it.",
}

# Local versus ranked width aggregation. The distinction is the Stage 0 defect.
W_LOCAL = 1.000
W_LOCAL_NOTE = (
    "exact, not fitted: E120 rung 5e weighted the isolated 7x7 grid by the "
    "realised local width histogram M={2:4,4:16,5:20,6:20,7:12,8:240}, which "
    "is E[f(M)] itself. No Jensen gap remains, so a local W above 1 would "
    "double-count.")
W_RANKED_CENTRAL = 1.33
W_RANKED_BAND = (1.00, 1.76)
W_RANKED_NOTE = (
    "live only in the ranked frame, where the board publishes "
    "effective_mean_draft_len and not the width histogram, so a convex f(M) "
    "must be read at a mean width. f(5)=6.52 against f(6)=3.30 is a factor "
    "2.0 across adjacent widths, so the gap is not negligible.")

# E116's null. The correction never multiplies this.
E116_ALPHA_BETA = 1.000
E116_ALPHA_BETA_BAND = (0.963, 1.038)

# Board resolution, from the byte-identical resample of the crown content.
BOARD_RESAMPLE_SPREAD_PCT = 0.374
LOCAL_LEG_RESOLUTION_PCT = 0.08
DECISION_LINES = {"parity": 0.53, "mode_proof": 1.86}

# Stage 0 registered terms that later evidence has since settled AGAINST this
# experiment. They are recorded here so no downstream reader picks them up.
FALSIFIED_STAGE0 = {
    "F_floor_1p43": {
        "registered": 1.43,
        "measured": 0.763,
        "status": "falsified",
        "evidence":
            "E120 rung 5e measured 0.763, which is 24 % below the floor Stage "
            "0 registered. The floor was a prediction, not a bound.",
    },
    "bandwidth_regime_law": {
        "registered":
            "the transfer factor rises with achieved bandwidth, r = -0.938 on "
            "the E123 isolated scan",
        "status": "does not replicate",
        "evidence":
            "thorfinn re-ran it in situ on this experiment's own window inside "
            "rung 5g and measured r = -0.067, permutation p = 0.855. Only M=3 "
            "kept a negative slope. Independently, E125 session 1 falsified "
            "the same law as stated: at mlp_gate_up M=4 the k1024 frame at "
            "achieved-rate ratio 0.675 and the consumer frame at 0.623 give "
            "opposite-sign effects, so the achieved rate does not order the "
            "factor.",
        "replacement":
            "the regime axis moves off bandwidth and onto the register file: "
            "specifically the distance from the shipped entry point to its "
            "next floor(budget / registers) occupancy cliff, and the direction "
            "an edit moves it. See `architecture_axis`.",
    },
    "branch_R_per_shape_gains": {
        "status": "void, unchanged",
        "evidence":
            "fractional per-shape gains cluster by inputs-per-group and not by "
            "draft width M, so a per-shape M law has no support.",
    },
}

# --- the third axis: the architecture the leg runs on -------------------------
#
# F4 named three anchors and warned that collapsing them is Advisor Error 94.
# They are three DIFFERENT transfers. Two of them hold the host fixed and move
# the frame; the third holds the frame fixed and moves the architecture. The
# class-by-regime table in this file is built entirely from one host, so it
# cannot reach the third anchor at all, let alone reproduce its sign. That is a
# structural limit of the instrument and is reported as one.
ARCH_ANCHOR = {
    "e121_local_g16s_leg_to_ranked_g17s_leg": {
        "submission": "cf9a9eda-fdb6-4d94-b090-5451db4ff9ea",
        "published_score": 3.26815344,
        "outcome": "rejected",
        "mechanism": "cross-simdgroup chunk-sum sharing",
        "mechanism_class": "occupancy_cliff",
        "frame_pair": "local g16s leg -> ranked g17s leg",
        "axis": "architecture",
        # In situ inside thorfinn's rung 5g, on the local architecture.
        "from_pct": 0.433,
        "from_interval_pct": (0.340, 0.526),
        "from_interval_note":
            "+-0.093, the sd F4 published for the paired local leg reading",
        # The ranked M5 run, as the board resolved it.
        "to_pct": -2.10,
        "to_interval_pct": (-2.474, -1.726),
        "to_interval_note":
            "+-BOARD_RESAMPLE_SPREAD_PCT 0.374, the byte-identical resample "
            "spread from F96. A ranked delta smaller than this cannot be "
            "resolved by one run.",
        "factor": -4.849,
        "factor_interval": (-7.276, -3.282),
        "arithmetic_check":
            "3.34136 * (1 - 0.0210) = 3.2712 against the published 3.26815",
    },
}

# `leg %` per `resident simdgroup %`, the one free coefficient in the census
# predictor. It is NOT fitted here: it is reported per architecture so the
# reader can see whether one value serves both.
ARCH_FRAMES = {
    "entry_point": {
        "key": "whole_e121_change",
        "field": "entry_fraction",
        "physical":
            "correct. `qmv_fast_crossrow_affine4_g64_wide` is a METAL_FUNC and "
            "the shipped `switch (ntg.x)` inlines every live width into one "
            "entry point, so one register allocation and therefore one "
            "occupancy serves every dispatched width.",
    },
    "isolated_bodies_f47_weighted": {
        "key": "whole_e121_change",
        "field": "body_f47_weighted_fraction",
        "physical":
            "a proxy. Per-width bodies never get their own register "
            "allocation in the shipped kernel, so this frame describes a "
            "machine that does not exist. It is reported because it is the "
            "frame the earlier residency tables used.",
    },
}


def load(name: str) -> dict:
    p = ART / name
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# A. why the isolated grid cannot supply its own transfer law
# --------------------------------------------------------------------------

def _slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def confound_audit(scan: dict) -> dict:
    """The E123 grid correlates price with achieved GB/s only across widths.

    Inside a fixed width the sign is not stable, so the pooled slope is a
    between-width effect wearing a bandwidth label. That is the reason the
    isolated grid cannot be asked for its own frame correction, and the reason
    Stage 1 had to move the regime at a fixed width.
    """
    cells = scan["cells"]
    arms = sorted({a for c in cells.values() for a in c["pct"]})
    out = {}
    for arm in arms:
        pooled_x, pooled_y = [], []
        by_width: dict[int, tuple[list, list]] = {}
        for key, cell in cells.items():
            if arm not in cell["pct"]:
                continue
            m = int(key.split("|m")[1])
            x, y = cell["gbs"], cell["pct"][arm]
            pooled_x.append(x)
            pooled_y.append(y)
            bx, by = by_width.setdefault(m, ([], []))
            bx.append(x)
            by.append(y)
        pooled = _slope(pooled_x, pooled_y)
        within = {m: _slope(bx, by) for m, (bx, by) in sorted(by_width.items())}
        signs = {m: (0 if s is None else (1 if s > 0 else -1))
                 for m, s in within.items()}
        nonzero = [s for s in signs.values() if s]
        out[arm] = {
            "pooled_slope_pct_per_gbs": pooled,
            "within_width_slope_pct_per_gbs": within,
            "within_width_sign": signs,
            "sign_stable_within_width": len(set(nonzero)) <= 1 if nonzero
                                        else None,
            "pooled_sign_matches_all_within":
                pooled is not None and bool(nonzero)
                and all(s == (1 if pooled > 0 else -1) for s in nonzero),
        }
    n_arms = len(out)
    n_stable = sum(1 for v in out.values() if v["sign_stable_within_width"])
    return {
        "source": scan["source"],
        "by_arm": out,
        "n_arms": n_arms,
        "n_arms_sign_stable_within_width": n_stable,
        "identified": n_stable == n_arms,
        "note":
            "The pooled achieved-GB/s slope in the isolated grid is confounded "
            "with the width axis. Where the within-width sign is not stable, "
            "the pooled slope is a between-width effect and cannot be read as "
            "a bandwidth law.",
    }


# --------------------------------------------------------------------------
# B. the class-by-regime table
# --------------------------------------------------------------------------

def frame_to_regime() -> dict:
    return {f: r for r, spec in REGIMES.items() for f in spec["frames"]}


def gather_points(laws: list) -> dict:
    """(class, session, shape, m, frame) -> price, from every input session.

    Sessions stay distinct in the key. Two sessions measured overlapping cells
    on different thermal histories, so their base prices are not
    interchangeable and a frame price is only ever divided by the base price
    of its own session.
    """
    pts: dict[tuple, float] = {}
    for session, law in laws:
        for klass, block in law["frame_law"].items():
            for p in block["points"]:
                key = (klass, session, p["shape"], p["m"], p["frame"])
                pts[key] = p["value_us_per_k_block"]
    return pts


def slope_through_origin(rows: list) -> float | None:
    den = sum(b * b for _, b in rows)
    return sum(a * b for a, b in rows) / den if den > 0 else None


def _band(v: list) -> list:
    if not v:
        return [None, None]
    v = sorted(v)
    return [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]]


def pairs_for(pts: dict, klasses: tuple, frames: tuple) -> list:
    """(frame price, own-session base price) over every matching cell."""
    rows = []
    for (klass, session, shape, m, frame), value in pts.items():
        if klass not in klasses or frame not in frames:
            continue
        base = pts.get((klass, session, shape, m, "base"))
        if base is None:
            continue
        rows.append((value, base))
    return rows


def cell_factor(pts: dict, klasses: tuple, frames: tuple) -> dict:
    """Transfer factor for one table cell: base price over frame price.

    The estimator is the origin-forced least squares slope of frame price on
    base price, which weights each measured cell by base squared. That is the
    weight the question asks for, because a cell whose isolated price is near
    zero carries no information about how a price transfers, and it is the
    maximum likelihood proportionality constant under equal per-cell noise.
    The linear ratio of sums is reported beside it as a sensitivity: the two
    disagree by up to a factor of two in the contended regime, and that
    disagreement is itself a result.
    """
    rows = pairs_for(pts, klasses, frames)
    if len(rows) < 2:
        return {"measured": False, "n_cells": len(rows),
                "note": "not measured: fewer than 2 cells in this table cell"}
    slope = slope_through_origin(rows)
    num = sum(a for a, _ in rows)
    den = sum(b for _, b in rows)
    boot = []
    rng = random.Random(20260822)
    for _ in range(BOOTSTRAP_DRAWS):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        s = slope_through_origin(draw)
        if s is not None:
            boot.append(s)
    slope_lo, slope_hi = _band(boot)
    # The factor is 1/slope, so a slope band that reaches zero maps to a factor
    # band that is unbounded above and changes sign. Such a cell must not be
    # allowed to "contain" an anchor: an unbounded interval would pass the kill
    # rule without evidence. Report it, flag it, and let the anchor test skip
    # it.
    identified = slope_lo is not None and slope_lo > 0
    return {
        "measured": True,
        "identified": identified,
        "n_cells": len(rows),
        "factor": 1.0 / slope if slope else None,
        "slope": slope,
        "slope_ci95": [slope_lo, slope_hi],
        "ci95": ([1.0 / slope_hi, 1.0 / slope_lo] if identified
                 else [None, None]),
        "unidentified_reason": None if identified else
        "the bootstrap slope band reaches zero, so the reciprocal is "
        "unbounded. The point estimate stands; the interval does not.",
        "factor_ratio_of_sums": (den / num) if num else None,
        "base_positive_frac": sum(1 for _, b in rows if b > 0) / len(rows),
    }


def class_regime_table(pts: dict) -> dict:
    """Rows are mechanism classes, columns are memory regimes."""
    present = {k for (k, *_rest) in pts}
    rows: dict[str, dict] = {}
    for klass in LADDER_CLASSES + ("sync",):
        members = SYNC_MEMBERS if klass == "sync" else (klass,)
        if not present.intersection(members):
            rows[klass] = {
                "measured": False,
                "note": "not measured: no session carried this class"}
            continue
        cells = {}
        for regime, spec in REGIMES.items():
            if regime == REFERENCE_REGIME:
                cells[regime] = {
                    "measured": True, "identified": True, "definitional": True,
                    "factor": 1.0,
                    "ci95": [1.0, 1.0], "n_cells": None,
                    "note": "unit by construction: the reference regime is "
                            "what the isolated price was measured in. It is "
                            "never evidence for or against an anchor"}
                continue
            cells[regime] = cell_factor(pts, members, spec["frames"])
        rows[klass] = {"measured": True, "members": list(members),
                       "by_regime": cells}
    return {
        "regime_definitions": {r: s["what_the_frame_does"]
                               for r, s in REGIMES.items()},
        "regime_frames": {r: list(s["frames"]) for r, s in REGIMES.items()},
        "rows": rows,
        "reading":
            "a factor above 1 means the regime hides the mechanism, so an "
            "isolated price over-predicts the in-situ effect. A factor below "
            "1 means the regime charges more for the mechanism than the "
            "isolated cell predicted.",
    }


# --------------------------------------------------------------------------
# C. the anchor test - validation only, never fitting
# --------------------------------------------------------------------------

def anchor_test(table: dict) -> dict:
    """Which regime column reproduces each anchor, and is one column shared?

    This is an identification test, not a fit. The table comes entirely from
    the frame axis. For each anchor the question is which regime columns of
    the anchor's own class contain the published factor. If one regime is
    common to both anchors, a single in-situ regime plus the class axis
    explains both, which is the pre-registered hypothesis. If no regime is
    common, the class axis alone cannot carry the split and the two anchors
    were measured in different regimes.
    """
    out = {}
    for name, anchor in ANCHORS.items():
        row = table["rows"].get(anchor["mechanism_class"], {})
        if not row.get("measured"):
            out[name] = {
                "class": anchor["mechanism_class"], "resolved": False,
                "note": "not measured: the anchor's class is absent from "
                        "every session"}
            continue
        hits, detail = [], {}
        for regime, cell in row["by_regime"].items():
            if not cell.get("measured"):
                detail[regime] = {"measured": False}
                continue
            lo, hi = cell["ci95"]
            ident = cell.get("identified", False)
            definitional = cell.get("definitional", False)
            width = (hi / lo) if ident and lo and lo > 0 else None
            resolving = (width is not None and width <= RESOLVING_BAND_RATIO
                         and not definitional)
            contains = ident and not definitional and lo <= anchor["factor"] <= hi
            covered = contains and resolving
            alt = anchor.get("alternative_reading_factor")
            detail[regime] = {
                "measured": True,
                "identified": ident,
                "definitional": definitional,
                "table_factor": cell["factor"],
                "table_ci95": cell["ci95"],
                "band_ratio": width,
                "resolving": resolving,
                "contains_anchor": contains,
                "covers_anchor": covered,
                "covers_alternative_reading":
                    ident and resolving and alt is not None and lo <= alt <= hi,
                "note": (None if ident else cell.get("unidentified_reason"))
                if resolving or not ident else
                "consistent but non-resolving: the band spans a factor of "
                "%.1f, so containing the anchor is not evidence" % width,
            }
            if covered:
                hits.append(regime)
        out[name] = {
            "class": anchor["mechanism_class"],
            "anchor_frame_as_published": anchor["frame"],
            "anchor_frame_note": anchor["frame_note"],
            "published_factor": anchor.get("published_factor",
                                           anchor["factor"]),
            "anchor_factor": anchor["factor"],
            "anchor_interval": list(anchor["measurement_interval"]),
            "alternative_reading_factor":
                anchor.get("alternative_reading_factor"),
            "prediction_side_quantified": anchor["prediction_side_quantified"],
            "prediction_side_note": anchor["prediction_side_note"],
            "regimes_that_reproduce_anchor": hits,
            "resolved": bool(hits),
            "by_regime": detail,
        }
    resolved = [v for v in out.values() if v.get("resolved")]
    shared: set = set()
    if resolved and len(resolved) == len(ANCHORS):
        shared = set.intersection(
            *[set(v["regimes_that_reproduce_anchor"]) for v in resolved])
    return {
        "by_anchor": out,
        "n_anchors": len(ANCHORS),
        "n_anchors_reproduced": len(resolved),
        "shared_regime": sorted(shared),
        "both_anchors_reproduced": len(resolved) == len(ANCHORS),
        "single_regime_explains_both": bool(shared),
        "kill_rule":
            "the table must reproduce BOTH anchors. A table that reproduces "
            "one and misses the other has picked a side.",
        "verdict": _anchor_verdict(len(resolved), len(ANCHORS), bool(shared)),
    }


def composite_diagnostic(table: dict) -> dict:
    """Can a mixed rung explain an anchor that a class-pure row cannot?

    Neither anchor rung is one class. Each deletes work and adds something
    back: E120 rung 5e deletes per-round arithmetic and adds one table read,
    so its halves are `deletion` and `ld`. E121 rung 2 -> 3 deletes the
    redundant per-simdgroup sums and adds an exchange plus two barriers, so
    its halves are `deletion` and `sync`. A two-part rung transfers as a
    harmonic mixture of its two class factors, weighted by each half's share
    of the isolated net effect:

        1 / F(w) = w / F_gain + (1 - w) / F_cost

    That map is a Mobius transform of w, so an unconstrained w reaches almost
    any target and "reachable" carries no information on its own. The physics
    supplies the constraint. Both rungs are a gain half plus a cost half whose
    signs oppose, so the gain half must exceed the net and w must be at least
    1. A w below 1 would require the added read or the added barriers to be a
    speedup, and a w below 0 would require the deletion itself to be a cost.
    The reported quantity is therefore w together with its admissibility, and
    a large admissible w is itself a warning: it means the mixture only works
    by cancelling two much larger opposed terms.
    """
    out: dict[str, dict] = {}
    for name, anchor in ANCHORS.items():
        gain_k, cost_k = anchor["component_classes"]
        g_row = table["rows"].get(gain_k, {})
        c_row = table["rows"].get(cost_k, {})
        if not (g_row.get("measured") and c_row.get("measured")):
            out[name] = {"available": False,
                         "note": "not measured: this session lacks the %s or "
                                 "%s row" % (gain_k, cost_k)}
            continue
        target = anchor["factor"]
        per_regime = {}
        for regime in REGIMES:
            if regime == REFERENCE_REGIME:
                continue
            g = g_row["by_regime"].get(regime, {})
            c = c_row["by_regime"].get(regime, {})
            if not (g.get("measured") and c.get("measured")):
                per_regime[regime] = {"measured": False}
                continue
            fg, fc = g.get("factor"), c.get("factor")
            if not fg or not fc or fg == fc:
                per_regime[regime] = {
                    "measured": True, "admissible": False,
                    "note": "no mixture: a class factor is missing or the two "
                            "classes coincide"}
                continue
            w = (1.0 / target - 1.0 / fc) / (1.0 / fg - 1.0 / fc)
            per_regime[regime] = {
                "measured": True,
                "gain_class": gain_k, "gain_factor": fg,
                "cost_class": cost_k, "cost_factor": fc,
                "gain_share_of_isolated_net": w,
                "admissible": w >= 1.0,
                "admissibility_rule":
                    "w >= 1 is required: the gain half must exceed the net "
                    "because the cost half opposes it",
                "cancellation_warning": w >= 3.0,
                "both_identified": bool(g.get("identified")
                                        and c.get("identified")),
            }
        out[name] = {
            "available": True,
            "class_as_labelled": anchor["mechanism_class"],
            "component_classes": [gain_k, cost_k],
            "anchor_factor": target,
            "by_regime": per_regime,
            "admissible_in": [r for r, v in per_regime.items()
                              if v.get("admissible")],
        }
    shared = [set(v["admissible_in"]) for v in out.values() if v.get("available")]
    return {
        "available": True,
        "by_anchor": out,
        "model": "1 / F(w) = w / F_gain + (1 - w) / F_cost",
        "w_meaning": "the gain half's share of the isolated net effect",
        "shared_admissible_regime":
            sorted(set.intersection(*shared)) if shared else [],
    }


def _anchor_verdict(n_ok: int, n: int, shared: bool) -> str:
    if n_ok < n:
        return "FAILS the kill rule: %d of %d anchors reproduced" % (n_ok, n)
    if shared:
        return ("passes: both anchors reproduced, and one regime column "
                "reproduces both once the class axis is allowed to differ")
    return ("passes the kill rule but the two anchors need DIFFERENT regime "
            "columns, so the class axis alone does not carry the split")


# --------------------------------------------------------------------------
# D. does one function of the achieved rate do as well?
# --------------------------------------------------------------------------

def marginal_summary(laws: list, pts: dict) -> dict:
    """Class-only, regime-only and class-by-regime, compared on one data set.

    The advisor asked for the marginal single-variable summary if it fits as
    well as the table. `mu` is the memory-bound fraction: the achieved rate
    over the bus share actually available, which is phi_eff where a competitor
    runs and phi_arm otherwise. Reporting the variance each model explains is
    what decides whether the table earns its extra cells.
    """
    mu_of = {}
    for session, law in laws:
        for block in law["frame_law"].values():
            for p in block["points"]:
                key = (session, p["shape"], p["m"], p["frame"])
                mu_of[key] = (p["phi_eff"] if p["frame"] == "consumer"
                              else p["phi_arm"])
    f2r = frame_to_regime()
    obs = []
    for (klass, session, shape, m, frame), value in pts.items():
        if frame == "base":
            continue
        base = pts.get((klass, session, shape, m, "base"))
        mu = mu_of.get((session, shape, m, frame))
        mu_base = mu_of.get((session, shape, m, "base"))
        if base is None or mu is None or mu_base is None:
            continue
        obs.append({"klass": klass, "regime": f2r.get(frame), "frame": frame,
                    "shape": shape, "m": m, "session": session,
                    "value": value, "base": base,
                    "mu": mu, "mu_base": mu_base, "d_mu": mu - mu_base})

    # Classes differ in absolute price by more than an order of magnitude --
    # the deletion of a whole add tree is worth microseconds per k-block and an
    # injected load is worth hundredths. Pooling raw residuals would let the
    # deletion class decide every model comparison on its own. Scaling each
    # class to unit root-mean-square base price leaves every within-class slope
    # unchanged and makes the classes contribute comparably.
    scale: dict[str, float] = {}
    for o in obs:
        scale.setdefault(o["klass"], 0.0)
    for klass in scale:
        sq = [o["base"] ** 2 for o in obs if o["klass"] == klass]
        rms = (sum(sq) / len(sq)) ** 0.5 if sq else 0.0
        scale[klass] = 1.0 / rms if rms > 0 else 0.0
    for o in obs:
        o["w"] = scale[o["klass"]]

    def rss(groups: dict) -> float:
        total = 0.0
        for rows in groups.values():
            pairs = [(o["value"] * o["w"], o["base"] * o["w"]) for o in rows]
            s = slope_through_origin(pairs)
            if s is None:
                continue
            total += sum((a - s * b) ** 2 for a, b in pairs)
        return total

    def group(keyfn) -> dict:
        g: dict = {}
        for o in obs:
            g.setdefault(keyfn(o), []).append(o)
        return g

    flat = rss({"all": obs})
    by_class = rss(group(lambda o: o["klass"]))
    by_regime = rss(group(lambda o: o["regime"]))
    by_both = rss(group(lambda o: (o["klass"], o["regime"])))
    return {
        "n_observations": len(obs),
        "residual_sum_of_squares": {
            "one_factor_for_everything":
                {"rss": flat, "parameters": 1},
            "class_only":
                {"rss": by_class,
                 "parameters": len(group(lambda o: o["klass"]))},
            "regime_only":
                {"rss": by_regime,
                 "parameters": len(group(lambda o: o["regime"]))},
            "class_by_regime":
                {"rss": by_both,
                 "parameters": len(group(lambda o: (o["klass"],
                                                    o["regime"])))},
        },
        "variance_explained_vs_flat": {
            "class_only": 1.0 - by_class / flat if flat else None,
            "regime_only": 1.0 - by_regime / flat if flat else None,
            "class_by_regime": 1.0 - by_both / flat if flat else None,
        },
        "regime_axis_dominates_class_axis":
            (by_regime < by_class) if flat else None,
        "mu_note":
            "mu is reported per observation as a covariate, not as the regime "
            "label. Stage 1 falsified the achieved-rate labelling: at "
            "mlp_gate_up M=4 the k1024 frame reaches phi 0.675 and the "
            "consumer frame phi 0.623, a similar achieved rate, and the two "
            "move the price in opposite directions.",
        "observations": obs,
    }


# --------------------------------------------------------------------------
# E. what the table changes, and what it must not touch
# --------------------------------------------------------------------------

def null_control() -> dict:
    return {
        "e116_alpha_times_beta": E116_ALPHA_BETA,
        "e116_band": list(E116_ALPHA_BETA_BAND),
        "share_term_corrected": False,
        "note":
            "the correction multiplies a per-cell mechanism effect only. The "
            "leg share is a measured decomposition of where the leg spends "
            "its time, and E116 found it composes exactly, so multiplying it "
            "by any frame term would break a measured null.",
        "passes": abs(E116_ALPHA_BETA - 1.0) < 0.04,
    }


def width_terms() -> dict:
    return {
        "local": {"harness": "local", "W": W_LOCAL, "note": W_LOCAL_NOTE},
        "ranked": {"harness": "ranked", "W": W_RANKED_CENTRAL,
                   "band": list(W_RANKED_BAND), "note": W_RANKED_NOTE},
        "stage0_defect":
            "Stage 0 applied the ranked W to a local comparison. That was "
            "wrong and it inflated the registered prediction.",
    }


def route_b(pre: dict, table: dict) -> dict:
    """The Route B ranked prediction, recomputed with the table.

    Route B is a wide-QMV deletion, so it reads the deletion row. This
    experiment does not identify which regime column the in-situ round sits
    in, so every measured column is reported and the spread across them IS the
    honest uncertainty.
    """
    isolated = pre["isolated_ranked_recomputed_on_7x7"]
    row = table["rows"].get("deletion", {})
    out = {"isolated_ranked_pct": isolated, "class": "deletion",
           "width_term_applied": "ranked", "W_ranked": W_RANKED_CENTRAL,
           "by_regime": {}}
    if not row.get("measured"):
        out["note"] = "not measured: the deletion row is absent"
        return out
    span = []
    for regime, cell in row["by_regime"].items():
        if not cell.get("measured") or cell.get("factor") is None:
            out["by_regime"][regime] = {"measured": False}
            continue
        base = isolated / W_RANKED_CENTRAL
        lo_f, hi_f = cell["ci95"]
        point = base / cell["factor"]
        band = ([base / hi_f, base / lo_f]
                if lo_f and hi_f and lo_f > 0 else [None, None])
        out["by_regime"][regime] = {"measured": True,
                                    "table_factor": cell["factor"],
                                    "ranked_pct": point,
                                    "ranked_pct_ci95": band}
        span.append(point)
        if band[0] is not None:
            span.extend(band)
    out["envelope_across_regimes"] = [min(span), max(span)] if span else None
    out["decision_lines"] = DECISION_LINES
    out["stage0_registered_primary_pct"] = pre["predictions"]["primary"][
        "ranked_pct"]
    out["stage0_band_80pct"] = pre["band_80pct"]
    # A decision is safe only when EVERY measured regime column falls on the
    # same side of the line, because this experiment does not identify which
    # column the in-situ round sits in.
    lo, hi = out["envelope_across_regimes"] or (None, None)
    out["decision"] = {
        line: {"line_pct": v,
               "clears_in_every_regime": lo is not None and lo > v,
               "clears_in_no_regime": hi is not None and hi < v,
               "verdict": "clears" if lo is not None and lo > v
               else "fails" if hi is not None and hi < v
               else "regime dependent: the answer changes with the column"}
        for line, v in DECISION_LINES.items()}
    out["resolution_note"] = (
        "F96: a byte-identical resample of the crown content published two "
        "medians %.3f %% apart, so a ranked prediction that moves by less "
        "than that cannot be validated against a single ranked run. The local "
        "leg resolves to about %.2f %%, so this table is aimed at local-leg "
        "decisions." % (BOARD_RESAMPLE_SPREAD_PCT, LOCAL_LEG_RESOLUTION_PCT))
    return out


# --------------------------------------------------------------------------

def architecture_axis(census: dict) -> dict:
    """Price the third anchor from the zero-GPU register census.

    The frame table cannot reach this anchor: every point in it comes from one
    host. The census can, because it runs the real AGX backend for both
    architectures and reports what each one allocates.

    The predictor is one line. `E121` moves the shipped entry point by one
    register on each architecture, in OPPOSITE directions, and on each
    architecture that one register crosses a `floor(budget / registers)` cliff.
    So the predicted transfer factor is the ratio of the two fractional
    occupancy changes, and it is negative by construction.
    """
    anchor = ARCH_ANCHOR["e121_local_g16s_leg_to_ranked_g17s_leg"]
    local, ranked = census["arches"]
    out = {"anchor": dict(anchor), "local_arch": local, "ranked_arch": ranked,
           "by_frame": {}}
    for frame, spec in ARCH_FRAMES.items():
        rung = census["ladder"][spec["key"]]
        a = rung[local][spec["field"]]
        b = rung[ranked][spec["field"]]
        factor = None if not a else b / a
        lo, hi = anchor["factor_interval"]
        out["by_frame"][frame] = {
            "physical": spec["physical"],
            "local_occupancy_change_pct": 100 * a,
            "ranked_occupancy_change_pct": 100 * b,
            "predicted_factor": factor,
            "sign_correct": factor is not None and factor < 0,
            "inside_anchor_interval":
                factor is not None and lo <= factor <= hi,
            "magnitude_shortfall":
                None if not factor else anchor["factor"] / factor,
            # leg percent per resident-simdgroup percent, one per architecture.
            "implied_leg_elasticity": {
                local: None if not a else anchor["from_pct"] / (100 * a),
                ranked: None if not b else anchor["to_pct"] / (100 * b)},
        }
    for frame, cell in out["by_frame"].items():
        e = cell["implied_leg_elasticity"]
        cell["elasticity_ratio_ranked_over_local"] = (
            None if not e[local] else e[ranked] / e[local])
    signs = [c["sign_correct"] for c in out["by_frame"].values()]
    inside = [c["inside_anchor_interval"] for c in out["by_frame"].values()]
    out["verdict"] = {
        "sign_reproduced": all(signs),
        "magnitude_reproduced": any(inside),
        "statement":
            "the census reproduces the SIGN of the third anchor in both "
            "frames and the MAGNITUDE in neither. The sign is the part no "
            "within-host frame axis can produce, and it is the part that "
            "cost the campaign a submission.",
        "residual_candidates": [
            "the leg elasticity of occupancy is not one and is not equal on "
            "the two architectures",
            "the QMV share of the leg differs between thorfinn's rung 5g "
            "local leg and the ranked eight-prompt leg",
            "the barrier and exchange instruction cost is a second term that "
            "the register census cannot see at all",
            "the local reading is one host, one public fixture and a short "
            "token window; the ranked reading is eight hidden prompts at 512 "
            "tokens",
        ],
    }
    ep = census["ladder"]["whole_e121_change"]
    out["mechanism"] = {
        "entry_registers": {
            arch: {"pre_e121": census["arms"]["pre_e121"][arch]["entry"][
                       "registers"],
                   "share_on": census["arms"]["share_on"][arch]["entry"][
                       "registers"]}
            for arch in (local, ranked)},
        "resident_simdgroups": {
            arch: {"pre_e121": ep[arch]["entry_from"],
                   "share_on": ep[arch]["entry_to"]}
            for arch in (local, ranked)},
        "binding_body": {
            arch: {arm: census["inlining_law"]["%s/%s" % (arch, arm)][
                "widest_body_width"] for arm in ("pre_e121", "share_on")}
            for arch in (local, ranked)},
        "statement":
            "on %s the pre-E121 ceiling was the NA=4 body at 94 registers, so "
            "cutting NA=4 to 90 let the NA=5 body at 93 become the ceiling and "
            "bought one simdgroup. On %s the pre-E121 ceiling was already the "
            "NA=5 body at 101 registers, which E121 cannot touch, and raising "
            "NA=4 from 90 to 102 made NA=4 the new ceiling and lost one "
            "simdgroup. The same edit, one register each way, on opposite "
            "sides of a floor-division cliff." % (local, ranked),
        "regime_variable":
            "not `does the register budget bind` -- nothing spills on either "
            "architecture. It is `how many registers separate the entry point "
            "from the next floor(budget / registers) cliff, and in which "
            "direction does the edit move it`. That is computable with no GPU "
            "time.",
    }
    cliffs = census["cliffs"]
    out["actionable_gate"] = {
        "rule":
            "before submitting a kernel edit, census the shipped entry point "
            "on the ranked architecture and check the cliff distance. An edit "
            "that adds registers while the entry point sits within its "
            "`registers_to_lose_one` margin buys a ranked regression no local "
            "measurement on this host can see.",
        "current_shipped_state": cliffs["%s/share_on" % ranked],
        "immediate_consequence":
            "the shipped entry point is at %d registers on %s and needs only "
            "%d fewer to recover a simdgroup, which is %+.2f %% occupancy. "
            "Reverting E121 does exactly that."
            % (cliffs["%s/share_on" % ranked]["registers"], ranked,
               cliffs["%s/share_on" % ranked]["registers_to_gain_one"],
               100 * (cliffs["%s/pre_e121" % ranked]["resident"]
                      / cliffs["%s/share_on" % ranked]["resident"] - 1)),
        "next_ceiling":
            "below that, %s is floored by the NA=5 body at 101 registers. "
            "Reaching 40 resident simdgroups needs the entry point at 99 or "
            "fewer, so it needs the NA=5 body cut by two as well." % ranked,
    }
    return out


def frame_pair_table(table: dict, arch: dict | None) -> dict:
    """The deliverable F4 asked for: correction(class, frame pair, regime).

    Three anchors, three frame pairs, one row each. Collapsing them into one
    number is Advisor Error 94, so they are never averaged here.
    """
    rows = {}
    for name, anchor in ANCHORS.items():
        klass = anchor["mechanism_class"]
        row = table["rows"].get(klass, {})
        by_regime = {}
        if row.get("measured"):
            for regime, cell in row["by_regime"].items():
                if not cell.get("measured"):
                    continue
                by_regime[regime] = {
                    "factor": cell["factor"], "ci95": cell["ci95"],
                    "identified": cell.get("identified", False),
                    "definitional": cell.get("definitional", False)}
        rows[name] = {
            "axis": "frame",
            "frame_pair": ("isolated cell -> in-situ round"
                           if name.startswith("e121")
                           else "kernel frame -> leg frame"),
            "host_changes": False,
            "mechanism_class": klass,
            "component_classes": list(anchor["component_classes"]),
            "anchor_factor": anchor["factor"],
            "anchor_interval": list(anchor["measurement_interval"]),
            "table_by_regime": by_regime,
            "instrument": "e125 frame sessions, one host, ungated ABBA",
        }
    if arch is not None:
        name = "e121_local_g16s_leg_to_ranked_g17s_leg"
        a = arch["anchor"]
        rows[name] = {
            "axis": "architecture",
            "frame_pair": a["frame_pair"],
            "host_changes": True,
            "mechanism_class": a["mechanism_class"],
            "component_classes": ["occupancy_cliff"],
            "anchor_factor": a["factor"],
            "anchor_interval": list(a["factor_interval"]),
            "predicted_by_regime": {
                frame: {"factor": cell["predicted_factor"],
                        "sign_correct": cell["sign_correct"],
                        "inside_anchor_interval": cell[
                            "inside_anchor_interval"]}
                for frame, cell in arch["by_frame"].items()},
            "instrument": "e125 zero-GPU register census, both architectures",
        }
    return {
        "rows": rows,
        "axes": ["mechanism class", "frame pair", "regime"],
        "never_collapsed":
            "the three rows are three different transfers. Two hold the host "
            "fixed and move the frame; the third holds the frame fixed and "
            "moves the architecture. Averaging them would hide the only sign "
            "change in the set.",
        "which_anchors_the_frame_table_fails":
            "both of them. The class-by-regime table reproduces neither "
            "frame-axis anchor under its own kill rule, and it cannot address "
            "the architecture anchor at all because every point in it comes "
            "from one host.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", action="append", default=[], metavar="NAME=FILE",
                    help="frame-law artifact, one per session")
    ap.add_argument("--scan", default="e123-bandwidth-scan.json")
    ap.add_argument("--prereg", default="routeb-prediction.json")
    ap.add_argument("--census", default="e121-census.json")
    ap.add_argument("--out", default="correction.json")
    args = ap.parse_args()

    laws = []
    for spec in (args.law or ["work=frame-law.json"]):
        name, _, fname = spec.partition("=")
        laws.append((name, load(fname)))

    pts = gather_points(laws)
    table = class_regime_table(pts)
    arch = architecture_axis(load(args.census))
    result = {
        "harness": "local",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "sessions": [n for n, _ in laws],
        "confound_audit": confound_audit(load(args.scan)),
        "width_terms": width_terms(),
        "null_control": null_control(),
        "class_regime_table": table,
        "anchor_test": anchor_test(table),
        "composite_diagnostic": composite_diagnostic(table),
        "architecture_axis": arch,
        "frame_pair_table": frame_pair_table(table, arch),
        "marginal_summary": marginal_summary(laws, pts),
        "route_b": route_b(load(args.prereg), table),
        "e124_noislands": E124_STATEMENT,
        "falsified_stage0_terms": FALSIFIED_STAGE0,
    }
    (ART / args.out).write_text(json.dumps(result, indent=2) + "\n")

    print("E125 stage 2 - class-by-regime transfer table  (local, ungated)\n")
    cols = [r for r in REGIMES if r != REFERENCE_REGIME]
    print("%-13s%s" % ("class", "".join("%-26s" % c for c in cols)))
    for klass, row in table["rows"].items():
        if not row.get("measured"):
            print("%-13s%s" % (klass, row["note"]))
            continue
        line = ""
        for c in cols:
            cell = row["by_regime"][c]
            if not cell.get("measured"):
                line += "%-26s" % "not measured"
            elif not cell.get("identified"):
                line += "%-26s" % ("%6.3f  unidentified" % cell["factor"])
            else:
                lo, hi = cell["ci95"]
                line += "%-26s" % ("%6.3f [%6.2f,%6.2f]"
                                   % (cell["factor"], lo, hi))
        print("%-13s%s" % (klass, line))

    at = result["anchor_test"]
    print("\nanchors (validation targets, never fitted)")
    for name, v in at["by_anchor"].items():
        if not v.get("resolved") and "note" in v:
            print("  %-28s %s" % (name, v["note"]))
            continue
        print("  %-28s class=%-9s factor=%.3f (%s frame, published %.3f)"
              % (name, v["class"], v["anchor_factor"],
                 v["anchor_frame_as_published"], v["published_factor"]))
        print("  %-28s   reproduced by: %s"
              % ("", ", ".join(v["regimes_that_reproduce_anchor"])
                 or "no regime"))
        for regime, d in v["by_regime"].items():
            if not d.get("measured"):
                continue
            print("  %-28s   %-20s %6.3f %s%s"
                  % ("", regime, d["table_factor"],
                     "reproduces" if d["covers_anchor"] else
                     "contains but non-resolving (band x%.1f)" % d["band_ratio"]
                     if d.get("contains_anchor") else
                     "unit by construction" if d.get("definitional") else
                     "unidentified" if not d["identified"] else "misses",
                     "  [covers the alternative reading %.3f]"
                     % v["alternative_reading_factor"]
                     if d.get("covers_alternative_reading") else ""))
    print("  verdict: %s" % at["verdict"])

    cd = result["composite_diagnostic"]
    if cd.get("available"):
        print("\ncomposite check: neither rung is class pure. %s" % cd["model"])
        print("  w is %s; w >= 1 is required" % cd["w_meaning"])
        for name, v in cd["by_anchor"].items():
            if not v.get("available"):
                print("  %-28s %s" % (name, v["note"]))
                continue
            print("  %-28s gain=%s cost=%s"
                  % (name, *v["component_classes"]))
            for regime, d in v["by_regime"].items():
                if "gain_share_of_isolated_net" not in d:
                    continue
                print("  %-28s   %-20s w=%+7.3f  %s"
                      % ("", regime, d["gain_share_of_isolated_net"],
                         ("admissible, but only by cancelling two much larger "
                          "opposed terms" if d["cancellation_warning"]
                          else "admissible") if d["admissible"]
                         else "inadmissible: the cost half would have to be a "
                              "speedup"))
        print("  shared admissible regime: %s"
              % (", ".join(cd["shared_admissible_regime"]) or "none"))

    print("\nthird axis: architecture. %s -> %s, from the zero-GPU census"
          % (arch["local_arch"], arch["ranked_arch"]))
    m = arch["mechanism"]
    for a in (arch["local_arch"], arch["ranked_arch"]):
        print("  %-14s entry %d -> %d registers, %d -> %d resident "
              "simdgroups, ceiling body NA%d -> NA%d"
              % (a, m["entry_registers"][a]["pre_e121"],
                 m["entry_registers"][a]["share_on"],
                 m["resident_simdgroups"][a]["pre_e121"],
                 m["resident_simdgroups"][a]["share_on"],
                 m["binding_body"][a]["pre_e121"],
                 m["binding_body"][a]["share_on"]))
    an = arch["anchor"]
    print("  anchor  %+.3f %% local -> %+.3f %% ranked, factor %.3f "
          "[%.2f, %.2f]" % (an["from_pct"], an["to_pct"], an["factor"],
                            *an["factor_interval"]))
    for frame, cell in arch["by_frame"].items():
        print("  %-30s predicts %+7.3f  sign %s  magnitude %s (short by "
              "%.1fx)" % (frame, cell["predicted_factor"],
                          "ok" if cell["sign_correct"] else "WRONG",
                          "ok" if cell["inside_anchor_interval"] else "misses",
                          abs(cell["magnitude_shortfall"])))
    print("  gate: %s" % arch["actionable_gate"]["immediate_consequence"])

    fp = result["frame_pair_table"]
    print("\ncorrection(mechanism class, frame pair, regime) - three anchors, "
          "three transfers")
    print("  %-38s %-12s %-30s %s"
          % ("frame pair", "class", "anchor", "instrument axis"))
    for name, row in fp["rows"].items():
        lo, hi = row["anchor_interval"]
        print("  %-38s %-12s %7.3f [%7.2f,%7.2f]      %s"
              % (row["frame_pair"], row["mechanism_class"],
                 row["anchor_factor"], lo, hi, row["axis"]))
    print("  fails: %s" % fp["which_anchors_the_frame_table_fails"])

    print("\nStage 0 terms settled against this experiment")
    for name, v in result["falsified_stage0_terms"].items():
        print("  %-26s %s" % (name, v["status"]))

    rb = result["route_b"]
    print("\nRoute B ranked prediction, deletion row, W_ranked=%.2f"
          % rb["W_ranked"])
    print("  isolated %.3f %%  ->  stage 0 registered %.3f %% [%.3f, %.3f]"
          % (rb["isolated_ranked_pct"], rb["stage0_registered_primary_pct"],
             *rb["stage0_band_80pct"]))
    for regime, v in rb["by_regime"].items():
        if not v.get("measured"):
            continue
        band = v["ranked_pct_ci95"]
        print("  %-20s %.3f %%%s"
              % (regime, v["ranked_pct"],
                 "  [%.3f, %.3f]" % tuple(band) if band[0] is not None
                 else "  band unidentified"))
    print("  envelope across regimes [%.3f, %.3f] %%"
          % tuple(rb["envelope_across_regimes"]))
    for line, v in rb["decision"].items():
        print("  vs %-11s %.2f %%  %s" % (line, v["line_pct"], v["verdict"]))

    print("\nE124 noislands: correction applied = %s, ranked stays %.2f to "
          "%.2f %%" % (result["e124_noislands"]["correction_applied"],
                       *result["e124_noislands"]["ranked_pct_unchanged"]))

    print("\nvariance explained against one factor for everything")
    for k, v in result["marginal_summary"][
            "variance_explained_vs_flat"].items():
        print("  %-16s %s" % (k, "n/a" if v is None else "%+.3f" % v))
    print("\nwrote %s" % (ART / args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
