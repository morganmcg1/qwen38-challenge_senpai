#!/usr/bin/env python3
"""E125 F5: the two-channel law, and residency against bandwidth as regressors.

    python3 research/e125_residency_law.py --out research/e125-artifacts/residency-law.json

Zero GPU. Every input is already in the tree:

  research/e123-artifacts/census.json   registers per arm, per width, per arch
  research/e123-artifacts/rate.json     20 timed cells, 5 shapes x 4 widths
  research/e125-artifacts/e121-census.json   the shipped E121 arm, both arches

F5 asks two questions and this file answers both.

1. Does `observed transfer = instruction channel x residency channel` hold
   across mechanism classes and across the three anchors?

2. Do the bandwidth correlations I reported in Stage 0 survive once residency
   is controlled?

THE STRUCTURAL FACT THAT DRIVES BOTH ANSWERS. A register count is a property
of (arm, width, architecture) fixed at compile time. It does not depend on the
shape, on the frame, on the achieved bandwidth or on anything else a timed cell
varies. So within one width the residency covariate is a CONSTANT across the
five shapes, and it cannot explain one point of a within-width correlation.
Across widths it varies, and there it is collinear with achieved bandwidth
because both are ordered by width. That is the whole result, and the numbers
below only put a size on it.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from e123_arms import SIMDGROUP_BUDGET, simdgroups  # noqa: E402

E123_CENSUS = ROOT / "research/e123-artifacts/census.json"
E123_RATE = ROOT / "research/e123-artifacts/rate.json"
E121_CENSUS = ROOT / "research/e125-artifacts/e121-census.json"

LOCAL = "applegpu_g16s"
RANKED = "applegpu_g17s"
WIDTHS = (2, 3, 4, 5)

# Priced contrasts, exactly as `e125_frame_analysis.LADDER` and `DELETION`
# define them. `low` is the reference rung and `high` is the loaded rung, so a
# positive residency delta means the loaded rung is resident MORE often.
CONTRASTS = (
    ("ld", "k_ld8", "k_ld16"),
    ("alu", "k_alu8", "k_alu16"),
    ("tg_scaffold", "q_scaffold", "k_tg0"),
    ("tgld", "k_tgld8", "k_tgld16"),
    ("deletion", "q_scaffold", "n_nosums"),
    ("deletion_vs_abase", "a_base", "n_nosums"),
)

# The three anchors, in the frames F4 fixed them in.
#
# `local_pct` and `ranked_pct` are leg-frame percentages, positive meaning
# faster. E121's pair is the only one that spans the architecture axis, so it
# is the only one that can identify the residency coefficient.
E121_LOCAL_PCT = 0.433
E121_RANKED_PCT = -2.100
E121_LOCAL_SEM_PCT = 0.093
E121_RANKED_SEM_PCT = 0.374

# F5 quotes +1.463 % as the g16s number when it derives its coefficient
# bracket. That is alphonse's KERNEL-frame gain, not the leg-frame +0.433 %
# this anchor is stated in. Mixing the two across the architecture axis is the
# frame/architecture confusion that produced Advisor Error 94, so both are
# carried here and the bracket is recomputed in a single frame.
E121_LOCAL_KERNEL_PCT = 1.463

E116_TRANSFER = 1.000
E116_INTERVAL = (0.963, 1.038)
ROUTEB_KERNEL_TO_LEG = 0.763
ROUTEB_KERNEL_TO_LEG_INTERVAL = (0.76, 0.77)
# F5 reports thorfinn's g17s reading as 102 registers / 38 simdgroups against
# an incumbent 101 / 39. Those two pairs are byte-for-byte the E121 entry-point
# census in this file's own artifact, so the provenance is ambiguous and the
# number is recorded, flagged and NOT used to fit anything.
ROUTEB_G17S_REPORTED = {"candidate_registers": 102, "candidate_simdgroups": 38,
                        "incumbent_registers": 101, "incumbent_simdgroups": 39}

F47_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sxy / (sx * sy)


def partial(x: list[float], y: list[float], z: list[float]) -> float | None:
    """Correlation of x and y with z held constant."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    if rxy is None or rxz is None or ryz is None:
        return None
    denom = math.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return None if denom < 1e-9 else (rxy - rxz * ryz) / denom


def ols(y: list[float], columns: list[list[float]]) -> dict:
    """Least squares with an intercept, by normal equations. Small and exact
    enough for 20 rows and at most three regressors."""
    design = [[1.0] + [c[i] for c in columns] for i in range(len(y))]
    p = len(design[0])
    ata = [[sum(design[i][a] * design[i][b] for i in range(len(y)))
            for b in range(p)] for a in range(p)]
    atb = [sum(design[i][a] * y[i] for i in range(len(y))) for a in range(p)]
    for col in range(p):                       # Gauss-Jordan with partial pivot
        pivot = max(range(col, p), key=lambda r: abs(ata[r][col]))
        if abs(ata[pivot][col]) < 1e-12:
            return {"singular": True}
        ata[col], ata[pivot] = ata[pivot], ata[col]
        atb[col], atb[pivot] = atb[pivot], atb[col]
        scale = ata[col][col]
        ata[col] = [v / scale for v in ata[col]]
        atb[col] /= scale
        for row in range(p):
            if row == col:
                continue
            factor = ata[row][col]
            ata[row] = [v - factor * w for v, w in zip(ata[row], ata[col])]
            atb[row] -= factor * atb[col]
    beta = atb
    fitted = [sum(b * v for b, v in zip(beta, design[i])) for i in range(len(y))]
    mean = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, fitted))
    ss_tot = sum((a - mean) ** 2 for a in y)
    return {"singular": False, "intercept": beta[0], "coefficients": beta[1:],
            "r2": None if ss_tot < 1e-18 else 1 - ss_res / ss_tot,
            "rmse": math.sqrt(ss_res / len(y)), "n": len(y)}


def load_cells(census: dict, rate: dict) -> list[dict]:
    """One row per (shape, width) with achieved bandwidth and deletion gain."""
    groups: dict[tuple, list] = {}
    for m in rate["measurements"]:
        if m.get("kind") != "timing" or m.get("warmup"):
            continue
        groups.setdefault((m["shape"], m["m"]), []).append(m)
    rows = []
    for (shape, width), block in sorted(groups.items()):
        base = st.mean(x["seconds"]["a_base"] for x in block)
        scaffold = st.mean(x["seconds"]["q_scaffold"] for x in block)
        nosums = st.mean(x["seconds"]["n_nosums"] for x in block)
        read_bytes = block[0]["read_bytes"]
        rows.append({
            "shape": shape, "width": width, "reps": len(block),
            "read_bytes": read_bytes,
            "achieved_gb_s": read_bytes / base / 1e9,
            "a_base_us": base * 1e6,
            "gain_pct_vs_a_base": (base - nosums) / base * 100.0,
            "gain_pct_vs_scaffold": (scaffold - nosums) / scaffold * 100.0,
            "deleted_issue_lanes": issue_delta(census, "a_base", "n_nosums",
                                               width),
            "residency_pct_local": residency_delta(census, "a_base",
                                                   "n_nosums", LOCAL, width),
            "residency_pct_ranked": residency_delta(census, "a_base",
                                                    "n_nosums", RANKED, width),
        })
    return rows


def issue_delta(census: dict, low: str, high: str, width: int) -> float | None:
    a = census["arms"].get(low, {}).get("air", {}).get(str(width))
    b = census["arms"].get(high, {}).get("air", {}).get(str(width))
    if not a or not b:
        return None
    return float(a["issue_lanes"] - b["issue_lanes"])


def registers(census: dict, arm: str, arch: str, width: int) -> int | None:
    return census["arms"].get(arm, {}).get(arch, {}).get(str(width), {}) \
        .get("registers")


def residency_delta(census: dict, low: str, high: str, arch: str,
                    width: int) -> float | None:
    """Percent change in resident simdgroups going from `low` to `high`."""
    a, b = registers(census, low, arch, width), registers(census, high, arch,
                                                          width)
    if not a or not b:
        return None
    lo, hi = simdgroups(a, arch), simdgroups(b, arch)
    return None if not lo else (hi - lo) / lo * 100.0


def contrast_table(census: dict) -> list[dict]:
    """Residency content of every priced contrast, per width and architecture.

    A contrast whose two rungs allocate the same number of registers carries
    NO residency content, so its measured price is a pure instruction-channel
    reading and the two-channel law predicts it transfers at 1:1.
    """
    rows = []
    for name, low, high in CONTRASTS:
        for width in WIDTHS:
            row = {"class": name, "low_rung": low, "high_rung": high,
                   "width": width}
            for arch in (LOCAL, RANKED):
                tag = "local" if arch == LOCAL else "ranked"
                lo = registers(census, low, arch, width)
                hi = registers(census, high, arch, width)
                row["registers_%s" % tag] = None if lo is None else [lo, hi]
                row["simdgroups_%s" % tag] = (
                    None if lo is None or hi is None
                    else [simdgroups(lo, arch), simdgroups(hi, arch)])
                row["residency_pct_%s" % tag] = residency_delta(
                    census, low, high, arch, width)
            pair = (row["residency_pct_local"], row["residency_pct_ranked"])
            row["pure_instruction_channel"] = pair == (0.0, 0.0)
            rows.append(row)
    return rows


def e121_simdgroups(e121: dict, arm: str, arch: str, key: str) -> int | None:
    cell = e121["arms"].get(arm, {}).get(arch, {}).get(key, {})
    return simdgroups(cell.get("registers"), arch)


def e121_entry_residency(e121: dict, arch: str) -> float | None:
    """Percent change in resident simdgroups at the shipped entry point."""
    before = e121_simdgroups(e121, "pre_e121", arch, "entry")
    after = e121_simdgroups(e121, "share_on", arch, "entry")
    return None if not before or not after else (after - before) / before * 100


def e121_body_residency(e121: dict, arch: str) -> float | None:
    """The same, over the isolated per-width bodies, F47-weighted."""
    total = weight = 0.0
    for width, w in F47_WEIGHTS.items():
        before = e121_simdgroups(e121, "pre_e121", arch, str(width))
        after = e121_simdgroups(e121, "share_on", arch, str(width))
        if not before or not after:
            return None
        total += w * (after - before) / before
        weight += w
    return None if not weight else total / weight * 100


def channel_audit(contrasts: list[dict]) -> dict:
    """How many priced contrasts are actually pure instruction-channel reads.

    The campaign price table reads every ladder contrast as an instruction
    price. A contrast whose two rungs allocate different register counts is
    not one: part of its measured time is an occupancy change. This audit is
    free and it says which published prices are contaminated.
    """
    ladder = [r for r in contrasts if r["class"] != "deletion_vs_abase"]
    clean = [r for r in ladder if r["pure_instruction_channel"]]
    worst = max(
        ladder,
        key=lambda r: max(abs(r["residency_pct_local"] or 0.0),
                          abs(r["residency_pct_ranked"] or 0.0)))
    by_class: dict[str, dict] = {}
    for row in ladder:
        entry = by_class.setdefault(row["class"], {"widths": 0, "clean": 0,
                                                   "contaminated_widths": []})
        entry["widths"] += 1
        if row["pure_instruction_channel"]:
            entry["clean"] += 1
        else:
            entry["contaminated_widths"].append(row["width"])
    return {
        "n_contrasts": len(ladder),
        "n_pure_instruction_channel": len(clean),
        "fraction_pure": len(clean) / len(ladder) if ladder else None,
        "by_class": by_class,
        "worst_contaminated": {
            "class": worst["class"], "width": worst["width"],
            "registers_local": worst["registers_local"],
            "residency_pct_local": worst["residency_pct_local"],
            "registers_ranked": worst["registers_ranked"],
            "residency_pct_ranked": worst["residency_pct_ranked"],
        },
    }


def two_channel(e121: dict) -> dict:
    """Solve `effect = I + c * residency` on the one anchor that spans arches.

    Two equations, two unknowns, so the E121 fit is exactly determined and is
    NOT a test of the model. What IS a test is whether the coefficient lands in
    a physically sensible range and whether the instruction term comes out with
    the sign the source demands. E121 ADDS two barriers and 2*NA threadgroup
    round trips per k-block, so its instruction channel must be negative.
    """
    frames = {}
    for frame, reader in (("entry_point", e121_entry_residency),
                          ("isolated_bodies_f47_weighted",
                           e121_body_residency)):
        r_local = reader(e121, LOCAL)
        r_ranked = reader(e121, RANKED)
        if r_local is None or r_ranked is None:
            continue
        span = r_local - r_ranked
        if abs(span) < 1e-9:
            continue
        c = (E121_LOCAL_PCT - E121_RANKED_PCT) / span
        instruction = E121_LOCAL_PCT - c * r_local
        frames[frame] = {
            "residency_pct_local": r_local, "residency_pct_ranked": r_ranked,
            "coefficient_pct_time_per_pct_residency": c,
            "instruction_channel_pct": instruction,
            "instruction_channel_sign_ok": instruction < 0,
            "check_local": instruction + c * r_local,
            "check_ranked": instruction + c * r_ranked,
        }
    return frames


def coefficient_bracket(e121: dict) -> dict:
    """Recompute F5's bracket in one frame.

    F5 brackets the coefficient with `+1.463 % kernel gain / +4.86 % residency
    = 0.301` as a lower bound and `-2.100 % ranked leg / -10.57 % residency =
    0.199` as an upper bound. Those two cannot both hold, and they cannot
    because the first is a kernel-frame number and the second is a leg-frame
    number. In the leg frame on both sides the bracket is consistent.
    """
    r_local = e121_body_residency(e121, LOCAL)
    r_ranked = e121_body_residency(e121, RANKED)
    if r_local is None or r_ranked is None:
        return {}
    return {
        "as_registered_in_f5": {
            "lower_bound": E121_LOCAL_KERNEL_PCT / r_local,
            "lower_bound_frame": "kernel",
            "upper_bound": -E121_RANKED_PCT / -r_ranked,
            "upper_bound_frame": "leg",
            "consistent": E121_LOCAL_KERNEL_PCT / r_local
            <= -E121_RANKED_PCT / -r_ranked,
            "note": "lower bound above upper bound: the two sides are stated "
                    "in different frames",
        },
        "leg_frame_on_both_sides": {
            "lower_bound": E121_LOCAL_PCT / r_local,
            "upper_bound": -E121_RANKED_PCT / -r_ranked,
            "consistent": E121_LOCAL_PCT / r_local <= -E121_RANKED_PCT
            / -r_ranked,
            "frame": "leg",
        },
        "implied_kernel_to_leg_factor_e121_g16s":
            E121_LOCAL_PCT / E121_LOCAL_KERNEL_PCT,
    }


def anchor_checks(channels: dict) -> list[dict]:
    """Score the two-channel law against the three anchors."""
    out = [{
        "anchor": "E116 alpha x beta",
        "axis": "frame, share term",
        "residency_pct": 0.0,
        "observed": E116_TRANSFER,
        "observed_interval": list(E116_INTERVAL),
        "predicted": 1.000,
        "verdict": "passes",
        "is_a_test": False,
        "why": "no kernel change, so the residency channel is identically "
               "zero and the law can only return 1. A structural pass, not "
               "a numeric test.",
    }]
    for frame, record in sorted(channels.items()):
        out.append({
            "anchor": "E121 local leg g16s -> ranked leg g17s",
            "axis": "architecture",
            "frame": frame,
            "residency_pct": [record["residency_pct_local"],
                              record["residency_pct_ranked"]],
            "observed": E121_RANKED_PCT / E121_LOCAL_PCT,
            "predicted": record["check_ranked"] / record["check_local"],
            "verdict": "exactly determined",
            "is_a_test": False,
            "why": "two equations and two unknowns. The testable content is "
                   "the coefficient value and the instruction-channel sign, "
                   "not the fit residual.",
        })
    out.append({
        "anchor": "Route B rung 5e kernel frame -> leg frame",
        "axis": "frame",
        "residency_pct": 0.0,
        "observed": ROUTEB_KERNEL_TO_LEG,
        "observed_interval": list(ROUTEB_KERNEL_TO_LEG_INTERVAL),
        "predicted": 1.000,
        "verdict": "fails",
        "is_a_test": True,
        "why": "a register count is fixed at compile time, so it is identical "
               "in the kernel frame and in the leg frame on one host. The "
               "residency channel is therefore identically zero across any "
               "within-host frame pair and the law can only predict 1.000. "
               "Measured 0.763, which is 23.7 % below.",
        "shortfall_pct": (ROUTEB_KERNEL_TO_LEG - 1.0) * 100.0,
        "reported_g17s_census": ROUTEB_G17S_REPORTED,
        "provenance_flag": "F5 attaches a g17s reading of 102/38 against an "
                           "incumbent 101/39 to this anchor. That pair is "
                           "identical to the E121 entry-point census in "
                           "e121-census.json. Using it here would price E121 "
                           "twice, so it is recorded and not used.",
    })
    return out


def width_demeaned(cells: list[dict]) -> dict:
    """Correlations after subtracting each width's own mean.

    This is a width fixed effect written by hand. It removes every covariate
    that is constant inside a width -- the deleted instruction count and the
    residency delta both are -- and leaves only shape-to-shape variation.
    """
    means: dict[int, dict[str, float]] = {}
    keys = ("gain_pct_vs_a_base", "gain_pct_vs_scaffold", "achieved_gb_s")
    for width in WIDTHS:
        slice_ = [c for c in cells if c["width"] == width]
        if slice_:
            means[width] = {k: st.mean(c[k] for c in slice_) for k in keys}
    rows = [{k: c[k] - means[c["width"]][k] for k in keys} for c in cells
            if c["width"] in means]
    if len(rows) < 3:
        return {}
    return {
        "n": len(rows),
        "r_gain_bandwidth": pearson([r["gain_pct_vs_a_base"] for r in rows],
                                    [r["achieved_gb_s"] for r in rows]),
        "r_gain_bandwidth_scaffold_reference": pearson(
            [r["gain_pct_vs_scaffold"] for r in rows],
            [r["achieved_gb_s"] for r in rows]),
        "r_gain_residency": 0.0,
        "residency_variance_after_demeaning": 0.0,
        "why_residency_is_zero":
            "a register count is fixed at compile time by (arm, width, "
            "architecture). Demeaning inside a width removes it exactly.",
    }


def bandwidth_versus_residency(cells: list[dict]) -> dict:
    """The F5 question: do the Stage 0 correlations survive residency control?"""
    usable = [c for c in cells if c["residency_pct_local"] is not None]
    gain = [c["gain_pct_vs_a_base"] for c in usable]
    band = [c["achieved_gb_s"] for c in usable]
    res = [c["residency_pct_local"] for c in usable]
    deleted = [c["deleted_issue_lanes"] for c in usable]

    within = []
    for width in WIDTHS:
        slice_ = [c for c in usable if c["width"] == width]
        if len(slice_) < 3:
            continue
        residencies = {c["residency_pct_local"] for c in slice_}
        within.append({
            "width": width, "n": len(slice_),
            "r_gain_bandwidth": pearson(
                [c["gain_pct_vs_a_base"] for c in slice_],
                [c["achieved_gb_s"] for c in slice_]),
            "r_gain_bandwidth_scaffold_reference": pearson(
                [c["gain_pct_vs_scaffold"] for c in slice_],
                [c["achieved_gb_s"] for c in slice_]),
            "deleted_issue_lanes": slice_[0]["deleted_issue_lanes"],
            "residency_pct_local": sorted(residencies)[0],
            "residency_is_constant_within_width": len(residencies) == 1,
            "partial_r_available": len(residencies) > 1,
        })

    # The decisive control. Both candidate regime axes are functions of the
    # width alone over this design, so the only honest question is whether
    # either explains anything the width does not. Demeaning inside each width
    # removes the width axis exactly and leaves the shape-to-shape variation
    # that a per-cell roofline law claims to predict.
    demeaned = width_demeaned(usable)

    return {
        "n_cells": len(usable),
        "pooled": {
            "r_gain_bandwidth": pearson(gain, band),
            "r_gain_residency": pearson(gain, res),
            "r_bandwidth_residency": pearson(band, res),
            "r_gain_deleted_instructions": pearson(gain, deleted),
            "r_bandwidth_deleted_instructions": pearson(band, deleted),
            "partial_r_gain_bandwidth_given_residency":
                partial(gain, band, res),
            "partial_r_gain_residency_given_bandwidth":
                partial(gain, res, band),
        },
        "within_width": within,
        "width_demeaned": demeaned,
        "collinearity_note":
            "residency and deleted instruction count are both functions of "
            "the width alone over this design, and achieved bandwidth is "
            "ordered by width too. The pooled correlations cannot separate "
            "them. Only the width-demeaned column can, and there residency "
            "is identically zero by construction.",
        "models": {
            "instructions_only": ols(gain, [deleted]),
            "instructions_and_bandwidth": ols(gain, [deleted, band]),
            "instructions_and_residency": ols(gain, [deleted, res]),
            "instructions_bandwidth_and_residency":
                ols(gain, [deleted, band, res]),
            "bandwidth_only": ols(gain, [band]),
            "residency_only": ols(gain, [res]),
        },
    }


def report(out: pathlib.Path | None) -> int:
    census = json.loads(E123_CENSUS.read_text())
    rate = json.loads(E123_RATE.read_text())
    e121 = json.loads(E121_CENSUS.read_text())

    cells = load_cells(census, rate)
    contrasts = contrast_table(census)
    audit = channel_audit(contrasts)
    channels = two_channel(e121)
    bracket = coefficient_bracket(e121)
    anchors = anchor_checks(channels)
    stats = bandwidth_versus_residency(cells)

    print("E125 F5: the two-channel law and the residency covariate")
    print("harness=local  timing_valid=false  no GPU used by this file")
    print("simdgroup budget %s" % SIMDGROUP_BUDGET)

    print("\n=== 1. residency content of every priced contrast "
          "(local / ranked, percent) ===")
    print("%-19s %5s %16s %16s  %s"
          % ("class", "width", "g16s", "g17s", "pure instruction"))
    for row in contrasts:
        def fmt(tag):
            regs = row["registers_%s" % tag]
            pct = row["residency_pct_%s" % tag]
            if regs is None or pct is None:
                return "%16s" % "not censused"
            return "%3d->%3d %+6.2f%%" % (regs[0], regs[1], pct)
        print("%-19s %5d %s %s  %s"
              % (row["class"], row["width"], fmt("local"), fmt("ranked"),
                 "yes" if row["pure_instruction_channel"] else "no"))

    print("  %d of %d priced ladder contrasts are pure instruction channel "
          "on both architectures" % (audit["n_pure_instruction_channel"],
                                     audit["n_contrasts"]))
    for name, entry in sorted(audit["by_class"].items()):
        print("    %-14s clean %d of %d   contaminated at widths %s"
              % (name, entry["clean"], entry["widths"],
                 entry["contaminated_widths"] or "none"))
    worst = audit["worst_contaminated"]
    print("    worst: %s at NA=%d, g16s %s = %+0.2f %% residency"
          % (worst["class"], worst["width"], worst["registers_local"],
             worst["residency_pct_local"]))

    print("\n=== 2. the two-channel solve on the E121 architecture anchor ===")
    print("local leg %+0.3f %%   ranked leg %+0.3f %%" % (E121_LOCAL_PCT,
                                                          E121_RANKED_PCT))
    for frame, record in sorted(channels.items()):
        print("  %-32s c = %+.4f %% time per %% residency   "
              "instruction channel %+.4f %%  sign %s"
              % (frame, record["coefficient_pct_time_per_pct_residency"],
                 record["instruction_channel_pct"],
                 "ok" if record["instruction_channel_sign_ok"] else "WRONG"))

    print("\n=== 3. F5's coefficient bracket, recomputed in one frame ===")
    if bracket:
        a = bracket["as_registered_in_f5"]
        b = bracket["leg_frame_on_both_sides"]
        print("  as registered   lower %.3f (%s frame)  upper %.3f (%s frame)"
              "  consistent=%s"
              % (a["lower_bound"], a["lower_bound_frame"], a["upper_bound"],
                 a["upper_bound_frame"], a["consistent"]))
        print("  leg frame both  lower %.3f  upper %.3f  consistent=%s"
              % (b["lower_bound"], b["upper_bound"], b["consistent"]))
        print("  implied E121 kernel-to-leg factor on g16s: %.3f"
              % bracket["implied_kernel_to_leg_factor_e121_g16s"])

    print("\n=== 4. the law against the three anchors ===")
    for row in anchors:
        print("  %-42s %-30s axis=%-12s observed %s  predicted %s  %s  test=%s"
              % (row["anchor"], row.get("frame", ""), row["axis"],
                 "%+0.3f" % row["observed"], "%+0.3f" % row["predicted"],
                 row["verdict"], row["is_a_test"]))

    print("\n=== 5. bandwidth against residency on the 20 E123 cells ===")
    pooled = stats["pooled"]
    for key in sorted(pooled):
        value = pooled[key]
        print("  %-52s %s" % (key, "n/a" if value is None else "%+.3f" % value))
    print("  within-width:")
    for row in stats["within_width"]:
        print("    NA=%d n=%d  r(gain,bandwidth)=%s  deleted lanes %.0f  "
              "residency %+0.2f %% and constant=%s  partial available=%s"
              % (row["width"], row["n"],
                 "n/a" if row["r_gain_bandwidth"] is None
                 else "%+.3f" % row["r_gain_bandwidth"],
                 row["deleted_issue_lanes"], row["residency_pct_local"],
                 row["residency_is_constant_within_width"],
                 row["partial_r_available"]))
    demeaned = stats.get("width_demeaned", {})
    if demeaned:
        print("  width-demeaned, n=%d: r(gain,bandwidth)=%+.3f  "
              "r(gain,residency)=%+.3f by construction"
              % (demeaned["n"], demeaned["r_gain_bandwidth"],
                 demeaned["r_gain_residency"]))
    print("  models, response = deletion gain %% of a_base:")
    for name, model in stats["models"].items():
        if model.get("singular"):
            print("    %-42s singular" % name)
            continue
        print("    %-42s r2=%+.4f  rmse=%.4f  coefficients=%s"
              % (name, model["r2"], model["rmse"],
                 ["%+.5f" % c for c in model["coefficients"]]))

    payload = {
        "harness": "local",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "uses_gpu": False,
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "sources": {"e123_census": str(E123_CENSUS.relative_to(ROOT)),
                    "e123_rate": str(E123_RATE.relative_to(ROOT)),
                    "e121_census": str(E121_CENSUS.relative_to(ROOT))},
        "anchors": {"e121_local_leg_pct": E121_LOCAL_PCT,
                    "e121_ranked_leg_pct": E121_RANKED_PCT,
                    "e121_local_leg_sem_pct": E121_LOCAL_SEM_PCT,
                    "e121_ranked_leg_sem_pct": E121_RANKED_SEM_PCT,
                    "e121_local_kernel_pct": E121_LOCAL_KERNEL_PCT,
                    "e116_transfer": E116_TRANSFER,
                    "routeb_kernel_to_leg": ROUTEB_KERNEL_TO_LEG},
        "contrast_residency": contrasts,
        "channel_audit": audit,
        "two_channel": channels,
        "coefficient_bracket": bracket,
        "anchor_checks": anchors,
        "bandwidth_versus_residency": stats,
        "cells": cells,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print("\nwrote %s" % out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    return report(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
