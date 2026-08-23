#!/usr/bin/env python3
"""E147: is the `e003a86d` / `7226dc9a` candidate-leg gap a base difference?

F8 measured a `+4.3978 %` weighted-five candidate-leg regression between
`e003a86d` and our `7226dc9a`, on a pair that is head-matched and
schedule-matched, and asked whether rung A or rung B caused it.

This script tests the competing explanation: the two rows were built from
different campaign trees, and the arms that differ are already priced on the
candidate leg by the campaign's own Rule 118 receipts.

Three arm values are read out of the live worktree with fail-closed regexes.
The `e003a86d` values are quoted from the ledger. The contrasts come from
FINDING 201, whose `cand8` column is the 8-prompt candidate-leg contrast in
percent, which is exactly the quantity F8 measured.

Zero GPU. Source reads and arithmetic only.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
LEDGER = ROOT / "senpai/campaign-ledger.md"
OUT = ROOT / "research/e147-base-diff.json"

# (arm, file, regex with one capture group, value for e003a86d, ledger line)
ARM_PROBES = [
    (
        "qmv_launch_grid",
        QWEN35,
        r'environment\["MLX_E120_QMV_GRID"\][\s\S]{0,200}?else \{ return \.(\w+) \}',
        "tight",
        51511,
    ),
    (
        "qmv_width_table",
        QWEN35,
        r"public static let compiledDefault = Table\.(\w+)",
        "shipped",
        51511,
    ),
    (
        "qmv_entry_points",
        QWEN35,
        r"public static let compiledDefault = Entry\.(\w+)",
        "tiered",
        51100,
    ),
    (
        "cluster_probe_fraction",
        QWEN35,
        r"qwen35DerivedClusterProbeFraction: Double = ([\d.]+)",
        "0.15",
        51511,
    ),
    (
        "depth_price_arm",
        SESSION,
        r'DepthPriceArm\(rawValue: requested\) \?\? \.(\w+)',
        "pb6",
        51511,
    ),
    (
        "depth_price_tier",
        SESSION,
        r"passBoundaryTierFactor = ([\d.]+)",
        "1.45",
        51511,
    ),
]

# FINDING 201, ledger line 51150. `cand8` is the 8-prompt candidate-leg
# contrast in percent under Rule 118. Negative means the candidate leg got
# faster. Each entry maps one arm change on OUR tree towards `e003a86d`.
#
# `sign` is +1 when the listed contrast already describes the direction our
# tree must move to reach `e003a86d`, and -1 when the ledger measured the
# opposite direction and the value must be inverted.
CONTRASTS = [
    {
        "arm": "qmv_launch_grid",
        "change": "wide -> tight",
        "ledger_row": "OURS   tight grid, onePass67",
        "ledger_pair": "623e77af -> 572b2cc4",
        "ledger_line": 51150,
        "cand8_pct": -3.7674,
        "sign": +1,
    },
    {
        "arm": "qmv_width_table",
        "change": "onePass67 -> shipped",
        "ledger_row": "F194   add onePass67 to tight base",
        "ledger_pair": "F194",
        "ledger_line": 51152,
        "cand8_pct": +0.3280,
        "sign": -1,  # the ledger ADDS the table; we REMOVE it
    },
    {
        "arm": "cluster_probe_fraction",
        "change": "0.25 -> 0.15",
        "ledger_row": "F192   probe 0.15 on tight base",
        "ledger_pair": "F192",
        "ledger_line": 51153,
        "cand8_pct": -0.1961,
        "sign": +1,
    },
]

# F8 section 2. Per-prompt candidate-leg d%, `e003a86d` -> `7226dc9a`.
# Positive means our leg is slower.
F8_CANDIDATE_DPCT = {
    "beagle": 5.1348,
    "drama": 4.0762,
    "republic": 3.7663,
    "essays": 3.7622,
    "medicine": 3.1803,
    "botany": 3.8434,
    "travel": 4.6656,
}
F8_WEIGHTED_FIVE_PCT = 4.3978
F8_WEIGHTED_FIVE_SD = 0.7204
F8_WEIGHTED_FIVE_N = 5


def read_arm(path: pathlib.Path, pattern: str) -> str:
    m = re.search(pattern, path.read_text())
    if not m:
        raise SystemExit(
            f"e147_base_diff: pattern did not match in {path.name}: {pattern}"
        )
    return m.group(1)


def ledger_line(n: int) -> str:
    lines = LEDGER.read_text().splitlines()
    return lines[n - 1].strip() if 0 < n <= len(lines) else ""


def main() -> int:
    ours = {}
    arm_rows = []
    for arm, path, pattern, theirs, cite in ARM_PROBES:
        got = read_arm(path, pattern)
        ours[arm] = got
        arm_rows.append(
            {
                "arm": arm,
                "ours_770a3ff2": got,
                "theirs_e003a86d": theirs,
                "differs": got != theirs,
                "source": f"{path.relative_to(ROOT)}",
                "ledger_line": cite,
            }
        )

    differing = {r["arm"] for r in arm_rows if r["differs"]}
    priced = {c["arm"] for c in CONTRASTS}
    unpriced = sorted(differing - priced)
    priced_but_equal = sorted(priced - differing)

    terms = []
    additive = 0.0
    multiplicative = 1.0
    for c in CONTRASTS:
        applies = c["arm"] in differing
        effect = c["sign"] * c["cand8_pct"] if applies else 0.0
        if applies:
            additive += effect
            multiplicative *= 1.0 + effect / 100.0
        terms.append(
            {
                **c,
                "applies": applies,
                "applied_cand8_pct": effect,
                "ledger_text": ledger_line(c["ledger_line"]),
            }
        )

    # `additive` and `multiplicative` describe how much FASTER e003a86d's leg
    # is. Invert to get how much SLOWER ours is, which is F8's sign.
    predicted_additive = -additive
    predicted_multiplicative = (1.0 / multiplicative - 1.0) * 100.0

    obs = list(F8_CANDIDATE_DPCT.values())
    obs_mean = statistics.mean(obs)
    obs_sd = statistics.stdev(obs)
    obs_se = obs_sd / len(obs) ** 0.5
    w5_se = F8_WEIGHTED_FIVE_SD / F8_WEIGHTED_FIVE_N**0.5

    comparisons = []
    for obs_name, obs_val, obs_se_val in (
        ("F8 weighted-five", F8_WEIGHTED_FIVE_PCT, w5_se),
        ("unweighted 7-prompt mean", obs_mean, obs_se),
    ):
        for pred_name, pred_val in (
            ("additive", predicted_additive),
            ("multiplicative", predicted_multiplicative),
        ):
            resid = obs_val - pred_val
            comparisons.append(
                {
                    "observed": obs_name,
                    "observed_pct": round(obs_val, 4),
                    "observed_se_pp": round(obs_se_val, 4),
                    "composition": pred_name,
                    "predicted_pct": round(pred_val, 4),
                    "residual_pp": round(resid, 4),
                    "residual_sigma": round(resid / obs_se_val, 3),
                }
            )

    max_abs_sigma = max(abs(c["residual_sigma"]) for c in comparisons)
    explained = max_abs_sigma < 2.0 and not unpriced

    report = {
        "experiment": "e147",
        "question": (
            "does the base tree difference explain F8's +4.3978 % "
            "candidate-leg gap between e003a86d and 7226dc9a?"
        ),
        "harness": "ranked",
        "our_base": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
        "their_row": "e003a86d",
        "arm_configuration": arm_rows,
        "arms_that_differ": sorted(differing),
        "arms_that_match": sorted(a for a in ours if a not in differing),
        "differing_arms_without_a_ledger_price": unpriced,
        "priced_contrasts_not_applicable": priced_but_equal,
        "contrast_terms": terms,
        "predicted_our_leg_slower_pct_additive": round(predicted_additive, 4),
        "predicted_our_leg_slower_pct_multiplicative": round(
            predicted_multiplicative, 4
        ),
        "observed_per_prompt_candidate_dpct": F8_CANDIDATE_DPCT,
        "observed_unweighted_mean_pct": round(obs_mean, 4),
        "observed_unweighted_sd_pp": round(obs_sd, 4),
        "observed_weighted_five_pct": F8_WEIGHTED_FIVE_PCT,
        "comparisons": comparisons,
        "max_abs_residual_sigma": max_abs_sigma,
        "e147_base_arm_gap_explains_four_percent": explained,
        "caveats": [
            "FINDING 201 measured every contrast on trees without pb6. The "
            "tight-grid saving is 1296.8 * ln(Mbar) microseconds per round "
            "(FINDING 200), so its percentage value depends on the width "
            "distribution, and pb6 changes that distribution. The three terms "
            "are therefore approximately, not exactly, transferable.",
            "The three differing arms are all draft-length neutral by the "
            "ledger's own statement at line 51527, which is why the pair "
            "stays schedule-matched while the leg time moves.",
            "This result prices the base gap. It does not by itself prove "
            "rung A and rung B are inert; it removes the evidence that they "
            "are not.",
        ],
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print("=== arm configuration: our base 770a3ff2 against e003a86d")
    hdr = f"{'arm':<24} {'ours':<12} {'e003a86d':<12} {'differs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in arm_rows:
        print(
            f"{r['arm']:<24} {r['ours_770a3ff2']:<12} "
            f"{r['theirs_e003a86d']:<12} {str(r['differs']):>8}"
        )

    print()
    print("=== FINDING 201 candidate-leg contrasts applied to the differing arms")
    hdr = f"{'arm':<24} {'change':<24} {'cand8 %':>9} {'sign':>5} {'applied %':>10}"
    print(hdr)
    print("-" * len(hdr))
    for t in terms:
        print(
            f"{t['arm']:<24} {t['change']:<24} {t['cand8_pct']:>9.4f} "
            f"{t['sign']:>5} {t['applied_cand8_pct']:>10.4f}"
        )

    print()
    print(
        f"predicted: our leg is slower by "
        f"{predicted_additive:.4f} % (additive) / "
        f"{predicted_multiplicative:.4f} % (multiplicative)"
    )

    print()
    print("=== observed against predicted")
    hdr = (
        f"{'observed':<26} {'obs %':>8} {'se pp':>7} {'model':<16} "
        f"{'pred %':>8} {'resid pp':>9} {'sigma':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in comparisons:
        print(
            f"{c['observed']:<26} {c['observed_pct']:>8.4f} "
            f"{c['observed_se_pp']:>7.4f} {c['composition']:<16} "
            f"{c['predicted_pct']:>8.4f} {c['residual_pp']:>9.4f} "
            f"{c['residual_sigma']:>7.3f}"
        )

    print()
    if unpriced:
        print(f"UNPRICED differing arms: {unpriced}")
    print(
        "e147_base_arm_gap_explains_four_percent = "
        f"{report['e147_base_arm_gap_explains_four_percent']} "
        f"(max |residual| {max_abs_sigma:.2f} sigma)"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
