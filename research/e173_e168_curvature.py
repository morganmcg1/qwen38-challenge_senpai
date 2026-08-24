#!/usr/bin/env python3
"""E173: re-run the tablePays boundary curvature test on the E168 paired data.

Input: the `e168-round-cost-pairs` W&B artifact (Askeladd, 26 legs, 5210
rounds). Download it first:

    python3 -c "import wandb; wandb.Api().artifact(
      'wandb-applied-ai-team/qwen38-mlx-challenge-senpai/'
      'e168-round-cost-pairs:latest').download('/tmp/e168-pairs')"

`harness=local`. Every source leg ran with `MLXFAST_LOCAL_COOL_GATE=0` and the
phase trace on, so `gate_qualified_for_timing=false`,
`timing_claims_permitted=false` and `trace_perturbs_timing=true`. Only
within-run differentials are meaningful. The trace tax is depth-independent
(about 592 us per round), so the second-difference estimator cancels it.

The estimator is curvature: round(m+1) - 2*round(m) + round(m-1). A tablePays
step at m = 4 raises curvature at m = 3 and lowers it at m = 4, so the
boundary-minus-non-boundary contrast is the identified quantity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
BOOTSTRAP = 4000
SEED = 173


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        rows.append(
            {
                "arm": r["arm"],
                "prompt": r["prompt"],
                "round_index": int(r["round_index"]),
                "ms": float(r["block_seconds"]) * 1e3,
                "m": int(r["m"]),
                "weight_passes": int(r["weight_passes"]),
            }
        )
    return rows


def curvature(med: dict[int, float], m: int) -> float | None:
    if all(k in med for k in (m - 1, m, m + 1)):
        return med[m + 1] - 2 * med[m] + med[m - 1]
    return None


def medians(rows: list[dict]) -> dict[int, float]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["m"]].append(r["ms"])
    return {m: statistics.median(v) for m, v in by.items()}


def stratified_resample(
    strata: dict[int, list[dict]], rng: random.Random
) -> list[dict]:
    out = []
    for rows in strata.values():
        out.extend(rng.choices(rows, k=len(rows)))
    return out


def contrast(rows: list[dict], boundary: int, control: int, label: str) -> dict:
    strata = collections.defaultdict(list)
    for r in rows:
        strata[r["m"]].append(r)
    med = medians(rows)
    b = curvature(med, boundary)
    c = curvature(med, control)
    if b is None or c is None:
        return {
            "label": label,
            "usable": False,
            "widths_present": sorted(med),
            "reason": "the estimator needs three consecutive widths at both "
            f"m = {boundary} and m = {control}",
        }
    rng = random.Random(SEED)
    diffs, bs, cs = [], [], []
    for _ in range(BOOTSTRAP):
        med_b = medians(stratified_resample(strata, rng))
        cb = curvature(med_b, boundary)
        cc = curvature(med_b, control)
        if cb is None or cc is None:
            continue
        bs.append(cb)
        cs.append(cc)
        diffs.append(cb - cc)
    diffs.sort()
    bs.sort()
    cs.sort()

    def ci(v: list[float]) -> list[float]:
        lo = v[int(0.025 * len(v))]
        hi = v[int(0.975 * len(v)) - 1]
        return [lo, hi]

    return {
        "label": label,
        "usable": True,
        "n_rounds": len(rows),
        "counts_by_m": {m: len(v) for m, v in sorted(strata.items())},
        "median_ms_by_m": {m: med[m] for m in sorted(med)},
        f"boundary_curvature_m{boundary}_ms": b,
        f"boundary_ci": ci(bs),
        f"control_curvature_m{control}_ms": c,
        f"control_ci": ci(cs),
        "contrast_ms": b - c,
        "contrast_ci": ci(diffs),
        "bootstrap": len(diffs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/tmp/e168-pairs/round_pairs.csv")
    ap.add_argument("--drop-first-rounds", type=int, default=2)
    args = ap.parse_args()

    rows = load(pathlib.Path(args.pairs))
    rows = [r for r in rows if r["round_index"] >= args.drop_first_rounds]

    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    print("arm coverage (rounds by verify width m)")
    for arm, arm_rows in sorted(by_arm.items()):
        counts = collections.Counter(r["m"] for r in arm_rows)
        print(f"  {arm:6s} n={len(arm_rows):5d} " + " ".join(
            f"m{m}={counts[m]}" for m in sorted(counts)))

    # One weight pass only. `groups(M)` steps at M = 6, and a group crossing is
    # a far larger effect than the one under test, so mixing pass counts would
    # swamp the estimator.
    single_pass = [r for r in rows if r["weight_passes"] == 1]

    results = [
        contrast(single_pass, 3, 4, "all arms, single weight pass"),
        contrast(
            [r for r in single_pass if r["arm"] == "adapt"],
            3,
            4,
            "adaptive arm only (m is endogenous)",
        ),
        contrast(
            [r for r in single_pass if r["arm"] != "adapt"],
            3,
            4,
            "imposed-depth arms only (m is exogenous)",
        ),
    ]

    for res in results:
        print(f"\n--- {res['label']}")
        if not res["usable"]:
            print(f"    not usable: {res['reason']}")
            print(f"    widths present: {res['widths_present']}")
            continue
        print(f"    n={res['n_rounds']} counts={res['counts_by_m']}")
        print(
            "    medians ms: "
            + " ".join(f"m{m}={v:.2f}" for m, v in res["median_ms_by_m"].items())
        )
        print(
            f"    boundary curvature(3) = {res['boundary_curvature_m3_ms']:.2f} ms "
            f"CI {res['boundary_ci'][0]:.2f}..{res['boundary_ci'][1]:.2f}"
        )
        print(
            f"    control  curvature(4) = {res['control_curvature_m4_ms']:.2f} ms "
            f"CI {res['control_ci'][0]:.2f}..{res['control_ci'][1]:.2f}"
        )
        print(
            f"    contrast = {res['contrast_ms']:+.2f} ms "
            f"CI {res['contrast_ci'][0]:+.2f}..{res['contrast_ci'][1]:+.2f}"
        )

    imposed = [r for r in single_pass if r["arm"] != "adapt"]
    imposed_counts = collections.Counter(r["m"] for r in imposed)
    coverage_note = (
        "The fixed-depth arms impose depth 2 (m = 3) and depth 7 (m = 8) only. "
        f"At a single weight pass they contribute m = 3: {imposed_counts[3]}, "
        f"m = 2: {imposed_counts[2]}, m = 4: {imposed_counts[4]}, "
        f"m = 5: {imposed_counts[5]}. The m = 2, 4 and 5 rows are truncated "
        "drafts, not imposed widths, so the exogenous-m curvature triple the "
        "estimator needs does not exist in this artifact. The pooled estimate "
        "is confounded by arm composition: the m = 3 stratum is dominated by "
        "the p2 arm while m = 4 and m = 5 come only from the adaptive arm, so "
        "any level difference between arms enters the second difference "
        "directly. The adaptive-arm estimate is the identified one here."
    )
    print(f"\nCOVERAGE: {coverage_note}")

    out = ROOT / "e173-artifacts/e168-curvature.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "e173-e168-boundary-curvature",
                "harness": "local",
                "source_artifact": "e168-round-cost-pairs (Askeladd, run k18flizk)",
                "gate_qualified_for_timing": False,
                "timing_claims_permitted": False,
                "trace_perturbs_timing": True,
                "trace_tax_note": "about 592 us per round, depth independent, so "
                "the second difference cancels it",
                "drop_first_rounds": args.drop_first_rounds,
                "rounds_after_drop": len(rows),
                "single_weight_pass_rounds": len(single_pass),
                "bootstrap_draws": BOOTSTRAP,
                "seed": SEED,
                "arm_coverage": {
                    arm: dict(
                        sorted(collections.Counter(r["m"] for r in v).items())
                    )
                    for arm, v in sorted(by_arm.items())
                },
                "coverage_note": coverage_note,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
