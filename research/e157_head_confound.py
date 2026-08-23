"""E157 R0: test FINDING 295's head claim against data already on disk.

FINDING 295 says the local candidate leg drafts with the pinned bf16 head while
the ranked candidate leg drafts with the declared q2/q4 head, and infers that
RULE 166's 2.75 divisor is contaminated by roughly 4.5 %.

Three checks, all zero GPU:

  1. Which head did the local legs behind the local row law actually load?
     `head_provenance` is in every report payload, so this is a file read.
  2. How much does the head choice actually cost per drafted row locally? The
     E79 gated census ran matched declared and pinned arms with per-phase
     timing, so the term FINDING 295 could only infer is already measured.
  3. Is the ranked within-prompt width sweep head-homogeneous? If deep cells
     ran a different head than shallow cells, the measured curvature could be
     head heterogeneity instead of a row price.

Check 3 is a threat to my own headline result, so it runs even though nothing
asked for it.

    python3 research/e157_head_confound.py
"""
from __future__ import annotations

import json
import os
import statistics

from e157_ranked_width_sweep import (
    SWEEP_PROMPT,
    PROMPT_NAMES,
    collect_cells,
    load_rows,
    lstsq,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "e157-artifacts")
LOCAL_LEG_DIR = ".mlxfast-private/e154/runs-r2a"
E79_ARMS = {
    "declared_census512": "research/e79-census-r5-gated-decl-a1-census512.json",
    "pinned_census512": "research/e79-census-r6-gated-pin-a1-census512.json",
    "declared_synchead512": "research/e79-census-r5-gated-decl-b1-synchead512.json",
    "pinned_synchead512": "research/e79-census-r6-gated-pin-b1-synchead512.json",
}
# mtp-head.manifest.json, the tree the ranked runner digest-verifies.
DECLARED_MANIFEST_BYTES = 427742600
PINNED_HEAD_BYTES = 849400347


def local_leg_heads() -> dict:
    """Which head did every leg behind the local row law load?"""
    seen = {}
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), LOCAL_LEG_DIR)
    for leg in sorted(os.listdir(root)):
        report = os.path.join(root, leg, "report.json")
        if not os.path.isfile(report):
            continue
        with open(report) as handle:
            head = json.load(handle).get("head_provenance") or {}
        seen[leg] = {
            "sha256": head.get("sha256"),
            "bytes": head.get("bytes"),
            "file_count": head.get("file_count"),
            "origin": head.get("origin"),
        }
    digests = {v["sha256"] for v in seen.values()}
    byte_counts = {v["bytes"] for v in seen.values()}
    single = digests.pop() if len(digests) == 1 else None
    only_bytes = byte_counts.pop() if len(byte_counts) == 1 else None
    return {
        "legs": seen,
        "single_head_across_legs": single is not None,
        "head_sha256": single,
        "head_bytes": only_bytes,
        # The report counts config.json beside model.safetensors; the manifest
        # digests the single safetensors tree.
        "safetensors_bytes_match_manifest": (
            only_bytes is not None
            and 0 < only_bytes - DECLARED_MANIFEST_BYTES < 8192
        ),
        "is_pinned_bf16_head": only_bytes == PINNED_HEAD_BYTES,
    }


def head_price_from_e79() -> dict:
    """Measured declared-versus-pinned head cost, matched and gated."""
    arms = {}
    for name, path in E79_ARMS.items():
        full = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        with open(full) as handle:
            leg = json.load(handle)["legs"][0]
        arms[name] = {
            "rounds_scored": leg["rounds_scored"],
            "mean_width": leg["M"],
            "tokens_per_round": leg["tokens_per_round"],
            "round_slope_ms_per_draft": leg["phase_fit"]["round"][
                "slope_ms_per_draft"
            ],
            "round_fixed_ms": leg["phase_fit"]["round"]["fixed_ms"],
            "round_median_ms": leg["phase_ms"]["round"]["median"],
            "draft_build_slope_ms_per_draft": leg["phase_fit"]["draft_build"][
                "slope_ms_per_draft"
            ],
            "draft_build_median_ms": leg["phase_ms"]["draft_build"]["median"],
            "verify_build_slope_ms_per_draft": leg["phase_fit"]["verify_build"][
                "slope_ms_per_draft"
            ],
        }

    # The synchead arms exist to attribute phase time; the census arms let the
    # draft step overlap the verify submit, so only synchead prices the head.
    decl = arms["declared_synchead512"]
    pin = arms["pinned_synchead512"]
    head_delta_ms = (
        pin["draft_build_slope_ms_per_draft"]
        - decl["draft_build_slope_ms_per_draft"]
    )
    round_delta_ms = (
        pin["round_slope_ms_per_draft"] - decl["round_slope_ms_per_draft"]
    )
    byte_ratio = PINNED_HEAD_BYTES / DECLARED_MANIFEST_BYTES
    return {
        "arms": arms,
        "harness": "local",
        "draft_step_slope_ratio_pinned_over_declared": (
            pin["draft_build_slope_ms_per_draft"]
            / decl["draft_build_slope_ms_per_draft"]
        ),
        "head_bytes_ratio_pinned_over_declared": byte_ratio,
        "draft_step_extra_ms_per_row_for_pinned": head_delta_ms,
        "round_slope_extra_ms_per_row_for_pinned": round_delta_ms,
        "round_slope_inflation_pinned_over_declared": (
            pin["round_slope_ms_per_draft"] / decl["round_slope_ms_per_draft"]
        ),
        "head_share_of_round_declared": (
            decl["draft_build_median_ms"] / decl["round_median_ms"]
        ),
        "head_share_of_round_pinned": (
            pin["draft_build_median_ms"] / pin["round_median_ms"]
        ),
        # Verify is the same target on both arms, so it is the null control.
        "verify_slope_ratio_pinned_over_declared": (
            pin["verify_build_slope_ms_per_draft"]
            / decl["verify_build_slope_ms_per_draft"]
        ),
    }


def head_of_entry(rows: list[dict]) -> dict[tuple[str, str], str]:
    """(submission id8, prompt name) -> head digest on the ranked receipt."""
    out = {}
    for row in rows:
        metrics = (row.get("officialMetrics") or {}).get("per_prompt") or []
        for entry in metrics:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
            if name:
                out[(row["id"][:8], name)] = str(
                    entry.get("head_provenance_sha256")
                )
    return out


def sweep_head_stratification(rows: list[dict]) -> dict:
    """Does the measured ranked curvature survive holding the head fixed?"""
    heads = head_of_entry(rows)
    cells = sorted(
        collect_cells(rows).get(SWEEP_PROMPT, []), key=lambda c: c["width"]
    )
    for c in cells:
        c["head"] = heads.get((c["id8"], SWEEP_PROMPT), "unknown")

    by_head: dict[str, list[dict]] = {}
    for c in cells:
        by_head.setdefault(c["head"], []).append(c)

    print(f"\n{SWEEP_PROMPT} cells by ranked head digest")
    print(f"{'id8':9s} {'width':>6s} {'us/rnd':>9s}  head")
    for c in cells:
        print(
            f"{c['id8']:9s} {c['width']:6.3f} "
            f"{c['clean_us_per_round']:9.0f}  {c['head'][:12]}"
        )

    result = {
        "harness": "ranked",
        "n_cells": len(cells),
        "n_distinct_heads": len(by_head),
        "cells_per_head": {h[:12]: len(v) for h, v in by_head.items()},
        "head_homogeneous": len(by_head) == 1,
    }

    # Does head correlate with width? That is the confound that would
    # manufacture curvature.
    widths = [c["width"] for c in cells]
    grand = statistics.fmean(widths)
    result["mean_width_by_head"] = {
        h[:12]: statistics.fmean([c["width"] for c in v])
        for h, v in by_head.items()
    }
    result["grand_mean_width"] = grand

    # Fit 1: the largest single-head subset, no head variation at all.
    largest = max(by_head.items(), key=lambda kv: len(kv[1]))
    subset = sorted(largest[1], key=lambda c: c["width"])
    result["largest_head"] = largest[0][:12]
    result["largest_head_cells"] = [
        {
            "id8": c["id8"],
            "width": c["width"],
            "rounds": c["rounds"],
            "clean_us_per_round": c["clean_us_per_round"],
        }
        for c in subset
    ]
    if len(subset) >= 4:
        sw = [c["width"] for c in subset]
        sc = [c["clean_us_per_round"] for c in subset]
        linear = lstsq([[1.0, w] for w in sw], sc)
        beta, stderr, r_squared, _ = linear
        result["single_head_linear"] = {
            "n": len(subset),
            "width_range": [min(sw), max(sw)],
            "fixed_us": beta[0],
            "marginal_us_per_row": beta[1],
            "marginal_stderr": stderr[1],
            "r_squared": r_squared,
        }
        if len(subset) >= 5 and len(set(sw)) >= 3:
            quad = lstsq([[1.0, w, w * w] for w in sw], sc)
            beta, stderr, r_squared, _ = quad
            result["single_head_quadratic"] = {
                "n": len(subset),
                "beta_us": beta,
                "curvature_us_per_row_squared": beta[2],
                "curvature_stderr": stderr[2],
                "curvature_sigma": (
                    beta[2] / stderr[2] if stderr[2] > 0 else None
                ),
                "r_squared": r_squared,
            }

    # Fit 1b: inside that one head, also control build speed with the same
    # submission's near-width-one plutarch cell.
    index_cells = {
        c["id8"]: c
        for c in collect_cells(rows).get("plutarch", [])
        if c["width"] < 1.5
    }
    paired = [c for c in subset if c["id8"] in index_cells]
    if len(paired) >= 4:
        design = [
            [1.0, index_cells[c["id8"]]["clean_us_per_round"], c["width"]]
            for c in paired
        ]
        target = [c["clean_us_per_round"] for c in paired]
        fitted = lstsq(design, target)
        if fitted:
            beta, stderr, r_squared, _ = fitted
            result["single_head_build_speed_controlled"] = {
                "n": len(paired),
                "width_range": [
                    min(c["width"] for c in paired),
                    max(c["width"] for c in paired),
                ],
                "intercept_us": beta[0],
                "index_loading": beta[1],
                "marginal_us_per_row": beta[2],
                "marginal_stderr": stderr[2],
                "r_squared": r_squared,
            }

    # Fit 2: every cell, with one dummy per head. Width effects are then
    # identified only within head.
    ordered = sorted(by_head, key=lambda h: -len(by_head[h]))
    reference = ordered[0]
    others = ordered[1:]
    design, target = [], []
    for c in cells:
        row = [1.0, c["width"], c["width"] ** 2]
        row += [1.0 if c["head"] == h else 0.0 for h in others]
        design.append(row)
        target.append(c["clean_us_per_round"])
    if len(cells) > len(design[0]) + 1:
        fitted = lstsq(design, target)
        if fitted:
            beta, stderr, r_squared, _ = fitted
            result["head_fixed_effect_quadratic"] = {
                "n": len(cells),
                "reference_head": reference[:12],
                "intercept_us": beta[0],
                "linear_us_per_row": beta[1],
                "linear_stderr": stderr[1],
                "curvature_us_per_row_squared": beta[2],
                "curvature_stderr": stderr[2],
                "curvature_sigma": (
                    beta[2] / stderr[2] if stderr[2] > 0 else None
                ),
                "r_squared": r_squared,
                "head_offsets_us": {
                    h[:12]: beta[3 + i] for i, h in enumerate(others)
                },
            }
    return result


def main() -> None:
    rows = load_rows()
    out = {
        "experiment": "e157-r0-finding-295-head-confound",
        "local_legs_behind_the_local_row_law": local_leg_heads(),
        "measured_head_price": head_price_from_e79(),
        "ranked_sweep_head_stratification": sweep_head_stratification(rows),
    }

    local = out["local_legs_behind_the_local_row_law"]
    price = out["measured_head_price"]
    strat = out["ranked_sweep_head_stratification"]

    print("\n--- FINDING 295 check 1: which head did the local law load? ---")
    print(f"single head across all legs : {local['single_head_across_legs']}")
    print(f"head sha256                 : {str(local['head_sha256'])[:16]}")
    print(f"head bytes                  : {local['head_bytes']}")
    print(f"matches declared manifest   : {local['safetensors_bytes_match_manifest']}")
    print(f"is pinned bf16 head         : {local['is_pinned_bf16_head']}")

    print("\n--- check 2: measured head price, harness=local, gated ---")
    print(
        "draft step slope pinned/declared : "
        f"{price['draft_step_slope_ratio_pinned_over_declared']:.3f}"
    )
    print(
        "head bytes  pinned/declared      : "
        f"{price['head_bytes_ratio_pinned_over_declared']:.3f}"
    )
    print(
        "verify slope pinned/declared     : "
        f"{price['verify_slope_ratio_pinned_over_declared']:.3f}  (null control)"
    )
    print(
        "round slope inflation if pinned  : "
        f"{price['round_slope_inflation_pinned_over_declared']:.3f}"
    )
    print(
        "head share of a round, declared  : "
        f"{price['head_share_of_round_declared']:.1%}"
    )
    print(
        "head share of a round, pinned    : "
        f"{price['head_share_of_round_pinned']:.1%}"
    )

    print("\n--- check 3: is the ranked width sweep head-homogeneous? ---")
    print(f"cells {strat['n_cells']}, distinct heads {strat['n_distinct_heads']}")
    print(f"cells per head : {strat['cells_per_head']}")
    print(f"mean width by head : "
          + ", ".join(f"{h}={w:.2f}" for h, w in strat["mean_width_by_head"].items()))
    for key in ("single_head_linear", "single_head_build_speed_controlled",
                "single_head_quadratic",
                "head_fixed_effect_quadratic"):
        if key in strat:
            print(f"{key}: {json.dumps(strat[key], indent=1)}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    path = os.path.join(ARTIFACTS, "e157_head_confound.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
