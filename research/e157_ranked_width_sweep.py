"""E157 R0: measure the ranked marginal row price within one prompt.

The eight prompts of a single receipt cannot separate a drafted row from an
accepted token, because the schedule moves them together at Pearson 0.98. The
board can separate them. Different submissions ran different schedules on the
same eight prompts and the same ranked host, so verify width varies WITHIN a
prompt across submissions.

The round count must be exact for this to work. It is exact whenever the
published constraints admit one multiple of the reduced denominator of
`effective_mean_draft_len`:

  * `rounds >= 512 / (1 + edl)` because accepted cannot exceed drafted;
  * `rounds >= non_drafting_round_count`;
  * `rounds * (1 - edl) <= non_drafting_round_count` when `edl < 1`, because a
    drafting round drafts at least one token;
  * `rounds <= 512`.

`drama` is the only prompt whose uniquely recovered cells span the full width
range, and `plutarch` supplies a near-width-one cell for almost every
submission, which indexes that submission's build speed. Two steps follow:

  1. index each submission by its plutarch round cost at width about one;
  2. regress the drama round cost on width, controlling for that index.

Sign convention in words: a positive width slope means one more verified row
costs more time. Every number is harness=ranked.

    python3 research/e157_ranked_width_sweep.py
"""
from __future__ import annotations

import json
import math
import os
from fractions import Fraction

CACHE = "/tmp/yukon-board/full.json"
DECODE_TOKENS = 512
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
INDEX_PROMPT = "plutarch"
SWEEP_PROMPT = "drama"
LOCAL_A_US = 23421.07
LOCAL_B_US = 15708.50


def load_rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["submissions"]


def exact_rounds(edl: float, non_drafting: int) -> int | None:
    frac = Fraction(edl).limit_denominator(DECODE_TOKENS)
    if abs(float(frac) - edl) > 1e-12:
        return None
    step = frac.denominator
    low = max(
        math.ceil(DECODE_TOKENS / (1.0 + edl) - 1e-9), int(non_drafting), step
    )
    candidates = [r for r in range(step, DECODE_TOKENS + 1, step) if r >= low]
    if edl < 1.0 and non_drafting > 0:
        candidates = [
            r for r in candidates if r * (1.0 - edl) <= non_drafting + 1e-9
        ]
    return candidates[0] if len(candidates) == 1 else None


def cell(entry: dict) -> dict | None:
    edl = entry.get("effective_mean_draft_len")
    non_drafting = entry.get("non_drafting_round_count")
    spt = entry.get("mtp_seconds_per_token_mean")
    prefill = entry.get("prefill_seconds_per_token")
    if None in (edl, non_drafting, spt, prefill):
        return None
    rounds = exact_rounds(edl, non_drafting)
    if rounds is None:
        return None
    return {
        "width": edl + 1.0,
        "rounds": rounds,
        "clean_us_per_round": 1e6 * DECODE_TOKENS * (spt - prefill) / rounds,
        "raw_ratio": entry.get("raw_ratio_of_means"),
        "serial_spt": entry.get("serial_seconds_per_token_mean"),
    }


def lstsq(design: list[list[float]], target: list[float]):
    n = len(design[0])
    normal = [
        [sum(r[i] * r[j] for r in design) for j in range(n)]
        + [sum(r[i] * v for r, v in zip(design, target))]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(normal[r][col]))
        if abs(normal[pivot][col]) < 1e-30:
            return None
        normal[col], normal[pivot] = normal[pivot], normal[col]
        for r in range(n):
            if r == col:
                continue
            factor = normal[r][col] / normal[col][col]
            for k in range(col, n + 1):
                normal[r][k] -= factor * normal[col][k]
    beta = [normal[i][n] / normal[i][i] for i in range(n)]
    predicted = [sum(b * v for b, v in zip(beta, r)) for r in design]
    residual = [y - q for y, q in zip(target, predicted)]
    dof = max(len(target) - n, 1)
    sigma2 = sum(r * r for r in residual) / dof
    # standard errors from the inverse normal matrix diagonal
    inverse_diagonal = []
    for i in range(n):
        unit = [1.0 if k == i else 0.0 for k in range(n)]
        work = [
            [sum(r[a] * r[b] for r in design) for b in range(n)] + [unit[a]]
            for a in range(n)
        ]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
            work[col], work[pivot] = work[pivot], work[col]
            for r in range(n):
                if r == col:
                    continue
                factor = work[r][col] / work[col][col]
                for k in range(col, n + 1):
                    work[r][k] -= factor * work[col][k]
        inverse_diagonal.append(work[i][n] / work[i][i])
    stderr = [math.sqrt(max(sigma2 * d, 0.0)) for d in inverse_diagonal]
    total = sum((y - sum(target) / len(target)) ** 2 for y in target)
    r_squared = 1.0 - sum(r * r for r in residual) / total if total > 0 else 0.0
    return beta, stderr, r_squared, residual


def collect_cells(rows: list[dict]) -> dict[str, list[dict]]:
    """Every uniquely recovered cell on the board, grouped by prompt."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        metrics = (row.get("officialMetrics") or {}).get("per_prompt")
        if not metrics:
            continue
        for entry in metrics:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
            built = cell(entry) if name else None
            if built:
                built["id8"] = row["id"][:8]
                built["tokens_per_round"] = DECODE_TOKENS / built["rounds"]
                built["rejected_per_round"] = (
                    built["width"] - built["tokens_per_round"]
                )
                grouped.setdefault(name, []).append(built)
    return grouped


def full_sweep_fits(rows: list[dict], out: dict) -> dict:
    """Two fits the eight-prompt cross-section cannot do.

    The first uses every uniquely recovered `drama` cell, which reaches width
    8.9, so the slope is not an extrapolation from shallow rounds. The second
    adds accepted tokens per round beside width; within one prompt across
    submissions the two no longer move together, so the drafted row price and
    the accepted token price separate.
    """
    grouped = collect_cells(rows)
    sweep = sorted(grouped.get(SWEEP_PROMPT, []), key=lambda c: c["width"])
    out["sweep_prompt_all_cells"] = sweep
    fits = {}
    if len(sweep) < 4:
        return fits

    widths = [c["width"] for c in sweep]
    costs = [c["clean_us_per_round"] for c in sweep]
    tokens = [c["tokens_per_round"] for c in sweep]

    print(f"\nall uniquely recovered {SWEEP_PROMPT} cells: {len(sweep)}")
    print(f"{'id8':9s} {'width':>6s} {'rounds':>7s} {'tok/rnd':>8s} {'us/rnd':>9s}")
    for c in sweep:
        print(
            f"{c['id8']:9s} {c['width']:6.3f} {c['rounds']:7d} "
            f"{c['tokens_per_round']:8.3f} {c['clean_us_per_round']:9.0f}"
        )

    plain = lstsq([[1.0, w] for w in widths], costs)
    if plain:
        beta, stderr, r_squared, _ = plain
        fits["sweep_prompt_width_only"] = {
            "n": len(sweep),
            "width_range": [min(widths), max(widths)],
            "fixed_us": beta[0],
            "marginal_us_per_row": beta[1],
            "marginal_stderr": stderr[1],
            "r_squared": r_squared,
        }
        print(
            f"\nwidth only, full range {min(widths):.2f}-{max(widths):.2f}:  "
            f"a {beta[0]:8.0f}  b {beta[1]:7.0f} +- {stderr[1]:5.0f}  "
            f"R2 {r_squared:.4f}"
        )

    quadratic = lstsq([[1.0, w, w * w] for w in widths], costs)
    if quadratic:
        beta, stderr, r_squared, _ = quadratic
        fits["sweep_prompt_width_quadratic"] = {
            "n": len(sweep),
            "beta_us": beta,
            "curvature_stderr": stderr[2],
            "r_squared": r_squared,
            "marginal_us_by_width": {
                str(w): beta[1] + beta[2] * (2.0 * w - 1.0) for w in range(1, 10)
            },
        }
        print(
            f"width quadratic: a {beta[0]:8.0f}  b {beta[1]:7.0f}  "
            f"c {beta[2]:6.0f} +- {stderr[2]:5.0f}  R2 {r_squared:.4f}"
        )
        for w in (1, 3, 5, 7, 8, 9):
            print(
                f"   marginal at width {w}: "
                f"{beta[1] + beta[2] * (2.0 * w - 1.0):7.0f} us"
            )

    # The single deepest cell carries most of the leverage. Report the slope
    # with and without it so the reader can price that dependence.
    trimmed = [c for c in sweep if c["width"] < 5.0]
    trimmed_fit = lstsq(
        [[1.0, c["width"]] for c in trimmed],
        [c["clean_us_per_round"] for c in trimmed],
    )
    if trimmed_fit:
        beta, stderr, r_squared, _ = trimmed_fit
        fits["sweep_prompt_width_only_below_five"] = {
            "n": len(trimmed),
            "width_range": [
                min(c["width"] for c in trimmed),
                max(c["width"] for c in trimmed),
            ],
            "fixed_us": beta[0],
            "marginal_us_per_row": beta[1],
            "marginal_stderr": stderr[1],
            "r_squared": r_squared,
        }
        print(
            f"width only, below 5:  a {beta[0]:8.0f}  b {beta[1]:7.0f} "
            f"+- {stderr[1]:5.0f}  R2 {r_squared:.4f}  n {len(trimmed)}"
        )
    return fits


def main() -> None:
    rows = load_rows()
    records = []
    for row in rows:
        metrics = (row.get("officialMetrics") or {}).get("per_prompt")
        if not metrics:
            continue
        cells = {}
        for entry in metrics:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
            if name in (INDEX_PROMPT, SWEEP_PROMPT):
                built = cell(entry)
                if built:
                    cells[name] = built
        if INDEX_PROMPT in cells and SWEEP_PROMPT in cells:
            records.append(
                {
                    "id8": row["id"][:8],
                    "official_score": row.get("officialScore"),
                    "commit": row.get("submissionCommitSha"),
                    "index_width": cells[INDEX_PROMPT]["width"],
                    "index_us_per_round": cells[INDEX_PROMPT]["clean_us_per_round"],
                    "sweep_width": cells[SWEEP_PROMPT]["width"],
                    "sweep_rounds": cells[SWEEP_PROMPT]["rounds"],
                    "sweep_us_per_round": cells[SWEEP_PROMPT]["clean_us_per_round"],
                }
            )
    records.sort(key=lambda r: r["sweep_width"])

    print(f"paired cells: {len(records)}")
    print(
        f"{'id8':9s} {'score':>10s} {'idx w':>6s} {'idx us/rnd':>11s} "
        f"{'swp w':>6s} {'swp rnds':>9s} {'swp us/rnd':>11s} {'ratio':>7s}"
    )
    for r in records:
        print(
            f"{r['id8']:9s} {r['official_score'] or float('nan'):10.5f} "
            f"{r['index_width']:6.3f} {r['index_us_per_round']:11.0f} "
            f"{r['sweep_width']:6.3f} {r['sweep_rounds']:9d} "
            f"{r['sweep_us_per_round']:11.0f} "
            f"{r['sweep_us_per_round'] / r['index_us_per_round']:7.3f}"
        )

    widths = [r["sweep_width"] for r in records]
    costs = [r["sweep_us_per_round"] for r in records]
    index = [r["index_us_per_round"] for r in records]

    out = {
        "experiment": "e157-r0-ranked-within-prompt-width-sweep",
        "harness": "ranked",
        "official_or_ranked_score": True,
        "index_prompt": INDEX_PROMPT,
        "sweep_prompt": SWEEP_PROMPT,
        "paired_cell_count": len(records),
        "cells": records,
        "fits": {},
    }

    plain = lstsq([[1.0, w] for w in widths], costs)
    if plain:
        beta, stderr, r_squared, _ = plain
        out["fits"]["width_only"] = {
            "fixed_us": beta[0],
            "marginal_us_per_row": beta[1],
            "marginal_stderr": stderr[1],
            "r_squared": r_squared,
        }
        print(
            f"\nwidth only:      a {beta[0]:8.0f}  b {beta[1]:7.0f} "
            f"+- {stderr[1]:5.0f}  R2 {r_squared:.4f}"
        )

    controlled = lstsq([[1.0, w, u] for w, u in zip(widths, index)], costs)
    if controlled:
        beta, stderr, r_squared, _ = controlled
        out["fits"]["width_with_build_speed_index"] = {
            "intercept_us": beta[0],
            "marginal_us_per_row": beta[1],
            "marginal_stderr": stderr[1],
            "index_loading": beta[2],
            "index_stderr": stderr[2],
            "r_squared": r_squared,
        }
        print(
            f"width + index:   c {beta[0]:8.0f}  b {beta[1]:7.0f} "
            f"+- {stderr[1]:5.0f}  index {beta[2]:6.3f} +- {stderr[2]:.3f}  "
            f"R2 {r_squared:.4f}"
        )
        marginal = beta[1]
        # The width-one round cost sets the denominator of the depth price.
        fixed = beta[0] + beta[2] * (sum(index) / len(index))
        out["e157_row_price_us"] = {
            "harness": "ranked",
            "identification": (
                "within prompt across submissions, exact recovered round "
                "counts, build speed controlled by the index prompt"
            ),
            "marginal_us_per_row": marginal,
            "marginal_stderr": stderr[1],
            "width_one_round_us": fixed + marginal,
            "h_marginal_ranked": marginal / (fixed + marginal),
            "local_marginal_us_per_row": LOCAL_B_US,
            "local_h_marginal": LOCAL_B_US / (LOCAL_A_US + LOCAL_B_US),
            "local_over_ranked_marginal": LOCAL_B_US / marginal,
        }
        print(
            f"\nranked marginal row {marginal:.0f} us, width-one round "
            f"{fixed + marginal:.0f} us, h {marginal / (fixed + marginal):.4f}"
        )
        print(
            f"local marginal row {LOCAL_B_US:.0f} us, h "
            f"{LOCAL_B_US / (LOCAL_A_US + LOCAL_B_US):.4f}, "
            f"local over ranked {LOCAL_B_US / marginal:.3f}x"
        )

    out["fits"].update(full_sweep_fits(rows, out))

    out_dir = "research/e157-artifacts"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "e157_ranked_width_sweep.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
