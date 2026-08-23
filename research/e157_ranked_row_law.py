"""E157 R0: recover the ranked round-cost law from published receipt rows.

THE ROWS-PER-ROUND REGRESSOR EXISTS. The trusted driver writes
`effectiveDraftLengths: rounds.map { $0.draftTokens.count }`
(`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift:295`), so the
published `effective_mean_draft_len` is the mean DRAFTED width per round and
not the accepted count. Verify width is `drafted + 1`, so a receipt publishes
the realised schedule directly, at full precision, with no fitted parameter.

Round counts are recovered, not fitted. `effective_mean_draft_len` is the
rational `drafted_total / rounds`, so the round count is a multiple of the
reduced denominator. Three published constraints prune the multiples:

  * accepted tokens cannot exceed drafted tokens, so
    `rounds >= decode_tokens / (1 + edl)`;
  * `rounds >= non_drafting_round_count`;
  * `rounds <= decode_tokens`.

Where more than one multiple survives, the surviving combination is chosen by
one joint cost fit across the eight prompts. POSITIVE CONTROL: the recovered
tokens per round must reproduce the eight values FINDING 281 published for the
crown receipt, which were derived by a different route.

Two laws are fitted on the same recovered rounds.

  LAW A  clean_us_per_round = a + b * width           (width = edl + 1)
  LAW C  clean_us_per_round = a + b * width + c * width^2

Sign convention in words: a positive `d_star - edl` means the marginal rule
wants to draft DEEPER than the schedule that actually ran; a negative value
means the schedule drafted too deep. Every number is harness=ranked unless the
field name says local.

    python3 research/e157_ranked_row_law.py [RECEIPT_ID8 ...]
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from fractions import Fraction

CACHE = "/tmp/yukon-board/full.json"
DECODE_TOKENS = 512
MAX_DEPTH_CAP = 8
SHIPPED_H = 0.18
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
# E154 R2, harness=local, M4 Pro 48 GiB: clean_round_us = a + b * width.
LOCAL_A_US = 23421.071843078098
LOCAL_B_US = 15708.498145888343
LOCAL_MEAN_WIDTH = 7.358974358974359
LOCAL_ROLLBACK_US = 262.87154073660724
# FINDING 281, harness=ranked, crown receipt ec24d59: tokens per round.
FINDING_281_TOKENS_PER_ROUND = {
    "plutarch": 1.0519,
    "drama": 2.0317,
    "travel": 2.4113,
    "beagle": 4.6545,
    "republic": 5.5054,
    "essays": 5.5628,
    "medicine": 5.6883,
    "botany": 6.3179,
}


def load_rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["submissions"]


def receipt(rows: list[dict], prefix: str) -> dict:
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{prefix}: {len(hits)} matches, need exactly one")
    return hits[0]


def round_candidates(edl: float, non_drafting: int) -> list[int]:
    frac = Fraction(edl).limit_denominator(DECODE_TOKENS)
    if abs(float(frac) - edl) > 1e-12:
        raise SystemExit(f"edl={edl!r} is not a small rational")
    step = frac.denominator
    low = max(
        math.ceil(DECODE_TOKENS / (1.0 + edl) - 1e-9), int(non_drafting), step
    )
    return [r for r in range(step, DECODE_TOKENS + 1, step) if r >= low]


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
    relative = [(y - q) / y for y, q in zip(target, predicted)]
    rmse = math.sqrt(sum(r * r for r in relative) / len(relative))
    return beta, rmse, relative


def design_for(widths: list[float], quadratic: bool) -> list[list[float]]:
    if quadratic:
        return [[1.0, w, w * w] for w in widths]
    return [[1.0, w] for w in widths]


def round_cost_us(beta: list[float], width: float) -> float:
    return sum(b * width ** k for k, b in enumerate(beta))


def marginal_us(beta: list[float], width: float) -> float:
    """Cost of the row that takes the verify batch from `width-1` to `width`."""
    return round_cost_us(beta, width) - round_cost_us(beta, width - 1.0)


def geometric_accepted(p: float, depth: float) -> float:
    whole = int(math.floor(depth))
    frac = depth - whole
    total = sum(p ** k for k in range(1, whole + 1))
    if frac > 0:
        total += frac * p ** (whole + 1)
    return total


def solve_conditional_p(accepted: float, depth: float) -> float:
    if depth <= 0 or accepted <= 0:
        return 0.0
    low, high = 1e-6, 0.9999999
    for _ in range(200):
        mid = 0.5 * (low + high)
        if geometric_accepted(mid, depth) < accepted:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def leg_ratio(p: float, depth: float, beta: list[float], rollback: float) -> float:
    tokens = 1.0 + geometric_accepted(p, depth)
    cost = round_cost_us(beta, depth + 1.0)
    if depth > 0:
        cost += rollback * (1.0 - p ** depth)
    return tokens / cost


def best_depth(p: float, beta: list[float], rollback: float) -> int:
    scores = [leg_ratio(p, float(d), beta, rollback) for d in range(MAX_DEPTH_CAP + 1)]
    return max(range(len(scores)), key=lambda d: scores[d])


def flat_price_depth(p: float, marginal: float, step: float, cap: int) -> int:
    """The shipped `costModelDepth` walk with a flat head-step price."""
    reach, expected, depth = 1.0, 0.0, 0
    for _ in range(cap):
        reach *= p
        threshold = marginal * (1.0 + expected) / (1.0 + depth * step)
        if reach <= threshold:
            break
        expected += reach
        depth += 1
    return depth


def prompt_table(row: dict) -> list[tuple[str, dict]]:
    entries = [
        (PROMPT_NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]), e)
        for e in row["officialMetrics"]["per_prompt"]
    ]
    entries.sort()
    return entries


def fit_over_combinations(entries, quadratic: bool):
    candidates = [
        round_candidates(e["effective_mean_draft_len"], e["non_drafting_round_count"])
        for _, e in entries
    ]
    widths = [e["effective_mean_draft_len"] + 1.0 for _, e in entries]
    decode_us = [
        1e6
        * DECODE_TOKENS
        * (e["mtp_seconds_per_token_mean"] - e["prefill_seconds_per_token"])
        for _, e in entries
    ]
    design = design_for(widths, quadratic)
    results = []
    for combo in itertools.product(*candidates):
        drafted = [
            e["effective_mean_draft_len"] * r for (_, e), r in zip(entries, combo)
        ]
        if any(DECODE_TOKENS - r > d + 1e-6 for r, d in zip(combo, drafted)):
            continue
        per_round = [t / r for t, r in zip(decode_us, combo)]
        fit = lstsq(design, per_round)
        if fit is None:
            continue
        beta, rmse, relative = fit
        if beta[0] <= 0:
            continue
        if any(marginal_us(beta, w) <= 0 for w in widths):
            continue
        results.append((rmse, combo, beta, relative))
    results.sort(key=lambda item: item[0])
    return results


LAW_FAMILY = {
    "A_width": lambda w, t: [1.0, w],
    "C_width_quadratic": lambda w, t: [1.0, w, w * w],
    "D_width_and_tokens": lambda w, t: [1.0, w, t],
    "E_tokens": lambda w, t: [1.0, t],
    "F_tokens_quadratic": lambda w, t: [1.0, t, t * t],
}


def compare_law_family(entries) -> dict:
    """Which regressor explains the ranked round cost: drafted width, or
    accepted tokens? The two move together across the eight prompts, so this
    comparison is the identification test, not a model-selection nicety."""
    candidates = [
        round_candidates(e["effective_mean_draft_len"], e["non_drafting_round_count"])
        for _, e in entries
    ]
    widths = [e["effective_mean_draft_len"] + 1.0 for _, e in entries]
    decode_us = [
        1e6
        * DECODE_TOKENS
        * (e["mtp_seconds_per_token_mean"] - e["prefill_seconds_per_token"])
        for _, e in entries
    ]
    out = {}
    for label, builder in LAW_FAMILY.items():
        best = None
        for combo in itertools.product(*candidates):
            drafted = [
                e["effective_mean_draft_len"] * r for (_, e), r in zip(entries, combo)
            ]
            if any(DECODE_TOKENS - r > d + 1e-6 for r, d in zip(combo, drafted)):
                continue
            tokens = [DECODE_TOKENS / r for r in combo]
            per_round = [u / r for u, r in zip(decode_us, combo)]
            fit = lstsq(
                [builder(w, t) for w, t in zip(widths, tokens)], per_round
            )
            if fit is None or fit[0][0] <= 0:
                continue
            if best is None or fit[1] < best[1]:
                best = (fit[0], fit[1], list(combo))
        beta, rmse, combo = best
        out[label] = {
            "beta_us": beta,
            "relative_rmse": rmse,
            "parameter_count": len(beta),
            "rounds": combo,
            "all_slopes_positive": all(b > 0 for b in beta[1:]),
        }
    return out


def analyse(row: dict) -> dict:
    entries = prompt_table(row)
    names = [n for n, _ in entries]
    widths = [e["effective_mean_draft_len"] + 1.0 for _, e in entries]

    laws = {}
    for label, quadratic in (("A_linear", False), ("C_quadratic", True)):
        results = fit_over_combinations(entries, quadratic)
        if not results:
            raise SystemExit(f"{label}: no admissible round-count combination fits")
        rmse, combo, beta, relative = results[0]
        near = [r for r in results if r[0] < 2.0 * rmse]
        h_at_6 = [marginal_us(r[2], 6.0) / round_cost_us(r[2], 1.0) for r in near]
        laws[label] = {
            "rounds": list(combo),
            "beta_us": beta,
            "relative_rmse": rmse,
            "worst_relative_residual": max(abs(x) for x in relative),
            "residual_by_prompt": dict(zip(names, relative)),
            "admissible_combination_count": len(results),
            "near_optimal_count": len(near),
            "h_marginal_at_width_6_range": [min(h_at_6), max(h_at_6)],
        }

    control = {}
    for (name, _), rounds in zip(entries, laws["A_linear"]["rounds"]):
        recovered = DECODE_TOKENS / rounds
        published = FINDING_281_TOKENS_PER_ROUND[name]
        control[name] = {
            "recovered_tokens_per_round": recovered,
            "finding_281_tokens_per_round_crown": published,
            "relative_difference": (recovered - published) / published,
        }
    control_worst = max(abs(v["relative_difference"]) for v in control.values())

    # Collinearity: can the receipt separate a drafted row from an accepted
    # token? FINDING 281 says the published slope is a blend. Measure it.
    rounds_a = laws["A_linear"]["rounds"]
    tokens = [DECODE_TOKENS / r for r in rounds_a]
    mean_w = sum(widths) / len(widths)
    mean_t = sum(tokens) / len(tokens)
    cov = sum((w - mean_w) * (t - mean_t) for w, t in zip(widths, tokens))
    var_w = sum((w - mean_w) ** 2 for w in widths)
    var_t = sum((t - mean_t) ** 2 for t in tokens)
    correlation = cov / math.sqrt(var_w * var_t)

    out = {
        "id8": row["id"][:8],
        "official_score": row["officialScore"],
        "commit": row["submissionCommitSha"],
        "created_utc": row["createdAt"],
        "harness": "ranked",
        "laws": laws,
        "law_family_comparison": compare_law_family(entries),
        "round_recovery_positive_control": {
            "reference": "FINDING 281 tokens per round, crown receipt ec24d59",
            "worst_relative_difference": control_worst,
            "by_prompt": control,
        },
        "row_versus_token_collinearity": {
            "pearson_width_vs_tokens_per_round": correlation,
            "reading": (
                "a drafted row and an accepted token move together across the "
                "eight prompts, so the receipt bounds the row price instead of "
                "identifying it"
            ),
        },
        "per_prompt": [],
    }

    for (name, entry), rounds in zip(entries, rounds_a):
        edl = entry["effective_mean_draft_len"]
        accepted = DECODE_TOKENS - rounds
        drafted = round(edl * rounds)
        conditional = solve_conditional_p(DECODE_TOKENS / rounds - 1.0, edl)
        record = {
            "prompt": name,
            "harness": "ranked",
            "edl_drafted_mean": edl,
            "verify_width_mean": edl + 1.0,
            "rounds": rounds,
            "drafted_total": drafted,
            "accepted_total": accepted,
            "tokens_per_round": DECODE_TOKENS / rounds,
            "pooled_accept_rate": accepted / drafted if drafted else 0.0,
            "non_drafting_rounds": entry["non_drafting_round_count"],
            "conditional_p_fitted": conditional,
            "clean_us_per_round": 1e6
            * DECODE_TOKENS
            * (
                entry["mtp_seconds_per_token_mean"]
                - entry["prefill_seconds_per_token"]
            )
            / rounds,
            "mtp_spt": entry["mtp_seconds_per_token_mean"],
            "serial_spt": entry["serial_seconds_per_token_mean"],
            "prefill_spt": entry["prefill_seconds_per_token"],
            "raw_ratio": entry["raw_ratio_of_means"],
        }
        for label in ("A_linear", "C_quadratic"):
            beta = laws[label]["beta_us"]
            rollback = LOCAL_ROLLBACK_US * (marginal_us(beta, edl + 1.0) / LOCAL_B_US)
            depth = best_depth(conditional, beta, rollback)
            gain = (
                leg_ratio(conditional, float(depth), beta, rollback)
                / leg_ratio(conditional, edl, beta, rollback)
                - 1.0
            )
            record[f"d_star_{label}"] = depth
            record[f"d_star_minus_edl_{label}"] = depth - edl
            record[f"modelled_gain_pct_{label}"] = 100.0 * gain
            record[f"h_marginal_at_run_width_{label}"] = marginal_us(
                beta, edl + 1.0
            ) / round_cost_us(beta, 1.0)
        record["d_star_shipped_flat_price"] = flat_price_depth(
            conditional, SHIPPED_H, SHIPPED_H, 7
        )
        out["per_prompt"].append(record)
    return out


def report(entry: dict) -> None:
    print(f"\n=== {entry['id8']}  score {entry['official_score']:.8f} ===")
    control = entry["round_recovery_positive_control"]
    print(
        "round recovery vs FINDING 281 tokens/round: worst relative difference "
        f"{100 * control['worst_relative_difference']:.3f} %"
    )
    for label, law in entry["laws"].items():
        beta = " ".join(f"{b:+.1f}" for b in law["beta_us"])
        lo, hi = law["h_marginal_at_width_6_range"]
        print(
            f"{label:12s} rmse {100 * law['relative_rmse']:6.3f} %  "
            f"worst {100 * law['worst_relative_residual']:6.3f} %  "
            f"beta_us [{beta}]  h(width 6) {lo:.4f}..{hi:.4f} over "
            f"{law['near_optimal_count']} near-optimal recoveries"
        )
    for label, law in entry["law_family_comparison"].items():
        beta = " ".join(f"{b:+.1f}" for b in law["beta_us"])
        print(
            f"  family {label:20s} k={law['parameter_count']} "
            f"rmse {100 * law['relative_rmse']:6.3f} %  "
            f"slopes positive {str(law['all_slopes_positive']):5s} [{beta}]"
        )
    print(
        "width vs tokens/round pearson "
        f"{entry['row_versus_token_collinearity']['pearson_width_vs_tokens_per_round']:.5f}"
    )
    print(
        f"{'prompt':9s} {'rounds':>6s} {'width':>6s} {'us/rnd':>8s} {'p cond':>7s} "
        f"{'hA':>6s} {'hC':>6s} {'d*A':>4s} {'d*C':>4s} {'d*ship':>6s} "
        f"{'gainA%':>7s} {'gainC%':>7s}"
    )
    for p in entry["per_prompt"]:
        print(
            f"{p['prompt']:9s} {p['rounds']:6d} {p['verify_width_mean']:6.3f} "
            f"{p['clean_us_per_round']:8.0f} {p['conditional_p_fitted']:7.4f} "
            f"{p['h_marginal_at_run_width_A_linear']:6.4f} "
            f"{p['h_marginal_at_run_width_C_quadratic']:6.4f} "
            f"{p['d_star_A_linear']:4d} {p['d_star_C_quadratic']:4d} "
            f"{p['d_star_shipped_flat_price']:6d} "
            f"{p['modelled_gain_pct_A_linear']:+7.2f} "
            f"{p['modelled_gain_pct_C_quadratic']:+7.2f}"
        )


def main() -> None:
    ids = sys.argv[1:] or ["0cf1637e", "75a21a4"]
    rows = load_rows()
    receipts = [analyse(receipt(rows, i)) for i in ids]
    for entry in receipts:
        report(entry)

    head = receipts[0]
    beta_a = head["laws"]["A_linear"]["beta_us"]
    beta_c = head["laws"]["C_quadratic"]["beta_us"]
    verdict = {
        "e157_row_price_us": {
            "harness": "ranked",
            "law_A_linear_marginal_us": beta_a[1],
            "law_C_quadratic_marginal_us_at_width": {
                str(w): marginal_us(beta_c, float(w)) for w in range(1, 9)
            },
            "local_marginal_us_at_mean_width_7_36": LOCAL_B_US,
            "local_over_ranked_at_matched_width_7": LOCAL_B_US
            / marginal_us(beta_c, 7.0),
            "local_over_ranked_width_averaged": LOCAL_B_US / beta_a[1],
            "identified": False,
            "reason": (
                "the eight ranked prompts vary verify width and accepted "
                "tokens together, so the marginal row price is bounded by the "
                "law family and not point identified"
            ),
        },
        "e157_marginal_rule_breakeven": {
            "harness": "ranked",
            "shipped_flat_head_step_cost_ratio": SHIPPED_H,
            "law_A_h_marginal_width_averaged": beta_a[1] / (beta_a[0] + beta_a[1]),
            "law_C_h_marginal_by_width": {
                str(w): marginal_us(beta_c, float(w)) / round_cost_us(beta_c, 1.0)
                for w in range(1, 9)
            },
            "sign_of_depth_error_depends_on_law": True,
        },
    }
    out = {
        "experiment": "e157-r0-ranked-row-law",
        "harness": "ranked",
        "source": "Yukon list endpoint officialMetrics.per_prompt, read only",
        "receipts": receipts,
        "verdict": verdict,
    }
    out_dir = "research/e157-artifacts"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "e157_ranked_row_law.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print("\nlaw C marginal us by verify width:")
    for w in range(1, 9):
        print(
            f"  width {w}: {marginal_us(beta_c, float(w)):8.0f} us   "
            f"h {marginal_us(beta_c, float(w)) / round_cost_us(beta_c, 1.0):.4f}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
