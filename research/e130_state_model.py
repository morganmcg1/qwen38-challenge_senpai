#!/usr/bin/env python3
"""E130-S: what does the runner state actually charge for?

FINDING 150 establishes that the 2026-08-22 board band is a runner state and
measures the offset at 817 us per drafting round. That number assumes the
answer. This script tests the assumption: it regresses the per-prompt candidate
leg offset on five rival covariates and reports which one explains it.

Each model is a single-parameter regression through the origin over the eight
prompts, so they are directly comparable on residual scatter.

    per_drafting_round   x = drafting rounds        fixed cost to enter drafting
    per_round            x = all rounds             fixed cost every round
    multiplicative       x = the leg itself         a uniformly slower machine
    per_draft_step       x = drafting rounds * Mbar cost per proposal-head call
    per_token            x = 512                    a flat per-prompt constant

The serial leg is the control. FINDING 150 claims the state does not touch it,
so no model should fit there and the offset should be null.

harness = ranked.

Usage:
    python3 research/e130_state_model.py --pair d3c491b5 48423d09 \
        --pair 48423d09 cf79f7df \
        --out research/e130-artifacts/rung8-state-model.json
"""

from __future__ import annotations

import argparse
import json
import math

TOKENS = 512

NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ORDER = [
    "plutarch",
    "drama",
    "travel",
    "beagle",
    "republic",
    "essays",
    "medicine",
    "botany",
]

# Proposal-head artifact size and the per-draft-step read, both from FINDING 143.
HEAD_ARTIFACT_MB = 427.0
HEAD_STEP_MB = 323.59


def load_board(path: str) -> dict:
    with open(path) as fh:
        rows = json.load(fh)
    if isinstance(rows, dict):
        rows = rows["submissions"]
    out = {}
    for r in rows:
        om = r.get("officialMetrics") or {}
        if om.get("per_prompt"):
            out[r["id"][:8]] = r
    return out


def prompts(row: dict) -> dict:
    d = {}
    for e in row["officialMetrics"]["per_prompt"]:
        d[NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8])] = e
    return d


def structure(entry: dict) -> dict:
    """Round accounting for one prompt, from the receipt's own fields."""
    mbar = entry["effective_mean_draft_len"]
    rounds = TOKENS / (mbar + 1.0)
    non_drafting = entry.get("non_drafting_round_count")
    drafting = max(0.0, rounds - non_drafting) if non_drafting is not None else float("nan")
    return {
        "mbar": mbar,
        "rounds": rounds,
        "non_drafting_rounds": non_drafting,
        "drafting_rounds": drafting,
        "draft_steps": drafting * mbar,
    }


def fit_origin(xs: list[float], ys: list[float]) -> dict:
    """Least squares through the origin, with scatter reported two ways."""
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return {"slope": float("nan"), "r2": float("nan"), "resid_cv": float("nan")}
    slope = sum(x * y for x, y in zip(xs, ys)) / sxx
    resid = [y - slope * x for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum(y * y for y in ys)
    n = len(xs)
    resid_sd = math.sqrt(ss_res / (n - 1)) if n > 1 else float("nan")
    mean_abs_y = sum(abs(y) for y in ys) / n
    # per-point implied slope, the quantity FINDING 150 averaged
    per_point = [y / x if x else float("nan") for x, y in zip(xs, ys)]
    finite = [v for v in per_point if math.isfinite(v)]
    mean_pp = sum(finite) / len(finite) if finite else float("nan")
    sd_pp = (
        math.sqrt(sum((v - mean_pp) ** 2 for v in finite) / (len(finite) - 1))
        if len(finite) > 1
        else float("nan")
    )
    return {
        "slope": slope,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "resid_sd": resid_sd,
        "resid_frac_of_mean_effect": resid_sd / mean_abs_y if mean_abs_y else float("nan"),
        "per_point_mean": mean_pp,
        "per_point_sd": sd_pp,
        "per_point_cv": abs(sd_pp / mean_pp) if mean_pp else float("nan"),
    }


def analyse_pair(board: dict, a: str, b: str) -> dict:
    """Offset of a relative to b, positive meaning a is slower."""
    pa, pb = prompts(board[a]), prompts(board[b])
    rows = []
    for name in ORDER:
        ea, eb = pa[name], pb[name]
        st = structure(ea)
        cand_delta_s = (
            ea["mtp_seconds_per_token_mean"] - eb["mtp_seconds_per_token_mean"]
        ) * TOKENS
        serial_delta_s = (
            ea["serial_seconds_per_token_mean"] - eb["serial_seconds_per_token_mean"]
        ) * TOKENS
        rows.append(
            {
                "prompt": name,
                **st,
                "cand_a_spt": ea["mtp_seconds_per_token_mean"],
                "cand_b_spt": eb["mtp_seconds_per_token_mean"],
                "cand_delta_pct": 100.0
                * cand_delta_s
                / (eb["mtp_seconds_per_token_mean"] * TOKENS),
                "cand_delta_us": cand_delta_s * 1e6,
                "serial_delta_pct": 100.0
                * serial_delta_s
                / (eb["serial_seconds_per_token_mean"] * TOKENS),
                "serial_delta_us": serial_delta_s * 1e6,
            }
        )

    def covariates(leg: str) -> dict:
        base = "cand_b_spt" if leg == "cand" else None
        return {
            "per_drafting_round": [r["drafting_rounds"] for r in rows],
            "per_round": [r["rounds"] for r in rows],
            "multiplicative": [
                (r[base] * TOKENS if base else prompts(board[b])[r["prompt"]][
                    "serial_seconds_per_token_mean"
                ] * TOKENS)
                for r in rows
            ],
            "per_draft_step": [r["draft_steps"] for r in rows],
            "per_token": [float(TOKENS) for _ in rows],
        }

    out = {"a": a, "b": b, "prompts": rows, "models": {}}
    for leg, key in (("candidate", "cand_delta_us"), ("serial", "serial_delta_us")):
        ys = [r[key] for r in rows]
        legtag = "cand" if leg == "candidate" else "serial"
        fits = {
            name: fit_origin(xs, ys) for name, xs in covariates(legtag).items()
        }
        ranked = sorted(
            (n for n in fits if math.isfinite(fits[n]["resid_frac_of_mean_effect"])),
            key=lambda n: fits[n]["resid_frac_of_mean_effect"],
        )
        mean_pct = sum(
            r["cand_delta_pct" if leg == "candidate" else "serial_delta_pct"]
            for r in rows
        ) / len(rows)
        out["models"][leg] = {
            "mean_delta_pct": mean_pct,
            "fits": fits,
            "best_model": ranked[0] if ranked else None,
            "ranking_by_residual": ranked,
        }

    # Physical reading of the winning candidate-leg model.
    best = out["models"]["candidate"]["best_model"]
    slope_us = out["models"]["candidate"]["fits"][best]["slope"] if best else float("nan")
    if best == "per_drafting_round" and math.isfinite(slope_us) and slope_us > 0:
        out["byte_reading"] = {
            "microseconds_per_drafting_round": slope_us,
            "note": (
                "If the state charges one extra DRAM traversal of a fixed "
                "object on every drafting round, this is the bandwidth that "
                "object would need."
            ),
            "implied_bandwidth_GBps_for_head_artifact": HEAD_ARTIFACT_MB
            / (slope_us * 1e-6)
            / 1000.0,
            "implied_bandwidth_GBps_for_one_draft_step_read": HEAD_STEP_MB
            / (slope_us * 1e-6)
            / 1000.0,
            "head_artifact_MB": HEAD_ARTIFACT_MB,
            "head_step_MB": HEAD_STEP_MB,
        }
    return out


def family_slopes(board: dict, family_path: str, reference: str) -> dict:
    """Fit the per-drafting-round offset of every schedule-family member.

    If the state is real and quantized, these slopes cluster at multiples of one
    state step instead of spreading continuously.
    """
    with open(family_path) as fh:
        family = json.load(fh)
    ref_prompts = prompts(board[reference])
    rows = []
    for member in family["members"]:
        sid = member["id"][:8]
        if sid not in board or sid == reference:
            continue
        pm = prompts(board[sid])
        xs, ys = [], []
        for name in ORDER:
            if name not in pm or name not in ref_prompts:
                break
            st = structure(pm[name])
            if not math.isfinite(st["drafting_rounds"]):
                continue
            xs.append(st["drafting_rounds"])
            ys.append(
                (
                    pm[name]["mtp_seconds_per_token_mean"]
                    - ref_prompts[name]["mtp_seconds_per_token_mean"]
                )
                * TOKENS
                * 1e6
            )
        if len(xs) != len(ORDER):
            continue
        fit = fit_origin(xs, ys)
        rows.append(
            {
                "receipt": sid,
                "solver": board[sid].get("solverUsername"),
                "score": board[sid].get("officialScore"),
                "created": board[sid].get("createdAt"),
                "slope_us_per_drafting_round": fit["slope"],
                "resid_frac_of_mean_effect": fit["resid_frac_of_mean_effect"],
            }
        )
    rows.sort(key=lambda r: r["slope_us_per_drafting_round"])

    # cluster the slopes with a simple gap cut at half a nominal state step
    step_guess = 900.0
    clusters: list[list[dict]] = []
    for r in rows:
        if clusters and (
            r["slope_us_per_drafting_round"]
            - clusters[-1][-1]["slope_us_per_drafting_round"]
            < step_guess / 2.0
        ):
            clusters[-1].append(r)
        else:
            clusters.append([r])
    summary = []
    for i, cl in enumerate(clusters):
        vals = [r["slope_us_per_drafting_round"] for r in cl]
        mean = sum(vals) / len(vals)
        sd = (
            math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            if len(vals) > 1
            else 0.0
        )
        summary.append(
            {
                "cluster": i,
                "n": len(cl),
                "slope_mean_us": mean,
                "slope_sd_us": sd,
                "slope_min_us": min(vals),
                "slope_max_us": max(vals),
                "solvers": sorted({r["solver"] for r in cl}),
                "earliest": min(r["created"] for r in cl),
                "latest": max(r["created"] for r in cl),
            }
        )
    gaps = [
        summary[i + 1]["slope_mean_us"] - summary[i]["slope_mean_us"]
        for i in range(len(summary) - 1)
    ]
    return {
        "reference": reference,
        "n_members_fitted": len(rows),
        "gap_cut_us": step_guess / 2.0,
        "clusters": summary,
        "cluster_gaps_us": gaps,
        "members": rows,
    }


def state_timeline(family: dict, since: str, gap_us: float = 400.0) -> dict:
    """Is the state an epoch the runner entered, or a draw made per run?

    An epoch predicts that every row before the switch sits in the old state and
    every row after it sits in the new one. A per-run draw predicts that the two
    states interleave throughout. The test needs the tree mechanism removed
    first, which is what the per-drafting-round slope does.
    """
    rows = [m for m in family["members"] if m["created"] >= since]
    rows.sort(key=lambda r: r["slope_us_per_drafting_round"])
    clusters: list[list[dict]] = []
    for r in rows:
        if clusters and (
            r["slope_us_per_drafting_round"]
            - clusters[-1][-1]["slope_us_per_drafting_round"]
            < gap_us
        ):
            clusters[-1].append(r)
        else:
            clusters.append([r])

    out = []
    for i, cl in enumerate(clusters):
        vals = [r["slope_us_per_drafting_round"] for r in cl]
        scores = [r["score"] for r in cl]
        mean = sum(vals) / len(vals)
        out.append(
            {
                "cluster": i,
                "n": len(cl),
                "slope_mean_us": mean,
                "slope_sd_us": (
                    math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                    if len(vals) > 1
                    else 0.0
                ),
                "score_mean": sum(scores) / len(scores),
                "score_min": min(scores),
                "score_max": max(scores),
                "n_solvers": len({r["solver"] for r in cl}),
                "earliest": min(r["created"] for r in cl),
                "latest": max(r["created"] for r in cl),
                "receipts": [r["receipt"] for r in cl],
            }
        )

    # A cluster of many different trees carries mechanism spread that can hide a
    # state step inside it. Split each large cluster at its widest internal gap
    # and keep the split only when it separates cleanly.
    for c, cl in zip(out, clusters):
        if len(cl) < 8:
            continue
        vals = [r["slope_us_per_drafting_round"] for r in cl]
        gap, at = max(
            (vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)
        )
        parts = [cl[: at + 1], cl[at + 1 :]]
        stats = []
        for part in parts:
            pv = [r["slope_us_per_drafting_round"] for r in part]
            mean = sum(pv) / len(pv)
            stats.append(
                {
                    "n": len(part),
                    "slope_mean_us": mean,
                    "slope_sd_us": (
                        math.sqrt(sum((v - mean) ** 2 for v in pv) / (len(pv) - 1))
                        if len(pv) > 1
                        else 0.0
                    ),
                    "n_solvers": len({r["solver"] for r in part}),
                    "receipts": [r["receipt"] for r in part],
                }
            )
        widest_sd = max(s["slope_sd_us"] for s in stats)
        merged = sorted(
            [(r["receipt"], "L") for r in parts[0]]
            + [(r["receipt"], "H") for r in parts[1]],
            key=lambda t: next(m["created"] for m in rows if m["receipt"] == t[0]),
        )
        labels = "".join(t[1] for t in merged)
        c["internal_split"] = {
            "gap_us": gap,
            "gap_over_widest_sd": gap / widest_sd if widest_sd else float("inf"),
            "clean": bool(widest_sd and gap > 2.0 * widest_sd),
            "step_us": stats[1]["slope_mean_us"] - stats[0]["slope_mean_us"],
            "parts": stats,
            "time_order": labels,
            "alternations": sum(
                1 for k in range(len(labels) - 1) if labels[k] != labels[k + 1]
            ),
            "max_possible_alternations": len(labels) - 1,
        }

    # Pair clusters whose score bands overlap: same tree generation, different
    # state. The slope difference between such a pair is one state step.
    pairs = []
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            a, b = out[i], out[j]
            overlap = not (a["score_min"] > b["score_max"] or b["score_min"] > a["score_max"])
            near_score = abs(a["score_mean"] - b["score_mean"]) < 0.05
            if overlap or near_score:
                merged = sorted(
                    [(r, "A") for r in a["receipts"]] + [(r, "B") for r in b["receipts"]],
                    key=lambda t: next(
                        m["created"] for m in rows if m["receipt"] == t[0]
                    ),
                )
                labels = [t[1] for t in merged]
                alternations = sum(
                    1 for k in range(len(labels) - 1) if labels[k] != labels[k + 1]
                )
                pairs.append(
                    {
                        "clusters": [i, j],
                        "step_us": b["slope_mean_us"] - a["slope_mean_us"],
                        "score_gap": a["score_mean"] - b["score_mean"],
                        "time_order": "".join(labels),
                        "alternations": alternations,
                        "max_possible_alternations": len(labels) - 1,
                        "epoch_would_predict_alternations": 1,
                    }
                )
    return {
        "since": since,
        "gap_cut_us": gap_us,
        "n_rows": len(rows),
        "clusters": out,
        "same_generation_pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--pair", nargs=2, action="append", required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--family-reference", default="cf79f7df")
    ap.add_argument("--since", default="2026-08-22")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    board = load_board(args.board)
    results = [analyse_pair(board, a, b) for a, b in args.pair]

    art = {
        "harness": "ranked",
        "experiment": "E130-S rung 8",
        "question": "what covariate explains the runner-state offset?",
        "models_tested": [
            "per_drafting_round",
            "per_round",
            "multiplicative",
            "per_draft_step",
            "per_token",
        ],
        "control": "the serial leg, which FINDING 150 says the state does not touch",
        "pairs": results,
        "finding_150_reconciliation": {
            "amends": "research/e130-artifacts/rung7-preregistration.json",
            "what_stays_valid": [
                "The within-tier sd is a within-STATE sd, because the 0.1 % tier "
                "cut is far narrower than the 1.15 % state step. The instrument "
                "calibration and the pre-registered decision thresholds are "
                "therefore unchanged.",
                "The rank diagnostic stands: inside one state the published "
                "median carries no rank information about candidate decode time.",
            ],
            "what_is_withdrawn": [
                "My interim-5 label 'this is one tree measured four times' is "
                "wrong. Speed tier 0 is one RUNNER STATE sampled across at least "
                "three different source trees. The trees differ by hundreds of "
                "lines and still agree to 0.06 %, so those mechanisms are worth "
                "approximately nothing and the tier is a state, not a tree.",
                "Any c derived from a cross-state contrast is invalid. c may be "
                "derived only against a row proven to share the state.",
            ],
        },
    }
    if args.family:
        art["family_quantization"] = family_slopes(
            board, args.family, args.family_reference
        )
        art["state_timeline"] = state_timeline(
            art["family_quantization"], args.since
        )
    with open(args.out, "w") as fh:
        json.dump(art, fh, indent=2)
        fh.write("\n")

    for res in results:
        print(f"\n=== {res['a']} minus {res['b']} ===")
        for leg in ("candidate", "serial"):
            m = res["models"][leg]
            print(f"  {leg} leg, mean offset {m['mean_delta_pct']:+.4f} %")
            for name in m["ranking_by_residual"]:
                f = m["fits"][name]
                print(
                    f"    {name:<20} slope {f['slope']:12.4f}  "
                    f"resid/effect {f['resid_frac_of_mean_effect']:7.4f}  "
                    f"per-point cv {f['per_point_cv']:7.4f}"
                )
            print(f"    best: {m['best_model']}")
        if "byte_reading" in res:
            br = res["byte_reading"]
            print(
                f"  byte reading: {br['microseconds_per_drafting_round']:.1f} us/round "
                f"=> {br['implied_bandwidth_GBps_for_head_artifact']:.0f} GB/s for the "
                f"{br['head_artifact_MB']:.0f} MB head, "
                f"{br['implied_bandwidth_GBps_for_one_draft_step_read']:.0f} GB/s for a "
                f"{br['head_step_MB']:.0f} MB step read"
            )
    if "family_quantization" in art:
        fq = art["family_quantization"]
        print(
            f"\n=== schedule-family quantization, {fq['n_members_fitted']} members "
            f"against {fq['reference']} ==="
        )
        for c in fq["clusters"]:
            print(
                f"  cluster {c['cluster']}  n={c['n']:3d}  "
                f"slope {c['slope_mean_us']:9.1f} +/- {c['slope_sd_us']:7.1f} us  "
                f"range [{c['slope_min_us']:.1f}, {c['slope_max_us']:.1f}]  "
                f"solvers {len(c['solvers'])}  first {c['earliest'][:19]}"
            )
        print(f"  gaps between clusters: "
              f"{', '.join('%.1f' % g for g in fq['cluster_gaps_us'])} us")
    if "state_timeline" in art:
        tl = art["state_timeline"]
        print(f"\n=== state timeline since {tl['since']}, {tl['n_rows']} rows, "
              f"gap cut {tl['gap_cut_us']:.0f} us ===")
        for c in tl["clusters"]:
            print(
                f"  c{c['cluster']}  n={c['n']:3d}  slope {c['slope_mean_us']:8.1f} "
                f"+/- {c['slope_sd_us']:6.1f}  score {c['score_mean']:.4f} "
                f"[{c['score_min']:.4f}, {c['score_max']:.4f}]  "
                f"solvers {c['n_solvers']:2d}  {c['earliest'][11:19]}..{c['latest'][11:19]}"
            )
            sp = c.get("internal_split")
            if sp and sp["clean"]:
                p0, p1 = sp["parts"]
                print(
                    f"      internal split step {sp['step_us']:7.1f} us, gap "
                    f"{sp['gap_us']:.1f} us = {sp['gap_over_widest_sd']:.1f} sd; "
                    f"n {p0['n']}/{p1['n']}, solvers {p0['n_solvers']}/{p1['n_solvers']}, "
                    f"alternations {sp['alternations']}/{sp['max_possible_alternations']}"
                )
                print(f"      time order: {sp['time_order']}")
        for p in tl["same_generation_pairs"]:
            print(
                f"  pair {p['clusters']}: step {p['step_us']:7.1f} us  "
                f"score gap {p['score_gap']:+.4f}  "
                f"alternations {p['alternations']}/{p['max_possible_alternations']} "
                f"(an epoch predicts 1)"
            )
            print(f"    time order: {p['time_order']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
