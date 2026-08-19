#!/usr/bin/env python3
"""E34: the ranked operating point, and what the width wall costs there.

Deliverables
------------
(a) Head reconciliation + the acceptance gap: recompute the declared-head tree
    digest locally and decompose ranked vs local acceptance.
(b) Re-cost the E25 r3 T(M) curve at the RANKED operating point instead of the
    local one, using exactly reconstructed ranked round counts.
(c) F / S / residuals from the absolute curve, and the cap-4 bracket.
(d) Whether `sdpaWidthWallDepthCap = 5` actually binds at ranked acceptance.

Everything here is offline: ranked telemetry (cached REST rows) plus the
forced-depth curves E25 r3 already measured. No GPU slot is used.

    python3 research/e34_ranked_operating_point.py --out research/e34-ranked-operating-point.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
from fractions import Fraction

import e34_cost_model as cm

REPO = pathlib.Path(__file__).resolve().parent.parent
TELEMETRY = REPO / ".mlxfast-private/ranked-telemetry.json"
CONTRACT = REPO / "fixtures/qwen3_8_27b_mtp_track.json"
MANIFEST = REPO / "mtp-head.manifest.json"
BOARD_TOP = "0cd0a6b4"
CENTRAL = ("beagle", "medicine")
DECODE_TOKENS = 512

# E25 r3 forced-depth curves, parent clock, mean ms per round; key is the
# verify width M = forced depth + 1. Same host, same session, same head; the
# two builds differ only in the dispatch table's IPG at M=5 and M=9.
CURVE_POST_E27 = {
    1: 68.07211760816902, 2: 71.36809877280531, 3: 78.82930860890971,
    4: 91.7211723955054, 5: 108.34561419068721, 6: 143.87177260576095,
    7: 156.54867119259305, 8: 170.32108455896378,
}
CURVE_PRE_E27 = {
    1: 68.61592777844133, 2: 71.60541055531337, 3: 78.91581688092384,
    4: 91.88500360438698, 5: 132.25655148672566, 6: 144.10291799115043,
    7: 156.78791111111112, 8: 170.13350669642856,
}
CURVE_SEM = {
    1: 0.1353630658705251, 2: 0.12784639381350452, 3: 0.16300056001430155,
    4: 0.17416452069907035, 5: 0.17557489478569693, 6: 0.1606515798266025,
    7: 0.15879082542901737, 8: 0.1635352964988384,
}
# Pass counts behind the pre-E27 curve, CONFIRMED by source read at `474c750`
# (the last accepted competitor snapshot, and the state in which the shipped
# depth cap was chosen): IPG {3:3, 4:4, 5:3, 6:3, 7:4, 8:4, 9:3}.
PASSES_PRE_E27 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}
IPG_PRE_E27 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# Provenance of the constant under test. `sdpaWidthWallDepthCap` was 4 in the
# organizer tree; ranked row 12b1c699 (audreyt, accepted 2.92521, 2026-08-16)
# raised it to 5 inside a six-way composite, reasoning "lifting cap to 5 behind
# same segmentation should unlock one more committed token per round". Under
# the table in force then, M=5 and M=6 both cost 2 weight passes, so that
# change crossed NO pass boundary and was free in the dominant cost term.
CAP_PROVENANCE = {
    "organizer_value": 4,
    "raised_to": 5,
    "raised_by_submission": "12b1c699",
    "raised_by_solver": "audreyt",
    "raised_by_score": 2.92520777238747,
    "raised_at": "2026-08-16",
    "composite_arms": 6,
    "table_when_raised": "474c750",
    "table_now": "0207de6 (E27, ours)",
}
LOCAL_ACCEPT_P = [0.6925972396486826, 0.5839572192513369, 0.5076586433260394,
                  0.41899441340782123, 0.38596491228070173]


def extended_curve(curve: dict[int, float] = None,
                   passes: dict[int, int] = None) -> dict[int, float]:
    """The measured curve plus an EXTRAPOLATED M=9 point.

    E25 r3 forced depths 0..7, so M=9 was never timed, but the shipped policy
    reaches it through `segmentedVerifyDepthCap = 8`. Clamping T(9) to T(8)
    would silently make the deepest rounds free and bias every no-wall arm in
    the direction the experiment is trying to test, so M=9 is extrapolated from
    the quadratic+step fit instead. It is the only modelled point in the curve.
    """
    curve = dict(curve or CURVE_POST_E27)
    if 9 not in curve:
        curve[9] = cm.fit_cost_model(curve, quadratic=True, passes=passes).predict(9)
    return curve


def board_curve() -> dict[int, float]:
    """Local T(M) under the pass structure the RANKED rows actually ran.

    Every scored row on the board predates E27, so its verify kernel streams
    the weights `PASSES_PRE_E27` times, not `dispatch_ipg()` times. Calibrating
    the ranked ledger against the post-E27 curve would credit the board with a
    speedup it never had and would silently move E27's own gain into the depth
    cap's column. E25 r3 measured both builds on one host in one session, so
    the board-matching curve is available as data rather than as a model.
    """
    return extended_curve(CURVE_PRE_E27, passes=PASSES_PRE_E27)


# --------------------------------------------------------------------------
# (a) head provenance
# --------------------------------------------------------------------------
def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_ledger(root: pathlib.Path) -> dict:
    """The ranked workflow's head-tree rule: sha256 over '<sha>  <rel>\\n' lines."""
    if not root.is_dir():
        return {"root": str(root), "present": False}
    entries = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name == "README.md":
                continue
            path = pathlib.Path(dirpath) / name
            entries.append((str(path.relative_to(root)), path))
    entries.sort()
    lines, files, total = [], {}, 0
    for rel, path in entries:
        digest, size = file_sha256(path), path.stat().st_size
        files[rel] = {"sha256": digest, "bytes": size}
        total += size
        lines.append("%s  %s\n" % (digest, rel))
    return {
        "root": str(root),
        "present": True,
        "file_count": len(entries),
        "total_bytes": total,
        "tree_digest": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "files": files,
    }


def head_reconciliation() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    cache = pathlib.Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
    trees = {name: tree_ledger(cache / name) for name in
             ("mtp-head-declared-q2q4", "mtp-head-declared-q2q4-run",
              "mtp-head-declared", "mtp-head")}
    ran = trees["mtp-head-declared-q2q4"]
    verified = (ran.get("tree_digest") == manifest["sha256"]
                and ran.get("total_bytes") == manifest["bytes"])
    return {
        "manifest": manifest,
        "trees": trees,
        "run_tree_matches_manifest": verified,
        "file_digest_of_model_safetensors":
            ran.get("files", {}).get("model.safetensors", {}).get("sha256"),
        "explanation": (
            "d038fd41 is the sha256 OF THE FILE model.safetensors; 559b24eb is the "
            "sha256 OF THE ONE-LINE TREE MANIFEST over that file. Same artifact, "
            "two digests of different objects."),
    }


# --------------------------------------------------------------------------
# ranked telemetry: exact reconstruction
# --------------------------------------------------------------------------
def prompt_names() -> dict[str, str]:
    import re
    out: dict[str, str] = {}
    contract = json.loads(CONTRACT.read_text())

    def walk(node):
        if isinstance(node, dict):
            if "sha256" in node:
                hit = re.search(r"pool-([a-z_]+)\.json", node.get("r2_path") or "")
                if hit:
                    out[node["sha256"]] = hit.group(1)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(contract)
    return out


def ranked_rows(prefix: str = BOARD_TOP) -> dict:
    subs = json.loads(TELEMETRY.read_text())["submissions"]
    row = next(s for s in subs if str(s["id"]).startswith(prefix))
    names = prompt_names()
    metrics = row["officialMetrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    out = {}
    for entry in metrics["per_prompt"]:
        name = names.get(entry["prompt_sha256"], entry["prompt_sha256"][:8])
        out[name] = entry
    return {"id": row["id"], "solver": row["solverUsername"],
            "score": row["officialScore"], "metrics": metrics, "per_prompt": out}


def all_rows() -> list[dict]:
    out = []
    for row in json.loads(TELEMETRY.read_text())["submissions"]:
        metrics = row.get("officialMetrics")
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        if isinstance(metrics, dict) and "per_prompt" in metrics:
            out.append(dict(row, officialMetrics=metrics))
    return out


def depth_echo_check() -> dict:
    """Settle whether `mtp_depth` reports realised depth or the offered ceiling.

    The advisor's sizing of (d) turned on reading `mtp_depth = 8` as realised
    depth. It is not: the field is `(8, 8)` on EVERY scored row, including the
    deliberate depth-0 serial controls whose realised draft length is exactly
    zero on all eight prompts. A field that is 8 when the run provably drafts
    nothing is an echo of the parent's offered ceiling.
    """
    rows = all_rows()
    pairs, controls, deepest = set(), [], []
    for row in rows:
        metrics = row["officialMetrics"]
        if row.get("officialScore") is None:
            continue
        pairs.add((metrics.get("mtp_depth"), metrics.get("mtp_max_draft_depth")))
        lens = [e["effective_mean_draft_len"] for e in metrics["per_prompt"]]
        record = {"id": row["id"][:8], "solver": row["solverUsername"],
                  "score": row["officialScore"],
                  "mtp_depth": metrics.get("mtp_depth"),
                  "mtp_max_draft_depth": metrics.get("mtp_max_draft_depth"),
                  "mean_draft_len": sum(lens) / len(lens),
                  "max_prompt_draft_len": max(lens)}
        if max(lens) == 0:
            controls.append(record)
        deepest.append(record)
    deepest.sort(key=lambda r: -r["max_prompt_draft_len"])
    return {
        "scored_rows": sum(1 for r in rows if r.get("officialScore") is not None),
        "distinct_depth_pairs": sorted(pairs),
        "is_config_echo": len(pairs) == 1,
        "zero_draft_controls": controls[:4],
        "zero_draft_control_count": len(controls),
        "deepest_rows": deepest[:4],
        "realised_draft_len_range": [min(r["mean_draft_len"] for r in deepest),
                                     max(r["mean_draft_len"] for r in deepest)],
    }


def ranked_depth_evidence(prompt: str = "beagle", backbone: str = "b53e4991") -> dict:
    """Ranked rows that drafted a NON-default amount on one central prompt.

    The counterfactual claims a shallower schedule is faster per token. The
    board cannot confirm it directly - no ranked row has ever run this
    dispatch table - but it can falsify the premise: if rows that draft more
    on `prompt` were also faster per token, the per-row price would not be the
    binding cost and the whole experiment would be pointless.
    """
    names = prompt_names()
    recs = []
    for row in all_rows():
        metrics = row["officialMetrics"]
        if not str(metrics.get("qwen_mtp_weights_hash", "")).startswith(backbone):
            continue
        if row.get("officialScore") is None:
            continue
        entries = {names.get(e["prompt_sha256"], e["prompt_sha256"][:8]): e
                   for e in metrics["per_prompt"]}
        entry = entries.get(prompt)
        if entry is None:
            continue
        rec = reconstruct(entry)["candidates"]
        best = rec[0] if rec else None
        recs.append({
            "id": row["id"][:8], "solver": row["solverUsername"],
            "status": row["status"], "score": row["officialScore"],
            "n": entry["effective_mean_draft_len"],
            "ratio": entry["raw_ratio_of_means"],
            "ms_per_token": 1000.0 * entry["mtp_seconds_per_token_mean"],
            "accept_rate": best["accept_rate"] if best else float("nan"),
            "tokens_per_round": best["tokens_per_round"] if best else float("nan"),
            "round_ms": best["round_ms"] if best else float("nan"),
        })
    recs.sort(key=lambda r: r["n"])
    default = statistics.mode([round(r["n"], 4) for r in recs])
    on_default = [r for r in recs if abs(r["n"] - default) < 1e-4]
    deeper = [r for r in recs if r["n"] > default + 1e-4]
    shallower = [r for r in recs if r["n"] < default - 1e-4]
    best_default = max(on_default, key=lambda r: r["ratio"])

    def summarise(group):
        if not group:
            return None
        rates = [r["accept_rate"] for r in group if r["accept_rate"] == r["accept_rate"]]
        return {"count": len(group),
                "best_ratio": max(r["ratio"] for r in group),
                "best_ms_per_token": min(r["ms_per_token"] for r in group),
                "median_accept_rate": statistics.median(rates) if rates else float("nan"),
                "rows": sorted(group, key=lambda r: -r["ratio"])[:6]}

    shallow = summarise(shallower)
    default_rate = statistics.median([r["accept_rate"] for r in on_default
                                      if r["accept_rate"] == r["accept_rate"]])
    return {
        "prompt": prompt, "backbone": backbone, "rows": len(recs),
        "default_n": default, "default_row_count": len(on_default),
        "default_median_accept_rate": default_rate,
        "best_on_default": best_default,
        "deeper_than_default": summarise(deeper),
        "shallower_than_default": shallow,
        "deeper_ever_beat_default_ratio":
            bool(deeper) and max(r["ratio"] for r in deeper) > best_default["ratio"],
        # The board's shallow rows are the WRONG control for a width cap. A cap
        # removes rows while acceptance per offered row is unchanged; these rows
        # are shallow because a weaker proposal head gets rejected sooner, which
        # loses tokens without saving a weight pass. If their accept rate is
        # below the default's, they cannot be read as evidence about capping.
        "shallow_rows_are_a_valid_cap_control":
            bool(shallow) and shallow["median_accept_rate"] >= default_rate,
    }


def identity_check(profile: dict, alpha: float = 0.99) -> dict:
    """Run the advisor's `R = (1 + a*n)/(1 + h*n)` identity, and say what it proves.

    It closes to machine precision on every prompt, but that is not evidence:
    with `a` fixed, `h` is DEFINED by `R` and `n`, so one equation determines
    the one remaining unknown and residual zero is arithmetic, not validation.
    What the reconstruction does test is the assumption hidden in the numerator.
    `1 + a*n` is meant to be tokens committed per round, and that quantity is
    known exactly from the ledger: `512 / rounds`. Comparing the two shows
    whether `a = 0.99` describes this run at all.
    """
    out = {}
    for name, entry in profile["prompts"].items():
        n = entry["n_draft_len"]
        ratio = entry["ratio"]
        h_bar = ((1.0 + alpha * n) / ratio - 1.0) / n if n else float("nan")
        replay = (1.0 + alpha * n) / (1.0 + h_bar * n) if n else float("nan")
        exact_tokens = entry["tokens_per_round"]
        out[name] = {
            "n": n, "ratio": ratio, "h_bar": h_bar,
            "replayed_ratio": replay,
            "replay_residual": replay - ratio,
            "assumed_tokens_per_round": 1.0 + alpha * n,
            "exact_tokens_per_round": exact_tokens,
            "tokens_per_round_error": (1.0 + alpha * n) - exact_tokens,
            "implied_alpha": (exact_tokens - 1.0) / n if n else float("nan"),
        }
    return {
        "alpha": alpha,
        "prompts": out,
        "max_abs_replay_residual": max(abs(v["replay_residual"]) for v in out.values()
                                       if v["replay_residual"] == v["replay_residual"]),
        "max_abs_tokens_per_round_error": max(abs(v["tokens_per_round_error"])
                                              for v in out.values()),
        "verdict": "identity closes by construction; alpha = 0.99 is not this "
                   "run's accept rate, so h_bar absorbs the numerator error and "
                   "is not a per-row cost ratio",
    }


def interp_local(width: float, curve: dict[int, float]) -> float:
    """Linear interpolation of the measured local curve at a fractional width."""
    lo = max(w for w in curve if w <= width) if any(w <= width for w in curve) else min(curve)
    hi = min(w for w in curve if w >= width) if any(w >= width for w in curve) else max(curve)
    if lo == hi:
        return curve[lo]
    frac = (width - lo) / (hi - lo)
    return curve[lo] + frac * (curve[hi] - curve[lo])


def reconstruct(entry: dict) -> dict:
    """Recover the INTEGER round ledger behind one ranked prompt.

    `effective_mean_draft_len` is a ratio of two integers - drafts PROPOSED
    over rounds - so its reduced denominator is the round count up to an
    integer multiple. Arithmetic pins most of the multiple:
    `rounds + accepted = 512`, `0 <= accepted <= proposed`, and a drafting
    round proposes at least one row. Where more than one multiple survives,
    only one puts the prompt's measured round time on the same cost curve as
    the rest of the submission, because round time is a monotone function of
    verify width on fixed hardware.
    """
    n = entry["effective_mean_draft_len"]
    frac = Fraction(n).limit_denominator(DECODE_TOKENS)
    candidates = []
    for mult in range(1, 9):
        rounds = frac.denominator * mult
        proposed = frac.numerator * mult
        accepted = DECODE_TOKENS - rounds
        if rounds > DECODE_TOKENS or accepted < 0 or accepted > proposed:
            continue
        if entry["non_drafting_round_count"] > rounds:
            continue
        drafting = rounds - entry["non_drafting_round_count"]
        if drafting > 0 and proposed < drafting:
            continue
        candidates.append({
            "rounds": rounds, "drafts_proposed": proposed, "drafts_accepted": accepted,
            "accept_rate": accepted / proposed if proposed else float("nan"),
            "round_ms": 1000.0 * entry["mtp_seconds_per_token_mean"] * DECODE_TOKENS / rounds,
            "mean_width": 1.0 + n,
            "tokens_per_round": DECODE_TOKENS / rounds,
            "multiple": mult,
        })
    return {"candidates": candidates}


def ranked_profile(prefix: str = BOARD_TOP) -> dict:
    """Exact per-prompt ledger for one ranked row, with the multiple resolved."""
    row = ranked_rows(prefix)
    raw = {}
    for name, entry in row["per_prompt"].items():
        rec = reconstruct(entry)
        for cand in rec["candidates"]:
            cand.update({
                "ratio": entry["raw_ratio_of_means"],
                "n_draft_len": entry["effective_mean_draft_len"],
                "serial_ms_per_token": 1000.0 * entry["serial_seconds_per_token_mean"],
                "mtp_ms_per_token": 1000.0 * entry["mtp_seconds_per_token_mean"],
                "non_drafting_rounds": entry["non_drafting_round_count"],
            })
        raw[name] = rec["candidates"]

    # Host-transfer scale from the prompts whose ledger is already unique.
    anchors = [(c[0]["mean_width"], c[0]["round_ms"]) for c in raw.values() if len(c) == 1]
    num = sum(interp_local(w, CURVE_POST_E27) * t for w, t in anchors)
    den = sum(interp_local(w, CURVE_POST_E27) ** 2 for w in (a[0] for a in anchors))
    scale = num / den
    out, selection = {}, {}
    for name, cands in raw.items():
        scored = sorted(
            cands,
            key=lambda c: abs(c["round_ms"] - scale * interp_local(c["mean_width"], CURVE_POST_E27)))
        best = scored[0]
        out[name] = dict(best)
        out[name]["alternatives"] = len(cands)
        selection[name] = {
            "chosen_multiple": best["multiple"],
            "candidates": [{"multiple": c["multiple"], "rounds": c["rounds"],
                            "accept_rate": c["accept_rate"], "round_ms": c["round_ms"],
                            "transfer_residual_ms":
                                c["round_ms"] - scale * interp_local(c["mean_width"], CURVE_POST_E27)}
                           for c in cands],
        }
    order = sorted(out, key=lambda k: out[k]["ratio"])
    by_width = sorted(out, key=lambda k: out[k]["mean_width"])
    monotone = all(out[by_width[i]]["round_ms"] <= out[by_width[i + 1]]["round_ms"] + 1e-9
                   for i in range(len(by_width) - 1))
    return {"id": row["id"], "solver": row["solver"], "score": row["score"],
            "prompts": out, "ratio_order": order,
            "host_transfer_scale": scale,
            "ledger_selection": selection,
            "width_time_monotone": monotone}


# --------------------------------------------------------------------------
# (d) does the wall bind: calibrate the policy to the ranked ledger
# --------------------------------------------------------------------------
def simulate(profile: list[float], sim: cm.PolicySim, rounds: int,
             seed: int = 0, offered: int = 8, margin_mean: float | None = None) -> dict:
    """Run the shipped policy against a per-position acceptance profile.

    `margin_mean` is the mean of an exponential draw for the pending primary's
    target top-2 gap, which drives the policy's confidence clamp. Modelling it
    as independent of that round's acceptance is a simplification: in the real
    decode the two are positively correlated, so this understates how often a
    short round is also a low-acceptance round.
    """
    rng = random.Random(seed)
    state = sim.new_state()
    widths: dict[int, int] = {}
    depth_accept: dict[tuple[int, int], int] = {}
    proposed = accepted_total = 0
    non_drafting = 0
    for _ in range(rounds):
        margin = rng.expovariate(1.0 / margin_mean) if margin_mean else None
        depth, _cap = sim.choose_depth(state, offered=offered, margin=margin)
        accepted = 0
        for i in range(depth):
            q = profile[min(i, len(profile) - 1)]
            if rng.random() < q:
                accepted += 1
            else:
                break
        width = depth + 1
        widths[width] = widths.get(width, 0) + 1
        depth_accept[(depth, accepted)] = depth_accept.get((depth, accepted), 0) + 1
        proposed += depth
        accepted_total += accepted
        if depth == 0:
            non_drafting += 1
        sim.record(state, accepted=accepted, drafted=depth)
    return {
        "widths": widths,
        "depth_accept": depth_accept,
        "mean_draft_len": proposed / rounds,
        "accept_rate": accepted_total / proposed if proposed else float("nan"),
        "tokens_per_round": 1.0 + accepted_total / rounds,
        "non_drafting_rounds": non_drafting,
        "mean_width": sum(w * c for w, c in widths.items()) / rounds,
        "mean_passes": sum(cm.weight_passes(w) * c for w, c in widths.items()) / rounds,
    }


def _summarise(runs: list[dict], rounds: int, trials: int, cost) -> dict:
    widths: dict[int, float] = {}
    for run in runs:
        for width, count in run["widths"].items():
            widths[width] = widths.get(width, 0.0) + count / (trials * rounds)
    widths = dict(sorted(widths.items()))
    return {
        "width_distribution": widths,
        "mean_width": sum(w * p for w, p in widths.items()),
        "mean_passes": sum(cm.weight_passes(w) * p for w, p in widths.items()),
        "tokens_per_round": statistics.mean(r["tokens_per_round"] for r in runs),
        "mean_draft_len": statistics.mean(r["mean_draft_len"] for r in runs),
        "accept_rate": statistics.mean(r["accept_rate"] for r in runs),
        "non_drafting_fraction": statistics.mean(r["non_drafting_rounds"] for r in runs) / rounds,
        "round_ms": sum(p * cost(w) for w, p in widths.items()) if cost else None,
        "fraction_at_M6": widths.get(6, 0.0),
        "fraction_ge_M6": sum(p for w, p in widths.items() if w >= 6),
        "fraction_at_M9": widths.get(9, 0.0),
    }


def calibrate(target_n: float, target_rate: float, target_round_ms: float,
              rounds: int, cost, *, sim: cm.PolicySim | None = None,
              trials: int = 24) -> dict:
    """Recover the round-level behaviour behind one ranked prompt's ledger.

    Three parameters - the acceptance profile `q_i = min(0.999, q0*gamma^i)`
    and the mean target top-2 margin that drives the confidence clamp - are
    fixed by three measured quantities: drafts per round, accept rate, and
    round time. The width distribution is then an OUTPUT, not an assumption.
    """
    sim = sim or cm.PolicySim()

    def run_set(q0, gamma, margin_mean, seeds):
        profile = [min(0.999, q0 * gamma ** i) for i in range(sim.max_depth)]
        return [simulate(profile, sim, rounds, seed=s, margin_mean=margin_mean)
                for s in range(seeds)]

    def fit_profile(margin_mean, seeds_coarse=4, seeds_fine=8):
        def loss_of(q0, gamma, seeds):
            runs = run_set(q0, gamma, margin_mean, seeds)
            n_hat = statistics.mean(r["mean_draft_len"] for r in runs)
            rate_hat = statistics.mean(r["accept_rate"] for r in runs)
            return (((n_hat - target_n) / target_n) ** 2
                    + ((rate_hat - target_rate) / max(target_rate, 1e-6)) ** 2)

        best = min(((loss_of(q0, g, seeds_coarse), q0, g)
                    for q0 in [0.30 + 0.035 * i for i in range(20)]
                    for g in [0.80 + 0.02 * i for i in range(11)]), key=lambda t: t[0])
        _l, q0c, gc = best
        best = min(((loss_of(q0, g, seeds_fine), q0, g)
                    for q0 in [max(0.05, min(0.999, q0c + 0.007 * (i - 5))) for i in range(11)]
                    for g in [max(0.50, min(1.02, gc + 0.004 * (i - 5))) for i in range(11)]),
                   key=lambda t: t[0])
        return best

    trace = []
    best = None
    for margin_mean in (None, 12.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0):
        loss, q0, gamma = fit_profile(margin_mean)
        runs = run_set(q0, gamma, margin_mean, trials)
        summary = _summarise(runs, rounds, trials, cost)
        gap = abs(summary["round_ms"] - target_round_ms) / target_round_ms
        total = loss + gap * gap
        trace.append({"margin_mean": margin_mean, "profile_loss": loss,
                      "q0": q0, "gamma": gamma, "total_loss": loss + gap * gap,
                      "round_ms": summary["round_ms"], "round_ms_gap": gap,
                      "mean_draft_len": summary["mean_draft_len"],
                      "accept_rate": summary["accept_rate"]})
        if best is None or total < best[0]:
            best = (total, margin_mean, q0, gamma, summary)

    _total, margin_mean, q0, gamma, summary = best
    profile = [min(0.999, q0 * gamma ** i) for i in range(sim.max_depth)]
    out = {
        "profile": profile, "q0": q0, "gamma": gamma, "margin_mean": margin_mean,
        "target_mean_draft_len": target_n, "target_accept_rate": target_rate,
        "target_round_ms": target_round_ms,
        "search_trace": trace,
    }
    out.update(summary)
    return out


# --------------------------------------------------------------------------
# (b)/(c) cost model on the ranked host
# --------------------------------------------------------------------------
def transfer_cost(scale: float, curve: dict[int, float] = None):
    """Ranked-host T(M) as the local measured curve scaled by one constant.

    The only assumption is that the ranked host runs the SAME cost SHAPE, which
    is testable: one free parameter has to reproduce eight prompts' round times.
    """
    curve = curve or extended_curve()
    return lambda width: scale * curve[min(max(width, min(curve)), max(curve))]


def fit_transfer_scale(profile: dict, calib: dict[str, dict],
                       curve: dict[int, float] = None) -> dict:
    """Least-squares scale between the local curve and ranked round times."""
    curve = curve or extended_curve()
    num = den = 0.0
    predicted_unit = {}
    for name, calibration in calib.items():
        unit = sum(p * curve[w] for w, p in calibration["width_distribution"].items())
        predicted_unit[name] = unit
        measured = profile["prompts"][name]["round_ms"]
        num += unit * measured
        den += unit * unit
    scale = num / den
    residuals = {name: profile["prompts"][name]["round_ms"] - scale * unit
                 for name, unit in predicted_unit.items()}
    measured = [profile["prompts"][n]["round_ms"] for n in predicted_unit]
    mean = sum(measured) / len(measured)
    ss_tot = sum((m - mean) ** 2 for m in measured)
    ss_res = sum(r * r for r in residuals.values())
    return {"scale": scale, "residuals_ms": residuals,
            "max_abs_residual_ms": max(abs(r) for r in residuals.values()),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def fit_ranked_cost(profile: dict, calib: dict[str, dict], *, quadratic: bool = False) -> dict:
    """Fit `T(M) = A + r*E[M] (+ g*E[M^2]) + S*E[passes]` across the 8 prompts.

    Each prompt contributes one equation because a round time averages
    linearly over that prompt's realised width distribution. This is the
    ranked host's OWN cost model, fitted without borrowing the local curve.

    The design uses `PASSES_PRE_E27` because that is the pass structure the
    board rows ran. `model` re-prices the SAME fitted `A, r, g, S` under our
    post-E27 table, so the counterfactual is evaluated on our kernel while its
    coefficients come from ranked measurements.
    """
    design, target, names = [], [], []
    for name, calibration in calib.items():
        widths = calibration["width_distribution"]
        e_m = sum(w * p for w, p in widths.items())
        e_m2 = sum(w * w * p for w, p in widths.items())
        e_pass = sum(PASSES_PRE_E27[w] * p for w, p in widths.items())
        row = [1.0, e_m] + ([e_m2] if quadratic else []) + [e_pass]
        design.append(row)
        target.append(profile["prompts"][name]["round_ms"])
        names.append(name)
    beta = cm._lstsq(design, target)
    idx = 2
    per_row2 = beta[idx] if quadratic else 0.0
    if quadratic:
        idx += 1
    suffix = "quad" if quadratic else "linear"
    model = cm.StepCostModel(intercept=beta[0], per_row=beta[1], per_pass=beta[idx],
                             per_row2=per_row2, table=cm.dispatch_ipg(),
                             name="ranked_" + suffix + "_our_kernel")
    board_model = cm.StepCostModel(intercept=beta[0], per_row=beta[1], per_pass=beta[idx],
                                   per_row2=per_row2, passes=PASSES_PRE_E27,
                                   name="ranked_" + suffix + "_board_kernel")
    residuals = {}
    for name, row, measured in zip(names, design, target):
        predicted = sum(b * x for b, x in zip(beta, row))
        residuals[name] = measured - predicted
    mean = sum(target) / len(target)
    ss_tot = sum((t - mean) ** 2 for t in target)
    ss_res = sum(r * r for r in residuals.values())
    return {
        "model": model, "board_model": board_model,
        "intercept_ms": beta[0], "per_row_ms": beta[1],
        "per_row2_ms": per_row2, "per_weight_pass_ms": beta[idx],
        "residuals_ms": residuals,
        "max_abs_residual_ms": max(abs(r) for r in residuals.values()),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def max_entropy_widths(mean_width: float, round_ms: float, cost,
                       widths=range(1, 10)) -> dict:
    """Least-committal width distribution matching E[M] and E[T(M)].

    The simulated distributions come from a policy reimplementation, so they
    inherit every modelling choice in `PolicySim`. This is the assumption-light
    control: of ALL distributions on 1..9 consistent with the two measured
    moments, it picks the one with maximum entropy. If the two disagree about
    how much mass sits at M >= 6, the wall verdict is a policy artefact; if
    they agree, the verdict survives without the policy model.
    """
    widths = list(widths)
    unit = [cost(w) for w in widths]

    def dist(l1, l2):
        raw = [math.exp(min(700.0, l1 * w + l2 * t)) for w, t in zip(widths, unit)]
        total = sum(raw)
        return [r / total for r in raw]

    def moments(l1, l2):
        p = dist(l1, l2)
        return (sum(w * q for w, q in zip(widths, p)),
                sum(t * q for t, q in zip(unit, p)), p)

    l1 = l2 = 0.0
    for _ in range(4000):  # Newton on the two dual variables
        m1, m2, p = moments(l1, l2)
        g = [m1 - mean_width, m2 - round_ms]
        if max(abs(g[0]) / max(mean_width, 1e-9), abs(g[1]) / max(round_ms, 1e-9)) < 1e-10:
            break
        v11 = sum(w * w * q for w, q in zip(widths, p)) - m1 * m1
        v12 = sum(w * t * q for w, t, q in zip(widths, unit, p)) - m1 * m2
        v22 = sum(t * t * q for t, q in zip(unit, p)) - m2 * m2
        det = v11 * v22 - v12 * v12
        if abs(det) < 1e-18:
            break
        step = [(v22 * g[0] - v12 * g[1]) / det, (-v12 * g[0] + v11 * g[1]) / det]
        damp = min(1.0, 1.0 / max(1e-9, max(abs(step[0]), abs(step[1])) * 4.0))
        l1 -= damp * step[0]
        l2 -= damp * step[1]
    m1, m2, p = moments(l1, l2)
    return {
        "width_distribution": {w: q for w, q in zip(widths, p)},
        "mean_width": m1, "round_ms": m2,
        "target_mean_width": mean_width, "target_round_ms": round_ms,
        "fraction_ge_M6": sum(q for w, q in zip(widths, p) if w >= 6),
        "fraction_at_M6": p[widths.index(6)] if 6 in widths else 0.0,
        "converged": (abs(m1 - mean_width) < 1e-6 * max(1.0, mean_width)
                      and abs(m2 - round_ms) < 1e-6 * max(1.0, round_ms)),
    }


# Depth-cap arms as `(sdpaWidthWallDepthCap, segmentedVerifyDepthCap)`.
# Capping the width wall ALONE does not keep a run inside the single-pass
# region: two consecutive full-accept rounds swap in `segmentedVerifyDepthCap`,
# so the escape hatch has to be capped too. `w4_s4` is the only arm that is
# genuinely one weight pass per round.
CAP_ARMS = {
    "w3_s3": (3, 3),   # below the single-pass top: closed negative, shape only
    "w4_s4": (4, 4),   # strictly single-pass, M <= 5
    "w4_s8": (4, 8),   # wall at 4, escape hatch untouched
    "w5_s5": (5, 5),   # shipped wall, no escape to M = 9
    "w5_s8": (5, 8),   # SHIPPED
    "w8_s8": (8, 8),   # no wall at all
}


def counterfactual(name: str, profile: dict, calib: dict, cost, *,
                   arm: str, trials: int = 24) -> dict:
    """Predicted ranked ratio for one prompt under a different depth-cap arm."""
    width_cap, segmented_cap = CAP_ARMS[arm]
    sim = cm.PolicySim(width_wall_cap=width_cap, segmented_cap=segmented_cap)
    rounds = profile["prompts"][name]["rounds"]
    runs = [simulate(calib["profile"], sim, rounds, seed=s,
                     margin_mean=calib["margin_mean"]) for s in range(trials)]
    summary = _summarise(runs, rounds, trials, cost)
    ms_per_token = summary["round_ms"] / summary["tokens_per_round"]
    serial = profile["prompts"][name]["serial_ms_per_token"]
    summary.update({
        "arm": arm,
        "width_wall_depth_cap": width_cap,
        "segmented_verify_depth_cap": segmented_cap,
        "ms_per_token": ms_per_token,
        "predicted_ratio": serial / ms_per_token,
    })
    return summary


def sensitivity(profile: dict, calib: dict, cost, *, names=CENTRAL,
                tolerance: float = 0.05, trials: int = 16) -> dict:
    """How much of the predicted gain survives the acceptance profile being wrong.

    Three measured numbers cannot uniquely pin three policy parameters, so the
    calibration has a near-degenerate direction. Every `(q0, gamma, margin)`
    triple in the search trace that still reproduces drafts-per-round, accept
    rate and round time to within `tolerance` is an equally admissible `p`.
    The reported quantity is the RELATIVE gain of the best arm over the shipped
    arm, because that ratio is what the counterfactual actually claims and it
    cancels the parts of the cost model common to both arms.
    """
    out = {}
    for name in names:
        entry = profile["prompts"][name]
        admissible = []
        for row in calib[name]["search_trace"]:
            n_err = abs(row["mean_draft_len"] - entry["n_draft_len"]) / entry["n_draft_len"]
            r_err = abs(row["accept_rate"] - entry["accept_rate"]) / entry["accept_rate"]
            if max(n_err, r_err, row["round_ms_gap"]) <= tolerance:
                admissible.append(row)
        arms = {}
        for row in admissible:
            p = [min(0.999, row["q0"] * row["gamma"] ** i) for i in range(8)]
            variant = dict(calib[name], profile=p, margin_mean=row["margin_mean"])
            best = counterfactual(name, profile, variant, cost, arm="w4_s4", trials=trials)
            ship = counterfactual(name, profile, variant, cost, arm="w5_s8", trials=trials)
            arms[repr((row["q0"], row["gamma"], row["margin_mean"]))] = {
                "q0": row["q0"], "gamma": row["gamma"], "margin_mean": row["margin_mean"],
                "mean_draft_len": row["mean_draft_len"], "accept_rate": row["accept_rate"],
                "round_ms": row["round_ms"],
                "shipped_ratio": ship["predicted_ratio"],
                "best_ratio": best["predicted_ratio"],
                "relative_gain": best["predicted_ratio"] / ship["predicted_ratio"] - 1.0,
                "shipped_fraction_ge_M6": ship["fraction_ge_M6"],
            }
        gains = [a["relative_gain"] for a in arms.values()]
        out[name] = {
            "admissible_count": len(admissible),
            "tolerance": tolerance,
            "variants": arms,
            "relative_gain_min": min(gains) if gains else float("nan"),
            "relative_gain_max": max(gains) if gains else float("nan"),
            "relative_gain_median": statistics.median(gains) if gains else float("nan"),
        }
    return out


def local_counters(path: str = "research/e25r2-timed.json") -> dict:
    """The trusted parent's realised-depth counters from an existing local run.

    The advisor asked for a fresh timed run to read
    `effective_max_draft_len`.  E34 is under a zero-GPU constraint, and the
    counter is already on disk, so this reads it instead.  The point of the
    table is the operating-point gap: the local fixture's realised depth does
    not reach the width wall on either leg, so no local run can decide a
    question about ranked rounds that sit two rows deeper.
    """
    src = REPO / path
    doc = json.loads(src.read_text())
    legs: dict[str, dict] = {}
    for name, entry in doc.get("per_prompt", {}).items():
        for leg in ("base", "candidate"):
            counters = (entry.get(leg) or {}).get("counters") or {}
            if not counters:
                continue
            legs.setdefault(leg, {})[name] = {
                "effective_max_draft_len": counters.get("effective_max_draft_len"),
                "effective_mean_draft_len": counters.get("effective_mean_draft_len"),
                "non_drafting_round_count": counters.get("non_drafting_round_count"),
                "accepted_draft_rate": counters.get("accepted_draft_rate"),
                "round_count": counters.get("round_count"),
                "replayed_rounds": counters.get("verify_block_replayed_round_count"),
            }
    summary = {}
    for leg, prompts in legs.items():
        maxima = [v["effective_max_draft_len"] for v in prompts.values()]
        means = [v["effective_mean_draft_len"] for v in prompts.values()]
        summary[leg] = {
            "max_draft_len_range": [min(maxima), max(maxima)],
            "max_width_M_reached": max(maxima) + 1,
            "mean_draft_len_range": [min(means), max(means)],
            "mean_width_M_range": [min(means) + 1, max(means) + 1],
            "reaches_shipped_wall_M6": max(maxima) + 1 >= 6,
        }
    return {
        "source": path,
        "per_prompt": legs,
        "summary": summary,
        "ranked_mean_width_range": None,  # filled by main() from the ledger
    }


def order_statistics(values: dict[str, float]) -> dict:
    """Published-score rule: mean of the two central order statistics of 8."""
    ranked = sorted(values.items(), key=lambda kv: kv[1])
    lo, hi = ranked[3], ranked[4]
    return {
        "median": (lo[1] + hi[1]) / 2.0,
        "central_prompts": [lo[0], hi[0]],
        "order": [name for name, _ in ranked],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e34-ranked-operating-point.json")
    ap.add_argument("--trials", type=int, default=24)
    args = ap.parse_args()

    table = cm.dispatch_ipg()
    single_pass_top = max(w for w in range(1, 10) if cm.weight_passes(w, table) == 1)
    single_pass_top_pre = max(w for w in range(1, 10) if PASSES_PRE_E27[w] == 1)
    result: dict = {
        "dispatch": {
            "ipg": {str(k): v for k, v in sorted(table.items())},
            "weight_passes": {str(w): cm.weight_passes(w, table) for w in range(1, 10)},
            "passes_per_row": {str(w): cm.rows_per_pass(w, table) for w in range(1, 10)},
            "source": str(cm.DISPATCH_SOURCE.relative_to(REPO)),
            "ipg_pre_e27": {str(k): v for k, v in sorted(IPG_PRE_E27.items())},
            "weight_passes_pre_e27": {str(w): PASSES_PRE_E27[w] for w in range(1, 10)},
            "single_pass_top_width_now": single_pass_top,
            "single_pass_top_width_pre_e27": single_pass_top_pre,
            "single_pass_top_depth_now": single_pass_top - 1,
            "single_pass_top_depth_pre_e27": single_pass_top_pre - 1,
        },
        "cap_provenance": dict(
            CAP_PROVENANCE,
            passes_at_M5_when_raised=PASSES_PRE_E27[5],
            passes_at_M6_when_raised=PASSES_PRE_E27[6],
            crossed_a_pass_boundary_when_raised=PASSES_PRE_E27[6] != PASSES_PRE_E27[5],
            passes_at_M5_now=cm.weight_passes(5, table),
            passes_at_M6_now=cm.weight_passes(6, table),
            crosses_a_pass_boundary_now=cm.weight_passes(6, table) != cm.weight_passes(5, table),
        ),
        "head": head_reconciliation(),
        "depth_echo": depth_echo_check(),
    }

    # (c) the absolute curve
    step = cm.fit_cost_model(CURVE_POST_E27, table=table)
    smooth = cm.fit_cost_model(CURVE_POST_E27, use_passes=False, table=table)
    quad_step = cm.fit_cost_model(CURVE_POST_E27, quadratic=True, table=table)
    quad_smooth = cm.fit_cost_model(CURVE_POST_E27, quadratic=True, use_passes=False, table=table)
    pre_step = cm.fit_cost_model(CURVE_PRE_E27, passes=PASSES_PRE_E27, name="pre_linear+step")
    result["local_curve"] = {
        "post_e27_ms": CURVE_POST_E27,
        "pre_e27_ms": CURVE_PRE_E27,
        "sem_ms": CURVE_SEM,
        "passes_post": {str(w): cm.weight_passes(w, table) for w in CURVE_POST_E27},
        "passes_pre": {str(w): PASSES_PRE_E27[w] for w in CURVE_PRE_E27},
        "increments_post_ms": {str(w): CURVE_POST_E27[w] - CURVE_POST_E27[w - 1]
                               for w in sorted(CURVE_POST_E27) if w - 1 in CURVE_POST_E27},
        "causal_pass_cost_ms": CURVE_PRE_E27[5] - CURVE_POST_E27[5],
        "fits": {name: model.as_dict() for name, model in
                 [("linear+step", step), ("linear+smooth", smooth),
                  ("quad+step", quad_step), ("quad+smooth", quad_smooth),
                  ("pre_linear+step", pre_step)]},
    }

    # ranked reconstruction
    profile = ranked_profile()
    result["ranked"] = profile
    result["identity_check"] = identity_check(profile)
    result["ranked_depth_evidence"] = {p: ranked_depth_evidence(p) for p in CENTRAL}

    # (d) calibrate every prompt against the ranked ledger. The cost model used
    # for calibration is the local curve scaled by one constant; the scale is
    # bootstrapped from the ledger, then refitted once from the calibrated width
    # distributions so it no longer depends on the mean-width interpolation.
    def calibrate_all(cost):
        return {name: calibrate(entry["n_draft_len"], entry["accept_rate"],
                                entry["round_ms"], entry["rounds"], cost,
                                trials=args.trials)
                for name, entry in profile["prompts"].items()}

    board = board_curve()
    scale0 = profile["host_transfer_scale"]
    calib = calibrate_all(transfer_cost(scale0, board))
    transfer = fit_transfer_scale(profile, calib, board)
    calib = calibrate_all(transfer_cost(transfer["scale"], board))
    transfer_final = fit_transfer_scale(profile, calib, board)
    result["calibration"] = calib
    result["host_transfer"] = {
        "bootstrap_scale": scale0,
        "refit_scale": transfer["scale"],
        "final": transfer_final,
        "board_curve_ms": board,
        "our_curve_ms": extended_curve(),
    }

    ranked_linear = fit_ranked_cost(profile, calib)
    ranked_quad = fit_ranked_cost(profile, calib, quadratic=True)
    kappa = transfer_final["scale"]
    scaled = transfer_cost(kappa)                 # our kernel
    scaled_board = transfer_cost(kappa, board)    # the kernel the board ran
    result["ranked_cost_fit"] = {
        "linear": {k: v for k, v in ranked_linear.items() if not k.endswith("model")},
        "quadratic": {k: v for k, v in ranked_quad.items() if not k.endswith("model")},
        "predicted_T_linear_our_kernel": {str(w): ranked_linear["model"].predict(w)
                                          for w in range(1, 10)},
        "predicted_T_linear_board_kernel": {str(w): ranked_linear["board_model"].predict(w)
                                            for w in range(1, 10)},
        "predicted_T_scaled_local_our_kernel": {str(w): scaled(w) for w in range(1, 10)},
        "predicted_T_scaled_local_board_kernel": {str(w): scaled_board(w) for w in range(1, 10)},
    }

    # (d) control: the same wall question answered without the policy model.
    result["max_entropy"] = {
        name: max_entropy_widths(calib[name]["mean_width"],
                                 profile["prompts"][name]["round_ms"], scaled_board)
        for name in profile["prompts"]
    }

    # (b)/(c) counterfactual arms. Each cost model appears twice: once on the
    # kernel the board ran, which must REPRODUCE the measured row, and once on
    # our post-E27 kernel, which is what a candidate would actually execute.
    # The difference between the two shipped-arm numbers is E27's own gain and
    # must not be attributed to the depth cap.
    models = {
        "scaled_local": scaled,
        "ranked_linear": ranked_linear["model"],
        "ranked_quadratic": ranked_quad["model"],
    }
    board_models = {
        "scaled_local": scaled_board,
        "ranked_linear": ranked_linear["board_model"],
        "ranked_quadratic": ranked_quad["board_model"],
    }
    caps = {}
    for name in profile["prompts"]:
        caps[name] = {
            label: {arm: counterfactual(name, profile, calib[name], cost,
                                        arm=arm, trials=args.trials)
                    for arm in CAP_ARMS}
            for label, cost in models.items()
        }
    result["counterfactual"] = caps

    def central_pair(label: str, arm: str) -> float:
        return statistics.mean(caps[name][label][arm]["predicted_ratio"] for name in CENTRAL)

    # Board-kernel replay: shipped policy on the kernel the board actually ran.
    # This is the ONLY arm with a measured truth value, so it sets the interval.
    replay = {}
    for label, cost in board_models.items():
        replay[label] = statistics.mean(
            counterfactual(name, profile, calib[name], cost, arm="w5_s8",
                           trials=args.trials)["predicted_ratio"]
            for name in CENTRAL)
    result["board_kernel_replay"] = replay

    # The published score is the median of ALL EIGHT prompts, so an arm that
    # lifts beagle and medicine could push them out of the central pair and
    # make the assignment's central-pair metric an overestimate. Re-rank every
    # arm to check rather than assume; which prompts are central is not
    # invariant across arms.
    board_replay_ratios = {
        label: {name: counterfactual(name, profile, calib[name], cost, arm="w5_s8",
                                     trials=args.trials)["predicted_ratio"]
                for name in profile["prompts"]}
        for label, cost in board_models.items()
    }
    result["score_order_statistics"] = {
        "measured": order_statistics(
            {n: e["ratio"] for n, e in profile["prompts"].items()}),
        "board_kernel_replay_shipped": {
            label: order_statistics(vals) for label, vals in board_replay_ratios.items()},
        "our_kernel": {
            label: {arm: order_statistics(
                {n: caps[n][label][arm]["predicted_ratio"] for n in profile["prompts"]})
                for arm in CAP_ARMS}
            for label in models},
    }

    measured_central = statistics.mean(profile["prompts"][n]["ratio"] for n in CENTRAL)
    modelled = {label: {arm: central_pair(label, arm) for arm in CAP_ARMS}
                for label in models}
    replay_error = {label: replay[label] - measured_central for label in models}
    # E27 alone, with the depth policy left exactly as shipped.
    e27_only = {label: modelled[label]["w5_s8"] - replay[label] for label in models}
    cap_only = {label: modelled[label]["w4_s4"] - modelled[label]["w5_s8"] for label in models}
    best_arm = {label: max(modelled[label], key=lambda a: modelled[label][a]) for label in models}
    best = {label: modelled[label][best_arm[label]] for label in models}
    at_best = statistics.mean(best.values())
    spread = max(best.values()) - min(best.values())
    err = max(abs(v) for v in replay_error.values())
    result["score_prediction"] = {
        "measured_central_pair": measured_central,
        "board_top_score": profile["score"],
        "board_kernel_replay": replay,
        "modelled_our_kernel": modelled,
        "shipped_replay_error": replay_error,
        "decomposition": {"e27_kernel_only": e27_only, "depth_cap_only": cap_only},
        "best_arm": best_arm,
        "predicted_ranked_central_pair_at_best_cap": at_best,
        "interval": [at_best - err - spread / 2.0, at_best + err + spread / 2.0],
        "interval_basis": "max |board-kernel replay error| across cost models, widened "
                          "by half the across-model spread at each model's own best arm",
    }

    median_arms = result["score_order_statistics"]["our_kernel"]
    best_median_arm = {label: max(median_arms[label],
                                  key=lambda a: median_arms[label][a]["median"])
                       for label in models}
    median_at_best = statistics.mean(
        median_arms[label][best_median_arm[label]]["median"] for label in models)
    median_replay_err = max(
        abs(result["score_order_statistics"]["board_kernel_replay_shipped"][label]["median"]
            - result["score_order_statistics"]["measured"]["median"])
        for label in models)
    central_pair_metric = result["score_prediction"][
        "predicted_ranked_central_pair_at_best_cap"]
    result["score_prediction"]["published_median"] = {
        "measured": result["score_order_statistics"]["measured"]["median"],
        "best_arm": best_median_arm,
        "predicted_at_best_arm": median_at_best,
        "board_kernel_replay_error": median_replay_err,
        "central_pair_minus_median": central_pair_metric - median_at_best,
        "central_pair_is_still_the_median_at_best_arm":
            all(median_arms[label][best_median_arm[label]]["central_prompts"] == list(CENTRAL)
                or sorted(median_arms[label][best_median_arm[label]]["central_prompts"])
                == sorted(CENTRAL)
                for label in models),
    }

    result["sensitivity"] = sensitivity(profile, calib, scaled, trials=max(8, args.trials // 2))

    counters = local_counters()
    counters["ranked_mean_width_range"] = [
        min(e["mean_width"] for e in profile["prompts"].values()),
        max(e["mean_width"] for e in profile["prompts"].values()),
    ]
    result["local_counters"] = counters

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(result, indent=1, default=str))
    print("wrote %s" % out)
    print(json.dumps({
        "head_verified": result["head"]["run_tree_matches_manifest"],
        "depth_echo": {k: result["depth_echo"][k] for k in
                       ("distinct_depth_pairs", "is_config_echo",
                        "zero_draft_control_count", "realised_draft_len_range")},
        "cap_provenance": result["cap_provenance"],
        "single_pass_top_depth": {
            "pre_e27": result["dispatch"]["single_pass_top_depth_pre_e27"],
            "now": result["dispatch"]["single_pass_top_depth_now"]},
        "identity": {k: result["identity_check"][k] for k in
                     ("max_abs_replay_residual", "max_abs_tokens_per_round_error",
                      "verdict")},
        "sensitivity": {n: {k: v for k, v in result["sensitivity"][n].items()
                            if k != "variants"} for n in CENTRAL},
        "causal_pass_cost_ms": result["local_curve"]["causal_pass_cost_ms"],
        "fit_max_resid_ms": {k: v["max_abs_residual_ms"]
                             for k, v in result["local_curve"]["fits"].items()},
        "host_transfer_scale": transfer_final["scale"],
        "host_transfer_max_resid_ms": transfer_final["max_abs_residual_ms"],
        "ranked_fit": {"linear_r2": ranked_linear["r_squared"],
                       "linear_max_resid_ms": ranked_linear["max_abs_residual_ms"],
                       "per_row_ms": ranked_linear["per_row_ms"],
                       "per_pass_ms": ranked_linear["per_weight_pass_ms"],
                       "quad_r2": ranked_quad["r_squared"],
                       "quad_per_pass_ms": ranked_quad["per_weight_pass_ms"]},
        "wall_binding": {
            name: {"policy_fraction_ge_M6": calib[name]["fraction_ge_M6"],
                   "maxent_fraction_ge_M6": result["max_entropy"][name]["fraction_ge_M6"],
                   "policy_mean_width": calib[name]["mean_width"],
                   "ledger_mean_width": profile["prompts"][name]["mean_width"]}
            for name in profile["prompts"]},
        "score_prediction": result["score_prediction"],
        "order_statistics": {
            "measured": result["score_order_statistics"]["measured"],
            "our_kernel_w5_s8": {l: result["score_order_statistics"]["our_kernel"][l]["w5_s8"]
                                 for l in models},
            "our_kernel_w4_s4": {l: result["score_order_statistics"]["our_kernel"][l]["w4_s4"]
                                 for l in models},
        },
        "local_counters": result["local_counters"]["summary"],
        "ranked_mean_width_range": result["local_counters"]["ranked_mean_width_range"],
    }, indent=1))


if __name__ == "__main__":
    main()
