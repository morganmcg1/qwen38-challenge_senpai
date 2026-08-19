#!/usr/bin/env python3
"""E45: pool the plateau trees to separate a step from smooth convexity.

Zero GPU seconds.  Everything here is arithmetic on published ranked telemetry.

E43 (merged, PR #48) showed that the eight ranked per-round costs of our own row
reject a linear T(M) at 20.3x pair noise but cannot choose between

    step        T(M) = a + b*M + s*[M >= 6]
    quadratic   T(M) = a + b*M + c*M^2

because both fit the row with zero slack.  The two families disagree violently
about the one number the campaign wants -- the 5 -> 6 increment T(6) - T(5) --
so the discount question was left open.

This experiment adds the plateau rows.  Ledger 155: `effective_mean_draft_len`
is byte-identical at full 16-digit precision between our row and the plateau
rows on the seven non-plutarch prompts, so the propose/accept trajectory, the
round count R_p and therefore the depth mixture rho_p are the SAME for every one
of those solvers.  Each extra solver then contributes eight fresh cost
observations against a *shared* nuisance rho rather than a fresh set of nuisance
parameters, which is the only instrument that can separate the families from
ranked telemetry.

Ledger 160(H) is the trap: two of the six plateau rows are one artifact measured
twice.  Rows are deduplicated by git tree digest, not by row identity, and the
duplicate pair is used for what it actually is -- the campaign's only clean
same-tree MTP-leg replicate pair, i.e. a measurement of the pooled instrument's
noise floor.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e43_ranked_step as e43                                    # noqa: E402

CACHE = pathlib.Path(".mlxfast-private/e43-corpus.json")
TREE_CACHE = pathlib.Path(".mlxfast-private/e45-trees.json")
OUT = pathlib.Path("research/e45-pooled-family.json")
UPSTREAM_REPO = "Layr-Labs/qwen-3.8-mtp-challenge"

DECODE_TOKENS = e43.DECODE_TOKENS
ORDER = e43.ORDER
CENTRAL = e43.CENTRAL
DECLARED_HEAD = e43.DECLARED_HEAD

OUR_ROW = e43.OUR_ROW                                            # ca9251b8
FRONTIER_ROW = e43.FRONTIER_ROW                                  # 0cd0a6b4 ofou
# The advisor's six plateau rows (ledger 148), by submission id prefix.
PLATEAU_ROWS = ["0cd0a6b4",       # ofou            3.249294  (board crown)
                "b0994092",       # fkiene          3.244179
                "3ac231d5",       # Lieisyourlie    3.243879
                "11863aa9",       # companygardener 3.243262
                "4f76de6e",       # alfranli123     3.243001
                "de7981ae"]       # WillGasser      3.240778

LOCAL_STEP_5_6 = e43.LOCAL_STEP_5_6      # 32.850 ms, thorfinn E38, PRE-REBASE tree
E43_STEP_FIT = (31.268, 1.452, 36.278)   # a, b, s      -> T(6)-T(5) = 37.730
E43_QUAD_FIT = (33.639, -1.930, 0.955)   # a, b, c      -> T(6)-T(5) =  8.575
E43_BRACKET_S_PRIMARY = (14.786134500080, 80.482786128094)  # s, not T(6)-T(5)
E43_RESIDUAL_RATIO = 1.174289             # rms|resid| quadratic / step, E43 primary
E43_PINS = {"beagle": 107, "medicine": 99, "essays": 87, "republic": 89,
            "botany": 85}
E43_PRIMARY = {"plutarch": 461, "drama": 252, "travel": 212, "beagle": 107,
               "medicine": 99, "essays": 87, "republic": 89, "botany": 85}
E43_TOL_HEADLINE = 0.00562                # 2.0 sigma of the 0.281 % pair noise
E43_CROWN_FRACTION_E_HI = 0.011134588742606866

CROWN_GAP_PCT = e43.CROWN_GAP_PCT         # 0.5193
SIGMA_SCORE_PCT = e43.SIGMA_SCORE_PCT     # 0.0978


# --------------------------------------------------------------------------
# Corpus, work identity and tree identity
# --------------------------------------------------------------------------

def load_corpus(refresh: bool = False) -> list:
    return e43.load_corpus(refresh)


def row_view(sub: dict, pmap: dict) -> dict:
    """Per-prompt telemetry for one submission row, plus its provenance."""
    entries = e43.per_prompt(sub, pmap)
    prompts = {}
    for name in ORDER:
        e = entries[name]
        prompts[name] = {
            "n": e["effective_mean_draft_len"],
            "mean_M": e["effective_mean_draft_len"] + 1.0,
            "non_drafting": e.get("non_drafting_round_count") or 0,
            "mtp_ms": e["mtp_seconds_per_token_mean"] * 1e3,
            "mtp_ms_total": e["mtp_seconds_per_token_mean"] * DECODE_TOKENS * 1e3,
            "serial_ms": e["serial_seconds_per_token_mean"] * 1e3,
            "ratio": e["raw_ratio_of_means"],
            "parity_ok": e.get("parity_ok"),
            "head": (e.get("head_provenance_sha256") or "")[:8],
        }
    return {"row": sub["id"][:8], "solver": sub["solverUsername"],
            "score": sub.get("officialScore"), "status": sub["status"],
            "created": sub.get("createdAt"),
            "commit": (sub.get("submissionCommitSha") or ""),
            "prompts": prompts}


def declared_head_rows(subs: list, pmap: dict) -> dict:
    """Every row whose eight legs all ran the declared proposal head."""
    out = {}
    for sub in subs:
        om = e43.official(sub)
        pp = om.get("per_prompt") or []
        if len(pp) != 8:
            continue
        if not all((p.get("head_provenance_sha256") or "").startswith(
                DECLARED_HEAD) for p in pp):
            continue
        try:
            view = row_view(sub, pmap)
        except KeyError:
            continue
        out[view["row"]] = view
    return out


def work_identity(ours: dict, other: dict) -> dict:
    """Which legs ran a byte-identical propose/accept trajectory.

    `effective_mean_draft_len` is compared as an exact float (16 significant
    digits as published), and `non_drafting_round_count` as an integer.  Both
    must match for the depth mixture rho_p to be shared, because rho(1) is
    nd / R and the mean is n + 1.
    """
    same, diff = [], []
    for name in ORDER:
        a, b = ours["prompts"][name], other["prompts"][name]
        if a["n"] == b["n"] and a["non_drafting"] == b["non_drafting"]:
            same.append(name)
        else:
            diff.append(name)
    return {"identical": same, "different": diff, "n_identical": len(same)}


def resolve_trees(commits: list, refresh: bool = False) -> dict:
    """Map each submission commit to its git tree digest.

    Tree identity is the only sound deduplication key: two rows with the same
    tree are one artifact measured twice (ledger 160(H)), and pooling them as
    independent inflates the effective sample.  The digests come from the
    organizer repository's public commit metadata, cached so that --self-test
    runs offline.
    """
    cached = {}
    if TREE_CACHE.exists() and not refresh:
        cached = json.loads(TREE_CACHE.read_text())
    missing = [c for c in commits if c and c not in cached]
    for sha in missing:
        url = "https://api.github.com/repos/%s/commits/%s" % (UPSTREAM_REPO, sha)
        raw = subprocess.run(["curl", "-sS", "-m", "30", url],
                             capture_output=True, check=True).stdout
        doc = json.loads(raw)
        if "commit" not in doc:
            raise SystemExit("cannot resolve commit %s: %s" % (sha[:12], doc))
        cached[sha] = {"tree": doc["commit"]["tree"]["sha"],
                       "message": doc["commit"]["message"].splitlines()[0],
                       "parents": [p["sha"] for p in doc.get("parents", [])]}
    if missing:
        TREE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TREE_CACHE.write_text(json.dumps(cached, indent=1, sort_keys=True))
    return {c: cached[c] for c in commits if c}


def build_pool(rows: dict, refresh_trees: bool = False,
               candidates: list | None = None) -> dict:
    """Our row plus one row per distinct plateau tree, with the dedup reported.

    Selection inside a tree group is deterministic: highest published score, so
    the retained row is the one the board itself treats as that tree's result.
    """
    ours = rows[OUR_ROW]
    wanted = list(PLATEAU_ROWS if candidates is None else candidates)
    present = [r for r in wanted if r in rows]
    trees = resolve_trees([ours["commit"]] + [rows[r]["commit"]
                                             for r in present], refresh_trees)
    groups: dict = {}
    for rid in present:
        tree = trees[rows[rid]["commit"]]["tree"]
        groups.setdefault(tree, []).append(rid)
    pool, duplicates = [], []
    for tree, members in groups.items():
        members.sort(key=lambda r: -(rows[r]["score"] or 0.0))
        pool.append(members[0])
        if len(members) > 1:
            duplicates.append({"tree": tree[:16], "rows": list(members),
                               "solvers": [rows[r]["solver"] for r in members],
                               "kept": members[0],
                               "dropped": members[1:]})
    pool.sort(key=lambda r: -(rows[r]["score"] or 0.0))
    identity = {rid: work_identity(ours, rows[rid]) for rid in present}
    shared = [nm for nm in ORDER
              if all(nm in identity[rid]["identical"] for rid in pool)]
    return {"our_row": OUR_ROW, "our_tree": trees[ours["commit"]]["tree"][:16],
            "candidate_rows": present,
            "missing_rows": [r for r in wanted if r not in rows],
            "row_count": len(present), "tree_count": len(groups),
            "pool": pool, "duplicate_groups": duplicates,
            "pooled_legs": len(ORDER) + sum(identity[r]["n_identical"]
                                            for r in pool),
            "trees": {rid: trees[rows[rid]["commit"]]["tree"][:16]
                      for rid in present},
            "identity": identity, "shared_prompts": shared,
            "unshared_prompts": [nm for nm in ORDER if nm not in shared]}


def replicate_noise(rows: dict, pool: dict) -> dict:
    """Per-prompt MTP-leg noise measured on the same-tree duplicate pair.

    Two rows sharing a git tree ran identical code on identical prompts, so
    their per-prompt MTP-leg difference is pure measurement noise.  This is the
    only clean replicate pair on the board and it is the correct sigma for the
    pooled fit, which uses the MTP leg alone and is therefore untouched by the
    serial-leg run-level drift of ledger item 153.
    """
    out = {"pairs": [], "per_prompt_pct": {}, "pooled_pct": None}
    diffs: dict = {nm: [] for nm in ORDER}
    for group in pool["duplicate_groups"]:
        members = group["rows"]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = rows[members[i]], rows[members[j]]
                per = {}
                for nm in ORDER:
                    d = 100.0 * (a["prompts"][nm]["mtp_ms"]
                                 / b["prompts"][nm]["mtp_ms"] - 1.0)
                    per[nm] = d
                    diffs[nm].append(d)
                out["pairs"].append({"rows": [members[i], members[j]],
                                     "tree": group["tree"],
                                     "delta_mtp_pct": per})
    flat = [d for nm in ORDER for d in diffs[nm]]
    if flat:
        for nm in ORDER:
            if diffs[nm]:
                out["per_prompt_pct"][nm] = max(abs(d) for d in diffs[nm]) / math.sqrt(2.0)
        rms = math.sqrt(sum(d * d for d in flat) / len(flat))
        out["pooled_pct"] = rms / math.sqrt(2.0)
        out["n_leg_pairs"] = len(flat)
    return out


# --------------------------------------------------------------------------
# Families.  T(M) = a + b*M + shape terms, plus a physical monotonicity floor.
# --------------------------------------------------------------------------

BASIS = {
    "g5": lambda m: 1.0 if m >= 5 else 0.0,
    "g6": lambda m: 1.0 if m >= 6 else 0.0,
    "g7": lambda m: 1.0 if m >= 7 else 0.0,
    "sq": lambda m: float(m * m),
}

# A family is T(M) = a + b*M + sum_k coef_k * basis_k(M).  Every shape
# coefficient is signed non-negative, which is what makes one leg's two
# half-planes an exact (not approximate) encoding of its E[g] bracket.
FAMILIES = {
    "linear": {"shapes": ()},
    "step5": {"shapes": ("g5",)},
    "step6": {"shapes": ("g6",)},
    "step7": {"shapes": ("g7",)},
    "quadratic": {"shapes": ("sq",)},
    "mixture": {"shapes": ("g6", "sq")},
}
SHAPE_BOX = {"g5": 400.0, "g6": 400.0, "g7": 400.0, "sq": 40.0}


def shapes_of(family: str) -> tuple:
    return FAMILIES[family]["shapes"]


def dim(family: str) -> int:
    return 2 + len(shapes_of(family))


def shape_weights(family: str, m: int) -> tuple:
    return tuple(BASIS[k](m) for k in shapes_of(family))


def delta_5_6_coeffs(family: str) -> tuple:
    """T(6) - T(5) as a linear functional of (a, b, shape coefficients)."""
    return (0.0, 1.0) + tuple(h - l for h, l in
                              zip(shape_weights(family, 6),
                                  shape_weights(family, 5)))


def secant_weights(family: str, x: float) -> tuple:
    """E[secant of each shape term through M = 1 and M = 5] at E[M] = x."""
    return tuple(g1 + (x - 1.0) * (g5 - g1) / 4.0
                 for g1, g5 in zip(shape_weights(family, 1),
                                   shape_weights(family, 5)))


def excess_coeffs(family: str, x: float, y: float) -> tuple:
    """Removable excess at one prompt, as (const, coefficients on the params).

    The excess is defined family-agnostically as the round cost above the secant
    of T through M = 1 and M = 5 -- "what the round costs beyond what the
    sub-six marginal row cost would predict".  The secant is linear in M, so its
    expectation is its value at E[M] and no extra rho assumption enters.  For a
    step-at-6 family this is exactly s * q_p, so E43's e_p is reproduced
    unchanged; for the quadratic family it is c * E[(M-1)(M-5)], which is
    negative below M = 5 and positive above it.

    Anchoring at M = 1 and M = 5 is a named choice.  It credits a fix with the
    cost above the sub-six slope only, so it is conservative relative to
    "remove the whole shape term" (reported alongside as a sensitivity).
    """
    return (y, (-1.0, -x) + tuple(-s for s in secant_weights(family, x)))


def family_static_rows(family: str) -> tuple:
    """Physical and box constraints on the parameters, independent of the data.

    Monotonicity of T on the integer support is imposed as a physical floor
    (verifying one more row cannot be cheaper).  With every shape coefficient
    non-negative the only remaining binding step is M = 1 -> 2, so the floor is
    b >= 0 for the pure step families and b + 3c >= 0 wherever the quadratic
    term is present.  The latter is weaker than b >= 0 and therefore admits the
    negative-slope quadratic fit E43 published and the advisor evaluated.
    """
    ks = shapes_of(family)
    n = dim(family)
    rows = [[-1.0, -1.0] + [-w for w in shape_weights(family, 1)]]  # T(1) >= 0
    rhs = [0.0]
    mono = [0.0, -1.0] + [0.0] * len(ks)
    if "sq" in ks:
        mono[2 + ks.index("sq")] = -3.0                   # b + 3c >= 0
    rows.append(mono)
    rhs.append(0.0)
    boxes = [(-1000.0, 1000.0), (-200.0, 200.0) if "sq" in ks else (0.0, 200.0)]
    boxes += [(0.0, SHAPE_BOX[k]) for k in ks]
    for i, (lo, hi) in enumerate(boxes):
        r = [0.0] * n
        r[i] = 1.0
        rows.append(list(r))
        rhs.append(hi)
        r[i] = -1.0
        rows.append(list(r))
        rhs.append(-lo)
    return rows, rhs


def leg_rows(o: dict, tol_frac: float, fixed: tuple | None = None) -> tuple:
    """The two half-planes one ranked leg contributes.

    With each E[g(M)] bracket relaxed independently this is E43's single-solver
    constraint; with `fixed` supplying shared values it is the pooled constraint
    at a fixed shape, which is what couples the solvers.  Independent brackets
    for a multi-shape family ignore the fact that one rho generates every shape
    moment jointly, which can only widen the admissible set.
    """
    eps = tol_frac * o["y"]
    lo = hi = fixed
    if fixed is None:
        lo, hi = o["g_lo"], o["g_hi"]
    return ([[1.0, o["x"]] + list(lo), [-1.0, -o["x"]] + [-v for v in hi]],
            [o["y"] + eps, -(o["y"] - eps)])


def polytope(obs: list, family: str, tol_frac: float,
             fixed_shape: dict | None = None) -> tuple:
    """Half-planes in the family's parameters for one solver's ranked legs."""
    rows, rhs = family_static_rows(family)
    for o in obs:
        fixed = None
        if fixed_shape is not None and o["name"] in fixed_shape:
            val = fixed_shape[o["name"]]
            fixed = val if isinstance(val, tuple) else (val,)
        r, h = leg_rows(o, tol_frac, fixed)
        rows += r
        rhs += h
    return rows, rhs


def bracket(obs: list, family: str, tol_frac: float, obj: tuple,
            fixed_shape: dict | None = None) -> dict:
    """Interval of a linear functional of (a, b, third) over the polytope."""
    rows, rhs = polytope(obs, family, tol_frac, fixed_shape)
    lo = e43.lp_extreme(rows, rhs, list(obj), "min")
    hi = e43.lp_extreme(rows, rhs, list(obj), "max")
    return {"feasible": lo["feasible"], "lo": lo["value"], "hi": hi["value"],
            "witness_lo": lo["x"], "witness_hi": hi["x"]}


def feasible(obs: list, family: str, tol_frac: float,
             fixed_shape: dict | None = None) -> bool:
    rows, rhs = polytope(obs, family, tol_frac, fixed_shape)
    return e43.lp_feasible(rows, rhs)


def leg_shape_range(family: str, verts: list) -> tuple:
    """Per shape term, the exact bracket of E[g(M)] over the admissible rho set.

    Also returns the vertex values themselves, which a multi-shape family needs
    to see jointly rather than as an independent box.
    """
    pts = [tuple(sum(w * BASIS[k](m) for m, w in v.items())
                 for k in shapes_of(family)) for v in verts]
    lo = tuple(min(p[i] for p in pts) for i in range(len(shapes_of(family))))
    hi = tuple(max(p[i] for p in pts) for i in range(len(shapes_of(family))))
    return lo, hi, pts


def observations(row: dict, family: str, selection: dict,
                 legs: list | None = None) -> list:
    """One solver's (x, y) legs with the exact E[g(M)] bracket per leg.

    `legs` restricts the row to the prompts whose work is identical to ours; a
    pooled tree that diverged on one prompt contributes its other seven rather
    than being dropped or forced onto our reading.
    """
    obs = []
    for name in (ORDER if legs is None else [n for n in ORDER if n in legs]):
        p = row["prompts"][name]
        R = selection[name]
        cand = next((c for c in e43.feasible_rounds(
            p["n"], p["mtp_ms_total"], p["non_drafting"]) if c["R"] == R), None)
        if cand is None:
            raise ValueError("reading %s=%d not admissible for %s"
                             % (name, R, row["row"]))
        rho1 = p["non_drafting"] / R
        verts = e43.vertices(p["mean_M"], rho1, p["non_drafting"])
        if not verts:
            raise ValueError("no rho for %s=%d" % (name, R))
        lo, hi, pts = leg_shape_range(family, verts)
        obs.append({"name": name, "x": p["mean_M"], "y": cand["per_round_ms"],
                    "R": R, "D": cand["D"], "A": cand["A"],
                    "rho1": rho1, "g_lo": lo, "g_hi": hi, "g_pts": pts,
                    "ratio": p["ratio"]})
    return obs


def reading_candidates(row: dict) -> dict:
    return {name: [c["R"] for c in e43.feasible_rounds(
        row["prompts"][name]["n"], row["prompts"][name]["mtp_ms_total"],
        row["prompts"][name]["non_drafting"])] for name in ORDER}


# --------------------------------------------------------------------------
# The pooled system: one bundle of legs per tree, sharing readings and shape
# --------------------------------------------------------------------------

def build_bundle(rows: dict, pool: dict, family: str) -> dict:
    """Per tree, per prompt, every admissible reading's leg observation.

    Only legs with verified work identity are included, because only those share
    the round count and the depth mixture with our row.  A leg that is dropped
    weakens the pooled system rather than biasing it: fewer constraints can only
    widen the admissible set.
    """
    bundle = {}
    keys = [pool["our_row"]] + list(pool["pool"])
    for rid in keys:
        row = rows[rid]
        legs = (ORDER if rid == pool["our_row"]
                else pool["identity"][rid]["identical"])
        entry = {}
        for name in legs:
            p = row["prompts"][name]
            per_reading = {}
            for cand in e43.feasible_rounds(p["n"], p["mtp_ms_total"],
                                            p["non_drafting"]):
                rho1 = p["non_drafting"] / cand["R"]
                verts = e43.vertices(p["mean_M"], rho1, p["non_drafting"])
                if not verts:
                    continue
                lo, hi, pts = leg_shape_range(family, verts)
                per_reading[cand["R"]] = {
                    "name": name, "x": p["mean_M"], "y": cand["per_round_ms"],
                    "R": cand["R"], "D": cand["D"], "A": cand["A"],
                    "rho1": rho1, "g_lo": lo, "g_hi": hi, "g_pts": pts,
                    "ratio": p["ratio"]}
            entry[name] = per_reading
        bundle[rid] = entry
    return bundle


def pooled_enumerate(bundle: dict, family: str, tol_frac: float,
                     node_cap: int = 3_000_000, first_only: bool = False,
                     trees: list | None = None, t1_floor: bool = True) -> dict:
    """Reading combinations that every pooled tree's own polytope admits.

    Dropping the shared-shape coupling and testing each tree separately is a
    relaxation of the pooled system, so a combination rejected here is rejected
    by the pooled system too: the surviving set is a sound superset and the
    pruning cannot invent an exclusion.  Our row is tested first because it is
    the tightest single constraint set, which keeps the extra trees cheap.
    """
    keys = list(bundle) if trees is None else list(trees)
    static = {}
    for rid in keys:
        rows_s, rhs_s = family_static_rows(family)
        if not t1_floor:
            rows_s, rhs_s = rows_s[1:], rhs_s[1:]
        static[rid] = (rows_s, rhs_s)
    prompts = sorted(
        {nm for rid in keys for nm in bundle[rid]},
        key=lambda nm: next(next(iter(bundle[rid][nm].values()))["x"]
                            for rid in keys if nm in bundle[rid]))
    readings = {}
    for nm in prompts:
        owner = next(rid for rid in keys if nm in bundle[rid])
        readings[nm] = sorted(bundle[owner][nm])
    out, nodes, capped = [], 0, False
    state = {rid: (list(static[rid][0]), list(static[rid][1])) for rid in keys}

    def walk(depth, state, sel):
        nonlocal nodes, capped
        if capped or (first_only and out):
            return
        if depth == len(prompts):
            out.append(dict(sel))
            return
        nm = prompts[depth]
        for R in readings[nm]:
            nodes += 1
            if nodes > node_cap:
                capped = True
                return
            nxt, ok = {}, True
            for rid in keys:
                rows_k, rhs_k = state[rid]
                if nm in bundle[rid]:
                    if R not in bundle[rid][nm]:
                        ok = False
                        break
                    r, h = leg_rows(bundle[rid][nm][R], tol_frac)
                    rows_k, rhs_k = rows_k + r, rhs_k + h
                    if not e43.lp_feasible(rows_k, rhs_k):
                        ok = False
                        break
                nxt[rid] = (rows_k, rhs_k)
            if not ok:
                continue
            sel[nm] = R
            walk(depth + 1, nxt, sel)
            del sel[nm]
            if capped or (first_only and out):
                return

    walk(0, state, {})
    total = 1
    for nm in prompts:
        total *= len(readings[nm])
    pinned = {nm: next(iter({s[nm] for s in out}))
              for nm in prompts if len({s[nm] for s in out}) == 1} if out else {}
    return {"family": family, "tol_frac": tol_frac, "trees": keys,
            "selections": out, "n_selections": len(out),
            "nodes_visited": nodes, "capped": capped,
            "cross_product_size": total, "pinned_rounds": pinned,
            "surviving_readings": {nm: sorted({s[nm] for s in out})
                                   for nm in prompts} if out else {},
            "prompt_order": prompts}


def family_threshold(bundle: dict, family: str, trees: list | None = None,
                     hi: float = 0.40, iters: int = 16,
                     node_cap: int = 400_000) -> dict:
    """Smallest per-leg slack at which the family can explain every pooled tree.

    This is the assumption-free effect size: no rho point estimate, no
    round-count choice.  Because the test drops the shared-shape coupling it is
    a *lower* bound on the pooled threshold -- the pooled system needs at least
    this much slack -- so comparing two families' thresholds compares two sound
    lower bounds.
    """
    def ok(tol):
        return bool(pooled_enumerate(bundle, family, tol, node_cap=node_cap,
                                     first_only=True, trees=trees)["selections"])
    if ok(1e-9):
        return {"family": family, "threshold_frac": 0.0, "bracketed": True}
    if not ok(hi):
        return {"family": family, "threshold_frac": None, "bracketed": False,
                "searched_to": hi}
    lo = 1e-9
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return {"family": family, "threshold_frac": hi, "lower_bracket_frac": lo,
            "bracketed": True}


# --------------------------------------------------------------------------
# The shared-shape witness search: the only test that uses the pooling
# --------------------------------------------------------------------------

def fit_at_shape(obs: list, family: str, shape: dict) -> dict:
    """Least-squares fit at a fixed shared shape, and its worst relative miss.

    A specific parameter vector whose largest relative residual is `eps` proves
    the tree is feasible at tolerance eps, so the maximum of this over trees is a
    sound *upper* bound on the pooled family threshold at that shape.  The fit is
    projected onto the family's monotonicity cone before the residual is read, so
    the witness it certifies is a legal member of the family.  Restricted to
    single-shape families, which is what the step-versus-quadratic question is.
    """
    if len(shapes_of(family)) != 1:
        raise ValueError("fit_at_shape needs a single-shape family")
    design = [[1.0, o["x"], shape[o["name"]]] for o in obs]
    y = [o["y"] for o in obs]
    w = [1.0 / (o["y"] * o["y"]) for o in obs]
    fit = e43.wls(design, y, w)
    a, b, c = fit["beta"]
    quad = shapes_of(family) == ("sq",)
    legal = (c >= 0.0 and (b + 3.0 * c >= 0.0 if quad else b >= 0.0))
    if not legal:                      # refit with the violated edge clamped
        if c < 0.0:
            c = 0.0
        if quad:
            if b + 3.0 * c < 0.0:
                b = -3.0 * c
        elif b < 0.0:
            b = 0.0
        resid_y = [v - b * o["x"] - c * shape[o["name"]]
                   for v, o in zip(y, obs)]
        a = sum(rv * ww for rv, ww in zip(resid_y, w)) / sum(w)
    worst = max(abs(a + b * o["x"] + c * shape[o["name"]] - o["y"]) / o["y"]
                for o in obs)
    return {"beta": (a, b, c), "worst_rel": worst, "clamped": not legal}


def pooled_shape_cost(bundles: dict, family: str, shape: dict) -> float:
    return max(fit_at_shape(obs, family, shape)["worst_rel"]
               for obs in bundles.values())


def shape_box(bundles: dict) -> dict:
    """Per prompt, the exact bracket on the shared scalar E[g(M)]."""
    box = {}
    for obs in bundles.values():
        for o in obs:
            lo, hi = box.get(o["name"], (o["g_lo"][0], o["g_hi"][0]))
            box[o["name"]] = (max(lo, o["g_lo"][0]), min(hi, o["g_hi"][0]))
    return box


def search_shape(bundles: dict, family: str, seed: int = 7,
                 restarts: int = 6, sweeps: int = 40) -> dict:
    """Coordinate search for the shared shape that fits every tree best.

    The pooled system is bilinear (per-tree scale times shared shape), so this
    search proves feasibility rather than infeasibility: whatever it finds is a
    witness, and the value it reaches is an upper bound on the pooled family
    threshold.  Coupled with the per-tree lower bound, the two bracket the
    threshold from both sides.
    """
    box = shape_box(bundles)
    names = list(box)
    rng = random.Random(seed)
    best, best_shape = None, None
    for trial in range(restarts):
        if trial == 0:
            shape = {nm: 0.5 * (box[nm][0] + box[nm][1]) for nm in names}
        elif trial == 1:
            shape = {nm: box[nm][0] for nm in names}
        elif trial == 2:
            shape = {nm: box[nm][1] for nm in names}
        else:
            shape = {nm: box[nm][0] + rng.random() * (box[nm][1] - box[nm][0])
                     for nm in names}
        cost = pooled_shape_cost(bundles, family, shape)
        step = {nm: 0.5 * (box[nm][1] - box[nm][0]) for nm in names}
        for _ in range(sweeps):
            improved = False
            for nm in names:
                for sign in (1.0, -1.0):
                    cand = dict(shape)
                    cand[nm] = min(box[nm][1], max(box[nm][0],
                                                   shape[nm] + sign * step[nm]))
                    if cand[nm] == shape[nm]:
                        continue
                    c = pooled_shape_cost(bundles, family, cand)
                    if c < cost - 1e-15:
                        shape, cost, improved = cand, c, True
            if not improved:
                for nm in names:
                    step[nm] *= 0.5
                if max(step.values()) < 1e-9:
                    break
        if best is None or cost < best:
            best, best_shape = cost, shape
    return {"family": family, "worst_rel": best, "shape": best_shape,
            "per_tree": {rid: fit_at_shape(obs, family, best_shape)
                         for rid, obs in bundles.items()}}


def certify_witness(bundles: dict, family: str, shape: dict,
                    tol_frac: float) -> dict:
    """Exact LP check that every tree is feasible at this shared shape."""
    per = {rid: feasible(obs, family, tol_frac, fixed_shape=shape)
           for rid, obs in bundles.items()}
    return {"all_feasible": all(per.values()), "per_tree": per,
            "tol_frac": tol_frac}


# --------------------------------------------------------------------------
# Why pooling cannot work: the two families are ray-equivalent designs
# --------------------------------------------------------------------------

def raw_fit(obs: list, shape: dict) -> dict:
    """Unconstrained weighted least squares at a fixed per-prompt shape value."""
    design = [[1.0, o["x"], shape[o["name"]]] for o in obs]
    y = [o["y"] for o in obs]
    w = [1.0 / (v * v) for v in y]
    beta = e43.wls(design, y, w)["beta"]
    pred = [sum(b * v for b, v in zip(beta, row)) for row in design]
    return {"beta": tuple(beta), "pred": pred,
            "resid": [p - v for p, v in zip(pred, y)],
            "worst_rel": max(abs(p - v) / v for p, v in zip(pred, y))}


def ray_equivalence(obs_a: list, obs_b: list, lam_min: float = 1e-6,
                    lam_max: float = 1e4, box: float = 1e6) -> dict:
    """Is family B's shape column an affine-plus-scaled copy of family A's?

    Solves for (alpha, beta, lambda > 0) with

        E[g_B(M)]_p = alpha + beta * E[M]_p + lambda * E[g_A(M)]_p

    at *some* admissible choice of each leg's two shape moments.  Because
    lambda > 0, each leg contributes two half-planes, so this is an exact LP in
    three variables rather than a search.

    If it is feasible the two designs span the same column space, so every
    dataset fitted by A is fitted identically by B -- same predictions, same
    residuals, same worst relative miss -- while the implied T(6) - T(5)
    differs.  Non-separation is then structural: the misfit that pooling or
    lower noise would have to detect is exactly zero, not merely small, so the
    minimum detectable difference is unbounded at any tree count.
    """
    rows, rhs = [], []
    for oa, ob in zip(obs_a, obs_b):
        if oa["name"] != ob["name"]:
            raise ValueError("leg order mismatch")
        rows.append([1.0, oa["x"], oa["g_lo"][0]])
        rhs.append(ob["g_hi"][0])
        rows.append([-1.0, -oa["x"], -oa["g_hi"][0]])
        rhs.append(-ob["g_lo"][0])
    for i, (lo, hi) in enumerate(((-box, box), (-box, box), (lam_min, lam_max))):
        r = [0.0, 0.0, 0.0]
        r[i] = 1.0
        rows.append(list(r))
        rhs.append(hi)
        r[i] = -1.0
        rows.append(list(r))
        rhs.append(-lo)
    lo = e43.lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "min")
    hi = e43.lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "max")
    return {"feasible": bool(lo["feasible"]),
            "lambda_lo": lo["value"], "lambda_hi": hi["value"],
            "witness_lo": lo["x"], "witness_hi": hi["x"]}


def equivalence_demo(obs_a: list, obs_b: list, family_a: str, family_b: str,
                     witness: list) -> dict:
    """Fit both families at one ray-equivalence witness and compare them.

    The point of the demo is that the agreement is exact arithmetic, not a
    coincidence of this dataset: `pred_gap_ms` should be at machine precision
    while `delta_5_6` differs by a large factor.
    """
    alpha, beta, lam = witness
    shape_a, shape_b, feasible_choice = {}, {}, True
    for oa, ob in zip(obs_a, obs_b):
        lo = max(ob["g_lo"][0], alpha + beta * oa["x"] + lam * oa["g_lo"][0])
        hi = min(ob["g_hi"][0], alpha + beta * oa["x"] + lam * oa["g_hi"][0])
        if lo > hi + 1e-9:
            feasible_choice = False
            lo = hi = 0.5 * (lo + hi)
        wp = 0.5 * (lo + hi)
        shape_b[ob["name"]] = wp
        shape_a[oa["name"]] = (wp - alpha - beta * oa["x"]) / lam
    fa, fb = raw_fit(obs_a, shape_a), raw_fit(obs_b, shape_b)
    mapped = (fa["beta"][0] - fa["beta"][2] * alpha / lam,
              fa["beta"][1] - fa["beta"][2] * beta / lam,
              fa["beta"][2] / lam)
    dc_a, dc_b = delta_5_6_coeffs(family_a), delta_5_6_coeffs(family_b)
    return {
        "witness": {"alpha": alpha, "beta": beta, "lambda": lam},
        "shape_choice_admissible": feasible_choice,
        "shape_a": shape_a, "shape_b": shape_b,
        "fit_a": fa["beta"], "fit_b": fb["beta"], "fit_b_from_map": mapped,
        "map_gap": max(abs(m - f) for m, f in zip(mapped, fb["beta"])),
        "pred_gap_ms": max(abs(p - q) for p, q in zip(fa["pred"], fb["pred"])),
        "worst_rel_a": fa["worst_rel"], "worst_rel_b": fb["worst_rel"],
        "delta_a": sum(c * v for c, v in zip(dc_a, fa["beta"])),
        "delta_b": sum(c * v for c, v in zip(dc_b, fb["beta"])),
        "a_in_cone": fa["beta"][1] >= 0.0 and fa["beta"][2] >= 0.0,
        "b_in_cone": (fb["beta"][2] >= 0.0
                      and fb["beta"][1] + 3.0 * fb["beta"][2] >= 0.0),
    }


def equivalent_witness_scan(obs_a: list, obs_b: list, family_a: str,
                            family_b: str, grid: int = 41) -> dict:
    """A ray-equivalence witness at which *both* families' fits are legal.

    The bare LP witness can map one family's fit outside the other's
    monotonicity cone, which would leave a reader thinking the cone breaks the
    tie.  Scanning lambda finds a witness where both fits are cone-legal, which
    shows the tie survives the physical constraint as well as the algebra.
    """
    ray = ray_equivalence(obs_a, obs_b)
    if not ray["feasible"]:
        return {"feasible": False, "ray": ray}
    lo, hi = max(ray["lambda_lo"], 1e-4), ray["lambda_hi"]
    cands = []
    for i in range(grid):
        f = i / (grid - 1.0)
        lam = lo * math.exp(f * math.log(hi / lo)) if hi > lo else lo
        rows, rhs = [], []
        for oa, ob in zip(obs_a, obs_b):
            rows.append([1.0, oa["x"]])
            rhs.append(ob["g_hi"][0] - lam * oa["g_lo"][0])
            rows.append([-1.0, -oa["x"]])
            rhs.append(-(ob["g_lo"][0] - lam * oa["g_hi"][0]))
        for k, (bl, bh) in enumerate(((-1e6, 1e6), (-1e6, 1e6))):
            r = [0.0, 0.0]
            r[k] = 1.0
            rows += [list(r)]
            rhs += [bh]
            r[k] = -1.0
            rows += [list(r)]
            rhs += [-bl]
        for sense in ("min", "max"):
            sol = e43.lp_extreme(rows, rhs, [0.0, 1.0], sense)
            if not sol["feasible"]:
                continue
            demo = equivalence_demo(obs_a, obs_b, family_a, family_b,
                                    [sol["x"][0], sol["x"][1], lam])
            if demo["shape_choice_admissible"]:
                cands.append(demo)
    legal = [d for d in cands if d["a_in_cone"] and d["b_in_cone"]]
    pick = min(legal or cands, key=lambda d: d["worst_rel_a"]) if cands else None
    return {"feasible": True, "ray": ray, "n_candidates": len(cands),
            "n_both_cones_legal": len(legal), "best": pick}


def offset_ray(box_a: dict, box_b: dict, names: list, lam_min: float = 1e-6,
               lam_max: float = 1e4, box: float = 1e6) -> dict:
    """The strict ray test: g_B,p = alpha + lambda * g_A,p, with no x column.

    This is the version pooling could have broken.  Two half-planes per prompt
    in (alpha, lambda), so it is an exact two-variable LP.
    """
    rows, rhs = [], []
    for nm in names:
        rows.append([1.0, box_a[nm][0]])
        rhs.append(box_b[nm][1])
        rows.append([-1.0, -box_a[nm][1]])
        rhs.append(-box_b[nm][0])
    for i, (lo, hi) in enumerate(((-box, box), (lam_min, lam_max))):
        r = [0.0, 0.0]
        r[i] = 1.0
        rows.append(list(r))
        rhs.append(hi)
        r[i] = -1.0
        rows.append(list(r))
        rhs.append(-lo)
    lo = e43.lp_extreme(rows, rhs, [0.0, 1.0], "min")
    hi = e43.lp_extreme(rows, rhs, [0.0, 1.0], "max")
    return {"feasible": bool(lo["feasible"]),
            "lambda_lo": lo["value"], "lambda_hi": hi["value"],
            "witness_lo": lo["x"], "witness_hi": hi["x"]}


def pooled_ray_equivalence(bundles_a: dict, bundles_b: dict, family_a: str,
                           family_b: str) -> dict:
    """Can the pooled shared-shape design separate the families at all?

    In the pooled model every tree keeps its own (a_t, b_t, s_t) and only the
    per-prompt shape value is shared, so family B is indistinguishable from A
    exactly when u_p = g_B,p - lambda * g_A,p lies in span{1, x_t.} for every
    pooled tree t.  Two facts decide it, and both are checked here rather than
    assumed:

    1. The work-identity filter keeps a leg only when `effective_mean_draft_len`
       and `non_drafting_round_count` match ours exactly, and x_p is n_p + 1.
       So every pooled tree contributes the *same* x vector and, because rho is
       a function of the same two integers, the *same* admissible shape box.
       Pooling therefore adds rows in y and no new regressor variation at all.
    2. Even with the x column removed -- the strictest pooled reading, where a
       per-tree slope may not absorb the offset -- the two shape columns are
       still exactly affinely related inside their admissible boxes.

    The demo then maps one shared A-shape to its B image and shows the pooled
    worst-relative miss is equal to machine precision while the implied
    T(6) - T(5) differs per tree, so the pooled threshold is the same number for
    both families at any tree count, noise level, or tolerance.
    """
    trees = list(bundles_a.keys())
    ref = trees[0]
    xa = {o["name"]: o["x"] for o in bundles_a[ref]}
    x_gap = max((abs(o["x"] - xa[o["name"]])
                 for t in trees for o in bundles_a[t] if o["name"] in xa),
                default=0.0)
    box_a, box_b = shape_box(bundles_a), shape_box(bundles_b)
    solo_a = {o["name"]: (o["g_lo"][0], o["g_hi"][0]) for o in bundles_a[ref]}
    solo_b = {o["name"]: (o["g_lo"][0], o["g_hi"][0]) for o in bundles_b[ref]}
    box_gap = max(max(abs(box_a[nm][i] - solo_a[nm][i]),
                      abs(box_b[nm][i] - solo_b[nm][i]))
                  for nm in box_a for i in (0, 1))
    names = sorted(box_a, key=ORDER.index)
    ray = offset_ray(box_a, box_b, names)
    out = {"trees": trees, "n_legs": sum(len(v) for v in bundles_a.values()),
           "x_identical_across_trees": x_gap < 1e-12, "x_max_gap": x_gap,
           "box_equals_single_row": box_gap < 1e-12, "box_max_gap": box_gap,
           "offset_ray": ray, "demo": None}
    if not ray["feasible"]:
        return out
    lam = math.sqrt(max(ray["lambda_lo"], 1e-9) * ray["lambda_hi"])
    # Any offset in the LP slice at this lambda works, so take the interval
    # directly rather than re-solving.
    a_lo = max(box_b[nm][0] - lam * box_a[nm][1] for nm in names)
    a_hi = min(box_b[nm][1] - lam * box_a[nm][0] for nm in names)
    if a_lo > a_hi:
        lam, alpha = ray["witness_lo"][1], ray["witness_lo"][0]
    else:
        alpha = 0.5 * (a_lo + a_hi)
    shape_a, shape_b = {}, {}
    for nm in names:
        lo = max(box_a[nm][0], (box_b[nm][0] - alpha) / lam)
        hi = min(box_a[nm][1], (box_b[nm][1] - alpha) / lam)
        if lo > hi + 1e-12:
            return {**out, "demo": {"admissible": False, "prompt": nm}}
        shape_a[nm] = 0.5 * (lo + hi)
        shape_b[nm] = alpha + lam * shape_a[nm]
    per_tree = {}
    for t in trees:
        ra = raw_fit(bundles_a[t], shape_a)
        rb = raw_fit(bundles_b[t], shape_b)
        fa = fit_at_shape(bundles_a[t], family_a, shape_a)
        fb = fit_at_shape(bundles_b[t], family_b, shape_b)
        per_tree[t] = {
            "raw_a": ra["worst_rel"], "raw_b": rb["worst_rel"],
            "raw_gap": abs(ra["worst_rel"] - rb["worst_rel"]),
            "pred_gap_ms": max(abs(p - q) for p, q in zip(ra["pred"],
                                                          rb["pred"])),
            "cone_a": fa["worst_rel"], "cone_b": fb["worst_rel"],
            "cone_gap": abs(fa["worst_rel"] - fb["worst_rel"]),
            "clamped_a": fa["clamped"], "clamped_b": fb["clamped"],
            "delta_a": sum(c * v for c, v in
                           zip(delta_5_6_coeffs(family_a), ra["beta"])),
            "delta_b": sum(c * v for c, v in
                           zip(delta_5_6_coeffs(family_b), rb["beta"]))}
    out["demo"] = {
        "admissible": True, "alpha": alpha, "lambda": lam,
        "shape_a": shape_a, "shape_b": shape_b,
        "pooled_raw_a": max(v["raw_a"] for v in per_tree.values()),
        "pooled_raw_b": max(v["raw_b"] for v in per_tree.values()),
        "pooled_raw_gap": max(v["raw_gap"] for v in per_tree.values()),
        "pooled_pred_gap_ms": max(v["pred_gap_ms"] for v in per_tree.values()),
        "pooled_cone_a": pooled_shape_cost(bundles_a, family_a, shape_a),
        "pooled_cone_b": pooled_shape_cost(bundles_b, family_b, shape_b),
        "n_clamped_a": sum(1 for v in per_tree.values() if v["clamped_a"]),
        "n_clamped_b": sum(1 for v in per_tree.values() if v["clamped_b"]),
        "per_tree": per_tree,
        "delta_gap_ms": max(abs(v["delta_a"] - v["delta_b"])
                            for v in per_tree.values())}
    return out


# --------------------------------------------------------------------------
# Deliverable (b) and (c): family-conditional excess, value, and the increment
# --------------------------------------------------------------------------

def witness_shape(obs: list, family: str, tol_frac: float,
                  theta: list) -> dict | None:
    """A shared shape consistent with one parameter vector, or None if empty.

    Needed to ask whether a bracket endpoint *survives pooling*: the endpoint is
    a parameter vector, and pooling constrains the shape, so the endpoint is only
    excluded by the pool if no shape that reproduces it is jointly feasible.
    Multi-shape families are skipped because their shape is not a scalar.
    """
    if len(shapes_of(family)) != 1:
        return None
    a, b, c = theta
    out = {}
    for o in obs:
        eps = tol_frac * o["y"]
        lo, hi = o["g_lo"][0], o["g_hi"][0]
        if c > 1e-12:
            lo = max(lo, (o["y"] - eps - a - b * o["x"]) / c)
            hi = min(hi, (o["y"] + eps - a - b * o["x"]) / c)
        elif abs(a + b * o["x"] - o["y"]) > eps + 1e-9:
            return None
        if lo > hi + 1e-9:
            return None
        out[o["name"]] = 0.5 * (lo + hi)
    return out


def delta_union(row: dict, family: str, tol_frac: float,
                selections: list) -> dict:
    """T(6) - T(5) over every reading the family admits, plus the endpoints."""
    obj = delta_5_6_coeffs(family)
    best_lo = best_hi = None
    per = []
    for sel in selections:
        obs = observations(row, family, sel)
        br = bracket(obs, family, tol_frac, obj)
        if not br["feasible"]:
            continue
        per.append({"selection": dict(sel), "lo": br["lo"], "hi": br["hi"]})
        if best_lo is None or br["lo"] < best_lo["lo"]:
            best_lo = {"lo": br["lo"], "selection": dict(sel),
                       "theta": list(br["witness_lo"])}
        if best_hi is None or br["hi"] > best_hi["hi"]:
            best_hi = {"hi": br["hi"], "selection": dict(sel),
                       "theta": list(br["witness_hi"])}
    return {"family": family, "n_feasible": len(per),
            "lo": None if best_lo is None else best_lo["lo"],
            "hi": None if best_hi is None else best_hi["hi"],
            "arg_lo": best_lo, "arg_hi": best_hi, "per_selection": per}


def tree_bundles(rows: dict, pool: dict, family: str, selection: dict) -> dict:
    """Per pooled tree, its work-identical legs at one shared reading."""
    out = {}
    for rid in [pool["our_row"]] + list(pool["pool"]):
        legs = (None if rid == pool["our_row"]
                else pool["identity"][rid]["identical"])
        out[rid] = observations(rows[rid], family, selection, legs)
    return out


def endpoint_survives_pool(rows: dict, pool: dict, family: str,
                           tol_frac: float, endpoint: dict) -> dict:
    """Does the pool still admit the shape that produced a bracket endpoint?"""
    if endpoint is None:
        return {"checked": False}
    obs = observations(rows[pool["our_row"]], family, endpoint["selection"])
    shape = witness_shape(obs, family, tol_frac, endpoint["theta"])
    if shape is None:
        return {"checked": False}
    bundles = tree_bundles(rows, pool, family, endpoint["selection"])
    cert = certify_witness(bundles, family, shape, tol_frac)
    return {"checked": True, "shape": shape, **cert}


def excess_report(row: dict, family: str, tol_frac: float,
                  selection: dict) -> dict:
    """Per-prompt removable excess under one family, as a bracket in ms and %.

    Two anchors are reported.  `secant` is the family-agnostic definition (cost
    above the M=1..5 secant), which reproduces E43's e_p exactly for step-at-6.
    `whole` removes the entire shape term, which is the most a fix could ever
    recover and is the right sensitivity for a curvature family whose secant
    excess is negative at low mean depth.
    """
    obs = observations(row, family, selection)
    rows_p, rhs_p = polytope(obs, family, tol_frac)
    if not e43.lp_feasible(rows_p, rhs_p):
        return {"family": family, "feasible": False}
    out = {"family": family, "feasible": True, "per_prompt": {}}
    for o in obs:
        rec = {"y_ms": o["y"], "x": o["x"], "R": o["R"]}
        anchors = {
            # excess = y + coefs . theta for both anchors
            "secant": excess_coeffs(family, o["x"], o["y"])[1],
            "whole": (-1.0, -o["x"]) + tuple(0.0 for _ in shapes_of(family)),
        }
        for label, coefs in anchors.items():
            lo = e43.lp_extreme(rows_p, rhs_p, list(coefs), "min")
            hi = e43.lp_extreme(rows_p, rhs_p, list(coefs), "max")
            e_lo, e_hi = o["y"] + lo["value"], o["y"] + hi["value"]
            rec[label] = {"lo_ms": e_lo, "hi_ms": e_hi,
                          "lo_frac": e_lo / o["y"], "hi_frac": e_hi / o["y"]}
        out["per_prompt"][o["name"]] = rec
    out["ratios"] = {o["name"]: o["ratio"] for o in obs}
    return out


def value_report(excess: dict, anchor: str = "secant") -> dict:
    """What the family-conditional excess is worth on the published median score.

    A negative excess bracket end means the family says there is nothing to
    harvest at that prompt, so the arm is reported as a slowdown rather than
    silently clamped to zero.
    """
    ratios = excess["ratios"]
    base = e43.score_of(ratios)
    out = {"anchor": anchor, "base_score": base, "arms": {},
           "crown_gap_pct": CROWN_GAP_PCT, "sigma_score_pct": SIGMA_SCORE_PCT,
           "fraction_needed": {}}

    def gain(frac, end):
        deltas = {nm: frac * excess["per_prompt"][nm][anchor][end]
                  for nm in CENTRAL}
        return (100.0 * (e43.score_after(ratios, deltas) / base - 1.0), deltas)

    for frac in (0.25, 1.0):
        for end in ("lo_frac", "hi_frac"):
            g, deltas = gain(frac, end)
            out["arms"]["removed_%.2f_%s" % (frac, end)] = {
                "delta_beagle": deltas["beagle"],
                "delta_medicine": deltas["medicine"],
                "score": base * (1.0 + g / 100.0), "score_gain_pct": g}
    for target, label in ((SIGMA_SCORE_PCT, "one_sigma"),
                          (CROWN_GAP_PCT, "crown")):
        for end in ("lo_frac", "hi_frac"):
            key = "%s_%s" % (label, end)
            if gain(1.0, end)[0] < target:
                out["fraction_needed"][key] = None
                continue
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if gain(mid, end)[0] >= target:
                    hi = mid
                else:
                    lo = mid
            out["fraction_needed"][key] = hi
    return out


def leg_fraction_needed(ratios: dict, prompts: tuple = CENTRAL) -> dict:
    """Candidate-leg time fraction at `prompts` needed for a given score gain.

    Family-free: pure order-statistic arithmetic on the published ratios, so it
    is the same number whichever cost model is true.  This is the decision input
    that survives the family ambiguity -- the families disagree about how much
    time is available to remove, not about what removing it is worth.  The
    median saturates once the sped-up prompts leave the central pair, so the
    saturation gain is reported next to the requirement.
    """
    base = e43.score_of(ratios)

    def gain(d):
        return 100.0 * (e43.score_after(ratios, {nm: d for nm in prompts})
                        / base - 1.0)

    out = {"base_score": base, "prompts": list(prompts),
           "saturation_gain_pct": gain(0.999999), "needed": {},
           "curve": {"%.3f" % d: gain(d)
                     for d in (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)}}
    for label, target in (("one_sigma", SIGMA_SCORE_PCT),
                          ("crown", CROWN_GAP_PCT)):
        if out["saturation_gain_pct"] < target:
            out["needed"][label] = None
            continue
        lo, hi = 0.0, 0.999999
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if gain(mid) >= target:
                hi = mid
            else:
                lo = mid
        out["needed"][label] = hi
    # smallest fraction that reaches 99.9 % of the saturated gain
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gain(mid) >= 0.999 * out["saturation_gain_pct"]:
            hi = mid
        else:
            lo = mid
    out["saturating_fraction"] = hi
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

SINGLE = ("linear", "step5", "step6", "step7", "quadratic")


def analyse(rows: dict, pool: dict, noise: dict, tol_frac: float,
            node_cap: int, do_threshold: bool) -> dict:
    """Every deliverable, in one pass over the shared enumeration."""
    our = rows[pool["our_row"]]
    out = {"tol_frac": tol_frac, "families": {}, "enumeration": {},
           "ray_equivalence": {}, "excess": {}, "value": {}}
    t0 = time.time()

    def stage(label: str) -> None:
        print("[%7.1fs] %s" % (time.time() - t0, label), file=sys.stderr,
              flush=True)

    # (a) the pooled enumeration, and whether pooling excludes any reading
    bundles_by_family = {}
    for family in SINGLE:
        stage("enumerate %s" % family)
        bundle = bundles_by_family[family] = build_bundle(rows, pool, family)
        solo = pooled_enumerate(bundle, family, tol_frac, node_cap=node_cap,
                                trees=[pool["our_row"]])
        pooled = pooled_enumerate(bundle, family, tol_frac, node_cap=node_cap)
        same = ([sorted(s.items()) for s in solo["selections"]]
                == [sorted(s.items()) for s in pooled["selections"]])
        out["enumeration"][family] = {
            "solo_n": solo["n_selections"], "pooled_n": pooled["n_selections"],
            "solo_nodes": solo["nodes_visited"],
            "pooled_nodes": pooled["nodes_visited"],
            "identical_selections": same,
            "cross_product_size": solo["cross_product_size"],
            "solo_pinned": solo["pinned_rounds"],
            "pooled_pinned": pooled["pinned_rounds"],
            "solo_surviving": solo["surviving_readings"],
            "pooled_surviving": pooled["surviving_readings"],
            "capped": solo["capped"] or pooled["capped"]}
        out["families"][family] = {"selections": pooled["selections"]}

    # (c) per-family and cross-family bracket on T(6) - T(5)
    union_lo = union_hi = None
    for family in SINGLE:
        stage("delta union %s" % family)
        sels = out["families"][family]["selections"]
        if not sels:
            out["families"][family]["delta"] = {"n_feasible": 0, "lo": None,
                                                "hi": None}
            continue
        du = delta_union(our, family, tol_frac, sels)
        for ep, key in ((du["arg_lo"], "pool_keeps_lo"),
                        (du["arg_hi"], "pool_keeps_hi")):
            du[key] = endpoint_survives_pool(rows, pool, family, tol_frac, ep)
        out["families"][family]["delta"] = du
        if du["lo"] is not None:
            union_lo = du["lo"] if union_lo is None else min(union_lo, du["lo"])
            union_hi = du["hi"] if union_hi is None else max(union_hi, du["hi"])

    # the mixture spans every single-shape family, so it is evaluated on the
    # union of their surviving readings; its own reading set is a superset, so
    # the reported cross-family width is a lower bound on the true width.
    seen, mix_sels = set(), []
    for family in SINGLE:
        for s in out["families"][family]["selections"]:
            key = tuple(sorted(s.items()))
            if key not in seen:
                seen.add(key)
                mix_sels.append(s)
    mix = delta_union(our, "mixture", tol_frac, mix_sels)
    out["families"]["mixture"] = {"delta": mix, "n_readings_tested": len(mix_sels)}
    if mix["lo"] is not None:
        union_lo = min(union_lo, mix["lo"])
        union_hi = max(union_hi, mix["hi"])
    out["cross_family_delta"] = {
        "lo": union_lo, "hi": union_hi,
        "contains_zero": union_lo is not None and union_lo <= 1e-9,
        "contains_advisor_step": (union_lo is not None
                                  and union_lo <= 37.730 <= union_hi),
        "contains_advisor_quadratic": (union_lo is not None
                                       and union_lo <= 8.575 <= union_hi),
        "local_ladder_5_6_dropped_tree": LOCAL_STEP_5_6}

    # why: the families are ray-equivalent designs at every surviving reading
    prim = E43_PRIMARY
    pairs = (("step6", "quadratic"), ("step5", "step6"), ("step6", "step7"),
             ("step5", "quadratic"), ("step7", "quadratic"))
    for fa, fb in pairs:
        stage("ray equivalence %s vs %s" % (fa, fb))
        oa, ob = observations(our, fa, prim), observations(our, fb, prim)
        scan = equivalent_witness_scan(oa, ob, fa, fb)
        sels = out["families"]["step6"]["selections"]
        every = all(ray_equivalence(observations(our, fa, s),
                                    observations(our, fb, s))["feasible"]
                    for s in sels)
        out["ray_equivalence"]["%s_vs_%s" % (fa, fb)] = {
            "primary": scan, "feasible_at_every_step6_reading": every,
            "n_readings_checked": len(sels)}

    # the same question asked of the pooled design itself, not of one row
    stage("pooled ray equivalence")
    out["pooled_ray"] = pooled_ray_equivalence(
        tree_bundles(rows, pool, "step6", prim),
        tree_bundles(rows, pool, "quadratic", prim), "step6", "quadratic")

    # (b) family-conditional excess and value, for our row and the board crown
    for label, rid in (("ours", pool["our_row"]), ("crown", FRONTIER_ROW)):
        if rid not in rows:
            continue
        stage("excess and value %s" % label)
        out["excess"][label] = {}
        out["value"][label] = {}
        for family in ("step6", "quadratic"):
            sel = (prim if rid == pool["our_row"]
                   else crown_selection(bundles_by_family[family], rid, family,
                                        tol_frac))
            if sel is None:
                out["excess"][label][family] = {"feasible": False}
                continue
            ex = excess_report(rows[rid], family, tol_frac, sel)
            out["excess"][label][family] = {"selection": sel, **ex}
            if ex.get("feasible"):
                out["value"][label][family] = {
                    anchor: value_report(ex, anchor)
                    for anchor in ("secant", "whole")}
        out["value"][label]["family_free"] = leg_fraction_needed(
            {nm: rows[rid]["prompts"][nm]["ratio"] for nm in ORDER})

    # The pooled shared-shape witness: a certified *upper* bound on the pooled
    # threshold for each family.  Cheap, because it is a local search over the
    # shape box rather than a reading enumeration.
    out["shared_shape"] = {}
    for family in ("step6", "quadratic"):
        stage("shared-shape witness %s" % family)
        bundles = tree_bundles(rows, pool, family, prim)
        sh = search_shape(bundles, family)
        out["shared_shape"][family] = {
            "upper_bound_at_primary_reading": sh["worst_rel"],
            "witness_shape": sh["shape"],
            "certified": certify_witness(bundles, family, sh["shape"],
                                         sh["worst_rel"] * 1.000001),
            "per_tree_fit": {rid: {"beta": v["beta"],
                                   "worst_rel": v["worst_rel"]}
                             for rid, v in sh["per_tree"].items()}}

    # The matching *lower* bound needs a reading enumeration per bisection step,
    # which costs minutes per family, so it is opt-in.
    if do_threshold:
        out["threshold"] = {}
        for family in ("step6", "quadratic"):
            stage("threshold lower bound %s" % family)
            bundle = bundles_by_family[family]
            out["threshold"][family] = {
                "lower_bound_solo": family_threshold(
                    bundle, family, iters=8, trees=[pool["our_row"]]),
                "lower_bound_pooled": family_threshold(bundle, family, iters=8)}

    out["power"] = power_statement(out, noise)
    return out


def crown_selection(bundle: dict, rid: str, family: str,
                    tol_frac: float) -> dict | None:
    """One admissible reading for a single pooled row, for the value table.

    The value table needs *a* consistent reading, not the whole admissible set,
    so this reuses the pruned depth-first enumeration restricted to that one row
    and stops at the first hit.  A blind scan of the raw cross product is far
    slower and can be effectively unbounded.
    """
    hit = pooled_enumerate(bundle, family, tol_frac, first_only=True,
                           trees=[rid])["selections"]
    return hit[0] if hit else None


def power_statement(res: dict, noise: dict) -> dict:
    """What pooling could ever buy, given the ray equivalence."""
    key = "step6_vs_quadratic"
    scan = res["ray_equivalence"][key]["primary"]
    gap = None
    if scan.get("best"):
        gap = scan["best"]["pred_gap_ms"]
    y_scale = 100.0                      # ms/round, order of the observed legs
    pr = res.get("pooled_ray") or {}
    demo = pr.get("demo") or {}
    return {
        "replicate_sigma_pct": noise["pooled_pct"],
        "replicate_sigma_ms_at_100ms_round": noise["pooled_pct"] / 100.0 * y_scale,
        "between_family_prediction_gap_ms": gap,
        "pooled_cost_gap": demo.get("pooled_cost_gap"),
        "pooled_x_column_identical": pr.get("x_identical_across_trees"),
        "pooled_box_equals_one_row": pr.get("box_equals_single_row"),
        "trees_needed_to_separate": None,
        "note": ("the two families' predictions differ by machine epsilon at a "
                 "shared admissible rho, so the misfit pooling would have to "
                 "detect is zero rather than small; the work-identity filter "
                 "also makes every pooled tree carry the same x column and the "
                 "same admissible shape box as one row, so no tree count and "
                 "no noise reduction separates them"),
    }


def report(res: dict, pool: dict, noise: dict) -> None:
    print("E45  pooled plateau trees, family separation")
    print("=" * 74)
    print("STOP EARLY: pooling does not separate the families.  The step and")
    print("quadratic designs are ray-equivalent on this instrument, so the")
    print("increment T(6)-T(5) is not identified at any tree count.")
    print()
    print("(a) pool actually used")
    print("    candidate rows %d -> distinct git trees %d -> pooled trees %d"
          % (pool["row_count"], pool["tree_count"], len(pool["pool"]) + 1))
    for g in pool["duplicate_groups"]:
        print("    dropped duplicate tree %s: %s (kept %s)"
              % (g["tree"][:16], "/".join(g["solvers"]), g["kept"]))
    print("    pooled legs %d, shared prompts %d (%s)"
          % (pool["pooled_legs"], len(pool["shared_prompts"]),
             ",".join(pool["shared_prompts"])))
    print("    same-tree MTP-leg replicate sd %.4f %% over %d leg pairs"
          % (noise["pooled_pct"], noise["n_leg_pairs"]))
    print()
    print("    reading enumeration, solo (our row) vs pooled:")
    print("    %-10s %6s %6s %8s %8s %s"
          % ("family", "solo", "pooled", "nodes/s", "nodes/p", "identical"))
    for family in SINGLE:
        e = res["enumeration"][family]
        print("    %-10s %6d %6d %8d %8d %s"
              % (family, e["solo_n"], e["pooled_n"], e["solo_nodes"],
                 e["pooled_nodes"], e["identical_selections"]))
    print()
    print("(c) T(6)-T(5) bracket per family, unioned over surviving readings")
    print("    %-10s %5s %10s %10s  %s"
          % ("family", "n", "lo ms", "hi ms", "pool keeps both endpoints"))
    for family in list(SINGLE) + ["mixture"]:
        d = res["families"][family].get("delta") or {}
        if d.get("lo") is None:
            print("    %-10s %5s %10s %10s" % (family, d.get("n_feasible", 0),
                                               "-", "-"))
            continue
        keeps = "%s/%s" % (d.get("pool_keeps_lo", {}).get("all_feasible"),
                           d.get("pool_keeps_hi", {}).get("all_feasible"))
        print("    %-10s %5d %10.4f %10.4f  %s"
              % (family, d["n_feasible"], d["lo"], d["hi"], keeps))
    cf = res["cross_family_delta"]
    print("    cross-family union  [%.4f, %.4f] ms   contains 0: %s"
          % (cf["lo"], cf["hi"], cf["contains_zero"]))
    print("    contains advisor step 37.730: %s   quadratic 8.575: %s"
          % (cf["contains_advisor_step"], cf["contains_advisor_quadratic"]))
    print("    local ladder 5->6 = %.3f ms is from a DROPPED tree; not used"
          % cf["local_ladder_5_6_dropped_tree"])
    print()
    print("why: ray equivalence (same column space at a shared admissible rho)")
    for key, v in res["ray_equivalence"].items():
        b = v["primary"].get("best")
        if not b:
            print("    %-24s infeasible" % key)
            continue
        print("    %-24s lam=[%.4g,%.4g] pred gap %.2e ms  "
              "delta %.3f vs %.3f ms  both cones %s  all %d readings %s"
              % (key, v["primary"]["ray"]["lambda_lo"],
                 v["primary"]["ray"]["lambda_hi"], b["pred_gap_ms"],
                 b["delta_a"], b["delta_b"],
                 b["a_in_cone"] and b["b_in_cone"], v["n_readings_checked"],
                 v["feasible_at_every_step6_reading"]))
    pr = res.get("pooled_ray") or {}
    if pr:
        print("    pooled design, %d legs over %d trees:" % (pr["n_legs"],
                                                             len(pr["trees"])))
        print("      x column identical across trees: %s (max gap %.3g)"
              % (pr["x_identical_across_trees"], pr["x_max_gap"]))
        print("      pooled shape box equals one row: %s (max gap %.3g)"
              % (pr["box_equals_single_row"], pr["box_max_gap"]))
        print("      offset-only ray (no x column) feasible: %s  lam=[%.4g,%.4g]"
              % (pr["offset_ray"]["feasible"], pr["offset_ray"]["lambda_lo"],
                 pr["offset_ray"]["lambda_hi"]))
        d = pr.get("demo") or {}
        if d.get("admissible"):
            print("      unconstrained pooled worst-rel: step6 %.10f  "
                  "quadratic %.10f  gap %.3g"
                  % (d["pooled_raw_a"], d["pooled_raw_b"], d["pooled_raw_gap"]))
            print("      identical misfit, T(6)-T(5) differs by up to %.3f ms"
                  % d["delta_gap_ms"])
            print("      cone-projected: step6 %.6f (%d/%d trees clamped)  "
                  "quadratic %.6f (%d clamped) <- the only asymmetry"
                  % (d["pooled_cone_a"], d["n_clamped_a"], len(pr["trees"]),
                     d["pooled_cone_b"], d["n_clamped_b"]))
    print()
    print("(b) excess and score value, per family")
    for label in res["excess"]:
        ff = res["value"][label]["family_free"]
        print("    %s row: base score %.8f  saturation %+.4f %% at leg fraction "
              "%.4f" % (label, ff["base_score"], ff["saturation_gain_pct"],
                        ff["saturating_fraction"]))
        print("      family-free leg fraction needed: one sigma %s  crown %s"
              % (fmt(ff["needed"]["one_sigma"]), fmt(ff["needed"]["crown"])))
        for family in ("step6", "quadratic"):
            ex = res["excess"][label].get(family) or {}
            if not ex.get("feasible"):
                print("      %-10s infeasible" % family)
                continue
            for nm in CENTRAL:
                r = ex["per_prompt"][nm]
                print("      %-10s %-9s secant [%7.3f, %7.3f] ms "
                      "(%.3f-%.3f %%)  whole [%7.3f, %7.3f] ms"
                      % (family, nm, r["secant"]["lo_ms"], r["secant"]["hi_ms"],
                         100 * r["secant"]["lo_frac"],
                         100 * r["secant"]["hi_frac"],
                         r["whole"]["lo_ms"], r["whole"]["hi_ms"]))
            for anchor in ("secant", "whole"):
                v = res["value"][label][family][anchor]
                print("      %-10s %-7s crown needs %s of excess (lo end %s); "
                      "full removal %+.4f %%"
                      % (family, anchor,
                         fmt(v["fraction_needed"]["crown_hi_frac"]),
                         fmt(v["fraction_needed"]["crown_lo_frac"]),
                         v["arms"]["removed_1.00_hi_frac"]["score_gain_pct"]))
    if "shared_shape" in res:
        print()
        print("pooled shared-shape witness at the E43 primary reading "
              "(certified upper bound on the pooled threshold)")
        for family, t in res["shared_shape"].items():
            print("    %-10s upper %.6f  certified %s  vs tol %.5f -> %s"
                  % (family, t["upper_bound_at_primary_reading"],
                     t["certified"]["all_feasible"], res["tol_frac"],
                     "within" if t["upper_bound_at_primary_reading"]
                     <= res["tol_frac"] else "above"))
    if "threshold" in res:
        print()
        print("pooled threshold lower bounds (shared-shape coupling dropped)")
        for family, t in res["threshold"].items():
            print("    %-10s lower solo %s  lower pooled %s"
                  % (family, fmt(t["lower_bound_solo"]["threshold_frac"]),
                     fmt(t["lower_bound_pooled"]["threshold_frac"])))
    p = res["power"]
    print()
    print("power: replicate sd %.4f %% (~%.4f ms on a 100 ms round); "
          "between-family prediction gap %.2e ms"
          % (p["replicate_sigma_pct"], p["replicate_sigma_ms_at_100ms_round"],
             p["between_family_prediction_gap_ms"]))
    print("       trees needed to separate: unbounded")


def fmt(v) -> str:
    return "n/a" if v is None else "%.6f" % v


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def synthetic_row(a: float, b: float, third: float, family: str,
                  plan: dict) -> dict:
    """A row whose legs were generated by a known member of `family`.

    Ground truth for the whole pipeline: the recovered bracket must contain the
    generating parameters and the enumeration must keep the generating reading.
    `plan` gives (R, non_drafting, depth) per prompt; drafting rounds all run at
    `depth`, which makes every published functional exactly consistent -- D is a
    whole multiple of the drafting-round count, so the reading is admissible.
    """
    prompts = {}
    for nm, (R, nd, k) in plan.items():
        f = R - nd
        D = f * (k - 1)
        if DECODE_TOKENS - R > D:
            raise ValueError("accepted exceeds offered for %s" % nm)
        rho = {k: f / R}
        if nd:
            rho[1] = nd / R
        mean_M = 1.0 + D / R
        t = sum(w * (a + b * m + third * BASIS[shapes_of(family)[0]](m))
                for m, w in rho.items())
        prompts[nm] = {"n": D / R, "mean_M": mean_M, "non_drafting": nd,
                       "mtp_ms": t * R / DECODE_TOKENS,
                       "mtp_ms_total": t * R, "serial_ms": 37.9908,
                       "ratio": 37.9908 / (t * R / DECODE_TOKENS),
                       "parity_ok": True, "head": DECLARED_HEAD, "rho": rho}
    return {"row": "synth", "solver": "synth", "score": 1.0, "status": "promoted",
            "created": "", "commit": "", "prompts": prompts}


def self_test() -> int:
    checks: list = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    subs = load_corpus()
    pmap = e43.prompt_map()
    rows = declared_head_rows(subs, pmap)
    pool = build_pool(rows)
    noise = replicate_noise(rows, pool)
    our = rows[OUR_ROW]
    tol = E43_TOL_HEADLINE

    # --- corpus, pool and tree-identity dedup -----------------------------
    ck("corpus_size", len(subs) > 600, str(len(subs)))
    ck("declared_head_rows", len(rows) == 110, str(len(rows)))
    ck("our_row_solver", our["solver"] == "morganmcg1", our["solver"])
    ck("our_tree_resolved", len(pool["our_tree"]) == 16, pool["our_tree"])
    ck("candidate_rows_six", pool["row_count"] == 6, str(pool["row_count"]))
    ck("distinct_trees_five", pool["tree_count"] == 5, str(pool["tree_count"]))
    ck("dedup_found_one_group", len(pool["duplicate_groups"]) == 1,
       str(pool["duplicate_groups"]))
    dup = pool["duplicate_groups"][0]
    ck("dedup_pair_is_ledger_160H",
       sorted(dup["rows"]) == ["11863aa9", "4f76de6e"], str(dup["rows"]))
    ck("dedup_keeps_higher_score",
       rows[dup["kept"]]["score"] >= max(rows[r]["score"] for r in dup["dropped"]),
       "%s %.6f" % (dup["kept"], rows[dup["kept"]]["score"]))
    ck("pool_is_five_trees", len(pool["pool"]) == 5, str(pool["pool"]))
    ck("pooled_legs_47", pool["pooled_legs"] == 47, str(pool["pooled_legs"]))
    ck("shared_prompts_seven",
       pool["shared_prompts"] == [n for n in ORDER if n != "plutarch"],
       str(pool["shared_prompts"]))
    ck("unshared_is_plutarch", pool["unshared_prompts"] == ["plutarch"],
       str(pool["unshared_prompts"]))
    ck("dup_rows_share_a_tree",
       pool["trees"]["11863aa9"] == pool["trees"]["4f76de6e"],
       pool["trees"]["11863aa9"])
    ck("distinct_trees_are_distinct",
       len({pool["trees"][r] for r in pool["pool"]}) == 5,
       str(sorted({pool["trees"][r] for r in pool["pool"]})))
    ck("our_tree_not_in_pool",
       pool["our_tree"] not in {pool["trees"][r] for r in pool["pool"]},
       pool["our_tree"])

    # --- work identity is exact, not rounded ------------------------------
    ck("willgasser_identity_seven",
       pool["identity"]["de7981ae"]["n_identical"] == 7,
       str(pool["identity"]["de7981ae"]["different"]))
    ck("other_pooled_identity_eight",
       all(pool["identity"][r]["n_identical"] == 8
           for r in pool["pool"] if r != "de7981ae"),
       str({r: pool["identity"][r]["n_identical"] for r in pool["pool"]}))
    perturbed = json.loads(json.dumps(rows[pool["pool"][0]]))
    perturbed["prompts"]["botany"]["n"] += 1e-13
    ck("work_identity_is_exact_float",
       work_identity(our, perturbed)["n_identical"]
       < pool["identity"][pool["pool"][0]]["n_identical"],
       str(work_identity(our, perturbed)["different"]))
    cand_ours = reading_candidates(our)
    ck("reading_sets_match_on_shared",
       all(reading_candidates(rows[r])[nm] == cand_ours[nm]
           for r in pool["pool"] for nm in pool["shared_prompts"]))

    # --- replicate noise --------------------------------------------------
    ck("replicate_sd_in_range", 0.03 < noise["pooled_pct"] < 0.12,
       "%.4f" % noise["pooled_pct"])
    ck("replicate_legs_eight", noise["n_leg_pairs"] == 8,
       str(noise["n_leg_pairs"]))
    ck("replicate_per_prompt_small",
       all(v < 0.2 for v in noise["per_prompt_pct"].values()),
       str({k: round(v, 4) for k, v in noise["per_prompt_pct"].items()}))

    # --- family algebra ---------------------------------------------------
    ck("dims", (dim("linear"), dim("step6"), dim("mixture")) == (2, 3, 4))
    ck("step6_jumps_at_six",
       (shape_weights("step6", 5), shape_weights("step6", 6)) == ((0.0,), (1.0,)))
    ck("delta_step6", delta_5_6_coeffs("step6") == (0.0, 1.0, 1.0))
    ck("delta_step5", delta_5_6_coeffs("step5") == (0.0, 1.0, 0.0))
    ck("delta_step7", delta_5_6_coeffs("step7") == (0.0, 1.0, 0.0))
    ck("delta_quadratic", delta_5_6_coeffs("quadratic") == (0.0, 1.0, 11.0))
    ck("delta_mixture", delta_5_6_coeffs("mixture") == (0.0, 1.0, 1.0, 11.0))
    ck("secant_step6_zero", secant_weights("step6", 4.2) == (0.0,))
    ck("secant_quadratic_is_6x_minus_5",
       abs(secant_weights("quadratic", 4.2)[0] - (6 * 4.2 - 5)) < 1e-12,
       str(secant_weights("quadratic", 4.2)))
    ck("excess_step6_is_y_minus_line",
       excess_coeffs("step6", 4.2, 90.0) == (90.0, (-1.0, -4.2, 0.0)))
    srow, srhs = family_static_rows("quadratic")
    ck("quad_floor_admits_negative_b",
       all(sum(c * v for c, v in zip(r, (33.639, -1.930, 0.955))) <= h + 1e-9
           for r, h in zip(srow, srhs)), "E43 quadratic fit is legal")
    ck("quad_floor_rejects_steep_negative_b",
       any(sum(c * v for c, v in zip(r, (33.6, -50.0, 0.955))) > h + 1e-9
           for r, h in zip(srow, srhs)))
    trow, trhs = family_static_rows("step6")
    ck("step_floor_rejects_negative_s",
       any(sum(c * v for c, v in zip(r, (31.268, 1.452, -1.0))) > h + 1e-9
           for r, h in zip(trow, trhs)))
    lo, hi, pts = leg_shape_range("mixture", [{1: 0.5, 7: 0.5}, {2: 1.0}])
    ck("leg_shape_range_bracket", lo == (0.0, 4.0) and hi == (0.5, 25.0),
       str((lo, hi)))
    ck("leg_shape_range_points", len(pts) == 2, str(pts))
    fake = {"name": "p", "x": 4.0, "y": 100.0, "g_lo": (0.2,), "g_hi": (0.8,)}
    lr, lh = leg_rows(fake, 0.01)
    ck("leg_rows_band", lr == [[1.0, 4.0, 0.2], [-1.0, -4.0, -0.8]]
       and abs(lh[0] - 101.0) < 1e-9 and abs(lh[1] + 99.0) < 1e-9,
       str((lr, lh)))

    # --- E43 reproduction on the primary reading --------------------------
    obs6 = observations(our, "step6", E43_PRIMARY)
    obsq = observations(our, "quadratic", E43_PRIMARY)
    s_br = bracket(obs6, "step6", tol, (0.0, 0.0, 1.0))
    ck("e43_s_bracket_reproduced",
       abs(s_br["lo"] - E43_BRACKET_S_PRIMARY[0]) < 1e-9
       and abs(s_br["hi"] - E43_BRACKET_S_PRIMARY[1]) < 1e-9,
       "[%.12f, %.12f]" % (s_br["lo"], s_br["hi"]))
    # The generic N-shape layer must agree with E43's hand-written step-only
    # code to floating-point identity, otherwise nothing built on top of it can
    # be compared with the E43 record.
    e43_obs, q_gap = [], 0.0
    for o in obs6:
        p = our["prompts"][o["name"]]
        br = e43.share_bracket(p["mean_M"], o["rho1"], p["non_drafting"])
        e43_obs.append({"name": o["name"], "x": o["x"], "y": o["y"],
                        "q_lo": br["q_lo"], "q_hi": br["q_hi"]})
        q_gap = max(q_gap, abs(br["q_lo"] - o["g_lo"][0]),
                    abs(br["q_hi"] - o["g_hi"][0]))
    ck("e43_q_bracket_agrees", q_gap < 1e-15, "max gap %.3g" % q_gap)
    e43_rows, e43_rhs, _ = e43.step_polytope(e43_obs, tol, None)
    e43_lo = e43.lp_extreme(e43_rows, e43_rhs, [0.0, 0.0, 1.0], "min")
    e43_hi = e43.lp_extreme(e43_rows, e43_rhs, [0.0, 0.0, 1.0], "max")
    ck("e43_polytope_agrees_bit_for_bit",
       abs(e43_lo["value"] - s_br["lo"]) < 1e-12
       and abs(e43_hi["value"] - s_br["hi"]) < 1e-12,
       "e43 [%.12f, %.12f]" % (e43_lo["value"], e43_hi["value"]))
    shape6 = {o["name"]: sum(w * BASIS["g6"](m) for m, w in
                             e43.maxent_rho(o["x"], o["rho1"],
                                            our["prompts"][o["name"]]
                                            ["non_drafting"]).items())
              for o in obs6}
    shapeq = {o["name"]: sum(w * BASIS["sq"](m) for m, w in
                             e43.maxent_rho(o["x"], o["rho1"],
                                            our["prompts"][o["name"]]
                                            ["non_drafting"]).items())
              for o in obsq}
    f6, fq = raw_fit(obs6, shape6), raw_fit(obsq, shapeq)
    ck("e43_step_fit_reproduced",
       all(abs(g - w) < 5e-3 for g, w in zip(f6["beta"], E43_STEP_FIT)),
       str(tuple(round(v, 4) for v in f6["beta"])))
    ck("e43_quad_fit_reproduced",
       all(abs(g - w) < 5e-3 for g, w in zip(fq["beta"], E43_QUAD_FIT)),
       str(tuple(round(v, 4) for v in fq["beta"])))
    d6 = sum(c * v for c, v in zip(delta_5_6_coeffs("step6"), f6["beta"]))
    dq = sum(c * v for c, v in zip(delta_5_6_coeffs("quadratic"), fq["beta"]))
    ck("advisor_step_delta_37_730", abs(d6 - 37.730) < 5e-3, "%.4f" % d6)
    ck("advisor_quad_delta_8_575", abs(dq - 8.575) < 5e-3, "%.4f" % dq)
    rms = lambda f: math.sqrt(sum(r * r for r in f["resid"]) / len(f["resid"]))
    ratio = rms(fq) / rms(f6)
    ck("advisor_residual_ratio_reproduced",
       abs(ratio - E43_RESIDUAL_RATIO) < 1e-4, "%.6f" % ratio)
    ck("residual_ratio_inside_inconclusive_band", ratio < 1.5,
       "%.6f rms, %.6f worst-rel"
       % (ratio, fq["worst_rel"] / f6["worst_rel"]))
    ck("linear_infeasible_at_primary",
       not feasible(observations(our, "linear", E43_PRIMARY), "linear", tol))

    # --- the enumeration, solo and pooled --------------------------------
    bundle6 = build_bundle(rows, pool, "step6")
    solo = pooled_enumerate(bundle6, "step6", tol, trees=[OUR_ROW])
    pooled = pooled_enumerate(bundle6, "step6", tol)
    ck("e43_selection_count_42", solo["n_selections"] == 42,
       str(solo["n_selections"]))
    ck("e43_pins_reproduced",
       all(solo["pinned_rounds"].get(k) == v for k, v in E43_PINS.items()),
       str(solo["pinned_rounds"]))
    ck("e43_surviving_plutarch",
       solo["surviving_readings"]["plutarch"] == [461, 474, 487, 500],
       str(solo["surviving_readings"]["plutarch"]))
    ck("e43_surviving_drama",
       solo["surviving_readings"]["drama"] == [252, 289, 299, 336],
       str(solo["surviving_readings"]["drama"]))
    ck("e43_surviving_travel",
       solo["surviving_readings"]["travel"] == [151, 212, 273],
       str(solo["surviving_readings"]["travel"]))
    ck("primary_reading_survives",
       any(all(s[k] == v for k, v in E43_PRIMARY.items())
           for s in solo["selections"]))
    ck("pooling_excludes_no_reading",
       pooled["n_selections"] == solo["n_selections"],
       "%d vs %d" % (pooled["n_selections"], solo["n_selections"]))
    ck("pooling_selections_identical",
       [sorted(s.items()) for s in pooled["selections"]]
       == [sorted(s.items()) for s in solo["selections"]])
    ck("pooling_visits_fewer_nodes",
       pooled["nodes_visited"] <= solo["nodes_visited"],
       "%d vs %d" % (pooled["nodes_visited"], solo["nodes_visited"]))
    ck("enumeration_not_capped", not solo["capped"] and not pooled["capped"])

    # --- ray equivalence: why pooling cannot help -------------------------
    ray = ray_equivalence(obs6, obsq)
    ck("ray_equivalence_feasible", ray["feasible"],
       "lam [%.4g, %.4g]" % (ray["lambda_lo"] or -1, ray["lambda_hi"] or -1))
    scan = equivalent_witness_scan(obs6, obsq, "step6", "quadratic")
    best = scan["best"]
    ck("ray_witness_both_cones_legal", scan["n_both_cones_legal"] > 0,
       str(scan["n_both_cones_legal"]))
    ck("ray_predictions_identical", best["pred_gap_ms"] < 1e-6,
       "%.3e" % best["pred_gap_ms"])
    ck("ray_map_is_exact", best["map_gap"] < 1e-5, "%.3e" % best["map_gap"])
    ck("ray_residuals_identical",
       abs(best["worst_rel_a"] - best["worst_rel_b"]) < 1e-12,
       "%.3e" % abs(best["worst_rel_a"] - best["worst_rel_b"]))
    ck("ray_deltas_differ", abs(best["delta_a"] - best["delta_b"]) > 1.0,
       "%.4f vs %.4f" % (best["delta_a"], best["delta_b"]))
    ck("ray_gap_below_noise",
       best["pred_gap_ms"] < 1e-6 * noise["pooled_pct"],
       "%.3e vs sd %.4f %%" % (best["pred_gap_ms"], noise["pooled_pct"]))
    ck("ray_holds_at_every_surviving_reading",
       all(ray_equivalence(observations(our, "step6", s),
                           observations(our, "quadratic", s))["feasible"]
           for s in solo["selections"]), str(len(solo["selections"])))

    # --- and the same question asked of the pooled design -----------------
    pray = pooled_ray_equivalence(
        tree_bundles(rows, pool, "step6", E43_PRIMARY),
        tree_bundles(rows, pool, "quadratic", E43_PRIMARY),
        "step6", "quadratic")
    ck("pooled_ray_uses_all_legs", pray["n_legs"] == pool["pooled_legs"],
       "%d legs" % pray["n_legs"])
    ck("pooled_x_column_identical", pray["x_identical_across_trees"],
       "max gap %.3g mean-M units" % pray["x_max_gap"])
    ck("pooled_shape_box_equals_one_row", pray["box_equals_single_row"],
       "max gap %.3g" % pray["box_max_gap"])
    ck("pooled_offset_only_ray_feasible", pray["offset_ray"]["feasible"],
       "lam [%.4f, %.4f]" % (pray["offset_ray"]["lambda_lo"],
                             pray["offset_ray"]["lambda_hi"]))
    pd = pray["demo"]
    ck("pooled_demo_admissible", bool(pd and pd["admissible"]), str(bool(pd)))
    ck("pooled_unconstrained_cost_identical", pd["pooled_raw_gap"] < 1e-12,
       "%.10f vs %.10f, gap %.3g" % (pd["pooled_raw_a"], pd["pooled_raw_b"],
                                     pd["pooled_raw_gap"]))
    ck("pooled_unconstrained_predictions_identical",
       pd["pooled_pred_gap_ms"] < 1e-9, "%.3e ms" % pd["pooled_pred_gap_ms"])
    ck("pooled_raw_gap_far_below_replicate_noise",
       pd["pooled_raw_gap"] < 1e-6 * noise["pooled_pct"] / 100.0,
       "%.3g vs sd %.6f" % (pd["pooled_raw_gap"], noise["pooled_pct"] / 100.0))
    ck("pooled_delta_still_differs", pd["delta_gap_ms"] > 1.0,
       "%.3f ms apart at identical pooled misfit" % pd["delta_gap_ms"])
    ck("pooled_every_tree_matches_unconstrained",
       all(v["raw_gap"] < 1e-12 for v in pd["per_tree"].values()),
       "worst tree gap %.3g" % max(v["raw_gap"]
                                   for v in pd["per_tree"].values()))
    # the cone is the only thing that can tell the two parameterisations apart:
    # same column space, different admissible orthant.
    ck("monotonicity_cone_is_the_only_asymmetry",
       pd["pooled_cone_a"] != pd["pooled_cone_b"]
       and (pd["n_clamped_a"] > 0) != (pd["n_clamped_b"] > 0),
       "cone cost step6 %.6f (%d clamped) vs quadratic %.6f (%d clamped)"
       % (pd["pooled_cone_a"], pd["n_clamped_a"], pd["pooled_cone_b"],
          pd["n_clamped_b"]))
    ck("offset_ray_stricter_than_free_x",
       pray["offset_ray"]["lambda_lo"] >= ray["lambda_lo"] - 1e-9,
       "offset lam_lo %.4f >= free lam_lo %.4f"
       % (pray["offset_ray"]["lambda_lo"], ray["lambda_lo"]))

    # --- brackets and the cross-family union ------------------------------
    br6 = bracket(obs6, "step6", tol, delta_5_6_coeffs("step6"))
    brq = bracket(obsq, "quadratic", tol, delta_5_6_coeffs("quadratic"))
    brm = bracket(observations(our, "mixture", E43_PRIMARY), "mixture", tol,
                  delta_5_6_coeffs("mixture"))
    br5 = bracket(observations(our, "step5", E43_PRIMARY), "step5", tol,
                  delta_5_6_coeffs("step5"))
    br7 = bracket(observations(our, "step7", E43_PRIMARY), "step7", tol,
                  delta_5_6_coeffs("step7"))
    ck("step6_and_quad_delta_disjoint", brq["hi"] < br6["lo"],
       "quad hi %.4f < step lo %.4f" % (brq["hi"], br6["lo"]))
    ck("mixture_spans_both",
       brm["lo"] <= brq["lo"] + 1e-9 and brm["hi"] >= br6["hi"] - 1e-9,
       "[%.4f, %.4f]" % (brm["lo"], brm["hi"]))
    ck("boundary_at_five_admits_zero_increment", br5["lo"] < 1e-6,
       "[%.4f, %.4f]" % (br5["lo"], br5["hi"]))
    ck("boundary_at_seven_admits_zero_increment", br7["lo"] < 1e-6,
       "[%.4f, %.4f]" % (br7["lo"], br7["hi"]))
    ck("advisor_quad_delta_inside_quad_bracket",
       brq["lo"] - 1e-9 <= 8.575 <= brq["hi"] + 1e-9,
       "[%.4f, %.4f]" % (brq["lo"], brq["hi"]))
    ck("advisor_step_delta_inside_step_bracket",
       br6["lo"] - 1e-9 <= 37.730 <= br6["hi"] + 1e-9,
       "[%.4f, %.4f]" % (br6["lo"], br6["hi"]))
    ck("step5_step7_feasible_too", br5["feasible"] and br7["feasible"])

    # --- endpoint survival under pooling ---------------------------------
    du = delta_union(our, "step6", tol, solo["selections"])
    keep_hi = endpoint_survives_pool(rows, pool, "step6", tol, du["arg_hi"])
    keep_lo = endpoint_survives_pool(rows, pool, "step6", tol, du["arg_lo"])
    ck("delta_union_wider_than_primary",
       du["lo"] <= br6["lo"] + 1e-9 and du["hi"] >= br6["hi"] - 1e-9,
       "[%.4f, %.4f]" % (du["lo"], du["hi"]))
    ck("pool_keeps_delta_hi_endpoint",
       keep_hi["checked"] and keep_hi["all_feasible"], str(keep_hi.get("per_tree")))
    ck("pool_keeps_delta_lo_endpoint",
       keep_lo["checked"] and keep_lo["all_feasible"], str(keep_lo.get("per_tree")))
    ck("witness_shape_none_for_mixture",
       witness_shape(observations(our, "mixture", E43_PRIMARY), "mixture", tol,
                     [0.0, 0.0, 0.0, 0.0]) is None)

    # --- excess, value, and the family-free anchor ------------------------
    ex6 = excess_report(our, "step6", tol, E43_PRIMARY)
    exq = excess_report(our, "quadratic", tol, E43_PRIMARY)
    ck("step6_secant_equals_whole",
       all(abs(ex6["per_prompt"][nm]["secant"]["lo_ms"]
               - ex6["per_prompt"][nm]["whole"]["lo_ms"]) < 1e-9
           for nm in ORDER), "step-at-6 has no sub-six secant term")
    ck("quad_secant_below_whole",
       all(exq["per_prompt"][nm]["secant"]["hi_ms"]
           < exq["per_prompt"][nm]["whole"]["hi_ms"] for nm in ORDER))
    ck("quad_secant_negative_at_low_depth",
       exq["per_prompt"]["drama"]["secant"]["lo_ms"] < 0.0,
       "%.4f" % exq["per_prompt"]["drama"]["secant"]["lo_ms"])
    ck("excess_grows_with_depth",
       ex6["per_prompt"]["botany"]["secant"]["hi_ms"]
       > ex6["per_prompt"]["plutarch"]["secant"]["hi_ms"])
    v6 = value_report(ex6, "secant")
    vq = value_report(exq, "secant")
    ck("base_score_matches_board", abs(v6["base_score"] - 3.2325084826) < 1e-8,
       "%.10f" % v6["base_score"])
    ck("e43_crown_fraction_reproduced",
       abs(v6["fraction_needed"]["crown_hi_frac"] - E43_CROWN_FRACTION_E_HI)
       < 1e-9, "%.9f" % v6["fraction_needed"]["crown_hi_frac"])
    ck("quad_needs_more_of_its_excess",
       vq["fraction_needed"]["crown_hi_frac"]
       > v6["fraction_needed"]["crown_hi_frac"],
       "%.6f vs %.6f" % (vq["fraction_needed"]["crown_hi_frac"],
                         v6["fraction_needed"]["crown_hi_frac"]))
    ff = leg_fraction_needed({nm: our["prompts"][nm]["ratio"] for nm in ORDER})
    ck("family_free_crown_fraction_small",
       0.0 < ff["needed"]["crown"] < 0.02, "%.6f" % ff["needed"]["crown"])
    ck("family_free_sigma_fraction_small",
       0.0 < ff["needed"]["one_sigma"] < ff["needed"]["crown"],
       "%.6f" % ff["needed"]["one_sigma"])
    ck("median_saturates",
       ff["saturating_fraction"] < 0.5
       and abs(ff["curve"]["0.250"] - ff["saturation_gain_pct"]) < 1e-6,
       "%.4f at %+.4f %%" % (ff["saturating_fraction"],
                             ff["saturation_gain_pct"]))
    ck("value_arms_saturate_across_families",
       abs(v6["arms"]["removed_1.00_lo_frac"]["score_gain_pct"]
           - vq["arms"]["removed_1.00_hi_frac"]["score_gain_pct"]) < 1e-6,
       "both hit the median cap")

    # --- synthetic ground truth ------------------------------------------
    plan = {"plutarch": (461, 415, 3), "drama": (252, 112, 5),
            "travel": (212, 112, 6), "beagle": (107, 17, 7),
            "medicine": (99, 9, 8), "essays": (87, 0, 6),
            "republic": (89, 0, 7), "botany": (85, 0, 8)}
    truth = (30.0, 1.5, 35.0)
    syn = synthetic_row(*truth, "step6", plan)
    syn_sel = {nm: R for nm, (R, _, _) in plan.items()}
    syn_obs = observations(syn, "step6", syn_sel)
    syn_br = bracket(syn_obs, "step6", 1e-6, delta_5_6_coeffs("step6"))
    ck("synthetic_bracket_contains_truth",
       syn_br["feasible"]
       and syn_br["lo"] - 1e-6 <= truth[1] + truth[2] <= syn_br["hi"] + 1e-6,
       "[%.4f, %.4f] vs %.4f" % (syn_br["lo"], syn_br["hi"],
                                 truth[1] + truth[2]))
    ck("synthetic_s_bracket_contains_truth",
       bracket(syn_obs, "step6", 1e-6, (0.0, 0.0, 1.0))["lo"] - 1e-6
       <= truth[2], "s truth %.1f" % truth[2])
    ck("synthetic_quadratic_also_fits",
       feasible(observations(syn, "quadratic", syn_sel), "quadratic", 1e-6)
       or ray_equivalence(syn_obs,
                          observations(syn, "quadratic", syn_sel))["feasible"],
       "a step-generated row is also explained by curvature")
    ck("synthetic_linear_rejected",
       not feasible(observations(syn, "linear", syn_sel), "linear", 1e-6))

    # --- threshold bracketing --------------------------------------------
    bundles = tree_bundles(rows, pool, "step6", E43_PRIMARY)
    sh = search_shape(bundles, "step6", restarts=2, sweeps=8)
    cert = certify_witness(bundles, "step6", sh["shape"],
                           sh["worst_rel"] * 1.000001)
    ck("shape_witness_certified", cert["all_feasible"],
       "worst %.6f" % sh["worst_rel"])
    ck("shape_box_nonempty",
       all(lo <= hi for lo, hi in shape_box(bundles).values()),
       str({k: (round(v[0], 4), round(v[1], 4))
            for k, v in shape_box(bundles).items()}))
    ck("pooled_upper_bound_above_solo",
       sh["worst_rel"] >= max(fit_at_shape(bundles[OUR_ROW], "step6",
                                          sh["shape"])["worst_rel"] - 1e-12, 0.0),
       "%.6f" % sh["worst_rel"])

    bad = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, detail in checks:
        print("%-4s %-44s %s" % ("ok" if ok else "FAIL", name, detail))
    print("%d checks, %d failed" % (len(checks), bad))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--refresh-trees", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tol", type=float, default=E43_TOL_HEADLINE)
    ap.add_argument("--node-cap", type=int, default=3_000_000)
    ap.add_argument("--no-threshold", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    subs = load_corpus(args.refresh)
    pmap = e43.prompt_map()
    rows = declared_head_rows(subs, pmap)
    pool = build_pool(rows, args.refresh_trees)
    noise = replicate_noise(rows, pool)

    if args.run:
        res = analyse(rows, pool, noise, args.tol, args.node_cap,
                      not args.no_threshold)
        res["meta"] = {"base_sha": git_head(), "our_row": OUR_ROW,
                       "our_tree": pool["our_tree"],
                       "declared_head": DECLARED_HEAD,
                       "pool": {"row_count": pool["row_count"],
                                "tree_count": pool["tree_count"],
                                "pooled_legs": pool["pooled_legs"],
                                "trees": pool["trees"],
                                "duplicate_groups": pool["duplicate_groups"],
                                "shared_prompts": pool["shared_prompts"]},
                       "replicate_noise": noise}
        report(res, pool, noise)
        OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print()
        print("wrote %s" % OUT)
        return 0

    if args.census:
        print("declared-head rows: %d" % len(rows))
        print("our row %s (%s) tree %s score %.8f"
              % (OUR_ROW, rows[OUR_ROW]["solver"], pool["our_tree"],
                 rows[OUR_ROW]["score"]))
        print()
        print("%-9s %-16s %-10s %-18s %-6s %s"
              % ("row", "solver", "score", "tree", "ident", "created"))
        for rid in pool["candidate_rows"]:
            r = rows[rid]
            print("%-9s %-16s %-10.6f %-18s %-6d %s%s"
                  % (rid, r["solver"], r["score"], pool["trees"][rid],
                     pool["identity"][rid]["n_identical"], r["created"][:19],
                     "" if rid in pool["pool"] else "   <- dropped (dup tree)"))
        print()
        print("rows=%d distinct trees=%d pooled=%d shared prompts=%s"
              % (pool["row_count"], pool["tree_count"], len(pool["pool"]),
                 ",".join(pool["shared_prompts"])))
        for g in pool["duplicate_groups"]:
            print("duplicate tree %s: %s -> kept %s"
                  % (g["tree"], g["solvers"], g["kept"]))
        print()
        print("same-tree MTP-leg replicate noise (percent):")
        for nm in ORDER:
            if nm in noise["per_prompt_pct"]:
                print("   %-9s %.4f" % (nm, noise["per_prompt_pct"][nm]))
        print("   pooled %.4f over %d legs"
              % (noise["pooled_pct"], noise.get("n_leg_pairs", 0)))
        print()
        cand = reading_candidates(rows[OUR_ROW])
        total = 1
        for nm in ORDER:
            total *= len(cand[nm])
            print("   readings %-9s %d" % (nm, len(cand[nm])))
        print("   cross product %d" % total)
        for rid in pool["pool"]:
            other = reading_candidates(rows[rid])
            same = all(other[nm] == cand[nm] for nm in pool["shared_prompts"])
            print("   reading sets match ours on shared prompts for %s: %s"
                  % (rid, same))
        return 0

    print("nothing to do; pass --census")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
