#!/usr/bin/env python3
"""E113 rung 0 - where the depth-price boundary is, and whether the schedule
has room on the current tree.

Two frames, kept in separate named tables so a local cancellation term cannot
leak into ranked pricing (program.md, "Ranked And Local Causal Boundaries").

  harness=local   E106 per-width round census on the current tree.
  harness=ranked  round cost recovered from our own post-E100 official
                  receipts, the same instrument as research/ranked_cost_curve.py
                  but refit after E100 moved the dispatch table.

Usage:
    python3 research/board_per_prompt.py fetch      # writes /tmp/yukon-board/full.json
    python3 research/e113_depth_boundary.py
"""

import json
import math
import statistics
from fractions import Fraction

BOARD = "/tmp/yukon-board/full.json"
T = 512

PROMPTS = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
OUR_RECEIPTS = ["b8b8b860", "44559d02"]
# Reviewed round counts, research/ranked_cost_curve.py. The maximal
# tokens-per-round rule pins all eight except drama, where 168 would make drama
# dearer than travel at a lower verify width.
ROUNDS = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
          "republic": 93, "essays": 92, "medicine": 90, "botany": 81}

# ---------------------------------------------------------------- cost tables

# harness=local. E106 unforced per-width census, W&B 19kgn6xi, current tree.
# M=2 and M=5 are SINGLE-ROUND samples; M=6/7/8 have n=4/3/10. The census
# serialises every command buffer, so these are GPU-clock microseconds and are
# explicitly not timing-valid.
E106_CENSUS = {2: 78208.218, 5: 113709.824, 6: 133563.9155,
               7: 144138.8057, 8: 156949.9529}
E106_N = {2: 1, 5: 1, 6: 4, 7: 3, 8: 10}
# M=3, M=4 interpolated on the straight line E106 measured between M=2 and M=5;
# M=1 extrapolated one step below M=2. All three are INFERRED, not measured.
_G1_STEP = (E106_CENSUS[5] - E106_CENSUS[2]) / 3.0
LOCAL_CURRENT = {
    1: E106_CENSUS[2] - _G1_STEP,
    2: E106_CENSUS[2],
    3: E106_CENSUS[2] + _G1_STEP,
    4: E106_CENSUS[2] + 2 * _G1_STEP,
    5: E106_CENSUS[5],
    6: E106_CENSUS[6], 7: E106_CENSUS[7], 8: E106_CENSUS[8],
}
LOCAL_MEASURED = {2, 5, 6, 7, 8}

# harness=ranked, route A. Finding 31 identity round_us = G * W / rate with the
# ranked rate table, re-partitioned for the post-E100 dispatch switch.
W_GB = 14.41235
RANKED_RATE = {1: (1, 462.3), 2: (1, 409.8), 3: (1, 368.0), 4: (1, 333.9),
               5: (1, 272.2),            # [5] after E100, one wide group
               6: (2, 477.7), 7: (2, 426.6), 8: (2, 385.3)}
RANKED_F31 = {M: g * W_GB / r * 1e6 for M, (g, r) in RANKED_RATE.items()}

# E92 per-position conditional acceptance, W&B ytcemy51.
E92_Q = [0.9659, 0.9652, 0.9543, 0.9486, 0.9487, 0.9859, 0.9451, 0.8333]
MAXDEPTH = 8
SHIP_H = 0.18

RESULTS = {}


# ------------------------------------------------------------------- receipts

def load_board():
    full = json.load(open(BOARD))
    rows = full["submissions"] if isinstance(full, dict) else full
    return [r for r in rows if isinstance(r, dict)]


def per_prompt(row):
    om = row.get("officialMetrics") or {}
    out = {}
    for e in om.get("per_prompt") or []:
        out[PROMPTS.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8])] = e
    return out


def receipt_rows(rows, prefix):
    hits = [r for r in rows if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("prefix %r matched %d rows" % (prefix, len(hits)))
    row = hits[0]
    e = per_prompt(row)
    out = {}
    for name, rec in e.items():
        dl = rec["effective_mean_draft_len"]
        f = Fraction(dl).limit_denominator(600)
        R = ROUNDS[name]
        if f.denominator and R % f.denominator:
            raise SystemExit("%s: reviewed round count %d is not a multiple of "
                             "the recovered denominator %d" % (name, R, f.denominator))
        mtp = rec["mtp_seconds_per_token_mean"]
        accepted = T - R
        drafts = dl * R
        out[name] = {
            "dl": dl, "M": dl + 1.0, "R": R,
            "mtp_us_per_token": mtp * 1e6,
            "round_us": T * mtp / R * 1e6,
            "accept": accepted / drafts if drafts > 0 else float("nan"),
            "raw": rec["raw_ratio_of_means"],
        }
    return row, out


# ------------------------------------------------------------------ fits

def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def two_line(points, brk):
    """Split at brk: group 1 is M < brk, group 2 is M >= brk."""
    g1 = [p for p in points if p[0] < brk]
    g2 = [p for p in points if p[0] >= brk]
    if len(g1) < 2 or len(g2) < 2:
        return None
    a1, c1 = linfit([p[0] for p in g1], [p[1] for p in g1])
    a2, c2 = linfit([p[0] for p in g2], [p[1] for p in g2])
    res = []
    for M, y in points:
        pred = (a1 + c1 * M) if M < brk else (a2 + c2 * M)
        res.append((y - pred) / y)
    rss = sum(r * r for r in res)
    return {"brk": brk, "a1": a1, "c1": c1, "a2": a2, "c2": c2,
            "rss": rss, "maxres": max(abs(r) for r in res),
            "step": (a2 + c2 * brk) - (a1 + c1 * brk)}


# ----------------------------------------------------- acceptance and pricing

def shaped_q(accept, realised_depth, shape=E92_Q):
    """Scale the E92 shape so the modelled per-draft accept rate at the
    realised mean depth equals the receipt's measured per-draft accept rate."""
    d = max(1, int(round(realised_depth)))

    def rate(k):
        q = [min(0.999999, k * s) for s in shape]
        reach, tot = 1.0, 0.0
        for i in range(d):
            reach *= q[i]
            tot += reach
        return tot / d

    lo, hi = 0.0, 1.0 / max(shape)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rate(mid) < accept:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return [min(0.999999, k * s) for s in shape], k


def expected_tokens(q, d):
    reach, tot = 1.0, 1.0
    for i in range(d):
        reach *= q[i]
        tot += reach
    return tot


def cost_per_token(curve, q, d):
    return curve[d + 1] / expected_tokens(q, d)


def greedy_depth(q, price_marginal, cap=7):
    """Exact reproduction of costModelDepth's walk with a given price vector."""
    cum = [1.0]
    run = 1.0
    for m in price_marginal:
        run += m
        cum.append(run)
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= q[depth]
        thr = price_marginal[depth] * (1.0 + expected) / cum[depth]
        if not reach > thr:
            break
        expected += reach
        depth += 1
    return depth


def uniform_price():
    return [SHIP_H] * MAXDEPTH


def boundary_price(width, tier):
    within = MAXDEPTH * SHIP_H / (MAXDEPTH - 1 + tier)
    m = [within] * MAXDEPTH
    m[width - 2] = within * tier
    return m


# ------------------------------------------------------------------- report

def marginal_table(name, curve, measured=frozenset()):
    print("\n=== %s: current-tree round cost and marginal cost ===" % name)
    print("  %-3s %-8s %11s %11s %8s %s"
          % ("M", "part", "round_us", "marginal", "vs prev", "provenance"))
    prev = None
    for M in range(1, 9):
        g = RANKED_RATE[M][0]
        part = {1: "[1]", 2: "[2]", 3: "[3]", 4: "[4]", 5: "[5]",
                6: "[3+3]", 7: "[4+3]", 8: "[4+4]"}[M]
        mar = "" if prev is None else "%11.1f" % (curve[M] - prev)
        rel = ""
        if prev is not None and M >= 3:
            base = curve[2] - curve[1]
            rel = "%8.2f" % ((curve[M] - prev) / base) if base else ""
        prov = "measured" if M in measured else "INFERRED"
        print("  %-3d %-8s %11.1f %s %s  G=%d %s"
              % (M, part, curve[M], mar, rel, g, prov))
        prev = curve[M]


def main():
    rows = load_board()
    print("=" * 78)
    print("E113 RUNG 0 - the depth-price boundary on the current tree")
    print("=" * 78)

    # ---- local frame -----------------------------------------------------
    marginal_table("harness=local (E106 census 19kgn6xi)", LOCAL_CURRENT,
                   LOCAL_MEASURED)
    print("  n per measured width: %s" % E106_N)
    print("  local tier ratio at the step into width 6: %.3f"
          % ((LOCAL_CURRENT[6] - LOCAL_CURRENT[5]) / _G1_STEP))

    # ---- ranked frame, route A ------------------------------------------
    marginal_table("harness=ranked route A (Finding 31 identity)", RANKED_F31,
                   frozenset())
    print("  ranked tier ratio at the step into width 6: %.3f"
          % ((RANKED_F31[6] - RANKED_F31[5]) / (RANKED_F31[5] - RANKED_F31[4])))
    print("  ranked tier ratio at the step into width 5: %.3f"
          % ((RANKED_F31[5] - RANKED_F31[4]) / (RANKED_F31[4] - RANKED_F31[3])))

    # ---- ranked frame, route B: our own post-E100 receipts ---------------
    print("\n=== harness=ranked route B: round cost from our post-E100 receipts ===")
    per_receipt = {}
    for pref in OUR_RECEIPTS:
        row, d = receipt_rows(rows, pref)
        per_receipt[pref] = d
        print("\n  receipt %s  official %.8f" % (pref, row["officialScore"]))
        print("    %-9s %6s %7s %11s %9s %9s"
              % ("prompt", "R", "M", "round_us", "us/token", "accept"))
        for name in sorted(d, key=lambda n: d[n]["M"]):
            r = d[name]
            print("    %-9s %6d %7.3f %11.1f %9.1f %9.4f"
                  % (name, r["R"], r["M"], r["round_us"],
                     r["mtp_us_per_token"], r["accept"]))

    pts = []
    for pref in OUR_RECEIPTS:
        for name, r in per_receipt[pref].items():
            pts.append((r["M"], r["round_us"], name, pref))
    xy = [(p[0], p[1]) for p in pts]

    print("\n  where is the ranked cost boundary? two-line fits, break at M=b")
    print("    %-5s %10s %10s %10s %10s %11s %9s"
          % ("break", "a1", "c1", "a2", "c2", "step_us", "max|res|"))
    fits = {}
    for brk in (4, 5, 6, 7):
        f = two_line(xy, brk)
        if f is None:
            continue
        fits[brk] = f
        print("    M>=%-2d %10.1f %10.1f %10.1f %10.1f %11.1f %8.2f%%"
              % (brk, f["a1"], f["c1"], f["a2"], f["c2"], f["step"],
                 100 * f["maxres"]))
    a, c = linfit([p[0] for p in xy], [p[1] for p in xy])
    res = [(y - (a + c * M)) / y for M, y in xy]
    print("    single line %.1f + %.1f * M   max|res| %.2f%%"
          % (a, c, 100 * max(abs(r) for r in res)))

    RESULTS["local_current_curve"] = dict(LOCAL_CURRENT)
    RESULTS["local_measured_widths"] = sorted(LOCAL_MEASURED)
    RESULTS["e106_n_per_width"] = dict(E106_N)
    RESULTS["local_tier_ratio_into_6"] = \
        (LOCAL_CURRENT[6] - LOCAL_CURRENT[5]) / _G1_STEP
    RESULTS["ranked_f31_curve"] = dict(RANKED_F31)
    RESULTS["ranked_tier_ratio_into_6"] = \
        (RANKED_F31[6] - RANKED_F31[5]) / (RANKED_F31[5] - RANKED_F31[4])
    RESULTS["ranked_tier_ratio_into_5"] = \
        (RANKED_F31[5] - RANKED_F31[4]) / (RANKED_F31[4] - RANKED_F31[3])
    RESULTS["receipt_rows"] = {p: per_receipt[p] for p in OUR_RECEIPTS}
    RESULTS["ranked_route_b_fits"] = {
        "break_M%d" % b: {k: f[k] for k in ("a1", "c1", "a2", "c2", "step",
                                            "maxres")}
        for b, f in fits.items()}
    RESULTS["ranked_route_b_single_line"] = {"a": a, "c": c,
                                             "maxres": max(abs(r) for r in res)}

    # ---- rung 0 proper: per-prompt optimum vs realised -------------------
    for label, curve in (("harness=ranked route A (F31)", RANKED_F31),
                         ("harness=local  (E106 census)", LOCAL_CURRENT)):
        print("\n=== per-prompt cost-per-emitted-token optimum, %s ===" % label)
        print("  cpt(d) = round_us(M=d+1) / expected emitted tokens(d)")
        print("  %-9s %7s %7s %6s %8s %8s | %s"
              % ("prompt", "accept", "realM", "d*", "cpt(d*)", "cpt(dreal)",
                 "cpt(d) for d = 0 .. 7"))
        gaps = []
        d0 = per_receipt[OUR_RECEIPTS[0]]
        for name in sorted(d0, key=lambda n: d0[n]["M"]):
            r = d0[name]
            q, _k = shaped_q(r["accept"], r["M"] - 1.0)
            cpt = [cost_per_token(curve, q, d) for d in range(0, 8)]
            best, dstar = min((c, d) for d, c in enumerate(cpt))
            dreal = min(7, max(0, int(round(r["M"] - 1.0))))
            gap = 100 * (cpt[dreal] - best) / best
            gaps.append((name, gap))
            RESULTS.setdefault("rung0_per_prompt", {}) \
                   .setdefault(label.split("(")[0].strip(), {})[name] = {
                "accept": r["accept"], "realised_mean_width": r["M"],
                "d_star": dstar, "d_realised": dreal, "cpt_at_d_star": best,
                "cpt_at_d_realised": cpt[dreal], "gap_pct": gap,
                "cpt_curve": cpt}
            print("  %-9s %7.4f %7.3f %6d %8.0f %8.0f | %s"
                  % (name, r["accept"], r["M"], dstar, best, cpt[dreal],
                     " ".join("%6.0f" % c for c in cpt)))
        near = [g for _n, g in gaps if g < 0.20]
        print("  prompts within 0.20%% of the optimum: %d of %d  -> KILL RULE 0 %s"
              % (len(near), len(gaps),
                 "FIRES" if len(near) >= 6 else "does not fire"))
        RESULTS.setdefault("rung0_gaps", {})[label.split("(")[0].strip()] = {
            "gap_pct": dict(gaps),
            "n_within_0p20pct": len(near),
            "kill_rule_0_fires": len(near) >= 6,
        }
        print("  CAVEAT: cpt(dreal) prices the realised MEAN depth as if it were "
              "a fixed depth.\n  The shipped walk is adaptive; E99 measured the "
              "adaptive walk 2.58 %% BETTER than\n  the best fixed depth on the "
              "ranked curve, so these gaps overstate the prize.")

    # ---- what the arms would choose --------------------------------------
    print("\n=== what each price arm chooses, per prompt (greedy walk, cap 7) ===")
    tier_ship = 2.0301
    tier_fit = (LOCAL_CURRENT[6] - LOCAL_CURRENT[5]) / _G1_STEP
    arms = {
        "ship": uniform_price(),
        "pb6": boundary_price(6, tier_ship),
        "pb6fit": boundary_price(6, tier_fit),
        "pb5": boundary_price(5, tier_ship),
    }
    print("  arm marginal vectors")
    for a_, m in arms.items():
        print("    %-7s %s" % (a_, " ".join("%.4f" % v for v in m)))
    print("\n  %-9s %7s %6s %6s %8s %8s %6s"
          % ("prompt", "realM", "ship", "pb6", "pb6fit", "pb5", "opt"))
    d0 = per_receipt[OUR_RECEIPTS[0]]
    for name in sorted(d0, key=lambda n: d0[n]["M"]):
        r = d0[name]
        q, _k = shaped_q(r["accept"], r["M"] - 1.0)
        chosen = {a_: greedy_depth(q, m) for a_, m in arms.items()}
        costs = [(cost_per_token(RANKED_F31, q, d), d) for d in range(0, 8)]
        _b, dstar = min(costs)
        print("  %-9s %7.3f %6d %6d %8d %8d %6d"
              % (name, r["M"], chosen["ship"], chosen["pb6"],
                 chosen["pb6fit"], chosen["pb5"], dstar))


# ------------------------------------------------------- rung 1: trace replay

def walk(q, price, cap, margin, lookahead_index=None):
    """costModelDepth's walk, optionally with a multi-step block test at one
    price index. `q` is the pre-round positionAcceptEMA vector."""
    cum = [1.0]
    run = 1.0
    for m in price:
        run += m
        cum.append(run)
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = q[depth]
        if margin == margin:
            if depth == 0:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
            elif depth == 1:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
        reach *= p
        thr = price[depth] * (1.0 + expected) / cum[depth]
        if reach > thr:
            expected += reach
            depth += 1
            continue
        if lookahead_index is not None and depth == lookahead_index:
            best_k, best_gain = 0, 0.0
            bcost, btok, r = 0.0, 0.0, reach
            for k in range(1, cap - depth + 1):
                if k > 1:
                    r *= q[depth + k - 1]
                bcost += price[depth + k - 1]
                btok += r
                gain = (cum[depth] / (1.0 + expected)
                        - (cum[depth] + bcost) / (1.0 + expected + btok))
                if gain > best_gain:
                    best_gain, best_k = gain, k
            if best_k > 1:
                r = reach
                for k in range(best_k):
                    if k:
                        r *= q[depth + k]
                    expected += r
                depth += best_k
                continue
        break
    return depth


def replay(tags, out_root="research/out"):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "e99_oracle", "research/e99_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from pathlib import Path

    tier_local = (LOCAL_CURRENT[6] - LOCAL_CURRENT[5]) / _G1_STEP
    tier_ranked = (RANKED_F31[5] - RANKED_F31[4]) / (RANKED_F31[4] - RANKED_F31[3])
    arms = {
        "ship": (uniform_price(), None),
        "pb6": (boundary_price(6, 2.0301), None),
        "pb6fit": (boundary_price(6, tier_local), None),
        "pb5": (boundary_price(5, 2.0301), None),
        "pb5fit": (boundary_price(5, tier_ranked), None),
        "look6": (boundary_price(6, tier_local), 4),
        "look5": (boundary_price(5, tier_ranked), 3),
    }
    curves = {
        "ranked": lambda M: mod.ranked_round_us(M),
        "local_current": lambda M: LOCAL_CURRENT[M],
    }

    print("\n" + "=" * 78)
    print("E113 RUNG 1 - replay on recorded current-tree rounds")
    print("=" * 78)
    for tag in tags:
        meta = dict(
            line.split("=", 1)
            for line in Path("%s/%s/meta.txt" % (out_root, tag)).read_text()
            .splitlines() if "=" in line)
        score = json.load(open("%s/%s/score.json" % (out_root, tag)))
        cap_off = score["metrics"]["mtp_depth"]
        rounds = mod.parse_trace(Path("%s/%s/trace.txt" % (out_root, tag)),
                                 tag, cap_off)
        # The parent offers at most the tokens left in the fixed decode window,
        # so the final rounds are budget-capped, not schedule-capped.
        emitted = 0
        # the prefill primary is committed by the first round, so the parent
        # offers over tokens + 1 emissions
        budget = int(meta["tokens"]) + 1
        for r in rounds:
            r.cap = min(r.cap, max(0, budget - emitted - 1))
            emitted += r.accepted + 1
        print("\n--- %s  base %s  tokens %s  rounds %d  offered cap %d ---"
              % (tag, meta["base_sha"][:8], meta["tokens"], len(rounds), cap_off))
        mism = sum(1 for r in rounds
                   if walk(r.ema, uniform_price(), r.cap, r.margin) != r.depth)
        print("    positive control, ship walk reproduces the recorded depth on "
              "%d of %d rounds" % (len(rounds) - mism, len(rounds)))
        if mism:
            print("    CONTROL FAILED - the replay does not model the shipped "
                  "rule; every number below is void")
        print("    mean recorded width %.3f  accept %.4f"
              % (statistics.fmean(r.depth for r in rounds) + 1.0,
                 sum(r.accepted for r in rounds)
                 / max(1, sum(r.depth for r in rounds))))
        for treatment in ("observed", "impute"):
            print("    treatment=%s" % treatment)
            base = None
            print("      %-8s %10s %10s %8s %8s"
                  % ("arm", "us/token", "mean depth", "vs ship", "curve"))
            for cname, curve in curves.items():
                base = None
                for aname, (price, la) in arms.items():
                    res = mod.evaluate_policy(
                        rounds, curve, treatment,
                        lambda r, p=price, l=la: walk(r.ema, p, r.cap,
                                                      r.margin, l))
                    if base is None:
                        base = res["us_per_token"]
                    print("      %-8s %10.1f %10.3f %+7.2f%% %8s"
                          % (aname, res["us_per_token"], res["mean_depth"],
                             100 * (res["us_per_token"] - base) / base, cname))
                    RESULTS.setdefault("rung1_trace", {}) \
                           .setdefault(tag, {}) \
                           .setdefault(treatment, {}) \
                           .setdefault(cname, {})[aname] = {
                        "us_per_token": res["us_per_token"],
                        "mean_depth": res["mean_depth"],
                        "vs_ship_pct": 100 * (res["us_per_token"] - base) / base}
        RESULTS.setdefault("rung1_control", {})[tag] = {
            "rounds": len(rounds), "mismatches": mism,
            "base_sha": meta["base_sha"], "tokens": int(meta["tokens"]),
            "offered_cap": cap_off,
            "mean_recorded_width":
                statistics.fmean(r.depth for r in rounds) + 1.0}


def per_prompt_replay(tag="e101ctl512", out_root="research/out"):
    """Rung 1b. Replay every arm at each ranked prompt's realised width.

    The recorded EMA vectors carry realistic round-to-round dispersion but come
    from one easy public prompt at mean width 7.359. Scaling them by a single
    per-prompt factor until the SHIP walk reproduces that prompt's realised mean
    draft length gives a width sweep (Rule 37) that keeps the dispersion.
    Token yield comes from the prompt's calibrated acceptance profile, not from
    the recorded acceptances, so the counterfactual is independent of the
    trajectory the ship policy happened to take.
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "e99_oracle", "research/e99_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    meta = dict(line.split("=", 1) for line
                in Path("%s/%s/meta.txt" % (out_root, tag)).read_text()
                .splitlines() if "=" in line)
    cap_off = json.load(open("%s/%s/score.json"
                             % (out_root, tag)))["metrics"]["mtp_depth"]
    rounds = mod.parse_trace(Path("%s/%s/trace.txt" % (out_root, tag)),
                             tag, cap_off)
    for r in rounds:
        r.cap = min(r.cap, 7)

    rows = load_board()
    _row, rec = receipt_rows(rows, OUR_RECEIPTS[0])

    tier_local = (LOCAL_CURRENT[6] - LOCAL_CURRENT[5]) / _G1_STEP
    tier_ranked = (RANKED_F31[5] - RANKED_F31[4]) / (RANKED_F31[4] - RANKED_F31[3])
    arms = {
        "ship": (uniform_price(), None),
        "pb6": (boundary_price(6, 2.0301), None),
        "pb6fit": (boundary_price(6, tier_local), None),
        "pb5": (boundary_price(5, 2.0301), None),
        "look6": (boundary_price(6, tier_local), 4),
        "look5": (boundary_price(5, tier_ranked), 3),
        # NEGATIVE CONTROL. E99 rung 5 shipped exactly this action - clamp to
        # depth 3, the top of the then-current G = 1 band, whenever the pending
        # primary's top-2 margin is at or below 9.4375. Its ranked-curve model
        # predicted +3.222 %. The official M5 run measured -6.077 %. Any arm
        # whose predicted gain comes from the same clamp direction inherits
        # that refutation.
        "e99gate": (uniform_price(), "e99"),
    }

    def mean_depth(lam, price, la):
        ema = None
        tot = 0
        for r in rounds:
            ema = [min(0.999999, lam * v) for v in r.ema]
            tot += walk(ema, price, r.cap, r.margin, la)
        return tot / len(rounds)

    print("\n" + "=" * 78)
    print("E113 RUNG 1b - per-prompt replay at the realised ranked widths")
    print("=" * 78)
    print("  trace %s, %d rounds, EMA dispersion preserved, level recalibrated"
          % (tag, len(rounds)))

    ship_price = uniform_price()
    out = {a: {} for a in arms}
    lams = {}
    for name in sorted(rec, key=lambda n: rec[n]["M"]):
        target = rec[name]["dl"]
        lo, hi = 0.0, 1.4
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mean_depth(mid, ship_price, None) < target:
                lo = mid
            else:
                hi = mid
        lams[name] = 0.5 * (lo + hi)

    print("\n  %-9s %7s %8s %8s | %s"
          % ("prompt", "realdl", "lambda", "shipdl", "arm mean depth / ranked us per token / delta%"))
    published = {a: [] for a in arms}
    for name in sorted(rec, key=lambda n: rec[n]["M"]):
        r0 = rec[name]
        lam = lams[name]
        q, _k = shaped_q(r0["accept"], r0["M"] - 1.0)
        base_cost = None
        line = []
        ship_depths = None
        for aname, (price, la) in arms.items():
            cost = tokens = 0.0
            depths = []
            for r in rounds:
                ema = [min(0.999999, lam * v) for v in r.ema]
                if la == "e99":
                    d = walk(ema, price, r.cap, r.margin)
                    if r.margin <= 9.4375:
                        d = min(d, 3)
                else:
                    d = walk(ema, price, r.cap, r.margin, la)
                cost += RANKED_F31[d + 1]
                tokens += expected_tokens(q, d)
                depths.append(d)
            upt = cost / tokens
            if base_cost is None:
                base_cost = upt
                ship_depths = depths
            delta = 100 * (upt - base_cost) / base_cost
            out[aname][name] = delta
            published[aname].append(r0["raw"] / (1.0 + delta / 100.0)
                                    if r0["raw"] else float("nan"))
            clamp = sum(1 for a, b in zip(depths, ship_depths) if a < b) \
                / len(depths)
            line.append("%s %.2f/%.0f/%+.2f%%/clamp %.2f"
                        % (aname, statistics.fmean(depths), upt, delta, clamp))
            RESULTS.setdefault("rung1b_per_prompt", {}) \
                   .setdefault(name, {})[aname] = {
                "mean_depth": statistics.fmean(depths),
                "ranked_us_per_token": upt, "vs_ship_pct": delta,
                "clamp_share": clamp}
        print("  %-9s %7.3f %8.4f %8.3f | %s"
              % (name, r0["dl"], lam,
                 mean_depth(lam, ship_price, None), "  ".join(line)))

    print("\n  predicted published median (Finding 16 weighting is implicit in "
          "the median)")
    print("    %-8s %14s %10s" % ("arm", "median", "vs ship"))
    base_med = None
    for aname in arms:
        vals = sorted(published[aname])
        med = 0.5 * (vals[3] + vals[4])
        if base_med is None:
            base_med = med
        print("    %-8s %14.8f %+9.4f%%"
              % (aname, med, 100 * (med - base_med) / base_med))
        RESULTS.setdefault("rung1b_published", {})[aname] = {
            "median": med, "vs_ship_pct": 100 * (med - base_med) / base_med,
            "per_prompt_raw": dict(zip(sorted(rec, key=lambda n: rec[n]["M"]),
                                       published[aname]))}
    print("    KILL RULE 1 threshold: 0.20 %% predicted published gain")
    RESULTS["rung1b_lambda"] = lams
    RESULTS["kill_rule_1_fires"] = all(
        v["vs_ship_pct"] < 0.20
        for a, v in RESULTS["rung1b_published"].items()
        if a in ("pb6", "pb6fit", "look6"))


if __name__ == "__main__":
    from pathlib import Path
    main()
    replay(["e101ctl512"])
    per_prompt_replay()
    Path("research/e113-artifacts").mkdir(parents=True, exist_ok=True)
    Path("research/e113-artifacts/rung01.json").write_text(
        json.dumps(RESULTS, indent=2, sort_keys=True))
    print("\nwrote research/e113-artifacts/rung01.json")
