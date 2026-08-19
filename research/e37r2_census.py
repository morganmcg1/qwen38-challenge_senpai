#!/usr/bin/env python3
"""E37 r2: dispatched verify-width census from the TRUSTED parent's journal.

The r1 census parsed the phase trace. This one does not need it:
`QwenRuntimeMTPDriver.swift:289` writes `effectiveDraftLengths` as one element
per round, so the histogram is a trusted-parent record rather than a
solver-side print, and it carries no perturbation caveat.

Sections
--------
census    per-arm depth/M histogram, accept rate, row and token shares
control   traced vs untraced element-wise, same arm and same geometry
ranked    exact round count and accept rate for each ranked prompt, recovered
          from the published rationals
bracket   assumption-free bounds on the ranked M>=6 share (vertex enumeration)
maxent    the max-entropy inference class, scored against measured data
payoff    E38/E33 cell value under the corrected sigma and headroom

    python3 research/e37r2_census.py
"""
from __future__ import annotations

import collections
import fractions
import json
import math
import pathlib

RUNS = pathlib.Path(".mlxfast-private/e37/runs")
RUNS_R1 = pathlib.Path(".mlxfast-private/e37/runs-r1-traced")
TELEMETRY = pathlib.Path(".mlxfast-private/ranked-telemetry.json")
FRONTIER = pathlib.Path("senpai/frontier-state.json")
OUT = pathlib.Path("research/results/e37")
DECODE_TOKENS = 512

PROMPT_SHA8 = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
               "a2ea8b60": "essays", "00142a44": "medicine",
               "c1ec5866": "plutarch", "ea82dcb5": "republic",
               "3b10cb4d": "travel"}

# Ascending by ranked ratio on the board frontier row; ranks 4-5 are scored.
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]
# Published to 12 decimals by the ranked worker; keyed by prompt name.
RANKED_N = {
    "plutarch": 0.154004106776, "drama": 2.297619047619,
    "travel": 2.655660377358, "beagle": 4.532710280374,
    "medicine": 4.767676767677, "essays": 5.425287356322,
    "republic": 5.269662921348, "botany": 5.776470588235,
}
RANKED_RAW_P = {
    "plutarch": 1.2560334838, "drama": 1.9231089575, "travel": 2.1895159531,
    "beagle": 3.1433255794, "medicine": 3.3552623916, "essays": 3.3906635754,
    "republic": 3.4143725007, "botany": 3.4490615187,
}
RANKED_NONDRAFT = {"plutarch": 449}  # every other prompt publishes 0

SIGMA_SCORE = 0.000923      # E35 / ledger 131, replaces 0.078 %
MAX_DEPTH = 8               # Constants.swift:331, hard-closed


def meta(arm: str, root: pathlib.Path = RUNS) -> dict:
    """`meta.txt` as a dict; the runner writes one `key=value` per line."""
    out = {}
    for line in (root / arm / "meta.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def mtp_leg(arm: str, root: pathlib.Path = RUNS) -> dict:
    """The MTP timed leg's report. Leg 03 is the depth-0 serial control."""
    reports = sorted((root / arm / "reports").glob("*-mtp-timed.json"))
    legs = [json.load(open(p)) for p in reports]
    mtp = [d for d in legs if d["effective_max_draft_len"] > 0]
    if len(mtp) != 1:
        raise SystemExit("%s: expected exactly one drafting leg, got %d"
                         % (arm, len(mtp)))
    return mtp[0]


def census(arm: str, root: pathlib.Path = RUNS) -> dict:
    d = mtp_leg(arm, root)
    depths = d["effective_draft_lengths"]
    rounds = len(depths)
    accepted, rejected = d["accepted_draft_total"], d["rejected_draft_total"]
    offered = accepted + rejected
    hist = collections.Counter(x + 1 for x in depths)          # M = depth + 1
    rows = sum(m * c for m, c in hist.items())
    tokens = d["decode_token_count"]

    def share(pred, weight):
        num = sum(weight(m) * c for m, c in hist.items() if pred(m))
        den = sum(weight(m) * c for m, c in hist.items())
        return num / den

    return {
        "arm": arm,
        "round_count": rounds,
        "round_count_reported": d["round_count"],
        "mean_depth": d["effective_mean_draft_len"],
        "max_depth": d["effective_max_draft_len"],
        "mean_M": sum(depths) / rounds + 1,
        "max_M": max(depths) + 1,
        "non_drafting_round_count": d.get("non_drafting_round_count", 0),
        "offered": offered, "accepted": accepted, "rejected": rejected,
        "accept_rate": accepted / offered if offered else float("nan"),
        "decode_tokens": tokens,
        "dispatched_rows": rows,
        "identity_R_plus_A": rounds + accepted,
        "all_tokens_matched": d["all_tokens_matched"],
        "residual_divergence_count": d.get("residual_divergence_count"),
        "M_hist": dict(sorted(hist.items())),
        "round_share_ge6": share(lambda m: m >= 6, lambda m: 1),
        "row_share_ge6": share(lambda m: m >= 6, lambda m: m),
        "w6_round": share(lambda m: m == 6, lambda m: 1),
        "w6_row": share(lambda m: m == 6, lambda m: m),
        "depths": depths,
    }


def ranked_candidates(name: str, raw_p: dict) -> dict:
    """Feasible (R, D) readings of one ranked prompt's published mean depth.

    `effective_mean_draft_len` is D/R for integers D (offered drafts) and R
    (rounds), so its published decimal reduces to p/q and R is a multiple of q.
    Every emitted token is a primary or an accepted draft, so R + A = 512, and
    A <= D forces R >= 512/(1+n).  That identity is not assumed: the control
    section verifies it exactly on every local run.
    """
    n = RANKED_N[name]
    frac = fractions.Fraction(n).limit_denominator(4096)
    q = frac.denominator
    lo = DECODE_TOKENS / (1.0 + n)                 # from accept rate <= 1
    feasible = []
    for k in range(1, DECODE_TOKENS // q + 1):
        R = k * q
        if R < lo - 1e-9 or R > DECODE_TOKENS:
            continue
        D = round(n * R)
        if abs(n - D / R) > 1e-9:
            continue
        feasible.append((R, D, DECODE_TOKENS / (R * raw_p[name])))
    return {"name": name, "n": n, "reduced": (frac.numerator, q),
            "R_min_from_accept": lo, "feasible": feasible}


def resolve_ranked(raw_p: dict) -> dict:
    """Pin one reading per prompt by requiring rho(M) to be non-decreasing.

    rho = 512/(R*raw_p) is the candidate round cost in pinned-serial
    token-costs. A round verifying more rows, having also run more head steps,
    cannot cost less, so rho must not fall as mean M rises. Prompts whose
    feasible set is a singleton anchor the curve and the rest follow.  The
    smallest feasible multiple is NOT always right -- drama's is not -- so the
    criterion, not the heuristic, is what does the work.
    """
    cands = {nm: ranked_candidates(nm, raw_p) for nm in ORDER}
    by_M = sorted(ORDER, key=lambda nm: RANKED_N[nm])
    # Global enumeration, so uniqueness is proven rather than assumed. The
    # search space is the product of the per-prompt feasible sets, a few
    # thousand combinations at most.
    solutions = []

    def walk(i: int, chosen: list, last_rho: float) -> None:
        if i == len(by_M):
            solutions.append(list(chosen))
            return
        for f in cands[by_M[i]]["feasible"]:
            if f[2] < last_rho - 1e-9:
                continue
            chosen.append(f)
            walk(i + 1, chosen, f[2])
            chosen.pop()

    walk(0, [], 0.0)
    if len(solutions) != 1:
        raise SystemExit("monotone rho admits %d readings, not 1"
                         % len(solutions))
    picked = dict(zip(by_M, solutions[0]))
    out = {}
    for nm in ORDER:
        R, D, rho = picked[nm]
        rejected = [f for f in cands[nm]["feasible"] if f[0] != R]
        out[nm] = {"R": R, "D": D, "A": DECODE_TOKENS - R,
                   "alpha": (DECODE_TOKENS - R) / D, "rho": rho,
                   "mean_M": RANKED_N[nm] + 1.0,
                   "reduced": cands[nm]["reduced"], "rejected": rejected}
    return out


def fit_rho(resolved: dict) -> tuple[float, float, float]:
    """Least-squares rho = a + b*meanM over the eight ranked prompts.

    Over-determined: eight points, two parameters, so unlike the retracted
    R = (1+alpha*n)/(1+hbar*n) identity this fit CAN fail. Its residuals are
    the check on the round-count reconstruction -- a misread R moves one
    prompt's rho by a factor of ~2 and would stand out immediately.
    """
    xs = [resolved[nm]["mean_M"] for nm in ORDER]
    ys = [resolved[nm]["rho"] for nm in ORDER]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
         / sum((x - mx) ** 2 for x in xs))
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1.0 - ss_res / ss_tot


def bracket(n: float, allow_zero: bool, rho_ab: tuple | None = None) -> dict:
    """Exact bounds on the M>=6 round and row share at fixed mean depth n.

    Feasible set: {p on depths S : sum p = 1, sum d*p = n, p >= 0}.  With two
    equalities every vertex has at most two support points, so enumerating
    ordered pairs is exhaustive rather than a search.  S excludes depth 0 when
    the ranked worker publishes non_drafting_round_count = 0.
    """
    S = list(range(0 if allow_zero else 1, MAX_DEPTH + 1))
    out = {}
    weights = [("round", lambda d: 1.0), ("row", lambda d: d + 1.0)]
    if rho_ab is not None:
        a, b = rho_ab
        # Round TIME share, the quantity a cell's phi actually is. Row share
        # over-weights wide rounds because it ignores the fixed per-round cost.
        weights.append(("time", lambda d: a + b * (d + 1.0)))
    for label, weight in weights:
        vals = []
        for d1 in S:
            for d2 in S:
                if d1 == d2:
                    continue
                if not (min(d1, d2) <= n <= max(d1, d2)):
                    continue
                p1 = (n - d2) / (d1 - d2)
                p2 = 1.0 - p1
                if p1 < -1e-12 or p2 < -1e-12:
                    continue
                pts = ((d1, p1), (d2, p2))
                num = sum(p * weight(d) for d, p in pts if d >= 5)
                den = sum(p * weight(d) for d, p in pts)
                vals.append(num / den)
        for d in S:                                   # degenerate single-point
            if abs(d - n) < 1e-12:
                vals.append(1.0 if d >= 5 else 0.0)
        out[label] = (min(vals), max(vals))
    return out


def maxent(n: float, allow_zero: bool) -> dict:
    """Max-entropy distribution on the depth support at fixed mean n.

    This is the inference *class* E34 used as its cross-check. Scoring it
    against a measured histogram tests the method, not any one prediction.
    """
    S = list(range(0 if allow_zero else 1, MAX_DEPTH + 1))
    lo, hi = -50.0, 50.0
    for _ in range(400):                              # bisect on the tilt
        lam = 0.5 * (lo + hi)
        w = [math.exp(lam * d) for d in S]
        z = sum(w)
        mean = sum(d * x for d, x in zip(S, w)) / z
        if mean < n:
            lo = lam
        else:
            hi = lam
    w = [math.exp(lam * d) for d in S]
    z = sum(w)
    p = {d: x / z for d, x in zip(S, w)}
    rows = sum(pd * (d + 1) for d, pd in p.items())
    return {"p": p,
            "round_ge6": sum(pd for d, pd in p.items() if d >= 5),
            "row_ge6": sum(pd * (d + 1) for d, pd in p.items() if d >= 5) / rows,
            "w6_round": p.get(5, 0.0),
            "w6_row": p.get(5, 0.0) * 6 / rows}


def score_of(ratios: dict) -> float:
    s = sorted(ratios.values())
    return 0.5 * (s[3] + s[4])


def scored_rows() -> list:
    """Every telemetry row that carries a complete eight-prompt official score."""
    out = []
    for s in json.load(open(TELEMETRY))["submissions"]:
        om = s.get("officialMetrics")
        if isinstance(om, str):
            om = json.loads(om)
        if (isinstance(s.get("officialScore"), (int, float)) and om
                and om.get("prompt_count") == 8):
            out.append(dict(s, officialMetrics=om))
    return out


def per_prompt(row: dict) -> dict:
    """Prompt name -> per-prompt block, keyed by the published prompt digest."""
    return {PROMPT_SHA8[p["prompt_sha256"][:8]]: p
            for p in row["officialMetrics"]["per_prompt"]}


def pinned_row(rows: list, key: str) -> dict:
    """The exact submission `senpai/frontier-state.json` pins under `key`."""
    want = json.load(open(FRONTIER))[key]["id"]
    hit = [r for r in rows if r["id"] == want]
    if len(hit) != 1:
        raise SystemExit("frontier %s -> %s not in telemetry cache" % (key, want))
    return hit[0]


def identical_tree_pairs(rows: list) -> list:
    """Score deltas between rows that share a tree, i.e. direct sigma draws.

    Ledger 131 says nobody has ever resubmitted an identical tree, which is why
    sigma_score has to be imported rather than measured.  This rechecks that on
    the cached board instead of taking it on trust.
    """
    deltas = []
    for field in ("submissionCommitSha", "promotedSourceRef"):
        by = collections.defaultdict(list)
        for r in rows:
            if r.get(field):
                by[r[field]].append(r)
        for group in by.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i]["officialScore"], group[j]["officialScore"]
                    deltas.append(abs(a - b) / ((a + b) / 2))
    return deltas


def r_prime(row: dict) -> dict:
    """alphonse's E35 estimator: the row's own serial leg, averaged over the pool.

    Keeps the row's own serial so box speed still cancels, but averages it over
    the eight prompts so per-prompt serial noise falls by sqrt(8).  Ledger 131
    accepts this and retires `R* = global_bar[p] / cand_p`.
    """
    blocks = per_prompt(row)
    bar = sum(b["serial_seconds_per_token_mean"]
              for b in blocks.values()) / len(blocks)
    return {nm: bar / b["mtp_seconds_per_token_mean"] for nm, b in blocks.items()}


def main() -> None:
    print("=" * 78)
    print("CENSUS -- trusted-parent journal, 512 decode tokens, offered depth 8")
    print("=" * 78)
    arms = [p.name for p in sorted(RUNS.iterdir()) if (p / "reports").is_dir()]
    cs = {}
    for arm in list(arms):
        try:
            c = census(arm)
        except (SystemExit, ValueError) as exc:       # arm still in flight
            print("\n-- %s SKIPPED: %s" % (arm, exc))
            arms.remove(arm)
            continue
        cs[arm] = c
        print("\n-- %s" % arm)
        print("   rounds %-4d  mean M %.4f  max M %d  nondrafting %d"
              % (c["round_count"], c["mean_M"], c["max_M"],
                 c["non_drafting_round_count"]))
        print("   offered %-4d accepted %-4d rejected %-4d accept rate %.4f"
              % (c["offered"], c["accepted"], c["rejected"], c["accept_rate"]))
        print("   R + A = %d (must be %d)   tokens matched %s  divergence %s"
              % (c["identity_R_plus_A"], c["decode_tokens"],
                 c["all_tokens_matched"], c["residual_divergence_count"]))
        print("   M hist %s" % c["M_hist"])
        print("   share M>=6: round %.4f  row %.4f   w(M=6): round %.4f row %.4f"
              % (c["round_share_ge6"], c["row_share_ge6"],
                 c["w6_round"], c["w6_row"]))

    print()
    print("=" * 78)
    print("CONTROL -- does the phase trace perturb the counts it reports?")
    print("=" * 78)
    for arm in arms:
        if not (RUNS_R1 / arm / "reports").is_dir():
            continue
        # A backup copy is not a second run: two directories that record the
        # same start instant are one measurement, so the comparison is vacuous.
        if meta(arm).get("started") == meta(arm, RUNS_R1).get("started"):
            print("%-16s VACUOUS: both directories hold the same run (%s); "
                  "no untraced re-run exists"
                  % (arm, meta(arm).get("started")))
            continue
        a, b = census(arm), census(arm, RUNS_R1)
        same = a["depths"] == b["depths"]
        print("%-16s untraced rounds %-4d traced rounds %-4d  element-wise %s"
              % (arm, a["round_count"], b["round_count"],
                 "IDENTICAL" if same else "*** DIFFERENT ***"))
        if not same:
            diff = [(i, x, y) for i, (x, y)
                    in enumerate(zip(a["depths"], b["depths"])) if x != y]
            print("   first differing rounds: %s" % diff[:6])

    print()
    print("=" * 78)
    print("RANKED -- exact round count and accept rate from published rationals")
    print("=" * 78)
    ranked = resolve_ranked(RANKED_RAW_P)
    print("%-10s %-12s %-5s %-5s %-5s %-7s %-7s %-9s"
          % ("prompt", "n = D/R", "R", "D", "A", "alpha", "rho", "tok/round"))
    for name in ORDER:
        r = ranked[name]
        print("%-10s %-12s %-5d %-5d %-5d %.4f  %.4f  %.3f"
              % (name, "%d/%d" % r["reduced"], r["R"], r["D"], r["A"],
                 r["alpha"], r["rho"], DECODE_TOKENS / r["R"]))
        if r["rejected"]:
            print("           rejected by monotone rho: %s"
                  % ["R=%d/rho=%.3f" % (rr, vv) for rr, _, vv in r["rejected"]])
    a, b, r2 = fit_rho(ranked)
    print("\nrho(meanM) = %.4f + %.4f*meanM     R^2 = %.6f  (8 points, 2 params)"
          % (a, b, r2))
    print("%-10s %-9s %-9s %-9s" % ("prompt", "meanM", "rho", "residual"))
    for name in ORDER:
        r = ranked[name]
        print("%-10s %-9.4f %-9.4f %+.5f"
              % (name, r["mean_M"], r["rho"], r["rho"] - (a + b * r["mean_M"])))
    print("implied M=1 round cost %.4f of a pinned serial token; per-row %.4f"
          % (a + b, b))

    print()
    print("=" * 78)
    print("BRACKET -- assumption-free bounds on the ranked M>=6 share")
    print("=" * 78)
    print("%-10s %-20s %-20s %-20s"
          % ("prompt", "round [min,max]", "ROW [min,max]", "TIME [min,max]"))
    brackets = {}
    for name in ("beagle", "medicine"):
        allow_zero = RANKED_NONDRAFT.get(name, 0) > 0
        bk = brackets[name] = bracket(RANKED_N[name], allow_zero, (a, b))
        print("%-10s [%.4f, %.4f]     [%.4f, %.4f]     [%.4f, %.4f]"
              % (name, bk["round"][0], bk["round"][1], bk["row"][0],
                 bk["row"][1], bk["time"][0], bk["time"][1]))
    print("depth 0 excluded for both: the ranked worker publishes "
          "non_drafting_round_count = 0")

    print()
    print("=" * 78)
    print("MAXENT -- the E34 inference class, scored against measured data")
    print("=" * 78)
    print("%-16s %-8s %-10s %-10s %-10s %-10s"
          % ("arm", "mean d", "meas r>=6", "maxent", "meas row>=6", "maxent"))
    for arm, c in cs.items():
        m = maxent(c["mean_depth"], c["non_drafting_round_count"] > 0)
        print("%-16s %-8.4f %-10.4f %-10.4f %-10.4f %-10.4f"
              % (arm, c["mean_depth"], c["round_share_ge6"], m["round_ge6"],
                 c["row_share_ge6"], m["row_ge6"]))
    for name in ("beagle", "medicine"):
        m = maxent(RANKED_N[name], RANKED_NONDRAFT.get(name, 0) > 0)
        print("%-16s %-8.4f %-10s %-10.4f %-10s %-10.4f"
              % ("ranked " + name, RANKED_N[name], "-", m["round_ge6"],
                 "-", m["row_ge6"]))

    print()
    print("=" * 78)
    print("PAYOFF -- OUR row, corrected sigma and headroom")
    print("=" * 78)
    rows = scored_rows()
    us, top = pinned_row(rows, "ourBestRankedRow"), pinned_row(rows, "boardTop")
    pu, pt = per_prompt(us), per_prompt(top)
    ours = {nm: b["raw_ratio_of_means"] for nm, b in pu.items()}
    theirs = {nm: b["raw_ratio_of_means"] for nm, b in pt.items()}
    bad_n = [nm for nm in ORDER
             if abs(pu[nm]["effective_mean_draft_len"] - RANKED_N[nm]) > 5e-7]
    bad_p = [nm for nm in ORDER
             if abs(ours[nm] - pu[nm]["serial_seconds_per_token_mean"]
                    / pu[nm]["mtp_seconds_per_token_mean"]) > 1e-9]
    print("our row %s   score %.11f (reported %.11f)"
          % (us["id"][:8], score_of(ours), us["officialScore"]))
    print("board top %s score %.11f (reported %.11f)"
          % (top["id"][:8], score_of(theirs), top["officialScore"]))
    print("draft lengths differing from the RANKED section: %s" % (bad_n or "none"))
    print("rows where raw_p != serial/candidate: %s" % (bad_p or "none"))

    sig = SIGMA_SCORE
    pairs = identical_tree_pairs(rows)
    print("identical-tree pairs on the cached board: %d -- sigma_score cannot be "
          "measured here, so it is imported" % len(pairs))
    print("sigma_score = %.4f %% (ledger 131 / E35, effective on the crown's "
          "steep profile); 2 sigma = %.4f %%" % (100 * sig, 200 * sig))

    print()
    print("gap to the board top, two labelled estimators:")
    print("   R  official ratio-of-medians   %+.4f %%   (ours/top - 1)"
          % (100 * (score_of(ours) / score_of(theirs) - 1)))
    rp_us, rp_top = r_prime(us), r_prime(top)
    gap_rp = score_of(rp_us) / score_of(rp_top) - 1.0
    print("   R' mean_8(own serial)/cand_p   %+.4f %%   (ledger 131 quotes "
          "-0.561 %%)" % (100 * gap_rp))
    print("      R' scores: ours %.6f  top %.6f  (%.1f sigma)"
          % (score_of(rp_us), score_of(rp_top), abs(gap_rp) / sig))

    print()
    ordered = sorted(ours.items(), key=lambda kv: kv[1])
    print("our ascending order: %s" % [nm for nm, _ in ordered])
    print("scored cells (ranks 4,5) = %s, %s; binding upper neighbour = %s at "
          "raw_p %.6f" % (ordered[3][0], ordered[4][0], ordered[5][0], ordered[5][1]))
    ceiling = ordered[5][1]
    for name in ("beagle", "medicine"):
        head = ceiling / ours[name] - 1.0
        lifted = dict(ours)
        lifted[name] = ceiling
        gain = score_of(lifted) / score_of(ours) - 1.0
        print("%-9s raw_p %.6f -> ceiling %.6f  headroom %+.3f %%  "
              "saturated score %+.4f %% (%.1f sigma)"
              % (name, ours[name], ceiling, 100 * head, 100 * gain, gain / sig))
    both = dict(ours, beagle=ceiling, medicine=ceiling)
    print("%-9s                                        both saturated  "
          "score %+.4f %% (%.1f sigma)"
          % ("", 100 * (score_of(both) / score_of(ours) - 1),
             (score_of(both) / score_of(ours) - 1) / sig))

    print()
    print("per 1 %% speedup of a beagle cell holding candidate-leg TIME share phi"
          "  (sigma_score = %.4f %%):" % (100 * sig))
    phis = {}
    for phi in (0.15, 0.20, 0.2166, 0.25, 0.2734, 0.30, 0.50):
        lifted = dict(ours)
        lifted["beagle"] = ours["beagle"] / (1 - 0.01 * phi)
        gain = score_of(lifted) / score_of(ours) - 1.0
        phis[phi] = gain
        print("   phi %.4f -> score %+.4f %%  (%.2f sigma; %.1f %% cell speedup "
              "needed for 2 sigma)"
              % (phi, 100 * gain, gain / sig, 2 * sig / gain))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({
        "decode_tokens": DECODE_TOKENS,
        "offered_max_depth": MAX_DEPTH,
        "census": cs,
        "traced_control": {
            arm: {"untraced": census(arm)["depths"],
                  "traced": census(arm, RUNS_R1)["depths"]}
            for arm in arms
            if (RUNS_R1 / arm / "reports").is_dir()
            and meta(arm).get("started") != meta(arm, RUNS_R1).get("started")},
        "run_meta": {arm: meta(arm) for arm in arms},
        "ranked": ranked,
        "rho_fit": {"a": a, "b": b, "r2": r2},
        "bracket": brackets,
        "payoff": {
            "our_row_id": us["id"], "our_score": score_of(ours),
            "top_row_id": top["id"], "top_score": score_of(theirs),
            "our_raw_p": ours, "top_raw_p": theirs,
            "gap_R_pct": 100 * (score_of(ours) / score_of(theirs) - 1),
            "gap_Rprime_pct": 100 * gap_rp,
            "sigma_score_pct": 100 * sig,
            "identical_tree_pairs": len(pairs),
            "binding_neighbour": ordered[5][0], "ceiling": ceiling,
            "headroom_pct": {nm: 100 * (ceiling / ours[nm] - 1)
                             for nm in ("beagle", "medicine")},
            "saturated_score_pct": {
                nm: 100 * (score_of(dict(ours, **{nm: ceiling}))
                           / score_of(ours) - 1)
                for nm in ("beagle", "medicine")},
            "phi_to_score_pct": {"%.4f" % k: 100 * v for k, v in phis.items()},
        },
    }, open(OUT / "r2-census.json", "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % (OUT / "r2-census.json"))


if __name__ == "__main__":
    main()
