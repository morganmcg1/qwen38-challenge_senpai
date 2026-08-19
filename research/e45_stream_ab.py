#!/usr/bin/env python3
"""E45 r2: identify the cost of one marginal QMV weight stream from the board's
own cross-solver A/B experiments, and refit the ranked row on source-derived
streams(M) instead of a fitted breakpoint in M.

WHAT THIS FILE IS FOR
---------------------
r1 of this assignment asked whether pooling plateau rows could separate a
step-in-M cost family from a quadratic-in-M one. That question is closed:
thorfinn's E41 falsifies any quadratic model-free (first differences of a
quadratic are monotone; the measured ones drop 22.846 ms after the boundary),
and the r1 step indicator was placed at 5->6, which is not a boundary of the
tree the pooled rows were measured on. See `census HEAD`: the shipped stream
vector is 1,1,2,2,2,2,3 with boundaries at 4->5 and 8->9.

r2 therefore asks a different and answerable question: WHAT DOES ONE MARGINAL
WEIGHT STREAM COST, measured on the ranked host, using rival submissions that
differ from each other ONLY in the QMV width->IPG dispatch table?

IDENTIFIABILITY, STATED UP FRONT BECAUSE IT BOUNDS EVERY NUMBER BELOW
--------------------------------------------------------------------
The intended estimator was delta_leg(p) = n_8(p) * delta_T(8) / tokens(p), with
a free falsification test comparing the cross-prompt ratio of delta_leg against
the cross-prompt ratio of n_8. THAT TEST CANNOT BE RUN. `officialMetrics`
per-prompt records `effective_mean_draft_len`, `non_drafting_round_count`,
`mtp_seconds_per_token_mean`, `serial_seconds_per_token_mean`,
`raw_ratio_of_means` and `accepted_pair_count`. There is NO per-width round
histogram anywhere in the board payload, so n_8(p) is not observable and
delta_T(8) in milliseconds is NOT identified from ranked data. Only the product
n_8(p) * delta_T(8) is, i.e. the leg-level effect.

Two consequences:
  * every effect here is reported in leg-relative terms (per cent of candidate
    seconds/token, and score points), never as a per-round millisecond cost;
  * the falsification test is replaced by a NEGATIVE CONTROL that the data does
    support: one scored prompt is almost entirely non-drafting
    (effective_mean_draft_len 0.1540, 449 non-drafting rounds), so it reaches
    wide widths far less often than the others. A dispatch-table change at
    width 8 must move that prompt much less than the drafting-heavy prompts. If
    it moves them all alike, the effect is not coming from the width-8 cell.

Leg-relative is also the decision-relevant currency: the open question is
whether raising the wide helper's bound to NA=5 repays its register cost in
SCORE, and score is built from leg ratios, not from per-round milliseconds.

UNIT OF ANALYSIS
----------------
The experimental unit is a GIT TREE, not a board row. Identical submitted
content has an identical tree SHA, so rows sharing a tree SHA are REPLICATES of
one experiment and are averaged when a tree acts as an arm. That same structure
supplies the noise floor for free: the spread across repeated submissions of a
byte-identical tree is pure ranked-host replication noise, with no code
difference of any kind to explain it.

Every row carrying `officialMetrics` is used regardless of accepted/rejected
status. Status reflects whether a submission improved the frontier, which is
downstream of the timing being measured; filtering on it would condition on the
outcome.

USAGE
    python3 research/e45_stream_ab.py --self-test
    python3 research/e45_stream_ab.py --run
"""
import argparse
import collections
import difflib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stream_dispatch_census as C  # noqa: E402
import e43_ranked_step as e43  # noqa: E402

CORPUS = pathlib.Path(".mlxfast-private/e43-corpus.json")
TREE_CACHE = pathlib.Path(".mlxfast-private/e45-tree-census.json")
OUT = pathlib.Path("research/e45-stream-ab.json")

DECODE_TOKENS = 512
PROMPT_COUNT = 8
STRUCT = re.compile(r'^[\s{}]*(case\s+\d+\s*:|break\s*;|default\s*:)?\s*$')
CELL_TOKEN = "qmv_fast_crossrow_affine4_g64_m<"


# ---------------------------------------------------------------- pure helpers

def spread(xs):
    """Dispersion summary that is honest about n=1 and n=2."""
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    d = {"n": n, "mean": mean, "min": min(xs), "max": max(xs),
         "range": max(xs) - min(xs)}
    d["sd"] = statistics.stdev(xs) if n >= 2 else None
    d["rel_range"] = (d["range"] / mean) if mean else None
    d["rel_sd"] = (d["sd"] / mean) if (d["sd"] is not None and mean) else None
    return d


def classify_kernel_diff(a_files, b_files):
    """('dispatch-only'|'confounded'|'identical', n_cell_lines, n_other_lines).

    'dispatch-only' means every changed line in BOTH kernel files either names a
    qmv_fast_crossrow_affine4_g64_m<...> instantiation or is pure switch
    scaffolding. Such a pair differs in the dispatch table and in nothing else,
    which is the only configuration that isolates stream count.

    This test exists because the census fingerprint does NOT isolate the
    dispatch table: it excludes both kernel files from the hash, and those files
    are where the mechanism lives. Two trees sharing a fingerprint can differ by
    an entire extra kernel -- 070f1189 vs b428c300 differ by 101 lines including
    a whole qmv_fast_singlerow_affine2_g64 definition.
    """
    changed = []
    for pa, pb in zip(a_files, b_files):
        if pa is None or pb is None:
            return "unreadable", 0, 0
        for line in difflib.unified_diff(pa, pb, n=0, lineterm=''):
            if line.startswith(('---', '+++', '@@')):
                continue
            if line.startswith(('-', '+')):
                changed.append(line[1:])
    if not changed:
        return "identical", 0, 0
    cell = [l for l in changed if CELL_TOKEN in l]
    other = [l for l in changed
             if CELL_TOKEN not in l and not STRUCT.match(l)]
    return ("dispatch-only" if not other else "confounded"), len(cell), len(other)


def legal_ipgs(m, lo=2, hi=4):
    """IPG values legal at width M under the current NA bound.

    The wide helper is instantiated for NA in {2,3,4}; a cell is legal when
    IPG is in range and M % IPG != 1 (a lone trailing row is not emitted).
    """
    return [g for g in range(lo, hi + 1) if m % g != 1]


def min_streams(m, lo=2, hi=4):
    opts = legal_ipgs(m, lo, hi)
    return min(math.ceil(m / g) for g in opts) if opts else None


# --------------------------------------------------------------- board corpus

def load_rows():
    subs = json.loads(CORPUS.read_text())["submissions"]
    out = []
    for s in subs:
        om = s.get("officialMetrics")
        if not om:
            continue
        om = json.loads(om) if isinstance(om, str) else om
        pp = om.get("per_prompt")
        if not pp or len(pp) != PROMPT_COUNT:
            continue
        if om.get("decode_tokens") != DECODE_TOKENS:
            continue
        out.append({
            "id": s["id"], "pfx": s["id"][:8], "status": s.get("status"),
            "score": s.get("officialScore"), "solver": s.get("solverUsername"),
            "created": s.get("createdAt"),
            "depth": om.get("mtp_depth"),
            "max_depth": om.get("mtp_max_draft_depth"),
            # Covariates that could explain a difference between byte-identical
            # trees without invoking measurement noise. All are checked for
            # invariance inside each replicate set.
            "covariates": {
                "weights_hash": om.get("qwen_mtp_weights_hash"),
                "heads": [p["head_provenance_sha256"] for p in pp],
                "policy": [om.get("mode"), om.get("aggregation"),
                           om.get("median_rule"), om.get("scoring_normalized"),
                           om.get("score_anchor"),
                           om.get("decode_speedup_ceiling"),
                           om.get("decode_speedup_floor"),
                           om.get("pairs_per_prompt")],
                "rho": [p["effective_mean_draft_len"] for p in pp],
                "nondraft": [p["non_drafting_round_count"] for p in pp],
            },
            "per_prompt": [{
                "sha": p["prompt_sha256"][:8],
                "rho": p["effective_mean_draft_len"],
                "nondraft": p["non_drafting_round_count"],
                "mtp": p["mtp_seconds_per_token_mean"],
                "serial": p["serial_seconds_per_token_mean"],
                "ratio": p["raw_ratio_of_means"],
                "parity": p.get("parity_ok"),
            } for p in pp],
        })
    return out


def tree_census(refresh):
    """{submission-id-prefix: {tree, table, streams, fp, qh, qcpp}}."""
    if TREE_CACHE.exists() and not refresh:
        return json.loads(TREE_CACHE.read_text())
    census = {}
    scanned = with_table = 0
    for name, obj in C.submission_refs():
        scanned += 1
        tbl = C.dispatch_table(obj)
        if tbl is None:
            continue
        fp = C.non_kernel_fingerprint(obj)
        if fp is None:
            continue
        with_table += 1
        tree = C.run(["git", "rev-parse", obj + "^{tree}"]).stdout.strip()
        blobs = {}
        for line in C.run(["git", "ls-tree", "-r", obj]).stdout.splitlines():
            try:
                meta, path = line.split("\t", 1)
            except ValueError:
                continue
            if path in C.KERNEL_PATHS:
                blobs[path] = meta.split()[2]
        census[name[:8]] = {
            "ref": obj, "tree": tree, "fp": fp,
            "table": {str(k): v for k, v in sorted(tbl.items())},
            "streams": {str(k): v for k, v in sorted(C.streams(tbl).items())},
            "qh": blobs.get(C.QH), "qcpp": blobs.get(C.QCPP),
        }
    payload = {"scanned": scanned, "with_table": with_table, "trees": census}
    TREE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TREE_CACHE.write_text(json.dumps(payload))
    return payload


def kernel_text(ref):
    out = []
    for path in (C.QH, C.QCPP):
        r = C.run(["git", "cat-file", "-p", "%s:%s" % (ref, path)])
        out.append(r.stdout.splitlines() if r.returncode == 0 else None)
    return out


# ------------------------------------------------------------------- analysis

def replication_floor(rows, census):
    """Score and per-prompt spread across repeated submissions of ONE tree.

    These rows share a tree SHA, so the submitted bytes are identical. Any
    spread is ranked-host replication noise: scheduling, thermal state, driver
    and OS variation between submissions. No code difference exists to explain
    it.
    """
    by_tree = collections.defaultdict(list)
    for r in rows:
        c = census["trees"].get(r["pfx"])
        if c:
            by_tree[c["tree"]].append(r)
    sets = []
    for tree, rs in sorted(by_tree.items()):
        if len(rs) < 2:
            continue
        scores = [r["score"] for r in rs if r["score"] is not None]
        if len(scores) < 2:
            continue
        per_prompt = []
        for i in range(PROMPT_COUNT):
            per_prompt.append({
                "sha": rs[0]["per_prompt"][i]["sha"],
                "rho": rs[0]["per_prompt"][i]["rho"],
                "mtp": spread(r["per_prompt"][i]["mtp"] for r in rs),
                "ratio": spread(r["per_prompt"][i]["ratio"] for r in rs),
                "serial": spread(r["per_prompt"][i]["serial"] for r in rs),
            })
        cov = {k: len({json.dumps(r["covariates"][k], sort_keys=True)
                       for r in rs})
               for k in rs[0]["covariates"]}
        sets.append({
            "tree": tree[:12],
            "rows": [{"pfx": r["pfx"], "solver": r["solver"],
                      "status": r["status"], "score": r["score"],
                      "created": r["created"]} for r in rs],
            "streams": census["trees"][rs[0]["pfx"]]["streams"],
            "score": spread(scores),
            "per_prompt": per_prompt,
            "distinct_covariate_values": cov,
            "all_covariates_invariant": all(v == 1 for v in cov.values()),
        })
    sets.sort(key=lambda s: -s["score"]["n"])
    all_rel_sd = [s["score"]["rel_sd"] for s in sets
                  if s["score"]["rel_sd"] is not None]
    all_rel_rng = [s["score"]["rel_range"] for s in sets]
    # Pooled within-tree sd of the published score, in relative terms.
    num = den = 0.0
    for s in sets:
        n = s["score"]["n"]
        if s["score"]["sd"] is None:
            continue
        num += (n - 1) * (s["score"]["sd"] / s["score"]["mean"]) ** 2
        den += (n - 1)
    # Per-prompt pooled within-tree relative sd of the candidate leg. The
    # near-non-drafting prompt is expected to be much quieter: replication noise
    # in this benchmark is dominated by speculative GPU work, not by the host
    # baseline, and that expectation is itself a check on the floor.
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    rho_of = {}
    for s in sets:
        for pp in s["per_prompt"]:
            m = pp["mtp"]
            if m["sd"] is None or not m["mean"]:
                continue
            acc[pp["sha"]][0] += (m["n"] - 1) * (m["sd"] / m["mean"]) ** 2
            acc[pp["sha"]][1] += (m["n"] - 1)
            rho_of[pp["sha"]] = pp["rho"]
    per_prompt_noise = {
        k: {"rho": rho_of[k], "rel_sd": math.sqrt(v[0] / v[1]), "dof": v[1]}
        for k, v in acc.items() if v[1]}
    return {
        "sets": sets,
        "n_sets": len(sets),
        "n_rows": sum(s["score"]["n"] for s in sets),
        "n_sets_all_covariates_invariant": sum(
            1 for s in sets if s["all_covariates_invariant"]),
        "per_prompt_noise": per_prompt_noise,
        "pooled_rel_sd": math.sqrt(num / den) if den else None,
        "max_rel_range": max(all_rel_rng) if all_rel_rng else None,
        "median_rel_sd": statistics.median(all_rel_sd) if all_rel_sd else None,
        "recorded_sigma_score_pct": e43.SIGMA_SCORE_PCT,
        "crown_gap_pct": e43.CROWN_GAP_PCT,
    }


def tree_arms(rows, census):
    """{tree: {'rows': [...], 'streams': ..., 'fp': ..., 'ref': ...}}"""
    by_tree = collections.defaultdict(list)
    for r in rows:
        c = census["trees"].get(r["pfx"])
        if c:
            by_tree[c["tree"]].append(r)
    arms = {}
    for tree, rs in by_tree.items():
        c = census["trees"][rs[0]["pfx"]]
        pp = []
        for i in range(PROMPT_COUNT):
            pp.append({
                "sha": rs[0]["per_prompt"][i]["sha"],
                "rho": rs[0]["per_prompt"][i]["rho"],
                "nondraft": rs[0]["per_prompt"][i]["nondraft"],
                "mtp": sum(r["per_prompt"][i]["mtp"] for r in rs) / len(rs),
                "serial": sum(r["per_prompt"][i]["serial"] for r in rs) / len(rs),
                "ratio": sum(r["per_prompt"][i]["ratio"] for r in rs) / len(rs),
            })
        scores = [r["score"] for r in rs if r["score"] is not None]
        arms[tree] = {
            "ref": c["ref"], "fp": c["fp"], "streams": c["streams"],
            "table": c["table"], "n_rows": len(rs),
            "rows": [r["pfx"] for r in rs],
            "solvers": sorted({r["solver"] for r in rs}),
            "score": (sum(scores) / len(scores)) if scores else None,
            "per_prompt": pp,
        }
    return arms


def work_identical(a, b):
    """Same schedule => same per-prompt draft profile, to full precision."""
    for pa, pb in zip(a["per_prompt"], b["per_prompt"]):
        if pa["sha"] != pb["sha"]:
            return False
        if pa["rho"] != pb["rho"] or pa["nondraft"] != pb["nondraft"]:
            return False
    return True


def ab_pairs(arms):
    """Dispatch-only cross-arm tree pairs that also have ranked metrics."""
    by_fp = collections.defaultdict(list)
    for tree, a in arms.items():
        by_fp[a["fp"]].append(tree)
    text = {}
    pairs = []
    for fp, trees in sorted(by_fp.items()):
        if len({json.dumps(arms[t]["table"], sort_keys=True) for t in trees}) < 2:
            continue
        for i in range(len(trees)):
            for j in range(len(trees)):
                if i >= j:
                    continue
                ta, tb = trees[i], trees[j]
                A, B = arms[ta], arms[tb]
                if A["table"] == B["table"]:
                    continue
                for t in (ta, tb):
                    if t not in text:
                        text[t] = kernel_text(arms[t]["ref"])
                kind, ncell, nother = classify_kernel_diff(text[ta], text[tb])
                widths = sorted(
                    int(m) for m in set(A["streams"]) | set(B["streams"])
                    if A["streams"].get(m) != B["streams"].get(m))
                pairs.append({
                    "fp": fp[:12], "kind": kind,
                    "cell_lines": ncell, "other_lines": nother,
                    "widths": widths,
                    "a": {"tree": ta[:12], "rows": A["rows"],
                          "n_rows": A["n_rows"], "solvers": A["solvers"],
                          "score": A["score"],
                          "streams": {str(w): A["streams"].get(str(w))
                                      for w in widths}},
                    "b": {"tree": tb[:12], "rows": B["rows"],
                          "n_rows": B["n_rows"], "solvers": B["solvers"],
                          "score": B["score"],
                          "streams": {str(w): B["streams"].get(str(w))
                                      for w in widths}},
                    "work_identical": work_identical(A, B),
                })
    return pairs


def pair_effect(arms, pair):
    """Per-prompt leg effect of going from arm a to arm b."""
    A = arms[[t for t in arms if t[:12] == pair["a"]["tree"]][0]]
    B = arms[[t for t in arms if t[:12] == pair["b"]["tree"]][0]]
    per = []
    for pa, pb in zip(A["per_prompt"], B["per_prompt"]):
        per.append({
            "sha": pa["sha"], "rho": pa["rho"], "nondraft": pa["nondraft"],
            "mtp_a": pa["mtp"], "mtp_b": pb["mtp"],
            "d_mtp_rel": pb["mtp"] / pa["mtp"] - 1.0,
            "serial_a": pa["serial"], "serial_b": pb["serial"],
            "d_serial_rel": pb["serial"] / pa["serial"] - 1.0,
            "ratio_a": pa["ratio"], "ratio_b": pb["ratio"],
            "d_ratio_rel": pb["ratio"] / pa["ratio"] - 1.0,
        })
    return per


def flip(per):
    """Re-express a per-prompt effect with the two arms swapped."""
    out = []
    for e in per:
        f = dict(e)
        for k in ("mtp", "serial", "ratio"):
            f[k + "_a"], f[k + "_b"] = e[k + "_b"], e[k + "_a"]
            f["d_%s_rel" % k] = f[k + "_b"] / f[k + "_a"] - 1.0
        out.append(f)
    return out


def dose_at_width(arms, pairs, width):
    """Dispatch-only, work-identical pairs whose ONLY differing width is `width`.

    Oriented so arm `lo` has fewer weight streams than arm `hi`, making a
    positive `d_mtp_rel` mean "more streams is slower".
    """
    keep = []
    for p in pairs:
        if p["kind"] != "dispatch-only" or p["widths"] != [width]:
            continue
        if not p["work_identical"]:
            continue
        sa = p["a"]["streams"][str(width)]
        sb = p["b"]["streams"][str(width)]
        if sa is None or sb is None or sa == sb:
            continue
        per = pair_effect(arms, p)
        if sa < sb:
            low_arm, high_arm, oriented, s_lo, s_hi = p["a"], p["b"], per, sa, sb
        else:
            low_arm, high_arm, oriented, s_lo, s_hi = (
                p["b"], p["a"], flip(per), sb, sa)
        keep.append({
            "fp": p["fp"], "width": width,
            "streams_lo": s_lo, "streams_hi": s_hi,
            "cell_lines": p["cell_lines"],
            "lo": low_arm, "hi": high_arm,
            "per_prompt": oriented,
            "mean_d_mtp_rel": sum(e["d_mtp_rel"] for e in oriented) / PROMPT_COUNT,
            "mean_d_ratio_rel": sum(e["d_ratio_rel"] for e in oriented) / PROMPT_COUNT,
            "d_score_rel": ((high_arm["score"] / low_arm["score"] - 1.0)
                            if (high_arm["score"] and low_arm["score"]) else None),
        })
    return keep


def dose_summary(doses, floor):
    """Cluster the +1-stream effect by fingerprint group and test it.

    Pairs within one fingerprint group are NOT independent: 6 of the 7 width-8
    groups contribute pairs that share an arm, so the same tree appears in many
    pairs. Treating 14 pairs as 14 replicates would overstate the evidence by
    roughly sqrt(14/7). The independent unit is the fingerprint group, so each
    group contributes one value.

    The matched comparison is against the per-prompt replication floor, not
    against zero: a pair is a difference of two independent submissions, so its
    noise is sqrt(2) x the single-submission replication sd.
    """
    ds = [x for x in doses if x["streams_hi"] - x["streams_lo"] == 1]
    if not ds:
        return None
    bygrp = collections.defaultdict(list)
    for x in ds:
        bygrp[x["fp"]].append(x)
    groups = []
    for fp, xs in sorted(bygrp.items()):
        sc = [x["d_score_rel"] for x in xs if x["d_score_rel"] is not None]
        groups.append({
            "fp": fp, "n_pairs": len(xs),
            "mean_d_mtp_rel": sum(x["mean_d_mtp_rel"] for x in xs) / len(xs),
            "mean_d_score_rel": (sum(sc) / len(sc)) if sc else None,
            "streams": sorted({(x["streams_lo"], x["streams_hi"]) for x in xs}),
        })
    vals = [g["mean_d_mtp_rel"] for g in groups]
    n = len(vals)
    mean = sum(vals) / n
    sd = statistics.stdev(vals) if n >= 2 else None
    se = (sd / math.sqrt(n)) if sd else None

    # Per-prompt sign consistency across ALL pairs, excluding the control prompt.
    perp = collections.defaultdict(list)
    for x in ds:
        for e in x["per_prompt"]:
            perp[e["sha"]].append(e["d_mtp_rel"])
    ctrl = min(perp, key=lambda k: floor["per_prompt_noise"][k]["rho"]
               if k in floor["per_prompt_noise"] else 9e9)
    prompts = []
    for k, v in perp.items():
        pn = floor["per_prompt_noise"].get(k, {})
        m = sum(v) / len(v)
        prompts.append({
            "sha": k, "rho": pn.get("rho"), "is_control": k == ctrl,
            "mean_d_mtp_rel": m,
            "sd_d_mtp_rel": statistics.stdev(v) if len(v) >= 2 else None,
            "replication_rel_sd": pn.get("rel_sd"),
            "expected_pair_rel_sd": (math.sqrt(2) * pn["rel_sd"]
                                     if pn.get("rel_sd") else None),
        })
    prompts.sort(key=lambda p: -(p["rho"] or 0))
    drafting = [p for p in prompts if not p["is_control"]]
    pos = sum(1 for p in drafting if p["mean_d_mtp_rel"] > 0)
    return {
        "n_pairs": len(ds), "n_groups": n,
        "groups": groups,
        "cluster_mean_d_mtp_rel": mean,
        "cluster_sd": sd, "cluster_se": se,
        "cluster_t": (mean / se) if se else None,
        "groups_positive": sum(1 for v in vals if v > 0),
        "per_prompt": prompts,
        "drafting_prompts_positive": pos,
        "drafting_prompts_total": len(drafting),
        # Two-sided sign test over drafting prompts. These prompts are not
        # independent of each other either (they share the same pairs), so this
        # is a consistency statement about direction, NOT an independent p-value.
        "sign_test_two_sided_p": (
            2 * sum(math.comb(len(drafting), k)
                    for k in range(max(pos, len(drafting) - pos),
                                   len(drafting) + 1))
            / 2 ** len(drafting)),
    }


def negative_control(doses):
    """The near-non-drafting prompt must move less than the drafting-heavy ones.

    Cheapest available falsification: a width-8 dispatch cell can only matter in
    rounds that actually dispatch at width 8. The prompt with rho 0.1540 and 449
    non-drafting rounds reaches wide widths far less often than a prompt with
    rho 5.78 and zero non-drafting rounds.
    """
    out = []
    for d in doses:
        per = d["per_prompt"]
        idx = min(range(len(per)), key=lambda i: per[i]["rho"])
        heavy = [i for i in range(len(per)) if i != idx]
        control = abs(per[idx]["d_mtp_rel"])
        others = [abs(per[i]["d_mtp_rel"]) for i in heavy]
        out.append({
            "fp": d["fp"], "width": d["width"],
            "streams": [d["streams_lo"], d["streams_hi"]],
            "control_sha": per[idx]["sha"], "control_rho": per[idx]["rho"],
            "control_nondraft": per[idx]["nondraft"],
            "control_abs_d_mtp_rel": control,
            "heavy_mean_abs_d_mtp_rel": sum(others) / len(others),
            "control_is_smaller": control < sum(others) / len(others),
            "control_rank": sorted(
                range(len(per)),
                key=lambda i: abs(per[i]["d_mtp_rel"])).index(idx),
        })
    return out


LADDER_TREE = "04ad6bf1"   # E25/E27 instrument tree that e43.LOCAL_LADDER was measured on


def streams_refit(rows, census):
    """(a) Refit the width ladder on source-derived streams(M).

    THE TREE MATTERS AND IT IS NOT HEAD. `e43.LOCAL_LADDER` is annotated
    "E25/E27 instrument", so it was measured while E27 had M5=5/M9=5 in the
    dispatch table. That tree is 04ad6bf1: streams 1,1,1,2,2,2,2, ONE boundary
    at 5->6. The tree we ship is 1,1,2,2,2,2,3 with boundaries at 4->5 and 8->9.

    Fitting HEAD's stream vector to this ladder is a tree mismatch and produces a
    NEGATIVE stream coefficient -- more streams apparently making the kernel
    faster -- which is how the mismatch announces itself. Both vectors are fitted
    here so the error stays visible instead of being silently corrected.

    On 04ad6bf1 the indicator [M>=6] and streams(M) are the SAME one-parameter
    regressor, since streams = 1 + [M>=6] there. That is why r1's step-at-6 fit
    this ladder well: it was the stream model in disguise, correct for the
    ladder's tree and wrong as a statement about the shipped tree.
    """
    st_head = C.streams(C.dispatch_table("HEAD"))
    tbl_ladder = C.dispatch_table(LADDER_TREE)
    st_ladder = C.streams(tbl_ladder) if tbl_ladder else None
    tbl = C.dispatch_table("HEAD")
    st = st_head
    ladder = {m: t for m, t in sorted(e43.LOCAL_LADDER.items()) if m in st}
    ms = sorted(ladder)

    def ols(cols, y):
        n = len(cols)
        A = [[sum(cols[i][k] * cols[j][k] for k in range(len(y)))
              for j in range(n)] for i in range(n)]
        b = [sum(cols[i][k] * y[k] for k in range(len(y))) for i in range(n)]
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[p] = A[p], A[i]
            b[i], b[p] = b[p], b[i]
            if abs(A[i][i]) < 1e-12:
                return None
            for r in range(n):
                if r == i:
                    continue
                f = A[r][i] / A[i][i]
                for c in range(i, n):
                    A[r][c] -= f * A[i][c]
                b[r] -= f * b[i]
        return [b[i] / A[i][i] for i in range(n)]

    y = [ladder[m] for m in ms]
    one = [1.0] * len(ms)
    lin = [float(m) for m in ms]
    quad = [float(m * m) for m in ms]
    strm = [float(st[m]) for m in ms]

    def fit(name, cols, labels):
        beta = ols(cols, y)
        if beta is None:
            return {"name": name, "feasible": False}
        pred = [sum(beta[i] * cols[i][k] for i in range(len(cols)))
                for k in range(len(ms))]
        resid = [y[k] - pred[k] for k in range(len(ms))]
        return {
            "name": name, "feasible": True,
            "coef": dict(zip(labels, beta)),
            "pred": {str(ms[k]): pred[k] for k in range(len(ms))},
            "resid": {str(ms[k]): resid[k] for k in range(len(ms))},
            "max_abs_resid": max(abs(r) for r in resid),
            "rms_resid": math.sqrt(sum(r * r for r in resid) / len(resid)),
        }

    strm_ladder = ([float(st_ladder[m]) for m in ms] if st_ladder else None)
    fits = [
        fit("streams_HEAD_tree_MISMATCHED", [one, strm, lin],
            ["const", "stream", "M"]),
        fit("linear", [one, lin], ["const", "M"]),
        fit("quadratic", [one, lin, quad], ["const", "M", "M2"]),
    ]
    if strm_ladder:
        fits.insert(0, fit("streams_ladder_tree_%s" % LADDER_TREE,
                           [one, strm_ladder, lin], ["const", "stream", "M"]))
    # r1's misplaced indicator, kept so the correction is auditable.
    for br in (5, 6, 7):
        ind = [1.0 if m >= br else 0.0 for m in ms]
        fits.append(fit("step_at_%d" % br, [one, ind, lin],
                        ["const", "step", "M"]))
    d1 = {str(m): ladder[m] - ladder[m - 1] for m in ms if m - 1 in ladder}
    return {
        "ladder_tree": LADDER_TREE,
        "ladder_tree_streams": ({str(k): v for k, v in sorted(st_ladder.items())}
                                if st_ladder else None),
        "ladder_tree_boundaries": ([list(b) for b in C.boundaries(st_ladder)]
                                   if st_ladder else None),
        "step_at_6_equals_ladder_stream_model": (
            st_ladder is not None
            and all((st_ladder[m] - 1) == (1 if m >= 6 else 0) for m in ms)),
        "head_table": {str(k): v for k, v in sorted(tbl.items())},
        "head_streams": {str(k): v for k, v in sorted(st.items())},
        "boundaries": [list(b) for b in C.boundaries(st)],
        "ladder": {str(m): ladder[m] for m in ms},
        "first_differences": d1,
        "quadratic_falsified_by_nonmonotone_d1": (
            any(list(d1.values())[i] > list(d1.values())[i + 1]
                for i in range(len(d1) - 1))),
        "max_d1_drop": max(
            (list(d1.values())[i] - list(d1.values())[i + 1]
             for i in range(len(d1) - 1)), default=None),
        "fits": fits,
        "stream_minimality": {
            str(m): {"legal_ipgs": legal_ipgs(m), "min_streams": min_streams(m),
                     "shipped": st[m], "is_minimal": st[m] == min_streams(m)}
            for m in ms},
    }


# ------------------------------------------------------------------ self-test

def self_test():
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    ck("legal_ipg_M3", legal_ipgs(3) == [3, 4], legal_ipgs(3))
    ck("legal_ipg_M4", legal_ipgs(4) == [2, 4], legal_ipgs(4))
    ck("legal_ipg_M5", legal_ipgs(5) == [3], legal_ipgs(5))
    ck("legal_ipg_M6", legal_ipgs(6) == [2, 3, 4], legal_ipgs(6))
    ck("legal_ipg_M7", legal_ipgs(7) == [4], legal_ipgs(7))
    ck("legal_ipg_M8", legal_ipgs(8) == [2, 3, 4], legal_ipgs(8))
    ck("legal_ipg_M9", legal_ipgs(9) == [3], legal_ipgs(9))
    ck("min_streams_M8_is_2", min_streams(8) == 2, min_streams(8))
    ck("min_streams_M9_is_3", min_streams(9) == 3, min_streams(9))
    ck("min_streams_M5_is_2", min_streams(5) == 2, min_streams(5))

    ck("spread_none_on_empty", spread([]) is None)
    s = spread([1.0, 3.0])
    ck("spread_mean", s["mean"] == 2.0, s["mean"])
    ck("spread_range", s["range"] == 2.0, s["range"])
    ck("spread_sd", abs(s["sd"] - math.sqrt(2)) < 1e-12, s["sd"])
    ck("spread_sd_none_at_n1", spread([5.0])["sd"] is None)
    ck("spread_rel_range", abs(s["rel_range"] - 1.0) < 1e-12, s["rel_range"])

    a = ["case 8:", "  qmv_fast_crossrow_affine4_g64_m<T, 8, 4>(", "  break;"]
    b = ["case 8:", "  qmv_fast_crossrow_affine4_g64_m<T, 8, 3>(", "  break;"]
    kind, nc, no = classify_kernel_diff([a, a], [b, b])
    ck("dispatch_only_detected", kind == "dispatch-only", kind)
    ck("dispatch_only_cell_lines", nc == 4, nc)
    ck("dispatch_only_no_other", no == 0, no)
    c = b + ["METAL_FUNC void extra_kernel() {", "  int z = 1;", "}"]
    kind2, _, no2 = classify_kernel_diff([a, a], [c, c])
    ck("extra_kernel_is_confounded", kind2 == "confounded", kind2)
    ck("confounded_counts_other", no2 > 0, no2)
    kind3, _, _ = classify_kernel_diff([a, a], [a, a])
    ck("identical_kernel", kind3 == "identical", kind3)
    kind4, _, _ = classify_kernel_diff([None, a], [b, b])
    ck("unreadable_fails_closed", kind4 == "unreadable", kind4)

    # A pair differing ONLY by switch scaffolding is not a stream contrast.
    d = ["case 8:", "  qmv_fast_crossrow_affine4_g64_m<T, 8, 4>(", "  break;",
         "}"]
    kind5, _, _ = classify_kernel_diff([a, a], [d, d])
    ck("scaffolding_only_is_dispatch_only", kind5 == "dispatch-only", kind5)

    pa = {"per_prompt": [{"sha": "x", "rho": 1.0, "nondraft": 0}]}
    pb = {"per_prompt": [{"sha": "x", "rho": 1.0, "nondraft": 0}]}
    pc = {"per_prompt": [{"sha": "x", "rho": 1.0000001, "nondraft": 0}]}
    pd = {"per_prompt": [{"sha": "y", "rho": 1.0, "nondraft": 0}]}
    ck("work_identical_true", work_identical(pa, pb))
    ck("work_identical_rho_strict", not work_identical(pa, pc))
    ck("work_identical_prompt_strict", not work_identical(pa, pd))

    per = [{"sha": "p", "rho": 4.0, "nondraft": 0,
            "mtp_a": 0.04, "mtp_b": 0.05, "d_mtp_rel": 0.25,
            "serial_a": 0.08, "serial_b": 0.08, "d_serial_rel": 0.0,
            "ratio_a": 2.0, "ratio_b": 1.6, "d_ratio_rel": -0.2}]
    f = flip(per)[0]
    ck("flip_swaps_arms", f["mtp_a"] == 0.05 and f["mtp_b"] == 0.04, f["mtp_a"])
    ck("flip_inverts_delta", abs(f["d_mtp_rel"] - (-0.2)) < 1e-12,
       f["d_mtp_rel"])
    ck("flip_is_involution",
       abs(flip(flip(per))[0]["d_mtp_rel"] - 0.25) < 1e-12,
       flip(flip(per))[0]["d_mtp_rel"])
    ck("flip_ratio_delta", abs(f["d_ratio_rel"] - 0.25) < 1e-12,
       f["d_ratio_rel"])

    ck("census_streams_formula", C.streams({8: 3}) == {8: 3}, C.streams({8: 3}))
    ck("census_streams_ceil", C.streams({8: 4}) == {8: 2}, C.streams({8: 4}))
    ck("boundaries_detect", C.boundaries({4: 1, 5: 2}) == [(4, 5, 1, 2)],
       C.boundaries({4: 1, 5: 2}))

    ok = sum(1 for _, o, _ in checks if o)
    for name, o, detail in checks:
        if not o:
            print("FAIL %-40s %s" % (name, detail))
    print("self-test: %d/%d passed" % (ok, len(checks)))
    return 0 if ok == len(checks) else 1


# ------------------------------------------------------------------------ run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--refresh-trees", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.run:
        ap.print_help()
        return 2

    t0 = time.time()

    def stage(msg):
        print("[%7.1fs] %s" % (time.time() - t0, msg), file=sys.stderr)

    stage("tree census")
    census = tree_census(args.refresh_trees)
    stage("board rows")
    rows = load_rows()
    stage("replication floor")
    floor = replication_floor(rows, census)
    stage("tree arms")
    arms = tree_arms(rows, census)
    stage("a/b pairs")
    pairs = ab_pairs(arms)
    stage("doses")
    doses = {str(w): dose_at_width(arms, pairs, w) for w in (4, 8)}
    stage("dose summary")
    summaries = {w: dose_summary(d, floor) for w, d in doses.items()}
    stage("negative control")
    negs = {w: negative_control(d) for w, d in doses.items()}
    stage("streams refit")
    refit = streams_refit(rows, census)

    payload = {
        "provenance": {
            "base_sha": C.run(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "trees_scanned": census["scanned"],
            "trees_with_table": census["with_table"],
            "distinct_fingerprints": len({c["fp"] for c in
                                          census["trees"].values()}),
            "board_rows_total": len(json.loads(CORPUS.read_text())["submissions"]),
            "board_rows_with_metrics": len(rows),
            "distinct_trees_with_metrics": len(arms),
        },
        "identifiability": {
            "n_8_observable": False,
            "per_prompt_metric_keys": sorted(
                rows[0]["per_prompt"][0].keys()) if rows else [],
            "note": ("no per-width round histogram exists in officialMetrics, "
                     "so delta_T(8) in ms is not identified; only the leg-level "
                     "product n_8(p)*delta_T(8) is"),
        },
        "replication_floor": floor,
        "pair_classification": dict(collections.Counter(
            "%s|%s" % (p["kind"], p["widths"]) for p in pairs)),
        "pairs": pairs,
        "doses": doses,
        "dose_summary": summaries,
        "negative_control": negs,
        "streams_refit": refit,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    stage("wrote %s" % OUT)

    print("trees scanned                      : %d" % census["scanned"])
    print("trees with a dispatch table        : %d" % census["with_table"])
    print("distinct non-kernel fingerprints   : %d"
          % payload["provenance"]["distinct_fingerprints"])
    print()
    print("board rows with officialMetrics    : %d" % len(rows))
    print("distinct trees behind them         : %d" % len(arms))
    print()
    print("REPLICATION FLOOR (byte-identical trees, repeated submissions)")
    print("  replicate sets                   : %d over %d rows"
          % (floor["n_sets"], floor["n_rows"]))
    if floor["pooled_rel_sd"] is not None:
        print("  pooled within-tree score rel sd  : %.4f %%"
              % (100 * floor["pooled_rel_sd"]))
        print("  widest within-tree score range   : %.4f %%"
              % (100 * floor["max_rel_range"]))
        print("  campaign recorded sigma_score    : %.4f %%"
              % floor["recorded_sigma_score_pct"])
        print("  crown gap                        : %.4f %%"
              % floor["crown_gap_pct"])
    print()
    print("  covariate-invariant sets         : %d / %d"
          % (floor["n_sets_all_covariates_invariant"], floor["n_sets"]))
    print("  per-prompt replication rel sd:")
    for k, v in sorted(floor["per_prompt_noise"].items(),
                       key=lambda kv: -kv[1]["rho"]):
        print("     %s rho %6.4f -> %.4f %%" % (k, v["rho"], 100 * v["rel_sd"]))
    print()
    for w in ("8", "4"):
        s = summaries[w]
        print("DOSE AT WIDTH %s: %d clean dispatch-only work-identical pairs "
              "in %d independent fingerprint groups" % (w, s["n_pairs"],
                                                        s["n_groups"]))
        for g in s["groups"]:
            print("   fp %s  %d pair(s)  mean d_mtp %+.4f %%  d_score %s"
                  % (g["fp"], g["n_pairs"], 100 * g["mean_d_mtp_rel"],
                     ("%+.4f %%" % (100 * g["mean_d_score_rel"]))
                     if g["mean_d_score_rel"] is not None else "n/a"))
        print("   CLUSTER: mean %+.4f %%  sd %.4f %%  se %.4f %%  t = %.2f  "
              "(%d/%d groups positive)"
              % (100 * s["cluster_mean_d_mtp_rel"], 100 * s["cluster_sd"],
                 100 * s["cluster_se"], s["cluster_t"],
                 s["groups_positive"], s["n_groups"]))
        print("   drafting prompts positive: %d/%d  (sign consistency p=%.4f)"
              % (s["drafting_prompts_positive"], s["drafting_prompts_total"],
                 s["sign_test_two_sided_p"]))
        print("   per-prompt effect vs matched replication noise:")
        for p in s["per_prompt"]:
            print("     %s rho %6.4f %s d_mtp %+7.4f %%  pair sd %s  "
                  "expected pair sd %s"
                  % (p["sha"], p["rho"] or 0,
                     "CONTROL" if p["is_control"] else "       ",
                     100 * p["mean_d_mtp_rel"],
                     ("%.4f %%" % (100 * p["sd_d_mtp_rel"]))
                     if p["sd_d_mtp_rel"] else "n/a",
                     ("%.4f %%" % (100 * p["expected_pair_rel_sd"]))
                     if p["expected_pair_rel_sd"] else "n/a"))
        n = negs[w]
        print("   negative control passes: %d / %d pairs"
              % (sum(1 for x in n if x["control_is_smaller"]), len(n)))
        print()
    r = refit
    print("STREAMS REFIT")
    print("  ladder tree %s streams %s boundaries %s"
          % (r["ladder_tree"], r["ladder_tree_streams"],
             r["ladder_tree_boundaries"]))
    print("  HEAD          streams %s boundaries %s"
          % (r["head_streams"], r["boundaries"]))
    print("  step_at_6 IS the ladder-tree stream model: %s"
          % r["step_at_6_equals_ladder_stream_model"])
    for f in r["fits"]:
        if f["feasible"]:
            print("  %-34s max|resid| %8.4f  rms %8.4f  %s"
                  % (f["name"], f["max_abs_resid"], f["rms_resid"],
                     " ".join("%s=%.4f" % kv for kv in f["coef"].items())))
    print("  shipped table stream-minimal at all widths: %s"
          % all(v["is_minimal"] for v in r["stream_minimality"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
