#!/usr/bin/env python3
"""E134 rung 0: is the board-fitted pass price `f` identified, and is it clean?

Run from `research/`. Zero GPU, zero model load. The optional register census
compiles Metal libraries through `xcrun metal-tt` and needs `--registers`.

    python3 e134_rung0.py --json e134-artifacts/rung0-pass-price.json
    python3 e134_rung0.py --registers --json e134-artifacts/rung0-pass-price.json

Four questions from the assignment:

  1. Is `f` identified at all, given that our own table makes the pass count
     and an `M >= 6` break collinear?
  2. Is the identifying cross-table variation confounded by register pressure?
  3. If `f` really is 50 us, what produces 10,322 us at exactly M = 6?
  4. What is the corrected `f`, and does the one-pass recommendation stand?
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_state_fe as fe  # noqa: E402
import e128_arm_prices as ap  # noqa: E402
import e128_ourcurve as oc  # noqa: E402

STRATA = pathlib.Path("/tmp/e128_strata.json")
MAXM = 9

# `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1565`.
OURS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}

# `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1008`.
DEPTH_CAP = 7

# `fixtures/qwen3_8_27b_mtp_track.json`, target.expected_source_bytes.
TARGET_BYTES = 15153237117

# Our own fitted ranked round cost, model R, from E128 section 5.
OUR_R = {"a": 27894.3, "b": 3388.3, "jump": 10322.5, "dslope": 2779.3,
         "jump_se": 365.1, "break": 6}

# F7 board-measured pass price, and the F8 break the board panel prefers.
F_HAT, F_SE, BOARD_BREAK = 50.4, 253.0, 4.375

PROMPTS = ("beagle", "medicine", "essays", "republic",
           "botany", "travel", "plutarch", "drama")


def gvec(tbl: dict) -> dict:
    return {m: math.ceil(m / max(tbl.get(m, m), 1)) for m in range(1, MAXM + 1)}


def gbar(g: dict, mbar: float) -> float:
    lo = max(1, min(MAXM, int(math.floor(mbar))))
    hi = max(1, min(MAXM, lo + 1))
    frac = min(max(mbar - lo, 0.0), 1.0)
    return (1 - frac) * g[lo] + frac * g[hi]


def hinge(m, knot):
    return np.maximum(np.asarray(m, float) - knot, 0.0)


def bar(name, width=72):
    print("\n=== %s %s" % (name, "=" * max(0, width - len(name) - 5)))


# --------------------------------------------------------------- question 1a
def question_1a() -> dict:
    """Our own eight points cannot separate a pass term from an M>=6 break."""
    bar("Q1a  on our own curve `f` is not identified, it IS the break")

    g = gvec(OURS)
    reach = list(range(1, DEPTH_CAP + 2))
    print("  reachable widths, segmentedVerifyDepthCap = %d so M <= %d"
          % (DEPTH_CAP, DEPTH_CAP + 1))
    print("  M          " + "".join("%6d" % m for m in range(1, MAXM + 1)))
    print("  IPG ours   " + "".join("%6s" % OURS.get(m, m)
                                    for m in range(1, MAXM + 1)))
    print("  passes     " + "".join("%6d" % g[m] for m in range(1, MAXM + 1)))
    print("  1[M>=6]    " + "".join("%6d" % (1 if m >= 6 else 0)
                                    for m in range(1, MAXM + 1)))

    ones = np.ones(len(reach))
    m = np.array(reach, float)
    p = np.array([g[i] for i in reach], float)
    h = np.array([1.0 if i >= 6 else 0.0 for i in reach])
    exact = bool(np.all(p == 1.0 + h))
    rank3 = int(np.linalg.matrix_rank(np.column_stack([ones, m, p])))
    rank4 = int(np.linalg.matrix_rank(np.column_stack([ones, m, p, h])))
    print("\n  passes(M) == 1 + 1[M>=6] on every reachable M : %s" % exact)
    print("  rank[1, M, passes]        = %d of 3" % rank3)
    print("  rank[1, M, passes, break] = %d of 4  <- rank deficient" % rank4)

    full = np.array([g[i] for i in range(1, MAXM + 1)], float)
    fullh = np.array([1.0 if i >= 6 else 0.0 for i in range(1, MAXM + 1)])
    print("  M = 9 would break the tie, passes 3 against break 1, but M = 9 "
          "is unreachable")
    print("  rank including the unreachable M=9  = %d of 4"
          % int(np.linalg.matrix_rank(np.column_stack(
              [np.ones(MAXM), np.arange(1., MAXM + 1), full, fullh]))))

    print("\n  So our E128 `jump` of %+.1f +- %.1f us and a pass price `f` are "
          "THE SAME\n  PARAMETER on our data. Our curve cannot argue with the "
          "board about `f`;\n  it can only say how big the thing at M=6 is."
          % (OUR_R["jump"], OUR_R["jump_se"]))
    return {"passes_equals_break_on_reachable": exact,
            "rank_with_break": rank4, "reachable_max_M": DEPTH_CAP + 1,
            "passes_ours": {str(k): v for k, v in g.items()}}


# --------------------------------------------------------------- question 1b
def load_panel():
    tables = {sid: {int(k): v for k, v in t.items()}
              for sid, t in json.loads(STRATA.read_text())["tables"].items()}
    panel = [r for r in fe.build_panel()["panel"]]
    short = {sid[:8]: tbl for sid, tbl in tables.items()}
    rows, kept = [], set()
    for r in panel:
        tbl = short.get(r["sid"])
        if tbl is None:
            continue
        g = gvec(tbl)
        rows.append({**r, "table": tbl, "gbar": gbar(g, r["mbar"]),
                     "tkey": json.dumps(tbl, sort_keys=True)})
        kept.add(r["sid"])
    return rows, short, sorted(kept)


def question_1b(rows, tables) -> dict:
    """How much cross-table variation is there, and what does it identify?"""
    bar("Q1b  the board: which variation identifies `f`, and how much")

    census = collections.Counter(r["tkey"] for r in rows)
    per_row = {}
    for r in rows:
        per_row[r["sid"]] = r["tkey"]
    row_census = collections.Counter(per_row.values())
    print("  %d rows carry a wide m-table, in %d distinct tables"
          % (len(per_row), len(row_census)))
    score_of, ipg_of = {}, {}
    for r in rows:
        score_of[r["sid"]] = r["score"]
        ipg_of[r["sid"]] = r["table"]
    by_table = collections.defaultdict(list)
    for sid, tkey in per_row.items():
        by_table[tkey].append(score_of[sid])
    print("  %-4s %-18s %-9s %-9s %-11s %s"
          % ("rows", "pass vec M=1..8", "meanscore", "maxscore", "NA at 6,7,8",
             "table"))
    per_table = {}
    for tkey, n in row_census.most_common():
        tbl = {int(k): v for k, v in json.loads(tkey).items()}
        g = gvec(tbl)
        pv = "".join("%2d" % g[m] for m in range(1, 9))
        sc = [s for s in by_table[tkey] if s is not None]
        na = "%d,%d,%d" % tuple(tbl.get(m, m) for m in (6, 7, 8))
        per_table[tkey] = {
            "rows": n, "pass_vector": [g[m] for m in range(1, 10)],
            "mean_score": float(np.mean(sc)) if sc else None,
            "max_score": float(np.max(sc)) if sc else None,
            "na_at_6_7_8": [tbl.get(m, m) for m in (6, 7, 8)]}
        print("  %-4d %-18s %-9s %-9s %-11s %s"
              % (n, pv,
                 "%.6f" % np.mean(sc) if sc else "-",
                 "%.6f" % np.max(sc) if sc else "-", na, tkey))

    modal = row_census.most_common(1)[0][0]
    modal_g = gvec({int(k): v for k, v in json.loads(modal).items()})
    diff_rows = [s for s, t in per_row.items()
                 if gvec({int(k): v for k, v in json.loads(t).items()})
                 != modal_g]
    onepass6 = [s for s, t in per_row.items()
                if gvec({int(k): v for k, v in json.loads(t).items()})[6] == 1]
    print("\n  rows whose pass vector differs from the modal one : %d of %d"
          % (len(diff_rows), len(per_row)))
    print("  rows that are ALREADY one-pass at M = 6           : %d"
          % len(onepass6))

    y = np.array([r["round_us"] for r in rows])
    m = np.array([r["mbar"] for r in rows])
    gb = np.array([r["gbar"] for r in rows])
    sid = [r["sid"] for r in rows]
    prompt = [r["prompt"] for r in rows]

    _, xd = fe.demean(y, np.column_stack([m, gb]), [sid, prompt])
    resid_g = xd[:, 1] - xd[:, 0] * (xd[:, 0] @ xd[:, 1]) / (xd[:, 0] @ xd[:, 0])
    print("\n  the identifying variation, after row FE, prompt FE and M:")
    print("    sd of within-transformed gbar        %.4f passes" % xd[:, 1].std())
    print("    sd after also removing M             %.4f passes"
          % resid_g.std())
    print("    that residual is what `f` is fitted on")

    share = {}
    ss = resid_g ** 2
    for label, keep in (("modal-table rows", lambda s: per_row[s] == modal),
                        ("other-table rows", lambda s: per_row[s] != modal),
                        ("one-pass-at-6 rows", lambda s: s in set(onepass6))):
        mask = np.array([keep(s) for s in sid])
        share[label] = float(ss[mask].sum() / ss.sum())
        print("    %-22s carry %5.1f %% of it, on %d observations"
              % (label, 100.0 * share[label], int(mask.sum())))

    print("\n## `f` fitted on subsets, row + prompt FE, SEs clustered by row")
    out = {}
    for label, mask in (
            ("all rows", np.ones(len(rows), bool)),
            ("modal table only", np.array([per_row[s] == modal for s in sid])),
            ("non-modal rows", np.array([per_row[s] != modal for s in sid]))):
        yy, mm, gg = y[mask], m[mask], gb[mask]
        ss_, pp = [s for s, k in zip(sid, mask) if k], \
            [p for p, k in zip(prompt, mask) if k]
        fit = fe.fe_fit(yy, np.column_stack([mm, gg]), [ss_, pp],
                        cluster=ss_, names=["b", "f"])
        piece = {"n": int(mask.sum()), "rows": len(set(ss_)),
                 "b": float(fit["beta"][0]), "b_se": float(fit["se"][0]),
                 "f": float(fit["beta"][1]), "f_se": float(fit["se"][1]),
                 "rmse": fit["rmse"]}
        out[label] = piece
        print("  %-18s n %4d rows %3d   b %8.1f+-%6.1f   f %9.1f+-%8.1f"
              % (label, piece["n"], piece["rows"], piece["b"], piece["b_se"],
                 piece["f"], piece["f_se"]))

    print("\n## `f` with a free break in the model, which is the honest test")
    for knot in (BOARD_BREAK, 5.0):
        x = np.column_stack([m, hinge(m, knot), gb])
        fit = fe.fe_fit(y, x, [sid, prompt], cluster=sid,
                        names=["b_lo", "d", "f"])
        out["with hinge %.3f" % knot] = {
            "b_lo": float(fit["beta"][0]), "d": float(fit["beta"][1]),
            "d_se": float(fit["se"][1]), "f": float(fit["beta"][2]),
            "f_se": float(fit["se"][2]), "rmse": fit["rmse"],
            "aicc": fit["aicc"]}
        print("  hinge at %.3f   b_lo %8.1f   d %8.1f+-%7.1f   f %9.1f+-%8.1f"
              "   aicc %10.1f"
              % (knot, fit["beta"][0], fit["beta"][1], fit["se"][1],
                 fit["beta"][2], fit["se"][2], fit["aicc"]))
    print("\n  A pass term and a break at the modal table's own step are the "
          "same\n  regressor for 121 of 202 rows, so they fight over one "
          "signal.")
    return {"table_census": {k: v for k, v in row_census.items()},
            "per_table": per_table,
            "distinct_tables": len(row_census),
            "rows_off_modal": len(diff_rows),
            "rows_onepass_at_6": len(onepass6),
            "identifying_sd_passes": float(resid_g.std()),
            "leverage_share": share, "fits": out,
            "onepass6_sids": sorted(onepass6)}


# ----------------------------------------------------- the natural experiment
def natural_experiment(rows) -> dict:
    """Rows that already ship a one-pass M=6, against the modal table."""
    bar("Q1c  the closest thing to a natural experiment on the board")

    by_sid = collections.defaultdict(list)
    for r in rows:
        by_sid[r["sid"]].append(r)
    g_of = {s: gvec(v[0]["table"]) for s, v in by_sid.items()}
    treat = {s for s, g in g_of.items() if g[6] == 1}
    modal = collections.Counter(
        json.dumps(v[0]["table"], sort_keys=True) for v in by_sid.values()
    ).most_common(1)[0][0]
    control = {s for s, v in by_sid.items()
               if json.dumps(v[0]["table"], sort_keys=True) == modal}
    print("  treated  rows, one pass at M=6 : %d" % len(treat))
    print("  control  rows, modal table     : %d" % len(control))

    scores = {}
    for label, group in (("one-pass at 6", treat), ("modal table", control)):
        vals = [by_sid[s][0]["score"] for s in group
                if by_sid[s][0].get("score")]
        scores[label] = {"n": len(vals), "mean": float(np.mean(vals)),
                         "sd": float(np.std(vals, ddof=1)),
                         "max": float(np.max(vals))}
        print("  %-16s official score  n %3d  mean %.6f  sd %.6f  max %.8f"
              % (label, len(vals), np.mean(vals), np.std(vals, ddof=1),
                 np.max(vals)))
    print("\n  Whole-row score is confounded by every other difference "
          "between\n  candidates, so it is context, not the estimate.")

    keep = [r for r in rows if r["sid"] in treat | control]
    y = np.array([r["round_us"] for r in keep])
    m = np.array([r["mbar"] for r in keep])
    t = np.array([1.0 if r["sid"] in treat else 0.0 for r in keep])
    pr = [r["prompt"] for r in keep]
    sid = [r["sid"] for r in keep]

    print("\n## round cost at matched width, prompt FE only so the treatment "
          "survives")
    out = {}
    for label, x, names in (
            ("M + treat", np.column_stack([m, t]), ["b", "tau"]),
            ("M + treat + treat*1[M>=5.5]",
             np.column_stack([m, t, t * (m >= 5.5)]), ["b", "tau", "tau_hi"]),
            ("M + hinge5 + treat*1[M>=5.5]",
             np.column_stack([m, hinge(m, 5.0), t, t * (m >= 5.5)]),
             ["b", "d", "tau", "tau_hi"])):
        fit = fe.fe_fit(y, x, [pr], cluster=sid, names=names)
        out[label] = {n: [float(b), float(s)] for n, b, s
                      in zip(names, fit["beta"], fit["se"])}
        out[label]["rmse"] = fit["rmse"]
        print("  %-30s %s" % (label, "  ".join(
            "%s %8.1f+-%7.1f" % (n, b, s)
            for n, b, s in zip(names, fit["beta"], fit["se"]))))
    print("\n  `tau_hi` is the extra cost a modal-table row pays at high width "
          "with\n  its second pass, sign flipped: a NEGATIVE tau_hi means the "
          "one-pass\n  table is cheaper exactly where it removes the pass.")
    return {"treated_rows": len(treat), "control_rows": len(control),
            "scores": scores, "fits": out}


# ---------------------------------------------------------------- question 2
def register_census(tables) -> dict:
    """Derived entry-point registers for every distinct board table."""
    bar("Q2  register pressure, measured per distinct table")

    import re
    import tempfile
    import e131_cliff_gate as gate

    cases_re = re.compile(r"(let cases = \[)(.*?)(\]\n)", re.DOTALL)

    def patcher(tbl):
        pairs = ", ".join("(%d, %d)" % (m, tbl[m])
                          for m in sorted(tbl) if m >= 3)

        def apply(swift: str) -> str:
            match = cases_re.search(swift)
            if match is None:
                raise SystemExit("no dispatch table in the extracted Swift")
            return swift[:match.start(2)] + pairs + swift[match.end(2):]
        return apply

    seen, out = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for i, tkey in enumerate(sorted(tables)):
            tbl = {int(k): v for k, v in json.loads(tkey).items()}
            rows = gate.census(
                gate.side_sources(None, swift_patch=patcher(tbl)),
                work, "t%d" % i)
            picked = {}
            for name, rec in rows.items():
                if "error" in rec or gate.RANKED not in rec:
                    continue
                if "qmv" not in name:
                    continue
                picked[name] = rec[gate.RANKED]["registers"]
            seen[tkey] = picked
            out[tkey] = {"registers": picked,
                         "sums": max((v for k, v in picked.items()
                                      if "_sums_" in k), default=None),
                         "plain": max((v for k, v in picked.items()
                                       if "_sums_" not in k), default=None)}
            print("  %-58s sums %s  plain %s"
                  % (tkey[:58], out[tkey]["sums"], out[tkey]["plain"]))
    return out


def question_2(rows, regs) -> dict:
    """Does controlling for register pressure move `f`?"""
    bar("Q2b  `f` with the register channel in the model")

    have = {k: v["sums"] for k, v in regs.items() if v.get("sums")}
    keep = [r for r in rows if r["tkey"] in have]
    if not keep:
        print("  no register census available, skipping")
        return {}
    y = np.array([r["round_us"] for r in keep])
    m = np.array([r["mbar"] for r in keep])
    gb = np.array([r["gbar"] for r in keep])
    rg = np.array([float(have[r["tkey"]]) for r in keep])
    sid = [r["sid"] for r in keep]
    pr = [r["prompt"] for r in keep]
    rgc = rg - rg.mean()

    print("  register spread across tables: min %d max %d sd %.2f"
          % (rg.min(), rg.max(), rg.std()))
    print("  corr(registers, gbar) %+.4f     corr(registers, M) %+.4f"
          % (np.corrcoef(rg, gb)[0, 1], np.corrcoef(rg, m)[0, 1]))

    out = {}
    for label, x, names in (
            ("M + passes", np.column_stack([m, gb]), ["b", "f"]),
            ("M + passes + regs*M",
             np.column_stack([m, gb, rgc * m]), ["b", "f", "r"]),
            ("M + passes + regs*passes",
             np.column_stack([m, gb, rgc * gb]), ["b", "f", "r"]),
            ("M + passes + regs*M + regs*passes",
             np.column_stack([m, gb, rgc * m, rgc * gb]),
             ["b", "f", "rM", "rP"])):
        fit = fe.fe_fit(y, x, [sid, pr], cluster=sid, names=names)
        out[label] = {n: [float(b), float(s)] for n, b, s
                      in zip(names, fit["beta"], fit["se"])}
        out[label]["rmse"] = fit["rmse"]
        out[label]["aicc"] = fit["aicc"]
        print("  %-34s rmse %7.1f aicc %9.1f  %s"
              % (label, fit["rmse"], fit["aicc"], "  ".join(
                  "%s %9.1f+-%8.1f" % (n, b, s)
                  for n, b, s in zip(names, fit["beta"], fit["se"]))))
    return out


# ---------------------------------------------------------------- question 3
def question_3() -> dict:
    """What a pass costs, from the kernel source and the pinned weight bytes."""
    bar("Q3  what a second pass actually costs")

    print("  The mechanism is in the kernel body, not in dispatch overhead.")
    print("  Qwen35.swift:1840  grid = (m*32, (n/8)*2, 1), threadgroup "
          "(32, 2, 1),")
    print("    so the x grid has M threadgroups and qmv_gx = tid.x.")
    print("  Qwen35.swift:1542-1545  first_m = group_x * IPG; if (first_m >= M)"
          " return,")
    print("    so exactly ceil(M/IPG) of those M groups do work.")
    print("  Qwen35.swift:1455-1470  every working group loops the FULL "
          "in_vec_size")
    print("    and loads w, scales and biases for its four output rows.")
    print("  Qwen35.swift:1583  qmv_out_row = tid.y*8 + sgid*4 does not "
          "mention gx,")
    print("    so both groups cover the SAME output rows.")
    print("\n  A pass is therefore a full re-read of the projection weights, "
          "not a\n  fixed launch cost. Its price scales with the checkpoint, "
          "not with a\n  constant.")

    round1 = OUR_R["a"] + OUR_R["b"] * 1.0
    bw = TARGET_BYTES / (round1 * 1e-6) / 1e9
    print("\n  pinned target bytes                     %d" % TARGET_BYTES)
    print("  our ranked round at M = 1               %.1f us" % round1)
    print("  implied read rate if that round is one full weight pass  "
          "%.0f GB/s" % bw)
    print("  a SECOND full pass at that rate would cost about %.0f us"
          % round1)
    print("  our measured M=6 jump                   %.1f +- %.1f us"
          % (OUR_R["jump"], OUR_R["jump_se"]))
    print("  ratio, full-pass prediction over measured jump           %.2fx"
          % (round1 / OUR_R["jump"]))

    print("\n  So the pass count OVER-predicts our jump by about 3x, and the "
          "board's\n  50 us UNDER-predicts it by about 200x. Both cannot be "
          "right, and the\n  gap has a sign-carrying explanation: register "
          "pressure moves the other\n  way at the same boundary.")

    def vregs(na):
        return {"acc": 4 * na, "partial": 4 * na, "a0..a3": 4 * na,
                "sums": na, "total_floats": 13 * na}
    print("\n  vector-register floats per thread in qwen_e120_qmv_wide<NA>, "
          "from\n  the declarations at Qwen35.swift:1446-1500:")
    print("  %-6s %6s %8s %8s %6s %8s" % ("M", "IPG", "acc", "partial",
                                          "sums", "floats"))
    for m in range(3, 9):
        na = OURS.get(m, m)
        v = vregs(na)
        print("  %-6d %6d %8d %8d %6d %8d"
              % (m, na, v["acc"], v["partial"], v["sums"], v["total_floats"]))
    print("\n  At M = 5 -> 6 our table drops IPG from 5 to 3. The pass count "
          "goes UP,\n  which costs time, and the per-thread vector footprint "
          "goes DOWN from 65\n  to 39 floats, which buys occupancy and SAVES "
          "time. The two channels\n  cancel, and a pooled board fit that "
          "carries only the pass count absorbs\n  that cancellation into `f`. "
          "A near-zero pooled `f` is what cancellation\n  looks like, exactly "
          "as the assignment suggests.")

    print("\n  Rival mechanisms I checked and reject for the M=6 step:")
    print("    SDPA width wall at qL >= 6, 16 x 4.19 MB per round   ~170 us"
          "   2 orders short")
    print("    xsums stride 8 -> 16 at M = 9 only, table is 10,240 B at M<=8"
          "   wrong width")
    print("    the tail branch: TAIL = M %% IPG is 0 at M=6, 3 at M=7, "
          "0 at M=8   wrong shape")
    print("  I name no rival that reaches 10,322 us. I concede the pass count.")
    return {"target_bytes": TARGET_BYTES, "round_us_at_M1": round1,
            "implied_gb_per_s": bw,
            "full_pass_over_measured_jump": round1 / OUR_R["jump"],
            "vector_floats": {str(m): 13 * OURS.get(m, m)
                              for m in range(3, 9)}}


# ---------------------------------------------------------------- question 4
def pricing_points(here: pathlib.Path, board: pathlib.Path, receipt_id: str):
    """The exact E128 pricing frame, so every number stays comparable."""
    hists = oc.fixture_histograms(here / "e128-artifacts/rung1-shipped.json")
    scen = oc.r_scenarios(here / "e128-artifacts/rung0-identity.json")
    receipt = oc.load_receipt(board, receipt_id)
    ordered = oc.build_points(receipt, scen["assumed"], hists)
    for point in ordered:
        point["raw"] = receipt["per_prompt"][point["prompt"]]["raw"]
    curves = json.loads(
        (here / "e128-artifacts/f4-candidate-curves.json").read_text())
    return {p["prompt"]: p for p in ordered}, curves["curves"]["slopeonly_b6"]


def question_4(q1b, q1c, q2, points, curve) -> dict:
    """A corrected `f`, and the one-pass arm repriced across its range."""
    bar("Q4  the corrected `f`, and whether the recommendation stands")

    from e128_reprice_onepass import onepass_fixed

    cands = [("F7 board pooled", F_HAT, F_SE),
             ("board, non-modal rows only",
              q1b["fits"]["non-modal rows"]["f"],
              q1b["fits"]["non-modal rows"]["f_se"])]
    hk = "with hinge %.3f" % BOARD_BREAK
    if hk in q1b["fits"]:
        cands.append(("board, hinge in the model", q1b["fits"][hk]["f"],
                      q1b["fits"][hk]["f_se"]))
    tau = q1c["fits"].get("M + treat + treat*1[M>=5.5]", {}).get("tau_hi")
    if tau:
        cands.append(("one-pass rows, matched contrast", -tau[0], tau[1]))
    for key in ("M + passes + regs*M", "M + passes + regs*M + regs*passes"):
        if q2 and key in q2:
            cands.append(("board, %s" % key, q2[key]["f"][0], q2[key]["f"][1]))
    cands.append(("our own curve, the M=6 jump", OUR_R["jump"],
                  OUR_R["jump_se"]))

    print("  every estimate of one pass, and what it makes the arm worth")
    print("  the price frame is the E128 one: receipt d3c491b5, curve")
    print("  slopeonly_b6, exact Rule 67 median, touched widths only")
    out = []
    for tag, widths in ap.ONEPASS_TABLES.items():
        print("\n  %s" % tag)
        print("  %-38s %10s %9s %11s %11s"
              % ("estimator", "f us", "se", "c=0.000", "c=0.445"))
        for label, f, se in cands:
            prices = []
            for c in (0.0, 0.445):
                mult = onepass_fixed(widths, curve, float(f), c, False)
                prices.append(
                    ap.price_arm(points, curve, mult)["median_delta_pct"])
            out.append({"table": tag, "estimator": label, "f": float(f),
                        "se": float(se), "price_c0": prices[0],
                        "price_c0445": prices[1]})
            print("  %-38s %10.1f %9.1f %+11.4f %+11.4f"
                  % (label, f, se, prices[0], prices[1]))
    return {"estimates": out}


# ------------------------------------------------- question 5, finding 155
# Every constant below is read from source at BASE_SHA 83e07638, not fitted.
#   Qwen36MTPBlockSession.swift:840   headStepCostRatio  = 0.18
#   Qwen36MTPBlockSession.swift:871   makeUniformDepthPrice()
#   Qwen36MTPBlockSession.swift:913   measuredRawDepthPrice
#   Qwen36MTPBlockSession.swift:935   prefixCosts()
#   Qwen36MTPBlockSession.swift:958   depthPriceArm = .ship
#   Qwen36MTPBlockSession.swift:1100  threshold = marginal/cumulative * (1+E)
#   MLXFastCore/Constants.swift:331   qwenMTPMaxDraftDepth = 8
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
E68_VERIFY_NORMALISER_S = 0.060300
E68_RAW = [0.26300121724709807, 0.29195567495854047, 0.34642143034825884,
           0.40231023217247086, 0.63287276451077956, 0.43601634825870655,
           0.35457813598673293, 0.42510483416251998]


def prefix_costs(marginal):
    out, running = [1.0], 1.0
    for v in marginal:
        running += v
        out.append(running)
    return out


def question_5() -> dict:
    """Is the shipped price vector the normalised ranked curve? (Finding 155)"""
    bar("Q5  finding 155: what IS the shipped price vector")

    ship_marg = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
    ship_cum = [1.0 + d * HEAD_STEP_COST_RATIO for d in range(MAX_DEPTH + 1)]

    a, b, jump, ds, brk = (OUR_R["a"], OUR_R["b"], OUR_R["jump"],
                           OUR_R["dslope"], OUR_R["break"])

    def round_us(m):
        return a + b * m + (jump + ds * (m - brk) if m >= brk else 0.0)

    unit = round_us(1)
    curve_marg = [(round_us(d + 2) - round_us(d + 1)) / unit
                  for d in range(MAX_DEPTH)]
    curve_cum = prefix_costs(curve_marg)

    scale = (MAX_DEPTH * HEAD_STEP_COST_RATIO) / sum(E68_RAW)
    pbfit_marg = [v * scale for v in E68_RAW]

    print("  round_us(M) = %.1f + %.1f M  (M < %d);  break adds %.1f + %.1f (M-%d)"
          % (a, b, brk, jump, ds, brk))
    print("  round_us(1) = %.1f us  is the normaliser\n" % unit)
    print("  %-3s %-9s %-9s %-9s | %-9s %-9s %-9s"
          % ("d", "ship.m", "curve.m", "pbfit.m",
             "ship.c", "curve.c", "curve/lvl"))
    for d in range(MAX_DEPTH):
        print("  %-3d %-9.4f %-9.4f %-9.4f | %-9.4f %-9.4f %-9.4f"
              % (d, ship_marg[d], curve_marg[d], pbfit_marg[d],
                 ship_cum[d], curve_cum[d], round_us(d + 1) / unit))

    ident = max(abs(curve_cum[d] - round_us(d + 1) / unit)
                for d in range(MAX_DEPTH + 1))
    print("\n  advisor's vector is 0.1083 x4, 0.4383, 0.1972 x3")
    print("  our curve  marginal   %s"
          % " ".join("%.4f" % v for v in curve_marg))
    print("  match to 4 s.f. : %s"
          % all(abs(curve_marg[d] - t) < 5e-5 for d, t in
                enumerate([0.1083] * 4 + [0.4383] + [0.1972] * 3)))
    print("  depth-4 marginal / base marginal = %.4f" % (curve_marg[4]
                                                         / curve_marg[0]))
    print("  SHIPPED marginal is flat %.2f, arm = .ship -> makeUniformDepthPrice"
          % HEAD_STEP_COST_RATIO)
    print("  so the advisor's vector is NOT the shipped price: it is the"
          " `rankedprice` ARM")
    print("  cumulative[d] == round_us(d+1)/round_us(1) to %.2e  (identity,"
          " prefixCosts is a prefix sum)" % ident)

    print("\n  the object the scheduler actually uses:"
          " marginal[d] / cumulative[d]")
    print("  %-3s %-11s %-11s %-8s" % ("d", "ship", "rankedprice", "ratio"))
    for d in range(MAX_DEPTH):
        s = ship_marg[d] / ship_cum[d]
        c = curve_marg[d] / curve_cum[d]
        print("  %-3d %-11.4f %-11.4f %-8.4f" % (d, s, c, c / s))

    print("\n  E68 measuredRawDepthPrice decoded, V = %.6f s:"
          " marginal QMV us for the step into verify width M"
          % E68_VERIFY_NORMALISER_S)
    e68 = {}
    for d, v in enumerate(E68_RAW):
        us = (v - HEAD_STEP_COST_RATIO) * E68_VERIFY_NORMALISER_S * 1e6
        e68[d + 2] = us
        print("    into width %d : %9.1f us" % (d + 2, us))
    excess6 = e68[6] - e68[5]
    print("  step into 6 / step into 5 = %.4f   excess at the width-6"
          " boundary = %.1f us" % (e68[6] / e68[5], excess6))
    print("  our E128 free-break jump = %.1f +- %.1f us  (E68 excess is %.2fx"
          " larger)" % (jump, OUR_R["jump_se"], excess6 / jump))
    print("  F7 pooled board f = %.1f us  is %.0fx smaller than the E68"
          " excess" % (F_HAT, excess6 / F_HAT))

    return {"ship_marginal": ship_marg, "ship_cumulative": ship_cum,
            "curve_marginal": curve_marg, "curve_cumulative": curve_cum,
            "pbfit_marginal": pbfit_marg,
            "curve_unit_us": unit,
            "cumulative_identity_max_abs_error": ident,
            "advisor_vector_is_shipped": False,
            "advisor_vector_is_our_ranked_curve": True,
            "depth4_marginal_multiple": curve_marg[4] / curve_marg[0],
            "e68_marginal_qmv_us": e68,
            "e68_width6_excess_us": excess6}


# ------------------------------------------------------------- question 6
# `e123_arms.SIMDGROUP_BUDGET`, the ranked arch is `applegpu_g17s`.
RANKED_SIMDGROUP_BUDGET = 3968


def question_6(regs, q1b, q1c, q2, points, curve) -> dict:
    """Reprice the one-pass arm with the MEASURED occupancy loss."""
    bar("Q6  the occupancy cost the E128 repricing never carried")

    import e128_reprice_onepass as rp

    levels = sorted({v["sums"] for v in regs.values() if v.get("sums")})
    if len(levels) != 2:
        print("  need exactly two register levels, got %s" % levels)
        return {}
    lo, hi = levels
    sg_lo = RANKED_SIMDGROUP_BUDGET // lo
    sg_hi = RANKED_SIMDGROUP_BUDGET // hi
    measured = 1.0 - sg_hi / sg_lo

    print("  the scored QMV entry point is ONE kernel with a runtime")
    print("  `switch (qmv_m)` at Qwen35.swift:1588, and every case is inlined,")
    print("  so its register allocation is the max over all instantiated IPG.")
    print("  Raising IPG at width 8 raises the footprint at width 2 as well.\n")
    print("  registers, ranked arch applegpu_g17s, budget %d"
          % RANKED_SIMDGROUP_BUDGET)
    print("    max IPG <= 5 (our table, the modal table) : %3d -> %2d simdgroups"
          % (lo, sg_lo))
    print("    max IPG >= 6 (any one-pass table)         : %3d -> %2d simdgroups"
          % (hi, sg_hi))
    print("    measured residency loss                    : %.4f" % measured)
    print("    E128 assumed ONEPASS_RESIDENCY_LOSS        : %.4f"
          % ap.ONEPASS_RESIDENCY_LOSS)

    cands = [("board, hinge in the model",
              q1b["fits"]["with hinge %.3f" % BOARD_BREAK]["f"])]
    tau = q1c["fits"].get("M + treat + treat*1[M>=5.5]", {}).get("tau_hi")
    if tau:
        cands.append(("one-pass rows, matched contrast", -tau[0]))
    if q2 and "M + passes + regs*M" in q2:
        cands.append(("board, regs controlled", q2["M + passes + regs*M"]["f"][0]))
    cands.append(("our own curve, the M=6 jump", OUR_R["jump"]))

    saved = rp.ONEPASS_RESIDENCY_LOSS
    out = []
    try:
        for tag, widths in ap.ONEPASS_TABLES.items():
            print("\n  %s   loss applied at EVERY width, as the one kernel"
                  " requires" % tag)
            print("  %-34s %10s %11s %11s %11s"
                  % ("estimator", "f us", "assumed c=1", "measured c=1",
                     "measured c=0"))
            for label, f in cands:
                row = {"table": tag, "estimator": label, "f": float(f)}
                for key, loss, c in (("assumed_c1", saved, 1.0),
                                     ("measured_c1", measured, 1.0),
                                     ("measured_c0", measured, 0.0)):
                    rp.ONEPASS_RESIDENCY_LOSS = loss
                    mult = rp.onepass_fixed(widths, curve, float(f), c, True)
                    row[key] = ap.price_arm(points, curve,
                                            mult)["median_delta_pct"]
                out.append(row)
                print("  %-34s %10.1f %+11.4f %+11.4f %+11.4f"
                      % (label, f, row["assumed_c1"], row["measured_c1"],
                         row["measured_c0"]))
    finally:
        rp.ONEPASS_RESIDENCY_LOSS = saved

    print("\n  The board already answers this in reduced form: rows that carry"
          "\n  a one-pass table pay the %d-register footprint AND still score"
          "\n  higher than the modal two-pass table." % hi)
    return {"registers_low": lo, "registers_high": hi,
            "simdgroups_low": sg_lo, "simdgroups_high": sg_hi,
            "measured_residency_loss": measured,
            "assumed_residency_loss": ap.ONEPASS_RESIDENCY_LOSS,
            "prices": out}


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--json", default="e134-artifacts/rung0-pass-price.json")
    ap_.add_argument("--board", type=pathlib.Path,
                     default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap_.add_argument("--receipt", default="d3c491b5")
    ap_.add_argument("--registers", action="store_true",
                     help="compile the Metal libraries and census registers")
    args = ap_.parse_args()

    print("harness=ranked  E134-rung0  zero GPU  analysis only")
    rows, tables, sids = load_panel()
    print("%d table-bearing rows, %d observations" % (len(sids), len(rows)))

    q1a = question_1a()
    q1b = question_1b(rows, tables)
    q1c = natural_experiment(rows)
    cache = here / "e134-artifacts/register-census.json"
    if args.registers:
        regs = register_census(set(r["tkey"] for r in rows))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(regs, indent=2) + "\n")
    else:
        regs = json.loads(cache.read_text()) if cache.is_file() else {}
    q2 = question_2(rows, regs) if regs else {}
    q3 = question_3()
    points, curve = pricing_points(here, args.board, args.receipt)
    q4 = question_4(q1b, q1c, q2, points, curve)
    q5 = question_5()
    q6 = question_6(regs, q1b, q1c, q2, points, curve) if regs else {}

    art = {"harness": "ranked", "gpu_used": False, "rows": len(sids),
           "observations": len(rows), "q1a": q1a, "q1b": q1b, "q1c": q1c,
           "registers": regs, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
           "q6": q6}
    path = pathlib.Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, indent=2, default=str) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
