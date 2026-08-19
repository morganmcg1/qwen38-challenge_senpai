#!/usr/bin/env python3
"""E46 synthesis: is the 20.291 ms step a weight STREAM or a GROUP WIDTH?

    python3 research/e46_analyze.py --base1 e46-base-r1 --arm1 e46-arm-r1 \
        --arm2 e46-arm-r2 --base2 e46-base-r2 \
        [--json-out research/e46-artifacts/e46-metrics.json]

Two independent readings of the same evidence:

  step 2   the width curve on the SHIPPED NA<=4 table, whose stream vector
           [1,1,2,2,2,2,3] moves the boundaries to 4->5 and 8->9 and removes
           the 5->6 boundary E41 fitted on. argmax d1 names the mechanism.
  step 3   two fixed-M contrasts where a*M cancels exactly:
             A  M=6 IPG 3->4   streams 2->2   width 3+3 -> 4+2
             B  M=8 IPG 4->3   streams 2->3   width 4+4 -> 3+3+2

Runs are ABBA (base, arm, arm, base): both builds share sweep position 2.5, so
the arithmetic mean of each pair cancels linear session drift exactly. That is
what makes the ungated thermal mode usable here. Every threshold comes from
research/e46_prereg.py, committed before the first GPU second.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e46_prereg as P  # noqa: E402

CURVE = ".mlxfast-private/qmv-curve/%s"
SHIPPED_IPG = dict(zip(P.WIDTHS, P.TABLES["01f69e1"]["ipg"]))
SHIPPED_STREAMS = dict(zip(P.WIDTHS, P.TABLES["01f69e1"]["streams"]))
ARM_IPG = {**SHIPPED_IPG, 6: 4, 8: 3}
CONTRAST_M = {"A": 6, "B": 8}


def streams(m: int, ipg: int) -> int:
    return math.ceil(m / ipg)


def load(tag: str):
    d = json.load(open(os.path.join(CURVE % tag, "vendored.json")))
    ident = {}
    ipath = os.path.join(CURVE % tag, "identity.txt")
    if os.path.exists(ipath):
        for line in open(ipath):
            line = line.strip()
            if line.startswith("run-qmv-curve: "):
                for part in line[len("run-qmv-curve: "):].split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        ident[k] = v
                continue
            m = re.match(r"^([a-z0-9_]+)=(.*)$", line)
            if m:
                ident[m.group(1)] = m.group(2)
    return d, ident


def t_of_m(d):
    """T(M) in seconds, and each scored shape's own seconds_per_call at each M."""
    total, per_shape = {}, {}
    for sh in d["shapes"]:
        cpv = sh["calls_per_verify"]
        for r in sh["rows"]:
            m = r["m"]
            total[m] = total.get(m, 0.0) + cpv * r["seconds_per_call"]
            per_shape.setdefault(m, {})[sh["name"]] = r["seconds_per_call"]
    return total, per_shape


def dispatch_at(d, m):
    paths, strm, ipg = set(), set(), set()
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == m:
                paths.add(r["in_kernel_path"])
                strm.add(r["weight_streams"])
                ipg.add(r["inputs_per_group"])
    return sorted(paths), sorted(strm), sorted(ipg)


def bitwise_bad(d):
    bad = []
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if (not r.get("row0_bitwise_matches_m1", True)
                    or r.get("row0_max_abs_delta_vs_m1", 0)):
                bad.append((sh["name"], r["m"], r.get("row0_max_abs_delta_vs_m1")))
    return bad


def jit_spread(d, m):
    """max over shapes of (mean - min)/min at width m, percent."""
    worst = 0.0
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == m and r["seconds_per_call_min"] > 0:
                worst = max(worst, (r["seconds_per_call"] - r["seconds_per_call_min"])
                            / r["seconds_per_call_min"] * 100.0)
    return worst


def fit(ms, ts, stream_vec):
    """Least squares T = c + b*streams + a*M; returns (a, b, c, max_abs_resid)."""
    X = np.array([[1.0, stream_vec[m], float(m)] for m in ms])
    y = np.array([ts[m] for m in ms])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(coef[2]), float(coef[1]), float(coef[0]), float(np.max(np.abs(resid)))


def sign_test_p(n_agree, n):
    """Two-sided exact sign-test p for n_agree of n in the same direction."""
    k = max(n_agree, n - n_agree)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base1", required=True)
    ap.add_argument("--arm1", required=True)
    ap.add_argument("--arm2", default="")
    ap.add_argument("--base2", default="")
    ap.add_argument("--e27-base", default="", help="prior-art base curve to compare")
    ap.add_argument("--e27-arm", default="", help="prior-art IPG-falsification arm")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    order = [("base1", args.base1), ("arm1", args.arm1)]
    if args.arm2:
        order.append(("arm2", args.arm2))
    if args.base2:
        order.append(("base2", args.base2))

    runs = {}
    for key, tag in order:
        d, ident = load(tag)
        t, per_shape = t_of_m(d)
        runs[key] = dict(tag=tag, d=d, ident=ident, t=t, per_shape=per_shape)

    base_keys = [k for k in ("base1", "base2") if k in runs]
    arm_keys = [k for k in ("arm1", "arm2") if k in runs]
    widths = sorted(set.intersection(*[set(runs[k]["t"]) for k in runs]))
    sweep = [m for m in widths if m in P.WIDTHS]

    out = {"tags": {k: runs[k]["tag"] for k in runs}, "widths": widths,
           "order": [t for _, t in order]}

    print("=" * 92)
    print("E46 -- weight streams vs group width, at fixed M, on the shipped table")
    print("=" * 92)

    print("\n[0] PROVENANCE  (ABBA: %s)" % " -> ".join(t for _, t in order))
    for key, _ in order:
        i = runs[key]["ident"]
        print(f"  {key:<6} {runs[key]['tag']:<14} head={i.get('head','?')[:12]} "
              f"dirty={i.get('dirty','?')} reps={i.get('reps','?')} "
              f"inner={i.get('inner','?')} host={i.get('host','?')}")
        print(f"         cool_gate={i.get('cool_gate_vendored','?')} "
              f"entry_C={i.get('gpu_temp_c_before_vendored','?')} "
              f"exit_C={i.get('gpu_temp_c_after_vendored','?')}")
    out["identity"] = {k: runs[k]["ident"] for k in runs}
    entry = [float(runs[k]["ident"][key]) for k in runs
             for key in ("gpu_temp_c_before_vendored",)
             if runs[k]["ident"].get(key, "").replace(".", "", 1).isdigit()]
    if entry:
        out["entry_temp_c"] = {"min": min(entry), "max": max(entry),
                               "spread": max(entry) - min(entry)}
        print(f"  entry temperature spread across arms: "
              f"{min(entry):.1f}..{max(entry):.1f} C "
              f"(delta {max(entry)-min(entry):.1f} C)")
    print("  cool_gate_passed_real_gate=false, gate_qualified_for_timing=false: "
          "directional causal evidence within this counterbalanced session, "
          "never a gate-qualified or ranked number.")

    print("\n[1] DISPATCH READBACK  (parsed from the header each build compiled)")
    ident_ok = True
    out["dispatch"] = {}
    for m in sweep:
        row = {}
        for key in runs:
            want_ipg = ARM_IPG[m] if key.startswith("arm") else SHIPPED_IPG[m]
            want_str = streams(m, want_ipg)
            paths, strm, ipg = dispatch_at(runs[key]["d"], m)
            good = (ipg == [want_ipg] and strm == [want_str]
                    and paths == ["qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>"
                                  % (m, want_ipg)])
            ident_ok &= good
            row[key] = dict(ok=good, path=paths, streams=strm, ipg=ipg,
                            want_ipg=want_ipg, want_streams=want_str)
        out["dispatch"][m] = row
        marks = " ".join(f"{k}:{'OK' if v['ok'] else 'BAD'}" for k, v in row.items())
        note = "  <-- contrast A" if m == 6 else ("  <-- contrast B" if m == 8 else "")
        print(f"  M={m}  base ipg={SHIPPED_IPG[m]} str={streams(m, SHIPPED_IPG[m])}"
              f"   arm ipg={ARM_IPG[m]} str={streams(m, ARM_IPG[m])}   {marks}{note}")
        for k, v in row.items():
            if not v["ok"]:
                print(f"        {k} got path={v['path']} ipg={v['ipg']} "
                      f"streams={v['streams']}; want ipg=[{v['want_ipg']}] "
                      f"streams=[{v['want_streams']}]")
    out["dispatch_readback_ok"] = bool(ident_ok)
    print(f"  every build dispatched as designed: "
          f"{'YES' if ident_ok else 'NO -- STOP, the contrast is not the one registered'}")

    print("\n[2] FIDELITY  (row0 bitwise vs M=1)")
    out["bitwise_failures"] = {}
    for key in runs:
        bad = bitwise_bad(runs[key]["d"])
        out["bitwise_failures"][key] = bad
        print(f"  {key:<6} failures: {len(bad)}" + (f"  {bad[:4]}" if bad else ""))

    print("\n[3] JIT LEAK CHECK  (mean-vs-min spread, %; the arm has 2 new "
          "instantiations)")
    out["jit_spread_pct"] = {}
    for m in sweep:
        vals = {k: jit_spread(runs[k]["d"], m) for k in runs}
        out["jit_spread_pct"][m] = vals
        note = "  <-- A" if m == 6 else ("  <-- B" if m == 8 else "")
        print("  M=%d  " % m + "  ".join(f"{k} {v:6.2f}" for k, v in vals.items())
              + note)

    # ---------------------------------------------------------------- step 2
    print("\n[4] STEP 2 -- width curve on the SHIPPED NA<=4 table, base builds only")
    print(f"  {'M':>3} {'streams':>8} {'IPG':>4} " +
          " ".join(f"{k:>10}" for k in base_keys) + f" {'mean(ms)':>10}")
    tbase = {}
    for m in sweep:
        vals = [runs[k]["t"][m] * 1e3 for k in base_keys]
        tbase[m] = st.fmean(vals)
        print(f"  {m:>3} {SHIPPED_STREAMS[m]:>8} {SHIPPED_IPG[m]:>4} " +
              " ".join(f"{v:10.3f}" for v in vals) + f" {tbase[m]:10.3f}")
    out["T_base_ms"] = tbase
    out["T_base_per_run_ms"] = {k: {m: runs[k]["t"][m] * 1e3 for m in sweep}
                                for k in base_keys}

    d1 = {f"{m}->{m+1}": tbase[m + 1] - tbase[m]
          for m in sweep[:-1] if m + 1 in tbase}
    # Each base build measures the whole curve, so the same step is measured twice
    # and their disagreement is that step's own replication floor.
    floor_d1 = {}
    if len(base_keys) == 2:
        for m in sweep[:-1]:
            if m + 1 not in tbase:
                continue
            reps = [(runs[k]["t"][m + 1] - runs[k]["t"][m]) * 1e3 for k in base_keys]
            floor_d1[f"{m}->{m+1}"] = abs(reps[0] - reps[1])
    pred_d1 = dict(zip(P.D1_LABELS, P.PRED_D1))
    print(f"\n  {'step':>7} {'measured d1':>12} {'H_streams pred':>15} "
          f"{'own floor':>10}   note")
    for lab in P.D1_LABELS:
        if lab not in d1:
            continue
        note = ("STREAM BOUNDARY (H_streams)" if lab in ("4->5", "8->9") else "")
        if lab == "5->6":
            note = "H_M6breakpoint predicts argmax here"
        print(f"  {lab:>7} {d1[lab]:12.3f} {pred_d1[lab]:15.3f} "
              f"{floor_d1.get(lab, float('nan')):10.3f}   {note}")
    argmax_d1 = max(d1, key=d1.get)
    ranked = sorted(d1.items(), key=lambda kv: -kv[1])
    step2_streams = argmax_d1 in ("4->5", "8->9")
    step2_m6 = argmax_d1 == "5->6"
    print(f"\n  argmax d1 = {argmax_d1}  ({d1[argmax_d1]:.3f} ms); "
          f"runner-up {ranked[1][0]} ({ranked[1][1]:.3f} ms)")
    print(f"  H_streams (argmax in 4->5 / 8->9): "
          f"{'SUPPORTED' if step2_streams else 'not supported'}")
    print(f"  H_M6breakpoint (argmax = 5->6):    "
          f"{'SUPPORTED -- stop rule 1 FIRES' if step2_m6 else 'FALSIFIED'}")
    out["d1_ms"] = d1
    out["d1_replicate_floor_ms"] = floor_d1
    out["argmax_d1"] = argmax_d1
    out["step2"] = {"supports_H_streams": bool(step2_streams),
                    "supports_H_M6breakpoint": bool(step2_m6),
                    "ranked": ranked}

    a, b, c, resid = fit(sweep, tbase, SHIPPED_STREAMS)
    a_m, b_m, c_m, resid_m = fit(sweep, tbase,
                                 {m: (1 if m >= 6 else 0) for m in sweep})
    print(f"\n  refit on the SHIPPED stream vector: T(M) = {c:.3f} + {b:.3f}*"
          f"streams(M) + {a:.3f}*M   max|resid| {resid:.3f} ms")
    print(f"  E41 on the NA<=5 table:             T(M) = {P.E41_C:.3f} + "
          f"{P.E41_B:.3f}*streams(M) + {P.E41_A:.3f}*M")
    print(f"  same data refit on an [M>=6] indicator instead: "
          f"max|resid| {resid_m:.3f} ms (b={b_m:.3f})")
    print(f"  the stream regressor fits this table "
          f"{'BETTER' if resid < resid_m else 'WORSE'} than the M>=6 indicator")
    out["refit_shipped_streams"] = {"a_per_row": a, "b_per_stream": b,
                                    "c_intercept": c, "max_abs_resid_ms": resid}
    out["refit_m6_indicator"] = {"a_per_row": a_m, "b_indicator": b_m,
                                 "c_intercept": c_m, "max_abs_resid_ms": resid_m}

    # ---------------------------------------------------------------- step 3
    if not arm_keys:
        print("\n[5] STEP 3 -- no arm runs yet")
        return 0

    print("\n[5] STEP 3 -- fixed-M contrasts; ABBA arithmetic means cancel linear "
          "drift")
    tarm = {m: st.fmean(runs[k]["t"][m] * 1e3 for k in arm_keys) for m in widths}
    mde = {}
    for m in widths:
        spans = []
        if len(base_keys) == 2:
            spans.append(abs(runs["base1"]["t"][m] - runs["base2"]["t"][m]) * 1e3)
        if len(arm_keys) == 2:
            spans.append(abs(runs["arm1"]["t"][m] - runs["arm2"]["t"][m]) * 1e3)
        mde[m] = max(spans) if spans else float("nan")
    out["T_arm_ms"] = tarm
    out["T_arm_per_run_ms"] = {k: {m: runs[k]["t"][m] * 1e3 for m in widths}
                              for k in arm_keys}
    out["mde_ms"] = mde

    print(f"  {'M':>3} {'base(ms)':>10} {'arm(ms)':>10} {'delta':>9} {'MDE':>8} "
          f"{'|d|>MDE':>8}  role")
    tb_all = {m: st.fmean(runs[k]["t"][m] * 1e3 for k in base_keys) for m in widths}
    deltas = {}
    for m in widths:
        dl = tarm[m] - tb_all[m]
        deltas[m] = dl
        sig = abs(dl) > mde[m] if mde[m] == mde[m] else False
        role = ("contrast A (streams 2->2, width 3+3 -> 4+2)" if m == 6 else
                "contrast B (streams 2->3, width 4+4 -> 3+3+2)" if m == 8 else
                "untreated control")
        print(f"  {m:>3} {tb_all[m]:10.3f} {tarm[m]:10.3f} {dl:9.3f} {mde[m]:8.3f} "
              f"{('YES' if sig else 'no'):>8}  {role}")
    out["delta_ms"] = deltas

    # The two builds differ only in the M=6 and M=8 table cells, so every other
    # width compiled to identical code. Whatever the controls move by is the
    # session's own noise, and a contrast smaller than that is not readable.
    ctl = [m for m in widths if m not in (6, 8)]
    ctl_bad = [m for m in ctl if mde[m] == mde[m] and abs(deltas[m]) > mde[m]]
    ctl_worst = max(ctl, key=lambda m: abs(deltas[m]))
    print(f"\n  untreated controls exceeding their own replicate floor: "
          f"{ctl_bad or 'none'}")
    print(f"  worst untreated control move: M={ctl_worst} {deltas[ctl_worst]:+.3f} ms "
          f"({deltas[ctl_worst]/tb_all[ctl_worst]*100:+.2f} %) -- the contrasts must "
          f"beat this to be readable as the edit rather than the session")
    out["controls_exceeding_floor"] = ctl_bad
    out["control_worst"] = {"m": ctl_worst, "delta_ms": deltas[ctl_worst],
                            "pct": deltas[ctl_worst] / tb_all[ctl_worst] * 100}

    print("\n  contrast A   M=6, IPG 3 -> 4   streams 2 -> 2   width 3+3 -> 4+2")
    dA, mdeA = deltas[6], mde[6]
    a_null = abs(dA) <= mdeA if mdeA == mdeA else None
    print(f"    delta_A = {dA:+.3f} ms   MDE(6) = {mdeA:.3f} ms   "
          f"{'|delta| <= MDE -> NULL' if a_null else '|delta| > MDE -> REAL EFFECT'}")
    print(f"    H_streams predicts 0 (streams unchanged):       "
          f"{'CONSISTENT' if a_null else 'INCONSISTENT'}")
    print(f"    H_groupwidth predicts > 0 (widest 3 -> 4 rows): "
          f"{'CONSISTENT' if (not a_null and dA > 0) else 'INCONSISTENT'}")

    print("\n  contrast B   M=8, IPG 4 -> 3   streams 2 -> 3   width 4+4 -> 3+3+2")
    dB, mdeB = deltas[8], mde[8]
    lo_s, hi_s = P.B_BAND_STRICT
    lo_l, hi_l = P.B_BAND_LENIENT
    b_strict = lo_s <= dB <= hi_s
    b_lenient = lo_l <= dB <= hi_l
    b_real = abs(dB) > mdeB if mdeB == mdeB else None
    print(f"    delta_B = {dB:+.3f} ms   MDE(8) = {mdeB:.3f} ms   "
          f"{'REAL EFFECT' if b_real else 'below the replicate floor'}")
    print(f"    H_streams predicts +{P.E41_B:.3f} ms   strict "
          f"[{lo_s:.3f}, {hi_s:.3f}] {'IN' if b_strict else 'OUT'}   lenient "
          f"[{lo_l:.3f}, {hi_l:.3f}] {'IN' if b_lenient else 'OUT'}")
    print(f"    H_groupwidth predicts < 0 (widest 4 -> 3 rows): "
          f"{'CONSISTENT' if dB < 0 else 'INCONSISTENT'}")
    print(f"    H_M6breakpoint predicts 0 (M fixed):            "
          f"{'CONSISTENT' if not b_real else 'INCONSISTENT'}")
    out["contrast"] = {
        "A": {"m": 6, "delta_ms": dA, "mde_ms": mdeA, "null": bool(a_null)},
        "B": {"m": 8, "delta_ms": dB, "mde_ms": mdeB, "real": bool(b_real),
              "in_strict_band": bool(b_strict), "in_lenient_band": bool(b_lenient),
              "band_strict": list(P.B_BAND_STRICT),
              "band_lenient": list(P.B_BAND_LENIENT)},
    }

    print("\n[6] PER-SHAPE SIGN TEST  (8 scored shapes, distribution-free)")
    out["sign_test"] = {}
    for name, m in CONTRAST_M.items():
        pb = {sh: st.fmean(runs[k]["per_shape"][m][sh] for k in base_keys)
              for sh in runs["base1"]["per_shape"][m]}
        pa = {sh: st.fmean(runs[k]["per_shape"][m][sh] for k in arm_keys)
              for sh in runs["arm1"]["per_shape"][m]}
        rows = [(sh, (pa[sh] - pb[sh]) * 1e6, (pa[sh] / pb[sh] - 1) * 100)
                for sh in sorted(pb) if sh in pa]
        pos = sum(1 for _, dd, _ in rows if dd > 0)
        p = sign_test_p(pos, len(rows))
        print(f"  contrast {name} (M={m}):  {pos}/{len(rows)} shapes slower under "
              f"the arm; two-sided sign-test p = {p:.4f}")
        for sh, dd, pct in rows:
            print(f"    {sh:<34} {dd:+9.2f} us/call  {pct:+7.2f} %")
        out["sign_test"][name] = {"m": m, "n_positive": pos, "n": len(rows),
                                  "p_two_sided": p,
                                  "per_shape_us": {sh: dd for sh, dd, _ in rows},
                                  "per_shape_pct": {sh: pc for sh, _, pc in rows}}

    if args.e27_base and args.e27_arm:
        print("\n[7] INDEPENDENT PRIOR REPLICATION  (E27 `7b5183d`, a different "
              "base tree, n=1)")
        pb, pbi = load(args.e27_base)
        pa, pai = load(args.e27_arm)
        tpb, _ = t_of_m(pb)
        tpa, _ = t_of_m(pa)
        print(f"  prior base {args.e27_base} head={pbi.get('head','?')[:12]} "
              f"widths={pbi.get('widths','?')}")
        print(f"  prior arm  {args.e27_arm} head={pai.get('head','?')[:12]} "
              f"widths={pai.get('widths','?')}")
        # The prior arm swept a SHORTER width list than its base, so each width
        # sat at a different sweep position -- and therefore a different GPU
        # temperature -- in the two runs. That is the confound E46 removes by
        # sweeping 1..9 in every arm.
        if pbi.get("widths") != pai.get("widths"):
            print("  NOTE: the prior base and arm swept different width lists, so "
                  "each width sat at a different thermal position in the two runs; "
                  "E46 matches the sweep by construction.")
        print(f"\n  {'M':>3} {'prior d':>9} {'prior %':>8} {'E46 d':>9} "
              f"{'E46 %':>8}  role")
        prior = {}
        for m in sorted(set(tpb) & set(tpa)):
            dp = (tpa[m] - tpb[m]) * 1e3
            pp = (tpa[m] / tpb[m] - 1) * 100
            prior[m] = {"delta_ms": dp, "pct": pp}
            role = ("contrast A" if m == 6 else "contrast B" if m == 8 else
                    "prior-only cell" if m == 4 else "control")
            here = (f"{deltas[m]:9.3f} {deltas[m]/tb_all[m]*100:8.2f}"
                    if m in deltas else f"{'-':>9} {'-':>8}")
            print(f"  {m:>3} {dp:9.3f} {pp:8.2f} {here}  {role}")
        # A is a predicted NULL and B a predicted POSITIVE, so they replicate in
        # different senses: A by both landing inside the noise, B by both being a
        # large positive step. A sign match on a null would mean nothing.
        agree = {}
        if 6 in prior and 6 in deltas:
            agree["A"] = bool(abs(prior[6]["pct"]) < 1.0 and abs(dA) <= max(mdeA, 1.0))
            print(f"\n  contrast A replicates as a NULL: "
                  f"prior {prior[6]['pct']:+.2f} %, E46 {dA/tb_all[6]*100:+.2f} % "
                  f"-> {'YES' if agree['A'] else 'NO'}")
        if 8 in prior and 8 in deltas:
            agree["B"] = bool(prior[8]["delta_ms"] > 0 and dB > 0)
            print(f"  contrast B replicates as a POSITIVE STEP: "
                  f"prior {prior[8]['delta_ms']:+.3f} ms ({prior[8]['pct']:+.2f} %), "
                  f"E46 {dB:+.3f} ms ({dB/tb_all[8]*100:+.2f} %) "
                  f"-> {'YES' if agree['B'] else 'NO'}")
        out["prior_replication"] = {
            "base_tag": args.e27_base, "arm_tag": args.e27_arm,
            "base_head": pbi.get("head"), "arm_head": pai.get("head"),
            "base_widths": pbi.get("widths"), "arm_widths": pai.get("widths"),
            "sweep_matched": pbi.get("widths") == pai.get("widths"),
            "delta": prior, "replicates": agree,
        }

    print("\n[8] VERDICT")
    verdict, mech = [], None
    if step2_m6:
        verdict.append("stop rule 1 FIRES: step 2's argmax d1 is 5->6")
        mech = "H_M6breakpoint"
    elif step2_streams:
        verdict.append(f"step 2: argmax d1 = {argmax_d1}, a stream boundary on the "
                       "shipped table -> H_streams")
    else:
        verdict.append(f"step 2: argmax d1 = {argmax_d1}, neither a stream boundary "
                       "nor 5->6 -> both readings incomplete")
    if a_null and b_real and b_lenient:
        verdict.append("step 3: A null at fixed streams, B a positive step of the "
                       "registered size at a stream boundary -> H_streams")
        mech = mech or "H_streams"
    elif not a_null and dA > 0:
        verdict.append("step 3: A moved with group width at constant streams -> "
                       "H_groupwidth is at least partly right")
        mech = mech or "H_groupwidth"
    elif not b_real:
        verdict.append("step 3: B did not move at a stream boundary -> H_streams "
                       "does not survive at fixed M")
        mech = mech or "neither"
    else:
        verdict.append("step 3: mixed; see the contrast rows above")
        mech = mech or "mixed"
    for line in verdict:
        print("  " + line)
    surviving = mech or "H_streams"
    names = {
        "H_streams": "weight-stream count ceil(M/IPG): the number of threadgroups "
                     "that each re-read the whole weight tile",
        "H_groupwidth": "widest group's row count / group balance",
        "H_M6breakpoint": "a property of the width M itself, not of the table",
        "neither": "unidentified: neither registered reading survives",
        "mixed": "partly identified: see the contrast rows",
    }
    print(f"\n  SURVIVING HYPOTHESIS: {surviving}")
    print(f"  the mechanism should now be called: {names[surviving]}")
    out["verdict"] = {"lines": verdict, "surviving": surviving,
                      "mechanism_name": names[surviving]}

    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(json.dumps(out, indent=2,
                                                          sort_keys=True, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
