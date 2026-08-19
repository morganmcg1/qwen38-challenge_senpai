#!/usr/bin/env python3
"""E41 synthesis: the K-tile distance ladder against the pre-registration.

    python3 research/e41_analyze.py --base e41-base-r1 --arm e41-arm-r1 \
        [--base2 e41-base-r2] [--json-out research/e41-artifacts/e41-metrics.json]

One arm build carries every rung, at a different M each, so the whole ladder is
measured in ONE session and the rungs cannot drift relative to each other. The
base build differs from it by nothing but the dispatch table. Every registered
number comes from research/e41_prereg.py, committed before the kernel existed;
nothing here re-derives a prediction.

--base2 is an optional second base run after the arm, so between-session drift is
bracketed rather than assumed.
"""
import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e41_prereg as P  # noqa: E402

CURVE = ".mlxfast-private/qmv-curve/%s"

# What each width's dispatch readback must say, from the arm dispatch table. The
# readback is parsed out of the header the build actually compiled, so this is a
# check on the built artifact, not on my memory of it.
EXPECT_ARM = {
    3: "qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true, 2, 2, 1>",
    4: "qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true, 2, 2, 64>",
    6: "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true, 2, 1>",
    7: "qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true, 2, 2, 1>",
    8: "qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true, 2, 2, 4>",
}
EXPECT_BASE = {
    3: "qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true>",
    4: "qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>",
    5: "qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>",
    6: "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>",
    7: "qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>",
    8: "qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>",
    9: "qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>",
}
# The NA=4 ladder, in re-read distance order. M=4 is the top rung (K in one
# tile), M=8 the discriminating rung, M=7 the adjacency bound.
LADDER = [(4, "KT=64 no locality"), (8, "KT=4  locality"), (7, "KT=1  adjacency")]
DISCRIMINATOR_M = 8
ANCHOR_M = 6


def load(tag):
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


def c_round(d):
    """C_round(M) in seconds, and each shape's contribution at each M."""
    total, per_shape = {}, {}
    for sh in d["shapes"]:
        cpv = sh["calls_per_verify"]
        for r in sh["rows"]:
            m = r["m"]
            total[m] = total.get(m, 0.0) + cpv * r["seconds_per_call"]
            per_shape.setdefault(m, {})[sh["name"]] = r["seconds_per_call"]
    return total, per_shape


def dispatch_at(d, m):
    paths, streams, ipg = set(), set(), set()
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == m:
                paths.add(r["in_kernel_path"])
                streams.add(r["weight_streams"])
                ipg.add(r["inputs_per_group"])
    return sorted(paths), sorted(streams), sorted(ipg)


def bitwise_bad(d):
    bad = []
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if (not r.get("row0_bitwise_matches_m1", True)
                    or r.get("row0_max_abs_delta_vs_m1", 0)):
                bad.append((sh["name"], r["m"], r.get("row0_max_abs_delta_vs_m1")))
    return bad


def jit_spread(d, m):
    """max over shapes of (mean - min)/min at width m, in percent.

    A template instantiation that JIT-compiled inside the timed window makes one
    timed region far slower than the rest, so the mean sits well above the min at
    that width and nowhere else. Every K-tile rung is a fresh instantiation, so
    this is the check that matters most in this experiment.
    """
    worst = 0.0
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == m and r["seconds_per_call_min"] > 0:
                worst = max(worst, (r["seconds_per_call"] - r["seconds_per_call_min"])
                            / r["seconds_per_call_min"] * 100.0)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base2", default="")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    todo = [("base", args.base), ("arm", args.arm)]
    if args.base2:
        todo.append(("base2", args.base2))
    runs = {}
    for key, tag in todo:
        d, ident = load(tag)
        c, per_shape = c_round(d)
        runs[key] = dict(tag=tag, d=d, ident=ident, c=c, per_shape=per_shape)

    widths = sorted(set(runs["base"]["c"]) & set(runs["arm"]["c"]))
    treated = [m for m in widths if m in P.ARM_MAP]
    control = [m for m in widths if m in P.UNTREATED]

    out = {"base_tag": args.base, "arm_tag": args.arm, "base2_tag": args.base2,
           "widths": widths, "treated": treated, "control": control}

    print("=" * 88)
    print("E41 -- K-tile re-read distance ladder, one arm build, one session")
    print("=" * 88)

    print("\n[0] PROVENANCE")
    for key in runs:
        i = runs[key]["ident"]
        print(f"  {key:<6} {runs[key]['tag']:<14} head={i.get('head','?')[:12]} "
              f"dirty={i.get('dirty','?')} reps={i.get('reps','?')} "
              f"inner={i.get('inner','?')} host={i.get('host','?')}")
        print(f"         cool_gate={i.get('cool_gate_vendored','?')} "
              f"entry_C={i.get('gpu_temp_c_before_vendored','?')} "
              f"exit_C={i.get('gpu_temp_c_after_vendored','?')}")
    out["identity"] = {k: runs[k]["ident"] for k in runs}

    print("\n[1] DISPATCH READBACK (parsed from the header each build compiled)")
    ident_ok = True
    for m in sorted(set(EXPECT_BASE) | set(EXPECT_ARM)):
        row = {}
        for key, expect in (("base", EXPECT_BASE), ("arm", EXPECT_ARM)):
            if m not in expect or key not in runs:
                continue
            paths, streams, ipg = dispatch_at(runs[key]["d"], m)
            good = paths == [expect[m]]
            ident_ok &= good
            row[key] = (good, paths, streams, ipg)
        marks = " ".join(f"{k}:{'OK' if v[0] else 'BAD'}" for k, v in row.items())
        print(f"  M={m}  {marks}")
        for k, v in row.items():
            if not v[0]:
                print(f"        {k} got {v[1]}")
                print(f"        {k} want [{ (EXPECT_BASE if k=='base' else EXPECT_ARM)[m] !r}]")
    out["dispatch_readback_ok"] = ident_ok
    print(f"  all instantiations as designed: {'YES' if ident_ok else 'NO -- STOP'}")

    print("\n[2] FIDELITY")
    bad_base, bad_arm = bitwise_bad(runs["base"]["d"]), bitwise_bad(runs["arm"]["d"])
    print(f"  base row0-vs-M1 bitwise failures: {len(bad_base)}")
    print(f"  arm  row0-vs-M1 bitwise failures: {len(bad_arm)}")
    if bad_arm:
        for row in bad_arm[:8]:
            print(f"    {row}")
    out["bitwise_failures"] = {"base": bad_base, "arm": bad_arm}

    print("\n[3] JIT LEAK CHECK (mean-vs-min spread, %; each rung is a new instantiation)")
    out["jit_spread_pct"] = {}
    for m in widths:
        jb, ja = jit_spread(runs["base"]["d"], m), jit_spread(runs["arm"]["d"], m)
        out["jit_spread_pct"][m] = {"base": jb, "arm": ja}
        flag = "  <-- treated" if m in treated else ""
        print(f"  M={m}  base {jb:7.2f}   arm {ja:7.2f}{flag}")

    print("\n[4] C_round(M), ms, and the raw arm/base ratio")
    print(f"  {'M':>3} {'base':>9} {'arm':>9} {'arm/base':>9}  role")
    out["c_round_ms"] = {}
    for m in widths:
        cb, ca = runs["base"]["c"][m], runs["arm"]["c"][m]
        role = P.ARM_MAP[m][1] if m in P.ARM_MAP else "untreated control"
        out["c_round_ms"][m] = {"base": cb * 1e3, "arm": ca * 1e3, "ratio_raw": ca / cb}
        print(f"  {m:>3} {cb*1e3:9.3f} {ca*1e3:9.3f} {ca/cb:9.4f}  {role}")

    print("\n[5] CONTROLS: untreated widths must not move")
    spread = {m: runs["arm"]["c"][m] / runs["base"]["c"][m] for m in control}
    drift = st.median(spread.values())
    worst = max(abs(v - 1) for v in spread.values())
    ctl_ok = worst <= P.CONTROL_BAND
    print(f"  control widths {control}: " +
          "  ".join(f"M{m}={v:.4f}" for m, v in sorted(spread.items())))
    print(f"  median drift {drift:.4f}; worst |ratio-1| {worst:.4f}; "
          f"registered band {P.CONTROL_BAND:.4f}  {'PASS' if ctl_ok else 'FAIL'}")
    if "base2" in runs:
        br = {m: runs["base2"]["c"][m] / runs["base"]["c"][m] for m in widths}
        print("  base2/base bracket (pure session drift, every width): "
              f"median {st.median(br.values()):.4f} "
              f"range {min(br.values()):.4f}..{max(br.values()):.4f}")
        out["base2_over_base"] = br
    out["control"] = {"spread": spread, "median_drift": drift, "worst_abs_dev": worst,
                      "band": P.CONTROL_BAND, "pass": ctl_ok}

    print("\n[6] ANCHOR: does the session replicate E38's arm(a)?")
    rho_anchor = (runs["arm"]["c"][ANCHOR_M] / runs["base"]["c"][ANCHOR_M]) / drift
    lo, hi = P.PRED_M6_ANCHOR
    anchor_ok = lo <= rho_anchor <= hi
    print(f"  M={ANCHOR_M} rho (drift-adjusted) = {rho_anchor:.4f}")
    print(f"  E38 measured {P.E38_ARM_A_M6:.4f}; registered band "
          f"[{lo:.4f}, {hi:.4f}]  {'REPLICATES' if anchor_ok else 'DOES NOT REPLICATE'}")
    if not anchor_ok:
        print("  a miss here invalidates the SESSION, not the ladder: the tax the "
              "ladder is trying to recover is not the tax E38 measured.")
    out["anchor"] = {"m": ANCHOR_M, "rho": rho_anchor, "band": [lo, hi], "pass": anchor_ok}

    print("\n[7] THE LADDER (NA=4, drift-adjusted rho; only the k_tile constant differs)")
    rho = {}
    for m, label in LADDER:
        rho[m] = (runs["arm"]["c"][m] / runs["base"]["c"][m]) / drift
        print(f"  M={m}  {label:<20} rho = {rho[m]:.4f}   tax = {rho[m]-1:+.4f}")
    out["ladder_rho"] = rho

    top = rho[LADDER[0][0]]
    disc = rho[DISCRIMINATOR_M]
    bound = rho[LADDER[2][0]]
    loc_frac = P.locality_recovery(top, disc)
    tot_frac = P.total_recovery(top, bound)
    print(f"\n  locality recovery  (KT=64 -> KT=4) = {loc_frac:+.3f} of the tax "
          f"({top-disc:+.4f} absolute)")
    print(f"  total recovery     (KT=64 -> KT=1) = {tot_frac:+.3f} of the tax "
          f"({top-bound:+.4f} absolute)")
    print(f"  registered thresholds: MEM >= {P.LOCALITY_STEP_MEM:.0%}, "
          f"ILP <= {P.LOCALITY_STEP_ILP:.0%}, step must clear "
          f"{P.CONTROL_BAND:.4f} to count")

    verdict = P.verdict(top, disc, bound)
    out["locality_recovery"] = loc_frac
    out["total_recovery"] = tot_frac
    out["verdict"] = verdict

    print(f"\n[8] PRE-REGISTERED VERDICT\n  {verdict}")
    if verdict.startswith("MEM"):
        print("  -> deliverable (b) is licensed. The census priced it: NA=6 at r=1 with")
        print("     4 tiles live is 105 registers, so (b) needs no threadgroup memory")
        print("     and no barrier -- it is loop interchange, not E33.")
    elif verdict.startswith("ILP"):
        print("  -> K-tiling is dead. R2 is the halved register tile and the extra")
        print("     loop, not the activation re-read. Per the assignment I stop here")
        print("     and do not build (b).")
    else:
        print("  -> partial. Report the fraction; (b)'s ceiling is that fraction of R2.")

    print("\n[9] CROSS-NA CHECK (NA=3: M=6 sequential tax vs M=3 K-tiled adjacency)")
    rho3 = (runs["arm"]["c"][3] / runs["base"]["c"][3]) / drift
    print(f"  M=6 rho {rho_anchor:.4f} (r=2, BPC=1, full-K passes)")
    print(f"  M=3 rho {rho3:.4f} (r=2, BPC=2, KT=1)")
    na3_rec = P.total_recovery(rho_anchor, rho3)
    print(f"  NA=3 recovery vs the anchor tax = {na3_rec:+.3f}")
    print("  This pair is confounded (both tiles live AND adjacency change), so it")
    print("  only corroborates direction. If NA=3 recovers while the NA=4 ladder is")
    print("  flat, the recovery came from the extra accumulator liveness -- ILP.")
    out["na3"] = {"rho_m6": rho_anchor, "rho_m3": rho3, "recovery": na3_rec}

    print("\n[10] PER-SHAPE AT THE DISCRIMINATING STEP")
    print(f"  {'shape':<34}{'base us':>10}{'M4 rho':>9}{'M8 rho':>9}{'step':>9}")
    for name in runs["base"]["per_shape"][DISCRIMINATOR_M]:
        b4 = runs["base"]["per_shape"][4].get(name)
        a4 = runs["arm"]["per_shape"][4].get(name)
        b8 = runs["base"]["per_shape"][DISCRIMINATOR_M].get(name)
        a8 = runs["arm"]["per_shape"][DISCRIMINATOR_M].get(name)
        if not all((b4, a4, b8, a8)):
            continue
        r4, r8 = (a4 / b4) / drift, (a8 / b8) / drift
        print(f"  {name:<34}{b8*1e6:10.2f}{r4:9.4f}{r8:9.4f}{r4-r8:+9.4f}")

    print("\n[11] VALUE, CONDITIONAL AND KERNEL-LEVEL ONLY")
    print(f"  psi*phi = {P.PSI_PHI_BACKSOLVED} is BACK-SOLVED from the crown, not")
    print("  measured. Every score figure below inherits that; I do not claim it.")
    print(f"  the measured M=6 tax alone is worth {P.score_pct(rho_anchor):+.4f} % of "
          "score if paid")
    recovered = rho_anchor - (rho_anchor - 1.0) * max(loc_frac, 0.0)
    print(f"  the same tax with this ladder's recovery applied: "
          f"{P.score_pct(recovered):+.4f} %")
    print("  neither number is a gain over the shipped base, which pays no tax. The")
    print("  tax only matters as the price deliverable (b) must pay to buy one weight")
    print("  pass at NA=6, so (b)'s value is that weight-pass saving MINUS whatever")
    print("  tax survives K-tiling. E41 measures the second term only.")
    print(f"  crown {P.CROWN_PCT:.4f} %, engineerable gap {P.GAP_PCT:.4f} %, "
          f"sigma_score {P.SIGMA_SCORE_PCT:.4f} %")
    print("  no E2E leg was run: the predicted move is far inside the n=4 MDE.")
    out["value"] = {
        "psi_phi_backsolved": P.PSI_PHI_BACKSOLVED,
        "psi_phi_is_measured": False,
        "tax_score_pct_if_paid": P.score_pct(rho_anchor),
        "tax_score_pct_after_recovery": P.score_pct(recovered),
    }

    gates = {"dispatch_readback": ident_ok, "fidelity": not bad_arm,
             "controls": ctl_ok, "anchor": anchor_ok}
    out["gates"] = gates
    print("\n[12] GATES")
    for k, v in gates.items():
        print(f"  {k:<20} {'PASS' if v else 'FAIL'}")
    out["all_gates_pass"] = all(gates.values())

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
