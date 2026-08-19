#!/usr/bin/env python3
"""E38 synthesis: three cost-curve arms against the pre-registration.

    python3 research/e38_analyze.py --base e38-base-r1 --arm-a e38-arma-r1 \
        --arm-b e38-armb-r1 [--json-out research/e38-artifacts/e38-metrics.json]

Every registered number comes from research/e38_prereg.py, which was committed
before the kernel existed.  Nothing here re-derives a prediction; it only
compares.
"""
import argparse
import json
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e38_prereg as P  # noqa: E402

CURVE = ".mlxfast-private/qmv-curve/%s"


def load(tag):
    d = json.load(open(os.path.join(CURVE % tag, "vendored.json")))
    ident = {}
    ipath = os.path.join(CURVE % tag, "identity.txt")
    if os.path.exists(ipath):
        for line in open(ipath):
            line = line.strip()
            m = re.match(r"^(?:run-qmv-curve: )?([a-z0-9_]+)=(.*)$", line)
            if m:
                ident[m.group(1)] = m.group(2)
            elif line.startswith("run-qmv-curve: "):
                for part in line[len("run-qmv-curve: "):].split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        ident[k] = v
    return d, ident


def c_round(d):
    """C_round(M) in seconds, and the per-shape contributions at each M."""
    total, per_shape = {}, {}
    for sh in d["shapes"]:
        cpv = sh["calls_per_verify"]
        for r in sh["rows"]:
            m = r["m"]
            total[m] = total.get(m, 0.0) + cpv * r["seconds_per_call"]
            per_shape.setdefault(m, {})[sh["name"]] = r["seconds_per_call"]
    return total, per_shape


def dispatch_at(d, m):
    """The distinct kernel paths / stream counts the arm actually dispatched."""
    paths, streams, ipg = set(), set(), set()
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if r["m"] == m:
                paths.add(r["in_kernel_path"])
                streams.add(r["weight_streams"])
                ipg.add(r["inputs_per_group"])
    return sorted(paths), sorted(streams), sorted(ipg)


def bitwise_ok(d):
    bad = []
    for sh in d["shapes"]:
        for r in sh["rows"]:
            if not r.get("row0_bitwise_matches_m1", True) or r.get("row0_max_abs_delta_vs_m1", 0):
                bad.append((sh["name"], r["m"], r.get("row0_max_abs_delta_vs_m1")))
    return bad


def jit_spread(d, m):
    """max over shapes of (mean - min)/min at width m, in percent.

    A template instantiation that JIT-compiled inside the timed window would
    make one timed region enormously slower than the rest, so the mean would sit
    far above the min at that width and nowhere else.
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
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--treated", type=int, default=6)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    arms = {}
    for key, tag in (("base", args.base), ("a", args.arm_a), ("b", args.arm_b)):
        d, ident = load(tag)
        c, per_shape = c_round(d)
        arms[key] = dict(tag=tag, d=d, ident=ident, c=c, per_shape=per_shape)

    M = args.treated
    widths = sorted(set(arms["base"]["c"]) & set(arms["a"]["c"]) & set(arms["b"]["c"]))
    widths = [m for m in widths if m <= 9]
    control = [m for m in widths if m not in (M, 1)]

    print("=" * 84)
    print(f"E38 -- three arms, treated width M={M}")
    print("=" * 84)

    print("\n[0] ARM IDENTITY (dispatch readback, not an assumption)")
    expect = {
        "base": ("qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>", 2, 3),
        "a": ("qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true, 2>", 2, 3),
        "b": ("qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true, 2, true>", 1, 6),
    }
    ident_ok = True
    for key in ("base", "a", "b"):
        paths, streams, ipg = dispatch_at(arms[key]["d"], M)
        want_path, want_stream, want_ipg = expect[key]
        good = (paths == [want_path] and streams == [want_stream] and ipg == [want_ipg])
        ident_ok &= good
        print(f"  {key:<5} {arms[key]['tag']:<16} {'OK ' if good else 'BAD'} "
              f"streams={streams} ipg={ipg}")
        print(f"        {paths}")
        if not good:
            print(f"        expected {want_path!r} streams=[{want_stream}] ipg=[{want_ipg}]")

    print("\n[1] C_round(M), ms")
    print(f"  {'M':>3} {'base':>9} {'arm a':>9} {'arm b':>9} {'a/base':>8} {'b/base':>8} {'b/a':>8}")
    for m in widths:
        cb_, ca_, cbb = arms["base"]["c"][m], arms["a"]["c"][m], arms["b"]["c"][m]
        print(f"  {m:>3} {cb_*1e3:9.3f} {ca_*1e3:9.3f} {cbb*1e3:9.3f} "
              f"{ca_/cb_:8.4f} {cbb/cb_:8.4f} {cbb/ca_:8.4f}")

    drift_a = st.median(arms["a"]["c"][m] / arms["base"]["c"][m] for m in control)
    drift_b = st.median(arms["b"]["c"][m] / arms["base"]["c"][m] for m in control)
    spread_a = [arms["a"]["c"][m] / arms["base"]["c"][m] for m in control]
    spread_b = [arms["b"]["c"][m] / arms["base"]["c"][m] for m in control]

    ratio_a_raw = arms["a"]["c"][M] / arms["base"]["c"][M]
    ratio_b_raw = arms["b"]["c"][M] / arms["base"]["c"][M]
    ratio_a = ratio_a_raw / drift_a
    ratio_b = ratio_b_raw / drift_b

    print("\n[2] CONTROLS (untreated widths must not move)")
    print(f"  control widths {control}")
    print(f"  arm a: median drift {drift_a:.4f}  spread {min(spread_a):.4f}..{max(spread_a):.4f}")
    print(f"  arm b: median drift {drift_b:.4f}  spread {min(spread_b):.4f}..{max(spread_b):.4f}")
    worst = max(max(abs(v - 1) for v in spread_a), max(abs(v - 1) for v in spread_b))
    print(f"  registered tolerance |ratio-1| <= {P.CONTROL_TOL:.4f}; worst observed {worst:.4f}  "
          f"{'PASS' if worst <= P.CONTROL_TOL else 'FAIL'}")

    print("\n[3] M=1 GLOBAL NULL (no `case 1:` exists in either dispatch tier)")
    if 1 in widths:
        p1 = dispatch_at(arms["b"]["d"], 1)[0]
        r1a = arms["a"]["c"][1] / arms["base"]["c"][1]
        r1b = arms["b"]["c"][1] / arms["base"]["c"][1]
        print(f"  M=1 path: {p1}")
        print(f"  a/base {r1a:.4f}   b/base {r1b:.4f}   "
              f"{'PASS' if max(abs(r1a-1), abs(r1b-1)) <= P.CONTROL_TOL else 'CHECK'}")
        print("  a serial-leg speedup would LOWER the published score, so this is a gate.")

    print("\n[4] PRIMARY vs PRE-REGISTRATION")
    print(f"  e38/m6_per_row_cost_ratio (raw)            = {ratio_b_raw:.4f}")
    print(f"  e38/m6_per_row_cost_ratio (drift-adjusted) = {ratio_b:.4f}   <-- PRIMARY")
    print(f"  registered {P.REGISTERED_RATIO:.3f} band [{P.REGISTERED_BAND[0]:.3f}, "
          f"{P.REGISTERED_BAND[1]:.3f}]  "
          f"{'INSIDE' if P.REGISTERED_BAND[0] <= ratio_b <= P.REGISTERED_BAND[1] else 'OUTSIDE'}")
    print(f"  assignment expectation {P.ADVISOR_RATIO:.2f}  "
          f"{'CONSISTENT' if ratio_b <= 0.96 else 'FALSIFIED'}")
    print(f"  route 1 predicted {P.route1()['ratio_b']:.4f}; route 2 predicted {P.route2()['ratio']:.4f}")

    print("\n[5] REGISTERED RELATIONS, MEASURED")
    rels = {r["key"]: r for r in P.registered_relations()}
    measured = {
        "R1_weight_pass": ratio_a - ratio_b,
        "R2_activation_doubling": ratio_a - 1.0,
        "R3_serialization": P.E33_RATIO_M6 - ratio_b,
    }
    for key, val in measured.items():
        rel = rels[key]
        inside = rel["lo"] <= val <= rel["hi"]
        print(f"  {key:<24} measured {val:+.4f}   registered {rel['point']:+.4f} "
              f"in [{rel['lo']:+.4f}, {rel['hi']:+.4f}]  {'INSIDE' if inside else 'OUTSIDE'}")
    print(f"      R3 discriminates: my account needed +0.0250, the 0.84 account needed +0.1750.")

    print("\n[6] PER-SHAPE AT M=6")
    r2 = {r["name"]: r for r in P.route2()["rows"]}
    alias = {"linear_attn.in_proj_fused_qkvzba": "linear_attn.in_proj_fused"}
    print(f"  {'shape':<34}{'base us':>10}{'a/base':>9}{'b/base':>9}{'E38 pred':>10}{'E33':>9}")
    for sh in arms["base"]["d"]["shapes"]:
        name = sh["name"]
        key = alias.get(name, name)
        bs = arms["base"]["per_shape"][M].get(name)
        aa = arms["a"]["per_shape"][M].get(name)
        bb = arms["b"]["per_shape"][M].get(name)
        if not bs:
            continue
        pred = r2.get(key, {}).get("e38_point")
        e33 = r2.get(key, {}).get("r33")
        print(f"  {name:<34}{bs*1e6:10.2f}{aa/bs:9.4f}{bb/bs:9.4f}"
              f"{(pred if pred else float('nan')):10.4f}{(e33 if e33 else float('nan')):9.4f}")

    print("\n[7] JIT / PSO PRE-FLIGHT (advisor comment 5337266846)")
    print("  worst (mean - min)/min per width, percent.  A template instantiation")
    print("  that compiled inside the timed window would spike only at M=6.")
    print(f"  {'M':>3} {'base':>9} {'arm a':>9} {'arm b':>9}")
    for m in widths:
        print(f"  {m:>3} {jit_spread(arms['base']['d'], m):9.2f} "
              f"{jit_spread(arms['a']['d'], m):9.2f} {jit_spread(arms['b']['d'], m):9.2f}")

    print("\n[8] BITWISE (harness-native row0 check against the M=1 reference)")
    for key in ("base", "a", "b"):
        bad = bitwise_ok(arms[key]["d"])
        print(f"  {key:<5} {'all rows bitwise-identical to M=1' if not bad else bad}")

    print("\n[9] EFFECTIVE GEOMETRY (advisor comment 5337327566)")
    dev = arms["base"]["d"]["device"]
    arch = dev["architecture"]
    cls = dev.get("architecture_class")
    mlx_default = {"p": (20, 40), "g": (40, 40), "s": (50, 50), "d": (50, 50)}.get(cls, (40, 40))
    print(f"  device {arch} class '{cls}' mem={dev['memory_size_bytes']/2**30:.0f} GiB")
    print(f"  QwenQMVCostCurveTests never constructs QwenRuntimeMTPWorker, so")
    print(f"  RuntimeStartupMemoryPolicy never runs and MLX's own arch defaults apply:")
    print(f"    MLX_MAX_OPS_PER_BUFFER={mlx_default[0]}  MLX_MAX_MB_PER_BUFFER={mlx_default[1]}")
    print(f"    env overrides present: "
          f"{[k for k in os.environ if k.startswith('MLX_MAX') or k.startswith('DARKBLOOM_')] or 'none'}")
    print(f"  ranked box would be 512 MB / 50 ops with residency ON (>=96 GiB gated,")
    print(f"  no env override, so not testable on this host).")
    for key in ("base", "a", "b"):
        i = arms[key]["ident"]
        print(f"  {key:<5} cool_gate={i.get('cool_gate_vendored', '?'):<18} "
              f"entry={i.get('gpu_temp_c_before_vendored', '?')[:6]:>6}C "
              f"exit={i.get('gpu_temp_c_after_vendored', '?')[:6]:>6}C "
              f"dirty={i.get('dirty', '?')} head={i.get('head', '?')[:12]}")

    print("\n[10] VERDICT")
    if ratio_b <= P.DECISIVE_RATIO:
        verdict = "DECISIVE WIN"
    elif ratio_b <= P.SHIP_RATIO:
        verdict = "SHIP"
    elif ratio_b >= 1 - P.CONTROL_TOL:
        verdict = "NULL"
    else:
        verdict = "SUB-THRESHOLD (real but below 1 sigma)"
    print(f"  {verdict}   (ship <= {P.SHIP_RATIO:.4f}, decisive <= {P.DECISIVE_RATIO:.4f}, "
          f"null >= {1-P.CONTROL_TOL:.4f})")
    print(f"  predicted decode-leg movement = {P.leg_movement_pct(ratio_b):+.3f}%  "
          f"projected score {P.score_gain_pct(ratio_b):+.3f}%")
    run_e2e = ratio_b <= P.E2E_RATIO
    print(f"  E2E rule (ratio <= {P.E2E_RATIO:.4f}): "
          f"{'RUN IT' if run_e2e else 'SKIP -- below instrument resolution'}")

    out = dict(
        primary=dict(name="e38/m6_per_row_cost_ratio", value=ratio_b, raw=ratio_b_raw,
                     registered=P.REGISTERED_RATIO, band=list(P.REGISTERED_BAND),
                     advisor=P.ADVISOR_RATIO),
        ratio_a=ratio_a, ratio_a_raw=ratio_a_raw,
        drift_a=drift_a, drift_b=drift_b,
        control_worst_abs_dev=worst, control_tol=P.CONTROL_TOL,
        relations={k: dict(measured=v, registered=rels[k]["point"],
                           lo=rels[k]["lo"], hi=rels[k]["hi"]) for k, v in measured.items()},
        c_round_ms={key: {str(m): arms[key]["c"][m] * 1e3 for m in widths} for key in arms},
        dispatch_identity_ok=bool(ident_ok),
        verdict=verdict, run_e2e=bool(run_e2e),
        leg_movement_pct=P.leg_movement_pct(ratio_b),
        score_gain_pct=P.score_gain_pct(ratio_b),
        geometry=dict(architecture=arch, architecture_class=cls,
                      mlx_default_ops_per_buffer=mlx_default[0],
                      mlx_default_mb_per_buffer=mlx_default[1],
                      policy_applied=False,
                      env_overrides=[k for k in os.environ
                                     if k.startswith("MLX_MAX") or k.startswith("DARKBLOOM_")]),
        arms={key: dict(tag=arms[key]["tag"], identity=arms[key]["ident"]) for key in arms},
    )
    if args.json_out:
        parent = os.path.dirname(args.json_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1, sort_keys=True)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
