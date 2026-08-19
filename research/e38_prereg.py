#!/usr/bin/env python3
"""E38 pre-registration: every prediction, threshold and decision rule, as code.

Committed BEFORE the kernel is written and BEFORE any E38 measurement exists, so
the numbers below cannot be tuned to the answer.  Run it to reproduce the
registered values:

    python3 research/e38_prereg.py                # full registration report
    python3 research/e38_prereg.py --self-test    # invariants only

E38's question
--------------
E33 made M=6 a single weight pass via `_m<T,6,6,true,2>` with a SEQUENTIAL row
loop, and measured 1.0150 (1.5% slower).  That change bundled two effects:

  (i)  the activation tile doubled (each threadgroup's row block re-reads the
       inputs it needs), and
  (ii) the grid halved, because IPG=6 makes ceil(M/IPG)=1 working x-block where
       the shipped IPG=3 has 2.

E38 unbundles them.  It keeps (i) exactly and undoes (ii) by placing the two row
blocks in two DIFFERENT threadgroups -- the ones the host already launches and
that currently exit immediately at `if (first_m >= M) return;`.

  arm (a) control  = _m<T,6,3,true,2> sequential : 2 x-groups, 2 weight passes
  arm (b) repair   = _m<T,6,6,true,2> x-blocked  : 2 x-blocks,  1 weight pass
  base             = _m<T,6,3,true,4>            : 2 x-groups, 2 weight passes

so that

  (a) - base  isolates the doubled activation reads   (grid, weights, MACs equal)
  (a) - (b)   isolates exactly one weight pass        (grid, acts,   MACs equal)

Sources of the arithmetic
-------------------------
backend/metal/quantized.cpp:251-254   grid_dims(M, ceil(N/8), B), 32x2 threads
kernels/quantized.h:1171-1172         first_m = tid.x*IPG; if (first_m>=M) return
kernels/quantized.h:968-1066          _wide: rows_per_simd=4, values_per_thread=16
E33 result document, sections 8.1/8.2 (measured cost curve + per-shape table)
"""
import argparse
import math

# ---------------------------------------------------------------------------
# Measured inputs.  All of these predate E38 and are quoted, not fitted here.
# ---------------------------------------------------------------------------

# E33 section 8.1: per-round QMV cost C_round(M) in ms, base tree, this host.
# The base kernel files are byte-identical between E33's base 4e5dc2b and E38's
# base 54248ce (verified with git diff), so these are a valid comparison basis.
BASE_C_ROUND_MS = {
    1: 58.676, 2: 63.212, 3: 72.507, 4: 82.774, 5: 96.163,
    6: 128.843, 7: 138.694, 8: 149.490, 9: 164.443,
}
E33_C_ROUND_MS = {
    1: 58.996, 2: 63.379, 3: 72.427, 4: 82.493, 5: 96.058,
    6: 130.781, 7: 138.988, 8: 149.536, 9: 165.198,
}

# Shipped >=4096-output tier, read off the switch in quantized.h: M -> IPG.
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}

# E33 section 8.2, measured at M=6: (name, n, base ms/call, calls, E33 ratio).
SHAPES_M6 = [
    ("mlp.gate_up_fused",        34816, 0.8403, 64, 0.9941),
    ("mlp.down",                  5120, 0.4750, 64, 1.0592),
    ("linear_attn.in_proj_fused", 16480, 0.4293, 48, 0.9947),
    ("linear_attn.out_proj",      5120, 0.1883, 48, 1.0492),
    ("full_attn.qkv_proj_fused",  14336, 0.3763, 16, 1.0148),
    ("full_attn.o_proj",          5120, 0.1937, 16, 1.0414),
    ("head.lm_head",            248320, 5.7060,  1, 0.9830),
    ("head.compact_draft_vocab", 98336, 0.0000,  0, 0.9868),
]

# Campaign constants (advisor comment 5337266846, corrected values).
PHI_LOCAL = 0.201          # M=6 share of QMV cost, this fixture (cost share)
PHI_RANKED_ROWS = 0.217    # askeladd's M=6 dispatched-ROW share, ranked beagle
PSI_MEASURED = 0.228       # QMV share of decode wall, E33 measurement, THIS host
SCORE_CHAIN = 0.4827       # raw_p[beagle] / (2 * score), exact
SIGMA_SCORE_PCT = 0.0923   # run-to-run sigma of the published score, percent
GAP_TO_FIRST_PCT = 0.561   # our 3.23250848 vs bar 3.24929398547457, percent
E2E_RESOLUTION_PCT = 0.30  # E33's --local-iterate instrument, n=2, 64 tokens

# E33's register law, measured by AIR probe.  Do NOT interpolate across r.
REG_LAW = {1: (15, 12), 2: (15, 17), 4: (20, 21)}  # rows_per_simd -> (a, b) in a+b*NA


# ---------------------------------------------------------------------------
# Issue-unit model.  U counts per-threadgroup instruction issue for one k-block,
# summed over the threadgroups that do work.
#
#   32*rows*NA   fused-multiply-add issue   (rows outputs x NA inputs)
#   16*rows      weight load + affine dequant issue (values_per_thread=16)
#   48*NA        activation load / broadcast issue
#
# The three coefficients are structural (they come from values_per_thread=16 and
# the vector widths in _wide), not fitted.
# ---------------------------------------------------------------------------

def issue_units(rows, na):
    return 32 * rows * na + 16 * rows + 48 * na


def arm_units(m, ipg, rows_per_simd, blocks_in_x):
    """(total U, weight passes, activation lane-units, MAC units) for one arm."""
    row_blocks = 4 // rows_per_simd
    x_groups = math.ceil(m / ipg)
    total_u = mac = wgt = act = 0
    for g in range(x_groups):
        na = min(ipg, m - g * ipg)
        for _ in range(row_blocks):
            total_u += issue_units(rows_per_simd, na)
            mac += 32 * rows_per_simd * na
            wgt += 16 * rows_per_simd
            act += 48 * na
    # blocks_in_x changes WHERE the row blocks run, never how many there are.
    del blocks_in_x
    return total_u, x_groups, act, mac, wgt


ARMS = {
    "base":  dict(m=6, ipg=3, rows_per_simd=4, blocks_in_x=False),
    "a_ctl": dict(m=6, ipg=3, rows_per_simd=2, blocks_in_x=False),
    "b_rep": dict(m=6, ipg=6, rows_per_simd=2, blocks_in_x=True),
    "e33":   dict(m=6, ipg=6, rows_per_simd=2, blocks_in_x=False),
}


def fit_cost_model():
    """Least-squares fit of T = a + b*U + c*(passes-1) over the base ladder.

    M=1 is excluded: no `case 1:` exists in either tier, so M=1 never reaches the
    crossrow kernel at all and belongs to a different cost family.
    """
    rows = []
    for m in sorted(SHIPPED_IPG):
        u, passes, _, _, _ = arm_units(m, SHIPPED_IPG[m], 4, False)
        rows.append((1.0, float(u), float(passes - 1), BASE_C_ROUND_MS[m]))
    n = len(rows)
    # Normal equations for 3 unknowns, solved with plain Gaussian elimination so
    # the script has no numpy dependency.
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * r[3] for r in rows) for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda k: abs(ata[k][i]))
        ata[i], ata[p] = ata[p], ata[i]
        atb[i], atb[p] = atb[p], atb[i]
        for k in range(i + 1, 3):
            f = ata[k][i] / ata[i][i]
            for j in range(i, 3):
                ata[k][j] -= f * ata[i][j]
            atb[k] -= f * atb[i]
    x = [0.0, 0.0, 0.0]
    for i in reversed(range(3)):
        x[i] = (atb[i] - sum(ata[i][j] * x[j] for j in range(i + 1, 3))) / ata[i][i]
    resid = [(rows[k][3] - (x[0] + x[1] * rows[k][1] + x[2] * rows[k][2]))
             for k in range(n)]
    return x[0], x[1], x[2], resid


# ---------------------------------------------------------------------------
# Route 1: differential issue-cost model.
# ---------------------------------------------------------------------------

def route1():
    a, b, c, resid = fit_cost_model()
    u_base, p_base, act_base, mac_base, wgt_base = arm_units(**ARMS["base"])
    u_a, p_a, act_a, mac_a, wgt_a = arm_units(**ARMS["a_ctl"])
    u_b, p_b, act_b, mac_b, wgt_b = arm_units(**ARMS["b_rep"])
    u_33, p_33, *_ = arm_units(**ARMS["e33"])
    t_base = BASE_C_ROUND_MS[6]

    # Differential form: the intercept and base's own fit residual cancel.
    d_b = b * (u_b - u_base) + c * (p_b - p_base)
    d_a = b * (u_a - u_base) + c * (p_a - p_base)
    # E33 has the same U and passes as arm (b); its measured excess over that
    # prediction is the grid-thinning penalty S that E38 exists to remove.
    d_33_pred = b * (u_33 - u_base) + c * (p_33 - p_base)
    s_e33 = (E33_C_ROUND_MS[6] - t_base) - d_33_pred

    return dict(
        a=a, b=b, c=c, resid=resid, s_e33=s_e33,
        u=dict(base=u_base, a=u_a, b=u_b, e33=u_33),
        passes=dict(base=p_base, a=p_a, b=p_b, e33=p_33),
        act=dict(base=act_base, a=act_a, b=act_b),
        mac=dict(base=mac_base, a=mac_a, b=mac_b),
        wgt=dict(base=wgt_base, a=wgt_a, b=wgt_b),
        ratio_b=(t_base + d_b) / t_base,
        ratio_a=(t_base + d_a) / t_base,
    )


# ---------------------------------------------------------------------------
# Route 2: per-shape residual analysis of E33's measured table.
#
# E33's working grid was ceil(N/8) threadgroups (1 x-block).  Base and E38 both
# run 2 x-blocks per row-group.  Shapes whose E33 grid was already large enough
# to saturate the machine paid no grid penalty, so for THOSE shapes E33's
# measured ratio is the pure traffic effect -- which is exactly what E38 keeps.
# Shapes below that knee get the median above-knee ratio as a point estimate and
# the full above-knee range as their band.
# ---------------------------------------------------------------------------

KNEE_TG = 1900  # between qkv's 1792 (penalised) and in_proj's 2060 (clean)


def route2():
    rows = []
    for name, n, ms, calls, r33 in SHAPES_M6:
        y_blocks = math.ceil(n / 8)
        rows.append(dict(name=name, n=n, y_blocks=y_blocks, ms=ms, calls=calls,
                         r33=r33, e33_tgs=y_blocks, base_tgs=2 * y_blocks,
                         e38_tgs=2 * y_blocks, above_knee=y_blocks >= KNEE_TG))
    clean = sorted(r["r33"] for r in rows if r["above_knee"])
    lo, hi = clean[0], clean[-1]
    mid = clean[len(clean) // 2] if len(clean) % 2 else 0.5 * (clean[len(clean) // 2 - 1] + clean[len(clean) // 2])
    for r in rows:
        r["e38_point"] = r["r33"] if r["above_knee"] else mid
        r["e38_lo"], r["e38_hi"] = (r["r33"], r["r33"]) if r["above_knee"] else (lo, hi)
    tot = sum(r["ms"] * r["calls"] for r in rows)

    def blend(key):
        return sum(r["ms"] * r["calls"] * r[key] for r in rows) / tot

    return dict(rows=rows, clean=clean, mid=mid, lo=lo, hi=hi, total_ms=tot,
                ratio=blend("e38_point"), ratio_lo=blend("e38_lo"),
                ratio_hi=blend("e38_hi"))


# ---------------------------------------------------------------------------
# Registered numbers, thresholds and decision rules.
# ---------------------------------------------------------------------------

REGISTERED_RATIO = 0.99
REGISTERED_BAND = (0.96, 1.005)
ADVISOR_RATIO = 0.84
CONTROL_TOL = 0.0046   # max |ratio-1| over E33's untreated M in {2..5,7,8,9}
E33_RATIO_M6 = E33_C_ROUND_MS[6] / BASE_C_ROUND_MS[6]

# Cost of the second full weight stream at this shape mix, two independent reads
# of the SAME measured ladder:
#   17.24 ms  least-squares c term over M=2..9
#   20.59 ms  E33's own decomposition of the M=5 -> M=6 step (32.68 ms total,
#             minus one 12.09 ms lane increment)
WEIGHT_PASS_MS = (17.24, 20.59)


def registered_relations():
    """Within-experiment identities.  These need no cross-session model at all:
    all three arms are measured in ONE locked session against ONE fresh base."""
    t = BASE_C_ROUND_MS[6]
    r1 = route1()
    # Removing one weight stream removes its DRAM bytes (the fitted c term) AND
    # its 64 weight-load/dequant issue units.  Both must be counted.
    wgt_issue_ms = r1["b"] * (r1["wgt"]["a"] - r1["wgt"]["b"])
    return [
        dict(
            key="R1_weight_pass",
            expr="ratio(a) - ratio(b)",
            claim="removing exactly one full weight stream, everything else equal",
            lo=0.13, hi=0.20,
            point=(wgt_issue_ms + r1["c"]) / t,
            note=f"= ({wgt_issue_ms:.2f} ms issue + {r1['c']:.2f} ms DRAM) / {t:.2f} ms; "
                 f"the ladder's other read of the pass cost is {WEIGHT_PASS_MS[1]:.2f} ms",
        ),
        dict(
            key="R2_activation_doubling",
            expr="ratio(a) - 1",
            claim="cost of doubling the activation tile at unchanged grid and weights",
            lo=0.05, hi=0.20, point=r1["ratio_a"] - 1.0,
            note="THE quantity E33 could not separate; arm (a) measures it directly",
        ),
        dict(
            key="R3_serialization",
            expr=f"{E33_RATIO_M6:.4f} - ratio(b)",
            claim="grid-thinning / lost-parallelism penalty E33 paid, isolated",
            lo=0.00, hi=0.20,
            point=E33_RATIO_M6 - REGISTERED_RATIO,
            note="my account predicts ~+0.025; the 0.84 account requires ~+0.175",
        ),
    ]

# Register predictions, from E33's measured r=2 law.  regs = 15 + 17*NA.
REG_PREDICTIONS = {
    "crossrow_m6_ipg6_r2 (_wide<T,6,DN,2>)": (117, (113, 121)),
    "arm (a) _wide<T,3,DN,2>":               (66, (62, 70)),
}
REG_CEILING = 125  # shipped _wide<T,5,...,4> high-water: 20 + 21*5


def leg_movement_pct(ratio, phi=PHI_LOCAL, psi=PSI_MEASURED):
    """Predicted candidate-leg decode movement, percent, for a QMV M=6 ratio."""
    return (1.0 - ratio) * phi * psi * 100.0


def score_gain_pct(ratio, phi=PHI_LOCAL, psi=PSI_MEASURED):
    return SCORE_CHAIN * (1.0 - ratio) * phi * psi * 100.0


def ratio_for_leg_movement(pct, phi=PHI_LOCAL, psi=PSI_MEASURED):
    return 1.0 - (pct / 100.0) / (phi * psi)


def ratio_for_score_gain(pct, phi=PHI_LOCAL, psi=PSI_MEASURED):
    return 1.0 - (pct / 100.0) / (SCORE_CHAIN * phi * psi)


# Ship when the projected published-score gain reaches 1 sigma; call the result
# decisive on its own only at 2 sigma.  Run the E2E only where the local
# instrument can actually see the leg move.
SHIP_RATIO = ratio_for_score_gain(SIGMA_SCORE_PCT)
DECISIVE_RATIO = ratio_for_score_gain(2 * SIGMA_SCORE_PCT)
E2E_RATIO = ratio_for_leg_movement(E2E_RESOLUTION_PCT)


def report():
    r1, r2 = route1(), route2()
    print("=" * 78)
    print("E38 PRE-REGISTRATION -- committed before any E38 code or measurement")
    print("=" * 78)

    print("\n[1] ARM ARITHMETIC  (per k-block, summed over working threadgroups)")
    print(f"  {'arm':<8}{'U':>7}{'x-blocks':>10}{'passes':>8}{'MAC':>7}{'act':>7}{'wgt':>7}")
    for key, label in (("base", "base"), ("a_ctl", "(a)"), ("b_rep", "(b)"), ("e33", "E33")):
        u, p, act, mac, wgt = arm_units(**ARMS[key])
        xb = math.ceil(ARMS[key]["m"] / ARMS[key]["ipg"])
        xb *= (4 // ARMS[key]["rows_per_simd"]) if ARMS[key]["blocks_in_x"] else 1
        print(f"  {label:<8}{u:>7}{xb:>10}{p:>8}{mac:>7}{act:>7}{wgt:>7}")
    print("  => (a) vs base: MAC and weight issue identical, activations 2x  "
          f"({r1['act']['base']} -> {r1['act']['a']})")
    print("  => (a) vs (b):  MAC and activations identical, weight issue halved "
          f"({r1['wgt']['a']} -> {r1['wgt']['b']}), passes 2 -> 1")

    print("\n[2] ROUTE 1 -- issue-cost model fitted to the MEASURED base ladder")
    print(f"  T = {r1['a']:.2f} + {r1['b']:.5f}*U + {r1['c']:.2f}*(passes-1)   ms")
    print("  residuals (ms), M=2..9: " + " ".join(f"{v:+.2f}" for v in r1["resid"]))
    print(f"  E33 grid-thinning penalty S (measured - model) = {r1['s_e33']:+.2f} ms "
          f"({100*r1['s_e33']/BASE_C_ROUND_MS[6]:+.2f}% of base)")
    print(f"  predicted arm (b) ratio = {r1['ratio_b']:.4f}")
    print(f"  predicted arm (a) ratio = {r1['ratio_a']:.4f}")
    print("  reading: E38 does recover S, but the +224 issue units of the doubled")
    print("           activation tile very nearly cancel the saved weight pass.")

    print("\n[3] ROUTE 2 -- per-shape residual analysis of E33's measured table")
    print(f"  above-knee (>= {KNEE_TG} row-groups) E33 ratios: "
          + ", ".join(f"{v:.4f}" for v in r2["clean"]) + f"  median {r2['mid']:.4f}")
    print(f"  {'shape':<28}{'ceil(N/8)':>10}{'knee':>6}{'E33':>9}{'E38 pred':>10}{'calls':>7}")
    for r in r2["rows"]:
        print(f"  {r['name']:<28}{r['y_blocks']:>10}{'ok' if r['above_knee'] else 'BELOW':>6}"
              f"{r['r33']:>9.4f}{r['e38_point']:>10.4f}{r['calls']:>7}")
    print(f"  blended round ratio = {r2['ratio']:.4f}  "
          f"(band {min(r2['ratio_lo'], r2['ratio_hi']):.4f}..{max(r2['ratio_lo'], r2['ratio_hi']):.4f})")

    print("\n[3b] REGISTERED WITHIN-EXPERIMENT RELATIONS  (no cross-session model)")
    print("  All three arms are measured in ONE locked session against ONE fresh base,")
    print("  so these are identities over that session's own numbers.")
    for rel in registered_relations():
        pt = f"  point {rel['point']:+.4f}" if "point" in rel else ""
        print(f"  {rel['key']:<24} {rel['expr']:<22} in [{rel['lo']:+.4f}, {rel['hi']:+.4f}]{pt}")
        print(f"      {rel['claim']}")
        print(f"      {rel['note']}")
    print("  Consistency: ratio(b) = ratio(a) - R1 exactly.  The weakest link in route 1")
    print("  is the assumption that an activation-issue unit costs the same as a MAC")
    print("  unit; that assumption is what sets arm (a) at "
          f"{r1['ratio_a']:.4f}.  Arm (a) MEASURES it.")
    print("  If arm (a) lands materially below that, the activation tile is cheaper than")
    print("  modelled, ratio(b) drops with it, and that is precisely the pathway by which")
    print("  the 0.84 account could still be right -- with no appeal to my model at all.")
    print(f"  Break-even for arm (a): ratio(a) = {1 + registered_relations()[0]['point']:.4f} "
          "puts arm (b) exactly at parity with base.")

    print("\n[4] REGISTERED PRIMARY PREDICTION")
    print(f"  e38/m6_per_row_cost_ratio = {REGISTERED_RATIO:.3f}   "
          f"band [{REGISTERED_BAND[0]:.3f}, {REGISTERED_BAND[1]:.3f}]")
    print(f"  advisor's stated expectation = {ADVISOR_RATIO:.2f}  -- I register against it.")
    print("  Falsifier for MY number: any ratio <= 0.96 means the serialization")
    print("  account was right and my activation-cost account was wrong.")
    print("  Falsifier for 0.84: any ratio >= 0.96 means halving the weight pass")
    print("  cannot pay for the doubled activation tile at this arithmetic intensity.")

    print("\n[5] CONTROLS")
    print(f"  M in {{2,3,4,5,7,8,9}} are untreated; predict |ratio-1| <= {CONTROL_TOL:.4f}")
    print("  (that tolerance is E33's own worst untreated-M deviation, M=9 at 1.0046)")
    print("  M=1 is a global null: neither dispatch tier has a `case 1:`, so M=1")
    print("  never reaches the crossrow kernel.  Any M=1 movement invalidates the run.")
    print("  A serial-leg (M=1) speedup would LOWER the published score, so this")
    print("  control is a correctness gate, not a bonus.")

    print("\n[6] REGISTER PREDICTIONS (AIR probe, measured BEFORE production compile)")
    for name, (pred, band) in REG_PREDICTIONS.items():
        print(f"  {name:<40} {pred:>4}  band [{band[0]}, {band[1]}]")
    print(f"  gate: production cell must stay <= {REG_CEILING} (shipped _wide<T,5,..,4> high-water)")
    print("  gate: no `[N x <6 x float>]` accumulator spill in the AIR listing")

    print("\n[7] E2E DECISION RULE  (registered before looking at any E2E number)")
    print(f"  psi = {PSI_MEASURED:.3f} is MEASURED (QMV share of decode wall, E33, this host);")
    print("  phi = {:.3f} local cost share; ranked ROW share is >= {:.3f} and is NOT".format(PHI_LOCAL, PHI_RANKED_ROWS))
    print("  the same quantity -- do not substitute one for the other.")
    print(f"  predicted leg movement = (1-ratio) * {PHI_LOCAL} * {PSI_MEASURED} = (1-ratio) * "
          f"{PHI_LOCAL*PSI_MEASURED*100:.3f}%")
    for r in (ADVISOR_RATIO, 0.90, 0.934, REGISTERED_RATIO, 1.0150):
        print(f"    ratio {r:.4f} -> leg {leg_movement_pct(r):+.3f}% , score {score_gain_pct(r):+.3f}%")
    print(f"  instrument resolution (E33, n=2, 64 tokens) ~ {E2E_RESOLUTION_PCT:.2f}%")
    print(f"  RULE: run the 512-token >=4-paired-leg E2E iff ratio <= {E2E_RATIO:.4f}.")
    print("  Otherwise the E2E cannot resolve the effect and running it would only")
    print("  manufacture a noise number; report this arithmetic instead.")
    print(f"  Even at the advisor's {ADVISOR_RATIO:.2f}, the score gain is "
          f"{score_gain_pct(ADVISOR_RATIO):.3f}% against a {GAP_TO_FIRST_PCT:.3f}% gap to #1,")
    print("  so E38 alone does not take rank 1 under the MEASURED psi.  psi is the")
    print("  swing factor: at psi=0.40 the same ratio would give "
          f"{score_gain_pct(ADVISOR_RATIO, psi=0.40):.3f}%.")

    print("\n[8] SHIP / KILL")
    print(f"  SHIP     iff ratio <= {SHIP_RATIO:.4f}: projected score gain reaches 1 sigma "
          f"({SIGMA_SCORE_PCT:.4f}%).")
    print(f"  DECISIVE iff ratio <= {DECISIVE_RATIO:.4f}: projected score gain reaches 2 sigma "
          f"({2*SIGMA_SCORE_PCT:.4f}%),")
    print("           i.e. one ranked run could confirm it on its own.")
    print(f"  NULL  if ratio >= {1-CONTROL_TOL:.4f} (inside the control band): report a null,")
    print("        do not chase, leave the shipped cell alone.")
    print("  In every non-ship case the kernel is REVERTED so no slower cell can sit")
    print("  on the submission path.  Research artifacts and the (g) side-fix stay.")

    print("\n[9] RUNTIME GEOMETRY (advisor comment 5337327566 -- outranks the rest)")
    print("  Every arm's meta must record: MLX_MAX_MB_PER_BUFFER, MLX_MAX_OPS_PER_BUFFER,")
    print("  the resolved startup memory profile, whether the low-memory stderr notice")
    print("  appeared, and whether wireResidentWeightsIfEnabled took the ticket.")
    print("  OPEN QUESTION registered here: the cost-curve harness is")
    print("  `swift test --filter QwenQMVCostCurveTests`, which may never construct")
    print("  QwenRuntimeMTPWorker and therefore may run under NEITHER geometry (plain")
    print("  MLX defaults).  This is verified and reported before the primary result.")
    print("  A ranked-geometry arm (DARKBLOOM_STARTUP_MEMORY_PROFILE=full,")
    print("  MLX_MAX_MB_PER_BUFFER=512, MLX_MAX_OPS_PER_BUFFER=50) runs LAST and is a")
    print("  third arm, never a replacement: the two primary ratio arms stay in one")
    print("  geometry so the ratio stays paired.")


def self_test():
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    u_base, p_base, act_base, mac_base, wgt_base = arm_units(**ARMS["base"])
    u_a, p_a, act_a, mac_a, wgt_a = arm_units(**ARMS["a_ctl"])
    u_b, p_b, act_b, mac_b, wgt_b = arm_units(**ARMS["b_rep"])
    u_33, p_33, act_33, mac_33, wgt_33 = arm_units(**ARMS["e33"])

    chk(u_base == 1184, f"base U {u_base} != 1184")
    chk(u_a == 1472, f"(a) U {u_a} != 1472")
    chk(u_b == 1408, f"(b) U {u_b} != 1408")
    chk(u_33 == u_b, "E33 and arm (b) must have identical issue units")
    chk(p_33 == p_b == 1, "single-pass arms must report 1 weight pass")
    chk(p_base == p_a == 2, "base and control must report 2 weight passes")

    # The two isolations E38 is built on.
    chk(mac_a == mac_base and wgt_a == wgt_base and act_a == 2 * act_base,
        "(a) vs base must differ ONLY by doubled activation issue")
    chk(mac_a == mac_b and act_a == act_b and wgt_b * 2 == wgt_a,
        "(a) vs (b) must differ ONLY by one weight pass")

    # Coverage: ceil(N/8) row-groups * 2 simdgroups * ROW_BLOCKS * ROWS_PER_SIMD.
    for _, n, *_ in SHAPES_M6:
        chk(n % 8 == 0, f"shape n={n} is not a multiple of 8")
        chk(math.ceil(n / 8) * 2 * (4 // 2) * 2 == n, f"row coverage broken for n={n}")

    # Register law self-consistency: shipped _wide<T,5,..,4> is the ceiling.
    a4, b4 = REG_LAW[4]
    chk(a4 + b4 * 5 == REG_CEILING, "register ceiling must equal the shipped r=4 cell")
    a2, b2 = REG_LAW[2]
    chk(a2 + b2 * 6 == REG_PREDICTIONS["crossrow_m6_ipg6_r2 (_wide<T,6,DN,2>)"][0],
        "production register prediction must follow the r=2 law")
    chk(a2 + b2 * 3 == REG_PREDICTIONS["arm (a) _wide<T,3,DN,2>"][0],
        "arm (a) register prediction must follow the r=2 law")

    # x-block placement must not change total work, only where it happens.
    seq = arm_units(m=6, ipg=6, rows_per_simd=2, blocks_in_x=False)
    xbl = arm_units(m=6, ipg=6, rows_per_simd=2, blocks_in_x=True)
    chk(seq == xbl, "row-block placement must be work-neutral in this model")

    # Both routes must land inside the registered band.
    r1, r2 = route1(), route2()
    for label, val in (("route1", r1["ratio_b"]), ("route2", r2["ratio"])):
        chk(REGISTERED_BAND[0] <= val <= REGISTERED_BAND[1],
            f"{label} prediction {val:.4f} outside registered band {REGISTERED_BAND}")
    chk(r1["s_e33"] > 0, "E33's grid-thinning penalty must be positive")

    # Decision thresholds must be ordered and consistent.
    chk(DECISIVE_RATIO < E2E_RATIO < SHIP_RATIO < 1 - CONTROL_TOL,
        f"threshold order broken: decisive {DECISIVE_RATIO:.4f} e2e {E2E_RATIO:.4f} "
        f"ship {SHIP_RATIO:.4f} null {1-CONTROL_TOL:.4f}")
    chk(abs(score_gain_pct(SHIP_RATIO) - SIGMA_SCORE_PCT) < 1e-9,
        "ship threshold must equal exactly 1 sigma of the published score")
    chk(abs(score_gain_pct(DECISIVE_RATIO) - 2 * SIGMA_SCORE_PCT) < 1e-9,
        "decisive threshold must equal exactly 2 sigma of the published score")
    chk(abs(leg_movement_pct(E2E_RATIO) - E2E_RESOLUTION_PCT) < 1e-9,
        "e2e threshold must equal exactly the instrument resolution")
    chk(leg_movement_pct(REGISTERED_RATIO) < E2E_RESOLUTION_PCT,
        "at the registered prediction the E2E must be unresolvable")

    # Base ladder consistency: the fit must not be doing the work alone.
    chk(max(abs(v) for v in r1["resid"]) < 3.0,
        "base-ladder fit residual exceeds 3 ms; model is not trustworthy")

    # The registered relations must be mutually consistent with route 1.
    rels = {r["key"]: r for r in registered_relations()}
    for rel in rels.values():
        chk(rel["lo"] <= rel["point"] <= rel["hi"],
            f"{rel['key']} point {rel['point']:+.4f} outside its own band")
    chk(abs((r1["ratio_a"] - rels["R1_weight_pass"]["point"]) - r1["ratio_b"]) < 1e-9,
        "R1 must be exactly the arm(a)->arm(b) step implied by route 1")
    chk(abs(E33_RATIO_M6 - 1.0150) < 5e-5, "E33 M=6 ratio must reproduce the published 1.0150")

    if fails:
        print(f"SELF-TEST FAILED ({len(fails)}):")
        for f in fails:
            print("  - " + f)
        raise SystemExit(1)
    print("E38 prereg self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        self_test()
        print()
        report()
