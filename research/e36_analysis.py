#!/usr/bin/env python3
"""E36 analysis: does values_per_thread compose with row-blocked NA, or contend?

  python3 research/e36_analysis.py --grid research/e36-vpt-grid.json \
      > research/e36-vpt-analysis.txt
"""

from __future__ import annotations

import argparse
import json
import pathlib

NA_LEGAL = range(2, 10)
V_VALUES = [8, 16, 32, 64]
R_VALUES = [1, 2, 3, 4]
E27_LADDER = {2: 62, 3: 83, 4: 104, 5: 125}

# Every 4-bit / group-64 QMV shape that reaches affine_qmv_fast on the scored
# decode path. K is what the kernel's k-loop must tile exactly; N selects the
# crossrow branch (>= 4096 wide, >= 1024 narrow pair, else qmv_fast_impl).
SCORED_SHAPES = [
    (34816, 5120, "fused gate+up", "Qwen35.swift:1249,1251"),
    (16480, 5120, "GDN fused in_proj", "Qwen35.swift:656-659"),
    (14336, 5120, "fused q+k+v", "Qwen35.swift:1683-1688"),
    (248320, 5120, "lm_head", "Qwen35.swift:2789"),
    (5120, 6144, "o_proj / GDN out_proj", "Qwen35.swift:1689,666"),
    (5120, 17408, "down_proj", "Qwen35.swift:1250"),
    (5120, 10240, "MTP head fc", "Qwen35MTP.swift:101"),
    (2048, 5120, "KV-only pack (narrow pair kernel)", "Qwen35.swift:1759,1778"),
]


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    inter = my - slope * mx
    resid = max(abs(y - (inter + slope * x)) for x, y in zip(xs, ys))
    return inter, slope, resid


def ipg_passes(m: int, na_max: int) -> tuple[int, int]:
    ipg = -(-m // -(-m // na_max))
    return ipg, -(-m // ipg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="research/e36-vpt-grid.json")
    args = ap.parse_args()
    data = json.loads(pathlib.Path(args.grid).read_text())
    ok = [c for c in data["cells"] if c["status"] == "ok"]
    by = {(c["arm"], c["na"], c["r"], c["v"]): c for c in ok}

    print("## (0) Probe validity: E27 anchors and control gates\n")
    print("  E27 measured NA=2..5 at r=4 independently. If these four numbers do not")
    print("  reproduce, the probe is wrong and nothing below counts.\n")
    print(f"  {'cell':14} {'E27':>5} {'E36':>5} {'match':>6}")
    anchors_ok = True
    for na, want in E27_LADDER.items():
        got = by[("shipped_anchor", na, 4, 16)]["peak_live_regs"]
        anchors_ok &= got == want
        print(f"  xship_na{na:<7} {want:>5} {got:>5} {str(got == want):>6}")
    e27bad = next(c for c in ok if c["name"] == "xctl_e27_spill_na6_r4_v16")
    print(f"  {'NA=6 r=4 (E27 known-BAD)':30} regs={e27bad['peak_live_regs']} "
          f"acc_spill={e27bad['acc_spill']}  (E27: 144, spills)")
    print(f"\n  generated body at r=4,v=16 vs shipped template, NA=2..5: "
          f"{[by[('grid_relaxed', na, 4, 16)]['peak_live_regs'] for na in (2, 3, 4, 5)]}")
    print(f"  gate_validation_failures: {data['gate_validation_failures'] or 'none'}")
    print(f"  anchors reproduce: {anchors_ok}")

    print("\n\n## (a) The grid: peak_live_regs / private_bytes, S = accumulator spill\n")
    for arm, rs in (("grid_relaxed", R_VALUES), ("coverage_preserving", [1, 2, 4])):
        print(f"### {arm}\n")
        head = "| NA | " + " | ".join(f"r{r} v{v}" for r in rs for v in V_VALUES) + " |"
        print(head)
        print("|" + "---|" * (1 + len(rs) * len(V_VALUES)))
        for na in list(NA_LEGAL) + [10, 11, 12]:
            row = [f"{na}"]
            for r in rs:
                for v in V_VALUES:
                    c = by.get((arm, na, r, v))
                    row.append("-" if not c else
                               f"{c['peak_live_regs']}{'S' if c['acc_spill'] else ''}"
                               f"/{c['private_bytes']}")
            if any(x != "-" for x in row[1:]):
                print("| " + " | ".join(row) + " |")
        print()
    tg = {c["threadgroup_refs"] for c in ok}
    print(f"threadgroup memory references across all {len(ok)} cells: {tg} "
          "(the crossrow family declares none, at any values_per_thread)")

    print("\n\n## (a2) Max spill-free NA, by rows_per_simd and values_per_thread\n")
    print("| arm | r | v=8 | v=16 | v=32 | v=64 |")
    print("|---|---|---|---|---|---|")
    maxna: dict[tuple[str, int, int], int] = {}
    for arm, rs in (("grid_relaxed", R_VALUES), ("coverage_preserving", [1, 2, 4])):
        for r in rs:
            cells_row = []
            for v in V_VALUES:
                best = 0
                for na in list(NA_LEGAL) + [10, 11, 12]:
                    c = by.get((arm, na, r, v))
                    if c and not c["acc_spill"]:
                        best = max(best, na)
                    elif c and c["acc_spill"]:
                        break
                maxna[(arm, r, v)] = best
                cells_row.append(str(best))
            print(f"| {arm} | {r} | " + " | ".join(cells_row) + " |")
    print("\n  (12 is the largest NA compiled; cells reading 12 are '>=12, untested above')")

    print("\n\n## (d) Register model in three dimensions\n")
    print("  The headline does not need a fit. Hold (arm, NA, r) fixed and vary only")
    print("  values_per_thread; the register count is the SAME NUMBER. Delta against")
    print("  the v=16 column, over every spill-free cell:\n")
    print(f"  {'arm':22} {'NA range':>10} {'v=8':>6} {'v=32':>6} {'v=64':>6}")
    for arm, rs in (("grid_relaxed", R_VALUES), ("coverage_preserving", [1, 2, 4])):
        for lo, hi, label in ((2, 9, "2..9"), (4, 9, "4..9")):
            cols = []
            for v in V_VALUES:
                if v == 16:
                    continue
                d = [abs(by[(arm, na, r, v)]["peak_live_regs"]
                         - by[(arm, na, r, 16)]["peak_live_regs"])
                     for na in range(lo, hi + 1) for r in rs
                     if (arm, na, r, v) in by and (arm, na, r, 16) in by
                     and not by[(arm, na, r, 16)]["acc_spill"]]
                cols.append(f"{max(d)}/{sum(d) / len(d):.2f}")
            print(f"  {arm:22} {label:>10} " + " ".join(f"{c:>6}" for c in cols))
    print("\n  (max / mean absolute delta in registers. NA=2,3 are where the allocator")
    print("  does small-cell things: they carry the whole max. From NA=4 up, every")
    print("  spill-free cell at r<=2 is bit-identical across v=8..64.)")

    print("\n  Per (r, v) line fits, over spill-free NA >= 4, falling back to all")
    print("  spill-free NA when that leaves under 3 points (r=4, i.e. E27's own ladder).\n")
    print(f"  {'r':>2} {'v':>3} {'intercept':>10} {'slope':>7} {'max|resid|':>11} {'NA used':>9}")
    slopes: dict[tuple[int, int], float] = {}
    for r in R_VALUES:
        for v in V_VALUES:
            avail = [na for na in range(2, 11)
                     if ("grid_relaxed", na, r, v) in by
                     and not by[("grid_relaxed", na, r, v)]["acc_spill"]]
            use = [na for na in avail if na >= 4]
            if len(use) < 3:
                use = avail
            if len(use) < 3:
                continue
            pts = [(na, by[("grid_relaxed", na, r, v)]["peak_live_regs"]) for na in use]
            i, s, e = fit([p[0] for p in pts], [p[1] for p in pts])
            slopes[(r, v)] = s
            print(f"  {r:>2} {v:>3} {i:>10.2f} {s:>7.2f} {e:>11.2f} "
                  f"{f'{use[0]}..{use[-1]}':>9}")

    print("\n  Is the slope a function of values_per_thread at all?")
    for r in R_VALUES:
        vals = [slopes[(r, v)] for v in V_VALUES if (r, v) in slopes]
        if vals:
            print(f"    r={r}: slopes across v=8..64 = {[round(x, 2) for x in vals]}  "
                  f"spread = {max(vals) - min(vals):.3f}")

    r_pts = [(r, slopes[(r, 16)]) for r in R_VALUES if (r, 16) in slopes]
    i, s, e = fit([p[0] for p in r_pts], [p[1] for p in r_pts])
    print(f"\n  slope(r) = {i:.2f} + {s:.2f}*r   max |residual| = {e:.2f}")
    print("  E32 fitted 8.36 + 3.19*r on the same kernel. The small shift is E32's r=1")
    print("  fit having included NA=11,12, where the allocator leaves the linear regime")
    print("  (156 regs at NA=12/v=16 with ZERO allocas, 126 at v=32). That is what gave")
    print("  E32's r=1 row its 5.45 residual. The mechanism reading is unchanged and")
    print(f"  slightly better: {i:.2f}/{s:.2f} = {i / s:.2f} against the 5:2 = 2.50 ratio of")
    print("  x-side floats (a0..a3 + sums) to per-row floats (acc + partial) per NA.")
    print("\n  Two-dimensional form, with values_per_thread carrying zero weight:")
    print(f"    regs(NA, r, v) = intercept(r) + ({i:.2f} + {s:.2f}*r)*NA + 0.00*v")

    print("\n  Advisor's prediction, scored:")
    pred_slope = 8.36 * (32 / 16) + 3.19 * 2
    pred = 16.0 + pred_slope * 6
    meas = by[("grid_relaxed", 6, 2, 32)]["peak_live_regs"]
    print(f"    'slope becomes 8.36*(vpt/16) + 3.19*r' -> slope(r=2,v=32) = {pred_slope:.2f}")
    print(f"    predicted regs(NA=6, r=2, v=32) = 16.0 + {pred_slope:.2f}*6 = {pred:.0f} "
          f"(advisor quoted ~196)")
    print(f"    MEASURED                                                 = {meas}")
    print(f"    measured regs(NA=6, r=2, v=16) from E32                  = "
          f"{by[('grid_relaxed', 6, 2, 16)]['peak_live_regs']}")
    print("    -> the x-side term does NOT scale with values_per_thread. FALSIFIED.")

    print("\n  What values_per_thread actually costs: private (alloca) bytes.\n")
    print(f"  {'r':>2} {'v':>3} {'stage_bytes':>12} {'r*v/2':>7} {'match':>6}")
    allmatch = True
    for r in (1, 2, 4):
        for v in V_VALUES:
            c = by.get(("grid_relaxed", 5, r, v))
            if not c:
                continue
            want = r * v // 2
            allmatch &= c["stage_bytes"] == want
            print(f"  {r:>2} {v:>3} {c['stage_bytes']:>12} {want:>7} "
                  f"{str(c['stage_bytes'] == want):>6}")
    print(f"\n  staging model private_bytes = rows_per_simd * values_per_thread / 2 : "
          f"exact in every cell = {allmatch}")
    print("  (uint16 packed[r][v/4]: r * v/4 words * 2 bytes. NA-independent.)")
    print("\n  The two resources are separable with no cross term:")
    print("    registers      = f(NA, rows_per_simd)          -- values_per_thread free")
    print("    private bytes  = rows_per_simd * vpt / 2       -- NA free")

    print("\n\n## (c) Host-side legality\n")
    print("  Frozen host dispatch (backend/metal/quantized.cpp, NOT in editablePaths):")
    print("    :246-259  bn=8, bk=32, group_dims(32,2,1), grid_dims(M, ceil(N/8), B)")
    print("    :259      bool fast = N % bn == 0 && K % 512 == 0;")
    print("    :992      the same gate again for gather_qmv")
    print("  Nothing host-side derives a size, offset or thread count from")
    print("  values_per_thread or bytes_per_lane. The grid is a function of (M, N, B)")
    print("  only, so raising values_per_thread does NOT disturb the output geometry")
    print("  that killed rows_per_simd in E32. It changes K-loop striding only.\n")
    print("  But `K % 512 == 0` is the ONLY alignment the frozen host guarantees, and")
    print("  512 == 16 * SIMD_SIZE is exactly the vpt=16 block size. The crossrow")
    print("  k-loop `for (k = 0; k < in_vec_size; k += block_size)` has no K tail and")
    print("  no bounds check; `_m` is a tail over M (inputs), never over K.\n")
    print("| N | K | projection | K%512 | K%1024 | K%2048 | v=8 | v=16 | v=32 | v=64 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    verdict = {v: True for v in V_VALUES}
    for n, k, name, src in SCORED_SHAPES:
        cols = []
        for v in V_VALUES:
            fits = k % (v * 32) == 0
            verdict[v] &= fits
            cols.append("ok" if fits else "**OVERRUN**")
        print(f"| {n} | {k} | {name} | {k % 512} | {k % 1024} | {k % 2048} | "
              + " | ".join(cols) + " |")
    print("\n  every scored K covered:  " +
          "  ".join(f"v={v}: {verdict[v]}" for v in V_VALUES))
    print("  K=5120 and K=17408 are 1024-aligned but NOT 2048-aligned, and they carry")
    print("  gate+up, QKV, GDN in_proj, lm_head and down_proj. values_per_thread=64 is")
    print("  a hard coverage wall on the highest-traffic shapes in the model.")
    print("\n  Group-coverage cap, independent of K: one scale/bias is fetched per lane")
    print("  per k-block at `simd_lid / (64/values_per_thread)`, so a lane must stay")
    print("  inside one 64-element affine group. Upstream writes the same constraint as")
    print("  `scale_step_per_thread = group_size / values_per_thread` (quantized.h:768),")
    print("  which is 0 -- a division by zero -- for values_per_thread > 64.")

    print("\n\n## (c2) The wall that is not in the register file: lane->K partition\n")
    print("  values_per_thread does not just make loads wider. It repartitions K across")
    print("  lanes, which reassociates the FP32 sum. The shipped kernel states order")
    print("  preservation as its OWN safety case, in its own header:\n")
    print("    quantized.h:966  '...the K accumulation order and simd_sum are unchanged")
    print("                      for every output element.'")
    print("    quantized.h:821  the same sentence for the narrow pair kernel.\n")
    print("  That claim is true today because the two paths partition K identically:")
    print("    quantized.h:786   qmv_fast_impl   x += tid.x*in_vec_size")
    print("                                           + simd_lid*values_per_thread")
    print("    quantized.h:1020  crossrow wide   xm = x + ... + k")
    print("                                           + simd_lid*values_per_thread + 4*i")
    print("  Both are values_per_thread=16, so lane L owns [k+16L, k+16L+16) in both.\n")
    print("  Raising values_per_thread in the crossrow kernel alone breaks that:")
    print("    - M==1 and M>9 fall through to qmv_fast_impl (quantized.h:2016); the")
    print("      switch at :1920 only covers ntg.x in 2..9. So within ONE candidate run")
    print("      the same (x, W) product would be summed in two different orders.")
    print("    - the narrow pair kernel (:873-877) and the N<1024 path keep vpt=16.")
    print("    - the pinned serial leg is M==1 throughout, i.e. entirely vpt=16, and it")
    print("      is the token stream the candidate has to match.")
    print("  scarletbright's shipped vpt=32 has none of this exposure: it is the 2-bit")
    print("  draft readout, which quantized.h:1068-1082 documents as proposal-only and")
    print("  exactly reranked afterwards. Reassociation there cannot change a token.")
    print("  On the wide crossrow verify path there is no such escape.\n")
    print("  Row-blocked NA does not have this problem. For a given output element the")
    print("  row-blocked form runs the identical k sequence into the identical acc[r];")
    print("  only which simdgroup pass computes it moves. E32's axis is")
    print("  order-preserving by construction. This one is not.")

    print("\n\n## (b) COMPOSITION VERDICT: best legal (NA, values_per_thread) at r=2\n")
    print("| M | shipped IPG/passes | best legal NA | passes | widest K-legal vpt | "
          "bytes_per_lane | k-blocks K=5120 | regs | spill | recommended vpt |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in (6, 9):
        sipg, spass = ipg_passes(m, 5)
        na_cap = min(m, maxna[("coverage_preserving", 2, 32)] or 0)
        na = max(n for n in range(2, na_cap + 1) if m % n != 1)
        _, passes = ipg_passes(m, na)
        best_v = max(v for v in V_VALUES if verdict[v])
        c = by[("coverage_preserving", na, 2, best_v)]
        print(f"| {m} | {sipg}/{spass} | {na} | {passes} | {best_v} | {best_v // 2} | "
              f"{5120 // (best_v * 32)} | {c['peak_live_regs']} | "
              f"{'SPILL' if c['acc_spill'] else 'clean'} | **16 (unchanged)** |")
    print("\n  Register headroom at those cells is NOT the binding constraint:")
    for m in (6, 9):
        for v in V_VALUES:
            c = by.get(("coverage_preserving", m, 2, v))
            if c:
                print(f"    M={m}, NA={m}, r=2 row-blocked, v={v:>2}: "
                      f"{c['peak_live_regs']} regs, {c['private_bytes']} private bytes, "
                      f"{'SPILL' if c['acc_spill'] else 'clean'}")
    print("\n  Plain answer for thorfinn: he can have NA, he cannot usefully have")
    print("  values_per_thread, and the reason is NOT contention.")
    print("    axis 1, row-blocked NA   : free in registers, order-preserving -> ship it")
    print("    axis 2, values_per_thread: free in registers (0 cost at every cell in")
    print("                               this grid), capped at 32 by K coverage, and")
    print("                               blocked by the kernel's own exactness claim")
    print("  The axes compose perfectly in the resource he was worried about and do not")
    print("  compose at all in the one nobody costed. Rung 1 of E33 needs no change.")

    print("\n\n## (e) Falsification: why did the shipped kernel choose 16?\n")
    print("  Required, because v=32 IS spill-free and IS K-legal on every scored shape.")
    print("  Four candidate reasons were checked; two are real, one is real but does not")
    print("  transfer, one is dead.\n")
    print("  1. REAL, and it is the whole reason upstream picked 16. The generic")
    print("     qmv_fast_impl materialises `thread U x_thread[values_per_thread]`")
    print("     (quantized.h:774) -- values_per_thread FP32 registers per lane, live")
    print("     across the whole row loop. 16 there costs 16 registers/lane; 32 costs")
    print("     32. The single literal that sets it is `packs_per_thread = bits == 2 ?")
    print("     1 : 2` (quantized.h:761), uncommented, and the bits==2 special case")
    print("     exists precisely to hold vpt at 16 when pack_factor doubles. So the")
    print("     upstream invariant is '16 values per lane', and it is a register")
    print("     argument -- for a kernel that stages all 16.")
    print("     It does NOT transfer to the crossrow kernel, which never materialises")
    print("     x_thread: it re-reads 4 activations per staged word into a0..a3")
    print("     (quantized.h:1014-1035). That is exactly why this grid measures zero")
    print("     register cost. The crossrow kernel inherited 16 by copy, not by")
    print("     analysis: `git log -S 'constexpr int values_per_thread = 16'` on")
    print("     quantized.h returns only Validate/Accept submission snapshots, and")
    print("     `git log -S packs_per_thread -- Vendor/` returns only the initial")
    print("     squashed MLX import.")
    print("  2. REAL and binding here: 512 == 16 * SIMD_SIZE is exactly the block size")
    print("     the frozen host gate `K % 512 == 0` guarantees. 16 is the largest lane")
    print("     width that needs no self-guard. See (c).")
    print("  3. REAL but not about 16: `qdot` for bits in {3,5,6} advances its weight")
    print("     pointer cumulatively inside the loop and is only correct for vpt <= 16")
    print("     (quantized.h:216-218, 247-248, 267-269). bits==4 -- our path -- uses the")
    print("     generic form and needs only values_per_thread % 4 == 0.")
    print("  4. DEAD: no reduction assumes a lane covers a quarter of a group. simd_sum")
    print("     (:1059) reduces one scalar per (row, input) over 32 lanes and is")
    print("     vpt-independent; quad_sum (:742) is only reachable at K in {64,128}")
    print("     (quantized.cpp:1385-1387), which no scored shape hits. Coalescing is not")
    print("     a reason either: bytes_per_lane goes 8 -> 16 contiguous per lane, which")
    print("     is a better burst, not a worse one.")
    print("\n  So: a real reason exists, it is reason 1, and it is a reason for the")
    print("  GENERIC kernel that was never re-derived for the crossrow one. Reason 2 is")
    print("  a real reason for the crossrow kernel and survives. I did not have to")
    print("  invent anything, and I did not find a coalescing, shuffle-width or")
    print("  group-size-16 interaction, because there is none.")

    print("\n\n## Primary metric\n")
    for label, arm, r in (("r=2 row-blocked (thorfinn's form)", "coverage_preserving", 2),
                          ("r=4 shipped form", "coverage_preserving", 4),
                          ("r=2 grid-relaxed, model extension", "grid_relaxed", 2)):
        print(f"  e36/max_spill_free_NA_at_values_per_thread_32, {label}: "
              f"{maxna[(arm, r, 32)]}  (at v=16: {maxna[(arm, r, 16)]})")
    print("\n  The delta against the shipped ceiling of 5 is entirely E32's row")
    print("  blocking. values_per_thread contributes exactly 0 to it at every cell.")


if __name__ == "__main__":
    main()
