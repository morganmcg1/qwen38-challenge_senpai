#!/usr/bin/env python3
"""E147 rung E-1a: static verification of the grid-stride retile dispatch.

Replays the in-kernel grid-stride loop over the frozen host launch grid and
proves, for every scored projection, that the retiled tile grid is visited
exactly once per tile with no origin outside the matrix.

Implements the F7 ruling:
  * no divisibility gate at all (no `M % 128`, no `N % BN`);
  * correctness comes from the grid-stride loop, not from a precondition;
  * report `e147_rungE_launched_vs_required` and
    `e147_rungE_grid_stride_max_iters` for the seven scored projections at
    M=512 and M=511.

A second probe sweeps the M values F7 named as under-covering, to show that
the grid-stride form is correct there too and that M=511 needs no exclusion.

Zero GPU. Pure arithmetic replay plus source assertions.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
OUT = ROOT / "research/e147-retile-arm.json"

# (name, host BM, host BN, retiled BM, retiled BN, note)
PATHS = [
    ("local_non_nax", 32, 32, 64, 16, "applegpu_g16s, this Mac, affine_qmm_t"),
    ("ranked_nax", 64, 64, 128, 32, "applegpu_g17s, ranked M5, affine_qmm_t_nax"),
]

# (name, N) for the seven scored GEMM projections.
PROJECTIONS = [
    ("gdn.in_proj", 16480),
    ("gdn.out_proj", 5120),
    ("fa.qkv", 14336),
    ("fa.o_proj", 5120),
    ("mlp.gate_up", 34816),
    ("mlp.down", 5120),
    ("lm_head", 248320),
]

# The two M values the scored prefill and head-priming phases actually use.
M_VALUES = [512, 511]

# F7 named these as under-covering under the un-strided form.  M=511 is in the
# list to show it is safe and needs no exclusion.
M_PROBE = [64, 129, 192, 320, 448, 511, 512]

VARIANTS = (
    "correct",
    "no_bounds_check",
    "no_stride",
    "host_tiles_as_arm",
    "swapped",
    "off_by_one",
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def replay(M, N, host_bm, host_bn, bm, bn, variant="correct"):
    """Replay every launched thread's dispatch under one variant.

    Returns (visits, max_iters, launched, required).  `visits` maps a tile
    origin (y_row, y_col) to how many times it was computed.  Origins outside
    the matrix are kept as keys so over-coverage shows up instead of being
    silently dropped.
    """
    tiles_x_host = ceil_div(N, host_bn)
    tiles_y_host = ceil_div(M, host_bm)
    launched = tiles_y_host * tiles_x_host

    tiles_x = ceil_div(N, bn)
    tiles_y = ceil_div(M, bm)
    required = tiles_y * tiles_x

    if variant == "host_tiles_as_arm":
        stride = required  # bug: stride by the retiled count, not the launched count
    elif variant == "off_by_one":
        stride = launched + 1
    else:
        stride = launched

    single_pass = variant in ("no_bounds_check", "no_stride")
    bounds_checked = variant != "no_bounds_check"

    visits: dict[tuple[int, int], int] = {}
    max_iters = 0

    for ty in range(tiles_y_host):
        for tx in range(tiles_x_host):
            t = ty * tiles_x_host + tx
            iters = 0
            while True:
                if single_pass and iters >= 1:
                    break
                if bounds_checked and t >= required:
                    break
                if variant == "swapped":
                    y_row = (t % tiles_x) * bm
                    y_col = (t // tiles_x) * bn
                else:
                    y_row = (t // tiles_x) * bm
                    y_col = (t % tiles_x) * bn
                visits[(y_row, y_col)] = visits.get((y_row, y_col), 0) + 1
                iters += 1
                t += stride
            max_iters = max(max_iters, iters)

    return visits, max_iters, launched, required


def expected_origins(M, N, bm, bn):
    return {
        (y * bm, x * bn)
        for y in range(ceil_div(M, bm))
        for x in range(ceil_div(N, bn))
    }


def classify(visits, M, N, bm, bn):
    want = expected_origins(M, N, bm, bn)
    got = set(visits)
    missing = want - got
    extra = got - want
    doubled = {k for k, v in visits.items() if v > 1}
    overrun = {(r, c) for (r, c) in got if r >= M or c >= N}
    ok = not (missing or extra or doubled or overrun)
    return {
        "ok": ok,
        "missing": len(missing),
        "extra": len(extra),
        "doubled": len(doubled),
        "overrun": len(overrun),
        "first_overrun": list(sorted(overrun)[0]) if overrun else None,
        "first_missing": list(sorted(missing)[0]) if missing else None,
    }


def sweep(m_values, variant="correct"):
    rows = []
    for path_name, host_bm, host_bn, bm, bn, note in PATHS:
        for proj, N in PROJECTIONS:
            for M in m_values:
                visits, max_iters, launched, required = replay(
                    M, N, host_bm, host_bn, bm, bn, variant=variant
                )
                rows.append(
                    {
                        "path": path_name,
                        "note": note,
                        "projection": proj,
                        "N": N,
                        "M": M,
                        "host_tile": f"{host_bm}x{host_bn}",
                        "arm_tile": f"{bm}x{bn}",
                        "launched": launched,
                        "required": required,
                        "surplus": launched - required,
                        "max_iters": max_iters,
                        **classify(visits, M, N, bm, bn),
                    }
                )
    return rows


def source_assertions():
    header = HEADER.read_text()
    twin = TWIN.read_text()
    stride_decl = "const int tiles_x_host = (N + kHostBN - 1) / kHostBN;"
    loop_re = re.compile(
        r"for \(int t = int\(tid\.y\) \* tiles_x_host \+ int\(tid\.x\); t < required;"
    )
    return {
        "header_arm_default_off": "constexpr bool kE147RetileOn = false;" in header,
        "twin_arm_default_off": "constexpr bool kE147RetileOn = false;" in twin,
        "header_arm_on_absent": "constexpr bool kE147RetileOn = true;" not in header,
        "twin_arm_on_absent": "constexpr bool kE147RetileOn = true;" not in twin,
        "header_grid_stride_present": stride_decl in header,
        "twin_grid_stride_present": stride_decl in twin,
        "header_grid_stride_loop_present": bool(loop_re.search(header)),
        "twin_grid_stride_loop_present": bool(loop_re.search(twin)),
        "header_retile_ok_gate_absent": "retile_ok" not in header,
        "twin_retile_ok_gate_absent": "retile_ok" not in twin,
        "header_m_mod_128_gate_absent": "% 128 == 0" not in header,
        "twin_m_mod_128_gate_absent": "% 128 == 0" not in twin,
    }


def main() -> int:
    scored_rows = sweep(M_VALUES)
    probe_rows = sweep(M_PROBE)

    controls = []
    for variant in VARIANTS:
        rows = sweep(M_PROBE, variant=variant)
        rejected = [r for r in rows if not r["ok"]]
        expect_all_ok = variant == "correct"
        controls.append(
            {
                "variant": variant,
                "cases": len(rows),
                "cases_rejected": len(rejected),
                "expect_all_ok": expect_all_ok,
                "control_passed": (not rejected) if expect_all_ok else bool(rejected),
                "first_failure": (
                    {
                        k: rejected[0][k]
                        for k in (
                            "path",
                            "projection",
                            "M",
                            "launched",
                            "required",
                            "missing",
                            "doubled",
                            "overrun",
                            "first_overrun",
                            "first_missing",
                        )
                    }
                    if rejected
                    else None
                ),
            }
        )

    # The exact case F7 named: ranked NAX gdn.in_proj at M=512 over-covers by
    # four threads, so the un-strided form walks off the end of y.
    f7 = {}
    for variant in ("correct", "no_bounds_check"):
        visits, max_iters, launched, required = replay(
            512, 16480, 64, 64, 128, 32, variant=variant
        )
        v = classify(visits, 512, 16480, 128, 32)
        f7[variant] = {
            "launched": launched,
            "required": required,
            "surplus": launched - required,
            "max_iters": max_iters,
            **v,
        }
    f7["matches_f7_arithmetic"] = (
        f7["correct"]["launched"] == 2064
        and f7["correct"]["required"] == 2060
        and f7["correct"]["ok"]
        and f7["no_bounds_check"]["overrun"] == 4
        and f7["no_bounds_check"]["first_overrun"] == [512, 0]
    )

    src = source_assertions()
    scored_ok = all(r["ok"] for r in scored_rows)
    probe_ok = all(r["ok"] for r in probe_rows)
    controls_ok = all(c["control_passed"] for c in controls)
    src_ok = all(src.values())

    report = {
        "experiment": "e147",
        "rung": "E-1a",
        "form": "grid-stride, no divisibility gate (F7 section 3)",
        "e147_rungE_launched_vs_required": [
            {
                "path": r["path"],
                "projection": r["projection"],
                "M": r["M"],
                "launched": r["launched"],
                "required": r["required"],
                "surplus": r["surplus"],
                "max_iters": r["max_iters"],
                "ok": r["ok"],
            }
            for r in scored_rows
        ],
        "e147_rungE_grid_stride_max_iters": max(r["max_iters"] for r in scored_rows),
        "e147_rungE_grid_stride_max_iters_probe": max(
            r["max_iters"] for r in probe_rows
        ),
        "scored_rows": scored_rows,
        "probe_rows": probe_rows,
        "positive_controls": controls,
        "f7_named_case_ranked_nax_gdn_in_proj_M512": f7,
        "source_assertions": src,
        "scored_rows_ok": scored_ok,
        "probe_rows_ok": probe_ok,
        "all_controls_passed": controls_ok,
        "all_source_assertions_passed": src_ok,
        "verdict": "PASS"
        if (scored_ok and probe_ok and controls_ok and src_ok and f7["matches_f7_arithmetic"])
        else "FAIL",
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    hdr = (
        f"{'path':<14} {'projection':<14} {'N':>7} {'M':>4} {'host':>7} {'arm':>8} "
        f"{'launch':>7} {'req':>7} {'surp':>5} {'iters':>5} {'ok':>5}"
    )
    print("=== scored M values (e147_rungE_launched_vs_required)")
    print(hdr)
    print("-" * len(hdr))
    for r in scored_rows:
        print(
            f"{r['path']:<14} {r['projection']:<14} {r['N']:>7} {r['M']:>4} "
            f"{r['host_tile']:>7} {r['arm_tile']:>8} {r['launched']:>7} "
            f"{r['required']:>7} {r['surplus']:>5} {r['max_iters']:>5} "
            f"{str(r['ok']):>5}"
        )

    print()
    print("=== F7 under-coverage probe, correct form (worst row per M)")
    print(f"{'M':>4} {'ceil(M/64)':>10} {'parity':>7} {'max_iters':>9} {'all_ok':>7}")
    for M in M_PROBE:
        sub = [r for r in probe_rows if r["M"] == M]
        c = ceil_div(M, 64)
        print(
            f"{M:>4} {c:>10} {'even' if c % 2 == 0 else 'odd':>7} "
            f"{max(r['max_iters'] for r in sub):>9} "
            f"{str(all(r['ok'] for r in sub)):>7}"
        )

    print()
    print("=== positive controls (swept over the probe M set)")
    for c in controls:
        print(
            f"  {c['variant']:<18} rejected {c['cases_rejected']:>3}/{c['cases']:<3} "
            f"passed={c['control_passed']}"
        )
        if c["first_failure"]:
            f = c["first_failure"]
            print(
                f"      first: {f['path']}/{f['projection']} M={f['M']} "
                f"launched={f['launched']} required={f['required']} "
                f"missing={f['missing']} doubled={f['doubled']} "
                f"overrun={f['overrun']} first_overrun={f['first_overrun']}"
            )

    print()
    print("=== F7 named case: ranked NAX gdn.in_proj, N=16480, M=512")
    for variant in ("correct", "no_bounds_check"):
        d = f7[variant]
        print(
            f"  {variant:<18} launched={d['launched']} required={d['required']} "
            f"surplus={d['surplus']} max_iters={d['max_iters']} "
            f"overrun={d['overrun']} first_overrun={d['first_overrun']} "
            f"ok={d['ok']}"
        )
    print(f"  matches_f7_arithmetic = {f7['matches_f7_arithmetic']}")

    print()
    print("=== source assertions")
    for k, v in src.items():
        print(f"  {k:<38} {v}")

    print()
    print(
        f"e147_rungE_grid_stride_max_iters = "
        f"{report['e147_rungE_grid_stride_max_iters']} (scored), "
        f"{report['e147_rungE_grid_stride_max_iters_probe']} (probe)"
    )
    print(f"verdict = {report['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
