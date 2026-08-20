#!/usr/bin/env python3
"""Turn one E83 session's `decomposition.json` into the rung-1 and rung-2 report.

    usage: research/e83_report.py research/out/TAG/decomposition.json

Everything printed here is `harness=local` on this host. The local seed-prefill
GEMM share does not determine the ranked one; the transfer band at the end makes
that explicit instead of hiding it behind a single number.
"""
from __future__ import annotations

import json
import statistics
import sys

# Rung 0 static accounting, recomputed by research/e83_prefill_accounting.py.
GEMM_TFLOP = 24.9375123
NON_GEMM_GFLOP = 169.5

# H-221: the promoted `qwen35DualRMSNorm` removed one head dispatch per step and
# bought about this much. The question is whether the same per-boundary price
# applies at seed width, where each dispatch carries 512x the work.
H221_MS_PER_BOUNDARY = 0.35

PHASES = [
    "p1_cache_alloc",
    "p2_target_forward_build",
    "p3_target_forward_eval",
    "p4_tail_norm_lmhead",
    "p5_top_two",
    "p6_final_eval",
    "p7_host_readback",
]


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def spread(xs: list[float]) -> str:
    if len(xs) < 2:
        return "-"
    return f"{1e3 * min(xs):.1f}-{1e3 * max(xs):.1f}"


def section(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    payload = json.load(open(sys.argv[1]))
    blocks = payload["blocks"]
    ident = payload["identity"]

    print(f"# E83 seed-prefill decomposition -- {payload['experiment']}")
    print(f"\nharness={payload['harness']}  "
          f"gate_qualified_for_timing={payload['gate_qualified_for_timing']}  "
          f"device={ident.get('device')}  host={ident.get('host')}")
    print(f"seed_length={ident['seed_length']}  reps={ident['reps']}  "
          f"layers={ident['num_hidden_layers']}  hidden={ident['hidden_size']}")

    begins = [b for b in blocks if b.get("kind") == "begin"]
    clean = [b for b in begins if b.get("arm") == "baseline"
             and not b.get("stall_phase")]
    whole = [b for b in clean if not b["phased"]]
    phased = [b for b in clean if b["phased"]]

    section("Rung 1 -- end-to-end `begin()`")
    if whole:
        w = [b["begin_seconds"] for b in whole]
        print(f"unphased begin(): median {1e3 * med(w):.1f} ms  "
              f"range {spread(w)} ms  n={len(w)}")
        print(f"  build      {1e3 * med([b['build_seconds'] for b in whole]):.1f} ms")
        print(f"  final eval {1e3 * med([b['final_eval_seconds'] for b in whole]):.1f} ms")
        print(f"  readback   {1e3 * med([b['readback_seconds'] for b in whole]):.1f} ms")
    if phased:
        p = [b["begin_seconds"] for b in phased]
        print(f"phased begin():   median {1e3 * med(p):.1f} ms  "
              f"range {spread(p)} ms  n={len(p)}")
        if whole:
            obs = med(p) - med(w)
            print(f"observer cost of phasing: {1e3 * obs:+.1f} ms "
                  f"({100 * obs / med(w):+.1f}%)")

    census = next((b for b in blocks if b.get("kind") == "boundary_census"), None)
    per_phase = (census or {}).get("per_phase", {})

    section("Rung 1 -- per-phase decomposition with boundary counts")
    print("| phase | median ms | % of phased | dispatches | cmd buffers | "
          "distinct kernels |")
    print("|---|---:|---:|---:|---:|---:|")
    total_phase = 0.0
    rows = []
    for name in PHASES:
        samples = [b["phases"][name] for b in phased if name in (b.get("phases") or {})]
        if not samples:
            continue
        ms = 1e3 * med(samples)
        total_phase += ms
        c = per_phase.get(name, {})
        rows.append((name, ms, c))
    for name, ms, c in rows:
        share = 100 * ms / med(p) / 1e3 if phased else float("nan")
        print(f"| `{name}` | {ms:.1f} | {share:.1f}% | "
              f"{c.get('dispatches', '-')} | "
              f"{c.get('command_buffer_commits', '-')} | "
              f"{len(c.get('kernels', {})) or '-'} |")
    if phased:
        remainder = 1e3 * med(p) - total_phase
        print(f"| **unattributed remainder** | **{remainder:.1f}** | "
              f"**{100 * remainder / (1e3 * med(p)):.1f}%** | | | |")

        section("H-221 -- is the remainder a fixed per-boundary tax?")
        totals = (census or {}).get("totals", {})
        d = totals.get("dispatches")
        cb = totals.get("command_buffer_commits")
        forced = 23  # 22 prefill-ladder asyncEvals + the terminating eval
        print(f"remainder = {remainder:.2f} ms")
        for label, count in (("dispatch", d), ("command buffer", cb),
                             ("forced evaluation point", forced)):
            if count:
                per = remainder / count
                verdict = ("consistent with H-221"
                           if 0.5 * H221_MS_PER_BOUNDARY <= per
                           <= 2.0 * H221_MS_PER_BOUNDARY else "NOT H-221")
                print(f"  per {label:<24} ({count:>6}): {per:8.4f} ms  "
                      f"[{verdict}]")
        print(f"  H-221 reference: {H221_MS_PER_BOUNDARY} ms per removed head "
              "dispatch at decode width")

    section("Rung 1 -- positive control")
    stalls = [b for b in begins if b.get("stall_phase")]
    if not stalls:
        print("NO POSITIVE CONTROL RAN -- attribution above is unverified.")
    for target in sorted({b["stall_phase"] for b in stalls}):
        arm = [b for b in stalls if b["stall_phase"] == target]
        injected = arm[0].get("stall_millis", 0)
        print(f"\ninjected {injected} ms into `{target}`:")
        ok = True
        for name in PHASES:
            base = [b["phases"][name] for b in phased
                    if name in (b.get("phases") or {})]
            hit = [b["phases"][name] for b in arm if name in (b.get("phases") or {})]
            if not base or not hit:
                continue
            delta = 1e3 * (med(hit) - med(base))
            flag = ""
            if name == target:
                flag = " <-- target"
                if abs(delta - injected) > 0.5 * injected:
                    ok = False
                    flag += "  MISATTRIBUTED"
            elif abs(delta) > 0.5 * injected:
                ok = False
                flag = "  LEAKED"
            print(f"  {name:<26} {delta:+8.2f} ms{flag}")
        print(f"  control verdict: {'PASS' if ok else 'FAIL'}")

    section("H-221 at seed width -- prefill ladder discontinuity")
    ladder = [b for b in blocks if b.get("kind") == "ladder_step"]
    if not ladder:
        print("no ladder sweep in this session")
    else:
        by_w: dict[int, list[float]] = {}
        for b in ladder:
            by_w.setdefault(b["seed_length"], []).append(b["begin_seconds"])
        widths = sorted(by_w)
        m = {w: 1e3 * med(by_w[w]) for w in widths}
        print("The prefill ladder arms at exactly `dim(1) >= 512`. Below it no "
              "evaluation point is forced; at and above it 22 are.\n")
        print("| width | forced eval points | n | median begin ms | us/token |")
        print("|---:|---:|---:|---:|---:|")
        for w in widths:
            print(f"| {w} | {22 if w >= 512 else 0} | {len(by_w[w])} | "
                  f"{m[w]:.1f} | {1e3 * m[w] / w:.1f} |")
        off = [w for w in widths if w < 512]
        on = [w for w in widths if w >= 512]
        if len(off) >= 2 and on:
            n = len(off)
            sx, sy = sum(off), sum(m[w] for w in off)
            sxx = sum(w * w for w in off)
            sxy = sum(w * m[w] for w in off)
            denom = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / denom
            icept = (sy - slope * sx) / n
            print(f"\nladder-off fit: begin_ms = {slope:.4f} * width "
                  f"+ {icept:.2f}  (from widths {off})")
            for w in on:
                resid = m[w] - (slope * w + icept)
                print(f"  width {w}: residual {resid:+.2f} ms "
                      f"= {resid / 22:+.4f} ms per forced evaluation point")
            print("\nA negative residual means the ladder pays for itself: "
                  "overlapping host graph build with GPU execution wins more "
                  "than the boundaries cost.")

    section("Rung 2 -- in-situ family tax at M=512")
    pairs: dict[str, dict[str, list[float]]] = {}
    for b in begins:
        name = b.get("pair_arm")
        if not name:
            continue
        pairs.setdefault(name, {}).setdefault(b["arm"], []).append(b["begin_seconds"])
    null_tax = None
    if "null" in pairs:
        arm = pairs["null"]
        if "null" in arm and "baseline" in arm:
            null_tax = med(arm["null"]) - med(arm["baseline"])
            print(f"null control (same wrapper, full width): "
                  f"{1e3 * null_tax:+.2f} ms -- this is pure wrapper overhead "
                  "and is subtracted from every family below.\n")
    print("| family | tax ms | % of begin | flop share |")
    print("|---|---:|---:|---:|")
    for name, arm in pairs.items():
        if name == "null" or "baseline" not in arm or name not in arm:
            continue
        tax = med(arm[name]) - med(arm["baseline"]) - (null_tax or 0.0)
        share = 100 * tax / med(w) if whole else float("nan")
        print(f"| `{name}` | {-1e3 * tax:.1f} | {-share:.1f}% | |")
    print("\n`tax` is time REMOVED by pinning that family to one row, so a "
          "positive number is the family's in-situ cost at M=512.")

    section("Rung 2 -- isolated roofline")
    iso = [b for b in blocks if b.get("kind") == "isolated_quantized_matmul"]
    print("| family | M | K | N | layers | ms/call | TFLOP/s | GB/s | "
          "modelled ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    modelled_total = 0.0
    for b in sorted(iso, key=lambda x: -x.get("modelled_prefill_seconds", 0)):
        ms = 1e3 * b["seconds_median"]
        mod = 1e3 * b["modelled_prefill_seconds"]
        if b["m"] >= 512:
            modelled_total += mod
        print(f"| `{b['family']}` | {b['m']} | {b['k']} | {b['n']} | "
              f"{b['layers']} | {ms:.3f} | {b['tflop_per_second']:.2f} | "
              f"{b['gb_per_second']:.1f} | {mod:.1f} |")
    sdpa = next((b for b in blocks if b.get("kind") == "isolated_sdpa"), None)
    if sdpa:
        print(f"| `sdpa` | | | | {sdpa.get('layers')} | "
              f"{1e3 * sdpa['seconds_median']:.3f} | | | "
              f"{1e3 * sdpa.get('modelled_prefill_seconds', 0):.1f} |")

    section("Stop rule")
    if whole and modelled_total:
        begin_ms = 1e3 * med(w)
        gemm_share = 100 * modelled_total / begin_ms
        print(f"end-to-end begin(): {begin_ms:.1f} ms")
        print(f"modelled GEMM sum:  {modelled_total:.1f} ms")
        print(f"GEMM share:         {gemm_share:.1f}%")
        peak = GEMM_TFLOP / (modelled_total / 1e3) / 1e12 * 1e12
        print(f"implied prefill rate: {GEMM_TFLOP / (modelled_total / 1e3):.2f} "
              "TFLOP/s over the modelled GEMM time")
        _ = peak
        if gemm_share >= 90:
            print("\n>= 90%: LOCAL prefill is GEMM-bound and closed on this "
                  "host. Report the bound and stop.")
        else:
            print(f"\n< 90%: {100 - gemm_share:.1f}% is not GEMM. Name the "
                  "lever and make exactly one change.")

        section("Local -> ranked transfer band")
        other_ms = begin_ms - modelled_total
        print("The stop rule is evaluated on the local number above. The "
              "ranked consequence is a different quantity, because NAX does "
              "not accelerate every part equally.\n")
        print("| g (GEMM speedup) | n (non-GEMM speedup) | ranked non-GEMM share |")
        print("|---:|---:|---:|")
        for g, n in ((7.62, 1.0), (7.62, 1.5), (7.62, 3.0), (7.62, 7.62)):
            r = (other_ms / n) / (modelled_total / g + other_ms / n)
            print(f"| {g} | {n} | {100 * r:.1f}% |")
        print("\ng = 7.62 is the measured local-to-ranked ratio of the whole "
              "seed leg. Only the n = g row reproduces the local share; every "
              "other row opens prefill wider on M5 than it looks here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
