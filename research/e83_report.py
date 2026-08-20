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


def head_report(payload: dict) -> None:
    ident = payload["identity"]
    steps = [b for b in payload["blocks"] if b.get("kind") == "pinned_head_step"]
    total = ident["head_total_bytes"]
    gemm = ident["head_gemm_bytes"]
    declared = ident["declared_manifest_bytes"]

    print(f"\n# E83 follow-on -- pinned proposal-head step "
          f"({payload['experiment']})")
    print(f"\nharness={payload['harness']}  "
          f"gate_qualified_for_timing={payload['gate_qualified_for_timing']}  "
          f"host={ident['host']}")
    print(f"head={ident['head_path']}  total_bytes={total:,}  "
          f"gemm_bytes={gemm:,}  kv_cache_length={ident['cache_length']}")

    section("Measured step time and byte rate")
    print("| rows | step ms | total bytes/ms | GEMM bytes/ms | GB/s (total) |")
    print("|---:|---:|---:|---:|---:|")
    for b in sorted(steps, key=lambda x: x["rows"]):
        ms = 1e3 * b["seconds_median"]
        print(f"| {b['rows']} | {ms:.3f} | {b['total_bytes_per_ms'] / 1e6:.1f} M | "
              f"{b['gemm_bytes_per_ms'] / 1e6:.1f} M | "
              f"{b['total_bytes'] / b['seconds_median'] / 1e9:.1f} |")
    print("\nThe step runs both pre-fc norms, the `fc` join, the single decoder "
          "layer's attention and MLP, and the output norm. It omits RoPE and "
          "the cache append, so the step time is a LOWER bound on a real head "
          "round and the byte rate is an UPPER bound.")

    section("Byte budget for a replacement head")
    m1 = next((b for b in steps if b["rows"] == 1), None)
    if m1:
        rate = m1["total_bytes_per_ms"]
        print(f"measured rate at 1 row: {rate / 1e6:.1f} MB per millisecond")
        print(f"pinned head  ({total:,} bytes): "
              f"{1e3 * m1['seconds_median']:.3f} ms measured")
        print(f"declared manifest head ({declared:,} bytes) at the same rate: "
              f"{declared / rate:.3f} ms predicted")
        print("\nThe declared head is not in this checkout, so its number is a "
              "PREDICTION from the measured rate, not a measurement. A head "
              "change is worth its cost only when the accepted tokens it adds "
              "outweigh the milliseconds its bytes cost at this rate.")
    widths = sorted(b["rows"] for b in steps)
    if len(widths) >= 2:
        lo = next(b for b in steps if b["rows"] == widths[0])
        hi = next(b for b in steps if b["rows"] == widths[-1])
        span = widths[-1] - widths[0]
        marginal = 1e3 * (hi["seconds_median"] - lo["seconds_median"]) / span
        print(f"\nmarginal cost of one extra row, {widths[0]} -> {widths[-1]}: "
              f"{marginal:+.4f} ms/row against a "
              f"{1e3 * lo['seconds_median']:.3f} ms fixed step. "
              "A near-flat slope means the head step is weight-traffic bound, "
              "so proposing more rows per step is close to free and shrinking "
              "the head is the only lever that moves its cost.")


def gate_report(payload: dict) -> None:
    """Rung 3 -- the two prefill-width fusion gates.

    Reports both an unpaired arm median and a per-rep paired delta against
    `gate_baseline`. The paired form is the one to trust: each rep contains
    every arm, so a monotone thermal trend cancels inside the rep and only the
    arm difference survives.
    """
    blocks = [b for b in payload["blocks"] if b.get("kind") == "gate_arm"]
    ident = payload["identity"]

    print(f"# E83 rung 3 -- prefill-width fusion gates -- {payload['experiment']}")
    print(f"\nharness={payload['harness']}  "
          f"gate_qualified_for_timing={payload['gate_qualified_for_timing']}  "
          f"device={ident.get('device')}  host={ident.get('host')}")
    print(f"seed_length={ident['seed_length']}  gate_reps={ident.get('gate_reps')}  "
          f"arms={ident.get('gate_arms')}")

    order = ["gate_baseline", "gate_g1", "gate_g2", "gate_g1g2"]
    names = [n for n in order if any(b["gate_arm"] == n for b in blocks)]
    by_arm = {n: [b for b in blocks if b["gate_arm"] == n] for n in names}

    section("Arm medians (unpaired)")
    print("| arm | in_proj bound | gate_up bound | n | median begin ms | "
          "range ms | saving vs baseline ms |")
    print("|---|---:|---:|---:|---:|---|---:|")
    base_med = med([b["begin_seconds"] for b in by_arm.get("gate_baseline", [])])
    for n in names:
        xs = [b["begin_seconds"] for b in by_arm[n]]
        m = med(xs)
        saving = "" if n == "gate_baseline" else f"{1e3 * (base_med - m):+.1f}"
        b0 = by_arm[n][0]
        print(f"| {n} | {b0['fused_in_proj_max_rows']} | "
              f"{b0['fused_gate_up_max_rows']} | {len(xs)} | {1e3 * m:.1f} | "
              f"{spread(xs)} | {saving} |")

    section("Paired per-rep delta vs gate_baseline (positive = arm is faster)")
    print("| arm | n pairs | median delta ms | min ms | max ms | "
          "reps arm faster |")
    print("|---|---:|---:|---:|---:|---:|")
    base_by_rep = {b["rep"]: b["begin_seconds"]
                   for b in by_arm.get("gate_baseline", [])}
    paired: dict[str, list[float]] = {}
    for n in names:
        if n == "gate_baseline":
            continue
        deltas = [1e3 * (base_by_rep[b["rep"]] - b["begin_seconds"])
                  for b in by_arm[n] if b["rep"] in base_by_rep]
        paired[n] = deltas
        if not deltas:
            continue
        wins = sum(1 for d in deltas if d > 0)
        print(f"| {n} | {len(deltas)} | {med(deltas):+.1f} | {min(deltas):+.1f} | "
              f"{max(deltas):+.1f} | {wins}/{len(deltas)} |")

    section("Exactness screen -- seed-boundary top-two evidence")
    primaries = {b["gate_arm"]: {b2["first_primary"] for b2 in by_arm[b["gate_arm"]]}
                 for b in blocks}
    all_primary = set()
    for s in primaries.values():
        all_primary |= s
    print(f"first_primary across every arm and rep: {sorted(all_primary)}")
    print("VERDICT: " + ("identical" if len(all_primary) == 1
                         else "DIVERGENT -- the fusion changed the argmax"))
    tops: dict[str, set] = {}
    for n in names:
        tops[n] = {tuple(b.get("top2_values") or []) for b in by_arm[n]}
    base_top = tops.get("gate_baseline", set())
    for n in names:
        same = tops[n] == base_top and len(tops[n]) == 1
        print(f"  {n}: distinct top2 tuples={len(tops[n])} "
              f"bit-identical-to-baseline={same}")
        if len(tops[n]) == 1 and n == "gate_baseline":
            print(f"    baseline top2 = {next(iter(tops[n]))}")
        elif tops[n] != base_top:
            print(f"    arm top2 = {sorted(tops[n])[:2]}")

    section("De-risk assertion -- no fused pack built inside a timed arm")
    befores = {b["pack_builds_before"] for b in blocks}
    afters = {b["pack_builds_after"] for b in blocks}
    print(f"pack_builds_before={sorted(befores)}  pack_builds_after={sorted(afters)}")
    ok = len(befores) == 1 and befores == afters and next(iter(befores)) > 0
    print("VERDICT: " + ("PASS -- packs resident from warm, none built while timed"
                         if ok else "FAIL -- a pack was built inside a timed arm"))

    section("Thermal record")
    for n in names:
        ent = [b["gpu_temp_entry_c"] for b in by_arm[n] if "gpu_temp_entry_c" in b]
        ext = [b["gpu_temp_exit_c"] for b in by_arm[n] if "gpu_temp_exit_c" in b]
        if ent:
            print(f"  {n}: entry median {med(ent):.2f} C "
                  f"({min(ent):.2f}-{max(ent):.2f})  "
                  f"exit median {med(ext):.2f} C" if ext else "")
    print(f"\ncool_gate_passed_real_gate={payload['cool_gate_passed_real_gate']}  "
          f"official_or_ranked_score={payload['official_or_ranked_score']}")

    section("Stop rule")
    combined = paired.get("gate_g1g2", [])
    if combined:
        c = med(combined)
        print(f"combined gate_g1g2 paired median saving = {c:+.1f} ms")
        if c < 40.7:
            print("VERDICT: NOT USEFUL -- below the 40.7 ms noise band. "
                  "Report the bound and close.")
        elif c < 60.0:
            print("VERDICT: INCONCLUSIVE -- between 40.7 and 60 ms. "
                  "Report for the advisor to decide.")
        else:
            print("VERDICT: LOCAL WINNER -- at or above 60 ms. Run the full "
                  "exactness and pre-submit chain.")


def prefill_report(payload: dict) -> None:
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
        base_s = [b["begin_seconds"] for b in whole]
        print(f"unphased begin(): median {1e3 * med(base_s):.1f} ms  "
              f"range {spread(base_s)} ms  n={len(base_s)}")
        print(f"  build      {1e3 * med([b['build_seconds'] for b in whole]):.1f} ms")
        print(f"  final eval {1e3 * med([b['final_eval_seconds'] for b in whole]):.1f} ms")
        print(f"  readback   {1e3 * med([b['readback_seconds'] for b in whole]):.1f} ms")
    if phased:
        p = [b["begin_seconds"] for b in phased]
        print(f"phased begin():   median {1e3 * med(p):.1f} ms  "
              f"range {spread(p)} ms  n={len(p)}")
        if whole:
            obs = med(p) - med(base_s)
            print(f"observer cost of phasing: {1e3 * obs:+.1f} ms "
                  f"({100 * obs / med(base_s):+.1f}%)")

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
    calib = next(
        (b for b in blocks if b.get("kind") == "stall_calibration"), None)
    if calib:
        print(f"stall calibration: `usleep({calib['nominal_millis']} ms)` on this "
              f"host delivers median {1e3 * calib['seconds_median']:.2f} ms "
              f"(min {1e3 * calib['seconds_min']:.2f}, "
              f"max {1e3 * calib['seconds_max']:.2f}, n={calib['samples']}).")
        print("The control is scored against the DELIVERED stall. Scoring it "
              "against the requested stall reports a false misattribution.")
    else:
        print("NO STALL CALIBRATION in this session: the target-phase delta is "
              "scored against the REQUESTED stall, which `usleep` overshoots.")
    for target in sorted({b["stall_phase"] for b in stalls}):
        arm = [b for b in stalls if b["stall_phase"] == target]
        requested = arm[0].get("stall_millis", 0)
        delivered = [1e3 * b["stall_actual_seconds"] for b in arm
                     if b.get("stall_actual_seconds") is not None]
        injected = med(delivered) if delivered else (
            1e3 * calib["seconds_median"] if calib else requested)
        source = "delivered in-band" if delivered else (
            "calibrated" if calib else "requested")
        print(f"\ninjected {requested} ms into `{target}` "
              f"({injected:.2f} ms {source}):")
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
        # The GEMM tile count in M steps at multiples of 32, so 496/504/511/512
        # all issue ceil(M/32) = 16 tiles while 520/528 issue 17. A width above
        # 512 therefore mixes the ladder step with a +6.25% tile step and
        # cannot be used to price boundaries.
        tiles = {w: -(-w // 32) for w in widths}
        base_tiles = tiles.get(512)
        contaminated = [w for w in widths if tiles[w] != base_tiles]
        if contaminated:
            print(f"\ntile counts (M/32, rounded up): "
                  f"{ {w: tiles[w] for w in widths} }")
            print(f"EXCLUDED from the fit -- different tile count, so the "
                  f"arithmetic step is not smooth across them: {contaminated}")
        off = [w for w in widths if w < 512 and tiles[w] == base_tiles]
        on = [w for w in widths if w >= 512 and tiles[w] == base_tiles]
        noise_band = 1e3 * (max(base_s) - min(base_s)) if whole else 0.0
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
                verdict = ("INSIDE the noise band -- not distinguishable "
                           "from zero" if abs(resid) <= noise_band
                           else "outside the noise band")
                print(f"  width {w}: residual {resid:+.2f} ms "
                      f"= {resid / 22:+.4f} ms per forced evaluation point "
                      f"[{verdict}]")
            print(f"\nbaseline noise band at width 512: {noise_band:.1f} ms")
            print("A negative residual would mean the ladder pays for itself. "
                  "H-221 predicts +0.35 ms per boundary, i.e. "
                  f"{22 * H221_MS_PER_BOUNDARY:.1f} ms over 22 boundaries -- "
                  "compare that against the noise band before believing "
                  "either sign.")

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
        share = 100 * tax / med(base_s) if whole else float("nan")
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
        # `gate_up_fused_unused` is the 5120->34816 pack the decode path builds
        # and prefill never calls: `Qwen35FusedMLP.callAsFunction` gates it on
        # `x.dim(-2) <= 16`. It is measured to price the road not taken, so it
        # must not enter the executed-path total.
        if b["m"] >= 512 and not any(t in b["family"] for t in ("unused", "probe")):
            modelled_total += mod
        print(f"| `{b['family']}` | {b['m']} | {b['k']} | {b['n']} | "
              f"{b['layers']} | {ms:.3f} | {b['tflop_per_second']:.2f} | "
              f"{b['gb_per_second']:.1f} | {mod:.1f} |")
    swiglu = next((b for b in blocks if b.get("kind") == "isolated_swiglu"), None)
    sdpa = next((b for b in blocks if b.get("kind") == "isolated_sdpa"), None)
    if sdpa:
        print(f"| `sdpa` | | | | {sdpa.get('layers')} | "
              f"{1e3 * sdpa['seconds_median']:.3f} | | | "
              f"{1e3 * sdpa.get('modelled_prefill_seconds', 0):.1f} |")

    if swiglu:
        section("Named lever -- the SwiGLU activation is not fused at seed width")
        u = 1e3 * swiglu["unfused_modelled_prefill_seconds"]
        f = 1e3 * swiglu["fused_modelled_prefill_seconds"]
        s = 1e3 * swiglu["saving_modelled_prefill_seconds"]
        print("`Qwen35FusedMLP.callAsFunction` gates its compiled fused form "
              "on `x.dim(-2) <= 16`, so decode gets one launch and the "
              "512-row seed gets `silu` then multiply with a materialized "
              "intermediate.\n")
        print(f"unfused, 64 layers: {u:.1f} ms")
        print(f"fused,   64 layers: {f:.1f} ms")
        print(f"available saving:   {s:.1f} ms")
        if begins:
            b0 = med([b["begin_seconds"] for b in begins
                      if b.get("arm") == "baseline" and not b["phased"]
                      and not b.get("stall_phase")])
            print(f"                    = {100 * s / (1e3 * b0):.2f}% of the "
                  "local seed prefill")

    ba = next((b for b in iso if b["family"] == "gdn.in_proj_ba_fused_probe"), None)
    b_cell = next((b for b in iso if b["family"] == "gdn.in_proj_b"), None)
    a_cell = next((b for b in iso if b["family"] == "gdn.in_proj_a"), None)
    if ba and b_cell and a_cell:
        section("Named lever -- the two 5120->48 GDN projections")
        pair = 1e3 * (b_cell["modelled_prefill_seconds"]
                      + a_cell["modelled_prefill_seconds"])
        fused_ba = 1e3 * ba["modelled_prefill_seconds"]
        print(f"`in_proj_b` + `in_proj_a` as shipped: {pair:.1f} ms "
              f"for {100 * 0.0242 / GEMM_TFLOP:.3f}% of the prefill FLOP")
        print(f"one 5120->96 projection instead:      {fused_ba:.1f} ms")
        print(f"available saving:                     {pair - fused_ba:.1f} ms")
        print("\nThese two cells are inside the GEMM total, so the >=90% stop "
              "rule still reads 'closed'. GEMM-bound is not the same as "
              "GEMM-optimal, and this is the largest single inefficiency the "
              "sweep found.")

    section("Stop rule")
    if whole and modelled_total:
        begin_ms = 1e3 * med(base_s)
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
        noise = 1e3 * (max(base_s) - min(base_s))
        print(f"non-GEMM residual: {other_ms:+.1f} ms")
        print(f"baseline noise band (max-min over {len(base_s)} unpinned reps): "
              f"{noise:.1f} ms\n")
        if other_ms <= noise:
            print("The residual is inside the noise band, so the honest "
                  "statement is a BOUND, not a value: non-GEMM work in the "
                  f"seed prefill costs at most {max(other_ms, noise):.0f} ms, "
                  f"i.e. at most {100 * max(other_ms, noise) / begin_ms:.1f}% "
                  "of the local leg. Every isolated GEMM is measured with its "
                  "own eval and no overlap, so the sum OVERSTATES in-situ GEMM "
                  "time; that is why the residual can land at or below zero.\n")
            other_ms = noise
        print("The stop rule is evaluated on the local number above. The "
              "ranked consequence is a different quantity, because NAX does "
              "not accelerate every part equally. Rows below use the BOUND.\n")
        print("| g (GEMM speedup) | n (non-GEMM speedup) | ranked non-GEMM share |")
        print("|---:|---:|---:|")
        for g, n in ((7.62, 1.0), (7.62, 1.5), (7.62, 3.0), (7.62, 7.62)):
            r = (other_ms / n) / (modelled_total / g + other_ms / n)
            print(f"| {g} | {n} | {100 * r:.1f}% |")
        print("\ng = 7.62 is the measured local-to-ranked ratio of the whole "
              "seed leg. Even the worst row (NAX helps non-GEMM not at all) "
              "leaves the ranked non-GEMM share small, because the local "
              "non-GEMM bound is itself small.")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        payload = json.load(open(path))
        kinds = {b.get("kind") for b in payload["blocks"]}
        if "pinned_head_step" in kinds:
            head_report(payload)
        elif "gate_arm" in kinds:
            gate_report(payload)
        else:
            prefill_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
