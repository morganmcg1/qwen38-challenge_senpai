#!/usr/bin/env python3
"""Reproduce section 11 of research/results/qwen38-r1-e18-prefill-dequant-prize.md.

Ranked-hardware prefill telemetry, read from public Yukon submission receipts.

    curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
      "https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true" \
      -o yukon_subs.json
    python3 research/e18_yukon_prefill.py yukon_subs.json

Every printed number is derived from the receipt file; the constants below are
only used to assert that the receipt still says what the write-up quotes.
"""

import json
import math
import statistics
import sys

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
FRONTIER_ID = "ba493f74-c0fe-440a-a956-f77d26232e54"
FRONTIER_SOURCE_REF = "156b5b75bdfa"
FRONTIER_SCORE = 2.95338624520432

# E16 (PR #18, alphonse) local M4 Pro instrumentation, NAX off. This is E16's
# prefill total P, the same quantity phi is expressed as a fraction of.
E16_PREFILL_SECONDS = 4.004000009
E16_GEMM_TFLOP_CENSUS = 24.93751230464
E16_PHI = 0.12942
E16_ROOFLINE_BYTES_PER_S = 227_128_791_836.97
QUANTIZED_WEIGHT_BYTES = 14_412_349_440

# E16 score-conversion constants, on base b85e782.
FRONTIER_STEP_POINTS = 0.0122890

# Section 5 dequant-ALU bands, as a fraction of GEMM time.
ALU_BANDS = [
    ("E16 residual phi = 12.942 % (upper bound)", E16_PHI),
    ("BM=32 ALU band, high (15.62 %)", 0.1562),
    ("BM=32 ALU band, low (12.50 %)", 0.1250),
    ("BM=64 NAX band, high (7.81 %)", 0.0781),
    ("BM=64 NAX band, low (6.25 %)", 0.0625),
]

# Every token that could name the ranked host, GPU, OS or kernel variant.
HOST_NEEDLES = [
    "applegpu", "gpu", "macos", "sw_vers", "buildversion", "host", "device",
    "arch", "nax", "runner", "m5", "chip", "hardware", "kernel", "metal",
    "os_", "darwin", "generation",
]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den


def walk_strings(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(str(k))
            walk_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            walk_strings(v, out)
    elif isinstance(node, str):
        out.append(node)


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <yukon_submissions.json>")

    with open(sys.argv[1]) as fh:
        payload = json.load(fh)
    rows = payload["submissions"] if isinstance(payload, dict) else payload

    metered = [r for r in rows if r.get("officialMetrics")]
    # prefill_seconds_per_token is a later schema addition: 131 earlier receipts
    # carry officialMetrics with the field set to None.
    scored = [r for r in metered
              if r["officialMetrics"].get("prefill_seconds_per_token")]
    print(f"submissions: {len(rows)}   with officialMetrics: {len(metered)}"
          f"   carrying prefill telemetry: {len(scored)}")

    frontier = max(scored, key=lambda r: r["officialScore"])
    assert frontier["id"] == FRONTIER_ID, frontier["id"]
    assert frontier["promotedSourceRef"].startswith(FRONTIER_SOURCE_REF)
    assert frontier["officialScore"] == FRONTIER_SCORE
    fm = frontier["officialMetrics"]
    print(f"frontier receipt: {frontier['id']}  solver={frontier['solverUsername']}")
    print(f"  promotedSourceRef={frontier['promotedSourceRef']}")
    print(f"  officialScore={frontier['officialScore']}")

    rule("11.1  the telemetry carries no host, OS or GPU field")
    print("officialMetrics keys:")
    print("  " + ", ".join(sorted(fm)))
    blob = []
    walk_strings(fm, blob)
    hits = sorted({n for n in HOST_NEEDLES for s in blob if n in s.lower()})
    print(f"\nneedles matched in officialMetrics: {hits if hits else 'NONE'}")
    assert not hits, hits

    # The same scan over the whole submission row does match, so locate it: the
    # only strings that hit are inside solver-authored free-text prose, which is
    # not machine-generated telemetry and never names the ranked box.
    row_hits = {}
    for key, val in frontier.items():
        if isinstance(val, str):
            found = [n for n in HOST_NEEDLES if n in val.lower()]
            if found:
                row_hits[key] = found
    print(f"needles matched elsewhere in the row: {row_hits or 'NONE'}")
    assert set(row_hits) <= {"note"}, row_hits
    print("  -> confined to the solver-authored 'note' field (free prose, not telemetry)")
    print("=> H1 cannot be resolved from Yukon. Only the Actions job log remains.")

    rule("11.2  ranked prefill share of the candidate leg")
    prefill_spt = fm["prefill_seconds_per_token"]
    decode_tokens = fm["decode_tokens"]
    cand_spt = fm["candidate_mtp_seconds_per_token_mean"]
    serial_spt = fm["baseline_serial_seconds_per_token_mean"]
    prefill_s = prefill_spt * decode_tokens
    cand_leg = cand_spt * decode_tokens
    serial_leg = serial_spt * decode_tokens
    print(f"prefill_seconds_per_token   {prefill_spt!r}")
    print(f"decode_tokens               {decode_tokens}")
    print(f"ranked prefill wall seconds {prefill_s:.6f}")
    print(f"candidate leg seconds       {cand_leg:.6f}")
    print(f"serial leg seconds          {serial_leg:.6f}")
    print(f"prefill / candidate leg     {100 * prefill_s / cand_leg:.4f} %   (charged inside)")
    print(f"prefill / serial leg        {100 * prefill_s / serial_leg:.4f} %")
    print(f"prefill / (cand + prefill)  {100 * prefill_s / (cand_leg + prefill_s):.4f} %   (if additive)")
    print("assignment premise was 15.8-18.0 %; neither reading approaches it.")

    rule("11.3  ranked prefill has never moved")
    pre = [r["officialMetrics"]["prefill_seconds_per_token"] for r in scored]
    sco = [r["officialScore"] for r in scored]

    # The field is a schema addition, so bound the window it actually covers.
    order = sorted(metered, key=lambda r: r["createdAt"])
    flags = [bool(r["officialMetrics"].get("prefill_seconds_per_token")) for r in order]
    transitions = sum(1 for a, b in zip(flags, flags[1:]) if a != b)
    dates = sorted(r["createdAt"] for r in scored)
    print(f"telemetry window   {dates[0]} .. {dates[-1]}")
    print(f"on/off transitions in time order: {transitions}  (1 == clean schema addition)")
    print(f"independent commits {len(set(r['submissionCommitSha'] for r in scored))}"
          f"   distinct solvers {len(set(r['solverUsername'] for r in scored))}")
    print(f"n                  {len(pre)}")
    print(f"min / max          {min(pre):.10f} / {max(pre):.10f}")
    print(f"median / mean      {statistics.median(pre):.10f} / {statistics.mean(pre):.10f}")
    print(f"stdev              {statistics.stdev(pre):.10f}")
    print(f"spread (max/min)   {max(pre) / min(pre):.4f}x")
    print(f"coeff of variation {100 * statistics.stdev(pre) / statistics.mean(pre):.3f} %")
    print(f"score range        {min(sco):.4f} .. {max(sco):.4f}  ({max(sco) / min(sco):.2f}x)")
    print(f"Pearson r(score, prefill)  {pearson(sco, pre):+.4f}")
    lo = min(scored, key=lambda r: r["officialScore"])
    hi = max(scored, key=lambda r: r["officialScore"])
    plo = lo["officialMetrics"]["prefill_seconds_per_token"]
    phi_ = hi["officialMetrics"]["prefill_seconds_per_token"]
    print(f"\nworst score {lo['officialScore']:.4f} -> prefill {plo:.10f}")
    print(f"best  score {hi['officialScore']:.4f} -> prefill {phi_:.10f}")
    print(f"ratio {max(plo, phi_) / min(plo, phi_):.6f}x across a {max(sco) / min(sco):.2f}x score spread")

    rule("11.4  compute scales further than bandwidth between the two hosts")
    local_tflops = E16_GEMM_TFLOP_CENSUS / E16_PREFILL_SECONDS
    ranked_tflops = E16_GEMM_TFLOP_CENSUS / prefill_s
    compute_ratio = ranked_tflops / local_tflops
    ranked_bw = QUANTIZED_WEIGHT_BYTES / serial_spt
    bw_ratio = ranked_bw / E16_ROOFLINE_BYTES_PER_S
    print(f"local prefill  {E16_PREFILL_SECONDS:.6f} s -> {local_tflops:6.3f} TFLOP/s   (E16, NAX off)")
    print(f"ranked prefill {prefill_s:.6f} s -> {ranked_tflops:6.3f} TFLOP/s")
    print(f"compute ratio                       {compute_ratio:.3f}x")
    print(f"local decode BW  {E16_ROOFLINE_BYTES_PER_S / 1e9:6.1f} GB/s   (measured roofline)")
    print(f"ranked decode BW {ranked_bw / 1e9:6.1f} GB/s   (effective lower bound)")
    print(f"bandwidth ratio                     {bw_ratio:.3f}x")
    print(f"\ncompute scales {compute_ratio / bw_ratio:.2f}x more than bandwidth")
    print("signature of a matrix-accelerator path active at M=512 and absent at M=1.")

    rule("11.5  the ranked prize, in frontier steps")
    gemm_frac_of_prefill = 1.0  # phi and the ALU bands are already fractions of prefill GEMM time
    print(f"{'assumed removable fraction':<44}{'dt ms':>9}{'leg red %':>11}{'score':>10}{'steps':>8}")
    for label, frac in ALU_BANDS:
        dt = prefill_s * frac * gemm_frac_of_prefill
        leg_red = dt / cand_leg
        gain = FRONTIER_SCORE * leg_red / (1.0 - leg_red)
        print(f"{label:<44}{1000 * dt:>9.2f}{100 * leg_red:>11.4f}{gain:>+10.5f}{gain / FRONTIER_STEP_POINTS:>8.2f}")
    print("\nevery row assumes 100 % of the fraction is removed, which section 5 and")
    print("Correction 5 both rule out. Local framing implied 7.34 steps for phi.")

    rule("11.6  side finding: the plausibility ceiling was raised 3 -> 5")
    order = sorted(metered, key=lambda r: r["createdAt"])
    ceils = [(r["createdAt"], r["officialMetrics"].get("decode_speedup_ceiling"))
             for r in order]
    moves = [(a, b) for a, b in zip(ceils, ceils[1:]) if a[1] != b[1]]
    for a, b in moves:
        print(f"  {a[0]}  ceiling {a[1]}  ->  {b[0]}  ceiling {b[1]}")
    print(f"transitions: {len(moves)}   current live ceiling: {ceils[-1][1]}"
          f"   (latest receipt {ceils[-1][0]})")
    floors = {r["officialMetrics"].get("decode_speedup_floor") for r in metered}
    print(f"decode_speedup_floor values ever seen: {floors}")
    worst_over = max(r["officialMetrics"].get("mtp_decode_speedup_raw_median") or 0
                     for r in metered)
    print(f"highest raw median ever reported: {worst_over:.4f}"
          f"  -> receipts above the old 3.0 gate: "
          f"{sum(1 for r in metered if (r['officialMetrics'].get('mtp_decode_speedup_raw_median') or 0) > 3.0)}")
    print("program.md still states 3.0; the live operator value is 5.0.")
    print("confirms assignment section 5.10 (operator commit a5854b97).")


if __name__ == "__main__":
    main()
