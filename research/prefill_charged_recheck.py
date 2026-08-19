#!/usr/bin/env python3
"""Re-adjudicate ledger item 122: is the seed prefill inside the scored window?

Item 122 tested two hypotheses against the board:

    A   raw = serial_spt / mtp_spt
    B   raw = (pf + serial_spt) / (pf + mtp_spt)

A held to 3.9e-11 and B failed by 5-7 %, and the ledger concluded that prefill
is not scored. That inference does not follow. The trusted source says

    QwenRuntimeMTPDriver.swift:94    let started = Date()
    QwenRuntimeMTPDriver.swift:95    client.beginMTPDecode(...)   <- the prefill
    QwenRuntimeMTPDriver.swift:197   decodeSeconds = now - started
    QwenRuntimeMTP.swift:347-349     seedPrefillSeconds is "deliberately NOT
                                     subtracted from decodeSeconds"

so mtp_spt ALREADY contains the prefill. Hypothesis B therefore charges it
TWICE, and refuting B says nothing about whether it is charged ONCE. A is
consistent with both readings and cannot discriminate.

What the field does tell us is the exact leverage. With leg = 512 * spt and
K = 512 * prefill_spt,

    raw_p = 512 * serial_spt / (K + R_p * round_ms_p)

so a fractional prefill win converts to score at K/leg_p and a fractional
round-cost win converts at 1 - K/leg_p. This script reports both, and the
ranked-versus-local prefill host ratio, which is the transfer question that
actually decides whether the term is reachable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
API = "https://api.yukon.org/api"
WINDOW = 512

# research/e25r2-timed.json, M4 Pro, 16 legs of 512 decode tokens.
M4_PREFILL_SECONDS = 3.9938
# E1 depth-0 round cost transfer, research/results/qwen38-r1-e1-depth-cost-curve.md
M4_DEPTH0_ROUND_MS = 65.0094
RANKED_DEPTH0_ROUND_MS = 31.470


def fetch(path: str):
    tok = os.environ.get("YUKON_API_TOKEN", "")
    if not tok:
        raise SystemExit("YUKON_API_TOKEN is not set")
    req = urllib.request.Request(
        API + path, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


BENCHMARK = "5d1ee4d7-80bd-4555-b182-6505f26ef495"


def find_submission(prefix: str) -> dict:
    d = fetch("/benchmarks/%s/submissions?all=true" % BENCHMARK)
    rows = d.get("submissions", d if isinstance(d, list) else [])
    for r in rows:
        if str(r.get("id", "")).startswith(prefix):
            if r.get("officialMetrics", {}).get("per_prompt"):
                return r
            full = fetch("/submissions/" + r["id"])
            return full.get("submission", full)
    raise SystemExit("submission %s not found among %d rows" % (prefix, len(rows)))


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ca9251b8"
    sub = find_submission(prefix)
    per = sub["officialMetrics"]["per_prompt"]

    rows = []
    for p in sorted(per, key=lambda x: x["effective_mean_draft_len"]):
        name = PROMPT_NAMES[p["prompt_sha256"][:8]]
        s = p["serial_seconds_per_token_mean"]
        m = p["mtp_seconds_per_token_mean"]
        pf = p["prefill_seconds_per_token"]
        raw = p["raw_ratio_of_means"]
        leg = WINDOW * m
        k = WINDOW * pf
        rows.append({
            "prompt": name, "serial_spt": s, "mtp_spt": m, "prefill_spt": pf,
            "raw": raw, "leg_ms": leg * 1000.0, "k_ms": k * 1000.0,
            "share": k / leg, "serial_share": k / (WINDOW * s),
        })

    print("submission %s  score %s" % (prefix, sub.get("officialScore")))
    print()
    print("%-9s %11s %11s %10s %10s %9s %9s" % (
        "prompt", "leg_ms", "K_ms", "K/leg", "K/serleg", "round_x", "prefill_x"))
    for r in rows:
        print("%-9s %11.1f %11.2f %9.2f%% %9.2f%% %9.4f %9.4f" % (
            r["prompt"], r["leg_ms"], r["k_ms"], 100.0 * r["share"],
            100.0 * r["serial_share"], 1.0 - r["share"], r["share"]))

    ks = [r["k_ms"] for r in rows]
    print()
    print("K is a near-constant %.2f-%.2f ms (spread %.2f %%), as a fixed 512-token"
          % (min(ks), max(ks), 100.0 * (max(ks) - min(ks)) / min(ks)))
    print("seed prefill must be.")

    pair = [r for r in rows if r["prompt"] in ("beagle", "medicine")]
    mr = sum(1.0 - r["share"] for r in pair) / 2.0
    mp = sum(r["share"] for r in pair) / 2.0
    print()
    print("MEDIAN PAIR beagle+medicine (the 4th and 5th order statistics)")
    print("  a fractional ROUND-COST win converts to score at   x%.4f" % mr)
    print("  a fractional PREFILL win converts to score at      x%.4f" % mp)
    print("  halving the candidate prefill would be worth       %+.3f %% of score"
          % (100.0 * 0.5 * mp))

    ranked_k = sum(ks) / len(ks) / 1000.0
    host = M4_PREFILL_SECONDS / ranked_k
    rnd = M4_DEPTH0_ROUND_MS / RANKED_DEPTH0_ROUND_MS
    print()
    print("TRANSFER: the ranked host is %.2fx faster than this M4 Pro on the" % host)
    print("compute-bound prefill (%.4f s vs %.4f s) but only %.2fx faster on"
          % (ranked_k, M4_PREFILL_SECONDS, rnd))
    print("latency-bound depth-0 decode rounds (%.3f ms vs %.3f ms)."
          % (RANKED_DEPTH0_ROUND_MS, M4_DEPTH0_ROUND_MS))
    print("The %.2fx gap is the qmm_nax signature: quantized.cpp:473 takes the" % (host / rnd))
    print("neural-accelerator GEMM when is_nax_available(), which needs GPU gen")
    print(">= 17. This M4 Pro is applegpu_g16s, gen 16 (research/archprobe.m), so")
    print("it CANNOT execute the kernel family that runs the ranked prefill.")
    print()
    print("512-token prefill arithmetic: 2 * 27e9 * 512 = %.3e FLOPs" % (2 * 27e9 * 512))
    print("  ranked  %.4f s -> %6.2f TFLOP/s" % (ranked_k, 2 * 27e9 * 512 / ranked_k / 1e12))
    print("  M4 Pro  %.4f s -> %6.2f TFLOP/s  (E16 measured dense-bf16 ceiling 7.401)"
          % (M4_PREFILL_SECONDS, 2 * 27e9 * 512 / M4_PREFILL_SECONDS / 1e12))
    return 0


if __name__ == "__main__":
    sys.exit(main())
