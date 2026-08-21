#!/usr/bin/env python3
"""Read one E101 rung 4 ABBA session and report the drift-corrected contrast.

    usage: research/e101_abba_analyse.py [LABEL]

Every leg contributes `mtp_seconds_per_token` from its own score.json plus the
identity and fidelity fields the assignment requires. The base legs sit at odd
positions and the kernel legs at even ones, so an ordinary least-squares line
through the base legs gives the session drift and its residual scatter gives
the session null. The contrast is the kernel mean against the base line
evaluated at the kernel positions, which removes monotone drift to first order.
"""

import glob
import json
import os
import re
import statistics
import sys

FIELDS = (
    "mtp_seconds_per_token",
    "serial_seconds_per_token",
    "mtp_decode_speedup",
    "effective_mean_draft_len",
    "accepted_draft_rate",
    "all_tokens_matched",
    "residual_divergence_count",
    "head_provenance_sha256",
    "decode_tokens",
    "mtp_depth",
)
META = (
    "e101_arm",
    "e101_position",
    "MLX_E101_ROW_TOP32",
    "worker_sha256",
    "post_run_worker_sha256",
    "gpu_temp_entry_c",
    "gpu_temp_exit_c",
    "cool_gate_passed_real_gate",
    "gate_qualified_for_timing",
    "base_sha",
    "sandbox",
    "exit",
    "trace_rounds",
)


def read_meta(path):
    meta = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                meta[key] = value
    return meta


def chain_us(trace_path):
    values = []
    if not os.path.exists(trace_path):
        return None
    with open(trace_path) as handle:
        for line in handle:
            match = re.search(r"\bd_chain_us=(\d+)", line)
            if match:
                values.append(int(match.group(1)))
    return statistics.mean(values) if values else None


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope, my - slope * mx


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "r4"
    legs = []
    for out in sorted(glob.glob(f"research/out/e101{label}p*")):
        meta_path = os.path.join(out, "meta.txt")
        score_path = os.path.join(out, "score.json")
        if not (os.path.exists(meta_path) and os.path.exists(score_path)):
            print(f"skip {out}: incomplete")
            continue
        meta = read_meta(meta_path)
        metrics = json.load(open(score_path))["metrics"]
        leg = {"tag": os.path.basename(out)}
        leg.update({k: meta.get(k) for k in META})
        leg.update({k: metrics.get(k) for k in FIELDS})
        leg["mean_d_chain_us"] = chain_us(os.path.join(out, "trace.txt"))
        leg["position"] = int(meta["e101_position"])
        legs.append(leg)

    if not legs:
        print("no legs found")
        return 1

    print(
        f"{'tag':14s} {'arm':4s} {'pos':>3s} {'mtp_s/tok':>11s} {'speedup':>8s} "
        f"{'d_len':>7s} {'acc':>7s} {'chain_us':>9s} {'inC':>5s} {'outC':>5s} "
        f"{'match':>5s} {'div':>4s}"
    )
    for leg in legs:
        print(
            f"{leg['tag']:14s} {leg['e101_arm']:4s} {leg['position']:3d} "
            f"{leg['mtp_seconds_per_token']:11.9f} "
            f"{leg['mtp_decode_speedup']:8.5f} "
            f"{leg['effective_mean_draft_len']:7.4f} "
            f"{leg['accepted_draft_rate']:7.5f} "
            f"{(leg['mean_d_chain_us'] or 0):9.1f} "
            f"{leg['gpu_temp_entry_c'] or '':>5s} {leg['gpu_temp_exit_c'] or '':>5s} "
            f"{str(leg['all_tokens_matched']):>5s} "
            f"{leg['residual_divergence_count']:4d}"
        )

    base = [leg for leg in legs if leg["e101_arm"] == "off"]
    kern = [leg for leg in legs if leg["e101_arm"] == "on"]
    bx = [leg["position"] for leg in base]
    by = [leg["mtp_seconds_per_token"] for leg in base]
    kx = [leg["position"] for leg in kern]
    ky = [leg["mtp_seconds_per_token"] for leg in kern]

    slope, intercept = fit(bx, by)
    residuals = [y - (slope * x + intercept) for x, y in zip(bx, by)]
    base_mean = statistics.mean(by)
    kern_mean = statistics.mean(ky)
    predicted = [slope * x + intercept for x in kx]
    contrast = kern_mean - statistics.mean(predicted)
    session_null = (
        statistics.pstdev(residuals) / base_mean * 100 if base_mean else float("nan")
    )

    print()
    print(f"base legs           n={len(base)} mean={base_mean:.9f} s/token")
    print(f"kernel legs         n={len(kern)} mean={kern_mean:.9f} s/token")
    print(f"raw contrast        {(kern_mean - base_mean) / base_mean * 100:+.4f} %")
    print(f"base drift slope    {slope:+.3e} s/token per leg "
          f"({slope / base_mean * 100:+.4f} %/leg)")
    print(f"base residuals      {['%+.4f%%' % (r / base_mean * 100) for r in residuals]}")
    print(f"session null (1 sd) {session_null:.4f} %")
    print(f"drift-corrected     {contrast / base_mean * 100:+.4f} %")
    print()
    print("fidelity invariants (must be identical across every leg)")
    for key in ("effective_mean_draft_len", "accepted_draft_rate",
                "all_tokens_matched", "residual_divergence_count",
                "head_provenance_sha256", "decode_tokens", "mtp_depth",
                "worker_sha256", "base_sha"):
        values = sorted({str(leg[key]) for leg in legs})
        flag = "OK " if len(values) == 1 else "MOVED"
        print(f"  {flag} {key}: {values if len(values) > 1 else values[0]}")

    chains = {
        arm: statistics.mean(
            [leg["mean_d_chain_us"] for leg in legs if leg["e101_arm"] == arm]
        )
        for arm in ("off", "on")
    }
    print()
    print(f"arm witness  mean d_chain_us off={chains['off']:.1f} "
          f"on={chains['on']:.1f} "
          f"delta={chains['on'] - chains['off']:+.1f} us/round")
    return 0


if __name__ == "__main__":
    sys.exit(main())
