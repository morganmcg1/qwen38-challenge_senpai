#!/usr/bin/env python3
"""Read one E101 ABBA block and report drift-corrected, contamination-filtered
contrasts.

    usage: research/e101_abba_analyse.py [LABEL]

LABEL selects the block: `r5` reads the production legs `research/out/e101r5p*`
and `r5s` reads the sync-head legs `research/out/e101r5sp*`.

Every leg contributes `mtp_seconds_per_token` from its own score.json, the
per-round trace counters, and the identity and fidelity fields the assignment
requires. Base legs sit at the odd positions and kernel legs at the even ones,
so an ordinary least-squares line through the base legs gives the session drift
and its residual scatter gives the session null.

CONTAMINATION FILTER. Rung 4 label r4 produced three legs whose host-side trace
fields all inflated three to four times together while the GPU-side fields did
not, and `host_thread_cpu_ns` rose with them. The worker thread burned that CPU
itself, so this is host-side scheduling, not GPU behaviour, and it struck an
`off` leg as well as two `on` legs.

The stall is a property of single rounds, so the filter runs at round level. A
round is stalled when its `host_thread_cpu_ns` exceeds `ROUND_CPU_RATIO` times
the median over the pooled rounds of the whole block. That threshold is one
absolute number for both arms, and the arms differ in host CPU by about one
percent against a stall factor of three, so the filter cannot favour an arm.
Every leg reports the fraction of rounds it kept. A leg that keeps less than
`MIN_CLEAN_FRACTION` is excluded from the `mtp_seconds_per_token` contrast,
which is a whole-leg score that no round filter can repair.

PER-ROUND STATISTIC. Each leg reduces its surviving rounds with the median,
which the source's own E86 comparison also uses. Round 1 is dropped outright
because it pays the cold cost.
"""

import glob
import json
import os
import re
import statistics
import sys

ROUND_CPU_RATIO = 1.25
MIN_CLEAN_FRACTION = 0.75

# f1 ranked pricing table: (prompt, mean drafts per round, mean round us).
# A latency-class saving of `u` us per draft removes `u * drafts` us from each
# ranked round, so the median-pair gain is the median over prompts of
# `u * drafts / round`. Never multiply that by an amplification factor as well:
# dividing by the ranked round IS the amplification.
RANKED_ROUNDS = (("beagle", 4.3818, 57502.0), ("essays", 5.087, 59723.0))

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
    "e101_block",
    "MLX_E101_ROW_TOP32",
    "worker_sha256",
    "post_run_worker_sha256",
    "gpu_temp_entry_c",
    "gpu_temp_exit_c",
    "cool_gate_passed_real_gate",
    "gate_qualified_for_timing",
    "base_sha",
    "sandbox",
    "sync_head",
    "exit",
    "trace_rounds",
)
# Counters reported for both arms. `d_submit2_us` is the head-chain GPU drain
# under --sync-head and is meaningless without it; `d_chain_us` is host graph
# construction in every mode.
COUNTERS = (
    "round_us",
    "draft_build_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
    "host_thread_cpu_ns",
    "d",
    "acc",
)


def price_us_per_draft(us):
    """Median-pair percent gain the f1 table predicts for `us` per draft."""
    gains = sorted(us * drafts / rnd * 100 for _, drafts, rnd in RANKED_ROUNDS)
    mid = len(gains) // 2
    if len(gains) % 2:
        return gains[mid]
    return (gains[mid - 1] + gains[mid]) / 2


def read_meta(path):
    meta = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                meta[key] = value
    return meta


def read_rounds(path):
    """Every traced round after the cold first one, as integer field maps."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as handle:
        for line in handle:
            if not line.startswith("mtp-trace: round="):
                continue
            fields = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
            if fields.get("round", 0) <= 1:
                continue
            match = re.search(r"\bsel_env=(\S+)", line)
            fields["sel_env"] = match.group(1) if match else ""
            rows.append(fields)
    return rows


def reduce_rounds(rows):
    """Mean of every counter over the rounds handed in.

    The mean, not the median, because the per-draft price divides a per-round
    delta by drafts per round and the ranked f1 table is built from mean rounds
    and mean drafts. A median round carries the modal draft count instead, so
    mixing a median round with a mean draft count would price two different
    populations against each other. The round stall filter already removed the
    outliers a median would have been protecting against.
    """
    out = {}
    if not rows:
        return out
    for key in COUNTERS:
        values = [row[key] for row in rows if key in row]
        if values:
            out[key] = statistics.mean(values)
    for key in ("sel_fused", "sel_argpart"):
        values = [row[key] for row in rows if key in row]
        if values:
            out[key] = max(values)
    sources = {row["sel_env"] for row in rows if row.get("sel_env")}
    out["sel_env"] = "/".join(sorted(sources)) if sources else ""
    return out


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope, my - slope * mx


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def load(label):
    legs = []
    pattern = re.compile(rf"^e101{re.escape(label)}p\d+[bk]$")
    for out in sorted(glob.glob(f"research/out/e101{label}p*")):
        if not pattern.match(os.path.basename(out)):
            continue
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
        leg["rounds"] = read_rounds(os.path.join(out, "trace.txt"))
        leg["position"] = int(meta["e101_position"])
        legs.append(leg)
    return sorted(legs, key=lambda leg: leg["position"])


def classify(legs):
    """Drop stalled rounds against one pooled block threshold, then reduce."""
    pooled = [row["host_thread_cpu_ns"] for leg in legs for row in leg["rounds"]
              if "host_thread_cpu_ns" in row]
    reference = statistics.median(pooled) if pooled else 0
    threshold = reference * ROUND_CPU_RATIO
    for leg in legs:
        rows = leg["rounds"]
        kept = [row for row in rows
                if row.get("host_thread_cpu_ns", 0) <= threshold] if reference \
            else rows
        leg["kept"] = len(kept)
        leg["total"] = len(rows)
        leg["kept_fraction"] = len(kept) / len(rows) if rows else 0.0
        leg["clean"] = leg["kept_fraction"] >= MIN_CLEAN_FRACTION
        leg["trace"] = reduce_rounds(kept or rows)
        leg["trace_all"] = reduce_rounds(rows)
    return reference, threshold


def counter_contrast(legs, key, drafts):
    off = [leg["trace"][key] for leg in legs
           if leg["e101_arm"] == "off" and key in leg["trace"]]
    on = [leg["trace"][key] for leg in legs
          if leg["e101_arm"] == "on" and key in leg["trace"]]
    if not off or not on:
        return None
    delta = statistics.mean(on) - statistics.mean(off)
    return {
        "off": statistics.mean(off), "on": statistics.mean(on),
        "off_sd": sd(off), "on_sd": sd(on),
        "delta": delta, "per_draft": delta / drafts if drafts else float("nan"),
        "n_off": len(off), "n_on": len(on),
    }


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "r5"
    legs = load(label)
    if not legs:
        print(f"no legs found for label {label}")
        return 1
    reference, threshold = classify(legs)
    sync = {leg["sync_head"] for leg in legs}

    print(f"block e101{label}  sync_head={'/'.join(sorted(str(s) for s in sync))}"
          f"  pooled round cpu median={reference / 1e6:.2f} M ns"
          f"  stall threshold={threshold / 1e6:.2f} M ns")
    print()
    print(f"{'tag':14s} {'arm':4s} {'pos':>3s} {'kept':>9s} {'cpu Mns':>8s} "
          f"{'mtp_s/tok':>11s} {'round_us':>9s} {'dbuild':>8s} "
          f"{'d_chain':>8s} {'dsub2':>8s} {'vbuild':>8s} {'eval':>8s} "
          f"{'d':>4s} {'sel_env':>8s} {'fus':>5s} {'arg':>4s} {'inC':>6s} "
          f"{'outC':>6s}")
    for leg in legs:
        t = leg["trace"]
        entry = float(leg["gpu_temp_entry_c"] or "nan")
        exit_c = float(leg["gpu_temp_exit_c"] or "nan")
        print(
            f"{leg['tag']:14s} {leg['e101_arm']:4s} {leg['position']:3d} "
            f"{leg['kept']:4d}/{leg['total']:<4d} "
            f"{t.get('host_thread_cpu_ns', 0) / 1e6:8.2f} "
            f"{leg['mtp_seconds_per_token']:11.9f} "
            f"{t.get('round_us', 0):9.0f} {t.get('draft_build_us', 0):8.0f} "
            f"{t.get('d_chain_us', 0):8.0f} {t.get('d_submit2_us', 0):8.0f} "
            f"{t.get('verify_build_us', 0):8.0f} {t.get('eval_wall_us', 0):8.0f} "
            f"{t.get('d', 0):4.1f} {t.get('sel_env', '') or '-':>8s} "
            f"{t.get('sel_fused', 0):5d} {t.get('sel_argpart', 0):4d} "
            f"{entry:6.1f} {exit_c:6.1f}"
        )

    clean = [leg for leg in legs if leg["clean"]]
    dropped = [f"{leg['tag']}({leg['kept_fraction']:.0%})"
               for leg in legs if not leg["clean"]]
    print()
    print(f"round stall filter    keeps rounds with host_thread_cpu_ns <= "
          f"{ROUND_CPU_RATIO} x pooled median")
    print(f"leg score filter      needs kept_fraction >= {MIN_CLEAN_FRACTION:.0%}; "
          f"drops {dropped or 'nothing'}")

    for name, subset in (("ALL LEGS", legs), ("CLEAN LEGS", clean)):
        base = [leg for leg in subset if leg["e101_arm"] == "off"]
        kern = [leg for leg in subset if leg["e101_arm"] == "on"]
        print()
        print(f"--- {name}: n_off={len(base)} n_on={len(kern)} ---")
        if not base or not kern:
            print("  one arm is empty; no contrast")
            continue
        by = [leg["mtp_seconds_per_token"] for leg in base]
        ky = [leg["mtp_seconds_per_token"] for leg in kern]
        bx = [leg["position"] for leg in base]
        kx = [leg["position"] for leg in kern]
        slope, intercept = fit(bx, by)
        residuals = [y - (slope * x + intercept) for x, y in zip(bx, by)]
        base_mean = statistics.mean(by)
        kern_mean = statistics.mean(ky)
        contrast = kern_mean - statistics.mean(slope * x + intercept for x in kx)
        null = sd(residuals) / base_mean * 100 if len(residuals) > 1 else float("nan")
        print(f"  mtp_s/tok      off={base_mean:.9f} on={kern_mean:.9f}")
        print(f"  raw contrast   {(kern_mean - base_mean) / base_mean * 100:+.4f} %")
        print(f"  drift slope    {slope:+.3e} s/token/leg "
              f"({slope / base_mean * 100:+.4f} %/leg)")
        print(f"  base residuals {['%+.4f%%' % (r / base_mean * 100) for r in residuals]}")
        print(f"  session null   {null:.4f} % (1 sd of the base residuals)")
        print(f"  drift-corr.    {contrast / base_mean * 100:+.4f} %")

        drafts = statistics.mean(
            [leg["trace"]["d"] for leg in subset if leg["trace"].get("d")] or [0])
        print(f"  drafts/round   {drafts:.4f} (mean over the kept rounds)")
        print(f"  {'counter':16s} {'off':>10s} {'on':>10s} {'sd_off':>8s} "
              f"{'sd_on':>8s} {'delta/rnd':>10s} {'delta/draft':>12s} "
              f"{'f1 % pair':>10s}")
        for key in ("round_us", "draft_build_us", "d_head1_us", "d_submit1_us",
                    "d_chain_us", "d_submit2_us", "verify_build_us",
                    "eval_wall_us"):
            row = counter_contrast(subset, key, drafts)
            if row is None:
                continue
            print(f"  {key:16s} {row['off']:10.1f} {row['on']:10.1f} "
                  f"{row['off_sd']:8.1f} {row['on_sd']:8.1f} "
                  f"{row['delta']:+10.1f} {row['per_draft']:+12.2f} "
                  f"{price_us_per_draft(-row['per_draft']):+10.4f}")

    print()
    print("fidelity invariants (must be identical across every leg)")
    for key in ("effective_mean_draft_len", "accepted_draft_rate",
                "all_tokens_matched", "residual_divergence_count",
                "head_provenance_sha256", "decode_tokens", "mtp_depth",
                "worker_sha256", "base_sha", "sandbox", "sync_head",
                "cool_gate_passed_real_gate", "gate_qualified_for_timing"):
        values = sorted({str(leg[key]) for leg in legs})
        flag = "OK " if len(values) == 1 else "MOVED"
        print(f"  {flag} {key}: {values if len(values) > 1 else values[0]}")

    print()
    print("arm witness (the selection path each leg actually executed)")
    for leg in legs:
        t = leg["trace"]
        source = t.get("sel_env", "")
        if not source:
            print(f"  n/a {leg['tag']:14s} worker predates the sel_* witness")
            continue
        expected_fused = source != "0"
        took_fused = t.get("sel_fused", 0) > 0
        took_arg = t.get("sel_argpart", 0) > 0
        ok = (took_fused == expected_fused and took_arg != expected_fused
              and expected_fused == (leg["e101_arm"] == "on"
                                     or source == "unset"))
        print(f"  {'OK ' if ok else 'WRONG'} {leg['tag']:14s} "
              f"arm={leg['e101_arm']:3s} sel_env={source:>6s} "
              f"fused={t.get('sel_fused', 0):6d} "
              f"argpart={t.get('sel_argpart', 0):6d}")

    print()
    print("f1 ranked price of a latency-class saving (harness=ranked model)")
    for us in (10, 15, 22.5, 40, 42.6, 60, 89.38, 104.21):
        print(f"  {us:8.2f} us/draft -> {price_us_per_draft(us):+.4f} % median pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
