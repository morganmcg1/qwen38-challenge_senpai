#!/usr/bin/env python3
"""E212: the per-round verify cost law R_local(m) on the current composed
surface (cap-8 + staged (9,5) + selective-m6 single-pass).

    usage: research/e212_report.py SESSION [SESSION ...]
                                   [--out research/e212-report.json]

harness=local. Every leg runs with the per-round phase trace on and the cool
gate off, so no number here is a gate-qualified timing claim and none of them
is a candidate speed claim. Only the SHAPE of a phase against the verified
width m, and the size of a step between adjacent widths, are read.

Per width the script reports the median per-round wall of every traced phase,
a bootstrap CI of that median over rounds, and the spread between the two
counterbalanced legs of the same width. A width whose two legs disagree by
more than the FINDING 503 session noise floor is flagged rather than averaged
silently.

The step test is round-weighted with the SHIPPED schedule's own served-width
distribution, taken from the `ref` legs of the same session: removing the step
at boundary m can only pay on rounds the shipped schedule serves at width >= m.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out" / "e212"

# The first rounds of a leg pay cold-cache and first-shape costs that no
# steady-state round pays. E182 dropped the same two.
DROP_FIRST_ROUNDS = 2

# FINDING 503: session-to-session noise floor of a traced per-round mean.
NOISE_FLOOR_MS = 0.24

PHASES = [
    "round_us",
    "draft_build_us",
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
    "band_pre_us",
    "band_gdn_mixer_us",
    "band_gdn_mlp_us",
    "band_fa_mixer_us",
    "band_fa_mlp_us",
]
COUNTERS = ["acc", "band_fwd", "xs_fill", "xs_hit", "cfg_miss"]

# The layer-family device bands, emitted only by a band-sync arm.
BANDS = [
    "band_pre_us",
    "band_gdn_mixer_us",
    "band_gdn_mlp_us",
    "band_fa_mixer_us",
    "band_fa_mlp_us",
]

# Fused routed QMV cells of one decode round: (name, k, n, invocations).
# Same table as research/e206_width_map.py.
CELLS = [
    ("mlp.gate_up", 5120, 34816, 64),
    ("mlp.down", 17408, 5120, 64),
    ("gdn.in_proj", 5120, 16480, 48),
    ("gdn.out_proj", 6144, 5120, 48),
    ("fa.qkv", 5120, 14336, 16),
    ("fa.o_proj", 6144, 5120, 16),
    ("lm_head", 5120, 248320, 1),
]
# affine 4-bit, group 64: 0.5 B/element payload plus a bf16 scale and a bf16
# bias per 64 elements.
BYTES_PER_ELEMENT = 0.5 + 4.0 / 64.0

# CURRENT shipped plan, Vendor/.../Qwen35.swift: staged pairs with the E208
# (9,5) entry, and E195 selective single-pass at m = 6 for every decode cell
# except mlp.down.
STAGED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}
SINGLEPASS_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3}

# The dated law this census replaces: E182 unperturbed legs, harness=local,
# M4 Pro, cap-7 surface, before E195 / E193 / E208 (research/e206_width_map.py).
E182_ROUND_MS = {1: 65.297, 4: 77.311, 5: 90.486, 6: 126.081, 7: 137.362,
                 8: 145.774, 9: 185.826}


def cell_variant(m: int, cell: str) -> str:
    """`qwen35QMVVariant`, transcribed."""
    if m == 6 and cell != "mlp.down":
        return "singlepass"
    return "staged"


def cell_groups(m: int, cell: str) -> int:
    """Weight passes this cell pays at width m under the shipped plan."""
    if m <= 1:
        return 1
    ipg = (SINGLEPASS_IPG if cell_variant(m, cell) == "singlepass"
           else STAGED_IPG)[m]
    return (m + ipg - 1) // ipg


def weight_bytes(m: int) -> float:
    """Weight bytes one round streams through the fused QMV cells at width m."""
    total = 0.0
    for name, k, n, invocations in CELLS:
        total += (k * n * BYTES_PER_ELEMENT) * invocations * cell_groups(m, name)
    return total


def parse_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def parse_rounds(path: pathlib.Path) -> list[dict[str, float]]:
    """Steady-state rounds of the LAST traced session in the file.

    `--local-iterate` runs the pinned serial leg and then the MTP leg from the
    same build, and both append to this file. Each session opens with
    `mtp-trace: begin`, so the last session is the MTP leg.
    """
    sessions: list[list[dict[str, float]]] = []
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if line.startswith("mtp-trace: begin"):
            sessions.append([])
            continue
        if not line.startswith("mtp-trace: round=") or not sessions:
            continue
        kv = dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
        if "round" not in kv or float(kv["round"]) <= DROP_FIRST_ROUNDS:
            continue
        row = {"round": float(kv["round"]), "d": float(kv.get("d", -1))}
        for field in PHASES + COUNTERS:
            if field in kv:
                row[field] = float(kv[field])
        sessions[-1].append(row)
    return sessions[-1] if sessions else []


def boot_ci(values: list[float], reps: int = 2000, seed: int = 12212) -> tuple[float, float]:
    """Percentile bootstrap CI of the median, in the units given."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(0.025 * reps)]
    hi = medians[min(reps - 1, int(0.975 * reps))]
    return (lo, hi)


def read_leg(leg_dir: pathlib.Path) -> dict | None:
    meta = parse_meta(leg_dir / "meta.txt")
    if not meta:
        return None
    arm = meta.get("e212_arm", "?")
    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})
    rounds = parse_rounds(leg_dir / "trace.txt")

    if arm.startswith("w") or arm.startswith("b"):
        depth = int(arm[1:])
        kept = [r for r in rounds if int(r["d"]) == depth]
        width = depth + 1
    else:
        kept = rounds
        width = None

    record = {
        "leg": leg_dir.name,
        "session": meta.get("e212_session"),
        "arm": arm,
        "width_m": width,
        "band_sync": meta.get("band_sync") == "1",
        "leg_index": int(meta.get("e212_leg", 0)),
        "rounds_traced": len(rounds),
        "rounds_at_width": len(kept),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c") or None,
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c") or None,
        "worker_sha256": meta.get("worker_sha256"),
        "worker_digest_stable": meta.get("worker_digest_stable"),
        "base_sha": meta.get("base_sha"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "decode_tokens": metrics.get("decode_tokens"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
        "phases_ms": {},
        "phase_ci_ms": {},
    }
    for phase in PHASES:
        values = [r[phase] / 1000.0 for r in kept if phase in r]
        if not values:
            continue
        record["phases_ms"][phase] = {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "n": len(values),
        }
        lo, hi = boot_ci(values)
        record["phase_ci_ms"][phase] = [lo, hi]
    accs = [r["acc"] for r in kept if "acc" in r]
    if accs:
        record["accepted_mean"] = statistics.fmean(accs)
        record["tokens_per_round_mean"] = statistics.fmean(accs) + 1.0
    if width is None and rounds:
        histogram: dict[int, int] = {}
        for r in rounds:
            histogram[int(r["d"]) + 1] = histogram.get(int(r["d"]) + 1, 0) + 1
        record["served_width_histogram"] = {str(k): v
                                            for k, v in sorted(histogram.items())}
        record["served_rounds"] = len(rounds)
    return record


def aggregate(legs: list[dict]) -> dict:
    widths: dict[int, dict] = {}
    for leg in legs:
        if leg["width_m"] is None or leg["band_sync"]:
            continue
        widths.setdefault(leg["width_m"], {"legs": []})["legs"].append(leg)

    table = {}
    for m, bucket in sorted(widths.items()):
        entry: dict = {"legs": [leg["leg"] for leg in bucket["legs"]],
                       "n_legs": len(bucket["legs"])}
        for phase in PHASES:
            medians = [leg["phases_ms"][phase]["median"]
                       for leg in bucket["legs"] if phase in leg["phases_ms"]]
            if not medians:
                continue
            entry.setdefault("phase_ms", {})[phase] = {
                "value": statistics.fmean(medians),
                "legs": medians,
                "leg_spread": max(medians) - min(medians),
            }
        cis = [leg["phase_ci_ms"].get("round_us") for leg in bucket["legs"]
               if leg["phase_ci_ms"].get("round_us")]
        if cis:
            entry["round_ms_ci_within_leg"] = cis
        entry["rounds_at_width"] = [leg["rounds_at_width"] for leg in bucket["legs"]]
        entry["accepted_mean"] = statistics.fmean(
            [leg["accepted_mean"] for leg in bucket["legs"] if "accepted_mean" in leg])
        entry["tokens_per_round_mean"] = entry["accepted_mean"] + 1.0
        entry["mtp_seconds_per_token"] = statistics.fmean(
            [leg["mtp_seconds_per_token"] for leg in bucket["legs"]
             if leg["mtp_seconds_per_token"]])
        entry["weight_bytes_per_round"] = weight_bytes(m)
        entry["weight_passes"] = {name: cell_groups(m, name)
                                  for name, _, _, _ in CELLS}
        round_entry = entry.get("phase_ms", {}).get("round_us")
        if round_entry:
            entry["round_ms"] = round_entry["value"]
            entry["round_ms_leg_spread"] = round_entry["leg_spread"]
            entry["leg_spread_exceeds_noise_floor"] = (
                round_entry["leg_spread"] > NOISE_FLOOR_MS)
        table[m] = entry
    return table


def aggregate_band(legs: list[dict]) -> dict:
    """Per-width band medians from the band-sync arms.

    ATTRIBUTION ONLY. A band arm drains the device at every mixer and MLP
    boundary, so its round total is inflated and is never a round cost. Only
    the SHARE of a step across bands is read, and only after the bands are
    checked to account for the band-arm round they were measured in.
    """
    widths: dict[int, list[dict]] = {}
    for leg in legs:
        if leg["band_sync"] and leg["width_m"] is not None:
            widths.setdefault(leg["width_m"], []).append(leg)

    table = {}
    for m, bucket in sorted(widths.items()):
        entry: dict = {"legs": [leg["leg"] for leg in bucket],
                       "n_legs": len(bucket)}
        for phase in PHASES:
            medians = [leg["phases_ms"][phase]["median"]
                       for leg in bucket if phase in leg["phases_ms"]]
            if not medians:
                continue
            entry.setdefault("phase_ms", {})[phase] = {
                "value": statistics.fmean(medians),
                "legs": medians,
                "leg_spread": max(medians) - min(medians),
            }
        bands = {b: entry["phase_ms"][b]["value"]
                 for b in BANDS if b in entry.get("phase_ms", {})}
        entry["band_sum_ms"] = sum(bands.values())
        round_ms = entry.get("phase_ms", {}).get("round_us", {}).get("value")
        if round_ms:
            entry["band_arm_round_ms"] = round_ms
            entry["band_unaccounted_ms"] = round_ms - entry["band_sum_ms"]
        table[m] = entry
    return table


def band_steps(band_table: dict) -> list[dict]:
    """Split each adjacent band-arm step across the traced bands."""
    out = []
    ms = sorted(band_table)
    for lo, hi in zip(ms, ms[1:]):
        if hi != lo + 1:
            continue
        a, b = band_table[lo], band_table[hi]
        per_band = {}
        for band in BANDS:
            av = a.get("phase_ms", {}).get(band, {}).get("value")
            bv = b.get("phase_ms", {}).get(band, {}).get("value")
            if av is None or bv is None:
                continue
            per_band[band] = bv - av
        total = sum(per_band.values())
        out.append({
            "boundary": f"{lo}->{hi}",
            "band_arm_round_step_ms": (b.get("band_arm_round_ms", 0.0)
                                       - a.get("band_arm_round_ms", 0.0)),
            "band_step_total_ms": total,
            "band_step_ms": per_band,
            "band_share": {k: (v / total if total else 0.0)
                           for k, v in per_band.items()},
        })
    return out


def ipg_law(table: dict) -> dict:
    """Read R(m) as (weight passes) x (per-pass cost at that group width).

    Every width except m = 6 dispatches one staged `(m, IPG)` pair for all
    seven decode cells, so `(G, IPG)` is a scalar there and two widths that
    share an IPG differ only in pass count. m = 6 is excluded: the shipped
    plan is per cell there, so no single `(G, IPG)` describes it.
    """
    uniform = {}
    for m, entry in table.items():
        if m < 2 or m == 6 or "round_ms" not in entry:
            continue
        ipg = STAGED_IPG[m]
        uniform[m] = {"ipg": ipg, "groups": (m + ipg - 1) // ipg,
                      "round_ms": entry["round_ms"]}

    ladder = {str(v["ipg"]): v["round_ms"]
              for m, v in sorted(uniform.items()) if v["groups"] == 1}
    pairs = []
    for m_hi, hi in sorted(uniform.items()):
        if hi["groups"] != 2:
            continue
        for m_lo, lo in sorted(uniform.items()):
            if lo["groups"] == 1 and lo["ipg"] == hi["ipg"]:
                # R = c + G(f - c) at fixed IPG gives c from the pair.
                implied_c = 2.0 * lo["round_ms"] - hi["round_ms"]
                pairs.append({
                    "ipg": hi["ipg"],
                    "m_one_pass": m_lo, "m_two_pass": m_hi,
                    "round_ms_one_pass": lo["round_ms"],
                    "round_ms_two_pass": hi["round_ms"],
                    "ratio": hi["round_ms"] / lo["round_ms"],
                    "implied_fixed_overhead_ms": implied_c,
                })
    return {
        "excluded_widths": [6],
        "single_pass_ipg_ladder_ms": ladder,
        "ipg_ladder_steps_ms": {
            f"{a}->{b}": ladder[str(b)] - ladder[str(a)]
            for a, b in zip(sorted(int(k) for k in ladder),
                            sorted(int(k) for k in ladder)[1:])
        },
        "fixed_ipg_pass_doubling": pairs,
    }


def served_distribution(legs: list[dict]) -> dict:
    total: dict[int, int] = {}
    for leg in legs:
        for width, count in (leg.get("served_width_histogram") or {}).items():
            total[int(width)] = total.get(int(width), 0) + count
    n = sum(total.values())
    share = {m: c / n for m, c in sorted(total.items())} if n else {}
    tail = {}
    running = 0.0
    for m in sorted(share, reverse=True):
        running += share[m]
        tail[m] = running
    return {"counts": {str(k): v for k, v in sorted(total.items())},
            "share": {str(k): v for k, v in share.items()},
            "share_at_or_above": {str(k): v for k, v in sorted(tail.items())},
            "rounds": n}


def steps(table: dict, distribution: dict) -> list[dict]:
    out = []
    at_or_above = {int(k): v for k, v in
                   (distribution.get("share_at_or_above") or {}).items()}
    for m in sorted(table):
        if m - 1 not in table:
            continue
        lo = table[m - 1].get("round_ms")
        hi = table[m].get("round_ms")
        if lo is None or hi is None:
            continue
        raw = hi - lo
        weight = at_or_above.get(m, 0.0)
        out.append({
            "boundary": f"{m - 1}->{m}",
            "m": m,
            "raw_step_ms": raw,
            "shipped_share_at_or_above_m": weight,
            "round_weighted_step_ms": raw * weight,
            "weight_bytes_step": table[m]["weight_bytes_per_round"]
                                 - table[m - 1]["weight_bytes_per_round"],
            "leg_spread_lo": table[m - 1].get("round_ms_leg_spread"),
            "leg_spread_hi": table[m].get("round_ms_leg_spread"),
        })
    return out


def fit(table: dict) -> dict:
    """Smooth quadratic against the kernel-level weight-pass model."""
    ms = [m for m in sorted(table) if m >= 2 and "round_ms" in table[m]]
    if len(ms) < 4:
        return {}
    y = [table[m]["round_ms"] for m in ms]
    base_bytes = weight_bytes(2)

    def solve(columns: list[list[float]]) -> tuple[list[float], float]:
        k = len(columns)
        a = [[sum(columns[i][r] * columns[j][r] for r in range(len(y)))
              for j in range(k)] for i in range(k)]
        b = [sum(columns[i][r] * y[r] for r in range(len(y))) for i in range(k)]
        for i in range(k):
            pivot = max(range(i, k), key=lambda r: abs(a[r][i]))
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
            for r in range(i + 1, k):
                factor = a[r][i] / a[i][i]
                for c in range(i, k):
                    a[r][c] -= factor * a[i][c]
                b[r] -= factor * b[i]
        coefficients = [0.0] * k
        for i in reversed(range(k)):
            coefficients[i] = (b[i] - sum(a[i][c] * coefficients[c]
                                          for c in range(i + 1, k))) / a[i][i]
        residual = sum(
            (y[r] - sum(coefficients[i] * columns[i][r] for i in range(k))) ** 2
            for r in range(len(y)))
        return coefficients, residual

    def aicc(rss: float, k: int) -> float:
        n = len(y)
        value = n * math.log(max(rss, 1e-12) / n) + 2 * k
        if n - k - 1 > 0:
            value += (2 * k * (k + 1)) / (n - k - 1)
        return value

    ones = [1.0] * len(ms)
    linear = [float(m) for m in ms]
    quadratic = [float(m * m) for m in ms]
    passes = [weight_bytes(m) / base_bytes for m in ms]

    smooth_coefficients, smooth_rss = solve([ones, linear, quadratic])
    step_coefficients, step_rss = solve([ones, linear, passes])
    linear_coefficients, linear_rss = solve([ones, linear])

    return {
        "widths": ms,
        "round_ms": y,
        "smooth_quadratic": {"coefficients": smooth_coefficients,
                             "rss": smooth_rss, "aicc": aicc(smooth_rss, 4)},
        "weight_pass_step": {"coefficients": step_coefficients,
                             "rss": step_rss, "aicc": aicc(step_rss, 4)},
        "linear": {"coefficients": linear_coefficients,
                   "rss": linear_rss, "aicc": aicc(linear_rss, 3)},
        "normalised_weight_bytes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--out", default=str(ROOT / "e212-report.json"))
    args = parser.parse_args()

    legs: list[dict] = []
    for session in args.sessions:
        root = OUT_ROOT / session
        for leg_dir in sorted(root.glob("leg*")):
            record = read_leg(leg_dir)
            if record:
                legs.append(record)
    if not legs:
        print("e212_report: no legs found")
        return 1

    reference_legs = [leg for leg in legs if leg["arm"] == "ref"]
    distribution = served_distribution(reference_legs)
    table = aggregate(legs)
    band_legs = [leg for leg in legs if leg["band_sync"]]
    band_table = aggregate_band(legs)

    report = {
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_score": False,
        "sessions": args.sessions,
        "noise_floor_ms": NOISE_FLOOR_MS,
        "legs": legs,
        "shipped_served_width_distribution": distribution,
        "width_table": {str(k): v for k, v in table.items()},
        "steps": steps(table, distribution),
        "shape_fit": fit(table),
        "dated_table_e182_round_ms": E182_ROUND_MS,
        "band_arm_legs": [leg["leg"] for leg in band_legs],
        "band_table": {str(k): v for k, v in band_table.items()},
        "band_steps": band_steps(band_table),
        "ipg_law": ipg_law(table),
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    print(f"e212: {len(legs)} legs, {len(table)} widths -> {args.out}")
    print(f"{'m':>3} {'round_ms':>9} {'spread':>7} {'rounds':>7} "
          f"{'acc':>5} {'e182_ms':>8} {'delta':>8}")
    for m, entry in sorted(table.items()):
        dated = E182_ROUND_MS.get(m)
        delta = f"{entry['round_ms'] - dated:8.2f}" if dated else "       -"
        print(f"{m:>3} {entry['round_ms']:9.3f} "
              f"{entry.get('round_ms_leg_spread', float('nan')):7.3f} "
              f"{sum(entry['rounds_at_width']):7d} "
              f"{entry['accepted_mean']:5.2f} "
              f"{dated if dated else 0:8.2f} {delta}")
    print("\nsteps (round-weighted with the shipped served-width distribution)")
    for step in report["steps"]:
        print(f"  {step['boundary']:>6}  raw {step['raw_step_ms']:8.3f} ms  "
              f"share>= {step['shipped_share_at_or_above_m']:5.3f}  "
              f"weighted {step['round_weighted_step_ms']:7.3f} ms/round")

    if band_table:
        short = {b: b[len("band_"):-len("_us")] for b in BANDS}
        print("\nband arms (ATTRIBUTION ONLY, syncs inflate every round)")
        print(f"{'m':>3} {'armround':>9} {'bandsum':>8} {'unacct':>7} "
              + " ".join(f"{short[b]:>10}" for b in BANDS))
        for m, entry in sorted(band_table.items()):
            vals = entry.get("phase_ms", {})
            print(f"{m:>3} {entry.get('band_arm_round_ms', 0.0):9.3f} "
                  f"{entry['band_sum_ms']:8.3f} "
                  f"{entry.get('band_unaccounted_ms', 0.0):7.3f} "
                  + " ".join(f"{vals.get(b, {}).get('value', 0.0):10.3f}"
                             for b in BANDS))
        print("\nband split of each step (share of the band-measured step)")
        for step in report["band_steps"]:
            parts = " ".join(
                f"{short[b]} {step['band_step_ms'][b]:+7.3f} "
                f"({step['band_share'][b]:+.0%})"
                for b in BANDS if b in step["band_step_ms"])
            print(f"  {step['boundary']:>5} total {step['band_step_total_ms']:7.3f} ms  "
                  f"armround {step['band_arm_round_step_ms']:7.3f} ms")
            print(f"        {parts}")

    law = report["ipg_law"]
    print("\nsingle-pass cost against group width IPG (m=6 excluded, mixed plan)")
    for ipg, ms in sorted(law["single_pass_ipg_ladder_ms"].items(), key=lambda kv: int(kv[0])):
        print(f"  IPG {ipg}  f = {ms:8.3f} ms/round")
    for k, v in law["ipg_ladder_steps_ms"].items():
        print(f"  IPG {k:>4}  step {v:7.3f} ms/round")
    print("\nsecond weight pass at fixed IPG")
    for p in law["fixed_ipg_pass_doubling"]:
        print(f"  IPG {p['ipg']}  m={p['m_one_pass']} (G=1) {p['round_ms_one_pass']:8.3f} -> "
              f"m={p['m_two_pass']} (G=2) {p['round_ms_two_pass']:8.3f}  "
              f"ratio {p['ratio']:5.3f}  implied fixed overhead "
              f"{p['implied_fixed_overhead_ms']:6.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
