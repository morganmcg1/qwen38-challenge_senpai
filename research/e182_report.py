#!/usr/bin/env python3
"""E182: phase-resolved round cost against verify width M.

    usage: research/e182_report.py [--root research/out/e182]
                                   [--out research/out/e182/report.json]

harness=local. Every leg ran with the per-round trace on and the cool gate off,
so no number here is a gate-qualified timing claim. The band arm additionally
drains the device 129 times per verify forward, so its ROUND time is inflated
by construction; only the shape of a band against M is read from it.

Two shape models are fitted to every phase and compared by AICc:

  smooth     p(M) = a + b*M + c*M^2
  step       p(M) = a + b*M + s*groups(M), groups from the replica width plan
                    (Qwen35.swift:1715-1728), i.e. the kernel-level prediction

`groups(M)` is 1 for M in 1..5, 2 for M in 6..8 and 3 for M = 9.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import statistics

DROP_FIRST_ROUNDS = 2

PHASES = [
    "round_us",
    "draft_build_us",
    "d_head1_us",
    "d_chain_us",
    "d_submit1_us",
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
COUNTERS = ["acc", "band_fwd", "xs_fill", "xs_hit"]


def groups(m: int) -> int:
    """Active replica QMV input groups at verify width m."""
    inputs_per_group = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
    if m <= 1:
        return 1
    ipg = inputs_per_group[m]
    return (m + ipg - 1) // ipg


def parse_leg(path: pathlib.Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        kv = dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
        if "round" not in kv or float(kv["round"]) <= DROP_FIRST_ROUNDS:
            continue
        row = {"round": float(kv["round"]), "d": float(kv.get("d", "nan"))}
        for field in PHASES + COUNTERS:
            if field in kv:
                row[field] = float(kv[field])
        rows.append(row)
    return rows


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        out[key] = value
    return out


def fit(xs: list[float], ys: list[float], basis) -> tuple[list[float], float]:
    """Least squares on the given basis; returns coefficients and RSS."""
    cols = [basis(x) for x in xs]
    k = len(cols[0])
    ata = [[sum(c[i] * c[j] for c in cols) for j in range(k)] for i in range(k)]
    atb = [sum(c[i] * y for c, y in zip(cols, ys)) for i in range(k)]
    for i in range(k):
        pivot = max(range(i, k), key=lambda r: abs(ata[r][i]))
        ata[i], ata[pivot] = ata[pivot], ata[i]
        atb[i], atb[pivot] = atb[pivot], atb[i]
        if abs(ata[i][i]) < 1e-12:
            return [float("nan")] * k, float("nan")
        for r in range(k):
            if r == i:
                continue
            f = ata[r][i] / ata[i][i]
            for c in range(i, k):
                ata[r][c] -= f * ata[i][c]
            atb[r] -= f * atb[i]
    beta = [atb[i] / ata[i][i] for i in range(k)]
    rss = sum(
        (y - sum(b * c for b, c in zip(beta, col))) ** 2
        for y, col in zip(ys, cols)
    )
    return beta, rss


def aicc(rss: float, n: int, k: int) -> float:
    if rss <= 0 or n <= k + 2:
        return float("nan")
    return n * math.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def shape_verdict(ms: list[int], vals: list[float]) -> dict:
    xs = [float(m) for m in ms]
    smooth, rss_s = fit(xs, vals, lambda m: [1.0, m, m * m])
    step, rss_t = fit(xs, vals, lambda m: [1.0, m, float(groups(int(m)))])
    n = len(xs)
    a_s, a_t = aicc(rss_s, n, 4), aicc(rss_t, n, 4)
    return {
        "smooth_coeffs": smooth,
        "smooth_rmse_us": math.sqrt(rss_s / n) if rss_s == rss_s else None,
        "step_coeffs": step,
        "step_rmse_us": math.sqrt(rss_t / n) if rss_t == rss_t else None,
        "aicc_smooth": a_s,
        "aicc_step": a_t,
        "preferred": (
            "step" if a_t == a_t and a_s == a_s and a_t < a_s else "smooth"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/out/e182")
    parser.add_argument("--out", default="research/out/e182/report.json")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    legs = []
    for meta_path in sorted(root.glob("*/leg*/meta.txt")):
        meta = read_meta(meta_path)
        trace = meta_path.parent / "trace.txt"
        if not trace.exists():
            continue
        rows = parse_leg(trace)
        if not rows:
            continue
        legs.append(
            {
                "arm": meta["e182_arm"],
                "leg": int(meta["e182_leg"]),
                "m": int(meta["verify_width_m"]),
                "band_sync": meta.get("band_sync", "0") == "1",
                "entry_c": meta.get("gpu_temp_entry", ""),
                "exit_c": meta.get("gpu_temp_exit", ""),
                "exit_code": meta.get("exit", ""),
                "rounds": len(rows),
                "rows": rows,
            }
        )

    per_leg = []
    pooled: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for leg in legs:
        summary = {
            k: leg[k]
            for k in (
                "arm", "leg", "m", "band_sync", "entry_c", "exit_c",
                "exit_code", "rounds",
            )
        }
        for field in PHASES + COUNTERS:
            vals = [r[field] for r in leg["rows"] if field in r]
            if vals:
                summary[f"{field}_p50"] = statistics.median(vals)
        per_leg.append(summary)
        kind = "band" if leg["band_sync"] else "trace"
        pooled[(kind, leg["m"])].extend(leg["rows"])

    tables = {}
    for kind in ("trace", "band"):
        table = []
        for m in sorted(k[1] for k in pooled if k[0] == kind):
            rows = pooled[(kind, m)]
            entry = {"m": m, "groups": groups(m), "n": len(rows)}
            for field in PHASES + COUNTERS:
                vals = [r[field] for r in rows if field in r]
                if vals:
                    entry[field] = statistics.median(vals)
                    entry[f"{field}_iqr"] = (
                        statistics.quantiles(vals, n=4)[2]
                        - statistics.quantiles(vals, n=4)[0]
                        if len(vals) >= 4
                        else 0.0
                    )
            table.append(entry)
        tables[kind] = table

    shapes = {}
    for kind, table in tables.items():
        ms = [row["m"] for row in table]
        for field in PHASES:
            vals = [row.get(field) for row in table]
            if any(v is None for v in vals) or len(ms) < 6:
                continue
            if max(vals) < 50.0:
                continue
            shapes[f"{kind}.{field}"] = shape_verdict(ms, vals)

    report = {
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "trace_perturbs_timing": True,
        "drop_first_rounds": DROP_FIRST_ROUNDS,
        "legs": per_leg,
        "tables": tables,
        "shapes": shapes,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for kind, table in tables.items():
        print(f"\n== {kind} arm (median us per round) ==")
        cols = [f for f in PHASES if any(f in row for row in table)]
        print("  M  G     n " + " ".join(f"{c[:-3]:>14}" for c in cols))
        for row in table:
            cells = " ".join(f"{row.get(c, float('nan')):>14.1f}" for c in cols)
            print(f"{row['m']:>3} {row['groups']:>2} {row['n']:>5} {cells}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
