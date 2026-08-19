#!/usr/bin/env python3
"""E56 ABBA analysis: read the four legs and apply the pre-registered rules.

  python3 research/e56_analyze.py [--out research/e56-abba.json]

The null arm is the pair of `base` legs. Two byte-identical builds measured in
one session bound what this instrument calls a difference when there is none,
so the base-to-base spread -- not a nominal noise figure -- is the bar the
candidate has to clear.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDER = [("baseA", "base"), ("schedB", "sched"),
         ("schedB2", "sched"), ("baseA2", "base")]
NULL_FLOOR_PCT = 0.0629
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")

CORRECTNESS = ("all_tokens_matched", "residual_divergence_count",
               "public_drift_tripwire_passed", "mtp_depth", "decode_tokens",
               "head_provenance_sha256", "uses_pinned_mtp_head")


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                out[key.strip()] = value.strip()
    return out


def width_histogram(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    legs, current, last = [], [], -1
    for line in path.read_text(errors="replace").splitlines():
        match = ROUND_RE.search(line)
        if not match:
            continue
        index, depth, accepted = (int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)))
        if index <= last and current:
            legs.append(current)
            current = []
        last = index
        current.append((depth, accepted))
    if current:
        legs.append(current)
    drafting = [leg for leg in legs if any(d > 0 for d, _ in leg)]
    if not drafting:
        return {}
    leg = max(drafting, key=len)
    widths = Counter(d + 1 for d, _ in leg)
    rounds = len(leg)
    return {
        "rounds": rounds,
        "mean_verify_width": sum(d + 1 for d, _ in leg) / rounds,
        "share": {w: round(widths.get(w, 0) / rounds, 5) for w in range(1, 10)},
        "count": {w: widths.get(w, 0) for w in range(1, 10)},
    }


def load_legs() -> list[dict]:
    legs = []
    for tag, arm in ORDER:
        out_dir = ROOT / "research" / "out" / tag
        score_path = out_dir / "score.json"
        if not score_path.exists():
            legs.append({"tag": tag, "arm": arm, "status": "missing"})
            continue
        score = json.loads(score_path.read_text())
        legs.append({
            "tag": tag,
            "arm": arm,
            "status": "ok",
            "score": score.get("score"),
            "passed": score.get("passed"),
            "metrics": score.get("metrics", {}),
            "meta": read_meta(out_dir / "meta.txt"),
            "widths": width_histogram(out_dir / "trace.txt"),
        })
    return legs


def arm_values(legs: list[dict], arm: str, key: str) -> list[float]:
    return [leg["metrics"][key] for leg in legs
            if leg["status"] == "ok" and leg["arm"] == arm
            and isinstance(leg["metrics"].get(key), (int, float))]


def contrast(legs: list[dict], key: str) -> dict | None:
    base, sched = arm_values(legs, "base", key), arm_values(legs, "sched", key)
    if not base or not sched:
        return None
    mb, ms = st.mean(base), st.mean(sched)
    return {
        "base": base,
        "sched": sched,
        "base_mean": mb,
        "sched_mean": ms,
        "delta_pct": 100.0 * (ms / mb - 1.0) if mb else None,
        "base_null_spread_pct": (100.0 * (max(base) - min(base)) / mb
                                 if len(base) > 1 and mb else None),
        "sched_spread_pct": (100.0 * (max(sched) - min(sched)) / ms
                             if len(sched) > 1 and ms else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="research/e56-abba.json")
    args = parser.parse_args()

    legs = load_legs()
    report = {"legs": legs, "null_floor_pct": NULL_FLOOR_PCT, "contrasts": {}}

    print(f"{'tag':<9}{'arm':<7}{'serial s/tok':>14}{'mtp s/tok':>14}"
          f"{'speedup':>10}{'draft':>8}{'acc':>8}")
    for leg in legs:
        if leg["status"] != "ok":
            print(f"{leg['tag']:<9}{leg['arm']:<7}  MISSING")
            continue
        m = leg["metrics"]
        print(f"{leg['tag']:<9}{leg['arm']:<7}"
              f"{m.get('serial_seconds_per_token', float('nan')):>14.8f}"
              f"{m.get('mtp_seconds_per_token', float('nan')):>14.8f}"
              f"{m.get('mtp_decode_speedup', float('nan')):>10.5f}"
              f"{m.get('effective_mean_draft_len', float('nan')):>8.3f}"
              f"{m.get('accepted_draft_rate', float('nan')):>8.3f}")

    print()
    for key, label in (("mtp_decode_speedup", "local ratio (PRIMARY)"),
                       ("mtp_seconds_per_token", "candidate s/token"),
                       ("serial_seconds_per_token", "serial leg (must not move)"),
                       ("effective_mean_draft_len", "mean draft length"),
                       ("accepted_draft_rate", "accepted draft rate")):
        row = contrast(legs, key)
        if row is None:
            continue
        report["contrasts"][key] = row
        null = row["base_null_spread_pct"]
        print(f"{label:<30} base {row['base_mean']:.8f}  sched {row['sched_mean']:.8f}"
              f"  {row['delta_pct']:+.4f} %"
              + (f"  | null arm spread {null:.4f} %" if null is not None else ""))

    print()
    print("Correctness and provenance (must be identical across all arms):")
    for key in CORRECTNESS:
        values = {leg["tag"]: leg["metrics"].get(key)
                  for leg in legs if leg["status"] == "ok"}
        unique = set(map(str, values.values()))
        verdict = "IDENTICAL" if len(unique) == 1 else "DIFFERS"
        shown = next(iter(unique)) if len(unique) == 1 else values
        print(f"  {key:<32}{verdict}  {shown}")
    report["correctness"] = {
        key: {leg["tag"]: leg["metrics"].get(key)
              for leg in legs if leg["status"] == "ok"}
        for key in CORRECTNESS}

    print()
    print("Verify-width share by arm (the mechanism readout):")
    print(f"{'tag':<9}{'rounds':>7}{'mean W':>8}" +
          "".join(f"{w:>8}" for w in range(1, 10)))
    for leg in legs:
        widths = leg.get("widths") or {}
        if not widths:
            continue
        print(f"{leg['tag']:<9}{widths['rounds']:>7}{widths['mean_verify_width']:>8.3f}"
              + "".join(f"{widths['share'][w]:>8.3f}" for w in range(1, 10)))

    print()
    print("Thermal record, taken from each leg's own benchmark.sh output:")
    for leg in legs:
        meta = leg.get("meta") or {}
        print(f"  {leg['tag']:<9} entry={meta.get('entry_gpu_temp_c')}"
              f" exit={meta.get('exit_gpu_temp_c')}"
              f" gate_passes={meta.get('cool_gate_passes')}"
              f" gate_skips={meta.get('cool_gate_skips')}"
              f" cool_gate_passed_real_gate={meta.get('cool_gate_passed_real_gate')}")

    print()
    print("Arm provenance. The checkout blob is HEAD on every leg; the arm is")
    print("selected by which prebuilt worker binary the leg ran.")
    for leg in legs:
        meta = leg.get("meta") or {}
        print(f"  {leg['tag']:<9} arm={meta.get('e56_arm')}"
              f" arm_schedule_blob={str(meta.get('arm_schedule_blob'))[:12]}"
              f" worker_sha256={str(meta.get('worker_sha256'))[:12]}"
              f" metallib_sha256={str(meta.get('metallib_sha256'))[:12]}"
              f" checkout_schedule_blob={str(meta.get('checkout_schedule_blob'))[:12]}")

    out_path = ROOT / args.out
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
