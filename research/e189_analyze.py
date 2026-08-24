#!/usr/bin/env python3
"""E189 analysis: numerical gate table, per-cell probe medians, ladder legs.

    research/e189_analyze.py gate  research/out/e189/<tag>/gate.json
    research/e189_analyze.py probe research/out/e189/<tag>/probe.json
    research/e189_analyze.py ladder research/out/e189/ladder

harness=local, not gate-qualified, no official score.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import median


def load(path: str):
    with open(path) as handle:
        return json.load(handle)


def gate(path: str) -> int:
    payload = load(path)
    print(f"active_plan={payload['active_plan']} arm={payload['qmv_arm']}")
    print(
        f"{'cell':<14}{'m':>3}{'table':>7}{'elems':>10}{'differ':>8}"
        f"{'max_ulp':>9}{'max_abs':>12}{'G_staged':>10}{'G_single':>10}"
    )
    for row in payload["cells"]:
        print(
            f"{row['cell']:<14}{row['m']:>3}{str(row['use_table']):>7}"
            f"{row['elements']:>10}{row['differing']:>8}{row['max_ulp']:>9}"
            f"{row['max_abs_diff']:>12.6g}{row['staged_groups']:>10}"
            f"{row['single_pass_groups']:>10}"
        )
    print()
    print("positive control (must differ):")
    for row in payload["positive_control"]:
        print(
            f"  m={row['m']} table={row['use_table']} differ={row['differing']}"
            f" max_ulp={row['max_ulp']} max_abs={row['max_abs_diff']:.6g}"
        )
    worst = payload["worst_max_ulp"]
    total = payload["total_differing"]
    tripped = all(row["differing"] > 0 for row in payload["positive_control"])
    print()
    print(f"worst_max_ulp={worst} total_differing={total} control_tripped={tripped}")
    return 0 if total == 0 and tripped else 1


def probe(path: str) -> int:
    payload = load(path)
    samples = payload["samples"]
    cells: dict[tuple[str, int, str], list[float]] = {}
    invocations: dict[str, int] = {}
    groups: dict[tuple[str, int, str], int] = {}
    for s in samples:
        cells.setdefault((s["cell"], s["m"], s["arm"]), []).append(s["microseconds"])
        invocations[s["cell"]] = s["invocations_per_round"]
        groups[(s["cell"], s["m"], s["arm"])] = s["groups"]
    names = sorted({c for c, _, _ in cells})
    widths = sorted({m for _, m, _ in cells})
    print(
        f"{'cell':<14}{'m':>3}{'staged_us':>11}{'single_us':>11}"
        f"{'delta_us':>10}{'pct':>8}{'G_s':>5}{'G_p':>5}{'inv':>5}{'round_ms':>10}"
    )
    round_delta = {m: 0.0 for m in widths}
    for name in names:
        for m in widths:
            a = cells.get((name, m, "staged"))
            b = cells.get((name, m, "singlepass"))
            if not a or not b:
                continue
            sa, sb = median(a), median(b)
            delta = sb - sa
            per_round_ms = delta * invocations[name] / 1000.0
            round_delta[m] += per_round_ms
            print(
                f"{name:<14}{m:>3}{sa:>11.1f}{sb:>11.1f}{delta:>10.1f}"
                f"{100.0 * delta / sa:>7.1f}%"
                f"{groups[(name, m, 'staged')]:>5}"
                f"{groups[(name, m, 'singlepass')]:>5}"
                f"{invocations[name]:>5}{per_round_ms:>10.2f}"
            )
    print()
    for m in widths:
        print(f"projected round delta at m={m}: {round_delta[m]:+.2f} ms/round")
    temps = [t["gpu_temp_c"] for t in payload.get("temperatures", [])]
    if temps:
        print(f"gpu_temp_c min={min(temps):.1f} max={max(temps):.1f}")
    return 0


ROUND_RE = re.compile(r"round_us=(\d+)")


def ladder(root: str) -> int:
    base = Path(root)
    legs = []
    for meta_path in sorted(base.glob("*/leg*/meta.txt")):
        meta = dict(
            line.split("=", 1)
            for line in meta_path.read_text().splitlines()
            if "=" in line
        )
        leg_dir = meta_path.parent
        rounds = []
        trace = leg_dir / "trace.txt"
        if trace.exists():
            for line in trace.read_text().splitlines():
                if line.startswith("mtp-trace: round="):
                    match = ROUND_RE.search(line)
                    if match:
                        rounds.append(int(match.group(1)) / 1000.0)
        score = {}
        score_path = leg_dir / "score.json"
        if score_path.exists():
            score = load(str(score_path)).get("metrics", {})
        # Round 1 carries first-round warmup, so drop it from the median.
        body = rounds[1:] if len(rounds) > 2 else rounds
        legs.append(
            {
                "arm": meta.get("e189_arm", "?"),
                "leg": int(meta.get("e189_leg", "0")),
                "plan": meta.get("width_plan", "?"),
                "m": int(meta.get("verify_width_m", "0")),
                "round_ms_median": median(body) if body else float("nan"),
                "rounds": len(body),
                "mtp_s_per_token": score.get("mtp_seconds_per_token"),
                "serial_s_per_token": score.get("serial_seconds_per_token"),
                "speedup": score.get("mtp_decode_speedup"),
                "matched": score.get("all_tokens_matched"),
                "acc": score.get("accepted_draft_rate"),
                "entry_c": meta.get("gpu_temp_entry"),
                "exit_c": meta.get("gpu_temp_exit"),
            }
        )
    print(
        f"{'leg':>4}{'arm':>6}{'m':>3}{'round_ms':>10}{'mtp_s/tok':>11}"
        f"{'speedup':>9}{'matched':>9}{'entry_c':>9}{'exit_c':>9}"
    )
    for leg in sorted(legs, key=lambda item: item["leg"]):
        entry = float(leg["entry_c"]) if leg["entry_c"] not in (None, "unavailable") else float("nan")
        exit_c = float(leg["exit_c"]) if leg["exit_c"] not in (None, "unavailable") else float("nan")
        mtp = leg["mtp_s_per_token"]
        print(
            f"{leg['leg']:>4}{leg['arm']:>6}{leg['m']:>3}"
            f"{leg['round_ms_median']:>10.2f}"
            f"{(mtp if mtp is not None else float('nan')):>11.5f}"
            f"{(leg['speedup'] if leg['speedup'] is not None else float('nan')):>9.4f}"
            f"{str(leg['matched']):>9}{entry:>9.1f}{exit_c:>9.1f}"
        )
    print()
    print(f"{'m':>3}{'staged_ms':>11}{'single_ms':>11}{'delta_ms':>10}{'pct':>8}"
          f"{'staged_s/tok':>14}{'single_s/tok':>14}{'tok_delta_pct':>15}")
    for m in sorted({leg["m"] for leg in legs}):
        s = [leg for leg in legs if leg["m"] == m and leg["plan"] == "staged"]
        p = [leg for leg in legs if leg["m"] == m and leg["plan"] == "singlepass"]
        if not s or not p:
            continue
        sm = median([leg["round_ms_median"] for leg in s])
        pm = median([leg["round_ms_median"] for leg in p])
        st = median([leg["mtp_s_per_token"] for leg in s if leg["mtp_s_per_token"]])
        pt = median([leg["mtp_s_per_token"] for leg in p if leg["mtp_s_per_token"]])
        print(
            f"{m:>3}{sm:>11.2f}{pm:>11.2f}{pm - sm:>10.2f}"
            f"{100.0 * (pm - sm) / sm:>7.1f}%{st:>14.5f}{pt:>14.5f}"
            f"{100.0 * (pt - st) / st:>14.2f}%"
        )
    bad = [leg for leg in legs if leg["matched"] is not True]
    if bad:
        print(f"\nLEGS WITHOUT all_tokens_matched: {[leg['leg'] for leg in bad]}")
        return 1
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    section, path = sys.argv[1], sys.argv[2]
    if section == "gate":
        return gate(path)
    if section == "probe":
        return probe(path)
    if section == "ladder":
        return ladder(path)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
