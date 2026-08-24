#!/usr/bin/env python3
"""E195 analysis: numerical gate table, per-cell probe medians, ladder legs.

    research/e195_analyze.py gate  research/out/e195/<tag>/gate.json
    research/e195_analyze.py probe research/out/e195/<tag>/probe.json
    research/e195_analyze.py ladder research/out/e195/ladder

harness=local, not gate-qualified, no official score.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import median, stdev


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
DEPTH_RE = re.compile(r"round=\d+ d=(\d+) acc=(\d+)")
PLAN_RE = re.compile(r"qmv_plan=(\S+)")
SP_RE = re.compile(r"qmv_sp=(\d+)")


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
        # The local-iterate leg traces the serial reference rounds (d=0) and the
        # drafting rounds (d>0) into one file. Only the drafting rounds carry
        # the verified width this experiment changes.
        drafting: list[float] = []
        serial: list[float] = []
        accepted: list[int] = []
        proposed = 0
        traced_plan = None
        qmv_sp = 0
        trace = leg_dir / "trace.txt"
        if trace.exists():
            for line in trace.read_text().splitlines():
                if not line.startswith("mtp-trace: round="):
                    continue
                round_match = ROUND_RE.search(line)
                depth_match = DEPTH_RE.search(line)
                if not round_match or not depth_match:
                    continue
                plan_match = PLAN_RE.search(line)
                if plan_match:
                    traced_plan = plan_match.group(1)
                sp_match = SP_RE.search(line)
                if sp_match:
                    qmv_sp = int(sp_match.group(1))
                value = int(round_match.group(1)) / 1000.0
                if int(depth_match.group(1)) > 0:
                    drafting.append(value)
                    accepted.append(int(depth_match.group(2)))
                    proposed += int(depth_match.group(1))
                else:
                    serial.append(value)
        score = {}
        score_path = leg_dir / "score.json"
        if score_path.exists():
            score = load(str(score_path)).get("metrics", {})
        # The first drafting round carries warmup, so drop it from the median.
        body = drafting[1:] if len(drafting) > 2 else drafting
        legs.append(
            {
                "arm": meta.get("e195_arm", "?"),
                "leg": int(meta.get("e195_leg", "0")),
                "plan": meta.get("width_plan", "?"),
                "m": int(meta.get("verify_width_m", "0")),
                "round_ms_median": median(body) if body else float("nan"),
                "serial_ms_median": median(serial[1:]) if len(serial) > 2 else float("nan"),
                "accepted_mean": sum(accepted) / len(accepted) if accepted else float("nan"),
                "accepted_total": sum(accepted),
                "proposed": proposed,
                "rounds": len(drafting),
                "traced_plan": traced_plan,
                "qmv_sp": qmv_sp,
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
        f"{'leg':>4}{'arm':>6}{'m':>3}{'draft_ms':>10}{'serial_ms':>11}"
        f"{'acc':>6}{'mtp_s/tok':>11}{'speedup':>9}{'matched':>9}"
        f"{'entry_c':>9}{'exit_c':>9}{'traced':>12}{'qmv_sp':>9}"
    )
    for leg in sorted(legs, key=lambda item: item["leg"]):
        entry = float(leg["entry_c"]) if leg["entry_c"] not in (None, "unavailable") else float("nan")
        exit_c = float(leg["exit_c"]) if leg["exit_c"] not in (None, "unavailable") else float("nan")
        mtp = leg["mtp_s_per_token"]
        print(
            f"{leg['leg']:>4}{leg['arm']:>6}{leg['m']:>3}"
            f"{leg['round_ms_median']:>10.2f}{leg['serial_ms_median']:>11.2f}"
            f"{leg['accepted_mean']:>6.2f}"
            f"{(mtp if mtp is not None else float('nan')):>11.5f}"
            f"{(leg['speedup'] if leg['speedup'] is not None else float('nan')):>9.4f}"
            f"{str(leg['matched']):>9}{entry:>9.1f}{exit_c:>9.1f}"
            f"{str(leg['traced_plan']):>12}{leg['qmv_sp']:>9}"
        )
    # Every arm is compared with the staged control at the SAME width, on
    # absolute candidate figures. The local serial-to-MTP ratio is reported per
    # leg but never used as the headline: both local legs run one candidate
    # build, so a change that also speeds the serial leg cancels there.
    print()
    print(
        f"{'m':>3}{'plan':>12}{'legs':>5}{'draft_ms':>10}{'sigma':>8}"
        f"{'d_vs_staged':>12}{'pct':>8}{'s/tok':>10}{'tok_pct':>9}"
        f"{'serial_ms':>11}"
    )
    for m in sorted({leg["m"] for leg in legs}):
        at_width = [leg for leg in legs if leg["m"] == m]
        control = [leg for leg in at_width if leg["plan"] == "staged"]
        if not control:
            continue
        base_ms = median([leg["round_ms_median"] for leg in control])
        base_tok = median(
            [leg["mtp_s_per_token"] for leg in control if leg["mtp_s_per_token"]]
        )
        for plan in ["staged", "singlepass", "selective", "selective7"]:
            arm = [leg for leg in at_width if leg["plan"] == plan]
            if not arm:
                continue
            values = [leg["round_ms_median"] for leg in arm]
            arm_ms = median(values)
            sigma = stdev(values) if len(values) > 1 else 0.0
            toks = [leg["mtp_s_per_token"] for leg in arm if leg["mtp_s_per_token"]]
            arm_tok = median(toks) if toks else float("nan")
            serial = median([leg["serial_ms_median"] for leg in arm])
            print(
                f"{m:>3}{plan:>12}{len(arm):>5}{arm_ms:>10.2f}{sigma:>8.2f}"
                f"{arm_ms - base_ms:>12.2f}"
                f"{100.0 * (arm_ms - base_ms) / base_ms:>7.1f}%"
                f"{arm_tok:>10.5f}"
                f"{100.0 * (arm_tok - base_tok) / base_tok:>8.2f}%"
                f"{serial:>11.2f}"
            )

    # RULE 179: the arms must have run the SAME schedule. A difference in
    # rounds, proposed drafts or accepted drafts means the comparison priced a
    # different trajectory, not the kernel plan.
    print()
    print(f"{'m':>3}{'plan':>12}{'rounds':>8}{'proposed':>10}{'accepted':>10}")
    for m in sorted({leg["m"] for leg in legs}):
        for plan in sorted({leg["plan"] for leg in legs if leg["m"] == m}):
            arm = [leg for leg in legs if leg["m"] == m and leg["plan"] == plan]
            print(
                f"{m:>3}{plan:>12}"
                f"{sorted({leg['rounds'] for leg in arm})!s:>8}"
                f"{sorted({leg['proposed'] for leg in arm})!s:>10}"
                f"{sorted({leg['accepted_total'] for leg in arm})!s:>10}"
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
