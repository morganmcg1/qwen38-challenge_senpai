#!/usr/bin/env python3
"""E124 stage 1: acceptance per island arm, with an interval and an exactness
cross-check. No timing claim is made or read here.

ESTIMATOR. The acceptance rate is a RATIO of two sums over rounds, not a mean
of Bernoulli trials, and draft decisions inside one round are not independent:
a rejection at position j truncates every deeper position, so the round is the
independent unit. The point estimate is `sum(accepted) / sum(proposed)` and the
interval is a nonparametric CLUSTER bootstrap that resamples whole rounds. The
delta-method ratio SE is reported beside it as a closed-form check.

Arms are NOT paired. The emitted token sequence is identical across arms by the
exactness gate, but the round structure is not: an arm that accepts less needs
more rounds for the same 512 tokens. So the arm difference uses an unpaired
bootstrap over the two independent round sets.

EXACTNESS. `mtp-row:` lines carry the target's exact top-two evidence at every
emitted position. Two arms must agree on the ordered top-1 token at every
position. The top-two VALUES may differ in their last bits when a position is
evaluated at a different verify width, so a value difference is reported but is
not a failure; a token difference is a failure.

  python3 research/e124_accept.py --tags e124s1p1all e124s1p2none ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

OUT = Path("research/out")
BOOTSTRAP = 20_000
SEED = 20260822


def read_rounds(tag: str) -> list[dict]:
    path = OUT / tag / "trace.txt"
    rounds = []
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        rounds.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)})
    return rounds


def read_rows(tag: str) -> list[tuple[int, int, int, str]]:
    path = OUT / tag / "trace.txt"
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"^mtp-row: pos=(\d+) ids=(-?\d+),(-?\d+) v=(\S+)", line)
        if m:
            rows.append((int(m[1]), int(m[2]), int(m[3]), m[4]))
    return rows


def read_score(tag: str) -> dict:
    path = OUT / tag / "score.json"
    return json.loads(path.read_text())["metrics"] if path.exists() else {}


def read_meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def ratio(rounds: list[dict]) -> float:
    den = sum(r["d"] for r in rounds)
    return sum(r["acc"] for r in rounds) / den if den else float("nan")


def delta_method_se(rounds: list[dict]) -> float:
    n = len(rounds)
    if n < 2:
        return float("nan")
    r = ratio(rounds)
    xbar = statistics.mean(x["d"] for x in rounds)
    resid = sum((x["acc"] - r * x["d"]) ** 2 for x in rounds)
    return math.sqrt(resid / (n - 1)) / (math.sqrt(n) * xbar)


def bootstrap_ratios(rounds: list[dict], rng: random.Random) -> list[float]:
    n = len(rounds)
    draws = []
    for _ in range(BOOTSTRAP):
        sample = [rounds[rng.randrange(n)] for _ in range(n)]
        draws.append(ratio(sample))
    return draws


def conditionals(rounds: list[dict], max_pos: int = 8) -> list[dict]:
    """P(position j accepted | j proposed and 1..j-1 accepted)."""
    out = []
    for j in range(1, max_pos + 1):
        eligible = [r for r in rounds if r["d"] >= j and r["acc"] >= j - 1]
        if not eligible:
            out.append({"position": j, "eligible": 0})
            continue
        hit = sum(1 for r in eligible if r["acc"] >= j)
        p = hit / len(eligible)
        se = math.sqrt(p * (1 - p) / len(eligible))
        out.append({"position": j, "eligible": len(eligible),
                    "accepted": hit, "p": p, "se": se})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--control", default=None,
                    help="tag of the arm `all` control; default is the first "
                         "tag whose meta says e124_arm=all")
    ap.add_argument("--kill-line-pt", type=float, default=0.21)
    ap.add_argument("--out", default="research/out/e124-acceptance.json")
    args = ap.parse_args()

    rng = random.Random(SEED)
    legs = {}
    for tag in args.tags:
        meta = read_meta(tag)
        rounds = read_rounds(tag)
        score = read_score(tag)
        draws = bootstrap_ratios(rounds, rng) if rounds else []
        draws_sorted = sorted(draws)
        legs[tag] = {
            "tag": tag,
            "arm": meta.get("e124_arm", "?"),
            "arm_witness": meta.get("e124_arm_witness", "<absent>"),
            "head_provenance_sha256": score.get("head_provenance_sha256"),
            "tokens": meta.get("tokens"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "rounds": len(rounds),
            "proposed": sum(r["d"] for r in rounds),
            "accepted": sum(r["acc"] for r in rounds),
            "acceptance_rate": ratio(rounds) if rounds else None,
            "acceptance_se_delta_method": delta_method_se(rounds) if rounds else None,
            "acceptance_ci95_cluster_bootstrap": (
                [draws_sorted[int(0.025 * BOOTSTRAP)],
                 draws_sorted[int(0.975 * BOOTSTRAP)]] if draws else None),
            "effective_mean_draft_len_trace": (
                statistics.mean(r["d"] for r in rounds) if rounds else None),
            "accepted_per_round": (
                statistics.mean(r["acc"] for r in rounds) if rounds else None),
            "verify_width_M": (
                statistics.mean(r["d"] for r in rounds) + 1 if rounds else None),
            "mean_round_us": (
                statistics.mean(r["round_us"] for r in rounds) if rounds else None),
            "mean_draft_build_us": (
                statistics.mean(r["draft_build_us"] for r in rounds) if rounds else None),
            "score_accepted_draft_rate": score.get("accepted_draft_rate"),
            "score_effective_mean_draft_len": score.get("effective_mean_draft_len"),
            "all_tokens_matched": score.get("all_tokens_matched"),
            "residual_divergence_count": score.get("residual_divergence_count"),
            "mtp_seconds_per_token": score.get("mtp_seconds_per_token"),
            "serial_seconds_per_token": score.get("serial_seconds_per_token"),
            "mtp_decode_speedup": score.get("mtp_decode_speedup"),
            "conditionals": conditionals(rounds),
            "_draws": draws,
            "_rows": read_rows(tag),
        }

    control = args.control or next(
        (t for t, v in legs.items() if v["arm"] == "all"), args.tags[0])
    ctrl = legs[control]

    print(f"control arm `{ctrl['arm']}` = {control}\n")
    print("tag                arm   rounds  proposed  accepted  acceptance  "
          "ci95                 eff_d   M      matched  div")
    for tag, v in legs.items():
        ci = v["acceptance_ci95_cluster_bootstrap"]
        cis = f"[{ci[0]:.6f},{ci[1]:.6f}]" if ci else "n/a"
        print(f"{tag:<18} {v['arm']:<5} {v['rounds']:>6}  {v['proposed']:>8}  "
              f"{v['accepted']:>8}  {v['acceptance_rate']:.6f}  {cis:<20} "
              f"{v['effective_mean_draft_len_trace']:.4f}  "
              f"{v['verify_width_M']:.4f}  {str(v['all_tokens_matched']):<7} "
              f"{v['residual_divergence_count']}")

    print("\ndelta against the control, in ABSOLUTE POINTS "
          f"(kill line {args.kill_line_pt} pt)")
    print("arm    d_acceptance_pt  ci95_pt                 verdict")
    for tag, v in legs.items():
        if tag == control:
            continue
        diff = [(a - b) * 100.0
                for a, b in zip(v["_draws"], ctrl["_draws"])]
        diff.sort()
        lo, hi = diff[int(0.025 * BOOTSTRAP)], diff[int(0.975 * BOOTSTRAP)]
        point = (v["acceptance_rate"] - ctrl["acceptance_rate"]) * 100.0
        killed = point < -args.kill_line_pt
        v["delta_acceptance_pt_vs_control"] = point
        v["delta_acceptance_pt_ci95"] = [lo, hi]
        v["killed_by_acceptance"] = killed
        print(f"{v['arm']:<6} {point:>+15.4f}  [{lo:>+8.4f},{hi:>+8.4f}]  "
              f"{'KILLED' if killed else 'survives'}")

    print("\nper-position acceptance conditionals, "
          "P(pos j accepted | j proposed and 1..j-1 accepted)")
    header = "arm    " + "".join(f"{j:>9}" for j in range(1, 9))
    print(header)
    for tag, v in legs.items():
        cells = []
        for c in v["conditionals"]:
            cells.append(f"{c['p']:>9.4f}" if c.get("eligible") else f"{'-':>9}")
        print(f"{v['arm']:<6} " + "".join(cells))
    print("arm    " + "".join(f"{c['eligible']:>9}" for c in ctrl["conditionals"])
          + "   <- control eligible counts")

    print("\nexactness cross-check against the control "
          "(ordered top-1 token at every emitted position)")
    ctrl_rows = ctrl["_rows"]
    for tag, v in legs.items():
        rows = v["_rows"]
        n = min(len(rows), len(ctrl_rows))
        tok_mismatch = sum(1 for i in range(n) if rows[i][1] != ctrl_rows[i][1])
        pos_mismatch = sum(1 for i in range(n) if rows[i][0] != ctrl_rows[i][0])
        val_mismatch = sum(1 for i in range(n) if rows[i][3] != ctrl_rows[i][3])
        v["rows"] = len(rows)
        v["row_top1_mismatch_vs_control"] = tok_mismatch
        v["row_pos_mismatch_vs_control"] = pos_mismatch
        v["row_value_mismatch_vs_control"] = val_mismatch
        print(f"{v['arm']:<6} rows={len(rows):<6} compared={n:<6} "
              f"top1_mismatch={tok_mismatch:<5} pos_mismatch={pos_mismatch:<5} "
              f"value_mismatch={val_mismatch}")

    for v in legs.values():
        v.pop("_draws", None)
        v.pop("_rows", None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"control": control, "kill_line_pt": args.kill_line_pt,
         "bootstrap_resamples": BOOTSTRAP, "seed": SEED, "legs": legs},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
