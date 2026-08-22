#!/usr/bin/env python3
"""E124 stage 1: acceptance per island arm, STRATIFIED. No timing claim here.

WHY STRATIFIED (F92). 100 % of the published median's marginal weight sits on
hidden prompts that accept 0.83-0.90 at depth 4.4-6.1. Every legacy local prose
fixture accepts 0.44-0.52 at depth ~2.5, which is the zero-weight cluster. An
arm delta pooled over both regimes answers no question the score asks, and the
sign can differ, because deep-position rejections are where a bf16-only Q
projection first diverges. So:

  stratum H  decisive. The kill line and the break-even apply HERE ONLY.
  stratum L  secondary. It informs the mechanism. It cannot kill or pass an arm.

ESTIMATOR. The acceptance rate is a RATIO of two sums over rounds, not a mean
of Bernoulli trials, and draft decisions inside one round are not independent:
a rejection at position j truncates every deeper position, so the round is the
independent unit. The point estimate is `sum(accepted) / sum(proposed)` and the
interval is a nonparametric CLUSTER bootstrap that resamples whole rounds
within each seed. The seed set is treated as fixed, so the interval describes
sampling over rounds, not over prompts.

PAIRING. Arms are paired at the SEED level and unpaired at the ROUND level. The
emitted token sequence is identical across arms by the exactness gate, but the
round structure is not: an arm that accepts less needs more rounds for the same
512 tokens. So a per-seed delta uses an unpaired bootstrap over the two
independent round sets of that seed, and the stratum delta pools those.

EXACTNESS. `mtp-row:` lines carry the target's exact top-two evidence at every
emitted position. Two arms must agree on the ordered top-1 token at every
position of the same seed. The top-two VALUES may differ in their last bits
when a position is evaluated at a different verify width, so a value difference
is reported but is not a failure; a token difference is a failure.

  python3 research/e124_accept.py \
      --runs-template '.mlxfast-private/e122/runs-e124-arm-{arm}/{seed}' \
      --stratum H=benchfixture,beagle_a --stratum L=medicine,travel
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

BOOTSTRAP = 20_000
SEED = 20260822
MAX_POS = 8


def read_rounds(run_dir: Path) -> list[dict]:
    rounds = []
    for line in (run_dir / "trace.txt").read_text(errors="replace").splitlines():
        if line.startswith("mtp-trace: round="):
            rounds.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)})
    return rounds


def read_rows(run_dir: Path) -> list[tuple[int, int, int, str]]:
    rows = []
    for line in (run_dir / "trace.txt").read_text(errors="replace").splitlines():
        m = re.match(r"^mtp-row: pos=(\d+) ids=(-?\d+),(-?\d+) v=(\S+)", line)
        if m:
            rows.append((int(m[1]), int(m[2]), int(m[3]), m[4]))
    return rows


def read_report(run_dir: Path) -> dict:
    path = run_dir / "report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep:
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


def bootstrap_sums(rounds: list[dict], rng: random.Random) -> list[tuple[int, int]]:
    """Resample whole rounds; return (accepted, proposed) per resample.

    Sums, not ratios, so a stratum can pool several seeds inside one resample.
    """
    n = len(rounds)
    draws = []
    for _ in range(BOOTSTRAP):
        acc = prop = 0
        for _ in range(n):
            r = rounds[rng.randrange(n)]
            acc += r["acc"]
            prop += r["d"]
        draws.append((acc, prop))
    return draws


def conditionals(rounds: list[dict]) -> list[dict]:
    """P(position j accepted | j proposed and 1..j-1 accepted)."""
    out = []
    for j in range(1, MAX_POS + 1):
        eligible = [r for r in rounds if r["d"] >= j and r["acc"] >= j - 1]
        if not eligible:
            out.append({"position": j, "eligible": 0})
            continue
        hit = sum(1 for r in eligible if r["acc"] >= j)
        p = hit / len(eligible)
        out.append({"position": j, "eligible": len(eligible), "accepted": hit,
                    "p": p, "se": math.sqrt(p * (1 - p) / len(eligible))})
    return out


def pooled_conditionals(per_seed: list[list[dict]]) -> list[dict]:
    out = []
    for j in range(MAX_POS):
        elig = sum(s[j].get("eligible", 0) for s in per_seed)
        hit = sum(s[j].get("accepted", 0) for s in per_seed)
        if not elig:
            out.append({"position": j + 1, "eligible": 0})
            continue
        p = hit / elig
        out.append({"position": j + 1, "eligible": elig, "accepted": hit,
                    "p": p, "se": math.sqrt(p * (1 - p) / elig)})
    return out


def percentiles(values: list[float]) -> list[float]:
    v = sorted(values)
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]


def load_leg(arm: str, seed: str, run_dir: Path, rng: random.Random) -> dict:
    rounds = read_rounds(run_dir)
    report = read_report(run_dir)
    meta = read_meta(run_dir)
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir),
        "prompt_file": meta.get("prompt_file"),
        "prompt_sha256": meta.get("prompt_sha256"),
        "golden_sha256": meta.get("golden_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "timing_valid": meta.get("timing_valid"),
        "rounds": len(rounds),
        "proposed": sum(r["d"] for r in rounds),
        "accepted": sum(r["acc"] for r in rounds),
        "acceptance_rate": ratio(rounds),
        "acceptance_se_delta_method": delta_method_se(rounds),
        "mean_depth": statistics.mean(r["d"] for r in rounds) if rounds else None,
        "accepted_per_round": statistics.mean(r["acc"] for r in rounds) if rounds else None,
        "decode_tokens": report.get("decode_token_count"),
        "all_tokens_matched": report.get("all_tokens_matched"),
        "residual_divergence_count": report.get("residual_divergence_count"),
        "report_accepted_draft_rate": report.get("accepted_draft_rate"),
        "conditionals": conditionals(rounds),
        "_draws": bootstrap_sums(rounds, rng) if rounds else [],
        "_rows": read_rows(run_dir),
    }


def analyse_stratum(name: str, seeds: list[str], legs: dict, args) -> dict:
    decisive = name == args.decisive_stratum
    out = {"decisive": decisive, "seeds": seeds, "seed_count": len(seeds), "arms": {}}
    if not seeds:
        return out

    for arm in args.arms:
        acc = sum(legs[s][arm]["accepted"] for s in seeds)
        prop = sum(legs[s][arm]["proposed"] for s in seeds)
        draws = [sum(legs[s][arm]["_draws"][b][0] for s in seeds)
                 / sum(legs[s][arm]["_draws"][b][1] for s in seeds)
                 for b in range(BOOTSTRAP)]
        out["arms"][arm] = {
            "pooled_acceptance_rate": acc / prop,
            "pooled_ci95": percentiles(draws),
            "pooled_accepted": acc,
            "pooled_proposed": prop,
            "pooled_rounds": sum(legs[s][arm]["rounds"] for s in seeds),
            "pooled_mean_depth": prop / sum(legs[s][arm]["rounds"] for s in seeds),
            "conditionals": pooled_conditionals(
                [legs[s][arm]["conditionals"] for s in seeds]),
            "per_seed": {s: {k: v for k, v in legs[s][arm].items()
                             if not k.startswith("_")} for s in seeds},
            "_draws": draws,
        }

    ctrl = out["arms"][args.control_arm]
    for arm in args.arms:
        v = out["arms"][arm]
        diff = sorted((a - b) * 100.0 for a, b in zip(v["_draws"], ctrl["_draws"]))
        point = (v["pooled_acceptance_rate"] - ctrl["pooled_acceptance_rate"]) * 100.0
        v["delta_acceptance_pt_vs_control"] = point
        v["delta_acceptance_pt_ci95"] = [diff[int(0.025 * BOOTSTRAP)],
                                         diff[int(0.975 * BOOTSTRAP)]]
        v["kill_line_applies"] = decisive
        v["killed_by_acceptance"] = bool(
            decisive and arm != args.control_arm and point < -args.kill_line_pt)
        for s in seeds:
            leg, cleg = legs[s][arm], legs[s][args.control_arm]
            d = sorted((a[0] / a[1] - b[0] / b[1]) * 100.0
                       for a, b in zip(leg["_draws"], cleg["_draws"]))
            ps = v["per_seed"][s]
            ps["delta_acceptance_pt_vs_control"] = (
                leg["acceptance_rate"] - cleg["acceptance_rate"]) * 100.0
            ps["delta_acceptance_pt_ci95"] = [d[int(0.025 * BOOTSTRAP)],
                                              d[int(0.975 * BOOTSTRAP)]]
            rows, crows = leg["_rows"], cleg["_rows"]
            n = min(len(rows), len(crows))
            ps["rows"] = len(rows)
            ps["rows_compared"] = n
            ps["row_top1_mismatch_vs_control"] = sum(
                1 for i in range(n) if rows[i][1] != crows[i][1])
            ps["row_pos_mismatch_vs_control"] = sum(
                1 for i in range(n) if rows[i][0] != crows[i][0])
            ps["row_value_mismatch_vs_control"] = sum(
                1 for i in range(n) if rows[i][3] != crows[i][3])
        v.pop("_draws")
    return out


def write_and_print(result: dict, args) -> None:
    for name, st in result["strata"].items():
        if not st["seeds"]:
            print(f"\n=== stratum {name}: no complete seed ===")
            continue
        tag = "DECISIVE" if st["decisive"] else "secondary, cannot kill or pass an arm"
        print(f"\n=== stratum {name} ({tag}), {st['seed_count']} seeds: "
              f"{', '.join(st['seeds'])} ===")
        print(f"{'arm':<5} {'rounds':>7} {'proposed':>9} {'accepted':>9} "
              f"{'accept':>9} {'ci95':<21} {'depth':>6} {'d_pt':>9} "
              f"{'d_ci95_pt':<21} verdict")
        for arm in args.arms:
            v = st["arms"][arm]
            lo, hi = v["pooled_ci95"]
            dlo, dhi = v["delta_acceptance_pt_ci95"]
            verdict = ("control" if arm == args.control_arm
                       else "KILLED" if v["killed_by_acceptance"]
                       else "survives" if st["decisive"] else "informative only")
            print(f"{arm:<5} {v['pooled_rounds']:>7} {v['pooled_proposed']:>9} "
                  f"{v['pooled_accepted']:>9} {v['pooled_acceptance_rate']:>9.6f} "
                  f"[{lo:.6f},{hi:.6f}] {v['pooled_mean_depth']:>6.3f} "
                  f"{v['delta_acceptance_pt_vs_control']:>+9.4f} "
                  f"[{dlo:>+8.4f},{dhi:>+8.4f}] {verdict}")

        print("\nper-position acceptance, "
              "P(pos j accepted | j proposed and 1..j-1 accepted)")
        print("arm   " + "".join(f"{j:>9}" for j in range(1, MAX_POS + 1)))
        for arm in args.arms:
            cells = "".join(f"{c['p']:>9.4f}" if c.get("eligible") else f"{'-':>9}"
                            for c in st["arms"][arm]["conditionals"])
            print(f"{arm:<5} " + cells)
        print("elig  " + "".join(
            f"{c['eligible']:>9}" for c in st["arms"][args.control_arm]["conditionals"]))

        if st["decisive"] and st["seed_count"] < 3:
            print("\nfewer than 3 seeds in the decisive stratum: the per-seed "
                  "deltas below are DIRECTIONAL, not a stratum verdict")
        print("\nper seed, delta against the control in absolute points")
        print(f"{'seed':<20} " + "".join(
            f"{a:>26}" for a in args.arms if a != args.control_arm))
        for s in st["seeds"]:
            cells = ""
            for arm in args.arms:
                if arm == args.control_arm:
                    continue
                ps = st["arms"][arm]["per_seed"][s]
                lo, hi = ps["delta_acceptance_pt_ci95"]
                cells += (f"{ps['delta_acceptance_pt_vs_control']:>+9.4f}"
                          f"[{lo:>+7.3f},{hi:>+7.3f}]")
            print(f"{s:<20} " + cells)

        print("\nexactness cross-check against the control, per seed")
        for arm in args.arms:
            if arm == args.control_arm:
                continue
            for s in st["seeds"]:
                ps = st["arms"][arm]["per_seed"][s]
                print(f"{arm:<5} {s:<20} rows={ps['rows']:<5} "
                      f"compared={ps['rows_compared']:<5} "
                      f"top1_mismatch={ps['row_top1_mismatch_vs_control']:<5} "
                      f"pos_mismatch={ps['row_pos_mismatch_vs_control']:<5} "
                      f"value_mismatch={ps['row_value_mismatch_vs_control']:<5} "
                      f"matched={ps['all_tokens_matched']} "
                      f"div={ps['residual_divergence_count']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-template",
                    default=".mlxfast-private/e122/runs-e124-arm-{arm}/{seed}")
    ap.add_argument("--arms", nargs="+", default=["all", "none", "q", "kv"])
    ap.add_argument("--control-arm", default="all")
    ap.add_argument("--stratum", action="append", required=True,
                    metavar="NAME=id,id,...")
    ap.add_argument("--decisive-stratum", default="H")
    ap.add_argument("--kill-line-pt", type=float, default=0.21)
    ap.add_argument("--out", default="research/out/e124-acceptance.json")
    args = ap.parse_args()

    strata = {}
    for spec in args.stratum:
        name, _, ids = spec.partition("=")
        strata[name] = [i for i in ids.split(",") if i]

    rng = random.Random(SEED)
    result = {
        "control_arm": args.control_arm,
        "decisive_stratum": args.decisive_stratum,
        "kill_line_pt": args.kill_line_pt,
        "bootstrap_resamples": BOOTSTRAP,
        "seed": SEED,
        "strata": {},
    }

    for name, seeds in strata.items():
        present, legs = [], {}
        for seed in seeds:
            arms_here = {}
            for arm in args.arms:
                run_dir = Path(args.runs_template.format(arm=arm, seed=seed))
                if (run_dir / "trace.txt").exists():
                    arms_here[arm] = load_leg(arm, seed, run_dir, rng)
            if len(arms_here) == len(args.arms):
                present.append(seed)
                legs[seed] = arms_here
            else:
                missing = sorted(set(args.arms) - set(arms_here))
                print(f"stratum {name}: skipping {seed}, missing arms {missing}")
        result["strata"][name] = analyse_stratum(name, present, legs, args)

    write_and_print(result, args)


if __name__ == "__main__":
    main()
