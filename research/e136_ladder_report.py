#!/usr/bin/env python3
"""Report one E136 launch-column ladder session and fit the three cost laws.

THE QUESTION. E135 deleted no-op QMV threadgroup columns and moved absolute
candidate MTP seconds per token by 1.806 %. It cannot say what a column costs.
`wide -> tight` changed the column count by a different factor at every width,
so a flat cost per drafting round, a linear cost in the launched column count
and a logarithmic cost in the column ratio all fit that one point.

THE READOUT. `tightN` launches N times the working column count at every
routed width and buys no arithmetic, so the launched column census per leg is
known exactly from the plan and the fixture's own per-round width histogram.
Each rung's absolute candidate MTP seconds per token is converted to
microseconds per drafting round and regressed against three predictors:

  flat     1 per round, so a constant; the ladder must be flat
  linear   launched columns per round
  log      sum of ln(columns) over the round's routed dispatch columns

The fits are compared on residual standard deviation against the pure
replicate noise of the session, so a law is rejected when its residual is
larger than the noise it must explain, not merely because another law fits
better.

DRIFT. One replicate is a palindrome, so every arm has the same mean position
and a monotone drift in leg index cancels to first order in every contrast.
The least-squares fit also carries a centred global leg index, and the paired
palindrome estimate is printed next to it as an independent check.

GATE LABELS. The session is ungated by construction. Entry and exit
temperature per leg and the entry spread per arm are printed, and
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are reproduced
verbatim: an ungated reading is directional evidence inside its own
counterbalanced session and nothing more.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import statistics

OUT = pathlib.Path("research/out")
ARMS = ("wide", "tight", "tight2", "tight4", "tight8")


def pad_factor(arm: str) -> int:
    return int(arm[len("tight"):]) if arm.startswith("tight") and arm != "tight" else 1


def read_meta(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def plan_from(witness: str) -> dict[int, int]:
    body = witness.split("/", 1)[1]
    return {
        int(cell.split(":")[0]): int(cell.split(":")[1]) for cell in body.split(",")
    }


def width_histogram(trace: pathlib.Path) -> collections.Counter[int]:
    """Verify width per round is the draft count plus the committed primary."""
    hist: collections.Counter[int] = collections.Counter()
    for line in trace.read_text().splitlines():
        if line.startswith("mtp-trace: round="):
            hist[int(re.search(r" d=(\d+)", line).group(1)) + 1] += 1
    return hist


def census(hist: collections.Counter[int], plan: dict[int, int]) -> dict[str, dict]:
    """Launched columns per leg, and the log predictor, for every rung."""
    out = {}
    for arm in ARMS:
        cols = 0
        logsum = 0.0
        for m, count in hist.items():
            if m not in plan:  # `default: break`, no routed QMV at this width
                continue
            working = -(-m // plan[m])
            launched = m if arm == "wide" else working * pad_factor(arm)
            cols += count * launched
            logsum += count * math.log(launched)
        out[arm] = {"columns": cols, "logsum": logsum}
    return out


def legs(label: str, phase: str) -> list[dict]:
    found = []
    for d in sorted(OUT.glob(f"e136{label}k*")) + sorted(OUT.glob(f"e136{label}t*")):
        meta = read_meta(d / "meta.txt")
        if meta.get("e136_phase") != phase:
            continue
        arm = meta.get("e136_grid")
        if arm not in ARMS:
            continue
        score = d / "score.json"
        if not score.exists():
            print(f"  !! {d.name}: no score.json")
            continue
        metrics = json.loads(score.read_text()).get("metrics", {})
        if not metrics.get("all_tokens_matched", False):
            print(f"  !! {d.name}: all_tokens_matched is not true")
        found.append({"dir": d, "meta": meta, "metrics": metrics, "arm": arm})
    return found


def solve(rows: list[list[float]], rhs: list[float]) -> list[float]:
    """Least squares by normal equations with Gaussian elimination."""
    p = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(p)] + [
        sum(r[i] * y for r, y in zip(rows, rhs))] for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        for r in range(p):
            if r == col:
                continue
            f = a[r][col]
            a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [a[i][p] for i in range(p)]


def fit(name: str, predictor: dict[str, float] | None,
        data: list[tuple[str, int, float]], noise: float) -> dict:
    """One drift-corrected least-squares fit of round microseconds.

    `predictor` of `None` is the flat law: intercept and drift only, so its
    design matrix stays full rank instead of carrying an all-zero column.
    """
    centre = statistics.mean(i for _, i, _ in data)
    if predictor is None:
        design = [[1.0, i - centre] for _, i, _ in data]
    else:
        design = [[1.0, predictor[a], i - centre] for a, i, _ in data]
    y = [v for _, _, v in data]
    beta = solve(design, y)
    resid = [v - sum(b * x for b, x in zip(beta, row)) for row, v in zip(design, y)]
    dof = len(y) - len(beta)
    sd = math.sqrt(sum(r * r for r in resid) / dof) if dof > 0 else float("nan")
    return {
        "name": name,
        "slope": 0.0 if predictor is None else beta[1],
        "intercept": beta[0],
        "drift": beta[-1],
        "residual_sd": sd,
        "ratio_to_noise": sd / noise if noise else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="L1")
    ap.add_argument(
        "--trace-census",
        default="research/out/e135x512/trace.txt",
        help="traced leg of the same fixture and token window, used only for "
             "the per-round width histogram; the token stream is bit identical "
             "across rungs so one histogram serves the whole session")
    ap.add_argument("--json", help="write the machine-readable summary here")
    args = ap.parse_args()

    rows = legs(args.label, "ladder")
    if not rows:
        print("e136_ladder_report: no ladder legs found")
        return 3

    # The plan witness lives in the pipeline log the witness legs recorded.
    plan = None
    for arm in ARMS:
        p = OUT / f"e136{args.label}w{arm}" / "pipelines.json"
        if p.exists():
            plan = plan_from(json.loads(p.read_text())["plan"])
            break
    if plan is None:
        print("e136_ladder_report: no witness pipeline log, cannot read the plan")
        return 3

    hist = width_histogram(pathlib.Path(args.trace_census))
    rounds = sum(hist.values())
    cen = census(hist, plan)

    print(f"e136 ladder session {args.label}")
    print(f"  plan                {plan}")
    print(f"  round width census  {dict(sorted(hist.items()))}  rounds={rounds}")
    print()
    print("  rung      launched columns/leg   columns/round   sum ln(cols)/leg")
    for arm in ARMS:
        c = cen[arm]
        print(f"  {arm:8s}  {c['columns']:19d}   {c['columns']/rounds:13.3f}"
              f"   {c['logsum']:16.2f}")
    print()

    workers = {r["meta"].get("worker_sha256", "?") for r in rows}
    commits = {r["meta"].get("e136_session_commit", "?") for r in rows}
    print(f"  legs {len(rows)}   workers {sorted(workers)}   commits {sorted(commits)}")
    if len(workers) != 1:
        print("  !! the session did not time one binary")
    gate = {r["meta"].get("gate_qualified_for_timing", "?") for r in rows}
    real = {r["meta"].get("cool_gate_passed_real_gate", "?") for r in rows}
    print(f"  gate_qualified_for_timing {sorted(gate)}"
          f"   cool_gate_passed_real_gate {sorted(real)}")
    print()

    tokens = int(rows[0]["metrics"]["decode_tokens"])
    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    def us_per_round(spt: float) -> float:
        return spt * tokens / rounds * 1e6

    print(f"  per-arm absolute candidate MTP time ({tokens} tokens, "
          f"{rounds} rounds/leg)")
    print("  rung      n   mtp s/token      sd          us/round     "
          "vs tight     entry C")
    base = statistics.mean(m["metrics"]["mtp_seconds_per_token"]
                           for m in by_arm["tight"]) if by_arm["tight"] else None
    summary = {}
    for arm in ARMS:
        arm_rows = by_arm.get(arm, [])
        if not arm_rows:
            continue
        v = [m["metrics"]["mtp_seconds_per_token"] for m in arm_rows]
        temps = [float(m["meta"].get("gpu_temp_entry_c", "nan")) for m in arm_rows]
        sd = statistics.stdev(v) if len(v) > 1 else float("nan")
        rel = (statistics.mean(v) / base - 1.0) * 100.0 if base else float("nan")
        print(f"  {arm:8s} {len(v):2d}   {statistics.mean(v):.9f}  {sd:.9f}  "
              f"{us_per_round(statistics.mean(v)):11.1f}  {rel:+8.3f} %   "
              f"{min(temps):.1f}-{max(temps):.1f}")
        summary[arm] = {
            "n": len(v),
            "mtp_seconds_per_token_mean": statistics.mean(v),
            "mtp_seconds_per_token_sd": sd,
            "us_per_round": us_per_round(statistics.mean(v)),
            "pct_vs_tight": rel,
            "launched_columns_per_leg": cen[arm]["columns"],
            "entry_temp_c_min": min(temps),
            "entry_temp_c_max": max(temps),
        }
    print()

    # Exactness across the whole session: the rungs must produce one token
    # stream, so these two derived numbers must be digit identical everywhere.
    for key in ("effective_mean_draft_len", "accepted_draft_rate",
                "residual_divergence_count", "all_tokens_matched"):
        seen = {repr(r["metrics"].get(key)) for r in rows}
        flag = "" if len(seen) == 1 else "   !! NOT IDENTICAL ACROSS RUNGS"
        print(f"  {key:28s} {sorted(seen)}{flag}")
    print()

    # The three laws, fitted on the padded rungs only. `wide` is held out as an
    # anchor so it can test each law out of sample.
    ladder_rows = [(r["arm"], int(r["meta"]["e136_leg_index"]),
                    us_per_round(r["metrics"]["mtp_seconds_per_token"]))
                   for r in rows if r["arm"] != "wide"]
    noise = statistics.mean(
        statistics.stdev([us_per_round(m["metrics"]["mtp_seconds_per_token"])
                          for m in by_arm[a]])
        for a in ARMS if a != "wide" and len(by_arm.get(a, [])) > 1)

    fits = [
        fit("flat    (constant)", None, ladder_rows, noise),
        fit("linear  (columns) ",
            {a: cen[a]["columns"] / rounds for a in ARMS}, ladder_rows, noise),
        fit("log     (ln cols) ",
            {a: cen[a]["logsum"] / rounds for a in ARMS}, ladder_rows, noise),
    ]
    print(f"  pooled within-arm noise on us/round   {noise:.1f}")
    print("  law                    slope            residual sd   sd/noise"
          "   wide out-of-sample")
    wide_us = (us_per_round(statistics.mean(
        m["metrics"]["mtp_seconds_per_token"] for m in by_arm["wide"]))
        if by_arm.get("wide") else None)
    tight_us = us_per_round(base) if base else None
    for f in fits:
        pred = ""
        if wide_us is not None and tight_us is not None:
            key = ("columns" if "linear" in f["name"]
                   else "logsum" if "log" in f["name"] else None)
            if key is None:
                want = tight_us
            else:
                want = (tight_us + f["slope"]
                        * (cen["wide"][key] - cen["tight"][key]) / rounds)
            f["wide_predicted_us_per_round"] = want
            f["wide_out_of_sample_error_us"] = wide_us - want
            pred = f"   predict {want:9.1f} vs {wide_us:9.1f}  ({wide_us - want:+8.1f})"
        print(f"  {f['name']}  {f['slope']:14.4f}   {f['residual_sd']:10.1f}"
              f"   {f['ratio_to_noise']:7.2f}{pred}")
    print()

    # Paired palindrome contrast, which needs no model at all.
    print("  paired contrast against tight inside each replicate")
    reps = sorted({int(r["meta"]["e136_replicate"]) for r in rows})
    for arm in ARMS:
        if arm == "tight":
            continue
        deltas = []
        for rep in reps:
            a = [us_per_round(r["metrics"]["mtp_seconds_per_token"])
                 for r in by_arm.get(arm, []) if int(r["meta"]["e136_replicate"]) == rep]
            t = [us_per_round(r["metrics"]["mtp_seconds_per_token"])
                 for r in by_arm["tight"] if int(r["meta"]["e136_replicate"]) == rep]
            if a and t:
                deltas.append(statistics.mean(a) - statistics.mean(t))
        if not deltas:
            continue
        sd = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
        added = (cen[arm]["columns"] - cen["tight"]["columns"]) / rounds
        per_col = statistics.mean(deltas) / added if added else float("nan")
        print(f"  {arm:8s} {statistics.mean(deltas):+10.1f} us/round   "
              f"sd {sd:8.1f}   n {len(deltas)}   "
              f"{added:+8.3f} columns/round   {per_col:+8.2f} us/column")

    traced = legs(args.label, "trace")
    if traced:
        print()
        print("  traced attribution legs, per-round phase means in us")
        phases = ("draft_build_us", "verify_build_us", "eval_wall_us",
                  "readout_us", "commit_us", "upkeep_us", "round_us")
        print("  rung      n  " + "".join(f"{p:>17s}" for p in phases))
        tby: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
        for r in traced:
            t = r["dir"] / "trace.txt"
            if not t.exists():
                continue
            acc = {p: [] for p in phases}
            for line in t.read_text().splitlines():
                if not line.startswith("mtp-trace: round="):
                    continue
                for p in phases:
                    mm = re.search(rf" {p}=(\d+)", line)
                    if mm:
                        acc[p].append(int(mm.group(1)))
            tby[r["arm"]].append({p: statistics.mean(v) for p, v in acc.items() if v})
        for arm in ARMS:
            if arm not in tby:
                continue
            cells = "".join(
                f"{statistics.mean(d[p] for d in tby[arm]):17.1f}" for p in phases)
            print(f"  {arm:8s} {len(tby[arm]):2d}  {cells}")
        if "tight" in tby and "tight8" in tby:
            print("  delta     " + "".join(
                f"{statistics.mean(d[p] for d in tby['tight8']) - statistics.mean(d[p] for d in tby['tight']):17.1f}"
                for p in phases))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "label": args.label,
            "rounds": rounds,
            "tokens": tokens,
            "width_histogram": {str(k): v for k, v in sorted(hist.items())},
            "census": cen,
            "arms": summary,
            "fits": fits,
            "pooled_noise_us_per_round": noise,
            "wide_us_per_round": wide_us,
            "tight_us_per_round": tight_us,
        }, indent=2, default=str))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
