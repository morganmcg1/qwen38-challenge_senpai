#!/usr/bin/env python3
"""E174 screen: witness the four arms, then price two independent contrasts.

`harness=local`. Both contrasts are within one counterbalanced gated session on
one host and one worker binary. No ranked transfer is claimed.

  o - s   replace 127 fused-norm epilogues with 127 standalone fills and their
          host kernel records. E174's mechanism, sign flipped.
  f - r   add 257 standalone fills and their host kernel records with the
          consumer held fixed. One dispatch plus one kernel record, together.

The serial leg is the null control: `Qwen35CustomQMV.routable` refuses at M = 1
and `Qwen35XSumsSidecar.wants` needs `tablePays(m) >= 4`, so no arm can reach
the serial leg. A serial effect that is not small next to the MTP effect means
the session drifted and the MTP number is not trustworthy.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "out"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

ARMS = ("s", "o", "r", "f")
PALINDROME = ("s", "o", "r", "f", "f", "r", "o", "s")

# What each arm must show in the per-round census, from the source: `sumtable`
# takes the table branch and either hits the sidecar or fills; `replica` never
# reaches the table branch at all; `fill_noconsume` always fills.
EXPECTED_CENSUS = {
    "s": {"hit": 127, "fill": 130},
    "o": {"hit": 0, "fill": 257},
    "r": {"hit": 0, "fill": 0},
    "f": {"hit": 0, "fill": 257},
}

CONTRASTS = {
    "o_minus_s": {
        "base": "s",
        "test": "o",
        "cells": 127,
        "mechanism": "replace 127 fused-norm epilogues with 127 standalone "
        "fills and their host kernel records",
    },
    "f_minus_r": {
        "base": "r",
        "test": "f",
        "cells": 257,
        "mechanism": "add 257 standalone fills and their host kernel records, "
        "consumer held fixed",
    },
}


def meta(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def census(path: pathlib.Path) -> dict[str, object]:
    """Per-round `xs_hit` and `xs_fill` from a traced leg."""
    seen = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("mtp-trace: round="):
                continue
            got = {}
            for key in ("round", "xs_hit", "xs_fill"):
                m = re.search(rf"\b{key}=(\d+)", line)
                if m:
                    got[key] = int(m.group(1))
            if len(got) == 3:
                seen.append(got)
    if len(seen) < 3:
        return {"rounds_traced": len(seen)}
    hits = [b["xs_hit"] - a["xs_hit"] for a, b in zip(seen, seen[1:])]
    fills = [b["xs_fill"] - a["xs_fill"] for a, b in zip(seen, seen[1:])]
    return {
        "rounds_traced": len(seen),
        "hit": statistics.median(hits),
        "fill": statistics.median(fills),
        "hit_set": sorted(set(hits)),
        "fill_set": sorted(set(fills)),
    }


def leg(tag: str) -> dict[str, object] | None:
    out = OUT / tag
    score_path = out / "score.json"
    if not score_path.exists():
        return None
    score = json.loads(score_path.read_text())
    m = meta(out / "meta.txt")
    metrics = score.get("metrics", {})
    return {
        "tag": tag,
        "arm": m.get("e174_arm"),
        "position": int(m.get("e174_position", 0) or 0),
        "mtp_spt": metrics.get("mtp_seconds_per_token"),
        "serial_spt": metrics.get("serial_seconds_per_token"),
        "edl": metrics.get("effective_mean_draft_len"),
        "accept": metrics.get("accepted_draft_rate"),
        "matched": metrics.get("all_tokens_matched"),
        "entry_c": float(m.get("gpu_temp_entry_c") or "nan"),
        "exit_c": float(m.get("gpu_temp_exit_c") or "nan"),
        "cool_gate_passed_real_gate": m.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": m.get("gate_qualified_for_timing"),
        "worker_sha256": m.get("worker_sha256"),
        "base_sha": m.get("base_sha"),
        "dirty": m.get("dirty_candidate_paths"),
        "host": m.get("host"),
        "chip": m.get("chip"),
        "tokens": m.get("tokens"),
        "census": census(out / "trace.txt"),
    }


def wandb_run(label: str):
    """Resume the session's single W&B run, creating it on the first call.

    RULE 374: a session longer than one hour publishes each leg as that leg
    finishes, so a session that dies at leg six still leaves five legs of
    evidence. Every leg is a separate process, so the run id is held on disk
    and every call resumes the same run.
    """
    import wandb

    path = OUT / f"e174-screen-{label}-wandb.txt"
    run_id = path.read_text().strip() if path.exists() else wandb.util.generate_id()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow",
        name=f"e174-screen-{label}",
        job_type="screen",
        tags=["e174", "harness:local", "xsums", "fill-coefficient", "gated"],
        config={
            "experiment": "e174-xsums-epilogue-extension",
            "harness": "local",
            "official_or_ranked_score": False,
            "design": "palindrome s o r f f r o s, four arms, two contrasts",
            "arms": {
                "s": "shipped sumtable: 127 epilogues, 130 standalone fills",
                "o": "MLX_E174_XSUMS_SIDECAR=off: 0 epilogues, 257 fills",
                "r": "MLX_E120_QMV_ARM=replica: 0 fills, no consume",
                "f": "MLX_E120_QMV_ARM=fill_noconsume: 257 fills, no consume",
            },
            "contrasts": {k: v["mechanism"] for k, v in CONTRASTS.items()},
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id + "\n")
    return run


def publish_leg(label: str, tag: str) -> int:
    entry = leg(tag)
    if entry is None:
        print(f"e174_screen: no score.json for {tag}; nothing to publish")
        return 0
    run = wandb_run(label)
    census_stats = entry.pop("census", {})
    payload = {f"leg/{k}": v for k, v in entry.items() if not isinstance(v, dict)}
    payload.update({f"census/{k}": v for k, v in census_stats.items()
                    if not isinstance(v, list)})
    payload["leg/arm_index"] = ARMS.index(entry["arm"]) if entry["arm"] in ARMS else -1
    run.log(payload)
    run.finish()
    print(f"e174_screen: published {tag} to run {run.id}")
    return 0


def witness(label: str) -> int:
    report = {"step": "witness", "harness": "local", "arms": {}}
    ok = True
    for arm in ARMS:
        entry = leg(f"e174{label}w{arm}") or {}
        c = entry.get("census", {})
        want = EXPECTED_CENSUS[arm]
        arm_ok = c.get("hit") == want["hit"] and c.get("fill") == want["fill"]
        # The control: this arm's census must fail every other arm's
        # expectation, otherwise the switch never reached the worker.
        distinct = all(
            not (c.get("hit") == other["hit"] and c.get("fill") == other["fill"])
            for name, other in EXPECTED_CENSUS.items()
            if name != arm and other != want
        )
        ok = ok and arm_ok and distinct
        report["arms"][arm] = {
            "tag": entry.get("tag"),
            "expected": want,
            "observed": {"hit": c.get("hit"), "fill": c.get("fill")},
            "rounds_traced": c.get("rounds_traced"),
            "passes_own_expectation": arm_ok,
            "fails_every_other_expectation": distinct,
            "all_tokens_matched": entry.get("matched"),
        }
    report["passed"] = ok
    path = OUT / f"e174-witness-{label}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def fit(legs: list[dict], test_arm: str) -> dict[str, float]:
    """Least squares of `mtp_spt ~ intercept + effect*is_test + drift*position`.

    Returns the effect as a percentage of the base arm's mean, with the
    standard error from the residuals. With four legs and three parameters the
    drift-fitted model has one degree of freedom, so the no-drift model is
    reported beside it exactly as E135 did.
    """
    ys = [x["mtp_spt"] for x in legs]
    ts = [1.0 if x["arm"] == test_arm else 0.0 for x in legs]
    ps = [float(x["position"]) for x in legs]
    base_mean = statistics.fmean(
        [y for y, t in zip(ys, ts) if t == 0.0]
    )

    def ols(cols: list[list[float]]) -> tuple[list[float], float, int]:
        n = len(ys)
        k = len(cols)
        # Normal equations by Gaussian elimination; k is 2 or 3 here.
        a = [[sum(ci[i] * cj[i] for i in range(n)) for cj in cols] + [
            sum(ci[i] * ys[i] for i in range(n))
        ] for ci in cols]
        for col in range(k):
            piv = max(range(col, k), key=lambda r: abs(a[r][col]))
            a[col], a[piv] = a[piv], a[col]
            if abs(a[col][col]) < 1e-18:
                return [float("nan")] * k, float("nan"), 0
            for r in range(k):
                if r == col:
                    continue
                fct = a[r][col] / a[col][col]
                for c in range(col, k + 1):
                    a[r][c] -= fct * a[col][c]
        beta = [a[i][k] / a[i][i] for i in range(k)]
        resid = [
            ys[i] - sum(beta[j] * cols[j][i] for j in range(k)) for i in range(n)
        ]
        dof = n - k
        sigma2 = sum(r * r for r in resid) / dof if dof > 0 else float("nan")
        return beta, sigma2, dof

    ones = [1.0] * len(ys)
    out: dict[str, float] = {"base_mean_spt": base_mean}
    for name, cols in (
        ("no_drift", [ones, ts]),
        ("drift_fitted", [ones, ts, ps]),
    ):
        beta, sigma2, dof = ols(cols)
        # se of the effect coefficient: sqrt(sigma2 * (X'X)^-1 [1,1]).
        n = len(ys)
        k = len(cols)
        xtx = [[sum(cols[i][z] * cols[j][z] for z in range(n)) for j in range(k)]
               for i in range(k)]
        inv11 = invert_element(xtx, 1)
        se = math.sqrt(sigma2 * inv11) if sigma2 == sigma2 and inv11 == inv11 else float("nan")
        out[f"effect_pct_{name}"] = beta[1] / base_mean * 100
        out[f"se_pp_{name}"] = se / base_mean * 100
        out[f"dof_{name}"] = dof
        if name == "drift_fitted":
            out["drift_pct_per_leg"] = beta[2] / base_mean * 100
    return out


def invert_element(m: list[list[float]], idx: int) -> float:
    """Element `[idx][idx]` of the inverse of a small symmetric matrix."""
    k = len(m)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(m)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-18:
            return float("nan")
        d = aug[col][col]
        for c in range(2 * k):
            aug[col][c] /= d
        for r in range(k):
            if r == col:
                continue
            f = aug[r][col]
            for c in range(2 * k):
                aug[r][c] -= f * aug[col][c]
    return aug[idx][k + idx]


def report(label: str) -> int:
    legs = []
    for position, arm in enumerate(PALINDROME, start=1):
        entry = leg(f"e174{label}p{position}{arm}")
        if entry:
            legs.append(entry)

    out: dict[str, object] = {
        "experiment": "e174-xsums-epilogue-extension",
        "step": "screen",
        "harness": "local",
        "label": label,
        "legs": legs,
        "contrasts": {},
    }

    if legs:
        out["identity"] = {
            "base_sha": legs[0]["base_sha"],
            "worker_sha256": legs[0]["worker_sha256"],
            "host": legs[0]["host"],
            "chip": legs[0]["chip"],
            "tokens": legs[0]["tokens"],
            "dirty_candidate_paths": legs[0]["dirty"],
            "one_worker_all_legs": len({x["worker_sha256"] for x in legs}) == 1,
            "all_legs_gate_qualified": all(
                x["gate_qualified_for_timing"] == "true" for x in legs
            ),
            "all_tokens_matched": all(x["matched"] for x in legs),
            "entry_c_spread": (
                max(x["entry_c"] for x in legs) - min(x["entry_c"] for x in legs)
            ),
        }

    print("=== legs ===")
    print(
        f"{'tag':<20}{'arm':<5}{'pos':<5}{'mtp s/tok':<14}{'serial s/tok':<14}"
        f"{'edl':<10}{'accept':<11}{'entry_C':<10}{'exit_C':<10}{'matched'}"
    )
    for x in legs:
        print(
            f"{x['tag']:<20}{x['arm']:<5}{x['position']:<5}{x['mtp_spt']:<14.8f}"
            f"{x['serial_spt']:<14.8f}{x['edl']:<10.6f}{x['accept']:<11.8f}"
            f"{x['entry_c']:<10.2f}{x['exit_c']:<10.2f}{x['matched']}"
        )

    for name, spec in CONTRASTS.items():
        pair = [x for x in legs if x["arm"] in (spec["base"], spec["test"])]
        if len(pair) < 4:
            out["contrasts"][name] = {"error": f"only {len(pair)} legs"}
            continue
        edls = {round(x["edl"], 6) for x in pair}
        accepts = {round(x["accept"], 8) for x in pair}
        schedule_identical = len(edls) == 1 and len(accepts) == 1
        result = {
            "mechanism": spec["mechanism"],
            "cells": spec["cells"],
            "legs": [x["tag"] for x in pair],
            "schedule_identical": schedule_identical,
            "edl_values": sorted(edls),
            "accept_values": sorted(accepts),
        }
        if not schedule_identical:
            result["error"] = (
                "arms disagree on the schedule; this is not a pure cost contrast"
            )
        else:
            mtp = fit(pair, spec["test"])
            serial_pair = [dict(x, mtp_spt=x["serial_spt"]) for x in pair]
            serial = fit(serial_pair, spec["test"])
            round_us = statistics.fmean([x["mtp_spt"] for x in pair]) * statistics.fmean(
                [x["edl"] for x in pair]
            ) * 1e6
            effect = mtp["effect_pct_no_drift"]
            se = max(mtp["se_pp_no_drift"], mtp["se_pp_drift_fitted"])
            result.update(
                {
                    "mtp": mtp,
                    "serial_null": serial,
                    "local_round_us": round_us,
                    "us_per_round": round_us * effect / 100,
                    "us_per_cell": round_us * effect / 100 / spec["cells"],
                    "us_per_cell_2se_upper": round_us
                    * (effect + 2 * se)
                    / 100
                    / spec["cells"],
                    "pct_2se_interval": [effect - 2 * se, effect + 2 * se],
                }
            )
        out["contrasts"][name] = result

    print("\n=== contrasts ===")
    for name, r in out["contrasts"].items():
        print(f"\n{name}: {r.get('mechanism', r.get('error'))}")
        if "mtp" in r:
            m = r["mtp"]
            print(
                f"  effect {m['effect_pct_no_drift']:+.4f} % "
                f"(no drift, se {m['se_pp_no_drift']:.4f} pp, "
                f"{m['dof_no_drift']:.0f} dof)"
            )
            print(
                f"  effect {m['effect_pct_drift_fitted']:+.4f} % "
                f"(drift fitted, se {m['se_pp_drift_fitted']:.4f} pp, "
                f"{m['dof_drift_fitted']:.0f} dof, "
                f"drift {m['drift_pct_per_leg']:+.4f} %/leg)"
            )
            print(
                f"  serial null {r['serial_null']['effect_pct_no_drift']:+.4f} %"
            )
            print(
                f"  {r['us_per_round']:+.1f} us/round over {r['cells']} cells "
                f"= {r['us_per_cell']:+.4f} us/cell "
                f"(2se upper {r['us_per_cell_2se_upper']:+.4f}), "
                f"local round {r['local_round_us']:.0f} us"
            )

    path = OUT / f"e174-screen-{label}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")

    run = wandb_run(label)
    run.summary["screen"] = out
    for name, r in out["contrasts"].items():
        if "mtp" not in r:
            continue
        run.summary[f"{name}/effect_pct"] = r["mtp"]["effect_pct_no_drift"]
        run.summary[f"{name}/se_pp"] = r["mtp"]["se_pp_no_drift"]
        run.summary[f"{name}/us_per_cell"] = r["us_per_cell"]
        run.summary[f"{name}/us_per_cell_2se_upper"] = r["us_per_cell_2se_upper"]
        run.summary[f"{name}/serial_null_pct"] = r["serial_null"][
            "effect_pct_no_drift"
        ]
    run.finish()
    print(f"published the screen summary to run {run.id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["witness", "report", "leg"])
    ap.add_argument("--label", default="s1")
    ap.add_argument("--tag")
    args = ap.parse_args()
    if args.command == "leg":
        return publish_leg(args.label, args.tag)
    return witness(args.label) if args.command == "witness" else report(args.label)


if __name__ == "__main__":
    raise SystemExit(main())
