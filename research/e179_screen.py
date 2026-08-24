#!/usr/bin/env python3
"""E179 screen: price the launch-config cache end to end.

`harness=local`. One counterbalanced gated session, one host, one worker
binary. No ranked transfer is claimed.

  n - c   remove the cache and rebuild an `mlx_fast_metal_kernel_config` on
          every one of the round's 387 kernel records. E179's mechanism, sign
          flipped, so a positive `n - c` effect means the cache is faster.

The serial leg is the null control: `Qwen35CustomQMV.routable` refuses at M = 1,
so no arm can reach the serial leg. A serial effect that is not small next to
the MTP effect means the session drifted and the MTP number is not trustworthy.

The `fit`, `invert_element`, `meta` and `leg` helpers follow E174's screen so
the two sessions are read the same way.
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

ARMS = ("c", "n")
PALINDROME = ("c", "n", "n", "c", "c", "n", "n", "c")

# Kernel records of one decode round, from the E174 census: 257 table-paying
# QMV cells plus 130 standalone chunk-sum fills.
RECORDS_PER_ROUND = 387

CONTRAST = {
    "n_minus_c": {
        "base": "c",
        "test": "n",
        "cells": RECORDS_PER_ROUND,
        "mechanism": "rebuild the launch config on every one of the round's "
        "387 kernel records instead of serving it from the geometry cache",
    }
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
    """Per-round `cfg_hit` and `cfg_miss` deltas from a traced leg."""
    seen = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("mtp-trace: round="):
                continue
            got = {}
            for key in ("round", "cfg", "cfg_hit", "cfg_miss"):
                m = re.search(rf"\b{key}=(\d+)", line)
                if m:
                    got[key] = int(m.group(1))
            if len(got) == 4:
                seen.append(got)
    if len(seen) < 3:
        return {"rounds_traced": len(seen)}
    hits = [b["cfg_hit"] - a["cfg_hit"] for a, b in zip(seen, seen[1:])]
    misses = [b["cfg_miss"] - a["cfg_miss"] for a, b in zip(seen, seen[1:])]
    return {
        "rounds_traced": len(seen),
        "arm_flag": seen[-1]["cfg"],
        "hit": statistics.median(hits),
        "miss": statistics.median(misses),
        "hit_set": sorted(set(hits)),
        "miss_set": sorted(set(misses)),
        "distinct_geometries": seen[-1]["cfg_miss"],
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
        "arm": m.get("e179_arm"),
        "position": int(m.get("e179_position", 0) or 0),
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
    """Resume the session's single W&B run, creating it on the first call."""
    import wandb

    path = OUT / f"e179-screen-{label}-wandb.txt"
    run_id = path.read_text().strip() if path.exists() else wandb.util.generate_id()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow",
        name=f"e179-screen-{label}",
        job_type="screen",
        tags=["e179", "harness:local", "config-cache", "host-cost", "gated"],
        config={
            "experiment": "e179-replica-config-cache",
            "harness": "local",
            "official_or_ranked_score": False,
            "design": "palindrome c n n c c n n c, two arms, one contrast",
            "arms": {
                "c": "shipped: launch configs served from the geometry cache",
                "n": "MLX_E179_CFG_CACHE_ARM=off: incumbent MLXFastKernel launcher",
            },
            "records_per_round": RECORDS_PER_ROUND,
            "contrast": CONTRAST["n_minus_c"]["mechanism"],
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id + "\n")
    return run


def publish_leg(label: str, tag: str) -> int:
    entry = leg(tag)
    if entry is None:
        print(f"e179_screen: no score.json for {tag}; nothing to publish")
        return 0
    run = wandb_run(label)
    census_stats = entry.pop("census", {})
    payload = {f"leg/{k}": v for k, v in entry.items() if not isinstance(v, dict)}
    payload.update(
        {f"census/{k}": v for k, v in census_stats.items() if not isinstance(v, list)}
    )
    payload["leg/arm_index"] = ARMS.index(entry["arm"]) if entry["arm"] in ARMS else -1
    run.log(payload)
    run.finish()
    print(f"e179_screen: published {tag} to run {run.id}")
    return 0


def witness(label: str) -> int:
    """Both arms reached the worker, and the cache behaves as declared.

    `c` must serve nearly every record from the cache and must stop taking
    misses once the geometry set is warm. `n` must never touch the cache. The
    two arms must emit the same tokens with the same schedule.
    """
    report = {"step": "witness", "harness": "local", "arms": {}}
    entries = {}
    ok = True
    for arm in ARMS:
        entry = leg(f"e179{label}w{arm}") or {}
        entries[arm] = entry
        c = entry.get("census", {})
        hit = c.get("hit")
        miss = c.get("miss")
        if arm == "c":
            arm_ok = (
                c.get("arm_flag") == 1
                and isinstance(hit, (int, float))
                and hit > 0
                and miss == 0
            )
        else:
            arm_ok = c.get("arm_flag") == 0 and hit == 0 and miss == 0
        ok = ok and arm_ok and bool(entry.get("matched"))
        report["arms"][arm] = {
            "tag": entry.get("tag"),
            "observed": {
                "cfg": c.get("arm_flag"),
                "hit_per_round": hit,
                "miss_per_round": miss,
                "distinct_geometries": c.get("distinct_geometries"),
            },
            "rounds_traced": c.get("rounds_traced"),
            "passes_own_expectation": arm_ok,
            "all_tokens_matched": entry.get("matched"),
        }

    # The identity control: the two arms must agree digit for digit on the
    # schedule they produced. The cache changes host bookkeeping only.
    if all(entries.get(a) for a in ARMS):
        identical = (
            round(entries["c"]["edl"], 8) == round(entries["n"]["edl"], 8)
            and round(entries["c"]["accept"], 8) == round(entries["n"]["accept"], 8)
        )
        report["schedule_identical"] = identical
        report["edl"] = [entries[a]["edl"] for a in ARMS]
        report["accept"] = [entries[a]["accept"] for a in ARMS]
        ok = ok and identical
    else:
        report["schedule_identical"] = False
        ok = False

    report["passed"] = ok
    path = OUT / f"e179-witness-{label}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def fit(legs: list[dict], test_arm: str) -> dict[str, float]:
    """Least squares of `mtp_spt ~ intercept + effect*is_test + drift*position`."""
    ys = [x["mtp_spt"] for x in legs]
    ts = [1.0 if x["arm"] == test_arm else 0.0 for x in legs]
    ps = [float(x["position"]) for x in legs]
    base_mean = statistics.fmean([y for y, t in zip(ys, ts) if t == 0.0])

    def ols(cols: list[list[float]]) -> tuple[list[float], float, int]:
        n = len(ys)
        k = len(cols)
        a = [
            [sum(ci[i] * cj[i] for i in range(n)) for cj in cols]
            + [sum(ci[i] * ys[i] for i in range(n))]
            for ci in cols
        ]
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
        resid = [ys[i] - sum(beta[j] * cols[j][i] for j in range(k)) for i in range(n)]
        dof = n - k
        sigma2 = sum(r * r for r in resid) / dof if dof > 0 else float("nan")
        return beta, sigma2, dof

    ones = [1.0] * len(ys)
    out: dict[str, float] = {"base_mean_spt": base_mean}
    for name, cols in (("no_drift", [ones, ts]), ("drift_fitted", [ones, ts, ps])):
        beta, sigma2, dof = ols(cols)
        n = len(ys)
        k = len(cols)
        xtx = [
            [sum(cols[i][z] * cols[j][z] for z in range(n)) for j in range(k)]
            for i in range(k)
        ]
        inv11 = invert_element(xtx, 1)
        se = (
            math.sqrt(sigma2 * inv11)
            if sigma2 == sigma2 and inv11 == inv11
            else float("nan")
        )
        out[f"effect_pct_{name}"] = beta[1] / base_mean * 100
        out[f"se_pp_{name}"] = se / base_mean * 100
        out[f"dof_{name}"] = dof
        if name == "drift_fitted":
            out["drift_pct_per_leg"] = beta[2] / base_mean * 100
    return out


def invert_element(m: list[list[float]], idx: int) -> float:
    """Element `[idx][idx]` of the inverse of a small symmetric matrix."""
    k = len(m)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(m)
    ]
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
        entry = leg(f"e179{label}p{position}{arm}")
        if entry:
            legs.append(entry)

    out: dict[str, object] = {
        "experiment": "e179-replica-config-cache",
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

    for name, spec in CONTRAST.items():
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
            round_us = (
                statistics.fmean([x["mtp_spt"] for x in pair])
                * statistics.fmean([x["edl"] for x in pair])
                * 1e6
            )
            effect = mtp["effect_pct_no_drift"]
            se = max(mtp["se_pp_no_drift"], mtp["se_pp_drift_fitted"])
            result.update(
                {
                    "mtp": mtp,
                    "serial_null": serial,
                    "local_round_us": round_us,
                    "us_per_round": round_us * effect / 100,
                    "us_per_record": round_us * effect / 100 / spec["cells"],
                    "pct_2se_interval": [effect - 2 * se, effect + 2 * se],
                    "clears_mue_0p30pct": effect - 2 * se >= 0.30,
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
            print(f"  serial null {r['serial_null']['effect_pct_no_drift']:+.4f} %")
            print(
                f"  {r['us_per_round']:+.1f} us/round over {r['cells']} records "
                f"= {r['us_per_record']:+.4f} us/record, "
                f"local round {r['local_round_us']:.0f} us"
            )
            print(
                f"  2se interval [{r['pct_2se_interval'][0]:+.4f}, "
                f"{r['pct_2se_interval'][1]:+.4f}] %, "
                f"clears MUE 0.30 %: {r['clears_mue_0p30pct']}"
            )

    path = OUT / f"e179-screen-{label}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")

    run = wandb_run(label)
    run.summary["screen"] = out
    for name, r in out["contrasts"].items():
        if "mtp" not in r:
            continue
        run.summary[f"{name}/effect_pct"] = r["mtp"]["effect_pct_no_drift"]
        run.summary[f"{name}/se_pp"] = r["mtp"]["se_pp_no_drift"]
        run.summary[f"{name}/us_per_round"] = r["us_per_round"]
        run.summary[f"{name}/us_per_record"] = r["us_per_record"]
        run.summary[f"{name}/serial_null_pct"] = r["serial_null"]["effect_pct_no_drift"]
        run.summary[f"{name}/clears_mue"] = r["clears_mue_0p30pct"]
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
