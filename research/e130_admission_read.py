#!/usr/bin/env python3
"""E130 rung 10, F13 section 4: read the admission probe.

Answers three pre-registered questions and states, for each, what would have
refuted it. There is no timing in this experiment, so nothing here needs a
thermal gate, a counterbalanced order, or a noise scale.

  usage: research/e130_admission_read.py --prefix e130-adm [--wandb]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MIB = 1 << 20

# F13 section 4. Recorded here so the reader cannot be tuned to its own data.
PREREGISTERED = {
    "s512": "unwired bytes = 0 in every draw; any draw that leaves the head "
            "out refutes the model",
    "s64_bimodal": "bimodal, one mode near 0 and one near 430 MB, slow rate "
                   "between 1/3 and 1/2; unimodal and small in every draw "
                   "refutes the model",
    "s64_shape": "when the head is out it is out as a set of tensors, not as "
                 "one 407.93 MiB object; only ever all-or-nothing refutes the "
                 "set hypothesis",
}

# The proposal head, from the E130 rung 8 traffic measurement.
HEAD_BYTES = 427_742_600


def parse_kv(line: str) -> dict:
    out = {}
    for key, value in re.findall(r"([a-z_0-9^]+)=([^\s]+)", line):
        out[key] = value
    return out


def parse_sizes(blob: str) -> list[int]:
    return [int(x) for x in blob.split(",") if x]


def read_leg(out: Path) -> dict:
    meta = {}
    meta_path = out / "meta.txt"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value

    # `active` at the sizing instant, per pid, from the session probe. This is
    # a sum of buffer lengths; the admission probe reports page-rounded sizes,
    # so the difference IS the page rounding tax.
    active_by_pid: dict[str, int] = {}
    resources_by_pid: dict[str, int] = {}
    residency = out / "residency.log"
    if residency.exists():
        for line in residency.read_text().splitlines():
            if "phase=sizing" not in line:
                continue
            fields = parse_kv(line)
            active_by_pid[fields["pid"]] = int(fields["active"])
            resources_by_pid[fields["pid"]] = int(fields["resources"])

    # One local leg starts several model-holding workers with DIFFERENT jobs:
    # reference, serial control and MTP. They are not repeated draws of one
    # quantity, so they are never pooled. Worker start order is stable and pids
    # increase with it, so the rank of a pid inside its leg is a usable role
    # label without touching the trusted parent.
    admission = out / "admission.log"
    admission_lines = (
        [l for l in admission.read_text().splitlines()
         if l.startswith("e130-admission")]
        if admission.exists() else [])
    pids_in_order = sorted({parse_kv(l)["pid"] for l in admission_lines},
                           key=int)
    worker_index = {pid: i + 1 for i, pid in enumerate(pids_in_order)}

    draws = []
    if admission_lines:
        for line in admission_lines:
            fields = parse_kv(line)
            unwired_top = parse_sizes(fields.get("unwired_top", ""))
            wired_top = parse_sizes(fields.get("wired_top", ""))
            pid = fields["pid"]
            total = int(fields["total_bytes"])
            active = active_by_pid.get(pid)
            draws.append({
                "pid": pid,
                "worker_index": worker_index[pid],
                "leg": out.name,
                "phase": fields["phase"],
                "arm": meta.get("arm"),
                "slack_mb": int(meta.get("slack_mb", -1)),
                "capacity": int(fields["capacity"]),
                "wired_count": int(fields["wired_count"]),
                "wired_bytes": int(fields["wired_bytes_sum"]),
                "unwired_count": int(fields["unwired_count"]),
                "unwired_bytes": int(fields["unwired_bytes"]),
                "total_bytes": total,
                "unwired_top": unwired_top,
                "wired_top": wired_top,
                "unwired_hist": fields.get("unwired_hist", ""),
                "active_at_sizing": active,
                "resources_at_sizing": resources_by_pid.get(pid),
                # Page-rounded bytes minus buffer-length bytes. Measured, not
                # bounded, and only defined when both probes saw the same pid.
                "page_tax_bytes": (total - active) if active else None,
            })
    return {"tag": out.name, "meta": meta, "draws": draws}


def spread(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "distinct": len(set(values)),
        "values": sorted(values),
    }


def resize_verdict(draws: list[dict]) -> dict:
    """The greedy fill. This is where the tower lottery was supposed to live."""
    by_arm: dict[str, list[dict]] = {}
    for draw in draws:
        by_arm.setdefault(draw["arm"], []).append(draw)

    out = {}
    for arm, arm_draws in sorted(by_arm.items()):
        unwired = [d["unwired_bytes"] for d in arm_draws]
        taxes = [d["page_tax_bytes"] for d in arm_draws
                 if d["page_tax_bytes"] is not None]
        out[arm] = {
            "draws": len(arm_draws),
            "fully_admitted": sum(1 for u in unwired if u == 0),
            # A draw leaving a tenth of the head out would be a "slow" draw.
            "slow_draws": sum(1 for u in unwired if u >= HEAD_BYTES // 10),
            "unwired_bytes": spread(unwired),
            "wired_bytes": spread([d["wired_bytes"] for d in arm_draws]),
            "wired_count": spread([d["wired_count"] for d in arm_draws]),
            "page_tax_bytes": spread(taxes),
            "page_tax_mib_mean": (
                sum(taxes) / len(taxes) / MIB if taxes else None),
        }
    return out


def steady_verdict(draws: list[dict], resize_draws: list[dict]) -> dict:
    """Steady state, grouped by worker role. Roles are never pooled.

    `slack_used` is how much of the allowance the post-sizing growth actually
    took. It is the quantity the shipped literal controls.
    """
    resize_by_pid = {d["pid"]: d for d in resize_draws}
    last: dict[str, dict] = {}
    for draw in draws:
        last[draw["pid"]] = draw

    grouped: dict[str, list[dict]] = {}
    for draw in last.values():
        key = f"{draw['arm']}/w{draw['worker_index']}"
        grouped.setdefault(key, []).append(draw)

    out = {}
    for key, group in sorted(grouped.items()):
        used = []
        for draw in group:
            base = resize_by_pid.get(draw["pid"])
            if base is not None:
                used.append(draw["wired_bytes"] - base["wired_bytes"])
        out[key] = {
            "draws": len(group),
            "slack_used_bytes": spread(used),
            "slack_used_mib_mean": (
                sum(used) / len(used) / MIB if used else None),
            "unwired_bytes": spread([d["unwired_bytes"] for d in group]),
            "unwired_count": spread([d["unwired_count"] for d in group]),
            "headroom_bytes": spread(
                [d["capacity"] - d["wired_bytes"] for d in group]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e130-adm")
    ap.add_argument("--root", type=Path, default=Path("research/out"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    legs = [read_leg(p) for p in sorted(args.root.glob(f"{args.prefix}-*"))
            if p.is_dir()]
    all_draws = [d for leg in legs for d in leg["draws"]]
    resize_draws = [d for d in all_draws if d["phase"] == "resize"]

    steady_draws = [d for d in all_draws if d["phase"] == "steady"]
    result = {
        "experiment": "e130-admission",
        "timed": False,
        "question": "what does the greedy fill admit, and what is the page "
                    "rounding tax in measured bytes",
        "preregistered": PREREGISTERED,
        "legs": [leg["tag"] for leg in legs],
        "resize_draws": resize_draws,
        "steady_draws": steady_draws,
        "resize_verdict": resize_verdict(resize_draws),
        "steady_verdict": steady_verdict(steady_draws, resize_draws),
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("=== resize: the greedy fill ===")
    for arm, v in result["resize_verdict"].items():
        print(f"  {arm:5s} draws={v['draws']} "
              f"fully_admitted={v['fully_admitted']} "
              f"slow={v['slow_draws']} "
              f"distinct_wired_bytes={v['wired_bytes']['distinct']}")
        print(f"        unwired MiB  min={v['unwired_bytes']['min'] / MIB:.3f} "
              f"max={v['unwired_bytes']['max'] / MIB:.3f}")
        if v["page_tax_mib_mean"] is not None:
            print(f"        page tax MiB min="
                  f"{v['page_tax_bytes']['min'] / MIB:.3f} "
                  f"max={v['page_tax_bytes']['max'] / MIB:.3f} "
                  f"mean={v['page_tax_mib_mean']:.3f}")

    print()
    print("=== steady state, by arm and worker role ===")
    for key, v in result["steady_verdict"].items():
        used = v["slack_used_bytes"]
        print(f"  {key:12s} draws={v['draws']} "
              f"slack_used MiB {used.get('min', 0) / MIB:8.2f}..."
              f"{used.get('max', 0) / MIB:8.2f} "
              f"distinct={used.get('distinct')}")
        print(f"               unwired MiB "
              f"{v['unwired_bytes']['min'] / MIB:8.1f}..."
              f"{v['unwired_bytes']['max'] / MIB:8.1f}  "
              f"headroom B {v['headroom_bytes']['min']}..."
              f"{v['headroom_bytes']['max']}")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130adm",
            name="e130adm",
            resume="allow",
            config={"experiment": "e130-admission",
                    "preregistered": PREREGISTERED},
            save_code=True,
        )
        payload = {}
        for arm, v in result["resize_verdict"].items():
            payload[f"e130_adm_{arm}_resize_draws"] = v["draws"]
            payload[f"e130_adm_{arm}_resize_fully_admitted"] = (
                v["fully_admitted"])
            payload[f"e130_adm_{arm}_resize_slow_draws"] = v["slow_draws"]
            payload[f"e130_adm_{arm}_resize_unwired_max_mib"] = (
                v["unwired_bytes"]["max"] / MIB)
            payload[f"e130_adm_{arm}_page_tax_mib"] = v["page_tax_mib_mean"]
        for key, v in result["steady_verdict"].items():
            tag = key.replace("/", "_")
            payload[f"e130_adm_{tag}_slack_used_mib"] = v["slack_used_mib_mean"]
            payload[f"e130_adm_{tag}_unwired_max_mib"] = (
                v["unwired_bytes"]["max"] / MIB)
        run.log(payload)
        run.summary.update(payload)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
