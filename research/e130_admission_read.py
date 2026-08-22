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

    draws = []
    admission = out / "admission.log"
    if admission.exists():
        for line in admission.read_text().splitlines():
            if not line.startswith("e130-admission"):
                continue
            fields = parse_kv(line)
            unwired_top = parse_sizes(fields.get("unwired_top", ""))
            wired_top = parse_sizes(fields.get("wired_top", ""))
            pid = fields["pid"]
            total = int(fields["total_bytes"])
            active = active_by_pid.get(pid)
            draws.append({
                "pid": pid,
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


def verdicts(resize_draws: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for draw in resize_draws:
        by_arm.setdefault(draw["arm"], []).append(draw)

    result = {"preregistered": PREREGISTERED, "by_arm": {}}
    for arm, draws in sorted(by_arm.items()):
        unwired = [d["unwired_bytes"] for d in draws]
        clean = [u for u in unwired if u == 0]
        # A draw that leaves at least a tenth of the head unwired is "slow".
        slow = [u for u in unwired if u >= HEAD_BYTES // 10]
        taxes = [d["page_tax_bytes"] for d in draws
                 if d["page_tax_bytes"] is not None]
        result["by_arm"][arm] = {
            "draws": len(draws),
            "fully_admitted": len(clean),
            "slow_draws": len(slow),
            "slow_rate": len(slow) / len(draws) if draws else None,
            "unwired_bytes_min": min(unwired) if unwired else None,
            "unwired_bytes_max": max(unwired) if unwired else None,
            "unwired_bytes_values": sorted(unwired),
            "page_tax_bytes_min": min(taxes) if taxes else None,
            "page_tax_bytes_max": max(taxes) if taxes else None,
            "page_tax_mib_mean": (
                sum(taxes) / len(taxes) / MIB if taxes else None),
        }
    return result


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

    result = {
        "experiment": "e130-admission",
        "timed": False,
        "question": "what does the greedy fill admit, and what is the page "
                    "rounding tax in measured bytes",
        "legs": [leg["tag"] for leg in legs],
        "resize_draws": resize_draws,
        "steady_draws": [d for d in all_draws if d["phase"] == "steady"],
        "verdicts": verdicts(resize_draws),
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("=== E130 admission probe, resize-time draws ===")
    for arm, v in result["verdicts"]["by_arm"].items():
        print(f"  {arm:5s} draws={v['draws']} fully_admitted="
              f"{v['fully_admitted']} slow={v['slow_draws']} "
              f"rate={v['slow_rate']}")
        print(f"        unwired MiB min={v['unwired_bytes_min'] / MIB:.2f} "
              f"max={v['unwired_bytes_max'] / MIB:.2f}")
        if v["page_tax_mib_mean"] is not None:
            print(f"        page tax MiB mean={v['page_tax_mib_mean']:.2f} "
                  f"min={v['page_tax_bytes_min'] / MIB:.2f} "
                  f"max={v['page_tax_bytes_max'] / MIB:.2f}")
    print()
    for draw in resize_draws:
        top = ",".join(str(s) for s in draw["unwired_top"][:6])
        print(f"  pid={draw['pid']} arm={draw['arm']} "
              f"unwired={draw['unwired_bytes'] / MIB:8.2f} MiB "
              f"n={draw['unwired_count']:5d} top=[{top}]")

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
        for arm, v in result["verdicts"]["by_arm"].items():
            payload[f"e130_adm_{arm}_draws"] = v["draws"]
            payload[f"e130_adm_{arm}_slow_rate"] = v["slow_rate"]
            payload[f"e130_adm_{arm}_unwired_max_mib"] = (
                v["unwired_bytes_max"] / MIB)
            payload[f"e130_adm_{arm}_page_tax_mib"] = v["page_tax_mib_mean"]
        run.log(payload)
        run.summary.update(payload)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
