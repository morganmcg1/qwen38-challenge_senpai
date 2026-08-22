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

# The head the local wrapper actually attaches. `wrapper.err` records
# `mtp_head=<cache>/qwen3.8-27b-mtp-v1/mtp-head`, which is the organizer-pinned
# BF16 head, NOT the smaller declared head in `mtp-head.manifest.json`. These
# are its tensor byte sizes read from the safetensors header, largest first.
# Resident dtype equals on-disk dtype for a BF16 head, so each tensor is one
# buffer of the same size.
PINNED_HEAD_TENSOR_BYTES = [
    178_257_920, 178_257_920, 178_257_920,  # mlp up, gate, down
    125_829_120,                            # self_attn.q_proj
    104_857_600,                            # fc
    62_914_560,                             # self_attn.o_proj
    10_485_760, 10_485_760,                 # self_attn.v_proj, k_proj
    10_240, 10_240, 10_240, 10_240, 10_240,  # five RMSNorm gains
    512, 512,                               # q_norm, k_norm
]
PINNED_HEAD_RESIDENT_BYTES = sum(PINNED_HEAD_TENSOR_BYTES)


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
                "wired_hist": fields.get("wired_hist", ""),
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


def parse_hist(blob: str) -> dict[int, int]:
    """`2^24:28,2^23:74` -> {24: 28, 23: 74}. A class holds [2^k, 2^(k+1))."""
    out: dict[int, int] = {}
    for item in blob.split(","):
        if not item:
            continue
        exponent, _, count = item.partition(":")
        out[int(exponent.removeprefix("2^"))] = int(count)
    return out


def composition_verdict(resize_draws: list[dict],
                        steady_draws: list[dict]) -> dict:
    """Is the head in or out of `unwired_set_`, and at what rate?

    Three independent arguments, none of which needs the head to be labelled
    inside the allocator:

      1. At the greedy fill nothing is left out at all, so the head is in.
      2. A weight dropped after the fill would leave its wired size class one
         member short. No wired class ever loses a member.
      3. No head tensor byte size is ever seen among the largest unwired
         buffers.

    A plain size bound is NOT used. One worker role holds a 254 MB unwired
    scratch buffer that is larger than every head tensor, so "bigger than the
    largest unwired buffer" proves nothing for that role.
    """
    largest_unwired = max(
        (max(d["unwired_top"]) for d in resize_draws + steady_draws
         if d["unwired_top"]), default=0)

    excluded_at_resize = sum(1 for d in resize_draws if d["unwired_bytes"] > 0)

    # A weight admitted at the fill could still be dropped later. The residency
    # set would have to shed it, so its size class would lose a member.
    resize_by_pid = {d["pid"]: parse_hist(d["wired_hist"]) for d in resize_draws}
    shrink_events = []
    for draw in steady_draws:
        base = resize_by_pid.get(draw["pid"])
        if base is None:
            continue
        now = parse_hist(draw["wired_hist"])
        for exponent, count in base.items():
            if now.get(exponent, 0) < count:
                shrink_events.append(
                    {"pid": draw["pid"], "leg": draw["leg"],
                     "class": exponent, "resize": count,
                     "steady": now.get(exponent, 0)})

    head_sizes = set(PINNED_HEAD_TENSOR_BYTES)
    seen_unwired = {size for d in resize_draws + steady_draws
                    for size in d["unwired_top"]}
    head_sizes_unwired = sorted(head_sizes & seen_unwired)
    # `unwired_top` is truncated, so only sizes at or above its smallest entry
    # are decidable. Report that threshold instead of implying full coverage.
    top_floor = min(
        (min(d["unwired_top"]) for d in resize_draws + steady_draws
         if len(d["unwired_top"]) >= 16), default=0)
    decidable = sorted(b for b in head_sizes if b >= top_floor)

    # Where the extra allowance goes. Grouped by role, never pooled across
    # roles, and reported as the LAST steady draw of each worker.
    last: dict[str, dict] = {}
    for draw in steady_draws:
        last[draw["pid"]] = draw
    by_role: dict[str, list[dict]] = {}
    for draw in last.values():
        by_role.setdefault(f"{draw['arm']}/w{draw['worker_index']}",
                           []).append(draw)

    unwired_mass = {}
    for key, group in sorted(by_role.items()):
        classes: dict[int, list[int]] = {}
        for draw in group:
            for exponent, count in parse_hist(draw["unwired_hist"]).items():
                classes.setdefault(exponent, []).append(count)
        unwired_mass[key] = {
            "draws": len(group),
            "unwired_count_mean": sum(d["unwired_count"] for d in group)
                                  / len(group),
            "unwired_mib_mean": sum(d["unwired_bytes"] for d in group)
                                / len(group) / MIB,
            "largest_unwired_bytes": max(max(d["unwired_top"]) for d in group
                                         if d["unwired_top"]),
            "class_count_mean": {
                f"2^{k}": sum(v) / len(group) for k, v in sorted(classes.items())
            },
            # Lower bound on the bytes each class holds: count * 2^k.
            "class_lower_bound_mib": {
                f"2^{k}": sum(v) / len(group) * (1 << k) / MIB
                for k, v in sorted(classes.items())
            },
        }

    # What the extra 448 MiB of allowance actually admitted: the size classes
    # that shrink in the unwired set when the arm changes, at equal role.
    admitted_delta = {}
    for index in sorted({int(k.split("/w")[1]) for k in unwired_mass}):
        small = unwired_mass.get(f"s64/w{index}")
        large = unwired_mass.get(f"s512/w{index}")
        if small is None or large is None:
            continue
        per_class = {}
        for name in sorted(set(small["class_count_mean"])
                           | set(large["class_count_mean"])):
            delta = (small["class_count_mean"].get(name, 0.0)
                     - large["class_count_mean"].get(name, 0.0))
            if abs(delta) >= 0.5:
                per_class[name] = {
                    "count_delta": delta,
                    "bytes_lower_bound_mib": (
                        delta * (1 << int(name.removeprefix("2^"))) / MIB),
                }
        admitted_delta[f"w{index}"] = {
            "unwired_mib_delta": (small["unwired_mib_mean"]
                                  - large["unwired_mib_mean"]),
            "unwired_count_delta": (small["unwired_count_mean"]
                                    - large["unwired_count_mean"]),
            "by_class": per_class,
        }

    return {
        "largest_unwired_bytes_over_all_draws": largest_unwired,
        "admitted_delta_s64_to_s512": admitted_delta,
        "resize_draws_with_any_exclusion": excluded_at_resize,
        "resize_draws": len(resize_draws),
        "head_exclusion_rate_at_fill": (
            excluded_at_resize / len(resize_draws) if resize_draws else None),
        "pinned_head_resident_bytes": PINNED_HEAD_RESIDENT_BYTES,
        "unwired_top_decidable_floor_bytes": top_floor,
        "head_tensor_sizes_decidable": decidable,
        "head_byte_share_decidable": (
            sum(b for b in PINNED_HEAD_TENSOR_BYTES if b >= top_floor)
            / PINNED_HEAD_RESIDENT_BYTES),
        "head_tensor_sizes_seen_unwired": head_sizes_unwired,
        "wired_class_shrink_events": shrink_events,
        "steady_unwired_by_role": unwired_mass,
    }


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
        "composition_verdict": composition_verdict(resize_draws, steady_draws),
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

    comp = result["composition_verdict"]
    print()
    print("=== is the head in or out of unwired_set_ ===")
    print(f"  largest unwired buffer, all {len(resize_draws) + len(steady_draws)} draws: "
          f"{comp['largest_unwired_bytes_over_all_draws']} B")
    print(f"  resize draws with any exclusion: "
          f"{comp['resize_draws_with_any_exclusion']}/{comp['resize_draws']} "
          f"= {comp['head_exclusion_rate_at_fill'] * 100:.1f}%")
    print(f"  pinned head resident bytes: {comp['pinned_head_resident_bytes']}")
    print(f"  wired size classes that ever lost a member: "
          f"{len(comp['wired_class_shrink_events'])}")
    decidable_tensors = sum(1 for b in PINNED_HEAD_TENSOR_BYTES
                            if b >= comp["unwired_top_decidable_floor_bytes"])
    print(f"  unwired_top is decidable at or above "
          f"{comp['unwired_top_decidable_floor_bytes']} B: "
          f"{len(comp['head_tensor_sizes_decidable'])} distinct sizes, "
          f"{decidable_tensors}/{len(PINNED_HEAD_TENSOR_BYTES)} head tensors, "
          f"{comp['head_byte_share_decidable'] * 100:.2f}% of head bytes")
    print(f"  head tensor sizes ever seen unwired: "
          f"{comp['head_tensor_sizes_seen_unwired'] or 'none'}")

    print()
    print("=== what the extra 448 MiB of allowance admitted ===")
    for role, v in comp["admitted_delta_s64_to_s512"].items():
        print(f"  {role} unwired {v['unwired_mib_delta']:+8.1f} MiB "
              f"count {v['unwired_count_delta']:+8.1f}")
        for name, c in sorted(v["by_class"].items(),
                              key=lambda kv: -abs(kv[1]["bytes_lower_bound_mib"])):
            print(f"      {name:>5s} {c['count_delta']:+7.1f} buffers "
                  f">= {c['bytes_lower_bound_mib']:+8.1f} MiB")

    print()
    print("=== steady unwired composition, by arm and worker role ===")
    for key, v in comp["steady_unwired_by_role"].items():
        print(f"  {key:12s} count={v['unwired_count_mean']:8.1f} "
              f"bytes={v['unwired_mib_mean']:8.1f} MiB "
              f"largest={v['largest_unwired_bytes']} B")
        top = sorted(v["class_lower_bound_mib"].items(),
                     key=lambda kv: -kv[1])[:5]
        print("               heaviest classes (>= MiB): " + ", ".join(
            f"{k} {v['class_count_mean'][k]:.0f}x >={mib:.0f}"
            for k, mib in top))

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
        payload["e130_adm_head_exclusion_rate_at_fill"] = (
            comp["head_exclusion_rate_at_fill"])
        payload["e130_adm_largest_unwired_bytes"] = (
            comp["largest_unwired_bytes_over_all_draws"])
        payload["e130_adm_head_sizes_seen_unwired"] = len(
            comp["head_tensor_sizes_seen_unwired"])
        payload["e130_adm_wired_class_shrink_events"] = (
            len(comp["wired_class_shrink_events"]))
        for key, v in comp["steady_unwired_by_role"].items():
            tag = key.replace("/", "_")
            payload[f"e130_adm_{tag}_unwired_count"] = v["unwired_count_mean"]
            payload[f"e130_adm_{tag}_unwired_mib"] = v["unwired_mib_mean"]
        run.log(payload)
        run.summary.update(payload)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
