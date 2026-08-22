#!/usr/bin/env python3
"""E130 rung 10: price the wired residency slack against its three bounds.

Rung 9 measured how much the scored window allocates and keeps live after
`Qwen36MTPBlockSession.wireResidentWeightsIfEnabled` sizes the wired ticket.
This script turns that measurement into the three bounds F10 asked for, so the
proposed constant is justified by arithmetic that can be re-run rather than by
a number quoted in a comment.

    target = activeMemory * fraction + slackMB << 20
    target = min(target, maxRecommendedWorkingSetBytes - 256 MiB)

Bound 1  the slack must exceed measured persistent growth, with margin for the
         page rounding the residency set charges but the sizing input does not.
Bound 2  `active + slack` must stay far below `maxrec - 256 MiB`, or the clamp
         silently truncates the ticket.
Bound 3  the slack must stay small against the live tower, so that the design
         property in the doc comment survives: genuinely large scratch must
         still fail the fit test and stay on the commit-free unwired path.

The fit test is per buffer and greedy, in `ResidencySet::insert`:

    if (wired_set_->allocatedSize() + buf->allocatedSize() <= capacity_)
        addAllocation(buf); commit();          // one driver commit per buffer
    else
        unwired_set_.insert(buf);              // commit-free

Two consequences drive bound 1 and bound 3. The set charges page-rounded
`allocatedSize()` while the sizing input sums `buf->length()`, so the tower's
page rounding is deducted from the slack before any growth can be wired. And
because admission is first-come rather than by importance, a slack smaller than
demand admits whichever buffers arrive first, which is a mechanism for run to
run variation rather than a fixed offset.

Usage
-----
    python3 research/e130_rung10_bounds.py \
        --rung9 research/e130-artifacts/rung9-allocation-growth.json \
        --residency-log research/out/e130-rung10-t128/residency.log \
        --out research/e130-artifacts/rung10-slack-bounds.json --wandb
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30

# Apple Silicon `vm_page_size`. The Metal allocator rounds every standalone
# buffer up to this granularity, and `MTLResidencySet.allocatedSize()` reports
# the rounded size.
PAGE_BYTES = 16 << 10

# The margin `wireResidentWeightsIfEnabled` keeps below the recommended working
# set before it clamps the ticket.
CLAMP_MARGIN_BYTES = 256 << 20

CURRENT_SLACK_MIB = 64
PROPOSED_SLACK_MIB = 512

# Ranked host memory. The 96 GiB guard means the wired path only runs on a host
# at or above that size; F10 prices bound 2 on a 128 GiB host.
RANKED_PHYSMEM_BYTES = 128 * GIB

FIELD = re.compile(r"([a-z_0-9]+)=([A-Za-z0-9_.+-]+)")


def read_resources(path: Path) -> dict:
    """Live Metal buffer count at each sizing instant, grouped by pid."""
    per_pid: dict[int, dict] = {}
    for raw in path.read_text().splitlines():
        if "e130-residency" not in raw or "phase=sizing" not in raw:
            continue
        row = {
            k: (int(v) if re.fullmatch(r"-?[0-9]+", v) else v)
            for k, v in FIELD.findall(raw)
        }
        pid = row.get("pid")
        if pid is None or pid in per_pid:
            continue
        per_pid[pid] = row
    counts = [r["resources"] for r in per_pid.values() if "resources" in r]
    slacks = sorted({r.get("slack_mb") for r in per_pid.values()})
    actives = sorted({r.get("active") for r in per_pid.values()})
    return {
        "log": str(path),
        "process_count": len(per_pid),
        "resources_at_sizing": counts,
        "resources_min": min(counts) if counts else None,
        "resources_max": max(counts) if counts else None,
        "resources_median": statistics.median(counts) if counts else None,
        "slack_mb_in_build": slacks,
        "active_at_sizing_distinct": actives,
        "resource_limit": sorted(
            {r.get("reslimit") for r in per_pid.values() if "reslimit" in r}
        ),
    }


def bound_one(growth_mib: float, slack_mib: float, resources: int | None) -> dict:
    """Slack must cover persistent growth plus the tower's page-rounding tax."""
    headroom = slack_mib - growth_mib
    out = {
        "name": "slack exceeds measured persistent growth",
        "measured_persistent_growth_mib": growth_mib,
        "slack_mib": slack_mib,
        "covers_growth": slack_mib > growth_mib,
        "ratio_slack_over_growth": slack_mib / growth_mib,
        "headroom_after_growth_mib": headroom,
    }
    if resources is None:
        out["page_rounding"] = "not measured"
        return out
    # `numResources` counts heap-suballocated buffers too, and those are never
    # inserted individually, so this is an upper bound on residency-set entries.
    expected = resources * (PAGE_BYTES / 2) / MIB
    worst = resources * (PAGE_BYTES - 1) / MIB
    out["page_rounding"] = {
        "live_buffers_at_sizing_upper_bound": resources,
        "page_bytes": PAGE_BYTES,
        "expected_tax_mib": expected,
        "worst_case_tax_mib": worst,
        "expected_tax_vs_current_slack_pct": 100.0 * expected / CURRENT_SLACK_MIB,
        "expected_tax_vs_proposed_slack_pct": 100.0 * expected / slack_mib,
        "survives_expected_tax": headroom > expected,
        "survives_worst_case_tax": headroom > worst,
        "current_slack_survives_expected_tax": (
            CURRENT_SLACK_MIB - growth_mib
        ) > expected,
    }
    return out


def bound_two(active_bytes: int, slack_mib: float, physmem_bytes: int,
              maxrec_bytes: int, label: str) -> dict:
    """`active + slack` must stay far below the clamp at `maxrec - 256 MiB`."""
    ceiling = maxrec_bytes - CLAMP_MARGIN_BYTES
    target = active_bytes + int(slack_mib) * MIB
    headroom = ceiling - active_bytes
    return {
        "name": f"ticket stays below the clamp ({label})",
        "host": label,
        "physmem_bytes": physmem_bytes,
        "physmem_gib": physmem_bytes / GIB,
        "maxrec_bytes": maxrec_bytes,
        "maxrec_gib": maxrec_bytes / GIB,
        "maxrec_over_physmem": maxrec_bytes / physmem_bytes,
        "clamp_ceiling_gib": ceiling / GIB,
        "active_at_sizing_gib": active_bytes / GIB,
        "requested_target_gib": target / GIB,
        "clamp_is_inactive": target < ceiling,
        "headroom_for_slack_mib": headroom / MIB,
        "headroom_over_proposed_slack": (headroom / MIB) / slack_mib,
        "proposed_slack_pct_of_headroom": 100.0 * slack_mib / (headroom / MIB),
    }


def bound_three(active_bytes: int, slack_mib: float, growth_mib: float,
                scratch_mib: float, pool_mib: float) -> dict:
    """Slack must stay small against the tower so scratch still fails the fit."""
    tower_mib = active_bytes / MIB
    residual = slack_mib - growth_mib
    return {
        "name": "design property survives: large scratch still fails the fit test",
        "tower_at_sizing_mib": tower_mib,
        "tower_at_sizing_gib": tower_mib / 1024.0,
        "slack_pct_of_tower": 100.0 * slack_mib / tower_mib,
        "current_slack_pct_of_tower": 100.0 * CURRENT_SLACK_MIB / tower_mib,
        "peak_live_scratch_mib": scratch_mib,
        "peak_pool_growth_mib": pool_mib,
        "residual_headroom_after_persistent_mib": residual,
        "residual_pct_of_peak_live_scratch": 100.0 * residual / scratch_mib,
        "residual_pct_of_peak_pool": 100.0 * residual / pool_mib,
        "pct_of_live_scratch_still_unwired": 100.0 * (1.0 - residual / scratch_mib),
        # Admission is first-come, so the residual is filled by whichever
        # buffers arrive first rather than by the persistent set only. The
        # design property is about aggregate scratch, not about any one buffer.
        "admission_is_first_come": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung9", type=Path,
                    default=Path("research/e130-artifacts/rung9-allocation-growth.json"))
    ap.add_argument("--residency-log", type=Path, default=None)
    ap.add_argument("--slack-mib", type=float, default=PROPOSED_SLACK_MIB)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    rung9 = json.loads(args.rung9.read_text())
    verdict = rung9["verdict"]
    growth = verdict["mtp_persistent_growth_mib_by_tokens"]["512"]
    scratch = verdict["mtp_peak_growth_mib_by_tokens"]["512"]

    leg512 = next(l for l in rung9["legs"] if l["decode_tokens"] == 512)
    procs = leg512["processes"]
    active_bytes = procs[0]["active_at_sizing"]
    assert len({p["active_at_sizing"] for p in procs}) == 1, "sizing input drifted"
    local_maxrec = procs[0]["maxrec"]
    local_physmem = procs[0]["physmem"]
    pool = max(p["pool_growth_max_mib"] for p in procs)

    resources = None
    resource_reading = None
    if args.residency_log is not None and args.residency_log.exists():
        resource_reading = read_resources(args.residency_log)
        resources = resource_reading["resources_max"]

    # The ranked host's recommended working set is not measured here. Scale it
    # by the ratio this host reports, and also give Apple's more conservative
    # 0.75 rule, so bound 2 is reported as a range rather than a single figure.
    ratio = local_maxrec / local_physmem
    ranked_bounds = [
        bound_two(active_bytes, args.slack_mib, RANKED_PHYSMEM_BYTES,
                  int(RANKED_PHYSMEM_BYTES * ratio),
                  f"128 GiB ranked, extrapolated at this host's {ratio:.5f}"),
        bound_two(active_bytes, args.slack_mib, RANKED_PHYSMEM_BYTES,
                  int(RANKED_PHYSMEM_BYTES * 0.75),
                  "128 GiB ranked, conservative 0.75 rule"),
    ]

    result = {
        "experiment": "e130-rung10",
        "question": "is 512 MiB the right wired residency slack",
        "proposed_slack_mib": args.slack_mib,
        "current_slack_mib": CURRENT_SLACK_MIB,
        "shortfall_of_current_slack_mib": growth - CURRENT_SLACK_MIB,
        "resource_reading": resource_reading,
        "bound_1_covers_growth": bound_one(growth, args.slack_mib, resources),
        "bound_2_below_clamp": {
            "local_measured": bound_two(
                active_bytes, args.slack_mib, local_physmem, local_maxrec,
                f"{local_physmem / GIB:.0f} GiB local, measured"),
            "ranked_modelled": ranked_bounds,
        },
        "bound_3_design_property": bound_three(
            active_bytes, args.slack_mib, growth, scratch, pool),
        "bit_exactness": (
            "the constant reaches only WiredMemoryTicket(size:) and thence the "
            "MTLResidencySet capacity; it never reaches a tensor, a kernel "
            "argument, a dispatch shape, a dtype, a reduction order or a "
            "scheduling decision, so no emitted token can change"
        ),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)

    b1 = result["bound_1_covers_growth"]
    b3 = result["bound_3_design_property"]
    print()
    print(f"bound 1  slack {args.slack_mib:.0f} MiB vs growth {growth:.2f} MiB "
          f"= {b1['ratio_slack_over_growth']:.2f}x, headroom "
          f"{b1['headroom_after_growth_mib']:.2f} MiB")
    if resources is not None:
        pr = b1["page_rounding"]
        print(f"         page rounding upper bound {pr['live_buffers_at_sizing_upper_bound']} "
              f"buffers -> {pr['expected_tax_mib']:.1f} MiB expected "
              f"({pr['expected_tax_vs_current_slack_pct']:.0f} % of the old 64 MiB slack)")
    for b in ranked_bounds:
        print(f"bound 2  {b['host']}: headroom {b['headroom_for_slack_mib'] / 1024:.1f} GiB "
              f"= {b['headroom_over_proposed_slack']:.0f}x the proposed slack, "
              f"clamp inactive {b['clamp_is_inactive']}")
    print(f"bound 3  slack is {b3['slack_pct_of_tower']:.2f} % of the "
          f"{b3['tower_at_sizing_gib']:.2f} GiB tower; "
          f"{b3['pct_of_live_scratch_still_unwired']:.1f} % of live scratch still "
          f"fails the fit test")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130rung10",
            name="e130rung10",
            resume="allow",
            config={
                "experiment": "e130-rung10",
                "proposed_slack_mib": args.slack_mib,
                "current_slack_mib": CURRENT_SLACK_MIB,
            },
            save_code=True,
        )
        run.log({
            "e130_rung10_proposed_slack_mib": args.slack_mib,
            "e130_rung10_measured_growth_mib": growth,
            "e130_rung10_slack_over_growth": b1["ratio_slack_over_growth"],
            "e130_rung10_headroom_after_growth_mib": b1["headroom_after_growth_mib"],
            "e130_rung10_slack_pct_of_tower": b3["slack_pct_of_tower"],
            "e130_rung10_pct_scratch_still_unwired": b3["pct_of_live_scratch_still_unwired"],
            "e130_rung10_clamp_headroom_gib": ranked_bounds[1]["headroom_for_slack_mib"] / 1024,
            "e130_rung10_clamp_headroom_over_slack": ranked_bounds[1]["headroom_over_proposed_slack"],
            "e130_rung10_resources_at_sizing": resources or 0,
        })
        if args.out is not None:
            artifact = wandb.Artifact("e130-rung10-slack-bounds", type="analysis")
            artifact.add_file(str(args.out))
            run.log_artifact(artifact)
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
