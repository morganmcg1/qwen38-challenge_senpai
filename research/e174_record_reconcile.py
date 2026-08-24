#!/usr/bin/env python3
"""E174 step 2b: reconcile the CPU record probe with the end-to-end screen.

The screen measured one standalone fill dispatch PLUS its host kernel record at
4.0-5.3 us of marginal candidate MTP wall time. The probe measures the same
call's two phases in ISOLATION, with nothing overlapping it:

    record   host graph-node construction, MLXFastKernel.callAsFunction with
             no eval. This is exactly the work item (2) proposes to cache.
    eval     encode, dispatch and compute.

The isolated total is larger than the marginal end-to-end cost, because in a
real round the host builds records while the GPU runs earlier work. So the
probe cannot be read straight across. This script brackets the record's share
of the marginal cost between the two defensible overlap assumptions, and
applies the advisor's 1 us arithmetic stop rule to both ends rather than to a
single convenient number.

It then prices the shipped tree's 130 unserved standalone fills producer-side,
which is the pricing section the advisor asked for.

Everything here is `harness=local`. No ranked coefficient is derived.
"""

from __future__ import annotations

import json
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "out"

BRACKET = (4.0, 5.3)
# Arm `s` round time from the screen artifact, the same leg the bracket's
# lower bound (`o - s`) was measured on.
ARM_S_ROUND_US = 181992.05465423755
# Census: standalone fills per round in the shipped tree, measured over 77
# differenced rounds of a 512-token leg.
SHIPPED_FILLS_PER_ROUND = 130
MIN_USEFUL_EFFECT_PCT = 0.30


def main() -> None:
    probe = json.loads((OUT / "e174-record-probe.json").read_text())
    cells = probe["cells"]

    # The larger batch is the better estimator: the same per-call work is
    # averaged over 8x more calls, so any fixed per-batch overhead is diluted
    # 8x. The batch-scaling control already showed the two agree to 4.4%.
    record = statistics.median(
        c["batch_256"]["record_us_per_call_median"] for c in cells.values()
    )
    evaluate = statistics.median(
        c["batch_256"]["eval_us_per_call_median"] for c in cells.values()
    )
    isolated_total = record + evaluate
    record_share = record / isolated_total

    # Reading 1, uniform overlap: both phases hide the same fraction of their
    # isolated cost behind other work, so the record keeps its isolated share
    # of the marginal cost. This is the conservative end for item (2).
    uniform = [record_share * b for b in BRACKET]
    # Reading 2, serial record: the host record is on the critical path 1:1 and
    # the fill's tiny dispatch hides completely. The record then accounts for
    # the whole marginal cost, capped by its own isolated value.
    serial = [min(record, b) for b in BRACKET]

    price_us = [SHIPPED_FILLS_PER_ROUND * b for b in BRACKET]
    price_pct = [100.0 * u / ARM_S_ROUND_US for u in price_us]
    # The same price at the probe's own per-record figure, which is what a
    # producer-side epilogue actually removes: one host record, not a record
    # plus a dispatch.
    price_probe_us = SHIPPED_FILLS_PER_ROUND * record
    price_probe_pct = 100.0 * price_probe_us / ARM_S_ROUND_US

    report = {
        "experiment": "E174-step2b-reconciliation",
        "harness": "local",
        "gate_qualified_for_timing": False,
        "screen_run": "7ueick4f",
        "marginal_bracket_us_per_dispatch_plus_record": list(BRACKET),
        "isolated_record_us_per_call": round(record, 4),
        "isolated_eval_us_per_call": round(evaluate, 4),
        "isolated_total_us_per_call": round(isolated_total, 4),
        "isolated_record_share_of_total": round(record_share, 4),
        "isolated_total_over_marginal": [
            round(isolated_total / b, 3) for b in BRACKET
        ],
        "record_us_in_marginal_uniform_overlap": [round(v, 3) for v in uniform],
        "record_us_in_marginal_serial_record": [round(v, 3) for v in serial],
        "record_min_over_both_readings_us": round(min(uniform + serial), 3),
        "advisor_stop_rule_threshold_us": 1.0,
        "item2_dead_by_arithmetic": min(uniform + serial) < 1.0,
        "batch_scaling_control_failures": probe["batch_scaling_control_failures"],
        "producer_side_removal_price": {
            "fills_per_round": SHIPPED_FILLS_PER_ROUND,
            "arm_s_round_us": ARM_S_ROUND_US,
            "us_per_round_at_bracket": [round(v, 1) for v in price_us],
            "pct_of_round_at_bracket": [round(v, 4) for v in price_pct],
            "us_per_round_at_probe_record": round(price_probe_us, 1),
            "pct_of_round_at_probe_record": round(price_probe_pct, 4),
            "min_useful_effect_pct": MIN_USEFUL_EFFECT_PCT,
            "ceiling_clears_min_useful": price_pct[1] >= MIN_USEFUL_EFFECT_PCT,
            "note": "CEILING. It assumes the producer epilogue that removes "
            "each fill is free. It is not: on these producers the epilogue is "
            "a sparse post-barrier tail, and the screen's f - o contrast shows "
            "GPU work on this pipeline is not free either.",
        },
    }

    path = OUT / "e174-record-reconcile.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
