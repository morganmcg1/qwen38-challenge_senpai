#!/usr/bin/env python3
"""Consolidate the F7 item-3 outputs into one committed artifact."""

import json
from collections import Counter
from pathlib import Path

OUT = Path("research/e128-artifacts/f7-strata.json")


def main():
    strata = json.load(open("/tmp/e128_strata.json"))
    curve = json.load(open("/tmp/e128_strata_curve.json"))
    high = json.load(open("/tmp/e128_strata_highwidth.json"))

    g6 = Counter()
    for key in strata["strata"]:
        g6[int(key.split(",")[5])] += len(strata["strata"][key])

    doc = {
        "harness": "ranked",
        "gpu_seconds": 0,
        "source": {
            "trees": "/tmp/tree_ids.json",
            "board": "/tmp/yukon-board/full.json",
            "dispatch_path": "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/"
                             "metal/kernels/quantized.h",
            "dispatch_symbol": "qmv_fast_crossrow_affine4_g64_m<T, M, IPG>",
            "branch": "out_vec_size >= 4096",
        },
        "coverage": {
            "trees_scanned": 456,
            "tables_found": 202,
            "crossrow_no_m_table": 220,
            "no_crossrow": 34,
            "reference_schedule_rows_with_local_tree": 0,
            "table_rows_on_reference_schedule": 0,
        },
        "strata": {
            key: {"n": len(sids), "g": [int(c) for c in key.split(",")]}
            for key, sids in strata["strata"].items()
        },
        "g6_histogram": dict(g6),
        "per_stratum_hinge": curve["per_stratum"],
        "joint_full_range": curve["joint"],
        "high_width": high,
        "f_hat_us": high["twoway"]["f"],
        "f_se_us": high["twoway"]["f_se"],
        "f_ci95_us": [
            high["twoway"]["f"] - 1.96 * high["twoway"]["f_se"],
            high["twoway"]["f"] + 1.96 * high["twoway"]["f_se"],
        ],
        "verdict": "Model R. The pass level price is statistically zero and "
                   "the break does not move with the stratum table above M=5.",
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote %s" % OUT)
    print("f = %.1f +- %.1f us   CI95 [%.1f, %.1f]" % (
        doc["f_hat_us"], doc["f_se_us"], *doc["f_ci95_us"]))


if __name__ == "__main__":
    main()
