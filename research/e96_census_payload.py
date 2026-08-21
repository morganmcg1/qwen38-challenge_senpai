#!/usr/bin/env python3
"""Turn one isolated census table into an E96 W&B analysis payload.

    usage: research/e96_census_payload.py KERNELS_W5_TXT OUT_JSON

Input is the saved stdout of `research/e95_verify_census.py kernels ... --width=5`
run on a leg with `MLX_E58_BUFFER_LIMIT_OPS=0`, which is the only setting that
puts one dispatch in one command buffer.

The payload carries three columns per family: the E95 modelled line the E96
brief was written from, the isolated census time measured here, and the dose
slope E96 measured directly. The two ratios are the rung 3 deliverable.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# us per round the E95 least-squares `fixed` split attributed to each family.
MODELLED_US = {
    "GDN recurrent step": 8112.6,
    "fused residual + RMSNorm": 1187.1,
    "GDN prework": 475.3,
    "q_norm + k_norm + RoPE": 210.5,
    "full-attention KV cache write": 166.5,
}
# us per round E96 measured with a bit-exact repeat dose ladder.
DOSE_US = {
    "GDN recurrent step": 861.0,
    "fused residual + RMSNorm": 298.0,
}
# Families whose bytes are the transformed weight stream itself.
STREAMING = ("MLP gate_up", "out_proj and MLP down_proj", "GDN in_proj",
             "lm_head", "full-attention fused QKV")

ROW = re.compile(
    r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S.*)$")


def family(label):
    for key in MODELLED_US:
        if label.startswith(key):
            return key
    return None


def main():
    text = pathlib.Path(sys.argv[1]).read_text()
    rows = []
    for line in text.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        us, n, per, mb, gbs, label = match.groups()
        rows.append([label.strip(), float(us), float(n), float(per),
                     float(mb), float(gbs)])
    rounds = int(re.search(r"rounds=(\d+)", text).group(1))

    stream_us = sum(r[1] for r in rows
                    if any(r[0].startswith(s) for s in STREAMING))
    stream_mb = sum(r[1 + 3] * r[2] for r in rows
                    if any(r[0].startswith(s) for s in STREAMING))
    total_us = sum(r[1] for r in rows)

    metrics = {
        "census_rounds": rounds,
        "census_verify_width": 5,
        "census_ops_per_buffer": 0,
        "census_isolated_total_us_per_round": total_us,
        "census_streaming_us_per_round": stream_us,
        "census_streaming_mb_per_round": stream_mb,
        "census_streaming_gbs": stream_mb / 1024.0 / (stream_us / 1e6),
        "census_streaming_share_of_isolated_total_pct":
            100.0 * stream_us / total_us,
        "census_non_streaming_us_per_round": total_us - stream_us,
    }
    compare = []
    for label, us, n, per, mb, gbs in rows:
        key = family(label)
        if key is None:
            continue
        modelled = MODELLED_US[key]
        dose = DOSE_US.get(key)
        compare.append([
            key, modelled, us, dose,
            modelled / us,
            (modelled / dose) if dose else None,
            (us / dose) if dose else None,
            n, per, gbs,
        ])
        slug = key.split(",")[0].replace(" ", "_").replace("+", "and")
        metrics[f"{slug}_isolated_us"] = us
        metrics[f"{slug}_modelled_us"] = modelled
        metrics[f"{slug}_modelled_over_isolated"] = modelled / us
        if dose:
            metrics[f"{slug}_dose_us"] = dose
            metrics[f"{slug}_modelled_over_dose"] = modelled / dose
            metrics[f"{slug}_isolated_over_dose"] = us / dose

    payload = {
        "config": {
            "census_leg": pathlib.Path(sys.argv[1]).parent.name,
            "buffer_limit_ops": 0,
            "buffer_limit_mb": 1,
            "timing_valid": False,
            "public_drift_tripwire_run": True,
        },
        "metrics": metrics,
        "tables": {
            "isolated_kernels": {
                "columns": ["kernel", "us_per_round", "n_per_round",
                            "us_per_dispatch", "mb_per_dispatch", "gbs"],
                "rows": rows,
            },
            "census_versus_truth": {
                "columns": ["family", "modelled_us", "isolated_us", "dose_us",
                            "modelled_over_isolated", "modelled_over_dose",
                            "isolated_over_dose", "n_per_round",
                            "us_per_dispatch", "gbs"],
                "rows": compare,
            },
        },
    }
    pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n")
    for key, value in metrics.items():
        print(f"   {key:<52} {value}")


if __name__ == "__main__":
    main()
