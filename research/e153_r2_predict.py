#!/usr/bin/env python3
"""Pre-register the local ABBA prediction implied by the E153 R2 microbenchmark.

The isolated-kernel probe (`research/out/e153-r2-timing.json`) measures the
merged-vs-split saving per full-attention layer. This script projects it onto
the gated end-to-end session so the ABBA result can be scored against a number
that existed before the session ran.

Frames (Rule 144): every percentage below is `harness=local`, decode frame,
negative = merged faster. The projection uses the E149 Arm A realised width
histogram for the medpair prompts, which is the best available estimate of how
often the `qL >= 6` guard admits a round on beagle_a and essays_montaigne.
"""

from __future__ import annotations

import json
from pathlib import Path

TIMING = Path("research/out/e153-r2-timing.json")
FULL_ATTENTION_LAYERS = 16
# E149 Arm A realised verify-width histogram, medpair prompts, both arms.
E149_WIDTH_HIST = {2: 14, 3: 134, 4: 114, 5: 132, 6: 86, 7: 18, 8: 188}
# E149 Arm A decode-frame per-prompt conversions: us/round per 1 %.
LOCAL_CONVERSION = {
    "beagle_a": (180.90360304887872, 0.12655319916705654, 0.478),
    "essays_montaigne": (88.27733666929998, 0.07225045990323364, 0.522),
}
GATED_LEG_SD_PCT = 0.052
RANKED_P_M_GE_6 = 0.5861
RULE_134 = 524.5
C1_PREDICTED_US_PER_ROUND = 154.4987


def main() -> None:
    rows = json.loads(TIMING.read_text())["rows"]
    scored = [r for r in rows if r["width"] in (6, 7, 8)]

    per_kl = {}
    for kl in sorted({r["kL"] for r in scored}):
        sel = [r for r in scored if r["kL"] == kl]
        per_kl[kl] = sum(r["saving_us_per_layer"] for r in sel) / len(sel)

    mean_layer = sum(r["saving_us_per_layer"] for r in scored) / len(scored)
    eligible_round_us = mean_layer * FULL_ATTENTION_LAYERS

    total_rounds = sum(E149_WIDTH_HIST.values())
    eligible = sum(v for k, v in E149_WIDTH_HIST.items() if k >= 6)
    p_local = eligible / total_rounds
    local_round_us = eligible_round_us * p_local

    per_prompt = {}
    weighted_pct = 0.0
    for name, (us, pct, weight) in LOCAL_CONVERSION.items():
        conversion = us / pct
        effect = local_round_us / conversion
        per_prompt[name] = {
            "us_per_round_per_pct": conversion,
            "predicted_pct": -effect,
            "weight": weight,
        }
        weighted_pct += weight * effect

    floor = GATED_LEG_SD_PCT * (0.478**2 + 0.522**2) ** 0.5

    out = {
        "probe": "e153_r2_local_abba_prediction",
        "harness": "local",
        "frame": "decode; negative = merged faster",
        "saving_us_per_layer_by_kL": per_kl,
        "saving_us_per_layer_mean_widths_6_8": mean_layer,
        "saving_us_per_eligible_round": eligible_round_us,
        "local_medpair_p_m_ge_6": p_local,
        "local_predicted_us_per_round": local_round_us,
        "per_prompt": per_prompt,
        "local_predicted_pct_medpair": -weighted_pct,
        "medpair_gated_floor_pct": floor,
        "predicted_sigma_vs_floor": weighted_pct / floor,
        "ranked_p_m_ge_6": RANKED_P_M_GE_6,
        "ranked_predicted_us_per_round": eligible_round_us * RANKED_P_M_GE_6,
        "ranked_predicted_pct_rule134": (
            eligible_round_us * RANKED_P_M_GE_6 / RULE_134
        ),
        "c1_predicted_us_per_round": C1_PREDICTED_US_PER_ROUND,
        "microbench_over_c1": (
            eligible_round_us * RANKED_P_M_GE_6 / C1_PREDICTED_US_PER_ROUND
        ),
    }
    Path("research/e153-r2-prediction.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
