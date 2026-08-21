#!/usr/bin/env python3
"""E94: compare two arms inside one ABBA session, per cap.

usage:
  research/e94_arm_compare.py research/e94-artifacts/rung2.json [BASE_ARM]

Each cap holds four legs in the order A, B, B, A. The arm effect is the mean of
the two B legs against the mean of the two A legs. The session null is the
spread inside each same-arm pair, which shares an arm and differs only in
position.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    doc = json.loads(Path(sys.argv[1]).read_text())
    base_arm = sys.argv[2] if len(sys.argv) > 2 else "ship"
    by_cap = defaultdict(lambda: defaultdict(list))
    for leg in doc["legs"]:
        meta = leg["meta"]
        by_cap[int(meta["e94_cap"])][meta["e94_arm"]].append(leg)

    nulls = []
    print("cap | arm | legs | s/token mean | pair spread % | round us/token mean"
          " | rounds | mean depth | eff draft | acc rate | matched")
    for cap in sorted(by_cap):
        for arm in sorted(by_cap[cap], key=lambda a: a != base_arm):
            legs = by_cap[cap][arm]
            sec = [l["score"]["mtp_seconds_per_token"] for l in legs]
            busy = [l["round_us_per_token"] for l in legs]
            spread = (abs(sec[0] - sec[1]) / statistics.fmean(sec) * 100
                      if len(sec) == 2 else float("nan"))
            if len(sec) == 2:
                nulls.append(spread)
            matched = all(l["score"]["all_tokens_matched"] for l in legs) and \
                all(l["score"]["residual_divergence_count"] == 0 for l in legs)
            print(f"{cap:3d} | {arm:6s} | {len(legs)} | "
                  f"{statistics.fmean(sec):.6f} | {spread:5.2f} | "
                  f"{statistics.fmean(busy):8.0f} | {legs[0]['rounds']:4d} | "
                  f"{legs[0]['mean_chosen_depth']:.3f} | "
                  f"{legs[0]['score']['effective_mean_draft_len']:.4f} | "
                  f"{legs[0]['score']['accepted_draft_rate']:.4f} | {matched}")

    print()
    print("cap | arm | delta s/token % | delta round us/token % | "
          "delta rounds | delta eff draft")
    for cap in sorted(by_cap):
        if base_arm not in by_cap[cap]:
            continue
        base = by_cap[cap][base_arm]
        base_sec = statistics.fmean(
            l["score"]["mtp_seconds_per_token"] for l in base)
        base_busy = statistics.fmean(l["round_us_per_token"] for l in base)
        for arm, legs in by_cap[cap].items():
            if arm == base_arm:
                continue
            sec = statistics.fmean(
                l["score"]["mtp_seconds_per_token"] for l in legs)
            busy = statistics.fmean(l["round_us_per_token"] for l in legs)
            print(f"{cap:3d} | {arm:6s} | {(sec - base_sec) / base_sec * 100:+7.2f} "
                  f"| {(busy - base_busy) / base_busy * 100:+7.2f} | "
                  f"{legs[0]['rounds'] - base[0]['rounds']:+4d} | "
                  f"{legs[0]['score']['effective_mean_draft_len'] - base[0]['score']['effective_mean_draft_len']:+.4f}")

    print()
    print(f"same-arm pair spreads %: {[round(n, 2) for n in nulls]}")
    print(f"session null: max {max(nulls):.2f} %, "
          f"rms {statistics.fmean(n * n for n in nulls) ** 0.5:.2f} %")

    print()
    print("worker digests")
    digests = defaultdict(set)
    for leg in doc["legs"]:
        meta = leg["meta"]
        digests[meta["e94_arm"]].add(meta["worker_sha256"])
    for arm, shas in digests.items():
        print(f"  {arm}: {sorted(s[:16] for s in shas)}")


if __name__ == "__main__":
    main()
