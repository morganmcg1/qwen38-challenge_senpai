#!/usr/bin/env python3
"""Print an E210 census document as a readable leg-by-leg idle map.

    usage: research/e210_summary.py DOC.json [DOC.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def show(doc: dict) -> None:
    print("#" * 72)
    print(f"{doc['tag']}  stall_us={doc['stall_us']}  "
          f"matched={doc['all_tokens_matched']}  "
          f"gate_qualified_for_timing={doc['gate_qualified_for_timing']}")
    for name, leg in doc["legs"].items():
        rounds = leg["rounds_analysed"]
        print("=" * 72)
        print(f"{name}  rounds_analysed={rounds} skipped={leg['rounds_skipped']}")
        print(f"  leg span {leg['leg_span_us'] / 1000:.1f} ms | "
              f"parent {leg['parent_leg_us'] / 1000:.1f} ms | "
              f"delta {leg['leg_span_minus_parent_leg_us'] / 1000:.2f} ms")
        print(f"  GPU busy {leg['leg_gpu_busy_us'] / 1000:.1f} ms | "
              f"idle {leg['leg_gpu_idle_us'] / 1000:.1f} ms = "
              f"{leg['leg_gpu_idle_pct']:.3f}% "
              f"(production {leg['leg_gpu_idle_pct_production']:.3f}%)")
        print(f"  instrument idle {leg['instrument_idle_us'] / 1000:.2f} ms")
        print("  segments:")
        for segment, block in leg["segments"].items():
            per = block["idle_us_per_window"]
            print(f"    {segment:24s} span={block['span_us'] / 1000:9.1f}ms "
                  f"busy={block['gpu_busy_us'] / 1000:9.1f}ms "
                  f"idle={block['gpu_idle_us'] / 1000:8.2f}ms "
                  f"idle%={block['gpu_idle_pct_of_segment']:6.2f} "
                  f"median/window={per.get('median', 0):8.1f}us "
                  f"p90={per.get('p90', 0):8.1f}us")
        print("  share of leg span %:",
              {k: round(v, 2) for k, v in leg["segment_share_of_leg_pct"].items()})
        print("  share of leg idle %:",
              {k: round(v, 2) for k, v in leg["idle_share_of_leg_idle_pct"].items()})
        print("  budget crosscheck:",
              {k: (round(v, 2) if isinstance(v, float) else v)
               for k, v in leg["budget_crosscheck"].items()})
        slices = leg["coherent_slices"]
        print("  coherent slices:",
              {k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in slices.items()
               if isinstance(v, (int, float))})
        for phase, block in slices["by_phase"].items():
            print(f"    starts in {phase:16s} n={block['count']:5d} "
                  f"total={block['total_us'] / 1000:8.1f}ms "
                  f"per_round={block['us_per_round']:8.1f}us "
                  f"max={block['max_us']:9.1f}us")
        print("   largest slices:",
              [(e["round"], e["phase"], round(e["us"], 1))
               for e in slices["largest"]])
        print("  phases (per analysed round):")
        for phase, block in sorted(leg["phases"].items(),
                                   key=lambda kv: -kv[1]["gpu_idle_us"]):
            print(f"    {phase:16s} span/round={block['span_us'] / rounds:9.1f}us "
                  f"idle/round={block['gpu_idle_us'] / rounds:8.1f}us "
                  f"idle%={block['gpu_idle_pct_of_segment']:6.2f}")


def main() -> None:
    for path in sys.argv[1:]:
        show(json.loads(Path(path).read_text()))


if __name__ == "__main__":
    main()
