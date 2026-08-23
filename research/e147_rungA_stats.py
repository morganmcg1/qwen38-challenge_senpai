#!/usr/bin/env python3
"""Read the E147 rung A reduction and print the numbers the report quotes.

The contrast is formed on `seed_prefill_seconds`, the charged component, and
never on the whole timed leg. The prefill is about a quarter of a local leg
here, so a leg-level difference would dilute the effect by roughly 4x and
report a real 1 % component change as 0.25 % of leg time.
"""

import json
import math
import pathlib
import statistics
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "research/e147-rungA.json"
    payload = json.loads(pathlib.Path(path).read_text())
    legs = payload["legs"]

    base = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "base"]
    cand = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "cand"]
    mean_base = statistics.fmean(base)
    mean_cand = statistics.fmean(cand)
    sd_base = statistics.stdev(base)
    sd_cand = statistics.stdev(cand)
    stderr = math.sqrt(sd_base**2 / len(base) + sd_cand**2 / len(cand))
    diff = mean_cand - mean_base
    # Welch, 13 df here, so the two-sided 95 % multiplier is 2.16.
    half = 2.16 * stderr

    print(f"legs                       {len(legs)}")
    print(f"base  n={len(base)} mean {mean_base:.6f} s  sd {sd_base:.6f} "
          f"({100 * sd_base / mean_base:.3f} %)")
    print(f"cand  n={len(cand)} mean {mean_cand:.6f} s  sd {sd_cand:.6f} "
          f"({100 * sd_cand / mean_cand:.3f} %)")
    print(f"diff  {diff:+.6f} s   se {stderr:.6f}   t {diff / stderr:+.2f}")
    print(f"pct   {100 * diff / mean_base:+.4f} %   95 % CI "
          f"[{100 * (diff - half) / mean_base:+.4f}, "
          f"{100 * (diff + half) / mean_base:+.4f}] %")
    print(f"rank separation: max cand {max(cand):.6f} < min base {min(base):.6f}"
          f" -> {max(cand) < min(base)}")

    ranked_spt = 0.001028291
    print(f"local base s/token {mean_base / 512:.7f}   ranked {ranked_spt:.9f}"
          f"   level factor {(mean_base / 512) / ranked_spt:.2f}x")

    for prompt in sorted({leg["prompt"] for leg in legs}):
        rows = [leg for leg in legs if leg["prompt"] == prompt]
        pre = statistics.fmean(
            [r["seed_prefill_seconds"] for r in rows if r["arm"] == "base"])
        leg_s = statistics.fmean(
            [r["decode_seconds"] for r in rows if r["arm"] == "base"])
        print(f"{prompt:<18} prefill {pre:.4f} s is "
              f"{100 * pre / leg_s:.2f} % of the {leg_s:.4f} s timed leg")

    entries = [leg["entry_c"] for leg in legs]
    print(f"entry temp C min {min(entries):.1f} max {max(entries):.1f}")


if __name__ == "__main__":
    main()
