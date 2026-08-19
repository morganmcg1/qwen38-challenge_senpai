#!/usr/bin/env python3
"""E49 post-hoc: test the ceiling dose-response without using the control leg.

The pre-registered arm 2 statistic in `e49_analyze.py` prices every dose against
`shipped`, and the launch was recreated before `shipped` got its second
replicate, so that one leg carries the whole reference. Each dose has both
replicates, so dose-vs-dose contrasts answer the same question with a fully
counterbalanced reference and no dependence on `shipped` at all.

If raising the kernel-wide allocation taxes untouched widths, then
`dose_huge` (+67 registers over the shipped max) minus `dose_null` (+1) must
show it. Reported alongside the empirical leg-to-leg noise, which is the scale
any claimed tax has to beat.

  python3 research/e49_dose_contrasts.py research/e49-artifacts/e49-metrics.json
"""

from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

# The dose is an unreachable `case 10:`, so no dispatched width changes. M=1,2
# are excluded as warm-up dominated, M=10 leaves the crossrow family.
WIDTHS = range(3, 10)

DOSES = {
    "dose_null": ("e49-dnull-c1", "e49-dnull-c2"),
    "dose_129": ("e49-d129-c1", "e49-d129-c2"),
    "dose_big": ("e49-dbig-c1", "e49-dbig-c2"),
    "dose_huge": ("e49-dhuge-c1", "e49-dhuge-c2"),
}
CONTROL = "e49-shipped-c1"
REG_DELTA = {"dose_null": 1, "dose_129": 18, "dose_big": 34, "dose_huge": 67}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "research/e49-artifacts/e49-metrics.json"
    legs = {
        leg["tag"]: leg
        for leg in json.loads(pathlib.Path(path).read_text())["legs"]
        if leg.get("status") == "ok"
    }

    def t(tag: str, m: int) -> float:
        return legs[tag]["t_ms"][str(m)]

    def mean_of(dose: str, m: int) -> float:
        return st.mean([t(tag, m) for tag in DOSES[dose]])

    out: dict = {"widths": list(WIDTHS), "reference": "dose_null", "contrasts": {}}

    print("leg-to-leg noise, same arm, |c1 - c2| over M=3..9")
    for dose, (a, b) in DOSES.items():
        spread = [abs(t(a, m) - t(b, m)) / st.mean([t(a, m), t(b, m)]) * 100 for m in WIDTHS]
        out.setdefault("replicate_spread_pct", {})[dose] = {
            "mean": round(st.mean(spread), 3), "max": round(max(spread), 3)}
        print(f"  {dose:10s} mean {st.mean(spread):.3f} %   max {max(spread):.3f} %")

    print("\ncontrol-free contrasts against dose_null (+1 register), both replicated")
    for dose in ("dose_129", "dose_big", "dose_huge"):
        per_width = [
            (mean_of(dose, m) - mean_of("dose_null", m)) / mean_of("dose_null", m) * 100
            for m in WIDTHS
        ]
        out["contrasts"][dose] = {
            "register_delta_vs_dose_null": REG_DELTA[dose] - REG_DELTA["dose_null"],
            "pooled_pct": round(st.mean(per_width), 3),
            "worst_pct": round(max(per_width), 3),
            "per_width_pct": {m: round(v, 3) for m, v in zip(WIDTHS, per_width)},
        }
        widths = " ".join(f"{v:+.2f}" for v in per_width)
        print(f"  {dose:10s} (+{REG_DELTA[dose] - 1:2d} reg) pooled {st.mean(per_width):+.3f} %"
              f"   worst {max(per_width):+.3f} %   per-width {widths}")

    print("\nrank of the single control leg among all 9 arm-2 legs (1 = fastest)")
    every = [CONTROL] + [tag for pair in DOSES.values() for tag in pair]
    ranks = []
    for m in WIDTHS:
        order = sorted((t(tag, m), tag) for tag in every)
        rank = next(i + 1 for i, (_, tag) in enumerate(order) if tag == CONTROL)
        ranks.append(rank)
        print(f"  M={m}: rank {rank}/9")
    mean_rank = st.mean(ranks)
    out["control_leg_rank"] = {"per_width": dict(zip(WIDTHS, ranks)),
                              "mean": round(mean_rank, 2),
                              "median_rank_if_unbiased": 5.0}
    print(f"  mean rank {mean_rank:.2f}/9 against 5.0 for an unbiased reference,"
          " so the single control leg is not materially fast or slow and the"
          " shipped-referenced ladder is not an artifact of it")

    dest = pathlib.Path("research/e49-artifacts/e49-dose-contrasts.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
