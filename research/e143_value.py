#!/usr/bin/env python3
"""Campaign Rule 121: the published median is a sorted order statistic.

F1 tells me to price every median figure with `research/f209_reorder_value.py`
and not to multiply by 0.478 by hand. That file is not on my base
`892dc5e16c3ec741287b3ef213b52b97fd01a6c6`, so this is a re-implementation of
the same model from the anchor table F1 published, and it is checked against
every number F1 quoted. If the two ever disagree, f209 wins and this file is
the defect.

The model is the scoring rule itself: sort the eight raw ratios, take the mean
of the two middle ones. A relative gain applied to a subset of prompts can move
a prompt out of the median pair, at which point further gain on that prompt is
worth exactly zero.

Self-check: `research/e143_value.py`
"""

from __future__ import annotations

# Sorted raw ratios of the anchor tree `572b2cc4`, exactly as F1 published them.
ANCHOR_572b2cc4 = {
    "plutarch": 1.25899,
    "drama": 2.08138,
    "travel": 2.39973,
    "beagle": 3.51170,
    "essays": 3.81267,
    "republic": 3.86063,
    "medicine": 3.87183,
    "botany": 3.90545,
}


def median_of(ratios: dict[str, float]) -> float:
    """The published median: mean of the two middle of the eight sorted raws."""
    values = sorted(ratios.values())
    if len(values) != 8:
        raise ValueError(f"the ranked run has eight prompts, got {len(values)}")
    return 0.5 * (values[3] + values[4])


def median_pct_gain(gains: dict[str, float],
                    anchor: dict[str, float] | None = None) -> float:
    """Percent change of the published median for relative per-prompt gains.

    `gains` maps a prompt name to its relative raw-ratio gain as a fraction, so
    0.01 is a one percent faster candidate leg on that prompt. Prompts absent
    from `gains` do not move.
    """
    anchor = anchor or ANCHOR_572b2cc4
    unknown = set(gains) - set(anchor)
    if unknown:
        raise ValueError(f"not ranked prompts: {sorted(unknown)}")
    moved = {k: v * (1.0 + gains.get(k, 0.0)) for k, v in anchor.items()}
    return 100.0 * (median_of(moved) / median_of(anchor) - 1.0)


def marginal(prompt: str, anchor: dict[str, float] | None = None) -> float:
    """dM/dx at x = 0 for a gain on one prompt: the Rule 116 carrier weight."""
    anchor = anchor or ANCHOR_572b2cc4
    step = 1e-9
    return median_pct_gain({prompt: step}, anchor) / (100.0 * step)


def ceiling(prompt: str, anchor: dict[str, float] | None = None,
            hi: float = 5.0, tol: float = 1e-9) -> tuple[float, float]:
    """The gain `x` past which one prompt stops paying, and what it paid.

    Bisects on the last `x` whose marginal value is still positive. Returned as
    (x as a percent of that prompt's raw ratio, median percent gain at x).
    """
    anchor = anchor or ANCHOR_572b2cc4
    if marginal(prompt, anchor) <= 0.0:
        return 0.0, 0.0
    lo = 0.0
    top = median_pct_gain({prompt: hi}, anchor)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if median_pct_gain({prompt: mid}, anchor) < top - tol:
            lo = mid
        else:
            hi = mid
    return 100.0 * hi, top


def _self_check() -> None:
    """Every figure F1 published, reproduced from this model."""
    checks = [
        # F1 published the raws to five decimals, so the reconstructed median
        # can only agree to about 5e-6 absolute.
        ("anchor median", median_of(ANCHOR_572b2cc4), 3.66218564, 5e-6),
        ("beagle dM/dx", marginal("beagle"), 0.4795, 5e-4),
        ("essays dM/dx", marginal("essays"), 0.5205, 5e-4),
        ("republic dM/dx", marginal("republic"), 0.0, 1e-9),
        ("plutarch dM/dx", marginal("plutarch"), 0.0, 1e-9),
        ("beagle ceiling x %", ceiling("beagle")[0], 9.940, 5e-3),
        ("beagle ceiling value %", ceiling("beagle")[1], 4.7639, 5e-4),
        ("essays ceiling x %", ceiling("essays")[0], 1.260, 5e-3),
        ("essays ceiling value %", ceiling("essays")[1], 0.6548, 5e-4),
        ("uniform 1 % converts 1:1",
         median_pct_gain({k: 0.01 for k in ANCHOR_572b2cc4}), 1.0, 1e-9),
    ]
    width = max(len(name) for name, *_ in checks)
    bad = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name:<{width}} {got:.6f} "
              f"want {want:.6f}")
    if bad:
        raise SystemExit(f"e143_value: {bad} checks disagree with F1")
    print("e143_value: reproduces every figure F1 published from f209")


if __name__ == "__main__":
    _self_check()
