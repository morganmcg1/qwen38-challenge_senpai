"""E148 Finding 241 population tests, requested in advisor feedback F3.6.

harness=local for both statistics. Both read the R-B lattice only; no GPU, no
scored-surface change.

1. `e148_lattice_slots_population` — fit the one-mode band plus the 0.3 to 0.8
   shoulder to (a) one broad Gaussian and (b) a three-component mixture pinned
   at s/3, 2s/3 and s with a common width, and compare by BIC. This asks
   whether the shoulder is a separate lattice component or the low tail of one
   broad mode.
2. `e148_at_zero_set_step_sensitivity` — recompute the at-zero row set over the
   whole attributable corpus at steps 729, 805.4, 879 and 903.7, and report how
   many rows change class.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RB = HERE / "e148-rb.json"
OUT = HERE / "e148-f241.json"

STEP_US = 879.0
AT_ZERO_THRESHOLD_STEPS = 0.35
CANDIDATE_STEPS_US = [729.0, 805.4, 879.0, 903.7]
BAND_LO_STEPS, BAND_HI_STEPS = 0.30, 1.40
MIN_SIGMA_US = 5.0


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def one_gaussian_bic(xs: list[float]) -> dict[str, float]:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    sigma = max(math.sqrt(var), MIN_SIGMA_US)
    ll = sum(gaussian_logpdf(x, mu, sigma) for x in xs)
    k = 2
    return {"mu_us": mu, "sigma_us": sigma, "loglik": ll, "params": k,
            "bic": k * math.log(n) - 2.0 * ll}


def mixture_loglik(xs: list[float], s: float, sigma: float,
                   weights: tuple[float, float, float]) -> float:
    locs = (s / 3.0, 2.0 * s / 3.0, s)
    total = 0.0
    for x in xs:
        acc = 0.0
        for w, mu in zip(weights, locs):
            if w > 0.0:
                acc += w * math.exp(gaussian_logpdf(x, mu, sigma))
        total += math.log(acc) if acc > 0.0 else -1e9
    return total


def fit_mixture(xs: list[float]) -> dict[str, float]:
    """EM over the three pinned components, with s and sigma on a grid.

    s and sigma are scanned rather than optimised so the result does not depend
    on an optimiser's starting point; the weights are solved to convergence by
    EM at every grid point.
    """
    best: dict[str, float] | None = None
    n = len(xs)
    s_grid = [700.0 + 2.0 * i for i in range(151)]        # 700 .. 1000
    sigma_grid = [20.0 + 5.0 * i for i in range(37)]      # 20 .. 200
    for s in s_grid:
        locs = (s / 3.0, 2.0 * s / 3.0, s)
        for sigma in sigma_grid:
            w = [1.0 / 3.0] * 3
            for _ in range(200):
                num = [0.0, 0.0, 0.0]
                for x in xs:
                    dens = [w[j] * math.exp(gaussian_logpdf(x, locs[j], sigma))
                            for j in range(3)]
                    tot = sum(dens)
                    if tot <= 0.0:
                        continue
                    for j in range(3):
                        num[j] += dens[j] / tot
                new = [v / n for v in num]
                if max(abs(new[j] - w[j]) for j in range(3)) < 1e-9:
                    w = new
                    break
                w = new
            ll = mixture_loglik(xs, s, sigma, (w[0], w[1], w[2]))
            if best is None or ll > best["loglik"]:
                best = {"s_us": s, "sigma_us": sigma, "loglik": ll,
                        "w_third": w[0], "w_two_thirds": w[1], "w_full": w[2]}
    assert best is not None
    # Free parameters: s, sigma, and two of the three weights.
    k = 4
    best["params"] = k
    best["bic"] = k * math.log(n) - 2.0 * best["loglik"]
    return best


def main() -> None:
    rb = json.loads(RB.read_text())
    points = rb["lattice_points"]
    k_us = [(p["row"], p["steps_exact"] * STEP_US) for p in points]

    band = [v for _, v in k_us
            if BAND_LO_STEPS * STEP_US <= v <= BAND_HI_STEPS * STEP_US]
    one = one_gaussian_bic(band)
    mix = fit_mixture(band)
    delta_bic = one["bic"] - mix["bic"]

    if delta_bic > 10.0:
        shoulder = ("component: the three-slot mixture beats one broad "
                    "Gaussian by more than 10 BIC")
    elif delta_bic < -10.0:
        shoulder = ("tail: one broad Gaussian beats the three-slot mixture by "
                    "more than 10 BIC, so Finding 241 holds on the null block "
                    "only")
    else:
        shoulder = ("undecided: |delta BIC| <= 10, the population cannot "
                    "separate the two models")

    sensitivity = []
    baseline: set[str] | None = None
    for step in CANDIDATE_STEPS_US:
        thresh = AT_ZERO_THRESHOLD_STEPS * step
        at_zero = {row for row, v in k_us if abs(v) < thresh}
        if step == STEP_US:
            baseline = at_zero
        sensitivity.append({"step_us": step, "threshold_us": round(thresh, 2),
                            "n_at_zero": len(at_zero), "rows": sorted(at_zero)})
    assert baseline is not None
    rc = json.loads((HERE / "e148-rc.json").read_text())
    top_ten = {r["row"] for r in sorted(rc["decode_ranked_table"],
                                        key=lambda r: r["corrected_total_pct"]
                                        )[:10]}
    k_by_row = dict(k_us)
    changed_max = 0
    for entry in sensitivity:
        rows = set(entry.pop("rows"))
        movers = sorted(rows ^ baseline)
        entry["rows_changing_class_vs_879"] = len(movers)
        entry["movers"] = [{"row": r, "k_us": round(k_by_row[r], 1),
                            "in_rc_top_ten": r in top_ten} for r in movers]
        changed_max = max(changed_max, len(movers))

    magnitudes = sorted(abs(v) for _, v in k_us)
    largest_at_zero = max((abs(v) for _, v in k_us
                           if abs(v) < AT_ZERO_THRESHOLD_STEPS * STEP_US),
                          default=0.0)
    smallest_refused = min((abs(v) for _, v in k_us
                            if abs(v) >= AT_ZERO_THRESHOLD_STEPS * STEP_US),
                           default=float("inf"))

    out = {
        "harness": "local",
        "frame": "decode frame, fitted k microseconds per drafting round",
        "step_us_campaign_constant": STEP_US,
        "band_selection_steps": [BAND_LO_STEPS, BAND_HI_STEPS],
        "band_n": len(band),
        "one_gaussian": one,
        "three_slot_mixture": mix,
        "e148_lattice_slots_population": {
            "delta_bic_one_minus_mixture": round(delta_bic, 2),
            "verdict": shoulder,
        },
        "e148_at_zero_set_step_sensitivity": {
            "rows_considered": len(k_us),
            "per_step": sensitivity,
            "max_rows_changing_class_vs_879": changed_max,
            "largest_k_at_zero_us": round(largest_at_zero, 1),
            "smallest_k_refused_us": round(smallest_refused, 1),
            "invariance_window_us": [
                round(largest_at_zero / AT_ZERO_THRESHOLD_STEPS, 1),
                round(smallest_refused / AT_ZERO_THRESHOLD_STEPS, 1)],
        },
        "corpus_k_abs_quantiles_us": {
            "p50": round(magnitudes[len(magnitudes) // 2], 1),
            "p90": round(magnitudes[int(0.90 * len(magnitudes))], 1),
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"band n = {len(band)} rows in steps [{BAND_LO_STEPS}, "
          f"{BAND_HI_STEPS}]")
    print(f'one Gaussian   mu {one["mu_us"]:.1f} sigma {one["sigma_us"]:.1f} '
          f'BIC {one["bic"]:.2f}')
    print(f'3-slot mixture s {mix["s_us"]:.1f} sigma {mix["sigma_us"]:.1f} '
          f'w {mix["w_third"]:.3f}/{mix["w_two_thirds"]:.3f}/'
          f'{mix["w_full"]:.3f} BIC {mix["bic"]:.2f}')
    print(f"delta BIC (one - mixture) = {delta_bic:+.2f}  -> {shoulder}")
    print()
    for entry in sensitivity:
        print(f'step {entry["step_us"]:>7.1f}  thresh '
              f'{entry["threshold_us"]:>7.2f}  n_at_zero '
              f'{entry["n_at_zero"]:>4d}  changed '
              f'{entry["rows_changing_class_vs_879"]:>3d}')
    print(f"largest k at zero {largest_at_zero:.1f} us, smallest k refused "
          f"{smallest_refused:.1f} us")


if __name__ == "__main__":
    main()
