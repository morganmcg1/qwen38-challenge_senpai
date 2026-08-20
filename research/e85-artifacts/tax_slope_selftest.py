#!/usr/bin/env python3
"""Check that `research/e85_tax_slope.py` recovers a planted per-buffer price.

Builds synthetic palindromic sessions with a known cost per added intermediate,
a linear thermal drift, and leg noise at the size this host actually shows.
A single 12-leg draw cannot pin the drift nuisance term, so the estimators are
judged the way they are used: over many draws, check that the recovered slope
is unbiased, that the 95 percent interval covers the planted value about 95
percent of the time, and that the reported verdict follows the E85 stop rule.
"""
from __future__ import annotations

import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from e85_tax_slope import read_legs, ols_slope, pass_slopes  # noqa: E402

DRAFT_LEN = 6.358974359
ACCEPTED = 0.877016129
DPT = DRAFT_LEN / (1.0 + DRAFT_LEN * ACCEPTED)
BASE = 0.03174120
DRIFT_PER_LEG = 20e-6
RESID_SD = 135e-6
LEVELS = [0, 48, 192]
REPEATS = 2
DRAWS = 300

HEADER = ("leg\ttax\tmtp_s_per_tok\tserial_s_per_tok\tratio\tmean_draft_len"
          "\taccepted_rate\tmatched\ttemp_in\ttemp_out\tseconds")


def session_text(cost_us_per_buffer: float, seed: int) -> str:
    rng = random.Random(seed)
    order = LEVELS + LEVELS[::-1]
    per_unit = cost_us_per_buffer * 1e-6 * DPT
    lines = [HEADER]
    leg = 0
    for _ in range(REPEATS):
        for tax in order:
            leg += 1
            drift = DRIFT_PER_LEG * (leg - 1)
            mtp = BASE + per_unit * tax + drift + rng.gauss(0.0, RESID_SD)
            serial = 0.07434882 + drift + rng.gauss(0.0, RESID_SD)
            lines.append("\t".join(str(v) for v in (
                leg, tax, mtp, serial, serial / mtp, DRAFT_LEN, ACCEPTED,
                "True", 61.0, 62.0, 215)))
    return "\n".join(lines) + "\n"


def check(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got:.3f} (want {want:.3f} +/- {tol})")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    for planted in (2.0, 14.5):
        print(f"planted cost per buffer = {planted} us, {DRAWS} draws")
        ols_hat: list[float] = []
        pass_hat: list[float] = []
        covered = 0
        with tempfile.TemporaryDirectory() as tmp:
            legs = Path(tmp) / "legs.tsv"
            for draw in range(DRAWS):
                legs.write_text(session_text(planted, seed=draw))
                rows = read_legs(legs)
                ols = ols_slope(rows, "mtp_s_per_tok")
                ols_hat.append(ols["slope"] * 1e6 / DPT)
                pass_hat.append(
                    pass_slopes(rows, "mtp_s_per_tok", len(LEVELS))["slope"] * 1e6 / DPT)
                lo = ols["ci95_lo"] * 1e6 / DPT
                hi = ols["ci95_hi"] * 1e6 / DPT
                covered += lo <= planted <= hi

        check("ols mean", statistics.fmean(ols_hat), planted, 0.25)
        check("pass mean", statistics.fmean(pass_hat), planted, 0.25)
        check("ols ci95 coverage %", 100.0 * covered / DRAWS, 95.0, 5.0)
        print(f"  ols sd across draws = {statistics.stdev(ols_hat):.3f} us/buffer")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "legs.tsv").write_text(session_text(planted, seed=0))
            out = subprocess.run(
                [sys.executable, str(RESEARCH / "e85_tax_slope.py"), str(root)],
                capture_output=True, text=True, check=True).stdout
        expect = "terminal-negative" if planted < 5.0 else "inconclusive"
        ok = f'"verdict": "{expect}' in out
        print(f"  {'ok  ' if ok else 'FAIL'} verdict starts {expect}")
        if not ok:
            raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
