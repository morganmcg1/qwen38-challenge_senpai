#!/usr/bin/env python3
"""Cross-module equivalence check for the E150 R2 one-shot reference arm.

`e150_r2_sequential` re-implements the linearised depth rule directly on the
hazard vector so it can interleave sequential readbacks. At AUC 0.5 that
re-implementation must collapse onto the arm R0.5 already priced through the
shipped walk. The R2 module's own control only compares its sequential arms
against its own one-shot arm, so it cannot see a shared offset. This script
closes that gap by pricing both implementations on one environment.
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e140_cells import install  # noqa: E402
from e150_lib import build_env  # noqa: E402
import e150_r05  # noqa: E402
import e150_r2_sequential as seq  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.45642623901367185)
    args = ap.parse_args()

    env = build_env(windows=args.windows, seeds=args.seeds)
    install(env.measured)

    print("mu %.9f  windows %d  seeds %d" % (args.mu, args.windows, args.seeds))
    print("\n  %-28s %10s %10s %10s" % ("cell", "r05", "r2seq", "diff pp"))
    worst = 0.0
    for name, clamp in (("noclamp", {}), ("clamped", None)):
        a = e150_r05.price(env, "shipped", "ratio", clamp, args.mu)
        b = seq.price(env, args.mu, 0.5, clamp, "oneshot")
        diff = b["median_pct_mean"] - a["median_pct_mean"]
        worst = max(worst, abs(diff))
        print("  %-28s %+10.4f %+10.4f %+10.4f"
              % (name, a["median_pct_mean"], b["median_pct_mean"], diff))
        print("  %-28s %10.4f %10.4f"
              % ("  weighted depth", a["weighted_mean_depth"],
                 b["weighted_mean_depth"]))

    print("\n  worst |diff| %.6f pp" % worst)
    print("  equivalent %s" % (worst < 1e-9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
