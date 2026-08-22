#!/usr/bin/env python3
"""Pre-register the E129 receipt predictions before the submission is sent.

    python3 research/e129_prereg.py --ref 0c6191b7

The instrument is F154: the unweighted 8-prompt mean of
``mtp_seconds_per_token_mean``. It is 5.17x sharper than the published median
because it uses all eight prompts instead of the two central order statistics.
Detection uses this mean; pricing still uses the F83 ranked weights.

harness=ranked throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ORDER = [
    "plutarch", "drama", "travel", "beagle",
    "republic", "essays", "medicine", "botany",
]
# F19 section 4. Each theory predicts a fractional change in candidate-leg
# seconds per token for templating plus the {6:6:4, 7:7:4} one-pass table.
THEORIES = {
    "pass-count break (Edward's M-curve)": -0.086,
    "instruction census, F87 haircut": -0.0256,
    "askeladd compile-only model": -0.026,
    "Edward cross-sectional f (null)": +0.0005,
}
# The advisor's own band, F19.
BAND = (-0.09, -0.03)
CENTRAL = -0.05
# F154 standard error of the instrument, as a fraction.
SE = 0.000187
# F76 mode index. A weighted sum over 100*ln(mtp_seconds_per_token_mean_p) that
# separates the two host states the ranked runner alternates between. The two
# states differ by roughly 0.8 % of candidate-leg time, which is 43 times the
# F154 standard error, so the state must be classified before any receipt pair
# is read as a code effect.
F76_WEIGHTS = {
    "plutarch": -0.3852, "drama": +0.0215, "travel": +0.4945, "beagle": +0.2068,
    "medicine": -0.1480, "republic": -0.0917, "essays": -0.0041, "botany": -0.0939,
}
F76_THRESHOLD = -12.9


def load() -> dict:
    raw = json.load(BOARD.open())
    rows = raw if isinstance(raw, list) else raw["submissions"]
    out = {}
    for r in rows:
        sid = (r.get("id") or "")[:8]
        om = r.get("officialMetrics") or {}
        if om.get("per_prompt"):
            out[sid] = r
    return out


def leg_means(row: dict) -> dict[str, float]:
    return {
        NAMES[e["prompt_sha256"][:8]]: e["mtp_seconds_per_token_mean"]
        for e in row["officialMetrics"]["per_prompt"]
    }


def mode_index(legs: dict[str, float]) -> float:
    return sum(F76_WEIGHTS[n] * 100.0 * math.log(legs[n]) for n in ORDER)


def pick(by: dict, prefix: str) -> tuple[str, dict]:
    hits = [k for k in by if k.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d receipts" % (prefix, len(hits)))
    return hits[0], by[hits[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="receipt id prefix to predict against")
    ap.add_argument("--peer", action="append", default=[],
                    help="extra receipt id prefix to classify alongside the reference")
    args = ap.parse_args()

    by = load()
    sid, row = pick(by, args.ref)
    hits = [sid]
    legs = leg_means(row)
    ref = sum(legs[n] for n in ORDER) / len(ORDER)

    print("harness=ranked  instrument=F154 unweighted 8-prompt candidate-leg mean")
    print("reference receipt %s  solver %s  published %.6f  %s"
          % (hits[0], row.get("solverUsername"), row["officialScore"],
             row["createdAt"][:19]))
    print("reference unweighted mean  %.6f s/token" % ref)
    print("per prompt: %s"
          % "  ".join("%s=%.6f" % (n, legs[n]) for n in ORDER))
    print("instrument se              %.6f s/token  (%.4f %%)" % (ref * SE, SE * 100))
    print()
    print("%-38s %9s %12s" % ("theory", "predicted", "predicted mean"))
    for name, frac in THEORIES.items():
        print("%-38s %+8.2f%% %12.6f" % (name, frac * 100, ref * (1 + frac)))
    print()
    print("%-38s %+8.2f%% %12.6f" % ("advisor band, fast edge", BAND[0] * 100, ref * (1 + BAND[0])))
    print("%-38s %+8.2f%% %12.6f" % ("advisor band, central", CENTRAL * 100, ref * (1 + CENTRAL)))
    print("%-38s %+8.2f%% %12.6f" % ("advisor band, slow edge", BAND[1] * 100, ref * (1 + BAND[1])))
    print("%-38s %+8.2f%% %12.6f" % ("census refutation threshold", -1.0, ref * 0.99))

    print()
    print("F76 host-state classification, threshold %.2f (below is the fast state)"
          % F76_THRESHOLD)
    print("%-9s %-16s %10s %10s %9s" % ("receipt", "solver", "published", "mean", "F76"))
    for prefix in [args.ref] + args.peer:
        psid, prow = pick(by, prefix)
        plegs = leg_means(prow)
        pmean = sum(plegs[n] for n in ORDER) / len(ORDER)
        idx = mode_index(plegs)
        print("%-9s %-16s %10.6f %10.6f %9.4f %s"
              % (psid, (prow.get("solverUsername") or "")[:16], prow["officialScore"],
                 pmean, idx, "fast" if idx < F76_THRESHOLD else "slow"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
