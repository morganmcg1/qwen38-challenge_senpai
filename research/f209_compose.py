#!/usr/bin/env python3
"""Rule 121 composition forecast.

Compose several measured ranked contrasts onto one anchor row and read the
published median the way the workflow computes it: apply each contrast's
PER-PROMPT candidate-leg delta, re-sort the eight raw ratios, average sorted
positions 3 and 4.

A contrast is named as a board pair `A:B`, meaning "the change that turns A
into B". Its per-prompt effect is taken from the candidate leg only, because
the serial leg is not causally reachable from candidate-editable code
(Rule 118, and senpai/verify-ranked-score-boundary.sh).

    raw_new / raw_old = mtp_old / mtp_new

Use `~A:B` to apply the contrast in reverse, which prices REMOVING a
mechanism that A:B added.

usage
  python3 research/f209_compose.py --anchor 572b2cc4 \
      --contrast 08b67f12:1760479a  --label width2 \
      --contrast ed608e64:08b67f12  --label probe015 \
      --contrast ~ed608e64:0b2f0014 --label drop_onepass67
"""
import argparse
import json
import os

BOARD = os.environ.get("YUKON_BOARD", "/tmp/yukon-board/full.json")
NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
ROWS = json.load(open(BOARD))["submissions"]


def per_prompt(sid):
    for r in ROWS:
        if (r.get("id") or "")[:8] != sid:
            continue
        pp = (r.get("officialMetrics") or {}).get("per_prompt")
        if not pp:
            raise SystemExit("%s has no per_prompt evidence" % sid)
        return r, {NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]): e
                   for e in pp}
    raise SystemExit("%s not on the board" % sid)


def median(vec):
    s = sorted(vec.values())
    return 0.5 * (s[3] + s[4])


def ranks(vec):
    return [k for k, _ in sorted(vec.items(), key=lambda kv: kv[1])]


def contrast_factors(spec):
    """Return per-prompt multiplicative factors on the RAW ratio."""
    reverse = spec.startswith("~")
    a, b = spec.lstrip("~").split(":")
    _, va = per_prompt(a)
    _, vb = per_prompt(b)
    out = {}
    for nm in va:
        ma = va[nm]["mtp_seconds_per_token_mean"]
        mb = vb[nm]["mtp_seconds_per_token_mean"]
        f = ma / mb                      # raw gain factor of A -> B
        out[nm] = (1.0 / f) if reverse else f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--contrast", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    args = ap.parse_args()

    row, pp = per_prompt(args.anchor)
    vec = {k: v["raw_ratio_of_means"] for k, v in pp.items()}
    m0 = median(vec)
    print("anchor %s  %s  %.8f" % (args.anchor, row.get("solverUsername"), m0))
    print("  order %s" % " < ".join(ranks(vec)))

    labels = args.label + ["c%d" % i for i in range(len(args.contrast))]
    running = m0
    print("\nper-contrast per-prompt raw gain, percent")
    hdr = "  %-16s" % "contrast" + "".join(
        "%9s" % n for n in ["beagle", "essays", "republic", "medicine",
                            "botany", "drama", "travel", "plutarch"])
    print(hdr)
    for spec, lab in zip(args.contrast, labels):
        f = contrast_factors(spec)
        line = "  %-16s" % lab + "".join(
            "%+8.3f " % ((f[n] - 1.0) * 100.0)
            for n in ["beagle", "essays", "republic", "medicine",
                      "botany", "drama", "travel", "plutarch"])
        print(line)
        vec = {k: v * f[k] for k, v in vec.items()}
        m = median(vec)
        print("      -> median %.8f   step %+7.4f %%   cumulative %+7.4f %%"
              % (m, (m / running - 1.0) * 100.0, (m / m0 - 1.0) * 100.0))
        running = m

    print("\nfinal sorted vector")
    for i, (nm, v) in enumerate(sorted(vec.items(), key=lambda kv: kv[1])):
        tag = "  <== median pair" if i in (3, 4) else ""
        print("  %d %-9s %9.5f%s" % (i, nm, v, tag))
    print("\nFORECAST %.8f   (%+.4f %% on the anchor)"
          % (median(vec), (median(vec) / m0 - 1.0) * 100.0))
    print("  order change: %s" % ("YES" if ranks(vec) != ranks(
        {k: v["raw_ratio_of_means"] for k, v in pp.items()}) else "no"))


if __name__ == "__main__":
    main()
