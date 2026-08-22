#!/usr/bin/env python3
"""Classify the draw mode of a ranked receipt, then read it mode corrected.

    usage: research/e126_modeindex.py RECEIPT.json [--label ID]
           research/e126_modeindex.py --board PREFIX [PREFIX ...]
           research/e126_modeindex.py --selftest
           research/e126_modeindex.py --anchor-check

The published median of one ranked run is not comparable with another run's
published median, because the hidden prompt pool draws in two modes (Rule 63).
F76 separates them with a zero-sum weighting of the per-prompt candidate decode
times:

    index = sum_p  w_p * 100 * ln(mtp_seconds_per_token_mean_p)

The weights sum to zero, so a uniform multiplicative speedup cancels exactly
and only a non-uniform change to the tree moves the level. One mode flip is
1.000 index units, same-mode sd is 0.116, per-run noise is 0.0817 and the
threshold is -12.9. Weights, calibration and anchors are the advisor's, from
`_advisor_scratch/modeindex.py`, relayed on PR #127.

A single ranked run is never a regression before it is classified (Rule 71).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

WEIGHTS = {
    "plutarch": -0.3852,
    "drama": +0.0215,
    "travel": +0.4945,
    "beagle": +0.2068,
    "medicine": -0.1480,
    "republic": -0.0917,
    "essays": -0.0041,
    "botany": -0.0939,
}

THRESHOLD = -12.9
FAST_ANCHOR = -13.3
SLOW_ANCHOR = -12.4
MODE_FLIP_UNITS = 1.000
SAME_MODE_SD = 0.116
PER_RUN_NOISE = 0.0817

# Every anchor the advisor recomputed against the live board this cycle. A
# published median is only comparable with another one at the same mode.
ANCHORS = {
    "bc070b7b": ("francip", 3.35922017, -13.4103, "promoted"),
    "7358c89f": ("newjordan", 3.35206897, -13.3421, "accepted"),
    "51b9bf85": ("vibecodooor", 3.35025879, -13.3497, "accepted"),
    "44559d02": ("morganmcg1", 3.34351272, -13.4917, "rejected"),
    "b8b8b860": ("morganmcg1", 3.33412148, -13.3723, "rejected"),
    "276aa2c2": ("hadakang", 3.33849825, -13.3588, "accepted"),
    "f04b102e": ("morganmcg1", 3.32824629, -13.2906, "accepted"),
    "8819b108": ("audreyt", 3.32794961, -13.1427, "accepted"),
    "7bef7d4c": ("morganmcg1", 3.29792433, -12.5202, "rejected"),
    "ec778a91": ("Amal-David", 3.34664074, -13.4240, "crown content redraw"),
}

# `7bef7d4c` is the only receipt of ours with a published SLOW draw and a known
# corrected value, so it fixes the correction that a slow index implies.
SLOW_REFERENCE = ("7bef7d4c", 3.29792433, 3.34136)

# Pre-registered on PR #127 before `cf9a9eda` existed.
READINGS = (
    (3.3500, None, "the E121 transfer chain is confirmed"),
    (3.3420, 3.3500, "partial. The chain holds in sign, the level is "
                     "over-priced"),
    (None, 3.3420, "the chain is refuted and F85's class table needs "
                   "rebuilding"),
)


def index_of(per_prompt: dict[str, float]) -> float:
    missing = sorted(set(WEIGHTS) - set(per_prompt))
    if missing:
        raise SystemExit("receipt is missing prompts: %s" % ", ".join(missing))
    return sum(w * 100.0 * math.log(per_prompt[p]) for p, w in WEIGHTS.items())


def classify(index: float) -> tuple[str, float]:
    """Mode label and its distance from the threshold, in same-mode sd."""
    label = "fast" if index < THRESHOLD else "slow"
    return label, abs(index - THRESHOLD) / SAME_MODE_SD


def correct(published: float, index: float) -> float:
    """Published median restated on the fast-mode scale.

    The correction is anchored on `7bef7d4c`, the one receipt of ours with a
    published slow draw and a corrected value the advisor has already fixed.
    A fast draw needs no correction.
    """
    if classify(index)[0] == "fast":
        return published
    _, raw, corrected = SLOW_REFERENCE
    return published * (corrected / raw)


def reading(corrected: float) -> str:
    for lo, hi, text in READINGS:
        if (lo is None or corrected >= lo) and (hi is None or corrected < hi):
            return text
    raise AssertionError("readings must cover the line")


def per_prompt_from(doc: dict) -> tuple[dict[str, float], float | None]:
    """Pull the per-prompt candidate decode times out of a Yukon receipt.

    Yukon has published several shapes for this table, so every plausible
    container is searched rather than assuming one. The value wanted is the
    candidate mean seconds per token, never the ratio and never the baseline.
    """
    wanted = ("mtp_seconds_per_token_mean", "candidate_mtp_seconds_per_token_"
              "mean", "candidate_seconds_per_token_mean")
    for container in ("per_prompt", "prompts", "legs", "results", "rows"):
        rows = doc.get(container)
        if rows is None and isinstance(doc.get("metrics"), dict):
            rows = doc["metrics"].get(container)
        if not rows:
            continue
        if isinstance(rows, dict):
            rows = [dict(row, prompt=name) for name, row in rows.items()]
        out = {}
        for row in rows:
            name = row.get("prompt") or row.get("name") or row.get("id")
            for key in wanted:
                if row.get(key):
                    out[str(name)] = float(row[key])
                    break
        if len(out) >= len(WEIGHTS):
            return out, doc.get("score") or doc.get("published_score")
    raise SystemExit(
        "no per-prompt candidate seconds per token found. Top-level keys: %s"
        % ", ".join(sorted(doc)))


BOARD_CACHE = pathlib.Path("/tmp/yukon-board/full.json")

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def board_rows() -> list[dict]:
    if not BOARD_CACHE.exists():
        raise SystemExit("run `python3 research/board_per_prompt.py fetch` "
                         "first; %s is absent" % BOARD_CACHE)
    return json.loads(BOARD_CACHE.read_text())


def board_receipt(prefix: str) -> dict:
    """Restate one Yukon board row in the receipt shape this tool reads."""
    hits = [r for r in board_rows() if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d board rows" % (prefix, len(hits)))
    row = hits[0]
    metrics = row.get("officialMetrics") or {}
    per_prompt = []
    for entry in metrics.get("per_prompt") or []:
        name = PROMPT_NAMES.get(str(entry.get("prompt_sha256", ""))[:8])
        if name:
            per_prompt.append(dict(entry, prompt=name))
    return {
        "submission_id": row.get("id"),
        "solver": row.get("solverUsername"),
        "status": row.get("promotionStatus") or row.get("status"),
        "score": row.get("officialScore"),
        "per_prompt": per_prompt,
    }


def anchor_check() -> int:
    """Recompute every advisor anchor index from its own board row.

    The advisor relayed ten anchor indices but not the per-prompt vectors
    behind them. The Yukon list endpoint carries those vectors, so the weight
    vector and the formula can be checked numerically rather than only at the
    decision layer.
    """
    bad = []
    print("  %-9s %-12s %11s %9s %9s %8s" % ("id", "solver", "published",
                                             "advisor", "recomputed", "delta"))
    for key, (solver, published, index, _status) in ANCHORS.items():
        try:
            doc = board_receipt(key[:8])
        except SystemExit as exc:
            bad.append("%s: %s" % (key, exc))
            continue
        got, _ = per_prompt_from(doc)
        mine = index_of(got)
        delta = mine - index
        print("  %-9s %-12s %11.8f %+9.4f %+10.4f %+8.4f"
              % (key[:8], solver, published, index, mine, delta))
        if abs(delta) > 0.5 * SAME_MODE_SD:
            bad.append("%s recomputed %+.4f against the advisor's %+.4f"
                       % (key, mine, index))
    for line in bad:
        print("  FAIL", line)
    print("  anchor check %s" % ("FAILED" if bad else "passed"))
    return 1 if bad else 0


def selftest() -> int:
    """Reproduce the advisor's anchor arithmetic, and prove it can fail.

    The anchor indices are given, not the per-prompt vectors behind them, so
    the check is on the decision layer: every anchor labelled FAST must
    classify fast, the one labelled SLOW must classify slow, and the slow
    correction must reproduce the value the advisor published for it.
    """
    bad = []
    for key, (solver, published, index, _status) in ANCHORS.items():
        label = classify(index)[0]
        expect = "slow" if key == SLOW_REFERENCE[0] else "fast"
        if label != expect:
            bad.append("%s (%s) classified %s, expected %s"
                       % (key, solver, label, expect))
        print("  %-9s %-12s published %.8f  index %+8.4f  %s  corrected "
              "%.5f" % (key, solver, published, index, label.upper(),
                        correct(published, index)))

    got = correct(SLOW_REFERENCE[1], -12.5202)
    if abs(got - SLOW_REFERENCE[2]) > 5e-5:
        bad.append("slow correction gave %.5f, expected %.5f"
                   % (got, SLOW_REFERENCE[2]))

    # The relayed weights sum to -1e-4, not to zero, so a uniform speedup
    # leaks |sum(w)| * 100 * |ln r| units instead of cancelling exactly. The
    # leak must stay far below the per-run noise or the index would confuse a
    # level change with a mode flip.
    flat = {p: 0.0300 for p in WEIGHTS}
    faster = {p: v * 0.95 for p, v in flat.items()}
    leak = abs(index_of(flat) - index_of(faster))
    if leak > 0.10 * PER_RUN_NOISE:
        bad.append("a uniform 5 percent speedup moved the index %.5f units"
                   % leak)
    print("  weight sum %+.6f, a uniform 5 percent speedup leaks %.6f units, "
          "%.2f percent of the %.4f per-run noise"
          % (sum(WEIGHTS.values()), leak, 100.0 * leak / PER_RUN_NOISE,
             PER_RUN_NOISE))

    # Positive control: the comparison must be able to fail. Bias the two
    # prompts carrying the largest opposite weights and the index must move by
    # more than one mode flip.
    skewed = dict(flat)
    skewed["travel"] *= 1.10
    skewed["plutarch"] *= 0.90
    moved = abs(index_of(skewed) - index_of(flat))
    if moved < MODE_FLIP_UNITS:
        bad.append("positive control moved only %.4f units" % moved)
    print("  positive control moves %.4f units, more than the %.3f unit mode "
          "flip" % (moved, MODE_FLIP_UNITS))

    for line in bad:
        print("  FAIL", line)
    print("  selftest %s" % ("FAILED" if bad else "passed"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", nargs="?", type=pathlib.Path)
    ap.add_argument("--label", default="receipt")
    ap.add_argument("--board", nargs="+", metavar="PREFIX")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--anchor-check", action="store_true")
    args = ap.parse_args()

    if args.anchor_check:
        return anchor_check()
    if args.board:
        rc = 0
        for prefix in args.board:
            doc = board_receipt(prefix)
            rc |= report(doc, "%s (%s, %s)" % (doc["submission_id"][:8],
                                               doc["solver"], doc["status"]))
        return rc
    if args.selftest or args.receipt is None:
        return selftest()

    return report(json.loads(args.receipt.read_text()), args.label)


def report(doc: dict, label: str) -> int:
    per_prompt, published = per_prompt_from(doc)
    index = index_of(per_prompt)
    mode, sd = classify(index)

    print("harness=ranked. %s, %d prompts" % (label, len(per_prompt)))
    for prompt, weight in sorted(WEIGHTS.items(), key=lambda kv: -abs(kv[1])):
        value = per_prompt[prompt]
        print("  %-10s w %+7.4f  s/tok %.8f  contribution %+8.4f"
              % (prompt, weight, value, weight * 100.0 * math.log(value)))
    print("\nindex %+8.4f  -> %s, %.1f same-mode sd from the -12.9 threshold"
          % (index, mode.upper(), sd))
    print("fast anchor %.1f, slow anchor %.1f, one flip %.3f units"
          % (FAST_ANCHOR, SLOW_ANCHOR, MODE_FLIP_UNITS))

    if published is None:
        print("\nno published median in the receipt, so no corrected reading")
        return 0
    corrected = correct(float(published), index)
    print("\npublished %.8f  ->  mode corrected %.5f" % (published, corrected))
    print("pre-registered reading: %s" % reading(corrected))
    print("crown %.5f (bc070b7b), crown content redraw %.5f (ec778a91)"
          % (ANCHORS["bc070b7b"][1], ANCHORS["ec778a91"][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
