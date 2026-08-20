#!/usr/bin/env python3
"""Check the isolated QMV curve against real decode rounds.

Rung 1 measures `C(M)` with a kernel microbenchmark: each shape is timed on
its own, so nothing overlaps. The depth scheduler pays a different cost, the
verify pass inside a live round, where the GPU overlaps work and the host runs
alongside it. If the microbenchmark curve had the wrong *shape*, the whole E68
premise would rest on an artefact.

The committed per-round traces answer this for free. `Qwen36MTPBlockSession`
emits `d=<draftCount>` and `eval_wall_us`, the GPU-owned segment of the round,
and verify width is `d + 1`. Grouping `eval_wall_us` by width gives the in-situ
curve directly, with no new GPU time.

The comparison that matters is the ratio `in-situ / isolated` at each width. A
constant ratio means the microbenchmark over-states the level but models the
shape correctly, and the E68 depth-price vector is unaffected, because it
divides one measured step by a measured width-1 pass and the constant cancels.

Trace files come from earlier campaign work on earlier bases, so absolute
milliseconds are NOT comparable with the current tip. Only the per-width ratio
is being read here.
"""

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?"
    r"eval_wall_us=(\d+).*?round_us=(\d+)")

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TRACES = [
    "research/results/e37/medicine-rounds.txt",
    "research/results/e37/natural_history-rounds.txt",
]


def parse(path):
    rows = ROUND_RE.findall(pathlib.Path(path).read_text())
    by = collections.defaultdict(list)
    for _round, draft, accepted, eval_us, round_us in rows:
        by[int(draft) + 1].append({
            "eval_ms": int(eval_us) / 1000.0,
            "round_ms": int(round_us) / 1000.0,
            "tokens": int(accepted) + 1,
        })
    return by


def summarise(by, isolated):
    out = {}
    widths = sorted(by)
    prev_eval = None
    prev_round = None
    for w in widths:
        v = by[w]
        ev = statistics.median(x["eval_ms"] for x in v)
        rd = statistics.median(x["round_ms"] for x in v)
        tok = statistics.mean(x["tokens"] for x in v)
        out[w] = {
            "rounds": len(v),
            "eval_ms": ev,
            "round_ms": rd,
            "eval_step_ms": None if prev_eval is None else ev - prev_eval,
            "round_step_ms": None if prev_round is None else rd - prev_round,
            "tokens_per_round": tok,
            "ms_per_token": rd / tok,
            "isolated_ms": isolated.get(w),
            "insitu_over_isolated": (ev / isolated[w]
                                     if isolated.get(w) else None),
        }
        prev_eval, prev_round = ev, rd
    return out


def isolated_curve(path):
    payload = json.loads(pathlib.Path(path).read_text())
    return {int(k): v["median_s"] * 1e3
            for k, v in payload["curve_seconds"]["shipped"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung1",
                    default="research/e68-artifacts/e68-rung1.json")
    ap.add_argument("--trace", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()

    isolated = isolated_curve(REPO / args.rung1)
    traces = args.trace or DEFAULT_TRACES
    payload = {"isolated_ms": isolated, "traces": {}}

    for t in traces:
        by = parse(REPO / t)
        if not by:
            print("no rounds parsed from %s" % t, file=sys.stderr)
            continue
        table = summarise(by, isolated)
        payload["traces"][t] = table
        print("== %s" % t)
        print("   M     n   eval ms  eval step  isolated  in/iso   ms/token")
        print("\n".join(
            "  %2d  %4d  %8.3f  %9s  %8.3f  %6.4f  %9.3f"
            % (w, r["rounds"], r["eval_ms"],
               "" if r["eval_step_ms"] is None
               else "%+.3f" % r["eval_step_ms"],
               r["isolated_ms"] or float("nan"),
               r["insitu_over_isolated"] or float("nan"),
               r["ms_per_token"])
            for w, r in sorted(table.items())))
        ratios = [r["insitu_over_isolated"] for r in table.values()
                  if r["insitu_over_isolated"]]
        if len(ratios) > 1:
            lo, hi = min(ratios), max(ratios)
            print("   in-situ / isolated: %.4f to %.4f, spread %.1f%% of mean"
                  % (lo, hi, 100.0 * (hi - lo) / statistics.mean(ratios)))
        steps = {w: r["eval_step_ms"] for w, r in table.items()
                 if r["eval_step_ms"] is not None}
        if 5 in steps and 6 in steps:
            print("   in-situ premise: 4->5 %+.3f ms, 5->6 %+.3f ms, "
                  "inverted=%s"
                  % (steps[5], steps[6], steps[6] > steps[5]))

    if args.out:
        pathlib.Path(REPO / args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str))
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
