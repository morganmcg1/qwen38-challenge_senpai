#!/usr/bin/env python3
"""E128 rung 0 -- an exact offline reimplementation of the shipped depth walk.

`costModelDepth` (Sources/MLXFastModel/Qwen36MTPBlockSession.swift) chooses the
draft depth of every round from three inputs the session snapshots BEFORE it
proposes anything: the per-position acceptance EMAs, the pending primary's
target top-2 margin, and the width cap in force. `snapshotScheduleSignal`
writes all three into the round's trace line, together with the extension walk
itself (`sched=<depth>:<p>/<reach>/<threshold>;`) and the depth the round
actually drafted (`d=`).

This module replays the walk in Python from the recorded inputs and compares
it with the recorded outputs, so a counterfactual priced offline can be trusted
to describe the shipped scheduler rather than a paraphrase of it.

Three agreements are reported, strongest first:

  1. `sched` agreement -- the replayed `%.6f` walk string is byte-identical to
     the shipped one. This checks p, reach and threshold at every step.
  2. depth agreement -- the replayed depth equals `d=`.
  3. EMA agreement -- the EMA vector this replayer carries forward from round
     k's outcome equals the vector round k+1 recorded. This checks
     `recordAcceptOutcome`, which the counterfactual simulation needs and the
     depth walk alone does not exercise.

  usage: research/e128_replay.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) ")
SIGNAL_RE = re.compile(
    r"\barm=(\w+) m=([-\d.naif]+) streak=(\d+) cap=(\d+) ema=([\d.,]+) "
    r"sched=(.*)$"
)

MAX_DEPTH = 8  # Qwen36MTPLimits.maxDepth
HEAD_STEP_COST_RATIO = 0.18  # Qwen36MTPBlockSession.headStepCostRatio
ACCEPT_EMA_ALPHA = 0.15  # Qwen36MTPBlockSession.acceptEMAAlpha
SEGMENTED_VERIFY_DEPTH_CAP = 7
EMA_PRIOR = [0.85 * 0.98 ** i for i in range(MAX_DEPTH)]

# makeUniformDepthPrice(). `cumulative` repeats the tip's closed form rather
# than accumulating, because the shipped code does: 1.0 + 3.0 * 0.18 and
# 1.0 + 0.18 + 0.18 + 0.18 differ by one ulp.
PRICE_MARGINAL = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
PRICE_CUMULATIVE = [1.0 + i * HEAD_STEP_COST_RATIO for i in range(MAX_DEPTH + 1)]


def cost_model_depth(
    ema: list[float],
    margin: float,
    offered_depth: int = MAX_DEPTH,
    width_cap: int = SEGMENTED_VERIFY_DEPTH_CAP,
    margin_scale_0: float | None = 2.0,
    margin_scale_1: float | None = 3.0,
    marginal: list[float] | None = None,
    cumulative: list[float] | None = None,
    margin_mode: str = "min",
) -> tuple[int, str, list[float]]:
    """The shipped walk. Returns (depth, trace string, per-step p).

    `margin_scale_0` / `margin_scale_1` of None delete that override, which is
    how the `nomargin*` arms are expressed. A NaN margin also disables both,
    matching `pendingTop2` being nil or shorter than two entries.

    `margin_mode` selects how the margin confidence combines with the EMA.
    `"min"` is the shipped strictly-downward override. `"replace"` trusts the
    margin outright and `"max"` lets it raise the EMA, so the two of them
    together measure whether the downward-only restriction is what holds depth
    below the ranked optimum. Neither is shipped behaviour.
    """
    marginal = PRICE_MARGINAL if marginal is None else marginal
    cumulative = PRICE_CUMULATIVE if cumulative is None else cumulative
    cap = min(min(offered_depth, MAX_DEPTH), width_cap)
    if cap <= 0:
        return 0, "", []
    reach = 1.0
    expected = 0.0
    depth = 0
    steps = []
    walked_p = []
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = ema[depth]
        scale = None
        if depth == 0 and have_margin and margin_scale_0 is not None:
            scale = margin_scale_0
        elif depth == 1 and have_margin and margin_scale_1 is not None:
            scale = margin_scale_1
        if scale is not None:
            conf = 1.0 / (1.0 + math.exp(-margin / scale))
            p = {"min": min(p, conf), "replace": conf,
                 "max": max(p, conf)}[margin_mode]
        reach *= p
        threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
        steps.append("%d:%.6f/%.6f/%.6f;" % (depth, p, reach, threshold))
        walked_p.append(p)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth, "".join(steps), walked_p


def record_accept_outcome(
    ema: list[float], accepted_count: int, draft_count: int,
    stopped_early: bool = False,
) -> list[float]:
    """`recordAcceptOutcome`, including the capped optimism transfer.

    `stopped_early` is the shipped guard for a stop token accepted as the last
    draft. The trace does not carry draft token ids, so the replayer assumes
    False and reports any resulting EMA divergence instead of hiding it.
    """
    out = list(ema)
    alpha = ACCEPT_EMA_ALPHA
    for index in range(min(accepted_count, len(out))):
        out[index] += alpha * (1.0 - out[index])
    if accepted_count < draft_count and not stopped_early \
            and accepted_count < len(out):
        out[accepted_count] += alpha * (0.0 - out[accepted_count])
    elif accepted_count == draft_count and draft_count > 0 \
            and accepted_count < len(out):
        if out[accepted_count] < 0.95:
            out[accepted_count] += alpha * (0.95 - out[accepted_count])
    return out


def read_meta(run_dir: Path) -> dict:
    meta = {}
    path = run_dir / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if _:
                meta[key] = value
    return meta


def read_rounds(run_dir: Path) -> list[dict]:
    rounds = []
    for line in (run_dir / "trace.txt").read_text(errors="replace").splitlines():
        head = ROUND_RE.match(line)
        if not head:
            continue
        signal = SIGNAL_RE.search(line)
        if not signal:
            continue
        rounds.append({
            "round": int(head.group(1)),
            "depth": int(head.group(2)),
            "accepted": int(head.group(3)),
            "arm": signal.group(1),
            "margin": float(signal.group(2)),
            "streak": int(signal.group(3)),
            "cap": int(signal.group(4)),
            "ema": [float(v) for v in signal.group(5).split(",")],
            "sched": signal.group(6),
        })
    return rounds


def format_ema(ema: list[float]) -> str:
    return ",".join("%.6f" % value for value in ema)


def validate_leg(run_dir: Path, offered_depth: int) -> dict:
    """Two independent replays of the same leg.

    STATIC replay feeds each round the EMA vector the trace recorded. The trace
    prints `%.6f`, so a p, reach or threshold that lands near a printing
    boundary can differ in its last digit for that reason alone.

    FORWARD replay carries the replayer's own full-precision EMA state from the
    shipped prior through `recordAcceptOutcome`, and never reads a recorded EMA
    again. It is the strict test: it reproduces the whole per-round state
    machine from the first round, and its EMA string must equal the shipped one
    at every round.
    """
    meta = read_meta(run_dir)
    rounds = read_rounds(run_dir)
    forced = meta.get("forced_depth", "none")
    total_tokens = int(meta.get("tokens", "512"))
    static = {"sched": 0, "depth": 0}
    forward = {"sched": 0, "depth": 0, "ema": 0}
    ema_max_abs = 0.0
    divergences = []
    state = list(EMA_PRIOR)
    emitted = 0
    for index, record in enumerate(rounds):
        # `QwenRuntimeMTPDriver.swift:141-150`. The worker is never told how
        # much of the window remains, so the parent narrows its own offer at
        # the tail. Without this the last round of every leg replays too deep.
        offer = max(1, min(offered_depth, MAX_DEPTH, total_tokens - emitted - 1))
        record["offer"] = offer
        emitted += min(1 + record["accepted"], total_tokens - emitted)
        depth, sched, _ = cost_model_depth(
            record["ema"], record["margin"],
            offered_depth=offer, width_cap=record["cap"])
        static["sched"] += sched == record["sched"]
        static["depth"] += depth == record["depth"]

        if format_ema(state) == format_ema(record["ema"]):
            forward["ema"] += 1
        elif len(divergences) < 8:
            divergences.append({
                "round": record["round"], "kind": "ema",
                "shipped": format_ema(record["ema"]),
                "replayed": format_ema(state)})
        ema_max_abs = max(
            ema_max_abs,
            max(abs(a - b) for a, b in zip(state, record["ema"])))

        f_depth, f_sched, _ = cost_model_depth(
            state, record["margin"],
            offered_depth=offer, width_cap=record["cap"])
        forward["sched"] += f_sched == record["sched"]
        if f_depth == record["depth"]:
            forward["depth"] += 1
        elif len(divergences) < 8:
            divergences.append({
                "round": record["round"], "kind": "depth",
                "margin": record["margin"],
                "shipped": record["depth"], "replayed": f_depth,
                "shipped_sched": record["sched"], "replayed_sched": f_sched})
        state = record_accept_outcome(
            state, record["accepted"], record["depth"])
    n = len(rounds)
    scale = (1.0 / n) if n else 0.0
    return {
        "run_dir": str(run_dir),
        "prompt_id": run_dir.name,
        "forced_depth": forced,
        "rounds": n,
        "offered_depth": offered_depth,
        "static_sched_agreement": static["sched"] * scale,
        "static_depth_agreement": (
            static["depth"] * scale if forced == "none" else None),
        "sched_agreement": forward["sched"] * scale,
        # A forced-depth leg pins the depth outside the cost model, so its
        # recorded `d=` is not a cost-model output and cannot be scored here.
        "depth_agreement": (
            forward["depth"] * scale if forced == "none" else None),
        "ema_agreement": forward["ema"] * scale,
        "ema_max_abs_error": ema_max_abs,
        "all_tokens_matched": meta.get("all_tokens_matched"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "divergences": divergences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--offered-depth", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    legs = [validate_leg(d, args.offered_depth) for d in args.run_dirs]
    total = sum(leg["rounds"] for leg in legs)
    sched_hits = sum(leg["sched_agreement"] * leg["rounds"] for leg in legs)
    depth_legs = [leg for leg in legs if leg["depth_agreement"] is not None]
    depth_rounds = sum(leg["rounds"] for leg in depth_legs)
    depth_hits = sum(
        leg["depth_agreement"] * leg["rounds"] for leg in depth_legs)
    ema_rounds = sum(leg["rounds"] for leg in legs)
    ema_hits = sum(
        leg["ema_agreement"] * leg["rounds"] for leg in legs)

    print("%-22s %7s %7s %9s %9s %9s %9s" % (
        "leg", "rounds", "forced", "sched", "depth", "ema", "d(static)"))
    for leg in legs:
        print("%-22s %7d %7s %9.4f %9s %9.4f %9s" % (
            leg["prompt_id"], leg["rounds"], leg["forced_depth"],
            leg["sched_agreement"],
            "n/a" if leg["depth_agreement"] is None
            else "%.4f" % leg["depth_agreement"],
            leg["ema_agreement"],
            "n/a" if leg["static_depth_agreement"] is None
            else "%.4f" % leg["static_depth_agreement"]))
    summary = {
        "legs": len(legs),
        "rounds": total,
        "sched_agreement": sched_hits / total if total else 0.0,
        "depth_agreement": depth_hits / depth_rounds if depth_rounds else None,
        "depth_rounds": depth_rounds,
        "ema_agreement": ema_hits / ema_rounds if ema_rounds else 0.0,
        "ema_rounds": ema_rounds,
        "ema_max_abs_error": max(
            (leg["ema_max_abs_error"] for leg in legs), default=0.0),
    }
    print("\npooled: rounds=%d sched=%.6f depth=%s ema=%.6f (max |dema|=%.3e)"
          % (total, summary["sched_agreement"],
             "n/a" if summary["depth_agreement"] is None
             else "%.6f" % summary["depth_agreement"],
             summary["ema_agreement"], summary["ema_max_abs_error"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"summary": summary, "legs": legs}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
