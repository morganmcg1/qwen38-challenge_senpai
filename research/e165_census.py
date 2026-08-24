#!/usr/bin/env python3
"""E165 Stage 0: decompose one MTP round and find the GPU-idle window.

usage: research/e165_census.py TAG [TAG ...] [--json PATH]

Every number comes from the `mtp-anchor:` lines, which carry absolute mach
uptime nanoseconds at fourteen boundaries of the round. The round period is
therefore reconstructed ACROSS the round boundary, which is where the window
this experiment targets actually lives: the trusted parent owns the protocol
turnaround between `t_tail_done` of round N and `t_round0` of round N+1, and
the GPU is idle for all of it.

THE GPU-IDLE WINDOW.

The round's single blocking `eval` returns at `t_eval_done` with every
submitted command buffer complete, so the device owns no work from that
instant. The next submission is the head chain's first `asyncEval`, issued
immediately after `t_head1_built`. The window between them is

    idle_raw = t_head1_built(N+1) - t_eval_done(N)

and it contains, in order: the host readout, the accept walk and commit, the
head-history upkeep, the trace instrument, the protocol turnaround, and the
next round's pre-draft and head-1 graph build.

INSTRUMENT COST IS SUBTRACTED, NOT ASSUMED SMALL. The row dump and the two
trace writes sit inside that window and are worth tens of microseconds against
a target of hundreds, so both are bracketed by their own anchors and removed:

    idle = idle_raw - row_trace(N) - trace_emit(N)

`row_trace` is bracketed in-round; `trace_emit` is measured after the write and
reported in the NEXT round's line as `t_prev_trace_done`, which is the only
ordering that can carry it. A trace without those fields reports idle_raw and
says so.

The reject path submits a recurrent-state prefetch inside `commit`, so its
idle window is shorter than the anchors show. Accept and reject rounds are
reported separately and the fit uses full-accept rounds only.

IDLE OR SLOW. The round period splits into exactly two measured parts:

    round_period = gpu_idle_window + gpu_covered

`gpu_idle_window` is PROVABLY idle — the blocking eval is a full barrier, so
at `t_eval_done` the device owns nothing and it owns nothing again until the
next `asyncEval`. `gpu_covered` is the span in which at least one command
buffer is enqueued; it is an UPPER bound on GPU-busy, because bubbles between
enqueued kernels are invisible to a host clock. So

    idle_fraction >= gpu_idle_window / round_period          (measured)
    busy_fraction <= gpu_covered    / round_period           (measured)

A schedule fix can only ever recover the first term. If it is small, the round
is slow, not idle, and the host-reordering family of mechanisms is capped at
that value whatever else is true.

COUNTER HAZARD, campaign record. `verify_window` (`verify_build_us`) is NOT
host graph construction. E86 removed the ladder and measured the split: host
encode of the whole 64-layer verify graph is 2,294 us, so under the shipped
ladder that counter is about 97 % GPU wait. Only `verify_pipeline`
(`verify_window + gpu_verify_eval`) is meaningful, and a ladder-off leg is the
only way to recover host encode from it.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"


def read_meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    if not path.exists():
        return {}
    return dict(
        line.partition("=")[::2]
        for line in path.read_text().splitlines()
        if "=" in line
    )


def read_score(tag: str) -> dict:
    """Leg score, with the nested `metrics` object lifted to the top level."""
    path = OUT / tag / "score.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    flat = {k: v for k, v in raw.items() if k != "metrics"}
    flat.update(raw.get("metrics", {}))
    return flat


def anchors(tag: str) -> list[dict]:
    """Every `mtp-anchor` round of one leg, in file order, per pid."""
    path = OUT / tag / "trace.txt"
    if not path.exists():
        return []
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-anchor: "):
            continue
        fields = {}
        ok = True
        for token in line[len("mtp-anchor: "):].split():
            key, sep, value = token.partition("=")
            if not sep:
                ok = False
                break
            try:
                fields[key] = int(value)
            except ValueError:
                ok = False
                break
        if ok and "t_round0" in fields:
            rounds.append(fields)
    return rounds


def components(rounds: list[dict]) -> list[dict]:
    """Per-round component microseconds, including the cross-boundary window.

    The last round of a leg has no successor and therefore no idle window, so
    it contributes every in-round component and no window.
    """
    out = []
    for index, cur in enumerate(rounds):
        nxt = rounds[index + 1] if index + 1 < len(rounds) else None
        if nxt is not None and nxt["pid"] != cur["pid"]:
            nxt = None
        if nxt is not None and nxt["t_round0"] < cur["t_tail_done"]:
            nxt = None

        us = lambda a, b: (b - a) / 1000.0  # noqa: E731

        row_trace = 0.0
        if cur.get("t_row_trace_done", 0) and cur.get("t_row_trace0", 0):
            row_trace = us(cur["t_row_trace0"], cur["t_row_trace_done"])

        record = {
            "round": cur["round"],
            "d": cur["d"],
            "acc": cur["acc"],
            "full_accept": cur["acc"] == cur["d"],
            # Host graph build for the head chain, both submits included.
            "host_draft_build": us(cur["t_round0"], cur["t_draft_built"]),
            "host_draft_pre": us(cur["t_round0"], cur["t_flush_built"]),
            "host_head1_build": us(cur["t_flush_built"], cur["t_head1_built"]),
            "submit1": us(cur["t_head1_built"], cur["t_submit1"]),
            "host_chain_build": us(cur["t_submit1"], cur["t_chain_built"]),
            "submit2": us(cur["t_chain_built"], cur["t_draft_built"]),
            "snapshot": us(cur["t_draft_built"], cur["t_snapshot_done"]),
            # ~97 % GPU wait under the shipped ladder (E86); never read as
            # host op-count evidence.
            "verify_window": us(cur["t_snapshot_done"], cur["t_verify_built"]),
            "gpu_verify_eval": us(cur["t_verify_built"], cur["t_eval_done"]),
            "readout": us(cur["t_eval_done"], cur["t_read_done"]),
            "commit": us(cur["t_read_done"], cur["t_commit_done"]),
            "upkeep": us(cur["t_commit_done"], cur["t_tail_done"]),
            "row_trace": row_trace,
            "round_total": us(cur["t_round0"], cur["t_tail_done"]),
        }
        record["upkeep_net"] = record["upkeep"] - row_trace
        record["host_post_eval_tail"] = (
            record["readout"] + record["commit"] + record["upkeep_net"]
        )
        # The only counter that may be quoted as a verify cost (E86 hazard).
        record["verify_pipeline"] = (
            record["verify_window"] + record["gpu_verify_eval"]
        )
        # First enqueue of the round to the barrier that drains it.
        record["gpu_covered"] = us(cur["t_head1_built"], cur["t_eval_done"])
        if nxt is not None:
            trace_emit = 0.0
            if nxt.get("t_prev_trace_done", 0):
                trace_emit = us(cur["t_tail_done"], nxt["t_prev_trace_done"])
            gap = us(cur["t_tail_done"], nxt["t_round0"]) - trace_emit
            idle_raw = us(cur["t_eval_done"], nxt["t_head1_built"])
            record["trace_emit"] = trace_emit
            record["protocol_gap"] = gap
            record["next_pre_submit"] = (
                us(nxt["t_round0"], nxt["t_head1_built"])
            )
            record["gpu_idle_window_raw"] = idle_raw
            record["gpu_idle_window"] = idle_raw - row_trace - trace_emit
            record["round_period"] = (
                us(cur["t_round0"], nxt["t_round0"]) - trace_emit - row_trace
            )
        out.append(record)
    return out


FIT_KEYS = [
    "gpu_idle_window",
    "host_post_eval_tail",
    "readout",
    "commit",
    "upkeep",
    "upkeep_net",
    "host_draft_pre",
    "host_head1_build",
    "protocol_gap",
    "next_pre_submit",
    "host_draft_build",
    "host_chain_build",
    "submit1",
    "submit2",
    "snapshot",
    "verify_window",
    "gpu_verify_eval",
    "verify_pipeline",
    "gpu_covered",
    "round_period",
    "round_total",
]

# Terms that partition `round_period` with no double counting, in wall order.
# `row_trace` and `trace_emit` are removed from both sides, so a residual here
# is genuinely unattributed rather than instrument cost.
#
# `gpu_covered` runs from `submit1` through `gpu_verify_eval`; every other term
# is inside the provably idle window.
BUDGET_COVERED = [
    "submit1",
    "host_chain_build",
    "submit2",
    "snapshot",
    "verify_window",
    "gpu_verify_eval",
]
BUDGET_IDLE = [
    "readout",
    "commit",
    "upkeep_net",
    "protocol_gap",
    "host_draft_pre",
    "host_head1_build",
]
BUDGET_KEYS = BUDGET_COVERED + BUDGET_IDLE


def median_of(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if key in r]
    return st.median(values) if values else None


def budget_of(records: list[dict]) -> dict | None:
    """Close the round period over the measured terms and name the residual.

    Only rounds that carry a successor have a period, so the last round of a
    leg drops out. Medians are taken per term rather than per round: the
    residual then reports how well a MEDIAN round closes, which is the
    quantity every share in the report is quoted against.
    """
    have = [r for r in records if "round_period" in r]
    if not have:
        return None
    period = st.median(r["round_period"] for r in have)
    terms = {k: median_of(have, k) for k in BUDGET_KEYS}
    covered = sum(terms[k] or 0.0 for k in BUDGET_COVERED)
    idle = sum(terms[k] or 0.0 for k in BUDGET_IDLE)
    return {
        "rounds": len(have),
        "round_period": period,
        "terms": terms,
        "covered_us": covered,
        "idle_us": idle,
        "residual_us": period - covered - idle,
        # Measured bounds, not estimates: see the module docstring.
        "idle_fraction": idle / period,
        "busy_fraction": covered / period,
        "measured_idle_window_us": median_of(have, "gpu_idle_window"),
    }


def contrasts(legs: list[dict]) -> dict:
    """The three between-leg questions the census exists to answer."""
    out: dict[str, dict] = {}

    def pick(**want) -> list[dict]:
        return [
            leg for leg in legs
            if all(leg.get(k) == v for k, v in want.items())
        ]

    # 1. Does the instrument change the thing it measures? Same pin, same
    #    build, same fixture: the only difference is the trace, so the two
    #    seconds-per-token values bound its cost directly.
    for untraced in pick(trace="0"):
        peers = [
            leg for leg in legs
            if leg["trace"] != "0" and leg["ladder"] == untraced["ladder"]
            and leg["sync_head"] != "1"
            and leg["pinned_depth"] == untraced["pinned_depth"]
        ]
        if not peers or untraced["mtp_seconds_per_token"] is None:
            continue
        traced = st.mean(
            leg["mtp_seconds_per_token"] for leg in peers
            if leg["mtp_seconds_per_token"] is not None
        )
        ratio = traced / untraced["mtp_seconds_per_token"]
        out[f"instrument neutrality d={untraced['pinned_depth']}"] = {
            "untraced_tag": untraced["tag"],
            "traced_tags": ",".join(leg["tag"] for leg in peers),
            "untraced_s_per_tok": untraced["mtp_seconds_per_token"],
            "traced_s_per_tok": traced,
            "traced_over_untraced": round(ratio, 6),
            "within_5_percent": abs(ratio - 1.0) <= 0.05,
        }

    # 2. The ladder hides host encode inside the eval wait. Removing it is the
    #    only way to read the two apart: with no rungs, `verify_window` is the
    #    host encode and `gpu_verify_eval` is the verify GPU wall.
    for off in pick(ladder="off"):
        peers = [
            leg for leg in legs
            if leg["ladder"] == "default" and leg["trace"] != "0"
            and leg["sync_head"] != "1"
            and leg["pinned_depth"] == off["pinned_depth"]
            and leg["budget"]
        ]
        if not peers or not off["budget"]:
            continue
        base_pipe = st.mean(
            leg["median"]["verify_pipeline"] for leg in peers)
        base_period = st.mean(leg["budget"]["round_period"] for leg in peers)
        host_encode = off["median"]["verify_window"]
        gpu_wall = off["median"]["gpu_verify_eval"]
        out[f"ladder exposure d={off['pinned_depth']}"] = {
            "ladder_off_tag": off["tag"],
            "shipped_tags": ",".join(leg["tag"] for leg in peers),
            "host_encode_H_us": round(host_encode, 1),
            "verify_gpu_wall_us": round(gpu_wall, 1),
            "shipped_verify_pipeline_us": round(base_pipe, 1),
            # What the shipped ladder still fails to hide.
            "residual_encode_exposure_us": round(base_pipe - gpu_wall, 1),
            "ladder_off_round_period_us": round(
                off["budget"]["round_period"], 1),
            "shipped_round_period_us": round(base_period, 1),
            "ladder_worth_us_per_round": round(
                off["budget"]["round_period"] - base_period, 1),
        }

    # 3. Draining the chain moves head GPU execute into `submit2`, which is
    #    the only host-visible measurement of the GPU work a cross-round
    #    prefetch could move into the idle window.
    drained = sorted(
        (leg for leg in pick(sync_head="1") if leg["median"].get("submit2")),
        key=lambda leg: leg["pinned_depth"] or "",
    )
    if drained:
        body = {
            leg["tag"] + f" d={leg['pinned_depth']}":
                f"head_chain_gpu_wall={leg['median']['submit2']:.1f} us "
                f"idle_window={leg['median']['gpu_idle_window']:.1f} us"
            for leg in drained
        }
        points = [
            (float(leg["pinned_depth"]), leg["median"]["submit2"])
            for leg in drained if leg["pinned_depth"]
        ]
        if len(points) >= 2:
            intercept, slope = fit(points)
            body["head_chain_gpu_intercept_us"] = round(intercept, 1)
            body["head_chain_gpu_per_draft_us"] = round(slope, 1)
        out["head chain GPU wall, sync-head legs"] = body

    return out


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares intercept and slope of y against d."""
    n = len(points)
    if n < 2:
        return (points[0][1], 0.0) if n else (float("nan"), float("nan"))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 0.0)
    slope = (n * sxy - sx * sy) / denom
    return ((sy - slope * sx) / n, slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    ap.add_argument(
        "--accept-only",
        action="store_true",
        help="fit on full-accept rounds only (the reject path submits a "
             "recurrent prefetch inside the idle window)",
    )
    args = ap.parse_args()

    report = {"legs": [], "fit": {}}
    per_depth: dict[int, dict[str, float]] = {}

    for tag in args.tags:
        meta = read_meta(tag)
        score = read_score(tag)
        recs = components(anchors(tag))
        use = [r for r in recs if r["full_accept"]] if args.accept_only else recs
        depths = sorted({r["d"] for r in recs})
        leg = {
            "tag": tag,
            "rounds": len(recs),
            "rounds_used": len(use),
            "depths": depths,
            "sync_head": meta.get("sync_head"),
            "trace": meta.get("trace"),
            "ladder": meta.get("e165_ladder", "default"),
            "pinned_depth": meta.get("e165_pinned_depth"),
            "cool_gate": meta.get("cool_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "worker_sha256": meta.get("worker_sha256"),
            "base_sha": meta.get("base_sha"),
            "full_accept_rounds": sum(1 for r in recs if r["full_accept"]),
            "mtp_seconds_per_token": score.get("mtp_seconds_per_token"),
            "serial_seconds_per_token": score.get("serial_seconds_per_token"),
            "mtp_decode_speedup": score.get("mtp_decode_speedup"),
            "effective_mean_draft_len": score.get("effective_mean_draft_len"),
            "accepted_draft_rate": score.get("accepted_draft_rate"),
            "all_tokens_matched": score.get("all_tokens_matched"),
            "residual_divergence_count": score.get("residual_divergence_count"),
            "decode_tokens": score.get("decode_tokens"),
            "median": {k: median_of(use, k) for k in FIT_KEYS},
        }
        report["legs"].append(leg)

        print(f"\n=== {tag}  rounds={len(recs)} used={len(use)} "
              f"depths={depths} sync_head={meta.get('sync_head')} "
              f"ladder={meta.get('e165_ladder', 'default')} "
              f"gate={meta.get('cool_gate')} ===")
        print(f"  s/tok={leg['mtp_seconds_per_token']} "
              f"edl={leg['effective_mean_draft_len']} "
              f"acc_rate={leg['accepted_draft_rate']} "
              f"matched={leg['all_tokens_matched']} "
              f"full_accept={leg['full_accept_rounds']}/{len(recs)}")
        for key in FIT_KEYS:
            value = leg["median"][key]
            if value is not None:
                print(f"  {key:24s} {value:12.1f} us")

        budget = budget_of(use) if use else None
        leg["budget"] = budget
        if budget:
            period = budget["round_period"]
            print(f"  -- closed budget, median round_period "
                  f"{period:.1f} us --")
            for key in BUDGET_KEYS:
                value = budget["terms"].get(key)
                if value is None:
                    continue
                where = "covered" if key in BUDGET_COVERED else "IDLE"
                print(f"     {key:22s} {value:10.1f} us "
                      f"{100.0 * value / period:6.2f} %  {where}")
            print(f"     {'residual':22s} {budget['residual_us']:10.1f} us "
                  f"{100.0 * budget['residual_us'] / period:6.2f} %")
            print(f"     idle_fraction >= {100.0 * budget['idle_fraction']:.2f}"
                  f" %   busy_fraction <= "
                  f"{100.0 * budget['busy_fraction']:.2f} %")

        # Only the plain traced legs carry the depth fit. Sync-head drains the
        # chain, ladder-off moves GPU work across the encode boundary, and an
        # untraced leg has no anchors at all.
        if (len(depths) == 1 and meta.get("sync_head") != "1"
                and meta.get("trace") != "0"
                and meta.get("e165_ladder", "default") == "default"):
            per_depth.setdefault(depths[0], []).append(leg["median"])

    # Replicated depths average before the fit, so a bracketing replicate
    # damps thermal drift instead of tilting the slope.
    depth_median = {
        d: {k: st.mean([m[k] for m in legs if m.get(k) is not None])
            for k in FIT_KEYS
            if any(m.get(k) is not None for m in legs)}
        for d, legs in per_depth.items() if legs
    }
    fit_depths = sorted(depth_median)
    if len(fit_depths) >= 2:
        print(f"\n=== component fit against pinned depth d in {fit_depths} ===")
        print(f"  {'component':24s} {'intercept us':>14s} {'per-draft us':>14s}")
        for key in FIT_KEYS:
            points = [
                (float(d), depth_median[d][key])
                for d in fit_depths
                if depth_median[d].get(key) is not None
            ]
            if len(points) < 2:
                continue
            intercept, slope = fit(points)
            report["fit"][key] = {
                "intercept_us": intercept,
                "per_draft_us": slope,
                "points": points,
            }
            print(f"  {key:24s} {intercept:14.1f} {slope:14.1f}")

    report["contrasts"] = contrasts(report["legs"])
    for name, body in report["contrasts"].items():
        print(f"\n=== {name} ===")
        for key, value in body.items():
            print(f"  {key:34s} {value}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
