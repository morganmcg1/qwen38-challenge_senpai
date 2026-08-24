#!/usr/bin/env python3
"""Reduce one E202 in-process arm-switched session (research-only).

The session rotates three arms at the round boundary inside every leg:

    arm(round) = [SHIPPED, BARRIER-LAST, BARRIER-ALL][(round + offset) % 3]

so ANY three consecutive rounds cover all three arms exactly once. The primary
estimator is therefore a within-leg triple contrast: inside one triple the leg,
the thermal state, the process, the head and the local key length are shared,
and a smooth drift in round cost cancels to first order. The design is balanced
by construction because `eval()` changes no computed value, so the per-round
width sequence is identical in every leg.

Two contrasts are reported:

    sync   = mean(BARRIER-LAST) - mean(SHIPPED)     per-eval() sync-cost proxy
    inner  = mean(BARRIER-ALL)  - mean(BARRIER-LAST)
    net    = inner - 4 * sync                       interior de-overlap

`net` is the assignment's decision contrast. Note the direction of its bias:
`sync` is measured at the group boundary, where the barrier also de-overlaps the
group from the following ops, so 4*sync OVER-subtracts if boundary and interior
barriers cost the same for reasons other than exposed dispatch latency. `net` is
therefore conservative against the "overlap confirmed" reading.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

ARM_NAMES = {0: "SHIPPED", 1: "BARRIER-LAST", 2: "BARRIER-ALL"}
EXPECTED_BARRIERS = {0: 0, 1: 1, 2: 5}


def parse_trace(path: pathlib.Path) -> list[dict]:
    rounds = []
    if not path.exists():
        return rounds
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-trace: "):
            continue
        rec = {}
        for field in line[len("mtp-trace: "):].split():
            if "=" not in field:
                continue
            key, _, value = field.partition("=")
            rec[key] = value
        if "round" in rec and "round_us" in rec:
            rounds.append(rec)
    return rounds


def to_int(rec: dict, key: str, default: int = -1) -> int:
    try:
        return int(rec[key])
    except (KeyError, ValueError):
        return default


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def serial_null_rounds(rounds: list[dict]) -> list[dict]:
    """The timed serial control leg: qL == 1 in every round.

    The split-cell branch never serves width 1, so the arm rotation is causally
    unreachable there. These rounds are the in-session null control RULE 388
    asks for, and they come from the same process, thermal state and schedule as
    the MTP rounds.
    """
    serial = [r for r in rounds if to_int(r, "serial_body", 0) == 1]
    if not serial:
        return []
    groups: dict[int, list[dict]] = {}
    for rec in serial:
        groups.setdefault(to_int(rec, "e202_pid"), []).append(rec)
    pid = max(groups, key=lambda p: len(groups[p]))
    return groups[pid]


def select_timed_mtp_rounds(rounds: list[dict], score: dict) -> tuple[list[dict], dict]:
    """Pick the worker process that produced the SCORED MTP decode leg.

    A local-submit leg spawns several model-holding workers (reference row
    generation, the timed serial leg, the timed MTP leg) that all append to one
    trace file. The scored leg is identified by its total decode time, which the
    trusted parent reports independently as `mtp_seconds_per_token`: no name,
    order or pid convention is assumed.
    """
    metrics = score.get("metrics", {})
    tokens = metrics.get("decode_tokens")
    spt = metrics.get("mtp_seconds_per_token")
    groups: dict[int, list[dict]] = {}
    for rec in rounds:
        if to_int(rec, "serial_body", 0) == 1:
            continue
        groups.setdefault(to_int(rec, "e202_pid"), []).append(rec)
    if not groups:
        return [], {"reason": "no non-serial rounds in trace"}
    if tokens is None or spt is None:
        pid = max(groups, key=lambda p: len(groups[p]))
        return groups[pid], {"pid": pid, "match": "fallback-longest"}

    target_us = tokens * spt * 1e6
    best_pid, best_ratio = None, None
    for pid, recs in groups.items():
        total = sum(to_int(r, "round_us", 0) for r in recs)
        ratio = total / target_us if target_us else math.inf
        if best_ratio is None or abs(math.log(max(ratio, 1e-9))) < abs(
            math.log(max(best_ratio, 1e-9))
        ):
            best_pid, best_ratio = pid, ratio
    return groups[best_pid], {
        "pid": best_pid,
        "traced_over_parent_time_ratio": best_ratio,
        "candidate_pids": {str(p): len(r) for p, r in groups.items()},
    }


def validate_witness(recs: list[dict], offset: str) -> dict:
    """RULE 391(b): a leg must prove which arm every round executed."""
    problems = []
    armed_rounds = 0
    serving_rounds = 0
    for rec in recs:
        arm = to_int(rec, "e202_arm")
        sel = to_int(rec, "e202_sel")
        calls = to_int(rec, "e202_calls")
        barriers = to_int(rec, "e202_barriers")
        rnd = to_int(rec, "round")
        if arm < 0 or sel < 0 or calls < 0 or barriers < 0:
            problems.append(f"round {rnd}: witness field missing")
            continue
        if offset == "-":
            if sel != 0 or arm != 0:
                problems.append(f"round {rnd}: bare leg selected arm {arm}")
        else:
            if sel != 1:
                problems.append(f"round {rnd}: arm selector inactive (sanitizer drop)")
            if arm != (rnd + int(offset)) % 3:
                problems.append(f"round {rnd}: arm {arm} off schedule")
        if calls > 0:
            serving_rounds += 1
            if barriers != calls * EXPECTED_BARRIERS[max(arm, 0)]:
                problems.append(
                    f"round {rnd}: {barriers} barriers for {calls} calls on arm {arm}"
                )
            if arm != 0 and barriers == 0:
                problems.append(f"round {rnd}: VOID, barrier arm served no barrier")
        if to_int(rec, "e202_barriers", 0) > 0:
            armed_rounds += 1
    return {
        "valid": not problems,
        "problems": problems[:20],
        "problem_count": len(problems),
        "rounds": len(recs),
        "branch_serving_rounds": serving_rounds,
        "armed_rounds": armed_rounds,
    }


def census(recs: list[dict]) -> dict:
    """Served-width histogram of the qL 6...9 split-cell branch, per round."""
    hist: dict[str, int] = {}
    for rec in recs:
        calls = to_int(rec, "e202_calls", 0)
        width = to_int(rec, "e202_width", 0)
        if calls > 0 and width > 0:
            hist[str(width)] = hist.get(str(width), 0) + calls
    return dict(sorted(hist.items()))


def census_first_n_calls(recs: list[dict], limit: int = 128) -> dict:
    """Served-width histogram over the first `limit` TIMED split-cell calls.

    The warm-up pass (a uniform 32-per-width prefix, counted by the separate
    round-1 census witness) is already outside `recs`, so this window is the
    first 128 calls of steady decode. It shows that even a post-warm-up window
    of E198's size is unrepresentative of the leg census, which is the second
    half of the FINDING 522(b) evidence.
    """
    hist: dict[str, int] = {}
    seen = 0
    for rec in recs:
        calls = to_int(rec, "e202_calls", 0)
        width = to_int(rec, "e202_width", 0)
        if calls <= 0 or width <= 0:
            continue
        if seen >= limit:
            break
        take = min(calls, limit - seen)
        hist[str(width)] = hist.get(str(width), 0) + take
        seen += take
    return dict(sorted(hist.items()))


def round_histogram(recs: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for rec in recs:
        if to_int(rec, "e202_calls", 0) > 0:
            width = str(to_int(rec, "e202_width", 0))
            hist[width] = hist.get(width, 0) + 1
    hist["no-split-branch"] = sum(1 for r in recs if to_int(r, "e202_calls", 0) == 0)
    return dict(sorted(hist.items()))


def triples(recs: list[dict], require_serving: bool) -> list[dict]:
    """Within-leg triple contrasts over three consecutive rounds.

    A triple is used only when all three rounds carry the same served width and
    the same split-cell call count, so the only difference inside it is the arm.
    Triples do not overlap: a sliding window would reuse each round three times
    and understate the dispersion of the contrast.
    """
    by_round = {to_int(r, "round"): r for r in recs}
    out = []
    consumed: set[int] = set()
    for start in sorted(by_round):
        if start in consumed:
            continue
        group = [by_round.get(start + i) for i in range(3)]
        if any(g is None for g in group):
            continue
        arms = [to_int(g, "e202_arm") for g in group]
        if sorted(arms) != [0, 1, 2]:
            continue
        widths = {to_int(g, "e202_width") for g in group}
        calls = {to_int(g, "e202_calls") for g in group}
        if len(widths) != 1 or len(calls) != 1:
            continue
        serving = to_int(group[0], "e202_calls") > 0
        if require_serving != serving:
            continue
        times = {}
        for arm, g in zip(arms, group):
            times[arm] = to_int(g, "round_us")
        out.append(
            {
                "start_round": start,
                "width": widths.pop(),
                "calls": calls.pop(),
                "t": times,
            }
        )
        consumed.update({start, start + 1, start + 2})
    return out


def mean_ci(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "sigma": None, "two_sigma": None}
    mean = statistics.fmean(values)
    sigma = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "sem": sigma,
        "two_sigma": 2 * sigma if sigma is not None else None,
    }


def contrasts(tris: list[dict]) -> dict:
    sync = [(t["t"][1] - t["t"][0]) / 1000.0 for t in tris]
    inner = [(t["t"][2] - t["t"][1]) / 1000.0 for t in tris]
    net = [i - 4 * s for i, s in zip(inner, sync)]
    total = [(t["t"][2] - t["t"][0]) / 1000.0 for t in tris]
    calls = [t["calls"] for t in tris]
    result = {
        "sync_ms_per_round": mean_ci(sync),
        "inner_ms_per_round": mean_ci(inner),
        "net_interior_ms_per_round": mean_ci(net),
        "all_minus_shipped_ms_per_round": mean_ci(total),
        "mean_split_calls_per_round": statistics.fmean(calls) if calls else None,
    }
    if calls and statistics.fmean(calls) > 0:
        per_call = statistics.fmean(calls)
        result["sync_us_per_barrier"] = (
            result["sync_ms_per_round"]["mean"] * 1000.0 / per_call
        )
        result["inner_us_per_barrier"] = (
            result["inner_ms_per_round"]["mean"] * 1000.0 / (4 * per_call)
        )
    return result


def verdict(inner: dict) -> str:
    """Decide on the DIRECT interior contrast, not on the subtracted form.

    `inner` = BARRIER-ALL - BARRIER-LAST already contains the marginal sync cost
    of four extra `eval()` calls, and that cost cannot be negative, so `inner` is
    an UPPER BOUND on the interior exposed dispatch latency. A bound needs no
    sync-cost estimate and so carries none of its variance. The assignment's
    subtracted form `net = inner - 4 * sync` is reported alongside; it assumes
    the per-barrier cost is additive, which the measurement itself can test.

    Labels follow the advisor ruling on PR 199 (comment 5402432090). `inner`
    bounds the exposed interior latency FROM ABOVE, so only the small-`inner`
    branch is decisive; a large `inner` is equally consistent with pure marginal
    sync cost of the four extra barriers and cannot be separated from overlap at
    this instrument's resolution. E198 priced the interior dispatch latency the
    FINDING 507 family would have to recover at 1.6 ms/round.
    """
    e198_priced_ms_per_round = 1.6
    mean = inner.get("mean")
    two_sigma = inner.get("two_sigma")
    if mean is None:
        return "VOID"
    upper = mean + (two_sigma or 0.0)
    if upper < 0.2:
        return "non-transfer-confirmed"
    if mean >= e198_priced_ms_per_round:
        return (
            "interior-latency-not-cheaply-recoverable; consistent with overlap; "
            "overlap vs marginal-sync cost not separable at this resolution"
        )
    if upper < e198_priced_ms_per_round:
        return "non-transfer-confirmed (below the E198 priced 1.6 ms/round)"
    return "ambiguous-band-below-1.6-with-interval-open"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    legs = sorted(p for p in args.session.glob("leg*") if p.is_dir())
    if not legs:
        print(f"e202_reduce: no legs under {args.session}", file=sys.stderr)
        return 2

    leg_reports = []
    all_serving: list[dict] = []
    all_idle: list[dict] = []
    all_serial: list[dict] = []
    valid = True
    for leg in legs:
        meta = read_meta(leg / "meta.txt")
        score_path = leg / "score.json"
        score = json.loads(score_path.read_text()) if score_path.exists() else {}
        rounds = parse_trace(leg / "trace.txt")
        timed, selection = select_timed_mtp_rounds(rounds, score)
        offset = meta.get("e202_offset", "-")
        witness = validate_witness(timed, offset)
        valid = valid and witness["valid"]
        serving = triples(timed, require_serving=True)
        idle = triples(timed, require_serving=False)
        serial = triples(serial_null_rounds(rounds), require_serving=False)
        all_serving.extend(serving)
        all_idle.extend(idle)
        all_serial.extend(serial)
        metrics = score.get("metrics", {})
        leg_reports.append(
            {
                "leg": leg.name,
                "offset": offset,
                "gpu_temp_entry": meta.get("gpu_temp_entry"),
                "gpu_temp_exit": meta.get("gpu_temp_exit"),
                "worker_sha256": meta.get("worker_sha256_before"),
                "worker_digest_stable": meta.get("worker_digest_stable"),
                "head_provenance_sha256": meta.get("head_provenance_sha256"),
                "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
                "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
                "all_tokens_matched": metrics.get("all_tokens_matched"),
                "residual_divergence_count": metrics.get("residual_divergence_count"),
                "decode_tokens": metrics.get("decode_tokens"),
                "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
                "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
                "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
                "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
                "accepted_draft_rate": metrics.get("accepted_draft_rate"),
                "traced_worker": selection,
                "witness": witness,
                "split_call_census_by_width": census(timed),
                "split_call_census_first_128_calls": census_first_n_calls(timed, 128),
                "round_census_by_width": round_histogram(timed),
                "serving_triples": len(serving),
                "leg_mean_round_ms": (
                    statistics.fmean([to_int(r, "round_us") / 1000.0 for r in timed])
                    if timed
                    else None
                ),
            }
        )

    serving_contrast = contrasts(all_serving)
    idle_contrast = contrasts(all_idle)
    serial_contrast = contrasts(all_serial)
    by_width = {
        str(width): contrasts([t for t in all_serving if t["width"] == width])
        for width in sorted({t["width"] for t in all_serving})
    }

    total_census: dict[str, int] = {}
    warmup_census: dict[str, int] = {}
    for rep in leg_reports:
        for width, count in rep["split_call_census_by_width"].items():
            total_census[width] = total_census.get(width, 0) + count
        for width, count in rep["split_call_census_first_128_calls"].items():
            warmup_census[width] = warmup_census.get(width, 0) + count

    leg_means = [r["leg_mean_round_ms"] for r in leg_reports if r["leg_mean_round_ms"]]
    report = {
        "experiment": "e202-eval-barrier-dispatch-settlement",
        "harness": "local",
        "session": str(args.session),
        "witnesses_valid": valid,
        "legs": leg_reports,
        "contrasts_branch_serving_rounds": serving_contrast,
        "contrasts_branch_serving_rounds_by_width": by_width,
        "contrasts_null_control_non_serving_rounds": idle_contrast,
        "contrasts_null_control_serial_leg": serial_contrast,
        "session_split_call_census_by_width": dict(sorted(total_census.items())),
        "session_split_call_census_first_128_calls_per_leg": dict(
            sorted(warmup_census.items())
        ),
        "leg_level_null_control": {
            "note": "every leg carries the same arm composition, so this spread "
            "is drift only (FINDING 518 re-measurement)",
            "leg_mean_round_ms": leg_means,
            "spread_ms": max(leg_means) - min(leg_means) if leg_means else None,
            "stdev_ms": statistics.stdev(leg_means) if len(leg_means) > 1 else None,
        },
        "verdict": verdict(serving_contrast["inner_ms_per_round"]) if valid else "VOID",
        "positive_control_sync_detected": (
            (serving_contrast["sync_ms_per_round"]["mean"] or 0.0)
            > 2 * (serving_contrast["sync_ms_per_round"]["two_sigma"] or 0.0)
        ),
    }

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
