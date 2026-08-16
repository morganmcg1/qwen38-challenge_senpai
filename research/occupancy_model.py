#!/usr/bin/env python3
"""fb2 occupancy / blended-ratio model for the segmented streak gate.

Models `costModelDepth` + the fullAcceptStreak ladder as a Markov chain over
the streak counter under a homogeneous per-position acceptance probability q,
and blends the measured round-cost curve C(d) to a predicted `raw` score.

The chain is exactly the shipped rule:
    widthCap = streak >= gate ? segmentedVerifyDepthCap : sdpaWidthWallDepthCap
    depth    = greedy marginal walk (h = headStepCostRatio) truncated at widthCap
    full accept (prob q^depth) -> streak+1 ; any reject -> streak = 0

Simplification: the shipped depth-0 confidence clamp
`p = min(EMA[0], sigmoid(margin/2))` is dropped, because a single q carries no
margin. Observed margins are large (top-1 margin ~17 logits at the one
inspected round => conf ~0.9998), so the clamp is inert in the realised runs.
"""
from __future__ import annotations

import argparse
import json

H = 0.20  # Qwen36MTPBlockSession.headStepCostRatio
SHALLOW_CAP = 4  # sdpaWidthWallDepthCap
DEEP_CAP = 8  # segmentedVerifyDepthCap

# Measured M4 Pro round cost, milliseconds, indexed by draft depth d
# (rows verified = d + 1). Source: research/fb7_head_rebase.py parent-side
# block latency on Run I / Run J.
MEASURED_C_LOCAL_MS = {2: 79.70, 4: 126.40, 5: 146.50, 6: 168.30, 7: 189.70, 8: 217.40}
# Kink-free linear fill for depths never observed: 12.2 ms + 22.5 ms/row.
LINEAR_INTERCEPT_MS = 12.2
LINEAR_PER_ROW_MS = 22.5
# Head rebase: resident BF16 head reads 610,464,691 B/forward more than the
# declared 4-bit g64 manifest => 2.6893 ms/draft at 227 GB/s.
HEAD_DELTA_MS_PER_DRAFT = 2.6893

SERIAL_S_PER_TOKEN = 0.073532175738364458  # Run I pinned serial leg, M4 Pro

# Measured gap between the timed decode phase and the sum of parent-side block
# requests: 4011.9 ms (Run I) and 4013.2 ms (Run J) over a 512-token window,
# i.e. seed prefill plus parent work the block timer never sees. It is
# schedule-invariant to 0.03% across those two arms and no gating change can
# touch it, so the blended ratio has to carry it or it overstates every gate.
NON_BLOCK_OVERHEAD_MS_PER_TOKEN = 4012.5 / 512.0


def round_cost_ms(depth: int, rebase_head: bool) -> float:
    if depth in MEASURED_C_LOCAL_MS:
        c = MEASURED_C_LOCAL_MS[depth]
    else:
        c = LINEAR_INTERCEPT_MS + LINEAR_PER_ROW_MS * (depth + 1)
    if rebase_head:
        c -= HEAD_DELTA_MS_PER_DRAFT * depth
    return c


TRACE_ROUND = __import__("re").compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?cap=(\d+)"
)


def position_acceptance(trace_path: str, only_deep: bool | None = None) -> dict:
    """Realised per-position acceptance from a trace: for a round of depth d
    that accepted a drafts, positions 0..a-1 observed a success and position a
    observed a failure when a < d. Deeper positions were never reached.

    `only_deep` selects rounds by the cap the gate handed them, which is the
    conditioning the gate actually applies: True keeps deep-cap rounds, False
    keeps shallow-cap rounds, None keeps every round."""
    hit = [0] * DEEP_CAP
    seen = [0] * DEEP_CAP
    rounds = 0
    with open(trace_path) as fh:
        for line in fh:
            m = TRACE_ROUND.search(line)
            if not m:
                continue
            d, a, cap = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if only_deep is not None and (cap > SHALLOW_CAP) != only_deep:
                continue
            rounds += 1
            for i in range(min(a, d)):
                hit[i] += 1
                seen[i] += 1
            if a < d:
                seen[a] += 1
    return {
        "rounds": rounds,
        "hits": hit,
        "observations": seen,
        "acceptance": [hit[i] / seen[i] if seen[i] else None for i in range(DEEP_CAP)],
        "pooled_acceptance": sum(hit) / sum(seen) if sum(seen) else None,
    }


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def shifted_profile(profile: list[float], weights: list[int], q: float) -> list[float]:
    """Shift a measured per-position profile in log-odds until its
    observation-weighted mean equals q, preserving the decay shape."""
    import math

    logits = [math.log(p / (1.0 - p)) for p in profile]
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        shifted = [_sigmoid(l + mid) for l in logits]
        wsum = sum(weights)
        mean = sum(shifted[i] * weights[i] for i in range(len(shifted))) / wsum
        if mean > q:
            hi = mid
        else:
            lo = mid
    return [_sigmoid(l + 0.5 * (lo + hi)) for l in logits]


def shifted_pair(
    ps: list[float],
    ws: list[int],
    pd: list[float],
    wd: list[int],
    q: float,
) -> tuple[list[float], list[float]]:
    """Shift the shallow-cap and deep-cap profiles by one shared log-odds
    offset until their pooled observation-weighted mean equals q.

    A shared offset is what makes this a difficulty sweep: it moves the whole
    prompt to a harder or easier regime while preserving the measured gap
    between rounds the gate held shallow and rounds it let run deep. Shifting
    each subpopulation to q separately would erase that gap, which is exactly
    the conditioning the gate exists to exploit."""
    import math

    ls = [math.log(p / (1.0 - p)) for p in ps]
    ld = [math.log(p / (1.0 - p)) for p in pd]
    wsum = sum(ws) + sum(wd)
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        mean = (
            sum(_sigmoid(ls[i] + mid) * ws[i] for i in range(len(ls)))
            + sum(_sigmoid(ld[i] + mid) * wd[i] for i in range(len(ld)))
        ) / wsum
        if mean > q:
            hi = mid
        else:
            lo = mid
    d = 0.5 * (lo + hi)
    return [_sigmoid(l + d) for l in ls], [_sigmoid(l + d) for l in ld]


def greedy_depth(p: list[float], cap: int) -> int:
    """Replay costModelDepth against a per-position acceptance vector."""
    if cap <= 0:
        return 0
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= p[depth]
        threshold = H * (1.0 + expected) / (1.0 + depth * H)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def expected_accepted(p: list[float], depth: int) -> float:
    """E[accepted drafts] = sum over k of P(first k positions all accept)."""
    total, reach = 0.0, 1.0
    for k in range(depth):
        reach *= p[k]
        total += reach
    return total


def full_accept_prob(p: list[float], depth: int) -> float:
    prob = 1.0
    for k in range(depth):
        prob *= p[k]
    return prob


def stationary(gate: int, ps: list[float], pd: list[float]):
    """Streak-chain stationary distribution over states 0..gate. States below
    the gate run the shallow-cap acceptance profile, states at or above it run
    the deep-cap profile -- the conditioning the gate exists to exploit."""
    n = gate + 1
    profs = [pd if s >= gate else ps for s in range(n)]
    depths = [
        greedy_depth(profs[s], DEEP_CAP if s >= gate else SHALLOW_CAP)
        for s in range(n)
    ]
    accept = [full_accept_prob(profs[s], depths[s]) for s in range(n)]
    pi = [1.0 / n] * n
    for _ in range(20000):
        nxt = [0.0] * n
        for s in range(n):
            f = accept[s]
            nxt[min(s + 1, gate)] += pi[s] * f
            nxt[0] += pi[s] * (1.0 - f)
        delta = sum(abs(nxt[i] - pi[i]) for i in range(n))
        pi = nxt
        if delta < 1e-15:
            break
    return pi, depths, profs


def evaluate(gate: int, ps: list[float], pd: list[float], rebase_head: bool) -> dict:
    """gate>=1: shipped ladder. gate==0: deep cap always. gate<0: shallow only."""
    if gate == 0:
        pi, depths, profs = [1.0], [greedy_depth(pd, DEEP_CAP)], [pd]
    elif gate < 0:
        pi, depths, profs = [1.0], [greedy_depth(ps, SHALLOW_CAP)], [ps]
    else:
        pi, depths, profs = stationary(gate, ps, pd)

    mean_cost = sum(pi[s] * round_cost_ms(depths[s], rebase_head) for s in range(len(pi)))
    mean_tokens = sum(
        pi[s] * (1.0 + expected_accepted(profs[s], depths[s])) for s in range(len(pi))
    )
    mean_depth = sum(pi[s] * depths[s] for s in range(len(pi)))
    p_depth8 = sum(pi[s] for s in range(len(pi)) if depths[s] == 8)
    deep_share = sum(pi[s] for s in range(len(pi)) if depths[s] > SHALLOW_CAP)
    ms_per_token = mean_cost / mean_tokens + NON_BLOCK_OVERHEAD_MS_PER_TOKEN
    return {
        "gate": gate,
        "profile_shallow": [round(x, 6) for x in ps],
        "profile_deep": [round(x, 6) for x in pd],
        "stationary": [round(x, 6) for x in pi],
        "depth_by_state": depths,
        "mean_effective_depth": mean_depth,
        "p_depth_8": p_depth8,
        "deep_round_share": deep_share,
        "accepted_tokens_per_round": mean_tokens,
        "mean_round_cost_ms": mean_cost,
        "ms_per_token": ms_per_token,
        "raw": SERIAL_S_PER_TOKEN / (ms_per_token / 1000.0),
    }


def crossover(gate_a: int, gate_b: int, rebase_head: bool, mk) -> float | None:
    """Lowest q in (0.50, 0.999] at which raw(gate_a) overtakes raw(gate_b)."""
    lo, hi = 0.50, 0.999
    f = lambda x: evaluate(gate_a, *mk(x), rebase_head)["raw"] - evaluate(
        gate_b, *mk(x), rebase_head
    )["raw"]
    if f(hi) <= 0 or f(lo) >= 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/occupancy-fb2.json")
    ap.add_argument(
        "--ema-threshold",
        type=float,
        default=0.90,
        help="tau for the EMA-conditioned gate arm (EMA[4] >= tau opens deep)",
    )
    ap.add_argument(
        "--profile-from",
        help="trace log whose realised per-position acceptance shapes the sweep; "
        "without it every position shares the same q",
    )
    args = ap.parse_args()

    measured = None
    measured_shallow = None
    measured_deep = None
    if args.profile_from:
        measured = position_acceptance(args.profile_from)
        measured_shallow = position_acceptance(args.profile_from, only_deep=False)
        measured_deep = position_acceptance(args.profile_from, only_deep=True)

        def _clean(stats):
            base = [
                min(max(x, 0.05), 0.995) if x is not None else 0.9
                for x in stats["acceptance"]
            ]
            return base, [max(w, 1) for w in stats["observations"]]

        ps0, ws0 = _clean(measured_shallow)
        pd0, wd0 = _clean(measured_deep)
        mk = lambda q: shifted_pair(ps0, ws0, pd0, wd0, q)
    else:
        mk = lambda q: ([q] * DEEP_CAP, [q] * DEEP_CAP)

    qs = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.98]
    out: dict = {
        "measured_position_acceptance": measured,
        "measured_position_acceptance_shallow_cap_rounds": measured_shallow,
        "measured_position_acceptance_deep_cap_rounds": measured_deep,
        "constants": {
            "headStepCostRatio": H,
            "sdpaWidthWallDepthCap": SHALLOW_CAP,
            "segmentedVerifyDepthCap": DEEP_CAP,
            "serial_s_per_token": SERIAL_S_PER_TOKEN,
            "measured_round_cost_ms_local": MEASURED_C_LOCAL_MS,
            "head_delta_ms_per_draft": HEAD_DELTA_MS_PER_DRAFT,
            "ema_threshold": args.ema_threshold,
        },
        "frames": {},
    }

    for frame, rebase in (("local", False), ("head_rebased", True)):
        rows = []
        for q in qs:
            profile = mk(q)
            row = {"q": q}
            for label, gate in (
                ("shallow_only", -1),
                ("gate_3", 3),
                ("gate_2", 2),
                ("gate_1", 1),
                ("no_gate", 0),
            ):
                row[label] = evaluate(gate, *profile, rebase)
            # EMA-conditioned arm: with a homogeneous q the per-position EMA
            # concentrates on q, so the EMA test degenerates to a prompt-level
            # threshold on top of the streak ladder.
            row["gate_ema"] = dict(
                row["gate_2"] if q >= args.ema_threshold else row["shallow_only"]
            )
            row["gate_ema"]["gate"] = f"ema(tau={args.ema_threshold})"
            rows.append(row)
        out["frames"][frame] = {
            "rows": rows,
            "crossovers": {
                "gate_3_overtakes_shallow_only": crossover(3, -1, rebase, mk),
                "gate_2_overtakes_shallow_only": crossover(2, -1, rebase, mk),
                "gate_1_overtakes_shallow_only": crossover(1, -1, rebase, mk),
                "no_gate_overtakes_shallow_only": crossover(0, -1, rebase, mk),
                "gate_1_overtakes_gate_2": crossover(1, 2, rebase, mk),
                "gate_2_overtakes_gate_3": crossover(2, 3, rebase, mk),
            },
        }

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    for frame in ("local", "head_rebased"):
        print(f"\n=== frame: {frame} ===")
        print(
            f"{'q':>6} {'depth(sh/deep)':>15} {'P(d=8)':>8} {'tok/rnd':>8} "
            f"{'raw:sh':>7} {'raw:g3':>7} {'raw:g2':>7} {'raw:g1':>7} {'raw:none':>8}"
        )
        for row in out["frames"][frame]["rows"]:
            sh = row["shallow_only"]["depth_by_state"][0]
            dp = row["no_gate"]["depth_by_state"][0]
            print(
                f"{row['q']:>6.2f} {f'{sh}/{dp}':>15} "
                f"{row['gate_2']['p_depth_8']:>8.4f} "
                f"{row['gate_2']['accepted_tokens_per_round']:>8.3f} "
                f"{row['shallow_only']['raw']:>7.4f} {row['gate_3']['raw']:>7.4f} "
                f"{row['gate_2']['raw']:>7.4f} {row['gate_1']['raw']:>7.4f} "
                f"{row['no_gate']['raw']:>8.4f}"
            )
        print("crossovers:", json.dumps(out["frames"][frame]["crossovers"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
