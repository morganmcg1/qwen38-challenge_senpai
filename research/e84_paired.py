#!/usr/bin/env python3
"""Paired per-round analysis of an E84 palindrome timing session.

Every arm in an E84 session is bit-exact with every other arm: the four
builds emit the same tokens and therefore the same (draft, accepted)
sequence in every round. Round i of arm X is then a like-for-like pair with
round i of the base arm, so a paired statistic over rounds removes the
round-to-round workload variation that dominates a leg total.

Estimator
---------
The session order is a palindrome: base a b ab ab b a base. Each arm holds
one early leg and one late leg placed symmetrically about the centre, so a
statistic that averages the two symmetric positions cancels linear
leg-position drift exactly.

The point estimate is the mean of the two symmetric per-pair medians:

    E3 = ( median_i(arm_early[i] - base_early[i])
         + median_i(arm_late[i]  - base_late[i]) ) / 2

`--selftest` reruns the simulation study that selected E3. Against a known
injected effect, with linear drift and one-sided multi-ms scheduling
spikes, E3 had the lowest RMSE of every estimator tried, and its
round-cluster bootstrap CI covered the truth 92-97 % of the time against a
nominal 95 %. The naive leg total had 2-3x the RMSE of E3.

Position-endpoint rule
----------------------
E86 measured that the first and last legs of a session carry inflated host
phases (1146 and 3228 us/round against 635-690 us/round for interior legs).
A palindrome cancels linear drift, but it does not cancel that endpoint
premium: it places both extremes on whichever arm sits at position 1 and
position N. Never give the reference arm both extremes, because every
arm-minus-reference number then becomes an upper bound instead of an
estimate. Add a throwaway arm at position 0 and at the last position, and
start the palindrome at position 1. Arm-to-arm contrasts inside the
palindrome stay position-balanced either way.

The null control below pairs the two reference legs. It prices leg-position
drift and leg noise. It does not price the endpoint premium, because both
of its legs are endpoints.

harness=local for every number produced here.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

# Non-overlapping split of round_us, in emission order.
PHASES = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
# draft_build_us is itself split by these; a sub-decomposition, not an addend.
SUBPHASES = [
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
]

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?)")


def parse_trace(path: Path) -> list[list[dict]]:
    """Split one append-mode trace file into sessions of rounds.

    The sink is opened O_APPEND once per process and a leg runs several
    workers against the same path, so one file holds several sessions.
    """
    sessions: list[list[dict]] = []
    current: list[dict] | None = None
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("mtp-trace: begin"):
            current = []
            sessions.append(current)
            continue
        if not raw.startswith("mtp-trace: round="):
            continue
        if current is None:
            current = []
            sessions.append(current)
        current.append({k: float(v) for k, v in KV.findall(raw)})
    return sessions


def describe(sessions: list[list[dict]]) -> list[dict]:
    out = []
    for i, s in enumerate(sessions):
        if not s:
            out.append({"index": i, "rounds": 0})
            continue
        out.append(
            {
                "index": i,
                "rounds": len(s),
                "rounds_with_drafts": sum(1 for r in s if r.get("d", 0) > 0),
                "mean_d": round(statistics.fmean(r.get("d", 0) for r in s), 4),
                "total_round_ms": round(
                    sum(r.get("round_us", 0) for r in s) / 1000.0, 1
                ),
            }
        )
    return out


def select_session(sessions, want: str, tag: str):
    if want != "auto":
        return sessions[int(want)]
    spec = [s for s in sessions if s and any(r.get("d", 0) > 0 for r in s)]
    if len(spec) != 1:
        raise SystemExit(
            f"{tag}: expected exactly one speculative session, found {len(spec)}.\n"
            + json.dumps(describe(sessions), indent=2)
            + "\nRe-run with --session <index> after inspecting it."
        )
    return spec[0]


def sequence(session) -> tuple:
    return tuple((int(r.get("d", -1)), int(r.get("acc", -1))) for r in session)


def field_of(session, field) -> list[float]:
    return [r.get(field, 0.0) for r in session]


def e3_point(p0: list[float], p1: list[float], idx=None) -> float:
    if idx is None:
        return (statistics.median(p0) + statistics.median(p1)) / 2.0
    return (
        statistics.median([p0[i] for i in idx])
        + statistics.median([p1[i] for i in idx])
    ) / 2.0


def e3_ci(p0, p1, rng, reps) -> tuple[float, float]:
    """Round-cluster bootstrap: resample rounds, apply to both pairings."""
    n = len(p0)
    vals = [
        e3_point(p0, p1, [rng.randrange(n) for _ in range(n)]) for _ in range(reps)
    ]
    vals.sort()
    return vals[int(0.025 * reps)], vals[min(int(0.975 * reps), reps - 1)]


def build_pairs(by_arm, arm, field):
    """Symmetric pairing: early arm leg with early base leg, late with late."""
    A, B = by_arm[arm], by_arm["base"]
    return (
        [x - y for x, y in zip(field_of(A[0], field), field_of(B[0], field))],
        [x - y for x, y in zip(field_of(A[1], field), field_of(B[1], field))],
    )


def selftest() -> int:
    """Simulation study behind the estimator choice."""
    N = 78
    LEGS = ["base", "a", "b", "ab", "ab", "b", "a", "base"]
    TRUTH = {"base": 0.0, "a": -300.0, "b": +100.0, "ab": -290.0}

    def sim(rng, drift, noise, spike_p):
        shape = [200000 + rng.gauss(0, 25000) for _ in range(N)]
        return [
            [
                shape[r]
                + TRUTH[arm]
                + drift * i
                + rng.gauss(0, noise)
                + (rng.uniform(2000, 12000) if rng.random() < spike_p else 0.0)
                for r in range(N)
            ]
            for i, arm in enumerate(LEGS)
        ]

    def sym(legs, arm):
        A = [legs[i] for i, a in enumerate(LEGS) if a == arm]
        B = [legs[i] for i, a in enumerate(LEGS) if a == "base"]
        return (
            [A[0][r] - B[0][r] for r in range(N)],
            [A[1][r] - B[1][r] for r in range(N)],
        )

    def est_e3(legs, arm):
        return e3_point(*sym(legs, arm))

    def est_e1(legs, arm):
        A = [legs[i] for i, a in enumerate(LEGS) if a == arm]
        B = [legs[i] for i, a in enumerate(LEGS) if a == "base"]
        return statistics.median(
            (A[0][r] + A[1][r]) / 2 - (B[0][r] + B[1][r]) / 2 for r in range(N)
        )

    def est_legtotal(legs, arm):
        A = [legs[i] for i, a in enumerate(LEGS) if a == arm]
        B = [legs[i] for i, a in enumerate(LEGS) if a == "base"]
        return (
            statistics.fmean(sum(x) for x in A) - statistics.fmean(sum(x) for x in B)
        ) / N

    ests = {
        "E3 sym-pair-median": est_e3,
        "E1 legmean-median": est_e1,
        "E5 leg-total": est_legtotal,
    }
    regimes = [
        ("drift 400us/leg, noise 900us, spikes 5%",
         dict(drift=400, noise=900, spike_p=0.05)),
        ("drift 1200us/leg, noise 600us, spikes 8%",
         dict(drift=1200, noise=600, spike_p=0.08)),
        ("no drift, noise 300us, spikes 3%",
         dict(drift=0, noise=300, spike_p=0.03)),
    ]
    print("Estimator RMSE against known truth (us/round), 400 replicates")
    for label, kw in regimes:
        print(f"\n  {label}")
        for name, f in ests.items():
            cells = []
            for a in ("a", "b", "ab"):
                xs = [f(sim(random.Random(1000 + s), **kw), a) for s in range(400)]
                bias = statistics.fmean(xs) - TRUTH[a]
                rmse = statistics.fmean((x - TRUTH[a]) ** 2 for x in xs) ** 0.5
                cells.append(f"{a}: bias{bias:+6.0f} rmse{rmse:6.0f}")
            print(f"    {name:>20}  " + "  ".join(cells))

    print("\nRound-cluster bootstrap CI coverage for E3 (nominal 0.95), 200 replicates")
    for label, kw in regimes:
        row = []
        for a in ("a", "b", "ab"):
            hit = 0
            for s in range(200):
                rng = random.Random(5000 + s)
                p0, p1 = sym(sim(rng, **kw), a)
                lo, hi = e3_ci(p0, p1, rng, 1500)
                hit += lo <= TRUTH[a] <= hi
            row.append(f"{a}: {hit / 200:.3f}")
        print(f"    {label:>42}  " + "  ".join(row))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=".mlxfast-private/e84/runs")
    ap.add_argument(
        "--legs",
        default="q1:base,q2:a,q3:b,q4:ab,q5:ab,q6:b,q7:a,q8:base",
        help="comma list of <tag>:<arm> in run order; must be a palindrome",
    )
    ap.add_argument("--session", default="auto")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument(
        "--base-abs",
        type=float,
        default=0.031942655,
        help="untraced base candidate_mtp_seconds_per_token (harness=local)",
    )
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    rng = random.Random(args.seed)
    order = [tuple(p.split(":")) for p in args.legs.split(",")]
    arms = [a for _t, a in order]
    if arms != arms[::-1]:
        raise SystemExit(f"leg order is not a palindrome: {arms}")
    runs = Path(args.runs_dir)

    raw = {}
    for tag, _arm in order:
        p = runs / tag / "round-trace.txt"
        if not p.exists():
            raise SystemExit(f"missing trace: {p}")
        raw[tag] = parse_trace(p)

    if args.list:
        for tag, _arm in order:
            print(tag, json.dumps(describe(raw[tag])))
        return 0

    sel = {tag: select_session(raw[tag], args.session, tag) for tag, _ in order}

    # 1. Pairing validity: every leg must run the identical round sequence.
    distinct: dict[tuple, list[str]] = {}
    for tag, _a in order:
        distinct.setdefault(sequence(sel[tag]), []).append(tag)
    print("## 1. Pairing validity  (harness=local)")
    print(f"distinct_round_sequences = {len(distinct)}")
    for i, (s, tags) in enumerate(distinct.items()):
        drafted = sum(d for d, _a in s)
        acc = sum(a for _d, a in s)
        rate = acc / drafted if drafted else float("nan")
        print(
            f"  seq[{i}] legs={','.join(tags)} rounds={len(s)} "
            f"drafted={drafted} accepted={acc} accept_rate={rate:.4f}"
        )
    if len(distinct) != 1:
        print(
            "\nPAIRING INVALID: the legs did not run the same round sequence. "
            "Report leg totals only; the paired medians below are not valid."
        )
        return 2
    n_rounds = len(next(iter(distinct)))
    print(f"pairing = VALID ({n_rounds} rounds, identical in all 8 legs)\n")

    by_arm: dict[str, list] = {}
    for tag, arm in order:
        by_arm.setdefault(arm, []).append(sel[tag])

    base_round = [
        statistics.fmean(leg[i].get("round_us", 0.0) for leg in by_arm["base"])
        for i in range(n_rounds)
    ]
    base_med_round = statistics.median(base_round)

    # 2. Paired per-round effect, primary estimator E3.
    print("## 2. Paired per-round delta vs base, estimator E3  (harness=local)")
    hdr = (
        f"{'arm':>4} {'E3 us':>9} {'boot CI95 us':>21} {'E3 %':>9} "
        f"{'CI95 %':>19} {'E1 us':>9} {'legtotal %':>11} {'neg/pos':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    results = []
    leg_secs = {t: sum(field_of(sel[t], "round_us")) / 1e6 for t, _a in order}
    base_leg_tot = statistics.fmean(leg_secs[t] for t, a in order if a == "base")
    armlist = [a for a in ("a", "b", "ab") if a in by_arm]
    for arm in armlist:
        p0, p1 = build_pairs(by_arm, arm, "round_us")
        pt = e3_point(p0, p1)
        lo, hi = e3_ci(p0, p1, rng, args.reps)
        e1 = statistics.median(
            [
                statistics.fmean(leg[i].get("round_us", 0.0) for leg in by_arm[arm])
                - base_round[i]
                for i in range(n_rounds)
            ]
        )
        allpairs = p0 + p1
        lt = (
            100.0
            * (statistics.fmean(leg_secs[t] for t, a in order if a == arm) - base_leg_tot)
            / base_leg_tot
        )
        r = {
            "arm": arm,
            "e3_us": pt,
            "e3_ci95_us": [lo, hi],
            "e3_pct": 100.0 * pt / base_med_round,
            "e3_ci95_pct": [100.0 * lo / base_med_round, 100.0 * hi / base_med_round],
            "e1_us": e1,
            "leg_total_pct": lt,
            "n_negative": sum(1 for d in allpairs if d < 0),
            "n_positive": sum(1 for d in allpairs if d > 0),
            "excludes_zero": (lo < 0 and hi < 0) or (lo > 0 and hi > 0),
        }
        results.append(r)
        print(
            f"{arm:>4} {pt:>9.1f} [{lo:>8.1f},{hi:>8.1f}] {r['e3_pct']:>+9.4f} "
            f"[{r['e3_ci95_pct'][0]:>+8.4f},{r['e3_ci95_pct'][1]:>+8.4f}] "
            f"{e1:>9.1f} {lt:>+11.4f} {r['n_negative']:>4}/{r['n_positive']:<4}"
        )
    print(
        "\nE3 = mean of the two symmetric per-pair medians (primary). "
        "E1 = median of leg-averaged deltas (cross-check). "
        "legtotal = naive sum over the same traced legs.\n"
        "The percentage columns are shares of the instrumented round window, "
        "not of the leg. Section 6 converts the us to seconds per token."
    )

    # 3. Null control. The two reference legs sit at the extreme positions, so
    # this pair is deliberately NOT drift-cancelled: it prices leg-position
    # drift plus leg noise, the error the paired design is meant to remove.
    # It does not price the endpoint premium itself, because both legs of the
    # pair are endpoints.
    first_tag, last_tag = order[0][0], order[-1][0]
    print(
        f"\n## 3. Null control: last reference leg {last_tag} minus "
        f"first reference leg {first_tag}  (harness=local)"
    )
    q1, q8 = sel[first_tag], sel[last_tag]
    dnull = [b - a for a, b in zip(field_of(q1, "round_us"), field_of(q8, "round_us"))]
    nmed = statistics.median(dnull)
    nlo, nhi = e3_ci(dnull, dnull, rng, args.reps)
    print(
        f"median {nmed:+.1f} us  boot CI95 [{nlo:+.1f}, {nhi:+.1f}]  "
        f"= {100.0 * nmed / base_med_round:+.4f} % of round  "
        f"neg/pos {sum(1 for d in dnull if d < 0)}/{sum(1 for d in dnull if d > 0)}"
    )

    # 4. Leg totals for the traced legs.
    print("\n## 4. Traced leg totals  (harness=local)")
    for tag, arm in order:
        print(f"  {tag:>4} {arm:>5} {leg_secs[tag]:>12.6f} s")

    # 5. Phase decomposition, E3 per phase.
    print("\n## 5. E3 paired median by phase, us  (harness=local)")
    print(f"{'phase':>18} " + " ".join(f"{a:>10}" for a in armlist))
    phase_rows = {}
    for field in PHASES + ["--"] + SUBPHASES:
        if field == "--":
            print(f"{'(draft_build split)':>18}")
            continue
        cells = [e3_point(*build_pairs(by_arm, a, field)) for a in armlist]
        phase_rows[field] = cells
        print(f"{field:>18} " + " ".join(f"{c:>+10.1f}" for c in cells))

    # 6. Absolute candidate seconds per token.
    tpr = args.tokens / n_rounds
    traced_base_round_s = statistics.fmean(base_round) / 1e6
    untraced_base_round_s = args.base_abs * tpr
    overhead = (
        100.0 * (traced_base_round_s - untraced_base_round_s) / untraced_base_round_s
    )
    print(
        f"\n## 6. Absolute conversion  (harness=local, {args.tokens} tokens / "
        f"{n_rounds} rounds = {tpr:.4f} tokens/round)"
    )
    print(
        f"instrumented round window = {traced_base_round_s * 1000:.3f} ms "
        "(sum of the round_us phases)\n"
        f"leg wall per round        = {untraced_base_round_s * 1000:.3f} ms "
        f"(from {args.base_abs:.9f} s/token)\n"
        f"instrumented coverage     = {100.0 + overhead:.1f} %  "
        "-> round_us does not span the whole decode leg, so a percentage of "
        "the instrumented round is not a percentage of the leg. The us are "
        "unaffected, and the conversion below divides them by tokens per "
        "round only."
    )
    for r in results:
        d = r["e3_us"] * 1e-6 / tpr
        lo = r["e3_ci95_us"][0] * 1e-6 / tpr
        hi = r["e3_ci95_us"][1] * 1e-6 / tpr
        print(
            f"  {r['arm']:>2}: candidate_mtp_seconds_per_token = "
            f"{args.base_abs + d:.9f} (delta {d:+.3e}, "
            f"CI95 [{lo:+.3e}, {hi:+.3e}]) = "
            f"{100.0 * d / args.base_abs:+.4f} % of untraced base"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "harness": "local",
                    "estimator": "E3 mean of symmetric per-pair medians",
                    "n_rounds": n_rounds,
                    "distinct_round_sequences": len(distinct),
                    "arms": results,
                    "null_q8_minus_q1": {
                        "median_us": nmed,
                        "ci95_us": [nlo, nhi],
                        "pct": 100.0 * nmed / base_med_round,
                    },
                    "traced_leg_seconds": leg_secs,
                    "phase_median_us": {
                        k: dict(zip(armlist, v)) for k, v in phase_rows.items()
                    },
                    "traced_base_round_s": traced_base_round_s,
                    "untraced_base_round_s": untraced_base_round_s,
                    "tracing_overhead_pct": overhead,
                    "base_abs_s_per_token": args.base_abs,
                    "tokens_per_round": tpr,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
