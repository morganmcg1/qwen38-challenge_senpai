#!/usr/bin/env python3
"""Research-only (qwen38-r1-e17-curve-transfer-and-refit, Q2/Q3).

A faithful Python port of the shipped marginal-depth gate, used to answer the
two analytical questions that were given no timing slot:

  Q2  why the shipped h vector is non-monotone, what a monotone-by-construction
      refit looks like, and what depth histogram / round count each candidate
      curve would produce -- reported BEFORE the timing arms are read.
  Q3  whether the measured 0.655..0.898% advantage of the 0.20 scalar over the
      0.18 scalar is explained by a gate flip, computed rather than re-measured.

WHY A SIMULATOR AND NOT A TRACE.  `MLX_QWEN_MTP_TRACE=1` writes only hexfloat
top-2 row dumps (`Qwen36MTPBlockSession.swift` traceRow/traceWrite); no reach,
threshold or per-round EMA value is logged anywhere, and adding that logging
would need a third research build plus non-timed passes this assignment does not
fund.  What the timed reports DO carry is `effective_draft_lengths` (the exact
per-round chosen depth), `round_count`, `accepted_draft_total` and
`rejected_draft_total`.  That is enough to FIT the latent acceptance process per
prompt and then run the real gate forward under a different h vector.

The fit is deliberately 3-parameter against 3 targets, so it cannot absorb an
arbitrary depth histogram: the histogram SHAPE is a prediction, not an input.

Falsification built in: the same fitted process is used to predict the FLAT18
arm, which E17 measures.  If predicted-vs-measured FLAT18 depth histograms
disagree, every Q2/Q3 number here is unreliable and is reported as such.

Ported verbatim from Sources/MLXFastModel/Qwen36MTPBlockSession.swift:
  * headStepCostRatioByDepth (l.575), sdpaWidthWallDepthCap = 5 (l.610),
    segmentedVerifyDepthCap = 8, segmentedStreakGate = 3 (l.616-617)
  * positionAcceptEMA prior 0.85 * 0.98**i (l.496), acceptEMAAlpha = 0.15
  * costModelDepth (l.621), recordAcceptOutcome (l.669),
    fullAcceptStreak update (l.1126)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAX_DEPTH = 8
SDPA_WIDTH_WALL_DEPTH_CAP = 5
SEGMENTED_VERIFY_DEPTH_CAP = 8
SEGMENTED_STREAK_GATE = 3
ACCEPT_EMA_ALPHA = 0.15
EMA_PRIOR = [0.85 * (0.98 ** i) for i in range(MAX_DEPTH)]
OPTIMISM_CAP = 0.95

SHIPPED_CURVE = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]
# The five-arm forced-depth marginals quoted in the shipped vector's own doc
# comment (l.564-566).  Only h[0..3] were measured directly.
MEASURED_MARGINALS = [0.0971, 0.1152, 0.2482, 0.3761]


def refit_curve() -> list[float]:
    """Monotone-by-construction refit.

    h[0..3] are replaced by the direct forced-depth marginals, which are already
    strictly increasing (0.0971 < 0.1152 < 0.2482 < 0.3761).  h[4..7] were never
    measured directly; the shipped tail is non-monotone there too (0.2919,
    0.3000, 0.2870, 0.3909 -- h[6] < h[5]).  Monotonicity is imposed with a
    running maximum, which is the minimal edit that cannot invent a cost the
    stack was never observed to pay: every entry is either a measured marginal
    or the largest cost already established at a shallower step.
    """
    h = list(MEASURED_MARGINALS) + SHIPPED_CURVE[len(MEASURED_MARGINALS):]
    for i in range(1, len(h)):
        h[i] = max(h[i], h[i - 1])
    return h


def scalar_curve(value: float) -> list[float]:
    return [value] * MAX_DEPTH


@dataclass
class GateState:
    """Exactly the mutable gate state of one Qwen36MTPBlockSession."""

    h: list[float]
    ema: list[float] = field(default_factory=lambda: list(EMA_PRIOR))
    streak: int = 0

    def cost_model_depth(self, offered_depth: int, margin: float | None) -> int:
        width_cap = (
            SEGMENTED_VERIFY_DEPTH_CAP
            if self.streak >= SEGMENTED_STREAK_GATE
            else SDPA_WIDTH_WALL_DEPTH_CAP
        )
        cap = min(min(offered_depth, MAX_DEPTH), width_cap)
        if cap <= 0:
            return 0
        reach = 1.0
        expected = 0.0
        cum_h = 0.0
        depth = 0
        while depth < cap:
            p = self.ema[depth]
            if margin is not None:
                if depth == 0:
                    p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
                elif depth == 1:
                    p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
            reach *= p
            threshold = self.h[depth] * (1.0 + expected) / (1.0 + cum_h)
            if not reach > threshold:
                break
            expected += reach
            cum_h += self.h[depth]
            depth += 1
        return depth

    def record_accept_outcome(self, accepted: int, drafted: int) -> None:
        for i in range(min(accepted, len(self.ema))):
            self.ema[i] += ACCEPT_EMA_ALPHA * (1.0 - self.ema[i])
        # `stoppedEarly` needs the emitted token ids; a 512-token fixed window
        # never terminates on a stop token (the parent continues the trajectory
        # past EOS), so the early-stop branch is unreachable here.
        if accepted < drafted and accepted < len(self.ema):
            self.ema[accepted] += ACCEPT_EMA_ALPHA * (0.0 - self.ema[accepted])
        elif accepted == drafted and drafted > 0 and accepted < len(self.ema):
            if self.ema[accepted] < OPTIMISM_CAP:
                self.ema[accepted] += ACCEPT_EMA_ALPHA * (
                    OPTIMISM_CAP - self.ema[accepted]
                )
        self.streak = self.streak + 1 if accepted == drafted else 0


@dataclass
class Process:
    """The latent per-prompt acceptance process. Three free parameters.

    q[i] = a * b**i           per-position conditional accept probability
    margin ~ Exponential(m)   the pendingTop2 top-2 logit gap that clamps
                              positions 0 and 1
    """

    a: float
    b: float
    m: float

    def q(self, i: int) -> float:
        return min(1.0, max(0.0, self.a * (self.b ** i)))


@dataclass
class SimResult:
    depth_hist: dict[int, int]
    round_count: int
    accepted: int
    rejected: int
    tokens: int

    @property
    def mean_depth(self) -> float:
        total = sum(d * n for d, n in self.depth_hist.items())
        return total / max(self.round_count, 1)

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(self.accepted + self.rejected, 1)

    @property
    def max_depth_seen(self) -> int:
        return max((d for d, n in self.depth_hist.items() if n), default=0)

    def cost_units(self, h: list[float]) -> float:
        """Round cost in batched-verify-forward units, using the SAME h the
        gate optimises against: one verify forward plus the head steps.  This is
        the model's own currency, not a wall-clock claim."""
        total = 0.0
        for d, n in self.depth_hist.items():
            total += n * (1.0 + sum(h[:d]))
        return total


def simulate(
    h: list[float],
    proc: Process,
    tokens: int = 512,
    offered_depth: int = MAX_DEPTH,
    trials: int = 200,
    seed: int = 20260817,
) -> SimResult:
    rng = random.Random(seed)
    hist: dict[int, int] = {d: 0 for d in range(MAX_DEPTH + 1)}
    rounds = accepted_total = rejected_total = tokens_total = 0
    for _ in range(trials):
        state = GateState(h=list(h))
        emitted = 0
        while emitted < tokens:
            margin = rng.expovariate(1.0 / proc.m) if proc.m > 0 else None
            depth = state.cost_model_depth(offered_depth, margin)
            accepted = 0
            for i in range(depth):
                if rng.random() < proc.q(i):
                    accepted += 1
                else:
                    break
            state.record_accept_outcome(accepted, depth)
            hist[depth] += 1
            rounds += 1
            accepted_total += accepted
            rejected_total += depth - accepted
            emitted += 1 + accepted
        tokens_total += emitted
    return SimResult(hist, rounds, accepted_total, rejected_total, tokens_total)


def fit_process(
    h: list[float],
    target_mean_depth: float,
    target_accept_rate: float,
    target_rounds_per_token: float,
    trials: int = 40,
) -> tuple[Process, float]:
    """Coarse-to-fine grid search on (a, b, m).

    Three targets, three parameters, so the depth HISTOGRAM and the round count
    under a DIFFERENT h vector remain predictions rather than fitted values.
    """
    best: tuple[Process, float] | None = None
    grid_a = [0.60 + 0.02 * i for i in range(21)]
    grid_b = [0.86 + 0.01 * i for i in range(15)]
    grid_m = [1.0 + 0.5 * i for i in range(15)]
    for a in grid_a:
        for b in grid_b:
            for m in grid_m:
                proc = Process(a, b, m)
                res = simulate(h, proc, trials=trials)
                rpt = res.round_count / max(res.tokens, 1)
                loss = (
                    ((res.mean_depth - target_mean_depth) / target_mean_depth) ** 2
                    + ((res.accept_rate - target_accept_rate) / target_accept_rate) ** 2
                    + ((rpt - target_rounds_per_token) / target_rounds_per_token) ** 2
                )
                if best is None or loss < best[1]:
                    best = (proc, loss)
    assert best is not None
    return best


def threshold_table(h: list[float], q: list[float]) -> list[dict[str, float]]:
    """The exact gate algebra, tabulated.

    Extend from depth d to d+1 iff f(d+1) > f(d) with f(d) = (1+E_d)/(1+H_d),
    E_d = sum_{j<d} prod_{k<=j} p_k, H_d = sum_{j<d} h[j].  Cross-multiplying:

        r_d (1 + H_d) > (1 + E_d) h[d],  r_d = prod_{k<=d} p_k

    i.e. reach > h[d] (1 + E_d) / (1 + H_d), term for term the shipped code.
    """
    rows = []
    reach = 1.0
    expected = 0.0
    cum_h = 0.0
    for d in range(min(len(h), len(q))):
        reach *= q[d]
        thr = h[d] * (1.0 + expected) / (1.0 + cum_h)
        rows.append(
            {
                "depth": d,
                "h": h[d],
                "reach": reach,
                "threshold": thr,
                "slack": reach - thr,
                "opens": reach > thr,
                "value_at_d": (1.0 + expected) / (1.0 + cum_h),
            }
        )
        if reach > thr:
            expected += reach
            cum_h += h[d]
    return rows


def load_measured(runs_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not runs_root.is_dir():
        return out
    for run_dir in sorted(runs_root.iterdir()):
        report = run_dir / "reports" / "04-mtp-timed.json"
        if not report.is_file():
            continue
        try:
            doc = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        out[run_dir.name] = doc
    return out


def fmt_hist(hist: dict[int, int], rounds: int) -> str:
    parts = []
    for d in range(MAX_DEPTH + 1):
        n = hist.get(d, 0)
        if n:
            parts.append(f"d{d}:{100.0 * n / max(rounds, 1):.1f}%")
    return " ".join(parts)


def report_curves(args: argparse.Namespace) -> None:
    curves = {
        "shipped": SHIPPED_CURVE,
        "refit": refit_curve(),
        "flat0.18": scalar_curve(0.18),
        "flat0.20": scalar_curve(0.20),
    }

    print("=== Q2a: the non-monotone shipped fit ===")
    print(f"{'d':>2}  {'shipped':>8}  {'measured':>9}  {'refit':>8}  note")
    ref = refit_curve()
    for d in range(MAX_DEPTH):
        meas = f"{MEASURED_MARGINALS[d]:.4f}" if d < len(MEASURED_MARGINALS) else "-"
        note = []
        if d > 0 and SHIPPED_CURVE[d] < SHIPPED_CURVE[d - 1]:
            note.append("shipped NON-MONOTONE here")
        if d < len(MEASURED_MARGINALS):
            err = 100.0 * (SHIPPED_CURVE[d] - MEASURED_MARGINALS[d]) / MEASURED_MARGINALS[d]
            note.append(f"shipped {err:+.1f}% vs measured")
        print(
            f"{d:>2}  {SHIPPED_CURVE[d]:>8.4f}  {meas:>9}  {ref[d]:>8.4f}  "
            + "; ".join(note)
        )

    proc = Process(args.a, args.b, args.m)
    q = [proc.q(i) for i in range(MAX_DEPTH)]
    print()
    print("=== Q2b: exact gate-threshold algebra at the converged EMA ===")
    print(f"q (per-position accept) = {[round(v, 4) for v in q]}")
    for name, h in curves.items():
        rows = threshold_table(h, q)
        opened = sum(1 for r in rows if r["opens"])
        print(f"\n-- {name}: gate opens to depth {opened}")
        print(f"{'d':>2}  {'h[d]':>7}  {'reach':>8}  {'thresh':>8}  {'slack':>9}  {'f(d)':>7}  open")
        for r in rows:
            print(
                f"{r['depth']:>2}  {r['h']:>7.4f}  {r['reach']:>8.5f}  "
                f"{r['threshold']:>8.5f}  {r['slack']:>+9.5f}  "
                f"{r['value_at_d']:>7.4f}  {'yes' if r['opens'] else 'NO'}"
            )

    print()
    print("=== Q2c / Q3: simulated depth histogram and round count per curve ===")
    print(f"(prior process a={proc.a} b={proc.b} m={proc.m}, "
          f"{args.tokens}-token window, offered depth 8, {args.trials} trials)")
    # Cost is scored under the TRUTH curve, not under each schedule's own belief.
    # Scoring under the shipped vector would credit it for understating h[0..1]:
    # its own f(d) is inflated by exactly the fitting error being diagnosed.
    truth = refit_curve()
    base_truth = base_own = None
    for name, h in curves.items():
        res = simulate(h, proc, tokens=args.tokens, trials=args.trials)
        rpt = res.round_count / max(res.tokens, 1)
        cost = res.cost_units(truth) / max(res.tokens, 1)
        own = res.cost_units(h) / max(res.tokens, 1)
        if base_truth is None:
            base_truth, base_own = cost, own
        print(
            f"{name:>9}: mean_depth={res.mean_depth:5.3f} max={res.max_depth_seen} "
            f"rounds/token={rpt:.4f} accept_rate={res.accept_rate:.4f}"
        )
        print(
            f"{'':>9}  cost/token under measured truth={cost:.4f} "
            f"({100.0 * (cost - base_truth) / base_truth:+.2f}% vs shipped); "
            f"under own belief={own:.4f}"
        )
        print(f"{'':>9}  hist {fmt_hist(res.depth_hist, res.round_count)}")

    print()
    print("NOTE: cost/token is in the gate's own verify-forward units under the")
    print("measured marginals; it is an ordering claim, not a seconds claim.")
    print("The gap between the two columns for `shipped` IS the fitting error.")


CURVES = {
    "shipped": SHIPPED_CURVE,
    "refit": refit_curve(),
    "flat0.18": scalar_curve(0.18),
    "flat0.20": scalar_curve(0.20),
}


def hist_cost_per_token(hist: dict[int, int], h: list[float], tokens: int) -> float:
    """Cost of a MEASURED round sequence, in batched-verify-forward units.

    A round that drafts d tokens costs one verify forward plus d head steps:
    1 + sum(h[:d]).  This is the gate's own cost model, evaluated on the depths
    the gate actually chose, so it needs no acceptance model at all.
    """
    return sum(n * (1.0 + sum(h[:d])) for d, n in hist.items()) / tokens


def report_costcheck(args: argparse.Namespace) -> None:
    """Q3, and the load-bearing validation for Q2.

    The cost model is tested directly: take each arm's MEASURED per-round depth
    sequence, price it under a candidate h vector, and compare the predicted
    cost ratio between arms with the measured decode seconds/token ratio.

    Prefill is subtracted HERE and only here.  It is ~3.995 s of arm-independent
    work in every leg, so leaving it in dilutes every relative decode claim by a
    constant factor; the ranked score, by contrast, is prefill-INCLUSIVE and is
    never computed this way.  See research/e17-notes.md section 2.
    """
    labels = args.labels.split(",")
    arms = dict(kv.split("=", 1) for kv in args.arm_map.split(",") if kv)
    runs_root = Path(args.runs_root)
    data = {}
    for label in labels:
        mtp = json.loads((runs_root / label / "reports" / "04-mtp-timed.json").read_text())
        hist: dict[int, int] = {}
        for d in mtp["effective_draft_lengths"]:
            hist[d] = hist.get(d, 0) + 1
        tokens = mtp["decode_token_count"]
        spt = mtp["parent_measured_seconds_per_token"]
        prefill = mtp["seed_prefill_seconds"]
        data[label] = {
            "hist": hist,
            "tokens": tokens,
            "spt_inclusive": spt,
            "spt_decode": spt - prefill / tokens,
            "rounds": len(mtp["effective_draft_lengths"]),
            "curve": arms.get(label, "?"),
        }

    print("=== measured arms (prose golden, 512 decode tokens) ===")
    print(f"{'label':>6} {'curve':>9} {'rounds':>7} {'mean_d':>7} "
          f"{'spt_incl':>10} {'spt_decode':>11}  hist")
    for label, d in data.items():
        mean_d = sum(k * v for k, v in d["hist"].items()) / d["rounds"]
        hist_s = " ".join(f"d{k}:{v}" for k, v in sorted(d["hist"].items()))
        print(f"{label:>6} {d['curve']:>9} {d['rounds']:>7} {mean_d:>7.3f} "
              f"{d['spt_inclusive']:>10.6f} {d['spt_decode']:>11.6f}  {hist_s}")

    print()
    print("=== cost model priced on the MEASURED depth sequences ===")
    for cname, h in CURVES.items():
        tag = "  <- measured marginals (truth candidate)" if cname == "refit" else ""
        print(f"\n-- priced under h = {cname}{tag}")
        for label, d in data.items():
            c = hist_cost_per_token(d["hist"], h, d["tokens"])
            print(f"{label:>6}: cost/token = {c:.6f}")

    print()
    print("=== prediction test: pairwise cost ratio vs measured decode ratio ===")
    print("(a pair is a real test only because the depth sequences differ; both")
    print(" arms ran the same prompt, host, window and head)")
    order = list(data)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            meas_incl = 100.0 * (data[b]["spt_inclusive"] - data[a]["spt_inclusive"]) \
                / data[b]["spt_inclusive"]
            meas_dec = 100.0 * (data[b]["spt_decode"] - data[a]["spt_decode"]) \
                / data[b]["spt_decode"]
            print(f"\n{a} vs {b}: {a} is {meas_dec:+.3f}% cheaper on DECODE seconds "
                  f"({meas_incl:+.3f}% on prefill-inclusive seconds)")
            for cname, h in CURVES.items():
                ca = hist_cost_per_token(data[a]["hist"], h, data[a]["tokens"])
                cb = hist_cost_per_token(data[b]["hist"], h, data[b]["tokens"])
                pred = 100.0 * (cb - ca) / cb
                print(f"{'':>4} h={cname:>9}: predicted {pred:+.3f}%  "
                      f"(error vs decode {pred - meas_dec:+.3f} pp)")


def report_fit(args: argparse.Namespace) -> None:
    runs = load_measured(Path(args.runs_root))
    if not runs:
        print(f"no timed reports under {args.runs_root}", file=sys.stderr)
        raise SystemExit(1)
    print("=== fit the latent process to each measured arm, then predict the other ===")
    rows = []
    for label, doc in runs.items():
        mean_depth = doc.get("effective_mean_draft_len")
        accept_rate = doc.get("accepted_draft_rate")
        rounds = doc.get("round_count")
        tokens = doc.get("decode_token_count") or doc.get("token_count")
        if None in (mean_depth, accept_rate, rounds, tokens):
            print(f"{label}: missing counters, skipped", file=sys.stderr)
            continue
        h = SHIPPED_CURVE if label.endswith("CURVE") else scalar_curve(0.18)
        proc, loss = fit_process(h, mean_depth, accept_rate, rounds / tokens)
        rows.append((label, proc, loss, mean_depth, accept_rate, rounds / tokens))
        print(
            f"{label:>26}: measured mean_depth={mean_depth:.3f} "
            f"accept={accept_rate:.4f} rounds/tok={rounds / tokens:.4f} "
            f"-> a={proc.a:.3f} b={proc.b:.3f} m={proc.m:.2f} loss={loss:.5f}"
        )
    if not rows:
        raise SystemExit(1)
    print()
    print("=== cross-prediction (fit on one arm, predict the other) ===")
    for label, proc, _loss, *_ in rows:
        other = scalar_curve(0.18) if label.endswith("CURVE") else SHIPPED_CURVE
        other_name = "flat0.18" if label.endswith("CURVE") else "shipped"
        res = simulate(other, proc, tokens=args.tokens, trials=args.trials)
        print(
            f"{label:>26} -> {other_name:>9}: predicted mean_depth="
            f"{res.mean_depth:.3f} rounds/tok="
            f"{res.round_count / max(res.tokens, 1):.4f} "
            f"accept={res.accept_rate:.4f}"
        )
        print(f"{'':>26}    hist {fmt_hist(res.depth_hist, res.round_count)}")
    print()
    print("=== the same fitted processes, applied to the UNMEASURED curves ===")
    for label, proc, _loss, *_ in rows:
        for name, h in (("refit", refit_curve()), ("flat0.20", scalar_curve(0.20))):
            res = simulate(h, proc, tokens=args.tokens, trials=args.trials)
            print(
                f"{label:>26} -> {name:>9}: mean_depth={res.mean_depth:.3f} "
                f"rounds/tok={res.round_count / max(res.tokens, 1):.4f} "
                f"accept={res.accept_rate:.4f} "
                f"cost/tok={res.cost_units(SHIPPED_CURVE) / max(res.tokens, 1):.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=".mlxfast-private/e17/runs")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--a", type=float, default=0.86, help="q[0] of the prior process")
    ap.add_argument("--b", type=float, default=0.94, help="geometric decay of q")
    ap.add_argument("--m", type=float, default=4.0, help="mean top-2 margin")
    ap.add_argument("--fit", action="store_true", help="fit against measured arms")
    ap.add_argument(
        "--costcheck",
        action="store_true",
        help="price measured depth sequences under each candidate h and compare "
        "with measured decode seconds",
    )
    ap.add_argument("--labels", default="Hp3,Sp3,S20p")
    ap.add_argument(
        "--arm-map",
        default="Hp3=shipped,Sp3=flat0.18,S20p=flat0.20",
        help="LABEL=CURVE pairs naming which h vector each measured arm ran",
    )
    args = ap.parse_args()
    if args.costcheck:
        report_costcheck(args)
    elif args.fit:
        report_fit(args)
    else:
        report_curves(args)


if __name__ == "__main__":
    main()
