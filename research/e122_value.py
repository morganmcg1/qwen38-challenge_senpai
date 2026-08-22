#!/usr/bin/env python3
"""E122 rung 0 -- price the margin as a depth conditioner.

    usage: research/e122_value.py FORCED_RUN_DIR [...] [--json OUT.json]

AUC answers "does the margin separate an accepted draft from a rejected one".
It does not answer the question the experiment exists to settle, which is
"how much decode time would a margin-conditioned depth schedule save". This
script answers the second question directly, on the FORCED-DEPTH arm only.

WHY THE FORCED ARM. With the depth pinned, every round reports its full
accept run length L, the number of drafts accepted before the first rejection.
Under the shipped policy L is censored at a depth the margin already chose, so
the same computation on that arm would be circular.

THE COST MODEL IS THE SHIPPED ONE, not a fit made here:

    time(d)   = V * (1 + 0.18 * d)      Qwen36MTPBlockSession.headStepCostRatio
    tokens(d) = 1 + min(L, d)

so a schedule's cost per token, in units of one verify forward V, is

    sum_r (1 + 0.18 * d_r) / sum_r (1 + min(L_r, d_r)).

NOT A MEASUREMENT. Every number below is this model evaluated on traced
acceptance data. `harness=local`, `timing_valid=false`. The model ignores
rejected-work cache traffic, rollback and replay, so it flatters any policy
that drafts deeper than it should. Read it as an upper bound on the pool.

WHAT IT REPORTS. Three policy classes on the same rounds:

  constant   one depth for every round; the class E99 already optimised
  margin     depth is a function of the margin bin alone; the shippable class
  oracle     depth may depend on L itself; unreachable, so it bounds the pool

The decision number is the share of the oracle pool the margin class captures,
measured out of sample so a bin table cannot fit its own noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e122_auc import quantile, read_meta, read_rounds  # noqa: E402

HEAD_STEP_COST_RATIO = 0.18


def cost(depth: int) -> float:
    return 1.0 + HEAD_STEP_COST_RATIO * depth


def tokens(run_length: int, depth: int) -> int:
    return 1 + min(run_length, depth)


def cost_per_token(rounds: list[dict], depths: list[int]) -> float:
    total_cost = sum(cost(d) for d in depths)
    total_tokens = sum(tokens(r["L"], d) for r, d in zip(rounds, depths))
    return total_cost / total_tokens


def best_constant(rounds: list[dict], dmax: int) -> tuple[int, float]:
    scored = [(cost_per_token(rounds, [d] * len(rounds)), d)
              for d in range(dmax + 1)]
    value, depth = min(scored)
    return depth, value


def dinkelbach(rounds: list[dict], dmax: int, assign) -> tuple[float, list[int]]:
    """Minimise sum(cost)/sum(tokens) over the policy class `assign` defines.

    `assign(rounds, lam, dmax)` returns the depth list that minimises
    sum(cost(d) - lam * tokens(d)) inside the class. Dinkelbach's iteration
    then converges to the class optimum from above.
    """
    depths = [0] * len(rounds)
    lam = cost_per_token(rounds, depths)
    for _ in range(64):
        depths = assign(rounds, lam, dmax)
        nxt = cost_per_token(rounds, depths)
        if abs(nxt - lam) < 1e-13:
            lam = nxt
            break
        lam = nxt
    return lam, depths


def assign_oracle(rounds: list[dict], lam: float, dmax: int) -> list[int]:
    return [min(range(dmax + 1),
                key=lambda d: cost(d) - lam * tokens(r["L"], d))
            for r in rounds]


def bin_index(edges: list[float], margin: float) -> int:
    for i, edge in enumerate(edges):
        if margin <= edge:
            return i
    return len(edges)


def fit_bin_depths(rounds: list[dict], edges: list[float], lam: float,
                   dmax: int) -> list[int]:
    """The depth each margin bin should take at the current lambda."""
    table = []
    for index in range(len(edges) + 1):
        members = [r for r in rounds if bin_index(edges, r["margin"]) == index]
        if not members:
            table.append(0)
            continue
        table.append(min(
            range(dmax + 1),
            key=lambda d: sum(cost(d) - lam * tokens(r["L"], d)
                              for r in members)))
    return table


def apply_bin_depths(rounds: list[dict], edges: list[float],
                     table: list[int]) -> list[int]:
    return [table[bin_index(edges, r["margin"])] for r in rounds]


def margin_edges(rounds: list[dict], bins: int) -> list[float]:
    values = sorted(r["margin"] for r in rounds)
    return [quantile(values, i / bins) for i in range(1, bins)]


def evaluate(rounds: list[dict], dmax: int, bins: int) -> dict:
    const_depth, const_cpt = best_constant(rounds, dmax)
    oracle_cpt, oracle_depths = dinkelbach(rounds, dmax, assign_oracle)

    edges = margin_edges(rounds, bins)

    def assign_margin(rs, lam, dm):
        return apply_bin_depths(rs, edges, fit_bin_depths(rs, edges, lam, dm))

    margin_cpt, margin_depths = dinkelbach(rounds, dmax, assign_margin)
    margin_table = fit_bin_depths(rounds, edges, margin_cpt, dmax)

    # Out of sample: fit the bin table on alternate rounds, spend it on the
    # rest. Alternating splits keep both halves spread over the whole decode
    # and over every prompt, which a prompt-wise split would not.
    train = [r for i, r in enumerate(rounds) if i % 2 == 0]
    test = [r for i, r in enumerate(rounds) if i % 2 == 1]
    held_out = None
    held_out_table = None
    if train and test:
        train_edges = margin_edges(train, bins)

        def assign_train(rs, lam, dm):
            return apply_bin_depths(
                rs, train_edges, fit_bin_depths(rs, train_edges, lam, dm))

        lam_train, _ = dinkelbach(train, dmax, assign_train)
        held_out_table = fit_bin_depths(train, train_edges, lam_train, dmax)
        held_out = cost_per_token(
            test, apply_bin_depths(test, train_edges, held_out_table))

    def gain(reference: float, candidate: float | None) -> float | None:
        if candidate is None or candidate <= 0.0:
            return None
        return (reference / candidate - 1.0) * 100.0

    # The held-out figure must be compared with the best constant ON THE SAME
    # held-out rounds, or the split itself shows up as a gain.
    test_const_cpt = best_constant(test, dmax)[1] if test else None

    return {
        "rounds": len(rounds),
        "dmax": dmax,
        "bins": bins,
        "head_step_cost_ratio": HEAD_STEP_COST_RATIO,
        "mean_run_length": statistics.fmean([r["L"] for r in rounds]),
        "run_length_hist": {
            str(k): sum(1 for r in rounds if r["L"] == k)
            for k in range(dmax + 1)},
        "best_constant_depth": const_depth,
        "best_constant_cost_per_token": const_cpt,
        "oracle_cost_per_token": oracle_cpt,
        "oracle_mean_depth": statistics.fmean(oracle_depths),
        "oracle_pool_pct": gain(const_cpt, oracle_cpt),
        "margin_cost_per_token": margin_cpt,
        "margin_mean_depth": statistics.fmean(margin_depths),
        "margin_pool_pct_in_sample": gain(const_cpt, margin_cpt),
        "margin_bin_edges": edges,
        "margin_bin_depths": margin_table,
        "held_out_margin_cost_per_token": held_out,
        "held_out_constant_cost_per_token": test_const_cpt,
        "held_out_bin_depths": held_out_table,
        "margin_pool_pct_held_out": (
            gain(test_const_cpt, held_out) if test_const_cpt else None),
        "captured_share_of_oracle_pool_pct": (
            None if not gain(const_cpt, oracle_cpt)
            else 100.0 * (gain(const_cpt, margin_cpt) or 0.0)
            / gain(const_cpt, oracle_cpt)),
    }


def self_test(bins: int) -> int:
    """Prove the estimator can report both a null and a full capture.

    A value model that cannot fail is not evidence. The null arm gives the
    conditioner a margin with no information about the run length, and the
    positive arm gives it a margin that determines the run length. The held-out
    figure is the one that must separate them.
    """
    import random

    rng = random.Random(122)
    size = 4000
    null_rounds, perfect_rounds = [], []
    for _ in range(size):
        length = min(7, int(rng.expovariate(0.6)))
        null_rounds.append({"margin": rng.random() * 10.0, "L": length})
        length = min(7, int(rng.expovariate(0.6)))
        perfect_rounds.append({"margin": length + rng.random() * 1e-3,
                               "L": length})

    ok = True
    for name, rounds in (("null", null_rounds), ("perfect", perfect_rounds)):
        block = evaluate(rounds, 7, bins)
        print(f"{name:<8} d*={block['best_constant_depth']} "
              f"oracle_pool={block['oracle_pool_pct']:.2f}% "
              f"margin_in_sample={block['margin_pool_pct_in_sample']:.2f}% "
              f"margin_held_out={block['margin_pool_pct_held_out']:.2f}% "
              f"bins={block['margin_bin_depths']}")
        held = block["margin_pool_pct_held_out"]
        if name == "null" and not (held is not None and held < 0.5):
            print("self-test: the null arm reported a pool", file=sys.stderr)
            ok = False
        if name == "perfect" and not (held is not None and held > 5.0):
            print("self-test: the positive arm reported no pool", file=sys.stderr)
            ok = False
    print("e122_value self-test: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    if args.self_test:
        return self_test(args.bins)
    if not args.run_dirs:
        parser.error("give at least one forced-depth run directory")

    pooled: list[dict] = []
    per_prompt = {}
    forced_seen = set()
    for run_dir in args.run_dirs:
        meta = read_meta(run_dir)
        forced = meta.get("forced_depth")
        if forced in (None, "none"):
            print(f"e122_value: {run_dir} is not a forced-depth arm "
                  f"(forced_depth={forced}); L would be censored", file=sys.stderr)
            return 2
        forced_seen.add(forced)
        rounds = read_rounds(run_dir)
        prompt = meta.get("prompt_id", run_dir.name)
        for r in rounds:
            r["prompt"] = prompt
            # With the depth pinned, acceptance stops at the first rejection,
            # so the accepted count IS the run length. It is right censored
            # only in a round that accepted every draft.
            r["L"] = r["accepted"]
            r["censored"] = r["accepted"] >= r["depth"]
        per_prompt[prompt] = rounds
        pooled.extend(rounds)

    if len(forced_seen) != 1:
        print(f"e122_value: mixed forced depths {sorted(forced_seen)}",
              file=sys.stderr)
        return 2
    dmax = int(next(iter(forced_seen)))

    result = {
        "experiment": "e122-target-margin-conditioned-draft-depth",
        "rung": 0,
        "harness": "local",
        "timing_valid": False,
        "official_or_ranked_score": False,
        "model": "shipped cost model, not a measurement",
        "censored_rounds": sum(1 for r in pooled if r["censored"]),
        "pooled": evaluate(pooled, dmax, args.bins),
        "prompts": {p: evaluate(rs, dmax, args.bins)
                    for p, rs in per_prompt.items() if len(rs) > 4 * args.bins},
    }

    text = ["E122 rung 0 -- modelled value of a margin-conditioned depth schedule",
            "harness=local  timing_valid=false  official_or_ranked_score=false",
            "cost/token is modelled in units of one verify forward, not seconds",
            "",
            f"{'scope':<18}{'n':>6}{'meanL':>8}{'d*':>4}{'const':>9}"
            f"{'margin':>9}{'oracle':>9}{'pool%':>8}{'margin%':>9}"
            f"{'held%':>8}{'share%':>8}"]

    def row(name: str, block: dict) -> str:
        def num(value, width, digits=4):
            if value is None:
                return f"{'--':>{width}}"
            return f"{value:>{width}.{digits}f}"
        return (f"{name:<18}{block['rounds']:>6}"
                f"{block['mean_run_length']:>8.3f}"
                f"{block['best_constant_depth']:>4}"
                f"{num(block['best_constant_cost_per_token'], 9)}"
                f"{num(block['margin_cost_per_token'], 9)}"
                f"{num(block['oracle_cost_per_token'], 9)}"
                f"{num(block['oracle_pool_pct'], 8, 2)}"
                f"{num(block['margin_pool_pct_in_sample'], 9, 2)}"
                f"{num(block['margin_pool_pct_held_out'], 8, 2)}"
                f"{num(block['captured_share_of_oracle_pool_pct'], 8, 1)}")

    text.append(row("pooled", result["pooled"]))
    for prompt, block in sorted(result["prompts"].items()):
        text.append(row(prompt, block))
    text.append("")
    text.append(f"pooled margin bin edges  {['%.4f' % e for e in result['pooled']['margin_bin_edges']]}")
    text.append(f"pooled bin depths        {result['pooled']['margin_bin_depths']}")
    text.append(f"held-out bin depths      {result['pooled']['held_out_bin_depths']}")
    text.append(f"run length histogram     {result['pooled']['run_length_hist']}")
    text.append(f"right-censored rounds    {result['censored_rounds']}")

    report = "\n".join(text)
    print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        args.json.with_suffix(".txt").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
