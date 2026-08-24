#!/usr/bin/env python3
"""E182: does the measured local phase table price ranked width better than
the bare quadratic?

    usage: research/e182_ranked_fold.py [--report research/out/e182/report.json]
                                        [rounds|survival|fit|price|all]

FINDING 456 fitted `R(M) = 30.640 - 0.939*M + 0.576*M^2 ms` (`harness=ranked`)
to 16 ranked (width, round-cost) observations. A quadratic is a smooth
surrogate: it has no mechanism behind it and it prices +7.7 ms between M = 7
and M = 8 where the replica dispatch grid predicts no new work at all.

E182 measures the round's phase shape directly on local M4 (`harness=local`).
This script feeds that measured shape back into the E177 ranked estimator as
extra named cost laws and lets the ranked data judge them against the
quadratic on equal terms:

  phase-local        R = F + s * phi_round(M)
                     phi_round is the measured local round shape, normalised
                     to phi(1) = 0 and phi(9) = 1. Two free parameters, so it
                     is the strictest possible test: one scale carries the
                     whole width dependence.
  phase-local+rows   R = F + s * phi_round(M) + b * M
                     Three parameters, the same count as the quadratic, so AIC
                     compares the SHAPES rather than the parameter budgets.
  phase-split        R = F + s1 * phi_verify(M) + s2 * phi_head(M)
                     The two measured phases enter separately. This is the
                     phase decomposition's own prediction of ranked width cost.

Everything transferred here is a SHAPE. Every coefficient is refitted on the
ranked data, so no local millisecond ever enters a ranked number (FINDING 449:
local M4 coefficients do not transfer; the M5 host answers for its own scale).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402

WIDTHS = list(range(1, 10))


def normalised(table: list[dict], field: str) -> dict[int, float]:
    """Local median of `field` per width, rescaled to 0 at M=1 and 1 at M=9."""
    raw = {row["m"]: row[field] for row in table if field in row}
    missing = [m for m in WIDTHS if m not in raw]
    if missing:
        raise SystemExit(f"e182: {field} missing widths {missing}")
    lo, hi = raw[1], raw[9]
    if hi <= lo:
        raise SystemExit(f"e182: {field} is not increasing in M")
    return {m: (raw[m] - lo) / (hi - lo) for m in WIDTHS}


def band_sum(row: dict) -> float:
    return sum(
        row[f]
        for f in (
            "band_pre_us",
            "band_gdn_mixer_us",
            "band_gdn_mlp_us",
            "band_fa_mixer_us",
            "band_fa_mlp_us",
        )
    )


def expectation(shape: dict[int, float], probs: list[float]) -> float:
    return sum(q * shape[1 + d] for d, q in enumerate(probs))


def install_shapes(phi_round, phi_verify, phi_head) -> None:
    """Register the measured local shapes as named ranked cost laws."""
    e177.COST_MODELS["phase-local"] = (
        ["F", "s"],
        lambda probs: [1.0, expectation(phi_round, probs)],
    )
    e177.COST_MODELS["phase-local+rows"] = (
        ["F", "s", "b"],
        lambda probs: [
            1.0,
            expectation(phi_round, probs),
            e177._rows(probs),
        ],
    )
    e177.COST_MODELS["phase-split"] = (
        ["F", "s_verify", "s_head"],
        lambda probs: [
            1.0,
            expectation(phi_verify, probs),
            expectation(phi_head, probs),
        ],
    )


def ranked_state():
    """Rebuild the E177 ranked state: receipts, survival tails, observations."""
    rows = e177.load_board()
    data = {}
    for label, prefix, cap, _note in e177.RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(prefix))
        vec = e177.per_prompt(row)
        data[label] = {
            "cap": cap,
            "vec": vec,
            "rec": e177.recover_rounds(vec),
        }
    surv = {}
    for name in e177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        t0 = 1.0 - a["nondraft"] / a["rounds"]
        _err, _lam, _kappa, s = e177.fit_survival(c["edl"], a["edl"], t0)
        surv[name] = s
    observations = []
    for label in ("A", "C"):
        cap = data[label]["cap"]
        for name in e177.ORDER:
            rec = data[label]["rec"][name]
            observations.append(
                (label, name, e177.width_probs(surv[name], cap),
                 rec["R"], rec["rounds"])
            )
    mu = {}
    for name in e177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        d_edl = a["edl"] - c["edl"]
        mu[name] = (a["abar"] - c["abar"]) / d_edl if d_edl > 1e-9 else 0.0
    return data, surv, observations, mu


def walk(data, surv, mu, name, model, target_cap):
    """E177's incremental walk from the paid cap-7 receipt to `target_cap`."""
    rec = data["A"]["rec"][name]
    cap, abar, R = data["A"]["cap"], rec["abar"], rec["R"]
    t = surv[name]
    while cap < target_cap:
        abar += t[cap] * mu[name]
        R += t[cap] * (e177.cost_at_width(model, cap + 2)
                       - e177.cost_at_width(model, cap + 1))
        cap += 1
    while cap > target_cap:
        abar -= t[cap - 1] * mu[name]
        R -= t[cap - 1] * (e177.cost_at_width(model, cap + 1)
                           - e177.cost_at_width(model, cap))
        cap -= 1
    n_rounds = e177.TOKENS / (1.0 + abar)
    return n_rounds * R


def priced_median(data, surv, mu, model, cap):
    raws = []
    for name in e177.ORDER:
        vec = data["A"]["vec"][name]
        total = walk(data, surv, mu, name, model, cap)
        raws.append(vec["serial"] / (total / e177.TOKENS + vec["prefill"]))
    return e177.published_median(raws)


def dump(path: pathlib.Path) -> None:
    data, surv, observations, mu = ranked_state()
    core = e177.fit_cost(observations, ["A", "C"], free_intercepts=[])
    a_only = e177.fit_cost(
        [o for o in observations if o[0] == "A"], ["A"], free_intercepts=[]
    )
    out = {}
    for key, model in core.items():
        out[key] = {
            "harness": "ranked",
            "names": model["names"],
            "beta_ms": [1000 * b for b in model["beta"]],
            "wrmse_ms": model["wrmse_ms"],
            "max_resid_ms": model["max_resid_ms"],
            "aic": model["aic"],
            "r_by_width_ms": {
                str(m): 1000 * e177.cost_at_width(model, m) for m in WIDTHS
            },
            "r1_anchor_err_ms": 1000 * e177.cost_at_width(model, 1) - 30.2519,
            "priced": {
                str(cap): priced_median(data, surv, mu, model, cap)
                for cap in (4, 5, 6, 8)
            },
        }
        if key in a_only:
            p4 = priced_median(data, surv, mu, a_only[key], 4)
            out[key]["a_only_wrmse_ms"] = a_only[key]["wrmse_ms"]
            out[key]["a_only_cap4_pred"] = p4
            out[key]["a_only_cap4_err_pct"] = 100 * (
                p4 / e177.CAP4_PUBLISHED - 1
            )
    path.write_text(
        json.dumps(
            {
                "harness": "ranked",
                "cap4_published": e177.CAP4_PUBLISHED,
                "cap7_published": e177.BEST_A,
                "crown": e177.CROWN,
                "r1_anchor_ms": 30.2519,
                "models": out,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", default="research/out/e182/report.json")
    parser.add_argument("--json", default="research/out/e182/ranked_fold.json")
    parser.add_argument("stage", nargs="?", default="all")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    trace = report["tables"]["trace"]
    band = report["tables"]["band"]

    phi_round = normalised(trace, "round_us")
    for row in band:
        row["band_total_us"] = band_sum(row)
    phi_verify = normalised(band, "band_total_us")
    phi_head = normalised(band, "d_submit2_us")

    print("shapes transferred from E182 (harness=local, normalised, unitless)")
    print("    M  phi_round  phi_verify  phi_head")
    for m in WIDTHS:
        print(
            "  %3d %10.4f %11.4f %9.4f"
            % (m, phi_round[m], phi_verify[m], phi_head[m])
        )

    install_shapes(phi_round, phi_verify, phi_head)
    dump(pathlib.Path(args.json))

    sys.argv = [sys.argv[0], args.stage]
    return e177.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
