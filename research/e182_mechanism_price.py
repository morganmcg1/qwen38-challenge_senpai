#!/usr/bin/env python3
"""E182: price the one mechanism the phase table names.

    usage: research/e182_mechanism_price.py [--report research/out/e182/report.json]

The measured local phase table says the round's width dependence is a STEP in
`groups(M)`, the replica QMV input-group count, and that every one of the four
GPU bands steps together at the two group crossings (M 5->6 and M 8->9). A
group is one extra streaming pass over the 4-bit backbone weights, so the
mechanism the table names is:

    make the routed QMV cells verify M rows in ONE weight pass at M >= 6
    instead of ceil(M / IPG) passes.

This script prices the ceiling of that mechanism on the ranked board. The
counterfactual removes ALL super-linear cost above M = 5: above the first group
crossing the round grows at the same slope it had inside group 1.

    R'(M) = R(M)                              for M <= 5
    R'(M) = R(5) + (M - 5) * (R(5) - R(4))    for M >= 6

That is an upper bound, not a prediction. It assumes the extra passes are free
and that nothing else about the wider round costs anything super-linear.

Both ranked laws are priced, because they disagree about how much step is
actually present on the M5 host:

  quadratic          FINDING 456, the best-fitting ranked law, no group
                     structure at all.
  phase-local+rows   the measured local shape refitted on ranked data, which
                     does carry the group structure.

`harness=ranked` for every published number here. `harness=local` only for the
shape that names the mechanism.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e182_ranked_fold as fold  # noqa: E402

LAWS = ("quadratic", "phase-local+rows", "phase-split", "phase-local")


class FlatAbove5:
    """A fitted law with all super-linear cost above width 5 removed."""

    def __init__(self, model):
        self.base = {m: e177.cost_at_width(model, m) for m in range(1, 10)}
        slope = self.base[5] - self.base[4]
        self.table = dict(self.base)
        for m in range(6, 10):
            self.table[m] = self.base[5] + (m - 5) * slope

    def at(self, m: int) -> float:
        return self.table[m]

    def saving(self, probs) -> float:
        """Expected per-round saving under a width mixture."""
        return sum(
            q * (self.base[1 + d] - self.table[1 + d])
            for d, q in enumerate(probs)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="research/out/e182/report.json")
    parser.add_argument("--out", default="research/out/e182/mechanism.json")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    trace = report["tables"]["trace"]
    band = report["tables"]["band"]
    for row in band:
        row["band_total_us"] = fold.band_sum(row)

    fold.install_shapes(
        fold.normalised(trace, "round_us"),
        fold.normalised(band, "band_total_us"),
        fold.normalised(band, "d_submit2_us"),
    )

    data, surv, observations, mu = fold.ranked_state()
    fits = e177.fit_cost(observations, ["A", "C"], free_intercepts=[])

    out = {"harness": "ranked", "laws": {}}
    print("mechanism ceiling: one weight pass for every width (harness=ranked)")
    print(
        "  %-18s %9s %9s %9s %9s %9s"
        % ("law", "R6 saved", "R7 saved", "R8 saved", "cap7 new", "gain %")
    )
    for key in LAWS:
        model = fits[key]
        law = FlatAbove5(model)
        raws, raws_base = [], []
        rounds_at_wide = 0.0
        rounds_total = 0.0
        for name in e177.ORDER:
            vec = data["A"]["vec"][name]
            rec = data["A"]["rec"][name]
            probs = e177.width_probs(surv[name], 7)
            # Acceptance is unchanged: the mechanism changes only how the same
            # rows are dispatched, so the round count stays at the paid value.
            new_r = rec["R"] - law.saving(probs)
            raws.append(
                vec["serial"]
                / (rec["rounds"] * new_r / e177.TOKENS + vec["prefill"])
            )
            raws_base.append(
                vec["serial"] / (rec["rounds"] * rec["R"] / e177.TOKENS
                                 + vec["prefill"])
            )
            wide = sum(q for d, q in enumerate(probs) if 1 + d >= 6)
            rounds_at_wide += wide * rec["rounds"]
            rounds_total += rec["rounds"]
        new = e177.published_median(raws)
        base = e177.published_median(raws_base)
        out["laws"][key] = {
            "r_by_width_ms": {str(m): 1000 * law.base[m] for m in range(1, 10)},
            "r_flat_by_width_ms": {
                str(m): 1000 * law.at(m) for m in range(1, 10)
            },
            "saved_ms": {
                str(m): 1000 * (law.base[m] - law.at(m)) for m in range(6, 10)
            },
            "published_base": base,
            "published_ceiling": new,
            "gain_pct": 100 * (new / base - 1),
            "wide_round_share": rounds_at_wide / rounds_total,
        }
        print(
            "  %-18s %9.2f %9.2f %9.2f %9.6f %9.3f"
            % (
                key,
                1000 * (law.base[6] - law.at(6)),
                1000 * (law.base[7] - law.at(7)),
                1000 * (law.base[8] - law.at(8)),
                new,
                100 * (new / base - 1),
            )
        )

    share = out["laws"]["quadratic"]["wide_round_share"]
    out["wide_round_share"] = share
    out["crown"] = e177.CROWN
    out["paid_cap7"] = e177.BEST_A
    print(
        "\n  rounds verifying 6 or more rows, receipt A round-weighted: %.1f %%"
        % (100 * share)
    )
    print("  paid cap-7 %.8f, crown %.8f" % (e177.BEST_A, e177.CROWN))
    for key in LAWS:
        block = out["laws"][key]
        print(
            "  %-18s ceiling %.8f  %+0.3f %%  crown cleared: %s"
            % (
                key,
                block["published_ceiling"],
                block["gain_pct"],
                block["published_ceiling"] > e177.CROWN,
            )
        )

    path = pathlib.Path(args.out)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
