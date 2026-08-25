#!/usr/bin/env python3
"""E209 cross-check: price the depth-policy family on the RANKED instrument.

    python3 research/e209_ranked.py [--json OUT]

WHY THIS EXISTS. `e209_desk.py` measures the controller family on the E168 `p7`
corpus, harness=local. That corpus is where FINDING 540 measured its +2.589%
ceiling. This script asks the separate question the score actually turns on:
does the SAME family move the published median on the ranked instrument?

The instrument is the FINDING 520 survival-pinned latent-q model, imported from
`e201_online_cap` / `e203_stage0b` without modification, exactly as E207 used
it. Everything here is labelled harness=ranked. The two harnesses are never
mixed, and no local millisecond is carried into a ranked number.

WHAT IT REPORTS.

  * the published median of every constant depth, against receipt A and the
    shipped cap-7 and cap-8 rules;
  * the per-ranked-prompt best fixed depth, which is the ranked analogue of
    FINDING 540's per-prompt best-fixed ceiling;
  * the per-prompt depth that each ranked prompt prefers, which is what shows
    whether the local prose corpus and the ranked population want the same
    level at all.

The per-prompt-best-fixed row is an ORACLE: it chooses each prompt's depth in
hindsight. It is an upper bound for any causal per-prompt controller on this
instrument, in the same sense as FINDING 540 (a).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e197_refit as E  # noqa: E402
import e201_online_cap as O  # noqa: E402
import e203_stage0b as S  # noqa: E402


def build():
    inst = O.build_instrument()
    laws, _ratio9 = O.cost_laws(inst["fit"])
    return inst, laws, O.cap_grid(inst, laws), laws["smooth"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",
                        default="research/e209-artifacts/ranked-crosscheck.json")
    args = parser.parse_args()

    inst, _laws, grid, law = build()

    # Per-prompt raw speedup of every constant depth.
    const_raw = {
        name: {d: S.assemble(inst, law, name, S.items_const(inst, name, d))
               ["raw"] for d in range(S.MAXD + 1)}
        for name in S.ORDER}
    shipped_raw = {name: grid["smooth"][name][8]["raw"] for name in S.ORDER}
    shipped_cap7_raw = {name: grid["smooth"][name][7]["raw"]
                        for name in S.ORDER}

    const_median = {d: e177.published_median(
        [const_raw[name][d] for name in S.ORDER]) for d in range(S.MAXD + 1)}
    ship_cap7 = e177.published_median(
        [shipped_cap7_raw[name] for name in S.ORDER])
    ship_cap8 = e177.published_median([shipped_raw[name] for name in S.ORDER])

    # Ranked analogue of FINDING 540 (a): each prompt takes its own best fixed
    # depth, chosen in hindsight.
    best_per_prompt = {name: max(const_raw[name], key=const_raw[name].get)
                       for name in S.ORDER}
    oracle_fixed = e177.published_median(
        [const_raw[name][best_per_prompt[name]] for name in S.ORDER])

    lines = ["E209 ranked cross-check (harness=ranked, FINDING 520 instrument)",
             "receipt A %.6f   crown %.6f" % (E.BEST_A, E.CROWN), ""]
    lines.append("published median by CONSTANT depth")
    for d in range(S.MAXD + 1):
        lines.append("  d=%d  %.6f  (%+7.2f%% vs A, %+7.2f%% vs shipped cap8)"
                     % (d, const_median[d],
                        100.0 * (const_median[d] - E.BEST_A) / E.BEST_A,
                        100.0 * (const_median[d] - ship_cap8) / ship_cap8))
    lines.append("")
    lines.append("shipped rule cap7 %.6f   cap8 %.6f" % (ship_cap7, ship_cap8))
    lines.append("per-prompt best fixed depth (ORACLE) %.6f  "
                 "(%+.2f%% vs shipped cap8)"
                 % (oracle_fixed,
                    100.0 * (oracle_fixed - ship_cap8) / ship_cap8))
    lines.append("")
    lines.append("per-ranked-prompt preferred fixed depth")
    for name in S.ORDER:
        lines.append("  %-22s d*=%d  raw %.4f   shipped cap8 raw %.4f"
                     % (name, best_per_prompt[name],
                        const_raw[name][best_per_prompt[name]],
                        shipped_raw[name]))
    lines.append("")
    lines.append("local prose corpus preferred fixed depth: 2 to 3 "
                 "(E209 desk, harness=local)")
    lines.append("ranked instrument preferred fixed depth : %d "
                 "(median of d* above)"
                 % statistics.median(sorted(best_per_prompt.values())))

    # With eight prompts the published median is the mean of the two middle
    # raws, so only two prompts set it. Naming them shows which acceptance
    # regime a candidate must improve to move the score at all.
    ranking = sorted(S.ORDER, key=lambda name: shipped_raw[name])
    setters = ranking[3:5]
    lines.append("")
    lines.append("WHICH PROMPTS SET THE PUBLISHED MEDIAN")
    lines.append("  shipped cap8 raws, sorted: %s"
                 % "  ".join("%s %.4f" % (n, shipped_raw[n]) for n in ranking))
    lines.append("  the median is the mean of the two middle raws: %s and %s"
                 % (setters[0], setters[1]))
    for name in setters:
        lines.append("    %-22s shipped %.4f  best fixed d=%d %.4f  (%+.2f%%)"
                     % (name, shipped_raw[name], best_per_prompt[name],
                        const_raw[name][best_per_prompt[name]],
                        100.0 * (const_raw[name][best_per_prompt[name]]
                                 - shipped_raw[name]) / shipped_raw[name]))
    lines.append("  both median-setting prompts prefer deep drafting "
                 "(d*=%s), the regime the local prose corpus never visits."
                 % ",".join(str(best_per_prompt[n]) for n in setters))
    payload_setters = {
        "median_setting_prompts": setters,
        "median_setting_detail": {
            name: {"shipped_cap8_raw": shipped_raw[name],
                   "best_fixed_depth": best_per_prompt[name],
                   "best_fixed_raw": const_raw[name][best_per_prompt[name]],
                   "best_fixed_vs_shipped_pct":
                       100.0 * (const_raw[name][best_per_prompt[name]]
                                - shipped_raw[name]) / shipped_raw[name]}
            for name in setters}}

    text = "\n".join(lines)
    print(text)

    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload_setters, **{
        "harness": "ranked",
        "receipt_A": E.BEST_A, "crown": E.CROWN,
        "const_published_median": const_median,
        "shipped_cap7": ship_cap7, "shipped_cap8": ship_cap8,
        "oracle_per_prompt_fixed": oracle_fixed,
        "oracle_vs_shipped_cap8_pct":
            100.0 * (oracle_fixed - ship_cap8) / ship_cap8,
        "per_prompt_best_depth": best_per_prompt,
        "per_prompt_const_raw": const_raw,
        "per_prompt_shipped_cap8_raw": shipped_raw,
    }), indent=1, sort_keys=True) + "\n")
    out.with_suffix(".txt").write_text(text + "\n")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
