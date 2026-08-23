"""E146: print every overflow table as Markdown for the PR comment.

The structured result summary is length limited, so the full tables live in one
PR comment. This script writes them from the same JSON artifacts the W&B run
reads, so the comment cannot drift from the run.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as handle:
        return json.load(handle)


def fmt(value, spec):
    return "n/a" if value is None else spec % value


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    cohort = load("e146-cohort.json")
    power = load("e146-power.json")
    modes = load("e146-modes.json")
    classify = load("e146-classify.json")
    legs = load("e146-rc-legs.json")
    l305 = load("e146-ledger305.json")
    ranked, local = power["ranked"], power["local"]

    print("## T1. Bar cohort at `684821ed`, prefill removed, %d rows"
          % cohort["cohort_size"])
    print()
    group_of = {}
    for g in cohort["groups"]:
        for row_id in g["ids"]:
            group_of[row_id] = g["name"]
    rows = sorted(cohort["rows"], key=lambda r: r["k_us_decode"])
    print(table(["id", "solver", "group", "k us/dr decode", "k us/dr total",
                 "decode8 %", "prefill8 %", "resid pp"],
                [[r["id"], r["solver"], group_of.get(r["id"], "?"),
                  "%.1f" % r["k_us_decode"], "%.1f" % r["k_us_total"],
                  "%+.4f" % r["decode8_pct"], "%+.4f" % r["prefill_mean8_pct"],
                  "%.4f" % r["residual_sd_pp"]] for r in rows]))
    print()
    print(table(["group", "n", "k mean us/dr", "k sd", "decode8 mean %",
                 "resid mean pp"],
                [[g["name"], g["n"], "%+.1f" % g["k_mean"], "%.1f" % g["k_sd"],
                  "%+.4f" % g["decode8_mean"], "%.4f" % g["residual_mean"]]
                 for g in cohort["groups"]]))
    print()

    print("## T2. State-corrected rows, both frames, step %.1f us/dr"
          % cohort["step_us_zero_delta_only"])
    print()
    crows = sorted(cohort["corrected_rows"], key=lambda r: r["corrected_total8_pct"])
    print(table(["id", "steps", "raw decode %", "corr decode %", "raw total %",
                 "corr total %"],
                [[r["id"], "%.2f" % r["steps"], "%+.4f" % r["raw_decode8_pct"],
                  "%+.4f" % r["corrected_decode8_pct"],
                  "%+.4f" % r["raw_total8_pct"],
                  "%+.4f" % r["corrected_total8_pct"]] for r in crows]))
    print()

    print("## T3. Headline step sensitivity, anchor `572b2cc4` to `1db9d63e`")
    print()
    print(table(["step source", "step us/dr", "corrected decode %",
                 "corrected total leg %"],
                [["raw, no correction", "0.0", "%+.4f" % l305["raw_decode_pct"],
                  "%+.4f" % l305["raw_total_pct"]]]
                + [[e["step_source"], "%.1f" % e["step_us"],
                    "%+.4f" % e["corrected_decode_pct"],
                    "%+.4f" % e["corrected_total_pct"]] for e in l305["steps"]]))
    print()
    print("Ledger 305 published `%+.4f` in the decode frame; the E146 replay of"
          % l305["advisor_decode_pct"])
    print("that same step gives `%+.4f`, an agreement of `%.4f` pp."
          % (l305["e146_replay_of_advisor_step_decode_pct"],
             l305["e146_advisor_agreement_pp"]))
    print()

    print("## T4. Mode separation, permuted null, 20000 permutations")
    print()
    print(table(["scheme", "feature", "AUC", "one-sided p", "null p95",
                 "n high", "n main"],
                [[scheme, a["feature"], "%.4f" % a["auc"],
                  "%.5f" % a["p_value_one_sided"], "%.4f" % a["null_p95"],
                  a["n_pos"], a["n_neg"]]
                 for scheme, entries in (("in sample", cohort["auc_in_sample"]),
                                         ("out of sample",
                                          cohort["auc_out_of_sample"]))
                 for a in entries]))
    print()

    print("## T5. Published-leg precision on the %d pure-nuisance pairs"
          % ranked["pairs"])
    print()
    print(table(["published field", "n", "sd over all pairs pp",
                 "sd over flat pairs pp"],
                [[name, v["n"], "%.4f" % v["sd_pct"], "%.4f" % v["flat_sd_pct"]]
                 for name, v in sorted(
                     ranked["published_leg_precision"].items())]))
    print()

    print("## T6. Corrected MDE table, harness=ranked")
    print()
    print(table(["design", "sd pp", "2-sigma MDE at n=1 pp", "note"],
                [[d["design"], "%.4f" % d["sd_pct"], "%.4f" % d["mde_n1_pct"],
                  d["note"]] for d in ranked["designs"]]))
    print()
    print(table(["true effect pp", "single contrasts needed, paired replicate",
                 "needed, mode classified"],
                [[k, v, max(1, int(round(v * (
                    ranked["e146_mde_2sigma_pct_mode_classified"]
                    / ranked["e146_mde_2sigma_pct_paired_replicate"]) ** 2)))]
                 for k, v in sorted(ranked["contrasts_needed"].items(),
                                    key=lambda kv: float(kv[0]))]))
    print()

    print("## T7. R-C local fixed-binary legs, `beagle_a`, 512 tokens")
    print()
    print(table(["leg", "kind", "s/token", "blocks only s", "seed prefill s",
                 "rounds", "draft len", "accept", "entry C", "exit C",
                 "gate wait s", "matched"],
                [[l["label"], l["kind"],
                  "%.7f" % l["metrics"]["parent_measured_seconds_per_token"],
                  "%.4f" % l["metrics"]["blocks_only_seconds"],
                  "%.4f" % l["metrics"]["seed_prefill_seconds"],
                  l["metrics"]["round_count"],
                  "%.4f" % l["metrics"].get("effective_mean_draft_len", 0.0),
                  "%.4f" % l["metrics"].get("accepted_draft_rate", 0.0),
                  "%.2f" % l["gpu_temp_entry_c"], "%.2f" % l["gpu_temp_exit_c"],
                  int(l["cool_gate_wait_seconds"]),
                  l["metrics"]["all_tokens_matched"]] for l in legs["legs"]]))
    print()
    print(table(["n legs per arm", "2-sigma MDE all legs %",
                 "2-sigma MDE steady state %", "80 % power MDE %"],
                [[n, "%.4f" % local["mde_2sigma_by_n"][n],
                  "%.4f" % local["mde_2sigma_steady_by_n"][n],
                  "%.4f" % local["mde_by_n"][n]]
                 for n in ("1", "2", "4", "8")]))
    print()

    print("## T8. Per-prompt exposure of the state, from the anchor")
    print()
    print(table(["prompt", "drafting share of rounds", "high mean %",
                 "flat mean %"],
                [[p, "%.4f" % v["drafting_share"], "%+.4f" % v["high_mean_pct"],
                  "%+.4f" % v["flat_mean_pct"]]
                 for p, v in modes["per_prompt"].items()]))
    print()

    print("## T9. Findings at risk from the run-mode correction")
    print()
    print(table(["row", "anchor", "finding", "k us/dr", "steps", "call",
                 "raw total %", "corr total %"],
                [[e["row"], e["anchor"], e["finding"],
                  fmt(e["k_us_per_drafting_round"], "%+.1f"),
                  fmt(e["steps_above_family_flat"], "%.2f"), e["mode_call"],
                  fmt(e["raw_total_mean8_pct"], "%+.4f"),
                  fmt(e["corrected_total_mean8_pct"], "%+.4f")]
                 for e in classify["e146_findings_at_risk"]]))

    offschedule = load("e146-offschedule.json")
    if offschedule:
        print()
        print("## T10. Off-schedule anchor inversion panel")
        print()
        print(table(["anchor", "role", "AUC", "null p95", "p one-sided",
                     "n high", "n main", "calibrated call"],
                    [[e["anchor"], e["role"], "%.4f" % e["auc"],
                      "%.4f" % e["null_p95"], "%.5f" % e["p_value_one_sided"],
                      e["n_high"], e["n_main"], e.get("calibrated_call", "-")]
                     for e in offschedule["anchor_inversion_panel"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
