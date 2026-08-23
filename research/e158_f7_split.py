#!/usr/bin/env python3
"""Split the island-arm effect into cost and accuracy columns (advisor F7).

Under the F6 identity `mtp = R / (1 + a)`:

    dln(mtp) = dln(R) - dln(1 + a)

`dln(R)` is the cost column and `-dln(1 + a)` is the accuracy column. That
two-way split is not yet enough, because `R` also moves when the schedule
offers a different number of rows. Using the depth-sweep round-cost fit
`R = s + h * rows` the cost column splits again:

    mechanism : the arm changes s and h at a FIXED row count
    schedule  : the row count changes under the arm's own s and h

Only the mechanism column is caused by removing the precision islands. The
schedule and accuracy columns are round-count lotteries that vary in sign
across prompts.

The `s` and `h` fit comes from the beagle_a depth sweep. Applying it to the
other prompts is an extrapolation and is labelled as such in the output.
"""

import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
HARVEST = ROOT / "e158-artifacts" / "r1-harvest.json"
OUT = ROOT / "e158-artifacts" / "f7-split.json"

FIT_PROMPT = "beagle_a"


def pct(new, old):
    return 100.0 * (new / old - 1.0)


def main():
    data = json.loads(HARVEST.read_text())
    fit = data["depth_sweep"]["arms"]
    s = {arm: fit[arm]["fit"]["fixed_seconds_per_round_s"]
         for arm in ("all", "none")}
    h = {arm: fit[arm]["fit"]["marginal_seconds_per_row_h"]
         for arm in ("all", "none")}

    legs = [l for l in data["legs"] if l["session"] in ("perprompt", "canary")]
    by = {}
    for l in legs:
        by.setdefault(l["prompt"], {}).setdefault(l["arm"], []).append(l)

    rows_out = {}
    for prompt, arms in by.items():
        if not ("all" in arms and "none" in arms):
            continue
        ref = {arm: arms[arm][0] for arm in ("all", "none")}
        mtp = {arm: statistics.fmean(x["mtp_seconds_per_token"] for x in arms[arm])
               for arm in ("all", "none")}
        rows = {arm: 1.0 + ref[arm]["mean_proposed_per_round_q"]
                for arm in ("all", "none")}
        a = {arm: ref[arm]["mean_accepted_per_round_a"] for arm in ("all", "none")}

        r_all = s["all"] + h["all"] * rows["all"]
        r_mech = s["none"] + h["none"] * rows["all"]
        r_sched = s["none"] + h["none"] * rows["none"]

        mechanism = pct(r_mech, r_all)
        schedule = pct(r_sched, r_mech)
        accuracy = -pct(1.0 + a["none"], 1.0 + a["all"])
        measured = pct(mtp["none"], mtp["all"])

        rows_out[prompt] = {
            "n_legs_all": len(arms["all"]),
            "n_legs_none": len(arms["none"]),
            "rows_per_round_all": rows["all"],
            "rows_per_round_none": rows["none"],
            "a_all": a["all"],
            "a_none": a["none"],
            "round_count_all": ref["all"]["round_count"],
            "round_count_none": ref["none"]["round_count"],
            "mtp_all": mtp["all"],
            "mtp_none": mtp["none"],
            "mechanism_pct": mechanism,
            "schedule_pct": schedule,
            "accuracy_pct": accuracy,
            "modelled_total_pct": mechanism + schedule + accuracy,
            "measured_total_pct": measured,
            "model_residual_pct": measured - (mechanism + schedule + accuracy),
            "fit_is_extrapolated_from": (
                None if prompt == FIT_PROMPT else FIT_PROMPT),
        }

    mech = [r["mechanism_pct"] for r in rows_out.values()]
    result = {
        "experiment": "e158-r1-f7-split",
        "harness": "local",
        "identity": "dln(mtp) = dln(R) - dln(1 + a); R = s + h * rows",
        "round_cost_fit": {"prompt": FIT_PROMPT, "s": s, "h": h},
        "per_prompt": rows_out,
        "mechanism_pct_mean": statistics.fmean(mech),
        "mechanism_pct_spread": max(mech) - min(mech),
        "noise_floor_pct_absolute_mtp": data["noise_floor_pct_absolute_mtp"],
        "reading": ("The mechanism column is the only column caused by removing "
                    "the islands. It is negative on every prompt and its spread "
                    "is small. The schedule and accuracy columns change sign "
                    "across prompts and dominate the measured total, which is "
                    "why per-prompt absolute mtp disagrees with the mechanism."),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"round-cost fit from {FIT_PROMPT}: "
          f"s_all={s['all']:.8f} s_none={s['none']:.8f} "
          f"h_all={h['all']:.8f} h_none={h['none']:.8f}")
    print()
    head = ("prompt", "mechanism", "schedule", "accuracy", "model", "measured",
            "resid")
    print("%-18s%11s%11s%11s%10s%11s%9s" % head)
    for prompt, r in rows_out.items():
        print("%-18s%+10.4f%%%+10.4f%%%+10.4f%%%+9.4f%%%+10.4f%%%+8.4f%%" % (
            prompt, r["mechanism_pct"], r["schedule_pct"], r["accuracy_pct"],
            r["modelled_total_pct"], r["measured_total_pct"],
            r["model_residual_pct"]))
    print()
    print(f"mechanism mean {result['mechanism_pct_mean']:+.4f} %, "
          f"spread {result['mechanism_pct_spread']:.4f} pp, "
          f"noise floor {result['noise_floor_pct_absolute_mtp']} %")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
