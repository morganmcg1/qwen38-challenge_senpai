"""E146 R-A, F2 protocol: the self-selecting cohort at the bar, and the AUCs.

harness=ranked throughout.

WHAT CHANGED FROM MY FIRST CUT. F227 showed that `mtp_seconds_per_token_mean`
already contains the charged 512-token seed prefill, about 8.45 per cent of the
leg. The state is charged per drafting round and the prefill runs none, so the
prefill has nowhere legitimate to land in a per-drafting-round fit and is
absorbed into k. Removing it is not a refinement, it is a correction: it moves
`43925f29` from -179.1 to about -9 and reassigns its whole effect to a prefill
cut, which is where it belongs.

THE CIRCULARITY THAT MATTERS, AND HOW IT IS HANDLED. The cohort labels are cut
from k. Therefore:

  * AUC(k) is 1.0 by construction. It is reported as a construction, never as
    evidence. Campaign Rule 104 asks for a permuted null; for k the permuted
    null is the honest number and the observed value is meaningless.
  * AUC(residual) shares the same labels but is a DIFFERENT statistic - the
    magnitude of the component orthogonal to the basis, not the component along
    it. It is informative, but it is still measured against labels derived from
    the other coordinate of the same fit, so it is reported as in-sample.
  * The out-of-sample test is section 5. There the label for a row comes from
    that row's OWN zero-delta replicate contrast, and the statistic comes from a
    DIFFERENT anchor, the row's family consensus. No quantity is shared between
    the label and the feature. That is the number to trust.
"""

import json
import math
import os
import random
import sys

import e146_lib as L
import e146_family as F
import e146_pairs as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-cohort.json")

BAR = "684821ed"
OUR_ROW = "1db9d63e"
OUR_ANCHOR = "572b2cc4"
# The flush-fold warm receipts behind FINDING 220 and Campaign Rule 128.
F220_ROWS = ["749b4c7e", "3ba6ee9d", "9f9b4790", "7e5172fa", "3d75f016", "32dde61d"]
# Rows whose label is known from a public zero-delta declaration, not from k.
DECLARED_NULLS = ["106573b9", "3a18ff21", "b8e0f27c", "aff3b543", "64508884",
                  "e7770562"]
GAP_CUT = 400.0
PERMUTATIONS = 20000


def auc(pos, neg):
    """Mann-Whitney AUC: P(a positive scores above a negative), ties at 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def permuted_auc(values, labels, seed=146):
    """Null distribution of AUC under label exchange. Campaign Rule 104."""
    rng = random.Random(seed)
    pool = list(labels)
    out = []
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        pos = [v for v, l in zip(values, pool) if l]
        neg = [v for v, l in zip(values, pool) if not l]
        out.append(auc(pos, neg))
    out.sort()
    return out


def auc_with_null(values, labels, name):
    pos = [v for v, l in zip(values, labels) if l]
    neg = [v for v, l in zip(values, labels) if not l]
    observed = auc(pos, neg)
    null = permuted_auc(values, labels)
    p = sum(1 for v in null if v >= observed) / float(len(null))
    return {
        "feature": name, "auc": observed, "n_pos": len(pos), "n_neg": len(neg),
        "null_median": null[len(null) // 2],
        "null_p95": null[int(0.95 * len(null))],
        "p_value_one_sided": p,
    }


def split_on_gaps(items, cut):
    """1-D clustering by gap width. `items` is a sorted list of (value, tag)."""
    groups = [[items[0]]]
    for item in items[1:]:
        if item[0] - groups[-1][-1][0] > cut:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def cohort_of(anchor, rows):
    """Every scored row whose eight draft lengths are digit-identical.

    Digit-identical draft lengths prove that the same schedule ran on the same
    trajectory, so the anchor's round counts apply to every member and only time
    can differ. This is the F2 self-selecting rule: no hand picking.
    """
    key = anchor.draft_key()
    return [r for r in rows
            if r.draft_key() == key and r.id8 != anchor.id8 and r.has_prefill()]


# The frame matters and is easy to get wrong. The state is estimated on the
# decode-only response because that is where it lives. But the ranked score is
# computed on the WHOLE timed leg, prefill included, so the same absolute number
# of seconds is a smaller percentage of the scored leg, by 1/(1 + prefill share)
# ~ 0.922. Pricing a submission in the decode frame overstates the correction by
# about 8 per cent of itself.
step_pct_of = L.state_pct


def main():
    path, rows = L.load()
    by_id = {r.id8: r for r in rows}
    anchor = by_id[BAR]
    members = cohort_of(anchor, rows)
    print("board %s, %d scored rows" % (path, len(rows)))
    print("cohort anchored on %s: %d rows with digit-identical draft lengths"
          % (BAR, len(members)))

    fits = []
    for row in members:
        fit = L.fit_k(anchor, row, basis_row=anchor, field="decode")
        old = L.fit_k(anchor, row, basis_row=anchor, field="total")
        fits.append({"row": row, "fit": fit, "old": old})
    fits.sort(key=lambda e: e["fit"]["k_us_per_drafting_round"])

    print("\n=== 1. the cohort, prefill removed ===")
    print("%9s %9s %9s %8s %9s %9s  %-8s %-14s %s" % (
        "k us/dr", "k(total)", "decode8%", "resid", "prefill%", "score",
        "id", "solver", "created"))
    for entry in fits:
        f, o, r = entry["fit"], entry["old"], entry["row"]
        print("%9.1f %9.1f %+9.4f %8.4f %+9.4f %9.6f  %-8s %-14s %s" % (
            f["k_us_per_drafting_round"], o["k_us_per_drafting_round"],
            f["cand_mean8_pct"], f["residual_sd_pp"],
            f.get("prefill_mean8_pct", float("nan")), r.score or float("nan"),
            r.id8, r.solver[:14], r.created[11:19]))

    groups = split_on_gaps(
        [(e["fit"]["k_us_per_drafting_round"], e) for e in fits], GAP_CUT)
    print("\n=== 2. gap clustering at %.0f us, no hand picking ===" % GAP_CUT)
    names = ["A main", "B/C", "D out", "E", "F"]
    summary_groups = []
    for i, group in enumerate(groups):
        ks = [v for v, _ in group]
        resid = [e["fit"]["residual_sd_pp"] for _, e in group]
        dec8 = [e["fit"]["cand_mean8_pct"] for _, e in group]
        summary_groups.append({
            "name": names[i] if i < len(names) else "G%d" % i,
            "n": len(group), "k_mean": L.mean(ks), "k_sd": L.sd(ks),
            "residual_mean": L.mean(resid), "decode8_mean": L.mean(dec8),
            "ids": [e["row"].id8 for _, e in group],
        })
        print("%-8s n=%2d  k %+9.1f sd %7.1f  decode8 %+8.4f  resid %.4f"
              % (summary_groups[-1]["name"], len(group), L.mean(ks), L.sd(ks),
                 L.mean(dec8), L.mean(resid)))

    low = [e for e in fits if e["fit"]["k_us_per_drafting_round"] <= GAP_CUT]
    high = [e for e in fits if GAP_CUT < e["fit"]["k_us_per_drafting_round"] <= 1800.0]
    step_us = L.mean([e["fit"]["k_us_per_drafting_round"] for e in high]) - \
        L.mean([e["fit"]["k_us_per_drafting_round"] for e in low])
    step_se = math.sqrt(
        L.sd([e["fit"]["k_us_per_drafting_round"] for e in high]) ** 2 / len(high)
        + L.sd([e["fit"]["k_us_per_drafting_round"] for e in low]) ** 2 / len(low))
    p_high = len(high) / float(len(high) + len(low))
    print("\nSTATE STEP low -> high = %+.1f us per drafting round, se %.1f"
          % (step_us, step_se))
    print("  = %+.4f %% of the decode-only 8-prompt mean"
          % step_pct_of(anchor, anchor, step_us))
    print("P(high) = %d/%d = %.4f" % (len(high), len(high) + len(low), p_high))
    print("residual inflation high/low = %.2fx"
          % (L.mean([e["fit"]["residual_sd_pp"] for e in high])
             / L.mean([e["fit"]["residual_sd_pp"] for e in low])))

    print("\n=== 2b. the UNBIASED step: zero-delta rows only ===")
    print("Both cohort estimates are biased and in opposite directions. The")
    print("high group contains real mechanisms, whose own cost adds to k. The")
    print("low group contains the mid cluster. A row that DECLARES a zero-delta")
    print("resample of the anchor has no mechanism at all, so its whole k IS")
    print("the step, with no cohort construction in between.")
    print("The step is a band and it differs between schedule families, so the")
    print("estimate used to correct a crown-lineage row must come from the")
    print("crown lineage. Two independent sources qualify, both zero-delta:")
    print("  (a) a cohort row that declares a zero-delta resample of the bar;")
    print("  (b) a validated pure-nuisance pair whose target shares the bar's")
    print("      eight draft lengths, so the pair sits in the same family.")
    zd_values = []
    for e in fits:
        if (e["row"].id8 in DECLARED_NULLS
                and e["fit"]["k_us_per_drafting_round"] > GAP_CUT):
            zd_values.append((e["row"].id8, e["fit"]["k_us_per_drafting_round"],
                              "declared resample of the bar"))
    bar_key = anchor.draft_key()
    for pair in P.pairs(rows):
        k_pair = pair["fit"]["k_us_per_drafting_round"]
        if k_pair > GAP_CUT and pair["target"].draft_key() == bar_key:
            zd_values.append((pair["replicate"].id8, k_pair,
                              "pair against %s" % pair["target"].id8))
    seen_zd = {}
    for name, value, why in zd_values:
        seen_zd.setdefault(name, (value, why))
    step_zd = None
    if seen_zd:
        for name, (value, why) in sorted(seen_zd.items(), key=lambda t: t[1][0]):
            print("  %-9s k = %+8.1f us per drafting round  (%s)"
                  % (name, value, why))
        vals = [v for v, _ in seen_zd.values()]
        step_zd = L.mean(vals)
        print("  unbiased crown-family step = %+.1f us, sd %.1f, n=%d"
              % (step_zd, L.sd(vals), len(vals)))
        print("  F152/F172 measured 930.9 us by a completely different route.")
        print("  Difference %.1f us." % abs(step_zd - 930.9))

    print("\n=== 3. what the prefill correction moved ===")
    for wanted in ("43925f29", "e7770562"):
        entry = next((e for e in fits if e["row"].id8 == wanted), None)
        if entry is None:
            print("%-8s not on this board export" % wanted)
            continue
        f, o = entry["fit"], entry["old"]
        print("%-8s k %+9.1f (was %+9.1f)  decode8 %+8.4f  prefill %+8.4f  resid %.4f"
              % (wanted, f["k_us_per_drafting_round"], o["k_us_per_drafting_round"],
                 f["cand_mean8_pct"], f.get("prefill_mean8_pct", float("nan")),
                 f["residual_sd_pp"]))

    print("\n=== 4. AUC in the cohort, labels cut from k ===")
    labels = [e["fit"]["k_us_per_drafting_round"] > GAP_CUT for e in fits]
    feats = {
        "fit_residual_sd_pp": [e["fit"]["residual_sd_pp"] for e in fits],
        "k_us_per_drafting_round":
            [e["fit"]["k_us_per_drafting_round"] for e in fits],
        "NEGATIVE CONTROL prefill_mean8_pct":
            [e["fit"].get("prefill_mean8_pct", 0.0) for e in fits],
        "NEGATIVE CONTROL abs prefill_mean8_pct":
            [abs(e["fit"].get("prefill_mean8_pct", 0.0)) for e in fits],
    }
    in_sample = []
    print("%-38s %6s %6s %6s %8s" % ("feature", "AUC", "null50", "null95", "p"))
    for name, values in feats.items():
        res = auc_with_null(values, labels, name)
        in_sample.append(res)
        print("%-38s %6.3f %6.3f %6.3f %8.4f"
              % (name, res["auc"], res["null_median"], res["null_p95"],
                 res["p_value_one_sided"]))
    print("k is 1.000 by construction: the labels were cut from it.")
    print("The prefill control must sit at the null. If it does not, the")
    print("cohort is contaminated by a real prefill mechanism, not by state.")

    print("\n=== 5. OUT OF SAMPLE: label from the row's own replicate, ===")
    print("===    statistic from a different anchor, the family     ===")
    ps = P.pairs(rows)
    default_step = step_us
    labelled = []
    for pair in ps:
        k_pair = pair["fit"]["k_us_per_drafting_round"]
        if abs(k_pair) <= 150.0:
            continue
        # In a zero-delta pair exactly one member carries the state. The slower
        # member does. That label uses only the pair, never the family.
        if k_pair > 0:
            row, label = pair["replicate"], True
        else:
            row, label = pair["target"], True
        other = pair["target"] if k_pair > 0 else pair["replicate"]
        for candidate, lab in ((row, True), (other, False)):
            placed = F.classify_in_family(candidate, rows, default_step)
            if placed is None:
                continue
            fam_anchor = by_id.get(placed["clusters"][0][0][1])
            if fam_anchor is None or fam_anchor.id8 == candidate.id8:
                continue
            if not (fam_anchor.has_prefill() and candidate.has_prefill()):
                continue
            fit = L.fit_k(fam_anchor, candidate, basis_row=fam_anchor,
                          field="decode")
            labelled.append({
                "id": candidate.id8, "label": lab,
                "residual_sd_pp": fit["residual_sd_pp"],
                "k_us": fit["k_us_per_drafting_round"],
                "family_anchor": fam_anchor.id8,
            })
    seen = {}
    for item in labelled:
        seen.setdefault((item["id"], item["label"]), item)
    labelled = list(seen.values())
    ids_both = {i for i, _ in seen}
    conflict = [i for i in ids_both
                if (i, True) in seen and (i, False) in seen]
    for bad in conflict:
        labelled = [x for x in labelled if x["id"] != bad]
    print("labelled rows %d (%d dropped for conflicting labels)"
          % (len(labelled), len(conflict)))
    out_of_sample = []
    if len(labelled) >= 6:
        labs = [x["label"] for x in labelled]
        if any(labs) and not all(labs):
            for name, key in (("fit_residual_sd_pp", "residual_sd_pp"),
                              ("k_us_per_drafting_round", "k_us")):
                res = auc_with_null([x[key] for x in labelled], labs, name)
                out_of_sample.append(res)
                print("%-38s %6.3f %6.3f %6.3f %8.4f"
                      % (name, res["auc"], res["null_median"], res["null_p95"],
                         res["p_value_one_sided"]))
    else:
        print("too few independently labelled rows for an AUC")

    print("\n=== 6. the declared nulls, label from the public note ===")
    print("%-9s %9s %8s %9s  %s" % ("id", "k us/dr", "resid", "decode8%", "call"))
    declared = []
    for name in DECLARED_NULLS:
        entry = next((e for e in fits if e["row"].id8 == name), None)
        if entry is None:
            print("%-9s not in the bar cohort on this export" % name)
            continue
        f = entry["fit"]
        call = "HIGH" if f["k_us_per_drafting_round"] > GAP_CUT else "MAIN"
        declared.append({"id": name, "k": f["k_us_per_drafting_round"],
                         "residual_sd_pp": f["residual_sd_pp"],
                         "decode8_pct": f["cand_mean8_pct"], "call": call})
        print("%-9s %9.1f %8.4f %+9.4f  %s"
              % (name, f["k_us_per_drafting_round"], f["residual_sd_pp"],
                 f["cand_mean8_pct"], call))
    n_high = sum(1 for d in declared if d["call"] == "HIGH")
    if declared:
        print("%d of %d declared nulls drew the slow state" % (n_high, len(declared)))

    print("\n=== 7. FINDING 220: does it survive the state correction? ===")
    f220 = []
    for name in F220_ROWS:
        entry = next((e for e in fits if e["row"].id8 == name), None)
        if entry is None:
            print("%-9s not in the bar cohort on this export" % name)
            continue
        f = entry["fit"]
        k = f["k_us_per_drafting_round"]
        steps = 1 if k > GAP_CUT else 0
        corrected = f["cand_mean8_pct"] - steps * step_pct_of(anchor, entry["row"],
                                                              step_us)
        f220.append({"id": name, "k": k, "raw_decode8_pct": f["cand_mean8_pct"],
                     "steps": steps, "corrected_decode8_pct": corrected,
                     "residual_sd_pp": f["residual_sd_pp"]})
        print("%-9s k %+9.1f  raw %+8.4f  steps %d  corrected %+8.4f"
              % (name, k, f["cand_mean8_pct"], steps, corrected))
    survives = None
    if len(f220) >= 3:
        raw_sd = L.sd([e["raw_decode8_pct"] for e in f220])
        cor_sd = L.sd([e["corrected_decode8_pct"] for e in f220])
        cor_mean = L.mean([e["corrected_decode8_pct"] for e in f220])
        print("\nspread across the %d receipts of ONE mechanism:" % len(f220))
        print("  raw       sd %.4f pp, range %.4f"
              % (raw_sd, max(e["raw_decode8_pct"] for e in f220)
                 - min(e["raw_decode8_pct"] for e in f220)))
        print("  corrected sd %.4f pp, range %.4f, mean %+.4f"
              % (cor_sd, max(e["corrected_decode8_pct"] for e in f220)
                 - min(e["corrected_decode8_pct"] for e in f220), cor_mean))
        # F220 claimed the mechanism is base dependent. That claim needs the
        # receipts to disagree AFTER the nuisance is removed.
        survives = 0.0 if cor_sd < 0.30 else 1.0
        print("\ne146_f220_survives_state_correction = %.1f" % survives)
        print("  F220 read the spread of these receipts as base dependence.")
        print("  After one state step is removed the receipts agree to %.4f pp"
              % cor_sd)
        print("  and the mechanism reads %+.4f %% on the decode leg." % cor_mean)

    print("\n=== 8. state-corrected values for the named rows ===")
    wanted = [OUR_ROW, "9f9b4790", "c24f1755", "c54de844", "452b0055",
              "e718a6d3", "9aea1921", "02742bf0", "1760479a", "3ba6ee9d",
              "749b4c7e", "b8e0f27c", "43925f29", "106573b9", "08b67f12",
              "a323a8ed", "890594e9", "aff3b543", "64508884", "e7770562"]
    corrected_rows = []
    use_step = step_zd or step_us
    print("step %.1f us per drafting round, the crown-family zero-delta value."
          % use_step)
    print("`corr total` is the frame a submission is priced in; `corr decode`")
    print("is the frame the state is estimated in.")
    print("%-9s %9s %6s %10s %10s %10s %9s" % (
        "id", "k us/dr", "steps", "raw d8%", "corr decode", "corr total",
        "prefill%"))
    for name in wanted:
        entry = next((e for e in fits if e["row"].id8 == name), None)
        if entry is None:
            continue
        f = entry["fit"]
        t = L.fit_k(anchor, entry["row"], basis_row=anchor, field="total")
        k = f["k_us_per_drafting_round"]
        steps = int(round(k / use_step)) if k > GAP_CUT else 0
        cd = f["cand_mean8_pct"] - steps * step_pct_of(
            anchor, entry["row"], use_step, "decode")
        ct = t["cand_mean8_pct"] - steps * step_pct_of(
            anchor, entry["row"], use_step, "total")
        corrected_rows.append({
            "id": name, "k_us": k, "raw_decode8_pct": f["cand_mean8_pct"],
            "raw_total8_pct": t["cand_mean8_pct"], "steps": steps,
            "corrected_decode8_pct": cd, "corrected_total8_pct": ct,
            "prefill_mean8_pct": f.get("prefill_mean8_pct"),
            "residual_sd_pp": f["residual_sd_pp"],
        })
        print("%-9s %9.1f %6d %+10.4f %+10.4f %+10.4f %+9.4f"
              % (name, k, steps, f["cand_mean8_pct"], cd, ct,
                 f.get("prefill_mean8_pct", float("nan"))))

    print("\n=== 9. our own row, both frames, all three step estimates ===")
    print("The SCORE is computed on the whole timed leg, so the headline must")
    print("be quoted in the total-leg frame. The decode frame is where the")
    print("state is estimated, not where a submission is priced.")
    ours = by_id[OUR_ROW]
    own = by_id[OUR_ANCHOR]
    own_placed = F.classify_in_family(ours, rows, step_us)
    anchor_placed = F.classify_in_family(own, rows, step_us)

    step_choices = [("cohort high-low", step_us)]
    if step_zd:
        step_choices.append(("zero-delta rows", step_zd))
    step_choices.append(("F152/F172 independent", 930.9))

    frames = []
    for frame_name, a in (("bar %s" % BAR, anchor),
                          ("assigned %s" % OUR_ANCHOR, own)):
        dec = L.fit_k(a, ours, basis_row=a, field="decode")
        tot = L.fit_k(a, ours, basis_row=a, field="total")
        print("\n%s" % frame_name)
        print("  raw total leg  %+8.4f %%" % tot["cand_mean8_pct"])
        print("  raw decode     %+8.4f %%   k %+.1f"
              % (dec["cand_mean8_pct"], dec["k_us_per_drafting_round"]))
        print("  %-24s %12s %12s" % ("step estimate", "corr total", "corr decode"))
        entry = {"frame": frame_name, "anchor": a.id8,
                 "raw_total_pct": tot["cand_mean8_pct"],
                 "raw_decode_pct": dec["cand_mean8_pct"],
                 "k_us": dec["k_us_per_drafting_round"], "corrected": {}}
        for label, value in step_choices:
            ct = tot["cand_mean8_pct"] - step_pct_of(a, ours, value, "total")
            cd = dec["cand_mean8_pct"] - step_pct_of(a, ours, value, "decode")
            entry["corrected"][label] = {"total_pct": ct, "decode_pct": cd,
                                         "step_us": value}
            print("  %-24s %+12.4f %+12.4f" % (label, ct, cd))
        frames.append(entry)
    print("\nadvisor F2 quotes the bar frame decode leg: +2.2109 -> +0.4248")
    if anchor_placed:
        print("anchor %s places at %.2f steps above its own family flat"
              % (OUR_ANCHOR, anchor_placed["steps_above_low"]))
    if own_placed:
        print("row    %s places at %.2f steps above its own family flat"
              % (OUR_ROW, own_placed["steps_above_low"]))

    headline_step = step_zd or step_us
    bar_frame = frames[0]
    own_frame = frames[1]
    headline = own_frame["corrected"][
        "zero-delta rows" if step_zd else "cohort high-low"]["total_pct"]
    print("\ne146_1db9d63e_mode_corrected_candidate_mean_pct = %+.4f %%"
          % headline)
    print("  anchored on %s, total timed leg, one state step of %.1f us."
          % (OUR_ANCHOR, headline_step))
    print("  against the brief's raw %+.4f %%." % own_frame["raw_total_pct"])

    payload = {
        "harness": "ranked",
        "board_path": path,
        "board_rows_scored": len(rows),
        "bar": BAR,
        "cohort_size": len(members),
        "gap_cut_us": GAP_CUT,
        "groups": summary_groups,
        "step_us_per_drafting_round": step_us,
        "step_se_us": step_se,
        "step_us_zero_delta_only": step_zd,
        "step_us_f152_independent": 930.9,
        "step_pct_of_decode_mean8": step_pct_of(anchor, anchor, step_us),
        "step_pct_of_total_mean8": step_pct_of(anchor, anchor, step_us, "total"),
        "p_high": p_high,
        "n_high": len(high),
        "n_low": len(low),
        "residual_inflation": (L.mean([e["fit"]["residual_sd_pp"] for e in high])
                               / L.mean([e["fit"]["residual_sd_pp"] for e in low])),
        "auc_in_sample": in_sample,
        "auc_out_of_sample": out_of_sample,
        "declared_nulls": declared,
        "f220": f220,
        "e146_f220_survives_state_correction": survives,
        "corrected_rows": corrected_rows,
        "our_row": {
            "frames": frames,
            "headline_step_us": headline_step,
            "e146_1db9d63e_mode_corrected_candidate_mean_pct": headline,
            "raw_candidate_mean8_pct": own_frame["raw_total_pct"],
            "bar_frame_decode_corrected_pct":
                bar_frame["corrected"]["zero-delta rows" if step_zd
                                       else "cohort high-low"]["decode_pct"],
            "anchor_steps_above_family_flat":
                anchor_placed["steps_above_low"] if anchor_placed else None,
            "row_steps_above_family_flat":
                own_placed["steps_above_low"] if own_placed else None,
        },
        "rows": [{
            "id": e["row"].id8, "solver": e["row"].solver,
            "created": e["row"].created, "score": e["row"].score,
            "k_us_decode": e["fit"]["k_us_per_drafting_round"],
            "k_us_total": e["old"]["k_us_per_drafting_round"],
            "decode8_pct": e["fit"]["cand_mean8_pct"],
            "total8_pct": e["old"]["cand_mean8_pct"],
            "residual_sd_pp": e["fit"]["residual_sd_pp"],
            "prefill_mean8_pct": e["fit"].get("prefill_mean8_pct"),
        } for e in fits],
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
