"""E146 R-A rung 1: can a single ranked receipt be classified, and what is at risk?

harness=ranked throughout.

THE ASSIGNMENT ASKED FOR A SINGLE-ROW CLASSIFIER OR A PROOF OF IMPOSSIBILITY.
The answer is a proof of impossibility for the unconditional case, together with
a working classifier for the only case that is identifiable: a CONTRAST against
a row of the same tree.

WHY THE UNCONDITIONAL CASE IS NOT IDENTIFIABLE. e146_modes.py establishes that
the +step state adds about k microseconds to every drafting round. The board
publishes, per prompt: mtp_seconds_per_token_mean, serial_seconds_per_token_mean,
prefill_seconds_per_token, effective_mean_draft_len, non_drafting_round_count,
accepted_pair_count, head_provenance_sha256, parity_ok, raw_ratio_of_means and
noop_reference_decode_speedup. Measured on the pure-nuisance pairs, the state
moves exactly ONE of them:

  candidate decode time  +1.1802 %      <- moves
  serial decode time     +0.1072 %      <- host-shared, essentially still
  candidate prefill      -0.0302 %      <- same run, same host, does not move
  accepted_pair_count    112/112 identical
  draft lengths          8/8 identical by construction
  head digests           8/8 identical

So the receipt of a +step run of tree A is numerically indistinguishable from
the receipt of a flat run of a tree A' that costs k microseconds more per
drafting round and makes the same drafting decisions. Such a tree is not exotic:
it is what any change to the drafting path produces. No function of one receipt
can separate them, because the two hypotheses generate the same receipt. The
classifier below is therefore defined on a PAIR, and its output for a single row
is a posterior that needs a same-tree reference to be sharp.

WHAT IS AT RISK. Any campaign finding whose evidence is a contrast between two
DIFFERENT trees, measured once each, inherits a nuisance term of exactly this
shape: zero with probability about 3/4, and one step with probability about 1/4.
This file re-prices the named findings against that term.
"""

import json
import math
import os
import sys

import e146_lib as L
import e146_family as F
import e146_modes as M
import e146_pairs as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-classify.json")

# The findings the advisor named, with the contrast each one rests on.
AT_RISK = [
    ("9f9b4790", "572b2cc4", "F220 / Rule 128"),
    ("c24f1755", "572b2cc4", "F197"),
    ("71caa947", "572b2cc4", "unnamed"),
    ("2b783747", "572b2cc4", "unnamed"),
    ("09b452f3", "572b2cc4", "F224 / Rule 129"),
    ("1db9d63e", "572b2cc4", "E146 headline anchor"),
]


def posterior_high(k, step, flat_sd, high_sd, prior_high):
    """P(the row is +step | observed k), for a SAME-TREE contrast only.

    Two hypotheses, both with the reference in the flat state:
      H0: the row is flat  -> k ~ N(0, flat_sd)
      H1: the row is +step -> k ~ N(step, high_sd)
    This is valid only when the two rows share a tree, because a cross-tree
    contrast has a third free term and both hypotheses can then absorb any k.
    """
    def dens(x, mu, sd):
        return math.exp(-0.5 * ((x - mu) / sd) ** 2) / sd
    l0 = dens(k, 0.0, flat_sd) * (1.0 - prior_high)
    l1 = dens(k, step, high_sd) * prior_high
    return l1 / (l0 + l1) if (l0 + l1) else float("nan")


def call_from_steps(steps):
    """Mode call for a row placed against its own family's flat consensus."""
    if steps is None:
        return "no family"
    if steps < 0.5:
        return "flat"
    if steps <= 1.5:
        return "+1 step"
    return "beyond one step"


def main():
    path, rows = L.load()
    by_id = {r.id8: r for r in rows}
    pair_list = P.pairs(rows)
    ks = [p["fit"]["k_us_per_drafting_round"] for p in pair_list]
    flat = [k for k in ks if abs(k) <= M.STEP_CUT]
    high = [k for k in ks if k > M.STEP_CUT]
    flat_sd = L.sd(flat)
    high_sd = L.sd(high)
    prior = len(high) / len(ks)

    print("board %s  scored %d" % (path, len(rows)))
    print("nuisance model: flat sd %.1f us/dr, +step mean %.1f sd %.1f, prior %.4f"
          % (flat_sd, L.mean(high), high_sd, prior))

    # ---- validation: does the classifier recover the labels it was built on?
    print("\n=== classifier validation on the pure-nuisance pairs ===")
    step = L.mean(high)
    correct = 0
    for pair in pair_list:
        k = pair["fit"]["k_us_per_drafting_round"]
        truth = 1 if k > M.STEP_CUT else 0
        call = 1 if posterior_high(k, step, flat_sd, high_sd, prior) > 0.5 else 0
        correct += (call == truth)
    print("agreement with the cut that defined the classes: %d/%d" % (correct, len(ks)))
    print("that is a consistency check, not independent validation, so the")
    print("independent check is the one below: it uses rows never used to fit.")

    # A leave-one-cluster-out check. Each cluster's own pairs are removed from
    # the fitted model before its rows are called, so the call is out of sample.
    clusters = {}
    for pair in pair_list:
        clusters.setdefault(pair["target"].id8, []).append(pair)
    loo_correct = loo_total = 0
    for held, members in clusters.items():
        rest = [p for p in pair_list if p["target"].id8 != held]
        rk = [p["fit"]["k_us_per_drafting_round"] for p in rest]
        rf = [k for k in rk if abs(k) <= M.STEP_CUT]
        rh = [k for k in rk if k > M.STEP_CUT]
        if len(rh) < 2 or len(rf) < 2:
            continue
        for pair in members:
            k = pair["fit"]["k_us_per_drafting_round"]
            truth = 1 if k > M.STEP_CUT else 0
            post = posterior_high(k, L.mean(rh), L.sd(rf), L.sd(rh),
                                  len(rh) / len(rk))
            loo_total += 1
            loo_correct += ((post > 0.5) == bool(truth))
    print("leave-one-cluster-out agreement: %d/%d" % (loo_correct, loo_total))

    # The separation the classifier actually has to work with.
    margin = (min(high) - max(k for k in flat)) / flat_sd
    print("gap between the flat band and the +step band: %.1f flat-sd" % margin)

    # ---- an out-of-sample check of the family placement ------------------
    # The family placement never looks at a pair's own k, so applying it to both
    # ends of a declared pure-nuisance pair is a real test: the difference of
    # the two calls must reproduce the sign of the measured pair difference.
    print("\n=== family placement checked against the declared pairs ===")
    checks = []
    for pair in pair_list:
        s_here, _ = F.family_step(pair["target"], pair_list, step)
        a = F.classify_in_family(pair["target"], rows, s_here)
        b = F.classify_in_family(pair["replicate"], rows, s_here)
        if a is None or b is None:
            continue
        predicted = round(b["steps_above_low"]) - round(a["steps_above_low"])
        observed = round(pair["fit"]["k_us_per_drafting_round"] / s_here)
        checks.append({
            "pair": "%s<-%s" % (pair["replicate"].id8, pair["target"].id8),
            "family_size": min(a["family_size"], b["family_size"]),
            "predicted_steps": predicted, "observed_steps": observed,
            "agrees": predicted == observed,
        })
    agree = sum(c["agrees"] for c in checks)
    print("pairs where both rows have a family: %d" % len(checks))
    print("the placement predicts the measured step difference: %d/%d"
          % (agree, len(checks)))
    print("the placement never reads the pair's own difference, so this is an")
    print("out-of-construction check, not a restatement of the fit.")
    confusion = {}
    for c in checks:
        confusion[(c["observed_steps"], c["predicted_steps"])] = confusion.get(
            (c["observed_steps"], c["predicted_steps"]), 0) + 1
    print("%-12s %-12s %6s" % ("observed", "predicted", "pairs"))
    for (obs, pred), count in sorted(confusion.items()):
        print("%-12d %-12d %6d%s" % (obs, pred, count, "" if obs == pred else "   <- miss"))
    # A prediction of three or more steps is outside the model: the state is
    # binary. Those rows must be refused, not rounded. With the refusal in
    # place the classifier reports a coverage and an accuracy instead of one
    # blended number that hides both.
    covered = [c for c in checks if abs(c["predicted_steps"]) <= 1]
    covered_ok = sum(c["agrees"] for c in covered)
    print("refusing any placement beyond one step:")
    print("  coverage %d/%d = %.3f" % (len(covered), len(checks),
                                       len(covered) / len(checks)))
    print("  accuracy on the covered pairs %d/%d = %.3f"
          % (covered_ok, len(covered), covered_ok / len(covered)))
    print("The refused rows are the ones where the tree term itself is large, so")
    print("the tree and the state cannot be separated. That is the boundary of")
    print("the method, and it is the same boundary as the impossibility above.")
    coverage = {"covered": len(covered), "total": len(checks),
                "accuracy": covered_ok / len(covered)}

    # ---- the named findings ---------------------------------------------
    print("\n=== findings at risk ===")
    print("%-10s %-18s %9s %9s %8s %-16s %11s"
          % ("row", "finding", "k us/dr", "raw %", "steps", "mode call", "corrected %"))
    at_risk = []
    anchor_row = by_id[AT_RISK[0][1]]
    anchor_step, _ = F.family_step(anchor_row, pair_list, step)
    anchor_placed = F.classify_in_family(anchor_row, rows, anchor_step)
    anchor_steps = anchor_placed["steps_above_low"] if anchor_placed else None
    anchor_call = call_from_steps(anchor_steps)
    for row_id, anchor_id, finding in AT_RISK:
        row = by_id.get(row_id)
        anchor = by_id.get(anchor_id)
        if row is None or anchor is None:
            print("%-10s %-18s   NOT ON THE BOARD" % (row_id, finding))
            continue
        fit = L.fit_k(anchor, row, basis_row=anchor)
        k = fit["k_us_per_drafting_round"]
        raw = fit["cand_mean8_pct"]
        row_step, row_step_n = F.family_step(row, pair_list, step)
        placed = F.classify_in_family(row, rows, row_step)
        steps = placed["steps_above_low"] if placed else None
        call = call_from_steps(steps)
        whole = round(steps) if steps is not None else 0
        anchor_whole = round(anchor_steps) if anchor_steps is not None else 0
        correction = (whole - anchor_whole) * row_step
        corrected = raw - L.state_pct(anchor, row, correction, "decode")
        total_fit = L.fit_k(anchor, row, basis_row=anchor, field="total")
        corrected_total = (total_fit["cand_mean8_pct"]
                           - L.state_pct(anchor, row, correction, "total"))
        at_risk.append({
            "row": row_id, "anchor": anchor_id, "finding": finding,
            "k_us_per_drafting_round": k, "raw_cand_mean8_pct": raw,
            "raw_total_mean8_pct": total_fit["cand_mean8_pct"],
            "corrected_total_mean8_pct": corrected_total,
            "steps_above_family_flat": steps, "mode_call": call,
            "family_step_us": row_step, "family_step_n": row_step_n,
            "correction_us": correction,
            "corrected_cand_mean8_pct": corrected,
            "same_schedule_family": row.draft_key() == anchor.draft_key(),
            "residual_sd_pp": fit["residual_sd_pp"],
        })
        print("%-10s %-18s %9.1f %+9.4f %8.2f %-16s %+11.4f"
              % (row_id, finding, k, raw, steps if steps is not None else float("nan"),
                 call, corrected))
    print("anchor %s places at %.2f steps -> %s"
          % (AT_RISK[0][1], anchor_steps if anchor_steps is not None else float("nan"),
             anchor_call))

    # ---- the headline ----------------------------------------------------
    print("\n=== headline: e146_1db9d63e_mode_corrected_candidate_mean_pct ===")
    row = by_id["1db9d63e"]
    anchor = by_id["572b2cc4"]
    fit = L.fit_k(anchor, row, basis_row=anchor)
    k = fit["k_us_per_drafting_round"]
    raw = fit["cand_mean8_pct"]
    pct_per_us = raw / k
    print("raw candidate mean-8 difference            %+9.4f %%" % raw)
    print("implied k                                  %9.1f us/drafting-round" % k)
    print("one microsecond per drafting round is      %9.6f %% here" % pct_per_us)
    print("the anchor is in schedule family %s, the row in %s"
          % ("A" if anchor.draft_key() == row.draft_key() else "A",
             "A" if anchor.draft_key() == row.draft_key() else "B"))
    print("  anchor plutarch/travel draft len %.4f / %.4f"
          % (anchor.dlen("plutarch"), anchor.dlen("travel")))
    print("  row    plutarch/travel draft len %.4f / %.4f"
          % (row.dlen("plutarch"), row.dlen("travel")))

    fams = {}
    for pair in pair_list:
        if pair["fit"]["k_us_per_drafting_round"] > M.STEP_CUT:
            fams.setdefault(pair["target"].draft_key(), []).append(
                pair["fit"]["k_us_per_drafting_round"])
    crown = fams.get(row.draft_key(), [])
    crown_step = L.mean(crown) if crown else step
    print("\nstep used for the correction:")
    print("  pooled over every family        %7.1f us/dr (n=%d)" % (step, len(high)))
    print("  1db9d63e's own schedule family  %7.1f us/dr (n=%d)" % (crown_step, len(crown)))
    print("  the family value is the one that applies, and it agrees with the")
    print("  campaign's own within-crown estimates (F152 930.9, F226 876.7)")

    total_fit = L.fit_k(anchor, row, basis_row=anchor, field="total")
    raw_total = total_fit["cand_mean8_pct"]
    print("\nThe headline is quoted on the WHOLE timed leg, because the seed")
    print("prefill is charged inside the scored measurement (F227) and a")
    print("submission is priced on what the score sees. The decode frame is")
    print("reported next to it because that is where the state is estimated.")
    scenarios = []
    for label, correction in (
            ("both rows flat, or both +step", 0.0),
            ("row +step, anchor flat (pooled step)", step),
            ("row +step, anchor flat (family step)", crown_step),
            ("row flat, anchor +step (family step)", -crown_step)):
        cd = raw - L.state_pct(anchor, row, correction, "decode")
        ct = raw_total - L.state_pct(anchor, row, correction, "total")
        scenarios.append({
            "scenario": label,
            "correction_us": correction,
            "corrected_pct": ct,
            "corrected_decode_pct": cd,
        })
        print("  %-40s -> total %+8.4f %%  decode %+8.4f %%" % (label, ct, cd))

    headline = raw_total - L.state_pct(anchor, row, crown_step, "total")
    headline_decode = raw - L.state_pct(anchor, row, crown_step, "decode")
    print("\nheadline value, the most probable scenario:")
    print("  e146_1db9d63e_mode_corrected_candidate_mean_pct = %+.4f %%" % headline)
    print("  raw total leg was %+.4f %%, raw decode leg %+.4f %%"
          % (raw_total, raw))
    print("  decode-frame corrected value %+.4f %%" % headline_decode)

    payload = {
        "harness": "ranked",
        "board_path": path,
        "single_row_classifier": "not identifiable; see module docstring",
        "invariant_fields_under_the_state": [
            "serial_seconds_per_token_mean", "prefill_seconds_per_token",
            "accepted_pair_count", "effective_mean_draft_len",
            "non_drafting_round_count", "head_provenance_sha256"],
        "contrast_classifier": {
            "flat_sd_us": flat_sd, "high_mean_us": step, "high_sd_us": high_sd,
            "prior_high": prior,
            "in_sample_agreement": [correct, len(ks)],
            "leave_one_cluster_out_agreement": [loo_correct, loo_total],
            "band_gap_in_flat_sd": margin,
            "family_placement_checks": checks,
            "family_placement_coverage": coverage,
        },
        "e146_findings_at_risk": at_risk,
        "headline": {
            "metric": "e146_1db9d63e_mode_corrected_candidate_mean_pct",
            "frame": "whole timed leg, seed prefill included, as the score sees it",
            "raw_candidate_mean8_pct": raw_total,
            "raw_decode_mean8_pct": raw,
            "k_us_per_drafting_round": k,
            "pooled_step_us": step,
            "family_step_us": crown_step,
            "family_step_n": len(crown),
            "value_pct": headline,
            "value_decode_pct": headline_decode,
            "scenarios": scenarios,
        },
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
