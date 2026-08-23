"""E146 R-A rung 1b: the crown schedule family, read against one anchor.

harness=ranked.

WHY THIS EXISTS. A posterior on the run mode of ONE row is flat whenever the two
rows being contrasted have different trees, because a tree that costs k more per
drafting round and a +step draw produce the same receipt. The way out is not a
better estimator on two rows, it is more rows: inside one schedule family the
same eight draft lengths are produced by many different trees, so the empirical
distribution of k across the family shows whether k piles up at the quantum.

THE OBJECTIVE SAME-DECISION FINGERPRINT. `accepted_pair_count` is published per
prompt and decoding is deterministic, so two runs of the same tree must publish
the same eight values. On the 57 solver-declared pure-nuisance pairs, all 424
prompt slots match exactly, which is what makes those pairs trustworthy without
believing the solver's note. The same fingerprint is applied here to every row
in the family, so `same decisions as the anchor` is a measured property.

Two rows with the same fingerprint made the same drafting decisions. They may
still differ in kernel speed, so a matching fingerprint does NOT prove a pure
nuisance contrast. It does prove that no schedule or acceptance change is mixed
into the contrast, which removes the largest confound.
"""

import json
import math
import os
import sys

import e146_lib as L
import e146_modes as M
import e146_pairs as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-family.json")
ANCHOR = "572b2cc4"
ROW = "1db9d63e"


def fingerprint(row):
    return tuple(repr(row.prompts[p]["accepted_pair_count"]) for p in L.PROMPT_ORDER)


def family_step(row, pair_list, default, cut=300.0):
    """The step measured by declared replicates INSIDE this row's family.

    Section 6 of e146_modes.py shows the step is a band, not a constant, and
    that it differs between families by about 40 per cent. Placing a row with
    another family's step is what turns 1.32 steps into 1.59 steps and then
    rounds it to the wrong integer, so the family value is used wherever one
    exists.
    """
    vals = [p["fit"]["k_us_per_drafting_round"] for p in pair_list
            if p["fit"]["k_us_per_drafting_round"] > cut
            and p["target"].draft_key() == row.draft_key()]
    return (L.mean(vals) if vals else default), len(vals)


def classify_in_family(row, rows, step, window_hours=14.0):
    """Where does `row` sit inside its own schedule family?

    The family is every row with the same eight draft lengths AND the same eight
    accepted-pair counts, inside a calendar window around `row`. The window
    matters: solvers copy the promoted frontier within hours, so a family read
    over days mixes genuine progress into the same axis as the run mode.

    The family's k values are then split at gaps wider than half a step. The
    lowest cluster is the flat-mode consensus, because a run mode can only make
    a row SLOWER than its own tree allows.
    """
    fp = fingerprint(row)
    here = P._hours(row.created)
    family = [r for r in rows
              if r.draft_key() == row.draft_key() and fingerprint(r) == fp
              and abs(P._hours(r.created) - here) <= window_hours]
    if len(family) < 4:
        return None
    ks = sorted(((L.fit_k(row, r, basis_row=row)["k_us_per_drafting_round"], r)
                 for r in family), key=lambda t: t[0])
    clusters = [[ks[0]]]
    for value in ks[1:]:
        if value[0] - clusters[-1][-1][0] > step / 2.0:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    low = clusters[0]
    low_centre = L.mean([v for v, _ in low])
    excess = -low_centre
    return {
        "family_size": len(family),
        "clusters": [[(v, r.id8) for v, r in c] for c in clusters],
        "low_cluster_size": len(low),
        "low_centre_us": low_centre,
        "row_excess_over_low_us": excess,
        "steps_above_low": excess / step,
        "window_hours": window_hours,
    }


def main():
    path, rows = L.load()
    by_id = {r.id8: r for r in rows}
    anchor = by_id[ANCHOR]
    target = by_id[ROW]

    print("board %s  scored %d" % (path, len(rows)))
    print("anchor %s (%s, %s, score %.4f)"
          % (ANCHOR, anchor.solver, anchor.created[:16], anchor.score))
    print("row    %s (%s, %s, score %.4f)"
          % (ROW, target.solver, target.created[:16], target.score))

    anchor_fp = fingerprint(anchor)
    row_fp = fingerprint(target)
    print("\nsame-decision fingerprint (accepted_pair_count, eight prompts):")
    print("  anchor %s" % " ".join(anchor_fp))
    print("  row    %s" % " ".join(row_fp))
    print("  identical: %s" % (anchor_fp == row_fp))

    twins = [r for r in rows if r.id8 != ROW and fingerprint(r) == row_fp
             and r.draft_key() == target.draft_key()]
    print("\nrows sharing the row's exact decisions and schedule: %d" % len(twins))
    for twin in sorted(twins, key=lambda r: r.created):
        fit = L.fit_k(twin, target, basis_row=twin)
        print("  %-10s %-14s %-18s score %.4f  k vs row %+9.1f  cand8 %+8.4f %%"
              % (twin.id8, twin.solver[:14], twin.created[:16], twin.score,
                 fit["k_us_per_drafting_round"], fit["cand_mean8_pct"]))

    # ---- the family, read against the anchor -----------------------------
    family = [r for r in rows if r.draft_key() == target.draft_key()]
    family.sort(key=lambda r: r.created)
    print("\n=== every row of the row's schedule family, against %s ===" % ANCHOR)
    print("%-10s %-14s %-17s %8s %9s %9s %7s %5s"
          % ("id", "solver", "created", "score", "k us/dr", "cand8 %", "resid", "fp"))
    entries = []
    for r in family:
        fit = L.fit_k(anchor, r, basis_row=anchor)
        same_fp = fingerprint(r) == anchor_fp
        entries.append({
            "id": r.id8, "solver": r.solver, "created": r.created,
            "score": r.score,
            "k_us_per_drafting_round": fit["k_us_per_drafting_round"],
            "cand_mean8_pct": fit["cand_mean8_pct"],
            "residual_sd_pp": fit["residual_sd_pp"],
            "same_decisions_as_anchor": same_fp,
        })
        print("%-10s %-14s %-17s %8.4f %9.1f %+9.4f %7.4f %5s"
              % (r.id8, r.solver[:14], r.created[:16], r.score or float("nan"),
                 fit["k_us_per_drafting_round"], fit["cand_mean8_pct"],
                 fit["residual_sd_pp"], "y" if same_fp else "n"))

    ks = sorted(e["k_us_per_drafting_round"] for e in entries)
    print("\nfamily k spread: %.1f .. %.1f" % (ks[0], ks[-1]))
    gap = M.largest_gap(ks, min(ks), max(ks))
    print("widest empty interval inside the family: %.1f us, %.1f -> %.1f"
          % (gap["width"], gap["from"], gap["to"]))

    # Rows that made the anchor's exact decisions are the cleanest sub-sample:
    # no schedule change and no acceptance change is mixed into their k.
    same = [e for e in entries if e["same_decisions_as_anchor"]]
    print("\nrows in the family with the ANCHOR's exact decisions: %d" % len(same))
    for e in sorted(same, key=lambda e: e["k_us_per_drafting_round"]):
        print("  %-10s %-14s k %+9.1f  cand8 %+8.4f %%  score %.4f"
              % (e["id"], e["solver"][:14], e["k_us_per_drafting_round"],
                 e["cand_mean8_pct"], e["score"] or float("nan")))
    if len(same) > 1:
        vals = [e["k_us_per_drafting_round"] for e in same]
        print("  spread %.1f .. %.1f, sd %.1f" % (min(vals), max(vals), L.sd(vals)))

    # ---- what the family says about the row ------------------------------
    pair_list = P.pairs(rows)
    high = [p["fit"]["k_us_per_drafting_round"] for p in pair_list
            if p["fit"]["k_us_per_drafting_round"] > M.STEP_CUT]
    fam_high = [p["fit"]["k_us_per_drafting_round"] for p in pair_list
                if p["fit"]["k_us_per_drafting_round"] > M.STEP_CUT
                and p["target"].draft_key() == target.draft_key()]
    band = (min(high), max(high))
    row_k = next(e["k_us_per_drafting_round"] for e in entries if e["id"] == ROW)
    print("\nrow k against the anchor: %+.1f us/dr" % row_k)
    print("board-wide +step band: %.1f .. %.1f" % band)
    print("this family's +step values: %s"
          % ", ".join("%.1f" % v for v in sorted(fam_high)))
    print("the row's k lies inside the board-wide band: %s"
          % ("yes" if band[0] <= row_k <= band[1] else "no"))

    # ---- classify each named row inside its OWN family -------------------
    print("\n=== each row placed inside its own same-decision family ===")
    step_for = L.mean(high)
    placements = {}
    for rid in [ANCHOR, ROW, "9f9b4790", "c24f1755", "1760479a", "02742bf0",
                "684821ed", "4debb1df"]:
        r = by_id.get(rid)
        if r is None:
            continue
        placed = classify_in_family(r, rows, step_for)
        placements[rid] = placed
        if placed is None:
            print("%-10s family too small to place" % rid)
            continue
        print("%-10s family %3d  clusters %s  excess over the flat consensus "
              "%+8.1f us/dr = %.2f steps"
              % (rid, placed["family_size"],
                 "/".join(str(len(c)) for c in placed["clusters"]),
                 placed["row_excess_over_low_us"], placed["steps_above_low"]))
    print("A row at about 0 steps ran flat. A row at about 1 step carried the")
    print("state. A row between the two cannot be called from board data alone.")

    payload = {
        "harness": "ranked",
        "board_path": path,
        "family_placements": placements,
        "anchor": ANCHOR,
        "row": ROW,
        "anchor_fingerprint": list(anchor_fp),
        "row_fingerprint": list(row_fp),
        "fingerprints_identical": anchor_fp == row_fp,
        "row_decision_twins": [t.id8 for t in twins],
        "family_size": len(family),
        "family": entries,
        "family_k_min": ks[0],
        "family_k_max": ks[-1],
        "family_widest_gap": gap,
        "rows_with_anchor_decisions": [e["id"] for e in same],
        "board_high_band": list(band),
        "family_high_values": sorted(fam_high),
        "row_k_us_per_drafting_round": row_k,
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
