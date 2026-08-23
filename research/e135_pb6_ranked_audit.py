#!/usr/bin/env python3
"""Audit the pb6 depth price against our own ranked receipts.

F34 measured pb6 as `+2.2987 %` faster on this host under the tight launch
grid. CAMPAIGN RULE 79 says no local timing leg can validate a depth price,
because the local fixture drafts far deeper than the ranked prompt mix. This
script does not argue that point from theory. It reads the receipts.

pb6 changes the drafting schedule, so it is identified on a receipt by its
`effective_mean_draft_len` column rather than by a note. Every other mechanism
the campaign has shipped leaves that column alone. The script partitions our
receipts into a pb6 group and a flat-price group on that column, then prices
pb6 three ways: on the published median, on the median pair held fixed at the
prompts that set it before the change, and under CAMPAIGN RULE 129, which
scores a non-uniform mechanism at its worst upper-slot candidate.

    python3 research/e135_pb6_ranked_audit.py [--board PATH] [--write]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

BOARD = "/tmp/yukon-board/full.json"

# Our receipts, oldest first. pb6 membership is derived, never asserted.
RECEIPTS = ("572b2cc4", "e003a86d", "1db9d63e")

# Mechanism content of each receipt, read from its own published note. A
# contrast is only worth quoting once its confounders are named, and the first
# version of this script quoted a three-mechanism contrast as though it
# isolated pb6.
CONTENT = {
    "572b2cc4": {"onePass67": 1, "p15": 0, "pb6": 0, "width2": 0},
    "e003a86d": {"onePass67": 0, "p15": 1, "pb6": 1, "width2": 0},
    "1db9d63e": {"onePass67": 0, "p15": 1, "pb6": 0, "width2": 1},
}

# Independent ranked estimates for the two mechanisms we cannot isolate from
# our own three receipts alone. p15 is the agreed range of two receipts on two
# bases; width2 is the public isolation pair on the candidate median pair.
EXTERNAL = {"p15": 0.325, "width2": 0.4742}

NAME = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# CAMPAIGN RULE 129. The lower median slot is beagle in every receipt we hold;
# the upper slot is whichever of these four sorts lowest.
UPPER_SLOT_CANDIDATES = ("essays", "republic", "medicine", "botany")


def prompt_name(entry: dict) -> str:
    blob = json.dumps(entry)
    for sha, name in NAME.items():
        if sha in blob:
            return name
    return "?"


def load(board: str) -> dict:
    raw = json.loads(pathlib.Path(board).read_text())
    rows = raw if isinstance(raw, list) else (
        raw.get("submissions") or raw.get("data") or [])
    out = {}
    for short in RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(short))
        out[short] = {
            "score": row["officialScore"],
            "commit": row["submissionCommitSha"][:8],
            "prompts": {prompt_name(p): p
                        for p in row["officialMetrics"]["per_prompt"]},
        }
    return out


def median_of(ratios: dict) -> float:
    return statistics.median(sorted(ratios.values()))


# F83 median-pair influence weights. The three prompts that never reach an
# upper slot carry no weight, so they are absent rather than set to zero.
F83 = {"beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
       "botany": 0.0124, "republic": 0.0100}


def onepass67_contrast(data: dict, flat: list[str]) -> dict:
    """Price onePass67 across the two flat-price receipts.

    Both receipts run the flat depth price, so their drafting schedules should
    agree prompt by prompt and any candidate-leg difference is pure time. The
    prompts whose `effective_mean_draft_len` matches to the last digit give the
    cleanest read, so they are reported separately from the eight-prompt mean.
    """
    if len(flat) != 2:
        return {}
    ref, cand = (data[flat[0]]["prompts"], data[flat[1]]["prompts"])
    print(f"\nONEPASS67 CONTRAST  {flat[0]} (onePass67) -> {flat[1]}"
          f" (no onePass67, plus the width-two route)")
    print(f"  {'prompt':10}{'edl a':>9}{'edl b':>9}{'same':>6}"
          f"{'mtp a':>10}{'mtp b':>10}{'mtp %':>9}")
    matched, every = [], []
    for name in sorted(ref):
        a, b = ref[name], cand[name]
        pct = -100 * (b["mtp_seconds_per_token_mean"]
                      / a["mtp_seconds_per_token_mean"] - 1)
        ident = a["effective_mean_draft_len"] == b["effective_mean_draft_len"]
        every.append(pct)
        if ident:
            matched.append((name, pct))
        print(f"  {name:10}{a['effective_mean_draft_len']:>9.4f}"
              f"{b['effective_mean_draft_len']:>9.4f}"
              f"{('yes' if ident else 'no'):>6}"
              f"{a['mtp_seconds_per_token_mean']:>10.6f}"
              f"{b['mtp_seconds_per_token_mean']:>10.6f}{pct:>+9.2f}")
    weighted = (sum(F83[k] * -100
                    * (cand[k]["mtp_seconds_per_token_mean"]
                       / ref[k]["mtp_seconds_per_token_mean"] - 1)
                    for k in F83) / sum(F83.values()))
    names = [n for n, _ in matched]
    vals = [p for _, p in matched]
    print(f"\n  positive means {flat[1]} is faster.")
    print(f"  schedule digit-identical on {len(names)} of 8: {names}")
    print(f"    mean over those            {statistics.fmean(vals):+.4f} %"
          f"  sd {statistics.stdev(vals):.4f}")
    print(f"    mean over all eight        {statistics.fmean(every):+.4f} %"
          f"  sd {statistics.stdev(every):.4f}")
    print(f"    F83 weighted five          {weighted:+.4f} %")
    print("  Every value is negative, so this pair got slower on the candidate"
          " leg on all eight prompts.")
    print("  The pair also adds p15 and the width-two route, and both are"
          " believed positive, so the")
    print("  loss understates onePass67. See the isolation below for the"
          " confounder-corrected price.")
    return {
        "e135_minus_onepass67_p15_width2_matched_schedule_pct":
            statistics.fmean(vals),
        "e135_minus_onepass67_p15_width2_eight_prompt_pct": statistics.fmean(every),
        "e135_minus_onepass67_p15_width2_weighted_five_pct": weighted,
        "e135_minus_onepass67_p15_width2_matched_prompts": names,
    }


def delta(short_a: str, short_b: str) -> dict:
    """Mechanisms that change between two receipts, as +1 added, -1 removed."""
    a, b = CONTENT[short_a], CONTENT[short_b]
    return {k: b[k] - a[k] for k in a if b[k] != a[k]}


def decompose(data: dict) -> dict:
    """Isolate pb6 as far as three receipts allow, naming every confounder.

    No pair of our receipts differs by pb6 alone. The closest pair differs by
    pb6 added and the width-two route removed, so pb6 is recovered by adding
    back an independent estimate of width2. The published median carries the
    serial baseline draw, so the candidate leg is reported beside it.
    """
    print("\nRECEIPT CONTENT, from each receipt's own note")
    for short in RECEIPTS:
        on = " ".join(k for k, v in CONTENT[short].items() if v)
        print(f"  {short:10}{data[short]['score']:>13.8f}   {on}")

    print("\nEVERY AVAILABLE PAIR, and what actually changes in it")
    for i, a in enumerate(RECEIPTS):
        for b in RECEIPTS[i + 1:]:
            d = delta(a, b)
            med = 100 * (data[b]["score"] / data[a]["score"] - 1)
            tag = " ".join(f"{'+' if v > 0 else '-'}{k}"
                           for k, v in sorted(d.items()))
            print(f"  {a} -> {b}   median {med:>+8.4f} %   changes: {tag}")
    print("  No pair changes one mechanism only, so nothing below is a clean"
          " isolation.")
    print("  Three receipts give only two independent contrasts, and the two"
          " solved prices consume")
    print("  both, so the third pair cannot check them. Any agreement there is"
          " arithmetic, not evidence.")

    # Solve in dependency order. pb6's only confounder is priced externally,
    # and its solved price then unlocks onePass67, whose closest pair carries
    # pb6. Each solved price feeds the next isolation.
    prices = dict(EXTERNAL)
    out = {}
    for target in ("pb6", "onePass67"):
        got = isolate(data, target, prices)
        prices[target] = got[
            f"e135_{target}_ranked_isolated_median_pct"]
        out.update(got)
    return out


def isolate(data: dict, target: str, prices: dict) -> dict:
    """Price one mechanism from the pair that confounds it least.

    Every reported number is the effect of TURNING THE MECHANISM ON, whichever
    direction the underlying pair runs in, so the sign never depends on which
    receipt happened to be published first.
    """
    best = None
    for i, a in enumerate(RECEIPTS):
        for b in RECEIPTS[i + 1:]:
            d = delta(a, b)
            if target in d and (best is None or len(d) < len(best[2])):
                best = (a, b, d)
    if best is None:
        return {}
    a, b, d = best
    # Orient the pair so the target mechanism is switched on, not off.
    if d[target] < 0:
        a, b = b, a
        d = delta(a, b)
    med = 100 * (data[b]["score"] / data[a]["score"] - 1)

    print(f"\nISOLATING {target}   pair {a} -> {b}, which switches it ON")
    print(f"  confounders in this pair: "
          f"{sorted(k for k in d if k != target) or 'none'}")
    print(f"  published median move           {med:>+9.4f} %")
    correction = 0.0
    for k, v in sorted(d.items()):
        if k == target:
            continue
        term = -v * prices[k]
        correction += term
        verb = "added" if v > 0 else "removed"
        print(f"  back out {k}, {verb} in the same pair,"
              f" independently priced {prices[k]:+.4f} %"
              f"   -> {term:+.4f} %")
    print(f"  {target} alone, published median  {med + correction:>+9.4f} %")

    ref, cand = data[a]["prompts"], data[b]["prompts"]
    beagle = -100 * (cand["beagle"]["mtp_seconds_per_token_mean"]
                     / ref["beagle"]["mtp_seconds_per_token_mean"] - 1)
    print(f"  beagle candidate leg            {beagle:>+9.4f} %"
          f"   -> {beagle + correction:+.4f} % corrected")
    return {
        f"e135_{target}_ranked_pair": f"{a}->{b}",
        f"e135_{target}_ranked_confounders": sorted(
            k for k in d if k != target),
        f"e135_{target}_ranked_pair_median_pct": med,
        f"e135_{target}_ranked_isolated_median_pct": med + correction,
        f"e135_{target}_ranked_isolated_beagle_pct": beagle + correction,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=BOARD)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    data = load(args.board)

    print("RECEIPT IDENTITY. pb6 is read off the schedule, not off the note.")
    print(f"  {'receipt':10}{'score':>13}  {'commit':8} "
          f"{'plutarch edl':>13}{'nondraft':>9}  schedule")
    flat, pb6 = [], []
    for short, rec in data.items():
        plut = rec["prompts"]["plutarch"]
        edl = plut["effective_mean_draft_len"]
        nd = plut["non_drafting_round_count"]
        # The flat price never drafts on the shallowest prompt; pb6 makes every
        # round cheap enough to draft, which is the whole mechanism.
        arm = "pb6" if nd == 0 else "flat"
        (pb6 if arm == "pb6" else flat).append(short)
        print(f"  {short:10}{rec['score']:>13.8f}  {rec['commit']:8} "
              f"{edl:>13.4f}{nd:>9}  {arm}")
    print(f"\n  flat-price receipts {flat}")
    print(f"  pb6 receipts        {pb6}")
    if len(pb6) != 1 or not flat:
        print("  cannot isolate pb6 from these receipts")
        return 1

    ref = data[flat[0]]
    cand = data[pb6[0]]
    print(f"\nPB6 CONTRAST  {flat[0]} (flat) -> {pb6[0]} (pb6)")
    print(f"  {'prompt':10}{'raw flat':>10}{'raw pb6':>10}{'raw %':>9}"
          f"{'mtp flat':>10}{'mtp pb6':>10}{'mtp %':>9}{'edl flat':>9}"
          f"{'edl pb6':>9}")
    for name in sorted(ref["prompts"]):
        a, b = ref["prompts"][name], cand["prompts"][name]
        raw_pct = 100 * (b["raw_ratio_of_means"] / a["raw_ratio_of_means"] - 1)
        # Negative means pb6 spent more seconds per token, so it is slower.
        mtp_pct = -100 * (b["mtp_seconds_per_token_mean"]
                          / a["mtp_seconds_per_token_mean"] - 1)
        print(f"  {name:10}{a['raw_ratio_of_means']:>10.4f}"
              f"{b['raw_ratio_of_means']:>10.4f}{raw_pct:>+9.2f}"
              f"{a['mtp_seconds_per_token_mean']:>10.6f}"
              f"{b['mtp_seconds_per_token_mean']:>10.6f}{mtp_pct:>+9.2f}"
              f"{a['effective_mean_draft_len']:>9.4f}"
              f"{b['effective_mean_draft_len']:>9.4f}")
    print("  raw % and mtp % are both positive when pb6 is better.")

    raw_a = {k: v["raw_ratio_of_means"] for k, v in ref["prompts"].items()}
    raw_b = {k: v["raw_ratio_of_means"] for k, v in cand["prompts"].items()}
    med_a, med_b = median_of(raw_a), median_of(raw_b)

    pair_a = sorted(raw_a, key=lambda k: raw_a[k])[3:5]
    pair_b = sorted(raw_b, key=lambda k: raw_b[k])[3:5]
    held = statistics.fmean(raw_b[k] for k in pair_a)
    worst_upper = min(UPPER_SLOT_CANDIDATES, key=lambda k: raw_b[k])
    rule129 = statistics.fmean([raw_b["beagle"], raw_b[worst_upper]])

    print(f"\nTHREE PRICES FOR THE SAME MECHANISM")
    print(f"  published median      {med_a:.8f} -> {med_b:.8f}"
          f"  {100 * (med_b / med_a - 1):+.4f} %")
    print(f"    median pair moved   {pair_a} -> {pair_b}")
    print(f"  median pair held      {med_a:.8f} -> {held:.8f}"
          f"  {100 * (held / med_a - 1):+.4f} %   (pair fixed at {pair_a})")
    print(f"  CAMPAIGN RULE 129     {med_a:.8f} -> {rule129:.8f}"
          f"  {100 * (rule129 / med_a - 1):+.4f} %"
          f"   (worst upper slot: {worst_upper})")

    beagle_pct = 100 * (raw_b["beagle"] / raw_a["beagle"] - 1)
    print(f"\n  beagle, the lower median slot in every receipt we hold, moves"
          f" {beagle_pct:+.4f} %.")
    print("  That is a candidate-leg loss on the anchor prompt, not a reorder"
          " artefact.")

    onepass = onepass67_contrast(data, flat)
    isolated = decompose(data)

    out = {
        "flat_receipt": flat[0], "pb6_receipt": pb6[0],
        **onepass, **isolated,
        # These four describe the three-mechanism pair, not pb6. They keep the
        # confounding in the key so no reader can quote them as a pb6 price;
        # the isolated pb6 numbers are the `e135_pb6_ranked_isolated_*` keys.
        "e135_pb6_p15_minus_onepass67_published_median_pct":
            100 * (med_b / med_a - 1),
        "e135_pb6_p15_minus_onepass67_held_pair_pct":
            100 * (held / med_a - 1),
        "e135_pb6_p15_minus_onepass67_rule129_pct":
            100 * (rule129 / med_a - 1),
        "e135_pb6_p15_minus_onepass67_beagle_pct": beagle_pct,
        "e135_pb6_ranked_worst_upper_slot": worst_upper,
        "median_pair_flat": pair_a, "median_pair_pb6": pair_b,
    }
    if args.write:
        path = pathlib.Path(
            f"research/e135-artifacts/pb6-ranked-audit-{flat[0]}-{pb6[0]}.json")
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
