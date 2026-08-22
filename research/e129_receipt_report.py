#!/usr/bin/env python3
"""Decompose the E129 rung-0 ranked receipt against the crown and the comparison run.

    python3 research/e129_receipt_report.py [--new PREFIX] [--old PREFIX] [--crown PREFIX]

Reads the board cache written by ``research/board_per_prompt.py fetch``. Reports
the published median, its central two order statistics, the per-prompt candidate
leg change, the serial leg null, and the pre-registered band comparison.
harness=ranked throughout.
"""

from __future__ import annotations

import argparse
import json
import pathlib

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
# The pre-registration, in the unit it was written in: the unweighted
# eight-prompt candidate-leg mean, seconds per token. The reference is
# `0c6191b7`, whose mean is 0.014758363 and whose published score is 3.512706.
# Each entry is the fractional change that reading predicts.
PREREG_REFERENCE = "0c6191b7"
PREREG = {
    "one-pass line, parameter-free": -0.0759,
    "pass count alone": -0.0860,
    "instruction count": -0.0256,
    "schedule model": -0.0260,
    "cross-sectional fit": +0.0005,
}
PREREG_BAND = {"band low": -0.09, "central": -0.05, "band high": -0.03,
               "refuted worse than": -0.01}


def receipt(rows: list[dict], prefix: str) -> tuple[dict, dict]:
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d rows" % (prefix, len(hits)))
    row = hits[0]
    per_prompt = {
        NAMES[e["prompt_sha256"][:8]]: e
        for e in row["officialMetrics"]["per_prompt"]
    }
    return row, per_prompt


def central_pair(per_prompt: dict) -> tuple[tuple[str, float], tuple[str, float]]:
    order = sorted(per_prompt.items(), key=lambda kv: kv[1]["raw_ratio_of_means"])
    return (
        (order[3][0], order[3][1]["raw_ratio_of_means"]),
        (order[4][0], order[4][1]["raw_ratio_of_means"]),
    )


def schedule_signature(per_prompt: dict) -> tuple:
    """What the scheduler did, per prompt, to nine decimal places.

    This is an EXACT check, not a noise band. Over our twenty-two receipts
    that carry per-prompt data these two fields take exactly seven distinct
    values, and one of them covers fourteen consecutive receipts across
    forty-one hours and several unrelated candidate archives. The schedule is
    therefore invariant to a kernel change and moves only when the scheduler
    itself moves, so any difference at all is a real confound rather than a
    draw.
    """
    return tuple(
        (name, round(per_prompt[name]["effective_mean_draft_len"], 9),
         per_prompt[name]["non_drafting_round_count"])
        for name in NAMES.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", default="623e77af")
    parser.add_argument("--old", default=PREREG_REFERENCE)
    parser.add_argument("--crown", default="48423d09")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e129-receipt-report.json"))
    args = parser.parse_args()

    rows = json.loads(BOARD.read_text())
    new, npp = receipt(rows, args.new)
    old, opp = receipt(rows, args.old)
    crown, cpp = receipt(rows, args.crown)

    print("harness=ranked")
    for tag, row, per_prompt in (
        (args.new, new, npp), (args.old, old, opp), (args.crown, crown, cpp)
    ):
        lower, upper = central_pair(per_prompt)
        print("%-9s %-9s score %.8f  central two %s %.6f | %s %.6f"
              % (tag, row["status"], row["officialScore"],
                 lower[0], lower[1], upper[0], upper[1]))

    score = new["officialScore"]
    print()
    print("vs crown %s        %+.4f %%" % (args.crown, (score / crown["officialScore"] - 1) * 100))
    print("vs %s              %+.4f %%" % (args.old, (score / old["officialScore"] - 1) * 100))

    # VALIDITY GATE, read before anything is priced. The two schedule fields
    # are bit-identical across archives that share a scheduler, so a single
    # difference means the two receipts did not run the same schedule and the
    # candidate-leg change is not attributable to the kernel arm.
    print()
    same_schedule = schedule_signature(npp) == schedule_signature(opp)
    print("validity: schedule identical to %s ... %s"
          % (args.old, "yes" if same_schedule else "NO, ARM IS CONFOUNDED"))
    if not same_schedule:
        print("%-9s %14s %14s %10s %10s"
              % ("prompt", "draftlen new", "draftlen old", "ndr new", "ndr old"))
        for name in NAMES.values():
            a, b = npp[name], opp[name]
            if (a["effective_mean_draft_len"] != b["effective_mean_draft_len"]
                    or a["non_drafting_round_count"]
                    != b["non_drafting_round_count"]):
                print("%-9s %14.9f %14.9f %10d %10d"
                      % (name, a["effective_mean_draft_len"],
                         b["effective_mean_draft_len"],
                         a["non_drafting_round_count"],
                         b["non_drafting_round_count"]))

    # The pre-registration, in its own unit: the unweighted eight-prompt
    # candidate-leg mean. The published score is not the right unit for it,
    # because the ranked serial leg moves independently of anything we ship.
    ref_mean = old["officialMetrics"]["candidate_mtp_seconds_per_token_mean"]
    got_mean = new["officialMetrics"]["candidate_mtp_seconds_per_token_mean"]
    observed = got_mean / ref_mean - 1.0
    print()
    print("pre-registered candidate-leg mean, negative is faster")
    print("reference %s                          %.9f s/tok"
          % (args.old, ref_mean))
    print("observed                                      %.9f s/tok  %+.3f %%"
          % (got_mean, observed * 100))
    for label, frac in sorted(PREREG.items(), key=lambda kv: kv[1]):
        print("  %-30s predicted %+6.2f %% -> %.9f   miss %+.3f pp"
              % (label, frac * 100, ref_mean * (1 + frac),
                 (observed - frac) * 100))
    for label, frac in sorted(PREREG_BAND.items(), key=lambda kv: kv[1]):
        print("  %-30s %+6.2f %% -> %.9f"
              % (label, frac * 100, ref_mean * (1 + frac)))
    inside = PREREG_BAND["band low"] <= observed <= PREREG_BAND["band high"]
    refuted = observed > PREREG_BAND["refuted worse than"]
    print("  inside the pre-registered band ... %s" % ("yes" if inside else "no"))
    print("  refuted at the claimed size ...... %s" % ("YES" if refuted else "no"))

    print()
    print("%-9s %8s %11s %11s %11s" % ("prompt", "M", "cand gain%", "serial d%", "raw gain%"))
    table = []
    for name in NAMES.values():
        a, b = npp[name], opp[name]
        width = a["effective_mean_draft_len"] + 1.0
        cand = (1.0 - a["mtp_seconds_per_token_mean"] / b["mtp_seconds_per_token_mean"]) * 100
        serial = (a["serial_seconds_per_token_mean"] / b["serial_seconds_per_token_mean"] - 1) * 100
        raw = (a["raw_ratio_of_means"] / b["raw_ratio_of_means"] - 1) * 100
        table.append((width, name, cand, serial, raw))
    table.sort()
    for width, name, cand, serial, raw in table:
        print("%-9s %8.3f %11.3f %11.3f %11.3f" % (name, width, cand, serial, raw))

    routed = [t for t in table if t[0] >= 3.0]
    mean_routed = sum(t[2] for t in routed) / len(routed)
    mean_all = sum(t[2] for t in table) / len(table)
    print("candidate leg, %d routed prompts   mean %+.3f %%" % (len(routed), mean_routed))
    print("candidate leg, all eight          mean %+.3f %%" % mean_all)
    print("serial leg null, all eight        mean %+.3f %%"
          % (sum(t[3] for t in table) / len(table)))

    metrics = new["officialMetrics"]
    heads = {e["head_provenance_sha256"] for e in metrics["per_prompt"]}
    print()
    print("candidate mean %.9f s/tok, serial mean %.9f s/tok, pooled ratio %.6f"
          % (metrics["candidate_mtp_seconds_per_token_mean"],
             metrics["baseline_serial_seconds_per_token_mean"],
             metrics["mtp_decode_speedup_pooled_ratio_of_means"]))
    print("parity_all_ok %s, pairs %d, decode tokens %d, distinct heads %d"
          % (metrics["parity_all_ok"], metrics["accepted_pair_count"],
             metrics["decode_tokens"], len(heads)))
    print("promotionStatus %s, promotedSourceRef %s, finished %s"
          % (new["promotionStatus"], new["promotedSourceRef"], new["promotionFinishedAt"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "harness": "ranked",
        "new": args.new,
        "old": args.old,
        "crown": args.crown,
        "official_score": score,
        "crown_score": crown["officialScore"],
        "old_score": old["officialScore"],
        "prereg": PREREG,
        "prereg_band": PREREG_BAND,
        "prereg_reference_candidate_mean": ref_mean,
        "observed_candidate_mean": got_mean,
        "observed_candidate_change": observed,
        "inside_prereg_band": inside,
        "refuted_at_claimed_size": refuted,
        "schedule_identical_to_reference": same_schedule,
        "per_prompt": [
            {"prompt": n, "width": w, "cand_gain_pct": c,
             "serial_delta_pct": s, "raw_gain_pct": r}
            for w, n, c, s, r in table
        ],
        "candidate_leg_routed_mean_pct": mean_routed,
        "candidate_leg_all_mean_pct": mean_all,
        "promotion": {
            "status": new["promotionStatus"],
            "source_ref": new["promotedSourceRef"],
            "finished_at": new["promotionFinishedAt"],
        },
    }, indent=1))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
