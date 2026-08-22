"""Validate the rational round-count recovery against the hard-coded ROUNDS
vector that `rankedcurve.py` uses for reference-schedule rows."""

from collections import Counter

from e128_rounds import load_rows, per_prompt, recover_rounds

ROUNDS = {
    "plutarch": 487,
    "drama": 252,
    "travel": 212,
    "beagle": 110,
    "republic": 93,
    "essays": 92,
    "medicine": 90,
    "botany": 81,
}
REF_BEAGLE = 4.381818181818182


def main():
    rows = load_rows()
    agree = Counter()
    mult = Counter()
    nref = 0
    shown = 0
    for r in rows:
        e = per_prompt(r)
        if len(e) != 8:
            continue
        if abs(e["beagle"]["effective_mean_draft_len"] - REF_BEAGLE) > 1e-9:
            continue
        nref += 1
        for name, entry in e.items():
            rr, m = recover_rounds(
                entry["effective_mean_draft_len"], entry["non_drafting_round_count"]
            )
            mult[m] += 1
            agree[(name, rr == ROUNDS[name])] += 1
            if shown < 8 and nref == 1:
                print(
                    f"  {name:10s} dl={entry['effective_mean_draft_len']!r:22s} "
                    f"n0={entry['non_drafting_round_count']:4d}  recovered={rr}  "
                    f"mult={m}  rankedcurve={ROUNDS[name]}"
                )
                shown += 1
    print(f"\nreference-schedule rows: {nref}")
    print("agreement with the rankedcurve ROUNDS vector:")
    for name in ROUNDS:
        ok = agree[(name, True)]
        bad = agree[(name, False)]
        print(f"  {name:10s} agree {ok:5d}   differ {bad:5d}")
    print("\nmultiplicity of the legal round window (1 = uniquely pinned):")
    for k, n in sorted(mult.items()):
        print(f"  mult={k}  n={n}")

    print("\nrecovery over ALL rows with eight prompts:")
    tot = Counter()
    for r in rows:
        e = per_prompt(r)
        if len(e) != 8:
            continue
        for name, entry in e.items():
            rr, m = recover_rounds(
                entry["effective_mean_draft_len"], entry["non_drafting_round_count"]
            )
            tot["total"] += 1
            if rr is None:
                tot["failed"] += 1
            elif m == 1:
                tot["unique"] += 1
            else:
                tot[f"mult{min(m,4)}"] += 1
    for k, v in sorted(tot.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
