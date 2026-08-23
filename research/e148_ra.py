"""E148 R-A: the six-prompt cohort structure of the whole scored board.

harness=ranked, zero GPU.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import e146_lib as L
import e148_lib as E

OUT = os.path.join(HERE, "e148-ra.json")


def main():
    path, rows = E.load_rows()
    by_id = {r.id8: r for r in rows}
    all_groups, cohorts = E.build_cohorts(rows)

    # The eight-prompt key is the F234 comparison, not the instrument.
    eight = {}
    for row in rows:
        eight.setdefault(row.draft_key(), []).append(row)

    classified = sum(len(v) for v in cohorts.values())
    unclassifiable = len(rows) - classified

    print("board %s" % path)
    print("scored rows with a complete eight-prompt receipt: %d" % len(rows))
    print("")
    print("e148_cohort_count                %d" % len(cohorts))
    print("e148_rows_classified             %d" % classified)
    print("e148_rows_unclassifiable         %d" % unclassifiable)
    print("cohorts of any size (incl. singletons) %d" % len(all_groups))
    print("eight-prompt cohorts of any size       %d  (F234 comparison)"
          % len(eight))

    ranked = sorted(cohorts.items(), key=lambda kv: -len(kv[1]))
    print("\n=== the five largest cohorts ===")
    print("%-4s %6s %10s %10s %10s %10s  %s"
          % ("#", "n", "best", "worst", "median", "span_h", "example"))
    top = []
    for i, (key, members) in enumerate(ranked[:5]):
        scores = sorted(m.score for m in members if m.score is not None)
        hours = [L.PROMPT_ORDER and 0 for _ in members]
        first, last = members[0].created, members[-1].created
        entry = {
            "rank": i + 1, "n": len(members),
            "best_score": scores[-1] if scores else None,
            "worst_score": scores[0] if scores else None,
            "median_score": scores[len(scores) // 2] if scores else None,
            "first_created": first, "last_created": last,
            "example_id": members[0].id8,
            "draft_lengths": {p: members[0].dlen(p) for p in E.COHORT_PROMPTS},
            "non_drafting": {p: members[0].non_drafting(p) for p in E.COHORT_PROMPTS},
        }
        top.append(entry)
        print("%-4d %6d %10.5f %10.5f %10.5f %10s  %s"
              % (i + 1, len(members), entry["best_score"], entry["worst_score"],
                 entry["median_score"], first[5:10] + ">" + last[5:10],
                 members[0].id8))

    # What the six-prompt key bought over the eight-prompt key.
    merged = 0
    for key, members in cohorts.items():
        sub = set(m.draft_key() for m in members)
        if len(sub) > 1:
            merged += 1
    print("\nF234 check: %d six-prompt cohorts contain more than one "
          "eight-prompt schedule key" % merged)
    print("largest eight-prompt cohorts: %s"
          % ", ".join(str(len(v)) for v in
                      sorted(eight.values(), key=len, reverse=True)[:5]))
    print("largest six-prompt cohorts:   %s"
          % ", ".join(str(len(v)) for v in
                      sorted(cohorts.values(), key=len, reverse=True)[:5]))

    # Which cohorts hold the rows the campaign cares about.
    print("\n=== where the named rows sit ===")
    named = ["684821ed", "572b2cc4", "1db9d63e", "43925f29", "5cdc9c17",
             "e7770562", "f7d59543", "7e5172fa", "3d75f016", "e003a86d"]
    placement = {}
    for rid in named:
        row = by_id.get(rid)
        if row is None:
            print("%-9s not on the scored board" % rid)
            placement[rid] = None
            continue
        key = E.cohort_key(row)
        size = len(all_groups[key])
        placement[rid] = {"cohort_size": size,
                          "classified": size >= E.MIN_COHORT,
                          "score": row.score, "solver": row.solver}
        print("%-9s cohort n=%-4d %-14s score %s"
              % (rid, size, row.solver[:14],
                 "%.5f" % row.score if row.score else "none"))

    # Parent attribution coverage: this is what R-C and R-D depend on.
    trees = E.tree_index(rows)
    kinds = {}
    for row in rows:
        _, kind = E.resolve_parent(row, trees, by_id, cohorts)
        kinds[kind] = kinds.get(kind, 0) + 1
    print("\n=== parent attribution over the %d scored rows ===" % len(rows))
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print("  %-24s %4d" % (kind, count))

    payload = {
        "harness": "ranked", "board_path": path,
        "e148_scored_rows": len(rows),
        "e148_cohort_count": len(cohorts),
        "e148_rows_classified": classified,
        "e148_rows_unclassifiable": unclassifiable,
        "e148_cohorts_any_size": len(all_groups),
        "e148_eight_prompt_cohorts": len(eight),
        "e148_cohorts_merged_by_six_prompt_key": merged,
        "min_cohort_size": E.MIN_COHORT,
        "cohort_prompts": E.COHORT_PROMPTS,
        "largest_five": top,
        "named_row_placement": placement,
        "parent_attribution": kinds,
    }
    E.dump(OUT, payload)


if __name__ == "__main__":
    main()
