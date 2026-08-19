"""Extract the values_per_thread prior art from the ranked Yukon board.

Section (f) of the E36 report rests on this. Reproduce with:

    curl -H "Authorization: Bearer $YUKON_API_TOKEN" \
      https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions \
      -o /tmp/e36_board.json
    python3 research/e36_board_evidence.py /tmp/e36_board.json \
      research/e36-board-vpt-evidence.json

The raw response is ~8 MB and is deliberately not committed; this writes the
compact slice the campaign actually needs.
"""

import json
import re
import sys

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
TERM = "values_per_thread"

# The two archives that changed values_per_thread on the wide affine4/g64
# crossrow verify kernel and failed the official parity gate, plus the next
# accepted row where the same solver wrote down the lesson.
PARITY_FAILURES = ("5c74b78b", "6154a6f1")
LESSON_ROW = "11863aa9"


def excerpts(note, term=TERM, pad=420):
    out, seen = [], set()
    for m in re.finditer(term, note):
        start, end = max(0, m.start() - pad), min(len(note), m.end() + pad)
        bucket = start // (2 * pad)
        if bucket in seen:
            continue
        seen.add(bucket)
        out.append(re.sub(r"\s+", " ", note[start:end]).strip())
    return out


def main(src, dst):
    payload = json.load(open(src))
    rows = payload if isinstance(payload, list) else payload.get("submissions", [])
    hits = [r for r in rows if r.get("note") and TERM in r["note"]]
    by_short = {r["id"][:8]: r for r in rows}

    doc = {
        "source": f"GET https://api.yukon.org/api/benchmarks/{BENCHMARK_ID}/submissions",
        "captured_by": "E36 / qwen-askeladd",
        "board_rows_total": len(rows),
        "notes_mentioning_values_per_thread": len(hits),
        "submissions": [
            {
                "id": r["id"],
                "solver": r["solverUsername"],
                "created_at": r["createdAt"],
                "official_score": r.get("officialScore"),
                "status": r.get("status"),
                "promotion_status": r.get("promotionStatus"),
                "rejection_reason": r.get("rejectionReason"),
                "excerpts": excerpts(r["note"]),
            }
            for r in sorted(hits, key=lambda r: r["createdAt"])
        ],
        "verify_path_parity_failures": [
            {
                "id": by_short[s]["id"],
                "solver": by_short[s]["solverUsername"],
                "created_at": by_short[s]["createdAt"],
                "status": by_short[s]["status"],
                "rejection_reason": by_short[s].get("rejectionReason"),
                "note_head": re.sub(r"\s+", " ", by_short[s]["note"][:700]),
            }
            for s in PARITY_FAILURES
        ],
    }

    lesson = by_short[LESSON_ROW]
    match = re.search(r".{600}verify reduction tree.{120}", lesson["note"], re.S)
    doc["lesson_recorded_by_next_accepted_row"] = {
        "id": lesson["id"],
        "solver": lesson["solverUsername"],
        "official_score": lesson["officialScore"],
        "status": lesson["status"],
        "promotion_status": lesson["promotionStatus"],
        "excerpt": re.sub(r"\s+", " ", match.group(0)) if match else None,
    }

    with open(dst, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(
        f"{dst}: {len(hits)} {TERM} notes of {len(rows)} board rows, "
        f"{len(doc['verify_path_parity_failures'])} verify-path parity failures"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
