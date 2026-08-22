"""Board-side depth scan: does the shipped draft depth win on the real board?

E128 prices counterfactual schedulers inside our own model. This script is the
independent check. It reads every public Yukon receipt, groups the per-prompt
rows by `effective_mean_draft_len`, and reports the best raw ratio reached at
each depth. Our model plays no part in it.

    python3 research/e128_board_depth_scan.py \
        --board /tmp/yukon-board/full.json \
        --json research/e128-artifacts/board-depth-scan.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# F92 published shipped depth and F83 ranked-median weight per prompt.
SHIPPED_DEPTH = {
    "beagle": 4.382, "medicine": 5.256, "essays": 5.087, "botany": 6.148,
    "republic": 4.989, "drama": 2.298, "travel": 2.656, "plutarch": 0.154,
}
WEIGHT = {
    "beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598, "botany": 0.0124,
    "republic": 0.0100, "drama": 0.0, "travel": 0.0, "plutarch": 0.0,
}


def load_rows(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text())
    subs = raw["submissions"] if isinstance(raw, dict) else raw
    rows: dict[str, list[dict]] = defaultdict(list)
    for sub in subs:
        metrics = sub.get("officialMetrics") or {}
        for entry in metrics.get("per_prompt") or []:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
            if name is None:
                continue
            rows[name].append({
                "raw": entry["raw_ratio_of_means"],
                "draft_len": entry["effective_mean_draft_len"],
                "id": sub["id"][:8],
                "status": sub["status"],
                "solver": sub["solverUsername"],
                "score": sub["officialScore"],
            })
    return rows


def scan(rows: list[dict], shipped_depth: float, top_n: int) -> dict:
    by_depth: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_depth[round(row["draft_len"], 3)].append(row)

    depths = []
    for depth, group in by_depth.items():
        best = max(group, key=lambda r: r["raw"])
        depths.append({
            "draft_len": depth,
            "rows": len(group),
            "best_raw": best["raw"],
            "best_id": best["id"],
            "best_solver": best["solver"],
            "median_raw": statistics.median(r["raw"] for r in group),
        })
    depths.sort(key=lambda d: -d["best_raw"])

    ranked = sorted(rows, key=lambda r: -r["raw"])
    key = round(shipped_depth, 3)
    at_shipped = [d for d in depths if d["draft_len"] == key]
    others = [d for d in depths if d["draft_len"] != key]
    lead = None
    if at_shipped and others:
        lead = (at_shipped[0]["best_raw"] / others[0]["best_raw"] - 1.0) * 100.0

    # How deep does the shipped depth's run of top rows go before another
    # depth appears? This is the claim that is easiest to falsify.
    run = 0
    for row in ranked:
        if round(row["draft_len"], 3) != key:
            break
        run += 1

    return {
        "total_rows": len(rows),
        "distinct_draft_lens": len(by_depth),
        "shipped_draft_len": shipped_depth,
        "rows_at_shipped_draft_len": len(by_depth.get(key, [])),
        "top_row_run_at_shipped_draft_len": run,
        "best_raw_at_shipped": at_shipped[0]["best_raw"] if at_shipped else None,
        "best_raw_at_any_other_depth": others[0]["best_raw"] if others else None,
        "shipped_lead_pct": lead,
        "top_rows": ranked[:top_n],
        "by_draft_len": depths,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = load_rows(args.board)
    out = {"harness": "ranked", "source": str(args.board), "prompts": {}}
    order = sorted(rows, key=lambda p: -WEIGHT[p])
    for prompt in order:
        group = rows[prompt]
        result = scan(group, SHIPPED_DEPTH[prompt], args.top)
        result["f83_weight"] = WEIGHT[prompt]
        out["prompts"][prompt] = result

        print("\n=== %s (F83 weight %.4f): %d rows, %d distinct draft lengths,"
              " shipped %.3f" % (
                  prompt, WEIGHT[prompt], result["total_rows"],
                  result["distinct_draft_lens"], result["shipped_draft_len"]))
        print("leading run of top rows at shipped depth: %d" %
              result["top_row_run_at_shipped_draft_len"])
        print("best at shipped %.4f   best elsewhere %.4f   lead %+.2f%%" % (
            result["best_raw_at_shipped"], result["best_raw_at_any_other_depth"],
            result["shipped_lead_pct"]))
        print("%10s %6s %10s %11s  %s" % (
            "draft_len", "rows", "best raw", "median raw", "best row"))
        for entry in result["by_draft_len"][:8]:
            mark = " <- shipped" if entry["draft_len"] == round(
                result["shipped_draft_len"], 3) else ""
            print("%10.3f %6d %10.4f %11.4f  %s %s%s" % (
                entry["draft_len"], entry["rows"], entry["best_raw"],
                entry["median_raw"], entry["best_id"], entry["best_solver"],
                mark))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
