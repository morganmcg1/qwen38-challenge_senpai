#!/usr/bin/env python3
"""E124 stage 0.5: read the acceptance regime of each candidate seed.

Reads the shipped-schedule legs written by research/e122_rung0_session.sh and
reports, per seed, the numbers F92 needs to stratify stage 1:

  rounds, mean_depth, mean_accepted, accept_rate_of_drafted,
  the position histogram of the first rejected draft,
  the accept rate over the first 128 versus the last 128 decoded tokens,
  and the longest repeated token n-gram in the continuation.

The last two are the degeneration diagnostics: a steep first-to-last climb or
a long repeated run means a high accept rate is greedy-decode degeneration,
not a property of the seed.

  python3 research/e124_regime.py --runs-dir .mlxfast-private/e122/runs-e124
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)\b")
SPLIT_TOKENS = 128


def read_trace(path: Path) -> list[tuple[int, int]]:
    """Return [(drafted, accepted)] in round order."""
    rounds: list[tuple[int, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND_RE.match(line)
        if m:
            rounds.append((int(m.group(2)), int(m.group(3))))
    return rounds


def longest_repeated_ngram(ids: list[int], cap: int = 64) -> int:
    """Largest n <= cap for which some n-gram of `ids` occurs at least twice."""
    best = 0
    for n in range(1, min(cap, len(ids) // 2) + 1):
        seen = set()
        hit = False
        for i in range(len(ids) - n + 1):
            g = tuple(ids[i : i + n])
            if g in seen:
                hit = True
                break
            seen.add(g)
        if not hit:
            break
        best = n
    return best


def longest_immediate_repeat(ids: list[int], cap: int = 32) -> int:
    """Longest block that is immediately repeated back to back."""
    best = 0
    for n in range(1, min(cap, len(ids) // 2) + 1):
        for i in range(len(ids) - 2 * n + 1):
            if ids[i : i + n] == ids[i + n : i + 2 * n]:
                best = n
                break
    return best


def analyse(seed_id: str, run_dir: Path, golden: Path | None) -> dict:
    report = json.loads((run_dir / "report.json").read_text())
    rounds = read_trace(run_dir / "trace.txt")
    drafted = sum(d for d, _ in rounds)
    accepted = sum(a for _, a in rounds)

    first_reject = Counter()
    for d, a in rounds:
        first_reject["none" if a == d else str(a)] += 1

    # Bucket rounds by the decode position at which they start.
    pos = 0
    head_d = head_a = tail_d = tail_a = 0
    total_tokens = sum(a + 1 for _, a in rounds)
    tail_start = max(0, total_tokens - SPLIT_TOKENS)
    for d, a in rounds:
        if pos < SPLIT_TOKENS:
            head_d += d
            head_a += a
        if pos >= tail_start:
            tail_d += d
            tail_a += a
        pos += a + 1

    out = {
        "seed_id": seed_id,
        "rounds": report["round_count"],
        "decode_tokens": report["decode_token_count"],
        "mean_depth": report["effective_mean_draft_len"],
        "mean_accepted": accepted / len(rounds) if rounds else None,
        "accept_rate_of_drafted": report["accepted_draft_rate"],
        "accept_rate_of_drafted_from_trace": accepted / drafted if drafted else None,
        "accepted_draft_total": report["accepted_draft_total"],
        "rejected_draft_total": report["rejected_draft_total"],
        "all_tokens_matched": report["all_tokens_matched"],
        "residual_divergence_count": report["residual_divergence_count"],
        "trace_rounds": len(rounds),
        "first_rejection_position_histogram": dict(sorted(first_reject.items())),
        "accept_rate_first_128": head_a / head_d if head_d else None,
        "accept_rate_last_128": tail_a / tail_d if tail_d else None,
    }
    if golden is not None and golden.exists():
        emitted = json.loads(golden.read_text())["emitted_tokens"]
        out["longest_repeated_ngram_tokens"] = longest_repeated_ngram(emitted)
        out["longest_immediate_repeat_tokens"] = longest_immediate_repeat(emitted)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=".mlxfast-private/e122/runs-e124")
    ap.add_argument("--goldens-dir", default=".mlxfast-private/e122/goldens")
    ap.add_argument("--manifest", default="research/e124-corpus-manifest.json")
    ap.add_argument("--extra", nargs="*", default=[], help="ids from another runs dir")
    ap.add_argument("--extra-runs-dir", default=".mlxfast-private/e122/runs")
    ap.add_argument("--out", default="research/out/e124-regime.json")
    ap.add_argument("--keep-threshold", type=float, default=0.80)
    args = ap.parse_args()

    domains = {s["id"]: s["domain"] for s in json.loads(Path(args.manifest).read_text())["seeds"]}
    goldens = Path(args.goldens_dir)
    rows = []
    for seed_id, root in [(i, Path(args.runs_dir)) for i in sorted(domains)] + [
        (i, Path(args.extra_runs_dir)) for i in args.extra
    ]:
        run_dir = root / seed_id
        if not (run_dir / "report.json").exists():
            print(f"{seed_id:20s} MISSING {run_dir}")
            continue
        row = analyse(seed_id, run_dir, goldens / f"{seed_id}-rows-513.json")
        row["domain"] = domains.get(seed_id, "reference")
        row["stratum"] = "H" if row["accept_rate_of_drafted"] >= args.keep_threshold else "L"
        rows.append(row)

    rows.sort(key=lambda r: -r["accept_rate_of_drafted"])
    result = {
        "keep_threshold": args.keep_threshold,
        "split_tokens": SPLIT_TOKENS,
        "stratum_h_count": sum(1 for r in rows if r["stratum"] == "H"),
        "seeds": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    hdr = f"{'seed':20s} {'domain':9s} {'rnds':>5s} {'depth':>6s} {'acc/rnd':>8s} {'accept':>7s} {'f128':>6s} {'l128':>6s} {'rep':>4s} S"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['seed_id']:20s} {r['domain']:9s} {r['rounds']:5d} "
            f"{r['mean_depth']:6.3f} {r['mean_accepted']:8.3f} "
            f"{r['accept_rate_of_drafted']:7.4f} "
            f"{(r['accept_rate_first_128'] or 0):6.3f} {(r['accept_rate_last_128'] or 0):6.3f} "
            f"{r.get('longest_repeated_ngram_tokens', -1):4d} {r['stratum']}"
        )
    print(f"\nstratum H (>= {args.keep_threshold}): {result['stratum_h_count']} seeds -> {out}")


if __name__ == "__main__":
    main()
