#!/usr/bin/env python3
"""Read one ranked receipt against named reference receipts (E135 F38/F39 §6).

Reports, per reference:

  * the candidate-leg delta on the two prompts that occupy the median pair,
    which is the headline under the campaign rules;
  * the 8-prompt candidate-leg mean, sd and se;
  * the F83 weighted-five mean.

Also reports the prefill delta against one named reference and the Rule 133
schedule read (plutarch first, then drama), which is the Rule 114 confirmation
that the intended depth-price arm reached the submitted binary.

Sign convention: POSITIVE means OURS IS FASTER.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import urllib.request

BOARD_URL = (
    "https://api.yukon.org/api/benchmarks/"
    "5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true"
)

# fixtures/qwen3_8_27b_mtp_track.json prompt digests, named for reading.
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}

# F83 median-pair influence weights. The three prompts absent from this map
# carry no weight in the published median (Rule 70).
F83_WEIGHTS = {
    "beagle": 0.4862,
    "medicine": 0.2508,
    "essays": 0.1598,
    "botany": 0.0124,
    "republic": 0.0100,
}

# Rule 148. At a published median of 3.5 or above, the median pair is drawn
# from a much narrower set than F83 assumed: beagle always holds the lower
# slot, and essays holds the upper slot in nine cases out of ten. F83 remains
# below for contrast, because it is the weight vector most of the campaign
# ledger was priced under.
RULE148_WEIGHTS = {
    "beagle": 0.5000,
    "essays": 0.4474,
    "republic": 0.0329,
    "medicine": 0.0197,
}

# Rule 147. Per-prompt candidate-channel detectability, in microseconds per
# round at two standard deviations on ONE receipt. A per-prompt candidate delta
# below its own entry here is inside that prompt's single-receipt noise.
RULE147_DETECT_US_PER_ROUND = {
    "essays": 44.5,
    "medicine": 66.9,
    "republic": 70.9,
    "beagle": 119.5,
    "drama": 123.8,
    "botany": 129.2,
    "travel": 150.1,
    "plutarch": 211.9,
}

# F216 noise floors, in percentage points.
PER_PROMPT_LEG_NOISE_PP = 0.0737
AT_ZERO_MDE_PP = 0.1154
PREFILL_NULL_BAND_PCT = 0.15


def load_board(path: str | None) -> list[dict]:
    if path and os.path.exists(path):
        raw = json.load(open(path))
    else:
        token = os.environ["YUKON_API_TOKEN"]
        req = urllib.request.Request(
            BOARD_URL, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=120) as fh:
            raw = json.load(fh)
        if path:
            json.dump(raw, open(path, "w"))
    if isinstance(raw, list):
        return raw
    return raw.get("submissions", raw.get("data", []))


def find(board: list[dict], prefix: str) -> dict:
    hits = [r for r in board if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{prefix}: matched {len(hits)} rows, need exactly 1")
    return hits[0]


def prompt_name(entry: dict) -> str:
    for key in ("prompt_sha256", "prompt_sha", "prompt_id", "prompt"):
        value = entry.get(key)
        if isinstance(value, str):
            head = value[:8]
            if head in PROMPT_NAMES:
                return PROMPT_NAMES[head]
            return head
    raise SystemExit(f"cannot name prompt from {sorted(entry)}")


def per_prompt(row: dict) -> dict[str, dict]:
    metrics = row.get("officialMetrics") or {}
    entries = metrics.get("per_prompt") or []
    if not entries:
        raise SystemExit(f"{row['id'][:8]}: no per_prompt block")
    return {prompt_name(e): e for e in entries}


def summarise(deltas: dict[str, float]) -> tuple[float, float, float]:
    values = list(deltas.values())
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, se


def median_pair(row: dict) -> tuple[str, str]:
    """The two prompts at sorted ranks 3 and 4 of the published raw ratios."""
    ranked = sorted(
        per_prompt(row).items(), key=lambda kv: kv[1]["raw_ratio_of_means"]
    )
    return ranked[3][0], ranked[4][0]


def compare(ours: dict, ref: dict, field: str) -> dict[str, float]:
    a, b = per_prompt(ours), per_prompt(ref)
    shared = sorted(set(a) & set(b))
    out = {}
    for name in shared:
        ref_value = b[name][field]
        out[name] = (ref_value - a[name][field]) / ref_value * 100.0
    return out


def candidate_us_per_round(row: dict) -> dict[str, float]:
    """Microseconds of candidate MTP time per decode round, by prompt.

    The receipt publishes no round count, so rounds are taken as
    `decode_tokens / (1 + effective_mean_draft_len)`. That is the campaign
    convention behind Rule 134 and Rule 147: for the beagle leg it reproduces
    `BEAGLE_TOKENS_PER_ROUND = 5.3818` exactly. It treats every draft as
    accepted, so it understates the round count and overstates microseconds per
    round. It is used here only to compare against the Rule 147 table, which
    was built on the same convention, so the bias cancels in that comparison.
    """
    tokens = (row.get("officialMetrics") or {}).get("decode_tokens") or 512
    return {
        name: e["mtp_seconds_per_token_mean"] * 1e6
        * (1.0 + e["effective_mean_draft_len"])
        for name, e in per_prompt(row).items()
    } if tokens else {}


def weighted(deltas: dict[str, float], weights: dict[str, float]) -> float:
    live = [n for n in weights if n in deltas]
    return sum(weights[n] * deltas[n] for n in live) / sum(
        weights[n] for n in live)


def report_reference(ours: dict, ref: dict, label: str) -> None:
    deltas = compare(ours, ref, "mtp_seconds_per_token_mean")
    lo, hi = median_pair(ours)
    mean, sd, se = summarise(deltas)
    us_round = candidate_us_per_round(ours)

    print(f"\n### e135_ranked_cand_w5_vs_{label}  (ref median {ref['officialScore']:.8f})")
    print(f"  median pair, OUR sorted ranks 3 and 4: {lo}, {hi}")
    print(f"    {lo:<9} {deltas.get(lo, float('nan')):+9.4f} %")
    print(f"    {hi:<9} {deltas.get(hi, float('nan')):+9.4f} %")
    pair = [deltas[n] for n in (lo, hi) if n in deltas]
    if len(pair) == 2:
        print(f"    pair mean {statistics.fmean(pair):+9.4f} %")
    print(f"  8-prompt mean {mean:+.4f} %  sd {sd:.4f}  se {se:.4f}")
    print(f"  Rule 148 weighted four {weighted(deltas, RULE148_WEIGHTS):+.4f} %")
    print(f"  F83 weighted five      {weighted(deltas, F83_WEIGHTS):+.4f} %")
    same = sum(1 for v in deltas.values() if v > 0)
    print(f"  sign test: {same}/{len(deltas)} prompts faster")
    print("  per prompt, against the Rule 147 candidate-channel floor:")
    print("    prompt        delta      us/round   R147 2sd   detected  r148")
    for name in sorted(deltas, key=lambda n: -deltas[n]):
        moved = abs(deltas[name]) / 100.0 * us_round.get(name, float("nan"))
        floor = RULE147_DETECT_US_PER_ROUND.get(name, float("nan"))
        mark = "yes" if moved >= floor else "no"
        print(f"    {name:<9} {deltas[name]:+9.4f} % {moved:9.1f} "
              f"{floor:10.1f} {mark:>10}  {RULE148_WEIGHTS.get(name, 0.0):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", help="our receipt id prefix")
    ap.add_argument(
        "--vs",
        action="append",
        default=[],
        help="reference receipt id prefix; repeat, tightest first",
    )
    ap.add_argument("--prefill-vs", default="684821ed")
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    args = ap.parse_args()

    board = load_board(args.board)
    ours = find(board, args.ours)

    print(f"OUR RECEIPT {ours['id'][:8]}  status={ours.get('status')} "
          f"promotion={ours.get('promotionStatus')} "
          f"score={ours.get('officialScore')}")
    print(f"commit {ours.get('submissionCommitSha')}")

    print("\n## Rule 133 schedule read: plutarch first, then drama")
    mine = per_prompt(ours)
    for name in ("plutarch", "drama"):
        entry = mine.get(name)
        if entry is None:
            print(f"  {name}: absent")
            continue
        print(f"  {name:<9} edl {entry['effective_mean_draft_len']:.4f}  "
              f"non_drafting_rounds {entry['non_drafting_round_count']}  "
              f"raw {entry['raw_ratio_of_means']:.4f}")
    print("  ship signature: plutarch edl near 0.155 with about 449 "
          "non-drafting rounds")
    print("  pb6 signature:  plutarch edl near 2.70 with zero "
          "non-drafting rounds")

    for label in args.vs:
        report_reference(ours, find(board, label), label)

    ref = find(board, args.prefill_vs)
    prefill = compare(ours, ref, "prefill_seconds_per_token")
    mean, sd, se = summarise(prefill)
    print(f"\n### e135_prefill_pct_vs_{args.prefill_vs}")
    print(f"  8-prompt mean {mean:+.4f} %  sd {sd:.4f}  se {se:.4f}")
    verdict = "INSIDE" if abs(mean) <= PREFILL_NULL_BAND_PCT else "OUTSIDE"
    print(f"  null band +/-{PREFILL_NULL_BAND_PCT:.2f} %: {verdict}")
    print(f"  (per-prompt leg noise {PER_PROMPT_LEG_NOISE_PP:.4f} pp, "
          f"at-zero MDE {AT_ZERO_MDE_PP:.4f} pp)")


if __name__ == "__main__":
    main()
