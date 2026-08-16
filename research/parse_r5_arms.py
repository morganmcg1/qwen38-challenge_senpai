"""Summarize depth/cap schedule for the two r5 arms and compare with r4 predictions."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?streak_in=(\d+) cap=(\d+)")

ARMS = {
    "control": ("r5-control-cap8-gate3-512", 8),
    "winner": ("r5-winner-cap7-gate3-512", 7),
}

TIMED_KEYS = (
    "declared_rows_total",
    "reference_checked_row_total",
    "effective_mean_draft_len",
    "effective_max_draft_len",
    "accepted_draft_total",
    "rejected_draft_total",
    "accepted_draft_rate",
    "round_count",
    "seed_token_count",
    "decode_token_count",
    "decode_seconds",
    "first_block_seconds",
    "p50_block_request_seconds_after_first",
    "max_block_request_seconds_after_first",
    "non_drafting_round_count",
    "verify_block_replayed_round_count",
    "all_tokens_matched",
    "residual_divergence_count",
    "parity_all_ok",
    "uses_pinned_mtp_head",
    "max_rejected_tail_logit_delta",
)

SCORE_KEYS = (
    "mtp_decode_speedup",
    "mtp_seconds_per_token",
    "serial_seconds_per_token",
    "public_drift_tripwire_passed",
)


def summarize(trace_path):
    depth = Counter()
    cap = Counter()
    rounds = accepted = rejected = 0
    for line in trace_path.read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        _, d, acc, _, c = (int(x) for x in m.groups())
        rounds += 1
        depth[d] += 1
        cap[c] += 1
        accepted += acc
        rejected += d - acc
    return {
        "rounds": rounds,
        "depth_histogram": dict(sorted(depth.items())),
        "cap_histogram": dict(sorted(cap.items())),
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "mean_draft_len": sum(d * n for d, n in depth.items()) / rounds if rounds else None,
    }


out = {}
for name, (tag, cap) in ARMS.items():
    summary = {"tag": tag, "segmented_verify_depth_cap": cap, "segmented_streak_gate": 3}
    summary.update(summarize(ROOT / f"trace-{tag}.log"))
    timed = json.loads((ROOT / f"capture-{tag}" / "04-mtp-timed.json").read_text())
    score = json.loads((ROOT / f"score-{tag}.json").read_text())["metrics"]
    summary.update({k: timed[k] for k in TIMED_KEYS if k in timed})
    summary.update({k: score[k] for k in SCORE_KEYS if k in score})
    summary["head_provenance_sha256"] = timed.get("head_provenance", {}).get("sha256")
    out[name] = summary

pred = {
    "depth_histogram": {3: 1, 4: 29, 5: 2, 6: 3, 7: 46},
    "cap_histogram": {4: 29, 7: 52},
    "rounds": 81,
    "accepted_draft_tokens": 431,
    "rejected_draft_tokens": 38,
    "effective_mean_draft_len": 5.790123456790123,
    "declared_rows_total": 550,
}
w = out["winner"]
checks = {k: (pred[k] == w.get(k)) for k in pred}
out["winner_vs_prediction"] = {"prediction": pred, "matches": checks, "all_match": all(checks.values())}

c, wn = out["control"], out["winner"]
out["winner_vs_control"] = {
    "seconds_per_token_pct": (wn["mtp_seconds_per_token"] / c["mtp_seconds_per_token"] - 1) * 100,
    "ratio_pct": (wn["mtp_decode_speedup"] / c["mtp_decode_speedup"] - 1) * 100,
}

(ROOT / "r5-arm-summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
json.dump(out, sys.stdout, indent=2, sort_keys=True)
print()
