#!/usr/bin/env python3
"""Collect the E87 arm-C liveness control into one record.

    usage: research/e87_liveness_report.py TAG [TAG ...] --out FILE

The control passes when the damaged arm, whose `draft_cluster.perm` is
reversed, drops the accepted-draft rate to zero while the undamaged arms hold
theirs. That is the only evidence that the cluster index, and not a silent
fall-back to the dense readout, produced the timed drafts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
FIELDS = ("decode_tokens", "effective_mean_draft_len", "accepted_draft_rate",
          "all_tokens_matched", "mtp_seconds_per_token",
          "serial_seconds_per_token")


def leg(tag: str) -> dict:
    meta = dict(
        line.split("=", 1)
        for line in (OUT / tag / "meta.txt").read_text().splitlines()
        if "=" in line)
    metrics = json.loads((OUT / tag / "score.json").read_text())["metrics"]
    return {"tag": tag, "arm": meta["e87_arm"], "head_dir": meta["head_dir"],
            "worker_sha256": meta["worker_sha256"],
            **{k: metrics.get(k) for k in FIELDS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = [leg(t) for t in args.tags]
    damaged = [a for a in arms if a["arm"].endswith("-damaged")]
    intact = [a for a in arms if not a["arm"].endswith("-damaged")]
    passed = (bool(damaged) and bool(intact)
              and all(a["accepted_draft_rate"] == 0 for a in damaged)
              and all(a["accepted_draft_rate"] > 0.9 for a in intact)
              and all(a["all_tokens_matched"] for a in arms))

    report = {
        "control": "reversed draft_cluster.perm",
        "harness": "local",
        "official_or_ranked_score": False,
        "passed": passed,
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for a in arms:
        print(f"{a['arm']:<16} d={a['effective_mean_draft_len']:<8} "
              f"acc={a['accepted_draft_rate']:<8} "
              f"matched={a['all_tokens_matched']} "
              f"mtp={a['mtp_seconds_per_token']:.6f}")
    print(f"liveness control passed: {passed}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
