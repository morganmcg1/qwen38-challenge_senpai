#!/usr/bin/env python3
"""Patch the finished E202 run's verdict label and census summary in place.

The session run `lpwbno36` was published with the reducer's pre-ruling verdict
string `overlap-confirmed`. The advisor ruling on PR 199 (comment 5402432090)
withdraws that label: `inner` bounds exposed interior latency from ABOVE, so a
large `inner` cannot separate overlap from marginal synchronisation cost. This
rewrites the summary fields of the SAME run rather than publishing a new one,
so the evidence pointer in the terminal result stays valid. No metric value is
touched.
"""
import argparse
import json
import pathlib

import wandb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--project", default="qwen38-mlx-challenge-senpai")
    ap.add_argument("--run", default="lpwbno36")
    ap.add_argument("--report", type=pathlib.Path,
                    default=pathlib.Path("research/e202-session.json"))
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run}")

    before = run.summary.get("verdict")
    run.summary["verdict"] = report["verdict"]
    run.summary["verdict_superseded"] = before
    run.summary["verdict_ruling"] = (
        "PR 199 comment 5402432090: inner is an upper bound, so a large inner "
        "does not uniquely identify overlap"
    )
    run.summary["decision_threshold_ms_per_round"] = 1.6
    run.summary["report"] = json.dumps(report)
    for width, calls in report["session_split_call_census_by_width"].items():
        run.summary[f"census/steady_calls_qL{width}"] = calls
    for width, calls in report[
        "session_split_call_census_first_128_calls_per_leg"
    ].items():
        run.summary[f"census/first128_calls_qL{width}"] = calls
    run.summary.update()

    print(f"e202_wandb_patch: {args.run} verdict {before!r} -> "
          f"{report['verdict']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
