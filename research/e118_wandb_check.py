#!/usr/bin/env python3
"""Check `e118_wandb_log.py` before and after it publishes.

Why this exists. `e118_wandb_log.py` writes six runs in one pass. When the
analysis script changed `cost_model.screen_prediction` from one dict per arm
to a dict of per-width readings, the logger raised `KeyError` on run four.
Runs one to three were already live and runs five and six never happened, so
the group was left half published and the failure only showed up after a
network round trip per run.

    --dry     builds every table against the committed summary.json with a
              stub `wandb`, so a schema drift fails in under a second and
              nothing is written to the server.
    --verify  reads the six runs back from the server and asserts that each
              reached `finished` and carries the summary keys a reader of
              the report would go looking for.

Neither mode is part of the experiment. They exist so that "the numbers are
in W&B" is a checked claim rather than an assumed one.
"""
from __future__ import annotations

import argparse
import sys
import types

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"

# The keys a reader arrives looking for, per run. Only the load-bearing ones:
# this is a smoke test, not a mirror of the logger.
EXPECTED = {
    "e118arms1": (
        "primary_metric_name",
        "primary_metric_arm",
        "e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base",
        "kill_rule_pct",
        "kill_rule_cleared",
        "kill_rule_cleared_anywhere_in_identified_set",
        "primary_metric_identified_local_lo",
        "primary_metric_identified_local_hi",
        "positive_control_failure_count",
        "discriminator_verdict",
    ),
    "e118stat1": (),
    "e118spil1": (),
    "e118cost1": (
        "pct_per_instruction_ld",
        "pct_per_instruction_alu",
        "pct_per_instruction_shuf",
        "us_per_instruction_ld",
        "shuffle_over_load_price_ratio",
        "ilp_four_minus_two_pct",
    ),
    "e118rng21": (
        "rung2_best_exact_round_weighted_pct",
        "rung2_bar_cleared",
        "rung2_standing_pct_excluding_local_spill",
        "rung2_excluding_local_spill_coverage",
    ),
    "e118hst01": (
        "sumshoist_round_weighted_pct",
        "capture_of_ceiling_weighted",
        "shippable_from_research",
        "excludes_table_production",
    ),
}


class _Table:
    """Stands in for `wandb.Table` and enforces the column contract."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = []

    def add_data(self, *values):
        if len(values) != len(self.columns):
            raise AssertionError(
                "row has %d values but the table declares %d columns %s"
                % (len(values), len(self.columns), self.columns))
        self.rows.append(values)


class _Run:
    def __init__(self, name):
        self.name = name
        self.summary = {}
        self.logged = {}

    def log(self, payload):
        self.logged.update(payload)

    def finish(self):
        pass


def dry_run() -> int:
    seen: list[_Run] = []

    def _init(**kwargs):
        run = _Run(kwargs.get("name"))
        seen.append(run)
        return run

    stub = types.ModuleType("wandb")
    stub.Table = _Table
    stub.init = _init
    sys.modules["wandb"] = stub

    sys.path.insert(0, "research")
    import e118_wandb_log as logger  # noqa: PLC0415

    # `start` resolves a deterministic run id and may touch the API; in a dry
    # run the only thing that matters is that it hands back a run object.
    logger.start = lambda job_type, name, config: _init(name=name)

    failures = 0
    for name, fn in logger.RUNS.items():
        print("==", name)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - the point is to report it
            failures += 1
            print("   FAILED: %s: %s" % (type(exc).__name__, exc))
            continue
        run = seen[-1]
        for key, val in sorted(run.logged.items()):
            if isinstance(val, _Table):
                print("   table %-34s %3d rows x %d cols"
                      % (key, len(val.rows), len(val.columns)))
                if not val.rows:
                    failures += 1
                    print("      EMPTY TABLE")
        missing = [k for k, v in sorted(run.summary.items()) if v is None]
        print("   summary keys: %d (%d null)"
              % (len(run.summary), len(missing)))
    print("DRY_RUN_FAILED" if failures else "DRY_RUN_OK")
    return 1 if failures else 0


def verify() -> int:
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    failures = 0
    for run_id, keys in EXPECTED.items():
        try:
            run = api.run("%s/%s" % (PROJECT, run_id))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("%-10s UNREACHABLE: %s" % (run_id, exc))
            continue
        tables = sorted(
            k for k, v in run.summary._json_dict.items()
            if isinstance(v, dict) and v.get("_type") == "table-file")
        print("%-10s state=%-9s tables=%d group=%s"
              % (run_id, run.state, len(tables), run.group))
        if run.state != "finished":
            failures += 1
            print("   NOT FINISHED")
        for key in keys:
            if key not in run.summary:
                failures += 1
                print("   MISSING %s" % key)
            else:
                print("   %-58s %s" % (key, run.summary[key]))
    print("VERIFY_FAILED" if failures else "VERIFY_OK")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true",
                      help="build every table offline against summary.json")
    mode.add_argument("--verify", action="store_true",
                      help="read the six runs back from W&B")
    args = ap.parse_args()
    return dry_run() if args.dry else verify()


if __name__ == "__main__":
    raise SystemExit(main())
