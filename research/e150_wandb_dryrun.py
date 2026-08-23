#!/usr/bin/env python3
"""Run research/e150_wandb_log.py against a stub W&B and check what it emits.

WHY THIS EXISTS. A publisher defect costs a whole real run to discover, and the
run is already published by the time the traceback appears. Two defects were
found this way and neither would have raised locally: a summary key that read
a schema the artifact no longer had, so it silently logged None; and a nested
dict written into a table cell, which W&B accepts and then renders as nothing.

WHAT IT CHECKS, beyond "it did not crash":

1.  No summary value is a container. A dict or list in `run.summary` is a
    silent data loss in the W&B UI.
2.  No table cell is a container, for the same reason.
3.  Every table row has exactly as many cells as the table has columns. W&B
    truncates or pads instead of raising.
4.  Reports which summary keys are None, since a stale key schema reads None
    rather than failing.

It imports the real publisher, so it exercises the real code path and cannot
drift from it.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent


class StubTable:
    def __init__(self, columns=None, data=None, **_kw):
        self.columns = list(columns or [])
        self.data = [list(r) for r in (data or [])]


class StubRun:
    def __init__(self):
        self.id = "dryrun"
        self.url = "https://wandb.local/dryrun"
        self.summary: dict = {}
        self.logged: dict = {}

    def log(self, payload):
        self.logged.update(payload)

    def finish(self):
        pass


def install_stub() -> StubRun:
    run = StubRun()
    mod = types.ModuleType("wandb")
    mod.Table = StubTable
    mod.init = lambda **_kw: run
    mod.run = run
    sys.modules["wandb"] = mod
    return run


def main() -> int:
    run = install_stub()
    sys.path.insert(0, str(HERE))
    publisher = importlib.import_module("e150_wandb_log")
    sys.argv = ["e150_wandb_log", "--run-name", "dryrun"]
    rc = publisher.main()

    problems: list[str] = []
    for key, value in run.summary.items():
        if isinstance(value, (dict, list, tuple, set)):
            problems.append(
                f"summary[{key!r}] is a {type(value).__name__}; W&B drops it")

    for name, tbl in run.logged.items():
        if not isinstance(tbl, StubTable):
            continue
        ncol = len(tbl.columns)
        for i, row in enumerate(tbl.data):
            if len(row) != ncol:
                problems.append(
                    f"table {name!r} row {i} has {len(row)} cells,"
                    f" {ncol} columns")
            for j, cell in enumerate(row):
                if isinstance(cell, (dict, list, tuple, set)):
                    problems.append(
                        f"table {name!r} row {i} column"
                        f" {tbl.columns[j] if j < ncol else j!r} is a"
                        f" {type(cell).__name__}; W&B renders it as nothing")

    nones = sorted(k for k, v in run.summary.items() if v is None)

    print("-- e150 wandb dry run --")
    print(f"  publisher rc      {rc}")
    print(f"  summary keys      {len(run.summary)}")
    print(f"  tables            {len(run.logged)}")
    for name, tbl in sorted(run.logged.items()):
        if isinstance(tbl, StubTable):
            print(f"    {name:34s} {len(tbl.columns)} cols"
                  f" x {len(tbl.data)} rows")
    print(f"  summary keys that are None ({len(nones)}):")
    for k in nones:
        print(f"    {k}")
    if problems:
        print(f"  PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"    {p}")
    else:
        print("  no shape or container problems")
    return 1 if problems or rc != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
