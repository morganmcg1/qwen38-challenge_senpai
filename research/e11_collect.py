#!/usr/bin/env python3
"""Research-only (qwen38-r1-e11-depth-lever-showdown): assemble the per-arm
evidence table from the captured local-iterate runs, and publish each arm to W&B.

Depth is read from the TRUSTED PARENT's `effective_draft_lengths` rather than
from the candidate's own phase trace, so the depth histogram that decides which
lever ships is not self-reported. The trace is still parsed, as a cross-check:
if the two disagree the arm is not usable.

usage:
  research/e11_collect.py .mlxfast-private/e11/runs --arms C1 C2 H K [--wandb]
"""

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e11-depth-lever-showdown"
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
H_RE = re.compile(r"mtp-trace: begin .*h=([0-9.,eE+-]+)")


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def read_meta(run_dir):
    meta = {}
    path = run_dir / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def load_reports(run_dir):
    """Return (serial, mtp) decode reports from the captured CLI stdout."""
    serial = mtp = None
    for path in sorted(glob.glob(str(run_dir / "reports" / "*.json"))):
        try:
            obj = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or "decode_token_count" not in obj:
            continue
        if obj.get("is_serial_control") is True:
            serial = obj
        elif obj.get("is_serial_control") is False:
            mtp = obj
    return serial, mtp


def trace_depths(run_dir):
    """Chosen depths on the drafting leg, from the candidate's own trace."""
    legs, cur, last = [], [], -1
    curves = []
    for path in sorted(glob.glob(str(run_dir / "trace.txt*"))):
        with open(path, errors="replace") as fh:
            for line in fh:
                h = H_RE.search(line)
                if h:
                    curves.append(h.group(1))
                m = ROUND_RE.search(line)
                if not m:
                    continue
                r, d = int(m.group(1)), int(m.group(2))
                if r <= last and cur:
                    legs.append(cur)
                    cur = []
                last = r
                cur.append(d)
        if cur:
            legs.append(cur)
            cur, last = [], -1
    drafting = [lg for lg in legs if any(lg)]
    return (drafting[-1] if drafting else []), curves


def collect(runs_root, label):
    run_dir = runs_root / label
    meta = read_meta(run_dir)
    score_path = run_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = dict(score.get("metrics") or {})
    serial, mtp = load_reports(run_dir)
    depths, curves = trace_depths(run_dir)

    row = {
        "label": label,
        "arm": meta.get("arm", ""),
        "cli_sha256": meta.get("cli_sha256", ""),
        "worker_sha256": meta.get("worker_sha256", ""),
        "source_sha256": meta.get("source_sha256", ""),
        "head_dir": meta.get("head_dir", ""),
        "tokens": int(meta.get("tokens", 0) or 0),
        "started": meta.get("started", ""),
        "finished": meta.get("finished", ""),
        "h_curve_traced": curves[-1] if curves else "",
    }
    row.update({k: v for k, v in metrics.items()
                if isinstance(v, (int, float, bool)) or v is None
                or k in ("head_provenance_sha256",)})
    if mtp:
        for key in ("round_count", "accepted_draft_total", "rejected_draft_total",
                    "non_drafting_round_count", "effective_mean_draft_len",
                    "effective_max_draft_len", "target_tail_total",
                    "declared_rows_total", "reference_checked_row_total",
                    "rejected_rows_reference_checked", "emitted_token_total",
                    "verify_block_replayed_round_count", "all_tokens_matched",
                    "residual_divergence_count", "parity_all_ok",
                    "max_draft_depth_bound", "peak_ram_gb",
                    "mlx_peak_memory_bytes"):
            if key in mtp:
                row[key] = mtp[key]
        row["head_provenance_sha256"] = (mtp.get("head_provenance") or {}).get(
            "sha256", row.get("head_provenance_sha256", ""))
        parent = mtp.get("effective_draft_lengths") or []
        row["parent_depth_hist"] = dict(sorted(Counter(parent).items()))
        row["parent_depth_n"] = len(parent)
    if serial:
        row["serial_round_count"] = serial.get("round_count")
        row["serial_all_tokens_matched"] = serial.get("all_tokens_matched")
    row["trace_depth_hist"] = dict(sorted(Counter(depths).items()))
    row["trace_depth_n"] = len(depths)
    # The parent's journal and the candidate's trace must describe the same
    # schedule; a mismatch means the trace does not describe the timed leg.
    row["depth_sources_agree"] = (
        bool(row.get("parent_depth_hist"))
        and row.get("parent_depth_hist") == row.get("trace_depth_hist"))
    return row


def to_wandb(row, group, notes):
    import wandb

    config = {
        "experiment": EXPERIMENT,
        "label": row["label"],
        "build_arm": row["arm"],
        "cli_sha256": row["cli_sha256"],
        "worker_sha256": row["worker_sha256"],
        "source_sha256": row["source_sha256"],
        "head_dir": row["head_dir"],
        "decode_tokens": row["tokens"],
        "h_curve_traced": row["h_curve_traced"],
        "head_provenance_sha256": row.get("head_provenance_sha256", ""),
        "git_head": sh("git", "rev-parse", "HEAD"),
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
    }
    summary = {k: v for k, v in row.items()
               if isinstance(v, (int, float, bool)) and k != "tokens"}
    run = wandb.init(project=PROJECT, entity=ENTITY, group=group,
                     name=f"e11-{row['label']}", config=config, notes=notes,
                     reinit=True)
    for src, prefix in (("parent_depth_hist", "depth_parent"),
                        ("trace_depth_hist", "depth_trace")):
        total = sum((row.get(src) or {}).values()) or 1
        for depth, count in (row.get(src) or {}).items():
            summary[f"{prefix}/d{depth}_count"] = count
            summary[f"{prefix}/d{depth}_share"] = count / total
    run.summary.update(summary)
    table = wandb.Table(columns=["depth", "rounds", "share"],
                        data=[[d, c, c / (row["parent_depth_n"] or 1)]
                              for d, c in (row.get("parent_depth_hist") or {}).items()])
    run.log({"parent_depth_histogram": table})
    url = run.url
    run.finish()
    return run.id, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--group", default=EXPERIMENT)
    ap.add_argument("--notes", default="")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rows = []
    for label in args.arms:
        if not (args.runs_root / label).is_dir():
            print(f"e11_collect: no run directory for {label}", file=sys.stderr)
            continue
        row = collect(args.runs_root, label)
        if args.wandb and os.environ.get("WANDB_API_KEY"):
            row["wandb_run_id"], row["wandb_url"] = to_wandb(
                row, args.group, args.notes)
        rows.append(row)

    print(json.dumps(rows, indent=2, sort_keys=False))
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
