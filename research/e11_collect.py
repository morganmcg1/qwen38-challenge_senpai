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
from statistics import median

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


def round_cost_stats(serial, mtp):
    """Per-depth round cost and the stall-guardrail ratio, from the parent's
    own timing journal.

    C(0) comes from the depth-0 serial control leg and C(d) from the drafting
    leg grouped by the parent's `effective_draft_lengths`, so the cost curve the
    schedule is priced against is measured rather than assumed. Round 0 is
    dropped everywhere: it carries first-round warmup.
    """
    out = {}
    base = None
    if serial:
        wall = (serial.get("block_request_seconds") or [])[1:]
        if wall:
            base = median(wall)
            out["cost/C0_ms"] = base * 1e3
            out["cost/C0_n"] = len(wall)
            out["stall/serial_max_over_p50"] = max(wall) / median(wall)
    if not mtp:
        return out
    wall = (mtp.get("block_request_seconds") or [])[1:]
    depths = (mtp.get("effective_draft_lengths") or [])[1:]
    if wall:
        out["stall/mtp_max_over_p50"] = max(wall) / median(wall)
    if not (wall and depths and len(wall) == len(depths)):
        return out
    per_depth = {}
    for seconds, depth in zip(wall, depths):
        per_depth.setdefault(depth, []).append(seconds)
    for depth in sorted(per_depth):
        cost = median(per_depth[depth])
        out[f"cost/C{depth}_ms"] = cost * 1e3
        out[f"cost/C{depth}_n"] = len(per_depth[depth])
        if base:
            out[f"cost/C{depth}_over_C0"] = cost / base
    if base:
        levels = {0: base}
        levels.update({d: median(v) for d, v in per_depth.items() if d > 0})
        for depth in sorted(levels):
            if depth + 1 in levels:
                out[f"cost/h{depth}_marginal"] = (
                    levels[depth + 1] - levels[depth]) / base
    return out


def arm_h_form(arm):
    """The head-cost constant as it appeared in the source that was actually
    compiled for this arm, read back from the build archive rather than from the
    working tree, which moves between arms.
    """
    src = Path(".mlxfast-private/e11/bins") / arm / "source.swift"
    if not (arm and src.exists()):
        return ""
    lines = src.read_text().splitlines()
    for index, line in enumerate(lines):
        if "headStepCostRatio" not in line or "let " not in line:
            continue
        chunk = [line.strip()]
        # `]` is tested on the right of `=` only: the type annotation on the
        # left (`[Double]`) also carries one.
        while (chunk[-1].endswith(("[", ","))
               and "]" not in chunk[-1].split("=", 1)[-1]):
            index += 1
            chunk.append(lines[index].strip())
        return " ".join(chunk)
    return ""


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
        "thermal_before": meta.get("thermal_before", ""),
        "thermal_after": meta.get("thermal_after", ""),
        "h_curve_traced": curves[-1] if curves else "",
        "h_form": arm_h_form(meta.get("arm", "")),
        "run_head_sha": meta.get("head_sha", ""),
        "run_worktree_dirty": int(meta.get("dirty", 0) or 0),
    }
    row.update(round_cost_stats(serial, mtp))
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
                    "mlx_peak_memory_bytes", "decode_seconds",
                    "seed_prefill_seconds", "prefill_seconds_per_token",
                    "p50_block_request_seconds_after_first",
                    "max_block_request_seconds_after_first",
                    "accepted_draft_rate", "accepted_draft_token_total",
                    "rejected_draft_row_total", "declared_row_total"):
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
        for key in ("decode_seconds", "seed_prefill_seconds",
                    "prefill_seconds_per_token",
                    "p50_block_request_seconds_after_first",
                    "max_block_request_seconds_after_first"):
            if key in serial:
                row[f"serial_{key}"] = serial[key]
    row["trace_depth_hist"] = dict(sorted(Counter(depths).items()))
    row["trace_depth_n"] = len(depths)
    row["pass"] = meta.get("pass", "timed")
    row["mlx_qwen_env"] = meta.get("mlx_qwen_env", "")
    row["golden"] = meta.get("golden", "<default>")
    # A timed arm deliberately runs with no MLX_QWEN_MTP_* name set, so it has
    # no trace to agree with; the parent's journal is then the sole and
    # sufficient depth source. Only a fingerprint pass can contradict itself.
    row["depth_sources_agree"] = (
        None if not row["trace_depth_hist"]
        else (bool(row.get("parent_depth_hist"))
              and row.get("parent_depth_hist") == row.get("trace_depth_hist")))
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
        "h_form": row.get("h_form", ""),
        "base_sha": sh("git", "merge-base", "HEAD",
                       "origin/senpai/qwen38-mtp-r1"),
        "run_head_sha": row.get("run_head_sha", ""),
        "pass": row.get("pass", "timed"),
        "mlx_qwen_env": row.get("mlx_qwen_env", ""),
        "golden": row.get("golden", "<default>"),
        "head_provenance_sha256": row.get("head_provenance_sha256", ""),
        "thermal_before": row.get("thermal_before", ""),
        "thermal_after": row.get("thermal_after", ""),
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
