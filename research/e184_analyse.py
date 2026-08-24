#!/usr/bin/env python3
"""E184 prefill cost decomposition: reduce the session artifacts to tables.

Inputs, all produced by `research/e184-run.sh session`:
  research/e184-phases-<arm>.jsonl   in-path bracketed phase table per forward
  research/capture-e184-<arm>/       trusted CLI reports kept by capture-cli.sh
  research/e184-scan-probe.json      isolated gated-delta scan probe

The trusted `seed_prefill_seconds` field is the anchor. The bracketed total is
always smaller or larger by the instrument's own overhead plus whatever the
brackets do not cover, and that residual is reported instead of hidden.
"""

import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["a1-off", "a2-coarse", "a3-fine", "a4-fine", "a5-coarse", "a6-off"]


def trusted_fields(arm):
    """Every (leg, seed_prefill_seconds, decode_seconds, ...) the CLI reported."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "research", f"capture-e184-{arm}", "*"))):
        try:
            with open(path) as handle:
                doc = json.load(handle)
        except (json.JSONDecodeError, IsADirectoryError, UnicodeDecodeError):
            continue
        for node in walk(doc):
            if isinstance(node, dict) and "seed_prefill_seconds" in node:
                out.append({
                    "file": os.path.basename(path),
                    "seed_prefill_seconds": node.get("seed_prefill_seconds"),
                    "decode_seconds": node.get("decode_seconds"),
                    "prefill_seconds_per_token": node.get("prefill_seconds_per_token"),
                    "seconds_per_token": node.get("seconds_per_token"),
                    "mode": node.get("mode") or node.get("label") or node.get("arm"),
                })
    return out


def walk(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


PREFIX = "mlxfast-worker: "


def phase_lines(arm):
    """Profiler lines rescued from the arm log.

    The worker cannot write files, so each line arrives on its stderr and the
    trusted parent re-emits it prefixed with `mlxfast-worker: `.
    """
    lines, diags = [], []
    sources = [
        os.path.join(ROOT, "research", f"e184-phases-{arm}.jsonl"),
        os.path.join(ROOT, "research", f"e184-log-{arm}.txt"),
    ]
    for path in sources:
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as handle:
            ingest(handle, lines, diags)
    return lines, diags


def ingest(handle, lines, diags):
    for raw in handle:
        index = raw.find(PREFIX)
        payload = raw[index + len(PREFIX):].strip() if index >= 0 else raw.strip()
        if payload.startswith("{"):
            try:
                doc = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if doc.get("e184_prefill_profile"):
                lines.append(doc)
            elif doc.get("e184_diag"):
                diags.append(doc)


def extra_arms():
    """Any other arm tags present in the research directory."""
    tags = set()
    for path in glob.glob(os.path.join(ROOT, "research", "e184-log-*.txt")):
        tags.add(os.path.basename(path)[len("e184-log-"):-len(".txt")])
    for path in glob.glob(os.path.join(ROOT, "research", "capture-e184-*")):
        tags.add(os.path.basename(path)[len("capture-e184-"):])
    return sorted(tags)


def summarise_phases(lines):
    """Mean seconds per phase across the forwards recorded in one arm."""
    per_phase = {}
    totals = []
    for line in lines:
        totals.append(line["bracketed_total_seconds"])
        for phase in line["phases"]:
            per_phase.setdefault(phase["phase"], []).append(phase["seconds"])
    rows = []
    grand = sum(statistics.mean(v) for v in per_phase.values())
    for label in sorted(per_phase):
        values = per_phase[label]
        mean = statistics.mean(values)
        rows.append({
            "phase": label,
            "mean_seconds": mean,
            "share_of_bracketed": mean / grand if grand else 0.0,
            "n": len(values),
            "sd_rel": (statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0),
        })
    return rows, totals


def main():
    report = {"arms": {}, "probe": None}
    probe_path = os.path.join(ROOT, "research", "e184-scan-probe.json")
    if os.path.exists(probe_path):
        with open(probe_path) as handle:
            report["probe"] = json.load(handle)

    for arm in ARMS + [a for a in extra_arms() if a not in ARMS]:
        lines, diags = phase_lines(arm)
        rows, totals = summarise_phases(lines) if lines else ([], [])
        trusted = trusted_fields(arm)
        if not lines and not diags and not trusted:
            continue
        report["arms"][arm] = {
            "forwards_recorded": len(lines),
            "bracketed_total_seconds": totals,
            "phases": rows,
            "diagnostics": diags,
            "trusted": trusted,
        }

    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
