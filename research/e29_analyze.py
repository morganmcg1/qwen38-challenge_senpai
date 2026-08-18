#!/usr/bin/env python3
"""Research-only (qwen38-r1-e29): attribute the per-round host tail from the
session's own six-way phase trace.

E20 produced the 0.6641 s / 10.26 % round-overhead headline as a SUBTRACTIVE
RESIDUAL: overhead(M) = block_m1(M) - attributed_m1(M), summed over widths
(research/e20_analyze.py:417,444). That quantity also defines target_work, and
its own fit has a NEGATIVE fixed floor (-1.46 ms + 4.58 ms/draft token), which
is unphysical. This script replaces the residual with directly measured host
spans already emitted by Qwen36MTPBlockSession:

    round_us = draft_build + verify_build + eval_wall + readout + commit + upkeep

`eval_wall` is the only segment that contains GPU time. The other five are host
work serialized against the device, so

    host_tail_us = round_us - eval_wall_us

is the directly measured analogue of E20's residual, decomposed into named
components without any subtraction against a second build.

Trace lines are appended by every worker that runs a session (reference,
verify, timed), so rounds are segmented on a non-increasing round counter.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Six-way split emitted at Qwen36MTPBlockSession.swift:1146-1155.
SEGMENTS = (
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
)
# Every segment except eval_wall is host work the device could be overlapping.
HOST_SEGMENTS = tuple(s for s in SEGMENTS if s != "eval_wall_us")

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")

# E21 512-token observed width histogram (draft_len+1 -> rounds). The local
# --local-iterate window is short and depth-8 heavy, so the raw per-round mean
# is NOT the ranked-relevant weighting; both are reported.
E21_WIDTH_HISTOGRAM = {1: 193, 2: 995, 3: 583, 4: 167, 5: 9}


@dataclass
class Round:
    session: int
    index: int
    width: int
    accepted: int
    seg: dict[str, int]
    round_us: int

    @property
    def host_tail_us(self) -> int:
        return self.round_us - self.seg["eval_wall_us"]


@dataclass
class Aggregate:
    rounds: int = 0
    seg_us: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(SEGMENTS, 0))
    round_us: int = 0

    def add(self, r: Round) -> None:
        self.rounds += 1
        self.round_us += r.round_us
        for k in SEGMENTS:
            self.seg_us[k] += r.seg[k]

    @property
    def host_tail_us(self) -> int:
        return self.round_us - self.seg_us["eval_wall_us"]

    @property
    def unaccounted_us(self) -> int:
        """round_us minus the six segments.

        The stamps tile the round contiguously, so this should be ~0. A
        non-zero value means the trace does not actually tile the round and
        every share below is suspect.
        """
        return self.round_us - sum(self.seg_us.values())


def parse_trace(path: Path) -> list[Round]:
    rounds: list[Round] = []
    session = 0
    prev_index = None
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND_RE.match(line.strip())
        if not m:
            continue
        index = int(m.group(1))
        if prev_index is not None and index <= prev_index:
            session += 1
        prev_index = index
        kv = dict(KV_RE.findall(m.group(4)))
        missing = [s for s in SEGMENTS if s not in kv]
        if missing or "round_us" not in kv:
            continue
        rounds.append(Round(
            session=session,
            index=index,
            width=int(m.group(2)) + 1,
            accepted=int(m.group(3)),
            seg={s: int(kv[s]) for s in SEGMENTS},
            round_us=int(kv["round_us"]),
        ))
    return rounds


def block_seconds(reports_dir: Path) -> dict[str, object]:
    """Per-leg parent-observed round timing from the captured CLI reports.

    03-* is the serial leg and 04-* the MTP leg (capture-cli.sh numbers each
    invocation), so the MTP leg's block_request_seconds is the parent-side
    quantity the trace's round_us must be reconciled against.
    """
    out: dict[str, object] = {}
    for report in sorted(reports_dir.glob("*.json")):
        try:
            doc = json.loads(report.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        blocks = doc.get("block_request_seconds")
        if not isinstance(blocks, list) or not blocks:
            continue
        out[report.name] = {
            "rounds": len(blocks),
            "sum_s": sum(blocks),
            "mean_ms": 1e3 * statistics.fmean(blocks),
            "decode_seconds": doc.get("decode_seconds"),
            "effective_draft_lengths":
                doc.get("effective_draft_lengths") is not None,
        }
    return out


def reweight(by_width: dict[int, Aggregate],
             histogram: dict[int, int]) -> dict[str, object] | None:
    """Per-round means reweighted onto an external width histogram.

    Widths absent from the trace cannot be reweighted; they are reported so a
    partial cover is never silently presented as a full one.
    """
    covered = {m: n for m, n in histogram.items() if m in by_width}
    if not covered:
        return None
    total = sum(covered.values())
    shares: dict[str, float] = {}
    for key in SEGMENTS:
        shares[key] = sum(
            n * by_width[m].seg_us[key] / by_width[m].rounds
            for m, n in covered.items()) / total
    round_mean = sum(
        n * by_width[m].round_us / by_width[m].rounds
        for m, n in covered.items()) / total
    host = round_mean - shares["eval_wall_us"]
    return {
        "histogram": histogram,
        "covered_widths": sorted(covered),
        "missing_widths": sorted(set(histogram) - set(covered)),
        "covered_rounds": total,
        "round_mean_ms": round_mean / 1e3,
        "host_tail_mean_ms": host / 1e3,
        "host_tail_share_pct": 100.0 * host / round_mean if round_mean else None,
        "segment_mean_ms": {k: v / 1e3 for k, v in shares.items()},
        "segment_share_of_round_pct": {
            k: 100.0 * v / round_mean for k, v in shares.items()
        } if round_mean else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    doc: dict[str, object] = {}
    for run_dir in args.run_dirs:
        trace = run_dir / "rounds.trace"
        entry: dict[str, object] = {"run_dir": str(run_dir)}
        meta = run_dir / "meta.txt"
        if meta.is_file():
            entry["meta"] = dict(
                line.split("=", 1) for line in meta.read_text().splitlines()
                if "=" in line)
        reports = run_dir / "reports"
        if reports.is_dir():
            entry["parent_block_seconds"] = block_seconds(reports)

        if not trace.is_file():
            entry["trace"] = "absent"
            doc[run_dir.name] = entry
            print(f"{run_dir.name}: no rounds.trace", file=sys.stderr)
            continue

        rounds = parse_trace(trace)
        if not rounds:
            entry["trace"] = "empty"
            doc[run_dir.name] = entry
            print(f"{run_dir.name}: rounds.trace has no parsable rounds",
                  file=sys.stderr)
            continue

        # The timed MTP leg is the last session in the appended file; earlier
        # sessions are the reference/verify workers.
        last = max(r.session for r in rounds)
        timed = [r for r in rounds if r.session == last]

        overall = Aggregate()
        for r in timed:
            overall.add(r)
        by_width: dict[int, Aggregate] = {}
        for r in timed:
            by_width.setdefault(r.width, Aggregate()).add(r)

        entry["sessions_in_file"] = last + 1
        entry["timed_session_rounds"] = overall.rounds
        entry["accepted_draft_total"] = sum(r.accepted for r in timed)
        entry["unaccounted_us"] = overall.unaccounted_us
        entry["totals_ms"] = {
            "round": overall.round_us / 1e3,
            "host_tail": overall.host_tail_us / 1e3,
            **{k: overall.seg_us[k] / 1e3 for k in SEGMENTS},
        }
        entry["share_of_round_pct"] = {
            "host_tail": 100.0 * overall.host_tail_us / overall.round_us,
            **{k: 100.0 * overall.seg_us[k] / overall.round_us
               for k in SEGMENTS},
        }
        entry["host_tail_composition_pct"] = {
            k: 100.0 * overall.seg_us[k] / overall.host_tail_us
            for k in HOST_SEGMENTS
        } if overall.host_tail_us else None
        entry["per_width"] = {
            str(m): {
                "rounds": a.rounds,
                "round_mean_ms": a.round_us / a.rounds / 1e3,
                "host_tail_mean_ms": a.host_tail_us / a.rounds / 1e3,
                "segment_mean_ms": {
                    k: a.seg_us[k] / a.rounds / 1e3 for k in SEGMENTS},
            }
            for m, a in sorted(by_width.items())
        }
        entry["reweighted_e21_512"] = reweight(by_width, E21_WIDTH_HISTOGRAM)
        doc[run_dir.name] = entry

        print(f"\n=== {run_dir.name}: {overall.rounds} timed rounds "
              f"({last + 1} sessions in file) ===")
        print(f"  unaccounted (round_us - sum segments): "
              f"{overall.unaccounted_us / 1e3:.2f} ms  "
              f"({100.0 * overall.unaccounted_us / overall.round_us:+.3f}% of round)")
        print(f"  {'segment':<18} {'total_ms':>10} {'%round':>8} {'%host':>8}"
              f" {'mean_ms':>9}")
        for k in SEGMENTS:
            tot = overall.seg_us[k] / 1e3
            pct = 100.0 * overall.seg_us[k] / overall.round_us
            hp = (100.0 * overall.seg_us[k] / overall.host_tail_us
                  if k != "eval_wall_us" and overall.host_tail_us else float("nan"))
            print(f"  {k:<18} {tot:>10.2f} {pct:>7.2f}% {hp:>7.2f}%"
                  f" {tot / overall.rounds:>9.3f}")
        print(f"  {'HOST TAIL':<18} {overall.host_tail_us / 1e3:>10.2f} "
              f"{100.0 * overall.host_tail_us / overall.round_us:>7.2f}%"
              f" {'':>8} {overall.host_tail_us / overall.rounds / 1e3:>9.3f}")
        print(f"  {'ROUND':<18} {overall.round_us / 1e3:>10.2f}")

        print(f"  {'M':>3} {'rounds':>7} {'round_ms':>9} {'host_ms':>8}"
              f" {'host%':>7} {'dbuild':>7} {'vbuild':>7} {'eval':>7}"
              f" {'readout':>8} {'commit':>7} {'upkeep':>7}")
        for m, a in sorted(by_width.items()):
            g = lambda k: a.seg_us[k] / a.rounds / 1e3  # noqa: E731
            print(f"  {m:>3} {a.rounds:>7} {a.round_us / a.rounds / 1e3:>9.3f}"
                  f" {a.host_tail_us / a.rounds / 1e3:>8.3f}"
                  f" {100.0 * a.host_tail_us / a.round_us:>6.2f}%"
                  f" {g('draft_build_us'):>7.3f} {g('verify_build_us'):>7.3f}"
                  f" {g('eval_wall_us'):>7.3f} {g('readout_us'):>8.3f}"
                  f" {g('commit_us'):>7.3f} {g('upkeep_us'):>7.3f}")

        rw = entry["reweighted_e21_512"]
        if rw:
            print(f"  reweighted onto E21 512-token histogram "
                  f"{rw['covered_widths']} (missing {rw['missing_widths']}): "
                  f"round {rw['round_mean_ms']:.3f} ms, "
                  f"host tail {rw['host_tail_mean_ms']:.3f} ms "
                  f"({rw['host_tail_share_pct']:.2f}%)")

        pb = entry.get("parent_block_seconds") or {}
        for name, stats in pb.items():
            print(f"  parent {name}: {stats['rounds']} rounds, "
                  f"sum {stats['sum_s']:.4f} s, mean {stats['mean_ms']:.3f} ms,"
                  f" decode_seconds={stats['decode_seconds']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(doc, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
