"""Dump the ordered dispatch window that precedes each N=5120 projection.

E106 attributes part of the `gdn.out_proj` excess to its neighbour. That claim
only means something if the neighbour is identified from encode order rather
than assumed, so this reducer prints the exact kernel sequence in front of each
N=5120 victim and the byte-identical `fa.o_proj` control.

Reads the raw census JSONL, so it needs no GPU and no rerun.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

from e106_phase_sweep import PRED_TO_TENSOR, SHAPE_RE, short


def windows(path, phase_want, width_want, back):
    """(tensor, window) -> [us] for every N=5120 dispatch at one width."""
    cells = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        rounds = collections.defaultdict(list)
        for rnd, ordinal, _w, shape_id, gpu_ns in rec["trace"]:
            rounds[rnd].append((ordinal, shape_id, gpu_ns))
        for _rnd, rows in rounds.items():
            seq = sorted(rows)
            for pos, (_ordinal, shape_id, gpu_ns) in enumerate(seq):
                match = parsed[shape_id]
                if match is None:
                    continue
                if match.group("phase") != phase_want:
                    continue
                if not match.group("kernel").startswith("affine_qmv_fast"):
                    continue
                if int(match.group("gy")) != 640:
                    continue
                if int(match.group("gx")) != width_want:
                    continue
                prev = None
                for look in range(pos - 1, -1, -1):
                    prior = parsed[seq[look][1]]
                    if prior is not None:
                        prev = short(prior.group("kernel"))
                        break
                tensor = PRED_TO_TENSOR.get(prev, "unknown(%s)" % prev)
                window = []
                lags = []
                for look in range(max(0, pos - back), pos):
                    prior = parsed[seq[look][1]]
                    if prior is None:
                        continue
                    window.append("%s grid=%sx%s"
                                  % (short(prior.group("kernel")),
                                     prior.group("gx"), prior.group("gy")))
                    lags.append(seq[look][2] / 1e3)
                cells[(tensor, tuple(window))].append((gpu_ns / 1e3, lags))
    return cells


def victims(path, phase_want, width_want):
    """Yield (tensor, slot-in-group, full signature, us) per N=5120 dispatch.

    `slot` counts Gated DeltaNet blocks since the last full-attention block, so
    a neighbour effect and a position effect can be told apart.
    """
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        shapes = rec.get("trace_shapes", [])
        parsed = [SHAPE_RE.match(s) for s in shapes]
        rounds = collections.defaultdict(list)
        for rnd, ordinal, _w, sid, ns in rec["trace"]:
            rounds[rnd].append((ordinal, sid, ns))
        for _rnd, rows in rounds.items():
            seq = sorted(rows)
            since_fa = 0
            for pos, (_o, sid, ns) in enumerate(seq):
                m = parsed[sid]
                if m is None:
                    continue
                if m.group("phase") != phase_want:
                    continue
                if not m.group("kernel").startswith("affine_qmv_fast"):
                    continue
                if int(m.group("gy")) != 640:
                    continue
                if int(m.group("gx")) != width_want:
                    continue
                prev = None
                for look in range(pos - 1, -1, -1):
                    p = parsed[seq[look][1]]
                    if p is not None:
                        prev = short(p.group("kernel"))
                        break
                tensor = PRED_TO_TENSOR.get(prev, "unknown(%s)" % prev)
                if tensor == "fa.o_proj":
                    since_fa = 0
                    slot = 0
                elif tensor == "gdn.out_proj":
                    since_fa += 1
                    slot = since_fa
                else:
                    slot = -1
                yield tensor, slot, shapes[sid], ns / 1e3


def report_slots(path, phase, width):
    by_slot = collections.defaultdict(list)
    sigs = collections.defaultdict(collections.Counter)
    for tensor, slot, sig, us in victims(path, phase, width):
        by_slot[(tensor, slot)].append(us)
        sigs[tensor][sig] += 1

    print("=" * 74)
    print("victim cost by position in the hybrid layer group, width=%d" % width)
    print("%-16s %-5s %6s %10s %8s" % ("tensor", "slot", "n", "mean us", "sd"))
    for key in sorted(by_slot):
        v = by_slot[key]
        print("%-16s %-5d %6d %10.2f %8.2f"
              % (key[0], key[1], len(v), statistics.fmean(v),
                 statistics.pstdev(v) if len(v) > 1 else 0.0))
    print()
    print("distinct kernel signatures per tensor")
    for tensor in sorted(sigs):
        for sig, n in sigs[tensor].most_common():
            print("  %-16s n=%-5d %s" % (tensor, n, sig))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", type=pathlib.Path)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--back", type=int, default=8)
    ap.add_argument("--top", type=int, default=2)
    args = ap.parse_args()

    report_slots(args.census, args.phase, args.width)

    cells = windows(args.census, args.phase, args.width, args.back)
    if not cells:
        print("no N=5120 dispatches at phase=%s width=%d"
              % (args.phase, args.width))
        return

    by_tensor = collections.defaultdict(list)
    for (tensor, window), costs in cells.items():
        by_tensor[tensor].append((window, costs))

    for tensor in sorted(by_tensor):
        entries = sorted(by_tensor[tensor], key=lambda kv: -len(kv[1]))
        total = sum(len(c) for _w, c in entries)
        print("=" * 74)
        print("%s   %d dispatches, %d distinct windows"
              % (tensor, total, len(entries)))
        for window, samples in entries[: args.top]:
            victims = [v for v, _lags in samples]
            print("  n=%-4d victim mean %8.2f us  sd %6.2f"
                  % (len(victims), statistics.fmean(victims),
                     statistics.pstdev(victims) if len(victims) > 1 else 0.0))
            depth = len(window)
            ahead = 0.0
            for slot, kern in enumerate(window):
                lag = depth - slot
                costs = [lags[slot] for _v, lags in samples
                         if len(lags) == depth]
                mean = statistics.fmean(costs) if costs else float("nan")
                ahead += mean
                print("        -%-2d  %-34s %8.2f us" % (lag, kern, mean))
            print("        window GPU time ahead of victim: %.1f us" % ahead)
        print()


if __name__ == "__main__":
    main()
