#!/usr/bin/env python3
"""Research-only: decompose a control/candidate draft-bits pair from one job log.

    research/draft_bits_phase2.py JOB_LOG [--arms 4,3]

Reads the `mtp-trace:` stream that MLX_QWEN_MTP_TRACE=1 emits and reports, per
arm, the draft-head provenance line, the round schedule, the first-round graph
build penalty against same-depth steady-state rounds, and the paired per-round
delta between the two arms.

The per-round term and the acceptance term are reported separately on purpose:
they answer different questions, and collapsing them hides whether a precision
change bought bandwidth or changed which tokens were drafted.
"""
import argparse
import re
import statistics as st
import sys

ARM_RE = re.compile(r"run-draft-bits-phase\d: arm .*?bits=(\d)")
KV_RE = re.compile(r"(\w+)=([\d.]+)")
INT_KV_RE = re.compile(r"(\w+)=(\d+)")


def parse(path):
    """Group trace records by arm and by worker instance.

    `slot` is the 1-based position of the arm invocation in the log, which is
    what distinguishes the two replicates of the same bit width in a
    counterbalanced Phase 3 order; `arm` is the bit width and repeats.
    """
    arm = None
    slot = 0
    blocks = []
    cur = None
    for line in open(path, errors="replace"):
        m = ARM_RE.search(line)
        if m:
            arm = m.group(1)
            slot += 1
        if "draft-head materialised" in line:
            kv = dict(KV_RE.findall(line))
            cur = {
                "arm": arm,
                "slot": slot,
                "bits": kv["bits"],
                "source_bits": kv["source_bits"],
                "requant_ms": float(kv["requant_ms"]),
                "rounds": [],
            }
            blocks.append(cur)
        if "mtp-trace: round=" in line and cur is not None:
            cur["rounds"].append(
                {k: int(v) for k, v in INT_KV_RE.findall(line)})
    return blocks


def med(rows, key):
    return st.median(r[key] for r in rows)


FIELDS = ("draft_build_us", "verify_build_us", "eval_wall_us", "readout_us",
          "commit_us", "upkeep_us", "round_us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--arms", default="4,3")
    args = ap.parse_args()
    arms = args.arms.split(",")

    blocks = parse(args.log)

    print("== draft-head provenance (every worker instance) ==")
    for b in blocks:
        print("  arm=%s bits=%s source_bits=%s requant_ms=%.3f rounds=%d"
              % (b["arm"], b["bits"], b["source_bits"], b["requant_ms"],
                 len(b["rounds"])))
    for arm in arms:
        rq = [b["requant_ms"] for b in blocks if b["arm"] == arm]
        print("  arm=%s requant_ms n=%d mean=%.3f min=%.3f max=%.3f"
              % (arm, len(rq), sum(rq) / len(rq), min(rq), max(rq)))

    timed = {}
    for b in blocks:
        if b["rounds"]:
            if b["arm"] in timed:
                sys.exit("arm %s has more than one timed instance" % b["arm"])
            timed[b["arm"]] = b["rounds"]
    missing = [a for a in arms if a not in timed]
    if missing:
        sys.exit("no timed rounds for arm(s): %s" % ",".join(missing))

    print("\n== round schedule ==")
    for arm in arms:
        r = timed[arm]
        print("  arm=%s rounds=%d depths=%s" % (
            arm, len(r), ",".join(str(x["d"]) for x in r)))
        print("    proposed=%d accepted=%d emitted=%d sum_round_ms=%.3f" % (
            sum(x["d"] for x in r), sum(x["acc"] for x in r),
            len(r) + sum(x["acc"] for x in r),
            sum(x["round_us"] for x in r) / 1000.0))

    print("\n== warm-graph audit: round 1 vs same-depth steady state ==")
    for arm in arms:
        r = timed[arm]
        first, d1 = r[0], r[0]["d"]
        same = [x for x in r[1:] if x["d"] == d1]
        print("  arm=%s round1 d=%d %s" % (arm, d1, " ".join(
            "%s=%d" % (f, first[f]) for f in FIELDS)))
        if not same:
            print("    no later round at d=%d; per-depth medians follow" % d1)
            for dd in sorted({x["d"] for x in r[1:]}):
                s = [x for x in r[1:] if x["d"] == dd]
                print("      d=%d n=%d %s" % (dd, len(s), " ".join(
                    "%s=%d" % (f, med(s, f)) for f in FIELDS)))
            continue
        print("    steady d=%d n=%d %s" % (d1, len(same), " ".join(
            "%s=%d" % (f, med(same, f)) for f in FIELDS)))
        print("    round1 excess: %s" % " ".join(
            "%s=%+d" % (f, first[f] - med(same, f)) for f in FIELDS))

    if len(arms) != 2:
        return

    ctl, cand = timed[arms[0]], timed[arms[1]]
    print("\n== paired per-round delta (arm %s - arm %s) ==" % (arms[0], arms[1]))
    if [x["d"] for x in ctl] != [x["d"] for x in cand]:
        print("  depth schedules DIFFER; pairing by index anyway")
    n = min(len(ctl), len(cand))
    pairs = [(ctl[i], cand[i]) for i in range(n)
             if ctl[i]["d"] == cand[i]["d"]]
    print("  paired rounds=%d of %d (same depth)" % (len(pairs), n))
    for f in FIELDS:
        d = [a[f] - b[f] for a, b in pairs]
        print("    %-16s median=%+8.1f us  mean=%+8.1f us" % (
            f, st.median(d), sum(d) / len(d)))
    # Steady state only: drop round 1, which carries the graph build.
    steady = [(a, b) for a, b in pairs[1:]]
    dr = [a["round_us"] - b["round_us"] for a, b in steady]
    readouts = sum(a["d"] for a, _ in steady)
    print("  steady-state rounds=%d  total round_us delta=%+d  readouts=%d"
          % (len(steady), sum(dr), readouts))
    if readouts:
        print("  implied saving per draft-head readout=%+.1f us"
              % (sum(dr) / readouts))


if __name__ == "__main__":
    main()
