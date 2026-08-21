#!/usr/bin/env python3
"""E89 rung 0c: is the per-round host state positional, periodic, or leg-local?

Reads one e89 session and reports, for every leg, the wall-clock start, the
per-round slow fraction and the run-length structure of slow rounds. A blocky
run-length structure means the state has memory across rounds. An i.i.d.
structure means every round draws the state again.
"""
import json
import os

import statistics
import sys

OUT = "research/out"


def leg_dirs(prefix):
    names = [d for d in os.listdir(OUT) if d.startswith(prefix + "-")]
    return sorted(names, key=lambda d: os.path.getmtime(os.path.join(OUT, d)))


def read_meta(path):
    meta = {}
    p = os.path.join(path, "meta.txt")
    if not os.path.exists(p):
        return meta
    for line in open(p):
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    return meta


def read_rounds(path):
    rows = []
    trace = os.path.join(path, "trace.txt")
    if not os.path.exists(trace):
        return rows
    for line in open(trace, errors="ignore"):
        if not line.startswith("mtp-trace: round="):
            continue
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                fields[k] = v
        rows.append(fields)
    rows.sort(key=lambda r: int(r["round"]))
    return rows


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return None


# The eight host-owned phases. These are always in the trace, so probe-off legs
# are comparable with probe-on legs.
HOST_PHASES = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
               "d_chain_us", "readout_us", "commit_us", "upkeep_us")


def host_sum_us(row):
    total = 0.0
    for key in HOST_PHASES:
        v = fnum(row, key)
        if v is None:
            return None
        total += v
    return total


def runs(flags):
    out = []
    if not flags:
        return out
    cur, n = flags[0], 1
    for f in flags[1:]:
        if f == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = f, 1
    out.append((cur, n))
    return out


def main():
    prefix = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dirs = leg_dirs(prefix)

    # global cut on host phase sum, from every post-warmup round in the session
    allhost = []
    per_leg = {}
    for d in dirs:
        rows = read_rounds(os.path.join(OUT, d))[warmup:]
        host = [host_sum_us(r) for r in rows]
        host = [h for h in host if h is not None]
        per_leg[d] = (rows, host)
        allhost += host
    if not allhost:
        print("no rounds found")
        return
    lo = statistics.median(sorted(allhost)[: len(allhost) // 2])
    cut = 2.0 * lo
    print(f"clean-band median {lo:.0f} us, slow cut {cut:.0f} us, "
          f"{len(allhost)} post-warmup rounds over {len(dirs)} legs\n")

    t0 = min(os.path.getmtime(os.path.join(OUT, d)) for d in dirs)
    print(f"{'leg':<22}{'pos':>4}{'end_s':>8}{'n':>5}{'slowfrac':>10}"
          f"{'longest_slow_run':>18}{'longest_clean_run':>19}{'flips':>7}")
    rec = []
    for pos, d in enumerate(dirs):
        rows, host = per_leg[d]
        if not host:
            print(f"{d:<22}{pos:>4}{os.path.getmtime(os.path.join(OUT, d)) - t0:>8.0f}"
                  f"{0:>5}")
            continue
        flags = [h > cut for h in host]
        rl = runs(flags)
        ls = max([n for f, n in rl if f], default=0)
        lc = max([n for f, n in rl if not f], default=0)
        frac = sum(flags) / len(flags)
        end = os.path.getmtime(os.path.join(OUT, d)) - t0
        print(f"{d:<22}{pos:>4}{end:>8.0f}{len(flags):>5}{frac:>10.2f}"
              f"{ls:>18}{lc:>19}{len(rl) - 1:>7}")
        rec.append({"leg": d, "pos": pos, "end_s": end, "n": len(flags),
                    "slow_fraction": frac, "longest_slow_run": ls,
                    "longest_clean_run": lc, "flips": len(rl) - 1,
                    "parts": read_meta(os.path.join(OUT, d)).get("e89_parts", ""),
                    "forced_qos": read_meta(os.path.join(OUT, d)).get("e89_force_qos", "")})

    # blockiness: compare the observed flip count with the i.i.d. expectation
    tot_flips = sum(r["flips"] for r in rec)
    exp_flips = 0.0
    for r in rec:
        p = r["slow_fraction"]
        exp_flips += 2 * p * (1 - p) * (r["n"] - 1)
    print(f"\nrun-length structure: observed flips {tot_flips}, "
          f"i.i.d. expectation {exp_flips:.0f}")
    if exp_flips > 0:
        print(f"  blockiness ratio {tot_flips / exp_flips:.2f} "
              f"(<1 means the state has memory across rounds)")

    json.dump({"cut_us": cut, "clean_band_us": lo, "legs": rec},
              open("research/e89-timeline.json", "w"), indent=2)
    print("\nwrote research/e89-timeline.json")


if __name__ == "__main__":
    main()
