#!/usr/bin/env python3
"""E89 rung 0c: name the CPU execution-rate state and price the candidate fix.

Rung 0b established that the slow drafting round runs the identical instruction
stream at 0.75x the clock and 0.46x the work per cycle, with the thread on-core
for the whole phase. Two questions remain.

  1. Is the slow mode a lower clock on the same cluster, or a move to the
     efficiency cluster? `e89_core_a` and the calibrated probe clock answer
     this directly, so it is a measurement rather than an inference.
  2. Does forced userInteractive QoS remove the state, and what does that buy
     in absolute candidate seconds per token?

Every timing here is `harness=local` and ungated. Nothing here is a score.

  usage: research/e89_rung0c.py PREFIX [WARMUP_ROUNDS]
"""
import itertools
import json
import math
import os
import statistics
import sys

OUT = "research/out"
WARMUP = 8

# The eight host-owned phases, present whether or not the probe is on.
HOST_PHASES = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
               "d_chain_us", "readout_us", "commit_us", "upkeep_us")

# Counters reported per round. Ratios are slow over fast.
COUNTERS = [
    ("host phase sum us", "host_sum_us"),
    ("host thread cpu ns", "host_thread_cpu_ns"),
    ("round thread cpu ns", "round_thread_cpu_ns"),
    ("probe ns, chain only", "e89_probe_ns"),
    ("probe span ns", "e89_probe_span_ns"),
    ("probe cycles", "e89_probe_cyc"),
    ("probe instructions", "e89_probe_ins"),
    ("instructions", "e89_instr"),
    ("cycles", "e89_cycles"),
    ("user cpu ns", "e89_user_ns"),
    ("system cpu ns", "e89_sys_ns"),
    ("thread user ns", "e89_thr_user_ns"),
    ("thread system ns", "e89_thr_sys_ns"),
    ("voluntary switches", "e89_nvcsw"),
    ("involuntary switches", "e89_nivcsw"),
    ("minor faults", "e89_minflt"),
    ("major faults", "e89_majflt"),
    ("pageins", "e89_pageins"),
    ("task compressed bytes", "e89_vm_comp"),
    ("task decompressions", "e89_vm_decomp"),
    ("task swapins", "e89_vm_swapin"),
    ("task footprint mb", "e89_vm_footprint_mb"),
    ("task compressed mb", "e89_vm_compressed_mb"),
    ("host free mb", "e89_host_free_mb"),
    ("host compressor mb", "e89_host_compressor_mb"),
    ("host pageins", "e89_host_pageins"),
    ("host decompressions", "e89_host_decomp"),
    ("host swapins", "e89_host_swapin"),
    ("verify build us", "verify_build_us"),
    ("eval wall us", "eval_wall_us"),
    ("submit2 us", "d_submit2_us"),
    ("round us", "round_us"),
    ("mlx cache mb", "e89_cache_mb"),
    ("mlx active mb", "e89_active_mb"),
]


def leg_dirs(prefix):
    names = [d for d in os.listdir(OUT) if d.startswith(prefix + "-")]
    return sorted(names, key=lambda d: os.path.getmtime(os.path.join(OUT, d)))


def read_meta(path):
    meta = {}
    p = os.path.join(path, "meta.txt")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
    return meta


def read_score(path):
    p = os.path.join(path, "score.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p)).get("metrics", {})
    except (ValueError, OSError):
        return {}


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


def host_sum_us(row):
    total = 0.0
    for key in HOST_PHASES:
        v = fnum(row, key)
        if v is None:
            return None
        total += v
    return total


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def ratio(slow, fast):
    if fast in (None, 0) or slow is None:
        return None
    return slow / fast


def permutation_p(a, b, iters=200000):
    """One-sided: is `a` larger than `b`? Exact when the split is small."""
    obs = statistics.mean(a) - statistics.mean(b)
    pool = list(a) + list(b)
    n = len(a)
    combos = list(itertools.combinations(range(len(pool)), n))
    if len(combos) > iters:
        return None, len(combos)
    hits = 0
    for combo in combos:
        x = [pool[i] for i in combo]
        y = [pool[i] for i in range(len(pool)) if i not in combo]
        if statistics.mean(x) - statistics.mean(y) >= obs - 1e-12:
            hits += 1
    return hits / len(combos), len(combos)


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    prefix = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else WARMUP
    dirs = leg_dirs(prefix)
    if not dirs:
        print(f"no legs for prefix {prefix}")
        return

    legs = []
    allhost = []
    for pos, d in enumerate(dirs):
        path = os.path.join(OUT, d)
        rows = read_rounds(path)[warmup:]
        meta = read_meta(path)
        score = read_score(path)
        host = [host_sum_us(r) for r in rows]
        allhost += [h for h in host if h is not None]
        legs.append({"leg": d, "pos": pos, "rows": rows, "meta": meta,
                     "score": score, "host": host})

    if not allhost:
        print("no post-warmup rounds")
        return

    # One global cut, from the clean half of every post-warmup round.
    clean_band = statistics.median(sorted(allhost)[: len(allhost) // 2])
    cut = 2.0 * clean_band
    print(f"clean band median {clean_band:.0f} us, slow cut {cut:.0f} us, "
          f"{len(allhost)} post-warmup rounds over {len(dirs)} legs")
    print("harness=local, ungated, cool_gate_passed_real_gate=false, "
          "gate_qualified_for_timing=false, official_or_ranked_score=false\n")

    # ---- per-leg -----------------------------------------------------------
    print("PER LEG")
    print(f"{'leg':<24}{'arm':<7}{'qos':<16}{'sync':>5}{'pos':>4}{'n':>4}"
          f"{'prev':>7}{'host_us':>9}{'GHz':>6}{'cores':>16}{'s/token':>10}")
    for leg in legs:
        rows, host = leg["rows"], leg["host"]
        flags = [h is not None and h > cut for h in host]
        prev = sum(flags) / len(flags) if flags else 0.0
        arm = leg["leg"][len(prefix) + 1:].rsplit("-", 1)[0]
        cyc = med([fnum(r, "e89_probe_cyc") for r in rows])
        span = med([fnum(r, "e89_probe_span_ns") for r in rows])
        ghz = cyc / span if cyc and span else 0.0
        cores = {}
        for r in rows:
            c = r.get("e89_core_a")
            if c is not None:
                cores[c] = cores.get(c, 0) + 1
        top = ",".join(f"{k}:{v}" for k, v in
                       sorted(cores.items(), key=lambda kv: -kv[1])[:3])
        leg.update(prevalence=prev, flags=flags, arm=arm, ghz=ghz, cores=cores)
        print(f"{leg['leg']:<24}{arm:<7}{leg['meta'].get('e89_force_qos',''):<16}"
              f"{leg['meta'].get('e89_sync_head',''):>5}{leg['pos']:>4}"
              f"{len(flags):>4}{prev:>7.2f}{med(host) or 0:>9.0f}{ghz:>6.3f}"
              f"{top:>16}"
              f"{leg['score'].get('mtp_seconds_per_token', 0) or 0:>10.5f}")

    # ---- arm summary and the fix test --------------------------------------
    print("\nPER ARM")
    arms = {}
    for leg in legs:
        arms.setdefault(leg["arm"], []).append(leg)
    print(f"{'arm':<10}{'n':>3}{'prevalence':>12}{'host_us':>9}{'GHz':>7}"
          f"{'s/token':>10}{'spread':>9}")
    for arm, group in sorted(arms.items()):
        prevs = [g["prevalence"] for g in group]
        spt = [g["score"].get("mtp_seconds_per_token") for g in group]
        spt = [s for s in spt if s]
        print(f"{arm:<10}{len(group):>3}{statistics.mean(prevs):>12.3f}"
              f"{med([med(g['host']) for g in group]) or 0:>9.0f}"
              f"{statistics.mean([g['ghz'] for g in group]):>7.3f}"
              f"{statistics.mean(spt) if spt else 0:>10.5f}"
              f"{(max(spt) - min(spt)) if len(spt) > 1 else 0:>9.5f}")

    if "ctl" in arms and "uix" in arms:
        a = [g["prevalence"] for g in arms["ctl"]]
        b = [g["prevalence"] for g in arms["uix"]]
        p, n = permutation_p(a, b)
        print(f"\nFIX TEST, prevalence, ctl over uix")
        print(f"  ctl mean {statistics.mean(a):.3f} n={len(a)}   "
              f"uix mean {statistics.mean(b):.3f} n={len(b)}")
        print(f"  exact one-sided permutation p={p} over {n} relabelings"
              if p is not None else f"  {n} relabelings, too many to enumerate")
        sa = [g["score"].get("mtp_seconds_per_token") for g in arms["ctl"]]
        sb = [g["score"].get("mtp_seconds_per_token") for g in arms["uix"]]
        sa = [s for s in sa if s]
        sb = [s for s in sb if s]
        if sa and sb:
            p2, n2 = permutation_p(sa, sb)
            delta = statistics.mean(sa) - statistics.mean(sb)
            print(f"  absolute candidate s/token ctl {statistics.mean(sa):.5f} "
                  f"uix {statistics.mean(sb):.5f}  "
                  f"delta {delta:+.5f} ({100 * delta / statistics.mean(sa):+.2f} %)")
            print(f"  exact one-sided permutation p={p2} over {n2} relabelings"
                  if p2 is not None else f"  {n2} relabelings")

    # ---- pooled discriminator ---------------------------------------------
    fast, slow = [], []
    for leg in legs:
        for row, flag in zip(leg["rows"], leg["flags"]):
            (slow if flag else fast).append(row)
    print(f"\nDISCRIMINATOR, {len(fast)} fast and {len(slow)} slow "
          f"post-warmup rounds pooled over every leg")
    print(f"{'quantity':<26}{'fast':>16}{'slow':>16}{'ratio':>8}")
    summary = {}
    for label, key in COUNTERS:
        f = med([fnum(r, key) for r in fast])
        s = med([fnum(r, key) for r in slow])
        if f is None and s is None:
            continue
        r = ratio(s, f)
        summary[key] = {"fast": f, "slow": s, "ratio": r}
        print(f"{label:<26}{f if f is not None else 0:>16.0f}"
              f"{s if s is not None else 0:>16.0f}"
              f"{r if r is not None else 0:>8.2f}")

    # host_sum_us is emitted only when the probe runs; fall back to the phases.
    fh = med([host_sum_us(r) for r in fast])
    sh = med([host_sum_us(r) for r in slow])
    print(f"{'host phases, wall':<26}{fh or 0:>16.0f}{sh or 0:>16.0f}"
          f"{ratio(sh, fh) or 0:>8.2f}")

    # ---- clock, work per cycle and occupancy -------------------------------
    print("\nRATE DECOMPOSITION")
    for label, group in (("fast", fast), ("slow", slow)):
        cyc = med([fnum(r, "e89_cycles") for r in group])
        usr = med([fnum(r, "e89_user_ns") for r in group])
        ins = med([fnum(r, "e89_instr") for r in group])
        pc = med([fnum(r, "e89_probe_cyc") for r in group])
        ps = med([fnum(r, "e89_probe_span_ns") for r in group])
        pi = med([fnum(r, "e89_probe_ins") for r in group])
        hw = med([host_sum_us(r) for r in group])
        hc = med([fnum(r, "host_thread_cpu_ns") for r in group])
        print(f"  {label:<5} process clock {cyc / usr if cyc and usr else 0:6.3f} GHz"
              f"   process IPC {ins / cyc if ins and cyc else 0:6.3f}"
              f"   probe clock {pc / ps if pc and ps else 0:6.3f} GHz"
              f"   probe IPC {pi / pc if pi and pc else 0:6.3f}"
              f"   host occupancy {hc / (hw * 1000) if hc and hw else 0:6.4f}")

    # ---- core cluster ------------------------------------------------------
    print("\nCORE OCCUPANCY, e89_core_a")
    for label, group in (("fast", fast), ("slow", slow)):
        hist = {}
        for r in group:
            c = r.get("e89_core_a")
            if c is not None:
                hist[c] = hist.get(c, 0) + 1
        total = sum(hist.values()) or 1
        cells = " ".join(f"cpu{k}:{100 * v / total:.0f}%" for k, v in
                         sorted(hist.items(), key=lambda kv: int(kv[0])))
        print(f"  {label:<5} {cells}")
    print("  probe clock by core, over every post-warmup round")
    bycore = {}
    for r in fast + slow:
        c = r.get("e89_core_a")
        pc, ps = fnum(r, "e89_probe_cyc"), fnum(r, "e89_probe_span_ns")
        if c is None or not pc or not ps:
            continue
        bycore.setdefault(c, []).append(pc / ps)
    for c in sorted(bycore, key=lambda k: int(k)):
        v = bycore[c]
        print(f"    cpu{c:<3} n={len(v):<6} median {statistics.median(v):.3f} GHz")

    # ---- stuck-leg rate ----------------------------------------------------
    obs = [leg for leg in legs if leg["meta"].get("e89_force_qos") == "none"
           and leg["meta"].get("e89_sync_head") != "1"]
    stuck = [leg for leg in obs if leg["prevalence"] > 0.5]
    lo, hi = wilson(len(stuck), len(obs))
    print(f"\nunforced legs {len(obs)}, majority-slow {len(stuck)}, "
          f"rate {len(stuck) / len(obs) if obs else 0:.3f}, "
          f"Wilson 95 % CI [{lo:.3f}, {hi:.3f}]")

    doc = {"prefix": prefix, "cut_us": cut, "clean_band_us": clean_band,
           "harness": "local", "gate_qualified_for_timing": False,
           "official_or_ranked_score": False,
           "legs": [{k: leg[k] for k in ("leg", "pos", "arm", "prevalence", "ghz")}
                    | {"forced_qos": leg["meta"].get("e89_force_qos"),
                       "parts": leg["meta"].get("e89_parts"),
                       "sync_head": leg["meta"].get("e89_sync_head"),
                       "host_med_us": med(leg["host"]),
                       "mtp_seconds_per_token":
                           leg["score"].get("mtp_seconds_per_token"),
                       "cores": leg["cores"]}
                    for leg in legs],
           "discriminator": summary}
    json.dump(doc, open("research/e89-rung0c.json", "w"), indent=2)
    print("\nwrote research/e89-rung0c.json")


if __name__ == "__main__":
    main()
