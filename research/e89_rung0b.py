#!/usr/bin/env python3
"""E89 rung 0b: is the host thread slow, or is it busy?

Reads the per-round `e89_*` fields the host-state probe emits and answers the
question three independent ways.

  1. `e89_instr` is the exact retired-instruction count for the round. A busy
     thread retires more instructions. A slow thread retires the same number.
     This field alone separates the two hypotheses.
  2. `e89_probe_ns` is a fixed 20,000-iteration dependent integer chain run
     on the same thread just before the round. It touches no memory and its
     instruction count is constant, so its wall time is a direct readout of
     issue latency.
  3. The per-QoS CPU-time split says which service class accrued the round's
     CPU nanoseconds.

usage: research/e89_rung0b.py PREFIX [PREFIX ...]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
HEADER_RE = re.compile(r"^e89-probe: header (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
        "readout_us", "commit_us", "upkeep_us"]
WARMUP = 5
QOS_NAMES = {0: "unspecified", 9: "background", 17: "utility", 21: "default",
             25: "userInitiated", 33: "userInteractive"}


def read_meta(path: Path) -> dict:
    return dict(line.partition("=")[::2]
                for line in path.read_text().splitlines() if "=" in line)


def load(tag: str) -> tuple[list[dict], list[dict]]:
    rounds, headers = [], []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        h = HEADER_RE.match(line)
        if h:
            rec = dict(kv.split("=", 1) for kv in h.group(1).split() if "=" in kv)
            headers.append(rec)
            continue
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {k: float(v) for k, v in KV_RE.findall(m.group(4))}
        rec["round"] = int(m.group(1))
        rec["d"] = int(m.group(2))
        rec["acc"] = int(m.group(3))
        rec["HOSTSUM"] = sum(rec[k] for k in HOST)
        if "e89_thr_user_ns" in rec:
            rec["THRCPU"] = rec["e89_thr_user_ns"] + rec["e89_thr_sys_ns"]
            # The share of the round the submitting thread spent ON a CPU.
            # A slow thread stays near its clean share; a blocked thread drops.
            rec["THRCPU_frac"] = rec["THRCPU"] / (rec["round_us"] * 1000)
            rec["HOSTCPU_frac"] = rec["THRCPU"] / (rec["HOSTSUM"] * 1000)
        rounds.append(rec)
    return rounds, headers


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--json-out", default="research/e89-rung0b.json")
    args = ap.parse_args()

    legs = []
    for prefix in args.prefixes:
        for p in sorted(OUT.glob(prefix + "-*")):
            if not (p / "trace.txt").exists():
                continue
            rounds, headers = load(p.name)
            if not rounds or "e89_probe_ns" not in rounds[0]:
                print(f"skip {p.name}: no probe fields")
                continue
            meta = read_meta(p / "meta.txt")
            score = {}
            sp = p / "score.json"
            if sp.exists():
                score = json.loads(sp.read_text())
            legs.append({"tag": p.name, "arm": meta.get("e89_force_qos", "?"),
                         "position": int(meta.get("e89_position", -1)),
                         "meta": meta, "rounds": rounds, "headers": headers,
                         "score": score})
    if not legs:
        raise SystemExit("no probe legs found")

    # A round is slow when its dependent-chain probe is above the midpoint of
    # the pooled bimodal distribution. The probe never touches the workload, so
    # this classification is independent of the host-phase sum it is tested
    # against.
    probe_all = sorted(r["e89_probe_ns"] for l in legs for r in l["rounds"][WARMUP:]
                       if l["arm"] == "none")
    lo = st.median(probe_all[: len(probe_all) // 4])
    hi = st.median(probe_all[-len(probe_all) // 4:])
    cut = (lo + hi) / 2
    print(f"probe distribution over the {len(probe_all)} post-warmup observation "
          f"rounds:\n  p25-band median {lo:.0f} ns   p75-band median {hi:.0f} ns"
          f"   cut {cut:.0f} ns")

    print(f"\n{'leg':<20}{'qos':<16}{'pos':>4}{'host_med':>9}{'probe_med':>10}"
          f"{'instr_med':>11}{'cyc_med':>10}{'user_ms':>8}{'qos':>4}{'role':>5}"
          f"{'pri':>4}{'tids':>5}{'cache_mb':>9}{'slowfrac':>9}{'trans':>6}")
    for leg in legs:
        rs = leg["rounds"]
        post = rs[WARMUP:]
        h = [r["HOSTSUM"] for r in post]
        pr = [r["e89_probe_ns"] for r in post]
        slow = [r["e89_probe_ns"] > cut for r in post]
        frac = sum(slow) / len(slow)
        # First round after the warm-up at which the leg leaves the slow state
        # and never returns.
        trans = ""
        if 0 < frac < 1:
            for i in range(len(slow)):
                if not any(slow[i:]):
                    trans = str(post[i]["round"])
                    break
        leg["host_med"] = st.median(h)
        leg["probe_med"] = st.median(pr)
        leg["instr_med"] = st.median([r["e89_instr"] for r in post])
        leg["cycles_med"] = st.median([r["e89_cycles"] for r in post])
        leg["user_ms"] = sum(r["e89_user_ns"] for r in post) / 1e6
        leg["slow_frac"] = frac
        leg["transition_round"] = trans
        leg["tids"] = len({r["e89_tid"] for r in rs})
        leg["cache_mb"] = st.median([r["e89_cache_mb"] for r in post])
        leg["active_mb"] = st.median([r["e89_active_mb"] for r in post])
        q = int(post[0]["e89_qos"])
        print(f"{leg['tag']:<20}{leg['arm']:<16}{leg['position']:>4}"
              f"{leg['host_med']:>9.0f}{leg['probe_med']:>10.0f}"
              f"{leg['instr_med']:>11.0f}{leg['cycles_med']:>10.0f}"
              f"{leg['user_ms']:>8.0f}{q:>4}{int(post[0]['e89_role']):>5}"
              f"{int(post[0]['e89_curpri']):>4}{leg['tids']:>5}"
              f"{leg['cache_mb']:>9.0f}{frac:>9.2f}{trans:>6}")

    # ---- the discriminator, at round granularity over the observation legs
    obs = [r for l in legs if l["arm"] == "none" for r in l["rounds"][WARMUP:]]
    slow = [r for r in obs if r["e89_probe_ns"] > cut]
    fast = [r for r in obs if r["e89_probe_ns"] <= cut]
    print(f"\n{'=' * 78}\nRUNG 0B DISCRIMINATOR, pooled over {len(obs)} "
          f"post-warmup observation rounds\n  {len(slow)} slow-probe rounds, "
          f"{len(fast)} fast-probe rounds")
    if not slow or not fast:
        print("  ONE STATE ONLY. No discrimination is possible from this session.")
    else:
        print(f"\n{'quantity':<24}{'fast round':>14}{'slow round':>14}{'ratio':>9}")
        rows = {}
        for key, label in [
            ("e89_probe_ns", "cpu probe ns"),
            ("THRCPU", "thread cpu ns"),
            ("e89_thr_user_ns", "thread user ns"),
            ("e89_thr_sys_ns", "thread system ns"),
            ("e89_instr", "instructions"),
            ("e89_cycles", "cycles"),
            ("e89_user_ns", "user cpu ns"),
            ("e89_sys_ns", "system cpu ns"),
            ("HOSTSUM", "host phase sum us"),
            ("d_submit1_us", "d_submit1 us"),
            ("d_head1_us", "d_head1 us"),
            ("commit_us", "commit us"),
            ("eval_wall_us", "eval wall us"),
            ("verify_build_us", "verify build us"),
            ("round_us", "round us"),
            ("e89_cache_mb", "mlx cache mb"),
            ("e89_active_mb", "mlx active mb"),
        ]:
            a = st.median([r[key] for r in fast])
            b = st.median([r[key] for r in slow])
            rows[key] = {"fast": a, "slow": b, "ratio": b / a if a else None}
            print(f"{label:<24}{a:>14.0f}{b:>14.0f}"
                  + (f"{b / a:>9.2f}" if a else f"{'n/a':>9}"))

        for key, label in [("e89_qos_def", "qos default"),
                           ("e89_qos_util", "qos utility"),
                           ("e89_qos_ui", "qos userInitiated"),
                           ("e89_qos_uix", "qos userInteractive"),
                           ("e89_qos_bg", "qos background"),
                           ("e89_qos_maint", "qos maintenance")]:
            a = st.median([r[key] for r in fast])
            b = st.median([r[key] for r in slow])
            rows[key] = {"fast": a, "slow": b}
            print(f"{label:<24}{a:>14.0f}{b:>14.0f}"
                  + (f"{b / a:>9.2f}" if a else f"{'n/a':>9}"))

        ic = st.median([r["e89_instr"] for r in slow]) / st.median(
            [r["e89_instr"] for r in fast])
        pc = st.median([r["e89_probe_ns"] for r in slow]) / st.median(
            [r["e89_probe_ns"] for r in fast])
        hc = st.median([r["HOSTSUM"] for r in slow]) / st.median(
            [r["HOSTSUM"] for r in fast])
        print(f"\n  instruction ratio {ic:.2f}   probe ratio {pc:.2f}   "
              f"host phase ratio {hc:.2f}")
        tc = st.median([r["THRCPU"] for r in slow]) / st.median(
            [r["THRCPU"] for r in fast])
        print(f"  thread cpu ratio {tc:.2f}")
        if ic < 1.2 and pc > 1.5:
            verdict = "THREAD IS SLOW: same work, longer issue latency"
        elif ic > 2.0 and pc < 1.2:
            verdict = "THREAD IS BUSY: more retired work at unchanged speed"
        elif pc < 1.2 and ic < 1.2 and tc < 1.2 and hc > 2.0:
            verdict = ("THREAD IS BLOCKED: host phases inflate while the thread "
                       "neither runs slower nor retires more work")
        else:
            verdict = "MIXED: no discriminator is clean, report and stop"
        print(f"  verdict: {verdict}")
        rows["verdict"] = verdict
        rows["ratios"] = {"instructions": ic, "cpu_probe": pc,
                          "host_phase_sum": hc, "thread_cpu": tc}

        r_ph = pearson([r["e89_probe_ns"] for r in obs],
                       [r["HOSTSUM"] for r in obs])
        r_ih = pearson([r["e89_instr"] for r in obs], [r["HOSTSUM"] for r in obs])
        print(f"\n  round-level pearson r, probe vs host phase sum: {r_ph:.3f}")
        print(f"  round-level pearson r, instructions vs host phase sum: "
              f"{r_ih:.3f}")

    # ---- stuck-leg rate over the observation legs
    obs_legs = [l for l in legs if l["arm"] == "none"]
    stuck = [l for l in obs_legs if l["slow_frac"] > 0.5]
    part = [l for l in obs_legs if 0.02 < l["slow_frac"] <= 0.5]
    k, n = len(stuck), len(obs_legs)
    ci = wilson(k, n)
    print(f"\nstuck-leg rate {k}/{n} = {k / n:.3f}, Wilson 95 % CI "
          f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  partly slow legs (2 % to 50 % of rounds): {len(part)}")
    print("  per-leg slow fraction: "
          + " ".join(f"{l['slow_frac']:.2f}" for l in obs_legs))

    # ---- controls
    print("\ncontrols")
    for arm in sorted({l["arm"] for l in legs if l["arm"] != "none"}):
        cl = [l for l in legs if l["arm"] == arm]
        base = [l for l in obs_legs if l["slow_frac"] < 0.5]
        bp = st.median([l["probe_med"] for l in base]) if base else float("nan")
        bh = st.median([l["host_med"] for l in base]) if base else float("nan")
        print(f"  {arm:<16} n={len(cl)}  probe_med="
              + ", ".join(f"{l['probe_med']:.0f}" for l in cl)
              + f"  (clean obs leg {bp:.0f})  host_med="
              + ", ".join(f"{l['host_med']:.0f}" for l in cl)
              + f"  (clean obs leg {bh:.0f})")

    # ---- absolute candidate time, clean legs only
    print("\nabsolute candidate seconds per token, harness=local")
    for label, group in [("clean observation legs",
                          [l for l in obs_legs if l["slow_frac"] < 0.5]),
                         ("stuck observation legs", stuck)]:
        spt = [l["score"].get("candidate_mtp_seconds_per_token")
               for l in group if l["score"].get("candidate_mtp_seconds_per_token")]
        if spt:
            print(f"  {label:<26} n={len(spt)}  mean={st.mean(spt):.7f}  "
                  f"min={min(spt):.7f}  max={max(spt):.7f}")
        else:
            print(f"  {label:<26} n=0")

    # ---- process identity across legs
    print("\nprocess identity: header instr0 per leg (a continuing counter means "
          "one process)")
    for leg in legs:
        vals = [h.get("instr0") for h in leg["headers"]]
        tids = [h.get("tid") for h in leg["headers"]]
        print(f"  {leg['tag']:<20} instr0={vals}  tid={tids}  "
              f"forced={[h.get('forced_qos') for h in leg['headers']]}")

    payload = {
        "experiment": "e89-drafting-round-host-state", "rung": "0b",
        "harness": "local", "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False, "official_or_ranked_score": False,
        "probe_cut_ns": cut,
        "discriminator": rows if slow and fast else None,
        "stuck_leg_rate": {"k": k, "n": n, "wilson95": ci},
        "legs": [{kk: vv for kk, vv in l.items()
                  if kk not in ("rounds", "headers")} for l in legs],
    }
    Path(ROOT / args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
