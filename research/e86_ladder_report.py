#!/usr/bin/env python3
"""Summarise one E86 decode-ladder timed session into a table + JSON.

usage: research/e86_ladder_report.py PREFIX [--out FILE] [--ref default]

Two jobs:

1.  Rung 0 decomposition. The `off` arm removes every asyncEval inside the
    verify-build window, so with `--sync-head` no GPU work is in flight there:
    `verify_build_us` becomes pure host encode `H` and `eval_wall_us` becomes
    the full GPU cost `G` of the verify graph. Every other arm's
    `H + G - (verify_build_us + eval_wall_us)` is the overlap the ladder buys.

2.  Rung 1 ranking on ABSOLUTE candidate MTP seconds per token, against the
    session null measured by the two `default` legs at the ends of the
    palindrome.

The ladder is pure enqueue timing, so the round count and rows_per_token must
be identical on every arm. The report fails loudly when they are not.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")
PHASES = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us",
          "d_submit2_us", "verify_build_us", "eval_wall_us", "readout_us",
          "commit_us", "upkeep_us", "draft_build_us", "round_us")


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k] = v
    return meta


def head_bytes(head_dir: str) -> tuple[int, int]:
    d = Path(head_dir)
    if not d.is_dir():
        return (-1, -1)
    files = sorted(p for p in d.rglob("*") if p.is_file())
    return (sum(p.stat().st_size for p in files), len(files))


def parse_trace(path: Path) -> dict:
    rounds = []
    for line in path.read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1)), "d": int(m.group(2)), "acc": int(m.group(3))}
        rec.update({k: float(v) for k, v in KV_RE.findall(m.group(4))})
        rounds.append(rec)
    if not rounds:
        return {}
    tokens = sum(r["acc"] + 1 for r in rounds)
    # Median per phase: the first round pays cold-kernel first touch and would
    # otherwise dominate a mean over ~78 rounds.
    out = {f"{k}_med": st.median(r[k] for r in rounds) for k in PHASES}
    out.update({f"{k}_mean": st.mean(r[k] for r in rounds) for k in PHASES})
    return out | {
        "rounds": len(rounds),
        "tokens_committed": tokens,
        "rounds_per_512": len(rounds) * 512.0 / tokens,
        "rows_per_token": sum(r["d"] + 1 for r in rounds) / tokens,
        "mean_d": st.mean(r["d"] for r in rounds),
        "mean_acc": st.mean(r["acc"] for r in rounds),
        "round_us_total": sum(r["round_us"] for r in rounds),
    }


def leg(prefix: str, tag: str) -> dict:
    d = OUT / tag
    meta = read_meta(d / "meta.txt")
    score = json.loads((d / "score.json").read_text())
    m = score["metrics"]
    arm, _, rep = tag[len(prefix) + 1:].rpartition("-")
    hb, hn = head_bytes(meta.get("head_dir", ""))
    rec = {
        "tag": tag,
        "arm": arm,
        "rep": int(rep),
        "ladder": meta.get("ladder"),
        "sync_head": meta.get("sync_head"),
        "candidate_mtp_seconds_per_token": m["mtp_seconds_per_token"],
        "serial_seconds_per_token": m["serial_seconds_per_token"],
        "local_ratio": m["mtp_decode_speedup"],
        "decode_tokens": m["decode_tokens"],
        "all_tokens_matched": m["all_tokens_matched"],
        "accepted_draft_rate": m["accepted_draft_rate"],
        "effective_mean_draft_len": m["effective_mean_draft_len"],
        "residual_divergence_count": m["residual_divergence_count"],
        "head_provenance_sha256": m["head_provenance_sha256"],
        "uses_pinned_mtp_head": m["uses_pinned_mtp_head"],
        "head_dir": meta.get("head_dir"),
        "head_loaded_bytes": hb,
        "head_loaded_files": hn,
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "base_sha": meta["base_sha"],
        "worker_sha256": meta["worker_sha256"],
        "started": meta["started"],
    }
    rec.update(parse_trace(d / "trace.txt"))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="default")
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(args.prefix + "-") and (p / "score.json").exists())
    legs = [leg(args.prefix, t) for t in tags]
    legs.sort(key=lambda r: r["started"])

    arms: dict[str, list[dict]] = {}
    for r in legs:
        arms.setdefault(r["arm"], []).append(r)

    summary = {}
    for arm, rows in arms.items():
        spt = [r["candidate_mtp_seconds_per_token"] for r in rows]
        summary[arm] = {
            "n": len(rows),
            "ladder": rows[0]["ladder"],
            "spt_mean": st.mean(spt),
            "spt_legs": spt,
            "spt_spread_pct": (max(spt) - min(spt)) / st.mean(spt) * 100.0,
            "ratio_mean": st.mean(r["local_ratio"] for r in rows),
            "rounds_legs": [r["rounds"] for r in rows],
            "rows_per_token_legs": [round(r["rows_per_token"], 6) for r in rows],
            "verify_build_us_med": st.mean(r["verify_build_us_med"] for r in rows),
            "eval_wall_us_med": st.mean(r["eval_wall_us_med"] for r in rows),
            "d_submit2_us_med": st.mean(r["d_submit2_us_med"] for r in rows),
            "round_us_med": st.mean(r["round_us_med"] for r in rows),
            "all_tokens_matched": all(r["all_tokens_matched"] for r in rows),
            "temp_entry_c": [round(r["gpu_temp_entry_c"], 1) for r in rows],
        }

    ref = args.ref if args.ref in summary else sorted(summary)[0]
    base = summary[ref]["spt_mean"]
    null_pct = summary[ref]["spt_spread_pct"]
    for s in summary.values():
        s["spt_delta_pct_vs_ref"] = (s["spt_mean"] - base) / base * 100.0
        s["beats_session_null"] = s["spt_delta_pct_vs_ref"] < -null_pct

    # Bit-exactness witness: enqueue timing may not move work.
    round_sets = {tuple(sorted(set(s["rounds_legs"]))) for s in summary.values()}
    rows_sets = {tuple(sorted(set(s["rows_per_token_legs"]))) for s in summary.values()}
    bit_exact_work = len(round_sets) == 1 and len(rows_sets) == 1

    decomp = None
    if "off" in summary:
        H = summary["off"]["verify_build_us_med"]
        G = summary["off"]["eval_wall_us_med"]
        decomp = {
            "host_encode_H_us": H,
            "gpu_execute_G_us": G,
            "H_plus_G_us": H + G,
            "max_H_G_us": max(H, G),
            "off_round_us_med": summary["off"]["round_us_med"],
            "gpu_idle_fraction_of_round_off": H / summary["off"]["round_us_med"],
            "per_arm": {},
        }
        for arm, s in summary.items():
            win = s["verify_build_us_med"] + s["eval_wall_us_med"]
            decomp["per_arm"][arm] = {
                "verify_plus_eval_us": win,
                "overlap_recovered_us": (H + G) - win,
                "overlap_fraction_of_H": ((H + G) - win) / H,
                "round_us_med": s["round_us_med"],
                "round_delta_vs_off_us": s["round_us_med"] - summary["off"]["round_us_med"],
            }

    print(f"session null (|spread| of the two `{ref}` legs): {null_pct:.3f} %\n")
    print(f"{'arm':<26} {'n':>2} {'s/tok mean':>12} {'Δ% vs ref':>10} {'spread%':>8} "
          f"{'vbuild us':>10} {'eval us':>9} {'round us':>9} {'rounds':>10} {'rows/tok':>9}")
    for arm in sorted(summary, key=lambda a: summary[a]["spt_mean"]):
        s = summary[arm]
        print(f"{arm:<26} {s['n']:>2} {s['spt_mean']:>12.6f} "
              f"{s['spt_delta_pct_vs_ref']:>+10.3f} {s['spt_spread_pct']:>8.3f} "
              f"{s['verify_build_us_med']:>10.0f} {s['eval_wall_us_med']:>9.0f} "
              f"{s['round_us_med']:>9.0f} {str(s['rounds_legs']):>10} "
              f"{s['rows_per_token_legs'][0]:>9.4f}")

    print(f"\nbit_exact_work (identical rounds and rows/token on every arm): {bit_exact_work}")

    if decomp:
        print("\nrung 0 decomposition, from the `off` arm (median us per round):")
        print(f"  host encode  H = {decomp['host_encode_H_us']:.0f} us")
        print(f"  gpu execute  G = {decomp['gpu_execute_G_us']:.0f} us")
        print(f"  H + G          = {decomp['H_plus_G_us']:.0f} us")
        print(f"  max(H, G)      = {decomp['max_H_G_us']:.0f} us")
        print(f"  off round      = {decomp['off_round_us_med']:.0f} us")
        print(f"  GPU idle fraction of the off round = "
              f"{decomp['gpu_idle_fraction_of_round_off'] * 100:.1f} %")
        print(f"\n  {'arm':<26} {'vbuild+eval':>12} {'overlap us':>11} "
              f"{'overlap/H':>10} {'round Δ vs off':>15}")
        for arm in sorted(decomp["per_arm"], key=lambda a: decomp["per_arm"][a]["round_us_med"]):
            p = decomp["per_arm"][arm]
            print(f"  {arm:<26} {p['verify_plus_eval_us']:>12.0f} "
                  f"{p['overlap_recovered_us']:>11.0f} "
                  f"{p['overlap_fraction_of_H'] * 100:>9.1f}% "
                  f"{p['round_delta_vs_off_us']:>+15.0f}")

    print("\nper-leg detail (chronological):")
    for r in legs:
        print(f"  {r['tag']:<40} spt={r['candidate_mtp_seconds_per_token']:.6f} "
              f"rounds={r['rounds']} rows/tok={r['rows_per_token']:.4f} "
              f"vbuild={r['verify_build_us_med']:.0f} eval={r['eval_wall_us_med']:.0f} "
              f"T={r['gpu_temp_entry_c']:.1f}->{r['gpu_temp_exit_c']:.1f}C "
              f"matched={r['all_tokens_matched']}")

    doc = {
        "prefix": args.prefix,
        "experiment": "e86-decode-asynceval-ladder-and-host-gpu-split",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "host": "Apple M4 Pro 48GB (not ranked M5)",
        "reference_arm": ref,
        "session_null_pct": null_pct,
        "bit_exact_work": bit_exact_work,
        "decomposition": decomp,
        "legs": legs,
        "summary": summary,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
