#!/usr/bin/env python3
"""Summarise one E82 head-cost timed session into a per-leg table + JSON.

usage: research/e82_headcost_report.py PREFIX [--out FILE]

Reports the fields the advisor requires per leg: absolute candidate
seconds/token, the local serial:mtp ratio, draft_build_us, rounds for the
decode window, head_provenance_sha256, the loaded head byte count, entry and
exit GPU temperature, and the two ungated-mode flags verbatim.
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
    comp = {k: st.mean(r[k] for r in rounds)
            for k in ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
                      "d_chain_us", "d_submit2_us", "eval_wall_us", "readout_us",
                      "commit_us", "upkeep_us")}
    return comp | {
        "rounds": len(rounds),
        "tokens_committed": tokens,
        "rounds_per_512": len(rounds) * 512.0 / tokens,
        "rows_per_token": sum(r["d"] + 1 for r in rounds) / tokens,
        "mean_d": st.mean(r["d"] for r in rounds),
        "mean_acc": st.mean(r["acc"] for r in rounds),
        "draft_build_us_total": sum(r["draft_build_us"] for r in rounds),
        "draft_build_us_per_round": st.mean(r["draft_build_us"] for r in rounds),
        "verify_build_us_per_round": st.mean(r["verify_build_us"] for r in rounds),
        "round_us_total": sum(r["round_us"] for r in rounds),
    }


def leg(tag: str) -> dict:
    d = OUT / tag
    meta = read_meta(d / "meta.txt")
    score = json.loads((d / "score.json").read_text())
    m = score["metrics"]
    hb, hn = head_bytes(meta.get("head_dir", ""))
    rec = {
        "tag": tag,
        "arm": tag.rsplit("-", 2)[1],
        "rep": int(tag.rsplit("-", 1)[1]),
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
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir() if p.name.startswith(args.prefix + "-"))
    legs = [leg(t) for t in tags]
    legs.sort(key=lambda r: r["started"])

    arms: dict[str, list[dict]] = {}
    for r in legs:
        arms.setdefault(r["arm"], []).append(r)

    ref = "declared"
    summary = {}
    for arm, rows in arms.items():
        spt = [r["candidate_mtp_seconds_per_token"] for r in rows]
        summary[arm] = {
            "n": len(rows),
            "spt_mean": st.mean(spt),
            "spt_legs": spt,
            "spt_spread_pct": (max(spt) - min(spt)) / st.mean(spt) * 100.0,
            "ratio_mean": st.mean(r["local_ratio"] for r in rows),
            "rounds_legs": [r["rounds"] for r in rows],
            "rounds_per_512_legs": [round(r["rounds_per_512"], 2) for r in rows],
            "rows_per_token_legs": [round(r["rows_per_token"], 4) for r in rows],
            "draft_build_us_per_round": st.mean(r["draft_build_us_per_round"] for r in rows),
            "head_loaded_bytes": rows[0]["head_loaded_bytes"],
            "head_provenance_sha256": sorted({r["head_provenance_sha256"] for r in rows}),
            "all_tokens_matched": all(r["all_tokens_matched"] for r in rows),
        }
    base = summary[ref]["spt_mean"]
    for arm, s in summary.items():
        s["spt_delta_pct_vs_declared"] = (s["spt_mean"] - base) / base * 100.0

    print(f"{'arm':<12} {'n':>2} {'s/tok mean':>12} {'Δ% vs decl':>11} "
          f"{'spread%':>8} {'ratio':>7} {'rounds':>14} {'dbuild us/rd':>13} {'head MB':>9}")
    for arm in sorted(summary, key=lambda a: summary[a]["spt_mean"]):
        s = summary[arm]
        print(f"{arm:<12} {s['n']:>2} {s['spt_mean']:>12.6f} "
              f"{s['spt_delta_pct_vs_declared']:>+11.3f} {s['spt_spread_pct']:>8.2f} "
              f"{s['ratio_mean']:>7.4f} {str(s['rounds_legs']):>14} "
              f"{s['draft_build_us_per_round']:>13.0f} "
              f"{s['head_loaded_bytes'] / 1e6:>9.2f}")

    print("\nper-leg detail (chronological):")
    for r in legs:
        print(f"  {r['tag']:<28} spt={r['candidate_mtp_seconds_per_token']:.6f} "
              f"ratio={r['local_ratio']:.4f} rounds={r['rounds']} "
              f"rows/tok={r['rows_per_token']:.4f} "
              f"dbuild={r['draft_build_us_per_round']:.0f}us "
              f"T={r['gpu_temp_entry_c']:.1f}->{r['gpu_temp_exit_c']:.1f}C "
              f"matched={r['all_tokens_matched']} "
              f"head={r['head_provenance_sha256'][:12]}/{r['head_loaded_bytes']}")

    doc = {
        "prefix": args.prefix,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "host": "Apple M4 Pro 48GB (not ranked M5)",
        "legs": legs,
        "summary": summary,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
