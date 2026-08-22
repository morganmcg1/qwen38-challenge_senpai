#!/usr/bin/env python3
"""Read the E134 same-binary ABBA arm session and price ship -> pb6.

The estimator is the within-replicate contrast. With order A B B A both arms
have mean position 2.5, so subtracting the two ship legs from the two pb6 legs
inside one replicate cancels any linear thermal drift exactly. That is the
whole reason the order is what it is, and it is why the per-replicate contrast,
not the pooled arm mean, is the number that gets reported.

Two bases are printed for every contrast:

  decode      parent_measured_seconds_per_token, which is what the arm moves
  ranked      (seed_prefill_seconds + decode_seconds) / decode_token_count

The ranked basis is the honest one for the score, because the ranked leg times
seed processing and decoding in one window and no depth policy touches the
seed. The decode basis is the honest one for the mechanism. Reporting only the
first would overstate the score effect by about 1.31x.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RUNS_PARENT = Path(".mlxfast-private/e128/runs-abba")
PROMPT_ID = "benchfixture"
PREREG = Path("research/e134-artifacts/abba-preregistration.json")


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def collect(label: str) -> list[dict]:
    legs = []
    for slot_dir in sorted(RUNS_PARENT.glob(f"{label}k*p*")):
        meta_path = slot_dir / PROMPT_ID / "meta.txt"
        report_path = slot_dir / PROMPT_ID / "report.json"
        if not meta_path.exists() or not report_path.exists():
            continue
        meta = read_meta(meta_path)
        report = json.loads(report_path.read_text())
        decode_spt = report["parent_measured_seconds_per_token"]
        tokens = report["decode_token_count"]
        ranked_spt = (
            report["seed_prefill_seconds"] + report["decode_seconds"]
        ) / tokens
        legs.append(
            {
                "slot": slot_dir.name,
                "arm": meta.get("e134_arm_requested", "?"),
                "replicate": int(meta.get("e134_replicate", "0")),
                "position": int(meta.get("e134_position", "0")),
                "witness": meta.get("e134_arm_witness", "absent"),
                "rounds_want": meta.get("e134_arm_witness_rounds_want", "?"),
                "round_count": report["round_count"],
                "mean_draft": report["effective_mean_draft_len"],
                "declared_rows": report["declared_rows_total"],
                "decode_spt": decode_spt,
                "ranked_spt": ranked_spt,
                "seed_prefill_seconds": report["seed_prefill_seconds"],
                "all_tokens_matched": report["all_tokens_matched"],
                "residual_divergence_count": report["residual_divergence_count"],
                "entry_c": meta.get("gpu_temp_entry_c", ""),
                "exit_c": meta.get("gpu_temp_exit_c", ""),
                "timing_valid": meta.get("timing_valid", "?"),
                "cool_gate_passed_real_gate": meta.get(
                    "cool_gate_passed_real_gate", "?"),
                "gate_qualified_for_timing": meta.get(
                    "gate_qualified_for_timing", "?"),
                "commit": meta.get("e134_session_commit", "?"),
                "worker": meta.get("e134_session_worker_sha256", "?"),
            }
        )
    return legs


def pct(ship: float, pb6: float) -> float:
    return (ship - pb6) / ship * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="s1")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    legs = collect(args.label)
    if not legs:
        print(f"e134_abba_report: no legs under {RUNS_PARENT}/{args.label}k*")
        return 1

    print(f"E134 same-binary ABBA arm session, label {args.label}")
    print(f"legs {len(legs)}   fixture {PROMPT_ID}")
    print()

    # Identity. One commit and one worker binary across the whole session is
    # the entire point of the design, so it is asserted rather than described.
    commits = {leg["commit"] for leg in legs}
    workers = {leg["worker"] for leg in legs}
    print(f"session_commit        {' '.join(sorted(commits))}")
    print(f"session_worker_sha256 {' '.join(sorted(workers))}")
    identity_ok = len(commits) == 1 and len(workers) == 1
    print(f"one binary, one commit: {identity_ok}")
    print()

    print("gate labels, verbatim from every leg")
    for field in ("timing_valid", "cool_gate_passed_real_gate",
                  "gate_qualified_for_timing"):
        values = sorted({leg[field] for leg in legs})
        print(f"  {field:<28} {' '.join(values)}")
    print("  official_or_ranked_score     false")
    print()

    exact_ok = all(
        leg["all_tokens_matched"] is True
        and leg["residual_divergence_count"] == 0
        for leg in legs
    )
    witness_bad = [leg for leg in legs if leg["witness"] != "ok"]
    print(f"exactness on every leg: {exact_ok}")
    print(f"witness mismatches:     {len(witness_bad)}")
    for leg in witness_bad:
        print(f"  {leg['slot']} wanted {leg['rounds_want']} rounds,"
              f" ran {leg['round_count']}")
    print()

    header = ("slot", "arm", "rep", "pos", "rounds", "draft", "rows",
              "decode_us", "ranked_us", "inC", "outC", "wit")
    print(f"{header[0]:<14}{header[1]:<6}{header[2]:>4}{header[3]:>4}"
          f"{header[4]:>7}{header[5]:>8}{header[6]:>6}{header[7]:>11}"
          f"{header[8]:>11}{header[9]:>7}{header[10]:>7}  {header[11]}")
    for leg in legs:
        print(f"{leg['slot']:<14}{leg['arm']:<6}{leg['replicate']:>4}"
              f"{leg['position']:>4}{leg['round_count']:>7}"
              f"{leg['mean_draft']:>8.3f}{leg['declared_rows']:>6}"
              f"{leg['decode_spt'] * 1e6:>11.1f}"
              f"{leg['ranked_spt'] * 1e6:>11.1f}"
              f"{leg['entry_c'] or '-':>7}{leg['exit_c'] or '-':>7}"
              f"  {leg['witness']}")
    print()

    entries = [float(leg["entry_c"]) for leg in legs if leg["entry_c"]]
    if entries:
        print(f"entry temperature spread {min(entries):.1f} to"
              f" {max(entries):.1f} C"
              f"  (range {max(entries) - min(entries):.1f} C)")
    seeds = [leg["seed_prefill_seconds"] for leg in legs]
    print(f"seed prefill {min(seeds):.3f} to {max(seeds):.3f} s,"
          f" arm-invariant and untouchable by any depth policy")
    print()

    usable = [leg for leg in legs if leg["witness"] == "ok"]
    replicates = sorted({leg["replicate"] for leg in usable})
    contrasts = []
    print("within-replicate contrast, ship -> pb6"
          " (positive means pb6 is faster)")
    print(f"{'rep':>4}{'ship_decode':>13}{'pb6_decode':>12}{'decode_pct':>12}"
          f"{'ranked_pct':>12}{'d_rounds':>10}")
    for rep in replicates:
        rows = [leg for leg in usable if leg["replicate"] == rep]
        ship = [leg for leg in rows if leg["arm"] == "ship"]
        pb6 = [leg for leg in rows if leg["arm"] == "pb6"]
        if len(ship) != 2 or len(pb6) != 2:
            print(f"{rep:>4}  incomplete: {len(ship)} ship, {len(pb6)} pb6")
            continue
        ship_d = statistics.fmean(leg["decode_spt"] for leg in ship)
        pb6_d = statistics.fmean(leg["decode_spt"] for leg in pb6)
        ship_r = statistics.fmean(leg["ranked_spt"] for leg in ship)
        pb6_r = statistics.fmean(leg["ranked_spt"] for leg in pb6)
        d_rounds = (statistics.fmean(leg["round_count"] for leg in pb6)
                    - statistics.fmean(leg["round_count"] for leg in ship))
        entry = {
            "replicate": rep,
            "ship_decode_spt": ship_d,
            "pb6_decode_spt": pb6_d,
            "decode_pct": pct(ship_d, pb6_d),
            "ship_ranked_spt": ship_r,
            "pb6_ranked_spt": pb6_r,
            "ranked_pct": pct(ship_r, pb6_r),
            "delta_rounds": d_rounds,
        }
        contrasts.append(entry)
        print(f"{rep:>4}{ship_d * 1e6:>13.1f}{pb6_d * 1e6:>12.1f}"
              f"{entry['decode_pct']:>+12.4f}{entry['ranked_pct']:>+12.4f}"
              f"{d_rounds:>+10.1f}")
    print()

    if not contrasts:
        print("e134_abba_report: no complete replicate; nothing to price")
        return 1

    decode_pcts = [entry["decode_pct"] for entry in contrasts]
    ranked_pcts = [entry["ranked_pct"] for entry in contrasts]
    decode_mean = statistics.fmean(decode_pcts)
    ranked_mean = statistics.fmean(ranked_pcts)
    decode_sd = statistics.stdev(decode_pcts) if len(decode_pcts) > 1 else 0.0
    ranked_sd = statistics.stdev(ranked_pcts) if len(ranked_pcts) > 1 else 0.0

    print(f"PRIMARY   decode basis  {decode_mean:+.4f} % "
          f"(sd {decode_sd:.4f}, n {len(decode_pcts)})")
    print(f"SECONDARY ranked basis  {ranked_mean:+.4f} % "
          f"(sd {ranked_sd:.4f}, n {len(ranked_pcts)})")
    if ranked_mean:
        print(f"dilution factor decode/ranked {decode_mean / ranked_mean:.3f}")
    print()

    verdict = "PROCEED"
    if decode_mean <= 0.0:
        verdict = "BLOCK"
    elif decode_mean < 1.0:
        verdict = "ESCALATE"
    if witness_bad or not exact_ok or not identity_ok:
        verdict = "INVALID"
    print(f"pre-registered verdict on the primary: {verdict}")
    if PREREG.exists():
        rule = json.loads(PREREG.read_text())["decision_rule"]
        print(f"  rule: {rule.get(verdict.lower(), rule['invalidate'])}")
    print()
    print("This is directional causal evidence inside one counterbalanced")
    print("session on one fixture. It is not gate-qualified and it is not a")
    print("ranked score. The ruling is one-sided: this host's width-6 cliff is")
    print("steeper than the runner's, so a win here transfers nothing.")

    if args.json:
        payload = {
            "label": args.label,
            "fixture": PROMPT_ID,
            "identity_ok": identity_ok,
            "exactness_ok": exact_ok,
            "witness_mismatches": len(witness_bad),
            "session_commit": sorted(commits),
            "session_worker_sha256": sorted(workers),
            "entry_temp_c": entries,
            "legs": legs,
            "contrasts": contrasts,
            "decode_pct_mean": decode_mean,
            "decode_pct_sd": decode_sd,
            "ranked_pct_mean": ranked_mean,
            "ranked_pct_sd": ranked_sd,
            "verdict": verdict,
            "harness": "local",
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
