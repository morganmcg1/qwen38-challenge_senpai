#!/usr/bin/env python3
"""Read one rung-5e ABBA session and report the pre-registered quantities.

    usage: research/e120_rung5e_report.py research/out/TAG

The headline is ABSOLUTE candidate MTP seconds per token against the same-binary
`off` arm. The local serial-to-MTP ratio is reported beside it because the
mechanism is confined to the MTP leg, but the absolute number is the claim.
"""

import argparse
import json
import statistics
from pathlib import Path

# Pre-registered in research/e120-results.md at commit 31b33371.
PREDICTED_GAIN_PCT = 2.5
NOTIFY_BELOW_PCT = 1.25
NOTIFY_ABOVE_PCT = 5.0
SERIAL_CONTROL_TOLERANCE_PCT = 0.5


def load(out_dir: Path):
    legs = []
    for p in sorted(out_dir.glob("score.*.json")):
        label = p.name[len("score.") : -len(".json")]
        m = json.loads(p.read_text())["metrics"]
        legs.append({"label": label, "arm": label.split("-", 1)[1], "metrics": m})
    temps = {}
    arms_file = out_dir / "arms.jsonl"
    if arms_file.exists():
        for line in arms_file.read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                temps[a["label"]] = a
    return legs, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    legs, temps = load(args.out_dir)
    if not legs:
        raise SystemExit(f"no score.*.json under {args.out_dir}")

    print("== legs ==")
    print(
        f"{'leg':<14}{'tokens':>7}{'depth':>6}{'serial_spt':>12}{'mtp_spt':>12}"
        f"{'ratio':>8}{'eff_draft':>10}{'acc_rate':>9}{'matched':>8}{'div':>5}"
        f"{'T_in':>7}{'T_out':>7}"
    )
    for leg in legs:
        m = leg["metrics"]
        t = temps.get(leg["label"], {})
        ti = t.get("gpu_temp_entry_c")
        to = t.get("gpu_temp_exit_c")
        print(
            f"{leg['label']:<14}{m['decode_tokens']:>7}{m['mtp_depth']:>6}"
            f"{m['serial_seconds_per_token']:>12.6f}{m['mtp_seconds_per_token']:>12.6f}"
            f"{m['mtp_decode_speedup']:>8.4f}{m['effective_mean_draft_len']:>10.3f}"
            f"{m['accepted_draft_rate']:>9.4f}{str(m['all_tokens_matched']):>8}"
            f"{m['residual_divergence_count']:>5}"
            f"{(f'{ti:.1f}' if ti is not None else '-'):>7}"
            f"{(f'{to:.1f}' if to is not None else '-'):>7}"
        )

    by_arm = {}
    for leg in legs:
        by_arm.setdefault(leg["arm"], []).append(leg["metrics"])

    def med(arm, key):
        return statistics.median(m[key] for m in by_arm[arm])

    print("\n== per-arm medians ==")
    print(f"{'arm':<12}{'n':>3}{'serial_spt':>12}{'mtp_spt':>12}{'ratio':>9}{'eff_draft':>11}")
    for arm in by_arm:
        print(
            f"{arm:<12}{len(by_arm[arm]):>3}{med(arm, 'serial_seconds_per_token'):>12.6f}"
            f"{med(arm, 'mtp_seconds_per_token'):>12.6f}"
            f"{med(arm, 'mtp_decode_speedup'):>9.4f}"
            f"{med(arm, 'effective_mean_draft_len'):>11.3f}"
        )

    if "off" not in by_arm or len(by_arm) < 2:
        print("\nsingle-arm session: no comparison")
        return

    cand = next(a for a in by_arm if a != "off")
    base_mtp, cand_mtp = med("off", "mtp_seconds_per_token"), med(cand, "mtp_seconds_per_token")
    base_ser, cand_ser = med("off", "serial_seconds_per_token"), med(cand, "serial_seconds_per_token")
    gain = 100 * (base_mtp - cand_mtp) / base_mtp
    serial_delta = 100 * (cand_ser - base_ser) / base_ser
    ratio_gain = 100 * (med(cand, "mtp_decode_speedup") / med("off", "mtp_decode_speedup") - 1)

    print(f"\n== headline: absolute candidate MTP seconds per token, {cand} against off ==")
    print(f"  off  mtp_spt      {base_mtp:.6f}")
    print(f"  {cand:<4} mtp_spt      {cand_mtp:.6f}")
    print(f"  absolute gain     {gain:+.3f}%   (pre-registered {PREDICTED_GAIN_PCT:.2f}%)")
    print(f"  local ratio gain  {ratio_gain:+.3f}%")
    print(f"\n== controls ==")
    print(
        f"  serial spt delta  {serial_delta:+.3f}%   "
        f"({'PASS' if abs(serial_delta) < SERIAL_CONTROL_TOLERANCE_PCT else 'FAIL'}, "
        f"tolerance {SERIAL_CONTROL_TOLERANCE_PCT}%)"
    )
    matched = all(m["all_tokens_matched"] for ms in by_arm.values() for m in ms)
    div = max(m["residual_divergence_count"] for ms in by_arm.values() for m in ms)
    print(f"  all_tokens_matched every leg   {matched}")
    print(f"  residual_divergence_count max  {div}")

    verdict = "within band, continue to 5f"
    if gain < NOTIFY_BELOW_PCT or gain > NOTIFY_ABOVE_PCT:
        verdict = (
            f"OUTSIDE the {NOTIFY_BELOW_PCT} to {NOTIFY_ABOVE_PCT}% notify band: "
            "report to the advisor before continuing"
        )
    print(f"\n  verdict: {verdict}")

    summary = {
        "candidate_arm": cand,
        "base_mtp_seconds_per_token": base_mtp,
        "candidate_mtp_seconds_per_token": cand_mtp,
        "absolute_mtp_gain_pct": gain,
        "local_ratio_gain_pct": ratio_gain,
        "serial_spt_delta_pct": serial_delta,
        "all_tokens_matched": matched,
        "residual_divergence_count_max": div,
        "predicted_gain_pct": PREDICTED_GAIN_PCT,
        "notify_band_pct": [NOTIFY_BELOW_PCT, NOTIFY_ABOVE_PCT],
        "verdict": verdict,
        "legs": legs,
    }
    (args.out_dir / "rung5e_report.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out_dir / 'rung5e_report.json'}")


if __name__ == "__main__":
    main()
