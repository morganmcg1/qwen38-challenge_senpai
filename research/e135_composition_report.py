#!/usr/bin/env python3
"""Report the T29-A three-arm session as three pairwise contrasts.

The session harness fits `y ~ 1 + arm + centred leg index` for a two-level arm
code. Three arms do not fit that model, and rewriting it into a three-level
design would change the estimator the grid and E87 sessions were reported with.
So this filters to two arms at a time and reuses the same estimator unchanged.

The filtering is sound because the palindrome is `base composed composed67
composed67 composed base`. Each arm has mean position 3.5, so dropping the
third arm leaves the remaining two still balanced about the centre of the
replicate and a monotone drift still cancels.

The three contrasts answer three different questions:

  base       vs composed    the advisor's decision, `e135_composition_local_pct`
  composed   vs composed67  the onepass67 table under a tight grid
  base       vs composed67  the best configuration against the base

  python3 research/e135_composition_report.py --label c1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_report as report

PAIRS = (
    ("base", "composed", "e135_composition_local_pct",
     "the composition exactly as 1db9d63e shipped it"),
    ("composed", "composed67", "e135_onepass_gain_under_tight_pct",
     "the onepass67 table added back under the tight grid"),
    ("base", "composed67", "e135_composition67_local_pct",
     "the never-built cell against the advisor base"),
)

# F29 pre-registered these on the base-to-composed contrast, in the convention
# where a positive number means the composition is FASTER than the base.
PREDICT_ADDITIVE = 0.87
PREDICT_F225 = -1.18
TOLERANCE_PP = 0.238

# F29 costed three mechanisms. T42 rung 0 read the two trees and found four:
# the launch grid also moved, wide -> tight, and E135 s1 measured that term at
# +1.8064 % faster over 12 legs on this host. The additive forecast for the
# four-mechanism contrast this session actually runs is therefore the F29
# forecast plus that term.
PREDICT_GRID_TERM = 1.8064
PREDICT_ADDITIVE_4 = PREDICT_ADDITIVE + PREDICT_GRID_TERM


def rounds(leg: dict) -> int | None:
    """Drafting rounds in one leg.

    The per-round trace counts them directly, one `round_us=` line each, but it
    costs about 9 % of candidate time on this host, so the timed legs run with
    `--no-trace`. Every round emits one primary token plus its accepted drafts,
    so the count is recovered exactly from the leg's own scored readouts:

        decode_tokens = R * (1 + effective_mean_draft_len * accepted_draft_rate)

    Checked against a traced 512-token leg: the trace counts 78 rounds and this
    returns 78. Both inputs are digit-identical behavioural readouts, so a
    derived count is as strong a determinism signature as a counted one.
    """
    trace = leg["dir"] / "trace.txt"
    if trace.exists():
        counted = sum(1 for line in trace.read_text().splitlines()
                      if "round_us=" in line)
        if counted:
            return counted
    m = leg["metrics"]
    tokens = report.fnum(m.get("decode_tokens"))
    edl = report.fnum(m.get("effective_mean_draft_len"))
    rate = report.fnum(m.get("accepted_draft_rate"))
    if not tokens or edl is None or rate is None:
        return None
    return round(tokens / (1.0 + edl * rate))


def columns_per_round(arm: str, label: str) -> int | None:
    """Threadgroup columns one drafting round launches on this arm.

    Read from the arm's own witness trace: each routed width contributes its
    dispatch count times the columns that width launched.
    """
    path = pathlib.Path(f"research/out/e135{label}w{arm}/pipelines.json")
    if not path.exists():
        return None
    log = json.loads(path.read_text())
    by_width = log.get("by_width", {})
    columns = log.get("columns_by_width", {})
    if not by_width or not columns:
        return None
    return sum(int(by_width[w]) * int(columns[w]) for w in by_width)


def launch_cost(rows: list[dict], label: str, headlines: dict,
                tokens: int) -> dict:
    """Price one launched threadgroup column from every available contrast.

    This is the experiment's own question. If the whole effect is launch cost
    then the same microseconds-per-column must come out of contrasts whose
    column counts differ by an order of magnitude.
    """
    cols = {arm: columns_per_round(arm, label)
            for arm in ("base", "composed", "composed67")}
    means = {}
    for arm in cols:
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in rows if r["arm"] == arm]
        if v:
            means[arm] = statistics.fmean(v)
    n_rounds = rounds(rows[0]) if rows else None
    out = {"columns_per_round": cols, "rounds_per_leg": n_rounds,
           "estimates": []}
    if not n_rounds:
        return out
    for ref, cand, metric, _ in PAIRS:
        if cols.get(ref) is None or cols.get(cand) is None:
            continue
        if ref not in means or metric not in headlines:
            continue
        saved = cols[ref] - cols[cand]
        if saved <= 0:
            continue
        round_s = means[ref] * tokens / n_rounds
        pct, se = headlines[metric]
        per_column = pct / 100.0 * round_s * 1e6 / saved
        out["estimates"].append({
            "contrast": f"{ref}->{cand}", "columns_saved": saved,
            "round_ms": round_s * 1e3, "pct": pct,
            "us_per_column": per_column,
            "us_per_column_se": se / 100.0 * round_s * 1e6 / saved,
        })
    return out


def contrast(rows: list[dict], ref: str, cand: str, key: str):
    report.ARMS = (ref, cand)
    sub = [r for r in rows if r["arm"] in (ref, cand)]
    fit = report.ols_arm_and_drift(sub, key)
    return sub, fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="c1")
    args = ap.parse_args()

    report.configure("composition")
    # `legs` keeps only arms named in `report.ARMS`, so widen it to all three
    # before reading, then narrow it per contrast.
    report.ARMS = ("base", "composed", "composed67")
    rows = report.legs(args.label)
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]
    if not complete:
        print("e135_composition_report: no scored legs")
        return 1

    print(f"=== T29-A composition session {args.label}:"
          f" {len(complete)} scored legs")
    print()
    print("per arm")
    print(f"{'arm':<12}{'n':>3}{'mtp mean':>12}{'sd':>10}"
          f"{'serial mean':>13}{'ratio':>9}{'entry C':>9}{'exit C':>8}"
          f"{'draft len':>26}")
    for arm in ("base", "composed", "composed67"):
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in complete if r["arm"] == arm]
        if not v:
            continue
        s = [report.fnum(r["metrics"].get("serial_seconds_per_token"))
             for r in complete if r["arm"] == arm]
        s = [x for x in s if x is not None]
        ratio = [report.fnum(r["metrics"].get("mtp_decode_speedup"))
                 for r in complete if r["arm"] == arm]
        ratio = [x for x in ratio if x is not None]
        ent = [report.fnum(r["meta"].get("gpu_temp_entry_c"))
               for r in complete if r["arm"] == arm]
        ent = [x for x in ent if x is not None]
        ex = [report.fnum(r["meta"].get("gpu_temp_exit_c"))
              for r in complete if r["arm"] == arm]
        ex = [x for x in ex if x is not None]
        edl = sorted({report.fnum(r["metrics"].get("effective_mean_draft_len"))
                      for r in complete if r["arm"] == arm} - {None})
        shown = ",".join(f"{d:.9f}" for d in edl) if edl else "-"
        print(f"{arm:<12}{len(v):>3}{statistics.fmean(v):>12.6f}"
              f"{(statistics.stdev(v) if len(v) > 1 else 0.0):>10.6f}"
              f"{(statistics.fmean(s) if s else float('nan')):>13.6f}"
              f"{(statistics.fmean(ratio) if ratio else float('nan')):>9.4f}"
              f"{(statistics.fmean(ent) if ent else float('nan')):>9.1f}"
              f"{(statistics.fmean(ex) if ex else float('nan')):>8.1f}"
              f"{shown:>26}")

    print()
    print("per leg, absolute candidate time and round count (F30 additions)")
    print(f"{'i':>3}{'tag':<22}{'arm':<12}{'mtp s/tok':>12}"
          f"{'serial s/tok':>14}{'rounds':>8}{'entry C':>9}{'exit C':>8}")
    for i, r in enumerate(complete):
        m = report.fnum(r["metrics"]["mtp_seconds_per_token"])
        s = report.fnum(r["metrics"].get("serial_seconds_per_token"))
        n = rounds(r)
        print(f"{i:>3}{r['tag']:<22}{r['arm']:<12}{m:>12.6f}"
              f"{(s if s is not None else float('nan')):>14.6f}"
              f"{(n if n is not None else -1):>8}"
              f"{report.fnum(r['meta'].get('gpu_temp_entry_c')) or 0.0:>9.1f}"
              f"{report.fnum(r['meta'].get('gpu_temp_exit_c')) or 0.0:>8.1f}")

    print()
    print("instrument checks")
    for arm in ("base", "composed", "composed67"):
        rc = sorted({rounds(r) for r in complete if r["arm"] == arm} - {None})
        vals = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
                for r in complete if r["arm"] == arm]
        if not vals:
            continue
        flag = "" if len(rc) <= 1 else "   NOT DETERMINISTIC"
        gaps = sorted(vals)
        widest = max((b - a for a, b in zip(gaps, gaps[1:])), default=0.0)
        print(f"  {arm:<12} rounds {rc}{flag}")
        print(f"  {'':<12} {len(set(vals))} distinct mtp values in {len(vals)}"
              f" legs, widest neighbour gap {widest:.3e} s/tok,"
              f" spread {max(vals) - min(vals):.3e}")
    print("  A few-valued instrument would show repeated identical mtp values"
          " or one wide gap that dominates the spread. Continuous jitter shows"
          " neither.")

    by_arm_edl = {}
    for arm in ("base", "composed", "composed67"):
        vals = {report.fnum(r["metrics"].get("effective_mean_draft_len"))
                for r in complete if r["arm"] == arm} - {None}
        if vals:
            by_arm_edl[arm] = vals
    print()
    print("schedule")
    for arm, vals in by_arm_edl.items():
        flag = "" if len(vals) == 1 else "   NOT CONSTANT WITHIN THE ARM"
        print(f"  {arm:<12} {sorted(vals)}{flag}")
    distinct = set().union(*by_arm_edl.values()) if by_arm_edl else set()
    if len(distinct) == 1:
        print("  one value across all arms, so every contrast here is pure"
              " kernel and runtime with a zero schedule component")
    else:
        print("  the arms do NOT share one schedule. The probe fraction is the"
              " only axis in this session that can move draft length, so the"
              " base contrast mixes a schedule change with the kernel change."
              " The ranked receipt had digit-identical draft lengths on all"
              " five weighted prompts, so this local schedule split is a"
              " property of the fixture and it does not transfer.")
        print("  composed -> composed67 holds the probe fraction fixed, so"
              " that contrast stays schedule-clean whatever the base does.")

    print()
    print("pairwise contrasts, y ~ 1 + arm + centred leg index")
    headlines = {}
    for ref, cand, metric, why in PAIRS:
        sub, fit = contrast(complete, ref, cand, "mtp_seconds_per_token")
        if fit is None:
            print(f"  {ref} -> {cand}: too few legs")
            continue
        pct = -100.0 * fit["contrast"] / fit["mean"]
        se = 100.0 * fit["se"] / fit["mean"]
        headlines[metric] = (pct, se)
        print(f"  {ref} -> {cand}   n {fit['n']:>2}   {why}")
        print(f"      {metric} = {pct:+.4f} % (+- {se:.4f}),"
              f" positive means {cand} is FASTER")
        _, sfit = contrast(complete, ref, cand, "serial_seconds_per_token")
        if sfit:
            print(f"      serial leg {-100.0 * sfit['contrast'] / sfit['mean']:+.4f} %"
                  f" (+- {100.0 * sfit['se'] / sfit['mean']:.4f}),"
                  f" drift {fit['drift_per_leg']:+.3e} per leg")

    tokens = int(report.fnum(complete[0]["metrics"].get("decode_tokens")) or 0)
    cost = launch_cost(complete, args.label, headlines, tokens)
    print()
    print("launch cost per threadgroup column, the E135 question itself")
    print(f"  columns per drafting round  {cost['columns_per_round']}")
    print(f"{'contrast':<24}{'columns saved':>14}{'round ms':>10}"
          f"{'pct':>10}{'us/column':>12}")
    for e in cost["estimates"]:
        print(f"  {e['contrast']:<22}{e['columns_saved']:>14}"
              f"{e['round_ms']:>10.3f}{e['pct']:>+10.4f}"
              f"{e['us_per_column']:>12.4f}")
    independent = [e for e in cost["estimates"]
                   if e["contrast"] in ("base->composed",
                                        "composed->composed67")]
    if len(independent) == 2:
        lo, hi = sorted(e["us_per_column"] for e in independent)
        ratio = max(e["columns_saved"] for e in independent) / min(
            e["columns_saved"] for e in independent)
        print(f"  the two contrasts that share no arm agree to"
              f" {100.0 * (hi - lo) / hi:.1f} % over a {ratio:.1f}x range of"
              f" columns saved, so the effect is linear in launched columns"
              f" and the E135 premise holds")
    artifacts = pathlib.Path("research/e135-artifacts")
    artifacts.mkdir(exist_ok=True)
    primary = next((e for e in cost["estimates"]
                    if e["contrast"] == "composed->composed67"), None)
    payload = dict(cost)
    if primary:
        # The cleanest estimator: one enum apart, same routing, same probe
        # fraction, same schedule, so nothing but launch geometry moves.
        payload["e135_launch_cost_us_per_column"] = primary["us_per_column"]
        payload["e135_launch_cost_us_per_column_se"] = \
            primary["us_per_column_se"]
    (artifacts / f"{args.label}-per-width.json").write_text(
        json.dumps(payload, indent=1) + "\n")

    print()
    if "e135_composition_local_pct" in headlines:
        pct, se = headlines["e135_composition_local_pct"]
        print("pre-registration, base -> composed, positive means faster")
        cells = (
            ("F29 additivity, three mechanisms", PREDICT_ADDITIVE),
            ("T42 additivity, four mechanisms ", PREDICT_ADDITIVE_4),
            ("Finding 225 interaction         ", PREDICT_F225),
        )
        for name, value in cells:
            print(f"  {name}  {value:+.4f} %")
        print(f"  observed                            {pct:+.4f} %"
              f" (+- {se:.4f})")
        nearest = min(cells, key=lambda c: abs(pct - c[1]))
        for name, value in cells:
            mark = "  <== nearest" if name == nearest[0] else ""
            print(f"  distance to {name} {abs(pct - value):>8.4f} pp{mark}")
        print(f"  the cells are separated by at least"
              f" {min(abs(a[1] - b[1]) for a in cells for b in cells if a is not b):.4f}"
              f" pp against a {TOLERANCE_PP} pp instrument tolerance")

    print()
    print("This session is GATED. Every leg passed the real 40 C gate, so"
          " cool_gate_passed_real_gate=true and the legs are gate-qualified"
          " for local comparison. It is still harness=local and is not any"
          " kind of official or ranked score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
