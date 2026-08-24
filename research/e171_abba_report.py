#!/usr/bin/env python3
"""E171: read the gated two-binary ABBA session and apply the advisor stop rule.

Everything here is `harness=local`. No number in this run is an official or
ranked score, and none may be compared with a receipt.

The contrast is S1 (instrumentation stripped) against S0 (instrumentation
present), both carrying the E165 cross-round head-chain prefetch as the shipped
default. The primary quantity is decode-only seconds per token, which removes
the seed prefill from the leg on the same clock origin.

Gates applied before any number is priced:

  * every leg passed the real 40 C gate;
  * every leg reports `all_tokens_matched` true against reference rows that
    were generated once from S0, so the strip is checked across arms;
  * the accept ledger is single valued across all legs. A deletion that moves
    `effective_mean_draft_len`, the accepted total, the round count or the
    accepted-draft rate deleted something that was not instrumentation;
  * each leg's binary witness agrees with its arm label.

Usage:
  python3 research/e171_abba_report.py [--wandb] [--run-name NAME]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics as st
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "research" / "out" / "e171" / "abba"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# Advisor r1 brief, section 6, against a per-leg host timing noise of 0.120 %.
CONFIRM_PCT = 0.30  # S1 faster by this much or more: confirmed.
TOLERATE_PCT = -0.15  # S1 slower by more than this: stop, do not fire.

# Advisor r1 brief, section 5. Ranked projection from receipt B (180db842,
# 3.704653995) with the decode tax removed and E165 applied at 0.2872 %.
CROWN = 3.729110
RANKED_PROJECTION = {"2s_low": 3.738813, "central": 3.752028, "2s_high": 3.765338}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def load_legs() -> list[dict[str, str]]:
    with (OUT / "legs.tsv").open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_session() -> dict[str, str]:
    session: dict[str, str] = {}
    for line in (OUT / "session.txt").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            session[key] = value
    return session


def contrast(baseline: list[float], candidate: list[float]) -> tuple[float, float]:
    """Percent change of candidate against baseline, and its 2 sigma.

    Variance is pooled within arm, so a real arm effect does not inflate the
    interval that has to contain it.
    """
    mu_b, mu_c = st.fmean(baseline), st.fmean(candidate)
    effect = (mu_c - mu_b) / mu_b * 100.0
    residuals: list[float] = []
    dof = 0
    for values in (baseline, candidate):
        mu = st.fmean(values)
        residuals += [v - mu for v in values]
        dof += len(values) - 1
    if dof == 0:
        return effect, float("nan")
    var = sum(r * r for r in residuals) / dof
    n = (len(baseline) + len(candidate)) / 2
    two_se = 2 * (2 * var / n) ** 0.5 / mu_b * 100.0
    return effect, two_se


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="e171-r1-strip-abba")
    args = parser.parse_args()

    legs = load_legs()
    session = load_session()
    if not legs:
        print("e171_abba_report: no legs")
        return 1

    problems: list[str] = []
    for leg in legs:
        if leg["matched"] != "true":
            problems.append(f"leg {leg['leg']} ({leg['arm']}): all_tokens_matched false")
        if int(leg["control"]) < 1:
            problems.append(f"leg {leg['leg']}: symbol table not readable")
        witness = int(leg["witness"])
        if leg["arm"] == "s0" and witness < 1:
            problems.append(f"leg {leg['leg']}: s0 binary carries no instrumentation")
        if leg["arm"] == "s1" and witness != 0:
            problems.append(f"leg {leg['leg']}: s1 binary still carries instrumentation")

    ledger_fields = ("edl", "accepted", "rounds", "accept_rate")
    ledger = {f: sorted({leg[f] for leg in legs}) for f in ledger_fields}
    for field, values in ledger.items():
        if len(values) != 1:
            problems.append(f"ledger field {field} is not single valued: {values}")

    gates = []
    for index, leg in enumerate(legs, start=1):
        meta = (OUT / f"leg{leg['leg']}-{leg['arm']}" / "meta.txt").read_text()
        entry = dict(
            line.split("=", 1) for line in meta.splitlines() if "=" in line
        )
        gates.append(entry)
        if entry.get("cool_gate_passed_real_gate") != "true":
            problems.append(f"leg {leg['leg']}: real 40 C gate not passed")

    by_arm: dict[str, dict[str, list[float]]] = {
        "s0": {"decode": [], "full": []},
        "s1": {"decode": [], "full": []},
    }
    for leg in legs:
        by_arm[leg["arm"]]["decode"].append(float(leg["decode_only_spt"]))
        by_arm[leg["arm"]]["full"].append(float(leg["spt"]))

    decode_effect, decode_2s = contrast(by_arm["s0"]["decode"], by_arm["s1"]["decode"])
    full_effect, full_2s = contrast(by_arm["s0"]["full"], by_arm["s1"]["full"])
    improvement = -decode_effect

    if improvement >= CONFIRM_PCT:
        verdict = "CONFIRMED: freeze and fire"
    elif improvement >= TOLERATE_PCT:
        verdict = "WITHIN TOLERANCE: freeze and fire, and report the disagreement"
    else:
        verdict = "STOP: S1 is slower than the tolerance; do not fire"

    print(f"commit           {session.get('git_head', git('rev-parse', 'HEAD'))}")
    print(f"tokens           {session.get('tokens')}  depth {session.get('depth')}")
    print(f"schedule         {session.get('schedule')}")
    print(f"worker s0        {session.get('worker_s0_sha256', '')[:12]}"
          f"  witness {session.get('worker_s0_witness')}"
          f"  control {session.get('worker_s0_control')}")
    print(f"worker s1        {session.get('worker_s1_sha256', '')[:12]}"
          f"  witness {session.get('worker_s1_witness')}"
          f"  control {session.get('worker_s1_control')}")
    print(f"golden rows      {session.get('golden_rows')} from s0"
          f"  sha12 {session.get('golden_sha256', '')[:12]}")
    print()
    for leg, gate in zip(legs, gates):
        print(
            f"leg {leg['leg']} {leg['arm']}"
            f"  entry {gate.get('gpu_temp_entry_c')}C"
            f"  exit {gate.get('gpu_temp_exit_c')}C"
            f"  matched {leg['matched']}"
            f"  decode_spt {float(leg['decode_only_spt']):.9f}"
            f"  spt {float(leg['spt']):.9f}"
            f"  edl {leg['edl']}  acc {leg['accepted']}  rounds {leg['rounds']}"
        )
    print()
    for field, values in ledger.items():
        print(f"ledger {field:12s} {values[0]}")
    print()
    print(f"s0 decode spt    {st.fmean(by_arm['s0']['decode']):.9f}"
          f"  n={len(by_arm['s0']['decode'])}")
    print(f"s1 decode spt    {st.fmean(by_arm['s1']['decode']):.9f}"
          f"  n={len(by_arm['s1']['decode'])}")
    print(f"decode effect    {decode_effect:+.4f} % +- {decode_2s:.4f} (2 sigma)")
    print(f"full-leg effect  {full_effect:+.4f} % +- {full_2s:.4f} (2 sigma)")
    print(f"improvement      {improvement:+.4f} % (positive means s1 faster)")
    print(f"verdict          {verdict}")
    if problems:
        print()
        for problem in problems:
            print(f"PROBLEM {problem}")
    print()
    print("harness=local. Not a ranked score. The ranked prediction for this "
          "strip comes from receipts A and C, not from this session.")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=args.run_name,
            job_type="abba",
            config={
                "harness": "local",
                "experiment": "e171-r1-instrumentation-strip",
                "commit": session.get("git_head"),
                "base_sha": "589793323fca5d1570870b7a044355944f959fa9",
                "tokens": int(session.get("tokens", 0)),
                "depth": int(session.get("depth", 0)),
                "schedule": session.get("schedule"),
                "host": session.get("host"),
                "mem_bytes": int(session.get("mem_bytes", 0)),
                "worker_s0_sha256": session.get("worker_s0_sha256"),
                "worker_s1_sha256": session.get("worker_s1_sha256"),
                "golden_source_arm": "s0",
                "golden_sha256": session.get("golden_sha256"),
                "cool_gate": "real 40 C gate on every timed leg",
                "confirm_pct": CONFIRM_PCT,
                "tolerate_pct": TOLERATE_PCT,
                "crown": CROWN,
                "ranked_projection": RANKED_PROJECTION,
            },
        )
        table = wandb.Table(
            columns=[
                "leg", "arm", "worker_sha12", "witness", "control",
                "entry_c", "exit_c", "matched", "decode_only_spt", "spt",
                "decode_s", "prefill_s", "edl", "accepted", "rounds",
                "accept_rate", "cool_gate_passed_real_gate",
                "gate_qualified_for_timing",
            ]
        )
        for leg, gate in zip(legs, gates):
            table.add_data(
                int(leg["leg"]), leg["arm"], leg["worker_sha12"],
                int(leg["witness"]), int(leg["control"]),
                gate.get("gpu_temp_entry_c"), gate.get("gpu_temp_exit_c"),
                leg["matched"], float(leg["decode_only_spt"]), float(leg["spt"]),
                float(leg["decode_s"]), float(leg["prefill_s"]), leg["edl"],
                leg["accepted"], leg["rounds"], leg["accept_rate"],
                gate.get("cool_gate_passed_real_gate"),
                gate.get("gate_qualified_for_timing"),
            )
        run.log(
            {
                "legs": table,
                "s0_decode_spt_mean": st.fmean(by_arm["s0"]["decode"]),
                "s1_decode_spt_mean": st.fmean(by_arm["s1"]["decode"]),
                "s0_spt_mean": st.fmean(by_arm["s0"]["full"]),
                "s1_spt_mean": st.fmean(by_arm["s1"]["full"]),
                "decode_effect_pct": decode_effect,
                "decode_effect_2sigma_pct": decode_2s,
                "full_leg_effect_pct": full_effect,
                "full_leg_effect_2sigma_pct": full_2s,
                "improvement_pct": improvement,
                "ledger_single_valued": len(problems) == 0,
                "problem_count": len(problems),
            }
        )
        run.summary["verdict"] = verdict
        run.summary["problems"] = problems
        print(f"wandb {run.url}")
        run.finish()

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
