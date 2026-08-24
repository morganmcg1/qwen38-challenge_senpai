#!/usr/bin/env python3
"""E165 Stage 2: publish the gated ABBA contrast for the head-chain prefetch.

Everything here is `harness=local`. No number in this run is an official or
ranked score and none may be compared with a receipt. Unlike the Stage 0
census these legs ARE gate qualified: every timed leg passed the real 40 C
gate, so `cool_gate_passed_real_gate` and `gate_qualified_for_timing` are
carried through verbatim per leg rather than asserted in prose.

The leg statistics come from `e165_prefetch.load_legs`, the same loader the
terminal report uses, so the published table cannot drift from the reported
verdict.

Usage:
  python3 research/e165_wandb_abba.py --run-name e165-r0-stage2-prefetch \
      --label pf --base-sha <BASE_SHA>
"""

from __future__ import annotations

import argparse
import pathlib
import statistics as st
import subprocess

import wandb

from e165_prefetch import MINIMUM_USEFUL_PCT, load_legs

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# Advisor LAW 377. A local measurement has no ranked value until its channel is
# named. This mechanism removes host work that is fixed per decode round, so it
# prices on the `per_round_fixed` channel and on no other.
CHANNEL = "per_round_fixed"
RANKED_TRANSFER = 0.34
RANKED_DEFICIT_US = 85.0

# Advisor I7, section 5. The published-score model, evaluated at the two
# prompts that set the median.
PROMPTS = {
    "beagle": {"round_ms": 44.983, "decode_share": 0.9035, "weight": 0.4781},
    "essays": {"round_ms": 48.932, "decode_share": 0.8951, "weight": 0.5219},
}
LAST_RECEIPT = 3.70465399
REVERT_FACTOR = 1.002010
CROWN = 3.729110


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def contrast(values_off: list[float], values_on: list[float]) -> tuple[float, float]:
    """Percent effect and its 2 sigma, pooled within arm exactly as reported."""
    mu_off, mu_on = st.fmean(values_off), st.fmean(values_on)
    effect = (mu_on - mu_off) / mu_off * 100.0
    residuals, dof = [], 0
    for values in (values_off, values_on):
        mu = st.fmean(values)
        residuals += [v - mu for v in values]
        dof += len(values) - 1
    var = sum(r * r for r in residuals) / dof
    n = (len(values_off) + len(values_on)) / 2
    two_se = 2 * (2 * var / n) ** 0.5 / mu_off * 100.0
    return effect, two_se


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", default="pf")
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()

    legs = load_legs(args.label)
    reps = sorted({leg["rep"] for leg in legs})
    mtp = {a: [x["mtp"] for x in legs if x["arm"] == a] for a in ("off", "on")}
    serial = {a: [x["serial"] for x in legs if x["arm"] == a] for a in ("off", "on")}

    effect, two_se = contrast(mtp["off"], mtp["on"])
    serial_effect, serial_two_se = contrast(serial["off"], serial["on"])

    rounds = legs[0]["rounds"]
    tokens = legs[0]["tokens"]
    mu_off = st.fmean(mtp["off"])
    delta_us_round = (st.fmean(mtp["on"]) - mu_off) * tokens / rounds * 1e6
    ranked_us_round = delta_us_round * RANKED_TRANSFER

    published_pct = 0.0
    for spec in PROMPTS.values():
        of_round = -ranked_us_round / 1e3 / spec["round_ms"] * 100.0
        published_pct += spec["weight"] * of_round * spec["decode_share"]
    composed = LAST_RECEIPT * REVERT_FACTOR * (1.0 + published_pct / 100.0)

    ledger = sorted({(x["rounds"], x["proposed"], x["accepted"]) for x in legs})
    exact = all(x["matched"] and not x["divergences"] for x in legs)
    gated = all(x["real_gate"] == "true" for x in legs)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.run_name,
        job_type="e165-stage2-abba",
        tags=["e165", "harness=local", "gated", "prefetch", CHANNEL],
        config={
            "harness": "local",
            "experiment": "e165-per-round-fixed-cost",
            "stage": "2-abba-contrast",
            "mechanism": "cross-round head-chain prefetch",
            "arm_env": "MLX_E165_HEAD_PREFETCH",
            "base_sha": args.base_sha,
            "session_commit": legs[0]["commit"],
            "worker_sha256": legs[0]["worker_sha256"].strip(),
            "decode_tokens": tokens,
            "rounds_per_leg": rounds,
            "schedule": "natural, no pinned depth",
            "trace": "off on every timed leg",
            "design": "gated ABBA palindrome off on on off off on on off",
            "replicates": len(reps),
            "legs": len(legs),
            "host": "Apple M4 Pro 48 GB",
            "channel": CHANNEL,
            "ranked_transfer_per_local_ms": RANKED_TRANSFER,
            "minimum_useful_pct": -MINIMUM_USEFUL_PCT,
        },
    )

    leg_columns = [
        "tag", "arm", "rep", "position", "mtp_s_per_tok", "serial_s_per_tok",
        "edl", "accepted_draft_rate", "rounds", "proposed", "accepted",
        "entry_c", "exit_c", "cool_gate_passed_real_gate",
        "gate_qualified_for_timing", "all_tokens_matched",
        "residual_divergence_count", "worker_sha256", "post_run_worker_sha256",
    ]
    leg_table = wandb.Table(columns=leg_columns)
    for leg in legs:
        leg_table.add_data(
            leg["tag"], leg["arm"], leg["rep"], leg["position"], leg["mtp"],
            leg["serial"], leg["edl"], leg["acc"], leg["rounds"],
            leg["proposed"], leg["accepted"], leg["entry_c"], leg["exit_c"],
            leg["real_gate"], leg["qualified"], leg["matched"],
            leg["divergences"], leg["worker_sha256"].strip(),
            leg["post_run_worker_sha256"].strip(),
        )
    run.log({"abba/legs": leg_table})

    rep_table = wandb.Table(
        columns=["rep", "n_off", "n_on", "effect_pct", "two_sigma_pct",
                 "us_per_round"])
    for rep in reps:
        sub = [x for x in legs if x["rep"] == rep]
        off = [x["mtp"] for x in sub if x["arm"] == "off"]
        on = [x["mtp"] for x in sub if x["arm"] == "on"]
        eff, se = contrast(off, on)
        rep_table.add_data(
            str(rep), len(off), len(on), eff, se,
            (st.fmean(on) - st.fmean(off)) * tokens / rounds * 1e6)
    eff_all, se_all = effect, two_se
    rep_table.add_data("pooled", len(mtp["off"]), len(mtp["on"]), eff_all,
                       se_all, delta_us_round)
    run.log({"abba/replicates": rep_table})

    run.summary.update({
        "effect_pct": effect,
        "two_sigma_pct": two_se,
        "us_per_round_local": delta_us_round,
        "us_per_round_ranked": ranked_us_round,
        "ranked_deficit_us_per_round": RANKED_DEFICIT_US,
        "ranked_repayment_multiple": -ranked_us_round / RANKED_DEFICIT_US,
        "serial_null_pct": serial_effect,
        "serial_null_two_sigma_pct": serial_two_se,
        "mtp_mean_off": mu_off,
        "mtp_mean_on": st.fmean(mtp["on"]),
        "accept_ledger_tuples": len(ledger),
        "accept_ledger": str(ledger[0]),
        "ledger_identical_across_arms": len(ledger) == 1,
        "all_legs_exact": exact,
        "all_legs_real_gate": gated,
        "clears_zero_at_two_sigma": effect + two_se < 0,
        "meets_predeclared_minimum": effect <= -MINIMUM_USEFUL_PCT,
        "modelled_published_gain_pct": published_pct,
        "modelled_composed_score": composed,
        "modelled_crown": CROWN,
        "modelled_shortfall_pct": (CROWN - composed) / composed * 100.0,
    })

    print(f"run_id {run.id}")
    print(f"run_url {run.url}")
    print(f"effect {effect:+.4f} % (2 sigma {two_se:.4f} %)")
    print(f"per round local {delta_us_round:+.1f} us "
          f"ranked {ranked_us_round:+.1f} us")
    print(f"modelled published {published_pct:+.4f} % "
          f"composed {composed:.6f} crown {CROWN:.6f}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
