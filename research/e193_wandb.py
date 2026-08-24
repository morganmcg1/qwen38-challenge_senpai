#!/usr/bin/env python3
"""E193 Stage 1: publish the ungated ABBA contrast for the E165 round-start
head-chain prefetch, re-tested alone on the current tree.

Everything here is `harness=local`. No number in this run is an official or
ranked score and none may be compared with a receipt.

WHY THIS IS NOT research/e165_wandb_abba.py
-------------------------------------------
That script is retained prior art and must not be reused as-is. It hardcodes
`RANKED_TRANSFER = 0.34` and publishes a modelled composed score built on it.
That coefficient is WITHDRAWN: the decomposition that produced it was struck
(FINDING 404), Entry 372 recorded tau < 0 on the same channel, the receipt pair
that was to measure tau never scored it (tau remains UNMEASURED), and the
standing ruling on PR 173 is explicit -- "price any local win as a mechanism
result; a receipt prices ranked". Republishing 0.34 would put a withdrawn
constant back into the campaign record wearing a fresh run id.

This run therefore publishes the measured local mechanism effect and NOTHING
derived from an unmeasured transfer coefficient. The ranked expectation is
carried only as the assignment's predeclared band, tagged as a prior, so the
receipt can later be scored against a prediction that was written before it.

The gate flags are carried through verbatim per leg. These legs are ungated
under the standing three conditions, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` appear in the leg table exactly as measured;
the session is ABBA-counterbalanced and entry/exit temperatures are published
alongside the effect.

Leg statistics come from `e165_prefetch.load_legs`, the same loader the
terminal report uses, so the published table cannot drift from the verdict.

Usage:
  python3 research/e193_wandb.py --run-name e193-stage1-prefetch-screen \
      --label e193 --base-sha <BASE_SHA>
"""

from __future__ import annotations

import argparse
import pathlib
import statistics as st
import subprocess

import wandb

from e165_prefetch import load_legs

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# Assignment predeclaration (E193 brief), recorded as a PRIOR only. E185 priced
# the round-start seam at 1060.8 us/round = 0.73% of coherent host work that
# converts at ~100%; E165's historical recovery rate of 58% puts the expected
# mechanism gain at 0.38-0.73%.
PRIOR_RANKED_BAND_PCT = (0.38, 0.73)
# E188 Stage 1 crown pricing, validated against crownGapDecomposition.
PRIOR_P_BEAT_CROWN = {0.38: 0.347, 0.50: 0.413, 0.73: 0.546, 0.0: 0.172}
# E165's own historical local result on its own tree, for the retest contrast.
E165_HISTORICAL_LOCAL_PCT = -0.2872
E165_HISTORICAL_US_PER_ROUND = -539.0
# Campaign minimum useful effect, single mechanism.
MUE_PCT = 0.39


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
    parser.add_argument("--label", default="e193")
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()

    legs = load_legs(args.label)
    mtp = {a: [x["mtp"] for x in legs if x["arm"] == a] for a in ("off", "on")}
    serial = {a: [x["serial"] for x in legs if x["arm"] == a] for a in ("off", "on")}

    effect, two_se = contrast(mtp["off"], mtp["on"])
    # The serial leg runs the SAME candidate binary in both arms and never
    # touches the prefetch, so its contrast is a null channel: a non-zero
    # serial effect means session drift leaked past the palindrome.
    serial_effect, serial_two_se = contrast(serial["off"], serial["on"])

    rounds = legs[0]["rounds"]
    tokens = legs[0]["tokens"]
    mu_off = st.fmean(mtp["off"])
    delta_us_round = (st.fmean(mtp["on"]) - mu_off) * tokens / rounds * 1e6

    ledger = sorted({(x["rounds"], x["proposed"], x["accepted"]) for x in legs})
    exact = all(x["matched"] and not x["divergences"] for x in legs)
    entry_temps = [x["entry_c"] for x in legs]

    clears_zero = effect + two_se < 0
    verdict = (
        "CONTINUE: the local win survives on the current tree"
        if clears_zero
        else "STOP: the 2 sigma interval contains zero"
    )

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.run_name,
        job_type="e193-stage1-abba",
        tags=["e193", "e165-retest", "harness=local", "ungated", "prefetch"],
        config={
            "harness": "local",
            "experiment": "e193-e165-round-start-prefetch-retest",
            "stage": "1-screen-abba",
            "mechanism": "round-start head-chain prefetch, re-tested alone",
            "arm_env": "MLX_E165_HEAD_PREFETCH",
            "base_sha": args.base_sha,
            "session_commit": legs[0]["commit"],
            "worker_sha256": legs[0]["worker_sha256"].strip(),
            "decode_tokens": tokens,
            "rounds_per_leg": rounds,
            "schedule": "natural, no pinned depth",
            "trace": "off on every timed leg",
            "design": "ungated ABBA palindrome off on on off off on on off",
            "legs": len(legs),
            "host": "Apple M4 Pro 48 GB",
            "local_mode": "--local-iterate",
            "receipt_d_snapshot": "376a43f3",
            "port_delta": "gate polarity only; machinery byte-identical to receipt D",
            "ranked_transfer_tau": "UNMEASURED - deliberately not applied",
            "prior_ranked_band_pct": list(PRIOR_RANKED_BAND_PCT),
            "e165_historical_local_pct": E165_HISTORICAL_LOCAL_PCT,
            "mue_pct": MUE_PCT,
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

    prior_table = wandb.Table(
        columns=["assumed_real_gain_pct", "p_beat_crown"])
    for gain, prob in sorted(PRIOR_P_BEAT_CROWN.items()):
        prior_table.add_data(gain, prob)
    run.log({"prior/p_beat_crown": prior_table})

    run.summary.update({
        "effect_pct": effect,
        "two_sigma_pct": two_se,
        "us_per_round_local": delta_us_round,
        "serial_null_pct": serial_effect,
        "serial_null_two_sigma_pct": serial_two_se,
        "mtp_mean_off": mu_off,
        "mtp_mean_on": st.fmean(mtp["on"]),
        "accept_ledger_tuples": len(ledger),
        "accept_ledger": str(ledger[0]),
        "ledger_identical_across_arms": len(ledger) == 1,
        "all_legs_exact": exact,
        "all_legs_real_gate": False,
        "gate_qualified_for_timing": False,
        "entry_temp_spread_c": max(entry_temps) - min(entry_temps),
        "entry_temp_min_c": min(entry_temps),
        "entry_temp_max_c": max(entry_temps),
        "clears_zero_at_two_sigma": clears_zero,
        "reproduces_e165_historical": (
            effect < 0 and abs(effect - E165_HISTORICAL_LOCAL_PCT) < two_se),
        "e165_historical_local_pct": E165_HISTORICAL_LOCAL_PCT,
        "e165_historical_us_per_round": E165_HISTORICAL_US_PER_ROUND,
        "verdict": verdict,
    })

    print(f"run_id {run.id}")
    print(f"run_url {run.url}")
    print(f"effect {effect:+.4f} % (2 sigma {two_se:.4f} %)")
    print(f"per round local {delta_us_round:+.1f} us over {rounds} rounds")
    print(f"serial null {serial_effect:+.4f} % (2 sigma {serial_two_se:.4f} %)")
    print(f"verdict {verdict}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
