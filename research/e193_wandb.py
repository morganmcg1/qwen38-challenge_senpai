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
import json
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


def publish_leg(leg: dict, base_sha: str, tokens: int) -> tuple[str, str]:
    """One W&B run per timed leg, so every leg has its own id and URL."""
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"e193-leg-{leg['position']}-{leg['arm']}",
        job_type="e193-stage1-leg",
        tags=["e193", "e165-retest", "harness=local", "ungated",
              f"arm={leg['arm']}"],
        config={
            "harness": "local",
            "experiment": "e193-e165-round-start-prefetch-retest",
            "arm": leg["arm"],
            "palindrome_position": leg["position"],
            "tag": leg["tag"],
            "base_sha": base_sha,
            "session_commit": leg["commit"],
            "worker_sha256": leg["worker_sha256"].strip(),
            "decode_tokens": tokens,
            "local_mode": "--local-iterate",
            "arm_env": "MLX_E165_HEAD_PREFETCH",
        },
    )
    run.summary.update({
        "mtp_seconds_per_token": leg["mtp"],
        "serial_seconds_per_token": leg["serial"],
        "effective_mean_draft_len": leg["edl"],
        "accepted_draft_rate": leg["acc"],
        "drafting_rounds": leg["rounds"],
        "drafts_proposed": leg["proposed"],
        "drafts_accepted": leg["accepted"],
        "gpu_temp_entry_c": leg["entry_c"],
        "gpu_temp_exit_c": leg["exit_c"],
        "cool_gate_passed_real_gate": leg["real_gate"],
        "gate_qualified_for_timing": leg["qualified"],
        "all_tokens_matched": leg["matched"],
        "residual_divergence_count": leg["divergences"],
        "post_run_worker_sha256": leg["post_run_worker_sha256"].strip(),
    })
    run_id, url = run.id, run.url
    run.finish()
    return run_id, url


# Tree digests of the two head caches a --local-submit run can reach, resolved
# against the rule in Sources/MLXFastTrustedHarness/QwenMTPHeadDeclaration.swift.
HEAD_DIGESTS = {
    "declared":
        "dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9",
    "pinned":
        "3a7fed842fd881b5a697273345e21ff7b8aa664f1eb177c9c58d8ba1137745b9",
}


def publish_confirmation(
    score_path: pathlib.Path, base_sha: str, candidate_sha: str,
    worker_sha256: str, entry_temps_c: list[float], head_class: str,
) -> tuple[str, str]:
    """One 512-token gated --local-submit confirmation.

    Unlike the Stage 1 screen legs this run IS gate-qualified: both timed legs
    passed the real 40C cool gate. It carries no arm contrast, because there is
    only one arm once the switch is deleted.

    `head_class` records WHICH proposal head the run measured. benchmark-qwen-mtp.sh
    takes the head directory from MLXFAST_QWEN_MTP_HEAD_DIR and does not read the
    tracked manifest, so a clean-environment launch reaches the pinned cache while
    the research leg runner reaches the declared one. Only a `declared` run
    confirms a candidate that ships a head declaration.
    """
    payload = json.loads(score_path.read_text())
    metrics = payload["metrics"]

    expected = HEAD_DIGESTS[head_class]
    actual = metrics["head_provenance_sha256"]
    if actual != expected:
        raise SystemExit(
            f"head_provenance_sha256 {actual} does not match --head-class "
            f"{head_class} ({expected}); refusing to publish a mislabelled run"
        )

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"e193-stage2-confirmation-512-{head_class}-head",
        job_type="e193-stage2-confirmation",
        tags=["e193", "e165-retest", "harness=local", "gate-qualified",
              "local-submit", "tokens=512", f"head={head_class}"],
        config={
            "harness": "local",
            "experiment": "e193-e165-round-start-prefetch-retest",
            "stage": "stage2-confirmation",
            "base_sha": base_sha,
            "candidate_sha": candidate_sha,
            "worker_sha256": worker_sha256,
            "local_mode": "--local-submit",
            "decode_tokens": metrics["decode_tokens"],
            "mtp_depth": metrics["mtp_depth"],
            "cool_gate_passed_real_gate": True,
            "gate_qualified_for_timing": True,
            "gpu_temp_entry_c_serial": entry_temps_c[0],
            "gpu_temp_entry_c_mtp": entry_temps_c[1],
            "arm": "on-unconditional",
            "head_class": head_class,
            "confirms_submitted_configuration": head_class == "declared",
        },
    )
    run.summary.update({
        "local_submit_passed": payload["passed"],
        "local_decode_speedup": payload["score"],
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "accepted_draft_rate": metrics["accepted_draft_rate"],
        "all_tokens_matched": metrics["all_tokens_matched"],
        "residual_divergence_count": metrics["residual_divergence_count"],
        "public_drift_tripwire_passed": metrics["public_drift_tripwire_passed"],
        "uses_pinned_mtp_head": metrics["uses_pinned_mtp_head"],
        "head_provenance_sha256": metrics["head_provenance_sha256"],
        # The local ratio is NOT a ranked estimate: both legs run the candidate
        # build and the reference rows are candidate-generated.
        "rankable": metrics["rankable"],
        "not_rankable_reason": metrics["not_rankable_reason"],
    })
    run_id, url = run.id, run.url
    run.finish()
    return run_id, url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", default="e193")
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--per-leg", action="store_true",
                        help="also publish one run per timed leg")
    parser.add_argument("--confirmation-json", type=pathlib.Path,
                        help="publish ONLY the 512-token gated confirmation")
    parser.add_argument("--head-class", choices=sorted(HEAD_DIGESTS),
                        help="which proposal head the confirmation measured")
    parser.add_argument("--candidate-sha")
    parser.add_argument("--worker-sha256")
    parser.add_argument("--entry-temps-c", nargs=2, type=float,
                        metavar=("SERIAL", "MTP"))
    args = parser.parse_args()

    if args.confirmation_json:
        if not args.head_class:
            parser.error("--confirmation-json requires --head-class")
        run_id, url = publish_confirmation(
            args.confirmation_json, args.base_sha, args.candidate_sha,
            args.worker_sha256, args.entry_temps_c, args.head_class)
        print(f"confirmation {run_id}  {url}")
        return 0

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

    leg_runs: list[tuple[str, str, str]] = []
    if args.per_leg:
        for leg in legs:
            run_id, url = publish_leg(leg, args.base_sha, tokens)
            leg_runs.append((leg["tag"], run_id, url))
            print(f"leg {leg['tag']:24s} {run_id}  {url}")

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

    if leg_runs:
        index = wandb.Table(columns=["tag", "run_id", "url"])
        for tag, run_id, url in leg_runs:
            index.add_data(tag, run_id, url)
        run.log({"abba/leg_runs": index})

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
