#!/usr/bin/env python3
"""E82: stream the precision-island verdict to W&B.

Two halves of one decision, in one run:

  1. the untimed acceptance screen over the island arms, which says the
     islands buy no measurable draft acceptance;
  2. the six-leg ungated palindromic timed session, which says removing them
     nevertheless makes the candidate slower.

The session is ABBA-counterbalanced and ungated by design, so every leg
carries cool_gate_passed_real_gate=false and gate_qualified_for_timing=false
verbatim. It is directional causal evidence inside one session, never a score.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import wandb

from e82_wandb import ENTITY, EXPERIMENT, PROJECT, log_builds, log_screen, table

# Ranked-M5 evidence for the one bit-exact route through the same structure,
# supplied by the advisor from askeladd's submissions. Used only to check this
# session's cost model against a ranked measurement.
ASKELADD_MECHANISM_A = {
    "bytes_removed": 5_898_240,
    "pooled_mean7_pct": -0.172,
    "pooled_se_pct": 0.052,
    "submissions": ["c37b4f67", "9383f9a4", "11a9412a"],
}
DECLARED_TENSOR_BYTES = 427_742_600


def log_timed(run, path: Path, reference: str = "declared") -> dict:
    doc = json.loads(path.read_text())
    legs, summary = doc["legs"], doc["summary"]

    leg_cols = ["tag", "arm", "rep", "started", "candidate_mtp_seconds_per_token",
                "serial_seconds_per_token", "local_ratio", "rounds",
                "rows_per_token", "mean_d", "mean_acc", "draft_build_us_per_round",
                "verify_build_us_per_round", "round_us_total", "d_submit1_us",
                "d_submit2_us", "all_tokens_matched", "residual_divergence_count",
                "accepted_draft_rate", "head_provenance_sha256",
                "head_loaded_bytes", "head_loaded_files", "gpu_temp_entry_c",
                "gpu_temp_exit_c", "cool_gate_passed_real_gate",
                "gate_qualified_for_timing", "base_sha", "worker_sha256"]
    run.log({"timed/legs": table(leg_cols, legs)})

    base = summary[reference]
    arm_rows = []
    for arm, s in summary.items():
        arm_rows.append({
            "arm": arm, "is_reference": arm == reference, "n_legs": s["n"],
            "spt_mean": s["spt_mean"], "spt_legs": json.dumps(s["spt_legs"]),
            "spt_delta_pct_vs_declared": s["spt_delta_pct_vs_declared"],
            "spt_within_arm_spread_pct": s["spt_spread_pct"],
            "local_ratio_mean": s["ratio_mean"],
            "rounds_legs": json.dumps(s["rounds_legs"]),
            "rows_per_token_legs": json.dumps(s["rows_per_token_legs"]),
            "draft_build_us_per_round": s["draft_build_us_per_round"],
            "head_loaded_bytes": s["head_loaded_bytes"],
            "head_bytes_delta_pct":
                (s["head_loaded_bytes"] - base["head_loaded_bytes"])
                / base["head_loaded_bytes"] * 100.0,
            "head_provenance_sha256": json.dumps(s["head_provenance_sha256"]),
            "all_tokens_matched": s["all_tokens_matched"],
            # A head property is stable across the palindrome's repeat visits.
            "rounds_reproducible": len(set(s["rounds_legs"])) == 1,
        })
        run.log({
            f"timed/{arm}/candidate_seconds_per_token": s["spt_mean"],
            f"timed/{arm}/spt_delta_pct_vs_declared": s["spt_delta_pct_vs_declared"],
            f"timed/{arm}/local_ratio": s["ratio_mean"],
            f"timed/{arm}/rounds": s["rounds_legs"][0],
            f"timed/{arm}/rows_per_token": s["rows_per_token_legs"][0],
            f"timed/{arm}/draft_build_us_per_round": s["draft_build_us_per_round"],
            f"timed/{arm}/head_loaded_bytes": s["head_loaded_bytes"],
        })
    run.log({"timed/by_arm": table(sorted({k for r in arm_rows for k in r}), arm_rows)})
    return doc


def log_decomposition(run, doc: dict, reference: str = "declared") -> None:
    arms: dict[str, list[dict]] = {}
    for r in doc["legs"]:
        arms.setdefault(r["arm"], []).append(r)

    def m(a, k):
        return st.mean(r[k] for r in arms[a])

    rows = []
    for a in arms:
        ro, db, vb = m(a, "rounds"), m(a, "draft_build_us_per_round"), m(a, "verify_build_us_per_round")
        tot = m(a, "round_us_total")
        rows.append({
            "arm": a, "rounds": ro, "rows_per_token": m(a, "rows_per_token"),
            "draft_build_total_us": db * ro, "verify_build_total_us": vb * ro,
            "eval_wait_and_rest_us": tot - db * ro - vb * ro, "round_us_total": tot,
            "draft_build_share_pct": db * ro / tot * 100.0,
            "verify_build_share_pct": vb * ro / tot * 100.0,
        })
    run.log({"timed/cost_split": table(sorted({k for r in rows for k in r}), rows)})

    ref, cand = reference, "noislands"
    ddb, ndb = m(ref, "draft_build_us_per_round"), m(cand, "draft_build_us_per_round")
    dtot = m(ref, "round_us_total")
    head_delta_pct = ((arms[cand][0]["head_loaded_bytes"] - arms[ref][0]["head_loaded_bytes"])
                      / arms[ref][0]["head_loaded_bytes"] * 100.0)
    bytes_to_draft_build = ((ndb - ddb) / ddb * 100.0) / head_delta_pct
    draft_build_share = ddb * m(ref, "rounds") / dtot
    bytes_to_candidate = bytes_to_draft_build * draft_build_share

    # The one bit-exact route through the same dead K/V structure has a ranked
    # measurement. Predicting it from this session's conversion is the only
    # available cross-host check on the cost model.
    predicted = (ASKELADD_MECHANISM_A["bytes_removed"] / DECLARED_TENSOR_BYTES
                 * 100.0) * bytes_to_candidate
    measured = ASKELADD_MECHANISM_A["pooled_mean7_pct"]
    run.log({
        "model/bytes_to_draft_build_conversion": bytes_to_draft_build,
        "model/draft_build_share_of_round": draft_build_share,
        "model/bytes_to_candidate_conversion": bytes_to_candidate,
        "model/e79_head_to_median_factor": 0.0843,
        "model/mechanism_a_predicted_pct": -predicted,
        "model/mechanism_a_measured_m5_pct": measured,
        "model/mechanism_a_residual_sigma":
            (measured + predicted) / ASKELADD_MECHANISM_A["pooled_se_pct"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e82-precision-island-timed")
    ap.add_argument("--timed", default="research/e82-headcost-islands.json")
    ap.add_argument("--screen", default="research/e82-accept-islands.json")
    ap.add_argument("--builds", nargs="*",
                    default=["research/e82-build-raw-noislands.json",
                             "research/e82-build-raw-qonly.json",
                             "research/e82-build-master-best.json",
                             "research/e82-build-master-ls.json"])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    doc = json.loads(Path(args.timed).read_text())
    leg0 = doc["legs"][0]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name,
        job_type="timed-session", notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "harness": "local",
            "local_mode": "--local-iterate",
            "timed": True,
            "sync_head": True,
            "decode_tokens": leg0["decode_tokens"],
            "offered_depth": 8,
            "arms": sorted(doc["summary"]),
            "reference_arm": "declared",
            "leg_order": "palindrome (ABBA-counterbalanced)",
            "n_legs": len(doc["legs"]),
            "cool_gate": 0,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "host": leg0.get("host", "ip-10-231-2-22.ec2.internal"),
            "chip": "Apple M4 Pro",
            "memory_bytes": 51539607552,
            "ranked_host": False,
            "base_sha": leg0["base_sha"],
            "worker_sha256": leg0["worker_sha256"],
            "declared_head_bytes": DECLARED_TENSOR_BYTES,
            "askeladd_mechanism_a": ASKELADD_MECHANISM_A,
        },
    )

    log_builds(run, [Path(p) for p in args.builds if Path(p).exists()])
    screen_path = Path(args.screen)
    if screen_path.exists():
        log_screen(run, screen_path, prefix="screen_islands")
    timed = log_timed(run, Path(args.timed))
    log_decomposition(run, timed)

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
