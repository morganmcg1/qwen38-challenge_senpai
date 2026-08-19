#!/usr/bin/env python3
"""Log the E37 dispatched-verify-width (M) census to W&B.

E37 measures no time -- the trace that produces the counts perturbs the round it
counts -- so there is no metric series to stream. The durable record is the
per-prompt width census, the depth-8 gate cross-tabulation, the exact ranked
M>=6 bracket derived from published telemetry, and the score payoff frame.

  python3 research/e37_wandb_log.py research/results/e37/census.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e37_width_census import RANKED, RANKED_N, ranked_ge6_bound, score_of  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EVIDENCE = pathlib.Path("research/results/e37")


def meta(arm: str) -> dict:
    out = {}
    for line in (EVIDENCE / f"{arm}-meta.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def main() -> None:
    census = json.loads(pathlib.Path(sys.argv[1]).read_text())
    arms = sorted(census)
    m0 = meta(arms[0])

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="e37-draft-width-census-beagle-medicine",
        job_type="census",
        tags=["e37", "counts-only", "no-timing-claim", "qwen38-mtp-v1"],
        config={
            "assignment": "qwen38-r1-e37-draft-width-census-beagle-medicine",
            "revision": "r1", "pr": 42,
            "base_sha": "abf6d79f92b97e3c47856be9c1d7798e6dc5a6b5",
            "head_sha": m0["head_sha"], "worktree_dirty": m0["dirty"],
            "host": "Apple M4 Pro Mac16,11 applegpu_g16s (NOT ranked M5)",
            "decode_tokens": int(m0["tokens"]),
            "offered_depth": int(m0["offered_depth"]),
            "head_tree_digest": "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71",
            "head_safetensors_sha256": m0["head_safetensors_sha256"],
            "worker_sha256": m0["worker_sha256"], "cli_sha256": m0["cli_sha256"],
            "sdpa_width_wall_depth_cap": 5, "segmented_verify_depth_cap": 8,
            "segmented_streak_gate": 2, "head_step_cost_ratio": 0.18,
            # Preserved verbatim: this run is not gate-qualified and no number
            # it produces may be compared as a timing measurement.
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "timing_claims_permitted": False,
            "trace_perturbs_timing": True,
        })

    width = wandb.Table(columns=[
        "arm", "M", "rounds", "round_share", "rows", "row_share", "token_share"])
    gate = wandb.Table(columns=[
        "arm", "drafting_rounds", "caps_seen", "gate_open_rounds",
        "gate_open_share", "max_streak", "chose_depth_gt5_while_open"])
    for arm in arms:
        c = census[arm]
        for k in sorted(c["hist"], key=int):
            width.add_data(arm, int(k), c["hist"][k], c["round_share"][k],
                           int(k) * c["hist"][k], c["row_share"][k],
                           c["token_share"][k])
        g = c["gate"]
        gate.add_data(arm, g["drafting_rounds"], str(g["caps_seen"]),
                      g["gate_open_rounds"], g["gate_open_share"],
                      g["max_streak"], g["deep_rounds_when_open"])

    base = score_of(RANKED)
    bracket = wandb.Table(columns=[
        "prompt", "ranked_raw_p", "ranked_n", "ranked_mean_M",
        "min_round_share_ge6", "max_round_share_ge6",
        "min_row_share_ge6", "max_row_share_ge6",
        "min_row_share_ge6_if_depth5_ceiling",
        "headroom_pct_of_raw_p", "score_if_fully_realised"])
    for nm in ("beagle", "medicine"):
        b8, b5 = ranked_ge6_bound(RANKED_N[nm], 8), ranked_ge6_bound(RANKED_N[nm], 5)
        others = sorted(v for k, v in RANKED.items() if k != nm)
        capped = dict(RANKED, **{nm: others[4]})
        bracket.add_data(
            nm, RANKED[nm], RANKED_N[nm], b8["mean_m"],
            b8["min_round_share_ge6"], b8["max_round_share_ge6"],
            b8["min_row_share_ge6"], b8["max_row_share_ge6"],
            b5["min_row_share_ge6"],
            100.0 * (others[4] / RANKED[nm] - 1.0), score_of(capped))

    run.log({"width_census": width, "depth8_gate": gate, "ranked_ge6_bracket": bracket})

    nh, med = census["natural_history"], census["medicine"]
    beagle = ranked_ge6_bound(RANKED_N["beagle"], 8)
    run.summary.update({
        "local/beagle_proxy_max_M": nh["max_width"],
        "local/beagle_proxy_mean_M": nh["mean_width"],
        "local/beagle_proxy_row_share_ge6": nh["row_share_ge6"],
        "local/medicine_max_M": med["max_width"],
        "local/medicine_mean_M": med["mean_width"],
        "local/medicine_row_share_ge6": med["row_share_ge6"],
        "local/rounds_choosing_depth_gt5": 0,
        "local/gate_open_rounds_total": (nh["gate"]["gate_open_rounds"]
                                         + med["gate"]["gate_open_rounds"]),
        "ranked/beagle_min_row_share_ge6": beagle["min_row_share_ge6"],
        "ranked/medicine_min_row_share_ge6":
            ranked_ge6_bound(RANKED_N["medicine"], 8)["min_row_share_ge6"],
        "ranked/board_top_score": base,
        "ranked/our_best_score": 3.23250848263467,
        "ranked/sigma_score_points": 0.00078 * base,
        "verdict/m_ge_6_reachable_locally": False,
        "verdict/deep_regime_M7_9_reachable_locally": False,
    })

    art = wandb.Artifact("e37-width-census", type="census")
    for p in sorted(EVIDENCE.iterdir()):
        art.add_file(str(p))
    run.log_artifact(art)
    print(f"logged {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
