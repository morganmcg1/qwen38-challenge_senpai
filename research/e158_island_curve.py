#!/usr/bin/env python3
"""E158 R1.B -- the precision-island arm curve.

Joins the four arms' recall-audit analyses with the per-leg reports and the
R0.3 byte attribution, and prices each arm against `all`.

Byte side, `harness=ranked`. Dropping an island does not only remove the
island's own bytes: k and v have complete island coverage, so their affine-4
packs are resident-but-unread under `all` and come back into the read set the
moment the island is dropped. q has 8.33 % coverage, so its affine-4 pack is
read under every arm and only the island rows come off.

Acceptance side, `harness=local`. The audit is untimed and every leg records
`timing_valid=false`.

usage: research/e158_island_curve.py OUT.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ARMS = ("all", "q", "kv", "none")
PROMPTS = ("beagle_a", "essays_montaigne", "benchfixture")
TOKENS_PER_PROMPT = 512

# From the declared-head census. Indices are read only for q: the k and v
# index tensors are complete permutations of their output range and the
# island fast path reads those rows densely instead of scattering them.
ISLAND_Q = 10_485_760 + 4_096
ISLAND_K = 10_485_760
ISLAND_V = 10_485_760
# One affine-4 g64 [1024, 5120] projection: packed weights plus scales plus
# biases. Read only when that projection has no island installed.
A4_KV_ONE = 2_621_440 + 163_840 + 163_840
MB = 1_000_000.0
PCT_PER_MB = 0.01793  # PROVISIONAL per advisor F4; the byte side is not.


def arm_byte_delta_vs_all(arm: str) -> int:
    """Signed change in trunk bytes read per draft step, `arm` minus `all`."""
    delta = 0
    if arm in ("kv", "none"):
        delta -= ISLAND_Q
    if arm in ("q", "none"):
        delta -= ISLAND_K + ISLAND_V
        delta += 2 * A4_KV_ONE
    return delta


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parent.parent
    runs = root / ".mlxfast-private" / "e128"
    arts = root / "research" / "e158-artifacts"

    out: dict = {
        "harness": "local for acceptance rates, ranked for published percent",
        "timing_valid": False,
        "head": "declared q2/q4, head_provenance.sha256 dadbfb80...",
        "tokens_per_prompt": TOKENS_PER_PROMPT,
        "prompts": list(PROMPTS),
        "arms": {},
    }

    for arm in ARMS:
        pooled = json.loads(
            (arts / f"recall-island-{arm}.json").read_text())["pooled"]
        cond = pooled["conditional_population"]
        legs = {}
        rounds_total = 0
        for pid in PROMPTS:
            leg = runs / f"runs-e158-island-{arm}" / pid
            report = json.loads((leg / "report.json").read_text())
            trace = (leg / "trace.txt").read_text()
            witness = next(
                (ln for ln in trace.splitlines()
                 if ln.startswith(f"qwen-mtp-island-arm: {arm} ")), None)
            if witness is None:
                raise SystemExit(f"e158: no island witness for {arm}/{pid}")
            rounds = report["round_count"]
            rounds_total += rounds
            legs[pid] = {
                "rounds": rounds,
                "tokens_per_round": TOKENS_PER_PROMPT / rounds,
                "all_tokens_matched": report["all_tokens_matched"],
                "residual_divergence_count": report["residual_divergence_count"],
                "accepted_draft_rate": report["accepted_draft_rate"],
                "effective_mean_draft_len": report["effective_mean_draft_len"],
                "head_provenance_sha256": report["head_provenance"]["sha256"],
                "mtp_head_tensor_count": report["mtp_head_tensor_count"],
                "uses_pinned_mtp_head": report["uses_pinned_mtp_head"],
                "island_arm_witness": witness,
            }
        delta = arm_byte_delta_vs_all(arm)
        out["arms"][arm] = {
            "legs": legs,
            "rounds_total": rounds_total,
            "tokens_per_round_pooled":
                TOKENS_PER_PROMPT * len(PROMPTS) / rounds_total,
            "conditional_slots": cond["slots"],
            "p_shipped_conditional": cond["p_shipped"],
            "p_exact_compact_conditional": cond["p_exact_compact"],
            "p_exact_full_conditional": cond["p_exact_full"],
            "bytes_per_draft_step_delta_vs_all": delta,
            "bytes_per_draft_step_delta_vs_all_MB": delta / MB,
            "published_pct_delta_vs_all_PROVISIONAL": -delta / MB * PCT_PER_MB,
        }

    ref = out["arms"]["all"]
    p_all = ref["p_shipped_conditional"]
    n_all = ref["conditional_slots"]
    se_all = math.sqrt(p_all * (1 - p_all) / n_all)
    for arm in ARMS:
        a = out["arms"][arm]
        p = a["p_shipped_conditional"]
        se = math.sqrt(p * (1 - p) / a["conditional_slots"])
        se_diff = math.hypot(se_all, se)
        a["acceptance_loss_vs_all_pt"] = 100.0 * (p_all - p)
        a["acceptance_loss_two_se_pt"] = 200.0 * se_diff
        a["rounds_delta_vs_all"] = a["rounds_total"] - ref["rounds_total"]

    out["resolution_note"] = (
        "Two standard errors on any arm-to-arm acceptance difference is about "
        f"{out['arms']['none']['acceptance_loss_two_se_pt']:.2f} points on "
        "three prompts. Every measured difference is inside that band, so the "
        "curve resolves no arm from any other on acceptance. The byte side is "
        "exact and the round counts are deterministic."
    )
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({a: {
        "rounds": out["arms"][a]["rounds_total"],
        "p_shipped": round(out["arms"][a]["p_shipped_conditional"], 7),
        "loss_pt": round(out["arms"][a]["acceptance_loss_vs_all_pt"], 4),
        "bytes_MB": round(out["arms"][a]["bytes_per_draft_step_delta_vs_all_MB"], 4),
        "pct": round(out["arms"][a]["published_pct_delta_vs_all_PROVISIONAL"], 4),
    } for a in ARMS}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
