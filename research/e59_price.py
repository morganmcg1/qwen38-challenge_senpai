#!/usr/bin/env python3
"""E59 pricing: convert the measured M=5 route win into a ranked-score band.

Three independent readings of the same mechanism are reported side by side:

  1. bottom up  -- the isolated M=5 cell win (rung 3) multiplied by this
                   fixture's own cost-weighted M=5 share and by the QMV share
                   of an MTP round;
  2. top down   -- the measured end-to-end round-cost move (rung 4);
  3. ranked     -- reading 2 rescaled by the published band for how much the
                   local fixture under-weights M=5 against the ranked mixture.

  python3 research/e59_price.py --out research/e59-artifacts/e59-pricing.json

Every constant is copied from a named source. Nothing is re-derived here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "research/e59-artifacts"
CELL_METRICS = ARTIFACTS / "e59-cell-metrics.json"
E2E_METRICS = ARTIFACTS / "e59-e2e-metrics.json"
REPRICING = REPO / "research/e53-scored-repricing.json"

BASE_ARM = "shipped"
TREATED_WIDTH = 5

# PR 62, from the advisor's local fixture study. Cost-weighted share of the MTP
# leg's QMV work by verify width on this fixture.
PUBLISHED_LOCAL_SHARES = {
    9: 53.45, 6: 25.19, 8: 7.55, 5: 5.07, 7: 4.71, 4: 3.50, 2: 0.54,
}
# PR 62: the local fixture under-weights M=5 against the ranked mixture by this
# factor band, so a local M=5 move must be rescaled before it is a board move.
RANKED_M5_REWEIGHT = (2.40, 4.41)
# PR 62 corrected pricing, 189(E). Ranked speed-up percent predicted for this
# mechanism under two mixture models.
PREREG_RANKED_BANDS = {
    "e48": (0.5626, 0.6636),
    "e53_mid": (0.8424, 0.9937),
}
# PR 62: how much ranked speed-up closes the gap to the promoted frontier, and
# the smallest ranked move the board can resolve at two standard deviations.
DEFICIT_PCT = 0.5367
RANKED_MDE_2SD_PCT = 0.283


def cost_to_speed_pct(cost_pct: float) -> float:
    """A cost move of -x % is a speed move of +x/(1-x/100) %."""
    return -100.0 * cost_pct / (100.0 + cost_pct)


def local_shares(cell: dict, hist: dict[str, int]) -> dict:
    """Cost-weighted width shares from this session's own two measurements.

    The isolated per-width cost curve comes from the rung 3 base arm and the
    round-width histogram comes from the rung 4 base arm, so the share is
    measured here rather than inherited.
    """
    base_legs = [leg for leg in cell["legs"]
                 if leg["status"] == "ok" and leg["arm"] == "iso_m5_ipg3"]
    if not base_legs:
        return {}
    per_width: dict[int, float] = {}
    for m_str in base_legs[0]["t_ms"]:
        m = int(m_str)
        values = [leg["t_ms"][m_str] for leg in base_legs if m_str in leg["t_ms"]]
        per_width[m] = sum(values) / len(values)

    weighted = {}
    for m_str, count in hist.items():
        m = int(m_str)
        if m in per_width:
            weighted[m] = count * per_width[m]
    total = sum(weighted.values())
    if not total:
        return {}
    return {m: 100.0 * v / total for m, v in sorted(weighted.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-pricing.json")
    args = ap.parse_args()

    if not CELL_METRICS.exists() or not E2E_METRICS.exists():
        raise SystemExit("e59_price: run rung 3 and rung 4 analysis first")
    cell = json.loads(CELL_METRICS.read_text())
    e2e = json.loads(E2E_METRICS.read_text())
    psi = json.loads(REPRICING.read_text())["psi_mtp"]

    route = e2e["verdicts"].get("route_arm") or cell["rung3_gate"].get(
        "route_for_rung4")
    arm_key = {"m5_rb2": "m5_rb2", "m5_rbx": "m5_rbx"}.get(route, route)
    if arm_key not in e2e["arms"]:
        raise SystemExit("e59_price: rung 4 has no %s arm" % arm_key)

    net_pair = "N1" if arm_key == "m5_rb2" else "N2"
    net_cell_win_pct = cell["pairs"][net_pair]["delta_pct"]

    hist = e2e["arms"][BASE_ARM]["width_histogram"]
    measured_shares = local_shares(cell, hist)
    share5 = measured_shares.get(TREATED_WIDTH)

    bottom_up = None
    if share5 is not None:
        bottom_up = psi["point"] * (share5 / 100.0) * net_cell_win_pct

    top_down_leg = e2e["arms"][arm_key]["delta_vs_base_leg_pct"]
    top_down_round = e2e["arms"][arm_key]["delta_vs_base_round_cost_pct"]

    ranked_cost_band = tuple(top_down_round * f for f in RANKED_M5_REWEIGHT)
    ranked_speed_band = tuple(sorted(cost_to_speed_pct(c) for c in ranked_cost_band))

    payload = {
        "route_arm": arm_key,
        "net_cell_win_pct_at_m5": net_cell_win_pct,
        "psi_mtp": psi,
        "measured_local_cost_shares_pct": measured_shares,
        "published_local_cost_shares_pct": PUBLISHED_LOCAL_SHARES,
        "local_m5_share_pct": share5,
        "local_m5_share_published_pct": PUBLISHED_LOCAL_SHARES[TREATED_WIDTH],
        "bottom_up_predicted_local_round_cost_pct": bottom_up,
        "measured_local_leg_pct": top_down_leg,
        "measured_local_round_cost_pct": top_down_round,
        "bottom_up_minus_top_down_pct": (
            None if bottom_up is None else round(bottom_up - top_down_round, 4)),
        "ranked_m5_reweight_band": RANKED_M5_REWEIGHT,
        "ranked_cost_band_pct": ranked_cost_band,
        "ranked_speed_band_pct": ranked_speed_band,
        "prereg_ranked_bands_pct": PREREG_RANKED_BANDS,
        "deficit_to_close_pct": DEFICIT_PCT,
        "ranked_mde_2sd_pct": RANKED_MDE_2SD_PCT,
        "closes_deficit_at_worst": bool(ranked_speed_band[0] >= DEFICIT_PCT),
        "closes_deficit_at_best": bool(ranked_speed_band[1] >= DEFICIT_PCT),
        "clears_ranked_mde_at_worst": bool(
            ranked_speed_band[0] >= RANKED_MDE_2SD_PCT),
        "caveat": (
            "the ranked band is a rescaling of one local one-prompt "
            "measurement, not a ranked measurement. Only the official M5 "
            "runner produces a ranked score."
        ),
    }

    print("E59 pricing for route %s" % arm_key)
    print("  isolated M=5 net cell win        %+.3f %%" % net_cell_win_pct)
    if share5 is not None:
        print("  measured local M=5 cost share    %.2f %%  (published %.2f %%)"
              % (share5, PUBLISHED_LOCAL_SHARES[TREATED_WIDTH]))
        print("  bottom-up local round-cost       %+.4f %%" % bottom_up)
    print("  measured local leg move          %+.4f %%" % top_down_leg)
    print("  measured local round-cost move   %+.4f %%  <- converts to rank"
          % top_down_round)
    print("  ranked cost band (x%.2f..%.2f)    %+.4f .. %+.4f %%"
          % (*RANKED_M5_REWEIGHT, *ranked_cost_band))
    print("  ranked speed band                %+.4f .. %+.4f %%"
          % ranked_speed_band)
    for name, band in PREREG_RANKED_BANDS.items():
        print("  prereg %-8s                  %+.4f .. %+.4f %%"
              % (name, *band))
    print("  deficit %.4f %%, ranked MDE at 2 sd %.4f %%"
          % (DEFICIT_PCT, RANKED_MDE_2SD_PCT))
    print("  closes deficit at worst: %s   clears MDE at worst: %s"
          % (payload["closes_deficit_at_worst"],
             payload["clears_ranked_mde_at_worst"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
