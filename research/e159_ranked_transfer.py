#!/usr/bin/env python3
"""E159 F7: identify the ranked round, then price local-to-ranked transfer.

Step 1 collapses FINDING 319. The board publishes `effective_mean_draft_len`
as an exact rational `proposed / rounds`, so only round counts that make
`q * rounds` an integer are possible. Intersecting that lattice with the
FINDING 319 bounds identifies five prompts outright; the monotone round-cost
law identifies the other three.

Step 2 divides each prompt's candidate decode time by its now-exact round
count, which gives `R_ranked` per prompt as a point, not an interval.

Step 3 compares the ranked round-cost curve with this student's local pinned
depth sweep, in seconds and in target-weight passes.

Step 4 derives the local-to-ranked amplification `k` for a pure byte-removal
arm, with an interval over the bandwidth anchor, and retrodicts the one
receipt pair that measures it.

harness=ranked for every board quantity; harness=local for every sweep
quantity. The two are never mixed inside one number without a stated model.

Usage:
  python3 research/e159_ranked_transfer.py
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e159-artifacts"
CACHE = os.environ.get("RECEIPT_CACHE", "/tmp/yukon_board.json")

ANCHOR = "5a9f130a"
CROWN = "ec24d591"

NAME = {
    "c1ec5866": "plutarch", "4b9e88cd": "drama", "3b10cb4d": "travel",
    "919318e1": "beagle", "00142a44": "medicine", "ea82dcb5": "republic",
    "a2ea8b60": "essays", "192fb621": "botany",
}

# FINDING 319 (ledger 335) round bounds for the anchor receipt.
FINDING_319_BOUNDS = {
    "plutarch": (457.9, 512.0), "drama": (206.0, 303.6),
    "travel": (173.2, 265.6), "beagle": (99.6, 181.6),
    "medicine": (83.3, 164.8), "republic": (85.5, 164.7),
    "essays": (85.8, 166.8), "botany": (76.3, 163.8),
}

DECODE_TOKENS = 512
TARGET_WEIGHT_PASS_BYTES = 14_417_640_448
HEAD_WEIGHT_PASS_BYTES = 427_742_600      # mtp-head.manifest.json
PINNED_SERIAL_ROUND_SECONDS = 0.0379709

# askeladd's B1 cell, quoted in PR 159 F7. The two arms drafted
# bit-identically there, so the measurement isolates cost.
B1_BYTES_PER_DRAFT_STEP = 25_558_528
B1_LOCAL_ROUND_SECONDS = 0.1385799
B1_LOCAL_DELTA_R_SECONDS = -0.00057694
B1_LOCAL_Q = 505 / 119.0                  # 393 accepted + 112 rejected
B1_LOCAL_PUBLISHED_PERCENT = 0.4181

# thorfinn's leaf16 receipt pair, the only measured transfer point.
LEAF16_LOCAL_PERCENT = 0.4207
LEAF16_RANKED_PERCENT = 0.5291
LEAF16_MEASURED_K = LEAF16_RANKED_PERCENT / LEAF16_LOCAL_PERCENT


def load_board():
    with open(CACHE) as handle:
        payload = json.load(handle)
    rows = payload["submissions"] if isinstance(payload, dict) else payload
    return {row["id"][:8]: row for row in rows}


def per_prompt(row):
    out = {}
    for entry in row["officialMetrics"]["per_prompt"]:
        key = entry["prompt_sha256"][:8]
        out[NAME.get(key, key)] = entry
    return out


def round_lattice(q, lo, hi):
    """Round counts that make `q * rounds` an integer, inside the bounds."""
    return [r for r in range(1, DECODE_TOKENS + 1)
            if abs(q * r - round(q * r)) < 1e-9 and lo - 0.5 <= r <= hi + 0.5]


def ols(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    sd = (sse / (n - 2)) ** 0.5 if n > 2 else float("nan")
    return {
        "intercept": intercept, "slope": slope, "n": n,
        "r_squared": 1.0 - sse / sst if sst else float("nan"),
        "residual_sd": sd,
        "se_slope": sd / sxx ** 0.5 if sxx else float("nan"),
        "se_intercept": sd * (1.0 / n + mx * mx / sxx) ** 0.5 if sxx else float("nan"),
        "residuals": resid,
    }


def identify_rounds(prompts):
    """Collapse FINDING 319 to a point for every prompt."""
    stage1 = {}
    for name, entry in prompts.items():
        q = entry["effective_mean_draft_len"]
        lo, hi = FINDING_319_BOUNDS[name]
        stage1[name] = round_lattice(q, lo, hi)

    decode = {
        name: DECODE_TOKENS * (entry["mtp_seconds_per_token_mean"]
                               - entry["prefill_seconds_per_token"])
        for name, entry in prompts.items()
    }
    unique = {n: c[0] for n, c in stage1.items() if len(c) == 1}

    # The identified prompts fix a monotone, convex R(q) curve. For an
    # ambiguous prompt, only one lattice point can sit on it: the others are
    # off by an integer factor and land far below the zero-draft round.
    anchors = sorted(
        (prompts[n]["effective_mean_draft_len"], decode[n] / r)
        for n, r in unique.items())

    def predict(q):
        if q <= anchors[0][0]:
            return anchors[0][1]
        for (q0, r0), (q1, r1) in zip(anchors, anchors[1:]):
            if q <= q1:
                return r0 + (r1 - r0) * (q - q0) / (q1 - q0)
        (q0, r0), (q1, r1) = anchors[-2], anchors[-1]
        return r1 + (r1 - r0) * (q - q1) / (q1 - q0)

    resolved = {}
    for name, candidates in stage1.items():
        q = prompts[name]["effective_mean_draft_len"]
        target = predict(q)
        best = min(candidates, key=lambda r: abs(decode[name] / r - target))
        resolved[name] = {
            "rounds": best,
            "lattice": candidates,
            "uniquely_identified_by_lattice": len(candidates) == 1,
            "predicted_R_seconds": target,
            "selected_R_seconds": decode[name] / best,
            "runner_up_R_seconds": sorted(
                (abs(decode[name] / r - target), decode[name] / r)
                for r in candidates)[1][1] if len(candidates) > 1 else None,
        }
    return resolved, decode


def ranked_table(prompts, resolved, decode):
    rows = []
    for name, entry in sorted(
        prompts.items(),
        key=lambda kv: kv[1]["effective_mean_draft_len"],
    ):
        rounds = resolved[name]["rounds"]
        q = entry["effective_mean_draft_len"]
        proposed = round(q * rounds)
        accepted = DECODE_TOKENS - rounds
        r_round = decode[name] / rounds
        bytes_round = TARGET_WEIGHT_PASS_BYTES + q * HEAD_WEIGHT_PASS_BYTES
        rows.append({
            "prompt": name,
            "rounds": rounds,
            "lattice": resolved[name]["lattice"],
            "uniquely_identified_by_lattice":
                resolved[name]["uniquely_identified_by_lattice"],
            "finding_319_bounds": list(FINDING_319_BOUNDS[name]),
            "q_proposed_per_round": q,
            "proposed_total": proposed,
            "accepted_draft_total": accepted,
            "a_accepted_per_round": accepted / rounds,
            "alpha_accept_fraction": (accepted / proposed) if proposed else None,
            "feasible_accepted_le_proposed": accepted <= proposed,
            "decode_seconds": decode[name],
            "R_ranked_seconds": r_round,
            "bytes_per_round": bytes_round,
            "achieved_bandwidth_bytes_per_second": bytes_round / r_round,
            "R_in_target_weight_passes": r_round / (
                TARGET_WEIGHT_PASS_BYTES / (bytes_round / r_round)),
            "mtp_seconds_per_token_mean": entry["mtp_seconds_per_token_mean"],
            "serial_seconds_per_token_mean":
                entry["serial_seconds_per_token_mean"],
            "prefill_seconds_per_token": entry["prefill_seconds_per_token"],
            "raw_ratio_of_means": entry["raw_ratio_of_means"],
            "non_drafting_round_count": entry["non_drafting_round_count"],
        })
    return rows


def median8(values):
    values = sorted(values)
    return (values[3] + values[4]) / 2


def published_effect(rows, saving_seconds_per_round):
    """Recompute the published median after a per-round candidate saving."""
    base = median8([r["raw_ratio_of_means"] for r in rows])
    moved = []
    for row in rows:
        saved = saving_seconds_per_round(row) * row["rounds"]
        mtp = row["mtp_seconds_per_token_mean"] - saved / DECODE_TOKENS
        moved.append(row["serial_seconds_per_token_mean"] / mtp)
    after = median8(moved)
    return {
        "median_before": base,
        "median_after": after,
        "published_percent": 100.0 * (after / base - 1.0),
    }


def ranked_batch_1(rows):
    """Ranked zero-draft round, anchored on the lowest-q prompt.

    The OLS intercept of a convex curve sits below every measured point, so
    it is not a batch-1 estimate. plutarch drafts 0.156 rows per round; back
    out that little drafting with the lowest measured segment marginal.
    """
    low, second = rows[0], rows[1]
    marginal = ((second["R_ranked_seconds"] - low["R_ranked_seconds"])
                / (second["q_proposed_per_round"] - low["q_proposed_per_round"]))
    return low["R_ranked_seconds"] - marginal * low["q_proposed_per_round"]


# Saving shapes. A head or draft-readout arm removes bytes once per draft
# step; a target arm removes them once per round; a row arm removes them once
# per verified row.
SAVING_SHAPES = {
    "per_draft_step": lambda row: row["q_proposed_per_round"],
    "per_round": lambda row: 1.0,
    "per_verify_row": lambda row: 1.0 + row["q_proposed_per_round"],
}


def transfer(rows, local, ranked_batch1):
    """Derive k for a pure byte-removal arm, with a bandwidth-anchor interval."""
    local_batch1_bandwidth = TARGET_WEIGHT_PASS_BYTES / local["R0_seconds"]
    ranked_batch1_bandwidth = TARGET_WEIGHT_PASS_BYTES / ranked_batch1
    pinned_serial_bandwidth = (
        TARGET_WEIGHT_PASS_BYTES / PINNED_SERIAL_ROUND_SECONDS)

    b1_bytes_local = B1_LOCAL_Q * B1_BYTES_PER_DRAFT_STEP
    save_bandwidth_local = b1_bytes_local / -B1_LOCAL_DELTA_R_SECONDS

    def k_at(scale, shape="per_draft_step"):
        bandwidth = save_bandwidth_local * scale
        effect = published_effect(
            rows,
            lambda row: (SAVING_SHAPES[shape](row) * B1_BYTES_PER_DRAFT_STEP
                         / bandwidth),
        )
        return effect["published_percent"]

    anchors = {
        "candidate_batch_1": ranked_batch1_bandwidth / local_batch1_bandwidth,
        "pinned_serial": pinned_serial_bandwidth / local_batch1_bandwidth,
    }

    out = {
        "local_batch_1_round_seconds": local["R0_seconds"],
        "local_batch_1_bandwidth_bytes_per_second": local_batch1_bandwidth,
        "ranked_batch_1_round_seconds": ranked_batch1,
        "ranked_batch_1_bandwidth_bytes_per_second": ranked_batch1_bandwidth,
        "pinned_serial_bandwidth_bytes_per_second": pinned_serial_bandwidth,
        "b1_bytes_removed_per_local_round": b1_bytes_local,
        "b1_local_save_bandwidth_bytes_per_second": save_bandwidth_local,
        "b1_local_published_percent": B1_LOCAL_PUBLISHED_PERCENT,
        "anchors": {},
    }
    for label, scale in anchors.items():
        percent = k_at(scale)
        out["anchors"][label] = {
            "bandwidth_scale": scale,
            "ranked_save_bandwidth_bytes_per_second":
                save_bandwidth_local * scale,
            "b1_published_ranked_percent": percent,
            "k": percent / B1_LOCAL_PUBLISHED_PERCENT,
        }

    # Invert the model on the one measured receipt pair.
    lo, hi = 1.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if k_at(mid) / B1_LOCAL_PUBLISHED_PERCENT > LEAF16_MEASURED_K:
            lo = mid
        else:
            hi = mid
    leaf16_scale = 0.5 * (lo + hi)
    out["leaf16_measured_k"] = LEAF16_MEASURED_K
    out["leaf16_implied_bandwidth_scale"] = leaf16_scale
    out["leaf16_scale_over_candidate_batch_1_anchor"] = (
        leaf16_scale / anchors["candidate_batch_1"])
    out["anchors"]["leaf16_calibrated"] = {
        "bandwidth_scale": leaf16_scale,
        "ranked_save_bandwidth_bytes_per_second":
            save_bandwidth_local * leaf16_scale,
        "b1_published_ranked_percent": k_at(leaf16_scale),
        "k": LEAF16_MEASURED_K,
    }

    ks = [v["k"] for v in out["anchors"].values()]
    out["k_interval"] = [min(ks), max(ks)]
    out["k_central_candidate_batch_1"] = out["anchors"]["candidate_batch_1"]["k"]

    # Deliverable 5: does k depend on which bytes an arm removes? Each shape
    # needs its OWN local baseline, because the same byte count removed once
    # per round is a different local saving from one removed per draft step.
    local_shape_units = {
        "per_draft_step": B1_LOCAL_Q,
        "per_round": 1.0,
        "per_verify_row": 1.0 + B1_LOCAL_Q,
    }
    out["k_by_saving_shape"] = {}
    for shape, units in local_shape_units.items():
        local_delta = units * B1_BYTES_PER_DRAFT_STEP / save_bandwidth_local
        fraction = local_delta / B1_LOCAL_ROUND_SECONDS
        local_percent = 100.0 * fraction / (1.0 - fraction)
        row = {"local_published_percent": local_percent}
        for label, anchor in out["anchors"].items():
            row[label] = k_at(anchor["bandwidth_scale"], shape) / local_percent
        out["k_by_saving_shape"][shape] = row
    return out


def roofline_multiples(rows, local, budget, ranked_batch1):
    """R over each host's own byte roofline, at the same batch-1 bandwidth."""
    local_bandwidth = TARGET_WEIGHT_PASS_BYTES / local["R0_seconds"]
    ranked_bandwidth = TARGET_WEIGHT_PASS_BYTES / ranked_batch1
    out = []
    adaptive = budget["adaptive_operating_point"]
    cells = [("local", "pinned_D=%d" % row["D"], row["q_proposed_per_round"],
              row["R_decode_seconds_mean"], local_bandwidth)
             for row in budget["dense_depth_table"]]
    cells.append(("local", "adaptive", adaptive["q_proposed_per_round"],
                  adaptive["R_decode_seconds"], local_bandwidth))
    cells.append(("local", "askeladd_b1_cell", B1_LOCAL_Q,
                  B1_LOCAL_ROUND_SECONDS, local_bandwidth))
    cells.extend(("ranked", row["prompt"], row["q_proposed_per_round"],
                  row["R_ranked_seconds"], ranked_bandwidth) for row in rows)
    for harness, cell, q, seconds, bandwidth in cells:
        payload = TARGET_WEIGHT_PASS_BYTES + q * HEAD_WEIGHT_PASS_BYTES
        roofline = payload / bandwidth
        out.append({
            "harness": harness, "cell": cell, "q_proposed_per_round": q,
            "R_seconds": seconds, "bytes_per_round": payload,
            "roofline_seconds": roofline,
            "R_over_roofline": seconds / roofline,
            "R_over_target_only_roofline": seconds / (
                TARGET_WEIGHT_PASS_BYTES / bandwidth),
        })
    return out


def main() -> None:
    board = load_board()
    prompts = per_prompt(board[ANCHOR])
    resolved, decode = identify_rounds(prompts)
    rows = ranked_table(prompts, resolved, decode)

    qs = [r["q_proposed_per_round"] for r in rows]
    rs = [r["R_ranked_seconds"] for r in rows]
    ranked_fit = ols(qs, rs)

    budget = json.loads((ARTIFACTS / "e159_round_budget.json").read_text())
    dense = {row["D"]: row for row in budget["dense_depth_table"]}
    local = {
        "R0_seconds": dense[0]["R_decode_seconds_mean"],
        "h_chord_0_7_seconds": budget["rho_variants"][
            "chord_0_7_shipped_cap"]["h_seconds"],
        "h_chord_0_8_seconds": budget["rho_variants"]["chord_0_8"]["h_seconds"],
        "h_ols_seconds": budget["h_slope_seconds"],
        "s_ols_seconds": budget["s_fixed_seconds"],
        "h_head_seconds": budget["h_head_seconds"],
    }

    # Marginal cost between adjacent ranked prompts, the ranked analogue of
    # the local per-width step table.
    ranked_steps = [
        {
            "from_prompt": a["prompt"], "to_prompt": b["prompt"],
            "from_q": a["q_proposed_per_round"], "to_q": b["q_proposed_per_round"],
            "marginal_seconds_per_proposed_row": (
                (b["R_ranked_seconds"] - a["R_ranked_seconds"])
                / (b["q_proposed_per_round"] - a["q_proposed_per_round"])),
        }
        for a, b in zip(rows, rows[1:])
    ]

    ranked_s = ranked_batch_1(rows)
    ranked_h = ranked_fit["slope"]
    local_s = local["R0_seconds"]
    roofline = {
        "target_weight_pass_bytes": TARGET_WEIGHT_PASS_BYTES,
        "head_weight_pass_bytes": HEAD_WEIGHT_PASS_BYTES,
        "local_batch_1_round_seconds": local_s,
        "local_batch_1_bandwidth": TARGET_WEIGHT_PASS_BYTES / local_s,
        "ranked_batch_1_round_seconds": ranked_s,
        "ranked_batch_1_bandwidth": TARGET_WEIGHT_PASS_BYTES / ranked_s,
        "ranked_batch_1_method": (
            "plutarch R minus the plutarch-to-drama marginal times its q; the "
            "OLS intercept of a convex curve is not a batch-1 estimate"),
        "ranked_ols_intercept_seconds": ranked_fit["intercept"],
        "finding_319_s_bounds": [0.02925, 0.03365],
        "ranked_batch_1_inside_finding_319": (
            0.02925 <= ranked_s <= 0.03365),
        "pinned_serial_round_seconds": PINNED_SERIAL_ROUND_SECONDS,
        "pinned_serial_bandwidth":
            TARGET_WEIGHT_PASS_BYTES / PINNED_SERIAL_ROUND_SECONDS,
        "candidate_batch_1_faster_than_pinned_serial_percent":
            100.0 * (PINNED_SERIAL_ROUND_SECONDS / ranked_s - 1.0),
        "local_h_over_s_chord_0_7": local["h_chord_0_7_seconds"] / local_s,
        "local_h_over_s_chord_0_8": local["h_chord_0_8_seconds"] / local_s,
        "local_h_over_s_ols": local["h_ols_seconds"] / local["s_ols_seconds"],
        "ranked_h_over_s": ranked_h / ranked_s,
        "local_over_ranked_h_over_s_chord_0_7":
            (local["h_chord_0_7_seconds"] / local_s) / (ranked_h / ranked_s),
        "ranked_8h_over_s": 8.0 * ranked_h / ranked_s,
        "finding_319_8h_over_s_upper_bound": 1.580,
        "ranked_8h_over_s_inside_finding_319":
            8.0 * ranked_h / ranked_s <= 1.580,
        "local_h_in_target_weight_passes":
            local["h_chord_0_7_seconds"] / local_s,
        "ranked_h_in_target_weight_passes": ranked_h / ranked_s,
        "h_is_per_proposed_row": True,
    }

    coefficient = transfer(rows, local, ranked_s)
    multiples = roofline_multiples(rows, local, budget, ranked_s)

    doc = {
        "experiment": "e159-r1-ranked-transfer",
        "anchor_receipt": ANCHOR,
        "crown_receipt": CROWN,
        "harness_of_board_quantities": "ranked",
        "harness_of_sweep_quantities": "local",
        "official_or_ranked_score": False,
        "decode_tokens": DECODE_TOKENS,
        "round_identification_method": (
            "effective_mean_draft_len is the exact rational proposed/rounds, "
            "so q*rounds must be an integer; intersect that lattice with the "
            "FINDING 319 bounds, then select on the monotone R(q) curve"),
        "prompts": rows,
        "rounds_total": sum(r["rounds"] for r in rows),
        "finding_319_rounds_total_bounds": [1263, 1917],
        "ranked_fit": ranked_fit,
        "ranked_marginal_steps": ranked_steps,
        "local": local,
        "roofline": roofline,
        "roofline_multiples": multiples,
        "transfer_coefficient": coefficient,
        "crown_gap_percent": 100.0 * (
            board[CROWN]["officialScore"] / board[ANCHOR]["officialScore"]
            - 1.0),
    }
    out = ARTIFACTS / "e159_ranked_transfer.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")

    print("ranked round identification, receipt", ANCHOR)
    print("%-9s %7s %-22s %8s %8s %8s %9s %8s" % (
        "prompt", "rounds", "lattice", "q", "a", "alpha", "R_ms", "GB/s"))
    for row in rows:
        print("%-9s %7d %-22s %8.4f %8.4f %8.4f %9.4f %8.1f" % (
            row["prompt"], row["rounds"], str(row["lattice"]),
            row["q_proposed_per_round"], row["a_accepted_per_round"],
            row["alpha_accept_fraction"], 1e3 * row["R_ranked_seconds"],
            row["achieved_bandwidth_bytes_per_second"] / 1e9))
    print("total rounds %d  (FINDING 319 bound [1263, 1917])"
          % doc["rounds_total"])
    print()
    print("ranked R(q) fit: s %.6f  h %.6f  R2 %.4f  8h/s %.4f" % (
        ranked_s, ranked_h, ranked_fit["r_squared"],
        roofline["ranked_8h_over_s"]))
    print("ranked marginal by segment (ms per proposed row):")
    for step in ranked_steps:
        print("  %-9s -> %-9s  q %.2f -> %.2f   %7.3f" % (
            step["from_prompt"], step["to_prompt"], step["from_q"],
            step["to_q"], 1e3 * step["marginal_seconds_per_proposed_row"]))
    print()
    print("batch-1 round: local %.4f ms (%.1f GB/s), ranked %.4f ms (%.1f GB/s)"
          % (1e3 * local_s, roofline["local_batch_1_bandwidth"] / 1e9,
             1e3 * ranked_s, roofline["ranked_batch_1_bandwidth"] / 1e9))
    print("h/s  local chord 0..7 %.4f   ranked %.4f   ratio %.2f" % (
        roofline["local_h_over_s_chord_0_7"], roofline["ranked_h_over_s"],
        roofline["local_over_ranked_h_over_s_chord_0_7"]))
    print()
    print("round cost over its own byte roofline")
    for cell in multiples:
        print("  %-7s %-16s q %.3f  R %7.3f ms  roofline %7.3f ms  x%.3f" % (
            cell["harness"], cell["cell"], cell["q_proposed_per_round"],
            1e3 * cell["R_seconds"], 1e3 * cell["roofline_seconds"],
            cell["R_over_roofline"]))
    print()
    print("transfer coefficient k, B1 head arm, crown gap %+.4f %%"
          % doc["crown_gap_percent"])
    for label, value in coefficient["anchors"].items():
        print("  %-20s scale %.3f  ranked %+.4f %%  k %.3f  %s" % (
            label, value["bandwidth_scale"],
            value["b1_published_ranked_percent"], value["k"],
            "TAKES CROWN" if value["b1_published_ranked_percent"]
            > doc["crown_gap_percent"] else "misses"))
    print("  k interval [%.3f, %.3f]   leaf16 measured %.3f" % (
        coefficient["k_interval"][0], coefficient["k_interval"][1],
        coefficient["leaf16_measured_k"]))
    print("  leaf16 implies bandwidth scale %.3f, which is %.3f x the "
          "candidate batch-1 anchor" % (
              coefficient["leaf16_implied_bandwidth_scale"],
              coefficient["leaf16_scale_over_candidate_batch_1_anchor"]))
    print("  k by saving shape (each shape against its own local baseline):")
    for shape, values in coefficient["k_by_saving_shape"].items():
        print("    %-16s local %+.4f %%   %s" % (
            shape, values["local_published_percent"],
            "  ".join("%s %.3f" % (label, value)
                      for label, value in values.items()
                      if label != "local_published_percent")))
    print()
    print("wrote", out)


if __name__ == "__main__":
    main()
