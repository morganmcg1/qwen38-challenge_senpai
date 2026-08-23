"""E159 step 4.2: price the cross-row QMV input-group partition.

The candidate's wide affine-4/g64 QMV picks inputs-per-group from a fixed width
table and launches ceil(m/ipg) input groups. Each group re-reads the whole
weight matrix, so a group-count increase costs one extra weight pass. The table
raises the group count at exactly two widths, 6 and 9, and the local dense depth
sweep shows a cost cliff at exactly those two widths and nowhere else.
"""

import json
import pathlib

ART = pathlib.Path(__file__).resolve().parent / "e159-artifacts"

# Local dense fixed-depth sweep, harness=local, M4 Pro, from
# e159_round_budget.json -> dense_depth_table. The width law is read from the
# decode series, which is the target forward the QMV cells belong to; the round
# series carries the proposal head and session work as well.
R_DEC = [65.813, 69.484, 71.561, 78.323, 91.634, 127.084, 138.065, 145.655, 185.427]
R_TOT = [73.634, 84.929, 94.310, 108.164, 128.098, 168.468, 186.355, 199.772, 244.281]
ACCEPTED = [0.0, 0.9807, 1.9091, 2.8209, 3.6545, 4.2784, 5.1687, 5.9189, 6.5294]

# Qwen35.swift:1568 width table and Qwen35.swift:1715 activeInputGroups.
INPUTS_PER_GROUP = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}


def groups(m: int) -> int:
    ipg = INPUTS_PER_GROUP[m]
    return (m + ipg - 1) // ipg


def mtp_ms_per_token(r_tot: float, accepted: float) -> float:
    return r_tot / (accepted + 1.0)


def main() -> None:
    widths = list(range(2, 10))
    marginal = {w: R_DEC[i] - R_DEC[i - 1] for i, w in zip(range(1, 9), widths)}
    delta_groups = {w: groups(w) - (groups(w - 1) if w > 2 else 1) for w in widths}

    rows = []
    for w in widths:
        rows.append(
            {
                "width": w,
                "marginal_ms": round(marginal[w], 3),
                "input_groups": groups(w),
                "delta_groups": delta_groups[w],
            }
        )

    flat = [marginal[w] for w in widths if delta_groups[w] == 0]
    cliff = [marginal[w] for w in widths if delta_groups[w] > 0]

    # Neighbour baselines for the two cliff widths.
    base6 = (marginal[5] + marginal[7]) / 2.0
    base9 = (marginal[7] + marginal[8]) / 2.0
    excess6 = marginal[6] - base6
    excess9 = marginal[9] - base9

    base_mtp = mtp_ms_per_token(R_TOT[4], ACCEPTED[4])

    def swept(save6: float, save9: float, max_depth: int):
        out = {}
        for d in range(0, max_depth + 1):
            save = 0.0
            if d >= 5:
                save += save6
            if d >= 8:
                save += save9
            out[d] = mtp_ms_per_token(R_TOT[d] - save, ACCEPTED[d])
        return out

    cap7 = swept(excess6, 0.0, 7)
    cap8 = swept(excess6, excess9, 8)
    best7 = min(cap7.values())
    best8 = min(cap8.values())

    # Adaptive arm, same session: mtp 28.68 ms/token, 5.564 accepted per round,
    # 78 rounds, proposed-depth histogram below. Verify width is depth + 1.
    hist = {3: 4, 4: 7, 5: 5, 6: 3, 7: 59}
    n_rounds = sum(hist.values())
    frac_deep = sum(v for k, v in hist.items() if k + 1 >= 6) / n_rounds
    mtp_adaptive = 28.68
    accepted_adaptive = 5.564
    round_ms = mtp_adaptive * (accepted_adaptive + 1.0)
    new_adaptive = (round_ms - frac_deep * excess6) / (accepted_adaptive + 1.0)
    g = (mtp_adaptive - new_adaptive) / mtp_adaptive

    result = {
        "experiment": "e159-step4.2-qmv-input-group-partition",
        "harness": "local",
        "official_or_ranked_score": False,
        "rule79_not_evidence": True,
        "host": "M4 Pro",
        "source_of_law": {
            "width_table": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1568",
            "active_input_groups": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1715",
            "grid_x_extent": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1834,1883",
            "weight_read_indexing": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1455-1472",
        },
        "marginal_series": "decode phase (R_dec), not the full round",
        "per_width": rows,
        "flat_marginal_ms": {
            "mean": round(sum(flat) / len(flat), 3),
            "min": round(min(flat), 3),
            "max": round(max(flat), 3),
            "n": len(flat),
        },
        "cliff_marginal_ms": {
            "mean": round(sum(cliff) / len(cliff), 3),
            "widths": [w for w in widths if delta_groups[w] > 0],
        },
        "excess_over_neighbours_ms": {
            "width_6": round(excess6, 3),
            "width_9": round(excess9, 3),
            "tree_priced_full_weight_pass_ms": 25.0,
        },
        "placement_null_probability": 1.0 / 28.0,
        "projection_fixed_depth": {
            "baseline_argmin_mtp_ms": round(base_mtp, 4),
            "cap7_only_width6_fixed": {
                "by_depth": {str(k): round(v, 4) for k, v in cap7.items()},
                "argmin_mtp_ms": round(best7, 4),
                "pct": round(100.0 * (best7 - base_mtp) / base_mtp, 3),
            },
            "cap8_width6_and_width9_fixed": {
                "argmin_mtp_ms": round(best8, 4),
                "pct": round(100.0 * (best8 - base_mtp) / base_mtp, 3),
            },
        },
        "projection_adaptive_arm": {
            "rounds": n_rounds,
            "fraction_of_rounds_at_verify_width_ge_6": round(frac_deep, 4),
            "round_ms": round(round_ms, 3),
            "mtp_ms_per_token_now": mtp_adaptive,
            "mtp_ms_per_token_projected": round(new_adaptive, 4),
            "candidate_leg_saving_g": round(g, 4),
            "implied_median_move_pct_rule176": round(100.0 * g / (1.0 - g), 2),
        },
    }

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "e159_qmv_group_law.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print("width  marg_ms  groups  d_groups")
    for r in rows:
        print(
            "  %d   %8.3f     %d      %+d"
            % (r["width"], r["marginal_ms"], r["input_groups"], r["delta_groups"])
        )
    print()
    print("flat marginals   n=%d mean %.3f  range [%.3f, %.3f]"
          % (len(flat), sum(flat) / len(flat), min(flat), max(flat)))
    print("cliff marginals  widths %s  mean %.3f"
          % ([w for w in widths if delta_groups[w] > 0], sum(cliff) / len(cliff)))
    print("width-6 excess over interp(w5,w7)=%.3f  ->  %.3f ms" % (base6, excess6))
    print("width-9 excess over mean(w7,w8)=%.3f   ->  %.3f ms" % (base9, excess9))
    print()
    print("fixed-depth argmin  %.4f -> %.4f ms/tok (%.2f%%) with width 6 fixed, cap 7"
          % (base_mtp, best7, 100.0 * (best7 - base_mtp) / base_mtp))
    print("fixed-depth argmin  %.4f -> %.4f ms/tok (%.2f%%) with widths 6 and 9 fixed, cap 8"
          % (base_mtp, best8, 100.0 * (best8 - base_mtp) / base_mtp))
    print()
    print("adaptive arm  %.4f -> %.4f ms/tok  g=%.4f  implied median %+.2f%%"
          % (mtp_adaptive, new_adaptive, g, 100.0 * g / (1.0 - g)))
    print("wrote", out)


if __name__ == "__main__":
    main()
