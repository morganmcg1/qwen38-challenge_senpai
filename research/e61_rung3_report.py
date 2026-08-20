#!/usr/bin/env python3
"""E61 rung 3: does the single weight stream at M=6 pay on a whole 512-token leg?

Rung 3 is the confirmation run, not a screen. It measures the arm effect on the
CANDIDATE MTP seconds-per-token directly, which is the headline; every projection
sits beside that number, never in front of it.

Design: `shipped:warm shipped:base t6:m6 t6:m6b shipped:base2`, two legs per tag.
The warm-up tag is declared and discarded. The remaining eight legs are ABBA in
time, so `shipped` occupies leg positions 1,2,7,8 and `t6` occupies 3,4,5,6.
Both arms therefore have mean position 4.5 and the arm effect is orthogonal to
linear thermal or clock drift. `time ~ arm + leg_position` is fitted on those
eight legs.

Timing used the permitted local ungated protocol, so the three gate flags travel
verbatim and this is not a ranked or official score.

  python3 research/e61_rung3_report.py --out research/e61-rung3.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics

RUNS = pathlib.Path(".mlxfast-private/e61/runs")
PROJECTION = pathlib.Path("research/e61-artifacts/e61-projection.json")

# tag -> (arm, discarded)
PLAN = (
    ("warm", "shipped", True),
    ("base", "shipped", False),
    ("m6", "t6", False),
    ("m6b", "t6", False),
    ("base2", "shipped", False),
)


def pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def read_meta(tag: str) -> dict:
    out = {}
    p = RUNS / tag / "meta.txt"
    for line in p.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def read_legs() -> list[dict]:
    """One record per timed leg, in the order they ran."""
    legs = []
    position = 0
    for tag, arm, discarded in PLAN:
        meta = read_meta(tag)
        for path in sorted(glob.glob(str(RUNS / tag / "score-*.json"))):
            leg_index = int(pathlib.Path(path).stem.split("-")[1])
            metrics = json.loads(pathlib.Path(path).read_text())["metrics"]
            rep = RUNS / tag / "reports" / ("leg-%d" % leg_index)
            serial = json.loads((rep / "03-mtp-timed.json").read_text())
            mtp = json.loads((rep / "04-mtp-timed.json").read_text())
            if not serial.get("is_serial_control"):
                raise SystemExit("%s: 03-mtp-timed.json is not the serial control" % rep)
            if mtp.get("is_serial_control"):
                raise SystemExit("%s: 04-mtp-timed.json is the serial control" % rep)
            if not discarded:
                position += 1
            legs.append({
                "tag": tag,
                "arm": arm,
                "discarded": discarded,
                "leg_index": leg_index,
                "position": None if discarded else position,
                "head_sha": meta.get("head_sha"),
                "dirty": meta.get("dirty"),
                "twin_digests": meta.get("twin_digests"),
                "binary_assert_m6_na": meta.get("e61_binary_assert_m6_na"),
                "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                "effective_mean_draft_len": metrics["effective_mean_draft_len"],
                "accepted_draft_rate": metrics["accepted_draft_rate"],
                "all_tokens_matched": metrics["all_tokens_matched"],
                "residual_divergence_count": metrics["residual_divergence_count"],
                "decode_tokens": metrics["decode_tokens"],
                # E55 falsifiers.
                "serial_p50_block_seconds": serial["p50_block_request_seconds"],
                "serial_seed_prefill_seconds": serial["seed_prefill_seconds"],
                "serial_round_count": serial["round_count"],
                "mtp_seed_prefill_seconds": mtp["seed_prefill_seconds"],
                "mtp_p50_block_seconds": mtp["p50_block_request_seconds"],
                "mtp_round_count": mtp["round_count"],
                "mtp_decode_seconds": mtp["decode_seconds"],
            })
    return legs


def ols_arm_and_position(legs: list[dict]) -> dict:
    """Fit y = b0 + b1*arm + b2*position by normal equations (3x3 solve)."""
    rows = [(1.0, 1.0 if l["arm"] == "t6" else 0.0, float(l["position"]),
             l["mtp_seconds_per_token"]) for l in legs]
    n = len(rows)
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    xty = [sum(r[i] * r[3] for r in rows) for i in range(3)]

    # Gauss-Jordan on the 3x4 augmented matrix.
    aug = [xtx[i] + [xty[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(aug[r][c]))
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(3):
            if r == c:
                continue
            f = aug[r][c]
            aug[r] = [v - f * w for v, w in zip(aug[r], aug[c])]
    beta = [aug[i][3] for i in range(3)]

    fitted = [beta[0] + beta[1] * r[1] + beta[2] * r[2] for r in rows]
    resid = [r[3] - f for r, f in zip(rows, fitted)]
    dof = n - 3
    s2 = sum(e * e for e in resid) / dof
    # Standard error of b1 needs (X'X)^-1[1][1]; recover it by solving again.
    inv = []
    for k in range(3):
        aug2 = [xtx[i][:] + [1.0 if i == k else 0.0] for i in range(3)]
        for c in range(3):
            p = max(range(c, 3), key=lambda r: abs(aug2[r][c]))
            aug2[c], aug2[p] = aug2[p], aug2[c]
            piv = aug2[c][c]
            aug2[c] = [v / piv for v in aug2[c]]
            for r in range(3):
                if r == c:
                    continue
                f = aug2[r][c]
                aug2[r] = [v - f * w for v, w in zip(aug2[r], aug2[c])]
        inv.append([aug2[i][3] for i in range(3)])
    se_b1 = (s2 * inv[1][1]) ** 0.5
    base_mean = statistics.fmean(
        l["mtp_seconds_per_token"] for l in legs if l["arm"] == "shipped")
    return {
        "model": "mtp_seconds_per_token ~ arm + leg_position",
        "n": n,
        "intercept": beta[0],
        "arm_coef_seconds_per_token": beta[1],
        "arm_coef_pct_of_shipped_mean": beta[1] / base_mean * 100.0,
        "arm_coef_stderr": se_b1,
        "arm_coef_t": beta[1] / se_b1,
        "position_coef_seconds_per_leg": beta[2],
        "position_coef_pct_per_leg": beta[2] / base_mean * 100.0,
        "residual_sd_seconds": s2 ** 0.5,
        "dof": dof,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e61-rung3.json")
    args = ap.parse_args()

    legs = read_legs()
    timed = [l for l in legs if not l["discarded"]]
    warm = [l for l in legs if l["discarded"]]

    by_arm = {}
    for arm in ("shipped", "t6"):
        vals = [l["mtp_seconds_per_token"] for l in timed if l["arm"] == arm]
        by_arm[arm] = {
            "legs": len(vals),
            "mtp_seconds_per_token_mean": statistics.fmean(vals),
            "mtp_seconds_per_token_min": min(vals),
            "mtp_seconds_per_token_max": max(vals),
            "same_arm_spread_pct": (max(vals) - min(vals)) / statistics.fmean(vals) * 100.0,
        }

    shipped_mean = by_arm["shipped"]["mtp_seconds_per_token_mean"]
    t6_mean = by_arm["t6"]["mtp_seconds_per_token_mean"]
    abs_delta = t6_mean - shipped_mean
    L = pct(t6_mean, shipped_mean)

    # The advisor retracted the 0.0629 % floor. The conservative session null is
    # this session's own largest same-arm spread.
    null_pct = max(by_arm[a]["same_arm_spread_pct"] for a in by_arm)

    # Tag-level drift controls: same arm, different tag, different build.
    def tag_mean(tag: str) -> float:
        return statistics.fmean(
            l["mtp_seconds_per_token"] for l in timed if l["tag"] == tag)

    drift = {
        "shipped_base_vs_base2_pct": pct(tag_mean("base2"), tag_mean("base")),
        "t6_m6_vs_m6b_pct": pct(tag_mean("m6b"), tag_mean("m6")),
    }

    proj = json.loads(PROJECTION.read_text())
    C = proj["inputs"]["measured_m6_cell_delta_pct"]
    f6_local = proj["local"]["f6"]
    f6_ranked = proj["ranked_e53"]["headline"]["f6"]
    f6_ranked_min = proj["ranked_e53"]["f6_min"]
    f6_ranked_max = proj["ranked_e53"]["f6_max"]
    psi_assumed = proj["inputs"]["psi_mtp"]

    implied_psi = L / (C * f6_local)

    # Ranked projection driven by the MEASURED transfer, not by the assumed psi.
    def ranked(f6: float) -> float:
        return C * f6 * implied_psi

    # E55 falsifiers. None of these may move with the arm: the serial leg is
    # depth-0 and never reaches the wide multi-row dispatch, and prefill is
    # outside the drafting loop.
    def arm_mean(key: str, arm: str) -> float:
        return statistics.fmean(l[key] for l in timed if l["arm"] == arm)

    def metric_null_pct(key: str) -> float:
        """A falsifier must be judged against ITS OWN metric's within-arm spread.

        Judging every metric against the headline metric's null is wrong in both
        directions: it flags quiet metrics that are simply noisier, and it hides
        real movement in metrics that are quieter.
        """
        spreads = []
        for arm in ("shipped", "t6"):
            vals = [l[key] for l in timed if l["arm"] == arm]
            spreads.append((max(vals) - min(vals)) / statistics.fmean(vals) * 100.0)
        return max(spreads)

    falsifiers = {}
    for name, key in (("serial_leg_seconds_per_token", "serial_seconds_per_token"),
                      ("serial_round_cost_p50_seconds", "serial_p50_block_seconds"),
                      ("serial_seed_prefill_seconds", "serial_seed_prefill_seconds"),
                      ("mtp_seed_prefill_seconds", "mtp_seed_prefill_seconds")):
        s, t = arm_mean(key, "shipped"), arm_mean(key, "t6")
        own_null = metric_null_pct(key)
        by_position = [(l["position"], l[key]) for l in sorted(
            timed, key=lambda x: x["position"])]
        falsifiers[name] = {
            "shipped": s, "t6": t, "delta_pct": pct(t, s),
            "own_metric_null_pct": own_null,
            "inside_own_metric_null": abs(pct(t, s)) < own_null,
            "by_leg_position": by_position,
            "enters_headline": False,
        }

    schedule_identical = (
        len({l["effective_mean_draft_len"] for l in timed}) == 1
        and len({l["accepted_draft_rate"] for l in timed}) == 1
        and len({l["mtp_round_count"] for l in timed}) == 1)

    correctness_ok = (
        all(l["all_tokens_matched"] for l in legs)
        and all(l["residual_divergence_count"] == 0 for l in legs)
        and all(l["decode_tokens"] == 512 for l in legs))

    # The headline is decode-only. Prefill drift therefore cannot reach it, which
    # is checked here rather than assumed.
    headline_excludes_prefill = all(
        abs(l["mtp_decode_seconds"] / l["decode_tokens"]
            - l["mtp_seconds_per_token"]) < 1e-12 for l in timed)

    report = {
        "experiment": "qwen38-r1-e61-single-weight-stream-qmv-m6",
        "rung": 3,
        "role": "confirmation run, not a screen",
        "base_sha": "d2139c924c7a7d98ca6026eea63867c2776abbca",
        "host": "Apple M4 Pro, 48 GiB",
        "decode_tokens": 512,
        "mtp_depth": 8,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "warm_up_legs_declared_and_discarded": [l["tag"] for l in warm],
        "legs": legs,
        "by_arm": by_arm,
        "headline": {
            "metric": "candidate MTP seconds per token, absolute",
            "shipped": shipped_mean,
            "t6": t6_mean,
            "absolute_delta_seconds_per_token": abs_delta,
            "relative_delta_pct": L,
        },
        "session_null": {
            "conservative_null_pct": null_pct,
            "definition": "largest same-arm spread in this session",
            "retracted_floor_pct": 0.0629,
            "effect_multiple_of_null": abs(L) / null_pct,
        },
        "drift_controls": drift,
        "regression": ols_arm_and_position(timed),
        "transfer": {
            "measured_m6_cell_delta_pct_C": C,
            "f6_local_from_own_round_census": f6_local,
            "own_round_census": proj["inputs"]["local_rounds"],
            "C_times_f6_local_pct": C * f6_local,
            "implied_psi_mtp_local_leg": implied_psi,
            "psi_assumed_by_advisor_repricing": psi_assumed,
            "predicted_L_if_psi_assumed_pct": C * f6_local * psi_assumed,
            "measured_L_pct": L,
        },
        "projection_local_fixture_weighted_pct": L,
        "projection_ranked_mixture_weighted": {
            "basis": "E53 ranked width mixture x measured transfer",
            "f6_ranked_headline": f6_ranked,
            "f6_ranked_min": f6_ranked_min,
            "f6_ranked_max": f6_ranked_max,
            "ranked_leg_delta_pct_headline": ranked(f6_ranked),
            "ranked_leg_delta_pct_min_f6": ranked(f6_ranked_min),
            "ranked_leg_delta_pct_max_f6": ranked(f6_ranked_max),
            "ranked_qmv_delta_pct": C * f6_ranked,
        },
        "falsifiers": falsifiers,
        "schedule_identical_across_all_legs": schedule_identical,
        "correctness_ok_every_leg": correctness_ok,
        "headline_excludes_prefill": headline_excludes_prefill,
        "wired_residency_active": False,
        "caveats": [
            "The local-to-ranked projection assumes wired residency does not "
            "change the bandwidth ladder. That is untested here and is E62's "
            "question. No number above is adjusted for it.",
            "Timing is ungated. It is directional causal evidence inside this "
            "counterbalanced session, and it is not a gate-qualified or ranked "
            "score.",
            "The ranked mixture weights come from E53's fitted width mixture, "
            "not from a ranked measurement.",
        ],
    }

    bands = proj["decision_bands"]
    if L <= bands["promote_at_or_below_pct"]:
        verdict = "promote"
    elif L < bands["report_only_below_pct"]:
        verdict = "report-only"
    else:
        verdict = "stop"
    report["decision_bands"] = bands
    report["verdict"] = verdict
    report["verdict_gated_on"] = {
        "effect_beats_session_null": abs(L) > null_pct,
        "correctness_ok_every_leg": correctness_ok,
        "schedule_identical": schedule_identical,
        "falsifiers_quiet": all(
            f["inside_own_metric_null"] for f in falsifiers.values()),
        "headline_excludes_prefill": headline_excludes_prefill,
    }

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("pos tag    arm      mtp s/tok    serial s/tok  speedup")
    for l in legs:
        print("%3s %-6s %-8s %.8f  %.8f  %.6f  %s"
              % (l["position"] if l["position"] else "-", l["tag"], l["arm"],
                 l["mtp_seconds_per_token"], l["serial_seconds_per_token"],
                 l["mtp_decode_speedup"], "DISCARDED" if l["discarded"] else ""))
    print()
    print("shipped mean  %.8f  (spread %.4f %%)"
          % (shipped_mean, by_arm["shipped"]["same_arm_spread_pct"]))
    print("t6      mean  %.8f  (spread %.4f %%)"
          % (t6_mean, by_arm["t6"]["same_arm_spread_pct"]))
    print("HEADLINE      %+.8f s/tok   = %+.4f %%" % (abs_delta, L))
    print("session null  %.4f %%   effect is %.2fx the null"
          % (null_pct, abs(L) / null_pct))
    r = report["regression"]
    print("OLS arm coef  %+.8f s/tok = %+.4f %%  (t = %.2f, position %+.5f %%/leg)"
          % (r["arm_coef_seconds_per_token"], r["arm_coef_pct_of_shipped_mean"],
             r["arm_coef_t"], r["position_coef_pct_per_leg"]))
    print("implied psi   %.4f   (advisor assumed %.4f)" % (implied_psi, psi_assumed))
    print("ranked proj   %+.4f %% (f6=%.4f), range %+.4f .. %+.4f %%"
          % (ranked(f6_ranked), f6_ranked, ranked(f6_ranked_max), ranked(f6_ranked_min)))
    print("falsifiers (each judged against its own metric's within-arm spread):")
    for k, v in falsifiers.items():
        print("  %-32s %+.4f %%  (own null %.4f %%)  %s"
              % (k, v["delta_pct"], v["own_metric_null_pct"],
                 "quiet" if v["inside_own_metric_null"] else "MOVED"))
    print("headline excludes prefill: %s" % headline_excludes_prefill)
    print("\nverdict: %s -> %s" % (verdict, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
