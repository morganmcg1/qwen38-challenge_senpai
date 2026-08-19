#!/usr/bin/env python3
"""E55: does the isolated M=9 two-stream win survive on the shipped table?

    research/e55_analyze.py --arms base m9two base2 [--wandb]

The candidate moves QMV dispatch `case 9` from `<T,9,3,true>` (3 weight streams)
to `<T,9,5,true>` (2 streams, 5+4) and relaxes the NA static_assert to admit
NA=5. Nothing else changes, and the diff is byte-neutral in the JIT twin.

PRIMARY is absolute candidate seconds per token, not the local ratio. Both local
legs share one build, so a change that speeds the target generally would cancel
in serial/mtp; this one is dispatched only at M>=2 and so should move the MTP leg
alone. That asymmetry is the falsifier, not a convenience.

Design is E48's bracket, not leg-level ABBA: each arm needs a full two-root
rebuild, so the candidate sits BETWEEN two byte-identical base arms. base-vs-base2
gives the same-session null and the drift estimate; the bracket mean is the
drift-corrected reference for the candidate.

Predictions were registered in research/e55-prereg.md BEFORE any GPU second, and
the constants below are copied from it verbatim. Reproducing that file's numbers
here is the point: the selection verdict is computed, not chosen after the fact.

Every leg here ran with MLXFAST_LOCAL_COOL_GATE=0. Nothing printed is
gate-qualified, ranked-equivalent, or an official score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e42_analyze import histogram, load_legs, mean, summarise_arm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / ".mlxfast-private/e55/runs"
PREREG = ROOT / "research/e55-prereg.md"
BASE_SHA = "a35bb006fd47785dc916241df63ec8780bda8e5c"
CANDIDATE_ARM = "m9two"
BASE_ARMS = ("base", "base2")

# --- pre-registered, from research/e55-prereg.md -----------------------------
# psi_mtp (E48) x M=9 cell win (E49 Arm 1) = % of MTP leg moved per unit f9,
# where f9 is the M=9 share of candidate-leg verify QMV cost.
PSI_MTP = 0.693391
M9_CELL_WIN_PCT = 12.255
SENSITIVITY_PCT_PER_F9 = 8.49751
# Competing f9 estimates. The measurement selects between them.
HYPOTHESES = {
    "e48_mine": 21.630,
    "edward_e53_upper": 8.900,
    "edward_e53_lower": 4.600,
}
# E48's byte-identical base2 rebuild, same host, same ungated protocol.
NULL_RAW_P_PCT = 0.0629
NULL_MTP_PCT = 0.0497
NULL_SERIAL_PCT = -0.0133
NULL_BASE_PER_LEG_RAW_P_SPREAD_PCT = 0.105
# Stop rule 2: within 3x the MTP-leg null floor is "did not transfer".
NULL_GUARD_MULTIPLE = 3.0
nan = float("nan")


def load_meta(arm: str) -> dict:
    """Provenance and thermal record written by research/e42-run.sh."""
    meta: dict[str, str] = {}
    for line in (RUNS / arm / "meta.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        meta[key] = value

    def gpu_c(raw: str | None) -> float:
        if not raw:
            return nan
        for field in raw.split():
            if field.startswith("gpu_temp="):
                return float(field[len("gpu_temp=") : -1])
        return nan

    legs = sorted(k for k in meta if k.startswith("leg") and k.endswith("_thermal_after"))
    return {
        "head_sha": meta["head_sha"],
        "twin_digests": meta["twin_digests"],
        "metallib_fingerprint": meta.get("metallib_fingerprint"),
        "started": meta.get("started"),
        "entry_gpu_temp_c": gpu_c(meta.get("thermal_before")),
        "exit_gpu_temp_c": gpu_c(meta.get("thermal_after")),
        "per_leg_exit_gpu_temp_c": [gpu_c(meta[k]) for k in legs],
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
    }


def load_arm(arm: str) -> dict:
    import e42_analyze

    e42_analyze.RUNS = RUNS
    return summarise_arm(arm, load_legs(arm), None)


def pct(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def selection_verdict(d_mtp_pct: float) -> dict:
    """The pre-registered selection table, applied to the measured MTP-leg move.

    f9_implied inverts the sensitivity, so it is only meaningful if the MTP leg
    actually moved down. A positive move is reported as its own outcome rather
    than folded into a magnitude.
    """
    # The sensitivity is % of MTP leg per UNIT f9, so inverting it yields a
    # fraction; the table is stated in percent.
    f9_implied = 100.0 * abs(d_mtp_pct) / SENSITIVITY_PCT_PER_F9
    guard = NULL_GUARD_MULTIPLE * NULL_MTP_PCT
    if d_mtp_pct > guard:
        selects = "neither: composed cell is SLOWER (prereg answer 2 of Risk 3)"
    elif d_mtp_pct > -guard:
        selects = "neither: within 3x the null floor, the isolated win did not transfer"
    elif d_mtp_pct < -2.2:
        selects = "neither: move exceeds every registered mixture (f9_implied > 26 %)"
    elif f9_implied < 3.5:
        selects = "neither: f9_implied below edward's lower bound"
    elif d_mtp_pct <= -1.30:
        selects = "E48 mixture (mine)"
    elif d_mtp_pct <= -1.05:
        selects = "between the mixtures, nearer E48"
    elif d_mtp_pct <= -0.50:
        selects = "edward E53 interval"
    else:
        selects = "edward E53 lower bound or below"
    return {
        "d_mtp_leg_pct": d_mtp_pct,
        "f9_implied_pct": f9_implied,
        "sensitivity_pct_per_f9": SENSITIVITY_PCT_PER_F9,
        "selects": selects,
        "null_guard_pct": guard,
        "beyond_null_guard": abs(d_mtp_pct) > guard,
        "null_multiple": abs(d_mtp_pct) / NULL_MTP_PCT if NULL_MTP_PCT else nan,
    }


def predictions() -> dict:
    return {
        "psi_mtp": PSI_MTP,
        "m9_cell_win_pct": M9_CELL_WIN_PCT,
        "sensitivity_pct_per_f9": SENSITIVITY_PCT_PER_F9,
        "serial_leg_pct": 0.0,
        "serial_leg_basis": (
            "the out_vec_size >= 4096 switch has no case 1; M=1 falls to "
            "default: break and reaches qmv_fast_impl untouched"
        ),
        "by_hypothesis": {
            name: {
                "f9_pct": f9,
                "predicted_mtp_leg_pct": -f9 * SENSITIVITY_PCT_PER_F9 / 100.0,
                "x_mtp_null": abs(f9 * SENSITIVITY_PCT_PER_F9 / 100.0) / NULL_MTP_PCT,
            }
            for name, f9 in HYPOTHESES.items()
        },
        "null_floors_pct": {
            "raw_p": NULL_RAW_P_PCT,
            "mtp_leg": NULL_MTP_PCT,
            "serial_leg": NULL_SERIAL_PCT,
            "base_per_leg_raw_p_spread": NULL_BASE_PER_LEG_RAW_P_SPREAD_PCT,
        },
    }


def prediction_errors(d_mtp_pct: float) -> dict:
    out = {}
    for name, rec in predictions()["by_hypothesis"].items():
        predicted = rec["predicted_mtp_leg_pct"]
        out[name] = {
            "predicted_mtp_leg_pct": predicted,
            "measured_mtp_leg_pct": d_mtp_pct,
            "error_pct_points": d_mtp_pct - predicted,
            "ratio_measured_over_predicted": d_mtp_pct / predicted if predicted else nan,
        }
    return out


def session_null(arms: dict) -> dict:
    """base vs base2: two byte-identical builds, one session, same protocol.

    This is the only null that shares this session's thermal history, so it
    supersedes E48's imported floors whenever both base arms are present.
    """
    if not all(a in arms for a in BASE_ARMS):
        return {"available": False, "reason": "needs both base and base2"}
    a, b = (arms[k] for k in BASE_ARMS)
    return {
        "available": True,
        "mtp_leg_pct": pct(b["mtp_seconds_per_token"], a["mtp_seconds_per_token"]),
        "serial_leg_pct": pct(b["serial_seconds_per_token"], a["serial_seconds_per_token"]),
        "raw_p_pct": pct(b["raw_p"], a["raw_p"]),
        "note": "byte-identical source; any move here is drift plus rebuild noise",
    }


def bracket_reference(arms: dict, key: str) -> dict:
    """Drift-corrected reference for the candidate, from the surrounding bases.

    With arms run base -> m9two -> base2 and drift monotone in time, the mean of
    the two base arms estimates the base level AT the candidate's slot. Using
    base alone instead leaves the whole session's drift in the effect.
    """
    present = [a for a in BASE_ARMS if a in arms]
    values = [arms[a][key] for a in present]
    return {
        "arms_used": present,
        "value": mean(values),
        "bracketed": len(present) == 2,
        "half_spread_pct": (
            50.0 * abs(pct(values[1], values[0])) if len(values) == 2 else nan
        ),
    }


def compare(arms: dict) -> dict:
    """Candidate against the bracket mean (primary) and base alone (secondary)."""
    cand = arms[CANDIDATE_ARM]
    out: dict = {}
    for label, key in (
        ("mtp_leg", "mtp_seconds_per_token"),
        ("serial_leg", "serial_seconds_per_token"),
        ("raw_p", "raw_p"),
    ):
        ref = bracket_reference(arms, key)
        entry = {
            "candidate": cand[key],
            "bracket_reference": ref["value"],
            "bracket_arms": ref["arms_used"],
            "bracketed": ref["bracketed"],
            "pct_vs_bracket": pct(cand[key], ref["value"]),
            "bracket_half_spread_pct": ref["half_spread_pct"],
        }
        if "base" in arms:
            entry["pct_vs_base_only"] = pct(cand[key], arms["base"][key])
        out[label] = entry
    return out


def fidelity(arms: dict) -> dict:
    """Stop rule 5 is a hard stop, so it is reported as a gate not a metric."""
    return {
        arm: {
            "all_tokens_matched": rec["all_tokens_matched"],
            "row_ledger_closes": rec["row_ledger_closes"],
            "round_count": rec["round_count"],
            "declared_rows_total": rec["declared_rows_total"],
            "accepted_draft_rate": rec["accepted_draft_rate"],
            "effective_mean_draft_len": rec["effective_mean_draft_len"],
        }
        for arm, rec in arms.items()
    }


def width_mix_identical(arms: dict) -> dict:
    """A schedule shift would move the M=9 share and confound the cell win.

    The candidate touches only which kernel serves M=9, never the schedule, so
    the histogram must be identical across arms. If it is not, the effect is not
    a pure cell substitution and the sensitivity constant no longer applies.
    """
    hists = {arm: json.dumps(rec["width_histogram"], sort_keys=True) for arm, rec in arms.items()}
    m9 = {arm: rec["width_histogram"].get(9, 0) for arm, rec in arms.items()}
    return {
        "identical_across_arms": len(set(hists.values())) == 1,
        "m9_rounds": m9,
        "histograms": {arm: json.loads(h) for arm, h in hists.items()},
    }


def leg_spread(arms: dict) -> dict:
    out = {}
    for arm, rec in arms.items():
        legs = rec["mtp_decode_seconds_all"]
        out[arm] = {
            "legs": len(legs),
            "mtp_decode_seconds_all": legs,
            "mtp_decode_seconds_sd_pct": rec.get("mtp_decode_seconds_sd_pct", 0.0),
            "serial_decode_seconds_all": rec["serial_decode_seconds_all"],
        }
    return out


def stop_rules(arms: dict, cmp: dict, null: dict) -> dict:
    """The registered rules, evaluated. Rule 1 was cleared before timing."""
    d_mtp = cmp["mtp_leg"]["pct_vs_bracket"]
    d_ser = cmp["serial_leg"]["pct_vs_bracket"]
    guard = NULL_GUARD_MULTIPLE * NULL_MTP_PCT
    serial_null = abs(null["serial_leg_pct"]) if null.get("available") else abs(NULL_SERIAL_PCT)
    matched = all(rec["all_tokens_matched"] for rec in arms.values())
    ledger = all(rec["row_ledger_closes"] for rec in arms.values())
    return {
        "rule1_vec5_compiles_and_lanes_exact": {
            "verdict": "PASS",
            "when": "cleared before any GPU timing second",
            "evidence": "research/e55_vec5_probe.metal, research/e55_vec5_check.swift",
        },
        "rule2_within_3x_null": {
            "triggered": abs(d_mtp) <= guard,
            "d_mtp_leg_pct": d_mtp,
            "guard_pct": guard,
            "meaning": "the isolated win did not transfer; stop and report",
        },
        "rule3_promotion_chain": {
            "triggered": d_mtp <= -1.0 and matched,
            "d_mtp_leg_pct": d_mtp,
            "all_tokens_matched": matched,
        },
        "rule4_serial_leg_moved": {
            "triggered": abs(d_ser) > NULL_GUARD_MULTIPLE * max(serial_null, abs(NULL_SERIAL_PCT)),
            "d_serial_leg_pct": d_ser,
            "serial_null_pct": serial_null,
            "meaning": "the dispatch model is wrong; M=1 should not reach case 9",
        },
        "rule5_bitwise_delta_at_m_le_9": {
            "hard_stop": not (matched and ledger),
            "all_tokens_matched": matched,
            "row_ledger_closes": ledger,
            "note": "512-token exactness is the gate; local rows are candidate-generated",
        },
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def strip(rec: dict) -> dict:
    return {
        k: v
        for k, v in rec.items()
        if not isinstance(v, (list, dict)) or k.endswith("histogram")
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "m9two", "base2"])
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    present = [a for a in args.arms if (RUNS / a / "meta.txt").exists()]
    missing = [a for a in args.arms if a not in present]
    arms = {a: load_arm(a) for a in present}
    provenance = {a: load_meta(a) for a in present}

    temps = [m["entry_gpu_temp_c"] for m in provenance.values() if m["entry_gpu_temp_c"] == m["entry_gpu_temp_c"]]
    payload: dict = {
        "experiment": "e55-compose-m9-two-stream-on-shipped-table",
        "assignment_id": "qwen38-r1-e55-compose-m9-two-stream-on-shipped-table",
        "pr": 57,
        "base_sha": BASE_SHA,
        "head_sha": git_head(),
        "host": "local-m4-pro",
        "ranked_host": "m5-qwen38-27b-mtp",
        "fixture": "correctness_prompts/public_longcopy_gate_english_512_256.json",
        "decode_tokens": 512,
        "offered_depth": 8,
        "design": "bracket: base -> m9two -> base2 (declared deviation from strict ABBA)",
        "primary_metric": "absolute candidate (MTP-leg) seconds per token",
        "arms_requested": args.arms,
        "arms_present": present,
        "arms_missing": missing,
        "prereg": str(PREREG.relative_to(ROOT)),
        "predictions": predictions(),
        "provenance": provenance,
        "entry_gpu_temp_spread_c": (max(temps) - min(temps)) if temps else nan,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "arms": {a: strip(rec) for a, rec in arms.items()},
        "leg_spread": leg_spread(arms),
        "fidelity": fidelity(arms),
        "width_mix": width_mix_identical(arms),
        "session_null": session_null(arms),
    }

    if CANDIDATE_ARM in arms:
        cmp = compare(arms)
        payload["comparison"] = cmp
        payload["selection"] = selection_verdict(cmp["mtp_leg"]["pct_vs_bracket"])
        payload["prediction_errors"] = prediction_errors(cmp["mtp_leg"]["pct_vs_bracket"])
        payload["stop_rules"] = stop_rules(arms, cmp, payload["session_null"])
    else:
        payload["comparison"] = {"available": False, "reason": f"{CANDIDATE_ARM} arm not run yet"}

    print(json.dumps(payload, indent=2, default=str))
    if args.wandb:
        log_wandb(payload)
    return 0


def log_wandb(payload: dict) -> None:
    import wandb

    arms = payload["arms"]
    tag = "-".join(payload["arms_present"]) or "empty"
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="qwen38-mlx-challenge-senpai",
        name=f"e55-m9-two-stream-{tag}",
        job_type="analysis",
        tags=["e55", "qwen3.8-27b-mtp-v1", "qmv-dispatch", "ungated-local", "pr57"],
        config={
            k: payload[k]
            for k in (
                "assignment_id",
                "pr",
                "base_sha",
                "head_sha",
                "host",
                "fixture",
                "decode_tokens",
                "offered_depth",
                "design",
                "primary_metric",
                "arms_present",
                "cool_gate_passed_real_gate",
                "gate_qualified_for_timing",
                "official_or_ranked_score",
            )
        },
    )
    summary: dict = {
        "arms_present": ",".join(payload["arms_present"]),
        "entry_gpu_temp_spread_c": payload["entry_gpu_temp_spread_c"],
        "width_mix_identical_across_arms": payload["width_mix"]["identical_across_arms"],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    for name, rec in payload["predictions"]["by_hypothesis"].items():
        summary[f"prereg/{name}/f9_pct"] = rec["f9_pct"]
        summary[f"prereg/{name}/predicted_mtp_leg_pct"] = rec["predicted_mtp_leg_pct"]
    summary["prereg/sensitivity_pct_per_f9"] = payload["predictions"]["sensitivity_pct_per_f9"]
    summary["prereg/serial_leg_pct"] = 0.0

    for arm, rec in arms.items():
        for key in (
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "raw_p",
            "accepted_draft_rate",
            "effective_mean_draft_len",
            "round_count",
            "declared_rows_total",
            "all_tokens_matched",
            "row_ledger_closes",
            "p50_block_request_seconds",
            "seed_prefill_seconds",
            "mtp_decode_seconds",
            "serial_decode_seconds",
            "mtp_decode_seconds_sd_pct",
        ):
            if key in rec:
                summary[f"{arm}/{key}"] = rec[key]
        summary[f"{arm}/m9_rounds"] = payload["width_mix"]["m9_rounds"].get(arm)
        meta = payload["provenance"][arm]
        summary[f"{arm}/entry_gpu_temp_c"] = meta["entry_gpu_temp_c"]
        summary[f"{arm}/exit_gpu_temp_c"] = meta["exit_gpu_temp_c"]
        summary[f"{arm}/head_sha"] = meta["head_sha"]
        summary[f"{arm}/twin_digests"] = meta["twin_digests"]

    null = payload["session_null"]
    if null.get("available"):
        for key in ("mtp_leg_pct", "serial_leg_pct", "raw_p_pct"):
            summary[f"null/{key}"] = null[key]

    cmp = payload.get("comparison") or {}
    if cmp.get("available") is not False:
        for label, rec in cmp.items():
            summary[f"delta/{label}_pct_vs_bracket"] = rec["pct_vs_bracket"]
            summary[f"delta/{label}_pct_vs_base_only"] = rec.get("pct_vs_base_only")
            summary[f"delta/{label}_bracketed"] = rec["bracketed"]
        sel = payload["selection"]
        summary["selection/selects"] = sel["selects"]
        summary["selection/f9_implied_pct"] = sel["f9_implied_pct"]
        summary["selection/null_multiple"] = sel["null_multiple"]
        summary["selection/beyond_null_guard"] = sel["beyond_null_guard"]
        for name, rec in payload["prediction_errors"].items():
            summary[f"error/{name}_pct_points"] = rec["error_pct_points"]
        for rule, rec in payload["stop_rules"].items():
            for field in ("triggered", "hard_stop", "verdict"):
                if field in rec:
                    summary[f"stop/{rule}"] = rec[field]

    arm_table = wandb.Table(
        columns=[
            "arm",
            "legs",
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "raw_p",
            "m9_rounds",
            "accepted_draft_rate",
            "all_tokens_matched",
            "row_ledger_closes",
            "entry_gpu_temp_c",
            "exit_gpu_temp_c",
        ]
    )
    for arm, rec in arms.items():
        meta = payload["provenance"][arm]
        arm_table.add_data(
            arm,
            rec["legs"],
            rec["mtp_seconds_per_token"],
            rec["serial_seconds_per_token"],
            rec["raw_p"],
            payload["width_mix"]["m9_rounds"].get(arm),
            rec["accepted_draft_rate"],
            rec["all_tokens_matched"],
            rec["row_ledger_closes"],
            meta["entry_gpu_temp_c"],
            meta["exit_gpu_temp_c"],
        )
    pred_table = wandb.Table(
        columns=["hypothesis", "f9_pct", "predicted_mtp_leg_pct", "measured_mtp_leg_pct", "error_pct_points"]
    )
    for name, rec in payload["predictions"]["by_hypothesis"].items():
        err = (payload.get("prediction_errors") or {}).get(name, {})
        pred_table.add_data(
            name,
            rec["f9_pct"],
            rec["predicted_mtp_leg_pct"],
            err.get("measured_mtp_leg_pct"),
            err.get("error_pct_points"),
        )
    leg_table = wandb.Table(columns=["arm", "leg", "mtp_decode_seconds", "serial_decode_seconds"])
    for arm, rec in payload["leg_spread"].items():
        for i, (m, s) in enumerate(
            zip(rec["mtp_decode_seconds_all"], rec["serial_decode_seconds_all"])
        ):
            leg_table.add_data(arm, i, m, s)
    width_table = wandb.Table(columns=["arm", "m", "rounds"])
    for arm, hist in payload["width_mix"]["histograms"].items():
        for m, count in sorted(hist.items(), key=lambda kv: int(kv[0])):
            width_table.add_data(arm, int(m), count)

    run.summary.update({k: v for k, v in summary.items() if v is not None})
    run.log(
        {
            "arms": arm_table,
            "predictions_vs_measured": pred_table,
            "legs": leg_table,
            "width_histograms": width_table,
        }
    )
    run.finish()
    print(f"wandb_run_url={run.url}", file=sys.stderr)
    print(f"wandb_run_id={run.id}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
