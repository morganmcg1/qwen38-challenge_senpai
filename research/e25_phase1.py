#!/usr/bin/env python3
"""Research-only (qwen38-r1-e25-per-row-draft-price): reduce the Phase 1 runs.

E25 Phase 1 tests thorfinn's E22 follow-up #1 on the GPU: does the measured
per-row draft price (arm PRICE / "arm D") beat the shipped scalar price (arm
BASE) on MTP TRUE DECODE, and does the realised depth histogram match the
Phase 0 offline prediction?

Primary metric, per PR #29 section 8: per-prompt MTP true decode
`decode_seconds - seed_prefill_seconds`, headline = MEDIAN OF 8.
`decode_seconds` is prefill-INCLUSIVE (calibration fact (c)), which is exactly
the contamination E25 was told to correct, so the prefill term is subtracted on
BOTH arms and the pooled figure is reported as secondary only.

Usage:
  research/e25_phase1.py [--runs DIR] [--out JSON] [--wandb]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e21_trace import parse_trace  # noqa: E402

PROMPTS = [
    "english", "narrative", "technical", "dramatic",
    "travel", "philosophy", "natural_history", "medicine",
]
ARMS = ["BASE", "PRICE"]

# Pre-registered in the PR #29 Phase 0 comment, BEFORE any GPU was spent.
PREREG = {
    "depth_histogram": {1: 193, 2: 1419, 3: 335},
    "depth_ge_4_rounds": 0,
    "pooled_true_decode_gain_pct": 5.2745,
    "median_of_8_true_decode_gain_pct": 4.6881,
    "rows_proposed_base": 4645,
    "rows_proposed_price": 4036,
    "mean_depth_base": 2.3857,
    "mean_depth_price": 2.0729,
    "per_prompt_gain_pct": {
        "english": 4.223, "narrative": 5.024, "technical": 3.899,
        "dramatic": 9.775, "travel": 4.352, "philosophy": 5.768,
        "natural_history": 3.050, "medicine": 6.940,
    },
    "tokens_lost_on_tape": 98,
}

SERIAL, MTP = "03-mtp-timed.json", "04-mtp-timed.json"


def read_meta(d: Path) -> dict:
    meta = {}
    p = d / "meta.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def read_leg(d: Path, name: str) -> dict | None:
    p = d / "reports" / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def true_decode(leg: dict) -> float:
    return leg["decode_seconds"] - leg["seed_prefill_seconds"]


def load_run(runs: Path, label: str) -> dict | None:
    d = runs / label
    if not d.is_dir():
        return None
    serial, mtp = read_leg(d, SERIAL), read_leg(d, MTP)
    if serial is None or mtp is None:
        return None
    meta = read_meta(d)
    score = {}
    if (d / "score.json").exists():
        score = json.loads((d / "score.json").read_text())

    # Both legs of one --local-iterate run share the same build, so the serial
    # leg is a per-run control: its cross-arm spread is that prompt's noise
    # floor and the reason a same-arm regression is distinguishable from drift.
    out = {
        "label": label,
        "arm": meta.get("arm"),
        "worker_sha256": meta.get("worker_sha256"),
        "source_sha256": meta.get("source_sha256"),
        "head_sha": meta.get("head_sha"),
        "dirty": meta.get("dirty"),
        "pass": meta.get("pass"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "thermal_before": meta.get("thermal_before"),
        "thermal_after": meta.get("thermal_after"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "exit": meta.get("exit"),
        "serial_leg_seconds": serial["decode_seconds"],
        "serial_prefill_seconds": serial["seed_prefill_seconds"],
        "serial_true_decode": true_decode(serial),
        "serial_rounds": serial["round_count"],
        "serial_all_tokens_matched": serial["all_tokens_matched"],
        "mtp_leg_seconds": mtp["decode_seconds"],
        "mtp_prefill_seconds": mtp["seed_prefill_seconds"],
        "mtp_true_decode": true_decode(mtp),
        "mtp_rounds": mtp["round_count"],
        "mtp_emitted": mtp["emitted_token_total"],
        "mtp_declared_rows": mtp["declared_rows_total"],
        "mtp_accepted_rows": mtp["accepted_draft_total"],
        "mtp_rejected_rows": mtp["rejected_draft_total"],
        "mtp_accepted_rate": mtp["accepted_draft_rate"],
        "mtp_mean_depth": mtp["effective_mean_draft_len"],
        "mtp_max_depth": mtp["effective_max_draft_len"],
        "mtp_non_drafting_rounds": mtp["non_drafting_round_count"],
        "mtp_replayed_rounds": mtp.get("verify_block_replayed_round_count"),
        "mtp_p50_block_seconds": mtp.get("p50_block_request_seconds_after_first"),
        "mtp_all_tokens_matched": mtp["all_tokens_matched"],
        "mtp_residual_divergence": mtp["residual_divergence_count"],
        "mtp_parity_all_ok": mtp.get("parity_all_ok"),
        "mtp_rejected_rows_reference_checked": mtp.get("rejected_rows_reference_checked"),
        "mtp_max_rejected_tail_logit_delta": mtp.get("max_rejected_tail_logit_delta"),
        "head_sha256": mtp.get("head_provenance", {}).get("sha256"),
        "uses_pinned_mtp_head": mtp.get("uses_pinned_mtp_head"),
        "local_iterate_speedup": score.get("score"),
        "local_iterate_passed": score.get("passed"),
    }
    hist = {}
    for n in mtp.get("effective_draft_lengths") or []:
        hist[int(n)] = hist.get(int(n), 0) + 1
    out["depth_histogram_from_report"] = dict(sorted(hist.items()))
    return out


def correctness_ok(r: dict) -> bool:
    return bool(
        r["mtp_all_tokens_matched"]
        and r["serial_all_tokens_matched"]
        and r["mtp_residual_divergence"] == 0
        and r["mtp_parity_all_ok"]
        and r["local_iterate_passed"]
        and r["uses_pinned_mtp_head"]
        and r["mtp_emitted"] == 512
        and r["exit"] == "0"
    )


def probe_histograms(runs: Path) -> dict:
    """Realised depth histogram per prompt, from the traced (untimed) passes."""
    out = {}
    for arm in ARMS:
        per_prompt, pooled = {}, {}
        for p in PROMPTS:
            t = runs / f"probe-{p}-{arm}" / "trace.txt"
            if not t.exists():
                continue
            rounds = parse_trace(t)
            h = {}
            for r in rounds:
                h[r.depth] = h.get(r.depth, 0) + 1
            per_prompt[p] = {
                "histogram": dict(sorted(h.items())),
                "rounds": len(rounds),
                "rows": sum(r.depth for r in rounds),
                "mean_depth": (sum(r.depth for r in rounds) / len(rounds)) if rounds else 0.0,
            }
            for k, v in h.items():
                pooled[k] = pooled.get(k, 0) + v
        if per_prompt:
            rounds = sum(v["rounds"] for v in per_prompt.values())
            rows = sum(v["rows"] for v in per_prompt.values())
            out[arm] = {
                "per_prompt": per_prompt,
                "pooled_histogram": dict(sorted(pooled.items())),
                "pooled_rounds": rounds,
                "pooled_rows": rows,
                "pooled_mean_depth": rows / rounds if rounds else 0.0,
                "depth_ge_4_rounds": sum(v for k, v in pooled.items() if k >= 4),
            }
    return out


def reduce_runs(runs_root: Path) -> dict:
    timed = {a: {} for a in ARMS}
    for p in PROMPTS:
        for a in ARMS:
            r = load_run(runs_root, f"{p}-{a}")
            if r is not None:
                timed[a][p] = r

    paired = [p for p in PROMPTS if p in timed["BASE"] and p in timed["PRICE"]]
    per_prompt = {}
    for p in paired:
        b, c = timed["BASE"][p], timed["PRICE"][p]
        bt, ct = b["mtp_true_decode"], c["mtp_true_decode"]
        # The serial legs are byte-identical work under two different builds, so
        # their spread bounds the drift any arm effect has to clear.
        s_b, s_c = b["serial_true_decode"], c["serial_true_decode"]
        per_prompt[p] = {
            "base_true_decode_s": bt,
            "price_true_decode_s": ct,
            "true_decode_gain_pct": (bt - ct) / bt * 100.0,
            "base_leg_s": b["mtp_leg_seconds"],
            "price_leg_s": c["mtp_leg_seconds"],
            "leg_gain_pct": (b["mtp_leg_seconds"] - c["mtp_leg_seconds"]) / b["mtp_leg_seconds"] * 100.0,
            "base_serial_true_decode_s": s_b,
            "price_serial_true_decode_s": s_c,
            "serial_spread_pct": abs(s_b - s_c) / ((s_b + s_c) / 2) * 100.0,
            "base_local_speedup": b["local_iterate_speedup"],
            "price_local_speedup": c["local_iterate_speedup"],
            "base_mean_depth": b["mtp_mean_depth"],
            "price_mean_depth": c["mtp_mean_depth"],
            "base_rounds": b["mtp_rounds"],
            "price_rounds": c["mtp_rounds"],
            "base_declared_rows": b["mtp_declared_rows"],
            "price_declared_rows": c["mtp_declared_rows"],
            "base_accepted_rate": b["mtp_accepted_rate"],
            "price_accepted_rate": c["mtp_accepted_rate"],
            "base_max_depth": b["mtp_max_depth"],
            "price_max_depth": c["mtp_max_depth"],
            "base_depth_histogram": b["depth_histogram_from_report"],
            "price_depth_histogram": c["depth_histogram_from_report"],
            "base_correct": correctness_ok(b),
            "price_correct": correctness_ok(c),
            "order": "BASE_first" if PROMPTS.index(p) % 2 == 0 else "PRICE_first",
        }

    gains = [per_prompt[p]["true_decode_gain_pct"] for p in paired]
    sums = {
        a: sum(timed[a][p]["mtp_true_decode"] for p in paired) for a in ARMS
    }
    pooled_gain = (sums["BASE"] - sums["PRICE"]) / sums["BASE"] * 100.0 if paired else 0.0

    pooled_hist = {a: {} for a in ARMS}
    for a in ARMS:
        for p in paired:
            for k, v in timed[a][p]["depth_histogram_from_report"].items():
                pooled_hist[a][int(k)] = pooled_hist[a].get(int(k), 0) + v

    serial_spreads = [per_prompt[p]["serial_spread_pct"] for p in paired]
    return {
        "prompts_paired": paired,
        "prompts_missing": [p for p in PROMPTS if p not in paired],
        "per_prompt": per_prompt,
        "headline": {
            "metric": "e25/mtp_true_decode_gain_pct_median_of_8",
            "median_of_8": statistics.median(gains) if gains else None,
            "n_prompts": len(gains),
            "pooled_gain_pct": pooled_gain,
            "mean_gain_pct": statistics.fmean(gains) if gains else None,
            "min_gain_pct": min(gains) if gains else None,
            "max_gain_pct": max(gains) if gains else None,
            "all_prompts_positive": all(g > 0 for g in gains) if gains else None,
            "base_true_decode_total_s": sums["BASE"],
            "price_true_decode_total_s": sums["PRICE"],
        },
        "noise_floor": {
            "serial_spread_pct_max": max(serial_spreads) if serial_spreads else None,
            "serial_spread_pct_mean": statistics.fmean(serial_spreads) if serial_spreads else None,
            # The advisor's stop rule: the within-arm control spread must not
            # swamp the between-arm effect.
            "effect_clears_serial_spread": (
                (statistics.median(gains) > max(serial_spreads)) if gains and serial_spreads else None
            ),
        },
        "correctness": {
            "all_pass": all(
                per_prompt[p]["base_correct"] and per_prompt[p]["price_correct"] for p in paired
            ) if paired else None,
            "failures": [
                p for p in paired
                if not (per_prompt[p]["base_correct"] and per_prompt[p]["price_correct"])
            ],
        },
        "pooled_depth_histogram_timed": {a: dict(sorted(pooled_hist[a].items())) for a in ARMS},
        "timed_runs": timed,
    }


def compare_prereg(reduced: dict, probes: dict) -> dict:
    hl = reduced["headline"]
    # Prefer the traced probe histogram when present, else the timed reports.
    if "PRICE" in probes:
        realised = probes["PRICE"]["pooled_histogram"]
        ge4 = probes["PRICE"]["depth_ge_4_rounds"]
        source = "traced_probe"
    else:
        realised = reduced["pooled_depth_histogram_timed"].get("PRICE", {})
        ge4 = sum(v for k, v in realised.items() if int(k) >= 4)
        source = "timed_report_effective_draft_lengths"
    pred = PREREG["depth_histogram"]
    keys = sorted({int(k) for k in realised} | set(pred))
    out = {
        "histogram_source": source,
        "predicted": {str(k): pred.get(k, 0) for k in keys},
        "realised": {str(k): int(realised.get(k, realised.get(str(k), 0))) for k in keys},
        "depth_ge_4_predicted": PREREG["depth_ge_4_rounds"],
        "depth_ge_4_realised": ge4,
        "unreachability_holds_at_runtime": ge4 == 0,
        "pooled_gain_predicted_pct": PREREG["pooled_true_decode_gain_pct"],
        "pooled_gain_realised_pct": hl["pooled_gain_pct"],
        "median_of_8_predicted_pct": PREREG["median_of_8_true_decode_gain_pct"],
        "median_of_8_realised_pct": hl["median_of_8"],
    }
    if hl["median_of_8"] is not None:
        out["median_of_8_error_pp"] = hl["median_of_8"] - PREREG["median_of_8_true_decode_gain_pct"]
        out["pooled_error_pp"] = hl["pooled_gain_pct"] - PREREG["pooled_true_decode_gain_pct"]
    per = {}
    for p, v in reduced["per_prompt"].items():
        if p in PREREG["per_prompt_gain_pct"]:
            per[p] = {
                "predicted_pct": PREREG["per_prompt_gain_pct"][p],
                "realised_pct": v["true_decode_gain_pct"],
                "error_pp": v["true_decode_gain_pct"] - PREREG["per_prompt_gain_pct"][p],
            }
    out["per_prompt"] = per
    return out


def log_wandb(payload: dict) -> list[dict]:
    import wandb

    runs = []
    group = "qwen38-r1-e25-per-row-draft-price"
    hl = payload["reduced"]["headline"]
    pre = payload["prereg_comparison"]
    reduced = payload["reduced"]

    cfg_common = {
        "assignment_id": "qwen38-r1-e25-per-row-draft-price",
        "revision_id": "r1",
        "pr": 29,
        "credit": "thorfinn E22 follow-up #1",
        "base_sha": "0d2eef9cac75d890de06a5eef4fd686c3c34c1ef",
        "result_commit": payload["provenance"]["head_sha"],
        "host": payload["provenance"]["host"],
        "tokens": 512,
        "mode": "local-iterate",
        "prompts": PROMPTS,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "cool_gate_temp_c": 40,
        "cool_gate_bypass_reason": "host idles above the compile-time 40C gate",
        "measured_row_step_ratio": [0.0, 0.095904, 0.152261, 0.442442],
        "head_step_cost_ratio": 0.18,
        "prereg": PREREG,
    }

    r = wandb.init(
        project="qwen38-mlx-challenge-senpai",
        entity="wandb-applied-ai-team",
        group=group,
        job_type="phase1-contrast",
        name="e25-phase1-price-vs-base",
        config=cfg_common,
        reinit=True,
    )
    summary = {
        "e25/mtp_true_decode_gain_pct_median_of_8": hl["median_of_8"],
        "e25/mtp_true_decode_gain_pct_pooled": hl["pooled_gain_pct"],
        "e25/mtp_true_decode_gain_pct_mean": hl["mean_gain_pct"],
        "e25/mtp_true_decode_gain_pct_min": hl["min_gain_pct"],
        "e25/mtp_true_decode_gain_pct_max": hl["max_gain_pct"],
        "e25/all_prompts_positive": hl["all_prompts_positive"],
        "e25/n_prompts": hl["n_prompts"],
        "e25/base_true_decode_total_s": hl["base_true_decode_total_s"],
        "e25/price_true_decode_total_s": hl["price_true_decode_total_s"],
        "e25/serial_spread_pct_max": reduced["noise_floor"]["serial_spread_pct_max"],
        "e25/serial_spread_pct_mean": reduced["noise_floor"]["serial_spread_pct_mean"],
        "e25/effect_clears_serial_spread": reduced["noise_floor"]["effect_clears_serial_spread"],
        "e25/correctness_all_pass": reduced["correctness"]["all_pass"],
        "e25/depth_ge_4_realised": pre["depth_ge_4_realised"],
        "e25/unreachability_holds_at_runtime": pre["unreachability_holds_at_runtime"],
        "e25/median_of_8_predicted_pct": pre["median_of_8_predicted_pct"],
        "e25/median_of_8_error_pp": pre.get("median_of_8_error_pp"),
        "e25/pooled_predicted_pct": pre["pooled_gain_predicted_pct"],
        "e25/pooled_error_pp": pre.get("pooled_error_pp"),
        "e25/histogram_source": pre["histogram_source"],
    }
    for p, v in reduced["per_prompt"].items():
        summary[f"e25/gain_pct/{p}"] = v["true_decode_gain_pct"]
        summary[f"e25/base_true_decode_s/{p}"] = v["base_true_decode_s"]
        summary[f"e25/price_true_decode_s/{p}"] = v["price_true_decode_s"]
        summary[f"e25/base_mean_depth/{p}"] = v["base_mean_depth"]
        summary[f"e25/price_mean_depth/{p}"] = v["price_mean_depth"]
        summary[f"e25/base_accepted_rate/{p}"] = v["base_accepted_rate"]
        summary[f"e25/price_accepted_rate/{p}"] = v["price_accepted_rate"]
        summary[f"e25/base_declared_rows/{p}"] = v["base_declared_rows"]
        summary[f"e25/price_declared_rows/{p}"] = v["price_declared_rows"]
        summary[f"e25/base_rounds/{p}"] = v["base_rounds"]
        summary[f"e25/price_rounds/{p}"] = v["price_rounds"]
        summary[f"e25/serial_spread_pct/{p}"] = v["serial_spread_pct"]
        summary[f"e25/base_local_speedup/{p}"] = v["base_local_speedup"]
        summary[f"e25/price_local_speedup/{p}"] = v["price_local_speedup"]
    for k, v in (payload.get("phase0") or {}).get("wandb_scalars", {}).items():
        summary[k] = v
    r.summary.update({k: v for k, v in summary.items() if v is not None})

    # A per-prompt table makes the prediction/realisation comparison inspectable
    # in the UI rather than only in the artifact.
    cols = ["prompt", "order", "predicted_gain_pct", "realised_gain_pct", "error_pp",
            "base_true_decode_s", "price_true_decode_s", "base_mean_depth",
            "price_mean_depth", "serial_spread_pct", "base_correct", "price_correct"]
    tbl = wandb.Table(columns=cols)
    for p in reduced["prompts_paired"]:
        v = reduced["per_prompt"][p]
        pp = pre["per_prompt"].get(p, {})
        tbl.add_data(
            p, v["order"], pp.get("predicted_pct"), v["true_decode_gain_pct"],
            pp.get("error_pp"), v["base_true_decode_s"], v["price_true_decode_s"],
            v["base_mean_depth"], v["price_mean_depth"], v["serial_spread_pct"],
            v["base_correct"], v["price_correct"],
        )
    r.log({"e25/per_prompt": tbl})

    hist_tbl = wandb.Table(columns=["depth", "predicted_rounds", "realised_rounds"])
    for k in sorted(pre["predicted"], key=int):
        hist_tbl.add_data(int(k), pre["predicted"][k], pre["realised"].get(k, 0))
    r.log({"e25/depth_histogram": hist_tbl})

    art = wandb.Artifact("e25-phase1", type="analysis")
    with art.new_file("e25-phase1.json") as f:
        json.dump(payload, f, indent=1, default=str)
    if payload["provenance"].get("phase0_path"):
        art.add_file(payload["provenance"]["phase0_path"], name="e25-phase0.json")
    r.log_artifact(art)

    runs.append({"run_id": r.id, "url": r.url, "name": r.name})
    r.finish()
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".mlxfast-private/e25/runs")
    ap.add_argument("--out", default="research/e25-phase1.json")
    ap.add_argument("--phase0", default="research/e25-phase0.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    if not runs_root.is_dir():
        print(f"e25_phase1: no runs at {runs_root}", file=sys.stderr)
        return 2

    reduced = reduce_runs(runs_root)
    probes = probe_histograms(runs_root)
    pre = compare_prereg(reduced, probes)

    phase0 = {}
    p0 = Path(args.phase0)
    if p0.exists():
        d = json.loads(p0.read_text())
        phase0 = {
            "instrument_gate": d.get("instrument_gate"),
            "ledger": d.get("ledger"),
            "round_time_table": d.get("round_time_table"),
            "unreachability_depth3": d.get("unreachability_depth3"),
            "binding_constraint": d.get("binding_constraint"),
            "wandb_scalars": {
                "e25/phase0_h_prefill_free_both_sides": 0.131697,
                "e25/phase0_h_leg_anchored": 0.166344,
                "e25/phase0_h_traced_over_leg_anchor": 0.069730,
                "e25/phase0_step_ratio_d3": 0.442442,
                "e25/phase0_per_row_0p18x8_bit_identical": 1,
                "e25/phase0_tape_rounds": 1947,
                "e25/phase0_tokens_lost_arm_D": 98,
                "e25/phase0_rows_saved_arm_D": 609,
            },
        }

    payload = {
        "experiment": "qwen38-r1-e25-per-row-draft-price",
        "credit": "thorfinn's E22 follow-up #1 (arm C is his design; arm D is the "
                  "no-deepen truncation of it that was pre-registered)",
        "provenance": {
            "head_sha": os.popen("git rev-parse HEAD").read().strip(),
            "base_sha": "0d2eef9cac75d890de06a5eef4fd686c3c34c1ef",
            "host": os.uname().nodename,
            "runs_root": str(runs_root),
            "phase0_path": str(p0) if p0.exists() else None,
        },
        "prereg": PREREG,
        "reduced": reduced,
        "probe_histograms": probes,
        "prereg_comparison": pre,
        "phase0": phase0,
    }

    Path(args.out).write_text(json.dumps(payload, indent=1, default=str))

    hl = reduced["headline"]
    print(f"prompts paired : {len(reduced['prompts_paired'])} "
          f"{reduced['prompts_paired']}")
    if reduced["prompts_missing"]:
        print(f"MISSING        : {reduced['prompts_missing']}")
    print(f"correctness    : all_pass={reduced['correctness']['all_pass']} "
          f"failures={reduced['correctness']['failures']}")
    print(f"MEDIAN OF 8    : {hl['median_of_8']}  (pre-registered "
          f"{PREREG['median_of_8_true_decode_gain_pct']})")
    print(f"pooled         : {hl['pooled_gain_pct']}  (pre-registered "
          f"{PREREG['pooled_true_decode_gain_pct']})")
    print(f"per-prompt     : min={hl['min_gain_pct']} max={hl['max_gain_pct']} "
          f"all_positive={hl['all_prompts_positive']}")
    print(f"serial spread  : max={reduced['noise_floor']['serial_spread_pct_max']} "
          f"clears={reduced['noise_floor']['effect_clears_serial_spread']}")
    print(f"depth>=4       : realised={pre['depth_ge_4_realised']} "
          f"(pre-registered 0) source={pre['histogram_source']}")
    print(f"histogram      : predicted={pre['predicted']}")
    print(f"                 realised ={pre['realised']}")
    for p in reduced["prompts_paired"]:
        v = reduced["per_prompt"][p]
        pp = pre["per_prompt"].get(p, {})
        print(f"  {p:<16} {v['true_decode_gain_pct']:+8.3f}%  "
              f"pred {pp.get('predicted_pct', float('nan')):+7.3f}%  "
              f"depth {v['base_mean_depth']:.3f}->{v['price_mean_depth']:.3f}  "
              f"serial spread {v['serial_spread_pct']:.3f}%  {v['order']}")

    if args.wandb:
        for r in log_wandb(payload):
            print(f"wandb: {r['run_id']} {r['url']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
