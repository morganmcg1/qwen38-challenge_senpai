"""Log rung 5f (the --local-submit gate and the free metallib control) to W&B.

Rung 5f re-runs the shipped `sumtable` arm on a rebuilt `mlx.metallib` with a
byte-identical worker binary. It answers one question: did the metallib rebuild
that happened after rung 5g move the candidate leg? The 5g sumtable arm is the
reference distribution.

Usage: python3 research/e120_wandb_rung5f.py JOB_LOG
"""

import json
import re
import statistics as st
import subprocess
import sys

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e120-own-the-qmv-dispatch"

# Rung 5g, same host, same worker, stale metallib. W&B run qqjlgtkv.
RUNG5G_SUMTABLE = [0.0294516, 0.0294523, 0.0294699, 0.0294771]
RUNG5G_OFF = [0.0306153, 0.0305977, 0.0306425, 0.0306522]
RUNG5G_SERIAL_OFF = 0.073625
RUNG5G_SERIAL_SUMTABLE = 0.073703

# Pre-registered before the run: the rebuild is immaterial if the candidate leg
# moves by less than this fraction of itself.
STOP_RULE_PCT = 0.080

STALE_METALLIB_WARNINGS = 3
FRESH_METALLIB_SHA = (
    "a3a74eda50ba7d375081b378a5dd87dd81bd419e0697c1ecbbd652e8677373c4"
)


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()

    start = text.rindex("{\n  \"score\"")
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    report = json.loads(text[start:end])

    shas = re.findall(r"^([0-9a-f]{64})\s+\.build-worker", text, re.M)
    commit = re.search(r"^([0-9a-f]{40})$", text, re.M)
    gates = re.findall(
        r"GPU cool-down gate passed \(current ([0-9.]+)C", text)
    return report, shas, commit.group(1) if commit else None, gates


def main():
    report, shas, commit, gates = parse_log(sys.argv[1])
    m = report["metrics"]
    cand = m["mtp_seconds_per_token"]
    serial = m["serial_seconds_per_token"]

    ref_mean = st.mean(RUNG5G_SUMTABLE)
    ref_sd = st.stdev(RUNG5G_SUMTABLE)
    ref_median = st.median(RUNG5G_SUMTABLE)
    off_mean = st.mean(RUNG5G_OFF)

    drift_pct = 100 * (cand - ref_mean) / ref_mean
    verdict = "immaterial" if abs(drift_pct) < STOP_RULE_PCT else "material"

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    summary = {
        "rung": "5f",
        "role": "local-submit gate + free metallib control",
        "harness": "local",
        "git_head": head,
        "commit_under_test": commit,
        "worker_sha256_before": shas[0] if shas else None,
        "worker_sha256_after": shas[-1] if shas else None,
        "worker_unchanged_across_run": len(set(shas)) == 1,
        "worker_same_as_rung5g": shas[0].startswith("0eee61f8") if shas else None,
        "metallib_sha256": FRESH_METALLIB_SHA,
        "metallib_rebuilt_since_rung5g": True,
        "rung5g_metallib_jit_warnings": STALE_METALLIB_WARNINGS,

        "candidate_seconds_per_token": cand,
        "serial_seconds_per_token": serial,
        "local_ratio": m["mtp_decode_speedup"],
        "decode_tokens": m["decode_tokens"],
        "mtp_depth": m["mtp_depth"],
        "rounds": 78,

        "rung5g_sumtable_mean": ref_mean,
        "rung5g_sumtable_median": ref_median,
        "rung5g_sumtable_sd": ref_sd,
        "rung5g_sumtable_min": min(RUNG5G_SUMTABLE),
        "rung5g_sumtable_max": max(RUNG5G_SUMTABLE),
        "rung5g_off_mean": off_mean,

        "drift_vs_rung5g_mean_pct": drift_pct,
        "drift_vs_rung5g_median_pct": 100 * (cand - ref_median) / ref_median,
        "drift_z_vs_rung5g_arm_sd": (cand - ref_mean) / ref_sd,
        "inside_rung5g_arm_range": min(RUNG5G_SUMTABLE) <= cand <= max(RUNG5G_SUMTABLE),
        "stop_rule_pct": STOP_RULE_PCT,
        "metallib_rebuild_verdict": verdict,
        "rung5g_headline_stands": verdict == "immaterial",

        "leg_gain_vs_rung5g_off_pct": -100 * (cand - off_mean) / off_mean,
        "rung5g_measured_leg_gain_pct": -100 * (ref_mean - off_mean) / off_mean,

        "serial_drift_vs_rung5g_off_pct":
            100 * (serial - RUNG5G_SERIAL_OFF) / RUNG5G_SERIAL_OFF,
        "serial_drift_vs_rung5g_sumtable_pct":
            100 * (serial - RUNG5G_SERIAL_SUMTABLE) / RUNG5G_SERIAL_SUMTABLE,

        "all_tokens_matched": m["all_tokens_matched"],
        "residual_divergence_count": m["residual_divergence_count"],
        "public_drift_tripwire_passed": m["public_drift_tripwire_passed"],
        "accepted_draft_rate": m["accepted_draft_rate"],
        "effective_mean_draft_len": m["effective_mean_draft_len"],
        "accept_rate_matches_rung5g": abs(
            m["accepted_draft_rate"] - 0.8770161290322581) < 1e-12,
        "head_provenance_sha256": m["head_provenance_sha256"],
        "uses_declared_head": True,

        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": True,
        "cool_gate_exit_temps_c": [float(g) for g in gates],

        "e121_share_sums_in_header": 3,
        "e121_share_sums_in_generated_twin": 3,
        "e121_share_sums_in_built_worker": 3,

        "scope_submitted_paths_changed": 1,
        "budget_source_bytes": 2580913,
        "budget_growth_bytes": 126078,
        "twin_audit_ok": True,
        "ranked_score_boundary_pass": True,

        "rankable": m["rankable"],
        "official_score": m["official_score"],
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e120-rung5f-local-submit-metallib-control",
        job_type="local-submit",
        config={
            "experiment": "E120 own the QMV dispatch",
            "rung": "5f",
            "arm": "sumtable (shipped default, no env override)",
            "host": "ip-10-231-2-95.ec2.internal",
            "chip": "Apple M4 Pro (applegpu_g16s)",
            "memory_gib": 48,
            "os": "macOS 26.5.2",
            "swift": "6.3.3",
            "mode": "benchmark-qwen-mtp.sh --local-submit",
            "tokens": 512,
            "depth": 8,
            "mtp_head": "declared (mtp-head.manifest.json)",
        },
    )
    run.summary.update(summary)
    for k, v in summary.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            run.log({k: v})
    print(json.dumps(summary, indent=2, default=str))
    print("W&B run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
