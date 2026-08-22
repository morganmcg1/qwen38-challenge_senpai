#!/usr/bin/env python3
"""Analyse an E136 rung-2 ABBA session.

    usage: research/e136_analyse.py [--label r2] [--json OUT.json]

Reads every `research/out/e136<label>p*` leg and reports:

1. The headline. Absolute candidate MTP seconds per token, which is the only
   quantity the ranked numerator responds to. Arm C1 changes work that runs
   ONLY inside the candidate MTP leg, so the local serial-to-MTP ratio is a
   valid second readout here rather than a cancelling one. Both are printed
   and the absolute number leads.

2. The mechanism. Draft depth, accepted tokens, realised acceptance and the
   per-round latency split, so a change in absolute time can be attributed to
   the shortlist rather than to a different round mix.

3. The exactness fields the arm must not move: `all_tokens_matched` and
   `residual_divergence_count` on the accepted stream.

The arm witness comes from meta.txt, which `research/e136_abba.sh` fills from
the trace: a C1 leg must show a positive `e136_c1_draft_steps` with
`e136_shipped_selection_draft_steps` at zero, and a base leg the reverse.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND_RE = re.compile(
    r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?"
    r"draft_build_us=(\d+) .*?round_us=(\d+)")
BEGIN_RE = re.compile(r"^mtp-trace: begin seed=(\d+)")

# One added microsecond per draft step is this many ranked percent, from the
# E136 rung-0 conversion: 174.1 us per draft step per ranked percent.
US_PER_DRAFT_STEP_PER_PCT = 174.1


def read_meta(path):
    meta = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.strip().partition("=")
                meta[key] = value
    return meta


def read_trace(path):
    blocks = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            begin = BEGIN_RE.match(line)
            if begin:
                current = {"seed": int(begin.group(1)), "rounds": []}
                blocks.append(current)
                continue
            match = ROUND_RE.match(line)
            if match and current is not None:
                current["rounds"].append({
                    "round": int(match.group(1)),
                    "d": int(match.group(2)),
                    "acc": int(match.group(3)),
                    "draft_build_us": int(match.group(4)),
                    "round_us": int(match.group(5)),
                })
    return blocks


def pick_mtp_block(blocks):
    """The last block that actually drafted."""
    chosen = None
    for block in blocks:
        if block["rounds"] and any(e["d"] > 0 for e in block["rounds"]):
            chosen = block
    return chosen


def load_leg(directory):
    meta_path = os.path.join(directory, "meta.txt")
    score_path = os.path.join(directory, "score.json")
    if not os.path.exists(meta_path) or not os.path.exists(score_path):
        return None
    meta = read_meta(meta_path)
    with open(score_path, encoding="utf-8") as handle:
        score = json.load(handle)
    metrics = score.get("metrics", {})
    leg = {
        "tag": os.path.basename(directory),
        "arm": meta.get("e136_arm"),
        "position": int(meta.get("e136_position", 0)),
        "flag": meta.get("MLX_E136_C1_SKETCH"),
        "c1_draft_steps": int(meta.get("e136_c1_draft_steps", 0)),
        "shipped_selection_draft_steps":
            int(meta.get("e136_shipped_selection_draft_steps", 0)),
        "status": meta.get("status"),
        "tokens": int(meta.get("tokens", 0)),
        "entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
        "exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
        "worker_sha256": meta.get("worker_sha256"),
        "worker_sha256_after_leg": meta.get("worker_sha256_after_leg"),
        "session_commit": meta.get("session_commit"),
        "base_sha": meta.get("base_sha"),
        "dirty_candidate_paths": meta.get("dirty_candidate_paths"),
        "cool_gate_passed_real_gate":
            meta.get("cool_gate_passed_real_gate") == "true",
        "gate_qualified_for_timing":
            meta.get("gate_qualified_for_timing") == "true",
        "mtp_s_per_tok": metrics.get("mtp_seconds_per_token"),
        "serial_s_per_tok": metrics.get("serial_seconds_per_token"),
        "speedup": metrics.get("mtp_decode_speedup"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "mean_draft_len": metrics.get("effective_mean_draft_len"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "decode_tokens": metrics.get("decode_tokens"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
    }
    leg["arm_witnessed"] = (
        (leg["c1_draft_steps"] > 0 and leg["shipped_selection_draft_steps"] == 0)
        if leg["flag"] == "1"
        else (leg["c1_draft_steps"] == 0
              and leg["shipped_selection_draft_steps"] > 0))

    trace_path = os.path.join(directory, "trace.txt")
    if os.path.exists(trace_path):
        block = pick_mtp_block(read_trace(trace_path))
        if block is not None:
            rounds = block["rounds"]
            leg["rounds"] = len(rounds)
            leg["median_round_us"] = statistics.median(
                e["round_us"] for e in rounds)
            leg["median_draft_build_us"] = statistics.median(
                e["draft_build_us"] for e in rounds)
            leg["mean_d"] = statistics.mean(e["d"] for e in rounds)
            leg["mean_acc"] = statistics.mean(e["acc"] for e in rounds)
            # Realised acceptance: accepted drafts over drafts proposed. This
            # is the rule-107 quantity, not one minus a shortlist miss rate.
            proposed = sum(e["d"] for e in rounds)
            leg["realised_acceptance"] = (
                sum(e["acc"] for e in rounds) / proposed if proposed else None)
            leg["draft_steps"] = proposed + len(rounds)
    return leg


def summarise(legs, key):
    out = {}
    for arm in ("off", "on"):
        values = [leg[key] for leg in legs
                  if leg["arm"] == arm and leg.get(key) is not None]
        if not values:
            continue
        out[arm] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    for arm in out:
        block = out[arm]
        block["se"] = block["sd"] / math.sqrt(block["n"]) if block["n"] else 0.0
    if "off" in out and "on" in out:
        base = out["off"]["mean"]
        cand = out["on"]["mean"]
        out["delta"] = cand - base
        out["delta_pct"] = 100.0 * (cand - base) / base if base else None
        # Two independent arm means, so the variances add. The ABBA order
        # counterbalances drift in the MEAN; it does not pair the legs, so a
        # paired statistic would overstate the precision.
        se = math.hypot(out["off"]["se"], out["on"]["se"])
        out["se_delta"] = se
        out["se_delta_pct"] = 100.0 * se / base if base else None
        out["two_sigma_pct"] = 2.0 * out["se_delta_pct"] if base else None
    return out


def session_identity(legs):
    """Assert the fields that must not move across a comparable session.

    The stale-worker defect of 2026-08-22 got past a whole session because the
    only thing checked was the source tree. `worker_sha256` is the fingerprint
    of the binary that actually decoded the tokens, and it is the one field
    that can witness a `Qwen35.swift` edit: `.build/release/mlxfast-swift`
    carries no model code and does not relink for one.

    `base_sha` is allowed to move when the session spans a research-tooling
    commit, because `research/*.py` is not linked into the worker. That is
    reported and must be justified by a diff, never assumed.
    """
    def distinct(key):
        return sorted({leg.get(key) for leg in legs if leg.get(key)})

    workers = distinct("worker_sha256")
    after = distinct("worker_sha256_after_leg")
    commits = distinct("session_commit")
    bases = distinct("base_sha")
    dirty = sorted({leg.get("dirty_candidate_paths") for leg in legs})

    out = {
        "worker_sha256": workers,
        "worker_sha256_after_leg": after,
        "session_commit": commits,
        "base_sha": bases,
        "dirty_candidate_paths": dirty,
        "worker_uniform": len(workers) == 1 and len(set(workers + after)) == 1,
        "session_commit_uniform": len(commits) == 1,
        "base_sha_uniform": len(bases) == 1,
        "no_dirty_candidate_paths": dirty == ["0"],
    }
    notes = []
    notes.append(
        f"worker_sha256 uniform across legs and unchanged by every leg: "
        f"{out['worker_uniform']} ({workers[0][:16] if workers else 'none'}...)")
    notes.append(f"session_commit uniform: {out['session_commit_uniform']} "
                 f"({commits[0][:12] if commits else 'none'})")
    notes.append(f"dirty candidate paths on every leg: 0 -> "
                 f"{out['no_dirty_candidate_paths']}")
    if out["base_sha_uniform"]:
        notes.append(f"base_sha uniform: True ({bases[0][:12]})")
    else:
        notes.append(
            f"base_sha MOVED mid-session across {len(bases)} commits "
            f"({', '.join(b[:12] for b in bases)}). This is only acceptable if "
            f"the intervening commits touch no candidate path; the identical "
            f"worker_sha256 above is the evidence that the timed binary did "
            f"not change. Verify with: git diff --name-only "
            f"{bases[0][:12]}..{bases[-1][:12]}")
    out["notes"] = notes
    # The timed binary changing mid-session invalidates every comparison, so
    # this one is fatal rather than reported.
    assert out["worker_uniform"], (
        f"the timed worker binary changed during the session: {workers} / {after}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="r2")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    pattern = os.path.join(ROOT, "research", "out", f"e136{args.label}p*")
    legs = [leg for leg in (load_leg(d) for d in sorted(glob.glob(pattern)))
            if leg]
    if not legs:
        print(f"no legs matched {pattern}", file=sys.stderr)
        return 1
    legs.sort(key=lambda leg: leg["position"])

    print(f"{'tag':<14}{'arm':<5}{'pos':<4}{'wit':<5}{'mtp s/tok':>12}"
          f"{'serial s/tok':>14}{'ratio':>8}{'d':>6}{'acc':>6}"
          f"{'in C':>7}{'out C':>7}{'match':>7}{'div':>5}")
    for leg in legs:
        print(f"{leg['tag']:<14}{str(leg['arm']):<5}{leg['position']:<4}"
              f"{'y' if leg['arm_witnessed'] else 'NO':<5}"
              f"{leg['mtp_s_per_tok']:>12.6f}{leg['serial_s_per_tok']:>14.6f}"
              f"{leg['speedup']:>8.4f}{leg.get('mean_d', 0):>6.2f}"
              f"{leg.get('mean_acc', 0):>6.2f}"
              f"{leg['entry_c']:>7.1f}{leg['exit_c']:>7.1f}"
              f"{str(leg['all_tokens_matched']):>7}"
              f"{str(leg['residual_divergence_count']):>5}")

    keys = ("mtp_s_per_tok", "serial_s_per_tok", "speedup", "mean_draft_len",
            "realised_acceptance", "median_round_us", "median_draft_build_us",
            "entry_c")
    identity = session_identity(legs)
    print("\n-- session identity --")
    for line in identity["notes"]:
        print(f"  {line}")
    report = {
        "legs": legs,
        "label": args.label,
        "identity": identity,
        "all_arms_witnessed": all(leg["arm_witnessed"] for leg in legs),
        "all_tokens_matched": all(leg["all_tokens_matched"] for leg in legs),
        "accepted_stream_divergences": sum(
            leg["residual_divergence_count"] or 0 for leg in legs),
        "entry_temp_spread_c": (
            max(leg["entry_c"] for leg in legs)
            - min(leg["entry_c"] for leg in legs)),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }
    for key in keys:
        report[key] = summarise(legs, key)

    print("\n-- arm contrast (`on` is the C1 sketch shortlist) --")
    for key in keys:
        block = report[key]
        if "delta_pct" not in block:
            continue
        print(f"{key:<26} off={block['off']['mean']:.6f} "
              f"(sd {block['off']['sd']:.6f}, n {block['off']['n']})  "
              f"on={block['on']['mean']:.6f} "
              f"(sd {block['on']['sd']:.6f}, n {block['on']['n']})  "
              f"delta={block['delta_pct']:+.4f} %")

    # The headline is stated on the absolute candidate leg, sign flipped so a
    # faster candidate reads positive.
    mtp = report["mtp_s_per_tok"]
    if "delta_pct" in mtp:
        report["e136_c1_candidate_leg_pct"] = -mtp["delta_pct"]
        acc = report["realised_acceptance"]
        if "delta" in acc:
            report["e136_realised_acceptance_delta_pp"] = 100.0 * acc["delta"]
        headline = report["e136_c1_candidate_leg_pct"]
        band = mtp.get("two_sigma_pct") or 0.0
        report["e136_c1_candidate_leg_pct_two_sigma"] = band
        print(f"\ne136_c1_candidate_leg_pct = {headline:+.4f} "
              f"+/- {band:.4f} (2 sigma) "
              f"(positive means the C1 candidate leg is faster)")
        # F4 section 8 states the advance bar as +0.6 %; the PR body section D
        # states it as +0.30 %. The contradiction is unresolved, so the verdict
        # is reported against both and the experiment advances on neither
        # without a ruling.
        report["stop_rule_verdict"] = {
            key: {"bar": bar, "action": action,
                  "point_clears": headline >= bar,
                  "lower_2sigma_clears": headline - band >= bar}
            for key, bar, action in (
                ("advance_pr_body_0.30", 0.30, "advance if clears"),
                ("advance_f4_0.60", 0.60, "advance if clears"),
                ("close_below_0.25", 0.25, "CLOSE if it does NOT clear"))
        }
        print("\n-- stop rules, reported against both stated bars --")
        for key, verdict in report["stop_rule_verdict"].items():
            print(f"{key:<24} bar={verdict['bar']:.2f} "
                  f"clears={'yes' if verdict['point_clears'] else 'no':<4} "
                  f"lower2sigma={'yes' if verdict['lower_2sigma_clears'] else 'no':<4} "
                  f"[{verdict['action']}]")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
