#!/usr/bin/env python3
"""E145: read every collected leg and build the R1 and R2 tables.

`harness=local`. Zero GPU.

A leg is `.mlxfast-private/e128/e145/SLOT/FIXTURE/{meta.txt,report.json}`.
`meta.txt` carries the identity tuple and the gate evidence; `report.json`
carries the timing, the per-round block seconds and the per-round drafted
depth, from which the realised verify width is `depth + 1`.

Usage:
  python3 research/e145_read.py --json research/e145-artifacts/legs.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEGS = ROOT / ".mlxfast-private/e128/e145"

# The curve every depth-price decision descends from, as the E145 assignment
# quotes it: rows 1..9, `per_drafting_round`, microseconds per round. It is a
# RANKED-scale curve, rebuilt by inverting ranked receipts. It was never read
# off a decode, which is what E145 R2 exists to fix.
REPLAYED_RANKED_US = {
    1: 31173.0, 2: 34619.0, 3: 38065.0, 4: 41511.0, 5: 44958.0,
    6: 61199.0, 7: 62825.0, 8: 70315.0, 9: 75639.0,
}

# The uncorrected `slopeonly_b6` fit the four E134 item-2 forms all sit on.
PRE_ARM_RANKED_US = {
    1: 31171.46872642797, 2: 34617.54053327561, 3: 38063.612340123254,
    4: 41509.684146970896, 5: 44955.75595381854, 6: 59666.585107748295,
    7: 64990.11647244296, 8: 70313.64783713763, 9: 75637.1792018323,
}

# `AttentionUtils.swift:104-126`: the wide-decode exactness chunk splits a
# 6..9-row causal decode attention into two sdpa calls. Widths 1..5 take one.
FULL_ATTENTION_LAYERS = 16


def read_meta(path: pathlib.Path) -> dict:
    """`meta.txt` is append-only, so a repeated key means the LAST wins."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def sdpa_calls(width: int) -> int:
    return 2 if 6 <= width <= 9 else 1


def load_leg(slot_dir: pathlib.Path) -> dict | None:
    for fixture_dir in sorted(slot_dir.iterdir()):
        if not fixture_dir.is_dir():
            continue
        meta_path = fixture_dir / "meta.txt"
        report_path = fixture_dir / "report.json"
        if not meta_path.exists() or not report_path.exists():
            continue
        meta = read_meta(meta_path)
        report = json.loads(report_path.read_text())
        depths = report.get("effective_draft_lengths") or []
        blocks = report.get("block_request_seconds") or []
        widths = [d + 1 for d in depths]
        hist = collections.Counter(widths)
        total = max(1, len(widths))
        rounds = report["round_count"]
        # The first block carries first-call warm misses the later rounds do
        # not pay, so every per-round statistic here excludes it and says so.
        tail = blocks[1:] if len(blocks) > 1 else blocks
        return {
            "slot": slot_dir.name,
            "fixture": fixture_dir.name,
            "arm": meta.get("e145_arm_requested",
                            meta.get("depth_price_arm_requested", "unset")),
            "pin": meta.get("e145_pin_requested", "none"),
            "position": int(
                meta.get("e145_r1_position")
                or meta.get("e145_r2_position")
                or meta.get("e145_r6_position")
                or 0),
            "residency": meta.get("e145_r6_residency", "unset"),
            "wired_gate_gib": meta.get("e145_r6_gate_gib", "unset"),
            "probe_lines": int(meta.get("e145_r6_probe_lines", 0)),
            "probe_applied": int(meta.get("e145_r6_probe_applied", 0)),
            "probe_refused": int(meta.get("e145_r6_probe_refused", 0)),
            "tokens": int(meta.get("tokens", 0)),
            "leg_kind": meta.get("e145_leg_kind", "timed"),
            "real_cool_gate_taken":
                meta.get("e145_real_cool_gate_taken", "false") == "true",
            # `e128_session.sh` writes this literal unconditionally because it
            # never takes a gate itself. `e145_lib.sh` takes the real 40 C gate
            # through the unmodified `benchmark.sh` entry point before the leg,
            # so both are carried and neither is overwritten.
            "e128_literal_cool_gate_passed_real_gate":
                meta.get("cool_gate_passed_real_gate"),
            "cool_gate_line": meta.get("e145_cool_gate_line", ""),
            "gate_wait_s": float(meta.get("e145_gate_wait_s") or "nan"),
            "gate_entry_temp_c": float(
                meta.get("e145_gate_entry_temp_c") or "nan"),
            "leg_exit_temp_c": float(
                meta.get("e145_leg_exit_temp_c") or "nan"),
            "phase_trace": meta.get("phase_trace", "?"),
            "timing_valid": meta.get("timing_valid", "?"),
            # CAMPAIGN RULE 128. A timing difference between two arms is only
            # attributable to the arm if the warm phase left both processes in
            # the same residency and cache state.
            "warm_telemetry_present":
                meta.get("e145_warm_telemetry_present", "false") == "true",
            "warm": {k[len("e145_warm_"):]: v for k, v in meta.items()
                     if k.startswith("e145_warm_")
                     and not k.endswith("_line")
                     and k != "e145_warm_telemetry_present"},
            "commit": meta.get("e145_session_commit", meta.get("base_sha", "")),
            "worker_sha256": meta.get("worker_sha256", ""),
            "head_sha256": meta.get("head_manifest_tree_sha256", ""),
            "all_tokens_matched": report["all_tokens_matched"],
            "residual_divergence_count": report["residual_divergence_count"],
            # `decode_seconds == seed_prefill_seconds + sum(blocks)` and
            # `parent_measured_seconds_per_token == decode_seconds / tokens`,
            # verified on every leg by `basis_residual_s` below. The ranked leg
            # times seed processing and decoding together, so the seed-
            # inclusive figure is the ranked-equivalent basis. The seed prefill
            # is about 4.0 s and no depth policy can move it, so the block-only
            # basis is the sensitive one for a schedule arm.
            "spt": report["parent_measured_seconds_per_token"],
            "spt_blocks_only": sum(blocks) / report["decode_token_count"]
                               if blocks else None,
            "decode_seconds": report["decode_seconds"],
            "blocks_seconds": sum(blocks),
            "seed_prefill_seconds": report.get("seed_prefill_seconds"),
            "basis_residual_s": report["decode_seconds"] - sum(blocks)
                                - (report.get("seed_prefill_seconds") or 0.0),
            "round_count": rounds,
            "non_drafting_round_count": report.get("non_drafting_round_count"),
            "mean_draft_len": report["effective_mean_draft_len"],
            "max_draft_len": report["effective_max_draft_len"],
            "accepted_draft_rate": report["accepted_draft_rate"],
            "accepted_draft_total": report["accepted_draft_total"],
            "rejected_draft_total": report["rejected_draft_total"],
            "replayed_round_count":
                report.get("verify_block_replayed_round_count"),
            "declared_rows_total": report.get("declared_rows_total"),
            "is_serial_control": report.get("is_serial_control"),
            "mtp_depth": report.get("mtp_depth"),
            "block_us_median": 1e6 * statistics.median(tail) if tail else None,
            "block_us_mean": 1e6 * statistics.fmean(tail) if tail else None,
            "block_us_first": 1e6 * blocks[0] if blocks else None,
            "round_us_from_blocks": 1e6 * sum(blocks) / rounds
                                    if rounds else None,
            "round_us_from_decode": 1e6 * report["decode_seconds"] / rounds
                                    if rounds else None,
            "width_hist": {str(w): hist[w] / total for w in sorted(hist)},
            "width_counts": {str(w): hist[w] for w in sorted(hist)},
            "width_mass_ge6": sum(v for w, v in hist.items() if w >= 6) / total,
            "sdpa_calls_per_verify": {
                str(w): sdpa_calls(w) * FULL_ATTENTION_LAYERS
                for w in sorted(hist)},
            "blocks": blocks,
            "depths": depths,
        }
    return None


def collect() -> list[dict]:
    if not LEGS.exists():
        return []
    out = []
    for slot_dir in sorted(LEGS.iterdir()):
        if not slot_dir.is_dir():
            continue
        leg = load_leg(slot_dir)
        if leg is not None:
            out.append(leg)
    return out


def pct(a: float, b: float) -> float:
    """`a` against `b`, in percent. Positive means `a` is larger."""
    return 100.0 * (a - b) / b


def r1_tables(legs: list[dict], prefix: str = "r1-") -> dict:
    """Arm tables for one session family.

    `r1b-` is a second session of the same `beagle_a` arm comparison, so it
    gets exactly this treatment and is then compared with `r1-`. The prefixes
    carry their trailing hyphen on purpose: `r1-` must not swallow `r1b-`.
    """
    r1 = [leg for leg in legs if leg["slot"].startswith(prefix)]
    by_fixture: dict[str, dict] = {}
    for leg in r1:
        by_fixture.setdefault(leg["fixture"], {}).setdefault(
            leg["arm"], []).append(leg)

    out = {}
    for fixture, arms in sorted(by_fixture.items()):
        entry: dict = {"arms": {}}
        for arm, arm_legs in sorted(arms.items()):
            spts = [leg["spt"] for leg in arm_legs]
            entry["arms"][arm] = {
                "legs": len(arm_legs),
                "positions": [leg["position"] for leg in arm_legs],
                "spt_mean": statistics.fmean(spts),
                "spt_values": spts,
                "spt_sd": statistics.stdev(spts) if len(spts) > 1 else 0.0,
                "spt_blocks_only_mean": statistics.fmean(
                    [leg["spt_blocks_only"] for leg in arm_legs]),
                "seed_prefill_seconds": [leg["seed_prefill_seconds"]
                                         for leg in arm_legs],
                "round_counts": sorted({leg["round_count"]
                                        for leg in arm_legs}),
                "mean_draft_len": sorted({round(leg["mean_draft_len"], 10)
                                          for leg in arm_legs}),
                "accepted_draft_rate": sorted({
                    round(leg["accepted_draft_rate"], 10)
                    for leg in arm_legs}),
                "accepted_draft_total": sorted({leg["accepted_draft_total"]
                                                for leg in arm_legs}),
                "rejected_draft_total": sorted({leg["rejected_draft_total"]
                                                for leg in arm_legs}),
                "replayed_round_count": sorted({leg["replayed_round_count"]
                                                for leg in arm_legs}),
                "width_hist": arm_legs[0]["width_hist"],
                "width_mass_ge6": arm_legs[0]["width_mass_ge6"],
                "block_us_median": statistics.fmean(
                    [leg["block_us_median"] for leg in arm_legs]),
                "entry_temps": [leg["gate_entry_temp_c"] for leg in arm_legs],
                "exit_temps": [leg["leg_exit_temp_c"] for leg in arm_legs],
                "gate_waits_s": [leg["gate_wait_s"] for leg in arm_legs],
                "all_tokens_matched": all(leg["all_tokens_matched"]
                                          for leg in arm_legs),
                "residual_divergence_count": sorted({
                    leg["residual_divergence_count"] for leg in arm_legs}),
            }
        ship = entry["arms"].get("ship")
        pb6 = entry["arms"].get("pb6")
        serial = entry["arms"].get("serial")
        if ship and pb6:
            entry["pb6_vs_ship_pct"] = pct(pb6["spt_mean"], ship["spt_mean"])
            entry["pb6_vs_ship_blocks_only_pct"] = pct(
                pb6["spt_blocks_only_mean"], ship["spt_blocks_only_mean"])
            entry["width_mass_ge6_shift_pp"] = 100.0 * (
                ship["width_mass_ge6"] - pb6["width_mass_ge6"])
            # CAMPAIGN RULE 114 and RULE 101. Decoding is deterministic for a
            # fixed fixture, budget, build and arm, so the round count is an
            # exact behavioural signature of the arm that RAN. The witness is
            # only a witness if it can fail, so record whether the two arms'
            # signatures differ at all.
            entry["witness"] = {
                "ship_round_counts": ship["round_counts"],
                "pb6_round_counts": pb6["round_counts"],
                "ship_signature_stable": len(ship["round_counts"]) == 1,
                "pb6_signature_stable": len(pb6["round_counts"]) == 1,
                "arms_separated": (set(ship["round_counts"])
                                   != set(pb6["round_counts"])),
                "control_fails_against_other_arm": (
                    set(ship["round_counts"]) != set(pb6["round_counts"])),
            }
        if serial:
            for arm_name in ("ship", "pb6"):
                arm = entry["arms"].get(arm_name)
                if arm:
                    arm["local_serial_to_mtp_ratio"] = (
                        serial["spt_mean"] / arm["spt_mean"])
            if ship and pb6:
                entry["ratio_delta_pct"] = pct(
                    serial["spt_mean"] / pb6["spt_mean"],
                    serial["spt_mean"] / ship["spt_mean"])
        out[fixture] = entry
    return out


def r1b_replicate(r1: dict, r1b: dict) -> dict:
    """Does the R1 arm effect survive into a second session?

    CAMPAIGN RULE 119: the session is the unit that drifts, so an effect seen
    once in one session is not yet reproduced. Both sessions decode the same
    fixture with the same build, so the round count, mean draft length and
    acceptance rate must match exactly; only the timing may move. A mismatch
    in those signatures means the two sessions did not run the same work and
    the timing comparison is void.
    """
    out: dict = {}
    for fixture in sorted(set(r1) & set(r1b)):
        a, b = r1[fixture], r1b[fixture]
        if "pb6_vs_ship_pct" not in a or "pb6_vs_ship_pct" not in b:
            continue
        signatures_match = all(
            a["arms"][arm][field] == b["arms"][arm][field]
            for arm in ("ship", "pb6")
            for field in ("round_counts", "mean_draft_len",
                          "accepted_draft_rate")
            if arm in a["arms"] and arm in b["arms"])
        out[fixture] = {
            "r1_pct": a["pb6_vs_ship_pct"],
            "r1b_pct": b["pb6_vs_ship_pct"],
            "gap_pp": b["pb6_vs_ship_pct"] - a["pb6_vs_ship_pct"],
            "r1_blocks_pct": a["pb6_vs_ship_blocks_only_pct"],
            "r1b_blocks_pct": b["pb6_vs_ship_blocks_only_pct"],
            "blocks_gap_pp": (b["pb6_vs_ship_blocks_only_pct"]
                              - a["pb6_vs_ship_blocks_only_pct"]),
            "same_sign": ((a["pb6_vs_ship_pct"] < 0)
                          == (b["pb6_vs_ship_pct"] < 0)),
            "work_signatures_match": signatures_match,
            "r1_entry_temps": [t for arm in a["arms"].values()
                               for t in arm["entry_temps"]],
            "r1b_entry_temps": [t for arm in b["arms"].values()
                                for t in arm["entry_temps"]],
        }
    return out


def r2_tables(legs: list[dict]) -> dict:
    timed = [leg for leg in legs
             if leg["slot"].startswith("r2-") and leg["pin"] != "none"]
    traced = [leg for leg in legs if leg["slot"].startswith("r2trace-")]
    by_pin: dict[int, list[dict]] = {}
    for leg in timed:
        by_pin.setdefault(int(leg["pin"]), []).append(leg)

    widths = {}
    for pin, pin_legs in sorted(by_pin.items()):
        width = pin + 1
        medians = [leg["block_us_median"] for leg in pin_legs]
        from_blocks = [leg["round_us_from_blocks"] for leg in pin_legs]
        hist = pin_legs[0]["width_hist"]
        leak = 1.0 - hist.get(str(width), 0.0)
        widths[width] = {
            "pin": pin,
            "legs": len(pin_legs),
            "positions": [leg["position"] for leg in pin_legs],
            "block_us_median_mean": statistics.fmean(medians),
            "block_us_median_values": medians,
            "block_us_median_sd": (statistics.stdev(medians)
                                   if len(medians) > 1 else 0.0),
            "round_us_from_blocks_mean": statistics.fmean(from_blocks),
            "round_us_from_blocks_values": from_blocks,
            "spt_mean": statistics.fmean([leg["spt"] for leg in pin_legs]),
            "spt_blocks_only_mean": statistics.fmean(
                [leg["spt_blocks_only"] for leg in pin_legs]),
            "seed_prefill_seconds": sorted(
                {round(leg["seed_prefill_seconds"], 4) for leg in pin_legs}),
            "round_counts": sorted({leg["round_count"] for leg in pin_legs}),
            "mean_draft_len": sorted({round(leg["mean_draft_len"], 6)
                                      for leg in pin_legs}),
            "max_draft_len": sorted({leg["max_draft_len"]
                                     for leg in pin_legs}),
            "accepted_draft_rate": sorted({round(leg["accepted_draft_rate"], 6)
                                           for leg in pin_legs}),
            "rejected_draft_total": sorted({leg["rejected_draft_total"]
                                            for leg in pin_legs}),
            "replayed_round_count": sorted({leg["replayed_round_count"]
                                            for leg in pin_legs}),
            "width_hist": hist,
            "pin_leak_fraction": leak,
            "head_step_forwards_per_round": pin,
            "verify_forwards_per_round": 1,
            "sdpa_calls_per_verify": sdpa_calls(width) * FULL_ATTENTION_LAYERS,
            "entry_temps": [leg["gate_entry_temp_c"] for leg in pin_legs],
            "exit_temps": [leg["leg_exit_temp_c"] for leg in pin_legs],
            "all_tokens_matched": all(leg["all_tokens_matched"]
                                      for leg in pin_legs),
        }

    ordered = sorted(widths)
    for index, width in enumerate(ordered):
        entry = widths[width]
        if index:
            previous = widths[ordered[index - 1]]
            entry["step_us_from_previous"] = (
                entry["block_us_median_mean"]
                - previous["block_us_median_mean"])
        replayed = REPLAYED_RANKED_US.get(width)
        if replayed is not None:
            entry["replayed_ranked_us"] = replayed
            entry["measured_over_replayed"] = (
                entry["block_us_median_mean"] / replayed)

    steps = {}
    for index in range(1, len(ordered)):
        low, high = ordered[index - 1], ordered[index]
        measured = widths[high]["block_us_median_mean"] \
            - widths[low]["block_us_median_mean"]
        replayed = REPLAYED_RANKED_US[high] - REPLAYED_RANKED_US[low]
        steps[f"{low}->{high}"] = {
            "measured_us": measured,
            "replayed_ranked_us": replayed,
            "measured_over_replayed": measured / replayed if replayed else None,
        }

    return {
        "widths": widths,
        "steps": steps,
        "traced_legs": {int(leg["pin"]): {
            "round_count": leg["round_count"],
            "mean_draft_len": leg["mean_draft_len"],
            "accepted_draft_rate": leg["accepted_draft_rate"],
            "width_hist": leg["width_hist"],
        } for leg in traced},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="research/e145-artifacts/legs.json")
    args = parser.parse_args()

    legs = collect()
    blob = {
        "harness": "local",
        "leg_count": len(legs),
        "legs": legs,
        "r1": r1_tables(legs),
        "r1b": r1_tables(legs, "r1b-"),
        "r2": r2_tables(legs),
        "replayed_ranked_us": REPLAYED_RANKED_US,
        "pre_arm_ranked_us": PRE_ARM_RANKED_US,
    }
    blob["r1b_replicate"] = r1b_replicate(blob["r1"], blob["r1b"])
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")

    print(f"e145_read: {len(legs)} legs -> {out}")
    for fixture, rep in blob["r1b_replicate"].items():
        print(f"\n[R1b] second-session replicate, {fixture}")
        print(f"  pb6 vs ship   R1 {rep['r1_pct']:+.4f} %"
              f"   R1b {rep['r1b_pct']:+.4f} %"
              f"   gap {rep['gap_pp']:+.4f} pp")
        print(f"  blocks only   R1 {rep['r1_blocks_pct']:+.4f} %"
              f"   R1b {rep['r1b_blocks_pct']:+.4f} %"
              f"   gap {rep['blocks_gap_pp']:+.4f} pp")
        print(f"  same sign {rep['same_sign']}"
              f" ; work signatures match {rep['work_signatures_match']}")
    for fixture, entry in blob["r1"].items():
        arms = entry["arms"]
        print(f"\n[R1] {fixture}")
        for arm in ("serial", "ship", "pb6"):
            if arm not in arms:
                continue
            a = arms[arm]
            print(f"  {arm:7s} spt={a['spt_mean']:.9f} sd={a['spt_sd']:.9f}"
                  f" rounds={a['round_counts']} draft={a['mean_draft_len']}"
                  f" ge6={a['width_mass_ge6']:.4f}"
                  f" ratio={a.get('local_serial_to_mtp_ratio', float('nan')):.6f}")
        if "pb6_vs_ship_pct" in entry:
            print(f"  pb6 vs ship: {entry['pb6_vs_ship_pct']:+.4f} %"
                  f" (blocks only"
                  f" {entry['pb6_vs_ship_blocks_only_pct']:+.4f} %)"
                  f"   width mass >=6 shift"
                  f" {entry['width_mass_ge6_shift_pp']:+.2f} pp")

    r2 = blob["r2"]["widths"]
    if r2:
        print("\n[R2] measured live per-round cost, microseconds")
        for width in sorted(r2):
            e = r2[width]
            print(f"  M={width} pin={e['pin']}"
                  f" median={e['block_us_median_mean']:.1f}"
                  f" sd={e['block_us_median_sd']:.1f}"
                  f" fromblocks={e['round_us_from_blocks_mean']:.1f}"
                  f" leak={e['pin_leak_fraction']:.4f}"
                  f" replayed={e.get('replayed_ranked_us')}"
                  f" ratio={e.get('measured_over_replayed', float('nan')):.4f}")
        print("\n[R2] steps")
        for name, step in blob["r2"]["steps"].items():
            print(f"  {name}: measured={step['measured_us']:+.1f} us"
                  f" replayed={step['replayed_ranked_us']:+.1f} us"
                  f" ratio={step['measured_over_replayed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
