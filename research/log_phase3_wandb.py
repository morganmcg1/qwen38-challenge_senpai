#!/usr/bin/env python3
"""Research-only: publish the E15 Phase 3 counterbalanced ABBA A/B to W&B.

usage:
  research/log_phase3_wandb.py .mlxfast-private/draft-bits/e15-r1-p3 \
      --log <run_job log> [--order 4,3,3,4] [--group G] [--notes ...]

Replicate directories are `<prefix>-p<position>-b<bits>`. Phase 2 published a
single control/candidate pair whose arms started 7.96C apart; this run repeats
each arm twice in an ABBA order, so three separable estimates are published:

  * per-pair deltas, whose spread bounds the residual after counterbalancing,
  * the ABBA-combined delta, which cancels drift linear in arm position,
  * the control-vs-control delta between the two identical control replicates,
    which is this host's own reproducibility and the floor the candidate beats.

The arms are temperature-EQUALIZED to a common start, not cool-gate qualified:
this host's ambient GPU floor sits above the 40C gate, so `gate_qualified_for_
timing` stays false and the result remains a directional screen.
"""

import argparse
import json
import statistics as st
from pathlib import Path

import wandb

from draft_bits_attribution import aggregate, attribute
from draft_bits_phase2 import FIELDS, parse
from draft_bits_phase3 import adjacent_pairs, replicate_dirs, round_delta
from log_phase2_wandb import (AMDAHL_KEYS, ENTITY, LEG_KEYS, PROJECT,
                              parse_identity, peak_rss)


def steady_per_readout(pairs):
    steady = pairs[1:]
    total = sum(a["round_us"] - b["round_us"] for a, b in steady)
    readouts = sum(a["d"] for a, _ in steady)
    return total, readouts, (total / readouts if readouts else float("nan"))


def trace_tables(blocks, order, control, ctl_slots, cand_slots, summary):
    """Publish the per-round trace decomposition, when a trace log exists.

    `run-amdahl-measurement.sh` re-points MLXFAST_SWIFT_BIN at
    research/capture-cli.sh and never exports MLX_QWEN_MTP_TRACE, so a Phase 3
    log carries no round records and none of this is available. The parent's own
    timed report still carries the result, so the absence degrades the
    decomposition rather than the headline.
    """
    reqt = wandb.Table(columns=["slot", "instance", "bits", "source_bits",
                                "requant_ms", "round_count"])
    per_slot_instance = {}
    for b in blocks:
        i = per_slot_instance[b["slot"]] = per_slot_instance.get(b["slot"], 0) + 1
        reqt.add_data(b["slot"], i, int(b["bits"]), int(b["source_bits"]),
                      b["requant_ms"], len(b["rounds"]))
    requant = [b["requant_ms"] for b in blocks if b["requant_ms"] > 0]
    summary["requant_ms_mean"] = st.mean(requant) if requant else 0.0
    summary["requant_ms_max"] = max(requant) if requant else 0.0

    timed = {}
    for b in blocks:
        if b["rounds"]:
            assert b["slot"] not in timed, f"slot {b['slot']} timed twice"
            timed[b["slot"]] = b["rounds"]
    assert sorted(timed) == list(range(1, len(order) + 1)), \
        f"expected timed rounds for slots 1..{len(order)}, got {sorted(timed)}"

    scheds = {tuple((x["d"], x["acc"]) for x in timed[s]) for s in timed}
    summary["acceptance/distinct_schedules"] = len(scheds)
    summary["headline/acceptance_term_pct"] = 0.0 if len(scheds) == 1 else None
    assert len(scheds) == 1, \
        "depth/acceptance schedules diverge; the acceptance term is not zero"

    per_pair = []
    for a, b in adjacent_pairs(order):
        ctl, cand = (a, b) if order[a - 1] == control else (b, a)
        pairs = round_delta(timed[ctl], timed[cand])
        total, readouts, per = steady_per_readout(pairs)
        per_pair.append(per)
        tag = f"pair_p{ctl}_p{cand}"
        summary[f"{tag}/paired_rounds"] = len(pairs)
        summary[f"{tag}/steady_round_us_delta_total"] = total
        summary[f"{tag}/steady_readout_count"] = readouts
        summary[f"{tag}/implied_us_saved_per_readout"] = per
    summary["pairs/implied_us_saved_per_readout_spread"] = \
        max(per_pair) - min(per_pair)

    n = min(len(timed[s]) for s in timed)
    depths = [timed[ctl_slots[0]][i]["d"] for i in range(n)]
    combined, rows = [], wandb.Table(
        columns=["round", "depth", "accepted"]
        + [f"p{s}_{f}" for s in sorted(timed) for f in FIELDS]
        + [f"abba_delta_{f}" for f in FIELDS])
    for i in range(n):
        if any(timed[s][i]["d"] != depths[i] for s in timed):
            continue
        row = {"d": depths[i]}
        for f in FIELDS:
            row[f] = (st.mean(timed[s][i][f] for s in ctl_slots)
                      - st.mean(timed[s][i][f] for s in cand_slots))
        combined.append(row)
        ref = timed[ctl_slots[0]][i]
        rows.add_data(ref["round"], ref["d"], ref["acc"],
                      *[timed[s][i][f] for s in sorted(timed) for f in FIELDS],
                      *[row[f] for f in FIELDS])
    summary["abba/paired_rounds"] = len(combined)
    for f in FIELDS:
        d = [r[f] for r in combined]
        summary[f"abba/delta/{f}_median_us"] = st.median(d)
        summary[f"abba/delta/{f}_mean_us"] = st.mean(d)
    steady = combined[1:]
    total = sum(r["round_us"] for r in steady)
    readouts = sum(r["d"] for r in steady)
    summary["abba/steady_round_us_delta_total"] = total
    summary["abba/steady_readout_count"] = readouts
    summary["abba/implied_us_saved_per_readout"] = total / readouts

    if len(ctl_slots) == 2:
        a, b = ctl_slots
        pairs = round_delta(timed[a], timed[b])
        total, readouts, per = steady_per_readout(pairs)
        summary["noise/control_vs_control_paired_rounds"] = len(pairs)
        summary["noise/control_vs_control_per_readout_us"] = per

    return {"requant_provenance": reqt, "rounds": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=Path)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--order", default="4,3,3,4")
    ap.add_argument("--control", default="4")
    ap.add_argument("--group", default="qwen38-r1-e15-draft-readout-3bit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    order = args.order.split(",")
    reps = replicate_dirs(str(args.prefix.parent), args.prefix.name, order)
    ctl_slots = [s for s, b in enumerate(order, 1) if b == args.control]
    cand_slots = [s for s, b in enumerate(order, 1) if b != args.control]

    docs, ident = {}, {}
    for slot, _bits, d in reps:
        d = Path(d)
        docs[slot] = json.loads((d / "amdahl.json").read_text())
        ident[slot] = parse_identity(d / "identity.txt")
        ident[slot]["peak_rss_bytes"] = peak_rss(d / "rusage.txt")

    prov = docs[1]["provenance"]
    config = {
        "experiment": "e15-phase3-draft-readout-3bit-abba",
        "phase": 3,
        "order": order,
        "control_bits": int(args.control),
        "candidate_bits": int(next(b for b in order if b != args.control)),
        "control_slots": ctl_slots,
        "candidate_slots": cand_slots,
        "base_sha": prov["base_sha"],
        "head_sha": prov["head_sha"],
        "host_chip": prov["host_chip"],
        "host_os": prov["host_os"],
        "host_memsize_bytes": int(prov["host_memsize_bytes"]),
        "mode": prov["mode"],
        "decode_tokens": docs[1]["mtp_leg"]["emitted_token_total"],
        "seed_token_count": docs[1]["mtp_leg"]["seed_token_count"],
        "cool_gate": ident[1].get("cool_gate"),
        "gate_qualified_for_timing": all(
            ident[s].get("cool_gate") == "passed_real_gate" for s in docs),
        "settle_target_c": ident[1].get("settle_target_c"),
        "temperature_equalized": True,
        "git_head_dirty": ident[1].get("dirty"),
        "segmented_streak_gate": ident[1].get("segmented_streak_gate"),
        "qmv_fast_crossrow_m8_ipg": ident[1].get("m8_ipg"),
        "worker_sha256": ident[1].get("worker_sha256"),
        "worker_sha256_distinct_across_arms": len(
            {ident[s].get("worker_sha256") for s in docs}),
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, group=args.group,
                     job_type="phase3-abba-timing", config=config,
                     notes=args.notes or "E15 Phase 3 counterbalanced ABBA A/B: "
                     "per-pair, ABBA-combined, and control-vs-control deltas")

    legs = wandb.Table(columns=["slot", "bits", "leg", "all_tokens_matched",
                                "emitted_token_total", "seconds_per_token",
                                "gpu_temp_c_before", "gpu_temp_c_after",
                                "cool_gate", "settle_reached_c", "settle_min_c",
                                "settle_waited_s"])
    summary = {}
    for slot, bits, _d in reps:
        doc = docs[slot]
        for leg in ("mtp_leg", "serial_leg"):
            node = doc[leg]
            legs.add_data(slot, int(bits), leg.replace("_leg", ""),
                          bool(node["all_tokens_matched"]),
                          node["emitted_token_total"],
                          node["parent_measured_seconds_per_token"],
                          float(ident[slot].get("gpu_temp_c_before", "nan")),
                          float(ident[slot].get("gpu_temp_c_after", "nan")),
                          ident[slot].get("cool_gate"),
                          ident[slot].get("settle_reached_c"),
                          ident[slot].get("settle_min_c"),
                          ident[slot].get("settle_waited_s"))
        summary[f"p{slot}/cool_gate_passed_real_gate"] = (
            ident[slot].get("cool_gate") == "passed_real_gate")
        for k in LEG_KEYS:
            if k in doc["mtp_leg"]:
                summary[f"p{slot}/mtp/{k}"] = doc["mtp_leg"][k]
        for k in AMDAHL_KEYS:
            summary[f"p{slot}/{k}"] = doc["amdahl"][k]
        summary[f"p{slot}/bits"] = int(bits)
        summary[f"p{slot}/peak_rss_bytes"] = ident[slot]["peak_rss_bytes"]
        for k in ("gpu_temp_c_before", "gpu_temp_c_after"):
            summary[f"p{slot}/{k}"] = float(ident[slot].get(k, "nan"))

    tables = {"legs": legs}
    blocks = parse(args.log)
    summary["trace/available"] = bool(blocks)
    if blocks:
        tables.update(trace_tables(blocks, order, args.control, ctl_slots,
                                   cand_slots, summary))

    # The parent's own timed report is the authority for work actually done, and
    # unlike the trace it is always present. Price the mechanism from Phase 1's
    # separately measured readout cost so the prompt-specific trajectory term
    # cannot be quoted as part of the bandwidth result.
    agg = aggregate(str(args.prefix.parent), args.prefix.name,
                    [int(b) for b in order])
    attr = attribute(agg, int(args.control), config["candidate_bits"])
    work = wandb.Table(columns=["bits", "replicates", "rounds", "rows",
                                "draft_calls", "accept_rate",
                                "decode_work_seconds", "prefill_seconds",
                                "leg_seconds", "local_score"])
    for bits in sorted(agg, reverse=True):
        a = agg[bits]
        work.add_data(bits, a["replicates"], a["rounds"], a["rows"], a["calls"],
                      a["accept_rate"], a["work"], a["prefill"], a["leg"],
                      a["score"])
        for k in ("rounds", "rows", "calls", "accept_rate", "work", "prefill",
                  "leg", "score"):
            summary[f"work/b{bits}/{k}"] = a[k]
    tables["work"] = work

    transfer = wandb.Table(columns=[
        "label", "head", "head_mb", "readout_share_of_draft_bytes_pct",
        "draft_step_ms", "decode_work_seconds", "mechanism_pct_of_decode_work",
        "mechanism_score_pct"])
    for t in attr.pop("transfer"):
        transfer.add_data(*[t[c] for c in transfer.columns])
        summary[f"transfer/{t['head']}/mechanism_score_pct"] = \
            t["mechanism_score_pct"]
        summary[f"transfer/{t['head']}/readout_share_of_draft_bytes_pct"] = \
            t["readout_share_of_draft_bytes_pct"]
    tables["transfer_model"] = transfer
    for k, v in attr.items():
        summary[f"attribution/{k}"] = v
    summary["headline/mechanism_only_score_pct"] = \
        attr["mechanism_only_score_pct"]
    summary["headline/trajectory_share_of_delta_pct"] = \
        attr["term_trajectory_share_pct"]

    def mean_leg(slots, leg, key="parent_measured_seconds_per_token"):
        return st.mean(docs[s][leg][key] for s in slots)

    # Two estimators of the same contrast. `pooled` compares arm means, which is
    # what the ABBA order was built for. `mean_of_halves` averages the two
    # nearly thermally matched adjacent-run ratios (p1-p2, p4-p3); agreement
    # between the two is the evidence that no position/thermal term survives.
    halves = [(a, b) if order[a - 1] == args.control else (b, a)
              for a, b in adjacent_pairs(order)]
    for leg, tag in (("mtp_leg", "mtp"), ("serial_leg", "serial")):
        c = mean_leg(ctl_slots, leg)
        k = mean_leg(cand_slots, leg)
        summary[f"abba/{tag}_seconds_per_token_control"] = c
        summary[f"abba/{tag}_seconds_per_token_candidate"] = k
        summary[f"headline/{tag}_seconds_per_token_pct"] = 100.0 * (k - c) / c
        halved = []
        for ct, cd in halves:
            key = "parent_measured_seconds_per_token"
            halved.append(100.0 * (docs[cd][leg][key] - docs[ct][leg][key])
                          / docs[ct][leg][key])
        for i, (ct, cd) in enumerate(halves, 1):
            summary[f"halves/{tag}_h{i}_p{ct}_vs_p{cd}_pct"] = halved[i - 1]
        summary[f"headline/{tag}_seconds_per_token_pct_mean_of_halves"] = \
            st.mean(halved)
        summary[f"headline/{tag}_pooled_minus_mean_of_halves_pp"] = \
            100.0 * (k - c) / c - st.mean(halved)
        summary[f"halves/{tag}_spread_pp"] = max(halved) - min(halved)
    summary["headline/serial_drift_pct"] = \
        summary["headline/serial_seconds_per_token_pct"]

    c = mean_leg(ctl_slots, "mtp_leg", "steady_seconds_per_token")
    k = mean_leg(cand_slots, "mtp_leg", "steady_seconds_per_token")
    summary["headline/steady_mtp_seconds_per_token_pct"] = 100.0 * (k - c) / c

    c = st.mean(docs[s]["amdahl"]["measured_local_score"] for s in ctl_slots)
    k = st.mean(docs[s]["amdahl"]["measured_local_score"] for s in cand_slots)
    summary["headline/local_score_pct"] = 100.0 * (k - c) / c
    summary["headline/ranked_modelled_score_fixed_serial"] = (
        st.mean(docs[s]["amdahl"]["ranked_window_leg_seconds_serial"]
                for s in ctl_slots)
        / st.mean(docs[s]["amdahl"]["ranked_window_leg_seconds_mtp"]
                  for s in cand_slots))

    a, b = ctl_slots[0], ctl_slots[-1]
    ca = docs[a]["mtp_leg"]["parent_measured_seconds_per_token"]
    cb = docs[b]["mtp_leg"]["parent_measured_seconds_per_token"]
    summary["noise/control_vs_control_mtp_pct"] = 100.0 * (cb - ca) / ca

    run.log(tables)
    run.summary.update(summary)
    for k in sorted(summary):
        if k.startswith(("headline/", "noise/", "pairs/", "acceptance/",
                         "attribution/", "transfer/", "trace/",
                         "abba/implied", "abba/steady", "abba/paired")):
            print("%-58s %s" % (k, summary[k]))
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
