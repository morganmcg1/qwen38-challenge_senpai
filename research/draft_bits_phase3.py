#!/usr/bin/env python3
"""Research-only: analyse a counterbalanced ABBA draft-bits timing experiment.

    research/draft_bits_phase3.py JOB_LOG --prefix TAG_PREFIX [--order 4,3,3,4]

Phase 2 compared one control run against one candidate run that started 7.96C
hotter, so its per-round delta and its thermal drift were confounded. Phase 3
runs each arm twice in an ABBA order and this script separates the two.

Three estimates are reported and they answer different questions:

  * per-pair deltas, one per adjacent (control, candidate) pair, whose spread
    bounds the residual after counterbalancing;
  * the ABBA-combined delta, which averages the replicates of each arm and so
    cancels any drift that is linear in position;
  * the control-vs-control delta between the two identical arm-4 replicates at
    positions 1 and 4, which measures this host's own reproducibility over the
    span of the experiment and is the noise floor the candidate must beat.

The per-round term and the acceptance term stay separate throughout, because a
precision change that buys bandwidth and one that changes which tokens were
drafted are different findings.
"""
import argparse
import json
import os
import statistics as st
import sys

from draft_bits_phase2 import FIELDS, parse


def replicate_dirs(root, prefix, order):
    return [(i + 1, b, os.path.join(root, "%s-p%d-b%s" % (prefix, i + 1, b)))
            for i, b in enumerate(order)]


def adjacent_pairs(order):
    """Pair each consecutive control/candidate neighbour, consuming both."""
    pairs, i = [], 0
    while i + 1 < len(order):
        if order[i] != order[i + 1]:
            pairs.append((i + 1, i + 2))
            i += 2
        else:
            i += 1
    return pairs


def round_delta(ctl, cand):
    n = min(len(ctl), len(cand))
    return [(ctl[i], cand[i]) for i in range(n) if ctl[i]["d"] == cand[i]["d"]]


def report_pair(label, ctl, cand):
    pairs = round_delta(ctl, cand)
    print("  %s: paired rounds=%d of %d" % (label, len(pairs), min(len(ctl), len(cand))))
    for f in FIELDS:
        d = [a[f] - b[f] for a, b in pairs]
        print("    %-16s median=%+8.1f us  mean=%+8.1f us"
              % (f, st.median(d), sum(d) / len(d)))
    steady = pairs[1:]
    dr = [a["round_us"] - b["round_us"] for a, b in steady]
    readouts = sum(a["d"] for a, _ in steady)
    per = sum(dr) / readouts if readouts else float("nan")
    print("    steady rounds=%d total_round_us=%+d readouts=%d per_readout=%+.1f us"
          % (len(steady), sum(dr), readouts, per))
    return per


TIMED_REPORT = "reports/04-mtp-timed.json"


def load_timed_reports(root, prefix, order):
    out = {}
    for slot, bits, d in replicate_dirs(root, prefix, order):
        path = os.path.join(d, TIMED_REPORT)
        if os.path.exists(path):
            out[slot] = json.load(open(path))
    return out


def work_report(reports, order, ctl_slots, cand_slots):
    """Split the leg delta into a work term and a cost-per-work term.

    The readout mechanism can only show up as cheaper work per verified row. A
    precision change that also moves the draft trajectory changes how many rows
    exist at all, and that part is prompt-specific, so the two must not be
    quoted as one number.
    """
    print("\n== work actually performed (parent's own timed report) ==")
    per_bits = {}
    for slot, bits in enumerate(order, 1):
        r = reports[slot]
        work = r["decode_seconds"] - r["seed_prefill_seconds"]
        row = {
            "bits": bits,
            "rounds": r["round_count"],
            "rows": r["declared_rows_total"],
            "accepted": r["accepted_draft_total"],
            "rejected": r["rejected_draft_total"],
            "mean_draft": r["effective_mean_draft_len"],
            "work": work,
            "per_row": work / r["declared_rows_total"],
            "per_round": work / r["round_count"],
            "first": r["first_block_seconds"],
            "p50_after_first": r["p50_block_request_seconds_after_first"],
        }
        per_bits.setdefault(bits, []).append(row)
        print("  slot=%d bits=%s rounds=%d rows=%d accepted=%d rejected=%d "
              "mean_draft=%.4f" % (slot, bits, row["rounds"], row["rows"],
                                   row["accepted"], row["rejected"],
                                   row["mean_draft"]))
        print("       decode_work=%.6fs per_row=%.7fs per_round=%.7fs "
              "first_block=%.6fs p50_after_first=%.6fs"
              % (work, row["per_row"], row["per_round"], row["first"],
                 row["p50_after_first"]))
        print("       matched=%s residual_divergence=%d rows_declared=%d "
              "rows_reference_checked=%d emitted=%d replayed_rounds=%d"
              % (r["all_tokens_matched"], r["residual_divergence_count"],
                 r["declared_rows_total"], r["reference_checked_row_total"],
                 r["emitted_token_total"], r["verify_block_replayed_round_count"]))

    for bits, rows in sorted(per_bits.items()):
        shapes = {(x["rounds"], x["rows"], x["accepted"]) for x in rows}
        print("  bits=%s replicates=%d distinct work shapes=%d %s"
              % (bits, len(rows), len(shapes),
                 "(work is a deterministic function of bit width)"
                 if len(shapes) == 1 else "WARNING: work varied across replicates"))

    ctl_rows = st.mean(reports[s]["declared_rows_total"] for s in ctl_slots)
    cand_rows = st.mean(reports[s]["declared_rows_total"] for s in cand_slots)
    ctl_rounds = st.mean(reports[s]["round_count"] for s in ctl_slots)
    cand_rounds = st.mean(reports[s]["round_count"] for s in cand_slots)
    ctl_work = st.mean(reports[s]["decode_seconds"] - reports[s]["seed_prefill_seconds"]
                       for s in ctl_slots)
    cand_work = st.mean(reports[s]["decode_seconds"] - reports[s]["seed_prefill_seconds"]
                        for s in cand_slots)
    d_rows = 100.0 * (cand_rows - ctl_rows) / ctl_rows
    d_rounds = 100.0 * (cand_rounds - ctl_rounds) / ctl_rounds
    d_work = 100.0 * (cand_work - ctl_work) / ctl_work
    d_per_row = 100.0 * ((cand_work / cand_rows) - (ctl_work / ctl_rows)) / (ctl_work / ctl_rows)
    d_per_round = 100.0 * ((cand_work / cand_rounds) - (ctl_work / ctl_rounds)) / (ctl_work / ctl_rounds)
    print("\n== decode-only decomposition (candidate - control) ==")
    print("  decode_work            %+.4f%%   (%.6fs -> %.6fs)"
          % (d_work, ctl_work, cand_work))
    print("  rows term              %+.4f%%   (%.1f -> %.1f verified rows)"
          % (d_rows, ctl_rows, cand_rows))
    print("  cost-per-row term      %+.4f%%   <- the readout mechanism"
          % d_per_row)
    print("  rounds term            %+.4f%%   (%.1f -> %.1f rounds)"
          % (d_rounds, ctl_rounds, cand_rounds))
    print("  cost-per-round term    %+.4f%%" % d_per_round)
    check = 100.0 * ((1 + d_rows / 100.0) * (1 + d_per_row / 100.0) - 1)
    print("  rows x cost-per-row    %+.4f%%   (reconstructs decode_work: residual %+.4f pp)"
          % (check, check - d_work))
    return {"d_work": d_work, "d_rows": d_rows, "d_per_row": d_per_row,
            "d_rounds": d_rounds, "d_per_round": d_per_round}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--order", default="4,3,3,4")
    ap.add_argument("--root", default=".mlxfast-private/draft-bits")
    ap.add_argument("--control", default="4")
    args = ap.parse_args()
    order = args.order.split(",")

    blocks = parse(args.log)
    timed = {}
    for b in blocks:
        if b["rounds"]:
            if b["slot"] in timed:
                sys.exit("slot %s has more than one timed instance" % b["slot"])
            timed[b["slot"]] = b["rounds"]
    ctl_slots = [s for s, b in enumerate(order, 1) if b == args.control]
    cand_slots = [s for s, b in enumerate(order, 1) if b != args.control]

    # The amdahl harness never exports MLX_QWEN_MTP_TRACE, so a leg measurement
    # carries no per-round trace. The parent's own timed report covers the same
    # ground more authoritatively, so degrade instead of refusing to analyse.
    have_trace = sorted(timed) == list(range(1, len(order) + 1))
    if not have_trace:
        print("== no per-round trace in this log (slots with rounds: %s) =="
              % (sorted(timed) or "none"))
        print("  falling back to the parent's timed reports and leg measurements")

    if have_trace:
        print("== draft-head provenance (every worker instance) ==")
        for b in blocks:
            print("  slot=%s bits=%s source_bits=%s requant_ms=%.3f rounds=%d"
                  % (b["slot"], b["bits"], b["source_bits"], b["requant_ms"],
                     len(b["rounds"])))

        print("\n== acceptance term ==")
        sched = {}
        for slot, bits in enumerate(order, 1):
            r = timed[slot]
            key = tuple((x["d"], x["acc"]) for x in r)
            sched.setdefault(key, []).append("p%d-b%s" % (slot, bits))
            print("  slot=%d bits=%s rounds=%d proposed=%d accepted=%d emitted=%d"
                  % (slot, bits, len(r), sum(x["d"] for x in r),
                     sum(x["acc"] for x in r), len(r) + sum(x["acc"] for x in r)))
        if len(sched) == 1:
            print("  all %d replicates share one (depth, accepted) schedule"
                  " => acceptance term is exactly 0" % len(order))
        else:
            print("  WARNING: %d distinct schedules: %s"
                  % (len(sched), list(sched.values())))

        print("\n== per-pair per-round delta (control - candidate) ==")
        per_pair = []
        for a, b in adjacent_pairs(order):
            ctl, cand = (a, b) if order[a - 1] == args.control else (b, a)
            per_pair.append(report_pair("pair p%d(b%s) - p%d(b%s)"
                                        % (ctl, order[ctl - 1], cand, order[cand - 1]),
                                        timed[ctl], timed[cand]))
        if len(per_pair) > 1:
            print("  per-readout across pairs: %s  spread=%.1f us"
                  % (" ".join("%+.1f" % p for p in per_pair),
                     max(per_pair) - min(per_pair)))

        print("\n== ABBA-combined delta (mean control slots %s - mean candidate slots %s) =="
              % (ctl_slots, cand_slots))
        n = min(len(timed[s]) for s in timed)
        depths = [timed[ctl_slots[0]][i]["d"] for i in range(n)]
        combined = []
        for i in range(n):
            if any(timed[s][i]["d"] != depths[i] for s in timed):
                continue
            row = {"d": depths[i]}
            for f in FIELDS:
                row[f] = (st.mean(timed[s][i][f] for s in ctl_slots)
                          - st.mean(timed[s][i][f] for s in cand_slots))
            combined.append(row)
        print("  paired rounds=%d of %d" % (len(combined), n))
        for f in FIELDS:
            d = [r[f] for r in combined]
            print("    %-16s median=%+8.1f us  mean=%+8.1f us"
                  % (f, st.median(d), sum(d) / len(d)))
        steady = combined[1:]
        tot = sum(r["round_us"] for r in steady)
        readouts = sum(r["d"] for r in steady)
        print("    steady rounds=%d total_round_us=%+.0f readouts=%d per_readout=%+.1f us"
              % (len(steady), tot, readouts, tot / readouts))

        if len(ctl_slots) == 2:
            print("\n== control-vs-control noise floor (p%d vs p%d, identical arms) =="
                  % tuple(ctl_slots))
            report_pair("ctl p%d - ctl p%d" % tuple(ctl_slots),
                        timed[ctl_slots[0]], timed[ctl_slots[1]])

    reports = load_timed_reports(args.root, args.prefix, order)
    if sorted(reports) == list(range(1, len(order) + 1)):
        work_report(reports, order, ctl_slots, cand_slots)
    else:
        print("\n== missing %s for slots %s, skipping work decomposition =="
              % (TIMED_REPORT, sorted(set(range(1, len(order) + 1)) - set(reports))))

    print("\n== legs, temperatures, and ABBA-combined headline ==")
    legs = {}
    for slot, bits, d in replicate_dirs(args.root, args.prefix, order):
        path = os.path.join(d, "amdahl.json")
        if not os.path.exists(path):
            print("  slot=%d bits=%s MISSING %s" % (slot, bits, path))
            continue
        r = json.load(open(path))
        ident = {}
        for line in open(os.path.join(d, "identity.txt")):
            for tok in line.replace("run-draft-bits-arm:", "").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    ident[k] = v
        legs[slot] = {
            "bits": bits,
            "mtp": r["mtp_leg"]["parent_measured_seconds_per_token"],
            "serial": r["serial_leg"]["parent_measured_seconds_per_token"],
            "matched": (r["mtp_leg"]["all_tokens_matched"]
                        and r["serial_leg"]["all_tokens_matched"]),
            "score": r["amdahl"]["measured_local_score"],
            "before": float(ident.get("gpu_temp_c_before", "nan")),
            "after": float(ident.get("gpu_temp_c_after", "nan")),
            "ident": ident,
        }
        print("  slot=%d bits=%s matched=%s mtp_spt=%.10f serial_spt=%.10f "
              "temp %.2f->%.2f C" % (slot, bits, legs[slot]["matched"],
                                     legs[slot]["mtp"], legs[slot]["serial"],
                                     legs[slot]["before"], legs[slot]["after"]))
        print("       cool_gate=%s settle_reached=%s settle_min=%s settle_waited=%s"
              " streak_gate=%s m8_ipg=%s worker=%s"
              % (ident.get("cool_gate"), ident.get("settle_reached_c"),
                 ident.get("settle_min_c"), ident.get("settle_waited_s"),
                 ident.get("segmented_streak_gate"), ident.get("m8_ipg"),
                 (ident.get("worker_sha256") or "?")[:12]))
    if len(legs) != len(order):
        return

    digests = {legs[s]["ident"].get("worker_sha256") for s in legs}
    print("  worker digests distinct=%d (one digest => every arm ran the same "
          "binary and differed only by MLX_QWEN_MTP_DRAFT_BITS)" % len(digests))

    # Reported two ways because they answer different questions. Pooled compares
    # arm means and is the estimator the ABBA order was designed for. The mean of
    # the two half-contrasts (p1-p2 and p4-p3) is a ratio-of-adjacent-runs
    # estimator: each half is nearly thermally matched, so agreement between the
    # two numbers is the evidence that no position/thermal term survives.
    halves = [(a, b) if order[a - 1] == args.control else (b, a)
              for a, b in adjacent_pairs(order)]
    for field in ("mtp", "serial"):
        c = st.mean(legs[s][field] for s in ctl_slots)
        k = st.mean(legs[s][field] for s in cand_slots)
        pooled = 100.0 * (k - c) / c
        half_deltas = [100.0 * (legs[cs][field] - legs[ct][field]) / legs[ct][field]
                       for ct, cs in halves]
        mean_half = st.mean(half_deltas)
        print("  %-6s control=%.10f candidate=%.10f" % (field, c, k))
        print("         pooled_delta=%+.4f%%  half_deltas=%s  mean_of_halves=%+.4f%%"
              "  pooled-vs-halves=%+.4f pp"
              % (pooled, " ".join("%+.4f%%" % d for d in half_deltas),
                 mean_half, pooled - mean_half))
    a, b = ctl_slots
    floor = 100.0 * (legs[b]["mtp"] - legs[a]["mtp"]) / legs[a]["mtp"]
    print("  control-vs-control mtp drift p%d->p%d = %+.4f%%  (noise floor)"
          % (a, b, floor))
    print("  half-contrast spread = %.4f pp vs noise floor %.4f pp"
          % (max(half_deltas) - min(half_deltas), abs(floor)))


if __name__ == "__main__":
    main()
