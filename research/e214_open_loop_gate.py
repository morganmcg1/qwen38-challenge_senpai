#!/usr/bin/env python3
"""E214 open-loop agreement gate: the shipped table IS the desk table.

    python3 research/e214_open_loop_gate.py RUN_DIR [--tokens N] [--json OUT]

harness=local for the traced run, harness=ranked for the desk projection this
gate protects. No timing claim is made or read here.

WHAT THIS GATE PROVES, in the order the evidence has to hold.

  1. TABLE IDENTITY. The binary prints its own price table once per traced run
     (`mtp-price:` in hexfloat, `Qwen36MTPBlockSession.tracedDepthPrice`). The
     gate reads that line, so the numbers under test are the numbers the walk
     compared against, not a second transcription of the artifact. They must
     equal `receipt_proof.forward_guarded` of
     `research/e211-artifacts/step-price.json` bit for bit.

  2. GRID AGREEMENT. The shipped table must reproduce the fitted DEPTH MAP:
     `e211_step_price.price_cuts` of the printed marginal must equal the
     artifact cut points, and `e200_desk_price.greedy_depth` must return the
     mapped depth at every one of the 4001 quadrature nodes. This is the desk
     gate's own standard (`e211_step_price.verify_price`).

  3. ROUND-FOR-ROUND AGREEMENT. Every traced round records the state the
     scheduler snapshotted before it proposed anything (`ema=`, `m=`, `cap=`)
     and the walk it then ran (`sched=`, `d=`). Replaying that state through
     `e128_replay.cost_model_depth` under the printed table must return the
     same depth at every round, and every field of the replayed walk must
     agree with the printed walk to the resolution the trace prints. Open
     loop: the recorded state is replayed, never regenerated, so the EMA
     feedback of a moved schedule cannot hide a disagreement.

     The trace prints `ema=` and `sched=` rounded to 6 decimals, so the
     replay reads inputs that are already quantized to 1e-6 and its running
     acceptance product can land one unit either side of the printed field.
     Byte identity of the walk string is therefore NOT a well-posed
     criterion; it is reported as a diagnostic only. The criterion is exact
     depth agreement plus `field_max_ulp <= 1`, one unit in the last printed
     place. Gate 4 shows this floor is far below any real disagreement.

  4. POSITIVE CONTROL. The same comparison is run under the uniform price the
     candidate replaced. It must FAIL, or the gate proves nothing.

Exit status is 0 only when every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_replay as R128  # noqa: E402
import e200_desk_price as D  # noqa: E402
import e200_trace_replay as T200  # noqa: E402
import e211_step_price as E211  # noqa: E402

ARTIFACT = pathlib.Path(__file__).resolve().parent / "e211-artifacts" \
    / "step-price.json"
ARTIFACT_SHA256 = \
    "3c5295cc1d05816e01da571a8053e73aa09ed1cc9e72e0ea192aaa415318f4ae"
ARTIFACT_KEY = ("receipt_proof", "forward_guarded")
SHIPPED_ARM = "stepq"
# `snapshotScheduleSignal` prints `ema=` and `sched=` at 6 decimals, so a
# replay from those quantized inputs may miss a printed field by one unit in
# the last printed place. Distances are measured in these units throughout.
QUANT_DECIMALS = 6
QUANT_UNITS = 1
MAXD = E211.MAXD
QGRID = E211.QGRID


def read_price_line(run_dir: pathlib.Path) -> dict:
    """The binary's own table, from the one `mtp-price:` line it prints."""
    hits = [line for line in
            (run_dir / "trace.txt").read_text(errors="replace").splitlines()
            if line.startswith("mtp-price:")]
    if not hits:
        raise SystemExit("no mtp-price: witness line in %s/trace.txt" % run_dir)
    fields = {}
    for token in hits[0][len("mtp-price:"):].split():
        key, _, value = token.partition("=")
        fields[key] = value
    price = {"arm": fields["arm"]}
    for key in ("marginal", "cumulative"):
        price[key] = [float.fromhex(v) for v in fields[key].split(",")]
    price["witness_lines"] = len(hits)
    price["witness_identical"] = len(set(hits)) == 1
    return price


def artifact_table() -> dict:
    digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    payload = json.loads(ARTIFACT.read_text())
    node = payload
    for key in ARTIFACT_KEY:
        node = node[key]
    return {
        "sha256": digest,
        "sha256_expected": ARTIFACT_SHA256,
        "sha256_match": digest == ARTIFACT_SHA256,
        "key": "/".join(ARTIFACT_KEY),
        "cuts": list(node["cuts"]),
        "thresholds": list(node["thresholds"]),
        "marginal": list(node["price_marginal"]),
        "cumulative": list(node["price_cumulative"]),
    }


def gate_table_identity(price: dict, want: dict) -> dict:
    return {
        "arm": price["arm"],
        "arm_is_shipped": price["arm"] == SHIPPED_ARM,
        "witness_lines": price["witness_lines"],
        "witness_identical": price["witness_identical"],
        "artifact_sha256_match": want["sha256_match"],
        "marginal_bit_identical": price["marginal"] == want["marginal"],
        "cumulative_bit_identical": price["cumulative"] == want["cumulative"],
        "max_abs_marginal_delta": max(
            abs(a - b) for a, b in zip(price["marginal"], want["marginal"])),
        "shipped_marginal": price["marginal"],
        "artifact_marginal": want["marginal"],
    }


def gate_grid(price: dict, want: dict) -> dict:
    """The desk gate's own standard, applied to the shipped numbers."""
    cuts = E211.cuts_of(want["cuts"])
    got_cuts = E211.price_cuts(price["marginal"])
    agreement, mismatches = E211.verify_price(cuts, price)
    depth_map = np.searchsorted(np.array(cuts[1:MAXD + 1]),
                                np.arange(QGRID), side="right")
    return {
        "grid_nodes": QGRID,
        "cuts_shipped": got_cuts[1:MAXD + 1],
        "cuts_artifact": want["cuts"],
        "cuts_match": got_cuts == cuts,
        "greedy_agreement": agreement,
        "greedy_mismatches": mismatches,
        "depth_map_reach": {str(k): int(np.sum(depth_map >= k))
                            for k in range(1, MAXD + 1)},
    }


def replay_rounds(rounds: list[dict], price: dict) -> dict:
    """Depth and walk-string agreement of the recorded state under `price`."""
    depth_hits = 0
    sched_hits = 0
    max_ulp = 0
    steps = 0
    fields = 0
    fields_off = 0
    mismatches = []
    for record in rounds:
        depth, sched, _ = R128.cost_model_depth(
            record["ema"], record["margin"],
            offered_depth=record["offer"], width_cap=record["cap"],
            marginal=price["marginal"], cumulative=price["cumulative"])
        steps += record["sched"].count(";")
        depth_hits += depth == record["depth"]
        sched_hits += sched == record["sched"]
        shipped = T200._sched_fields(record["sched"])
        replayed = T200._sched_fields(sched)
        if len(shipped) != len(replayed):
            mismatches.append({"round": record["round"], "kind": "shape",
                               "shipped_sched": record["sched"],
                               "replayed_sched": sched})
            continue
        deltas = [abs(round((a - b) * 10 ** QUANT_DECIMALS))
                  for a, b in zip(shipped, replayed)]
        fields += len(deltas)
        fields_off += sum(1 for d in deltas if d)
        ulp = max(deltas, default=0)
        max_ulp = max(max_ulp, ulp)
        if ulp > QUANT_UNITS or depth != record["depth"]:
            mismatches.append({
                "round": record["round"], "kind": "value", "ulp": ulp,
                "shipped_depth": record["depth"], "replayed_depth": depth,
                "shipped_sched": record["sched"], "replayed_sched": sched})
    total = len(rounds)
    return {
        "rounds_checked": total,
        "steps_checked": steps,
        "depth_agreement": depth_hits / total if total else 0.0,
        "sched_byte_identical": sched_hits / total if total else 0.0,
        "fields_checked": fields,
        "fields_off_by_quantum": fields_off,
        "field_max_ulp": max_ulp,
        "mismatches_total": len(mismatches),
        "mismatches": mismatches[:8],
    }


def depth_histogram(rounds: list[dict]) -> dict:
    hist = [0] * (MAXD + 1)
    for record in rounds:
        hist[record["depth"]] += 1
    total = len(rounds) or 1
    return {
        "depth_hist": hist,
        "edl": sum(i * n for i, n in enumerate(hist)) / total,
        "accepted_mean": sum(r["accepted"] for r in rounds) / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=64,
                        help="decode window of the traced leg; sets the tail "
                             "offer narrowing the driver applies")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    rounds = T200.read_rounds_with_cost(args.run_dir)
    if not rounds:
        print("no traced rounds in %s" % args.run_dir, file=sys.stderr)
        return 2
    T200.attach_offers(rounds, args.tokens)

    price = read_price_line(args.run_dir)
    want = artifact_table()
    identity = gate_table_identity(price, want)
    grid = gate_grid(price, want)
    live = replay_rounds(rounds, price)
    control = replay_rounds(rounds, D.ship_price())
    census = depth_histogram(rounds)
    meta = R128.read_meta(args.run_dir)
    score_path = args.run_dir / "score.json"
    score = (json.loads(score_path.read_text()) if score_path.exists() else {})
    metrics = score.get("metrics", {})

    passed = (
        identity["arm_is_shipped"]
        and identity["witness_identical"]
        and identity["artifact_sha256_match"]
        and identity["marginal_bit_identical"]
        and identity["cumulative_bit_identical"]
        and grid["cuts_match"]
        and grid["greedy_agreement"] == 1.0
        and grid["greedy_mismatches"] == 0
        and live["depth_agreement"] == 1.0
        and live["field_max_ulp"] <= QUANT_UNITS
        and live["mismatches_total"] == 0
        and control["mismatches_total"] > 0
        # The control has to fail by far more than the trace's own printing
        # resolution, or a passing live arm would prove nothing.
        and control["field_max_ulp"] > 100 * QUANT_UNITS
    )

    print("E214 open-loop agreement gate")
    print("leg            %s" % args.run_dir)
    print("candidate      %s" % meta.get("base_sha", "?")[:12])
    print("worker         %s" % meta.get("worker_sha256", "?")[:16])
    print("head           %s" % metrics.get("head_provenance_sha256", "?")[:16])
    print("mode / tokens  %s / %s" % (metrics.get("mode", "?"),
                                      metrics.get("decode_tokens", "?")))
    print("exactness      all_tokens_matched=%s residual_divergence=%s"
          % (metrics.get("all_tokens_matched", "?"),
             metrics.get("residual_divergence_count", "?")))
    print()
    print("1. table identity (witness printed by the binary)")
    print("   arm                    %s (shipped=%s)"
          % (identity["arm"], identity["arm_is_shipped"]))
    print("   witness lines          %d, identical=%s"
          % (identity["witness_lines"], identity["witness_identical"]))
    print("   artifact sha256        %s" % want["sha256"])
    print("   marginal bit-identical %s (max |delta| %.3e)"
          % (identity["marginal_bit_identical"],
             identity["max_abs_marginal_delta"]))
    print("   cumulative bit-ident.  %s" % identity["cumulative_bit_identical"])
    print("   shipped marginal       %s"
          % "  ".join("%.6f" % v for v in identity["shipped_marginal"]))
    print()
    print("2. grid agreement over %d quadrature nodes" % grid["grid_nodes"])
    print("   cuts shipped           %s" % grid["cuts_shipped"])
    print("   cuts artifact          %s" % grid["cuts_artifact"])
    print("   cuts match             %s" % grid["cuts_match"])
    print("   greedy agreement       %.4f (%d mismatching nodes)"
          % (grid["greedy_agreement"], grid["greedy_mismatches"]))
    print()
    print("3. round-for-round agreement on the traced fixture")
    print("   rounds / walk steps    %d / %d"
          % (live["rounds_checked"], live["steps_checked"]))
    print("   depth agreement        %.4f (criterion: 1.0000)"
          % live["depth_agreement"])
    print("   max field distance     %d units of 1e-%d (criterion: <= %d)"
          % (live["field_max_ulp"], QUANT_DECIMALS, QUANT_UNITS))
    print("   fields off by a unit   %d of %d"
          % (live["fields_off_by_quantum"], live["fields_checked"]))
    print("   sched byte identical   %.4f (diagnostic; the trace prints 1e-%d)"
          % (live["sched_byte_identical"], QUANT_DECIMALS))
    print("   mismatches             %d" % live["mismatches_total"])
    for bad in live["mismatches"]:
        print("   MISMATCH round %s: %s" % (bad["round"], bad))
    print()
    print("4. positive control (the uniform price this candidate replaced)")
    print("   depth agreement        %.4f" % control["depth_agreement"])
    print("   mismatches             %d (must be > 0)"
          % control["mismatches_total"])
    print("   max field distance     %d units (must be > %d)"
          % (control["field_max_ulp"], 100 * QUANT_UNITS))
    print()
    print("leg schedule census: rounds=%d edl=%.3f acc_mean=%.3f hist=%s"
          % (live["rounds_checked"], census["edl"], census["accepted_mean"],
             census["depth_hist"]))
    print()
    print("GATE %s" % ("PASS" if passed else "FAIL"))

    payload = {
        "harness": "local trace, ranked desk table",
        "run_dir": str(args.run_dir),
        "tokens": args.tokens,
        "meta": meta,
        "score_metrics": metrics,
        "artifact": {k: v for k, v in want.items() if k != "marginal"},
        "table_identity": identity,
        "grid": grid,
        "round_for_round": live,
        "positive_control": control,
        "quantization": {"trace_decimals": QUANT_DECIMALS,
                         "field_tolerance_units": QUANT_UNITS,
                         "control_min_units": 100 * QUANT_UNITS},
        "census": census,
        "pass": passed,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("wrote %s" % args.json)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
