#!/usr/bin/env python3
"""E220 verdict: every integer-path variant against the base, per scored cell.

Reads the per-variant census artifacts and decides each variant on three static
channels, in this order:

  1. FP sequence digest. The dequant block's ordered floating-point opcode
     sequence must be identical to the base. A variant that changes it has
     reassociated and is dead (FINDING 404, closedAxes.splitKAsABitExactLever).
  2. AGX machine code. `text_sha8` identical to the base on both generations
     means the backend emits literally the same instructions, so the variant
     cannot change the run time by any amount. `text_bytes` larger than the
     base means the variant added instructions.
  3. Register pressure. New spill bytes, or a register count that crosses the
     FINDING 549 grid, is a stop signal for that variant.

The two `probe_*` variants are COST ORACLES. They compute the wrong answer by
construction and are never candidates; they price what a perfect extraction
would be worth, so the negative result is a bound and not a null.

  python3 research/e220_report.py
"""

from __future__ import annotations

import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e220-artifacts"
OUT = ARTIFACTS / "e220-report.json"

BASE = "base"
ORACLES = ("probe_zero_shift", "probe_min_extract")
CANDIDATES = ("bfe", "vecload", "word32", "hoist", "hoist_scalar", "all")
ARCHES = ("g16s", "g17s")

# The advisor's measured decomposition of the NA = 5 pass (FINDING 570,
# Entry 438). `b` is the byte-proportional term the census attacks.
B_TERM_MS_PER_ROUND = 26.434
B_TERM_SHARE_OF_PASS = 0.310
# Student promotion bar for this assignment.
PROMOTION_BAR_MS = 1.0


def load(variant: str) -> dict:
    path = ARTIFACTS / f"e220_census_{variant}.json"
    if not path.exists():
        raise SystemExit(f"missing census artifact {path}")
    return json.loads(path.read_text())


def verdict(base: dict, cand: dict, name: str) -> dict:
    cells = sorted(base["cells"])
    fp_base = {k: v["dequant_block"]["fp_sequence_sha8"]
               for k, v in base["instantiations"].items()}
    fp_cand = {k: v["dequant_block"]["fp_sequence_sha8"]
               for k, v in cand["instantiations"].items()}
    fp_identical = fp_base == fp_cand

    per_cell = {}
    text_deltas = {arch: [] for arch in ARCHES}
    identical_text = True
    new_spill = False
    for cell in cells:
        entry = {}
        for arch in ARCHES:
            b, c = base["cells"][cell][arch], cand["cells"][cell][arch]
            delta = c["text_bytes"] - b["text_bytes"]
            text_deltas[arch].append(100.0 * delta / b["text_bytes"])
            identical_text &= c["text_sha8"] == b["text_sha8"]
            new_spill |= c["spill_bytes"] > b["spill_bytes"]
            entry[arch] = {
                "base_text_bytes": b["text_bytes"],
                "text_bytes": c["text_bytes"],
                "text_delta_bytes": delta,
                "text_delta_pct": 100.0 * delta / b["text_bytes"],
                "text_identical": c["text_sha8"] == b["text_sha8"],
                "base_registers": b["registers"],
                "registers": c["registers"],
                "base_spill_bytes": b["spill_bytes"],
                "spill_bytes": c["spill_bytes"],
            }
        per_cell[cell] = entry

    mean_delta = {arch: statistics.fmean(text_deltas[arch]) for arch in ARCHES}
    same_size = all(
        cell[arch]["text_delta_bytes"] == 0
        for cell in per_cell.values() for arch in ARCHES)
    is_oracle = name in ORACLES
    if is_oracle:
        call = "cost oracle, not a candidate"
    elif not fp_identical:
        call = "dead: FP sequence changed"
    elif identical_text:
        call = "no-op: AGX machine code byte-identical on both generations"
    elif same_size:
        call = "no-op: same AGX instruction count, different encoding only"
    elif new_spill:
        call = "dead: new register spill"
    elif max(mean_delta.values()) >= 0:
        call = "dead: no fewer instructions than the base"
    else:
        call = "carry forward"
    return {
        "variant": name,
        "is_cost_oracle": is_oracle,
        "fp_sequence_identical": fp_identical,
        "agx_text_identical": identical_text,
        "introduces_spill": new_spill,
        "mean_text_delta_pct": mean_delta,
        "dequant_block_int_ops": {
            k: v["dequant_block"]["int_ops_total"]
            for k, v in cand["instantiations"].items()},
        "per_cell": per_cell,
        "verdict": call,
    }


def main() -> int:
    base = load(BASE)
    report = {
        "experiment": "e220-dequant-integer-path",
        "harness": "local",
        "measurement": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "base_source_sha8": base["source_sha8"],
        "width_plan_rule": base["width_plan_rule"],
        "reachable_instantiations": base["reachable_instantiations"],
        "base_dequant_block_sequence": base["instantiations"]["NA5_tbl"][
            "dequant_block_sequence"],
        "base_dequant_block_int_ops": base["instantiations"]["NA5_tbl"][
            "dequant_block"]["int_ops"],
        "variants": {},
    }
    for name in CANDIDATES + ORACLES:
        report["variants"][name] = verdict(base, load(name), name)

    # The prize: what a hypothetical one-operation-per-nibble extraction would
    # remove, priced from the oracle that keeps the FP tree intact.
    prize = report["variants"]["probe_min_extract"]["mean_text_delta_pct"]
    report["perfect_extraction_prize"] = {
        "definition": ("base minus probe_min_extract, mean over the scored "
                       "cells; the FP sequence digest is unchanged, so this "
                       "isolates the integer extraction"),
        "instruction_share_pct": {a: -v for a, v in prize.items()},
        "ms_per_round_on_b_term": {
            a: -v / 100.0 * B_TERM_MS_PER_ROUND for a, v in prize.items()},
        "ms_per_round_on_whole_na5_pass": {
            a: -v / 100.0 * B_TERM_MS_PER_ROUND / B_TERM_SHARE_OF_PASS
            for a, v in prize.items()},
        "promotion_bar_ms_per_round": PROMOTION_BAR_MS,
        "reachable": False,
        "why_unreachable": ("extract_bits, the only MSL spelling that can ask "
                            "for a single-instruction nibble extract, compiles "
                            "to byte-identical AGX machine code"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")

    print("base dequant block, per 4 packed values (AIR, NA5_tbl):")
    for op, count in report["base_dequant_block_int_ops"].items():
        print(f"  {op:6s} {count}")
    print("\nvariant verdicts:")
    print(f"  {'variant':18s} {'oracle':>6s} {'fp=':>4s} {'text=':>5s}"
          f" {'spill+':>6s} {'g16 d%':>7s} {'g17 d%':>7s}  verdict")
    for name, v in report["variants"].items():
        print(f"  {name:18s} {str(v['is_cost_oracle']):>6s}"
              f" {str(v['fp_sequence_identical']):>4s}"
              f" {str(v['agx_text_identical']):>5s}"
              f" {str(v['introduces_spill']):>6s}"
              f" {v['mean_text_delta_pct']['g16s']:+7.2f}"
              f" {v['mean_text_delta_pct']['g17s']:+7.2f}  {v['verdict']}")
    prize_block = report["perfect_extraction_prize"]
    print("\nprize from a perfect one-op-per-nibble extraction (UNREACHABLE):")
    for arch in ARCHES:
        print(f"  {arch}: {prize_block['instruction_share_pct'][arch]:.2f}% of "
              f"QMV machine code -> "
              f"{prize_block['ms_per_round_on_b_term'][arch]:.2f} ms/round on "
              f"the b term, "
              f"{prize_block['ms_per_round_on_whole_na5_pass'][arch]:.2f} "
              f"ms/round on the whole NA=5 pass "
              f"(bar {PROMOTION_BAR_MS:.1f})")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
