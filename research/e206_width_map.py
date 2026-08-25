#!/usr/bin/env python3
"""E206: cap-7 / cap-8 served-width censuses, m=9 work map, G=3 exposure.

Counts and desk arithmetic only. No timing contrast between caps or surfaces
(RULE 79 / FINDING 521). Every number is tagged harness=local.

Inputs:
  - E202 raw per-round traces (cap-7 surface), untracked, this host:
        research/out/e202/session-submit-abba1/<leg>/trace.txt
  - E199 cap-8 per-round W&B tables, vendored under research/e206-inputs/.

Output: research/e206-width-work-map.json
"""

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent
E202_SESSION = ROOT / "out" / "e202" / "session-submit-abba1"
INPUTS = ROOT / "e206-inputs"
OUT = ROOT / "e206-width-work-map.json"

# Fused routed QMV cells: (name, k, n, invocations per round).
CELLS = [
    ("mlp.gate_up", 5120, 34816, 64),
    ("mlp.down", 17408, 5120, 64),
    ("gdn.in_proj", 5120, 16480, 48),
    ("gdn.out_proj", 6144, 5120, 48),
    ("fa.qkv", 5120, 14336, 16),
    ("fa.o_proj", 6144, 5120, 16),
    ("lm_head", 5120, 248320, 1),
]
FAMILY = {
    "mlp.gate_up": "mlp",
    "mlp.down": "mlp",
    "gdn.in_proj": "gdn",
    "gdn.out_proj": "gdn",
    "fa.qkv": "fa",
    "fa.o_proj": "fa",
    "lm_head": "lm_head",
}
# affine 4-bit, group 64: 0.5 B/element payload + one bf16 scale and one bf16
# bias per 64 elements.
BYTES_PER_ELEMENT = 0.5 + 4.0 / 64.0

# Shipped staged QMV width plan, Qwen35.swift:1582.
STAGED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
# E195 selective plan, Qwen35.swift:1641: single pass at m == 6 only,
# and not for mlp.down / unlisted.
SINGLEPASS_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3}

# FINDING 485 / E197 local step law, harness=local, M4 Pro.
STEP_LAW = {"intercept_ms": 18.92, "per_row_ms": 6.51, "per_group_ms": 36.03}


def groups(m: int, ipg: int) -> int:
    return -(-m // ipg)


def shipped_groups(m: int, cell: str) -> int:
    if m == 6 and cell != "mlp.down":
        return groups(m, SINGLEPASS_IPG[m])
    return groups(m, STAGED_IPG[m])


def parse_e202_traces():
    """Cap-7 all-rounds served-width census from the raw E202 traces."""
    if not E202_SESSION.is_dir():
        return None
    field = re.compile(r"(\w+)=([^\s]+)")
    legs = {}
    for leg_dir in sorted(E202_SESSION.iterdir()):
        trace = leg_dir / "trace.txt"
        if not trace.is_file():
            continue
        census, accepted, proposed, rounds = {}, {}, {}, 0
        for line in trace.read_text().splitlines():
            if not line.startswith("mtp-trace: round="):
                continue
            rec = dict(field.findall(line))
            if rec.get("serial_body") == "1":
                continue
            d, acc = int(rec["d"]), int(rec["acc"])
            width = d + 1
            census[width] = census.get(width, 0) + 1
            proposed[width] = proposed.get(width, 0) + d
            accepted[width] = accepted.get(width, 0) + acc
            rounds += 1
        legs[leg_dir.name] = {
            "rounds": rounds,
            "census_by_served_width": census,
            "drafts_proposed_by_width": proposed,
            "drafts_accepted_by_width": accepted,
        }
    return legs


def load_cap8_tables():
    """Cap-8 per-round tables (round, drafts, rows, block_request_seconds)."""
    runs = {}
    for path in sorted(INPUTS.glob("cap8-per-round-*.json")):
        run_id = path.stem.rsplit("-", 1)[-1]
        table = json.loads(path.read_text())
        cols = table["columns"]
        rows = [dict(zip(cols, r)) for r in table["data"]]
        census, latency, drafts = {}, {}, {}
        for r in rows:
            w = int(r["rows"])
            census[w] = census.get(w, 0) + 1
            latency.setdefault(w, []).append(float(r["block_request_seconds"]))
            drafts[w] = drafts.get(w, 0) + int(r["drafts"])
        runs[run_id] = {
            "rounds": len(rows),
            "census_by_served_width": census,
            "drafts_proposed_by_width": drafts,
            "block_request_ms_mean_by_width": {
                w: 1000.0 * statistics.fmean(v) for w, v in latency.items()
            },
            "block_request_ms_total": 1000.0 * sum(sum(v) for v in latency.values()),
        }
    return runs


def weight_byte_table():
    per_cell, total = {}, 0.0
    for name, k, n, inv in CELLS:
        bytes_one = k * n * BYTES_PER_ELEMENT
        bytes_round = bytes_one * inv
        per_cell[name] = {
            "k": k,
            "n": n,
            "invocations_per_round": inv,
            "weight_bytes_per_invocation": bytes_one,
            "weight_bytes_per_round_per_pass": bytes_round,
            "family": FAMILY[name],
        }
        total += bytes_round
    for name, rec in per_cell.items():
        rec["share_of_pass"] = rec["weight_bytes_per_round_per_pass"] / total
    by_family = {}
    for name, rec in per_cell.items():
        fam = rec["family"]
        by_family[fam] = by_family.get(fam, 0.0) + rec["share_of_pass"]
    return per_cell, total, by_family


def m9_plan_map():
    """What each family runs at m=9 under the shipped plan, and the options."""
    cells = {}
    for name, k, n, inv in CELLS:
        cells[name] = {
            "cell_key_k_n": [k, n],
            "invocations_per_round": inv,
            "shipped_variant_at_m9": "staged",
            "shipped_ipg_at_m9": STAGED_IPG[9],
            "shipped_groups_at_m9": groups(9, STAGED_IPG[9]),
            "covered_by_e195_selective_plan_at_m9": False,
            "groups_at_m6_shipped": shipped_groups(6, name),
            "groups_at_m8_shipped": shipped_groups(8, name),
        }
    options = []
    for ipg in range(2, 10):
        tail = 9 % ipg
        legal = tail != 1
        options.append(
            {
                "ipg": ipg,
                "groups": groups(9, ipg),
                "tail_rows": tail,
                "legal_under_static_assert": legal,
                "max_ipg_proven_in_shipped_plan": ipg <= 6,
            }
        )
    return cells, options


def step_law_prediction(m, g):
    return (
        STEP_LAW["intercept_ms"]
        + STEP_LAW["per_row_ms"] * m
        + STEP_LAW["per_group_ms"] * g
    )


def main():
    cap7 = parse_e202_traces()
    cap8 = load_cap8_tables()
    per_cell, pass_bytes, family_share = weight_byte_table()
    m9_cells, m9_options = m9_plan_map()

    cap7_total = {}
    if cap7:
        for leg in cap7.values():
            for w, c in leg["census_by_served_width"].items():
                cap7_total[w] = cap7_total.get(w, 0) + c

    # Step-law consistency check against the cap-8 per-round latencies.
    # Within one run only; this is NOT a cap or policy contrast.
    law_check = []
    ref = cap8.get("7y6ap5l8")
    if ref:
        for w in sorted(ref["census_by_served_width"]):
            if w < 2:
                continue
            g_uniform = groups(w, STAGED_IPG[w])
            law_check.append(
                {
                    "served_width": w,
                    "rounds": ref["census_by_served_width"][w],
                    "measured_block_request_ms_mean": ref[
                        "block_request_ms_mean_by_width"
                    ][w],
                    "step_law_ms_uniform_staged_G": step_law_prediction(w, g_uniform),
                    "uniform_staged_G": g_uniform,
                }
            )

    doc = {
        "experiment": "e206-cap8-width-work-map",
        "harness": "local",
        "campaign_base": "40be5063a136725f0ce73ed81b2f8deac5ce184d",
        "notes": [
            "Counts and desk arithmetic only. No cap-8 vs cap-7 timing contrast.",
            "Cap-7 census surface: 7f10e147 (E202). Cap-8 census surface: "
            "c47c7284 = b9ee228c + segmentedVerifyDepthCap=8; b9ee228c has a "
            "zero-byte scored diff against the campaign base 40be5063.",
        ],
        "cap7_census": {
            "source": "E202 raw traces, W&B lpwbno36, 14 legs x 512 tokens",
            "per_leg": cap7,
            "total_by_served_width": cap7_total,
        },
        "cap8_census": {
            "source": "E199 W&B per_round tables (7y6ap5l8, exdb3vyt), "
            "512 tokens, gate-qualified",
            "runs": cap8,
        },
        "weight_byte_model": {
            "bytes_per_element_affine4_g64": BYTES_PER_ELEMENT,
            "per_cell": per_cell,
            "weight_bytes_per_round_per_pass": pass_bytes,
            "family_share_of_pass": family_share,
        },
        "m9_dispatch_map": {
            "shipped_staged_ipg_table": STAGED_IPG,
            "shipped_singlepass_ipg_table": SINGLEPASS_IPG,
            "e195_selective_predicate": "m == 6 && cell != .mlpDown "
            "(Qwen35.swift:1641)",
            "cells": m9_cells,
            "ipg_options_at_m9": m9_options,
        },
        "step_law": STEP_LAW,
        "step_law_consistency_check_within_7y6ap5l8": law_check,
    }
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")

    if cap7_total:
        print("\ncap-7 all-round served-width census (14 legs):")
        n7 = sum(cap7_total.values())
        for w in sorted(cap7_total):
            print(f"  qL={w}: {cap7_total[w]:5d}  {100*cap7_total[w]/n7:6.2f}%")
    print("\ncap-8 served-width census:")
    for run_id, rec in cap8.items():
        n8 = rec["rounds"]
        row = "  ".join(
            f"qL{w}={rec['census_by_served_width'][w]}"
            for w in sorted(rec["census_by_served_width"])
        )
        print(f"  {run_id} ({n8} rounds): {row}")
    print(f"\nweight bytes per round per pass: {pass_bytes/1e9:.3f} GB")
    for fam, share in sorted(family_share.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:8s} {100*share:5.1f}%")
    print("\nstep-law check inside 7y6ap5l8 (harness=local, within one leg):")
    for rec in law_check:
        print(
            f"  qL={rec['served_width']} n={rec['rounds']:3d} "
            f"measured={rec['measured_block_request_ms_mean']:7.2f} ms  "
            f"law(G={rec['uniform_staged_G']})="
            f"{rec['step_law_ms_uniform_staged_G']:7.2f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
