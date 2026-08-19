#!/usr/bin/env python3
"""Research-only (qwen38-r1-e55-compose-m9-two-stream-on-shipped-table).

Answers the two corrections in advisor comment 12 with data that already
exists, no GPU second.

Section A -- the seed prefill is inside MY local timed leg too.
  The advisor established that the ranked driver starts its clock before the
  512-token prefill, so ranked raw_p carries a dilution of 8.44..9.05 %. My own
  captured `mtp-timed` reports carry `seed_prefill_seconds` and per-round
  `block_request_seconds`, so the same question is decidable on my instrument
  from data I already hold. It is: the local MTP leg dilution is ~24 %, roughly
  3x the ranked figure. That makes the undiluted round-cost change directly
  measurable and it changes the conversion the advisor asked me to state.

Section B -- the register census obeys an exact law with zero free parameters.
  The advisor's §4 hard gate asks for one integer and calls it the most
  valuable number a student can return this round. Combining the E32 affine
  ladder he quoted with the census's own `peak_live_values` field yields
  reg = 20 + 21*max(NA) + 4*[two distinct NA groups], which reproduces all six
  distinct observed configurations exactly. The law then answers the gate for
  the whole NA=5 family rather than for this candidate alone.

Inputs are the committed census artifact and the private per-leg timed reports.
The extracted per-leg scalars are copied into the output artifact so the
analysis is reproducible from the artifact alone.

  python3 research/e55_comment12_corrections.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CENSUS = REPO / "research" / "e55-reg-census.json"
RUNS = REPO / ".mlxfast-private" / "e55" / "runs"
OUT = REPO / "research" / "e55-comment12-corrections.json"

ARMS = ("base", "m9two", "base2")
LEGS = (1, 2)
# 03 is the depth-0 serial control, 04 the speculative candidate leg. Read the
# discriminator out of the report rather than trusting the file number.
PHASE_FILES = {"serial": "03-mtp-timed.json", "mtp": "04-mtp-timed.json"}

PSI_MTP = 0.693391
M9_CELL_WIN_PCT = 12.255
NULL_MTP_PCT = 0.0497
NULL_SERIAL_PCT = -0.0133
NULL_GUARD_MULTIPLE = 3.0

# Advisor comment 12 §1, from the ranked receipt.
RANKED = {
    "beagle": {"leg_ms": 6233.1, "k_ms": 525.96, "dilution": 0.91552},
    "medicine": {"leg_ms": 5820.7, "k_ms": 527.04, "dilution": 0.90953},
}
RANKED_K_MS_RANGE = (525.7, 528.2)
RANKED_SEED_TOKENS = 512
# Advisor comment 12 §5 / ledger 186(D).
TRANSFER_DIVISOR = 3.55
ADVISOR_PREFILL_RATIO = 7.58
ADVISOR_DECODE_ROUND_RATIO = 2.14
# Advisor comment 12 §4, from E32.
LADDER_R4 = (20.0, 21.0)
LADDER_R2 = (16.0, 15.0)
# Advisor comment 12 §3 prereg column, local MTP leg.
PREREG_LOCAL = {"e48_mixture": -1.838, "edward_upper": -0.756, "edward_lower": -0.391}
PREREG_F9 = {"e48_mixture": 21.630, "edward_upper": 8.9, "edward_lower": 4.6}


def fail(msg: str) -> None:
    print(f"e55-comment12: {msg}", file=sys.stderr)
    sys.exit(1)


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


# ---------------------------------------------------------------- section A

def read_leg(arm: str, leg: int, phase: str) -> dict:
    path = RUNS / arm / "reports" / f"leg-{leg}" / PHASE_FILES[phase]
    if not path.is_file():
        fail(f"missing timed report {path}")
    r = json.loads(path.read_text())
    want_serial = phase == "serial"
    if bool(r["is_serial_control"]) != want_serial:
        fail(f"{path} is_serial_control={r['is_serial_control']} but read as {phase}")
    blocks = r["block_request_seconds"]
    if len(blocks) != int(r["round_count"]):
        fail(f"{path} has {len(blocks)} block times for {r['round_count']} rounds")
    return {
        "decode_seconds": float(r["decode_seconds"]),
        "seed_prefill_seconds": float(r["seed_prefill_seconds"]),
        "prefill_seconds_per_token": float(r["prefill_seconds_per_token"]),
        "seed_token_count": int(r["seed_token_count"]),
        "decode_token_count": int(r["decode_token_count"]),
        "round_count": int(r["round_count"]),
        "declared_rows_total": int(r["declared_rows_total"]),
        "sum_block_seconds": float(sum(blocks)),
        "parent_measured_seconds_per_token": float(r["parent_measured_seconds_per_token"]),
        "all_tokens_matched": bool(r["all_tokens_matched"]),
    }


legs: dict[str, dict] = {}
for arm in ARMS:
    for leg in LEGS:
        for phase in PHASE_FILES:
            legs[f"{arm}/leg-{leg}/{phase}"] = read_leg(arm, leg, phase)

# The containment test. If the prefill is inside decode_seconds then
# decode - prefill - sum(blocks) is a small non-negative parent overhead; if it
# is outside then decode - sum(blocks) is.
containment = {}
for key, d in legs.items():
    inside = d["decode_seconds"] - d["seed_prefill_seconds"] - d["sum_block_seconds"]
    outside = d["decode_seconds"] - d["sum_block_seconds"]
    containment[key] = {
        "residual_if_prefill_inside_s": inside,
        "residual_if_prefill_outside_s": outside,
        "residual_inside_frac_of_leg": inside / d["decode_seconds"],
        "residual_outside_frac_of_leg": outside / d["decode_seconds"],
        "inside_reading_is_tighter": abs(inside) < abs(outside),
        "inside_residual_is_non_negative": inside >= 0.0,
        # The advisor's K = seed_tokens x prefill_seconds_per_token.
        "advisor_K_seconds": d["seed_token_count"] * d["prefill_seconds_per_token"],
        "advisor_K_matches_seed_prefill_rel": abs(
            d["seed_token_count"] * d["prefill_seconds_per_token"] - d["seed_prefill_seconds"]
        ) / d["seed_prefill_seconds"],
    }

prefill_is_inside = all(
    c["inside_reading_is_tighter"] and c["inside_residual_is_non_negative"]
    for c in containment.values()
)
advisor_k_formula_holds = all(
    c["advisor_K_matches_seed_prefill_rel"] < 5e-3 for c in containment.values()
)


def arm_phase_mean(arm: str, phase: str, field: str) -> float:
    vals = [legs[f"{arm}/leg-{leg}/{phase}"][field] for leg in LEGS]
    return sum(vals) / len(vals)


# Per-arm leg totals and round-cost totals. `round_cost` is the sum of the
# per-round block request times, i.e. the leg with the prefill and the parent
# overhead removed. It is the quantity a per-round kernel change acts on.
arm_stats = {}
for arm in ARMS:
    entry = {}
    for phase in PHASE_FILES:
        leg_s = arm_phase_mean(arm, phase, "decode_seconds")
        pf_s = arm_phase_mean(arm, phase, "seed_prefill_seconds")
        rc_s = arm_phase_mean(arm, phase, "sum_block_seconds")
        tok = int(arm_phase_mean(arm, phase, "decode_token_count"))
        entry[phase] = {
            "leg_seconds": leg_s,
            "seed_prefill_seconds": pf_s,
            "round_cost_seconds": rc_s,
            "parent_overhead_seconds": leg_s - pf_s - rc_s,
            "seconds_per_token": leg_s / tok,
            "round_cost_seconds_per_token": rc_s / tok,
            "prefill_frac_of_leg": pf_s / leg_s,
            "round_cost_frac_of_leg": rc_s / leg_s,
            "dilution": 1.0 - pf_s / leg_s,
        }
    arm_stats[arm] = entry

# base and base2 bracket the candidate in time; their mean is the baseline.
base_bracket = {}
for phase in PHASE_FILES:
    for field in arm_stats["base"][phase]:
        base_bracket.setdefault(phase, {})[field] = (
            arm_stats["base"][phase][field] + arm_stats["base2"][phase][field]
        ) / 2.0

effects = {}
for phase in PHASE_FILES:
    cand, base_b = arm_stats["m9two"][phase], base_bracket[phase]
    leg_pct = pct(cand["leg_seconds"], base_b["leg_seconds"])
    rc_pct = pct(cand["round_cost_seconds"], base_b["round_cost_seconds"])
    # Algebraic undilution assumes the prefill is exactly common mode. It is
    # not: the prefill jitters by ~0.2 % between legs. The direct sum of the
    # per-round block times needs no such assumption, so it is the headline;
    # the recovery is kept only to check the accounting.
    recovered = leg_pct / base_b["dilution"]
    d_prefill = cand["seed_prefill_seconds"] - base_b["seed_prefill_seconds"]
    predicted_gap_pp = d_prefill / base_b["round_cost_seconds"] * 100.0
    effects[phase] = {
        "leg_pct": leg_pct,
        "round_cost_pct": rc_pct,
        "round_cost_pct_recovered_from_leg": recovered,
        "identity_gap_pp": recovered - rc_pct,
        # The whole gap must be the arm-to-arm prefill difference, which proves
        # the leg has no term beyond prefill + rounds + negligible overhead.
        "identity_gap_pp_predicted_from_prefill_jitter": predicted_gap_pp,
        "identity_gap_unexplained_pp": abs((recovered - rc_pct) - predicted_gap_pp),
        "amplification_round_over_leg": (
            rc_pct / leg_pct if abs(leg_pct) > NULL_GUARD_MULTIPLE * NULL_MTP_PCT else None
        ),
        "base_null_leg_pct": pct(
            arm_stats["base2"][phase]["leg_seconds"], arm_stats["base"][phase]["leg_seconds"]
        ),
        "base_null_round_cost_pct": pct(
            arm_stats["base2"][phase]["round_cost_seconds"],
            arm_stats["base"][phase]["round_cost_seconds"],
        ),
        # The prefill runs qmm, not the qmv_fast switch this candidate edits, so
        # its arm-to-arm change is a third null measured inside every leg.
        "prefill_pct": pct(cand["seed_prefill_seconds"], base_b["seed_prefill_seconds"]),
        "base_null_prefill_pct": pct(
            arm_stats["base2"][phase]["seed_prefill_seconds"],
            arm_stats["base"][phase]["seed_prefill_seconds"],
        ),
        "parent_overhead_frac_of_leg": base_b["parent_overhead_seconds"] / base_b["leg_seconds"],
    }

round_cost_identity_agrees = all(
    e["identity_gap_unexplained_pp"] < 5e-3 for e in effects.values()
)
serial_falsifier_holds = abs(effects["serial"]["round_cost_pct"]) < NULL_GUARD_MULTIPLE * 0.05
prefill_falsifier_holds = all(
    abs(e["prefill_pct"]) < 1.0 for e in effects.values()
)

# Local versus ranked dilution, and the prefill transfer ratio.
local_pf_spt = sum(
    d["prefill_seconds_per_token"] for d in legs.values()
) / len(legs)
ranked_pf_spt_range = (
    RANKED_K_MS_RANGE[0] / 1000.0 / RANKED_SEED_TOKENS,
    RANKED_K_MS_RANGE[1] / 1000.0 / RANKED_SEED_TOKENS,
)
prefill_ratio_range = (
    local_pf_spt / ranked_pf_spt_range[1],
    local_pf_spt / ranked_pf_spt_range[0],
)
ranked_dilution_mean = sum(v["dilution"] for v in RANKED.values()) / len(RANKED)

dilution_block = {
    "local_mtp_leg_dilution": base_bracket["mtp"]["dilution"],
    "local_mtp_prefill_frac_of_leg": base_bracket["mtp"]["prefill_frac_of_leg"],
    "local_serial_leg_dilution": base_bracket["serial"]["dilution"],
    "local_serial_prefill_frac_of_leg": base_bracket["serial"]["prefill_frac_of_leg"],
    "ranked_dilution_beagle": RANKED["beagle"]["dilution"],
    "ranked_dilution_medicine": RANKED["medicine"]["dilution"],
    "ranked_dilution_mean": ranked_dilution_mean,
    "local_over_ranked_dilution_penalty": ranked_dilution_mean / base_bracket["mtp"]["dilution"],
    "local_prefill_seconds_per_token": local_pf_spt,
    "ranked_prefill_seconds_per_token_range": list(ranked_pf_spt_range),
    "prefill_transfer_ratio_range": list(prefill_ratio_range),
    "advisor_prefill_transfer_ratio": ADVISOR_PREFILL_RATIO,
    "prefill_transfer_ratio_rel_gap_vs_advisor": min(
        abs(r - ADVISOR_PREFILL_RATIO) / ADVISOR_PREFILL_RATIO for r in prefill_ratio_range
    ),
}

# The conversion the advisor asked me to state, under both readings of where
# the local dilution already sits. I report both rather than choose.
measured_leg_pct = effects["mtp"]["leg_pct"]
measured_rc_pct = effects["mtp"]["round_cost_pct"]
conversion = {
    "reading_L_literal": {
        "what": "advisor comment 12 §3: score = measured local leg change x ranked dilution",
        "dscore_pct": -measured_leg_pct * ranked_dilution_mean,
        "dscore_pct_transfer_floor": -measured_leg_pct * ranked_dilution_mean / TRANSFER_DIVISOR,
    },
    "reading_R_undiluted": {
        "what": "undilute the local leg to round cost, then apply the ranked dilution",
        "dscore_pct": -measured_rc_pct * ranked_dilution_mean,
        "dscore_pct_transfer_floor": -measured_rc_pct * ranked_dilution_mean / TRANSFER_DIVISOR,
    },
    "readings_differ_by_factor": (measured_rc_pct / measured_leg_pct) if measured_leg_pct else None,
    "note": (
        "psi_mtp was calibrated on a local leg that this artifact now shows carries "
        "a ~24 % prefill dilution. Reading L therefore charges a local dilution and a "
        "ranked dilution to the same quantity unless psi_mtp is already a round-cost "
        "coefficient. I cannot settle which from the numbers I hold; the advisor owns psi_mtp."
    ),
}

# f9 back-solve, now carrying the ranked dilution the advisor added.
def f9_from_dscore(ds_pct: float) -> float:
    return ds_pct / (PSI_MTP * M9_CELL_WIN_PCT / 100.0 * ranked_dilution_mean)


f9_backsolve = {
    "sensitivity_dscore_pct_per_unit_f9": PSI_MTP * M9_CELL_WIN_PCT * ranked_dilution_mean,
    "implied_f9_pct_for_dscore_4p5": f9_from_dscore(4.5),
    "implied_f9_pct_for_dscore_4p7": f9_from_dscore(4.7),
    "implied_f9_pct_for_e49_m9_half_1p3625": f9_from_dscore(1.3625),
    "e48_mixture_f9_pct": PREREG_F9["e48_mixture"],
    "edward_upper_f9_pct": PREREG_F9["edward_upper"],
    "note": (
        "Adding the ranked dilution RAISES the ranked f9 implied by the quoted "
        "+4.5..+4.7 % prize, so the §5a tension I flagged widens rather than closes."
    ),
}

prereg_check = {}
for name, predicted in PREREG_LOCAL.items():
    prereg_check[name] = {
        "predicted_local_leg_pct": predicted,
        "measured_local_leg_pct": measured_leg_pct,
        "residual_pp": measured_leg_pct - predicted,
        "measured_over_predicted": measured_leg_pct / predicted,
    }

# ---------------------------------------------------------------- section B

census = json.loads(CENSUS.read_text())
arms_by_name = {a["name"]: a for a in census["arms"] if a.get("status") == "ok"}

# Collect every distinct (set of NA group values) -> register observation the
# census contains, across all arms, and check for internal contradictions.
observations: dict[tuple, dict] = {}
for name, arm in arms_by_name.items():
    for width, cell in arm["width_cells"].items():
        key = tuple(sorted(cell["na_cells"]))
        rec = {
            "peak_live_regs": int(cell["peak_live_regs"]),
            "peak_live_values": int(cell["peak_live_values"]),
            "allocas": int(cell["allocas"]),
        }
        prev = observations.get(key)
        if prev is None:
            observations[key] = dict(rec, seen_in=[f"{name}:M{width}"])
        else:
            if (prev["peak_live_regs"], prev["peak_live_values"], prev["allocas"]) != (
                rec["peak_live_regs"], rec["peak_live_values"], rec["allocas"]
            ):
                fail(f"census contradicts itself for NA groups {key}")
            prev["seen_in"].append(f"{name}:M{width}")

# The mixed-group penalty is not a fitted constant: it is the census's own
# peak_live_values delta between the uniform and the mixed configurations.
uniform_plv = sorted({o["peak_live_values"] for k, o in observations.items() if len(k) == 1})
mixed_plv = sorted({o["peak_live_values"] for k, o in observations.items() if len(k) > 1})
if len(uniform_plv) != 1 or len(mixed_plv) != 1:
    fail(f"peak_live_values is not constant within each group class: {uniform_plv} {mixed_plv}")
mixed_penalty = mixed_plv[0] - uniform_plv[0]
uniform_allocas = sorted({o["allocas"] for k, o in observations.items() if len(k) == 1})
mixed_allocas = sorted({o["allocas"] for k, o in observations.items() if len(k) > 1})


def predict_regs(na_groups, ladder=LADDER_R4, penalty=None) -> float:
    intercept, slope = ladder
    p = mixed_penalty if penalty is None else penalty
    key = tuple(sorted(set(na_groups)))
    return intercept + slope * max(key) + (p if len(key) > 1 else 0)


law_rows = []
for key, obs in sorted(observations.items()):
    pred = predict_regs(key)
    law_rows.append({
        "na_groups": list(key),
        "class": "uniform" if len(key) == 1 else "mixed",
        "observed_regs": obs["peak_live_regs"],
        "predicted_regs": pred,
        "residual": obs["peak_live_regs"] - pred,
        "peak_live_values": obs["peak_live_values"],
        "allocas": obs["allocas"],
        "seen_in": obs["seen_in"],
    })
law_exact = all(row["residual"] == 0 for row in law_rows)
law_max_abs_residual = max(abs(row["residual"]) for row in law_rows)

# Negative control: the r=2 ladder the advisor also quoted must NOT fit.
r2_rows = [
    {
        "na_groups": list(k),
        "observed_regs": o["peak_live_regs"],
        "predicted_regs": predict_regs(k, ladder=LADDER_R2),
    }
    for k, o in sorted(observations.items())
]
r2_max_abs_residual = max(abs(r["observed_regs"] - r["predicted_regs"]) for r in r2_rows)

# Negative control: perturb one observation and the law must break.
perturbed_key = (4, 5)
perturbed_breaks = (
    observations[perturbed_key]["peak_live_regs"] + 1 - predict_regs(perturbed_key)
) != 0

# What the law says about the §4 hard gate for the WHOLE NA=5 family.
BASE_CEILING = int(census["kernel_wide_reg_max"]["base_na4_table"])
CAND_CEILING = int(census["kernel_wide_reg_max"]["m9two_candidate"])
E27_CEILING = int(census["kernel_wide_reg_max"]["e27_both_cells"])


def groups_for(m: int, ipg: int) -> list[int]:
    out, left = [], m
    while left > 0:
        take = min(ipg, left)
        out.append(take)
        left -= take
    return out


na5_family = []
for m in range(3, 10):
    g = groups_for(m, 5)
    if 5 not in g:
        continue
    na5_family.append({
        "width": m,
        "groups_at_ipg5": g,
        "class": "uniform" if len(set(g)) == 1 else "mixed",
        # The wide helper still asserts NA >= 2, so a tail group of one row does
        # not compile without a second, separate relaxation.
        "buildable_under_na_ge_2": min(g) >= 2,
        "predicted_cell_regs": predict_regs(g),
    })
min_na5_cell = min(
    r["predicted_cell_regs"] for r in na5_family if r["buildable_under_na_ge_2"]
)
gate_block = {
    "advisor_gate_target_regs": 108,
    "base_kernel_wide_reg_max": BASE_CEILING,
    "candidate_kernel_wide_reg_max": CAND_CEILING,
    "e27_kernel_wide_reg_max": E27_CEILING,
    "candidate_reads_108": CAND_CEILING == 108,
    "candidate_reads_129": CAND_CEILING == 129,
    "branch": (
        "A_ceiling_held_transfer_expected" if CAND_CEILING == BASE_CEILING
        else "B_e27_confound_reproduced_report_and_stop"
    ),
    "na5_family": na5_family,
    "min_cell_regs_over_any_na5_width": min_na5_cell,
    # Any table holding at least one NA=5 group has a kernel-wide max at least
    # as large as that group's own cell, so the gate is unreachable for the
    # whole family, not just for this candidate.
    "no_na5_table_can_read_108": min_na5_cell > 108,
    "only_uniform_na5_width_in_scored_range": [
        r["width"] for r in na5_family if r["class"] == "uniform"
    ],
    "predicted_ceiling_for_case7_or_case8_na5": predict_regs(groups_for(7, 5)),
}

# ---------------------------------------------------------------- verdict

controls = {
    "nc_prefill_inside_reading_is_tighter_on_every_leg": prefill_is_inside,
    "nc_advisor_K_formula_reproduces_seed_prefill": advisor_k_formula_holds,
    "nc_leg_budget_closes_prefill_plus_rounds_only": round_cost_identity_agrees,
    "nc_serial_round_cost_falsifier_holds": serial_falsifier_holds,
    "nc_prefill_falsifier_holds": prefill_falsifier_holds,
    "nc_local_dilution_is_not_the_ranked_dilution": (
        abs(base_bracket["mtp"]["dilution"] - ranked_dilution_mean) > 0.05
    ),
    "nc_prefill_transfer_ratio_matches_advisor_within_5pct": (
        dilution_block["prefill_transfer_ratio_rel_gap_vs_advisor"] < 0.05
    ),
    "nc_register_law_is_exact_on_every_configuration": law_exact,
    "nc_mixed_penalty_equals_peak_live_values_delta": mixed_penalty == 4,
    "nc_r2_ladder_does_not_fit": r2_max_abs_residual > 5,
    "nc_law_rejects_a_perturbed_observation": perturbed_breaks,
    "nc_gate_branch_is_B_not_A": gate_block["branch"].startswith("B_"),
    "nc_no_na5_table_can_read_108": gate_block["no_na5_table_can_read_108"],
    "nc_all_arms_matched_tokens": all(d["all_tokens_matched"] for d in legs.values()),
}
verdict_ok = all(controls.values())

payload = {
    "experiment": "qwen38-r1-e55-compose-m9-two-stream-on-shipped-table",
    "answers_advisor_comment": 5347182851,
    "host": "Apple M4 Pro",
    "gpu_seconds_spent": 0,
    "section_a_prefill_dilution": {
        "prefill_is_inside_the_local_timed_leg": prefill_is_inside,
        "per_leg_scalars": legs,
        "containment": containment,
        "arm_stats": arm_stats,
        "base_bracket": base_bracket,
        "effects": effects,
        "dilution": dilution_block,
        "conversion": conversion,
        "f9_backsolve": f9_backsolve,
        "prereg_check_local_leg": prereg_check,
    },
    "section_b_register_law": {
        "law": "peak_live_regs = 20 + 21*max(NA) + 4*[two distinct NA group sizes]",
        "ladder_from_advisor_e32_r4": {"intercept": LADDER_R4[0], "slope": LADDER_R4[1]},
        "mixed_penalty": mixed_penalty,
        "mixed_penalty_source": (
            f"census peak_live_values {uniform_plv[0]} uniform vs {mixed_plv[0]} mixed; "
            f"allocas {uniform_allocas} vs {mixed_allocas}"
        ),
        "free_parameters_fitted_by_me": 0,
        "rows": law_rows,
        "exact": law_exact,
        "max_abs_residual": law_max_abs_residual,
        "r2_ladder_rows": r2_rows,
        "r2_ladder_max_abs_residual": r2_max_abs_residual,
        "hard_gate": gate_block,
    },
    "negative_controls": controls,
    "verdict_ok": verdict_ok,
}

OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(f"prefill_is_inside_local_leg={prefill_is_inside}")
print(f"local_mtp_dilution={base_bracket['mtp']['dilution']:.6f} "
      f"(prefill is {base_bracket['mtp']['prefill_frac_of_leg']*100:.3f} % of the leg)")
print(f"ranked_mtp_dilution_mean={ranked_dilution_mean:.6f}")
print(f"measured_mtp_leg_pct={measured_leg_pct:+.4f}")
print(f"measured_mtp_round_cost_pct={measured_rc_pct:+.4f} "
      f"(recovered {effects['mtp']['round_cost_pct_recovered_from_leg']:+.4f}, "
      f"gap {effects['mtp']['identity_gap_pp']:+.4f} pp, "
      f"unexplained {effects['mtp']['identity_gap_unexplained_pp']:.2e} pp)")
print(f"serial_round_cost_pct={effects['serial']['round_cost_pct']:+.4f} "
      f"prefill_pct={effects['mtp']['prefill_pct']:+.4f}")
print(f"prefill_transfer_ratio={prefill_ratio_range[0]:.3f}..{prefill_ratio_range[1]:.3f} "
      f"vs advisor {ADVISOR_PREFILL_RATIO}")
print(f"register_law_exact={law_exact} max_abs_residual={law_max_abs_residual} "
      f"mixed_penalty={mixed_penalty}")
print(f"gate_branch={gate_block['branch']} candidate_max={CAND_CEILING} base_max={BASE_CEILING}")
print(f"no_na5_table_can_read_108={gate_block['no_na5_table_can_read_108']} "
      f"min_na5_cell={min_na5_cell}")
for k, v in controls.items():
    if not v:
        print(f"CONTROL FAILED: {k}", file=sys.stderr)
print(f"verdict_ok={verdict_ok}")
print(f"wrote {OUT.relative_to(REPO)}")
sys.exit(0 if verdict_ok else 1)
