#!/usr/bin/env python3
"""Read the E139 zero-noise acceptance channel and price the two held riders.

    usage: research/e139_analyse.py [--runs DIR] [--json OUT]

THE CHANNEL. Every leg decodes a fixed 512-token window against a reference
trajectory generated once, on this base, under the shipped default. The
quantities this script reads -- `round_count`, `accepted_draft_total`,
`effective_mean_draft_len` -- are pure functions of (arm, fixture, window,
offered depth). They carry no thermal, scheduling or clock noise, so ONE leg
per arm resolves them exactly and a second leg of the same arm must be
digit-identical.

WHAT THE CHANNEL COSTS. The parent counts exactly 512 emitted tokens and each
round emits one primary token plus its accepted drafts, so

    round_count + accepted_draft_total = 512 or 513

(513 when the window closes inside a round and its last accepted draft is not
emitted). Round count and accepted tokens are therefore the SAME observable
up to that one token, and the channel's finest step is one round. Its
resolution is

    quantisation_floor_pct = 100 / round_count_ship

which is 1.28 % on a fixture that drafts 6.36 deep and 0.385 % on one that
drafts 2.20 deep. Zero variance is not the same as fine resolution: the
channel can prove a null to within one round and no finer.

PRICING (CAMPAIGN RULE 107 net, CAMPAIGN RULE 116 weighting, harness=ranked).
For each rider,

    acceptance_cost_pct = 100 * (R_arm - R_ship) / R_ship
    net_median_pct      = (gross_byte_pct_local
                           - acceptance_cost_pct * ACCEPTANCE_TRANSFER)
                          * LOCAL_GROSS_TO_MEDIAN_GAIN

RULE 116. The published score is `median(raw_1 .. raw_8)`, which is the mean
of the 4th and 5th sorted values, so only `beagle` and `essays` carry any
marginal weight. The probe fraction removes bytes from every draft step on
every prompt over the same 12,292 centroid rows, so its RELATIVE effect is
uniform across prompts. Multiplying all eight raw ratios by a common factor
leaves the sorted order unchanged, so the median scales by that same factor.
This mechanism class needs no per-prompt weighting at all: one multiply.

A width-concentrated mechanism does NOT have that property and must be
repriced by sorting. Do not reuse this constant for one.

The acceptance term is a round-count ratio, a structural quantity with no
local/ranked byte content, so it is carried at 1.0 and that assumption is
printed, never hidden.

ASSUMPTION, STATED. `acceptance_cost_pct` treats every round as costing the
same. It does not: a round that verifies more rows costs more. Reading the
round-count ratio as a time ratio is therefore first-order only, and the
output names the direction of that error per rider.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# E139 F2 / FINDING 192 retired the 1.3x frame haircut FOR THIS FAMILY, and
# E139 F4 / FINDING 196 replaced what is left of it with a single measured
# transfer. The haircut was proposed because local `benchfixture` drafts 6.359
# deep against a ranked frame near 4.9, which would predict a measured/model
# ratio near 0.77. Two independent receipts for the same one-constant change,
# read on the median-pair identity, give 0.951 against the UNSCALED byte
# model. So the draft-path byte model transfers essentially intact and the
# whole local-to-published conversion is this one multiply.
LOCAL_GROSS_TO_MEDIAN_GAIN = 0.95

# A round-count ratio has no byte content and no draft-path share to shrink,
# so it is carried across harnesses at 1.0. Printed so a reviewer can reject
# it.
ACCEPTANCE_TRANSFER = 1.0

# Byte model for the probe-fraction ladder, local per-cent of the candidate
# leg, gross of any acceptance cost. Unscaled. Every entry is the measured
# `pct_head_share_7` of that rung minus the same field at p=0.25, read from
# research/e139-probe-ladder-screen.json, so this table IS the screen's own
# byte accounting rather than a fit to it.
GROSS_BYTE_PCT_LOCAL = {
    "unset": 0.0,
    "0.25": 0.0000, "0.20": 0.1700, "0.175": 0.2550, "0.15": 0.3403,
    "0.125": 0.4253, "0.10": 0.5103, "0.09": 0.5444, "0.08": 0.5784,
    "0.075": 0.5956, "0.07": 0.6125, "0.06": 0.6465, "0.05": 0.6806,
    "0.04": 0.7147, "0.03": 0.7487, "0.02": 0.7828, "0.015": 0.7997,
    "0.01": 0.8168, "0.0075": 0.8251, "0.005": 0.8337,
}

# The fp32 rerank tiebreak changes ARITHMETIC only: the same float lands in
# the same float slot of the same buffer. It moves no bytes, so its whole
# value is whatever it does to acceptance.
FP32_GROSS_BYTE_PCT_LOCAL = 0.0

# The rival's receipts for the probe rider at p=0.15, the number this channel
# was built to reconstruct. Two independent receipts on two different bases,
# read off the CANDIDATE leg rather than the raw serial/candidate ratio
# (Rule 112: the pinned serial leg is 9.6 times noisier and importing it buys
# nothing for a candidate-side mechanism), and evaluated on the RULE 116
# median-pair identity rather than the withdrawn F83 weights. Supplied by
# E139 F4 / FINDING 196.
RIVAL_MEDIAN_RECEIPTS_AT_P015 = {
    "b6cb0fea->02742bf0 wide-grid base": 0.3866,
    "ed608e64->08b67f12 tight-grid base": 0.2603,
}
RIVAL_MEDIAN_PCT_AT_P015 = 0.3235
# F2 supplied 0.0948 for the F83 reading of the same two receipts. The
# median-pair reading has a wider spread between them, 0.1263 against 0.1050,
# so the same derivation gives a proportionally wider interval. Scaled rather
# than reused, and the wider value is the one carried.
RIVAL_MEDIAN_2SIGMA_AT_P015 = 0.0948 * (0.1263 / 0.1050)


def arm_probe_key(arm: str) -> str:
    """`p015` -> `0.15`, `fp32p002` -> `0.02`, `ship` and `fp32` -> `unset`."""
    if arm in ("ship", "fp32"):
        return "unset"
    for prefix in ("fp32p0", "p0"):
        if arm.startswith(prefix):
            return f"0.{arm[len(prefix):]}"
    raise KeyError(f"unknown arm {arm}")


def arm_is_fp32(arm: str) -> bool:
    return arm == "fp32" or arm.startswith("fp32p")

# The digits that must repeat exactly across two legs of the same arm.
CHANNEL_KEYS = ["round_count", "accepted_draft_total", "rejected_draft_total",
                "effective_mean_draft_len", "effective_max_draft_len",
                "accepted_draft_rate", "declared_rows_total"]

PROVENANCE_KEYS = ["base_sha", "worker_sha256", "worker_sha256_after_leg",
                   "head_manifest_tree_sha256", "dirty_candidate_paths",
                   "host", "memory_gib", "cli_sha256"]


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def load_legs(runs: pathlib.Path) -> list[dict]:
    legs = []
    for d in sorted(runs.iterdir()):
        meta_path = d / "meta.txt"
        if not meta_path.is_file():
            continue
        meta = read_meta(meta_path)
        report_path = d / "report.json"
        report = (json.loads(report_path.read_text())
                  if report_path.is_file() else {})
        # meta.txt records what the session asked for; these come from the
        # parent's own accounting and are the row-side view of the same
        # channel.
        for k in ("decode_token_count", "emitted_token_total",
                  "declared_rows_total", "reference_checked_row_total",
                  "non_drafting_round_count", "parity_all_ok"):
            if k in report:
                meta[k] = str(report[k])
        meta["report"] = report
        meta["dir"] = d.name
        legs.append(meta)
    return legs


def num(meta: dict, key: str):
    v = meta.get(key)
    if v in (None, "", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None


def leg_key(meta: dict) -> tuple:
    return (meta["arm"], meta["prompt_id"], meta["tokens"],
            meta["offered_depth"])


def check_legs(legs: list[dict]) -> list[str]:
    bad = []
    for m in legs:
        for got, want, name in (("witness_fp32_gate", "expected_fp32_gate",
                                 "fp32 gate"),
                                ("witness_probe_gate", "expected_probe_gate",
                                 "probe gate"),
                                ("witness_probes", "expected_probes",
                                 "probes")):
            if m.get(got) != m.get(want):
                bad.append(f"{m['dir']}: {name} witnessed {m.get(got)}, "
                           f"wanted {m.get(want)}")
        if m.get("all_tokens_matched") != "true":
            bad.append(f"{m['dir']}: all_tokens_matched="
                       f"{m.get('all_tokens_matched')}")
        if m.get("residual_divergence_count") != "0":
            bad.append(f"{m['dir']}: residual_divergence_count="
                       f"{m.get('residual_divergence_count')}")
        if m.get("worker_sha256") != m.get("worker_sha256_after_leg"):
            bad.append(f"{m['dir']}: the worker moved during the leg")
        if m.get("exit") != "0":
            bad.append(f"{m['dir']}: exit={m.get('exit')}")
        if m.get("dirty_candidate_paths") != "0":
            bad.append(f"{m['dir']}: dirty_candidate_paths="
                       f"{m.get('dirty_candidate_paths')}")
        if m.get("decode_token_count") != m.get("tokens"):
            bad.append(f"{m['dir']}: decoded "
                       f"{m.get('decode_token_count')} of {m.get('tokens')}")
        if m.get("emitted_token_total") != m.get("tokens"):
            bad.append(f"{m['dir']}: emitted "
                       f"{m.get('emitted_token_total')} of {m.get('tokens')}")
        if m.get("parity_all_ok") != "True":
            bad.append(f"{m['dir']}: parity_all_ok={m.get('parity_all_ok')}")
        if m.get("declared_rows_total") != m.get("reference_checked_row_total"):
            bad.append(f"{m['dir']}: declared "
                       f"{m.get('declared_rows_total')} rows but the parent "
                       f"checked {m.get('reference_checked_row_total')}")
    return bad


def determinism(legs: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for m in legs:
        groups.setdefault(leg_key(m), []).append(m)

    positive = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        digits = {tuple(m.get(k) for k in CHANNEL_KEYS) for m in members}
        positive.append({
            "arm": key[0], "prompt_id": key[1], "tokens": key[2],
            "offered_depth": key[3], "legs": [m["dir"] for m in members],
            "identical": len(digits) == 1,
            "values": sorted(str(d) for d in digits),
        })

    # Negative polarity: two groups that share a fixture and window but differ
    # in exactly one asked dimension must NOT read the same digits.
    negative = []
    keys = sorted(groups)
    names = ("arm", "prompt_id", "tokens", "offered_depth")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[1] != b[1] or a[2] != b[2]:
                continue
            differs = [n for n, x, y in zip(names, a, b) if x != y]
            if len(differs) != 1:
                continue
            va = tuple(groups[a][0].get(k) for k in CHANNEL_KEYS)
            vb = tuple(groups[b][0].get(k) for k in CHANNEL_KEYS)
            # A depth cap that sits above the deepest draft the scheduler ever
            # asked for cannot change any digit, so an identical reading there
            # says nothing about the channel. Separate "did not bind" from
            # "did not resolve".
            binds = None
            if differs[0] == "offered_depth":
                reach = [num(groups[k][0], "effective_max_draft_len")
                         for k in (a, b)]
                offers = [int(a[3]), int(b[3])]
                if all(r is not None for r in reach):
                    binds = max(reach) > min(offers)
            # Only a pair that contains the shipped arm says anything about
            # the instrument. Two treatment arms agreeing is a result about
            # those treatments, and scoring it as a channel failure would be
            # a category error: it would let a null treatment condemn the
            # instrument that measured it.
            if "ship" not in (a[0], b[0]):
                role = "cross-treatment"
            elif differs[0] == "offered_depth":
                role = "control"
            else:
                role = "arm-liveness"
            negative.append({
                "a": f"{a[0]}/{a[1]}/offer{a[3]}",
                "b": f"{b[0]}/{b[1]}/offer{b[3]}",
                "differs_in": differs[0],
                "role": role,
                "distinguished": va != vb,
                "perturbation_binds": binds,
                "round_count_a": groups[a][0].get("round_count"),
                "round_count_b": groups[b][0].get("round_count"),
            })
    return {"positive": positive, "negative": negative}


def price(legs: list[dict]) -> dict:
    ref = {}
    for m in legs:
        if m["arm"] == "ship" and m["offered_depth"] == "8":
            ref.setdefault(m["prompt_id"], m)

    rows = []
    for m in legs:
        if m["offered_depth"] != "8":
            continue
        base = ref.get(m["prompt_id"])
        if base is None or m["dir"] == base["dir"] or m["arm"] == "ship":
            continue
        r_ship, r_arm = num(base, "round_count"), num(m, "round_count")
        if r_ship is None or r_arm is None:
            continue
        arm = m["arm"]
        acc_cost = 100.0 * (r_arm - r_ship) / r_ship
        gross_local = GROSS_BYTE_PCT_LOCAL[arm_probe_key(arm)]
        if arm_is_fp32(arm):
            gross_local += FP32_GROSS_BYTE_PCT_LOCAL
        gross_median = gross_local * LOCAL_GROSS_TO_MEDIAN_GAIN
        a_ship, a_arm = (num(base, "accepted_draft_rate"),
                         num(m, "accepted_draft_rate"))
        rw_ship, rw_arm = (num(base, "declared_rows_total"),
                           num(m, "declared_rows_total"))
        rows.append({
            "arm": arm,
            "prompt_id": m["prompt_id"],
            "rep": m["rep"],
            "round_count_ship": r_ship,
            "round_count_arm": r_arm,
            "delta_rounds": r_arm - r_ship,
            "accepted_ship": num(base, "accepted_draft_total"),
            "accepted_arm": num(m, "accepted_draft_total"),
            "accept_rate_ship": a_ship,
            "accept_rate_arm": a_arm,
            "accept_delta_pp": (None if a_ship is None or a_arm is None
                                else 100.0 * (a_arm - a_ship)),
            "mean_draft_ship": num(base, "effective_mean_draft_len"),
            "mean_draft_arm": num(m, "effective_mean_draft_len"),
            "declared_rows_ship": rw_ship,
            "declared_rows_arm": rw_arm,
            # Second cost proxy. Rounds price the fixed per-round overhead;
            # verified rows price the width-dependent part. Reporting both
            # brackets the constant-cost-per-round assumption instead of
            # hiding it.
            "row_cost_pct": (None if not rw_ship
                             else 100.0 * (rw_arm - rw_ship) / rw_ship),
            "quantisation_floor_pct": 100.0 / r_ship,
            "acceptance_cost_pct": acc_cost,
            "gross_byte_pct_local": gross_local,
            "gross_byte_pct_median": gross_median,
            "net_median_pct": gross_median - acc_cost * ACCEPTANCE_TRANSFER,
            "rerank_drafts": num(m, "witness_fp32_rerank_drafts"),
            "probes": num(m, "witness_probes"),
        })
    return {"reference": {k: v["dir"] for k, v in ref.items()}, "rows": rows}


def per_arm(rows: list[dict]) -> list[dict]:
    arms: dict[str, list[dict]] = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    out = []
    for arm, rs in sorted(arms.items()):
        nets = [r["net_median_pct"] for r in rs]
        out.append({
            "arm": arm,
            "fixtures": [r["prompt_id"] for r in rs],
            "delta_rounds": [r["delta_rounds"] for r in rs],
            "acceptance_cost_pct": [r["acceptance_cost_pct"] for r in rs],
            "accept_delta_pp": [r["accept_delta_pp"] for r in rs],
            "net_median_pct_per_fixture": nets,
            "net_median_pct_mean": sum(nets) / len(nets),
            "net_median_pct_worst": min(nets),
            "gross_byte_pct_median": rs[0]["gross_byte_pct_median"],
            "acceptance_null": all(r["delta_rounds"] == 0 for r in rs),
            "acceptance_resolved": all(r["delta_rounds"] != 0 for r in rs),
            "tightest_bound_pct": min(r["quantisation_floor_pct"] for r in rs),
        })
    return out


def reconstruct_rival(summary: list[dict]) -> dict:
    p015 = next((a for a in summary if a["arm"] == "p015"), None)
    if p015 is None:
        return {"available": False}
    gross = p015["gross_byte_pct_median"]
    implied = gross - RIVAL_MEDIAN_PCT_AT_P015
    measured = max(p015["acceptance_cost_pct"])
    bound = p015["tightest_bound_pct"]
    predicted_net = gross - measured * ACCEPTANCE_TRANSFER
    return {
        "available": True,
        "rival_median_receipts": RIVAL_MEDIAN_RECEIPTS_AT_P015,
        "rival_median_pct": RIVAL_MEDIAN_PCT_AT_P015,
        "rival_median_2sigma": RIVAL_MEDIAN_2SIGMA_AT_P015,
        "gross_byte_pct_median": gross,
        "measured_acceptance_cost_pct_worst": measured,
        "predicted_net_pct": predicted_net,
        "measured_over_model": (RIVAL_MEDIAN_PCT_AT_P015 / predicted_net
                                if predicted_net else None),
        "agrees_within_rival_2sigma":
            abs(predicted_net - RIVAL_MEDIAN_PCT_AT_P015)
            <= RIVAL_MEDIAN_2SIGMA_AT_P015,
        # What the ranked receipt would need the acceptance cost to be, and
        # whether this channel could have seen a cost that small at all. If
        # the implied cost is inside one round, a null leg neither confirms
        # nor refutes it.
        "implied_acceptance_cost_pct": implied,
        "channel_bound_pct": bound,
        "channel_can_resolve_implied_cost": abs(implied) >= bound,
        "residual_pct": measured - implied,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".mlxfast-private/e139/runs")
    ap.add_argument("--json")
    args = ap.parse_args()

    legs = load_legs(pathlib.Path(args.runs))
    if not legs:
        print(f"e139_analyse: no legs under {args.runs}", file=sys.stderr)
        return 1

    row = "  {:<34} {:>4} {:>8} {:>8} {:>8} {:>9} {:>7} {:>3} {:>6} {:>6}"
    print("=== legs ===")
    print(row.format("dir", "R", "accepted", "rejected", "mean_d", "accept",
                     "matched", "div", "probes", "rerank"))
    for m in legs:
        print(row.format(
            m["dir"], m.get("round_count", "?"),
            m.get("accepted_draft_total", "?"),
            m.get("rejected_draft_total", "?"),
            f"{float(m['effective_mean_draft_len']):.4f}"
            if m.get("effective_mean_draft_len") else "?",
            f"{float(m['accepted_draft_rate']):.6f}"
            if m.get("accepted_draft_rate") else "?",
            m.get("all_tokens_matched", "?"),
            m.get("residual_divergence_count", "?"),
            m.get("witness_probes", "?"),
            m.get("witness_fp32_rerank_drafts", "?")))

    prov = {k: sorted({str(m.get(k)) for m in legs}) for k in PROVENANCE_KEYS}
    print("\n=== provenance (one value per key or the session is mixed) ===")
    for k, v in prov.items():
        # `base_sha` moves whenever a research-only file is committed between
        # legs. What must not move is the built artifact, so a spread in
        # `base_sha` under a single `worker_sha256` is recorded, not flagged.
        soft = k == "base_sha" and len(prov["worker_sha256"]) == 1
        mark = "" if len(v) == 1 else ("   (research-only commits; one worker)"
                                       if soft else "   <-- MIXED")
        print(f"  {k}: {', '.join(v)}{mark}")

    bad = check_legs(legs)
    print("\n=== witness and exactness ===")
    for b in bad:
        print(f"  FAIL {b}")
    if not bad:
        print(f"  all {len(legs)} legs: arm witnessed, tokens matched, zero "
              "residual divergence, clean candidate tree, worker still")

    det = determinism(legs)
    print("\n=== instrument, positive polarity "
          "(same arm twice must be identical) ===")
    for g in det["positive"]:
        print(f"  {g['arm']}/{g['prompt_id']}/offer{g['offered_depth']} "
              f"x{len(g['legs'])}: "
              + ("IDENTICAL" if g["identical"] else "DIFFERS  <-- FAIL"))
        if not g["identical"]:
            for v in g["values"]:
                print(f"      {v}")
    if not det["positive"]:
        print("  no repeated arm: positive polarity NOT demonstrated")

    print("\n=== instrument, negative polarity "
          "(one changed knob must be distinguished) ===")
    for g in det["negative"]:
        if g["role"] == "cross-treatment":
            continue
        if g["perturbation_binds"] is False:
            verdict = "VACUOUS (cap never bound)"
        elif g["distinguished"]:
            verdict = "DISTINGUISHED"
        elif g["role"] == "control":
            verdict = "IDENTICAL  <-- CHANNEL FAIL"
        else:
            verdict = "IDENTICAL  (arm is null, not a channel fault)"
        print(f"  [{g['role']:<13}] {g['a']} vs {g['b']}: "
              f"{verdict}   R {g['round_count_a']} -> {g['round_count_b']}")
    live = sorted({g["a"].split("/")[1] for g in det["negative"]
                   if g["distinguished"]
                   and g["perturbation_binds"] is not False
                   and g["role"] in ("control", "arm-liveness")})
    print(f"  fixtures where some single knob moved the channel: "
          f"{live or 'NONE'}")
    if not det["negative"]:
        print("  no single-knob pair: negative polarity NOT demonstrated")

    priced = price(legs)
    summary = per_arm(priced["rows"])
    print("\n=== acceptance channel, priced against ship (harness=ranked) ===")
    print(f"  reference legs: {priced['reference']}")
    print(f"  LOCAL_GROSS_TO_MEDIAN_GAIN={LOCAL_GROSS_TO_MEDIAN_GAIN} "
          f"(gross byte term only)   "
          f"ACCEPTANCE_TRANSFER={ACCEPTANCE_TRANSFER}")
    prow = "  {:<6} {:<16} {:>3} {:>6} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}"
    print(prow.format("arm", "fixture", "dR", "dRows", "rows%", "accept_pp",
                      "floor%", "cost%", "gross%", "net%"))
    for r in priced["rows"]:
        print("  {:<6} {:<16} {:>3} {:>6} {:>8.4f} {:>9.4f} {:>8.4f} "
              "{:>8.4f} {:>8.4f} {:>8.4f}".format(
                  r["arm"], r["prompt_id"], r["delta_rounds"],
                  r["declared_rows_arm"] - r["declared_rows_ship"],
                  r["row_cost_pct"], r["accept_delta_pp"] or 0.0,
                  r["quantisation_floor_pct"],
                  r["acceptance_cost_pct"], r["gross_byte_pct_median"],
                  r["net_median_pct"]))

    print("\n=== per-arm verdict ===")
    for a in summary:
        state = ("NULL (no round moved on any fixture)" if a["acceptance_null"]
                 else "RESOLVED on every fixture" if a["acceptance_resolved"]
                 else "SPLIT across fixtures")
        print(f"  {a['arm']}: acceptance {state}; net_median_pct mean "
              f"{a['net_median_pct_mean']:+.4f} worst "
              f"{a['net_median_pct_worst']:+.4f}; the channel resolves no "
              f"finer than {a['tightest_bound_pct']:.4f} %")

    rival = reconstruct_rival(summary)
    print("\n=== rival receipt reconstruction (p=0.15) ===")
    if not rival["available"]:
        print("  no p015 leg")
    else:
        for name, v in rival["rival_median_receipts"].items():
            print(f"  receipt {name:<36} {v:+.4f} %")
        print(f"  pooled ranked receipt        "
              f"{rival['rival_median_pct']:+.4f} % "
              f"+/- {rival['rival_median_2sigma']:.4f} (2 sigma)")
        print(f"  my gross byte term           "
              f"{rival['gross_byte_pct_median']:+.4f} % "
              f"(x{LOCAL_GROSS_TO_MEDIAN_GAIN} median transfer)")
        print(f"  minus measured acceptance    "
              f"{rival['measured_acceptance_cost_pct_worst']:+.4f} %")
        print(f"  => my predicted net          "
              f"{rival['predicted_net_pct']:+.4f} %")
        print(f"  measured / model             "
              f"{rival['measured_over_model']:.4f}")
        print(f"  agrees within rival 2 sigma? "
              f"{rival['agrees_within_rival_2sigma']}")
        print(f"  implied acceptance cost      "
              f"{rival['implied_acceptance_cost_pct']:+.4f} % "
              f"(what the receipt would need)")
        print(f"  channel resolution           "
              f"{rival['channel_bound_pct']:.4f} % (one round)")
        print(f"  channel can resolve it?      "
              f"{rival['channel_can_resolve_implied_cost']}")

    print("\n=== composition additivity ===")
    singles = {a["arm"]: a for a in summary}
    seen = False
    for composed, parts in (("fp32p015", ("fp32", "p015")),
                            ("fp32p010", ("fp32", "p010"))):
        if composed not in singles:
            continue
        seen = True
        if all(p in singles for p in parts):
            got = singles[composed]["net_median_pct_mean"]
            want = sum(singles[p]["net_median_pct_mean"] for p in parts)
            print(f"  {composed}: measured {got:+.4f} % vs sum of parts "
                  f"{want:+.4f} % (delta {got - want:+.4f} %)")
        else:
            print(f"  {composed}: measured but a single-rider part is missing")
    if not seen:
        print("  no composed arm measured")

    payload = {
        "harness": "ranked",
        "local_gross_to_median_gain": LOCAL_GROSS_TO_MEDIAN_GAIN,
        "acceptance_transfer": ACCEPTANCE_TRANSFER,
        "gross_byte_pct_local": GROSS_BYTE_PCT_LOCAL,
        "legs": [{k: v for k, v in m.items() if k != "report"} for m in legs],
        "provenance": prov,
        "failures": bad,
        "determinism": det,
        "priced": priced,
        "per_arm": summary,
        "rival_reconstruction": rival,
    }
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
