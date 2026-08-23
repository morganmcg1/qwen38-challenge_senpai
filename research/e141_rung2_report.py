#!/usr/bin/env python3
"""Price the E141 shipped -> full draft-vocabulary arm from the ABBA session.

The estimator is the within-replicate contrast. With order A B B A both arms
sit at mean position 2.5, so subtracting the two shipped legs from the two full
legs inside one replicate cancels linear thermal drift exactly. The
per-replicate contrast, not the pooled arm mean, is the reported number.

TWO BASES, as E134 established:

  decode  parent_measured_seconds_per_token, what the mechanism moves
  ranked  (seed_prefill_seconds + decode_seconds) / decode_token_count

The ranked basis is the headline, because the ranked leg times seed processing
and decoding in one window and no draft-vocabulary change touches the seed.
Reporting only the decode basis overstates the score effect.

WHY NO LOCAL-RATIO CORRECTION. The ranked numerator is the runner's own
prebuilt serial baseline, which no candidate edit reaches, so no psi_serial
share is ever subtracted. Both legs here run the candidate binary, but the arm
is a run-time selector confined to the candidate MTP leg, so nothing cancels.

WHY THE LOCAL PERCENTAGE IS STILL NOT THE RANKED PERCENTAGE. CAMPAIGN RULE 115:
convert a per-round absolute mechanism through absolute microseconds per round,
never through the local percentage. Split the effect into two terms:

  full_spt / shipped_spt = (N_full / N_shipped) * (1 + delta_us / round_us)

  term 1  N_full / N_shipped   a dimensionless round-count ratio. Acceptance is
          deterministic, so this transfers to the ranked harness unchanged.
  term 2  delta_us / round_us  a fixed absolute cost over the round length. The
          local benchfixture round is about 195,000 us; the medpair-weighted
          ranked round is 52,726 us (F189 arm B). The same absolute cost is a
          much larger fraction of the shorter ranked round.

So this experiment can and does show a positive local percentage while the
ranked value is negative. Only term 2 is rescaled:

  net_ranked_pct = 100 * (1 - (N_full / N_shipped) * (1 + f_ranked))
  f_ranked       = kappa * delta_us_local / ranked_round_us

kappa is the M4 Pro -> M5 absolute-cost transfer and is class-dependent, so
this reports the bracket rather than one false point estimate:

  kappa = 0.646  Rule 115 per-round overhead class, the unfavourable end.
  kappa = 0.270  the bandwidth carve-out Rule 115 states for a mechanism such
                 as C1, where the ranked host moves bytes faster so the
                 absolute cost shrinks with the round and f is invariant. That
                 is 52,726 / 195,171 expressed on the same kappa axis, and it
                 makes f_ranked equal f_local. This mechanism reads 2.53x more
                 coarse table per draft, so it has real bandwidth character.

If the conclusion is the same at both ends, the host-transfer class does not
matter and the result is robust.

COST DECOMPOSITION. Per-round cost splits into work proportional to the leaf
count, which the probe fraction cannot touch, and work proportional to
probes * rowsPerLeaf, which it scales linearly. That split is what lets the
p=0.10 figure be derived; it is a model, and it is labelled as one.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Rule 116: the published median rides on these two prompts alone.
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}

ROWS_PER_LEAF = 8
SHIPPED_PADDED = 98_336
FULL_PADDED = 248_320
PROBE_FRACTION = 0.25

# Rule 115 conversion constants, all from senpai/campaign-ledger.md.
RANKED_ROUND_US = 52_726.0  # medpair-weighted ranked round, F189 arm B
LOCAL_REFERENCE_ROUND_US = 195_171.0  # benchfixture anchor quoted by Rule 115
KAPPA_OVERHEAD = 0.646  # absolute M4 Pro -> M5, per-round overhead class
KAPPA_BANDWIDTH = RANKED_ROUND_US / LOCAL_REFERENCE_ROUND_US  # f invariant

# The advisor's pre-registered price: recoverable mass * coefficient.
PREREGISTERED_COEFFICIENTS = (65.0, 203.0)


def leaves(padded: int) -> int:
    return padded // ROWS_PER_LEAF


def probes(padded: int, fraction: float) -> int:
    return max(1, math.ceil(fraction * leaves(padded)))


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if _:
            out[key] = value
    return out


def collect(runs_parent: Path, label: str) -> list[dict]:
    legs = []
    for slot_dir in sorted(runs_parent.glob(f"{label}k*p*")):
        for prompt_dir in sorted(slot_dir.iterdir()):
            meta_path = prompt_dir / "meta.txt"
            report_path = prompt_dir / "report.json"
            if not meta_path.exists() or not report_path.exists():
                continue
            meta = read_meta(meta_path)
            report = json.loads(report_path.read_text())
            tokens = report["decode_token_count"]
            decode_spt = report["parent_measured_seconds_per_token"]
            ranked_spt = (
                report["seed_prefill_seconds"] + report["decode_seconds"]
            ) / tokens
            rounds = report["round_count"]
            legs.append(
                {
                    "slot": slot_dir.name,
                    "prompt": prompt_dir.name,
                    "arm": meta.get("e141_arm_requested", "?"),
                    "prefix_exported": meta.get("e141_prefix_exported", "?"),
                    "replicate": int(meta.get("e141_replicate", "0")),
                    "position": int(meta.get("e141_position", "0")),
                    "witness": meta.get("e141_witness", "absent"),
                    "rounds_want": meta.get("e141_witness_rounds_want", "?"),
                    "round_count": rounds,
                    "mean_draft": report["effective_mean_draft_len"],
                    "accepted_draft_rate": report["accepted_draft_rate"],
                    "decode_spt": decode_spt,
                    "ranked_spt": ranked_spt,
                    "decode_seconds": report["decode_seconds"],
                    "seed_prefill_seconds": report["seed_prefill_seconds"],
                    "decode_tokens": tokens,
                    "us_per_round": 1e6 * report["decode_seconds"] / rounds,
                    "all_tokens_matched": report["all_tokens_matched"],
                    "residual_divergence_count": report[
                        "residual_divergence_count"
                    ],
                    "entry_c": meta.get("gpu_temp_entry_c", ""),
                    "exit_c": meta.get("gpu_temp_exit_c", ""),
                    "timing_valid": meta.get("timing_valid", "?"),
                    "cool_gate_passed_real_gate": meta.get(
                        "cool_gate_passed_real_gate", "?"
                    ),
                    "gate_qualified_for_timing": meta.get(
                        "gate_qualified_for_timing", "?"
                    ),
                    "commit": meta.get("e141_session_commit", "?"),
                    "worker": meta.get("e141_session_worker_sha256", "?"),
                }
            )
    return legs


def contrasts(legs: list[dict], prompt: str, basis: str) -> list[float]:
    """Per-replicate 100 * (1 - full/shipped) on one prompt and one basis."""
    out = []
    reps = sorted({leg["replicate"] for leg in legs if leg["prompt"] == prompt})
    for rep in reps:
        rows = [
            leg
            for leg in legs
            if leg["prompt"] == prompt
            and leg["replicate"] == rep
            and leg["witness"] != "MISMATCH"
        ]
        shipped = [leg[basis] for leg in rows if leg["arm"] == "shipped"]
        full = [leg[basis] for leg in rows if leg["arm"] == "full"]
        if len(shipped) != 2 or len(full) != 2:
            continue
        out.append(100.0 * (1.0 - statistics.mean(full) / statistics.mean(shipped)))
    return out


def ranked_conversion(
    prompts: dict, round_ratio: dict[str, float], kappa: float
) -> dict:
    """Rule 115: rescale only the absolute per-round cost term.

    The round-count ratio is deterministic and transfers unchanged; the added
    microseconds are re-expressed over the ranked round instead of the local
    one. Returns the medpair-weighted net and the per-prompt terms.
    """
    per_prompt: dict[str, dict] = {}
    for prompt, ratio in round_ratio.items():
        added = prompts[prompt]["added_us_per_round_at_p025"]
        if added is None:
            return {}
        f_local = added / prompts[prompt]["shipped_us_per_round"]
        f_ranked = kappa * added / RANKED_ROUND_US
        per_prompt[prompt] = {
            "round_count_ratio": ratio,
            "rounds_term_pct": 100.0 * (1.0 - ratio),
            "added_us_per_round_local": added,
            "f_local_pct": 100.0 * f_local,
            "f_ranked_pct": 100.0 * f_ranked,
            "net_ranked_pct": 100.0 * (1.0 - ratio * (1.0 + f_ranked)),
            "net_local_pct": 100.0 * (1.0 - ratio * (1.0 + f_local)),
        }
    return {
        "kappa": kappa,
        "ranked_round_us": RANKED_ROUND_US,
        "harness": "ranked",
        "seed_prefill_dilution": (
            "NOT modelled. The ranked leg times seed processing and decoding in "
            "one window, and the arm cannot touch prefill, so the true ranked "
            "effect is this value scaled by the decode share of the leg. That "
            "shrinks the magnitude of either sign. Ignoring it therefore "
            "OVERSTATES a positive result, so it is anti-conservative for the "
            "promotion decision and conservative for a refutation."
        ),
        "per_prompt": per_prompt,
        "net_ranked_pct": sum(
            MEDPAIR[p] * per_prompt[p]["net_ranked_pct"] for p in MEDPAIR
        ),
        "rounds_term_pct": sum(
            MEDPAIR[p] * per_prompt[p]["rounds_term_pct"] for p in MEDPAIR
        ),
        "cost_term_pct": sum(
            MEDPAIR[p]
            * (per_prompt[p]["net_ranked_pct"] - per_prompt[p]["rounds_term_pct"])
            for p in MEDPAIR
        ),
    }


def mean_by(legs: list[dict], prompt: str, arm: str, key: str) -> float | None:
    vals = [
        leg[key]
        for leg in legs
        if leg["prompt"] == prompt and leg["arm"] == arm
        and leg["witness"] != "MISMATCH"
    ]
    return statistics.mean(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s1")
    ap.add_argument("--runs", default=".mlxfast-private/e128/runs-e141")
    ap.add_argument("--out", default="research/e141-rung2.json")
    ap.add_argument("--rung3", default="research/e141-rung3.json")
    ap.add_argument("--census", default="research/e141-census.json")
    args = ap.parse_args()

    legs = collect(Path(args.runs), args.label)
    if not legs:
        print("e141_rung2_report: no legs found")
        return

    report: dict = {
        "label": args.label,
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_or_ranked_score": False,
        "estimator": "within-replicate ABBA contrast, mean position 2.5 per arm",
        "legs": legs,
        "prompts": {},
    }

    for prompt in MEDPAIR:
        rows = [leg for leg in legs if leg["prompt"] == prompt]
        if not rows:
            continue
        entry = [float(leg["entry_c"]) for leg in rows if leg["entry_c"]]
        blob: dict = {
            "leg_count": len(rows),
            "entry_temp_c_min": min(entry) if entry else None,
            "entry_temp_c_max": max(entry) if entry else None,
            "entry_temp_c_spread": (max(entry) - min(entry)) if entry else None,
            "mismatched_legs": sum(1 for leg in rows if leg["witness"] == "MISMATCH"),
            "divergences": sum(
                (leg["residual_divergence_count"] or 0)
                + (0 if leg["all_tokens_matched"] else 1)
                for leg in rows
            ),
        }
        for basis in ("decode_spt", "ranked_spt"):
            values = contrasts(legs, prompt, basis)
            blob[f"{basis}_gain_pct_per_replicate"] = values
            blob[f"{basis}_gain_pct"] = statistics.mean(values) if values else None
            blob[f"{basis}_gain_pct_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        for arm in ("shipped", "full"):
            for key in ("round_count", "us_per_round", "decode_spt", "ranked_spt",
                        "mean_draft", "accepted_draft_rate"):
                blob[f"{arm}_{key}"] = mean_by(legs, prompt, arm, key)
        blob["added_us_per_round_at_p025"] = (
            blob["full_us_per_round"] - blob["shipped_us_per_round"]
            if blob["full_us_per_round"] and blob["shipped_us_per_round"]
            else None
        )
        report["prompts"][prompt] = blob

    have = [p for p in MEDPAIR if p in report["prompts"]]
    if len(have) == len(MEDPAIR):
        def medpair(key: str) -> float | None:
            vals = [report["prompts"][p][key] for p in MEDPAIR]
            if any(v is None for v in vals):
                return None
            return sum(MEDPAIR[p] * report["prompts"][p][key] for p in MEDPAIR)

        report["e141_net_local_ranked_basis_pct"] = medpair("ranked_spt_gain_pct")
        report["e141_net_decode_pct"] = medpair("decode_spt_gain_pct")
        added = medpair("added_us_per_round_at_p025")

        # Rule 115. The deciding number is the ranked conversion, not the
        # measured local percentage. Take the round-count ratio from rung 3,
        # where it is exact and deterministic, and cross-check it against the
        # timing legs, which must reproduce it if the arm selector worked.
        rung3 = json.loads(Path(args.rung3).read_text())
        d3 = rung3["delta"]
        ratio = {
            p: d3["round_count_full"][p] / d3["round_count_shipped"][p]
            for p in MEDPAIR
        }
        report["round_count_ratio_source"] = args.rung3
        report["round_count_ratio"] = ratio
        report["round_count_ratio_from_timing_legs"] = {
            p: (
                report["prompts"][p]["full_round_count"]
                / report["prompts"][p]["shipped_round_count"]
            )
            for p in MEDPAIR
        }
        report["round_count_ratio_agrees_with_rung3"] = all(
            abs(ratio[p] - report["round_count_ratio_from_timing_legs"][p]) < 1e-9
            for p in MEDPAIR
        )

        conversions = {
            "overhead_class": ranked_conversion(
                report["prompts"], ratio, KAPPA_OVERHEAD
            ),
            "bandwidth_class": ranked_conversion(
                report["prompts"], ratio, KAPPA_BANDWIDTH
            ),
        }
        report["rule115_conversion"] = conversions
        nets = [c["net_ranked_pct"] for c in conversions.values() if c]
        if nets:
            # Primary metric. Report the unfavourable end as the headline and
            # keep the whole bracket, because the host-transfer class for this
            # mechanism is not settled by this experiment.
            report["e141_net_ranked_pct"] = min(nets)
            report["e141_net_ranked_pct_bracket"] = [min(nets), max(nets)]
            report["e141_net_ranked_pct_gross_rounds_only"] = conversions[
                "overhead_class"
            ]["rounds_term_pct"]

            census = json.loads(Path(args.census).read_text())
            recoverable_pct = census["medpair"]["recoverable_pct_full_vocabulary"]
            recoverable = recoverable_pct / 100.0
            report["e141_medpair_recoverable_fraction_pct"] = recoverable_pct
            # The advisor priced this as recoverable mass * coefficient. State
            # which coefficient each side of the ledger actually implies.
            report["preregistered_price"] = {
                "coefficients": list(PREREGISTERED_COEFFICIENTS),
                "predicted_net_ranked_pct": [
                    recoverable * c for c in PREREGISTERED_COEFFICIENTS
                ],
                "implied_coefficient_gross_rounds": (
                    conversions["overhead_class"]["rounds_term_pct"] / recoverable
                ),
                "implied_coefficient_net": {
                    name: c["net_ranked_pct"] / recoverable
                    for name, c in conversions.items()
                    if c
                },
            }
        report["e141_added_us_per_round_at_p025"] = added
        report["e141_added_us_per_round_at_p010"] = None
        report["e141_added_us_per_round_at_p010_is_derived"] = True

        if added is not None:
            # Split the added per-round cost into the part the probe fraction
            # cannot touch and the part it scales. Centroid work follows the
            # leaf count; cluster-row scoring follows probes * rowsPerLeaf.
            d_leaves = leaves(FULL_PADDED) - leaves(SHIPPED_PADDED)
            d_rows_025 = ROWS_PER_LEAF * (
                probes(FULL_PADDED, PROBE_FRACTION)
                - probes(SHIPPED_PADDED, PROBE_FRACTION)
            )
            d_rows_010 = ROWS_PER_LEAF * (
                probes(FULL_PADDED, 0.10) - probes(SHIPPED_PADDED, PROBE_FRACTION)
            )
            # One equation, two unknowns, so state the bracket rather than a
            # false point estimate: all cost on centroids gives no p sensitivity
            # at all, all cost on row scoring scales the whole term.
            all_centroid = added
            all_rows = added * d_rows_010 / d_rows_025 if d_rows_025 else None
            report["e141_added_us_per_round_at_p010"] = (
                statistics.mean([all_centroid, all_rows])
                if all_rows is not None
                else None
            )
            report["p010_model"] = {
                "note": (
                    "DERIVED, not measured. qwen35DerivedClusterProbeFraction "
                    "(Qwen35.swift:4937) is a plain let on another experiment's "
                    "surface and cannot be selected at run time from this branch."
                ),
                "delta_leaves": d_leaves,
                "delta_rows_scored_at_p025": d_rows_025,
                "delta_rows_scored_at_p010": d_rows_010,
                "bracket_all_cost_on_centroids_us": all_centroid,
                "bracket_all_cost_on_row_scoring_us": all_rows,
            }

        report["divergences_total"] = sum(
            report["prompts"][p]["divergences"] for p in MEDPAIR
        )
        report["mismatched_legs_total"] = sum(
            report["prompts"][p]["mismatched_legs"] for p in MEDPAIR
        )

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for prompt, blob in report["prompts"].items():
        print(f"\n== {prompt} ==")
        print(
            f"  rounds    shipped {blob['shipped_round_count']} "
            f"full {blob['full_round_count']}"
        )
        print(
            f"  us/round  shipped {blob['shipped_us_per_round']:.1f} "
            f"full {blob['full_us_per_round']:.1f} "
            f"added {blob['added_us_per_round_at_p025']:+.1f}"
        )
        for basis in ("decode_spt", "ranked_spt"):
            vals = blob[f"{basis}_gain_pct_per_replicate"]
            mean = blob[f"{basis}_gain_pct"]
            sd = blob[f"{basis}_gain_pct_stdev"]
            sd_text = f" sd {sd:.4f}" if sd is not None else ""
            print(
                f"  {basis:11s} {mean:+.4f} %{sd_text}  "
                f"per replicate {[round(v, 4) for v in vals]}"
            )
        print(
            f"  entry temp spread {blob['entry_temp_c_spread']} C  "
            f"divergences {blob['divergences']}  "
            f"witness mismatches {blob['mismatched_legs']}"
        )

    if "e141_net_ranked_pct" in report:
        print("\n== harness=local, measured ==")
        print(
            "  net on ranked basis "
            f"{report['e141_net_local_ranked_basis_pct']:+.4f} %   "
            f"net on decode basis {report['e141_net_decode_pct']:+.4f} %"
        )
        print(
            "\n== harness=ranked, Rule 115 conversion "
            f"(ranked round {RANKED_ROUND_US:.0f} us) =="
        )
        agree = report["round_count_ratio_agrees_with_rung3"]
        print(
            f"  round-count ratio from rung 3 reproduced by timing legs: {agree}"
        )
        for name, conv in report["rule115_conversion"].items():
            print(
                f"  {name:15s} kappa {conv['kappa']:.3f}  "
                f"rounds {conv['rounds_term_pct']:+.4f} % "
                f"cost {conv['cost_term_pct']:+.4f} % "
                f"-> net {conv['net_ranked_pct']:+.4f} %"
            )
        lo, hi = report["e141_net_ranked_pct_bracket"]
        print(
            f"\ne141_net_ranked_pct = {report['e141_net_ranked_pct']:+.4f} "
            f"(headline, unfavourable end; bracket {lo:+.4f} to {hi:+.4f})"
        )
        pre = report["preregistered_price"]
        plo, phi = pre["predicted_net_ranked_pct"]
        print(
            f"  pre-registered band {plo:+.4f} to {phi:+.4f} % "
            f"from coefficients {pre['coefficients']}"
        )
        print(
            "  implied coefficient: gross rounds "
            f"{pre['implied_coefficient_gross_rounds']:.1f}, net "
            + ", ".join(
                f"{n} {v:.1f}" for n, v in pre["implied_coefficient_net"].items()
            )
        )
        print(
            "e141_added_us_per_round_at_p025 = "
            f"{report['e141_added_us_per_round_at_p025']:+.1f}"
        )
        if report.get("e141_added_us_per_round_at_p010") is not None:
            m = report["p010_model"]
            print(
                "e141_added_us_per_round_at_p010 = "
                f"{report['e141_added_us_per_round_at_p010']:+.1f} DERIVED "
                f"(bracket {m['bracket_all_cost_on_row_scoring_us']:+.1f} to "
                f"{m['bracket_all_cost_on_centroids_us']:+.1f})"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
