#!/usr/bin/env python3
"""E217: does an explicitly staged weight tile pay the wide-QMV per-pass fixed
cost once instead of twice at a `G == 2` width?

    usage: research/e217_report.py SESSION [SESSION ...]
                                   [--out research/e217-artifacts/e217-report.json]

harness=local. Every leg runs with the per-round phase trace on and the cool
gate off, so no number here is a gate-qualified timing claim and none of them
is a candidate speed claim.

SIGN CONVENTION. A contrast named `X - off` is the SAVING of mapping `X`
against the shipped `split` mapping:

    saving_ms(X) = mean(off round_ms) - mean(X round_ms)

so POSITIVE means `X` is FASTER. This is the convention E216 reported
FINDING 565 in (`coop - off = -1.323 ms/round` pooled, meaning coop LOST).

PAIRING. The mappings are bit-exact against each other, so a leg emits the
same token stream, the same draft decisions and therefore the same round
sequence whatever the mapping is. Round index `r` is the same round of work in
every arm, which makes this a paired design. A palindrome such as
`off coop staged staged coop off` gives every arm two legs placed
symmetrically about the session midpoint, so the per-round arm value is the
mean of its two legs and monotone thermal drift cancels to first order.

CENSUS. Per-width savings are reported on their own and then round-weighted
twice: once with the public single-fixture census of the `off` legs of this
session, and once with the pooled eight-prompt census
`research/e215-artifacts/width-census-stepq.json` (FINDING 564). The pooled
number is the one the promotion rule reads.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out" / "e217"
POOLED_CENSUS = ROOT / "e215-artifacts" / "width-census-stepq.json"

# The first rounds of a leg pay cold-cache and first-shape costs that no
# steady-state round pays. E182, E205 and E212 dropped the same two.
DROP_FIRST_ROUNDS = 2

# FINDING 503: session-to-session noise floor of a traced per-round mean.
NOISE_FLOOR_MS = 0.24

# The sole promotion gate for E217, on the pooled-census-weighted `staged - off`
# saving (advisor amendment, PR #215).
PROMOTION_GATE_MS = 1.0

# Widths a `G == 2` mapping can move. Every other width keeps the shipped
# mapping by construction, so 2..5 is the control band.
MOVED_WIDTHS = [6, 7, 8, 9]
CONTROL_WIDTHS = [2, 3, 4, 5]

ARMS = ["off", "coop", "staged"]

PHASES = [
    "round_us",
    "draft_build_us",
    "d_pre_us",
    "d_flush_us",
    "d_head1_us",
    "d_submit1_us",
    "d_chain_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
COUNTERS = ["acc", "xs_fill", "xs_hit", "cfg_miss"]

# Mapping witnesses emitted once per round. Only a `G == 2` dispatch increments
# one of these, so they prove which mapping the moved widths really took.
WITNESSES = ["qmv_split_g2", "qmv_coop", "qmv_staged"]


def parse_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def parse_rounds(path: pathlib.Path) -> list[dict[str, float]]:
    """Steady-state rounds of the LAST traced session in the file.

    `--local-iterate` runs the pinned serial leg and then the MTP leg from the
    same build, and both append to this file. Each session opens with
    `mtp-trace: begin`, so the last session is the MTP leg.
    """
    sessions: list[list[dict[str, float]]] = []
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if line.startswith("mtp-trace: begin"):
            sessions.append([])
            continue
        if not line.startswith("mtp-trace: round=") or not sessions:
            continue
        kv = dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
        if "round" not in kv or float(kv["round"]) <= DROP_FIRST_ROUNDS:
            continue
        row = {"round": float(kv["round"]), "d": float(kv.get("d", -1))}
        for field in PHASES + COUNTERS + WITNESSES:
            if field in kv:
                row[field] = float(kv[field])
        sessions[-1].append(row)
    return sessions[-1] if sessions else []


def read_leg(leg_dir: pathlib.Path) -> dict | None:
    meta = parse_meta(leg_dir / "meta.txt")
    if not meta.get("e217_arm"):
        return None
    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})
    rounds = parse_rounds(leg_dir / "trace.txt")

    histogram: dict[int, int] = {}
    for r in rounds:
        width = int(r["d"]) + 1
        histogram[width] = histogram.get(width, 0) + 1

    # The mapping witness. The counters are cumulative over the process, so the
    # increment across the kept rounds is what this leg's rounds dispatched.
    witness: dict[str, float] = {}
    if rounds:
        for name in WITNESSES:
            if name in rounds[0] and name in rounds[-1]:
                witness[name] = rounds[-1][name] - rounds[0][name]
                witness[name + "_final"] = rounds[-1][name]

    return {
        "leg": leg_dir.name,
        "session": meta.get("e217_session"),
        "arm": meta.get("e217_arm"),
        "mapping": meta.get("qmv_group_mapping"),
        "leg_index": int(meta.get("e217_leg", 0)),
        "session_order": meta.get("e217_session_order"),
        "rounds_traced": len(rounds),
        "served_width_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c") or None,
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c") or None,
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "worker_sha256_before": meta.get("worker_sha256_before"),
        "worker_sha256_after": meta.get("worker_sha256_after"),
        "worker_digest_stable": meta.get("worker_digest_stable"),
        "base_sha": meta.get("base_sha"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "decode_tokens": metrics.get("decode_tokens"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
        "mapping_dispatch_witness": witness,
        "_rounds": rounds,
    }


def arm_series(legs: list[dict], arm: str, phase: str) -> dict[int, dict[int, float]]:
    """`round index -> width -> mean ms` over the legs of one arm.

    Averaging an arm's two counterbalanced legs is what cancels monotone drift.
    """
    per_round: dict[int, list[tuple[int, float]]] = {}
    for leg in legs:
        if leg["arm"] != arm:
            continue
        for r in leg["_rounds"]:
            if phase not in r:
                continue
            per_round.setdefault(int(r["round"]), []).append(
                (int(r["d"]) + 1, r[phase] / 1000.0))
    out: dict[int, tuple[int, float]] = {}
    for index, samples in per_round.items():
        widths = {w for w, _ in samples}
        if len(widths) != 1:
            # The arms diverged in draft decisions, which a bit-exact mapping
            # cannot do. Drop the round and report it.
            continue
        out[index] = (samples[0][0], statistics.fmean(v for _, v in samples),
                      len(samples))
    return out


def paired_contrast(legs: list[dict], a: str, b: str, phase: str) -> dict:
    """`a - b` as a SAVING: positive means `a` is faster than `b`.

    Rounds are paired by round index, which is the same unit of work in both
    arms because the mappings are bit-exact.
    """
    sa = arm_series(legs, a, phase)
    sb = arm_series(legs, b, phase)
    shared = sorted(set(sa) & set(sb))
    by_width: dict[int, list[float]] = {}
    width_mismatch = 0
    for index in shared:
        wa, va, na = sa[index]
        wb, vb, nb = sb[index]
        if wa != wb:
            width_mismatch += 1
            continue
        by_width.setdefault(wa, []).append(vb - va)

    table = {}
    for width, diffs in sorted(by_width.items()):
        n = len(diffs)
        mean = statistics.fmean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else float("nan")
        sem = sd / math.sqrt(n) if n > 1 else float("nan")
        table[width] = {
            "n_rounds": n,
            "saving_ms": mean,
            "stdev_ms": sd,
            "sem_ms": sem,
            "two_sigma_ms": 2.0 * sem if n > 1 else float("nan"),
            "ci95_ms": [mean - 1.96 * sem, mean + 1.96 * sem] if n > 1
                       else [float("nan")] * 2,
            "excludes_zero": bool(n > 1 and abs(mean) > 2.0 * sem),
            "above_noise_floor": bool(abs(mean) > NOISE_FLOOR_MS),
        }
    return {
        "contrast": f"{a} - {b}",
        "sign": "positive means the first arm is faster",
        "phase": phase,
        "paired_rounds": len(shared),
        "width_mismatched_rounds": width_mismatch,
        "by_width": table,
    }


def weighted(contrast: dict, shares: dict[int, float], min_n: int = 2) -> dict:
    """Round-weight per-width savings by a served-width census.

    Only widths this session measured with at least `min_n` paired rounds carry
    an interval, so they alone set the primary number. Every excluded width is
    reported with its census share and its reason, because an excluded width is
    not a zero-saving width. A width the census serves that no `G == 2` mapping
    can move does have a true saving of zero, and that is stated per width
    rather than assumed.
    """
    total = 0.0
    variance = 0.0
    covered = 0.0
    excluded = {}
    for width, share in sorted(shares.items()):
        entry = contrast["by_width"].get(width)
        if entry is None or entry["n_rounds"] < min_n:
            if share > 0:
                excluded[str(width)] = {
                    "census_share": share,
                    "n_rounds": 0 if entry is None else entry["n_rounds"],
                    "point_estimate_ms": None if entry is None
                                         else entry["saving_ms"],
                    "moved_width": width in MOVED_WIDTHS,
                }
            continue
        total += share * entry["saving_ms"]
        variance += (share * entry["sem_ms"]) ** 2
        covered += share
    sem = math.sqrt(variance)
    return {
        "saving_ms_per_round": total,
        "sem_ms": sem,
        "ci95_ms": [total - 1.96 * sem, total + 1.96 * sem],
        "excludes_zero": bool(abs(total) > 2.0 * sem),
        "min_paired_rounds_per_width": min_n,
        "census_share_covered": covered,
        "census_share_excluded": excluded,
    }


def public_census(legs: list[dict]) -> dict[int, float]:
    """Served-width shares of the `off` legs of this session."""
    counts: dict[int, int] = {}
    for leg in legs:
        if leg["arm"] != "off":
            continue
        for width, n in leg["served_width_histogram"].items():
            counts[int(width)] = counts.get(int(width), 0) + n
    total = sum(counts.values())
    return {w: n / total for w, n in sorted(counts.items())} if total else {}


def pooled_census() -> dict[int, float]:
    data = json.loads(POOLED_CENSUS.read_text())
    shares = data["pooled_all_prompts"]["share_by_served_width"]
    return {int(k): v for k, v in shares.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="research/e217-artifacts/e217-report.json")
    args = ap.parse_args()

    legs: list[dict] = []
    for session in args.sessions:
        base = OUT_ROOT / session
        for leg_dir in sorted(base.glob("leg*")):
            leg = read_leg(leg_dir)
            if leg:
                legs.append(leg)
    if not legs:
        print("e217: no legs found")
        return 2

    integrity = {
        "legs": len(legs),
        "all_tokens_matched": all(leg["all_tokens_matched"] is True for leg in legs),
        "worker_digest_stable": all(
            leg["worker_digest_stable"] == "true" for leg in legs),
        "single_worker_binary": len({leg["worker_sha256_before"] for leg in legs}) == 1,
        "single_base_sha": len({leg["base_sha"] for leg in legs}) == 1,
        "residual_divergence_total": sum(
            leg["residual_divergence_count"] or 0 for leg in legs),
        "arms_present": sorted({leg["arm"] for leg in legs}),
        "legs_per_arm": {arm: sum(1 for leg in legs if leg["arm"] == arm)
                         for arm in ARMS},
    }

    # The mapping witness, per arm. An `off` leg must show zero `coop` and zero
    # `staged` dispatches; a `staged` leg must show zero `split_g2` and zero
    # `coop`, and a non-zero `staged` count.
    witness = {}
    for arm in ARMS:
        totals = {name: 0.0 for name in WITNESSES}
        for leg in legs:
            if leg["arm"] != arm:
                continue
            for name in WITNESSES:
                totals[name] += leg["mapping_dispatch_witness"].get(name, 0.0)
        witness[arm] = totals
    expected = {"off": "qmv_split_g2", "coop": "qmv_coop", "staged": "qmv_staged"}
    integrity["mapping_witness_clean"] = all(
        witness.get(arm, {}).get(expected[arm], 0) > 0
        and all(witness[arm][other] == 0
                for other in WITNESSES if other != expected[arm])
        for arm in ARMS if arm in witness)

    # Thermal record and the bound it puts on the confound. The two legs of one
    # arm sit at opposite ends of the palindrome, so the spread of their
    # whole-leg seconds per token bounds how much entry temperature could have
    # moved that arm.
    entries = [float(leg["gpu_temp_entry_c"]) for leg in legs
               if leg["gpu_temp_entry_c"]]
    thermal = {
        "entry_temp_c": [float(leg["gpu_temp_entry_c"]) for leg in legs],
        "exit_temp_c": [float(leg["gpu_temp_exit_c"]) for leg in legs],
        "entry_temp_spread_c": max(entries) - min(entries) if entries else None,
        "arm_leg_spread_mtp_spt": {},
    }
    for arm in ARMS:
        values = [leg["mtp_seconds_per_token"] for leg in legs
                  if leg["arm"] == arm and leg["mtp_seconds_per_token"]]
        if len(values) == 2:
            thermal["arm_leg_spread_mtp_spt"][arm] = {
                "values": values,
                "abs_spread": abs(values[0] - values[1]),
                "relative_spread": abs(values[0] - values[1])
                                   / statistics.fmean(values),
            }

    shares_public = public_census(legs)
    shares_pooled = pooled_census()

    contrasts = {}
    for a, b in [("staged", "off"), ("coop", "off"), ("staged", "coop")]:
        block = {}
        for phase in PHASES:
            c = paired_contrast(legs, a, b, phase)
            if not c["by_width"]:
                continue
            if phase == "round_us":
                c["weighted_public_census"] = weighted(c, shares_public)
                c["weighted_pooled_census"] = weighted(c, shares_pooled)
                # Supplementary only. It folds in the single-round widths, which
                # carry a point estimate and no interval.
                c["weighted_pooled_census_all_widths"] = weighted(
                    c, shares_pooled, min_n=1)
                c["moved_widths"] = {
                    str(w): c["by_width"].get(w) for w in MOVED_WIDTHS}
                c["control_band"] = {
                    str(w): c["by_width"].get(w) for w in CONTROL_WIDTHS}
            block[phase] = c
        contrasts[f"{a}_minus_{b}"] = block

    headline = contrasts["staged_minus_off"]["round_us"]["weighted_pooled_census"]
    report = {
        "experiment": "e217-staged-dequant",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "sessions": args.sessions,
        "sign_convention": "a contrast `X - off` is the SAVING of X; positive means X is faster",
        "noise_floor_ms": NOISE_FLOOR_MS,
        "promotion_gate": {
            "rule": "pooled-census-weighted `staged - off` >= +1.0 ms/round",
            "threshold_ms": PROMOTION_GATE_MS,
            "measured_ms": headline["saving_ms_per_round"],
            "sem_ms": headline["sem_ms"],
            "met": bool(headline["saving_ms_per_round"] >= PROMOTION_GATE_MS),
        },
        "coop_replication": {
            "finding_565_pooled_ms": -1.323,
            "predeclared_agreement_band_ms": 0.5,
        },
        "integrity": integrity,
        "thermal": thermal,
        "mapping_dispatch_witness": witness,
        "census_public": {str(k): v for k, v in shares_public.items()},
        "census_pooled": {str(k): v for k, v in shares_pooled.items()},
        "contrasts": contrasts,
        "legs": [{k: v for k, v in leg.items() if k != "_rounds"} for leg in legs],
    }

    coop_pooled = contrasts["coop_minus_off"]["round_us"][
        "weighted_pooled_census"]["saving_ms_per_round"]
    report["coop_replication"]["measured_pooled_ms"] = coop_pooled
    report["coop_replication"]["agrees_with_finding_565"] = bool(
        abs(coop_pooled - (-1.323)) <= 0.5)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"e217: {len(legs)} legs, integrity {integrity}")
    for name, block in contrasts.items():
        c = block["round_us"]
        print(f"\n{name} (round_us, ms/round saved; positive = first arm faster)")
        for width in sorted(c["by_width"]):
            e = c["by_width"][width]
            print(f"  m={width}  n={e['n_rounds']:4d}  "
                  f"{e['saving_ms']:+8.3f} +/- {e['two_sigma_ms']:.3f} (2 sem)"
                  f"{'  *' if e['excludes_zero'] else ''}")
        if "weighted_pooled_census" in c:
            pub = c["weighted_public_census"]
            pool = c["weighted_pooled_census"]
            allw = c["weighted_pooled_census_all_widths"]
            print(f"  public census  {pub['saving_ms_per_round']:+8.3f} "
                  f"+/- {1.96 * pub['sem_ms']:.3f} (ci95), "
                  f"share {pub['census_share_covered']:.3f}")
            print(f"  pooled census  {pool['saving_ms_per_round']:+8.3f} "
                  f"+/- {1.96 * pool['sem_ms']:.3f} (ci95), "
                  f"share {pool['census_share_covered']:.3f}")
            print(f"  pooled n>=1    {allw['saving_ms_per_round']:+8.3f} "
                  f"(supplementary, share {allw['census_share_covered']:.3f})")
    print(f"\ne217: promotion gate {report['promotion_gate']}")
    print(f"e217: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
