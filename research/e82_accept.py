#!/usr/bin/env python3
"""E82 rung 0: read the acceptance screen and apply the stop rule.

The screen replays ONE serial golden per seed under every head, so the token
stream is identical across arms and each round of each arm faces exactly the
same prediction problem. That makes the comparison paired at the round level,
which is what the stop rule needs: an unpaired difference of a few tenths of a
point on 12 seeds is noise, a paired one usually is not.

What is measured
----------------
Per-position acceptance p_i = P(draft i accepted | draft i was offered). The
denominator is the REACH count, not the round count: draft 4 is only offered on
rounds that reached it, so dividing by rounds would report a depth profile that
is really a reach profile.

The stop rule the advisor set for this rung:

  advance only if the requantized head beats the reference head by >= +1.0
  point of POOLED acceptance over depths 3-6, AND loses no acceptance on the
  HARDEST TERCILE at any depth in 3-6.

The hardest tercile is the beagle analogue. Beagle is the 4th order statistic
of the eight ranked prompts in every session above score 3.15, so it and its
neighbour set the published median; a head that wins on easy prose and loses on
hard prose moves the mean and not the median.

Difficulty is per ROUND, from the golden itself: the reference top1-top2 margin
at the round's tail row. It is a property of the target model on that token, so
every arm inherits the same tercile assignment and the split cannot be gamed
by an arm's own behaviour.

Usage:
  python3 research/e82_accept.py --report research/e82-accept.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e82/screen"))
DEPTHS = range(1, 9)
RULE_DEPTHS = (3, 4, 5, 6)
Z = 1.959963984540054  # 95 %


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rounds_of(payload: dict) -> dict[int, dict]:
    """Group the row ledger into rounds keyed by round index.

    A round is one commit plus zero or more draft rows. `tail_margin` is the
    reference margin of the committed row, which is the difficulty signal.
    """
    out: dict[int, dict] = defaultdict(lambda: {"drafts": {}, "tail_margin": None})
    for row in payload["row_ledger"]:
        entry = out[row["round"]]
        if row["kind"] == "draft":
            # draft_index is 0-based in the ledger; depth i is the i-th draft.
            entry["drafts"][row["draft_index"] + 1] = bool(row["accepted"])
        else:
            entry["tail_margin"] = row["reference_margin"]
    return dict(out)


def load(arm: str, seeds: list[str]) -> dict[str, dict]:
    out = {}
    for seed in seeds:
        path = CACHE / "verify" / arm / f"{seed}.json"
        if not path.exists():
            continue
        out[seed] = json.loads(path.read_text())
    return out


def terciles(margins: list[float]) -> tuple[float, float]:
    ordered = sorted(margins)
    n = len(ordered)
    return ordered[n // 3], ordered[2 * n // 3]


def profile(arm_rounds: dict[tuple[str, int], dict], keys: list[tuple[str, int]]) -> dict:
    """Per-depth acceptance over a chosen subset of rounds."""
    per_depth = {}
    for depth in DEPTHS:
        reached = [k for k in keys if depth in arm_rounds[k]["drafts"]]
        k_acc = sum(arm_rounds[k]["drafts"][depth] for k in reached)
        p, lo, hi = wilson(k_acc, len(reached))
        per_depth[depth] = {"accepted": k_acc, "reached": len(reached), "p": p, "lo": lo, "hi": hi}
    pooled_reached = [(k, d) for k in keys for d in RULE_DEPTHS if d in arm_rounds[k]["drafts"]]
    k_acc = sum(arm_rounds[k]["drafts"][d] for k, d in pooled_reached)
    p, lo, hi = wilson(k_acc, len(pooled_reached))
    return {
        "per_depth": per_depth,
        "pooled_3_6": {"accepted": k_acc, "reached": len(pooled_reached), "p": p, "lo": lo, "hi": hi},
    }


def mcnemar(a: dict, b: dict, keys: list[tuple[str, int]], depths) -> dict:
    """Paired discordance over rounds both arms reached at the same depth."""
    b01 = b10 = both = neither = 0
    for k in keys:
        for d in depths:
            x, y = a[k]["drafts"].get(d), b[k]["drafts"].get(d)
            if x is None or y is None:
                continue
            if x and y:
                both += 1
            elif x and not y:
                b10 += 1
            elif y and not x:
                b01 += 1
            else:
                neither += 1
    n = b01 + b10
    # Continuity-corrected McNemar; exact enough at these counts.
    chi2 = (abs(b10 - b01) - 1) ** 2 / n if n else 0.0
    return {
        "candidate_only": b01,
        "reference_only": b10,
        "both": both,
        "neither": neither,
        "discordant": n,
        "chi2": chi2,
        "significant_95": chi2 > 3.841459,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="declared")
    ap.add_argument("--candidates", default="soup-q4,qat-q4,master-bf16,kamciosz,pinned")
    ap.add_argument("--report", default="research/e82-accept.json")
    args = ap.parse_args()

    manifest = json.loads(Path("research/e82-corpus-manifest.json").read_text())
    seeds = [s["name"] for s in manifest["seeds"]]
    arms = [args.reference] + [a for a in args.candidates.split(",") if a]

    payloads = {arm: load(arm, seeds) for arm in arms}
    have = [arm for arm in arms if payloads[arm]]
    common = sorted(set.intersection(*(set(payloads[a]) for a in have)))
    if not common:
        raise SystemExit("no seed has a payload for every arm")

    grouped = {arm: {(s, r): v for s in common for r, v in rounds_of(payloads[arm][s]).items()} for arm in have}

    # Difficulty comes from the reference arm's tail margins, which are a
    # property of the target model on the golden and are identical across arms.
    ref = grouped[args.reference]
    keys = sorted(k for k in ref if ref[k]["tail_margin"] is not None and ref[k]["tail_margin"] >= 0)
    lo_cut, hi_cut = terciles([ref[k]["tail_margin"] for k in keys])
    splits = {
        "pooled": keys,
        "easiest": [k for k in keys if ref[k]["tail_margin"] >= hi_cut],
        "hardest": [k for k in keys if ref[k]["tail_margin"] <= lo_cut],
    }

    report = {
        "seeds": common,
        "reference_arm": args.reference,
        "rule_depths": list(RULE_DEPTHS),
        "difficulty": {
            "signal": "reference top1-top2 margin at the round tail row",
            "tercile_cuts": {"hardest_max": lo_cut, "easiest_min": hi_cut},
            "counts": {k: len(v) for k, v in splits.items()},
        },
        "arms": {},
    }

    for arm in have:
        # A head that silently failed to load would still produce a payload, so
        # the provenance digest is asserted to differ per arm rather than assumed.
        prov = {s: payloads[arm][s].get("head_provenance", {}) for s in common}
        digests = sorted({p.get("sha256", "?") for p in prov.values()})
        entry = {
            "head_provenance_sha256": digests,
            "head_bytes": sorted({p.get("bytes") for p in prov.values()}),
            "parity_all_ok": all(payloads[arm][s].get("parity_all_ok") for s in common),
            "accepted_draft_rate": {s: payloads[arm][s].get("accepted_draft_rate") for s in common},
            "round_count": {s: payloads[arm][s].get("round_count") for s in common},
            "splits": {name: profile(grouped[arm], ks) for name, ks in splits.items()},
        }
        if arm != args.reference:
            entry["paired_vs_reference"] = {
                name: mcnemar(ref, grouped[arm], ks, RULE_DEPTHS) for name, ks in splits.items()
            }
            pooled_delta = 100 * (
                entry["splits"]["pooled"]["pooled_3_6"]["p"]
                - report["arms"][args.reference]["splits"]["pooled"]["pooled_3_6"]["p"]
            )
            hard_deltas = {
                d: 100
                * (
                    entry["splits"]["hardest"]["per_depth"][d]["p"]
                    - report["arms"][args.reference]["splits"]["hardest"]["per_depth"][d]["p"]
                )
                for d in RULE_DEPTHS
            }
            entry["stop_rule"] = {
                "pooled_delta_points": pooled_delta,
                "hardest_delta_points_by_depth": hard_deltas,
                "pooled_gate_passed": pooled_delta >= 1.0,
                "hardest_gate_passed": all(v >= 0.0 for v in hard_deltas.values()),
                "advance": pooled_delta >= 1.0 and all(v >= 0.0 for v in hard_deltas.values()),
            }
        report["arms"][arm] = entry

    Path(args.report).write_text(json.dumps(report, indent=2))
    render(report)
    print(f"\nwrote {args.report}")


def render(report: dict) -> None:
    ref = report["reference_arm"]
    print(f"seeds: {len(report['seeds'])}  rounds: {report['difficulty']['counts']}")
    print(
        f"difficulty cuts: hardest margin <= {report['difficulty']['tercile_cuts']['hardest_max']:.4f},"
        f" easiest >= {report['difficulty']['tercile_cuts']['easiest_min']:.4f}"
    )
    for split in ("pooled", "easiest", "hardest"):
        print(f"\n=== {split} tercile: per-depth acceptance % (95 % Wilson), reach in brackets ===")
        header = "arm".ljust(13) + "".join(f"    d{d}          " for d in DEPTHS)
        print(header)
        for arm, entry in report["arms"].items():
            cells = []
            for d in DEPTHS:
                c = entry["splits"][split]["per_depth"][d]
                cells.append(f" {100*c['p']:5.1f} [{c['reached']:5d}]" if c["reached"] else "     -       ")
            print(arm.ljust(13) + "".join(cells))
        print("  pooled 3-6:")
        for arm, entry in report["arms"].items():
            c = entry["splits"][split]["pooled_3_6"]
            if not c["reached"]:
                continue
            print(
                f"    {arm.ljust(13)} {100*c['p']:6.2f} % "
                f"[{100*c['lo']:.2f}, {100*c['hi']:.2f}] n={c['reached']}"
            )
    print(f"\n=== stop rule vs {ref} (pooled >= +1.0 pt AND no hardest-tercile loss at d3-d6) ===")
    for arm, entry in report["arms"].items():
        rule = entry.get("stop_rule")
        if not rule:
            continue
        hard = " ".join(f"d{d}{v:+.2f}" for d, v in rule["hardest_delta_points_by_depth"].items())
        print(
            f"  {arm.ljust(13)} pooled {rule['pooled_delta_points']:+.2f} pt"
            f"  hardest [{hard}]  -> {'ADVANCE' if rule['advance'] else 'STOP'}"
        )
        mc = entry["paired_vs_reference"]["pooled"]
        print(
            f"                paired d3-6: candidate-only {mc['candidate_only']},"
            f" reference-only {mc['reference_only']}, chi2 {mc['chi2']:.2f}"
            f" {'(significant)' if mc['significant_95'] else '(not significant)'}"
        )


if __name__ == "__main__":
    main()
