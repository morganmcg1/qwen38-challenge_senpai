#!/usr/bin/env python3
"""E110 rung 4: collect the official ranked receipt for the `xv4` arm.

    YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
    python3 research/e110_rung4_receipt.py [--board PATH] [--out PATH]

Rung 3 passed every pre-submit gate, so the arm went to the official M5 runner
as submission `7bef7d4c`. This collector joins three sources into one record:

  * the Yukon board row, which carries the official score, the rejection
    reason, `parity_all_ok` and the eight per-prompt candidate seconds per
    token;
  * the two reference submissions the advisor named, `b8b8b860` and
    `44559d02`, so the candidate leg is compared against a measured board
    neighbour rather than against a local number; and
  * `rung1c-census.json`, which lets the receipt be graded directly against
    the register and occupancy prediction that was pre-registered for it.

Percentages are log-percent, matching `board_prompt_instrument.py`, so they
compare directly with the floors that tool measures.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
CENSUS = pathlib.Path("research/out/e110/rung1c-census.json")
OUT = pathlib.Path("research/out/e110/rung4-receipt.json")

CANDIDATE = "7bef7d4c"
REFERENCES = ("b8b8b860", "44559d02")

# The public prompt names the board reports only as SHA-256 digests.
PROMPT = {
    "919318e117fd04fd827e4cbc82abc30f37f6aea0e1c2609b27e3679c64c25fca": "beagle",
    "192fb6218ae7c1950dffb24b13ce3608857162bcd17853d3892f1d70876ad313": "botany",
    "4b9e88cdbe5aa2e76c03dcd76c5d2166e1e89c40ec2f777720aa87b225d7e1ff": "drama",
    "a2ea8b60458057ae731c11ff841eb95ad66f6f785b9bc8a6ef169d59ccca679b": "essays",
    "00142a4454b01e746b517c35eb23d1f248cd934bc3a743250b03745b9e711986": "medicine",
    "c1ec58669d032878b7fd82d811132bca19e79c91e19bf1414fe47c9b6d16e09e": "plutarch",
    "ea82dcb5931a2d78a42cebd4fb4189bf875a31c947b67792782e455f88a155a1": "republic",
    "3b10cb4dcae4b7f8222b1594861c4bbcc72aa890d72b46e97caee994884c82a5": "travel",
}

# The `board_prompt_instrument.py` probe definitions.
TARGET_PROBE = ("plutarch",)
DRAFT_PROBE = ("beagle", "botany", "essays", "medicine", "republic")

# Measured by `board_prompt_instrument.py --noise` at receipt time. The
# constants the tool currently applies are much smaller for TARGET, so both are
# recorded and the sigma is reported against the measured column.
FLOOR_MEASURED = {"TARGET": 0.5655, "DRAFT": 0.1780}
FLOOR_CONSERVATIVE = {"TARGET": 0.7998, "DRAFT": 0.2518}
FLOOR_IN_USE = {"TARGET": 0.0945, "DRAFT": 0.0952}

ROUND_WEIGHTS = {"2": 0.024, "3": 0.275, "4": 0.667, "5": 0.034}
GAMMA = 0.01346
REGISTER_FILE = {"applegpu_g16s": 384 * 1024, "applegpu_g17s": 496 * 1024}

# E110 rung-2 matched ABBA on the local M4 Pro host, in absolute candidate MTP
# seconds per token. The whole point of this rung is that the ranked host
# disagrees with it in sign.
LOCAL_ABBA_PCT = -0.7498
LOCAL_ABBA_CI = [-1.0393, -0.4602]


def logpct(a: float, b: float) -> float:
    return math.log(a / b) * 100.0


def rows(board: pathlib.Path) -> list[dict]:
    doc = json.loads(board.read_text())
    if isinstance(doc, dict):
        for key in ("submissions", "items", "data", "results"):
            if key in doc:
                return doc[key]
    return doc


def find(all_rows: list[dict], prefix: str) -> tuple[dict, dict]:
    for row in all_rows:
        if str(row.get("id", "")).startswith(prefix):
            metrics = row["officialMetrics"]
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
            return row, metrics
    raise SystemExit(f"submission {prefix} not on the board")


def per_prompt(metrics: dict) -> dict[str, dict]:
    return {PROMPT[p["prompt_sha256"]]: p for p in metrics["per_prompt"]}


def occupancy(registers: int, arch: str) -> int:
    return REGISTER_FILE[arch] // (128 * registers)


def census_grade(path: pathlib.Path) -> dict:
    """Grade the receipt against the pre-registered register prediction.

    The advisor pre-registered the g17s NA2 83->93 and NA5 98->101 pair as the
    first suspect for any ranked underperformance. Weighting every width by the
    realised verify-width histogram tests that claim instead of quoting the two
    regressing widths alone.
    """
    doc = json.loads(path.read_text())
    base, arm = doc["arms"]["a_base"], doc["arms"]["xv4"]
    widths, totals = [], {"d_reg_local": 0.0, "d_reg_ranked": 0.0,
                          "d_air_lines": 0.0, "d_omega_ranked_pct": 0.0,
                          "d_omega_local_pct": 0.0}
    for width, weight in ROUND_WEIGHTS.items():
        cell = {"width": int(width), "round_weight": weight}
        for arch, tag in (("applegpu_g16s", "local"), ("applegpu_g17s", "ranked")):
            rb, ra = base[arch][width]["registers"], arm[arch][width]["registers"]
            cell[f"registers_{tag}_base"] = rb
            cell[f"registers_{tag}_arm"] = ra
            cell[f"spill_{tag}_base"] = base[arch][width]["spill_bytes"]
            cell[f"spill_{tag}_arm"] = arm[arch][width]["spill_bytes"]
            ratio = occupancy(rb, arch) / occupancy(ra, arch)
            cell[f"d_omega_{tag}_pct"] = (ratio ** GAMMA - 1) * 100
            totals[f"d_reg_{tag}"] += weight * (ra - rb)
            totals[f"d_omega_{tag}_pct"] += weight * cell[f"d_omega_{tag}_pct"]
        cell["device_loads_base"] = base["air"][width]["device_loads"]
        cell["device_loads_arm"] = arm["air"][width]["device_loads"]
        cell["air_lines_base"] = base["air"][width]["air_lines"]
        cell["air_lines_arm"] = arm["air"][width]["air_lines"]
        totals["d_air_lines"] += weight * (cell["air_lines_arm"]
                                           - cell["air_lines_base"])
        widths.append(cell)
    return {"per_width": widths, "weighted": totals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=pathlib.Path, default=BOARD)
    ap.add_argument("--census", type=pathlib.Path, default=CENSUS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    if not args.board.exists():
        print(f"no {args.board}; run `board_per_prompt.py fetch` first",
              file=sys.stderr)
        return 1

    all_rows = rows(args.board)
    row, metrics = find(all_rows, CANDIDATE)
    cand = per_prompt(metrics)
    refs = {name: find(all_rows, name) for name in REFERENCES}

    prompts = []
    for name in sorted(cand, key=lambda n: -cand[n]["mtp_seconds_per_token_mean"]):
        entry = cand[name]
        rec = {
            "prompt": name,
            "candidate_seconds_per_token": entry["mtp_seconds_per_token_mean"],
            "serial_seconds_per_token": entry["serial_seconds_per_token_mean"],
            "raw_ratio_of_means": entry["raw_ratio_of_means"],
            "effective_mean_draft_len": entry["effective_mean_draft_len"],
            "non_drafting_round_count": entry["non_drafting_round_count"],
            "parity_ok": entry["parity_ok"],
            "head_provenance_sha256": entry["head_provenance_sha256"],
        }
        for ref, (_, rmetrics) in refs.items():
            other = per_prompt(rmetrics)[name]["mtp_seconds_per_token_mean"]
            rec[f"vs_{ref}_logpct"] = logpct(
                entry["mtp_seconds_per_token_mean"], other)
        prompts.append(rec)

    by_name = {p["prompt"]: p for p in prompts}
    probes = {}
    for probe, members in (("TARGET", TARGET_PROBE), ("DRAFT", DRAFT_PROBE)):
        for ref in REFERENCES:
            effect = statistics.fmean(
                by_name[m][f"vs_{ref}_logpct"] for m in members)
            probes[f"{probe}_vs_{ref}"] = {
                "members": list(members),
                "effect_logpct": effect,
                "sigma_vs_measured_floor": effect / FLOOR_MEASURED[probe],
                "sigma_vs_floor_in_use": effect / FLOOR_IN_USE[probe],
                "floor_measured_pct": FLOOR_MEASURED[probe],
                "floor_conservative_pct": FLOOR_CONSERVATIVE[probe],
                "floor_in_use_pct": FLOOR_IN_USE[probe],
            }

    # The edited kernel is the wide multi-row path, so it is reached only when
    # the candidate drafts. Splitting on that is the causal test.
    groups = {}
    for tag, pick in (("drafting", lambda p: p["non_drafting_round_count"] == 0),
                      ("non_drafting", lambda p: p["non_drafting_round_count"] > 0)):
        members = [p for p in prompts if pick(p)]
        groups[tag] = {
            "prompts": [p["prompt"] for p in members],
            "n": len(members),
            "mean_logpct_vs_b8b8b860": statistics.fmean(
                p["vs_b8b8b860_logpct"] for p in members),
        }

    cand_mean = metrics["candidate_mtp_seconds_per_token_mean"]
    aggregate = {
        f"candidate_time_vs_{ref}_logpct": logpct(
            cand_mean, rmetrics["candidate_mtp_seconds_per_token_mean"])
        for ref, (_, rmetrics) in refs.items()
    }

    doc = {
        "experiment": "e110",
        "rung": 4,
        "arm": "xv4",
        "harness": "ranked",
        "submission_id": row["id"],
        "created_at": row["createdAt"],
        "official_score": row["officialScore"],
        "improved": row["improved"],
        "promotion_status": row.get("promotionStatus"),
        "rejection_reason": row.get("rejectionReason"),
        "yukon_commit": metrics["commit"],
        "official_metrics": {k: v for k, v in metrics.items() if k != "per_prompt"},
        "per_prompt": prompts,
        "probes": probes,
        "groups": groups,
        "aggregate": aggregate,
        "local_abba_pct": LOCAL_ABBA_PCT,
        "local_abba_ci": LOCAL_ABBA_CI,
        "local_to_ranked_swing_pp": (
            groups["drafting"]["mean_logpct_vs_b8b8b860"] - LOCAL_ABBA_PCT),
        "census_grade": census_grade(args.census),
        "references": {
            ref: {"id": rrow["id"], "official_score": rrow["officialScore"],
                  "candidate_mtp_seconds_per_token_mean":
                      rmetrics["candidate_mtp_seconds_per_token_mean"]}
            for ref, (rrow, rmetrics) in refs.items()
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.out}")
    print(f"official score {doc['official_score']}  "
          f"rejection: {doc['rejection_reason']}")
    print(f"parity_all_ok {metrics['parity_all_ok']}  "
          f"aggregate candidate time {aggregate}")
    for tag, rec in groups.items():
        print(f"  {tag:<13} n={rec['n']} "
              f"mean {rec['mean_logpct_vs_b8b8b860']:+.4f} %")
    print(f"  weighted census prediction "
          f"{doc['census_grade']['weighted']['d_omega_ranked_pct']:+.4f} % "
          f"of QMV kernel time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
