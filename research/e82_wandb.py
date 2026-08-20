#!/usr/bin/env python3
"""E82: stream the head provenance audit, the builds and the acceptance screen to W&B.

One run holds every fact another agent would need to reproduce or overturn the
rung-0 decision: which published head is a real fine-tune and by how much, what
requantizing each trunk costs, whether each built arm meets the three hard
constraints, and how the arms compare on the untimed acceptance screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e82-requantize-the-only-genuinely-retrained-head"


def cell(value):
    # W&B table cells must be scalars; nested audit blocks are kept as JSON
    # text rather than dropped, so nothing in the record is lost.
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[cell(row.get(c)) for c in columns])
    return t


def log_audit(run, path: Path) -> None:
    audit = json.loads(path.read_text())
    run.log({"audit/raw": wandb.Table(columns=["json"], data=[[json.dumps(audit, indent=2)]])})
    for section, payload in audit.items():
        if isinstance(payload, dict) and all(isinstance(v, dict) for v in payload.values()):
            columns = ["tensor"] + sorted({k for v in payload.values() for k in v})
            rows = [{"tensor": name, **values} for name, values in payload.items()]
            run.log({f"audit/{section}": table(columns, rows)})


def log_builds(run, reports: list[Path]) -> None:
    constraint_rows, damage_rows, island_rows = [], [], []
    for path in reports:
        report = json.loads(path.read_text())
        checks = report["constraints"]
        constraint_rows.append(
            {
                "tag": report["tag"],
                "submission_eligible": checks.get("submission_eligible"),
                "bytes": checks["bytes"],
                "reference_bytes": checks["reference_bytes"],
                "reference_artifact": checks.get("reference_artifact"),
                "bytes_delta_pct": checks["bytes_delta_pct"],
                "tensor_count": checks["tensor_count"],
                "draft_lm_head_byte_identical": all(checks["draft_lm_head_byte_identical"].values()),
                "norms_byte_identical_to_source": all(checks["norms_byte_identical_to_source"].values()),
                "shapes_match_reference": all(checks["shapes_match_reference"].values()),
                "tree_sha256": report["tree_sha256"],
                "tree_bytes": report["tree_bytes"],
                "trunk_source_sha256": report["metadata"]["trunk_source_sha256"],
            }
        )
        for name, row in report["quantization_damage"].items():
            damage_rows.append({"tag": report["tag"], "tensor": name, **row})
        for proj, row in report.get("islands", {}).items():
            island_rows.append({"tag": report["tag"], "projection": proj, **row})

    run.log(
        {
            "builds/constraints": table(sorted({k for r in constraint_rows for k in r}), constraint_rows),
            "builds/quantization_damage": table(
                sorted({k for r in damage_rows for k in r}), damage_rows
            ),
            "builds/precision_islands": table(
                sorted({k for r in island_rows for k in r}), island_rows
            ),
        }
    )


def log_screen(run, path: Path) -> None:
    screen = json.loads(path.read_text())
    ref = screen["reference_arm"]

    work_rows, depth_rows, pooled_rows, paired_rows, rule_rows = [], [], [], [], []
    for arm, entry in screen["arms"].items():
        w = entry["work"]
        work_rows.append(
            {
                "arm": arm,
                "is_reference": arm == ref,
                "head_sha256": (entry["head_provenance_sha256"] or ["?"])[0],
                "head_bytes": (entry["head_bytes"] or [None])[0],
                "parity_all_ok": entry["parity_all_ok"],
                "mean_rounds_per_512": w["mean_rounds_per_512"],
                "rounds_delta_pct": entry.get("rounds_delta_pct", 0.0),
                "mean_accepted_per_round": w["mean_accepted_per_round"],
                "mean_drafted_per_round": w["mean_drafted_per_round"],
                "rows_per_token": w["rows_per_token"],
            }
        )
        for split, block in entry["splits"].items():
            for depth, cell in block["per_depth"].items():
                depth_rows.append({"arm": arm, "split": split, "depth": int(depth), **cell})
            pooled_rows.append({"arm": arm, "split": split, **block["pooled_3_6"]})
        for split, mc in entry.get("paired_vs_reference", {}).items():
            paired_rows.append({"arm": arm, "split": split, **mc})
        rule = entry.get("stop_rule")
        if rule:
            rule_rows.append(
                {
                    "arm": arm,
                    "pooled_delta_points": rule["pooled_delta_points"],
                    "pooled_gate_passed": rule["pooled_gate_passed"],
                    "hardest_gate_passed": rule["hardest_gate_passed"],
                    "advance": rule["advance"],
                    **{f"hardest_delta_d{d}": v for d, v in rule["hardest_delta_points_by_depth"].items()},
                }
            )
        # Scalars so the arms are directly plottable, not only inspectable.
        run.log(
            {
                f"screen/{arm}/mean_rounds_per_512": w["mean_rounds_per_512"],
                f"screen/{arm}/mean_accepted_per_round": w["mean_accepted_per_round"],
                f"screen/{arm}/rows_per_token": w["rows_per_token"],
                f"screen/{arm}/pooled_accept_d3_d6": entry["splits"]["pooled"]["pooled_3_6"]["p"],
                f"screen/{arm}/hardest_accept_d3_d6": entry["splits"]["hardest"]["pooled_3_6"]["p"],
                f"screen/{arm}/easiest_accept_d3_d6": entry["splits"]["easiest"]["pooled_3_6"]["p"],
            }
        )

    run.log(
        {
            "screen/work": table(sorted({k for r in work_rows for k in r}), work_rows),
            "screen/per_depth": table(sorted({k for r in depth_rows for k in r}), depth_rows),
            "screen/pooled_d3_d6": table(sorted({k for r in pooled_rows for k in r}), pooled_rows),
            "screen/paired_mcnemar": table(sorted({k for r in paired_rows for k in r}), paired_rows),
            "screen/stop_rule": table(sorted({k for r in rule_rows for k in r}), rule_rows),
        }
    )
    return screen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e82-requantized-head-screen")
    ap.add_argument("--audit", default="research/e82-head-audit.json")
    ap.add_argument("--island-replay", default="research/e82-island-rule-replay.json")
    ap.add_argument("--builds", nargs="*", default=sorted(str(p) for p in Path("research").glob("e82-build-*.json")))
    ap.add_argument("--screen", default="research/e82-accept.json")
    ap.add_argument("--corpus", default="research/e82-corpus-manifest.json")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    screen_path = Path(args.screen)
    screen = json.loads(screen_path.read_text()) if screen_path.exists() else {}

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        job_type="research-screen",
        notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "harness": "local",
            "verb": "mtp-verify",
            "timed": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "offered_depth": 8,
            "decode_tokens": screen.get("steps", 512),
            "seed_tokens": corpus["seed_tokens"],
            "seeds": [s["name"] for s in corpus["seeds"]],
            "domains": sorted({s["domain"] for s in corpus["seeds"]}),
            "tokenizer_sha256": corpus["tokenizer_sha256"],
            "reference_arm": screen.get("reference_arm"),
            "rule_depths": screen.get("rule_depths"),
            "declared_head_bytes": 427742600,
        },
    )

    run.log(
        {
            "corpus/seeds": table(
                ["name", "domain", "gutenberg_id", "tokens", "chars", "sha256"], corpus["seeds"]
            )
        }
    )
    log_audit(run, Path(args.audit))
    replay = Path(args.island_replay)
    if replay.exists():
        run.log(
            {
                "audit/island_rule_replay": wandb.Table(
                    columns=["json"], data=[[replay.read_text()]]
                )
            }
        )
    log_builds(run, [Path(p) for p in args.builds])
    if screen:
        log_screen(run, screen_path)

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
