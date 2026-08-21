"""Publish a recorded --local-submit gate score.json as a W&B run.

Usage: e87_gate_wandb_publish.py <out-dir> <run-name> [key=value ...]
"""

import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def main():
    out = pathlib.Path(sys.argv[1])
    name = sys.argv[2]
    extra = dict(kv.split("=", 1) for kv in sys.argv[3:])
    d = json.loads((out / "score.json").read_text())
    metrics = d["metrics"]
    meta = {}
    meta_path = out / "meta.txt"
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        job_type="local-submit-gate",
        group="e87-draw2",
        tags=["e87", "draw2", "local-submit", "gate", "arm-c", "section8"],
        config={
            "experiment": "E87",
            "student": "qwen-thorfinn",
            "pr": 89,
            "assignment_id": "qwen38-r1-e87-coarse-draft-shortlist-traffic",
            "revision_id": "r2",
            "harness": "local",
            "host": "ip-10-231-2-95.ec2.internal",
            "chip": "Apple M4 Pro Mac16,11",
            "track_id": d["track_id"],
            "primary_metric": "mtp_seconds_per_token",
            "primary_direction": "minimize",
            **meta,
            **extra,
        },
    )
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
    flat["score"] = d["score"]
    flat["passed"] = d["passed"]
    run.log(flat)
    run.summary.update(flat)
    for k, v in metrics.items():
        if isinstance(v, str):
            run.summary[k] = v
    art = wandb.Artifact(f"{name}-score", type="measurement")
    art.add_file(str(out / "score.json"))
    if meta_path.exists():
        art.add_file(str(meta_path))
    run.log_artifact(art)
    rid, url = run.id, run.url
    run.finish()
    print(json.dumps({"run_id": rid, "url": url}, indent=1))


if __name__ == "__main__":
    main()
