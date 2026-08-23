"""E146 R-C: copy the fixed-binary leg evidence out of the private run directory.

harness=local. The raw run directory is gitignored, so this writes the small
audit fields into `research/e146-rc-legs.json` where another agent can read the
same numbers the R-C statistics were computed from.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEGS = os.path.join(ROOT, ".mlxfast-private", "e146", "rc", "legs.jsonl")
OUT = os.path.join(HERE, "e146-rc-legs.json")

LEG_FIELDS = [
    "label", "kind", "leg_depth", "started_utc", "finished_utc",
    "gpu_temp_entry_c", "gpu_temp_exit_c", "cool_gate_wait_seconds",
    "cool_gate_passed_real_gate", "gate_qualified_for_timing",
    "worker_sha256_after", "worker_unchanged", "exit_code",
]
METRIC_FIELDS = [
    "parent_measured_seconds_per_token", "blocks_only_seconds",
    "seed_prefill_seconds", "prefill_seconds_per_token", "decode_seconds",
    "decode_token_count", "round_count", "non_drafting_round_count",
    "block_count", "effective_mean_draft_len", "effective_max_draft_len",
    "accepted_draft_rate", "accepted_draft_total", "rejected_draft_total",
    "emitted_token_total", "declared_rows_total", "reference_checked_row_total",
    "all_tokens_matched", "parity_all_ok", "residual_divergence_count",
    "is_serial_control", "first_block_seconds", "p50_block_request_seconds",
    "mtp_depth",
]


def main():
    if not os.path.exists(LEGS):
        print("no R-C legs at %s" % LEGS)
        return 1
    legs = []
    for line in open(LEGS):
        leg = json.loads(line)
        record = {k: leg[k] for k in LEG_FIELDS if k in leg}
        record["metrics"] = {k: leg["metrics"][k] for k in METRIC_FIELDS
                             if k in leg["metrics"]}
        legs.append(record)
    doc = {
        "harness": "local",
        "host": "aws-mac student qwen-askeladd",
        "prompt": "beagle_a",
        "decode_tokens": 512,
        "offered_depth": 8,
        "cool_gate": "real 40C gate, MLXFAST_LOCAL_COOL_GATE unset",
        "order": "mtp,serial,mtp,mtp,mtp,serial,mtp,mtp,mtp,serial,mtp",
        "source": ".mlxfast-private/e146/rc/legs.jsonl",
        "legs": legs,
    }
    with open(OUT, "w") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
    print("wrote %s with %d legs" % (OUT, len(legs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
