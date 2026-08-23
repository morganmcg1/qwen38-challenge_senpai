"""E144 R-B2: prove the requantized head leaves the coarse screen's readout alone.

The E143 coarse screen reads the 2-bit `draft_lm_head` and reranks the shortlist
through the BF16 precision islands. If those tensors are byte-identical between
the incumbent and the new head, the screen's arithmetic is bit-identical for any
given hidden state, so `recall_at_32` can only move through a change in the
hidden state itself, never through the readout.

This script settles the byte question. It does not settle the hidden-state
question, which needs the R-C replay.
"""

import argparse
import hashlib
import json
import os

from e144_ra import DECLARED
from e144_st import SafeTensors

READOUT_PREFIXES = ("draft_lm_head", "precision_islands")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="requantized head safetensors")
    parser.add_argument("--out", default="e144-rb2.json")
    arguments = parser.parse_args()

    declared = SafeTensors(DECLARED)
    candidate = SafeTensors(arguments.candidate)

    if sorted(declared.header) != sorted(candidate.header):
        raise SystemExit("tensor name sets differ; the heads are not interchangeable")

    identical, changed = [], []
    for name in declared.header:
        left_dtype, left_shape = declared.info(name)[:2]
        right_dtype, right_shape = candidate.info(name)[:2]
        if (left_dtype, left_shape) != (right_dtype, right_shape):
            raise SystemExit(f"{name}: dtype or shape changed, {left_dtype}{left_shape} -> {right_dtype}{right_shape}")
        left = declared.raw(name).tobytes()
        right = candidate.raw(name).tobytes()
        (identical if left == right else changed).append(
            {
                "tensor": name,
                "dtype": left_dtype,
                "shape": list(left_shape),
                "bytes": len(left),
                "sha256": hashlib.sha256(left).hexdigest()[:16],
            }
        )

    readout = [e for e in identical + changed if e["tensor"].startswith(READOUT_PREFIXES)]
    readout_changed = [e for e in changed if e["tensor"].startswith(READOUT_PREFIXES)]

    report = {
        "experiment": "e144",
        "rung": "R-B2",
        "harness": "local",
        "candidate": arguments.candidate,
        "declared": DECLARED,
        "tensor_count": len(declared.header),
        "identical_tensor_count": len(identical),
        "changed_tensor_count": len(changed),
        "changed_tensors": [e["tensor"] for e in changed],
        "readout_tensor_count": len(readout),
        "readout_tensors_byte_identical": not readout_changed,
        "readout_tensors": [e["tensor"] for e in readout],
        "payload_bytes_declared": sum(e["bytes"] for e in identical + changed),
        "e144_head_bytes_delta": os.path.getsize(arguments.candidate) - os.path.getsize(DECLARED),
        "e144_head_payload_bytes_delta": 0,
        "conclusion": (
            "the coarse screen's readout arithmetic is bit-identical, so recall_at_32 can "
            "only move through the hidden state the trunk produces"
            if not readout_changed
            else "the readout changed; recall_at_32 must be re-measured directly"
        ),
    }

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
