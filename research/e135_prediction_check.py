#!/usr/bin/env python3
"""Score the four predictions frozen before the `572b2cc4` receipt was read.

P1 the candidate 8-prompt mean decode time must FALL; a rise of 0.10 % or more
   falsifies it.
P2 the pinned serial baseline must not move; 0.5 % in either direction
   falsifies it.
P3 `effective_mean_draft_len` must be equal to the digit on every prompt,
   because the launch grid changes dispatch cost and not behaviour.
P4 parity must hold on all eight prompts.
"""

import glob
import json
import sys

P1_FALSIFY_PCT = 0.10
P2_FALSIFY_PCT = 0.50


def load(prefix: str) -> dict:
    hits = [p for p in glob.glob("research/out/receipts/*.json") if prefix in p]
    if not hits:
        raise SystemExit("no cached receipt for %s" % prefix)
    return json.load(open(hits[0]))


before = load(sys.argv[1])
after = load(sys.argv[2])
old, new = before["officialMetrics"], after["officialMetrics"]

candidate = 100.0 * (new["candidate_mtp_seconds_per_token_mean"]
                     / old["candidate_mtp_seconds_per_token_mean"] - 1.0)
serial = 100.0 * (new["baseline_serial_seconds_per_token_mean"]
                  / old["baseline_serial_seconds_per_token_mean"] - 1.0)

print("P1 candidate mtp s/token mean  %.9f -> %.9f  %+.4f %%  %s"
      % (old["candidate_mtp_seconds_per_token_mean"],
         new["candidate_mtp_seconds_per_token_mean"], candidate,
         "HELD" if candidate < P1_FALSIFY_PCT else "FALSIFIED"))
print("P2 pinned serial s/token mean  %.9f -> %.9f  %+.4f %%  %s"
      % (old["baseline_serial_seconds_per_token_mean"],
         new["baseline_serial_seconds_per_token_mean"], serial,
         "HELD" if abs(serial) < P2_FALSIFY_PCT else "FALSIFIED"))

by_hash = {p["prompt_sha256"]: p for p in old["per_prompt"]}
identical = 0
print("\n%-10s %-22s %-22s %-9s %s"
      % ("prompt", "draft len before", "draft len after", "identical", "raw ratio"))
for p in new["per_prompt"]:
    q = by_hash[p["prompt_sha256"]]
    same = q["effective_mean_draft_len"] == p["effective_mean_draft_len"]
    identical += same
    print("%-10s %-22r %-22r %-9s %.8f -> %.8f"
          % (p["prompt_sha256"][:8], q["effective_mean_draft_len"],
             p["effective_mean_draft_len"], same,
             q["raw_ratio_of_means"], p["raw_ratio_of_means"]))

print("\nP3 schedule identical on %d of %d prompts  %s"
      % (identical, len(new["per_prompt"]),
         "HELD" if identical == len(new["per_prompt"]) else "FALSIFIED"))
parity = new["parity_all_ok"] and all(p["parity_ok"] for p in new["per_prompt"])
print("P4 parity on all eight prompts  %s" % ("HELD" if parity else "FALSIFIED"))

for name, metrics in (("before", old), ("after", new)):
    pairs = sorted((p["raw_ratio_of_means"], p["prompt_sha256"][:8])
                   for p in metrics["per_prompt"])
    print("%-6s medpair 4th=%s 5th=%s  median %.8f"
          % (name, pairs[3], pairs[4], (pairs[3][0] + pairs[4][0]) / 2))
