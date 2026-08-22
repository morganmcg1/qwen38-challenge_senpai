#!/usr/bin/env python3
"""Tripwire: prove the composed tree still carries edward's pb6 pass boundary.

pb6 changes the round schedule, so the schedule is the cheapest proof that the
rebase kept it. The pre-pb6 ship commit measured a mean draft length of
6.358974358974359 over 78 rounds on this fixture. pb6 trades draft length for
cheaper verification and measures about 5.854 over about 82 rounds.

A composed leg that reproduces the pre-pb6 number to the digit means the rebase
dropped pb6 and nothing downstream of this check is a composition.
"""
import json
import sys

PRE_PB6_MEAN_DRAFT = 6.358974358974359
EXPECTED_PB6_MEAN_DRAFT = 5.854
TOLERANCE = 0.40

path = sys.argv[1]
with open(path) as handle:
    metrics = json.load(handle)["metrics"]

mean_draft = metrics["effective_mean_draft_len"]
accept = metrics["accepted_draft_rate"]
tokens = metrics["decode_tokens"]
rounds = tokens / (mean_draft * accept) if mean_draft * accept else float("nan")

print(f"effective_mean_draft_len {mean_draft!r}")
print(f"accepted_draft_rate      {accept!r}")
print(f"decode_tokens            {tokens}")
print(f"implied rounds           {rounds:.2f}")
print(f"pre-pb6 signature        {PRE_PB6_MEAN_DRAFT!r}")
print(f"expected pb6 signature   ~{EXPECTED_PB6_MEAN_DRAFT} (+/- {TOLERANCE})")

if mean_draft == PRE_PB6_MEAN_DRAFT:
    print("FAIL pb6 tripwire: the leg reproduces the pre-pb6 schedule exactly; "
          "the rebase dropped pb6")
    sys.exit(1)

delta = abs(mean_draft - EXPECTED_PB6_MEAN_DRAFT)
if delta > TOLERANCE:
    print(f"FAIL pb6 tripwire: mean draft is {delta:.4f} from the expected pb6 "
          f"schedule, outside the {TOLERANCE} band; inspect before shipping")
    sys.exit(1)

print(f"PASS pb6 tripwire: schedule moved off pre-pb6 and sits {delta:.4f} "
      f"from the expected pb6 value")
sys.exit(0)
