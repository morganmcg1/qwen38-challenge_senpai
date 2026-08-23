#!/usr/bin/env python3
"""Tripwire: prove which depth-price arm the timed binary actually shipped.

A depth price changes the round schedule, so the realised schedule is the
cheapest proof of which arm ran. CAMPAIGN RULE 114 applies: read the arm from
the run's own `score.json`, never from the variable the leg was asked with,
because an unrecognised `MLX_E134_DEPTH_PRICE_ARM` falls back to the compiled
default and is indistinguishable from a selector that never arrived.

E135 F33 restored the compiled default to `.pb6`. F23 had reverted it to
`.ship` on ranked receipt `e003a86d`, and the anchor-inversion panel then
showed that receipt drew the slow decode state (AUC 0.1964, p 0.993), so the
receipt priced a state draw and not the mechanism. The wanted arm is therefore
`pb6`, which sits at about 5.854 on this fixture. The `ship` arm signature is
6.358974358974359.

RULE 101. The two signatures differ by 0.505, so the band is set to 0.20 to
keep them disjoint. The check requires the observed schedule to match the
wanted arm AND to be outside the other arm's band, and it prints both
distances, so a leg that cannot discriminate is reported as a failure rather
than a pass.
"""
import argparse
import json
import sys

SIGNATURES = {
    "ship": 6.358974358974359,
    "pb6": 5.854,
}
TOLERANCE = 0.20

parser = argparse.ArgumentParser()
parser.add_argument("score_json")
parser.add_argument("--want", choices=sorted(SIGNATURES), default="pb6")
args = parser.parse_args()

with open(args.score_json) as handle:
    metrics = json.load(handle)["metrics"]

mean_draft = metrics["effective_mean_draft_len"]
accept = metrics["accepted_draft_rate"]
tokens = metrics["decode_tokens"]
rounds = tokens / (mean_draft * accept) if mean_draft * accept else float("nan")

other = next(a for a in SIGNATURES if a != args.want)
d_want = abs(mean_draft - SIGNATURES[args.want])
d_other = abs(mean_draft - SIGNATURES[other])

print(f"effective_mean_draft_len {mean_draft!r}")
print(f"accepted_draft_rate      {accept!r}")
print(f"decode_tokens            {tokens}")
print(f"implied rounds           {rounds:.2f}")
print(f"wanted arm               {args.want} @ {SIGNATURES[args.want]!r} "
      f"(distance {d_want:.4f})")
print(f"other arm                {other} @ {SIGNATURES[other]!r} "
      f"(distance {d_other:.4f})")
print(f"band                     +/- {TOLERANCE}")

if d_want > TOLERANCE:
    print(f"FAIL arm tripwire: the leg is {d_want:.4f} from the {args.want} "
          f"schedule, outside the {TOLERANCE} band")
    sys.exit(1)

if d_other <= TOLERANCE:
    print(f"FAIL arm tripwire: the leg is also within {TOLERANCE} of the "
          f"{other} schedule, so this witness cannot discriminate the arms")
    sys.exit(1)

print(f"PASS arm tripwire: the leg matches {args.want} to {d_want:.4f} and is "
      f"{d_other:.4f} from {other}, so the check could have failed")
sys.exit(0)
