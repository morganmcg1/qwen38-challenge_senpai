"""Recover the per-prompt round count R from a board row.

`effective_mean_draft_len` is reported as D/R with D and R both non-negative
integers: D is the total number of drafts proposed over the window and R is the
number of rounds. The board publishes ten significant digits, so the smallest R
that makes D integral is recoverable by rational reconstruction. This removes
the F1 defect that forced `rankedcurve.py` to hard-code one ROUNDS vector and
to keep only reference-schedule rows.

Bounds:
  R <= 512                      every round emits at least the primary token
  R >= 512 / (1 + dl)           a round emits at most 1 + drafts tokens
  R >= non_drafting_round_count those rounds are a subset of all rounds
"""

import json
from fractions import Fraction

PROMPTS = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
TOKENS = 512
BOARD = "/tmp/yukon-board/full.json"


def load_rows(path=BOARD):
    raw = json.load(open(path))
    rows = raw
    if isinstance(raw, dict):
        for k in ("submissions", "rows", "data", "items"):
            if k in raw:
                rows = raw[k]
                break
    return [r for r in rows if isinstance(r, dict)]


def per_prompt(row):
    om = row.get("officialMetrics") or {}
    out = {}
    for e in om.get("per_prompt") or []:
        name = PROMPTS.get((e.get("prompt_sha256") or "")[:8])
        if name:
            out[name] = e
    return out


def recover_rounds(dl, n0, tokens=TOKENS):
    """Smallest round count R consistent with dl = D/R and the token budget.

    Returns (R, multiplicity). Multiplicity counts the legal R values that
    reproduce the printed decimal; 1 means the round count is uniquely pinned.
    """
    if dl == 0:
        return (n0 if n0 else tokens), 1
    frac = Fraction(repr(dl)).limit_denominator(tokens)
    base = frac.denominator
    lo = max(1, n0, int(tokens / (1.0 + dl)))
    cands = [k * base for k in range(1, tokens // base + 1) if lo <= k * base <= tokens]
    if not cands:
        return None, 0
    return cands[0], len(cands)
