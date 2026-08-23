"""E146 shared board instrument: rounds, the drafting-round basis, and the k fit.

harness=ranked for every quantity computed here. Each input is a published
per-prompt field from a ranked receipt; nothing local enters this file.

THE ROUND COUNT IS NOT PUBLISHED. The board publishes `effective_mean_draft_len`
(drafts proposed per round, averaged over ALL rounds) and the exact
`non_drafting_round_count`, but no round count and no accept rate. F219 closes
that with the identity

    R = 512 / (1 + a*d)

where `a` is the fraction of proposed drafts that are accepted. `a` is a
property of the prompt and the schedule, not of the candidate's kernels, so it
is recovered once per prompt from the F219 anchor table and carried to other
draft lengths through a geometric chain model: a per-step acceptance `p` implies
`a*d = p(1 - p^d)/(1 - p)`. Inside one draft-length cluster no extrapolation
happens at all, because `d` is digit-identical to the anchor's.

Verified against the live receipt for the anchor `1760479a`: the eight published
`effective_mean_draft_len` values reproduce the F219 table to the printed
digits, and every implied drafting-round count is non-negative and at least the
published `512 - non_drafting_round_count` floor.
"""

import json
import math
import os

DECODE_TOKENS = 512

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
PROMPT_ORDER = [
    "beagle", "botany", "drama", "essays",
    "medicine", "plutarch", "republic", "travel",
]

# FINDING 219, advisor comment 5383899804: the ranked anchor table for
# `1760479a`. prompt -> (candidate seconds per token, mean draft length, rounds).
F219_ANCHOR = {
    "beagle": (0.0106863, 4.3818, 110.00),
    "botany": (0.0096534, 6.1481, 81.04),
    "drama": (0.0178217, 2.2976, 252.01),
    "essays": (0.0098320, 5.0870, 92.04),
    "medicine": (0.0097197, 5.2556, 90.01),
    "plutarch": (0.0300828, 0.1557, 486.76),
    "republic": (0.0097098, 4.9892, 93.00),
    "travel": (0.0156178, 2.6479, 212.33),
}

LIVE_BOARD = "/tmp/yukon-board/full.json"
FROZEN_BOARD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "board-per-prompt-2026-08-23.json")

FROZEN_FIELDS = [
    "accepted_pair_count", "effective_mean_draft_len", "head_provenance_sha256",
    "mtp_seconds_per_token_mean", "non_drafting_round_count",
    "noop_reference_decode_speedup", "parity_ok", "prefill_seconds_per_token",
    "raw_ratio_of_means", "serial_seconds_per_token_mean",
]


def _anchor_accept_rate(prompt):
    _, draft_len, rounds = F219_ANCHOR[prompt]
    if draft_len <= 0:
        return 0.0
    return (DECODE_TOKENS / rounds - 1.0) / draft_len


def _chain_accepted(step_p, width):
    if width <= 0:
        return 0.0
    if step_p >= 1.0:
        return width
    return step_p * (1.0 - step_p ** width) / (1.0 - step_p)


def _implied_step_p(prompt):
    """Per-step acceptance behind the anchor's (a, d) for this prompt."""
    _, draft_len, _ = F219_ANCHOR[prompt]
    target = _anchor_accept_rate(prompt) * draft_len
    low, high = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _chain_accepted(mid, draft_len) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


STEP_P = {name: _implied_step_p(name) for name in PROMPT_ORDER}
ANCHOR_ACCEPT = {name: _anchor_accept_rate(name) for name in PROMPT_ORDER}


def round_count(prompt, draft_len):
    """R for this prompt at this mean draft length, through the F219 identity."""
    if draft_len <= 0:
        return float(DECODE_TOKENS)
    accepted_per_round = _chain_accepted(STEP_P[prompt], draft_len)
    return DECODE_TOKENS / (1.0 + accepted_per_round)


class Row(object):
    """One ranked receipt, reduced to the eight per-prompt fields E146 needs."""

    __slots__ = ("id8", "id", "solver", "created", "score", "status",
                 "promotion", "note", "source_ref", "commit", "prompts")

    def __init__(self, raw, per_prompt):
        self.id = raw.get("id") or raw.get("id8")
        self.id8 = self.id[:8]
        self.solver = raw.get("solverUsername") or raw.get("solver") or ""
        self.created = raw.get("createdAt") or ""
        self.score = raw.get("officialScore")
        self.status = raw.get("status")
        self.promotion = raw.get("promotionStatus")
        self.note = raw.get("note") or ""
        self.source_ref = raw.get("promotedSourceRef")
        self.commit = raw.get("submissionCommitSha")
        self.prompts = per_prompt

    def cand(self, prompt):
        return self.prompts[prompt]["mtp_seconds_per_token_mean"]

    def prefill(self, prompt):
        return self.prompts[prompt].get("prefill_seconds_per_token")

    def has_prefill(self):
        return all(self.prefill(p) for p in PROMPT_ORDER)

    def decode(self, prompt):
        """Candidate leg with the charged 512-token seed prefill removed.

        FINDING 227: the trusted driver starts its clock before the seed
        prefill, so `mtp_seconds_per_token_mean` contains it. A per-drafting-round
        overhead cannot live in the prefill, which runs no drafting rounds at
        all, so leaving the prefill in the response gives a per-leg prefill
        change nowhere legitimate to land and it is absorbed into k.
        Confirmed on this board in `e146_prefill.py`: the implied absolute seed
        prefill is 0.52649 s with a CV of 0.108 % across eight prompts whose
        candidate times differ by 3.1x.
        """
        pre = self.prefill(prompt)
        return self.cand(prompt) - pre if pre else self.cand(prompt)

    def serial(self, prompt):
        return self.prompts[prompt]["serial_seconds_per_token_mean"]

    def dlen(self, prompt):
        return self.prompts[prompt]["effective_mean_draft_len"]

    def non_drafting(self, prompt):
        return self.prompts[prompt]["non_drafting_round_count"]

    def head(self, prompt):
        return (self.prompts[prompt].get("head_provenance_sha256") or "")[:12]

    def draft_key(self):
        """T32 schedule fingerprint: the eight draft lengths, exact digits."""
        return tuple(repr(self.dlen(p)) for p in PROMPT_ORDER)

    def raw_ratio(self, prompt):
        return self.prompts[prompt]["raw_ratio_of_means"]

    def published_median(self):
        vals = sorted(self.raw_ratio(p) for p in PROMPT_ORDER)
        return 0.5 * (vals[3] + vals[4])

    def cand_mean8(self):
        return sum(self.cand(p) for p in PROMPT_ORDER) / 8.0

    def basis(self):
        """drafting rounds per second of candidate leg, per prompt.

        A once-per-drafting-round overhead of `k` microseconds moves the
        candidate leg of prompt p by `k * 1e-6 * basis_p * 100` per cent.
        """
        out = {}
        for p in PROMPT_ORDER:
            rounds = round_count(p, self.dlen(p))
            drafting = rounds - self.non_drafting(p)
            total = DECODE_TOKENS * self.cand(p)
            out[p] = drafting / total
        return out

    def drafting_rounds(self):
        return {p: round_count(p, self.dlen(p)) - self.non_drafting(p)
                for p in PROMPT_ORDER}


def _per_prompt_from_live(raw):
    om = raw.get("officialMetrics")
    if not isinstance(om, dict):
        return None
    entries = om.get("per_prompt")
    if not isinstance(entries, list) or len(entries) != 8:
        return None
    out = {}
    for entry in entries:
        name = PROMPTS.get((entry.get("prompt_sha256") or "")[:8])
        if name is None:
            return None
        if entry.get("mtp_seconds_per_token_mean") in (None, 0):
            return None
        if entry.get("serial_seconds_per_token_mean") in (None, 0):
            return None
        out[name] = entry
    return out if len(out) == 8 else None


def _per_prompt_from_frozen(raw, fields):
    out = {}
    for name, values in (raw.get("per_prompt") or {}).items():
        if name not in PROMPT_ORDER:
            return None
        entry = dict(zip(fields, values))
        if not entry.get("mtp_seconds_per_token_mean"):
            return None
        if not entry.get("serial_seconds_per_token_mean"):
            return None
        out[name] = entry
    return out if len(out) == 8 else None


def load(path=None):
    """Scored rows from either the live payload or the committed export."""
    path = path or (LIVE_BOARD if os.path.exists(LIVE_BOARD) else FROZEN_BOARD)
    with open(path) as handle:
        payload = json.load(handle)
    rows = []
    if isinstance(payload, dict) and "rows" in payload:
        fields = payload.get("per_prompt_fields") or FROZEN_FIELDS
        for raw in payload["rows"]:
            per_prompt = _per_prompt_from_frozen(raw, fields)
            if per_prompt:
                rows.append(Row(raw, per_prompt))
    else:
        if isinstance(payload, dict):
            payload = payload.get("submissions") or []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            if not isinstance(raw.get("officialScore"), (int, float)):
                continue
            per_prompt = _per_prompt_from_live(raw)
            if per_prompt:
                rows.append(Row(raw, per_prompt))
    rows.sort(key=lambda r: r.created)
    return path, rows


def leg(row, prompt, field="decode"):
    """The response variable: the candidate leg, with or without the prefill.

    `decode` is the default because the state being measured is charged per
    drafting round and the seed prefill runs none. `total` reproduces the
    pre-F227 estimator and exists only for the before-and-after comparison.
    """
    if field == "decode":
        return row.decode(prompt)
    if field == "total":
        return row.cand(prompt)
    if field == "prefill":
        return row.prefill(prompt)
    if field == "serial":
        return row.serial(prompt)
    raise ValueError(field)


def pct_diff(anchor, row, field="decode"):
    """Per-prompt candidate-leg percentage difference, row against anchor."""
    return {p: 100.0 * (leg(row, p, field) / leg(anchor, p, field) - 1.0)
            for p in PROMPT_ORDER}


def serial_pct_diff(anchor, row):
    return {p: 100.0 * (row.serial(p) / anchor.serial(p) - 1.0) for p in PROMPT_ORDER}


def design_matrix(anchor, row, unit="drafting_round", field="decode"):
    """Percentage moved per microsecond of per-unit overhead carried by `row`.

    The model is t_p(row) = t_p(anchor) + k * count_p(row). Dividing by the
    ANCHOR's time is what makes the fitted k a property of the row rather than
    of its own slowdown: using the row's own time shrinks the basis exactly
    where k is large and inflates k by that same factor.
    """
    out = {}
    for p in PROMPT_ORDER:
        rounds = round_count(p, row.dlen(p))
        if unit == "drafting_round":
            count = rounds - row.non_drafting(p)
        elif unit == "round":
            count = rounds
        elif unit == "non_drafting_round":
            count = row.non_drafting(p)
        elif unit == "draft_step":
            count = row.dlen(p) * rounds
        elif unit == "verify_row":
            count = rounds * (1.0 + row.dlen(p))
        elif unit == "emitted_token":
            count = float(DECODE_TOKENS)
        elif unit == "flat":
            count = DECODE_TOKENS * leg(anchor, p, field) * 1e6 / 100.0
        else:
            raise ValueError(unit)
        out[p] = count / (DECODE_TOKENS * leg(anchor, p, field)) * 1e-6 * 100.0
    return out


def fit_k(anchor, row, basis_row=None, weights=None, field="decode"):
    """Least squares through the origin of per-prompt %diff on the basis.

    Returns k in microseconds added to every drafting round, the R2 of the fit
    against the flat-percent null (which is R2 = 0 by construction of the
    through-origin form), the per-prompt residual sd in percentage points, and
    the observed and predicted vectors.
    """
    diffs = pct_diff(anchor, row, field)
    names = list(PROMPT_ORDER)
    if weights is None:
        weights = {p: 1.0 for p in names}
    design = design_matrix(anchor, basis_row or row, "drafting_round", field)
    num = sum(weights[p] * design[p] * diffs[p] for p in names)
    den = sum(weights[p] * design[p] * design[p] for p in names)
    k = num / den if den else float("nan")
    pred = {p: k * design[p] for p in names}
    resid = [diffs[p] - pred[p] for p in names]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum(diffs[p] * diffs[p] for p in names)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    resid_sd = math.sqrt(ss_res / len(names))
    out = {
        "k_us_per_drafting_round": k,
        "r2_through_origin": r2,
        "residual_sd_pp": resid_sd,
        "cand_mean8_pct": sum(diffs[p] for p in names) / 8.0,
        "observed_pct": diffs,
        "predicted_pct": pred,
        "residual_pct": {p: diffs[p] - pred[p] for p in names},
        "field": field,
    }
    if anchor.has_prefill() and row.has_prefill():
        pre = pct_diff(anchor, row, "prefill")
        out["prefill_mean8_pct"] = sum(pre[p] for p in names) / 8.0
    return out


def fit_on_basis(anchor, row, unit, field="decode"):
    """The same through-origin fit against a rival physical unit.

    unit is one of `drafting_round`, `round`, `verify_row`, `draft_step`,
    `emitted_token`, `non_drafting_round`, `flat`.
    """
    names = list(PROMPT_ORDER)
    design = design_matrix(anchor, row, unit, field)
    diffs = pct_diff(anchor, row, field)
    num = sum(design[p] * diffs[p] for p in names)
    den = sum(design[p] * design[p] for p in names)
    k = num / den if den else float("nan")
    resid = [diffs[p] - k * design[p] for p in names]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum(diffs[p] * diffs[p] for p in names)
    mean_diff = sum(diffs[p] for p in names) / len(names)
    ss_flat = sum((diffs[p] - mean_diff) ** 2 for p in names)
    return {
        "k": k,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "residual_sd_pp": math.sqrt(ss_res / len(names)),
        "shape_r2_vs_flat": 1.0 - ss_res / ss_flat if ss_flat else float("nan"),
        "predicted_pct": {p: k * design[p] for p in names},
    }


def state_pct(anchor, row, step_us, field="decode"):
    """The 8-prompt mean percentage that `step_us` per drafting round buys.

    This is the ONE definition of the correction. The model is additive in
    seconds, so the correction is the mean over prompts of the per-prompt
    percentage that `step_us` moves, and nothing else. In particular it is not
    `raw * step / k`: that proportional form silently assumes the whole observed
    difference lies along the basis, which is false whenever the fit has a
    residual, and the residual is exactly what is large on the rows that carry
    the state.

    `field` selects the frame. A submission is priced on the whole timed leg,
    so `total` is the frame for any reported score effect; `decode` is the frame
    the state is estimated in.
    """
    design = design_matrix(anchor, row, "drafting_round", field)
    return step_us * sum(design[p] for p in PROMPT_ORDER) / len(PROMPT_ORDER)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def sd(values, ddof=1):
    values = list(values)
    if len(values) <= ddof:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - ddof))
