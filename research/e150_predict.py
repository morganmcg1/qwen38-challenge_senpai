#!/usr/bin/env python3
"""E150: the pre-draft observer, the hazard model, and the deployable walker.

`harness=local instrument`. Zero GPU.

THE INTEGRITY BOUNDARY OF THIS FILE, stated once and enforced structurally.

`Observer` and `features` may read only quantities the scored session already
holds at the moment it chooses the round's depth, BEFORE any draft for that
round exists:

  * `pendingTop2`, the target's top-two margin on the pending primary token,
    produced by the PREVIOUS round's verify. `e134_rung1.parse_trace` gates
    the identity `margin[r] == rows[r-1][-1]` on every attached round, so this
    is the previous verify's last row and nothing else.
  * the previous round's per-position margin rows, which the previous verify
    already computed and the session already logs.
  * the previous rounds' accepted counts, the running accept base rate, the
    accept streak, and `positionAcceptEMA`, all of which the shipped session
    already maintains.
  * `offeredDepth`, which the parent hands the session each round.

`features` never receives `ctx["capability"]`. The label is read only by the
collection harness in `e150_r1.py`, never by anything that runs inside a
priced round. Nothing here reads prompt identity, prompt length, a prompt
hash, any per-request state that survives a request, or the benchmark phase.

The fitted coefficients are an input-independent table. They are fitted on the
public forced-depth fixture traces, which is the same class of object as the
shipped `EMA_PRIOR` constant and the shipped depth price table.
"""
from __future__ import annotations

import math
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_replay import (  # noqa: E402
    EMA_PRIOR,
    MAX_DEPTH,
    PRICE_CUMULATIVE,
    PRICE_MARGINAL,
    SEGMENTED_VERIFY_DEPTH_CAP,
)
from e140_lookahead import walk_argmax  # noqa: E402
from e134_rung2 import walk  # noqa: E402
from e150_lib import clamp_state, slope_of  # noqa: E402

EMA_PRIOR_LIST = list(EMA_PRIOR)

# Shared pre-draft columns, in a fixed order. `ema_i` is appended per position
# by `design_row`, so the per-position model reads the shipped state entry for
# exactly the boundary it is predicting.
SHARED_COLUMNS = (
    "margin", "sig2", "sig3",
    "pm_mean", "pm_min", "pm_slope", "pm_first", "prev_len",
    "prev_acc", "acc_mean3", "streak", "base_rate", "offer",
)

# The information ladder of E150 R2. Every rung is a column subset of the same
# fitted model family, so a rung difference is an information difference and
# not a model-family difference.
LADDER = {
    "L0c": (),
    "L1e": (),
    "L2": ("margin", "sig2", "sig3"),
    "L3": ("margin", "sig2", "sig3",
           "pm_mean", "pm_min", "pm_slope", "pm_first", "prev_len"),
    "L4": SHARED_COLUMNS,
}
# `L0c` is the constant control and reads no state at all, not even the EMA.
USES_EMA = {"L0c": False, "L1e": True, "L2": True, "L3": True, "L4": True}


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Observer:
    """Per-window pre-draft state, reconstructed from what the walk is handed.

    `e134_rung2.simulate` does not pass the previous round's accepted count to
    the chooser, but it does pass `base_rate = accepted_total / drafted_total`
    computed over the rounds already closed. The chooser knows every depth it
    chose, so `accepted_total` is recoverable exactly and the previous
    accepted count is its first difference. `e150_r1.py` gates that
    reconstruction against the simulator's own accepted counts and refuses to
    fit if a single round disagrees.

    A window restarts when the walk is handed `EMA_PRIOR` again. At that point
    `accepted_total` and `drafted_total` are both zero, so the reset is exact
    and not a heuristic.
    """

    __slots__ = ("drafted", "acc_total", "prev_acc", "recent", "streak",
                 "last_depth", "rounds")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.drafted = 0
        self.acc_total = 0
        self.prev_acc = 0
        self.recent = []
        self.streak = 0
        self.last_depth = 0
        self.rounds = 0

    def observe(self, ema, base_rate: float) -> None:
        if ema == EMA_PRIOR_LIST:
            self.reset()
            return
        acc_total = (int(round(base_rate * self.drafted))
                     if self.drafted else 0)
        acc = acc_total - self.acc_total
        self.acc_total = acc_total
        if self.last_depth > 0 and acc == self.last_depth:
            self.streak += 1
        else:
            self.streak = 0
        self.prev_acc = acc
        self.recent.append(acc)
        if len(self.recent) > 3:
            self.recent.pop(0)
        self.rounds += 1

    def chose(self, depth: int) -> None:
        self.last_depth = depth
        self.drafted += depth


def features(margin: float, offer: int, prev_rows, obs: Observer,
             base_rate: float) -> dict:
    """The pre-draft observable vector. It never sees this round's drafts."""
    if prev_rows:
        pm_mean = sum(prev_rows) / len(prev_rows)
        pm_min = min(prev_rows)
        pm_first = prev_rows[0]
        pm_slope = slope_of(prev_rows)
    else:
        pm_mean = pm_min = pm_first = pm_slope = 0.0
    safe = margin if not math.isnan(margin) else 0.0
    return {
        "margin": safe,
        "sig2": sigmoid(safe / 2.0),
        "sig3": sigmoid(safe / 3.0),
        "pm_mean": pm_mean,
        "pm_min": pm_min,
        "pm_slope": pm_slope,
        "pm_first": pm_first,
        "prev_len": float(len(prev_rows)),
        "prev_acc": float(obs.prev_acc),
        "acc_mean3": (sum(obs.recent) / len(obs.recent)
                      if obs.recent else 0.0),
        "streak": float(obs.streak),
        "base_rate": base_rate,
        "offer": float(offer),
    }


# --------------------------------------------------------------- the model

def irls(x: np.ndarray, y: np.ndarray, ridge: float = 1.0,
         steps: int = 25) -> np.ndarray:
    """Ridge-penalised logistic regression by Newton iteration.

    Column 0 is the intercept and carries no penalty. The penalty is on
    standardised columns, so it is scale free.
    """
    n, k = x.shape
    beta = np.zeros(k)
    beta[0] = math.log(max(1e-6, min(1 - 1e-6, y.mean()))
                       / max(1e-6, 1.0 - y.mean()))
    penalty = np.eye(k) * ridge
    penalty[0, 0] = 0.0
    for _ in range(steps):
        eta = np.clip(x @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        grad = x.T @ (y - mu) - penalty @ beta
        hess = (x.T * w) @ x + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


class Hazard:
    """One conditional acceptance model per draft position.

    Position `i` predicts `P(K > i | K >= i, x)`, which is exactly the object
    the shipped walk multiplies into `reach`. `positionAcceptEMA` is a leaky
    marginal estimate of the same quantity, so this model is a drop-in
    replacement for the state vector and not a change to the decision rule.
    """

    def __init__(self, rung: str, columns, use_ema: bool, mean, sd, betas,
                 temperature: float = 1.0):
        self.rung = rung
        self.columns = tuple(columns)
        self.use_ema = use_ema
        self.temperature = temperature
        self.mean = np.asarray(mean, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.betas = [np.asarray(b, dtype=float) for b in betas]
        nbase = len(self.columns)
        self._alpha = np.array([b[0] for b in self.betas])
        self._base = np.array([b[1:1 + nbase] for b in self.betas])
        self._ema = (np.array([b[1 + nbase] for b in self.betas])
                     if use_ema else None)
        self._base_mean = self.mean[:nbase]
        self._base_sd = self.sd[:nbase]
        self._ema_mean = float(self.mean[nbase]) if use_ema else 0.0
        self._ema_sd = float(self.sd[nbase]) if use_ema else 1.0

    def width(self) -> int:
        return len(self.columns) + (1 if self.use_ema else 0)

    def state(self, feat: dict, ema) -> list[float]:
        """`P(K > i | K >= i, x)` for every draft position, in one pass.

        The base columns do not vary with the position and the shipped EMA
        entry does, so the two halves are evaluated separately and added.
        """
        eta = self._alpha
        if self.columns:
            row = np.fromiter((feat[name] for name in self.columns),
                              dtype=float, count=len(self.columns))
            eta = eta + self._base @ ((row - self._base_mean)
                                      / self._base_sd)
        if self._ema is not None:
            eta = eta + self._ema * ((np.asarray(ema[:MAX_DEPTH])
                                      - self._ema_mean) / self._ema_sd)
        if self.temperature != 1.0:
            eta = eta * self.temperature
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        return np.clip(p, 1e-4, 1.0 - 1e-4).tolist()

    def at_temperature(self, temperature: float) -> "Hazard":
        """A sharpened copy. `T > 1` trades calibration for decisiveness.

        The walk consumes the state vector inside a product and then inside a
        ratio, so a calibrated probability is not automatically the state that
        minimises seconds per token. `T` is fitted on training prompts only.
        """
        return Hazard(self.rung, self.columns, self.use_ema, self.mean,
                      self.sd, self.betas, temperature)

    def to_json(self) -> dict:
        return {"rung": self.rung, "columns": list(self.columns),
                "use_ema": self.use_ema, "mean": self.mean.tolist(),
                "sd": self.sd.tolist(), "temperature": self.temperature,
                "betas": [b.tolist() for b in self.betas]}


def design(rows: dict, columns, use_ema: bool, position: int) -> np.ndarray:
    """Raw design columns for one position, before standardisation."""
    parts = [rows[name] for name in columns]
    if use_ema:
        parts.append(rows["ema"][:, position])
    if not parts:
        return np.zeros((len(rows["capability"]), 0))
    return np.column_stack(parts)


def fit_hazard(rows: dict, rung: str, ridge: float = 1.0) -> Hazard:
    """Fit every position on one training pool."""
    columns = LADDER[rung]
    use_ema = USES_EMA[rung]
    cap = rows["capability"]
    sample = design(rows, columns, use_ema, 0)
    if sample.shape[1]:
        mean = sample.mean(axis=0)
        sd = sample.std(axis=0)
        sd[sd < 1e-9] = 1.0
    else:
        mean = np.zeros(0)
        sd = np.ones(0)
    betas = []
    for i in range(MAX_DEPTH):
        mask = cap >= i
        y = (cap[mask] > i).astype(float)
        raw = design(rows, columns, use_ema, i)[mask]
        if raw.shape[1]:
            # `ema_i` is the last column and moves with `i`, so it needs the
            # position-0 standardisation to stay comparable across positions.
            z = (raw - mean) / sd
            x = np.column_stack([np.ones(len(z)), z])
        else:
            x = np.ones((len(y), 1))
        if len(y) < 50 or y.max() == y.min():
            beta = np.zeros(x.shape[1])
            rate = float(y.mean()) if len(y) else 0.5
            beta[0] = math.log(max(1e-6, min(1 - 1e-6, rate))
                               / max(1e-6, 1.0 - rate))
        else:
            beta = irls(x, y, ridge=ridge)
        betas.append(beta)
    return Hazard(rung, columns, use_ema, mean, sd, betas)


# ------------------------------------------------------------- the walkers


def walk_ratio(state, margin: float, offer: int, price, lam: float,
               cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP) -> int:
    """Dinkelbach step for the ratio of sums the score actually measures.

    `walk_argmax` maximises the PER-ROUND ratio `(1 + E[A_d]) / C_d`, and so
    does `oracle_depth`. The published score is a ratio of SUMS, and the
    greedy step that minimises `sum C / sum T` is to maximise
    `lam * (1 + E[A_d]) - C_d` at the current running ratio `lam = sum C /
    sum T`. The two agree only when every round has the same ratio.

    `lam` is the session's own running cost per emitted token in the same
    normalised units as `cumulative`. It carries no information about the
    round being chosen, so this is a decision-rule change and not an
    information change.
    """
    _, cumulative = price or (PRICE_MARGINAL, PRICE_CUMULATIVE)
    cap = min(min(offer, MAX_DEPTH), cap_limit)
    if cap <= 0:
        return 0
    best, best_value = 0, lam * 1.0 - cumulative[0]
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = state[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        reach *= p
        expected += reach
        depth += 1
        value = lam * (1.0 + expected) - cumulative[depth]
        if value > best_value:
            best_value, best = value, depth
    return best


class RatioTracker:
    """The session's own running normalised cost per emitted token."""

    __slots__ = ("cost", "tokens")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.cost = 0.0
        self.tokens = 0.0

    def lam(self):
        """`None` until this session has closed a round.

        There is no defensible constant to seed the running ratio with: any
        constant is a tuned hyperparameter fitted on the very traces being
        priced. The first round of every session therefore falls back to the
        per-round rule, which needs no session history.
        """
        return self.cost / self.tokens if self.tokens > 0 else None

    def close(self, depth: int, accepted: int, cumulative) -> None:
        self.cost += cumulative[depth]
        self.tokens += 1.0 + accepted


def apply_rule(rule: str, state, margin: float, offer: int, price, lam=None,
               cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP):
    if rule == "argmax":
        return min(walk_argmax(state, margin, offer, price=price), cap_limit)
    if rule == "ratio":
        if lam is None:
            return min(walk_argmax(state, margin, offer, price=price),
                       cap_limit)
        return walk_ratio(state, margin, offer, price, lam, cap_limit)
    return min(walk(state, margin, offer, None, None, None, price), cap_limit)


def predictor_walker(model: Hazard, rule: str, clamp, record=None,
                     fixed_lam=None):
    """The deployable chooser. One `Observer`, one state vector, one rule.

    `fixed_lam` replaces the session's running ratio with a constant. It is
    the control that separates "the ratio rule adapts usefully" from "the
    per-round rule is simply biased toward the wrong depth on this curve".
    """
    obs = Observer()
    tracker = RatioTracker()

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        ema_list = list(ema)
        if ema_list == EMA_PRIOR_LIST:
            tracker.reset()
        elif rule == "ratio":
            cumulative = (price or (PRICE_MARGINAL, PRICE_CUMULATIVE))[1]
            tracker.close(obs.last_depth, obs.prev_acc, cumulative)
        obs.observe(ema_list, ctx["base_rate"])
        feat = features(margin, offer, ctx["prev_rows"], obs,
                        ctx["base_rate"])
        state = model.state(feat, ema_list)
        if record is not None:
            record(state, ctx["capability"])
        lam = tracker.lam() if fixed_lam is None else fixed_lam
        if clamp is None:
            depth = apply_rule(rule, state, margin, offer, price, lam)
        else:
            depth = apply_rule(rule, clamp_state(state, margin, clamp),
                               float("nan"), offer, price, lam)
        obs.chose(depth)
        return depth

    return chooser


def surrogate_seconds_per_token(model: Hazard, rows: dict, rule: str, clamp,
                                price, stride: int = 7) -> float:
    """Offline round-level cost per token, on training prompts only.

    This is the objective a hyperparameter is chosen against. It holds the
    round trajectory fixed, so it ignores the feedback from the chosen depth
    into the next round's state; that makes it a selector, never a price.
    """
    _, cumulative = price
    cap_values = rows["capability"]
    ema_rows = rows["ema"]
    total_cost = total_tokens = 0.0
    lam = None
    for _ in range(2):
        total_cost = total_tokens = 0.0
        for j in range(0, len(cap_values), stride):
            feat = {name: rows[name][j] for name in SHARED_COLUMNS}
            state = model.state(feat, ema_rows[j])
            margin = feat["margin"]
            offer = int(feat["offer"])
            if clamp is None:
                depth = apply_rule(rule, state, margin, offer, price, lam)
            else:
                depth = apply_rule(rule, clamp_state(state, margin, clamp),
                                   float("nan"), offer, price, lam)
            total_cost += cumulative[depth]
            total_tokens += 1.0 + min(int(cap_values[j]), depth)
        lam = total_cost / total_tokens
    return total_cost / total_tokens


def select_temperature(model: Hazard, rows: dict, rule: str, clamp, price,
                       grid, stride: int = 7) -> tuple[float, list]:
    """Pick `T` by the offline surrogate on the training prompts only."""
    scored = [(t, surrogate_seconds_per_token(model.at_temperature(t), rows,
                                              rule, clamp, price, stride))
              for t in grid]
    best = min(scored, key=lambda row: row[1])
    return best[0], [{"temperature": t, "surrogate_cost_per_token": c}
                     for t, c in scored]


def fixed_state_walker(state_of, rule: str, clamp, record=None,
                       fixed_lam=None,
                       cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP):
    """A walker whose state vector comes from a caller-supplied function.

    `state_of(ema, margin, offer, ctx) -> list[float]`. The controls use it:
    the truth control returns the realised hit vector, the marginal control
    returns the prompt's true per-position conditional vector, and the noise
    ladders return a perturbed hit vector.

    Every arm built on this walker already reads `ctx["capability"]`, so it is
    a non-implementable ceiling. That is why the running ratio may be closed
    from the realised capability in the same round it is chosen, instead of
    being deferred a round as the deployable `predictor_walker` must do.
    """
    tracker = RatioTracker()

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        ema_list = list(ema)
        if ema_list == EMA_PRIOR_LIST:
            tracker.reset()
        state = state_of(ema_list, margin, offer, ctx)
        if record is not None:
            record(state, ctx["capability"])
        lam = tracker.lam() if fixed_lam is None else fixed_lam
        if clamp is None:
            depth = apply_rule(rule, state, margin, offer, price, lam,
                               cap_limit)
        else:
            depth = apply_rule(rule, clamp_state(state, margin, clamp),
                               float("nan"), offer, price, lam, cap_limit)
        if rule == "ratio":
            cumulative = (price or (PRICE_MARGINAL, PRICE_CUMULATIVE))[1]
            tracker.close(depth, min(int(ctx["capability"]), depth), cumulative)
        return depth
    return chooser


def hit_vector(capability: int) -> list[float]:
    return [1.0 if i < capability else 0.0 for i in range(MAX_DEPTH)]
