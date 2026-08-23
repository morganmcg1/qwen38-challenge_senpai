#!/usr/bin/env python3
"""E150 shared harness: one replay environment, one scorer, one arm shape.

`harness=local instrument`. Zero GPU. Every number this module produces is a
replayed offline price against a measured per-width round cost curve, never a
timed leg. Campaign Rule 79 forbids a local timing leg from publishing a
depth-price or schedule-policy contrast, and this module never times anything.

The environment is E145 R7's environment, rebuilt here rather than re-read
from an artifact so that R0, R1 and R2 all price against one cache, one seed
set and one receipt.

Three preconditions this module owns, because getting any of them wrong voids
every downstream number:

  1. `e140_cells.install(curve)` must run BEFORE `e128_price.make_policy`,
     because `make_policy` freezes the ranked price table at construction
     time. `assert_price_ordering` is the copyable assertion E145 R7-5
     published for this.
  2. The E128 leg attachment gate must report zero mismatches, otherwise the
     recorded per-position margin vectors are not aligned with the rounds
     that carry them.
  3. The base arm of every ratio is the shipped tree simulated on the SAME
     installed curve, so a curve change cannot leak into an arm contrast.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
import e140_cells  # noqa: E402
from e128_price import RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e128_replay import EMA_PRIOR, MAX_DEPTH  # noqa: E402
from e134_rung2 import build_legs, median_pct, simulate  # noqa: E402
from e140_cells import install, transfer_cache  # noqa: E402
from e145_r7 import MAX_WIDTH, admissibility  # noqa: E402
from e145_r7_state import curves_and_prices  # noqa: E402

ARTIFACTS = HERE / "e150-artifacts"
DEFAULT_RUNS = HERE.parent / ".mlxfast-private/e128/runs-forced"
DEFAULT_ACCEPT = HERE / "e128-artifacts/rung1-forced.json"
DEFAULT_BOARD = pathlib.Path("/tmp/yukon-board/full.json")
DEFAULT_RECEIPT = "d3c491b5"

EMA_PRIOR_LIST = list(EMA_PRIOR)

# E145 R7-2 and R7-3 cells this campaign already published on the measured
# curve. Every E150 rung reproduces the ones it depends on rather than citing
# them, and these are the values the reproduction must land on.
R7_SHIPPED_PCT = 0.1338226619729229
R7_CLAMP_NONE_PCT = R7_SHIPPED_PCT - 0.5667827116888124
R7_TRUEP_ARGMAX_PCT = -0.380622197388301
R7_CLAIRVOYANT_ARGMAX_PCT = 4.95412243417301
R7_ORACLE_DEPTH_PCT = R7_SHIPPED_PCT + 6.216971808451828
REPRO_TOLERANCE_PP = 0.02


class Env:
    """The replay environment every E150 rung prices against."""

    def __init__(self, cache, seeds, receipt, windows, points, measured,
                 replayed, measured_price, replayed_price, admitted, gate):
        self.cache = cache
        self.seeds = seeds
        self.receipt = receipt
        self.windows = windows
        self.points = points
        self.measured = measured
        self.replayed = replayed
        self.measured_price = measured_price
        self.replayed_price = replayed_price
        self.admitted = admitted
        self.gate = gate
        self._base = {}

    def identity(self) -> dict:
        return {
            "harness": "local",
            "frame": "decode",
            "gpu_used": False,
            "windows": self.windows,
            "seeds": list(self.seeds),
            "receipt": self.receipt["id"],
            "receipt_score": self.receipt["score"],
            "width1_anchor": "measured",
            "admissible_widths": sorted(self.admitted),
            "attachment_gate": self.gate,
        }

    def base(self, curve_name: str, curve, seed: str, prompt: str) -> dict:
        """The shipped tree on `curve`, memoised per curve, seed and prompt."""
        key = (curve_name, seed, prompt)
        if key not in self._base:
            entry = self.cache[(seed, prompt)]
            install(curve)
            self._base[key] = simulate(
                None, entry["factory"](entry["p_target"]), self.windows)
        return self._base[key]


def build_env(windows: int = 200, fit_windows: int = 60, seed: int = 128,
              seeds: int = 6, anchor: str = "measured",
              runs: pathlib.Path = DEFAULT_RUNS,
              accept: pathlib.Path = DEFAULT_ACCEPT,
              board: pathlib.Path = DEFAULT_BOARD,
              receipt_prefix: str = DEFAULT_RECEIPT) -> Env:
    points, measured, replayed, measured_price, replayed_price = (
        curves_and_prices(anchor))
    admitted = set(admissibility(points)["admissible_widths"])
    receipt = load_board_receipt(board, receipt_prefix)
    legs, gate = build_legs(accept, runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    seed_list = [seed + i for i in range(seeds)]
    cache = transfer_cache(legs, windows, fit_windows, seed_list)
    return Env(cache, seed_list, receipt, windows, points, measured, replayed,
               measured_price, replayed_price, admitted, gate)


def assert_price_ordering(curve, price) -> None:
    """Precondition 1, as the one-line assertion E145 R7-5 published.

    Every caller that builds an `e128_price` policy against `curve` must run
    this first. It fails loudly when the frozen ranked price table belongs to
    a different curve than the one being simulated.
    """
    e140_cells.install(curve)
    assert e128_price.ranked_price_table()[0] == price[0], \
        "call e140_cells.install(curve) before e128_price.make_policy"


def score_arm(env: Env, curve_name: str, curve, price, make_walker,
              adjust=None) -> dict:
    """One arm's receipt-weighted median percent against the shipped tree.

    `make_walker(seed, prompt, entry) -> chooser | None`. A `None` chooser
    runs the shipped walk, which is how an `adjust`-only E128 arm is priced.
    """
    values, per_prompt, per_prompt_depth = [], {}, {}
    hist = [0] * (MAX_DEPTH + 2)
    cap_hist = [0] * (MAX_DEPTH + 2)
    rounds = 0
    depths, accepts = [], []
    for seed in env.seeds:
        ratios = {}
        for prompt in RANKED_PROMPTS:
            entry = env.cache[(seed, prompt)]
            base = env.base(curve_name, curve, seed, prompt)
            walker = make_walker(seed, prompt, entry)
            install(curve)
            run = simulate(adjust, entry["factory"](entry["p_target"]),
                           env.windows, price=price, walker=walker)
            ratio = run["us_per_token"] / base["us_per_token"]
            ratios[prompt] = {"ratio": ratio}
            per_prompt.setdefault(prompt, []).append(ratio)
            per_prompt_depth.setdefault(prompt, []).append(run["mean_depth"])
            for index, count in enumerate(run["depth_counts"]):
                hist[index] += count
            for index, count in enumerate(run["cap_counts"]):
                cap_hist[index] += count
            rounds += run["rounds"]
            weight = RANKED_PROMPTS[prompt]["weight"]
            depths.append(weight * run["mean_depth"])
            accepts.append(weight * run["accept_rate"])
        values.append(median_pct(env.receipt, ratios))
    n = len(env.seeds)
    return {
        "median_pct_mean": statistics.fmean(values),
        "median_pct_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_pct_values": values,
        "weighted_mean_depth": sum(depths) / n,
        "weighted_accept_rate": sum(accepts) / n,
        "rounds": rounds,
        "width_histogram": {str(i + 1): hist[i] / rounds
                            for i in range(MAX_WIDTH)},
        "capability_histogram": {str(i): cap_hist[i] / rounds
                                 for i in range(MAX_DEPTH + 1)},
        "frac_rounds_inadmissible": sum(
            hist[w - 1] for w in range(1, MAX_WIDTH + 1)
            if w not in env.admitted) / rounds,
        "per_prompt_ratio": {p: statistics.fmean(v)
                             for p, v in per_prompt.items()},
        "per_prompt_mean_depth": {p: statistics.fmean(v)
                                  for p, v in per_prompt_depth.items()},
    }


def clamp_state(state, margin: float, scales: dict[int, float] | None):
    """E145 R7-2's `clamped`, kept identical so the cells stay comparable."""
    if not scales or math.isnan(margin):
        return state
    out = list(state)
    for depth, scale in scales.items():
        if depth < len(out):
            out[depth] = min(out[depth],
                             1.0 / (1.0 + math.exp(-margin / scale)))
    return out


def slope_of(rows) -> float:
    n = len(rows)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(rows) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(rows))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def write_artifact(name: str, payload: dict) -> pathlib.Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
