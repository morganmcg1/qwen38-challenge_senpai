#!/usr/bin/env python3
"""E143 R1: split the in-vocabulary first divergences into C-b, C-c and C-d.

WHAT DECIDES THE SPLIT. The shipped proposal path is

    coarse   2-bit affine readout of the compact draft head over 98,330 rows
    window   the top `draftRerankCandidateCount` = 32 coarse rows
    rerank   an EXACT affine-4 readout of the target's own lm_head rows,
             restricted to the window, argmax

C-c IS IDENTICALLY ZERO AND THAT IS A PROOF, NOT A MEASUREMENT. If the head's
exact full-compact readout ranks `t*` first and `t*` is in the window, then
`t*` is the argmax of the exact readout over a subset that contains it, so the
rerank returns `t*`. The rerank cannot mis-order a token it would itself rank
first. The only reachable in-vocabulary channel is therefore

    C-b  the 2-bit coarse screen dropped `t*` before the exact rerank saw it

and everything else in vocabulary is

    C-d  the head's hidden state prefers a different continuation.

WHY THIS RUNS WITHOUT THE HEAD'S HIDDEN ROW. `Qwen35.swift:5870` records that
the shipped `draft_lm_head.*` is exactly `quantize(dequantize(exact compact
lm_head), 64, 2)`, verified bit for bit. Arm S below re-derives that and
asserts it. So the screen's failure mode is a property of a 2-bit versus 4-bit
readout of ONE matrix, and it can be measured on any in-distribution hidden
vector of the right scale. The E142 capture holds 3,664 real ones.

THREE ARMS (Rule 113).

  arm P  legacy audit. The device's own top-2 evidence for each captured row,
         used to prove the offline readout reproduces the scored one.
  arm S  the shipped arithmetic. The coarse tensors re-derived from the exact
         compact lm_head and asserted equal to the shipped ones bit for bit,
         then the shipped pipeline replayed at the true hidden rows.
  arm F  the treatment. The same pipeline at a SURROGATE head hidden, so the
         channel split can be read at the head's real operating point.

THE SURROGATE, AND WHAT MAKES IT A TEST RATHER THAN A FIT. The head's hidden is
a noisier estimate of the same next-token direction, so arm F perturbs the
captured hidden by an isotropic Gaussian of relative L2 norm `sigma` and
replays the whole pipeline. `sigma` is fitted on ONE observable - the simulated
miss rate matched to the measured non-C-a miss rate - and then VALIDATED on a
different one, the distribution of the target's rank of the proposal, which the
fit never sees. A surrogate that reproduces the second statistic is evidence.
One that does not is reported as a failed surrogate and C-b stays unresolved.

Usage:
  research/e143_r1.py [--sigmas 0.0,0.1,...] [--out research/e143-r1.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

HIDDEN = 5120
VOCAB = 248_320
GROUP = 64
COMPACT_PREFIX_COUNT = 98_304
COMPACT_CONTROL_START = 248_044
COMPACT_CONTROL_END = 248_070
COMPACT_REAL_COUNT = 98_330
COMPACT_PADDED_COUNT = 98_336
RERANK_CANDIDATES = 32
RECALL_K = (8, 16, 32, 64, 128, 256, 512, 1024)

MISS_TO_SCORE_PCT = 203.0
CARRIER_WEIGHT = {"beagle": 0.478, "essays": 0.522}
RANKED_P = {"beagle": 0.9341, "essays": 0.9647}
SEED_CARRIER = {"beagle_a": "beagle", "beagle_f": "beagle",
                "essays_montaigne": "essays"}

CACHE = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1"
DUMP = CACHE / "e142/verifyrows/hidden"
VERIFY = CACHE / "e142/verifyrows/verify"
HEAD = CACHE / "mtp-head-declared-run/model.safetensors"
TARGET = "weights/model-00003-of-00003.safetensors"


def compact_row_of(token: np.ndarray) -> np.ndarray:
    """Compact row of each token id, or -1 when the token has no compact row."""
    out = np.full(token.shape, -1, dtype=np.int64)
    low = (token >= 0) & (token < COMPACT_PREFIX_COUNT)
    out[low] = token[low]
    ctrl = (token >= COMPACT_CONTROL_START) & (token < COMPACT_CONTROL_END)
    out[ctrl] = COMPACT_PREFIX_COUNT + token[ctrl] - COMPACT_CONTROL_START
    return out


def load_trials() -> dict:
    """The 2,634 ON-TRAJECTORY draft trials, with their captured hidden rows.

    A round's on-trajectory draft rows are rows 0 through the first divergence
    inclusive, or every draft row when nothing diverged. Rows after the first
    divergence are speculative continuations of a wrong prefix, so they are not
    Bernoulli trials and they are excluded here.
    """
    seeds = sorted(p.stem for p in VERIFY.glob("*.json"))
    keep: dict[str, dict[int, int]] = {}
    meta_rows: list[dict] = []
    for seed in seeds:
        payload = json.loads((VERIFY / f"{seed}.json").read_text())
        rounds: dict[int, list[dict]] = defaultdict(list)
        for row in payload["row_ledger"]:
            rounds[row["round"]].append(row)
        per_round: dict[int, int] = {}
        for index in sorted(rounds):
            drafts = sorted((r for r in rounds[index] if r["kind"] == "draft"),
                            key=lambda r: r["draft_index"])
            if not drafts:
                continue
            diverged = next((r for r in drafts if not r["accepted"]), None)
            last = diverged["draft_index"] if diverged else len(drafts) - 1
            per_round[index] = last
            for r in drafts[:last + 1]:
                meta_rows.append({
                    "seed": seed,
                    "carrier": SEED_CARRIER.get(seed, "other"),
                    "round": index,
                    "draft_index": r["draft_index"],
                    "t_star": r["top2_tokens"][0],
                    "proposal": r["token"],
                    "accepted": r["accepted"],
                    "s1": r["top2_logits"][0],
                    "s2": r["top2_logits"][1],
                })
        keep[seed] = per_round

    hidden = np.zeros((len(meta_rows), HIDDEN), dtype=np.float32)
    cursor = 0
    for seed in seeds:
        for meta_path in sorted(DUMP.glob(f"{seed}.pid*.meta.i32")):
            stem = str(meta_path)[: -len(".meta.i32")]
            meta = np.fromfile(meta_path, dtype=np.int32)
            t2i = np.fromfile(stem + ".top2.i32", dtype=np.int32).reshape(-1, 2)
            xs = np.fromfile(stem + ".x.f32", dtype=np.float32).reshape(-1, HIDDEN)
            offset = 0
            for index, width in enumerate(meta.tolist()):
                last = keep[seed].get(index)
                if last is not None:
                    for k in range(last + 1):
                        row = meta_rows[cursor]
                        if row["round"] != index or row["draft_index"] != k:
                            raise SystemExit(
                                f"e143-r1: {seed}: ledger and dump disagree at "
                                f"round {index} draft {k}")
                        if int(t2i[offset + k, 0]) != row["t_star"]:
                            raise SystemExit(
                                f"e143-r1: {seed} round {index} draft {k}: dump "
                                f"argmax {int(t2i[offset + k, 0])} != ledger "
                                f"{row['t_star']}")
                        hidden[cursor] = xs[offset + k]
                        cursor += 1
                offset += width
    if cursor != len(meta_rows):
        raise SystemExit(f"e143-r1: filled {cursor} of {len(meta_rows)} hidden rows")
    return {"meta": meta_rows, "hidden": hidden}


def load_heads() -> dict:
    """The exact compact affine-4 rows, and the shipped 2-bit coarse copy.

    Arm S: the coarse tensors are re-derived here from the exact rows and
    asserted equal to the shipped ones, which is what makes a measurement on
    the re-derived arithmetic a measurement of the SHIPPED screen.
    """
    blob = mx.load(TARGET)
    take = mx.array(np.concatenate([
        np.arange(COMPACT_PREFIX_COUNT),
        np.arange(COMPACT_CONTROL_START, COMPACT_CONTROL_END)]).astype(np.int32))
    exact_w = blob["language_model.lm_head.weight"][take]
    exact_s = blob["language_model.lm_head.scales"][take]
    exact_z = blob["language_model.lm_head.biases"][take]
    mx.eval(exact_w, exact_s, exact_z)
    del blob

    head = mx.load(str(HEAD))
    ship_w = head["draft_lm_head.weight"][:COMPACT_REAL_COUNT]
    ship_s = head["draft_lm_head.scales"][:COMPACT_REAL_COUNT]
    ship_z = head["draft_lm_head.biases"][:COMPACT_REAL_COUNT]
    mx.eval(ship_w, ship_s, ship_z)
    del head

    derived_w = mx.zeros(ship_w.shape, dtype=ship_w.dtype)
    parts_w, parts_s, parts_z = [], [], []
    step = 16_384
    for start in range(0, COMPACT_REAL_COUNT, step):
        stop = min(start + step, COMPACT_REAL_COUNT)
        deq = mx.dequantize(exact_w[start:stop], exact_s[start:stop],
                            exact_z[start:stop], group_size=GROUP, bits=4)
        qw, qs, qz = mx.quantize(deq, group_size=GROUP, bits=2)
        mx.eval(qw, qs, qz)
        parts_w.append(qw)
        parts_s.append(qs)
        parts_z.append(qz)
        del deq
    derived_w = mx.concatenate(parts_w, axis=0)
    derived_s = mx.concatenate(parts_s, axis=0)
    derived_z = mx.concatenate(parts_z, axis=0)
    mx.eval(derived_w, derived_s, derived_z)

    provenance = {
        "weight_bit_identical": bool(mx.all(derived_w == ship_w).item()),
        "scales_bit_identical": bool(mx.all(derived_s == ship_s).item()),
        "biases_bit_identical": bool(mx.all(derived_z == ship_z).item()),
        "weight_mismatch_words": int(mx.sum(derived_w != ship_w).item()),
        "claim": ("Qwen35.swift:5870 - draft_lm_head is "
                  "quantize(dequantize(exact compact lm_head), 64, 2)"),
    }
    return {"exact": (exact_w, exact_s, exact_z),
            "coarse": (ship_w, ship_s, ship_z),
            "provenance": provenance}


def pipeline(H: mx.array, heads: dict, chunk: int) -> dict:
    """Replay the shipped proposal pipeline at a batch of hidden rows.

    Returns, per row: the exact compact argmax, the coarse rank of that argmax,
    the window's exact winner, and the coarse rank of a supplied reference row.
    """
    exact_w, exact_s, exact_z = heads["exact"]
    coarse_w, coarse_s, coarse_z = heads["coarse"]
    n = H.shape[0]
    exact_argmax = np.zeros(n, dtype=np.int64)
    coarse_rank_of_exact_argmax = np.zeros(n, dtype=np.int64)
    window_winner = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        x = H[start:start + chunk]
        rows = x.shape[0]
        coarse = mx.quantized_matmul(x, coarse_w, scales=coarse_s, biases=coarse_z,
                                     transpose=True, group_size=GROUP, bits=2)
        exact = mx.quantized_matmul(x, exact_w, scales=exact_s, biases=exact_z,
                                    transpose=True, group_size=GROUP, bits=4)
        arg = mx.argmax(exact, axis=1)
        idx = mx.arange(rows)
        coarse_at_arg = coarse[idx, arg]
        rank = (coarse > coarse_at_arg[:, None]).sum(axis=1) + 1
        kth = COMPACT_REAL_COUNT - RERANK_CANDIDATES
        window = mx.argpartition(coarse, kth=kth, axis=1)[:, kth:]
        win_scores = mx.take_along_axis(exact, window, axis=1)
        winner = mx.take_along_axis(window, mx.argmax(win_scores, axis=1)[:, None],
                                    axis=1).reshape(rows)
        mx.eval(arg, rank, winner)
        exact_argmax[start:start + rows] = np.array(arg)
        coarse_rank_of_exact_argmax[start:start + rows] = np.array(rank)
        window_winner[start:start + rows] = np.array(winner)
        del coarse, exact, window, win_scores
    return {"exact_argmax": exact_argmax,
            "coarse_rank_of_exact_argmax": coarse_rank_of_exact_argmax,
            "window_winner": window_winner}


def classify(result: dict, t_row: np.ndarray) -> dict:
    """Channel of every trial, given the pipeline's output and the truth row."""
    argmax_right = result["exact_argmax"] == t_row
    in_window = result["coarse_rank_of_exact_argmax"] <= RERANK_CANDIDATES
    proposal = result["window_winner"]
    miss = proposal != t_row
    channel_b = miss & argmax_right & ~in_window
    channel_c = miss & argmax_right & in_window
    channel_d = miss & ~argmax_right
    return {"miss": miss, "channel_b": channel_b, "channel_c": channel_c,
            "channel_d": channel_d, "proposal": proposal,
            "argmax_right": argmax_right, "in_window": in_window}


def wilson(successes: int, trials: int, z: float = 1.0) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials
                         + z * z / (4 * trials * trials)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def ranked_pct(rate: float) -> float:
    return MISS_TO_SCORE_PCT * rate * (CARRIER_WEIGHT["beagle"] + CARRIER_WEIGHT["essays"])


def target_rank_of(H: mx.array, ids: np.ndarray, chunk: int, target_chunk: int) -> np.ndarray:
    """The TARGET's full-vocabulary rank of a proposed token, per row.

    The validation statistic. Computed against the full 248,320-row lm_head,
    exactly as the measured version in stage B was, so the two are comparable.
    """
    blob = mx.load(TARGET)
    wq = blob["language_model.lm_head.weight"]
    scales = blob["language_model.lm_head.scales"]
    biases = blob["language_model.lm_head.biases"]
    n = H.shape[0]
    unique, inverse = np.unique(ids, return_inverse=True)
    take = mx.array(unique.astype(np.int32))
    deq = mx.dequantize(wq[take], scales[take], biases[take],
                        group_size=GROUP, bits=4).astype(mx.float32)
    logit = np.array(H @ deq.T)[np.arange(n), inverse]
    del deq
    ld = mx.array(logit.astype(np.float32))
    greater = mx.zeros((n,), dtype=mx.int32)
    for start in range(0, VOCAB, target_chunk):
        deq = mx.dequantize(wq[start:start + target_chunk],
                            scales[start:start + target_chunk],
                            biases[start:start + target_chunk],
                            group_size=GROUP, bits=4).astype(mx.float32)
        greater = greater + (H @ deq.T > ld[:, None]).sum(axis=1).astype(mx.int32)
        mx.eval(greater)
        del deq
    return np.array(greater).astype(np.int64) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmas", default="0.02,0.05,0.08,0.11,0.14,0.18,0.24")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--target-chunk", type=int, default=15520)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test on the first N trials only")
    parser.add_argument("--out", default="research/e143-r1.json")
    args = parser.parse_args()

    started = time.time()
    trials = load_trials()
    meta = trials["meta"]
    hidden = trials["hidden"]
    if args.limit:
        meta = meta[:args.limit]
        hidden = hidden[:args.limit]
    n = len(meta)
    t_star = np.array([m["t_star"] for m in meta], dtype=np.int64)
    accepted = np.array([m["accepted"] for m in meta])
    carrier = np.array([m["carrier"] for m in meta])
    t_row = compact_row_of(t_star)
    unproposable = t_row < 0
    print(f"e143-r1: {n} on-trajectory trials, {int(accepted.sum())} accepted, "
          f"{int((~accepted).sum())} divergences, "
          f"{int(unproposable.sum())} unproposable", flush=True)

    heads = load_heads()
    print("e143-r1: provenance " + json.dumps(heads["provenance"]), flush=True)
    if not all(heads["provenance"][k] for k in
               ("weight_bit_identical", "scales_bit_identical", "biases_bit_identical")):
        raise SystemExit("e143-r1: the shipped coarse head is not the re-derived one; "
                         "arm S does not reproduce the shipped arithmetic")

    H = mx.array(hidden)

    # ---- arm S: the shipped pipeline at the TRUE hidden rows ---------------
    base = pipeline(H, heads, args.chunk)
    print(f"e143-r1: arm S done {time.time() - started:.1f}s", flush=True)

    # arm P: the offline exact readout has to reproduce the device's argmax on
    # the rows whose device argmax has a compact row at all.
    proposable = ~unproposable
    argmax_matches_device = float(
        np.mean(base["exact_argmax"][proposable] == t_row[proposable]))

    recall = {}
    for k in RECALL_K:
        hits = int(np.sum(base["coarse_rank_of_exact_argmax"] <= k))
        recall[f"recall_at_{k}"] = hits / n
        recall[f"recall_at_{k}_ci68"] = wilson(hits, n)
    coarse_rank = base["coarse_rank_of_exact_argmax"]
    screen_loss_32 = float(np.mean(coarse_rank > RERANK_CANDIDATES))
    arm_s = {
        "rows": n,
        "arm_p_offline_argmax_matches_device": argmax_matches_device,
        "coarse_rank_of_exact_argmax_median": float(np.median(coarse_rank)),
        "coarse_rank_of_exact_argmax_p99": float(np.percentile(coarse_rank, 99)),
        "screen_loss_at_32": screen_loss_32,
        "screen_loss_at_32_ci68": wilson(int(np.sum(coarse_rank > RERANK_CANDIDATES)), n),
        **recall,
    }

    # ---- arm F: the surrogate head hidden ----------------------------------
    measured_miss = float(np.mean(~accepted & ~unproposable))
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal((n, HIDDEN)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    h_norm = np.linalg.norm(hidden, axis=1, keepdims=True)
    Noise = mx.array(noise * h_norm)

    sweep = []
    for sigma in [float(s) for s in args.sigmas.split(",") if s]:
        out = pipeline(H + sigma * Noise, heads, args.chunk)
        cls = classify(out, t_row)
        sim_miss = float(np.mean(cls["miss"] & ~unproposable))
        row = {
            "sigma": sigma,
            "simulated_miss_rate": sim_miss,
            "channel_b_rate": float(np.mean(cls["channel_b"])),
            "channel_c_rate": float(np.mean(cls["channel_c"])),
            "channel_d_rate": float(np.mean(cls["channel_d"] & ~unproposable)),
            "channel_b_events": int(cls["channel_b"].sum()),
            "channel_c_events": int(cls["channel_c"].sum()),
        }
        sweep.append((row, cls))
        print(f"e143-r1: sigma={sigma:.3f} miss={sim_miss:.4f} "
              f"C-b={row['channel_b_rate']:.5f} C-c={row['channel_c_rate']:.5f} "
              f"{time.time() - started:.1f}s", flush=True)

    # Calibrate on the miss rate only.
    best_row, best_cls = min(
        sweep, key=lambda item: abs(item[0]["simulated_miss_rate"] - measured_miss))

    # Validate on a statistic the calibration never saw.
    sim_miss_rows = np.where(best_cls["miss"] & ~unproposable)[0]
    sim_rank = target_rank_of(
        mx.array(hidden[sim_miss_rows]), best_cls["proposal"][sim_miss_rows].astype(np.int64),
        args.chunk, args.target_chunk)
    measured = json.loads(Path("research/e143-r0.json").read_text())
    measured_ranks = np.array(
        [r["target_rank_of_d"] for r in measured["records"] if "target_rank_of_d" in r])
    validation = {
        "statistic": "target rank of the proposal at a miss",
        "measured_median": float(np.median(measured_ranks)),
        "measured_le_32_share": float(np.mean(measured_ranks <= 32)),
        "measured_is_2_share": float(np.mean(measured_ranks == 2)),
        "measured_n": int(measured_ranks.size),
        "surrogate_median": float(np.median(sim_rank)),
        "surrogate_le_32_share": float(np.mean(sim_rank <= 32)),
        "surrogate_is_2_share": float(np.mean(sim_rank == 2)),
        "surrogate_n": int(sim_rank.size),
    }
    validation["median_ratio"] = (validation["surrogate_median"]
                                  / validation["measured_median"])
    validation["le_32_abs_delta"] = abs(
        validation["surrogate_le_32_share"] - validation["measured_le_32_share"])
    # A surrogate is usable when it lands the median rank within a factor of
    # two and the top-32 containment within 10 points. Both are coarse, but
    # they are the difference between "a noisier version of the same direction"
    # and "a different direction".
    validation["surrogate_usable"] = bool(
        0.5 <= validation["median_ratio"] <= 2.0
        and validation["le_32_abs_delta"] <= 0.10)

    cb_rate = best_row["channel_b_rate"]
    cb_ci = wilson(best_row["channel_b_events"], n)
    channel_b = {
        "rate": cb_rate,
        "rate_ci68": cb_ci,
        "events": best_row["channel_b_events"],
        "trials": n,
        "ranked_pct": ranked_pct(cb_rate),
        "ranked_pct_ci68": [ranked_pct(cb_ci[0]), ranked_pct(cb_ci[1])],
        "label": "derived",
    }
    channel_c = {
        "rate": best_row["channel_c_rate"],
        "events": best_row["channel_c_events"],
        "ranked_pct": ranked_pct(best_row["channel_c_rate"]),
        "note": ("zero by construction: the rerank is exact over the window, so "
                 "a token the head would rank first cannot be mis-ordered"),
    }
    channel_d = {
        "rate": best_row["channel_d_rate"],
        "ranked_pct": ranked_pct(best_row["channel_d_rate"]),
        "share_of_in_vocabulary_misses": (
            best_row["channel_d_rate"]
            / max(best_row["simulated_miss_rate"], 1e-12)),
        "label": "derived",
    }

    state = {
        "harness": "local",
        "measured_non_channel_a_miss_rate": measured_miss,
        "arm_s_true_hidden": arm_s,
        "arm_f_sweep": [row for row, _ in sweep],
        "arm_f_calibrated_sigma": best_row["sigma"],
        "arm_f_calibration_residual": abs(
            best_row["simulated_miss_rate"] - measured_miss),
        "arm_f_validation": validation,
        "channel_b": channel_b,
        "channel_c": channel_c,
        "channel_d": channel_d,
        "head_provenance": heads["provenance"],
        "elapsed_seconds": time.time() - started,
    }
    Path(args.out).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))
    print(f"e143-r1: wrote {args.out}")


if __name__ == "__main__":
    main()
