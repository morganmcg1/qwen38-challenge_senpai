#!/usr/bin/env python3
"""E142 rung 0: certified survival fractions for a screened verify readout.

The scored verify readout streams the whole affine-4 group-64 `lm_head`
(248,320 x 5,120) once per round and reduces it to two ids and two values per
row. This script asks, offline and with no timed GPU work, how much of that
vocabulary a PROVABLE screen can skip.

For a screen that supplies an upper bound `U_v >= logit_v` for every row, and a
threshold `tau` that is a proven lower bound on the true second-best exact
score `s2`, the shortlist `{v : U_v >= tau}` provably contains the true top two
and the certificate `max over excluded v of U_v < tau <= s2` holds by
construction. The certified survival fraction is the fraction of vocabulary
bytes that a screen cannot exclude at a valid threshold.

Three screens, as assigned:

  S-A  block max-norm.  n_b = max over the 8 rows of block b of ||w_v||_2,
       U_v = n_b * ||h||_2.  Zero build cost, loosest, the null screen.
  S-B  coarse affine-2 group-64 copy of lm_head, derived in process from the
       fixed target weights.  U_v = coarse_v + sum_g (s2_{v,g}/2) * L1_g(h).
  S-C  centroid plus radius branch and bound over leaves of 8 rows,
       U_L = c_L . h + r_L * ||h||_2.  Reported as an optimistic CEILING: the
       leaf radius comes from true 7-nearest-neighbour distances, which no
       clustering can beat, and the centroid term is replaced by the member's
       own exact logit.  A ceiling that cannot prune proves that a built index
       cannot prune either.

Three thresholds, all valid certificates, reported side by side so the value of
the draft-token seed is visible:

  oracle   tau = s2 as the shipped dense readout measured it.  The tightest
           threshold that exists; nothing can prune more than this.
  seed     tau = the smaller of two EXACT scores: the draft token this verify
           row checks, and the screen's own best-scoring row.  Two exact rows
           cost 5,760 bytes and their minimum is a proven lower bound on s2.
  noseed   tau = the second largest LOWER bound the screen itself supplies.
           No exact row is read before the screen runs.

Usage:
  research/e142_r0.py [--seeds A,B] [--rows-per-seed N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HIDDEN = 5120
VOCAB = 248320
GROUP = 64
GROUPS = HIDDEN // GROUP  # 80
# One exact affine-4 group-64 vocabulary row: 640 packed uint32 words plus a
# bf16 scale and a bf16 bias for each of its 80 groups.
EXACT_ROW_BYTES = 640 * 4 + GROUPS * 2 * 2  # 2880
DENSE_ROUND_BYTES = VOCAB * EXACT_ROW_BYTES  # 715,161,600
# A coarse affine-2 group-64 row: 5,120 two-bit codes plus a bf16 scale and
# bias per group.
COARSE2_ROW_BYTES = HIDDEN * 2 // 8 + GROUPS * 2 * 2  # 1600
ROWS_PER_LEAF = 8
LEAVES = VOCAB // ROWS_PER_LEAF  # 31040
NEG_INF = -float("inf")

HEAD_FILE = "weights/model-00003-of-00003.safetensors"
HEAD_PREFIX = "language_model.lm_head"


def load_capture(dump_dir: Path, seeds: list[str], rows_per_seed: int):
    """Rounds recorded by the E142 verify instrument, grouped by seed."""
    rounds = []
    for seed in seeds:
        shards = sorted(dump_dir.glob(f"{seed}.pid*.meta.i32"))
        if not shards:
            raise SystemExit(f"e142-r0: no dump shard for seed {seed}")
        taken = 0
        for meta_path in shards:
            stem = str(meta_path)[: -len(".meta.i32")]
            meta = np.fromfile(meta_path, dtype=np.int32)
            tok = np.fromfile(stem + ".tok.i32", dtype=np.int32)
            t2i = np.fromfile(stem + ".top2.i32", dtype=np.int32)
            t2v = np.fromfile(stem + ".top2.f32", dtype=np.float32)
            xs = np.fromfile(stem + ".x.f32", dtype=np.float32)
            total = int(meta.sum())
            if tok.size != total or t2i.size != 2 * total or xs.size != total * HIDDEN:
                raise SystemExit(
                    f"e142-r0: shard {stem} is inconsistent "
                    f"(rounds={meta.size} rows={total} tok={tok.size} x={xs.size})")
            o_tok = o_x = 0
            for index, width in enumerate(meta.tolist()):
                if rows_per_seed and taken >= rows_per_seed:
                    break
                rounds.append({
                    "seed": seed,
                    "round": index,
                    "width": width,
                    "tokens": tok[o_tok:o_tok + width].copy(),
                    "top2_ids": t2i[2 * o_tok:2 * (o_tok + width)].reshape(width, 2),
                    "top2_values": t2v[2 * o_tok:2 * (o_tok + width)].reshape(width, 2),
                    "hidden": xs[o_x:o_x + width * HIDDEN].reshape(width, HIDDEN),
                })
                o_tok += width
                o_x += width * HIDDEN
                taken += width
    return rounds


def load_head():
    blob = mx.load(HEAD_FILE)
    wq = blob[f"{HEAD_PREFIX}.weight"]
    scales = blob[f"{HEAD_PREFIX}.scales"]
    biases = blob[f"{HEAD_PREFIX}.biases"]
    assert list(wq.shape) == [VOCAB, HIDDEN // 8], wq.shape
    assert list(scales.shape) == [VOCAB, GROUPS], scales.shape
    return wq, scales, biases


def dequantize_chunk(wq, scales, biases, start, stop):
    return mx.dequantize(
        wq[start:stop], scales[start:stop], biases[start:stop],
        group_size=GROUP, bits=4).astype(mx.float32)


def coarse_two_bit(deq: mx.array):
    """An affine-2 group-64 copy of a dequantized weight chunk.

    Returns `(coarse, half_scale)`, where `half_scale[v, g]` is `s2_{v,g} / 2`:
    the exact worst-case per-element error of the copy inside group g. The copy
    is a pure function of the fixed target weights, so it ships no bytes and
    leaves `head_provenance_sha256` unchanged.
    """
    rows = deq.shape[0]
    grouped = deq.reshape(rows, GROUPS, GROUP)
    low = grouped.min(axis=2, keepdims=True)
    high = grouped.max(axis=2, keepdims=True)
    # Three intervals span the four levels of a 2-bit affine code.
    scale = mx.maximum((high - low) / 3.0, 1e-12)
    codes = mx.clip(mx.round((grouped - low) / scale), 0, 3)
    coarse = (codes * scale + low).reshape(rows, HIDDEN)
    return coarse, scale.reshape(rows, GROUPS) * 0.5


def merge_top2(t1, t2, b1, b2):
    """Running top two of a union, given the top two of each side."""
    return mx.maximum(t1, b1), mx.maximum(mx.minimum(t1, b1), mx.maximum(t2, b2))


def top2_of(values: mx.array):
    pair = mx.topk(values, 2, axis=1)
    return pair.max(axis=1), pair.min(axis=1)


def exact_of_pairs(H, wq, scales, biases, ids: np.ndarray) -> np.ndarray:
    """Exact logits for a small set of (verify row, vocabulary id) pairs."""
    flat = np.maximum(ids.reshape(-1), 0)
    unique, inverse = np.unique(flat, return_inverse=True)
    take = mx.array(unique.astype(np.int32))
    deq = mx.dequantize(
        wq[take], scales[take], biases[take], group_size=GROUP, bits=4
    ).astype(mx.float32)
    scores = np.array(H @ deq.T)
    rows = np.repeat(np.arange(ids.shape[0]), ids.shape[1])
    out = scores[rows, inverse.reshape(-1)].reshape(ids.shape).astype(np.float64)
    out[ids < 0] = NEG_INF
    return out


def sc_ceiling(wq, scales, biases, H, h_l2, taus, samples: int, chunk: int) -> dict:
    """An optimistic ceiling on what any leaf-of-8 centroid screen can prune.

    A leaf bound is `U_L = c_L . h + r_L * ||h||_2`. Whatever the clustering
    does, a leaf that holds row v also holds seven other rows, so its diameter
    is at least the distance from v to its 7th nearest neighbour and
    `r_L >= d_7(v) / 2`. Replacing `c_L . h` by the member's own exact logit is
    optimistic in the same direction, so the prune fraction computed here is an
    upper bound on the real index's prune fraction.
    """
    rng = np.random.default_rng(20260822)
    probe = np.sort(rng.choice(VOCAB, size=samples, replace=False)).astype(np.int32)
    take = mx.array(probe)
    probe_rows = mx.dequantize(
        wq[take], scales[take], biases[take], group_size=GROUP, bits=4
    ).astype(mx.float32)
    probe_norm2 = (probe_rows * probe_rows).sum(axis=1)
    best = mx.full((samples, ROWS_PER_LEAF), float("inf"))
    for start in range(0, VOCAB, chunk):
        deq = dequantize_chunk(wq, scales, biases, start, start + chunk)
        d2 = (probe_norm2[:, None] + (deq * deq).sum(axis=1)[None, :]
              - 2.0 * (probe_rows @ deq.T))
        best = mx.sort(
            mx.concatenate([best, mx.maximum(d2, 0.0)], axis=1), axis=1
        )[:, :ROWS_PER_LEAF]
        mx.eval(best)
        del deq, d2
    # Column 0 is the row itself at distance 0, so column 7 is the 7th nearest
    # neighbour: the closest a leaf of 8 can ever be.
    radius = np.sqrt(np.array(best)[:, ROWS_PER_LEAF - 1]) * 0.5
    probe_logits = np.array(H @ probe_rows.T)          # [N, samples]
    slack = np.array(h_l2)[:, None] * radius[None, :]  # [N, samples]
    upper = probe_logits + slack
    out = {
        "samples": samples,
        "leaf_radius_median": float(np.median(radius)),
        "leaf_radius_p05": float(np.percentile(radius, 5)),
        "row_norm_median": float(np.median(np.sqrt(np.array(probe_norm2)))),
        "slack_median": float(np.median(slack)),
        "probe_logit_median": float(np.median(probe_logits)),
    }
    for name, tau in taus.items():
        prune = (upper < tau[:, None]).mean(axis=1)
        out[f"prune_fraction_{name}_median"] = float(np.median(prune))
        out[f"prune_fraction_{name}_max"] = float(np.max(prune))
    return out


def slack_budget(s2, mean_logit, slack_a, slack_b, coarse_error) -> dict:
    """How far each bound is from the slack a certificate can afford.

    A vocabulary row is excluded only when `U_v < tau <= s2`, so the slack a
    screen may add to the AVERAGE vocabulary row is `s2 - mean_logit`. Anything
    larger keeps the whole vocabulary alive whatever the threshold is.

    The worst-case error of a b-bit affine copy of a group scales exactly as
    `1 / (2^b - 1)`, so the measured 2-bit bound also prices every other coarse
    width: `err_b = err_2 * 3 / (2^b - 1)`.
    """
    budget = np.median(s2 - mean_logit)
    err2 = float(np.median(slack_b))
    required_bits = float(np.log2(1.0 + 3.0 * err2 / budget))
    return {
        "budget_median": float(budget),
        "sa_slack_over_budget": float(np.median(slack_a) / budget),
        "sb_slack_over_budget": err2 / budget,
        "sb_actual_error_over_budget": float(np.median(coarse_error) / budget),
        "sb_required_bits_for_certificate": required_bits,
        "note": ("a coarse copy certifies only at more than "
                 f"{required_bits:.1f} bits, and the exact head is 4 bits"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--rows-per-seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=15520)
    parser.add_argument("--sc-ceiling-samples", type=int, default=4096)
    parser.add_argument("--skip-sc", action="store_true")
    parser.add_argument("--out", default="research/e142-r0.json")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir) if args.dump_dir else (
        Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows/hidden")
    seeds = [s for s in args.seeds.split(",") if s] or sorted(
        {p.name.split(".pid")[0] for p in dump_dir.glob("*.meta.i32")})

    started = time.time()
    rounds = load_capture(dump_dir, seeds, args.rows_per_seed)
    n_rounds = len(rounds)
    widths = np.array([r["width"] for r in rounds])
    hidden = np.concatenate([r["hidden"] for r in rounds], axis=0)
    s1_device = np.concatenate(
        [r["top2_values"][:, 0] for r in rounds]).astype(np.float64)
    s2_device = np.concatenate(
        [r["top2_values"][:, 1] for r in rounds]).astype(np.float64)
    top1_device = np.concatenate([r["top2_ids"][:, 0] for r in rounds])
    # Row m of a round checks the draft token at verify input position m + 1.
    # The last row of a round is the bonus row and checks no draft.
    draft_token = np.concatenate([np.append(r["tokens"][1:], -1) for r in rounds])
    seed_of_row = np.concatenate([[r["seed"]] * r["width"] for r in rounds])
    row_width = np.concatenate([np.full(r["width"], r["width"]) for r in rounds])
    first_row = np.concatenate([[0], np.cumsum(widths)])[:-1]
    n_rows = hidden.shape[0]
    max_width = int(widths.max())
    print(f"e142-r0: {n_rounds} rounds, {n_rows} verify rows, "
          f"widths {int(widths.min())}..{max_width}, seeds={','.join(seeds)}",
          flush=True)

    H = mx.array(hidden)
    h_l2 = mx.sqrt((H * H).sum(axis=1))
    h_abs = mx.abs(H)
    h_l1 = h_abs.sum(axis=1)
    h_l1_group = h_abs.reshape(n_rows, GROUPS, GROUP).sum(axis=2)
    mx.eval(h_l2, h_l1, h_l1_group)
    del h_abs

    wq, scales, biases = load_head()

    # ---- pass 1: exact top two, screen lower bounds, diagnostics -----------
    exact_t1 = mx.full((n_rows,), NEG_INF)
    exact_t2 = mx.full((n_rows,), NEG_INF)
    exact_t1_id = mx.zeros((n_rows,), dtype=mx.int32)
    lower_t1 = mx.full((n_rows,), NEG_INF)
    lower_t2 = mx.full((n_rows,), NEG_INF)
    coarse_t1 = mx.full((n_rows,), NEG_INF)
    coarse_t1_id = mx.zeros((n_rows,), dtype=mx.int32)
    slack_a = mx.zeros((n_rows,))
    slack_b = mx.zeros((n_rows,))
    coarse_abs_err = mx.zeros((n_rows,))
    exact_sum = mx.zeros((n_rows,))
    row_norms = []
    chunk = args.chunk
    assert VOCAB % chunk == 0, "chunk must divide the vocabulary"

    for start in range(0, VOCAB, chunk):
        deq = dequantize_chunk(wq, scales, biases, start, start + chunk)
        exact = H @ deq.T
        row_norm = mx.sqrt((deq * deq).sum(axis=1))
        coarse, half = coarse_two_bit(deq)
        del deq
        coarse_logit = H @ coarse.T
        err = h_l1_group @ half.T
        del coarse, half

        block_max = row_norm.reshape(chunk // ROWS_PER_LEAF, ROWS_PER_LEAF).max(axis=1)
        ub_a = h_l2[:, None] * mx.repeat(block_max, ROWS_PER_LEAF)[None, :]
        ub_b = coarse_logit + err
        lb_b = coarse_logit - err

        b1, b2 = top2_of(exact)
        exact_t1_id = mx.where(
            b1 > exact_t1, (mx.argmax(exact, axis=1) + start).astype(mx.int32),
            exact_t1_id)
        exact_t1, exact_t2 = merge_top2(exact_t1, exact_t2, b1, b2)
        l1, l2 = top2_of(lb_b)
        lower_t1, lower_t2 = merge_top2(lower_t1, lower_t2, l1, l2)
        c1 = coarse_logit.max(axis=1)
        coarse_t1_id = mx.where(
            c1 > coarse_t1, (mx.argmax(coarse_logit, axis=1) + start).astype(mx.int32),
            coarse_t1_id)
        coarse_t1 = mx.maximum(coarse_t1, c1)

        slack_a = slack_a + (ub_a - exact).sum(axis=1)
        slack_b = slack_b + (ub_b - exact).sum(axis=1)
        coarse_abs_err = coarse_abs_err + mx.abs(coarse_logit - exact).sum(axis=1)
        exact_sum = exact_sum + exact.sum(axis=1)
        mx.eval(exact_t1, exact_t2, exact_t1_id, lower_t1, lower_t2, coarse_t1,
                coarse_t1_id, slack_a, slack_b, coarse_abs_err, exact_sum, row_norm)
        row_norms.append(np.array(row_norm))
        del exact, coarse_logit, err, ub_a, ub_b, lb_b, row_norm
        print(f"  pass1 {start:>7}..{start + chunk:<7} {time.time() - started:6.1f}s",
              flush=True)
    row_norms = np.concatenate(row_norms)

    exact_t1_np = np.array(exact_t1).astype(np.float64)
    exact_t2_np = np.array(exact_t2).astype(np.float64)
    # This offline pass dequantizes to float32 and accumulates in float32,
    # while the scored QMV kernel multiplies in the model dtype. The two
    # arithmetics therefore disagree by a small amount that has to be measured
    # and carried, not assumed away: every offline bound in this report is
    # accurate only to about this gap.
    mismatch = np.array(exact_t1_id) != top1_device
    validation = {
        "offline_argmax_matches_device": float(np.mean(~mismatch)),
        "s1_abs_delta_max": float(np.max(np.abs(exact_t1_np - s1_device))),
        "s1_abs_delta_median": float(np.median(np.abs(exact_t1_np - s1_device))),
        "s2_abs_delta_max": float(np.max(np.abs(exact_t2_np - s2_device))),
        "s2_abs_delta_median": float(np.median(np.abs(exact_t2_np - s2_device))),
        "argmax_mismatch_rows": int(mismatch.sum()),
        "argmax_mismatch_s1_minus_s2_median": float(
            np.median((s1_device - s2_device)[mismatch])) if mismatch.any() else None,
        "all_rows_s1_minus_s2_median": float(np.median(s1_device - s2_device)),
    }
    arithmetic_tolerance = 10.0 * max(
        validation["s1_abs_delta_max"], validation["s2_abs_delta_max"])
    validation["arithmetic_tolerance"] = arithmetic_tolerance
    print("e142-r0: validation " + json.dumps(validation), flush=True)

    # ---- thresholds --------------------------------------------------------
    seed_ids = np.stack([draft_token, np.array(coarse_t1_id)], axis=1)
    seed_exact = exact_of_pairs(H, wq, scales, biases, seed_ids)
    tau_noseed = np.array(lower_t2).astype(np.float64)
    distinct = (seed_ids[:, 0] >= 0) & (seed_ids[:, 0] != seed_ids[:, 1])
    tau_seed = np.where(distinct, np.minimum(seed_exact[:, 0], seed_exact[:, 1]),
                        NEG_INF)
    tau_seed = np.maximum(tau_seed, tau_noseed)
    taus = {"oracle": s2_device, "seed": tau_seed, "noseed": tau_noseed}
    # A threshold above the true s2 would break the certificate. `oracle` is s2
    # by definition, so the two constructed thresholds are the ones under test,
    # and they are checked against the DEVICE s2 with the measured
    # offline-to-device arithmetic gap as the tolerance.
    threshold_check = {}
    for name in ("seed", "noseed"):
        excess = taus[name] - s2_device
        finite = np.isfinite(excess)
        threshold_check[name] = {
            "max_excess_over_device_s2": float(np.max(excess[finite])),
            "rows_over_tolerance": int(np.sum(excess[finite] > arithmetic_tolerance)),
        }
        if threshold_check[name]["rows_over_tolerance"]:
            raise SystemExit(
                f"e142-r0: threshold {name} exceeds s2 on "
                f"{threshold_check[name]['rows_over_tolerance']} of {n_rows} rows")
    print("e142-r0: thresholds " + json.dumps(threshold_check), flush=True)

    # ---- pass 2: survival at each valid threshold --------------------------
    screens = ["S-A", "S-B"]
    per_row = {s: {n: np.zeros(n_rows) for n in taus} for s in screens}
    per_round = {s: {n: np.zeros(n_rounds) for n in taus} for s in screens}
    tau_mx = {n: mx.array(t.astype(np.float32))[:, None] for n, t in taus.items()}
    # Rows of one round are contiguous, so the round-level union is at most
    # eight position-indexed ORs rather than a per-round Python loop.
    position = [(np.flatnonzero(widths > p), first_row[widths > p] + p)
                for p in range(max_width)]

    for start in range(0, VOCAB, chunk):
        deq = dequantize_chunk(wq, scales, biases, start, start + chunk)
        row_norm = mx.sqrt((deq * deq).sum(axis=1))
        coarse, half = coarse_two_bit(deq)
        del deq
        block_max = row_norm.reshape(chunk // ROWS_PER_LEAF, ROWS_PER_LEAF).max(axis=1)
        bounds = {
            "S-A": h_l2[:, None] * mx.repeat(block_max, ROWS_PER_LEAF)[None, :],
            "S-B": (H @ coarse.T) + (h_l1_group @ half.T),
        }
        del coarse, half, row_norm
        for screen in screens:
            for name in taus:
                keep = bounds[screen] >= tau_mx[name]
                mx.eval(keep)
                keep_np = np.array(keep)
                per_row[screen][name] += keep_np.sum(axis=1)
                union = np.zeros((n_rounds, keep_np.shape[1]), dtype=bool)
                for round_idx, row_idx in position:
                    union[round_idx] |= keep_np[row_idx]
                per_round[screen][name] += union.sum(axis=1)
                del keep, keep_np, union
        del bounds
        print(f"  pass2 {start:>7}..{start + chunk:<7} {time.time() - started:6.1f}s",
              flush=True)

    screen_bytes = {"S-A": 0, "S-B": VOCAB * COARSE2_ROW_BYTES}
    seed_bytes = {"oracle": 0, "seed": 2 * EXACT_ROW_BYTES, "noseed": 0}

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "seeds": seeds,
        "rounds": n_rounds,
        "rows": int(n_rows),
        "dense_round_bytes": DENSE_ROUND_BYTES,
        "exact_row_bytes": EXACT_ROW_BYTES,
        "validation": validation,
        "threshold_check": threshold_check,
        "hidden_geometry": {
            "l2_median": float(np.median(np.array(h_l2))),
            "l1_median": float(np.median(np.array(h_l1))),
            "l1_over_l2_median": float(np.median(np.array(h_l1 / h_l2))),
            "l1_over_l2_min": float(np.min(np.array(h_l1 / h_l2))),
            "l1_over_l2_max": float(np.max(np.array(h_l1 / h_l2))),
            "sqrt_hidden": float(np.sqrt(HIDDEN)),
        },
        "logit_geometry": {
            "s1_median": float(np.median(s1_device)),
            "s2_median": float(np.median(s2_device)),
            "s1_minus_s2_median": float(np.median(s1_device - s2_device)),
            "mean_logit_median": float(np.median(np.array(exact_sum) / VOCAB)),
            "row_norm_median": float(np.median(row_norms)),
            "cauchy_schwarz_median": float(
                np.median(np.array(h_l2)) * np.median(row_norms)),
            "coarse2_median_abs_error": float(
                np.median(np.array(coarse_abs_err) / VOCAB)),
            "sa_median_slack": float(np.median(np.array(slack_a) / VOCAB)),
            "sb_median_slack": float(np.median(np.array(slack_b) / VOCAB)),
        },
        "slack_budget": slack_budget(
            s2_device, np.array(exact_sum) / VOCAB, np.array(slack_a) / VOCAB,
            np.array(slack_b) / VOCAB, np.array(coarse_abs_err) / VOCAB),
        "width_histogram": {
            str(int(w)): int(np.sum(widths == w))
            for w in sorted(set(widths.tolist()))},
        "seed_value": {
            "draft_token_is_argmax_rate": float(
                np.mean(draft_token[draft_token >= 0]
                        == top1_device[draft_token >= 0])),
            "tau_seed_gap_to_s2_median": float(np.median(s2_device - tau_seed)),
            "tau_noseed_gap_to_s2_median": float(np.median(s2_device - tau_noseed)),
        },
        "screens": {},
    }

    for screen in screens:
        entry = {"screen_bytes": screen_bytes[screen], "thresholds": {}}
        for name in taus:
            rows = per_row[screen][name]
            union = per_round[screen][name]
            round_bytes = (screen_bytes[screen] + seed_bytes[name]
                           + union * EXACT_ROW_BYTES)
            frac = round_bytes / DENSE_ROUND_BYTES
            entry["thresholds"][name] = {
                "row_survival_fraction_median": float(np.median(rows) / VOCAB),
                "row_survival_fraction_p95": float(np.percentile(rows, 95) / VOCAB),
                "row_survival_fraction_worst": float(np.max(rows) / VOCAB),
                "round_byte_fraction_median": float(np.median(frac)),
                "round_byte_fraction_p95": float(np.percentile(frac, 95)),
                "round_byte_fraction_worst": float(np.max(frac)),
                "round_bytes_median": float(np.median(round_bytes)),
                "by_width": {
                    str(int(w)): {
                        "rounds": int(np.sum(widths == w)),
                        "row_survival_fraction_median": float(
                            np.median(rows[row_width == w]) / VOCAB),
                        "union_rows_median": float(np.median(union[widths == w])),
                        "byte_fraction_median": float(np.median(frac[widths == w])),
                        "byte_fraction_p95": float(
                            np.percentile(frac[widths == w], 95)),
                    } for w in sorted(set(widths.tolist()))},
                "by_seed": {
                    s: float(np.median(rows[seed_of_row == s]) / VOCAB)
                    for s in seeds},
            }
        report["screens"][screen] = entry

    if not args.skip_sc:
        report["S-C_ceiling"] = sc_ceiling(
            wq, scales, biases, H, h_l2, taus, args.sc_ceiling_samples, chunk)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"e142-r0: wrote {args.out} in {time.time() - started:.1f}s")
    for screen in screens:
        for name in taus:
            block = report["screens"][screen]["thresholds"][name]
            print(f"{screen} {name:7s} "
                  f"rows_median={block['row_survival_fraction_median']:.4f} "
                  f"bytes_median={block['round_byte_fraction_median']:.4f} "
                  f"bytes_p95={block['round_byte_fraction_p95']:.4f}")
    if not args.skip_sc:
        print("S-C ceiling " + json.dumps(report["S-C_ceiling"]))


if __name__ == "__main__":
    main()
