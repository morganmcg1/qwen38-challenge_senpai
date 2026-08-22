#!/usr/bin/env python3
"""E142 rung 0, screen S-D: a low-rank certified screen for the verify readout.

S-A, S-B and S-C all bound the logit through a WORST-CASE per-element weight
error or a radius in the full 5,120-dimensional space, so their slack grows
like `sqrt(5120)` relative to a typical inner product. S-D removes that factor
instead of fighting it.

Let `V_k` hold the top k right singular vectors of the fixed `lm_head` matrix
and let `P_k = V_k V_k^T`. For every vocabulary row,

    logit_v = w_v . h = (P_k w_v) . (P_k h) + (w_v - P_k w_v) . (h - P_k h)

and Cauchy-Schwarz on the SECOND term only gives

    |logit_v - proj_v(k)|  <=  rho_v(k) * ||h - P_k h||_2

with `proj_v(k) = (V_k^T w_v) . (V_k^T h)` and
`rho_v(k) = ||w_v - P_k w_v||_2`. Both `V_k` and `rho` are pure functions of
the fixed target weights, so they are input-independent derived tables that
ship no bytes and leave `head_provenance_sha256` unchanged. The bound holds for
every input whatever the basis is, so the certificate and the dense fallback
keep the emitted stream bit-identical by construction.

The screen reads `248,320 x k` projected coefficients instead of
`248,320 x 5,120` packed weights, so it pays only when the residual product
`rho_v * ||h_perp||` is small enough to exclude most of the vocabulary. That is
exactly what this script measures.

Usage:
  research/e142_r0_lowrank.py [--ranks 128,256,512,1024,2048] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from e142_r0 import (COARSE2_ROW_BYTES, DENSE_ROUND_BYTES, EXACT_ROW_BYTES,
                     GROUP, GROUPS, HIDDEN, ROWS_PER_LEAF, VOCAB,
                     dequantize_chunk, load_capture, load_head)


def gram(wq, scales, biases, chunk: int, started: float) -> np.ndarray:
    """`W^T W` for the whole vocabulary, accumulated in float32."""
    total = mx.zeros((HIDDEN, HIDDEN))
    for start in range(0, VOCAB, chunk):
        deq = dequantize_chunk(wq, scales, biases, start, start + chunk)
        total = total + deq.T @ deq
        mx.eval(total)
        del deq
        print(f"  gram {start:>7} {time.time() - started:6.1f}s", flush=True)
    return np.array(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--rows-per-seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=15520)
    parser.add_argument("--ranks", default="128,256,512,1024,2048")
    parser.add_argument("--coefficient-bytes", type=int, default=1,
                        help="bytes per projected coefficient in the byte model")
    parser.add_argument("--out", default="research/e142-r0-lowrank.json")
    args = parser.parse_args()

    ranks = sorted(int(r) for r in args.ranks.split(","))
    dump_dir = Path(args.dump_dir) if args.dump_dir else (
        Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows/hidden")
    seeds = [s for s in args.seeds.split(",") if s] or sorted(
        {p.name.split(".pid")[0] for p in dump_dir.glob("*.meta.i32")})

    started = time.time()
    rounds = load_capture(dump_dir, seeds, args.rows_per_seed)
    n_rounds = len(rounds)
    widths = np.array([r["width"] for r in rounds])
    first_row = np.concatenate([[0], np.cumsum(widths)])[:-1]
    max_width = int(widths.max())
    hidden = np.concatenate([r["hidden"] for r in rounds], axis=0)
    s2_device = np.concatenate(
        [r["top2_values"][:, 1] for r in rounds]).astype(np.float64)
    n_rows = hidden.shape[0]
    print(f"e142-lowrank: {n_rounds} rounds, {n_rows} rows, ranks={ranks}",
          flush=True)

    H = mx.array(hidden)
    h_norm2 = np.array((H * H).sum(axis=1)).astype(np.float64)
    wq, scales, biases = load_head()

    G = gram(wq, scales, biases, args.chunk, started)
    # `eigh` returns ascending eigenvalues, so reverse into descending energy.
    values, vectors = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    basis = np.ascontiguousarray(vectors[:, order]).astype(np.float32)
    energy = np.cumsum(values) / values.sum()
    print(f"e142-lowrank: weight energy at ranks "
          f"{ {k: round(float(energy[k - 1]), 4) for k in ranks} }", flush=True)

    V = mx.array(basis)
    g = H @ V                       # [N, HIDDEN] hidden coordinates in the basis
    mx.eval(g)
    g_energy = np.cumsum(np.array(g * g), axis=1)
    h_perp = {k: np.sqrt(np.maximum(h_norm2 - g_energy[:, k - 1], 0.0))
              for k in ranks}

    tau = mx.array(s2_device.astype(np.float32))[:, None]
    survivors_row = {k: np.zeros(n_rows) for k in ranks}
    survivors_round = {k: np.zeros(n_rounds) for k in ranks}
    slack_sum = {k: np.zeros(n_rows) for k in ranks}
    rho_sum = {k: 0.0 for k in ranks}
    position = [(np.flatnonzero(widths > p), first_row[widths > p] + p)
                for p in range(max_width)]
    perp_mx = {k: mx.array(h_perp[k].astype(np.float32))[:, None] for k in ranks}

    for start in range(0, VOCAB, args.chunk):
        deq = dequantize_chunk(wq, scales, biases, start, start + args.chunk)
        row_norm2 = (deq * deq).sum(axis=1)
        Z = deq @ V                 # [C, HIDDEN] weight coordinates in the basis
        mx.eval(Z, row_norm2)
        del deq
        z_energy = mx.cumsum(Z * Z, axis=1)
        for k in ranks:
            proj = g[:, :k] @ Z[:, :k].T
            rho = mx.sqrt(mx.maximum(row_norm2 - z_energy[:, k - 1], 0.0))
            upper = proj + perp_mx[k] * rho[None, :]
            keep = upper >= tau
            mx.eval(keep, rho)
            keep_np = np.array(keep)
            survivors_row[k] += keep_np.sum(axis=1)
            union = np.zeros((n_rounds, keep_np.shape[1]), dtype=bool)
            for round_idx, row_idx in position:
                union[round_idx] |= keep_np[row_idx]
            survivors_round[k] += union.sum(axis=1)
            slack_sum[k] += np.array((perp_mx[k] * rho[None, :]).sum(axis=1))
            rho_sum[k] += float(np.array(rho.sum()))
            del proj, rho, upper, keep, keep_np, union
        del Z, z_energy, row_norm2
        print(f"  screen {start:>7} {time.time() - started:6.1f}s", flush=True)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "seeds": seeds,
        "rounds": n_rounds,
        "rows": int(n_rows),
        "coefficient_bytes": args.coefficient_bytes,
        "dense_round_bytes": DENSE_ROUND_BYTES,
        "threshold": "oracle: tau = s2 as the shipped dense readout measured it",
        "ranks": {},
    }
    for k in ranks:
        screen_bytes = VOCAB * k * args.coefficient_bytes
        union = survivors_round[k]
        round_bytes = screen_bytes + union * EXACT_ROW_BYTES
        frac = round_bytes / DENSE_ROUND_BYTES
        report["ranks"][str(k)] = {
            "weight_energy": float(energy[k - 1]),
            "h_perp_over_h_median": float(
                np.median(h_perp[k] / np.sqrt(h_norm2))),
            "rho_mean": rho_sum[k] / VOCAB,
            "slack_median": float(np.median(slack_sum[k] / VOCAB)),
            "screen_bytes": screen_bytes,
            "row_survival_fraction_median": float(
                np.median(survivors_row[k]) / VOCAB),
            "row_survival_fraction_p95": float(
                np.percentile(survivors_row[k], 95) / VOCAB),
            "row_survival_fraction_worst": float(
                np.max(survivors_row[k]) / VOCAB),
            "round_byte_fraction_median": float(np.median(frac)),
            "round_byte_fraction_p95": float(np.percentile(frac, 95)),
            "round_byte_fraction_worst": float(np.max(frac)),
            "by_width": {
                str(int(w)): {
                    "rounds": int(np.sum(widths == w)),
                    "union_rows_median": float(np.median(union[widths == w])),
                    "byte_fraction_median": float(np.median(frac[widths == w])),
                } for w in sorted(set(widths.tolist()))},
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"e142-lowrank: wrote {args.out} in {time.time() - started:.1f}s")
    for k in ranks:
        cell = report["ranks"][str(k)]
        print(f"k={k:<5} energy={cell['weight_energy']:.4f} "
              f"hperp={cell['h_perp_over_h_median']:.4f} "
              f"slack={cell['slack_median']:8.3f} "
              f"rows={cell['row_survival_fraction_median']:.4f} "
              f"bytes={cell['round_byte_fraction_median']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
