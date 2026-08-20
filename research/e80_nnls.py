#!/usr/bin/env python3
"""Per-kernel GPU time from command-buffer signatures, by non-negative least squares.

Why this exists
---------------
Rung 0c chose `MLX_MAX_OPS_PER_BUFFER=1` to put one dispatch in one command
buffer, so that the buffer's `GPUEndTime - GPUStartTime` would BE that
dispatch's GPU time. The rung-2 debug leg refuted the premise. MLX honours the
limit, but an MLX op is not a Metal dispatch: the isolated leg still averages
2.04 dispatches per buffer, and only 0.4 % of verify dispatches ever land in a
buffer of their own. Direct per-dispatch pricing is not reachable by this
mechanism.

The same data supports a stronger estimator. Every buffer records an exact GPU
interval and the exact multiset of dispatches inside it, so each buffer is one
linear equation

    gpu_time(buffer) = sum over kernels k of  count(k, buffer) * t_k

in the unknown per-dispatch times `t_k`. The isolated leg gives thousands of
such equations over a few dozen unknowns, and the SAME kernel appears in many
different pairings, which is what makes the system identifiable. Solving it
under a non-negativity constraint recovers every kernel's time, not just the
0.4 % that happened to run alone.

What `t_k` means
----------------
Dispatches inside one buffer can overlap when MLX uses a concurrent encoder,
so the fitted `t_k` is the kernel's *effective serialised contribution* to GPU
busy time, not its wall duration in isolation. That is the correct quantity for
this census: the fitted times reconstruct the measured total by construction,
so shares of the verify block always close to 100 %. Fitting the default-mode
and isolated-mode legs separately and dividing gives the concurrency discount
the assignment asks for, per family.

Solver
------
Lawson-Hanson active-set NNLS. scipy is not installed on this host, so it is
implemented here against numpy alone. It terminates at the exact KKT point for
a full-rank active set, which plain `lstsq` plus clipping does not.
"""
from __future__ import annotations

import numpy as np


def nnls(A: np.ndarray, b: np.ndarray, max_iter: int | None = None,
         tol: float = 1e-10):
    """Lawson-Hanson non-negative least squares: min ||Ax - b|| with x >= 0."""
    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        if passive.all() or (w[~passive] <= tol).all():
            break
        candidates = np.where(~passive, w, -np.inf)
        passive[int(np.argmax(candidates))] = True
        while True:
            idx = np.where(passive)[0]
            s = np.zeros(n)
            s[idx] = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            if (s[idx] > 0).all():
                x = s
                break
            # Move to the constraint boundary and drop what went non-positive.
            blocking = idx[s[idx] <= 0]
            alpha = np.min(x[blocking] / (x[blocking] - s[blocking]))
            x = x + alpha * (s - x)
            passive &= x > tol
            x[~passive] = 0.0
            if not passive.any():
                break
        w = A.T @ (b - A @ x)
    return x, float(np.linalg.norm(A @ x - b))


def solve_kernel_times(rows):
    """Fit per-dispatch GPU times from buffer signatures.

    `rows` is an iterable of (counts, total_gpu_ns, n_buffers) where `counts`
    maps a dispatch key to how many times it occurs in ONE buffer of that
    signature, `total_gpu_ns` is the summed GPU time over all `n_buffers`
    buffers that carried it, and `n_buffers` is that count.

    Each signature contributes one equation, weighted by sqrt(n_buffers) so a
    signature seen 1249 times counts far more than one seen twice. Returns the
    fitted per-dispatch nanoseconds, plus fit diagnostics.
    """
    rows = [r for r in rows if r[2] > 0]
    if not rows:
        return {}, {}
    keys = sorted({k for counts, _, _ in rows for k in counts})
    index = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(rows), len(keys)))
    b = np.zeros(len(rows))
    weights = np.zeros(len(rows))
    for r, (counts, total_ns, n_buffers) in enumerate(rows):
        for k, c in counts.items():
            A[r, index[k]] = c
        b[r] = total_ns / n_buffers          # mean GPU ns for this signature
        weights[r] = np.sqrt(n_buffers)
    Aw = A * weights[:, None]
    bw = b * weights
    x, residual = nnls(Aw, bw)

    measured = float((b * weights**2).sum())
    predicted = float(((A @ x) * weights**2).sum())
    rank = int(np.linalg.matrix_rank(Aw))
    diag = {
        "signatures": len(rows),
        "kernels": len(keys),
        "rank": rank,
        "rank_deficient": rank < len(keys),
        "weighted_residual_ns": residual,
        "measured_weighted_ns": measured,
        "predicted_weighted_ns": predicted,
        "closure": predicted / measured if measured else None,
        "zero_fitted": [k for k in keys if x[index[k]] <= 0],
    }
    return {k: float(x[index[k]]) for k in keys}, diag
