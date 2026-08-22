#!/usr/bin/env python3
"""Time and validate the warm-time basis the C1 sketch can actually ship.

The screened cell `qlowrank256-N4096-p0.35` uses a QUERY-fitted basis: the top
eigenvectors of the hidden-state second moment, fitted on the E133 corpus. That
matrix is 5,120 x 256 and it cannot ship. The editable source budget has
103,166 bytes of growth left and the matrix is 2,621,440 bytes as bfloat16, so
it is 25x over before any text encoding, and shipping it as data would need a
loader outside this experiment's declared scope.

The only in-scope alternative is a basis that is a deterministic function of the
checkpoint. For an isotropic query the optimal rank-`k` projection is the top-`k`
eigenvector set of the ROW second moment, which is exactly the screen's
`lowrank` family, so there is no better checkpoint-only choice to search for.

This probe answers the two engineering questions that decide whether the Swift
build can compute that basis during warm:

  1. What does a direct `eigh` of the 5,120 x 5,120 row second moment cost on
     this host? Warm is untimed but it is not free.
  2. Does fixed-seed subspace iteration with QR reach the same subspace, and in
     how many iterations? Captured energy is the metric that matters, because
     the sketch error is `row . (I - P)(x - mu)`.

It also measures how much QUERY energy the row basis keeps, so the gap to the
query-fitted basis is a measured number instead of an assertion.

    usage: research/e136_basis_probe.py [--rank 256] [--iters 2,4,8,16]
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

import e87_head as H
import e133_index as IX
from e133_screen import Index, Screen, query_second_moments

SEED = 20260822


def second_moment(rows: mx.array, mu: mx.array, chunk: int = 8192) -> mx.array:
    """`(1/N) sum (row - mu)(row - mu)^T`, chunked so no float32 copy of the
    whole 98,336 x 5,120 table has to land at once."""
    acc = mx.zeros((H.HIDDEN, H.HIDDEN), mx.float32)
    for a in range(0, rows.shape[0], chunk):
        d = rows[a:a + chunk].astype(mx.float32) - mu
        acc = acc + mx.matmul(d.T, d)
        mx.eval(acc)
    return acc / rows.shape[0]


def subspace_iteration(cov: mx.array, rank: int, iters: int) -> mx.array:
    """Fixed-seed subspace iteration, reorthonormalized by QR each round.

    The sketch estimator is `(x - mu) P P^T (row - mu)`, so `P` must stay
    orthonormal or the estimator is not the inner product it claims to be.
    """
    q = mx.random.normal((H.HIDDEN, rank), key=mx.random.key(SEED))
    q, _ = mx.linalg.qr(q, stream=mx.cpu)
    for _ in range(iters):
        q, _ = mx.linalg.qr(mx.matmul(cov, q), stream=mx.cpu)
        mx.eval(q)
    return q


def captured(cov: mx.array, basis: mx.array) -> float:
    """`trace(P^T C P) / trace(C)`: the fraction of `C`'s energy `P` keeps."""
    num = float(mx.sum(mx.matmul(cov, basis) * basis).item())
    return num / float(mx.trace(cov).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--iters", default="2,4,8,16")
    ap.add_argument("--out", default="research/e136-basis-probe.json")
    args = ap.parse_args()

    t0 = time.time()
    screen = Screen(Index(IX.DEFAULT_OUT))
    t_load = time.time() - t0

    t0 = time.time()
    cov = second_moment(screen.rows, screen.mu)
    mx.eval(cov)
    t_cov = time.time() - t0

    t0 = time.time()
    values, vectors = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(values, vectors)
    t_eigh = time.time() - t0
    exact = vectors[:, ::-1][:, :args.rank]
    mx.eval(exact)

    report = {
        "load_seconds": t_load,
        "second_moment_seconds": t_cov,
        "eigh_5120_seconds": t_eigh,
        "rank": args.rank,
        "seed": SEED,
        "exact_row_energy": captured(cov, exact),
        "iterations": {},
    }
    print(f"load {t_load:.1f}s  second moment {t_cov:.1f}s  "
          f"eigh(5120) {t_eigh:.1f}s")
    print(f"exact row energy kept at rank {args.rank}: "
          f"{report['exact_row_energy']:.6f}")

    for k in (int(s) for s in args.iters.split(",")):
        t0 = time.time()
        q = subspace_iteration(cov, args.rank, k)
        dt = time.time() - t0
        # Both bases are orthonormal, so the mean squared singular value of
        # `exact^T q` is the fraction of the exact subspace the iterate spans.
        overlap = mx.matmul(exact.T, q)
        align = float(mx.sum(overlap * overlap).item()) / args.rank
        energy = captured(cov, q)
        report["iterations"][str(k)] = {
            "seconds": dt,
            "row_energy": energy,
            "subspace_alignment": align,
            "energy_ratio": energy / report["exact_row_energy"],
        }
        print(f"  iters {k:3d}  {dt:6.2f}s  row energy {energy:.6f}  "
              f"ratio {energy / report['exact_row_energy']:.6f}  "
              f"alignment {align:.6f}")

    qcov = query_second_moments(screen.mu)
    report["query_energy"] = {}
    for fold in ("beagle", "min_carriers"):
        c = mx.array(qcov[fold].astype("float32"))
        row_keep = captured(c, exact)
        qvalues, qvectors = mx.linalg.eigh(c, stream=mx.cpu)
        mx.eval(qvalues, qvectors)
        qbasis = qvectors[:, ::-1][:, :args.rank]
        report["query_energy"][fold] = {
            "kept_by_row_basis": row_keep,
            "kept_by_query_basis": captured(c, qbasis),
        }
        print(f"query energy on {fold}: row basis keeps {row_keep:.6f}, "
              f"query basis keeps "
              f"{report['query_energy'][fold]['kept_by_query_basis']:.6f}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
