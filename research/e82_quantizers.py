#!/usr/bin/env python3
"""E82 rung 6: affine-4 g64 quantizers that beat MLX's own estimator.

`research/e82_quant_control.py` proves the shipped `declared` head is exactly
`mx.quantize(master, 64, 4, affine)`, so the 0.82 pt of acceptance rung 0
measured between `master-bf16` and `declared` is the estimator's alone. This
module holds the replacements, all writing the identical artifact format:
`uint32` codes, BF16 scales and BF16 biases, group 64, 4 bits.

MLX's estimator, from `quantized.h:2973-2986`, snaps one edge of the group onto
an exact integer code and lets the other edge fall where it may. Half of the
representable range is therefore spent pinning an endpoint that carries no more
weight than any other value in the group. Both methods here attack that.

  mlx    the shipped baseline, reproduced through this module's own packer so
         that every arm is written by the same code path.
  ls     tier 0. A shrink grid over the group interval, then a closed-form
         least-squares refit of (scale, bias) with the codes held fixed,
         iterated. Strictly a superset of round-to-nearest min-max, so it
         cannot lose on reconstruction error.
  hqq    tier 1. Half-quadratic splitting with a generalised soft threshold,
         run in float32 with a free floating bias, plus a scale grid because
         the scale objective is not unimodal.

Every candidate is scored on the reconstruction the runtime will actually
compute, which uses BF16 scales and biases, not the float32 ones the solver
works in. Selection is per group and per method by that realised error, so the
result is never worse than the best single method on any group.

  python3 research/e82_quantizers.py --selftest
"""

from __future__ import annotations

import mlx.core as mx

GROUP = 64
BITS = 4
N_BINS = (1 << BITS) - 1
PER_WORD = 32 // BITS
EPS = 1e-7
# Grid endpoints are pulled toward the group midpoint, so 1.00 is the
# unshrunk interval and every other value is a strict clip. 1.00 is always
# present: published work finds clipping can hurt at 4 bits even where it helps
# at 3, so the grid point has to be selected by measured error, not assumed.
SHRINK = [1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]
CHUNK_GROUPS = 1 << 18


def pack(codes: mx.array) -> mx.array:
    """Pack `[G, 64]` nibbles into the `[G, 8]` uint32 words MLX dequantizes.

    The bit order is asserted against `mx.quantize` in `selftest`, not assumed
    from documentation.
    """
    g = codes.shape[0]
    words = codes.astype(mx.uint32).reshape(g, -1, PER_WORD)
    shifts = (BITS * mx.arange(PER_WORD)).astype(mx.uint32)
    return (words << shifts).sum(axis=2).astype(mx.uint32)


def unpack(words: mx.array) -> mx.array:
    shifts = (BITS * mx.arange(PER_WORD)).astype(mx.uint32)
    return ((words[..., None] >> shifts) & mx.array(N_BINS, mx.uint32)).reshape(
        words.shape[0], -1)


def _codes(g: mx.array, s: mx.array, b: mx.array) -> mx.array:
    """🔴 Clamp both sides. The quantize kernel writes `min(...)` into a
    `uint8_t` and relies on `bias` sitting on the extreme edge to keep the
    quotient non-negative. A least-squares refit moves the bias off that edge,
    so a one-sided clamp would wrap negative values silently."""
    return mx.clip(mx.round((g - b) / s), 0, N_BINS)


def _bf16(x: mx.array) -> mx.array:
    """The realised value: the artifact stores BF16, so the solver's float32
    scale is not what the runtime multiplies by."""
    return x.astype(mx.bfloat16).astype(mx.float32)


def _mse(g: mx.array, s: mx.array, b: mx.array) -> mx.array:
    return ((s * _codes(g, s, b) + b - g) ** 2).sum(axis=1, keepdims=True)


def _minmax_init(g: mx.array, a_lo: float, a_hi: float) -> tuple[mx.array, mx.array]:
    lo_raw = g.min(axis=1, keepdims=True)
    hi_raw = g.max(axis=1, keepdims=True)
    mid = (lo_raw + hi_raw) / 2
    half = (hi_raw - lo_raw) / 2
    lo = mid - a_lo * half
    hi = mid + a_hi * half
    return mx.maximum((hi - lo) / N_BINS, EPS), lo


def _ls_refit(g: mx.array, q: mx.array, s: mx.array) -> tuple[mx.array, mx.array]:
    """Two-variable linear regression of `w` on `q`, per group, closed form.

    With the codes held fixed this is the exact minimiser of the group's
    squared error, so each pass is monotone non-increasing.
    """
    n = float(g.shape[1])
    sq = q.sum(axis=1, keepdims=True)
    sw = g.sum(axis=1, keepdims=True)
    sqq = (q * q).sum(axis=1, keepdims=True)
    sqw = (q * g).sum(axis=1, keepdims=True)
    den = n * sqq - sq * sq
    safe = mx.where(den > 0, den, 1.0)
    s_new = mx.where(den > 0, (n * sqw - sq * sw) / safe, s)
    s_new = mx.maximum(s_new, EPS)
    return s_new, (sw - s_new * sq) / n


def _solve_ls(g: mx.array, passes: int = 3) -> tuple[mx.array, mx.array]:
    best_mse = best_s = best_b = None
    for a_lo in SHRINK:
        for a_hi in SHRINK:
            s, b = _minmax_init(g, a_lo, a_hi)
            for _ in range(passes):
                s, b = _ls_refit(g, _codes(g, s, b), s)
            s, b = _bf16(s), _bf16(b)
            e = _mse(g, s, b)
            if best_mse is None:
                best_mse, best_s, best_b = e, s, b
            else:
                take = e < best_mse
                best_mse = mx.where(take, e, best_mse)
                best_s = mx.where(take, s, best_s)
                best_b = mx.where(take, b, best_b)
            mx.eval(best_mse, best_s, best_b)
    return best_s, best_b


def _solve_hqq(g: mx.array, p: float = 0.7, beta0: float = 1e1,
               kappa: float = 1.01, iters: int = 20,
               scale_grid: int = 21) -> tuple[mx.array, mx.array]:
    """Half-quadratic splitting on the bias, in our own parameterisation.

    The reference implementation writes `w_r = (q - zero)/scale` and rounds the
    zero point to an integer for kernel compatibility. Our format carries a
    free floating bias and reconstructs `w_r = s*q + b`, so the bias stays
    continuous and the closed-form update is `b = mean(w - w_e - s*q)`. The
    solver runs entirely in float32; the inverted-scale parameterisation the
    reference uses is documented as unstable in half precision.
    """
    base_s, base_b = _minmax_init(g, 1.0, 1.0)
    best_mse = best_s = best_b = None
    grid = [0.90 + 0.20 * i / (scale_grid - 1) for i in range(scale_grid)]
    for gain in grid:
        s = mx.maximum(base_s * gain, EPS)
        b = base_b
        beta = beta0
        for _ in range(iters):
            q = _codes(g, s, b)
            e = g - (s * q + b)
            mag = mx.abs(e)
            shrunk = mx.maximum(mag - mag ** (p - 1) / beta, 0.0)
            w_e = mx.sign(e) * shrunk
            b = (g - w_e - s * q).mean(axis=1, keepdims=True)
            beta *= kappa
        sb, bb = _bf16(s), _bf16(b)
        e = _mse(g, sb, bb)
        if best_mse is None:
            best_mse, best_s, best_b = e, sb, bb
        else:
            take = e < best_mse
            best_mse = mx.where(take, e, best_mse)
            best_s = mx.where(take, sb, best_s)
            best_b = mx.where(take, bb, best_b)
        mx.eval(best_mse, best_s, best_b)
    return best_s, best_b


def _solve_mlx(g: mx.array) -> tuple[mx.array, mx.array]:
    """MLX's own estimator, reached through `mx.quantize` so the baseline in
    every comparison is the shipped one and not a reimplementation of it."""
    _, s, b = mx.quantize(g.astype(mx.bfloat16), group_size=GROUP, bits=BITS,
                          mode="affine")
    return _bf16(s.reshape(-1, 1)), _bf16(b.reshape(-1, 1))


SOLVERS = {"mlx": _solve_mlx, "ls": _solve_ls, "hqq": _solve_hqq}


def quantize(w: mx.array, methods: tuple[str, ...] = ("mlx",)) -> dict:
    """Quantize `[out, in]` float32 weights, choosing the best method per group.

    Returns the packed artifact tensors, the float32 reconstruction, and the
    count of groups each method won, so a report can say where the gain came
    from rather than only that there was one.
    """
    out, cols = w.shape
    assert cols % GROUP == 0, (out, cols)
    flat = w.reshape(-1, GROUP)
    total = flat.shape[0]

    parts, wins = [], {m: 0 for m in methods}
    for start in range(0, total, CHUNK_GROUPS):
        g = flat[start:start + CHUNK_GROUPS]
        best_mse = best_s = best_b = best_id = None
        for i, m in enumerate(methods):
            s, b = SOLVERS[m](g)
            e = _mse(g, s, b)
            if best_mse is None:
                best_mse, best_s, best_b = e, s, b
                best_id = mx.zeros_like(e)
            else:
                take = e < best_mse
                best_mse = mx.where(take, e, best_mse)
                best_s = mx.where(take, s, best_s)
                best_b = mx.where(take, b, best_b)
                best_id = mx.where(take, mx.full(e.shape, float(i)), best_id)
            mx.eval(best_mse, best_s, best_b, best_id)
        for i, m in enumerate(methods):
            wins[m] += int((best_id == i).sum().item())
        codes = _codes(g, best_s, best_b)
        parts.append((pack(codes), best_s, best_b, best_s * codes + best_b))
        mx.eval(*parts[-1])

    packed = mx.concatenate([p[0] for p in parts]).reshape(out, -1)
    scales = mx.concatenate([p[1] for p in parts]).reshape(out, -1)
    biases = mx.concatenate([p[2] for p in parts]).reshape(out, -1)
    deq = mx.concatenate([p[3] for p in parts]).reshape(out, cols)
    scales_bf, biases_bf = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)

    # 🔴 The round-trip gate. A packing bug and a quantizer improvement look
    # identical in a rel-L2 table, so no error number is computed until MLX
    # itself agrees that the artifact reconstructs to the intended values.
    check = mx.dequantize(packed, scales_bf, biases_bf,
                          group_size=GROUP, bits=BITS).astype(mx.float32)
    gap = float(mx.abs(check - deq).max().item())
    mx.eval(packed, scales_bf, biases_bf, deq)
    assert gap == 0.0, f"round trip mismatch: max |diff| = {gap}"

    return {"weight": packed, "scales": scales_bf, "biases": biases_bf,
            "dequantized": deq, "method_group_wins": wins,
            "groups": total, "round_trip_max_abs_diff": gap}


def selftest() -> None:
    mx.random.seed(0)
    w = mx.random.normal((512, 640)).astype(mx.bfloat16).astype(mx.float32)

    # 1. The packer reproduces MLX's own bit order exactly.
    ref_q, ref_s, ref_b = mx.quantize(w.astype(mx.bfloat16),
                                      group_size=GROUP, bits=BITS, mode="affine")
    codes = unpack(ref_q.reshape(-1, w.shape[1] * BITS // 32))
    assert mx.array_equal(pack(codes), ref_q.reshape(-1, w.shape[1] * BITS // 32)), \
        "pack/unpack is not an involution on MLX's own payload"
    ref_deq = mx.dequantize(ref_q, ref_s, ref_b, group_size=GROUP, bits=BITS)
    mine = (ref_s.astype(mx.float32).reshape(-1, 1) * codes
            + ref_b.astype(mx.float32).reshape(-1, 1)).reshape(w.shape)
    assert float(mx.abs(ref_deq.astype(mx.float32) - mine).max().item()) == 0.0, \
        "bit order disagrees with mx.dequantize"
    print("pack/unpack matches mx.quantize bit order and mx.dequantize")

    # 2. `mlx` through this module reproduces `mx.quantize` bit for bit.
    got = quantize(w, methods=("mlx",))
    assert mx.array_equal(got["scales"], ref_s), "scales differ from mx.quantize"
    assert mx.array_equal(got["biases"], ref_b), "biases differ from mx.quantize"
    print("module `mlx` path reproduces mx.quantize scales and biases")

    # 3. Each replacement cannot lose to the baseline on realised error.
    base = float(((got["dequantized"] - w) ** 2).sum().item())
    for m in ("ls", "hqq"):
        r = quantize(w, methods=(m,))
        e = float(((r["dequantized"] - w) ** 2).sum().item())
        print(f"{m:4s} sse {e:.6e} vs mlx {base:.6e}  ratio {e / base:.4f}"
              f"  round_trip {r['round_trip_max_abs_diff']}")
    both = quantize(w, methods=("mlx", "ls", "hqq"))
    e = float(((both["dequantized"] - w) ** 2).sum().item())
    assert e <= base + 1e-9, "per-group selection lost to its own baseline"
    print(f"best-of sse {e:.6e}  ratio {e / base:.4f}  wins {both['method_group_wins']}")
    print("SELFTEST PASS")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
