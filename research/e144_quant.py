"""Data-free affine quantizers for the E144 MTP head experiment.

Every quantizer here is a pure function of the master weights: no calibration
data, no activations, no training, no corpus. All three produce the identical
deployed format -- affine `bits`-bit, group `group_size` along the input
dimension, codes packed low-nibble-first into uint32 words, BF16 scales and
biases -- so they are byte-for-byte interchangeable in `mtp-head/`.

`mlx_rtn` is a faithful CPU transcription of MLX's `affine_quantize` Metal
kernel (Vendor/mlx-swift/.../kernels/quantized.h:2931). Its two quirks matter
for bit-exact reproduction and are reproduced deliberately:

  * `w_max` starts at 0 rather than -inf, so an all-negative group gets
    `w_max == 0`;
  * the grid is re-anchored on the larger-magnitude edge so that 0.0 is exactly
    representable, which makes `scale` NEGATIVE whenever |w_max| >= |w_min|.

The kernel also quantizes with the float32 scale and bias while storing them
rounded to BF16, so reconstruction error must be measured against the BF16
values actually shipped.
"""

import numpy as np

from e144_st import bf16_to_f32, f32_to_bf16

EPS = np.float32(1e-7)


def metal_round(values):
    """Round half AWAY FROM ZERO, matching Metal's `round()`.

    numpy's `round` breaks ties to even, which disagrees with Metal on exactly
    the half-integer codes and is enough to corrupt a bit-exact reproduction.
    """
    return np.trunc(values + np.copysign(np.float32(0.5), values))


def _groups(weight, group_size):
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"{columns} columns is not a multiple of {group_size}")
    return weight.reshape(rows * columns // group_size, group_size)


def pack_codes(codes, bits, rows):
    """Pack integer codes low-order-element-first into little-endian uint32."""
    per_word = 32 // bits
    flat = codes.reshape(rows, -1)
    if flat.shape[1] % per_word:
        raise ValueError("column count is not a multiple of the pack factor")
    lanes = flat.reshape(rows, -1, per_word).astype(np.uint32)
    shifts = (bits * np.arange(per_word, dtype=np.uint32)).reshape(1, 1, per_word)
    return np.bitwise_or.reduce(lanes << shifts, axis=2).astype(np.uint32)


def unpack_codes(packed, bits, columns):
    """Inverse of `pack_codes`."""
    per_word = 32 // bits
    shifts = (bits * np.arange(per_word, dtype=np.uint32)).reshape(1, 1, per_word)
    lanes = (packed[:, :, None] >> shifts) & np.uint32((1 << bits) - 1)
    return lanes.reshape(packed.shape[0], -1)[:, :columns].astype(np.uint8)


def dequantize(packed, scales, biases, bits, group_size, columns):
    """Reconstruct float32 weights exactly as the MLX dequantize kernel does."""
    codes = unpack_codes(packed, bits, columns).astype(np.float32)
    scale = bf16_to_f32(scales).repeat(group_size, axis=1)
    bias = bf16_to_f32(biases).repeat(group_size, axis=1)
    return codes * scale + bias


def _finish(weight, scale, bias, bits, group_size, rows, columns):
    """Assign codes from float32 (scale, bias), then store them as BF16.

    Returns the deployed triplet plus the float32 sum of squared error measured
    against the BF16 scale and bias that actually ship.
    """
    grouped = _groups(weight, group_size)
    n_bins = float((1 << bits) - 1)
    codes = np.clip(np.round((grouped - bias) / scale), 0.0, n_bins)

    scales_bf16 = f32_to_bf16(scale.reshape(rows, -1))
    biases_bf16 = f32_to_bf16(bias.reshape(rows, -1))
    packed = pack_codes(codes.astype(np.uint8), bits, rows)

    deployed_scale = bf16_to_f32(scales_bf16).reshape(-1, 1)
    deployed_bias = bf16_to_f32(biases_bf16).reshape(-1, 1)
    residual = grouped - (codes * deployed_scale + deployed_bias)
    return packed, scales_bf16, biases_bf16, float(np.sum(residual.astype(np.float64) ** 2))


def mlx_rtn(weight, bits=4, group_size=64):
    """MLX's stock round-to-nearest affine quantizer, transcribed exactly.

    Every intermediate stays float32 because the kernel's arithmetic is float32;
    letting numpy promote to float64 changes the codes on near-tie groups.
    """
    rows, columns = weight.shape
    grouped = _groups(weight, group_size)
    n_bins = np.float32((1 << bits) - 1)
    zero = np.float32(0.0)

    w_min = grouped.min(axis=1, keepdims=True)
    # The kernel seeds w_max at 0, not -inf.
    w_max = np.maximum(grouped.max(axis=1, keepdims=True), zero)

    scale = np.maximum((w_max - w_min) / n_bins, EPS)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)
    q0 = metal_round(edge / scale)
    at_zero = q0 == zero
    scale = np.where(at_zero, scale, edge / np.where(at_zero, np.float32(1.0), q0))
    scale = scale.astype(np.float32)
    bias = np.where(at_zero, zero, edge).astype(np.float32)

    # The kernel clamps only the upper end; the lower end is >= 0 by construction.
    codes = np.minimum(metal_round((grouped - bias) / scale), n_bins)

    scales_bf16 = f32_to_bf16(scale.reshape(rows, -1))
    biases_bf16 = f32_to_bf16(bias.reshape(rows, -1))
    packed = pack_codes(codes.astype(np.uint8), bits, rows)

    deployed_scale = bf16_to_f32(scales_bf16).reshape(-1, 1)
    deployed_bias = bf16_to_f32(biases_bf16).reshape(-1, 1)
    residual = grouped - (codes * deployed_scale + deployed_bias)
    return packed, scales_bf16, biases_bf16, float(np.sum(residual.astype(np.float64) ** 2))


def clip_search(weight, bits=4, group_size=64, low=0.60, high=1.00, steps=41):
    """Per-group MSE-optimal symmetric clipping search.

    Shrinks each group's [min, max] range by a fraction and keeps the fraction
    with the lowest sum of squared error. Fraction 1.00 is plain round-to-nearest,
    so the search can only improve on it.
    """
    rows, columns = weight.shape
    grouped = _groups(weight, group_size)
    n_bins = float((1 << bits) - 1)

    w_min = grouped.min(axis=1, keepdims=True)
    w_max = grouped.max(axis=1, keepdims=True)

    best_error = np.full((grouped.shape[0], 1), np.inf, dtype=np.float32)
    best_scale = np.empty((grouped.shape[0], 1), dtype=np.float32)
    best_bias = np.empty((grouped.shape[0], 1), dtype=np.float32)

    for fraction in np.linspace(high, low, steps, dtype=np.float32):
        lo = w_min * fraction
        hi = w_max * fraction
        scale = np.maximum((hi - lo) / n_bins, EPS)
        codes = np.clip(np.round((grouped - lo) / scale), 0.0, n_bins)
        error = np.sum((grouped - (codes * scale + lo)) ** 2, axis=1, keepdims=True)
        improved = error < best_error
        best_error = np.where(improved, error, best_error)
        best_scale = np.where(improved, scale, best_scale)
        best_bias = np.where(improved, lo, best_bias)

    return _finish(weight, best_scale, best_bias, bits, group_size, rows, columns)


def alternating_ls(weight, bits=4, group_size=64, iterations=12, init=None):
    """Per-group alternating least squares over the codes and the (scale, bias).

    Alternates between assigning codes by rounding and re-solving the
    two-parameter least-squares fit of (scale, bias) to the group. Each half-step
    is non-increasing in squared error, and the code assignment is accepted only
    when it does not increase the error, so the iteration cannot diverge.
    """
    rows, columns = weight.shape
    grouped = _groups(weight, group_size)
    n_bins = float((1 << bits) - 1)
    count = float(group_size)

    if init is None:
        w_min = grouped.min(axis=1, keepdims=True)
        w_max = grouped.max(axis=1, keepdims=True)
        scale = np.maximum((w_max - w_min) / n_bins, EPS)
        bias = w_min
    else:
        scale, bias = init

    def error_of(s, b):
        codes = np.clip(np.round((grouped - b) / s), 0.0, n_bins)
        return np.sum((grouped - (codes * s + b)) ** 2, axis=1, keepdims=True)

    best_error = error_of(scale, bias)
    best_scale, best_bias = scale.copy(), bias.copy()

    for _ in range(iterations):
        codes = np.clip(np.round((grouped - best_bias) / best_scale), 0.0, n_bins)

        # Least squares for w ~ codes * scale + bias over each group.
        sum_q = np.sum(codes, axis=1, keepdims=True)
        sum_qq = np.sum(codes * codes, axis=1, keepdims=True)
        sum_w = np.sum(grouped, axis=1, keepdims=True)
        sum_qw = np.sum(codes * grouped, axis=1, keepdims=True)

        determinant = count * sum_qq - sum_q * sum_q
        degenerate = np.abs(determinant) < 1e-12
        safe = np.where(degenerate, 1.0, determinant)
        scale = (count * sum_qw - sum_q * sum_w) / safe
        bias = (sum_qq * sum_w - sum_q * sum_qw) / safe
        scale = np.where(degenerate | (np.abs(scale) < EPS), best_scale, scale)
        bias = np.where(degenerate, best_bias, bias)

        error = error_of(scale, bias)
        improved = error < best_error
        if not np.any(improved):
            break
        best_error = np.where(improved, error, best_error)
        best_scale = np.where(improved, scale, best_scale)
        best_bias = np.where(improved, bias, best_bias)

    return _finish(weight, best_scale, best_bias, bits, group_size, rows, columns)


def _clip_init(grouped, bits, low, high, steps):
    """Best per-group clipped (scale, bias) over a fraction grid."""
    n_bins = np.float32((1 << bits) - 1)
    w_min = grouped.min(axis=1, keepdims=True)
    w_max = grouped.max(axis=1, keepdims=True)

    best_error = np.full((grouped.shape[0], 1), np.inf, dtype=np.float32)
    best_scale = np.empty((grouped.shape[0], 1), dtype=np.float32)
    best_bias = np.empty((grouped.shape[0], 1), dtype=np.float32)
    for fraction in np.linspace(high, low, steps, dtype=np.float32):
        lo = w_min * fraction
        hi = w_max * fraction
        scale = np.maximum((hi - lo) / n_bins, EPS)
        codes = np.clip(np.round((grouped - lo) / scale), 0.0, n_bins)
        error = np.sum((grouped - (codes * scale + lo)) ** 2, axis=1, keepdims=True)
        improved = error < best_error
        best_error = np.where(improved, error, best_error)
        best_scale = np.where(improved, scale, best_scale)
        best_bias = np.where(improved, lo, best_bias)
    return best_scale, best_bias, best_error


def best_of_breed(weight, bits=4, group_size=64, iterations=16, refine=9):
    """Clipping search, then alternating least squares, then a local 2D refine.

    Still a pure function of the weights. The refine pass sweeps a small
    multiplicative grid on the scale and an additive grid on the bias around the
    alternating-least-squares fixed point, which recovers most of the gap to the
    exhaustive affine optimum.
    """
    rows, columns = weight.shape
    grouped = _groups(weight, group_size)
    n_bins = np.float32((1 << bits) - 1)

    scale, bias, error = _clip_init(grouped, bits, 0.60, 1.00, 41)
    scale, bias, error = _als_loop(grouped, scale, bias, error, n_bins, group_size, iterations)

    step = scale.copy()
    for _ in range(3):
        candidates_scale = [scale * f for f in np.linspace(0.97, 1.03, refine, dtype=np.float32)]
        candidates_bias = [bias + o * step for o in np.linspace(-0.15, 0.15, refine, dtype=np.float32)]
        for trial_scale in candidates_scale:
            for trial_bias in candidates_bias:
                codes = np.clip(np.round((grouped - trial_bias) / trial_scale), 0.0, n_bins)
                trial_error = np.sum(
                    (grouped - (codes * trial_scale + trial_bias)) ** 2, axis=1, keepdims=True
                )
                improved = trial_error < error
                error = np.where(improved, trial_error, error)
                scale = np.where(improved, trial_scale, scale)
                bias = np.where(improved, trial_bias, bias)
        scale, bias, error = _als_loop(grouped, scale, bias, error, n_bins, group_size, 4)
        step = step * np.float32(0.35)

    return _finish(weight, scale, bias, bits, group_size, rows, columns)


def _als_loop(grouped, scale, bias, error, n_bins, group_size, iterations):
    """Shared alternating least-squares refinement, monotone in squared error."""
    count = np.float32(group_size)
    for _ in range(iterations):
        codes = np.clip(np.round((grouped - bias) / scale), 0.0, n_bins)
        sum_q = np.sum(codes, axis=1, keepdims=True)
        sum_qq = np.sum(codes * codes, axis=1, keepdims=True)
        sum_w = np.sum(grouped, axis=1, keepdims=True)
        sum_qw = np.sum(codes * grouped, axis=1, keepdims=True)

        determinant = count * sum_qq - sum_q * sum_q
        degenerate = np.abs(determinant) < 1e-12
        safe = np.where(degenerate, np.float32(1.0), determinant)
        trial_scale = (count * sum_qw - sum_q * sum_w) / safe
        trial_bias = (sum_qq * sum_w - sum_q * sum_qw) / safe
        trial_scale = np.where(degenerate | (np.abs(trial_scale) < EPS), scale, trial_scale)
        trial_bias = np.where(degenerate, bias, trial_bias)

        codes = np.clip(np.round((grouped - trial_bias) / trial_scale), 0.0, n_bins)
        trial_error = np.sum(
            (grouped - (codes * trial_scale + trial_bias)) ** 2, axis=1, keepdims=True
        )
        improved = trial_error < error
        if not np.any(improved):
            break
        error = np.where(improved, trial_error, error)
        scale = np.where(improved, trial_scale, scale)
        bias = np.where(improved, trial_bias, bias)
    return scale, bias, error


QUANTIZERS = {
    "rtn": mlx_rtn,
    "clip": clip_search,
    "als": alternating_ls,
    "best": best_of_breed,
}
