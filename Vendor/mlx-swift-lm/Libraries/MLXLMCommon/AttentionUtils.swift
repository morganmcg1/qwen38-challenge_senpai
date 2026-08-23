import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
///
/// LIMITATION (ContinuousBatchingV2 caches): when `cache` is a
/// `CBv2AttendingLayerCache`, the layer cache owns BOTH the KV update and
/// the attention computation, INCLUDING masking — the `mask` parameter is
/// DISCARDED on that path (v2 derives causal/window masks from per-row
/// absolute positions), and `sinks` are passed as nil. A non-adapted model
/// driven with CBv2 caches therefore silently loses any CUSTOM array mask
/// (e.g. bidirectional/prefix-LM or padding masks) and any attention
/// sinks; sinks-bearing or custom-mask models must call `updateAndAttend`
/// directly instead. Array masks fail HARD here — in release builds too
/// (`preconditionFailure`, not a debug-only assertion): silently swapping
/// the model's required mask for v2's causal/window mask would corrupt
/// output in exactly the builds users run (PR#62 review).
///
/// MULTI-ROW LIMITATION: this compatibility path is B == 1 ONLY. Legacy
/// (non-v2-adapted) models apply scalar RoPE via `KVCache.offset` BEFORE
/// calling this helper; a CBv2 cache's legacy `offset` is the MAX row
/// offset, so at B > 1 every shorter row would be silently mis-rotated.
/// B == 1 stays allowed (the scalar offset is exact for a single row).
/// Generic models must be v2-adapted — capture `positionOffsets` before
/// dispatch and call `updateAndAttend` directly — before they can serve
/// multi-row CBv2 batches. This fails loudly rather than mis-rotating.
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    // ContinuousBatchingV2 hook — see the LIMITATION notes above.
    if let v2 = cache as? CBv2AttendingLayerCache {
        if let violation = cbv2CustomMaskViolation(mask: mask, layerIndex: v2.layerIndex) {
            preconditionFailure(violation)
        }
        if let violation = cbv2LegacyAttentionBatchViolation(
            batch: queries.dim(0), layerIndex: v2.layerIndex)
        {
            preconditionFailure(violation)
        }
        return v2.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
    }
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        // WIDE-DECODE EXACTNESS CHUNK (B == 1, causal, 6 <= qL <= 9): the
        // fused sdpa vector path serves qL * gqa <= 32; above it the dispatch
        // changes kernel family and the accumulation order of every score —
        // the measured source of the MTP width wall's top-2 VALUE drift.
        // Splitting the queries at row 5 keeps both halves on the fused
        // vector path with windows that are BYTE-IDENTICAL to two
        // consecutive <= 5-row rounds at the same offsets: with bottom-right
        // causal alignment, chunk A (rows 0..<5) over keys[..<kL-(qL-5)]
        // gives row i the window a width-5 round would, and chunk B
        // (rows 5..) over the full keys gives row 5+j the window a follow-up
        // width-(qL-5) round would. Keys/values are re-sliced, not
        // recomputed — the only extra cost is one more pass over the KV
        // rows (a few MB), never over weights. Serial (qL == 1), the <= 5
        // verify widths, and prefill (qL > 9) are untouched.
        // The cache update above happens exactly once; both segments below
        // are read-only views of that single committed candidate window.
        let qL = queries.dim(2)
        let kL = cachedKeys.dim(2)
        if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL,
           case .causal = mask
        {
            // E153: one merged fused-vector call over the same keys. See
            // `Qwen35MergedSdpaVector` for the exactness argument and for the
            // key lengths it deliberately leaves to the split below.
            if let merged = qwen35MergedSdpaVector(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, qL: qL, kL: kL)
            {
                Qwen35MergedSdpaVector.recordWitness(
                    arm: .merged, qL: qL, kL: kL)
                return merged
            }
            Qwen35MergedSdpaVector.recordWitness(arm: .split, qL: qL, kL: kL)
            let split = 5
            let kSplit = kL - (qL - split)
            let outA = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., 0 ..< split, 0...],
                keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...],
                values: cachedValues[0..., 0..., 0 ..< kSplit, 0...],
                scale: scale,
                mask: .causal
            )
            let outB = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., split..., 0...],
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: .causal
            )
            return concatenated([outA, outB], axis: 2)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

// MARK: - E153 merged wide-decode SDPA

/// One fused vector SDPA call over the whole `qL x kL` causal window, as a
/// bit-exact replacement for the two-call query split in
/// `attentionWithCacheUpdate`.
///
/// WHY IT IS BIT-EXACT. `sdpa_vector` gives threadgroup `(head, row r)` the
/// keys `i <= N - tpg.y + r`. The split calls it twice:
///
///   chunk A  N = kL - (qL - 5), tpg.y = 5,      row r      -> i <= kL - qL + r
///   chunk B  N = kL,            tpg.y = qL - 5, row r - 5  -> i <= kL - qL + r
///   merged   N = kL,            tpg.y = qL,     row r      -> i <= kL - qL + r
///
/// The three forms therefore visit the SAME key set in the SAME per-row order.
/// A skipped key touches neither the running max, nor the running sum, nor the
/// output accumulator, so the float operation sequence of every row is
/// identical and the results agree bit for bit. The merged form saves one
/// kernel dispatch, one full KV stream, two query copies (both query chunks
/// are dim-2 slices, which MLX copies) and one `concatenated` per full
/// attention layer.
///
/// The kernel body below is a transcription of
/// `mlx/backend/metal/kernels/sdpa_vector.h` `sdpa_vector<T, D, V>` for
/// `has_mask = false`, `has_sinks = false`, `do_causal = true`, with the
/// hard-coded row-contiguous query offset replaced by the injected strides.
/// Every arithmetic expression is preserved verbatim, including `fast::exp`,
/// the accumulator update order and the two-stage simd reduction.
enum Qwen35MergedSdpaVector {
    static let kernelName = "qwen35_merged_sdpa_vector"
    static let threadgroupSize = 1024

    /// MLX routes to `sdpa_vector_2pass` at `kL >= 1024` on architectures
    /// whose name ends in `d` or `s` -- which includes the ranked M5 runner --
    /// and at `kL >= 4096` with GQA everywhere else. The 2-pass form splits
    /// the key range into blocks and combines partial softmax states, so it is
    /// NOT the accumulation this kernel reproduces. Refuse the whole band
    /// rather than depend on the host architecture string: `kL` reaches 1024
    /// only in the last rounds of a 512-seed, 512-decode leg.
    static let twoPassKeyLength = 1024

    static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]

    /// `0` selects the shipped two-call split. Any other value, including
    /// unset, selects the merged kernel. Only `MLX_`-prefixed names survive
    /// the runtime worker environment filter.
    static let enabled: Bool =
        ProcessInfo.processInfo.environment["MLX_E153_MERGED_SDPA"] != "0"

    /// Emit one witness line per distinct `(arm, qL)` the process reaches
    /// instead of one line per arm.
    static let censusEnabled: Bool =
        ProcessInfo.processInfo.environment["MLX_E153_SDPA_CENSUS"] == "1"

    enum Arm: String {
        case merged
        case split
    }

    private static let witnessLock = NSLock()
    nonisolated(unsafe) private static var witnessed: Set<String> = []
    // Read without the lock on the timed path. A lost race repeats one witness
    // line and changes nothing else, and the check must not allocate: this runs
    // once per full attention layer per round.
    nonisolated(unsafe) private static var mergedWitnessed = false
    nonisolated(unsafe) private static var splitWitnessed = false

    /// Rule 114: the timed leg must be able to state which arm it ran from its
    /// own output. Written to the configured trace file, which is opened
    /// `O_APPEND` -- the `mtp-timed` parent swallows worker stderr on a
    /// successful leg, so stderr alone is not readable evidence.
    static func recordWitness(arm: Arm, qL: Int, kL: Int) {
        if censusEnabled {
            let key = "\(arm.rawValue)/\(qL)"
            witnessLock.lock()
            let isNew = witnessed.insert(key).inserted
            witnessLock.unlock()
            guard isNew else { return }
        } else {
            switch arm {
            case .merged:
                guard !mergedWitnessed else { return }
                mergedWitnessed = true
            case .split:
                guard !splitWitnessed else { return }
                splitWitnessed = true
            }
        }
        var line = "qwen35-merged-sdpa: arm=\(arm.rawValue) qL=\(qL) kL=\(kL)"
        line += " enabled=\(enabled) census=\(censusEnabled)\n"
        let data = Data(line.utf8)
        if let path = ProcessInfo.processInfo
            .environment["MLX_QWEN_MTP_TRACE_PATH"], !path.isEmpty
        {
            let descriptor = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
            if descriptor >= 0 {
                let handle = FileHandle(
                    fileDescriptor: descriptor, closeOnDealloc: true)
                handle.write(data)
                try? handle.close()
                return
            }
        }
        FileHandle.standardError.write(data)
    }

    static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: kernelName,
        inputNames: ["queries", "keys", "values", "scale"],
        outputNames: ["out"],
        source: source,
        // The KV of one layer is several MB. A row-contiguous copy of it per
        // call would cost far more than the split this kernel replaces, and
        // the cache slice it receives is already exactly what MLX's own
        // `sdpa_vector` consumes through strides.
        ensureRowContiguous: false)

    static let source = """
          constexpr int BN = 32;
          constexpr int BD = 32;
          constexpr int qk_per_thread = D / BD;
          constexpr int v_per_thread = V / BD;
          typedef float U;

          const int H = queries_shape[1];
          const int kv_heads = keys_shape[1];
          const int N = keys_shape[2];
          const int gqa_factor = H / kv_heads;

          const int q_batch_head_idx = int(threadgroup_position_in_grid.x);
          const int q_seq_idx = int(threadgroup_position_in_grid.y);
          const int qL = int(threadgroups_per_grid.y);
          const int batch_idx = q_batch_head_idx / H;
          const int q_head_idx = q_batch_head_idx % H;
          const int kv_head_idx = q_head_idx / gqa_factor;
          const int simd_gid = int(simdgroup_index_in_threadgroup);
          const int simd_lid = int(thread_index_in_simdgroup);

          const int64_t q_last_stride = queries_strides[3];
          const int64_t k_last_stride = keys_strides[3];
          const int64_t v_last_stride = values_strides[3];
          const int64_t k_seq_stride = keys_strides[2];
          const int64_t v_seq_stride = values_strides[2];
          const int64_t inner_k_stride = int64_t(BN) * k_seq_stride;
          const int64_t inner_v_stride = int64_t(BN) * v_seq_stride;

          const device T* q_ptr = queries
              + batch_idx * queries_strides[0]
              + q_head_idx * queries_strides[1]
              + q_seq_idx * queries_strides[2]
              + int64_t(simd_lid) * qk_per_thread * q_last_stride;
          const device T* k_ptr = keys
              + batch_idx * keys_strides[0]
              + kv_head_idx * keys_strides[1]
              + int64_t(simd_gid) * k_seq_stride
              + int64_t(simd_lid) * qk_per_thread * k_last_stride;
          const device T* v_ptr = values
              + batch_idx * values_strides[0]
              + kv_head_idx * values_strides[1]
              + int64_t(simd_gid) * v_seq_stride
              + int64_t(simd_lid) * v_per_thread * v_last_stride;
          device T* o_ptr = out
              + int64_t(q_batch_head_idx * qL + q_seq_idx) * V
              + simd_gid * v_per_thread;

          thread U q[qk_per_thread];
          thread U k[qk_per_thread];
          thread U o[v_per_thread];
          threadgroup U outputs[BN * BD];
          threadgroup U max_scores[BN];
          threadgroup U sum_exp_scores[BN];

          for (int i = 0; i < qk_per_thread; i++) {
            q[i] = static_cast<U>(scale) * q_ptr[i * q_last_stride];
          }
          for (int i = 0; i < v_per_thread; i++) {
            o[i] = 0;
          }

          U max_score = Limits<U>::finite_min;
          U sum_exp_score = 0;
          const int causal_limit = N - qL + q_seq_idx;

          // Two spellings of one loop. The unit-stride form lets the compiler
          // widen the 8 element key read and value read into vector loads; the
          // general form is correctness insurance for a layout the scored path
          // never produces. Everything else is identical, so both agree bit
          // for bit with `sdpa_vector`.
          if (k_last_stride == 1 && v_last_stride == 1) {
            for (int i = simd_gid; i < N; i += BN) {
              if (i <= causal_limit) {
                for (int j = 0; j < qk_per_thread; j++) {
                  k[j] = k_ptr[j];
                }
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) {
                  score += q[j] * k[j];
                }
                score = simd_sum(score);
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int j = 0; j < v_per_thread; j++) {
                  o[j] = o[j] * factor + exp_score * v_ptr[j];
                }
              }
              k_ptr += inner_k_stride;
              v_ptr += inner_v_stride;
            }
          } else {
            for (int i = simd_gid; i < N; i += BN) {
              if (i <= causal_limit) {
                for (int j = 0; j < qk_per_thread; j++) {
                  k[j] = k_ptr[j * k_last_stride];
                }
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) {
                  score += q[j] * k[j];
                }
                score = simd_sum(score);
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int j = 0; j < v_per_thread; j++) {
                  o[j] = o[j] * factor + exp_score * v_ptr[j * v_last_stride];
                }
              }
              k_ptr += inner_k_stride;
              v_ptr += inner_v_stride;
            }
          }

          if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          max_score = max_scores[simd_lid];
          U new_max = simd_max(max_score);
          U factor = fast::exp(max_score - new_max);
          sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

          for (int i = 0; i < v_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          if (simd_lid == 0) {
            for (int i = 0; i < v_per_thread; i++) {
              o_ptr[i] = static_cast<T>(o[i]);
            }
          }
        """
}

/// Run the merged wide-decode SDPA kernel, or return nil when this call is
/// outside the band it reproduces exactly. A nil result is not an error: the
/// caller falls through to the shipped split.
///
/// The kernel assumes the innermost dimension of `out` is contiguous, which it
/// owns, and reads every input through its own strides.
public func qwen35MergedSdpaVector(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    qL: Int,
    kL: Int
) -> MLXArray? {
    guard Qwen35MergedSdpaVector.enabled else { return nil }
    guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }
    let heads = queries.dim(1)
    let kvHeads = keys.dim(1)
    let headDim = queries.dim(3)
    let valueDim = values.dim(3)
    guard queries.dim(0) == 1, keys.dim(0) == 1, values.dim(0) == 1,
        queries.dim(2) == qL, keys.dim(2) == kL, values.dim(2) == kL,
        values.dim(1) == kvHeads, keys.dim(3) == headDim,
        kvHeads > 0, heads > 0, heads % kvHeads == 0,
        headDim > 0, headDim % 32 == 0, headDim <= 256,
        valueDim > 0, valueDim % 32 == 0, valueDim <= 256,
        qL >= 1, kL >= qL,
        kL < Qwen35MergedSdpaVector.twoPassKeyLength,
        queries.dtype == keys.dtype, queries.dtype == values.dtype,
        Qwen35MergedSdpaVector.supportedDTypes.contains(queries.dtype)
    else { return nil }

    let outputs = Qwen35MergedSdpaVector.kernel(
        [queries, keys, values, MLXArray(scale)],
        template: [("T", queries.dtype), ("D", headDim), ("V", valueDim)],
        grid: (Qwen35MergedSdpaVector.threadgroupSize * heads, qL, 1),
        threadGroup: (Qwen35MergedSdpaVector.threadgroupSize, 1, 1),
        outputShapes: [[1, heads, qL, valueDim]],
        outputDTypes: [queries.dtype])
    return outputs[0]
}

/// Compile the merged kernel outside the timed window.
///
/// Rule 110: a warm-phase change is worth nothing unless it moves a pipeline
/// cache key. This one does the opposite work -- it moves the FIRST TOUCH of a
/// JIT Metal source compile, which costs tens to hundreds of milliseconds, out
/// of the scored leg. One call compiles the whole family: the pipeline is keyed
/// by the template arguments `(T, D, V)` only, while `qL` arrives through the
/// launch grid and `kL` through the key shape.
///
/// Returns true when the kernel actually ran, so a caller can assert the warm
/// was not silently skipped.
@discardableResult
public func warmQwen35MergedSdpaVector(
    keys: MLXArray, values: MLXArray, queryHeads: Int, scale: Float
) -> Bool {
    guard Qwen35MergedSdpaVector.enabled, keys.ndim == 4 else { return false }
    let warmKeyLength = min(
        keys.dim(2), Qwen35MergedSdpaVector.twoPassKeyLength - 1)
    guard warmKeyLength >= 8 else { return false }
    let warmKeys = keys[0..., 0..., 0 ..< warmKeyLength, 0...]
    let warmValues = values[0..., 0..., 0 ..< warmKeyLength, 0...]
    let queries = MLXArray.zeros(
        [1, queryHeads, 6, keys.dim(3)], dtype: keys.dtype)
    guard let out = qwen35MergedSdpaVector(
        queries: queries, keys: warmKeys, values: warmValues,
        scale: scale, qL: 6, kL: warmKeyLength)
    else { return false }
    eval(out)
    return true
}

/// Custom-mask guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the LIMITATION doc there). `.none`/`.causal` are subsumed by v2's own
/// position-derived masks; any other mode (custom `.array`/`.arrays`) would
/// be silently DISCARDED by `updateAndAttend`, so it must fail in ALL build
/// configurations — a debug-only `assertionFailure` compiles out of release
/// builds and lets the wrong mask ship (PR#62 review). Returns the failure
/// description for an illegal call, or nil when the call is allowed.
/// Internal (not private) so tests can pin the exact condition without
/// tripping the precondition.
func cbv2CustomMaskViolation(
    mask: MLXFast.ScaledDotProductAttentionMaskMode, layerIndex: Int
) -> String? {
    switch mask {
    case .none, .causal:
        return nil
    default:
        return """
            attentionWithCacheUpdate: a custom array mask was passed with a \
            CBv2 layer cache (layer \(layerIndex)). CBv2 caches own their \
            masks and DISCARD this parameter — the model must be v2-adapted \
            (call updateAndAttend with its own semantics) instead.
            """
    }
}

/// Multi-row guard for the CBv2 branch of `attentionWithCacheUpdate` (see
/// the MULTI-ROW LIMITATION doc there). Returns the failure description for
/// an illegal call, or nil when the call is allowed. Internal (not private)
/// so tests can pin the exact condition without tripping the precondition.
func cbv2LegacyAttentionBatchViolation(batch: Int, layerIndex: Int) -> String? {
    guard batch > 1 else { return nil }
    return """
        attentionWithCacheUpdate: a multi-row batch (B=\(batch)) reached the \
        legacy CBv2 compatibility path (layer \(layerIndex)). Legacy models \
        apply scalar RoPE via `KVCache.offset` — the MAX row offset — so \
        shorter rows would be silently mis-rotated at B > 1. This model must \
        be v2-adapted (read `positionOffsets` before dispatch and call \
        `updateAndAttend` directly) before multi-row CBv2 serving; B == 1 \
        remains supported.
        """
}
