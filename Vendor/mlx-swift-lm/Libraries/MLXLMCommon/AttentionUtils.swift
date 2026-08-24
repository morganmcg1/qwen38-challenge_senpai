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
    FusedRowAmortizedSDPA.Liveness.reached("awcu_entry")
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
        FusedRowAmortizedSDPA.Liveness.reached("awcu_else_qL\(qL)")
        if case .causal = mask {
            FusedRowAmortizedSDPA.Liveness.reached("causal_qL\(qL)")
        }
        if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL,
           case .causal = mask
        {
            if let fused = FusedRowAmortizedSDPA.attend(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale)
            {
                return fused
            }
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

/// One-pass fused SDPA for the wide-decode split (E198).
///
/// The shipped qL 6...9 branch issues TWO `sdpa_vector` dispatches: one over
/// the shortened window for rows 0...4 and one over the full window for the
/// remaining rows. This kernel replaces both with a single dispatch. Row `r`
/// attends keys `0 ... N - M + r`, which is the exact key range each global row
/// already receives from the split, so every row reproduces the split bit for
/// bit, including the top-2 evidence the split exists to protect
/// (`attentionWithCacheUpdate`, the qL 6...9 branch).
///
/// The kernel keeps the vendored per-row arithmetic exactly — 32 simdgroups x
/// 32 lanes, the same key-to-simdgroup assignment, the same online-softmax
/// update order, the same cross-simdgroup max/sum reduction and transposed
/// output combine — and it keeps one threadgroup per (head, row).
///
/// An earlier form of this kernel instead held all M rows of one head in
/// registers, to read each KV tile once per head rather than once per row.
/// Measured on M4 Pro, that form costs 1.4x to 4.9x more than M separate
/// one-row dispatches: collapsing the row into the threadgroup removes the
/// dispatch grid's only source of parallelism at these widths, and the KV
/// re-read it saves is not the binding resource at kL 517...1023.
///
/// Bit-exactness needs the full 1024-thread threadgroup: a narrower group would
/// change which keys a simdgroup accumulates and so change the summation order.
/// Register pressure that pushes the pipeline's `maxTotalThreadsPerThreadgroup`
/// below 1024 is therefore a hard wall, not a slow path, and MLX reports it as
/// a thrown dispatch error.
public enum FusedRowAmortizedSDPA {
    public static let headDim = 256
    public static let bn = 32
    public static let bd = 32

    /// Key length at which the vendored dispatch may switch to
    /// `sdpa_vector_2pass`. The fused kernel serves only shorter keys.
    public static let twoPassKeyLength = 1024

    /// Query-row counts the fused kernel serves. `DARKBLOOM_QWEN_FUSED_SDPA_ROWS`
    /// selects them for a measurement arm; with the variable unset the kernel
    /// is off and the shipped two-call split runs unchanged. RULE 198 requires
    /// this switch to become a compile-time constant before any submission.
    ///
    /// THE PREFIX IS LOAD BEARING (HARNESS DEFECT 28).
    /// `sanitizedRuntimeWorkerEnvironment` rebuilds
    /// the runtime worker's environment from empty and copies in only an
    /// allowlist, to stop submitted code from reading a benchmark-phase oracle.
    /// `MLXFAST_*` is excluded by design, so an `MLXFAST_`-named model-side arm
    /// switch is silently dropped and every arm measures the default. Only the
    /// `DARKBLOOM_` prefix is allowlisted for model-side participant opt-ins.
    public static let enabledRows: Set<Int> = {
        guard let raw = ProcessInfo.processInfo
            .environment["DARKBLOOM_QWEN_FUSED_SDPA_ROWS"]
        else {
            return []
        }
        return Set(raw.split(separator: ",").compactMap { Int($0) })
    }()

    /// RESEARCH PROBE (E198), delete before any submission. A null timing
    /// result cannot distinguish "the fused kernel ran and did not help" from
    /// "the fused kernel never ran", because `attend` declines silently. This
    /// counts served calls and declines by reason, and writes the totals to
    /// `DARKBLOOM_E198_LIVENESS_OUT` when the process exits. It is inert unless
    /// that variable is set, and no timed arm sets it.
    public enum Liveness {
        nonisolated(unsafe) private static var servedCounts: [Int: Int] = [:]
        nonisolated(unsafe) private static var declinedCounts: [String: Int] = [:]
        nonisolated(unsafe) private static var reachedCounts: [String: Int] = [:]

        /// The requested path, or a home-directory fallback. The fallback
        /// exists so that "no file" means "this code did not run" and can never
        /// also mean "the environment variable did not arrive". `HOME` is
        /// allowlisted by `sanitizedRuntimeWorkerEnvironment` and its value is
        /// known from outside the worker, so the search cannot look in the
        /// wrong place; `NSTemporaryDirectory()` is not, because this workspace
        /// overrides `TMPDIR` while CoreFoundation answers with /var/folders.
        nonisolated(unsafe) private static let envPath = ProcessInfo.processInfo
            .environment["DARKBLOOM_E198_LIVENESS_OUT"]
        nonisolated(unsafe) private static let path =
            envPath ?? (NSHomeDirectory() as NSString)
            .appendingPathComponent("e198-liveness.json")

        private static let armed: Bool = {
            atexit { Liveness.dump() }
            return true
        }()

        /// Count a call site upstream of `attend`, so a missing `attend` call
        /// is distinguishable from a declined one.
        /// Set true for one research leg only. A crash is the only witness that
        /// no file path, environment variable, or discarded stderr stream can
        /// swallow, so it separates "this code never runs" from "the evidence
        /// never reached me". It answered that question: the leg died with
        /// `reached qwen35_attn_entry_L512`, proving the path is live and the
        /// file channel was at fault. Never true in a timed or submitted build.
        private static let abortOnFirstReach = false

        public static func reached(_ key: String) {
            if abortOnFirstReach { fatalError("E198 LIVENESS: reached \(key)") }
            guard armed else { return }
            reachedCounts[key, default: 0] += 1
            flushIfDue()
        }

        // The worker can be killed rather than exited, so `atexit` alone can
        // leave no evidence. Flush on the first call and then periodically.
        nonisolated(unsafe) private static var calls = 0
        private static let flushEvery = 128

        static func served(rows: Int) {
            guard armed else { return }
            servedCounts[rows, default: 0] += 1
            flushIfDue()
        }

        static func declined(reason: String, rows: Int) {
            guard armed else { return }
            declinedCounts["\(reason)_m\(rows)", default: 0] += 1
            flushIfDue()
        }

        private static func flushIfDue() {
            calls += 1
            if calls == 1 || calls % flushEvery == 0 { dump() }
        }

        static func dump() {
            func object(_ pairs: [(String, Int)]) -> String {
                "{" + pairs.map { "\"\($0.0)\": \($0.1)" }.joined(separator: ", ") + "}"
            }
            let served = object(
                servedCounts.sorted { $0.key < $1.key }.map { ("\($0.key)", $0.value) })
            let declined = object(
                declinedCounts.sorted { $0.key < $1.key }.map { ($0.key, $0.value) })
            let reached = object(reachedCounts.sorted { $0.key < $1.key })
            let rowsEnv = ProcessInfo.processInfo
                .environment["DARKBLOOM_QWEN_FUSED_SDPA_ROWS"] ?? "<unset>"
            let json = """
                {"served_by_rows": \(served), "declined_by_reason": \(declined), \
                "reached": \(reached), "enabled_rows": \(enabledRows.sorted()), \
                "rows_env": "\(rowsEnv)", "out_env_set": \(envPath != nil), \
                "pid": \(ProcessInfo.processInfo.processIdentifier)}
                """
            do {
                try json.write(toFile: path, atomically: true, encoding: .utf8)
                FileHandle.standardError.write(
                    Data("e198-liveness: wrote \(path)\n".utf8))
            } catch {
                FileHandle.standardError.write(
                    Data("e198-liveness: FAILED \(path): \(error)\n".utf8))
            }
        }
    }

    private static func decline(_ reason: String, _ rows: Int) -> MLXArray? {
        Liveness.declined(reason: reason, rows: rows)
        return nil
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen_mtp_fused_row_amortized_sdpa",
        inputNames: ["queries", "keys", "values", "scale"],
        outputNames: ["out"],
        source: kernelSource,
        header: "#include <metal_simdgroup>\n",
        ensureRowContiguous: false
    )

    /// The fused output for a `[1, H, M, 256]` causal wide-decode step, or nil
    /// when this shape is not served and the caller must keep the split.
    ///
    /// `serving` is the set of query-row counts to fuse. Research harnesses
    /// pass an explicit set to price a width the shipped default does not
    /// serve; production takes the default.
    public static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        serving: Set<Int> = enabledRows
    ) -> MLXArray? {
        let rows = queries.dim(2)
        guard serving.contains(rows) else { return decline("width", rows) }
        guard queries.dim(0) == 1, keys.dim(0) == 1, values.dim(0) == 1,
            queries.dim(3) == headDim, keys.dim(3) == headDim,
            values.dim(3) == headDim,
            queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            values.dtype == .bfloat16
        else { return decline("shape_or_dtype", rows) }

        let heads = queries.dim(1)
        let kvHeads = keys.dim(1)
        guard kvHeads > 0, heads % kvHeads == 0, values.dim(1) == kvHeads,
            values.dim(2) == keys.dim(2), keys.dim(2) >= rows
        else { return decline("head_grouping", rows) }

        // The kernel reads each thread's 8 head-dim elements as one contiguous
        // run, which is what every cache layout in this tree provides.
        guard keys.strides[3] == 1, values.strides[3] == 1 else {
            return decline("kv_stride", rows)
        }

        // This kernel reproduces `sdpa_vector` arithmetic. At or above
        // `twoPassKeyLength` the vendored dispatch in
        // `scaled_dot_product_attention.cpp` can select `sdpa_vector_2pass`,
        // whose block-partitioned softmax combine rounds differently, so
        // serving that range would change emitted logits by one bf16 ulp.
        // Declining is unconditional rather than architecture-gated: the
        // vendored condition also reads the device architecture string, and a
        // fallback is always exact.
        guard keys.dim(2) < twoPassKeyLength else {
            return decline("two_pass_boundary", rows)
        }

        Liveness.served(rows: rows)
        return kernel(
            [queries, keys, values, MLXArray(scale)],
            template: [("M", rows)],
            grid: (bd, bn, heads * rows),
            threadGroup: (bd, bn, 1),
            outputShapes: [[1, heads, rows, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    /// The kernel body, public so an exactness harness can build its
    /// positive control from the exact shipped source with one substitution.
    public static let kernelSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int D = 256;
        constexpr int QK = D / BD;
        constexpr int VP = D / BD;
        threadgroup float outputs[BN * BD];
        threadgroup float max_scores[BN];
        threadgroup float sum_exp_scores[BN];

        const int simd_gid = int(simdgroup_index_in_threadgroup);
        const int simd_lid = int(thread_index_in_simdgroup);
        // One threadgroup per (head, query row). Holding the M rows inside a
        // single threadgroup instead removes the dispatch grid's only source
        // of parallelism at these widths.
        const int slot = int(threadgroup_position_in_grid.z);
        const int q_head = slot / M;
        const int q_row = slot % M;

        const int num_q_heads = int(queries_shape[1]);
        const int num_kv_heads = int(keys_shape[1]);
        const int kv_head = q_head / (num_q_heads / num_kv_heads);
        const int N = int(keys_shape[2]);

        const int64_t q_dim_stride = queries_strides[3];
        const int64_t k_seq_stride = keys_strides[2];
        const int64_t v_seq_stride = values_strides[2];

        thread float q[QK];
        thread float o[VP];

        const int64_t q_base = int64_t(q_head) * queries_strides[1]
            + int64_t(q_row) * queries_strides[2]
            + int64_t(simd_lid * QK) * q_dim_stride;
        for (int j = 0; j < QK; j++) {
            q[j] = scale
                * static_cast<float>(queries[q_base + int64_t(j) * q_dim_stride]);
        }
        for (int j = 0; j < VP; j++) {
            o[j] = 0;
        }
        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0;

        const device bfloat16_t* k_ptr = keys
            + int64_t(kv_head) * keys_strides[1]
            + int64_t(simd_gid) * k_seq_stride + int64_t(simd_lid * QK);
        const device bfloat16_t* v_ptr = values
            + int64_t(kv_head) * values_strides[1]
            + int64_t(simd_gid) * v_seq_stride + int64_t(simd_lid * VP);
        const int64_t inner_k_stride = int64_t(BN) * k_seq_stride;
        const int64_t inner_v_stride = int64_t(BN) * v_seq_stride;

        // Bottom-right causal alignment: row r attends keys 0 ... N - M + r.
        // This one rule reproduces both calls of the shipped split, whose
        // shortened head window and full tail window give each global row the
        // same key range.
        for (int i = simd_gid; i < N; i += BN) {
            if (i <= N - M + q_row) {
                float k[QK];
                for (int j = 0; j < QK; j++) {
                    k[j] = static_cast<float>(k_ptr[j]);
                }

                float score = 0;
                for (int j = 0; j < QK; j++) {
                    score += q[j] * k[j];
                }
                score = simd_sum(score);

                float new_max = max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                for (int j = 0; j < VP; j++) {
                    o[j] = o[j] * factor
                        + exp_score * static_cast<float>(v_ptr[j]);
                }
            }
            k_ptr += inner_k_stride;
            v_ptr += inner_v_stride;
        }

        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lid];
        float new_max = simd_max(max_score);
        float factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

        for (int j = 0; j < VP; j++) {
            outputs[simd_lid * BD + simd_gid] = o[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[j] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[j] = sum_exp_score == 0 ? o[j] : (o[j] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            device bfloat16_t* out_ptr = out + int64_t(slot) * D + simd_gid * VP;
            for (int j = 0; j < VP; j++) {
                out_ptr[j] = static_cast<bfloat16_t>(o[j]);
            }
        }
        """
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
