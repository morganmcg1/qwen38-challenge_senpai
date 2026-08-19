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
        let isCausal: Bool
        if case .causal = mask { isCausal = true } else { isCausal = false }
        let arm = SdpaWideChunkArm.current
        let wantsChunk = queries.dim(0) == 1 && isCausal && arm.wantsChunk(qL: qL, kL: kL)
        SdpaWideChunkTrace.call(
            qL: qL, kL: kL, batch: queries.dim(0), causal: isCausal,
            arm: arm, chunked: wantsChunk)
        if wantsChunk {
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

/// E57 RESEARCH INSTRUMENT — arm selector for the wide-decode exactness chunk.
///
/// The shipped predicate chunks every causal decode call at `6 <= qL <= 9`. The
/// trusted host dispatcher
/// (`mlx/backend/metal/scaled_dot_product_attention.cpp:685, :746-753`) shows
/// the `qL * gqa <= 32` threadgroup limit that motivated the chunk binds only
/// the `sdpa_vector_2pass` route, which is selected for `qL <= 8` only when the
/// architecture letter is `'d'` or `'s'` AND `kL >= 1024`. `narrow` tests that
/// reading, `off` is the positive control that must fail, and `base` is the
/// shipped predicate. One build therefore serves all three arms, so an arm
/// comparison carries no build difference. Default is the shipped predicate:
/// an unset or unknown environment value cannot change scored behaviour.
///
/// The name carries the `MLX_` prefix because
/// `sanitizedRuntimeWorkerEnvironment` admits only a small allowlist into the
/// runtime worker, and `MLXFAST_` is deliberately not part of it.
enum SdpaWideChunkArm: String {
    case base
    case narrow
    case off

    static let current: SdpaWideChunkArm = {
        let raw = ProcessInfo.processInfo.environment["MLX_E57_SDPA_CHUNK_ARM"] ?? ""
        return SdpaWideChunkArm(rawValue: raw) ?? .base
    }()

    func wantsChunk(qL: Int, kL: Int) -> Bool {
        guard qL >= 6, qL <= 9, kL >= qL else { return false }
        switch self {
        case .base: return true
        case .narrow: return qL >= 9 || kL >= 1024
        case .off: return false
        }
    }
}

/// E57 RESEARCH INSTRUMENT — one line per cached-KV decode SDPA call.
///
/// Records only the facts the trusted dispatcher keys on: batch, `qL`, `kL`,
/// causality, the selected arm and whether the chunk fired. The route each
/// call takes is derived offline from the quoted dispatcher conditions, so this
/// instrument stays free of any device query. Off unless
/// `MLX_E57_SDPA_TRACE=1`.
///
/// The sink mirrors `Qwen36MTPBlockSession.traceSink`: the `mtp-timed` parent
/// discards worker stderr, so a readable trace needs an `O_APPEND` file that
/// the reference, serial and decode workers can share. Every line carries the
/// writing process id, which is what separates the three legs of one local
/// run.
enum SdpaWideChunkTrace {
    static let enabled =
        ProcessInfo.processInfo.environment["MLX_E57_SDPA_TRACE"] == "1"

    private static let sink: FileHandle = {
        guard let path = ProcessInfo.processInfo
            .environment["MLX_E57_SDPA_TRACE_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()

    private static let pid = ProcessInfo.processInfo.processIdentifier

    static func call(
        qL: Int, kL: Int, batch: Int, causal: Bool,
        arm: SdpaWideChunkArm, chunked: Bool
    ) {
        guard enabled else { return }
        let reason: String
        if !chunked {
            reason = "none"
        } else if qL >= 9 {
            reason = "width9"
        } else if kL >= 1024 {
            reason = "kl1024"
        } else {
            reason = "widthonly"
        }
        let line = "e57-sdpa: pid=\(pid) qL=\(qL) kL=\(kL) b=\(batch) "
            + "causal=\(causal ? 1 : 0) arm=\(arm.rawValue) "
            + "chunk=\(chunked ? 1 : 0) reason=\(reason)\n"
        sink.write(Data(line.utf8))
    }
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
