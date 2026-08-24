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
            let split = 5
            let kSplit = kL - (qL - split)
            let barrier = Qwen35SplitCellBarrier.arm
            let keysA = cachedKeys[0..., 0..., 0 ..< kSplit, 0...]
            let valuesA = cachedValues[0..., 0..., 0 ..< kSplit, 0...]
            if barrier == .all { eval(keysA); eval(valuesA) }
            let outA = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., 0 ..< split, 0...],
                keys: keysA,
                values: valuesA,
                scale: scale,
                mask: .causal
            )
            if barrier == .all { eval(outA) }
            let outB = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., split..., 0...],
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: .causal
            )
            if barrier == .all { eval(outB) }
            let out = concatenated([outA, outB], axis: 2)
            if barrier != .shipped { eval(out) }
            Qwen35SplitCellBarrier.served(width: qL, barriers: barrier.barrierCount)
            return out
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

/// RESEARCH ONLY (E202), delete before any submission (RULE 198).
///
/// Selects one of three `eval()` barrier arms inside the shipped qL 6...9
/// split-cell branch above, to settle whether MLX overlaps that branch's five
/// dispatches. The arm is switched IN PROCESS at every round boundary
/// (`Qwen36MTPBlockSession.generateRound`), so all arms share one binary, one
/// head, one thermal state and one token stream (RULE 388). `eval()` changes
/// no computed value, so every arm emits the identical token stream and the
/// per-round width sequence is identical in every leg.
///
/// THE SWITCH PREFIX IS LOAD BEARING (HARNESS DEFECT 28). The runtime worker
/// rebuilds its environment from an allowlist that excludes `MLXFAST_*`, so
/// only a `DARKBLOOM_`-prefixed model-side opt-in survives (RULE 391(a)).
public enum Qwen35SplitCellBarrier {
    public enum Arm: Int {
        /// The unmodified shipped branch: no barrier.
        case shipped = 0
        /// One barrier after the final dispatch of the group. Prices the
        /// per-`eval()` sync cost without de-overlapping the interior.
        case last = 1
        /// One barrier after each of the five dispatches.
        case all = 2

        var barrierCount: Int {
            switch self {
            case .shipped: return 0
            case .last: return 1
            case .all: return 5
            }
        }
    }

    /// Round-rotation offset for the arm schedule. Unset means "never arm a
    /// barrier": the branch then runs exactly as shipped in every round.
    /// RULE 391(a) allowlisted name.
    private static let offset: Int? = ProcessInfo.processInfo
        .environment["DARKBLOOM_E202_BARRIER_OFFSET"].flatMap { Int($0) }

    private static let schedule: [Arm] = [.shipped, .last, .all]

    nonisolated(unsafe) public private(set) static var arm: Arm = .shipped

    /// RULE 391(b) witness, arm-cost-symmetric (RULE 391(c)): the counter work
    /// below is identical in all three arms, and only the `eval()` calls
    /// differ. `calls`/`barriers` are the per-round witness carried in the
    /// round trace line; `widthCensus` is the leg-wide served-width histogram.
    nonisolated(unsafe) public private(set) static var calls = 0
    nonisolated(unsafe) public private(set) static var barriers = 0
    nonisolated(unsafe) public private(set) static var lastWidth = 0
    nonisolated(unsafe) public private(set) static var widthCensus: [Int: Int] = [:]

    /// Whether this process ever armed a barrier. A leg that reads
    /// `armed=false` on a barrier arm is VOID, not null.
    public static var armSelectionActive: Bool { offset != nil }

    public static func beginRound(_ round: Int) {
        calls = 0
        barriers = 0
        lastWidth = 0
        guard let offset else {
            arm = .shipped
            return
        }
        arm = schedule[((round + offset) % schedule.count + schedule.count)
            % schedule.count]
    }

    @inline(__always)
    static func served(width: Int, barriers barrierCount: Int) {
        calls += 1
        barriers += barrierCount
        lastWidth = width
        widthCensus[width, default: 0] += 1
    }

    /// Compact census string for the round trace line.
    public static func censusWitness() -> String {
        widthCensus.sorted { $0.key < $1.key }
            .map { "\($0.key):\($0.value)" }
            .joined(separator: "|")
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
