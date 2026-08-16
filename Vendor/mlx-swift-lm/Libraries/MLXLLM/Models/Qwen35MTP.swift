// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

// MARK: - Host-cost trace (local research instrumentation)

/// Nanosecond accumulators for the head's host-side graph-build cost, gated by
/// the same `MLX_QWEN_MTP_TRACE=1` switch the block session uses. Decode is
/// single threaded, so plain statics are sufficient; the session resets them
/// around the sub-steps it wants attributed.
public enum Qwen35MTPHostTrace {
    public nonisolated(unsafe) static let enabled =
        ProcessInfo.processInfo.environment["MLX_QWEN_MTP_TRACE"] == "1"
    public nonisolated(unsafe) static var embedNs: UInt64 = 0
    public nonisolated(unsafe) static var fuseNs: UInt64 = 0
    public nonisolated(unsafe) static var maskNs: UInt64 = 0
    public nonisolated(unsafe) static var layerNs: UInt64 = 0
    public nonisolated(unsafe) static var normNs: UInt64 = 0
    public nonisolated(unsafe) static var attnNs: UInt64 = 0
    public nonisolated(unsafe) static var mlpNs: UInt64 = 0
    public nonisolated(unsafe) static var layerNormNs: UInt64 = 0

    public static func reset() {
        embedNs = 0
        fuseNs = 0
        maskNs = 0
        layerNs = 0
        normNs = 0
        attnNs = 0
        mlpNs = 0
        layerNormNs = 0
    }

    @inline(__always)
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
}

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            // Same fused gate/up MLP as the backbone layers; here the linears
            // stay bf16 and the fuse takes the plain-weight path. Head side —
            // proposal-only, no exactness constraint.
            _mlp.wrappedValue = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        guard Qwen35MTPHostTrace.enabled else {
            let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
        let t0 = Qwen35MTPHostTrace.now()
        let normed = inputLayerNorm(x)
        let t1 = Qwen35MTPHostTrace.now()
        let r = selfAttn(normed, mask: mask, cache: cache)
        let t2 = Qwen35MTPHostTrace.now()
        let h = x + r
        let postNormed = postAttentionLayerNorm(h)
        let t3 = Qwen35MTPHostTrace.now()
        let out = h + (mlp as! UnaryLayer)(postNormed)
        let t4 = Qwen35MTPHostTrace.now()
        Qwen35MTPHostTrace.layerNormNs += (t1 - t0) + (t3 - t2)
        Qwen35MTPHostTrace.attnNs += t2 - t1
        Qwen35MTPHostTrace.mlpNs += t4 - t3
        return out
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's pre-norm hidden state at position t with the embedding of
/// the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        let trace = Qwen35MTPHostTrace.enabled
        let t0 = trace ? Qwen35MTPHostTrace.now() : 0
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        let t1 = trace ? Qwen35MTPHostTrace.now() : 0
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        var fused = fc(concatenated([e, h], axis: -1))
        let t2 = trace ? Qwen35MTPHostTrace.now() : 0

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)
        let t3 = trace ? Qwen35MTPHostTrace.now() : 0

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }
        let t4 = trace ? Qwen35MTPHostTrace.now() : 0

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        let out = norm(fused)
        if trace {
            let t5 = Qwen35MTPHostTrace.now()
            Qwen35MTPHostTrace.embedNs += t1 - t0
            Qwen35MTPHostTrace.fuseNs += t2 - t1
            Qwen35MTPHostTrace.maskNs += t3 - t2
            Qwen35MTPHostTrace.layerNs += t4 - t3
            Qwen35MTPHostTrace.normNs += t5 - t4
        }
        return out
    }

    /// Run one proposal flush while omitting leading-row outputs that have no
    /// consumer. Every supplied row still participates in the fusion stage and
    /// contributes K/V state; only the final row needs a full decoder output.
    /// Multi-layer heads fail closed before mutating cache state.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
              hidden.dim(1) > 1,
              nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

}
