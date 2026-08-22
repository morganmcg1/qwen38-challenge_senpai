import Cmlx
import Foundation
import MLX
import MLXFast
import MLXFastModel
import Testing

/// E134 rung 6 — HYPOTHESIS 179, the `query_transposed` specialisation gate.
///
/// H179 asks whether the SDPA warm in `warmTargetLaterWindowSDPA` compiles a
/// different pipeline from the one the scored path first reaches. The MLX
/// pipeline cache key for the fused vector attention is
/// `kname + mask flag + qt|qnt + c|nc + sinks|nosinks`
/// (`scaled_dot_product_attention.cpp:374-378`), and `query_transposed` is
/// the one boolean that source inspection alone cannot settle.
///
/// This suite reads the boolean instead of arguing about it. It rebuilds
/// three query tensors with the exact expressions the two paths use, then
/// reports MLX's own `row_contiguous` flag for each.
///
/// The decisive detail is at `scaled_dot_product_attention.cpp:686-718`. In
/// vector mode MLX does NOT read the incoming flag directly. It first applies
/// `q_copy_unless`, and it makes a contiguous copy of any query that fails
/// that predicate. `query_transposed` is therefore
/// `!row_contiguous && q_copy_unless(q)`, not `!row_contiguous`.
@Suite("E134 rung 6: the SDPA query_transposed specialisation")
struct E134QueryContiguityTests {
    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"]
        == "1"

    static let batch = 1
    static let queryHeads = 24
    static let headDim = 256
    static let keyLength = 1024

    /// MLX's own flag, read through the C API the dispatcher reads.
    static func rowContiguous(_ array: MLXArray) -> Bool {
        eval(array)
        var flag = false
        _ = _mlx_array_is_row_contiguous(&flag, array.ctx)
        return flag
    }

    static func strides(_ array: MLXArray) -> [Int] {
        eval(array)
        let ndim = Int(mlx_array_ndim(array.ctx))
        guard ndim > 0, let raw = mlx_array_strides(array.ctx) else {
            return []
        }
        return (0 ..< ndim).map { Int(raw[$0]) }
    }

    static func shape(_ array: MLXArray) -> [Int] {
        eval(array)
        let ndim = Int(mlx_array_ndim(array.ctx))
        guard ndim > 0, let raw = mlx_array_shape(array.ctx) else { return [] }
        return (0 ..< ndim).map { Int(raw[$0]) }
    }

    /// `scaled_dot_product_attention.cpp:686-700`, transcribed. True means
    /// MLX uses the query as it stands; false means MLX replaces it with a
    /// contiguous copy before the dispatch.
    static func queryCopyUnless(_ array: MLXArray) -> Bool {
        if rowContiguous(array) { return true }
        let s = strides(array)
        let sh = shape(array)
        guard s.count == 4, sh.count == 4 else { return false }
        if sh[0] == 1 || sh[1] == 1 {
            let bidx = sh[0] == 1 ? 1 : 0
            return s[3] == 1 && s[2] == sh[3] * sh[bidx] && s[bidx] == sh[3]
        }
        return false
    }

    /// The func-const value the dispatcher ends up setting at slot 21.
    static func queryTransposed(_ array: MLXArray) -> Bool {
        !rowContiguous(array) && queryCopyUnless(array)
    }

    /// `warmTargetLaterWindowSDPA`, `Qwen36MTPBlockSession.swift:638-641` and
    /// `:664-667`.
    static func warmQuery(_ qL: Int) -> MLXArray {
        MLXArray.zeros([batch, queryHeads, qL, headDim], dtype: .bfloat16)
    }

    /// `Qwen35Attention.swift:165-193`: rms-normed `[B, L, H, D]`, transposed
    /// to `[B, H, L, D]`, then fused RoPE. The projection and the norm weight
    /// do not change any layout, so a shaped stand-in reproduces the layout
    /// the scored attention block hands to `attentionWithCacheUpdate`.
    static func scoredQuery(_ qL: Int) -> MLXArray {
        let normed = MLXArray.zeros(
            [batch, qL, queryHeads, headDim], dtype: .bfloat16)
        let transposed = normed.transposed(0, 2, 1, 3)
        return MLXFast.RoPE(
            transposed, dimensions: headDim, traditional: false,
            base: 1_000_000, scale: 1, offset: keyLength - qL)
    }

    /// `AttentionUtils.swift:127-140`.
    static func chunkA(_ queries: MLXArray) -> MLXArray {
        queries[0..., 0..., 0 ..< 5, 0...]
    }

    static func chunkB(_ queries: MLXArray) -> MLXArray {
        queries[0..., 0..., 5..., 0...]
    }

    @Test(
        "the warm query and both scored chunks compile the same specialisation",
        .enabled(if: E134QueryContiguityTests.runtimeEnabled))
    func specialisationMatches() {
        var lines: [String] = []
        func record(_ label: String, _ array: MLXArray) -> Bool {
            let qt = Self.queryTransposed(array)
            lines.append(
                "\(label) shape=\(Self.shape(array)) "
                + "strides=\(Self.strides(array)) "
                + "row_contiguous=\(Self.rowContiguous(array)) "
                + "q_copy_unless=\(Self.queryCopyUnless(array)) "
                + "query_transposed=\(qt) "
                + "specialisation=\(qt ? "_qt" : "_qnt")")
            return qt
        }

        var warm: [Int: Bool] = [:]
        var unchunked: [Int: Bool] = [:]
        for qL in 1 ... 9 {
            warm[qL] = record("warm  qL=\(qL)", Self.warmQuery(qL))
            unchunked[qL] = record(
                "scored qL=\(qL)", Self.scoredQuery(qL))
        }
        var chunked: [Int: Bool] = [:]
        for qL in 6 ... 9 {
            let queries = Self.scoredQuery(qL)
            let a = record("chunkA qL=\(qL)", Self.chunkA(queries))
            let b = record("chunkB qL=\(qL)", Self.chunkB(queries))
            chunked[qL] = a || b
        }
        print(lines.joined(separator: "\n"))

        // The gate. Every scored query the fused vector path can see must
        // land on the same specialisation the warm compiled.
        for qL in 1 ... 9 {
            #expect(warm[qL] == false)
            #expect(unchunked[qL] == false)
        }
        for qL in 6 ... 9 {
            #expect(chunked[qL] == false)
        }
    }

    /// Positive control. The predicate must be able to report `_qt`, or the
    /// gate above proves nothing. A `[B, H, L, D]` view of a row-contiguous
    /// `[B, L, H, D]` buffer is the layout `q_copy_unless` accepts without a
    /// copy, which is exactly the case that sets the func const.
    @Test(
        "the predicate reports _qt for a transposed query",
        .enabled(if: E134QueryContiguityTests.runtimeEnabled))
    func predicateCanReportTransposed() {
        let normed = MLXArray.zeros(
            [Self.batch, 8, Self.queryHeads, Self.headDim], dtype: .bfloat16)
        let transposed = normed.transposed(0, 2, 1, 3)
        #expect(Self.rowContiguous(transposed) == false)
        #expect(Self.queryCopyUnless(transposed) == true)
        #expect(Self.queryTransposed(transposed) == true)
    }

    /// Negative control on the other limb. A dim-2 slice of a row-contiguous
    /// `[B, H, L, D]` buffer fails `q_copy_unless`, so MLX copies it and the
    /// func const stays false. This is the limb that decides H179, so it is
    /// pinned separately from the gate.
    @Test(
        "a dim-2 slice fails q_copy_unless and is copied",
        .enabled(if: E134QueryContiguityTests.runtimeEnabled))
    func sliceIsCopiedNotSpecialised() {
        let queries = Self.scoredQuery(8)
        #expect(Self.rowContiguous(queries) == true)
        for chunk in [Self.chunkA(queries), Self.chunkB(queries)] {
            #expect(Self.rowContiguous(chunk) == false)
            #expect(Self.queryCopyUnless(chunk) == false)
            #expect(Self.queryTransposed(chunk) == false)
        }
    }
}
