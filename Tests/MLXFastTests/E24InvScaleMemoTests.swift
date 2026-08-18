import Foundation
import MLX
import Testing

// E24 Phase 2 — research-only. Never submitted.
//
// `Qwen35GatedDeltaNet.invScalePair` replaces four inline
// `MLXArray(<float>).asType(dtype)` sites with one per-layer, dtype-keyed memo.
// The speed argument is Phase 1's; this file guards the correctness argument,
// which has exactly three failure modes:
//
//   R1  the memoized constant differs from the value the inline expression
//       produced, so the model arithmetic silently changes;
//   R2  reusing one `MLXArray` as a kernel input many times perturbs it -- the
//       real novel hazard, since MLX may donate an input's buffer to its
//       output when that input is uniquely referenced;
//   R3  the dtype-keyed lookup serves a constant of the wrong dtype.
//
// These are properties of the memo pattern and of MLX's reuse semantics, which
// is what is actually exercised here -- no mocks, real arrays and real kernels.
// `invScalePair` is `private`, so this file cannot call it directly and does
// not prove the four call sites were rewired correctly. That is proved
// end-to-end instead, by the Phase 3 correctness run's `all_tokens_matched`
// and `residual_divergence_count == 0` against the pinned reference.
@Suite
struct E24InvScaleMemoTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// `weights/config.json: linear_key_head_dim`. Note `pow(invScale, 2)` is
    /// Float arithmetic and is *not* exactly 1/128, so the memo has to
    /// reproduce the inline expression rather than a tidied-up literal.
    private static let headKDim = 128
    private static let dtypes: [DType] = [.bfloat16, .float16, .float32]

    private static func inlinePair(_ dtype: DType) -> (sq: MLXArray, lin: MLXArray) {
        let invScale = pow(Float(headKDim), -0.5)
        return (
            sq: MLXArray(pow(invScale, 2)).asType(dtype),
            lin: MLXArray(invScale).asType(dtype)
        )
    }

    private static func bits(_ a: MLXArray) -> Data {
        eval(a)
        return a.asData()
    }

    private static func hex(_ d: Data) -> String {
        d.map { String(format: "%02x", $0) }.joined()
    }

    // R1 + R3.
    @Test(.enabled(if: E24InvScaleMemoTests.enabled))
    func memoizedConstantsAreBitIdenticalToTheInlineExpression() throws {
        var seen: [String: DType] = [:]
        for dtype in Self.dtypes {
            // The memo evaluates the expression once and hands the same array
            // back forever; the inline site evaluated it on every call.
            let memo = Self.inlinePair(dtype)
            for call in 0 ..< 8 {
                let fresh = Self.inlinePair(dtype)
                #expect(
                    Self.bits(memo.sq) == Self.bits(fresh.sq),
                    "sq constant drifted from the inline expression on call \(call) for \(dtype)"
                )
                #expect(
                    Self.bits(memo.lin) == Self.bits(fresh.lin),
                    "lin constant drifted from the inline expression on call \(call) for \(dtype)"
                )
            }
            #expect(memo.sq.dtype == dtype)
            #expect(memo.lin.dtype == dtype)

            // R3: a dict that silently served one dtype for another would pass
            // every check above, so require the encodings to be distinguishable.
            let key = Self.hex(Self.bits(memo.lin))
            #expect(
                seen[key] == nil,
                "\(dtype) encodes `lin` identically to \(String(describing: seen[key])); a mis-keyed lookup would be undetectable"
            )
            seen[key] = dtype
        }
    }

    // R2: the hazard that only exists because the array is now long-lived.
    @Test(.enabled(if: E24InvScaleMemoTests.enabled))
    func reusingOneConstantAcrossManyKernelsNeverPerturbsIt() throws {
        for dtype in Self.dtypes {
            let memo = Self.inlinePair(dtype)
            let before = Self.bits(memo.sq)

            // 48 GDN layers x many rounds; 64 reuses is far past the point
            // where a donated buffer would show up.
            for round in 0 ..< 64 {
                let x = MLXArray(
                    (0 ..< 32).map { Float($0) + Float(round) }
                ).asType(dtype)

                let viaMemo = memo.sq * x + memo.lin
                let viaInline = { () -> MLXArray in
                    let fresh = Self.inlinePair(dtype)
                    return fresh.sq * x + fresh.lin
                }()
                #expect(
                    Self.bits(viaMemo) == Self.bits(viaInline),
                    "round \(round) for \(dtype): memoized constants produced a different result than recomputing them"
                )
                #expect(
                    Self.bits(memo.sq) == before,
                    "round \(round) for \(dtype): the memoized constant's own buffer changed after being used as a kernel input"
                )
            }
        }
    }
}
