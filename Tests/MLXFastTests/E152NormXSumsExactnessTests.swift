import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

// E152 A1 -- the fused residual+RMSNorm dispatch also emits the chunk-sum table
// its own consumer would otherwise fill.
//
// 127 of the 257 wide QMV calls of one target forward read an activation that
// `qwen35FusedResidualRMSNorm` has just written: every layer's post-attention
// norm feeds `mlp.gate_up`, and every entry norm except layer 0's feeds
// `gdn.in_proj` or `fa.qkv`. Each of those consumers currently launches
// `qwen35_custom_affine4_g64_xsums_v1` to re-read the same bytes and reduce
// them. The fill is almost all encode overhead, so moving the reduction into a
// dispatch the round already pays should remove 127 boundaries and keep the
// arithmetic.
//
// "Keep the arithmetic" is the whole claim and it is checked here, not
// asserted. Three specific ways this can be wrong:
//
//   HAZARD 1 -- the epilogue re-derives the sum from registers or gathers it
//   across threads instead of re-reading the stored bytes. That is a second
//   reduction for the compiler to order and it will not match the standalone
//   fill.
//
//   HAZARD 2 -- the slot map is derived from the write loop. That loop covers
//   0..4095 with 1024 threads and 4096..5119 with threads 0..255, so a thread
//   that maps "the k-block I last wrote" to "the table entry I own" puts two of
//   the ten k-blocks on the wrong lane.
//
//   HAZARD 3 -- the table is read from the wrong buffer. `h` and `normed` are
//   the same shape and the same dtype and are stored by the same dispatch, so a
//   one-word slip compiles and runs.
//
// MLXLLM is a dependency *package* product, so it is not built with
// `-enable-testing` and `@testable` cannot reach the kernels. Both entry points
// are therefore read verbatim out of `Qwen35.swift` and JIT-compiled under
// their own name, input names, output names and `ensureRowContiguous` flag.
// MLX's custom-kernel cache keys on exactly that tuple, so these are the
// shipped kernels rather than copies of them. The consumer side needs no such
// reconstruction: `Qwen35CustomQMV.xsumsTable`, `matmulWithTable` and `matmul`
// are public.

private let e152QwenRelativePath = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
private let e152BaseKernelMarker = "qwen35FusedResidualRMSNormKernel = MLXFast.metalKernel("
private let e152BaseWrapperMarker = "func qwen35FusedResidualRMSNorm("
private let e152XSumsKernelMarker = "qwen35FusedResidualRMSNormXSumsKernel = MLXFast.metalKernel("
private let e152XSumsWrapperMarker = "func qwen35FusedResidualRMSNormProducing("
private let e152FillKernelMarker = "name: \"qwen35_custom_affine4_g64_xsums_v1\""
private let e152InlineControlMarker = "name: \"qwen35_dual_rms_norm_bf16_v1\""

private enum E152SourceError: Error, CustomStringConvertible {
    case missing(String)

    var description: String {
        switch self {
        case .missing(let what): return "could not locate \(what) in \(e152QwenRelativePath)"
        }
    }
}

private func e152RepoRoot(file: StaticString = #filePath) -> URL {
    if let override = ProcessInfo.processInfo.environment["MLXFAST_REPO_ROOT"], !override.isEmpty {
        return URL(fileURLWithPath: override)
    }
    // <root>/Tests/MLXFastTests/<this file>
    return URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func e152Lines() throws -> [String] {
    let url = e152RepoRoot().appendingPathComponent(e152QwenRelativePath)
    return try String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n")
}

/// The `source:` argument of the declaration at `marker`, as written.
///
/// Returns the raw text after `source:` so a test can tell an inline literal
/// from a named constant. The E152 split is only meaningful if BOTH entry
/// points name the same constant, and that is a property of this text.
private func e152SourceArgument(_ lines: [String], marker: String) throws -> String {
    guard let declIdx = lines.firstIndex(where: { $0.contains(marker) }) else {
        throw E152SourceError.missing(marker)
    }
    guard
        let idx = lines[declIdx ..< min(declIdx + 10, lines.count)].firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces).hasPrefix("source:")
        })
    else { throw E152SourceError.missing("source: for \(marker)") }
    return lines[idx].trimmingCharacters(in: .whitespaces)
        .dropFirst("source:".count)
        .trimmingCharacters(in: CharacterSet(charactersIn: " ,"))
}

/// The body of the first `"""` literal at or after `marker`.
///
/// The opening delimiter is searched for forwards from the marker rather than
/// assumed to be on it, because `name:` sits several argument lines above
/// `source: """`.
private func e152LiteralBody(_ lines: [String], marker: String) throws -> [String] {
    guard let start = lines.firstIndex(where: { $0.contains(marker) }) else {
        throw E152SourceError.missing(marker)
    }
    guard
        let open = lines[start...].firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces).hasSuffix("\"\"\"")
        })
    else { throw E152SourceError.missing("opening \"\"\" after \(marker)") }
    guard
        let close = lines[(open + 1)...].firstIndex(where: {
            let t = $0.trimmingCharacters(in: .whitespaces)
            return t == "\"\"\"" || t == "\"\"\","
        })
    else { throw E152SourceError.missing("closing \"\"\" after \(marker)") }
    return Array(lines[(open + 1) ..< close])
}

private func e152NamedBody(_ lines: [String], _ ident: String) throws -> [String] {
    let decl = "let \(ident) = \"\"\""
    guard let open = lines.firstIndex(where: { $0.contains(decl) }) else {
        throw E152SourceError.missing(decl)
    }
    guard
        let close = lines[(open + 1)...].firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces) == "\"\"\""
        })
    else { throw E152SourceError.missing("closing \"\"\" for \(ident)") }
    return Array(lines[(open + 1) ..< close])
}

/// The value expression of a chunk-sum fill, trimmed of indentation.
///
/// From `float s = 0.0f;` to the store, inclusive. This is the text that must
/// be character-identical between the standalone fill and the epilogue: same
/// loads, same three BF16 adds per group of four, same ascending float
/// accumulation, same store offset. Anything the compiler is free to reorder
/// differently is a different reduction.
private func e152ReductionCore(_ body: [String]) -> [String] {
    let trimmed = body.map { $0.trimmingCharacters(in: .whitespaces) }
    guard let start = trimmed.firstIndex(of: "float s = 0.0f;") else { return [] }
    guard let end = trimmed[start...].firstIndex(where: { $0.hasSuffix("] = s;") }) else {
        return []
    }
    return Array(trimmed[start ... end])
}

// MARK: - Source gates (no GPU)

@Suite("E152 producing-norm source gate")
struct E152NormXSumsSourceTests {

    @Test("both norm entry points name one shared source constant")
    func bothNormEntryPointsNameOneSharedSourceConstant() throws {
        let lines = try e152Lines()
        let base = try e152SourceArgument(lines, marker: e152BaseKernelMarker)
        let xsums = try e152SourceArgument(lines, marker: e152XSumsKernelMarker)

        #expect(base == "qwen35FusedResidualRMSNormSource")
        #expect(
            xsums == "qwen35FusedResidualRMSNormSource + qwen35FusedResidualRMSNormXSumsEpilogue",
            "the three-output kernel must reuse the shared body, not a copy of it")

        // Positive control: the reader can tell a named constant from an inline
        // literal, so the two assertions above are not vacuous.
        let inline = try e152SourceArgument(lines, marker: e152InlineControlMarker)
        #expect(inline == "\"\"\"", "expected an inline literal at the control kernel")
    }

    @Test("the reconstructed three-output source extends the two-output source exactly")
    func theReconstructedThreeOutputSourceExtendsTheTwoOutputSourceExactly() throws {
        let base = try loadFusedKernel(
            marker: e152BaseKernelMarker, wrapperMarker: e152BaseWrapperMarker)
        let xsums = try loadFusedKernel(
            marker: e152XSumsKernelMarker, wrapperMarker: e152XSumsWrapperMarker)

        #expect(base.name == "qwen35_fused_residual_rms_norm")
        #expect(xsums.name == "qwen35_fused_residual_rms_norm_xsums_v1")
        #expect(base.outputNames == ["h", "normed"])
        #expect(xsums.outputNames == ["h", "normed", "xsums"])
        #expect(base.inputNames == xsums.inputNames)
        #expect(base.ensureRowContiguous == false)
        #expect(xsums.ensureRowContiguous == false)
        #expect(xsums.declaredLsize == base.declaredLsize)
        #expect(xsums.declaredNReads == base.declaredNReads)
        #expect(xsums.wrapperThreadGroup == base.wrapperThreadGroup)
        #expect(xsums.wrapperGridMultiplier == base.wrapperGridMultiplier)

        #expect(
            xsums.source.hasPrefix(base.source),
            "the epilogue must be appended to the shipped body, not spliced into it")
        #expect(
            !base.source.contains("xsums"),
            "the two-output dispatch must have no table code in it at all")
    }

    @Test("the epilogue reduction is character-identical to the standalone fill")
    func theEpilogueReductionIsCharacterIdenticalToTheStandaloneFill() throws {
        let lines = try e152Lines()
        let fill = e152ReductionCore(try e152LiteralBody(lines, marker: e152FillKernelMarker))
        let epilogue = e152ReductionCore(
            try e152NamedBody(lines, "qwen35FusedResidualRMSNormXSumsEpilogue"))

        #expect(fill.count >= 6, "did not find the fill's value expression")
        #expect(fill == epilogue, "HAZARD 1: the two reductions have drifted apart")

        // Positive control: a reassociation the compiler would be free to make
        // if the text allowed it must be visible to this comparison.
        let reassociated = fill.map {
            $0.replacingOccurrences(
                of: "s += xv[0] + xv[1] + xv[2] + xv[3];",
                with: "s += (xv[0] + xv[1]) + (xv[2] + xv[3]);")
        }
        #expect(reassociated != fill, "the control did not change the text it targets")
        #expect(reassociated != epilogue)
    }

    @Test("the epilogue reduces the stored normed bytes and nothing else")
    func theEpilogueReducesTheStoredNormedBytesAndNothingElse() throws {
        let lines = try e152Lines()
        let epilogue = try e152NamedBody(lines, "qwen35FusedResidualRMSNormXSumsEpilogue")
        let pointer = epilogue.filter { $0.contains("bfloat16_t* xm =") }

        #expect(pointer.count == 1, "expected exactly one pointer setup in the epilogue")
        let base = epilogue.first { $0.contains("normed +") }
        #expect(base != nil, "HAZARD 3: the epilogue must read `normed`")
        #expect(
            !epilogue.contains(where: { $0.contains("h +") || $0.contains("h[") }),
            "HAZARD 3: the epilogue must not touch the residual output")

        // HAZARD 2: the slot map must come from the thread id over the whole
        // row, never from the write loop's own index.
        #expect(epilogue.contains(where: { $0.contains("threadgroup_barrier(mem_flags::mem_device)") }))
        #expect(epilogue.contains(where: { $0.contains("const int xs_lane = int(thread_id) % 32;") }))
        #expect(epilogue.contains(where: { $0.contains("const int xs_kb = int(thread_id) / 32;") }))
    }
}

// MARK: - GPU exactness

private struct E152Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray
}

private struct E152SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@Suite("E152 producing-norm exactness")
struct E152NormXSumsExactnessTests {

    static let enabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    static let k = 5120
    static let eps: Float = 1e-6
    static let groupSize = 64
    static let bits = 4

    static var kBlocks: Int { k / 512 }

    // MARK: fixtures

    private static func activations(m: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(seed)
        let x = MLXRandom.normal([m, k]).asType(.bfloat16)
        let r = MLXRandom.normal([m, k]).asType(.bfloat16)
        let w = (1.0 + 0.25 * MLXRandom.normal([k])).asType(.bfloat16)
        eval(x, r, w)
        return (x, r, w)
    }

    /// Full 32-bit packed nibbles, as in E120: `MLXRandom.randInt` tops out
    /// below 2^31 and would leave the high nibble permanently small.
    private static func weights(n: Int, seed: UInt64) -> E152Weights {
        var rng = E152SplitMix64(state: seed)
        var raw = [UInt32]()
        raw.reserveCapacity(n * k / 8)
        for _ in 0 ..< (n * k / 8) { raw.append(UInt32(truncatingIfNeeded: rng.next())) }
        let packed = MLXArray(raw).reshaped([n, k / 8])
        MLXRandom.seed(seed)
        let scales = MLXRandom.uniform(low: Float(0.004), high: Float(0.02), [n, k / groupSize])
            .asType(.bfloat16)
        let biases = MLXRandom.uniform(low: Float(-0.06), high: Float(0.06), [n, k / groupSize])
            .asType(.bfloat16)
        eval(packed, scales, biases)
        return E152Weights(packed: packed, scales: scales, biases: biases)
    }

    // MARK: kernel plumbing

    private static func runNorm(
        _ spec: ShippedFusedKernel, source: String? = nil, name: String? = nil,
        x: MLXArray, r: MLXArray, w: MLXArray, m: Int
    ) -> [MLXArray] {
        let kernel = MLXFast.metalKernel(
            name: name ?? spec.name,
            inputNames: spec.inputNames,
            outputNames: spec.outputNames,
            source: source ?? spec.source,
            ensureRowContiguous: spec.ensureRowContiguous)
        var shapes: [[Int]] = [x.shape, x.shape]
        var dtypes: [DType] = [.bfloat16, .bfloat16]
        if spec.outputNames.count == 3 {
            shapes.append([Self.kBlocks * 32 * Qwen35CustomQMV.sumsStride(m)])
            dtypes.append(.float32)
        }
        return kernel(
            [x, r, w, MLXArray(Self.eps)],
            grid: (m * spec.wrapperThreadGroup, 1, 1),
            threadGroup: (spec.wrapperThreadGroup, 1, 1),
            outputShapes: shapes,
            outputDTypes: dtypes)
    }

    /// The `[kBlocks, 32, m]` entries a table actually defines.
    ///
    /// A lane's stride is padded to 8 floats (16 at M = 9) so its entries stay
    /// in one cache line. Only the first `m` of each slot are ever written, so
    /// comparing the whole buffer compares uninitialised device memory.
    private static func defined(_ table: MLXArray, m: Int) -> MLXArray {
        table
            .reshaped([Self.kBlocks, 32, Qwen35CustomQMV.sumsStride(m)])[0..., 0..., 0 ..< m]
    }

    private static func differing(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    // MARK: tests

    @Test(
        "the produced table equals the standalone fill over normed, at every width",
        .enabled(if: E152NormXSumsExactnessTests.enabled))
    func theProducedTableEqualsTheStandaloneFillOverNormed() throws {
        let base = try loadFusedKernel(
            marker: e152BaseKernelMarker, wrapperMarker: e152BaseWrapperMarker)
        let xsums = try loadFusedKernel(
            marker: e152XSumsKernelMarker, wrapperMarker: e152XSumsWrapperMarker)

        for m in Qwen35CustomQMV.widths {
            let (x, r, w) = Self.activations(m: m, seed: 0xE152_0001 &+ UInt64(m))
            let two = Self.runNorm(base, x: x, r: r, w: w, m: m)
            let three = Self.runNorm(xsums, x: x, r: r, w: w, m: m)
            let standalone = Qwen35CustomQMV.xsumsTable(three[1])
            let wrongBuffer = Qwen35CustomQMV.xsumsTable(three[0])
            eval(two[0], two[1], three[0], three[1], three[2], standalone, wrongBuffer)

            #expect(
                Self.differing(three[0], two[0]) == 0,
                "m=\(m): the third output changed the residual `h`")
            #expect(
                Self.differing(three[1], two[1]) == 0,
                "m=\(m): the third output changed `normed`")
            #expect(
                Self.differing(Self.defined(three[2], m: m), Self.defined(standalone, m: m)) == 0,
                "m=\(m): HAZARD 1/2: the produced table is not the standalone fill")

            // HAZARD 3 control. `h` and `normed` are the same shape, the same
            // dtype and are stored by the same dispatch, so the comparison
            // above has to be able to see the wrong one.
            #expect(
                Self.differing(Self.defined(three[2], m: m), Self.defined(wrongBuffer, m: m)) > 0,
                "m=\(m): the control could not distinguish `h` from `normed`")
        }
    }

    @Test(
        "folding the second-pass k-blocks onto the first pass is visible",
        .enabled(if: E152NormXSumsExactnessTests.enabled))
    func foldingTheSecondPassKBlocksOntoTheFirstPassIsVisible() throws {
        let xsums = try loadFusedKernel(
            marker: e152XSumsKernelMarker, wrapperMarker: e152XSumsWrapperMarker)
        let m = 6
        let (x, r, w) = Self.activations(m: m, seed: 0xE152_0002)

        // The write loop covers 0..4095 with 1024 threads and 4096..5119 with
        // threads 0..255, so k-blocks 8 and 9 are exactly the entries a
        // pass-derived slot map would misplace. `% 8` folds them onto k-blocks
        // 0 and 1 -- a deterministic wrong answer in precisely that region.
        let folded = xsums.source.replacingOccurrences(
            of: "const int xs_kb = int(thread_id) / 32;",
            with: "const int xs_kb = (int(thread_id) / 32) % 8;")
        #expect(folded != xsums.source, "the control did not change the slot map")

        let good = Self.runNorm(xsums, x: x, r: r, w: w, m: m)
        let bad = Self.runNorm(
            xsums, source: folded, name: "qwen35_e152_slotmap_fold_control",
            x: x, r: r, w: w, m: m)
        eval(good[2], bad[2])

        let g = Self.defined(good[2], m: m)
        let b = Self.defined(bad[2], m: m)
        for kb in 0 ..< Self.kBlocks {
            let moved = Self.differing(g[kb], b[kb])
            if kb < 8 {
                #expect(moved == 0, "k-block \(kb) is outside the fold and must not move")
            } else {
                #expect(moved > 0, "k-block \(kb) is the two-pass region and must move")
            }
        }
    }

    @Test(
        "the consumer reads the produced table and is bit-identical to filling its own",
        .enabled(if: E152NormXSumsExactnessTests.enabled))
    func theConsumerReadsTheProducedTableAndIsBitIdenticalToFillingItsOwn() throws {
        let xsums = try loadFusedKernel(
            marker: e152XSumsKernelMarker, wrapperMarker: e152XSumsWrapperMarker)
        let m = 6
        let n = 4096
        let (x, r, w) = Self.activations(m: m, seed: 0xE152_0003)
        let out = Self.runNorm(xsums, x: x, r: r, w: w, m: m)
        let a = out[1]
        let table = out[2]
        let weights = Self.weights(n: n, seed: 0xE152_0003)
        eval(a, table)

        func consume(_ t: MLXArray?) throws -> MLXArray {
            let y = Qwen35CustomQMV.matmul(
                a, weights.packed, scales: weights.scales, biases: weights.biases,
                groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                xsums: t, arm: .sumTable)
            let unwrapped = try #require(y, "the cell must route at m=\(m), k=\(Self.k), n=\(n)")
            eval(unwrapped)
            return unwrapped
        }

        let filled = try consume(nil)
        let produced = try consume(table)
        let wrongLength = try consume(MLXArray.zeros([8], dtype: .float32))
        let wrongType = try consume(table.asType(.float16))

        #expect(
            Self.differing(produced, filled) == 0,
            "the produced table must give the incumbent answer exactly")
        #expect(
            Self.differing(wrongLength, filled) == 0,
            "a table of the wrong length must fall back to the fill path")
        #expect(
            Self.differing(wrongType, filled) == 0,
            "a table of the wrong dtype must fall back to the fill path")

        // Without this the three comparisons above would also pass if the
        // accepted table were quietly discarded. Entry 0 is (k-block 0, lane 0,
        // row 0), which every width reads, and the step is E120's 1024 rather
        // than one ulp: the entry is one of ten k-block terms and the output is
        // bf16, so a float32 ulp of the table rounds away and would make this
        // control fail for a reason that is not a defect.
        var oneHot = [Float](repeating: 0, count: table.size)
        oneHot[0] = 1024
        let poisoned = try consume(table + MLXArray(oneHot))
        #expect(
            Self.differing(poisoned, filled) > 0,
            "an accepted table that does not reach the kernel is not a fusion")
    }
}
