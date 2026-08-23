import Foundation
import MLX
import Testing

// E160 -- exactness audit of the fused SwiGLU producer.
//
// `qwen35_fused_swiglu_xsums_v1` replaces MLX's compiled `silu(a) * b` on the
// MLP decode path and emits the wide-QMV chunk-sum table of the activation from
// the same pass, so the routed `mlp.down` consumer reads a published table
// instead of launching `qwen35_custom_affine4_g64_xsums_v1`. Two claims carry
// the whole mechanism, and both are claims about bits:
//
//   1. the activation is bit-identical to the compiled `silu(a) * b`, and
//   2. the emitted table is bit-identical to the standalone fill's table over
//      those same activation bytes.
//
// Claim 1 rests on the kernel declaring `Sigmoid` and `Multiply` verbatim from
// `unary_ops.h` and `binary_ops.h` and applying them at `bfloat16_t` in the
// order `compiled.cpp` emits. Claim 2 rests on the epilogue being the fill
// kernel's body over the bytes this pass has just written. Both are source
// arguments, so they are measured here rather than trusted.
//
// The kernels are `private` inside MLXLLM, which is a dependency *product* and
// so is not built with `-enable-testing`. Rather than widen a submitted path
// just to test it, this suite reads the kernel sources verbatim out of
// `Qwen35.swift` at runtime and JIT-compiles them under the same names, input
// names, output names and `ensureRowContiguous` flags. A kernel that stops
// matching the audited text fails the source gate instead of drifting away
// from it silently.

private let qwen35RelativePath = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
private let headerMarker = "private let qwen35FusedSwiGLUHeader = \"\"\""
private let epilogueMarker = "? \"\"\""
private let bodyMarker = "    return \"\"\""
private let generatorMarker = "private func qwen35FusedSwiGLUSource(emitSums: Bool) -> String {"
private let fillNameMarker = "name: \"qwen35_custom_affine4_g64_xsums_v1\""

private enum SourceGateError: Error, CustomStringConvertible {
    case missing(String)

    var description: String {
        switch self {
        case .missing(let what): return "could not locate \(what) in \(qwen35RelativePath)"
        }
    }
}

private func repoRoot(file: StaticString = #filePath) -> URL {
    if let override = ProcessInfo.processInfo.environment["MLXFAST_REPO_ROOT"], !override.isEmpty {
        return URL(fileURLWithPath: override)
    }
    return URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func qwen35Lines() throws -> [String] {
    let url = repoRoot().appendingPathComponent(qwen35RelativePath)
    guard let text = try? String(contentsOf: url, encoding: .utf8) else {
        throw SourceGateError.missing("the model source file")
    }
    return text.components(separatedBy: "\n")
}

/// The value of the Swift multiline literal that opens on `open`, dedented the
/// way the compiler dedents it: by the indentation of the closing delimiter.
private func multilineLiteral(_ lines: [String], openingAt open: Int) throws -> String {
    guard
        let close = lines[(open + 1)...].firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces).hasPrefix("\"\"\"")
        })
    else { throw SourceGateError.missing("closing \"\"\" after line \(open)") }
    let indent = String(lines[close].prefix(while: { $0 == " " }))
    return lines[(open + 1) ..< close]
        .map { $0.hasPrefix(indent) ? String($0.dropFirst(indent.count)) : $0 }
        .joined(separator: "\n")
}

private func index(_ lines: [String], containing needle: String, from: Int = 0) throws -> Int {
    guard let i = lines[from...].firstIndex(where: { $0.contains(needle) }) else {
        throw SourceGateError.missing(needle)
    }
    return i
}

private struct ShippedSwiGLUSource {
    var header: String
    var epilogue: String
    var body: String
    var fillSource: String

    func source(emitSums: Bool) -> String {
        body.replacingOccurrences(
            of: "\\(sumsEpilogue)", with: emitSums ? epilogue : "")
    }
}

private func loadShippedSwiGLUSource() throws -> ShippedSwiGLUSource {
    let lines = try qwen35Lines()
    let header = try multilineLiteral(lines, openingAt: index(lines, containing: headerMarker))

    let generator = try index(lines, containing: generatorMarker)
    let epilogue = try multilineLiteral(
        lines, openingAt: index(lines, containing: epilogueMarker, from: generator))
    let body = try multilineLiteral(
        lines, openingAt: index(lines, containing: bodyMarker, from: generator))

    let fillName = try index(lines, containing: fillNameMarker)
    let fill = try multilineLiteral(
        lines, openingAt: index(lines, containing: "source: \"\"\"", from: fillName))

    guard body.contains("\\(sumsEpilogue)") else {
        throw SourceGateError.missing("the spliced \\(sumsEpilogue) segment")
    }
    return ShippedSwiGLUSource(
        header: header, epilogue: epilogue, body: body, fillSource: fill)
}

@Suite("Fused SwiGLU chunk-sum source gate")
struct QwenFusedSwiGLUSourceGateTests {

    @Test("the audited arithmetic is still the source the worker compiles")
    func theAuditedArithmeticIsStillTheSourceTheWorkerCompiles() throws {
        let s = try loadShippedSwiGLUSource()

        // The operator structs, verbatim from unary_ops.h and binary_ops.h.
        #expect(s.header.contains("auto y = 1 / (1 + metal::exp(metal::abs(x)));"))
        #expect(s.header.contains("return (x < 0) ? y : 1 - y;"))
        #expect(s.header.contains("return x * y;"))

        // The tape order compiled.cpp emits for silu(a) * b, at bfloat16_t.
        let withSums = s.source(emitSums: true)
        #expect(withSums.contains("bfloat16_t tmp_s = Sigmoid()(tmp_a);"))
        #expect(withSums.contains("bfloat16_t tmp_g = Multiply()(tmp_a, tmp_s);"))
        #expect(withSums.contains("ov[i] = Multiply()(tmp_g, bv[i]);"))

        // The epilogue is the fill body: three BF16 adds per group of four,
        // accumulated into a float that starts at zero, in ascending i.
        #expect(withSums.contains("s += xv[0] + xv[1] + xv[2] + xv[3];"))
        #expect(withSums.contains("float s = 0.0f;"))
        #expect(withSums.contains("threadgroup_barrier(mem_flags::mem_device);"))
        #expect(s.fillSource.contains("s += xv[0] + xv[1] + xv[2] + xv[3];"))

        // The shipped no-sums variant splices nothing, so the two kernels
        // differ by the epilogue alone.
        let withoutSums = s.source(emitSums: false)
        #expect(!withoutSums.contains("xsums["))
        #expect(!withoutSums.contains("threadgroup_barrier"))
        #expect(withSums.hasPrefix(withoutSums))
    }
}

@Suite("Fused SwiGLU chunk-sum bit-identity")
struct QwenFusedSwiGLUXSumsExactnessTests {

    static let enabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    /// The scored shape: `intermediateSize` is 17,408, so the gate-up output is
    /// twice that and the activation covers 34 chunk blocks.
    static let half = 17408
    static let widths = [4, 5, 8, 9]

    static func sumsStride(_ m: Int) -> Int { m <= 8 ? 8 : 16 }

    private func gateUp(rows: Int, seed: UInt64) -> MLXArray {
        var state = seed
        // A spread wide enough to cover the sigmoid's saturated tails, its
        // sign branch and the dense region around zero.
        let values = (0 ..< (rows * Self.half * 2)).map { _ -> Float in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float(state >> 40) / Float(1 << 24)
            return (u - 0.5) * 24.0
        }
        return MLXArray(values, [rows, Self.half * 2]).asType(.bfloat16)
    }

    private func compiledActivation(_ y: MLXArray) -> MLXArray {
        let body: @Sendable (MLXArray) -> MLXArray = { yy in
            let h = yy.dim(-1) / 2
            let a = yy[.ellipsis, ..<h]
            return (a * MLX.sigmoid(a)) * yy[.ellipsis, h...]
        }
        return compile(body)(y)
    }

    private func differingCells(_ a: MLXArray, _ b: MLXArray) -> Int {
        notEqual(a.asType(.float32), b.asType(.float32))
            .asType(.int32).sum().item(Int.self)
    }

    @Test(
        "the fused producer matches the compiled SwiGLU and the standalone fill",
        .enabled(if: QwenFusedSwiGLUXSumsExactnessTests.enabled))
    func theFusedProducerMatchesTheCompiledSwiGLUAndTheStandaloneFill() throws {
        let s = try loadShippedSwiGLUSource()
        let fused = MLXFast.metalKernel(
            name: "qwen35_fused_swiglu_xsums_v1",
            inputNames: ["y"],
            outputNames: ["out", "xsums"],
            source: s.source(emitSums: true),
            header: s.header,
            ensureRowContiguous: true)
        let fill = MLXFast.metalKernel(
            name: "qwen35_custom_affine4_g64_xsums_v1",
            inputNames: ["x"],
            outputNames: ["xsums"],
            source: s.fillSource,
            ensureRowContiguous: true)

        let kBlocks = Self.half / 512
        var controlsFired = 0

        for (i, rows) in Self.widths.enumerated() {
            let y = gateUp(rows: rows, seed: 0xE160_2026_0823 &+ UInt64(i))
            let stride = Self.sumsStride(rows)
            let out = fused(
                [y],
                grid: (128, kBlocks, rows),
                threadGroup: (128, 1, 1),
                outputShapes: [[rows, Self.half], [kBlocks * 32 * stride]],
                outputDTypes: [.bfloat16, .float32])
            let activation = out[0]
            let table = out[1].reshaped([kBlocks * 32, stride])[0..., ..<rows]

            let reference = compiledActivation(y)
            let referenceTable = fill(
                [reference],
                grid: (32, kBlocks, rows),
                threadGroup: (32, 1, 1),
                outputShapes: [[kBlocks * 32 * stride]],
                outputDTypes: [.float32])[0]
                .reshaped([kBlocks * 32, stride])[0..., ..<rows]
            eval(activation, table, reference, referenceTable)

            let nans = isNaN(reference.asType(.float32)).asType(.int32).sum()
                .item(Int.self)
            let activationDiff = differingCells(activation, reference)
            let tableDiff = differingCells(table, referenceTable)
            #expect(
                nans == 0,
                "m=\(rows): the reference activation has \(nans) NaN cells, so a bitwise comparison would be meaningless")
            #expect(
                activationDiff == 0,
                "m=\(rows): the fused activation differs from the compiled silu(a) * b in \(activationDiff) cells")
            #expect(
                tableDiff == 0,
                "m=\(rows): the emitted table differs from the standalone fill in \(tableDiff) cells")

            // POSITIVE CONTROL 1. A float32 sum over the same 16 activations
            // is the same value in exact arithmetic and a different value in
            // this one, because the fill adds three BF16 rounds per group of
            // four before it accumulates. A comparison that cannot see that
            // cannot see a re-association either.
            let f32 = reference.asType(.float32)
                .reshaped([rows, kBlocks * 32, 16]).sum(axis: -1)
                .transposed(1, 0)
            eval(f32)
            let reassociated = differingCells(table, f32)
            #expect(
                reassociated > 0,
                "m=\(rows): a float32 re-association produced the same bits as the fill, so this comparison cannot detect one")
            if reassociated > 0 { controlsFired += 1 }

            // POSITIVE CONTROL 2. One perturbed activation element must move
            // exactly one table cell.
            // MLXArray is a reference type, so the copy is explicit. Adding a
            // BF16 zero is exact for every finite value.
            let perturbed = reference + MLXArray(Float(0)).asType(.bfloat16)
            perturbed[0, 0] = MLXArray(reference[0, 0].item(Float.self) + 1.0)
                .asType(.bfloat16)
            let perturbedTable = fill(
                [perturbed],
                grid: (32, kBlocks, rows),
                threadGroup: (32, 1, 1),
                outputShapes: [[kBlocks * 32 * stride]],
                outputDTypes: [.float32])[0]
                .reshaped([kBlocks * 32, stride])[0..., ..<rows]
            eval(perturbedTable)
            let moved = differingCells(table, perturbedTable)
            #expect(
                moved > 0,
                "m=\(rows): perturbing one activation element left every table cell unchanged, so this comparison is blind")
            if moved > 0 { controlsFired += 1 }
        }

        #expect(controlsFired == Self.widths.count * 2)
    }
}
