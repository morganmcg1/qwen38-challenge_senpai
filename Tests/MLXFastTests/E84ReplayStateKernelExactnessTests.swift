import Foundation
import MLX
import MLXFast
import Testing

// E84 mechanism B -- exactness gate for the state-only Gated DeltaNet replay
// kernel.
//
// `Qwen35GatedDeltaNet.replayPrefix` reads only the boundary recurrent state,
// yet the vendored `gated_delta_step` kernel it called also produced the full
// `[1, T, Hv, Dv]` output tensor. `qwen35_gated_delta_replay_state` is that
// kernel with the `q` pointer, the `out` accumulator, its `simd_sum`, the `y`
// store and the mask branch removed. Everything that touches `state` is copied
// verbatim.
//
// The claim under test is that the surviving state recurrence is BIT-identical.
// It is a claim about fp32 register arithmetic under a different Metal
// compilation unit, so it is measured, not asserted. Both kernels are read out
// of the shipped Swift sources at runtime and JIT-compiled under their shipped
// names, input names and output names, so this suite cannot silently drift from
// what the worker runs.

private let gatedDeltaRelativePath =
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift"
private let qwen35RelativePath =
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

private let batchSize = 1
private let kHeads = 16
private let vHeads = 48
private let kHeadDim = 128
private let vHeadDim = 128

private enum SourceGateError: Error, CustomStringConvertible {
    case missing(String, String)

    var description: String {
        switch self {
        case .missing(let what, let file): return "could not locate \(what) in \(file)"
        }
    }
}

private struct ShippedKernel {
    var name: String
    var inputNames: [String]
    var outputNames: [String]
    var source: String
}

private func repoRoot(file: StaticString = #filePath) -> URL {
    if let override = ProcessInfo.processInfo.environment["MLXFAST_REPO_ROOT"],
       !override.isEmpty
    {
        return URL(fileURLWithPath: override)
    }
    return URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func quotedStrings(in line: String) -> [String] {
    var out: [String] = []
    var current: String?
    for ch in line {
        if ch == "\"" {
            if let c = current {
                out.append(c)
                current = nil
            } else {
                current = ""
            }
        } else if current != nil {
            current!.append(ch)
        }
    }
    return out
}

private func loadShippedKernel(
    relativePath: String, declarationMarker: String
) throws -> ShippedKernel {
    let url = repoRoot().appendingPathComponent(relativePath)
    let lines = try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: "\n")

    guard let declIdx = lines.firstIndex(where: { $0.contains(declarationMarker) })
    else { throw SourceGateError.missing(declarationMarker, relativePath) }
    guard
        let sourceOpen = lines[declIdx...].firstIndex(where: {
            let t = $0.trimmingCharacters(in: .whitespaces)
            return t.hasPrefix("let source = ") && t.hasSuffix("\"\"\"")
        })
    else { throw SourceGateError.missing("let source = \"\"\"", relativePath) }
    guard
        let sourceClose = lines[(sourceOpen + 1)...].firstIndex(where: {
            let t = $0.trimmingCharacters(in: .whitespaces)
            return t == "\"\"\"" || t == "\"\"\","
        })
    else { throw SourceGateError.missing("closing \"\"\"", relativePath) }

    let tail = lines[sourceClose ..< Swift.min(sourceClose + 12, lines.count)]
    guard let nameLine = tail.first(where: { $0.contains("name:") }),
          let name = quotedStrings(in: nameLine).first
    else { throw SourceGateError.missing("name:", relativePath) }
    guard let inLine = tail.first(where: { $0.contains("inputNames:") })
    else { throw SourceGateError.missing("inputNames:", relativePath) }
    guard let outLine = tail.first(where: { $0.contains("outputNames:") })
    else { throw SourceGateError.missing("outputNames:", relativePath) }

    return ShippedKernel(
        name: name,
        inputNames: quotedStrings(in: inLine),
        outputNames: quotedStrings(in: outLine),
        source: lines[(sourceOpen + 1) ..< sourceClose].joined(separator: "\n"))
}

/// The vendored kernel with `hasMask: false`, exactly as
/// `makeGatedDeltaKernel` builds it.
private func loadVendoredGatedDeltaKernel() throws -> ShippedKernel {
    var kernel = try loadShippedKernel(
        relativePath: gatedDeltaRelativePath,
        declarationMarker: "private func makeGatedDeltaKernel(hasMask: Bool)")
    kernel.source = kernel.source.replacingOccurrences(
        of: "\\(maskSource)", with: "true")
    // `suffix` is empty for the unmasked arm.
    kernel.name = kernel.name.replacingOccurrences(of: "\\(suffix)", with: "")
    kernel.inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    return kernel
}

private func loadReplayStateKernel() throws -> ShippedKernel {
    try loadShippedKernel(
        relativePath: qwen35RelativePath,
        declarationMarker:
            "private let qwen35GatedDeltaReplayStateKernel: MLXFast.MLXFastKernel?")
}

private func squeezed(_ text: String) -> String {
    text.components(separatedBy: .whitespacesAndNewlines).joined()
}

/// Do `needles` appear in `haystack` in this order, without overlap?
private func appearInOrder(_ needles: [String], in haystack: String) -> Bool {
    var cursor = haystack.startIndex
    for needle in needles {
        guard let found = haystack.range(of: needle, range: cursor ..< haystack.endIndex)
        else { return false }
        cursor = found.upperBound
    }
    return true
}

private struct SplitMix {
    private var state: UInt64
    init(_ seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func normal() -> Float {
        let u = Swift.max(unit(), 1e-12)
        let v = unit()
        return Float(
            (-2.0 * Foundation.log(u)).squareRoot()
                * Foundation.cos(2.0 * Double.pi * v))
    }

    mutating func floats(_ count: Int, _ body: (inout SplitMix) -> Float) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0 ..< count { out[i] = body(&self) }
        return out
    }
}

private struct ReplayInputs {
    var q: MLXArray
    var k: MLXArray
    var v: MLXArray
    var g: MLXArray
    var beta: MLXArray
    var state: MLXArray
}

/// Inputs in the ranges the live recurrence produces: `g = exp(-x)` and
/// `beta = sigmoid(x)` are both in (0, 1), q/k are RMS-normed and scaled, and
/// the carried state is fp32.
private func makeReplayInputs(seed: UInt64, rows T: Int) -> ReplayInputs {
    var rng = SplitMix(seed)
    let qkCount = batchSize * T * kHeads * kHeadDim
    let vCount = batchSize * T * vHeads * vHeadDim
    let gCount = batchSize * T * vHeads
    let stateCount = batchSize * vHeads * vHeadDim * kHeadDim

    let q = MLXArray(
        rng.floats(qkCount) { $0.normal() * 0.09 },
        [batchSize, T, kHeads, kHeadDim]).asType(.bfloat16)
    let k = MLXArray(
        rng.floats(qkCount) { $0.normal() * 0.09 },
        [batchSize, T, kHeads, kHeadDim]).asType(.bfloat16)
    let v = MLXArray(
        rng.floats(vCount) { $0.normal() },
        [batchSize, T, vHeads, vHeadDim]).asType(.bfloat16)
    let g = MLXArray(
        rng.floats(gCount) { Float(0.90 + 0.099 * $0.unit()) },
        [batchSize, T, vHeads])
    let beta = MLXArray(
        rng.floats(gCount) { Float($0.unit()) },
        [batchSize, T, vHeads])
    let state = MLXArray(
        rng.floats(stateCount) { $0.normal() * 0.3 },
        [batchSize, vHeads, vHeadDim, kHeadDim])
    return ReplayInputs(q: q, k: k, v: v, g: g, beta: beta, state: state)
}

private func vendoredBoundaryState(
    _ kernel: MLXFast.MLXFastKernel, _ inputs: ReplayInputs, rows T: Int
) -> MLXArray {
    kernel(
        [inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, inputs.state,
         MLXArray(T)],
        template: [
            ("InT", DType.bfloat16), ("StT", DType.float32),
            ("Dk", kHeadDim), ("Dv", vHeadDim),
            ("Hk", kHeads), ("Hv", vHeads),
        ],
        grid: (32, vHeadDim, batchSize * vHeads),
        threadGroup: (32, 4, 1),
        outputShapes: [[batchSize, T, vHeads, vHeadDim], inputs.state.shape],
        outputDTypes: [DType.bfloat16, DType.float32]
    )[1]
}

private func cloneBoundaryState(
    _ kernel: MLXFast.MLXFastKernel, _ inputs: ReplayInputs, rows T: Int
) -> MLXArray {
    kernel(
        [inputs.k, inputs.v, inputs.g, inputs.beta, inputs.state, MLXArray(T)],
        template: [
            ("StT", DType.float32),
            ("Dk", kHeadDim), ("Dv", vHeadDim),
            ("Hk", kHeads), ("Hv", vHeads),
        ],
        grid: (32, vHeadDim, batchSize * vHeads),
        threadGroup: (32, 4, 1),
        outputShapes: [inputs.state.shape],
        outputDTypes: [DType.float32]
    )[0]
}

private func bitPatterns(_ array: MLXArray) -> [UInt32] {
    array.asArray(Float.self).map { $0.bitPattern }
}

@Suite(.serialized)
struct E84ReplayStateKernelExactnessTests {

    /// The five state statements and their order are the whole exactness
    /// argument. Check them against both shipped sources.
    @Test
    func theCloneKeepsEveryStateStatementInOrder() throws {
        let vendored = try loadVendoredGatedDeltaKernel()
        let clone = try loadReplayStateKernel()

        #expect(vendored.name == "gated_delta_step")
        #expect(vendored.outputNames == ["y", "state_out"])
        #expect(clone.name == "qwen35_gated_delta_replay_state")
        #expect(clone.inputNames == ["k", "v", "g", "beta", "state_in", "T"])
        #expect(clone.outputNames == ["state_out"])
        #expect(
            !clone.inputNames.contains("q"),
            "the q pointer exists only to feed the removed output")

        let statements = [
            "state[i]=static_cast<float>(i_state[s_idx]);",
            "state[i]=state[i]*g_[hv_idx];",
            "kv_mem+=state[i]*k_[s_idx];",
            "kv_mem=simd_sum(kv_mem);",
            "autodelta=(v_[dv_idx]-kv_mem)*beta_[hv_idx];",
            "state[i]=state[i]+k_[s_idx]*delta;",
            "o_state[s_idx]=static_cast<StT>(state[i]);",
        ]
        let cloneText = squeezed(clone.source)
        let vendoredText = squeezed(vendored.source)
        #expect(appearInOrder(statements, in: vendoredText))
        #expect(appearInOrder(statements, in: cloneText))

        // What must be gone.
        #expect(!cloneText.contains("out+=state[i]*q_[s_idx];"))
        #expect(!cloneText.contains("out=simd_sum(out);"))
        #expect(!cloneText.contains("y[dv_idx]"))
        #expect(!cloneText.contains("q_+=Hk*Dk;"))
        #expect(!cloneText.contains("InT"), "InT would be an unused template parameter")

        // What must stay, because the pointer walk feeds the state chain.
        for advance in ["k_+=Hk*Dk;", "v_+=Hv*Dv;", "g_+=Hv;", "beta_+=Hv;"] {
            #expect(cloneText.contains(advance), "missing pointer advance \(advance)")
            #expect(vendoredText.contains(advance))
        }
    }

    /// Bit-identity of the boundary state at every replay width the tape can
    /// produce, with a positive control at each width.
    @Test
    func boundaryStateIsBitIdenticalAtEveryReplayWidth() throws {
        let vendoredSource = try loadVendoredGatedDeltaKernel()
        let cloneSource = try loadReplayStateKernel()
        let vendored = MLXFast.metalKernel(
            name: vendoredSource.name,
            inputNames: vendoredSource.inputNames,
            outputNames: vendoredSource.outputNames,
            source: vendoredSource.source)
        let clone = MLXFast.metalKernel(
            name: cloneSource.name,
            inputNames: cloneSource.inputNames,
            outputNames: cloneSource.outputNames,
            source: cloneSource.source)

        var totalCells = 0
        for T in 1 ... 8 {
            let inputs = makeReplayInputs(
                seed: 0x5EA1_74C5_0E84_1000 &+ UInt64(T), rows: T)
            let reference = vendoredBoundaryState(vendored, inputs, rows: T)
            let candidate = cloneBoundaryState(clone, inputs, rows: T)
            eval(reference, candidate)

            let referenceBits = bitPatterns(reference)
            let candidateBits = bitPatterns(candidate)
            #expect(referenceBits.count == candidateBits.count)
            let mismatches = zip(referenceBits, candidateBits).filter { $0 != $1 }.count
            #expect(
                mismatches == 0,
                "T = \(T): \(mismatches) of \(referenceBits.count) state values moved")
            totalCells += referenceBits.count

            // POSITIVE CONTROL. Advance one `v` element by one bf16 step at the
            // first timestep. Failure mode: the delta it feeds propagates
            // through the rank-1 state update, so the compared state MUST move.
            var perturbed = inputs
            perturbed.v = MLXArray(
                inputs.v.asType(.float32).asArray(Float.self).enumerated().map {
                    $0.offset == 0 ? $0.element + 1.0 : $0.element
                },
                inputs.v.shape).asType(.bfloat16)
            let control = cloneBoundaryState(clone, perturbed, rows: T)
            eval(control)
            let controlMismatches = zip(referenceBits, bitPatterns(control))
                .filter { $0 != $1 }.count
            #expect(
                controlMismatches > 0,
                "T = \(T): positive control did not move the state")
        }
        #expect(totalCells == 8 * batchSize * vHeads * vHeadDim * kHeadDim)
    }
}
