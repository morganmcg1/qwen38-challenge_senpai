import Foundation
import MLX
import Testing

// E28 / PR #33 -- exactness audit of the fused residual+RMSNorm kernel we
// inherited at the frontier.
//
// `qwen35FusedResidualRMSNormKernel` collapses the previous layer's exit add
// and this layer's entry RMSNorm into one launch (63 launches removed per
// forward). It fires for every interior layer whenever
// `hiddenStates.dtype == .bfloat16 && hiddenStates.dim(-1) == 5120`, i.e. the
// entire scored decode path. Its in-code claim is strong:
//
//     "the values are bit-identical to the sequential
//      `x = prevH + prevMLP; inputLayerNorm(x)` chain"
//
// Bit-identity is a claim about FP32 reduction order, so it is checked here
// rather than trusted. Source inspection of the kernel against MLX's
// `rms_looped` (`mlx/backend/metal/kernels/rms_norm.metal`) shows the element
// index map, the guard, `n_reads = 4`, the simd reduction tree, the
// `metal::precise::rsqrt` call and the output expression all agree. Exactly one
// parameter is not fixed by the source: MLX passes `lsize` in as
// `[[threads_per_threadgroup]]`, and for `axis_size > RMS_LOOPED_LIMIT` (4096;
// ours is 5120) `normalization.cpp` sets it to the pipeline's
// `maxTotalThreadsPerThreadgroup()`. The fused kernel instead hardcodes
// `constexpr uint lsize = 1024`. If those disagree the two kernels partition
// the 5120-term sum differently and the claim fails.
//
// `qwen35FusedResidualRMSNorm` is `internal` to MLXLLM and MLXLLM is a
// dependency *package* product, so it is not built with `-enable-testing` and
// `@testable` cannot reach it. Rather than widen a submitted path just to test
// it, this suite reads the kernel source verbatim out of `Qwen35.swift` at
// runtime and JIT-compiles it under the same name, input names, output names
// and `ensureRowContiguous` flag. MLX's custom-kernel cache keys on exactly
// that tuple, so this is the shipped kernel, not a copy of it -- and it cannot
// silently drift, which is the E26 content-gate lesson applied here.

private let kernelDeclMarker = "qwen35FusedResidualRMSNormKernel = MLXFast.metalKernel("
private let wrapperDeclMarker = "func qwen35FusedResidualRMSNorm("
private let qwen35RelativePath = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

private struct ShippedFusedKernel {
    var name: String
    var inputNames: [String]
    var outputNames: [String]
    var source: String
    var ensureRowContiguous: Bool
    var declaredLsize: Int
    var declaredNReads: Int
    var wrapperGridMultiplier: Int
    var wrapperThreadGroup: Int
}

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
    // <root>/Tests/MLXFastTests/<this file>
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

private func firstInt(in line: String) -> Int? {
    var digits = ""
    for ch in line {
        if ch.isNumber {
            digits.append(ch)
        } else if !digits.isEmpty {
            break
        }
    }
    return Int(digits)
}

private func loadShippedFusedKernel() throws -> ShippedFusedKernel {
    let url = repoRoot().appendingPathComponent(qwen35RelativePath)
    let lines = try String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n")

    guard let declIdx = lines.firstIndex(where: { $0.contains(kernelDeclMarker) }) else {
        throw SourceGateError.missing(kernelDeclMarker)
    }
    guard
        let sourceOpen = lines[declIdx...].firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces).hasSuffix("source: \"\"\"")
        })
    else { throw SourceGateError.missing("source: \"\"\"") }
    guard
        let sourceClose = lines[(sourceOpen + 1)...].firstIndex(where: {
            let t = $0.trimmingCharacters(in: .whitespaces)
            return t == "\"\"\"" || t == "\"\"\","
        })
    else { throw SourceGateError.missing("closing \"\"\"") }

    let header = lines[declIdx ..< sourceOpen]
    guard let nameLine = header.first(where: { $0.contains("name:") }),
        let name = quotedStrings(in: nameLine).first
    else { throw SourceGateError.missing("name:") }
    guard let inLine = header.first(where: { $0.contains("inputNames:") }) else {
        throw SourceGateError.missing("inputNames:")
    }
    guard let outLine = header.first(where: { $0.contains("outputNames:") }) else {
        throw SourceGateError.missing("outputNames:")
    }

    let body = lines[(sourceOpen + 1) ..< sourceClose]
    guard
        let ercLine = lines[sourceClose...].prefix(6).first(where: {
            $0.contains("ensureRowContiguous:")
        })
    else { throw SourceGateError.missing("ensureRowContiguous:") }

    guard let lsizeLine = body.first(where: { $0.contains("constexpr uint lsize") }),
        let lsize = firstInt(in: lsizeLine)
    else { throw SourceGateError.missing("constexpr uint lsize") }
    guard let nReadsLine = body.first(where: { $0.contains("constexpr uint n_reads") }),
        let nReads = firstInt(in: nReadsLine)
    else { throw SourceGateError.missing("constexpr uint n_reads") }

    guard let wrapIdx = lines.firstIndex(where: { $0.contains(wrapperDeclMarker) }) else {
        throw SourceGateError.missing(wrapperDeclMarker)
    }
    let wrapper = lines[wrapIdx ..< min(wrapIdx + 30, lines.count)]
    guard let gridLine = wrapper.first(where: { $0.contains("grid:") }),
        let gridMultiplier = firstInt(in: gridLine)
    else { throw SourceGateError.missing("grid:") }
    guard let tgLine = wrapper.first(where: { $0.contains("threadGroup:") }),
        let threadGroup = firstInt(in: tgLine)
    else { throw SourceGateError.missing("threadGroup:") }

    return ShippedFusedKernel(
        name: name,
        inputNames: quotedStrings(in: inLine),
        outputNames: quotedStrings(in: outLine),
        source: body.joined(separator: "\n"),
        ensureRowContiguous: ercLine.contains("true"),
        declaredLsize: lsize,
        declaredNReads: nReads,
        wrapperGridMultiplier: gridMultiplier,
        wrapperThreadGroup: threadGroup
    )
}

// MARK: - Source gate (no GPU)

@Suite("Fused residual RMSNorm source gate")
struct QwenFusedResidualRMSNormSourceTests {

    @Test("the fused kernel still has the shape this audit measured")
    func theFusedKernelStillHasTheShapeThisAuditMeasured() throws {
        let k = try loadShippedFusedKernel()
        #expect(k.name == "qwen35_fused_residual_rms_norm")
        #expect(k.inputNames == ["x", "r", "weight", "eps"])
        #expect(k.outputNames == ["h", "normed"])
        #expect(k.ensureRowContiguous == false)
        #expect(k.declaredNReads == 4, "must equal MLX RMS_N_READS to share a partition")
        #expect(
            k.declaredLsize == k.wrapperThreadGroup,
            "kernel assumes lsize \(k.declaredLsize) but wrapper dispatches \(k.wrapperThreadGroup)"
        )
        #expect(
            k.wrapperGridMultiplier == k.wrapperThreadGroup,
            "grid multiplier \(k.wrapperGridMultiplier) must be one threadgroup per row")
    }

    @Test("the two arithmetic properties the eager chain depends on are still in the source")
    func theTwoArithmeticPropertiesTheEagerChainDependsOnAreStillInTheSource() throws {
        let k = try loadShippedFusedKernel()
        let squeezed = k.source.components(separatedBy: .whitespacesAndNewlines).joined()

        // (1) the residual sum is rounded to bf16 BEFORE it is squared, which is
        // what the eager `rmsNorm(bf16(x + r))` does. Accumulating the fp32 sum
        // would be more accurate and therefore wrong here.
        #expect(
            squeezed.contains("bfloathi=bfloat(xi+ri);"),
            "residual sum must be rounded to bf16 before use")
        #expect(
            squeezed.contains("acc+=float(hi)*float(hi);"),
            "the bf16-rounded value, not the fp32 sum, must be squared")

        // (2) precise::rsqrt, matching rms_norm.metal. `metal::rsqrt` is a
        // different (fast) implementation and would break bit-identity.
        #expect(squeezed.contains("metal::precise::rsqrt("), "must use precise::rsqrt")
        #expect(!squeezed.contains("=metal::rsqrt("), "fast rsqrt would break bit-identity")
        #expect(!squeezed.contains("fast::rsqrt("), "fast rsqrt would break bit-identity")

        // The reduction tree itself: simd_sum, zero-init of local_sums by
        // simd_group 0, lane-0 publish, simd_sum over the per-simd sums.
        #expect(squeezed.contains("acc=simd_sum(acc);"))
        #expect(squeezed.contains("acc=simd_sum(local_sums[simd_thread]);"))
    }
}

// MARK: - Bit-identity against MLX (GPU)

private func splitMix64(_ state: inout UInt64) -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
}

private func gaussian(_ state: inout UInt64) -> Float {
    // Box-Muller on two open-interval uniforms; deterministic given the seed.
    let u1 = max(Float(splitMix64(&state) >> 11) * 0x1p-53, 1e-7)
    let u2 = Float(splitMix64(&state) >> 11) * 0x1p-53
    return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2)
}

private enum ActivationFamily: String, CaseIterable {
    case normal
    case outlier
    case tied
    case large

    /// Residual-stream-shaped synthetic activations. These are NOT captured
    /// model activations -- no 27B forward is run here -- so they are chosen to
    /// bracket the real distribution rather than reproduce it: `outlier`
    /// reproduces the handful of very-high-magnitude residual channels that
    /// dominate an RMS reduction, and `tied` makes most partial sums exactly
    /// representable so reassociation has nothing to disagree about.
    func sample(count: Int, state: inout UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            switch self {
            case .normal:
                out[i] = gaussian(&state)
            case .outlier:
                let g = gaussian(&state)
                out[i] = (splitMix64(&state) % 500 == 0) ? g * 220 : g
            case .tied:
                out[i] = Float(Int(splitMix64(&state) % 9)) * 0.5 - 2.0
            case .large:
                out[i] = gaussian(&state) * 30
            }
        }
        return out
    }
}

private struct BitCompare {
    var cells: Int
    var differing: Int
    var maxUlp: Int

    var rate: Double { cells == 0 ? 0 : Double(differing) / Double(cells) }
}

private func compareBits(_ a: MLXArray, _ b: MLXArray) -> BitCompare {
    let av = a.view(dtype: .uint16).asType(.int32)
    let bv = b.view(dtype: .uint16).asType(.int32)
    let d = MLX.abs(av - bv)
    let differing = d.asType(.bool).asType(.int32).sum().item(Int32.self)
    let maxUlp = d.max().item(Int32.self)
    return BitCompare(cells: a.size, differing: Int(differing), maxUlp: Int(maxUlp))
}

@Suite("Fused residual RMSNorm bit-identity")
struct QwenFusedResidualRMSNormExactnessTests {

    static let enabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    static let axisSize = 5120
    static let rows =
        Int(ProcessInfo.processInfo.environment["MLXFAST_FUSED_RMSNORM_ROWS"] ?? "") ?? 128
    static let eps: Float = 1e-6

    private func inputs(family: ActivationFamily, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        var state = seed
        let n = Self.rows * Self.axisSize
        let x = MLXArray(family.sample(count: n, state: &state), [Self.rows, Self.axisSize])
            .asType(.bfloat16)
        let r = MLXArray(family.sample(count: n, state: &state), [Self.rows, Self.axisSize])
            .asType(.bfloat16)
        var wState = seed &+ 0xABCD
        let w = MLXArray(
            (0 ..< Self.axisSize).map { _ in 1.0 + 0.25 * gaussian(&wState) }, [Self.axisSize]
        ).asType(.bfloat16)
        return (x, r, w)
    }

    private func runFused(
        _ k: ShippedFusedKernel, source: String, name: String, threadGroup: Int,
        x: MLXArray, r: MLXArray, w: MLXArray
    ) -> (h: MLXArray, normed: MLXArray) {
        let kernel = MLXFast.metalKernel(
            name: name,
            inputNames: k.inputNames,
            outputNames: k.outputNames,
            source: source,
            ensureRowContiguous: k.ensureRowContiguous)
        let out = kernel(
            [x, r, w, MLXArray(Self.eps)],
            grid: (Self.rows * threadGroup, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [.bfloat16, .bfloat16])
        return (out[0], out[1])
    }

    @Test(
        "the fused kernel is bit-identical to the eager add + MLX rmsNorm chain",
        .enabled(if: QwenFusedResidualRMSNormExactnessTests.enabled))
    func theFusedKernelIsBitIdenticalToTheEagerAddAndMLXRMSNormChain() throws {
        let k = try loadShippedFusedKernel()
        var report: [String: Any] = [
            "schema": "e28.fused_residual_rmsnorm.v1",
            "kernel": k.name,
            "declared_lsize": k.declaredLsize,
            "declared_n_reads": k.declaredNReads,
            "axis_size": Self.axisSize,
            "rows_per_family": Self.rows,
            "mlx_rms_looped_limit": 4096,
            "eps": Double(Self.eps),
        ]

        var families: [[String: Any]] = []
        var totalNormedDiffering = 0
        var totalResidualDiffering = 0
        var worstUlp = 0

        for (i, family) in ActivationFamily.allCases.enumerated() {
            let (x, r, w) = inputs(family: family, seed: 0x5E17_2026_0228 &+ UInt64(i))

            let fused = runFused(
                k, source: k.source, name: k.name, threadGroup: k.wrapperThreadGroup,
                x: x, r: r, w: w)

            let eagerH = x + r
            let eagerNormed = MLXFast.rmsNorm(eagerH, weight: w, eps: Self.eps)
            eval(fused.h, fused.normed, eagerH, eagerNormed)

            let hCmp = compareBits(fused.h, eagerH)
            let nCmp = compareBits(fused.normed, eagerNormed)
            totalResidualDiffering += hCmp.differing
            totalNormedDiffering += nCmp.differing
            worstUlp = max(worstUlp, nCmp.maxUlp)

            families.append([
                "family": family.rawValue,
                "cells": nCmp.cells,
                "residual_differing_cells": hCmp.differing,
                "normed_differing_cells": nCmp.differing,
                "normed_differing_rate": nCmp.rate,
                "normed_max_ulp": nCmp.maxUlp,
            ])

            #expect(
                hCmp.differing == 0,
                "\(family.rawValue): fused residual h differs from eager x + r in \(hCmp.differing) cells"
            )
            #expect(
                nCmp.differing == 0,
                "\(family.rawValue): fused normed differs from MLX rmsNorm in \(nCmp.differing) cells (max \(nCmp.maxUlp) ulp)"
            )
        }

        // Positive control. Re-partitioning the same 5120-term reduction across
        // 512 threads instead of 1024 is the exact failure mode the pass above
        // would hide if MLX's `rms_looped` did not also run at 1024, so the
        // harness has to be able to see it. If this control cannot separate the
        // two partitions then the pass above proves nothing about `lsize`.
        let controlThreadGroup = k.wrapperThreadGroup / 2
        let controlSource = k.source.replacingOccurrences(
            of: "constexpr uint lsize = \(k.declaredLsize);",
            with: "constexpr uint lsize = \(controlThreadGroup);")
        #expect(controlSource != k.source, "control must actually change lsize")

        let (cx, cr, cw) = inputs(family: .normal, seed: 0x5E17_2026_0228)
        let control = runFused(
            k, source: controlSource, name: k.name + "_lsize_control",
            threadGroup: controlThreadGroup, x: cx, r: cr, w: cw)
        let controlEagerH = cx + cr
        let controlEagerNormed = MLXFast.rmsNorm(controlEagerH, weight: cw, eps: Self.eps)
        eval(control.h, control.normed, controlEagerH, controlEagerNormed)

        let controlH = compareBits(control.h, controlEagerH)
        let controlNormed = compareBits(control.normed, controlEagerNormed)

        report["families"] = families
        report["total_residual_differing_cells"] = totalResidualDiffering
        report["total_normed_differing_cells"] = totalNormedDiffering
        report["max_normed_ulp"] = worstUlp
        report["control_lsize"] = controlThreadGroup
        report["control_residual_differing_cells"] = controlH.differing
        report["control_normed_differing_cells"] = controlNormed.differing
        report["control_normed_differing_rate"] = controlNormed.rate
        report["control_normed_max_ulp"] = controlNormed.maxUlp

        // The elementwise residual cannot depend on the reduction partition, so
        // a control that also broke `h` would mean the control changed
        // something other than the thing under test.
        #expect(
            controlH.differing == 0,
            "control changed the elementwise residual, so it is not an lsize-only control")
        #expect(
            controlNormed.differing > 0,
            "harness cannot distinguish a 512-thread reduction from a 1024-thread one, so the bit-identity pass says nothing about lsize"
        )

        if let out = ProcessInfo.processInfo.environment["MLXFAST_FUSED_RMSNORM_OUT"], !out.isEmpty {
            let data = try JSONSerialization.data(
                withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
        }
        print("E28_FUSED_RMSNORM \(report)")
    }
}
