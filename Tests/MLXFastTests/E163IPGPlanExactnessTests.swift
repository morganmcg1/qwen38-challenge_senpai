import CryptoKit
import Foundation
import MLX
import MLXLLM
import Testing

// E163 -- exactness gate for the minimum-accumulator QMV width plans.
//
// The arms move input rows out of one wide threadgroup into more, narrower
// ones: width 5 goes from one group of five rows to two groups of three and
// two, width 4 from one group of four to two of two, width 6 from two groups
// of three to three of two. The claim is that this cannot move a single output
// bit, because `qwen_e120_qmv_wide` gives every input row its own accumulator
// lane and sums only over `k`. The split across threadgroups decides which
// lanes a group carries, never the order or the type of any addition.
//
// A claim about bits is checked, not trusted. This suite compares the routed
// dispatch (`Qwen35CustomQMV.matmul`, the path the scored worker takes) with
// MLX's own `quantized_matmul` on the same operands, at every scored shape and
// at the touched widths plus untouched controls, and it reports the actual
// floating-point values it compared.
//
// Three properties make this a gate rather than a smoke test:
//
//   * The operands are real bf16 floats with both signs and non-integer
//     mantissas, so an accumulation-order change would show up as a rounding
//     difference. An integer-valued operand set can be reordered freely and
//     would hide exactly the defect this gate exists to catch.
//   * Every touched width carries a positive control that perturbs one
//     activation by one bf16 ulp and requires the comparison to fail. A gate
//     that cannot fail proves nothing.
//   * The control also records which output rows moved. One perturbed element
//     of the last row must change that row and no other, which is the row
//     independence that makes a regrouping exact by construction.
//
// The digest of every cell is written out, so the same suite run under each
// `MLX_E163_IPG_PLAN` arm produces files that `research/e163_exactness_compare.py`
// diffs cell by cell. That comparison covers the untouched widths too: they
// must be identical across arms.
//
// Enable with `MLXFAST_RUN_E163_EXACTNESS=1` and point `MLXFAST_E163_OUT` at
// the JSON destination.

private struct E163Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// The affine 4-bit group-64 transposed projections one target verify pass
/// reaches, with their per-pass dispatch counts (E137's live routing census).
private let e163Shapes: [E163Shape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
]

/// Widths 4, 5 and 6 are the ones the arms regroup. Widths 3, 7 and 8 keep the
/// shipped plan under every arm, so they prove the edit stays inside the table
/// entries it names. Width 8 is thorfinn's, and this suite must show it did not
/// move.
private let e163TouchedWidths = [4, 5, 6]
private let e163Widths = [3, 4, 5, 6, 7, 8]

private struct E163Weight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// Packed nibbles plus per-group affine scales and biases. The nibble stream is
/// a hash tile offset per row, so it is deterministic and free of structure the
/// kernel could exploit. The scales and biases are irregular non-integer bf16
/// values, which is what makes the products and the group sums land on real
/// rounding boundaries.
private func e163Weight(k: Int, n: Int) -> E163Weight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words])
        + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1.7e-6
    let scaleTile: [Float] = (0..<groups).map { index in
        0.00713 + 0.00391 * Float((index &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index in
        -0.0517 - 0.0233 * Float((index &* 23) % 53) / 53.0
    }
    let weight = E163Weight(
        w: w,
        scales: (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16),
        biases: (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16))
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

/// Activations with both signs and non-integer bf16 mantissas.
private func e163Activations(m: Int, k: Int) -> [Float] {
    var values = [Float](repeating: 0, count: m * k)
    for row in 0..<m {
        for col in 0..<k {
            let hashed = (row &* 7_919 &+ col &* 104_729 &+ 15_485_863) % 1_000_003
            values[row * k + col] = (Float(hashed) / 1_000_003.0 - 0.5) * 1.7320508
        }
    }
    return values
}

/// The next bf16 value above `value`, by incrementing the stored bit pattern.
/// One ulp is the smallest perturbation the operand can carry, so a comparison
/// that survives it is not a bitwise comparison at all.
private func e163NextBFloat16Up(_ value: Float) -> Float {
    let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    return Float(bitPattern: UInt32(bits &+ 1) << 16)
}

private func e163Digest(_ values: [Float]) -> String {
    var hasher = SHA256()
    values.withUnsafeBytes { hasher.update(bufferPointer: $0) }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func e163Hex(_ value: Float) -> String {
    String(format: "0x%08x", value.bitPattern)
}

@Suite
struct E163IPGPlanExactnessTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E163_EXACTNESS"] == "1"
    }

    @Test(.enabled(if: E163IPGPlanExactnessTests.enabled))
    func routedWidthPlanIsBitExact() throws {
        let outPath = try #require(
            ProcessInfo.processInfo.environment["MLXFAST_E163_OUT"],
            "MLXFAST_E163_OUT must name the JSON destination")
        let arm = ProcessInfo.processInfo.environment["MLX_E163_IPG_PLAN"] ?? "shipped"
        let plan = Qwen35CustomQMV.inputsPerGroupPlan

        var cells: [[String: Any]] = []
        for shape in e163Shapes {
            let weight = e163Weight(k: shape.k, n: shape.n)
            for m in e163Widths {
                var values = e163Activations(m: m, k: shape.k)
                let x = MLXArray(values, [m, shape.k]).asType(.bfloat16)

                let reference = quantizedMM(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
                let routed = Qwen35CustomQMV.matmul(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    groupSize: 64, bits: 4, mode: .affine)
                let candidate: MLXArray = try #require(
                    routed,
                    Comment(
                        rawValue: "the candidate route declined \(shape.name) at "
                            + "M=\(m), so this cell would compare MLX against itself"))
                eval(reference, candidate)

                // bf16 -> f32 is exact, so these are the kernels' own bits.
                let referenceValues = reference.asType(.float32).asArray(Float.self)
                let candidateValues = candidate.asType(.float32).asArray(Float.self)
                var mismatches = 0
                var maxAbsDelta: Float = 0
                for index in 0..<candidateValues.count
                where candidateValues[index].bitPattern != referenceValues[index].bitPattern {
                    mismatches += 1
                    maxAbsDelta = max(
                        maxAbsDelta, abs(candidateValues[index] - referenceValues[index]))
                }

                let ipg = Qwen35CustomQMV.inputsPerGroup(m)
                var cell: [String: Any] = [
                    "shape": shape.name,
                    "k": shape.k,
                    "n": shape.n,
                    "m": m,
                    "calls_per_verify": shape.callsPerVerify,
                    "elements": candidateValues.count,
                    "inputs_per_group": ipg,
                    "accumulator_lanes_na": ipg,
                    "active_groups": (m + ipg - 1) / ipg,
                    "tail_lanes": m % ipg == 0 ? ipg : max(m % ipg, 2),
                    "matches_incumbent_bitwise": mismatches == 0,
                    "mismatched_elements": mismatches,
                    "max_abs_delta_vs_incumbent": Double(maxAbsDelta),
                    "candidate_digest": e163Digest(candidateValues),
                    "incumbent_digest": e163Digest(referenceValues),
                    "sample_values": sampleValues(
                        candidate: candidateValues, incumbent: referenceValues,
                        m: m, n: shape.n),
                ]

                #expect(
                    mismatches == 0,
                    Comment(
                        rawValue: "\(shape.name) M=\(m): \(mismatches) of "
                            + "\(candidateValues.count) outputs differ from the "
                            + "incumbent, max_abs_delta \(maxAbsDelta)"))

                if e163TouchedWidths.contains(m) {
                    let row = m - 1
                    let column = 17
                    values[row * shape.k + column] = e163NextBFloat16Up(
                        values[row * shape.k + column])
                    let perturbedX = MLXArray(values, [m, shape.k]).asType(.bfloat16)
                    let perturbed = try #require(
                        Qwen35CustomQMV.matmul(
                            perturbedX, weight.w, scales: weight.scales,
                            biases: weight.biases, groupSize: 64, bits: 4, mode: .affine))
                    eval(perturbed)
                    let perturbedValues = perturbed.asType(.float32).asArray(Float.self)

                    var changedRows: Set<Int> = []
                    var changed = 0
                    for index in 0..<perturbedValues.count
                    where perturbedValues[index].bitPattern != candidateValues[index].bitPattern {
                        changed += 1
                        changedRows.insert(index / shape.n)
                    }
                    cell["positive_control"] = [
                        "perturbed_row": row,
                        "perturbed_column": column,
                        "perturbed_by": "one_bfloat16_ulp",
                        "rejects": changed > 0,
                        "changed_elements": changed,
                        "changed_rows": changedRows.sorted(),
                    ]
                    #expect(
                        changed > 0,
                        Comment(
                            rawValue: "\(shape.name) M=\(m): one ulp on row \(row) changed "
                                + "no output, so the comparison cannot fail"))
                    #expect(
                        changedRows == [row],
                        Comment(
                            rawValue: "\(shape.name) M=\(m): one ulp on row \(row) moved "
                                + "rows \(changedRows.sorted()), so the rows are not "
                                + "independent"))
                }

                cells.append(cell)
            }
        }

        let payload: [String: Any] = [
            "harness": "local",
            "arm": arm,
            "plan": plan.keys.sorted().map { ["m": $0, "inputs_per_group": plan[$0]!] },
            "touched_widths": e163TouchedWidths,
            "widths": e163Widths,
            "reference": "mlx quantized_matmul, same operands, same process",
            "cells": cells,
        ]
        try JSONSerialization
            .data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: outPath))
        print("E163_EXACTNESS_OUT \(outPath) arm=\(arm) cells=\(cells.count)")
    }

    /// Real output floats, so the report can quote what was compared rather
    /// than only a digest of it. The last row is the one a regrouping moves
    /// into a different threadgroup.
    private func sampleValues(
        candidate: [Float], incumbent: [Float], m: Int, n: Int
    ) -> [[String: Any]] {
        [(m - 1, 0), (m - 1, n / 2), (0, n - 1)].map { row, column in
            let index = row * n + column
            return [
                "row": row,
                "column": column,
                "candidate": Double(candidate[index]),
                "candidate_bits": e163Hex(candidate[index]),
                "incumbent": Double(incumbent[index]),
                "incumbent_bits": e163Hex(incumbent[index]),
            ]
        }
    }
}
