import Foundation
import MLX
import MLXRandom
import Testing

// E100 -- what does instantiating NA = 5 cost the widths that never use it?
//
// The QMV dispatcher is ONE kernel entry point. `qmv_fast_crossrow_affine4_g64_m`
// is selected by `ntg.x` inside that entry point, so every instantiated branch
// shares one register allocation. Raising the wide helper's bound to NA = 5 to
// collapse M = 5 to a single x-group therefore has two terms, not one:
//
//   BENEFIT  M = 5 and M = 9 read the weight matrix fewer times.
//   TAX      M = 1..4 and M = 6..8 keep their own dispatch entries but may pay
//            a lower occupancy, because E76 measures the shipped helper at 91
//            applegpu_g17s registers for NA <= 4 and 98 for NA = 5.
//
// The ranked board prices the benefit at -0.700 % +/- 0.285 % per stream
// removal, but every board contrast that carries a measurement moves IPG
// inside [2, 4], so the board cannot see the tax at all
// (`research/e100_na5_board.py`). This probe measures both terms locally, in
// the same session, at the six scored linear shapes.
//
// The tax term is a NULL: widths whose dispatch entry did not change must come
// back flat. That makes the experiment falsifiable in the direction that
// matters, because a flat tax and a large benefit is the only combination that
// justifies a ranked slot.
//
// EXACTNESS. Row m of the output is carried by lane m of an independent
// accumulator and `simd_sum` reduces along K WITHIN a row, so moving a row
// between x-groups must not change one bit. This probe tests that claim
// directly rather than assuming it: for each shape it digests every output row
// at every width and asserts that a given row is bit-identical across every
// width that reaches a cross-row kernel. `crossCheck` then compares the same
// digests against a float32 dequantised reference, which MUST differ; that is
// the positive control proving the digest can detect a reordering at all.
//
// Research instrument. Off unless `MLXFAST_RUN_E100_PROBE=1`. `Tests/` is never
// packaged into a submission. Within-session relative measurement, no thermal
// gate, no score.

private struct E100Cell {
    var shape: String
    var hidden: Int
    var outputs: Int
    var width: Int
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
    var rowDigests: [UInt64]
}

@Suite("E100 stream collapse probe")
struct E100StreamCollapseProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E100_PROBE"] == "1"

    static let groupSize = 64
    static let bits = 4
    static let widths = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    /// Every scored linear that reaches the `out_vec_size >= 4096` WIDE branch.
    /// `lm_head` is opt-in because it allocates about 715 MB of packed weights.
    static let shapes: [(name: String, hidden: Int, outputs: Int)] = {
        var s: [(name: String, hidden: Int, outputs: Int)] = [
            ("mlp.gate_up", 5120, 34816),
            ("gdn.in_proj", 5120, 16480),
            ("fa.qkv", 5120, 14336),
            ("mlp.down", 17408, 5120),
            ("gdn.out_proj", 6144, 5120),
        ]
        if ProcessInfo.processInfo.environment["MLXFAST_E100_LM_HEAD"] == "1" {
            s.append(("lm_head", 5120, 248320))
        }
        return s
    }()

    /// affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias.
    private static func packedBytes(hidden: Int, outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func timed(_ count: Int, _ body: () -> MLXArray) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            eval(body())
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// The floor every cell pays: one blocking `eval` of negligible GPU work.
    private static func evalOverheadMicroseconds() -> Double {
        let tiny = MLXArray(Array(repeating: Float(1), count: 16))
        eval(tiny)
        _ = timed(200) { tiny + 1 }
        return timed(4000) { tiny + 1 }
    }

    /// FNV-1a over the raw float32 bit patterns of one output row.
    private static func digest(_ values: ArraySlice<Float>) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in values {
            var bits = UInt64(v.bitPattern)
            for _ in 0 ..< 4 {
                h = (h ^ (bits & 0xff)) &* 0x0000_0100_0000_01b3
                bits >>= 8
            }
        }
        return h
    }

    private static func rowDigests(_ out: MLXArray, width: Int, outputs: Int)
        -> [UInt64]
    {
        let flat: [Float] = out.asType(.float32).asArray(Float.self)
        return (0 ..< width).map {
            digest(flat[($0 * outputs) ..< (($0 + 1) * outputs)])
        }
    }

    @Test(
        "collapsing an x-group is bit-exact and does not move other widths",
        .enabled(if: E100StreamCollapseProbeTests.enabled))
    func streamCollapse() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E100 overhead us=%.3f", overhead))

        var cells: [E100Cell] = []
        var control: [[String: Any]] = []

        for shape in Self.shapes {
            // Fixed seed per shape so both arms of an A/B build the identical
            // working set and their digests are directly comparable.
            MLXRandom.seed(UInt64(0xE100) &+ UInt64(shape.outputs))
            let dense = MLXRandom.normal([shape.outputs, shape.hidden])
                .asType(.bfloat16)
            let (packed, scales, biases) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed, scales, biases)

            let bytes = Self.packedBytes(hidden: shape.hidden,
                                         outputs: shape.outputs)
            // Aim at roughly 100 ms of integrated work per cell.
            let count = max(20, min(400, Int(1.0e11) / (bytes * 4)))

            // One activation block, sliced per width. Row m must be the same
            // input vector at every width, or the cross-width digest check
            // below compares different products and cannot detect a
            // reassociation.
            MLXRandom.seed(UInt64(0xBEEF) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([Self.widths.max()!, shape.hidden])
                .asType(.bfloat16)
            eval(block)

            var inputs: [Int: MLXArray] = [:]
            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                eval(x)
                inputs[width] = x
            }

            func call(_ width: Int) -> MLXArray {
                quantizedMM(
                    inputs[width]!, packed, scales: scales, biases: biases,
                    transpose: true, groupSize: Self.groupSize,
                    bits: Self.bits, mode: .affine)
            }
            func measure(_ width: Int) -> Double {
                _ = Self.timed(5) { call(width) }
                return Self.timed(count) { call(width) }
            }

            let forward = Self.widths.map(measure)
            let reverse = Array(Self.widths.reversed().map(measure).reversed())

            for (index, width) in Self.widths.enumerated() {
                let out = call(width)
                eval(out)
                let cell = E100Cell(
                    shape: shape.name, hidden: shape.hidden,
                    outputs: shape.outputs, width: width,
                    forwardMicroseconds: forward[index],
                    reverseMicroseconds: reverse[index],
                    replicates: count,
                    rowDigests: Self.rowDigests(
                        out, width: width, outputs: shape.outputs))
                cells.append(cell)
                let mean = (cell.forwardMicroseconds + cell.reverseMicroseconds) / 2
                let net = mean - overhead
                print(
                    "E100 cell shape=\(shape.name) k=\(shape.hidden) "
                        + "n=\(shape.outputs) m=\(width) "
                        + String(
                            format: "fwd_us=%.3f rev_us=%.3f net_us=%.3f "
                                + "one_pass_gb_s=%.1f",
                            cell.forwardMicroseconds, cell.reverseMicroseconds,
                            net, Double(bytes) / net / 1e3)
                        + " replicates=\(count)")
            }

            // Positive control: the same product through a dequantised float32
            // path. Its digests MUST differ, or the digest is not sensitive to
            // a reassociation and the exactness check above is vacuous.
            let ref = matmul(inputs[5]!.asType(.float32),
                             dense.asType(.float32).transposed())
                .asType(.bfloat16)
            eval(ref)
            let refDigests = Self.rowDigests(
                ref, width: 5, outputs: shape.outputs)
            let kernelDigests = cells.last(where: {
                $0.shape == shape.name && $0.width == 5
            })!.rowDigests
            control.append([
                "shape": shape.name,
                "differs": zip(refDigests, kernelDigests).contains { $0 != $1 },
            ])

            eval(MLXArray(0))
            Memory.clearCache()
        }

        // Row m must be bit-identical at every width that reaches a cross-row
        // kernel. M = 1 is excluded: `ntg.x == 1` has no `case` and falls
        // through to the generic single-row path.
        var mismatches: [[String: Any]] = []
        for shape in Self.shapes {
            let byWidth = cells.filter { $0.shape == shape.name && $0.width >= 2 }
            guard let reference = byWidth.max(by: { $0.width < $1.width })
            else { continue }
            for cell in byWidth where cell.width < reference.width {
                for row in 0 ..< cell.width
                where reference.rowDigests[row] != cell.rowDigests[row] {
                    mismatches.append([
                        "shape": shape.name, "width": cell.width, "row": row,
                    ])
                }
            }
        }
        for m in mismatches {
            print("E100 MISMATCH \(m)")
        }
        for c in control {
            print("E100 positive_control \(c)")
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E100_OUT"],
            !path.isEmpty
        {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "eval_overhead_us": overhead,
                "row_digest_mismatches": mismatches,
                "positive_control": control,
                "cells": cells.map {
                    [
                        "shape": $0.shape,
                        "hidden": $0.hidden,
                        "outputs": $0.outputs,
                        "packed_bytes": Self.packedBytes(
                            hidden: $0.hidden, outputs: $0.outputs),
                        "m": $0.width,
                        "forward_us": $0.forwardMicroseconds,
                        "reverse_us": $0.reverseMicroseconds,
                        "replicates": $0.replicates,
                        "row_digests": $0.rowDigests.map { String($0) },
                    ]
                },
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
        #expect(mismatches.isEmpty)
        #expect(control.allSatisfy { ($0["differs"] as? Bool) == true })
    }
}
