// Research-only (qwen38-r1-e55): Risk 1 gate driver.
//
// Answers three questions the advisor made a hard gate:
//   1. Does metal::vec<float,5> compile, and what is it? (sizeof, alignof,
//      lane independence under indexed write and under arithmetic)
//   2. Does every one of the 5 lanes carry the input row it should, with the
//      SAME accumulation the shipped NA<=4 type produces? The decisive test is
//      GPU-vs-GPU and bitwise: lane m of an NA=5 run must equal the same row
//      computed by a narrower, already-shipped NA on the same data, because the
//      wide helper's lanes are never reduced across each other.
//   3. Does the check actually fail when a lane is wrong? Three perturbed
//      kernels are positive controls and MUST be reported as caught.
//
//   xcrun -sdk macosx metal -std=metal3.1 -O2 -c research/e55_vec5_probe.metal -o /tmp/e55.air
//   xcrun -sdk macosx metallib /tmp/e55.air -o /tmp/e55.metallib
//   swiftc -O research/e55_vec5_check.swift -o /tmp/e55_vec5_check
//   /tmp/e55_vec5_check /tmp/e55.metallib

import Foundation
import Metal

let metallibPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/e55.metallib"

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write(Data("no Metal device\n".utf8))
    exit(1)
}
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: metallibPath))

struct ProbeParams {
    var K: Int32
    var rows: Int32
    var first_m: Int32
    var out_row: Int32
}

let K = 256
let rows = 4
let M = 5

// Deterministic, distinct-per-row inputs. Distinct rows are the point: a lane
// that carries the wrong row must produce a different number.
var wpk = [UInt16](repeating: 0, count: rows * (K / 4))
for r in 0..<rows {
    for i in 0..<(K / 4) {
        wpk[r * (K / 4) + i] = UInt16(truncatingIfNeeded: (r * 7919 + i * 104729) &* 2654435761)
    }
}
var scales = [Float](repeating: 0, count: rows * (K / 64))
var biases = [Float](repeating: 0, count: rows * (K / 64))
for r in 0..<rows {
    for g in 0..<(K / 64) {
        scales[r * (K / 64) + g] = 0.0031_25 * Float(1 + r) + 0.000_11 * Float(g)
        biases[r * (K / 64) + g] = -0.019_5 * Float(1 + r) - 0.001_3 * Float(g)
    }
}
// Each input row gets its own scale AND its own pattern, so row m is separable
// from row m' by value, not just by magnitude.
var x = [Float](repeating: 0, count: M * K)
for m in 0..<M {
    for k in 0..<K {
        let phase = Float(m) * 0.618_034 + Float(k) * 0.017_453
        x[m * K + k] = (Float(m + 1) * 0.25) * sin(phase) + 0.03 * Float((m * 31 + k) % 17)
    }
}

func makeBuffer<T>(_ a: [T]) -> MTLBuffer {
    return a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
let wBuf = makeBuffer(wpk)
let sBuf = makeBuffer(scales)
let bBuf = makeBuffer(biases)
let xBuf = makeBuffer(x)

func runWide(_ name: String, firstM: Int) -> [Float] {
    guard let fn = library.makeFunction(name: name) else {
        FileHandle.standardError.write(Data("missing function \(name)\n".utf8))
        exit(1)
    }
    let pipeline = try! device.makeComputePipelineState(function: fn)
    let yBuf = device.makeBuffer(length: M * rows * 4, options: .storageModeShared)!
    memset(yBuf.contents(), 0, M * rows * 4)
    var params = ProbeParams(K: Int32(K), rows: Int32(rows), first_m: Int32(firstM), out_row: 0)
    let pBuf = device.makeBuffer(bytes: &params, length: MemoryLayout<ProbeParams>.stride, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(wBuf, offset: 0, index: 0)
    enc.setBuffer(sBuf, offset: 0, index: 1)
    enc.setBuffer(bBuf, offset: 0, index: 2)
    enc.setBuffer(xBuf, offset: 0, index: 3)
    enc.setBuffer(yBuf, offset: 0, index: 4)
    enc.setBuffer(pBuf, offset: 0, index: 5)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let ptr = yBuf.contents().bindMemory(to: Float.self, capacity: M * rows)
    return Array(UnsafeBufferPointer(start: ptr, count: M * rows))
}

// ---- Q1: what is vec<float,5> ----
let layoutFn = library.makeFunction(name: "vec5_layout")!
let layoutPipeline = try device.makeComputePipelineState(function: layoutFn)
let outBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
memset(outBuf.contents(), 0, 16 * 4)
do {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(layoutPipeline)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}
let lay = Array(UnsafeBufferPointer(start: outBuf.contents().bindMemory(to: UInt32.self, capacity: 16), count: 16))

print("device=\(device.name)")
print("--- vec<float,N> layout as compiled for THIS device ---")
print("vec<float,2> sizeof=\(lay[0]) alignof=\(lay[1])")
print("vec<float,3> sizeof=\(lay[2]) alignof=\(lay[3])")
print("vec<float,4> sizeof=\(lay[4]) alignof=\(lay[5])")
print("vec<float,5> sizeof=\(lay[6]) alignof=\(lay[7])")
print("vec5_indexed_write_bleed_mask=\(lay[8])  (0 = every t[i]=v touched only lane i)")
print("vec5_lanewise_arith_fault_mask=\(lay[9])  (0 = v*2+1 stayed lane-local)")

// ---- Q2/Q3: lane fidelity, bitwise, GPU vs GPU ----
let na5 = runWide("wide_na5_faithful", firstM: 0)
let na4 = runWide("wide_na4_faithful", firstM: 0)
let na3lo = runWide("wide_na3_faithful", firstM: 0)
let na2hi = runWide("wide_na2_faithful", firstM: 3)

func bits(_ f: Float) -> UInt32 { return f.bitPattern }
func maxUlp(_ a: [Float], _ b: [Float], _ idx: [Int]) -> (Int, Double) {
    var worstUlp = 0
    var worstRel = 0.0
    for i in idx {
        let d = abs(Int64(bits(a[i])) - Int64(bits(b[i])))
        worstUlp = max(worstUlp, Int(d))
        if b[i] != 0 { worstRel = max(worstRel, Double(abs(a[i] - b[i]) / abs(b[i]))) }
    }
    return (worstUlp, worstRel)
}

// Reference index sets: NA=5 lane m holds input row m, output rows 0..3.
func idxForInputRow(_ m: Int) -> [Int] { return (0..<rows).map { m * rows + $0 } }

print("\n--- lane fidelity: NA=5 lane m must equal the SAME row under a shipped narrower NA ---")
var laneVerdicts: [String] = []
for m in 0..<4 {
    let ref = (m < 3) ? na3lo : na4          // rows 0..2 from NA=3, row 3 from NA=4
    let refName = (m < 3) ? "na3(first_m=0)" : "na4(first_m=0)"
    let (u, r) = maxUlp(na5, ref, idxForInputRow(m))
    let ok = (u == 0)
    laneVerdicts.append("lane\(m)=\(ok ? "EXACT" : "DIFF")")
    print("lane \(m) vs \(refName): max_ulp=\(u) max_rel=\(String(format: "%.3e", r))  \(ok ? "EXACT" : "MISMATCH")")
}
do {
    let (u, r) = maxUlp(na5, na2hi, idxForInputRow(4))
    let ok = (u == 0)
    laneVerdicts.append("lane4=\(ok ? "EXACT" : "DIFF")")
    print("lane 4 vs na2(first_m=3): max_ulp=\(u) max_rel=\(String(format: "%.3e", r))  \(ok ? "EXACT" : "MISMATCH")")
}
// Cross-check that the narrower references agree with each other where they
// overlap, so an "EXACT" verdict above cannot come from a broken reference.
do {
    let (u, _) = maxUlp(na3lo, na4, (0..<3).flatMap { idxForInputRow($0) })
    print("reference cross-check na3 vs na4 on rows 0..2: max_ulp=\(u) (must be 0)")
}

print("\n--- the 5 lanes carry DISTINCT rows (a lane-collapse would hide behind 'exact') ---")
for m in 0..<M {
    let v = idxForInputRow(m).map { na5[$0] }
    print(String(format: "input row %d -> outputs [% .6f, % .6f, % .6f, % .6f]", m, v[0], v[1], v[2], v[3]))
}
var distinct = true
for m in 0..<M {
    for n in (m + 1)..<M {
        if idxForInputRow(m).map({ bits(na5[$0]) }) == idxForInputRow(n).map({ bits(na5[$0]) }) { distinct = false }
    }
}
print("all_five_lane_outputs_distinct=\(distinct)")

// ---- Q3: positive controls. These MUST be caught. ----
print("\n--- positive controls: a deliberately wrong lane MUST be caught ---")
var controlsAllCaught = true
for (name, label) in [("wide_na5_swap04", "swap lanes 0 and 4"),
                      ("wide_na5_zero4", "zero lane 4 activations"),
                      ("wide_na5_leak34", "leak lane 3 into lane 4")] {
    let bad = runWide(name, firstM: 0)
    var caughtLanes: [Int] = []
    var detail: [String] = []
    for m in 0..<5 {
        let ref = (m < 3) ? na3lo : (m == 3 ? na4 : na2hi)
        let (u, r) = maxUlp(bad, ref, idxForInputRow(m))
        if u != 0 { caughtLanes.append(m) }
        detail.append("lane\(m):ulp=\(u),rel=\(String(format: "%.2e", r))")
    }
    let caught = !caughtLanes.isEmpty
    if !caught { controlsAllCaught = false }
    print("\(name) (\(label)): caught=\(caught) offending_lanes=\(caughtLanes)")
    print("    \(detail.joined(separator: "  "))")
}

let laneGatePassed = laneVerdicts.allSatisfy { $0.hasSuffix("EXACT") } && distinct
    && lay[8] == 0 && lay[9] == 0
print("\nRISK1_GATE lane_fidelity=\(laneGatePassed ? "PASS" : "FAIL") positive_controls=\(controlsAllCaught ? "PASS" : "FAIL")")
exit((laneGatePassed && controlsAllCaught) ? 0 : 1)
