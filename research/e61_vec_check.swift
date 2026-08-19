// Research-only (qwen38-r1-e61): rung 0.1 driver.
//
// Answers, for NA = 5, 6, 7 and 8, the three questions E55 answered at NA = 5:
//   1. What is `vec<float,N>` on this device: sizeof, alignof, indexed-write
//      lane bleed, lane-local arithmetic.
//   2. Does every lane carry the input row it should, bitwise, against the
//      shipped narrow types NA = 2 and NA = 4 computed on the same data?
//   3. Does the check fail when a lane is deliberately wrong? Three perturbed
//      kernels per NA are positive controls and MUST all be caught.
//
//   xcrun -sdk macosx metal -std=metal3.1 -O2 -c research/e61_vec_probe.metal -o /tmp/e61.air
//   xcrun -sdk macosx metallib /tmp/e61.air -o /tmp/e61.metallib
//   swiftc -O research/e61_vec_check.swift -o /tmp/e61_vec_check
//   /tmp/e61_vec_check /tmp/e61.metallib research/e61-artifacts/e61-vec-probe.json

import Foundation
import Metal

let metallibPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/e61.metallib"
let jsonPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : ""

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
let M = 8

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

// ---- Q1: what is vec<float,N> on this device ----
let layoutFn = library.makeFunction(name: "vec_layout")!
let layoutPipeline = try device.makeComputePipelineState(function: layoutFn)
let outBuf = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
memset(outBuf.contents(), 0, 32 * 4)
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
let lay = Array(UnsafeBufferPointer(start: outBuf.contents().bindMemory(to: UInt32.self, capacity: 32), count: 32))

print("device=\(device.name)")
print("--- vec<float,N> layout as compiled for THIS device ---")
var layoutJSON: [String] = []
var layoutClean = true
for n in 2...8 {
    let sz = lay[2 * (n - 2)], al = lay[2 * (n - 2) + 1]
    let bleed = lay[14 + (n - 2)], arith = lay[21 + (n - 2)]
    if bleed != 0 || arith != 0 { layoutClean = false }
    print("vec<float,\(n)> sizeof=\(sz) alignof=\(al) lane_bleed_mask=\(bleed) lanewise_arith_fault_mask=\(arith)")
    layoutJSON.append("\"\(n)\": {\"sizeof\": \(sz), \"alignof\": \(al), \"lane_bleed_mask\": \(bleed), \"arith_fault_mask\": \(arith)}")
}

// ---- Q2: lane fidelity, bitwise, GPU vs GPU ----
func bits(_ f: Float) -> UInt32 { return f.bitPattern }
func idxForInputRow(_ m: Int) -> [Int] { return (0..<rows).map { m * rows + $0 } }
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

// Two independent references over all 8 input rows, both from shipped NA<=4
// types: NA=2 at first_m 0,2,4,6 and NA=4 at first_m 0,4.
var refNA2 = [Float](repeating: 0, count: M * rows)
for f in stride(from: 0, to: M, by: 2) {
    let part = runWide("wide_na2_faithful", firstM: f)
    for m in f..<(f + 2) { for i in idxForInputRow(m) { refNA2[i] = part[i] } }
}
var refNA4 = [Float](repeating: 0, count: M * rows)
for f in stride(from: 0, to: M, by: 4) {
    let part = runWide("wide_na4_faithful", firstM: f)
    for m in f..<(f + 4) { for i in idxForInputRow(m) { refNA4[i] = part[i] } }
}
let allRows = (0..<M).flatMap { idxForInputRow($0) }
let (refUlp, _) = maxUlp(refNA2, refNA4, allRows)
print("\n--- reference cross-check: NA=2 against NA=4 on all 8 rows: max_ulp=\(refUlp) (must be 0) ---")

print("\n--- lane fidelity: lane m of an NA=N run must equal input row m under the shipped NA=2 helper ---")
var laneGate = true
var laneJSON: [String] = []
for n in 5...8 {
    let got = runWide("wide_na\(n)_faithful", firstM: 0)
    var worst = 0
    var perLane: [Int] = []
    for m in 0..<n {
        let (u, _) = maxUlp(got, refNA2, idxForInputRow(m))
        perLane.append(u)
        worst = max(worst, u)
    }
    print("NA=\(n) per-lane max_ulp against the shipped NA=2 helper: \(perLane)")
    var distinct = true
    for m in 0..<n {
        for p in (m + 1)..<n where idxForInputRow(m).map({ bits(got[$0]) }) == idxForInputRow(p).map({ bits(got[$0]) }) {
            distinct = false
        }
    }
    let ok = (worst == 0) && distinct
    if !ok { laneGate = false }
    print("NA=\(n): max_ulp_over_\(n)_lanes=\(worst) all_lanes_distinct=\(distinct)  \(ok ? "EXACT" : "MISMATCH")")
    laneJSON.append("\"\(n)\": {\"max_ulp\": \(worst), \"per_lane_max_ulp\": \(perLane), \"all_lanes_distinct\": \(distinct), \"exact\": \(ok)}")
}

// ---- Q3: positive controls. Every one MUST be caught. ----
print("\n--- positive controls: a deliberately wrong lane MUST be caught at every NA ---")
var controlsAllCaught = true
var controlJSON: [String] = []
for n in 5...8 {
    for (suffix, label) in [("swap", "swap lanes 0 and \(n - 1)"),
                            ("zerolast", "zero lane \(n - 1) activations"),
                            ("leak", "leak lane \(n - 2) into lane \(n - 1)")] {
        let bad = runWide("wide_na\(n)_\(suffix)", firstM: 0)
        var caughtLanes: [Int] = []
        for m in 0..<n {
            let (u, _) = maxUlp(bad, refNA2, idxForInputRow(m))
            if u != 0 { caughtLanes.append(m) }
        }
        let caught = !caughtLanes.isEmpty
        if !caught { controlsAllCaught = false }
        print("NA=\(n) \(suffix) (\(label)): caught=\(caught) offending_lanes=\(caughtLanes)")
        controlJSON.append("\"na\(n)_\(suffix)\": {\"caught\": \(caught), \"offending_lanes\": \(caughtLanes)}")
    }
}

let passed = laneGate && controlsAllCaught && layoutClean && refUlp == 0
print("\nE61_RUNG0_GATE lane_fidelity=\(laneGate ? "PASS" : "FAIL") positive_controls=\(controlsAllCaught ? "PASS" : "FAIL") layout_clean=\(layoutClean ? "PASS" : "FAIL") reference_cross_check=\(refUlp == 0 ? "PASS" : "FAIL")")

if !jsonPath.isEmpty {
    let json = """
    {
      "device": "\(device.name)",
      "probe": "research/e61_vec_probe.metal",
      "K": \(K), "rows": \(rows), "input_rows": \(M),
      "layout": {\(layoutJSON.joined(separator: ", "))},
      "reference_cross_check_na2_vs_na4_max_ulp": \(refUlp),
      "lane_fidelity": {\(laneJSON.joined(separator: ", "))},
      "positive_controls": {\(controlJSON.joined(separator: ", "))},
      "gate_passed": \(passed)
    }
    """
    try? FileManager.default.createDirectory(atPath: (jsonPath as NSString).deletingLastPathComponent,
                                             withIntermediateDirectories: true)
    try? json.write(toFile: jsonPath, atomically: true, encoding: .utf8)
    print("wrote \(jsonPath)")
}
exit(passed ? 0 : 1)
