// E154 F4 item 3: what read bandwidth can this host actually reach?
//
// FINDING 281 identifies the M-independent 25,409 us term in the ranked round
// as one full read of the scored linear weights, 14.412 GB, which implies
// 567 GB/s. That identification is only admissible if a host we use can reach
// that rate. This probe measures the ceiling directly with a grid-stride
// streaming read, which is the friendliest possible access pattern, so the
// number it returns is an upper bound on anything the model can achieve.
//
// Standalone on purpose: it touches no submitted path and needs no package
// graph. Build and run:
//
//   swiftc -O research/e154_bandwidth_probe.swift -o BIN \
//     -framework Metal -framework Foundation
//   BIN [buffer_gib] [iterations] [output_json_path]
//
// harness=local. Never an official or ranked score.

import Foundation
import Metal

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

// Grid-stride so consecutive lanes touch consecutive 16-byte vectors. XOR
// rather than add: it cannot be strength-reduced or vectorised away, and it
// keeps every load live.
kernel void stream_read(device const uint4 *src [[buffer(0)]],
                        device uint *dst [[buffer(1)]],
                        constant uint64_t &vec_count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint gsize [[threads_per_grid]]) {
    uint4 acc = uint4(0);
    for (uint64_t i = gid; i < vec_count; i += gsize) {
        acc ^= src[i];
    }
    dst[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("e154-bwprobe: " + message + "\n")
        .utf8))
    exit(1)
}

let arguments = CommandLine.arguments
let bufferGiB = arguments.count > 1 ? (Double(arguments[1]) ?? 4.0) : 4.0
let iterations = arguments.count > 2 ? (Int(arguments[2]) ?? 12) : 12

guard let device = MTLCreateSystemDefaultDevice() else {
    fail("no Metal device")
}
guard let queue = device.makeCommandQueue() else {
    fail("no command queue")
}

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: kernelSource, options: nil)
} catch {
    fail("kernel compile failed: \(error)")
}
guard let function = library.makeFunction(name: "stream_read") else {
    fail("no stream_read function")
}
let pipeline: MTLComputePipelineState
do {
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("pipeline failed: \(error)")
}

let bytes = Int(bufferGiB * 1024 * 1024 * 1024) & ~15
let vectorCount = UInt64(bytes / 16)

// Private storage keeps the allocation device-local and stops the CPU from
// paging it. The contents are irrelevant; only the traffic matters.
guard let source = device.makeBuffer(length: bytes,
                                     options: .storageModePrivate) else {
    fail("could not allocate \(bytes) bytes")
}

let threadgroupSize = pipeline.maxTotalThreadsPerThreadgroup
// A ceiling claim is only as good as its saturation evidence, so the launch
// geometry is swept rather than assumed. If the best rate sits at an interior
// point the grid was wide enough; if it sits at the last point the sweep was
// too narrow and the number is a floor on the ceiling.
let threadgroupSweep = [256, 512, 1024, 2048, 4096]
var count = vectorCount
var best = 0.0
var bestThreadgroups = 0
var geometries: [[String: Any]] = []

for threadgroups in threadgroupSweep {
    let totalThreads = threadgroups * threadgroupSize
    guard let sink = device.makeBuffer(length: totalThreads * 4,
                                       options: .storageModePrivate) else {
        fail("could not allocate the sink")
    }
    var samples: [Double] = []
    for _ in 0 ..< iterations {
        guard let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else {
            fail("no encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(sink, offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt64>.size, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1,
                                           depth: 1))
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()

        if let error = buffer.error {
            fail("dispatch failed: \(error)")
        }
        let seconds = buffer.gpuEndTime - buffer.gpuStartTime
        guard seconds > 0 else { continue }
        samples.append(Double(bytes) / seconds / 1e9)
    }
    guard !samples.isEmpty else { continue }
    // The first pass of each geometry warms residency and the pipeline, so it
    // is reported but never counted as the ceiling.
    let counted = samples.count > 1 ? Array(samples.dropFirst()) : samples
    let peak = counted.max() ?? 0.0
    if peak > best {
        best = peak
        bestThreadgroups = threadgroups
    }
    let sorted = counted.sorted()
    geometries.append([
        "threadgroups": threadgroups,
        "threads": totalThreads,
        "warmup_gbps": samples[0],
        "peak_gbps": peak,
        "mean_gbps": counted.reduce(0, +) / Double(counted.count),
        "median_gbps": sorted[sorted.count / 2],
        "all_gbps": samples,
    ])
}

guard best > 0 else { fail("no timed samples") }

let report: [String: Any] = [
    "probe": "e154-host-achievable-read-bandwidth",
    "harness": "local",
    "official_or_ranked_score": false,
    "device": device.name,
    "buffer_bytes": bytes,
    "buffer_gib": Double(bytes) / 1024 / 1024 / 1024,
    "iterations_per_geometry": iterations,
    "threads_per_threadgroup": threadgroupSize,
    "access_pattern": "grid-stride uint4, fully coalesced, XOR reduction",
    "e154_host_achievable_read_gbps": best,
    "peak_at_threadgroups": bestThreadgroups,
    "saturated": bestThreadgroups != threadgroupSweep.last,
    "geometries": geometries,
]
let data = try JSONSerialization.data(withJSONObject: report,
                                      options: [.prettyPrinted,
                                                .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
if arguments.count > 3 {
    try data.write(to: URL(fileURLWithPath: arguments[3]))
}
