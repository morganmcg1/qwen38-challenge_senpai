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
//   BIN [buffer_gib] [iterations]
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
// Enough threads to saturate every core without making the grid-stride loop
// so short that launch overhead dominates.
let threadgroups = 1024
let totalThreads = threadgroups * threadgroupSize
guard let sink = device.makeBuffer(length: totalThreads * 4,
                                   options: .storageModePrivate) else {
    fail("could not allocate the sink")
}

var count = vectorCount
var best = 0.0
var samples: [Double] = []

for iteration in 0 ..< iterations {
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
    let gbps = Double(bytes) / seconds / 1e9
    samples.append(gbps)
    best = max(best, gbps)
    // The first pass warms the residency and the pipeline, so it is reported
    // but never counted as the ceiling.
    if iteration == 0 { best = 0.0 }
}

guard !samples.isEmpty else { fail("no timed samples") }
let counted = samples.count > 1 ? Array(samples.dropFirst()) : samples
let mean = counted.reduce(0, +) / Double(counted.count)
let sorted = counted.sorted()
let median = sorted[sorted.count / 2]

let report: [String: Any] = [
    "probe": "e154-host-achievable-read-bandwidth",
    "harness": "local",
    "official_or_ranked_score": false,
    "device": device.name,
    "buffer_bytes": bytes,
    "buffer_gib": Double(bytes) / 1024 / 1024 / 1024,
    "iterations": iterations,
    "threadgroups": threadgroups,
    "threads_per_threadgroup": threadgroupSize,
    "access_pattern": "grid-stride uint4, fully coalesced, XOR reduction",
    "warmup_gbps": samples[0],
    "e154_host_achievable_read_gbps": best,
    "mean_gbps_excluding_warmup": mean,
    "median_gbps_excluding_warmup": median,
    "all_gbps": samples,
]
let data = try JSONSerialization.data(withJSONObject: report,
                                      options: [.prettyPrinted,
                                                .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
