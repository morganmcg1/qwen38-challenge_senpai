// Report the register-pressure-limited occupancy of each crossrow packing factor.
//
// On Apple GPUs `maxTotalThreadsPerThreadgroup` is capped by the register budget
// the back end assigned to a kernel, so it is the cheapest direct readout of the
// register cliff that the AIR text cannot show.
//
//   xcrun metal -std=metal3.1 -O2 -c research/crossrow_na_probe.metal \
//     -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/probe.air
//   xcrun metallib /tmp/probe.air -o /tmp/probe.metallib
//   swiftc -O research/crossrow_na_occupancy.swift -o /tmp/na_occupancy
//   /tmp/na_occupancy /tmp/probe.metallib

import Foundation
import Metal

let metallibPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/probe.metallib"

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write(Data("no Metal device\n".utf8))
    exit(1)
}
let library = try device.makeLibrary(URL: URL(fileURLWithPath: metallibPath))

print("device=\(device.name)")
print("name              maxThreads  execWidth  tgMemBytes")
for name in library.functionNames.sorted() {
    guard let function = library.makeFunction(name: name) else { continue }
    let pipeline = try device.makeComputePipelineState(function: function)
    print(name.padding(toLength: 18, withPad: " ", startingAt: 0)
        + String(pipeline.maxTotalThreadsPerThreadgroup).padding(toLength: 12, withPad: " ", startingAt: 0)
        + String(pipeline.threadExecutionWidth).padding(toLength: 11, withPad: " ", startingAt: 0)
        + String(pipeline.staticThreadgroupMemoryLength))
}
