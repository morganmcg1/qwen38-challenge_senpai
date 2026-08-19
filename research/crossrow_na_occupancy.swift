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
//   /tmp/na_occupancy /tmp/probe.metallib [name-substring-filter]
//
// The optional filter is required in practice on a full MLX metallib. Building a
// pipeline for a function that declares function constants raises a Metal
// validation assertion, which aborts the process instead of throwing, so it
// cannot be caught. Both the filter and the functionConstantsDictionary check
// keep such a function from ever reaching the pipeline call.

import Foundation
import Metal

let arguments = CommandLine.arguments
let metallibPath = arguments.count > 1 ? arguments[1] : "/tmp/probe.metallib"
let filter = arguments.count > 2 ? arguments[2] : nil

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write(Data("no Metal device\n".utf8))
    exit(1)
}
let library = try device.makeLibrary(URL: URL(fileURLWithPath: metallibPath))

print("device=\(device.name)")
print("metallib_function_total=\(library.functionNames.count)")
if let filter { print("filter=\(filter)") }
print("name maxThreads execWidth tgMemBytes")

var reported = 0
var skippedConstants = 0
for name in library.functionNames.sorted() {
    if let filter, !name.contains(filter) { continue }
    guard let function = library.makeFunction(name: name) else { continue }
    guard function.functionConstantsDictionary.isEmpty else {
        print("SKIP_FUNCTION_CONSTANTS \(name)")
        skippedConstants += 1
        continue
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    // Names are never truncated: an ambiguous kernel name would make the
    // arm-to-arm comparison meaningless.
    print("\(name) \(pipeline.maxTotalThreadsPerThreadgroup)"
        + " \(pipeline.threadExecutionWidth) \(pipeline.staticThreadgroupMemoryLength)")
    reported += 1
}
print("reported=\(reported) skipped_function_constants=\(skippedConstants)")
// An empty selection would make a downstream arm-to-arm diff pass trivially.
if reported == 0 {
    FileHandle.standardError.write(Data("no pipeline reported; filter matched nothing\n".utf8))
    exit(2)
}
