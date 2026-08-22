// E130 follow-up 1: MEASURE resident concurrency instead of deriving it.
//
// Every register-channel result in this campaign, including the arm E130 ships,
// reads residency out of the floor law
//
//     resident simdgroups = floor(REGISTER_BUDGET / registers_per_thread)
//
// The register count on the right is measured. The concurrency on the left is
// not: it is inferred. If the inference is wrong, a flat register ladder does
// not mean "occupancy does not buy time", it means "the register count never
// moved occupancy" and the whole axis is mispriced.
//
// e76 and e104 both attacked this with `maxTotalThreadsPerThreadgroup`, which
// is a driver-reported ceiling on ONE threadgroup, not a count of how many run
// at once. This probe counts the threadgroups that are actually resident
// together.
//
// METHOD. Each threadgroup registers itself in a device atomic on entry, spins
// on a dependent fma chain long enough that a whole wave overlaps, then
// deregisters. Thread 0 records the live count it observed at entry. The host
// takes the maximum over all threadgroups, which is the peak concurrent
// threadgroup count for the whole device.
//
// WHAT IT CAN AND CANNOT SAY. The number is device-wide: shader cores times
// per-core resident threadgroups. Absolute per-core simdgroups therefore need a
// core count this probe does not read. The floor law is still falsifiable
// without one, because the law predicts the RATIO between two register counts,
// and the ratio is core-count free.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//       research/e130_occupancy_probe.m -o /tmp/e130/occ
//   /tmp/e130/occ SOURCE.metal ITERS FUNCTION [FUNCTION ...]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static const uint32_t kThreadsPerGroup = 256;
static const uint32_t kGroups = 8192;
static const uint32_t kSeedFloats = 4096;
static const int kReplicates = 3;

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 4) {
      fprintf(stderr,
              "usage: %s SOURCE.metal ITERS FUNCTION [FUNCTION ...]\n",
              argv[0]);
      return 2;
    }
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (!source) {
      fprintf(stderr, "e130_occupancy_probe: cannot read %s\n", argv[1]);
      return 1;
    }
    int iters = atoi(argv[2]);
    if (iters <= 0) {
      fprintf(stderr, "e130_occupancy_probe: ITERS must be positive\n");
      return 2;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e130_occupancy_probe: no Metal device\n");
      return 2;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&err];
    if (!library) {
      fprintf(stderr, "e130_occupancy_probe: compile failed: %s\n",
              err.localizedDescription.UTF8String);
      return 1;
    }

    id<MTLBuffer> seed = [device newBufferWithLength:kSeedFloats * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    float *seedValues = (float *)seed.contents;
    for (uint32_t i = 0; i < kSeedFloats; ++i) {
      seedValues[i] = 1.0f + (float)(i % 17) * 1e-4f;
    }
    id<MTLBuffer> observed =
        [device newBufferWithLength:kGroups * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> live = [device newBufferWithLength:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

    fprintf(stdout, "device %s\n", device.name.UTF8String);
    fprintf(stdout, "threadgroups %u threads_per_group %u iters %d\n",
            kGroups, kThreadsPerGroup, iters);
    fprintf(stdout, "%-24s %8s %10s %12s %12s\n", "function", "maxTPTG",
            "peak_tg", "peak_thread", "seconds");

    NSMutableArray *rows = [NSMutableArray array];
    for (int a = 3; a < argc; ++a) {
      NSString *name = @(argv[a]);
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (!function) {
        fprintf(stderr, "e130_occupancy_probe: no function %s\n", argv[a]);
        return 1;
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&err];
      if (!pipeline) {
        fprintf(stderr, "e130_occupancy_probe: pipeline failed for %s: %s\n",
                argv[a], err.localizedDescription.UTF8String);
        return 1;
      }

      uint32_t peak = 0;
      double best = 0.0;
      for (int rep = 0; rep < kReplicates; ++rep) {
        memset(observed.contents, 0, kGroups * sizeof(uint32_t));
        *(uint32_t *)live.contents = 0;

        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:observed offset:0 atIndex:0];
        [encoder setBuffer:live offset:0 atIndex:1];
        [encoder setBuffer:seed offset:0 atIndex:2];
        int sink = -1;
        [encoder setBytes:&iters length:sizeof(iters) atIndex:3];
        [encoder setBytes:&sink length:sizeof(sink) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(kGroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {
          fprintf(stderr, "e130_occupancy_probe: %s failed: %s\n", argv[a],
                  buffer.error.localizedDescription.UTF8String);
          return 1;
        }

        double seconds = buffer.GPUEndTime - buffer.GPUStartTime;
        if (best == 0.0 || seconds < best) {
          best = seconds;
        }
        const uint32_t *values = (const uint32_t *)observed.contents;
        for (uint32_t g = 0; g < kGroups; ++g) {
          if (values[g] > peak) {
            peak = values[g];
          }
        }
      }

      uint32_t residual = *(uint32_t *)live.contents;
      if (residual != 0) {
        fprintf(stderr,
                "e130_occupancy_probe: %s left %u threadgroups registered; "
                "the entry and exit counts do not balance\n",
                argv[a], residual);
        return 1;
      }

      fprintf(stdout, "%-24s %8lu %10u %12u %12.6f\n", argv[a],
              (unsigned long)pipeline.maxTotalThreadsPerThreadgroup, peak,
              peak * kThreadsPerGroup, best);
      [rows addObject:@{
        @"function" : name,
        @"max_total_threads_per_threadgroup" :
            @(pipeline.maxTotalThreadsPerThreadgroup),
        @"thread_execution_width" : @(pipeline.threadExecutionWidth),
        @"static_threadgroup_memory_bytes" :
            @(pipeline.staticThreadgroupMemoryLength),
        @"peak_concurrent_threadgroups" : @(peak),
        @"peak_concurrent_threads" : @(peak * kThreadsPerGroup),
        @"seconds_min" : @(best),
      }];
    }

    NSDictionary *report = @{
      @"probe" : @"e130_occupancy_probe",
      @"device" : device.name,
      @"threadgroups_dispatched" : @(kGroups),
      @"threads_per_threadgroup" : @(kThreadsPerGroup),
      @"iters" : @(iters),
      @"replicates" : @(kReplicates),
      @"cells" : rows,
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingPrettyPrinted
                                                     error:&err];
    fprintf(stdout, "\nJSON %s\n", [[NSString alloc] initWithData:json
                                                         encoding:NSUTF8StringEncoding]
                                       .UTF8String);
    return 0;
  }
}
