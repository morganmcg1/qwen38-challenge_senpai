// E80 rung 0c -- decide the GPU-timestamp mechanism from device evidence.
//
// Research-only. Prints, as JSON, what this host actually supports:
//   * MTLCounterSampleBuffer sampling points (stage / dispatch / draw / blit /
//     tile boundary), which decides option (C);
//   * the counter sets the device publishes, and whether `timestamp` is one;
//   * whether a real dispatch-boundary sample resolves to moving GPU ticks;
//   * whether MTLCommandBuffer GPUStartTime / GPUEndTime move, which decides
//     option (B);
//   * the CPU/GPU timestamp correlation from `sampleTimestamps:gpuTimestamp:`,
//     which is what converts GPU ticks to nanoseconds.
//
// clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//   research/e80_counter_probe.m -o /tmp/e80-build/e80_counter_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *kSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void e80_touch(device float *out [[buffer(0)]],\n"
    "                      uint gid [[thread_position_in_grid]]) {\n"
    "  float acc = out[gid];\n"
    "  for (int i = 0; i < 256; ++i) { acc = fma(acc, 1.000001f, 1.0f); }\n"
    "  out[gid] = acc;\n"
    "}\n";

static NSString *boolStr(BOOL value) { return value ? @"true" : @"false"; }

int main(int argc, const char **argv) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"device"] = device.name;

    out[@"supports_counter_sampling"] = @{
      @"at_stage_boundary" :
          boolStr([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
      @"at_draw_boundary" :
          boolStr([device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary]),
      @"at_dispatch_boundary" :
          boolStr([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
      @"at_tile_dispatch_boundary" :
          boolStr([device supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary]),
      @"at_blit_boundary" :
          boolStr([device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary]),
    };

    NSMutableArray *sets = [NSMutableArray array];
    id<MTLCounterSet> timestampSet = nil;
    for (id<MTLCounterSet> set in device.counterSets) {
      NSMutableArray *counters = [NSMutableArray array];
      for (id<MTLCounter> counter in set.counters) {
        [counters addObject:counter.name];
      }
      [sets addObject:@{@"name" : set.name, @"counters" : counters}];
      if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
        timestampSet = set;
      }
    }
    out[@"counter_sets"] = sets;
    out[@"has_timestamp_counter_set"] = boolStr(timestampSet != nil);

    // CPU/GPU correlation, the only supported tick -> nanosecond conversion.
    MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    usleep(20000);
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    out[@"timestamp_correlation"] = @{
      @"cpu_delta" : @(cpu1 - cpu0),
      @"gpu_delta" : @(gpu1 - gpu0),
      @"gpu_ticks_per_cpu_ns" :
          @(cpu1 > cpu0 ? (double)(gpu1 - gpu0) / (double)(cpu1 - cpu0) : 0.0),
    };

    NSError *error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                             options:nil
                               error:&error];
    if (library == nil) {
      out[@"library_error"] = error.localizedDescription;
      printf("%s\n", [[[NSString alloc]
          initWithData:[NSJSONSerialization dataWithJSONObject:out
                                                       options:NSJSONWritingPrettyPrinted
                                                         error:nil]
              encoding:NSUTF8StringEncoding] UTF8String]);
      return 1;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"e80_touch"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buffer = [device newBufferWithLength:4 * 1024 * 1024
                                               options:MTLResourceStorageModeShared];

    // (B) command-buffer GPUStartTime / GPUEndTime.
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buffer offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(1024 * 1024, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    out[@"command_buffer_gpu_time"] = @{
      @"gpu_start_time" : @(cb.GPUStartTime),
      @"gpu_end_time" : @(cb.GPUEndTime),
      @"gpu_seconds" : @(cb.GPUEndTime - cb.GPUStartTime),
      @"kernel_start_time" : @(cb.kernelStartTime),
      @"kernel_end_time" : @(cb.kernelEndTime),
      @"usable" : boolStr(cb.GPUEndTime > cb.GPUStartTime),
    };

    // (C) MTLCounterSampleBuffer at dispatch boundary, only if advertised.
    if (timestampSet != nil &&
        [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
      MTLCounterSampleBufferDescriptor *desc =
          [[MTLCounterSampleBufferDescriptor alloc] init];
      desc.counterSet = timestampSet;
      desc.sampleCount = 4;
      desc.storageMode = MTLStorageModeShared;
      desc.label = @"e80-dispatch-boundary";
      id<MTLCounterSampleBuffer> samples =
          [device newCounterSampleBufferWithDescriptor:desc error:&error];
      if (samples == nil) {
        out[@"dispatch_boundary_probe"] =
            @{@"ok" : @"false", @"error" : error.localizedDescription};
      } else {
        id<MTLCommandBuffer> cb2 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
        [enc2 setComputePipelineState:pipeline];
        [enc2 setBuffer:buffer offset:0 atIndex:0];
        [enc2 sampleCountersInBuffer:samples atSampleIndex:0 withBarrier:NO];
        [enc2 dispatchThreads:MTLSizeMake(1024 * 1024, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc2 sampleCountersInBuffer:samples atSampleIndex:1 withBarrier:NO];
        [enc2 endEncoding];
        [cb2 commit];
        [cb2 waitUntilCompleted];
        NSData *resolved = [samples resolveCounterRange:NSMakeRange(0, 2)];
        if (resolved == nil || resolved.length < 2 * sizeof(MTLCounterResultTimestamp)) {
          out[@"dispatch_boundary_probe"] = @{@"ok" : @"false", @"error" : @"resolve failed"};
        } else {
          const MTLCounterResultTimestamp *ts =
              (const MTLCounterResultTimestamp *)resolved.bytes;
          out[@"dispatch_boundary_probe"] = @{
            @"ok" : @"true",
            @"t0" : @(ts[0].timestamp),
            @"t1" : @(ts[1].timestamp),
            @"delta_ticks" : @(ts[1].timestamp - ts[0].timestamp),
            @"cb_gpu_seconds" : @(cb2.GPUEndTime - cb2.GPUStartTime),
          };
        }
      }
    } else {
      out[@"dispatch_boundary_probe"] =
          @{@"ok" : @"false", @"error" : @"atDispatchBoundary not supported"};
    }

    NSData *json = [NSJSONSerialization dataWithJSONObject:out
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    printf("%s\n", [[[NSString alloc] initWithData:json
                                          encoding:NSUTF8StringEncoding] UTF8String]);
    return 0;
  }
}
