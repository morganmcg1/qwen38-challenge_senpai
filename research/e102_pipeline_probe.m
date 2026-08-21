// E102 rung 1 arbiter: runtime MTLComputePipelineState properties, read from
// the SAME metallib that research/agx_crossarch.py hands to `metal-tt`.
//
// E76's probe compiled from source through `newLibraryWithSource:`, which uses
// the driver's default language version and its own option set. `metal-tt`
// reads a metallib built with `-std=metal4.0 -O2 -fno-fast-math`. Comparing a
// register count from one binary with a pipeline limit derived from a
// different binary is not a reconciliation, so this probe loads the metallib
// directly and both numbers describe one object.
//
// `maxTotalThreadsPerThreadgroup` is the only public field a register
// allocation can move: the runtime caps a threadgroup at the thread count whose
// registers fit. It is therefore the arbiter between the static instruments.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e102/pipeline research/e102_pipeline_probe.m
//   /tmp/e102/pipeline probe.metallib kernel_name [kernel_name ...]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: %s LIB.metallib FUNCTION [FUNCTION ...]\n",
              argv[0]);
      return 2;
    }
    NSError *err = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&err];
    if (!lib) {
      fprintf(stderr, "cannot load %s: %s\n", argv[1],
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }
    printf("DEVICE name=%s max_threads_per_threadgroup=%lu\n",
           [device.name UTF8String],
           (unsigned long)device.maxThreadsPerThreadgroup.width);
    for (int i = 2; i < argc; i++) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(argv[i])];
      if (!fn) {
        fprintf(stderr, "missing function %s\n", argv[i]);
        return 1;
      }
      MTLComputePipelineDescriptor *desc =
          [MTLComputePipelineDescriptor new];
      desc.computeFunction = fn;
      MTLComputePipelineReflection *reflection = nil;
      id<MTLComputePipelineState> pso = [device
          newComputePipelineStateWithDescriptor:desc
                                        options:MTLPipelineOptionBufferTypeInfo
                                     reflection:&reflection
                                          error:&err];
      if (!pso) {
        fprintf(stderr, "pipeline %s failed: %s\n", argv[i],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      printf("PIPELINE function=%s max_total_threads_per_threadgroup=%lu "
             "thread_execution_width=%lu static_threadgroup_memory_bytes=%lu\n",
             argv[i], (unsigned long)pso.maxTotalThreadsPerThreadgroup,
             (unsigned long)pso.threadExecutionWidth,
             (unsigned long)pso.staticThreadgroupMemoryLength);
    }
    return 0;
  }
}
