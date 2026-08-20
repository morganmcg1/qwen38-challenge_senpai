// E76 calibration: read every register-derived limit Metal will report for a
// compute pipeline, so the advisor's register-to-occupancy arithmetic can be
// checked against the runtime instead of assumed.
//
// `maxTotalThreadsPerThreadgroup` is the only public field that a register
// allocation can move: the runtime caps a threadgroup at the number of threads
// whose registers fit. Pairing it with the register count that
// research/agx_crossarch.py reads for the same kernel turns the pair into a
// measurement of the register file, IF the field moves at all.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e76/occupancy research/e76_occupancy_probe.m
//   /tmp/e76/occupancy source.metal k_s8 k_s16 ...

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: %s SOURCE.metal FUNCTION [FUNCTION ...]\n",
              argv[0]);
      return 2;
    }
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1])
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:options
                                                error:&err];
    if (!lib) {
      fprintf(stderr, "compile failed: %s\n",
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
      id<MTLComputePipelineState> pso =
          [device newComputePipelineStateWithFunction:fn error:&err];
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
