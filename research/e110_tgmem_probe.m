// E110 rung 2 step 1: driver reading of the COMPOSED qmv entry point.
//
// Compiles each source with the same options MLX uses for a JIT library
// (fast math off, newest available language version), builds a pipeline state
// for every entry point it declares, and prints the driver's own
// staticThreadgroupMemoryLength and maxTotalThreadsPerThreadgroup.
//
// No kernel is dispatched, so this probe holds no model and does no GPU work
// beyond pipeline creation.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e110_tgmem_probe research/e110_tgmem_probe.m
//   /tmp/e110_tgmem_probe LABEL=PATH [LABEL=PATH ...]
//
// Research-only: nothing here is on the scored path.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "usage: e110_tgmem_probe LABEL=PATH [LABEL=PATH ...]\n");
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e110_tgmem_probe: no Metal device\n");
      return 1;
    }
    printf("{\n  \"device\": \"%s\",\n", [[device name] UTF8String]);
    printf("  \"max_threadgroup_memory_length\": %lu,\n",
           (unsigned long)device.maxThreadgroupMemoryLength);
    printf("  \"variants\": [\n");

    for (int i = 1; i < argc; i++) {
      const char *arg = argv[i];
      const char *eq = strchr(arg, '=');
      if (!eq) {
        fprintf(stderr, "e110_tgmem_probe: expected LABEL=PATH, got %s\n", arg);
        return 2;
      }
      NSString *label = [[NSString alloc] initWithBytes:arg
                                                 length:(NSUInteger)(eq - arg)
                                               encoding:NSUTF8StringEncoding];
      NSString *path = @(eq + 1);
      NSError *err = nil;
      NSString *src = [NSString stringWithContentsOfFile:path
                                                encoding:NSUTF8StringEncoding
                                                   error:&err];
      if (!src) {
        fprintf(stderr, "e110_tgmem_probe: cannot read %s\n", [path UTF8String]);
        return 1;
      }
      MTLCompileOptions *opts = [MTLCompileOptions new];
      if (@available(macOS 26.0, *)) {
        opts.languageVersion = MTLLanguageVersion4_0;
      } else {
        opts.languageVersion = MTLLanguageVersion3_1;
      }
      // MLX builds every JIT library with fast math off
      // (metal/device.cpp Device::build_library_).
      [opts setFastMathEnabled:NO];
      id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                options:opts
                                                  error:&err];
      if (!lib) {
        fprintf(stderr, "e110_tgmem_probe: %s failed to compile: %s\n",
                [label UTF8String],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      printf("    {\"label\": \"%s\", \"source\": \"%s\", \"entry_points\": [\n",
             [label UTF8String], [path UTF8String]);
      NSArray<NSString *> *names = [lib functionNames];
      names = [names sortedArrayUsingSelector:@selector(compare:)];
      for (NSUInteger f = 0; f < [names count]; f++) {
        NSString *name = names[f];
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
          fprintf(stderr, "e110_tgmem_probe: %s/%s pipeline failed: %s\n",
                  [label UTF8String], [name UTF8String],
                  err ? [[err localizedDescription] UTF8String] : "unknown");
          return 1;
        }
        printf("      {\"name\": \"%s\", \"max_threads\": %lu, "
               "\"static_threadgroup_memory\": %lu, \"threads_per_simd\": %lu}%s\n",
               [name UTF8String],
               (unsigned long)pso.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso.staticThreadgroupMemoryLength,
               (unsigned long)pso.threadExecutionWidth,
               f + 1 == [names count] ? "" : ",");
      }
      printf("    ]}%s\n", i + 1 == argc ? "" : ",");
    }
    printf("  ]\n}\n");
  }
  return 0;
}
