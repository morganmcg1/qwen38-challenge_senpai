// E73 rung 0, pipeline half: ask the toolchain what it will tell us about the
// occupancy of every legal `(M, IPG)` partition.
//
// This creates a compute pipeline state for each arm and reads the only
// occupancy figures Metal exposes publicly. It dispatches nothing and times
// nothing, so it is a compile probe and not a measurement.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e73/e73_pipeline_probe research/e73_pipeline_probe.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *source_path = NULL;
    const char *out_path = NULL;
    char *arm_list = NULL;
    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--source") && i + 1 < argc) source_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arm_list = strdup(argv[++i]);
      else {
        fprintf(stderr, "e73_pipeline_probe: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!source_path || !out_path || !arm_list) {
      fprintf(stderr,
              "e73_pipeline_probe: --source, --arms and --out are required\n");
      return 2;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *src = [NSString stringWithContentsOfFile:@(source_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e73_pipeline_probe: cannot read %s\n", source_path);
      return 1;
    }
    MTLCompileOptions *opts = [MTLCompileOptions new];
    if (@available(macOS 26.0, *)) {
      opts.languageVersion = MTLLanguageVersion4_0;
    } else {
      opts.languageVersion = MTLLanguageVersion3_1;
    }
    [opts setFastMathEnabled:NO];
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
      fprintf(stderr, "e73_pipeline_probe: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }

    NSMutableArray *rows = [NSMutableArray array];
    for (char *token = strtok(arm_list, ","); token; token = strtok(NULL, ",")) {
      NSString *fn_name = [NSString stringWithFormat:@"e73_cell_%s", token];
      id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
      if (!fn) {
        fprintf(stderr, "e73_pipeline_probe: missing function %s\n",
                [fn_name UTF8String]);
        return 1;
      }
      id<MTLComputePipelineState> pso =
          [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) {
        fprintf(stderr, "e73_pipeline_probe: pipeline %s failed: %s\n",
                [fn_name UTF8String],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      printf("ARM arm=%s max_total_threads_per_threadgroup=%lu "
             "static_threadgroup_memory_bytes=%lu thread_execution_width=%lu\n",
             token, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
             (unsigned long)pso.staticThreadgroupMemoryLength,
             (unsigned long)pso.threadExecutionWidth);
      fflush(stdout);
      [rows addObject:@{
        @"arm": @(token),
        @"max_total_threads_per_threadgroup":
            @(pso.maxTotalThreadsPerThreadgroup),
        @"static_threadgroup_memory_bytes": @(pso.staticThreadgroupMemoryLength),
        @"thread_execution_width": @(pso.threadExecutionWidth),
      }];
    }

    NSDictionary *report = @{
      @"experiment": @"e73",
      @"rung": @0,
      @"harness": @"compile-probe",
      @"device": device.name,
      @"arms": rows,
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:&err];
    [json writeToFile:@(out_path) atomically:YES];
    fprintf(stderr, "e73_pipeline_probe: wrote %s\n", out_path);
    return 0;
  }
}
