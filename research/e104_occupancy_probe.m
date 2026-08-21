// E104 rung 2: does register pressure cap occupancy as NA grows?
//
// The rung 1 sweep shows the load stream is flat in NA while the shipped
// kernel's time grows faster than either roofline floor. This probe asks the
// Metal compiler directly: for each arm and each NA, what is the largest
// threadgroup the driver will let us launch? On Apple GPUs that limit falls
// when a kernel needs more registers, so it is a direct read on occupancy.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//       research/e104_occupancy_probe.m -o /tmp/e104_occ
//   /tmp/e104_occ --dir /tmp/e104-arms

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *kArms[] = {"a_base", "l_loadonly", "z_noxload", "xw_widex"};
static const int kArmCount = 4;
static const int kWidths[] = {2, 3, 4, 5, 6};
static const int kWidthCount = 5;

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSString *dir = @"/tmp/e104-arms";
    for (int i = 1; i + 1 < argc; ++i) {
      if (strcmp(argv[i], "--dir") == 0) {
        dir = [NSString stringWithUTF8String:argv[i + 1]];
      }
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e104_occupancy_probe: no Metal device\n");
      return 2;
    }
    fprintf(stdout, "device: %s\n", device.name.UTF8String);
    fprintf(stdout, "max threads per threadgroup: %lu\n",
            (unsigned long)device.maxThreadsPerThreadgroup.width);
    fprintf(stdout, "\n%-12s %3s %10s %10s %12s\n", "arm", "NA", "maxTPTG",
            "execWidth", "tgMemBytes");

    NSMutableArray *rows = [NSMutableArray array];
    for (int a = 0; a < kArmCount; ++a) {
      NSString *path =
          [dir stringByAppendingPathComponent:
                   [NSString stringWithFormat:@"iso_%s.metal", kArms[a]]];
      NSError *err = nil;
      NSString *src = [NSString stringWithContentsOfFile:path
                                                encoding:NSUTF8StringEncoding
                                                   error:&err];
      if (!src) {
        fprintf(stderr, "e104_occupancy_probe: cannot read %s: %s\n",
                path.UTF8String, err.localizedDescription.UTF8String);
        return 2;
      }
      MTLCompileOptions *opts = [MTLCompileOptions new];
      opts.fastMathEnabled = NO;
      id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                options:opts
                                                  error:&err];
      if (!lib) {
        fprintf(stderr, "e104_occupancy_probe: compile failed for %s: %s\n",
                kArms[a], err.localizedDescription.UTF8String);
        return 2;
      }
      for (int w = 0; w < kWidthCount; ++w) {
        NSString *fn =
            [NSString stringWithFormat:@"e104_iso_na%d", kWidths[w]];
        id<MTLFunction> f = [lib newFunctionWithName:fn];
        if (!f) {
          fprintf(stderr, "e104_occupancy_probe: missing %s in %s\n",
                  fn.UTF8String, kArms[a]);
          return 2;
        }
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:f error:&err];
        if (!pso) {
          fprintf(stderr, "e104_occupancy_probe: pipeline failed %s %s: %s\n",
                  kArms[a], fn.UTF8String,
                  err.localizedDescription.UTF8String);
          return 2;
        }
        fprintf(stdout, "%-12s %3d %10lu %10lu %12lu\n", kArms[a], kWidths[w],
                (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                (unsigned long)pso.threadExecutionWidth,
                (unsigned long)pso.staticThreadgroupMemoryLength);
        [rows addObject:@{
          @"arm" : [NSString stringWithUTF8String:kArms[a]],
          @"na" : @(kWidths[w]),
          @"max_total_threads_per_threadgroup" :
              @(pso.maxTotalThreadsPerThreadgroup),
          @"thread_execution_width" : @(pso.threadExecutionWidth),
          @"static_threadgroup_memory_bytes" :
              @(pso.staticThreadgroupMemoryLength),
        }];
      }
    }

    for (int i = 1; i + 1 < argc; ++i) {
      if (strcmp(argv[i], "--out") == 0) {
        NSDictionary *doc = @{
          @"device" : device.name,
          @"device_max_threads_per_threadgroup" :
              @(device.maxThreadsPerThreadgroup.width),
          @"rows" : rows,
        };
        NSData *j = [NSJSONSerialization dataWithJSONObject:doc
                                                    options:NSJSONWritingPrettyPrinted
                                                      error:nil];
        [j writeToFile:[NSString stringWithUTF8String:argv[i + 1]]
            atomically:YES];
        fprintf(stdout, "\nwrote %s\n", argv[i + 1]);
      }
    }
    return 0;
  }
}
