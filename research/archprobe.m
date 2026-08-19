// Reports the Metal architecture string and the MLX selections keyed off it.
// Enumerates the device only; runs no compute and no timing.
//
//   clang -fobjc-arc -framework Metal -framework Foundation \
//         -o /tmp/archprobe research/archprobe.m && /tmp/archprobe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void printEnv(const char *name) {
  const char *v = getenv(name);
  printf("  %-32s = %s\n", name, v ? v : "(unset)");
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) {
      printf("NO_METAL_DEVICE\n");
      return 1;
    }
    NSString *name = [d name];
    printf("device.name         = %s\n", [name UTF8String]);
    if (@available(macOS 14.0, *)) {
      NSString *arch = [[d architecture] name];
      const char *a = [arch UTF8String];
      size_t n = strlen(a);
      printf("architecture.name   = %s\n", a);
      if (n >= 3) {
        int tens = a[n - 3] - '0';
        int ones = a[n - 2] - '0';
        int gen = tens * 10 + ones;
        char back = a[n - 1];
        printf("MLX arch_gen        = %d\n", gen);
        printf("MLX devc (back)     = '%c'\n", back);
        int two_pass = (back == 'd' || back == 's');
        printf("SDPA 2pass at KV>=1024 on this box? %s\n",
               two_pass ? "YES" : "NO");

        // device.cpp:924-926 gates the whole _nax kernel family on the
        // architecture generation, so a box below the threshold can never
        // execute the variants the ranked M5 is documented to use.
        int nax_threshold = (back == 'p') ? 18 : 17;
        printf("_nax gen threshold  = %d\n", nax_threshold);
        printf("_nax available (arch gate only)? %s\n",
               gen >= nax_threshold ? "YES" : "NO");

        // device.cpp:572-595 picks the command-buffer budget from `back`.
        int def_ops = 40, def_mb = 40;
        switch (back) {
          case 'p': def_ops = 20; def_mb = 40; break;
          case 'g': def_ops = 40; def_mb = 40; break;
          case 's': case 'd': def_ops = 50; def_mb = 50; break;
        }
        printf("MLX stock budget    = %d ops / %d mebi-elements\n",
               def_ops, def_mb);
      }
    } else {
      printf("architecture unavailable on this OS\n");
    }

    printf("threadgroup_memory_bytes = %lu\n",
           (unsigned long)d.maxThreadgroupMemoryLength);
    printf("recommended_wset_bytes   = %llu\n",
           (unsigned long long)d.recommendedMaxWorkingSetSize);
    printf("has_unified_memory       = %d\n", (int)d.hasUnifiedMemory);

    printf("environment as seen by this process:\n");
    printEnv("MLX_MAX_MB_PER_BUFFER");
    printEnv("MLX_MAX_OPS_PER_BUFFER");
    printEnv("MLX_METAL_GPU_ARCH");
    printEnv("DARKBLOOM_STARTUP_MEMORY_PROFILE");
  }
  return 0;
}
