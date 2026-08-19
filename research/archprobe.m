// Reports the Metal architecture string and GPU family that MLX uses to select
// kernel variants, plus the raw command-buffer budget environment as seen by a
// fresh process. Enumerates the device only; runs no compute and no timing.
//
//   clang -fobjc-arc -framework Metal -framework Foundation \
//         -o /tmp/archprobe research/archprobe.m && /tmp/archprobe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static void printEnv(const char *name) {
  const char *v = getenv(name);
  printf("  %-24s = %s\n", name, v ? v : "(unset)");
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }

    printf("device.name               = %s\n", dev.name.UTF8String);
    if (@available(macOS 14.0, *)) {
      printf("device.architecture.name  = %s\n",
             dev.architecture.name.UTF8String);
    } else {
      printf("device.architecture.name  = (needs macOS 14)\n");
    }

    struct {
      MTLGPUFamily family;
      const char *label;
    } families[] = {
        {MTLGPUFamilyApple7, "Apple7"},   {MTLGPUFamilyApple8, "Apple8"},
        {MTLGPUFamilyApple9, "Apple9"},   {MTLGPUFamilyMetal3, "Metal3"},
        {MTLGPUFamilyCommon3, "Common3"},
    };
    printf("gpu_families              =");
    for (size_t i = 0; i < sizeof(families) / sizeof(families[0]); i++) {
      if ([dev supportsFamily:families[i].family]) {
        printf(" %s", families[i].label);
      }
    }
    printf("\n");

    printf("max_threadgroup_threads   = %lu\n",
           (unsigned long)dev.maxThreadsPerThreadgroup.width);
    printf("threadgroup_memory_bytes  = %lu\n",
           (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("recommended_wset_bytes    = %llu\n",
           (unsigned long long)dev.recommendedMaxWorkingSetSize);
    printf("has_unified_memory        = %d\n", (int)dev.hasUnifiedMemory);
    printf("location_number           = %lu\n", (unsigned long)dev.locationNumber);

    printf("environment as seen by a fresh process:\n");
    printEnv("MLX_MAX_MB_PER_BUFFER");
    printEnv("MLX_MAX_OPS_PER_BUFFER");
    printEnv("DARKBLOOM_STARTUP_MEMORY_PROFILE");
    printEnv("MLXFAST_LOCAL_COOL_GATE");
  }
  return 0;
}
