// E72 rung 0: can this Mac emit or analyse GPU code for a generation other than
// the attached one, and can it report a *real* register allocation rather than
// an AIR-level proxy?
//
// Runs no timed compute. It compiles trivial kernels, builds pipeline states,
// and reads back whatever the Metal runtime will tell us about them.
//
//   clang -fobjc-arc -framework Metal -framework Foundation \
//         -o /tmp/e72_arch_probe research/e72_arch_probe.m && /tmp/e72_arch_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <stdio.h>

static void dumpClassProperties(const char *clsName) {
  Class cls = objc_getClass(clsName);
  if (!cls) {
    printf("  %s: CLASS NOT FOUND\n", clsName);
    return;
  }
  unsigned count = 0;
  objc_property_t *props = class_copyPropertyList(cls, &count);
  printf("  %s: %u properties\n", clsName, count);
  for (unsigned i = 0; i < count; ++i) {
    const char *name = property_getName(props[i]);
    const char *attrs = property_getAttributes(props[i]);
    printf("    %-44s %s\n", name, attrs);
  }
  free(props);

  unsigned mcount = 0;
  Method *methods = class_copyMethodList(cls, &mcount);
  printf("  %s: %u instance methods\n", clsName, mcount);
  for (unsigned i = 0; i < mcount; ++i) {
    const char *sel = sel_getName(method_getName(methods[i]));
    // Only the ones a family/arch override would plausibly be named after.
    if (strcasestr(sel, "arch") || strcasestr(sel, "family") ||
        strcasestr(sel, "target") || strcasestr(sel, "device") ||
        strcasestr(sel, "gpu") || strcasestr(sel, "platform") ||
        strcasestr(sel, "option") || strcasestr(sel, "compil")) {
      printf("    [sel] %s\n", sel);
    }
  }
  free(methods);
}

static NSString *kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"kernel void k_small(device float* o [[buffer(0)]],\n"
    @"                    device const float* a [[buffer(1)]],\n"
    @"                    uint i [[thread_position_in_grid]]) {\n"
    @"  float acc = 0;\n"
    @"  for (int j = 0; j < 8; ++j) acc = fma(a[i*8+j], a[i*8+j], acc);\n"
    @"  o[i] = acc;\n"
    @"}\n"
    @"kernel void k_big(device float* o [[buffer(0)]],\n"
    @"                  device const float* a [[buffer(1)]],\n"
    @"                  uint i [[thread_position_in_grid]]) {\n"
    @"  float acc[96];\n"
    @"  for (int j = 0; j < 96; ++j) acc[j] = a[i*96+j];\n"
    @"  for (int r = 0; r < 4; ++r)\n"
    @"    for (int j = 0; j < 96; ++j) acc[j] = fma(acc[j], acc[(j+7)%96], acc[(j+13)%96]);\n"
    @"  float s = 0;\n"
    @"  for (int j = 0; j < 96; ++j) s += acc[j];\n"
    @"  o[i] = s;\n"
    @"}\n";

int main(void) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("NO_METAL_DEVICE\n"); return 1; }
    printf("== attached device ==\n");
    printf("  name          = %s\n", [[dev name] UTF8String]);
    if (@available(macOS 14.0, *))
      printf("  architecture  = %s\n", [[[dev architecture] name] UTF8String]);
    printf("  maxThreadsPerThreadgroup = %lu x %lu x %lu\n",
           (unsigned long)dev.maxThreadsPerThreadgroup.width,
           (unsigned long)dev.maxThreadsPerThreadgroup.height,
           (unsigned long)dev.maxThreadsPerThreadgroup.depth);
    printf("  maxThreadgroupMemoryLength = %lu\n",
           (unsigned long)dev.maxThreadgroupMemoryLength);

    printf("\n== MTLDevice: any way to name a different architecture? ==\n");
    // MTLCopyAllDevices only reports physically present devices; there is no
    // public or private constructor that takes a target architecture.
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    printf("  MTLCopyAllDevices count = %lu\n", (unsigned long)all.count);
    for (id<MTLDevice> d in all) printf("    %s\n", [[d name] UTF8String]);
    if (@available(macOS 14.0, *)) {
      Class archCls = objc_getClass("MTLArchitecture");
      printf("  MTLArchitecture class present = %s\n", archCls ? "yes" : "no");
      dumpClassProperties("MTLArchitecture");
    }

    printf("\n== MTLCompileOptions surface (public + private) ==\n");
    dumpClassProperties("MTLCompileOptions");

    printf("\n== MTLComputePipelineDescriptor surface ==\n");
    dumpClassProperties("MTLComputePipelineDescriptor");

    printf("\n== real device register pressure proxy ==\n");
    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:opts error:&err];
    if (!lib) { printf("  COMPILE FAILED: %s\n", [[err description] UTF8String]); return 2; }
    for (NSString *fn in @[@"k_small", @"k_big"]) {
      id<MTLFunction> f = [lib newFunctionWithName:fn];
      MTLComputePipelineReflection *refl = nil;
      id<MTLComputePipelineState> ps =
          [dev newComputePipelineStateWithFunction:f
                                           options:MTLPipelineOptionBufferTypeInfo
                                        reflection:&refl
                                             error:&err];
      if (!ps) { printf("  %s: PIPELINE FAILED %s\n", [fn UTF8String], [[err description] UTF8String]); continue; }
      printf("  %-8s maxTotalThreadsPerThreadgroup=%lu  threadExecutionWidth=%lu  staticThreadgroupMemoryLength=%lu\n",
             [fn UTF8String],
             (unsigned long)ps.maxTotalThreadsPerThreadgroup,
             (unsigned long)ps.threadExecutionWidth,
             (unsigned long)ps.staticThreadgroupMemoryLength);
      dumpClassProperties(class_getName([ps class]));
    }

    printf("\n== MTLBinaryArchive: can it serialise for another device? ==\n");
    Class baCls = objc_getClass("MTLBinaryArchiveDescriptor");
    printf("  MTLBinaryArchiveDescriptor present = %s\n", baCls ? "yes" : "no");
    if (baCls) dumpClassProperties("MTLBinaryArchiveDescriptor");
    printf("  (an archive is produced by an id<MTLDevice>; there is no device for a\n"
           "   generation that is not attached, so this cannot target one.)\n");
  }
  return 0;
}
