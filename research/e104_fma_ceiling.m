// E104 rung 2 control: what arithmetic rate can this GPU actually retire?
//
// The rung 1 sweep shows the quantized matvec becomes arithmetic-bound above
// NA=4 at roughly 2.8-3.0 TFLOP/s of useful multiply-accumulate. That number
// only means something against the machine's own ceiling, so this measures it
// with a memory-free kernel: many independent accumulator chains fed by
// registers, compiled exactly the way MLX compiles its JIT libraries.
//
// Two forms are timed. "mul_add" writes `acc = acc * b + c`, which is what the
// quantized kernel writes and what the optimizer sees. "fma" calls
// metal::fma() explicitly. Comparing them prices fused against unfused issue
// on this hardware without guessing from an intermediate representation.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//       research/e104_fma_ceiling.m -o /tmp/e104_fma
//   /tmp/e104_fma

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

static const char *kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// CHAINS independent accumulators hide the ~4 cycle ALU latency, so the loop
// measures issue throughput rather than dependency stalls.
#define CHAINS 8

template <bool USE_FMA>
inline void ceiling_body(device float *out, uint gid, uint iters, float seed) {
  float acc[CHAINS];
  float b[CHAINS];
  float c[CHAINS];
  for (uint i = 0; i < CHAINS; ++i) {
    acc[i] = seed + float(i);
    b[i] = 1.0000001f + float(i) * 1e-7f;
    c[i] = 1e-6f * float(i + 1);
  }
  for (uint t = 0; t < iters; ++t) {
    for (uint i = 0; i < CHAINS; ++i) {
      if (USE_FMA) {
        acc[i] = fma(acc[i], b[i], c[i]);
      } else {
        acc[i] = acc[i] * b[i] + c[i];
      }
    }
  }
  float s = 0.0f;
  for (uint i = 0; i < CHAINS; ++i) s += acc[i];
  out[gid] = s;
}

[[kernel]] void ceiling_mul_add(device float *out [[buffer(0)]],
                                constant uint &iters [[buffer(1)]],
                                constant float &seed [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  ceiling_body<false>(out, gid, iters, seed);
}

[[kernel]] void ceiling_fma(device float *out [[buffer(0)]],
                            constant uint &iters [[buffer(1)]],
                            constant float &seed [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  ceiling_body<true>(out, gid, iters, seed);
}
)METAL";

static const int kChains = 8;

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    // Match MLX: Device::build_library_ sets fast math off.
    [opts setFastMathEnabled:NO];
    NSError *err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:@(kSource) options:opts error:&err];
    if (!lib) {
      fprintf(stderr, "compile failed: %s\n",
              err.localizedDescription.UTF8String);
      return 1;
    }

    const char *names[2] = {"ceiling_mul_add", "ceiling_fma"};
    id<MTLComputePipelineState> pso[2];
    for (int i = 0; i < 2; i++) {
      id<MTLFunction> f = [lib newFunctionWithName:@(names[i])];
      pso[i] = [device newComputePipelineStateWithFunction:f error:&err];
      if (!pso[i]) {
        fprintf(stderr, "pipeline %s failed: %s\n", names[i],
                err.localizedDescription.UTF8String);
        return 1;
      }
    }

    const uint32_t threads = 1u << 20;
    const uint32_t iters = 4096;
    id<MTLBuffer> out = [device newBufferWithLength:threads * sizeof(float)
                                            options:MTLResourceStorageModeShared];

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    printf("device: %s\n", device.name.UTF8String);
    printf("threads=%u iters=%u chains=%d\n\n", threads, iters, kChains);
    printf("%-16s %10s %12s %12s\n", "form", "ms", "GFLOP", "TFLOP/s");

    double best[2] = {0, 0};
    // Interleave the two forms so any drift affects both equally.
    for (int rep = 0; rep < 5; rep++) {
      for (int i = 0; i < 2; i++) {
        float seed = 1.0f;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso[i]];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBytes:&iters length:sizeof(iters) atIndex:1];
        [enc setBytes:&seed length:sizeof(seed) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        uint64_t t0 = mach_absolute_time();
        [cb commit];
        [cb waitUntilCompleted];
        double sec = (mach_absolute_time() - t0) * ns_per_tick * 1e-9;
        // One multiply-accumulate is two floating point operations.
        double flops = 2.0 * (double)threads * (double)iters * (double)kChains;
        double tflops = flops / sec / 1e12;
        if (tflops > best[i]) best[i] = tflops;
        if (rep >= 1) {
          printf("%-16s %10.3f %12.1f %12.3f\n", names[i], sec * 1e3,
                 flops / 1e9, tflops);
        }
      }
    }
    printf("\nbest mul_add = %.3f TFLOP/s\nbest fma     = %.3f TFLOP/s\n",
           best[0], best[1]);
    printf("fma / mul_add = %.3fx\n", best[1] / best[0]);
    return 0;
  }
}
