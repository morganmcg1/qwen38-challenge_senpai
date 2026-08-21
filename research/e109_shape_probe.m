// E109 rung 1: time one kernel family at several threadgroup shapes and say
// which of three mechanisms carries its intra-kernel residual.
//
// E105 rung 1 split each live dispatch family's per-dispatch interval into
// launch, memory and a residual. Two families are mostly residual and both
// run very narrow threadgroups:
//
//   GDN prework       11.36 us/dispatch, 6.92 us residual, 400 threadgroups
//                     of ONE simd group each
//   q/k norm + RoPE    9.17 us/dispatch, 5.80 us residual, 140 threadgroups
//                     of TWO simd groups each
//
// Folding the same total work into fewer, wider threadgroups separates the
// three candidate mechanisms, because they predict different curves:
//
//   H1 occupancy      time falls, then saturates once the machine is full
//   H2 dependent chain  time is flat: the critical path is inside one thread
//   H3 per-threadgroup granularity  time falls as 1/threadgroups
//
// The probe is spec-driven. research/e109_shape_arms.py writes the arm sources
// and a `spec_<family>.json` that names the buffers in binding order, the grid
// and the threadgroup of every arm. Nothing here is family-specific, so the
// same binary answers rung 1a and rung 1b.
//
// WHAT MAKES THE COMPARISON VALID.
//   * Every arm binds the SAME input buffers. Only the output buffers are
//     per-arm, so a difference can only come from kernel code.
//   * Arms are interleaved ABBA inside one session and the reported statistic
//     is a median over both directions, so monotone drift cancels to first
//     order.
//   * Every arm declared `exact_vs_arm0` is byte-compared against arm 0 after
//     timing. The fold is supposed to be bit-exact; the probe checks it rather
//     than asserting it.
//   * `dispatchThreads:` matches the live call. MLX custom kernels dispatch a
//     thread grid, not a threadgroup grid (E105 follow-up 4), so a probe that
//     used `dispatchThreadgroups:` would measure a different launch.
//
// Research-only: nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e109_shape_probe research/e109_shape_probe.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ARMS 12
#define MAX_BUFS 40

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);  // round-to-nearest-even
  return (uint16_t)(u >> 16);
}

static inline uint32_t xorshift32(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

static inline float unit(uint32_t *s) {
  return (float)(xorshift32(s) >> 8) / (float)(1u << 24);
}

static double seconds_since(uint64_t t0) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  uint64_t d = mach_absolute_time() - t0;
  return (double)d * (double)tb.numer / (double)tb.denom / 1e9;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

typedef struct {
  char name[64];
  int is_out;
  id<MTLBuffer> shared;          // inputs and constants
  id<MTLBuffer> per_arm[MAX_ARMS];  // outputs
  size_t length;
} Buf;

typedef struct {
  char name[64];
  char function[128];
  char source[256];
  MTLSize grid;
  MTLSize tg;
  int threadgroups;
  int simdgroups_per_tg;
  int shipped;
  int exact_vs_arm0;
  id<MTLComputePipelineState> pso;
  NSUInteger max_threads;
  NSUInteger exec_width;
  NSUInteger tg_mem;
} Arm;

static NSString *readTextFile(NSString *path) {
  NSError *err = nil;
  NSString *s = [NSString stringWithContentsOfFile:path
                                          encoding:NSUTF8StringEncoding
                                             error:&err];
  if (!s) {
    fprintf(stderr, "e109_shape_probe: cannot read %s: %s\n",
            [path UTF8String],
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  return s;
}

// `newLibraryWithSource:` has no include path, so the shared MLX preamble that
// every arm needs is substituted textually here. That keeps the arm files
// readable and keeps `xcrun metal -I <dir>` working as an independent syntax
// check of the same text.
static NSString *inlinePreamble(NSString *src, NSString *dir) {
  NSString *needle = @"#include \"mlx_preamble.h\"";
  if ([src rangeOfString:needle].location == NSNotFound) return src;
  NSString *pre =
      readTextFile([dir stringByAppendingPathComponent:@"mlx_preamble.h"]);
  pre = [pre stringByReplacingOccurrencesOfString:@"#pragma once"
                                       withString:@""];
  return [src stringByReplacingOccurrencesOfString:needle withString:pre];
}

static id number(NSDictionary *d, NSString *k) {
  id v = d[k];
  if (!v) {
    fprintf(stderr, "e109_shape_probe: spec is missing %s\n", [k UTF8String]);
    exit(1);
  }
  return v;
}

static void fillBuffer(Buf *b, NSDictionary *spec, uint32_t *seed) {
  NSString *kind = spec[@"kind"];
  if ([kind isEqualToString:@"i64"] || [kind isEqualToString:@"i32"] ||
      [kind isEqualToString:@"f32"]) {
    NSArray *values = number(spec, @"values");
    if ([kind isEqualToString:@"i64"]) {
      int64_t *p = (int64_t *)b->shared.contents;
      for (NSUInteger i = 0; i < values.count; i++)
        p[i] = (int64_t)[values[i] longLongValue];
    } else if ([kind isEqualToString:@"i32"]) {
      int32_t *p = (int32_t *)b->shared.contents;
      for (NSUInteger i = 0; i < values.count; i++)
        p[i] = (int32_t)[values[i] intValue];
    } else {
      float *p = (float *)b->shared.contents;
      for (NSUInteger i = 0; i < values.count; i++)
        p[i] = [values[i] floatValue];
    }
    return;
  }
  // Data inputs. The residual under test is latency, not values, so the only
  // requirement is that the arithmetic stays finite: exp, log1p and the RMS
  // reciprocal all behave on [-1, 1].
  NSString *dtype = spec[@"dtype"];
  size_t count = (size_t)[number(spec, @"count") unsignedLongLongValue];
  if ([dtype isEqualToString:@"bf16"]) {
    uint16_t *p = (uint16_t *)b->shared.contents;
    for (size_t i = 0; i < count; i++)
      p[i] = f32_to_bf16(2.0f * unit(seed) - 1.0f);
  } else {
    float *p = (float *)b->shared.contents;
    for (size_t i = 0; i < count; i++) p[i] = 2.0f * unit(seed) - 1.0f;
  }
}

static size_t bufferBytes(NSDictionary *spec) {
  NSString *kind = spec[@"kind"];
  if ([kind isEqualToString:@"i64"])
    return 8 * (size_t)[(NSArray *)number(spec, @"values") count];
  if ([kind isEqualToString:@"i32"] || [kind isEqualToString:@"f32"])
    return 4 * (size_t)[(NSArray *)number(spec, @"values") count];
  size_t count = (size_t)[number(spec, @"count") unsignedLongLongValue];
  size_t elem = [(NSString *)spec[@"dtype"] isEqualToString:@"bf16"] ? 2 : 4;
  return count * elem;
}

static void encode(id<MTLComputeCommandEncoder> enc, Arm *arm, Buf *bufs,
                   int nbuf, int arm_index) {
  [enc setComputePipelineState:arm->pso];
  for (int i = 0; i < nbuf; i++) {
    id<MTLBuffer> b = bufs[i].is_out ? bufs[i].per_arm[arm_index]
                                     : bufs[i].shared;
    [enc setBuffer:b offset:0 atIndex:(NSUInteger)i];
  }
  [enc dispatchThreads:arm->grid threadsPerThreadgroup:arm->tg];
}

// One measurement is `inner` back-to-back dispatches inside a single command
// buffer, which is how the live decode round issues them, and it amortises the
// command-buffer cost that is NOT under test here.
static double timeArm(id<MTLCommandQueue> q, Arm *arm, Buf *bufs, int nbuf,
                      int arm_index, int inner) {
  uint64_t t0 = mach_absolute_time();
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  for (int i = 0; i < inner; i++) encode(enc, arm, bufs, nbuf, arm_index);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  return seconds_since(t0) / (double)inner;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *spec_path = NULL;
    const char *out_path = NULL;
    int reps = 24, inner = 32;
    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner = atoi(argv[++i]);
      else { fprintf(stderr, "e109_shape_probe: bad arg %s\n", argv[i]); return 1; }
    }
    if (!spec_path) { fprintf(stderr, "e109_shape_probe: --spec is required\n"); return 1; }

    NSString *specFile = [NSString stringWithUTF8String:spec_path];
    NSString *dir = [specFile stringByDeletingLastPathComponent];
    NSError *err = nil;
    NSDictionary *spec =
        [NSJSONSerialization JSONObjectWithData:[readTextFile(specFile)
                                                    dataUsingEncoding:NSUTF8StringEncoding]
                                        options:0
                                          error:&err];
    if (!spec) { fprintf(stderr, "e109_shape_probe: bad spec json\n"); return 1; }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSArray *bufspecs = number(spec, @"buffers");
    NSArray *armspecs = number(spec, @"arms");
    int nbuf = (int)bufspecs.count, narm = (int)armspecs.count;
    if (nbuf > MAX_BUFS || narm > MAX_ARMS) {
      fprintf(stderr, "e109_shape_probe: spec is too large\n"); return 1;
    }

    Arm arms[MAX_ARMS] = {0};
    for (int a = 0; a < narm; a++) {
      NSDictionary *s = armspecs[a];
      snprintf(arms[a].name, sizeof arms[a].name, "%s",
               [(NSString *)number(s, @"name") UTF8String]);
      snprintf(arms[a].function, sizeof arms[a].function, "%s",
               [(NSString *)number(s, @"function") UTF8String]);
      snprintf(arms[a].source, sizeof arms[a].source, "%s",
               [(NSString *)number(s, @"source") UTF8String]);
      NSArray *g = number(s, @"grid"), *t = number(s, @"threadgroup");
      arms[a].grid = MTLSizeMake([g[0] unsignedLongValue],
                                 [g[1] unsignedLongValue],
                                 [g[2] unsignedLongValue]);
      arms[a].tg = MTLSizeMake([t[0] unsignedLongValue],
                               [t[1] unsignedLongValue],
                               [t[2] unsignedLongValue]);
      arms[a].threadgroups = [number(s, @"threadgroups") intValue];
      arms[a].simdgroups_per_tg = [number(s, @"simdgroups_per_threadgroup") intValue];
      arms[a].shipped = [s[@"shipped"] boolValue];
      arms[a].exact_vs_arm0 = [s[@"exact_vs_arm0"] boolValue];

      NSString *src = inlinePreamble(
          readTextFile([dir stringByAppendingPathComponent:
                                [NSString stringWithUTF8String:arms[a].source]]),
          dir);
      // Same two options MLX sets for every JIT library
      // (backend/metal/device.cpp:631). Fast math off changes rounding, and
      // the language version changes codegen, so an arm compiled any other way
      // would not be the shipped kernel.
      MTLCompileOptions *opt = [MTLCompileOptions new];
      opt.fastMathEnabled = NO;
      if (__builtin_available(macOS 26, *)) {
        opt.languageVersion = MTLLanguageVersion4_0;
      } else if (__builtin_available(macOS 15, *)) {
        opt.languageVersion = MTLLanguageVersion3_2;
      } else {
        opt.languageVersion = MTLLanguageVersion3_1;
      }
      id<MTLLibrary> lib = [device newLibraryWithSource:src options:opt error:&err];
      if (!lib) {
        fprintf(stderr, "e109_shape_probe: %s failed to compile: %s\n",
                arms[a].name, [[err localizedDescription] UTF8String]);
        return 1;
      }
      id<MTLFunction> fn = [lib newFunctionWithName:
                                   [NSString stringWithUTF8String:arms[a].function]];
      if (!fn) {
        fprintf(stderr, "e109_shape_probe: %s has no function %s\n",
                arms[a].name, arms[a].function);
        return 1;
      }
      arms[a].pso = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!arms[a].pso) {
        fprintf(stderr, "e109_shape_probe: %s pipeline failed: %s\n",
                arms[a].name, [[err localizedDescription] UTF8String]);
        return 1;
      }
      arms[a].max_threads = arms[a].pso.maxTotalThreadsPerThreadgroup;
      arms[a].exec_width = arms[a].pso.threadExecutionWidth;
      arms[a].tg_mem = arms[a].pso.staticThreadgroupMemoryLength;
      NSUInteger want = arms[a].tg.width * arms[a].tg.height * arms[a].tg.depth;
      fprintf(stderr,
              "e109_shape_probe: %-6s %-24s tg=%lux%lux%lu (%lu thr) "
              "tgs=%d max=%lu width=%lu tgmem=%lu\n",
              arms[a].name, arms[a].function,
              (unsigned long)arms[a].tg.width, (unsigned long)arms[a].tg.height,
              (unsigned long)arms[a].tg.depth, (unsigned long)want,
              arms[a].threadgroups, (unsigned long)arms[a].max_threads,
              (unsigned long)arms[a].exec_width, (unsigned long)arms[a].tg_mem);
      if (want > arms[a].max_threads) {
        fprintf(stderr,
                "e109_shape_probe: %s wants %lu threads but the pipeline caps "
                "at %lu -- this arm is not runnable and is reported as such\n",
                arms[a].name, (unsigned long)want,
                (unsigned long)arms[a].max_threads);
        return 2;
      }
    }

    Buf bufs[MAX_BUFS] = {0};
    uint32_t seed = 0x9e3779b9u;
    for (int i = 0; i < nbuf; i++) {
      NSDictionary *s = bufspecs[i];
      snprintf(bufs[i].name, sizeof bufs[i].name, "%s",
               [(NSString *)number(s, @"name") UTF8String]);
      bufs[i].is_out = [(NSString *)number(s, @"kind") isEqualToString:@"out"];
      bufs[i].length = bufferBytes(s);
      if (bufs[i].is_out) {
        for (int a = 0; a < narm; a++) {
          bufs[i].per_arm[a] = [device newBufferWithLength:bufs[i].length
                                                   options:MTLResourceStorageModeShared];
          memset(bufs[i].per_arm[a].contents, 0, bufs[i].length);
        }
      } else {
        bufs[i].shared = [device newBufferWithLength:bufs[i].length
                                             options:MTLResourceStorageModeShared];
        fillBuffer(&bufs[i], s, &seed);
      }
    }

    for (int a = 0; a < narm; a++)
      for (int w = 0; w < 3; w++) timeArm(queue, &arms[a], bufs, nbuf, a, inner);

    double *samples = calloc((size_t)narm * (size_t)reps * 2, sizeof(double));
    for (int r = 0; r < reps; r++) {
      for (int a = 0; a < narm; a++)
        samples[a * reps * 2 + r * 2] = timeArm(queue, &arms[a], bufs, nbuf, a, inner);
      for (int a = narm - 1; a >= 0; a--)
        samples[a * reps * 2 + r * 2 + 1] = timeArm(queue, &arms[a], bufs, nbuf, a, inner);
    }

    // Correctness after timing, so a zeroed output cannot be timed by mistake.
    for (int a = 0; a < narm; a++) {
      for (int i = 0; i < nbuf; i++)
        if (bufs[i].is_out) memset(bufs[i].per_arm[a].contents, 0, bufs[i].length);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      encode(enc, &arms[a], bufs, nbuf, a);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }

    FILE *out = out_path ? fopen(out_path, "w") : stdout;
    fprintf(out, "{\n  \"family\": \"%s\",\n",
            [(NSString *)number(spec, @"family") UTF8String]);
    fprintf(out, "  \"device\": \"%s\",\n", [[device name] UTF8String]);
    fprintf(out, "  \"reps\": %d,\n  \"inner\": %d,\n", reps, inner);
    fprintf(out, "  \"dispatch\": \"dispatchThreads\",\n");
    fprintf(out, "  \"arms\": [\n");
    int all_exact = 1;
    for (int a = 0; a < narm; a++) {
      int n = reps * 2;
      double *v = samples + a * n;
      double *sorted = malloc((size_t)n * sizeof(double));
      memcpy(sorted, v, (size_t)n * sizeof(double));
      qsort(sorted, (size_t)n, sizeof(double), cmp_double);
      double median = (n % 2) ? sorted[n / 2]
                              : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
      double mean = 0.0;
      for (int i = 0; i < n; i++) mean += v[i];
      mean /= n;
      double var = 0.0;
      for (int i = 0; i < n; i++) var += (v[i] - mean) * (v[i] - mean);
      var = n > 1 ? var / (n - 1) : 0.0;

      int exact = 1, mismatch_bytes = 0;
      if (a > 0 && arms[a].exact_vs_arm0) {
        for (int i = 0; i < nbuf; i++) {
          if (!bufs[i].is_out) continue;
          const uint8_t *p = bufs[i].per_arm[a].contents;
          const uint8_t *q0 = bufs[i].per_arm[0].contents;
          for (size_t b = 0; b < bufs[i].length; b++)
            if (p[b] != q0[b]) { exact = 0; mismatch_bytes++; }
        }
        if (!exact) all_exact = 0;
      }
      fprintf(out,
              "    {\"name\": \"%s\", \"function\": \"%s\", "
              "\"threadgroups\": %d, \"simdgroups_per_threadgroup\": %d, "
              "\"threads_per_threadgroup\": %lu, \"shipped\": %s, "
              "\"max_total_threads\": %lu, \"thread_execution_width\": %lu, "
              "\"threadgroup_memory_bytes\": %lu, "
              "\"us_per_dispatch_median\": %.4f, \"us_per_dispatch_min\": %.4f, "
              "\"us_per_dispatch_mean\": %.4f, \"us_per_dispatch_sd\": %.4f, "
              "\"exact_vs_arm0\": %s, \"checked_exact\": %s, "
              "\"mismatch_bytes\": %d}%s\n",
              arms[a].name, arms[a].function, arms[a].threadgroups,
              arms[a].simdgroups_per_tg,
              (unsigned long)(arms[a].tg.width * arms[a].tg.height * arms[a].tg.depth),
              arms[a].shipped ? "true" : "false",
              (unsigned long)arms[a].max_threads,
              (unsigned long)arms[a].exec_width, (unsigned long)arms[a].tg_mem,
              median * 1e6, sorted[0] * 1e6, mean * 1e6, sqrt(var) * 1e6,
              exact ? "true" : "false",
              (a > 0 && arms[a].exact_vs_arm0) ? "true" : "false",
              mismatch_bytes, a + 1 == narm ? "" : ",");
      free(sorted);
    }
    fprintf(out, "  ],\n  \"all_folds_bit_exact\": %s\n}\n",
            all_exact ? "true" : "false");
    if (out != stdout) fclose(out);
    free(samples);
    return 0;
  }
}
