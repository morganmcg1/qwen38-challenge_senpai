// E111 rung 1: isolated-dose harness for the one-byte affine-4 bias recoding.
//
// Measures, in one process and one queue, with palindrome ordering, at a
// scored linear shape of the wide affine-4 QMV as the entry switch selects it
// for M = 5: grid (5, N/8, 1) threadgroups of (32, 2, 1), NA = 5,
// DIRECT_NIBBLES = true, only tid.x == 0 doing work.
//
// Arms are described in research/e111_bias6_arms.metal. Fidelity runs before
// any timing:
//   1. a_shipped against a CPU reference in the kernel's own summation order,
//      with a perturbation control that proves the comparison can fail;
//   2. e_bias6 bit-identical to a_shipped over the whole output, with a
//      damaged-code control that proves that comparison can fail too.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e111/e111_bias6_ab research/e111_bias6_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kGroup = 64;
static const int kNA = 5;
static const int kRowsPerTG = 8;
static const int kValuesPerThread = 16;
static const int kBlockSize = 16 * 32;

enum { kArmA = 0, kArmNoBias, kArmNoSums, kArmBias1, kArmBias6, kArmConstW,
       kArmLoadOnly, kArmCount };
static const char *kArmNames[kArmCount] = {
    "a_shipped", "n_nobias", "n_nosums", "d_bias1",
    "e_bias6",   "b_constw", "c_loadonly"};
// Bytes of the 64-element group record each arm actually streams.
static const int kArmGroupBytes[kArmCount] = {36, 34, 36, 35, 35, 4, 36};
// Arms that keep every elementary product and the shipped summation order, so
// their whole output must be bit-identical to a_shipped.
static const int kExpectBitExact[kArmCount] = {1, 0, 0, 0, 1, 0, 0};

static int kK = 0, kN = 0;
static size_t kGroups = 0;

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline uint32_t xorshift32(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

static double g_ns_per_tick = 0.0;
static uint64_t g_session_start = 0;

static double seconds_since(uint64_t start) {
  return (double)(mach_absolute_time() - start) * g_ns_per_tick * 1e-9;
}

// CPU reference in the kernel's own summation order for one (row, m) output.
static float referenceCell(const uint8_t *w, const uint16_t *scales,
                           const uint16_t *biases, const uint16_t *x, int row,
                           int m) {
  const int in_vec_size_w = kK / 2;
  const int in_vec_size_g = kK / kGroup;
  float lane[32];
  for (int l = 0; l < 32; l++) lane[l] = 0.0f;
  for (int k = 0; k < kK; k += kBlockSize) {
    for (int l = 0; l < 32; l++) {
      uint16_t packed[4];
      memcpy(packed, w + (size_t)row * in_vec_size_w + k / 2 + l * 8, 8);
      const int gi = row * in_vec_size_g + k / 64 + l / 4;
      const float scale = bf16_to_f32(scales[gi]);
      const float bias = bf16_to_f32(biases[gi]);
      float sum = 0.0f, partial = 0.0f;
      for (int i = 0; i < 4; i++) {
        const uint16_t *xm = x + (size_t)m * kK + k + l * kValuesPerThread + 4 * i;
        const float x0 = bf16_to_f32(xm[0]), x1 = bf16_to_f32(xm[1]);
        const float x2 = bf16_to_f32(xm[2]), x3 = bf16_to_f32(xm[3]);
        // The kernel adds these four in BFLOAT, not in float, so the reference
        // must round after every add or the affine bias term disagrees in the
        // last place.
        uint16_t t = f32_to_bf16(x0 + x1);
        t = f32_to_bf16(bf16_to_f32(t) + x2);
        t = f32_to_bf16(bf16_to_f32(t) + x3);
        sum += bf16_to_f32(t);
        partial += x0 * (float)(packed[i] & 0x000f) +
                   x1 * (float)((packed[i] >> 4) & 0x000f) +
                   x2 * (float)((packed[i] >> 8) & 0x000f) +
                   x3 * (float)((packed[i] >> 12) & 0x000f);
      }
      lane[l] += scale * partial + sum * bias;
    }
  }
  for (int stride = 16; stride >= 1; stride >>= 1) {
    for (int l = 0; l < stride; l++) lane[l] += lane[l + stride];
  }
  return lane[0];
}

typedef struct {
  double wall_us;
  double gpu_us;
} Timing;

static Timing runArm(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                     id<MTLBuffer> __strong *w, int w_copies,
                     id<MTLBuffer> scales, id<MTLBuffer> biases,
                     id<MTLBuffer> codes, id<MTLBuffer> x, id<MTLBuffer> y,
                     uint64_t seed, int reps, int inner, int *slice) {
  const int in_vec_size = kK, out_vec_size = kN;
  MTLSize grid = MTLSizeMake(kNA, kN / kRowsPerTG, 1);
  MTLSize tg = MTLSizeMake(32, 2, 1);
  double gpu_total = 0.0;
  uint64_t t0 = mach_absolute_time();
  for (int rep = 0; rep < reps; rep++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      [enc setComputePipelineState:pso];
      [enc setBuffer:w[*slice] offset:0 atIndex:0];
      *slice = (*slice + 1) % w_copies;
      [enc setBuffer:scales offset:0 atIndex:1];
      [enc setBuffer:biases offset:0 atIndex:2];
      [enc setBuffer:codes offset:0 atIndex:3];
      [enc setBuffer:x offset:0 atIndex:4];
      [enc setBuffer:y offset:0 atIndex:5];
      [enc setBytes:&in_vec_size length:sizeof(int) atIndex:6];
      [enc setBytes:&out_vec_size length:sizeof(int) atIndex:7];
      [enc setBytes:&seed length:sizeof(uint64_t) atIndex:8];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    gpu_total += (cb.GPUEndTime - cb.GPUStartTime);
  }
  Timing t;
  t.wall_us = seconds_since(t0) / (double)(reps * inner) * 1e6;
  t.gpu_us = gpu_total / (double)(reps * inner) * 1e6;
  return t;
}

static Timing runStream(id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pso, id<MTLBuffer> src,
                        id<MTLBuffer> sink, size_t bytes, uint threadgroups,
                        int reps, int inner) {
  const uint n_vec4 = (uint)(bytes / 16);
  const uint tg_threads = 256;
  const uint total_threads = threadgroups * tg_threads;
  double gpu_total = 0.0;
  uint64_t t0 = mach_absolute_time();
  for (int rep = 0; rep < reps; rep++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      [enc setComputePipelineState:pso];
      [enc setBuffer:src offset:0 atIndex:0];
      [enc setBuffer:sink offset:0 atIndex:1];
      [enc setBytes:&n_vec4 length:sizeof(uint) atIndex:2];
      [enc setBytes:&total_threads length:sizeof(uint) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    gpu_total += (cb.GPUEndTime - cb.GPUStartTime);
  }
  Timing t;
  t.wall_us = seconds_since(t0) / (double)(reps * inner) * 1e6;
  t.gpu_us = gpu_total / (double)(reps * inner) * 1e6;
  return t;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *src_path = "research/e111_bias6_arms.metal";
    const char *blob_path = NULL;
    const char *out_path = NULL;
    const char *shape_name = "unknown";
    int blocks = 6, w_copies = 2, inner = 4, reps = 2, check_cells = 24;
    int warm = 3;
    uint stream_tgs = 2048;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--src") && i + 1 < argc) src_path = argv[++i];
      else if (!strcmp(argv[i], "--blob") && i + 1 < argc) blob_path = argv[++i];
      else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape_name = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-copies") && i + 1 < argc) w_copies = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--check-cells") && i + 1 < argc) check_cells = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--stream-tgs") && i + 1 < argc) stream_tgs = (uint)atoi(argv[++i]);
      else {
        fprintf(stderr, "e111_bias6_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path || !blob_path) {
      fprintf(stderr,
              "usage: e111_bias6_ab --blob BIN --out JSON [--shape NAME] "
              "[--src FILE] [--blocks N] [--w-copies N] [--inner N] "
              "[--reps N] [--check-cells N] [--warm N] [--stream-tgs N]\n");
      return 2;
    }

    // --- metadata blob ----------------------------------------------------
    FILE *bf = fopen(blob_path, "rb");
    if (!bf) {
      fprintf(stderr, "e111_bias6_ab: cannot read %s\n", blob_path);
      return 1;
    }
    uint32_t head[6];
    if (fread(head, sizeof(uint32_t), 6, bf) != 6 || head[0] != 0x31313145u) {
      fprintf(stderr, "e111_bias6_ab: bad blob header in %s\n", blob_path);
      return 1;
    }
    kK = (int)head[2];
    kN = (int)head[3];
    kGroups = (size_t)head[5];
    if ((size_t)kN * (kK / kGroup) != kGroups || kN % kRowsPerTG != 0 ||
        kK % kBlockSize != 0) {
      fprintf(stderr, "e111_bias6_ab: blob shape K=%d N=%d groups=%zu is not "
                      "a legal cell\n", kK, kN, kGroups);
      return 1;
    }

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;
    g_session_start = mach_absolute_time();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e111_bias6_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e111_bias6_ab: cannot read %s\n", src_path);
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
      fprintf(stderr, "e111_bias6_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }

    id<MTLComputePipelineState> pso[kArmCount];
    for (int a = 0; a < kArmCount; a++) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(kArmNames[a])];
      if (!fn) {
        fprintf(stderr, "e111_bias6_ab: no function %s\n", kArmNames[a]);
        return 1;
      }
      pso[a] = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso[a]) {
        fprintf(stderr, "e111_bias6_ab: pipeline %s failed: %s\n", kArmNames[a],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
    }
    id<MTLFunction> stream_fn = [lib newFunctionWithName:@"e111_stream"];
    id<MTLComputePipelineState> stream_pso =
        [device newComputePipelineStateWithFunction:stream_fn error:&err];
    if (!stream_pso) {
      fprintf(stderr, "e111_bias6_ab: stream pipeline failed\n");
      return 1;
    }

    const size_t w_bytes = (size_t)kN * (size_t)kK / 2;
    const size_t sb_bytes = kGroups * 2;
    const size_t code_bytes = kGroups;
    const size_t x_bytes = (size_t)kNA * kK * 2;
    const size_t y_bytes = (size_t)kNA * kN * 2;
    const size_t shipped_stream = kGroups * 36;

    if (w_copies < 1 || w_copies > 8) {
      fprintf(stderr, "e111_bias6_ab: --w-copies must be 1..8\n");
      return 2;
    }
    id<MTLBuffer> w[8];
    for (int c = 0; c < w_copies; c++) {
      w[c] = [device newBufferWithLength:w_bytes
                                 options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> scales = [device newBufferWithLength:sb_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [device newBufferWithLength:sb_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> codes = [device newBufferWithLength:code_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> x = [device newBufferWithLength:x_bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:y_bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [device newBufferWithLength:8192
                                             options:MTLResourceStorageModeShared];

    uint16_t *sp = (uint16_t *)scales.contents;
    uint16_t *bp = (uint16_t *)biases.contents;
    uint8_t *cp = (uint8_t *)codes.contents;
    if (fread(sp, 2, kGroups, bf) != kGroups ||
        fread(bp, 2, kGroups, bf) != kGroups ||
        fread(cp, 1, kGroups, bf) != kGroups) {
      fprintf(stderr, "e111_bias6_ab: short read on %s\n", blob_path);
      return 1;
    }
    fclose(bf);

    uint32_t rng = 0xC0FFEEu;
    uint8_t *w0 = (uint8_t *)w[0].contents;
    for (size_t i = 0; i < w_bytes; i += 4) {
      uint32_t v = xorshift32(&rng);
      memcpy(w0 + i, &v, 4);
    }
    for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, w_bytes);
    uint16_t *xp = (uint16_t *)x.contents;
    for (size_t i = 0; i < (size_t)kNA * kK; i++) {
      xp[i] = f32_to_bf16(((float)(xorshift32(&rng) % 2048) / 1024.0f) - 1.0f);
    }

    const uint64_t seed = 0x0123456789ABCDEFul;
    const int tg_count = kN / kRowsPerTG;

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e111_bias6_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out,
            "  \"shape\": \"%s\",\n  \"blob\": \"%s\",\n"
            "  \"k\": %d,\n  \"n\": %d,\n  \"group_size\": %d,\n"
            "  \"na\": %d,\n  \"grid_x\": %d,\n  \"threadgroups_y\": %d,\n"
            "  \"threads_per_threadgroup\": 64,\n  \"groups\": %zu,\n"
            "  \"weight_bytes\": %zu,\n  \"metadata_bytes\": %zu,\n"
            "  \"shipped_stream_bytes\": %zu,\n  \"w_copies\": %d,\n"
            "  \"blocks\": %d,\n  \"inner\": %d,\n  \"reps\": %d,\n"
            "  \"warm_passes\": %d,\n  \"order\": \"palindrome\",\n",
            shape_name, blob_path, kK, kN, kGroup, kNA, kNA, tg_count, kGroups,
            w_bytes, 2 * sb_bytes, shipped_stream, w_copies, blocks, inner, reps,
            warm);

    fprintf(out, "  \"pipelines\": {");
    for (int a = 0; a < kArmCount; a++) {
      fprintf(out,
              "%s\n    \"%s\": {\"max_threads\": %lu, \"exec_width\": %lu, "
              "\"tg_memory\": %lu}",
              a ? "," : "", kArmNames[a],
              (unsigned long)pso[a].maxTotalThreadsPerThreadgroup,
              (unsigned long)pso[a].threadExecutionWidth,
              (unsigned long)pso[a].staticThreadgroupMemoryLength);
    }
    fprintf(out, "\n  },\n");

    // --- fidelity, before any timing --------------------------------------
    {
      int slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, codes, x, y, seed,
             1, 1, &slice);
      const uint16_t *yp = (const uint16_t *)y.contents;
      double worst_rel = 0.0;
      int worst_row = -1, worst_m = -1, checked = 0;
      uint32_t pick = 0x1234567u;
      for (int i = 0; i < check_cells; i++) {
        const int row = (i < check_cells / 2) ? i : (int)(xorshift32(&pick) % kN);
        const int m = (int)(xorshift32(&pick) % kNA);
        const float want = referenceCell(w0, sp, bp, xp, row, m);
        const float got = bf16_to_f32(yp[(size_t)m * kN + row]);
        const double denom = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
        const double rel = fabs(want - got) / denom;
        if (rel > worst_rel) { worst_rel = rel; worst_row = row; worst_m = m; }
        checked++;
      }
      fprintf(stderr,
              "e111_bias6_ab: fidelity cells=%d worst_rel=%.3e row=%d m=%d\n",
              checked, worst_rel, worst_row, worst_m);
      fprintf(out,
              "  \"fidelity\": {\"cells_checked\": %d, \"worst_rel\": %.6e, "
              "\"worst_row\": %d, \"worst_m\": %d, \"tolerance\": 2.0e-2, "
              "\"pass\": %s},\n",
              checked, worst_rel, worst_row, worst_m,
              worst_rel < 2.0e-2 ? "true" : "false");

      // Positive control: perturb one packed weight row, re-run, and require
      // the clean reference to disagree. Then restore and require agreement.
      const size_t row_bytes = (size_t)kK / 2;
      const float clean_ref = referenceCell(w0, sp, bp, xp, 0, 0);
      uint8_t *saved = (uint8_t *)malloc(row_bytes);
      memcpy(saved, w0, row_bytes);
      for (size_t i = 0; i < row_bytes; i++) w0[i] = 0xFF;
      for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, row_bytes);
      slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, codes, x, y, seed,
             1, 1, &slice);
      const float pert_got = bf16_to_f32(((const uint16_t *)y.contents)[0]);
      const double ctrl_rel =
          fabs(clean_ref - pert_got) / fmax(fabs(clean_ref), 1e-6);
      memcpy(w0, saved, row_bytes);
      free(saved);
      for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, row_bytes);
      slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, codes, x, y, seed,
             1, 1, &slice);
      const float restored_got = bf16_to_f32(((const uint16_t *)y.contents)[0]);
      const double restored_rel =
          fabs(clean_ref - restored_got) / fmax(fabs(clean_ref), 1e-6);
      fprintf(stderr,
              "e111_bias6_ab: weight control perturbed_rel=%.3e "
              "restored_rel=%.3e\n", ctrl_rel, restored_rel);
      fprintf(out,
              "  \"weight_control\": {\"perturbed_rel\": %.6e, "
              "\"restored_rel\": %.6e, \"detected\": %s},\n",
              ctrl_rel, restored_rel,
              (ctrl_rel > 2.0e-2 && restored_rel < 2.0e-2) ? "true" : "false");
    }

    // --- arm-versus-shipped exactness, before any timing -------------------
    uint16_t *ref = (uint16_t *)malloc(y_bytes);
    {
      int slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, codes, x, y, seed,
             1, 1, &slice);
      memcpy(ref, y.contents, y_bytes);
      fprintf(out, "  \"arm_exactness\": [\n");
      for (int a = 0; a < kArmCount; a++) {
        memset(y.contents, 0, y_bytes);
        slice = 0;
        runArm(queue, pso[a], w, w_copies, scales, biases, codes, x, y, seed, 1,
               1, &slice);
        const uint16_t *got = (const uint16_t *)y.contents;
        size_t differing = 0;
        double worst = 0.0;
        for (size_t i = 0; i < y_bytes / 2; i++) {
          if (got[i] != ref[i]) {
            differing++;
            const double e = fabs(bf16_to_f32(got[i]) - bf16_to_f32(ref[i]));
            if (e > worst) worst = e;
          }
        }
        const int ok = kExpectBitExact[a] ? (differing == 0) : 1;
        fprintf(stderr,
                "e111_bias6_ab: exactness %-10s differing=%zu/%zu worst=%.3e%s\n",
                kArmNames[a], differing, y_bytes / 2, worst,
                (kExpectBitExact[a] && differing) ? "   <-- BIT-EXACT VIOLATION"
                                                  : "");
        fprintf(out,
                "%s    {\"arm\":\"%s\",\"expect_bit_exact\":%s,"
                "\"differing\":%zu,\"total\":%zu,\"worst_abs\":%.6e,"
                "\"pass\":%s}",
                a ? ",\n" : "", kArmNames[a],
                kExpectBitExact[a] ? "true" : "false", differing, y_bytes / 2,
                worst, ok ? "true" : "false");
      }
      fprintf(out, "\n  ],\n");
    }

    // --- code control: a damaged code must break e_bias6 -------------------
    {
      const uint8_t saved_code = cp[0];
      cp[0] ^= 0x01u;
      int slice = 0;
      memset(y.contents, 0, y_bytes);
      runArm(queue, pso[kArmBias6], w, w_copies, scales, biases, codes, x, y,
             seed, 1, 1, &slice);
      const uint16_t *got = (const uint16_t *)y.contents;
      size_t damaged_diff = 0;
      for (size_t i = 0; i < y_bytes / 2; i++) {
        if (got[i] != ref[i]) damaged_diff++;
      }
      cp[0] = saved_code;
      slice = 0;
      memset(y.contents, 0, y_bytes);
      runArm(queue, pso[kArmBias6], w, w_copies, scales, biases, codes, x, y,
             seed, 1, 1, &slice);
      got = (const uint16_t *)y.contents;
      size_t restored_diff = 0;
      for (size_t i = 0; i < y_bytes / 2; i++) {
        if (got[i] != ref[i]) restored_diff++;
      }
      fprintf(stderr,
              "e111_bias6_ab: code control damaged_diff=%zu restored_diff=%zu\n",
              damaged_diff, restored_diff);
      fprintf(out,
              "  \"code_control\": {\"damaged_differing\": %zu, "
              "\"restored_differing\": %zu, \"detected\": %s},\n",
              damaged_diff, restored_diff,
              (damaged_diff > 0 && restored_diff == 0) ? "true" : "false");
    }
    free(ref);

    // --- warm, then time --------------------------------------------------
    // The first dispatch of a session runs at a low clock. One warm pass is
    // not enough: the smoke run measured slot 0 at 1,394 us against 539 us for
    // the same arm in the mirror slot. Warm every arm several times and throw
    // the whole first palindrome away.
    int slice = 0;
    for (int pass = 0; pass < warm; pass++) {
      for (int a = 0; a < kArmCount; a++) {
        runArm(queue, pso[a], w, w_copies, scales, biases, codes, x, y, seed, 1,
               inner, &slice);
      }
    }
    for (int s = 0; s < 2 * kArmCount; s++) {
      const int a = s < kArmCount ? s : (2 * kArmCount - 1 - s);
      runArm(queue, pso[a], w, w_copies, scales, biases, codes, x, y, seed,
             reps, inner, &slice);
    }
    runStream(queue, stream_pso, w[0], sink, w_bytes, stream_tgs, 1, inner);

    fprintf(out, "  \"timing\": [\n");
    int first = 1;
    for (int b = 0; b < blocks; b++) {
      Timing t[2 * kArmCount];
      const double at = seconds_since(g_session_start);
      for (int s = 0; s < 2 * kArmCount; s++) {
        const int a = s < kArmCount ? s : (2 * kArmCount - 1 - s);
        t[s] = runArm(queue, pso[a], w, w_copies, scales, biases, codes, x, y,
                      seed, reps, inner, &slice);
      }
      fprintf(stderr, "e111_bias6_ab: block %d", b);
      for (int a = 0; a < kArmCount; a++) {
        const double gpu = 0.5 * (t[a].gpu_us + t[2 * kArmCount - 1 - a].gpu_us);
        fprintf(stderr, "  %s=%.2fus", kArmNames[a], gpu);
      }
      fprintf(stderr, "\n");
      for (int a = 0; a < kArmCount; a++) {
        const double gpu = 0.5 * (t[a].gpu_us + t[2 * kArmCount - 1 - a].gpu_us);
        const double wall = 0.5 * (t[a].wall_us + t[2 * kArmCount - 1 - a].wall_us);
        const double gbps =
            (double)(kGroups * (size_t)kArmGroupBytes[a]) / (gpu * 1e-6) / 1e9;
        fprintf(out,
                "%s    {\"arm\":\"%s\",\"block\":%d,\"session_elapsed_s\":%.3f,"
                "\"gpu_us\":%.4f,\"wall_us\":%.4f,\"group_bytes\":%d,"
                "\"stream_gbps\":%.3f,\"slot_lo_gpu_us\":%.4f,"
                "\"slot_hi_gpu_us\":%.4f}",
                first ? "" : ",\n", kArmNames[a], b, at, gpu, wall,
                kArmGroupBytes[a], gbps, t[a].gpu_us,
                t[2 * kArmCount - 1 - a].gpu_us);
        first = 0;
      }
    }
    fprintf(out, "\n  ],\n");

    // --- achievable bandwidth reference -----------------------------------
    fprintf(out, "  \"stream\": [\n");
    first = 1;
    for (int b = 0; b < blocks; b++) {
      const Timing t =
          runStream(queue, stream_pso, w[0], sink, w_bytes, stream_tgs, reps, inner);
      const double gbps = (double)w_bytes / (t.gpu_us * 1e-6) / 1e9;
      if (b == 0) {
        fprintf(stderr, "e111_bias6_ab: stream %.2fus %.1f GB/s\n", t.gpu_us, gbps);
      }
      fprintf(out,
              "%s    {\"block\":%d,\"bytes\":%zu,\"gpu_us\":%.4f,"
              "\"wall_us\":%.4f,\"gbps\":%.3f}",
              first ? "" : ",\n", b, w_bytes, t.gpu_us, t.wall_us, gbps);
      first = 0;
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e111_bias6_ab: wrote %s\n", out_path);
  }
  return 0;
}
