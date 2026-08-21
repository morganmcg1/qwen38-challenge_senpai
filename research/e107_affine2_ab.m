// E107 rung 0: isolated-dose harness for the coarse affine-2 draft readout.
//
// Measures, in one process and one queue, with palindrome ordering, at the
// exact scored cell (K = 5120, N = 98336, M = 1, grid (1, 12292, 1)
// threadgroups of (32, 2, 1)):
//
//   a_shipped   the transcribed shipped kernel
//   b_constw    weight loads removed, all arithmetic kept
//   c_loadonly  all loads kept, extract-and-fma removed
//   d_stream    idealised coalesced stream over the same byte counts
//
// The discriminator: if the readout is issue-rate bound, b_constw stays close
// to a_shipped and c_loadonly collapses. If it is bandwidth bound, b_constw
// collapses and c_loadonly stays close to a_shipped.
//
// Fidelity is checked before any timing: a_shipped is compared against a CPU
// reference over a row sample, with a positive control that perturbs one
// packed weight word and proves the comparison can fail.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e107_affine2_ab research/e107_affine2_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kK = 5120;          // in_vec_size
static const int kN = 98336;         // out_vec_size
static const int kGroup = 64;
static const int kRowsPerTG = 8;     // 2 simdgroups x 4 rows

enum { kArmA = 0, kArmB, kArmB2, kArmC, kArmE, kArmH, kArmF, kArmG,
       kArmI, kArmJ, kArmCount };
static const char *kArmNames[kArmCount] = {
    "a_shipped",  "b_constw",     "b2_maskalu", "c_loadonly",
    "e_floor",    "h_split",      "f_mask",     "g_bfe",
    "i_h_unroll", "j_f_nounroll"};
// Arms that keep every elementary product and the shipped summation order, so
// their whole output must be bit-identical to a_shipped.
static const int kExpectBitExact[kArmCount] = {1, 0, 0, 0, 0, 1, 1, 1,
                                               1, 1};

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

// CPU reference in the kernel's own summation order: per lane, 32 values into
// one float accumulator, five k-blocks into result[r], then a butterfly
// reduction over the 32 lanes.
static float referenceRow(const uint8_t *w, const uint16_t *scales,
                          const uint16_t *biases, const uint16_t *x, int row) {
  const int in_vec_size_w = kK / 4;
  const int in_vec_size_g = kK / kGroup;
  const int block_size = 32 * 32;
  float lane[32];
  for (int l = 0; l < 32; l++) lane[l] = 0.0f;
  for (int k = 0; k < kK; k += block_size) {
    for (int l = 0; l < 32; l++) {
      const uint8_t *ws = w + (size_t)row * in_vec_size_w + k / 4 + l * 8;
      uint64_t packed;
      memcpy(&packed, ws, 8);
      const int gi = row * in_vec_size_g + k / kGroup + (l * 32) / kGroup;
      const float scale = bf16_to_f32(scales[gi]);
      const float bias = bf16_to_f32(biases[gi]);
      const uint16_t *xm = x + k + l * 32;
      float xv[32], sum = 0.0f;
      for (int i = 0; i < 32; i += 4) {
        xv[i] = bf16_to_f32(xm[i]);
        xv[i + 1] = bf16_to_f32(xm[i + 1]);
        xv[i + 2] = bf16_to_f32(xm[i + 2]);
        xv[i + 3] = bf16_to_f32(xm[i + 3]);
        // The kernel adds these four in BFLOAT, not in float, so the
        // reference must round after every add or the affine bias term
        // disagrees in the last place.
        uint16_t t = f32_to_bf16(xv[i] + xv[i + 1]);
        t = f32_to_bf16(bf16_to_f32(t) + xv[i + 2]);
        t = f32_to_bf16(bf16_to_f32(t) + xv[i + 3]);
        sum += bf16_to_f32(t);
      }
      float accum = 0.0f;
      for (int j = 0; j < 32; j++) {
        accum += xv[j] * (float)((packed >> (2 * j)) & 0x03ull);
      }
      lane[l] += scale * accum + sum * bias;
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
                     id<MTLBuffer> __strong *w, int w_copies, id<MTLBuffer> scales,
                     id<MTLBuffer> biases, id<MTLBuffer> x, id<MTLBuffer> y,
                     uint64_t seed, int reps, int inner, int *slice) {
  const int in_vec_size = kK, out_vec_size = kN;
  MTLSize grid = MTLSizeMake(1, kN / kRowsPerTG, 1);
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
      [enc setBuffer:x offset:0 atIndex:3];
      [enc setBuffer:y offset:0 atIndex:4];
      [enc setBytes:&in_vec_size length:sizeof(int) atIndex:5];
      [enc setBytes:&out_vec_size length:sizeof(int) atIndex:6];
      [enc setBytes:&seed length:sizeof(uint64_t) atIndex:7];
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
                        id<MTLBuffer> sink, uint bytes, uint threadgroups,
                        int reps, int inner) {
  const uint n_vec4 = bytes / 16;
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
    const char *src_path = "research/e107_affine2_arms.metal";
    const char *out_path = NULL;
    int blocks = 8, w_copies = 2, inner = 8, reps = 2, check_rows = 96;
    uint stream_tgs = 2048;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--src") && i + 1 < argc) src_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--w-copies") && i + 1 < argc) w_copies = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--check-rows") && i + 1 < argc) check_rows = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--stream-tgs") && i + 1 < argc) stream_tgs = (uint)atoi(argv[++i]);
      else {
        fprintf(stderr, "e107_affine2_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path) {
      fprintf(stderr, "usage: e107_affine2_ab --out JSON [--src FILE] "
                      "[--blocks N] [--w-copies N] [--inner N] [--reps N] "
                      "[--check-rows N] [--stream-tgs N]\n");
      return 2;
    }

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;
    g_session_start = mach_absolute_time();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e107_affine2_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e107_affine2_ab: cannot read %s\n", src_path);
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
      fprintf(stderr, "e107_affine2_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }

    id<MTLComputePipelineState> pso[kArmCount];
    for (int a = 0; a < kArmCount; a++) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(kArmNames[a])];
      if (!fn) {
        fprintf(stderr, "e107_affine2_ab: no function %s\n", kArmNames[a]);
        return 1;
      }
      pso[a] = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso[a]) {
        fprintf(stderr, "e107_affine2_ab: pipeline %s failed: %s\n",
                kArmNames[a],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
    }
    id<MTLFunction> stream_fn = [lib newFunctionWithName:@"e107_d_stream"];
    id<MTLComputePipelineState> stream_pso =
        [device newComputePipelineStateWithFunction:stream_fn error:&err];
    if (!stream_pso) {
      fprintf(stderr, "e107_affine2_ab: stream pipeline failed\n");
      return 1;
    }

    const size_t w_bytes = (size_t)kN * kK / 4;
    const size_t g_count = (size_t)kN * (kK / kGroup);
    const size_t sb_bytes = g_count * 2;
    const size_t x_bytes = (size_t)kK * 2;
    const size_t y_bytes = (size_t)kN * 2;
    const size_t buffer_bytes = w_bytes + 2 * sb_bytes;

    id<MTLBuffer> w[8];
    if (w_copies < 1 || w_copies > 8) {
      fprintf(stderr, "e107_affine2_ab: --w-copies must be 1..8\n");
      return 2;
    }
    for (int c = 0; c < w_copies; c++) {
      w[c] = [device newBufferWithLength:w_bytes
                                 options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> scales = [device newBufferWithLength:sb_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [device newBufferWithLength:sb_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> x = [device newBufferWithLength:x_bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:y_bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> stream_src =
        [device newBufferWithLength:buffer_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [device newBufferWithLength:4096
                                             options:MTLResourceStorageModeShared];

    uint32_t rng = 0xC0FFEEu;
    uint8_t *w0 = (uint8_t *)w[0].contents;
    for (size_t i = 0; i < w_bytes; i += 4) {
      uint32_t v = xorshift32(&rng);
      memcpy(w0 + i, &v, 4);
    }
    for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, w_bytes);
    uint16_t *sp = (uint16_t *)scales.contents;
    uint16_t *bp = (uint16_t *)biases.contents;
    for (size_t i = 0; i < g_count; i++) {
      sp[i] = f32_to_bf16(0.01f + 0.001f * (float)(xorshift32(&rng) % 64));
      bp[i] = f32_to_bf16(-0.015f + 0.0005f * (float)(xorshift32(&rng) % 64));
    }
    uint16_t *xp = (uint16_t *)x.contents;
    for (int i = 0; i < kK; i++) {
      xp[i] = f32_to_bf16(((float)(xorshift32(&rng) % 2048) / 1024.0f) - 1.0f);
    }
    memcpy(stream_src.contents, w0, w_bytes);

    const uint64_t seed = 0x0123456789ABCDEFul;
    const int tg_count = kN / kRowsPerTG;

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e107_affine2_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out,
            "  \"k\": %d,\n  \"n\": %d,\n  \"group_size\": %d,\n"
            "  \"threadgroups\": %d,\n  \"threads_per_threadgroup\": 64,\n"
            "  \"rows_per_threadgroup\": %d,\n"
            "  \"weight_bytes\": %zu,\n  \"metadata_bytes\": %zu,\n"
            "  \"buffer_bytes\": %zu,\n  \"w_copies\": %d,\n"
            "  \"blocks\": %d,\n  \"inner\": %d,\n  \"reps\": %d,\n"
            "  \"order\": \"palindrome\",\n",
            kK, kN, kGroup, tg_count, kRowsPerTG, w_bytes, 2 * sb_bytes,
            buffer_bytes, w_copies, blocks, inner, reps);

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

    // --- fidelity, before any timing -------------------------------------
    {
      int slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, x, y, seed, 1, 1,
             &slice);
      const uint16_t *yp = (const uint16_t *)y.contents;
      double worst_rel = 0.0;
      int worst_row = -1, checked = 0;
      uint32_t pick = 0x1234567u;
      for (int i = 0; i < check_rows; i++) {
        int row = (i < check_rows / 2) ? i : (int)(xorshift32(&pick) % kN);
        const float want = referenceRow(w0, sp, bp, xp, row);
        const float got = bf16_to_f32(yp[row]);
        const double denom = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
        const double rel = fabs(want - got) / denom;
        if (rel > worst_rel) { worst_rel = rel; worst_row = row; }
        checked++;
      }
      fprintf(stderr, "e107_affine2_ab: fidelity rows=%d worst_rel=%.3e row=%d\n",
              checked, worst_rel, worst_row);
      fprintf(out,
              "  \"fidelity\": {\"rows_checked\": %d, \"worst_rel\": %.6e, "
              "\"worst_row\": %d, \"tolerance\": 2.0e-2, \"pass\": %s},\n",
              checked, worst_rel, worst_row, worst_rel < 2.0e-2 ? "true" : "false");

      // Positive control: perturb the packed weights of row 0, re-run, and
      // require the CLEAN reference to disagree with the new GPU value. Then
      // restore and require agreement to return.
      const size_t row_bytes = kK / 4;
      const float clean_ref = referenceRow(w0, sp, bp, xp, 0);
      uint8_t *saved = (uint8_t *)malloc(row_bytes);
      memcpy(saved, w0, row_bytes);
      for (size_t i = 0; i < row_bytes; i++) w0[i] = 0xFF;
      for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, row_bytes);
      slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, x, y, seed, 1, 1,
             &slice);
      const float pert_got = bf16_to_f32(((const uint16_t *)y.contents)[0]);
      const double ctrl_rel =
          fabs(clean_ref - pert_got) / fmax(fabs(clean_ref), 1e-6);
      memcpy(w0, saved, row_bytes);
      free(saved);
      for (int c = 1; c < w_copies; c++) memcpy(w[c].contents, w0, row_bytes);
      slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, x, y, seed, 1, 1,
             &slice);
      const float restored_got = bf16_to_f32(((const uint16_t *)y.contents)[0]);
      const double restored_rel =
          fabs(clean_ref - restored_got) / fmax(fabs(clean_ref), 1e-6);
      fprintf(stderr,
              "e107_affine2_ab: positive control perturbed_rel=%.3e "
              "restored_rel=%.3e\n", ctrl_rel, restored_rel);
      fprintf(out,
              "  \"positive_control\": {\"perturbed_rel\": %.6e, "
              "\"restored_rel\": %.6e, \"detected\": %s},\n",
              ctrl_rel, restored_rel,
              (ctrl_rel > 2.0e-2 && restored_rel < 2.0e-2) ? "true" : "false");
    }

    // --- arm-versus-shipped exactness, before any timing -------------------
    {
      uint16_t *ref = (uint16_t *)malloc(y_bytes);
      int slice = 0;
      runArm(queue, pso[kArmA], w, w_copies, scales, biases, x, y, seed, 1, 1,
             &slice);
      memcpy(ref, y.contents, y_bytes);
      fprintf(out, "  \"arm_exactness\": [\n");
      for (int a = 0; a < kArmCount; a++) {
        memset(y.contents, 0, y_bytes);
        slice = 0;
        runArm(queue, pso[a], w, w_copies, scales, biases, x, y, seed, 1, 1,
               &slice);
        const uint16_t *got = (const uint16_t *)y.contents;
        size_t differing = 0;
        double worst = 0.0;
        for (int i = 0; i < kN; i++) {
          if (got[i] != ref[i]) {
            differing++;
            const double e = fabs(bf16_to_f32(got[i]) - bf16_to_f32(ref[i]));
            if (e > worst) worst = e;
          }
        }
        const int ok = kExpectBitExact[a] ? (differing == 0) : 1;
        fprintf(stderr,
                "e107_affine2_ab: exactness %-10s differing=%zu/%d worst=%.3e%s\n",
                kArmNames[a], differing, kN, worst,
                (kExpectBitExact[a] && differing) ? "   <-- BIT-EXACT VIOLATION"
                                                  : "");
        fprintf(out,
                "%s    {\"arm\":\"%s\",\"expect_bit_exact\":%s,"
                "\"differing\":%zu,\"total\":%d,\"worst_abs\":%.6e,"
                "\"pass\":%s}",
                a ? ",\n" : "", kArmNames[a],
                kExpectBitExact[a] ? "true" : "false", differing, kN, worst,
                ok ? "true" : "false");
      }
      fprintf(out, "\n  ],\n");
      free(ref);
    }

    // --- warm, then time --------------------------------------------------
    int slice = 0;
    for (int a = 0; a < kArmCount; a++) {
      runArm(queue, pso[a], w, w_copies, scales, biases, x, y, seed, 1, inner,
             &slice);
    }
    runStream(queue, stream_pso, stream_src, sink, (uint)w_bytes, stream_tgs, 1,
              inner);

    fprintf(out, "  \"timing\": [\n");
    int first = 1;
    for (int b = 0; b < blocks; b++) {
      Timing t[2 * kArmCount];
      const double at = seconds_since(g_session_start);
      for (int s = 0; s < 2 * kArmCount; s++) {
        const int a = s < kArmCount ? s : (2 * kArmCount - 1 - s);
        t[s] = runArm(queue, pso[a], w, w_copies, scales, biases, x, y, seed,
                      reps, inner, &slice);
      }
      fprintf(stderr, "e107_affine2_ab: block %d", b);
      for (int a = 0; a < kArmCount; a++) {
        const double gpu = 0.5 * (t[a].gpu_us + t[2 * kArmCount - 1 - a].gpu_us);
        fprintf(stderr, "  %s=%.2fus", kArmNames[a], gpu);
      }
      fprintf(stderr, "\n");
      for (int a = 0; a < kArmCount; a++) {
        const double gpu = 0.5 * (t[a].gpu_us + t[2 * kArmCount - 1 - a].gpu_us);
        const double wall = 0.5 * (t[a].wall_us + t[2 * kArmCount - 1 - a].wall_us);
        fprintf(out,
                "%s    {\"arm\":\"%s\",\"block\":%d,\"session_elapsed_s\":%.3f,"
                "\"gpu_us\":%.4f,\"wall_us\":%.4f,\"slot_lo_gpu_us\":%.4f,"
                "\"slot_hi_gpu_us\":%.4f}",
                first ? "" : ",\n", kArmNames[a], b, at, gpu, wall,
                t[a].gpu_us, t[2 * kArmCount - 1 - a].gpu_us);
        first = 0;
      }
    }
    fprintf(out, "\n  ],\n");

    // --- achievable bandwidth reference -----------------------------------
    fprintf(out, "  \"stream\": [\n");
    first = 1;
    const size_t stream_bytes[2] = {w_bytes, buffer_bytes};
    const char *stream_names[2] = {"weights_only", "weights_plus_metadata"};
    for (int b = 0; b < blocks; b++) {
      for (int s = 0; s < 2; s++) {
        const Timing t = runStream(queue, stream_pso, stream_src, sink,
                                   (uint)stream_bytes[s], stream_tgs, reps,
                                   inner);
        const double gbps = (double)stream_bytes[s] / (t.gpu_us * 1e-6) / 1e9;
        if (b == 0) {
          fprintf(stderr, "e107_affine2_ab: stream %s %.2fus %.1f GB/s\n",
                  stream_names[s], t.gpu_us, gbps);
        }
        fprintf(out,
                "%s    {\"range\":\"%s\",\"block\":%d,\"bytes\":%zu,"
                "\"gpu_us\":%.4f,\"wall_us\":%.4f,\"gbps\":%.3f}",
                first ? "" : ",\n", stream_names[s], b, stream_bytes[s],
                t.gpu_us, t.wall_us, gbps);
        first = 0;
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e107_affine2_ab: wrote %s\n", out_path);
  }
  return 0;
}
