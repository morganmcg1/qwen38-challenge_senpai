// E98 rung 1b: price the metadata byte and the LUT lookup inside the SCORED
// affine-4/g64 `qmv_fast` kernel, at the scored verify widths.
//
// Rung 1a used a group-size ladder and could only reach `qmv_fast_impl` at
// M = 1, because the wide cross-row family is gated on `group_size == 64`
// (quantized.h:1916). This harness instead compiles three variants of the
// runtime-effective JIT string in ONE process and alternates them inside one
// thermal session, so the cross-row kernel itself can be measured.
//
//   (a) shipped   scale = scales[g], bias = biases[g]           36 B / 64 elem
//   (b) indexed   idx = ushort(scales[g]); scale = lut[2*idx],
//                 bias = lut[2*idx+1], lut at the front of biases 34 B / 64 elem
//   (c) constant  literals, no metadata read at all              32 B / 64 elem
//
// Arms (a) and (b) get DIFFERENT operand buffers that encode the SAME
// (scale, bias) per group, so their outputs must be bit-identical. That makes
// this a correctness proof of the indexed dequantisation as well as a timing
// measurement. Arm (c) changes values on purpose and is a timing-only upper
// bound; it is never quoted as a correctness-bearing result.
//
// Sources come from research/e98_variant_sources.py. Research-only: nothing
// here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e98_qmv_ab research/e98_qmv_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NARM 3
static const char *kArmName[NARM] = {"a_shipped", "b_indexed", "c_constant"};

// --- bf16, the scored activation and scale type ------------------------------

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);  // round-to-nearest-even
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

static inline float unit(uint32_t *s) {
  return (float)(xorshift32(s) >> 8) / (float)(1u << 24);
}

typedef struct {
  int n;
  int k;
  const char *name;
} Shape;

typedef struct {
  id<MTLBuffer> w;        // shared across arms
  id<MTLBuffer> x;        // shared across arms
  id<MTLBuffer> in_vec;
  id<MTLBuffer> out_vec;
  id<MTLBuffer> scales[NARM];
  id<MTLBuffer> biases[NARM];
  id<MTLBuffer> y[NARM];
  int n;
  int k;
  int lut_entries;
} Operands;

static double g_ns_per_tick = 0.0;
static uint64_t g_session_start = 0;

static double seconds_since(uint64_t start) {
  return (double)(mach_absolute_time() - start) * g_ns_per_tick * 1e-9;
}

// quantized.h:1929-1974. Weight streams per dispatch: qmv_fast_impl indexes the
// input row by tid.x so every one of the M grid columns streams the matrix,
// while the cross-row kernel returns early unless tid.x * IPG < M.
static int weightStreams(int m) {
  static const int ipg[10] = {0, 0, 0, 3, 4, 3, 3, 4, 4, 3};
  if (m == 1) return 1;
  if (m == 2) return 1;
  if (m >= 3 && m <= 9) return (m + ipg[m] - 1) / ipg[m];
  return m;
}

static double armReadBytes(const Operands *o, int arm, int m) {
  const double groups = (double)o->n * (double)o->k / 64.0;
  double nibbles = (double)o->n * (double)o->k / 2.0;
  double meta = arm == 0 ? 4.0 * groups : (arm == 1 ? 2.0 * groups : 0.0);
  double activations = (double)m * o->k * 2.0;
  return weightStreams(m) * (nibbles + meta) + activations;
}

static id<MTLComputePipelineState> buildArm(id<MTLDevice> device,
                                            const char *path, NSString *fn,
                                            const char *label,
                                            NSUInteger *out_regs) {
  NSString *src = [NSString stringWithContentsOfFile:@(path)
                                            encoding:NSUTF8StringEncoding
                                               error:nil];
  if (!src) {
    fprintf(stderr, "e98_qmv_ab: cannot read %s arm source %s\n", label, path);
    exit(1);
  }
  MTLCompileOptions *opts = [MTLCompileOptions new];
  if (@available(macOS 26.0, *)) {
    opts.languageVersion = MTLLanguageVersion4_0;
  } else {
    opts.languageVersion = MTLLanguageVersion3_1;
  }
  [opts setFastMathEnabled:NO];

  NSError *err = nil;
  uint64_t t0 = mach_absolute_time();
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
  double compile_s = seconds_since(t0);
  if (!lib) {
    fprintf(stderr, "e98_qmv_ab: %s arm failed to compile: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  id<MTLFunction> f = [lib newFunctionWithName:fn];
  if (!f) {
    fprintf(stderr, "e98_qmv_ab: %s arm has no function %s\n", label,
            [fn UTF8String]);
    exit(1);
  }
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:f error:&err];
  if (!pso) {
    fprintf(stderr, "e98_qmv_ab: %s arm pipeline failed: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  *out_regs = pso.maxTotalThreadsPerThreadgroup;
  fprintf(stderr,
          "e98_qmv_ab: %s arm compiled in %.2fs  src=%zu bytes  "
          "max_threads=%lu  tg_mem=%lu\n",
          label, compile_s, (size_t)[src length],
          (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.staticThreadgroupMemoryLength);
  return pso;
}

// Build one operand set per arm from a shared pool of `lut_entries` distinct
// (scale, bias) pairs. Arm (a) and arm (c) hold the pair values directly; arm
// (b) holds the uint16 index in `scales` and the table in the front of
// `biases`. Arms (a) and (b) therefore describe the identical dequantisation.
static Operands makeOperands(id<MTLDevice> device, Shape shape, int max_m,
                             int lut_entries) {
  Operands o = (Operands){};
  o.n = shape.n;
  o.k = shape.k;
  o.lut_entries = lut_entries;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;
  if ((size_t)(2 * lut_entries) > groups) {
    fprintf(stderr, "e98_qmv_ab: LUT of %d entries does not fit in %zu groups\n",
            lut_entries, groups);
    exit(1);
  }

  o.w = [device newBufferWithLength:words * 4
                            options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)max_m * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.in_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.out_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  for (int a = 0; a < NARM; a++) {
    o.scales[a] = [device newBufferWithLength:groups * 2
                                      options:MTLResourceStorageModeShared];
    o.biases[a] = [device newBufferWithLength:groups * 2
                                      options:MTLResourceStorageModeShared];
    o.y[a] = [device newBufferWithLength:(size_t)max_m * shape.n * 2
                                 options:MTLResourceStorageModeShared];
    memset(o.y[a].contents, 0, o.y[a].length);
  }

  uint32_t seed = 0x1234567u;
  uint32_t *wp = (uint32_t *)o.w.contents;
  for (size_t i = 0; i < words; i++) {
    wp[i] = xorshift32(&seed);
  }

  uint16_t *pool_s = malloc((size_t)lut_entries * 2);
  uint16_t *pool_b = malloc((size_t)lut_entries * 2);
  for (int i = 0; i < lut_entries; i++) {
    float s = 0.004f + 0.004f * unit(&seed);
    pool_s[i] = f32_to_bf16(s);
    pool_b[i] = f32_to_bf16(-7.5f * s);
  }

  uint16_t *sa = (uint16_t *)o.scales[0].contents;
  uint16_t *ba = (uint16_t *)o.biases[0].contents;
  uint16_t *sb = (uint16_t *)o.scales[1].contents;
  uint16_t *bb = (uint16_t *)o.biases[1].contents;
  uint16_t *sc = (uint16_t *)o.scales[2].contents;
  uint16_t *bc = (uint16_t *)o.biases[2].contents;
  memset(bb, 0, o.biases[1].length);
  for (int i = 0; i < lut_entries; i++) {
    bb[2 * i] = pool_s[i];
    bb[2 * i + 1] = pool_b[i];
  }
  for (size_t g = 0; g < groups; g++) {
    uint16_t idx = (uint16_t)(xorshift32(&seed) % (uint32_t)lut_entries);
    sa[g] = pool_s[idx];
    ba[g] = pool_b[idx];
    sb[g] = idx;  // raw uint16 in the bf16 slot
    sc[g] = pool_s[idx];
    bc[g] = pool_b[idx];
  }
  free(pool_s);
  free(pool_b);

  uint16_t *xp = (uint16_t *)o.x.contents;
  for (size_t i = 0; i < (size_t)max_m * shape.k; i++) {
    xp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  *(int *)o.in_vec.contents = shape.k;
  *(int *)o.out_vec.contents = shape.n;
  return o;
}

static void encodeDispatch(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso, Operands *o,
                           int arm, int m) {
  [enc setComputePipelineState:pso];
  [enc setBuffer:o->w offset:0 atIndex:0];
  [enc setBuffer:o->scales[arm] offset:0 atIndex:1];
  [enc setBuffer:o->biases[arm] offset:0 atIndex:2];
  [enc setBuffer:o->x offset:0 atIndex:3];
  [enc setBuffer:o->y[arm] offset:0 atIndex:4];
  [enc setBuffer:o->in_vec offset:0 atIndex:5];
  [enc setBuffer:o->out_vec offset:0 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m, (NSUInteger)(o->n / 8), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

static double runArm(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                     Operands *o, int arm, int m, int reps, int inner) {
  uint64_t t0 = mach_absolute_time();
  for (int r = 0; r < reps; r++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      encodeDispatch(enc, pso, o, arm, m);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  return seconds_since(t0) / (double)(reps * inner);
}

// Exact affine reference for one output element, in double, read from arm (a)'s
// operands. Proves the harness itself indexes k and the nibbles correctly.
static double referenceElement(Operands *o, int m, int row) {
  const uint32_t *w = (const uint32_t *)o->w.contents;
  const uint16_t *scales = (const uint16_t *)o->scales[0].contents;
  const uint16_t *biases = (const uint16_t *)o->biases[0].contents;
  const uint16_t *x = (const uint16_t *)o->x.contents;
  const int words_per_row = o->k / 8;
  const int groups_per_row = o->k / 64;
  double acc = 0.0;
  for (int g = 0; g < groups_per_row; g++) {
    double s = (double)bf16_to_f32(scales[row * groups_per_row + g]);
    double b = (double)bf16_to_f32(biases[row * groups_per_row + g]);
    double qdot = 0.0, xsum = 0.0;
    for (int word = 0; word < 8; word++) {
      uint32_t packed = w[row * words_per_row + g * 8 + word];
      for (int nib = 0; nib < 8; nib++) {
        int kk = g * 64 + word * 8 + nib;
        double xv = (double)bf16_to_f32(x[m * o->k + kk]);
        qdot += xv * (double)((packed >> (4 * nib)) & 0xf);
        xsum += xv;
      }
    }
    acc += s * qdot + b * xsum;
  }
  return acc;
}

static void dispatchOnce(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pso, Operands *o, int arm,
                         int m) {
  memset(o->y[arm].contents, 0, o->y[arm].length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeDispatch(enc, pso, o, arm, m);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *dir = "/tmp/e98-arms", *out_path = NULL;
    const char *fn_name = "affine_qmv_fast_bfloat16_t_64_4_false";
    const char *widths_arg = "1,5,6,7,8";
    const char *shapes_arg = "0,1,2";
    int pairs = 4, lut = 2658, samples = 64;
    double target_ms = 40.0;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--fn") && i + 1 < argc) fn_name = argv[++i];
      else if (!strcmp(argv[i], "--widths") && i + 1 < argc) widths_arg = argv[++i];
      else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes_arg = argv[++i];
      else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--lut") && i + 1 < argc) lut = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--target-ms") && i + 1 < argc) target_ms = atof(argv[++i]);
      else {
        fprintf(stderr, "e98_qmv_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path) {
      fprintf(stderr, "usage: e98_qmv_ab --out JSON [--dir ARMS] [--fn NAME] "
                      "[--widths L] [--pairs N] [--lut N] [--target-ms MS]\n");
      return 2;
    }

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;
    g_session_start = mach_absolute_time();

    int widths[32], n_widths = 0, max_m = 1;
    for (const char *p = widths_arg; *p && n_widths < 32;) {
      widths[n_widths] = atoi(p);
      if (widths[n_widths] > max_m) max_m = widths[n_widths];
      n_widths++;
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e98_qmv_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *fn = @(fn_name);
    id<MTLComputePipelineState> arm[NARM];
    NSUInteger max_threads[NARM];
    for (int a = 0; a < NARM; a++) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/arm_%c.metal", dir, 'a' + a);
      arm[a] = buildArm(device, path, fn, kArmName[a], &max_threads[a]);
    }

    const Shape all_shapes[] = {
        {34816, 5120, "mlp_gate_up_k5120_n34816"},
        {5120, 17408, "mlp_down_k17408_n5120"},
        {248320, 5120, "lm_head_k5120_n248320"},
    };
    const int n_all = (int)(sizeof(all_shapes) / sizeof(all_shapes[0]));
    Shape shapes[8];
    int n_shapes = 0;
    for (const char *p = shapes_arg; *p && n_shapes < 8;) {
      int idx = atoi(p);
      if (idx < 0 || idx >= n_all) {
        fprintf(stderr, "e98_qmv_ab: shape index %d out of range\n", idx);
        return 2;
      }
      shapes[n_shapes++] = all_shapes[idx];
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e98_qmv_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out, "  \"function\": \"%s\",\n  \"lut_entries\": %d,\n", fn_name, lut);
    fprintf(out, "  \"pairs\": %d,\n  \"order\": \"ABCCBA\",\n", pairs);
    fprintf(out, "  \"arms\": [\"%s\", \"%s\", \"%s\"],\n", kArmName[0],
            kArmName[1], kArmName[2]);
    fprintf(out, "  \"measurements\": [\n");

    int first_row = 1;
    for (int s = 0; s < n_shapes; s++) {
      Operands o = makeOperands(device, shapes[s], max_m, lut);
      fprintf(stderr, "e98_qmv_ab: shape %s  w=%.1fMB  lut=%d entries\n",
              shapes[s].name, (double)o.w.length / 1e6, lut);

      // --- fidelity, before any timing --------------------------------------
      // (a) against an exact double reference, and (b) against (a) BIT for BIT.
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        dispatchOnce(queue, arm[0], &o, 0, m);
        dispatchOnce(queue, arm[1], &o, 1, m);
        const uint16_t *ya = (const uint16_t *)o.y[0].contents;
        const uint16_t *yb = (const uint16_t *)o.y[1].contents;

        size_t differing = 0, total = (size_t)m * o.n;
        int first_bad_m = -1, first_bad_n = -1;
        for (size_t i = 0; i < total; i++) {
          if (ya[i] != yb[i]) {
            if (!differing) {
              first_bad_m = (int)(i / o.n);
              first_bad_n = (int)(i % o.n);
            }
            differing++;
          }
        }

        // A random 4-bit matrix against random signed activations produces
        // outputs that pass through zero, so a per-element relative error is
        // unbounded there. The scale-invariant statistic is the error RMS over
        // the signal RMS; it is what proves the harness indexes k, the nibble
        // order and the group stride correctly.
        double max_rel = 0.0, sq_err = 0.0, sq_want = 0.0;
        int count = 0;
        const int stride = o.n / samples > 0 ? o.n / samples : 1;
        for (int mm = 0; mm < m; mm++) {
          for (int row = 0; row < o.n; row += stride) {
            double want = referenceElement(&o, mm, row);
            double got = (double)bf16_to_f32(ya[mm * o.n + row]);
            double scale = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
            double rel = fabs(got - want) / scale;
            if (rel > max_rel) max_rel = rel;
            sq_err += (got - want) * (got - want);
            sq_want += want * want;
            count++;
          }
        }
        double rms = count && sq_want > 0.0 ? sqrt(sq_err / sq_want) : 0.0;

        fprintf(stderr,
                "e98_qmv_ab:   fidelity M=%d  a_vs_double max_rel=%.3e "
                "rms_over_signal=%.3e   b_vs_a differing=%zu/%zu\n",
                m, max_rel, rms, differing, total);
        if (differing) {
          fprintf(stderr,
                  "e98_qmv_ab:     FIRST MISMATCH m=%d n=%d a=0x%04x b=0x%04x\n",
                  first_bad_m, first_bad_n,
                  ya[(size_t)first_bad_m * o.n + first_bad_n],
                  yb[(size_t)first_bad_m * o.n + first_bad_n]);
        }
        fprintf(out,
                "%s    {\"kind\":\"fidelity\",\"shape\":\"%s\",\"m\":%d,"
                "\"a_vs_double_max_rel\":%.6e,\"a_vs_double_rms_over_signal\":%.6e,"
                "\"b_vs_a_differing\":%zu,\"b_vs_a_total\":%zu,"
                "\"bit_identical\":%s}",
                first_row ? "" : ",\n", shapes[s].name, m, max_rel, rms,
                differing, total, differing ? "false" : "true");
        first_row = 0;
      }

      // --- positive control -------------------------------------------------
      // Perturb one LUT entry so the comparison above is proven able to fail.
      {
        int m = widths[0];
        uint16_t *bb = (uint16_t *)o.biases[1].contents;
        uint16_t saved = bb[0];
        bb[0] = f32_to_bf16(bf16_to_f32(saved) * 1.5f);
        dispatchOnce(queue, arm[1], &o, 1, m);
        const uint16_t *ya = (const uint16_t *)o.y[0].contents;
        const uint16_t *yb = (const uint16_t *)o.y[1].contents;
        size_t differing = 0, total = (size_t)m * o.n;
        for (size_t i = 0; i < total; i++) {
          if (ya[i] != yb[i]) differing++;
        }
        bb[0] = saved;
        fprintf(stderr,
                "e98_qmv_ab:   positive control M=%d  one LUT scale perturbed "
                "-> differing=%zu/%zu\n", m, differing, total);
        fprintf(out,
                ",\n    {\"kind\":\"positive_control\",\"shape\":\"%s\","
                "\"m\":%d,\"differing\":%zu,\"total\":%zu,\"detected\":%s}",
                shapes[s].name, m, differing, total,
                differing ? "true" : "false");
        // Restore arm (a) outputs for the next shape's comparisons.
        dispatchOnce(queue, arm[1], &o, 1, m);
      }

      // --- calibrate, then warm every legal width for every arm -------------
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        // Two probes, second one used: the first dispatch of a cold cell can
        // be several times its steady cost and would starve `inner`.
        runArm(queue, arm[0], &o, 0, m, 1, 1);
        double probe = runArm(queue, arm[0], &o, 0, m, 1, 1);
        int inner = (int)(target_ms * 1e-3 / probe);
        if (inner < 1) inner = 1;
        if (inner > 64) inner = 64;
        const int reps = 3;

        for (int a = 0; a < NARM; a++) {
          runArm(queue, arm[a], &o, a, m, 1, inner);
        }

        for (int p = 0; p < pairs; p++) {
          // ABCCBA: linear drift inside the block cancels for every arm.
          const int order[6] = {0, 1, 2, 2, 1, 0};
          double t[6];
          double at = seconds_since(g_session_start);
          for (int slot = 0; slot < 6; slot++) {
            t[slot] = runArm(queue, arm[order[slot]], &o, order[slot], m, reps,
                             inner);
          }
          double sec[NARM];
          sec[0] = 0.5 * (t[0] + t[5]);
          sec[1] = 0.5 * (t[1] + t[4]);
          sec[2] = 0.5 * (t[2] + t[3]);
          fprintf(stderr,
                  "e98_qmv_ab:   %s M=%d block %d inner=%d  a=%.3fus "
                  "b=%.3fus c=%.3fus  a-b=%+.2f%% a-c=%+.2f%% ratio=%.3f\n",
                  shapes[s].name, m, p, inner, sec[0] * 1e6, sec[1] * 1e6,
                  sec[2] * 1e6, 100.0 * (sec[1] - sec[0]) / sec[0],
                  100.0 * (sec[2] - sec[0]) / sec[0],
                  (sec[0] - sec[2]) != 0.0
                      ? (sec[0] - sec[1]) / (sec[0] - sec[2])
                      : 0.0);
          fprintf(out,
                  ",\n    {\"kind\":\"timing\",\"shape\":\"%s\",\"m\":%d,"
                  "\"block\":%d,\"inner\":%d,\"reps\":%d,"
                  "\"weight_streams\":%d,"
                  "\"a_s\":%.9e,\"b_s\":%.9e,\"c_s\":%.9e,"
                  "\"a_read_bytes\":%.0f,\"b_read_bytes\":%.0f,"
                  "\"c_read_bytes\":%.0f,"
                  "\"session_elapsed_s\":%.3f,"
                  "\"slots\":[%.9e,%.9e,%.9e,%.9e,%.9e,%.9e]}",
                  shapes[s].name, m, p, inner, reps, weightStreams(m), sec[0],
                  sec[1], sec[2], armReadBytes(&o, 0, m), armReadBytes(&o, 1, m),
                  armReadBytes(&o, 2, m), at, t[0], t[1], t[2], t[3], t[4],
                  t[5]);
        }
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e98_qmv_ab: wrote %s\n", out_path);
  }
  return 0;
}
