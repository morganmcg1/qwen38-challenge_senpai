// E44 section 7.3: paired base-vs-candidate microbenchmark of the scored
// affine-4/g64 `qmv_fast` verify-width tree.
//
// The existing Tests/MLXFastTests/QwenQMVCostCurveTests.swift harness measures a
// WITHIN-BUILD C(M) curve. It cannot answer this question, because the two arms
// here are two different compiled kernel sources, and in this checkout that
// source is the JIT string inside mlx-generated/quantized.cpp (a C++ recompile),
// not the metallib. Swapping arms would mean swapping builds, so the two arms
// could never be interleaved inside one thermal session.
//
// This harness instead compiles BOTH runtime-effective JIT strings with
// newLibraryWithSource: in ONE process, using the same MTLCompileOptions the
// scored worker uses (device.cpp: LanguageVersion4_0, fastMath disabled), and
// alternates them ABBA inside a single session so monotone thermal drift
// cancels to first order. Sources come from research/jit_string_compile.py
// --emit (base arm from a git revision, candidate arm from the worktree).
//
// Research-only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e44_qmv_ab research/e44_qmv_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// --- deterministic inputs ----------------------------------------------------

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
  id<MTLBuffer> w;
  id<MTLBuffer> scales;
  id<MTLBuffer> biases;
  id<MTLBuffer> x;
  id<MTLBuffer> y;
  id<MTLBuffer> in_vec;
  id<MTLBuffer> out_vec;
  int n;
  int k;
} Operands;

static double g_ns_per_tick = 0.0;
static uint64_t g_session_start = 0;

static double seconds_since(uint64_t start) {
  return (double)(mach_absolute_time() - start) * g_ns_per_tick * 1e-9;
}

static id<MTLComputePipelineState> buildArm(id<MTLDevice> device,
                                           const char *path,
                                           NSString *fn,
                                           const char *label) {
  NSString *src = [NSString stringWithContentsOfFile:@(path)
                                            encoding:NSUTF8StringEncoding
                                               error:nil];
  if (!src) {
    fprintf(stderr, "e44_qmv_ab: cannot read %s arm source %s\n", label, path);
    exit(1);
  }
  MTLCompileOptions *opts = [MTLCompileOptions new];
  // device.cpp build_library_from_source: newest language version the OS
  // offers, fast math OFF. No include path: the string must be self-contained.
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
    fprintf(stderr, "e44_qmv_ab: %s arm failed to compile: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  id<MTLFunction> f = [lib newFunctionWithName:fn];
  if (!f) {
    fprintf(stderr, "e44_qmv_ab: %s arm has no function %s\n", label,
            [fn UTF8String]);
    exit(1);
  }
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:f
                                                                         error:&err];
  if (!pso) {
    fprintf(stderr, "e44_qmv_ab: %s arm pipeline failed: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  fprintf(stderr,
          "e44_qmv_ab: %s arm compiled in %.2fs  src=%zu bytes  "
          "max_threads=%lu  tg_mem=%lu\n",
          label, compile_s, (size_t)[src length],
          (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.staticThreadgroupMemoryLength);
  return pso;
}

static Operands makeOperands(id<MTLDevice> device, Shape shape, int max_m) {
  Operands o;
  o.n = shape.n;
  o.k = shape.k;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;   // 4-bit
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;  // group_size 64

  o.w = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
  o.scales = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.biases = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)max_m * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.y = [device newBufferWithLength:(size_t)max_m * shape.n * 2
                            options:MTLResourceStorageModeShared];
  o.in_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.out_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];

  uint32_t seed = 0x1234567u;
  uint32_t *wp = (uint32_t *)o.w.contents;
  for (size_t i = 0; i < words; i++) {
    wp[i] = xorshift32(&seed);  // eight independent 4-bit nibbles
  }
  uint16_t *sp = (uint16_t *)o.scales.contents;
  uint16_t *bp = (uint16_t *)o.biases.contents;
  for (size_t i = 0; i < groups; i++) {
    float s = 0.004f + 0.004f * unit(&seed);
    sp[i] = f32_to_bf16(s);
    bp[i] = f32_to_bf16(-7.5f * s);  // affine zero-point near the nibble centre
  }
  uint16_t *xp = (uint16_t *)o.x.contents;
  for (size_t i = 0; i < (size_t)max_m * shape.k; i++) {
    xp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  memset(o.y.contents, 0, o.y.length);
  *(int *)o.in_vec.contents = shape.k;
  *(int *)o.out_vec.contents = shape.n;
  return o;
}

static void encodeDispatch(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           Operands *o, int m) {
  [enc setComputePipelineState:pso];
  // backend/metal/quantized.cpp qmv_fast: for batch <= 1 only buffers 0..6 are
  // bound, because add_strides_and_shapes returns before the batch operands.
  [enc setBuffer:o->w offset:0 atIndex:0];
  [enc setBuffer:o->scales offset:0 atIndex:1];
  [enc setBuffer:o->biases offset:0 atIndex:2];
  [enc setBuffer:o->x offset:0 atIndex:3];
  [enc setBuffer:o->y offset:0 atIndex:4];
  [enc setBuffer:o->in_vec offset:0 atIndex:5];
  [enc setBuffer:o->out_vec offset:0 atIndex:6];
  // group_dims(32, 2, 1), grid_dims(M, ceil(N / bn), B) with bn = 8.
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m, (NSUInteger)(o->n / 8), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

static double runArm(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pso,
                     Operands *o, int m, int reps, int inner) {
  uint64_t t0 = mach_absolute_time();
  for (int r = 0; r < reps; r++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      encodeDispatch(enc, pso, o, m);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  return seconds_since(t0) / (double)(reps * inner);
}

// Exact affine reference for one output element, in double.
static double referenceElement(Operands *o, int m, int row) {
  const uint32_t *w = (const uint32_t *)o->w.contents;
  const uint16_t *scales = (const uint16_t *)o->scales.contents;
  const uint16_t *biases = (const uint16_t *)o->biases.contents;
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

typedef struct {
  double max_rel;   // vs the double reference
  double rms_rel;
  double max_rel_vs_base;
  // The element that produced max_rel, so a large relative number can be read
  // against the magnitude it was divided by instead of being taken at face
  // value: a near-cancelling output turns rounding-scale absolute error into a
  // large ratio.
  int worst_m;
  int worst_row;
  double worst_want;
  double worst_got;
} Fidelity;

static Fidelity checkArm(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pso,
                         Operands *o, int m, int samples,
                         const float *base_y) {
  memset(o->y.contents, 0, o->y.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeDispatch(enc, pso, o, m);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  const uint16_t *y = (const uint16_t *)o->y.contents;
  Fidelity f = {0};
  double sq = 0.0;
  int count = 0;
  const int stride = o->n / samples > 0 ? o->n / samples : 1;
  for (int mm = 0; mm < m; mm++) {
    for (int row = 0; row < o->n; row += stride) {
      double want = referenceElement(o, mm, row);
      double got = (double)bf16_to_f32(y[mm * o->n + row]);
      double scale = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
      double rel = fabs(got - want) / scale;
      if (rel > f.max_rel) {
        f.max_rel = rel;
        f.worst_m = mm;
        f.worst_row = row;
        f.worst_want = want;
        f.worst_got = got;
      }
      sq += rel * rel;
      if (base_y) {
        double bv = (double)base_y[mm * o->n + row / stride];
        double bs = fabs(bv) > 1e-6 ? fabs(bv) : 1e-6;
        double rb = fabs(got - bv) / bs;
        if (rb > f.max_rel_vs_base) f.max_rel_vs_base = rb;
      }
      count++;
    }
  }
  f.rms_rel = count ? sqrt(sq / (double)count) : 0.0;
  return f;
}

// One-hot readout of the packed-weight mapping.
//
// Setting x[0][k0] = 1 and everything else 0 makes the exact answer
// y[0][n] = scale[n][k0 / 64] * q[n][k0] + bias[n][k0 / 64], with no summation
// and therefore no cancellation. So this reads what k each arm actually paired
// with each nibble, against the weights the harness itself wrote. If an arm's
// outputs are a permutation of the expected list, this names the permutation;
// no reasoning about the packing convention is involved.
static void probeMapping(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> base,
                         id<MTLComputePipelineState> cand,
                         Operands *o, int m, int nprobe) {
  const int row = 0;  // read output row 0
  uint16_t *xp = (uint16_t *)o->x.contents;
  const uint32_t *w = (const uint32_t *)o->w.contents;
  const uint16_t *scales = (const uint16_t *)o->scales.contents;
  const uint16_t *biases = (const uint16_t *)o->biases.contents;
  const int groups_per_row = o->k / 64;
  const int words_per_row = o->k / 8;

  double *expect = calloc(nprobe, sizeof(double));
  for (int k0 = 0; k0 < nprobe; k0++) {
    double s = bf16_to_f32(scales[row * groups_per_row + k0 / 64]);
    double b = bf16_to_f32(biases[row * groups_per_row + k0 / 64]);
    uint32_t packed = w[row * words_per_row + k0 / 8];
    expect[k0] = s * (double)((packed >> (4 * (k0 % 8))) & 0xf) + b;
  }

  memset(o->x.contents, 0, o->x.length);
  fprintf(stderr,
          "e44_qmv_ab:   one-hot probe M=%d row=%d  "
          "(expect[k] = scale*nibble(k%%8 of word k/8) + bias)\n", m, row);
  fprintf(stderr, "e44_qmv_ab:     k0   expected        base      -> k?"
                  "        cand      -> k?\n");
  for (int k0 = 0; k0 < nprobe; k0++) {
    xp[k0] = f32_to_bf16(1.0f);
    double got[2];
    id<MTLComputePipelineState> psos[2] = {base, cand};
    for (int a = 0; a < 2; a++) {
      memset(o->y.contents, 0, o->y.length);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      encodeDispatch(enc, psos[a], o, m);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      got[a] = bf16_to_f32(((const uint16_t *)o->y.contents)[row]);
    }
    // Which k does each arm's answer correspond to, if any?
    int which[2] = {-1, -1};
    for (int a = 0; a < 2; a++) {
      double best = 1e30;
      for (int kk = 0; kk < nprobe; kk++) {
        double d = fabs(got[a] - expect[kk]);
        if (d < best) { best = d; which[a] = kk; }
      }
      // Only claim a match if it is far closer than bf16 store noise allows.
      if (best > 1e-3 * fabs(expect[which[a]]) + 1e-9) which[a] = -1;
    }
    fprintf(stderr,
            "e44_qmv_ab:    %3d  %+.6f  %+.6f  %4d  %+.6f  %4d\n",
            k0, expect[k0], got[0], which[0], got[1], which[1]);
    xp[k0] = 0;
  }
  free(expect);
}

// Exact coverage proof over every (m, n, k).
//
// The one-hot probe reads the k mapping but only exercises one term of one
// group, so it cannot see a group that is summed twice or skipped. This probe
// closes that gap without giving up exactness. Every scale is forced to 1 and
// every bias to 0, and x is set to 1 on exactly the eight k values packed in one
// weight word, zero everywhere else. The exact answer is then
//
//   y[m][n] = sum of the eight nibbles of w[n][word]
//
// an integer in [0, 120], which bfloat16 represents without loss, so the
// comparison can demand bit equality of the stored result rather than a
// tolerance. One dropped, duplicated or misplaced nibble moves the sum by at
// least 1 and is therefore visible. Sweeping the word index covers every k with
// per-nibble resolution, and a single dispatch yields every (m, n) at once, so
// the same sweep also proves the m and n tiling. It says nothing about the
// affine arithmetic -- that is what the random-x fidelity check is for.
static void probeCoverage(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> base,
                          id<MTLComputePipelineState> cand,
                          Operands *o, int m, int word_stride) {
  const int words_per_row = o->k / 8;
  const size_t n_groups = (size_t)o->n * o->k / 64;
  uint16_t *sp = (uint16_t *)o->scales.contents;
  uint16_t *bp = (uint16_t *)o->biases.contents;
  const uint16_t one = f32_to_bf16(1.0f);
  for (size_t i = 0; i < n_groups; i++) { sp[i] = one; bp[i] = 0; }
  memset(o->x.contents, 0, o->x.length);
  uint16_t *xp = (uint16_t *)o->x.contents;
  const uint32_t *w = (const uint32_t *)o->w.contents;
  const uint16_t *y = (const uint16_t *)o->y.contents;

  long checked = 0, bad[2] = {0, 0}, shown[2] = {0, 0};
  double worst_abs[2] = {0.0, 0.0};
  id<MTLComputePipelineState> psos[2] = {base, cand};
  const char *label[2] = {"base", "cand"};

  for (int word = 0; word < words_per_row; word += word_stride) {
    for (int mm = 0; mm < m; mm++) {
      for (int j = 0; j < 8; j++) xp[mm * o->k + word * 8 + j] = one;
    }
    for (int a = 0; a < 2; a++) {
      memset(o->y.contents, 0, o->y.length);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      encodeDispatch(enc, psos[a], o, m);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      for (int row = 0; row < o->n; row++) {
        uint32_t packed = w[row * words_per_row + word];
        int want = 0;
        for (int nib = 0; nib < 8; nib++) want += (packed >> (4 * nib)) & 0xf;
        for (int mm = 0; mm < m; mm++) {
          float got = bf16_to_f32(y[mm * o->n + row]);
          double d = fabs((double)got - (double)want);
          if (d > worst_abs[a]) worst_abs[a] = d;
          if (got != (float)want) {
            bad[a]++;
            if (shown[a] < 8) {
              fprintf(stderr,
                      "e44_qmv_ab:     %s MISMATCH word=%d k=[%d,%d) m=%d n=%d "
                      "want=%d got=%.4f\n",
                      label[a], word, word * 8, word * 8 + 8, mm, row, want,
                      got);
              shown[a]++;
            }
          }
          if (a == 0) checked++;
        }
      }
    }
    for (int mm = 0; mm < m; mm++) {
      for (int j = 0; j < 8; j++) xp[mm * o->k + word * 8 + j] = 0;
    }
  }
  fprintf(stderr,
          "e44_qmv_ab:   coverage M=%d words=%d(stride %d) elements=%ld  "
          "base bad=%ld worst_abs=%.1f  cand bad=%ld worst_abs=%.1f\n",
          m, (words_per_row + word_stride - 1) / word_stride, word_stride,
          checked, bad[0], worst_abs[0], bad[1], worst_abs[1]);
}

static void captureSampled(Operands *o, int m, int samples, float *out) {
  const uint16_t *y = (const uint16_t *)o->y.contents;
  const int stride = o->n / samples > 0 ? o->n / samples : 1;
  for (int mm = 0; mm < m; mm++) {
    int j = 0;
    for (int row = 0; row < o->n; row += stride) {
      out[mm * o->n + j] = bf16_to_f32(y[mm * o->n + row]);
      j++;
    }
  }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *base_path = NULL, *cand_path = NULL, *out_path = NULL;
    const char *fn_name = "affine_qmv_fast_bfloat16_t_64_4_false";
    const char *widths_arg = "1,2,3,4,5,6,7,8,9";
    int pairs = 5, reps = 25, inner = 20, samples = 32, probe = 0, coverage = 0;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--base") && i + 1 < argc) base_path = argv[++i];
      else if (!strcmp(argv[i], "--cand") && i + 1 < argc) cand_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--fn") && i + 1 < argc) fn_name = argv[++i];
      else if (!strcmp(argv[i], "--widths") && i + 1 < argc) widths_arg = argv[++i];
      else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--probe") && i + 1 < argc) probe = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--coverage") && i + 1 < argc) coverage = atoi(argv[++i]);
      else {
        fprintf(stderr, "e44_qmv_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!base_path || !cand_path || !out_path) {
      fprintf(stderr, "usage: e44_qmv_ab --base SRC --cand SRC --out JSON "
                      "[--fn NAME] [--widths L] [--pairs N] [--reps N] "
                      "[--inner N]\n");
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
      fprintf(stderr, "e44_qmv_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *fn = @(fn_name);
    id<MTLComputePipelineState> arm[2];
    arm[0] = buildArm(device, base_path, fn, "base");
    arm[1] = buildArm(device, cand_path, fn, "cand");

    const Shape shapes[] = {
        {5120, 5120, "attn_out_n5120_k5120"},
        {5120, 17408, "mlp_down_n5120_k17408"},
    };
    const int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e44_qmv_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out, "  \"function\": \"%s\",\n", fn_name);
    fprintf(out, "  \"pairs\": %d,\n  \"reps\": %d,\n  \"inner\": %d,\n",
            pairs, reps, inner);
    fprintf(out, "  \"order\": \"ABBA\",\n  \"measurements\": [\n");

    int first_row = 1;
    for (int s = 0; s < n_shapes; s++) {
      Operands o = makeOperands(device, shapes[s], max_m);
      fprintf(stderr, "e44_qmv_ab: shape %s  w=%.1fMB\n", shapes[s].name,
              (double)o.w.length / 1e6);

      // Diagnostics run instead of timing, never alongside it: both of them
      // overwrite the operands they read.
      if (probe > 0) {
        for (int wi = 0; wi < n_widths; wi++) {
          probeMapping(queue, arm[0], arm[1], &o, widths[wi], probe);
        }
        continue;
      }
      if (coverage > 0) {
        for (int wi = 0; wi < n_widths; wi++) {
          probeCoverage(queue, arm[0], arm[1], &o, widths[wi], coverage);
        }
        continue;
      }

      // --- fidelity, before any timing ---------------------------------------
      // The candidate is not expected to be bit-equal (the matrix unit fixes its
      // own 8-wide summation order), so the decisive cheap check is agreement
      // with an exact double reference: that is what proves the fragment map and
      // the nibble/k indexing are right. A wrong index shows up as a huge
      // relative error, not a rounding-scale one.
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        float *base_y = calloc((size_t)m * shapes[s].n, sizeof(float));
        Fidelity fb = checkArm(queue, arm[0], &o, m, samples, NULL);
        captureSampled(&o, m, samples, base_y);
        Fidelity fc = checkArm(queue, arm[1], &o, m, samples, base_y);
        fprintf(stderr,
                "e44_qmv_ab:   fidelity M=%d  base max_rel=%.3e rms=%.3e  "
                "cand max_rel=%.3e rms=%.3e  cand_vs_base=%.3e\n",
                m, fb.max_rel, fb.rms_rel, fc.max_rel, fc.rms_rel,
                fc.max_rel_vs_base);
        fprintf(stderr,
                "e44_qmv_ab:     worst element  base m=%d n=%d want=%+.6f "
                "got=%+.6f abs=%.2e   cand m=%d n=%d want=%+.6f got=%+.6f "
                "abs=%.2e\n",
                fb.worst_m, fb.worst_row, fb.worst_want, fb.worst_got,
                fabs(fb.worst_got - fb.worst_want), fc.worst_m, fc.worst_row,
                fc.worst_want, fc.worst_got,
                fabs(fc.worst_got - fc.worst_want));
        fprintf(out,
                "%s    {\"kind\":\"fidelity\",\"shape\":\"%s\",\"m\":%d,"
                "\"base_max_rel\":%.6e,\"base_rms_rel\":%.6e,"
                "\"cand_max_rel\":%.6e,\"cand_rms_rel\":%.6e,"
                "\"cand_vs_base_max_rel\":%.6e,"
                "\"base_worst\":{\"m\":%d,\"n\":%d,\"want\":%.6e,\"got\":%.6e},"
                "\"cand_worst\":{\"m\":%d,\"n\":%d,\"want\":%.6e,\"got\":%.6e}}",
                first_row ? "" : ",\n", shapes[s].name, m, fb.max_rel,
                fb.rms_rel, fc.max_rel, fc.rms_rel, fc.max_rel_vs_base,
                fb.worst_m, fb.worst_row, fb.worst_want, fb.worst_got,
                fc.worst_m, fc.worst_row, fc.worst_want, fc.worst_got);
        first_row = 0;
        free(base_y);
      }

      // --- warm every legal width for both arms ------------------------------
      for (int wi = 0; wi < n_widths; wi++) {
        for (int a = 0; a < 2; a++) {
          runArm(queue, arm[a], &o, widths[wi], 3, inner);
        }
      }

      // --- paired timing ----------------------------------------------------
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        for (int p = 0; p < pairs; p++) {
          // One pair is ABBA, so drift inside the pair cancels to first order
          // and the arm order is balanced without a second session.
          const int order[4] = {0, 1, 1, 0};
          double t[4];
          double at = seconds_since(g_session_start);
          for (int slot = 0; slot < 4; slot++) {
            t[slot] = runArm(queue, arm[order[slot]], &o, m, reps, inner);
          }
          double base_s = 0.5 * (t[0] + t[3]);
          double cand_s = 0.5 * (t[1] + t[2]);
          fprintf(stderr,
                  "e44_qmv_ab:   %s M=%d pair %d  base=%.3fus cand=%.3fus "
                  "delta=%+.2f%%\n",
                  shapes[s].name, m, p, base_s * 1e6, cand_s * 1e6,
                  100.0 * (cand_s - base_s) / base_s);
          fprintf(out,
                  ",\n    {\"kind\":\"timing\",\"shape\":\"%s\",\"m\":%d,"
                  "\"pair\":%d,\"base_s\":%.9e,\"cand_s\":%.9e,"
                  "\"session_elapsed_s\":%.3f,"
                  "\"slots\":[%.9e,%.9e,%.9e,%.9e]}",
                  shapes[s].name, m, p, base_s, cand_s, at,
                  t[0], t[1], t[2], t[3]);
        }
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e44_qmv_ab: wrote %s\n", out_path);
  }
  return 0;
}
