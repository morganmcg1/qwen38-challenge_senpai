// E104: measure rate(NA) for ONE wide x-group, and find what binds it.
//
// The whole local optimisation problem is `minimise G / rate(NA)`. G is settled
// by E100. This harness measures the second term directly: it forces every
// width M in [2, 6] through a single weight-reading x-group and times four
// arms of the SAME kernel inside one counterbalanced thermal session.
//
//   a_base      the shipped wide kernel.
//   l_loadonly  every device load and bf16 -> f32 conversion of a_base, with
//               the four-nibble FMA block collapsed to one. Timing-only.
//   z_noxload   weights and metadata only; the activation stream is never read.
//               Nothing in it scales with NA except the NA output writes, so it
//               prices the grid's NA - 1 early-return x-groups. Timing-only.
//   xw_widex    a_base with 16-byte activation loads instead of 8-byte ones.
//               Same operations, same operand types, same accumulation order,
//               so it must be BIT-IDENTICAL to a_base. The candidate fix.
//
// `--arms` replaces that default set, so the same harness also runs the rung
// 0.5 partition ladder and the arithmetic arms. A name suffixed `:diag` is a
// timing-only diagnostic and is exempt from the bit-for-bit check; every other
// arm must reproduce arm 0 exactly.
//
// `read_bytes` in the JSON always prices ONE weight stream. An arm whose
// partition reads the weights twice therefore needs its own multiplier, which
// research/e104_analysis.py applies from the arm's partition table.
//
// Sources come from research/e104_variant_sources.py. Research-only: nothing
// here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e104_rate_probe research/e104_rate_probe.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXARM 8
static int g_narm = 4;
static const char *kArmName[MAXARM] = {"a_base", "l_loadonly", "z_noxload",
                                       "xw_widex"};
// An arm named "fm_<base>" reads arm_<base>.metal but compiles it with fast
// math on. That contracts multiply-add into fma, which changes rounding, so
// such an arm is a timing-only diagnostic and can never ship.
#define FM_PREFIX "fm_"
// Diagnostic arms deliberately change the arithmetic, so they are never quoted
// as correctness-bearing. Every other arm must reproduce a_base bit for bit.
static const char *kDiagPrefix[] = {"l_", "z_", FM_PREFIX};
static int kArmExactVsBase[MAXARM] = {1, 0, 0, 1};

static int armIsDiagnostic(const char *name) {
  for (size_t i = 0; i < sizeof(kDiagPrefix) / sizeof(kDiagPrefix[0]); i++) {
    if (!strncmp(name, kDiagPrefix[i], strlen(kDiagPrefix[i]))) return 1;
  }
  return 0;
}

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
  id<MTLBuffer> w;
  id<MTLBuffer> x;
  id<MTLBuffer> in_vec;
  id<MTLBuffer> out_vec;
  id<MTLBuffer> scales;
  id<MTLBuffer> biases;
  id<MTLBuffer> y[MAXARM];
  int n;
  int k;
} Operands;

static double g_ns_per_tick = 0.0;
static uint64_t g_session_start = 0;
static const char *g_macmon = NULL;

static double seconds_since(uint64_t start) {
  return (double)(mach_absolute_time() - start) * g_ns_per_tick * 1e-9;
}

// GPU die temperature, or NaN when no sampler is available. An ungated session
// only supports a relative claim, so every timed cell records its own entry and
// exit temperature rather than one number for the whole run.
static double gpuTempC(void) {
  if (!g_macmon) return NAN;
  char cmd[1024];
  snprintf(cmd, sizeof(cmd),
           "%s pipe -s1 2>/dev/null | /opt/homebrew/bin/jq -r "
           "'.temp.gpu_temp_avg // empty'",
           g_macmon);
  FILE *p = popen(cmd, "r");
  if (!p) return NAN;
  char buf[64] = {0};
  if (!fgets(buf, sizeof(buf), p)) {
    pclose(p);
    return NAN;
  }
  pclose(p);
  return buf[0] ? atof(buf) : NAN;
}

// Every measured width runs as ONE weight-reading x-group: the arm sources
// route M in [2, 6] to IPG == M, so exactly one grid column reads weights and
// the other M - 1 return before their first load.
static double readBytes(const Operands *o, int m) {
  const double groups = (double)o->n * (double)o->k / 64.0;
  const double nibbles = (double)o->n * (double)o->k / 2.0;
  const double metadata = 4.0 * groups;  // bf16 scale + bf16 bias
  const double activations = (double)m * o->k * 2.0;
  return nibbles + metadata + activations;
}

static double flops(const Operands *o, int m) {
  return 2.0 * (double)o->n * (double)o->k * (double)m;
}

static id<MTLLibrary> buildArm(id<MTLDevice> device, const char *path,
                               const char *label, int fast_math) {
  NSString *src = [NSString stringWithContentsOfFile:@(path)
                                            encoding:NSUTF8StringEncoding
                                               error:nil];
  if (!src) {
    fprintf(stderr, "e104_rate_probe: cannot read %s arm source %s\n", label,
            path);
    exit(1);
  }
  MTLCompileOptions *opts = [MTLCompileOptions new];
  if (@available(macOS 26.0, *)) {
    opts.languageVersion = MTLLanguageVersion4_0;
  } else {
    opts.languageVersion = MTLLanguageVersion3_1;
  }
  // MLX builds every JIT library with fast math off (device.cpp
  // Device::build_library_), so the default matches the scored kernel.
  [opts setFastMathEnabled:fast_math ? YES : NO];
  NSError *err = nil;
  uint64_t t0 = mach_absolute_time();
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
  double compile_s = seconds_since(t0);
  if (!lib) {
    fprintf(stderr, "e104_rate_probe: %s arm failed to compile: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  fprintf(stderr, "e104_rate_probe: %s compiled in %.2fs  src=%zu bytes\n",
          label, compile_s, (size_t)[src length]);
  return lib;
}

// The scored dispatcher is one kernel whose register allocation is a max over
// every width branch, so a partition ladder built from it would confound the
// widest branch with the width under test. `--fn` may therefore carry a `%d`,
// which selects one isolated entry point per width and gives each cell its own
// allocation.
static id<MTLComputePipelineState> psoFor(id<MTLDevice> device,
                                          id<MTLLibrary> lib, NSString *fn,
                                          const char *label) {
  id<MTLFunction> f = [lib newFunctionWithName:fn];
  if (!f) {
    fprintf(stderr, "e104_rate_probe: %s arm has no function %s\n", label,
            [fn UTF8String]);
    exit(1);
  }
  NSError *err = nil;
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:f error:&err];
  if (!pso) {
    fprintf(stderr, "e104_rate_probe: %s arm pipeline failed: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  fprintf(stderr,
          "e104_rate_probe:   %s %s  max_threads=%lu  tg_mem=%lu\n", label,
          [fn UTF8String], (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.staticThreadgroupMemoryLength);
  return pso;
}

// One shared weight, metadata and activation buffer for every arm: the arms
// differ only in kernel code, so any operand difference would confound the
// bit-for-bit comparison.
static Operands makeOperands(id<MTLDevice> device, Shape shape, int max_m) {
  Operands o = (Operands){};
  o.n = shape.n;
  o.k = shape.k;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;

  o.w = [device newBufferWithLength:words * 4
                            options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)max_m * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.in_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.out_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.scales = [device newBufferWithLength:groups * 2
                                 options:MTLResourceStorageModeShared];
  o.biases = [device newBufferWithLength:groups * 2
                                 options:MTLResourceStorageModeShared];
  for (int a = 0; a < g_narm; a++) {
    o.y[a] = [device newBufferWithLength:(size_t)max_m * shape.n * 2
                                 options:MTLResourceStorageModeShared];
    memset(o.y[a].contents, 0, o.y[a].length);
  }

  uint32_t seed = 0x1234567u;
  uint32_t *wp = (uint32_t *)o.w.contents;
  for (size_t i = 0; i < words; i++) {
    wp[i] = xorshift32(&seed);
  }
  uint16_t *sp = (uint16_t *)o.scales.contents;
  uint16_t *bp = (uint16_t *)o.biases.contents;
  for (size_t g = 0; g < groups; g++) {
    float s = 0.004f + 0.004f * unit(&seed);
    sp[g] = f32_to_bf16(s);
    bp[g] = f32_to_bf16(-7.5f * s);
  }
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
  [enc setBuffer:o->scales offset:0 atIndex:1];
  [enc setBuffer:o->biases offset:0 atIndex:2];
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

// Exact affine reference for one output element, in double. Proves the harness
// itself indexes k, the nibble order and the group stride correctly.
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

static size_t countDiffering(Operands *o, int arm, int m, int *first_m,
                             int *first_n) {
  const uint16_t *ya = (const uint16_t *)o->y[0].contents;
  const uint16_t *yb = (const uint16_t *)o->y[arm].contents;
  size_t differing = 0, total = (size_t)m * o->n;
  *first_m = -1;
  *first_n = -1;
  for (size_t i = 0; i < total; i++) {
    if (ya[i] != yb[i]) {
      if (!differing) {
        *first_m = (int)(i / o->n);
        *first_n = (int)(i % o->n);
      }
      differing++;
    }
  }
  return differing;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *dir = "/tmp/e104-arms", *out_path = NULL;
    const char *fn_name = "affine_qmv_fast_bfloat16_t_64_4_false";
    const char *widths_arg = "2,3,4,5,6";
    const char *shapes_arg = "0,1,2,3,4";
    const char *arms_arg = NULL;
    int pairs = 4, samples = 48;
    double target_ms = 40.0;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--fn") && i + 1 < argc) fn_name = argv[++i];
      else if (!strcmp(argv[i], "--widths") && i + 1 < argc) widths_arg = argv[++i];
      else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes_arg = argv[++i];
      else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--macmon") && i + 1 < argc) g_macmon = argv[++i];
      else if (!strcmp(argv[i], "--target-ms") && i + 1 < argc) target_ms = atof(argv[++i]);
      else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arms_arg = argv[++i];
      else {
        fprintf(stderr, "e104_rate_probe: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path) {
      fprintf(stderr,
              "usage: e104_rate_probe --out JSON [--dir ARMS] [--fn NAME] "
              "[--widths L] [--shapes L] [--pairs N] [--samples N] "
              "[--macmon PATH] [--target-ms MS]\n");
      return 2;
    }

    if (arms_arg) {
      char buf[512];
      snprintf(buf, sizeof(buf), "%s", arms_arg);
      int n = 0;
      for (char *tok = strtok(buf, ","); tok && n < MAXARM;
           tok = strtok(NULL, ",")) {
        char *mark = strstr(tok, ":diag");
        if (mark) *mark = '\0';
        kArmName[n] = strdup(tok);
        kArmExactVsBase[n] = !mark && !armIsDiagnostic(tok);
        n++;
      }
      if (n < 2) {
        fprintf(stderr, "e104_rate_probe: --arms needs 2 to %d names\n",
                MAXARM);
        return 2;
      }
      g_narm = n;
      if (armIsDiagnostic(kArmName[0])) {
        fprintf(stderr,
                "e104_rate_probe: arm 0 is the exactness reference and must "
                "not be a diagnostic arm\n");
        return 2;
      }
    }

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;
    g_session_start = mach_absolute_time();

    int widths[32], n_widths = 0, max_m = 0;
    for (const char *p = widths_arg; *p && n_widths < 32;) {
      widths[n_widths] = atoi(p);
      if (widths[n_widths] > max_m) max_m = widths[n_widths];
      n_widths++;
      const char *c = strchr(p, ',');
      if (!c) break;
      p = c + 1;
    }

    Shape all_shapes[] = {
        {34816, 5120, "mlp_gate_up_k5120_n34816"},
        {16480, 5120, "gdn_in_proj_k5120_n16480"},
        {14336, 5120, "fa_qkv_k5120_n14336"},
        {5120, 17408, "mlp_down_k17408_n5120"},
        {5120, 6144, "gdn_out_proj_k6144_n5120"},
        // The two remaining scored output widths. The real n = 98336 readout is
        // 2-bit and reaches a different entry point, so this 4-bit point covers
        // the width for an exactness sweep and is not a scored-cell timing.
        {98336, 5120, "draft_readout_width_n98336_k5120_4bit"},
        {248320, 5120, "lm_head_k5120_n248320"},
    };
    Shape shapes[8];
    int n_shapes = 0;
    for (const char *p = shapes_arg; *p && n_shapes < 8;) {
      int idx = atoi(p);
      if (idx >= 0 && idx < (int)(sizeof(all_shapes) / sizeof(all_shapes[0]))) {
        shapes[n_shapes++] = all_shapes[idx];
      }
      const char *c = strchr(p, ',');
      if (!c) break;
      p = c + 1;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSString *arch = [[device architecture] name];

    const int per_width = strstr(fn_name, "%d") != NULL;
    id<MTLComputePipelineState> pso[MAXARM][32];
    for (int a = 0; a < g_narm; a++) {
      char path[1024];
      int fast_math = !strncmp(kArmName[a], FM_PREFIX, strlen(FM_PREFIX));
      const char *srcName =
          fast_math ? kArmName[a] + strlen(FM_PREFIX) : kArmName[a];
      snprintf(path, sizeof(path), "%s/arm_%s.metal", dir, srcName);
      id<MTLLibrary> lib = buildArm(device, path, kArmName[a], fast_math);
      for (int wi = 0; wi < n_widths; wi++) {
        if (!per_width && wi) {
          pso[a][wi] = pso[a][0];
          continue;
        }
        char name[256];
        snprintf(name, sizeof(name), fn_name, widths[wi]);
        pso[a][wi] = psoFor(device, lib, @(name), kArmName[a]);
      }
    }

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e104_rate_probe: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out, "  \"function\": \"%s\",\n  \"pairs\": %d,\n", fn_name, pairs);
    fprintf(out, "  \"order\": \"palindrome\",\n  \"arm_count\": %d,\n", g_narm);
    fprintf(out, "  \"arms\": [");
    for (int a = 0; a < g_narm; a++) {
      fprintf(out, "%s\"%s\"", a ? ", " : "", kArmName[a]);
    }
    fprintf(out, "],\n  \"measurements\": [\n");

    int first_row = 1;
    for (int s = 0; s < n_shapes; s++) {
      Operands o = makeOperands(device, shapes[s], max_m);
      fprintf(stderr, "e104_rate_probe: shape %s  w=%.1fMB\n", shapes[s].name,
              (double)o.w.length / 1e6);

      // --- fidelity, before any timing -----------------------------------
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        for (int a = 0; a < g_narm; a++) {
          dispatchOnce(queue, pso[a][wi], &o, a, m);
        }

        double max_rel = 0.0, sq_err = 0.0, sq_want = 0.0;
        int count = 0;
        const int stride = o.n / samples > 0 ? o.n / samples : 1;
        const uint16_t *ya = (const uint16_t *)o.y[0].contents;
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

        fprintf(out,
                "%s    {\"kind\":\"fidelity\",\"shape\":\"%s\",\"m\":%d,"
                "\"base_vs_double_max_rel\":%.6e,"
                "\"base_vs_double_rms_over_signal\":%.6e,\"arms\":[",
                first_row ? "" : ",\n", shapes[s].name, m, max_rel, rms);
        first_row = 0;
        fprintf(stderr,
                "e104_rate_probe:   fidelity M=%d  base_vs_double max_rel=%.3e "
                "rms=%.3e\n", m, max_rel, rms);
        for (int a = 1; a < g_narm; a++) {
          int fm = -1, fn_ = -1;
          size_t differing = countDiffering(&o, a, m, &fm, &fn_);
          size_t total = (size_t)m * o.n;
          fprintf(out,
                  "%s{\"arm\":\"%s\",\"exact_required\":%s,\"differing\":%zu,"
                  "\"total\":%zu,\"bit_identical\":%s,\"first_bad_m\":%d,"
                  "\"first_bad_n\":%d}",
                  a > 1 ? "," : "", kArmName[a],
                  kArmExactVsBase[a] ? "true" : "false", differing, total,
                  differing ? "false" : "true", fm, fn_);
          fprintf(stderr,
                  "e104_rate_probe:     %-11s vs base differing=%zu/%zu%s\n",
                  kArmName[a], differing, total,
                  kArmExactVsBase[a]
                      ? (differing ? "   *** EXACTNESS FAILURE ***" : "   exact")
                      : "   (diagnostic arm, difference expected)");
          if (kArmExactVsBase[a] && differing) {
            const uint16_t *yb = (const uint16_t *)o.y[a].contents;
            fprintf(stderr,
                    "e104_rate_probe:       FIRST MISMATCH m=%d n=%d "
                    "base=0x%04x arm=0x%04x\n",
                    fm, fn_, ya[(size_t)fm * o.n + fn_],
                    yb[(size_t)fm * o.n + fn_]);
          }
        }
        fprintf(out, "]}");
      }

      // --- positive control ------------------------------------------------
      // Perturb one activation and re-dispatch ONLY the exact arm, so the
      // bit-for-bit comparison above is proven able to fail.
      {
        int m = widths[0];
        uint16_t *xp = (uint16_t *)o.x.contents;
        uint16_t saved = xp[0];
        xp[0] = f32_to_bf16(bf16_to_f32(saved) * 1.5f + 0.25f);
        dispatchOnce(queue, pso[g_narm - 1][0], &o, g_narm - 1, m);
        int fm = -1, fn_ = -1;
        size_t differing = countDiffering(&o, g_narm - 1, m, &fm, &fn_);
        size_t total = (size_t)m * o.n;
        xp[0] = saved;
        fprintf(stderr,
                "e104_rate_probe:   positive control M=%d  one activation "
                "perturbed -> differing=%zu/%zu\n", m, differing, total);
        fprintf(out,
                ",\n    {\"kind\":\"positive_control\",\"shape\":\"%s\","
                "\"m\":%d,\"arm\":\"%s\",\"differing\":%zu,\"total\":%zu,"
                "\"detected\":%s}",
                shapes[s].name, m, kArmName[g_narm - 1], differing, total,
                differing ? "true" : "false");
        dispatchOnce(queue, pso[g_narm - 1][0], &o, g_narm - 1, m);
      }

      // --- calibrate, warm, then time ---------------------------------------
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        runArm(queue, pso[0][wi], &o, 0, m, 1, 1);
        double probe = runArm(queue, pso[0][wi], &o, 0, m, 1, 1);
        int inner = (int)(target_ms * 1e-3 / probe);
        if (inner < 1) inner = 1;
        if (inner > 64) inner = 64;
        const int reps = 3;

        for (int a = 0; a < g_narm; a++) {
          runArm(queue, pso[a][wi], &o, a, m, 1, inner);
        }

        double entry_c = gpuTempC();
        const int slots = 2 * g_narm;
        for (int p = 0; p < pairs; p++) {
          // AB..BA palindrome: linear drift inside the block cancels for every
          // arm at any arm count.
          double t[2 * MAXARM];
          double at = seconds_since(g_session_start);
          for (int slot = 0; slot < slots; slot++) {
            const int a = slot < g_narm ? slot : slots - 1 - slot;
            t[slot] = runArm(queue, pso[a][wi], &o, a, m, reps, inner);
          }
          double sec[MAXARM];
          for (int a = 0; a < g_narm; a++) {
            sec[a] = 0.5 * (t[a] + t[slots - 1 - a]);
          }
          const double bytes = readBytes(&o, m);
          const double fl = flops(&o, m);

          fprintf(stderr, "e104_rate_probe:   %s M=%d block %d inner=%d",
                  shapes[s].name, m, p, inner);
          for (int a = 0; a < g_narm; a++) {
            fprintf(stderr, "  %s=%.1fus/%.1fGBs", kArmName[a], sec[a] * 1e6,
                    bytes / sec[a] / 1e9);
          }
          fprintf(stderr, "  last_vs_base=%+.2f%%\n",
                  100.0 * (sec[g_narm - 1] - sec[0]) / sec[0]);

          fprintf(out,
                  ",\n    {\"kind\":\"timing\",\"shape\":\"%s\",\"m\":%d,"
                  "\"block\":%d,\"inner\":%d,\"reps\":%d,"
                  "\"read_bytes\":%.0f,\"flops\":%.0f,"
                  "\"session_elapsed_s\":%.3f,\"gpu_temp_entry_c\":%.2f,"
                  "\"seconds\":{",
                  shapes[s].name, m, p, inner, reps, bytes, fl, at, entry_c);
          for (int a = 0; a < g_narm; a++) {
            fprintf(out, "%s\"%s\":%.9e", a ? "," : "", kArmName[a], sec[a]);
          }
          fprintf(out, "},\"slots\":[");
          for (int slot = 0; slot < slots; slot++) {
            fprintf(out, "%s%.9e", slot ? "," : "", t[slot]);
          }
          fprintf(out, "]}");
        }
        double exit_c = gpuTempC();
        fprintf(out,
                ",\n    {\"kind\":\"thermal\",\"shape\":\"%s\",\"m\":%d,"
                "\"gpu_temp_entry_c\":%.2f,\"gpu_temp_exit_c\":%.2f,"
                "\"cool_gate_passed_real_gate\":false,"
                "\"gate_qualified_for_timing\":false}",
                shapes[s].name, m, entry_c, exit_c);
        fprintf(stderr,
                "e104_rate_probe:   %s M=%d thermal entry=%.1fC exit=%.1fC\n",
                shapes[s].name, m, entry_c, exit_c);
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e104_rate_probe: wrote %s\n", out_path);
  }
  return 0;
}
