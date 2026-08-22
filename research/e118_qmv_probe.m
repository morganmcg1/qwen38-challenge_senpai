// E118: time the metadata-load arms of the wide affine-4 QMV.
//
// This is E104's rate probe, extended in three ways that this experiment needs.
//
// 1. HARNESS DEFECT 16. `macmon pipe -s1` idles the GPU for about one second,
//    and the DVFS ramp back up costs a fixed 30 to 80 ms that is paid entirely
//    by the first arm timed after the sample. A fixed cost is not monotone
//    drift, so the ABBA palindrome mean does NOT cancel it, and slot 0 is arm 0,
//    the baseline. E115 measured a false +5.95 % winner from exactly this.
//    Here every temperature sample is followed by a discarded fixed-duration
//    ramp burst before any timed work, the sample itself is moved before the
//    per-arm warm-up rather than after it, and block 0 is still discarded by
//    research/e118_analysis.py. The per-arm forward-minus-reverse slot gap is
//    written to the JSON so the fix can be checked instead of assumed.
//
// 2. A ninth buffer, `packed_sb`, holding the same BF16 scale and bias values
//    interleaved into one uint32 per group. The `g_pack32` family reads it.
//    Every arm declares it, so no arm differs from another in its argument
//    table.
//
// 3. A positive control per exact arm, on BOTH operand paths that matter here.
//    One activation is perturbed and only the candidate arm is re-dispatched;
//    then one group's scale, bias and interleaved record are perturbed together
//    and only the candidate arm is re-dispatched. A bit-exactness claim is only
//    evidence if the comparison is proven able to fail on the field the arm
//    actually changed.
//
// Sources come from research/e118_arms.py. Research-only: nothing here is on
// the scored path, and this probe compiles its own copy of the kernel, so its
// numbers are NOT end-to-end and say nothing about the shipped worker until an
// arm is landed in the shipped file by whoever owns it.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e118_qmv_probe research/e118_qmv_probe.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXARM 16
enum { kMetaProbeRows = 8 };
static int g_narm = 0;
static const char *kArmName[MAXARM];
static int kArmExactVsBase[MAXARM];

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

// E111's Bias6 packer, the CPU half of `e118_bias_from_code`. Every operation
// after the single FP32 multiply is integer, so the two halves agree bit for
// bit and `e_bias6` can be required to match `a_base` exactly.
static inline uint16_t bias_bf16_from_code(float scale, uint32_t code) {
  float prod = -(float)(code & 0xFu) * scale;
  uint32_t u;
  memcpy(&u, &prod, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  u += ((code & 0x30u) << 12) - 0x10000u;
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
  id<MTLBuffer> packed_sb;
  id<MTLBuffer> bias_codes;
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
// exit temperature rather than one number for the whole run. The call idles the
// GPU: every caller must follow it with rampBurst before timing anything.
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

// `read_bytes` always prices ONE weight stream, as in E104 and E110.
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
                               const char *label) {
  NSString *src = [NSString stringWithContentsOfFile:@(path)
                                            encoding:NSUTF8StringEncoding
                                               error:nil];
  if (!src) {
    fprintf(stderr, "e118_qmv_probe: cannot read %s arm source %s\n", label,
            path);
    exit(1);
  }
  MTLCompileOptions *opts = [MTLCompileOptions new];
  if (@available(macOS 26.0, *)) {
    opts.languageVersion = MTLLanguageVersion4_0;
  } else {
    opts.languageVersion = MTLLanguageVersion3_1;
  }
  // MLX builds every JIT library with fast math off, so the default matches the
  // scored kernel.
  [opts setFastMathEnabled:NO];
  NSError *err = nil;
  uint64_t t0 = mach_absolute_time();
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
  double compile_s = seconds_since(t0);
  if (!lib) {
    fprintf(stderr, "e118_qmv_probe: %s arm failed to compile: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  fprintf(stderr, "e118_qmv_probe: %s compiled in %.2fs  src=%zu bytes\n",
          label, compile_s, (size_t)[src length]);
  return lib;
}

static id<MTLComputePipelineState> psoFor(id<MTLDevice> device,
                                          id<MTLLibrary> lib, NSString *fn,
                                          const char *label) {
  id<MTLFunction> f = [lib newFunctionWithName:fn];
  if (!f) {
    fprintf(stderr, "e118_qmv_probe: %s arm has no function %s\n", label,
            [fn UTF8String]);
    exit(1);
  }
  NSError *err = nil;
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:f error:&err];
  if (!pso) {
    fprintf(stderr, "e118_qmv_probe: %s arm pipeline failed: %s\n", label,
            err ? [[err localizedDescription] UTF8String] : "unknown");
    exit(1);
  }
  fprintf(stderr, "e118_qmv_probe:   %s %s  max_threads=%lu  tg_mem=%lu\n",
          label, [fn UTF8String], (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.staticThreadgroupMemoryLength);
  return pso;
}

// One shared weight, metadata and activation buffer for every arm: the arms
// differ only in kernel code, so any operand difference would confound the
// bit-for-bit comparison. `packed_sb` is derived from `scales` and `biases`, so
// the pack32 arms read exactly the values every other arm reads.
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
  o.packed_sb = [device newBufferWithLength:groups * 4
                                    options:MTLResourceStorageModeShared];
  o.bias_codes = [device newBufferWithLength:groups
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
  uint32_t *sbp = (uint32_t *)o.packed_sb.contents;
  uint8_t *cp = (uint8_t *)o.bias_codes.contents;
  for (size_t g = 0; g < groups; g++) {
    float s = 0.004f + 0.004f * unit(&seed);
    sp[g] = f32_to_bf16(s);
    uint32_t code = xorshift32(&seed) & 0x3fu;
    cp[g] = (uint8_t)code;
    bp[g] = bias_bf16_from_code(bf16_to_f32(sp[g]), code);
    sbp[g] = (uint32_t)sp[g] | ((uint32_t)bp[g] << 16);
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
  [enc setBuffer:o->packed_sb offset:0 atIndex:7];
  [enc setBuffer:o->bias_codes offset:0 atIndex:8];
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

// Harness defect 16. Every `gpuTempC` call leaves the GPU idle for about a
// second, and the clocks take 30 to 80 ms to come back. This burns a fixed
// wall-clock budget of discarded work so the ramp is paid before the first
// timed slot instead of inside it.
static void rampBurst(id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pso, Operands *o, int arm,
                      int m, double seconds) {
  uint64_t t0 = mach_absolute_time();
  while (seconds_since(t0) < seconds) {
    runArm(queue, pso, o, arm, m, 1, 2);
  }
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

typedef struct {
  size_t differing;
  int first_m;
  int first_n;
  int max_ulp;
  double max_rel;
} DiffReport;

// bf16 ordered index, so an ULP distance is a plain integer subtraction.
static int bf16_order(uint16_t bits) {
  return (bits & 0x8000u) ? (int)(0x8000u - (bits & 0x7fffu))
                          : (int)(0x8000u + bits);
}

static DiffReport countDiffering(Operands *o, int arm, int m) {
  const uint16_t *ya = (const uint16_t *)o->y[0].contents;
  const uint16_t *yb = (const uint16_t *)o->y[arm].contents;
  DiffReport r = (DiffReport){0, -1, -1, 0, 0.0};
  size_t total = (size_t)m * o->n;
  for (size_t i = 0; i < total; i++) {
    if (ya[i] == yb[i]) {
      continue;
    }
    if (!r.differing) {
      r.first_m = (int)(i / o->n);
      r.first_n = (int)(i % o->n);
    }
    r.differing++;
    int ulp = bf16_order(ya[i]) - bf16_order(yb[i]);
    if (ulp < 0) {
      ulp = -ulp;
    }
    if (ulp > r.max_ulp) {
      r.max_ulp = ulp;
    }
    double fa = (double)bf16_to_f32(ya[i]), fb = (double)bf16_to_f32(yb[i]);
    double den = fabs(fa) > fabs(fb) ? fabs(fa) : fabs(fb);
    if (den > 0.0) {
      double rel = fabs(fa - fb) / den;
      if (rel > r.max_rel) {
        r.max_rel = rel;
      }
    }
  }
  return r;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *dir = "/tmp/e118-arms", *out_path = NULL;
    const char *fn_name = "e118_iso_na%d";
    const char *widths_arg = "2,3,4,5";
    const char *shapes_arg = "0,1,2,3,4";
    const char *arms_arg = NULL;
    int pairs = 5, samples = 48;
    double target_ms = 40.0, ramp_ms = 150.0;

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
      else if (!strcmp(argv[i], "--ramp-ms") && i + 1 < argc) ramp_ms = atof(argv[++i]);
      else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arms_arg = argv[++i];
      else {
        fprintf(stderr, "e118_qmv_probe: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path || !arms_arg) {
      fprintf(stderr,
              "usage: e118_qmv_probe --out JSON --arms LIST [--dir ARMS] "
              "[--fn NAME] [--widths L] [--shapes L] [--pairs N] "
              "[--samples N] [--macmon PATH] [--target-ms MS] [--ramp-ms MS]\n");
      return 2;
    }

    {
      char buf[1024];
      snprintf(buf, sizeof(buf), "%s", arms_arg);
      int n = 0;
      for (char *tok = strtok(buf, ","); tok && n < MAXARM;
           tok = strtok(NULL, ",")) {
        char *mark = strstr(tok, ":diag");
        if (mark) *mark = '\0';
        kArmName[n] = strdup(tok);
        kArmExactVsBase[n] = mark ? 0 : 1;
        n++;
      }
      if (n < 2) {
        fprintf(stderr, "e118_qmv_probe: --arms needs 2 to %d names\n", MAXARM);
        return 2;
      }
      g_narm = n;
      if (!kArmExactVsBase[0]) {
        fprintf(stderr,
                "e118_qmv_probe: arm 0 is the exactness reference and must not "
                "be a diagnostic arm\n");
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
      snprintf(path, sizeof(path), "%s/arm_%s.metal", dir, kArmName[a]);
      id<MTLLibrary> lib = buildArm(device, path, kArmName[a]);
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
      fprintf(stderr, "e118_qmv_probe: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out, "  \"function\": \"%s\",\n  \"pairs\": %d,\n", fn_name, pairs);
    fprintf(out, "  \"order\": \"palindrome\",\n  \"arm_count\": %d,\n", g_narm);
    fprintf(out, "  \"ramp_ms\": %.1f,\n  \"target_ms\": %.1f,\n", ramp_ms,
            target_ms);
    fprintf(out, "  \"harness\": \"local\",\n");
    fprintf(out, "  \"defect16_fix\": \"temperature sampled before warm-up, "
                 "then a discarded ramp burst, then timing\",\n");
    fprintf(out, "  \"arms\": [");
    for (int a = 0; a < g_narm; a++) {
      fprintf(out, "%s\"%s\"", a ? ", " : "", kArmName[a]);
    }
    fprintf(out, "],\n  \"arm_exact_required\": [");
    for (int a = 0; a < g_narm; a++) {
      fprintf(out, "%s%s", a ? ", " : "",
              kArmExactVsBase[a] ? "true" : "false");
    }
    fprintf(out, "],\n  \"measurements\": [\n");

    int first_row = 1;
    for (int s = 0; s < n_shapes; s++) {
      Operands o = makeOperands(device, shapes[s], max_m);
      fprintf(stderr, "e118_qmv_probe: shape %s  w=%.1fMB\n", shapes[s].name,
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
                "e118_qmv_probe:   fidelity M=%d  base_vs_double max_rel=%.3e "
                "rms=%.3e\n", m, max_rel, rms);
        for (int a = 1; a < g_narm; a++) {
          DiffReport d = countDiffering(&o, a, m);
          size_t total = (size_t)m * o.n;
          fprintf(out,
                  "%s{\"arm\":\"%s\",\"exact_required\":%s,\"differing\":%zu,"
                  "\"total\":%zu,\"bit_identical\":%s,\"first_bad_m\":%d,"
                  "\"first_bad_n\":%d,\"max_ulp\":%d,\"max_rel\":%.6e}",
                  a > 1 ? "," : "", kArmName[a],
                  kArmExactVsBase[a] ? "true" : "false", d.differing, total,
                  d.differing ? "false" : "true", d.first_m, d.first_n,
                  d.max_ulp, d.max_rel);
          fprintf(stderr,
                  "e118_qmv_probe:     %-15s vs base differing=%zu/%zu "
                  "max_ulp=%d max_rel=%.3e%s\n",
                  kArmName[a], d.differing, total, d.max_ulp, d.max_rel,
                  kArmExactVsBase[a]
                      ? (d.differing ? "   *** EXACTNESS FAILURE ***" : "   exact")
                      : "   (diagnostic arm, difference expected)");
          if (kArmExactVsBase[a] && d.differing) {
            const uint16_t *yb = (const uint16_t *)o.y[a].contents;
            fprintf(stderr,
                    "e118_qmv_probe:       FIRST MISMATCH m=%d n=%d "
                    "base=0x%04x arm=0x%04x\n",
                    d.first_m, d.first_n,
                    ya[(size_t)d.first_m * o.n + d.first_n],
                    yb[(size_t)d.first_m * o.n + d.first_n]);
          }
        }
        fprintf(out, "]}");
      }

      // --- positive controls -------------------------------------------------
      // Two perturbations, one per operand path an arm of this experiment can
      // change, applied to every arm whose output must match `a_base`. Only the
      // candidate arm is re-dispatched, so a comparison that cannot fail is
      // visible immediately. The metadata perturbation moves `scales`,
      // `biases`, `packed_sb` and `bias_codes` together, so it reaches the
      // pack32 and Bias6 arms too.
      {
        const int m = widths[0];
        const size_t gpr = (size_t)o.k / 64;
        {
          const uint16_t *y0 = (const uint16_t *)o.y[0].contents;
          size_t zeros = 0;
          for (size_t i = 0; i < (size_t)m * o.n; i++) {
            if (y0[i] == 0) zeros++;
          }
          fprintf(stderr, "e118_qmv_probe:   DIAG y0 zeros=%zu/%zu gpr=%zu\n",
                  zeros, (size_t)m * o.n, gpr);
        }
        uint16_t *xp = (uint16_t *)o.x.contents;
        uint16_t *sp = (uint16_t *)o.scales.contents;
        uint16_t *bp = (uint16_t *)o.biases.contents;
        uint32_t *sbp = (uint32_t *)o.packed_sb.contents;
        uint8_t *cp = (uint8_t *)o.bias_codes.contents;
        for (int a = 1; a < g_narm; a++) {
          if (!kArmExactVsBase[a]) continue;
          size_t hit_x = 0, hit_meta = 0;

          uint16_t saved_x = xp[0];
          xp[0] = f32_to_bf16(bf16_to_f32(saved_x) * 1.5f + 0.25f);
          dispatchOnce(queue, pso[a][0], &o, a, m);
          hit_x = countDiffering(&o, a, m).differing;
          xp[0] = saved_x;

          // One group out of `in_vec_size / 64` contributes to an output that
          // is rounded to BF16, so a small perturbation can round away. The
          // factor has to dominate the whole row for the control to be a real
          // detector rather than a coincidence. The perturbation is spread over
          // rows taken from across the output so it cannot land only on rows a
          // given dispatch geometry leaves untouched.
          const size_t nmeta = (size_t)kMetaProbeRows * gpr;
          uint16_t *save_s = malloc(nmeta * sizeof(uint16_t));
          uint16_t *save_b = malloc(nmeta * sizeof(uint16_t));
          uint32_t *save_sb = malloc(nmeta * sizeof(uint32_t));
          uint8_t *save_c = malloc(nmeta);
          for (int j = 0; j < kMetaProbeRows; j++) {
            const size_t base = ((size_t)o.n * j / kMetaProbeRows + 1) * gpr;
            for (size_t g = 0; g < gpr; g++) {
              const size_t gi = base + g, si = (size_t)j * gpr + g;
              save_s[si] = sp[gi];
              save_b[si] = bp[gi];
              save_sb[si] = sbp[gi];
              save_c[si] = cp[gi];
              sp[gi] = f32_to_bf16(bf16_to_f32(save_s[si]) * 512.0f + 1e-3f);
              cp[gi] = (uint8_t)((save_c[si] + 9u) & 0x3fu);
              bp[gi] = bias_bf16_from_code(bf16_to_f32(sp[gi]), cp[gi]);
              sbp[gi] = (uint32_t)sp[gi] | ((uint32_t)bp[gi] << 16);
            }
          }
          dispatchOnce(queue, pso[a][0], &o, a, m);
          hit_meta = countDiffering(&o, a, m).differing;
          if (a == 1) {
            const uint16_t *y0 = (const uint16_t *)o.y[0].contents;
            const uint16_t *ya2 = (const uint16_t *)o.y[a].contents;
            for (int j = 0; j < kMetaProbeRows; j++) {
              const int col = (int)((size_t)o.n * j / kMetaProbeRows + 1);
              fprintf(stderr,
                      "e118_qmv_probe:   DIAG col=%d base=%.4f pert=%.4f\n", col,
                      bf16_to_f32(y0[col]), bf16_to_f32(ya2[col]));
            }
          }
          for (int j = 0; j < kMetaProbeRows; j++) {
            const size_t base = ((size_t)o.n * j / kMetaProbeRows + 1) * gpr;
            for (size_t g = 0; g < gpr; g++) {
              const size_t gi = base + g, si = (size_t)j * gpr + g;
              sp[gi] = save_s[si];
              bp[gi] = save_b[si];
              sbp[gi] = save_sb[si];
              cp[gi] = save_c[si];
            }
          }
          free(save_s);
          free(save_b);
          free(save_sb);
          free(save_c);

          dispatchOnce(queue, pso[a][0], &o, a, m);
          size_t clean = countDiffering(&o, a, m).differing;
          fprintf(stderr,
                  "e118_qmv_probe:   control %-15s x_hit=%zu meta_hit=%zu "
                  "restored_diff=%zu%s\n",
                  kArmName[a], hit_x, hit_meta, clean,
                  (hit_x && hit_meta && !clean) ? "" : "   *** CONTROL FAILED ***");
          fprintf(out,
                  ",\n    {\"kind\":\"positive_control\",\"shape\":\"%s\","
                  "\"m\":%d,\"arm\":\"%s\",\"activation_perturbed_differing\":"
                  "%zu,\"metadata_perturbed_differing\":%zu,"
                  "\"restored_differing\":%zu,\"detected\":%s}",
                  shapes[s].name, m, kArmName[a], hit_x, hit_meta, clean,
                  (hit_x && hit_meta && !clean) ? "true" : "false");
        }
      }

      // --- calibrate, sample temperature, ramp, warm, then time -------------
      for (int wi = 0; wi < n_widths; wi++) {
        int m = widths[wi];
        runArm(queue, pso[0][wi], &o, 0, m, 1, 1);
        double probe = runArm(queue, pso[0][wi], &o, 0, m, 1, 1);
        int inner = (int)(target_ms * 1e-3 / probe);
        if (inner < 1) inner = 1;
        if (inner > 64) inner = 64;
        const int reps = 3;

        // Defect 16: the temperature sample idles the GPU, so it goes here and
        // is followed by discarded work. The per-arm warm-up below then runs on
        // clocks that are already back up, and no timed slot pays the ramp.
        double entry_c = gpuTempC();
        rampBurst(queue, pso[0][wi], &o, 0, m, ramp_ms * 1e-3);

        for (int a = 0; a < g_narm; a++) {
          runArm(queue, pso[a][wi], &o, a, m, 1, inner);
        }

        const int slots = 2 * g_narm;
        for (int p = 0; p < pairs; p++) {
          // AB..BA palindrome: linear drift inside the block cancels for every
          // arm at any arm count. It does NOT cancel a fixed first-slot cost,
          // which is why the ramp is paid above and block 0 is discarded in
          // analysis.
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

          fprintf(stderr, "e118_qmv_probe:   %s M=%d block %d inner=%d",
                  shapes[s].name, m, p, inner);
          for (int a = 0; a < g_narm; a++) {
            fprintf(stderr, "  %s=%.1fus", kArmName[a], sec[a] * 1e6);
          }
          fprintf(stderr, "\n");

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
                "e118_qmv_probe:   %s M=%d thermal entry=%.1fC exit=%.1fC\n",
                shapes[s].name, m, entry_c, exit_c);
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e118_qmv_probe: wrote %s\n", out_path);
  }
  return 0;
}
