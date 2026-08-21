// E103 rung 2: price the arms of the scored full-attention decode SDPA kernel
// in isolation, in one process, in one thermal session.
//
// The scored dispatch is
//   sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks grid=24x5x1 tg=1024x1x1
// at verify width 5, and the `_nc` grid=24x1x1 form at width 1. The trusted
// dispatcher owns that grid, so only arms a, b, c and g could ever ship inside
// `sdpa_vector.h`. Arms d_pack* need an editable Swift-side custom kernel and
// are measured here purely to price that route before paying for it.
//
// Arms:
//   a_shipped_c   verbatim transcription of the shipped kernel
//   b_vecload_c   4-wide K/V loads, arithmetic order unchanged -> bit-identical
//   c_fastpath_c  b, plus skip the rescale when the running max does not move
//   d_pack{1,2,3,6}_c  P query heads per threadgroup, grid (24/P, M, 1)
//   e_resident_c  traffic-free control: K/V pointers never advance
//   f_nosoftmax_c softmax-free control
//   g_double_c    positive control: the key loop runs twice, must show ~2x
//
// Every arm is compared BIT FOR BIT against arm a. Arms b, d_pack1, d_pack2,
// d_pack3 and d_pack6 must match exactly; c may differ and is reported either
// way; e, f and g change the answer on purpose and are timing-only.
//
// K and V rotate over `--kv-copies` independent slices so the measurement sees
// a realistic cache footprint instead of one permanently resident tile.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e103_sdpa_ab research/e103_sdpa_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ARMS 16

typedef struct {
  const char *name;
  int pack;          // query heads per threadgroup; 1 for the unpacked arms
  int bit_exact;     // 1 when the arm must match arm a bit for bit
  int correctness;   // 0 for the timing-only controls
} ArmSpec;

static const ArmSpec kArms[] = {
    {"a_shipped_c", 1, 1, 1},  {"b_vecload_c", 1, 1, 1},
    {"c_fastpath_c", 1, 0, 1}, {"d_pack1_c", 1, 1, 1},
    {"d_pack2_c", 2, 1, 1},    {"d_pack3_c", 3, 1, 1},
    {"d_pack6_c", 6, 1, 1},    {"e_resident_c", 1, 0, 0},
    {"f_nosoftmax_c", 1, 0, 0}, {"g_double_c", 1, 0, 0},
    {"h_tailfree_c", 1, 0, 0}, {"j_launchonly_c", 1, 0, 0},
};
static const int kNumArms = (int)(sizeof(kArms) / sizeof(kArms[0]));

static const int kHeads = 24;
static const int kKVHeads = 4;
static const int kDim = 256;

// --- bf16 ------------------------------------------------------------------

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

static inline float unit(uint32_t *s) {
  return (float)(xorshift32(s) >> 8) / (float)(1u << 24);
}

static double g_ns_per_tick = 0.0;
static uint64_t g_session_start = 0;

static double seconds_since(uint64_t start) {
  return (double)(mach_absolute_time() - start) * g_ns_per_tick * 1e-9;
}

typedef struct {
  id<MTLBuffer> queries;
  id<MTLBuffer> keys;
  id<MTLBuffer> values;
  id<MTLBuffer> out[MAX_ARMS];
  int n;          // history length
  int m;          // verify rows
  int kv_copies;
  size_t kv_slice_bytes;
  float scale;
} Operands;

static Operands makeOperands(id<MTLDevice> device, int n, int m, int kv_copies) {
  Operands o = (Operands){};
  o.n = n;
  o.m = m;
  o.kv_copies = kv_copies;
  o.scale = 1.0f / sqrtf((float)kDim);
  const size_t q_elems = (size_t)kHeads * m * kDim;
  const size_t kv_elems = (size_t)kKVHeads * n * kDim;
  o.kv_slice_bytes = kv_elems * 2;

  o.queries = [device newBufferWithLength:q_elems * 2
                                  options:MTLResourceStorageModeShared];
  o.keys = [device newBufferWithLength:o.kv_slice_bytes * kv_copies
                               options:MTLResourceStorageModeShared];
  o.values = [device newBufferWithLength:o.kv_slice_bytes * kv_copies
                                 options:MTLResourceStorageModeShared];
  for (int a = 0; a < kNumArms; a++) {
    o.out[a] = [device newBufferWithLength:q_elems * 2
                                   options:MTLResourceStorageModeShared];
    memset(o.out[a].contents, 0, o.out[a].length);
  }

  uint32_t seed = 0x51ed270bu;
  uint16_t *qp = (uint16_t *)o.queries.contents;
  for (size_t i = 0; i < q_elems; i++) {
    qp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  uint16_t *kp = (uint16_t *)o.keys.contents;
  uint16_t *vp = (uint16_t *)o.values.contents;
  for (size_t i = 0; i < kv_elems * (size_t)kv_copies; i++) {
    kp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
    vp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  return o;
}

// Exact reference for one output element, in double, from slice 0.
static double referenceElement(const Operands *o, int head, int row, int dim) {
  const uint16_t *q = (const uint16_t *)o->queries.contents;
  const uint16_t *k = (const uint16_t *)o->keys.contents;
  const uint16_t *v = (const uint16_t *)o->values.contents;
  const int kv_head = head / (kHeads / kKVHeads);
  const size_t q_base = ((size_t)head * o->m + row) * kDim;
  const size_t kv_base = (size_t)kv_head * o->n * kDim;
  const int last = o->n - o->m + row;

  double best = -INFINITY;
  for (int i = 0; i <= last; i++) {
    double s = 0.0;
    for (int d = 0; d < kDim; d++) {
      s += (double)o->scale * bf16_to_f32(q[q_base + d]) *
           bf16_to_f32(k[kv_base + (size_t)i * kDim + d]);
    }
    if (s > best) best = s;
  }
  double denom = 0.0, num = 0.0;
  for (int i = 0; i <= last; i++) {
    double s = 0.0;
    for (int d = 0; d < kDim; d++) {
      s += (double)o->scale * bf16_to_f32(q[q_base + d]) *
           bf16_to_f32(k[kv_base + (size_t)i * kDim + d]);
    }
    double e = exp(s - best);
    denom += e;
    num += e * bf16_to_f32(v[kv_base + (size_t)i * kDim + dim]);
  }
  return denom == 0.0 ? 0.0 : num / denom;
}

static void encodeDispatch(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso, const Operands *o,
                           int arm, int slice) {
  const int pack = kArms[arm].pack;
  const int gqa = kHeads / kKVHeads;
  const int n = o->n;
  const size_t k_head_stride = (size_t)n * kDim;
  const size_t k_seq_stride = (size_t)kDim;
  const float scale = o->scale;
  const size_t off = (size_t)slice * o->kv_slice_bytes;

  [enc setComputePipelineState:pso];
  [enc setBuffer:o->queries offset:0 atIndex:0];
  [enc setBuffer:o->keys offset:off atIndex:1];
  [enc setBuffer:o->values offset:off atIndex:2];
  [enc setBuffer:o->out[arm] offset:0 atIndex:3];
  [enc setBytes:&gqa length:sizeof(int) atIndex:4];
  [enc setBytes:&n length:sizeof(int) atIndex:5];
  [enc setBytes:&k_head_stride length:sizeof(size_t) atIndex:6];
  [enc setBytes:&k_seq_stride length:sizeof(size_t) atIndex:7];
  [enc setBytes:&k_head_stride length:sizeof(size_t) atIndex:8];
  [enc setBytes:&k_seq_stride length:sizeof(size_t) atIndex:9];
  [enc setBytes:&scale length:sizeof(float) atIndex:10];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(kHeads / pack),
                                        (NSUInteger)o->m, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

static double runArm(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                     const Operands *o, int arm, int reps, int inner,
                     int *slice) {
  uint64_t t0 = mach_absolute_time();
  for (int r = 0; r < reps; r++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      encodeDispatch(enc, pso, o, arm, *slice);
      *slice = (*slice + 1) % o->kv_copies;
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  return seconds_since(t0) / (double)(reps * inner);
}

static void dispatchOnce(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pso, const Operands *o,
                         int arm) {
  memset(o->out[arm].contents, 0, o->out[arm].length);
  int slice = 0;
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeDispatch(enc, pso, o, arm, slice);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

// Bytes the arm must read from the K and V slices for one dispatch, if nothing
// were cached. Each of the H query rows streams the whole history of its KV
// head, so packing P heads divides that by P.
static double armReadBytes(const Operands *o, int arm) {
  const int pack = kArms[arm].pack;
  double kv = 2.0 * (double)o->n * kDim * 2.0;  // K and V, bf16
  return (double)kHeads / pack * (double)o->m * kv;
}

static double logicalBytes(const Operands *o) {
  return 2.0 * (double)kKVHeads * o->n * kDim * 2.0;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *src_path = "research/e103_sdpa_arms.metal";
    const char *out_path = NULL;
    const char *widths_arg = "1,2,3,4,5";
    const char *lens_arg = "512,768,1024";
    int pairs = 3, kv_copies = 16, inner_cap = 512;
    double target_ms = 40.0;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--src") && i + 1 < argc) src_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--widths") && i + 1 < argc) widths_arg = argv[++i];
      else if (!strcmp(argv[i], "--lens") && i + 1 < argc) lens_arg = argv[++i];
      else if (!strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--kv-copies") && i + 1 < argc) kv_copies = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inner-cap") && i + 1 < argc) inner_cap = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--target-ms") && i + 1 < argc) target_ms = atof(argv[++i]);
      else {
        fprintf(stderr, "e103_sdpa_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path) {
      fprintf(stderr, "usage: e103_sdpa_ab --out JSON [--src FILE] "
                      "[--widths L] [--lens L] [--pairs N] [--kv-copies N] "
                      "[--inner-cap N] [--target-ms MS]\n");
      return 2;
    }

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;
    g_session_start = mach_absolute_time();

    int widths[16], n_widths = 0;
    for (const char *p = widths_arg; *p && n_widths < 16;) {
      widths[n_widths++] = atoi(p);
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }
    int lens[16], n_lens = 0;
    for (const char *p = lens_arg; *p && n_lens < 16;) {
      lens[n_lens++] = atoi(p);
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "e103_sdpa_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e103_sdpa_ab: cannot read %s\n", src_path);
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
    uint64_t t0 = mach_absolute_time();
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
      fprintf(stderr, "e103_sdpa_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }
    fprintf(stderr, "e103_sdpa_ab: compiled %d arms in %.2fs\n", kNumArms,
            seconds_since(t0));

    id<MTLComputePipelineState> pso[MAX_ARMS];
    for (int a = 0; a < kNumArms; a++) {
      id<MTLFunction> f = [lib newFunctionWithName:@(kArms[a].name)];
      if (!f) {
        fprintf(stderr, "e103_sdpa_ab: no function %s\n", kArms[a].name);
        return 1;
      }
      pso[a] = [device newComputePipelineStateWithFunction:f error:&err];
      if (!pso[a]) {
        fprintf(stderr, "e103_sdpa_ab: pipeline %s failed: %s\n", kArms[a].name,
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      fprintf(stderr,
              "e103_sdpa_ab: %-14s max_threads=%lu tg_mem=%lu simd_width=%lu\n",
              kArms[a].name,
              (unsigned long)pso[a].maxTotalThreadsPerThreadgroup,
              (unsigned long)pso[a].staticThreadgroupMemoryLength,
              (unsigned long)pso[a].threadExecutionWidth);
      if (pso[a].maxTotalThreadsPerThreadgroup < 1024) {
        fprintf(stderr,
                "e103_sdpa_ab: FATAL %s cannot host 1024 threads (%lu)\n",
                kArms[a].name,
                (unsigned long)pso[a].maxTotalThreadsPerThreadgroup);
        return 1;
      }
    }

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e103_sdpa_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out, "  \"heads\": %d,\n  \"kv_heads\": %d,\n  \"head_dim\": %d,\n",
            kHeads, kKVHeads, kDim);
    fprintf(out, "  \"kv_copies\": %d,\n  \"pairs\": %d,\n", kv_copies, pairs);
    fprintf(out, "  \"order\": \"palindrome\",\n  \"arms\": [");
    for (int a = 0; a < kNumArms; a++) {
      fprintf(out, "%s\"%s\"", a ? ", " : "", kArms[a].name);
    }
    fprintf(out, "],\n  \"max_threads\": [");
    for (int a = 0; a < kNumArms; a++) {
      fprintf(out, "%s%lu", a ? ", " : "",
              (unsigned long)pso[a].maxTotalThreadsPerThreadgroup);
    }
    fprintf(out, "],\n  \"measurements\": [\n");

    int first_row = 1;
    for (int li = 0; li < n_lens; li++) {
      for (int wi = 0; wi < n_widths; wi++) {
        const int n = lens[li], m = widths[wi];
        Operands o = makeOperands(device, n, m, kv_copies);
        fprintf(stderr,
                "e103_sdpa_ab: cell N=%d M=%d  kv=%.1fMB x%d  logical=%.2fMB\n",
                n, m, (double)o.kv_slice_bytes / 1e6, kv_copies,
                logicalBytes(&o) / 1e6);

        // --- fidelity, before any timing ---------------------------------
        for (int a = 0; a < kNumArms; a++) {
          dispatchOnce(queue, pso[a], &o, a);
        }
        const uint16_t *ya = (const uint16_t *)o.out[0].contents;
        const size_t total = (size_t)kHeads * m * kDim;

        // arm a against an exact double reference, on a sparse grid.
        double max_abs = 0.0, sq_err = 0.0, sq_want = 0.0;
        for (int head = 0; head < kHeads; head += 5) {
          for (int row = 0; row < m; row++) {
            for (int dim = 0; dim < kDim; dim += 37) {
              double want = referenceElement(&o, head, row, dim);
              double got = bf16_to_f32(ya[((size_t)head * m + row) * kDim + dim]);
              double d = fabs(got - want);
              if (d > max_abs) max_abs = d;
              sq_err += d * d;
              sq_want += want * want;
            }
          }
        }
        double rms = sq_want > 0.0 ? sqrt(sq_err / sq_want) : 0.0;
        fprintf(stderr,
                "e103_sdpa_ab:   a vs double  max_abs=%.3e rms_over_signal=%.3e\n",
                max_abs, rms);
        fprintf(out,
                "%s    {\"kind\":\"reference\",\"n\":%d,\"m\":%d,"
                "\"max_abs\":%.6e,\"rms_over_signal\":%.6e}",
                first_row ? "" : ",\n", n, m, max_abs, rms);
        first_row = 0;

        for (int a = 1; a < kNumArms; a++) {
          const uint16_t *yb = (const uint16_t *)o.out[a].contents;
          size_t differing = 0;
          double worst = 0.0;
          for (size_t i = 0; i < total; i++) {
            if (ya[i] != yb[i]) {
              differing++;
              double d = fabs(bf16_to_f32(ya[i]) - bf16_to_f32(yb[i]));
              if (d > worst) worst = d;
            }
          }
          fprintf(stderr,
                  "e103_sdpa_ab:   %-14s vs a  differing=%zu/%zu worst=%.3e%s\n",
                  kArms[a].name, differing, total, worst,
                  (kArms[a].bit_exact && differing) ? "   <-- BIT-EXACT VIOLATION"
                                                    : "");
          fprintf(out,
                  ",\n    {\"kind\":\"fidelity\",\"n\":%d,\"m\":%d,"
                  "\"arm\":\"%s\",\"expect_bit_exact\":%s,"
                  "\"differing\":%zu,\"total\":%zu,\"worst_abs\":%.6e,"
                  "\"bit_identical\":%s}",
                  n, m, kArms[a].name, kArms[a].bit_exact ? "true" : "false",
                  differing, total, worst, differing ? "false" : "true");
        }

        // --- positive control: perturb one key, arm a must react ----------
        {
          uint16_t *kp = (uint16_t *)o.keys.contents;
          uint16_t saved = kp[0];
          kp[0] = f32_to_bf16(bf16_to_f32(saved) + 0.25f);
          uint16_t *snap = malloc(total * 2);
          memcpy(snap, ya, total * 2);
          dispatchOnce(queue, pso[0], &o, 0);
          size_t differing = 0;
          for (size_t i = 0; i < total; i++) {
            if (snap[i] != ya[i]) differing++;
          }
          free(snap);
          kp[0] = saved;
          dispatchOnce(queue, pso[0], &o, 0);
          fprintf(stderr,
                  "e103_sdpa_ab:   positive control: one key perturbed -> "
                  "differing=%zu/%zu\n", differing, total);
          fprintf(out,
                  ",\n    {\"kind\":\"positive_control\",\"n\":%d,\"m\":%d,"
                  "\"differing\":%zu,\"total\":%zu,\"detected\":%s}",
                  n, m, differing, total, differing ? "true" : "false");
        }

        // --- calibrate, warm, then time -----------------------------------
        int slice = 0;
        runArm(queue, pso[0], &o, 0, 1, 1, &slice);
        double probe = runArm(queue, pso[0], &o, 0, 1, 1, &slice);
        int inner = (int)(target_ms * 1e-3 / probe);
        if (inner < 1) inner = 1;
        if (inner > inner_cap) inner = inner_cap;
        const int reps = 2;
        for (int a = 0; a < kNumArms; a++) {
          runArm(queue, pso[a], &o, a, 1, inner, &slice);
        }

        for (int p = 0; p < pairs; p++) {
          double t[2 * MAX_ARMS];
          double at = seconds_since(g_session_start);
          for (int s = 0; s < 2 * kNumArms; s++) {
            int a = s < kNumArms ? s : (2 * kNumArms - 1 - s);
            t[s] = runArm(queue, pso[a], &o, a, reps, inner, &slice);
          }
          double sec[MAX_ARMS];
          for (int a = 0; a < kNumArms; a++) {
            sec[a] = 0.5 * (t[a] + t[2 * kNumArms - 1 - a]);
          }
          fprintf(stderr, "e103_sdpa_ab:   N=%d M=%d block %d inner=%d", n, m,
                  p, inner);
          for (int a = 0; a < kNumArms; a++) {
            fprintf(stderr, "  %s=%.2fus", kArms[a].name, sec[a] * 1e6);
          }
          fprintf(stderr, "\n");
          fprintf(out,
                  ",\n    {\"kind\":\"timing\",\"n\":%d,\"m\":%d,\"block\":%d,"
                  "\"inner\":%d,\"reps\":%d,\"session_elapsed_s\":%.3f,"
                  "\"logical_bytes\":%.0f,\"seconds\":{",
                  n, m, p, inner, reps, at, logicalBytes(&o));
          for (int a = 0; a < kNumArms; a++) {
            fprintf(out, "%s\"%s\":%.9e", a ? "," : "", kArms[a].name, sec[a]);
          }
          fprintf(out, "},\"stream_bytes\":{");
          for (int a = 0; a < kNumArms; a++) {
            fprintf(out, "%s\"%s\":%.0f", a ? "," : "", kArms[a].name,
                    armReadBytes(&o, a));
          }
          fprintf(out, "},\"slots\":[");
          for (int s = 0; s < 2 * kNumArms; s++) {
            fprintf(out, "%s%.9e", s ? "," : "", t[s]);
          }
          fprintf(out, "]}");
        }
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e103_sdpa_ab: wrote %s\n", out_path);
  }
  return 0;
}
