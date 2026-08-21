// E103 rung 2b: price the qL <= 5 partition that the trusted dispatcher forces
// on every verify width M >= 6.
//
// `supports_sdpa_vector` requires qL * gqa_factor <= 32, and with gqa 6 that
// caps qL at 5, so the editable Swift layer splits a width-M verify into a
// qL = 5 dispatch plus a qL = (M - 5) dispatch over the same history. Each
// dispatch pays the whole per-dispatch fixed cost, so the history is walked
// twice per layer at M >= 6.
//
// This harness measures, in one process and one queue, with palindrome
// ordering:
//
//   single    one dispatch, grid (24, M, 1), history N
//   split     two dispatches, grid (24, 5, 1) over history N-(M-5) then
//             grid (24, M-5, 1) over history N     <- what ships today
//   only5     the first leg of the split alone
//   onlyr     the second leg of the split alone
//
// The causal bound inside the kernel is `i <= N - tpg.y + q_seq_idx`, so the
// (N - r, qL = 5) leg reproduces rows 0..4 of the width-M problem exactly and
// the (N, qL = r) leg reproduces rows 5..M-1 exactly. That is asserted bit for
// bit before any timing, which also proves that merging the split back into one
// dispatch is an exactness-neutral change.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e103_split_ab research/e103_split_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kHeads = 24;
static const int kKVHeads = 4;
static const int kDim = 256;
static const int kMaxQL = 5;  // the trusted qL * gqa <= 32 cap at gqa = 6
static const int kNumVariants = 4;
static const char *kVariantNames[4] = {"single", "split", "only5", "onlyr"};

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
  id<MTLBuffer> keys;
  id<MTLBuffer> values;
  id<MTLBuffer> q_full;   // [24, M, D]
  id<MTLBuffer> q_head;   // [24, 5, D], rows 0..4 of q_full
  id<MTLBuffer> q_tail;   // [24, r, D], rows 5..M-1 of q_full
  id<MTLBuffer> o_full;
  id<MTLBuffer> o_head;
  id<MTLBuffer> o_tail;
  int n;                  // history length of the width-M problem
  int m;                  // verify rows
  int r;                  // remainder rows, m - 5
  int kv_copies;
  size_t kv_slice_bytes;
  float scale;
} Cell;

static Cell makeCell(id<MTLDevice> device, int n, int m, int kv_copies) {
  Cell c = (Cell){};
  c.n = n;
  c.m = m;
  c.r = m - kMaxQL;
  c.kv_copies = kv_copies;
  c.scale = 1.0f / sqrtf((float)kDim);

  const size_t kv_elems = (size_t)kKVHeads * n * kDim;
  c.kv_slice_bytes = kv_elems * 2;
  c.keys = [device newBufferWithLength:c.kv_slice_bytes * kv_copies
                               options:MTLResourceStorageModeShared];
  c.values = [device newBufferWithLength:c.kv_slice_bytes * kv_copies
                                 options:MTLResourceStorageModeShared];

  const size_t q_full_elems = (size_t)kHeads * m * kDim;
  const size_t q_head_elems = (size_t)kHeads * kMaxQL * kDim;
  const size_t q_tail_elems = (size_t)kHeads * c.r * kDim;
  c.q_full = [device newBufferWithLength:q_full_elems * 2
                                 options:MTLResourceStorageModeShared];
  c.q_head = [device newBufferWithLength:q_head_elems * 2
                                 options:MTLResourceStorageModeShared];
  c.q_tail = [device newBufferWithLength:q_tail_elems * 2
                                 options:MTLResourceStorageModeShared];
  c.o_full = [device newBufferWithLength:q_full_elems * 2
                                 options:MTLResourceStorageModeShared];
  c.o_head = [device newBufferWithLength:q_head_elems * 2
                                 options:MTLResourceStorageModeShared];
  c.o_tail = [device newBufferWithLength:q_tail_elems * 2
                                 options:MTLResourceStorageModeShared];
  memset(c.o_full.contents, 0, c.o_full.length);
  memset(c.o_head.contents, 0, c.o_head.length);
  memset(c.o_tail.contents, 0, c.o_tail.length);

  uint32_t seed = 0x51ed270bu;
  uint16_t *qp = (uint16_t *)c.q_full.contents;
  for (size_t i = 0; i < q_full_elems; i++) {
    qp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  uint16_t *kp = (uint16_t *)c.keys.contents;
  uint16_t *vp = (uint16_t *)c.values.contents;
  for (size_t i = 0; i < kv_elems * (size_t)kv_copies; i++) {
    kp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
    vp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }

  // The two split legs read exactly the same query rows as the single pass.
  uint16_t *qh = (uint16_t *)c.q_head.contents;
  uint16_t *qt = (uint16_t *)c.q_tail.contents;
  for (int h = 0; h < kHeads; h++) {
    for (int j = 0; j < kMaxQL; j++) {
      memcpy(qh + ((size_t)h * kMaxQL + j) * kDim,
             qp + ((size_t)h * m + j) * kDim, (size_t)kDim * 2);
    }
    for (int j = 0; j < c.r; j++) {
      memcpy(qt + ((size_t)h * c.r + j) * kDim,
             qp + ((size_t)h * m + kMaxQL + j) * kDim, (size_t)kDim * 2);
    }
  }
  return c;
}

// One `sdpa_a_shipped` dispatch. `n_eff` is the causal history length the leg
// sees; `head_stride` always describes the physical K and V layout, which is
// the width-M history, so a shorter leg reads a prefix of the same buffer.
static void encodeLeg(id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso, const Cell *c,
                      id<MTLBuffer> q, id<MTLBuffer> o, int rows, int n_eff,
                      int slice) {
  const int gqa = kHeads / kKVHeads;
  const size_t head_stride = (size_t)c->n * kDim;
  const size_t seq_stride = (size_t)kDim;
  const float scale = c->scale;
  const size_t off = (size_t)slice * c->kv_slice_bytes;

  [enc setComputePipelineState:pso];
  [enc setBuffer:q offset:0 atIndex:0];
  [enc setBuffer:c->keys offset:off atIndex:1];
  [enc setBuffer:c->values offset:off atIndex:2];
  [enc setBuffer:o offset:0 atIndex:3];
  [enc setBytes:&gqa length:sizeof(int) atIndex:4];
  [enc setBytes:&n_eff length:sizeof(int) atIndex:5];
  [enc setBytes:&head_stride length:sizeof(size_t) atIndex:6];
  [enc setBytes:&seq_stride length:sizeof(size_t) atIndex:7];
  [enc setBytes:&head_stride length:sizeof(size_t) atIndex:8];
  [enc setBytes:&seq_stride length:sizeof(size_t) atIndex:9];
  [enc setBytes:&scale length:sizeof(float) atIndex:10];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kHeads, (NSUInteger)rows, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

static void encodeVariant(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso, const Cell *c,
                          int variant, int *slice) {
  switch (variant) {
    case 0:  // single: one width-M dispatch
      encodeLeg(enc, pso, c, c->q_full, c->o_full, c->m, c->n, *slice);
      *slice = (*slice + 1) % c->kv_copies;
      break;
    case 1:  // split: what ships today
      encodeLeg(enc, pso, c, c->q_head, c->o_head, kMaxQL, c->n - c->r, *slice);
      encodeLeg(enc, pso, c, c->q_tail, c->o_tail, c->r, c->n, *slice);
      *slice = (*slice + 1) % c->kv_copies;
      break;
    case 2:  // only the qL = 5 leg
      encodeLeg(enc, pso, c, c->q_head, c->o_head, kMaxQL, c->n - c->r, *slice);
      *slice = (*slice + 1) % c->kv_copies;
      break;
    default:  // only the qL = r leg
      encodeLeg(enc, pso, c, c->q_tail, c->o_tail, c->r, c->n, *slice);
      *slice = (*slice + 1) % c->kv_copies;
      break;
  }
}

static double runVariant(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pso, const Cell *c,
                         int variant, int reps, int inner, int *slice) {
  uint64_t t0 = mach_absolute_time();
  for (int rep = 0; rep < reps; rep++) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int i = 0; i < inner; i++) {
      encodeVariant(enc, pso, c, variant, slice);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  return seconds_since(t0) / (double)(reps * inner);
}

static void runOnce(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                    const Cell *c, int variant) {
  int slice = 0;
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeVariant(enc, pso, c, variant, &slice);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *src_path = "research/e103_sdpa_arms.metal";
    const char *out_path = NULL;
    const char *widths_arg = "6,7,8";
    const char *lens_arg = "512,576,768,1024";
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
        fprintf(stderr, "e103_split_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!out_path) {
      fprintf(stderr, "usage: e103_split_ab --out JSON [--src FILE] "
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
      fprintf(stderr, "e103_split_ab: no Metal device\n");
      return 1;
    }
    NSString *arch = @"unknown";
    if (@available(macOS 14.0, *)) arch = [[device architecture] name];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@(src_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e103_split_ab: cannot read %s\n", src_path);
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
      fprintf(stderr, "e103_split_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"a_shipped_c"];
    if (!fn) {
      fprintf(stderr, "e103_split_ab: no function a_shipped_c\n");
      return 1;
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      fprintf(stderr, "e103_split_ab: pipeline failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }
    fprintf(stderr,
            "e103_split_ab: a_shipped_c max_threads=%lu tg_mem=%lu\n",
            (unsigned long)pso.maxTotalThreadsPerThreadgroup,
            (unsigned long)pso.staticThreadgroupMemoryLength);

    FILE *out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "e103_split_ab: cannot write %s\n", out_path);
      return 1;
    }
    fprintf(out, "{\n  \"device\": \"%s\",\n  \"architecture\": \"%s\",\n",
            [[device name] UTF8String], [arch UTF8String]);
    fprintf(out,
            "  \"heads\": %d,\n  \"kv_heads\": %d,\n  \"head_dim\": %d,\n"
            "  \"max_ql\": %d,\n  \"kv_copies\": %d,\n  \"pairs\": %d,\n"
            "  \"kernel\": \"a_shipped_c\",\n  \"order\": \"palindrome\",\n"
            "  \"variants\": [\"single\", \"split\", \"only5\", \"onlyr\"],\n",
            kHeads, kKVHeads, kDim, kMaxQL, kv_copies, pairs);
    fprintf(out, "  \"measurements\": [\n");

    int first_row = 1;
    for (int li = 0; li < n_lens; li++) {
      for (int wi = 0; wi < n_widths; wi++) {
        const int n = lens[li], m = widths[wi];
        if (m <= kMaxQL) {
          fprintf(stderr, "e103_split_ab: skipping M=%d, no split at M<=%d\n",
                  m, kMaxQL);
          continue;
        }
        Cell c = makeCell(device, n, m, kv_copies);
        fprintf(stderr,
                "e103_split_ab: cell N=%d M=%d (split %d+%d) kv=%.1fMB x%d\n",
                n, m, kMaxQL, c.r, (double)c.kv_slice_bytes / 1e6, kv_copies);

        // --- exactness, before any timing ---------------------------------
        runOnce(queue, pso, &c, 0);
        runOnce(queue, pso, &c, 1);
        const uint16_t *of = (const uint16_t *)c.o_full.contents;
        const uint16_t *oh = (const uint16_t *)c.o_head.contents;
        const uint16_t *ot = (const uint16_t *)c.o_tail.contents;
        size_t differing = 0, total = 0;
        double worst = 0.0;
        for (int h = 0; h < kHeads; h++) {
          for (int j = 0; j < m; j++) {
            const uint16_t *want = of + ((size_t)h * m + j) * kDim;
            const uint16_t *got =
                j < kMaxQL ? oh + ((size_t)h * kMaxQL + j) * kDim
                           : ot + ((size_t)h * c.r + (j - kMaxQL)) * kDim;
            for (int d = 0; d < kDim; d++) {
              total++;
              if (want[d] != got[d]) {
                differing++;
                double e = fabs(bf16_to_f32(want[d]) - bf16_to_f32(got[d]));
                if (e > worst) worst = e;
              }
            }
          }
        }
        fprintf(stderr,
                "e103_split_ab:   split vs single  differing=%zu/%zu worst=%.3e"
                "%s\n", differing, total, worst,
                differing ? "   <-- BIT-EXACT VIOLATION" : "");
        fprintf(out,
                "%s    {\"kind\":\"fidelity\",\"n\":%d,\"m\":%d,\"r\":%d,"
                "\"expect_bit_exact\":true,\"differing\":%zu,\"total\":%zu,"
                "\"worst_abs\":%.6e,\"bit_identical\":%s}",
                first_row ? "" : ",\n", n, m, c.r, differing, total, worst,
                differing ? "false" : "true");
        first_row = 0;

        // --- positive control: the comparison must be able to fail ---------
        {
          uint16_t *kp = (uint16_t *)c.keys.contents;
          uint16_t saved = kp[0];
          kp[0] = f32_to_bf16(bf16_to_f32(saved) + 0.25f);
          runOnce(queue, pso, &c, 1);  // only the split legs see the change
          size_t ctrl_diff = 0;
          for (int h = 0; h < kHeads; h++) {
            for (int j = 0; j < m; j++) {
              const uint16_t *want = of + ((size_t)h * m + j) * kDim;
              const uint16_t *got =
                  j < kMaxQL ? oh + ((size_t)h * kMaxQL + j) * kDim
                             : ot + ((size_t)h * c.r + (j - kMaxQL)) * kDim;
              for (int d = 0; d < kDim; d++) {
                if (want[d] != got[d]) ctrl_diff++;
              }
            }
          }
          kp[0] = saved;
          runOnce(queue, pso, &c, 1);
          fprintf(stderr,
                  "e103_split_ab:   positive control: one key perturbed -> "
                  "differing=%zu/%zu\n", ctrl_diff, total);
          fprintf(out,
                  ",\n    {\"kind\":\"positive_control\",\"n\":%d,\"m\":%d,"
                  "\"differing\":%zu,\"total\":%zu,\"detected\":%s}",
                  n, m, ctrl_diff, total, ctrl_diff ? "true" : "false");
        }

        // --- calibrate, warm, then time ------------------------------------
        int slice = 0;
        runVariant(queue, pso, &c, 0, 1, 1, &slice);
        double probe = runVariant(queue, pso, &c, 0, 1, 1, &slice);
        int inner = (int)(target_ms * 1e-3 / probe);
        if (inner < 1) inner = 1;
        if (inner > inner_cap) inner = inner_cap;
        const int reps = 2;
        for (int v = 0; v < kNumVariants; v++) {
          runVariant(queue, pso, &c, v, 1, inner, &slice);
        }

        for (int p = 0; p < pairs; p++) {
          double t[2 * 4];
          double at = seconds_since(g_session_start);
          for (int s = 0; s < 2 * kNumVariants; s++) {
            int v = s < kNumVariants ? s : (2 * kNumVariants - 1 - s);
            t[s] = runVariant(queue, pso, &c, v, reps, inner, &slice);
          }
          double sec[4];
          for (int v = 0; v < kNumVariants; v++) {
            sec[v] = 0.5 * (t[v] + t[2 * kNumVariants - 1 - v]);
          }
          fprintf(stderr, "e103_split_ab:   N=%d M=%d block %d inner=%d", n, m,
                  p, inner);
          for (int v = 0; v < kNumVariants; v++) {
            fprintf(stderr, "  %s=%.2fus", kVariantNames[v], sec[v] * 1e6);
          }
          fprintf(stderr, "  saving=%.2fus\n", (sec[1] - sec[0]) * 1e6);
          fprintf(out,
                  ",\n    {\"kind\":\"timing\",\"n\":%d,\"m\":%d,\"r\":%d,"
                  "\"block\":%d,\"inner\":%d,\"reps\":%d,"
                  "\"session_elapsed_s\":%.3f,\"seconds\":{",
                  n, m, c.r, p, inner, reps, at);
          for (int v = 0; v < kNumVariants; v++) {
            fprintf(out, "%s\"%s\":%.9e", v ? "," : "", kVariantNames[v],
                    sec[v]);
          }
          fprintf(out, "},\"slots\":[");
          for (int s = 0; s < 2 * kNumVariants; s++) {
            fprintf(out, "%s%.9e", s ? "," : "", t[s]);
          }
          fprintf(out, "]}");
        }
      }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    fprintf(stderr, "e103_split_ab: wrote %s\n", out_path);
  }
  return 0;
}
