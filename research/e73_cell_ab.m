// E73 rung 1: time every legal `(M, IPG)` group partition inside ONE process
// and ONE thermal session, on the frozen host geometry.
//
// The arms are entry points in one self-contained source string
// (research/e73_emit_arms.py), compiled here with the same MTLCompileOptions
// the scored worker uses, so no arm can differ by preamble text, compiler
// flags, or build freshness. Legs run in palindrome order, so a monotone
// thermal or clock drift cancels to first order and every arm sits at the same
// mean leg position.
//
// Each arm dispatches `grid_dims(M, ceil(N/8), 1)` with `group_dims(32, 2, 1)`,
// which is exactly what `backend/metal/quantized.cpp` gives the kernel at that
// M. The M x-slots are all launched and `ceil(M / IPG)` of them read weights;
// the rest return. That idle-slot cost is part of the partition and is
// therefore inside the measurement, not removed from it.
//
// x and y are allocated for 9 input rows and every arm uses its first M rows.
// One row's dot product does not depend on M or IPG, so one reference capture
// checks every arm: an arm that skips work shows differing elements instead of
// a faster leg.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e73/e73_cell_ab research/e73_cell_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define kMaxArms 24
#define kMaxRows 9

typedef struct {
  char name[16];
  int m;
  int ipg;
  int groups;
} Arm;

static Arm kArm[kMaxArms];
static int kArmCount = 0;
static int kOrder[2 * kMaxArms];
static int kLegs = 0;

static void buildOrder(void) {
  kLegs = 2 * kArmCount;
  for (int a = 0; a < kArmCount; a++) {
    kOrder[a] = a;
    kOrder[kLegs - 1 - a] = a;
  }
}

typedef struct {
  const char *name;
  int k;
  int n;
} Shape;

// Every distinct (K, N) affine-4 group-64 shape on the scored path.
// `full_attn.o_proj` is the same (6144, 5120) cell as `linear_attn.out_proj`,
// so one entry covers both of the seven scored linear shapes.
static const Shape kShapes[] = {
    {"linear_attn.in_proj_fused_qkvzba", 5120, 16480},
    {"full_attn.qkv_proj_fused", 5120, 14336},
    {"mlp.gate_up_fused", 5120, 34816},
    {"head.lm_head", 5120, 248320},
    {"linear_attn.out_proj", 6144, 5120},
    {"mlp.down", 17408, 5120},
};
static const int kShapeCount = 6;

static double g_ns_per_tick = 0.0;

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
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

static double sample_gpu_temp(const char *macmon) {
  if (!macmon || !macmon[0]) return NAN;
  char command[1024];
  snprintf(command, sizeof(command), "%s pipe -s1 2>/dev/null", macmon);
  FILE *pipe = popen(command, "r");
  if (!pipe) return NAN;
  static char buffer[65536];
  size_t got = fread(buffer, 1, sizeof(buffer) - 1, pipe);
  pclose(pipe);
  buffer[got] = 0;
  const char *key = strstr(buffer, "\"gpu_temp_avg\":");
  if (!key) return NAN;
  return atof(key + strlen("\"gpu_temp_avg\":"));
}

typedef struct {
  id<MTLBuffer> w, scales, biases, x, y, in_vec, out_vec;
  int n, k;
} Operands;

static Operands makeOperands(id<MTLDevice> device, Shape shape) {
  Operands o;
  o.n = shape.n;
  o.k = shape.k;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;

  o.w = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
  o.scales = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.biases = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)kMaxRows * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.y = [device newBufferWithLength:(size_t)kMaxRows * shape.n * 2
                            options:MTLResourceStorageModeShared];
  o.in_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.out_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];

  uint32_t seed = 0x1234567u;
  uint32_t *wp = (uint32_t *)o.w.contents;
  for (size_t i = 0; i < words; i++) wp[i] = xorshift32(&seed);
  uint16_t *sp = (uint16_t *)o.scales.contents;
  uint16_t *bp = (uint16_t *)o.biases.contents;
  for (size_t i = 0; i < groups; i++) {
    float s = 0.004f + 0.004f * unit(&seed);
    sp[i] = f32_to_bf16(s);
    bp[i] = f32_to_bf16(-7.5f * s);
  }
  uint16_t *xp = (uint16_t *)o.x.contents;
  for (size_t i = 0; i < (size_t)kMaxRows * shape.k; i++) {
    xp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  memset(o.y.contents, 0, o.y.length);
  *(int *)o.in_vec.contents = shape.k;
  *(int *)o.out_vec.contents = shape.n;
  return o;
}

static void encodeDispatch(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso, Operands *o,
                           int m) {
  [enc setComputePipelineState:pso];
  [enc setBuffer:o->w offset:0 atIndex:0];
  [enc setBuffer:o->scales offset:0 atIndex:1];
  [enc setBuffer:o->biases offset:0 atIndex:2];
  [enc setBuffer:o->x offset:0 atIndex:3];
  [enc setBuffer:o->y offset:0 atIndex:4];
  [enc setBuffer:o->in_vec offset:0 atIndex:5];
  [enc setBuffer:o->out_vec offset:0 atIndex:6];
  // grid_dims(M, N/8, 1) with group_dims(32, 2, 1): the exact geometry
  // quantized.cpp gives this kernel, held constant for every arm because
  // backend/metal/quantized.cpp is not an editable path.
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m, (NSUInteger)(o->n / 8), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

typedef struct {
  double gpu_seconds;
  double wall_seconds;
} Leg;

static Leg runLeg(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                  Operands *o, int inner, int m) {
  uint64_t t0 = mach_absolute_time();
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  for (int i = 0; i < inner; i++) encodeDispatch(enc, pso, o, m);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  Leg leg;
  leg.wall_seconds = (double)(mach_absolute_time() - t0) * g_ns_per_tick * 1e-9;
  leg.gpu_seconds = cb.GPUEndTime - cb.GPUStartTime;
  return leg;
}

static void captureOutput(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pso, Operands *o,
                          uint16_t *out, int m) {
  memset(o->y.contents, 0, o->y.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeDispatch(enc, pso, o, m);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  memcpy(out, o->y.contents, o->y.length);
}

static int parseArms(char *list) {
  for (char *token = strtok(list, ","); token; token = strtok(NULL, ",")) {
    if (kArmCount == kMaxArms) {
      fprintf(stderr, "e73_cell_ab: at most %d arms\n", kMaxArms);
      return 0;
    }
    int m = 0, ipg = 0;
    if (sscanf(token, "m%d_ipg%d", &m, &ipg) != 2 || m < 3 || m > kMaxRows ||
        ipg < 2 || ipg > 6) {
      fprintf(stderr, "e73_cell_ab: bad arm %s\n", token);
      return 0;
    }
    Arm arm;
    snprintf(arm.name, sizeof(arm.name), "%s", token);
    arm.m = m;
    arm.ipg = ipg;
    arm.groups = (m + ipg - 1) / ipg;
    kArm[kArmCount++] = arm;
  }
  return kArmCount;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *source_path = NULL;
    const char *out_path = NULL;
    const char *macmon = getenv("MLXFAST_MACMON_BIN");
    const char *shape_filter = NULL;
    char *arm_list = NULL;
    int reps = 9, warmup_reps = 1;
    double target_bytes = 12e9;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--source") && i + 1 < argc) source_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape_filter = argv[++i];
      else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arm_list = strdup(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--warmup-reps") && i + 1 < argc) warmup_reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--target-bytes") && i + 1 < argc) target_bytes = atof(argv[++i]);
      else if (!strcmp(argv[i], "--macmon") && i + 1 < argc) macmon = argv[++i];
      else {
        fprintf(stderr, "e73_cell_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!source_path || !out_path || !arm_list) {
      fprintf(stderr, "e73_cell_ab: --source, --arms and --out are required\n");
      return 2;
    }
    if (!parseArms(arm_list)) return 2;
    buildOrder();

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = (double)tb.numer / (double)tb.denom;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@(source_path)
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    if (!src) {
      fprintf(stderr, "e73_cell_ab: cannot read %s\n", source_path);
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
      fprintf(stderr, "e73_cell_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }

    id<MTLComputePipelineState> pso[kMaxArms];
    for (int a = 0; a < kArmCount; a++) {
      NSString *fn_name = [NSString stringWithFormat:@"e73_cell_%s", kArm[a].name];
      id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
      if (!fn) {
        fprintf(stderr, "e73_cell_ab: missing function %s\n", [fn_name UTF8String]);
        return 1;
      }
      pso[a] = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso[a]) {
        fprintf(stderr, "e73_cell_ab: pipeline %s failed: %s\n",
                [fn_name UTF8String],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      printf("ARM arm=%s m=%d ipg=%d groups=%d "
             "max_total_threads_per_threadgroup=%lu "
             "static_threadgroup_memory_bytes=%lu thread_execution_width=%lu\n",
             kArm[a].name, kArm[a].m, kArm[a].ipg, kArm[a].groups,
             (unsigned long)pso[a].maxTotalThreadsPerThreadgroup,
             (unsigned long)pso[a].staticThreadgroupMemoryLength,
             (unsigned long)pso[a].threadExecutionWidth);
      fflush(stdout);
    }

    NSMutableArray *shapeResults = [NSMutableArray array];
    double entry_c = sample_gpu_temp(macmon);
    fprintf(stderr, "e73_cell_ab: session entry gpu_temp_c=%.2f\n", entry_c);

    for (int s = 0; s < kShapeCount; s++) {
      @autoreleasepool {
        Shape shape = kShapes[s];
        if (shape_filter && strcmp(shape_filter, shape.name)) continue;
        Operands o = makeOperands(device, shape);
        // Bytes ONE weight stream reads: 4-bit packed data plus the g64 scale
        // and bias. An arm with `groups` streams moves `groups` times this.
        const double bytes_per_stream =
            (double)shape.n * (double)shape.k / 2.0
            + 2.0 * 2.0 * (double)shape.n * (double)shape.k / 64.0;
        int inner = (int)(target_bytes / bytes_per_stream);
        if (inner < 1) inner = 1;

        // Work conservation and bit-identity in one check. Row r of y does not
        // depend on M or IPG, so every arm must reproduce the reference rows it
        // covers, bit for bit.
        size_t elements = (size_t)kMaxRows * shape.n;
        uint16_t *reference = malloc(elements * 2);
        uint16_t *candidate = malloc(elements * 2);
        int ref = 0;
        for (int a = 1; a < kArmCount; a++) {
          if (kArm[a].m > kArm[ref].m) ref = a;
        }
        captureOutput(queue, pso[ref], &o, reference, kArm[ref].m);
        NSMutableDictionary *parity = [NSMutableDictionary dictionary];
        for (int a = 0; a < kArmCount; a++) {
          captureOutput(queue, pso[a], &o, candidate, kArm[a].m);
          size_t differing = 0, zero_rows = 0;
          for (int row = 0; row < kArm[a].m && row < kArm[ref].m; row++) {
            for (int i = 0; i < shape.n; i++) {
              size_t at = (size_t)row * shape.n + (size_t)i;
              if (reference[at] != candidate[at]) differing++;
            }
          }
          for (int row = 0; row < kArm[a].m; row++) {
            size_t nonzero = 0;
            for (int i = 0; i < shape.n && nonzero == 0; i++) {
              if (candidate[(size_t)row * shape.n + (size_t)i] != 0) nonzero = 1;
            }
            if (!nonzero) zero_rows++;
          }
          parity[@(kArm[a].name)] = @{@"differing": @(differing),
                                      @"zero_rows": @(zero_rows)};
          printf("PARITY shape=%s arm=%s m=%d ipg=%d differing=%zu "
                 "zero_rows=%zu compared_rows=%d\n",
                 shape.name, kArm[a].name, kArm[a].m, kArm[a].ipg, differing,
                 zero_rows, kArm[a].m < kArm[ref].m ? kArm[a].m : kArm[ref].m);
          fflush(stdout);
        }
        free(reference);
        free(candidate);

        double shape_entry_c = sample_gpu_temp(macmon);
        NSMutableArray *legs = [NSMutableArray array];
        for (int rep = -warmup_reps; rep < reps; rep++) {
          for (int position = 0; position < kLegs; position++) {
            int a = kOrder[position];
            Leg leg = runLeg(queue, pso[a], &o, inner, kArm[a].m);
            if (rep < 0) continue;  // declared discarded warm-up rep
            double per_dispatch = leg.gpu_seconds / (double)inner;
            double stream_bytes = bytes_per_stream * (double)kArm[a].groups;
            double gbps = stream_bytes * (double)inner / leg.gpu_seconds / 1e9;
            [legs addObject:@{
              @"rep": @(rep),
              @"position": @(position),
              @"arm": @(kArm[a].name),
              @"m": @(kArm[a].m),
              @"ipg": @(kArm[a].ipg),
              @"groups": @(kArm[a].groups),
              @"gpu_seconds": @(leg.gpu_seconds),
              @"wall_seconds": @(leg.wall_seconds),
              @"seconds_per_dispatch": @(per_dispatch),
              @"gbps": @(gbps),
            }];
            printf("LEG shape=%s rep=%d position=%d arm=%s m=%d ipg=%d "
                   "groups=%d inner=%d gpu_seconds=%.9f wall_seconds=%.9f "
                   "seconds_per_dispatch=%.12f gbps=%.6f\n",
                   shape.name, rep, position, kArm[a].name, kArm[a].m,
                   kArm[a].ipg, kArm[a].groups, inner, leg.gpu_seconds,
                   leg.wall_seconds, per_dispatch, gbps);
            fflush(stdout);
          }
        }
        double shape_exit_c = sample_gpu_temp(macmon);
        printf("SHAPE shape=%s k=%d n=%d inner=%d legs=%lu "
               "bytes_per_stream=%.0f entry_gpu_temp_c=%.2f exit_gpu_temp_c=%.2f\n",
               shape.name, shape.k, shape.n, inner, (unsigned long)legs.count,
               bytes_per_stream, shape_entry_c, shape_exit_c);
        fflush(stdout);
        fprintf(stderr, "e73_cell_ab: %s inner=%d legs=%lu entry=%.2fC exit=%.2fC\n",
                shape.name, inner, (unsigned long)legs.count, shape_entry_c,
                shape_exit_c);

        [shapeResults addObject:@{
          @"shape": @(shape.name),
          @"k": @(shape.k),
          @"n": @(shape.n),
          @"inner": @(inner),
          @"bytes_per_stream": @(bytes_per_stream),
          @"entry_gpu_temp_c": @(shape_entry_c),
          @"exit_gpu_temp_c": @(shape_exit_c),
          @"parity": parity,
          @"legs": legs,
        }];
      }
    }

    double exit_c = sample_gpu_temp(macmon);
    NSMutableArray *armRows = [NSMutableArray array];
    for (int a = 0; a < kArmCount; a++) {
      [armRows addObject:@{
        @"arm": @(kArm[a].name),
        @"m": @(kArm[a].m),
        @"ipg": @(kArm[a].ipg),
        @"groups": @(kArm[a].groups),
        @"max_total_threads_per_threadgroup":
            @(pso[a].maxTotalThreadsPerThreadgroup),
      }];
    }
    NSDictionary *report = @{
      @"experiment": @"e73",
      @"rung": @1,
      @"harness": @"local",
      @"device": device.name,
      @"reps": @(reps),
      @"warmup_reps": @(warmup_reps),
      @"target_bytes": @(target_bytes),
      @"session_entry_gpu_temp_c": @(entry_c),
      @"session_exit_gpu_temp_c": @(exit_c),
      @"arms": armRows,
      @"shapes": shapeResults,
    };
    NSError *json_error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingSortedKeys
                                                     error:&json_error];
    if (!json) {
      fprintf(stderr, "e73_cell_ab: JSON failed: %s\n",
              [[json_error localizedDescription] UTF8String]);
      return 1;
    }
    [json writeToFile:@(out_path) atomically:YES];
    fprintf(stderr, "e73_cell_ab: wrote %s exit gpu_temp_c=%.2f\n", out_path,
            exit_c);
    return 0;
  }
}
