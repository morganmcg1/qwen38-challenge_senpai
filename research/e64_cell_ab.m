// E64 rung 0b: time three isolated `qmv_fast_crossrow_affine4_g64_wide` arms at
// one NA inside ONE process and ONE thermal session.
//
// The arms are three entry points in one self-contained source string
// (research/e64_emit_arms.py), compiled here with the same MTLCompileOptions the
// scored worker uses, so no arm can differ by preamble text, compiler flags, or
// build freshness. Legs run in palindrome order, so a monotone thermal or clock
// drift cancels to first order and every arm sits at the same mean leg position.
//
// Research only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e64/e64_cell_ab research/e64_cell_ab.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define kMaxArms 8
// Palindrome over the selected arms: every arm has the same mean leg position,
// so a monotone drift across one rep cancels in the arm contrast, and the two
// legs of one arm are a same-arm null at maximum separation.
static const char *kArms[kMaxArms] = {"plain", "forced", "ballast"};
static int kArmCount = 3;
static int kOrder[2 * kMaxArms];
static int kLegs = 6;

// Output rows one threadgroup covers. `rows2` halves the rows per simdgroup, so
// it needs twice the threadgroups for the same output. Timing it on the default
// grid would leave half the output unwritten, and the parity check would still
// read zero differing on the rows it did write.
static int kArmRowsPerTG[kMaxArms] = {8, 8, 8, 8, 8, 8, 8, 8};

static int rowsPerThreadgroup(const char *arm) {
  return strcmp(arm, "rows2") == 0 ? 4 : 8;
}

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

static const Shape kShapes[] = {
    {"mlp.gate_up_fused", 5120, 34816},
    {"mlp.down", 17408, 5120},
    {"full_attn.qkv_proj_fused", 5120, 14336},
    {"linear_attn.in_proj_fused_qkvzba", 5120, 16480},
};
static const int kShapeCount = 4;

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
  id<MTLBuffer> w, scales, biases, x, y, in_vec, out_vec, na_vec;
  int n, k, rows;
} Operands;

static Operands makeOperands(id<MTLDevice> device, Shape shape, int na) {
  Operands o;
  o.n = shape.n;
  o.k = shape.k;
  o.rows = na;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;

  o.w = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
  o.scales = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.biases = [device newBufferWithLength:groups * 2 options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)na * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.y = [device newBufferWithLength:(size_t)na * shape.n * 2
                            options:MTLResourceStorageModeShared];
  o.in_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.out_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.na_vec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  *(int *)o.na_vec.contents = na;

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
  for (size_t i = 0; i < (size_t)na * shape.k; i++) {
    xp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  memset(o.y.contents, 0, o.y.length);
  *(int *)o.in_vec.contents = shape.k;
  *(int *)o.out_vec.contents = shape.n;
  return o;
}

static void encodeDispatch(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso, Operands *o,
                           int rows_per_tg) {
  [enc setComputePipelineState:pso];
  [enc setBuffer:o->w offset:0 atIndex:0];
  [enc setBuffer:o->scales offset:0 atIndex:1];
  [enc setBuffer:o->biases offset:0 atIndex:2];
  [enc setBuffer:o->x offset:0 atIndex:3];
  [enc setBuffer:o->y offset:0 atIndex:4];
  [enc setBuffer:o->in_vec offset:0 atIndex:5];
  [enc setBuffer:o->out_vec offset:0 atIndex:6];
  // Only `merged` declares buffer 7. Binding it for every arm keeps one
  // dispatch path and cannot change an arm that does not read it.
  [enc setBuffer:o->na_vec offset:0 atIndex:7];
  // One x-group covering NA input rows: the single-weight-stream geometry the
  // NA ladder measures. Output rows per threadgroup vary by arm.
  [enc dispatchThreadgroups:MTLSizeMake(1, (NSUInteger)(o->n / rows_per_tg), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

typedef struct {
  double gpu_seconds;
  double wall_seconds;
} Leg;

static Leg runLeg(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                  Operands *o, int inner, int rows_per_tg) {
  uint64_t t0 = mach_absolute_time();
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  for (int i = 0; i < inner; i++) encodeDispatch(enc, pso, o, rows_per_tg);
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
                          uint16_t *out, int rows_per_tg) {
  memset(o->y.contents, 0, o->y.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encodeDispatch(enc, pso, o, rows_per_tg);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  memcpy(out, o->y.contents, o->y.length);
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const char *source_path = NULL;
    const char *out_path = NULL;
    const char *macmon = getenv("MLXFAST_MACMON_BIN");
    const char *shape_filter = NULL;
    char *arm_list = NULL;
    int na = 5, reps = 21, warmup_reps = 1;
    double target_bytes = 24e9;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--source") && i + 1 < argc) source_path = argv[++i];
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
      else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape_filter = argv[++i];
      else if (!strcmp(argv[i], "--na") && i + 1 < argc) na = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--warmup-reps") && i + 1 < argc) warmup_reps = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--target-bytes") && i + 1 < argc) target_bytes = atof(argv[++i]);
      else if (!strcmp(argv[i], "--macmon") && i + 1 < argc) macmon = argv[++i];
      else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arm_list = strdup(argv[++i]);
      else {
        fprintf(stderr, "e64_cell_ab: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (!source_path) {
      fprintf(stderr, "e64_cell_ab: --source is required\n");
      return 2;
    }
    if (arm_list) {
      kArmCount = 0;
      for (char *token = strtok(arm_list, ","); token;
           token = strtok(NULL, ",")) {
        if (kArmCount == kMaxArms) {
          fprintf(stderr, "e64_cell_ab: at most %d arms\n", kMaxArms);
          return 2;
        }
        kArms[kArmCount++] = token;
      }
    }
    for (int a = 0; a < kArmCount; a++) {
      kArmRowsPerTG[a] = rowsPerThreadgroup(kArms[a]);
    }
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
      fprintf(stderr, "e64_cell_ab: cannot read %s\n", source_path);
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
      fprintf(stderr, "e64_cell_ab: compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      return 1;
    }

    id<MTLComputePipelineState> pso[kMaxArms];
    for (int a = 0; a < kArmCount; a++) {
      NSString *name = [NSString stringWithFormat:@"e64_cell_%s", kArms[a]];
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      if (!fn) {
        fprintf(stderr, "e64_cell_ab: missing function %s\n", [name UTF8String]);
        return 1;
      }
      pso[a] = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso[a]) {
        fprintf(stderr, "e64_cell_ab: pipeline %s failed: %s\n",
                [name UTF8String],
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return 1;
      }
      fprintf(stderr, "e64_cell_ab: arm %-8s max_threads=%lu tg_mem=%lu\n",
              kArms[a], (unsigned long)pso[a].maxTotalThreadsPerThreadgroup,
              (unsigned long)pso[a].staticThreadgroupMemoryLength);
    }

    NSMutableArray *shapeResults = [NSMutableArray array];
    double entry_c = sample_gpu_temp(macmon);
    fprintf(stderr, "e64_cell_ab: session entry gpu_temp_c=%.2f\n", entry_c);

    for (int s = 0; s < kShapeCount; s++) {
      Shape shape = kShapes[s];
      if (shape_filter && strcmp(shape_filter, shape.name)) continue;
      Operands o = makeOperands(device, shape, na);
      const double bytes_per_dispatch = (double)shape.n * (double)shape.k / 2.0;
      int inner = (int)(target_bytes / bytes_per_dispatch);
      if (inner < 1) inner = 1;

      // Parity: identical floating-point operations must give identical bits.
      size_t elements = (size_t)na * shape.n;
      uint16_t *reference = malloc(elements * 2);
      uint16_t *candidate = malloc(elements * 2);
      captureOutput(queue, pso[0], &o, reference, kArmRowsPerTG[0]);
      NSMutableDictionary *parity = [NSMutableDictionary dictionary];
      for (int a = 1; a < kArmCount; a++) {
        captureOutput(queue, pso[a], &o, candidate, kArmRowsPerTG[a]);
        size_t differing = 0;
        for (size_t i = 0; i < elements; i++) {
          if (reference[i] != candidate[i]) differing++;
        }
        parity[@(kArms[a])] = @(differing);
        fprintf(stderr, "e64_cell_ab: %s parity vs plain: %zu differing of %zu\n",
                kArms[a], differing, elements);
      }
      free(reference);
      free(candidate);

      double shape_entry_c = sample_gpu_temp(macmon);
      NSMutableArray *legs = [NSMutableArray array];
      for (int rep = -warmup_reps; rep < reps; rep++) {
        for (int position = 0; position < kLegs; position++) {
          int arm = kOrder[position];
          Leg leg = runLeg(queue, pso[arm], &o, inner, kArmRowsPerTG[arm]);
          if (rep < 0) continue;  // declared discarded warm-up rep
          [legs addObject:@{
            @"rep": @(rep),
            @"position": @(position),
            @"arm": @(kArms[arm]),
            @"gpu_seconds": @(leg.gpu_seconds),
            @"wall_seconds": @(leg.wall_seconds),
            @"seconds_per_dispatch": @(leg.gpu_seconds / (double)inner),
            @"gbps": @(bytes_per_dispatch * (double)inner / leg.gpu_seconds / 1e9),
          }];
        }
      }
      double shape_exit_c = sample_gpu_temp(macmon);
      fprintf(stderr,
              "e64_cell_ab: %s inner=%d legs=%lu entry=%.2fC exit=%.2fC\n",
              shape.name, inner, (unsigned long)legs.count, shape_entry_c,
              shape_exit_c);

      [shapeResults addObject:@{
        @"shape": @(shape.name),
        @"k": @(shape.k),
        @"n": @(shape.n),
        @"inner": @(inner),
        @"bytes_per_dispatch": @(bytes_per_dispatch),
        @"entry_gpu_temp_c": @(shape_entry_c),
        @"exit_gpu_temp_c": @(shape_exit_c),
        @"parity_differing_vs_plain": parity,
        @"legs": legs,
      }];
    }
    double exit_c = sample_gpu_temp(macmon);

    NSMutableArray *order = [NSMutableArray array];
    for (int position = 0; position < kLegs; position++) {
      [order addObject:@(kArms[kOrder[position]])];
    }

    NSDictionary *report = @{
      @"source": @(source_path),
      @"na": @(na),
      @"reps": @(reps),
      @"warmup_reps_discarded": @(warmup_reps),
      @"order": order,
      @"device": device.name ?: @"?",
      @"entry_gpu_temp_c": @(entry_c),
      @"exit_gpu_temp_c": @(exit_c),
      @"shapes": shapeResults,
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingPrettyPrinted
                                                     error:&err];
    if (out_path) {
      [json writeToFile:@(out_path) atomically:YES];
      fprintf(stderr, "e64_cell_ab: wrote %s\n", out_path);
    } else {
      fwrite(json.bytes, 1, json.length, stdout);
    }
    return 0;
  }
}
