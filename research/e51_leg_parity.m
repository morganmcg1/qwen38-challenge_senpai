// E51 step 0a: do the SERIAL leg (`qmv_fast_impl`, M == 1) and the MTP leg
// (`qmv_fast_crossrow_affine4_g64_wide`, M in 3..9) produce BIT-IDENTICAL
// affine-4/g64 output on the scored shapes, and does a dose to the pinned BF16
// bias tree break that identity?
//
// Both legs live in ONE compiled entry point, `affine_qmv_fast<T,64,4,false>`:
// the width is carried by `ntg.x`, so M == 1 falls through the switch to
// `qmv_fast_impl` and M in 3..9 reaches the wide cross-row cells. This harness
// therefore needs no model weights and no Swift build. It compiles each arm's
// runtime-effective JIT string with the scored MTLCompileOptions (language 4.0,
// fast math OFF, no include path -- device.cpp:631) and, per (shape, width),
// reports three independent comparisons:
//
//   legs      wide output at width M   vs  M == 1 output of the SAME arm
//   wide_r0   wide output at width M   vs  wide output of the reference arm
//   serial_r0 M == 1 output            vs  M == 1 output of the reference arm
//
// `legs` is the exactness question. `wide_r0` is the dose-bite check: it proves
// the arm changed the numbers the scored wide cell actually produces, which no
// AIR diff can establish. `serial_r0` must stay true for every arm, because the
// patch is confined to the wide cell; if it ever goes false the patch leaked.
//
// Untimed on purpose. Token streams and kernel outputs are deterministic, so no
// thermal gate applies and GPU contention cannot change a bit pattern.
//
// Research-only. Nothing here is on the scored path.
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//         -o /tmp/e51_leg_parity research/e51_leg_parity.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  const char *name;
  int k;
  int n;
} Shape;

// The affine-4/g64 projection shapes of the scored Qwen 3.8 27B tree that reach
// the `out_vec_size >= 4096` wide branch (research/qmv_cost_curve.py).
static const Shape kShapes[] = {
    {"linear_attn.in_proj_fused_qkvzba", 5120, 16480},
    {"linear_attn.out_proj", 6144, 5120},
    {"full_attn.qkv_proj_fused", 5120, 14336},
    {"full_attn.o_proj", 6144, 5120},
    {"mlp.gate_up_fused", 5120, 34816},
    {"mlp.down", 17408, 5120},
    {"head.lm_head", 5120, 248320},
};
static const int kShapeCount = (int)(sizeof(kShapes) / sizeof(kShapes[0]));

typedef struct {
  id<MTLBuffer> w;
  id<MTLBuffer> scales;
  id<MTLBuffer> biases;
  id<MTLBuffer> x;
  id<MTLBuffer> yWide;
  id<MTLBuffer> ySerial;
  id<MTLBuffer> inVec;
  id<MTLBuffer> outVec;
  int k;
  int n;
  int maxM;
} Operands;

static id<MTLComputePipelineState> buildArm(id<MTLDevice> device, NSString *path,
                                            NSString *label) {
  NSError *err = nil;
  NSString *src = [NSString stringWithContentsOfFile:path
                                            encoding:NSUTF8StringEncoding
                                               error:&err];
  if (!src) {
    fprintf(stderr, "e51_leg_parity: cannot read arm %s at %s\n",
            label.UTF8String, path.UTF8String);
    exit(1);
  }
  MTLCompileOptions *opts = [MTLCompileOptions new];
  if (@available(macOS 26.0, *)) {
    opts.languageVersion = MTLLanguageVersion4_0;
  } else {
    opts.languageVersion = MTLLanguageVersion3_1;
  }
  [opts setFastMathEnabled:NO];
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
  if (!lib) {
    fprintf(stderr, "e51_leg_parity: arm %s failed to compile: %s\n",
            label.UTF8String,
            err ? err.localizedDescription.UTF8String : "unknown");
    exit(1);
  }
  id<MTLFunction> fn =
      [lib newFunctionWithName:@"affine_qmv_fast_bfloat16_t_64_4_false"];
  if (!fn) {
    fprintf(stderr, "e51_leg_parity: arm %s has no affine_qmv_fast entry\n",
            label.UTF8String);
    exit(1);
  }
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    fprintf(stderr, "e51_leg_parity: arm %s pipeline failed: %s\n",
            label.UTF8String,
            err ? err.localizedDescription.UTF8String : "unknown");
    exit(1);
  }
  fprintf(stderr, "e51_leg_parity: arm %-10s compiled  src=%zu bytes\n",
          label.UTF8String, (size_t)src.length);
  return pso;
}

static Operands makeOperands(id<MTLDevice> device, Shape shape, int maxM) {
  Operands o;
  o.k = shape.k;
  o.n = shape.n;
  o.maxM = maxM;
  const size_t words = (size_t)shape.n * (size_t)shape.k / 8;
  const size_t groups = (size_t)shape.n * (size_t)shape.k / 64;

  o.w = [device newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
  o.scales = [device newBufferWithLength:groups * 2
                                 options:MTLResourceStorageModeShared];
  o.biases = [device newBufferWithLength:groups * 2
                                 options:MTLResourceStorageModeShared];
  o.x = [device newBufferWithLength:(size_t)maxM * shape.k * 2
                            options:MTLResourceStorageModeShared];
  o.yWide = [device newBufferWithLength:(size_t)maxM * shape.n * 2
                                options:MTLResourceStorageModeShared];
  o.ySerial = [device newBufferWithLength:(size_t)maxM * shape.n * 2
                                  options:MTLResourceStorageModeShared];
  o.inVec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  o.outVec = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  if (!o.w || !o.scales || !o.biases || !o.x || !o.yWide || !o.ySerial) {
    fprintf(stderr, "e51_leg_parity: allocation failed for %s\n", shape.name);
    exit(1);
  }

  uint32_t seed = 0x1234567u;
  uint32_t *wp = (uint32_t *)o.w.contents;
  for (size_t i = 0; i < words; i++) {
    wp[i] = xorshift32(&seed);
  }
  uint16_t *sp = (uint16_t *)o.scales.contents;
  uint16_t *bp = (uint16_t *)o.biases.contents;
  for (size_t i = 0; i < groups; i++) {
    float s = 0.004f + 0.004f * unit(&seed);
    sp[i] = f32_to_bf16(s);
    bp[i] = f32_to_bf16(-7.5f * s);
  }
  uint16_t *xp = (uint16_t *)o.x.contents;
  for (size_t i = 0; i < (size_t)maxM * shape.k; i++) {
    xp[i] = f32_to_bf16(2.0f * unit(&seed) - 1.0f);
  }
  memset(o.yWide.contents, 0, o.yWide.length);
  memset(o.ySerial.contents, 0, o.ySerial.length);
  *(int *)o.inVec.contents = shape.k;
  *(int *)o.outVec.contents = shape.n;
  return o;
}

// One dispatch of the shared entry point. `ntgX` selects the leg: 1 reaches
// `qmv_fast_impl`, 3..9 reach the wide cross-row cells.
static void encode(id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> pso, Operands *o,
                   id<MTLBuffer> y, size_t xOffset, size_t yOffset, int ntgX) {
  [enc setComputePipelineState:pso];
  [enc setBuffer:o->w offset:0 atIndex:0];
  [enc setBuffer:o->scales offset:0 atIndex:1];
  [enc setBuffer:o->biases offset:0 atIndex:2];
  [enc setBuffer:o->x offset:xOffset atIndex:3];
  [enc setBuffer:y offset:yOffset atIndex:4];
  [enc setBuffer:o->inVec offset:0 atIndex:5];
  [enc setBuffer:o->outVec offset:0 atIndex:6];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ntgX,
                                        (NSUInteger)(o->n / 8), 1)
      threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

static void runOnce(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                    Operands *o, int m) {
  memset(o->yWide.contents, 0, o->yWide.length);
  memset(o->ySerial.contents, 0, o->ySerial.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encode(enc, pso, o, o->yWide, 0, 0, m);
  for (int row = 0; row < m; row++) {
    encode(enc, pso, o, o->ySerial, (size_t)row * o->k * 2,
           (size_t)row * o->n * 2, 1);
  }
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) {
    fprintf(stderr, "e51_leg_parity: command buffer failed (status %ld)\n",
            (long)cb.status);
    exit(1);
  }
}

typedef struct {
  size_t compared;
  size_t mismatches;
  double maxAbsDelta;
  double maxRelDelta;
  long maxUlpDelta;
  size_t firstMismatch;
} Compare;

static Compare compareBits(const uint16_t *a, const uint16_t *b, size_t count) {
  Compare c = {count, 0, 0.0, 0.0, 0, SIZE_MAX};
  for (size_t i = 0; i < count; i++) {
    if (a[i] == b[i]) {
      continue;
    }
    c.mismatches++;
    if (c.firstMismatch == SIZE_MAX) {
      c.firstMismatch = i;
    }
    double av = bf16_to_f32(a[i]);
    double bv = bf16_to_f32(b[i]);
    double delta = fabs(av - bv);
    if (delta > c.maxAbsDelta) {
      c.maxAbsDelta = delta;
    }
    double denom = fabs(bv) > 0.0 ? fabs(bv) : 1.0;
    if (delta / denom > c.maxRelDelta) {
      c.maxRelDelta = delta / denom;
    }
    // bf16 bit patterns of the same sign are ordered, so the raw difference is
    // an ulp count. Opposite signs are reported as the sentinel -1.
    long ulps = ((a[i] ^ b[i]) & 0x8000u)
                    ? -1
                    : labs((long)(a[i] & 0x7fffu) - (long)(b[i] & 0x7fffu));
    if (ulps > c.maxUlpDelta) {
      c.maxUlpDelta = ulps;
    }
  }
  return c;
}

static void printCompare(FILE *out, const char *key, Compare c) {
  fprintf(out,
          "\"%s\":{\"equal\":%s,\"compared\":%zu,\"mismatches\":%zu,"
          "\"mismatch_fraction\":%.6g,\"max_abs_delta\":%.9g,"
          "\"max_rel_delta\":%.6g,\"max_ulp_delta\":%ld,"
          "\"first_mismatch\":%lld}",
          key, c.mismatches == 0 ? "true" : "false", c.compared, c.mismatches,
          c.compared ? (double)c.mismatches / (double)c.compared : 0.0,
          c.maxAbsDelta, c.maxRelDelta, c.maxUlpDelta,
          c.firstMismatch == SIZE_MAX ? -1LL : (long long)c.firstMismatch);
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSMutableArray<NSString *> *labels = [NSMutableArray array];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    NSMutableArray<NSNumber *> *widths = [NSMutableArray array];
    NSMutableSet<NSString *> *shapeFilter = [NSMutableSet set];
    const char *jsonPath = NULL;
    int repeats = 1;

    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--arm") && i + 1 < argc) {
        NSString *spec = @(argv[++i]);
        NSRange eq = [spec rangeOfString:@"="];
        if (eq.location == NSNotFound) {
          fprintf(stderr, "e51_leg_parity: --arm wants label=path\n");
          return 2;
        }
        [labels addObject:[spec substringToIndex:eq.location]];
        [paths addObject:[spec substringFromIndex:eq.location + 1]];
      } else if (!strcmp(argv[i], "--widths") && i + 1 < argc) {
        for (NSString *w in [@(argv[++i]) componentsSeparatedByString:@","]) {
          [widths addObject:@(w.intValue)];
        }
      } else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) {
        for (NSString *s in [@(argv[++i]) componentsSeparatedByString:@","]) {
          [shapeFilter addObject:s];
        }
      } else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) {
        repeats = atoi(argv[++i]);
      } else if (!strcmp(argv[i], "--json") && i + 1 < argc) {
        jsonPath = argv[++i];
      } else {
        fprintf(stderr, "e51_leg_parity: unknown argument %s\n", argv[i]);
        return 2;
      }
    }
    if (labels.count == 0) {
      fprintf(stderr, "e51_leg_parity: at least one --arm label=path required\n");
      return 2;
    }
    if (widths.count == 0) {
      for (int m = 3; m <= 9; m++) {
        [widths addObject:@(m)];
      }
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    fprintf(stderr, "e51_leg_parity: device %s\n", device.name.UTF8String);

    NSMutableArray<id<MTLComputePipelineState>> *psos = [NSMutableArray array];
    for (NSUInteger a = 0; a < labels.count; a++) {
      [psos addObject:buildArm(device, paths[a], labels[a])];
    }

    FILE *out = jsonPath ? fopen(jsonPath, "w") : stdout;
    fprintf(out, "{\"device\":\"%s\",\"repeats\":%d,\"entries\":[",
            device.name.UTF8String, repeats);
    bool firstEntry = true;

    int maxM = 1;
    for (NSNumber *w in widths) {
      maxM = w.intValue > maxM ? w.intValue : maxM;
    }

    for (int s = 0; s < kShapeCount; s++) {
      Shape shape = kShapes[s];
      if (shapeFilter.count && ![shapeFilter containsObject:@(shape.name)]) {
        continue;
      }
      Operands o = makeOperands(device, shape, maxM);
      // Reference-arm outputs, kept per width so every later arm compares
      // against the same bytes.
      NSMutableDictionary<NSNumber *, NSData *> *refWide =
          [NSMutableDictionary dictionary];
      NSMutableDictionary<NSNumber *, NSData *> *refSerial =
          [NSMutableDictionary dictionary];

      for (NSUInteger a = 0; a < labels.count; a++) {
        for (NSNumber *wn in widths) {
          int m = wn.intValue;
          NSData *wideFirst = nil;
          NSData *serialFirst = nil;
          bool selfStable = true;
          for (int rep = 0; rep < repeats; rep++) {
            runOnce(queue, psos[a], &o, m);
            NSData *wide = [NSData dataWithBytes:o.yWide.contents
                                          length:(size_t)m * shape.n * 2];
            NSData *serial = [NSData dataWithBytes:o.ySerial.contents
                                            length:(size_t)m * shape.n * 2];
            if (!wideFirst) {
              wideFirst = wide;
              serialFirst = serial;
            } else if (![wide isEqualToData:wideFirst] ||
                       ![serial isEqualToData:serialFirst]) {
              selfStable = false;
            }
          }
          const uint16_t *wide = (const uint16_t *)wideFirst.bytes;
          const uint16_t *serial = (const uint16_t *)serialFirst.bytes;
          size_t count = (size_t)m * shape.n;

          if (a == 0) {
            refWide[wn] = wideFirst;
            refSerial[wn] = serialFirst;
          }
          Compare legs = compareBits(wide, serial, count);
          Compare wideRef = compareBits(wide, (const uint16_t *)refWide[wn].bytes,
                                        count);
          Compare serialRef =
              compareBits(serial, (const uint16_t *)refSerial[wn].bytes, count);

          fprintf(out, "%s\n{\"arm\":\"%s\",\"shape\":\"%s\",\"k\":%d,\"n\":%d,"
                       "\"m\":%d,\"self_stable\":%s,",
                  firstEntry ? "" : ",", labels[a].UTF8String, shape.name,
                  shape.k, shape.n, m, selfStable ? "true" : "false");
          firstEntry = false;
          printCompare(out, "legs", legs);
          fprintf(out, ",");
          printCompare(out, "wide_vs_ref", wideRef);
          fprintf(out, ",");
          printCompare(out, "serial_vs_ref", serialRef);
          fprintf(out, "}");
          fflush(out);

          fprintf(stderr,
                  "  %-10s %-33s M=%d  legs %s (%zu/%zu, max|d|=%.3g, ulp<=%ld)"
                  "  wide_vs_ref %s  serial_vs_ref %s  self_stable %s\n",
                  labels[a].UTF8String, shape.name, m,
                  legs.mismatches == 0 ? "EQUAL   " : "DIVERGES",
                  legs.mismatches, legs.compared, legs.maxAbsDelta,
                  legs.maxUlpDelta,
                  wideRef.mismatches == 0 ? "equal" : "differs",
                  serialRef.mismatches == 0 ? "equal" : "differs",
                  selfStable ? "yes" : "NO");
        }
      }
    }
    fprintf(out, "\n]}\n");
    if (jsonPath) {
      fclose(out);
    }
  }
  return 0;
}
