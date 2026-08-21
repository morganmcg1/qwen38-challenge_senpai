/* E89 rung 0c: model-free reproduction of the per-drafting-round host state.
 *
 * The MTP host thread runs a short CPU burst (about 600 us) once per round and
 * then waits about 150 ms for the GPU. This program mimics only that duty cycle,
 * with no model, no GPU and no MLX, and measures what the CPU does to a burst.
 *
 * Two instruments run inside every burst:
 *   clock  a register-only dependent chain, so its rate is the core clock;
 *   mixed  a pointer chase plus branchy work, so its rate also shows IPC.
 *
 * Build: clang -O2 -o /tmp/e89_duty research/e89_duty_cycle.c
 */
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <sys/resource.h>

/* Darwin role is not exported by the public SDK headers. */
#ifndef PRIO_DARWIN_ROLE
#define PRIO_DARWIN_ROLE 6
#define PRIO_DARWIN_ROLE_UI_FOCAL 2
#endif
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile uint64_t sink;

#define CHASE_BYTES (4u << 20) /* larger than L2, so misses reach the SLC and DRAM */
#define CHASE_SLOTS (CHASE_BYTES / sizeof(uint32_t))
static uint32_t *chase;

static uint64_t chain(uint64_t x, long iters) {
  for (long i = 0; i < iters; i++) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 31;
    x += 0x9e3779b97f4a7c15ULL;
    x ^= x << 17;
    x += 0x165667b19e3779f9ULL;
    x ^= x >> 23;
  }
  return x;
}

static uint64_t mixed(uint64_t seed, long iters) {
  uint32_t p = (uint32_t)(seed % CHASE_SLOTS);
  uint64_t acc = seed;
  for (long i = 0; i < iters; i++) {
    p = chase[p];
    acc += p;
    if (acc & 1) acc = acc * 3 + 1;
    else acc >>= 1;
  }
  return acc + p;
}

struct sample {
  double ns, cyc, ins;
};

static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmpd);
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#define ROUNDS 60
#define CLOCK_ITERS 20000
#define MIXED_ITERS 40000

static void run(const char *label, long idle_us, long keepwarm_iters) {
  struct sample cs[ROUNDS], ms[ROUNDS];
  for (int r = 0; r < ROUNDS; r++) {
    if (idle_us) usleep((useconds_t)idle_us);
    if (keepwarm_iters) sink = chain(sink + 1, keepwarm_iters);

    struct rusage_info_v4 a, b, c;
    uint64_t t0, t1, t2;
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&a);
    t0 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    sink = chain(sink + 3, CLOCK_ITERS);
    t1 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&b);
    sink = mixed(sink + 5, MIXED_ITERS);
    t2 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&c);

    cs[r].ns = (double)(t1 - t0);
    cs[r].cyc = (double)(b.ri_cycles - a.ri_cycles);
    cs[r].ins = (double)(b.ri_instructions - a.ri_instructions);
    ms[r].ns = (double)(t2 - t1);
    ms[r].cyc = (double)(c.ri_cycles - b.ri_cycles);
    ms[r].ins = (double)(c.ri_instructions - b.ri_instructions);
  }

  double cn[ROUNDS], cg[ROUNDS], mn[ROUNDS], mi[ROUNDS], mg[ROUNDS];
  for (int r = 0; r < ROUNDS; r++) {
    cn[r] = cs[r].ns;
    cg[r] = cs[r].cyc / cs[r].ns;
    mn[r] = ms[r].ns;
    mg[r] = ms[r].cyc / ms[r].ns;
    mi[r] = ms[r].ins / ms[r].cyc;
  }
  printf("%-30s idle=%6ldus warm=%7ld | clock: %7.0fns %5.3fGHz | mixed: %8.0fns %5.3fGHz ipc=%5.3f\n",
         label, idle_us, keepwarm_iters, median(cn, ROUNDS), median(cg, ROUNDS),
         median(mn, ROUNDS), median(mg, ROUNDS), median(mi, ROUNDS));
}

/* Core-cluster calibration: which logical CPU indices belong to which cluster,
 * and what clock each one reaches. Runs the register-only chain continuously so
 * the duty-cycle effect above cannot contaminate the reading. */
#define MAP_SLICES 40
#define MAP_ITERS 200000

static void coremap(const char *label, qos_class_t qos) {
  if (qos != QOS_CLASS_UNSPECIFIED) pthread_set_qos_class_self_np(qos, 0);
  double ghz[64];
  long hits[64];
  memset(ghz, 0, sizeof ghz);
  memset(hits, 0, sizeof hits);
  for (int s = 0; s < MAP_SLICES; s++) {
    struct rusage_info_v4 a, b;
    size_t cpu = 0;
    pthread_cpu_number_np(&cpu);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&a);
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    sink = chain(sink + 7, MAP_ITERS);
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&b);
    size_t cpu2 = 0;
    pthread_cpu_number_np(&cpu2);
    if (cpu != cpu2 || cpu >= 64) continue; /* migrated mid-slice, unattributable */
    ghz[cpu] += (double)(b.ri_cycles - a.ri_cycles) / (double)(t1 - t0);
    hits[cpu] += 1;
  }
  printf("%-16s", label);
  for (int c = 0; c < 64; c++)
    if (hits[c]) printf("  cpu%d n=%ld %5.3fGHz", c, hits[c], ghz[c] / (double)hits[c]);
  printf("\n");
}

/* Placement under the round-like duty cycle. `coremap` above runs the chain
 * continuously, which is the easy case. This asks the question that matters:
 * for a thread that is idle 99.6 percent of the time, which scheduling policy
 * actually keeps it on the performance cluster? */
#define PLACE_ROUNDS 60

/* Hold the thread's duty cycle up during the GPU wait and see where the
 * scheduler puts it. `pct` is the target percentage of each 5 ms slice of the
 * wait that the thread spends running instead of sleeping.
 *
 * The spin is bounded by the wall clock, not by an iteration count, because an
 * iteration count silently under-delivers once the thread is demoted to a
 * slower core: that is exactly the state under test, so a fixed count makes the
 * high-duty arms unreachable. The duty cycle actually achieved is measured and
 * reported next to the target. */
#define SLICE_NS 5000000ULL

static void keepalive(const char *label, int pct) {
  long ecore = 0, pcore = 0, unknown = 0;
  double ghz = 0;
  long n = 0;
  uint64_t busy_ns = 0, wall_ns = 0;
  for (int r = 0; r < PLACE_ROUNDS; r++) {
    for (int slice = 0; slice < 30; slice++) { /* 30 x 5 ms = 150 ms */
      uint64_t s0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
      uint64_t spin_ns = SLICE_NS * (uint64_t)pct / 100;
      uint64_t s1 = s0;
      while (spin_ns && (s1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) - s0 < spin_ns)
        sink = chain(sink + 1, 2000);
      busy_ns += s1 - s0;
      if (pct < 100) usleep((useconds_t)((SLICE_NS - spin_ns) / 1000));
      wall_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - s0;
    }
    struct rusage_info_v4 a, b;
    size_t cpu = 0, cpu2 = 0;
    pthread_cpu_number_np(&cpu);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&a);
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    sink = chain(sink + 11, 300000);
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&b);
    pthread_cpu_number_np(&cpu2);
    if (cpu != cpu2 || cpu >= 64) { unknown++; continue; }
    if (cpu < 4) ecore++;
    else pcore++;
    ghz += (double)(b.ri_cycles - a.ri_cycles) / (double)(t1 - t0);
    n++;
  }
  printf("%-14s target=%3d%% actual=%5.1f%%  ecore=%2ld pcore=%2ld migrated=%2ld"
         "  burst %5.3fGHz\n",
         label, pct, wall_ns ? 100.0 * (double)busy_ns / (double)wall_ns : 0.0,
         ecore, pcore, unknown, n ? ghz / (double)n : 0.0);
}

static void placement(const char *label, int policy) {
  switch (policy) {
    case 1: pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); break;
    case 2:
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      setpriority(PRIO_DARWIN_ROLE, 0, PRIO_DARWIN_ROLE_UI_FOCAL);
      break;
    case 3: {
      /* Real-time. Apple documents that time-constraint threads run on the
       * performance cluster. Budget one round: 1 ms of work per 150 ms. */
      mach_timebase_info_data_t tb;
      mach_timebase_info(&tb);
      double ns_per_tick = (double)tb.numer / (double)tb.denom;
      struct thread_time_constraint_policy p;
      p.period = (uint32_t)(150e6 / ns_per_tick);
      p.computation = (uint32_t)(1e6 / ns_per_tick);
      p.constraint = (uint32_t)(5e6 / ns_per_tick);
      p.preemptible = 1;
      kern_return_t kr = thread_policy_set(
          mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY,
          (thread_policy_t)&p, THREAD_TIME_CONSTRAINT_POLICY_COUNT);
      if (kr != KERN_SUCCESS) printf("%-22s thread_policy_set rc=%d\n", label, kr);
      break;
    }
    case 4: {
      /* Affinity tags. Documented as a cache-sharing hint, and widely reported
       * to be unimplemented on Apple silicon, but it is the only placement API
       * left untested and the check is cheap. */
      struct thread_affinity_policy p = {.affinity_tag = 1};
      kern_return_t kr = thread_policy_set(
          mach_thread_self(), THREAD_AFFINITY_POLICY, (thread_policy_t)&p,
          THREAD_AFFINITY_POLICY_COUNT);
      printf("%-22s thread_policy_set rc=%d%s\n", label, kr,
             kr == KERN_NOT_SUPPORTED ? " (KERN_NOT_SUPPORTED)" : "");
      break;
    }
    default: break;
  }
  /* Policies 5 and above spin for `prewarm_us` immediately before the burst,
   * to ask whether a short recent-utilisation spike is enough to earn a
   * performance core, which would be far cheaper than a continuous spin. */
  long prewarm_us = policy >= 5 ? (policy == 5 ? 2000 : policy == 6 ? 10000 : 40000) : 0;
  long ecore = 0, pcore = 0, unknown = 0;
  double ghz[64];
  long hits[64];
  memset(ghz, 0, sizeof ghz);
  memset(hits, 0, sizeof hits);
  for (int r = 0; r < PLACE_ROUNDS; r++) {
    usleep((useconds_t)(150000 - prewarm_us));
    if (prewarm_us) {
      uint64_t w0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
      while (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - w0 < (uint64_t)prewarm_us * 1000)
        sink = chain(sink + 1, 2000);
    }
    struct rusage_info_v4 a, b;
    size_t cpu = 0;
    pthread_cpu_number_np(&cpu);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&a);
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    sink = chain(sink + 11, 300000); /* about 600 us at full clock */
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&b);
    size_t cpu2 = 0;
    pthread_cpu_number_np(&cpu2);
    if (cpu != cpu2 || cpu >= 64) { unknown++; continue; }
    if (cpu < 4) ecore++;
    else pcore++;
    ghz[cpu] += (double)(b.ri_cycles - a.ri_cycles) / (double)(t1 - t0);
    hits[cpu] += 1;
  }
  double tot = 0;
  long n = 0;
  for (int c = 0; c < 64; c++) {
    tot += ghz[c];
    n += hits[c];
  }
  printf("%-22s ecore=%2ld pcore=%2ld migrated=%2ld  mean %5.3fGHz  |", label, ecore,
         pcore, unknown, n ? tot / (double)n : 0.0);
  for (int c = 0; c < 64; c++)
    if (hits[c]) printf(" cpu%d:%ld", c, hits[c]);
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "placement") == 0) {
    placement("warmup, discard", 0);
    placement("A default", 0);
    placement("B userinteractive", 1);
    placement("C uix + UI_FOCAL role", 2);
    placement("D time-constraint RT", 3);
    placement("E affinity tag", 4);
    placement("F prewarm 2ms", 5);
    placement("G prewarm 10ms", 6);
    placement("H prewarm 40ms", 7);
    return 0;
  }
  if (argc > 1 && strcmp(argv[1], "keepalive") == 0) {
    keepalive("warmup", 0);
    static const int duty[] = {0, 10, 25, 40, 55, 70, 85, 95, 100};
    for (unsigned i = 0; i < sizeof duty / sizeof *duty; i++)
      keepalive("sweep", duty[i]);
    return 0;
  }
  if (argc > 1 && strcmp(argv[1], "coremap") == 0) {
    printf("perflevel0(P)=%s perflevel1(E)=%s\n", getenv("E89_P") ? getenv("E89_P") : "?",
           getenv("E89_E") ? getenv("E89_E") : "?");
    coremap("unspecified", QOS_CLASS_UNSPECIFIED);
    coremap("background", QOS_CLASS_BACKGROUND);
    coremap("utility", QOS_CLASS_UTILITY);
    coremap("default", QOS_CLASS_DEFAULT);
    coremap("userinitiated", QOS_CLASS_USER_INITIATED);
    coremap("userinteractive", QOS_CLASS_USER_INTERACTIVE);
    coremap("background2", QOS_CLASS_BACKGROUND);
    return 0;
  }
  chase = malloc(CHASE_BYTES);
  /* one random permutation cycle, so the chase cannot be prefetched */
  for (uint32_t i = 0; i < CHASE_SLOTS; i++) chase[i] = i;
  for (uint32_t i = CHASE_SLOTS - 1; i > 0; i--) {
    uint32_t j = (uint32_t)(((uint64_t)rand() << 15 ^ (uint64_t)rand()) % (i + 1));
    uint32_t t = chase[i];
    chase[i] = chase[j];
    chase[j] = t;
  }

  run("warmup, discard", 0, 0);
  run("A busy, no idle", 0, 0);
  run("B round-like, 150ms idle", 150000, 0);
  run("B round-like + 1.3ms warm", 150000, 200000);
  run("B round-like + 6.5ms warm", 150000, 1000000);
  run("A busy, no idle", 0, 0);
  run("B round-like, 150ms idle", 150000, 0);
  free(chase);
  return 0;
}
