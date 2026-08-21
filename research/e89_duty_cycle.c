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

int main(void) {
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
