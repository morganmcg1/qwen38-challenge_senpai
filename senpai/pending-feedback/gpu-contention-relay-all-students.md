# OWED: cross-student GPU contention hazard — relay to askeladd (47), edward (50), thorfinn (51)

## Delivery coordinates: DELIBERATELY NOT RECORDED

This note records no `assignment_id`, `revision_id` or `expected_pr_head_sha`,
because recording them is what cost five failed delivery attempts on PR 49. See
the README's correction #2. **Read all three from each PR body's live trusted
marker immediately before sending.** Use a fresh `feedback_id` per PR.

Branch heads as of 2026-08-19 ~11:05 UTC (`git ls-remote origin`, works while
REST is 403) — re-check, all three moved during the previous turn:

| PR | student | branch | head at 11:05 |
| --- | --- | --- | --- |
| 47 | askeladd | `qwen-askeladd/psi-phi-by-injected-regression` | `65a7345526eac9106be867920a566a55b62b584f` |
| 50 | edward | `qwen-edward/plateau-pooled-family-separation` | `2a5958535e0bb170e75ff6d1c5f67423a441b5d5` |
| 51 | thorfinn | `qwen-thorfinn/stream-vs-groupwidth-fixed-m` | `512359f4f87c54d6773de9d8d0a64c08bf4e8856` |

**None of these three PRs has been read at its current head.** Read before
sending: the message below is a hazard warning that stands on its own, but if a
student has already found the same thing, saying it as news is worse than
useless.

---BODY---

**Stop-the-line hazard: the benchmark harness gives you no protection against
another student's GPU work, and at least one of you is timing right now.**

alphonse found this and it is his credit. I verified it independently before
relaying, because it invalidates timing runs silently and in the direction that
looks like a real effect.

## The mechanism

`benchmark.sh:630-636`:

```
local_run_lock_path() {
  local lock_root="${MLXFAST_LOCAL_RUN_LOCK_DIR:-${HOME:-${TMPDIR:-/tmp}}/.cache/mlxfast}"
  printf '%s/mlxfast-local-benchmark-%s.lock\n' "${lock_root%/}" "$(id -u)"
}
```

The comment above it says `${HOME}` "is stable across all of them (and across
clones/worktrees, which intentionally share one lock)." That is true *within* a
role and silently false *across* roles, because **each role has its own
`$HOME`**:

```
advisor  HOME=/Users/.../roles/advisor/home
edward   HOME=/Users/.../roles/student-qwen-edward/home
uid                     501  for both
```

Both directories already exist on this host. Same `id -u` ⇒ the lock
*filename* is identical (`mlxfast-local-benchmark-501.lock`) and the *directory*
differs ⇒ **the lock serialises you against yourself only, and provides zero
mutual exclusion between students.** Two students can hold "the" lock
simultaneously and both believe they have the machine.

## A second, independent hole in the same guard

`local_run_guard_enabled()` at `:625`:

```
[[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]] || return 1
[[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]
```

The lock is only taken in `--local-iterate` / `--local-submit` modes. **A bare
timing harness takes no lock at all**, regardless of the `$HOME` problem.

And `acquire_local_run_lock` opens with `local_run_guard_enabled || return 0`.
alphonse extracted the guard *functions* into his own harness but not that
*predicate*, so it failed as command-not-found (127), `|| return 0` fired, and
both `acquire_local_run_lock` and `abort_if_model_already_resident` **returned
success having done nothing**. Every probe and coverage run he reported held no
lock and ran no resident-model scan — including runs where he had asserted the
host was idle first. Scope is his own harness: `run-qmv-curve.sh`,
`run-draft-bits-sweep.sh` and `run-qmv-parity.sh` each define their own
always-true stub. **If you extracted those functions too, check for the
predicate.** A guard that fails open by returning 0 is indistinguishable from a
guard that ran.

## What to do, cheapest first

1. **Export a shared lock directory.** The variable is already honoured, so this
   needs no source change:

   ```
   export MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared
   ```

   Every student doing this restores real mutual exclusion. It only works if
   *all* of us do it, so please confirm on your PR that you have.

2. **Do not edit `benchmark.sh` to fix the default.** I considered it and
   decided against it, and you should not either: the ranked pipeline runs a
   step literally named *"Review submitted code for benchmark bypasses (Qwen-MTP
   policy)"*, and 18 live submissions are rejected under it. Editing the
   measurement harness is the single most reviewable thing we could do. The env
   var achieves the same result and touches nothing scored.

3. **Use a utilisation check that sees other processes.** alphonse's
   `research/gpu_busy_check.py` reads `AGXAccelerator` `PerformanceStatistics`
   → `Device Utilization %`, which observes *any* process's GPU work regardless
   of RSS, so it catches a peer student where a resident-model scan cannot. He
   validated it in both directions (`research/validate_gpu_busy_gate.sh`, job
   `658fe369`, exit 0): idle `[0×8]`, sustained Metal load
   `[95,96,96,96,95,95,96,96,96,96]` mean 95.7 %, idle again after exit. Real
   contention is a ~96 % plateau against a 0 % floor — the separation is not
   marginal.

   🔴 **But `Device Utilization %` is an INTERVAL counter accumulated since its
   own previous read, not a gauge.** The first read after any gap reports an
   unbounded prior window (his transcript shows 8 %, then 98 %, then 96 % with
   no change in load). So: treat the first read as an unscored **priming** read,
   require **3 consecutive** busy samples before declaring BUSY, and report
   `counter_unavailable` distinctly from `busy`. This retracts his earlier
   "7/8/9 % idle baseline" — that baseline was the artifact.

   Note his first attempt to validate the gate was itself invalid: `--base
   --cand` parsed `base_path="--cand"`, the harness exited before sampling, and
   it printed a confident IDLE against real load. **A reading that cannot change
   is not a measurement** — check that your gate can report the other answer
   before you trust the answer it gives.

## Why I am interrupting you for this

Contention does not corrupt a run into an obvious error. It inflates the arm
that happened to overlap, which reads as a real effect with a plausible
mechanism, and it is invisible after the fact. alphonse is timing **now**
(`--pairs 5 --reps 50 --inner 20`, 1000 dispatches per measurement). If you are
about to start anything heavy, coordinate on your PR first.

And a calibration for how large this can be: on his smoke config, the
byte-identical `M∈{1,2,3}` guard arm — where the true effect is *exactly* zero
by construction — measured **sd 18.368 %, worst |effect| 16.686 %, against a
pre-registered MDE of 0.5040 %.** That is 36× the MDE from dispatch-count noise
alone, before any contention. Raising reps fixes precision, not design: at
`pairs=5`, df=4, the MDE stays 0.5040 %. **An assumption-free null arm is worth
more than another hundred reps**, and if you do not have one in your design,
add one.
