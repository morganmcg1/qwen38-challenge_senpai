# E29 — Round overhead and the host graph-build ladder

Student: `qwen-alphonse` · PR #34 · base `d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602`
Host: Apple M4 Pro `Mac16,11`, 48 GiB, 20 GPU cores, NAX OFF.
Local mode: `--local-iterate`, 256 decode tokens, depth 8, declared proposal head.

## Headline

Two results, one negative and one methodological.

1. **The verify forward is GPU-bound at the widths the scheduler actually uses.**
   Per-round "host tail" in the shipped build is *not* host CPU work: it is host
   time spent blocking on GPU backpressure. The genuinely removable host cost is
   **4.35 % of decode**, not the 10.26 % E20 recorded and not the 53.9 % a naive
   reading of the segment trace suggests.
2. **The decode `asyncEval` ladder is worth ~14 % on serial (M=1) decode and
   ~0 % on the MTP leg.** Because the local harness runs both legs on the
   candidate build, disabling the ladder produces a **+16.5 % local ratio
   "win" that is entirely an artefact of slowing the baseline leg.** On the
   ranked run the serial leg is pinned, so the real effect on score is the
   MTP-leg change: −0.11 %, i.e. noise.

## Attribution: the six-way per-round split

`Qwen36MTPBlockSession` was instrumented to tile `round_us` into six disjoint
segments. The split is exact: `unaccounted` is 0.08–0.09 ms per leg, **+0.001 %
of round time**, so no cost hides between the buckets.

Whole-leg validation (T1): trace round sum 6271.14 ms vs parent
`04-mtp-timed.json` 6.2784 s (0.1 %); prefill `begin` 4.0055 s;
6.278 + 4.006 = 10.28 s ≈ reported `decode_seconds` 10.2716 s. Prefill is 73.9 %
host build and is *not* part of true decode — `decode_seconds` is
prefill-inclusive and must not be used as the decode denominator.

### Why E20's 10.26 % was an over-estimate

E20 obtained overhead subtractively, `overhead(M) = block_m1(M) −
attributed_m1(M)` (`research/e20_analyze.py:417`). That folds host graph build
into `target_work` and produced an unphysical −1.46 ms fit floor. The direct
six-way split replaces the subtraction with a measurement.

## The ladder sweep

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
(`Qwen35TextModelInner.callAsFunction`, ~:2164-2245) fires `asyncEval` at decode
layers `0,1,9,19,29,39,49,57` — a 40-layer Laguna schedule scaled to 64 layers.
`ladderActive = inputs.dim(1) <= 9 || prefillLadder`, so it is live at **every**
MTP verify width, not just `S <= 2` as the stale in-code comment claimed (comment
corrected in this branch).

A file-scope `qwen35DecodeLadderRungs` now reads `MLX_QWEN_MTP_LADDER` once
(`default` / `off` / `front` / `dense` / explicit index list), shipping the
original schedule when unset.

| arm | rungs | serial s/tok | MTP s/tok | MTP Δ vs D0 | local ratio | steady round |
|---|---|---|---|---|---|---|
| D0 | default (8) | 0.0824759 | 0.0391876 | — | 2.1046 | 5741.67 ms |
| L0 | off (0) | 0.0959426 | 0.0391430 | −0.11 % | 2.4511 | 5721.76 ms |
| L1 | front (2) | 0.0944483 | 0.0391632 | −0.06 % | 2.4116 | 5721.87 ms |
| L2 | dense (17) | 0.0810077 | 0.0390464 | −0.36 % | 2.0747 | 5711.69 ms |

**The MTP leg is flat: 0.36 % total spread from 0 to 17 rungs**, below the
measurement noise floor established on this host (trace perturbation alone is
1.54 %, and the D0-vs-N1 same-config repeat differs by 0.86 %). The serial leg
spread over the same arms is **18.4 %**. Ladder placement is a large lever at
M=1 and a non-lever at the widths the scheduler actually picks.

`L2` was included as a falsification control: if interior rungs caused the
stalls, denser rungs had to be worse. Dense is instead nominally the *fastest*
arm on both legs, so the "interior rungs stall wide widths" hypothesis is
rejected rather than merely unsupported. The −0.36 % is not claimed as a win.

The serial ordering is monotone in rung count — dense (0.08101) < default
(0.08248) < front (0.09445) < off (0.09594) — and front-only recovers just 1.6 %
of the 16.3 % gap. So the serial benefit comes from **interior** rungs spreading
dispatch across the 64-layer depth, not from starting the GPU early.

All four arms: `all_tokens_matched: true`, `residual_divergence_count: 0`,
`effective_mean_draft_len` 6.5143, `accepted_draft_rate` 0.9737 — identical at
0, 2, 8 and 17 rungs, confirming the ladder is bit-identical as documented.

`D0` also guards the `switch` → `Set.contains` refactor: 0.0391876 s/tok vs the
untraced pre-refactor control `N1` 0.0395148 s/tok — no regression.

### The ladder only relocates time

With the ladder on, mid-forward `asyncEval` encodes dispatches inline on the
calling thread, so GPU wait lands in `verify_build_us`. With it off, all GPU work
lands at the terminal `eval()` and is measured as `eval_wall_us`. The sum is
conserved per width:

| M | D0 vbuild+eval | L0 vbuild+eval | diff |
|---|---|---|---|
| 2 | 70.379 ms | 71.656 ms | +1.8 % |
| 6 | 138.445 ms | 139.705 ms | +0.9 % |
| 7 | 147.351 ms | 149.431 ms | +1.4 % |
| 8 | 159.526 ms | 160.423 ms | +0.6 % |
| 9 | 198.231 ms | 197.925 ms | −0.2 % |

Total round time is likewise unchanged: 6028.75 ms (D0) vs 6022.23 ms (L0),
0.11 % apart, and steady round time across all four arms spans only
5711.69–5741.67 ms (0.53 %). The ladder moves 85–99 ms/round between buckets and
creates or destroys nothing at these widths.

The host-tail share is **non-monotonic** in rung count — off 4.42 %, front
5.77 %, dense 35.67 %, default 53.75 % — which is itself evidence for the
backpressure mechanism rather than for host work. Eight widely spaced rungs make
each flush cover ~8 layers, enough ops to push past the 10-command-buffer
ceiling and block the host; 17 rungs every 4 layers flush smaller batches that
stay nearer the ceiling and block less. Zero rungs move all GPU time to the
terminal `eval()`, where it is measured as `eval_wall_us` instead. Host-attributed
wait therefore peaks at intermediate rung density, which no host-CPU-work model
predicts.

### Therefore: `verify_build_us` is not host graph-build time

In the ladder-off arm the host segments contain only lazy graph construction,
because nothing forces work to the GPU mid-forward. That isolates the real host
cost:

| segment | L0 total | share of round |
|---|---|---|
| draft_build_us | 168.11 ms | 2.79 % |
| verify_build_us | 85.19 ms | 1.41 % |
| readout+commit+upkeep | 8.84 ms | 0.15 % |
| **host tail** | **262.23 ms** | **4.35 %** |
| eval_wall_us | 5760.00 ms | 95.65 % |

The whole 64-layer verify graph costs **2.4 ms/round** to construct. Round
overhead is therefore ~4.35 % of decode and ~95.6 % is GPU execution. Host-side
scheduling levers — launch fusion, `asyncEval` pipelining, per-round Swift setup
— are bounded above by that 4.35 %, and most of it is draft-chain work rather
than the verify forward.

Bookkeeping (`readout` + `commit` + `upkeep` = 0.15–0.19 %) is definitively not
the problem.

## Mechanism

The head-chain drain probe (`MLX_QWEN_MTP_TRACE_SYNC_HEAD=1`) moves 12–13.5 ms
per round out of `verify_build_us` into `draft_build_us`, ~2.3 ms per extra head
step — so ~10 ms of head-chain GPU time was hiding inside the verify bucket. The
remaining 62.8–89.8 ms of `verify_build_us` is neither head-chain GPU time nor op
construction: the dispatch inventory is constant at 969 ops for M=6..9, which
would imply 70–90 µs per op, ~20× a plausible host op-construction cost.

The mechanism is MLX backpressure. `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp:25`
sets `MAX_ACTIVE_TASKS = 10`, and inside `eval_impl`'s per-array loop (~:269-281)
the host calls `gpu::finalize` then `scheduler::wait_for_one()` once more than 10
command buffers are in flight — **per primitive, not per eval call**.
`needs_commit()` (`backend/metal/device.cpp:484-487`) commits every 50 ops or
50 MB (arch default for `'s'`, `:573-595`), so 969 dispatches per verify forward
is ~19 commits, well above the 10-buffer ceiling. That is why
`verify_build_us` grows 34.6 → 89.8 ms while the op count stays fixed: it tracks
GPU work per dispatch, not host work.

`asyncEval` cannot hide this. Verified against MLX 0.32.0 (JIT build,
`MLX_METAL_DEBUG` undefined): `eval()` and `asyncEval()` both build the tape and
encode every Metal dispatch inline on the calling thread; `asyncEval` only skips
the terminal wait.

## Full-accept economics

`round_ms / M` in steady state (M = draft_len + 1), an upper bound that assumes
full acceptance:

| M | rounds | D0 ms/token | L0 ms/token |
|---|---|---|---|
| 2 | 1 | 35.425 | 36.355 |
| 6 | 9 | 23.533 | 23.458 |
| 7 | 3 | 22.026 | 21.820 |
| 8 | 5 | **20.959** | **20.780** |
| 9 | 15 | 22.956 | 22.910 |

M=8 is the cheapest width, yet the scheduler picked M=9 in 15 of 33 steady
rounds. Running every round at M=8 would be 8.0 % (D0) / 8.5 % (L0) cheaper —
a full-acceptance bound only, and schedule choice is E25's territory.

The **M=9 cliff** is not: from M=8 to M=9 the row count grows 12.5 % but
`vbuild` grows 15.9 ms and `eval` 19.5–21.2 ms, ~22 %, at *identical* dispatch
count. That is a GPU shape/tiling inefficiency, and since the round cost is
GPU-bound it is the largest remaining prize visible from this experiment.

## Reproduction

```bash
bash research/rebuild.sh
bash research/e29-run.sh D0=1:8:default L0=1:8:off L1=1:8:front L2=1:8:dense
python3 research/e29_analyze.py .mlxfast-private/e29/runs/{T1,S1,D0,L0,L1,L2} \
  --json-out research/results/e29-analysis.json
```

`research/e29-run.sh LABEL=TRACE:DEPTH[:LADDER]`. Each arm runs its own serial
leg, so the ratio self-normalises against thermal drift. One arm ≈ 3 min.

## Follow-ups not implemented

- **Retune or delete the interior ladder rungs.** They are measurably worth
  nothing at M≥6 and ~14 % at M=1. A width-aware schedule (ladder only when
  `S <= 2`, matching what the comment always claimed) would keep the serial win
  and simplify the decode path.
- **Command-buffer geometry.** `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`
  are unexplored campaign-wide. External `MLX_*` injection is blocked by the
  worker allowlist, so this needs `setenv` from editable code *before* the first
  `metal::device()`; env accessors cache in function-local statics
  (`utils.h:178-188`) and `metal::device()` is a leaked singleton
  (`device.cpp:876-882`), so a late `setenv` is a silent no-op.
  `Sources/MLXFastModel/Qwen35RuntimeWeights.swift:45` sets only
  `MLX_MAX_MB_PER_BUFFER=128` and is dead on the scored path
  (`QwenRuntimeMTPWorker.swift:487` gates on `policy.isLowMemory`, 64 GiB).
- **`ConcurrentContext` barrier suppression** (`device.h:33-45`, `start_concurrent()`
  `:88-90`) has exactly one user tree-wide (`backend/metal/slicing.cpp:35`), so
  independent per-head and per-draft-row dispatches currently take an interposed
  barrier.
- **The M=9 cliff** — a ~22 % cost step for a 12.5 % row increase at constant
  dispatch count.
