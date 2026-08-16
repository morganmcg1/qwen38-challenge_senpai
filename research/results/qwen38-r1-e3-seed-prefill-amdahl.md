# Result — `qwen38-r1-e3-seed-prefill-amdahl` (revision `r1`)

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"max_reducible_fraction_of_begin","available":true,"value":0.12494091083733323},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

**Result label: `not useful` — for the Part-B mechanism.** Part A is complete and
is a successful, decisive quantification. The seed prefill is a *bigger* share of
the candidate leg than the assignment assumed (~23.9 % at the ranked window on
this host, versus the 13.4 % in the advisor's worked template), **and it is
irreducible**: 97.0 % of it is quantized GEMM already running at 87.1 % of this
machine's measured dense bf16 ceiling. The stop rule's second branch fired.

- **Student / branch:** `qwen-thorfinn` / `qwen-thorfinn/seed-prefill-amdahl-term`
- **Hypothesis and target cost:** the timed 512-token seed prefill (`begin`) is
  the campaign's Amdahl term; measure it, then cut it by >= 30 % to buy
  +0.077..+0.117 published score.
- **Decision:** **dead** for prefill-cutting. Part A answered, Part B is proven
  unreachable and was correctly not attempted. No candidate-source change is
  proposed.
- **`BASE_SHA`:** `e20268e9c2c1f35c2d75221d059e75bb95768ef6`
- **`UPSTREAM_SHA`:** `7351e62674bc600f0ca148d3a1b0604716a09db6`
- **Measurement commit:** `cdfe7a3943a275c3149d67f1b15514388894b6bb` (tooling only;
  the measured build is `BASE_SHA` semantics — no file under `editablePaths` was
  touched at any point)
- **Yukon promoted frontier used as reference:** submission
  `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, score `2.9042110287045`, `sourceRef
  7351e626`
- **Submitted candidate files:** **none.** No file in `benchmark.json ->
  editablePaths` was modified. `git diff BASE_SHA..HEAD --stat` touches only
  `research/` and `.gitignore`.
- **Supporting tooling files:** `research/capture-cli.sh`,
  `research/prefill_amdahl.py`, `research/run-amdahl-measurement.sh`,
  `research/prefill_floor.py`, `research/prefill_floor_summary.py`,
  `research/results/qwen38-r1-e3-seed-prefill-amdahl.md`, `.gitignore`
- **MTP head provenance and draft policy:** unchanged pinned head
  `hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaff972fd3ca2be08f3640603b47e9ce70a`;
  `head_provenance_sha256 = da336ce9894f859d4da39855ba35972fe8152224cf2fa0c2c7122b6cfcfc4e94`;
  `uses_pinned_mtp_head: true`. Draft policy untouched (`--mtp-depth 8`,
  `costModelDepth` not modified).
- **Assignment-scope preflight:**
  `./senpai/validate-assignment-scope.sh e20268e9... Sources/MLXFastModel/Qwen36MTPBlockSession.swift Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  -> `assignment scope OK`.
- **Editable bytes:** `./senpai/check-editable-budget.sh e20268e9...` ->
  `OK: source=2396110/3000000, headroom=603890, growth=0/262144,
  exempt=2410/2147483648, files=154`. Growth is exactly `0` because nothing
  submittable changed.
- **Scored-path reachability evidence:** verified in
  `Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` — `warmMTPDecode()`
  is untimed and completes *before* `let started = Date()`, which is immediately
  followed by `client.beginMTPDecode(seedTokens:)`. `decodeSeconds` is taken
  after the round loop. **The 512-token seed prefill is inside the timed window
  on both legs**, exactly as `fixtures/qwen3_8_27b_mtp_track.json` states.

---

## Evidence

- **Host:** AWS Mac, Apple **M4 Pro**, 20 GPU cores / 14 CPU cores,
  `hw.memsize = 51539607552` (48 GiB), macOS **26.5.2 (25F84)**, Swift **6.3.3**.
- **Memory profile:** automatic low-memory profile, identical on both legs and on
  every run reported here. Not changed at any point.
- **Thermal policy:** the wrapper's run lock and 40 °C cool gate were used
  unmodified; the gate waited before each timed leg. Nothing was bypassed.
- **Vendored MLX version:** `Vendor/mlx-swift/Package.swift` pins MLX-Swift
  **0.32.0**; the floor benchmark used pip `mlx` **0.32.0** — same kernels, no
  version skew.
- **Exact baseline command** (there is no candidate command — no code changed):

  ```bash
  research/run-amdahl-measurement.sh part-a-base-e20268e9
  # which is, in substance:
  #   MLXFAST_SWIFT_BIN=research/capture-cli.sh \
  #   MLXFAST_SCORE_PATH=<captured>/score.json \
  #   ./benchmark-qwen-mtp.sh --local-iterate
  # then:
  #   python3 research/prefill_amdahl.py --tag part-a-base-e20268e9 --wandb
  ```

  Run under `run_job` (30-minute cap). `research/capture-cli.sh` is a pass-through
  tee on the CLI's stdout reports; it changes no trusted code and no timing.
  `benchmark-qwen-mtp.sh` `mktemp`s its report dir and deletes it on `EXIT`, so the
  tee is the only way to retain `03-mtp-timed.json` / `04-mtp-timed.json`.
- **Tests and risk-based checks:** none required — **no source under test was
  changed**. The fidelity gates below come from the benchmark itself.
- **Exact-token and row-ledger verdict:** `all_tokens_matched: true` on both legs;
  `emitted_token_total = 64`; `declared_rows_total = 64`; **exact 64/64**.
- **Divergent tokens:** `residual_divergence_count = 0` on both legs.
  `public_drift_tripwire_passed: true`; `passed: true`.
- **Generated-twin audit:** not relevant, no Metal source touched.
- **Peak RAM / artifact size:** head unchanged; `weights/` 14 GB, digest
  `dabbabd181c9a5b03a0de013b7cb6195a0c890d68037ea329a095e6f9c7c9ec8`.
- **Official status:** not submitted. Nothing to submit.

### W&B runs (project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`)

| purpose | run id | URL | state |
| --- | --- | --- | --- |
| Part A leg decomposition (incl. per-round block tables for both legs) | `cwlqu3ok` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cwlqu3ok | finished |
| Prefill component floor / irreducibility proof | `ihnmmi1b` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ihnmmi1b | finished |

Both are in group `qwen38-r1-e3-seed-prefill-amdahl`.

---

## Part A — the measurement

### A0. Method, and why not `MLX_QWEN_MTP_TRACE=1`

The assignment asked for `begin` from `MLX_QWEN_MTP_TRACE=1`. **That trace is
unreachable on the Qwen path.** The trace writes to the *worker* process stderr,
and `QwenRuntimeMTPDriver` constructs the `mtp-timed`, `mtp-verify` and
`mtp-reference` phases with `forwardsWorkerStderr: false` (only the DFlash
`runBenchmark` path passes `true`), so the trace lines are discarded. Wrapping the
worker to recover them is blocked by `enforceMetallibFingerprint` and the sandbox's
`allowedExecutablePath`, and defeating either would invalidate the measurement.

So I recovered `P` by **parent-clock algebra** instead, which is strictly better
evidence because it uses only the clock the score is actually computed from. For
each leg the parent reports total `decode_seconds` and a per-round `latency`. The
per-round latency brackets only the block request; `requireStructurallySound` and
the result appends sit outside it. Therefore:

```
decode_seconds  =  P  +  sum(block_latency)  +  N * c
```

with `P` the seed prefill (everything before round 1) and `c` a constant
per-round parent overhead. Two legs give two equations in two unknowns, and
they have very different `N` (64 serial rounds vs 10 MTP rounds), so the system
is well-conditioned:

- `c = 0.00033844841851128475` s/round (338 µs — a plausible Swift-side
  structural-check + append cost)
- **`P = 4.008616434203254` s**

**Independent cross-check.** The two legs' raw residuals
(`decode_seconds - sum(block_latency)`) are `4.030277132987976` (serial) and
`4.012000918388367` (MTP). They agree to **18.28 ms, i.e. 0.45 %**, despite the
legs differing by 54 rounds and 2.7 s of decode work. Two nearly independent
measurements of a shared fixed cost landing within half a percent is strong
evidence that `P` is real and that the linear model is correct. `P` is a slight
*upper* bound (if the true `c` were 50 µs, `P` would fall by only 3.2 ms).

### A1 / A2. `P` absolute, and as a fraction of each leg

Local `--local-iterate` window (64 decode tokens, 512-token seed):

| quantity | serial leg (depth 0) | MTP leg (depth 8) |
| --- | ---: | ---: |
| `decode_seconds` (parent clock) | 8.31670892238617 | 5.6215879917144775 |
| `sum(block_latency)` | 4.286431789398193 | 1.6095870733261108 |
| rounds | 64 | 10 |
| residual (= `P` estimate) | 4.030277132987976 | 4.012000918388367 |
| **`P` fraction of leg** | **0.4819955190944846** | **0.7130754584134334** |
| `parent_measured_seconds_per_token` | 0.1299485769122839 | 0.08783731237053871 |
| steady-state s/token (excl. `P`) | 0.06628893292139447 | 0.024849048091305625 |
| `seed_token_count` | **512** | **512** |

At the 64-token local window prefill is **71.3 %** of the candidate leg. That
number is inflated by the short decode window and is *not* the ranked figure —
see A4.

### A3. Fitted leg model and sensitivity

`score = (P + D_s) / (P + D_c)` with `D_s = 4.308092488182916`,
`D_c = 1.6129715575112238` (decode work including the per-round `c` term):

- `modelled_local_score = 1.479423418194995`
- `measured_local_score = 1.479423418194995`

The model reproduces the measured local score **exactly** (to printed precision).
That is the fit quality check, and it passes.

Local-window sensitivity:

- `dScore/dP = -0.0852825605330034` per second
- **`0.00853` score points per 100 ms of prefill removed**
- `score_if_prefill_were_free = 2.6709041880628064` (= the decode-only speedup)
- `decode_only_speedup = 2.6709041880628064`, versus the measured `1.4794`
  end-to-end. **Prefill is currently costing 1.19 points of local speedup.**

### A4. Local seed length, and rescaling to the ranked window

**Answered explicitly: the local seed is exactly 512 tokens, identical to the
ranked leg.** `seed_token_count = 512` on both legs of `--local-iterate`. There is
no seed-length correction to make. The only difference from ranked is the *decode*
window: local `--local-iterate` decodes 64 tokens, `--local-submit` 128, ranked
512.

Because `P` is a fixed per-leg cost and the steady-state rate is measured
directly, rescaling to a 512-token decode window on **this host** is a clean
extrapolation (no new assumption beyond "steady state stays steady"):

| quantity | value |
| --- | ---: |
| `ranked_window_leg_seconds_serial` | 38.01415087117089 |
| `ranked_window_leg_seconds_mtp` | 16.753961536619396 |
| `ranked_window_prefill_fraction_of_serial_leg` | 0.10545063725843477 |
| **`ranked_window_prefill_fraction_of_mtp_leg`** | **0.23926379593516184** |
| `ranked_window_modelled_score` | 2.268964912452662 |
| `ranked_window_dscore_dprefill_per_second` | -0.07574118572965953 |
| **score points per 100 ms removed (ranked window, this host)** | **0.00757** |
| score if `P` cut 20 % | 2.332740208719386 |
| score if `P` cut 30 % | 2.367093722484924 |
| score if `P` cut 50 % | 2.441402646828216 |
| score if `P` were free | 2.668074827610685 |

**Headline for the campaign: prefill is ~23.9 % of the candidate leg, not
13.4 %.** The advisor's worked template assumed `begin ~= 0.9 s` from a pre-warm-fix
dev-box phase trace; the measured share on this host is roughly **2x** that
assumption. On *share of the leg* alone, Part A fully justified opening Part B.

**Ranked-`P` band — present as a band, not a fact.** Naively scaling this host to
the ranked box by the leg ratio `f = 19.453 / 38.014 = 0.512` gives ranked
`P ~= 2.05 s`, i.e. ~30 % of the 6.699 s ranked candidate leg. The advisor's 0.9 s
assumption gives 13.4 %. The truth is likely toward the **lower** end, for a
mechanical reason that A5 makes precise: **prefill is compute-bound while decode
is bandwidth-bound**, and M5 lifts compute (Neural Accelerators) considerably
more than it lifts memory bandwidth, so uniform scaling over-states ranked `P`.
I am reporting `13.4 %..30 %` as the honest band and **not** claiming a ranked
number. The irreducibility conclusion below does not depend on where in that band
the truth sits.

### A5 (new deliverable). Stall-guardrail margin

`max_block_request_seconds` / `p50_block_request_seconds`, from the parent's
per-round `latency` (full per-round tables are logged to run `cwlqu3ok`):

| leg | `max_block_s` | `p50_block_s` | **max/p50** | margin to 4.0x | max *is* first block? |
| --- | ---: | ---: | ---: | ---: | --- |
| serial (depth 0) | 0.1102290153503418 | 0.0649939775466919 | **1.6959881439961555** | 2.3040118560038447 | **yes** |
| MTP (depth 8) | 0.1901090145111084 | 0.16856098175048828 | **1.1278352352771444** | 2.8721647647228554 | **no** |

Excluding the first block: serial 1.5067561247154762, MTP unchanged at
1.1278352352771444 (the MTP maximum is the *last* round, not the first).

**Both legs are far inside the 4x rejection threshold here — but read this with
the fixture's calibration caveat, not as reassurance.** The fixture records
serial **5.72–5.90x** and depth-2 **3.30–3.36x** *at the 512-token window*. My
window is 64 tokens. `p50` is a median over 64 (or 10) rounds and is
window-insensitive, but the *first* block's one-time warm cost is the numerator,
and the ranked box, its warm state and its 512-round schedule are all different.
**Local max/p50 does not transfer**; treat the fixture's 3.30–3.36x as the live
figure and my numbers as a same-shape sanity check only. The guardrail is
enforced in the trusted parent on the ranked box and is not reproducible here.

Two observations that are transferable, and that I think matter more than my
ratio:

1. **On the serial leg the maximum genuinely is the first block** (0.1102 s vs a
   0.0650 s median, 1.70x), which independently confirms the fixture's
   "first block after the seed prefill is a one-time warmup, not a stall"
   diagnosis using a completely separate measurement.
2. **On the MTP leg the maximum is *not* the first block.** First block is
   0.17828190326690674 s, `p50` is 0.16856098175048828 s, and the maximum
   0.1901090145111084 s occurs at the end of the run. So on the candidate leg the
   first block is already only ~5.8 % above the median — **`warmAllDepths` is
   doing its job and there is essentially no first-touch warm cost left to
   remove.** That is important for the advisor's framing: this experiment was
   nominated as "the natural defence" against the guardrail, but the candidate
   leg's warm coverage is already effectively complete, so **`begin` /
   `warmAllDepths` work cannot buy guardrail margin either.** If the ranked
   depth-2 leg really sits at 3.30–3.36x, the numerator is coming from something
   other than residual first-touch JIT — most plausibly thermal or scheduler
   variance across a 512-round leg. That is a different investigation.

---

## The irreducibility floor — why Part B was not attempted

The stop rule permits stopping on "proven irreducible (e.g. `begin` is already at
the weight-streaming floor for one full pass — compute that floor and say so)". I
computed a **stronger and more appropriate** floor.

The weight-streaming floor is the wrong bound here, and it is worth saying why:
~14.1 GiB at this host's ~273 GB/s is ~55 ms, only **1.4 %** of `P`. Prefill is
**not** bandwidth-bound — it amortizes each weight read over 512 rows. (The
contrast is stark: serial *decode* moves 14.1 GiB in 0.0673 s = ~224 GB/s = **~82 %
of peak bandwidth**. Decode is bandwidth-saturated; prefill is compute-saturated.
These are two different machines' worth of bottleneck inside one leg.)

So I built the **compute floor** instead: `research/prefill_floor.py` times every
operation the 512-token prefill executes, at its exact scored shape, dtype and
quantization (affine 4-bit group-64), using MLX 0.32.0 — the same version the
Swift build vendors. Model geometry taken from `weights/config.json`: hidden
5120, intermediate 17408 (dense), 64 layers = 48 `linear_attention` + 16
`full_attention`, head_dim 256, 24 q-heads / 4 kv-heads, `attn_output_gate: true`,
conv kernel 4, vocab 248320.

| component | calls | total s | % of `P` | TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| `mlp:gate_up_down` | 64 | 2.7297653779387474 | 68.10 | 6.419 |
| `linear_attn:in_proj_fused_qkvzba` | 48 | 0.6420700326561928 | 16.02 | 6.459 |
| `linear_attn:out_proj` | 48 | 0.24591201916337013 | 6.13 | 6.288 |
| `full_attn:qkv_proj` | 16 | 0.18754867278039455 | 4.68 | 6.412 |
| `linear_attn:gated_delta_kernel_T512` | 48 | 0.12787800282239914 | 3.19 | 1.209 |
| `full_attn:o_proj` | 16 | 0.08186267130076885 | 2.04 | 6.296 |
| `full_attn:sdpa_causal_512` | 16 | 0.022179342806339264 | 0.55 | 2.324 |
| `linear_attn:conv1d_depthwise_k4` | 48 | 0.011751987040042877 | 0.29 | 0.171 |
| `head:lm_head_single_row` | 1 | 0.003095707972534001 | 0.08 | 0.821 |
| `head:final_norm_full_512` | 1 | 0.0001674171071499586 | 0.00 | 0.031 |
| **modelled total** | | **4.052231231587939** | **101.1** | |
| **measured `P`** | | **4.008616434203254** | 100.0 | |
| `ceiling:dense_bf16_gemm_512x5120x17408` | ref | — | — | **7.363** |

### The three conclusions this forces

**1. There is no dead work, no launch-gap slack, and nothing schedulable left in
`begin`.** The model *over*-predicts by **+1.1 %**. A sum of isolated per-op
medians is a best-case *serial-kernel upper bound* — it charges every kernel in
full with no overlap. The measured `P` comes in **below** it, which is only
possible because MLX's `asyncEval` pipelining already overlaps launches. There is
no gap between "the kernels the prefill must run" and "the time the prefill
takes". Any Part-B mechanism would have to make the *kernels themselves* faster,
which is outside this assignment's scope and is not what Part B proposed.

**2. The hard ceiling is 12.49 %, which is below the 20 % stop threshold.**
Quantized GEMM is **3.8871587738394737 s = 97.0 % of `P`**, running at **6.415
TFLOP/s = 87.1 % of the 7.363 TFLOP/s dense bf16 ceiling I measured on this same
host** (that ceiling matches theory: 20 cores x 128 ALU x 2 flop x ~1.5 GHz
~= 7.68 TFLOPS). If 4-bit dequantization became **entirely free** — physically
impossible; the weights are 4-bit affine group-64 and must be unpacked — the GEMM
would take 3.386318585352616 s and the total saving would be
**0.5008401884868579 s = 12.494 % of `P`**. The stop rule requires a mechanism
winning **>= 20 % of `begin`**. **The absolute upper bound is 12.49 %, so no
mechanism inside this assignment's scope can reach the threshold.** The stop rule
fires on the "proven irreducible" branch.

**3. Both proposed Part-B mechanisms are individually dead, for reasons the floor
makes explicit.**

- **Mechanism (1), dead work in `begin`.** The assignment asked whether the final
  norm runs over all 511 leading rows and whether head-history priming runs a
  512-row head forward inside the timed window. Measured: a full 512-row final
  norm is **0.167 ms = 0.004 % of `P`**. Even if it were entirely eliminated it
  is 4 parts in 100,000. And the base already routes only the last row through
  `applyLMHead` (3.1 ms, 0.08 %); `seedLogits` is a deliberately-dead lazy graph
  and `seedHiddenForPriming` retains pre-norm hidden state (~5 MB) without
  forcing a head forward. **`begin` was already audited and trimmed by whoever
  wrote it.** There is nothing here.

- **Mechanism (2), chunked prefill (512 vs 2x256 vs 4x128).** Predicted **<= 0**,
  for two independent reasons, and I chose not to spend a run confirming a
  prediction with no upside:
  1. 97 % of `P` is throughput-bound GEMM already at 87 % of the dense ceiling.
     Halving `M` halves arithmetic intensity per weight read. On a machine whose
     GEMM is already near its compute ceiling at `M = 512`, reducing `M` moves
     *away* from the ceiling, not toward it. There is no occupancy headroom to
     recover — 87.1 % is not a number with 13 % of "wrong kernel selection" in it.
  2. The Gated DeltaNet scan does not benefit. I read the kernel
     (`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift:128`): it is a
     **strictly sequential per-timestep recurrence**, `for (int t = 0; t < T; ++t)`
     with two `simd_sum` per step, grid `(32, Dv, B*Hv)`, threadgroup `(32,4,1)`,
     state held in registers as `float state[Dk/32]`. **`T` is an input, not part
     of the grid.** So 2x256 executes exactly the same number of sequential steps
     as 1x512, plus one extra kernel launch and one extra recurrent-state
     round-trip. It is strictly worse. The prefill already takes this path as a
     single chunk at `nConfirmed == 0`
     (`Qwen35.swift:1094-1105 -> processChunk -> gatedDeltaUpdateMemoG ->
     gatedDeltaKernel`), and `Qwen35TextModelInner.callAsFunction`
     (`Qwen35.swift:1865-1888`) already gates a `prefillLadder` on
     `inputs.dim(1) >= 512` with `asyncEval` at `i == 0 || i % 4 == 3` — the
     large-`M` path is the one that was deliberately built and tuned.

  Chunking would also have been the expensive branch: it changes the prefill
  shape, so it would require a new `warmAllDepths` entry warming the *exact*
  scored expression (the file's recorded 0.941 s vs 0.402 s first-block loss is
  the cost of getting that wrong), plus hexfloat cache-state equality proofs on
  KV and recurrent state. Spending that on a mechanism with a negative predicted
  sign and a 12.49 % ceiling would have been poor use of the allocation.

- **Mechanism (3), memory-profile interaction.** Not varied. The profile was held
  at the automatic low-memory profile for every measurement, as required. With
  the compute floor established, a profile change cannot move a term that is
  97 % compute-bound GEMM at 87 % of ceiling.

---

## Metric table

The template's baseline/candidate table does not apply cleanly — **there is no
candidate build**, because Part A's answer was to not build one. Filling it in
honestly:

| Metric | Baseline (`BASE_SHA`, measured) | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.1299485769122839 | n/a (no candidate) | — |
| MTP seconds/token | 0.08783731237053871 | n/a | — |
| local serial-relative speedup | 1.479423418194995 | n/a | — |
| effective mean draft length | 5.4 | n/a | — |
| accepted draft rate | 1.0 | n/a | — |
| `P` (seed prefill wall) | 4.008616434203254 s | n/a | — |
| max reducible fraction of `P` | 0.20 (stop-rule threshold) | 0.12494091083733323 (proven bound) | **-0.0751, below threshold** |

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts.

### On the ratio-cancellation warning

The assignment warned twice that a prefill win **cancels in the local serial:MTP
ratio**, because both local legs run the same candidate build, and instructed
that the primary metric be **absolute candidate leg wall** and **absolute `begin`
wall** against a fresh unchanged-`BASE_SHA` run. **Stating it explicitly as
required: the local ratio would have cancelled, and I did not use it as a signal
anywhere.** Everything in this report is absolute wall time on a fresh
unchanged-`BASE_SHA` run, decomposed on the parent clock. The only place a ratio
appears is `decode_only_speedup`, which is a derived quantity used to size the
prize, not a stop signal.

---

## Conclusion

**What happened and why.** Part A ran once, as budgeted, and produced a clean
two-equation decomposition of both legs on the parent's own clock, cross-validated
to 0.45 % by two nearly independent residuals. It found `P = 4.0086 s`, i.e.
**23.9 % of the candidate leg at the ranked window on this host — about twice the
share the assignment assumed.** On share alone Part B was justified. I then spent
one cheap analysis run (3.4 s) building the compute floor before spending a
full measurement allocation on a mechanism, and the floor closed the question: `P`
is 97.0 % quantized GEMM at 87.1 % of this machine's measured dense bf16 ceiling,
the sum-of-isolated-kernels model *over*-predicts the measured time by 1.1 %, and
the absolute maximum extractable under a physically impossible free-dequantization
assumption is **12.49 % — below the 20 % stop threshold.**

The assignment's premise was right and its arithmetic was right. The prize is
real and it is bigger than estimated. It is simply already collected: `begin` is
a dense 512x5120x17408-class GEMM sequence that the existing code already runs at
near-roofline efficiency, with its lazy graphs already trimmed and its warm
already load-bearing.

**Evidence for or against the mechanism.** Against, decisively: the +1.1 % model
over-prediction (no slack), 87.1 % of ceiling (no efficiency headroom), 0.004 %
for the final norm (mechanism 1 is noise), the sequential `for (t = 0; t < T; ++t)`
recurrence with `T` outside the grid (mechanism 2 has a negative predicted sign),
and 12.49 % < 20 % (the stop rule's own arithmetic).

**Prompt and M5 transfer risk.** Transfer of the *measurement* is unusually clean,
as the advisor noted: every leg on every prompt processes exactly 512 seed tokens,
so `P` is a property of the model and machine, not the prose, and there is no
depth-schedule contamination. Transfer of the *conclusion* is also robust: the
irreducibility argument is a roofline argument, and M5's Neural Accelerators
raise compute more than bandwidth, which pushes `P` **down** relative to decode
and makes the prefill share *smaller* on the ranked box, not larger. The one thing
that does **not** transfer is my max/p50 guardrail ratio (64-token window, wrong
box); the fixture's 3.30-3.36x remains the live figure.

**Smallest useful next action.** Redirect. The same decomposition that killed this
line points straight at where the time actually is. On the candidate leg:
`P = 4.009 s` is irreducible compute, but decode is `1.613 s` of work running at
**~82 % of peak memory bandwidth** — a completely different bottleneck with a
completely different lever set. And within `P` itself, the only component *not*
near roofline is `gated_delta_kernel_T512` at **1.209 TFLOP/s (3.19 % of `P`)**,
which is sequential by construction; a chunkwise-parallel Gated DeltaNet
formulation is the only structurally interesting prefill target left, and at
3.19 % of `P` it is worth at most ~0.024 score points even if it went to zero. I
do not recommend it.

**Recommendation: close.** Close the prefill-cutting line for this campaign at
this base. Do not re-open it unless (a) the target checkpoint's quantization
changes such that GEMM efficiency drops well below 87 % of ceiling, or (b) an M5
measurement shows `P` is a *larger* fraction of the ranked candidate leg than the
30 % top of my band *and* shows the ranked box's quantized GEMM running far from
its own roofline. Both are recorded here as the specific evidence that would
justify reopening.

**Two campaign-level findings worth carrying forward regardless of this
result:**

1. **`warmAllDepths` has essentially no first-touch cost left on the candidate
   leg** (first block 0.1783 s vs `p50` 0.1686 s, +5.8 %, and the run maximum is
   the *last* round). So this experiment cannot be the guardrail defence it was
   nominated as. If the ranked depth-2 leg is really at 3.30-3.36x, the numerator
   is not residual JIT.
2. **The candidate leg is two different machines.** Prefill is compute-bound
   (bandwidth floor is 1.4 % of `P`); decode is bandwidth-bound (~82 % of peak).
   Any optimization proposal should state which of the two it is attacking,
   because a technique that helps one is close to irrelevant for the other.

### Acknowledging the advisor's follow-up (`qwen38-r1-e3-fb1-sensitivity-and-guardrail`)

1. **Prompt-invariance:** confirmed and used. `seed_token_count = 512` on both
   legs, so the local-to-ranked path for this measurement needs no prompt
   correction. Recorded in A4.
2. **Worked sensitivity template:** re-measured on this host with my own numbers
   and stated assumptions, per instruction. My `P` share is ~2x the template's
   0.9 s assumption; I report the ranked-`P` band as 13.4 %..30 % and explain why
   I expect the lower end. The template's exchange rate
   (~0.043 score per 100 ms of candidate wall) is unchallenged by this result —
   it is the *reachable* prefill milliseconds that turned out to be zero.
3. **Stall guardrail:** delivered in A5 for both legs, with the explicit caveat
   that a 64-token local window does not reproduce the fixture's 512-window
   ratios, plus the two transferable observations above.
4. **Bit-identical hexfloat / exact warm:** not exercised, because no prefill
   shape or state-writing code was changed. Had mechanism (2) proceeded, it would
   have required both.

### Note on process

I do not have `post_assignment_comment` in this session, so I could not post the
Part-A number as an interim update before deciding not to run Part B. The
decision to stop is made under the assignment's own stop rule ("proven
irreducible ... compute that floor and say so"), and the floor is above.
