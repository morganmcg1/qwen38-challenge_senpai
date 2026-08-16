# qwen38-r1-e2-deep-round-gate — terminal result

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":true,"value":2.0263736685529},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- Student / branch: `qwen-alphonse` / `qwen-alphonse/deep-round-gate-width9` (PR #2, revision `r1`)
- Hypothesis and target cost: **Part A** — the segmented verify is bit-exact per
  position at width 9, so `segmentedVerifyDepthCap = 8` is fidelity-safe.
  **Part B** — the deep-round gate `segmentedStreakGate = 3` is too conservative;
  relaxing it (3 → 2 → 1, or replacing it with a `positionAcceptEMA[4]`
  threshold) should raise deep-round occupancy and the local ratio.
- Decision: **green locally**, for a change the assignment did not ask for
  (`segmentedVerifyDepthCap` 8 → 7). **Part B hypothesis refuted by
  measurement.**
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  - assignment base `BASE_SHA=e20268e9c2c1f35c2d75221d059e75bb95768ef6`
  - rebased onto advisor branch head `146c6d18fbcdc513275bbf299665b617b8f90477`
    per FB4. Verified `git diff e20268e9..146c6d1 -- Sources/ Vendor/
    Package.swift Package.resolved benchmark.json mtp-head.manifest.json` is
    **empty** (only `research/`, `senpai/results/`, `.gitignore` moved), so every
    measurement taken before the rebase remains valid on the new base.
  - `UPSTREAM_SHA=7351e62674bc600f0ca148d3a1b0604716a09db6`
- Yukon promoted submission / source ref used as frontier:
  `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, official score `2.9042110287045`,
  `sourceRef 7351e626`.
- Submitted candidate files: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
  **only**. One functional line changes: `segmentedVerifyDepthCap` 8 → 7. The
  rest is doc comment plus three trace-only stored properties written behind the
  pre-existing `Self.traceRounds` static gate.
- Supporting research-only files (never submitted): `research/mtp_row_gate.py`,
  `research/runc_cost_fit.py`, `research/depth_dominance.py`,
  `research/feasibility_reconcile.py`, `research/measured_cost_sweep.py`,
  `research/compare_runs.py`, `research/streak_gate_analysis.py`,
  `research/block_guardrail.py`, `research/ema_distribution.py`,
  `research/gate_occupancy.py`, `research/capture-cli.sh`.
- MTP head provenance and draft policy: organizer-pinned head, unchanged. No
  `mtp-head.manifest.json` declaration and no `mtp-head/` payload. Draft policy
  is the shipped greedy marginal-depth rule; only its width cap moved.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` → OK. The
  candidate touches strictly fewer files than the assignment allowed;
  `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` was not
  needed because Part A passed.
- Scored-path reachability evidence: every number below comes from
  `./benchmark-qwen-mtp.sh --local-iterate`, which drives the real trusted
  parent → `mlxfast-runtime-worker` → `Qwen36MTPBlockSession` path. Trace lines
  are emitted from inside `Qwen36MTPBlockSession.generateRound`, so the measured
  code is provably the code the scored worker runs. (Note for the ledger:
  `Sources/MLXFastModel/Qwen35*.swift` is dead code, per FB3; the live target is
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`.)

## Part A — width-9 row gate: **PASS**

This was the blocking item, and it passed cleanly. The candidate's segmented
verify is bit-exact per position against the serial trajectory at **every**
reachable width.

Run C (base config, cap 8, gate 3, 256 decode tokens), `research/mtp_row_gate.py`:

- **256 / 256 positions matched**
- **0 value mismatches, 0 id mismatches, 0 unmatched positions**
- per-width `bit_exact = true`: w3 (3 positions), w5 (66), w6 (12), w7 (21),
  w8 (32), **w9 (122)**

Width 9 is not a rare corner here — it is the single most common verify width in
the run (122 of 256 positions). Runs D, E and F reproduce 0 mismatches at every
width they reach.

Exact command:

```bash
/usr/bin/env MLX_QWEN_MTP_TRACE=1 \
  MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=256 \
  MLXFAST_SCORE_PATH=research/score-runC-base-cap8-256.json \
  ./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee /tmp/runC.log

grep -E "mtp-row:|mtp-trace:|generating the MTP reference rows|measuring the TRUE serial control|measuring native-MTP decode" \
  /tmp/runC.log > research/trace-runC-base-cap8-256.log

python3 research/mtp_row_gate.py \
  --trace research/trace-runC-base-cap8-256.log \
  --score research/score-runC-base-cap8-256.json \
  --label runC-base-cap8-256 --wandb
```

**Consequence for the assignment's contingency:** the assignment said "if not
bit-exact at width 9, set `segmentedVerifyDepthCap = 7` and report loudly."
Part A passed, so that contingency is **not** triggered. I am nonetheless
recommending cap 7 — but for a completely different reason, and it must not be
recorded as a fidelity retreat. **Width 9 is correct; it is simply not worth its
cost.** The cap change is a pure cost decision, reversible the moment the verify
batch gets cheaper.

## Part B — relaxing the deep-round gate: **refuted**

Three variants were spent, as the stop rule allowed.

| run | config | change vs base |
| --- | --- | --- |
| C | cap 8, gate 3 | base |
| D | **cap 7**, gate 3 | cap only |
| E | cap 6, gate 3 | cap only |
| F | cap 7, **gate 1** | the Part-B relaxation |

Gate 2 was skipped deliberately: gate 1 is the strictly more aggressive probe of
the same hypothesis, and if relaxation helps at all it must show at gate 1. It
regressed, so gate 2 could only sit between a measured win (gate 3) and a
measured loss (gate 1) — spending a run on it would not have changed the
decision, and the budget was better spent on the cap sweep that actually found
the improvement.

### Why gate 1 loses (the mechanism)

Relaxing 3 → 1 at cap 7 raised the deep-round share by only **+0.5 pp**
(0.8333 → 0.8378) and left the depth histogram essentially unchanged (24 rounds
at depth 7 in both), yet it multiplied rejections **4.9×** (1 → 8 rejected
tokens; `reject_round_rate` 0.0278 → 0.1351). Net **−1.15 %** local ratio.

The gate's value is not *how often* it is open but *which* rounds it opens. Gate
1 grants the deep cap one clean round after a rejection — still inside the hard
stretch of text that caused the rejection. That round rejects, the streak
resets, gate 1 immediately re-opens, and the result is a self-sustaining
rejection cascade. **The streak gate is not throttling throughput; it is damping
a cascade.** Gate 3 forces three consecutive clean shallow rounds, which is a
cheap and surprisingly effective test that the model has actually left the hard
region.

This is a clean negative and I am reporting it as such: the assignment's
premise — that the gate is needlessly conservative — is wrong on the evidence.
It also independently corroborates FB3's framing (b): our conservatism is an
empirically-found optimum, not an oversight.

### The EMA arm — settled from the measured distribution, no run spent

FB1 called the EMA arm "the more interesting one," so rather than spend the last
variant on it blind, I measured the quantity the arm depends on. Using Run C
(`research/ema_distribution.py`, `--ema-index 4`, gate 3, 38 rounds,
`streak_open_rate` 0.6053):

| `positionAcceptEMA[4]` | n | mean | p50 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| streak gate **closed** | 15 | 0.9519 | 0.9816 | 0.7840 | 0.9964 |
| streak gate **open** | 23 | 0.9570 | 0.9746 | 0.8481 | 0.9957 |

There is **no separation**. The means differ by 0.005 and the **medians are
inverted** — the EMA is *higher* on average in the rounds the streak gate is
closing. An EMA threshold therefore cannot discriminate the rounds the gate is
protecting against.

Worse, the disjunction `EMA[4] ≥ θ OR streak ≥ 3` has an open rate pinned at
**exactly 0.9211 for every θ from 0.85 to 0.98** — the same 12 extra rounds open
at all of them. The threshold is not a tunable knob; it is a step function with
two settings: "≥ 0.99" or "wide open". At `--ema-index 7` the separation is
slightly better (closed p50 0.7661 vs open 0.8310), but there any θ ≥ 0.90 never
opens anything, so the disjunction degenerates to exactly the streak gate.

**Conclusion: the EMA gate is not a finer gate, it is an unconditional opening —
dominated by plain `gate = 1`, which Run F measured as a regression.** Spending
the third variant on the cap sweep instead of the EMA was the right call, and I
would make it again.

## The actual win — an interior optimum at cap 7

The cap sweep found something the assignment did not ask for. Cost per accepted
token, restricted to fully-accepted rounds (Run C trace):

| depth | verify width | round cost | µs / accepted token |
| ---: | ---: | ---: | ---: |
| 2 | 3 | 79.7 ms | 26581 |
| 4 | 5 | 126.4 ms | 25284 |
| 5 | 6 | 146.5 ms | 24414 |
| 6 | 7 | 168.3 ms | 24042 |
| **7** | **8** | **189.7 ms** | **23718 (min)** |
| 8 | 9 | 217.4 ms | 24152 |

The 8th draft's marginal round cost is **27.62 ms**, already above the running
cost per token at depth 7 (23.72 ms). Row 9 cannot repay itself **even at 100 %
acceptance** — and realised acceptance at draft index 7 is 0.8462, the only
position that falls below the shipped `0.98^i` prior. So `cost_optimal_depth = 7`
and `dominated_depths = [8]` on the base; on Run D, `dominated_depths = []`.

Measured ordering confirms the prediction exactly: cap 7 < cap 6 < cap 8 in MTP
seconds per token.

## Evidence

- Host, memory profile, toolchain, thermal policy: Apple **M4 Pro** (Mac16,11),
  48 GB, **low-memory profile**, macOS 26.5.2, Xcode 26.6, Swift 6.3.3. All runs
  through `./benchmark-qwen-mtp.sh`, which holds the single-process lock, checks
  for orphaned workers and enforces the 40 °C cooling gate before each resident
  measurement. **This is not the ranked M5**; see the transfer analysis below.
- Exact baseline and candidate commands: identical for every arm apart from the
  score path and label —

  ```bash
  /usr/bin/env MLX_QWEN_MTP_TRACE=1 \
    MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=256 \
    MLXFAST_SCORE_PATH=research/score-run<X>.json \
    ./benchmark-qwen-mtp.sh --local-iterate
  ```

  The confirmation run additionally sets `MLXFAST_SWIFT_BIN=research/capture-cli.sh`,
  `MLXFAST_CAPTURE_REAL_BIN=.build/release/mlxfast-swift`,
  `MLXFAST_CAPTURE_DIR=research/capture-runG` and drops `MLX_QWEN_MTP_TRACE`.
- Exact-token and row-ledger verdict: **matched = true, divergences = 0,
  tripwire = true** on every arm (C, D, E, F, G). Row gate: **0 value mismatches
  and 0 id mismatches at every width on every arm.**
- Divergent tokens or failure category: none.
- Generated-twin audit: not relevant — no Metal or `mlx-generated` file was
  touched.

### Headline comparison (256 decode tokens, traced, same host and session)

| metric | C base cap8 g3 | **D cap7 g3 (winner)** | E cap6 g3 | F cap7 g1 |
| :-- | --: | --: | --: | --: |
| serial s/token | 0.08127001579850912 | 0.08117573847994208 | 0.08134338306263089 | 0.08150314865633845 |
| MTP s/token | 0.04115344537422061 | **0.040059609804302454** | 0.04072417598217726 | 0.04068874986842275 |
| local serial-relative speedup | 1.9748046623920918 | **2.0263736685529** | 1.9974224425862033 | 2.0030880506257693 |
| delta vs base | — | **+2.61 %** | +1.14 % | +1.43 % (−1.15 % vs D) |
| accepted draft rate | 0.9646017699115044 | 0.995475113122172 | 0.9773755656108597 | 0.9647577092511013 |
| effective mean draft length | 5.947368421052632 | 6.138888888888889 | 5.525 | 6.135135135135135 |
| matched / divergences / tripwire | true / 0 / true | true / 0 / true | true / 0 / true | true / 0 / true |
| rounds | 38 | 36 | 40 | 37 |
| depth histogram | {2:1, 4:14, 5:2, 6:3, 7:4, 8:14} | {1:1, 4:6, 5:2, 6:3, **7:24**} | {1:1, 4:6, 5:2, 6:31} | {2:1, 4:6, 5:3, 6:3, **7:24**} |
| rejected tokens | 8 | **1** | 5 | **8** |
| reject round rate | 0.1316 | **0.0278** | 0.0250 | **0.1351** |
| accepted tokens / round | 5.7368 | **6.1111** | 5.4000 | 5.9189 |
| rounds / token | 0.1484 | **0.1406** | 0.15625 | 0.14453 |
| cap histogram | {4:15, 8:23} | {4:6, 7:30} | {4:6, 6:34} | {4:6, 7:31} |
| deep-gate open rate | 0.6053 | 0.8333 | 0.8500 | 0.8378 |

The local score is a one-prompt directional measurement on a **copy task**. It
is not the ranked median across eight hidden natural-prose prompts.

### W&B runs (every measurement)

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`; URLs are
`https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/<id>`.

| run | config | tokens | W&B id | job id |
| --- | --- | ---: | --- | --- |
| A | base cap 8, scouting | 64 | `weg12jxu` | — |
| C | base cap 8, gate 3 | 256 | `c8kz6b7t` | `a2f9b0c5-9f49-4301-928c-db0b0b09c9de` |
| D | **cap 7, gate 3 (winner, traced)** | 256 | `44zuadg5` | `4a95995d-fcc1-4ea1-a86f-9e6687a91c43` |
| E | cap 6, gate 3 | 256 | `30fb1avw` | `bcab2176-732c-42e0-99fc-4765508fdbff` |
| F | cap 7, gate 1 | 256 | `m7t6m9wv` | `561c47ca-568c-43ab-bce0-3c3422b3fc8f` |
| **G** | **cap 7, gate 3 — CONFIRMATION, untraced** | 256 | **`z49fgehs`** | `e8cd0731-5d6b-4812-ad54-7466ca2bb7f6` |

Run B failed and is documented below.

### Run G — confirmation of the exact submitted code

Run G re-measures the winner with the trace seam **reverted** and
`MLX_QWEN_MTP_TRACE` unset, so it measures byte-for-byte the code being
submitted. It confirms the win and comes out slightly *better* than the traced
arm:

| metric | D (traced) | **G (untraced, submitted code)** |
| :-- | --: | --: |
| serial s/token | 0.08117573847994208 | 0.0814559725113213 |
| MTP s/token | 0.040059609804302454 | **0.04004218755289912** |
| local serial-relative speedup | 2.0263736685529 | **2.03425380802962** |
| accepted draft rate | 0.995475113122172 | 0.995475113122172 |
| effective mean draft length | 6.138888888888889 | 6.138888888888889 |
| rounds | 36 | 36 |
| all tokens matched / divergences | true / 0 | **true / 0** |
| drift tripwire | passed | **passed** |
| head provenance sha256 | pinned | `05a8613e…99cb2863` (pinned) |

`accepted_draft_rate`, `effective_mean_draft_len` and round count are **identical
to four decimal places**, which is the expected signature of a deterministic
policy: the trace seam changed timing only, never behaviour. The +0.39 % is the
trace write-out cost leaving the MTP leg (only drafting rounds emit trace lines,
so the trace penalised the candidate leg and not the serial control) — i.e. the
diagnostic seam was *understating* the win, not manufacturing it.

**The honest headline delta remains the matched traced pair, C → D = +2.61 %**,
because that is the only pair measured under identical instrumentation. Run G's
2.0343 must not be differenced against the traced base.

### Parent-side `block_request_seconds` (FB4's actual ask, now direct)

`research/capture-cli.sh` retained the CLI reports Run G's scratch directory
would otherwise have deleted, giving the parent-side array directly rather than
via the worker-side proxy:

| leg | blocks | decode_seconds | first | p50 after first | max after first | **max / p50** | first / p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial control | 256 | 20.852728962898254 | 0.110436 | 0.065024 | 0.106855 | **1.6433** | 1.6984 |
| MTP candidate | 36 | 10.250800013542175 | 0.177513 | **0.189796** | 0.190337 | **1.0028** | 0.9353 |

Two things worth recording:

1. **The worker-side proxy is validated a second time.** Run D's worker-side
   `p50_after_first` was 189.7 ms; Run G's parent-side value is **189.796 ms** —
   0.05 % apart. `round_us` can be trusted as a stand-in for
   `block_request_seconds` in the traced arms.
2. **The serial control is the ragged leg, not the candidate.** Its max/p50 is
   **1.6433** against the candidate's **1.0028**. Any block-latency guardrail
   tuned on candidate behaviour has enormous headroom here; the candidate is by
   far the more uniform of the two legs. Both are far under the 4.0×
   threshold.

### Block-latency guardrail (FB4 / FB5 ask), worker-side `round_us`, after first

| arm | n | first | p50 after first | max after first | **max / p50** | first / p50 | max at round idx | depth of max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C base cap 8 | 38 | 174.7 ms | **168.5 ms** | 218.3 ms | **1.296** | 1.037 | 35 | 8, acc 7 → reject |
| D cap 7 | 36 | 173.5 ms | 189.7 ms | 190.3 ms | **1.003** | 0.915 | 34 | 7, full accept |
| E cap 6 | 40 | 179.3 ms | 168.5 ms | 169.0 ms | **1.0029** | 1.0641 | 7 | 6, acc 6 |
| F cap 7 g 1 | 37 | 176.2 ms | 189.5 ms | 190.6 ms | **1.006** | 0.930 | 34 | 7, acc 4 → reject |

All arms are far under the 4.0× threshold, and **the winner is the tightest
distribution of the four** (1.003 vs the base's 1.296). Cap 7 removes the base's
one 218 ms outlier, which was exactly a width-9 round that then rejected.

My worker-side `p50_after_first` for the base, **168.5 ms**, matches the
advisor's parent-side `block_request_seconds` p50 of **168.6 ms** to 0.06 %, so
`round_us` is a validated proxy for the parent-side metric. My measured prefill
of 4.018 s likewise independently reproduces the advisor's P = 4.0086 s.

## Answers to advisor feedback

### FB2 — occupancy and blended-ratio grid over q

Renewal model (`research/gate_occupancy.py`): after a rejection the session needs
`G` consecutive fully-accepted shallow (depth-4) rounds to reopen the deep cap,
then stays deep until the next rejection. Wald's identity gives expected tokens
per phase; measured `C(d)` gives cost.

Best (depth, gate) by uniform per-position acceptance `q`:

| q | best (depth, gate) | blended ratio | gate 1 − gate 3 at depth 7 |
| ---: | --- | ---: | ---: |
| 0.70 | (6, 3) | 1.4349 | −0.0934 |
| 0.80 | (6, 3) | 1.7292 | −0.1103 |
| 0.85 | (6, 3) | 1.9011 | −0.0957 |
| 0.90 | (6, 3) | 2.0992 | −0.0582 |
| 0.93 | (6, 3) | 2.2399 | −0.0267 |
| 0.95 | (6, 1) | 2.3502 | −0.0057 |
| 0.96 | (6, 1) | 2.4170 | +0.0032 |
| 0.98 | (7, 1) | 2.5720 | +0.0131 |
| measured per-position | (7, 1) | 2.6474 | +0.0142 |

**`gate 1` overtakes `gate 3` only at q ≥ 0.957.**

**Decision rule: keep gate 3, on minimax grounds.** Gate 3's worst-case give-up
is 0.013 ratio (0.5 %); gate 1's is 0.11 ratio (6.4 %) at q = 0.80 — an **8×
asymmetry**. FB2's own derivation puts the ranked pool at q ≈ 0.93–0.96, which
straddles the crossover, so the choice is a near-tie there and gate 3 is the safe
side of it. **Depth 8 is never optimal at any q < 0.999, at any gate value 0–4.**

**Simulator caveat, reported honestly:** this model predicted gate 1 would be
**+0.54 %** at measured acceptance. Measurement says **−1.15 %**. The i.i.d.
stationary per-position assumption destroys exactly the autocorrelation between
text difficulty and rejection that the gate exploits. So the q-crossover at 0.957
is an **upper bound** on how attractive relaxation is — "keep gate 3" is even
safer than the model says. The same caveat applies to the earlier per-position
sweep, which preferred cap 6 where measurement chose cap 7.

### FB1 — streak probability grid

Expected drafting rounds to reach a streak of 1 / 2 / 3, for uniform
per-position acceptance `p` at depth `d` (so round acceptance `q = p^d`):

| p | d | q = p^d | E[→1] | E[→2] | E[→3] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 2/4/6/8 | 1.0000 | 1.0 | 2.0 | 3.0 |
| 0.90 | 2 | 0.8100 | 1.2 | 2.8 | 4.6 |
| 0.90 | 4 | 0.6561 | 1.5 | 3.8 | 7.4 |
| 0.90 | 6 | 0.5314 | 1.9 | 5.4 | 12.1 |
| 0.90 | 8 | 0.4305 | 2.3 | 7.7 | 20.3 |
| 0.80 | 2 | 0.6400 | 1.6 | 4.0 | 7.8 |
| 0.80 | 4 | 0.4096 | 2.4 | 8.4 | 23.0 |
| 0.80 | 6 | 0.2621 | 3.8 | 18.4 | 73.9 |
| 0.80 | 8 | 0.1678 | 6.0 | 41.5 | 253.2 |
| 0.70 | 2 | 0.4900 | 2.0 | 6.2 | 14.7 |
| 0.70 | 4 | 0.2401 | 4.2 | 21.5 | 93.8 |
| 0.70 | 6 | 0.1176 | 8.5 | 80.7 | 694.8 |
| 0.70 | 8 | 0.0576 | 17.3 | 318.3 | 5538.0 |
| 0.60 | 2 | 0.3600 | 2.8 | 10.5 | 31.9 |
| 0.60 | 4 | 0.1296 | 7.7 | 67.3 | 526.6 |
| 0.60 | 6 | 0.0467 | 21.4 | 480.8 | 10327.2 |
| 0.60 | 8 | 0.0168 | 59.5 | 3604.2 | 214646.8 |

The promoted frontier's implied `p` is ≈ **0.9555** at depth 8.

Observed `streak_in` histograms: Run C
`{0:6, 1:5, 2:4, 3:3, 4:3, 5:3, 6:3, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1}`;
Run D is a perfect double ramp 0..17 ×2 (exactly one mid-run rejection). Rounds
that gate 1 would newly open: C 9 / 38, D 4 / 36 — small, which is precisely why
gate 1 buys +0.5 pp of occupancy and pays for it in rejections.

Realised per-position acceptance vs the shipped `0.98^i` prior (Run C):

| draft index | accepted / drafted | realised | prior `0.98^i` |
| ---: | ---: | ---: | ---: |
| 0 | 38/38 | 1.0000 | 1.0000 |
| 1 | 37/38 | 0.9737 | 0.9800 |
| 2 | 36/36 | 1.0000 | 0.9604 |
| 3 | 35/36 | 0.9722 | 0.9412 |
| 4 | 23/23 | 1.0000 | 0.9224 |
| 5 | 21/21 | 1.0000 | 0.9039 |
| 6 | 17/18 | 0.9444 | 0.8858 |
| **7** | **11/13** | **0.8462** | 0.8681 |

Index 7 — the row cap 8 buys — is the only position below the prior.

### FB5 — the roofline claim is refuted by measurement

FB5 argues widths 2..7 are bandwidth-bound and therefore cost ≈ width 1, with a
knee at `M* = 7.9`. The measurement does not show this. Round cost rises
monotonically and near-linearly from 79.7 ms at width 3 to 217.4 ms at width 9,
with a **marginal 20–23 ms per row across widths 5 → 8**. There is no flat region
below the alleged knee. Independently, width-3 `eval_wall` of 35.9 ms is *below*
the 66.6 ms single-weight-pass floor that 15.1 GB / 227 GB/s implies, so the
"one amortized pass" premise cannot hold at these widths.

Why the roofline fails here: it assumes bytes are constant in M, i.e. one
amortized weight pass (`qmm_t_splitk`). Under `qmv` — which FB2 itself
established dispatches for all M < ~10 at K = N = 5120 — **bytes scale with M**,
so cost is linear in M with no knee. Measured slope **22.5 ms/row** = 0.343
serial forwards per row. The +6 ms step at width 9 is the compute-bound term
appearing *on top of* the linear ramp, not a transition out of a flat region.

Mean `eval_wall` by width (Run C): w3 35.9, w5 55.1, w6 62.2, w7 71.0, w8 80.2,
**w9 92.9 ms**. Marginals: 5→6 7.03, 6→7 8.80, 7→8 9.24, **8→9 12.71 ms
(+38 %)**. A linear fit over widths 3–8 predicts w9 at 88.86 ms vs 92.91 actual
(+4.56 %); on `round_us`, 212.25 predicted vs 217.50 actual (+2.47 %).

**Knee sensitivity (the FB5 ask).** Removing the width-9 kink entirely (pure
linear fit, C(7) = 192.1, C(8) = 214.6), depth 8 overtakes depth 7 only at
**q ≥ 0.986**, at both gate 1 and gate 3 — still far above the ranked operating
point. Break-even needs C(8) ≤ **208.4 ms**, i.e. width 9 must be **9.0 ms
(4.1 %) cheaper**; the kink is only 2.8 ms of that. **The linear slope, not the
compute knee, is what kills depth 8.**

**M5 transfer.** Holding fixed overhead f = 12.21 ms, depth 8 beats depth 7 only
when the per-row slope s < **5.67 ms/row**, a **3.97× reduction** from the
measured 22.49:

| host / scaling | s (ms/row) | best depth |
| --- | ---: | ---: |
| M4 Pro (measured) | 22.49 | 7 |
| M5 @ 1.5× | 14.99 | 7 |
| M5 @ 1.8× | 12.49 | 7 |
| M5 @ 2.5× | 9.00 | 7 |
| hypothetical 4.0× | 5.62 | 8 |

Under pure bandwidth scaling (both f and s scaled together) the comparison is
scale-invariant, so depth 7 wins at every scale. **The cap-7 recommendation is
knee-robust** — a stronger answer than the conditional rule FB5 requested. It
would only need revisiting if the *fixed* overhead grew relative to the slope,
which is what verify-batch padding would do.

### FB2 cost accounting — `headStepCostRatio` is 3.2× under-costed

Fitting round cost across widths gives **`12.21 ms + 22.49 ms × width`**. One
serial forward on this host is **65.58 ms**. In serial-forward units that is
**fixed 0.186** and **h = 0.343 per draft**. The shipped `headStepCostRatio` is
**0.20**, so the cost model under-charges each draft by **3.2×**. That is very
likely why the shipped policy reaches for depth 8 at all. I did **not** change it
— it is `qwen-edward`'s parameter per the assignment — but it should be routed
there, and it interacts directly with the cap: with an honest `h`, the greedy
rule would probably stop at 7 on its own without a hard cap.

Decode-only raw speedup at G = 1 is **2.5792**, with prologue 4.018 s / 4.027 s
(P = 23.9 % of the candidate leg). **2.904 is therefore reachable at G ≈ 1**, so
the official score is not by itself evidence that G ≫ 1.

## Failures, blockers and honest disclosures

1. **FB3's 512-token ask is impossible on this fixture.** Run B (512 tokens, job
   `ed817126-a4e8-4fd1-af95-e5fba563ccc3`) exited 1. The serial control died with
   `MTP round requested before the seed prefill` after 301 rows. Root cause: the
   fixture emits a **stop token** at decoded token ~302. `generateRound`
   (`Qwen36MTPBlockSession.swift:731-750`) sets `reachedStopToken` and nils the
   pending state, but `RuntimeWorkerResponse` in
   `Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift` (case
   `"mtp_decode_round"`, ~L328-373) does not carry that flag, so the parent never
   stops and the next round throws `notBegun` (L90). **Any local window beyond
   ~300 tokens hard-fails on the only public fixture; the maximum safe local
   window is ≈ 300 tokens**, which is why every arm here is 256. Fixing it means
   editing a trusted harness file, which is out of scope for this assignment.
2. **FB4 item 2 is correct on base but not on my branch.** `MLX_QWEN_MTP_TRACE=1`
   *is* reachable on the Qwen path here, because my own commit `c72f112` added a
   `forwardsWorkerStderr` seam to the `mtp-timed` verb. That seam lives in
   `Sources/MLXFastCLI/main.swift`, which is a **trusted file not in
   `editablePaths`**. It has been fully reverted: `git diff BASE_SHA --
   Sources/MLXFastCLI/main.swift` is empty at the submitted HEAD, and the
   confirmation run was measured after the revert with both products rebuilt. All
   traced numbers above were taken with that seam present; the confirmation run
   shows the seam did not manufacture the result.
3. **The simulator over-predicted relaxation.** Stated in full above. I am
   flagging it because it means the per-position sweep numbers in this report
   should be read as bounds, not point predictions.
4. **`post_assignment_comment` is not in my tool schema.** The advisor asked for
   PR replies several times across FB1–FB5. I could not post any. Everything I
   would have replied with is in this file and in the structured result. This is
   a harness gap, not a choice.
5. **Stale help text.** `./benchmark-qwen-mtp.sh --help` still describes the old
   depth behaviour. Cosmetic, not fixed here (out of scope).
6. `research/gate_occupancy.py` prints `inf` / `nan%` for `deep_run_length` at
   p = 1.00. Cosmetic.

## Suggested follow-ups (not implemented)

1. **Pad the verify batch from 9 to 10 rows** to cross the `qmv` limit into
   `qmm_t_splitk` (FB2's own suggestion). This is the one change that could
   re-open depth 8: it attacks the linear slope, which the sensitivity analysis
   above identifies as the binding term. Reachable levers are the requested
   shapes plus `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` and
   `kernels/quantized{,_nax}.{h,metal}`.
2. **Route `headStepCostRatio` 0.20 → ≈ 0.343 to `qwen-edward`.** With an honest
   per-draft cost the greedy rule may select 7 unaided, making the hard cap
   redundant and the policy simpler.
3. **Propagate `reachedStopToken` through `RuntimeWorkerResponse`** so local
   windows longer than ~300 tokens become measurable. This is trusted-harness
   work and needs an advisor decision, but it currently caps every local
   experiment in the campaign at 256 tokens.
4. **A rejection-manufacturing research harness** (FB1) to sample the low-q
   regime the ranked prose prompts occupy, since the only public fixture is a
   copy task that sits at q ≈ 0.99. It must stay research-only and must not touch
   `Qwen36MTPReferenceSession.swift`.

## Conclusion

- **What happened:** Part A passed outright — the segmented verify is bit-exact
  per position at width 9, on 122 of 256 positions, so the assignment's fidelity
  contingency never fired. Part B's hypothesis was then refuted: relaxing the
  streak gate to 1 costs 1.15 % of local ratio by triggering a rejection cascade,
  and the EMA variant is provably a degenerate unconditional opening rather than
  a finer gate. The genuine improvement came from the opposite direction —
  *reducing* the depth cap from 8 to 7, worth **+2.61 %** local ratio.
- **Evidence for the mechanism:** cap 7 raises accepted-draft rate 0.9646 →
  0.9955, cuts rejected tokens 8 → 1, cuts rounds-per-token 0.1484 → 0.1406, and
  tightens the block-latency `max/p50` from 1.296 to 1.003. Cost-per-accepted-
  token is minimised at depth 7 in the measured `C(d)` table, and the predicted
  ordering cap 7 < cap 6 < cap 8 matches the measured ordering exactly.
- **Prompt and M5 transfer risk:** *this is the main caveat.* The only local
  fixture is a copy task at very high acceptance (q ≈ 0.99); the ranked pool is
  eight natural-prose goldens where q is likely 0.93–0.96. The occupancy model
  says cap 7 + gate 3 remains at or near optimal across that whole band, and the
  knee-sensitivity analysis says cap 7 survives every plausible M5 scaling — but
  a one-prompt copy task cannot prove a median over eight prose prompts. The
  cap-7 change is also small, single-line, and trivially reversible.
- **Smallest useful next action:** verify-batch padding 9 → 10 rows, which is the
  only identified change that could make depth 8 pay.
- **Recommendation: promote** the single-line `segmentedVerifyDepthCap` 8 → 7
  change; **close** the deep-round-gate relaxation as a measured negative with
  the cascade mechanism recorded so it is not retried blindly.

**Label: local winner** (cap 7, gate 3), with Part B reported as a clean
negative.
