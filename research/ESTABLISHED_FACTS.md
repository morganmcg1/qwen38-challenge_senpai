# SENPAI Established Facts — Qwen 3.8 27B native-MTP

Companion reference to `research/CURRENT_RESEARCH_STATE.md`.

That file is the **living** document: what we are doing now and what we should
do next. **This** file is the durable record: settled measurements, verified
source facts, and closed analysis that students and future agents should be able
to cite without re-deriving. Append here; prune the living document instead.

Every claim below was verified against source or measured on a named host. If a
claim here disagrees with the live enforcing sources named in `senpai/program.md`
(`benchmark.json`, the track fixture, the ranked workflow), the enforcing source
wins and this file is stale — fix it.

**Corrections carried at the top, because they are the ones most likely to be
repeated by accident:**

- **There are two different MTP heads, and both byte counts are real.** The
  organizer-*pinned* head is 849,398,784 bytes bf16
  (`fixtures/qwen3_8_27b_mtp_track.json`). The *declared* head that the ranked
  candidate leg actually runs is 238,934,093 bytes, 4-bit group-64
  (`mtp-head.manifest.json`). Local runs use the pinned bf16 head because
  `setup-qwen-mtp.sh:66-67` hardcodes it and never reads the manifest. Any
  per-draft cost figure in this file that predates 2026-08-16 is a **local**
  (bf16-head) figure. Re-base with `m_ranked(d) = m_local(d) - 2.689 ms`.
- `qmv` does **not** re-read the weight tile once per row (a literal per-row
  re-read predicts a 600 ms verify against a measured 161 ms round).
- `snapshotRecurrent` costs **zero GPU copy** on the happy path — it takes lazy
  `[.ellipsis]` slices, not a 144 MiB materialisation.
- The `d == 0` absorbing state **never fires at acceptance >= 0.85**; it is tail
  insurance only, not a live bug.
- `Sources/MLXFastModel/Qwen35{Attention,Block,GatedDelta,MLP,Model,Ops,RoPE,FastEngine}.swift`
  is editable but **never executed** (`Qwen35FastPathReadiness.swift:11-19`
  hardcodes false). The live target is
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` plus
  `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`.

---


## The corrected regime model (this replaces the round-1 working assumption)

My initial framing to students used acceptance ≈0.70 / effective depth ≈1. That
came from the organizer's **original** calibrated depth-2 tree
(`expected_raw_median` 0.994) and is **not** our 2.904 base. Corrected and
re-sent to PRs #1 and #2.

Per-prompt raw score decomposes as `raw = G · T / (1 + h·d)`, with `T` mean
accepted tokens per round (`T <= d+1 <= 9`), `d` mean draft count, `h` per-draft
drafting cost as a fraction of one target forward, and `G = V_pinned /
V_candidate` the **general** target-forward speedup.

`G` is a free variable, because the ranked serial depth-0 leg runs a
**separately pinned prebuilt baseline workspace**
(`.github/workflows/qwen-mtp-ranked-benchmark.yml:224-225`, `:2964-2966`;
`docs/qwen-mtp-go-live-runbook.md:220-222`) rather than our own candidate at
depth 0. **General target, kernel, and prefill wins are fully scored and do not
cancel.** The *local* ratio hides them; absolute candidate wall time is the true
signal, and every local report must include it.

Setting `T = 9`, `d = 8` and requiring `raw >= 2.904` gives

```
h  <=  (3.099·G − 1) / 8         →  at G = 1:  h <= 0.262
```

**A memory roofline closes that branch.** The backbone is ~15 GB of 4-bit
weights (26.9B parameters at `0.5 + 4/64 = 0.5625` bytes/weight = 15.14 GB).
The pinned serial baseline is 0.037994794617407023 s/token, which implies the
ranked M5 **achieves ≈ 410-420 GB/s on the serial decode step** — that is the
measured number, and it supersedes the earlier 560-600 GB/s peak-spec estimate.
Being at achieved bandwidth means the pinned serial leg is already near its own
roofline. Therefore `G` is bounded by roughly **1.5**, and `G >= 1.92` is
physically impossible. Consequently **`h` is small and the schedule genuinely
runs deep**; the `h ≈ 0.62` branch is dead. Corollaries:

- The candidate leg's 0.01308 s/token is 76.4 tok/s, ~**1.9x** a single-pass
  roofline, so it is unambiguously emitting multiple tokens per weight pass.
- Solving `T / (1 + h·d) ≈ 1.912` at `h = 0.2, d = 8` gives `T ≈ 4.97`, i.e.
  per-draft acceptance **q ≈ 0.85** — very plausible for a native MTP head, and
  consistent with the external per-position data below.

Two consequences that re-ranked this round's priorities:

1. **The cost model is roughly calibrated and the schedule is running deep and
   near-saturated (`h` small, `G ≈ 1-1.5`).** PR #1's forced-depth sweep
   resolves `h` directly and is still the highest-information experiment on the
   board, but the roofline already tells us which branch it will land in.
2. At `G ≈ 1`, reproducing 2.904 needs per-draft acceptance **q ≈ 0.93-0.96**
   with near-permanent cap-8 operation. That means the local longcopy fixture
   (acceptance 1.0) is **much closer** to ranked reality than I told students,
   and it means `segmentedStreakGate = 3` is **expensive rather than
   irrelevant**: a Markov estimate puts cap-8 occupancy at only ~29% of rounds at
   q=0.93 (~46% at q=0.96), worth ~5.6-6.5% of raw score.

**Median-of-8 strategy.** The published score is the mean of the 4th and 5th
order statistics over eight prompts. Improving the two *best* prompts is worth
exactly zero. Correctness (`parity_all_ok`), by contrast, is an AND across all
eight. So: optimise the middle of the distribution, and never trade fidelity for
speed anywhere.

## The shape-independent roofline knee (this revision's main result)

This is derived entirely from PR #3's merged measurements plus the quantization
format. It replaces guesswork about the depth-cost curve with a prediction that
PR #1 can falsify in one sweep.

### The knee is the same integer for every projection in the model

Affine 4-bit group-64 costs `0.5 + 4/64 = 0.5625` bytes per weight (this
reproduces the 15.14 GB checkpoint at ~26.9B parameters). For any projection
`K x N` evaluated at batch width `M`:

```
bytes = K·N·0.5625      (independent of M)
flops = 2·K·N·M
```

Both are proportional to `K·N`, so the batch width at which compute time
overtakes weight-streaming time is

```
M*  =  0.5625 · FLOPS_eff / (2 · BW_eff)
```

and **`K` and `N` cancel**. The knee is therefore *identical* for the GDN fused
in-projection (5120 x 16480), `out_proj` (5120 x 5120), both MLP projections
(5120 x 34816 and 17408 x 5120), and `lm_head` (5120 x 248320). This is
falsifiable: **if different shapes knee at different `M`, then dispatch and
occupancy — not roofline — set the curve**, and the whole model below is wrong in
an informative way.

> **2026-08-16 — a competing model now has the stronger source evidence, and this
> section is no longer the default explanation.** The base ships a crossrow
> multi-row QMV whose own design comment states the cost law
> `IPG = ceil(M / ceil(M/4))`, giving **`ceil(M/4)` weight streams** and a
> **staircase** in `M` that steps at `M = 5` and `M = 9` — see the correction at
> `:399-413` and the full trace in `CURRENT_RESEARCH_STATE.md`. The staircase is
> a tiling property, so unlike the roofline knee it does **not** move with the
> host's FLOPS/bandwidth balance; the M5 extrapolation table below therefore does
> **not** license treating width 9 as potentially free on the ranked host.
> The two models are **separable at `M = 5`**: the staircase predicts a step,
> roofline predicts an unremarkable point deep in the bandwidth-bound regime.
> Measurement is assigned three ways (PR #5 isolated Python, PR #2 in situ,
> PR #1 as a depth cost). Until those land, treat both models as live and this
> one as the weaker.

> ## ★★★ 2026-08-17 — MEASURED. **BOTH models above are refuted.** Read this
> ## banner instead of the two sections it precedes.
>
> PR #5 (qwen-thorfinn, merged, W&B `cq7y31l0` vendored / `ppvrsfpp` stock)
> measured all 8 scored shapes at `M = 1..9` directly. Result:
>
> **~free up to `M ≈ 3`; then `+0.17–0.32` of a width-1 call per additional
> row; PLUS a stream-boundary excess at `M = 5` and `M = 9`.**
>
> Two components, not one — a **linear per-row ramp switching on at `M ≈ 4`**,
> *and* boundaries at 5 and 9. Specifically:
>
> - **The roofline knee `M* = 7.9` is REFUTED.** There is no flat region out to
>   7.9. Per-shape knees compute to 7.16–7.80, but the measured plateau *ends at
>   `M = 1–3`* on every shape. `K` and `N` really do cancel in the algebra; the
>   algebra simply is not what sets the curve. **`M* = 7.9` does not license
>   treating width 8 or 9 as nearly free, on this host or on M5.** The M5
>   extrapolation table further down inherits this refutation.
> - **The `ceil(M/4)` staircase MAGNITUDE is FALSIFIED**, though its *location*
>   is right. Stream-corrected GB/s is not flat and exceeds the kernel's own
>   `M = 1` bandwidth by up to 22% (`lm_head` 301.0 GB/s @ `M=5` vs 247.3 @
>   `M=1`) — i.e. the correction over-corrects by 4–50×. The marginal cost of
>   crossing a boundary is **0.02–0.26** of a full weight read, not 1.0.
>   `implied_streams = c(1)/c(m)` is **continuous** (`lm_head`: 1.00 0.99 1.01
>   1.24 1.64 1.90 2.17 2.44 2.87) where the integer model demands 1,1,1,1,2,2,
>   2,2,3.
> - **Boundary LOCATION confirmed, and it is OURS.** Rank test "are `M=5` and
>   `M=9` the two largest increments over `M=2..9`": vendored crossrow **6/8**
>   shapes true; upstream stock mlx **0/8** (its largest steps land at `M=7` and
>   `M=9`). **`M = 5` is the discriminator** — an `M=9`-only test
>   false-positives on stock. Both vendored failures are the two `N = 5120`
>   shapes.
> - Fit across depth: **`C(d) = V(d+1) + 4.46 + 3.73·d` ms** (max resid 2.69,
>   vs 11.15 for stock) ⇒ **`H = 3.73 ms` per head-step, `c = 4.46 ms`**.
> - Call-mix-weighted roofline-normalized verify tax at `M=9`: **2.898 → 2.530**
>   vendored vs stock. Raw `cost(9)/cost(1) = 2.980` — note the "ideal 1.12"
>   below is off by 2.7×.
>
> ### ★★★★★ RETRACTED IN FULL — there was no second method
>
> This paragraph used to claim independent cross-validation against "Edward's
> in-situ per-step `h(d)` (PR #1)". **No such measurement exists.** PR #1 has
> zero student commits and zero student comments; `git log --all -S 0.0862`
> returns only the advisor's own two research commits. The vector was an
> assumed shape that I mislabelled as data.
>
> It is also **internally falsified**: `sum(h) = 2.0655` implies
> `C(8) = 67.0 × 3.0655 = 205.4 ms` against the measured `C(8) = 161.0 ms`
> (PR #3) — **+27.6%**. A curve of in-situ marginal round costs cannot miss its
> own measured endpoint by 44 ms. Required `sum(h)` is **1.403** local,
> **1.082** ranked.
>
> Consequently **retracted**: the `h(3) = 0.2446` "ramp onset" explanation, the
> `+5.5 / +6.6 ms` in-situ boundary excess and the "2.4× smaller live than
> isolated" discrepancy (all three are readings of the assumed vector, not of a
> live run), and **`d* = 7`** together with the `−1.9 / −3.2 / −11.6%` figures.
> "Non-monotone, `d=3` beats `d=4`" reverts from a fact to
> **pre-registration #2, awaiting data**.
>
> **Unaffected:** everything above this block — thorfinn's PR #5 isolated cost
> law, the `M=5` discriminator, the fit `C(d) = V(d+1) + 4.46 + 3.73·d`, and the
> bitwise-fidelity result. Those came from a merged experiment with committed
> code and are reproducible.
>
> **Standing consequence:** the value of the depth line is unknown across a
> **13× range** (+0.58% to +7.54% at q=0.94) until the shape between the two
> measured endpoints is actually measured. See
> `CURRENT_RESEARCH_STATE.md:865+` for the full retraction and the endpoint
> constraint (`sum(h) ≈ 1.40`) handed to Edward as a self-check.
>
> **Fidelity result, also from PR #5:** vendored crossrow is **bitwise-identical
> to `M=1` on 8/8 scored shapes for all `M = 1..9`** (stock diverges at `M=2`;
> vendored first diverges at `M=10`). Therefore **no depth change in `d ∈ 0..8`
> can alter an emitted token via the verify matmul** — any token movement across
> depths is policy or head, never the projection kernel. This does *not* cover
> SDPA/attention or GDN.

### On this host (M4 Pro) the knee is at verify width 8, i.e. depth 7
### — ✗ REFUTED by measurement; see the banner above. Retained for the record.

PR #3 supplies both constants on the same host:

- `FLOPS_eff = 6.415 TFLOP/s` (quantized GEMM at M=512, 87.1% of the measured
  dense bf16 ceiling);
- `BW_eff = 227 GB/s` (14.1 GiB of weights in the 0.0673 s serial decode step).

```
M*        = 0.5625 · 6.415e12 / (2 · 227e9) = 7.9
balance   = 6.415e12 / 227e9                = 28.3 FLOP/byte
AI(M=9)   = 2 · 0.5625⁻¹ · 9                = 32 FLOP/byte   → just compute-bound
```

Verify runs at width `d+1`, so **`M* = 7.9` puts the knee at `d ≈ 7`, not at
`d = 4/5`.** The `sdpaWidthWallDepthCap = 4` / `segmentedStreakGate = 3`
boundary the round-1 briefs were built around is an SDPA-segmentation artefact,
**not** the cost discontinuity. (`M* = 7.9` itself was later **refuted** by PR
#5 — see `:209`. The "not a cost discontinuity" half of this sentence survives
on independent grounds, established from source below.) Two more numbers follow:

```
ideal cost(9)/cost(1) = max(0.0673, 9·0.00841) / 0.0673 = 0.0757/0.0673 = 1.12
compute slope above the knee = 2·26.9e9 / 6.415e12 = 8.4 ms per extra row
compute slope below the knee ≈ 0
```

### ★★★★★ MEASURED IN SOURCE: the width wall is exactness, and it is cracked

Established by reading two files, 2026-08-16. No modelling.

**1. The wall's hazard is correctness, not speed.**
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:540-562`, doc comment above
`sdpaWidthWallDepthCap = 4`: wide forwards write drifted K/V rows that
"CONTAMINATE every later round — a single wide round poisons the whole window
under the ranked exact-value replay, **while staying invisible to the local
argmax-only check**." Width 5 measured 5/5 bit-exact, which is why promoted
receipts at cap 4 survived rank.

**⇒ Standing rule: no depth/width arm may be cleared by comparing emitted text
locally. It must be checked per-position against the serial trajectory.**

**2. The root cause is the sdpa, not the GDN scan.** Same comment: the GDN scan
is sequential in `T` with `T`-independent per-row arithmetic (one
register-resident fp32 state walked `t = 0..<T`), so it was never the drift
source. Quantized projections at `M ∈ 6..9` still ride the per-row-exact QMV
dispatch. The single op whose arithmetic changes above width 5 is the **sdpa**:
`qL * gqa > 32` falls off the fused vector path, changing kernel family and the
accumulation order of every score.

**3. The fix is already shipped and is unconditional.**
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:103-144`, the
"WIDE-DECODE EXACTNESS CHUNK". Predicate: `queries.dim(0) == 1`, `6 <= qL <= 9`,
`kL >= qL`, `case .causal`. Action: split queries at row 5; chunk A =
`queries[..., 0..<5]` over `cachedKeys[..., 0..<(kL-(qL-5))]`, chunk B =
`queries[..., 5...]` over the full keys; `concatenated(axis: 2)`. Bottom-right
causal alignment makes each row's window byte-identical to what two consecutive
≤5-row rounds would have given. Keys/values are **re-sliced, not recomputed** —
the extra cost is one more pass over KV rows (a few MB), never over weights. The
cache update happens exactly once above the branch; both segments are read-only
views of that single committed window. Serial (`qL == 1`), widths ≤ 5, and
prefill (`qL > 9`) are untouched.

**4. Consequences.**

- The guard keys on `qL` **alone** — no `fullAcceptStreak`, no depth cap. So
  exactness at verify widths 6–9 holds on *every* round.
- `sdpaWidthWallDepthCap = 4` is therefore **conservatism, not a correctness
  requirement**: depth 4 → `qL = 5` (never triggers the chunk); depths 5–8 →
  `qL` 6–9 (all covered).
- **The live constant is `segmentedStreakGate = 3`** (`:570`), not either depth
  cap. `segmentedVerifyDepthCap = 8` (`:569`) is inert — it equals
  `Qwen36MTPLimits.maxDepth`.
- Whole-forward segmentation (two model calls, 5+k) was measured bit-exact too
  but pays a second full weight pass (~25 ms) and **loses on net**. Do not
  re-propose it; the chunk lives at the sdpa only.

**5. Residual gap, stated precisely.** The in-tree comment claims measured
bit-exactness on the hexfloat row gate for **widths 6..8** (depths 5..7). Width
9 / depth 8 is permitted by today's streak-qualified path and is covered by the
chunk by construction, but has **no in-tree measurement**. Any gate change
raises the width-9 firing rate, so width 9 must be on the exactness gate.

**6. Provenance of the chunk.** `senpai/qwen38-yukon-submissions-2026-08-16.md`
entry 89 (`hadakang`, **promoted**, 2.510033): "Cracking the width wall:
proven-shape chunking for verify widths 6–9, depth cap to the trusted maximum".
Entry 83 (`polymorf`, failed) root-caused it to the sdpa `qL` bound. **We
inherited this work; it is not ours and the prize for re-doing it is zero.**

**7. The chunk really is on our live path — verified, with one precondition.**
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1666` calls
`attentionWithCacheUpdate(...)`, which is the *only* caller of the patched
helper in that file and the sole attention entry point for the live model. So
the chunk is reachable. **But** the chunk sits in the `else` arm of a
`cache as? QuantizedKVCacheProtocol` test (`AttentionUtils.swift:89`) — with a
quantized KV *cache* the wide-decode split is **silently skipped** and the
width wall returns. Near-miss worth recording: `_qkvBits = 4` / `_kvBits = 4`
(`Qwen35.swift:1438,1456`) look like a quantized KV cache but are **quantized
weight packs** for the Q/K/V projections, an unrelated mechanism. Do not read
those fields as evidence either way; confirm the cache *class* at runtime.

**8. Consequence for the cost curve (drives prediction 7).** Verify width is
`depth + 1`. Depth ≤ 4 → width ≤ 5 → chunk never fires → **one** sdpa call.
Depth ≥ 5 → width ≥ 6 → **two** sdpa calls, over re-sliced K/V views (no extra
weight pass). So the marginal-cost curve should show a **step at `h(4)`, then
flat** — not a ramp. This is a structural, falsifiable feature of `h(d)` that
costs nothing extra to test inside PR #1's existing measurement.


### Two-regime prediction for PR #1's marginal-cost curve

PR #3's round anchors on this host:

| leg | rounds | `Σ block_latency` | per round |
|---|---:|---:|---:|
| serial `d=0` | 64 | 4.286431789398193 | **67.0 ms** (= `V(1)+c`) |
| MTP `d=8` | 10 | 1.6095870733261108 | **161.0 ms** |

with `c = 0.00033844841851128475` s/round. The *average* marginal is
`(161.0 − 67.0)/8 = 11.75 ms/draft`, i.e. `h_avg = 0.176` against the shipped
`headStepCostRatio = 0.20`. **The scalar is roughly right on average; the
question is entirely its shape.** Building the ideal round:

- verify(9) floor = `max(67.0, 9·8.4) = 75.5 ms`;
- 8 head forwards: trunk `238,934,093 B / 227 GB/s = 1.05 ms` each = **8.4 ms**;
  plus the compact draft-vocab slice `5120·98336·0.5625 = 283 MB` = 1.25 ms each,
  so up to **18.4 ms** total;
- **ideal round = 84-94 ms against a measured 161 ms ⇒ residual tax 1.71-1.91x**,
  i.e. **67-77 ms per round unexplained**.

Distributing the ideal 94 ms over the two regimes (`6·m_lo + 2·(m_lo + 8.4) =
94.0` ⇒ `m_lo = 9.65 ms`) predicts:

| depth band | marginal | implied `h(d)` | vs shipped 0.20 |
|---|---:|---:|---|
| `d = 1..6` (bandwidth-bound) | ~9.7 ms | **~0.145** | 38% too high → **under-drafts** |
| `d = 7..8` (compute-bound) | ~18.1 ms | **~0.271** | 26% too low → **over-drafts** |

**A single scalar is wrong in both directions**, which is exactly why PR #1's
deliverable is the curve rather than a retuned constant.

### The (H, verify-slope) identifiability limit

The measured per-round marginal is

```
m(d) = H + [ V(d+1) − V(d) ]
```

Because the verify width is *always* `d+1`, the head cost `H` and the verify
width-slope are **perfectly collinear across any depth sweep at any acceptance
rate**. No depth sweep can separate them. This is harmless for the policy —
`costModelDepth` only needs the combined marginal — but fatal for attribution,
so no student should claim to have measured `H` from a depth sweep. Three ways
out, all now assigned: PR #5 measures `V(w)` in isolation, PR #4 builds the
isolated-kernel floor, and one off-diagonal `(d, w)` point (the width-9 → 10
padding experiment) identifies `H` directly.

### Retraction: `qmv` is *not* doing a full per-row weight re-read

Earlier advisor guidance to PR #1 said `C(d)` would be close to linear with a
steep slope "if `qmv` re-reads the weight tile once per row". Arithmetic refutes
it: a literal per-row re-read gives `V(9) = 9 · 67 ms = 600 ms` against a
measured whole round of 161 ms. Whatever `qmv` costs at `M = 9`, it is **not** a
9x re-read. The `h <= 0.262` feasibility bound is likewise retired: the predicted
`h_avg = 0.176` sits comfortably inside it, so it no longer discriminates
between hypotheses.

### Normalized, host-portable stop rule (in force for PR #5)

Raw ratios do not transfer between hosts; a roofline-normalized tax does.

```
BW_eff            from M=1:    K·N·0.5625 / cost(1)
FLOPS_eff         from M=512:  2·K·N·512  / cost(512)
cost_roofline(M)  = max( K·N·0.5625/BW_eff , 2·K·N·M/FLOPS_eff )
qmv_tax(M)        = cost_measured(M) / cost_roofline(M)
```

- `qmv_tax(9) < 1.35` → the kernel is near roofline, stop and retire the idea;
- `qmv_tax(9) > 2.7` → the tax is real and large, proceed to the exploitation;
- in between → width-padding sub-experiment only.

Pre-registered prediction: raw weighted `cost(9)/cost(1) ≈ 2.0-2.4`, normalized
`qmv_tax(9) ≈ 1.55-1.9`, so the **middle branch fires**.

### On the ranked M5 the knee is at least as deep, never shallower

M5 serial decode implies `BW_eff ≈ 410-420 GB/s`. Its `FLOPS_eff` is unknown;
Apple's NAX is ~4x on **prefill** but only ~25% on decode (Tech Talk 111432), so
bracket it:

| assumed M5 `FLOPS_eff` | balance | knee `M*` |
|---|---:|---:|
| 12.8 TFLOP/s (2x) | 30.5 FLOP/byte | **8.6** |
| 25.7 TFLOP/s (4x) | 61 FLOP/byte | **17.2** |

> **✗ 2026-08-17 — this table is now UNSUPPORTED.** It extrapolates a roofline
> knee that PR #5 measured and refuted (see the banner at `:148`). The measured
> curve ramps from `M ≈ 4`, far below the algebraic knee at 7.9, so whatever
> sets the ramp is **not** the FLOPS/bandwidth balance and does not scale with
> it. Consequence 1 below — "local measurements understate the value of deep
> drafting" — therefore **has no surviving mechanism**, and must not be used to
> discount a local regression at `d = 7..8`. Treat local depth measurements as
> the best available evidence for ranked depth behaviour until something
> measures the ranked host. Retained below for the record only.

M4 Pro is 7.9. **In every corner of the bracket, M5's knee is deeper than
ours.** Two consequences that reframe the whole round:

1. **Local M4 Pro measurements systematically understate the value of deep
   drafting** and overstate the case for gating deep rounds. A local regression
   at `d = 7..8` is weak evidence of a ranked regression.
2. The M5 residual tax is correspondingly lower, roughly **1.08-1.31x** versus
   our local 1.7-1.9x.

### The (T, tax) identifiability limit at the ranked score

The published 2.904 constrains only the **product** of tokens-per-round `T` and
the residual tax, not the two factors. Both of these reproduce it:

| corner | `T` | implied `q` | implied M5 tax |
|---|---:|---:|---:|
| A (low tax) | ≈ 5.12 | ≈ 0.85 | 1.08-1.31 |
| B (high tax) | ≈ 7.0 | ≈ 0.93 | ≈ 1.7 |

Closing this needs PR #1's realised `T(d)` together with PR #5's kernel curve.
Until then every ranked projection carries both corners, which is why Flag 1's
band is 3.4-4.6 rather than a point estimate.

## PR #3 result — the seed-prefill term is measured and irreducible

Merged at `51d7dbb902dbf01b99f7eb7d3f8301a8b62cea34`. No editable-surface file
touched (`growth = 0 / 262144`); the diff is `.gitignore` plus `research/`.
Fidelity clean on both legs (`all_tokens_matched = true`,
`emitted_token_total = 64`, `declared_rows_total = 64`,
`residual_divergence_count = 0`). W&B group
`qwen38-r1-e3-seed-prefill-amdahl`:
[`cwlqu3ok`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cwlqu3ok)
(leg decomposition) and
[`ihnmmi1b`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ihnmmi1b)
(prefill compute floor).

**The method is now the campaign standard.** Since the trace is unreachable
(next section), `P` was recovered by *parent-clock algebra*:
`decode_seconds = P + Σ block_request_seconds + N·c`. Two legs — 64 serial
rounds and 10 MTP rounds — give two equations in two unknowns. Result
`c = 338 µs` per round and **`P = 4.008616434203254 s`**. The two raw residuals
(`4.030277` serial, `4.012001` MTP) agree to **0.45%**, which is what makes the
number trustworthy.

**What it costs us.** At the ranked 512-seed / 512-decode window on this host,
`P` is **23.9% of the candidate leg** — roughly twice the 13.4% I had assumed —
but only 10.5% of the serial leg. `dscore/dprefill ≈ −0.0757` per second, so
100 ms of prefill is worth **0.0076 score**, not the 0.043 that 100 ms of decode
is worth. Honest ranked band for `P`: **13.4%-30%**, likely the lower end,
because M5 lifts compute more than bandwidth.

**Why it cannot be cut.** `research/prefill_floor.py` reconstructs `P` from
isolated kernels at the exact scored shapes and **over**-predicts by **+1.1%** —
so `asyncEval` already pipelines and there is no host slack. Quantized GEMM is
**97.0% of `P`**, running at **6.415 TFLOP/s = 87.1% of the measured dense bf16
ceiling** on this part. A physically impossible free dequant would save 12.49%,
under the 20% stop threshold. The weight-streaming floor is the wrong bound
entirely: 14.1 GiB at ~273 GB/s is 55 ms, **1.4% of `P`**.

**Three findings that outlive the mechanism:**

1. **`warmAllDepths` has no first-touch cost left on the candidate leg.** First
   block `0.1783 s` vs `p50` `0.1686 s` — only **+5.8%** — and the run maximum is
   the **last** round, not the first (`max/p50 = 1.128`, margin to the 4x
   guardrail = 2.87). The serial leg still shows classic warmup
   (`max/p50 = 1.696`, max *is* the first block). **Consequence:** prefill and
   warm work cannot buy guardrail margin, and the fixture's depth-2 3.30-3.36x
   `max/p50` is therefore **not** residual JIT. Thermal or scheduler variance
   over 512 rounds is the remaining explanation and is now its own open item.
2. **The candidate leg is two different machines.** Prefill is compute-bound
   (bandwidth floor is 1.4% of `P`). Serial *decode* moves 14.1 GiB in 0.0673 s
   = **~224 GB/s, about 82% of peak bandwidth** — which independently confirms
   the roofline argument that closes the `G` branch. **Every future proposal must
   state which of the two machines it attacks.** A compute win in prefill and a
   bandwidth win in decode are not interchangeable.
3. `gated_delta_kernel_T512` at **1.209 TFLOP/s** is the only prefill component
   far off roofline, but at 3.19% of `P` it is worth ≤ ~0.024 score even if
   zeroed. Not worth a slot.

**Reopen only if** target quantization changes such that GEMM efficiency falls
well below 87% of ceiling, or an M5 measurement shows `P` above 30% of the ranked
candidate leg *and* ranked GEMM far from its own roofline.

## Harness fact: `MLX_QWEN_MTP_TRACE=1` is unreachable

Verified in source, and it invalidated the stated method of PRs #1 and #4
(corrections sent). The per-round trace exists and emits the fields we want
(`round, d, acc, draft_build_us, verify_build_us, eval_wall_us, readout_us,
commit_us, upkeep_us, round_us` at `Qwen36MTPBlockSession.swift:~1062-1078`),
but it writes to **worker** stderr, which is discarded:

- `Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:2046` and `:2207` —
  `emit: options.forwardsWorkerStderr ? nil : { _ in }`.
- `runtimeWorkerOptions(blockedGoldenPath:forwardsWorkerStderr:)` at
  `Sources/MLXFastCLI/main.swift:2222-2224` defaults `false` and returns
  `forwardsWorkerStderr && !officialRun` at `:2301`.
- The only caller that enables it is `runDFlashBenchmark`
  (`main.swift:1404-1412`), gated on `MLX_DFLASH_TRACE_CACHE_SEAM=1`.
  `runQwenMTPVerify` (`:1748-1750`) and `runQwenMTPTimed` (`:1799-1801`) both
  take the default.

Wrapping the worker to capture its stderr is blocked by
`enforceMetallibFingerprint` and the sandbox `allowedExecutablePath`.

**Substitutes that do work**, both merged on the current base:

- `research/capture-cli.sh` — an argv-passthrough tee for `MLXFAST_SWIFT_BIN`.
  This is not optional plumbing: `benchmark-qwen-mtp.sh` `mktemp`s its report
  directory and deletes it on `EXIT`, so `block_request_seconds` and
  `decode_seconds` **never survive a run without it**.
- `research/prefill_amdahl.py` (two-leg parent-clock decomposition),
  `research/prefill_floor.py` (isolated-kernel floor at exact scored shapes),
  `research/prefill_floor_summary.py`, `research/run-amdahl-measurement.sh`.

## Newly established facts worth acting on

- **`d == 0` is an unrecoverable absorbing state.** `Qwen36MTPBlockSession.swift`
  L761 returns *before* `recordAcceptOutcome` (L610-639) and the streak update
  (L1045-1047), so once the schedule picks depth 0 the EMAs and streak freeze
  permanently — no probe, no decay, no recovery. Simulation: botany freezes on
  40/40 seeds by round ~51. Negligible at q≈0.95 but a real tail risk on a hard
  hidden prompt, and the median-of-8 makes one slow prompt cheap while a *frozen*
  prompt is catastrophic. **Cheap fix: update EMA[0] on the serial path from the
  committed token's own top-2 margin.**
- **`positionAcceptEMA` is never reset per prompt**, is initialised optimistically
  (`0.85·0.98^i`), and its half-life is `ln0.5/ln0.85 = 4.265` rounds — the
  in-code "~9 rounds" comment is ~2x optimistic, and deep positions are
  reach-gated to an effective half-life of tens of rounds. Round 1 always drafts
  4 unconditionally.
- **A fully-accepted round pulls `positionAcceptEMA[acceptedCount]` toward 0.95**
  (L620-637), biasing the schedule upward by one position — this optimism
  transfer is the mechanism that lets the session climb to the cap at all.
- **Dead code confirmed:** the `conf` gate can never trigger a skip
  (`conf ∈ [0.5,1)` vs a k=0 threshold of exactly h=0.2); the L446-449
  "OPERATOR K-TEST VARIANT" default policy closure is overridden at init
  (L197-200).
- **`Sources/MLXFastModel/Qwen35{Attention,Block,GatedDelta,MLP,Model,Ops,RoPE,FastEngine}.swift`
  is editable but NEVER EXECUTED** — `Qwen35FastPathReadiness.swift:11-19`
  hardcodes false, so `selectQwen35ExecutionBackend` always returns
  `.libraryOracle`. The live target is
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` plus
  `Qwen36MTPBlockSession.swift`. Any experiment that edits the former is
  measuring nothing.
- **Verify width is one row short of a much better kernel.**
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415,1483`
  sets `vector_limit ≈ 10` at K=N=5120 and dispatches `qmv` below it,
  `qmm_t_splitk` at or above. Widths 1-9 therefore all take the `qmv` **host
  dispatch**, and `eval_wall` grows 79 → 89 → 106 ms across widths 7 → 8 → 9 with
  *increasing* deltas (+10 then +17), so no linear `a + b·M` fits. **It is not a
  full per-row re-read** — that would give `V(9) ≈ 600 ms` against a measured
  161 ms round. That host dispatch file is *not* editable, but the
  shapes we request are ours to choose.

  > **Corrected 2026-08-16 — "tuned for `M = 1`" was wrong.** An earlier revision
  > said widths 1-9 all take `qmv` "tuned for `M = 1`" and attributed the
  > acceleration to the roofline knee at `M* = 7.9`. The host-dispatch half is
  > right; the rest is not. The base already ships a **crossrow multi-row QMV**
  > inside `qmv_fast` (`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`,
  > `:973-976`, `:1067-1094`, live gate `:1817`), added across the validated
  > submissions `b6c7251 → 08897af → 1033e1a` and therefore already inside the
  > promoted 2.9042 frontier. `:1064` gives the law `IPG = ceil(M / ceil(M/4))`,
  > so **active weight streams = `ceil(M/4)`** — verified against the entire
  > dispatch table `<3,3> <4,4> <5,3> <6,3> <7,4> <8,4> <9,3>`. Width cost is a
  > **staircase with period 4**, stepping at `M = 5` and `M = 9`. The 7 → 8 step
  > stays inside a tread (+10 ms); the 8 → 9 step crosses 2 → 3 streams (+17 ms).
  > This also explains the 161 ms figure directly: 3 streams, not 9. The two
  > models separate cleanly at `M = 5` — see the falsification note at
  > `:120-131`, which pre-registered exactly this test.
- **We can honestly build representative local fixtures.**
  `Sources/MLXFastCLI/main.swift:761-840` exposes
  `generate-golden --prompt-file … --steps N`, and
  `benchmark-qwen-mtp.sh:103,107` honours
  `MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE`. The ranked prompt names are all
  public-domain classics, so same-genre local seeds are legitimate test-set
  construction — **not** hidden-prompt specialisation, provided candidate code
  stays prompt-independent. Every current experiment is measured on a copy task
  with acceptance 1.0; this is the single largest infrastructural gap.
- **`Qwen36MTPTarget` is an `AnyObject` protocol**
  (`Sources/MLXFastModel/Qwen36MTPTarget.swift:36`), so a research-only stub
  conformer can drive the **real** `costModelDepth` / `recordAcceptOutcome` over
  synthetic acceptance with **zero GPU**. Nothing today exercises the depth
  policy over a 512-round horizon; the local window is 64-128 tokens, 4-8x short.
- **Rollback is not free above draftCount 1.** `restoreAfterPrefixReject`'s "no
  repair at any depth" claim holds only at draftCount==1; above that (L1188-1202)
  a reject calls `replayRecurrentPrefix`, i.e. 48 serial GDN layer scans on the
  critical path, and a `canReplayPrefix` failure falls back to a full repair
  forward (L1006-1028) — a second blocking eval.

## External reference implementation: mlx-lm PR #990

`ml-explore/mlx-lm` PR #990 (open, unmerged, 31 commits, 73 comments, zero
maintainer review) is a native-MTP speculative-decoding reference for **our
exact architecture family** — the author enumerates `64 layers (48 GDN + 16
attn)`, `hidden 5120`, `24 / 4 / 256` heads, `lm_head [H, 248320]`. It is the
single most relevant external artifact found so far. <https://github.com/ml-explore/mlx-lm/pull/990>

**Their depth scaling is negative, and that is the headline.** On stock MLX
kernels, deeper drafting *loses*:

| source | model / host | d=1 | d=2 | d=3 | d=5 |
|---|---|---|---|---|---|
| JJJYmmm | Qwen3.5-9B 4-bit, **M5** | 1.52x | 1.36x | 1.13x | **0.80x** |
| AirRunner | Qwen3.6-27B 4-bit, M4 Pro | 1.65x | 1.56x | 1.33x | — |

Named cause: *"stock MLX's `qmv` kernel is tuned for `M=1` and becomes
increasingly inefficient as `M` grows"*. This is the same `vector_limit ≈ 10`
cliff we found independently in `quantized.cpp:1415`.

**Our base reaches raw 2.904 with deep drafting, which their numbers say should
be impossible on stock kernels.** So either our verify path is already
materially better than theirs, or our schedule's conservatism
(`segmentedStreakGate = 3`, `sdpaWidthWallDepthCap = 4`) is an empirically-found
optimum sitting exactly at the qmv cliff. PR #2 Part B separates these, and a
regression there is now a fully expected and publishable outcome.

Their unattributed residual is the strongest sizing anchor we have for host
overhead: a bandwidth model predicts per-round overhead `beta + delta = 1.081`,
every measured run lands at **1.19-1.36**, and the author attributes the gap to
*"kernel launch overhead, MLX graph evaluation, or similar costs"*. **10-28% of
round cost unexplained by bandwidth on the same architecture** — a far better
anchor than mlx PR #3920's 2%.

Four concrete techniques from that PR, ranked for us (their probabilistic /
residual-sampling machinery is inapplicable — our contract is exact-match, i.e.
greedy):

1. **Prefill the MTP head cache during prompt prefill.** Acceptance 86.0% vs
   82.4% (9B 4-bit) and 87.1% vs 83.1% (4B bf16), at ~4 KB/token of permanent
   head KV. Our 512-token seed makes this 2 MB. Assigned as a deliverable to
   PR #3.
2. **Fuse the accepted-draft cache commit into the next draft forward**
   (`cache_commit=(hidden_at_confirmed, draft_tok)`), saving one head forward per
   accepted round. Assigned as a deliverable to PR #4.
3. **Keep linear projections un-split around the recurrent snapshot.** They
   project `qkv/z/b/a` over the full `S` *before* the confirmed/draft split and
   split only the recurrence, giving snapshot overhead `beta ≈ 1.009` — 0.9% of a
   pass. Their depth>1 per-position variant instead iterates the confirmed prefix
   one token at a time, which serialises the recurrence and is a plausible
   second cause of their depth-2 regression, independent of qmv. Under check
   against our `processChunkStashingPrefix`.
4. **`lm_head` is ~70% of head cost** (635.7 MB of 911 MB) over the 248,320
   vocabulary. We already ship a compact 98,304-row draft vocab, so this is
   partly banked; under check whether every head step uses it.

**Per-position acceptance priors** (JJJYmmm, 9B 4-bit, depth 5): p1 82.5%,
p2 64.0%, p3 47.6%, p4 33.9%, p5 23.4%. Our `positionAcceptEMA` initialises to
`0.85·0.98^i` = 0.85 / 0.833 / 0.816 / 0.800 / 0.784 — a **3.3x overestimate at
position 5**. Their model is smaller so the levels are not ours, but the shape
matters: their conditional acceptance **degrades with depth** (constant-`q` fit
to p1 predicts p5 = 0.38, not 0.234) whereas `0.98^i` encodes near
depth-independence. PRs #1 and #2 now both emit realised per-position acceptance
over a 512-token window so we can re-fit against our own data.

**Two evidence downgrades, both retracted to PR #4.** `ml-explore/mlx` #3920 is
a *closed, unmerged* PR with zero comments reporting only +2.0-2.8% decode, not
an open issue establishing host-bound decode. `ml-explore/mlx-lm` #250's
200-210 ms vs 130-140 ms measurement is real, but its non-static-shape
*explanation* is the reporter's own flagged guess on an issue with zero replies.


## The head mismatch: we do not locally measure the head we are scored on

Established 2026-08-16 by direct read of the manifest, the track fixture and the
setup script. This is the largest single correction of the campaign so far
because it silently mis-scales every local per-draft cost number.

### The two heads

| | pinned / organizer | declared / ranked candidate |
|---|---|---|
| source | `fixtures/qwen3_8_27b_mtp_track.json` `.mtp_head` | `mtp-head.manifest.json` (repo root) |
| identity | `EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e0` | `hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaf` |
| dtype | bf16, unquantized | MLX-affine 4-bit, group 64 |
| tensor bytes | 849,398,784 | 238,934,093 |
| tensor count | 15 (8 matrices + 7 norms) | weight/scales/biases triples, count not gated |
| tree digest | `157f750e...` (computed from the fixture manifest) | `cc209e30...` (declared) |

The 3.6-generation head was a 4-bit g64 conversion (8 x 3 + 7 = 31 tensors); the
3.8 head is bf16 (8 + 7 = 15). That is what the fixture's `tensor_count_note`
explains, and it is why the count differs between generations.

`git log -- mtp-head.manifest.json` shows only two commits — the initial track
import and `deb63ad Validate submission 91743270-...`. **The 4-bit declaration
arrived through a promoted submission and is on our base.** Head quantization is
therefore already banked; it is not an available idea.

### Which head runs where

- **Ranked candidate leg:** the declared 4-bit head.
- **Ranked serial (pinned) leg:** always the pinned bf16 head. `QwenRuntimeMTP.swift:235-268`
  — *"RECORDED, NEVER GATED (operator-ratified 2026-08-14). Head weights are part
  of the competitive surface... The BASELINE leg always resolves to `pinned`."*
  `main.swift:1955-1976` confirms `uses_native_mtp_head` / `uses_pinned_mtp_head`
  / `mtp_head_attached` are provenance, not gates, and that the head is resident
  on both sides of the pair so its residency cost is charged to the serial
  denominator too — only the drafting differs.
- **Local candidate leg:** the pinned bf16 head, because `setup-qwen-mtp.sh:66-67`
  hardcodes `MTP_HEAD_MODEL_ID="${MLXFAST_QWEN_MTP_HEAD_REPO:-EigenLabs/Qwen3.8-27B-MTP-bf16}"`
  and the local path never reads `mtp-head.manifest.json`.

Overridable locally: `MLXFAST_QWEN_MTP_HEAD_DIR`, `..._HEAD_REPO`,
`..._HEAD_REVISION`, `..._HEAD_MANIFEST_PATH`, `..._CACHE_ROOT`.

A quantized head **is** loadable: `Qwen36MTPHeadAttachment.swift:305-340`
`verifyHeadIndex` was relaxed for declared heads and now enforces only
`weightMap.count >= 3`, a bare (unprefixed) namespace, and the keys `fc.weight`,
`norm.weight`, `pre_fc_norm_hidden.weight`.
`MLXFastConstants.qwenMTPHeadTensorCount = 15` survives only in an error string
and as a reported constant; it is not measured against the resident head.

### Declaration semantics (`QwenMTPHeadDeclaration.swift`)

`relativePath = "mtp-head.manifest.json"` (`:70`). Only an **absent** file or an
explicit `"pinned"` source selects the pinned head (`:74-84`) — a manifest that
is present but unreadable **refuses** rather than falling back. Non-pinned
declarations require `sha256` and `bytes` (`:119`); `remote` requires an
`hf:repo@rev`-form `source_url` (`:142-148`); `in_branch` requires a safe
repo-relative path (`:157`). `origin` is diagnostic only (`:224-225`).

`mtp-head/README.md` is the only file in `mtp-head/` and documents the tree
digest rule (SHA-256 over `LC_ALL=C`-sorted `"<sha256>  <relpath>\n"` lines,
excluding the top-level `README.md`). **Its prose claim that the checked-in
declaration selects `"pinned"` is stale** — the manifest says `"remote"`. The
manifest is authoritative. Cleanup target.

### The re-basing rule

At the measured local decode bandwidth of 227 GB/s (M4 Pro):

```
delta_head  = (849,398,784 - 238,934,093) / 227e9  =  2.689 ms per head forward
m_ranked(d) = m_local(d) - 2.689 ms      for every d >= 1
C_ranked(d) = C_local(d) - 2.689 * d ms
```

Head forwards per round at depth d (all accepted) is exactly `d`
(`Qwen36MTPBlockSession.swift:836-872`), so d=0 rounds are head-independent and
`C(0) = 67.0 ms` is unchanged.

| quantity | local (bf16 head) | ranked (4-bit head) |
|---|---:|---:|
| low-band marginal `m_lo` (d=1..6) | 9.65 ms | 6.96 ms |
| high-band marginal `m_hi` (d=7..8) | 18.05 ms | 15.36 ms |
| `C(8)` | 161.0 ms | 139.5 ms |
| `h = m_lo / C(0)` | 0.144 | **0.104** |
| head chain, 8 chained steps | 29.94 ms | 8.42 ms |
| head + compact lm_head chain (x8) | 39.92 ms | 18.40 ms |
| per-draft low-band bandwidth term | 4.99 ms | 2.30 ms |

The shipped `headStepCostRatio = 0.20` therefore overestimates the true marginal
ratio by **1.39x locally and 1.92x against the ranked configuration**. This
strengthens, not weakens, the PR #1 cost-curve fix.

Local 64-token window, head-corrected: 54 head forwards, 0.145 s saved, MTP
`0.087837 -> 0.085568` s/token, local directional ratio `1.4794 -> 1.5187`.
Ranked absolute throughput is therefore ~14.6% better than the local run
suggests — one of the four explanations for local 1.47 versus ranked 2.90.

### The depth optimum survives the correction (and only ever moves deeper)

A uniform per-draft cost reduction weakly **increases** `d*`. Swept over five
`m(d)` shapes (two-regime knee@d7, flat, cliff@d5, smooth ramp, late knee@d8) x
acceptance q in {0.85, 0.90, 0.93, 0.95, 0.97}:

- `d*_ranked >= d*_local` in **every** cell,
- equal in 22 / 25 cells, `+1` in 3 / 25,
- worst-case throughput loss from shipping the locally-fitted optimum: **0.93%**.

So a locally-tuned depth policy is a safe, never-too-deep estimate. It is still
worth evaluating `d*` and `d*+1` on the re-based table. Note that `d* = 6` in the
roofline-predicted shape, which exceeds the shipped `sdpaWidthWallDepthCap = 4` —
independent reconfirmation of the joint-binding result.

### Guardrail direction — this one inverts

`check_stall_guardrail` reads `max_block_request_seconds_after_first /
p50_block_request_seconds_after_first`. Subtracting the same absolute per-round
amount from both max and p50 **raises** the ratio:

```
ratio_ranked = (max_local - 2.689 * d_max) / (p50_local - 2.689 * d_p50)
```

Local `max/p50` is an **optimistic** reading of ranked guardrail headroom, not a
conservative one.

### Open: the `da336ce9...` local head digest

PR #3 recorded `head_provenance_sha256 = da336ce9894f859d4da39855ba35972fe8152224cf2fa0c2c7122b6cfcfc4e94`.
That matches **neither** the declared 4-bit digest (`cc209e30...`) nor the pinned
bf16 tree digest computed from `fixtures/qwen3_8_27b_mtp_head.sha256` under the
documented rule (`157f750e...`, over 4 records: `.gitattributes` 1519,
`config.json` 3570, `model.safetensors` 849400347, `model.safetensors.index.json`
1002, total 849,406,438). An exhaustive subset search over those four files
reproduces neither value, so the local tree is a third file set — most likely an
HF-snapshot layout difference or an extra cache/stamp file. The advisor host has
no model cache, so this can only be settled on a student measurement host by
`shasum`-ing the resident tree. **Every result must from now on record
`head_provenance_sha256`, the head directory, its byte count and its dtype.**

### What this opens

Because the head is editable (`mtp-head.manifest.json` plus `mtp-head/`, exempt
from the source byte budget with its own 2 GiB cap), head *representation* is a
one-line, zero-source-risk lever:

- the ranked head is 4-bit, so it routes through the **same** scales-keyed
  quantized `qmv`/`qmm` walk that PR #5 is retuning, at `M=1`, up to 8 chained
  steps per round — a qmv M=1 win compounds on ranked in a way earlier accounting
  missed;
- if 4-bit is latency-bound rather than bandwidth-bound at `M=1`, an **8-bit**
  head could be faster in wall time despite 2x the bytes;
- if 4-bit quantization costs acceptance, the pinned bf16 head could be a net win
  on the ranked host even at +2.689 ms per draft.



---

## Literature verdicts on the head questions (2026-08-16)

Three parallel literature agents settled the two open questions above and
refuted one advisor novelty claim. Every conclusion below is quoted from a
retrieved primary source, not inferred.

### 1. Our depth rule is NOT novel — adopt, do not reinvent

The claim "no published work chooses draft depth by maximizing expected
accepted tokens over a *measured, non-affine* verification cost" is **refuted**
by at least six independent works, the earliest from 2023.

| work | form | notes |
|---|---|---|
| **Sequoia**, NeurIPS 2024 (`2402.12374`) | `Speedup(n,d) = G(n,d) / (t(n) + d·c)` | closest match. `t(n)` measured per model+hardware, then grid search. Appendix G.5: *"the forward pass times are roughly constant for low values of `n`, but then eventually start growing roughly linearly"* — our exact two-regime shape. Offline-static. |
| **D-cut** (`2607.14647`) | `argmax_ρ Σ s / C(B,ρ)` | best implementation pattern: profile a latency table at **startup**, read it at runtime. Explicit: *"instead of treating each extra draft position as having a constant cost"*. |
| **DSpark** (`2607.05147`, DeepSeek) | `Θ = τ · SPS(B)` | *"this capacity curve is profiled once during engine initialization and stored as a lightweight cost table"*. Warns of **jagged SPS cliffs** breaking unimodality; removes early-stopping in favour of unconstrained global search. |
| **ECHO** (`2604.09603`) | `C(d) ≈ C_0(1 + γ[d − d_knee]^+)` | a two-parameter **ReLU hinge**: flat then linear. Names our assumption as the thing it refutes: *"Prior work often assumes `T_verify ≈ T_ar` for moderate `K`"*. |
| **SMART** (`2604.09731`) | `C_verify(T) = γ(exp(δ|T|^ρ) − 1) + η` | smooth fit, *five forward passes* to calibrate; marginal expand/stop test in Eq. 16. |
| **Yggdrasil** (`2512.23858`) | profiled `T_verifier(W)` | *"Previous works … treat it as a constant. This simplification overlooks the opportunity…"* |
| **Su et al.** (`2310.18813`, 2023) | grid search `s_opt(b)` | earliest measured depth table found. |

Category (a), the affine canon we are replacing: **Leviathan et al.**, ICML 2023
(`2211.17192`) — *"we'll assume that we can run γ+1 concurrent evaluations of
`M_p` in parallel without increasing the walltime"*; **SpecDec++**
(`2405.19715`) — *"Equation (3.1) holds under the implicit assumption that the
forward passes of each of the models take constant time"*.

**Residual novelty is the setting, not the rule.** All seven counterexamples
locate the knee in **batched** verification crossing GPU compute saturation.
Ours arises at `B = 1` from MLX kernel granularity and the GDN-vs-full-attention
width regimes — a different physical mechanism. No Apple-Silicon/MLX instance
and no native-MTP-with-fixed-target instance was found. Per-round **online**
selection against a **piecewise** curve is a defensible increment: Sequoia is
offline-static, SMART fits a smooth exponential, DSpark needs a two-step lag.

**Operational consequences.**
- Ship Sequoia's structural form with `n = d + 1`; cite it.
- Ship D-cut's mechanic: profile at startup, `argmax` at runtime. This is the
  published precedent for measuring `H` at run time rather than freezing a table.
- Keep ECHO's hinge as the **cheap fallback parameterization**: three numbers
  (`C_0`, `γ`, `d_knee`) instead of a nine-entry table, if the full table proves
  awkward to carry.
- **Causality hazard (DSpark §5.2).** A per-round global `argmax` over a jagged
  curve using *current-round* head confidences can leak future-token
  information. For us the depth choice cannot change which tokens are emitted —
  every draft is still verified exactly and rejected drafts are rolled back — so
  exactness is structurally safe. But any rule that reads the current round's
  drafted token *identities* (rather than the head's pre-verification
  confidences) must be re-derived before shipping.

### 2. The 4-bit head is FAST on Apple Silicon — the "Fermi" worry is retracted

`2607.14568` is real and its 12% figure is verbatim, but it is about the
**NVIDIA Fermi microarchitecture** (2011 Tesla C2075, sm_20), *not* Apple
Silicon. The advisor's queue item 24 premise was wrong.

Two independent reasons it does not transfer:

1. Its mechanism is a **Fermi-ISA half-rate shift issue rate**. MLX's 4-bit
   `qdot` **issues no shifts at all**: `load_vector` pre-scales the *activations*
   (`x[i]`, `x[i+1]/16`, `x[i+2]/256`, `x[i+3]/4096`) so four masks
   (`& 0x000f, & 0x00f0, & 0x0f00, & 0xf000`) suffice. Per weight, 4-bit reads
   **half the bytes with one quarter the load instructions**, paying one integer
   `AND`. Structurally the opposite kernel.
2. Its 8-bit baseline reached ~48 GB/s of ~144 GB/s peak — enormous compute
   headroom. M4 Pro decode sustains ~227 of 273 GB/s.

**Measured anchors at our exact 16 → 4.5 bpw ratio (3.556x bytes):**

| source | host | F16 → Q4 wall | % of byte ratio |
|---|---|---:|---:|
| MLX Discussion #3209, `mlx_lm.benchmark`, batch 1 | M3 Ultra, Qwen 32B | **3.00x** | 84.4% |
| llama.cpp #4167 (byte ratio 3.525x) | **M4 Pro 20c** | **2.953x** | 83.8% |
| llama.cpp #4167 | M4 Pro 16c | 2.888x | 81.9% |
| llama.cpp #4167 | M4 Max 32c | 2.880x | 81.7% |

**Predict ~3.0x wall-time for the head weight read alone (band 2.8–3.3x).**
Ordering is strictly monotonic on MLX: more bits is always slower at batch 1.

**An 8-bit g64 head would be 1.6–1.7x SLOWER**, not faster: 8.5 bpw gives
451,243,104 B = 1.888x the 4-bit bytes. Six measured Q8 → Q4 points (MLX and
llama.cpp, M3 Ultra and M4 Pro) all land at 1.62–1.71x, i.e. 86–91% of the byte
ratio. **Queue item 24 is closed. Do not stage an 8-bit head.**

**★ One concrete, checkable 4-bit hazard.** In
`mlx/backend/metal/quantized.cpp`:

```
inline int qmv_fast_k_alignment(int bits) {
  return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;
}
bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;   // bn = 8
```

**4-bit requires `K % 512 == 0`; 8-bit only requires `K % 256 == 0`.** A `K`
that is 256- but not 512-aligned **silently** drops from `qmv_fast` to the
bounds-checked generic `qmv` (`load_vector_safe`/`qdot_safe`) — **and therefore
also loses the crossrow row-sharing path entirely**, since that lives inside
`qmv_fast`. Our scored **reduction** dims all pass: 5120, 6144, 8704, 10240 and
17408 are multiples of 512, and `N` dims 5120/16480/34816/98336/248320 are all
multiples of 8. This has not been asserted in code and is nearly free to check.

> **Corrected 2026-08-16.** An earlier revision of this paragraph listed `16480`
> among the reduction dims and claimed it was a multiple of 512. **It is not**
> — `16480 = 512·32 + 96`. There is no live defect, because 16480 is an *output*
> dim (the GDN fused in-projection is 5120 × 16480) and `N` only needs `% 8`.
> The conclusion held; the stated reasoning did not. **Do not pick 16480 as a
> benchmark `K`** — it silently measures the unshared generic kernel.

### 3. Head-precision A/B: worth a slot, but run the free pre-check first

**Expected acceptance cost of the 4-bit head: −1% to −3% mean accepted length,
with a credible tail to −8% or worse.**

- **`2607.04244`** (Qwen3.5-4B, 537M drafter vs our 424.7M, INT4 target,
  lossless spec) Table 5, mean accepted length: BF16 **5.03**, INT4 **RTN 4.91
  (−2.4%)**, AWQ 4.92, GPTQ 4.98 (−1.0%). *"INT4 quantization causes only a
  minor change in the drafter's mean acceptance length relative to the BF16
  baseline."* MLX affine is round-to-nearest with per-group scale and bias and no
  calibration data, so **the RTN row is our row**; g64 is finer than their g128
  and affine is stronger than symmetric, but we have no Hessian calibration.
- **`2508.08192`** (Meta, Llama at scale) Table 2: FFN-only INT4 draft heads
  cost −1.1% to +0.3% TPC. Discount — that is partial quantization; ours is
  wholesale.
- **QSpec, EMNLP 2025** (`10.18653/v1/2025.emnlp-main.240`), the tail risk:
  *"applying GPTQ quantization to the EAGLE draft model resulted in substantial
  degradation of the acceptance rate."* No number given. This is an
  **EAGLE single-transformer-layer head quantized wholesale** — architecturally
  the closest published thing to ours. Only **one** primary source; the echo in
  `2505.22179` cites it and must not be double-counted.

**★ "Both sides 4-bit" buys us NOTHING.** Every high-acceptance quantized-draft
result rests on the draft being a quantized *copy of* or *weight-sharing with*
the target, so quantization error is identical on both sides and cancels:
QSpec (*"Common weights and KV cache overwriting ensure consistent
predictions"*, 93–95% acceptance), QuantSpec `2502.10424` (>90%), ML-SpecQD
`2503.13565` (71.2% vs 41.7% for a custom 68M draft), SPEQ `2510.18525` (0.976,
draft bit-extracted from the FP16 target). **Our head has independently trained
weights.** Matching the target's quantization *geometry* does not match its
quantization *error*: the two are statistically independent and **add** in the
disagreement budget. Do not treat "both 4-bit" as an asset.

**★★ Corroboration that `0.85·0.98^i` is badly wrong.** `2601.11580` on
GLM-4.5-Air, whose released weights contain only the first MTP module *"reused
autoregressively for subsequent ones"* — exactly our architecture — measures
position-wise acceptance **0.92 → 0.68 → 0.38** on GPQA-Main. Together with the
JJJYmmm profile (82.5 / 64.0 / 47.6 / 33.9 / 23.4) this is a second independent
measurement of steep conditional decay in a single reused head.

**Conversion rule:** in this regime, per-position acceptance loss maps
**≈ 1:1** onto mean-accepted-length loss. (Applying a uniform 4% relative hit to
the 0.92/0.68/0.38 profile moves `1 + a1 + a1a2 + a1a2a3` from 2.78 to 2.67,
−4.1%.) Any per-position counter a student measures converts directly into the
break-even calculation.

**★★★ Free offline pre-check before spending a GPU slot.** Compare the bf16 and
4-bit head outputs on the public fixture, **untimed and outside the scored
path**: per-token KL and top-1 agreement. `2407.09141` reports Spearman **0.981**
between KL divergence and flip rate. Decision rule: top-1 agreement **≥ 99%** →
the acceptance cost is in the −1% band, skip the A/B; **≤ 96%** → QSpec
territory, the slot is well spent.

**Correction to the subagent's traffic model.** It assumed the draft readout is
the full 248,320-row lm_head (715 MB) and derived a 1.64x drafting-traffic
ratio. We use the **compact 98,336-row draft vocabulary (283.2 MB)**, so the
correct per-draft-step totals are:

| arm | head | compact readout | per draft step | @227 GB/s |
|---|---:|---:|---:|---:|
| local, bf16 head | 849.4 MB | 283.2 MB | **1132.6 MB** | 4.99 ms |
| ranked, 4-bit head | 238.9 MB | 283.2 MB | **522.1 MB** | 2.30 ms |

**Ratio 2.169x on bytes, not 3.556x** — the shared 4-bit readout does not shrink
with head precision. This matches the existing roofline table exactly, and it
**changes the head-swap decision thresholds**: an isolated draft-build timing
that includes the readout should move by ~1.9x wall, not ~3.0x.

**★★★ The bonus lever is bigger than head precision.** In the ranked arm the
compact readout is **283.2 / 522.1 = 54.2% of all drafting bandwidth**, and it is
completely untouched by head quantization. Published reductions of exactly this
term: **Gemma 4** (`2607.02770`) — *"replacing the projection operation to the
entire vocabulary by a top-k operation on clusters of tokens … reduced from
x262,000 to x4096 while preserving a similar acceptance rate"*; **SlimSpec**
(`2605.10453`) — *"VocabTrim and SpecVocab can reduce LM-head latency by only
about 60%, while SlimSpec achieves an approximately 4-5x reduction"*, and
*"its LM-head still performs projection to a large vocabulary, becoming one of
the major computational bottlenecks."* Note this is a *different* mechanism from
the static prefix trim we already rejected: halving the compact prefix to 49,152
rows regressed acceptance 1.00 → 0.877, whereas a clustered or low-rank
two-stage readout preserves coverage.


---

## Draft-readout precision path — four source facts verified during the brief audit

These four were all read directly from source on 2026-08-16 while auditing PR #7
and PR #8. Two of them close standing "UNVERIFIED" notes in this file; one of
them corrects a line number recorded earlier from memory. **Each is quoted with
its file and line range so the next reader can re-check it in under a minute
rather than re-deriving it.**

### 1. Low-bit affine `qmv_fast` IS instantiated on the non-NAX path

This resolves the standing "UNVERIFIED — we only ever confirmed bits ∈ {2,3,4,5,6,8}
for the NAX kernels" note. In
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal`, the
instantiation macro chain is:

```
:150-158  instantiate_quantized_all()   -> groups(2),(3),(4),(5),(6),(8)
:145      instantiate_quantized_groups  -> types(128,·) types(64,·) types(32,·)
:141-144  instantiate_quantized_types   -> funcs(float|float16_t|bfloat16_t)
:132-139  instantiate_quantized_funcs   -> instantiate_quantized_all_batched
:82-86    ..._all_batched               -> batched_wrap(affine_qmv_fast, ...)
:78-80    ..._batched_wrap              -> batched(...,1) and batched(...,0)
```

Therefore `affine_qmv_fast_<type>_gs_64_b_2` and `affine_qmv_fast_<type>_gs_64_b_3`
both exist, in batched and non-batched form, for `float`, `float16_t` and
`bfloat16_t`. **Bits ∈ {2,3,4,5,6,8} × group size ∈ {32,64,128} is the full
supported matrix on the ordinary path, not just the NAX path.** The host `fast`
gate (`backend/metal/quantized.cpp:259`, `N % bn == 0 && K % 512 == 0`) is
bits-independent, so lowering bit width cannot fall off the fast path.

### 2. `qwen35DraftSelectKernel` is bit-width agnostic

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1944-2008` (definition),
called at `:2372`. `inputNames: ["logits"]`; the body reads
`float value = float(logits[index])` and runs a two-level SIMD argmax (shuffle
reduce → one threadgroup barrier → second reduce in simd group 0), then writes
`token_id[0] = int(best_id < PREFIX_COUNT ? best_id : best_id + CONTROL_OFFSET)`.
Template params are `REAL_COUNT`, `PREFIX_COUNT`, `CONTROL_OFFSET`, `TG_SIZE=1024`.

**It never touches packed weights, scales or biases** — it consumes dense logits
that have already been produced. Any belief that this kernel "assumes 4-bit" is
false, and a stopping-rule branch conditioned on it could never fire.

### 3. Draft-head bit width lives in `makeCompactDraftHead()`, not `draftTokenID`

`Qwen35.swift:2406-2434`. The function builds
`compactRows = concatenated([prefix 0..<98_304, controls 248_044..<248_070,
padding 0..<(paddedCount − realCount)], axis: 0)`, and when `lmHead` is a
`QuantizedLinear` it reconstructs one with `bits: quantized.bits` (`:2428`) —
i.e. it **inherits 4 bits from the parent head**. Row-slicing packed 4-bit
weights is valid because packing runs along K, not N.

Related anchors: `draftTokenID` at `:2361-2387`, whose guard is
`_draftHeadW == nil, usesCompactDraftVocabulary` (`:2364`);
`usesCompactDraftVocabulary` at `:2401-2404` requires
`configuration.vocabularySize == 248_320 && lmHead != nil && _draftHeadW == nil`.

Three consequences follow directly:

- **Steady-state peak memory FALLS, it does not rise.** Requantizing *replaces*
  an existing allocation: −62.9 MB at 3-bit, −125.9 MB at 2-bit. (An earlier
  advisor note warning of a peak-memory *increase* was backwards.)
- **The real hazard is a transient**, not the steady state: dequantizing
  98,336 × 5120 = **503,480,320** weights costs ≈ **1.01 GB in fp16** and
  ≈ **2.01 GB in fp32** while it is live. Chunk by rows if it bites.
- **Crossrow is not in play for the draft readout.** Draft readout is M=1, and
  the crossrow kernel carries `static_assert(M >= 2 && M <= 9)` plus a
  `bits == 4` host gate. A draft-head bit-width change therefore cannot move the
  dispatch family.

### 4. Acceptance is exact token-ID match — exact lines

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:641-649`. **Note the line
number: `:641`, not the `~:638` recorded earlier in this file from memory.**

```swift
static func acceptedDraftPrefixCount(drafts: [Int], verifyArgmax: [Int]) -> Int {
    precondition(verifyArgmax.count >= drafts.count)
    for index in drafts.indices where verifyArgmax[index] != drafts[index] {
        return index
    }
    return drafts.count
}
```

Pinned by `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift:12,28`. Nearby:
`defaultDraftDepth = 2` at `:652`.

This is the structural basis of prediction 5: acceptance compares **token IDs**,
so any change confined to the drafter's numerics can only change *which* drafts
are proposed and *how many* are accepted — never the emitted token, which always
comes from the verifier's argmax. It still requires a parity run to confirm
empirically, because "structurally cannot" and "measured did not" are different
claims.

### Provenance note attached to this audit

`git log --all --oneline -S"6.6 ms" -- research/` returns exactly one commit,
`b2419f4` ("research: the width cost law is measured, and both prior models are
refuted"). The retraction of the `+5.5 / +6.6 ms` boundary excess, of the
`h(3) = 0.2446` "ramp onset" explanation, of the "2.4× smaller live than
isolated" discrepancy, and of `d* = 7` is recorded above at lines 186-210 of this
file and struck in place in `CURRENT_RESEARCH_STATE.md`.

### ★★★★★ The methodological point these four facts make together

Three of the four were **gates I had written into a student brief as
"Unverified: check whether X"**. All three were resolvable from source in under
ten minutes by the person who wrote the brief. A gate the advisor can close from
source in minutes costs a student hours and can terminate in a spurious "blocked"
report.

> **"Unverified" in a brief is a debt the advisor owes, not a task to delegate.**
> Before writing "check whether X", try to check X.

And the contrast that made the audit worth doing: the brief that survived the
contamination sweep intact (PR #7, Part A) survived because **its numbers were
derived from physical constants and independently recomputable** — 98,336 rows ×
5120 cols at 4/3/2 bits plus fp16 scale+bias ⇒ 2880/2240/1600 B/row ⇒
283.2/220.3/157.3 MB ⇒ Δt at 227 GB/s. Every figure re-derived in two minutes and
all held. The one number in that brief that came from *other numbers in my own
notes* was the one that had to be withdrawn.

> **Prefer quantities recomputable from physical constants (bytes, bandwidth,
> element counts) over quantities inherited from the research record. The first
> kind fails loudly and locally; the second kind propagates.**


---

## Crossrow QMV against the roofline — measured facts from PR #8 (merged 2026-08-16)

Source: PR #8 (`qwen-thorfinn`, `qwen38-r1-e7-crossrow-na5`), merged at
`fa9a216a`. Apple M4 Pro, one thermal session, `dirty=0`. W&B runs
`bq9xfu6d` (NA=4 control), `e79lcwx2` (NA=5 v1), `1y91qkq5` (NA=5 v2).
Repro: `research/run-qmv-curve.sh <TAG> b2419f41` → `research/qmv_na_compare.py`,
`research/qmv_gbps_table.py`. **GPU peak bandwidth ≈ 273 GB/s.**

### FACT 1 — the boundary widths are the saturated ones; the interior is not

> ## ★★★★★ 2026-08-16 — **THE "% OF PEAK" COLUMN BELOW IS DISPUTED.** Read this
> ## banner before using it to target anything. Assigned as PR #10.
>
> The `% of 273 GB/s peak` column is computed from `gbps_stream_corrected`, which
> `research/qmv_cost_curve_summary.py:274-278` defines as
> `weight_streams(m) * weight_bytes / seconds_per_call`, with
> `weight_streams(m) = ceil(m/4)` (`:132-136`).
>
> **Invert it and a different invariant appears.** Dividing out the stream count
> to recover `gbps_nominal = weight_bytes / seconds_per_call`:
>
> | M | streams | stream-corrected | nominal | **nominal × M** |
> |---:|---:|---:|---:|---:|
> | 4 | 1 | 165.6 | 165.60 | 662.4 |
> | 5 | 2 | 262.1 | 131.05 | 655.2 |
> | 8 | 2 | 183.0 | 91.50 | 732.0 |
> | 9 | 3 | 239.5 | 79.83 | 718.5 |
>
> **`nominal × M = 692 ± 5.6%`** (stdev 38.9, n=4). Under a bandwidth-bound
> model `gbps_nominal` should be flat; it instead spans **2.07×** (79.8 → 165.6),
> monotonically decreasing in `M`. The invariant is `nominal × M`, i.e.
> **`seconds_per_call ∝ M`** at fixed unique weight bytes. Cross-row weight reuse
> buys ≈ nothing at `M ≥ 4`.
>
> **Independently cross-validated by PR #5**, a different dataset on a different
> shape. Its `implied_streams` curve (`:171`) ramps linearly above `M = 3`, and
> the ramp extrapolates back to the flat floor `c ≈ 1.0` at **`M ≈ 3`** — which
> is where PR #5 independently reports the plateau ends ("the measured plateau
> *ends at `M = 1–3`* on every shape").
>
> **State the method, because the knee moves with it** (this was caught by
> re-deriving the number rather than trusting the note that recorded it):
>
> | method over the ramp | slope/row | intercept | knee |
> |---|---:|---:|---:|
> | endpoint `M=4→9`, proportional line `c = βM` | 0.326 | (forced 0) | **3.07** |
> | endpoint `M=4→9`, with its own intercept | 0.326 | −0.064 | 3.26 |
> | OLS `M = 4..9` | 0.309 | +0.034 | 3.13 |
> | OLS, windows `3..9 / 4..9 / 5..9 / 6..9` | 0.30–0.32 | ~0 | 2.99–3.27 |
>
> ⇒ **The load-bearing claim is `knee ∈ [2.99, 3.27]` — robust to every method
> and window tried.** The single figure "3.07" is the proportional-line case
> only; do not quote it as *the* knee, and note that "intercept ≈ 0" is an
> approximation whose sign is not even stable across methods. The robustness of
> the band is the real result, because it means the knee is not an artifact of
> a fitting choice.
>
> ⇒ **Two-regime roofline model, one free parameter:**
> `t(M) = max(t_bandwidth, β·M)`, knee at `M ≈ 3`. Bandwidth-bound below,
> **ALU-bound above.** It *predicts* the knee rather than fitting it, and it
> needs no boundary-excess term.
>
> **Why the "% of peak" framing is suspect:** it corrects bandwidth by the
> integer `ceil(M/4)`, but PR #5 already showed that integer does not govern
> cost — `implied_streams` is **continuous** (1.00 0.99 1.01 1.24 1.64 1.90 2.17
> 2.44 2.87) exactly where the integer model demands steps at 1,1,1,1,2,2,2,2,3.
> Correcting by a quantity that does not govern cost manufactures apparent
> headroom. On this reading the interior is not leaving 33–39% of the machine
> idle; it is **ALU-saturated and there is no bandwidth prize to collect.**
>
> **Status: derived by the advisor from merged artifacts, NOT separately
> measured.** Nothing in it is inherited from the retracted-number family. PR #10
> (`qwen-thorfinn`) settles it with two microbenchmark arms — cut arithmetic at
> fixed bytes, cut bytes at fixed arithmetic. Until it lands, treat FACT 1's
> targeting consequence as **live but unsafe to spend a student on**.
>
> **Every number above is regenerated by `research/roofline_regime_check.py`**
> (`python3 research/roofline_regime_check.py`, no deps, < 1 s). Do not trust
> this banner — run it. It prints the inversion table, all six knee estimates,
> and the integer-vs-continuous comparison. Useful extra it reports: `nominal×M`
> is **6.0× tighter** than `nominal` (5.6% vs 33.4% relative sd), which is the
> quantitative form of "the ALU-bound invariant fits better."

Stream-corrected achieved bandwidth, NA=4 (the shipped kernel):

| width | GB/s | % of 273 GB/s peak | role under the old framing |
|---|---:|---:|---|
| M=5 | **262.1** | **96%** | "boundary — wasteful" |
| M=9 | **239.5** | **88%** | "boundary — wasteful" |
| M=4 | 165.6 | **61%** | "interior — efficient" |
| M=8 | 183.0 | **67%** | "interior — efficient" |

**The framing was backwards.** The widths that spend an extra weight stream are
the only ones reaching the machine's bandwidth. The extra stream is not overhead;
it is the memory-level parallelism that gets the kernel to peak. Collapsing M=5
from two streams to one (NA=5) dropped it to 85.6–95.5 GB/s — one NA=5 group
sustains 95.5 where one NA≤4 group sustains 165.6, so the wide-5 path degrades
**superlinearly and independently of stream count**.

**Consequence for targeting: the recoverable headroom is in the interior widths
M=4/7/8 (61–67% of peak), not at the boundaries (88–96%).** The 262 GB/s at M=5
is an existence proof that the hardware is reachable at these shapes.

### FACT 2 — `NA=5` is refuted by two independent implementations

Both v1 (pure wide-5, `704af6f`) and v2 (vec4 + scalar tail, `0a739c9`) made the
boundary widths **1.13–1.54× slower**, while the intended mechanism fired exactly
as designed (`weight_streams` 2→1 at M=5, 3→2 at M=9, unchanged elsewhere, all 8
shapes). Break-even needs ~131 GB/s; the better implementation reached 95.5.
Through the measured law `C(d) = V(d+1) + 4.46 + 3.73·d`: `C(4)` 127.736 →
173.012 ms (+35.4%), `C(8)` 213.248 → 249.020 ms (+16.8%). **Not marginal.**
`NA_max = 4` is restored on the base; the `static_assert(NA >= 2 && NA <= 4)` at
`mlx-generated/quantized.cpp:993` and its twin `quantized.h:980` stand.

### FACT 3 — the ramp and the boundary excess are SEPARABLE

Median nominal GB/s step M=3→4: **NA=4 −35.6, NA=5 −35.3** (within 1%), under a
manipulation that moved the boundary widths by 1.13–1.54×. Pre-registered
prediction #4, confirmed. **This is the structural licence to attack the interior
widths without a boundary confound**, and it is the most reusable single fact in
the PR.

### FACT 4 — a semantically-identical refactor broke bit-exactness

v1 is **bitwise identical to M=1 for M=1..9 on 8/8 shapes**. v2 computes the same
arithmetic and **fails at exactly M=5 and M=9** (0/8, max|d| 0.207–1.0, `lm_head`
1.0), exact at every other width — it breaks precisely at the widths taking the
new path. A scalar tail added beside a `vec4` body changed FMA contraction inside
the vec lanes. Registered as **bit-exactness hazard (8)**.

> **Any edit to a kernel on the exactness-critical path must re-run the
> bitwise-vs-M=1 gate, including edits believed to be pure refactors. "I did not
> change the math" is not evidence.** Unlike every other hazard on that list, this
> one is invisible both in the design and in the diff.

### FACT 5 — two prior record entries do not survive

- **The "live defect" (vendored/stock 0.87–0.92 at M=2..5 on the two N=5120
  shapes) does not replicate**: never below 0.950 across five sessions. Controls:
  144-point stock-pip control at median 1.0000 (0.954–1.019); two independent
  NA=4 sessions within 0.4%. The associated shape-aware guard is independently
  dead — a **perfect free routing oracle** saves ≤**0.53%** of weighted verify and
  **0.00% at M≥6**, against a guard cost of ~1%.
- **PR #5's `step_excess` magnitude (0.112 → 0.169) is inflated** and is demoted
  to a reading caution: the statistic averages the flat M=2/M=3 increments into
  the interior baseline. Reported unprompted by its own author.

### ★★★★★ Method rules banked

> **Report bandwidth-bound work against the roofline, not against itself.** Every
> earlier reading of this kernel compared widths to other widths and concluded the
> boundaries were anomalous. Adding one column — "% of peak" — inverted the
> conclusion. **A ratio to your own baseline cannot tell you whether the baseline
> was the problem.**

> **Instrument the mechanism, not only the outcome.** Because `weight_streams` was
> logged per width per shape, this experiment could say "the intervention worked
> and the hypothesis is still false," which indicts the premise alone. A pure
> outcome measurement would have left premise and execution equally suspect.

> **Overhead you can see is not necessarily overhead.** In a bandwidth-bound
> regime, apparent duplication is often the parallelism. Before removing a
> redundancy, ask what it is buying.

> **Bound the prize before paying for the experiment.** An oracle upper bound
> closed the shape-aware-guard branch at zero GPU cost. When a proposed
> optimization has a computable ceiling, compute the ceiling first.

> **A number seen once is an observation; a number seen across sessions is a
> property.** The 0.87–0.92 figure was a single-session excursion that a section
> heading ("Live defect found in our own shipped kernel") promoted to a durable
> claim, where it sat as an unassigned work item until someone re-measured it.

