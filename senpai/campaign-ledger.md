# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

Observed from Yukon and the organizer remote at `2026-08-17T11:46:41Z`.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `0b071ed9db211f17554bc5a13fb7381f14d709b3` |
| Best promoted submission | `ba493f74-c0fe-440a-a956-f77d26232e54` |
| Promoted source ref | `156b5b75bdfac82ae406487f531fd991e7fdfd30` |
| Official score | `2.95338624520432` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `156b5b75bdfac82ae406487f531fd991e7fdfd30` |

The promoted receipt above is the public Yukon frontier used to bootstrap this
campaign; it is not claimed as a Senpai-authored result.

Campaign commit `d098212` imports that exact promoted submitted surface. The
source is based on trusted organizer parent
`d077e68567827e8d926272df2245226d72b889ac`; relative to the previous promoted
`7351e62674bc600f0ca148d3a1b0604716a09db6` snapshot, its submitted delta spans
eight editable files (`+217/-41`), including the adaptive schedule, target and
head-state paths, quantized kernel twin, and proposal-head manifest. Trusted
base ancestry was reviewed separately and was not imported as solver code.

Campaign commit `8b85909` then reapplies the fixed-window post-EOS continuation
required by the current parent-owned 512-token contract. That overlay is not
part of the promoted Yukon receipt above and must pass exact 512-token replay
and official validation before it can itself be called promoted.

**Plausibility ceiling: `5.0`.** Raised from `3.0` by operator commit
`a5854b979499800a6f5f71a8d4fc14fd43ca4723` (2026-08-17, `AGENTS.md` +
`senpai/program.md` only) and readable at
`benchmark.json /scoring/decodeSpeedupCeiling` on base `b85e782`. It is a
fail-closed administrative gate, **not** a stop target and not an optimization
target (`senpai/program.md:21`). Headroom from the promoted `2.95338624520432`
is `+2.047` score — at the stale `-0.4335` calibration roughly `4.7` s off a
`~12.05` s candidate leg, i.e. about 39% of the whole MTP leg. No lever measured
this campaign is within an order of magnitude of that, so the ceiling changes
nothing operationally except that a large legitimate result must not be held
back. Docs corrected this session:
`research/ESTABLISHED_FACTS.md` and `research/CURRENT_RESEARCH_STATE.md` were
still stale at `3.0` and at the superseded `2.904` frontier.

## Base `b85e782`: what moved, and what students must re-derive

**The advisor branch has since advanced to `422db045`, and every step was
scored-path-inert.** Progression and verified deltas:
`b85e782` -> `e7cd780` (defect closure: tests and docs, 0 editable files) ->
`3c9317da` (merge PR #18, 13 files, **0 editable**) -> `af80b0fc` (merge PR #16,
0 editable, 0 source) -> `422db045` (laguna-map ceiling correction, 1 doc file).
The programmatic check that matters:
`git diff --stat b85e782 af80b0fc -- Sources/ Vendor/ benchmark.json fixtures/ .github/ Package.swift Package.resolved tools/ mtp-head.manifest.json`
is **empty**, and the two commits after it touch only `Tests/` and `senpai/`.
So every editable-path fact in this section still holds verbatim at `422db045`,
and a student rebasing from `b85e782` onto `422db045` inherits **no arm
information** — only a habitable test target and corrected documentation.

`b85e782` is the merge of the promoted-frontier sync line with the campaign
overlay line. `d098212` and `e6e6f81` are **divergent** parents, both ancestors
of `b85e782`; neither is an ancestor of the other. Measured editable deltas:

| From | Editable files changed | What they are |
| --- | --- | --- |
| `d098212` | 1 | `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (+24/-53) — the entire scored-path delta above the promoted frontier, i.e. exactly the fixed-window overlay |
| `ef16dea4` | 3 | the above, plus `Vendor/.../mlx-generated/quantized.cpp` (+5/-1) and `Vendor/.../kernels/quantized.h` (+15/-1) |
| `e6e6f81` | 3 of 7 changed files | same three; the other four are `Sources/MLXFastCLI/main.swift`, two test files, and one deleted test file |

**Two paired frontier changes ship together and must be read together.** The
promoted sync moved `segmentedStreakGate` from `3` to `2` *and* moved
`qmv_fast_crossrow_affine4_g64_m<T, 8, 4>` to `<T, 8, 3>` (a 3+3+2 split, not
4+4). The organizer's own in-tree comment says why they are one change: *gate 2
reaches the width-8 verify sooner, so that kernel fires more*. Verified gate
values: `ef16dea4` = 3, `e6e6f81` = 3, `b85e782` = 2 (gate 2 was already in
`d098212`).

Consequence, and it is load-bearing: **any result measured on `ef16dea4` or
`e6e6f81` had both halves of a deliberately-paired change at the wrong setting.**
That invalidates depth-8 / M=8 arithmetic, draft-depth histograms, and
`h(8)` estimates taken on those bases — not by making them wrong measurements,
but by making them measurements of a superseded configuration. It also
independently corroborates E14's register-cliff mechanism from the organizer's
side: `M = 9` profiles cheaper than `M = 8` (319 / 437 / 216 us for M = 7/8/9)
because the even split of 8 needs two simultaneous `vec<float,4>` accumulators.

## Cross-cutting defects found and closed this session

**1. The post-EOS `notBegun` defect — CLOSED, and the fix is alignment, not a
deviation.** The editable session at `b85e782` has no stop-token behaviour at
all: `reachedStopToken` is a constant `false` (`Qwen36MTPBlockSession.swift:167`),
the initialiser takes `stopTokens _: Set<Int>` as an ignored parameter (`:171`),
and `acceptedDraftPrefixCount` (`:672-679`) is a pure longest-common-prefix with
no EOS case. `stopTokens` occurrences dropped from 7 at `ef16dea4` to 1.

The decisive evidence is that **the trusted harness never had the defect**:
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` (non-editable) owns
the window by count — `:121 while emitted.count < options.totalTokenCount`,
`:141 let remaining = options.totalTokenCount - emitted.count`, overshoot
truncated at `:210-211`, and `grep -c reachedStopToken` on it is **0**. Its
`:173-176` comment calls `stopTokenInsideWindow` a defence of the *old*
normalised denominator. `benchmark.json /scoring/mtpEmptyDraftRoundsLegalNote`
states the same rule in the organizer's words. So the overlay is mandatory
alignment with the organizer's own driver. It is **insurance, not speedup**: it
enables legs that would otherwise fail, and it does not move the score. Submitting
the bare advisor base therefore banks nothing over the promoted receipt.

**2. The frontier-sync merge silently deletes our tests — a defect, twice.**
`Tests/` went from 58 files at `e6e6f81` to 57 at `b85e782`. Both losses are
delete/modify conflicts resolved the wrong way:

- `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` — deleted by `bc552e5`
  ("Retire the orphaned fixed-window EOS guard test"). This is the **fourth**
  time this literal has been removed: `f1a874d` added, `330b44e` reverted,
  `b219009` re-added, `bc552e5` deleted. Ledger row for `8b85909` below cites
  "focused `QwenMTPFixedWindowTests`: 2/2 passed" for a file that was not in the
  tree — a ledger/tree inconsistency caused entirely by this defect.
- `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` — 722 lines at `ef16dea4`,
  `e13a6fe` and `e6e6f81`; **absent** at `d098212` and `b85e782`; 792 lines on
  thorfinn's PR #16 head. It carries `QwenQMVCostCurveTests` and
  `QwenQMVParityTests`, including the bit-exact QMV parity suite that found the
  parity false-pass hole. PR #16 is the recovery vehicle and its r2 exit
  criterion is to KEEP the 792-line file at the conflict.

Standing rule from this: **a frontier sync may not reduce the test-file count.**
Compare `git ls-files Tests | wc -l` across the merge before publishing, and
resolve every delete/modify conflict in `Tests/` in favour of keep.

**3. The QMV parity harness could false-pass.** Found by thorfinn (E14) and
worth more than his negative result: a stale metallib meant an arm could be
"proved" bit-exact without the arm's code having been built. Hardened by
`68279a7` (stale-metallib audit) and `4a4014e` (rebuild around arm switches),
with a `perturb` positive control (x1.015625f) that must fire — it fired 56/96.
Any exactness claim made before that hardening is unverified. This generalises
the existing rule that `benchmark-qwen-mtp.sh` contains no `swift build`: the
robust defence is a per-arm `sha256` of the built worker recorded in metadata.

**4. `b85e782` did not compile its own test target — FIXED.** The frontier-sync
merge reformatted a multi-line string literal in
`Tests/MLXFastTests/QwenMTPVerbTests.swift` into a two-literal `+`
concatenation. Swift Testing's `#expect(_:_:)` takes `Comment?`, which is
`ExpressibleByStringLiteral` but **not** constructible from a `String`
*expression*, so the result was a hard error — `cannot convert value of type
'String' to expected argument type 'Comment?'` — and **the whole
`MLXFastTests` target failed to build at `b85e782`**. The change was pure
collateral reformatting: it is not part of the ceiling raise that the same
merge legitimately carried in `QwenMTPTrackNamingTests.swift`. Nothing was
wrong at `e6e6f81`, where the same message was a `"""` literal with a
backslash continuation; restoring that form fixes it. Verified: `swift build
--build-tests --force-resolved-versions` exit 0 in 12.6 s afterwards.

This one was load-bearing for the whole slate. Every student had just been told
to rebase onto `b85e782` and re-run the correctness gates, and askeladd's r2
exit criteria in particular ask him to reproduce a `swift test` comparison
against the base — all of which would have failed to build for reasons having
nothing to do with their work. Standing rule: **publish no advisor base without
first running `swift build --build-tests --force-resolved-versions` on it.** A
green student `swift test` at the *previous* base is not evidence about the new
one. Note also that `Tests/` is outside `editablePaths` (`AGENTS.md:197`), so
test restoration and repair cost nothing against the 3,000,000-byte source
budget or the 262,144-byte growth budget.

## Re-verified after the merge — and one half of it was WRONG

Two standing conclusions were checked against `b85e782` because the merge
touched `quantized.cpp` and `quantized.h`. **One survived; one was refuted by a
deeper trace on 2026-08-17 (advisor). The refuted text is corrected below rather
than deleted, because the way it was wrong is the reusable lesson.**

### REFUTED: "scored prefill takes `qmm_splitk` and never reaches `qmm()`/NAX"

What the earlier entry claimed: our transposed non-batched (`B == 1`)
projections take `qmm_splitk()` and `return` at `quantized.cpp:1418-1424`
before `qmm()` is ever called, therefore the prefill dequant prize is a
`qmm_splitk` problem and no NAX path is reachable on scored prefill.

**Both halves of that are false.** The outer dispatch does take `qmm_splitk`
at `:1418-1424`, but `qmm_splitk` (defined at `:776`) *delegates straight back*:

```
if (split_k <= 1) {
  return qmm(x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
}
```

`split_k > 1` requires `n_tiles * m_tiles <= 256` with `bm = bn = 32`. At the
scored prefill `M = 512`, `m_tiles = 16`, so it requires `n_tiles <= 16`, i.e.
**`N <= 512`**. Every scored prefill projection has `N >= 3072`:

| shape | M | N | K | n_tiles | m_tiles | split_k | actually runs |
|---|--:|--:|--:|--:|--:|--:|---|
| `mlp.gate_up_fused` | 512 | 17408 | 5120 | 544 | 16 | **1** | delegates to `qmm()` |
| `full_attn.qkv_proj_fused` | 512 | 7168 | 5120 | 224 | 16 | **1** | `qmm()` |
| `linear_attn.in_proj_fused_qkvzba` | 512 | 8192 | 5120 | 256 | 16 | **1** | `qmm()` |
| `mlp.down` | 512 | 5120 | 8704 | 160 | 16 | **1** | `qmm()` |
| `full_attn.o_proj` | 512 | 5120 | 3072 | 160 | 16 | **1** | `qmm()` |
| `head.lm_head` | 512 | 248320 | 5120 | 7760 | 16 | **1** | `qmm()` |
| control | 512 | 512 | 5120 | 16 | 16 | 2 | `qmm_t_splitk` |
| control | 512 | 128 | 5120 | 4 | 16 | 8 | `qmm_t_splitk` |

⇒ **the `qmm_t_splitk` kernels are dead code on this model**, and scored prefill
really runs `qmm()`. `qmm()` opens at `:684` with a NAX early return at
`:697-699` gated on `is_nax_available() && transpose && (K % 64 == 0) &&
(env::enable_tf32() || x.dtype() != float32)`. Every term except
`is_nax_available()` holds for our shapes (`transpose` is passed literally
`true`, `K % 64 == 0` for 5120/8704/3072, `x` is bf16). **So NAX *is* reachable
on scored prefill whenever the ranked host satisfies `is_nax_available()`.**

Consequence, and it is a first-order hazard: alphonse's E16 prefill dequant
overhead of **12.942% of `P`** was measured on the **non-NAX** path. If ranked
M5 satisfies `is_nax_available()`, the ranked prefill GEMM is `qmm_nax`, and
that number — plus any local optimization of `qmm()` — may transfer at zero.
The 11 `_nax` files are editable, and `AGENTS.md:259-260` makes them
first-class targets. This is why E18 (thorfinn, PR #20) leads with a host-only
reachability determination instead of a kernel edit.

**Process lesson, now a standing rule: a call-site trace is not a call-graph
trace. Follow every delegating `return` to a fixed point before concluding that
a function is never reached.** The earlier entry stopped one level too shallow
and read `qmm_splitk(...); return;` as terminal.

### SURVIVES: the decode QMV path is non-NAX on both hosts

`quantized_nax.h` (1681 lines) contains zero `qmv`, zero `qmv_fast`, zero
`crossrow` and zero `affine4` — only `affine_qmm_{t,n}_nax`,
`affine_gather_qmm_*_nax` and the two `qmm_*_nax_tgp_impl` helpers, against
`quantized.h`'s 2981 lines carrying `qmv_fast` ×28 and `crossrow` ×22. There is
no NAX competitor to `qmv_fast_crossrow_affine4_g64_m`, so thorfinn's cross-row
results and the frontier's `<T,8,4>`→`<T,8,3>` change stay on the ranked path.
This is now also confirmed *analytically* rather than only empirically:
`get_qmv_batch_limit(D, O, d)` returns **10** for our `D = 5120 > 4096` on the
local `default` arch-size branch, matching PR #5's measured `vector_limit = 10`.
Verify runs at `M <= 9 < 10`, so it never enters `qmm`/`qmm_splitk` at all —
which is why the refutation above does not disturb any decode conclusion.

`is_nax_available()` itself (`device.cpp:913`) is unchanged: a compile-time
`#ifdef MLX_METAL_NO_NAX -> return false` escape, otherwise a memoized `static
bool` requiring `__builtin_available(macOS 26.2, ...)` **and** `gen >= (arch ==
'p' ? 18 : 17)`. Local `applegpu_g16s` parses to gen `16`, suffix `s`,
threshold `17`, so NAX is **off locally** despite local macOS 26.5.2. It cannot
be forced on locally; only off, and only at compile time.

## The `key_len = 1024` fidelity hazard now has a mechanism

The campaign-level `key_len = 1024` positional residual
(`CURRENT_RESEARCH_STATE.md:2798-2809`, `ESTABLISHED_FACTS.md:1749`) was an
unexplained observation: PR #2 found **919/919 non-terminal width-9 rows
bit-exact**, with all 15 value mismatches in the final block at positions
**1022–1024**, and widths 2, 4 and 8 drifting there too. Four independent
observations said "positional, not width-driven". The mechanism is now located
and it is exactly positional:

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:743-753`

```
// We route to the 2 pass fused attention if
// - The device is large and the sequence length long
// - The sequence length is even longer and we have gqa
bool do_causal = do_causal_ && q.shape(2) > 1;
char devc = d.get_architecture().back();
if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
    (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
  sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
} else {
  sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
}
```

`k.shape(2)` is the key/cache length. **At `key_len >= 1024` the kernel family
changes from single-pass `sdpa_vector` to `sdpa_vector_2pass`, which computes a
blockwise partial softmax and combines partials in a second kernel — a
different floating-point reduction order over the same scores.** The ranked leg
is 512 seed + 512 decode, so `key_len` reaches 1024 at the very last decode
steps and nowhere earlier. That is precisely where, and only where, the
mismatches were seen.

Why this also explains the *width* observations: in single-pass `sdpa_vector`
the key loop is `for (int i = simd_gid; i < N; i += BN)` with `BN = 32`, a
reduction order that is a function of `N` alone and **independent of `qL`** —
which is why all widths agreed bit-exactly for 919/919 rows below the boundary.
In `sdpa_vector_2pass` the block count depends on `n_simds = gqa_factor *
q.shape(2)` (`:440-476`), so `qL` enters the reduction geometry for the first
time at the boundary.

Two facts that decide whether this matters for the score, neither yet
established, and they point in opposite directions:

- The `>= 1024` trigger is gated on `devc == 'd' || devc == 's'`, the **last
  character of the GPU architecture string**. Local is `applegpu_g16s` → `'s'`
  → it fires locally. If ranked M5 reports a suffix that is neither `d` nor
  `s`, the only surviving trigger is `k.shape(1) < q.shape(1) && k.shape(2) >=
  4096`, and a 1024-token ranked window never reaches 4096 — in which case
  **the residual is a local-only artifact and the item closes as not useful.**
- If ranked M5 *is* `d`- or `s`-suffixed, the scored leg changes attention
  kernel family inside the scored window, under a
  `trusted-sequential-reverification-exact-token-match` gate. A value drift
  that flips one near-tie argmax fails the whole leg.

Editability map for anyone who works this: the dispatch file
`backend/metal/scaled_dot_product_attention.cpp` is **NOT** editable, and
neither is `device.cpp`. `kernels/sdpa_vector.h`,
`kernels/scaled_dot_product_attention.metal` and
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` **are**
editable. `sdpa_vector.h` has **no** `mlx-generated` twin (only
`steel_attention.cpp` and `steel_attention_nax.cpp` exist there), so it is
AOT-only and any edit needs `tools/build-mlx-metallib.sh --all-build-roots`.
`AttentionUtils.swift` already contains the precedent for a Swift-level fix:
the WIDE-DECODE EXACTNESS CHUNK at `:104-142` splits queries at row 5 for
`6 <= qL <= 9` specifically to hold a dispatch decision steady, with
`supports_sdpa_vector` requiring `qL <= 8` and `qL * gqa_factor <= 32`
(`scaled_dot_product_attention.cpp:634-637`).

There is a free diagnostic lever: `:477` reads
`env::get_var("MLX_SDPA_BLOCKS", 0)` and overrides the block count when
positive. That changes the number of partial accumulators **without editing any
kernel**, so it can prove or refute reduction-order causation cheaply. It is a
*diagnostic only* — `MLX_`-prefixed variables are structurally unreachable in
the ranked worker (named blocker #1), so it can never be part of a submitted
fix.

## Scoring bounds: two different numbers

The plausibility ceiling is enforced at two levels and they are not the same
value. `benchmark.json:201` `decodeSpeedupCeilingNote`: the **published-median**
ceiling moved `3.0 -> 5.0` on 2026-08-17, and the box wrapper's **per-pair**
bound `MAX_PLAUSIBLE_SPEEDUP` moved `5.0 -> 8.0` in the same decision, "so the
aggregate ceiling stays strictly tighter than the per-pair bound"
(`.github/workflows/qwen-mtp-ranked-benchmark.yml:467` carries the same note).
Since aggregation is the median of per-prompt raw ratios, a single prompt may
legally land as high as `8.0` while the median it contributes to must stay at or
below `5.0`. The organizer's own stated reason for `5.0` being generous is worth
quoting when judging a suspicious number: "the authors' own exact-greedy
envelope tops out at 1.74x at a 32-token window, kernel work on this tower has
historically bought tens of percent, and 5.0 leaves room for both compounding."
Our promoted `2.95338624520432` is already well above that 1.74x envelope, which
is the honest measure of how much of this campaign's frontier is kernel work
rather than acceptance.

## Same-host baselines

| Base SHA | Host / memory profile | Toolchain | Head provenance | Command | Key metrics | Evidence location |
| --- | --- | --- | --- | --- | --- | --- |
| `7351e62674bc600f0ca148d3a1b0604716a09db6` | AWS Birch/Alphonse; Apple M4 Pro, 48 GB; automatic low-memory profile | macOS 26.5.2; Xcode 26.6; Swift 6.3.3 | pinned head SHA-256 `c3f8a09b3c2ff1a9b40c2c1a5f71236e2e57be31f861270c071e7ba909e18e64` | `MLXFAST_QWEN_MTP_LOCAL_WORK_DIR="$PWD/.mlxfast-local-qwen-mtp" yukon run` | pass; directional `1.4708805115725638`; exact `64/64`; serial `0.1292338595` s/token; MTP `0.0878615621` s/token; effective draft `5.4`; acceptance `1.0`; divergences `0` | ignored `score.aws-birch-alphonse.7351e626.baseline.json` (SHA-256 `0f166cdfcf0b3e1f33a438de5012c9e865c8c33ed1b7a20cc881a859eadc3b83`) and `local-docs/baselines/aws-birch-alphonse-7351e626/` |

That baseline ran on the detached promoted source. Its complete submitted
surface is identical to campaign import commit `ce159755`.

**That baseline is now stale for A/B use.** It was taken on `7351e626`, several
promotions back, and the scheduler moved from `0.20 / 4 / 7 / gate 3` to
`0.18 / 5 / 8 / gate 2` in between. No candidate may be compared against it on
this base; a fresh same-host baseline on the current advisor merge is required
first, and every comparison must be a matched pair measured in the same
serialized window.

**Measurement hygiene, from a 1000 s idle thermal soak on this host.** Idle GPU
settles near `38.7` to `40` °C at about `0.02` W and recovers within roughly
`50` s, but the soak also caught recurring foreign spikes to `65` to `83` °C at
`16` to `31` W (`t ~ 181-244`, `575-744`, `871-922` s). Those are *other
agents' GPU work on the same host*, and they corrupt timing. The standing rule
is therefore: **parallelize builds and analysis freely, but serialize every
timing measurement**, and sample temperature around each timed arm so a
contaminated arm can be identified and discarded rather than believed.

## Official campaign submissions

| Submission ID | Candidate SHA | Base SHA | Model | Score / status | Public note | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| _No Senpai campaign submissions yet._ | | | | | | |

## Novelty index

Use one row per causal mechanism, not per branch. `Reopen when` must name new
evidence or a changed condition; “try again” is not enough.

| Mechanism | Scored path / cost center | Hypothesis | Best evidence | Disposition | Reopen when |
| --- | --- | --- | --- | --- | --- |
| Exact-fill first K/V-cache retention | `KVCacheSimple.update`; charged 512-token seed in 16 full-attention layers | Retaining incoming K/V when an empty cache receives an exact multiple of its 256-row step removes zero-fill and full-slice replacement without changing shape, offset, or layout | Stock code allocates exact-size zero arrays and then overwrites every element; the transfer map specifies boundary and rollback tests | untested; ready | Close after exact 512+1...9, snapshot/rollback, layout, and matched seed timing; reopen only after cache/step changes or fresh profiling |
| Packed GDN prework for S=1...2 | 48 GDN layers on serial, adaptive-skip, and narrow verify calls | A small-width variant that explicitly sources required old conv-state rows can extend the promoted S=3...9 packed mixer | The current mixer passed exhaustive campaign checks; source inspection locates the S=2 boundary at old conv-state ownership | untested; medium risk | Proceed only with explicit old-state construction; stop on any cache/output mismatch or less than roughly 0.25 ms named-call saving |
| Corrected packed GDN beta | Remaining `sigmoid(b)` launch/intermediate on promoted S=3...9 mixer path | Produce beta inside the packed kernel with an exact correction for finite BF16 input `0xC0DB` | Prior exhaustive BF16 comparison found one 1-ulp exception; the remaining launch is small but repeated | untested; exactness constrained | Require all 65,536 BF16 encodings, including NaN/Inf, plus exact recurrence/cache proof; close if correction cost has no end-to-end payoff |
| Compiled fused-SwiGLU causal isolation | `Qwen35FusedMLP`, S<=9 target/head path | The compiled slice-SiLU-product expression removes one launch/intermediate and may contribute independently | Present in promoted `7351e626`; the four-hunk composite improved `2.876429` to `2.904211`, but no component was isolated | promoted in composite; individual sign unresolved | Run a matched `7351e626` on/off ablation before extending or composing it; require movement outside matched noise or direct structural evidence |
| Pair GQA query heads per K/V read | Short-query SDPA in 16 full-attention layers; six query heads share each KV head | Processing two Q heads per K/V read can reduce duplicated traffic while retaining independent per-head accumulation order | Qwen geometry exposes a 6:1 reuse seam; broader grouped-SDPA work has hit M5 register/pipeline limits | unresolved; resource-gated | Reopen only when a D256 pair-head kernel compiles on M5 with acceptable registers/occupancy and exact per-head outputs |
| Seed-prefill wall time inside the charged window | 512-token seed prefill, charged on **both** arms of every paired prompt | Because `raw_p = (P + D_s)/(P + D_m)`, cutting `P` raises every per-prompt ratio with no effect on any scored token | **CLOSED by E16 (PR #18) with `closure_error_seconds = 0`.** Prefill is 99.94% GPU: true CPU graph construction is `1.8` ms = `0.045%` of `P`, and `build_us` is enqueue back-pressure, not work. Budget: GEMM-at-ceiling `3.369302` s (84.148%) + non-GEMM `0.212714` s (5.313%) + residual `0.421984` s (10.539%); measured dense-bf16 ceiling `7.401388` TFLOP/s with GEMM achieving `6.414787` = 86.67% of it. Best interior schedule `list:0,1,2,5,11,23,47` moves serial prefill to `3.993803` s = `0.2547%` of `P`, `5.9x` below the `1.5%` bar | **closed; scheduling is dead, dequant is the residual** | Do not reopen the ladder/schedule. The one live term is **dequant overhead `0.518202` s = 12.942% of prefill = `0.090180` pts ~ 7.3 frontier steps**, minus a `0.096218` s overlap credit. That is a GEMM question, not the cross-row QMV decode path. **Corrected 2026-08-17: the GEMM in question is `qmm()` (`backend/metal/quantized.cpp:684`), not `qmm_splitk` — see the REFUTED subsection above** |
| Prefill dequant overhead | `qmm()` at `backend/metal/quantized.cpp:684`, or `qmm_nax` at `:473` if ranked M5 satisfies `is_nax_available()`, inside the charged 512-token seed prefill | The 12.942% of prefill spent on dequantisation is the single largest closed-budget residual on the scored path; removing even half of it is worth ~3.6 frontier steps | E16's closed budget (`closure_error_seconds = 0`, pipelined graph reproduces worker wall to 0.2%, build cost `0.00136` s); alphonse's attribution | **untested; highest expected value now; assigned as E18 (PR #20, thorfinn)** | Open now. Two falsifications must land before any kernel edit. (1) **Transfer**: E14's findings are all cross-row QMV (`qmv_fast_crossrow_*`, decode, `M<=9`); the prefill GEMM tiles at `bm = bn = 32` and amortises weight reads across 32 output columns, so the register-cliff mechanism has no obvious purchase — assume nothing carries over. (2) **Host**: `is_nax_available()` is false locally (`applegpu_g16s` -> gen 16, suffix `'s'`, threshold 17) but may be true on ranked M5, in which case the scored prefill GEMM is `qmm_nax` and alphonse's 12.942% — plus any local optimisation of `qmm()` — transfers at zero. Resolve the host question host-only before spending GPU |
| Compact draft-head readout precision | proposal-head readout matmul, one call per draft step | Reading the compact draft vocabulary at 3 bits instead of 4 removes 22.22% of the readout's bytes on a bandwidth-bound call | E15 (PR #17): bytes 4->3 `-22.22%`, time `-24.32%` (`1.16635` -> `0.88271` ms), bandwidth `+2.77%`, delta `283.64` us/readout, same `qmv_fast_impl` on both. End-to-end at 256 tokens: MTP steady s/token `-0.9983%`, all four legs exact, acceptance term exactly zero (byte-identical depth schedules over 35 rounds, requant argmax-lossless over all 230 proposals). Per-round attribution agrees with the microbenchmark to 1.4% | **local winner; strongest clean exact candidate; awaiting 512-token ABBA on `af80b0fc` or later** | Blocking issues are measurement, not mechanism: a `7.96` degC thermal gap between arms, a 256-token window where the contract requires 512, and a base that has since moved twice (`b85e782` -> `af80b0fc` -> `422db045`; the scored path is byte-identical across all three) |
| `key_len = 1024` SDPA two-pass dispatch boundary | `scaled_dot_product_attention.cpp:743-753` inside the scored 512-seed + 512-decode window; every full-attention layer | The positional exactness residual at decode positions 1022-1024 is a **kernel-family change**, not a width effect: at `k.shape(2) >= 1024` on a `'d'`- or `'s'`-suffixed GPU the dispatch switches `sdpa_vector` -> `sdpa_vector_2pass`, which changes the floating-point reduction order of every score | Four independent confirmations that the residual is positional, not width-driven (PR #2): 919/919 non-terminal width-9 rows bit-exact; all 15 value mismatches in the final block at 1022-1024; widths 2, 4 and 8 drift there too; the 128-token local-submit window ends at 640 and is clean. Width 4 is **outside** the `AttentionUtils` `6 <= qL <= 9` split, which exonerates that path. Mechanism located this session: single-pass `sdpa_vector` reduces with `BN = 32` over `N` alone so its order is **independent of `qL`** (hence 919/919 agreement below the boundary), whereas `sdpa_vector_2pass` sets its block count from `n_simds = gqa_factor * q.shape(2)`, so `qL` enters reduction geometry for the first time exactly at `N = 1024` | **mechanism located; consequence UNRESOLVED and host-dependent** | The single decisive fact is the **last character of the ranked M5 architecture string**. The `>= 1024` trigger is gated on `devc` in `{'d','s'}`; local `applegpu_g16s` fires, and no M5 arch string is recorded anywhere in the repo. If M5 is neither, the only surviving trigger is `k.shape(1) < q.shape(1) && k.shape(2) >= 4096`, which a 1024-token ranked window never reaches, and this **closes as a local-only artifact / not useful**. If M5 *is* `'d'` or `'s'`, the scored leg changes attention kernel family inside an exact-token-match gate and one flipped near-tie argmax fails a whole leg. Editability: the dispatch `.cpp` and `device.cpp` are **not** editable; `kernels/sdpa_vector.h`, `kernels/scaled_dot_product_attention.metal` and `MLXLMCommon/AttentionUtils.swift` are. `sdpa_vector.h` has **no `mlx-generated` twin**, so it is AOT-only and any edit needs `tools/build-mlx-metallib.sh --all-build-roots`. `MLX_SDPA_BLOCKS` (`:477-478`) changes the partial-accumulator count with no kernel edit but is **diagnostic only** — `MLX_`-prefixed vars are named blocker #1 |
| Compile-time group width `NA = 4` cliff | Cross-row QMV in the proposal head; `mtp-head.manifest.json` now declares 4-bit/group-64 | Something about `NA = 4` specifically, most plausibly register pressure or spilling, makes cross-row contraction regress; occupancy was refuted and the student withdrew the chain-depth story | E10 partitions **exactly** on compile-time group width: every `M` whose NA set contains 4 regresses, none without NA=4 does, zero overlap; ordered variant is bit-identical to control on all 96 cells, `max_abs_delta = 0`. E14 adds the register accounting: `sizeof(vec<float,5>) == 32` vs `16` for `vec<float,4>`, 13 NA-wide vectors per thread so 39 at NA=3, 52 at NA=4, 65 at NA=5, 104 at NA=5 padded; E13 found NA=5 compiles free with first spill at NA=6. **The organizer independently confirms the cliff**: the frontier moved `<T,8,4>` to `<T,8,3>` because the even split of 8 needs two simultaneous `vec<float,4>` accumulators, and `M=9` with three-lane vectors profiles cheaper than `M=8` | mechanism OPEN but now corroborated from two independent directions; magnitude ceiling about `1%` of crossrow QMV time only | Reopen by reading register and spill counts out of compiled AIR, not threadgroup size; now more relevant because the frontier head is affine4/g64 |
| Cross-row second weight pass | `qmv_fast_crossrow_*` in the proposal head; verify widths M=2..9 | Eliminating the second weight pass, or the row/NA-width tax, should recover a large fraction of verify time | **Measured and dead (E14, PR #16).** The second weight pass is worth only `+8.16%` drift-adjusted (`0.1161` h-units) because **~89% of it is cache-served**; per-shape excess is monotone in footprint (`head.lm_head` 682 MiB -> 12.07% down to `full_attn.o_proj` 16.88 MiB -> 1.32%), and structural `0.1115` agrees with interventional `0.1161` to 4%. One verify row at constant pass count costs ~`0.27` depth-0 rounds vs ~`0.11` for the pass, so the **row/NA-width tax is ~2.4x the weight-stream tax**. Arm A (`_m<T,5,2>`) is `+39.3%` slower at M=5 *and* fails parity 8/96; arm E (scalar `float[NA]` packing) is bit-identical (0/96, byte-identical output, sha256 `9e3c52a3df97856e...`) but a reproducible `+12.4%` regression | **negative; closed as a speedup, green as measurement** | Do not reopen the weight-pass framing. The tax is rows/width, not streaming — any future cross-row work must target row count |

## Experiment receipts

| Date | Branch / candidate | Mechanism | Base SHA | Local result | Official result | Result record |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | clean Yukon source / `7351e626` | untouched promoted-tree baseline | `7351e62674bc600f0ca148d3a1b0604716a09db6` | pass; M4-local directional score `1.4708805115725638`; public tripwire passed; exact `64/64` | not submitted; local result is non-rankable | Same-host baseline row above; submitted surface now matches campaign import `ce159755` |
| 2026-08-16 | `codex/sync-organizer-frontier-20260816` / `ce159755` | exact promoted editable-snapshot import | `eb2dc26caf48ac126e0f51df7db5130414ff1d94` | release build, overlay, budget, twin, and trusted-parity checks passed; full `swift test` reached product compilation but was blocked by unchanged organizer test-source type error at `QwenMTPVerbTests.swift:755` | adopted public promoted `e6c5ef35` at `2.9042110287045`; not a Senpai-authored submission | Source delta is exactly two editable Swift files (`+54/-14`); campaign records and novelty queue refreshed separately |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `d098212` | exact promoted editable-snapshot import | `83201aa98a71d42415e1c7e85e8bc96cf609d5cf` | preservation, overlay, and budget checks passed; inherited `quantized` twin comment drift recorded without changing promoted bytes | adopted public promoted `ba493f74` at `2.95338624520432`; not a Senpai-authored submission | Exact organizer source `156b5b75`; eight editable files changed relative to the prior promoted source |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-5` / `8b85909` | fixed-window continuation after EOS on the new promoted source | `d098212` | focused `QwenMTPFixedWindowTests`: 2/2 passed; full 512-token exact replay still required | not submitted; not promoted | Restores the campaign's parent-owned fixed-window behavior without altering the trusted fixture or parent. **The suite this row cites was deleted by the merge that produced `b85e782`; restored and extended on the advisor branch (see defect 2 above)** |
| 2026-08-17 | `qwen-thorfinn/ipg-weight-passes` / `852af07f` (PR #16, E14) | cross-row IPG weight passes: can the second weight pass or the NA-width tax be removed? | `ef16dea4` | **terminal negative.** 11 W&B runs in `qwen38-mlx-challenge-senpai` (`88khsek3` `bfk6o414` `97ieuck5` refs, `tu839z8z` armB, `e62r389y` armA, `a10cxpfs` armA3, `qnwqdh03` armD, `md5dlsm0` armE, `fdus9cxa` armE2, `2qvqo4z8` cross-arm, `sxau0sjl` Q4). Second pass `+8.16%` (~89% cache-served); row tax ~2.4x pass tax; arm A `+39.3%` at M=5 and 8/96 parity failures; arm E bit-identical but `+12.4%` slower. Measured `h`: 0.2537/0.3816/0.2735/0.2771/0.2759 for d=3..7, three independent h4 = 0.3816/0.3764/0.3880 (mean 0.3820 +-1.5%, shipped fit 1.7% low). Depth-4 break-even needs `h3 <= 0.3511` = a 6.48% cut = 1581.4 us off the 24404.5 us fourth draft step; threshold 1.0693 unreachable over 400,000 random vectors | not submitted; zero submitted-path files | `research/results/qwen38-r1-e14-ipg-weight-passes.md`. **Not useful as speedup / green as measurement.** Byproducts worth more than the result: the QMV parity false-pass hole (defect 3), the min-of-regions estimator (control noise +-0.5% -> +-0.27%), and the arm-E finding that scalar packing recovers ~32 of arm A's 39 points |
| 2026-08-17 | `qwen-askeladd/draft-readout-3bit-default` / `de82bc37` (PR #17, E15) | compact draft-head readout at 3 bits as the compiled default | `ef16dea4` | **terminal succeeded; local winner.** Phase 1 (`n2xr3wx5`, cool gate PASSED, reps=21, 98,336x5,120 g64 affine): 4-bit `1.16635` ms / 283,207,680 B / 242.81 GB/s, 3-bit `0.88271` ms / 220,272,640 B / 249.54 GB/s, 2-bit `0.67003` ms; host STREAM peak `227,787,321,075.54` B/s. Phase 2 (`hgdke1uo`, 256 tokens): MTP leg s/token `-0.7272%`, steady `-0.9983%`, `measured_local_score` `+0.8723%`, control-serial-fixed modelled `+0.8379%`, peak RSS `-56.6` MB. All 4 legs exact, divergence 0, acceptance term exactly zero, requant argmax-lossless over 230 proposals and untimed (`117.007` ms once per process). Win is host-side: `draft_build_us` +1029 and `verify_build_us` +881 = 1910 of 1915 us, `eval_wall_us` unchanged (+0.008%) | not yet submitted | `research/results/qwen38-r1-e15-*`. **Advisor prediction 1 refuted** (both bit widths take the same `qmv_fast_impl`). Blockers before submission: `7.96` degC arm-to-arm thermal gap, 256-token window vs the contract's 512, and base moved. r2 requires an ABBA `4,3,3,4` at 512 pinned to `af80b0fc` or later |
| 2026-08-17 | `qwen-alphonse/prefill-ladder-adjudication` / `9116d435` (PR #18, E16) | adjudicate the seed-prefill ladder: is prefill CPU-bound, and what is the closed budget? | `e13a6fe0` | **terminal succeeded; verdict merge.** 7 W&B runs, 14 timed phases all with `all_tokens_matched=true`, `residual_divergence_count=0`, `declared_rows_total = emitted_token_total = 64`, `accepted_draft_rate=1`, `uses_pinned_mtp_head=true`. Ladder ON `build_us=2,957,503` / `eval_wall_us=1,046,892`; ladder OFF `build_us=1,796` / `eval_wall_us=4,004,115`; total `P` moves only `+0.0015` s serial. `closure_error_seconds = 0`. Best schedule `3.993803` s vs baseline `4.004000` s (`-0.0102` s). Q5 re-measure `4.002279` s sits inside the pre-merge band `[3.993803024, 4.007063985]` | not submitted; **decision: revert the `DARKBLOOM_QWEN_PREFILL_LADDER` block out of `Vendor/.../Qwen35.swift` and merge with zero submitted-path changes** | `research/results/qwen38-r1-e16-*`, `research/floor-e16.json`. Strongest closure this campaign has produced. Retracted his own E12 73.8%/26.2% CPU/GPU split in writing; resolved the impossible `23.808` TFLOP/s rate to `6.2235` TFLOP/s = 84.5% of ceiling; corrected E3's never-probed hardcoded `"mlx_python": "0.32.0"` to the actual mlx 0.29.3 / Python 3.9.6. The submitted-path delta is worth `0.00178` pts = `0.14` frontier steps and sits behind an `MLX_`-prefixed env var, i.e. named blocker #1 — deleting it is the cheap, honest resolution |
| 2026-08-17 | `qwen-edward/curve-transfer-and-refit` / `e3f31dea` (PR #19, E17) | does the merged depth curve transfer to this base, and does the h-fit need refitting? | `e6e6f81` | **terminal succeeded; accepted. Label: local winner (mechanism), not submittable.** First 512-token, 4-prompt, prefill-inclusive paired measurement in the campaign. Held-out n=3: `median(raw|CURVE) = 1.666207` vs `median(raw|FLAT18) = 1.560481`, delta `+0.105726 = +6.775%`, `g_median = +5.284%`, curve wins 3/3 (all four, even-n: `+5.986%`, 4/4; he reports the more conservative held-out figure). **Mechanism nailed**: `Spearman(g, extra verify rows %) = +1.000` (n=4) vs `+0.400` for rows-per-accept — the curve wins by **declining redundant target verification**, not by accepting more. Scalar reaches `d5`/`d4`; curve `max_d = 3` on all four prompts. Decode-currency -> score conversion `0.84228` on his base, cross-checked against E11 (`7.577% x 0.84228 = 6.382%` vs `6.378%` measured). All 8 arms clean: matched True, divergence 0, parity True, 512/512, `declared == checked` rows, `mlx_qwen_env` empty, `dirty = 0`, pinned head, drift tripwire True, stall `1.227-1.738x` against a `4x` guardrail. Contract-6 worker freshness proved distinct per arm (`1651e64e...` vs `bb7db942...`). Measured `h`: d=1 `M2->3` fit **1.49x low** (`.0775` vs `.1152`); d=0 `.0842` vs `.0971`; d=2 `.2426` vs `.2482`; d=3 `.3754` vs `.3761` | not submitted; zero submitted-path delta | `research/results/qwen38-r1-e17-*`, `research/e17-notes.md`. **His own self-criticism is the most valuable line**: the honest advantage over a *well-chosen* scalar is ~`+4.5%`, and a monotone refit is predicted neutral-to-worse, so the shipped curve's edge is "accidental risk-pricing at depth 2, not superior cost modelling — a fragile foundation." **Independently converges with E14 on depth 4 never being optimal at `q <= 1`, by a different method.** Timing budget banked: `17.69` min mean / `19.05` max per 512-token prompt-pair with 2 arms and 2 cool gates. Delivered the `research/e11-notes.md` retraction — debt closed. Four advisor challenges, three of which he won: he was right about the stale laguna-map ceiling lines (fixed in `422db045`), right that his frontier citation was a faithful read of his own base, and right that **the blinding leak was mine** — labelling a section "quarantined" does not un-send it |
| 2026-08-17 | advisor `senpai/qwen38-mtp-r1` / `e7cd780` | close four cross-cutting defects found while reconciling students against `b85e782` | `b85e782` | 5 files, `+591/-12`, zero editable/scored-path files. Restored `Tests/MLXFastTests/QwenMTPFixedWindowTests.swift` at 368 lines (10 tests in 2 suites, all passing) after its **fourth** removal; fixed the `Comment?` type error at `QwenMTPVerbTests.swift:751-757` that meant **the entire `MLXFastTests` target did not compile at `b85e782`**; corrected the frontier and ceiling in `research/ESTABLISHED_FACTS.md` and `research/CURRENT_RESEARCH_STATE.md` | not submitted; documentation and tests only | The build defect was independently found by thorfinn (his follow-up #9) who attributed introduction to `ee977ae` and left it out of scope. Consequence larger than either of us logged: **any green `swift test` reported at `b85e782` was impossible.** Standing rule added to the checklist |
| 2026-08-17 | advisor `senpai/qwen38-mtp-r1` / `3c9317da` then `af80b0fc` | merge E16 (PR #18) then E14 (PR #16) | `e7cd780`, then `3c9317da` | Both merged with `accept_result_on_current_base` first. `b85e782 -> af80b0fc` **scored path is byte-identical**: `git diff --stat b85e782 af80b0fc -- Sources/ Vendor/ benchmark.json fixtures/ .github/ Package.swift Package.resolved tools/ mtp-head.manifest.json` is empty. E14's merge recovered the 792-line `QwenQMVCostCurveTests.swift`, taking `Tests/` from 57 files to 59. Habitability proved at `af80b0fc`: `swift build --build-tests -c debug --force-resolved-versions` exit 0 in `9.65` s; the two recovered suites run `15 tests / 4 suites / 0 failures` in `4.99` s | not submitted | E14 was delivered as a **merge instead of a rebase** because `submit_experiment_result` enforces fast-forward against the remote head; deviation accepted after proving merge tree `57451dd90e8c74e029196d8d482a9bfb9eea860d` byte-identical to the true rebase `95929f9`. E16's `gpu_cores = 20` deviation also accepted — his `system_profiler` probe was right and my `10` was a mis-scraped CPU performance-core count |
| 2026-08-17 | advisor `senpai/qwen38-mtp-r1` / `422db045` | correct three stale ceiling/frontier passages in `senpai/laguna-to-qwen-speedup-map.md` | `af80b0fc` | 1 file, `+55/-8`, zero editable/scored-path files. Habitability re-proved: `swift build --build-tests -c debug --force-resolved-versions` exit 0 in `4.07` s. The executive-conclusion headroom paragraph now carries the live frontier, ceiling `5.0` with `a5854b97` provenance, headroom `+2.047` (~69% multiplicative), and the fail-closed-gate framing; the old endpoint table is retained verbatim under a **SUPERSEDED** banner beside a new live table | not submitted; documentation only | Found because edward challenged a *different* pair of files. His cited lines were already fixed at `e7cd780` — he was reading his own base `e6e6f81`, outside my ancestry, so his copy was legitimately stale — but the challenge made me re-scan, and the re-scan found real staleness elsewhere. **A challenge that is wrong on its face can still be right about the class of error** |
| 2026-08-17 | `qwen-thorfinn/prefill-dequant-prize` / `dbfef047` (PR #20, E18) | is the 12.942% prefill dequant residual reachable, and on which kernel? | `422db045` | **r1 assigned, in flight.** Phase 1 is host-only and takes no benchmark lock: reproduce or refute my `split_k` arithmetic, instrument the dispatch *decision* for the six scored prefill shapes, and determine from runner-image / workflow / arch evidence whether ranked M5 satisfies `is_nax_available()`. **Phase 1 alone is a complete, mergeable result**, and an explicit UNRESOLVED with a named missing fact is a full pass | not submitted | Brief opens with my own self-correction: I concluded the scored prefill takes `qmm_splitk` after tracing only the outer dispatch, and asked him to check me because "I got the call graph wrong once already in this area; assume I can be wrong again." His parity false-pass hole from E14 and his per-arm worker `sha256` are now campaign requirements |

## Update checklist

1. Confirm the exact base, candidate, organizer, and promoted source SHAs.
2. Add or revise one novelty row with the mechanism's disposition.
3. Add the result receipt and the public submission receipt, if any.
4. Update same-host baseline rows whenever the base, host, head, or toolchain
   changes.
5. Keep `frontier-state.json` synchronized whenever organizer or promoted
   frontier pins change.
6. Before publishing a new advisor base, prove it is habitable: run `swift build
   --build-tests --force-resolved-versions` to exit 0, and check that
   `git ls-files Tests | wc -l` did not fall across the merge. Resolve every
   `Tests/` delete/modify conflict in favour of keep. Both checks exist because a
   frontier sync silently broke the test target and dropped two suites; neither
   failure is visible from a student's green run at the previous base.
7. Re-verify, do not assume, any standing conclusion that depends on a file the
   sync touched. Record the re-verification explicitly so the next reader can
   tell a checked claim from an inherited one.
8. A call-site trace is not a call-graph trace. Follow every delegating `return`
   to a fixed point before concluding that a function is never reached. This cost
   the campaign a wrong prefill conclusion for a full round.
9. Publish the advisor branch *before* creating an assignment that should
   inherit a fix: `create_assignment` takes `expected_base_sha` from the branch
   tip, so an unpublished correction is invisible to the new brief.
10. Never send a blinded student any information about another arm's outcome
    while their prompt set is still open, even under a "quarantined" heading.
    Labelling a section quarantined does not un-send it. Withhold until their
    last timed leg is on disk.

## Advisor process lessons, 2026-08-17

These are mechanism-free but they cost real time, so they belong in the ledger.

- **Students have no PR-comment tool.** A student's only GitHub write path is the
  terminal result. Reported by edward while explaining why he could not signal a
  GPU handoff. Consequence: **the advisor must relay every cross-student
  coordination message**, and any brief that says "wait for a handoff signal" is
  giving an instruction the student cannot follow. Briefs now state the GPU order
  explicitly and say that no signal will arrive.
- **A PR holding a terminal result is routed to review, and
  `send_assignment_feedback` is unavailable there** (it requires `status:wip` as
  the only active assignment status). `request_assignment_revision` does work in
  review state and carries an arbitrarily long comment, so it is the review-state
  channel — including for the case where the revision request is mostly
  adjudication of an accepted result.
- **Merging moves the base.** Every subsequent mutation's
  `expected_current_base_sha` must be re-read, not carried forward from the
  previous call.
- **A challenge that is wrong on its face can still be right about the class of
  error.** Edward cited two files as carrying stale ceiling text; both were
  already fixed on the advisor branch, and he was reading his own out-of-ancestry
  base. Re-scanning anyway found three genuinely stale passages in a third file.
  Credit the challenge, then check the class.
- **When a band and a point estimate disagree, the point estimate is the thing
  that needs defending** (carried forward from PR #2, still the most useful
  single line about my own failure mode). Related: a stop signal that is monotone
  in the mechanism I have in mind is not automatically monotone in the objective.
