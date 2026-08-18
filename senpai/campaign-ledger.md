# Qwen 3.8 Senpai Campaign Ledger

This is the compact, versioned index for advisor handoffs. Update it with every
terminal experiment and official receipt. Keep large local score artifacts out
of Git; link only reproducible commands, commits, and concise evidence.

Machine-readable frontier pins live in
[`frontier-state.json`](frontier-state.json). If this ledger and that file
disagree, stop and repair both before assigning or submitting work.

## Current frontier

The organizer remote and the promoted Yukon row were refreshed at
`2026-08-18T17:01:12.000Z`. The advisor branch `senpai/qwen38-mtp-r1` carries
the same pins in [`frontier-state.json`](frontier-state.json) (blob
`97bb6ab4`), verified equal to `origin/main`'s copy; the earlier
`d1530a4` / `bd007bc7` / `3.13098700135133` staleness warning is discharged.

| Field | Value |
| --- | --- |
| Organizer source | `Layr-Labs/qwen-3.8-mtp-challenge` |
| Organizer synced commit | `474c75013f333f119bdc465d849f23917b195b20` |
| Best promoted submission | `942e5ab2-1c46-4c50-b7c3-eaf948878ed0` |
| Promoted source ref | `474c75013f333f119bdc465d849f23917b195b20` |
| Official score | `3.2341518328631` |
| Campaign `BASE_SHA` | Fetch `origin/main`, then run `git rev-parse origin/main`; the Git ref is authoritative because a file cannot contain the hash of its own commit |
| Submitted solver snapshot | `474c75013f333f119bdc465d849f23917b195b20` |
| Previous frontier | `3a995c2b` @ `86fb1f0`, `3.23222998733732` |
| Promotion delta | `+0.00192184552578` score, `+0.0595%` |

The promoted receipts above are the public Yukon frontier used to bootstrap and
then to re-baseline this campaign; they are not claimed as Senpai-authored
results.

### Frontier history: superseded `caec88d4` landed this campaign's own kernel change first

Everything from here to the end of this subsection is **history preserved on
purpose**. `caec88d4` / `0d800b22` (`3.14642585386152`) is no longer the
frontier — the table above is. The narrative is kept because its three findings
are still load-bearing: the measured `+1.56%` scoop-risk instance, the
`research/crossrow-closure.md` closure of the crossrow kernel, and the
`editablePaths` contract boundary at the frozen 8-output host tile.

The complete diff of promoted `caec88d4` is **six lines across exactly the two
quantized twins this campaign owns**, flipping three template arguments to
`DIRECT_NIBBLES=true`:

```
case 3: qmv_fast_crossrow_affine4_g64_m<T, 3, 3>  ->  <T, 3, 3, true>
case 4: qmv_fast_crossrow_affine4_g64_m<T, 4, 4>  ->  <T, 4, 4, true>
case 5: qmv_fast_crossrow_affine4_g64_m<T, 5, 3>  ->  <T, 5, 3, true>
```

That is **byte-for-byte advisor commit `67856b5`**, committed here before the
promotion and not yet submitted at the time it landed. Semantic identity is
proven, not asserted, and the proof is cheap to re-run:

```sh
git --no-pager diff upstream/main HEAD -- \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/mlx-generated/quantized.cpp \
  | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' \
  | grep -vE '^[+-][[:space:]]*(//|\*|/\*)' | grep -vE '^[+-][[:space:]]*$'
```

This must return EMPTY (grep exit 1). Only comments differ.

Two consequences, both load-bearing.

First, the *size* of the win is now measured by someone else's ranked receipt
rather than by our local estimate: **`+1.56%`**. Any campaign document that
sized `DIRECT_NIBBLES` at M=3,4,5 near "5% of verify rounds" is refuted by this
row. The local `--local-submit` harness could not have told us this — see the
noise rule below.

Second, the delay between committing a kernel win and submitting it is a **real
scoop risk with a measured instance**. This is the scoop-risk entry in the update
checklist below (item 46 as of this revision). It was previously cross-referenced
as "checklist item 53", a number the checklist has never reached in any of its
revisions; the entry was written this session and the reference repointed.

The follow-on question — is there anything left in this kernel — is answered in
[`research/crossrow-closure.md`](../research/crossrow-closure.md): no. Seven
candidates are enumerated and closed there, including the two that matter most.
Every FMA fusion in the inner product is forbidden because it reassociates the
accumulation and `tokenFidelityGate` is
`trusted-sequential-reverification-exact-token-match`. And raising
`rows_per_simd` from 4 to 8 — bit-exact, `-9.6%` ALU per row at NA=3, the
single largest remaining win — is **blocked by the `editablePaths` contract**,
because it needs `bn` and `group_dims` from
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:251-254`,
which matches zero `editablePaths` entries. Treat the frozen 8-output host tile
as a contract boundary, not a kernel convention.

Campaign commit `006a369` imports the exact promoted submitted surface from
`474c75013f333f119bdc465d849f23917b195b20`. Relative to `86fb1f0`, it
restores the dedicated single-row affine-2/group-64 coarse-readout kernel and
keeps the executable M=8 affine-4/group-64 4+4 split. Replace semantics also
remove the prior full-memory residency and post-wire command-buffer policy.
No organizer policy, contract, fixture, workflow, guide, dependency, head
manifest, or other trusted file changed.

The declared head is
`hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae6282749a52e052496dd5300b4aa441df7301e8`,
tree digest `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`,
and 427,742,600 bytes. It preserves the promoted 4-bit/group-64 precision-island
head and adds the affine-2 compact `draft_lm_head` ABI: weight
`[98,336, 320]`, scales/biases `[98,336, 80]`, followed by an exact affine-4
rerank of a 32-token shortlist.

The exact promoted readable header carried a contradictory M=8 3+3+2 comment
beside the executable 4+4 call. Campaign commit `b4ed293` replaces only that
comment with the checked-in generated twin's accurate 4+4 description.
Executable kernel text is unchanged.

**Plausibility ceiling: `5.0`.** Raised from `3.0` by operator commit
`a5854b979499800a6f5f71a8d4fc14fd43ca4723` (2026-08-17, `AGENTS.md` +
`senpai/program.md` only) and readable at
`benchmark.json /scoring/decodeSpeedupCeiling` on base `b85e782`. It is a
fail-closed administrative gate, **not** a stop target and not an optimization
target (`senpai/program.md:21`). Headroom from the **current** promoted
`3.2341518328631` is `+1.7658481671369` score, i.e. the ceiling sits at
`1.546x` the live bar — a further **+54.60%** relative decode speedup would be
required to reach it. (The earlier form of this paragraph computed headroom from
the long-superseded `3.02460155382533` and converted it to seconds through a
stale `-0.4335` calibration; that conversion is deleted rather than re-derived,
because no current lever needs it.)
No lever measured this campaign is within an order of magnitude of that, so the
ceiling changes nothing operationally except that a large legitimate result
must not be held back. Docs corrected this session:
`research/ESTABLISHED_FACTS.md` and `research/CURRENT_RESEARCH_STATE.md` were
still stale at `3.0` and at the superseded `2.904` frontier.

## Base `b85e782`: what moved, and what students must re-derive

**This earlier advisor progression through `422db045` was entirely
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

### CORRECTED 2026-08-17 (advisor): the width story above was wrong, and the real mechanism is simpler

An earlier revision of this section said "in `sdpa_vector_2pass` the block count
depends on `n_simds = gqa_factor * q.shape(2)`, so `qL` enters the reduction
geometry for the first time at the boundary." **That is false inside our
window.** The host-side block count is computed at
`scaled_dot_product_attention.cpp:440-476`:

```
int gqa_factor = q.shape(1) / k.shape(1);
int n_simds = gqa_factor * q.shape(2);
char devc = d.get_architecture().back();
int N = k.shape(2);
int blocks;
if (devc == 's') {
  blocks = 64;
  if (N > 1024 && n_simds > 4) { N<=8192->128, N<=32768->256, N<=65536->512, else 1024 }
} else if (devc == 'd') {
  blocks = 128;
  if (n_simds <= 2 && N > 8192) blocks = 256;
  else if (n_simds >= 6) { N>=16384 && N<65536 -> 512, N>=65536 -> 1024 }
} else {
  blocks = (n_simds >= 4) ? 64 : 32;
}
```

The `n_simds` ladder is guarded by **`N > 1024`**, which is FALSE at
`N == 1024`. The ranked window tops out at exactly 1024, so on an `'s'`-suffixed
device `blocks = 64` unconditionally and **`qL` never touches the reduction
geometry at all.** The corrected mechanism has three parts, all verified against
the two key loops in `kernels/sdpa_vector.h`:

- **(a) Both partitions are independent of `N`.** Single-pass, `:98`:
  `for (int i = simd_gid; i < N; i += BN)` with `BN = 32`. Two-pass first
  kernel, `:263`: `for (int i = block_idx; i < N; i += blocks)`. Key `i` lands
  in accumulator `i % 32` or `i % blocks`; `N` only decides where the loop
  stops.
- **(b) The causal cut lands at the same absolute position in both.** Both loops
  compute `use_key = i <= (N - q_seq_len + q_seq_idx)`, and a row at absolute
  cache position `p` satisfies `p = N - q_seq_len + q_seq_idx`, so the cut is
  exactly `i <= p` **regardless of `N` or width**. The key *set* summed for a
  given token is therefore identical in serial and MTP at any width and any
  `N`. This is not a masking bug and not a width bug.
- **(c) The only difference is the stride: 32 versus `blocks = 64`.** Same key
  set, different grouping into partial accumulators, different summation tree,
  different rounding.

(a) and (b) together *force* the 919/919 bit-exact agreement below the
boundary — it is a theorem, not a coincidence — and (c) is the entire source of
the drift above it. Widths 2, 4 and 8 all drift because all three straddle the
boundary; width only changes how many tokens land on the wrong side of it.

**Quantitative pre-registered prediction.** Serial emits token `t` (1-indexed
1..512) with its query at cache position `511 + t` and `N = 512 + t`, so serial
crosses `N >= 1024` only at `t = 512`. MTP verifying width `qL` at key length
1024 covers tokens `513 - qL .. 512`, all two-pass. The family-mismatch band is
therefore the last `qL - 1` tokens before the final one, and nothing else:

| width `qL` | MTP two-pass tokens | serial two-pass | predicted mismatch band (token idx) | cache positions |
|---|---|---|---|---|
| 2 | 511..512 | 512 only | **511** | 1022 |
| 4 | 509..512 | 512 only | **509..511** | 1020..1022 |
| 5 | 508..512 | 512 only | **508..511** | 1019..1022 |
| 8 | 505..512 | 512 only | **505..511** | 1016..1022 |

Width 2 predicts a **single** position, cache 1022 — exactly the low edge of
PR #2's reported 1022–1024 band. Token-index versus cache-position conventions
must be reconciled before this is claimed as agreement.

**Model geometry** (`fixtures/qwen3_6_27b_config.json`, `text_config`):
`num_attention_heads = 24`, `num_key_value_heads = 4`, `hidden_size = 5120`,
`full_attention_interval = 4`, `attn_output_gate = true`,
`num_hidden_layers = 64`, `vocab_size = 248320`, `intermediate_size = 17408`,
quantization `group_size = 64`. So **`gqa_factor = 24/4 = 6`**, and
`supports_sdpa_vector`'s `qL * gqa_factor <= 32` gives **`qL <= 5`** — which
*independently explains* the otherwise unmotivated `let split = 5` in
`AttentionUtils.swift`. `full_attention_interval = 4` means only **16 of 64**
layers reach SDPA at all; the rest are GDN linear attention.

**One open discrepancy, honestly flagged:** the fixture declares
`head_dim = 256`, but the measured scored shapes imply **128**. With
`head_dim = 128` and the output gate on,
`24*128 + 4*128 + 4*128 + 24*128 = 7168` reproduces the observed
`full_attn.qkv_proj_fused` N exactly, and `full_attn.o_proj` K `= 3072 = 24*128`
exactly; `head_dim = 256` reproduces neither. The likely explanation is that the
fixture is Qwen **3.6** while the target is **3.8**. The conclusion is robust
either way because `sdpa_vector_supported_head_dim` admits `{64,96,128,256}`,
but `gqa_factor` is load-bearing for the `qL <= 5` derivation and **must be
resolved from the resident model config**, not from this fixture. No local HF
cache is present to check against.

**Scope calibration, so nobody over-invests.** `benchmark.json`'s own
description states that native MTP decode is measured exact-greedy against
serial **12/12 runs to 512 tokens, across all EOS branches**. That is organizer
evidence, on ranked hardware, that this hazard is **not currently breaking the
fidelity gate**. The value here is therefore (i) closing a campaign item open
since PR #2, (ii) retiring a **latent** risk that goes live the moment anything
moves widths or window length, and (iii) a bounded exactness-preserving repair
if one turns out to be needed. Expect a **Not useful** or **Unclear** label;
there is no speedup in it.

**Candidate repair (advisor derivation, never compiled or run).** The dispatch
is not editable, but it is a pure function of `k.shape(2)`, and
`AttentionUtils.swift` already re-slices keys at the call site. In the one
straddling round per leg (`kL >= 1024 && kL - qL < 1023`): chunk A = rows
`0 ..< qL-1` against `keys[0 ..< 1023]`, which stays single-pass and so matches
serial; chunk B = row `qL-1` against the full `keys`, which is two-pass and so
matches serial at `t = 512`. Causality survives because chunk A row `j` cuts at
`1023 - (qL-1) + j = 1024 - qL + j`, its true absolute position. Cost is one
round out of roughly 150 plus one extra pass over about 1024 KV rows. Five named
attacks, in priority order: **(v)** whether the KV cache's `step = 256` growth
makes `cachedKeys.dim(2)` exceed the logical key length — *check this first*;
**(i)** off-by-one, 1023 versus 1024; **(ii)** whether `kL` overshoots 1024
given the overlay commits `512 <= committed <= 520`; **(iii)** whether the
existing `6 <= qL <= 9` chunk and the new one compose or fight; **(iv)** whether
`concatenated` along axis 2 is bit-exact. Note the KV cache step is 256
(`KVCache.swift:388`), so 1024 is *not* uniquely a cache-growth boundary —
256/512/768 are too — which independently rules out cache reallocation as the
explanation.

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
Our promoted `3.02460155382533` is already well above that 1.74x envelope, which
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
| `4437d061-7366-422c-a51d-1679995307ed` | `95f8311d97bb22e0d827b4698ed486b664f4adf6` | `d1530a40` | `senpai` | **`rejected`**, `2.86126590369985` (`-0.28516`, `-28.73%` vs the then-bar) | `senpai/submissions/01-*` | Causally closed: the manifest pinned our own fine-tuned head (`hf:morgan/…dev20-k4b06-lr1e5-q4-g64-islands@97aa5363`, 270,404,736 B). The head was worse than the organizer's; the kernel content was exonerated by a comment-stripped SHA-256 identity against the parent. |
| `74d1bd3a-2e26-4bd2-9e2e-76c0a38cda3a` | `07b2f1b56f4abd714e1dbabd8d39c278cc5a4a9f` | `c80c023326bb933307706d506c0a2dbbff1a628f` | `senpai` | **`failed`** (no score; `2026-08-18T01:30Z`) | [`senpai/submissions/02-combined-e15-directnibbles.md`](submissions/02-combined-e15-directnibbles.md) | Ranked `failed`, not `rejected`: the workflow's correctness gate emitted no valid JSON payload (`qwen-mtp-ranked-benchmark.yml:2580`), stderr and artifacts are suppressed, and the run lands in the non-charged `score_present_validation_failed` bucket. This row's candidate SHA was recorded as `d11e01ea` in an earlier revision; Yukon reports `07b2f1b5`. |
| `b360b4c8-01db-4e0c-b67f-03df1e8acc2f` | `81640b7891259d1b24e7bfd23ede0d09a31ab27b` | `036fd9ca` | `senpai` | **`failed`** (no score; `2026-08-18T03:33Z`) | same E15 note | Same failure signature as `74d1bd3a` and the same E15 requant payload (`61c4d962`). Manifest-touching submissions fail at `55.6%` versus a `36.25%` population rate, which is the strongest single predictor available. |
| `9197ed62-621f-474d-bfba-e1efddd9dd4c` | announced `51c5b6688c7a54dfe3628e010ecb2321cf014034`; **Yukon reports `dbf91c6c4876a58b0decca130acf4…`** | `527306761f70e2c4024f347915328894db80c181` | `senpai` | **`rejected`**, `3.06938159465413` (`-0.173619`; created `2026-08-18T17:08Z`) | announced on issue #31 | Not Senpai-authored: launched by the operator. Submitted diff is **only `mtp-head.manifest.json`**, repointing the declared head to `hf:morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank@fd4a99c5`. **This is the second scored confirmation that a fine-tuned replacement draft head is worse than the organizer's declared head**: the two head-quality plays in this table are the only two rows that produced a score, and both were rejected (`2.86126590369985` for the Dev20 head at `4437d061`, `3.06938159465413` here). Do not re-propose a replacement head; that lever is closed by measurement, not by argument. Two secondary facts are worth keeping: (i) Yukon's `commit` column reports a repackaged commit (`dbf91c6c…`) and not the announced candidate (`51c5b668…`), exactly as it did for `74d1bd3a` (announced `d11e01ea`, reported `07b2f1b5`), so never key a row on the announced SHA alone; (ii) applying the delta rule verified on `4437d061` (`2.86126590369985 + 0.28516 = 3.14642590…`, the then-bar) to this row gives an implied bar of `3.06938159465413 + 0.173619 ≈ 3.2430006`, which is **higher than the promoted `3.2341518328631`** even though `upstream/main` is still `474c75013f333f119bdc465d849f23917b195b20` with zero newer commits — either a promotion was in flight at evaluation time or the delta baseline is not the promoted frontier. Treat `3.2341518328631` as the ledgered bar and `≈3.2430006` as an open question. **With this row terminal, all four submissions are terminal and the single in-flight slot is FREE for the first time in the campaign.** |

Benchmark run `5d1ee4d7-80bd-4555-b182-6505f26ef495`. The `74d1bd3a` row was
submitted with:

```sh
senpai/submit-official.sh c80c023326bb933307706d506c0a2dbbff1a628f \
  --model "senpai" \
  --note-file senpai/submissions/02-combined-e15-directnibbles.md
```

**Read the content of this submission carefully before interpreting its score.**
`origin/main` at `c80c023` does *not* contain promoted `caec88d4`, so the
submitted diff carries two things at once: `DIRECT_NIBBLES` at M=3,4,5, which
the frontier has since independently promoted at `+1.56%` and which is
therefore **no longer novel**, and the E15 3-bit compact draft readout
(`+0.8334%` ranked, PR #17, askeladd), which is **the only novel content in the
row**. Section 8 of the public note is the honest novelty disclosure. The
"expected combined score is roughly `3.172`" prediction that stood here is moot:
the row came back `failed` with no score at all, so it never tested the
prediction, and the bar has since moved to `3.2341518328631`.

🔴 **`c80c023326bb933307706d506c0a2dbbff1a628f` is no longer a legal
`BASE_SHA`.** The command above is preserved for provenance only. Since that
submission, `origin/main` advanced eighteen commits and four of the ninety
protected paths now differ at `c80c023`, so `submit-official.sh:382` rejects it.
The current legal base is `527306761f70e2c4024f347915328894db80c181` — but only
once the advisor branch descends from it, because `:186-194` also requires the
base to be an ancestor of `HEAD`. See update-checklist item 47.

Two mechanics facts that cost time to establish and should not be re-derived.
`yukon submit` enforces a hard **server-side limit of one in-flight submission
per account**, so while any row reads `validating` no further submission is
possible from any role; validation latency is on the order of hours, so do not
block on it. As of `2026-08-18` all four rows above are terminal
(`rejected`/`failed`/`failed`/`rejected`), so **the slot is free** — which makes
update-checklist item 46 live: the delay between committing a measured kernel
win and submitting it is itself a risk, because the frontier moves. Poll with
`yukon submissions` (plural), run from the linked repository directory.
And the submitter never passes `BASE_SHA` to `yukon` — it is
consumed only by the wrapper's own gates and the script ends
`exec yukon submit "${submit_args[@]}"`, so **yukon packages the advisor
branch HEAD's editable paths**. Non-editable files, including everything under
`Tests/` and `research/`, cannot cause a rejection.

Historical base rate for calibration: roughly 226 of about 482 Yukon rows are
`rejected`, and a `3.1149` score was rejected under the *old* `3.13099` bar.
Rejection is the modal outcome, not an anomaly.

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
| `key_len = 1024` SDPA two-pass dispatch boundary | `scaled_dot_product_attention.cpp:743-753` inside the scored 512-seed + 512-decode window; every full-attention layer | The positional exactness residual at decode positions 1022-1024 is a **kernel-family change**, not a width effect: at `k.shape(2) >= 1024` on a `'d'`- or `'s'`-suffixed GPU the dispatch switches `sdpa_vector` -> `sdpa_vector_2pass`, which changes the floating-point reduction order of every score | Four independent confirmations that the residual is positional, not width-driven (PR #2): 919/919 non-terminal width-9 rows bit-exact; all 15 value mismatches in the final block at 1022-1024; widths 2, 4 and 8 drift there too; the 128-token local-submit window ends at 640 and is clean. Width 4 is **outside** the `AttentionUtils` `6 <= qL <= 9` split, which exonerates that path. Mechanism located and then **corrected** this session — see the corrected subsection, which supersedes an earlier `n_simds` account that is false inside our window. The real mechanism is a stride change and nothing else: **(a)** both key loops partition independently of `N` (`i += 32` single-pass at `sdpa_vector.h:98`, `i += blocks` two-pass at `:263`), **(b)** both compute `use_key = i <= (N - q_seq_len + q_seq_idx)`, which for a row at absolute cache position `p` is exactly `i <= p` regardless of `N` or width, so the summed key *set* is identical in serial and MTP, and **(c)** the only difference is stride 32 versus `blocks = 64`. At `N == 1024` the host-side `n_simds` ladder is guarded by `N > 1024` and never entered, so **width does not affect `blocks` at all**. (a)+(b) therefore *force* the 919/919 agreement as a theorem. Pre-registered quantitative band: mismatches confined to the last `qL - 1` tokens before the final one — width 2 predicts the **single** cache position 1022, the exact low edge of PR #2's reported band | **mechanism located and corrected; consequence UNRESOLVED and host-dependent; assigned as E19 (PR #21, alphonse)** | The single decisive fact is the **last character of the ranked M5 architecture string**. The `>= 1024` trigger is gated on `devc` in `{'d','s'}`; local `applegpu_g16s` fires, and no M5 arch string is recorded anywhere in the repo. If M5 is neither, the only surviving trigger is `k.shape(1) < q.shape(1) && k.shape(2) >= 4096`, which a 1024-token ranked window never reaches, and this **closes as a local-only artifact / not useful**. If M5 *is* `'d'` or `'s'`, the scored leg changes attention kernel family inside an exact-token-match gate and one flipped near-tie argmax fails a whole leg. Editability: the dispatch `.cpp` and `device.cpp` are **not** editable; `kernels/sdpa_vector.h`, `kernels/scaled_dot_product_attention.metal` and `MLXLMCommon/AttentionUtils.swift` are. `sdpa_vector.h` has **no `mlx-generated` twin**, so it is AOT-only and any edit needs `tools/build-mlx-metallib.sh --all-build-roots`. `MLX_SDPA_BLOCKS` (`:477-478`) changes the partial-accumulator count with no kernel edit but is **diagnostic only** — `MLX_`-prefixed vars are named blocker #1 |
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
| 2026-08-17 | advisor `senpai/qwen38-mtp-r1` / `1bb627ab` | record the refuted `qmm_splitk` conclusion and the campaign's process lessons | `422db045` | 1 file, `+218/-29`, zero editable/scored-path files. Habitability proved first: `swift build --build-tests -c debug --force-resolved-versions` exit 0 in `0.91` s. Adds the REFUTED subsection (verbatim `qmm_splitk` delegation, the `N <= 512` derivation, the eight-row `split_k` table, the `qmm()` NAX early return at `:697-699`), the SURVIVES subsection, and three standing rules | not submitted; documentation only | The standing rule this produced is the most transferable thing in it: **"a call-site trace is not a call-graph trace."** I read the outer dispatch, saw `qmm_splitk(...); return;`, and stopped one level too shallow — the callee delegates straight back to `qmm()` whenever `split_k <= 1`, which is *every* scored prefill shape |
| 2026-08-17 | `qwen-alphonse/keylen-1024-residual` / `0d6853ca` (PR #21, E19) | close the campaign-level `key_len = 1024` positional exactness residual | `1bb627ab` | **r1 assigned, in flight.** Zero-GPU, host-only, takes no benchmark lock. Brief hands him the complete corrected mechanism (stride 32 vs `blocks = 64`, with the (a)/(b)/(c) argument and both verbatim key loops), the quantitative band table, `gqa_factor = 6` and the `split = 5` witness, and the candidate repair with five ranked attacks. **Deliverable 1 is to prove the live call path before anything else**; deliverables 1, 3 and 8 alone are a complete mergeable result if the `devc` question resolves against `'d'`/`'s'` in his first hour | not submitted | Both outcomes pre-registered so the result cannot be talked into significance either way. Brief also restates, verbatim, the **two advisor errors PR #2 caught me in**: ranking arms by `accepted_tokens_per_round` (anti-correlated with speed among cap-8 arms), and closing a direction on a point estimate that contradicted my own preregistered band — "when a band and a point estimate disagree, the point estimate is the thing that needs defending" |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-6` / `29f1ee4` | exact promoted editable-snapshot import | `1c57496` | preservation, overlay, budget, and trusted-parity checks passed; both changed QMV twins are byte-identical to promoted source `79683c63`; regeneration audit is locally blocked by the missing Xcode Metal Toolchain | adopted public promoted `14b53255` at `3.02460155382533`; not a Senpai-authored submission | Exact organizer source `79683c63`; two affine4/group-64 QMV kernel twins changed relative to the previous promoted source |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-6` / `28e591f` | fixed-window continuation after EOS on promoted source `79683c63` | `29f1ee4` | source overlay and trusted-parity checks passed; Swift test compilation remains blocked by the unchanged organizer `QwenMTPVerbTests.swift:755` type error, so full 512-token exact replay is still required | not submitted; not promoted | Reapplies the campaign's parent-owned fixed-window behavior as a separate overlay |
| 2026-08-17 | advisor `senpai/qwen38-mtp-r1` / `56e81ff` | merge current challenge frontier and reconcile local cooling policy | `3e6ca31` + campaign `main` `57cb86b` | complete test target builds; focused fixed-window suite passes 10 tests in 2 suites; QMV twins are byte-identical to official source `79683c63`; experiment ledger preserved | not submitted; adopted public promoted `14b53255` at `3.02460155382533` | Real 40C gate remains default; ungated local timing is permitted only under the recorded ABBA, temperature, and false-qualification protocol |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-ed4dfd6-r2` / `f04df93` | exact promoted editable-snapshot import | `1d573f6` | preservation, campaign overlay, and editable budget passed; canonical audit exposed one comment-only stale generated twin in the promoted bytes | adopted public promoted `39fdbf62-60e4-4ab7-bf09-0d1b5a0b618a` / `ed4dfd6` at `3.07714439121787`; not a Senpai-authored submission | Exact promoted source changes four submitted paths relative to `79683c6`; the campaign import changes five paths because it also removes the unpromoted fixed-window overlay; preserves the intervening campaign-only `program.md` update at `1d573f6` |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-ed4dfd6-r2` / `7ab7376` | canonical regeneration of promoted `quantized` twin | `f04df93` | comment-stripped source SHA is identical before/after; twin audit 29/29 and release build passed; full `swift test` remains blocked by unchanged organizer `QwenMTPVerbTests.swift:755` type error | not submitted; mechanical campaign repair only | Canonical output expands three comments to thirteen; no executable token changes; no AOT Metal source changed, so `mlx.metallib` rebuild is not applicable |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-bd007bc` / `c8dceb9` | exact promoted editable-snapshot import | `1d1eeda` | preservation, campaign overlay, editable budget, trusted parity, and release build passed; full `swift test` rebuilt products but remains blocked by the unchanged organizer `QwenMTPVerbTests.swift:755` type error | adopted public promoted `bd007bc7-e8ab-4919-baf4-d5e90068dd83` / `d1530a409848b82a0a1890141c1483875d1e0173` at `3.13098700135133`; not a Senpai-authored submission | Net executable delta from `ed4dfd6` is the M=7 direct-nibbles flag in the readable/generated QMV twins; no organizer policy or dependency files changed |
| 2026-08-17 | `codex/sync-organizer-frontier-20260817-bd007bc` / `08fb76a` | canonical regeneration of promoted `quantized` twin | `c8dceb9` | comment-stripped source SHA is identical before/after; twin audit 29/29; no AOT-only Metal source changed, so `mlx.metallib` rebuild is not applicable | not submitted; mechanical campaign repair only | Canonical output expands three comments to thirteen; no executable token changes |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-live-1` / `7906ea1` | exact promoted editable-snapshot import | `c80c023` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build passed; full `swift test` reaches the unchanged trusted `QwenMTPVerbTests.swift:755` `String`/`Comment?` compile defect | adopted public promoted `824dc272-b560-4dc6-bf6c-42f58944f4cb` / `8dabcfb` at `3.19351254799833`; not a Senpai-authored submission | Four editable files changed; current q2/q4 rerank head and both M=1 MTP-only QMV dispatches are present; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-live-1` / `7b740c7` | canonical regeneration of promoted `quantized` twin | `7906ea1` | comment-stripped source SHA is identical; `quantized` reproduces; 27/29 twins pass and both remaining NAX differences are isolated to compiler-owned system-section inventory with every vendored section byte-equal | not submitted; mechanical campaign repair only | Preserves exact promoted NAX blobs; no AOT-only Metal source changed, so `mlx.metallib` rebuild is not applicable |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-12d3756` / `d631025` | exact promoted editable-snapshot import | `23f2781` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build passed; full `swift test` reaches the unchanged trusted `QwenMTPVerbTests.swift:755` `String`/`Comment?` compile defect | adopted public promoted `578535f7-95e6-4f95-a34c-281b9dbbbffc` / `12d3756` at `3.19580475139646`; not a Senpai-authored submission | Two editable QMV twins changed: M=8 affine4/group-64 moves from 3+3+2 to 4+4, while the prior M=1 affine2 coarse-readout and narrow affine4 fast paths are absent; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-12d3756` / `cc32c73` | canonical regeneration of promoted `quantized` twin | `d631025` | generated delta is comment-only; twin audit 29/29, final release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Canonical output expands three comments to ten; executable tokens are unchanged; the fresh metallib covers the imported readable-header change |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-369cc05` / `ff5866b` | exact promoted editable-snapshot import | `d649aab` | preservation, campaign overlay, editable budget, trusted parity, exact 89-path promoted-blob checks, and release build passed; full `swift test` compiles the new Qwen35 source and reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `12864bc1-9c9e-4e3b-8964-e8b9e4da8d31` / `369cc05` at `3.21000579584503`; not a Senpai-authored submission | Current source combines the exact two-dispatch top-32 proposal shortlist with the restored M=8 3+3+2 QMV split; the replaced Qwen35 file does not retain the immediately prior `868cde8` fusion suite; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-369cc05` / `9d04199` | canonical regeneration of promoted `quantized` twin | `ff5866b` | generated delta is comment-only; explicit-toolchain twin audit 29/29, final release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Canonical output expands the abbreviated M=8 comments; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-dccba74` / `abca948` | exact promoted editable-snapshot import | `a187ec6` | preservation, campaign overlay, editable budget, trusted parity, exact promoted-blob checks, and release build pass; full `swift test` reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `72ce82dc-f751-485d-a7b3-94ab6471cf87` / `dccba745` at `3.22826053954006`; not a Senpai-authored submission | Two editable QMV twins changed: the proposal-only M=1 affine-2 fast path is restored and M=8 affine-4/group-64 moves from 3+3+2 to 4+4; proposal-head tree remains `559b24eb`; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-dccba74` / `0b19827` | reconcile promoted M=8 readable/twin comment | `abca948` | comment-only delta; explicit-toolchain twin audit 29/29, frozen release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Replaces a stale 3+3+2 narrative with the generated twin's accurate 4+4 description; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-86fb1f0` / `8afb5e8` | exact promoted editable-snapshot import | `14ef8c2` | preservation, campaign overlay, editable budget, trusted parity, exact five-path import, and release build pass; full `swift test` reaches the byte-identical inherited `QwenMTPVerbTests.swift:755` `String`/`Comment?` defect | adopted public promoted `3a995c2b-3c42-48e8-b982-f36a8abda0e7` / `86fb1f0` at `3.23222998733732`; not a Senpai-authored submission | Five editable paths changed; the full-memory residency/command-buffer policy and Qwen35 fusion suite are present, the dedicated M=1 affine-2 QMV is absent, and the proposal-head manifest is unchanged; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-86fb1f0` / `76b961f` | canonical regeneration of promoted `quantized` twin | `8afb5e8` | generated delta is comment-only; explicit-toolchain twin audit 29/29, frozen release build, and fresh AOT `mlx.metallib` build pass with Metal toolchain `32023.883` | not submitted; mechanical campaign repair only | Expands the abbreviated M=8 comment to the readable header's direct-nibble/IPG4 rationale; executable tokens are unchanged |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-474c750` / `006a369` | exact promoted editable-snapshot import | `50a5be6` | preservation, campaign overlay, editable budget, trusted parity, and exact 89-path import pass | adopted public promoted `942e5ab2-1c46-4c50-b7c3-eaf948878ed0` / `474c750` at `3.2341518328631`; not a Senpai-authored submission | Four editable paths changed; the M=1 affine-2 coarse-readout kernel is restored, executable M=8 remains 4+4, and the prior full-memory residency policy is removed; no trusted policy changed |
| 2026-08-18 | `codex/sync-organizer-frontier-20260818-474c750` / `b4ed293` | reconcile promoted M=8 readable/twin comment | `006a369` | comment-only delta; executable `<T,8,4,true>` call unchanged | not submitted; mechanical campaign repair only | Replaces the stale 3+3+2 narrative with the checked-in generated twin's 4+4 description |
| 2026-08-18 | advisor `senpai/qwen38-mtp-r1` / `d7619a7` | end-to-end validation of the merged tree **at the scored token count**: a 513-row golden, then four timed legs (512×depth-2, 512×serial control, 256×depth-2, 128×depth-2) | `d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602` | All four legs `all_tokens_matched = true` with `residual_divergence_count = 0`. The 512×depth-2 leg emitted 512 tokens over 176 rounds with 336 accepted, 16 rejected and 176 tails = **528 declared rows**, and `528 − 16 = 512` reproduces PR #30's finding F4 exactly (final target cache offset 1024). Reported-style ratio **2.1784×** (`0.0757904355` / `0.0347923830` seconds per token, both prefill-inclusive); true-decode ratio **2.5144×** (`34.517648` / `13.728098` s). E20's per-token numbers replicated within 3% on this tree (serial `67.417` ms vs `65.539`; depth-2 `26.813` vs `27.401`). Open lead recorded for follow-up: the 16 rejections at 512 have **no counterpart at 128 or 256**, where the accepted-draft rate is exactly `1.0`, and the reference's first stop token lands at index 300 — inside the 256→512 window. | not submitted (correctness/validation run, not a candidate) | job `a4bead4e-483d-4f1f-9e94-d2626a7f064f`, exit 0, 647.342 s; `research/e26-legs.sh` golden + legs; two harness defects found in that script and relayed (a `jq` selector on a non-existent `row_ledger` key silently writes 0-byte ledgers, and a `.matched` field name that should be `.all_tokens_matched`) |

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
11. **A mechanism that explains the data is not thereby the mechanism.** The
    first `key_len = 1024` account said `qL` enters the reduction geometry
    through `n_simds`; it explained every one of PR #2's four observations and it
    was still false, because the `n_simds` ladder is guarded by `N > 1024` and
    our window stops at exactly 1024. The test that caught it was demanding a
    *quantitative* prediction: the true mechanism (stride 32 vs 64) names an
    exact mismatch band per width, the false one named none. Before banking a
    mechanism, make it predict a number it could fail on.
12. **Verify a merge empirically; a merge you did not inspect is a hypothesis.**
    Run the merge, then compare the three blobs (merge-base, theirs, ours) for
    every file the merge could touch. This session's `origin/main` merge was
    predicted to be a sixteen-file integration and turned out to touch exactly
    one file, because seven of eight candidates were byte-identical between
    theirs and ours.
13. **Taking `--theirs` on a vendored file can import the frontier's own
    invariant violation.** Promoted `474c7501` reverted the `quantized.h` M=8
    comment to a "3+3+2 register cliff" narrative while leaving the code at
    `<T,8,4,true>`. A blind take inherits the contradiction; the campaign had to
    repair it at `b4ed293`.
14. **A waiver's negative control must attack the row that is live, not the row
    that was live when the waiver was written.** A control that mutates a
    superseded digest passes vacuously and proves nothing about the current
    twin.
15. **A gate that reads state from a different ref than your branch cannot be
    fixed by editing your branch.** `submit-official.sh:196-218` reads
    `senpai/frontier-state.json` from `origin/main`, not from `HEAD`; and
    `:186-194` requires `BASE_SHA` to be an ancestor of *both* `HEAD` and
    `origin/main`. The only repair is to move the ref the gate reads, or to
    merge it.
16. **A code comment's claim about the reference implementation is a testable
    hypothesis, not documentation.** The frontier's "319/437/216 µs M=7/8/9
    register cliff" comment was contradicted by E22's measured monotone
    139.325/177.758/186.422 ms. Measure before inheriting a number from prose.
17. **A job that exits nonzero may still have delivered its payload.** The
    declared-head provisioning job exited 1 solely because an optional
    `config.json` returned 404; the 427,742,600-byte `model.safetensors` was
    already on disk and tree-digest-correct. Read the log before re-running.
18. **A pinned-count constant may survive only as an error-message string.**
    `MLXFastConstants.qwenMTPHeadTensorCount = 15` looked like it would reject
    the 40-tensor declared head; it is referenced only inside a diagnostic
    message. The live checks are `weightMap.count >= 3`, absence of an `mtp.`
    prefix, and three named tensors.
19. **A `Validate`/`Accept` commit's content is not necessarily the frontier's
    intent.** `474c7501` is a single accept commit whose diff *reverts* two
    mechanisms and adds one; the accept ceremony carries no guarantee that the
    promoted overlay was authored on the current base.
20. **The upstream frontier is not monotone.** Promoted scores went
    3.19088 → 3.19351 → 3.19580 → 3.20989 → 3.21001 → 3.22826 → 3.23223 →
    3.23415, but the *content* regressed twice along the way. Never infer that a
    later promoted tree is a superset of an earlier one.
21. **The frontier ships its own offline verifiers as public-but-uncalled
    functions.** `qwen35VerifyDraftTop32(trials:seed:)` and
    `qwen35BenchDraftTop32(iters:)` exist with zero call sites. An uncalled
    verifier is not a gate; wire it into `Tests/` or it protects nothing.
22. **Byte-identical-to-merge-base on both sides means a frontier re-sync is a
    zero-conflict fast take.** Compute that before budgeting time for conflict
    resolution.
23. **A frontier insertion entirely below line N leaves every citation above N
    valid.** Check where the insertion starts before re-grepping the whole
    anchor table; and when a revert *removes* lines, expect a negative shift
    (this session: +80, then −83).
24. **Order a state-file bump after the merge commit that makes it true.** A
    `frontier-state.json` that names a commit the branch does not yet contain is
    a lie that the submit gate will not catch, because it reads state from
    `origin/main`.
25. **The bar moves in the third decimal.** Consecutive promotions differed by
    `+0.00192184552578` (`+0.0595%`). Any local measurement whose noise floor is
    ~1% cannot adjudicate a promotion-sized effect; size the experiment against
    the *delta*, not the score.
26. **A `swift build` that exits zero may have compiled nothing.** Count objects
    newer than a pre-build stamp and check whether the binary was relinked.
    Exit code alone is not a build proof, and neither is elapsed time.
27. **A `--theirs` resolution on a vendored file can delete a symbol a
    campaign-owned file still calls.** Grep the campaign surface for every
    symbol the resolution removes before committing.
28. **A ledger entry that says "closed by source inspection" is not a gate.**
    The stop-token continuation fix has now been fixed four times and lost five
    times (`f1a874d` → `330b44e` → `b219009` → `bc552e5` → `origin/main` via
    `006a369`). Only an executable content assertion survives a frontier sync.
29. **A reverted-then-re-submitted mechanism can return byte-for-byte.** E24's
    constant-scalar hoist arrived free at `86fb1f02` with payload
    `fd7a26d831cdc64c9ea0e437`. Race the frontier; do not bank an unpromoted
    mechanism as safely ours.
30. **Sync the *organizer* ref for content, but merge campaign `main` for
    submission ancestry.** These are two different gates and both must be
    satisfied: `:382` compares the protected surface `main_sha` ↔ `base_sha`
    (content), while `:186-194` requires `base_sha ∈ ancestors(HEAD) ∩
    ancestors(main_sha)` (ancestry). Satisfying one does not satisfy the other.
31. **A waiver dies by closure, not only by drift.** When the underlying
    question is answered, delete the waiver rather than refreshing its digests;
    a refreshed waiver for a closed question is a permanent false positive.
32. **A script that exits 0 on a clean tree is untested.** Exercise every gate
    against a deliberately dirty or wrong input before trusting a green run.
33. **Identify which product compiles the code you edited.** Only
    `mlxfast-runtime-worker` contains the campaign model surface;
    `.build/release/mlxfast-swift` has zero `Qwen36MTPBlockSession` symbols and
    no `Cmlx.build` directory at all. Building the CLI proves nothing about a
    model-surface edit.
34. **Elapsed time is not a build proof, and a short build is not suspicious.**
    A from-scratch worker release build is ~135–145 s here; an incremental no-op
    double build is ~23 s; a metallib rebuild is ~45–48 s. Recompiling zero
    objects and not relinking can be the correct outcome.
35. **A promoted submission can be a stale-base REPLACE overlay that silently
    reverts earlier frontier mechanisms.** `942e5ab2` discarded the
    wired-memory residency policy, the command-buffer geometry, and the M=8
    4+4 IPG split, added one 2-bit kernel, and still scored higher. Either the
    discarded three were worth ≈nothing, or the new kernel is worth more than
    the delta and is masked by three regressions.
36. **A supervised build job dirties the worktree through sibling
    `.build*.stale-<tag>/` directories, not `.build*/`.** `.gitignore` covers
    the latter only. Remove the stale siblings before asserting a clean tree.
37. **The Yukon poll subcommand is `yukon submissions` (plural), and it must run
    from the linked repository directory.** `yukon submission list` does not
    exist, and the plural form fails outside the workspace.
38. **A kernel-presence probe must be validated on a known-present symbol
    first.** `strings` on `mlx.metallib` finds nothing for kernels that are
    provably compiled in, so a negative result from an unvalidated probe is
    uninformative. Prove presence behaviourally through the dispatch gate.
39. **Read the real gate, not your reproduction of it.** Two "extra" gates in a
    local reproduction turned out to be verbatim script lines; one apparent
    invention was `submit-official.sh:382` itself.
40. **Test a "stale binary" hypothesis by asking which translation units that
    binary contains.** The CLI could not be stale with respect to a model-surface
    change because it never compiled that surface.
41. **A verifier must reuse the exact field name the script under test uses.** A
    checker reading `.matched` reported `null` forever against a harness that
    writes `.all_tokens_matched`.
42. **`2>/dev/null` on a `jq` extraction turns a missing field into a silent
    empty file.** `research/e26-legs.sh:156-157` filters a non-existent
    top-level `row_ledger` key, so every ledger artifact is 0 bytes with
    `sha256 = e3b0c442…` — the SHA-256 of the empty string. Never suppress
    `jq`'s stderr in an artifact-producing step.
43. **Estimate merge cost from the three-way diff shape, not the line-count
    asymmetry.** A 693-line ours versus 113-line theirs looked catastrophic; the
    `base..theirs` diff was `+38/−21` in two hunks and git auto-merged 100% of
    theirs' unique content.
44. **A ledger cross-reference is only as good as the number it cites.** This
    file carried "This is checklist item 53" while the checklist had never
    exceeded eleven items in any of its seventeen revisions. Re-check a forward
    reference whenever you renumber, and prefer naming the item over numbering
    it.
45. **A local ratio is a decoy; the absolute candidate wall time is the signal.**
    `decode_seconds` is prefill-inclusive (53.8% of it at 128 tokens, 22.9% at
    512 depth-2, 11.0% at 512 serial, 6.1989% ranked), so a `--local-submit`
    receipt dilutes a decode-only effect by ~2.2× and can invert the sign of a
    comparison.
46. **Minimise the delay between committing a kernel win and submitting it.**
    This is the scoop-risk item: the campaign held `DIRECT_NIBBLES` at M=3,4,5
    long enough for the frontier to promote the same mechanism independently at
    a measured `+1.56%`, which converted a novel result into a no-op. A
    committed-but-unsubmitted mechanism has no claim.
47. **`BASE_SHA` legality is an ancestry problem before it is a content
    problem.** `submit-official.sh:186-194` forces
    `base_sha ⊆ ancestors(merge_base(HEAD, origin/main)) ∪ {merge_base}`, while
    `:382` forces the protected surface at `base_sha` to equal the protected
    surface at `origin/main`. When the advisor branch has diverged from
    `origin/main`, *no* commit satisfies both, and the only repair is to merge
    `origin/main` into the advisor branch. Measured at this session's
    divergence: `c80c023` passed ancestry and failed `:382` on four paths;
    `5273067` passed `:382` with zero differing paths and failed ancestry until
    the merge landed.
48. **The tracker's identifiers are not the ones you announced; key every row on
    the receipt UUID.** Yukon's `commit` column reports a repackaged commit, not
    the candidate SHA the submitter announced. Observed twice: `74d1bd3a` was
    announced at `d11e01ea` and is reported as `07b2f1b5`; `9197ed62` was
    announced at `51c5b668` and is reported as `dbf91c6c`. The UUID is the only
    stable join key between an announcement, a ledger row, and a Yukon row, so
    record the announced SHA and the reported SHA as two separate facts and
    never assume a mismatch means the wrong tree was submitted.
49. **Read a result table's delta column as an arithmetic identity before
    quoting any part of it.** Yukon prints `diff` as
    `score − bar_at_evaluation`: `4437d061` scored `2.86126590369985` with
    `-0.28516`, and `2.86126590369985 + 0.28516 = 3.14642590…`, which is the
    `caec88d4` bar `3.14642585386152` to five decimals. That identity is worth
    having, because it lets a rejected row *measure the bar that was live when
    it ran* — applied to `9197ed62` it implies `≈3.2430006`, above the promoted
    `3.2341518328631`, which is a fact about the frontier that no organizer
    commit had yet published. The percentage printed beside the delta does not
    divide out against the score, the bar, or their difference, so do not quote
    it.

50. **A whole-file archive replacement silently evicts promoted mechanisms;
    ancestry is not mechanism currency.** A tree can descend from the commit
    that promoted a mechanism and still not contain it, because a later
    stale-base overlay replaced the file wholesale. This is not hypothetical:
    it is how `<T, 8, 4, true>` was reverted to `<T, 8, 3, true>` upstream
    while the comment above it kept citing a "register cliff" that E27 later
    refuted with the wrong sign. Before trusting any candidate tree, diff its
    *editable* paths against the frontier and confirm the mechanisms are
    present by content, not by ancestry.

51. **The tracker is an intelligence surface, not just a submission endpoint.**
    `yukon submissions --all` returns the entire board rather than our own
    rows; `yukon submission-note <uuid>` prints a competitor's full reasoning;
    `yukon notes list --author <user>` and `yukon benchmark show <id>` fill in
    the rest. Competitors' own notes supplied the batch-limit fact that proves
    crossrow changes reach the ranked box, three independent corroborations
    that a ranked `failed` is packaging rather than code, and a free negative
    (a wide target-top-2 reducer, −3.85 % at rank) that saved us building it.

52. **`editablePaths` comes from `benchmark.json` at `main_sha`, not from
    `senpai/frontier-state.json`.** `submit-official.sh:228` reads
    `git show "${main_sha}:benchmark.json"`. There are 89 entries, five of
    which are *directories*, and `protected_paths` is that list plus
    `benchmark.json` itself, so 90. A stale `frontier-state.json` therefore
    cannot block a submission, and reasoning about legality from it is
    reasoning about the wrong file.

53. **Yukon's printed percentage is `delta / 0.99245`,** where `0.99245` is the
    score an unmodified Qwen 3.8 tree makes. Verified on four rows spanning
    +2.15 % to −28.73 %. This amends item 49, which recorded that the
    percentage "does not divide out" — it does, just not against any quantity
    on the same row. The practical consequence is unpleasant: a printed
    "+2.15 %" is only about **+0.66 %** relative to a 3.21 frontier, because
    the denominator is the unmodified baseline and not the current bar.

54. **Our own submission notes were a severe one-sided leak.** Four notes
    published our trained head weights as publicly downloadable revisions with
    a demonstration that anonymous readback works, the full training recipe and
    sweep table, the dataset and sample construction, six unreleased forward
    levers, source paths with diff sizes, the measurement host's name, our
    thermal-gate weakness, campaign-internal file paths, and public W&B run
    URLs. Competitors state *in writing* that they mine each other's notes and
    branches. Policy is now **minimum-viable notes**: `program.md` requires only
    the exact lowercase `senpai` model label plus LLM and harness attribution.
    State the mechanism and the evidence; publish nothing else. This applies to
    `research/results/` files too — they are public.

55. **Never size an adaptive-policy result with a static-optimum model.** E27's
    own report understated a large realised decode win as +0.16 %, taken from
    `optimal_speedup_q100`, which optimises a *single static depth* across the
    whole run. The depth curve is flat near its optimum, so a 20 % cut at one
    width barely moves that argmax. The shipped policy is adaptive, and the
    changed widths carry real round mass, so the same change measured
    end-to-end is worth **−6.51 % decode time (×1.0697) on the longcopy
    fixture** and about **+2.49 %** re-costed against the shallower E21 tape.
    Both statements are arithmetically true; only the adaptive one is
    score-relevant. The student nearly talked himself out of the campaign's
    best mechanism.

56. **A bit-identical kernel change under a timing-blind policy is the only
    class of speedup with structurally zero token-fidelity risk — prefer it.**
    E27's kernel is bit-identical over 192/192 parity cells, and
    `costModelDepth()` reads the *constant* `headStepCostRatio`, never a
    measured timing. Therefore no selected depth and no emitted token can
    change, and this is provable before measuring rather than checked after.
    The end-to-end A/B confirmed it: all four legs returned an element-wise
    identical `effective_draft_lengths`, identical round, accept, reject and
    emit counts, `all_tokens_matched = true` and
    `residual_divergence_count = 0`. Contrast E25's arm D, which bought its
    speedup by refusing a draft row and lost 98 of 4098 tokens.

57. **`qmv_parity_compare.py` never exits nonzero.** Its verdict must be parsed
    from its text output. Any gate keyed on its exit code is vacuously green.
    Found and self-reported by the student whose own evidence chain it weakened.

58. **`qmv_cost_curve_summary.py:266` normalises every shape against M=1, so
    always include width 1 in `--widths`.** Omitting it makes the summary step
    exit nonzero *after* a sweep has already succeeded, which looks like a lost
    measurement and is not one.

59. **When a senpai GitHub mutation returns 403, retry the identical call
    before engineering around it.** Observed twice: a burst of successful
    mutations, then 403 on every endpoint, then spontaneous recovery on a later
    turn. Meanwhile the repository is public, so anonymous `curl` against
    `api.github.com` still reads PR bodies and comments at 60 requests/hour,
    and pre-registration ordering can be proven independently from
    `git log --format='%h %cI'` on a fetched `prhead/N` ref. A credential
    failure is not an outage, and it is not a reason to stop doing science.

60. **A metallib fingerprint is a sound witness for whether a kernel mechanism
    is present in a built tree.** Reconstructing the pre-merge kernel from
    source and rebuilding returned the metallib to *exactly* its former size
    and fingerprint (158,531,304 B / `12ad4a6a…`), and restoring the merged
    source returned it to 158,531,608 B / `1e359ea9…`. The 304-byte delta is
    precisely the two added template instantiations. This round-trip also gives
    a clean A/B method for kernel changes after they merge: `git checkout
    <pre-merge-sha> -- <the twins>`, rebuild, measure, restore, rebuild — with
    the fingerprint as the proof that each arm is what it claims to be. Note
    that `strings` on a metallib does **not** prove kernel presence; the
    fingerprint does.

61. **Yukon enforces a 5 KiB floor on the submission note, so "minimum-viable
    notes" cannot mean "short notes."** The submit refused a 3,848-byte note
    with a demand for "a complete, reproducible reasoning narrative." That
    collides head-on with item 54, and the resolution is a *shape*, not a length:
    go deep on the mechanism being submitted — which the merged diff already
    reveals to anyone who reads it, so describing it costs nothing — and stay
    silent on unreleased levers, head provenance, training recipes, host
    identity, W&B URLs, and anything about future direction. E27's note reached
    10.3 KiB while disclosing no asset a competitor could use, because the extra
    5 KiB went into failures, course corrections, and caveats. Confessing that
    our own harness undersold the result by 44x is honest, is exactly what the
    requirement asks for, and hands a rival nothing.

62. **An assertion string carrying a diagnostic message defeats a naive grep, and
    a gate keyed on a pattern that cannot occur is worse than no gate.** I built a
    tripwire around `static_assert(NA >= 2 && NA <= 5)` and it refused a build
    that was completely correct, because the real line is
    `static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");`
    — the trailing `)` never existed. The two-sided discipline that E28 applied
    to its content gate applies to *source-text* gates too: a new assertion must
    be shown to pass on a known-GOOD tree as well as fail on a known-BAD one.
    Confirm the count is 1, never merely that it is not zero.

63. **The frontier can advance on byte-identical code, so "sync to the new
    frontier" is sometimes vacuous — and the board's top is noise-limited.**
    `11863aa` was promoted at 3.24326223889754, superseding `4f76de6` at
    3.24300059379657, and `git diff --stat c0e34afd..5068eb8` is **empty**: the
    two organizer accept commits have the same tree,
    `b8642b81f72ff9214c74c654218a1bdc84fc2321`. The frontier advanced by
    re-measuring identical code, which puts ranked run-to-run noise at order
    **0.008%**. Two consequences. First, `program.md`'s "replay the candidate on
    the new base and measure it again" was satisfied without doing anything,
    because the new base *is* the base we measured on; always compare trees, not
    commit ids, before paying for a re-measure. Second, no result below roughly
    0.05% should ever be reported as a ranked win by anyone in this campaign.

64. **`--local-submit` must run with `MLXFAST_QWEN_MTP_HEAD_DIR` unset, exactly
    as the ranked runner does.** My gate script exported the research head
    directory `mtp-head-declared-q2q4`; `setup-qwen-mtp.sh:76` honours a pre-set
    value, and that directory is deliberately a *single-file* tree (the manifest
    pins its digest that way: "pinned to the runner's single-file
    model.safetensors tree digest"), so it has no `config.json` and
    `benchmark-qwen-mtp.sh:215` hard-refuses. Adding a `config.json` would have
    been the wrong fix twice over, because the tree digest covers every regular
    file except a top-level `README.md`. The declared head is resolved
    **runner-side, pre-sandbox**, and `QwenMTPHeadDeclaration.swift` says so
    explicitly: head provenance "never decides whether a run passes," having
    replaced three booleans that used to be pass-conditions with recorded fields.
    Setting that variable is right for `research/run-arms.sh` measurement and
    wrong for any `--local-iterate` or `--local-submit` invocation.

65. **`send_assignment_feedback` requires `status:wip`, so it is unavailable once
    a student posts a terminal result.** On a review-ready PR the comment vehicle
    is the `reason` field of `accept_result_on_current_base`, which is published
    as a PR comment. Compose the full adjudication there rather than discovering
    the restriction with a long comment in hand. Do not reach for
    `repair_assignment_routing` to force a PR back to `wip` for this — it exists
    for label drift, not for ordinary assignment decisions.

66. **Yukon CLI output is ANSI-colourised even when redirected to a file**, so
    the status column is `\033[32mpromoted\033[39m` and every column-position
    parse silently returns nothing. Strip escapes first
    (`sed 's/\x1b\[[0-9;]*m//g'`) before any `awk '$3=="promoted"'`. A parse that
    returns zero rows looks exactly like a board with no promoted rows, which is
    the worst possible failure mode for an intelligence query.

67. **THE RANKED SERIAL LEG IS NOT PINNED. It is the candidate's own build,
    measured in the same session.** I had recorded the opposite as an established
    fact, and it was load-bearing for E29's verdict and for every score
    projection I have written. Sources, three independent:
    `benchmark.json` `/scoring/scoreAnchor = "serial = 1.0"`,
    `/scoring/aggregation = "median_of_per_prompt_raw_serial_relative_speedup"`,
    `/scoring/noopReferenceRole = "informational_diagnostic_not_scored"`;
    `.github/workflows/qwen-mtp-ranked-benchmark.yml:128-129`
    `raw_p = mean(serial depth-0 seconds/token) / mean(MTP seconds/token)` then
    `score = median(raw_p over ALL 8 pool prompts)`; and `:3080-3090` in words —
    *"Both means come from the SAME thermally-gated session for that prompt, so
    the serial leg IS the normaliser and no pinned reference is consulted
    anywhere in the scoring path."*
    **Consequence: the score has TWO derivatives.** Anything that speeds up the
    `S=1` decode path LOWERS the score. Every experiment must now report BOTH
    legs; a ratio alone is ambiguous evidence and must be refused at review.
    Why I got it wrong: the pool's pinned `noop_decode_speedup` *was* the
    normaliser before the 2026-08-14 anchor move, and it is still carried in the
    contract and validated at `:3061-3068` — as a reported diagnostic only.

68. **The shipped `asyncEval` ladder is worth ~+20 % of SCORE if disabled,
    entirely by slowing the control leg. DECLINED, and escalated.**
    Mechanism: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:2118-2132`
    rung set `[0,1,9,19,29,39,49,57]` with an `MLX_QWEN_MTP_LADDER` override, and
    `:2196-2197` `ladderActive = inputs.dim(1) <= 9 || prefillLadder` — which
    covers the serial step and every legal MTP verify width.
    E29's four merged arms, schedules element-wise identical
    (`accepted_draft_total = 222` in all four):

    | arm | ladder | serial ms/tok | MTP ms/round | ratio vs default |
    |---|---|---:|---:|---:|
    | D0 | default | 66.8745 | 172.5586 | — |
    | L0 | off | 80.3441 | 172.2081 | **+20.38 %** |
    | L1 | front | 78.8334 | 172.1367 | +18.17 % |
    | L2 | dense | 65.3695 | 171.5106 | −1.65 % |

    The ladder helps the serial leg ~20 % and the MTP leg 0 %. It is therefore
    score-NEGATIVE by construction, and it is INHERITED organizer code, not ours.
    No mechanical guard exists — only `/scoring/decodeSpeedupFloor = 0.9` and
    `/scoring/decodeSpeedupCeiling = 5.0`. The static reviewer would probably not
    catch it either: `.github/scripts/run-submission-static-review.sh:433` lists
    *"MLX scheduling or synchronization changes"* among things it must NOT fail,
    and its `measurement-structure exploitation` clause is narrowly about
    input-keyed caching whose reuse requires the harness to repeat an identical
    computation.
    **DECISION: do not ship it, and do not hand it to a student as a shipping
    candidate.** A change whose entire effect is to slow the control so the ratio
    looks better is not optimization, whatever an automated judge would pass.
    Escalated to the human for a ruling; it stays a one-line change if they rule
    otherwise. Corroborating: with a ceiling of 5.0 and the top row at
    3.24326, nobody on the board is sitting at 4.x — so either nobody has found
    this or nobody is taking it.

69. **The command-buffer-geometry lever is closed, and it was closed by data we
    already owned.** The ladder IS the runtime commit-geometry knob: each rung's
    `asyncEval` forces a stream flush, i.e. one commit boundary. E29 therefore
    already covered 0 / 2 / 8 / 17 forced commits above MLX's automatic
    50-op/50-MB floor, and the MTP leg spread across that whole range is 0.61 %,
    non-monotone, below the 0.86 % repeat-noise floor. Going *below* MLX's
    automatic floor needs a patch to MLX internals. E31's accessible axis was
    fully measured before E31 was written — I found this one turn too late,
    after the assignment had gone out.

70. **The 53.86 % "host tail" is a ladder accounting artefact, not a cost.** Same
    four arms: host-tail share 53.86 / 4.35 / 5.66 / 35.86 % while round totals
    are 6028.7 / 6022.2 / 6015.8 / 5998.3 ms. The tail is where the host blocks
    *inside* `asyncEval` at the rungs. It moved 12× while the round moved 0.5 %.
    I had relayed that tail to two students as a thing worth attacking.

71. **Draft depth > 8 is hard-closed, permanently. Kill any proposal to widen
    beyond M=9 on sight.** `Sources/MLXFastCore/Constants.swift:331`
    `qwenMTPMaxDraftDepth = 8`, OPERATOR-RATIFIED; `MLXFastCore` is NOT in
    `editablePaths`; the TRUSTED parent enforces it in
    `QwenRuntimeMTPDriver.requireStructurallySound` over the draft count a round
    *actually* proposed; and the doc pre-empts the split-knob workaround — *"a
    submission that raised one and not the other would still be bounded by
    `qwenMTPMaxDraftDepth` at the parent."* Confirmed independently at
    `benchmark.json` `/scoring/mtpMaxDraftDepth = 8` and again in the
    static-review prompt. The wide-crossrow cells `M = 3..9` are the complete
    legal set, which is why that table stops where it does.

72. **`editablePaths` mixes DIRECTORY prefixes with exact file paths.**
    `Sources/MLXFastModel` and `Sources/MLXFastTransform` are directory entries;
    the other 87 are exact files. An exact-match membership test reports
    `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` as PROTECTED, which is
    false — it is the single most-edited file in the campaign. Test membership as
    "exact match OR path-prefix match". I briefly believed every scheduler
    experiment we had ever run was discarded at rank.

73. **Where the verify-forward GPU time actually is, and why the width lever is
    finished.** E20 time attribution: MLP **59 %**, GDN 28 %, full attention 8 %,
    LM head + top-two 5 %; MLP is also 65.1 % of the ~14.8 GB of byte traffic
    (three affine-4-bit 17408x5120 projections per layer x 64 layers). E23:
    dispatch count is NON-monotonic in M — 1096 at M=6..9 against 1544 at M=2 —
    so there is no dispatch-count argument for narrow widths and a
    dispatch-count argument against M=2 specifically. Weight passes are
    `ceil(M/NA)`; at NA=5 that is 1,2,2,2,2 for M=5..9, i.e. **already the
    minimum available for NA<=5 at every legal M**. The only remaining kernel
    prize is a lower pass count, which needs NA>=6, which spills (144 regs,
    2 allocas, against 125/1 at NA=5). `ceil(M/NA)` also bounds that prize
    before anyone builds it: NA=6 buys **M=6 only** (2 passes -> 1), which is 23
    of 78 rounds in the shipped depth histogram; M=9 is 34 of 78 and would need
    NA=9.

74. **`respond_to_human_issue` UPSERTS per `human_message_id`; a second call
    REPLACES the first response instead of appending.** I answered human message
    `5334188726` on issue 31 in one turn, then answered it again the next turn to
    escalate a new question, and the mutation returned
    `state: issue_response_upserted` with `resource_url` pointing at the SAME
    comment id `5334956828`. The earlier reply's content is gone from the thread.
    Nothing was lost that the ledger does not hold, but the failure mode is
    silent and the tool reports success. Rules: treat one human message as ONE
    reply slot, make that reply complete, and if a new question arises prefer
    waiting for a new human message over overwriting an answer nobody may have
    read yet.

75. **The GitHub REST API can 403 while git transport keeps working.** For
    roughly half a turn every `get_prs` / `send_assignment_feedback` /
    `create_assignment` call returned HTTP 403, while `git ls-remote`, `git fetch`
    and `publish_advisor_branch` all succeeded — they authenticate over a
    different path. It cleared with no action. The right response is not to idle:
    bank findings in the ledger and publish by push, keep local source analysis
    going, and retry the API afterwards. Do not run foreground sleep/poll loops
    waiting on it.

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
