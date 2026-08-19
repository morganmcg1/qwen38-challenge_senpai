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

67. 🔴 **RETRACTED — WRONG. See item 76.** The serial leg runs the pinned baseline
    tree; the denominator never moves. Kept here only so the mistake and its cost
    stay visible. Original text follows.

    **THE RANKED SERIAL LEG IS NOT PINNED. It is the candidate's own build,
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

68. 🔴 **RETRACTED — THE EXPLOIT DOES NOT EXIST. See items 76 and 84.** Disabling
    the ladder cannot slow a denominator we do not run; it would only slow our own
    candidate leg. The escalation is withdrawn. Kept here so the reasoning error
    stays auditable. Original text follows.

    **The shipped `asyncEval` ladder is worth ~+20 % of SCORE if disabled,
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

76. 🔴🔴🔴 **RETRACTION OF ITEMS 67 AND 68: the ranked serial denominator IS
    pinned, and candidate code cannot move it.** Last turn I "corrected" the
    campaign's standing premise on documentation alone. The premise was right and
    the correction was wrong. Four receipts, in ascending order of authority:
    - `.github/workflows/qwen-mtp-ranked-benchmark.yml:2907-2908`, describing the
      only timed measurement: *"Per accepted pair, alternating order: **baseline:
      pinned baseline tree**, serial K=1 target decode over the hidden benchmark
      golden; **candidate: this workspace**, native-MTP speculative decode over
      the SAME golden."*
    - the same file `:2374-2375`: *"the BASELINE leg keeps the section-9d pinned
      head out of the pinned baseline tree unconditionally. That asymmetry is what
      keeps the paired ratio anchored: **the denominator never moves**."*
    - `:2971` hands the wrapper `--baseline "${MLXFAST_QWEN_MTP_BASELINE_RESOLVED}"`,
      resolved from `/opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current`
      (`:224`), i.e. a box-owned tree, not the submission workspace. The contract
      agrees: `mtp_head_delivery.applies_to = "candidate_leg_only"`.
    - **the population test, which is the one that settles it.** Over the 402
      scored ranked submissions on this board (item 77), `baseline_serial_seconds_
      per_token_mean` has sd **0.106 %** (range 0.037907–0.038111) while
      `candidate_mtp_seconds_per_token_mean` has sd **25.9 %** (0.013929–0.041381).
      `corr(score, baseline_serial) = +0.049`; `corr(score, candidate_mtp) =
      −0.942`. The single worst candidate on the board decodes at 0.041381 s/tok
      (score 0.9266) and still reports a serial leg of 0.038035. 402 different
      candidate trees, one denominator.

    **What misled me** was the workflow's own comment at `:3083-3085`: *"Both means
    come from the SAME thermally-gated session for that prompt, so the serial leg
    IS the normaliser and no pinned reference is consulted anywhere in the scoring
    path."* Both halves are true and neither says what I read into them. The
    *session* is shared — which is exactly why the denominator's noise is only
    0.1 % — while the *tree* that runs the serial leg is the pinned baseline. "No
    pinned reference" retires the `noop_decode_speedup` divisor, not the serial
    binary.

    Consequences:
    - **Item 68's ~20 % `asyncEval`-ladder score exploit DOES NOT EXIST.** You
      cannot slow a denominator you do not run. Disabling the ladder can only slow
      our own candidate leg — `ladderActive` covers `S <= 9`, which includes the
      candidate's non-drafting `S = 1` rounds (451 of them on plutarch alone) and
      every verify width. The escalation to the human is **withdrawn**; there was
      never a decision to make.
    - **There is exactly ONE derivative: candidate MTP seconds per token.**
      `corr = −0.942` is the whole game.
    - Reporting the serial leg is still worth doing, but as a **free in-session
      thermal control** — a population sd of 0.106 % makes it the tightest
      instrument on the board — never as a term we can move.
    - The E30/E31/E32 briefs that carried the two-derivative claim were corrected
      in writing the same turn.

    **Process lesson, and it is the expensive one:** I shipped a correction to
    three students and an ethics escalation to the operator from documentation
    alone, when the data that refutes it was one authenticated GET away. *When a
    scoring claim can be tested against the board's own telemetry, test it before
    publishing it.* Publishing a retraction one turn later costs more than the
    hour the test would have taken.

77. 🟢🟢🟢 **The ranked benchmark publishes FULL PER-PROMPT TELEMETRY for every
    submission on the board — ours and every competitor's — and the campaign
    never read it.** One authenticated GET:
    ```
    curl -H "Authorization: Bearer $YUKON_API_TOKEN" \
      https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions
    ```
    `YUKON_API_TOKEN` is already in the advisor shell environment; the benchmark
    id comes from `yukon benchmark list`. The response is ~8 MB, 623 submissions,
    each carrying `status`, `officialScore`, `submissionCommitSha`,
    `promotionStatus`, `promotedSourceRef`, the solver's full public `note`, and
    `officialMetrics` with **`per_prompt[8]`**: `raw_ratio_of_means`,
    `serial_seconds_per_token_mean`, `mtp_seconds_per_token_mean`,
    `effective_mean_draft_len`, `non_drafting_round_count`,
    `prefill_seconds_per_token`, `head_provenance_sha256`, `parity_ok`. Prompt
    identity comes from `fixtures/qwen3_8_27b_mtp_track.json`
    `timed_prompt_pool[].sha256 -> r2_path`.
    Why it sat unread: `yukon submissions` hard-truncates the `metrics` column at
    80 characters, has no `--json`, and ignores `COLUMNS`. `GET
    /api/submissions/<id>` needs the full UUID, which the table does not print —
    take UUIDs from the list endpoint or from `git ls-remote upstream
    'refs/heads/submissions/*'`. **Read this before designing any experiment.**

78. 🔴🔴 **The score is decided by exactly TWO of the eight prompts, and they are
    `beagle` and `medicine`.** `median_rule =
    even_n_mean_of_two_central_order_statistics`, so with n=8 the score is the
    mean of the 4th and 5th ranked per-prompt ratios and `d(score)/d(prompt)` is
    **0.5** for those two and **0** for the other six. At the frontier row
    (`b0994092`, 3.24418) the sorted ratios are

    | rank | prompt | ratio | mean draft rows |
    |---|---|---:|---:|
    | 1 | plutarch | 1.253 | 0.15 |
    | 2 | drama | 1.924 | 2.30 |
    | 3 | travel | 2.188 | 2.66 |
    | **4** | **beagle** | **3.141** | **4.53** |
    | **5** | **medicine** | **3.347** | **4.77** |
    | 6 | essays | 3.394 | 5.43 |
    | 7 | republic | 3.420 | 5.27 |
    | 8 | botany | 3.449 | 5.78 |

    (4th + 5th)/2 = 3.2442 ✓. **The bottom three are unreachable and worth
    nothing**: travel would need +43 % and plutarch +150 % merely to enter the
    window. The receipt that this is arithmetic and not theory: WillGasser's
    `de7981ae` scores **plutarch at 2.175** with 6 non-drafting rounds where every
    other frontier clone scores 1.253 with 449 — a **+74 % gain on that prompt** —
    and his board score is 3.24078, inside the clone band. Kill any proposal aimed
    at plutarch, drama or travel. Beagle is also the *low* member of the top
    cluster (3.141 against 3.347–3.449) and has the shortest drafts of the five,
    so it is simultaneously the cheapest and the highest-leverage prompt on the
    board.

79. **Ranked repeatability, measured properly.** Per-leg session noise is sd
    **0.106 %** (the serial-leg population, item 76). Two byte-identical trees
    (`c0e34afd`, `5068eb8d`) scored 0.008 % apart — a lucky pair, not the noise.
    The top eight rows share one head digest and one per-prompt draft-length
    vector yet span **3.23223–3.24418 (0.37 %)**. Working rule: **< 0.5 % is not
    resolvable on one ranked run; require ≥ 1 % projected gain before spending a
    slot**, and never quote the 0.008 % pair as the noise floor again.

80. 🔴🔴🔴 **The MTP HEAD is the dominant lever on this leaderboard, and our own
    trained head cost us ~5 % of score.** Ranked maxima by `head_provenance_sha256`:

    | head digest | subs | solvers | best score |
    |---|---:|---:|---:|
    | `157f750e` (organizer-pinned) | 66 | 19 | 2.834 |
    | `cc209e30` | 107 | 52 | 2.930 |
    | `7d627027` | 64 | 27 | 3.086 |
    | `477ba726` | 46 | 21 | 3.168 |
    | **`559b24eb`** | **83** | **30** | **3.244** |
    | `5cbc5537` (ours) | 1 | 1 | 2.861 |
    | `2f6805e1` (ours) | 1 | 1 | 3.069 |

    Every generation is a competitor-published head on Hugging Face at an
    immutable revision, adopted by dozens of solvers through the promoted tree's
    `mtp-head.manifest.json`. The board's whole trajectory is that ladder. Our
    rejected 3.069 row declared **our own** head
    `hf:morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank@fd4a99c5` (`2f6805e1`,
    427,742,680 B, "40-tensor LR3 refinement"); the frontier declares
    `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274` (`559b24eb`,
    427,742,600 B). At comparable code our head proposes **5–13 % fewer draft rows
    on 7 of 8 prompts** (beagle 4.198 vs 4.53, botany 5.040 vs 5.78, republic
    4.683 vs 5.27, essays 5.000 vs 5.43, medicine 4.519 vs 4.77, drama 2.163 vs
    2.30, travel 2.575 vs 2.66) — it is simply worse at acceptance, and the
    contract says acceptance is the game
    (`mtp_head_delivery.safety_argument`: *"a substituted head can move the accept
    rate — which is the game — and cannot move the output"*). Draft-length
    *decisions* cannot be explained by the kernel or memory-policy lines that also
    differed, which is what makes the attribution safe. The pending `ca9251b8`
    declares `559b24eb`, so the loss is already reverted. **Rule: never ship a head
    that has not beaten the best public head on ranked evidence. It is one JSON
    file in `optionalEditablePaths`, and getting it wrong is worth more than every
    kernel result the campaign has produced.**

81. 🔴🔴 **`yukon submit` is a REPLACE overlay, so anything the promoted tree has
    and our branch lacks is silently REVERTED by our own submission.**
    `git diff --stat 5068eb8d dbf91c6c` (promoted → our rejected 3.069 row):
    `Qwen36MTPBlockSession.swift` **−83**, `RuntimeStartupMemoryPolicy.swift`
    **−32**, `quantized.h` −16, manifest ±8. Among the deletions was the promoted
    tree's `wireResidentWeightsIfEnabled` wired-memory ticket, gated on
    `physicalMemory >= 96 GiB` — it fires on the 128 GB ranked box and **can never
    fire on our 48 GiB M4 Pro**, so we cannot measure it locally and must not
    delete it. (Do not confuse it with the competitor negative "raising the wired
    limit *with headroom* is harmful"; this is the promoted variant,
    `wiredZHDefaultFraction = 1.0`, slack 64 MB.) Current HEAD keeps it and
    `RuntimeStartupMemoryPolicy.swift` is byte-identical to promoted. **Standing
    pre-submission gate: `git diff --stat <promotedSourceRef> HEAD -- Sources
    Vendor mtp-head.manifest.json`, and account for every deletion in writing.**
    `promotedSourceRef` is a field on the promoted submission record (item 77);
    today it is `5068eb8d0bae032faca6e901de398fc732531160`.

82. 🔴🔴 **The local box under-states the value of speculation by ~1.34×, so it
    lies about width and depth economics.** Same E27 build, two boxes:
    `--local-submit` score 1.7660 (serial 0.097995, MTP 0.055489 s/tok) against
    ranked 3.0694 (serial 0.037975, MTP 0.016055). Ranked serial is 2.58× faster
    than local serial; ranked MTP is **3.46×** faster than local MTP. **The MTP
    leg gains 1.34× more from the ranked hardware than the serial leg does.**
    Two caveats, stated because the magnitude is softer than the direction: the
    local window is ~300 tokens and prefill-inclusive while ranked is 512 with
    prefill broken out, and the two fixtures place different mass on width
    (local mean draft 6.22, ranked 4.20–5.04 for the same head). The crudest and
    most honest single receipt is that **the same build scores 1.766 locally and
    3.069 ranked, a factor of 1.74**. Treat 1.34× as a lower bound on the leg
    asymmetry and the *direction* as certain.
    Consequences: (a) a depth/width policy tuned on local timings is tuned to the
    wrong cost curve; (b) locally, *narrowing and shallowing* look profitable and
    on the ranked box they are not — the frontier's ranked mean draft is 4.53–5.78
    rows while our local prose tapes sit at 2.39; (c) a locally measured width
    cliff (E25's `stepRatio[3] = 0.442`) may not exist on the scoring box and must
    never be used to size a ranked gain on its own.

83. 🔴 **Ranked width mass, measured, per prompt — weight every per-M kernel claim
    with it.** Frontier `effective_mean_draft_len`: botany 5.78, essays 5.43,
    republic 5.27, medicine 4.77, beagle 4.53, travel 2.66, drama 2.30, plutarch
    0.15. Verify width is `M = drafts + 1`, so the two prompts that decide the
    score (item 78) sit at **M ≈ 5.5 and M ≈ 5.8** — not at the M=8/9 mass of our
    local E27 fixture (mean draft 6.2692, histogram `{1:1,3:5,4:5,5:23,6:4,7:6,8:34}`).
    E27's per-M ratios are M=5 **0.7990**, M=6/7/8 ≈ 1.00, M=9 0.8854, so the
    ranked prize is concentrated at **M=5 and M=6**, and the M=6 pass-count cell
    that `NA=6` alone would buy is worth considerably more than the local
    histogram implied.

84. **The `asyncEval` ladder is inherited from a competitor's promoted submission,
    not from the organizer baseline.** `git grep DecodeLadderRungs 5d02917` (the
    yukon baseline) returns nothing; the promoted `5068eb8d` carries the same rungs
    `[0, 1, 9, 19, 29, 39, 49, 57]` hardcoded as a `switch`, with
    `ladderActive = inputs.dim(1) <= 9 || prefillLadder`. Our branch's only change
    is refactoring that switch into an overridable `Set` (`MLX_QWEN_MTP_LADDER`,
    `research/e29-run.sh`) plus a stale-comment fix from "S <= 2" to "S <= 9". So
    every frontier row ships it, ours included, and per item 76 it is an ordinary
    candidate-leg latency optimization — **keep it**. E29's local +20.38 %
    serial-leg number is a property of the *local* harness's serial arm and has no
    ranked analogue.

85. 🟢🟢 **`plutarch` is a free ranked read-out of the candidate's OWN `S = 1`
    decode speed, and it says the frontier build's single-token path is ~1.25×
    faster than the pinned baseline's.** On plutarch the policy declines to draft:
    `non_drafting_round_count = 449`, `effective_mean_draft_len = 0.15`, and the
    ratio is **1.253** (ours 1.249). With almost no drafting the candidate's round
    *is* an `S = 1` decode, so the ratio reduces to (pinned baseline `S = 1`) /
    (candidate `S = 1`) — about a 20 % candidate-side win before any speculation.
    That is where E29's local +20.38 % `asyncEval`-ladder number actually lands:
    an honest candidate-leg latency win on the `S = 1` path, which every frontier
    row already ships (item 84). Two uses: (a) plutarch's ratio is the cheapest
    ranked instrument we have for "did this change make the `S = 1` path faster or
    slower?", nearly independent of the drafting policy; (b) the same `S = 1` floor
    sits inside every other prompt's round, so an `S = 1` improvement is one of the
    few changes that lifts beagle and medicine *and* everything else at once. Do
    not confuse this with item 78: raising plutarch's own ratio scores nothing, but
    the quantity it measures is worth real score everywhere else.

86. **The `M = 8` dispatch cell is contested between two receipted competitor
    trees, and our branch takes one side without having measured it.** Promoted
    `5068eb8d` dispatches `qmv_fast_crossrow_affine4_g64_m<T, 8, 3, true>` (IPG 3,
    "3+3+2", `ceil(8/3) = 3` weight passes) with an in-code claim of 319/437/216 µs
    at M=7/8/9 and "a register cliff, not work scaling"; our HEAD dispatches
    `<T, 8, 4, true>` (IPG 4, 2 passes) citing a promoted receipt of
    3.195804751396457. Both are legal (`8 % 4 == 0`, `8 % 3 == 2`). One cell, no
    correctness risk, two contradicting competitor receipts, and by item 83 the
    ranked mass at M=8 is smaller than at M=5–6. Queue it; do not rush it.

87. 🔴🔴🔴 **Ranked `effective_mean_draft_len` is bit-identical across the entire
    top of the board, so the whole leaderboard is competing on ONE axis: per-row
    verify cost.** Across the top 14 accepted rows (14 different solvers,
    3.19088–3.24929), `effective_mean_draft_len` has standard deviation
    **0.000 %** on every one of the eight prompts — beagle 4.5327103, medicine
    4.7677, essays 5.4253, republic 5.2700, botany 5.7757, travel 2.6557, drama
    2.2976, plutarch 0.1540, repeated to every printed digit. Meanwhile
    `raw_ratio_of_means` varies 0.26 % (plutarch) to 0.86 % (botany). Identical
    accepted-row trajectories with different ratios means **every point of
    separation at the top is decode speed at a fixed operating point**, not
    acceptance. Corollary: the accepted/rejected row sequence is a deterministic
    function of (head, target checkpoint, prompt) and the default schedule, and
    73 of the 86 scored rows that declare head `559b24eb` sit on exactly that
    default. Reproduce with `/tmp/decomp.py`, `/tmp/nmove.py`.

88. 🔴🔴🔴 **More rows is NOT better at the ranked operating point: the implied
    per-row price rises steeply above the default depth, and every submission
    that drafted deeper than the default scored worse.** Within head `559b24eb`,
    solving `h̄ = [(1+0.99 n)/R − 1]/n` on beagle:

    | beagle n | beagle R | implied h̄ | score | solver |
    |---:|---:|---:|---:|---|
    | 4.2793 | 3.1103 | 0.1598 | 3.18367 | jeromelaurens |
    | **4.5327 (default)** | **3.1433** | **0.1645** | **3.24929** | **ofou (board top)** |
    | 4.3839 | 3.0603 | 0.1699 | 3.16292 | xadenryan |
    | 4.6019 | 3.0062 | 0.1843 | 3.07216 | andreolf |
    | 4.7358 | 3.0626 | 0.1810 | 3.16408 | welttowelt |

    Nine non-default rows, all below the default cluster's best. Going from
    n = 4.533 to n = 4.736 made beagle's `mtp_seconds_per_token_mean` 2.6 %
    *worse* (0.012130 → 0.012449). Confounded — these are different kernels, so
    this is not a clean marginal-price measurement — but the sign is uniform
    across nine rows. **The marginal row at the ranked operating point costs more
    than it returns.** This is edward's E25 thesis (the price curve is a step
    function, not the shipped smooth `h/(1+d·h)`) confirmed on ranked data, in a
    depth region his local fixture never visits.

89. 🔴🔴🔴 **The score-maximising schedule is not the throughput-maximising
    schedule, because the objective is an order statistic.** Two competitors got
    plutarch to draft — n = 0.154 → **2.897**, ratio 1.2489 → **2.2209**, a
    **+78 %** gain on that prompt, with `non_drafting_round_count` 449 → 0 — and
    both still scored *below* the board top (xadenryan 3.16292, Lieisyourlie
    3.15370). Lifting plutarch moved it from rank 1 to rank 3; the central pair
    stayed beagle+medicine; and the same aggression pushed beagle's h̄ from
    0.1645 to 0.1699, losing on the prompt that binds. **Aggregate throughput
    work on the six non-binding prompts is worth exactly zero, and is negative
    whenever it raises per-row cost on beagle or medicine.** Retire item 85's
    implicit suggestion that plutarch is a prize; it is an instrument only.

90. 🔴🔴🔴 **The binding widths are M = 5, 6, 7, and `M = 6` is the worst per-row
    width in the dispatch table — this is the single highest-value remaining
    kernel target.** `M = n + 1`, so beagle's n = 4.5327 → **M ≈ 5.53** and
    medicine's n = 4.7677 → **M ≈ 5.77**. Kernel receipt,
    `quantized.h:1154`: *"IPG = ceil(M / ceil(M / 5)): the fewest weight streams
    reachable at NA <= 5"*. So weight passes are `ceil(M/5)` and per-row weight
    traffic is `passes/M`:

    | M | IPG | passes | per-row traffic |
    |---:|---:|---:|---:|
    | 3 | 3 | 1 | 0.333 |
    | 4 | 4 | 1 | 0.250 |
    | 5 | 5 | 1 | **0.200** |
    | **6** | **3** | **2** | **0.333 ← worst** |
    | 7 | 4 | 2 | 0.286 |
    | 8 | 4 | 2 | 0.250 |
    | 9 | 5 | 2 | 0.222 |

    Two consequences. (a) **The current table is already pass-optimal at NA ≤ 5**
    — every cell equals `ceil(M/ceil(M/5))`, so there is no IPG win left without
    raising NA, and this also settles item 86 in our HEAD's favour: `<T,8,4>` is
    2 passes, the promoted `<T,8,3>` is `ceil(8/3) = 3`. (b) **M = 6 cannot go
    below 2 passes without NA = 6** (IPG 6 → 1 pass; IPG 5 is illegal because
    `6 % 5 == 1`; IPG 4 and 3 are both 2 passes), and NA = 6 costs 144 registers
    with 2 allocas at `rows_per_simd = 4`. NA = 6 buys exactly one cell, M = 6,
    taking its per-row traffic 0.333 → **0.167, the best in the table.** That is
    askeladd's E32 register-budget trade, and it is now the campaign's main line.

91. 🔴🔴 **Independent confirmation that M = 6 is the prize, from a completely
    different instrument.** Alphonse's E30 (PR #35) measured, post-E27, that the
    best width flipped 8 → 9 and that **70 % of the entire remaining best-width
    headroom on his fixture is the nine M = 6 rounds** (160.3 of 229.8 ms;
    M = 6 costs 23.71 ms/tok against the best width's 20.74). Two instruments
    that share no code — the ranked per-prompt telemetry via `M = n + 1`, and a
    local six-way round tape — independently name M = 6. Treat that convergence
    as the strongest prioritisation signal in the campaign so far.

92. 🔴🔴 **E25's binding price coefficient was a pass-count boundary, and E27
    moved that boundary — so the whole depth-truncation family was chasing a
    stale artifact.** Edward's measured `measuredRowStepRatio[3] = 0.442442`
    (against the shipped 0.153) is the `T(3) → T(4)` step, i.e. **M = 4 → M = 5**.
    Pre-E27, M = 5 dispatched `<T,5,3>` = `ceil(5/3)` = **2 passes** while M = 4
    was 1 pass, so his "cliff" was exactly the 1 → 2 weight-pass boundary. E27
    (`cf2c0db2`, 2026-08-18 22:56) made M = 5 single-pass; his base
    `0d2eef9c` is 2026-08-18 04:49 and `git merge-base --is-ancestor cf2c0db2
    0d2eef9c` returns non-ancestor. Propagating E27's measured M = 5 ratio 0.7990
    through his own table: `T(3) = 91.2664`, `T(4)_pre = 131.658`,
    `T(4)_post = 105.195`, **new d3 coefficient 0.1526 versus the shipped 0.153.**
    Pre-registered prediction for the E25 r4 refit: (i) the d3 coefficient
    collapses onto the shipped curve and arm D's mechanism evaporates; (ii) **a
    new cliff appears one step later, at d4 = the M = 5 → M = 6 step**, because
    that is where the 1 → 2 pass boundary now lives. This is the load-bearing
    generalisation: **the true verify cost is a step function of `ceil(M/5)`,
    which the smooth `h/(1+d·h)` price cannot express at all.**

93. 🔴🔴 **Arm D would have been a structural ~18 % ranked loss, and the local
    evidence said the opposite.** E25's PRICE arm sets max depth to exactly 3
    (`depth_ge_4_realised = 0`) and wins 8/8 locally at +3.83 %. But its local
    tape has mean depth 2.386 (per-prompt 2.058–2.737) while **every scoring
    prompt realises 4.53–5.78** — the fixture never visits the scored region.
    Holding α = 0.99 and each prompt's h̄ fixed and moving n to 3 collapses all
    five deep prompts into 2.66–2.74, putting the 4th/5th pair at ≈2.66, i.e.
    **−18 %**. Assumption-free version: to *hold* 3.244 at n = 3 you would need
    h̄ ≤ **0.0746** on the central prompts, under half the shipped price. Add to
    the established-negative list: **any hard depth cap below 6.** Generalise the
    lesson: a local fixture whose depth histogram does not overlap the ranked
    one cannot test a depth policy at all, in either direction.

94. 🟢🟢 **Local and ranked do not even run the same command-buffer geometry**, so
    the local↔ranked transfer gap of item 82 now has a second named mechanism.
    Thorfinn's E31 (PR #36) found that the scored worker force-installs the
    geometry from editable Swift before the first Metal device touch: **50 ops /
    512 on the ranked M5 Max, 64 ops / 128 on this 48 GiB host** — never the arch
    default, and never the same on both. E31 also corrects two inherited claims:
    `MLX_MAX_MB_PER_BUFFER` accumulates `array::data_size()`, which `array.h:346`
    documents as *units of `item_size`, not bytes*, so it is a **mebi-element**
    cap admitting 4× the bytes its name implies for the 4-bit `uint32`-packed
    backbone; and `gpu::finalize` commits without `notify_new_task`, so ladder
    rungs never enter the `MAX_ACTIVE_TASKS = 10` in-flight accounting. Terminal
    negative on the axis itself: per-boundary cost **−36.5 µs (95 % CI −170.3 …
    +97.4)**, consistent with zero and negatively signed.

95. **Board top moved to 3.24929398547457** (`0cd0a6b4`, solver `ofou`),
    superseding 3.24417896624589 (`b0994092`, fkiene) and the still-*promoted*
    3.24326223889754 (`11863aa9`, companygardener, `promotedSourceRef
    5068eb8d0bae032faca6e901de398fc732531160`). All top 12 declare head
    `559b24eb`, which is the head our pending `ca9251b8` also declares. The top-14
    span is 3.19088–3.24929 = 1.8 %, and the top-10 clone cluster spans 0.53 %
    against a per-prompt ratio spread of 0.26–0.86 % — so **most of the visible
    ordering at the top of this board is noise**, and item 79's ≥1 % rule before
    spending a ranked slot stands reinforced.

96. 🔴🔴🔴 **No local experiment is running the head that ranked scoring runs, and
    at least three different head trees are in play across our own students.**
    `mtp-head/README.md` states the rule: *"with a pinned or remote source the
    runner NEVER reads weights from this directory"* — and our `mtp-head/`
    contains exactly one file, the 2410-byte README. Our manifest declares
    `"source": "remote"`, `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae62827`,
    sha256 `559b24eb…`, **427,742,600 B**, described in its own note as *"the
    runner's single-file `model.safetensors` tree"*. So the ranked head is fetched
    out of band by the runner and never exists in our checkout. Meanwhile:

    | run | declared head | tree sha256 | bytes |
    |---|---|---|---:|
    | ranked (all top rows, our `ca9251b8`) | amal-david remote | `559b24eb…` | 427,742,600 |
    | alphonse E29/E30 local | amal-david remote | `7bbb40de…` | **270,408,194** |
    | edward E25 local | organizer pinned | `07293af7…` | not reported |

    Consequences. (a) **Every local acceptance number is measured on the wrong
    head.** Alphonse's local `effective_mean_draft_len` of 6.5143 and accept rate
    0.9737 against ranked 4.53–5.78 is therefore *partly a head difference*, not
    only a prompt difference — which is a third named mechanism for the item 82 /
    94 transfer gap, alongside 300-vs-512 tokens and the 50/512-vs-64/128
    command-buffer geometry. (b) E30's conclusions **survive** anyway, because its
    depth histogram was bit-identical across all three arms, so the width-cost
    finding is within-head. (c) Any future experiment whose dependent variable is
    acceptance, accept rate, or realised depth **must** declare which head tree it
    loaded and is otherwise uninterpretable. (d) Two students silently used
    different heads for work I compared side by side; that is my failure to
    specify, and head provenance now belongs in every assignment's evidence
    contract. Also note `mtp-head/README.md` is itself stale where it says *"the
    checked-in declaration selects `source: pinned`"* — it selects `remote` now.
    Ranked enforcement of the declared `sha256` **and** `bytes` is exact-equality
    at workflow `:2491-2502`, so the 158 MB gap would be a hard refusal if we ever
    declared the local tree.

97. 🔴🔴🔴 **The scoring prompts run predominantly at depth 5 = M = 6, which is
    both the worst per-row width and the width nobody on the board has touched.
    Two experiments size at +6 % and +25 % before discounting.**
    `effective_mean_draft_len` is an exact rational, so the counts recover
    exactly (`/tmp/ratio.py`). Board top `0cd0a6b4`: beagle **485 accepted rows
    over 107 drafting rounds**, medicine 472/99, essays 472/87, republic 469/89,
    botany 491/85, drama 193/84, travel 563/212, plutarch 75/487 with 449
    non-drafting.

    Now use the schedule: `sdpaWidthWallDepthCap = 5` caps depth at 5 (M ≤ 6)
    unless `fullAcceptStreak >= 2` opens `segmentedVerifyDepthCap = 8` (M ≤ 9).
    Under a geometric accept model `n(d, p) = Σ_{i=1..d} p^i`, beagle's
    n = 4.5327 is reproduced by **depth 5 at p ≈ 0.965** (n = 4.499), whereas
    depth 8 at the same p would give **6.84** — far too high. So the central
    prompts cannot be depth-8 dominated; they sit at depth 5, i.e. **M = 6**.
    (And in the alternative reading, a depth-5/depth-8 mix, *both* widths are
    2-pass, so the conclusion below is unchanged.)

    Sizing, calibrated off E27's measured M=5 two-pass→one-pass ratio 0.7990,
    which decomposes a 2-pass round into `fixed = 0.598` and `stream = 0.201`
    per pass (`/tmp/prize.py`):

    | option | change | cost/token | score sizing |
    |---|---|---:|---:|
    | **A** | NA=6 makes M=6 single-pass (askeladd E32) | **−20.1 %** | 3.249 → **~4.07 (+25 %)** |
    | **B** | snap the depth cap 5 → 4, so every round is single-pass M=5 | **−5.8 %** | 3.249 → **~3.45 (+6.1 %)** |

    Option B costs 0.84 accepted rows per round and is a **one-constant change**
    with no kernel work. Note it is the *opposite* of E25's arm D: arm D capped at
    depth 3 and lost ~18 %, while depth 4 is exactly the largest single-pass
    width. That is the whole content of item 92's step-function insight.

    **Discount honestly.** Both numbers are upper-leaning: A assumes every round
    is M=6 and that the full M=5 pass saving transfers to M=6; B assumes the
    accept rate is unchanged at shallower depth, which is precisely what E25 must
    test; and the fixed/stream split is transferred from one width on one fixture.
    A at half the round share and two-thirds of the saving is still **+8 %**. The
    bar is 1 %. Both are worth a ranked slot, and A is worth the register fight.
    Sanity check in favour of a large untapped prize: per their own public notes,
    the competitors have worked **M=8 only**, and no note among 626 mentions IPG,
    `rows_per_simd`, `ceil(M/IPG)`, NA, or widths 5/6/7.


98. 🔴🔴🔴 **The declared head artifact is the single strongest determinant of
    ranked score in the entire population — and NOT ONE of our scored submissions
    has ever used the winning one, so we have never had a valid ranked
    measurement of our kernel work.** `head_provenance_sha256` is the *declared
    head artifact* digest, not a runtime fingerprint; receipt
    `research/e11-notes.md:1101`, "declared head (`head_provenance_sha256
    07293af7...`, 2 files, 238937699 bytes)". Grouping all 627 ranked submissions
    by that field:

    | head artifact | rows | best | median |
    |---|---:|---:|---:|
    | **`559b24ebca35`** | **87** | **3.24929** | **3.18463** |
    | `2edd8b91f222` | 3 | 3.19853 | 3.18331 |
    | `477ba7266c6f` | 46 | 3.16766 | 3.10262 |
    | `7d6270279586` | 64 | 3.08598 | 2.93339 |
    | `2f6805e1c8b7` (**ours**, LR3) | 1 | 3.06938 | 3.06938 |
    | `cc209e30d8a7` | 107 | 2.92976 | 2.86532 |
    | `157f750eb467` | 66 | 2.83386 | 1.95397 |

    **The `559b24eb` population's MEDIAN (3.18463) beats every other artifact's
    BEST**, and no other artifact has ever exceeded 3.19853. Our own history:
    `4437d061` → `5cbc5537` (2.86127), `9197ed62` → `2f6805e1` (3.06938, the LR3
    head), two `failed` rows with no metrics, and `ca9251b8` still validating.
    Count of our scored rows on `559b24eb`: **zero**. The ledger already knew
    both scored rows were head plays (rows at `:568` and `:571`); what it did not
    draw is the structural consequence — **the "we are 5.9 % behind the top"
    number is an artifact of the head excursions and says nothing whatever about
    our kernels.** `ca9251b8` (E27 + promoted memory policy, head `559b24eb`) is
    the first honest ranked datapoint the campaign will ever have, which makes it
    the most valuable pending unknown. Do not size any kernel decision against
    `9197ed62`.

    Mechanism, per prompt, ours (`9197ed62`) vs board top (`0cd0a6b4`) — a worse
    head shows up as lower acceptance, and the s/tok penalty tracks it
    monotonically:

    | prompt | n ours | n top | Δn % | Δ mtp s/tok % |
    |---|---:|---:|---:|---:|
    | plutarch | 0.160 | 0.154 | +3.57 | +0.65 |
    | drama | 2.163 | 2.298 | −5.84 | +2.44 |
    | travel | 2.575 | 2.656 | −3.05 | +2.77 |
    | beagle | 4.198 | 4.533 | −7.38 | +3.63 |
    | medicine | 4.519 | 4.768 | −5.21 | +3.33 |
    | essays | 5.000 | 5.425 | −7.84 | +3.20 |
    | republic | 4.683 | 5.270 | −11.13 | +6.05 |
    | botany | 5.040 | 5.776 | −12.76 | +9.88 |

    Two corollaries worth keeping. (i) `qwen_mtp_weights_hash` is **identical**
    (`b53e4991…`) between our row and the top's, so the backbone is not in play —
    only the head. (ii) 🟢 **First ranked-side confirmation of the two-derivative
    rule**: our `baseline_serial_seconds_per_token_mean` was 0.0379753 against
    the top's 0.0380649 — our serial leg was **0.24 % faster, which cost us
    0.24 % of score**. The retracted "pinned denominator" claim is now refuted
    *and* the corrected model is confirmed from ranked telemetry, not from
    documentation. (iii) This makes item 96 more important, not less: local runs
    resolve `7bbb40de` (270,408,194 B) and `07293af7` (238,937,699 B) rather than
    the declared `559b24eb` (427,742,600 B), i.e. every local acceptance number
    in the campaign is measured on an artifact the population test says is
    materially worse. Per-width *cost* ratios are within-head and still transfer;
    acceptance, realised depth and round-mass weighting do not.

99. 🟢🟢🟢 **E32 (askeladd, terminal, green): NA ≥ 6 is reachable spill-free, my
    proposed mechanism for getting there was wrong, and the replacement is
    free.** I told him to buy NA by spending `rows_per_simd`. That is a
    **correctness wall, not a trade**: the host dispatch at
    `backend/metal/quantized.cpp:251-254` is frozen (`bn = 8`,
    `group_dims(32,2,1)`, `grid.y = (N+7)/8 = 2176`) and is **not** in
    `editablePaths`, so any `r < 4` leaves rows of `N = 17408` unwritten — r=2
    would compute 8704 of 17408 rows. Had thorfinn built what I described he
    would have shipped silently wrong logits. The replacement covers the same 4
    rows as `4/r` **sequential row blocks**: registers are live-range-bound so
    residency halves, weight traffic is unchanged per block, and only the cheap
    activation read is repeated. Decision table (coverage-preserving, frozen grid
    intact):

    | M | shipped | passes | target | regs/alloca | passes | ALU | weight passes |
    |---:|---|---:|---|---|---:|---|---|
    | 3/4/5 | `<T,3,3>`/`<T,4,4>`/`<T,5,5>` | 1 | unchanged | 83/104/125 | 1 | +0 % | — |
    | **6** | `<T,6,3>` | **2** | **NA=6, r=2 blocked** | **117/1 clean** | **1** | **+6.4 %** | **−50 %** |
    | 7 | `<T,7,4>` | 2 | NA=7, r=2 blocked | 134/1 clean | 1 | +8.5 % | −50 % |
    | 8 | `<T,8,4>` | 2 | NA=8, r=2 blocked | 151/1 clean | 1 | +10.3 % | −50 % |
    | 9 | `<T,9,5>` | 2 | NA=9, r=2 blocked | 168/1 clean | 1 | +11.7 % | −50 % |

    🟢 **The de-risking fact: the primary target (NA=6, r=2, 117 regs) sits
    BELOW the 125-register `<T,5,5>` cell that E27 already ships and already
    measured fast.** My register model was **falsified**: the product model
    `35.5 + 5.96·(r·NA)` has max residual 49 regs and does not even determine the
    verdict (NA=6/r=4 spills at 144 while NA=12/r=2 is clean at 196). The correct
    model is affine in NA at fixed r, `slope(r) = 8.36 + 3.19·r`, max residual
    **0.25** — `r=2: 16 + 15·NA`, `r=4: 20 + 21·NA` — which reproduces E27's
    62/83/104/125 ladder digit-for-digit and splits exactly into r-independent
    x-side registers (5 floats/NA) and per-row acc/partial registers (2
    floats/NA/row). Also carried: `<T,6,5>` (alphonse's E30 follow-up suggestion)
    is **illegal** (`6 % 5 == 1` trips `static_assert(M % IPG != 1)`) and would
    not help anyway (still 2 passes); and E27's `<T,8,3,true>` counter-example —
    19 % slower with *fewer* registers — means spill-freedom is a gate, never a
    predictor.

100. 🟢 **The Yukon submissions API exposes competitors' `note` but no diff**, so
    our `senpai/**` ledger and advisor briefs are **not** competitor-visible
    through the API; only the `note` field is a leak, and only the static
    reviewer sees `submission_diff`. Full single-submission key set:
    `benchmarkId, claimedScore, createdAt, id, improved, note, officialMetrics,
    officialScore, promotedSourceRef, promotionFinishedAt, promotionReason,
    promotionSnapshotRef, promotionStatus, rejectionReason, solverAccountId,
    solverAvatarUrl, solverProfileUrl, solverUsername, status,
    submissionCommitSha, updatedAt`. Residual exposure is `submissionCommitSha` /
    `promotionSnapshotRef` if the repo is reachable. Instrument trap that cost a
    call: the submission envelope is **camelCase** (`solverUsername`,
    `officialScore`, `officialMetrics`) while the nested `per_prompt` rows are
    **snake_case** (`effective_mean_draft_len`, `head_provenance_sha256`,
    `raw_ratio_of_means`) — a snake_case filter on the envelope silently returns
    zero rows rather than erroring.


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

## 2026-08-19: the first honest ranked row. We are rank 9, not 5.9 % behind.

Submission `ca9251b8-58cd-4d90-9a52-fa05f5657216` (commit `2b0c36a0`, created
2026-08-18 22:44, resolved 00:41) is **the first Senpai submission in this
campaign that ran the declared head artifact**. It scored
**3.23250848263467**, `rejected` with `score did not improve current best` —
the benign class (61.4 % of all rejections; every rejection at score >= 3.0 is
this one).

### 101 — We are **rank 9 of 408** scored rows, 0.517 % behind #1

| rank | id8 | solver | score |
|---:|---|---|---:|
| 1 | `0cd0a6b4` | ofou | 3.24929399 |
| 2 | `b0994092` | fkiene | 3.24417897 |
| 3 | `11863aa9` | companygardener | 3.24326224 |
| 4 | `4f76de6e` | alfranli123 | 3.24300059 |
| 5 | `de7981ae` | WillGasser | 3.24077781 |
| 6 | `3ec77796` | xadenryan | 3.23460814 |
| 7 | `942e5ab2` | Kamciosz | 3.23415183 |
| 8 | `e9f38898` | Lieisyourlie | 3.23409850 |
| **9** | **`ca9251b8`** | **morganmcg1 (us)** | **3.23250848** |
| 10 | `efe01dcf` | paul-hf | 3.23244884 |
| 11 | `3a995c2b` | SSHdotCodes | 3.23222999 |
| 12 | `070f1189` | scarletbright | 3.23167066 |

**All top 14 rows are on head `559b24eb`.** The head is the gate into the top
cluster and we are now through it. Ranks 6..12 span **0.09 %** — seven rows
inside a tenth of a percent.

🔴 **This retires the "5.862 % median gap" framing entirely.** That number was
measured on head `2f6805e1` (our LR3 trained head) and was almost all head
artifact. Corrected standing: **3.23251, gap to #1 = 0.517 %.**

Score identity re-confirmed exactly on both rows: sorted per-prompt
`raw_ratio_of_means`, score = mean(4th, 5th) = **beagle + medicine**.
Ours: mean(3.12015, 3.34486) = 3.2325085. Top: mean(3.14333, 3.35526) =
3.2492940. Note that `officialMetrics.mtp_decode_speedup_median` (3.1202) is
**not** the score — do not read it as one.

### 102 — Acceptance is now a **closed** lever, by exact measurement

`effective_mean_draft_len` is **bit-identical to the board top on 8/8 prompts**:
plutarch 0.1540, drama 2.2976, travel 2.6557, beagle 4.5327, medicine 4.7677,
republic 5.2697, essays 5.4253, botany 5.7765. Difference `+0.000 %` on every
prompt. `qwen_mtp_weights_hash` identical (`b53e4991…`), `parity_all_ok` true,
`mtp_depth` **8** on both rows.

Consequences:

- Our manifest declaration is **correct and working**. Do not touch
  `mtp-head.manifest.json`.
- **Every point of separation in the top cluster is per-row cost.** Within the
  88 rows on head `559b24eb`, **73 carry this exact acceptance fingerprint** and
  they span **2.325595 → 3.249294 (+39.7 %)**, median 3.196209. Identical
  proposal/acceptance behaviour, 39.7 % of score range. Cost is the whole game.
- 🔴 `mtp_depth = 8`, not 5. My earlier inference "n = 4.533 implies depth 5 at
  p ~ 0.965" was **wrong**: the field is in `officialMetrics` and says 8. At
  depth 8, n = 4.5327 implies uniform p ~ **0.871**. Any sizing that assumed
  depth 5 or p = 0.965 must be redone.

### 103 — 🔴 Our serial leg is faster than the top's on **8/8 prompts**, and that costs us score

| prompt | serial ours | serial top | delta |
|---|---:|---:|---:|
| plutarch | 0.0379678 | 0.0380471 | −0.2085 % |
| drama | 0.0379057 | 0.0380357 | −0.3417 % |
| travel | 0.0379038 | 0.0380867 | −0.4800 % |
| beagle | 0.0379849 | 0.0381076 | −0.3220 % |
| medicine | 0.0380265 | 0.0381029 | −0.2005 % |
| republic | 0.0379579 | 0.0380236 | −0.1728 % |
| essays | 0.0378929 | 0.0379771 | −0.2219 % |
| botany | 0.0379546 | 0.0381388 | −0.4828 % |
| **mean** | **0.0379493** | **0.0380649** | **−0.304 %** |

Population serial sd is 0.23 % per leg, so any single prompt is unremarkable —
but **8/8 with the same sign is p = 2^-8 = 0.4 %**. This is systematic, not
noise. It is also not a box-speed effect: a faster box would have moved *both*
legs, and our MTP leg moved the **other** way.

`R = serial / mtp`, so the serial leg is the **numerator**. Making the
non-drafting decode path faster **lowers our score**. Sizing: had our serial
matched the top's, beagle → 3.12015 × 1.00322 = 3.13020 and medicine →
3.34486 × 1.00201 = 3.35158, giving **3.24089 — rank 5**, from *undoing* an
optimisation.

**Doctrine (competitor-confirmed).** newjordan, accepted 3.1935: *"The narrow
[1024, 4096) branch never fires in the serial leg … the 2-bit branch fires only
when `bits == 2 && out_vec_size == 98336`. So the serial numerator and the
serial-denominator band are byte-for-byte unchanged."* Every shipped
optimisation must be **shape-gated off the M=1 path**.

🔴 **The ethical line, stated once.** Shape-gating or reverting *our own*
optimisation so it no longer fires on the serial leg is legitimate engineering —
we simply stop shipping a change that is counterproductive under the metric.
**Deliberately injecting a pessimisation into the serial leg is benchmark
gaming** and `Review submitted code for benchmark bypasses` is a live rejection
class with 18 hits. Never cross that line.

### 104 — Our candidate leg is the **worst of the top 12** on beagle

Per-row verify cost via the exact re-parameterisation
`h̄ = [(1 + 0.99·n)/R − 1]/n`:

| id8 | solver | score | h̄ beagle | h̄ medicine | mtp beagle |
|---|---|---:|---:|---:|---:|
| `0cd0a6b4` | ofou | 3.249294 | **0.16452** | 0.14783 | **0.0121233** |
| `b0994092` | fkiene | 3.244179 | 0.16480 | 0.14867 | 0.0121385 |
| `11863aa9` | companygardener | 3.243262 | 0.16498 | 0.14872 | 0.0121454 |
| `4f76de6e` | alfranli123 | 3.243001 | 0.16602 | 0.14787 | 0.0121258 |
| `3ec77796` | xadenryan | 3.234608 | 0.16725 | 0.14860 | 0.0121261 |
| `942e5ab2` | Kamciosz | 3.234152 | 0.16635 | 0.14948 | 0.0121587 |
| `e9f38898` | Lieisyourlie | 3.234099 | 0.16563 | 0.15012 | 0.0121339 |
| **`ca9251b8`** | **us** | **3.232508** | **0.16738 (worst)** | 0.14894 | **0.0121740 (worst)** |
| `efe01dcf` | paul-hf | 3.232449 | 0.16660 | 0.14963 | 0.0121346 |
| `3a995c2b` | SSHdotCodes | 3.232230 | 0.16648 | 0.14978 | 0.0121476 |
| `070f1189` | scarletbright | 3.231671 | 0.16626 | 0.15009 | 0.0121559 |
| `a4a69447` | rinaldofesta | 3.229695 | 0.16712 | 0.14977 | 0.0121619 |

Our beagle `mtp_seconds_per_token_mean` is the **highest (slowest) of the top
12**, and our medicine h̄ is mid-pack. So we hold rank 9 *despite* the worst
wide-width candidate performance in the cluster.

Per-prompt MTP-leg penalty vs the top, ordered by n:

| prompt | n | mtp delta |
|---|---:|---:|
| plutarch | 0.154 | +0.049 % |
| drama | 2.298 | −0.008 % |
| travel | 2.656 | −0.037 % |
| beagle | 4.533 | **+0.418 %** |
| medicine | 4.768 | +0.110 % |
| republic | 5.270 | **+0.426 %** |
| essays | 5.425 | **+0.506 %** |
| botany | 5.776 | +0.206 % |

🔴 **The penalty is zero on every narrow prompt and appears only where n > 4.5.**
Narrow rounds (M <= 4) are at parity with the top; the wide widths are where we
lose. This is the first **ranked-side** localisation of our deficit to the wide
crossrow dispatch, and it is independent confirmation of E33's target. It is
also a caution: E27 made M=5 and M=9 single-pass and measured −6.56 % E2E
locally, yet on the ranked box our wide widths are the cluster's worst. Either
E27's local gain did not transfer, or it did and we would otherwise be far
worse. **We cannot distinguish without a control row on this head**, and no
duplicate-commit pairs exist in the 88-row population to calibrate from.

Corrected sizing for E33, replacing "+8 % to +25 % of score" (which was derived
from the phantom 5.9 % gap and is **wrong by an order of magnitude**): a
uniform 5.4 % reduction in h̄ on the central pair — e.g. a 0.82 per-row cost
ratio on M=6 rounds carrying ~30 % of round mass — yields beagle 3.1947 and
medicine 3.4215, i.e. **3.30809, +1.81 % over the top**. On a board where
ranks 6..12 span 0.09 %, +1.81 % is enormous. Size against **0.517 %**, not
against 5.9 %.

### 105 — 🔴 RETRACTION: "our M=6/IPG/NA line is not scooped" was false

I asserted, and repeated in four assignment briefs, that *"no note among 627
mentions IPG, `rows_per_simd`, `ceil(M/IPG)`, NA, or widths 5/6/7."* A correct
scan of all 628 notes (every row has a note):

| term | notes | top scorer mentioning it |
|---|---:|---|
| `crossrow` | **175** | `11863aa9` companygardener 3.24326 (accepted) |
| `qmv_fast` | **147** | `11863aa9` 3.24326 |
| `IPG` | **96** | `e9f38898` Lieisyourlie 3.23410 |
| `warmAllDepths` | **78** | **`0cd0a6b4` ofou 3.24929 — the #1, plus ranks 2, 3, 4** |
| `weight stream` | 62 | `e9f38898` 3.23410 |
| `sdpaWidthWallDepthCap` | **48** | `efe01dcf` paul-hf 3.23245 |
| `values_per_thread` | 13 | `72ce82dc` scarletbright 3.22826 (accepted) |
| `rows_per_simd` | 3 | `e9f38898` 3.23410 |

The earlier grep was broken (it must have run against a truncated row set).
**Lesson: a zero-hit result on a corpus scan is a claim about my tooling, not
about the world, and must be validated on a known-positive before it is
believed — let alone repeated to four students as "nobody else is here."** This
is the same discipline I demand of students' negative controls and I failed to
apply it to myself.

What survives: the specific **row-blocked `4/r` coverage-preserving
reformulation** of the wide crossrow QMV is not described in any note. The
**axis** is crowded; the mechanism is still ours.

### 106 — `values_per_thread = 32` is already shipped at rank, on a path we do not care about

From `e9f38898` (Lieisyourlie, rank 8), quoting the promoted `#530` body for
`qmv_fast_singlerow_affine2_g64<T>`: *"`values_per_thread = 32` — one `uint64`
load (32 packed 2-bit values) per output row per k-block … `rows_per_simd = 4`
… this halves packed loads and scale/bias lookups versus the generic 16-value
form, and quarters the k-block count."* Dispatch gate: `bits == 2 &&
out_vec_size == 98336 && ntg.x == 1`.

And explicitly untouched by them: *"Every `bits == 4` dispatch cell, including
the live M=8 `4+4` (`qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>`)."*

So: the axis is proven not to spill on ranked M5 hardware, on the **2-bit
single-row draft readout** — a path our own E31 measured at 0.013 % of round
time. **E36's target (the wide 4-bit crossrow MLP QMV, 59 % of verify time)
remains untouched by anyone.** The gate `ntg.x == 1` is also a textbook
worked example of item 103's shape-gating doctrine.

### 107 — 🔴 The FP32 reassociation legality boundary, and a ranked-measured hazard

Lieisyourlie's own numerics section: *"The wider lane coverage reassociates the
FP32 partial-sum tree (32 products accumulate per lane-interior before the
k-block add, versus 16 in the generic per-byte-quad order). That perturbs
rounding at the last-ulp level. It is legal **for this stage** by the promoted
draft-rerank contract: the coarse shortlist is approximate by design, the exact
affine-4 rerank … decides the proposal, and the unchanged target verification
decides every emitted token."*

The converse is measured. companygardener, on the **target** side: *"Half-footprint
splits each affine-4/group-64 group across eight lanes instead of four …
the pairing and the within-lane K tile change, so last-ulp target logits move.
The public fixture can pass … while a hidden prompt flips a near-tie argmax."*
That scored 3.22444 and was rejected; their follow-up accepted at 3.24326 says
simply *"Stay off the verify reduction tree."*

**Rule: reassociation is licensed on the coarse/draft path and forbidden on the
verify path.** E33 row-blocks a kernel used by the verify leg, so it must
preserve the per-row K-reduction order exactly, not merely produce
"numerically equivalent" results. Row blocking changes which rows a thread
visits, not the within-row accumulation order — but that must be *demonstrated*,
because the local fixture cannot detect the failure.

### 108 — 🔴 `headStepCostRatio` 0.18 → 0.16 is a clean isolated ranked A/B: −1.164 %

`a1326b4b` (Lieisyourlie, 2026-08-18 05:23, **3.15370**), base `036fd9ca` at
**3.19088**. Their own description: *"This archive changes only
`private static let headStepCostRatio = 0.16` … Amal's `Qwen35.swift` rerank,
`mtp-head.manifest.json`, kernels, warmup, streak gate, and depth caps are
byte-identical."*

**One constant, everything else byte-identical, −1.164 % at rank.** Lowering `h`
makes the cost model price a draft step *cheaper*, so the session drafts more
deeply — and that is measured worse. This is the ranked-side falsification of the
"`h = 0.18` is mis-calibrated, lower it" line that E25 kept circling, and it
converges with the already-established "deeper is worse at the ranked operating
point."

Note the direction carefully: it does **not** close `sdpaWidthWallDepthCap`
5 → 4, which pushes capped rounds *shallower* (M=6 → M=5). That is the opposite
sign and remains open.

Their own dead-axis table, worth carrying wholesale:

| archive | score | mechanism |
|---|---:|---|
| `3ba27d91` | 3.21191 REJ | fusion restore on top-32 tip |
| `b0f127d5` | 3.20407 REJ | AndNormed warm |
| `eee31a36` | 3.18005 REJ | mid-width nibble on Amal head |
| `a1326b4b` | 3.15370 REJ | **h 0.18 → 0.16** |
| `a56eecd4` | 3.12613 REJ | mid-width on welt head |
| `9100a4e…` | 3.07827 REJ | cold prior 0.90·0.99^d |

Also confirmed by them: *"`RuntimeStartupMemoryPolicy.swift` — SSH's full-memory
wire and 512 MiB / 50-op command-buffer geometry stay"* — independent agreement
with our E31 command-buffer geometry finding.

### 109 — `NAX` in competitor notes is Apple's M5 neural-accelerator path, **not** our NA

19 notes mention `NAX`; the usages are *"NAX M=512 tile"*, *"Paired GQA
attention and NAX unroll-count experiment"*, *"NAX paths are not in this diff."*
This is the same NAX our own ledger already adjudicates at lines 258 and 310
(*"SURVIVES: the decode QMV path is non-NAX on both hosts"*). Lieisyourlie's
*"do not copy … #532 NAX"* therefore does **not** pre-kill E33's NA axis or
E36's `values_per_thread` axis. I checked this specifically because a
dead-axis list from a rank-8 competitor naming our mechanism would have been
grounds to cancel two live assignments.

### 110 — `warmAllDepths` self-calibration is in the top 4 rows and we have never touched it

78 notes mention `warmAllDepths`, including **ranks 1, 2, 3 and 4**. The
mechanism, from AndreasHad04 (`a0110f2a`): *"`warmAllDepths` already dispatches a
verify at every legal width … OUTSIDE every scored window. It now runs that
sequence three times: pass 0 compiles …, passes 1 and 2 are timed and the
per-width minimum is kept."*

🔴 **This is the structural answer to the campaign's single biggest epistemic
problem.** Every width/depth constant we have argued about — `h = 0.18`,
`sdpaWidthWallDepthCap`, `costModelDepth`, the `F`/`S` split — is a
hand-tuned constant fitted on an M4 Pro with the wrong head artifact, and item
102 shows the ranked operating point is not the local one. A cost model
calibrated *on the ranked box, inside the untimed warm*, replaces all of them
with measurements. It also explains why item 108's hand-tuned `h` lost: the
constant was wrong for that tree, and a self-calibrating model would not have
been.

Adjacent unexploited items from the same scan, recorded for sequencing and
**not** yet verified against source by me:

- **Prefill is charged inside `decode_seconds` on both legs** (paul-hf,
  `efe01dcf`, rank 10): ~0.53 s of a ~9.5 % share on the candidate leg vs
  ~2.8 % on serial. Our measured prefill is 0.00102–0.00103 s/tok on both rows,
  within 0.4 % of the top — an *asymmetric* lever we have never worked.
- **The `K >= 1024` SDPA family boundary** (zeeshan8281 `4287f727`; corroborated
  by fkiene `b0994092`, rank 2, accepted): 512 seed + 512 decode crosses 1024
  mid-window, and *"the JIT tax is candidate-leg-only and does not cancel in the
  paired ratio."* Existing warm seeds only 512.
- **`costModelDepth`'s absorbing barrier** (WillGasser `de7981ae`, rank 5):
  *"`positionAcceptEMA[0] <= 0.18` is an absorbing state. Once a prompt enters
  it, the session cannot draft for the remainder of the window and cannot
  observe the evidence that would release it."* Matches plutarch exactly
  (n = 0.154, 427 non-drafting rounds of 512) — but item 101's order statistics
  prove plutarch is worth **exactly zero** unless it clears 3.143, and its
  best-ever value across 405 runs is 2.221. The barrier only matters if it can
  latch on beagle or medicine.

### 111 — 🔴 SELF-CORRECTION to item 103: the 8/8 serial sign test was real but **irrelevant**, and my sizing of it was wrong

Item 103 claimed our serial leg is systematically faster (8/8 prompts, mean
−0.304 %) and sized "matching the top's serial" at +0.26 % of score, i.e.
rank 5. **The sign test survives; the sizing does not.** I tested my own
independence assumption before telling four students, and it failed.

**Variance decomposition of the serial leg over all 88 rows on head `559b24eb`:**

- grand mean 0.0379908 s/tok
- between-run sd of the per-run mean: **0.1207 %**
- within-run sd across the 8 prompts: **0.1766 %**
- i.i.d. legs would give a between-run sd of 0.1766/√8 = 0.0625 %; observed is
  ~2× that, so a genuine **run-level offset of σ ≈ 0.10 %** exists on top of
  per-leg noise.

Our run's offset from the grand mean is **−0.109 %, i.e. 1.05 σ — entirely
unremarkable.** We are the 20th fastest of 88 (23rd percentile). The outlier is
in the other direction: **ofou (#1) has the slowest serial leg of all 88 rows**
(0.0380649, +0.195 % ≈ +1.9 σ), and their official margin over rank 2 is
0.157 %.

**The decisive error.** Only beagle and medicine are scored. Our fast serial legs
were on **travel (−0.26 % vs grand), essays (−0.28 %), drama (−0.20 %)** —
all non-central prompts, worth exactly zero. On the central pair our serial was
at or *above* the grand mean (beagle 0.0379849 vs 0.0379827; medicine 0.0380265
vs 0.0379903). **We had no serial deficit where it counts.** I aggregated over
prompts whose weight is zero, which is the exact mistake I warn students about
in every brief.

`corr(serial_mean, mtp_mean) = +0.043` over the population and `+0.050` in the
top cluster ⇒ **no box-speed effect**; the two legs are independent, so serial
variation is pure noise with respect to candidate-leg engineering — **and it is
being scored.** `corr(serial_mean, score) = +0.695` within the 21-row top
cluster: a slower serial leg predicts a higher score, exactly as `R = serial/mtp`
demands.

**What stands from 103:** the shape-gating doctrine, and the rule that a
serial-leg speedup is a red flag rather than a bonus. **What is withdrawn:** the
claim that we carry a self-inflicted serial optimisation worth +0.26 %. There is
no such lever. Do not go looking for it.

### 112 — Normalising the serial leg removes ~60 % of the visible spread at the top of the board

Recomputing every row's per-prompt ratio against the population per-prompt
**grand-mean serial numerator**, holding each run's own measured MTP leg fixed,
then re-deriving the median-of-8:

| official | id8 | solver | official | serial-normalised | delta | norm rank |
|---:|---|---|---:|---:|---:|---:|
| 1 | `0cd0a6b4` | ofou | 3.249294 | **3.239188** | −0.311 % | **1** |
| 2 | `b0994092` | fkiene | 3.244179 | 3.237870 | −0.194 % | 3 |
| 3 | `11863aa9` | companygardener | 3.243262 | 3.235729 | −0.232 % | 7 |
| 4 | `4f76de6e` | alfranli123 | 3.243001 | 3.236262 | −0.208 % | 5 |
| 5 | `de7981ae` | WillGasser | 3.240778 | **3.239135** | −0.051 % | **2** |
| 6 | `3ec77796` | xadenryan | 3.234608 | 3.237514 | +0.090 % | 4 |
| 7 | `942e5ab2` | Kamciosz | 3.234152 | 3.230224 | −0.121 % | 14 |
| 8 | `e9f38898` | Lieisyourlie | 3.234099 | 3.233800 | −0.009 % | 9 |
| **9** | **`ca9251b8`** | **us** | **3.232508** | **3.230830** | **−0.052 %** | **11** |
| 10 | `efe01dcf` | paul-hf | 3.232449 | 3.236053 | +0.112 % | 6 |
| 11 | `3a995c2b` | SSHdotCodes | 3.232230 | 3.233897 | +0.052 % | 8 |
| 15 | `72ce82dc` | scarletbright | 3.228261 | 3.217915 | −0.320 % | 21 |

**Official top-10 span 0.521 % → serial-normalised top-10 span 0.211 %.** About
60 % of the ordering at the top of this board is serial-leg variation rather
than candidate-leg engineering. Movements are large: WillGasser 5→2,
paul-hf 10→6, xadenryan 6→4; companygardener 3→7, Kamciosz 7→14,
scarletbright 15→21.

**Our true engineering position: `norm` 3.230830, −0.258 % from the best**
(ofou and WillGasser effectively tied at 3.2392). Our *official* rank is 9 and
our *normalised* rank is 11 — we were mildly helped, not hurt, by serial noise.

Three consequences that should govern the endgame:

1. 🔴 **The real engineering gap is 0.258 %, not 0.517 %.** Half the apparent
   gap is a tailwind ofou received.
2. 🔴 **Do not re-roll an identical tree.** Expected value is ~0 and the
   per-submission sd on score is order 0.1–0.2 %, which cannot reliably close
   0.517 %. Spend slots on real mechanism.
3. 🔴 **When E33 lands, expect ±0.2 % of measurement noise on the ranked
   result.** A single submission that lands within 0.2 % of the top is a coin
   flip, so a mechanism must be worth clearly more than that before we read a
   ranked row as confirmation. This is the honest replacement for the
   competitor folklore figure of "±1.5–3 % ranked noise", which our own
   88-row population contradicts: per-leg sd is 0.18 % and run-level sd 0.10 %.

**Method note.** This normalisation is only licensed because acceptance is
bit-identical across this population (item 102) and the two legs are
uncorrelated (item 111). It is a re-weighting of measured quantities, not a
model fit. The per-prompt grand-mean serials are tight — 0.0379805 (drama) to
0.0380036 (travel), a 0.06 % spread over 88 runs — so the reference is stable.

### 113 — 🟢 VERIFIED OPENING: our shape warm stops at KV 512, and the SDPA dispatch switches kernel families at KV 1024

Two competitor claims from the notes scan (item 110) turned out to be checkable
against our own source in minutes. Both check out, and together they describe a
cost we are paying inside every scored window.

**The boundary is real.** `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:745-753`:

```cpp
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

`k.shape(2)` is the live KV length. At KV >= 1024 the dispatch leaves
`sdpa_vector` for **`sdpa_vector_2pass`, a different kernel** with its own
pipeline. The second clause (GQA at KV >= 4096) cannot fire on this track.

**Our warm stops below it.** `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:308`
seeds the throwaway warm cache with exactly `Array(repeating: 0, count: 512)`,
and the surrounding comment shows we already learned this lesson once at a
smaller scale:

> *"Warming the legal widths behind an 8-token prefix left the long-prefix
> variants to materialise inside a later scored round (the ranked prompt-5
> receipt showed a repeatable 0.368 s one-off stall). Seed the throwaway cache
> at the track's real 512-token prefix so every width below compiles in the same
> long-context dispatch family as decode."*

**But the track is 512 seed + `decodeTokens` 512.** The live KV length therefore
runs 512 -> 1024 and **crosses the boundary mid-window**, so the first
`sdpa_vector_2pass` dispatch of the run happens *inside* the scored window. We
fixed the 8 -> 512 instance of exactly this bug and stopped one token short of
the real boundary.

**Why it does not cancel in the paired ratio.** With per-token means
`serial ~ 0.0380` and `mtp ~ 0.0121`, a one-off cost `C` shared by both legs adds
`c = C/512` to each and gives `(0.0380 + c)/(0.0121 + c)`, which is
**monotonically decreasing in `c`**. A shared one-off stall therefore *depresses*
the ratio, and removing it from both legs raises the score. fkiene (rank 2,
accepted 3.24418) states the stronger asymmetric form: *"Serial depth-0 does not
assemble `[primary] + drafts`. The JIT tax is therefore candidate-leg-only and
does not cancel in the paired ratio."* Either way the sign is in our favour.

**Why this is actionable now.** `warmAllDepths` is declared at
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:283` — **an editable path**.
Its callers (`Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:196,283,309`
and the `MLXFastHarness` twin) are **not** editable, but we do not need them: the
implementation is ours. The warm is documented as running *"OUTSIDE every scored
window"*, so extending it is free of scored cost.

**The one open uncertainty, which must be settled before this is believed.** The
boundary is gated on `d.get_architecture().back()` being `'d'` or `'s'`. Our host
is an M4 Pro and the ranked box is an M5 Max; **I have not established the
architecture character on either.** If the ranked box reports neither, the branch
never fires and this opening is empty. That is one print statement, and it is the
first thing the assignment must do.

**Sequencing.** This needs a GPU timing slot, which thorfinn holds for E33, so it
queues behind rung 1. It is currently the strongest candidate for the next round,
ahead of the other two adjacent levers in item 110, because the mechanism is
verified in source rather than inferred from a note.

**ADDENDUM (same turn), two facts that de-risk it substantially.**

1. 🟢 **The fix does not depend on resolving the architecture character.** The
   branch keys on `k.shape(2)`, the live KV length, which we control directly.
   Extending the throwaway warm cache past 1024 and re-warming the widths covers
   **both** kernel families whatever the arch char is, and it is strictly
   additive inside a warm that is already documented as running outside every
   scored window. So the *safety* of the change is arch-independent; the arch
   char only decides whether it *helps*. That inverts the risk profile — the
   experiment can proceed before the uncertainty is settled, which is not how I
   framed it above.
2. 🟢 **The arch gate is env-overridable**, so the uncertainty is testable on the
   M4 Pro without an M5. `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:560-562`:

   ```cpp
     arch_ = env::metal_gpu_arch();
     if (arch_.empty()) {
       arch_ = std::string(device_->architecture()->name()->utf8String());
   ```

   and `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/utils.h:205-208`:

   ```cpp
   inline const std::string& metal_gpu_arch() {
     static std::string gpu_arch_ = get_var("MLX_METAL_GPU_ARCH", "");
     return gpu_arch_;
   }
   ```

   ⚠️ **Use it only as a mechanism probe, never as a performance measurement.**
   The same string feeds `get_architecture_gen()` and the NAX availability check
   (`device.cpp:924-926`, `can_use_nax &= gen >= (arch == 'p' ? 18 : 17)`), so
   forcing it changes kernel selection elsewhere. It can answer "does the family
   switch at KV 1024 and does the first dispatch cost anything"; it cannot
   produce a transferable speedup number. This is the same trap as item 111 in
   miniature: a knob that makes a branch observable is not a knob that makes a
   measurement valid.

   Neither the local M4 Pro nor the ranked M5 Max arch string is established —
   there is no Python MLX in this environment, so it requires a build. It stays
   the first step, but it no longer blocks the work.

### 114 — 🟢 `warmAllDepths` currently compiles but never *measures*, and that is the campaign's epistemic fix

Reading `warmAllDepthShapes` end to end: it seeds a 512-token throwaway cache,
walks the head draft step `maxDepth` times, and warms the committed-history K/V
paths. **It times nothing.** There is no per-width measurement anywhere in it.

AndreasHad04 (`a0110f2a`) describes the unexploited form: *"`warmAllDepths`
already dispatches a verify at every legal width … OUTSIDE every scored window.
It now runs that sequence three times: pass 0 compiles …, passes 1 and 2 are
timed and the per-width minimum is kept."*

🔴 **This is the structural answer to the campaign's single biggest epistemic
problem.** Every width/depth constant we have argued about — `headStepCostRatio
= 0.18`, `sdpaWidthWallDepthCap = 5`, `costModelDepth = 700`, and the `F`/`S`
split in `T(M) = F·M + ceil(M/IPG)·S` — is hand-fitted on an M4 Pro against the
wrong head artifact, while item 102 shows the ranked operating point is not the
local one. A cost model calibrated **on the ranked box, inside the untimed
warm**, replaces every one of them with a measurement taken on the machine that
scores us.

It also explains item 108 mechanistically: `h` 0.18 -> 0.16 lost 1.164 % because
a hand-picked constant was wrong for that tree. A self-calibrating model cannot
make that class of error, and it is the only depth-policy change on the board
that does not require us to guess a number. Note that `warmAllDepths` appears in
**78 notes including ranks 1, 2, 3 and 4** — the top of the board is already
working this surface, so treat the timing-instrumented form as contested rather
than free.

### 115 — 🟢 The local GPU is `applegpu_g16s`: the SDPA two-pass branch *does* fire on this box, and the in-tree comment mislabels it

Item 113 left one open uncertainty: the `sdpa_vector_2pass` branch is gated on
`char devc = d.get_architecture().back()` being `'d'` or `'s'`, and I had
established neither host's architecture string. I settled it by compiling a
twelve-line Metal probe (`/tmp/archprobe.m`, `clang -framework Metal`) rather
than waiting for a full build:

```
device.name         = Apple M4 Pro
architecture.name   = applegpu_g16s
MLX arch_gen        = 16
MLX devc (back)     = 's'
SDPA 2pass at KV>=1024 on this box? YES
```

The string format is decoded by `device.cpp:566-572`: the last three characters
are `<tens><ones><tier>`, so `applegpu_g16s` is generation 16, tier `'s'`. The
tier switch at `device.cpp:573-595` is the in-tree documentation of the tier
characters, and it also sets the command-buffer geometry:

| tier | in-tree comment | `max_ops_per_buffer_` | `max_mb_per_buffer_` |
|---|---|---:|---:|
| `'p'` | `// phone` | 20 | 40 |
| `'g'` | `// base, pro` | 40 | 40 |
| `'s'` | `// max` | 50 | 50 |
| `'d'` | `// ultra` | 50 | 50 |
| default | `// default to medium` | 40 | 40 |

Two consequences, one of which is a warning about reading source comments as
facts.

1. 🔴 **The comment is wrong for this generation.** Our box is an M4 **Pro** and
   it reports tier `'s'`, which the comment labels `// max`. So the mapping from
   Apple marketing tier to MLX tier character is *not* what the comment says, and
   nobody should infer the ranked M5 Max's character from it. What we can say is
   that both `'s'` and `'d'` take the two-pass branch, and that a Max-class part
   is overwhelmingly unlikely to report `'p'` or `'g'`, so the branch almost
   certainly fires on the ranked box too.
2. 🟢 **Our local box already has the ranked box's command-buffer geometry** — 50
   ops, 50 MB — because `'s'` and `'d'` share the same limits. That is a real
   piece of good news for transferability, and it retroactively strengthens E31:
   the command-buffer geometry we measured locally is the geometry the ranked box
   uses, not a smaller-tier variant. Note also that this is where the "50-op"
   figure in E31 comes from; the 512 MiB number in my earlier summary of E31 is a
   different limit and the two should not be quoted together.

The env override `MLX_METAL_GPU_ARCH` (item 113 addendum) turned out not to be
needed to answer the question. It remains available, and remains a mechanism
probe only, for the reasons given there.

### 116 — 🔴 SELF-FALSIFICATION of item 113: the warm's 512-token seed is correctly sized, and the KV-1024 opening is worth approximately nothing

Item 113 claimed the scored track is "512 seed + 512 decode, so the live KV
length crosses the 1024 boundary **mid-window**", and I queued it as the next
round's strongest candidate. The fixture says otherwise.
`fixtures/qwen3_8_27b_mtp_track.json`, `timed_prompt_pool_note`:

> Each entry IS a timed MTP reference-rows golden … generated on box 3 with
> `mtp-verify --emitted <plan> --generate 513` … from a DIFFERENT **512-token
> prose seed** (8 distinct domains), and each returned
> `reference_self_consistent=true` with **513 rows**.

So the seed is 512 tokens and the window decodes 512 tokens (`decodeTokens 512`).
The live KV length therefore runs **512 → 1024**, and `k.shape(2) >= 1024` first
becomes true in the **final round of the window**, not mid-window. The warm seeds
the throwaway cache at exactly 512 — which covers the window's *entire* range
bar the last step. My "we fixed the 8 → 512 instance of this bug and stopped one
token short" was rhetorically satisfying and quantitatively empty: 512 is the
right number, and the one token we stop short of is the one that does not matter.

There is also a **ranked-data refutation** that is independent of the source
reading. The only thing this mechanism can produce is a **fixed, per-window,
one-off cost** (a pipeline creation for the second kernel family). A constant
absolute offset across all eight windows is a one-parameter model, and it is
**disfavoured at p ≈ 0.017** (χ² = 17.1 on 7 dof) against our measured
per-prompt deficits, because drama and travel show zero or slightly negative
deficit where a fixed cost predicts +0.18 % and +0.20 %. The model that does fit
is a wide-prompt-only rate step (item 118, p ≈ 0.30).

**Item 113 is demoted from "next round's strongest candidate" to low priority.**
It is not proven dead — a fixed cost confined to some sessions is not excluded —
but it no longer justifies a student-week, and the mechanism cannot explain the
deficit we actually have. This is the second time in two turns that item 113 has
been corrected by evidence I went looking for after writing it, and the lesson is
the same one as item 105 and item 111: *a mechanism verified in source is not a
mechanism sized against data.* I verified the branch existed and then sized it
from a track geometry I had not read.

### 117 — 🔴🔴 The track fixture publishes the organizers' own noise measurement, and it confirms the score identity exactly

`fixtures/qwen3_8_27b_mtp_track.json` has been in the tree the whole campaign and
I had never read it end to end. It contains, published and operator-ratified,
several things the campaign has been reconstructing by inference.

**(a) The score identity, confirmed independently of my empirical derivation.**
`scoring_semantics`:

> `aggregation`: `median_of_per_prompt_raw_serial_relative_speedup`
> `per_prompt_score`: `raw_p = mean(serial depth-0 seconds/token over p's accepted pairs) / mean(candidate seconds/token over the same pairs)`
> `published_score`: `median(raw_p over all timed prompts)`
> `median_rule`: `even_n_mean_of_two_central_order_statistics`
> `median_rule_note`: 8 is EVEN, so the rule matters: the score is the mean of the two central order statistics, NOT the lower-median rule used by the per-pair `mtp_decode_speedup_median` diagnostic

That is exactly the identity I reverse-engineered from two ranked rows — mean of
the 4th and 5th sorted `raw_ratio_of_means` — and it independently confirms that
`officialMetrics.mtp_decode_speedup_median` is *not* the score. It also confirms
`R = serial / candidate`, i.e. the serial leg is the numerator (item 103/111).
`prompt_selection` confirms there is no per-run draw: all 8 are always timed.

**(b) 🔴 A real noise measurement, which retires my ±0.2 % guidance.** The
`calibration` block reports six thermally-gated sessions of an **unmodified**
tree across **both** ranked boxes, with session medians:

```
box 2:  0.995231  0.993918  0.994609
box 3:  0.993847  0.993622  0.993008
```

Total spread 0.0022 = **0.12 %**; standard deviation **0.078 %** of the mean.
This is run-to-run noise **on the published score statistic itself**, measured by
the organizers, across boxes. My ±0.2 % figure (itself derived from the 88-row
serial variance decomposition in item 111) was conservative by about 2.5×.

Per-prompt, `noop_decode_speedup_spread_pct` gives beagle 0.144, botany 0.331,
drama 0.219, essays 0.155, medicine 0.207, plutarch 0.437, republic 0.269,
travel 0.167 — but `noop_decode_speedup_note` states these are conservative
propagations `sqrt(CV_serial² + CV_mtp²)` that **over-state true paired spread by
roughly 1.4–1.9×** because the two legs share a thermal session and are
positively correlated, and it publishes the **exact pair-level ratios where they
were available: beagle 0.104 %, botany 0.281 %, drama 0.116 %.** Every ranked row
has `accepted_pair_count = 1` on all eight prompts, so the pair-level figure is
the correct noise scale for a single submission's per-prompt ratio.

Consequences that change how we should be operating:

- 🟢 **Our 0.258 % serial-normalised gap to #1 is ≈ 3.3 σ — unambiguously real.**
- 🟢 **The detection threshold for a single ranked run is ≈ 0.16 % (2 σ), not
  0.5 %.** A clean 0.2 % is a measurable result. This materially lowers the bar
  every current assignment has to clear, and I have told all four students the
  wrong (harsher) number.
- 🔴 **The ranks 6–12 band spans 0.09 %, which is barely one σ.** Our nominal
  rank inside that band is noise. "Rank 9" and "rank 6" are the same result.
- 🔴 The competitor folklore "±1.5–3 % ranked noise" is wrong by an order of
  magnitude, and now demonstrably so from the organizers' own data.

**(c) 🟢 The organizers confirm the serial leg is prompt-independent.** From
`noop_decode_speedup_note`, on an outlier: *"the depth-0 leg does
prompt-independent work, so all serial readings are interchangeable and sit at
~0.03800."* This is direct validation of the serial-normalising method in item
112, and it improves the estimator: a run's serial offset should be estimated by
**pooling all eight of its serial readings**, not per-prompt. The pinned on-box
serial calibration is 0.037994794617407023, against our 88-row grand mean of
0.0379908 — agreement to 1 part in 10⁴.

**(d) 🟡 The central pair is tree-dependent.** The calibration block's per-prompt
raw ratios for a stock tree are beagle 0.9837, botany 0.8467, drama 0.9587,
essays 1.0044, medicine 1.0726, plutarch 0.9701, republic 1.0116, travel 1.0581;
the 4th and 5th sorted are **beagle and essays**, mean 0.99404 — reproducing
`expected_raw_median` 0.9940390645 exactly. On *our* tree the central pair is
beagle and medicine. So "only beagle and medicine carry weight" is a statement
about our current operating point, not a property of the track, and it can
rotate under a large enough change.

### 118 — 🟢 Our deficit against #1 is a clean step at the draft-width boundary: parity below n = 3, +0.38 ± 0.06 % above n = 4.5

Testing our per-prompt MTP-leg deficits against the organizers' pair-level noise
(item 117b) rather than eyeballing them changes the picture in a useful way.
Per-prompt deficit, absolute cost over the 512-token window, and significance
against the true paired σ:

| prompt | n | Δ% | Δ ms | σ% | σ | significant |
|---|---:|---:|---:|---:|---:|---|
| plutarch | 0.154 | +0.049 | +7.6 | 0.265 | +0.18 | no |
| drama | 2.298 | −0.008 | −0.8 | 0.116 | −0.07 | no |
| travel | 2.656 | −0.037 | −3.3 | 0.101 | −0.36 | no |
| **beagle** | 4.533 | **+0.418** | +26.0 | 0.104 | **+4.02** | **YES** |
| medicine | 4.768 | +0.110 | +6.4 | 0.125 | +0.87 | no |
| **republic** | 5.270 | **+0.426** | +24.3 | 0.163 | **+2.61** | **YES** |
| **essays** | 5.425 | **+0.506** | +29.0 | 0.094 | **+5.38** | **YES** |
| botany | 5.776 | +0.206 | +11.7 | 0.281 | +0.73 | no |

Model comparison, inverse-variance weighted:

| model | params | χ² / dof | p |
|---|---:|---|---|
| constant rate across all 8 | 1 | 25.9 / 7 | 0.0005 — rejected |
| constant absolute offset across all 8 | 1 | 17.1 / 7 | 0.017 — disfavoured |
| **two-level step at the width boundary** | **2** | **7.2 / 6** | **0.30 — fits** |

The step model reads: prompts with n < 3 are at **−0.019 ± 0.073 %** — parity
with the leader, indistinguishable from zero — and prompts with n > 4.5 are at
**+0.380 ± 0.056 %**, which is 6.8 σ from zero. One parameter, one boundary, and
it lands exactly where the wide crossrow dispatch changes behaviour.

Caveat I cannot remove with this fixture: **inside** the wide group, all five
windows have nearly the same wall time (5.67–6.23 s), so a per-token rate penalty
and a fixed ~22 ms per-window cost are numerically degenerate there (χ² 7.11 vs
7.20). What *is* excluded is a fixed cost applying to **all** prompts, because
the three narrow windows have very different durations (8.9–15.5 s) and show no
deficit. Separating rate from fixed cost within the wide group would need a
high-n prompt at a different window length, which the track does not provide.

Medicine is the persistent outlier in every model at about −2.2 σ: we are
essentially tied with the leader on medicine. That matters because medicine is
one of the two scoring prompts.

**Order-statistic saturation caps**, computed on our own sorted ratios:

```
1 plutarch 1.25280   2 drama 1.91668   3 travel 2.17980
4 beagle   3.12015  <- central        5 medicine 3.34486  <- central
6 essays   3.36612  <- caps the 5th   7 republic 3.39402   8 botany 3.42536
```

- **beagle has +7.88 % of headroom** before it would pass essays and stop paying
  at the margin. No saturation risk; the wide-width line has room.
- **medicine has only +0.64 % of headroom.** Beyond that, further medicine gains
  are worth zero and essays becomes the binding prompt. Nobody should be
  optimising medicine specifically.

Putting items 117 and 118 together gives the campaign's operating target in one
line: **the statistically real, addressable deficit is beagle's +0.418 %, worth
about +0.21 % of score at the 0.5 marginal weight of a central order statistic —
which is essentially the whole 0.258 % serial-normalised gap.** The wide-width
kernel line (E33) is aimed at exactly this, and is confirmed as the main line.

### 119 — 🟢 Acceptance closure is now established on two independent fields, and the realised-depth question has a local counter rather than a source read

**(a) A second identical fingerprint.** Item 102 established that
`effective_mean_draft_len` is bit-identical to the board leader on 8/8 prompts.
`non_drafting_round_count` is *also* identical on 8/8: zero on seven prompts and
**449 on plutarch** for both rows. Sixteen matching per-prompt quantities across
two independently developed trees. We and the leader run the same draft schedule
round for round; the entire difference between rank 1 and rank 10 is time per
round. Acceptance is closed, and now on two fields rather than one.

Incidentally this explains plutarch's degeneracy: ~449 of its ~504 rounds draft
nothing at all, which is why its ratio is 1.2528 and why it is worth exactly zero
to the score.

**(b) 🔴 "The candidate declares no depth."** `scoring_semantics`
→ `effective_depth_provenance`:

> The candidate declares no depth. `effective_mean_draft_len`,
> `effective_max_draft_len`, `non_drafting_round_count` and
> `effective_draft_lengths` are derived by the trusted parent from its own round
> journal and sealed per prompt and per run.

This settles the item 102 / 113 open question in the direction I flagged as
possible: `officialMetrics.mtp_depth: 8` and `mtp_max_draft_depth: 8` are
**configuration echoes, not realised depth**. So *both* of my competing
inferences — "depth 5 at p ≈ 0.965" and "depth 8 at p ≈ 0.871" — remain
unsupported, and no ranked field will settle it: `effective_max_draft_len` and
`effective_draft_lengths` are sealed per run but are **not** in the public API
per-prompt payload (which exposes only `accepted_pair_count`,
`effective_mean_draft_len`, `head_provenance_sha256`,
`mtp_seconds_per_token_mean`, `non_drafting_round_count`,
`noop_reference_decode_speedup`, `parity_ok`, `prefill_seconds_per_token`,
`prompt_sha256`, `raw_ratio_of_means`, `serial_seconds_per_token_mean`).

**(c) 🟢 But our own harness emits it.** `effective_max_draft_len` appears in
local timed artifacts as a per-prompt, per-leg counter — e.g. in
`research/e25r2-timed.json` at
`per_prompt/<name>/{base,candidate}/counters/effective_max_draft_len`, which in
that (Qwen 3.6-era) run read base 4–5 and candidate 3. So the realised-depth
question is answerable by **reading a counter from one local timed run on the
current tree**, not by source archaeology and not by inference from n. That is a
much cheaper and much more reliable route than the one I gave edward, and it is
the single most useful correction I can hand him.

### 120 — 🟢 E36 TERMINAL (merged, PR #41): `values_per_thread` is closed on the verify path by ranked parity failures, not by registers — and my register model is falsified

askeladd returned a complete terminal answer in one revision, zero GPU, zero
shipped bytes (11 files, all `research/`, growth 0/262144). Merged at
`1e3dc494`. Four results, in descending order of what they cost me.

**(a) 🔴 The axis is closed, and by somebody else's ranked failures.** The
re-ordered first question — *can `values_per_thread` be raised on the wide
crossrow QMV without altering the per-row reduction order?* — is answered **NO,
and not fixably: `values_per_thread` IS the lane→K partition
(`quantized.h:1020`).** No setting above or below 16 preserves the order. Then,
instructed to read the 13 `values_per_thread` notes (item 105's corrected scan),
he found the confirming evidence:

- companygardener `5c74b78b` — wide affine4/g64 QMV footprint cut for "width-3
  and width-4 target verification", vpt 16→8, block 512→256 — **FAILED the
  official untimed Qwen-MTP correctness-and-parity gate.**
- `6154a6f1` — same change at verify width 3 — **FAILED the same gate.**
- Their next row `11863aa9` (3.24326, accepted and promoted) records the rule:
  *"Target-side footprint cuts fail official MTP parity. Stay off the verify
  reduction tree."*

🔴 **The observation that makes this decisive is his, not mine: `5c74b78b` moved
vpt DOWN to 8 — the register-cheapest cell in the entire 262-cell grid (117 regs
at M=6/NA=6/r=2, 8 private bytes).** If this axis were a register trade, v=8 was
a free win. It failed parity twice anyway. **Registers were never the binding
constraint; the lane→K partition always was.** That also closes the v=8 column,
which local compilation could never have done, at zero M5 transfer risk.

This is the third independent confirmation of the item 107 reassociation
boundary, and the first that arrives as a *pair of ranked gate failures* rather
than a note's assertion.

**(b) 🔴 My register model is falsified by measurement, with a mechanism.** I
predicted `slope ≈ 8.36·(vpt/16) + 3.19·r`, i.e. 155 registers at
NA=6/r=2/vpt=32 (and I misquoted "~196" in the brief, from a different cell).
**Measured 106 — bit-identical to vpt=16.** Over 262 cells (2 arms × r∈{1,2,3,4}
× vpt∈{8,16,32,64} × NA 2..12), registers are **unchanged across vpt 8→64 for
every spill-free cell at NA≥4, r≤2**; the r-slopes 11/15/18/21 hold at every vpt
with zero spread at r=2.

My error was **structural, not calibration**: I assumed the kernel materialises
`x_thread`, so the x-side term should scale with vpt. It does not — crossrow
re-reads four activations into `a0..a3` (`quantized.h:1014-1035`). vpt's real
cost is **`r·vpt/2` private bytes in all 262 cells, and zero registers.**

Superseding fit, replacing mine everywhere: **`slope(r) = 8.00 + 3.30·r`, max
residual 0.40.** Validity gates passed before any new number was shown: E27
ladder reproduced digit-for-digit (62/83/104/125), known-BAD NA=6/r=4 gave
144+spill, 14/14 controls matched pre-declared verdicts.

**(c) 🟢 The composition verdict: the axes DO compose in registers, and it does
not matter.** `M=6/NA=6/r=2` is **117 registers at every one of vpt
8/16/32/64**; `M=9/NA=9/r=2` is 168 at all four. So E33's rung-1 headroom claim
(117 vs the 125 the shipped `<T,5,5>` already uses) is confirmed at four
independent vpt settings rather than one. thorfinn gets NA, not vpt — **but the
reason is the parity wall in (a), not contention.**

**(d) 🟢 My k-block arithmetic was right, and was then strengthened.** The vpt-64
wall reproduces exactly (`5120 % 2048 = 1024`), and additionally hits **K=17408
(down_proj)**; the ragged-block escape I invited him to find **does not exist** —
no K tail, no bounds check, and `_m` is a tail over M, not K. Six shapes overrun
at vpt 64, five of them on the wide crossrow band (his own post-submission
correction of "five scored projections", against his own published number, when
the verdict did not depend on it).

**(e) 🟢 Why the kernel chose 16, found rather than invented.** Generic
`qmv_fast_impl` materialises `x_thread[vpt]` at `:774`, sized by
`packs_per_thread = bits==2 ? 1 : 2` at `:761` — real upstream code, never
re-derived for crossrow, inherited by copy. Plus `512 = 16 × SIMD_SIZE`. Plus the
one that matters most: **16 is the pinned serial build's width, hence the only
parity-safe width on the verify path.**

**Reporting discipline worth recording as the campaign's standard.** He reported
`vpt_attributable_delta = 0` when the primary metric moved 5→9 and the +4 could
have been banked — it belongs to E32's row blocking. He withdrew his own
follow-up (order-preserving unroll-by-2) **before** anyone spent on it, because
xadenryan had already built it (`37911ea4`, `ace21f9d` failed policy review,
`d95b11d5` validating) and aimed it at M==1 — the wrong side of `R = serial/mtp`.

### 121 — 🟢 `get_qmv_batch_limit` resolves to 10 on the ranked box: E33's dispatch is live, and the Apple generation ladder is now anchored

askeladd escalated the one open risk that could have invalidated E33 wholesale,
and it is now closed. `get_qmv_batch_limit`
(`Vendor/.../backend/metal/quantized.cpp:84-126`) is consumed at `:1415/:1483`
and gated at `:1417` as `if (M >= vector_limit) { …qmm… }`, so **`qmv` runs only
when `M < vector_limit`.** For our shapes (D, O > 4096):

| `arch_gen` | tier | `vector_limit` | M=6..9 in `qmv`? |
|---|---|---:|---|
| **13 or 14** | non-`'d'` | **6** | 🔴 **no — all fall to qmm** |
| 13 or 14 | `'d'` | 12 | yes |
| anything else | non-`'d'` | **10** | 🟢 yes |
| anything else | `'d'` | 12 | 🟢 yes |

Had the ranked M5 parsed to arch_gen 13/14 non-`'d'`, **every one of M=6..9 would
have left `qmv` entirely** and the whole crossrow NA line — E32, E33, E36 — would
be tuning a dispatch the ranked box never executes. Correct escalation, and
correctly scoped: the M5 arch string genuinely is not in the repo.

**Resolved to 10, by three independent lines:**

1. 🔴 **An accepted ranked row states it outright.** `088f763b` (accepted,
   1.25132): *"At decode M=1 those three GEMVs all miss the NAX qmm path (**qmv
   batch limit is 10**; see research)."*
2. 🔴 **NAX liveness on the ranked box forces `gen ≥ 17`.** `52cdbf5e`: *"The M4
   Max does not support NAX/MPP tensor ops; **the ranked M5 box does**."*
   `device.cpp:924-926` gates `can_use_nax &= gen >= (arch == 'p' ? 18 : 17)`, so
   a live NAX path requires gen ≥ 17 and the 13/14 branch cannot fire. I checked
   this specifically because a *dead* NAX axis would have been consistent with
   gen < 17 and would NOT have closed the risk — the notes assert the path is
   live on the ranked box, not merely compiled.
3. 🟢 **The generation ladder is anchored by direct measurement.** Our probe
   (item 115) gives this box `applegpu_g16s`, **gen 16**. Competitor notes agree
   independently: `d9cbec0f` *"The available development machines are M3 Max and
   M4. They do not cross MLX's M5-generation NAX availability threshold"*, and
   *"The M4 correctly kept the NAX guard disengaged."* So M4 = 16, M5 = 17, and
   arch_gen 13/14 is M1/M2-era silicon.

Belt and braces: **every branch except gen∈{13,14}-non-`'d'` returns ≥ 10**, and
`'d'` returns 12 even in the 13/14 branch, so M=6..9 stay in `qmv` under every
configuration the ranked box could plausibly report.

🟢 **thorfinn is unblocked; E33 rung 1 needs no change.** I have asked him to add
a one-line assertion that the cell he times is actually reached — the cheapest
possible guard against this class of error now that the gate's location is known.

**The transferable lesson is the inverse of item 105.** askeladd scoped his
search to the repository, which was the right instinct, and the answer lived in
the competitor corpus he had already mined for `values_per_thread`. Item 105's
lesson was that a zero-hit corpus scan is a claim about tooling; this is its
complement — **the corpus is a source of positive facts about the ranked
environment, not only of prior-art kills.** Two of the three lines above are
competitor notes describing hardware we cannot touch.

### 122 — 🔴 Prefill is reported but NOT scored: every prefill optimisation is worth exactly zero, and item 110's "adjacent lever" was never a lever

Item 110 listed "prefill inside `decode_seconds`" as an adjacent unexploited
lever, on the strength of a paul-hf note putting prefill at ~9.5 % of the
candidate leg against ~2.8 % of the serial leg. A competitor built an entire
submission on the stronger form of the claim — `d9cbec0f`: *"targets a different
large budget: **the charged 512-token seed prefill** on the ranked M5."*

It is not charged. Tested on 32 cells — 8 prompts × 4 independent rows (ours plus
ranks 1, 2, 3) — against two hypotheses:

| hypothesis | worst relative error over 32 cells |
|---|---|
| `raw = serial_s_per_tok / mtp_s_per_tok` (prefill **excluded**) | **3.9 × 10⁻¹¹** |
| `raw = (pf + serial) / (pf + mtp)` (prefill **included**) | 5–7 % |

The first is exact to floating-point round-trip precision on every cell; the
second is not close. **`raw_ratio_of_means` is exactly the ratio of the two
per-token decode means, and `prefill_seconds_per_token` is sealed for information
only.** Tool: `research/prefill_charged.py`.

Consequences:

- 🔴 **Prefill optimisation is worth exactly 0.000 % of score.** Removed from
  item 110's adjacent-lever list permanently. Our prefill is already at parity
  with the leader (0.001027 vs 0.001031, within 0.4 %), so we were never going to
  gain there anyway, but the point is stronger than that: we could halve prefill
  and score identically.
- 🟡 **paul-hf's arithmetic is right and their conclusion is irrelevant.** ~9 % of
  the candidate leg's *wall time* is prefill — that follows mechanically from
  prefill ≈ 0.53 s against a candidate decode of ≈ 6.2 s and a serial decode of
  ≈ 19.4 s, and the apparent 9.5 %-vs-2.8 % asymmetry is nothing but the 3.2×
  decode ratio re-expressed. It is a true statement about wall time and a false
  statement about score.
- 🟢 `d9cbec0f` has status `failed`, so its premise was never tested at rank. Note
  the pattern: a competitor spent a submission on a lever that ten lines of
  arithmetic against their own published telemetry would have closed. That is the
  same failure mode as my item 113, and the same cure — **check what the score
  actually integrates before optimising anything.**

This also tightens item 117(a): the score is `median(serial_p/mtp_p)` over the
eight prompts under the even-n mean-of-two-central-order-statistics rule, with
*no* other term. There is no prefill component, no warm component, and no
prompt-count weighting. Everything outside the 512-token decode window is free,
and everything inside it is priced at the per-token mean.

---

## 123 — 🔴🔴 Every top-12 rival has **bit-identical acceptance** to ours. The entire competitive spread is wall-clock speed at fixed work.

The submissions *list* endpoint carries `officialMetrics` in full for all 414
scored rows — I had been fetching rows one at a time for two turns without
noticing. `research/rival_profile.py` now reads the whole board offline.

For the top 12 rows I compared `effective_mean_draft_len` and
`non_drafting_round_count` against ours, per prompt:

| row | score | `n` match | `non_drafting` match |
|---|---|---|---|
| 0cd0a6b4 (#1) | 3.24929399 | **8/8** | **8/8** |
| b0994092 | 3.24417897 | **8/8** | **8/8** |
| 3ac231d5 | 3.24387902 | **8/8** | **8/8** |
| 11863aa9 | 3.24326224 | **8/8** | **8/8** |
| 4f76de6e | 3.24300059 | **8/8** | **8/8** |
| de7981ae | 3.24077781 | 7/8 | 7/8 |
| b089fdd9 | 3.23871762 | **8/8** | **8/8** |
| 3ec77796 | 3.23460814 | **8/8** | **8/8** |

Not "similar" — *equal*, to all printed digits, on both fields, on eight prompts,
across seven independent solvers. `de7981ae` is the single exception and it
differs on exactly one prompt (item 125).

This settles a question I had been treating as open since item 119:

- 🔴 **Nobody at the top of this board is winning through acceptance, depth,
  width, or any other policy quantity.** The draft schedule is effectively frozen
  across the whole competitive cluster. Everyone dispatches the same rounds, at
  the same widths, accepting the same tokens.
- 🟢 **Therefore 100 % of the 0.52 % gap to #1 is wall-clock time to execute an
  identical instruction stream.** That is a kernel/dispatch problem, full stop.
- 🔴 This is a **strong endorsement of E33** (thorfinn, kernel internals) and a
  **strong disconfirmation of the premise of E34** (edward, depth-policy
  operating point). See item 125 for the direct experimental null on E34's class.

`qwen_mtp_weights_hash` is one single value across all 414 rows
(`b53e4991737cdf50827e518e7559628874d3ff6d5f63bebc057ddbb16a89e2cd`), which
independently confirms the head artifact is pinned for everyone and closes any
remaining "maybe they shipped a better head" hypothesis.

**Process note.** I reverse-engineered the score identity, the noise floor and
the acceptance fields over several turns from per-row fetches, and the entire
board was sitting in one list response the whole time. This is the same lesson as
item 117 (read the fixture that is already in your tree) one level out: **before
modelling a population, check whether the bulk endpoint already returns the
population.**

---

## 124 — 🔴🔴 The deficit is a reproducible per-prompt *fingerprint*, and it is **not** monotone in draft length. beagle and medicine are the puzzle.

Candidate `mtp_seconds_per_token_mean` advantage over us, per prompt, for the ten
rows above us (positive = rival faster). Acceptance is identical everywhere
(item 123), so these are pure speed differences on identical work:

| row | beagle | medicine | essays | republic | botany | drama | travel |
|---|---|---|---|---|---|---|---|
| 0cd0a6b4 | +0.418 | +0.110 | +0.506 | +0.426 | +0.206 | −0.008 | −0.037 |
| b0994092 | +0.293 | +0.148 | +0.510 | +0.436 | +0.180 | −0.010 | −0.126 |
| 3ac231d5 | +0.328 | +0.062 | +0.526 | +0.435 | +0.231 | +0.074 | −0.037 |
| 11863aa9 | +0.236 | +0.073 | +0.481 | +0.524 | +0.238 | +0.016 | −0.053 |
| 4f76de6e | +0.398 | −0.046 | +0.507 | +0.372 | +0.256 | +0.008 | −0.146 |
| de7981ae | +0.422 | +0.103 | +0.526 | +0.491 | +0.253 | +0.087 | +0.056 |
| b089fdd9 | +0.218 | +0.140 | +0.546 | +0.447 | +0.236 | +0.054 | −0.058 |
| 3ec77796 | +0.395 | +0.031 | +0.445 | +0.407 | +0.205 | −0.068 | −0.067 |
| e9f38898 | +0.330 | −0.131 | +0.470 | +0.451 | +0.248 | −0.052 | −0.014 |
| efe01dcf | +0.325 | +0.010 | +0.426 | +0.390 | +0.141 | −0.029 | −0.089 |

Read the columns, not the rows. Ten independent solvers agree to within ~0.1 %:

- **Deficit group** — essays ≈ +0.49 %, republic ≈ +0.44 %, beagle ≈ +0.34 %,
  botany ≈ +0.22 %. Every single rival beats us on all four.
- **Parity group** — medicine ≈ +0.05 %, drama ≈ +0.01 %, travel ≈ **−0.06 %**
  (we are *faster* than most rivals on travel). Signs are mixed, i.e. noise.

This is item 118's two-level step re-derived from ten rivals instead of one, and
it is far tighter than my χ² fit deserved to be. But it **breaks the story I
attached to it.** Mean draft lengths are:

```
deficit group : essays 5.425  republic 5.270  botany 5.776  beagle 4.533
parity group  : medicine 4.768  travel 2.656  drama 2.298
```

🔴 **beagle (n = 4.533) is in the deficit group; medicine (n = 4.768) is in the
parity group.** A monotone-in-`n` rule would order them the other way. Item 118's
"n < 3 parity / n > 4.5 deficit" split survives for six prompts and fails on
exactly the pair that carries the entire score — beagle and medicine are the 4th
and 5th order statistics, i.e. the only two prompts with non-zero weight.

So the single most valuable unknown in the campaign is now sharply posed:

> **Why are we at parity on medicine (n = 4.77) but 0.34 % behind on beagle
> (n = 4.53), when the two prompts are adjacent in every summary statistic the
> board exposes?**

A candidate mechanism worth stating because it is checkable without a GPU. The
dispatched kernel width is `M = offered_depth + 1` (the verify batch is the
primary row plus the drafted rows). Our own E27 M-table is *very* non-uniform:

```
M   1      2      3      4      5      6      7      8      9
    0.9292 0.9860 1.0020 0.9995 0.7990 1.0032 0.9995 1.0051 0.8854
```

We hold a large win at **M = 5** (0.799) and **M = 9** (0.885) and are flat
everywhere else. If medicine's rounds concentrate on a width where E27 already
won and beagle's concentrate on a width where E27 is flat, the parity/deficit
split follows immediately — and the prescription is to extend the M=5-class win
to the width beagle actually uses. Mean acceptance cannot distinguish the two
because it is a first moment; the *width histogram* can.

I cannot settle this from board data: `effective_max_draft_len` and
`effective_draft_lengths` are computed by the trusted parent and **not** exposed
(item 119). It needs the analytic dispatch census — assigned as E37.

🟢 Independent of the mechanism, two numbers are now firm and should drive
targeting: **beagle carries 0.5 marginal weight with +7.88 % of headroom before
it saturates, and medicine carries 0.5 marginal weight with only +0.64 %.**
Beagle is the only prompt on the board where we have both leverage and room.

---

## 125 — 🔴🔴 A rival found a **real bug in our shared baseline**, fixed it, improved one prompt by **+73.61 %**, and gained **+0.000000 %** of score. Exactly zero.

`de7981ae` (WillGasser, Claude Opus 5) is the one top-12 row whose acceptance
differs from ours (7/8, item 123). Its published note states the mechanism, and
the mechanism is correct:

> `costModelDepth` returns 0 whenever `positionAcceptEMA[0] <= headStepCostRatio`
> (0.18). A round that drafts nothing returns before `recordAcceptOutcome` is
> reached, and that function is the only place the EMA is ever updated. The
> estimate is therefore frozen at the value that caused the skip.
> `positionAcceptEMA[0] <= 0.18` is an absorbing state.

**Verified in our own source**, `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`:

- `costModelDepth` at `:700`. The walk enters with `depth = 0`, `reach = 1.0`,
  `expected = 0.0`, so `threshold = h·(1+0)/(1+0) = h = 0.18` exactly (`:630`,
  `headStepCostRatio = 0.18`). `reach = p = positionAcceptEMA[0]`, then
  `guard reach > threshold else break` ⇒ **`p ≤ 0.18` returns depth 0**.
- `positionAcceptEMA` is written **only** in `recordAcceptOutcome`
  (`:775-801`), which is reached only at `:1219`, inside the verify path.
- A depth-0 round never verifies, so it never updates the EMA that produced it.
  The state is genuinely absorbing.

**Confirmed in the data on both sides.** Our plutarch:
`non_drafting_round_count = 449` of ~504 rounds, `effective_mean_draft_len = 0.154`.
Theirs: `6` and `2.541`. They broke the trap. All eight top rivals — and we —
carry the identical 449, so the trap is deterministic and universal.

**And the score value of the fix, computed exactly** (`research/plutarch_zero.py`):

```
their plutarch raw_p        2.175010      (ours 1.252802,  +73.61 %)
score WITH their fix        3.24077781
score WITHOUT their fix     3.24077781
score value of the fix      +0.000000 %
```

Because plutarch is the **1st order statistic** before the fix and the **2nd**
after it (their sorted list: drama 1.9194, plutarch 2.1750, travel 2.1847,
beagle 3.1356, medicine 3.3460, …). It never reaches the central pair. A 73.61 %
improvement on a real bug, worth nothing, to six decimal places.

Consequences:

- 🔴 **E34's class is now experimentally null, not merely unsupported.** Item 123
  showed nobody at the top changed the draft policy; this shows what happens to
  the one solver who did. The absorbing-state fix is the largest single-prompt
  improvement anywhere on this board and it moved the score by zero.
- 🟢 **The order-statistic weighting is now empirically confirmed from a rival's
  row**, not just from my own sensitivity algebra. Item 111's lesson (never
  aggregate over zero-weight prompts) has an external, quantified witness.
- 🟡 **The bug is still a latent cliff on the two prompts that matter.** The trap
  fires when `positionAcceptEMA[0]` falls to 0.18. Beagle sits at n = 4.53 and
  medicine at 4.77, so both are far from it today and neither is at risk. But any
  change that perturbs the depth-0 acceptance estimate on beagle or medicine
  would drop that prompt from ~3.1–3.3 to ~1.25 and cost roughly **29 % of
  score**. E33 is required to be bit-exact and therefore cannot trip it; **E34
  edits this exact policy and can.** Edward has been warned explicitly.
- 🟢 Cheap insurance, correctly priced: fixing the absorbing state is worth
  0.000 % of score today and removes a 29 % downside that only a policy edit can
  trigger. It is worth doing *if and only if* we ship a policy edit.

**The lesson I want on the record.** A competent rival, with a correct diagnosis
of a real defect, published in a coherent note, produced a change with exactly
zero effect on the objective — because they optimised the mechanism they had
found rather than the quantity the score integrates. That is my item 113 and
`d9cbec0f`'s prefill submission a third time. **Before fixing anything, compute
the counterfactual score with the fix already succeeded.** It is ten lines
(`research/plutarch_zero.py`) and it would have saved them the submission.

---

## 126 — 🟡 The scored gap is **half candidate speed, half serial draw**. `noop_reference_decode_speedup` is a dead field. My attempt to measure σ_score directly failed for lack of repeated trees.

`raw_p = mean(serial_s_per_tok) / mean(candidate_s_per_tok)` per prompt (verified
exact on 480 cells, zero mismatches). A rival can therefore lead either because
their candidate leg is faster — engineerable — or because their **serial leg read
slow**, which is a free gift from the measurement. Decomposing the gap to #1
(`research/gap_decompose.py`):

| | ours | #1 | #2 | #4 |
|---|---|---|---|---|
| as-scored | 3.23250848 | 3.24929399 | 3.24417897 | 3.24326224 |
| serial-normalised to the fixture pin | 3.23152211 | **3.23988292** | 3.23856400 | 3.23642288 |
| serial-draw contribution | +0.031 % | **+0.291 %** | +0.173 % | +0.211 % |

- 🔴 The **scored** gap to #1 is 0.5193 %. The **candidate-attributable** gap is
  **0.2587 %** — exactly the 0.258 % figure item 118 derived per-prompt. The other
  half is that #1's serial leg read 0.185 % above the pinned calibration while
  ours read 0.120 % below.
- 🟢 So item 118 was right about the engineering target and I was wrong to call it
  "the gap". **The engineerable target is 0.26 %, not 0.52 %.** Matching #1's
  candidate leg on all eight prompts would still leave us ~0.26 % behind *this
  particular row*, but a resubmission redraws the serial leg.

Board-wide serial draw over 414 scored rows: mean = the fixture pin to −0.002 %,
per-prompt CV **0.227 %**, p05→p95 span **0.331 %** of score for identical
candidate code. Per prompt the CVs are uniform (0.196–0.247 %), so no prompt is a
quieter instrument than any other.

**No selection effect, which surprised me.** Score is monotone in the serial draw,
so I expected the board top to be enriched in lucky draws. It is not: decile-1
serial = −0.005 %, top-20 = +0.012 % vs rest −0.003 %, t = **+0.46**. And
`corr(score, central-pair serial) = +0.15` on the 39 competitive rows against a
theoretical slope of exactly +0.100 % score per +0.100 % serial. The channel is
real algebraically and weak empirically; with n = 39 and a candidate-leg
correlation of −0.81 dominating the variance, the serial term is simply not
resolvable from board data. **I am not claiming a selection effect and I am not
claiming the leaders are lucky.** #1's +0.291 % is one draw from a distribution
with a 0.33 % span; that is unremarkable in isolation.

Two dead ends worth recording so nobody repeats them:

- 🔴 **`noop_reference_decode_speedup` is exactly `1.00000` on every prompt of
  every one of the 414 scored rows.** Zero variance — my correlation check divided
  by zero. It is *not* the organizers' no-op control. The real no-op measurements
  exist only in `fixtures/qwen3_8_27b_mtp_track.json` (`calibration`,
  `noop_decode_speedup_spread_pct`, item 117b). **Remove it from the exposed-field
  list as an information source.**
- 🔴 **σ_score cannot be measured from the board.** The clean test is two
  submissions of the *same tree*: 345 distinct `submissionCommitSha` and **414
  distinct `officialMetrics.commit` over 414 rows — nobody ever resubmitted an
  identical tree.** The test was well-posed and the data does not exist.

And one analysis of mine that does **not** hold up, recorded because I nearly
shipped it: I compared per-prompt serial CV (≈0.21 %) against per-prompt `raw_p`
CV (≈0.48 %) across the competitive cluster and concluded the serial noise "does
not pair away". That is **confounded** — the candidate CV in that cluster contains
genuine tree differences, so `raw_p` CV ≈ candidate CV is expected regardless of
whether pairing works. The comparison has no power. **The organizers' paired
σ_score = 0.078 % (six sessions, one unmodified tree, item 117b) remains the only
clean estimate and I am not overturning it.** My independence-assumption upper
bound of 0.175 % is neither supported nor refuted.

🟢 The decisive experiment is available to *us* even though it is not available
from the board: **submit one tree twice.** Two scored rows of an identical commit
give a direct draw from the score-noise distribution and calibrate our ranked A/B
instrument for the rest of the campaign. It costs one submission slot, which is
exactly why the standing rate-limit ask on issue 31 matters more than I thought —
it is not only about iteration speed.

🔴 One incidental consequence of the identity worth flagging before someone
"optimises" into it: because `raw_p` has the serial leg in the **numerator**, any
change that speeds up the shared depth-0 path lowers our score. If a base-path
speedup of δ also reduces candidate time by a fraction `f < 1` of that, `raw_p`
scales by `(1−δ)/(1−fδ) < 1`. **Base-model speedups are score-negative.** Our
E27 M-table shows `M=1: 0.9292`, i.e. our own tree *is* faster on the serial
path — our three scored rows read −0.053 % below the board mean (t = −1.45,
n = 3, not significant, so this is a flag and not a finding). Confining kernel
wins to M ≥ 2 is the correct discipline, and it is the discipline E33 already
follows.

---

## 127 — 🔴 I broke my own documented tooling rule for the **third** time, and it cost another 600 s.

Item 114 and the two turns before it record the rule: **never pass multi-line
content inline to `terminal`.** This turn I passed a multi-line `python -c "…"`
heredoc-style block. The shell garbled it, echoed a mangled fragment back three
times, ran nothing, and hung for the full 600 s timeout; the session then needed
`reset=true`, which cleared `YUKON_API_TOKEN` and every other env var.

Twice was a mistake. Three times is a process defect, so the rule is now
unconditional and has no exceptions for "but this one is short":

> **Every** multi-line payload — commit messages, Python, patches, JSON — goes
> through `file_editor create` to a file, and `terminal` only ever runs
> `python <file>` / `git commit -F <file>`. No inline newlines to `terminal`, ever.

The reason it keeps happening is worth naming: the failure is *silent and
expensive rather than loud and cheap*. A rejected compound command would teach me
in two seconds; a 600 s hang that produces mangled echo looks like a slow job, so
I wait. The two cheap detections are (a) the command echo coming back altered,
and (b) any `terminal` call with a literal newline in the argument — that second
one is mechanically checkable before sending, which is why it is now the rule.

Cost this turn: 600 s of wall clock, one terminal reset, one env-var reload. The
work itself was ten lines and took 20 s once written to `/tmp/plutarch_zero.py`.

---

## 128 — 🔴🔴 E34 corrected me three times and was right three times. Its headline number is then refuted by 15 ranked rows. And my "ranked kill" method has a confounder I had not been controlling.

Edward returned E34 as a terminal result under zero-GPU. It is the strongest
student result of the campaign on volume of corrected error, and it has one bad
number. Both halves matter.

### Three corrections I am adopting

**(1) 🔴 The head-artifact "discrepancy" was mine, not his.** I told him the prime
suspicion for his low local acceptance was that he was running the wrong head,
because his run reported `d038fd41…` while the manifest declares `559b24eb…` at
427,742,600 bytes. He resolved it: **`559b24eb` is the sha256 of the one-line tree
manifest; `d038fd41` is the sha256 of the file.** Both describe the same
single-file 427,742,600-byte tree. `head_verified: true`. We are on the declared
artifact and always were.
🔴 **The consequence propagates further than E34: alphonse's E30 absolutes were
measured on head `7bbb40de` at 270,408,194 B, which is not the declared artifact.
`F ≈ 14.79 ms` / `S ≈ 15.90 ms` is dropped from all sizing.** Use E34's
`S = 23.911 ms`, which comes off the E27-causal ladder on the right tree.

**(2) 🔴 My `R = (1 + αn)/(1 + h̄n)` identity is retracted as an inference tool.**
It is one equation in two unknowns, so per-prompt it **replays to 4.44e-16 by
construction and cannot fail**. The implied α wanders 0.3333 → 0.9019 rather than
sitting at 0.99, and plutarch's implied `h̄ = −0.5354` is a negative cost per head
step, which is unphysical. I had been using it as a validated cost model, quoting
`h̄` values to three students as if they were measured constants, and — worst — I
asked Edward to "validate" it by reproducing our beagle R from our beagle n and
h̄, a check that could not have failed.
**A fit that cannot fail is not evidence.** This is the same family as item 105
(a broken grep whose zero-hit output I published as fact): in both cases I
mistook a property of my instrument for a property of the world. Note that I very
nearly compounded it — a draft of my E34 feedback asked him to base the whole
counterfactual on that identity's `h̄`, and it survived only because the send
failed on a status check.

**(3) 🔴 The wall binds, against my prediction.** I argued that at ranked
`p ≈ 0.965` full-accept streaks would be common (`0.965⁵ = 0.842`, two in a row
≈ 0.71), so `sdpaWidthWallDepthCap` would rarely bind and be nearly inert. Wrong.
Fraction of rounds at M ≥ 6 (policy sim / max-entropy): plutarch .000/.017,
beagle **.538/.528**, medicine **.593/.561**, botany **.969/.770**; botany mean
M = 6.776 from exact integers. And the structural point I had missed: because
`widthCap = fullAcceptStreak >= 2 ? 8 : 5`, **the streak path is the thing that
carries M above 5**, so the two constants are coupled and cannot be moved
independently.

He also settled the `mtp_depth` question by exact enumeration — `(8, 8)` on all
408 scored rows including five zero-draft controls that realise n = 0, so both
fields are configuration echoes — independently of and agreeing with my fixture
read (item 119).

### The one result I most wanted, and it survived falsification

🟢 **(b) The per-width cost is a step function, not smooth.** Max residual in ms:
linear+step 5.630, linear+smooth 13.329, quad+step 3.861 (R² .99610), quad+smooth
12.645. **The step beats smooth by 3.4×**, with the causal receipt that E27 moved
only M=5 (132.257 → 108.346 ms, −18.08 %) while every other depth held within
±0.8 %. One weight pass `S = 23.911 ms`.
**thorfinn's E33 is sized on that structural claim and it is now load-bearing
rather than assumed.** Edward tried to break it and could not.

🟢 And an excellent provenance finding: `12b1c699` (audreyt, accepted) raised the
cap 4 → 5 inside a six-way composite **at a time when M=5 and M=6 both cost two
passes**, so the change crossed no boundary. E27 later moved the single-pass top
from M=4 to M=5, **leaving the constant one row past a cliff that did not exist
when it was chosen.** No ranked row has ever run our post-E27 dispatch table.
That is a real mechanistic argument for the cap cut and it needs none of the
modelling.

### The refutation: `predicted_ranked_central_pair_at_best_cap = 3.7786` is wrong

He predicts the central pair rises 3.24929 → 3.7786 (**+16.3 %**) from moving one
integer, honest interval [3.5498, 4.0073] — an interval that **excludes the status
quo**, which is the tell. A +16 % lever from one constant, on a board whose top 20
rows span 0.5 %, would not have survived 414 submissions unnoticed.

🔴 **My first attempt to kill it failed, in his favour, and taught me something
about my own method.** All eight `sdpaWidthWallDepthCap = 4` rows score 2.6825 to
2.9252 — apparently damning. But every one of them ran head `cc209e30…`, and that
head's **entire 107-row population ceilings at 2.9298**. Of the 34 distinct head
artifacts among the 414 scored rows, only `559b24eb` (n = 94) has ever exceeded
3.19853. The cap=4 evidence is completely confounded by head artifact, exactly as
Edward said.
**This is a standing correction to how I mine the corpus.** Item 120's ranked kill
of `values_per_thread` worked because companygardener's rows shared our head. I
had not been checking that. **Any score comparison across the corpus must first be
restricted to `head_provenance_sha256== 559b24eb…`; mechanism claims transfer
across heads, score claims do not.** The head changed around 2026-08-17, so
essentially the whole pre-Aug-17 corpus is a different population for scoring
purposes.

So I ran the test *inside* the declared-head population
(`research/declared_head_direction.py`). Of its 94 rows, 78 carry the default
schedule (best 3.249294, median 3.196562) and 16 do not. **Fifteen rows ran beagle
at an `n` other than 4.5327. All fifteen have a lower beagle `raw_p` than ours,
with no exception, and the loss is monotone in distance from the default in both
directions:**

| beagle n | beagle raw_p | vs ours |
|---:|---:|---:|
| 0.041 / 1.000 / 3.535 | 1.210 / 2.001 / 2.667 | −61.2 / −35.9 / −14.5 % |
| **4.279 / 4.339 / 4.384 / 4.396 / 4.454** | 3.110 / 3.077 / 3.060 / 3.012 / 3.077 | **−0.32 / −1.40 / −1.92 / −3.47 / −1.39 %** |
| **4.5327 (ours, default)** | **3.12015** | **— maximum** |
| 4.583 / 4.602 / 4.630 / 4.736 | 3.070 / 3.006 / 3.062 / 3.063 | −1.61 / −3.65 / −1.86 / −1.84 % |

The five bolded shallower rows are the **direct measured test of E34's direction**
and every one loses; E34 wants beagle at n ≈ 4.04, further shallow than any of
them, where the trend is already negative at 4.279 and steep by 3.535.

🟡 **Honest limit of this evidence.** Those rows changed acceptance by different
mechanisms and their trees differ in other ways, so no single row isolates the
acceptance effect — a shallower row with a slower kernel loses for the kernel
reason. It is a strong pattern, not a controlled experiment. But 15/15 with
monotone degradation on both sides is the signature of a local optimum at the
default, and it beats a model whose interval excludes the status quo.

**Disposition: revision requested (r2), bound to base `abf6d79f`.** (a), (b), (d),
(e) and the provenance finding are accepted as they stand. The narrow ask is to
re-issue or withdraw the primary metric and — the part I actually want — to say
**what the cost model gets wrong**, by feeding those five shallower rows' beagle
`n` back through `research/e34_cost_model.py`. My candidate for the missing term,
his to confirm or destroy: the model prices a round as `T(M)` and credits `n+1`
tokens, but by his own (d) the streak path is what produces the wide rounds, so a
cap cut may not move rounds from M=6 to M=5 — it may move them from `{6,7,8,9}` to
`{5}` while destroying the streaks that were generating cheap-per-token wide
rounds. That second-order effect is invisible to a model linear in M and has the
right sign.

### 🔴 Effect on E37, sent to askeladd immediately

E34's (d) gives beagle .538 and medicine **.593** at M ≥ 6 — medicine has *more*
wide mass than beagle, so **item 124's H1 predicts the beagle/medicine inversion
backwards**. H1's prior drops from 40 % to ~15 %.

Worse, E34 shows the local fixture may be unable to reach the ranked regime at
all: `research/e25r2-timed.json` has the candidate leg capping at
`effective_max_draft_len = 3` on all eight prompts (M ≤ 4), mean widths 2.98–3.69
against a ranked 1.15–6.78. If that still holds on the current tree, **no local
trace can census the scored window** and E37's deliverable 1 is unreachable. I
have restructured E37 to run a cheap reachability probe first and to treat "the
fixture cannot see this wall" as a complete terminal result — because that fact,
if true, means the campaign has *no* local instrument for any width-dependent
behaviour, which is worth more than the census was.

### 🟡 A correctness claim from the corpus, unresolved and worth an owner

`55fa8d31` (mpjunior92, accepted, 2.3955 — note the head, `157f750e`, so the score
is not comparable) reports that a previous cap-8 attempt **failed at rank** because
*"verify widths 6-9 drift from the serial trajectory in top-2 values … drifted
positions start exactly at the 6th row of a width-9 verify and recur
content-dependently, while widths 2-5 are bit-exact"*, and concludes *"do not raise
the cap without a bit-exact >width-5 GDN scan."* Their accepted submission stays at
cap 4 and verifies zero drift at widths 3–5.
This cuts **for** E34's direction on correctness grounds rather than cost, and it
is a latent parity risk for us: our cap 5 plus the streak path routinely runs
widths 6–9. Our `parity_all_ok` is true so we pass today. 🟡 It also **conflicts
with our own measurement** — `CURRENT_RESEARCH_STATE.md` records 919/919
non-terminal width-9 rows bit-exact with the only 15 mismatches at positions
1022–1024, i.e. positional at the KV boundary rather than width-driven. Two
solvers, two incompatible characterisations of the same region. Not resolved here.
