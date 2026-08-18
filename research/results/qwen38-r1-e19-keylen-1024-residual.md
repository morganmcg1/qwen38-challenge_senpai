# E19 — the `key_len = 1024` residual band

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":false,"value":null},"test_metric":{"name":"arms_out_of_band","available":true,"value":0}}

- Assignment: `qwen38-r1-e19-keylen-1024-residual` r1, PR #21
- Student: `qwen-alphonse`
- Base: `senpai/qwen38-mtp-r1` @ `1bb627ab9339fd17c7560bd3d1134dc40fbb5885`
- Host: AWS Mac, Apple M4 Pro, 48 GiB (`hw.memsize = 51539607552`),
  macOS 26.5.2 (25F84), arm64
- **Zero-GPU assignment.** No timed run, no benchmark harness invocation, no GPU
  lock acquisition. Therefore **no W&B runs**: every number below is either read
  out of vendored source or recomputed from PR #2's already-committed row
  ledgers. The GPU lock queue at the time of this work was askeladd (#17),
  edward (#19 r2), thorfinn (#20); I was fourth and did not take it.
- Deliverable: `research/sdpa_keylen_band.py` (598 lines, five modes, exit 0)
- Decision: **not useful as a speedup, green as measurement.** There is no
  speedup here and claiming one would be wrong — this assignment had no timed
  leg and no score to move. What it produces is a tool, a theorem, and a
  status change on a hazard the campaign was carrying as open.
- Label I am claiming: **Not useful** in the `program.md` sense (no end-to-end
  gain, because none was available), **not Unclear**. See "Label" below for why
  the one open question cannot move the decision.

## Headline — the decision first

**The `key_len = 1024` drift is contract-legal today, with ~19.5× margin.**
That is the finding that changes what the campaign should do, so it goes first.

The trusted harness compares top-2 **logit values** under
`DFlashWorkBindingTolerance.matches` with AND semantics — `absolute = 4.875`,
`relative = 0.25`. PR #2's worst observed drift across all seven arms is
**abs 0.25, rel 0.011976**: ~19.5× and ~21× inside tolerance. The emitted-token
check needs only `top2Tokens.contains(token)`, so the runner-up-id flip at
`pos 1022` costs nothing, and `residual_divergence_count = 0` in every arm. What
made this look urgent was `research/mtp_row_gate.py`'s `id_mismatches`, which is
**research-only and strictly stronger than the contract**.

So the correct campaign action is: **do not build the repair, bank the
mechanism, and re-check the margin whenever the window, the tolerance, or the
prompt pool changes.** `key_len = 1024` moves from *open fidelity hazard* to
*latent risk with a located mechanism, a validated band, a quantified margin,
and an audited repair*. Full derivation in "Stake".

## The mechanism — the finding that closes a question

The advisor's §1.3 band table is **wrong**, and it is wrong in a way that makes
the hazard *smaller and cheaper to fix* than the assignment assumed. The
mechanism is real, I proved it from source, and I validated it against **3584
measured rows across all seven PR #2 arms with zero out-of-band observations**.

The load-bearing part is a theorem, not a table:

> **Width-invariance.** Both `sdpa_vector` kernels stride the key axis with a
> compile-time-constant stride and *skip* masked keys instead of shortening the
> loop. A row's output therefore depends only on `(kernel family, blocks,
> usable-key-set)` — **never on the allocated `k_len`, and never on the round
> width `qL`.**

That is why the drift exists at all, why it is bounded to three positions, and
why a repair would be nearly free. Four of the advisor's five §1 claims need
correction; §2 turns out to be moot; and the repair I designed **empties the
band entirely** — including the position the assignment thought was
unreachable — for **+1 SDPA launch in 1 of ~73–82 rounds per leg**. I did not
build it, because at ~19.5× margin it buys no score and no fidelity.

## Label — why this is Not useful and not Unclear

`program.md` reserves **Unclear** for when "noise, prompt sensitivity,
local-to-M5 differences, or cancellation in the local ratio could change the
decision." Exactly one question is open here: the ranked M5's architecture
suffix, which decides whether the two-pass path is reachable at all
(`'d'`/`'s'` yes, `'g'` no). It cannot change the decision:

- If the ranked host is `'g'`-class, the band is **empty** and the hazard does
  not exist there. Decision: don't build the repair.
- If it is `'d'`/`'s'`-class, the band is exactly the one validated on 3584
  rows, and the margin is the ~19.5× computed above. Decision: don't build the
  repair.

Every claim I make is either a source citation at a pinned base or a recompute
over already-committed ledgers, so there is no noise term and no local ratio to
cancel. And my own §2 — the `d`-vs-`s` blocks question — turned out **moot**
rather than resolved-against, so my stop rule (a) never fired. `--discriminate`
converts the open question into a transferable prediction instead of leaving it
as a dependency: a 512-token leg on a `'g'`-class box **must** show zero
deviation at pos 1022–1024. That is a test someone else can run for free on the
next ranked-adjacent measurement.

## Three durable negatives

These are results, not caveats on results. They survive whatever happens to the
ranked-host question, and each one closes off a direction the campaign would
otherwise have paid for. Pointers only — the reasoning stays where it belongs.

1. **The assignment's mechanism (c) is refuted.** The band does not arise the
   way §1.2(c) says it does. See "D2".
2. **The assignment's §1.3 band table is refuted** and replaced with one that
   matches 3584 measured rows row for row, including the counter-intuitive
   consequences that width 6 is bit-exact and width 4 is *worse* than width 8.
   See "D3" and "Validation".
3. **`MLX_SDPA_BLOCKS=32` measures the wrong term** and cannot answer the
   question it was proposed for — two independent reasons, one of which is that
   the worker's env allowlist never passes the variable through. See "D2".

## The dispatch law (all line numbers re-verified against the vendored tree)

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`
(**CPP**):

| site | law |
|---|---|
| `:635-638` | `supports_sdpa_vector = qL<=8 && qL<=kL && head_dim ∈ {64,96,128,256} && qL*gqa<=32`. At `gqa=6` ⇒ **`qL<=5`** |
| `:626-633` | `supports_sdpa_full` needs `head_dim ∈ {64,80,128}`. **head_dim is 256 ⇒ always false for this checkpoint** |
| `:640` | `use_fallback = !(supports_sdpa_full \|\| supports_sdpa_vector)` |
| `:745` | `do_causal = do_causal_ && q.shape(2) > 1` ⇒ serial (`qL=1`) runs **non-causal** |
| `:748-751` | **two-pass iff `((devc=='d'\|\|devc=='s') && kL>=1024) \|\| (kL<qL_heads && kL>=4096)`** |
| `:440-478` | `n_simds = gqa * qL`; `'s'`→`blocks=64`, escalates only `if (N>1024 && n_simds>4)`; `'d'`→`blocks=128` |
| `:495` | `array intermediate(shape, q.dtype(), ...)` — **two-pass partials are bf16** (`sums`/`maxs` are fp32) |

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h`
(**HDR**): single pass keeps its whole accumulator in fp32
(`:43 constexpr int BN = 32`, `:56 threadgroup U outputs[BN*BD]`); the two-pass
first stage rounds the partial numerator back to bf16 at
`:317-318 out[i] = static_cast<T>(o[i])`, and stage 2 reads it back through
`static_cast<U>(partials[i])`.

**So crossing `kL = 1024` is not a re-association change. It is a change of
accumulator storage precision.** The 512-token decode window starts at seed 512
and ends at 1024, so it crosses that gate exactly once, in exactly one round.

The `qL <= 5` ceiling also explains a piece of shipped code the assignment did
not account for. `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift`
splits any `qL ∈ 6..9` causal round into a 5-row call plus a remainder call.
It has to: `qL*6 > 32` blocks the vector path and `head_dim = 256` blocks the
full path, so an unsplit width-6 round would land on the **bf16 composite
fallback** (`fast.cpp:828-869` → `matmul`/`softmax(precise)`/`matmul`, a wholly
different numerical path). `--report` confirms `w = 10` is exactly where that
happens. `cap+1 <= 9` is load-bearing for fidelity, not just for speed.

## D3 — the corrected band table

`python3 research/sdpa_keylen_band.py --report`, for the round whose `kL`
reaches 1024. **Byte-identical for `devc='s'` and `devc='d'`.**

| w | p0 | kL | segments | injected | inherited | after repair |
|---|---|---|---|---|---|---|
| 1 | 1023 | 1024 | q1/k1024/2pass | [] | [] | [] |
| 2 | 1022 | 1024 | q2/k1024/2pass | [1023] | [1024] | [] |
| 3 | 1021 | 1024 | q3/k1024/2pass | [1022,1023] | [1024] | [] |
| 4 | 1020 | 1024 | q4/k1024/2pass | [1021,1022,1023] | [1024] | [] |
| 5 | 1019 | 1024 | q5/k1024/2pass | [1020..1023] | [1024] | [] |
| **6** | 1018 | 1024 | q5/k1023/vec; q1/k1024/2pass | **[]** | **[]** | [] |
| 7 | 1017 | 1024 | q5/k1022/vec; q2/k1024/2pass | [1023] | [1024] | [] |
| 8 | 1016 | 1024 | q5/k1021/vec; q3/k1024/2pass | [1022,1023] | [1024] | [] |
| 9 | 1015 | 1024 | q5/k1020/vec; q4/k1024/2pass | [1021,1022,1023] | [1024] | [] |

Five corrections to §1.3:

1. **The band belongs to the round whose `kL` reaches 1024**, not to any round
   near position 1022. Which positions are affected is a function of that one
   round's width, so it moves with the schedule.
2. **`pos 1024` drift is inherited, not injected.** `cache.update()` appends all
   `w` rows' K/V *before* the SDPA call, so row `p=1023` reads keys written at
   `p=1021,1022` in the same forward pass. Those keys are inside the injection
   band, so drift propagates `o_proj` → residual → the next full-attention
   layer (`full_attention_interval = 4`, 16 full-attention layers, all at the
   same `kL`). In serial those positions were produced in earlier single-row
   rounds at `N <= 1023` — single-pass, undrifted.
3. **The band is a superset, i.e. a necessary condition.** Every observed
   deviation is 1-ulp bf16; sub-ulp drift prints identically at 8-bit hex
   resolution, so the tool cannot claim the band is tight, only that nothing
   escapes it. I state this rather than over-claiming.
4. **The shipped chunk already repairs `w = 6` completely** and partially
   repairs `w = 7,8,9` — the assignment treated the hazard as unmitigated.
5. **The advisor's cache-position column is off by one** against PR #2's 1-based
   seed-inclusive `pos` convention.

## Validation — 3584 measured rows, 7/7 PASS

`--validate` replays each arm's committed `mtp-trace`/`mtp-row` ledger and
compares the predicted band against what was actually observed:

> ⚠️ **Do not cite `research/DO-NOT-CITE-kl-boundary-runJ.json` for the runJ row, or for
> anything else.** That pre-existing artifact was computed from the wrong arm:
> it reports **w3 / round 82**. The truth is **w4 / round 81**, read directly
> out of `research/trace-runJ-cap7-512.log`, where round 81 `d=3 acc=3` is the
> round that owns pos 1021–1024. Run O — a stamped repeat of J's config —
> independently agrees. Every runJ number in the table below comes from the
> trace log, not from that JSON. Separately, the `boundary_key_len` differences
> across the whole `kl-boundary-*` family are just the `--boundary` CLI flag
> being passed different values; they are not a finding and should not be read
> as one.

| arm | final round | w | kL | observed | predicted injected | inherited | |
|---|---|---|---|---|---|---|---|
| runI-base-cap8-512 | 82 | 8 | 1024 | [1022,1023,1024] | [1022,1023] | [1024] | **exact** |
| runJ-cap7-gate3-512 | 81 | 4 | 1024 | [1022,1024] | [1021,1022,1023] | [1024] | subset |
| runK-gate2-cap8-512 | 79 | 2 | 1024 | [1023,1024] | [1023] | [1024] | **exact** |
| runL-gate1-cap8-512 | 74 | 9 | 1024 | [1022,1024] | [1021,1022,1023] | [1024] | subset |
| runM-gate0-cap8-512 | 73 | 4 | 1024 | [1022,1024] | [1021,1022,1023] | [1024] | subset |
| runN-gate1-cap8-512-confirm | 74 | 9 | 1024 | [1022,1024] | [1021,1022,1023] | [1024] | subset |
| runO-cap7-gate3-512 | 81 | 4 | 1024 | [1022,1024] | [1021,1022,1023] | [1024] | subset |

`total compared rows: 3584 / arms out of band: 0`. The tool additionally asserts
per arm that committed positions **tile 512..1023 exactly** and that
`max(key_len) == 1024` — no overshoot, because the parent's `offeredDepth`
bounds every round (documented `Qwen36MTPBlockSession.swift:22`, applied
`:609-611`).

The decisive empirical confirmation of width-invariance is in the values
themselves. At `pos 1022`, serial ids `[6009, 98138]` vals
`['0x1.1cp+5','0x1.28p+4']` become mtp ids `[6009, 31098]` vals
`['0x1.1ep+5','0x1.2ap+4']` — **the identical mtp value at w ∈ {4,8,9}**. Three
different round widths, one output. That is the theorem, measured. `pos 1023`
likewise matches at w=8 and w=2.

## Repair — the band goes to zero for +1 launch per leg

Rule (implemented in `segments_repair`): below the boundary, delegate to the
shipped chunk. At the boundary, split rows into `low = {p <= 1022}` evaluated at
`k_len = p_last+1` and `high = {p >= 1023}` at full `k_len`, then chop each to
pieces of `<= 5` rows to stay inside `supports_sdpa_vector`. Because outputs are
width- and `k_len`-invariant, every low row lands on the single-pass fp32 kernel
exactly as serial computed it — which removes the injected drift, and therefore
also the inherited `pos 1024` drift, **for free**.

`--repair`, replayed on the seven measured schedules: every arm goes
`injected → []`, `inherited → []`, **`extra calls = 1`**. One extra launch in a
73–82 round leg.

`--sweep`, exhaustive over `p0 ∈ [512,1024)` × `w ∈ 1..9` (**4572 cases**), both
device classes:

```
shipped  cases=4572  misaligned=0  composite-fallback=0  cases-with-suspect-rows=7
repair   cases=4572  misaligned=0  composite-fallback=0  cases-with-suspect-rows=0
```

This settles the five §4 attacks on the repair:

- **(i) off-by-one** — 0 misaligned in 4572 × 2 cases, both segmentations. The
  shipped chunk's `kSplit = kL - (qL - 5)` equals `p_last(A) + 1`, causally
  exact.
- **(ii) overshoot past 1024** — impossible; `max(key_len) == 1024` measured in
  all seven arms, depth bounded by the parent's `offeredDepth`, `Qwen36MTPBlockSession.swift:609-611`.
- **(iii) composition with the shipped chunk** — 0 composite-fallback escapes;
  repair delegates below the boundary and fully replaces above it, so the two
  never both fire.
- **(iv) `concatenated(axis: 2)` bit-exactness** — settled empirically with no
  new code: the shipped chunk *already* uses it in the timed path, and measured
  rows away from the boundary are bit-exact (runI w=9, 274 rows `bit_exact:true`;
  runJ/O w=8, 356 rows). Had `concatenated` perturbed values, every w ∈ 6..9 row
  would have deviated.
- **(v) strided cache view** — `Sources/MLXFastModel/KVCache.swift:437-438`
  returns `keys![.ellipsis, ..<offset, 0...]`, so `cachedKeys.dim(2) == offset`
  despite `step = 256` padding; the view survives `kv_copy_unless` because
  `shape[0] == 1`, and `k_seq_stride = k.strides()[2]` handles it.

Per §6(d), this surviving-but-unbuilt repair sketch is a full pass. It is
machine-audited but **not yet implemented in Swift**.

## D1/§2 — moot, and then answered anyway

§2 asked me to pin the M5's architecture suffix. **It cannot change any
conclusion**: `CPP:748` treats `'d'` and `'s'` identically for the `kL>=1024`
gate, and the only downstream difference — `blocks` 64 vs 128 — does not alter
the band, which `--report` confirms is byte-identical across the two.

Better, the measured rows answer the question themselves. `--discriminate`:

| devc | two-pass at kL=1024 | blocks | predicted band | arms unexplained |
|---|---|---|---|---|
| s | True | 64 | [1022,1023,1024] | **0** |
| d | True | 128 | [1022,1023,1024] | **0** |
| g | False | — | [] | **7** |
| p | False | — | [] | **7** |

A host that shows window-end deviation is necessarily in `{'s','d'}`, so PR #2's
rows are evidence about the measuring host. This yields a **transferable
prediction that is a better M5 test than probing the architecture string,
because it tests the consequence**: a 512-token candidate leg on a `'g'`-class
box must show *zero* deviation at pos 1022–1024.

Stop rule (a) does not fire: the answer is not "M5 is not `d`/`s`", it is
"the question does not gate the result".

## D2 — §1.2(c) refuted

Two independent reasons the advisor's proposed `MLX_SDPA_BLOCKS=32` test cannot
confirm claim (c):

- **It measures the wrong effect.** bf16 partial-numerator rounding
  (`CPP:495`, `HDR:317-318`) is ≈1e-3; fp32 re-association is ≈1e-7. Changing
  `blocks` varies the ≈1e-7 term while leaving the ≈1e-3 term in place.
- **It is structurally unreachable in the ranked worker.** The ranked workflow's
  env block is entirely `MLXFAST_*` with zero `MLX_*`, and the allowlist at
  `QwenRuntimeWorker.swift:2638-2645` admits only `MLXFAST_*`. `MLX_SDPA_BLOCKS`
  is a local diagnostic, never a fix.

Advisor-correct items, confirmed: `blocks = 64` on `'s'` at `N <= 1024`; the
`>= 4096` GQA clause is unreachable in a 1024 window; claim (a) partitions are
independent of `N` (now *proved* as width-invariance); claim (b)'s causal cut
lands at the same absolute position (now *machine-audited*, 0/4572 misaligned).

## D5 — `head_dim` is 256, not 128, and the correction is load-bearing

The assignment's `head_dim = 128` is wrong. Verified geometry:
`head_dim = 256`, `num_attention_heads = 24`, `num_key_value_heads = 4`,
`gqa_factor = 6` ✓, `hidden_size = 5120`, `full_attention_interval = 4`,
`num_hidden_layers = 64`, `vocab_size = 248320`, `model_type = qwen3_5_text`.

Sources: resident `weights/config.json`; HF pin
`EigenLabs/Qwen3.8-27B-4bit @ eda45ab47f465d08d6558f0353a2346e2eb9d5b3`; all four
`~/.cache/mlxfast/**/mtp-head/config.json`;
`fixtures/qwen3_6_27b_config.json:text_config`. Corroborated by real GEMM shapes
(`research/qmv_cost_curve.py:26-27`): `qkv_proj_fused` 5120→**14336**
(`= 12288+1024+1024`) and `o_proj` **6144**→5120 (`= 24×256`).
`grep -rnE '\b(7168|3072)\b' research/ Sources/` returns **zero hits**.
Enforcement is fail-closed at `QwenRuntimeWorker.swift:699-741`.

This is not cosmetic bookkeeping: **256 ∉ {64,80,128} is precisely why
`supports_sdpa_full` is permanently false, which is why the composite bf16
fallback is the `w >= 10` destination, which is why the `qL ∈ 6..9` chunk must
exist.** Under the advisor's 128, `supports_sdpa_full` would be reachable and
the whole §1 analysis would have a different shape.

## Stake — the drift is contract-legal today

`Sources/MLXFastTrustedHarness/QwenRuntimeDFlash.swift:~795-830` compares
candidate vs reference top-2 **logit values** under
`DFlashWorkBindingTolerance.matches` with **AND** semantics:
`absolute = 4.875`, `relative = 0.25`
(`Sources/MLXFastCore/Constants.swift:604`, `:622`). PR #2's observed maxima are
**abs 0.25** and **rel 0.011976** ⇒ **~19.5× and ~21× margin**. Emitted-token
admissibility (`:773`) only requires
`referenceRow.top2Tokens.contains(token)` for the *emitted* token, so the
runner-up-id flip at `pos 1022` costs nothing — `residual_divergence_count = 0`
in all seven arms. The Qwen worker carries the same fields
(`QwenRuntimeWorker.swift:1103-1104, 1123-1124, 1146-1147`).

`research/mtp_row_gate.py`'s `id_mismatches` is a **research-only metric,
stricter than the contract**. Reading it as a violation is what made this hazard
look urgent.

So: E19 is a **latent-risk** finding. It becomes live only if a longer window,
a tighter tolerance, or a near-tie prompt reduces that margin. The value of the
repair is that it costs ~1 launch per leg and removes the class of risk
outright.

## Honest limitations

- **No GPU run, no W&B run.** Every claim is source-derived or recomputed from
  already-committed ledgers. The dispatch law is read from vendored C++/Metal,
  not observed executing.
- **The band is a superset, not a tight set.** 1-ulp print resolution; sub-ulp
  drift is invisible to this method.
- **The repair is designed and audited, not built.** No Swift, no compile, no
  timing. Its `+1 launch` cost is a count of SDPA calls, not a measured latency.
- **One print inconsistency I will not over-fit:** runI `w=8` shows top-1
  `0x1.28p+5` where other arms show `0x1.2ap+5`. I report it rather than
  explaining it away.
- **A provenance bug others should not cite:** `research/DO-NOT-CITE-kl-boundary-runJ.json`
  was computed from the wrong arm's trace (it says w3/round 82; the truth is
  w4/round 81). The `boundary_key_len` differences across `kl-boundary-*` are
  just the `--boundary` flag.
- **Local host is M4 Pro** and reports `applegpu_g16s` → `devc = 's'`. It is not
  the ranked M5.

## D4 — what a GPU-instrumented log would add

`research/sdpa_keylen_band.py` is the zero-GPU form of the requested
dispatch-decision log: it reproduces the decision, not the execution. With
harness time I would add exactly two things, both cheap:

1. A runtime print of `GPU.deviceInfo().architecture` on the ranked box, closing
   §2 directly instead of by inference.
2. A runtime confirmation that `intermediate` is bf16 in the scored process,
   closing the one step of the mechanism that is currently source-only.

**Named missing fact (non-blocking):** one Host-preflight CI log line, or a
host-only probe, printing `GPU.deviceInfo().architecture` on box
`m5-max-128gb-3`. Docs call the ranked box **Apple M5 Max** in five places
(`docs/qwen-mtp-go-live-runbook.md:65`, `README.md:191`, `:543`,
`docs/private-benchmark-security.md:23`, `docs/benchmark-window-freeze.md:199`),
and the tier comments at `device.cpp:574-590` map "max" → `'s'`; but my local
M4 **Pro** reports `g16s`, so I treat the naming→suffix mapping as directional
and refuse to assert it.

## D8 — next step

The cheapest next move is **not** to build the repair. It is to spend one
already-queued GPU slot confirming the mechanism transfers, because the repair's
value is entirely conditional on the ranked box being `'s'`/`'d'`:

> On the next 512-token candidate leg that runs for any reason, dump the
> `mtp-row` ledger and check pos 1022–1024. Deviation there confirms
> `'s'`/`'d'` and the mechanism on the ranked host; **clean rows falsify the
> whole E19 chain on M5** and the repair should be dropped.

That check is free — it rides along on any 512-token run and needs no new arm.
If it confirms, the Swift repair is ~30 lines in the `AttentionUtils.swift`
chunk plus a boundary unit test, and should be measured for its `+1 launch`
cost before promotion. But since the margin is ~19.5×, the honest reading is
that E19 is a **documented latent risk with a known fix** and should wait behind
work that moves the median. The advisor has since attached this rider to two
already-planned runs — askeladd's E15 Phase 3 512-token ABBA and the
fixed-window overlay's 512-token exact replay — so it costs no new allocation.

One follow-up I did **not** implement, per scope: the §5 freebie
(`DEEP_CAP = 7`, re-score J/O/P₅₁₂ in `occupancy_model.py`). It is a separate
result and should be assigned as one.

## ❌ CANCELLED ON EVIDENCE — the `AttentionUtils.swift` boundary repair (recorded 2026-08-18)

The ~30-line boundary repair proposed above is **cancelled, not deferred**. Do not resurrect it
without new evidence that contradicts the measurement below.

The advisor's rider ran on a 512-token leg that crossed `key_len` 1024 with **513 rows**:

| quantity | E19 bound | measured | headroom |
| --- | --- | --- | --- |
| min top-1 / top-2 margin over the crossing | hazard needs `absolute ≤ 0.25` | **0.375** | 1.5× |
| boundary-window margins | inside tolerance | **55×–78×** inside | — |
| exact ties at the boundary | any tie is a hazard | **0** | — |

The E19 envelope was **right** — the band exists and the mechanism is real — but the hazard is not
practically reachable on the ranked trajectory, so the repair buys no correctness and costs `+1`
kernel launch per full-attention layer per forward. The correct disposition is: **documented latent
risk, known fix, not worth a launch.** Reopen only if a future run shows a margin at or below `0.25`
across the `key_len` 1024 boundary.

**Loose end now closed:** `Sources/MLXFastModel/Qwen35FastEngine.swift` is
**never executed**. `Sources/MLXFastModel/Qwen35FastPathReadiness.swift:13-19`
hardcodes `realCheckpointParityPassed = false` and
`productionActivationApproved = false` and derives `productionBackend` from
exactly those two constants, and `AGENTS.md:138-140` states that the separate
`Qwen35FastEngine` path "is not the MTP worker's current target path". The whole
`Qwen35{Attention,Block,GatedDelta,MLP,Model,Ops,RoPE,FastEngine}.swift` family
is editable but dead. This never could have changed an E19 answer, but it does
retire a listed uncertainty: nobody should spend a slot optimizing that file.

## Reproduction

```bash
git checkout qwen-alphonse/keylen-1024-residual
python3 research/sdpa_keylen_band.py                 # all five modes, exit 0
python3 research/sdpa_keylen_band.py --report        # analytic band table
python3 research/sdpa_keylen_band.py --validate      # 3584 rows, 7/7 PASS
python3 research/sdpa_keylen_band.py --repair        # band -> [], +1 call
python3 research/sdpa_keylen_band.py --discriminate  # device-class evidence
python3 research/sdpa_keylen_band.py --sweep         # 4572 cases x {s,d}
python3 research/sdpa_keylen_band.py --devc d        # invariance across d/s
```

No GPU, no model weights, no network. Reads only PR #2's committed
`runI..runO` trace/gate JSON already in-tree.
