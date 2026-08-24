# E174 — Extend the fused-norm xsums epilogue to the 130 unserved producers

`harness=local` everywhere in this document. **No ranked transfer is claimed.**

- PR: #173, branch `qwen-alphonse/e174-xsums-epilogue-extension`, revision `r0`
- Base: `2b8037c3e5207fa2f5633e61d013a779492b05c2`
- Session commit: `9877d915f48a6d4ede58c5e6143ba9a59611e3f6`
- Worker: `1d9947bb01a48fec20a185b138ce1f08fc4e8f0cae7f0e4b8a79fce3b3e0adc4`
- Host: `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 48 GiB
- W&B: https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/7ueick4f
- Verdict on the assigned hypothesis: **unresolved, and held.** Not implemented.
- Delivered instead: first-hand per-fill and per-kernel-record coefficients,
  an exact additive decomposition of the shipped sumtable mechanism, and two
  retractions of my own earlier claims.

> **An earlier revision of this file said the axis was closed and that I would
> spend zero timed legs. Both statements were wrong.** The advisor approved a
> four-arm screen, I ran it, and it falsified my own argument for closing the
> axis. This document records what the measurement actually says.

## 1. What the assignment asked

Publish xsums from the producers of the 130 routed table-paying cells that have
no publishing producer, so those cells skip the standalone fill. The assignment
priced that at 0.650 ms/round of GPU fill work plus about 0.43 ms/round of host
admission work, both from my own E173 inventory, against a predeclared minimum
useful effect of **0.30 %** per-token local.

I objected before implementing, arguing that merged E160 evidence had closed the
axis. The advisor accepted the objection (ADVISOR ERROR 237, RULE 376), held the
implementation, and approved a screen to measure the coefficients first-hand.

## 2. Census — the 130 is measured, not source-derived

Two untimed 32-token traced legs, one worker, `dirty_candidate_paths=0`.
Artifact `research/out/e174-census-c1.json`.

| arm | `xs_hit`/round | `xs_fill`/round |
| --- | ---: | ---: |
| shipped | 127 | 130 |
| `MLX_E174_XSUMS_SIDECAR=off` | 0 | 257 |

Singleton counter sets on every traced round. Both arms digit-identical on
`all_tokens_matched`, `effective_mean_draft_len` and `accepted_draft_rate`, so
the switch changes only the fill count. This confirms the E173 source-derived
count of 130 exactly and closes ledger §332.3's standing ask.

## 3. The screen

Eight thermally gated 512-token legs in one session, palindrome
`s o r f f r o s`, so every arm has mean leg position 4.5 and a monotone drift
cancels to first order. All eight legs `cool_gate_passed_real_gate=true` and
`gate_qualified_for_timing=true`, entry-temperature spread 2.94 C, one worker
binary, `edl = 6.358974` and `accept = 0.87701613` digit-identical on all eight.

| arm | what it runs |
| --- | --- |
| `s` | shipped: 127 producer epilogues, 130 fills, table consumer `USE_TABLE=true` |
| `o` | `MLX_E174_XSUMS_SIDECAR=off`: 0 epilogues, 257 fills, table consumer `USE_TABLE=true` |
| `r` | `MLX_E120_QMV_ARM=replica`: 0 fills, `qwen35CustomAffine4QMVKernel` |
| `f` | `MLX_E120_QMV_ARM=fill_noconsume`: 257 fills, table consumer `USE_TABLE=false` |

Phase 1 witnessed all four arms from the run's own per-round census before any
timed leg ran. All four matched their source-predicted counters exactly, so no
arm was silently timing the shipped path.

### Results, absolute candidate MTP time

| contrast | grade | effect | µs/round | cells | µs/cell |
| --- | :-: | ---: | ---: | ---: | ---: |
| `o − s` | B | +0.3849 % | +699.1 | 127 | 5.505 |
| `f − o` | **A** | **+4.2997 %** | +7840.1 | — | — |
| `f − r` | C | +0.5190 % | +981.9 | 257 | 3.820 |
| `r − s` | C | +4.1605 % | +7557.3 | — | — |

Cleanliness is graded from source, not from the numbers. **A**: same kernel
object, same bindings, same grid, one declared thing differs. **B**: consumer
identical, more than one thing differs. **C**: different consumer pipeline
objects.

### Noise control

The serial leg is unreachable by every arm — `routable` refuses at `M = 1` — so
its arm-to-arm spread is a direct empirical null for a two-versus-two contrast
in this session: max **0.2232 %**, median 0.1117 % over the six pairs. `o − s`
clears it by 1.7×, `f − r` by 2.3×, `f − o` by 19×. Residual standard errors
give 7.4σ for `o − s`; the serial band argues for more caution than that, and
both are recorded.

## 4. The coefficient, as a bracket

`o − s` holds the consumer identical and additionally contains the producer's
epilogue cost, which is **≥ 0**, so it is a **lower** bound on one fill plus its
record. `f − r` additionally contains one extra buffer binding, also **≥ 0**, so
it is an **upper** bound.

| bound | source | µs/cell | 2σ |
| --- | --- | ---: | --- |
| lower | `o − s` | 5.505 | [4.03, 7.00] |
| upper | `f − r` | 3.820 | [2.39, 5.27] |

The bounds **cross**, so they cannot both be tight. They reconcile only inside
their overlapping 2σ region, or if the fill's marginal cost rises between the
130-fill and 257-fill regimes, which this session did not test.

> **Working value: one standalone fill dispatch plus its host kernel record
> costs about 4.0 to 5.3 µs of candidate MTP wall time on this host and tree.**

No point estimate is quoted. The crossing is reported rather than averaged away.

## 5. Two retractions

### 5.1 `f − r` is not the clean contrast; `o − s` is

I told the advisor that `f − r` holds the consumer fixed with `consume:` false
on both arms. Reading `Qwen35CustomQMV.matmul` again shows the opposite:
`replica` never enters the table branch, so it dispatches
`qwen35CustomAffine4QMVKernel` while `fill_noconsume` dispatches
`qwen35CustomAffine4QMVTableKernel` with one extra bound buffer. E135/F41
carried this caveat and I quoted its number without carrying the caveat.

### 5.2 E160 did not close the axis; it was underpowered by construction

The measured per-cell bracket predicts that removing E160's 64 fills **with a
completely free epilogue** would show `fuse − off` between −0.130 % and
−0.187 %. E160's 2σ intervals were `[-0.1752, +0.2181]` and
`[-0.2059, +0.2029]`.

**Its 2σ half-width (≈0.20 pp) was wider than the entire effect its own
mechanism could produce.** It could not have detected a perfectly free epilogue.
Its null is a failure to reject, not evidence of absence.

I therefore retract this claim from my blocker comment: *"The E173 cost premise
is therefore rejected at about 2σ, twice independently."* It was never tested
with enough power. My own E173 row — 0.650 ms / 130 cells = 5.0 µs per cell —
sits **inside** the bracket measured here. The row I apologised for was
approximately right; E135's 1.711 µs and E160's null were what misled me.

**Proposed amendment to RULE 376:** a merged null closes a mechanism axis only
if its confidence interval excludes the minimum useful effect. A null whose
interval is wider than the effect under test closes nothing.

## 6. Verdict on the assigned hypothesis

By the predeclared rule `|o − s| × 130/127 ≥ 0.30 %`, the premise **survives**:
0.3849 × 130/127 = **+0.394 %**.

**Unresolved, not falsified — and still not worth implementing.** The fills cost
real money. What remains unproven is that a producer *epilogue* can capture that
money on these particular producers. The source argument is untouched: no
producer of the 130 cells has a grid matching the fill grid `(32, kBlocks, m)`,
so each new epilogue is a sparse post-barrier tail. `o − s` measures the
epilogue on the **best available** producer, already in the tree; E174 must use
strictly worse ones. The ceiling is +0.394 % and the epilogue cost on worse
grids is unmeasured and could consume all of it.

Held per the advisor's ruling.

## 7. The sumtable mechanism, decomposed exactly

`f − o` is the grade-A contrast: same kernel, same bindings, same grid, fill
count held at 257, only the `USE_TABLE` template constant differs.

> **Reading the precomputed chunk-sum table instead of recomputing sums per
> output-row block is worth +4.2997 % = 7,840 µs/round of candidate MTP time.**

Exact additive decomposition, in absolute candidate seconds per token:

| component | contribution |
| --- | ---: |
| table consumption at fixed fill count (`o − f`) | **−0.00123291** |
| sidecar epilogue, 127 cells (`s − o`) | −0.00010994 |
| the fills themselves, 257 (`f − r`) | +0.00015440 |
| **total (`s − r`)** | **−0.00118845** |

### Relevance to receipt D

Receipt D's tree carried the campaign quantized kernels without the sidecar —
the arm `o` topology — and lost 1.34 % on ranked. Arm `o` measures only
**+0.385 %** slower than shipped here, because arm `o` still consumes the table
and merely self-fills. So the missing sidecar accounts for at most about a
quarter of D's loss and **something else explains the rest**. The 4.30 % figure
is the cost of losing table *consumption*, which is the failure mode if the
`_nax` variants do not consume the table.

## 8. Suggested follow-ups, which I did not implement

1. **Memoize the standalone fill on activation identity.**
   `Qwen35CustomQMV.matmul` calls `xsumsTable(x)` per cell with no memoization.
   The sidecar's `take()` already dedupes by activation identity, but only for
   published tables. Where several routed cells read the same activation — the
   standard transformer shape — each pays its own fill for an identical table.
   At 4.0–5.3 µs per fill every duplicate removed is worth that much, with no
   new kernel, no epilogue and no grid risk. Sizing needs one cheap census:
   distinct `x` identities among the fill calls in a round, against 257. If a
   quarter are duplicates that is ≈0.14–0.18 %. Not measured.
2. **Price item (2) against this bracket.** Removing 257 dispatch-plus-record
   pairs bought 982 µs/round = 0.52 %. Item (2) removes the record only, so its
   per-record value is strictly below 3.82 µs and its ceiling is ≈0.52 %, only
   if every record is cacheable, on the critical path, and the dominant half of
   the pair. Nothing here says the record is the larger half.
3. **Test whether the fill's marginal cost is regime-dependent.** That is the
   physical hypothesis that would reconcile the crossing bounds, and it decides
   whether removal work should be priced at 3.8 or 5.5 µs per cell.
4. **Carry method tags forward.** Both retractions here came from quoting a
   number without its caveat. An assignment citing a ledger row should carry
   that row's `method` tag and its stated caveats with it.

## 9. Reproduction

```bash
research/e174_census.sh          # census, 2 untimed traced legs
research/e174_screen.sh 512 s1   # 1 warmup + 4 witness + 8 gated 512-token legs
python3 research/e174_decompose.py
```

Artifacts: `research/out/e174-census-c1.json`,
`research/out/e174-witness-s1.json`, `research/out/e174-screen-s1.json`,
`research/out/e174-decompose-s1.json`. All under gitignored `research/out/`.
