# E174 — What one standalone chunk-sum fill costs, and where the 130 of them come from

PR #173, branch `qwen-alphonse/e174-xsums-epilogue-extension`, base
`2b8037c3e5207fa2f5633e61d013a779492b05c2`. `harness=local` throughout. No
ranked coefficient is derived from any number here.

The assigned hypothesis — extend the fused-norm xsums epilogue to the 130
unserved producers — is **held permanently** by advisor ruling F5 (ADVISOR
ERROR 237, RULE 376). This report covers the approved substitute programme: a
first-hand cost screen, a dedup census, and a CPU-side host record probe.

## Identity tuple

| field | value |
| --- | --- |
| base SHA | `2b8037c3e5207fa2f5633e61d013a779492b05c2` |
| screen session commit | `9877d915` |
| census session commit | `e3826bcc` |
| worker (census) | `5e7f0a94323fce4b6aaf86a4830b5d9071d736ffd09df9637bf23022cc33674d`, unchanged pre and post run |
| worker (screen) | `1d9947bb01a48fec20a185b138ce1f08fc4e8f0cae7f0e4b8a79fce3b3e0adc4` |
| host | `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 48 GiB |
| token window | 512 decode tokens, public local fixture |
| head | declared run tree |
| reference source | candidate-generated local reference rows |
| harness | `local` |

## 1. The coefficient (screen, 8 gated legs, W&B `7ueick4f`)

All eight legs `gate_qualified_for_timing=true`, entry-temperature spread
2.94 C, one worker binary, `dirty_candidate_paths=0`,
`all_tokens_matched=true`, `edl=6.358974` and `accept=0.87701613`
digit-identical on all eight.

| contrast | grade | effect | µs/round | cells | µs/cell |
| --- | :-: | ---: | ---: | ---: | ---: |
| `o − s` | B | +0.3849 % | +700.4 | 127 | 5.505 |
| `f − o` | **A** | **+4.2997 %** | +7840.1 | — | — |
| `f − r` | C | +0.5190 % | +984.4 | 257 | 3.820 |
| `r − s` | C | +4.1605 % | +7557.3 | — | — |

The two per-cell bounds cross, so the result is a bracket, not a point:

> **One standalone fill dispatch plus its host kernel record costs about
> 4.0 to 5.3 µs of candidate MTP wall time on this host and tree.**

Empirical null: the serial leg is unreachable by every arm (`routable` refuses
at M = 1), and its arm-to-arm spread is max 0.2232 %, median 0.1117 % over six
pairs.

## 2. Dedup census — a structural zero (W&B `rn4y98s1`)

Two untimed traced 512-token arms, 77 differenced rounds each, one worker.

| arm | fills/round | distinct activations/round | duplicate fraction |
| --- | ---: | ---: | ---: |
| `s` shipped | 130 | 130 | **0.000** |
| `o` sidecar off | 257 | 257 | **0.000** |

Three controls:

1. **Per-round instrument self-check.** Two references to one `MLXArray` and
   one separately allocated array must key as 2 distinct identities with
   exactly one `2x` bucket. `ok` on all 78 rounds of both arms.
2. **Counter/summary agreement.** Each round's differenced cumulative counters
   equal that round's own self-describing summary, on every round. The census
   does clear between rounds.
3. **In-situ identity control.** Arm `s` records 127 sidecar `take` hits per
   round, keying on the same `ObjectIdentifier` mechanism. Identity matching
   demonstrably works at this call site on these real activations.

A strong reference is held for every activation of the current round, so a
released tensor cannot have its address reused and merge two distinct
activations. That error would have **overstated** the duplicate fraction.

**Why it is zero.** The 257 cells sit in a fixed 2 : 1 : 1 ratio plus one:

| width | cells/round | consumer | producer |
| --- | ---: | --- | --- |
| 5120 | 129 | fused `gate_up` 64, `gdn.in_proj` 48, `fa.qkv` 16, `lm_head` 1 | the two RMSNorms |
| 6144 | 64 | `gdn.out_proj` 48, `fa.o_proj` 16 | attention output |
| 17408 | 64 | `mlp.down` 64 | SwiGLU output |

Qwen 3.8 fuses QKV and fuses gate/up, so every activation has exactly one QMV
consumer. There is nothing to memoize. The memoization follow-up I proposed
after the screen is dead, and I proposed it on an unfused-transformer shape
assumption I did not check against the source first.

**Independent cross-validation.** This width histogram (129 / 64 / 64) is
derived from a runtime trace on real activations. It reproduces FINDING 454's
source census (Entry 378, corrected E176) cell for cell, by a completely
different method.

## 3. Host kernel record probe (W&B `rn4y98s1`)

`MLXFastKernel.callAsFunction` is the record item (2) proposes to cache: an
`mlx_fast_metal_kernel_config`, the template args, the grid, the output args,
an input `vector_array`, and `mlx_fast_metal_kernel_apply`. MLX is lazy, so
that call returns an unevaluated node — the host has paid for the record and
nothing else. `eval` then pays the encode, the dispatch and the compute.

Three real activation widths, 9 repeats, batches 32 and 256, medians:

| width | record µs/call | eval µs/call | batch-scaling ratio |
| --- | ---: | ---: | ---: |
| 5120 | 5.751 | 7.969 | 1.044 |
| 6144 | 5.566 | 7.825 | 1.044 |
| 17408 | 5.518 | 8.186 | 1.042 |

**Positive control green.** Record construction is per-call work, so its
per-call cost must not track the batch size. The three ratios are 1.04, so an
8x change in batch moves the per-call figure by 4 %. A timer reading a fixed
per-batch overhead would have shown roughly 8x.

Record cost is also flat in `K` while the activation grows 3.4x, which is what
a fixed host overhead should do and what data-dependent work should not.

### Reconciliation with the screen, and the arithmetic stop rule

Isolated record 5.566 µs, isolated eval 7.969 µs, isolated total 13.535 µs.
The marginal end-to-end cost of the same call is 4.0–5.3 µs, so the isolated
total is 2.6–3.4x the marginal cost: in a real round the host builds records
while the GPU runs earlier work, and the probe cannot be read straight across.

Two defensible overlap assumptions bracket the record's share of the marginal
cost:

| reading | record share of the 4.0–5.3 µs |
| --- | --- |
| uniform overlap (both phases hide the same fraction) | 1.65 – 2.18 µs |
| serial record (host record on the critical path 1:1, tiny dispatch hides) | 4.0 – 5.3 µs |

**The advisor's stop rule does not fire under either reading.** The minimum is
1.65 µs, against a 1 µs threshold. Item (2) is **not dead by arithmetic**.

### The refinement that decides item (2), which I did not measure

Reading `MLXFastKernel.callAsFunction` in
`Vendor/mlx-swift/Source/MLX/MLXFastKernel.swift`, only part of the record is
cacheable:

| sub-step | cacheable? |
| --- | --- |
| `config_new`, template args, `set_grid`, `set_thread_group`, `add_output_arg`, `set_verbose`, `config_free` | **yes** — depends only on `(K, m)`, and a round has about 15 distinct `(K, m)` pairs |
| `new_mlx_vector_array(inputs)` | no — the input tensors change every call |
| `mlx_fast_metal_kernel_apply` and its output arrays | no — this is the graph node |

So item (2)'s real ceiling is the **config share** of 1.65–5.3 µs, not the
whole of it. I did not measure that share. Splitting it needs the C symbols
from the `Cmlx` module, which needs a `Package.swift` dependency edit, and
`program.md` forbids changing the package graph. That split is the first
question any item (2) assignment should answer, and it is cheap once someone
decides the package-graph rule permits a test-only dependency.

### This replaces, and raises, the E173 row

E173's `admission.kernel_record_construction` was 1.2842 ms/round, computed as
a residual of an entry-point total after three other measured rows, so it
inherited all of their errors. The shipped tree builds 257 QMV records plus
130 fill records = 387 records per round. At the probe's 5.566 µs that is
**2.154 ms/round** of isolated host record work, about 1.7x the residual. I
report the discrepancy rather than choosing; the probe is direct, but it is
also a tight loop with warm caches and no contention, which is a best case.

## 4. Pricing the 130 producer-side fills (the advisor's question)

Arm `s` round time 181,992 µs. Shipped standalone fills 130/round, measured.

| priced at | µs/round | % of round |
| --- | ---: | ---: |
| 4.0 µs (bracket low) | 520.0 | 0.286 % |
| 5.3 µs (bracket high) | 689.0 | 0.379 % |
| 5.566 µs (probe record) | 723.6 | 0.398 % |

Against the 0.30 % minimum useful effect, the ceiling clears it at the top of
the bracket and misses at the bottom.

**This is a ceiling and it assumes a free epilogue, which it is not.** Two
facts constrain the realizable value:

- No producer of the 130 cells has a grid matching the fill grid
  `(32, kBlocks, m)`. The attention output and SwiGLU producers are row-major
  and their epilogue is a sparse post-barrier tail — this is FINDING 359's
  diagnosed mechanism, and the census now names exactly which two producers
  would have to carry it.
- GPU work on this pipeline is **not** free. The grade-A contrast `f − o`
  measures +4.2997 % for adding per-output-row-block sum recomputation with
  the record count held fixed. So an epilogue tail costs real wall time.

The trade for each converted cell is therefore: remove one host record
(1.65–5.3 µs of marginal cost) and add one sparse post-barrier epilogue tail
(unmeasured GPU time). E160 measured this trade at 64 cells and found a null,
but its 2σ half-width of about 0.20 pp exceeded the entire effect its own
mechanism could produce (−0.130 % to −0.187 %), so its null is a failure to
reject, not evidence of absence.

**What would settle it, cheaply:** measure the epilogue tail cost alone, on one
producer, with the fill count held fixed — the same design discipline that made
`f − o` a grade-A contrast. If the tail is cheaper than the record it removes,
the axis reopens with a measured expectation instead of an argument.

## 5. Compliance and scope

- Every instrument is default-off and opt-in behind an `MLX_`-prefixed
  environment switch, witnessed in the built worker by
  `rebuild-and-assert-worker.sh --require`.
- `senpai/verify-ranked-score-boundary.sh` OK. Assignment scope OK, 2 submitted
  paths. Editable budget growth 8243/262144, source 2,637,289/3,000,000.
- **The scored surface carries research instruments and must be stripped
  before any freeze.** `MLX_E174_XSUMS_SIDECAR` (`b5fd35b1`), the dedup census
  and its counter (`71a6e773`, `e3826bcc`), and the `xs_uniq`/`xs_dedup` trace
  fields all sit in `Qwen35.swift` and `Qwen36MTPBlockSession.swift`. The
  shipped path reads one `static let` Bool per fill call. That is small, but
  section 3 shows the admission phase is host-sensitive, so I do not claim it
  is free.
- No timed pair was spent. The zero duplicate fraction removed the only thing
  step 2a would have measured.
