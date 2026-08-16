# E6-r1 — Compact draft-readout precision: 4-bit vs 3-bit vs 2-bit

Assignment `qwen38-r1-e6-draft-head-precision`, revision `r1`, PR #7.
Base `senpai/qwen38-mtp-r1` @ `b2419f41fe500bb7bb8ceea523352994a7fad0ea`.
Result commit: see PR result marker.

**Host: Apple M4 Pro, 20-core GPU, 48 GiB (51,539,607,552 B). This is NOT the
ranked M5 (`m5-qwen38-27b-mtp`). Every number here is directional evidence.**

Head pinned identically across all arms:
`eb481df38267db5c9d9db1f6a813fcc73e762d0af74fdb1bcb061724c815adfe`.

---

## 0. Answer in one line

Both low-bit arms beat the 4-bit control on ms/token and **neither costs
acceptance**. 3-bit gained +1.92 pp acceptance on this fixture; 2-bit is
acceptance-neutral to 16 significant digits. Per the stopping rule ("2-bit is
~free → report it and quantify the noise"), this experiment stops here.

| | 4-bit control | 3-bit | 2-bit |
|---|---:|---:|---:|
| ms/token @512 | 35.1193 | **34.4514 (−1.90 %)** | **34.6194 (−1.42 %)** |
| accepted_draft_rate | 0.890269 | **0.909474 (+2.16 %)** | 0.890269 (±0) |
| local ratio serial/MTP | 2.09455 | **2.13141** | 2.12441 |
| head bytes | 283,207,680 | 220,272,640 | 157,337,600 |
| parity_all_ok | ✅ | ✅ | ✅ |

Run-to-run noise on this host, measured from two independent 4-bit controls:
**0.0137 ms/token (0.039 %)** and **0.058 %** on the local ratio. Both wins are
25–36× that noise.

---

## 1. Step 0 — fidelity, confirmed from source

`Sources/MLXFastModel/Qwen36MTPBlockSession.swift:641-649`:

```swift
static func acceptedDraftPrefixCount(drafts: [Int], verifyArgmax: [Int]) -> Int {
    precondition(verifyArgmax.count >= drafts.count)
    for index in drafts.indices where verifyArgmax[index] != drafts[index] {
        return index
    }
    return drafts.count
}
```

Exact token-ID equality against the verify argmax; first mismatch wins; no
threshold, nothing probabilistic. Sole call site `:931`. Pinned by
`Tests/MLXFastTests/QwenMTPFixedWindowTests.swift:12,28` (I re-ran both:
`acceptedDraftPrefixCount([41,eos,73,89],[41,eos,73,97,101]) == 3`, and
`onlyATargetMismatchEndsTheAcceptedPrefix`).

**Therefore lowering *draft* precision cannot change the emitted stream — only
the acceptance rate.** The structural argument holds. It did not substitute for
the parity run; see §2.

## 2. Evidence contract item 1 — parity and emitted-stream identity

| | 4-bit control | 3-bit | 2-bit |
|---|---|---|---|
| `parity_all_ok` | True | True | True |
| `all_tokens_matched` | True | True | True |
| `residual_divergence_count` | 0 | 0 | 0 |
| `declared_rows_total` | 565 | 556 | 565 |
| `reference_checked_row_total` | 565 | 556 | 565 |
| `emitted_token_total` | 512 | 512 | 512 |
| `max_rejected_tail_logit_delta` | 0 | 0 | 0 |
| `ref_emitted_stream_sha256` | `da92be8a0dc02229485292258a502a3b7a1aa896fb6e23ca0071e019f65acf90` | **identical** | **identical** |

**Byte-identical, stated plainly: yes.** All three arms emitted the same 512
tokens with the same SHA-256, each reproduced its own reference with zero
residual divergences, and row accounting closed exactly
(`declared_rows_total == reference_checked_row_total`) in every arm.

## 3. Headline table — 512 decode tokens

`MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512`, same host, same session, one
model-holding process at a time.

| field | 4-bit control-b | 3-bit | 2-bit |
|---|---:|---:|---:|
| W&B run | `hrgew6pe` | `ey56o2j5` | `ue0l9ryy` |
| arm dir | `.mlxfast-private/draft-bits/e6-r1-bits4-control-b` | `…/e6-r1-bits3` | `…/e6-r1-bits2` |
| head total bytes | 283,207,680 | 220,272,640 | 157,337,600 |
| packed bytes | 251,740,160 | 188,805,120 | 125,870,080 |
| scale+bias bytes | 31,467,520 | 31,467,520 | 31,467,520 |
| `accepted_draft_rate` | 0.8902691511387164 | 0.9094736842105263 | 0.8902691511387164 |
| Δ acceptance vs control | — | **+0.0192045330718099 (+2.157 %)** | **0.0 (exactly)** |
| accepted / rejected | 430 / 53 | 432 / **43** | 430 / 53 |
| `round_count` | 82 | **81** | 82 |
| `draft_readouts_total` | 483 | 475 | 483 |
| readouts / round | 5.890243902439025 | 5.864197530864198 | 5.890243902439025 |
| `ms_per_token` | 35.119320498779416 | 34.45143951103091 | 34.61940819397569 |
| Δ ms/token | — | **−0.6678809877 (−1.90 %)** | **−0.4999123048 (−1.42 %)** |
| `decode_seconds` | 17.98109209537506 | 17.639137029647827 | 17.72513699531555 |
| `local_ratio_serial_over_mtp` | 2.0945467495553607 | 2.1314095445870165 | 2.1244104330733418 |
| Δ local ratio | — | **+0.03686 (+1.76 %)** | **+0.02986 (+1.43 %)** |
| `serial_ms_per_token` | 73.55905859731138 | 73.43012699857354 | 73.54583195410669 |
| `target_tail_total` | 82 | 81 | 82 |
| `verify_block_replayed_round_count` | 13 | 12 | 14 |
| `non_drafting_round_count` | 0 | 0 | 0 |
| `first_block_seconds` | 0.17038094997406006 | 0.17851901054382324 | 0.18025696277618408 |
| `p50_block_after_first` | 0.16892600059509277 | 0.16716301441192627 | **0.16558802127838135** |
| `max_block_after_first` | 0.22012102603912354 | 0.21704590320587158 | **0.21464908123016357** |
| `max_rss_bytes` | 16,215,195,648 | 16,215,212,032 | 16,230,760,448 |
| `gpu_temp_c_before` (°C) | 41.7988 | 44.9525 | 45.8138 |
| `cool_gate` | disabled_ambient_floor | disabled_ambient_floor | disabled_ambient_floor |

The three `serial_ms_per_token` values (73.559 / 73.430 / 73.546, spread
0.18 %) are the built-in control: the serial leg does not touch the draft head,
so its stability across arms shows the host was thermally and electrically
comparable across the three measurements.

## 4. Evidence contract item 4 — draft readout cost and achieved GB/s

Isolated kernel sweep through the vendored MLX the scored worker links
(`QwenQMVCostCurveTests.sweepCompactDraftReadoutOverBits`, round-robin
interleaved across bit widths so thermal drift cannot alias onto one arm;
`reps=21`, `inner_calls_per_rep=10`). Artifacts:
`.mlxfast-private/draft-bits-sweep/e6-r1-bits-interleaved/`.

| bits | weight_bytes | s/call | ms/step | Δ vs 4-bit | achieved GB/s | crossrow |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 283,207,680 | 0.0011655750 | 1.1656 | — | 242.98 | true |
| 3 | 220,272,640 | 0.0008833333 | 0.8833 | **−24.2 %** | 249.37 | false |
| 2 | 157,337,600 | 0.0006711041 | 0.6711 | **−42.4 %** | 234.45 | false |

Per-arm spread ±2.0 % / ±1.2 % / ±0.6 %. Device `applegpu_g16s`,
`nax_available: false`. STREAM ceiling on this host:
`peak_bandwidth_bytes_per_second = 227,128,791,836.97`;
`peak_flops_per_second = 7.529e12`.

**Achieved GB/s is flat across arms (234–249, all at or slightly above the
STREAM figure).** Two consequences:

1. The readout is purely bandwidth-bound. Bytes moved is the whole story, so
   the byte savings convert to time savings essentially one-for-one.
2. The `bits == 4`-only crossrow gate does not bite. The advisor was right in
   §3 of the feedback: the draft readout is M=1 and the crossrow kernel
   `static_assert`s `M >= 2`, so dropping off crossrow costs nothing. The 4-bit
   arm reports `crossrow: true` at the dispatcher level but achieves the same
   bandwidth as the non-crossrow arms.

## 5. Evidence contract item 5 — readouts per round and per-round saving

`draft_readouts_per_round` = **5.890** (control and 2-bit), **5.864** (3-bit).
This is the measured number the brief asked for in place of the advisor's
retracted ≈ −1.1 ms/round note.

Per-round saving from bandwidth alone, using the §4 microbenchmark:

- 3-bit: 5.890 × (1.1656 − 0.8833) = **−1.663 ms/round**
- 2-bit: 5.890 × (1.1656 − 0.6711) = **−2.913 ms/round**

Over 82 rounds and 512 tokens that is **−0.266 ms/token** (3-bit) and
**−0.466 ms/token** (2-bit) from bandwidth alone.

### The 2-bit arithmetic closes, which proves the arm was live

2-bit changed neither acceptance nor round count, so its entire gain must be
bandwidth. Measured:

```
(17.98109209537506 − 17.72513699531555) × 1000 / 483 = 0.530 ms saved per readout
microbenchmark prediction: 1.1655750 − 0.6711041 = 0.4945 ms per readout
agreement ratio = 1.07
```

A silent no-op would give exactly 0. 7 % above prediction is consistent with
reduced pressure on the shared memory system in the full model. See §8 for
three further independent proofs the low-bit head was actually installed.

### The 3-bit gain decomposes cleanly

Total `decode_seconds` change is −342 ms. Of that:

- readout component: 483 × 1.1656 − 475 × 0.8833 = **−143 ms**
- residual: **−199 ms**, accounted for by one fewer round (p50 block ≈ 167–169 ms)
  plus 10 fewer rejected verify rows.

Prediction and end-to-end measurement agree. **3-bit is faster than 2-bit
despite reading 62.9 MB more per step**, purely because its acceptance gain
removed an entire round.

## 6. Evidence contract item 2 — acceptance detail

Mean acceptance is in §3. **The per-position acceptance vector is not cheaply
obtainable** and I did not fabricate one:

`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift:290` sets
`ledger: retainLedger ? audit.rows : []`, and `retainLedger` is true only for
the `mtp-verify` verb, which is untimed and would not be the arm under test.
`Sources/MLXFastCLI/main.swift:2042` emits `row_ledger` only
`if !report.ledger.isEmpty`. Getting the vector therefore requires a second,
differently-configured, untimed run per arm. That is not free, and the
substitutes below answer the same question.

**Substitute 1 — per-round draft depth sequence** (`effective_draft_lengths`,
82/81/82 entries). Control:

```
4 4 4 5 5 6 6 6 7 7 7 7 8 8 8 8 4 4 4 4 4 4 8 8 8 8 8 8 4 4 4 4 8 8 8 8 4 4 4 8
8 8 8 8 4 4 4 4 4 4 4 4 4 8 4 4 4 4 4 4 4 8 4 4 4 4 4 4 4 8 8 8 8 8 8 8 8 8 8 8
8 7
```

Divergences from control:

| arm | first divergence | differing positions | detail |
|---|---:|---:|---|
| 2-bit | round **52** | 2 | (52: 4→8), (53: 8→4) |
| 3-bit | round **59** | 5 | (59: 4→8), (61: 8→4), (67: 4→7), (68: 4→7), (80: 8→1) |

**Substitute 2 — depth histogram.** Control and 2-bit: `{4:39, 5:2, 6:3, 7:5,
8:33}`. 3-bit: `{1:1, 4:37, 5:2, 6:3, 7:6, 8:32}`.

**Substitute 3 — reference margin distribution** (arm-invariant, from the
serial golden, `row_count` 513): mean 14.0390, p10 9.75, p50 15.0,
fraction with margin < 0.5 = 0.0039, fraction < 2.0 = 0.0292. Only ~0.4 % of
positions are near-ties, which is why coarse draft logits change so few
decisions.

## 7. Evidence contract item 6 — peak memory

**The deterministic measure is head size, and it goes *down*, confirming the
advisor's §5 correction that the original warning was backwards:**

| arm | compact draft head bytes | Δ vs 4-bit |
|---|---:|---:|
| 4-bit | 283,207,680 | — |
| 3-bit | 220,272,640 | **−62,935,040 (−62.9 MB)** |
| 2-bit | 157,337,600 | **−125,870,080 (−125.9 MB)** |

`_compactDraftHead` is already a separate allocation; requantizing shrinks it.
Steady-state footprint drops.

**`ru_maxrss` is flat and does not resolve this**: 16,215,195,648 /
16,215,212,032 / 16,230,760,448 (spread 0.1 %). Two reasons: the head is under
2 % of a 16 GB footprint, and the RSS peak occurs during checkpoint load and
transform, *before* the compact draft head is lazily built during warm-up. I
report it for completeness, not as the answer.

**Transient:** Route A as implemented requantizes in blocks of 8192 rows, so it
holds at most 8192 × 5120 × 2 B = **83.9 MB** of fp16 at a time, not the
~1.0 GB an unblocked dequantize of all 503M weights would need. The advisor's
hazard was real; blocking removes it. Route B (shipping a pre-quantized head)
would have no transient at all.

## 8. Was the low-bit head actually installed? Four independent proofs

This mattered because the 2-bit arm reproduced the control's aggregate
acceptance exactly, which is exactly what a silent no-op would look like.

1. **Per-round trajectory differs.** 2-bit diverges from the control at round
   52 (§6). A no-op cannot do that. The aggregate coincidence is a swap of an
   adjacent 4/8 pair, which preserves both the total and the multiset.
2. **`verify_block_replayed_round_count` differs**: 14 (2-bit) vs 13 (control)
   vs 12 (3-bit). This counter increments at
   `QwenRuntimeMTPDriver.swift:481` when a round has rejected-tail rows not
   backed by the golden, so it is a structural, non-timing statistic.
3. **Block latency is lower**: 2-bit p50 0.16559 s vs control 0.16893 s, max
   0.21465 s vs 0.22012 s. Both directions match cheaper readouts.
4. **The saved-milliseconds arithmetic closes to 7 %** against the independent
   kernel microbenchmark (§5).

Supporting plumbing checks: the worker binary contains the env-read string
(`grep -c MLX_QWEN_MTP_DRAFT_BITS .build-worker/release/mlxfast-runtime-worker`
= 1; the same grep on `.build/release/mlxfast-swift` = 0, confirming which
binary matters); `MLX_` is on the worker env allowlist
(`sanitizedRuntimeWorkerEnvironment`, `QwenRuntimeWorker.swift:2624-2655`); and
the on-device shape/byte math is pinned by `QwenQMVCostCurveTests`, whose
measured `weight_bytes` (283,207,680 / 220,272,640 / 157,337,600) match the
brief's theoretical 283.2 / 220.3 / 157.3 MB exactly.

Two independent 4-bit controls agreed on `accepted_draft_rate` to all 16 digits
(0.8902691511387164) and on `round_count` and `declared_rows_total`, which
establishes that acceptance is deterministic per (arm, fixture) and gives the
noise figures in §0.

## 9. Open question I could not close

3-bit is numerically *closer* to 4-bit than 2-bit is, yet 3-bit is the arm that
moved acceptance.

The affine scale is `s_b = (M − m)/(2^b − 1)`. In units of the 4-bit step
`s_4`, the 2-bit grid is `{0, 5, 10, 15}` — an **exact subgrid** of the 4-bit
grid `{0..15}` — with worst-case round-trip error 2.5·s_4. The 3-bit grid is
`{0, 15/7, 30/7, …}`, **incommensurate** with the 4-bit grid, worst-case error
1.07·s_4. So by max round-trip error 3-bit should perturb the logits *less*.

Two readings, and I cannot separate them with the evidence I have:

- **(a) Coincidence.** With ~480 draft rows and only 0.4 % near-ties (§6), the
  number of decisions genuinely at risk is a handful. Both arms flip a couple
  of borderline proposals; 2-bit's flips happened to cancel.
- **(b) Structure.** Landing on an exact subgrid preserves the *ordering* of
  logits more faithfully than max round-trip error suggests, because argmax
  cares about rank, not absolute error.

Decisive follow-up: log the proposed draft token IDs per round under a
non-timed verb and diff them across arms. That separates "same decisions" from
"different decisions, same count" directly. I did not run it because the
stopping rule had already fired.

## 10. Recommendation

**Two different rankings, and which one applies depends on generalization:**

- **Measured on this fixture: 3-bit > 2-bit > 4-bit.** 3-bit wins because its
  acceptance gain deleted a whole round.
- **Fixture-independent, bandwidth only: 2-bit > 3-bit > 4-bit**
  (−0.466 vs −0.266 ms/token). This is the part that cannot fail to
  generalize, because it is pure byte traffic on a bandwidth-bound kernel.

Honest reading: the +1.92 pp acceptance is exact and reproducible *for this
fixture*, but with ~480 draft rows a swing of 10 borderline proposals is
already 2 pp. **The defensible claim is that 3-bit is acceptance-neutral and
measured non-negative here**, not that it reliably improves acceptance.
Fixture-level uncertainty cannot be reduced by re-running — acceptance is
deterministic per arm — only by a second prompt, which the local harness does
not offer.

If exactly one arm is to be carried forward, **3-bit is the safer choice**: it
captures ~60 % of the bandwidth win with a strictly smaller quantization
perturbation, so its acceptance is less exposed on the hidden prompts. If the
M5 confirms 2-bit is also acceptance-neutral across prompts, 2-bit is worth
another 0.2 ms/token.

## 11. Caveats

- **M4 Pro, not the ranked M5.** Directional only.
- **`cool_gate=disabled_ambient_floor` on all three arms** — ambient never fell
  below the 40 °C gate, so the gate was disabled rather than satisfied. The
  arms are comparable *to each other* (stable `serial_ms_per_token`, entry
  temps 41.8/45.0/45.8 °C), but none is gate-qualified.
- **One public fixture.** `program.md` warns that a result on one prompt is not
  a result. Acceptance in particular is fixture-specific.
- **Local-ratio cancellation does not apply here.** The draft head is on the
  MTP leg only, so this change appears in the ratio *and* in absolute candidate
  ms/token. Both are reported; both agree.

## 12. Suggested follow-ups (not implemented)

1. **Multi-prompt acceptance validation.** The single highest-value next step.
   The 3-bit acceptance gain and the 2-bit neutrality are both single-fixture
   observations.
2. **Route B — ship a pre-quantized 3-bit compact head** in
   `mtp-head.manifest.json`, lifting the `_draftHeadW == nil` guards at
   `Qwen35.swift:2364` and `:2403`. Removes the load-time requantize entirely.
   Explicitly out of scope for this assignment.
3. **Diff proposed draft token IDs across arms** under a non-timed verb, to
   close §9.
4. **Try 5-bit and 6-bit.** Both are instantiated. If the acceptance/bandwidth
   curve is not monotonic between 2 and 4, the interior is worth one sweep.
5. **Fix the stale comment at `Qwen36MTPBlockSession.swift:466-469`**, which
   claims `MLX_QWEN_MTP_TRACE=1` flips worker stderr forwarding. It does not
   (`runQwenMTPTimed` uses the default `forwardsWorkerStderr: false`). Not in
   scope here; it cost me time.

## 13. Reproduction

```bash
# Kernel cost curve (cheap gate, no model resident)
research/run-draft-bits-sweep.sh e6-r1-bits-interleaved

# One arm each; BITS TAG [TOKENS] [BASE_SHA]
research/run-draft-bits-arm.sh 4 e6-r1-bits4-control-b 512 b2419f41fe500bb7bb8ceea523352994a7fad0ea
research/run-draft-bits-arm.sh 3 e6-r1-bits3           512 b2419f41fe500bb7bb8ceea523352994a7fad0ea
research/run-draft-bits-arm.sh 2 e6-r1-bits2           512 b2419f41fe500bb7bb8ceea523352994a7fad0ea

# Comparison + W&B roll-up (first arm dir is the control)
python3 research/draft_bits_arms.py \
  .mlxfast-private/draft-bits/e6-r1-bits4-control-b \
  .mlxfast-private/draft-bits/e6-r1-bits3 \
  .mlxfast-private/draft-bits/e6-r1-bits2 \
  --tag draft-bits-arms-e6-r1-final --wandb
```

Arms must run sequentially: the wrapper holds a single-model run lock.

## 14. Preflight

```
senpai/validate-assignment-scope.sh b2419f41… Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift  → OK
senpai/check-editable-budget.sh    b2419f41… → source=2397991/3000000 headroom=602009
                                                growth=3341/262144 exempt=2410 files=154
```

Only `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` is a submitted
path. `research/**` and `Tests/**` are research-only and are not packaged by
Yukon.

## 15. W&B runs

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`qwen38-r1-e6-draft-head-precision`.

| run | what |
|---|---|
| `hrgew6pe` | 4-bit control-b arm |
| `ey56o2j5` | 3-bit arm |
| `ue0l9ryy` | 2-bit arm |
| `pdk6ujaq` | analysis roll-up (`draft-bits-arms-e6-r1-final`) |

## 16. Defects found and fixed while building this

1. **Stale worker binary.** `benchmark-qwen-mtp.sh` never builds the worker: it
   drives the CLI directly at `:127` and only enters `benchmark.sh` via
   `--local-cool-gate-only`, which returns before the build gate at
   `benchmark.sh:1830`. Arms were silently measuring an old binary. Fixed by
   building the worker inside the arm script.
2. **SIGPIPE false negative.** `strings … | grep -q …` under
   `set -euo pipefail` returns 141 *on match*, so the plumbing check failed
   exactly when it succeeded. Fixed with `grep -c … || true`.
3. **Reference golden deleted by the harness EXIT trap**
   (`benchmark-qwen-mtp.sh:457-459`), which is why stream identity was
   initially unverifiable. Fixed: `research/capture-cli.sh` scans argv for
   `--output <path>` and rescues it to `${keep}/NN-<verb>-output.json`.
4. **File-based head provenance is impossible in the worker.** The `mtp-timed`
   verb runs the worker under a Seatbelt profile that is `(deny file-write*)`
   with only `/dev/null` allowed (`MLXFastCLI/main.swift:2626-2639`), stderr is
   discarded (`forwardsWorkerStderr: false`, drained at
   `QwenRuntimeWorker.swift:1737`), and stdout is the request protocol. I
   removed that unobservable channel and derive head bytes from the recorded
   `MLX_QWEN_MTP_DRAFT_BITS` plus fixed shape constants instead, with the §8
   proof chain replacing it.
