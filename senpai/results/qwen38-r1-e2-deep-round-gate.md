SENPAI-RESULT: assignment=qwen38-r1-e2-deep-round-gate revision=r4 student=qwen-alphonse status=succeeded label=local-winner primary_metric=local_serial_relative_speedup direction=maximize baseline=2.0947033499 candidate=2.1626147093 test_metric=all_tokens_matched test_value=1

# Deep-round gate at width 9 — Part A row gate + Part B cap/gate sweep

## Header

| field | value |
|---|---|
| student | `qwen-alphonse` |
| branch | `qwen-alphonse/deep-round-gate-width9` |
| PR | #2 |
| assignment / revision | `qwen38-r1-e2-deep-round-gate` / `r4` |
| `BASE_SHA` | `1eacf376e3ee82578df7f47ee47f51d1382a0dbc` (the r4 marker base). The advisor branch tip moved three times during this turn — to `b9767435ad9f64509173569e62d14a658f281598` (merged into this branch), then `ea2afad4b0649cf238c7d3bf0c289330acc75d83`, then `71767c0234e91590cb2f2c187ceab6a118861a61`. All three diffs are confined to `research/` (state docs plus new `qmv`/roofline analysis scripts, probes and `pr3_anchor_reconciliation.py`); none touches `Sources/`, `Vendor/`, fixtures or the manifest, so **no measurement here is invalidated**. |
| `UPSTREAM_SHA` | `7351e62674bc600f0ca148d3a1b0604716a09db6` |
| shipped configuration | `segmentedVerifyDepthCap = 7`, `segmentedStreakGate = 3`, `sdpaWidthWallDepthCap = 4` |
| candidate commit (measured) | `8e61e775e3e49ee94fafcba439944d7951f3f480` — Run O provenance stamp reads `head=8e61e775… dirty=0`, `pre-build cap=7 gate=3`, `post-build cap=7 gate=3`. |
| result head | the commit carrying this document. `git diff 8e61e77 HEAD -- Sources/` is **empty**: the compiled candidate at the result head is byte-identical to the tree Run O measured. Everything added since is `Tests/`, `research/` and `senpai/`. |
| Run J provenance — **caveat retired** | J predates `run-gate-arm.sh` and carries no launch stamp. It is now bracketed by **two** stamped repeats of its configuration: Run O at `head=8e61e775… dirty=0` and Run P₅₁₂ at `head=43438153… dirty=0`, both asserting `cap=7 gate=3` before *and* after the Swift build. All three arms agree bit-for-bit on every structural quantity, so J demonstrably compiled the shipped constants. |
| confirmation run (r4 stop rule) | **Run P₅₁₂**, tag `runP-cap7-gate3-512-confirm`, stamped at `43438153a5ca93f4e5050229e2be2ffb01238d1f` — *the exact head of the submitted result*. Launched after submission purely to close the provenance gap; it reproduced within 0.11 % and **changed no conclusion**. |
| host | Apple M4 Pro (`Mac16,11`), 48 GB, **low-memory profile** — *not* the ranked M5 |
| toolchain | macOS 26.5.2, Xcode 26.6, Swift 6.3.3 |
| local `vector_limit` | 10 (`applegpu_g16s`, gen 16) → widths 1..9 all take the `qmv`+crossrow family locally |
| declared head | `hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaf` (unchanged) |
| Yukon frontier at write time | submission `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, score `2.9042110287045`, `sourceRef 7351e626` |
| local mode | `./benchmark-qwen-mtp.sh --local-iterate`, `MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512` |

### Submitted candidate files

Exactly one file in `benchmark.json` `editablePaths` is changed:

- `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` — gating constants + trace-only stored properties + reconciled doc comments.

### Supporting (research-only, never submitted)

`research/*.py`, `research/*.sh`, `research/*.json`, `senpai/results/*.md`.

### Head provenance (`research/fb7_head_provenance.py`)

| field | value |
|---|---|
| `tree_digest_sha256` | `05a8613e3d86456f5df9bc8ab8c53daa5d19604c08d1b0bd215ad0d599cb2863` |
| `tree_digest_bytes` | 849,407,066 (payload 849,398,784 + header 1,555) |
| tensors | 15, **all BF16** — no quantization keys present |
| `resident_is_quantized` | `false` |
| `resident_matches_pinned_fixture` | `true` |
| `resident_matches_declared_manifest` | `false` |
| `payload_over_declared_ratio` | 3.5549 (declared 238,934,093 bytes) |
| `per_forward_extra_bytes` | 610,464,691 → **2.6893 ms/draft** at 227 GB/s |

The resident head is the pinned fixture but is materialised **BF16**, not 4-bit g64 as the manifest name implies. Every "head-rebased"/"ranked" figure below subtracts `2.6893 ms × drafts` from the measured round cost to estimate what the same schedule would cost with a genuinely 4-bit-g64-resident head. The head itself was **not** changed (out of scope; `headStepCostRatio` is owned by `qwen-edward`).

### Preflight gates (re-run on the final tree)

```
senpai/validate-assignment-scope.sh 1eacf376e3ee82578df7f47ee47f51d1382a0dbc \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift
  -> assignment scope OK: 1 submitted path(s)

senpai/check-editable-budget.sh 1eacf376e3ee82578df7f47ee47f51d1382a0dbc
  -> source=2398680/3000000 headroom=601320 growth=4030/262144
     exempt=2410/2147483648 files=154
```

Growth is 4,030 bytes against a 262,144-byte allowance (1.5 %), and all of it is constants, doc-comment text and the trace-only stored properties.

The full candidate diff against the base touches two files: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (+76/−4) and `Sources/MLXFastCLI/main.swift` (+10/−1). **`Sources/MLXFastCLI` is not in `benchmark.json` `editablePaths`**, so the CLI change is research-only and is never packaged into a submission — it is the stderr forwarding seam described under *Instrumentation disclosure*, and it is double-gated on `MLX_QWEN_MTP_TRACE=1` **and** `!officialRun`.

### Scored-path reachability

`Qwen36MTPBlockSession.costModelDepth` is the function the scored worker calls each round to choose the draft count; the three constants under test (`sdpaWidthWallDepthCap`, `segmentedVerifyDepthCap`, `segmentedStreakGate`) are its only inputs beyond `fullAcceptStreak`. Reachability is proven directly: every timed run below emits one `mtp-trace:` line per round from inside that function, carrying the `streak_in=`, `cap=` and `d=` actually used, and the round counts in those traces reconcile exactly with the trusted parent's `round_count` in `04-mtp-timed.json`.

---

## Instrumentation disclosure

`Sources/MLXFastCLI/main.swift` carries a **local-only** +10/−1 change: it forwards worker stderr when `MLX_QWEN_MTP_TRACE` is set, ANDed with `!officialRun`. I confirmed this path is **not** listed in `benchmark.json` `editablePaths`, so it is never packaged into a submission and cannot affect a ranked run. It is disclosed here because it is present in the branch. All `mtp-trace:` and `mtp-row:` evidence in this report comes through that seam.

The trace-only stored properties added to `Qwen36MTPBlockSession.swift` (`traceStreakIn`, `traceWidthCap`, `traceEMAIn`) *are* in a submitted file. They are written only under `if Self.traceRounds`, which is compiled against an environment flag that the official run does not set, and they are never read by the decision path.

---

## Process note — interim reporting was not possible

`post_assignment_comment` is **not present in my tool schema** for this revision, so I could not post the Part A verdict, the deviation-from-instruction notice, or the concurrency incident to PR #2 while the work was in flight, as the assignment asked. Everything is therefore consolidated into this terminal result. This is a harness limitation, not a choice; please read the "Deviation from instruction" and "Concurrency incident" sections as if they had arrived as interim comments hours earlier.

---

## Part A — hexfloat row gate at width 9

**Verdict: width 9 is bit-exact, so Part A on its own gives no reason to lower `segmentedVerifyDepthCap`. Width is not the drift variable; terminal-block position is.** (Cap 7 *is* nevertheless what ships — see "Part A remedy" below and Part B. It is a throughput decision, not a fidelity fix.)

Seven 512-decode-token runs were gated. Each produced `compared_rows = 512` and `unmatched_positions = 0` against the serial trajectory (**3,584 rows total**). The trusted-parent harness reported `all_tokens_matched = true` in all seven.

### Per-width aggregate (Runs I + J + K + L + M + N + O)

Recomputed directly from the seven `research/analysis-run*.json` row gates by `python3 research/partA_width_aggregate.py` (committed on this branch, together with the fifteen analysis and score JSONs it reads, so the advisor can reproduce every number below without re-running the GPU). The totals sum to 3,584.

| width | rows compared | value mismatches | top-1 id mismatches | note |
|---|---|---|---|---|
| 2 | 2 | 2 | 0 | Run K terminal block |
| 4 | 16 | 6 | 3 | terminal blocks of Runs J, O (4 rows each) and M (8 rows) |
| 5 | 687 | 0 | 0 | |
| 6 | 126 | 0 | 0 | |
| 7 | 216 | 0 | 0 | |
| 8 | 1002 | 3 | 1 | Run I terminal block |
| **9** | **1535** | **4** | **2** | Run L terminal block + its bit-identical repeat in Run N |
| **total** | **3584** | **15** | **6** | every mismatch sits in a terminal block |

Per-run width-9 rows: Run I 274 (all exact), Runs J and O 0 each (both executed cap 7, so width 9 is unreachable by construction), Run K 306 (all exact), Run L 308 with 2 mismatched rows, **Run M 339 (all exact)**, Run N 308 with the same 2 mismatched rows as L.

An earlier revision of this table was built before Run O existed and additionally carried a phantom `width 3` row imported from a 256-token run and a `width 4` count that included only Run M. Both are corrected above; no conclusion changes, because every added row is bit-exact or is itself a terminal block.

**Width-9 rows outside a terminal block: 1,517 / 1,517 bit-exact.** All four width-9 deviations in the whole campaign sit inside one 9-row terminal block and its exact repeat.

Run M is the strongest single disproof of the width hypothesis. It ran with the gate fully open, so it produced *more* width-9 rows than any other run (339) — and **every one of them is bit-exact**. Its only mismatches are in a **width-4** terminal block:

```
Run M row_gate.per_width
  "4": {compared 8,   value_mm 2, id_mm 1, bit_exact false}   <- terminal block
  "5": {compared 29,  value_mm 0, id_mm 0, bit_exact true}
  "6": {compared 18,  value_mm 0, id_mm 0, bit_exact true}
  "7": {compared 46,  value_mm 0, id_mm 0, bit_exact true}
  "8": {compared 72,  value_mm 0, id_mm 0, bit_exact true}
  "9": {compared 339, value_mm 0, id_mm 0, bit_exact true}
```

The widest verify width in the run is clean and the *narrowest* one is not. Whatever causes the deviation, it is not verify width.

### Run L is the decisive positive case; Run M is the decisive negative control

Run L is the only run whose *terminal* block was itself width 9 (round 74, `key_len = 1024`). It drifted at absolute positions 1022 and 1024 — with the **identical second-place token swap** that Run I showed in its width-**8** terminal block: mtp `[6009, 31098]` vs serial `[6009, 98138]` at position 1022. Run M's serial control log independently shows `ids = 6009, 98138` at position 1022. Same position, same token pair, three different widths. Drift is a function of terminal-block position, not of verify width.

### The `key_len = 1024` boundary is the actual variable

`research/kl-boundary-runL.json`: max `key_len` reached by width 5 = 898, width 6 = 904, width 7 = 918, width 8 = 934 — none touch the 1024 boundary. Width 9 reaches 1024 in exactly one round (round 74), and that is the round that drifts.

`research/kl-boundary-runM.json` closes the argument by reversing the roles:

| width | rounds | max `key_len` | rounds at the 1024 boundary |
|---|---|---|---|
| 4 | 2 | **1024** | **round 73** |
| 5 | 7 | 907 | none |
| 6 | 3 | 535 | none |
| 7 | 7 | 914 | none |
| 8 | 12 | 930 | none |
| 9 | 42 | 1020 | none |

In Run M, width 9 gets to within 4 tokens of the boundary across 42 rounds and never deviates; the single width-4 round that actually *reaches* 1024 is exactly the round that does. Across all seven runs the rule holds without exception: **a block deviates if and only if it is the block that closes the 1024-token window**, and never otherwise.

`research/kl-boundary-runN.json` reproduces Run L exactly — width 5 max 898, width 6 904, width 7 918, width 8 934, width 9 max 1024 with a single boundary round 74 — confirming that the boundary structure, like the schedule, is deterministic.

### The deviation is deterministic, not a floating-point race

Run N is the strongest statement available about the nature of the drift. It re-ran Run L's configuration and reproduced **the same two mismatched positions, in the same round, with byte-identical hexfloats and the same token ids**:

```
Run N row_gate.mismatch_samples
  pos 1022, width 9, round 74: mtp ids [6009, 31098]  serial ids [6009, 98138]
                               mtp     [0x1.1ep+5, 0x1.2ap+4]
                               serial  [0x1.1cp+5, 0x1.28p+4]
  pos 1024, width 9, round 74: mtp ids [286, 1658]    serial ids [286, 1658]
                               mtp     [0x1.2ap+5, 0x1.4ap+4]
                               serial  [0x1.2ap+5, 0x1.4ep+4]
```

This rules out nondeterministic reduction order, thermal effects, and scheduling races as causes. The terminal block takes a genuinely different, reproducible numerical path — consistent with a boundary-sized dispatch or padding difference at `key_len = 1024` — and it lands 2–4 ULP away every single time.

### Deviation magnitudes (Run L, hexfloat)

| position | rank | candidate | serial | abs | rel |
|---|---|---|---|---|---|
| 1022 | top-1 | `0x1.1ep+5` = 35.75 | `0x1.1cp+5` = 35.5 | 0.25 | 0.00704 |
| 1022 | 2nd | `0x1.2ap+4` = 18.625 | `0x1.28p+4` = 18.5 | 0.125 | 0.00676 |
| 1024 | 2nd | `0x1.4ap+4` = 20.625 | `0x1.4ep+4` = 20.875 | 0.25 | 0.011976 |

**Max absolute deviation 0.25; max relative deviation 0.011976** — roughly 2–4 ULP at bf16 logit magnitude. Widths 6, 7 and 8 in non-terminal blocks show **zero** deviation of any size, so there is no width-graded trend to report: the deviation is 0 everywhere except the terminal block, at any width.

### Top-2 ordering and identity

- **Top-1 token identity is preserved in 3,072 / 3,072 rows.** Every row counted as an "id mismatch" above is a disagreement in the *second*-place token only.
- Top-2 *ordering* is preserved wherever the top-2 identities agree. The five id mismatches are cases where the second-place slot is a near-tie and the reordering swaps in a different runner-up.
- Answering the assignment's question directly, at widths 6, 7, 8 and 9 in non-terminal blocks: **max absolute logit deviation 0, max relative deviation 0, top-2 ordering and identity fully preserved.** The only non-zero deviations anywhere in the campaign are the terminal-block rows tabulated above.

### Why this is not a ranked failure

The trusted parent compares **top-1 only**. In `QwenRuntimeMTPDriver.swift`, the emitted-token gate (≈ lines 211–219) is a zero-tolerance comparison of the emitted token id, and the rejected-tail replay (≈ lines 483–510) compares `.first` only, admitting a disagreement solely when `margin < tolerance.referenceMargin`. A second-place swap at a near-tie is exactly the case that check tolerates. Consistently, all seven runs report:

```
all_tokens_matched            = true
residual_divergence_count     = 0
max_rejected_tail_logit_delta = 0
parity_all_ok                 = true
non_drafting_round_count      = 0
uses_pinned_mtp_head          = true
declared_rows_total == reference_checked_row_total   (all runs)
```

The ranked window is 512 seed + 512 decode, so **every** ranked prompt terminates at `key_len = 1024`; this puts roughly 2–3 rows per prompt (≈ 16–24 rows over 8 prompts) into the drift zone. Those rows are still top-1 exact, so the ranked gate should pass, but the advisor should know the exposure exists.

### ★ The decisive Part A cross-tab — the mismatch tracks *position*, never *width*

This is the single strongest piece of Part A evidence and it only became available once seven arms existed. Each arm terminates its 1,024-position window with whatever width its schedule happened to choose. If width were the drift variable, the mismatching width would be constant across arms. It is not — **it is exactly the terminal round's width, every time**:

| run | cap | gate | per-width `compared / value-mismatch / id-mismatch / bit-exact` | mismatch positions | mismatch round | mismatch width |
|---|--:|--:|---|---|--:|--:|
| I | 8 | 3 | w5 165/0/0/T · w6 12/0/0/T · w7 21/0/0/T · **w8 40/3/1/F** · w9 274/0/0/**T** | 1022, 1023, 1024 | 82 (last) | 8 |
| J | 7 | 3 | **w4 4/2/1/F** · w5 119/0/0/T · w6 12/0/0/T · w7 21/0/0/T · w8 356/0/0/T | 1022, 1024 | 81 (last) | 4 |
| O | 7 | 3 | **w4 4/2/1/F** · w5 119/0/0/T · w6 12/0/0/T · w7 21/0/0/T · w8 356/0/0/T | 1022, 1024 | 81 (last) | 4 |
| K | 8 | 2 | **w2 2/2/0/F** · w5 117/0/0/T · w6 18/0/0/T · w7 37/0/0/T · w8 32/0/0/T · w9 306/0/0/**T** | 1023, 1024 | 79 (last) | 2 |
| L | 8 | 1 | w5 69/0/0/T · w6 27/0/0/T · w7 35/0/0/T · w8 73/0/0/T · **w9 308/2/1/F** | 1022, 1024 | 74 (last) | 9 |
| N | 8 | 1 | w5 69/0/0/T · w6 27/0/0/T · w7 35/0/0/T · w8 73/0/0/T · **w9 308/2/1/F** | 1022, 1024 | 74 (last) | 9 |
| M | 8 | 0 | **w4 8/2/1/F** · w5 29/0/0/T · w6 18/0/0/T · w7 46/0/0/T · w8 72/0/0/T · w9 339/0/0/**T** | 1022, 1024 | 73 (last) | 4 |

Four different terminal widths (2, 4, 8, 9) across seven arms, and in every arm the *only* imperfect width is the terminal one. Width 9 is compared **1,535 times** across I, K, L, M and N (Runs J and O run at cap 7 and never reach it); it is bit-exact in I (274), K (306) and M (339) — **919 of 919 rows** in the three arms where width 9 was *not* terminal — and imperfect only in L and N, the two arms where width 9 *was* the terminal width, and there only inside the terminal block itself.

**Part A's question is therefore answered `bit-exact = true` for width 9.** `compared_rows = 512`, `unmatched_positions = 0`, `all_tokens_matched = true` and `residual_divergence_count = 0` in all seven arms; the residue is 2–3 second-place bf16 hexfloat ulps in the last three rows of the window. Width 4 does not even enter the `AttentionUtils` segmented split (that path is gated `6 <= qL <= 9`), so arms J, O and M exonerate the deep-width path directly.

### ★ Part A remedy — the cap *was* lowered to 7, and it wins for an unrelated reason

The assignment's Part A branch was: *not bit-exact → lower `segmentedVerifyDepthCap` to 7*. My honest position is that **the premise for that branch was not met** — width 9 is exact — but I shipped cap 7 anyway, because Part B measures it as the best arm by a wide margin.

I want to be explicit that these are two different justifications and only the second one is load-bearing:

- Cap 7 does **not** fix the terminal-block drift. It cannot: the drift is at `key_len = 1024`, which every run reaches regardless of cap; lowering the cap only changes *which* width happens to close the window (Runs J and O move it from w8 to w4, and it is still there).
- Cap 7 **is** the fastest configuration measured, at **−3.085 % absolute candidate s/token** against the Run I control. That is the reason it is shipped.

So the shipped tree satisfies the assignment's Part-A-fails clause by coincidence, and satisfies Part B's bar on merit. The advisor should read the cap-7 decision as a throughput result, not as a fidelity remedy.

---
## Part B — cap × gate sweep

### Reproduction

Every arm was produced by the same committed runner, which stamps provenance, asserts the constants before *and* after the Swift build, and only then runs the benchmark:

```bash
# TAG  EXPECT_CAP  EXPECT_GATE  TOKENS  MODE
research/run-gate-arm.sh runI-base-cap8-512      8 3 512 --local-iterate   # control
research/run-gate-arm.sh runK-gate2-cap8-512     8 2 512 --local-iterate
research/run-gate-arm.sh runL-gate1-cap8-512     8 1 512 --local-iterate
research/run-gate-arm.sh runM-gate0-cap8-512     8 0 512 --local-iterate
research/run-gate-arm.sh runN-gate1-cap8-512-confirm 8 1 512 --local-iterate
research/run-gate-arm.sh runO-cap7-gate3-512     7 3 512 --local-iterate   # winner
research/run-gate-arm.sh runP-cap7-gate3-localsubmit-128 7 3 128 --local-submit
research/run-gate-arm.sh runP-cap7-gate3-512-confirm 7 3 512 --local-iterate  # third repeat, stamped at the submitted head
```

Runs I, J and J₂ predate the runner and were driven by the same environment by hand; that is precisely why J₂'s assigned and executed configurations diverged, why Run O exists to reproduce J under a stamped build, and why Run P₅₁₂ then repeated it a third time stamped at the submitted head. All sweep arms are `--local-iterate` at 512 decode tokens (`MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512`), one public fixture, M4 Pro, 40 °C cool gate honoured on every launch. Run I is a **fresh unchanged-base control measured on this host in this session** — not a historical number.

### Headline — the winner is cap 7 / gate 3, and the *cap* is the dominant lever

| run | cap | gate | local ratio | Δ ratio vs I | s/token | **Δ s/token** | W&B |
|---|--:|--:|---|---|---|---|---|
| **I** | 8 | 3 | 2.0947033499 | — (control) | 0.03510386 | — | `txwiiulo` |
| **J** | **7** | **3** | **2.1636873696** | **+3.293 %** | **0.03403967** | **−3.032 %** | `iwy987kn` |
| **O** | **7** | **3** | **2.1615420490** | **+3.191 %** | **0.03400226** | **−3.138 %** | `p6yyq9ep` |
| **P₅₁₂** | **7** | **3** | **2.1623244386** | **+3.228 %** | **0.03403510** | **−3.045 %** | `bp9kysyu` |
| J₂ | 7 | 2 | 2.1243836568 | +1.417 % | 0.03452479 | −1.650 % | `ixu99guw` |
| K | 8 | 2 | 2.1019606601 | +0.346 % | 0.03502188 | −0.234 % | `sc05c6tg` |
| L | 8 | 1 | 2.1288141130 | +1.628 % | 0.03456269 | −1.542 % | `lluppgt1` |
| N | 8 | 1 (repeat of L) | 2.1311965111 | +1.742 % | 0.03458426 | −1.480 % | `y2uqpe8a` |
| M | 8 | 0 | 2.0600336024 | −1.655 % | 0.03567602 | **+1.630 %** | `l4zj9qxi` |

Serial-leg s/token per run: I 0.07353218, J 0.07365120, O 0.07349732, P₅₁₂ 0.07359493, J₂ 0.07334391, K 0.07361460, L 0.07357754, N 0.07370585, M 0.07349379 — **spread 0.4935 %**, so the ratio column is not being moved by an unstable denominator. Restricted to the four arms that matter for the headline comparison (I, J, O, P₅₁₂) the serial spread is **0.209 %**.

**Winner, all three repeats:** cap 7 / gate 3, mean s/token **0.03402568** (mean ratio **2.1625179524**), **−3.071 %** against the control, **+3.237 %** on the ratio.

The published primary metric is left at the two-repeat mean **2.1626147093**, because that is the number the timing evidence was submitted under. The three-repeat mean is **2.1625179524**, a difference of **−0.0045 %** — an order of magnitude below repeat noise, and not worth restating the whole result for. Either figure supports the same decision.

#### Run-to-run noise, measured three times

The schedule is a deterministic function of the configuration, so a repeat is a pure timing measurement with the decision path pinned. Three independent repeats of the winner plus one of a gate arm:

| set | config | s/token values | mean | **spread** |
|---|---|---|---|---|
| **J ↔ O ↔ P₅₁₂** | **cap 7 / gate 3** | 0.03403967 / 0.03400226 / 0.03403510 | **0.03402568** | **0.1100 %** |
| L ↔ N | cap 8 / gate 1 | 0.03456269 / 0.03458426 | 0.03457347 | **0.0624 %** |

Adding a third repeat did **not** widen the band: the three-arm spread (0.1100 %) is the same as the original J↔O pair (0.1099 %), so 0.11 % is a stable estimate of this host's timing noise rather than an artefact of two samples. **The −3.071 % effect is 28× the repeat noise.**

J, O and P₅₁₂ are bit-identical in every structural quantity — `accepted_draft_rate` 0.9189765458422174, `effective_mean_draft_len` 5.790123456790123, 81 rounds, depth histogram `{3:1, 4:29, 5:2, 6:3, 7:46}`, cap histogram `{4:29, 7:52}`, 431 accepted / 38 rejected, `declared_rows_total` = `reference_checked_row_total` = 550, 10 replayed verify blocks, and the *same two* terminal-round mismatch hexfloats — to sixteen digits.

#### The 2 × 2: cap and gate interact, with a sign reversal

| | gate 3 | gate 2 | effect of relaxing gate 3 → 2 |
|---|---|---|---|
| **cap 8** | 0.03510386 (I) | 0.03502188 (K) | **−0.234 %** (helps slightly) |
| **cap 7** | **0.03402096** (J, O) | 0.03452479 (J₂) | **+1.481 %** (hurts) |
| effect of lowering cap 8 → 7 | **−3.085 %** | −1.419 % | |

The cap main effect at gate 3 is **−3.085 %**; the largest gate effect anywhere in the sweep is −1.5 %. **`segmentedVerifyDepthCap` is the dominant lever and it is very far from inert** — this directly contradicts the r4 feedback's retraction #3. The interaction is real: relaxing the gate is mildly good at cap 8 and clearly bad at cap 7, because at cap 7 the gate is the only thing still holding back rounds that were going to reject anyway.

**Against the assignment's bar:** the expected result was mean chosen depth up, rounds-per-token down, and absolute candidate s/token down by **≥ 2 %**. Cap 7 / gate 3 delivers **−3.085 %**, clearing the bar with 1.5× margin. The gate-only arms do not: the best of those is −1.511 % (L/N mean).

### Repeatability — the schedule is a deterministic function of the configuration

Two arms were deliberately run twice: **L↔N** (cap 8 / gate 1) and **J↔O** (cap 7 / gate 3, the winner). Both repeats are structurally *bit-identical* to their originals, so each pair isolates pure timing noise with the decision path frozen.

Run N re-ran Run L's exact configuration on the committed tree (`head=03d43894…`, `dirty=0`, pre-build and post-build asserts both `cap=8 gate=1`). Every structural quantity came back **identical**, not merely close:

| quantity | Run L | Run N |
|---|---|---|
| depth histogram | `{4:18, 5:5, 6:5, 7:10, 8:36}` | `{4:18, 5:5, 6:5, 7:10, 8:36}` |
| rounds | 74 | 74 |
| accepted / rejected drafts | 438 / 47 | 438 / 47 |
| `effective_mean_draft_len` | 6.5540540540540544 | 6.5540540540540544 |
| `accepted_draft_rate` | 0.90309278350515465 | 0.90309278350515465 |
| `declared_rows_total` | 559 | 559 |
| `verify_block_replayed_round_count` | 12 | 12 |
| per-position acceptance profile | identical | identical |
| row-gate mismatches | pos 1022 + 1024, round 74 | pos 1022 + 1024, round 74, **same hexfloats** |
| gate counterfactual, KL boundary | identical | identical |

Run O did the same for Run J (cap 7 / gate 3) and reproduced it exactly: depth histogram `{3:1, 4:29, 5:2, 6:3, 7:46}`, 81 rounds, 431 accepted / 38 rejected drafts, `effective_mean_draft_len` 5.790123456790123, `accepted_draft_rate` 0.9189765458422174, and the *same* terminal-round mismatch hexfloats at positions 1022 and 1024.

#### ★ Run P₅₁₂ — the third repeat, stamped at the submitted head

The winner was run a **third** time, `research/run-gate-arm.sh runP-cap7-gate3-512-confirm 7 3 512 --local-iterate`, stamped `head=43438153a5ca93f4e5050229e2be2ffb01238d1f dirty=0` with pre-build and post-build asserts both `cap=7 gate=3`. That head is the exact commit the r4 result was submitted at, which is the point of the run.

Run P₅₁₂ is **byte-identical to Run O in every non-timing output**, verified by direct comparison of the committed artifacts rather than by eye:

| quantity | Run O | Run P₅₁₂ |
|---|---|---|
| depth histogram | `{3:1, 4:29, 5:2, 6:3, 7:46}` | identical |
| cap histogram | `{4:29, 7:52}` | identical |
| rounds | 81 | 81 |
| accepted / rejected drafts | 431 / 38 | 431 / 38 |
| `declared_rows_total` / `reference_checked_row_total` | 550 / 550 | 550 / 550 |
| `verify_block_replayed_round_count` | 10 | 10 |
| `accepted_draft_rate` | 0.9189765458422174 | identical |
| `effective_mean_draft_len` | 5.790123456790123 | identical |
| row-gate `per_width` table | w4 4/2/1, w5 119/0/0, w6 12/0/0, w7 21/0/0, w8 356/0/0 | **`==` identical** |
| row-gate mismatch samples | pos 1022 + 1024, round 81 | **`==` identical**, same token ids *and* same hexfloats |
| per-position acceptance profile | `0.951 0.974 0.960 0.972 0.980 1.000 0.955` | identical |
| decode seconds | 17.4092 | 17.4260 |

The `per_width` and `mismatch_samples` rows were compared with a structural equality test on the two `analysis-run*.json` artifacts, not by inspection; both returned `True`. Only the timings differ.

*Artifact-hygiene note.* My first pass wrote this run's analysis to `research/analysis-runP.json`, which was already occupied by the 128-token `--local-submit` fidelity analysis (both runs use the tag prefix `runP`). The committed 128-token artifact was silently overwritten in the worktree. It was recovered from git and the confirmation run now lives at `research/analysis-runP-512-confirm.json`; both are committed side by side. Nothing measured was lost, but the near-miss is worth recording — the two `runP` tags should have been disambiguated at launch.

**What the third repeat bought.** Not precision — the spread did not move. It bought **provenance**: Run J was the highest-scoring arm and the only unstamped one, so the headline rested partly on inferring J's build from its outputs. There are now two stamped arms reproducing J bit-for-bit, one of them at the submitted head itself, which converts that inference into direct evidence. The caveat in the header is retired on this basis.

**Correction to my earlier claim.** The previous revision of this section asserted that "repeating an arm a third time would buy nothing; only a different host or a different prompt can move the conclusion." That was right about *precision* and wrong about *provenance*, and I should not have written it as a blanket statement. The precision half held exactly — the third sample reproduced to 0.11 % and shifted the mean by −0.0045 % — but a repeat is also the only way to attach a launch stamp to an arm measured before the stamping runner existed, and that was a real open weakness in my own report header.

**The candidate schedule is a deterministic function of the configuration.** Nothing in the gate or cap path depends on wall-clock time, thermal state, or scheduling order. That makes each repeat a clean measurement of *timing* noise with the decision path held exactly fixed:

| pair | config | s/token A | s/token B | spread | ratio A | ratio B | spread |
|---|---|---|---|---|---|---|---|
| L↔N | cap 8 / gate 1 | 0.03456269 | 0.03458426 | **0.0624 %** | 2.1288141130 | 2.1311965111 | 0.112 % |
| **J↔O** | **cap 7 / gate 3** | 0.03403967 | 0.03400226 | **0.1099 %** | 2.1636873696 | 2.1615420490 | 0.099 % |

**Run-to-run noise is ≤ 0.11 % in s/token. The winning effect is −3.085 %, i.e. 28× the largest observed repeat spread.** The eight serial control legs — which never change — spread by 0.4935 % end to end, so even the *drift of the measurement frame itself* is six times smaller than the effect.

The corollary is that the remaining error in the sweep is not sampling noise, it is host and prompt transfer. Repeating an arm a third time would buy nothing; only a different host or a different prompt can move the conclusion.

### ★ Correction — the advisor's designated stop signal did not fire

The assignment named accepted-tokens-per-round "the stop signal". Recomputed consistently across all arms (emitted/round = 512 / rounds; accepted-drafts/round = `accepted_draft_total` / rounds):

| run | cap | gate | rounds | emitted/round | accepted-drafts/round | rounds/token | mean depth | mean round ms | ratio |
|---|---|---|---|---|---|---|---|---|---|
| I | 8 | 3 | 82 | 6.2439 | 5.2439 | 0.16016 | 5.890 | 170.13 | 2.0947 |
| **J** | **7** | **3** | 81 | 6.3210 | 5.3210 | 0.15820 | 5.790 | **165.29** | **2.1637** |
| **O** | **7** | **3** | 81 | 6.3210 | 5.3210 | 0.15820 | 5.790 | **165.16** | **2.1615** |
| K | 8 | 2 | 79 | 6.4810 | 5.4810 | 0.15430 | 6.127 | 175.93 | 2.1020 |
| L | 8 | 1 | 74 | 6.9189 | 5.9189 | 0.14453 | 6.554 | 184.74 | 2.1288 |
| N | 8 | 1 | 74 | 6.9189 | 5.9189 | 0.14453 | 6.554 | 184.82 | 2.1312 |
| **M** | 8 | **0** | **73** | **7.0137** | **6.0137** | **0.14258** | **7.000** | 195.12 | **2.0600** |

Within the cap-8 family, accepted-tokens-per-round rises **monotonically all the way through gate 0**, rounds-per-token falls monotonically, and mean depth rises monotonically — yet the score peaks at gate 1 and then falls. **The worst arm, M, maximises all three of the assignment's named indicators.** The signal is not merely uninformative here; over the cap-8 arms it is *anti-correlated* with speed.

The winner makes the point from the other side. Cap 7 has a **lower** mean depth than every cap-8 arm except I (5.790 vs 6.127 / 6.554 / 7.000) and a middling accepted-tokens-per-round (5.3210). On the assignment's stop signal it looks like the second-weakest arm in the sweep. It is the fastest by 3 %.

What actually separates the arms is **mean round latency at constant emitted tokens**. Every arm emits exactly 512 tokens, so `s/token = mean_round_ms × rounds / 512`; cap 7 is the only arm that raises emitted-per-round above the control *and* lowers milliseconds-per-round at the same time. Equivalently, in **µs per emitted token**, driven by the reject count:

| run | cap | gate | µs/token (`mean_round_us × rounds / 512`) | rejected drafts | accepted drafts |
|---|---|---|---|---|---|
| I | 8 | 3 | 27 246.7 | 53 | 430 |
| **J** | **7** | **3** | **26 149.2** | **38** | 431 |
| **O** | **7** | **3** | **26 129.0** | **38** | 431 |
| K | 8 | 2 | 27 145.9 | 51 | 433 |
| L | 8 | 1 | 26 700.7 | 47 | 438 |
| N | 8 | 1 | 26 712.3 | 47 | 438 |
| M | 8 | 0 | 27 819.4 | **72** (+89 % vs J/O) | 439 |

Cap 7 is the minimum of this column at **26 129–26 149 µs/token, −4.1 % of in-round decode time versus the control**. It ranks the arms in exactly the order the score does.

Two mechanisms, both about wasted width-9 work:

- **Gate 0 (M)** buys one extra accepted draft over the whole run and pays 25 extra rejections for it. An ungated schedule sends every round to width 9 including the rounds that were about to reject, and a rejected width-9 round costs a full ~218 ms block plus rollback for zero emitted tokens.
- **Cap 7 (J/O)** removes width 9 *entirely*. Rejections fall from 53 to 38 — a 28 % cut — because the marginal 9th draft was the one being rejected, and width 9 is also the only width that needs a third weight stream (`ceil(9/4) = 3` vs `ceil(8/4) = 2`). The arm gives up one round's worth of depth and gets back both the third stream and the rejection tail.

The stop-signal recommendation therefore stands and strengthens: **retire accepted-tokens-per-round for depth work and use µs per emitted token plus the reject count.** Those two rank all seven arms correctly; the named signal ranks the worst arm first.

### Fidelity and accounting (all arms)

```
all_tokens_matched            = true    (I, J, O, J2, K, L, M, N, P)
residual_divergence_count     = 0
max_rejected_tail_logit_delta = 0
parity_all_ok                 = true
non_drafting_round_count      = 0
uses_pinned_mtp_head          = true
public_drift_tripwire_passed  = true
head_provenance_sha256        = 05a8613e3d86456f5df9bc8ab8c53daa5d19604c08d1b0bd215ad0d599cb2863
```

Per-arm row accounting (`declared_rows_total == reference_checked_row_total`, computed as
`sum over rounds of (depth + 1)`):

| arm | cap | gate | mode | seed | final cache offset | declared rows |
|---|--:|--:|---|--:|--:|--:|
| I | 8 | 3 | iterate 512 | 512 | 1024 | 565 |
| **J** | **7** | **3** | iterate 512 | 512 | 1024 | **550** |
| **O** | **7** | **3** | iterate 512 | 512 | 1024 | **550** |
| J₂ | 7 | 2 | iterate 512 | 512 | 1024 | 561 |
| K | 8 | 2 | iterate 512 | 512 | 1024 | 563 |
| L | 8 | 1 | iterate 512 | 512 | 1024 | 559 |
| N | 8 | 1 | iterate 512 | 512 | 1024 | 559 |
| M | 8 | 0 | iterate 512 | 512 | 1024 | 584 |
| **P** | **7** | **3** | **submit 128** | 512 | 640 | **130** |

Cap 7 also declares the fewest rows of any 512-token arm (550 vs 565 for the control, 584 for gate 0) — it does 2.7 % less target verification work than the control while emitting the same 512 tokens.

### Run P — `--local-submit` fidelity check on the winner

The winning configuration was re-run under `--local-submit` (`research/run-gate-arm.sh runP-cap7-gate3-localsubmit-128 7 3 128 --local-submit`, stamped `head=44e7a8d6… dirty=0`, pre- and post-build asserts both `cap=7 gate=3`):

```
score                         = 1.7150844246386372
passed                        = true
all_tokens_matched            = true
residual_divergence_count     = 0
public_drift_tripwire_passed  = true
uses_pinned_mtp_head          = true
serial_seconds_per_token      = 0.09674164094030857
mtp_seconds_per_token         = 0.05640634335577488
effective_mean_draft_len      = 5.842105263157895
accepted_draft_rate           = 0.990990990990991
rounds = 19, accepted 110, rejected 1, depth hist {1:1, 4:3, 5:2, 6:3, 7:10}
row gate: compared 128, value mismatches 0, id mismatches 0
  per width -> 2: 1 rows, 5: 15, 6: 12, 7: 21, 8: 79 — every width bit-exact
```

Two things matter here.

**First, the row gate is completely clean, including width 8 across 79 rows.** The 512-token arms all showed a 1–2 ulp top-2 drift confined to the terminal round; Run P's decode window ends at position **640**, not 1024, and there is no drift at all. That is the fourth independent confirmation that the residual is a function of decode *position* near `key_len = 1024`, not of draft width.

**Second, the 128-token submit window is too short to show the cap-7 win**, and I am reporting that plainly rather than quoting it as supporting evidence. Against the only prior `--local-submit` artifact on this branch (Run H, revision r1, old base `e20268e9`, cap 8 / gate 3):

| | Run H (cap 8, r1 base) | Run P (cap 7, this base) | delta |
|---|---|---|---|
| score | 1.7177867866350764 | 1.7150844246386372 | −0.157 % |
| serial s/token | 0.09683839138597250 | 0.09674164094030857 | −0.100 % |
| MTP s/token | 0.05637392960488796 | 0.05640634335577488 | **+0.058 %** |
| all tokens matched | true | true | — |
| residual divergences | 0 | 0 | — |

The MTP difference is +0.058 %, i.e. *half* the J↔O repeat noise and across two different bases — it is a null. The reason is structural, not a contradiction: at 128 tokens the run is 19 rounds long and the ~4.02 s fixed prologue dominates, so the per-round saving that cap 7 delivers has almost no window to accumulate. `--local-submit` is a fidelity gate here, and it passes; the speed claim rests entirely on the matched 512-token `--local-iterate` arms.

### Depth histograms

Width = depth + 1; weight streams = `ceil(width / 4)`.

| depth | width | streams | I (c8 g3) | **J (c7 g3)** | **O (c7 g3)** | J₂ (c7 g2) | K (c8 g2) | L (c8 g1) | N (c8 g1) | M (c8 g0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 2 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 3 | 4 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 2 |
| 4 | 5 | 2 | 39 | 29 | 29 | 27 | 29 | 18 | 18 | 7 |
| 5 | 6 | 2 | 2 | 2 | 2 | 3 | 3 | 5 | 5 | 3 |
| 6 | 7 | 2 | 3 | 3 | 3 | 3 | 6 | 5 | 5 | 7 |
| 7 | 8 | 2 | 5 | **46** | **46** | 48 | 4 | 10 | 10 | 12 |
| 8 | 9 | **3** | 33 | **0** | **0** | 0 | 36 | 36 | 36 | **42** |

Within the cap-8 family the shape is bimodal: the schedule is almost always either at the shallow wall (depth 4) or at the cap (depth 8). Relaxing the gate moves mass from the first mode to the second (39 → 29 → 18 → 7 at depth 4; 33 → 36 → 36 → 42 at depth 8).

The cap-7 arms have a different signature, and it is the reason they win. Depth 8 is structurally unreachable, so all 33 of the control's width-9 rounds collapse onto width 8, which needs only **two** weight streams instead of three. The deep mode is *preserved* — 46 deep rounds versus the control's 5 + 33 = 38 — while the third-stream dispatch disappears from the run entirely. Cap 7 is not "less drafting"; it is the same amount of drafting done in a cheaper dispatch class.

### Step 0 — does the gate actually bind? (`research/cap_binding.py`)

Both caps are read from the per-round `cap=` field that `costModelDepth` emits, so this is the configuration the round *executed*, not one inferred from the git log. Reproduce with `python3 research/cap_binding.py --trace research/trace-runI-base-cap8-512.log --label runI --gate 3 --deep-cap 8`; artifacts are `research/cap-binding-run{I,J,J2,K,L,M,N,O}.json`.

| arm | gate | cap | rounds | rounds gate-closed (streak < gate, ceiling 4) | frac | rounds at deep cap | frac | mean depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **I (control)** | 3 | 8 | 82 | **39** | **0.4756** | **33** | **0.4024** | 5.890 |
| J | 3 | 7 | 81 | 29 | 0.3580 | 46 | 0.5679 | 5.790 |
| O (J repeat) | 3 | 7 | 81 | 29 | 0.3580 | 46 | 0.5679 | 5.790 |
| J₂ | 2 | 7 | 82 | 27 | 0.3293 | 48 | 0.5854 | 5.841 |
| K | 2 | 8 | 79 | 29 | 0.3671 | 36 | 0.4557 | 6.127 |
| L | 1 | 8 | 74 | 18 | 0.2432 | 36 | 0.4865 | 6.554 |
| N (L repeat) | 1 | 8 | 74 | 18 | 0.2432 | 36 | 0.4865 | 6.554 |
| M | 0 | 8 | 73 | 0 | 0.0000 | 42 | 0.5753 | 7.000 |

Three answers, all from the control:

1. **The cap binds.** 33 of 82 rounds (40.24 %) realised depth 8. The r3 prediction was that this would come back at ~0, which would have closed the experiment; it did not, and the experiment was live. The cost test in `costModelDepth` does *not* stop the loop at ~5 on this fixture — the 5.4 effective draft length that motivated that prediction is an average over a **bimodal** distribution (39 rounds pinned at depth 4, 33 at depth 8), not a mode.
2. **The gate binds hard.** 39 of 82 rounds (**47.56 %**) ran with `fullAcceptStreak < 3`, i.e. with a real ceiling of 4 rather than 8. That lands inside the r4 prediction band of 29–87 %, between the simulated "easy" (29.0 %) and "mid" (66.5 %) prose regimes.
3. **The gate lever moves exactly as designed — it is just not the biggest one.** At cap 8 the closed fraction falls monotonically 0.4756 → 0.3671 → 0.2432 → 0.0000 as the gate goes 3 → 2 → 1 → 0, and mean depth rises monotonically 5.890 → 6.127 → 6.554 → 7.000. The r1/r4 hypothesis is therefore mechanically confirmed: the streak really does withhold deep rounds, and relaxing it really does release them. It also *pays* at cap 8 — gate 2 −0.234 %, gate 1 −1.511 % — until gate 0 removes the gate entirely and reverses to +1.630 %. But the cap effect at gate 3 is **−3.085 %, roughly twice the best gate effect anywhere in the sweep**, and it is obtained *without* releasing any additional deep rounds. The gate was the right thing to suspect; it was not the largest thing wrong.

J↔O and L↔N are bit-identical in every column, which is the same determinism the headline rests on.

*Artifact-naming correction:* the previously committed `research/cap-binding-runJ.json` was generated from `trace-runJ-gate2-512.log` — the **J₂** arm — under the filename `runJ`. The eight artifacts above are regenerated so each filename matches the arm it describes, and all eight now reconcile cell-for-cell with the depth-histogram table. No number in this report changed: the histogram table already carried the correct J and J₂ columns, so the defect was in the artifact filename, not in the analysis.

### Block latency after the first block (fb7)

| run | rounds | p50 after first | p50 depth/idx | max after first | max depth/idx | max idx ÷ rounds | max round was a rejection | local ratio | ranked (head-rebased) | shift |
|---|---|---|---|---|---|---|---|---|---|---|
| I | 82 | 168.405 ms | d6 / 6 | 220.227 ms | d8 / 62 | 0.7561 | yes | 1.3077 | 1.3050 | −0.0027 |
| J | 82 | 189.799 ms | d7 / 10 | 192.265 ms | d7 / 28 | 0.3415 | yes | 1.0130 | 1.0144 | +0.0014 |
| K | 79 | 189.725 ms | d7 / 11 | 220.172 ms | d8 / 57 | 0.7215 | yes | 1.1605 | 1.1624 | +0.0019 |
| L | 74 | 191.290 ms | d7 / 52 | 226.734 ms | d8 / 74 | 1.0000 | no | 1.1853 | 1.1899 | +0.0046 |
| M | 73 | 217.394 ms | d8 / 15 | 221.644 ms | d8 / 32 | 0.4384 | no | 1.0195 | 1.0217 | +0.0021 |
| **N** | 74 | 191.087 ms | d7 / 54 | 227.225 ms | d8 / 74 | 1.0000 | no | 1.1891 | 1.1942 | +0.0051 |

The head-rebased column applies the measured head delta of **2.689271766519824 ms per draft** at 227 GB/s (the resident head is 849,398,784 payload bytes of BF16 against 238,934,093 declared, i.e. 610,464,691 extra bytes per forward). Rebasing **raises** the max/p50 ratio in five of the six runs in that corpus (I–N), so the ranked host would see a slightly *worse* tail dispersion than the local host, not a better one. The shift is small (≤ 0.006) in every case.

Run N reproduces Run L's tail structure: max block at the terminal round (index 74 of 74), not a rejection, depth 8; p50 at depth 7. The p50 and max block times agree with L to within 0.2 % (191.087 vs 191.290 ms; 227.225 vs 226.734 ms), which is the same tight repeatability the headline shows.

Run L's max block is its own terminal round (index 74 of 74, fraction 1.0000) and was **not** a rejection — it is the round that closes the 1024 window, which ties the latency tail back to the Part A finding rather than to gate policy. Run M's max/p50 ratio collapses to 1.0195 because with no gate almost every round is depth 8, so there is no shallow population left to pull the median down.

Cost curve (measured ms per **accepted** token, local / head-rebased ranked):

| depth | local | ranked |
|---|---|---|
| 2 | 26.567 | 24.774 |
| 4 | 25.280 | 23.129 |
| 5 | 24.417 | 22.176 |
| 6 | 24.043 | 21.738 |
| **7** | **23.712** | **21.359** |
| 8 | 24.156 | 21.765 |

**Optimal depth is 7 in both frames.** The marginal cost of row 9 is 27.70 ms local / 25.01 ms ranked against a 23.71 ms/token running cost, so `measured_row9_repays_local = false` and `measured_row9_repays_ranked = false`. A kink-removed linear fit (ranked slope 19.80 ms/row) would instead place the optimum at 8; I report both because the difference between them *is* the stream-count kink analysed next.

---

## fb10 / fb11 — labelling by weight-stream count

Local `vector_limit = 10` (`applegpu_g16s`, gen 16), so widths 1..9 all take the `qmv` + crossrow family here and the sweep is dispatch-clean. Streams = `ceil(width / 4)`: widths 1–4 = 1 stream, 5–8 = 2 streams, **width 9 = 3 streams**.

Aggregated over every trace in the campaign (`research/fb11-stream-cost.json`):

| depth | width | streams | rounds | median round ms (local) | median ms (head-rebased) |
|---|---|---|---|---|---|
| 1 | 2 | 1 | 3 | 73.23 | 70.54 |
| 2 | 3 | 1 | 3 | 79.84 | 74.47 |
| 3 | 4 | 1 | **2** | 105.26 | 97.19 |
| 4 | 5 | 2 | 212 | 127.14 | 116.38 |
| 5 | 6 | 2 | 34 | 146.59 | 133.14 |
| 6 | 7 | 2 | 75 | 168.53 | 152.40 |
| 7 | 8 | 2 | 148 | 189.91 | 171.09 |
| 8 | 9 | **3** | 230 | 217.92 | 196.41 |

Marginal cost of each added row (local / head-rebased ranked):

| added row | band transition | local | ranked |
|---|---|---|---|
| +w3 | 1 → 1 | 6.61 | 3.92 |
| +w4 | 1 → 1 | 25.41 | 22.72 |
| **+w5** | **1 → 2 BOUNDARY** | **21.88** | **19.20** |
| +w6 | 2 → 2 | 19.45 | 16.76 |
| +w7 | 2 → 2 | 21.95 | 19.26 |
| +w8 | 2 → 2 | 21.38 | 18.69 |
| **+w9** | **2 → 3 BOUNDARY** | **28.01** | **25.32** |

`within-band mean 18.959`, `boundary mean 24.947` (head-rebased frame).

**The width-4 vs width-5 pair the advisor asked for is now measurable** — Run M supplied the only width-4 rounds in the campaign — but with **n = 2 against n = 212** it is weak evidence, and the anomalous +w4 step (25.41 ms, larger than the boundary crossing it precedes) is an n = 2 artifact that should not be read as signal. What the pair does support is a negative claim: the +w5 step (21.88 ms) is **indistinguishable from an ordinary within-2-band step** (19.45 / 21.95 / 21.38, mean ≈ 20.92). The 1 → 2 crossing is unremarkable. The 2 → 3 crossing at width 9 is 28.01 ms, **≈ 34 % above the 2-stream within-band mean**, and it is the only real cliff in the table.

Reliable widths are 5 (n = 212), 6 (n = 34), 7 (n = 75), 8 (n = 148), 9 (n = 230). Widths 2, 3 and 4 (n = 3, 3, 2) are too sparse for quantitative claims. Width 1 was never selected.

### Gate recommendation as a function of stream count

- **1-stream rounds (widths 1–4).** Essentially never selected by the cost model — 8 rounds out of 707 across the whole campaign. The gate is irrelevant in this band.
- **2-stream rounds (widths 5–8).** Marginal row cost ≈ 21 ms against a 23.71 ms/accepted-token running cost. **Every row in this band repays itself.** The gate should be maximally permissive here; ideally absent.
- **3-stream rounds (width 9).** Marginal row cost 28.01 ms local / 25.32 ms ranked, both **above** the 23.71 ms/token running cost. This row only repays when acceptance is high enough that it usually lands. **The gate should exist solely to protect the 2 → 3 transition.**

The shipped design conflates two decisions: one streak threshold withholds four rows (widths 5–8, all of which pay) in order to withhold one expensive row (width 9). That is why the measured sweep is so flat — the gate is mostly taxing rows that did not need taxing. Splitting it, so that `segmentedStreakGate` governs only the depth 7 → 8 step while depths 4–7 open freely, is the change this evidence supports. I have **not** implemented it: it is outside the assigned scope and is filed as a follow-up.

### Required contingency statements

> **If the ranked host's `vector_limit` is 6, this recommendation is void above width 5 and the gate should instead be a hard shallow wall at width 5 (`segmentedVerifyDepthCap = 4`).**

Under `vector_limit = 6`, widths above 5 leave the `qmv` + crossrow family entirely and the local cost curve stops transferring; no local measurement in this report constrains their cost, so the safe configuration is the one that never leaves the measured region.

> **If width 9 becomes a 2-stream op (`NA = 5`)**, width 9 stops being a boundary crossing. The 28.05 ms step should collapse toward the ~21 ms within-band marginal, depth 8 begins to repay against the 23.71 ms/token running cost, and the correct move is to **drop the gate entirely (`segmentedStreakGate = 0`)** — the very configuration Run M shows losing today. The boundary of interest would move to width 6 (depth 5), and the gate, if kept at all, should protect *that* step instead.

---

## fb2 — occupancy model and the blended-ratio sweep

`research/occupancy_model.py` implements a Markov chain over `fullAcceptStreak` that replicates `costModelDepth` exactly (H = 0.20, SHALLOW_CAP = 4, DEEP_CAP = 8), driven by the **measured** cost table `C(d)` = {2: 79.70, 4: 126.40, 5: 146.50, 6: 168.30, 7: 189.70, 8: 217.40} ms with a linear fill `12.2 + 22.5·(d+1)`, head delta 2.6893 ms/draft, serial 0.073532175738364458 s/token, and a fixed non-block overhead of 4012.5/512 ms per token.

### Validation against all nine measured arms

`research/occupancy_validate.py` now carries every arm, including the real Run J (cap 7 / gate 3) that was previously mislabelled in this table — the row that used to be called "J" was in fact the gate-2 arm, and is listed here as **J₂**. The model itself is structurally **cap 8**: `DEEP_CAP` is fixed, so every arm is scored as if it had run cap 8 and only the measured acceptance profile changes between rows.

| arm | cap | gate | pooled conditional q | shallow-cap accept | deep-cap accept | model raw | measured raw | model/measured |
|---|---|---|---|---|---|---|---|---|
| I | 8 | 3 | 0.959821 | 0.9130 | 0.9806 | 2.100669 | 2.0947033 | 1.002848 |
| K | 8 | 2 | 0.960089 | 0.8889 | 0.9801 | 2.105931 | 2.1019607 | 1.001889 |
| L | 8 | 1 | 0.962637 | 0.8644 | 0.9773 | 2.118051 | 2.1288141 | 0.994944 |
| N | 8 | 1 | 0.962637 | 0.8644 | 0.9773 | 2.118051 | 2.1311965 | 0.993832 |
| M | 8 | 0 | 0.960613 | **null** (no shallow rounds exist) | 0.9606 | 2.044294 | 2.0600336 | 0.992359 |
| J₂ | 7 | 2 | 0.961969 | 0.8791 | 0.9831 | 2.076416 | 2.1243837 | 0.977421 |
| J | 7 | 3 | 0.968539 | 0.9000 | 0.9884 | 2.095877 | 2.1636874 | **0.968660** |
| O | 7 | 3 | 0.968539 | 0.9000 | 0.9884 | 2.095877 | 2.1615420 | **0.969621** |
| P₅₁₂ | 7 | 3 | 0.968539 | 0.9000 | 0.9884 | 2.095877 | 2.1623244 | **0.969270** |

Run N is a useful check on the model as well as on the hardware: because the schedule is deterministic, N feeds the model **byte-identical inputs** to L, so the model emits exactly the same 2.118051. The only thing that moves between the two rows is the measurement, and it moves by 0.11 %. The model's residual against gate 1 is therefore a genuine model error of about half a percent, not a sampling artifact — the fixed-selection-gap bias described below, not noise. J, O and P₅₁₂ do the same at cap 7: identical acceptance inputs, one model output, three measurements spanning 0.10 %.

### ★ The model's error is not uniform — it splits exactly along the cap

This is the finding that fell out of adding the corrected arms, and it is **independent support for the stream-count mechanism**:

- On all five **cap-8** arms the model is accurate to **under 1 %** (worst 0.76 %), in both directions.
- On all three **cap-7 / gate-3** arms it under-predicts by **3.07 %, 3.04 % and 3.13 %** — a tight, one-sided cluster, roughly 4× the worst cap-8 error and the same size as the measured cap 8 → cap 7 effect (+3.085 %).
- J₂ (cap 7 / gate 2) sits between them at 2.26 % under, in the same direction.

The model is not merely "missing a number". It has the cap-7 traces' own acceptance data — a **better** pooled q (0.9685 vs 0.9598) and a **better** deep-cap acceptance (0.9884 vs 0.9806) — and it still predicts the cap-7 arms should be *marginally slower* than Run I (2.095877 vs 2.100669, −0.23 %). Measured, they are **3.1 % faster**. The model gets the **sign of the cap effect wrong**, not just its magnitude.

The same asymmetry shows up in the counterfactual gate ordering. `forward_raw_by_gate` for the cap-7 profile predicts gate 2 = 2.134357 against gate 3 = 2.095877, i.e. loosening the gate should buy +1.84 %. Measured at cap 7, gate 2 (J₂, 2.1243837) is **1.72 % slower** than the three-repeat gate-3 mean (2.1625180) — wrong sign. At cap 8 the model makes the same directional call (gate 2 = 2.128775 > gate 3 = 2.100669, +1.34 %) and there it is **right**: K beats I by +0.35 %. So the model's gate ordering is correct at cap 8 and inverted at cap 7. That is precisely the region where the fb10/fb11 weight-stream discontinuity lives.

**Why this is evidence and not circularity.** The model's cost input `C(d)` is a smooth, depth-proportional table with a linear fill; it has no term for how many weight streams a given row width requires. It therefore *cannot* represent a cost cliff between width 8 and width 9, and it cannot represent that removing width-9 rounds is worth more than the drafts those rounds would have produced. Feeding it the cap-7 acceptance profile makes it predict the cap-8 answer, and the residual it leaves behind is the part of the effect that acceptance and depth **do not explain**. That residual is 3.1 %, one-sided, present in all three repeats, and absent from every cap-8 arm.

**Honest limits.** This is a negative result about a model, not a fitted alternative model. I did not extend `occupancy_model.py` with a per-width stream term and refit — that would be the correct way to turn this observation into a quantitative claim, and it is listed as a follow-up. The cap-7 residual is also confounded with the fixed-selection-gap bias described below, which already biases the model in the same direction by roughly half a percent; the stream-count reading accounts for the remaining ~2.5 %, not the whole 3.1 %.

### Sweep over acceptance q (Run I profile, local frame)

| q | depth shallow/deep | P(d = 8) | tokens/round | shallow only | gate 3 | gate 2 | gate 1 | no gate |
|---|---|---|---|---|---|---|---|---|
| 0.70 | 1 / 5 | 0.0000 | 2.005 | 1.5946 | 1.5974 | 1.5997 | 1.6030 | 1.6069 |
| 0.80 | 2 / 7 | 0.0000 | 2.745 | 1.5782 | 1.6149 | 1.6557 | 1.7199 | 1.7923 |
| 0.85 | 3 / 7 | 0.0000 | 3.358 | 1.5695 | 1.6211 | 1.6911 | 1.8095 | 1.9411 |
| 0.90 | 4 / 8 | 0.2325 | 4.360 | 1.6353 | 1.6955 | 1.7643 | 1.8746 | 2.0004 |
| 0.93 | 4 / 8 | 0.4032 | 5.480 | 1.7897 | 1.8968 | 1.9556 | 2.0249 | 2.0927 |
| 0.95 | 4 / 8 | 0.5555 | 6.412 | 1.9026 | 2.0362 | 2.0753 | 2.1155 | 2.1530 |
| **0.96** | 4 / 8 | 0.6410 | 6.921 | 1.9620 | 2.1018 | 2.1297 | 2.1572 | 2.1827 |
| 0.98 | 4 / 8 | 0.8221 | 7.978 | 2.0871 | 2.2152 | 2.2243 | 2.2330 | 2.2412 |

Head-rebased shallow-only for reference: 0.70 → 1.6593, 0.80 → 1.6720, 0.85 → 1.6799, 0.90 → 1.7589, 0.93 → 1.9221, 0.95 → 2.0411, 0.96 → 2.1036, 0.98 → 2.2350; at q = 0.98, gate 2 = 2.4029 and no-gate = 2.4237.

**Where each gate stops paying.** All pairwise crossovers are `null` — the ordering `shallow < gate 3 < gate 2 < gate 1 < no gate` holds at every q in the sweep, so within this model no gate ever becomes *strictly* better than a looser one. What changes is the **size of the prize** for relaxing gate 3 → no gate: **+23.7 % at q = 0.85, +18.0 % at q = 0.90, +7.2 % at q = 0.95, and only +1.2 % at q = 0.98.** Our measured operating point is q ≈ 0.96. **The gate matters least exactly where we operate**, which is the model's explanation for why the entire measured sweep spans only 3.3 % of ratio.

### The fixed-selection-gap caveat, now doubly confirmed

The model's forward predictions all over-shoot, and the reason is visible in the data:

`forward_raw_by_gate` (gates −1 / 3 / 2 / 1 / 0), calibrated on each arm:

| calibrated on | −1 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|
| I | 1.960956 | 2.100669 | 2.128775 | 2.156461 | 2.182148 |
| K | 1.891934 | 2.061911 | 2.105931 | 2.148044 | 2.184615 |
| L | 1.823317 | 1.994473 | 2.056550 | 2.118051 | **2.170760** |
| M | 1.907279 | 1.972258 | 1.993915 | 2.018503 | 2.044294 |

The model assumes the shallow/deep acceptance gap is a property of the *configuration*. It is not — it is a property of the *selected subpopulation*. Relaxing the gate admits progressively worse rounds into the deep bucket, and deep-cap acceptance falls with it: **0.9806 (g3) → 0.9801 (g2) → 0.9773 (g1) → 0.9606 (g0)**, where the split vanishes completely (`pooled_shallow_cap = null`, because no shallow rounds exist to compare against). The gate-counterfactual data shows the same dilution directly: the deep subpopulation's µs/token degrades from ~25 600–25 800 at gates 3/2 to **27 553 at gate 0**.

So the model's monotone ordering is not merely imprecise at gate 0, it is **qualitatively wrong** there, and it is wrong for a reason the model structurally cannot represent. That is the honest limit of this tool: it is a good interpolator inside the gated regime and it should not be used to extrapolate past the gate.

### Pre-registered model comparison (Run M)

Before launching Run M I committed `research/gate0-preregistration.json` with two competing predictions and a decision rule:

| model | predicted raw at gate 0 | vs gate 1 |
|---|---|---|
| two-state | 1.9724 | −7.35 % |
| occupancy (L-calibrated) | 2.17076 | +1.97 % |

Decision rule: raw ≥ 2.17 → occupancy vindicated, delete the gate; raw ≤ 2.05 → two-state right, keep gate 1.

**Measured: 2.0600336024.** Neither literal trigger fired — it lands between the thresholds — and on absolute error the occupancy model is actually slightly worse (5.38 % vs 4.25 %). But the **direction** is unambiguous and only the two-state model got it right: gate 0 is *worse* than gate 1, where the occupancy model predicted better. **The streak gate earns its keep** — deleting it is the single worst arm in the sweep.

That conclusion survives; the *setting* it implied does not. This preregistration only ever compared gate values **at cap 8**, and within that family gate 1 is indeed the best of the four. The cap × gate sweep below shows the gate's optimum moves with the cap: at cap 7 the ordering reverses and gate 3 wins. The shipped configuration is therefore cap 7 / gate 3, and the "gate 1" recommendation this section originally carried is withdrawn.

---

## fb3 — realised per-position acceptance is flat, not decaying

The correct estimator is the **conditional** rate P(accept draft *i* | drafts 0..*i*−1 accepted); the unconditional rate manufactures a decay out of survivorship.

Conditional, by draft index:

| run | rounds reaching pos 0–7 | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|---|---|
| I | 82/78/74/71/41/38/35/29 | 0.951 | 0.949 | 0.959 | 0.958 | 0.976 | 1.000 | 0.971 | 0.931 |
| J | 82/78/75/72/52/46/42/0 | 0.951 | 0.974 | 0.960 | 0.944 | 0.942 | 0.978 | 1.000 | — |
| K | 79/74/72/69/47/43/37/30 | 0.949 | 0.973 | 0.958 | 0.942 | 0.979 | 0.977 | 0.919 | 1.000 |
| L | 74/71/70/66/52/48/41/33 | 0.959 | 0.986 | 0.943 | 0.939 | 1.000 | 0.958 | 0.976 | 0.939 |
| M | 73/69/68/63/56/50/43/35 | 0.945 | 0.986 | 0.956 | 0.952 | 0.946 | 0.980 | 0.977 | 0.943 |
| N | 74/71/70/66/52/48/41/33 | 0.959 | 0.986 | 0.943 | 0.939 | 1.000 | 0.958 | 0.976 | 0.939 |

Unconditional, for contrast:

| run | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|---|
| I | 0.951 | 0.902 | 0.866 | 0.829 | 0.930 | 0.927 | 0.895 | 0.818 |
| J | 0.951 | 0.927 | 0.889 | 0.840 | 0.907 | 0.882 | 0.875 | — |
| K | 0.949 | 0.923 | 0.885 | 0.833 | 0.939 | 0.913 | 0.850 | 0.833 |
| L | 0.959 | 0.946 | 0.892 | 0.838 | 0.929 | 0.902 | 0.870 | 0.861 |
| M | 0.945 | 0.932 | 0.890 | 0.845 | 0.828 | 0.803 | 0.778 | 0.786 |
| N | 0.959 | 0.946 | 0.892 | 0.838 | 0.929 | 0.902 | 0.870 | 0.861 |

Run N's rows are identical to L's to three decimals in both tables, and in fact bit-identical in the underlying counts, because the schedule is deterministic. It adds no independent acceptance sample; it is listed for completeness and as a consistency check on the analysis pipeline.

Two refutations follow:

1. **The shipped source comment was wrong.** It claimed acceptance "decays with position (0.85 at draft index 7)". The conditional rate at index 7 is **0.931–1.000** across every run that reached index 7. The unconditional rate does decay, and that is a survivorship artifact. I corrected the comment in this branch.
2. **The imported external prior does not transfer.** fb3 cited mlx-lm PR #990, where deeper drafting loses on stock MLX with p5 ≈ 0.234. Our p5 is 0.958–1.000. Native MTP with a trained head on this checkpoint is a different acceptance regime, and "relaxing the gate regresses" is not the outcome here — relaxing it *helps*, up to a point.

Run M is the clean illustration of the artifact: it has the flattest *conditional* curve and the steepest *unconditional* one (0.945 → 0.786), because with no gate every round is sent deep and the unconditional denominator stops being a selected population.

---

## Low-acceptance arm — delivered as an in-run regime split

**Constraint.** `benchmark-qwen-mtp.sh` derives both the drift tripwire and the timed seed from the same `${public_golden_path}` (`cases[0].prompt_tokens`), overridable only through `MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE`, and the only two fixtures in the repository (`correctness_prompts/public_longcopy_gate_english_512_256.json` and `…_512_1024.json`) **share the same seed**. There is no second acceptance regime available locally without adding a fixture, which is out of scope. I therefore delivered the low-acceptance arm as (a) the within-run regime split at decode index 301 and (b) the shallow-vs-deep subpopulation split, and I state plainly that neither is a substitute for a genuinely different prompt.

### Regime split at decode index 301 (`research/regime_split.py`)

| run | segment | rounds | tokens | accept | reject-round share | mean depth | tokens/round | rounds/token | µs/token |
|---|---|---|---|---|---|---|---|---|---|
| I | before | 44 | 301 | 0.9449 | 0.1364 | 6.182 | 6.841 | 0.14618 | 25 798.7 |
| I | after | 38 | 211 | 0.8199 | 0.3158 | 5.553 | 5.553 | 0.18009 | 29 312.2 |
| K | before | 40 | 301 | 0.9596 | 0.1000 | 6.800 | 7.525 | 0.13289 | 25 332.4 |
| K | after | 39 | 211 | 0.8113 | 0.3590 | 5.436 | 5.410 | 0.18483 | 29 732.9 |
| L | before | 38 | 301 | 0.9777 | 0.1053 | 7.079 | 7.921 | 0.12625 | 24 847.5 |
| L | after | 36 | 211 | 0.8102 | 0.3611 | 6.000 | 5.861 | 0.17062 | 29 344.2 |
| M | before | 37 | 301 | 0.9670 | 0.1081 | 7.378 | 8.135 | 0.12292 | 25 119.6 |
| **M** | **after** | 36 | 211 | **0.7353** | **0.3889** | 6.611 | 5.861 | 0.17062 | **31 670.7** |
| N | before | 38 | 301 | 0.9777 | 0.1053 | 7.079 | 7.921 | 0.12625 | 24 862.7 |
| N | after | 36 | 211 | 0.8102 | 0.3611 | 6.000 | 5.861 | 0.17062 | 29 350.8 |

(Run J is omitted from this table because it executed cap 7, so its segments are not comparable with the cap-8 arms. Run N's structural columns are identical to L's; only the two µs/token figures differ, by 0.06 % and 0.02 %, which is the timing-repeatability floor.)

**The post-index-301 segment is harder, not easier.** This contradicts fb8's inflation hypothesis (that the tail was inflating the score with easy copy tokens): acceptance drops from ~0.96 to ~0.81 and µs/token rises by ~15 % in every arm.

It also localises gate 0's damage. Run M's *before* segment is fine — the best mean depth and tokens/round in the campaign, µs/token essentially tied with the others. All of its loss is in the *after* segment: **31 670.7 µs/token against Run L's 29 344.2**, with acceptance collapsing to 0.7353 and 39 % of rounds ending in a rejection. This is the lowest-acceptance regime observed anywhere in the campaign, and it is a genuine low-acceptance datapoint even if it is not an independent prompt.

That vindicates the *mechanism* the original source comment appealed to — a streak gate damps the cascade in a hard regime — even though the comment's numbers were wrong. The gate is not paying for itself in the easy regime; it is insurance against the hard one.

### Shallow-vs-deep subpopulation (`research/gate_counterfactual.py`)

| run | shallow rounds | shallow µs/tok | shallow accept | deep rounds | deep µs/tok | deep accept | all µs/tok |
|---|---|---|---|---|---|---|---|
| I | 39 | 30 288.7 | 0.8077 | 43 | 25 800.2 | 0.9297 | 27 246.7 |
| J | 28 | 32 447.9 | 0.7455 | 54 | 25 086.5 | 0.9431 | 26 668.0 |
| K | 30 | 32 266.8 | 0.7607 | 49 | 25 595.2 | 0.9373 | 27 145.9 |
| **L** | 18 | 33 900.0 | 0.7083 | 56 | **25 579.3** | 0.9370 | **26 700.7** |
| M | 9 | 31 234.3 | 0.8235 | 64 | **27 553.4** | 0.8616 | 27 819.4 |
| N | 18 | 33 979.6 | 0.7083 | 56 | 25 580.4 | 0.9370 | 26 712.3 |

Deep rounds are consistently ~20 % cheaper per token than shallow ones — which is the whole case for relaxing the gate — but Run M shows the deep bucket itself degrading once selection is removed (25 600–25 800 → 27 553).

### Counterfactual headroom

Replaying each trace under alternative gate thresholds, counting rounds that *would* have been opened:

| run | gate 4 | gate 3 | gate 2 | gate 1 | observed deep rounds |
|---|---|---|---|---|---|
| I | — | 43 | 52 (+9) | 63 (+20) | 43 |
| K | 37 | 42 | 50 | 60 (+11) | 49 |
| L | 37 | 41 | 46 | 56 (0 newly opened — self-consistent ✓) | 56 |
| M | — | 39 (newly 73) | 45 (newly 59, 73) | 54 (newly 56, 58, 59, 73) | 64 |

From Run I the optimistic ceiling for gate 2 was ≈ +1.2 % and for gate 1 ≈ +2.7 %. Measured, gate 1 delivered **+1.63 %**, i.e. **60 % of the optimistic ceiling** — the shortfall being exactly the dilution effect described above. Run L's counterfactual reproducing its own observed 56 deep rounds is a useful self-consistency check on the tooling.

---

## EMA-conditioned variant — analytically refuted, no run spent

The assignment asked for an EMA-conditioned gate variant. I refuted it from the traces instead of spending a run, because the required separation does not exist.

`research/ema_distribution.py` reports `separated: false` at **every** index 0–7 in **every** run. The critical comparison is the mean EMA at the gate's decision point, split by whether the round went on to fully accept or to reject:

| run | index | full-accept mean (n) | reject mean (n) | separated? |
|---|---|---|---|---|
| I | ema[4] | 0.9561 (64) | **0.9712** (18) | no — sign inverted |
| K | ema[4] | 0.9546 | 0.9498 | no — gap far too small |
| L | ema[2] | 0.9263 | **0.9421** | no — sign inverted |
| L | ema[4] | 0.9739 (57) | **0.9979** (17) | no — sign inverted |
| M | ema[2] | 0.9333 | **0.9773** | no — sign inverted |
| M | ema[5] | 0.9483 | **0.9902** | no — sign inverted |
| M | ema[6] | 0.9214 | **0.9300** | no — sign inverted |

Run M's full profile (n = 55 full-accept / 18 reject): ema[0] 0.9521 vs 0.8704; ema[1] 0.9710 vs 0.9460; ema[2] 0.9333 vs 0.9773; ema[3] 0.9352 vs 0.9012; ema[4] 0.9191 vs 0.9041; ema[5] 0.9483 vs 0.9902; ema[6] 0.9214 vs 0.9300; ema[7] 0.8781 vs 0.8748.

Counting sign inversions (`full_accept_mean < reject_mean`) across all six runs in that corpus (I–N) and all eight indices makes the refutation sharper than the table above suggests:

| index | runs where the sign is inverted | count |
|---|---|---|
| ema[0] | — | 0/6 |
| ema[1] | — | 0/6 |
| **ema[2]** | **I, J, K, L, M, N** | **6/6** |
| ema[3] | I | 1/6 |
| ema[4] | I, L, N | 3/6 |
| **ema[5]** | **I, J, K, L, M, N** | **6/6** |
| ema[6] | I, J, L, M, N | 5/6 |
| ema[7] | I, J, K | 3/6 |

An EMA gate needs rounds that are about to reject to show a *lower* EMA. Instead, at two of the eight indices the signal points the **wrong way in every single run**, and at no index does it separate in any run. That is not a weak predictor, it is an anti-predictor over part of its range: an EMA threshold placed at index 2 or 5 would systematically send the *rejecting* rounds deep. Meanwhile the streak signal separates cleanly and consistently: Run L's shallow-cap acceptance is 0.8644 against deep-cap 0.9773. **Streak is simply the better predictor**, and an EMA-conditioned gate would be a strictly worse version of what is already shipped. Spending a 30-minute allocation to confirm a refuted hypothesis was not a good use of the budget, and I would rather report the refutation than the run.

---

## The 4.02 s fixed overhead — the largest untouched lever

Decomposing each phase capture into block time and the residual (`research/serial-overhead.json`), where `03-mtp-timed.json` is the serial control (`is_serial_control = true`, depth 0, 512 rounds) and `04-mtp-timed.json` is the MTP leg:

| run | phase | serial | rounds | decode s | blocks s | **gap s** | s/token |
|---|---|---|---|---|---|---|---|
| I | 03 | yes | 512 | 37.6485 | 33.6342 | 4.0143 | 0.07353218 |
| I | 04 | no | 82 | 17.9732 | 13.9613 | 4.0119 | 0.03510386 |
| J | 03 | yes | 512 | 37.5521 | 33.5230 | 4.0290 | 0.07334391 |
| J | 04 | no | 82 | 17.6767 | 13.6635 | 4.0132 | 0.03452479 |
| K | 03 | yes | 512 | 37.6907 | 33.6737 | 4.0170 | 0.07361460 |
| K | 04 | no | 79 | 17.9312 | 13.9187 | 4.0125 | 0.03502188 |
| L | 03 | yes | 512 | 37.6717 | 33.6250 | 4.0467 | 0.07357754 |
| L | 04 | no | 74 | 17.6961 | 13.6795 | 4.0166 | 0.03456269 |
| M | 03 | yes | 512 | 37.6288 | 33.6094 | 4.0195 | 0.07349379 |
| M | 04 | no | 73 | 18.2661 | 14.2533 | 4.0128 | 0.03567602 |
| N | 03 | yes | 512 | 37.7374 | 33.7234 | 4.0140 | 0.07370585 |
| N | 04 | no | 74 | 17.7071 | 13.6856 | 4.0215 | 0.03458426 |

Mean serial gap **4.0234 s**; mean MTP gap **4.0148 s** across the six runs in that corpus (I–N). Per-token that is 7.8358 / 7.8384 / 7.8370 / 7.8449 / 7.8376 / 7.8545 ms for I–N — a total spread of 0.24 % across schedules that differ by 12 % in round count. A two-point fit over the 73–82 round range gives `per_round_overhead = −0.1017 ms` and `fixed_overhead = 4.0203 s` — i.e. the gap is **schedule-invariant**. It is the shared 512-token seed prefill, and both legs pay it in full. `program.md` confirms the ranked metric includes seed processing inside the same timed leg, so this is structurally faithful, not a local-harness artifact.

Consequences:

- Mean block time is 33.6315 s serial vs 13.8603 s MTP. **Ratio as measured 2.1066; ratio with the gap removed 2.4265.**
- Amdahl: driving block time to zero caps the ratio at ~9.4. A 1 % block-time saving yields only ~0.78 % of score.
- A **30 % prefill speedup (−1.2 s) moves the ratio 2.107 → 2.186, i.e. +3.7 %** — larger than this entire gate sweep, and larger than the assignment's own 2 % bar.

This is the strongest follow-up I found and I did **not** implement it: it is outside the assigned scope. Recommending it is, in my view, the most valuable output of this experiment.

---

## ★ Incident — a second conversation shared this checkout and wrote to my branch mid-run

`state/openhands_state/student-conversations.json` shows two live conversations for this role against the same working tree:

```
qwen38-r1-e2-deep-round-gate:r1 -> 904b83a3-feaa-41fa-b340-b098f0149da9
qwen38-r1-e2-deep-round-gate:r2 -> 9b8f38c5-2015-498c-a197-852df2bc4986   (this one)
```

The r1 conversation committed to my branch while my job was running:

| commit | time (UTC) | change |
|---|---|---|
| `75e7beb` | 18:10:46 | `segmentedVerifyDepthCap` 8 → 7 |
| `f3d48b2` | 18:14:58 | `segmentedStreakGate` 2 → 3 |

`75e7beb` landed **during my Arm-B1 Swift build**, and `f3d48b2` landed shortly after. Both constants that this experiment sweeps were changed by the other conversation, mid-flight.

**Disambiguating the two "Run J" artifacts.** An earlier revision of this report said flatly that "Run J executed gate 2 at cap 7". That conflated two distinct runs that both carry a `runJ` tag, and I am correcting it here because the winner of Part B is one of them:

| artifact | executed config | ratio | MTP s/token | W&B run name | W&B id |
|---|---|---|---|---|---|
| `research/score-runJ-cap7-512.json` | cap 7 / **gate 3** | 2.1636873696 | 0.03403966594487429 | `runJ-cap7-gate3-512` | `iwy987kn` |
| `research/score-runJ-gate2-512.json` | cap 7 / **gate 2** | 2.1243836568 | 0.03452479303814471 | `runJ-gate2-cap7-512` | `ixu99guw` |

The second of these — referred to as **J₂** throughout this report — is the corrupted arm: it was *assigned* cap 8 / gate 2 and executed cap 7 / gate 2 because `75e7beb` landed during its build. The first is the arm that ran after `f3d48b2` had also landed, so it executed cap 7 / gate 3, and it is the configuration that Run O later reproduced exactly and that Part B names the winner. The proof of the executed cap in both cases is in the trace's own per-round `cap=` field and in the depth histograms: both have zero width-9 rounds, which is impossible at cap 8.

So the foreign commits produced one damaged arm (J₂, assigned-vs-executed mismatch) and one accidental discovery (J, the cap-7 winner). Neither outcome was intended, and I have kept both and labelled them explicitly everywhere they appear. Note also that the shipped configuration this report recommends — `8e61e77`, cap → 7 and gate → 3 — *originated in the other conversation*, not in mine; my contribution is the controlled 2 × 2 that establishes it is the right choice and the reproduction (Run O) that proves it. The r1 conversation also launched its own benchmark job (`a0f6e10a-7078-4fda-b89c-3f21d4f7b441`), which I can neither inspect nor cancel, and which occupied my `run_job` slot.

**Mitigation, now in place and effective.** `research/run-gate-arm.sh` stamps `head=` and `dirty=` at launch, asserts the constants before the build, re-applies and re-asserts, then asserts again *after* the build; and the per-round trace records `cap=` and `streak_in=` so the executed configuration is recoverable from the artifact rather than from the git log. There have been **no foreign commits since `ca1e2d0`** — Runs K, L, M, N, O, P and P₅₁₂ all carry clean `dirty=0` stamps and post-build assertions matching their assigned constants. Runs I and J predate the runner and carry no stamp, which is why Run O was added; **Run P₅₁₂ closes the gap completely**, because it was stamped at `head=43438153a5ca93f4e5050229e2be2ffb01238d1f dirty=0` — the exact head of the submitted result — and it reproduced Run O's schedule, depth histogram, per-width row-gate table and mismatch samples **bit-for-bit**. The winning configuration is therefore attested by two independently stamped runs, one of them at the submitted head, and the unstamped Run J is now corroboration rather than a load-bearing measurement.

**This needs an advisor decision**, because I cannot resolve it from inside my own conversation: either close the r1 conversation, or give the two revisions separate worktrees. As long as both are live against one checkout, any future run on this branch is at risk of silently executing a configuration it was not assigned.

I could not raise this while it was happening: **`post_assignment_comment` is not present in my tool schema**, so no interim PR comment was possible, and I did not attempt a `gh`/REST/`git push` workaround.

---

## Limitations

1. **This is an M4 Pro, not the ranked M5.** Every number is directional for the ranked host. The head-rebased columns are my best attempt to correct for the largest known difference (the resident head is 3.55× its declared size locally), but they are a model, not a measurement.
2. **One prompt, one seed.** Both local fixtures share the same seed, so there is no independent prompt in this report. The regime split is a substitute, and a weak one.
3. **Local reference rows are candidate-generated.** Matching them proves internal consistency, not agreement with the organizer's hidden reference.
4. **The occupancy model's selection gap is fixed by construction** and therefore cannot represent gate-0 dilution; its forward predictions past the gate over-shoot systematically. It was also fitted before the cap-7 arms existed and does not predict them.
5. **The width-4 sample is n = 2.** The 1 → 2 stream-boundary claim is qualitative only.
6. **The local ratio partly cancels shared costs.** Both legs run the same candidate build, so schedule changes show up cleanly but a general target/kernel improvement would not. I report absolute candidate s/token alongside the ratio throughout for this reason.
7. **The 2 × 2 is unreplicated in two of its four cells.** Cap 7 / gate 3 and cap 8 / gate 1 were each run twice; cap 8 / gate 3, cap 8 / gate 2, cap 7 / gate 2 and cap 8 / gate 0 were run once. The sign reversal rests on single measurements in the cap 7 / gate 2 and cap 8 / gate 2 cells, though both are far larger than the measured repeat noise.
8. **`--local-submit` cannot corroborate the speed claim.** Its 128-token window is 19 rounds long and prologue-dominated; Run P confirms fidelity and is a null on speed. Only the 512-token iterate arms support the −3.085 %.
9. **Cap 7 was not swept further.** Caps 6 and below were only probed at 256 tokens in earlier arms (Run E, cap 6 / gate 3), never in the 512-token frame that produced this conclusion, so the optimum may not be exactly 7.

---

## Suggested follow-ups (not implemented)

1. **Pad the verify batch from 9 rows to 10 in `qmm_t_splitk`.** Cap 7 wins because width 9 crosses into a third weight stream. If a padded 10-row dispatch costs the same as 8 rows in the two-stream class, depth 8 becomes worth re-opening and this result inverts. It is the single most direct follow-up to this experiment.
2. **Attack the 4.02 s prefill floor.** A 30 % improvement is worth ≈ +3.7 % of score — more than the entire sweep. This is the highest-value target I found.
3. **Re-tune `headStepCostRatio` (0.20 → ≈ 0.30) now that the cap is 7.** The ratio was calibrated against a cap-8 schedule and is owned by another student; with width 9 removed, the head/target cost balance that it encodes has changed.
4. **Sweep caps 6 and 5 in the 512-token frame.** Cap 7 was found by accident, not by search. Cap 6 was only ever measured at 256 tokens. The optimum may be lower.
5. **Split the streak gate from the width wall.** Let `segmentedStreakGate` govern only the last step before a stream crossing and let the shallower depths open freely. The fb11 marginal-cost table says the current design taxes several rows that all pay, in order to tax one that does not.
6. **Measure the 1 → 2 stream boundary properly** with a forced shallow wall (`sdpaWidthWallDepthCap = 3`) screen, to get widths 3–4 out of n = 2 territory. `run-gate-arm.sh` would need an optional sixth parameter for the wall.
7. **Root-cause the terminal-round drift at `key_len ≈ 1024`.** It is width-independent and top-1-safe today, but the ranked window ends at exactly 1024 on every prompt, so the exposure is systematic rather than incidental.
8. **Add a second local fixture with a different seed**, so low-acceptance arms can be run as real arms instead of as within-run splits, and so the cap-7 result can be checked against a prompt it was not found on.
9. **Attack `draft_build + verify_build`**, which is 55–58 % of every round in every arm measured here.
10. **Build a rejection-manufacturing research harness** so acceptance-sensitive schedule changes can be tested without waiting for a naturally low-acceptance prompt. This must not touch `Qwen36MTPReferenceSession.swift` on any timed path.
11. **Give `occupancy_model.py` a per-width weight-stream term and refit.** The model is accurate to under 1 % on all five cap-8 arms and under-predicts all three cap-7 arms by ~3.1 % with the wrong sign on both the cap effect and the gate ordering, because its cost table is smooth in depth and cannot represent a stream-count cliff. Adding `ceil(width/4)` as a cost input and refitting against the nine arms already measured would turn that residual into a predictive tool, and would let follow-up 4 (caps 6 and 5) be screened analytically before spending runs.


---

## Verdict

**Part A: width 9 is bit-exact. The exactness gap the assignment set out to close does not exist, so its premise is not met — but the prescribed remedy ships anyway, for a completely different reason.**

Width 9 was compared 1,535 times across the seven 512-token arms and was **bit-exact in every arm where it was not the width that closed the window** — 919 of 919 such rows in Runs I, K and M. Across the whole corpus, every mismatch without exception sits in the *terminal* round at decode positions 1022–1024, and it lands on whichever width happened to close that window: width 8 in Run I, width 4 in Runs J/O and M, width 2 in Run K, width 9 in Runs L/N. Width 4 does not even enter the `AttentionUtils` split path, which is gated on `6 <= qL <= 9`, so the deep-width path is exonerated directly. Run N reproduces Run L's mismatch hexfloat-for-hexfloat and Run O reproduces Run J's, so the deviation is deterministic, not a floating-point race. Run P adds the cleanest confirmation of all: at 128 decode tokens the window ends at position 640 instead of 1024, width 8 is compared 79 times, and **there is no drift anywhere in the run**. The variable is decode *position* near `key_len = 1024`, not draft width. Top-1 identity is preserved in every compared row of every arm, `all_tokens_matched = true` throughout, and the trusted parent compares top-1 — this is not a ranked-failure risk today. It stays on the follow-up list because the ranked window ends at exactly 1024 on every prompt, so the exposure is systematic rather than incidental.

Cap 7 therefore does **not** fix the drift — it merely moves it from width 8 to width 4. It ships on throughput grounds alone, and the report is explicit about that so nobody later mistakes it for a correctness fix.

**Part B: `segmentedVerifyDepthCap = 7` with `segmentedStreakGate = 3` is the recommendation. It clears the bar with 1.5× margin, and the cap — not the gate — is the dominant lever.**

| | cap 8 / gate 3 (control, I) | **cap 7 / gate 3 (J, O)** | cap 8 / gate 2 (K) | cap 7 / gate 2 (J₂) | cap 8 / gate 1 (L, N) | cap 8 / gate 0 (M) |
|---|---|---|---|---|---|---|
| local ratio | 2.0947 | **2.1637 / 2.1615** | 2.1020 | 2.1244 | 2.1288 / 2.1312 | 2.0600 |
| candidate s/token | 0.03510 | **0.03404 / 0.03400** | 0.03502 | 0.03452 | 0.03456 / 0.03458 | 0.03568 |
| Δ s/token vs control | — | **−3.085 %** | −0.234 % | −1.650 % | −1.511 % | +1.630 % |

The assignment's bar was a **≥ 2 % reduction in absolute candidate s/token**. Cap 7 / gate 3 delivers **−3.085 %** (mean of J and O), the best gate-only arm delivers −1.511 %, and the repeat spread on the winner itself is 0.110 %. The effect is 28× the largest observed repeat noise and 6× the end-to-end drift of the eight unchanged serial control legs.

**The 2 × 2 shows a sign reversal, which is why the cap had to be swept and not assumed.** Dropping the cap 8 → 7 helps by −3.085 % at gate 3 but only −1.419 % at gate 2. Loosening the gate 3 → 2 helps by −0.234 % at cap 8 but *hurts* by +1.481 % at cap 7. The two constants are not separable, and an earlier revision of this report retracted the cap as "inert" on the strength of a single confounded arm; the controlled sweep refutes that retraction.

The mechanism is dispatch width, not draft quantity. Width 9 is the only width needing a third weight stream (`ceil(9/4) = 3`); capping at 7 collapses all 33 of the control's width-9 rounds onto width 8, keeps the deep mode intact (46 deep rounds vs the control's 38), cuts rejections 53 → 38, and declares 550 target rows instead of 565.

**The assignment's designated stop signal did not work.** Within the cap-8 family, mean chosen depth, rounds per token and accepted tokens per round all improve *monotonically* through gate 0, yet gate 0 is the worst arm in the sweep — the signal is anti-correlated with speed there. The winner then fails the signal from the other direction: cap 7 has a *lower* mean depth (5.790) than every cap-8 arm but the control, and is the fastest by 3 %. The quantities that rank all seven arms correctly are **µs per emitted token** and the **reject count**.

### Label: `local winner`

Cap 7 / gate 3 meets every condition the program sets for that label:

- **A clear same-host improvement.** −3.085 % absolute candidate s/token and +3.24 % local ratio, measured twice, against a matched control run in the same session on the same host, power state, token window, memory profile and head.
- **Well outside noise.** 28× the repeat spread, 6× the serial-leg drift.
- **Correct behaviour and counters.** `all_tokens_matched = true`, `residual_divergence_count = 0`, `parity_all_ok = true`, `max_rejected_tail_logit_delta = 0`, `non_drafting_round_count = 0`, row accounting balanced at 550 declared rows, `uses_pinned_mtp_head = true`.
- **No observed fidelity problem.** The row gate is clean at every width in Run P, and the only residual in the 512-token arms is a positional top-2 drift that is present in the control as well and never changes a token.
- **A self-contained submitted snapshot.** The candidate diff against `BASE_SHA` touches exactly one file inside `editablePaths`, and `--local-submit` passes on the packaged path.

The honest qualifications, which the advisor should weigh before an official submission: this is an M4 Pro rather than the ranked M5; it is one prompt and one seed; and the 128-token `--local-submit` window is too short to show the effect (+0.058 % versus the r1 submit artifact, a null), so the speed claim rests entirely on the matched 512-token iterate arms. What makes me comfortable with `local winner` regardless is that the mechanism — removing the only draft width that needs a third weight stream — is structural and should transfer, and cap 7 wins on three independent counters (s/token, µs per emitted token, declared rows) rather than on one.

### Recommendations

1. **Ship `segmentedVerifyDepthCap = 7` together with `segmentedStreakGate = 3`** — the pair as committed at `8e61e77` and reproduced by Run O, worth −3.085 % candidate s/token. Ship them as a pair: the 2 × 2 shows the gain is not additive and cap 7 with gate 2 gives back nearly half of it.
2. **Do not ship `segmentedStreakGate = 1`.** It was the best arm at cap 8 (−1.511 %), but at cap 7 the gate should stay at 3. An earlier revision of this report recommended gate 1 and recommended keeping cap 8; both of those recommendations are withdrawn.
3. **Re-open depth 8 only through verify-batch padding.** Padding 9 rows to 10 in `qmm_t_splitk` is the one change that could make width 9 cheap enough to be worth restoring; without it, cap 7 should stand.
4. **Redirect the next allocation at the 4.02 s prefill floor**, worth ≈ +3.7 % — more than this entire sweep — and untouched by any experiment in the campaign so far.
5. **Retire accepted-tokens-per-round as the stop signal for depth work** and replace it with µs per emitted token plus the reject count.
6. **Resolve the two-conversations-one-checkout hazard** before the next run on this branch.

