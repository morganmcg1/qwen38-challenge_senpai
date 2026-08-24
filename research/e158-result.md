# E158 result: the head's residual is capacity, not precision — and the three bf16 precision islands are dead weight

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e158_head_precision_recoverable_pp","available":true,"value":-0.00731},"secondary_metric":{"name":"e158_island_pct_vs_all_PROVISIONAL_none","available":true,"value":0.45835},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- **Student / branch:** `qwen-askeladd` /
  `qwen-askeladd/e158-head-precision-census`
- **Hypothesis and target cost:** E155 left 12.44 pp of marginal acceptance
  loss (9.67 pp conditional) that neither the ANN index nor the vocabulary trim
  can reach. That residual has two possible causes. Either the declared head is
  *quantized too hard* and a wider-precision head would recover acceptance
  (precision), or the head's single MTP layer simply cannot predict those
  tokens at any precision (capacity). E158 separates them by re-running the
  same exact-readout audit under the organizer-pinned **bf16** head, which is
  2.85x heavier per proposal slot. R1.B then prices the declared head's own
  three bf16 precision islands (`q`, `k`, `v`) by ablating them.
- **Decision:** **close the head-precision axis, and drop the precision
  islands.** `e158_verdict = close_head_precision`,
  `e158_island_verdict = drop_the_islands`. No candidate change is proposed by
  this branch; the submitted surface is byte-identical to the PR base. The
  island result is a concrete, cheap, *positive* recommendation for whoever
  next rebuilds the head.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:**
  `5f60bea8e154163e562287f1317a845ef10b0b44` (PR base, advisor branch
  `senpai/qwen38-mtp-r1`) / `0863b06ac16e26e48fc06e97444095b00feb66d4` / no
  candidate commit. The audit build was `305787da`; the instrument is reverted
  at `a5d7fc98` and the head of this branch is a research-only tree.
- **Yukon promoted submission / source ref used as frontier:** the bar is
  `ec24d591` at `3.72911001`, source `0863b06a`, as recorded in
  `senpai/frontier-state.json` at 15:57Z. I did not query Yukon and I did not
  call a submission; the advisor's brief explicitly forbade both.
- **Candidate build fingerprint:** audit-ON worker
  `b3f5d6cfdce13f645e6ea2b32eae4b637fe0f84d52e68a50e4b4ff49e6d91c25`;
  reverted, audit-free worker at the head of this branch
  `696416f78f1bfbf0103ec433a5d18d5b0738d118002dd4ce7049933f58a69ff3`, rebuilt
  and asserted with
  `senpai/rebuild-and-assert-worker.sh --forbid-symbol installIfRequested
  --forbid-symbol QwenMTPRecallAudit` (PASS, 576,978 symbols extracted).
- **Submitted-surface / generated-twin / metallib digests:** unchanged. No
  Metal source and no submitted Swift source differ from the PR base at the
  head of this branch.
  `git diff --stat 5f60bea8 -- Sources Vendor Package.swift Package.resolved
  mtp-head.manifest.json mtp-head` is **empty**.
- **Submitted candidate files:** none.
  `senpai/check-editable-budget.sh 5f60bea8…` reports
  `growth_attributable = 0`.
- **Supporting test, tooling, or documentation files:**
  `research/e158_head_census.py`, `research/e158_draft_step_bytes.py`,
  `research/e158_r1_session.sh`, `research/e158_island_session.sh`,
  `research/e158_island_curve.py`, `research/e158_wandb_log.py`,
  `research/e158-artifacts/*.json`, and this document. None is a submitted
  path.
- **MTP head provenance, digest, and draft policy:** two heads, deliberately.
  The **declared** head is
  `head_provenance.sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  (run tree, `config.json` + `model.safetensors`), manifest digest
  `559b24eb…`, origin
  `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274`, 427,742,600 bytes,
  **40 tensors**. The **pinned** head is the organizer default,
  as-staged tree digest `6bab9c82…`, 849,400,347 bytes, **15 tensors**, all
  BF16. Draft policy is the shipped adaptive schedule at offered depth 8 on
  every leg.
- **Token window, fixture, reference source, and harness:** 512 decode tokens
  behind a 512-token seed, public fixtures `beagle_a`, `essays_montaigne` and
  `benchfixture`, checked-in E128 goldens. **`harness=local` on every measured
  rate; `harness=ranked` on every conversion to published percent.**
- **Exact cell:** the declared head's proposal readout — the derived cluster
  ANN index over the affine-2 g64 `draft_lm_head` (`[98336, 5120]`, stored as
  `[98336, 320]` U32 plus `[98336, 80]` bf16 scales and biases), 6,146 leaves,
  922 probes, 14,752 refined rows, then a 32-candidate affine-4 rerank against
  the target `lm_head` — and, for the pinned arm, the *absence* of that
  readout.
- **Official causal path and score equation:** untouched. This branch changes
  no scored code; `senpai/verify-ranked-score-boundary.sh` passes.
- **Assignment-scope preflight:** `senpai/validate-assignment-scope.sh` takes
  `BASE_SHA SUBMITTED_PATH…` and correctly reports `research` as outside
  `editablePaths`. The real scope evidence is the empty submitted-surface diff
  quoted above.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `source=2652191/3000000 growth=197356/262144` for `growth_enforced` against
  `770a3ff2`, and `growth=0/262144` for `growth_attributable` against the PR
  base. I consumed no team growth budget.
- **Scored-path reachability evidence:** the audit is gated by
  `MLX_E155_RECALL_AUDIT`, absent from the shipped default, proven off on four
  sides (section 6). At the head of this branch the hook does not exist on the
  submitted surface at all.
- **Written promotion rule and verdict:** the brief pre-registered
  `>= 1.0 pp` recoverable means build a wider head, `0.3` to `1.0 pp` means one
  bounded arm, `< 0.3 pp` means close. Measured **`-0.0073 pp`** — the wrong
  sign. **Verdict: close.** For the islands the brief asked for the cheapest
  arm whose acceptance loss stays under the advisor's provisional break-even;
  `none` loses `0.0605 pt` against a `0.1717 pt` provisional threshold and
  saves the most bytes. **Verdict: drop all three.**
- **Pre-official evidence budget / timed legs used:** zero timed legs, zero
  gated legs, no thermal gate, no ABBA, no palindrome. Nineteen untimed
  512-token decodes (6 audit legs for R1, 12 for the island curve, 1 gate
  probe).
- **Frozen candidate SHA, if promoted for submission:** none.
- **Specific evidence that invalidated the frozen SHA, if any:** not
  applicable.
- **Submission owner / read-only receipt-watcher job ID:** not applicable. I
  did not call Yukon.
- **W&B:** run `t6x06104`,
  <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/t6x06104>,
  group `e158-head-precision-census`. Artifact
  `e158-head-precision-census` carries all ten JSON artifacts and all
  nineteen raw per-slot JSONL files.

## Evidence

- **Host, instance, chip, memory profile, toolchain, thermal policy:** Apple
  M4 Pro, 20-core `applegpu_g16s`, 48 GiB. No cool gate was taken and none is
  claimed: every artifact records `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `timing_valid=false`,
  `official_or_ranked_score=false`.
- **Exact commands:**
  - `research/e158_head_census.py`
  - `research/e158_draft_step_bytes.py`
  - `research/e158_r1_session.sh declared beagle_a essays_montaigne benchfixture`
  - `research/e158_r1_session.sh pinned beagle_a essays_montaigne benchfixture`
  - `research/e158_island_session.sh all,q,kv,none beagle_a essays_montaigne benchfixture`
  - `research/e158_island_curve.py`, `research/e155_gate_probe.sh beagle_a`,
    `research/e158_wandb_log.py`
- **Cheapest real falsification gate and positive-control verdict:** the
  permutation control, run on both heads. Pairing each slot's shipped id with
  the *next* slot's target collapses `p_shipped` from `0.872921` to `0.015979`
  on the declared head (85.694 pp) and from `0.881588` to `0.010274` on the
  pinned head (87.131 pp). The comparison is capable of failing on both arms.
- **Tests and risk-based checks, in execution order:** safetensors header
  census on both heads; tree-digest recipe verification against the recorded
  `head_provenance.sha256`; head-swap round trip in both directions;
  golden-row exactness on all eighteen audit legs; audit row-count identity
  against `sum(effective_draft_lengths)`; gate probe; permutation control;
  masked-winner control; `ann_outside_compact_set` census; instrument revert
  and forbidden-symbol worker rebuild; then the three gate scripts on the final
  head.
- **Exact-token and row-ledger verdict:** `all_tokens_matched = true` and
  `residual_divergence_count = 0` on all eighteen audit legs and on the gate
  probe.
- **Divergent tokens or failure category:** none.
- **Generated-twin audit:** not relevant; no Metal source changed.
- **Peak RAM or head/artifact size:** the audit adds one dense
  full-vocabulary `lm_head` projection over all 248,320 ids per proposal slot,
  inside the measured block. Every audit leg is therefore non-timing by
  construction. Raw per-slot JSONL totals 2.3 MB and lives outside Git under
  `.mlxfast-private/e158/`.

### 1. Which head does each harness actually load?

`e158_heads_differ_local_vs_ranked = true`. The two heads on this machine are
different trees:

| | pinned (organizer default) | declared (`mtp-head.manifest.json`) |
|---|---|---|
| tree digest | `6bab9c82…` (as staged, 5 files) | `559b24eb…` bare / `dadbfb80…` run tree |
| file bytes | 849,400,347 | 427,742,600 |
| tensors | 15 | **40** |
| dtypes | BF16 only | BF16 + U32 + I32 |
| vocabulary projection | **none** | `draft_lm_head`, affine-2 g64 over `[98336, 5120]` |

`e158_head_swap_works = true` in both directions, and
`e158_tree_digest_recipe_verified = true`: recomputing the digest recipe over
the declared run tree reproduces `dadbfb80…` exactly.

**This corrects advisor FINDING 295.** `benchmark-qwen-mtp.sh` drafts with the
pinned bf16 head, but my E155 and E158 legs run through
`research/e128_session.sh:84`, which points the loader at the declared run
tree. Every E155 leg carried `head_provenance.sha256 = dadbfb80…`, so E155
measured the *shipped* head, not the pinned one. The E158 declared arm
reproduces E155's conditional bucket C to 16 significant figures
(`9.667673716012084` here versus E155's `9.6677`, on 1,503 slots) — an exact
determinism replication on a rebuilt worker, which is the strongest available
proof that the two experiments loaded the same weights. The advisor has
accepted this correction.

### 2. RULE 173: which provenance fields are witnesses

Four fields that read like head witnesses are not:

| field | why it is not a witness |
|---|---|
| `head_provenance.origin` | the manifest URL; identical for both trees I loaded from the same origin |
| `head_provenance.source` | `"remote"` vs `"local"`; describes fetch, not content |
| `uses_pinned_mtp_head` | true whenever *a* head drafted; true on both arms |
| `uses_native_mtp_head` | same; true on both arms |
| `mtp_head_tensor_count` | **compile-time constant 15**, not read from the file — the declared head has 40 tensors and still reports 15 |

Only `head_provenance.sha256` identifies the tree. E155's own result document
records "15 tensors" for the declared head; that number came from this
constant and is wrong. The correct figure is 40.

### 3. Coverage

| head | prompt | slots | rounds | mean offered depth | conditional slots |
|---|---|---:|---:|---:|---:|
| declared | beagle_a | 505 | 119 | 4.2437 | 442 |
| declared | essays_montaigne | 502 | 146 | 3.4384 | 430 |
| declared | benchfixture | 496 | 78 | 6.3590 | 452 |
| declared | **pooled** | **1503** | **343** | **4.3819** | **1324** |
| pinned | beagle_a | 482 | 121 | 3.9835 | 441 |
| pinned | essays_montaigne | 488 | 147 | 3.3197 | 432 |
| pinned | benchfixture | 491 | 77 | 6.3766 | 450 |
| pinned | **pooled** | **1461** | **345** | **4.2348** | **1323** |

The two arms are not the same slot population — a different head proposes a
different number of drafts — so all comparisons below use the conditional
population (slots where the shipped path had a genuine choice), which the two
arms match to within one slot (1,324 versus 1,323).

### 4. R1 — the precision decomposition

Six rates on the conditional population:

| | pinned bf16 | declared q2/q4 | pinned − declared |
|---|---:|---:|---:|
| `p_shipped` | 0.9002268 | 0.9010574 | −0.0008306 |
| `p_exact_compact` | 0.9002268 | 0.9018127 | −0.0015859 |
| `p_exact_full` | **0.9032502** | **0.9033233** | **−0.0000731** |

Three-bucket split, conditional, both partitions exhaustive:

| bucket | pinned bf16 | declared q2/q4 |
|---|---:|---:|
| A — selection defect (index) | 0.0000 pp | 0.0755 pp |
| B — candidate-set trim (vocabulary) | 0.3023 pp | 0.1511 pp |
| C — irreducible head disagreement | **9.6750 pp** | **9.6677 pp** |

**`e158_head_precision_recoverable_pp = −0.0073`.** Spending 2.85x the bytes
per proposal slot on a fully bf16 head recovers *negative* acceptance: one
slot in 1,323. Bucket C is flat to 0.008 pp across a 2.85x precision change.
The residual is capacity, not precision.

Priced against the advisor's F4 exchange rates, all **PROVISIONAL**:

| term | value |
|---|---:|
| `e158_bf16_bytes_per_draft_step_delta_MB` | 834.583 |
| `e158_precision_gain_pct_PROVISIONAL` | −0.0195 % |
| `e158_precision_cost_pct_PROVISIONAL` | +14.9641 % |
| `e158_precision_net_pct_PROVISIONAL` | **−14.9836 %** |

Break-even needed 4.054 acceptance points. The measured value is negative, so
the margin against the promotion rule is roughly 555x on the gain side and the
sign is wrong besides. No plausible refit of the F4 rate changes this verdict:
the acceptance term is zero within one slot regardless of what a point is
worth.

### 5. R0.3 — why the pinned head costs 834.58 MB more per draft step

Per proposal slot, at the current tree geometry (6,146 leaves, 922 probes,
14,752 refined rows):

| | trunk | readout | total |
|---|---:|---:|---:|
| declared q2/q4 | 264,494,080 | 33,528,960 | **298,023,040** |
| pinned bf16 | 849,398,784 | 283,207,680 | **1,132,606,464** |

The readout term is the surprise. The pinned head has **no vocabulary
projection of its own** (`has_full_vocab_projection = false`,
`has_compact_vocab_projection = false`). Without a `draft_lm_head`,
`buildDerivedClusterIndex` (`Sources/MLXFastModel/Qwen35.swift:6146-6152`)
guards on `_draftHeadW` and never builds an ANN index at all, so
`draftTokenID` falls through to `_compactDraftHead` **dense** — a full
98,336-row affine-4 g64 gather from the target `lm_head` every slot,
283,207,680 bytes.

The audit counters confirm this independently: `index_miss = 0` on 1,461
pinned slots versus `4` on 1,503 declared slots. A dense readout cannot miss
its index because it has none. **This corrects the advisor's F2 point 3 and
their 603.79 MB byte model**; the true delta is 834.58 MB.

The trunk term splits per matrix. `research/e158-artifacts/draft-step-bytes.json`
carries the full table; the three MLP matrices dominate at 128.12 MB each,
`q_proj` adds 79.95 MB, `fc` 75.37 MB, `o_proj` 45.22 MB, and `k_proj`/`v_proj`
add **nothing** — they are already bf16 in the declared head, because the
precision islands cover them completely. That last row is what makes section 8
worth running.

Also worth recording: the declared head keeps 5,906,432 bytes resident that
draft-time never reads (the affine-4 `k_proj`/`v_proj` packs and the k/v island
index vectors), because the bf16 islands supersede them.

### 6. Gate witness, four sides

1. **Source:** the audit hook only fires under `MLX_E155_RECALL_AUDIT`, which
   no shipped path sets.
2. **Two-sided probe:** `research/e155_gate_probe.sh` with
   `MLX_E155_RECALL_AUDIT_PATH` *set* and `MLX_E155_RECALL_AUDIT` *unset*, 64
   tokens, gives `audit_file_created = false`, `audit_rows_written = 0`,
   `leg_exit_code = 0`. The path variable alone does nothing. Saved as
   `research/e158-artifacts/gate-probe.json`; E155's own probe artifact was
   restored untouched.
3. **Revert:** `git apply -R research/e155-patches/recall-audit.patch` leaves
   zero references to the instrument anywhere under `Sources/`.
4. **Binary:** the rebuilt worker
   `696416f7…` passes `--forbid-symbol installIfRequested --forbid-symbol
   QwenMTPRecallAudit` over 576,978 extracted symbols. The audit build
   `b3f5d6cf…` is a different binary and is named in every audit artifact.

`e158_exact_path_is_gated_off_by_default = true`.

### 7. Positive controls (RULE 101)

| control | declared | pinned |
|---|---:|---:|
| permutation magnitude | 85.694 pp | 87.131 pp |
| masked-winner magnitude | 83.832 pp | 84.736 pp |
| `masked_control_moved` | 1,503 / 1,503 | 1,461 / 1,461 |
| `ann_outside_compact_set` | 0 | 0 |
| `index_miss` | 4 | 0 |

Both arms move the full slot population under both controls, so neither arm's
null result is a dead instrument.

### 8. R1.B — the precision-island curve

The declared head carries three bf16 precision islands over otherwise affine-4
projections: `q` (a partial row set), `k` and `v` (complete permutations). I
built a driver that installs any subset, and ran four arms x three prompts.
Every leg is witnessed by its own `trace.txt`, because `e128_session.sh`
overrides `MLX_QWEN_MTP_TRACE_PATH` per leg — the witness string names the arm
and both install flags.

| arm | rounds | conditional `p_shipped` | loss vs `all` | bytes/step Δ | published % (PROVISIONAL) |
|---|---:|---:|---:|---:|---:|
| all | 343 | 0.9010574 | — | 0 | — |
| q | 346 | 0.8967596 | +0.4298 pt | −15,073,280 | +0.2703 % |
| kv | 344 | 0.8996226 | +0.1435 pt | −10,489,856 | +0.1881 % |
| **none** | **342** | 0.9004525 | **+0.0605 pt** | **−25,563,136** | **+0.4583 %** |

Two standard errors on any pairwise acceptance difference is **2.32 pt** on
three prompts. Every measured difference is inside that band, so **the
acceptance side of this curve is a null result** and the ordering is noise —
which the curve itself shows, since `q` (one island kept) scores worse than
`none` (no islands kept). That is not a physically possible monotone response;
it is sampling.

The decision therefore rests on the two exact quantities:

- **Bytes.** Dropping all three islands removes 31,461,376 bytes of bf16 island
  weight but returns the 5,898,240 bytes of affine-4 `k`/`v` packs to the read
  set, so the net saving is **25,563,136 bytes**, not 31.46 MB. That is exact
  arithmetic over the safetensors header, not a measurement.
- **Round counts.** These are deterministic given the head and the prompt.
  `none` needs 342 rounds for 3x512 tokens; `all` needs 343. Removing the
  islands does not cost a round.

`e158_island_verdict = drop_the_islands`, worth **+0.4583 %** provisional,
which matches the advisor's composition table entry of `+0.458 %`.

The honest framing: I cannot prove `none` is *better* than `all` on
acceptance. I can prove it is not measurably worse on 1,326 slots, that it
reads 25.56 MB less per proposal slot, and that it finishes the same work in
one fewer round. For a head rebuild that is a free simplification.

### 9. Provisional-percent caveat

Advisor F4 retracted `+2.6701 %` per acceptance point, the 148.9 MB
break-even, and the 43,114 µs modelled round as self-confirming (RULE 166 /
FINDING 286 / ADVISOR ERROR 198). Every percent figure in this document that
derives from those rates is labelled PROVISIONAL, in the artifacts and in the
W&B summary keys. The quantities that do **not** depend on the retracted rates
are the six acceptance rates, the three-bucket splits, the byte censuses, the
round counts, and both verdicts — because both verdicts are decided by margins
of 100x or by exact byte arithmetic, not by the conversion constant.

## Suggested follow-ups (not implemented)

1. **Rebuild the declared head without the islands.** This is the one
   actionable output of E158: 25.56 MB per proposal slot for no measurable
   acceptance cost. It needs a head rebuild, which the brief forbade this
   round.
2. **Push the trunk ladder down, not the precision up.** Bucket C is flat
   across a 2.85x precision change, so the head is not precision-starved. The
   advisor's trunk ladder (a4 264.5 MB / a3 202.2 MB / a2 150.5 MB) is the
   interesting direction, and E158 predicts the acceptance cost of a3 or a2
   will also be small. Someone should measure it with this same driver.
3. **Attack capacity, not the readout.** 9.67 pp of conditional loss survives
   an exact full-vocabulary argmax on both heads. Only a different head
   *architecture* — more layers, or a different conditioning input — can move
   it. That is a large, slow experiment and should be scoped deliberately.
4. **Fix `mtp_head_tensor_count`.** It is a compile-time constant that lies
   about the loaded file. It is telemetry, not a scored path, but it has
   already produced one wrong number in a result document.
5. **Reclaim the 5.9 MB of never-read residency** in the declared head if the
   islands stay. Cheap, but strictly smaller than follow-up 1.

---

# R1 terminal result — the B1 receipt, the sealed E159 curve, and my reading of F18 section 3

Revision `r1`. Written 2026-08-24 on host `ip-10-231-2-227.ec2.internal`
(Apple M4 Pro, 48 GiB). Advisor F18 section 7 asked for three things and this
section delivers all three. Every number carries `harness=ranked` or
`harness=local`.

- `harness=ranked` means the official M5 runner. The serial numerator comes
  from the runner-owned prebuilt baseline workspace; the candidate denominator
  comes from the submitted workspace. Candidate code cannot move the numerator.
- `harness=local` means this M4 Pro host, one public fixture, both legs built
  from the candidate binary.

The two harnesses are never mixed inside one equation.

## R1.1 Official receipt `35a8a9de` — terminal, rejected (`harness=ranked`)

B1 is the island-arm default flip: the three bf16 precision islands in the
proposal head are switched off, which removes three dispatches and
`25,563,136` bytes per proposal slot.

| field | value |
| --- | --- |
| submission | `35a8a9de-14a2-40e1-920c-03bfd8f969ae` |
| submitted commit | `08f811009c9628785ba2ec0f19ca2e8fb3795689` |
| status | **rejected**, `rejectionReason = "score did not improve current best"` |
| `officialScore` | **3.6940622131456** |
| our previous best `5a9f130a` | 3.70784519415395 |
| validating | 2026-08-23T23:31:28Z |
| resolved | 2026-08-24T01:11:09Z (1 h 41 m) |
| `parity_all_ok` | `true`, 8 of 8 pairs |
| head on both receipts | `559b24eb…` |

The rejection is administrative. Parity passed on every pair, so B1 is
**correct and slower on the metric that ranks**, not broken.

### Per-prompt candidate leg, B1 against `5a9f130a` (`harness=ranked`)

| prompt | base `mtp_s/t` | B1 `mtp_s/t` | d ms | d % | base `edl` | B1 `edl` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `192fb621` | 0.009648 | 0.009743 | +0.095 | +0.984 % | 6.148 | 5.988 |
| `ea82dcb5` | 0.009697 | 0.009869 | +0.172 | +1.774 % | 4.989 | 4.796 |
| `00142a44` | 0.009704 | 0.009558 | −0.146 | −1.505 % | 5.256 | 5.281 |
| `a2ea8b60` | 0.009823 | 0.009746 | −0.077 | −0.784 % | 5.087 | 5.022 |
| `919318e1` | 0.010696 | 0.010651 | −0.045 | −0.419 % | 4.382 | 4.268 |
| `3b10cb4d` | 0.015638 | 0.015635 | −0.003 | −0.019 % | 2.648 | 2.623 |
| `4b9e88cd` | 0.017880 | 0.017466 | −0.414 | −2.316 % | 2.298 | 2.285 |
| `c1ec5866` | 0.030121 | 0.030131 | +0.010 | +0.032 % | 0.156 | 0.172 |

B1 is faster on 5 of 8 prompts. The mean candidate time falls from
`0.0141511` to `0.0141000` s/token, **−0.361 %**.

### The published median fell by an order statistic, not by speed

| | rank 4 | rank 5 | median |
| --- | ---: | ---: | ---: |
| base `5a9f130a` | 3.5453 (`919318e1`) | 3.8704 (`a2ea8b60`) | 3.7078452 |
| B1 `35a8a9de` | 3.5544 (`919318e1`) | 3.8337 (`ea82dcb5`) | 3.6940622 |

`ea82dcb5` regressed 1.774 %, fell from rank 7 to rank 5 and became the upper
central value. `a2ea8b60` improved and left the central pair. The median delta
is `(+0.0091 − 0.0367)/2 = −0.0138`, which reproduces the published
`−0.372 %` exactly. **A candidate that is 0.361 % faster on the mean lost
0.372 % on the median.** B1 also widened the top-four spread from 1.62 % to
3.38 %.

### The accept ledger moved, so this was never a runtime contrast

Effective draft length fell on 6 of 8 prompts. Regressing the per-prompt time
change on the draft-length change gives pearson `r = −0.746` and a slope of
**−11.9 % candidate time per accepted draft token**. Where B1 kept its drafts
it got faster; where it lost drafts it got slower. This is CAMPAIGN LAW 364
visible in the ranked data: the islands were buying acceptance, and the round
saving did not cover the loss.

### Five findings from the 949-receipt board pull (`harness=ranked`)

These were posted in full on the PR. Summary here for the record.

**A. The ranked serial noise floor.** Across 949 receipts the serial leg has
sd `0.125 %` and range `0.816 %`. Consecutive-receipt drift is median
`0.131 %`, p90 `0.271 %`. Daily mean drifted from `+0.031 %` (08-17) to
`−0.149 %` (08-23). Two coupling tests: the within-receipt slope of candidate
on serial is `+0.885` (`r = +0.019`, `n = 7592`), and over 117 consecutive
near-tied receipt pairs the slope is `+0.711` (`r = +0.208`, `p ≈ 0.03`). The
legs are partially coupled at 0.7 to 0.9, not independent.

**B. The top of the board runs our schedule.** There are 190 distinct
draft-length signatures among 949 receipts. **90 receipts across 20 users
carry ours exactly**, `[4.382, 0.156, 5.087, 2.648, 5.256, 2.298, 4.989,
6.148]`, and that set includes the crown. Speculation policy is saturated at
the top of the board. The contest there is pure candidate runtime.

**C. Empirical ranked elasticity.** Over the top cohort,
`published_score = 8.1526 − 0.31450 × candidate_mtp_ms`, `r = −0.958`,
residual sd `0.01047` points, which is `0.284 %` of score. **1 % of candidate
time is worth 1.2144 % of published score** (`harness=ranked`).

**D. About half the crown's lead is one anomalous serial draw.** The crown's
serial leg on `a2ea8b60` is `38.44804` ms against a field mean of `37.98997`
and sd `0.08931`, so `z = +5.13`, the 99.58th percentile of 949 receipts. Its
candidate leg on that same prompt is 0.170 % *faster* than the field, so the
draw is independent numerator noise and not thermal throttling. Rescoring on a
common numerator puts the true gap at **+0.205 %** (crown's serial) or
**+0.208 %** (ours), not the headline `+0.5735 %`. A cohort fit attributes
48 % of the crown's advantage to candidate time and 52 % to residual draw. The
advisor's independent draw-neutralisation in F18 reached `+0.2071 %`, which
agrees with both of my estimates.

**E. Dispersion is the largest single lever we are not pulling.** Equalising
our four fastest prompts at their own mean is worth **+0.500 %** published
with zero mean speedup, against only `+0.149 %` of headroom to the crown. This
is not prompt specialisation. It is reducing the dispersion of schedule
efficiency across draft-length regimes, and it is legitimate because the
schedule is input-independent.

## R1.2 E159 — the sealed gated pinned-width curve (`harness=local`)

Branch `qwen-askeladd/e159-pinned-width-curve`, commit `a9c0f7a6`, base
`001f8e24`. Kept local and unpushed under F17 section 3. Session job
`ce78f077`, exit 0, 2214 s. W&B run
[`i58bdswa`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/i58bdswa),
state `finished`, namespace `e159g/*`.

Eleven legs, one thermal session, `MLX_E159_FIXED_DRAFT_DEPTH=d` for the ten
pinned legs and the shipped adaptive policy for the eleventh. Every leg passed
the **real 40 C gate**: `cool_gate_passed_real_gate=true`,
`gate_qualified_for_timing=true`, entry temperature 39.37 C to 39.90 C. Every
leg produced `matched=true` and `residual_divergence_count=0` against the
public golden, and the width histogram witnessed the pin exactly on every
pinned leg.

| leg | pin `d` | rows `M` | rounds | `R` ms | sd | entry C | accepted draft rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asc1 | 1 | 2 | 259 | 69.403 | 1.742 | 39.37 | 0.9807 |
| asc2 | 2 | 3 | 176 | 71.674 | 2.795 | 39.60 | 0.9545 |
| asc3 | 3 | 4 | 134 | 78.124 | 2.504 | 39.66 | 0.9475 |
| asc4 | 4 | 5 | 109 | 91.640 | 2.903 | 39.61 | 0.9243 |
| asc5 | 5 | 6 | 96 | 127.141 | 4.775 | 39.58 | 0.8685 |
| asc6 | 6 | 7 | 82 | 138.853 | 4.756 | 39.56 | 0.8758 |
| asc7 | 7 | 8 | 73 | 146.332 | 6.660 | 39.48 | 0.8642 |
| desc4 | 4 | 5 | 109 | 91.362 | 1.402 | 39.90 | 0.9243 |
| desc5 | 5 | 6 | 96 | 126.764 | 3.869 | 39.37 | 0.8685 |
| desc6 | 6 | 7 | 82 | 138.591 | 2.400 | 39.86 | 0.8758 |
| natural | — | — | 78 | 138.288 | — | 39.34 | 0.8770 |

### The replicates satisfy CAMPAIGN LAW 364 by construction

The three ascending/descending replicate pairs have **digit-identical accept
ledgers**: 109/109 rounds at `M=5`, 96/96 at `M=6`, 82/82 at `M=7`, with the
accepted draft rate equal to 16 significant figures. These are pure runtime
contrasts. Ascending legs are `+0.306 ms` slower, sd `0.062 ms`, on all three
pairs, so `0.062 ms` is the leg-to-leg measurement interval used everywhere
below.

### The measured curve

| `M` | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R(M)` ms | 69.403 | 71.674 | 78.124 | 91.501 | 126.952 | 138.722 | 146.332 |
| `dR(M)` ms | — | +2.271 | +6.450 | +13.377 | **+35.451** | +11.770 | +7.610 |

All `dR` carry the same `± 0.062 ms` interval. `M=5`, `M=6` and `M=7` are
two-leg means.

### Prediction 1 — `STEP > 0`, order +10 to +18 ms. Sign confirmed, magnitude too large

| estimator | definition | `STEP` ms |
| --- | --- | ---: |
| linear | `R(6) − (s1 + h1·6)` | **+31.091 ± 9.717** |
| second difference | `dR(6) − dR(5)` | **+22.075** |
| quadratic | `R(6) − quadratic through M=3,4,5` | **+15.148** |
| advisor prediction | F17 section 5 | +13.9 (range 10 to 18) |

Every estimator is positive, so CAMPAIGN LAW 354's sign is confirmed and the
width-plan axis does not reopen. Only the most model-dependent estimator sits
inside the advisor's range. The spread across estimators is larger than any
one interval, which is the honest statement of how well this curve pins the
step.

### Prediction 2 — `h1 = h2`. Agrees only vacuously

| band | widths | `n` | `dof` | intercept `s` ms | slope `h` ms/row | worst residual ms | SSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `G=1` | 2,3,4,5 | 4 | 2 | 52.216 ± 12.982 | 7.2743 ± 3.5333 | −3.189 | 31.211 |
| `G=2` | 6,7,8 | 3 | 1 | 69.505 ± 16.927 | 9.6901 ± 2.4019 | +1.387 | 2.884 |

`h2 − h1 = +2.416 ± 4.272`, so the intervals overlap. That is not support for
the prediction. The band `G=1` interval is `± 3.53` on a slope of `7.27`; it
would fail to reject almost any hypothesis. The test is uninformative, not
passed.

### Correction to F17 section 6 — I cannot report `s1` prominently

**`R(M)` is not two straight lines.** Inside band `G=1` the marginal cost
*rises* monotonically, `+2.271 → +6.450 → +13.377`; inside band `G=2` it
*falls*, `+35.451 → +11.770 → +7.610`. Band 1 is convex and band 2 is
concave. Fitting a straight line to a convex arc puts a `−3.189 ms` residual
at the ends and drags the intercept down.

`s1 = 52.216 ms` is therefore an artifact of that misfit, not a measurement of
the per-round fixed cost. F17 section 6 asked me to report it prominently as
the local counterpart of the ranked intercept `16.158 ms` and the target of
the next campaign round. **I decline, and I recommend the advisor does not
adopt it.** Any deficit priced through `s1` inherits the misfit. The right
local instrument for the fixed term is a curve that either includes `M=1` and
`M=0` or models the convexity, and that is a separate experiment.

### Correction to F17 section 5 — the descending replicates land 1 on band 1 and 2 on band 2

F17 says the `d = 6, 5, 4` descending replicates put two extra points on band
`G=1` and one on band `G=2`. They pin `M = 7, 6, 5`, so one lands on band
`G=1` (`M=5`) and two on band `G=2` (`M=6, 7`). The band fits above use the
correct placement: band `G=1` has one two-leg point and band `G=2` has two.

Note also that the field `verify_passes_ceil_rows_over_5` in the curve
artifact is a **misnomer**. It is `ceil(M/5)`, which equals CAMPAIGN LAW 354's
weight-stream count `ceil(M/IPG)` for the shipped IPG table over `M = 2..8`.
It is a band label, not a measured verify-pass count. I did not measure verify
passes.

### Instrument validation against a number this session never saw

The tree comment at `Sources/MLXFastModel/Qwen36MTPBlockSession.swift:960-961`
records E68 rung 1, job `21ac5458`, at `13.405 ms` for the same marginal row.
This session measured `dR(5) = 13.377 ms`. That is **0.21 % agreement** across
a different experiment, a different driver and a different session, and it is
the strongest evidence that the pinned instrument measures what it claims to.

### Ranked transfer — no verify-row penalty (`harness=local` to `harness=ranked`)

The mean marginal row over `M = 2..8` is `12.822 ms`, which is **2.403x** the
ranked per-row law `5.3351 ms/row` (FINDING 352). RULE 192's decode transfer
factor is `2.947x`. Local marginal rows are therefore *cheaper* than RULE 192
predicts, not more expensive, so there is no ranked verify-row penalty hiding
here. This confirms the correction I made in the F16 response. The individual
widths straddle the factor: `M=6` transfers at `6.645x` and every other width
at `1.43x` to `2.51x`.

### Deliverable 4 — blind reconciliation, discharged

`research/e159_gated_curve.py` refuses any leg whose label contains `natural`
or `unpinned`. It produced `R_pred` from the pinned curve and thorfinn's
traced natural width histogram alone, and I posted `R_pred` on the PR while
the natural leg was still sealed.

| quantity | value |
| --- | ---: |
| `R_pred` (posted before unsealing) | 136.805 ms |
| `R_measured` (natural leg) | 138.288 ms |
| difference | **+1.483 ms (+1.084 %)** |
| within-leg sem, `n = 77` | 2.345 ms |
| leg-to-leg sd | 0.062 ms |
| histogram mean rows | 7.35936 |
| measured mean rows | 7.35897 |
| row difference | −0.00039 |

The prediction is inside one within-leg standard error and outside the
leg-to-leg interval. The pinned curve predicts an unpinned mixed-width leg to
about 1 %, which is good enough to price policy changes and not good enough to
price a 0.1 % runtime change.

### The result that matters most — a fixed width beats the shipped adaptive policy

Every leg emitted the **same 512 tokens** and matched the same golden, so this
is a *policy* comparison at identical output, not a runtime contrast. CAMPAIGN
LAW 364's ban on runtime elasticity does not apply, because nothing here is
priced with an elasticity: these are directly measured wall times for
identical work.

| policy | `M` | ms per token | vs shipped |
| --- | ---: | ---: | ---: |
| pin 1 | 2 | 35.205 | +67.35 % |
| pin 2 | 3 | 24.723 | +17.52 % |
| pin 3 | 4 | 20.510 | −2.51 % |
| **pin 4** | **5** | **19.530** | **−7.17 %** |
| pin 5 | 6 | 23.831 | +13.28 % |
| pin 6 | 7 | 22.271 | +5.86 % |
| pin 7 | 8 | 20.899 | −0.66 % |
| shipped adaptive | mixed | 21.037 | 0 |

**Pinning at `M=5` finishes the same 512 tokens 7.17 % faster than the shipped
adaptive policy**, 9.9997 s of decode against 10.7711 s. The shipped policy is
beaten by both `M=5` and `M=8`, so it is not sitting on either end of a
tradeoff — it is spending most of its rounds in the expensive `M≥6` region
that the `M=6` cliff makes uneconomic.

The marginal price of a marginal token makes the cliff explicit, against a
19.53 ms/token baseline at the optimum:

| step | ms per extra token per round |
| --- | ---: |
| `M` 2→3 | 2.44 |
| `M` 3→4 | 7.07 |
| `M` 4→5 | 15.26 |
| **`M` 5→6** | **55.73** |
| `M` 6→7 | 12.93 |
| `M` 7→8 | 9.89 |

Crossing into band `G=2` costs 55.73 ms for one extra token per round when a
token costs 19.53 ms. Nothing in bands 6 and 7 pays that back; only `M=8`
recovers to roughly break-even.

The measured acceptance function that the F18 section 4 prize model needs is:

| `M` | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted draft rate | 0.9807 | 0.9545 | 0.9475 | 0.9243 | 0.8685 | 0.8758 | 0.8642 |

Two caveats before anyone ships a fixed width. This is one public fixture on
one local host, and the ranked prompts span a much wider draft-length range,
from `edl = 0.156` to `edl = 6.148`. A fixed width that wins on a long-copy
fixture may lose badly on `c1ec5866`, which barely drafts at all. The honest
claim is narrower and still strong: **the shipped cost model is mispricing the
band boundary, and correcting that pricing is worth several percent locally.**
That is exactly F18's step 2b.

### Independent corroboration from edward's ungated session

Blind protocol discharged (see R1.4), I then read
`research/e159-artifacts/e159_cap_replay.json` from edward's session. His
board-deconvolved cap scan over 5751 cells finds `argmax cap = 3` under
`board_deconvolved_width6_at_neighbour_mean` for **+3.379 % median**, and it is
the only scenario in his artifact with `loo_stable = true`. That scenario is
defined by *healing the `M=6` cost to its neighbours' mean*, which is the same
mechanism my curve measures directly as the `+35.451 ms` cliff. Two
independent methods, one ranked-deconvolved and one locally gated, both say
the `M=6` step is the dominant avoidable cost and that a narrow cap beats the
shipped policy.

## R1.3 My reading of F18 section 3

### The `R` column mixes speed with row count

F18 section 3 defines `R = (mtp_spt − prefill_spt) × (1 + edl)`. The factor
`1 + edl` is the number of rows **proposed** per round. The number of tokens
**emitted** per round is `1 + edl × accepted_draft_rate`. Dividing
seconds-per-token by proposed rows therefore produces a quantity that falls
whenever the candidate proposes fewer rows, even when every emitted token got
slower.

Decomposing B1 against `5a9f130a` into a realised per-token term and a row
term (`harness=ranked`, both terms exact from the receipts):

| prompt | d decode per token | d rows | `dR` as F18 defines it | real sign of per-token speed |
| --- | ---: | ---: | ---: | --- |
| drama | −2.448 % | −0.378 % | −2.818 % | faster |
| travel | +0.002 % | −0.675 % | −0.673 % | **slower** |
| beagle | −0.415 % | −2.118 % | −2.524 % | faster |
| republic | +1.965 % | −3.228 % | −1.326 % | **slower** |
| essays | −0.855 % | −1.075 % | −1.921 % | faster |
| medicine | −1.676 % | +0.405 % | −1.278 % | faster |
| botany | +1.104 % | −2.239 % | −1.160 % | **slower** |
| plutarch | +0.041 % | +1.418 % | +1.460 % | slower |

**The sign disagrees on 3 of the 7 drafting prompts.** On travel, republic and
botany, `R` fell only because the row denominator shrank. So "R fell on all
seven" overstates the case: B1 bought realised per-token speed on 4 of 7 and
lost it on 3.

### This does not refute the F18 section 4 prize, and I want to be clear about that

The `φ_acc = 0` cell of the section 4 table is arithmetically
`d(spt) + d(rows)`, which is exactly the `dR` column above. So the prize model
is **internally consistent under mechanism (b)** — policy re-selection with
acceptance unchanged — because under (b) the row change is the mechanism
rather than a confound. It is invalid under mechanism (a), where the row
change is a *symptom* of lost draft quality and must not be credited as a
saving. F18 already flags exactly this, and E166 step 1 is the right test. I
am reporting the decomposition so the advisor can see how much of the section
3 column depends on that split: on three prompts, all of it.

### `FINDING 352`'s law does not reproduce the section 3 `R` values

Applying `R = 16.158 + 5.3351 × rows` to the section 3 rows gives a worst
residual of `+21.823 ms` and `SSE = 953.4`, against the claimed `SSE = 0.82`
and worst `0.51 ms`. A direct least-squares fit on those same values gives
`R = 50.157 + 0.8220 × rows`, `SSE = 58.7`. The section 3 `R` is not monotone
in rows at all: drama has 3.298 rows at 55.574 ms while republic has 5.989
rows at 51.917 ms.

The two advisor tables are therefore using **incompatible definitions of `R`**
and should be reconciled before either is used to price anything. My reading
of why: any cross-prompt ranked fit confounds the width mix with prompt
identity, because a prompt's `edl` and its intrinsic difficulty move together.
That confound is the whole reason a pinned local curve was worth 2214 s of GPU
time — it holds the width fixed and lets the mix vary by construction.

## R1.4 Blind protocol — discharged, with the timestamp

I did not open edward's `research/e159-artifacts/` until
**2026-08-24T01:38:13Z**. By that time my own curve artifact and
reconciliation artifact were written, committed at `a9c0f7a6`, and published
to W&B run `i58bdswa`. The pre-registered `R_pred = 136.805` was posted on the
PR before the natural leg was unsealed, and it was produced by a script that
refuses to read any leg labelled `natural` or `unpinned`.

## R1.5 Suggested follow-ups (not implemented)

1. **Retune the cost model at the band boundary (F18 step 2b).** The shipped
   `headStepCostRatio` and `measuredRawDepthPrice` at
   `Qwen36MTPBlockSession.swift:966-975` do not know that `M=6` costs 35.5 ms
   more than `M=5`. Feeding the measured `dR(M)` into the price is the single
   cheapest way to capture most of the 7.17 % local gap.
2. **Measure the `M=6` step at fixed rows.** My step is measured across rows
   and inherits the band `G=1` convexity. Alphonse's session `c6063e2e` holds
   rows at 5 and forces `IPG=3`. If the two numbers agree, the step is a
   weight-stream price. If they disagree, part of my `+31.091` is the
   convexity, and the linear estimator should be retired in favour of the
   second difference.
3. **Fit the fixed per-round term properly.** `s1` is unusable. A curve that
   includes `M=1` and `M=0` would measure the intercept directly and give F17
   section 6 the local number it actually wants.
4. **Check whether a fixed width survives the ranked prompt spread.** The
   local optimum `M=5` was measured on one long-copy fixture. The ranked pool
   spans `edl` 0.156 to 6.148, and `c1ec5866` barely drafts. A width plan that
   is fixed *per band* rather than globally fixed is the compliant middle
   ground, since the plan stays input-independent.
5. **Price the dispersion lever from finding E.** `+0.500 %` published for
   zero mean speedup is larger than the `+0.149 %` of headroom to the crown,
   and no one on the board is pulling it.
6. **Reconcile the two `R` definitions** in F18 section 3 and FINDING 352
   before either is used again.
