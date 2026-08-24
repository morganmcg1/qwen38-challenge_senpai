# E167 terminal result — the maintained base was shipped, measured, and the evidence it produced changed the campaign's premise

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":"180db84","primary_metric":{"name":"ranked_published_score","available":true,"value":3.70465399491642},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}
```

- Student / branch: `qwen-edward` / `qwen-edward/e167-ship-the-maintained-base`
- Assignment: `e167-ship-the-maintained-base`, PR 166, revision `r0`
- Decision: **shipped and measured; no score improvement; the by-products are the value**
- `BASE_SHA` used for the submission: `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`
- Assignment base: `40eae74b8901074bdd613b89b4ddef4c8faf7bf7`; candidate commit `a69e3de131d626e72503e1a5510c489634603318`
- Yukon frontier at submission time: crown `ec24d591`, newjordan, `3.729110011`, source ref `0863b06ac16e26e48fc06e97444095b00feb66d4` (unmoved when checked immediately before submitting)
- Harness: `ranked` for the receipt, `local` for every timed session below
- Host for all local work: Apple M4 Pro, gen 16, 48 GiB, `is_nax_available() == false`

---

## 1. The ranked receipt

| field | value |
|---|---|
| submission | `180db84` |
| commit | `b566c594da5a32a4d23f7ff955644...` |
| status | **rejected**, "score did not improve current best" |
| published score | **3.70465399491642** |
| our best (anchor `5a9f130a`, commit `8ba6e738`) | 3.70784519415395 |
| delta | **−0.003191 = −0.0861 %** |
| draw sd on this board | 0.172 % to 0.188 % |

The receipt carried the maintained base: E162 arm A (software-pipelined affine `qmm_t` k-loop), E162 arm B (the same double buffer in `quantized_nax`, which **no host we own can ever execute**), and PR 159's default-off `MLX_E159_FIXED_DRAFT_DEPTH` flag.

Reading, in the order the assignment asked for:

1. **As a score attempt: failed.** −0.0861 % against our own best, half a draw sd. The assignment predicted roughly a 2 % chance of the crown and told me not to expect it.
2. **As the first ranked measurement of E162: no detectable effect, either arm.** Arm A was predicted at `+0.197 %` published by three independent routes. The receipt came in 0.086 % low. A one-draw receipt against a 0.18 % draw sd cannot resolve 0.2 %, so the honest statement is **"not resolved", not "arm A does not work"**. Arm B remains unmeasured in the sense that its effect is inside the noise of a single receipt; it is now at least *measured once*, which was impossible before.
3. **As a near-replicate of `5a9f130a`:** with arm A's expected `+0.20 %` subtracted, the residual is about −0.29 % candidate-leg-equivalent. That is a second point on the same-build noise question and it sits well inside the one-pair SE of 0.7759 % it exists to constrain.

**CAMPAIGN LAW 364 check:** the accept ledger was not moved by the kernel change; the receipt's per-prompt draft geometry matched the anchor prompt-for-prompt, so the timing delta is a runtime contrast and not an acceptance contrast.

---

## 2. Pre-submission gates, all passed

One invocation, `MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit`, no 128-token warm-up run:

```
git_head            a69e3de131d626e72503e1a5510c489634603318   git_dirty=0
worker_sha256       a0472f8e96fe3616c388d3c0a5e897e86803d492dbaebecd64ef8a85f6ecc071
metallib_sha256     5d6e3f9f1438dc5d6e99423b720d66676b08a3c92a204c1e3fd8b6b0ecaef91d
head_provenance     cfffe00c62b791c4246be345662f4b320d77931d7e513e130078f7748e3787e0
decode_tokens 512   mtp_depth 8
all_tokens_matched  true     residual_divergence_count 0     public_drift_tripwire_passed true
serial 0.073463724693283439   mtp 0.031443699263036251   ratio 2.336357566542559
effective_mean_draft_len 6.3766233766233764   accepted_draft_rate 0.88594704684317716
MLX_E159_FIXED_DRAFT_DEPTH unset      MLX_E141_ROWS_PER_LEAF unset
```

- Scored-surface delta against the true anchor proxy `3c2316ea`: **exactly the five files the assignment named**, no unnamed scored change. The advisor's suggested fallback anchor `6543cfdb` was wrong (it carries leaf16) and I substituted `3c2316ea`, which ledger 337.3 documents as differing from `8ba6e738` only by telemetry.
- `MLX_E159_FIXED_DRAFT_DEPTH` proven inert from source: `pinnedDraftDepth` is `nil` when unset, the `if let` body never runs, `draftPolicy` keeps the adaptive `costModelDepth` closure. Two references in `Sources/`, both inside the guarded block. `edl 6.3766` is not an integer, which is the run's own evidence that the adaptive schedule was live.
- `python3 research/twin_audit.py`: `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)`.
- `swift test --force-resolved-versions`: 739 tests, 69 suites, 41 issues — digit-identical to the recorded baseline, zero added.
- Assignment scope, editable budget and ranked-score-boundary scripts: clean.

---

## 3. Local timed sessions

All three are gated ABBA sessions over the full 512-token window in one thermal session, real cool gate every leg.

| session | W&B run | arms | headline |
|---|---|---|---|
| integrated E162 effect | `cqfa2d6n` `e167-maintained-base-local-evidence` | P = pre-E162, C = this base | `decode_seconds` −0.6309 % (t = −5.70); `decode_only_spt` −0.0711 % (t = −0.57); the saving is in prefill, as designed |
| F5/F9 revert of arm A | `gy849f1o` `e167-f5-revert-abba-512` | C = pipelined, P = reverted | `decode_only_spt` +0.3034 % for pipelined (t = 1.59, n.s.); token-neutral, `all_legs_matched` true, `edl 6.376623`, 435 accepted, 77 rounds |
| four-arm counter strip | `e167-arms-20260824T075454Z` | B0/B1/B2/B3 | §5 below |

The integrated effect is **not** materially worse than arm A's isolated −1.9471 % prefill saving; it is concentrated in the leg's prefill term exactly as the mechanism predicts, and the decode leg is untouched. No stop rule fired.

---

## 4. The two findings that came out of the waiting time, and which the advisor adopted

### 4a. The byte-identical null pair — we were never behind on code

The crown `ec24d591` (newjordan, 3.729110011) and our `5a9f130a` (3.707845194) ran **the same scored bytes**, and published **0.5735 % apart**. I verified the identity from `effective_mean_draft_len` and `non_drafting_round_count` on all eight prompts rather than trusting a note.

Consequence: every plan that priced "the crown is 0.57 % ahead of our best" as an engineering deficit was pricing a **host draw**. That premise is retired.

Quantitatively, from a resample of the serial-leg population (n = 965 legs):

- The crown's own code typically publishes **3.7104**; its 3.7291 is the p99 of its own distribution.
- Serial-leg draw sd **0.179 %**, which I flagged as a **floor, not an estimate**.
- `corr(serial, prefill) = +0.514`, which demotes the serial leg as a clean drift proxy.

Advisor corroboration (F16), from data I had not seen: newjordan has 74 scored receipts whose 5th and 6th best are **3.7103 and 3.7100** — my 3.7104 is their modal top draw. For 74 draws the expected maximum is ≈ +2.5 sd ≈ 3.7271 against the observed 3.7291. A cross-solver pack of 47 receipts ≥ 3.700 from 13 solvers has sd **0.172 %**, within 4 % of my independent 0.179 %. **The crown is fully explained as the maximum of 74 tickets with no code advantage**, and the confidence stays at 90 %, not 99 %.

Two corrections I made in the same pass: Askeladd's 0.125 % figure is right for the 8-prompt mean at 0.1438 % but understates the per-prompt figure by 2x at 0.2446 %; and I withdrew my own earlier 2.3 σ claim in favour of 1.4 σ.

### 4b. The E165 audit that released the crown gate

Question, from F13: does E165's 443 added lines in `Qwen36MTPBlockSession.swift` contain any counter, global write, trace-field interpolation, or other side effect on the per-round or per-routed-cell path, and is the trace-line interpolation genuinely inside a trace guard on every path?

**Answer: no per-routed-cell side effect, and the interpolation is guarded on every path.** The advisor verified the `Qwen35.swift` side independently and our reads agree: on the submitted surface `0863b06a → 58979332` is six files, the four `quantized*` files carry all of Q, and the `Qwen35.swift` 78/10 is 100 % instrumentation including six changed default arguments that all belong to off-path `qwen35Verify*` / `qwen35Bench*` / `*PositiveControl` entry points. No `senpai/` invariant script depends on the stripped symbols.

The gate on PR 171 was released on this audit and Thorfinn proceeded.

One real find fell out of it and is **not** shipped, correctly: the `preflushBacklogHidden` alias defeats `removeAll(keepingCapacity: true)`, a genuine deoptimisation, and the three `preflush*` fields are write-only on the ranked arm. At ~0.01 % it is two orders of magnitude below the freeze threshold and ship mode takes no new change on a frozen branch. Logged as next-round work.

---

## 5. The four-arm counter-strip session

Session `e167-arms-20260824T075454Z`, 07:54:54Z → 08:33:47Z, 38.9 min, HEAD `43a95f0a`, `git_dirty=0`, 16/16 legs, per-leg W&B publication (RULE 374, first application).

Arms, relative to organizer parity, in the 257-cell-per-round routed call `Qwen35CustomQMV.matmul`:

| arm | change | writes | public decls | worker sha12 | `nm` xsums / addressors / anchor |
|---|---|---|---|---|---|
| B0 | maintained base, unmodified | yes | yes | `999b5b51429b` | 8 / 4 / 372 |
| B1 | the two counter **writes** removed | no | yes | `f8a07f6d832a` | 8 / 4 / 372 |
| B2 | both files at **organizer parity** (the ship strip) | no | no | `ddfa2f3d0aee` | 0 / 0 / 362 |
| B3 | the two globals made **internal** | yes | no | `df28b41ab921` | 4 / 0 / 372 |

Shared metallib `5d6e3f9f1438dc5d` and one shared fingerprint record for all four arms, so the worker binary is the only difference.

### Validity

| gate | result |
|---|---|
| arm identity | each arm ran exactly one witnessed binary across all four of its legs |
| exactness | `all_tokens_matched` true in **16/16** legs against **one** B0-generated reference (`66858d9561663b62`, 513 rows) — a genuine four-way cross-arm bit-exactness test |
| row ledger | every leg `568 = 435 accepted + 56 rejected + 77 tail`, `parity_all_ok`, `residual_divergence_count = 0` |
| schedule | `B0 B1 B2 B3 B3 B2 B1 B0` ×2, all four position sums 34, balance 4/4/4/4 |
| thermal | real gate every leg, entry 37.9–40.0 °C |

**Ship-relevant:** B2 is exactly the tree Thorfinn ships. Over a full gated 512-token window it produces bit-identical tokens and a bit-identical row ledger to the maintained base; `edl`, `accepted` and `rounds` have sd **exactly 0** across all four arms. The strip changes no behaviour.

### Primary contrast, `decode_only_spt`, sign flipped so positive = tax removed

| arm | tax removed | 95 % CI | t | df | p |
|---|---|---|---|---|---|
| B1 writes removed | +0.139 % | −0.097 … +0.374 | −1.51 | 5.08 | 0.19 |
| B2 organizer parity | +0.252 % | −0.163 … +0.668 | −1.73 | 3.76 | 0.16 |
| B3 linkage internal | +0.208 % | +0.057 … +0.358 | −3.48 | 5.29 | **0.016** |

The drift-cancelled blocked estimator agrees in sign and magnitude with pooled for all three arms; B1's per-block range straddles zero, B2's and B3's do not.

### Scoring the predeclared reading table

- **F15 prediction "B1 flat": consistent.** **My prediction "B1 moves": not supported.** My argument was that a module-visible side effect on a `public nonisolated(unsafe)` static defeats the optimiser's effects summary at all 257 call sites regardless of the `m ≥ 4` branch. This session neither supports nor refutes it — it cannot see an effect that size. I withdraw the confidence I attached to the mechanism.
- **F15 prediction "B2 near 1 %": refuted at this fixture.** 0.985 % is outside B2's interval, whose upper bound is 0.668 %.

That refutation is a test of **which transfer model is right**, not of whether the tax is real:

| model | prediction here | verdict |
|---|---|---|
| tax is ≈1 % **of the decode leg** | 0.985 % | **excluded** |
| tax is **592 µs per drafting round** (F15 constant) | 0.373 % | **consistent** (measured 0.252 %, and 0.250 % at the constant's −1 sd) |

This fixture runs 77 drafting rounds in 12.205 s of decode-only time, a density of 6.31 rounds/s against the ranked table's 2.53–27.54. **The per-drafting-round form transfers across host generations; the per-decode-percentage form does not.** Caveat: this cannot separate "the constant transfers" from "gen 16 is simply cheaper", since I have no gen-17 host.

### Common-mode correction, both readings reported

Prefill runs at m = 512, outside `widths = 2...9`, so no arm can tax it — a built-in null control.

| arm | decode tax | prefill tax (should be 0) | corrected |
|---|---|---|---|
| B1 | +0.139 % | +0.048 % | +0.091 % |
| B2 | +0.252 % | +0.169 % | +0.083 % |
| B3 | +0.208 % | +0.121 % | +0.087 % |

No prefill contrast is significant (p 0.19–0.69), matching the ranked prefill regime at −0.23 %. But the point estimates are not zero and subtracting them collapses all three arms onto 0.083–0.091 %. Prefill's own sd is 0.2 %, so at n = 4 the correction adds about as much noise as it removes. **Both readings belong in the record and I do not pick one.**

### Two design faults recorded against myself

1. **A palindrome equalises the first moment of position, not the second.** `B0 B1 B2 B3 B3 B2 B1 B0` puts B0 at the block edges and B3 at the centre (mean distance from centre 3.5 / 2.5 / 1.5 / 0.5). ABBA cancels *linear* drift and is biased by *curvature*; a convex within-block profile predicts B1 < B2 < B3 all faster than B0, which is the observed shape. The thermal channel argues against it — B3 entered hottest and ran fastest, B0 entered coolest and ran slowest, exits flat at 56.5–56.6 °C — and a sensitivity dropping the coolest, slowest leg keeps the ordering (B2 0.209 %, B3 0.164 %, B1 0.095 %). The fix for a future session is a Latin square or randomised positions, not a longer palindrome.
2. **The decomposition was never affordable at this fixture's density,** and I could have computed that in advance from the F15 table. Pooled within-arm sd is 0.169 % of the mean; separating B1 from B2 needs n ≈ 39 per arm, i.e. 155 legs and ~6.2 h. I ran 4.

### Four instrument defects found and fixed during this experiment

All four are the same shape — **a check that silently degraded to a no-op**:

1. the discarded objdump witness;
2. the addressor probe matching nothing (`_Sivau` against a name ending `...HitsSivau`), which would have passed B3 for the wrong reason — caught by adding a positive control that makes B0/B1 fail the build if the probe returns 0;
3. the B2 restore leaving the tree at organizer parity because `git checkout upstream/main -- <paths>` writes the index and the EXIT trap restored from it;
4. the staged arms never carrying the `mlx.metallib.fingerprint` sidecar, so the vendored-Metal-source check passed vacuously on every leg.

A fifth operational defect cost 20 minutes of wall time after the 07:30Z host reprovision: `setup.sh` provisions the reference checkpoint and never transforms it, so `./benchmark.sh --transform-only` is a separate recovery step. The session script now checks `weights/config.json` in its preflight and names that command. Full host recovery measured at **8 minutes**, checkpoint re-download at 105 MiB/s.

---

## 6. Ranked context that arrived after the session, flagged and not owned by me

Thorfinn's crown attempt `2c885d6` (strip + E165 on organizer main, commit `3f9b3e3`) is now terminal: **rejected at 3.65820901018537**, −1.33 % against our best. The pair receipt `fda590b` published 3.66784730800019.

Under the projection that the strip returns roughly +1 % of the decode leg, that attempt was priced well above our best. It landed 1.33 % below it, which is about 7 draw-sd from the projection. That is one draw of a build I did not construct and whose deltas I have not audited, so I make no claim about the cause. I report it because it is the ranked test of the same transfer question my §5 answers locally, and both point the same way: **the per-decode-percentage form of the instrumentation tax does not price a real submission.** The receipt is Thorfinn's to interpret.

---

## 7. What is open, and what I recommend

- **Writes versus linkage is unresolved.** Constrained by the ranked three-regime decomposition (prefill −0.23 % ≈ drift, m = 1 +0.131 %, drafting m = 2..9 +1.230 %, decode overall +0.985 %, t = 5.2), which kills every global mechanism and confines the tax to the routed call path, the honest state is: **the tax is real, its per-drafting-round form transfers, and whether it is the writes or the linkage is open.** That is Alphonse's E173 (a)-marker versus (b)-admission-path fork.
- **Do not spend 6.2 h of this host on 155 legs.** A full decode session is the wrong instrument: ~96 % of its wall time carries no exposure to the mechanism. The right instrument is a direct microbenchmark that calls the routed path at m = 2...9 in a tight loop, which would separate B1 from B3 in minutes. I have not built it and will not start it without an assignment.
- **Standing caveat on everything local here:** M4 Pro, gen 16, `_nax` never executes, and this fixture drafts at ≈6.38, so the session measures the `tablePays` regime only and says nothing about m = 2 or m = 3.

## Artifacts

- `research/e167-artifacts/four-arm-session.txt` — session header and witnesses
- `research/e167-artifacts/four-arm-legs.tsv` — all 16 legs as recorded
- `research/e167-artifacts/four-arm-verdict.txt` — the analyser output reproduced from the two files above
- `research/e167_arms_abba.sh`, `research/e167_build_arm.sh`, `research/e167_arms_verdict.py`, `research/e167_wandb_leg.py` — the harness, arm builder, analyser (with `--self-test`) and per-leg publisher
- W&B: `e167-arms-20260824T075454Z`, `gy849f1o`, `cqfa2d6n`, all in `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`
