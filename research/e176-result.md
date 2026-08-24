# E176 result: Q has no decode consumer, and receipt C is the outlier

`assignment_id` `e176-q-decode-consumer-census`, revision `r0`.
Base `senpai/qwen38-mtp-r1` at `6368dc25265bd05461df4f070a18f4d97a18bc12`.

W&B: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/vxdn8h60>
(run `vxdn8h60`).

Every number carries a harness label. No GPU decode leg was timed. Task 4, the
local GPU probe, was not needed: the desk answer at the routing threshold is
not close.

## 1. Cell census (`harness=source`)

Reproduce with `python3 research/e176_cell_census.py`.

### 1.1 One entry point, one routing decision

Every quantized projection in the scored decode path goes through exactly one
Swift entry point, `Qwen35Ops.linear`
(`Sources/MLXFastModel/Qwen35Ops.swift:36`), which calls MLX
`quantizedMM(transpose: true)`. MLX then decides in
`QuantizedMatmul::eval_gpu`
(`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415`):

```cpp
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) { /* qmm_splitk or qmm */ return; }
dispatch_qmv(...);
```

Q (RULE 377) edits only `qmm_t`, `qmm_t_nax` and `qmm_t_splitk`. A cell
consumes Q only if it reaches `M >= vector_limit` with `transpose_ == true`.

### 1.2 The thirteen cells

497 quantized-projection calls per decode round, 13 distinct cells. Shapes
follow from `fixtures/qwen3_6_27b_config.json`: hidden 5120, intermediate
17408, vocab 248320, 24 attention heads and 4 KV heads at head_dim 256, GDN
16 key heads and 48 value heads at dim 128, 48 GDN layers and 16
full-attention layers.

| cell | K | N | calls/round | source |
| --- | --- | --- | --- | --- |
| gdn.in_qkv | 5120 | 10240 | 48 | `Qwen35GatedDelta.swift:241` |
| gdn.in_z | 5120 | 6144 | 48 | `Qwen35GatedDelta.swift:245` |
| gdn.in_b | 5120 | 48 | 48 | `Qwen35GatedDelta.swift:254` |
| gdn.in_a | 5120 | 48 | 48 | `Qwen35GatedDelta.swift:255` |
| gdn.out | 6144 | 5120 | 48 | `Qwen35GatedDelta.swift:346` |
| attn.q | 5120 | 6144 | 16 | `Qwen35Attention.swift:143` |
| attn.k | 5120 | 1024 | 16 | `Qwen35Attention.swift:162` |
| attn.v | 5120 | 1024 | 16 | `Qwen35Attention.swift:163` |
| attn.o | 6144 | 5120 | 16 | `Qwen35Attention.swift:211` |
| mlp.gate | 5120 | 17408 | 64 | `Qwen35MLP.swift:28` |
| mlp.up | 5120 | 17408 | 64 | `Qwen35MLP.swift:29` |
| mlp.down | 17408 | 5120 | 64 | `Qwen35MLP.swift:30` |
| lm_head | 5120 | 248320 | 1 | `Qwen35FastEngine.swift:263` |

**Correction to my own interim comment.** That comment cited sites in a file
called `Qwen35.swift` near lines 1737 and 1904. No such file exists in this
tree. Those references were wrong. The table above replaces them and I read
every row cell by cell.

### 1.3 The limit is one constant per host class

`get_qmv_batch_limit(D, O, d)` (`quantized.cpp:84-125`) is guarded on
`arch_gen == 13 || arch_gen == 14`, and each size branch tests
`D <= 2048 && O <= 2048`, then `D <= 4096 && O <= 4096`, then falls through.

**Every cell has K >= 5120 > 4096, and `D` is `K`.** No cell can take either
narrow branch, so the limit does not depend on `N` at all:

| host class | limit, all 13 cells | first M that consumes Q | provenance |
| --- | --- | --- | --- |
| gen 13/14, size s | 6 | **6** | M1/M2 class, not in this campaign |
| gen 13/14, size d | 12 | none at M <= 9 | Ultra |
| gen 16, size s | 10 | none at M <= 9 | **first-hand probe, this host** |
| gen >= 17, size s | 10 | none at M <= 9 | ranked M5 |
| gen >= 17, size d | 12 | none at M <= 9 | Ultra |

`research/e176_arch_probe.swift` reads Metal device metadata only. On this host
it reports `applegpu_g16s`. `device.cpp:565-572` parses `arch_gen` from the two
digits before the final character, so that is gen 16, size `s`. The ranked M5
must satisfy `is_nax_available`, which requires `arch_gen >= 17`
(`device.cpp:913-925`), so it takes the same fall-through and the same limit
of 10.

### 1.4 Required statement

Maximum decode width is `M = 1 committed primary + 8 drafts = 9`.

**The set of Q-consuming cells at `M <= 5` is EMPTY, and it is equally empty at
`M = 6, 7, 8, 9`, on this host and on the ranked M5.** Q's only live consumer
inside a timed leg is the 512-token seed prefill.

The tree already says so independently. `Qwen36MTPBlockSession.swift:1049` and
`:1658` both record that projections at `M` in 6..9 stay on the per-row-exact
QMV dispatch because the host limit is "10+ on this generation for these
shapes".

### 1.5 The small-N hypothesis is refuted cell by cell

F2 relayed a characterization in which Q's decode consumers are the
non-routable small-`N` cells, for example a kv projection at `n=1024` that
fails an `n >= 4096` test. Three separate parts of that do not hold here.

1. There is no custom quantized matvec replica in this tree. The only
   `MLXFast.metalKernel` uses are the two linear top-two evidence kernels at
   `Qwen36MTPBlockSession.swift:2209` and `:2261`. Every projection reaches MLX.
2. `get_qmv_batch_limit` gates on a **conjunction** of `D` and `O`. attn.k and
   attn.v at `N=1024`, and gdn.in_b and gdn.in_a at `N=48`, all still have
   `D = 5120 > 4096`, so they land in the same fall-through and get the same
   limit of 10.
3. Falling through to MLX does not reach Q anyway. At `M < vector_limit` the
   dispatch is `dispatch_qmv`, which Q does not touch. Even the
   `transpose_ == false` path, whose limit is the constant 4, reaches
   `affine_qmm_n`, and Q does not touch `qmm_n` either.

### 1.6 The M = 9 row, for the reopened declamp question

At `M = 9` on gen 16 and on gen >= 17, the count of cells that reach `qmm_t` or
`qmm_t_nax` is **0 of 497**. A declamp to `M = 9` cannot create a Q consumer
and cannot be priced through Q.

The only host class where `M = 9` crosses the limit is gen 13/14 size s, where
the limit is 6 and **all 497 calls** switch at `M >= 6`. That is a cheap,
falsifiable explanation for a local step law that does not transfer: if any
local host reports gen 13 or 14, its `M >= 6` behaviour is a different kernel
family from the ranked runner's. Running `research/e176_arch_probe.swift` on
each student Mac settles it in seconds.

## 2. Per-prompt null calibration (`harness=ranked`)

Board snapshot: 976 scored 512-token receipts, 192 distinct schedule
fingerprints, 115 in the organizer-main-schedule cluster.

Scalar widths:

- Per-receipt common offset, robust sd **0.1421 %**; pairwise 0.2011 %.
- Per-prompt serial residual sd 0.155 % to 0.171 %.
- **Adopted per-prompt pairwise 1 sigma 0.298 % to 0.314 %**, from
  `sqrt(2 * serial_resid_p^2 + common_null^2)`.
- Contaminated ceiling from the candidate cluster 0.68 % to 0.97 %.

### 2.1 Sign structure of the byte-identical pair, crown `ec24d591` vs A `5a9f130a`

| prompt | serial % | prefill % | decode % | leg % |
| --- | --- | --- | --- | --- |
| plutarch | −0.0935 | −0.2456 | −0.0049 | −0.0131 |
| drama | −0.2427 | −0.1009 | −0.5710 | −0.5440 |
| travel | −0.2137 | −0.3742 | −0.4705 | −0.4641 |
| beagle | −0.1513 | −0.2499 | −0.2473 | −0.2475 |
| republic | −0.4306 | −0.1471 | −0.1742 | −0.1713 |
| essays | **+1.1256** | −0.3802 | −0.1459 | −0.1705 |
| medicine | −0.0075 | −0.0218 | −0.0927 | −0.0852 |
| botany | +0.0541 | −0.2373 | −0.1748 | −0.1815 |
| **mean8** | **+0.0051** | **−0.2196** | **−0.2351** | **−0.2346** |
| **sd8** | 0.4768 | 0.1254 | 0.1916 | 0.1816 |
| **negative** | 6/8 | **8/8** | **8/8** | **8/8** |

This is the decisive property, and a scalar sigma hides it. **The null is not
sign-symmetric.** Two byte-identical candidate builds differ by a coherent
whole-receipt offset of about −0.23 % that appears on all eight prompts in the
candidate channel, with a per-prompt spread of only 0.18 % around it.

Two consequences.

1. A whole-receipt shift of about +-0.25 % in the candidate channel is
   **ordinary**, and averaging over prompts does not reduce it. Any
   cross-receipt claim smaller than that is unsafe.
2. The serial channel does not carry the same offset: mean8 +0.0051 %, 6/8
   negative, with one large excursion at essays of +1.1256 %. The serial leg is
   byte-identical by construction, so that essays value is pure measurement
   noise, and it is why essays is a weak lever throughout this analysis.

## 3. Mechanism fit (`harness=ranked`)

One-parameter fits to the `B - C` vector, 7 dof: prefill (34.7 +- 6.6 ms/leg,
chi2 14.38), round (249.9 +- 56.3 us/round, chi2 22.41), uniform
(−0.5754 +- 0.1093 %, chi2 14.41). The round model is refuted: it needs
plutarch at −0.82 % and plutarch measures +0.05 %, a +3.1 sigma miss.

Two-parameter fit: per leg **+35.5 +- 12.5 ms** (2.8 sigma), per round
**−7.9 +- 107 us** (0.1 sigma, consistent with zero, 95 % upper bound
218 us/round).

**Essays as a falsification lever: it does not work, in either direction.**
Essays sits at −0.39 sigma against zero, and the three one-parameter models
predict −0.687, −0.417 and −0.575 %, a spread of 0.29 pp that is inside the
per-prompt sigma. Section 2.1 explains why: essays carries the largest serial
excursion in the byte-identical control. Plutarch is the only prompt that
discriminates, because its edl of 0.156 makes round count and prefill share
diverge.

### 3.1 The channel split, with no model

Each receipt carries per-prompt `prefill_seconds_per_token`, so
`leg = 512 * mtp_seconds_per_token_mean`,
`prefill = 512 * prefill_seconds_per_token`, `decode = leg - prefill`.

Prefill share of the candidate leg: plutarch 3.42 %, drama 5.71 %, travel
6.47 %, beagle 9.51 %, republic 10.49 %, essays 10.42 %, medicine 10.47 %,
botany 10.60 %; mean7 9.10 %.

`B - A` is the clean Q instrument, because A is organizer-pure and both trees
were inspected first hand:

| prompt | prefill % | decode % | leg % |
| --- | --- | --- | --- |
| plutarch | +1.9225 | −0.0579 | +0.0098 |
| drama | +2.1709 | −0.1685 | −0.0340 |
| travel | +1.7964 | +0.1604 | +0.2684 |
| beagle | +1.7070 | −0.0839 | +0.0888 |
| republic | +2.0347 | +0.1444 | +0.3450 |
| essays | +1.9133 | +0.0178 | +0.2167 |
| medicine | +2.0869 | +0.0709 | +0.2846 |
| botany | +1.8552 | +0.0391 | +0.2331 |
| **mean8** | **+1.9359** | **+0.0153** | **+0.1766** |
| **sd8** | 0.1544 | 0.1136 | 0.1379 |
| **negative** | **0/8** | 3/8 | 1/8 |

Q's prefill effect is 8/8 the same sign. Q's decode effect is 3/8 negative with
a mean of +0.0153 % and a spread of 0.11 %, which is the sign-symmetric null of
section 2.1 with no offset at all. That is what "no consumer" looks like in a
measurement.

Prefill-only closure test, mean7, where a seed-only factor must satisfy
`leg % = prefill % * prefill share`:

| pair | content | prediction | measured leg | residual |
| --- | --- | --- | --- | --- |
| B − A | Q + I | +0.1782 | +0.2004 | **+0.0222** |
| C − A | I | −0.0231 | +0.8765 | +0.8996 |
| B − C | Q as previously derived | +0.1996 | −0.6691 | −0.8687 |
| D − B | E165 − I | +0.0061 | +0.6878 | +0.6817 |
| crown − A | byte-identical null | −0.0199 | −0.2663 | −0.2464 |

Only `B - A` closes, and it closes at the null scale.

**Q is prefill-only: +1.94 % of prefill, which is +10 to +12 ms per leg, and a
decode effect indistinguishable from zero. Net candidate leg +0.20 % slower.**
This reproduces Edward's E175 local +1.936 %, 8/8 same sign (`harness=local`,
W&B `8bc65oel`) to three digits, so the prefill penalty transfers from local to
ranked.

## 4. The quantitative implication: receipt C is the outlier

FINDING 450 makes the Q decode term identically zero. Then `B - C` must equal
its prefill-only prediction, and it does not:

| quantity | value |
| --- | --- |
| prefill share, mean7 | 9.0960 % |
| `B - C` prefill, mean7 | +2.1947 % |
| predicted `B - C` leg | **+0.1996 %** |
| measured `B - C` leg, mean7 | **−0.6691 %** |
| **unexplained, located in C** | **−0.8687 %** |

Q-prefill pushes B the wrong way for FINDING 440: it makes B *slower* than C by
about +0.20 %, not faster by 0.67 %. So `B - C` needs a non-code component of
about **−0.87 %** sitting in receipt C.

The independent check agrees. `C - A` is an inert-instrumentation-only
contrast, yet its decode channel is **+0.9889 % mean7** while its prefill
channel is −0.2315 %. Against the whole-receipt common-offset sigma of
0.1421 % that is **7.0 sigma**, and it is about four times the −0.23 % coherent
offset that section 2.1 measures between two byte-identical builds. `C - A`
decode is 7/8 positive with a spread of 0.56 %, so it is not one bad prompt.

**Conclusion: receipt C is an outlier.** Its candidate decode channel is slow by
about +0.9 % for a reason that is not in its diff. FINDING 440's attribution of
−0.6691 % to Q is therefore conditional on receipt C, and the E175 receipt is
the test.

The `+0.0070 %` additivity residual that validated FINDING 440 cannot detect
this. `(C - A) + (B - C) + (D - B) = D - A` is an identity in receipt values and
holds for any C, however anomalous.

This converges with Edward's independent C/A anomaly of −1.08 % published on an
inert-only contrast. It would also re-open D's E165 verdict, but only after the
E175 receipt confirms the model. One step at a time.

## 5. Registered prediction for E175 (`15017ddf`, frozen `9c4fefe8`)

Applying the measured per-prompt `B - A` channel factors to A's own per-prompt
legs and recomputing the published median gives **3.702087**, against A's
`3.70784519415395`, a change of **−0.155 %**.

| model | predicted published |
| --- | --- |
| advisor, ranked route | 3.7328 / 3.7267 |
| Edward, prefill-adjusted | 3.7218 / 3.7157 |
| **this census, model 3** | **3.7021**, band 3.700 to 3.704 |

The three models are separated by far more than the +-0.25 % whole-receipt null
of section 2.1, so the receipt discriminates cleanly. If E175 lands near 3.702,
Q is a net cost, it should leave the ship set, and receipt C is confirmed as the
outlier. If it lands near 3.727, this census is wrong about the live call path
and section 1 must be re-derived.

## 6. Open conflict: FINDING 444

FINDING 444 derives a local `Q` leg effect near **−0.88 %**. At the 23.4 %
local prefill share, and with the measured local prefill penalty of +1.936 %,
that requires about **−1.33 % of local decode**. Section 1 forbids that channel
on any gen 16 or gen >= 17 host.

The two results cannot both be right. Two measurements settle it, in this
order.

1. **Free, seconds.** Run `research/e176_arch_probe.swift` on the host that
   produced FINDING 444. If it reports gen 13 or 14, the limit there is 6, all
   497 calls switch to `qmm_t` at `M >= 6`, and FINDING 444 is real but
   host-specific and non-transferable. If it reports gen 16, the local decode
   channel is as empty as the ranked one and FINDING 444 is an artifact.
2. **One ABBA-counterbalanced local session.** Organizer-pure against
   organizer-pure plus Q, on one host, with **prefill seconds and decode
   seconds logged separately** instead of only the leg total. The existing local
   evidence reports only the leg, and that is exactly the aggregation that lets
   a +1.94 % prefill penalty and a spurious decode term hide in one number.

## 7. cap-4 (`harness=ranked`, moot for composition)

Recorded because I measured it before F2 declared the composition dead.

Receipt `90c131dc` is organizer main with one literal changed,
`segmentedVerifyDepthCap 7 -> 4`. Rejected at **3.54742900664627**, which is
**−4.33 %** against A at `3.70784519415395`.

| prompt | A edl | cap-4 edl | prefill % | decode % | leg % |
| --- | --- | --- | --- | --- | --- |
| plutarch | 0.156 | 0.156 | −0.2014 | −0.1145 | −0.1175 |
| drama | 2.298 | 2.298 | +0.1513 | −0.2175 | −0.1963 |
| travel | 2.648 | 2.591 | +0.2160 | −0.0468 | −0.0295 |
| beagle | 4.382 | 3.292 | −0.1937 | **+4.6343** | +4.1686 |
| republic | 4.989 | 3.562 | −0.1887 | **+5.6161** | +5.0001 |
| essays | 5.087 | 3.525 | +0.0070 | **+5.1855** | +4.6421 |
| medicine | 5.256 | 3.597 | +0.0173 | **+4.2324** | +3.7856 |
| botany | 6.148 | 3.827 | −0.0980 | −0.3191 | −0.2955 |
| **mean** | | | **−0.0363** (8) | **+2.7264** (7) | **+2.4393** (7) |

cap-4 is a clean decode-channel factor: prefill flat within the null on all
eight prompts, and the whole cost in decode. It is the exact mirror of Q, which
is prefill-only.

**Ruling, for the record: do not compose.** Q pays nothing at `M <= 5` because
it pays nothing at any decode `M`. cap-4 moves only the decode denominator, the
one channel Q does not touch, so composition cannot create a Q consumer and Q's
fixed prefill penalty rides on top unchanged. Composing the measured `B - A`
factors onto cap-4 gives a published median of **3.542212**, a further
−0.147 %.

**Anomaly worth its own question.** botany is clipped hardest, edl 6.148 to
3.827, and is the one clipped prompt that does not regress, at −0.32 % decode,
while beagle, republic, essays and medicine are clipped less and lose 4 % to
6 %. Whatever makes botany's deep drafts worthless is a real signal about the
adaptive depth walk, and it may be useful to Askeladd's E177 cost(M) fit.

## 8. Reproduction

```bash
python3 research/e176_cell_census.py                 # task 1, source only
YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
python3 research/e176_q_consumer_census.py           # tasks 2 and 3
WANDB_API_KEY=... python3 research/e176_wandb.py
swift research/e176_arch_probe.swift                 # Metal metadata only
```

## 9. Suggested follow-ups, not implemented

1. Run `research/e176_arch_probe.swift` on every student Mac. It is free and it
   either explains or kills FINDING 444 and the non-transferring local step law
   in one reading.
2. Re-audit every finding derived from receipt C `fda590bb`. The additivity
   check that validated FINDING 440 is an identity and cannot detect a
   contaminated leg.
3. Report the null as a coherent whole-receipt offset of about −0.23 % plus a
   0.18 % per-prompt spread, not as a single symmetric sigma. Sign structure
   across the eight prompts is a stronger test than any mean.
4. A QMM-path win needs decode `M >= 10`, which the eight-draft cap forbids, or
   a change to the qmv path instead. Further `qmm_t` work cannot move ranked
   decode at all.
