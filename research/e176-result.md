# E176 result: Q has no decode consumer, and the cap-4 composition is refused

`assignment_id` `e176-q-decode-consumer-census`, revision `r0`.
Base `senpai/qwen38-mtp-r1` at `6368dc25265bd05461df4f070a18f4d97a18bc12`.

W&B: <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/qpty8v51>
(run `qpty8v51`). Run `hjgb79i4` is an earlier upload of the same analysis. It
put the `B - C` channel numbers under `B - A` column names, so read `qpty8v51`
only.

Every number below carries a harness label. No GPU decode leg was timed for
this experiment. Task 4 was not needed, because the desk answer at the routing
threshold is not close.

## 1. Consumer census (`harness=source`)

### 1.1 FINDING 390 is wrong

`get_qmv_batch_limit` in `backend/metal/quantized.cpp:84-125` returns the
`6 / 10 / 14` limits that FINDING 390 quotes **only** inside
`if (arch_gen == 13 || arch_gen == 14)`. Every other generation returns
`10 / 12 / 18`, and `arch_size == 'd'` returns `32 / 18 / 12`.

| branch | limits | minimum `vector_limit` | provenance |
| --- | --- | --- | --- |
| `arch_gen` 13 or 14 | 6 / 10 / 14 | 6 | not the ranked runner |
| `arch_gen` 16, `arch_size` s | 10 / 12 / 18 | 10 | first-hand probe, this host |
| `arch_gen` >= 17 | 10 / 12 / 18 | 10 | inferred for ranked M5 |
| `arch_size` d | 32 / 18 / 12 | 12 | not this chip |

`research/e176_arch_probe.swift` reads Metal device metadata only. On this host
it reports `architecture applegpu_g16s`, so `arch_gen` is 16 and `arch_size` is
`s`, and the limit is 10 for every Qwen cell. The ranked M5 must satisfy
`is_nax_available` (`device.cpp:913-925`), which requires `arch_gen >= 17`, so
the ranked runner takes the same `10 / 12 / 18` table.

**The minimum `vector_limit` over every branch in the function is 10.**

### 1.2 Decode never reaches the QMM threshold

Maximum decode width is `M = 1 primary + 8 drafts = 9`. `9 < 10`, so every
transposed decode cell routes to `dispatch_qmv`. It never reaches `qmm`,
`qmm_nax`, or `qmm_splitk`.

Wide quantized matmul call sites reached once per decode round, all
affine-4 group-64, `transpose = true`, `x = [1, M, K]` row contiguous:

| site | calls / round | K | N | source |
| --- | --- | --- | --- | --- |
| `mlp.gate_up` | 64 | 5120 | 34816 | `Qwen35.swift:1737` |
| `mlp.down` | 64 | 17408 | 5120 | `Qwen35.swift:1741` |
| `gdn.in_proj` | 48 | 5120 | 16480 | `Qwen35.swift:1743` |
| `gdn.out_proj` | 48 | 6144 | 5120 | `Qwen35.swift:1744` |
| `fa.qkv` | 16 | 5120 | 14336 | `Qwen35.swift:1746` |
| `fa.o_proj` | 16 | 6144 | 5120 | `Qwen35.swift:1747` |
| `lm_head` | 1 | 5120 | 248320 | `Qwen35.swift:1737` |
| **total** | **257** | | | |

No site has `x.size() / K != M`, so no site takes a reshape path that would
change the routing width. The only `M >= 10` event in the whole request is the
untimed seed-prime head-fc flush at `M` near 512.

### 1.3 What Q actually touches

Q is RULE 377, commits `2679ef6c` and `232b9fda`:
`QuantizedBlockLoader::shift_dst`, `qmm_t_pipelined_k_loop`, the `qmm_t_impl`
rewrite, and `Ws[BN*BK_padded] -> Ws[2*BN*BK_padded]` in `affine_qmm_t`,
`affine_qmm_t_splitk` and `affine_gather_qmm_t`. Nothing in Q touches qmv,
crossrow, or `qmm_n`. `qmm_t_splitk` is inside the blast radius, but ledger
entry 258 already proves it is dead code on this model.

### 1.4 Required statement

**The set of Q-consuming cells at `M <= 5` is empty. It is also empty at
`M = 6, 7, 8, 9`. Q's only live consumer inside a timed leg is the 512-token
seed prefill.**

### 1.5 Anomalies found while counting

These are observations, not part of the ruling.

- The MTP head fc at `Qwen35MTP.swift:195` and `:229` (K=10240, N=5120) is
  un-routed and reaches the MLX crossrow qmv path.
- The 2-bit centroid readout at `Qwen35.swift:5823` has N=12292, so `N % 8 == 4`.
- `gatherQuantizedMM` at `Qwen35.swift:5853` has `w.ndim == 3` and B=1844, so it
  reaches `gather_qmv`.
- One `transpose = false` site at `KVCache.swift:2124` is unreachable.

## 2. Per-prompt null calibration (`harness=ranked`)

Board snapshot: 976 scored 512-token receipts, 192 distinct schedule
fingerprints. The organizer-main-schedule cluster holds 115 receipts.

The serial leg is byte-identical by construction across all of these receipts,
so its spread is pure measurement noise.

- Per-receipt common offset, robust sd: **0.1422 %**; pairwise **0.2011 %**.
- Per-prompt serial residual sd: 0.155 % to 0.171 %.
- **Adopted per-prompt pairwise 1 sigma: 0.298 % to 0.314 %**, from
  `sqrt(2 * serial_resid_p^2 + common_null^2)`.
- Contaminated upper bound from the 115-receipt candidate cluster: 0.68 % to
  0.97 %. Use it only as a ceiling; it contains real candidate differences.
- Byte-identical control pair, crown `ec24d591` against A `5a9f130a`:
  mean −0.2346 %, sd of the eight prompts 0.1816 %.

At sigma near 0.30 %, the original `B - C` vector is significant on travel
(−3.8 sigma), medicine (−3.0), beagle (−2.9), republic (−2.2) and drama (−2.2),
and is **not** significant on plutarch (+0.39), essays (−0.39) or botany
(−0.80).

## 3. Mechanism fit, and the split that settles it (`harness=ranked`)

One-parameter fits to the `B - C` vector, seven degrees of freedom:

| model | beta | se | chi2 |
| --- | --- | --- | --- |
| prefill, constant s per leg | 34.7 ms | 6.6 ms | 14.38 |
| round, constant s per round | 249.9 us | 56.3 us | 22.41 |
| uniform, constant per cent | −0.5754 % | 0.1093 % | 14.41 |

The round model is refuted: it needs plutarch at −0.82 % but plutarch measures
+0.05 %, a +3.1 sigma miss.

Two-parameter fit: per leg **+35.5 +- 12.5 ms** (2.8 sigma), per round
**−7.9 +- 107 us** (0.1 sigma, consistent with zero, 95 % upper bound
218 us per round).

**Essays as a falsification lever.** The assignment asked whether essays
discriminates. It does not, in either direction. Essays sits at −0.39 sigma
against zero, and all three one-parameter models predict essays within
0.29 percentage points of each other (−0.687, −0.417, −0.575). Essays cannot
separate the prefill model from the uniform model, and it cannot reject the
round model either. Plutarch is the only prompt that discriminates, because its
edl of 0.156 makes its round count and its prefill share diverge. The
discriminating evidence is therefore plutarch, not essays.

### 3.1 The decisive channel split, with no model at all

Each receipt carries a per-prompt `prefill_seconds_per_token` field, so the leg
splits directly:

```text
leg     = 512 * mtp_seconds_per_token_mean
prefill = 512 * prefill_seconds_per_token
decode  = leg - prefill
```

Prefill share of the candidate leg: plutarch 3.42 %, drama 5.71 %, travel
6.47 %, beagle 9.51 %, republic 10.49 %, essays 10.42 %, medicine 10.47 %,
botany 10.60 %.

`B - A` per prompt, the clean Q instrument, because both trees were inspected
first hand and only A is free of the instrumentation that contaminates C:

| prompt | prefill % | decode % |
| --- | --- | --- |
| plutarch | +1.9225 | −0.0579 |
| drama | +2.1709 | −0.1685 |
| travel | +1.7964 | +0.1604 |
| beagle | +1.7070 | −0.0839 |
| republic | +2.0347 | +0.1444 |
| essays | +1.9133 | +0.0178 |
| medicine | +2.0869 | +0.0709 |
| botany | +1.8552 | +0.0391 |
| **mean** | **+1.9359** | **+0.0258**, sd 0.11 % |

Factor system by channel, mean8 prefill / mean7 decode / mean7 leg:

| pair | content | prefill | decode | leg |
| --- | --- | --- | --- | --- |
| B − A | Q + I | +1.9359 | +0.0258 | +0.2004 |
| C − A | I | −0.2315 | +0.9889 | +0.8765 |
| B − C | Q as previously derived | +2.1724 | −0.9524 | −0.6691 |
| D − B | E165 − I | +0.0692 | +0.7579 | +0.6878 |
| D − A | Q + I + E165 | +2.0064 | +0.7841 | +0.8899 |
| crown − A | byte-identical null | −0.2196 | −0.2680 | −0.2663 |

Prefill-only closure test, mean7. A factor that acts only on the seed must show
`leg % = prefill % * prefill share`. This form does not depend on the
subtraction being exact, so it survives the probe caveat on
`prefill_seconds_per_token`:

| pair | prediction | measured leg | residual |
| --- | --- | --- | --- |
| B − A | +0.1782 | +0.2004 | **+0.0222** |
| C − A | −0.0231 | +0.8765 | +0.8996 |
| B − C | +0.1996 | −0.6691 | −0.8687 |
| D − B | +0.0061 | +0.6878 | +0.6817 |
| crown − A | −0.0199 | −0.2663 | −0.2464 (null scale) |

Only `B - A` closes, and it closes at the null scale.

### 3.2 What this means

- **Q is prefill-only.** It costs +1.94 % of prefill, which is +10 ms to +12 ms
  per leg. Its decode effect is +0.026 % with sd 0.11 %, and all eight prompts
  sit inside +-0.17 %. Net candidate leg: **+0.20 % slower**.
- This reproduces Edward's E175 local measurement of **+1.936 % slower prefill,
  8 of 8 same sign** (W&B `8bc65oel`, `harness=local`) to three digits. The
  penalty transfers from the local host to the ranked runner exactly.
- **FINDING 440's `Q = −0.6691 %` is an artifact of receipt C** (`fda590bb`,
  already flagged unfetchable by FINDING 443). B and C do not carry the same
  instrumentation: C carries +0.90 % of decode cost that B does not, so
  `B - C` credits Q with removing a cost Q never added.
- The `+0.0070 %` additivity residual cannot detect this. `(C - A) + (B - C) +
  (D - B) = D - A` is an identity in receipt values and holds for any C.
- **E167's original reading was correct**: decode flat, leg about +0.23 %
  slower.

### 3.3 Score prediction for `A + Q`

Applying the measured per-prompt `B - A` channel factors to A's own per-prompt
legs and recomputing the published median gives **3.702087**, which is
−0.155 % against A's `3.707845`. Compare the advisor's 3.7328 and 3.7267 and
Edward's 3.7218 and 3.7157. **Q should come out of the ship set.**

## 4. cap-4 composition ruling: NO

The cap-4 receipt landed while this analysis was running.
`90c131dc` is organizer main with one literal changed,
`segmentedVerifyDepthCap 7 -> 4`. Status **rejected**, published
**3.54742900664627**, which is **−4.33 %** against organizer-pure A at
`3.70784519415395`.

cap-4 against A, by channel:

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

cap-4 behaves exactly as a decode-channel factor: prefill is flat within the
null on all eight prompts, and the whole cost is decode.

**The ruling.** Q pays nothing at `M <= 5` because it pays nothing at any
decode `M`. Section 1 shows the consumer set is empty at every width the decode
path can reach, and section 3 confirms it in the ranked measurement. cap-4
changes only the decode denominator, which is the channel Q does not touch.
Composition therefore cannot create a Q consumer, and Q's fixed prefill penalty
rides on top unchanged.

Composing the measured per-prompt `B - A` factors onto each base and recomputing
the published median:

| base | published | reconstructed | + Q | delta |
| --- | --- | --- | --- | --- |
| organizer-pure A | 3.707845 | 3.707845 | 3.702087 | −0.1553 % |
| cap-4 | 3.547429 | 3.547429 | 3.542212 | −0.1471 % |

**Do not compose Q with cap-4.** cap-4 alone already costs 4.33 % of published
score, and adding Q costs a further 0.147 %. There is no width at which the
composition pays.

### 4.1 A cap-4 anomaly worth a separate question

botany is clipped hardest, edl 6.148 down to 3.827, and it is the one clipped
prompt that does **not** regress: decode −0.32 %. beagle, republic, essays and
medicine are clipped less and each regress by 4 % to 6 %. Whatever makes
botany's deep drafts worthless is a real and separate signal about the adaptive
depth walk. I did not chase it; it belongs in its own experiment.

## 5. Open conflict, stated honestly

FINDING 444 derives a local `Q` leg effect near **−0.88 %**. At the 23.4 %
local prefill share, and with the measured local prefill penalty of +1.936 %,
that would need about **−1.33 % of local decode**. That is the same channel
this experiment shows is empty. The two results cannot both be right.

I am not papering over it. The settling measurement is a matched local pair,
organizer-pure against organizer-pure plus Q, on one host in one
ABBA-counterbalanced session, with **prefill seconds and decode seconds logged
separately** rather than only the leg total. If that pair reproduces a local
decode gain, section 1's source census is wrong about the live call path and
the routing claim must be re-derived. If it shows local decode flat, FINDING 444
is measuring something other than Q.

## 6. Reproduction

```bash
YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
python3 research/e176_q_consumer_census.py     # writes research/e176-q-census.json
WANDB_API_KEY=... python3 research/e176_wandb.py
swift research/e176_arch_probe.swift           # Metal device metadata only
```

## 7. Suggested follow-ups, not implemented

1. Run the section 5 settling pair. It is cheap and it closes the only open
   conflict in the Q story.
2. Ask why botany tolerates a 2.32-token edl clip with no decode cost while
   beagle loses 4.6 %. The adaptive depth walk is spending real time on drafts
   that four other prompts need and botany does not.
3. If a QMM-path win is still wanted, it needs a mechanism that raises decode
   `M` to 10 or more, or a change to the qmv path instead. Optimizing `qmm_t`
   further cannot move this model's decode at all.
4. Re-audit any other finding that was derived from receipt C `fda590bb`. The
   additivity check that validated FINDING 440 is an identity and cannot detect
   a contaminated leg.
