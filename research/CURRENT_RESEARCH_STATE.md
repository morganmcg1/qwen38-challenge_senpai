# SENPAI Research State

- **2026-08-21 15:05 UTC.** Campaign active, no round limit.
- 🔴🔴🔴 **WE HELD THE FRONTIER FOR NINE MINUTES AND LOST IT TWICE.**
  `f04b102e` was promoted at 11:13:00Z with **3.32824629**. hadakang's
  `276aa2c2` took it at 11:22:52Z with **3.33849825** on a bare RESAMPLE of our
  own tree. vibecodooor's `51b9bf85` took it at 13:00:36Z with **3.35025879**
  on a real one-dispatch rerank fusion built on our tree. We are **rank 6 by
  published score** and **rank 3 by serial-free** (ours 3.33711595, hadakang
  3.33753284, vibecodooor 3.33979539).
- 🔴 **`87b654b2` (edward, margin-gated one-bit clamp) is VALIDATING** since
  13:47:16Z. Expected receipt window 14:45Z–15:50Z. Pre-registered reading:
  ≥3.42 full transfer, 3.36–3.42 half to three quarters, 3.33–3.36 weak,
  3.30–3.33 near null, <3.30 refuted.
- **Most recent human research direction:** Issue #22 — execute aggressively
  toward the winning frontier. No new human instruction since.
- Campaign base: `5c2c3b8b613841d0d9677d4540e6a08e8bd40759`, the merge of PR
  #102 (E100), on top of `97511edb` (PR #104), `9e6d1563` (ledger 249),
  `ad8403f1` (PR #100), `0c7c3930` (ledger 248), `d5075d4c` (PR #89),
  `cd0a89da` (PR #99), `4d937ce3` (PR #97) and `09315fac` (PR #98).
- ✅ **The advisor branch CONTAINS the tree that all three crowns are built
  on.** `git grep` for `buildDerivedClusterIndex` and `qwen_mtp_probe_sort`
  matches `Qwen35.swift` at HEAD.
- `BASE_SHA` for every submit call: `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
  Verified an ancestor of the campaign base.
- Organizer `upstream/main`: **`41bad1c6f124f8f0c7f324cf60e95cd2c4de2ca6`**,
  which is vibecodooor's `51b9bf85`. The whole advance from our `23ef7556`
  touches exactly two files, `Qwen35.swift` (+93/−90) and
  `mtp-head.manifest.json` (2 lines), and **both are editable paths. Zero
  trusted or organizer drift.** `frontier-state.json` on `origin/main` records
  `syncedCommit 0c90733d`, and `0c90733d` is an ancestor of `41bad1c6`, so the
  submit guard at `senpai/submit-official.sh:220-226` passes unchanged. No
  organizer sync and no replay are required before the next submission.
- 🔴 **The `41bad1c6` rerank kernel is queued for import by thorfinn (E101).**
  Take the `Qwen35.swift` hunks only; the manifest hunk is hadakang's resample
  marker and carries no scored behaviour.

## 🔴🔴🔴 FINDING 31. THE ROUND IS AN IDENTITY IN `G / rate(partition)`

```
round_us  =  G  x  14.41235 GB  /  rate(partition)
```

Reproduces every measured local round to better than 0.03 %. This is not a fit;
it is Finding 21 with the residual folded into `rate`. Its value is that it
separates the two things a partition change can do: change `G`, or change
`rate`.

| partition | M | local GB/s | vs NA=1 | ranked GB/s | vs NA=1 |
|---|--:|--:|--:|--:|--:|
| `[1]` | 1 | 223.6 | 1.000 | 462.3 | 1.000 |
| `[2]` | 2 | 206.6 | 0.924 | 409.8 | 0.886 |
| `[3]` | 3 | 192.7 | 0.862 | 368.0 | 0.796 |
| `[4]` | 4 | 167.1 | 0.747 | 333.9 | 0.722 |
| `[3+2]` | 5 | 228.6 | 1.022 | 542.8 | 1.174 |
| `[3+3]` | 6 | 209.1 | 0.935 | 477.7 | 1.033 |
| `[4+3]` | 7 | 191.6 | 0.857 | 426.6 | 0.923 |
| `[4+4]` | 8 | 175.8 | 0.786 | 385.3 | 0.833 |
| `[3+3+3]` | 9 | 211.9 | 0.948 | — | — |

Local one-group overhead against `max(DRAM floor 52,792 µs, FMA floor)`:
NA=1 **1.221×**, NA=2 1.322×, NA=3 1.416×, NA=4 1.634×, NA=5 **1.959×**
(103,404 µs measured post-collapse). At NA=5 the kernel reads at ~55 % of DRAM
peak and computes at ~47 % of the FMA ceiling **simultaneously** — a latency or
issue-rate problem, not a resource wall, and therefore repairable.

**Prize:** NA=5 at the 1.221× overhead NA=1 already demonstrates would be
64,459 µs against 103,404 today, i.e. **−37.7 % on 24.1 % of rounds**.
**ASSIGNED as E104 / PR #106.**

## 🔴🔴🔴 FINDING 32. THE GROUP-SCALING FACTOR — WHY FEWER WEIGHT STREAMS PAYS LOCALLY AND NOT ON THE RANKED BOX

Collapsing M=5 from `[3+2]` to `[5]` halves the bytes and removes one concurrent
instruction stream. Under Finding 31 the trade is exact:

```
time([3+2]) = 2W / r2      time([5]) = W / r1
collapse gain = 1 - r2/(2 r1) = 1 - A/2 ,   A = r2 / r1
```

`A = 2` means two x-groups deliver exactly twice one group's aggregate
bandwidth, so halving the byte count buys **nothing**. The entire value of the
fewer-streams theme is the distance of `A` below 2.

| | A | collapse gain |
|---|--:|--:|
| local, from alphonse's measured round (139.4 GB/s one-group) | **1.640** | **+18.0 %** |
| ranked, route 1: group-scaling advantage, **no receipt data** | **2.040** | −2.0 % |
| ranked, route 2: the two rival receipts | **1.994** [1.964, 2.024] | −0.3 ± 1.5 % |

Route 1 uses only our own ranked cost curve: `[3]`→`[3+2]` scales aggregate
bandwidth ×1.186 locally and ×1.475 ranked, an advantage of ×1.244. Route 2 is
`dW = −0.070 ± 0.360 pp` over `ca9251b8` and `3ff80e86`, which shipped exactly
this mechanism. **The two routes are independent and they agree.**

Falsification at an M=5 share of 0.24 of G=2 rounds: if the ranked box kept its
own NA ladder, `dW` would be −1.65 pp (**4.4 σ**); if it behaved like the local
box, −4.32 pp (**11.8 σ**). Robust to the share — even 0.10 gives ~1.9 σ.

**Consequences:**

1. **The ranked G boundary does NOT move to M=6.** Edward's interim-7
   composition prediction is refuted; `marginGateDepth` stays at **3** for
   anything submitted.
2. **New general rule for the stop list:** any "fewer weight streams" idea must
   be priced on the **ranked** group-scaling factor, not the local one. They
   differ by 1.24×, which is the difference between +18 % and 0 %.
3. **E100 stays merged** — a large, clean, correctly measured local winner worth
   ≈ −0.03 % ± 0.40 % published once the R=91→98 register tax (+0.0974 %) nets
   against the width benefit.
4. **`rate(NA)` is the largest remaining lever.** One wide ranked x-group at
   NA=5 streams **272 GB/s** while two together stream **543 GB/s**: the machine
   has ≥ 2× the outstanding-load capacity one wide group uses.

Ranked prize if `rate(NA)` is repaired (G=2 leg effect at share 0.24):
×1.10 → −2.25 % · ×1.25 → −4.86 % · ×1.50 → −8.05 % · ×1.75 → −10.33 % ·
×2.00 → −12.04 %. **Even a 10 % lift is 8× the published detection floor.**

## 🔴 FINDING 23. BYTE-SHARE PRICING OVER-PRICES BY ABOUT 5x IN THIS KERNEL FAMILY

Source: askeladd's E98 terminal kill, PR #100, ledger 249.1 to 249.3.

> In the `qmv_fast_crossrow_affine4_g64_m` and `_wide` family, logical byte
> share multiplied by achieved bandwidth over-prices byte-reduction work by
> about 5x. The correct discriminator is **"does the removed byte reach DRAM
> once per read, or is it replayed from cache".**

Scored cells read at 88.7 % to 98.9 % of the 273 GB/s DRAM peak **on logical
bytes** — `lm_head` at M=5 reaches 269.9 GB/s — yet an 11.1 % logical byte cut
returns only about 0.17 of the predicted time. Alphonse reached the same
conclusion independently from the ranked board, fitting a
proportional-to-model coefficient of `rho = 0.204`.

Consequences:

1. 🔴 **THEME 1, bytes per weight, IS CLOSED.**
2. Finding 21's 88.6 % streaming share is a census of where round time goes.
   It is **not** a licence to price byte removal.
3. Repriced and shelved: the 12-bit packed index, the head-side affine-2
   metadata index, entropy coding, and every weight-byte-removal proposal.
4. Finding 22's stream-class factor of 1.00 still holds where the removed
   bytes genuinely reach DRAM.

## 🔴 FINDING 24 (AMENDED). YUKON DEDUPLICATES ON ARCHIVE CONTENT, AND ANY BYTE DEFEATS THE DEDUPE

FACT 27 v3. The dedupe key is archive content. It is defeated by any changed
byte anywhere in the archive, **including the 583-character free-text `note`
field of `mtp-head.manifest.json`**, which has no scored behaviour at all.

🔴 **ADVISOR ERROR 55.** I recorded that no rival could redraw a promoted tree
and sent that to thorfinn as settled. hadakang redrew OUR promoted tree within
nine minutes, using one character in a free-text note field, and took the
crown. Finding 24's observation was correct; my inference from it was not.

**Advisor policy, unchanged:** we do NOT buy resample draws with cosmetic
diffs. `program.md` says "Do not send duplicate official submissions." An
unused note file sits at `senpai/submission-note-e101-draw3.md`.

## 🔴 FINDING 25. THE PERFECT SAME-ARCHIVE REPLICATE PAIR

`research/board_pair_decompose.py f04b102e 276aa2c2`. hadakang's `276aa2c2` is
our own promoted tree resubmitted. Identical `qwen_mtp_weights_hash`,
identical `head_provenance_sha256`, identical `decode_tokens` 512, identical
`mtp_max_draft_depth` 8, and a **digit-identical draft schedule on all eight
prompts**.

```
candidate leg   hada faster by  -0.0189 %   sd 0.0511
serial leg      hada slower by  +0.2125 %   sd 0.3989
serial-free   ours 3.33711595   hada 3.33753284   gap +0.0125 %
published     ours 3.32824629   hada 3.33849825   gap +0.3080 %
```

🔴 **The published gap is 25x the serial-free gap. The entire crown change was
the serial-baseline lottery.** This is the cleanest available confirmation of
Finding 20 and of thorfinn's frame decomposition, and it cost us no
submission.

## 🔴 FINDING 26. THE RIVAL RERANK FUSION, AND AN INDEPENDENT RANKED CALIBRATION OF FINDING 22

`research/board_pair_decompose.py f04b102e 51b9bf85`. Same schedule on all
eight prompts.

```
candidate leg   vibe faster by  +0.0268 %  sd 0.0730   (unweighted over 8)
serial leg      vibe slower by  +0.1956 %  sd 0.3346
serial-free   ours 3.33711595   vibe 3.33979539   gap +0.0803 %
published     ours 3.32824629   vibe 3.35025879   gap +0.6614 %
```

Under Finding 16 the median pair is beagle plus essays, giving
`(0.0477 + 0.1103)/2 = +0.079 %`, which reproduces the measured `+0.0803 %`.
**Eight ninths of their headline is the serial lottery. One ninth is a real
mechanism.**

🔴 **THE CALIBRATION.** Their edit removes about one net dispatch and changes
the launch geometry, worth roughly 10 to 15 us per draft (Finding 4 prices
`affine_gather_qmv` plus the top-32 finalize at 29.64 us/draft combined).
Finding 22's latency-class table predicts `+0.323 %` at 40 us/draft, therefore
`+0.081 %` to `+0.121 %` at 10 to 15 us. **Measured `+0.0803 %`.** Finding
22's latency-class pricing survived a test it could have failed. It does not
repeat advisor error 53.

**Their exact source delta**, importable, in
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`:

- deleted `qwen35DraftRerankKernel`, the 32-lane value/id reducer;
- deleted `gatherQuantizedMM` over 32 independent `N=1` affine-4 rows, the
  `_compactDraftGatherW/S/Z/Lhs` batch views, the `MLX_E85_GATHER_QMM` gate
  and `qwen35GatherQMMRerankEnabled`;
- added `qwen35DraftSelectedAffine4RerankKernel`, name
  `qwen_mtp_draft_selected_affine4_rerank_g64_v1`, inputs
  `["x","candidate_ids","weight","scales","biases"]`, output `["token_id"]`,
  `grid (256,1,1)`, `threadGroup (256,1,1)`, `outputShapes [[1,1]]`,
  `outputDTypes [.int32]`, constants `TG_SIZE 256, TOPK 32, SIMD_SIZE 32,
  NSIMD 8, K 5120, K_WORDS 640, K_GROUPS 80, VALUES_PER_LANE 16, BLOCK 512`,
  and `static_assert(NSIMD * 4 == TOPK)`. Eight SIMDgroups own four rows each;
  nibble masks `0x000f/0x00f0/0x0f00/0xf000` with `xv` pre-scaled by
  `1, 1/16, 1/256, 1/4096`; `result += scale*accum + sum*bias`; `simd_sum`;
  `float(bfloat16_t(reduced))` into `threadgroup float exact_scores[32]`; and
  `sg == 0` runs the incumbent `qwen_draft_selected_rerank_better` shuffle-down
  total order. The `PREFIX_COUNT` and `CONTROL_OFFSET` mapping is unchanged;
- moved the `qwen35Top32RealCount` / `qwen35Top32K` drift guard from inside
  the dense fallback branch to the top of `draftTokenIDWithDeclaredRerank`.

Their public gate discipline: a 32-token smoke run checked 41/41 declared
rows, and a 512-token gate checked 522/522 declared rows with zero residual
divergence.

## 🔴 FINDING 27. THE DISPATCHER CENSUS

Source: askeladd's E102 rung 1, PR #104, ledger 249.12. CPU only, no GPU time,
no candidate file touched. Entry cell
`affine_qmv_fast<bfloat16_t, 64, 4, false>`, `g16s` local and `g17s` through
`xcrun metal-tt`.

**27a. The dispatcher allocates exactly what its widest inlined body
allocates.** Seven of seven arms, both GPU generations, including the 16-byte
`g16s` spill frame at NA=6. The `switch (ntg.x)` adds no register pressure at
all. **E76's and E97's body-scope censuses were the correct instrument all
along.**

**27b. 🔴 `peak_live_regs` IS NOT A REGISTER COUNT.** It is AIR SSA liveness:
1.7x above machine allocation at body scope, 1.8x at entry-point scope. It is
order-preserving only (A 163 < C 182 < B 183 < D 201 matches `g17s`
91 < 98 = 98 < 111). **Use it as a relative screen, never as a level.** The
campaign has been quoting 163 and 125 as if they were registers.

**27c. The runtime `MTLComputePipelineState` arbiter is SATURATED.** Every arm
and every cell reports `maxTotalThreadsPerThreadgroup = 1024`,
`threadExecutionWidth = 32`, `staticThreadgroupMemoryLength = 0`, and the
device maximum is also 1024. Register counts of 91, 98 and 111 all permit a
full 1024-thread threadgroup. `metal-tt` is the only instrument that separates
the two generations.

**27d. DEAD-WIDTH PRUNING IS DEAD.** Arms A, E and F are byte-identical GPU
objects (`g16s` digest `8c9c4255`, `g17s` `44096e70`); the compiler folds the
emptied `case 9:` into `default:`. Arm G, pruned on both branches, still holds
94/91. Only machine text moves, 126,984 to 114,736 `g17s` bytes, which
converts to 0.0000 % under the E77-corrected occupancy law. **Rung 2 closed
with a negative and was never built.**

**Also settled:** arm B and arm C are identical in registers and in pipeline
properties. The unreachable `<T,9,5,true>` costs zero registers and 9.6 KB of
cold text. Dropping it is right, and the prize is zero. It is NOT why
`ca9251b8` lost.

### 🔴 The first zero-parameter ranked prediction of the campaign

E77-corrected law `S(R) = floor(496 KiB / (128 R))` and
`Omega = (32/S)^0.01346`. Arm B has `R=98`, `S=40`, therefore **+0.0974 %** on
every QMV dispatch at every width.

| group | prompts | measured | prediction | residual |
|---|---|---:|---:|---:|
| G = 1 | plutarch, drama, travel | **+0.1068 %** | **+0.0974 %** | +0.0094 pp |
| G = 2 | beagle, medicine, republic, essays, botany | **+0.5516 %** | +0.0974 % | **+0.4542 pp** |

Within-mode per-run serial-free sd is 0.113 %, so the G=1 agreement is inside
noise and the G=2 residual is four times it.

🔴 **A widening confined to one width still taxes every other width, because
the entry point is shared. Every future kernel proposal must price that flat
tax before its own gain.** The G=2 residual falls entirely on M=5 rounds, since
M=9 is unreachable, implying about +1.9 % on the M=5 round (marked INFERENCE)
where the byte model predicts −18.75 % and alphonse measured −17.7 % in
isolation. Two suspects remain, and registers are not one of them: the four
non-kernel files in `ca9251b8`, or a `g17s` execution effect at NA=5 that
registers cannot see.


## 🔴 FINDING 22. THE TRANSFER LAW HAS TWO CLASSES. PRICE EVERY MECHANISM WITH THE RIGHT ONE

Source: thorfinn's E87 terminal result on PR #89, self-corrected against his own
receipt, plus my reprice in `research/finding22_reprice.py`. Ledger 248.

**The law:**

```
ranked delta_us / local delta_us  =  (local achieved rate) / (ranked achieved rate)
```

For DRAM-bound work both rates are the machine's streaming bandwidth, so the
ratio is 249.55 / 542.8 = 0.460 and the PERCENTAGE is preserved. For
latency-bound work neither rate scales with DRAM bandwidth, the ratio is about
1.0 (measured 0.98), and the PERCENTAGE is amplified by
`local_round / ranked_round` = 2.401 at M=5.

```
STREAM  work            ranked % = local % x 1.0     (0.460 x 2.401 = 1.104)
LATENCY work            ranked % = local % x 2.40    (0.980 x 2.401 = 2.353)
HEAD BYTE removal       x 0.236                       MEASURED, E87 arm C
ACCEPTANCE loss         x 1.0                         accounting identity
```

The sanity check the law must pass is that a DRAM-bound saving keeps its
percentage, because the item and the round it divides into scale together. It
does, at 1.104. That is why the latency branch is credible.

**Evidence.** Section 8 removes fixed dispatch latency, not bytes. Priced with
the 0.236 byte factor it was +0.0095 %; measured in the serial-free frame it was
**+0.1117 %**, an understatement of about **12x**. Thorfinn's forward prediction
with no fitted parameter, from the isolated census rate 12.84 us/draft and
public ranked round times, gives +0.1036 % on the median pair, 93 % agreement.
A board regression concurs at 12.53 us/draft (se 5.73, t 2.19).

**RETIRED: Finding 13's derived transfer factors.** The "fixed / launch"
transfer of 0.670 is wrong: a fixed-class local cost of 65,674 us transferring
at 0.98 would need 64,361 us of a 55,870 us ranked round. Finding 13's "fixed"
bucket is streaming work that the marginal-per-row model failed to attribute,
because that model counted only marginal per-row cost and never the G=2 base
streams. Finding 21's direct census supersedes the split. **Keep only the
measured head factor 0.236 and the acceptance factor 1.0. Delete the derived
verify factor 1.532 and the derived fixed factor 0.670 from all pricing.**

**The corrected closure threshold.** Compare a LOCAL cost against:

```
STREAM-class item is dead below    0.160 % local
LATENCY-class item is dead below   0.067 % local   (0.115 % on the published floor)
```

Every item closed between those two bounds was closed on the wrong test.

**The reprice of the E96 census** (local M=5 round 127,533 us, ranked M=5 round
53,108 us, DRAM peak 273 GB/s, DRAM-bound cut at 60 % of peak):

| family | us/rnd | GB/s | %peak | class | local % | ranked % |
|---|---:|---:|---:|---|---:|---:|
| MLP gate_up | 48381.86 | 265.8 | 97.4 | stream | 37.937 | 41.883 |
| out_proj + down_proj | 36559.21 | 238.1 | 87.2 | stream | 28.666 | 31.649 |
| GDN in_proj | 17675.04 | 258.4 | 94.7 | stream | 13.859 | 15.301 |
| lm_head | 5269.31 | 271.9 | 99.6 | stream | 4.132 | 4.562 |
| attn fused QKV + gate | 5163.37 | 256.5 | 94.0 | stream | 4.049 | 4.470 |
| GDN recurrent step | 1421.13 | 212.5 | 77.8 | stream | 1.114 | 1.230 |
| **SDPA over FA history** | 1267.00 | ~53 | ~19 | **latency** | 0.993 | **2.386** |
| **fused residual + RMSNorm** | 771.54 | 27.0 | 9.9 | **latency** | 0.605 | **1.453** |
| **GDN prework** | 543.39 | 32.6 | 11.9 | **latency** | 0.426 | **1.023** |
| q/k norm + RoPE | 149.85 | - | - | latency | 0.117 | 0.282 |
| KV cache write | 89.10 | - | - | latency | 0.070 | 0.168 |
| MTP top-2 | 56.13 | - | - | latency | 0.044 | 0.106 |
| STREAM subtotal | 114469.92 | | | | 89.757 | 99.094 |
| **LATENCY subtotal** | **2877.01** | | | | **2.256** | **5.417** |

After the measured isolation discount (calibrated by the two dose ladders: GDN
step 1421.13 isolated against 861.0 dose = 1.65x; fused norm 771.54 against
298.0 = 2.59x), the latency pool is **2.09 % to 3.28 % of the ranked round**,
not the 0.87 % to 1.37 % I had been pricing.

**REVIVED by the reprice:**

- **fused residual + RMSNorm.** Dose 298.01 us/pass/round, R2 0.9506. Local
  0.234 % is below the 0.277 % published floor, which is why E96 rung 3a closed
  it. Ranked **0.561 %**, which is 2.0x the published floor.
- **SDPA over the full-attention history.** Carried at "0.4 % to 0.6 %
  corrected"; discounted ranked **0.92 % to 1.45 %**. Largest single latency
  item. 79.19 us per dispatch is far above launch overhead, so this is
  inefficiency, not launch cost: about 4.2 MB per layer per round at 79.19 us
  implies 53 GB/s, 19 % of peak. Its true factor sits between 0.46 and 0.98; at
  a conservative 0.7 it is still 0.64 % to 1.01 % ranked.
- **GDN prework.** Ranked 1.023 % isolated, **0.40 % to 0.62 %** discounted.

**STAYS CLOSED: the GDN recurrent step.** Stream-class at 212.5 GB/s, 77.8 % of
peak, so its percentage does not amplify: 0.675 % local, 0.745 % ranked, with
little headroom, and the scored path reaches the non-editable `GatedDelta.swift`.

**No live assignment's stop rule moves.** E98, E99 and E100 are stream-class or
schedule-class, and E99 is already priced on the ranked curve. The head-side
affine-2 metadata idea is a genuine byte change, so 0.236 stays correct there
and its 0.17 % shelving stands.

**ADVISOR ERROR 52.** I accepted a byte factor for a latency mechanism and then
retired the E87 selection chain on it. Repriced, that chain is +0.918 % on the
median pair as an f16 bound, and the realizable part is +0.32 % to +0.72 %.
E101 (thorfinn, PR #103) reopens it.

## 🔴 FINDING 21. THE ROUND IS AT LEAST 82 % DRAM WEIGHT STREAMING, AND THE TRANSFORM THAT WRITES THOSE WEIGHTS IS OURS

This finding reorganises the whole campaign. Read it before pricing anything.

**21a. The floor.** The transformed target weights total **14.4123 GB**. The
student M4 Pro has a DRAM peak of about **273 GB/s**, so one full weight stream
cannot take less than **52,792 us**.

| M | G | measured round busy | minimum streaming time | streaming share |
|--:|--:|---:|---:|---:|
| 1 | 1 | 64,445 us | 52,792 us | **>= 81.9 %** |
| 5 | 2 | 126,103 us | 105,584 us | **>= 83.7 %** |
| 9 | 3 | 204,029 us | 158,376 us | **>= 77.6 %** |

The achieved rate implied by `G * 14.4123 GB / round` is 223.6, 232.2 and
219.7 GB/s, which is **82 to 85 % of the DRAM ceiling**. The ranked M5 at M = 5
implies about **542 GB/s** on the same accounting.

🔴 **Only two quantities can move the score by a large amount: the number of
bytes per weight, and the number of full weight streams per round, `G`.** One
extra stream costs about 52,800 us, which is **42 % of the M = 5 round**.
Everything inside the width-independent term `a` = 10,919.5 us lives inside
**8.7 %** of the round, and no single item in it clears the detection floor.

Caveat on method: a single-`S` fitted model `cost = G*S + k*M` does not fit both
dispatch bands, because `G` varies per tensor. **Use the floor argument — total
weight bytes divided by DRAM peak, against the measured round — and never a
fitted `S`.**

**21b. The transform is candidate-owned and the whole field has left it
untouched.** The ranked workflow step `Transform Qwen target in bench sandbox`
at `.github/workflows/qwen-mtp-ranked-benchmark.yml:1669-1700` runs, inside the
submission sandbox, with the log line `running submitted transform`:

```
.build/release/mlxfast-swift transform --reference "${MLXFAST_QWEN_MTP_TARGET_DIR}" --output weights
```

The pinned artifact is the **source** checkpoint. The `weights/` directory the
ranked target loads is produced by **our** code, and
`Sources/MLXFastTransform/` is editable. `qwen_mtp_weights_hash` is a **TOCTOU
guard, not a pin**: the workflow hashes the transform output at `:1703` and
re-checks it at `:2791-2821` to detect a change **during** the run. It is never
compared with a repository constant.

🔴 **690 of 690 scored board runs report the identical hash
`b53e4991737cdf50827e518e7559628874d3ff6d5f63bebc057ddbb16a89e2cd`.** No
submission from any solver has ever changed a byte of the transformed weight
representation.

**21c. The mechanism is already written and has no reader.**
`Sources/MLXFastTransform/AffineMetadataCoding.swift`, 438 lines, already builds
the uint16 (scale, bias) index: `pairToIndex: [UInt32: UInt16]`, a 65,536-entry
lookup table, the pair packed as `UInt32(scale) | (UInt32(bias) << 16)` over two
bf16 halves, emitting `<stem>.metadata_indices` and `<stem>.metadata_lut` into
the shard `mlxfast-projection-metadata.safetensors`. It is called from
`Transform.swift:268` and gated to the Laguna `.gemma4` family. **There is no
runtime consumer anywhere in `Sources/`.**

**21d. The arithmetic.**

```
affine-4 g64 today   32 B nibbles + 2 B scale + 2 B bias = 36 B / 64 elements
with a uint16 index  32 B nibbles + 2 B index            = 34 B / 64 elements
byte reduction       2 / 36 = 5.56 %
against >= 82 % streaming share -> local round floor >= 4.55 %
```

E97's metadata census makes this lossless and exact: 498 tensors, 420,208,640
groups, 1.68 GB of metadata; **zero** tensors have 256 or fewer distinct pairs
(minimum 911), so an 8-bit table is impossible, but the **maximum** is 7,846, so
an aligned uint16 index is lossless for all 498 and costs only 5.17 MB of
tables.

The engineering crux is buffer plumbing:
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` is **not**
editable and passes exactly three arrays (`w`, `scales`, `biases`) with fixed
shapes. Three routes exist: repurpose `scales` as the bitcast index and
`biases` as the table; bypass the MLX op with `MLXFast.metalKernel`, as
`Qwen35.swift` already does for Gated DeltaNet; or index the bias only.

---

## Board and submission state

- 🔴🔴🔴 **THE CROWN CHANGED HANDS THREE TIMES IN UNDER TWO HOURS.**

  ```
  11:13:00Z  f04b102e  morganmcg1   3.32824629  promoted   E87 arm C + section 8   tree 23ef7556
  11:22:52Z  276aa2c2  hadakang     3.33849825  promoted   a RESAMPLE OF OUR TREE  tree ca061247
  13:00:36Z  51b9bf85  vibecodooor  3.35025879  promoted   a MECHANISM ON OUR TREE tree 41bad1c6
  ```

  `51b9bf85` detail: `submissionCommitSha a8d37e2c…`, `promotedSourceRef
  41bad1c6…`, created 11:41:47Z, note begins `Model: GPT 5.6 Sol`,
  `baseline_serial_seconds_per_token_mean 0.03801052`,
  `candidate_mtp_seconds_per_token_mean 0.01528845`.

- **Live board top, 13:05Z.**

  ```
  51b9bf85  vibecodooor    accepted  3.35025879 promoted   11:41:47Z
  276aa2c2  hadakang       accepted  3.33849825 promoted   11:22:52Z
  e18433d4  audreyt        rejected  3.33813079 -          11:43:51Z
  71eddde4  newjordan      rejected  3.33695815 -          11:25:37Z
  f04b102e  morganmcg1     accepted  3.32824629 promoted   10:17:01Z   <- ours
  8819b108  audreyt        accepted  3.32794961 promoted   02:31:06Z
  0c2807a4  newjordan      rejected  3.32774046 -          09:44:09Z
  4cb3c9b7  vibecodooor    rejected  3.32552796 -          07:12:05Z
  214d92aa  vibecodooor    accepted  3.32529025 promoted   01:54:24Z
  a321a008  audreyt        rejected  3.32466460 -          07:33:51Z
  VALIDATING: f372c980 fkiene | 91623861 andreolf | 6737e312 Lieisyourlie
              9cb82a2f ivanfioravanti | 4a36ad31 newjordan | 27057395 jonathan308
              6ce628f0 audreyt
  ```

- 🔴 **Both losses are decomposed. Neither is a mechanism deficit we cannot
  answer.** See Findings 25 and 26 above. hadakang's whole gain is the serial
  lottery: their serial-free advantage is `+0.0125 %` against a published
  advantage of `+0.3080 %`, a factor of 25. vibecodooor's real mechanism is
  worth `+0.0803 %` serial-free against a published `+0.6614 %`, so eight
  ninths of their headline is also the lottery.

- 🔴 **THE YUKON IN-FLIGHT SLOT HAS BEEN IDLE SINCE 11:13Z AND IS RESERVED FOR
  EDWARD.** E99's margin gate measures `+3.222 %` at cap 8 and `+3.126 %` at
  cap 5 on the ranked us-per-token statistic. Predicted published is
  `3.32825 x 1.031 ≈ 3.431`, a margin of `+2.4 %` over the live frontier
  against a published single-run sd of `0.277 %`. **Submit clearance was issued
  at 13:12Z.**

- 🔴 **Do NOT rebase E99 onto `41bad1c6` before submitting.** A rebase costs a
  rebuild, a fresh exactness gate and a fresh timed session, and buys at most
  the `+0.08 %` that thorfinn will import separately. Guard preconditions were
  verified at 13:10Z: `0c90733d` is an ancestor of `41bad1c6`, there is no
  trusted drift, and `BASE_SHA` is unchanged.

- 🔴 **`cb8aeefb`, the same mechanism without section 8, is 3.33341827
  serial-free.** Senpai holds three of the top five serial-free trees. The
  serial-free ladder at the top is now: vibecodooor 3.33979539, hadakang
  3.33753284, **ours 3.33711595**, ours 3.33341827.

- 🔴 **Draw 1 of the old resample ladder, `84b9ef7b`, scored 3.30142229 and was
  rejected. The Q-row rider was a real regression**, not a bad lottery draw: a
  discriminating regression on the eight prompts gives a per-round coefficient
  that is null (t = −0.91) and a **per-drafting-round coefficient of +820.3 us
  (t = +7.62)**. Mechanism: narrowing the Q pack under `islandFastPathReady()`
  makes `live.count` indivisible by the steel GEMM's `bn`, so all 16
  full-attention layers drop to `MN_naligned` at verify width above 1.
  **Reopen only in a width-aware form.**

- 🔴 **The crown is a max statistic.** ox-alpha submitted one unchanged tree
  three times: `70aa42aa` 3.32279, `a321a008` 3.32466, `8819b108` **3.32795**.
  Across four repeat-tree triples the published sd is 0.243 % and the max of
  three draws sits +0.233 % above the mean. hadakang's own note discloses an
  eleven-draw series on one tree spanning 3.23520 to 3.33850, a range of
  **3.2 %** with zero mechanism change.

- 🔴 **Finding 20 — the serial baseline is a second independent lottery.** The
  run-level serial random effect is **0.0821 %**, on top of a within-run
  prompt-to-prompt sd of 0.2120 %. Thermal coupling is refuted; lag-1
  autocorrelation is **−0.268**, anti-persistent. Our fourteen serial draws
  average −0.060 % against a field mean of +0.001 %.

- ⚠️ **`sigma_b = 0.078 %` is model-conditional, not measured**, and it is
  refuted at `t = +2.23` on 18 mode-matched replicate pairs. The empirical
  floors need no model: published **0.277 %**, serial-free **0.160 %**.

- 🔴 **STANDING POLICY, UPDATED. WE DO NOT BUY DRAWS WITH COSMETIC DIFFS.**
  Yukon's dedupe is defeated by any byte, including the free-text `note` field
  of `mtp-head.manifest.json` (Finding 24). That makes a bare resample
  mechanically possible and `program.md` forbids it: "Do not send duplicate
  official submissions." Every submission from here must carry a real measured
  mechanism. The single in-flight slot is a scarce research instrument, not a
  lottery ticket.

## 0. THE MEASUREMENT FLOOR. Read this before pricing anything.

Measured on 18 byte-identical same-mode replicate pairs from the 669-row board
by `research/board_replicate_floor.py`, and reproduced independently by alphonse:

| statistic | median abs pair gap | max | pair sd | per-run sd |
|---|---:|---:|---:|---:|
| **published** `(raw_beagle + raw_essays)/2` | **0.1907 %** | 0.6833 % | 0.277 % | **0.196 %** |
| **serial-free** (board-mean serial substituted) | **0.1194 %** | 0.3449 % | 0.160 % | **0.113 %** |

**One ranked pair resolves nothing below `0.55 %` published or `0.32 %`
serial-free.** Always price on the serial-free statistic: it is `1.73x` quieter
for free, because it divides out the runner's serial lottery (sd `0.166 %`).

Consequences that are now campaign policy:

- A `7/7` same-sign per-prompt result is the signature of a **run-level common
  shift**, not of mechanism strength. Tight per-prompt spread does not rescue it.
- Sub-floor mechanisms are priced from the **local device model**, never from
  the board, and they **ride** in a submission whose headline is above the floor.
- The promoted crown is a max-statistic. **Now measured, not inferred**: across
  four independent repeat-tree triples the published sd is `0.243 %` and the max
  of three draws sits **`+0.233 %` above the mean of those three**. ox-alpha's
  own three receipts of one unchanged tree read 3.32279, 3.32466 and 3.32795.
  The crown's true mechanism value is near **3.3202**. This revises the earlier
  `0.4 %` to `0.6 %` estimate downward and makes it concrete.
- Any promoted lever whose published delta is below `+0.0106` is below the
  floor. Stop citing those as evidence that a lever works.

### 0a. THE THIRD STATISTIC, AND THE QUIETEST ONE: IDENTIFIED ROUND COST `L`

`research/board_same_schedule.py`. Select every board run whose
`effective_mean_draft_len` is bit-identical on all eight prompts to the crown,
which removes the schedule as a confounder and leaves 54 runs that can differ
only in what a round costs. Fit the five `G = 2` prompts **centered on the width
centroid** `M = 6.1723`, so the level and the slope are orthogonal:

```
round_us(M) = L + S * (M - Mbar)

L : median 61,566.2 us   sd 0.90 %   noise about 0.09 %   -> identified, 10x
S : median  7,231.7 us   sd 2.73 %   within-run se 205 us -> NOT identified
```

- **`L` is the quietest official statistic available.** It averages five prompts
  instead of two. Use it to rank mechanisms; use serial-free to predict a
  published draw.
- 🔴 **Never fit the raw intercept.** A five-point line over `M` in
  `[5.38, 7.15]` extrapolated to `M = 0` see-saws: a run with low botany noise
  reads as a low slope and a high intercept with no mechanism behind it. Ledger
  243's `(a1, c1, a2, c2)` fit is valid as a population fit over 50 runs and
  invalid per run.
- 🔴 **The per-row verify slope is not resolvable by one official run.** In 54
  runs no solver has ever lowered it. The only resolved movements are five
  target-verify-path edits that raised it by 2.6 % to 10.8 %. **A mechanism that
  moves only the slope cannot be confirmed by a receipt; it must be confirmed
  locally.** A mechanism that moves the level can be confirmed by one run.

Retired by this: the "same-mode residual sd 0.1025 %" constant, the ticket
model built on it, and the single-pair prices for E84 (`-0.109 %`), E85
(`-0.199 %` and `+0.022 %`) and the `8819b108` Q-row shrink (`+0.035 %`). See
ledgers 240 and 244.

---

## 0b. THE DEPTH-4 DOMINANCE THEOREM. Local only. Advisor error 43.

> 🔴 **CORRECTION, 2026-08-21.** This theorem holds on our M4 Pro and **not** on
> the ranked M5. The ranked cost curve of section 0d gives
> `C(M=5)/C(M=4) = 53,108/43,162 = 1.2304`, which is **below** the 1.25 ceiling
> the proof needs. Depth 4 is not dominated on the machine that scores us. The
> proof below is correct; only its second measured input changes. Keep it as the
> local statement and never quote it as a ranked one.
>
> Ranked flat-`q` crossovers, against the local ones:
> depth 3 versus 4, ranked `q* = 0.9682`, local never;
> depth 3 versus 7, ranked `0.9253`, local `0.9728`;
> depth 4 versus 7, ranked `0.9098`, local `0.8428`.
>
> 🔴 And flat-`q` ranked modelling is itself invalid for pricing the schedule.
> Every measured ranked accept rate, 0.834 to 0.903, sits below the 0.9253
> crossover, so flat `q` says depth 3 everywhere, yet the measured adaptive
> schedule beats every flat-`q` fixed depth by about 10 %. Ranked acceptance is
> strongly heteroscedastic and the shipped schedule already exploits it.

E92 measured the production round-busy cost at every verify width. Depth 4 is
the most expensive draft depth of 2 through 8, by `20.0 %`, because verify
width 5 is where `G = ceil(M/4)` increments in the `quantized.h:1924-1977` WIDE
switch. The marginal step into width 5 is `39,865.7 us`, which is `3.48x` the
step into 4 and `3.40x` the step into 6.

The theorem needs no cost-model fit. With `a_i = prod_{j<=i} q_j`, acceptance
probabilities at most 1 give `a1 >= a2 >= a3 >= a4`, so

```
Y(4)/Y(3) = 1 + a4/(1 + a1 + a2 + a3) <= 1 + a4/(1 + 3*a4) <= 1.25
```

for **every acceptance profile that can exist**. Measured against it,
`C(w5)/C(w4) = 126,103.1/86,237.4 = 1.4623`. Since `1.25 < 1.4623`, a depth-4
round is dominated by a depth-3 round unconditionally. Two measured numbers,
one combinatorial inequality, no rescaling and no acceptance estimate.

Margins, from `research/depth4_dominance.py`:

| normalisation | M4 Pro measured | M5, cliff `1.126x` flatter |
|---|---:|---:|
| `C(w5)/C(w4)` against the `1.25` ceiling | `17.0 %` | `12.8 %` |
| marginal step into width 5 must fall | `45.9 %` | `39.1 %` |
| rescaled `m4` must fall | `25.5 %` | `16.2 %` |

`get_qmv_batch_limit` branches only on `arch_gen == 13 || 14`, so the boundary
**location** cannot move between gen 16 and gen 17. **`snap4` transfers.**

**What is settled and what is not.** The dominance is settled. The *share* of
ranked tokens carried by depth-4 rounds is not, and it is the entire multiplier
on the prize: `+0.80 %` at the local `6.4 %` share, `+1.86 %` at `15 %`,
`+2.68 %` at ledger 207's `21.6 %`. The ranked walk sits at mean chosen depth
`4.3818` on beagle and `5.0870` on essays against the local fixture's `6.359`,
so the local share understates the ranked one. E94's cap-4, cap-5 and cap-8
screens measure it; the depth histogram is the primary output.

**Guard rail.** `h = 0.32`, uniform shallowing, scored `2.84585` ranked, which
is `-14 %`. Uniform shallowing is catastrophic because depths 1 and 2 cost
`35,240` and `25,748 us` per token against depth 3's `22,504`. Only a
**targeted** guard is alive. `amin` and `amine92` stay screens.

Advisor error 37: I first published this margin as `29 %`, holding `Y3` fixed
while letting `r4` reach 1, which is inconsistent. Edward's `0.735 %` reads a
near-degenerate coefficient of a rearranged inequality and is not the decision
margin either. See ledger 241.

---

## 0c. THE WHITE-BOX ROUND MODEL. Five constants that predict the machine.

Askeladd fitted the target verify pass from a per-dispatch census on his Mac at
384 tokens (E95 rung 2). Combined with his E93 head model:

```
verify_us(M) = 10,920 + 27,377 * G + 10,268 * M
head_us(d)   =  2,560 +  2,226.5 * (d - 1)
M = d + 1,   G = ceil(M / IPG),   IPG = ceil(M / ceil(M / 4))
```

The three verify parameters are separately identified. `c = 10,268` comes from
within-`G` variation at M = 3 to 4, 5 to 6 and 6 to 8, which all give the same
number. `b = 27,377` comes from the `G` step at M = 4 to 5. `a` is the residual.

**It predicts edward's independent E92 production sweep to within `0.66 %` at
depths 3 through 8** — a different Mac, a different session, a different
instrument, a different token window. Errors are `+0.17`, `+0.12`, `+0.66`,
`+0.54`, `-0.13` and `-0.21 %`. It fails only *below* its fitted range, at
d = 1 (`-12.0 %`) and d = 2 (`-1.2 %`).

This is the first white-box cost model of the scored round the campaign has
held. Everything in section 0b, the depth-price defect below, and the pricing
of every schedule arm now comes out of it.

### The shipped depth price is wrong at exactly one cell

`Qwen36MTPBlockSession.swift:904-911 makeUniformDepthPrice()` is the live arm.
It prices `T(d) = V + d*h*V` with `h = 0.18` and `V` flat in width, so every
step costs `11,600 us`.

| step into verify width M | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| measured marginal us | 5,330 | 5,003 | 11,459 | **39,866** | 11,740 | 12,589 | 13,526 | 40,072 |
| shipped price us | 11,600 | 11,600 | 11,600 | **11,600** | 11,600 | 11,600 | 11,600 | 11,600 |

**It under-prices the step into verify width 5 by `3.44x`**, and with
`segmentedVerifyDepthCap = 7` that is the only `G` boundary in the legal range.
The stale `measuredRawDepthPrice` at `:946-955` puts the cliff at width 6, one
position too late, which is why `pbfit` won `3.5 %` on the old dispatch table
and lost it on the crown table.

### The optimum is bimodal: 3 or 7, never 4, 5 or 6

Cost per accepted token at flat `q`, cap 7, head cost included: **depth 4 never
wins at any `q` at or below 1.0**, and is `16.9 %` worse than depth 3 even at
perfect acceptance. **Depth 3 wins below `q = 0.9728`; depth 7 wins above it.
Depths 4, 5 and 6 are never optimal.** E92 measured flat `q = 0.9551` and
hot-head depth-7 rounds back-solve to `q` near `0.966`, both below the
crossover — which is exactly why fixed depth 3 at `22,504 us/token` beat
adaptive depth 7 at `22,986`.

### The open contradiction inside the model

`b = 27,377 us` for a `14,412 MB` weight pass reads as **`526.4 GB/s`, which is
`1.99x` the `265.0 GB/s` DRAM ceiling**. The same contradiction appears from
the other end: at M = 1 the round busy is `64,445 us` and the weight stream
alone needs `54,385 us`, leaving at most `10,060 us` for all non-qmv work while
`a` alone is `10,920 us`.

**Leading hypothesis:** every x-group reads the full weight tensor, but if the
`G` groups run concurrently the later groups hit in the system-level cache.
Raising `G` would then add latency and issue cost rather than bytes, `b` would
not be a bandwidth term, and **byte reduction could not reach most of it.**
Askeladd's isolated `qmv_fast_crossrow_affine4_g64_wide` probe at `100.3 MB`
against `11.8 MB` cache-resident discriminates this directly and is running.

Until it settles, **no byte-reduction mechanism may be priced against the `b`
term**, and the `43-45 %` bytes / `46-49 %` arithmetic split stays out of the
ledger.


---

## 0d. THE RANKED COST CURVE AND THE TRANSFER TABLE. Read this before pricing anything.

Until 2026-08-21 every price in this campaign was a **local** price with an
unmeasured transfer to the ranked M5. The transfer is now measured for one work
class and derived for the other two, and the three differ by a factor of six.

### The ranked curve, recovered from the official board

`effective_mean_draft_len` in `officialMetrics.per_prompt` is the exact rational
`total_drafts / total_rounds`, where the denominator counts non-drafting rounds
too. `Fraction(dl).limit_denominator()` recovers the ranked round count. With 512
decode tokens and `mtp_seconds_per_token_mean` that gives the ranked cost of one
round at a known verify width `M = drafts + 1`.

Fitted independently on **50 official runs on the reference schedule**:

```
harness=ranked, M5
G=1, M=1..4 : round_us = 27,181.5 + 3,995.1 * M     cv(a1) 0.85 %, cv(c1) 4.73 %
G=2, M=5..8 : round_us = 16,943.2 + 7,233.0 * M     cv(a2) 5.57 %, cv(c2) 2.79 %
```

| M | G | ranked us | local us | ratio | ranked marginal | local marginal |
|--:|--:|---:|---:|---:|---:|---:|
| 1 | 1 | 31,177 | 64,445 | 2.07 | — | — |
| 2 | 1 | 35,172 | 69,776 | 1.98 | 3,995 | 5,330 |
| 3 | 1 | 39,167 | 74,778 | 1.91 | 3,995 | 5,003 |
| 4 | 1 | 43,162 | 86,237 | 2.00 | 3,995 | 11,459 |
| **5** | **2** | **53,108** | **126,103** | 2.37 | **9,946** | **39,866** |
| 6 | 2 | 60,341 | 137,843 | 2.28 | 7,233 | 11,740 |
| 7 | 2 | 67,574 | 150,431 | 2.23 | 7,233 | 12,589 |
| 8 | 2 | 74,807 | 163,957 | 2.19 | 7,233 | 13,526 |

🔴 **The ranked group-boundary cliff is +23.0 %. The local cliff is +46.2 %.**
That is why `h = 0.32`, a uniform shallowing, scored 2.84585 = -14 % ranked.

Two structural differences, both load-bearing:

1. Locally the per-row slope is **flat** at 12,494.5 us across the boundary and
   the whole step sits in the group term `b`. On M5 the slope **nearly doubles**,
   3,995 to 7,233. The machines differ in shape, not only in scale.
2. Because `c1 != c2`, the form `a + bG + cM` is **not identifiable** on ranked
   data. A naive least-squares fit with free round counts returns `b = -20,374.7`.
   Use two independent lines plus the physical constraints.

**Caveat to carry on every use: the ranked round counts are inferred, not
measured.** Internal validation is that four parameters fit eight prompts to 1.3 %
and stay stable across 50 independent runs. Instrument
`_advisor_scratch/rankedcurve.py`.

Ranked round counts per 512 tokens: beagle 110, essays 92, republic 93,
medicine 90, botany 81, travel 212, drama 252, plutarch 487 of which 449 are
non-drafting. Ranked accept rates: beagle 0.834, botany 0.866, medicine 0.892,
essays 0.897, republic 0.903, travel 0.533, drama 0.449, plutarch 0.333.

### The work-class transfer table

Combining the ranked curve with the five-constant local model of section 0c and
the **measured** arm C head transfer, at the ranked beagle width `M = 5.3818`:

| work class | local us | local share | ranked us | ranked share | transfer |
|---|---:|---:|---:|---:|---:|
| proposal head | 10,090 | 7.70 % | 1,019 | 1.82 % | **0.237** measured |
| **per-row verify** | 55,261 | 42.18 % | 36,088 | **64.59 %** | **1.532** derived |
| fixed / launch | 65,674 | 50.12 % | 18,763 | 33.58 % | **0.670** derived |
| round | 131,024 | | 55,870 | | machine 2.345x |

🔴 **This is the strategic fact of the campaign.** M5 is much faster than our
M4 Pro at bandwidth-bound streaming, which is where the proposal head lives, and
only 1.53x faster at per-row verify work. **We have been spending our strongest
students on the axis with the worst transfer.** The axis with the best transfer is
two thirds of the ranked round and completely un-attributed.

The head transfer is measured, not modelled. Arm C is the first mechanism whose
ranked and local effects were both measured on the same tree with no confound:
a 29 % head saving worth 2.233 % locally at the ranked draft depth produced
**+0.529 %** on ranked beagle. Transfer 0.237 depth-corrected, 0.327 uncorrected;
thorfinn independently inferred 0.350 from the aggregate published move.

### The four pricing rules

```
head-side local gain    -> multiply by 0.24 to 0.35
per-row verify gain     -> multiply by about 1.5
fixed / launch gain     -> multiply by about 0.67
acceptance loss         -> multiply by 1.0, ALWAYS
```

The last one is the trap. A proposal the drafter fails to retrieve is rejected by
the target on any machine, so an acceptance penalty never shrinks on transfer
while the byte gain that bought it shrinks by three. This is what moved
`derived15` from an accepted +0.23 % to +0.30 % to an unknown sign in the range
-0.5 % to +0.14 %.

### PREFILL IS NOT IN THE SCORE

For every prompt of every scored run,
`raw_ratio_of_means == serial_seconds_per_token_mean / mtp_seconds_per_token_mean`,
exact to all printed digits. Mode string `qwen-mtp-paired-decode-only`.
`prefill_seconds_per_token` is reported and never enters `raw_p`.

Every "decode share" multiplier the campaign has used was wrong. **The correct
multiplier is 1.0.** Ranked prefill is about 0.527 s per leg, so M5 prefill is
7.6x faster than our local 4.04 s, but it buys nothing either way.

---


## 0e. THE MEDIAN IS LOCKED, AND THE CENSUS METHOD HAS A CEILING

Two results from ledger 245. Both change how work is priced. Read them before
section 1, which they supersede.

### 0e.1 The exact score function, and the exact value of each prompt

Instrument: `python3 research/board_median_lock.py`. It sorts each run's eight
`raw_ratio_of_means` ascending, records which prompt occupies each rank, then
replays the median-of-eight rule under a multiplier on one prompt at a time.
That gives the exact derivative and the exact ceiling of every prompt, with no
model and no fitting.

**Rank occupancy over the 81 published runs at or above 3.25:**

| rank | occupant |
|---|---|
| 4 | **beagle, 100.0 % — every one of 81 runs** |
| 5 | essays 66.7 %, medicine 19.8 %, republic 7.4 %, botany 6.2 % |

**The score is therefore exactly:**

```text
published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)
```

Only the first term is free. The second is pinned by a four-prompt cluster that
spans less than 1.6 %, so improving essays alone simply hands the 5th slot to
republic. Exact single-prompt value at the crown `8819b108`:

| prompt | raw ratio | published gain per 1 % | ceiling | reached at |
|---|---:|---:|---:|---:|
| **beagle** | 3.185167 | **+0.4785 %** | **+4.6625 %** | 9.8 % |
| essays | 3.470732 | +0.3721 % | **+0.3721 %** | 0.8 % |
| travel | 2.188496 | 0 | +4.6625 % | 59.8 %, unreachable |
| republic, medicine, botany, drama, plutarch | | 0 | 0 | — |
| **uniform, all eight** | | **+1.0000 %** | unbounded | |

The same shape holds on our own `cb8aeefb`: beagle +0.4801 % per point with a
+4.5146 % ceiling, essays +0.5199 % per point with a +0.5269 % ceiling, every
other prompt zero.

**Four consequences for how we assign work:**

1. **A beagle-only mechanism is worth 12.5 times an essays-only mechanism.**
   Essays saturates after 0.8 % and pays nothing after that.
2. Uniform mechanisms keep the full 1.0 multiplier and remain the best value per
   unit of engineering. Nothing here demotes them.
3. After uniform work is exhausted, **every remaining prompt-specific
   microsecond belongs to beagle**, which still has 4.66 % of untouched ceiling.
4. 🔴 **Beagle's deficit is an acceptance deficit, not a cost deficit.** Beagle
   accepts 0.834 at mean draft length 4.382; essays accepts 0.897 at 5.087. Their
   round costs sit on the same shared curve. Beagle is simply the least
   predictable prompt in the pool.

🔴 **The low-acceptance regime is the highest-value unexploited axis in this
campaign.** Prompt detection is illegal, but a schedule that behaves better when
**observed** acceptance is low is legal, general and worth far more than one
tuned for high acceptance. The campaign has implicitly tuned for the opposite.
Beagle is also the only scoring prompt above the verify group boundary, at
`M = 5.382` against a boundary at `M = 5`, so every boundary-price decision is
made on the one prompt that sets half the score.

### 0e.2 MLX dispatches are concurrent, so a per-dispatch census is an upper bound

Source, all in `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp`,
which is **not** editable:

- `:545-548` every compute encoder is created with `MTL::DispatchTypeConcurrent`.
- `:363-374 maybeInsertBarrier()` inserts a buffer memory barrier **only** when
  `needs_barrier_` is set.
- `:322-325` and `:344-350` set `needs_barrier_` from whole-`MTL::Buffer`
  overlap between this dispatch and the previous one.

Three consequences:

1. **Independent dispatches inside one command buffer overlap in wall time.**
   Summing per-dispatch intervals double-counts.
2. `MLX_E58_BUFFER_LIMIT_MB=1` measures a cost the round does not pay. It
   serialises concurrent work and charges every kernel a full submit and drain.
   An isolated per-kernel time is an **upper bound** on round contribution.
3. **The error size is predictable.** A kernel that saturates the machine cannot
   overlap much, so isolated is close to true. A kernel far below peak leaves the
   machine free, overlaps, and inflates roughly in proportion.

This is exactly why E93's head census cross-validated to 0.7 % against
thorfinn's round-level arm C delta — those are DRAM-saturating GEMVs — while the
Gated DeltaNet step, censused at 37.2 GB/s or about one eighth of the machine,
came out about eight times too large.

**What survives:** every round-level and leg-level measurement. Askeladd's E95
rung 2 width model, edward's E92 ladder, the ranked M5 cost curve in section 0d,
the identified level `L` in section 0a, and the per-row verify slope that E97 is
built on. A round-level marginal cannot be inflated by concurrency, because
overlap is already priced into the wall time.

**What needs a caveat:** every per-dispatch attribution. E95 rung 3 and
thorfinn's E87 §8 isolated chain cost of 113.78 us/draft are upper bounds.

🔴 **A lever this exposes, candidate and unassigned.** MLX tracks dependencies at
whole-buffer granularity, not array-slice granularity. Two dispatches touching
disjoint slices of one buffer still trigger a full-encoder barrier and lose all
concurrency. Our editable surface writes into shared buffers, for example
`KVCache.swift:398` and `:434` `slice_update` across the 16 full-attention
layers, and the Gated DeltaNet state writes. `device.cpp` is not editable, so
the barrier policy is fixed, but **what we ask it to do is entirely editable
Swift**.

### 0e.3 Advisor error 45

I read a per-dispatch census rate of 37.2 GB/s, one eighth of the machine, as
evidence of headroom. It was evidence that the census method does not apply. **A
measured rate far below peak is first a validity signal about the instrument and
only second a signal about the workload.** I built a whole assignment on the
inverted reading and the student refuted it in one session.

---

## 1. The scoring statistic. Superseded in detail by section 0e.

**The published score is exactly `(raw_beagle + raw_essays) / 2`** at the current
frontier, and section 0e gives the general form and the exact per-prompt
derivatives. The score is
the median of eight per-prompt ratios, and for eight values the median is the
mean of the 4th and 5th sorted. On every high-scoring submission the 4th is
beagle and the 5th is essays, exact to eight decimal places:

| submission | 4th | 5th | mean of the two | published |
|---|---|---|---:|---:|
| `8819b108` crown | beagle 3.185167 | essays 3.470732 | 3.32794961 | 3.32794961 |
| `214d92aa` | beagle 3.181589 | essays 3.468991 | 3.32529025 | 3.32529025 |
| `0dd455f0` | beagle 3.187837 | essays 3.451471 | 3.31965392 | 3.31965392 |
| `8e83c6b3` | beagle 3.178054 | essays 3.459828 | 3.31894061 | 3.31894061 |
| `83f0b282` ours | beagle 3.177222 | essays 3.450347 | 3.31378448 | 3.31378448 |

**Travel, drama, plutarch, republic, medicine and botany contribute nothing.**

Two asymmetric margins, and they are not the same:

- **Beagle has 31 % of headroom below it.** Travel, the next value down, is at
  2.19. Beagle stays 4th under any mechanism we can build, so a beagle
  improvement of `x %` always moves the score by about `0.48 x %`.
- **Essays has only 0.6 % to 1.0 % of headroom above it.** Republic sits just
  above. **An essays-only improvement saturates at roughly 0.7 %**, after which
  republic becomes the 5th value and further essays gains pay nothing.

A uniform improvement across all prompts preserves the ordering and pays in
full. A prompt-selective improvement must land on beagle to pay without limit.

### `mean7` is retired

I priced this campaign on the mean over the seven drafting prompts. That
statistic is wrong and it cost us the crown once already. Compare our E84
against ox-alpha's `8819b108`, which is the same idea carried further:

| prompt | our E84 | ox-alpha | who wins |
|---|---:|---:|---|
| **beagle, sets score** | −0.116 % | **−0.139 %** | them |
| **essays, sets score** | −0.103 % | **−0.150 %** | them |
| travel | **−0.229 %** | −0.156 % | us, pays nothing |
| drama | **−0.170 %** | −0.051 % | us, pays nothing |
| `mean7` | **−0.137 %** | −0.131 % | us |
| **score statistic** | −0.109 % | **−0.145 %** | **them** |

The score-statistic gap is 0.035 %. The serial-free gap between the two
submissions is 0.0357 %. They agree to three decimals. **Our mechanism won on
the prompts that pay nothing and lost on the two that are the entire score.**

Instrument: `python3 research/board_per_prompt.py pair <base> <candidate>`
prints both statistics and marks the two score-setting rows.
`python3 research/board_per_prompt.py setters` prints the 4th and 5th values
and both margins.

### The score lives at verify width 5 and 6

| prompt | drafts | mean verify width | sets score |
|---|---:|---:|---|
| **beagle** | 4.382 | **5.38** | **yes, 4th** |
| **essays** | 5.087 | **6.09** | **yes, 5th** |
| botany | 6.148 | 7.15 | no |
| travel | 2.656 | 3.66 | no |
| plutarch | 0.154 | 1.15 | no |

Ledger 207 already had M5 and M6 carrying 57.5 % of ranked round cost. This is
why. **Our local fixture runs at mean verify width 7.27, roughly two widths
above the scoring point**, over-weighting M9 by 7.58 times and under-weighting
M5 by 3.76 times. A mechanism that helps at M9 and not at M5 or M6 looks
excellent locally and scores zero.

---

## 1b. Where we stand on the board

### Serial-free, top of the board, 991 rows, 13:05Z on 08-21

| rank | id | serial-free | published | status | who |
|---:|---|---:|---:|---|---|
| 1 | `51b9bf85` | **3.33979539** | **3.35025879** | promoted, CROWN | vibecodooor |
| 2 | `276aa2c2` | 3.33753284 | 3.33849825 | promoted | hadakang |
| **3** | **`f04b102e` ours** | **3.33711595** | **3.32824629** | **promoted** | morganmcg1 |
| **4** | **`cb8aeefb` ours** | **3.33341827** | 3.32345770 | rejected | morganmcg1 |
| 5 | `8819b108` | 3.31678843 | 3.32794961 | promoted | audreyt |
| **6** | **`83f0b282` ours** | 3.31560530 | 3.31378448 | rejected | morganmcg1 |
| **7** | **`87e6421b` ours** | 3.31489386 | 3.30652180 | rejected | morganmcg1 |

🔴 **The top three trees are the SAME TREE plus increments.** `276aa2c2` is our
own `f04b102e` resubmitted with one character changed in a free-text note field
and nothing else. `51b9bf85` is our tree plus one real rerank-fusion kernel
worth `+0.0803 %` serial-free. Rank 2 versus rank 3 is `+0.0125 %`, well under
the `0.160 %` serial-free floor, so **hadakang and we are statistically the
same tree, which is exactly what they are.** Rank 1 versus rank 3 is
`+0.0803 %`, still under the floor as a single pair but supported by a
mechanism we can read in their source and price independently.

🔴 **The mechanism race is now a two-tree race and both branches of it run
through our own code.** Everything ahead of rank 5 descends from E87 arm C plus
section 8. We hold ranks 3 and 4 outright and the rivals hold ranks 1 and 2 on
top of the tree we published.

### What the two crown moves were

`214d92aa` is `0dd455f0` plus a Metal kernel that reads the affine-4 embedding
rows inside the dual-RMSNorm-concat kernel. **That is our own E85 arm (b).**

**Its ranked value is not measurable and is about `0.02 %`.** The pair
`0dd455f0 -> 214d92aa` gave `-0.199 %` and our own pair `83f0b282 -> 87e6421b`
gave `+0.022 %`. Both are below the `0.32 %` serial-free floor, they differ by
`0.8` pair sigma, and neither is evidence. The correct price comes from E85's own
device measurement: head GPU `2292.849 -> 2285.283` us per draft is `-0.33 %` of
the head pass, and at the `6.3 %` ranked head share that is `-0.021 %` of round
time. Advisor errors 30 and 35 are both instances of pricing this mechanism from
the board instead of from the device.

`8819b108` is `8e83c6b3` plus 264 lines in one file: island dead-work
elimination in the proposal-head projections, applied to K/V **and Q**. Our E84
is the K/V half only. The missing Q half shrinks the `q_proj` quantized pack
from 12,288 rows to the 11,264 live rows and replaces the `putAlong` scatter
with `concatenated` plus `take`. It saves 2,949,120 bytes per draft step,
0.6895 % of the head read, and is worth about **+0.035 %** on the score
statistic. It is assigned to askeladd as the default arm of E93 rung 4.

### The byte law is an average and must not be applied per tensor

The E82 law, 0.0815 % of candidate time per 1 % of head bytes, predicts
+0.056 % for the Q shrink. Measured increment: **+0.0063 %, standard error
0.0233**, so the prediction sits at the top of the interval. `q_proj` is a 35 MB
read and edward's corrected curve shows reads that size are partly cache-served
at 276 to 430 GB/s against 261 to 265 GB/s in the plateau. **Price a byte
removal against the size-matched achievable rate, not the flat coefficient.**
A directly measured mechanism such as E87 arm C does not need the law at all.

---

## 2. The two mechanisms that decide this campaign

### 2.1 E89 — the ranked measurement lottery is efficiency-core placement

Every ranked run draws a binary host state, independently per run, that lives
only in the drafting path and costs about 0.9 ms per drafting round. It is worth
**1.016 % of serial-free score on our own tree** and 1.409 % median across 22
pairs of other people's trees.

**Alphonse has named the mechanism with a direct measurement.** Per-round
`pthread_cpu_number_np` shows fast rounds on cpu 9, 10, 11 and slow rounds 85 %
on cpu 0 to 3. A zero-GPU probe separates two multiplicative components:
cluster placement (`background` never leaves the E cluster and never exceeds
2.600 GHz; `userinteractive` reaches 4.513 GHz on a P core, a 1.74x ratio) and
DVFS residency (a P core at 0.4 % duty only reaches 3.67 to 3.75 GHz).

**The fix is one line**: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
behind a per-thread guard, called before the round clock starts. A pilot on one
binary, back to back, 128 tokens: slow-round prevalence 1.00 -> 0.06, host phase
median 3,339 -> 632 us, **`mtp_seconds_per_token` 0.053969 -> 0.053772, +0.365 %,
bit-exact with identical `effective_mean_draft_len` to sixteen digits**.

**Secondary benefit, possibly larger than the primary.** The host state is what
destroys our local paired estimators. Thorfinn's composed-tree pair lost 58 of
63 paired rounds to it. The fix repairs the campaign's measurement instrument.

**One open discriminator.** E-core placement scales every host phase, including
the ones that run on non-drafting rounds. Scaled to ranked, that predicts a
plutarch mode effect of about +0.5 %. Observed plutarch mode sd is 0.032 % with
r = +0.043. A 16x miss. It does not change the ship decision but it must be
reconciled or flagged in the submission note.

### 2.2 E87 arm C — the largest single mechanism on the board

A two-stage IVF shortlist over the coarse draft readout: 12,292 clusters of 8
rows, 3,073 probed. It cuts the coarse stage 157,337,600 -> 59,001,600 bytes and
the whole per-draft head read 427,738,112 -> 329,402,112 bytes, a 22.99 %
reduction, with all tokens matched on every leg.

**Local: -1.688 % leg total, -2.582 % paired per-round over 63 clean rounds at
63/63 sign agreement, Mann-Whitney exact p = 1/126.** Consolidated ranked price
**+1.5 % to +1.75 %**, which is about 10 sigma against the 0.166 % serial
lottery.

**It survives composition with E85 and E90 unchanged.** The merge onto the
campaign base produced zero conflicts. Arm C replaces the producer of
`candidateIDs`; E85 and E90 replace the consumer. The absolute per-draft saving
is **-619.9 us on the composed tree against -616.4 us on the r1 tree**. Arm C
removes bytes, E90 removes dispatches and copies, and they compose additively.

**The delivery blocker is now removed.** In r1 the submitted surface was a no-op
because `mtp-head.manifest.json` still named the declared remote head, the head
artifact was 605 MB against a 25 MiB archive cap, and Hugging Face publication
returns 401 on the advisor host and on two student Macs. Option B-prime (a Swift
source table) is closed on SwiftPM resources and the 262,144-byte growth budget.
Option C's clean form is closed on the archive cap.

**Thorfinn's r2 rung 1 opened load-time derivation, and the derived partition is
better than the one we shipped in r1.** A balanced bisecting 2-means rule,
`research/e87_bisect.py --balance half`, no RNG, 14 levels, **4.87 s**, cheap
enough to run inside a model load:

| partition | probe `p` | misses / 18,092 | worst-domain `m` | gate 3.0e-3 |
|---|---:|---:|---:|---|
| **bisect, derived** | 0.25 | 4 | **2.266e-4** | pass, 13.2x inside |
| bisect, derived | 0.15 | 10 | 7.554e-4 | pass, 4.0x inside |
| plain k-means, r1 | 0.25 | 11 | 1.079e-3 | pass |
| plain k-means, r1 | 0.15 | 36 | 3.237e-3 | **fail** |

It also removes the FlashHead weak-domain failure mode: k-means put 10 of its 11
misses in narrative, the derived rule splits its 4 misses evenly.

**Provenance is closed.** `research/e87-coarse-identity.json` shows the shipped
`mtp.draft_lm_head.{weight,scales,biases}` is bit-identical to
`quantize(dequantize(exact affine-4 g64 compact lm_head rows), 64, 2)` across
all 157,337,600 bytes. So the permuted row table is a pure reordering of shipped
bytes, the centroids are leaf means of exact rows, and no requantization occurs
anywhere. The whole mechanism now lives in `Sources/` and `Vendor/`: **no custom
head, no manifest declaration, no Hugging Face, nothing in the archive.**

**Ship `p = 0.25`, not `p = 0.15`.** His optimum of `byte gain - 206.6 * m`
prefers 0.15 at +2.017 % against +1.827 %. The `206.6` is the least trustworthy
number in the campaign, the misses are about 0.45 per leg at 0.15 so no local
measurement can resolve the penalty at either point, and the downside if the
coefficient is understated is one-sided. `p = 0.25` also has a measured local
anchor at exactly 22.99 % byte removal from the r1 session. `p = 0.15` is the
immediate follow-up submission; two ranked runs at two probe fractions on one
partition give the m-penalty coefficient directly.

**One harness defect cost him a leg and is now documented.** His runtime log
channel produced nothing because `benchmark.sh:1294` writes `(deny file-write*)`
into the runtime worker seatbelt profile with only `/dev/null` allowed. The
shipped trace sink has the same failure mode at
`Qwen36MTPBlockSession.swift:788-794`, falling back to a stderr that `mtp-timed`
swallows. **Untimed capture legs need `MLXFAST_NO_SANDBOX=1`.**

---

## 3. Current research focus

Findings 23, 26 and 27 replace the previous theme set. **Two whole themes died
this round and one opened.** The campaign now has one very large schedule
lever, one newly external-validated kernel theme, one dispatch-count pool, and
two bookkeeping levers.

**Theme 1 — BYTES PER WEIGHT. 🔴 CLOSED BY FINDING 23. Do not assign it.**
E98 built all three JIT variants, proved the uint16 (scale, bias) index is
bit-lossless in-kernel across 21 cells and up to 1,986,560 outputs, and then
measured the removal as **−0.35 % against a minimum useful 1.5 %**. In this
kernel family logical byte share multiplied by achieved bandwidth over-prices
byte reduction by about 5x. The whole theme rested on that multiplication.
Everything downstream of it — the 12-bit packed index, the head-side affine-2
metadata index, entropy coding — is shelved with it.

**Theme 2a — STREAMS PER ROUND, KERNEL SIDE. E100, alphonse, PR #102.
🔴 CEILING CORRECTED DOWNWARD AND THE END-TO-END RESULT IS A NULL.**

- **Reach.** At NA ≤ 5 there is exactly ONE collapsible width and it is
  **M = 5**. M=6 needs NA=6 because IPG=5 leaves tail 1, forbidden by
  `static_assert(M % IPG != 1)`; M=7 needs NA=7; M=8 needs NA=8; M=9 is
  unreachable under `segmentedVerifyDepthCap = 7`. **Delete the "77 % of width
  mass" headline. The honest ceiling is the M=5 round share, ledger 207 puts it
  at 24.1 %, times the M=5 gain.**
- **Isolated gain is real and large.** Two real builds, 0 of 45 across-arm row
  digest mismatches, positive control differs in 5 of 5 shapes: M=5
  **−17.7 % ± 1.7 %**, M=9 −9.8 % ± 0.9 %, plus a tax of **+0.475 % ± 0.404 %**
  on the byte-identical-dispatch widths 6, 7 and 8.
- 🔴 **End to end it is a null.** 512-token depth-8 `B A A B`, all four legs
  exact with identical draft statistics: collapse is **+0.095 %** against a
  base within-arm spread of 0.032 %. The 64-token sessions gave +0.167 % at
  depth 8 and −0.098 % at depth 4. At depth 4, 12 of 13 rounds already run at
  M = 5, so **reach is not the binding constraint** in that session, and the
  reconstructed round still moves only −1.10 %.
- 🔴 **The 17x discrepancy is the real finding.** Alphonse's round
  reconstruction from measured per-shape cell costs tracks the base arm to
  within a few percent (54.15 ms at M=1, 114.68 ms at M=5, 123.94 ms at width
  6.4 against a measured 154.87 ms round) but predicts the collapse arm falls
  25.3 ms per round where the session measures 1.47 ms. Two readings survive:
  **(a)** the scored decode does not reach the wide `affine_qmv_fast` dispatch
  at all, or **(b)** it reaches it with the second weight stream already
  resident. Reading (a) would re-price E76, E97, E98 and E102 as measurements
  of an unexecuted path.
- **Two controls are ordered ahead of any further timing.** Control 1 is a free
  `MLX_E58_BUFFER_LIMIT_OPS=0 MLX_E58_BUFFER_LIMIT_MB=1` **kernel-name census**
  at forced depth 4 and forced depth 5 on the unmodified tip, reporting the
  verbatim dispatched kernel name strings for the target trunk projections;
  the JIT name carries the template parameters. Control 2 is a **dose leg**,
  `case 6 -> <T, 6, 2, true>`, legal today with no `static_assert` change and
  no register change, one 64-token depth-8 leg with immediate revert.
- 🔴 **The `rows_per_simd` escape is DEAD.** E76 measured `rps2` at
  **+14.16 %** slower at NA=5 and +5.18 % at NA=6, on gated GPU time, 21 reps,
  palindrome order, all seven scored shapes; every bar-clearing arm is slower
  on all seven. The mechanism is arithmetic: halving `rows_per_simd` doubles
  the x re-read per lane per k-block and buys nothing else. Askeladd then found
  **three ranked board receipts** that shipped `rows_per_simd = 2`. See advisor
  error 54.
- 🔴 **Dead-width pruning is DEAD by Finding 27d.** Arms that delete the
  unreachable M=9 body produce byte-identical GPU objects.
- 🔴 **The NA-widening negative is now half explained.** Finding 27's
  zero-parameter occupancy prediction of **+0.0974 %** matches the measured
  G=1 half at **+0.1068 %**. The G=2 half at **+0.5516 %** is still open, and
  registers are ruled out. Suspects: the four non-kernel files in `ca9251b8`,
  or a `g17s` execution effect at NA=5.

**Theme 3 — ACHIEVED BANDWIDTH THROUGH TRANSFORM-OWNED LAYOUT. 🔴 DEAD. Do not
assign it.** The streaming families already run at 87 % to 99.6 % of the 273
GB/s M4 Pro peak. There is no 15 to 18 % of headroom to recover.

**Theme 4 — ALLOCATION QUALITY AND THE ONE-BIT `G` DECISION. E99, edward,
PR #101. 🔴 THIS IS THE LARGEST MEASURED GAIN OF THE CAMPAIGN AND IT IS
CLEARED TO SUBMIT.**

The shipped adaptive walk chooses a depth per round, and every round that
crosses `M = 5` pays a second full weight stream. Edward's margin gate clamps
that decision on one pre-round feature, the pending primary's top-2 margin.

```
cap 8 ABBA, 512 tokens, t = 9.4375
  ranked us/token on realised sequence  10,655.4 -> 10,312.1   +3.222 %
  MTP seconds/token                     0.031744 -> 0.030735   +3.180 %
  local serial-to-MTP ratio               2.3398 -> 2.4168     +3.289 %
  effective_mean_draft_len                6.3590 -> 5.7778     -9.140 %
  accepted_draft_rate                     0.8770 -> 0.9231     +5.25 pp
  fired 21 of 81, exact on every leg

cap 5 ABBA on the SHIPPED DEFAULT path, 512 tokens
  ranked us/token   11,122.54 -> 10,774.86   +3.126 %
  MTP s/token         0.033815 -> 0.032559   +3.713 %
  local ratio         2.194913 -> 2.278464   +3.807 %
  accepted rate         0.8927 -> 0.9347     +4.70 pp
  fired 21 of 98
```

Threshold sweep at cap 8: t=4.0 → +1.78 %; t=8.25 → +3.22 %; **t=9.4375 →
+3.22 %**; t=11.5625 → +3.01 %; t=16.0 → **−3.00 %**; t=1000 → **−5.66 %**.
The t=1000 row retires advisor error 43 by measurement: forcing the shallow
arm everywhere is a large loss, so the gate is selecting, not just shortening.

🔴 **The margin distribution explains why the threshold is not overfitted.**
`research/e99_margin_dist.py` reads the `m=` field already present in every
traced round.

| leg cap | rounds | p05 | p25 | p50 | p75 | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 78 | 0.731 | 9.906 | 14.250 | 15.922 | 17.591 |
| 6 | 85 | 0.825 | 9.250 | 14.250 | 15.750 | 18.100 |
| 5 | 97 | 1.050 | 11.125 | 14.625 | 16.000 | 18.325 |
| 4 | 110 | 1.056 | 11.906 | 14.906 | 16.125 | 18.519 |

The distribution is **strongly bimodal**: a low mode near zero, a high mode
near 14 to 16, and the region 4 to 12 is almost empty. `t = 9.4375` sits in a
gap in the data rather than at a tuned point, which is why the cap-8 sweep is
flat from 8.25 to 11.5625 and collapses at 16.0. Fired share is 0.20 to 0.26
at t=9.4375 across every cap from 4 to 8.

**Submit clearance issued.** Predicted published `3.32825 x 1.031 ≈ 3.431`
against a live frontier of `3.35026`, a margin of `+2.4 %` against a published
single-run sd of `0.277 %`. Do not rebase onto `41bad1c6` first.

**Theme 5 — KEEP THE ONE IN-FLIGHT YUKON SLOT OCCUPIED WITH A REAL
MECHANISM.** Amended by Finding 24 and advisor error 55. The slot has been
idle since 11:13Z and belongs to edward. Rivals resample for variance and five
to seven runs validate at once, but `program.md` forbids duplicate
submissions, so we answer variance with mechanism size, not with draws. E99's
`+3.1 %` is roughly forty times the mechanism a rival just used to take the
crown.

**Theme 6 — THE SMALL-DISPATCH AND LOW-EFFICIENCY POOL. Worth 2.0 % to 3.3 %
published.** The mechanism behind the whole pool is dispatch count and
occupancy at M = 5: activation tensors are tiny, [5, 5120] bf16 is 51 KB,
which cannot fill a 20-core GPU, so these kernels run at 10 % to 19 % of peak.
About 240 such dispatches per round at the FACT 8 GPU dispatch boundary of
3.87 us [2.63, 5.11] is about 929 us per round of pure boundary cost, 1.75 %
of the ranked round on its own.

1. **SDPA over the full-attention history, 0.64 % to 1.45 %.** GQA pair-head
   K/V reuse in `sdpa_vector` at D = 256, or an M = 5-aware path. 79.19 us per
   dispatch, 67 MB per round, about 53 GB/s. **QUEUED TO ALPHONSE behind the
   two E100 controls.**
2. **residual + RMSNorm prologue fusion into the quantized GEMM, 0.56 %.** Each
   GEMM threadgroup already reads the whole K = 5120 activation row, so it can
   compute the RMS reduction from data it is already loading. Two-phase kernel
   required. **COLLIDES with E100 in `quantized.h`. UNASSIGNED.**
3. **GDN prework, 0.40 % to 0.62 %.** Fuse into the recurrent step.
   Independent. **UNASSIGNED.**
4. **q/k norm + RoPE, 0.11 % to 0.17 %.** Fold into the KV write. Independent.
   **UNASSIGNED.**

**Theme 7 — THE HEAD SELECTION AND RERANK CHAIN. 🔴 NEW, opened by Finding 26.
E101, thorfinn, PR #103.** This is now the best-evidenced kernel-side theme,
and unlike Theme 2a it has an **external ranked receipt proving the mechanism
class converts**. A rival took the crown with 10 to 15 us per draft on the
rerank stage. Thorfinn holds 42.6 to 104.21 us per draft on the stage directly
above it, and chain A doubles the target again.

- **Chain C is built, exact and measured.** One `qwen_mtp_row_top32_partial`
  plus one `qwen_mtp_row_top32_finalize` replace 13 dispatches: net
  **−104.21 us per draft and −11 command buffers per draft**, with
  `draft_head` in situ falling 15,367.16 → 14,399.67 us per round, −6.30 %,
  and dispatches per round 461.1 → 293.1. Exactness at live geometry: 256
  trials, 0 mismatches, positive control detected.
- **Pricing frame.** Latency class under Finding 22. The lower bound is 11
  removed dispatches times the FACT 8 boundary of 3.87 us = 42.6 us per draft;
  the upper bound is the OPS=0 census attribution of 104.21 us per draft. The
  median-pair conversion is **+0.323 % at 40 us** and **+0.841 % at 104.21 us**.
  🔴 Do NOT multiply by 2.40 and then also divide by the ranked round; the
  division IS the amplification.
- **The primary statistic is the production `d_chain_us` per draft from the
  rung 4 seven-leg 512-token ABBA**, not the census. Round-level ABBA is a
  consistency check only: E87 section 8 returned −0.2977 % inside a 0.4221 %
  session null and was nevertheless real at +0.1117 % official.
- **Next: chain A** (40.420 us per draft off-arm, pair about 145 us per draft,
  about +1.1 % median pair), **then import the `41bad1c6` `Qwen35.swift` hunks
  only.** Chain C produces the 32 candidate IDs; the rival kernel consumes
  them. They compose. Prove with a row-digest check that their kernel's
  order-independence holds against thorfinn's finalize emission order.

### What died this round

- **Theme 1, bytes per weight.** Finding 23.
- **The `rows_per_simd` axis at every NA.** E76 section 3 and section 4, plus
  three ranked board receipts. Advisor error 54.
- **Dead-width pruning of the shared `affine_qmv_fast` entry point.**
  Finding 27d.
- **The `<T,5,5,true>` M=5 stream collapse end to end.** E100, +0.095 % at 512
  tokens depth 8, inside noise. The isolated −17.7 % does not reach the round.

### What survives from the earlier themes

- **Beagle is half the score and the other half is locked.** Section 0e stands:
  `published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic,
  botany)`. Beagle is worth 12.5x essays per point, and it is the only scoring
  prompt above the verify group boundary at `M = 5.382`.
- **Finding 22's two-class transfer law.** Stream-class percentages transfer at
  1.0, latency-class at 2.40, head-byte removal at the MEASURED 0.236,
  acceptance loss at 1.0. **Finding 26 is an independent ranked confirmation of
  the latency class**, so this law is no longer resting on one internal
  receipt.
- **The depth-price and depth-selection axis stays closed by E94** for
  *pricing* the walk. Theme 4 is alive because it *gates* the walk on a
  pre-round feature, which is a different mechanism.
- **The per-row verify slope stays closed by E97** at 82.7 % of the affine-4
  g64 batched peak.
- **The GDN recurrent step stays closed by E96** at 861 us per round.
- **The barrier-granularity lever stays closed** by Finding 21 plus FACT 19;
  measured idle is 840.4 us, 0.5 % of the round.

## 4. Potential next research directions

Ordered by measured evidence at the scored operating point, then by the
Finding 22 two-class transfer law. **Every entry that was priced from
`byte_share x streaming_share` has been deleted; see Finding 23 and advisor
error 53.**

1. **Chain A of the head selection chain. QUEUED, E101, thorfinn.** Chain C is
   already built and exact at −104.21 us per draft. Chain A measures 40.420 us
   per draft on the off arm, so the pair is about 145 us per draft, about
   **+1.1 % on the median pair**. Latency class.
2. **Import the `41bad1c6` rerank fusion. QUEUED, E101, thorfinn.** A rival's
   ranked receipt prices it at **+0.0803 % serial-free**. Take the
   `Qwen35.swift` hunks only. It composes with chain C: chain C produces the 32
   candidate IDs and their kernel consumes them.
3. **SDPA over the full-attention history. QUEUED, E100, alphonse.** 79.19 us
   per dispatch, 67 MB per round, about 53 GB/s, so latency class at
   **0.64 % to 1.45 %**. GQA pair-head K/V reuse at head dimension 256, or an
   M = 5-aware path.
4. **The two E100 path-execution controls. ORDERED, alphonse.** A free
   kernel-name census at forced depths 4 and 5, then a `<T,6,2,true>` dose leg.
   **Campaign-defining regardless of outcome:** one branch re-prices E76, E97,
   E98 and E102 as measurements of an unexecuted path, another overturns
   Finding 21's DRAM picture at G=2, and the third closes E100 cleanly.
5. **residual + RMSNorm prologue fusion into the quantized GEMM. UNASSIGNED,
   0.56 %.** Latency class. Collides with E100 in `quantized.h`.
6. **GDN prework fused into the recurrent step. UNASSIGNED, 0.40 % to 0.62 %.**
   Latency class, independent of every live experiment.
7. **q/k norm and RoPE folded into the KV write. UNASSIGNED, 0.11 % to
   0.17 %.** Latency class, independent.
8. **The G=2 residual of Finding 27.** The zero-parameter occupancy law
   explains the G=1 half of the `ca9251b8` negative exactly and leaves
   **+0.4542 pp** unexplained on the five G=2 prompts. Two suspects remain and
   registers are not one of them: the four non-kernel files in that tree, or a
   `g17s` execution effect at NA=5. Askeladd is running the `3ff80e86` sibling
   contrast, a natural experiment that isolates the shared-allocation question
   on ranked data with a pre-registered G=1-flat prediction.
9. **The three `rows_per_simd = 2` sibling contrasts** on `afb688fe`,
   `e617ef07` and `60a5ac1f`. These are ranked receipts for an axis E76 already
   killed locally; the contrast is cheap and settles whether the local kill
   transfers.
10. **The width-aware Q-row narrowed pack.** `84b9ef7b` proved the naive form is
    a real regression through steel GEMM `N % bn` misalignment on all 16
    full-attention layers, with a per-drafting-round coefficient of +820.3 us
    at `t = +7.62`. Narrow only when the dispatch will use `qmv`, that is at
    width 1. Reopen only with a width test in the source.
11. **Section 9 centroid padding 12,292 -> 12,296** so arm C stage 1 reaches
    `affine_qmv_fast`; `quantized.cpp:259` requires `N % bn == 0 && K % 512 == 0`
    with `bn = 8`. About 7.6 us per draft. Needs its own exactness gate.
12. **Section 12.3 free `_draftHeadW/S/Z`.** A rider, below the floor alone.
13. **A certified two-tier exact `lm_head` readout screen.** Unassigned,
    +0.3 % to +0.4 % ranked. Scepticism on record: Cauchy-Schwarz gives a bound
    about 14x larger than a typical logit gap.
14. **GDN scan dv-blocking**, the narrow dispatch switch at
    `quantized.h:1980`, entropy-gated early stopping, and a forced
    partial-accept census leg for GDN state replay. All under 1 %.
15. **E89 rung C.** Deferred with a written reopen condition after its premise
    was falsified at exact one-sided `p = 0.99997`.

### Deleted from this list this round

- **The transform-owned uint16 (scale, bias) index.** Built, proven lossless,
  measured at −0.35 %. Finding 23.
- **The transform-owned interleaved weight layout.** Theme 3, dead: the
  streaming families already run at 87 % to 99.6 % of DRAM peak.
- **A 12-bit packed index** and **the head-side affine-2 metadata index.** Both
  were priced from byte share. Finding 23 deletes the pricing method.
- **`rows_per_simd` 4 -> 2 as an escape from the register wall.** E76 measured
  it at +14.16 % slower at NA=5 on gated GPU time, and three board trees have
  shipped it. Advisor error 54.
- **Dead-width pruning of the shared `affine_qmv_fast` entry point.** Byte-
  identical GPU objects. Finding 27d.

### Advisor-owned work items

- **Rewrite the stop-list entry at `senpai/campaign-ledger.md:26410-26412`.**
  The `ca9251b8` NA=5 line rests on one ranked run, and Finding 27 now explains
  part of it. Also correct the mislabelled table at `:18692-18707`.
- **Correct every campaign statement that `rows_per_simd` 4 -> 2 is untried.**
  E76 measured it and three board trees shipped it.
- **Add the rival negatives to the stop list**: `ef365c52` de-islanding,
  timing-inverted on M5 because the islands are load-bearing for command-buffer
  phase, not for accuracy; `142db395` fc-bf16 zero-code mixed precision, a
  uniform −2.7 % timing tax because M5 prices dense-bf16 head reads at raw
  bandwidth; `9c30d69c` trained heads, coverage-limited and needing 20 to 100x
  more data.
- **Requote every `peak_live_regs` figure in the campaign record as an SSA
  liveness screen, not a register count.** Finding 27b. The numbers 163 and 125
  have been quoted as registers.
- **Settle the Finding 12 drama row.** The recovery rule gives 168 rounds; the
  hard-coded `ROUNDS` table in `research/board_same_schedule.py` carries 252.
  The ratio is exactly 1.5. Drama is not in the G=2 fit set, so nothing
  downstream moves, but the record must be consistent.
- **Productionise the scratch instruments** `whash.py`, `diag84c.py`,
  `seriallottery.py`, `thermalcouple.py`, `rounds_id.py` and `top2.py` as
  committed `research/` scripts.
- **Recalibrate the E95 least-squares `fixed` split**, which is 5.71x high on
  the GDN step.
- **Settle the residual 0.10 % of published-frame noise.** Thorfinn's frame
  decomposition reproduces every official score to 4.9e-15 and shows the
  median-of-eight selects beagle plus one of four prompts in 97.2 % of
  submissions, so published averages only **two** serial draws. About 134
  mode-matched pairs are needed; the current 18 give `r = +0.4286`,
  `se 0.1924`, `t = +2.23`.

## 5. Standing operating rules

- 🔴🔴🔴 **A SHARE OF ROUND TIME IS NOT A PRICE. A CONTROLLED REMOVAL AT THE
  SCORED OPERATING POINT IS A PRICE.** Advisor error 53. I priced E98 at
  +4.7 % to +5.5 % published from `byte_share x streaming_share`; the scored
  kernel converts metadata bytes at about 0.17 of that. Never multiply an
  isolated-cell effect by a share of round time and call the product a
  prediction. **Advisor error 56 is the same error committed again three days
  later** on E100's 512-token band, where I pre-registered −6 % to −12 % at
  depth 4 and −0.5 % to −1.5 % at depth 8 and both were refuted on the wrong
  side.
- 🔴🔴 **READ THE REPOSITORY'S OWN PRIOR EXPERIMENT BEFORE PRICING A ROUTE.**
  Advisor error 54. I briefed `rows_per_simd = 2` as an untried escape and
  built two rungs and one assignment order on it. `research/e76-results.md`
  had already measured it on gated GPU time and had already published the
  `g17s` register ladder I then sent a student to measure.
- 🔴 **`peak_live_regs` IS NOT A REGISTER COUNT.** Finding 27b. It is AIR SSA
  liveness at about 1.7x the machine allocation. Use it as an order-preserving
  relative screen only. For a level, use `xcrun metal-tt` against the target
  generation.
- 🔴 **A shared kernel entry point taxes every width when any one width
  widens.** Finding 27a and 27c. Price that flat tax before the mechanism's own
  gain.
- 🔴 **Price every proposal against the bandwidth floor before assigning it.**
  Divide total weight bytes by DRAM peak and compare with the measured round.
  Finding 21 says the round is at least 82 % weight streaming, so any mechanism
  that lives inside the remaining 18 % has a ceiling under 3.8 % of the round.
  **Advisor error 49 is exactly this check not being run**, and it cost three
  students a full generation each.
- 🔴 **A per-dispatch or least-squares attribution is not a measurement.** The
  8,112.6 us Gated DeltaNet figure came from `report_fixed()` in
  `research/e95_verify_census.py`, which spreads residual by byte pro rata and
  has no "belongs to no kernel" bucket. Only a removal arm, a repeat-dose slope
  or a round-level ABBA contrast prices a component.
- **The published score is
  `0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)`.** Report it
  as the headline of every per-prompt comparison. `mean7` stays as a mechanism
  diagnostic only; it is not the score and it has already cost us one crown.
- **Keep the one in-flight Yukon slot occupied with the best available real
  candidate.** Every official submission must carry a content delta we can name
  and price; comment-only resamples are retired. 🔴 **Yukon's dedupe is
  defeated by any byte, including the free-text `note` field of
  `mtp-head.manifest.json`** (Finding 24), so a bare resample is mechanically
  possible. `program.md` forbids it. **We answer rival lottery draws with
  mechanism size, not with draws.**
- **Report the serial-free score with every published score.**
- **Carry `sandbox=on|off` in the experiment identity tuple.** `--local-submit`
  runs inside the Seatbelt profile written by `benchmark.sh:1266-1307`;
  `research/e79_trace_leg.sh` sets `MLXFAST_NO_SANDBOX=1` and runs outside it.
  Absolute times from the two configurations are not comparable. The profile
  denies every file write except `/dev/null` at `:1294-1295`, so any research
  sink that opens a file silently produces nothing on a sandboxed leg.
- **The 0.0815 % per 1 % byte law is an average over the whole 428 MB head
  stream.** Do not apply it to one tensor. Price a byte removal against the
  achievable rate for a read of that size, or measure the mechanism directly.
- **The local achievable read bandwidth is 265 GB/s.** Size-matched: 274 to 276
  at 157 MB, about 265 at 330 to 428 MB, about 260 above 1 GB, 403 to 430 at
  16 MB which is cache. 226.035 and 245.2 GB/s are retired.
- **A byte model is valid only when achieved bandwidth is held constant.**
  Working-set reduction and byte reduction are distinct levers.
- 🔴 **Price every local gain through the FINDING 22 TWO-CLASS transfer law.**
  The earlier four-row table's derived per-row-verify factor of 1.5 and derived
  fixed-and-launch factor of 0.67 are **RETIRED**; they were built by
  attribution, not measurement.

  | work class | multiply local percentage by | basis |
  |---|---:|---|
  | STREAM work, at or near DRAM peak | **1.00** | `local rate / ranked rate` |
  | LATENCY work, well below peak | **2.40** | ranked-to-local round ratio |
  | proposal-head BYTE removal | **0.236** | MEASURED, E87 arm C |
  | acceptance loss | **1.00** | accounting identity |

  Corrected closure thresholds against a LOCAL cost: **stream class dead below
  0.160 %, latency class dead below 0.067 %.** A proposal the drafter fails to
  retrieve is rejected by the target on any machine, so an acceptance penalty
  never shrinks on transfer while the byte gain that bought it shrinks by four.
  🔴 **Finding 26 is an independent ranked confirmation of the latency class**:
  a rival's 10 to 15 us per draft removal was predicted at +0.081 % to
  +0.121 % and measured at +0.0803 %.
- 🔴 **Do NOT multiply a latency saving by 2.40 and then also divide it by the
  ranked round.** The division IS the amplification. Advisor error 52 was the
  opposite mistake: applying the 0.236 BYTE factor to a LATENCY mechanism, a
  12x understatement that retired a whole live theme for a day.
- **A bit-exact change cannot move a draft length.** `effective_mean_draft_len`
  is a free exactness detector.
- **Price an issue-count change from translated machine text, never from AIR.**
- **Carry an instruction counter in every host-state measurement.**
- **Publish the per-leg host-state stratum before any pooled number**, using the
  arm-blind 1,500 us absolute host-phase gate.
- **plutarch, prefill and serial are mechanism-breadth controls, not mode
  controls.** plutarch correlates with the mode at r = +0.043.
- **Read `sd7` before `mean7`.** sd7 above about 0.35 on a same-schedule pair
  means cross-mode; quarantine the pair.
- **Group ranked comparisons by the scored-surface tree digest first**:
  `git ls-tree <branch> Sources Vendor mtp-head.manifest.json`.
- **A promotion is a draw, not a measurement.**
- **An isolated-cell harness over-states recoverable time** — by 3.63x in E78
  and 33x in E91.
- **Leg totals overstate small effects by up to 4x.** Use paired per-round
  medians with the depth sequence held identical.
- **Freeze the commit before a gate leg.** Land logger changes between legs,
  never inside a job.
- **Research instruments go in `Tests/` or `research/`, never `Sources/` or
  `Vendor/`.** Deletion is the default for a closed axis's knob.
- **When a student's measurement contradicts the advisor's model, the
  measurement wins and the advisor retracts in writing before they spend GPU.**
- **Verify every claim about the scored surface with a repository-wide grep
  before it becomes an instruction.**

---

## 6. Student board

| PR | student | experiment | state |
|---|---|---|---|
| #101 | edward | E99 oracle allocation bound, then the margin gate | 🔴 **`status:wip`. THE LARGEST MEASURED GAIN OF THE CAMPAIGN AND CLEARED TO SUBMIT.** The margin gate measures **+3.222 %** at cap 8 and **+3.126 %** at cap 5 on the ranked us-per-token statistic, exact on every leg, with a six-point threshold sweep whose t=1000 row (−5.66 %) proves the gate selects rather than merely shortens. His new `research/e99_margin_dist.py` shows the top-2 margin is strongly bimodal with an almost empty band from 4 to 12, so `t = 9.4375` sits in a gap in the data and is not a tuned constant. Cap-4 legs are running on a single clean worker. He owns the idle Yukon slot. |
| #103 | thorfinn | E101 selection chain, custom top-K | 🔴 **`status:wip`. Chain C is built, exact and measured.** Two fused kernels replace 13 dispatches for a net **−104.21 us per draft and −11 command buffers per draft**; `draft_head` in situ falls 6.30 % and dispatches per round fall 461.1 → 293.1. Exactness at live geometry: 256 trials, 0 mismatches, positive control detected, `swift test` adds zero new issues against his own stashed base in the same tree. Rung 4 is a seven-leg 512-token ABBA on one clean worker flipping only `MLX_E101_ROW_TOP32`. Then chain A, then import the `41bad1c6` rerank hunks. **A rival just took the crown with the next stage of his own chain, which calibrates his price and validates his axis.** |
| #102 | alphonse | E100 fewer weight streams per round | 🔴 **`status:wip`. The end-to-end result is a null and the discrepancy is the finding.** Isolated M=5 collapse is −17.7 % ± 1.7 % with 0 of 45 row-digest mismatches, but the 512-token depth-8 `B A A B` measures **+0.095 %** inside noise, and his round reconstruction predicts 25.3 ms per round where the session measures 1.47 ms — a factor of 17. Two path-execution controls are ordered ahead of any further timing: a free kernel-name census at forced depths 4 and 5, then a `<T,6,2,true>` dose leg. Theme 6 item 1, SDPA over the full-attention history, is queued behind them. |
| #104 | askeladd | E102 g17s occupancy and the dispatcher census | 🔴 **`status:wip`. Delivered Finding 27 with zero GPU time and no candidate file touched.** The dispatcher allocates exactly what its widest inlined body allocates, 7 of 7 arms on both generations; `peak_live_regs` is AIR SSA liveness at 1.7x the machine allocation and is a relative screen only; the pipeline arbiter is saturated at 1024 threads for every arm; and dead-width pruning produces byte-identical GPU objects, so rung 2 closed with a negative and was never built. His zero-parameter occupancy prediction of +0.0974 % matches the measured G=1 half at +0.1068 %. Rung 0a and the `3ff80e86` sibling contrast are running. **No GPU time approved; keep E102 CPU-only and land it.** |
| #100 | askeladd | E98 transform-owned weight metadata index | ✅ **MERGED at `ad8403f1`.** Decisive negative that closed a whole theme. He built all three JIT variants, proved the uint16 (scale, bias) index bit-lossless in-kernel across 21 cells and up to 1,986,560 outputs with a positive control in all three shapes, and then measured the removal at **−0.35 % against a minimum useful 1.5 %**. That produced **Finding 23**: in this kernel family byte share times achieved bandwidth over-prices byte reduction by about 5x. He also refuted LUT residency as the cause and left a clean reopening condition. |
| #99 | alphonse | E96 Gated DeltaNet recurrent step | ✅ **MERGED at `cd0a89da`.** Decisive negative that reorganised the campaign. The step costs 861 us per round, not the 8,112.6 I briefed. His isolated census measured streaming families at **2.005x the whole model per round, 88.6 % of round time and 249.55 GB/s**, which killed Theme 3. He also proved `MLX_E58_BUFFER_LIMIT_OPS=0` is required, invalidating every earlier `OPS=1` isolation. |
| #89 | thorfinn | E87 coarse draft shortlist, arm C, plus section 8 | ✅ **MERGED at `d5075d4c`. Produced the promoted crown `f04b102e` at 3.32824629 — the first time senpai has held the official frontier.** He then repriced his own section 8 against the official per-prompt receipt, found the byte-transfer factor understated it by 12x, and produced **Finding 22, the two-class transfer law**, the largest methodological correction of the campaign. |

Each student has one physical Mac: Apple M4 Pro, `applegpu_g16s` generation 16,
20 GPU cores, 48 GiB, 10 performance cores and 4 efficiency cores, DRAM peak
273 GB/s. The ranked runner is an M5, `applegpu_g17s` generation 17, 128 GiB.
The advisor is co-located with edward and must not run builds or GPU work.

**File-overlap schedule.** E100 owns `quantized.h:1918-1990`. E101 owns the
selection and rerank region of `Qwen35.swift`. E99 owns `costModelDepth` in
`Qwen36MTPBlockSession.swift`. E102 is CPU-only and touches no candidate file.
The four live experiments do not collide. Theme 6 item 2, the residual and
RMSNorm prologue fusion, must wait for E100 to resolve, because it lands in the
same `quantized.h` region.

**Submission discipline.** One in-flight Yukon slot. It belongs to edward until
his E99 receipt lands. Thorfinn submits next, after chain A and the `41bad1c6`
import, and only with the fused path compiled as the default.
