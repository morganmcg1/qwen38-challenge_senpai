# Advisor brief, 2026-08-19: the score is two prompts and one width

**Why this is a file and not four PR comments.** The GitHub REST API has been
returning HTTP 403 for every read and every mutation for the last hour (the
item-75 rate-limit pattern; git transport is unaffected). I could not deliver
this as PR feedback, so it is on the advisor branch instead. **Everyone rebases
onto this branch, so read the section with your name on it.** I will mirror the
per-student sections into the PRs as soon as the API recovers; if you see both,
they are the same content and this file is the original.

Full derivations and receipts are campaign-ledger items **87–95**.

---

## 0. A retraction that affects three of you

Last cycle I told alphonse, thorfinn and askeladd that the ranked ratio has
**two** movable derivatives because the serial baseline is measured in the same
thermally-gated session as the candidate. **That is wrong and I withdraw it.**

Same *session*, pinned *tree*. Receipts:

- the ranked workflow runs `--baseline "${MLXFAST_QWEN_MTP_BASELINE_RESOLVED}"`
  = `/opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current`
- the head-delivery contract is `mtp_head_delivery.applies_to =
  "candidate_leg_only"`
- workflow `:2374-2375`: *"the BASELINE leg keeps the section-9d pinned head out
  of the pinned baseline tree unconditionally … the denominator never moves."*
- population test over 402 scored rows: `baseline_serial_seconds_per_token_mean`
  sd **0.106 %** (0.037907–0.038111) against
  `candidate_mtp_seconds_per_token_mean` sd **25.9 %**;
  `corr(score, serial) = +0.049`, `corr(score, candidate_mtp) = −0.942`

What misled me was `:3083-3085` — *"no pinned reference is consulted anywhere in
the scoring path"* — which retires the `noop_decode_speedup` divisor, not the
serial binary. **There is exactly one derivative: candidate MTP s/tok.**

Consequences: the ~20 % "ladder exploit" I escalated **does not exist** and the
escalation is withdrawn. The serial leg is still worth reporting as a free
in-session thermal control; at 0.106 % population sd it is the tightest
instrument on the board.

My own lesson, recorded in the ledger: I shipped a correction to three students
and an ethics escalation on documentation alone, when one authenticated GET would
have refuted it.

---

## 1. Use this tool. It changes what counts as a good experiment here

```bash
python3 research/ranked_telemetry.py --refresh --top 14
python3 research/ranked_telemetry.py --profile 0cd0a6b4    # current board top
python3 research/ranked_telemetry.py --price beagle         # R vs n vs implied hbar
python3 research/ranked_telemetry.py --spread               # per-prompt noise
```

It wraps `GET /api/benchmarks/5d1ee4d7-.../submissions` with the
`YUKON_API_TOKEN` already in the campaign env. 626 submissions, each with
`officialMetrics.per_prompt[8]` carrying `raw_ratio_of_means`,
`serial_seconds_per_token_mean`, `mtp_seconds_per_token_mean`,
**`effective_mean_draft_len`**, `non_drafting_round_count`, `parity_ok`,
`head_provenance_sha256` — plus every competitor's full public `note`. This sat
unread all campaign because `yukon submissions` truncates the `metrics` column at
80 characters, ignores `COLUMNS`, and has no `--json`.

---

## 2. The score is two prompts

`median_rule = even_n_mean_of_two_central_order_statistics` with n=8, so the
score is the mean of the **4th and 5th ranked** per-prompt ratios. Board top
(ofou, `0cd0a6b4`, **3.24929398547457**):

| rank | prompt | ratio | `effective_mean_draft_len` | mean M |
|---|---|---:|---:|---:|
| 1 | plutarch | 1.2560 | 0.154 (449 non-drafting) | 1.15 |
| 2 | drama | 1.9231 | 2.298 | 3.30 |
| 3 | travel | 2.1895 | 2.656 | 3.66 |
| **4** | **beagle** | **3.1433** | **4.533** | **5.53** |
| **5** | **medicine** | **3.3553** | **4.768** | **5.77** |
| 6 | essays | 3.3907 | 5.425 | 6.43 |
| 7 | republic | 3.4144 | 5.270 | 6.27 |
| 8 | botany | 3.4491 | 5.776 | 6.78 |

`∂score/∂prompt` is **0.5 for beagle and medicine, 0 for the other six.** The
bottom three are unreachable: travel would need +43 %, plutarch +150 %.

**Kill any proposal aimed at plutarch, drama or travel.** Receipt that this is
not theoretical: two competitors got plutarch to draft — n 0.154 → **2.897**,
ratio 1.2489 → **2.2209**, a **+78 %** gain on that prompt, non-drafting rounds
449 → 0 — and **both still scored below the board top** (xadenryan 3.16292,
Lieisyourlie 3.15370), because plutarch merely moved from rank 1 to rank 3 while
the same aggression pushed beagle's implied per-row price from 0.1645 to 0.1699.
Aggregate throughput work on the six non-binding prompts is worth exactly zero,
and is *negative* when it raises per-row cost on the two that bind.

### 2a. And the whole board is competing on one axis

Across the top 14 accepted rows from 14 different solvers (3.19088–3.24929),
`effective_mean_draft_len` has standard deviation **0.000 %** on every one of the
eight prompts — identical to every printed digit — while `raw_ratio_of_means`
varies 0.26–0.86 %. Identical accepted-row trajectories with different ratios
means **every point of separation at the top is per-row verify cost, not
acceptance.**

And deeper is not better. Within the same head `559b24eb`, solving
`hbar = [(1+0.99n)/R − 1]/n` on beagle:

| beagle n | beagle R | implied hbar | score | solver |
|---:|---:|---:|---:|---|
| 4.2793 | 3.1103 | 0.1598 | 3.18367 | jeromelaurens |
| **4.5327 (default)** | **3.1433** | **0.1645** | **3.24929** | **ofou** |
| 4.3839 | 3.0603 | 0.1699 | 3.16292 | xadenryan |
| 4.6019 | 3.0062 | 0.1843 | 3.07216 | andreolf |
| 4.7358 | 3.0626 | 0.1810 | 3.16408 | welttowelt |

Nine rows drafted away from the default; all nine scored below the default
cluster's best. Going n 4.533 → 4.736 made beagle 2.6 % *worse* per token. These
are different kernels so it is confounded, but the sign is uniform. **The
marginal row at the ranked operating point costs more than it returns.**

Read `hbar` as a re-parameterisation of the observed `(R, n)` under an assumed
`alpha`, **not** a measured price: one equation, two unknowns. Use it to compare
rows at equal `n`.

---

## 3. The main line: make M=6 a single weight pass

`M = n + 1`, so the binding widths are M ≈ 5.5–5.8, i.e. the round distribution
straddles M=5, 6, 7. `quantized.h:1154` states the dispatch rule: *"IPG =
ceil(M / ceil(M / 5)): the fewest weight streams reachable at NA <= 5."* So
`passes = ceil(M/5)` and per-row weight traffic is `passes/M`:

| M | IPG | passes | per-row traffic |
|---:|---:|---:|---:|
| 3 | 3 | 1 | 0.333 |
| 4 | 4 | 1 | 0.250 |
| 5 | 5 | 1 | **0.200** |
| **6** | **3** | **2** | **0.333 ← worst in the table** |
| 7 | 4 | 2 | 0.286 |
| 8 | 4 | 2 | 0.250 |
| 9 | 5 | 2 | 0.222 |

**The table is already pass-optimal at NA ≤ 5** — every cell equals
`ceil(M/ceil(M/5))` — so there is no IPG win left anywhere without raising NA.
That also settles the contested M=8 cell in our HEAD's favour: `<T,8,4>` is 2
passes, the promoted `<T,8,3>` is `ceil(8/3) = 3`. Stop queueing it.

M=6 cannot reach one pass at NA ≤ 5: IPG 6 needs NA=6; **IPG 5 is illegal**
(`6 % 5 == 1` trips `static_assert(M % IPG != 1)`); IPG 4 and 3 are both 2
passes. **NA=6 buys exactly one cell, M=6, and takes it from 0.333 to 0.167 —
the best per-row traffic in the table — directly under beagle and medicine.**

Two independent confirmations:

- **Alphonse's E30**, a local six-way round tape sharing no code with the ranked
  telemetry, finds that post-E27 the best width flipped 8 → 9 and **70 % of all
  remaining best-width headroom is the M=6 rounds** (160.3 of 229.8 ms; M=6 at
  23.71 ms/tok against the best width's 20.74).
- **The competitors have not touched it.** scarletbright (`72ce82dc`, +1.84 %)
  says outright that *"every other `bits == 4` dispatch cell, the pair kernels,
  the `_m`/wide tables … are all untouched."* SSHdotCodes (`3a995c2b`) worked
  **M=8 only**, and described the mechanism exactly as we do — *"the grouping
  changes weight-tile stream count, not row arithmetic … avoiding the third
  weight stream paid by 3+3+2"* — for **+0.23 %** from that single cell
  (`578535f7`). Across all 626 notes, nobody mentions IPG, `rows_per_simd`,
  `ceil(M/IPG)`, NA, or widths 5/6/7.

### 3a. Register model, so there is something to falsify

Live per-thread registers in `qmv_fast_crossrow_affine4_g64_wide`:
`acc[rows_per_simd]` and `partial[rows_per_simd]` are each `RS × NA` floats;
`sums` and `a0..a3` are `5 × NA`; `packed[RS][4]` is `2 × RS`; `scale_local` and
`bias_local` are `2 × RS`. Hence

```
regs(NA, RS) ~= 4 + (2*RS + 13)*NA + 4*RS
```

**Exact** against the measured AIR ladder at RS=4: NA=2 → 62, NA=3 → 83,
NA=4 → 104, NA=5 → 125 (all measured); NA=6 → 146 predicted against 144
measured. Extrapolating:

| RS | NA | predicted regs | second alloca? |
|---:|---:|---:|---|
| 4 | 5 | 125 | no (measured) |
| 4 | 6 | 146 | **yes — measured 144, 2 allocas** |
| 2 | 6 | **114** | **predicted no** |
| 2 | 7 | 131 | predicted marginal |

**Pre-registered prediction: `rows_per_simd = 2` with NA=6 lands near 114
registers with one alloca, clearing the spill cliff and making M=6 single-pass
buildable.** The RS coefficient is derived from source structure, not measured,
so it is a structural extrapolation and the probe is the falsifier.

### 3b. The two costs you must measure rather than assume

- **Launch geometry is coupled.** `out_row = int(tid.y) * 8 + int(simd_gid) * 4`
  (`quantized.h:1176`) hardcodes 4 rows per simdgroup and 8 per threadgroup
  against `constexpr int num_simdgroups = 2`. RS=2 means
  `tid.y * 4 + simd_gid * 2` and a doubled grid y-extent. Getting this wrong
  produces silently wrong logits, so gate on bit-identical output first.
- **Activation re-reads double.** Each simdgroup streams `RS` weight rows but
  *all* `NA` activation vectors over the full `in_vec_size`. At RS=4, NA=6,
  K=5120 that is 10,240 B of weights against 61,440 B of activations per
  simdgroup — activations are already **6×** the weight traffic per simdgroup.
  Halving RS doubles the simdgroup count and so doubles activation reads. The
  reason it can still win is that the ~61 KB activation slab lives in cache while
  the weight stream is the DRAM cost, and passes are what multiply DRAM weight
  traffic. **That is a hypothesis about which level of the hierarchy binds, and
  it is the real content of the experiment.** Report L2-versus-DRAM reasoning,
  not just round time.

Encouraging outside data point: scarletbright shipped a **doubled** per-lane
footprint (`values_per_thread = 32`) to the ranked M5, pre-registered spilling as
their falsifier, and it did not trigger. Different kernel and bit width, so it
does not transfer — but it weakens the assumption that the ranked box has less
register headroom than we fear.

---

## 4. Per student

### askeladd — E32, PR #37. This is now the campaign's main line.

Everything in §3 is yours. Grid unchanged; **prediction 3 (the register model) is
the priority** and the M=9 cell matters less than I implied, because E30 closed
it and it is no longer the expensive width.

Do the **compile-only AIR read first** — `research/crossrow_na_probe.metal`
already carries the NA=6 arm behind `CROSSROW_NA_PROBE_WIDE`, so RS=2 × NA=6
costs **zero GPU time** to settle. Report that number before spending a timing
slot. Also note the `static_assert(NA >= 2 && NA <= 5)` at `quantized.h:980` is
now the live ceiling (E27 raised it from 4), and the probe's comment saying
"NA > 4 trips the production assert" is stale.

### edward — E25, PR #29. Do NOT open the 4 GPU-h arm D confirmation.

Your terminal result is good work and your §7.3 ranked-transfer caveat was
right. Two things you did not have:

**(a) Arm D would be a structural ~18 % ranked loss.** PRICE sets max depth to
exactly 3 (`depth_ge_4_realised = 0`). Every scoring prompt realises 4.53–5.78,
and the two that decide the score sit at 4.53 and 4.77 — **your local tape's mean
depth of 2.386 (per-prompt 2.058–2.737) never visits the scored region at all.**
Holding alpha and each prompt's `hbar` fixed and moving n to 3 collapses the five
deep prompts into 2.66–2.74, putting the central pair at ≈2.66. Assumption-free
version: to *hold* 3.244 at n=3 you would need `hbar <= 0.0746` on the central
prompts, under half the shipped price.

**(b) Your binding coefficient was repaired by a merge 18 hours after you
measured it.** `M = n + 1`, so your `T(3)` is M=4 and your `T(4)` is **M=5**, and
`measuredRowStepRatio[3] = 0.442442` is the M=4 → M=5 step. Pre-E27, M=5
dispatched `<T,5,3>` = `ceil(5/3)` = **2 passes** while M=4 was 1 pass — your
"cliff" **was the 1 → 2 weight-pass boundary**. E27 (`cf2c0db2`, 22:56) made M=5
single-pass; your base `0d2eef9c` is 04:49 and
`git merge-base --is-ancestor cf2c0db2 0d2eef9c` returns non-ancestor.
Propagating E27's measured M=5 ratio 0.7990 through your own table:

```
T(3)      = 91.2664 ms                    (M=4, untreated)
T(4)_pre  = 91.2664 * 1.442442 = 131.658  (M=5, 2 passes)
T(4)_post = 131.658 * 0.7990   = 105.195  (M=5, 1 pass)
new step  = (105.195 - 91.2664)/91.2664 = 0.1526
```

**0.1526 against the shipped 0.153.** Predictions for the refit: (i) the d3
coefficient collapses onto the shipped curve and arm D's mechanism evaporates;
(ii) **a new cliff appears one step later, at d4 — the M=5 → M=6 step** — because
that is where the 1 → 2 pass boundary now lives. Register your own intervals
before you run.

The load-bearing generalisation is yours and it is worth more than arm D ever
was: **the true verify cost is a step function of `ceil(M/5)`, and the shipped
smooth `h/(1+d·h)` price cannot express that at all.**

Deliverable, rebased onto this branch, one build and one tape on your existing
fixture and no more: refit the price vector with the same instrument gate (the
one that reproduced all 1947 taped depths bit-identically — that gate is why I
trust this); **remove the arm D edit from `Sources/`**, because a hard depth-3 cap
must not land in the research base; keep the §3 prefill correction, the four
confound controls, the §5 fixed-window accounting and the §5.1 adjudication,
which all stand on their own. Then solve `(alpha, hbar)` per prompt from the
ranked telemetry instead of assuming alpha — implied per-row price is ≈0.165
beagle, 0.149 medicine, 0.155–0.164 republic/essays/botany against 0.306 drama
and 0.248 travel, so the shipped scalar is mildly conservative on the deep
prompts and badly wrong on the shallow ones. **A price that is depth-conditioned
toward buying more rows on the deep prompts is the interesting arm, and it is the
opposite sign to arm D.**

If the refit shows the cliff is gone, say so plainly. That closes a whole family
of depth-truncation proposals and is a fully respectable result.

### alphonse — E30, PR #35. Accepted, and it is the best-designed experiment of the campaign.

The width-specificity argument is the right design: one treated width moving
~20 ms while four untreated widths stay flat within ±1.3 ms in the same arm, with
D0b entering 20 °C hotter and still measuring M=9 slightly faster. That is a
clean natural experiment and the two-instrument agreement to 0.04–0.09 pp on
`bound_M8` means it does not rest on your own trace. **My threshold of < 2.0
failed and your registered point of 2.58 was right** — logged as an advisor
calibration miss.

Three follow-ups. First, your residual analysis is what promotes M=6 to the main
line, so state that conclusion prominently — right now it is buried in the P1
table. Second, one provenance discrepancy to chase: you report the declared head
as sha256 `7bbb40de…`, 270,408,194 B, while the ranked rows for that same
`hf:amal-david/...@ae62827` report `head_provenance_sha256`
`559b24eb…` at 427,742,600 B. The ranked number is a **tree digest** (sha256 over
`"<file sha256>  <relative path>\n"` lines, `LC_ALL=C` sorted, top-level
`README.md` excluded) so the digests are not comparable — but **the byte counts
should be**, and 270 MB against 428 MB is a 158 MB gap. Either our local head
tree is not what the ranked box materialises, in which case no local acceptance
measurement transfers, or one of the two counts is scoped differently. Worth one
`find -type f | xargs wc -c`. Third, note the ranked box enforces declared
sha256 **and** byte count by exact equality at workflow `:2491-2502`.

### thorfinn — E31, PR #36. Accepted as a terminal negative, and the corrections are worth more than the negative.

Three deliverables, all clean: a per-boundary cost of **−36.5 µs (95 % CI −170.3
… +97.4)** closes the axis; `MLX_MAX_MB_PER_BUFFER` is a **mebi-element** cap
because `buffer_sizes_` accumulates `array::data_size()`, which `array.h:346`
documents as units of `item_size` — 4× the implied bytes for the 4-bit
`uint32`-packed backbone; and `gpu::finalize` commits without `notify_new_task`,
so ladder rungs never enter the `MAX_ACTIVE_TASKS = 10` accounting. Zero timed
arms and it still retired a direction — that is the right call.

Your **50 ops/512 ranked against 64 ops/128 on this host** finding is now
load-bearing far beyond E31: it is the second named mechanism for the local↔ranked
transfer gap, and it means no local dispatch-batching measurement transfers
without an explicit argument. Please promote it out of the audit and into the
ledger framing.

Also: one item of mine to retract to you specifically. Item 4 of my previous
feedback treated the `asyncEval` ladder as a scoring artifact. It is not — it is
an honest **candidate-side S=1 win**, and plutarch is its ranked read-out (449
non-drafting rounds, ratio 1.2560, i.e. our S=1 decode is ~1.25× the pinned
baseline's). Your commit-geometry negative stands independently. Your next
assignment is **not** M=8 — §3 settles that cell in our favour — so expect
productionising whatever askeladd's E32 probe returns.

---

## 5. Standing calibrations, unchanged

- **Ranked noise.** Per-leg repeatability 0.106 %; byte-identical trees agreed to
  0.008 %; the **top-10 cluster spans 0.53 %** against per-prompt ratio spreads
  of 0.26–0.86 %. Most of the visible ordering at the top of that board is noise.
  **Require ≥1 % projected gain before spending a ranked slot; under 0.5 % is
  unresolvable.**
- **Never project a local ratio.** The same build scored **1.766 local against
  3.069 ranked**. Direction certain, magnitude soft — 300 versus 512 tokens,
  different prefill handling, a different width mass, and per E31 not even the
  same command-buffer geometry.
- **A local fixture whose depth histogram does not overlap the ranked one cannot
  test a depth policy at all, in either direction.** That is the generalised form
  of the arm D lesson and it applies to every schedule experiment from here on.
- **Pre-submission gate.** `yukon submit` is a REPLACE overlay, so run
  `git diff --stat <promotedSourceRef> HEAD -- Sources Vendor
  mtp-head.manifest.json` and account for **every** deletion before submitting.
  `promotedSourceRef = 5068eb8d0bae032faca6e901de398fc732531160`.
- **Never ship an MTP head that has not beaten the best public head on ranked
  evidence.** Our own LR3 head cost ~5 %: it proposed 5–13 % fewer accepted rows
  on 7 of 8 prompts than `559b24eb`, which the whole frontier ships.
