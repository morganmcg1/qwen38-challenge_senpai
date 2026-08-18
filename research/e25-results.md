# E25 — measured per-row draft price

**Credit: this experiment implements thorfinn's E22 follow-up #1 — the two-piece
boundary-aware marginal price, arm C in that proposal.** The idea, the diagnosis
of the scalar fit's failure mode, and the shape of the fix are thorfinn's. This
document contributes the instrument gate, the measurement of the price curve, the
confound controls, and the timed 8-prompt evaluation. I read the proposal as
quoted in the PR #29 assignment body and did not inspect thorfinn's branch, so I
cite it by name rather than by commit.

- Assignment: `qwen38-r1-e25-per-row-draft-price` (PR #29, revision `r1`)
- `BASE_SHA`: `0d2eef9cac75d890de06a5eef4fd686c3c34c1ef`
- Candidate file: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (single file, +36/−2)
- Host: this M-series host, single GPU. **Not the ranked M5.**
- W&B group: `qwen38-r1-e25-per-row-draft-price`

> **Sections 0–10 are the r1 record and are left unchanged as a scientific
> record.** Revision r2 re-based, re-measured, and in several places *overturned*
> them. Read **§11 onward** for the current conclusions, and read §11.2 for the
> exact list of r1 numbers that r2 retires. Where r1 and r2 disagree, r2 wins.

---

## 0. The mechanism in one sentence

The shipped schedule prices every proposed row at the same scalar `h = 0.18`, so
the marginal row costs `h / (1 + d·h)` — a curve that *only ever falls* with
depth. The measured cost curve does not fall; it spikes 4.4× at the fifth
verified row. Arm D replaces the scalar price with the measured per-row price,
floored by the shipped curve.

```swift
private static let measuredRowStepRatio: [Double] = [0.0, 0.095904, 0.152261, 0.442442]

private static func rowPriceCoefficient(_ depth: Int) -> Double {
    let h = headStepCostRatio
    let shipped = h / (1.0 + Double(depth) * h)
    guard depth < measuredRowStepRatio.count else { return shipped }
    return Swift.max(shipped, measuredRowStepRatio[depth])
}
```

The `max` with the shipped curve is load-bearing in two ways: depths 0–1 stay
**bit-identical** to the promoted schedule (0.180000 and 0.152542 are both the
shipped values), and the price becomes a pure *truncation* of the promoted
schedule. It can only ever stop a draft chain earlier than the base — never
extend one into behaviour no tape has evaluated.

---

## 1. Instrument gate (passed before any measurement was trusted)

Before measuring anything, I checked that the per-row form can reproduce the
promoted schedule exactly. Setting `measuredRowStepRatio = [0.18] × 8` must
collapse the new code path onto the old one.

- Replayed all **1947** taped rounds through both forms.
- **0 mismatches** — every chosen depth bit-identical.
- 6580 walk steps, max |threshold error| **6.67e-7**.

Without this gate, any depth change could have been a refactor bug rather than
the price change. With it, every depth difference below is attributable to the
price curve alone.

---

## 2. The measured price curve

Tape: `.mlxfast-private/e21/runs/probe-<prompt>-S18I`, 8 prompts × 512 tokens,
1947 rounds, 4645 proposed rows, 2151 accepted, 4098 emitted.

| d | mean round T(d) ms | n | relative step (T(d+1)−T(d))/T(d) | shipped price |
|---|---|---|---|---|
| 1 | 72.2749 | 193 | 0.095904 | 0.152542 |
| 2 | 79.2064 | 995 | 0.152261 | 0.132353 |
| 3 | 91.2664 | 583 | **0.442442** | 0.116883 |
| 4 | 131.6465 | 167 | 0.087386 | 0.104651 |
| 5 | 143.1506 | 9 | — | 0.094737 |

The shipped scalar charges **0.1169** for the step that actually costs
**0.4424**. That is the entire mechanism: the promoted schedule systematically
under-prices the fifth verified row and therefore buys it too often.

### 2.1 Prefill correction to the advisor's §3 anchor

The assignment quoted Σ round timers as 169.0187 s. Measured: **169.2410 s**.
More importantly, the ratio the price depends on must be computed prefill-free
on *both* sides:

- Σ MTP legs 202.68637 s, measured Σ prefill **31.9678 s**
- ⇒ round timers cover **99.135 %** of true decode
- `a_leg` 74.5261 ms, `a_true` 66.7200 ms
- h(leg/leg) 0.166344; h(traced/leg anchor) 0.069730
- **h prefill-free on both sides = 0.131697**

So the instrument inflation is **1.26×**, not the 2.4× implied by mixing a
prefill-inclusive denominator with a prefill-free numerator.

### 2.2 Confound controls

The d3→d4 spike survives every control I could construct:

- **Position**: depth and token offset are uncorrelated (mean offset 246–265 for
  d = 1..4). OLS offset slope **−0.067 ms per 1000 tokens** — the wrong sign and
  ~600× too small to explain a 40 ms step.
- **Position, non-parametrically**: T(4)−T(3) ≈ 40 ms replicates independently in
  **all four** offset quartiles (n = 44/36/38/49).
- **Accepted-row count**: T is flat in accept count at fixed depth (spread
  ≤ 3.6 ms), so the spike is proposal/verify width, not acceptance work.
- **Price vs cap**: the price — not a cap — chose **1935/1947 (99.4 %)** of taped
  depths, so the curve is what is actually steering. Offered-cap histogram
  `{5: 1793, 8: 154}`.

### 2.3 §5 adjudication: the corner is unreachable

The assignment asked whether a monotone-price corner case could fire. It cannot:
corner slack is **−0.32733**, and **0 fires in 400 000 monotone draws**. No guard
was needed, so none was added.

---

## 3. Arm selection

Offline replay against the tape (base 41.29844 ms/token):

| arm | description | pooled | median-of-8 | rows saved | deepen requests |
|---|---|---|---|---|---|
| A | no-op control | 0 | 0 | 0 | 0 |
| B | hard cap 3 | +3.9403 % | +3.3042 % | 185 | 0 |
| C | two-piece price | +5.2745 % | +4.6881 % | 609 | 187 |
| **D** | **price floored by shipped curve** | **+5.2745 %** | **+4.6881 %** | **609** | **0** |

D matches C's gain exactly while making **zero** deepen requests — it never asks
for a depth the promoted schedule would not have offered. That is why D was
chosen over C: identical measured benefit, strictly smaller behavioural surface.

---

## 4. Timed 8-prompt result

Two arms built from the same tree with **no patching**: `BASE` is
`0d2eef9c…:Qwen36MTPBlockSession.swift`, `PRICE` is `HEAD:…`. Counterbalanced
ABBA within one session. Unit is the MTP leg's
`decode_seconds / decode_token_count`, which `assert_scored_unit()` pins to the
trusted `score.json` per-token metric on every run.

### 4.1 Headline

**`e25/mtp_true_decode_gain_pct_median_of_8` = +3.835 %** (baseline `0.0`,
maximize). Pooled `+3.834 %`, mean `+3.845 %`, min `+2.095 %`, max `+5.919 %`.
**Arm PRICE wins on all 8 of 8 prompts.** Correctness `all_pass = True`,
`failures = []` — every leg matched its golden and closed its row ledger.

`depth_ge_4_realised = 0`, exactly as pre-registered, sourced from the timed
runs' own `effective_draft_lengths` (not from a separate sweep).

| prompt | order | base s/tok | price s/tok | **gain** | pre-reg | realised/pred | serial spread |
|---|---|---|---|---|---|---|---|
| english | A→B | 0.0496610 | 0.0483411 | **+2.658 %** | 4.223 | 0.629 | 0.166 % |
| narrative | B→A | 0.0501758 | 0.0482309 | **+3.876 %** | 5.024 | 0.772 | 0.208 % |
| technical | A→B | 0.0468828 | 0.0456082 | **+2.719 %** | 3.899 | 0.697 | 0.064 % |
| dramatic | B→A | 0.0475128 | 0.0450053 | **+5.278 %** | 9.775 | 0.540 | 0.418 % |
| travel | A→B | 0.0498239 | 0.0479341 | **+3.793 %** | 4.352 | 0.872 | 0.068 % |
| philosophy | B→A | 0.0485857 | 0.0464367 | **+4.423 %** | 5.768 | 0.767 | 0.029 % |
| natural_history | A→B | 0.0520129 | 0.0509232 | **+2.095 %** | 3.050 | 0.687 | 0.029 % |
| medicine | B→A | 0.0497588 | 0.0468133 | **+5.919 %** | 6.940 | 0.853 | 0.429 % |

`order` is the pre-registered ABBA schedule (`A` = BASE, `B` = PRICE), derived
from the index of the prompt in the canonical 8-prompt list rather than from
argument order, so it cannot drift with how the runs were batched. `serial
spread` is the run-to-run spread of the *pinned serial* leg for that prompt; it
is the noise floor of this rig and is 6–130× smaller than the effect it sits
next to (`clears = True`, max spread `0.429 %` vs min effect `2.095 %`).

Realised depth histogram `{1: 192, 2: 1264, 3: 515}` against the pre-registered
`{1: 193, 2: 1419, 3: 335}`: the arm shifts mass from d2 to d3 more than the
tape predicted, because on the live fixed window it also has to fill the extra
rounds discussed in §5.1.

### 4.2 Per-arm counters

| prompt | arm | rounds | declared rows | accepted | rejected | accept rate | mean depth | max depth | non-drafting | true decode s | replayed | p50 block s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english | BASE | 251 | 818 | 261 | 306 | 0.4603 | 2.2590 | 4 | 0 | 25.4264 | 102 | 0.08080 |
| english | PRICE | 255 | 778 | 258 | 265 | 0.4933 | 2.0510 | 3 | 0 | 24.7507 | 86 | 0.08027 |
| narrative | BASE | 250 | 836 | 263 | 323 | 0.4488 | 2.3440 | 4 | 0 | 25.6900 | 108 | 0.08087 |
| narrative | PRICE | 251 | 794 | 262 | 281 | 0.4825 | 2.1633 | 3 | 0 | 24.6942 | 99 | 0.08066 |
| technical | BASE | 231 | 790 | 281 | 278 | 0.5027 | 2.4199 | 5 | 0 | 24.0040 | 96 | 0.08127 |
| technical | PRICE | 235 | 757 | 278 | 244 | 0.5326 | 2.2213 | 3 | 0 | 23.3514 | 87 | 0.08026 |
| dramatic | BASE | 217 | 811 | 295 | 299 | 0.4966 | 2.7373 | 5 | 0 | 24.3266 | 94 | 0.08710 |
| dramatic | PRICE | 226 | 760 | 286 | 248 | 0.5356 | 2.3628 | 3 | 0 | 23.0427 | 85 | 0.08133 |
| travel | BASE | 251 | 828 | 262 | 315 | 0.4541 | 2.2988 | 5 | 0 | 25.5098 | 105 | 0.08106 |
| travel | PRICE | 250 | 792 | 263 | 279 | 0.4852 | 2.1680 | 3 | 0 | 24.5423 | 93 | 0.07978 |
| philosophy | BASE | 236 | 821 | 276 | 309 | 0.4718 | 2.4788 | 4 | 0 | 24.8759 | 105 | 0.08140 |
| philosophy | PRICE | 238 | 777 | 274 | 265 | 0.5083 | 2.2647 | 3 | 0 | 23.7756 | 92 | 0.08091 |
| natural_history | BASE | 275 | 841 | 237 | 329 | 0.4187 | 2.0582 | 4 | 0 | 26.6306 | 112 | 0.08005 |
| natural_history | PRICE | 276 | 805 | 236 | 293 | 0.4461 | 1.9167 | 3 | 0 | 26.0727 | 94 | 0.07963 |
| medicine | BASE | 236 | 847 | 276 | 335 | 0.4517 | 2.5890 | 5 | 0 | 25.4765 | 114 | 0.08878 |
| medicine | PRICE | 240 | 773 | 272 | 261 | 0.5103 | 2.2208 | 3 | 0 | 23.9684 | 96 | 0.08077 |

`true decode s` is the 512-token leg total; divide by 512 for the s/tok column
in §4.1. The counters say the mechanism did exactly what it was built to do and
nothing else:

- **`max depth` is exactly 3 on every PRICE leg**, and BASE reaches 4 or 5 on
  every prompt. The d3 coefficient `0.442` is the binding price; d4/d5 never
  clear it. This is the arm's signature, and it is present 8/8.
- **Accept rate rises on every prompt** (e.g. medicine `0.4517 → 0.5103`,
  dramatic `0.4966 → 0.5356`). Rejected rows fall on all 8 while accepted rows
  fall by at most 9, so the arm is removing rows that were going to be thrown
  away.
- **`non_drafting_rounds = 0` in all 16 legs.** Arm D declines *depth*, never
  drafting itself, so it never takes the zero-draft path. This matters for
  legality (§6): the round always proposes, verifies and reports.
- `replayed` rounds fall on all 8 prompts (e.g. medicine `114 → 96`), consistent
  with fewer rejected tails to repair.
- `p50 block seconds` falls on all 8, i.e. the win is per-round latency and not
  an artefact of the leg total.

---

## 5. Two self-corrections

### 5.1 Phase 0's projection is an upper bound

Phase 0 modelled arm D as **losing 98 emitted tokens**. That was wrong. The
trusted parent owns a fixed 512-token window and continues the trajectory, so a
shallower schedule does not emit fewer tokens — it spends **extra rounds**.

The measured fixed-window accounting over all 8 prompts:

| quantity | BASE | PRICE | delta | Phase 0 predicted |
|---|---|---|---|---|
| tokens emitted per leg | 512 | 512 | 0 | — |
| tokens lost | 0 | 0 | **0** | 98 |
| rounds | 1947 | 1971 | **+24** | — |
| declared rows | 6592 | 6236 | **−356** | −609 |

`tokens_emitted_all_legs_512 = True` on all 16 legs, so `realised_tokens_lost`
is **0** and Phase 0's 98 lost tokens simply do not exist on the live rig. The
cost reappears as `extra_rounds_spent_by_arm = +24` and as rows-saved coming in
at 356 rather than the predicted 609.

The row shortfall and the extra rounds are the same fact: each extra round
re-adds a depth-1 row plus a primary commit. `realised_rows_saved` 356 = 609
predicted − 229 rows the arm had to re-propose, and PRICE's proposed-row count
is `6236 − 1971 = 4265` against 4036 predicted, i.e. `+229`. So the 24 extra
rounds and the 229 re-proposed rows close against each other exactly.

**The BASE arm independently reproduced the E21 tape.** BASE ran 1947 rounds,
which is exactly the tape's 1947 rounds, and its 6592 declared rows decompose as
`1947 primary + 4645 proposed` — the tape's proposed-row count to the row. Phase
0's instrument gate showed the model *replays* the tape; this shows the live
binary *regenerates* it. The realised/predicted comparison below is therefore a
comparison of the arm against itself on the same trajectory, not against a
different decode.

I tested the extra-rounds explanation rather than asserting it. Per prompt:

| prompt | extra rounds | rows saved | realised/pred |
|---|---|---|---|
| travel | −1 | 36 | 0.872 |
| natural_history | +1 | 36 | 0.687 |
| narrative | +1 | 42 | 0.772 |
| philosophy | +2 | 44 | 0.767 |
| english | +4 | 40 | 0.629 |
| technical | +4 | 33 | 0.697 |
| medicine | +4 | 74 | 0.853 |
| dramatic | +9 | 51 | 0.540 |

`extra_rounds_vs_attenuation_pearson_r = −0.7405` (n = 8), with
`realised_over_predicted_gain_ratio_mean = 0.7270`. The extremes behave as the
mechanism requires: `travel` spent −1 rounds and kept 87 % of its predicted gain,
`dramatic` spent +9 and kept 54 %. The shortfall is therefore accounted for
rather than left as an unmodelled effect.

Honest note on that correlation: at n = 6 it was −0.9666 and adding the last two
prompts moved it to −0.7405, so extra rounds are the dominant term but not the
only one — `medicine` kept 85 % despite +4 rounds because it also saved by far
the most rows (74). I am reporting the weaker n = 8 figure rather than the n = 6
one I saw first.

**Phase 0's `+4.688 % median-of-8` must be read as an upper bound.** The
pre-registration was left unmodified and the timed matrix is the ground truth.

### 5.2 A prefill double-subtraction bug in my own reducer

`true_decode()` subtracted `seed_prefill_seconds` from `decode_seconds`, but
`decode_seconds` already excludes prefill. Verified empirically:
`decode_seconds / decode_token_count` reproduces the trusted
`score.json:metrics.{serial,mtp}_seconds_per_token` exactly. Fixed, and
`assert_scored_unit()` now pins both legs of every run to the trusted score file
so this class of unit error cannot recur silently. No number in this document is
affected — the fix landed before the timed matrix was reduced.

I also deleted a broken `probe_histograms()` path: `parse_trace` returns lists,
not depth-bearing objects. Realised histograms now come from the timed runs' own
`effective_draft_lengths`, so histogram and timing can never come from different
passes, and no extra traced sweep is needed.

---

## 6. Legality

The mechanism is a change to the **draft count chosen before proposal**.
`costModelDepth` runs in the depth-choice path (`Qwen36MTPBlockSession.swift:199`)
and returns a value in `[0, offeredDepth]`. `program.md` states the candidate
"may choose any draft count from zero up to the limit offered by the parent",
so this is the contract-blessed knob, not an edge of it.

Specifically, arm D:

- **declines no rows mid-verify.** The depth is fixed before any row is proposed,
  so there is no unevaluated-verify-row question (`AGENTS.md:231-233`). Every row
  that is proposed is fully evaluated and its exact top-two evidence reported.
- **does not change the target answer.** No head, weight, tokenizer, or target
  change; `uses_pinned_mtp_head=true` on all runs (`AGENTS.md:242-244`).
- **leaves no rejected state reachable.** Rollback and replay are untouched;
  `mtp_replayed_rounds` moves only because the depth distribution moved
  (`AGENTS.md:237-238`).
- **is input-independent.** `measuredRowStepRatio` is a compile-time constant
  table, in the same class as the input-independent weight/kernel/shape tables
  `AGENTS.md` explicitly permits. It does not read the prompt, phase, reference
  leg, or baseline.
- **cannot deepen.** The `max` with the shipped curve makes the price a pure
  truncation of the promoted schedule (0 deepen requests across 1947 taped
  rounds), so it cannot reach behaviour the promoted base has not already run.

What this does **not** license: the constants are fit on *this host's* prose
tape. They are a claim about relative per-row cost on this stack, not a universal
truth, and §8 below is the direct consequence.

---

## 7. Caveats

### 7.1 Binaries are not bit-reproducible on this host

Rebuilding the same two source blobs produced different worker digests
(`2c614e24…`/`85f6a9ce…` → `7cd4cf39…`/`83e700a7…`). Swift release builds are
not bit-reproducible here, so **arm identity rides on the source blobs, not the
binary digests**:

| arm | source blob (git SHA-1) | scored worker sha256 |
|---|---|---|
| BASE | `9f4610d5ee97f46e96f37dafd6c1238205283776` | `7cd4cf39cdb2d56a418f23a268289c12b86f45db…` |
| PRICE | `26c809044f82f556cef3baea6dbcffcfd8f51fca` | `83e700a7dab9d47dbaceddfb33ab0023a5e6fac2…` |

`research/e25-build.sh` enforces distinct worker+source digests across arms, an
**identical** trusted-driver digest (`c9bfcaf9…` both arms), and zero
`Qwen36MTPBlockSession` symbols in each installed driver — so the trusted driver
is provably invariant and the change is provably confined to the scored worker.

### 7.2 Cool-gate bypass

Advisor-authorised in PR #29 §8, conditional on ABBA-in-one-session. All four
flags are appended verbatim to every `meta.txt`
(`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`cool_gate_temp_c=40`, `cool_gate_bypass_reason=host idles above the compile-time
40C gate`). Entry/exit temperatures are recorded per arm, and the serial-control
spread is reported next to every effect.

### 7.3 ⚠️ Ranked transfer: I expect this to lose on the hidden pool

**This is the most important caveat in the document and it argues against
submitting.**

Source comments in the file I edited record **ranked** evidence pointing the
opposite way:

| ranked change | score | vs bar |
|---|---|---|
| uniform `h = 0.32` (`fc62d1aa`) | 2.84585 | −3 % |
| uniform `h = 0.15` | 2.667 | loses |
| uniform `h = 0.14` | 2.766 | loses |

Live accepted bar: **3.14642585386152**.

The `h = 0.32` note records ranked per-prompt mean drafts falling
4.35/4.89/5.78/5.33/5.04 → 3.36/4.01/4.53/4.03/4.76 and concludes **"this pool
rewards depth"**.

The problem is a distribution mismatch, and it is severe:

- The **ranked pool decodes at mean depth ≈ 4.3–5.8.**
- My **tape has n = 167 at d4, n = 9 at d5, and nothing above 5.** Local prose
  means are ≈ 2.4.
- Arm D's d3 coefficient is **0.442** against the shipped **0.153** — so at
  exactly the depths the hidden pool lives at, **arm D is more aggressive than
  `h = 0.32`, which already lost by 3 %.**

So a clean 8/8 local win at depth ≈ 2.2 is weak evidence about a pool at depth
≈ 5. The curve is measured where the ranked pool does not operate, and
extrapolated into the region where it does using n = 9.

**Recommendation: do not submit to Yukon without a ranked-representative
check.** The local result is real and the mechanism is sound; the open question
is entirely whether the price curve holds at d ≥ 4, and this experiment cannot
answer that.

---

## 8. Suggested follow-ups (not implemented)

1. **Re-measure the price curve at ranked depths.** The single highest-value
   next step. Force depth 4–6 on a prose tape to get real n at d4/d5/d6, then
   refit. If T(5)→T(6) is cheap, arm D's aggression at d3 is simply wrong for the
   ranked pool and a two-piece curve with a *falling* tail would beat it.
2. **A depth-conditioned price rather than a global one.** The ranked pool and the
   local prose pool appear to sit on genuinely different cost curves. A price
   parameterised by observed accept-rate regime would let one schedule serve
   both, and E25's `rowPriceCoefficient` is the natural seam for it.
3. **Prompt-mix diagnosis.** Worth understanding *why* the ranked pool sustains
   depth ≈ 5 when local prose sustains ≈ 2.4 before fitting any more curves;
   every price experiment is currently being fit on the wrong distribution.

## 9. Blocked on

- **Ranked-representative validation** is blocked on access to a prompt set that
  decodes at mean depth ≈ 5. Every local prose fixture I have decodes at ≈ 2.4,
  which is precisely why §7.3 cannot be settled here.
- **Bit-reproducible arm attribution** is blocked on the Swift toolchain's
  non-determinism on this host (§7.1). Mitigated via source-blob identity, not
  solved.

## 10. Out of scope, noted

Under arm D's price, `segmentedVerifyDepthCap = 8` and `segmentedStreakGate = 2`
become dead code — the price never reaches a depth where either can fire. Both
were left in place as out of scope for this assignment.

---

# Part II — revision r2

- Assignment: `qwen38-r1-e25-per-row-draft-price` (PR #29, revision **`r2`**)
- `BASE_SHA`: **`d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602`** (was `0d2eef9…` in r1)
- Promoted frontier at the time of writing: source `474c75013f333f119bdc465d849f23917b195b20`,
  submission `942e5ab2-1c46-4c50-b7c3-eaf948878ed0`, **score `3.2341518328631`**
- Host: **Apple M4 Pro / `Mac16,11` / `applegpu_g16s`**, single GPU. **Not the ranked M5.**
- W&B group: `qwen38-r1-e25-per-row-draft-price`
- Research-only additions: `research/e25r2_rule.py`, `research/e25r2_refit.py`,
  `research/e25r2_policy.py`, `research/e25r2_log_wandb.py`,
  `research/e25r2-force-depth.sh`, `research/e25r2-pool.json`,
  `research/e25r2-policy.json`. **No candidate source file changed in r2.**

## 11. What r2 asked and what it found

### 11.1 Headline

The advisor's objection is **correct, and stronger than stated**. Arm D is not an
aggressive price; it is a **hard `DEEP_CAP = 3`** that happens to be spelled as a
price. §12 proves this three independent ways: analytically, by numerical
identity against an explicit `DEEP_CAP = 3` arm, and by the depth histogram of
the *actually shipped* arm-D binary at runtime.

Worse for arm D: on the re-measured curve, **the entire price-vector design
space is worth at most a few tenths of a percent**, roughly an order of magnitude
less than arm D's r1 measured `+3.83 %` (§15). r1's headline is therefore not a
price effect that a better price can extend — it is very largely the `DEEP_CAP`,
and a cap is exactly the lever the in-source ranked history says **loses**
(§19.1). **I do not recommend arm D, and I do not recommend a Swift successor to
it.** E25 closes as a *negative result with a proof*, and the mechanism it
disproves is worth more than the `+3.83 %` it originally reported.

### 11.2 r1 numbers that r2 retires

| r1 claim | status in r2 | why |
|---|---|---|
| `+3.83 %` median gain is a *price* result | **retired** | it is dominated by the implied `DEEP_CAP = 3` (§12, §15) |
| `measuredRowStepRatio = [0, 0.0959, 0.1523, 0.4424]` is the right curve | **re-measured** | new base changed both `T(d)` and acceptance `p` (§11.3) |
| depth 7 is the pre-registered rate optimum (assumes `p = 1`) | **refuted by measurement** | §13, §19.4 |
| the in-source h-sweep "supports" arm D | **retired as a misreading** | §19.1 |
| ranked bar `3.13098700135133` / `d1530a40` / `bd007bc7-…` | **stale** | live bar is `3.2341518328631` (§17) |
| declared head was attached in r1 | **false — r1 used a truncated head** | §19.2 |
| `uses_pinned_mtp_head: true` means "the organizer head" | **false** | §19.3 |

### 11.3 Why every r1 timing number had to be re-measured

The frontier delta `0d2eef9` → `d7619a7` touches all three terms of the price
model, so **no r1 timing transfers, including ratio-normalised ones**:

1. **`T(d)` proposal term.** A new 2-bit single-row draft kernel
   `qmv_fast_singlerow_affine2_g64` (`kernels/quantized.h:1084`, gate
   `bits == 2 && out_vec_size == 98336 && ntg.x == 1`) now serves the draft
   readout. It fires on **every** head step, so it rescales the per-row head
   cost `H` directly.
2. **Acceptance `p`.** `draftTokenIDWithDeclaredRerank` (`Qwen35.swift:3142`)
   replaced the plain argmax readout with an exact top-32 shortlist
   (`qwen35DraftTop32` @2657) plus a declared rerank kernel (@2380). This is
   proposal-side only, so it is legal, but it changes *which* token is proposed
   and therefore changes measured acceptance.
3. **`T(d)` verify term.** A fused residual RMSNorm (kernel @1480, wrapper
   @1582, `boundaryFused` @2075, switch @2175, gated on
   `bfloat16 && dim(-1) == 5120`) removes 63 launches per forward.

The only *width*-dependent change in the delta is `case 8` (depth 7) moving
`IPG` 3 → 4, which matters only at the deepest forced leg.

Recomposition onto `d7619a7` was clean: `git merge-tree` returned rc 0 and a
clean tree, and neither `measuredRowStepRatio` nor `rowPriceCoefficient` exists
on the new base, so the r1 patch applies as a pure addition.

## 12. Deliverable (a): arm D is `DEEP_CAP = 3`, proved three ways

### 12.1 The admissibility theorem

The shipped walk (`Qwen36MTPBlockSession.swift`, `costModelDepth`) is:

```text
reach = 1 ; expected = 0
for depth in 0 ..< cap:
    p        = p(depth)                       # ema / sigmoid-clamped at d0,d1
    reach   *= p
    threshold = c(depth) * (1 + expected)
    guard reach > threshold else break
    expected += reach
```

So row `depth + 1` is added **iff** `reach_depth > c_depth · (1 + expected_depth)`.
Because `reach ≤ 1`, the loosest possible requirement is obtained at the most
favourable acceptance history, `p = 1` at every earlier depth, which gives
`reach = 1` and `expected = depth`. Hence

```text
row depth+1 is reachable for SOME acceptance history
  ⟺  1 > c(depth) · (1 + depth)
  ⟺  c(depth) < 1 / (depth + 1)
```

This is a *hard* feasibility statement, not a probabilistic one: if
`c(depth) ≥ 1/(depth+1)` then no acceptance history whatsoever reaches depth
`depth + 1`. Ceilings:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `1/(depth+1)` | 1.0 | 0.5 | 0.3333 | **0.25** | 0.2 | 0.1667 | 0.1429 | 0.125 |

Arm D sets `c(3) = 0.442442`. Since `0.442442 > 0.25`, **depth 4 is
unconditionally unreachable under arm D**, and therefore so is every deeper row.
Arm D's effective maximum draft depth is exactly **3**.

Two useful corollaries:

- The **shipped scalar** form `c(d) = h/(1 + d·h)` satisfies
  `h/(1+d·h) < 1/(d+1) ⟺ h < 1`, so the shipped price is **wall-free at every
  depth for every legal `h`**. Walls are a thing arm D introduced, not a thing
  the schedule had.
- Arm D's `max(h/(1+d·h), measured_c(d))` differs from BASE **only at `d ≥ 2`**,
  because `measured_c(0) = 0 < 0.18` and `measured_c(1) < c_shipped(1)`.

`research/e25r2_rule.py` checks this numerically: 800,000 random acceptance
histories drawn over the legal `ema` simplex fire depth ≥ 4 under arm D
**zero** times.

### 12.2 Numerical identity against an explicit `DEEP_CAP = 3`

In the offline replay engine (§14), I ran arm D beside an arm that keeps the
shipped scalar `h = 0.18` and simply clamps `cap` at 3. The two arms produce
**identical** depth histograms, round counts, emitted-token counts, and modelled
times on every prompt. An arm that is numerically indistinguishable from
`DEEP_CAP = 3` *is* `DEEP_CAP = 3`.

The same table shows the converse: `DEEP_CAP = 4` and `DEEP_CAP = 5` are both
numerically identical to unmodified BASE, i.e. the shipped price already almost
never wants depth ≥ 4 on this pool. That is what makes the cap the whole story.

### 12.3 The runtime depth histogram of the real arm-D binary

The two arguments above are offline. The third is the shipped `PRICE` binary
itself, traced end-to-end on the `english` fixture with the declared head
(`research/e25r2_policy.py --observed-arm PRICE`, report key
`observed_runtime_depths`):

```text
rounds = 239   depth_histogram = {1: 30, 2: 169, 3: 40}
max_depth_observed = 3        rounds_at_depth_ge_4 = 0
width_cap_histogram = {5: 222, 8: 17}   mean_depth = 2.0418
tokens_per_round = 1.9874
```

`rounds_at_depth_ge_4 = 0` with 17 rounds where the width cap was 8 and thus
*would have permitted* depth 4–8. The cap that bound those rounds was arm D's
price, not the hardware wall.

## 13. Deliverables (b) and (c): the price curve re-measured, with real `n` at depth 4–7

Artifact: `research/e25r2-pool.json` (38,895 B). Reproduce with

```bash
python3 research/e25r2_refit.py \
  --prompts english,narrative,technical,dramatic,travel,philosophy,natural_history,medicine \
  --out research/e25r2-pool.json
```

The forced-depth probe binary FORCE round-robins the taken depth over
`[0,1,2,3,4,5,6,7]` and removes the streak gate, so every depth gets real,
comparable `n` on every prompt instead of only the depths the shipped rule
happens to visit. All eight prompts contributed: 1954 rounds traced, **1826
analysed** after dropping the first 8 and last 8 rounds of each prompt
(8 warmup + 8 tail × 8 prompts = 128).

### 13.1 `T(d)` on the trusted-parent clock

| d | n | mean ms | sem | median ms | weight-stream passes |
|---|---|---|---|---|---|
| 0 | 232 | 68.616 | 0.141 | 68.722 | 1 |
| 1 | 232 | 71.605 | 0.122 | 71.446 | 1 |
| 2 | 231 | 78.916 | 0.145 | 78.692 | 1 |
| 3 | 228 | 91.885 | 0.156 | 91.995 | 1 |
| 4 | 228 | 132.257 | 0.140 | 133.350 | **2** |
| 5 | 226 | 144.103 | 0.190 | 144.817 | 2 |
| 6 | 225 | 156.788 | 0.136 | 157.515 | 2 |
| 7 | 224 | 170.133 | 0.138 | 170.827 | 2 |

The step from `d = 3` to `d = 4` costs **40.4 ms**, against 13.0 ms for the step
before it and 11.8 ms for the step after it. That single discontinuity is the
whole story of this experiment, and §12 of the r1 report already identified its
mechanism: at `d = 4` the verify block is `M = 5` rows, which is the first width
that needs a **second** pass over the 4-bit backbone weights
(`ceil(M / IPG)`, `_m<T,5,3,true>`).

### 13.2 The step ratios, and the one that is inadmissible

`c_d = (T(d+1) − T(d)) / T(d)` is the quantity arm D calls a price.

| d | measured `c_d` | admissibility ceiling `1/(d+1)` | admissible? | r1 `c_d` | shipped scalar |
|---|---|---|---|---|---|
| 0 | 0.043568 | 1.0 | yes | — | 0.18 |
| 1 | 0.102093 | 0.5 | yes | 0.095904 | 0.15254 |
| 2 | 0.164342 | 0.3333 | yes | 0.152261 | 0.13235 |
| 3 | **0.439371** | **0.25** | **NO** | **0.442442** | 0.11688 |
| 4 | 0.089570 | 0.2 | yes | — | — |
| 5 | 0.088027 | 0.16667 | yes | — | — |
| 6 | 0.085119 | 0.142857 | yes | — | — |

Two things matter here.

First, **r1's `c_3` replicates on the new base**: 0.439371 against r1's 0.442442,
a 0.7 % difference, even though the base changed the draft kernel to 2-bit,
changed acceptance with top-32 + rerank, and fused residual RMSNorm. The price
curve was *not* stale in its decisive coefficient. Deliverable (b) is answered,
and the answer is that re-measuring did not rescue arm D.

Second, `c_3` is the **only** violation of the §12 admissibility ceiling, and it
is a 76 % overshoot rather than a marginal one. This is why arm D behaves as a
hard cap: a price of 0.4394 at depth 3 cannot be paid by any acceptance belief,
because even a *certain* accept of all four pending rows only returns
`1/(3+1) = 0.25` of a round.

### 13.3 The realised rate: constant-depth argmax is 2, not 7

Dividing measured `T(d)` by measured tokens-per-round gives the rate a
*constant-depth* schedule would achieve.

| d | tokens/round | ms/token | vs best |
|---|---|---|---|
| 0 | 1.0000 | 68.616 | −44.77 % |
| 1 | 1.7284 | 41.428 | −8.52 % |
| **2** | **2.0823** | **37.899** | **best** |
| 3 | 2.3421 | 39.232 | −3.40 % |
| 4 | 2.4079 | 54.926 | −27.13 % |
| 5 | 2.2743 | 63.360 | −38.19 % |
| 6 | 2.5644 | 61.139 | −34.08 % |
| 7 | 2.3661 | 71.905 | −41.51 % |

`argmax = 2` on both the mean and the median `T`. A 5,000-draw bootstrap over
rounds puts the argmax share at **{1: 0.0005, 2: 0.80075, 3: 0.19875}** and
never selects 4 or deeper. Depth 2 and depth 3 are within 3.4 % of each other,
which is why the shipped adaptive rule oscillating between them is reasonable;
depth 4 and beyond are not close.

Per-prompt argmax: `dramatic 2, english 2, medicine 3, narrative 3,
natural_history 1, philosophy 2, technical 2, travel 2`. No prompt wants depth
4 or deeper.

**This is the answer to deliverable (c).** The forced legs at `d = 4, 5, 6`
(and 7) that r2 asked for do exist now with real `n ≈ 225` each, and they show
that the deep rows are not merely mispriced — they are genuinely bad. Depth 4
buys 15.6 % more tokens per round than depth 2 for 67.6 % more time.

### 13.4 The cost model, and what it says the price *should* be

Fitting `T = a + b·d + c·ceil((d+1)/IPG)` over all eight depths:

```text
T(d) = 30.120 + 10.172·d + 32.378·ceil((d+1)/IPG)   R² = 0.99271, max|resid| 6.118 ms
```

`b = 10.172 ms` is the per-draft-row marginal cost, which is the proposal head
running once per draft token. `c = 32.378 ms` is the cost of one extra pass over
the backbone weights. The model explains 99.3 % of the variance in `T(d)` from
those two terms alone, which is strong independent support for the mechanism in
§12.3 and means the `d = 3 → 4` step is **structural**, not noise or thermal
drift.

### 13.5 Confounder controls carried forward from r1

* **Position / thermal drift.** OLS of round time on token position gives
  **+3.0151 ms per 1000 tokens** (n = 1826) — i.e. about +1.5 ms across a
  512-token leg, an order of magnitude below the 40.4 ms step being measured.
  Splitting the pool into position quartiles reproduces the depth ordering
  **perfectly in all four**:

  | quartile | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
  |---|---|---|---|---|---|---|---|---|
  | q0 | 68.6 | 71.3 | 78.9 | 92.0 | 132.5 | 144.0 | 156.8 | 170.1 |
  | q1 | 68.8 | 72.0 | 79.5 | 92.2 | 132.6 | 144.4 | 157.2 | 170.4 |
  | q2 | 68.3 | 71.5 | 78.3 | 91.7 | 131.9 | 143.6 | 156.4 | 169.9 |
  | q3 | 68.6 | 71.6 | 78.9 | 91.6 | 132.0 | 144.4 | 156.7 | 170.2 |

  Because FORCE round-robins depth, depth is orthogonal to position by
  construction; these quartiles confirm it rather than repair it.
* **Instrument agreement.** `core_minus_parent` is −0.97 to −1.03 ms and the
  trace-I/O `upkeep_bias` is 0.245–0.269 ms, i.e. ≤ 0.21 % of round time. Real
  in mechanism, immaterial in size, and common-mode across depths.
* **Fidelity.** All 8 prompts: `all_tokens_matched = true`, `parity_all_ok =
  true`, `residual_divergence_count = 0`, `decode_token_count = 512`, declared
  head `dadbfb806d80` / 427,746,170 B.
* **Row accounting.** All 8 prompts: `accepted_agrees`, `proposed_agrees`,
  `rows_closed` all true, `target_tail_total = 0`,
  `max_rejected_tail_logit_delta = 0`.

### 13.6 Conditional acceptance on the pooled tape

| position | p | n |
|---|---|---|
| 0 | 0.6926 ± 0.0116 | 1594 |
| 1 | 0.5840 ± 0.0161 | 935 |
| 2 | 0.5077 ± 0.0234 | 457 |
| 3 | 0.4190 ± 0.0369 | 179 |
| 4 | 0.3860 ± 0.0645 | 57 |
| 5 | 0.6875 ± 0.1159 | 16 |
| 6 | 0.4000 ± 0.2191 | 5 |

Acceptance decays smoothly through position 4 and then goes uninformative on
`n ≤ 16`. There is no acceptance cliff at position 3 — **the cliff is entirely
on the cost side**, which is the single most important structural fact in this
report.

## 14. The instrument, and a defect in it that I found and fixed

### 14.1 The trusted-parent per-round clock

r1 measured `T(d)` from the editable side's own `round_us` trace field. r2
measures it on **the clock that produces the score**: the trusted parent writes
`block_request_seconds` and `effective_draft_lengths` into
`reports/04-mtp-timed.json`, and pairing them index-wise gives the per-round
wall time and the depth actually taken, attributed by the parent's journal.

This is strictly better in three ways: it needs nothing from the editable side,
it cannot be biased by trace I/O, and — unlike the trace, which emits no line for
a depth-0 round — **it covers `d = 0`**, which is the serial control the whole
price model is normalised against. `timer_agreement_ms` in the pooled report
cross-checks the editable timers against it; the residual `upkeep_us` trace-I/O
bias is real in mechanism but ≤ 0.21 % of round time, so it changes no ordering.

### 14.2 The replay engine and its acid test

Every field the depth walk consumes is in the trace: `m=` (the margin feeding
the `d0`/`d1` sigmoid clamps), `ema=` (all eight EMAs), and `cap=`. Because the
walk is a pure function of those, **any** price vector can be replayed on
**every** recorded round with zero unevaluable rounds. The acid test is to
replay the *shipped* price and require that the reconstructed depth equal the
recorded `shipped=` mark on every round.

### 14.3 The defect: offer-bound tail rounds

That acid test failed, at 800 of 801 rounds. The one failure was round 234 of
235 on `dramatic`.

Root cause: the trace's `cap=` field is written by
`snapshotScheduleSignal(widthCap:)` and therefore records **`widthCap` only**,
not the real `cap = min(offeredDepth, maxDepth, widthCap)`. Near the end of the
fixed 512-token window the trusted parent offers fewer rows than eight, so those
rounds are **offer-bound**, and the replay — which only knows `widthCap` —
over-predicts their depth. Nothing is wrong with the candidate; the *instrument*
was blind to one term.

Fix: exclude the tail of each prompt, symmetric with the existing warm-up
exclusion. The bound is exact rather than heuristic: `offeredDepth < 8` requires
at most 8 tokens remaining in the window, and every round emits at least one
token, so **at most the last 8 rounds of a prompt can be offer-bound**. Setting
`TAIL_ROUNDS = 8` (`= MAX_DEPTH`) in both `research/e25r2_refit.py` and
`research/e25r2_policy.py` restores **`replay_exact = True`**.

I record this because it is the kind of defect that silently biases a depth
study *towards shallow*: offer-bound rounds look like rounds where the schedule
chose to stop early, and they are concentrated exactly where the window ends.
r1's reducer had no tail exclusion at all.

## 15. Offline policy replay: the whole price design space

Artifact: `research/e25r2-policy.json`. Reproduce with

```bash
python3 research/e25r2_policy.py --refit research/e25r2-pool.json \
  --prompts english,narrative,technical,dramatic,travel,philosophy,natural_history,medicine \
  --out research/e25r2-policy.json
```

The forced-depth tape records, for every round, the acceptance outcome at every
position up to the forced depth. That makes the tape **replayable**: for any
candidate price vector I can recompute the depth the greedy rule would have
chosen and score it against the measured `T(d)` table, without spending a GPU
run per arm. 1578 rounds are evaluable this way, and the engine reproduces the
shipped arm exactly (`replay_exact = 1578/1578`, §14.2).

### 15.1 Every arm in the design space

| arm | ms/token | mean depth | vs base | depth histogram |
|---|---|---|---|---|
| `arm_d_r1_measured` | **39.8026** | 1.994 | **+0.322 %** | {1: 107, 2: 1373, 3: 98} |
| `arm_d_refit_measured` | 39.8407 | 2.092 | +0.226 % | {1: 107, 2: 1219, 3: 252} |
| `base_shipped_deep_cap_3` | 39.8407 | 2.092 | +0.226 % | {1: 107, 2: 1219, 3: 252} |
| `coordinate_optimum` | 39.8458 | 1.981 | +0.213 % | {1: 31, 2: 1546, 3: 1} |
| `base_shipped_h0.18` | 39.9308 | 2.099 | 0.000 % | {1: 107, 2: 1219, 3: 241, 4: 11} |
| `base_shipped_deep_cap_4` | 39.9308 | 2.099 | 0.000 % | identical to base |
| `base_shipped_deep_cap_5` | 39.9308 | 2.099 | 0.000 % | identical to base |
| `free_deep_rows` | 79.2277 | 7.525 | −49.600 % | {1: 107, 8: 1471} |

Read this table from the bottom up, because the controls carry most of the
information.

* **`free_deep_rows`** prices every deep row at zero, so the greedy rule always
  runs to depth 8. It loses **49.6 %**. The deep rows are not a pricing artefact
  that a better coefficient could unlock; they are expensive, and any rule that
  buys them loses badly.
* **`deep_cap_4` and `deep_cap_5` reproduce the base exactly** — same ms/token
  to four decimals, same histogram. The shipped rule at `h = 0.18` already
  almost never goes past depth 4 (11 rounds out of 1578, 0.7 %). So capping at 4
  or 5 is a no-op: there is nothing there to cut.
* **`deep_cap_3` gains +0.226 %** by removing exactly those 11 depth-4 rounds.
  That is the *entire* mechanism available to a cap.

### 15.2 The numerical identity that settles deliverable (a)

`arm_d_refit_measured` and `base_shipped_deep_cap_3` are **the same row**:
39.8407 ms/token, mean depth 2.092, and the byte-identical histogram
{1: 107, 2: 1219, 3: 252}.

That is not a coincidence or a rounding coincidence — it is the operational
definition of equivalence. Feeding the refit measured price vector into the
greedy rule produces, round for round across 1578 rounds, the same decisions as
deleting arm D entirely and writing `DEEP_CAP = 3` above the shipped `h = 0.18`
rule. **The advisor's objection is confirmed.** Arm D is not a price that
happens to be steep; it is a hard cap wearing a price's clothing, and §12's
proof says why: `c_3 = 0.4394 > 1/4` makes depth 4 unreachable for *every*
acceptance belief, so the coefficient's numeric value is irrelevant beyond the
fact that it exceeds the ceiling.

### 15.3 The best arm found anywhere in this replay is +0.322 %

Coordinate ascent over the eight coefficients converges to
`[0.18, 0.12, 0.22, 0.1169, 0.1047, 0.0947, 0.0865, 0.0796]` at +0.213 %, which
is **worse** than `arm_d_r1_measured` at +0.322 %.

I want to be precise about what that does and does not establish. Coordinate
ascent is a **local** search, and the fact that it is beaten by a
hand-constructed arm proves the objective is not concave in these coordinates
and that the search did not find the global optimum. So the honest claim is:

> The best arm found anywhere in this replay is **+0.322 %**.

not "+0.322 % is the provable ceiling of the price design space." I did not
prove a ceiling. What I can say is that four independent probes of the space —
the r1 measured vector, the refit vector, a family of explicit depth caps, and a
coordinate search — all land between +0.21 % and +0.33 %, and the one arm that
reaches outside that band (`free_deep_rows`) loses 49.6 %. The design space
looks flat and small, and nothing in it is worth a submission.

### 15.4 Belief calibration

The replay needs the rule's acceptance beliefs to match reality, or its depth
choices would be fiction. Predicted 2.153 tokens/round against actual 2.2548, a
ratio of **0.9549**; I apply the reciprocal `shrink = 1.0472` so the replayed
rule is neither systematically optimistic nor pessimistic. Per-forced-depth
ratios are 0.9336 / 0.9349 / 0.9082 / 0.9486 / 1.0251 / 0.9145 / 1.0178 for
`d = 1..7` — flat, with no depth-dependent bias that would distort the deep-row
decisions specifically.

### 15.5 The shipped PRICE binary agrees at runtime

The replay is a model, so it needs a runtime check. Tracing the **real** PRICE
binary on `english` gives `depth_histogram = {1: 30, 2: 169, 3: 40}`,
`max_depth_observed = 3`, and `rounds_at_depth_ge_4 = 0` over 239 rounds, with
`mean_depth = 2.0418`. Meanwhile `width_cap_histogram = {5: 222, 8: 17}` shows
the hardware and streak gate **would have permitted** depth 4–8 in every one of
those rounds. Arm D's own price is what held it at 3. The replay and the binary
agree.

## 16. Deliverable (d): why a "measured but non-prohibitive" `d ≥ 3` price cannot exist

The advisor asked for a variant whose `d ≥ 3` prices are measured rather than
prohibitive. **That variant does not exist**, and the reason is a theorem plus a
measurement, not a failure of search:

- §12.1: row 4 is reachable **iff** `c(3) < 0.25`.
- §13: the *measured* `c(3)` on the new base is ≈ `0.44`, because `T(3) → T(4)`
  crosses the M = 5 weight-stream pass cliff (`ceil(M/IPG)` goes 1 → 2 at
  `M = 5`, i.e. depth 4). The cost really does jump ~44 % there.

So "measured" and "non-prohibitive" are contradictory requirements at `d = 3` on
this hardware. Any variant satisfying the letter of (d) must **stop using the
measured value at `d = 3`** — at which point it is no longer a measured price,
it is a hand-chosen one, and §15 shows hand-chosen prices are worth at most a
few tenths of a percent anyway.

The wall is a property of the hardware, not of arm D's `max(...)`. Arm D's only
sin is being honest about it in a rule that cannot survive honesty.

**This is why I did not write a Swift arm G.** Adding a fourth `rowPrice*`
variant to the candidate would spend a build and a timed allocation to
re-discover a proof. §15's replay covers the same design space exhaustively and
for free, on the parent's own clock.

The lever that *would* pay is collapsing the cliff itself — making depth 4 cost
one weight pass instead of two. The advisor assigned that to **thorfinn as E27**,
and I deliberately did not touch it.

## 17. Deliverable (e): projections re-anchored on the live bar

r1 anchored on submission `bd007bc7-…` / source `d1530a40` / score
`3.13098700135133`. All three are stale. The live highest-scoring `promoted` row
is:

```text
submission 942e5ab2-1c46-4c50-b7c3-eaf948878ed0
source     474c75013f333f119bdc465d849f23917b195b20
score      3.2341518328631
```

There is no price-vector candidate I would submit against a `3.2341518328631`
bar: §15's modelled ceiling for the whole design space is a few tenths of a
percent even if local gain transferred one-for-one, and §19.1 gives a mechanism
by which arm D specifically should *lose* several percent on the ranked pool.

I did not submit, and could not have: the submission slot was busy for the whole
revision (receipt `9197ed62-621f-474d-bfba-e1efddd9dd4c` validating), so
`senpai/submit-official.sh` was correctly not run.

## 18. Deliverables (f) and (g): head choice

Every r2 leg — serial control, verify, and timed MTP, on all arms — ran the
**declared** head:

```text
MLXFAST_QWEN_MTP_HEAD_DIR=$ADH/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-q2q4-run
head_origin  hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae6282749a52e052496dd5300b4aa441df7301e8
model.safetensors  427,742,600 B  sha256 d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1
```

This is the right control rather than the organizer head, and the reason is
mechanical, exactly as the advisor's calibration fact (g) implies: the declared
head's `draft_lm_head.weight` is U32 `[98336, 320]`, i.e. **2-bit**, which is
precisely what the new `qmv_fast_singlerow_affine2_g64` gate keys on
(`bits == 2 && out_vec_size == 98336`). **Head choice and per-row draft cost are
therefore coupled** — measuring the price curve under the organizer's 4-bit head
would measure a draft step the ranked candidate never executes.

Two provenance notes worth keeping:

- The manifest `sha256` `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`
  is the **single-file tree digest**, not the digest of `model.safetensors`. They
  are different numbers for the same head; confusing them wastes an hour.
- `research/fetch-declared-head.sh` stages a sibling run tree with
  `model.safetensors` **hardlinked** plus the organizer head-family
  `config.json` (byte-identical, 3570 B). Only `benchmark-qwen-mtp.sh:215` needs
  that `config.json`; the ranked workflow fetches `model.safetensors` alone and
  `verifyHeadTree` (`Qwen36MTPHeadAttachment.swift:215-265`) never reads it. So
  the sibling tree is a local-harness convenience, not a deviation from the
  ranked head.

## 19. r2 self-corrections

### 19.1 I misread the in-source ranked h-sweep, and it argues *against* arm D

In r1 I cited the `headStepCostRatio` sweep recorded in
`Qwen36MTPBlockSession.swift` as support for arm D. That was wrong twice.

First, the cross-era error: those scores (newjordan `2.91995`, hadakang
`2.92976`, `4650c96e` `2.93524`, `fc62d1aa` `2.84585`) come from **different
frontier eras** and are not comparable to each other or to today's bar. Citing
them as a ladder is exactly the cross-commit comparison `program.md` forbids.

Second, and more importantly, the one *within-era* comparison in that history
points the other way. Raising `h` 0.18 → 0.32 (`fc62d1aa`) scored `2.84585`, a
clean **−3 %** against its own base, with the serial leg flat
(`0.038092 → 0.038070`) and candidate decode time **up 0.95 %**. Its mechanism
was purely that it **shortened every draft**: ranked mean draft lengths went
`4.35 / 4.89 / 5.78 / 5.33 / 5.04` → `3.36 / 4.01 / 4.53 / 4.03 / 4.76`. With
`h = 0.15` (`2.667`) and `h = 0.14` (`2.766`) also failing, `h` is bracketed on
both sides and **`0.18` is a true local optimum**. The in-source comment's own
conclusion is *"this pool rewards depth"*.

**Therefore the ranked-transfer prediction for arm D is negative.** Ranked mean
draft length at `h = 0.18` is 4.35–5.78 (≈ 5.08). Arm D caps at 3, so it
truncates the *majority* of ranked rounds — roughly a 40 % cut in mean depth,
against the 19 % cut that cost `h = 0.32` three percent. I expect arm D to
regress materially on the hidden pool. My r1 recommendation against submitting
it was right, but for a weaker reason than the real one.

The local fixture cannot see this, because local prose decodes at mean depth
≈ 2.0 (§12.3) where a cap at 3 is nearly free. **The local pool and the ranked
pool sit in different depth regimes**, and that gap — not the price curve — is
the binding uncertainty in this whole line of work.

### 19.2 r1 ran a truncated head and I did not notice

r1's `E11_HEAD_DIR` pointed at a staged tree of **238,934,129 B**, sha256
`c934b40f…`, against the declared head's 427,742,600 B. That is a real r1
defect: a partially-fetched head silently changes proposal quality, hence
acceptance, hence every price number r1 measured. It is one of the reasons §11.3
insists the curve be re-measured rather than transferred.

### 19.3 `uses_pinned_mtp_head: true` does not mean "the organizer head"

I spent time in r1 treating that report field as head provenance. It is not:
`main.swift:1962-1966` sets it from `report.usesNativeMTPHead`, i.e. **"did this
leg draft at all"**. Serial legs report `false` and MTP legs report `true`
regardless of which head is attached. Real provenance is
`head_provenance.origin` / `head_sha256` / `head_bytes`. Had I read the field
correctly in r1, §19.2 would have been caught before the timed runs.

### 19.4 My depth-7 pre-registration was wrong

r1 pre-registered depth 7 as the rate optimum, on a `p = 1` assumption. Measured
acceptance is nothing like 1 — conditional `p` falls from ≈ 0.70 at `d0` to
≈ 0.36–0.51 by `d3`–`d4` — and the measured realised-rate argmax is shallow
(§13). The pre-registration is **refuted**; `e25r2/r1_prereg_refuted` records
this in W&B rather than quietly dropping it.

## 20. r2 caveats

### 20.1 The FORCE arm measures *unconditional* acceptance

To get real `n` at every depth, the FORCE arm removes the streak gate (width cap
pinned at `segmentedVerifyDepthCap = 8`) and cycles forced depth 0..7. That makes
measured acceptance the **round-robin average over all rounds**, whereas the
shipped rule only drafts deep *after* a good full-accept streak, so its
conditional acceptance at `d ≥ 3` may exceed the cycle average.

This is the right unbiased estimator for "what would depth `d` cost and yield on
a random round", and it is exactly why the constant-depth argmax being shallow
does **not** license the conclusion that the adaptive rule should be shallower.
I flag it because it is the single easiest way to over-read §13.

### 20.2 Stale `mlx.metallib` is a known, non-masking condition

Every run log carries the metallib mismatch warning (recorded `6639cc59…` vs
current `12ad4a6a…`). I deliberately did **not** rebuild it. The Metal delta
`0d2eef9` → `d7619a7` is exactly `quantized.cpp` (the runtime-effective JIT
twin) plus `kernels/quantized.h`, so the affected kernels are JIT-compiled from
the C++ source string and are live regardless of the metallib;
`python3 research/twin_audit.py` reports `TWIN AUDIT OK: 29 runtime-effective
twin(s)`, exit 0. The condition is common-mode across all three arms, and the
ranked runner rebuilds via `./setup.sh` anyway.

### 20.3 `T(8)` is modelled, not measured

The forced cycle covers depths 0..7. Depth 8 (`M = 9`, the second pass cliff) is
reported from the pass-count model only, and is labelled as such wherever it
appears.

### 20.4 Binaries are still not bit-reproducible

As in r1 (§7.1), Swift release builds are not bit-identical on this host. Arm
identity therefore rides on **source-blob digests**, recorded per arm:

| arm | `Qwen36MTPBlockSession.swift` blob | notes |
|---|---|---|
| BASE | `7ce81abe55275f6b25712026ba3d5908396ebe30` | unmodified `d7619a7` |
| PRICE | `64673bab9c024d38639c291d71269be8ac0046ef` | arm D recomposed |
| FORCE | `7ce81abe…` + patch `0ff4ab29…` | instrument only, never a candidate |

### 20.5 Cool-gate bypass

`MLXFAST_LOCAL_COOL_GATE=0` was set for all r2 legs, with the four required
disclosures written verbatim into every run's `meta.txt`. Absolute
seconds-per-token across prompts should be read with that in mind; all
comparisons in §13 and §15 are within-run or same-session.


## 21. Matched BASE vs PRICE timing on base `d7619a7`, and what it retracts

### 21.1 What was measured

Sixteen timed legs: eight prompts x {BASE, PRICE}, ABBA-ordered within each
pair, all on base `d7619a7`, the declared `q2-q4-rerank-v1` head
(`d038fd41...`, 427,746,170 B), 512 seed + 512 decode tokens per leg, on the
local Apple M4 Pro (`Mac16,11`, `applegpu_g16s`) -- **not** the ranked M5.

Report: `research/e25r2-timed.json`, produced by `research/e25r2_timed.py`.
It **exits 0 with `gates.all_pass = true`, zero gate failures and no missing
legs**. The gates check row accounting, ledger closure, rejected-tail logit
equality, emitted-token counts, parent-clock agreement, arm identity and
`meta dirty == 0` on every leg.

Per-leg `mlxfast-runtime-worker` sha256 was identical across every leg of a
given arm (BASE `fbbf7cfc...`, PRICE `e5eb0e09...`), so arm identity is
established by binary digest, not by trust.

### 21.2 Headline

`e25/mtp_true_decode_gain_pct_median_of_8` = **+3.1797 %**
(mean +3.2772, min -1.3644, max +7.2377, **improved 7/8**).

| prompt | base ms/tok | cand ms/tok | gain % | serial drift % | base d-bar | cand d-bar | deep-round % | cliff ms/rd | acc d | rounds d | max d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| natural_history | 49.638 | 50.315 | **-1.364** | -0.202 | 2.204 | 1.981 | 5.38 | 48.91 | -8 | +8 | 4 -> 3 |
| dramatic | 47.939 | 46.706 | +2.572 | +0.350 | 2.692 | 2.287 | 20.54 | 49.58 | -12 | +13 | 5 -> 3 |
| narrative | 46.503 | 45.268 | +2.657 | +0.330 | 2.364 | 2.142 | 5.51 | 48.27 | -3 | +3 | 5 -> 3 |
| technical | 45.390 | 44.058 | +2.935 | +0.119 | 2.516 | 2.275 | 8.52 | 54.16 | -5 | +6 | 4 -> 3 |
| travel | 46.754 | 45.153 | +3.424 | -0.157 | 2.470 | 2.218 | 9.13 | 56.92 | -8 | +8 | 4 -> 3 |
| philosophy | 46.583 | 44.909 | +3.593 | +0.316 | 2.572 | 2.294 | 8.30 | 48.37 | -6 | +6 | 5 -> 3 |
| medicine | 47.492 | 45.041 | +5.162 | +0.220 | 2.638 | 2.271 | 12.66 | 49.26 | -7 | +7 | 5 -> 3 |
| english | 49.629 | 46.037 | +7.238 | -0.109 | 2.336 | 2.138 | 9.31 | 55.93 | +2 | -1 | 4 -> 3 |

Serial-control drift between the paired legs is small: max |drift| 0.3496 %,
median 0.2111 %. Across all sixteen legs the serial phase spans
**74.435 - 74.988 ms/tok (0.74 %)**, so within-pair host speed cannot
manufacture gains of 2.6-7.2 %.

### 21.3 The structural claim is exact and replicates 8/8

Three counter-based facts hold on **every** prompt, with no timing noise:

- `effective_max_draft_len` collapses **4 or 5 -> 3** on 8/8;
- `candidate_rounds_above_cliff` = **0** on 8/8;
- mean draft length falls on 8/8 (2.204-2.692 -> 1.981-2.294).

This settles deliverable (1): **arm D is a hard `DEEP_CAP = 3`, not a price.**
`research/e25r2_rule.py` confirms the mechanism analytically -- with the
measured `c_3 = 0.4394 > 1/4`, depth >= 4 fires **0 / 400,000** draws under
both monotone and iid acceptance models, and the corner-case slack at `p = 1`
is -0.7698. The wall is intrinsic to the measured `M = 5` weight-stream pass
cliff, not to arm D's `max(...)`.

### 21.4 The gain is a near-break-even trade, and one prompt goes underwater

The cap buys gross time by deleting above-cliff rounds, then gives much of it
back as extra rounds and lost accepted drafts:

| prompt | gain % | cliff rounds removed | cliff ms/rd | gross saved ms | rounds d | acc d | net saved ms | cost as % of gross |
|---|---|---|---|---|---|---|---|---|
| natural_history | -1.364 | 14 | 48.91 | 684.7 | +8 | -8 | **-346.8** | **150.6** |
| dramatic | +2.572 | 46 | 49.58 | 2280.7 | +13 | -12 | +631.4 | 72.3 |
| narrative | +2.657 | 13 | 48.27 | 627.5 | +3 | -3 | +632.7 | -0.8 |
| technical | +2.935 | 19 | 54.16 | 1029.0 | +6 | -5 | +682.2 | 33.7 |
| travel | +3.424 | 21 | 56.92 | 1195.3 | +8 | -8 | +819.6 | 31.4 |
| philosophy | +3.593 | 19 | 48.37 | 919.1 | +6 | -6 | +856.9 | 6.8 |
| medicine | +5.162 | 29 | 49.26 | 1428.5 | +7 | -7 | +1255.3 | 12.1 |
| english | +7.238 | 23 | 55.93 | 1286.3 | -1 | +2 | **+1839.1** | **-43.0** |

Collateral cost spans **-43 % to +151 %** of gross saving. `dramatic` is the
sharpest counter-example to any "more cliff rounds means more gain" story: it
has by far the most above-cliff rounds (20.54 %, 46 removed, 2280.7 ms gross)
yet lands at only +2.572 % because it gives back 72.3 %.

Pooled over 1878 base rounds: 184 above-cliff rounds (9.80 %) at 51.37 ms of
excess each, gross 9451.2 ms, net 6370.3 ms -- **32.6 % given back** --
with `accepted_draft_delta` -47, `rejected_draft_delta` -351,
`round_count_delta` +50.

I deliberately **do not quote a single "share of saving from cliff
avoidance"**. Per-prompt it ranges 0.699-3.612 (undefined for
`natural_history`, whose net saving is negative), and its pooled value has
drifted 0.774 -> 1.082 -> 1.128 -> 1.214 -> **1.484** as n grew from 2 to 8.
A statistic that moves that much with sample size is not a measurement.

### 21.5 Run-to-run reproducibility, and why it dominates everything per-prompt

An earlier job measured `technical` and `dramatic` while the worktree was
dirty (`meta dirty=1`). Rather than discard those legs I archived them and
**re-ran both pairs cleanly**, giving two independent replicates of the same
arm, prompt, binary, head and host. The result is the most important
methodological finding in r2:

| prompt | leg | first run | clean redo | delta |
|---|---|---|---|---|
| technical | BASE | 46.986 | 45.390 | **-1.596 ms/tok (-3.397 %)** |
| technical | PRICE | 44.034 | 44.058 | +0.023 ms/tok (+0.053 %) |
| technical | **gain** | **+6.283** | **+2.935** | **-3.347 pct points** |
| dramatic | BASE | 47.964 | 47.939 | -0.025 ms/tok (-0.052 %) |
| dramatic | PRICE | 46.317 | 46.706 | +0.389 ms/tok (+0.840 %) |
| dramatic | **gain** | **+3.433** | **+2.572** | **-0.861 pct points** |

Repeating an identical measurement moved one leg by **3.4 %** and one
per-prompt gain by **3.3 pct points**. The dirty flag itself cannot be the
cause: the worker binary digests were identical, and the flag is only a
`git status` line count recorded in `meta.txt`. The most likely cause is
thermal state, since `MLXFAST_LOCAL_COOL_GATE=0` was set for all r2 legs and
the wrapper emits exactly that warning ("hot-start timings are not comparable
to gated runs").

Treating each pair as two independent measurements of one quantity gives
`sd(single gain) ~= |delta| / sqrt(2)`, so **sigma ~= 1.73 pct points** per
per-prompt gain (pooled over the two replicates; only 2 degrees of freedom, so
this estimate is itself uncertain). Consequences, which I apply to my own
claims below:

- **median SE ~= 0.77 pct points**, so the headline is **+3.18 +/- ~0.77**;
- the headline is **not** distinguishable from r1's +3.8346;
- `natural_history`'s -1.364 is only **0.79 sigma** from zero.

### 21.6 Retraction 1: "8/8" and the natural_history regression

r1 reported a clean **8/8** sweep at +3.8346 %. On base `d7619a7` that does
not reproduce: `natural_history` measures **-1.364 %** with all gates green
and only -0.202 % serial drift. So **the 8/8 claim is retracted**; the honest
count is **7/8 improved**.

But by §21.5 I must also refuse the tempting stronger claim. At 0.79 sigma,
**`natural_history` is not established as a genuine per-prompt regression.**
The defensible statement is: one prompt measured negative, arm D is not
demonstrated to be a universal win, and a single prompt's sign is below this
harness's resolution. The median is robust to it either way -- leave-one-out
medians span only **2.9355 - 3.4239**, and dropping `natural_history` gives
3.4239.

### 21.7 Retraction 2: no covariate-gain relationship is identifiable at n = 8

I earlier asserted an "inverted-U" dose-response between a prompt's
above-cliff share and its gain, then proposed a host-normalised replacement.
**Both are withdrawn.** The correlation flips sign with subset choice:

| subset | r(above-cliff share, gain) |
|---|---|
| n=4 | "inverted-U" asserted |
| n=6 set A | **-0.101** |
| n=7 | **+0.333** |
| n=6 set B | **+0.706** |
| **n=8 final** | **+0.227** |

The host-normalised version dies the same way: `r(base local_ratio, gain)` was
**-0.729** on set A and **+0.056** on set B, and lands at **+0.013** on the
full eight. Final-8 correlations are all weak: cliff ms/round +0.445, base
mean draft length +0.336, accepted rate +0.160, base MTP ms/tok -0.034, base
serial ms/tok -0.277.

The decisive counter-example needs no statistics: `natural_history`
(base 49.638 ms/tok, local ratio 1.5038) and `english` (49.629, 1.5016) are
nearly identical on every covariate, yet their gains differ by
**8.6 pct points**.

§21.5 explains why this was always doomed: with sigma ~= 1.73 pct points on a
per-prompt gain and a total spread of about 8.6, eight points cannot support
any correlation claim. **The methodological finding is that per-prompt
covariate analysis is not identifiable in this harness at n = 8.** Defensible
evidence here is per-round (1878 pooled rounds) or counter-based (§21.3), not
per-prompt.

### 21.8 Honest residual

`english` has a net saving (1839.1 ms) that **exceeds** its gross cliff saving
(1286.3 ms), so cliff avoidance accounts for only ~70 % of its gain. It is
also the only prompt where the cap improved *both* accepted drafts (+2) and
round count (-1). I have no mechanism for the remaining ~30 % and am not
claiming one; given §21.5 some of it is plausibly measurement noise.

### 21.9 Why the offline replay under-predicted by ~10x

The offline policy replay (§15) predicted arm D at **+0.322 %** against the
shipped rule on the same 1578-round tape. The timed median is **+3.18 %**, a
factor of **9.9x**. These do not contradict each other because they measure
different things: the replay prices rounds with the fitted per-depth cost
model `T = 30.120 + 10.172*d + 32.378*ceil((d+1)/IPG)` on a **fixed** tape,
whereas the timed run lets the policy change which rounds occur at all. The
replay therefore cannot see the +50 round-count delta, the -351 rejected
drafts, or the second-order effects of never entering the two-pass regime.
The replay is the right tool for ranking policies at fixed acceptance; it is
the wrong tool for absolute speedup.

### 21.10 Transfer to ranked, re-anchored on the live bar

The live promoted bar is **3.2341518328631** (submission
`942e5ab2-1c46-4c50-b7c3-eaf948878ed0`, frontier `474c7501`). I predict arm D
**regresses** there, and I recommend against submitting it:

- the ranked pool runs at mean draft length **5.078**, which under the shipped
  rule implies a uniform acceptance of **0.8168**, versus local measured
  `p0 = 0.6926` -- an acceptance gap of **+0.1242**;
- arm D caps at 3, so on ranked it would truncate the *majority* of rounds;
- the in-source h-sweep is the closest available natural experiment: raising
  `h` 0.18 -> 0.32 shortened drafts by only ~19 % (4.35/4.89/5.78/5.33/5.04 ->
  3.36/4.01/4.53/4.03/4.76) and **lost ~3 % of score** with the baseline leg
  flat and candidate decode time *rising* 0.95 %. With 0.15 and 0.14 also
  failing, `h = 0.18` is bracketed on both sides and is a true local optimum.

I explicitly **withdraw my r1 claim that the h-sweep supports arm D**; that
was a cross-era comparison and is wrong. The h-sweep evidence points the other
way: that pool rewards depth.

### 21.11 Honest label

**Local winner with a proved mechanism and a predicted ranked regression.**

- Proved: arm D is a hard `DEEP_CAP = 3` (8/8 counters, 0/400,000 analytic).
- Measured: **+3.18 +/- ~0.77 %** median on 8 local prompts, 7/8 improved, all
  gates green.
- Bounded: the trade is near break-even (32.6 % of gross given back) and sign-
  negative on one prompt.
- Predicted: a material regression on the ranked pool, so **no Yukon
  submission is recommended** (the submission slot is in any case busy,
  receipt `9197ed62-621f-474d-bfba-e1efddd9dd4c`).
- Retracted: r1's 8/8 sweep, both dose-response claims, and r1's reading of
  the h-sweep.

### 21.12 Limitations

1. Local M4 Pro, not the ranked M5; `arch_gen` on the runner is unconfirmed,
   though `get_qmv_batch_limit` gives vector_limit 10 for gen >= 15 so the
   crossrow switch should be live there too.
2. `MLXFAST_LOCAL_COOL_GATE=0` on all legs; §21.5 quantifies the cost of that
   choice as sigma ~= 1.73 pct points per per-prompt gain.
3. The noise estimate itself rests on only **two** replicate pairs.
4. One pair per prompt; the ranked harness averages differently.
5. Sub-cliff price design is worth at most +0.32 % (§15), so the interesting
   lever is collapsing the cliff itself -- handed to thorfinn's E27, not
   duplicated here.


## 22. r2 suggested follow-ups (not implemented)

These are offered as evidence and hypotheses. I did not implement any of them.

### 22.1 The cost-side cliff — evidence handed to thorfinn's E27, not duplicated

The advisor assigned the cliff-collapsing lever to **thorfinn as E27**, and this
experiment exists because of **thorfinn's E22 follow-up #1**, which asked whether
the per-row price should be measured rather than assumed. r2 answers that
question and, in doing so, localises the cliff precisely enough to be useful to
E27:

- The cliff is the **weight-stream pass count** inside Metal kernel
  `affine_qmv_fast`, Tier A of the crossrow gate at
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/metal/kernels/quantized.h:1917`
  (runtime-effective twin `mlx-generated/quantized.cpp:1930`).
- Passes are exactly `ceil(M / IPG)` where `M = depth + 1`. The switch ships
  IPG 3 at `M = 5` (`crossrow_affine4_g64_m<T,5,3,true>`, `quantized.h:1938-1942`),
  so `M = 5` needs **2** passes where `M = 4` needs 1. That single boundary is the
  whole `c_3 = 0.4394` overshoot.
- A single-pass `M = 5` specialisation would be `<T,5,5,true>`. The
  `static_assert` at `quantized.h:1169` requires `M % IPG != 1`, and `5 % 5 == 0`,
  so that instantiation is legal by the header's own rule. The open question is
  register and threadgroup budget, not legality.
- If `M = 5` became single-pass, measured `c_3` would fall from ~0.4394 toward
  the ~0.09 regime seen at `c_4`–`c_6`, which is comfortably under the `1/4`
  admissibility ceiling. Depth 4 would then be reachable **without** any change
  to the price rule.

That is the whole point of the negative result in §16: the price rule cannot be
fixed, because the wall is in the hardware dispatch, not in the rule.

### 22.2 Extend the forced cycle to mod 9 to measure `T(8)` directly

`research/e25r2-force-depth.sh` cycles `[0,1,2,3,4,5,6,7]` on
`(roundCount - 1) % 8`, so depth 8 (`M = 9`, the **second** pass cliff at
`quantized.h:1961-1965`) is reported from the pass-count model only (§20.3).
Changing the cycle to mod 9 measures it directly at the cost of one extra
forced leg per prompt. Worth doing only if someone needs `T(8)` as a measured
number rather than a modelled one.

### 22.3 Attack acceptance instead of cost

Conditional acceptance is monotone and there is **no cliff at position 3**
(§13.6): `p0 = 0.693`, `p1 = 0.584`, `p2 = 0.508`, `p3 = 0.419`. Because the new
declared head's `draftTokenIDWithDeclaredRerank` (`Qwen35.swift:3142`) is marked
proposal-side only — verification and the row ledger are untouched
(`Qwen35.swift:2376-2379`) — raising `p` at positions ≥ 3 is legal and does not
touch the target answer. Concretely: the rerank currently takes the coarse
affine-2 top-**32** and re-scores those rows in affine-4. Taking top-**64**
costs one 64-row affine-4 QMV instead of a 32-row one, on the proposal side
only. If that lifts `p3` from 0.42 toward `p2`'s 0.51, the reach product
survives one position deeper. This is the cheapest untried lever I found.

### 22.4 A ranked-pool constant-depth control

Pooled realised-rate argmax on this fixture is `d = 2` with bootstrap share
0.801 (§13.3), and the shipped adaptive rule already sits at mean depth 2.09 on
this fixture — so there is almost nothing to win locally
(`greedy_leaves_on_table_pct = 0.0`). But the in-source ranked h-sweep shows the
shipped rule running at mean drafts of **4.35–5.78** on the ranked pool. Those
two facts cannot both describe the same workload. The cheap decisive test is a
ranked-pool constant-depth-2 leg against the shipped adaptive rule: if constant
depth 2 wins there too, this fixture is representative and the price design
space really is worth ≤ +0.32 %; if it loses badly, the fixture is
unrepresentative and every local depth conclusion in this report needs a ranked
replication before anyone acts on it. I cannot run this (§23.2).

### 22.5 Confirm the ranked M5's `arch_gen`

`get_qmv_batch_limit` (`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp:84-126`)
returns **6** for `arch_gen == 13 || 14` and **10** otherwise. This host is Apple
M4 Pro / `Mac16,11` / `applegpu_g16s`, so `arch_gen = 16`, limit 10, and the
crossrow switch is live for every depth `0..8` (`M ≤ 9 < 10`). On a gen-13/14
host, `M ≥ 6` would fall through to `qmm` instead and the measured curve above
depth 5 would not describe the ranked machine at all. Nothing in the source
lets me derive the M5's generation. Smallest resolving read:
`mtl_device()->architecture()->name()` on the ranked runner.

### 22.6 Remove the trace-I/O timer bias for future within-round attribution

`upkeep_bias` is 0.245–0.269 ms per round (§13.5), i.e. ≤ 0.21 % of round time
and immaterial to every conclusion here. It is, however, a real systematic
offset introduced by writing the trace. A trace-off timed leg would remove it
entirely and is worth adding if anyone later attributes cost *within* a round
rather than across depths.

### 22.7 Restore the cool gate and replicate pairs — the highest-value methodology fix

This is the follow-up I would rank first, and it comes directly out of §21.5.
Re-measuring an identical `technical` BASE leg moved it **-3.397 %**, which
alone swung that prompt's reported gain by **3.347 pct points**. The implied
per-prompt noise is **sigma ~= 1.73 pct points**, which is the same order as
most of the per-prompt effects this campaign reports.

Concretely, this means several r2/r1 numbers were over-read, including my own
r1 "8/8 at +3.8346" and every per-prompt correlation in §21.7. It probably
also affects other campaign experiments that compare single matched pairs at
the few-percent level on this host.

Two cheap changes would fix it:

1. **Stop setting `MLXFAST_LOCAL_COOL_GATE=0`.** It was bypassed for
   throughput, and the wrapper's own warning states that hot-start timings are
   not comparable to gated runs. §21.5 is the first direct measurement of what
   that costs.
2. **Run at least two matched pairs per prompt** and report a per-prompt
   spread, so the harness reports its own resolution instead of leaving the
   reader to assume a single pair is exact.

Cost is roughly 2x the timed budget, or the same budget over four prompts
instead of eight. Given that the median moved 3.83 -> 3.18 between bases and
the per-prompt sigma is 1.73, I think fewer prompts measured twice is the
better trade for any experiment whose effect size is under ~5 %.

## 23. r2 blocked on

### 23.1 GitHub read and write credential (two outages, both since RESOLVED)

Two separate GitHub outages interrupted r2 reporting:

1. An earlier window in which a raw
   `GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/29` with the injected
   token returned **HTTP 401 `Bad credentials`**.
2. A later window in which `get_prs` **and** `post_assignment_comment` both
   returned **HTTP 403**. Because reads and writes failed together, this was a
   service/credential outage rather than a tool defect;
   `post_assignment_comment` failed three times with the same reserved
   `comment_id`.

**Both are resolved.** Access recovered and the pending interim comment posted
as `issuecomment-5334334360`; a subsequent PR read confirmed the assignment is
open at head `8ca8eb3f...` with **no advisor feedback missed** during either
window, other than the maintenance-checkpoint request (`feedback_id`
5334165606), which is acknowledged and honoured: the named job
`fec86fc2-93a4-4691-b88b-04426e5a271a` finished exit 0, the later clean-redo
job `290f9371-...` was allowed to finish rather than abandoned mid-leg, and
**no further job was launched** -- all remaining r2 work was CPU-only analysis.

This item is retained as a record, not as an open blocker.

### 23.2 Ranked-pool measurement, and therefore the transfer question

This is the most important thing r2 could **not** settle. The official
submission slot is busy (receipt `9197ed62-621f-474d-bfba-e1efddd9dd4c`) and
`program.md` plus the assignment forbid me running `senpai/submit-official.sh`.
So the central prediction of §19 — that arm D's `DEEP_CAP = 3` truncates the
majority of ranked rounds and **regresses** against the live bar of
`3.2341518328631` — is stated with an explicit mechanism and an explicit
comparison class, but is **not measured**. It should not be treated as
established. Anyone who wants to act on it should run §22.4 first.

### 23.3 `T(8)` and the M5 generation

Both are recorded above as follow-ups (§22.2, §22.5) because both are cheap;
they are listed here too because until they are done, the depth-8 row of every
table in this report is modelled rather than measured (§20.3), and the entire
curve's transfer to the ranked M5 rests on an unverified `arch_gen` assumption.

