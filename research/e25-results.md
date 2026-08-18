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

<!-- POOL:CURVE -->

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

<!-- POOL:POLICY -->

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

