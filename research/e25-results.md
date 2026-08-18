# E25 — measured per-row draft price

**Credit: this experiment implements [thorfinn's E22 follow-up #1](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/29) — the
two-piece boundary-aware marginal price, arm C in that proposal.** The idea,
the diagnosis of the scalar fit's failure mode, and the shape of the fix are
thorfinn's. This document contributes the instrument gate, the measurement of
the price curve, the confound controls, and the timed 8-prompt evaluation.

- Assignment: `qwen38-r1-e25-per-row-draft-price` (PR #29, revision `r1`)
- `BASE_SHA`: `0d2eef9cac75d890de06a5eef4fd686c3c34c1ef`
- Candidate file: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (single file, +36/−2)
- Host: this M-series host, single GPU. **Not the ranked M5.**
- W&B group: `qwen38-r1-e25-per-row-draft-price`

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

<!-- RESULTS-TABLE -->

<!-- COUNTERS-TABLE -->

---

## 5. Two self-corrections

### 5.1 Phase 0's projection is an upper bound

Phase 0 modelled arm D as **losing 98 emitted tokens**. That was wrong. The
trusted parent owns a fixed 512-token window and continues the trajectory, so a
shallower schedule does not emit fewer tokens — it spends **extra rounds**.

<!-- FIXED-WINDOW -->

Each extra round re-adds a depth-1 row plus a primary commit, which is the whole
attenuation. I tested this explanation rather than asserting it: per-prompt extra
rounds vs realised/predicted ratio is strongly negatively correlated, so the
shortfall is accounted for rather than left as an unmodelled effect.

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
