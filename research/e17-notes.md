# E17 — curve transfer and refit

Assignment `qwen38-r1-e17-curve-transfer-and-refit`, revision r1, PR #19.
Base `e6e6f81767e84cc8c39b48c09a4f5cac597cdbca` (PR #13 merged; the per-depth
`headStepCostRatioByDepth` curve is the compiled default).

---

## 1. Pre-registration (deliverable 14)

**Signed and committed before any E17 timing arm was launched.** The commit that
introduces this section contains no E17 run artifacts; `.mlxfast-private/e17/`
does not yet exist. That ordering is the only thing that makes the rest of this
document worth reading, because the question E17 asks — does a curve fitted on
one prompt survive a median over eight? — is exactly the question that
prompt-level or convention-level freedom can fake.

### 1.1 What is being compared

Two binaries differing in eight `Double` literals and nothing else:

| arm | `headStepCostRatioByDepth` |
| --- | --- |
| `CURVE` | `[0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]` (this branch's HEAD) |
| `FLAT18` | `[0.18 × 8]` |

`FLAT18` is the counterfactual rather than a revert of the pre-merge file
because the shipped extend test is

```text
extend at depth d  iff  reach_d > h[d] · (1 + expected_d) / (1 + cumH_d)
```

and with a flat vector `cumH_d = d·h`, so a flat `0.18` reproduces the retired
scalar rule `h(1+expected)/(1+d·h)` term for term. The contrast is therefore the
constants alone, not a partial revert of everything else that moved between
bases. `research/e17-build.sh` asserts, per arm, that the vector declaration is
present, that no scalar `headStepCostRatio` is declared, that the arm's own
literals are installed, and that `sdpaWidthWallDepthCap = 5`,
`segmentedVerifyDepthCap = 8` and `segmentedStreakGate = 3` are untouched. The
`FLAT18` patch is keyed on the exact shipped literal, so a base that re-fitted
the curve makes the build fail loudly instead of silently producing a third arm.

### 1.2 Prompt set and order — fixed here

Eight 512-token prose seeds, 512 decode steps, one golden each:

| # | id | register | prompt file | seed tokens |
| --- | --- | --- | --- | --- |
| 0 | `english` | expository history | `research/e11_prose_gate_english_512.txt` | 870 |
| 1 | `narrative` | literary narrative | `research/e17_prose_narrative_512.txt` | 981 |
| 2 | `technical` | technical exposition | `research/e17_prose_technical_512.txt` | 903 |
| 3 | `dramatic` | stage dialogue | `research/e17_prose_dramatic_512.txt` | 1146 |
| 4 | `travel` | first-person travel | `research/e17_prose_travel_512.txt` | 985 |
| 5 | `philosophy` | philosophical argument | `research/e17_prose_philosophy_512.txt` | 909 |
| 6 | `natural_history` | 19th-c. natural history | `research/e17_prose_natural_history_512.txt` | 1005 |
| 7 | `medicine` | clinical review | `research/e17_prose_medicine_512.txt` | 939 |

Seed-token counts are from `research/e17_token_check.py` against the target
tokenizer; `generate-golden` uses the first 512 tokens, so every prompt has a
full seed window with margin.

Commitments:

* **No prompt is added, dropped or reordered after timing starts**, and every
  prompt that completes is reported, including any that embarrasses the curve.
* `english` is E11's prompt and therefore **in-sample**: the shipped curve was
  fitted against marginals measured on it. It is retained as an anchor that ties
  E17 back to the r3 pair, and it is reported, but the **headline is the
  held-out 7**. The all-8 median is reported alongside it because 8 prompts
  mirror the ranked pool's shape and exercise the even-`n` rule.
* The other seven are original prose written for this experiment, in registers
  chosen to spread acceptance behaviour (dialogue and clinical prose should
  draft differently from narrative and travel). None is a copy or repetition
  task.
* `correctness_prompts/public_longcopy_gate_english_512*` — the copy fixture
  whose 0.89–0.95 accept rate flatters deep drafting — **is not used**, and
  nothing under `fixtures/` or `correctness_prompts/` is touched. Goldens land
  outside Git in `.mlxfast-private/e17/goldens/`.

### 1.3 Interleaving and order-effect control

* Arms are interleaved **within** prompt, never blocked by arm, so session-scale
  thermal drift cannot correlate with arm.
* The within-prompt order **alternates with the prompt's index**: even index →
  `CURVE, FLAT18`; odd index → `FLAT18, CURVE`. A systematic first-arm/second-arm
  effect (cold cache, cool-gate history) therefore cannot correlate with arm
  either.
* Every run is serialized through `research/await-lock-then-run.sh`, so one
  model-holding process is resident at a time and each arm passes the unmodified
  40 °C cool gate, drift tripwire and orphan scan.
* GPU/CPU temperature and power are sampled before and after every arm into that
  arm's `meta.txt`.
* Timed arms run with every `MLX_QWEN_MTP_*` name cleared (`research/e11-run.sh`
  unsets them), so no trace or research override can leak into a headline; the
  verbatim surviving list is recorded per arm. Both arms use the pinned/declared
  head at `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared`.

### 1.4 Metric convention — fixed here

For each prompt `p`:

```text
raw_p = serial_seconds_per_token / mtp_seconds_per_token
```

Both terms are read **verbatim** from `.parent_measured_seconds_per_token` in
that run's own reports — `03-mtp-timed.json` for the depth-0 serial control leg,
`04-mtp-timed.json` for the depth-8 arm leg. **Nothing is subtracted.** This is
prefill-inclusive by construction, and §2 below establishes that from source.
Each `--local-iterate` invocation measures its own byte-identical serial control,
so each prompt yields two independent serial legs and their spread is that
prompt's noise floor.

Reported statistics:

* `median(raw_p | CURVE)` and `median(raw_p | FLAT18)`, ranked median (with even
  `n`, the mean of the two middle values), over the held-out 7 and over all 8.
* `g_p = (mtp_spt_FLAT18 − mtp_spt_CURVE) / mtp_spt_FLAT18` — the curve's
  per-prompt gain on the MTP leg — and `g_median`. This is the same quantity r3
  published as `g = 6.378%`, so E17's `g_median` is directly comparable to it.
* Per-prompt full table, min/max spreads, win count, per-prompt serial noise
  floor, temperatures.
* Per arm per prompt: depth histogram (from `effective_draft_lengths`), round
  count, mean/max effective depth, accepted/rejected totals, accept rate,
  replayed-round count, row accounting, `all_tokens_matched`, `parity_all_ok`,
  `residual_divergence_count`, head sha256, binary sha256.

### 1.5 Stop rules — fixed here

| outcome | reading | action |
| --- | --- | --- |
| `g_median ≥ +4%` | the curve's r3 win is real at ranked scale | report a transferring win; the curve is the right lever to keep refining |
| `+1%` ≤ `g_median` < `+4%` | real but much smaller than the single-prompt 6.378% | publish the shrinkage honestly; the curve stays, but priority should shift to levers with more headroom |
| `g_median < +1%`, or the sign flips across prompts | **the r3 headline was substantially a one-prompt artifact** | publish as the most valuable negative available and stop refining `h`; a refit fitted on one prompt inherits the same defect |

### 1.6 Falsification statement

This experiment is designed so that the curve can lose, and the following would
falsify "the merged per-depth curve is a genuine ranked-scale improvement":

1. `g_median` over the held-out 7 falls below `+1%`, i.e. an order of magnitude
   below the `+6.378%` r3 published from `english` alone.
2. `g_p` changes sign across prompts — the curve helps some registers and hurts
   others — so a median over eight unseen prompts is a coin flip and the
   *published median* rather than the mean is what decides the score.
3. The curve's advantage on the held-out set is smaller than the per-prompt
   serial noise floor, in which case r3's win was inside measurement error all
   along and only looked clean because a single prompt was measured twice.
4. `CURVE`'s depth histogram on held-out prompts collapses to the same depths
   `FLAT18` chooses, which would mean the eight fitted constants are not
   actually changing the schedule off the fitting prompt and the r3 delta came
   from something else.

If any of 1–3 holds I will report the curve as a one-prompt artifact, in those
words, and will say so in the result summary rather than foregrounding the
in-sample anchor. I will not recover a headline by dropping prompts, switching to
the all-8 or decode-only variant after the fact, substituting the mean for the
median, or re-timing a losing prompt until it wins. Any repeat run will be
reported as a repeat, with both values shown.

The in-sample anchor supplies a fifth, diagnostic falsification: if `english`
itself no longer reproduces something close to r3's `g = 6.378%` on this base and
host, then the E17 harness — not the curve — is what changed, and no held-out
conclusion may be drawn until that is resolved.

---

## 2. The metric convention, from source (deliverable 12 support)

The advisor's r1 feedback is **right** that the ranked score is prefill-inclusive
and that E11's r3 prose claiming seed prefill is "outside the ranked score" was
wrong. The chain, in the trusted (non-editable) driver:

| # | source | what it establishes |
| --- | --- | --- |
| 1 | `Sources/MLXFastRuntime/QwenRuntimeMTPDriver.swift:94` — `let started = Date()` | the clock starts **before** any decode work |
| 2 | `…:95` — `beginMTPDecode(...)` | seed prefill happens **after** the clock started, so it is inside the span |
| 3 | `…:197` — `decodeSeconds = Date().timeIntervalSince(started)` | the span runs to the last emitted token and contains prefill |
| 4 | `Sources/MLXFastRuntime/QwenRuntimeMTP.swift:442` — `decodeSecondsPerToken = decodeSeconds / max(decodeTokenCount, 1)` | the whole span, prefill included, is divided by decode tokens only |
| 5 | `Sources/mlxfast-swift/main.swift:~2013-2028` | emits `parent_measured_seconds_per_token` from that value, and emits `seed_prefill_seconds` / `prefill_seconds_per_token` beside it marked *"Observability only — nothing above subtracts it — and omitted (never zero)"* |
| 6 | `benchmark-qwen-mtp.sh:651-652,726` | the reported ratio is built from `parent_measured_seconds_per_token`, unmodified |

So a 512-token seed is charged to a 512-token decode window, and both legs of
every pair carry it. There is no subtraction anywhere on the scored path, and
`seed_prefill_seconds` exists purely to be looked at.

**Consequence for prompt design, which is the reason this matters:** prefill is a
fixed per-leg cost that speculation cannot reduce, so it dilutes every ratio
toward `1.0`. Measured here, `P ≈ 3.995 s` against a serial leg of `38.2 s` and
an MTP leg of `23.7–25.3 s` — i.e. prefill is ~10 % of the serial leg and ~16 %
of the MTP leg. The prefill-inclusive ratio is therefore *lower* than the
decode-only ratio (`1.609` vs `1.733` for the curve arm), and any candidate's
apparent speedup is capped by how much of its leg is unspeculatable prefill.

## 3. Where the r1 arithmetic restatement went wrong

The convention correction is right; the numbers attached to it are not, and the
correction must not be recorded with them. `research/e17_analyse.py --r3`
recomputes both r3 arms from their own committed reports:

```text
Sp3 (scalar h=0.18)   serial 0.074569517513737082  mtp 0.049472851445898414
Hp3 (merged curve)    serial 0.074528021505102515  mtp 0.046317363390699029
measured seed_prefill_seconds: 3.994360 .. 3.998045 s (all four legs, n = 512)

                    Sp3        Hp3      pair delta
prefill-inclusive  1.507282   1.609073   0.101791   <-- equals what r3 published
decode-only        1.602116   1.732572   0.130456
```

The published r3 pair **already was** the prefill-inclusive pair, to six
decimals; likewise `g = 6.378%` already was the prefill-inclusive MTP-leg gain
(decode-only would have been `7.577%`). Applying the advisor's convention to r3
therefore changes no published number — only the surrounding prose was wrong.

The r1 feedback instead quotes `Sp3 = 1.437971`, `Hp3 = 1.521771`, "scored delta
17.67% smaller". Those come from adding `P/n` to values that already contained
it — charging seed prefill twice:

```text
(serial_spt + P/n) / (mtp_spt + P/n):  Sp3 = 1.438285   Hp3 = 1.521191
advisor r1 quoted                   :  Sp3 = 1.437971   Hp3 = 1.521771
pair delta 0.101791 -> 0.082906, i.e. 18.55% smaller (r1 quoted 17.67%)
```

The reconstruction lands within 0.03 % of both quoted ratios and reproduces the
"~18 % smaller delta" claim, which identifies the operation. The residual is
consistent with a single rounded `P` instead of each leg's own measured value.

Note also the direction: moving from decode-only to prefill-inclusive *shrinks* a
ratio, so had r3 really published decode-only numbers, the corrected values would
have been `1.602116 / 1.732572` — **larger**, not smaller. E17 uses the
prefill-inclusive convention throughout regardless, per §1.4, because that is
what the driver measures.
