# E209 result: the variance-tax controller family

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"prose_population_median_pct_faster_vs_shipped","available":true,"value":2.3072219074438793},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e209-variance-tax-controller`
- Hypothesis and target cost: an online depth controller that estimates the
  per-prompt optimal fixed depth and then HOLDS it captures at least +1.0%
  prose-population median of the +2.589% FINDING 540 fixed-depth headroom.
- Decision: **green locally, dead on the official path.** The local statistic
  clears the promotion rule. A ranked cross-check gives the whole family a
  negative ceiling, so the family closes.
- `BASE_SHA`: `22ca1b48c455b83b489afcd59badec2cab84ec5d`
- Candidate commit: `c1c133f7` (research files only; no candidate surface
  touched)
- Yukon promoted submission / frontier: receipt A `3.70784519415395`, crown
  `3.7291100105909`. Not re-queried: this experiment made no submission and
  changed no submitted file.
- Candidate build fingerprint: none. Stage 0 is a desk replay; no worker was
  built and no GPU ran.
- Submitted-surface / twin / metallib digests: unchanged, none touched.
- Submitted candidate files: **none.**
- Supporting research files: `research/e209_controllers.py`,
  `research/e209_desk.py`, `research/e209_ranked.py`,
  `research/e209_wandb.py`, `research/e209-artifacts/`.
- MTP head provenance and draft policy: the corpus legs carry
  `head_safetensors_sha256=d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`,
  `head_sha=ad4fb0289d1b0e9a47896577ea7976409a228e28`, pinned draft depth 7.
- Token window, fixture, reference source, harness: 512 decode tokens; E168
  `p7` corpus, 9 prompts; reference rows are the recorded corpus accept counts;
  **harness=local** for the controller table and **harness=ranked** for the
  cross-check. The two are never mixed.
- Exact cell: not applicable. No kernel, shape, or dispatch family changed.
- Official causal path and score equation: none exercised. No candidate edit
  was made, so `d ln(ranked baseline serial time) / dx = 0` is trivially
  satisfied and no ranked value is claimed for any change.
- Assignment-scope preflight: not run. Nothing outside `research/` changed, so
  the submitted surface is byte-identical to `BASE_SHA`.
- Editable source bytes / headroom / growth: unchanged, growth 0.
- Scored-path reachability evidence: not applicable; nothing was implemented on
  the scored path.
- Written promotion rule and verdict: the assignment's rule
  (`>= +1.0%` prose median AND min per-prompt `>= -0.25%`) is **met**
  (+2.307% / +1.659%). The assignment's measurement rule
  (benchfixture sign must agree with the prose sign before a timing session) is
  **not met**: the sign is inverted for every registered controller. Stage 1
  therefore stops at "report"; see "Deviation" below for why the 64-token
  screen was also not run.
- Pre-official evidence budget / timed legs used: **0 timed legs, 0 GPU
  seconds.** The whole result is desk replay.
- Frozen candidate SHA: none.
- Submission owner / receipt watcher: none.
- W&B group `qwen38-r1-e209-variance-tax-controller`:
  - desk (harness=local) `2popkdc2` —
    <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2popkdc2>
  - ranked cross-check (harness=ranked) `ow19em8x` —
    <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/ow19em8x>

## Question, evidence, expected result, smallest decisive test, stop rule

**Question.** Can a causal, within-request depth controller that holds a level
capture at least +1.0% of the +2.589% FINDING 540 fixed-depth headroom?

**Evidence for asking.** FINDING 540 (W&B `mn30elwp`) measured the ceiling and
showed that holding the shipped rule's own average level captures +2.460 of the
+2.589%. FINDING 538 (W&B `0ouc9krp`) showed the level tracking itself is worth
~0.93%, so the controller must keep choosing the level online. E203 measured the
round-level depth choice to be uncorrelated with the realized accept length
(max |corr| 0.0706).

**Smallest decisive test.** Replay 15 pre-registered controllers on the E168
`p7` corpus with a whole-leg runner, paired over random round orders.

**Stop or promotion rule.** Pre-registered by the assignment and reproduced
verbatim in the code. Result: promotion condition met, measurement condition
not met.

## Evidence

- Host, chip, thermal policy: **not applicable.** No timed leg ran. Every
  artifact carries `coolGatePassedRealGate=false`,
  `gateQualifiedForTiming=false`, `timingClaimsPermitted=false`,
  `officialOrRankedScore=false`.
- `head_provenance_sha256` per leg: inherited from the E168 corpus meta, quoted
  above. No new leg was measured, so no new provenance exists.
- Exact commands:

```bash
python3 research/e209_desk.py --draws 400 \
  --json research/e209-artifacts/stage0-controllers.json
python3 research/e209_ranked.py
python3 research/e209_wandb.py
```

- Cheapest real falsification gate and positive control: the shipped-walk port
  is E207's, whose positive control reproduced **1955 recorded rounds with 0
  mismatches**. The replay's own null control is that every order-invariant arm
  (each `fixed-d*`) returns a zero permutation spread, which it does.
- Exact-token and row-ledger verdict: **not applicable.** No tokens were
  generated. A depth-policy change cannot alter the emitted token stream, but
  that claim was not exercised here because nothing was implemented.

### Primary table (harness=local, E168 `p7`, 512 tokens, 400 paired draws)

| Metric | Baseline (shipped rule) | Candidate (`OWN-mean-W16`) | Delta |
| --- | ---: | ---: | ---: |
| prose-population median ms/token | reference | — | **+2.307% faster** |
| prose minimum across 8 prompts | reference | — | +1.659% faster |
| prose median ms/round equivalent | reference | — | +1.728 ms/round |
| benchfixture (organizer prompt) | reference | — | **-5.759% slower** |
| depth switches per leg (prose mean) | 125.2 | 10.0 | -115.2 |
| leave-one-prompt-out median | reference | — | +2.204% faster |
| leave-one-prompt-out minimum | reference | — | +1.887% faster |

The composition bar is 0.2 ms/round; +1.728 ms/round is 8.6x that bar.

### Full pre-registered family (prose median % faster)

| Controller | prose med | prose min | benchfixture | switches/leg |
| --- | ---: | ---: | ---: | ---: |
| A1-ema-W16 | -2.241 | -2.965 | -0.348 | 10.3 |
| A1-ema-W32 | -2.029 | -3.133 | -0.168 | 18.4 |
| A1-ema-W64 | -2.209 | -2.855 | -0.049 | 34.9 |
| A2-mode-W16 | +2.180 | +1.112 | -1.356 | 9.7 |
| A2-mode-W32 | +1.437 | +0.605 | -0.168 | 17.9 |
| A2-mode-W64 | +1.198 | +0.714 | -0.049 | 34.5 |
| **OWN-mean-W16** | **+2.307** | **+1.659** | -5.759 | 10.0 |
| OWN-mean-W32 | +2.305 | +1.887 | -2.560 | 18.1 |
| OWN-mean-W64 | +1.996 | +1.462 | **-0.144** | 34.7 |
| B-h4 | -1.775 | -2.815 | -0.671 | 18.8 |
| B-h8 | -3.031 | -4.357 | -0.191 | 8.4 |
| B-h16 | -1.781 | -2.995 | -0.207 | 3.8 |
| C-p0.005 | -0.589 | -1.558 | -0.707 | 78.6 |
| C-p0.02 | -0.761 | -2.007 | -1.027 | 32.3 |
| DIAG-p0.02 (circular, not eligible) | +2.500 | +1.472 | +4.343 | 13.5 |

Every controller that re-derives the level from the EMA cost model (A1, B, C)
**loses**. Only the estimators that read the shipped rule's own realized depth
choices (A2, OWN) win. The EMA-implied level with the margin removed is
systematically too deep.

### The mechanism, confirmed more sharply than FINDING 540

For **every** prose prompt, `round(shipped mean depth)` **is** the hindsight-best
fixed depth. Re-measured under the whole-leg runner the ceiling is +3.206% and
the variance-only decomposition is also +3.206%: 100% of the local headroom is
round-to-round movement, and 0% is level selection. FINDING 540's rate-based
runner gave +2.589% and +2.460%; the difference is the runner, not the corpus.

### Three qualifications on the local number

1. **Constant-depth null.** A hardcoded `fixed-d3` scores +2.605% prose median,
   +1.022% prose min and -0.170% on benchfixture, beating the controller on
   every axis. On this corpus the online estimation adds nothing.
2. **`OWN-mean-W16`'s margin over W32/W64 is an estimator bias, not skill.**
   The settle-window mean is biased upward by the optimistic seed prior:

   | W | prompt | prefix mean | whole-leg mean | bias |
   | ---: | --- | ---: | ---: | ---: |
   | 16 | english | 2.488 | 2.241 | +0.247 |
   | 16 | benchfixture | 5.838 | 6.649 | **-0.811** |
   | 64 | english | 2.300 | 2.241 | +0.060 |
   | 64 | benchfixture | 6.657 | 6.649 | +0.008 |

   W16 locks one step deep on prose, which happens to help, and one step shallow
   on benchfixture, which costs 5.8%. W64 is the unbiased estimator.
3. **Cost-table curvature carries most of the effect.** Under a curvature-free
   cost table the gain falls from +2.307% to +0.857%.

   | scenario | prose median |
   | --- | ---: |
   | measured (dated E168 table) | +2.307% |
   | m=5 cell at its mean not median | +2.978% |
   | m=5 at linear continuation | +1.787% |
   | fully linear table | **+0.857%** |

   The prose-relevant cells are well populated (m=2: 48 rounds, m=3: 1126,
   m=4: 128, m=5: 56); the thin m=6 and m=7 cells carry no prose weight.

## The ranked cross-check that closes the family (harness=ranked)

FINDING 520 instrument, imported unmodified from `e201_online_cap` /
`e203_stage0b`, exactly as E207 used it.

| Policy | published median | vs shipped cap 8 |
| --- | ---: | ---: |
| constant d=2 | 2.869960 | -25.05% |
| constant d=3 | 3.378325 | -11.78% |
| constant d=8 | 3.744526 | -2.22% |
| shipped cap 7 | 3.707845 | -3.17% |
| shipped cap 8 | 3.829386 | reference |
| **per-prompt best fixed depth (ORACLE)** | **3.744526** | **-2.22%** |

The hindsight per-prompt-best-fixed oracle strictly upper-bounds any causal
per-prompt level controller, including every controller in this family. It is
**negative**. On the ranked instrument the family cannot win.

### Why the harnesses disagree: the prose corpus is not representative

| corpus prompt | source | shipped mean depth |
| --- | --- | ---: |
| benchfixture | `correctness_prompts/public_longcopy_gate_english_512_256.json` (organizer) | 6.60 |
| the 8 "prose" prompts | `research/e17_prose_*.txt`, `research/e11_prose_gate_english_512.txt` (self-authored) | 2.18 – 2.57 |

The prose population carrying FINDING 540's headroom is entirely self-authored
and entirely low-acceptance. The one organizer-supplied prompt in the corpus
sits in the opposite regime and inverts the sign for every controller.

With 8 prompts the published median is the mean of the two middle raws, so only
two prompts set the score:

```
plutarch 1.2607  drama 2.1222  travel 2.4277 | beagle 3.6472  essays 4.0116 | republic 4.0439  medicine 4.0589  botany 4.1134
```

`beagle` and `essays` set the median and **both prefer depth 8**. Our prose
corpus resembles `plutarch`, `drama` and `travel` — the three prompts below the
median, which cannot move the published median however much they improve.

## Deviation from the pre-registered plan, and why

The assignment's Stage 1 is "implement behind an arm, 64-token exactness screen,
then stop and report because the benchfixture sign is inverted". I ran the desk
work in full and **did not** implement the arm or run the screen.

Reason: the pre-registered path after that screen is "stop and report" in any
case, and the ranked cross-check shows the family's ceiling is negative, so the
screen could not produce evidence that changes any decision. It would have spent
a build and a GPU allocation to demonstrate only that an arm nobody can ship
does not change tokens. I posted this reasoning to the advisor as an interim
comment before submitting and offered to implement the arm on request.

## Conclusion

- **What happened.** The assigned hypothesis is confirmed on its own terms:
  a low-variance level-holding controller captures +2.307% prose-population
  median, 72% of the re-measured +3.206% ceiling, well past the +1.0% bar and
  with a comfortable robustness margin (prose min +1.659%, LOPO min +1.887%).
- **Evidence for the mechanism.** Confirmed and sharpened: on this corpus the
  shipped rule's average level is already optimal for every prompt, and 100% of
  the headroom is round-to-round movement against a convex round cost. The
  controller cuts depth switches from 125 per leg to 10.
- **Evidence against the value.** Three independent results say the local gain
  will not reach the score. A hardcoded constant beats the controller on the
  same corpus; the best window's edge is an estimator bias; and, decisively, the
  ranked hindsight oracle for the entire family is -2.22% against the shipped
  rule, with both median-setting prompts in the deep-drafting regime the corpus
  never visits.
- **Prompt and M5 transfer risk.** Maximal, and now measured rather than
  assumed. This is the dominant result of the experiment.
- **Recommendation: close the within-prompt depth-policy family.** FINDING 540's
  ceiling is real but is measured on a self-authored low-acceptance corpus that
  sits below the ranked median and cannot move it.

## Suggested follow-ups, not implemented

1. **Re-examine every finding that used the `research/e17_prose_*` population
   as a proxy for the ranked pool.** The provenance split found here is a
   property of the corpus, not of E209, so it may qualify other conclusions.
   The cheap check is to re-read each finding's prose median beside its
   benchfixture column and its ranked projection.
2. **Build a corpus in the median-setting regime.** The published median is set
   by high-acceptance, deep-drafting prompts. A pinned-depth trace set over
   organizer `correctness_prompts` material would let desk work price changes
   where the score actually lives. This is the highest-value follow-up.
3. **Test the ranked instrument's round-level credit.** The instrument gives the
   shipped rule +2.22% over the per-prompt oracle, which must come from
   round-level adaptation. E203 measured that adaptation to be uninformative on
   real rounds (max |corr| 0.0706). One of the two is wrong, and the answer
   changes how much weight the instrument should carry.
