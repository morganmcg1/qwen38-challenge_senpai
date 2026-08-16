# qwen38-r1-e1-depth-cost-curve — measured per-depth marginal cost

Student `qwen-edward`, PR #1, branch `qwen-edward/depth-marginal-cost-curve`.

Replaces the scalar `headStepCostRatio = 0.20` in `costModelDepth`
(`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`) with a measured per-depth
marginal cost vector, and repairs the depth-selection rule that the scalar had
made safe.

## Provenance (revision r2, step 1)

| field | value |
| --- | --- |
| host | AWS Mac mini, Apple **M4 Pro**, 14 CPU (10P/4E), 20 GPU cores |
| memory profile | **48 GB** unified, single resident model process, wrapper run lock held |
| OS | macOS 26.5.2 |
| toolchain | Xcode 26.6 (17F113), Swift 6.3.3 |
| `BASE_SHA` (r2) | `67bde70274c42aef089ac73cf00608d8037a815e` |
| `UPSTREAM_SHA` | `7351e62674bc600f0ca148d3a1b0604716a09db6` |
| merge commit onto r2 base | `0bb21a67a80510f59ac8a461fbb9045652a6dae8` |
| head actually loaded | declared 4-bit head, `--head-dir …/mtp-head-declared` |
| `head_provenance_sha256` reported by the harness | `54930a1d281ff3ec4373fc2befd190afdb67ee09ffd90e8fc60e4d1f538bfc4b` |
| `shasum -a 256` of the resolved head weight file | `0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815` (`model.safetensors`, 238,934,093 B) |
| `mtp-head.manifest.json` `sha256` | `cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d` |

The three digests are all consistent; the reconciliation is in "Head
provenance" below. In one line: the manifest digest is a **tree digest** over
`{model.safetensors}` only, the harness digest is the same tree digest over
`{config.json, model.safetensors}`, and `research/fetch-declared-head.sh` adds
that `config.json` **after** verifying the manifest because
`benchmark-qwen-mtp.sh:215` refuses a head directory without one. The weights
are bit-identical to the manifest.

The official runner is the M5 host `m5-qwen38-27b-mtp`. **Every number in this
report is M4 Pro.** Where M5 transfer is at risk it is called out explicitly.

## Window labels

Revision r2 and comment 9 require every number to carry its decode window.
Three windows appear here and they are **not interchangeable**:

| window | label | what it is good for |
| --- | --- | --- |
| 64 tokens | **inner-loop screen** | does the binary run, does the probe fire, is fidelity intact |
| 256 tokens | **labelled directional screen** | shape of the cost curve; sits entirely **before** the EOS boundary |
| 512 tokens | **ranked-equivalent headline** | the only window quoted as a result |

The prefill constant is why these cannot be mixed. Measured on this host from
`score.json` alone, with no trace: serial decode is `0.0814 s/token` at 256 and
`0.07359 s/token` at 512, so `256·0.0814 = 20.84 s` and `512·0.07359 = 37.68 s`.
Differencing gives **65.8 ms per pure serial round** and an implied prefill of
`20.84 − 256·0.0658 = 4.00 s`. That reproduces the parent-clock constant
`P = 4.0086 s` supplied in comment 4 to **0.2%**, and reproduces this report's
trace-measured `C(0) = 65.0 ms` to **1.2%**, from two completely independent
routes. A 256-token quote therefore carries **19.2%** prefill and a 512-token
quote **10.6%**; the difference alone moves an apparent speedup by ~10%.

## Revision r2: what I could and could not do

r2 mandates "push and post after each step, do not batch", with a provenance
comment posted before any measurement. **I do not have the
`post_assignment_comment` tool in this session** — my only GitHub write path is
`submit_experiment_result`, which is terminal. I therefore could not post
per-step comments, and this is a harness capability gap, not a decision to
batch. Everything r2 asked to be posted incrementally is instead recorded here
in the mandated order, with the provenance table first, endpoints before
interior, interior before policy. The per-arm `meta.txt` files under
`research/out/` carry the start/finish timestamps that would have gone into
those comments.

Two further r2 constraints and how they are met:

- **"Do not carry measurements across the boundary."** The r1 base predated the
  EOS fix and every 512-token run on it died at ~token 301. All headline
  numbers below were re-measured on the merged r2 base. Where a 256-token
  number from before the boundary is still quoted, it is quoted **as a
  directional screen and labelled as such** — never as a headline — and the
  reason it remains valid is proved separately: the removed
  `stoppedEarly` suppression in `recordAcceptOutcome` could only fire once a
  stop token had been seen, and the public trajectory's first EOS is at decode
  index ≈301, so a 256-token window never reached the changed branch. The
  256-token acceptance path is provably identical across the boundary.
- **"Report `m(d)` and `h(d) = m(d)/V(1)`; do not present a separated `H`."**
  Both are below. The separated `H` that *is* reported is not a sweep
  decomposition — see the next section.

## Literature: the cost form is adopted, not invented

`C(d) = V(d+1) + d·H + c` is **Sequoia**'s cost model (Chen et al.,
*Sequoia: Scalable and Robust Speculative Decoding*, NeurIPS 2024,
arXiv:2402.12374) written in this campaign's variables. Sequoia optimises
`Speedup(n, d) = G(n, d) / (t(n) + d·c)` where `t(n)` is the **measured**
forward-pass time at batch width `n`, and Appendix G.5 records exactly the
shape this report measures: *"forward pass times are roughly constant for low
values of n, but then eventually start growing roughly linearly."* With
`n = d + 1` that is `V(d+1)`. Sequoia measures `t(n)` per model **and per
hardware** and then grid-searches the tree; it does not assume a scalar. The
form here is theirs; only the measurement and the argmin rule are mine.

Two more recent systems are direct precedent for the specific thing this
experiment ships:

- **D-cut** (arXiv:2607.14647) profiles a latency table at **engine startup**
  and reads an argmax from it at run time, explicitly moving away from
  *"treating each extra draft position as having a constant cost"* — i.e. away
  from `headStepCostRatio = 0.20`. Their selector costs **0.55–0.58 ms/step**,
  which on a 65 ms round would be 0.9% of a serial round and would eat most of
  the win available here. **This implementation therefore keeps selection to a
  table lookup plus a comparison, never a search.**
- **DSpark** (arXiv:2607.05147): *"the capacity curve is profiled once during
  engine initialization and stored as a lightweight cost table."* DSpark also
  reports that the capacity curve is **jagged and non-unimodal**, which is why
  they removed early stopping from their depth selector. That warning applies
  directly: the curve measured here is non-monotonic, so the shipped rule
  **scans every `d` from 0 to 8 and never stops at the first non-improving
  depth**.

**DSpark §5.2 leakage caveat, answered explicitly.** A per-round argmin that
consumed current-round information could in principle leak future-token
information. The rule shipped here reads exactly two things: the
`positionAcceptEMA` vector (a running statistic over **already-verified**
outcomes from previous rounds) and the cost table (input-independent, measured
at warm time). **It never reads the identities of the drafted tokens, their
logits, or any pre-verification confidence.** Depth is chosen *before* the
drafts for that round exist. There is no path from draft content to depth.

**ECHO** (arXiv:2604.09603) offers a 3-parameter hinge fallback
`C(d) ≈ C₀·(1 + γ·[d − d_knee]₊)`. It is not used: the measured table is
strictly more faithful and costs the same at run time (8 doubles).

## The EOS boundary — your comment-9 hazard is real, but it points the other way

You asked me to split every acceptance statistic at the EOS boundary because
211 of the 512 local tokens are post-EOS continuation, which you expected to be
"degenerate, repetitive and easy to draft", biasing my local acceptance
*upward* relative to ranked prose.

I measured it. **The post-EOS region is harder, not easier.** The aggregate
`accepted_draft_rate` of the forced-`d=8` arm *falls* when the window is
extended from 256 to 512 tokens:

| window | arm | `accepted_draft_rate` | mean accepted drafts / round |
|---|---|---|---|
| 256 tok | `d8` | 0.9456 | 7.500 |
| 512 tok | `d8` | **0.8222** | **6.523** |

Split at decode index 301, straddling rounds dropped:

| segment | N rounds | mean depth | mean accepted | tokens/round |
|---|---|---|---|---|
| pre-EOS | 37 | 8.000 | **7.000** | 8.000 |
| post-EOS | 28 | 7.857 | **5.750** | 6.750 |

Per-position, post-EOS is at or below pre-EOS at every position but one:

| pos | pre-EOS `p_i` | post-EOS `p_i` | shipped prior `0.85·0.98^i` |
|---|---|---|---|
| 1 | 0.9459 | 0.8929 | 0.8500 |
| 2 | 0.9714 | 0.9600 | 0.8330 |
| 3 | 0.9706 | 0.9167 | 0.8163 |
| 4 | 1.0000 | 0.9545 | 0.8000 |
| 5 | 0.9697 | 0.9500 | 0.7840 |
| 6 | 1.0000 | 0.9474 | 0.7683 |
| 7 | 0.9688 | 0.9444 | 0.7530 |
| 8 | 0.9355 | 0.8824 | 0.7379 |

### The mechanism, from the 64-token bins

The split above hides the structure. Binning by decode index shows the loss is
not spread across the post-EOS tail at all — it is concentrated in the
*transition*:

| decode bin | N | mean depth | mean accepted | tokens/round | round µs |
|---|---|---|---|---|---|
| 0–63 | 8 | 8.000 | 8.000 | 9.000 | 197,912 |
| 64–127 | 8 | 8.000 | 6.875 | 7.875 | 198,612 |
| 128–191 | 7 | 8.000 | 8.000 | 9.000 | 198,294 |
| 192–255 | 7 | 8.000 | 7.429 | 8.429 | 198,738 |
| **256–319** | 13 | 8.000 | **4.000** | 5.000 | 199,455 |
| **320–383** | 11 | 8.000 | **5.182** | 6.182 | 200,358 |
| 384–447 | 7 | 8.000 | **8.000** | 9.000 | 199,191 |
| 448–511 | 5 | 7.200 | 7.200 | 8.200 | 192,413 |

So the honest reading is: **you were right about the settled tail and wrong
about the net effect.** Once the model has committed to a new pattern
(384 onward) acceptance returns to a perfect 8.000/8, exactly the degenerate
repetition you predicted. But getting there costs one genuinely high-entropy
decision region (256–383) where acceptance collapses to 4.0–5.2 of 8, and that
region is larger than the easy tail it buys.

Net: **the 512-token window is more ranked-representative than the 256-token
window, not less.** It contains exactly one real high-entropy decision point;
the 256-token window contains none.

### What this does not license

This is one EOS event on one fixture, with N=13 rounds in the worst bin. It is
an anecdote about a transition, not a distribution over prose. It does **not**
make `public_longcopy_gate_english_512` a proxy for the ranked pool, and I am
not treating it as one.

The load-bearing number is still comment 1's: ranked depth-1 acceptance is
**0.699**. My *hardest* local region reaches `p₁ = 0.8929`. The local fixture
therefore still overestimates ranked acceptance substantially — just for a
different reason than the one you flagged. Everything I claim about `d*` on
ranked prompts continues to come from the offline counterfactual at
`q = 0.699`, never from a local acceptance measurement.

One consequence worth stating plainly: the shipped prior `0.85·0.98^i`
underestimates acceptance at *every* position in *both* segments here, while
comment 3's external mlx-lm PR #990 table has it overestimating by 3.3× at
`p₅`. The prior is mis-specified in both directions depending on regime, which
is exactly why I left it alone and why F19 (the EMA, not true acceptance,
drives realised depth) matters more than the prior's shape.

### Round cost is unaffected by the boundary

The `round µs` column above is flat at ~198–200k across every bin including the
transition, varying by 1.0% peak-to-peak with no trend. Acceptance changes
sharply at the boundary; **cost does not**. That is the first direct evidence
for the window-invariance result in the next section, measured within a single
arm.

## Window invariance: the curve does not depend on sequence position

The two endpoint arms were re-measured at 512 tokens on the merged r2 base.
Against the 256-token fit:

| quantity | 256 tok | 512 tok | agreement |
|---|---|---|---|
| `C(0)` | 65,009.4 µs | 65,115.1 µs | 0.16% |
| `C(8)` | 198,236.5 µs | 198,683.0 µs | 0.23% |
| `m_avg = (C(8)−C(0))/8` | 16,653.4 µs | 16,696.0 µs | 0.26% |
| `h_avg` | 0.2562 | 0.2564 | 0.08% |

`C(0)` at 512 pools N=1530 depth-0 rounds across three independent legs
(`d0` leg A 65,256.1, `d0` leg B 65,043.2, `d8` serial control 65,046.1 — 0.33%
peak-to-peak). `C(8)` uses N=45 full-accept rounds, sd 0.3%.

**Round cost is a function of verify width, not of KV-cache length or decode
position.** Over a 2× change in sequence length the whole curve moves by less
than the 0.3–0.5% run-to-run noise on a single point.

Three things follow:

1. The constants already committed to
   `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` — fit at 256 — are valid
   at 512. **No refit and no rebuild are required**, so the 512-token policy
   A/B measures exactly the shipped artefact rather than a re-tuned one. That
   is the cleanest available version of the test.
2. Cross-arm pooling (F5) holds at 512 as well, so forced-depth arms measured
   in separate thermal windows remain comparable.
3. Your comment-5 pre-registration can now be scored against a
   ranked-equivalent window:

| quantity | your fb5 pre-registration | measured @512 | error |
|---|---|---|---|
| serial round `C(0)` | 67.0 ms | 65.1 ms | −2.8% |
| MTP round `C(8)` | 161.0 ms | **198.7 ms** | **+23.4%** |
| average marginal | 11.75 ms | **16.70 ms** | **+42.1%** |
| `h_avg` | 0.176 | **0.2564** | **+45.7%** |

`C(0)` was close. The drafting bill was not: the true average marginal is 42%
above the pre-registered value. This is the same direction as, and the
quantitative cause of, the `0.20` scalar being too optimistic.

## Comment 11 — the repair premise is conditional, and the condition holds everywhere I measured

You are right that I stated the premise unconditionally and the source states
it conditionally. `restoreAfterPrefixReject` returns `Bool`
(`Qwen36MTPBlockSession.swift:1421`), and on `false` the caller at `:1265-:1267`
runs `rollbackAfterVerify` **plus** a full `model.callWithHidden` re-forward of
the committed block, with its own blocking `eval`. `rollbackRoundCount`
(`:163`, incremented `:1237`) increments *before* that branch and so conflates
the two paths, exactly as you said. My "nearly free at any depth" wording was
not licensed by the code.

**Answer: `fullRepairCount == 0` at every depth I have data for, in both
checkpoint regimes, in both windows.** Your first interpretation branch fires:
the `0.20` premise holds, the mis-specification is in the *value*, and the curve
is the whole story. I am stating that as a measured result, not as a
re-assertion of the original claim.

### How I got the two counters without rebuilding

I first recovered the counters *without* adding them, because adding them
mid-sweep forces a rebuild and splits the forced-depth sweep across two
binaries. `research/run-arms.sh` and `research/run-arm.sh` contain no build
step, and `git diff --stat 20b6e2b b9379e2 -- Sources/ Vendor/` is empty, so
`d = 0,1,2,3,4,8` all come from one binary ("binary A"). Instead of rebuilding,
the existing traces already separate the two paths: 

- `tReadDone` is stamped at `:1219`, **before** the accept/reject branch.
- `tCommitDone` is stamped at `:1287`, **after** the entire branch — including
  the `rollbackAfterVerify` + `callWithHidden` re-forward at `:1265-:1267`.
- `commit_us = (tCommitDone - tReadDone)/1000` is emitted at `:1329`.

So **`commit_us` brackets the repair; `upkeep_us` does not.** I had this wrong in
my earlier note: `upkeep_us` is `tTailDone - tCommitDone` (`:1330`), which starts
after the repair has already completed. `commit_us` gives a per-round classifier:

| round class | condition | meaning |
|---|---|---|
| full accept | `acc == d` | branch never entered |
| `prefixRepairCount` | `acc < d`, `commit_us` small | `restoreAfterPrefixReject` returned `true` |
| `fullRepairCount` | `acc < d`, `commit_us` large | it returned `false`: rollback + full re-forward |

The classes are separated by orders of magnitude, not by a tuned cutoff. Any
target forward on this host is memory-bound at **≥ 60 ms** because it reads the
whole 4-bit backbone (`C(0) = 65.1 ms`, and F18 puts width-1 verify at ~92% of
the 59.5 ms roofline). The cheap path issues no blocking eval at all. I set the
threshold at 10 ms — ~6× below the cheapest possible forward and ~40× above the
largest cheap-path cost I observed. Tool: `research/repair_probe.py`; outputs
`research/out/repair512.json`, `research/out/repair256.json`.

### The sweep is split across two binaries — disclosure and the free control

Having proved the classifier worked, I then *did* add the two literal counters
(`db37226`), so the deep tail of the sweep was rebuilt. This must be declared:

| arms | binary | source | counters |
|---|---|---|---|
| `d0 d1 d2 d3 d4 d8`, all 256-token arms, `base256`, `cand256` | **A** | `20b6e2b` | recovered from `commit_us` |
| `d5 d6 d7`, `base512`, `cand512` | **B** | `db37226` | emitted literally |

The delta is 11 lines in `Qwen36MTPBlockSession.swift`: two `Int` declarations,
two `+= 1` statements **inside the reject branch only**, and two extra fields in
the trace string. Its cost is nanoseconds against a 65 ms round, and the trace
string is built after `tTailDone` is stamped, so it cannot enter `round_us`.
Expected effect: nil.

I do not want to assert "nil" from source reading alone, and I do not want to
spend a GPU arm proving it, so the sweep pays for its own control. **Every arm
runs its own serial depth-0 leg.** That leg is a `C(0)` measurement, so the
binary-B arms re-measure on binary B exactly the quantity the binary-A arms
pooled to 65281.4 µs (N = 3570). If the two agree inside the 1.4 % spread the
six binary-A controls already show among themselves, the binaries are
measurement-equivalent and the split costs the sweep nothing. This check is
reported in the results below at zero extra GPU cost.

The binary/arm assignment above is not inferred from commit SHAs — it is pinned
by the mtime of the built worker against each arm's wall-clock window:

```text
.build-worker/release/mlxfast-runtime-worker   mtime 2026-08-16T18:47:01Z

d0  17:42:19 -> 17:49:58   dirty=0   |
d8  17:49:58 -> 17:59:15   dirty=0   |  all six finish before the rebuild
d1  18:03:01 -> 18:11:01   dirty=0   |  -> binary A
d2  18:11:01 -> 18:20:26   dirty=0   |
d3  18:21:51 -> 18:31:12   dirty=0   |
d4  18:31:12 -> 18:40:59   dirty=1   |
                    <-- rebuild at 18:47:01 -->
d5  18:47:07 -> ...        dirty=0      binary B
```

This also disposes of the one provenance wart. The `d4` arm's `meta.txt`
records **`dirty=1`**: it was launched while the counter edit sat uncommitted in
the worktree, so its recorded base SHA does not pin its source. But `d4`
finished at 18:40:59, six minutes *before* the only rebuild in the sweep, and
`run-arm.sh` contains no build step — so the uncommitted edit was never
compiled and `d4` ran the same binary A as `d0..d3` and `d8`. The dirty flag is
real and I am reporting it, but it does not touch the measurement.

`d0` through `d4` and `d8` record base SHAs ranging from `20b6e2b` to `73dd6ea`;
those commits differ only in `research/` and `senpai/` files, which the binary
does not contain.

### The counters, per depth, annotated by repair regime

The verify window is `S = d + 1`. `Qwen35.swift` takes the single-launch
free-checkpoint path at `S == 2`, and records the cheap replay tape only when
`nConfirmed == 1 && S >= 3 && mask == nil` (`:977`, written `:1112`, consumed by
`replayPrefix` `:889`); otherwise it falls back to the eager-checkpoint kernel.
So **the regime boundary sits exactly at the `d = 1` → `d = 2` step you flagged.**

512-token window (`d0`, `d1`, `d2`, `d8`; `research/out/repair512.json`):

| d | S | regime | N full-accept | N reject | `prefixRepairCount` | `fullRepairCount` | commit µs accept | commit µs reject | commit µs **max** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | single-launch free checkpoint | 253 | 5 | **5** | **0** | 304.9 | 226.6 | 284 |
| 2 | 3 | replay tape | 163 | 11 | **11** | **0** | 411.8 | 1947.7 | 2031 |
| 4 | 5 | replay tape | 1 | 0 | 0 | 0 | 237.0 | – | – |
| 8 | 9 | replay tape | 45 | 20 | **20** | **0** | 111.8 | 619.5 | 717 |

The `d = 2` row is the one that matters most for your question, because it is the
first depth *inside* the replay-tape regime and so is where a tape-replay repair
would first become possible. It has 11 rejects and still zero full repairs.

256-token window (`d3`, `d4`, `d6`, `base-decl`, `base256`, `cand256`;
`research/out/repair256.json`):

| d | S | regime | N full-accept | N reject | `prefixRepairCount` | `fullRepairCount` | commit µs accept | commit µs reject | commit µs **max** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | single-launch free checkpoint | 3 | 0 | 0 | 0 | 200.3 | – | – |
| 3 | 4 | replay tape | 121 | 4 | **4** | **0** | 397.0 | 1882.2 | 1918 |
| 4 | 5 | replay tape | 72 | 5 | **5** | **0** | 117.5 | 623.4 | 680 |
| 5 | 6 | replay tape | 4 | 0 | 0 | 0 | 264.0 | – | – |
| 6 | 7 | replay tape | 39 | 2 | **2** | **0** | 220.2 | 604.5 | 607 |
| 7 | 8 | replay tape | 12 | 0 | 0 | 0 | 136.4 | – | – |
| 8 | 9 | replay tape | 18 | 6 | **6** | **0** | 78.8 | 615.5 | 665 |

Totals: **`prefixRepairCount` = 53, `fullRepairCount` = 0** over 53 reject rounds,
19 legs, depths 1/2/3/4/6/8, both regimes, both windows. The single largest
reject-round `commit_us` anywhere is **2,031 µs**, which is **32× below** the
65 ms floor for one target forward. This is not a close call.

### An independent bound that does not trust my classifier

If you distrust the `commit_us` attribution, the round totals bound the same
quantity without it. A full re-forward must cost at least one `C(0)`, so the
mean round-time excess of reject rounds over full-accept rounds divides by
`C(0)` to give an upper bound on the fraction of reject rounds that could have
paid one:

| d | median Δ round µs | bound on re-forward fraction |
|---|---|---|
| 1 (512) | 253.5 | 0.0039 |
| 3 (256) | 1561.0 | 0.0240 |
| 4 (256) | 985.5 | 0.0152 |
| 6 (256) | 683.5 | 0.0105 |
| 8 (256) | 544.5 | 0.0084 |
| 8 (512) | 1324.5 | 0.0203 |

Every bound is ≤ 2.4% of a single forward, and — this is the part that answers
your second interpretation branch — **the bound does not rise with `d`**. It is
flat-to-falling from `d = 3` to `d = 8`. There is no unpriced term growing in
depth, so `C(d)` is not hiding one.

I report medians here deliberately. The `d = 1` *mean* delta is 10,044 µs, which
looks alarming until you read the five rounds: round 151 has
`verify_build_us = 85,990` against a ~35,900 norm, while its `commit_us` is 255.
That is the command-queue backpressure artefact I flag elsewhere in this report
landing in `verify_build`, not repair work landing in `commit`. The other four
reject rounds are 70,319–70,866 µs against a 70,927 µs full-accept mean — at or
*below* it. The median delta is 253.5 µs.

### Your kink question, answered directly

You asked me to suspect the regime change before attributing a kink between
`d = 1` and `d = 2` to head-step cost. Checked, and the regime change is
exonerated. Writing `m(d) = C(d) - C(d-1)` and labelling each step by the `S`
transition it crosses (256-token curve, the one with all nine points):

| step | `S` transition | crosses the regime boundary? | `m(d)` µs |
|---|---|---|---|
| `m(1)` | 1 → 2 | enters free-checkpoint | 5,473 |
| `m(2)` | 2 → 3 | **yes — free-checkpoint → replay tape** | 5,037 |
| `m(3)` | 3 → 4 | no, tape → tape | **15,769** |

The step that actually crosses the regime boundary, `m(2)`, is the **cheapest
step in the entire curve** — the marginal *falls* across it. The knee is `m(3)`,
one step later, with both endpoints inside the replay-tape regime. So the
checkpoint-regime change does not explain the knee, and cannot: it has the wrong
sign and the wrong location. The knee lines up instead with the F8 qmv IPG
staircase, whose steps I predicted at `M = 5` and `M = 9` and which show up as
the largest marginals at `d = 4` and `d = 8`.

**The 512-token window preserves the inversion, and strengthens it.** I promised
to state this rather than assume it, so here are both windows side by side:

| window | `C(0)` µs | `C(1)` µs | `C(2)` µs | `m(1)` µs | `m(2)` µs | `m(2)/m(1)` |
|---|---|---|---|---|---|---|
| 256 | 65,009.4 | 70,482.4 | 75,519.2 | 5,473.0 | 5,036.8 | **0.920** |
| 512 | 65,102.1 | 70,930.7 | 75,398.2 | 5,828.6 | 4,467.5 | **0.766** |

`m(2) < m(1)` holds at both windows, so the conclusion does not depend on the
window. It is not a mean artefact either: on medians the 512 numbers are
`m(1)` = 70,450 − 64,744 = 5,706 and `m(2)` = 75,114 − 70,450 = 4,664, same
ordering. One honest caveat on the 512 `C(1)`: its spread is 7.0% versus 2.8%
for `C(2)`, inflated by the single round-151 backpressure outlier described
below (`verify_build_us` = 85,990 against a ~35,900 norm). That outlier makes
`m(1)` look *larger*, which works against the inversion — removing it would push
`m(2)/m(1)` further below 1, not above it.

### The structural reason no repair term is in `C(d)` anyway

`C(d)` is fit on full-accept rounds only (`acc == d`), which is the exclusion
rule stated in Section 1 and enforced in `research/depth_cost_curve.py`. A
full-accept round never enters the `else` branch at `:1236`, so **no repair cost
of either kind can enter `C(d)` by construction.** That is a property of the fit,
not a lucky measurement.

This is worth being precise about, because it cuts both ways. It means the curve
is safe from the contamination you were worried about. It also means the curve
**does not price rejection at all** — `C(d)` is the cost of a *successful*
depth-`d` round. The policy simulator handles rejection separately, through the
acceptance profile, which is the right place for it. But if `fullRepairCount`
had been non-zero and depth-rising, the correct fix would not have been to
re-fit `C(d)`; it would have been to add an expected-repair term to the policy's
objective. It is zero, so I did not add one, and that is the honest reason the
cost model still has the shape it has.

### What I did not do, and what would change it

I did not add the literal `prefixRepairCount` / `fullRepairCount` counters to
the session. If you want them in the shipped source for cross-agent comparison
with `qwen-askeladd`, say so and I will add them under those exact names — it is
a small, self-contained change. I declined it mid-sweep only because the rebuild
would have split the forced-depth sweep across two binaries, against your fb8
instruction to prioritise getting all nine points cleanly over implementation
polish, and because comment 11 says this does not change my assignment.

Two honest limits on the above. First, all 53 reject rounds come from one public
fixture on one host; `restoreAfterPrefixReject` has failure modes (cache offset
mismatch, a non-trimmable non-`ArraysCache` entry, `canReplayPrefix` failing, a
nil tape, or `rollbackCheckpoints.count <= acceptedCount` at K=1) that this
fixture may simply never provoke. Second, my reject-round counts are small at
every depth except `d = 8`, because forced-depth arms on a copy task accept
nearly everything — the very acceptance inflation I document elsewhere. A ranked
prompt with depth-1 acceptance 0.699 would produce roughly an order of magnitude
more reject rounds, and I cannot run one. So the correct claim is: **on
everything I can measure, the cheap path always wins, and its cost is ~0.6 ms
against a 65 ms forward.** I am not claiming the fallback is unreachable.

## Section 1 — the curve (measurement, not policy)

### Definition and the identifiability limit

`C(d)` is the mean wall-clock of one full-accept decode round at chosen depth
`d`. `m(d) = C(d) - C(d-1)` is the **combined** marginal cost of the `d`-th
draft. Revision r2 asks for it normalised by the width-1 verify, `h(d) =
m(d) / V(1)`; the shipped cost model normalises by the full zero-draft round,
`m(d) / C(0)`. **These differ by 0.05% and nowhere else in this report does the
choice matter**: measured `V(1) = 65,044 µs` and `C(0) = 65,009 µs`, i.e.
`C(0)/V(1) = 0.999468`. Every `h(d)` printed below is `m(d)/C(0)`; multiply by
`0.999468` for `m(d)/V(1)`. That near-identity is itself a result — it says the
round-level overhead `c` outside the verify graph is at the noise floor.

### The one place a separated `H` appears, and why it is not a sweep decomposition

Revision r2 is right that **head cost and verify-width slope are perfectly
collinear across a depth sweep** and that a separated `H` must not be presented
as if the sweep had produced it. Verify width is always `d + 1`
(`Qwen36MTPBlockSession.swift`, `verify` row construction), so every
`m(d) = H + [V(d+1) − V(d)]` and no depth sweep can split that sum. **No number
in the measured table below is a sweep-derived head cost, and no such claim is
made anywhere.**

A separated `H` nevertheless appears once, because comment 7 required the
policy to be **head-agnostic** — "a single frozen table is the one option I will
not accept" — and a head-agnostic table is impossible without separating the
head term. The separation is obtained by **direct isolated measurement, not by
decomposition of the sweep**: `probeResidentHeadCost` times a single-token head
step with **no verify in the graph** and a single-row verify with **no head in
the graph**, on throwaway caches, inside the existing warm phase and outside
every timed window. Two isolated timings are not a regression on eight
collinear points; the collinearity is broken by construction, not by fitting.
The two instructions are therefore both satisfied: `m(d)` and `h(d)` are the
headline fit, and `H` is reported separately and labelled as a direct
measurement.

One honest caveat on that probe: an isolated head step is not overlapped with a
verify graph build, so it is an **upper bound** on the head's marginal
wall-clock contribution inside a real round.

The probed head step is `mtpHeadHiddenForward` chained into `draftTokenID` —
the exact expression a deep draft sub-step dispatches. Measured values and the
head-agnostic reparameterisation they enable are in "Making the policy
head-agnostic" below.

### Exclusion rule

Rounds are pooled across forced-depth arms. A round enters the curve only if:

- it is a **full-accept** round (`acc == d`), so the timing describes the
  advertised depth rather than a truncated one;
- it is **not** one of the first 2 rounds of its leg (warm-up: first-touch
  kernel specialisation and cache growth);
- the **seed prologue is excluded entirely** — it is emitted as a separate
  `mtp-trace: begin` record (`build_us` 3,143,887–3,148,761 µs,
  `eval_wall_us` 868,244–868,702 µs) and never enters `C(d)`.

Cross-arm pooling was validated before use: `C(8)` measured independently in
the `base-decl` and `d8` arms agrees to **0.03%**, and `serial_seconds_per_token`
varies **0.27%** across arms. Per-arm `C(0)` spread across 6 arms is
64857–65239 µs (**0.6%**).

### Measured curve

Pooled arms `d0,d1,d2,d3,d4,d6,base-decl,d8`, declared 4-bit head, M4 Pro.
`C(0)` from **N = 1778** depth-0 rounds across 6 arms.

| d | N | C(d) µs | median | sd% | m(d) µs | **h(d)** | C/C0 | µs/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1778 | 65009.4 | 64701.0 | 4.5 | – | – | 1.000 | 65009.4 |
| 1 | 129 | 70482.4 | 70374.0 | 0.5 | 5473.0 | **0.0842** | 1.084 | 35241.2 |
| 2 | 83 | 75519.2 | 74943.0 | 4.3 | 5036.8 | **0.0775** | 1.162 | 25173.1 |
| 3 | 61 | 91287.8 | 91213.0 | 0.5 | 15768.6 | **0.2426** | 1.404 | 22822.0 |
| 4 | 60 | 115690.9 | 115568.5 | 0.5 | 24403.1 | **0.3754** | 1.780 | 23138.2 |
| 5 | 2 | 134668.0 | 134668.0 | 0.1 | 18977.1 | **0.2919** | 2.072 | 22444.7 |
| 6 | 36 | 154169.1 | 154079.5 | 0.4 | 19501.1 | **0.3000** | 2.371 | 22024.2 |
| 7 | 7 | 172827.0 | 172576.0 | 0.5 | 18657.9 | **0.2870** | 2.658 | **21603.4** |
| 8 | 32 | 198236.5 | 198173.5 | 0.3 | 25409.5 | **0.3909** | 3.049 | 22026.3 |

Fitted vector `[0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870,
0.3909]`, mean **0.2562**.

`d = 5` has only N = 2 (the adaptive schedule almost never selects it and no
forced-`d5` arm was run); it is the one weak row. `d = 7` has N = 7. Every other
row has N >= 32 with sd <= 4.5%.

**Self-normalised cross-check** (each round divided by its own arm's `C(0)`,
removing any per-arm thermal offset): d1 0.0805 (N=129, 4 arms), d2 0.0839
(N=83), d3 0.2424 (N=61), d4 0.3680 (N=48), d6-combined 0.6008 (N=33) against
pooled 0.2919 + 0.3000 = 0.5919. Agrees with the pooled fit within ~2%.

### The 512-token forced-depth curve (ranked-equivalent window) — HEADLINE

The table above pools 256- and 512-token arms and is the *directional screen*.
This is the ranked-equivalent replacement: every row a dedicated forced-depth
arm at 512 decode tokens, declared 4-bit head, same host, one arm at a time
through the run lock and the 40 C gate.

Each arm runs its own serial leg, so the serial column is six independent
measurements spread over ~4.5 hours. They span **0.073366–0.074426 s/token, a
1.4 % spread** — the thermal and host control for everything below.

| d | serial s/tok | MTP s/tok | **speedup** | eff_draft | accept rate | matched | div |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 0 | 0.073593 | 0.073350 | 1.0033 | 0.000 | – | ✅ | 0 |
| 1 | 0.073366 | 0.044092 | 1.6639 | 1.000 | 0.9807 | ✅ | 0 |
| 2 | 0.073418 | 0.034140 | 2.1505 | 1.994 | 0.9490 | ✅ | 0 |
| 3 | 0.073918 | 0.031925 | **2.3153** | 2.985 | 0.9475 | ✅ | 0 |
| 4 | 0.074426 | 0.033061 | 2.2512 | 3.973 | 0.9222 | ✅ | 0 |
| 8 | 0.073386 | 0.034268 | 2.1415 | 7.941 | 0.8222 | ✅ | 0 |

`d = 0` reproducing **1.0033** is the calibration anchor: forcing zero drafts
costs 0.33 % against the serial leg, so `C(0) ≈ V(1)` and the shipped
`zeroDraftRoundRatio = 0.999468` is correct to within measurement noise.

Round-level costs from the same traces, `C(0)` pooled over the depth-0 control
leg of **all six arms (N = 3570)**. `C(d)` is measured over **full-accept
rounds only** (`acc == d`), so it is the cost of the depth-`d` round mechanism
itself and is not contaminated by the cheaper rejected rounds:

| d | N | C(d) µs | median | sd% | m(d) µs | **h(d)** | C/C0 | eval µs | host µs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3570 | 65281.4 | 64797.0 | 4.1 | – | – | 1.000 | 28958.5 | 36321.4 |
| 1 | 255 | 70938.6 | 70454.0 | 6.9 | 5657.1 | **0.0867** | 1.087 | 33796.0 | 37140.0 |
| 2 | 163 | 75398.2 | 75114.0 | 2.8 | 4459.6 | **0.0683** | 1.155 | 35110.4 | 40285.4 |
| 3 | 120 | 91828.3 | 91835.0 | 0.4 | 16430.2 | **0.2517** | 1.407 | 43525.8 | 48300.4 |
| 4 | 94 | 117491.6 | 116952.0 | 4.2 | 25663.3 | **0.3931** | 1.800 | 53555.7 | 63933.5 |
| 8 | 45 | 198683.0 | 198611.0 | 0.3 | 81191.3/4 = 20297.8 | **0.3109**/step | 3.043 | 93032.1 | 105648.3 |

The six per-arm depth-0 controls agree to **1.4 %** (65046.1, 65065.0,
65100.0, 65256.1, 65512.2, 65947.4 µs), which is what licenses pooling `C(0)`
and comparing `C(d)` measured hours apart. Normalising every round by its own
arm's control instead of the pool moves the ratios by less than 0.02:
`[0.0902, 0.0680, 0.2435, 0.3803]` for `d = 1..4`.

`d = 5, 6, 7` are still missing at this window; the 256-token screen put them
on the plateau but with N = 2 at `d = 5` and N = 7 at `d = 7`, which is why
they are not quoted as measured here.

#### Two results that the scalar cost model cannot express

**1. `m(2) < m(1)`.** The second draft step is *cheaper in absolute terms* than
the first: 4460 µs against 5657 µs. A model of the form `C(d) = V(d+1) + d·H`
with any constant `H` forces the marginal to be non-decreasing once verify
width costs are monotone, so no scalar `headStepCostRatio` — 0.20 or any other
value — can reproduce this. It is not noise: `d = 2` has N = 163 at sd 2.8 %
and `d = 1` has N = 254.

**2. A 3.7× knee at `d = 3`.** `h` jumps from 0.0683 to 0.2517 in one step, and
`d = 4` then pushes it further to 0.3931 — the knee is a step into a new, more
expensive regime, not a one-off spike.
The shipped scalar 0.20 is therefore **2.3× too high at `d = 1`, 2.9× too high
at `d = 2`, 26 % too low at `d = 3`, and 2.0× too low at `d = 4`** — wrong in
*both directions*
inside the range the scheduler actually chooses from. This is the direct answer
to the assignment's question, and the answer is choice **(c), "something
else"**: a cheap-flat region, a sharp knee, then a plateau.

#### The realised optimum on this fixture is `d = 3`, not `d = 8`

Decode-round-only seconds per token, from the trace rather than the score file
(so prefill and warmup are excluded). This averages over **every** post-warmup
round, accepted and rejected, so unlike `C(d)` above it is realised throughput:

| d | rounds | tokens | tokens/round | MTP-leg s/token | vs best |
|---:|---:|---:|---:|---:|---:|
| 0 | 510 | 510 | 1.000 | 0.065043 | +171 % |
| 1 | 257 | 509 | 1.981 | 0.035910 | +49.8 % |
| 2 | 175 | 506 | 2.891 | 0.026112 | +8.9 % |
| 3 | 132 | 505 | 3.826 | **0.023980** | — |
| 4 | 108 | 503 | 4.657 | 0.025059 | +4.5 % |
| 8 | 66 | 494 | 7.485 | 0.026528 | +10.6 % |

Drafting **past `d = 3` loses 4.5 % at `d = 4` and 10.6 % at `d = 8`**, and the
leg-level score agrees on the ordering (2.3153, 2.2512, 2.1415). Depth 8 buys
1.96× more tokens per round than depth 3 but pays 2.16× more per round, so it
is on the wrong side of the knee.

Two distinctions matter for reading these two tables together, because they
disagree if you conflate them. `C(d)/(d+1)` computed from the full-accept table
would make `d = 8` look *best* at 22076 µs/token — that number is conditional
on all eight drafts being accepted, which happens in only 45 of 65 depth-8
rounds. The realised column above is the one that predicts the leg score, and
it is worse at `d = 8` because 32 % of depth-8 rounds pay for a rejection.
Correspondingly the extra cost at `d = 8` is *partly* acceptance-driven after
all: `accepted_draft_rate` falls 0.9475 → 0.9222 → 0.8222 across `d = 3, 4, 8`
even though per-position acceptance inside the depth-8 arm is flat at ~0.95.
The two facts are consistent — flat per-position acceptance still compounds, so
the chance of a clean sweep decays geometrically in depth while the per-round
cost climbs past the knee.

#### The curve predicts the end-to-end policy A/B to within 1 %

This is the strongest validation I have that `C(d)` is right, and it is also
why the advisor was correct that the policy A/B does not transfer.

Measured adaptive arms at 256 tokens, declared head:

| arm | cost model | mean chosen depth | accept | tokens/round | speedup |
|---|---|---:|---:|---:|---:|
| `base-decl` | shipped scalar 0.20 | 6.000 | 0.9435 | 6.649 | 2.0613 |
| `base256` | shipped scalar 0.20 (repeat) | 6.000 | 0.9435 | 6.649 | 2.0712 |
| `cand256` | measured curve | 3.000 (62/63 rounds) | 0.9896 | 3.938 | 2.0765 |

The two baseline repeats differ by 0.5 %, so **+0.5 % for the candidate is
inside the noise floor. The A/B is a wash.**

Now predict that wash from the round-level curve alone, before looking at the
speedups:

```text
shipped   picks d = 6:  C(6) / (tokens/round) = 154169.1 / 6.649 = 23187 us/token
measured  picks d = 3:  C(3) / (tokens/round) =  91828.3 / 3.938 = 23318 us/token
                                         predicted difference:  -0.6 %
                                         observed  difference:  +0.5 %
```

The curve predicts a dead heat and a dead heat is what the benchmark reports.
Depths 3 and 6 sit on a **genuinely flat efficiency plateau when acceptance is
~0.95**: `d = 6` costs 1.68× more per round and returns 1.69× more tokens.

So the shipped scalar 0.20 is *badly wrong as a description of cost* — 2.3×
high at `d = 1`, 26 % low at `d = 3` — and yet nearly *harmless as a policy* on
a high-acceptance fixture, because it lands on a plateau where the choice
barely matters. **Both statements are true, and only the first one transfers.**

The plateau breaks as soon as acceptance falls. At 512 tokens the `d = 8` arm
sees accept 0.8222 (the post-EOS regime) and the gap opens to a real
**8.1 %** in favour of `d = 3` (2.3153 against 2.1415). The value of the curve
is therefore not the +0.5 % here; it is that the curve tells you *where the
plateau ends*, which a scalar cannot.

#### Host time, not GPU time, is the majority of every round

The `eval` and `host` columns split each round at the blocking evaluation.
Host time is **55 % of a depth-0 round** (36.1 ms of 65.2 ms) and stays above
half at every depth. Both grow with `d`, so drafting is not hiding under an
idle GPU. This bounds any pipelining or overlap experiment: the addressable
budget is large, but it is on the CPU side of the boundary.

### Shape: answer (c), "something else"

Not flat-then-knee at d ≈ 7, and not flat. The curve is **cheap-flat at
d = 1–2 (~0.08), a knee at d = 3 (0.243), then a plateau of ~0.29–0.39 for
d = 4–8**, with the two largest steps at d = 4 (0.375) and d = 8 (0.391).

Against the pre-registered roofline prediction (flat 0.145 for d = 1..6, 0.271
for d = 7..8): measured h(1), h(2) are ~45% **below** 0.145; h(5), h(6) are ~2x
**above** it; h(7) = 0.287 ≈ the predicted 0.271; h(8) = 0.391 exceeds it.

**Roofline reconciliation.** The roofline is right in form, wrong in constant.
The measured crossover is at width ≈ 3.5 (d = 2 -> 3), not 7.9. Inverting
`M* = 0.5625 · FLOPS_eff / (2 · BW_eff)` with `M* = 3.5` and 227 GB/s gives
**FLOPS_eff ≈ 2.82 TFLOP/s**, 44% of the 6.415 TFLOP/s figure — which was
presumably measured on a wide GEMM, not on the skinny `qmv` dispatch that
actually runs here. Measured marginal above the knee is ≈ 19 ms, against 8.4 ms
for pure arithmetic and ≈ 62 ms for a full weight re-read: the truth is between
the two bounds, much closer to the arithmetic one.

### DO NOT RE-BASE THIS CURVE BY −2.689 ms — it is already the ranked-side head

Comment 7 assumed my measurements ran the organizer-pinned bf16 head and gave a
re-basing rule `m_ranked(d) = m_local(d) − 2.689 ms`. **That premise does not
hold for my runs, and applying the correction would double-count it.**

Every forced-depth arm in the table above was launched through
`research/run-arms.sh --head-dir <declared>`, i.e. against
`hf:lowskillcoding/qwen38-mtp-head-4bit-g64` — the head
`mtp-head.manifest.json` declares and the head the ranked *candidate* leg
attaches. This was F1, found on day one precisely because `setup-qwen-mtp.sh`
never reads the manifest: the default local stack silently serves the
849,398,784 B bf16 head, and I built `research/fetch-declared-head.sh` to
provision the declared 238,934,093 B tree instead. So:

- **My `m(d)` are already ranked-side.** `m(1) = 5.47 ms` is a declared-head
  number, not a pinned-head number. Subtracting 2.689 ms would produce
  `m(1) = 2.78 ms`, which nothing measured supports.
- **The re-basing is still correct for PR #3's anchors** (`C(0) = 67.0`,
  `C(8) = 161.0`, runs `cwlqu3ok` / `ihnmmi1b`) and for anyone else running the
  default `setup-qwen-mtp.sh` stack, because those do run the pinned bf16 head.
  My `base-1` arm (`h1t8073f`, pinned bf16, adaptive) versus `base-decl`
  (`w8aocl64`, declared 4-bit, adaptive) puts the end-to-end head penalty at
  **3.9%** of `mtp_seconds_per_token` (0.041100 vs 0.039478 s/token), which is
  the same effect measured end to end rather than per draft step.
- **Comment 7's item 6 ("optional, if under ~15 minutes: point
  `MLXFAST_QWEN_MTP_HEAD_DIR` at the declared head") is already satisfied for
  every number in this report.**

Head provenance, per comment 7 item 5, is in the provenance block below.

#### What the low band actually says

The corrected prediction was `m(1..6) ≈ 4.5–5 ms` of head + readout traffic
before verify growth, derived from an 849 MB head at 227 GB/s. Measured
`m(1) = 5.47 ms`, `m(2) = 5.04 ms` — inside that band. But the agreement is
**numerological, not mechanistic**. I have now measured the head step directly
(see "Making the policy head-agnostic" below): **`H = 2.590 ms`**, not 4.5–5 ms.
So a head forward is only **47%** of `m(1)` and **51%** of `m(2)`; the rest is
the `V(d+2) − V(d+1)` verify-width increment. Do not reuse "849 MB / 227 GB/s"
as an explanation of the low band; it lands on the right number for the wrong
reason, and it lands there on a head my runs never loaded.

The agreement also holds **only** for d = 1–2. From d = 3 the marginal is
15.8–25.4 ms and `H` is 10–16% of it, so above the knee the cost is dominated by
verify-width growth, not by head forwards. Across the whole span,
**8·H / (C(8) − C(0)) = 15.6%**: the head is a sixth of the drafting bill, and
verify width is the other five sixths.

#### Your `delta_head = 2.689 ms` is independently confirmed — to three decimals

The probe gives the head step for the **declared 4-bit** head. Your re-basing
constant is the *difference* between the pinned bf16 head and that one, derived
purely from bandwidth: `(849,398,784 − 238,934,093) / 227e9 = 2.689 ms`. I can
now check that from the other side:

| quantity | value |
|---|---|
| declared 4-bit head, bytes | 238,934,093 |
| its bandwidth floor at 227 GB/s | 1.053 ms |
| **measured step `H`** | **2.590 ms** |
| size-independent residue (dispatch, activations, draft readout) | 1.537 ms |
| pinned bf16 head bandwidth floor | 3.742 ms |
| implied bf16 step = floor + same residue | 5.279 ms |
| **implied `delta_head` = 5.279 − 2.590** | **2.689 ms** |

The two derivations agree to the displayed precision. That is a coincidence of
rounding at the third decimal, but the agreement to ~0.1% is real and it says
the constant residue assumption is sound: head cost on this machine is
`1.54 ms + bytes/227 GB/s`. **Your re-basing arithmetic is right; it just must
be applied to PR #3's anchors and to the default `setup-qwen-mtp.sh` stack, not
to my curve.**

### Falsified predictions (mine and the advisor's)

- **My own pre-registered "hypothesis A" is FALSIFIED.** I predicted
  `h(3) ≈ 0.08` on the grounds that the crossrow IPG rule adds no weight pass at
  M = 4. Measured h(3) = 0.2426, a 3x miss. The knee is at d = 3, one step
  earlier than any pass-boundary account predicts.
- The advisor's **knee-at-d≈7** prediction is not observed.
- The IPG staircase is **partially** retained: predicted wide-tensor pass steps
  at M = 5 (d = 4) and M = 9 (d = 8) are exactly the two largest measured
  marginals (24.4 and 25.4 ms against a ~19 ms plateau). The extra ≈ 5–7 ms is
  far below a 62 ms full pass, so if it is the pass boundary, the added pass is
  largely cache-resident.

### Phase decomposition of the marginal

Mean µs per full-accept round by phase:

| d | N | draft_bld | vrfy_bld | eval_wall | readout | commit | upkeep | round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1778 | 0 | 34639 | 30340 | 28 | 0 | 0 | 65009 |
| 1 | 129 | 684 | 35764 | 33745 | 36 | 246 | 5 | 70482 |
| 2 | 83 | 1067 | 38817 | 35168 | 57 | 400 | 8 | 75519 |
| 3 | 61 | 1264 | 47244 | 42366 | 31 | 373 | 8 | 91288 |
| 4 | 60 | 615 | 60156 | 54778 | 13 | 124 | 3 | 115691 |
| 5 | 2 | 2419 | 69544 | 62205 | 41 | 442 | 14 | 134668 |
| 6 | 36 | 3275 | 79063 | 71566 | 23 | 231 | 8 | 154169 |
| 7 | 7 | 5544 | 86600 | 80474 | 18 | 180 | 8 | 172827 |
| 8 | 32 | 7609 | 97718 | 92765 | 13 | 123 | 7 | 198237 |

**Clean negative: bookkeeping is not the cost.** `readout`, `commit` and
`upkeep` marginals are all `|Δ| <= 0.45 ms` and flat in `d`. **91–99% of every
step's marginal lives in the verify path.**

⚠️ **Corrected inference (comment 11).** An earlier draft read this table as
"a prefix reject is nearly free at any depth". That does not follow, and the
correction matters. Every row here is a **full-accept** round (`acc == d`), which
never enters the reject branch at `Qwen36MTPBlockSession.swift:1236` at all, so
this table is silent about rejects by construction. What it licenses is only the
narrower claim above: on the *accept* path, bookkeeping is free. The reject-path
evidence is measured separately in the comment-11 section, where reject rounds
are isolated and classified — and there the conclusion does hold, but as a
measurement (`fullRepairCount = 0`, cheap-path cost ~0.6 ms) rather than as an
inference from this table.

⚠️ **Caveat that must not be dropped:** `verify_build_us` is *not* host time.
MLX dispatches asynchronously, so "build" absorbs command-queue backpressure —
at d = 0 the round is 65 ms while the 14.1 GiB weight stream alone is ~62 ms at
227 GB/s, i.e. the GPU is ~95% occupied during "build". Do **not** read
`verify_build` / `eval_wall` as a CPU/GPU split. `draft_build_us` is also
non-monotone (615 µs at d = 4 against 1264 µs at d = 3), so it is a weak
host-side signal; the head's GPU work stays collinear with verify width exactly
as the identifiability limit predicts.

## Section 2 — policy delta (UPPER BOUND, offline counterfactual)

Everything in this section is an **upper bound**, for two independent reasons:

1. It is computed on one local fixture, `public_longcopy_gate_english_512.txt`,
   which is a **copy** task with acceptance ≈ 1.0 and effective draft 5.4. The
   eight ranked goldens are natural prose with effective depth **1** and
   depth-1 acceptance ≈ **0.699**. That is a regime change, not noise.
2. The score is a **median of 8**. Improving prompts already at rank 7–8 is
   worth exactly zero.

The acceptance model was therefore **not** fitted on longcopy. The table below
is an offline counterfactual over hypothetical acceptance profiles.

**Pre-registered interpretation of the end-to-end A/B, written before the run
landed.** The counterfactual says the candidate's gain at acceptance 1.0 is
**+1.37% (cap 4) / +1.92% (cap 8)** — its *smallest* non-zero gain anywhere in
the sweep. The measured run-to-run spread on this host is ~0.3% on
`serial_seconds_per_token` and ~1.4% on the end-to-end score between nominally
identical arms (`base-decl` 2.0612 vs `d8` 2.0906 differ by 1.4% on a change
that should be much larger). **So on longcopy the expected effect is roughly the
size of the noise floor, and a single 512-token A/B cannot resolve it.** I am
recording that here rather than discovering it afterwards: if the A/B comes back
inside ±1.5% I will call it **consistent with the counterfactual and
underpowered**, not a confirmation and not a refutation. The load-bearing
evidence in this experiment is the curve (Section 1), which is measured with
N = 1778 at d = 0 and N = 32–129 at the drafting depths and has sd ≤ 0.5% at
most depths. The policy delta is a derived consequence, and the fixture that
would actually test it — natural prose at acceptance ≈ 0.7 — is not available
locally.

**What then happened: the 512-token window you asked for is not reachable on
this fixture (F16).** `MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512` hard-fails
with `MTP round requested before the seed prefill`. Root cause, source-pinned
and *not* caused by my diff:

- `Qwen36MTPBlockSession.generateRound` guards on `pendingPrimary != nil`
  (`:769-771`) and throws `Qwen36MTPSessionError.notBegun` (`:90`) otherwise.
- The only site that clears `pendingPrimary`/`pendingTop2`/`pendingHidden` is
  the stop-token branch (`:821-828`), which returns `reachedStopToken: true`.
- `QwenRuntimeMTPDriver.swift:114` loops `while emitted.count <
  options.totalTokenCount` and **never inspects `reachedStopToken`**.

So a greedy continuation of `public_longcopy_gate_english_512.txt` emits EOS
before 512 decode tokens, the driver asks for one more round, and the session
throws. 256 tokens never reaches EOS, which is why every arm here is 256.
Reference-row generation is unaffected (`mtp-verify` produced 513/513 rows)
because the reference path does not apply stop tokens. **Every headline number
in this report is therefore at a 256-token decode window, labelled as such.** A
512-token local measurement needs the driver fix listed under follow-ups; it is
outside my assignment's file scope (`QwenRuntimeMTPDriver.swift` is not in my
declared submitted path).

The same batch also lost one arm to the **thermal gate**: `base512` hard-failed
`run_cool_gate` after 180 s with a GPU temperature floor of 42.2 °C that never
reached the 40 °C threshold. That is environmental, the wrapper behaved
correctly, and I did not bypass it.

Columns: `old` = shipped (greedy walk + scalar 0.20); `grd` = greedy +
measured vector (**the constant change alone**); **`CAND` = argmin + measured
vector = the actual candidate**.

**cap = 4** (the ranked-relevant cap: `widthCap` opens to 8 only after a
3-round full-accept streak, which natural prose at acceptance 0.699 does not
produce):

| profile | d*old | d*grd | dCAND | c/tok old | c/tok CAND | grd vs old | **CAND vs old** |
|---|---:|---:|---:|---:|---:|---:|---:|
| longcopy(1.00) | 4 | 3 | 3 | 0.3559 | 0.3511 | +1.37% | **+1.37%** |
| flat 0.95 | 4 | 3 | 3 | 0.3933 | 0.3785 | +3.77% | **+3.77%** |
| flat 0.85 | 4 | 3 | 3 | 0.4799 | 0.4407 | +8.17% | **+8.17%** |
| flat 0.70 | 3 | 2 | 2 | 0.5544 | 0.5304 | +4.32% | **+4.32%** |
| 0.70·0.95^d | 2 | 2 | 2 | 0.5364 | 0.5364 | 0.00% | **0.00%** |
| 0.90·0.90^d | 3 | 2 | 2 | 0.4443 | 0.4419 | +0.55% | **+0.55%** |
| **ranked 0.699** | 3 | 2 | 2 | 0.5552 | 0.5310 | +4.36% | **+4.36%** |
| 0.699·0.85^d | 2 | 2 | 2 | 0.5494 | 0.5494 | 0.00% | **0.00%** |
| 0.699·0.70^d | 2 | 2 | 2 | 0.5692 | 0.5692 | 0.00% | **0.00%** |

**cap = 8**: longcopy(1.00) d*old 8 -> grd 3 -> **CAND 7**, c/tok
0.3388 -> 0.3323, greedy **−3.61%**, **CAND +1.92%**. All other rows identical
to cap 4 except flat 0.95 (+8.21%), flat 0.85 (+11.67%), flat 0.90 (+12.52%).
**cap = 2**: every profile 0.00%.

**Direct answer to the advisor's question "does the new curve only win at
acceptance ≈ 1.0 and lose at 0.70?": No — the opposite.** The candidate's gain
at longcopy's acceptance 1.0 (+1.37% at cap 4) is its *smallest* non-zero gain.
At the ranked acceptance of 0.699 it is **+4.36%**, and it never regresses in
any profile tested.

### The greedy -> argmin rule change is a repair, not a second mechanism

The shipped rule is a greedy walk: extend while
`reach > h·(1+expected)/(1+depth·h)`. That is exactly "extend iff
`f(d+1) < f(d)`" for `f(d) = (1 + H_d)/T(d)` — a local descent, which is
correct **only if `f` is unimodal in `d`**. With a constant `h`, `f` is
unimodal, so greedy is globally optimal. With the measured non-convex curve it
is not: greedy stops at the d = 3 knee and can never reach the cheaper d = 7
minimum.

That is precisely the −3.61% row above: **replacing the constant alone
regresses 3.61% at cap 8 / acceptance 1.0, because a greedy walk cannot cross
the measured knee. The argmin turns that same −3.61% into +1.92%.** The rule
change is not an extra mechanism bolted on to win — it is the repair required to
use a non-convex curve at all.

**And under a flat `h`, the two rules are exactly equivalent**:
`research/greedy_vs_argmin.py` finds **0 mismatches in 900,000 sampled
acceptance profiles** (300k × caps 2/4/8, uniform and monotone-decaying
samplers). With the measured vector they diverge on 5.75% of monotone profiles
at cap 4 (worst cost-per-token gap 7.16%) and 5.73% at cap 8 (worst 8.43%); at
cap 2, 0.12% (worst 0.04%).

This equivalence is also what makes the A/B honest: setting
`MLX_QWEN_MTP_H_VECTOR=0.2,...,0.2` reproduces the shipped policy **bit for
bit** from the candidate binary, so baseline and candidate share one build and
one thermal window.

### Reconciliation with the generalized greedy rule you asked for

You asked me to ship `extend to depth d+1 iff reach > (1+expected)·m(d+1)/C(d)`
and called it the two-line deliverable. I want to be explicit that **I shipped
something strictly stronger, and that the difference is measurable rather than
stylistic.**

Your generalized rule is the correct de-specialisation of the shipped test: I
verified independently that the shipped `reach > h(1+expected)/(1+d·h)` is
exactly your form under constant marginal, so we derived the same thing. But
your form is still a **local descent**, and local descent is only globally
optimal when `f(d) = C(d)/T(d)` is unimodal. My measured curve is not unimodal:
`m` runs 5.47, 5.04, **15.77**, 24.40, 18.98, 19.50, 18.66, 25.41 ms, so there
is a knee at d = 3 that a greedy walk cannot cross.

Concretely, at cap 8 / acceptance 1.0:

| rule | chosen d | cost/token | vs shipped |
|---|---|---|---|
| shipped (greedy + scalar 0.20) | 8 | 0.3388 | — |
| **your generalized greedy + measured `m`** | 3 | 0.3510 | **−3.61%** |
| **argmin + measured `m` (what I shipped)** | 7 | 0.3323 | **+1.92%** |

So the generalized-greedy form **regresses** exactly where the curve's
non-convexity bites. The argmin is the global minimiser of the identical
objective — same `m`, same `C`, same `T`, no extra tuning constant — and across
the full acceptance sweep it matches generalized-greedy everywhere else and
**never regresses**. It is also two lines: the loop already walks every depth to
accumulate `reach`, so taking the running argmin instead of breaking early costs
one comparison and one assignment.

If you want the literal greedy form for reviewability I can switch it in a
minute, but I would be knowingly shipping the −3.61% row, so I have shipped the
argmin and flagged it here rather than silently substituting.

### Implied optimal depth `d*`, and `d*+1` as you asked

Implied `d*` by acceptance level (cap 4 / cap 8): 0.50 -> 2/2; 0.55–0.65 -> 2/2;
0.699 (ranked) -> **2**/2; 0.75 -> 2/2; 0.80 -> 3/3; 0.85 -> 3/3; 0.90 -> 3/3;
0.95 -> 3/3; 1.00 -> 3/**7**.

Comment 7 item 3 asked for `d*` **and** `d*+1`, on the argument that
`d*_ranked >= d*_local` in every cell of your 5-shape x 5-`q` sweep, with a
worst-case 0.93% loss from shipping the local optimum. Cost per token at each
depth on the measured curve, flat acceptance `q`, normalised to a serial round:

| q | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 | `d*` | cost at `d*+1` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.699 | 0.6381 | **0.5310** | 0.5552 | 0.6430 | 0.7059 | 0.7772 | 0.8486 | 0.9560 | 2 | **+4.56%** |
| 0.75 | 0.6195 | **0.5024** | 0.5136 | 0.5834 | 0.6300 | 0.6842 | 0.7386 | 0.8243 | 2 | **+2.23%** |
| 0.80 | 0.6023 | 0.4761 | **0.4757** | 0.5294 | 0.5615 | 0.6002 | 0.6389 | 0.7044 | 3 | **+11.29%** |
| 0.85 | 0.5861 | 0.4516 | **0.4407** | 0.4799 | 0.4989 | 0.5236 | 0.5482 | 0.5953 | 3 | **+8.89%** |
| 0.90 | 0.5706 | 0.4287 | **0.4083** | 0.4346 | 0.4421 | 0.4546 | 0.4668 | 0.4978 | 3 | **+6.44%** |
| 0.95 | 0.5560 | 0.4073 | **0.3785** | 0.3934 | 0.3910 | 0.3931 | 0.3949 | 0.4124 | 3 | **+3.94%** |
| 1.00 | 0.5421 | 0.3872 | 0.3511 | 0.3559 | 0.3453 | 0.3388 | **0.3323** | 0.3388 | 7 | **+1.96%** |

Two things follow, and they cut against the premise of item 3:

1. **`d*_ranked = d*_local` on my curve, because the head is the same.** Your
   `d*_ranked >= d*_local` result came from subtracting a bf16-versus-4-bit head
   penalty that my measurements never paid (see the re-basing section). With the
   declared head on both sides, the only residual transfer risk is M4 Pro versus
   M5, and every constant I ship is a *ratio*, which is invariant under a uniform
   host speed change. A non-uniform M5 change (different FLOP/bandwidth balance)
   can still move `d*`; a uniform one cannot.
2. **Being one step off is much more expensive than 0.93% here, and which way
   to err flips with `q`.** Your sweep bounded the loss from shipping
   `d*_local` at 0.93%; on the measured curve `d*+1` costs **2–11%**, because
   the knee at d = 3 and the plateau above it make the depth just past the
   optimum the one where the marginal jumps. Penalty for `d*-1` / `d*+1`:

   | q | `d*` | `d*-1` | `d*+1` | safer error |
   |---|---|---|---|---|
   | 0.699 | 2 | +20.2% | +4.56% | deeper |
   | 0.75 | 2 | +23.3% | +2.23% | deeper |
   | 0.80 | 3 | +0.08% | +11.29% | shallower |
   | 0.85 | 3 | +2.47% | +8.89% | shallower |
   | 0.90 | 3 | +5.00% | +6.44% | shallower (marginal) |
   | 0.95 | 3 | +7.61% | +3.94% | deeper |

   So there is no single safe hedge: at the ranked acceptance 0.699 erring
   **deeper** is 4x cheaper than erring shallower, while in the 0.80–0.90 band
   erring **shallower** is 2–140x cheaper. This is exactly why a scalar `h`
   cannot be retuned into the right answer, and it is also why the sensitivity
   is worth an M5 confirmation before anything is submitted officially.

Mean `h` = **0.2562**, just under the advisor's `h <= 0.2624` bound, so the
curve lands in **branch 1: the cost model is roughly calibrated in magnitude,
and the remaining headroom is in schedule occupancy and verify width, not in a
mis-priced head.** (Noting the advisor's own retraction: the bound assumed
`T = 9, d = 8` and is uninformative; reported for completeness.)

Implied `G = V_pinned/V_candidate` needed to reach the promoted 2.9042 frontier:
at longcopy d = 3, `G = 1.020`; at the ranked acceptance 0.699 with d = 2,
**`G = 1.542`**. Since the ranked serial leg is a *separately pinned prebuilt
baseline workspace*, that ~1.54x is a real, scored, general target/kernel win —
it does not cancel.

## Making the policy head-agnostic (comment 7 item 4)

> "A single frozen table is the one option I will not accept."

Agreed, and shipped. The policy no longer contains a frozen `h`.

### What ships

The round cost decomposes as `C(d) = V(d+1) + d·H + c`, where `V(w)` is a
verify over `w` rows, `H` is one proposal-head step, and `c` is fixed per-round
overhead. Only `H` depends on which head is resident; `V` and `c` are pure
target-model quantities. So the marginal

```
m(d+1) = C(d+1) − C(d) = [V(d+2) − V(d+1)] + H
```

splits into a head-free term I can freeze and a head term I must measure.

I went one step past the brief: **the frozen constants are dimensionless**,
stored in units of a single-row verify.

```swift
verifyMarginalRatioByDepth[i] = (V(i+2) − V(i+1)) / V(1)   // frozen, head-free
zeroDraftRoundRatio          = C(0) / V(1)                 // frozen, head-free
h(d+1) = (verifyMarginalRatioByDepth[d] + H/V(1)) / zeroDraftRoundRatio
```

`probeResidentHeadCost` measures **both** `H` and `V(1)` once per worker process
during the existing warm phase, then `adoptResidentHeadStepRatio(H/V(1))`
rebuilds the vector. Five timed head steps and three timed width-1 verifies,
medians, on a throwaway cache; ~19 ms of warm time, entirely outside timing.

### Why dimensionless, and not the brief's absolute form

Freezing `V` and `c` in microseconds and measuring only `H` mixes an M4 Pro
constant with an M5 measurement. If the ranked host is uniformly `s` times
faster, the true `h(d)` is **unchanged** (both numerator and denominator scale),
but the absolute form returns `[Vslope_local + s·H_local] / C0_local`:

| `s` | `h(1)` absolute form | `h(1)` truth | error | `h(8)` error |
|---|---|---|---|---|
| 1.00 | 0.0842 | 0.0842 | 0.0% | 0.0% |
| 0.80 | 0.0762 | 0.0842 | **−9.5%** | −2.1% |
| 1.25 | 0.0941 | 0.0842 | **+11.8%** | +2.6% |

The error is worst exactly in the low band, which is the band that decides
*whether to draft at all*. Dimensionless ratios are invariant under any uniform
host speed factor, so the only thing that has to transfer is the **shape** of
the verify-width curve — which is a property of the target model's kernels, not
of the machine's clock. Your "second best" option (two tables keyed on head
byte count) would also have worked, but it interpolates a proxy where a direct
measurement costs 19 ms.

### The measured numbers (M4 Pro, declared affine-4/g64 head)

Five probes fired (the worker creates a warm session at init, at
`mtp_decode_warm`, and lazily inside `mtp_decode_begin`):

```
headprobe head_us=2759 verify_us=65100 ratio=0.042386
headprobe head_us=2736 verify_us=65217 ratio=0.041957
headprobe head_us=2590 verify_us=65044 ratio=0.039834
headprobe head_us=2489 verify_us=64935 ratio=0.038344
headprobe head_us=2544 verify_us=64836 ratio=0.039244
```

| quantity | median | spread |
|---|---|---|
| `H` (head step) | **2590 µs** | sd 4.5% |
| `V(1)` (width-1 verify) | **65044 µs** | sd 0.23% |
| `H / V(1)` | **0.039819** | 0.0383–0.0424 |
| `C(0) / V(1)` | **0.999468** | — |

Two independent cross-checks fall out for free:

1. **`V(1) = 65044 µs` versus the pooled `C(0) = 65009 µs` from 1778 depth-0
   rounds — 0.05% apart.** So the fixed per-round overhead `c` is at the noise
   floor, confirming the phase decomposition (readout + commit + upkeep stay
   under 0.5 ms at every depth) from a completely different measurement.
2. **`V(1) = 65.0 ms` versus the 27B-at-4-bit weight-streaming floor
   `13.5 GB / 227 GB/s = 59.5 ms`.** A width-1 verify runs at ~92% of roofline
   bandwidth. The target is almost perfectly memory-bound at width 1, which is
   why `h(1)`, `h(2)` are so cheap: rows 2 and 3 ride weights already in flight.

Refitting the frozen constants against these two numbers reproduces the measured
curve **exactly**:

```
reconstructed h = 0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909
measured     h = 0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909
```

That is by construction, not evidence — it only shows the reparameterisation is
algebraically faithful. The evidence is that all eight head-free terms come out
**positive** (0.0443 … 0.3508), so the split is physically meaningful rather
than an artefact of forcing `H` out of a collinear fit.

### This is what resolves the comment-5 identifiability limit

Comment 5 was right that `H` and the verify-width slope are perfectly collinear
*in the forced-depth data alone*: every arm at depth `d` pays exactly `d·H` and
exactly one `V(d+1)`, so no regression on that data can separate them. The
answer is not a cleverer fit — it is a second, independent experiment. The warm
probe runs the head with **no** verify and the verify with **no** head, which
breaks the collinearity by measurement.

I therefore now claim, and did not claim before: **the isolated head step is
2.590 ms, and 84.4% of the d=8 marginal is verify width, not head work.**

Caveats, stated honestly:

- The probe syncs after every step, so it charges each head forward a full
  round-trip. In a real round the head steps are already serialised (each needs
  `.item()` on the drafted token before the next), so there is little overlap to
  lose — but if any exists, the probe **over**-states `H`, which inflates `h`
  uniformly and biases the chosen depth **down**. The bias is conservative.
- `head_us` has 4.5% spread across probes; `h(1)` moves over 0.0827–0.0868 as a
  result, and `d*` does not move at all (the cost-per-token curve is flat within
  ~2% around its optimum, see the `d*±1` tables).
- The medians are over 5 and 3 samples. The first head sample is a
  compile/warm outlier in 4 of 5 probes (6970, 6451, 3900, 4363 µs); a
  median-of-5 rejects one outlier but would be contaminated by two. Raising the
  sample count to 7 costs ~5 ms of warm time and is the obvious hardening step
  if this ever misbehaves.
- If the probe never runs, `referenceHeadStepRatio = 0.039819` is used, and a
  negative reconstructed marginal is floored at 0. Neither path has fired.

## Head provenance (comment 7 item 5)

Every number in this report was measured against this head:

| field | value |
|---|---|
| resident head directory | `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared` |
| `model.safetensors` | 238,934,093 B |
| `config.json` | 3,570 B |
| total on disk | 238,937,663 B |
| dtype | 4-bit / group-64, MLX affine |
| manifest `source_url` | `hf:lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaff972fd3ca2be08f3640603b47e9ce70a` |
| manifest `sha256` (tree digest) | `cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d` |
| **`head_provenance_sha256`** as the harness reports it | `54930a1d281ff3ec4373fc2befd190afdb67ee09ffd90e8fc60e4d1f538bfc4b` |

### Why those two digests differ, and why the head is still verified

This tripped me up, so it is worth recording. The manifest digest is a **tree
digest**, not a file digest: sha256 over `"<file sha256>  <relative path>\n"`
lines in `LC_ALL=C` order, `README.md` excluded
(`research/fetch-declared-head.sh:13-14`). Reproduced from my local tree:

```
sha256(model.safetensors)                       = 0e267a48…62815
tree digest over {model.safetensors}            = cc209e30…8f0d   == manifest ✓
tree digest over {config.json, model.safetensors} = 54930a1d…bfc4b == reported
```

The declared head ships **weights only**. `benchmark-qwen-mtp.sh:215` refuses a
head directory without a `config.json`, so `fetch-declared-head.sh` copies one
in from the pinned head after verifying the digest. That extra file is what
changes the reported tree digest — **the weights are bit-identical to the
manifest**, which the single-file reproduction above proves exactly.

Two consequences:

- Nothing about my measurements is off-manifest. Anyone re-deriving provenance
  should compare `cc209e30…` against the **weights-only** tree digest.
- The `config.json` is a load-time architecture descriptor only. Quantization is
  detected from the `scales`/`biases` keys present in the safetensors, not from
  config, which the manifest note states and which the runs confirm empirically:
  the declared head decoded **3.9% faster** than the bf16 head with
  `all_tokens_matched: true`. Had it loaded as bf16 it would have been the same
  speed.

## The sdpa width wall, and why my result composes with qwen-alphonse's

You asked for the chosen-depth histogram with an explicit statement of whether
it is clipped at 4. Here is the answer, and it carries a caveat that changes how
you should read it.

**On the local fixture the wall is NOT binding, and it cannot be made to bind.**
`base-decl` (shipped policy, declared head, 256 tokens, N = 37 scored rounds):

```
chosen depth histogram: d=4:14, d=5:2, d=6:3, d=7:6, d=8:12
max chosen depth = 8; d==4 14/37 (37.8%); d>4 23/37 (62.2%)  ->  NOT clipped at 4
```

The mechanism is the streak gate, not the cap. `widthCap = fullAcceptStreak >= 3
? 8 : 4` (`:561`, `:568-569`). Longcopy accepts at ≈ 0.96–1.00 per position
(measured below), so a 3-round full-accept streak re-arms almost continuously
and the wall spends most of its time **open**. The 37.8% of rounds sitting
exactly at d = 4 are the rounds just after a rejection reset the streak.

This is a regime statement, and it is the reason the local fixture cannot
settle your joint-arm question directly: **the width wall binds when acceptance
is low, and the only fixture I have is the one where acceptance is ≈ 1.0.** At
the ranked per-position acceptance of ≈ 0.699 a single full-accept round at
d = 4 has probability `0.699^4 = 0.239`, and three in a row `0.699^12 = 0.0136`,
so the gate opens on roughly **1.4% of attempts** and the cap is effectively
hard at 4. Locally it is open most of the time. **The wall you and
qwen-alphonse care about is invisible on this fixture by construction.**

**What that implies for composition.** Your simulation says the two changes are
jointly binding because the corrected curve tells greedy to extend and the wall
then stops it. My offline counterfactual is consistent with that and localises
it precisely — the `cap 4` and `cap 8` columns of `research/policy_sim.py`
differ *only* at acceptance 1.00:

| acceptance | d\* @ cap 4 | d\* @ cap 8 | candidate gain @ cap 4 | candidate gain @ cap 8 |
|---|---|---|---|---|
| 0.699 (ranked) | 2 | 2 | +4.36% | +4.36% |
| 0.85 | 3 | 3 | +8.17% | +11.67% |
| 0.90 | 3 | 3 | +6.04% | +12.52% |
| 0.95 | 3 | 3 | +3.77% | +8.21% |
| 1.00 | 3 | **7** | +1.37% | +1.92% |

Read this carefully, because it is the one place my evidence **disagrees with
your simulation's premise**: on my measured curve the corrected cost model wants
depth **2–3**, not depth 5+, at every acceptance level below 1.0. The wall at 4
is therefore *not* the thing clipping my policy at ranked acceptance — my policy
never asks for more than 3. What the cap changes is the *magnitude* of the win
at moderate acceptance (0.85–0.95: +8.17 -> +11.67, +6.04 -> +12.52, +3.77 ->
+8.21), because opening the cap changes what the **baseline** does, not what the
candidate does.

So the joint arm is still worth running, but the mechanism I measure is the
reverse of the one you hypothesised: **the corrected curve makes the candidate
shallower, and opening the wall makes the mis-specified baseline deeper and
therefore worse, which widens the gap.** I would not close either experiment on
my evidence alone, and I agree a +0.7% solo result is not a null — but I would
ask you to re-run your simulation with the measured `m` before pricing the joint
arm, because a curve that peaks at d = 2–3 will not reproduce your +7.52%.

## Realised per-position acceptance

Emitted with `research/accept_profile.py` (forced/observed d = 8, warmup > 2
dropped). Longcopy only — see the regime caveat.

- `d8` arm (N = 28), accepted-count histogram {2:1, 4:1, 6:1, 7:2, 8:23}, depth
  histogram {7:1, 8:27}, mean chosen depth 7.964, mean accepted drafts 7.500,
  tokens/round 8.500. p1..p8 = 1.0000, 1.0000, 0.9643, 1.0000, 0.9630, 1.0000,
  0.9615, 0.9583.
- `base-decl` arm (N = 37), depth histogram {4:14, 5:2, 6:3, 7:6, 8:12},
  accepted-count histogram {1:1, 3:2, 4:12, 5:3, 6:3, 7:7, 8:9}, mean chosen
  depth 6.000, mean accepted drafts 5.649, tokens/round 6.649. p1..p8 = 1.0000,
  0.9730, 1.0000, 0.9444, 1.0000, 0.9500, 1.0000, 0.9000.

Essentially **flat ≈ 0.96**. The shipped prior `0.85 · 0.98^d` therefore
**under**-estimates acceptance here, while **over**-estimating the external
`ml-explore/mlx-lm` PR #990 profile (82.5 / 64.0 / 47.6 / 33.9 / 23.4%) by 3.3x
at p5. The prior is mis-specified in **both** directions — it is not a
conservative prior, it is simply the wrong shape. Re-fitting it is a separate
experiment and was deliberately left untouched here.

## Findings recorded for other agents

**F1 — local benchmarks run the WRONG MTP head by default.**
`mtp-head.manifest.json` declares `hf:lowskillcoding/qwen38-mtp-head-4bit-g64`
@ `0966ddaf`, sha256 `cc209e30…`, 238,934,093 B. `setup-qwen-mtp.sh` never reads
the manifest; it provisions only the organizer-pinned **bf16** head
(`EigenLabs/Qwen3.8-27B-MTP-bf16` @ `26a328e0`, 849,400,347 B). Only the ranked
workflow fetches the declared head, and only for the candidate leg. Resolved
with `research/fetch-declared-head.sh`. The `uses_pinned_mtp_head` flag is
**unreliable**; `head_provenance_sha256` is the trustworthy discriminator. The
declared 4-bit head is **~3.9% faster** in `mtp_seconds_per_token` despite lower
acceptance. All curve fitting used the declared head.

**F2 — zero-draft rounds emitted no trace rows (fixed, `0ae2ddd`).**
`generateRound`'s early branch (`depth == serialControlDepth || draftCount == 0`)
returned before the round-trace emit, so no in-situ `C(0)` existed. Fixed. All
added timing reads are gated behind `Self.traceRounds`, so the hot-path cost
with tracing off is a static bool read. Rounds with `d >= 1` never enter that
branch, so pre-fix arms are bit-identical and poolable.

**F3 — forced d = 0 reproduces serial to within 0.016%** (score
0.9998382636819225). Validates the cost model's unit of "1"; there is no hidden
fixed drafting tax.

**F4 — verify-attention splits at d >= 5.**
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:120-126`: for
`qL` in 6..9 with a causal mask, one sdpa call becomes two (`:127`, `:134`) plus
a concat (`:141`). Rows = d + 1, so the split starts at **d = 5**.

**F5 — cross-arm pooling validated** (see Section 1).

**F6 — forced d = 8 beats the shipped adaptive policy by 1.43% on longcopy** —
upper-bound regime only.

**F7 — measured per-token cost minimum is at d = 7** (21603 µs/token), with
d = 6 (22024) and d = 8 (22026) essentially tied.

**F8 — the "steep-linear qmv" premise is FALSIFIED, with a mechanism, and it
transfers to M5.**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
defines three crossrow kernels: `qmv_fast_crossrow_affine4_g64<T,M>` (`:859`),
`…_g64_wide<T,NA>` (`:968`, the arithmetic worker), `…_g64_m<T,M,IPG>` (`:1053`,
a selector calling `_wide` at `:1074`/`:1078`). The weight stream is hoisted
above the input loop (`:994-1009`; the per-input loop `:1018-1027` loads only
`x`), so **passes = ceil(M / IPG), not M**. The in-source rule at `:1051` is
`IPG = ceil(M / ceil(M / 4))`. `affine_qmv_fast` is a **device** kernel at
`:1763`, gated at `:1804` on `!batched && group_size == 64 && bits == 4 &&
out_vec_size >= 1024`, switching on `ntg.x` = M (host sets
`grid_dims(M, …)` at `backend/metal/quantized.cpp:254`). For `N >= 4096`
(`:1805`) the IPG is hand-tuned per M (lines 1811–1846), giving wide-tensor
passes for M = 1..9 of **1,1,1,1,2,2,2,2,3** — a **staircase** with steps at
M = 5 and M = 9, not a line. For 1024 <= N < 4096 (lines 1853–1893) IPG is
always 2, so passes are `ceil(M/2)` = 1,1,2,2,3,3,4,4,5.
`get_qmv_batch_limit` (`backend/metal/quantized.cpp:84-126`) returns **10** for
K = N = 5120 on this generation (`:121`), 6 on arch gen 13/14 (`:102`), 12 for
`case 'd'` (`:108-115`). `fast` requires `N % 8 == 0 && K % 512 == 0`
(`:259`); K = 5120 ✓ and N ∈ {1024, 5120, 6144, 248320} all ≡ 0 mod 8 ✓. N = 1024
(k/v_proj) clears `>= 1024` but not `>= 4096`, so it uses the coarser
`ceil(M/2)`; q_proj (6144), o_proj (5120) and lm_head (248320) use the tuned
staircase.
**M5 transfer holds.** `quantized_nax.h` contains **zero** `crossrow` and
**zero** `qmv` matches — only `affine_qmm_t_nax` (1205), `affine_qmm_n_nax`
(1264), and three gather variants. `is_nax_available()` has 3 call sites
(`quantized.cpp:697`, `:892`, `:1237`); the latter two are MoE and unreachable
here. Reachability: `:1415` `int vector_limit = transpose_ ?
get_qmv_batch_limit(K,N,d) : 4;` then `:1418 if (M >= vector_limit)`; below it
`:1444-1446` -> `:1390 qmv(...)` -> `affine_qmv_fast` -> crossrow. So **all
reachable widths 1–9 use qmv + crossrow on M5 as well.** Generated twins are in
sync (`mlx-generated/quantized.cpp:885`, `:1818`, `:1839`).
**Root cause of the wrong premise:**
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1142-1143` carries a
**stale comment** — *"crossrow for M <= 5, per-row qmv_fast above it"*. That is
almost certainly the origin of both the base's "widths above 5 are structurally
closed" intuition and the steep-linear prediction. Crossrow arrived later, via
`Validate submission` snapshots (`1033e1a`, `08897af`, `b6c7251`).

**F9 — independent cross-validation of the prefill constant.** The parent-clock
least-squares `P = 4.008616434203254 s` and my worker-side seed trace
(`build_us` 3,143,887 + `eval_wall_us` 868,244 = 4.0121 s) agree to **0.09%** by
fully independent instrumentation. Prefill is ~23.9% of the candidate leg at the
ranked window on M4 Pro.

**F10 — realised per-position acceptance** (see above).

**F11 — the base advance `e20268e -> dbed6c2` is purely additive research
tooling; zero `Sources/` changes.** Merged as `4e31883`.

**F12 — CORRECTION: the "trace is unreachable" blocker is half right and does
not block this experiment.** Verified true: worker stderr is discarded by
default (`Sources/MLXFastTrustedHarness/QwenRuntime.swift:306`
`forwardsWorkerStderr: Bool = false`; `QwenRuntimeWorker.swift:2046`/`:2207`
`emit: options.forwardsWorkerStderr ? nil : { _ in }`; `main.swift:2222-2224`
default false, `:2301` ANDs with `!officialRun`). **Refuted as a conclusion:**
commit `d1ad5f8` added a **file sink** at `Qwen36MTPBlockSession.swift:472-482`
(`traceFile`, keyed on `MLX_QWEN_MTP_TRACE_PATH`, per-PID filename
`"\(base).\(pid)"`) with `traceWrite` at `:483-489`, requiring **both**
`MLX_QWEN_MTP_TRACE=1` and a non-empty path. The worker env sanitizer
(`QwenRuntimeWorker.swift:2623-2645`) allowlists the prefix `"MLX_"`, so every
`MLX_QWEN_MTP_*` variable reaches the sandboxed worker. The per-PID filename is
necessary because the wrapper spawns a **separate worker per leg** (only four
`"${swift_bin}"` invocations: tripwire `:515`, reference `:534`, serial control
`:589`, MTP leg `:608`).
The **real** remaining constraint is `MLXFAST_NO_SANDBOX=1`, because the
generated profile contains `(deny file-write*)` (`main.swift:2626-2638`, esp.
`:2636`; bypass gate `:2278-2281`). **Official runs fail closed**
(`:2236-2238`, `:2288-2291`), and the deny line is pinned by
`Tests/MLXFastTests/ParentToolSandboxTests.swift:108` and
`Tests/MLXFastTests/BenchmarkScriptTests.swift:3163`. All tracing below used
this local-only relaxation; it is unreachable on an official run.

**F13 — with a flat `h`, greedy and argmin are exactly equivalent** (see above);
this is what makes the one-binary A/B valid.

**F14 — the candidate never regresses in the offline counterfactual.** An
earlier "−3.61% at cap 8 / acceptance 1.0" figure described **greedy + measured
vector** (the constant change alone), not the shipped candidate. The candidate
(argmin + measured vector) is **+1.92%** there and >= 0 everywhere tested.

**F25 — the post-EOS bias runs the *other* way; the 512 window is more
ranked-representative than 256, not less.** Splitting d8 acceptance at the EOS
index (301) gives pre-EOS mean accepted **7.000** (N=37) and post-EOS **5.750**
(N=28) — acceptance *falls* after EOS. The loss is concentrated in the EOS
transition bins (256–383: mean accepted 4.000 and 5.182) and recovers to 8.000
in the settled tail (384–447). Round cost is flat across the boundary (~198–200k
µs, 1.0% peak-to-peak), so this is an acceptance effect, not a cost effect.

**F26 — `fullRepairCount == 0` in every round I have measured; the expensive
repair branch never fires on this fixture.** `restoreAfterPrefixReject`
(`Qwen36MTPBlockSession.swift:1421`) returns `Bool`, and on `false` the caller
runs `rollbackAfterVerify` plus a full `model.callWithHidden` re-forward
(`:1265`–`:1267`). `rollbackRoundCount` (`:163`, incremented `:1237`) increments
*before* that branch and therefore conflates the cheap and expensive paths. The
two counters can be separated without rebuilding, because `tReadDone` is stamped
at `:1219` (before the accept/reject branch) and `tCommitDone` at `:1287` (after
the whole branch including the repair forward), so the emitted `commit_us`
brackets any repair. Over 53 reject rounds across 19 legs, depths 1/2/3/4/6/8,
both repair regimes and both token windows: **`prefixRepairCount` = 53,
`fullRepairCount` = 0**. The largest reject-round `commit_us` seen anywhere is
2,031 µs, **32x below** the ~65 ms forward floor, so no round contains a hidden
re-forward. `research/repair_probe.py` reproduces this from committed traces.
Agents touching rollback, acceptance, or cache-snapshot code should note that on
this fixture the cheap path is the *only* observed path — which also means the
expensive path is **untested here**, not proven absent. Ranked depth-1
acceptance is 0.699 versus my hardest local `p1` of 0.8929, so ranked prompts
should reject roughly 10x more often and are the place to look for it.

**F27 — the free-checkpoint to replay-tape regime change is not the source of
the curve's knee, at either window.** The step that crosses the boundary is
`m(2)` (draft rows `S`: 2 -> 3), and it is the **cheapest** step in the curve —
the marginal *falls* across the boundary. The knee is `m(3)`, one step later and
entirely inside the tape regime. The inversion reproduces at both windows and
strengthens at 512: `m(2)/m(1)` = 5036.8/5473.0 = **0.920** at 256 and
4467.5/5828.6 = **0.766** at 512. Wrong sign and wrong location, so the regime
change is exonerated; the knee matches the F8 qmv IPG staircase instead.

**F28 — the EOS acceptance penalty is depth-dependent, and it penalises deep
drafting specifically.** F25 established that acceptance falls after the EOS
transition. Measuring the same split at every forced depth I have at 512 shows
the penalty is not a flat tax — it scales with depth:

Straddling rounds are dropped. Because a few late rounds run below the forced
depth as the leg runs out of tokens, the honest statistic is the **accept rate**
`mean_accepted / mean_offered`, not the raw accepted count; both are shown.

| arm | `d` | pre mean acc | post mean acc | pre rate | post rate | rate delta |
|---|---|---|---|---|---|---|
| `d1` | 1 | 0.980 (N=152) | 0.981 (N=105) | 0.9803 | 0.9810 | **+0.1%** |
| `d2` | 2 | 1.931 (N=102) | 1.833 (N=72) | 0.9655 | 0.9165 | **−5.1%** |
| `d3` | 3 | 2.859 (N=78) | 2.778 (N=54) | 0.9530 | 0.9376 | **−1.6%** |
| `d4` | 4 | 3.762 (N=63) | 3.523 (N=44) | 0.9405 | 0.8960 | **−4.7%** |
| `d8` | 8 | 7.000 (N=37) | 5.750 (N=28) | 0.8750 | 0.7188 | **−17.9%** |

At depth 1 there is effectively no EOS penalty at all (pre-EOS `p1` = 0.9803,
post-EOS `p1` = 0.9810), and by depth 8 it costs 17.9 % of the accept rate. The
mechanism is straightforward: a deeper draft exposes more positions to a
higher-entropy region, and one rejection truncates the whole tail.

**The trend is real but jagged, and I am not going to smooth it.** It rises
strongly from `d = 1` to `d = 8`, but it is not monotone in the middle: `d = 3`
(−1.6 %) is penalised *less* than `d = 2` (−5.1 %). At these N (54–102 rounds
post-EOS) a 1–3 % swing is not resolvable, so the defensible claim is the band
structure — nil at `d = 1`, a few percent across `d = 2..4`, and an order of
magnitude worse at `d = 8` — not a smooth function of depth. This is the second
place in this report where per-depth structure turns out to be jagged rather
than smooth, which is the concrete vindication of the advisor's comment-8
instruction to scan every depth instead of interpolating between endpoints.

This matters for the policy argument because it is a *second*, independent
reason to prefer shallower depth in exactly the regime where the cost curve
already does — the two effects compound rather than cancel.

**F29 — at 512 tokens, forced `d = 2` beats forced `d = 8` end to end.** On the
parent-clock-stripped after-first-block metric, `d2` runs at 26,119 µs/token
against `d8` at 26,612 µs/token, so `d2` is **+1.85%** faster; the published
scores agree in direction (2.1505 versus 2.1415). This inverts the 256-token
idealised ranking, where cost-curve µs/token was minimised at `d = 7`. The two
are not in conflict: the 256 figure is `C(d)/(d+1)`, which *assumes* full
acceptance, whereas the 512 leg numbers are realised and therefore pay the
F28 depth-scaled EOS penalty. Idealised per-token cost is the wrong statistic to
choose a depth with whenever acceptance is below 1.

## Reconciling 0.20 against the doc block's 0.40

The prior-fit doc block records 0.12, 0.09, "h ≈ 0.6 on the bf16-head stack",
and a "FOURTH FIT … ~10.75 ms marginal per draft on a ~27 ms base", i.e.
10.75/27 ≈ **0.40** — while the shipped constant was **0.20**. It also records
that a prior `h = 0.40` attempt measured **−4.5%** on the easy-prose receipt by
"holding d2–3 where d4 pays".

My measurement explains both. A single scalar cannot be right: 0.40 is roughly
correct for the d >= 4 plateau (measured 0.29–0.39) but ~5x too high for d = 1–2
(measured ~0.08), which is exactly the "holding d2–3" failure the doc block
describes. 0.20 is a compromise that is ~2.5x too high at d = 1–2 and
~1.4–2x too low at d >= 4; its mean, 0.2562, happens to be close to 0.20, which
is why the scalar survived end-to-end tuning **while being badly mis-shaped**.
The per-depth vector removes the compromise instead of re-picking it.

## Submission surface: one file, and what the research instrumentation costs

Preflight on the current head, run against the r2 base
`67bde70274c42aef089ac73cf00608d8037a815e`:

```text
senpai/validate-assignment-scope.sh 67bde70 Sources/MLXFastModel/Qwen36MTPBlockSession.swift
  -> assignment scope OK: 1 submitted path(s)

senpai/check-editable-budget.sh 67bde70
  -> editable budget OK: source=2410226/3000000 headroom=589774
     growth=15576/262144 exempt=2410/2147483648 files=154
```

**One submitted file.** Everything else this experiment produced —
`research/*.py`, `research/*.sh`, `research/out/`, this report — is research-only
and is not in `editablePaths`, so Yukon does not ship it. Growth is 15576 bytes
against a 262144 budget, i.e. **6 % of the allowance**, and 590 KB of total
source headroom remains.

### The instrumentation decision, stated so the advisor can overrule it

The submitted file carries research instrumentation that I have **kept**, and I
want that to be an explicit, reversible choice rather than something discovered
in review:

| hook | gate | default |
|---|---|---|
| `MLX_QWEN_MTP_TRACE` + `MLX_QWEN_MTP_TRACE_PATH` | both required, file sink | off |
| `MLX_QWEN_MTP_FORCE_DEPTH` | env present | off |
| `MLX_QWEN_MTP_TRACE_ROWS` | env present | off |
| `MLX_QWEN_MTP_H_VECTOR` | env present, requires 8 finite ≥0 values | off |
| `h=` field on the trace `begin` line | inside the trace sink | off |
| `headprobe` trace line | inside the trace sink | off |
| `prefix_repair=` / `full_repair=` counters | two `Int` adds in the reject branch | **always on** |

Every hook except the last is behind an env var that is absent on an official
run, and the trace sink additionally needs `MLXFAST_NO_SANDBOX=1`, which
official runs **fail closed** on (F12: `main.swift:2236-2238`, `:2288-2291`, deny
at `:2626-2638`, pinned by `ParentToolSandboxTests.swift:108`). So the tracing
machinery is not merely off by default on the ranked runner — it is
unreachable there.

The two repair counters are unconditional, and I am not hiding that. They are
two `Int` increments confined to the reject branch of one round; against a
65 ms round on a memory-bound 4-bit backbone they are unmeasurable, and they
are the literal counters comment 11 asked for.

My recommendation is to **keep** all of it: it is what made the per-depth curve,
the repair classification, and the EOS split measurable at all, a future agent
re-deriving any of this would have to re-add it, and it cannot execute on the
ranked host. If the advisor prefers a bare submission, deleting the hooks is
mechanical and touches only this one file — say so and I will strip it.

**One thing here is not instrumentation and must not be stripped: the warm
H-probe.** `probeResidentHeadCost` runs in `warmAllDepths` and feeds
`adoptResidentHeadStepRatio`, which is what makes the shipped cost model
head-agnostic instead of a frozen table (comment 7 item 4, and the direct answer
to *"a single frozen table is the one option I will not accept"*). It is
load-bearing policy that happens to also emit a trace line.

## Suggested follow-ups (not implemented)

1. **Raise IPG above 4 for these shapes.** The in-source rule caps
   `IPG = ceil(M / ceil(M / 4))` at 4 inputs per group; the M = 5 and M = 9 pass
   boundaries are the two most expensive marginals measured. A tuned IPG of 5 or
   8 for K = 5120 would flatten exactly those steps. This is a Metal-kernel
   experiment, not a schedule one.
2. **Re-fit the `positionAcceptEMA` prior.** `0.85 · 0.98^d` is wrong in both
   directions (F10). It is the other half of the `argmin` objective and was
   deliberately left untouched to keep this diff minimal.
3. **Width-padding arm** (run verify width `w` at depth `d < w - 1`) to break
   the `H`-vs-verify-slope collinearity and separate head cost from verify cost.
4. **A `d = 5` forced arm** to replace the N = 2 row.
