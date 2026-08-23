# Turn the proposal head's BF16 precision islands off by default

## Context and goal

The campaign frontier is a heavily tuned depth-2 native-MTP schedule. Almost
every remaining schedule idea has already been tried, so this round asked a
different question: *nobody has ever changed the proposal head itself.* The
head is editable, it is read on every draft step, and its arithmetic precision
had never been treated as a tunable. The goal was to find out whether the head
is paying for precision it does not need.

## Environment and setup

Local host: Apple M4 Pro, 48 GB unified memory, macOS. Ranked host: the
organizer's M5 runner `m5-qwen38-27b-mtp`. Target: Qwen 3.8 27B, 64 layers
(48 Gated DeltaNet, 16 full attention), hidden 5120, vocab 248320, affine 4-bit
group-64 backbone. Build and fixtures came up through `./setup.sh` and
`./setup-qwen-mtp.sh`.

Two environment defects had to be repaired before any measurement was
trustworthy. `.build-worker/release/mlx.metallib` was missing, because
`swift build` does not produce it; it was restored only after
`tools/build-mlx-metallib.sh --print-fingerprint` proved the source fingerprint
`2050ebf1c1cf091ebbf35fceb7e9c1a9b399b7ed371b133d10b8546734efe7a7` matched.
`reference_weights/` was empty, which silently broke the harness `source_hash`.
Both were fixed before the timed work, and every leg reported in this note
carries identical `worker_sha256`, `cli_sha256` and
`metallib_source_fingerprint`.

## Prior work and baseline

The organizer's original calibrated depth-2 tree scored about 0.994. The
campaign has since promoted a long chain of schedule and kernel work. The
baseline for this experiment is the current campaign base commit, measured
fresh on the same host, in the same session, with the same token window and the
same declared head. No historical number is used as a comparison point.

## Hypothesis

The proposal head carries three BF16 "precision islands" on its q, k and v
projections. These islands exist to protect numerical accuracy. The hypothesis
was that the head's *drafting* accuracy does not need them, because a draft is
only a proposal: the fixed target re-checks every token, so a slightly worse
draft costs acceptance rate, not correctness. If the accuracy loss is small,
removing the islands removes real weight traffic from the bandwidth-bound
draft step and should be a net win.

## Approach selection and tradeoffs

Three arms were considered: (a) remove the islands entirely, (b) keep the
islands but shrink them, (c) re-quantise the whole head trunk to BF16. Arm (c)
was measured and was a null (-0.0073 pp), so it was dropped. Arm (b) adds a
tuning surface and a new configuration flag for a fraction of arm (a)'s effect,
so it was not built. Arm (a) is a one-line default change with no new
configuration surface, which is the smallest possible submitted diff and the
easiest to reason about and to revert. Arm (a) was selected.

The tradeoff accepted is a small loss of draft quality in exchange for less
head weight traffic per draft step. The measurements below price both sides of
that trade.

## The mechanism, in one line

`Qwen35IslandArm.fromEnvironment` returns `.none` instead of `.all` in
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. The three BF16
precision-island corrections on the proposal head's q, k and v projections stop
being read, and the head drafts from its declared affine-4 group-64 packs.

This is **45 bytes** of submitted growth in one editable file. It changes no
target weight, no tokenizer, no trusted code, and no head declaration. The
target answer is untouched; only the proposal head's own arithmetic changes, so
the candidate still verifies every emitted token against the fixed target.

## Headline result: the clean cell, with no model in it

On `beagle_a` the two arms draft **bit-identically** — the same 119 rounds and
the same accepted and rejected counts — so the schedule and accuracy terms are
exactly zero by construction and the whole effect is the mechanism:

```
beagle_a, candidate MTP seconds per token, none - all = -0.4163 %
```

That is a measured mechanism, not a fitted one.

## Gate-qualified confirmation on a second prompt

A four-leg counterbalanced palindrome (`all, none, none, all`) on
`benchfixture`, 512 decode tokens, every leg through the real 40 C thermal gate:

```
absolute candidate MTP seconds per token, none - all
  -0.7581 % +/- 0.0945 % (1 sigma)   = 8.0 sigma
entry temperatures 39.33 38.87 38.43 38.08 C, spread 1.25 C
all_tokens_matched true on all four legs, 0 residual divergences
round ledger closes exactly: 78/434/62 (all), 77/435/57 (none)
```

The serial control moves +0.0250 % (1.5 sigma), a null, as it must: the serial
leg decodes at depth 0 and never reads the proposal head.

## The cross-prompt estimate, labelled as extrapolated

Splitting each prompt into mechanism, schedule and accuracy columns:

```
mechanism column   -0.5227  -0.5065  -0.5527  -0.4993
                   mean -0.5203 %, spread 0.0534 pp
measured total     -0.4163  -1.4742  -0.9691  +0.7761   spread 2.25 pp
absolute-MTP noise floor 0.039 %
```

The mechanism is prompt-stable to 0.05 pp while the measured total swings
2.25 pp. The split separates a prompt lottery from a mechanism. **On
`plutarch_lives` the total has the opposite sign**, so this is reported as an
extrapolation and not as a per-prompt guarantee.

## Head provenance

Every leg above sealed

```
head_provenance.sha256 = dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9
```

which is the declared head tree, so these legs measure the head the ranked run
drafts with.

One trap worth publishing: **`head_provenance.origin` is not a witness.**
`computeQwenMTPHeadProvenance` takes `origin` from the declaration when
`source == remote`, but takes the digest from the directory actually loaded. A
run that loaded the organizer-pinned BF16 head can therefore seal a declared
`origin` string. Only `head_provenance.sha256` identifies the head that drafted.

## Acceptance repricing

The published cost of one acceptance point is **1.0093 %/pt**, not the 2.6701
we previously used. The old rate was fitted across a set in which the offered
draft depth co-moved with the acceptance rate, so it charged a depth change to
acceptance. A naive regression on that set returns the **wrong sign**. At a
fixed schedule the rate is `q / (1 + a)`, which is structurally pinned near
1 %/pt. Every earlier figure derived from 2.6701 is about 2.65x too optimistic.

## Transfer to the ranked host: a projection, not a measurement

The local figure does not carry to ranked uncorrected. The saved fraction is
`bytes_removed * steps / (BW * R)`, and all four terms differ between the local
Apple M4 Pro and the ranked M5. This arm removes head weight traffic, so it is
bandwidth-bound, and the estimated transfer coefficient is `k ~ 1.36`:

```
local -0.4163 %  ->  about +0.567 % published
```

**This is a projection.** We do not claim the crown. The projected gain sits
inside the run-to-run spread of the serial leg, whose standard deviation is
near 0.67 %, so a single receipt cannot separate this mechanism from the draw.

## Exact commands

```
./setup.sh && ./setup-qwen-mtp.sh
swift build -c release --force-resolved-versions
swift test --force-resolved-versions
python3 research/twin_audit.py
tools/build-mlx-metallib.sh --print-fingerprint
./benchmark-qwen-mtp.sh --local-iterate          # per-leg timed arms
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
  ./benchmark-qwen-mtp.sh --local-submit         # 512-token confirmation
senpai/validate-assignment-scope.sh <BASE_SHA> \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
senpai/check-editable-budget.sh <BASE_SHA>
senpai/verify-ranked-score-boundary.sh
```

Scope, budget and boundary checks all pass at the submitted commit: source
2,607,881 of 3,000,000 bytes, growth 153,046 of 262,144 bytes, 154 files, and
one allowlisted twin waiver out of 29 runtime-effective twins.

## Failures and course corrections

1. **The first regression estimate was wrong by 2.65x.** The acceptance price
   had been fitted on a set where offered draft depth co-moved with acceptance,
   so the fit charged a depth change to acceptance. Repricing at fixed schedule
   gave 1.0093 %/pt instead of 2.6701 %/pt and moved a related break-even from
   3.93 pp to 10.39 pp. This invalidated a whole family of earlier campaign
   projections and is the most important correction in this round.
2. **A three-column decomposition was rejected as a result.** Fitting
   mechanism, schedule and accuracy separately missed the measured round count
   on one prompt by 8.74 % and produced a mechanism term whose sign contradicted
   the directly measured cost column. Only the exact two-column split
   (`t = R / (1 + a)`, using measured R and a) is reported as a result. The
   three-column figures are labelled as extrapolation.
3. **A fitted per-step byte cost came out unphysical.** The fitted step time
   implies about 330.8 GB/s on a part specified near 273 GB/s, so the derived
   ratio inherits that error. It is reported with the caveat rather than used
   as a load-bearing number.
4. **Peak RAM is not reachable on the MTP path.** No repository file emits
   `peak_ram_gb` for this driver, so peak resident set was sampled externally
   with `ps` at 2 s and attributed per leg by start and finish timestamps. The
   worker peaks at 14.687 GB and the arm difference is a null (-0.0068 %).
   Separately, three dead head buffers hold 157,337,600 bytes, which is 1.00 %
   of high water, and are a clean future cleanup.
5. **Post-EOS continuation and the round ledger were verified, not assumed.**
   Every leg reports `all_tokens_matched` true with zero residual divergences,
   and `rounds + accepted - 512 = 0` exactly on both arms.

## What was learned

The head is not free, and its precision is a real lever: turning three BF16
islands off is worth roughly half a percent of candidate decode time with no
correctness cost. More broadly, the mechanism column is prompt-stable to
0.05 pp while the measured total swings 2.25 pp across prompts. Reporting a
single prompt's total as "the effect" is therefore unsound in this benchmark,
and the campaign should price mechanisms, not totals. The acceptance repricing
shows the same lesson from the other side: a regression fitted across a set
with a co-moving confounder can return the wrong sign.

## Next steps

- Build a zero-draft twin of this exact commit as an instrument. With 512
  rounds for 512 tokens it measures the per-step time directly and should
  settle the unphysical fitted bandwidth above. It is expected to score near
  1.17 and be rejected; it is a measuring device, not a candidate.
- Reclaim the 157 MB of dead head buffers.
- Re-derive every campaign projection that consumed the old 2.6701 %/pt price.
- Re-measure this arm on the ranked M5 frame rather than transferring it, since
  the projection coefficient is the weakest link in this note.

## Models, effort and harness

- Agent harness: **OpenHands**, run under the **Senpai** advisor/student
  research control plane (one advisor agent, four student agents, GitHub pull
  requests and labels as the only cross-agent protocol).
- Underlying LLM: **Anthropic Claude**, run at high reasoning effort for both
  the advisor and student roles.
- Local measurement host: Apple **M4 Pro**, 48 GB. The ranked host is the
  organizer's M5. All local numbers above are labelled `harness=local`; the
  transfer projection is the only ranked-frame claim and is marked as such.
