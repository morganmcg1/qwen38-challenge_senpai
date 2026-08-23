# Turn the proposal head's BF16 precision islands off by default

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

## Models, effort and harness

- Agent harness: **OpenHands**, run under the **Senpai** advisor/student
  research control plane (one advisor agent, four student agents, GitHub pull
  requests and labels as the only cross-agent protocol).
- Underlying LLM: **Anthropic Claude**, run at high reasoning effort for both
  the advisor and student roles.
- Local measurement host: Apple **M4 Pro**, 48 GB. The ranked host is the
  organizer's M5. All local numbers above are labelled `harness=local`; the
  transfer projection is the only ranked-frame claim and is marked as such.
