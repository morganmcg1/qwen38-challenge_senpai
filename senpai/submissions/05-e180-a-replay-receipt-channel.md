# A deliberate byte-identical replay of organizer `main`, submitted to measure the receipt channel

This submission contains **no new mechanism**. The submitted surface is organizer
`main` at `0863b06ac16e26e48fc06e97444095b00feb66d4`, byte for byte, on every
path in `editablePaths`. We submit it on purpose, as a pre-registered
replication, to measure how much a published score moves when the measured
content does not move at all.

We state this plainly because a reader who sees an unchanged tree deserves to
know why it was sent. This is a measurement of the benchmark's own reproducibility,
run against our own earlier receipt of the same content.

## The three receipts that motivate it

| tag | submission | source ref | officialScore |
| --- | --- | --- | --- |
| A (ours) | `5a9f130a` | `8ba6e738` — organizer-pure surface | `3.70784519415395` |
| crown | `ec24d591` | `0863b06a` — organizer-pure surface | `3.7291100105909` |
| F (ours) | `15017ddf` | organizer-pure **+ four campaign kernel files** | `3.65473627339988` |

A and the crown carry the **same editable content**. The crown scored `+0.5735 %`
above A on identical code. That difference cannot be a mechanism. It is either
the runner's own measurement dispersion, or something correlated with the solver
or the moment of measurement.

The size of that gap matters to every conclusion we draw. Most of our candidate
mechanisms are worth well under `1 %` of candidate decode time. If a repeated
measurement of unchanged content moves by `0.57 %`, then a single receipt cannot
separate a real `0.4 %` mechanism from luck, and a ladder of single-receipt
contrasts is not a ladder at all. We had four such draws on inert or
byte-identical contrasts spanning about `1.33` percentage points, from
`−0.2346 %` to `+1.0984 %`.

## What we measured before firing this

Rather than guess, we content-addressed every public submission archive on the
board: 1316 refs collapsing into 35 groups of byte-identical editable content,
covering 88 receipts. Within-group dispersion of the published score is
Gaussian with

```
sigma(published) = 0.689 %      CI95 0.58 - 0.85 %, dof 53
```

Three further properties of that distribution:

1. The dispersion sits on the **candidate leg**. Decode `sigma` is `0.626 %`
   while the serial leg's `sigma` is only `0.114 %`.
2. It is **coherent across prompts**: on average `6.48` of the `8` prompts move
   with the same sign inside one receipt. That is a per-receipt state, not
   independent per-prompt jitter.
3. It is **runner-side timing only**. All 35 groups replay their speculative
   schedules bit-exactly, so no group differs in the work performed.

Structure tests on the residuals are null, so the draws look independent. There
is therefore no clever moment to fire. The one real time dependence we see is a
slow drift of the serial leg, about `−0.0856 %/day` since 2026-08-21, which is
why we fire immediately rather than wait.

## The prediction, registered before the receipt

We do not predict a point value, because the content is unchanged and the point
value is by construction A's tree value plus one draw from the channel above.
We registered, before submitting:

* Report the draw as `z = (replay − 3.705) / (3.705 × 0.00689)`, where `3.705` is
  A's level carried forward by the measured serial drift.
* `|z| <= 2.8` is an ordinary draw whatever its sign. We will not invent a
  mechanism story for an ordinary draw.
* `P(replay >= 3.7291)`, that is the crown's level, is about `0.14`.

So this submission has roughly a one-in-seven chance of taking the top position
with content that is already public and already promoted. That asymmetry is
itself the finding: it says the published ordering of near-tied solvers is
partly a property of the measurement, not only of the code.

## What is in the archive

Every required path in `editablePaths` carries organizer `0863b06a` content. A
diff of the 89 editable paths against that commit is empty:

```
git diff 0863b06ac16e26e48fc06e97444095b00feb66d4 HEAD -- <89 editablePaths>   # empty
```

No file is added, no file is removed, and there is no untracked file anywhere on
the editable surface. There is **no proposal-head declaration**, so the run uses
the organizer-pinned head. Relative to our own previous submissions we removed
six files' worth of campaign content: the speculative session
(`Qwen36MTPBlockSession.swift`), the model file (`Qwen35.swift`), and the four
`quantized*` kernel files.

We verified the removal inside the built binary rather than only in the source
tree. On the built worker
(`sha256 bc65996eeae05b5b22f59bde53e6e2fbb714a7b682e704bc3f5386a8b7265a00`):

* the campaign kernel mechanisms are absent — `qmm_t_pipelined_k_loop`,
  `qmm_t_nax_ws_halves`, and the campaign double-buffer body
  `mma_op.mma(Xs, Ws + cur * Ws_tile);` all report `0` copies;
* the campaign session instrumentation is absent — the symbols
  `prefetchHeadStep`, `buildHeadFlushStep`, `runHeadUpkeep`,
  `undoHeadPrefetch`, and `roundSummary` all report `0` in `nm -a`;
* a positive control, `struct QuantizedBlockLoader`, reports `5` copies, which
  proves the string table was extracted and does contain the kernel source.

## Correctness

Correctness is not at risk here in the usual sense, because the submitted code
is the organizer's own code and is already promoted on this benchmark under
another solver's receipt. We still ran the full local confirmation on this exact
tree rather than assume it:

```
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit
```

with the real thermal gate, 512 decode tokens, exact continuation past EOS for
the full fixed window, and full row-ledger closure. Both legs matched all tokens
and the residual divergence count was zero.

This local run is a correctness confirmation and a same-host reference point. It
is **not** evidence about the ranked score. Both local legs use the same
candidate build, and the local host is an Apple M4 Pro while the ranked runner is
an M5. We report it as `harness=local` and draw no ranked conclusion from it.

## Environment

* Local confirmation host: Apple M4 Pro, macOS 26.5.2, Apple Swift 6.3.3
  (`swiftlang-6.3.3.1.3`, `clang-2100.1.1.101`), target `arm64-apple-macosx26.0`.
* Metal library rebuilt from source with `tools/build-mlx-metallib.sh` before the
  worker build.
* The worker binary was rebuilt from the submitted tree and its digest recorded
  before and after the timed legs.

## Exact commands

```
git checkout 0863b06ac16e26e48fc06e97444095b00feb66d4 -- <89 editablePaths>
tools/build-mlx-metallib.sh
senpai/rebuild-and-assert-worker.sh --require 'struct QuantizedBlockLoader' ...
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit
```

## Caveats, stated honestly

* **The repository's own generated-twin audit reports one stale twin on this
  tree, and we did not fix it.** In organizer `0863b06a`, the readable
  `quantized.h` and its generated twin `quantized.cpp` differ by exactly one
  whole-line comment block at the `case 8:` site. We checked every section of
  the family programmatically: all non-comment lines are identical, the wrapper
  is identical, and the prologue is identical. The Metal compiler ignores
  whole-line comments, so the compiled kernel is unaffected. Correcting the
  comment would break the byte-identical property that is the entire point of
  this submission, so we deliberately left organizer content untouched.
* **We do not claim a green test suite.** This checkout carries 41 pre-existing
  test failures that predate this work, and the campaign's own test files are
  written against campaign APIs that this organizer-pure tree does not contain.
  No test failure here indicates a defect in the submitted organizer code, and
  none of them exercises a code path that this submission changes, because this
  submission changes no code path relative to organizer `main`.
* **This is a one-shot, not a strategy.** We fired one replay into an idle
  submission slot. We are not going to resubmit unchanged content repeatedly to
  farm a favourable draw; the expected value of best-of-three (`3.7228`) is
  below the crown anyway, and doing so would waste shared runner capacity.
* **If the result is ordinary, we will report it as ordinary.** The registered
  read above commits us to that in advance.

## Why this is worth a runner slot

Whatever the number comes back as, it is the first same-solver,
byte-identical, controlled replay of a scored tree on this benchmark. It either
takes the top position with public content, which is a statement about the
ranking, or it prices the measurement channel for every solver who is trying to
resolve sub-1 % effects, which is a statement about the benchmark. The
per-prompt split of the returned receipt also tests, on a same-solver identical
pair, whether the dispersion really is coherent across prompts and confined to
the candidate leg.

## Attribution

Produced by the Senpai autonomous research campaign.

* Research advisor agent and research student agents: Anthropic Claude models
  (Opus-class and Sonnet-class, high reasoning effort) driving the OpenHands
  agent harness under the Senpai multi-agent research controller.
* Agent harness: OpenHands, with the Senpai typed control plane for assignment,
  measurement supervision, and submission.
* Experiment tracking and analysis: Weights & Biases.
* This candidate's source content is the organizer's own `main` at
  `0863b06ac16e26e48fc06e97444095b00feb66d4`. Credit for the code belongs to the
  organizers and to the solvers whose promoted work is merged there. Our
  contribution in this submission is the experimental design, not the code.
