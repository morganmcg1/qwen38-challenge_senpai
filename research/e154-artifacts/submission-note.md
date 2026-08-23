# Senpai campaign submission — calibration anchor for our own maintained base

**This submission claims no new mechanism.** It is a deliberate measurement of a
tree we have never measured, and we are spending a slot on it because we cannot
price our own recent work without it. We would rather say that plainly than
dress an anchor up as a candidate.

**Model label:** `senpai` is a campaign label, not a catalogued model. The
underlying agents are `anthropic/claude-opus-5` at reasoning effort `xhigh` for
the research advisor and the four research students, with
`anthropic/claude-sonnet-5` for short subordinate lookups. The agent harness is
OpenHands, driven by the W&B Senpai multi-agent research harness. The target
weights and the proposal head are the ones the benchmark declares and pins; no
model in that sentence is a language model we chose.

---

## 1. Why this submission exists

Our last ranked receipt measured a tree that is now three merges old. Since
then we merged three of our own experiments into the campaign base and have
never measured the result. Every composite we build from here would inherit
that unpriced block, and we would not be able to tell an imported gain from a
regression we introduced ourselves.

So this submission measures the base exactly as it stands, with **no edit at
all** to the submitted surface. The candidate byte-growth against our own
recorded base is `0`, from `senpai/check-editable-budget.sh`:

```
editable budget OK: source=2650331/3000000 bytes headroom=349669
growth=0/262144 exempt=2410/2147483648 files=154
```

We expect this row **not to promote**. Its value is the number, not the rank.

## 2. What is actually in the tree

Against the current promoted frontier's source ref
`0863b06ac16e26e48fc06e97444095b00feb66d4`, exactly five submitted paths differ:

| path | delta |
| --- | ---: |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | +980 / −500 |
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | +273 / −65 |
| `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h` | +158 / −64 |
| `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp` | +158 / −64 |
| `mtp-head.manifest.json` | +1 / −1 |

The two `quantized_nax` files are a readable Metal source and its
runtime-effective generated twin; they are kept in step and audited together by
`research/twin_audit.py`, which reports 29 runtime-effective twins in agreement.

The proposal head is the pinned, declared head. `mtp-head.manifest.json` names
`hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae6282749a52e052496dd5300b4aa441df7301e8`
with artifact digest
`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`; the
one-line difference against the frontier's manifest is descriptive text, not the
artifact.

## 3. Local evidence

One gated `--local-submit` run at the ranked token count, on the exact submitted
tree, with the runtime worker rebuilt from that tree immediately before the run
and re-hashed immediately after it. The worker digest was unchanged across the
leg:

```
worker_sha256 fbec57d0d58feee74c98c694434dc9ed9188af2359226ea566d9f2d86a4529cb
```

```json
{"score": 2.3186635755123963, "passed": true,
 "metrics": {"mode": "qwen-mtp-local-submit", "decode_tokens": 512,
  "mtp_depth": 8, "all_tokens_matched": true, "residual_divergence_count": 0,
  "uses_pinned_mtp_head": true, "effective_mean_draft_len": 6.3766233766233764,
  "accepted_draft_rate": 0.88594704684317716,
  "serial_seconds_per_token": 0.073790716705843806,
  "mtp_seconds_per_token": 0.031824675854295492,
  "public_drift_tripwire_passed": true, "official_score": false,
  "rankable": false}}
```

Row accounting closed on both legs: the serial control checked `512/512` rows at
depth 0, the MTP leg checked `568/568` rows over 77 rounds, and the reference
pass reported `rows=513 self_consistent=true chain_contradictions=0`. Both timed
legs passed the real 40 °C cool gate, at 39.1 °C and 39.2 °C entry.

**We do not offer that local ratio as a score.** Both local legs run the
candidate build and the reference rows are candidate-generated, so the harness
correctly marks the run `rankable: false`. It is an exactness and
row-accounting gate here, nothing more.

## 4. What we are not claiming

* No speedup, and no mechanism. The submitted surface is unchanged from our own
  maintained base.
* No transfer model. This row is intended to *supply* a measurement that our
  pricing models currently lack, not to test one.
* We hold no belief that this row clears the current bar, and we would report a
  drop as informative rather than as a failure.

## 5. Disclosures

1. We are spending a public submission slot on a diagnostic. We think that is
   the honest use of it, given that our alternative was to build a five-part
   composite on top of an unmeasured block.
2. The local harness cannot reproduce the hidden eight-prompt result. Its
   reference rows come from the candidate build, and its single public prompt is
   not the ranked pool.
3. This note was written by an AI research student agent (OpenHands) on behalf
   of the campaign owner.
