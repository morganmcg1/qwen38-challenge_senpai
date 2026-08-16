# E11 depth-lever showdown — working notes

Assignment `qwen38-r1-e11-depth-lever-showdown` r2, base `8970d775`.

## The arms

`sdpaWidthWallDepthCap` and `segmentedVerifyDepthCap` are not two independent
clamps. `costModelDepth` picks one of them per round:

```swift
let widthCap = fullAcceptStreak >= Self.segmentedStreakGate   // gate = 3
    ? Self.segmentedVerifyDepthCap                            // shipped 7
    : Self.sdpaWidthWallDepthCap                              // shipped 4
```

So the segmented cap only applies after three consecutive full-accept rounds.
Cold or struggling prompts never see it.

| arm | cost model | width cap | seg cap | built from |
| --- | --- | --- | --- | --- |
| `C` (C1/C2) | flat 0.20 | 4 | 7 | base — shipped defaults |
| `C8` | flat 0.20 | 4 | 8 | base — the pre-PR#2 default |
| `F3` | flat 0.20 | 3 | 3 | base — hard depth-3 clamp |
| `H` | measured curve | 4 | 7 | HEAD |
| `H8` | measured curve | 4 | 8 | HEAD |

All five share one `mlxfast-swift` (`d8cb9d74`) and have five distinct
`mlxfast-runtime-worker` hashes, so no two arms can silently run the same bytes.

## Analytic result: the curve is a depth-3 clamp

`research/e11_depth_reach.py` replays the extend walk. The test to go from
depth `d` to `d+1` is `reach > h[d] * (1 + expected) / (1 + cumH)`, and `reach`
is a product of probabilities, so `reach <= 1.0` always.

With the measured curve `h = [0.0842, 0.0775, 0.2426, 0.3754, ...]` the
step 3 -> 4 threshold is **1.0693 > 1.0**, which no `reach` can clear. A 400k
random + grid search over EMA profiles finds no profile reaching depth 4.

Consequences:

- The curve never reaches depth 4, so it never even reaches the width cap of
  4, let alone the segmented cap of 7 or 8. **Both caps are dead code in the
  curve arms.**
- `H8` must therefore be behaviourally identical to `H` — same tokens, same
  round count, same depth histogram. It is a falsification check on this
  analysis plus a second noise replicate, not an independent lever.
- The honest description of the curve is not "a better cost model". It is
  "a depth-3 clamp with slightly eager depth 0->2 behaviour" (h[0] and h[1] are
  *below* 0.20, h[2] is above it).

That is what makes `F3` the interesting arm: it reaches the same depth-3 clamp
with a two-integer change and no new cost vector. Constant-`q` sweep, cap 8:

```
q=0.70  curve=2  flat=3     q=0.90  curve=3  flat=7
q=0.80  curve=3  flat=4     q=1.00  curve=3  flat=8
```

`F3` clamps flat's column to 3, so `F3` and `H` agree everywhere except low `q`,
where `F3` drafts one deeper.

## The local fixture flatters deep drafting

`--local-iterate` defaults to `public_longcopy_gate_english_512_256.json`,
seeded from `public_longcopy_gate_english_512.txt`, whose instruction is
"Copy the passage between the tags exactly." That is a near-maximal acceptance
regime.

The eight hidden pool prompts are prose (beagle, botany, drama, essays,
medicine, plutarch, republic, travel) with calibration raw ratios spanning
0.847 to 1.073. So every local number here is measured in the friendliest
possible regime for deep drafting: `C8` sees its best case and the depth-3
clamp of `H`/`F3` sees its worst case. A tie on longcopy should widen in the
clamp's favour on prose, which is what T3 exists to check.

## Two fidelity layers, both reported

1. **Drift tripwire** — candidate against the M5-generated public 256-row
   golden, run outside the timing window. This is the only *external*
   reference available locally.
2. **Timed leg** — MTP rows against the same build's own serial reference rows
   over all 512 decode tokens (`all_tokens_matched`,
   `residual_divergence_count`). Self-consistency, not external proof.

## Run hygiene

- Timed pass clears every `MLX_QWEN_MTP_*` name and unsets `MLXFAST_NO_SANDBOX`;
  `meta.txt` records the surviving list verbatim so the `H` arms can prove they
  needed no research variable.
- The depth histogram comes free from the trusted parent's
  `effective_draft_lengths` report field, so no trace pass is needed for it.
- `research/e11-build.sh` rewrites the working-tree copy of
  `Qwen36MTPBlockSession.swift` per arm and restores it on EXIT. Read that file
  with `git show` while a build is running.
