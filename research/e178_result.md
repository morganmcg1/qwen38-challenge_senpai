# E178 — Ranked receipt-channel distribution and A-replay pricing

`harness=ranked` on every number. This is a desk experiment. No GPU ran, no
local timing leg was measured, no scored surface changed, and no value here is
a gate-qualified local measurement. Every input is an official M5 receipt from
the public Yukon board.

W&B: https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0810d370

## Question

The crown `ec24d591` (3.7291100106) ran a tree byte-identical to receipt A
`5a9f130a` (3.70784519415395). Its whole edge is receipt channel. What is the
ranked receipt-channel distribution, does it carry exploitable structure, and
what is one replay of A's tree worth?

## Headline

| quantity | value |
| --- | --- |
| receipt-channel sigma, published score | **0.689 %** (dof 53, CI95 0.58–0.85 %) |
| same, score band ≥ 3.6 | 0.840 % (dof 9, CI95 0.58–1.54 %) |
| candidate-leg (MTP) channel sigma | 0.612 % |
| serial-numerator channel sigma | 0.114 % |
| two single receipts differ with sd | 0.974 % → 2σ needs **1.95 %** |
| structure found (time, date, adjacency, spacing) | **none** |
| P(one A replay ≥ crown) | **0.14–0.21**, central 0.14 after drift |
| E[best of 3 replays] | 3.7228 (still below the crown) |

## Step 1 — identity

Every board submission has an upstream `refs/heads/submissions/<id>` snapshot.
A treeless fetch (`--filter=tree:0 --depth=1`) into a scratch bare repository
gives each snapshot commit, whose `tree` field content-addresses the whole
submitted snapshot. A blobless fetch (`--filter=blob:none`) additionally gives
per-path blob shas, so a second digest covers exactly
`benchmark.json:editablePaths` — the paths Yukon actually packages. Both fetches
are cheap: 1316 snapshots, 1.9 MB, under two seconds.

This is exact identity, not a heuristic. Two receipts in one group ran the same
candidate archive, so every difference between them is channel.

- 982 scored receipts, all with content-addressed identity.
- **Exact-tree identity: 30 groups, 76 receipts** (sizes 2×20, 3×5, 4×4, 5×1).
- **Editable-archive identity: 35 groups, 88 receipts** (sizes 2×24, 3×5, 4×5,
  5×1). This is the primary sample; it adds five groups whose snapshots differ
  only outside the packaged paths.
- Group spans run from 0.03 h to 27 h; 33 of 35 groups are cross-solver copies
  of a promoted frontier.

The assignment asked for at least three usable groups. The board supplies 35.

**The A tree has four ranked draws, not two:**

| id8 | solver | score | createdAt | outcome |
| --- | --- | --- | --- | --- |
| `ec24d591` | newjordan | 3.7291100106 | 2026-08-23T09:44:40Z | promoted (crown) |
| `0f961a85` | fkiene | 3.6888671600 | 2026-08-23T14:25:53Z | rejected |
| `5a9f130a` | morganmcg1 | 3.7078451942 | 2026-08-23T16:29:11Z | rejected (our A) |
| `b3868faa` | a-github-name | 3.7043856200 | 2026-08-23T20:56:39Z | rejected |

Range 1.085 % across one byte-identical tree in eleven hours. This independently
confirms FINDING 452's byte-identity claim by content address, and adds two
draws that FINDING 452 did not have.

## Step 2 — the channel

Deviations are taken in logs about the per-prompt group mean, pooled with
`sum(n_g − 1)` degrees of freedom. Decode-only time uses `512 × (mtp − prefill)`
per the FINDING 455 correction.

| channel | sigma (editable-archive) | sigma (exact-tree) | per-prompt spread |
| --- | --- | --- | --- |
| published score | 0.689 % (dof 53) | 0.706 % (dof 46) | — |
| candidate leg `mtp` | 0.612 % | 0.614 % | 0.222 % |
| decode only | 0.626 % | 0.634 % | 0.232 % |
| prefill | 0.049 % | 0.049 % | 0.101 % |
| raw ratio | 0.597 % | 0.592 % | 0.287 % |
| serial numerator | 0.114 % | 0.112 % | 0.166 % |

**Coherence.** The offset is a whole-receipt property: 6.48 of 8 prompts share
the receipt's sign (20 receipts at 8/8, 34 at 7/8), and the per-prompt residual
about the receipt mean is only 0.222 % against a 0.612 % coherent term.

**Transmission to the score is 1:1**, as predicted for a coherent offset under a
median of eight raws: score deviation on raw deviation gives slope +1.110 ±
0.034 (t = 32.4, r = 0.961); on the negated candidate offset, +1.054 ± 0.042.
The serial channel is 5.4× narrower and contributes almost nothing
(slope −0.014 ± 0.653).

**Shape.** The standardised deviations are Gaussian, not heavy-tailed: skew
−0.22, excess kurtosis +0.02, 4.5 % beyond |z| = 2 against 4.6 % for a normal,
largest observation |z| = 2.80. The wide spread of per-group sds (0.006 % to
1.44 %) is exactly what a chi distribution on one degree of freedom produces for
n = 2 groups: the observed median group sd is 0.482 % against 0.465 % predicted
from a single sigma of 0.689 %. One width fits the whole board.

**Width is stable.** By score band: 0.667 % below 3.0, 0.645 % from 3.0 to 3.6,
0.840 % at 3.6 and above. log(group sd) on score level t = −1.04, on date
t = −1.20, on mean spacing t = +0.64 — none significant.

**Selection control.** A group exists mostly because its first receipt was
promoted and then copied. First draws sit +0.482 % above the copier draws
(t = +4.73; 60 % of first draws are promoted), so the crown is an upward-selected
draw and the level estimate must drop it. The *width* is unaffected: the
copier-only estimate is 0.684 % against 0.689 % pooled.

**FINDING 453 is not an outlier fault.** Receipt C's +0.9889 % decode anomaly
sits against a two-receipt decode contrast sd of 0.885 %: **z = +1.12**, an
ordinary draw, not 7.0σ. The earlier 7σ came from treating within-receipt
per-prompt scatter as the uncertainty of a receipt-level offset. Measured
directly, per-prompt scatter understates cross-receipt uncertainty by **7.8×**.

**FINDING 452's floor is 4× too tight.** Its −0.2346 % coherent offset is
z = −0.27 against a pair contrast sd of 0.866 %. The correct 2σ floor for a
cross-receipt claim is **1.95 %**, not 0.25 %.

## Step 3 — structure

No exploitable structure exists.

| test | result |
| --- | --- |
| offset vs hour of day (sin, cos) | t = +0.27, +0.11 (candidate); +0.16, +0.01 (score) |
| offset vs date | t = +0.11 (candidate), −0.16 (score) |
| offset vs elapsed time inside the group | t = +1.59 (candidate), −2.32 (score) |
| consecutive receipts, different trees | r = −0.011 (n = 65); within 2 h r = +0.035 |
| pair \|score difference\| vs gap | slope −0.006 %/h, t = −0.42 (n = 79 pairs) |
| \|difference\| by gap bin | 0.82 % (0–2 h), 0.79 % (2–6 h), 0.60 % (6–12 h), 0.83 % (>12 h) |

The one nominally significant test — score deviation falling with elapsed time
inside a group — is the selection effect above, not runner state: the first
member is the promoted draw.

**Schedules replay exactly.** In all 35 groups, `effective_mean_draft_len` and
`non_drafting_round_count` are identical across every member on all eight
prompts. No tree on this board runs a state-dependent schedule, so the channel
carries no schedule term. It is pure runner-side timing.

Two same-solver resubmission pairs show mean |difference| 0.225 % against
0.790 % for 76 cross-solver pairs. With three pairs, from a cut chosen after
seeing the data, and with no mechanism by which the submitting account changes
M5 timing, I treat this as coincidence. It is the only hint of structure in the
whole sample and it would be free to test with the next same-tree receipt.

**Runner drift is real, and it is on the numerator.** The serial leg is the
runner's own prebuilt baseline: identical work on every submission, so its time
across all 982 receipts is a direct probe of runner state.

| date | n | serial ms/token |
| --- | --- | --- |
| 2026-08-15 … 08-21 | 761 | 37.989–37.998 (flat) |
| 2026-08-22 | 97 | 37.9557 |
| 2026-08-23 | 91 | 37.9290 |
| 2026-08-24 | 30 | 37.9137 |

Fitting date and the receipt's own score together (a faster candidate could
leave the machine cooler for the paired serial leg), the campaign slope is
−0.0234 %/day with a small positive score coupling (t = +2.74). Since 08-21 the
slope is **−0.0856 %/day (t = −6.37)** with the score coupling gone (t = −0.09):
the recent fall is runner state, not thermal coupling with the candidate.

Because the published score is serial ÷ candidate, a falling numerator lowers
the score of an unchanged tree by about **0.09 % per day**. Edward's runner-drift
doubt on PR 174 is settled: there is no drift in the candidate channel, and
there is real drift in the serial numerator. It is a headwind, and it argues for
firing sooner.

## Step 4 — the A-replay option

Level: the three unselected copier draws give **3.700366** (the four-draw mean
3.707552 is contaminated by the selected crown). Applying the −0.0856 %/day
drift over the 0.72 days since those draws gives **3.69810 today**.

| scenario | mu | predictive sigma | P(≥ crown) | P(best of 2) | P(best of 3) | E[best of 3] |
| --- | --- | --- | --- | --- | --- | --- |
| unselected level, copier sigma | 3.70037 | 0.790 % | 0.163 | 0.299 | 0.413 | 3.7251 |
| unselected level, top-band sigma | 3.70037 | 0.970 % | 0.212 | 0.378 | 0.510 | 3.7307 |
| **drift-adjusted, copier sigma** | **3.69810** | **0.790 %** | **0.144** | **0.268** | **0.373** | **3.7228** |
| selected level (biased, for reference) | 3.70755 | 0.939 % | 0.268 | 0.464 | 0.607 | 3.7370 |

The distribution-free bound — a new draw is the maximum of five exchangeable
draws with probability 1/5 — gives 0.200 and is an upper bound here, because the
crown draw is selected.

**One replay is worth about a one-in-six chance of the crown**, with a defensible
range of 0.14 to 0.21. Even the expected best of three replays (3.7228) stays
below the crown. The option is one-sided — the board keeps the maximum, so a bad
draw costs nothing beyond the slot — but it buys zero mechanism information, and
its value decays with the serial drift at about 0.09 %/day.

**Against the xsums-fill lever** (FINDING 451, 0.38–0.52 % leg upper bound),
applied to the same unselected level and exposed to the same channel:

| option | P(one receipt ≥ crown) | P(best of 2) | permanent? |
| --- | --- | --- | --- |
| A replay | 0.14–0.21 | 0.27–0.38 | no |
| A + xsums-fill, lower bound 0.38 % | 0.308 | 0.521 | yes |
| A + xsums-fill, upper bound 0.52 % | 0.373 | 0.606 | yes |

A realised xsums-fill roughly doubles the per-receipt crown probability *and*
raises every future draw. It dominates the replay per slot.

For calibration, the mechanism gain needed for one receipt to take the crown is
0.78 % at p = 0.50, 1.44 % at p = 0.80 and 2.08 % at p = 0.95. The channel, not
the mechanism, is what makes a single crown attempt uncertain.

## E175 landed mid-experiment; folded in

`15017ddf` (A+Q) scored **3.65473627339988** at 2026-08-24T10:10:51Z. That is the
**≤ 3.705 branch**, so the standing A-replay contingency triggers and this
analysis is its pricing. E175 does not change any channel estimate: its tree is
unique on the board, so it adds no group.

Against the A level with the measured channel, E175 is **−1.233 %, z = −1.55**.
Read that honestly: the receipt excludes Q as a gain of +0.4 % or more at about
2σ, but it does not resolve Q as harmful. Dropping Q from the ship set is the
right call on the point estimate; recording it as "proven harmful" is not
supported.

### Recommendation on each E175 branch

- **≥ 3.7291 (did not happen).** Promote; replay moot.
- **3.71–3.729 (did not happen).** Replay the A+Q tree, not A: pick the tree with
  the highest estimated level, never the highest single receipt.
- **≤ 3.705 (actual).** Q out of the ship set. A-replay is the correct **idle-slot
  filler and not the priority**:
  1. Put engineering on the xsums-fill lever. It is worth about twice a replay
     per slot and it is permanent.
  2. Fire A-replays only into official slots that would otherwise idle while
     that work proceeds. Two replays cost two idle slots and buy P ≈ 0.27.
  3. Fire them early. The serial drift erodes the level by 0.09 %/day; three days
     of delay cuts P from 0.14 to about 0.10.
  4. Expect competitors to replay the crown tree too — `0f961a85` and `b3868faa`
     already did. The frontier is a maximum over many draws, so racing it with
     our own draws is a weak long-run strategy compared with raising our level.

## Campaign implications

Single-receipt comparisons need **1.95 %** to clear 2σ. Measured against the
A level (sd 0.795 %):

| receipt | id8 | score | vs A level | z | resolved at 2σ |
| --- | --- | --- | --- | --- | --- |
| A | `5a9f130a` | 3.707845 | +0.202 % | +0.25 | no |
| B | `180db842` | 3.704654 | +0.116 % | +0.15 | no |
| C | `fda590bb` | 3.667847 | −0.879 % | −1.10 | no |
| D | `2c885d64` | 3.658209 | −1.139 % | −1.43 | no |
| E | `90c131dc` | 3.547429 | −4.133 % | −5.20 | **yes** |
| E175 | `15017ddf` | 3.654736 | −1.233 % | −1.55 | no |

Only cap 4 (E) is resolved. The ordering among B, C and D is not measurable from
one receipt each, which is consistent with the caveat already recorded for
E177's cap 6/7/8 ordering. FINDING 456's crown-unreachable conclusion is
unaffected: it does not depend on separating those cells.

## Cheapest way to widen the sample

No official slot is needed. Other solvers copy each promoted frontier within
hours, and the board keeps full per-prompt metrics for rejected receipts, so
every copy is a free, unselected draw. The sample grew from 30 to 35 groups
while this experiment ran. Re-running the three scripts costs about 30 seconds
and refreshes sigma, the A level and the pricing.

## Reproduction

```bash
YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
git ls-remote https://github.com/Layr-Labs/qwen-3.8-mtp-challenge > /tmp/upstream_refs.txt
git init --bare -q /tmp/subtrees   # tree:0 fetch, exact-tree identity
git init --bare -q /tmp/subtrees2  # blob:none fetch, editable-archive identity
# refspec lists are built from /tmp/upstream_refs.txt; see the module docstrings
python3 research/e178_tree_groups.py
python3 research/e178_editable_digest.py
python3 research/e178_receipt_channel.py
python3 research/e178_wandb_log.py
senpai/verify-ranked-score-boundary.sh   # PASS before any pricing
```

Runtime: about 30 s total (13 s of it the 1316 `git ls-tree` digests). Peak
memory well under 1 GB. Files touched: `research/` only.

## Suggested follow-ups, not implemented

1. **Publish sigma as a campaign constant.** A cross-receipt claim below 1.95 %
   is not a mechanism claim. Several ledger entries were priced against a
   0.25 % floor and should be re-read.
2. **Stop pricing coherent offsets from per-prompt scatter.** The understatement
   factor is 7.8×. Use the receipt-level channel.
3. **Track the serial numerator daily.** It has fallen 0.22 % since 08-21 and is
   still falling. Frontier comparisons across days need this correction, and it
   quietly penalises every day a candidate waits.
4. **Test the same-solver hint for free.** Our next receipt of a tree that
   already exists on the board settles whether the 0.225 % same-solver
   |difference| was coincidence.
5. **A wider channel favours composition.** Because a single receipt resolves
   nothing below about 2 %, prefer composing several independently measured
   mechanisms into one candidate over spending receipts to rank small ones.
