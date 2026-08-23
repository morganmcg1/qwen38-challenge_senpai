# SENPAI Research State

- 2026-08-23 03:10 UTC
- Most recent human research direction: none new this round. The standing
  direction remains the operator's campaign brief in `senpai/program.md`:
  maximise the official decode score on `qwen3.8-27b-mtp-v1`, submit the
  strongest legitimate candidate promptly, and never stop at synthesis.

## Where the campaign stands

The published crown is `3ba6ee9d` (Amal-David) at **3.70576324**, accepted
2026-08-22T23:49:58Z. Our own best promoted row is `572b2cc4` at 3.66218564.

Finding 213 is the important part of that sentence. The crown's candidate leg
is **slower** than the row it displaced (`1760479a`, 3.70355222): candidate
median-pair +0.0328 %, serial median-pair +0.0890 %. Draft lengths and
non-drafting counts are digit-identical across all eight prompts. The crown's
declared mechanism is an untimed warm that compiles a head fold expression
before the scored rounds, which Rule 110 prices at zero. **The crown moved
because its serial leg drew a fast ticket, not because its candidate got
faster.** The fastest promoted candidate leg on the board is still `1760479a`.

That splits the board into two different reference rows, which is now Campaign
Rule 124: anchor **value** tables on the published crown, because that is the
bar we have to clear; anchor **candidate** contrasts on the fastest promoted
candidate leg, because that is the physics. Today: bar `3ba6ee9d`, frontier
`1760479a`.

The noise model behind that (Finding 211, twenty true null pairs, nine solvers)
gives a per-row published sd of 0.159 %. Converted into a decision table:

| candidate median-pair gain over `1760479a` | expected median | P(new crown) |
|--:|--:|--:|
| +0.00 % | 3.70355 | 35 % |
| +0.10 % | 3.70726 | 60 % |
| +0.25 % | 3.71281 | 88 % |
| +0.40 % | 3.71837 | 98 % |

Anything below about +0.25 % of candidate median-pair gain is a coin flip
dressed as a result. That is the bar every assignment below is priced against.

## Current research focus

### 1. The proposal head is a submittable surface (Finding 214) — largest lever

I spent this campaign believing a re-quantized proposal head would need an
external artifact upload. That was wrong. `benchmark.json` puts `mtp-head/`
in `editablePaths`, marks it optional, exempts it from the 3 MB source budget
with its own 2 GiB cap, and `QwenMTPHeadDeclaration.swift` implements
`source: "in_branch"` against a repo-relative path. A head can ship in the
branch with zero new source bytes.

The prize is a single clean comparison already in the ledger: the
organizer-pinned bf16 head reaches 93.13 % acceptance; the shipped declared
head is **the same weights** put through naive round-to-nearest affine-4 g64
and reaches 92.31 %. That 0.82 pt is quantization damage and nothing else. The
bf16 head is unshippable because it costs 2× the bytes for 2× the time. A
better *data-free* quantizer at identical bytes and identical layout keeps the
head step, changes no Swift, and converts 1:1 under Rule 121 because it is a
uniform gain:

| recovered | raw gain | median | P(new crown) |
|--:|--:|--:|--:|
| +0.20 pt | +0.406 % | 3.72081 | > 98 % |
| +0.40 pt | +0.812 % | 3.73585 | > 99.9 % |
| +0.82 pt | +1.665 % | 3.76746 | > 99.9 % |

Recovering a quarter of the damage takes the crown. Nothing else on the books
is this large.

The qat-q4 *artifact* remains closed (ledger 282.6): different trained trunk,
McNemar chi-square 0.083, 2.7–3.4 % slower, licence. Only the quantizer axis is
open.

### 2. Register occupancy at verify width 6 — the next crown attempt

Thorfinn's F22. The `na6` kernel uses 96 registers on g16s and **105 on g17s**,
where the ceiling is 96 for full occupancy; the g17s simdgroup count drops to
37 against 42 at `na3`. Rolling the column-pair loop to get `na6` under 96 g17s
registers is priced at +0.5996 % of published median under Rule 121, band
[0, +0.93 %], and it is not capped by the essays saturation because the gain
shape is broad. Kill rule: close the arm if no variant reaches ≤96 g17s
registers without spilling more than it saves.

### 3. The scheduler's two-dimensional price plane

Finding 210 showed that `makeBoundaryDepthPrice` holds the total, so `tier` and
`within` are the same constant and E134's tier grid swept a one-dimensional
diagonal. The `(h, tier)` plane has never been searched. This matters beyond
its own gain: Campaign Rule 125 says an acceptance gain must be priced through
the scheduler's *response*, not at fixed depth, because E82 bought +0.85 pt of
acceptance and lost 2.7–3.4 % of time when the depth walk spent it on extra
width. The head axis in item 1 therefore depends on knowing the plane.

### 4. The compact draft vocabulary

Alphonse's E141. The head cannot propose one target token in about two hundred
because its vocabulary is a 98,330-entry prefix of 248,320. Two students have
now measured the channel and disagree by 1.7×: alphonse's corpus census says
beagle 0.98 % / essays 0.92 %, askeladd's live first-divergence census says
beagle 1.15 % / essays 0.00 %. The honest band for the fix is +0.35 % to
+0.90 % against an arm-A cost of 0.70–0.83 %, which makes the specified arm A a
coin flip. Arm B (generalising the cluster-row kernel so the widened table
costs 0.22 MB instead of 30 MB) is now required first. One free measurement
settles the disagreement: the draft-position split, asked of both students.

## Closed this round

- **The acceptance axis inside the shipped head.** Askeladd's E143 resolved
  92.23 % [91.22, 97.64] of beagle's acceptance gap to channel C-d, the
  head's own ranking quality, at every carrier and every interval end above my
  pre-registered 80 % line. Reassigning all sixteen unresolved rows still
  leaves 91.22 %. The residual +5.234 % of published median is unreachable
  without a different head — which is exactly why item 1 above is the follow-on.
- **C2, the precision islands.** Advisor Error 148: I priced it at +0.35 % from
  a byte count, which is three times the corrected gross and prices a change
  that never pays for the acceptance it spends. E82's own receipt has arm `q`
  at 1.801 % *slower*. Askeladd proved the arm reproduces C2's numerics
  bit-exactly with no source edit at all.
- **Advisor Error 150**, recorded above: the head-upload blocker I invented.

## Potential next research directions

- The width-independent GPU-work pool: about 3.7–3.8 % of the ranked round is
  work that does not scale with verify width. It is the largest uniform target
  on the books and it has no owner.
- Head-side confidence, per position, feeding the depth policy. Rung 0 is
  zero-GPU on the cached E142 verify-row capture. Estimated +0.5 %,
  beagle-weighted, band [0, +1.5 %].
- The head flush-width determinism census at `Qwen36MTPBlockSession.swift`
  `:1540-1580`, which cleans up part of E143's C-d residual.
- P4, the Gated DeltaNet S=2 mid-state write, which gates 151 MB per round on
  rejection. Highest correctness risk on the board; unowned.
- Finding 190: the verify-width cliff appears to move one width between two of
  our own bases. Check the E92 axis label before assigning any bisect.
- Composing the two measured held riders, worth +0.2 % to +0.3 % together.

## Standing constraints that shape all of the above

- **Rule 123**: the published median is
  `(beagle + min(essays, republic, medicine, botany)) / 2` — a worst case, not
  an average. Buffers at the current bar are medicine 0.825 %, republic
  1.187 %, botany 1.417 %, all tighter than last round. A uniform gain is the
  safest class because it cannot manufacture a new minimum.
- **Rule 72**: one shot per submission. No re-rolling for a better serial
  ticket. Finding 213 makes this rule load-bearing rather than decorative.
- **Rule 118**: price every ranked contrast on the candidate leg. Finding 211
  quantifies why — the candidate leg is a 3.08× more precise instrument than
  the serial leg.
