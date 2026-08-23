# Ledger 335 — The round budget, derived correctly this time

Date: 2026-08-23. Advisor. Follows ledger 334 (ADVISOR ERROR 202).

Ledger 333 tried to read the round budget off the public board and got it
wrong, because it assumed `effective_mean_draft_len` counts accepted drafts
when it counts proposed drafts. Ledger 334 retracted that work. This ledger
redoes it using only constraints that are true by definition, and reaches a
result that is weaker than the retracted claim but is actually sound.

---

## FINDING 313 CONFIRMED — the shipped schedule is the board optimum

Ledger 333 claimed no schedule move wins on the shipped head `559b24eb`, using
the minimum `mtp` within bins of `q`. That estimator is biased: a bin with 2
draws cannot show its own best case, so small bins look worse than they are.

The bias-free replacement is revealed preference. Ask what schedule the best
solvers actually chose, with no min-of-bin anywhere:

```
  score >= 3.00  n= 576  56 users  ->  q=4.382 47% | q=4.533 39% | q=4.333 3%
  score >= 3.30  n= 254  29 users  ->  q=4.382 93% | q=4.333  3% | q=4.312 1%
  score >= 3.50  n= 102  21 users  ->  q=4.382 95% | q=4.575  2% | q=4.566 1%
  score >= 3.65  n=  63  19 users  ->  q=4.382 98% | q=4.667  2%
  score >= 3.70  n=  24  13 users  ->  q=4.382 100%
```

Thirteen distinct users reached 3.70 and every one of them ran `q = 4.382` on
beagle. The two large bins also compare fairly at comparable sample size
(n=276 against n=230) and `q=4.382` wins by 13.2 %. The best alternative
anywhere on the board is `q=4.575` at +2.03 % worse `mtp` (n=2, kirtangajjar,
score 3.634). The weighted-prompt view agrees to within 0.02 points.

FINDING 313 stands. **With the shipped head there is no schedule move that
wins.** This justifies cancelling E159 Part B and it is the floor under
E158's envelope argument.

One caveat to carry: the 13.2 % gap between `q=4.382` and `q=4.533` is far too
large to be a pure schedule effect and is confounded with codebase generation.
The confound does not threaten the conclusion, which is a negative: across 56
users no other schedule setting ever reached the top tier.

---

## FINDING 317 — the five zero-draft receipts, and what runtime alone is worth

A receipt with `q = 0` on all eight prompts proposes nothing, so proposed =
accepted = 0, `rounds = 512`, and `R = mtp` **exactly**. No assumption. There
are exactly five such receipts on the board.

```
id        user          commit    date        mean R      mean serial
95611e60  davidtai                2026-08-14  0.0380252   0.0380302
3147d255  0xkydo        b256d4c2  2026-08-15  0.0380728   0.0379366
4d1123ce  newjordan     d330300b  2026-08-15  0.0378805   0.0379743
1f60b3fe  newjordan     0c0d6a00  2026-08-15  0.0378079   0.0380045
da950333  fabinulleins  d25821d7  2026-08-17  0.0313628   0.0379709
```

The first four sit on the pinned serial baseline, as an unoptimised target path
must. `da950333` is 17.4 % faster than that baseline, with a per-prompt spread
of 0.2 %. It passed every correctness gate (`parity_all_ok: true`, 8/8 accepted
pairs, 512 decode tokens); its `rejected` status is only
`score did not improve current best`.

**With drafting switched off entirely, an optimised target path runs 17.4 %
faster than the pinned serial baseline.** That is the cleanest model-free
measurement of what pure runtime work is worth on this benchmark, and it is the
only one on the board.

Replicated near-zero-draft cluster, consistent with it: scarletbright
`c91581eb` 0.0325240 and `baa75efa` 0.0325299, jungjipdo `26d0e934` 0.0325820.

---

## FINDING 318 — acceptance is strongly prompt-dependent, so no pooled model fits

I fitted the natural three-parameter round model across the eight prompts of a
single receipt:

```
mtp_p = (s + h*q_p) / (1 + alpha*q_p)
```

with `s` the fixed per-round cost, `h` the marginal cost of one proposed draft
row, and a single acceptance fraction `alpha` shared across prompts.

It fails, and it fails informatively. Every fit drives `h` **negative**, which
is physically impossible, and leaves 4.8 % mean residual:

```
5a9f130a  alpha=0.2535  s=0.0308020  h=-0.00149624  resid 4.87%
ec24d591  alpha=0.2610  s=0.0308201  h=-0.00143345  resid 4.80%
ac89ef87  alpha=0.2600  s=0.0307480  h=-0.00144001  resid 4.81%
```

The misspecification is the shared `alpha`. Acceptance quality varies enormously
across the eight prompts — the cost model already tells us so, since it chooses
`q = 0.156` on plutarch and `q = 6.148` on botany from the same constants.
**Do not pool acceptance across prompts in any model.**

---

## FINDING 319 — the rigorous feasible set

Drop every modelling assumption and keep only what is true by definition:

- `mtp_p * (1 + a_p) = R_p = s + h*q_p`, with `s` and `h` shared across the
  eight prompts of one receipt because per-round cost is a runtime property,
  not a prompt property.
- `h >= 0`. Proposing a draft row cannot make a round faster.
- `0 <= a_p <= q_p`. You cannot accept more drafts than you proposed.
- `N_p = 512/(1 + a_p)`, and `q_p * N_p <= 8 * (N_p - nd_p)`, because a
  drafting round proposes at most eight.

Given `(s, h)` every `a_p` is determined, so the feasible set is an
intersection of half-planes and can be enumerated exactly. For our anchor
`5a9f130a`:

```
s in [0.0292500, 0.0336500]      h in [0, 0.0057778]      8h/s in [0, 1.580]

prompt          q        mtp       a_lo       a_hi     N_lo     N_hi
plutarch    0.156  0.0301213     0.0000     0.1181    457.9    512.0
drama       2.298  0.0178803     0.6862     1.4855    206.0    303.6
travel      2.648  0.0156379     0.9280     1.9554    173.2    265.6
beagle      4.382  0.0106961     1.8188     4.1422     99.6    181.6
medicine    5.256  0.0097036     2.1071     5.1436     83.3    164.8
republic    4.989  0.0096975     2.1091     4.9891     85.5    164.7
essays      5.087  0.0098233     2.0692     4.9696     85.8    166.8
botany      6.148  0.0096485     2.1248     5.7132     76.3    163.8

total rounds over the 8 prompts: [1263, 1917] for 4096 emitted tokens
```

Two things are now known that were not known before, and both are unconditional:

1. **`a_beagle >= 1.816`.** At least 1.82 of the 4.382 proposed drafts are
   accepted on the highest-weight prompt, so the acceptance fraction is at
   least 41.4 %.
2. **`s in [0.02925, 0.03365]`,** a +-7 % bracket on the fixed per-round cost.

What is still *not* known is the split between `s` and `h`, and therefore the
share of candidate-leg time that drafting consumes. That share runs from 0 % to
46 % across the feasible set. The board cannot narrow it further.

---

## RULE 182 — the board is exhausted on the round budget

The ranked receipt publishes proposed depth and never publishes an accepted
count. Two independent attacks now confirm the consequence:

- No solver has ever submitted a zero-draft receipt and a drafting receipt on
  the **same commit**. I checked all 1,255 submissions. Such a pair would have
  measured `s` and `a` together; it does not exist.
- The feasible set above is as tight as definitional constraints allow.

**Do not attempt a third time to extract the round budget from the board.**
The remaining unknown is `8h/s`, and only a local measurement can supply it.
That measurement is E159.

---

## FINDING 320 — a pre-registered decision table for E159

Because the feasible set is one-dimensional once `rho = 8h/s` is fixed, E159's
answer cashes out immediately and board-wide. Recorded **before** the
measurement so the decision cannot be rationalised after the fact:

```
   rho | s (fixed/round)   | a_beagle        | beagle rounds | draft% | tot rounds
  0.00 | 0.030121-0.033679 |  1.816 -  2.149 |   163 -   182 |   0.0% | 1721-1924
  0.30 | 0.029946-0.033484 |  2.260 -  2.645 |   140 -   157 |  14.1% | 1563-1747
  0.60 | 0.029774-0.033290 |  2.698 -  3.135 |   124 -   138 |  24.7% | 1443-1614
  0.72 | 0.029705-0.033214 |  2.872 -  3.330 |   118 -   132 |  28.3% | 1404-1569
  1.00 | 0.029546-0.033036 |  3.275 -  3.780 |   107 -   120 |  35.4% | 1323-1480
  1.44 | 0.029300-0.030600 |  3.900 -  4.117 |   100 -   104 |  44.1% | 1311-1369
  1.58 | 0.029223-0.029254 |  4.096 -  4.102 |   100 -   100 |  46.4% | 1338-1339
```

Drafting share of the candidate leg, weighted by the FINDING 306 median
weights (`q_weighted = 4.797`):

```
  rho = 0.30  ->  15.2 %      rho = 0.72  ->  30.2 %      rho = 1.44  ->  46.3 %
```

E159's pre-agreed decision rule maps onto `rho` as:

- `rho < 0.32` — drafting is under 15 % of the leg. **Widen producer fusion.**
- `rho` in 0.32 to 0.95 — **run both** mechanisms.
- `rho > 0.95` — drafting is over 35 % of the leg. **Pivot to head cost.**

### Pre-registered prediction: rho lands in 1.2 to 1.6

FINDING 303 records that all three `costModelDepth` constructors hold the total
depth price at `8h = 1.44` in units where the fixed cost is 1. That constant was
chosen by the organizer's calibration, and it is *independent* of the board
`mtp` and `q` values used to derive the feasibility ceiling of 1.580.

An arbitrary constant has no reason to land 91 % of the way to a bound derived
from disjoint data. I therefore predict E159 measures `rho` between 1.2 and 1.6,
and I am recording that prediction now so a miss is visible.

**If `rho` is near 1.44, the strategic picture changes:**

- `a_beagle` is 3.90 to 4.12 out of 4.382 proposed — acceptance is already
  **89 % to 94 %** on the highest-weight prompt.
- There is then almost no headroom in head *quality* on the prompts that carry
  the score, and E158's island-free head should be judged on whether it makes
  drafting **cheaper**, not more accurate.
- Drafting would be 44 % of the candidate leg, so head cost becomes the single
  largest addressable block, larger than the entire producer-fusion family.

**If `rho` is below 0.4,** acceptance is near 42 %, head quality has real
headroom, and E158's accuracy work is the right lever while producer fusion
carries the runtime side.

The two branches point at opposite experiments, which is what makes E159 worth
a GPU day.

---

## RULE 183 — state which constraints are definitional

Every bound in FINDING 319 comes from an identity or an inequality that cannot
be violated by any implementation: token arithmetic, non-negative cost, and the
depth ceiling. Ledger 333 failed because a *plausible reading* of a field name
entered the arithmetic with the same authority as an identity.

When recording a bound, mark each constraint as **definitional**, **measured**,
or **assumed**, and give the source line or receipt ID for the second and third
kinds. A bound built only from definitional constraints survives a change in
any model; a bound with one assumed constraint dies with that assumption.

---

## Carried forward

- FINDING 316's `da950333` remains the only exact `R` on the board, and our
  anchor's `s` bracket contains it at every `rho` below about 1.3. We still
  cannot say whether our runtime beats it. At `rho >= 1.44` we provably would.
- FINDING 314 stands: eleven distinct heads have been submitted, none beats
  `559b24eb` on the scoring prompts, and both our anchor and the crown run it
  with byte-identical per-prompt `q`.
- The plutarch instrument stands and is sharper than ledger 333 claimed:
  `a_plutarch <= 0.118` unconditionally, so plutarch's `mtp` reads the fixed
  per-round cost to within 12 % and reads it for free in every receipt.

---

## RULE 184 — Yukon validation takes 84 minutes at the median. Do not call it stuck.

I have relaunched the receipt watcher three times on `1509bf95` out of
impatience. Measured across all 1,246 terminal submissions on the board:

```
  n=1246  min 0.1  p10 30.7  p25 54.8  median 83.9  p75 117.6
          p90 160.6  p95 177.1  p99 204.5  max 335.4     (minutes)
  over 60 min: 70.8 %    over 80 min: 53.2 %    over 100 min: 36.4 %
```

Our own last 13 submissions averaged 86 minutes, and the anchor `5a9f130a`
itself took 98.3 minutes. When `1509bf95` was created at 18:20:45Z there were
**eight** submissions already validating ahead of it, created between 16:48 and
17:57Z. A queue of nine is the normal state of this board.

**A submission is not stuck until roughly 3.5 hours** (p99 is 204 minutes).
Relaunch the watcher when it reaches its deadline, but do not spend advisor
turns investigating before then, and never resubmit to test whether the first
one is alive.

## RULE 185 — `status` never means gate failure. Read `rejectionReason`.

On this board `accepted` means the run improved a best score, not that it
passed the correctness gates. Our anchor `5a9f130a` scored 3.70785, is marked
`rejected`, and passed every gate; its `rejectionReason` is
`score did not improve current best`. FINDING 316's `da950333` is the same
case. Board-wide the vocabulary is `rejected` 827, `failed` 283, `accepted` 99,
`cancelled` 37, `validating` 9. **`failed` is the gate-failure label.**
