# E54 blind prediction for P1, written while P1 was still running

Committed before any P1 leg output was read. The P1 job was launched first, and
this note uses only E49's already-merged legs, PR #8, E48, E53 and the E27 board
anchor. It changes no pre-registered prediction in
`research/e54-artifacts/e54-prereg.json`, whose digest stays
`a8b62d8d0a8606423b6c51f1ec41054b7f0fea0c9f4067abbb46b8bb6792291c`. It replaces
one *derived* threshold inside the Law C branch with a measured one, and it moves
against the author's own preferred law.

## What changed, and why it is a correction rather than a revision

The pre-registration priced Law C's requirement as "the lone NA=5 group must
sustain < 82.8 GB/s", taking 165.6 GB/s from PR #8 as the rate `<T,5,3>`
achieves and halving it. That reference belongs to another host and to a
single-group configuration.

E49's shipped-table legs measure the right quantity directly on this host.
`research/e54_bandwidth.py` reads it out of those legs:

| working groups | group NA | achieved device rate |
|---|---|---|
| 1 | 2 | 222.3 GB/s |
| 1 | 3 | 199.3 GB/s |
| 1 | 4 | 175.2 GB/s |
| 2 | 3, 2 | **240.4 GB/s** |
| 2 | 3, 3 | 224.7 GB/s |
| 2 | 4, 3 | 208.4 GB/s |
| 2 | 4, 4 | 194.2 GB/s |
| 3 | 3, 3, 3 | 233.4 GB/s |

Two facts follow, and they pull in opposite directions.

1. **A lone group gets slower as its NA grows**, and almost exactly linearly:
   −23.0 then −24.1 GB/s per extra row. Extrapolating one step gives a lone
   NA=5 group **151.7 GB/s**. This is Law C's mechanism visible in existing data.
2. **Adding a second working group raises the achieved rate a lot.** `[3,2]` at
   240.4 GB/s beats lone NA=3 at 199.3 and lone NA=4 at 175.2. Concurrency, not
   width, is what fills the memory system.

Because `<T,5,3>` achieves 240.4 GB/s rather than 165.6, and `<T,5,5>` moves half
its traffic, the true break-even for the lone NA=5 group is
**120.2 GB/s**, not 82.8. That is a much easier target for Law C, so this
correction *helps* the advisor's prediction and hurts Law A′.

## The convergence

| route | inputs | lone NA=5 rate |
|---|---|---|
| extrapolated lone-group curve | E49 legs, this host | **151.7 GB/s** → P1 = **−20.7 %** |
| break-even | E49 legs, this host | **120.2 GB/s** → P1 = 0 % |
| E27 board anchor + E48 shares + E49 M=9 win | board, E48, E49 | **95.5 GB/s** → P1 = **+25.8 %** |
| direct measurement | PR #8, another host | **95.5 GB/s** |

The third and fourth rows agree to better than 0.1 GB/s, and nothing in the
third row's arithmetic uses a bandwidth figure. `solve_m5_delta` consumes only
the board score −0.3321 %, E48's width shares, the measured M=9 win +12.255 %
and the pinned leverage module; the conversion to GB/s then uses only this host's
measured break-even. So the agreement is not circular.

Under E53's mixture the same inversion needs 113.0–114.1 GB/s, which is 18–20 %
above PR #8. So this also separates the two mixtures.

## The prediction

**P1 regresses by roughly +25.8 %, and Law C is right.** The author's
pre-registered preference was Law A′ at −19.0 %. Recording the opposite
conclusion here, before the measurement, so the pre-registration cannot be
reinterpreted afterwards.

Sharper than its inputs deserve: E27's board score, the width shares and PR #8's
single-host figure all carry uncertainty that the two-decimal agreement hides.
Treat it as a strong convergence, not a proof. The measurement decides.

## What each outcome would then mean

- **P1 near +25.8 %** — Law C holds, E27 is fully explained by its own two
  halves, and E48's mixture is favoured over E53's. A whole class of
  "fewer weight streams is better" edits is unsafe when the saving leaves one
  group running alone.
- **P1 near −20.7 %** — the lone-group curve extrapolates correctly, Law A′
  holds, and E27's residual belongs to a mechanism nobody has named. E49 already
  bounded the shared-allocation tax at 0.213 % of QMV, far too small to cover it.
- **P1 inside the bar** — both models fail at this cell and the group-count story
  needs rebuilding.
- **P2 and P3** stay the constant-traffic test either way: A′ says null, the
  critical-path model says a large regression, Law C as briefed says a large win.
