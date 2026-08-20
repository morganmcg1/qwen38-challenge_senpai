# E62 rung 1 — design changes made before spending legs

## 1. The ladder runs at 512 tokens, not 256

The planned split was a cheap 256-token ladder (rung 1) followed by a
512-token confirmation of the winner (rung 2). Leg timing kills that split.

A leg's wall time is dominated by fixed cost, not by the decode window. The
64-token census leg ran 01:17:47 -> 01:19:40, 113 s wall, of which the two
timed decode phases account for

```
mtp    64 x 0.0874 = 5.6 s
serial 64 x 0.1303 = 8.3 s
                    ~14 s of ~113 s
```

Model load, transform, eight-depth shape warm and seed prefill are the rest.
Halving the decode window from 512 to 256 removes roughly half of the decode
seconds only, which on a ~4 min stock leg is on the order of 10 % of wall
time, while it costs about sqrt(2) in per-token timing noise.

Paying 40 % more noise to save 10 % of the clock is a bad trade, and 256-token
results are not ranked-equivalent under `program.md` anyway. So rungs 1 and 2
merge into one 512-token session that is directly comparable to the
`0.0340649975114502` anchor.

## 2. The primary screen is a ladder slope, not seven pairwise contrasts

Seven arms with one replicate pair each is weak for any single contrast: the
E60 same-arm spread at wide separation is about 0.16 %, against a minimum
useful effect of 0.15 %. But the physical question is whether decode time
*tilts* with commit granularity, and every leg informs that one slope.

`research/e62_analyze.py --trend-mb 4096` therefore fits

```
mtp_seconds_per_token ~ log2(OPS) + leg_position
```

across the constant-MB ladder and reports the slope with a 95 % CI, as percent
per doubling and across the full ladder span. The per-arm contrasts from
`regress` stay in the report but are descriptive.

## 3. Predicted geometry, from this session's own census fit

With the in-session fit `ops_per_dispatch = 1.6153` and the E60
`mb_per_dispatch = 10.58`:

```
dispatches_per_commit = min(OPS / 1.6153, MB / 10.58)
```

At `MB = 4096` the byte cap is 387 dispatches/commit, so it is non-binding for
every OPS point on the ladder. The ladder is purely OPS-driven by construction:

| arm | MB | OPS | predicted dispatches/commit |
|---|---|---|---|
| C | 4096 | 6 | 3.7 |
| D | 4096 | 12 | 7.4 |
| E | 4096 | 25 | 15.5 |
| B | 4096 | 50 | 31.0 (null control) |
| F | 4096 | 100 | 61.9 |
| G | 4096 | 200 | 123.8 |
| A | 512 | 50 | 31.0 (shipped) |

The ladder spans 3.7 to 123.8 dispatches per commit, a 33x range, with the
shipped geometry sitting at 31.0 near the middle.

**A vs B is the null control.** At the shipped `MB = 512, OPS = 50` the
measured census value was 30.954 dispatches/commit and the model gives
`min(30.95, 48.4) = 30.95`. OPS binds and MB does not, so A and B are the same
geometry reached by two different routes and must time the same. If A vs B is
not null, the instrument is wrong and the rung is void.

## 4. Order

One declared discarded warm-up leg, then the 14-leg palindrome

```
A B C D E F G G F E D C B A
```

Every arm has mean position 7.5, so monotone drift cancels in the arm means,
and `leg_position` absorbs the linear component in the fit. Note the
asymmetry this leaves: arm A's two legs sit 13 apart and arm G's sit 1 apart,
so A carries more drift variance than G under any *non*-linear drift. The A/B
null control is what detects that, because A and B share the same wide
position structure.

Estimated cost: 15 legs at about 4 min, roughly 60 min.

## 5. Residency regime

Every leg in this session runs on the stock arm on a 48 GiB host, so the
`:225` gate is false and `wired_residency_active=false` throughout, while the
ranked M5 runs with wiring on. See `rung1b-scope-finding.md`. If the ladder
produces a winner, that winner is re-checked once with wiring forced on before
any shipping claim.
