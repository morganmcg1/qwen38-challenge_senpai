# E44 r2 — method record for the narrow M ∈ {7,8} arm

This file records the parts of the r2 method that are fixed independently of the
timing outcome, so the outcome cannot retro-fit them.

## What changed relative to r1

r1 dispatched the `simdgroup_matrix` cell across the whole `M ∈ {4..9}` range and
was refuted: **−7.341 %** on net over the replaced widths, with 7 of 12 replaced
cells regressing with resolved intervals. r2 keeps the identical kernel construct
and narrows the dispatch to **M ∈ {7,8}** only, which were the two widths that
won in r1. `M ∈ {4,5,6,9}` are returned to their base scalar cells.

The rationale comment in the cell was also corrected. r1 justified the construct
by weight-stream halving; the r1 flat-cost measurement refuted that mechanism and
identified the real one (a fixed 8-row MMA cost, flat in M, which beats the
scalar cells only once the scalar per-row cost has risen far enough). The comment
now states the mechanism that the data supports.

## Base and provenance

- `BASE_SHA` = `9fe0dc5dbdb30af4c807ea71873df99e2da72aa2` (r1 used `efff400c`).
- The shipped-surface diff `efff400c → 9fe0dc5d` over
  `Sources Vendor benchmark.json mlx-generated` is **empty**, so r1's per-cell
  register table transfers to this base without remeasurement. The rebase was
  verified conflict-free.
- Host: Apple M4 Pro, `applegpu_g16s`, 48 GiB, Apple metal 32023.883.
  The ranked box is M5 Max, `applegpu_g17s`.

## Pre-registration discipline

`research/e44-r2-prereg.md` was committed in `502f715` **before the narrow
variant was compiled**, and posted to the PR. It fixes, in advance:

- the Gate A register bound and the predicted binding cell;
- the Gate B pass condition;
- the Gate C design (9 pairs, df = 8) and the requirement for an A/A control;
- the two-term score decomposition and its f-sensitivity table;
- the power caveat for the ceiling term.

## Resolution floor

The honest floor is the one the measurements actually achieve, not the one that
was hoped for. r1 pre-registered an MDE of 0.5040 % but its zero-effect guard
cells showed a worst |effect| of **0.628 %**. r2 therefore uses 0.628 % as the
floor, and additionally runs a dedicated **A/A control**: a session with the same
design in which both arms are built from the base sha, so the true effect is
exactly zero at *every* width rather than only at the three cheap guard widths.
If the control's worst |effect| is larger than 0.628 %, the control's number is
used instead. The summary tool refuses to accept a control whose two arms are not
byte-identical.

## Why the two score terms are reported separately

The candidate changes two things at once, and they push the score in **opposite
directions**:

- **Width term** — the MMA cell replaces the scalar cell at M ∈ {7,8}. These
  widths exist only on the MTP leg; the serial leg never dispatches them. With
  ψ_mtp = 0.6736, a 1 % reduction in MTP-leg QMV cost is worth **+0.674 %** of
  score. This term is **favourable** and is powered.
- **Ceiling term** — `qmv_fast` has a single `[[kernel]]` entry point and one
  shared register allocation taken as the max over all instantiated cells, so
  changing that ceiling perturbs **every** width including M = 1. Both legs speed
  up, but the serial leg is the more QMV-dominated of the two (ψ_serial = 0.8525
  vs ψ_mtp = 0.6736), so a uniform speedup helps the denominator more than the
  numerator. A 1 % uniform QMV reduction is worth **−0.179 %** of score. This
  term is **adverse**.

Aggregating them would hide the sign conflict. They are reported separately, each
with its sign attached, and the width term is additionally reported as a
sensitivity table over `f`, the share of MTP-leg QMV cost dispatched at the
touched widths, because E43 established that `f` is **not identified**.

## Power caveat, stated before the numbers existed

The ceiling term is **not powered by this design and cannot be**. E40 priced the
ceiling at 0.00959 % of cost per 1 % of register. The narrow arm's predicted
ceiling move is 108 → 104 = −3.70 %, worth about **0.0355 %** of score. Resolving
0.0355 % against a 0.628 % floor needs on the order of **880 pairs**. This design
runs 9. The ceiling term is therefore reported as a **bound**, never as a
measurement, and no claim is made that it was observed.
