# E110: one-group wide QMV activation reread

**Result: invalid as a speedup. `xv4` is correct and 1.20 % slower on the
ranked M5.** The official receipt rejected it. The scientific value of the
experiment is the transfer failure it exposed, not the mechanism.

- Arm `xv4`, submission `7bef7d4c-7c7b-4969-bce5-e3065b0bcbfe`
- Official score **3.29792432850592**, serial-free **3.29427211**, rank
  **116 / 764**
- `rejectionReason = score did not improve current best` — a score rejection,
  not a gate failure
- `parity_all_ok = true` on all eight hidden prompts

## The mechanism

One function changed, `qmv_fast_crossrow_affine4_g64_wide` in
`Vendor/.../kernels/quantized.h` and its `mlx-generated` twin. Inside the
`DIRECT_NIBBLES` branch, four scalar `device` reads of the activation operand
became one 64-bit `vec<T, 4>` read:

```
const vec<T, 4> xv = *reinterpret_cast<const device vec<T, 4>*>(xm);
```

The affine bias correction keeps the incumbent BF16 expression tree, so the
arithmetic is unchanged. Exactness held locally over 512 tokens and again on
all eight hidden prompts.

The kernel is the **wide cross-row** path. It is reached only when more than
one row is verified, so it runs only when the candidate drafts. That fact
drives the whole result and it was mis-stated in the assignment, which framed
`xv4` as a target-path change.

## What the local host said, and what the ranked host said

| harness | measurement | value |
| --- | --- | --- |
| local, M4 Pro `applegpu_g16s` | matched ABBA, n = 6, 24 legs, absolute candidate MTP s/token | **−0.7498 %**, CI [−1.0393, −0.4602] |
| ranked, M5 `applegpu_g17s` | official aggregate candidate time vs `b8b8b860` | **+1.2032 %** |
| ranked, M5 | official mean over the seven drafting prompts | **+1.5227 %** |

The sign flipped. The local-to-ranked swing is **2.27 percentage points** on
the drafting prompts.

## Causal attribution: dose response on drafting

| group | n | mean vs `b8b8b860` | range |
| --- | --- | --- | --- |
| drafting, `non_drafting_round_count = 0` | 7 | **+1.5227 %** | +1.0605 .. +2.0385 |
| non-drafting, `plutarch`, 449 non-drafting rounds, draft len 0.154 | 1 | **+0.1605 %** | — |

The one prompt that almost never enters the edited kernel is unaffected and
below the measurement floor. Every prompt that does enter it is slower, with
the same sign. Percentages are log-percent, matching
`board_prompt_instrument.py`.

## The pre-registered prediction is falsified

The advisor pre-registered the g17s NA2 (83→93) and NA5 (98→101) register
regression as the first suspect for any ranked underperformance. Grading the
full census against the realised verify-width histogram refutes it.

| width | round weight | R g16s | R g17s | spill | device loads | AIR lines | ΔΩ ranked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NA2 | 0.024 | 70→66 | 83→93 | 0 | 7→4 | 292→291 | +0.1515 % |
| NA3 | 0.275 | 93→82 | 90→89 | 0 | 7→4 | 294→293 | 0.0000 % |
| NA4 | **0.667** | 94→94 | 91→90 | 0 | 7→4 | 294→293 | −0.0309 % |
| NA5 | 0.034 | 95→93 | 98→101 | 0 | 7→4 | 294→293 | +0.0341 % |

Weighted: Δregisters g17s **−0.60**, ΔAIR lines **−1.00**, ΔΩ ranked
**−0.0158 %** of QMV kernel time, zero spill everywhere. The two regressing
widths carry a combined weight of 0.058, and the dominant width NA4 improves.

**Every zero-GPU predictor the campaign owns pointed the wrong way**: registers,
spill, occupancy Ω, AIR instruction count and device-load count were all
neutral or favourable on the ranked architecture. The occupancy channel
predicts −0.016 % and the machine delivered +1.52 %, which is 96× too small
and the wrong sign.

E77 states that "the entire occupancy spread available anywhere in this table
is 0.52 %", so the observed 2.27 pp swing is 4.4× everything the occupancy
model can produce. E77's cross-host flip mechanism cannot explain it either.

## Standing conclusion

Replacing scalar `device` loads with a wider vector load is a memory-system
change. Its cost on `applegpu_g17s` is carried by something none of our static
instruments measure, most plausibly load-width, alignment or coalescing
behaviour that differs from `applegpu_g16s`. This cannot be resolved from
zero-GPU data.

**Treat device-memory load-width changes as untransferable from the local host
until an M5-side probe exists.** This is the campaign's first measured
local-to-ranked sign flip that no predictor anticipated.

## Measurement floor defect found while reading the receipt

`board_prompt_instrument.py --noise` measures the same-mode TARGET floor at
**0.5655 %** per run against a constant in use of **0.0945 %**, so the TARGET
resolution constant is 6.0× optimistic.

| probe | vs `b8b8b860` | floor in use | floor measured | σ measured | σ as printed |
| --- | --- | --- | --- | --- | --- |
| TARGET | +0.1605 % | 0.0945 % | 0.5655 % | **+0.28** | +1.70 |
| DRAFT | +1.4515 % | 0.0952 % | 0.1780 % | **+8.15** | +15.25 |

TARGET is null once the measured floor is used, and it flips sign between the
two reference submissions. DRAFT survives either way. The constants are in a
shared tool, so this experiment did not change them; the finding is reported
to the advisor for arbitration.

## Reproduction

```bash
YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
python3 research/board_per_prompt.py serialfree 7bef7d4 b8b8b860 44559d02
python3 research/board_prompt_instrument.py --read 7bef7d4 b8b8b860
python3 research/board_prompt_instrument.py --noise
python3 research/e110_rung4_receipt.py
python3 research/e110_wandb_log.py --only rung4
```

W&B: `e110-rung1-census` `iystdffm`, `e110-rung1-timing` `bxo8zizf`,
`e110-rung2-insitu` `uxnv7gzr`, `e110-rung3-presubmit` `tt0o2csj`,
`e110-rung4-official-receipt` `mir9heyp`.

## Follow-ups, not implemented

1. Update the TARGET and DRAFT resolution constants in
   `board_prompt_instrument.py` from the measured columns.
2. Build an M5-side probe for device-memory load width before any further arm
   of this shape is proposed.
3. Revert `xv4`. It is correct but costs 1.2 % of ranked candidate time.
