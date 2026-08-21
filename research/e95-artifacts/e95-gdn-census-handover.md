# E95 -> GDN follow-up: how to measure the Gated DeltaNet dispatches directly

Author: qwen-askeladd (E95). Reader: the agent who owns the GDN follow-up.
No GPU work was done for this note. It only tells you how to run the two
existing instruments and which of their fields are direct measurements.

E95 reported that the GDN family is 85.1 % of the fixed verify term `a` and
that the GDN recurrent step costs 6.8x its DRAM expectation. That number came
from `fixed` mode, which is a **model output**, not a measurement. Do not
start from it. Start from the direct per-dispatch path below.

## 0. Re-seat the census instrument

The instrument is research-only and is **not** in the final E95 tree. Apply it
on top of your own base:

```bash
git apply research/e95-artifacts/e95-census-instrument.patch
```

It adds `Sources/MLXFastModel/E58DispatchCensus.swift` and three call sites.
Never submit it. Revert it before any timing leg you intend to publish as a
gate-qualified result, and before any submission scope check.

## 1. Run one leg with one MLX op per command buffer

```bash
research/e93_gputime_leg.sh gdn-iso 5 384 1
```

Arguments are `TAG DRAFTS TOKENS OPS_PER_BUFFER`. The verify width is
`M = DRAFTS + 1`, because the target checks the pending primary token plus
every draft in the same forward pass. Output lands in `research/out/gdn-iso/`.

Launch it through `run_job`, not the terminal. A 384-token leg takes about
3.5 minutes plus the build.

`OPS_PER_BUFFER=1` sets `MLX_E58_BUFFER_LIMIT_OPS=1`. That is the field that
decides what you can measure:

| setting | one command buffer contains | `exclusive_kernels` | what a buffer interval means |
| --- | --- | --- | --- |
| unset (default) | many MLX ops | empty | in-situ total for a mixed group |
| `1` | exactly one MLX op | populated | one kernel's own GPU time |

`MLX_E58_BUFFER_LIMIT_OPS=1` removes intra-buffer concurrency, so every
isolated kernel time **over-states** the kernel's in-situ cost. The ratio of
the isolated total to the in-situ total is the concurrency discount. E95
measured that discount for the whole verify phase; you must measure it again
for the GDN subset, because the discount is not uniform across families.

Run the unset-geometry leg too, so you have both totals:

```bash
research/e93_gputime_leg.sh gdn-insitu 5 384
```

## 2. Read the direct tables

```bash
research/e95_verify_census.py kernels research/out/gdn-iso/census.jsonl --width=6
research/e95_verify_census.py buffers research/out/gdn-insitu/census.jsonl --width=6
research/e95_verify_census.py gdn     research/out/gdn-iso/census.jsonl
```

`kernels` prints per-kernel isolated GPU microseconds per dispatch and per
round, taken from `exclusive_kernels`. This is the table you want. The GDN
dispatches appear under these shape prefixes:

- `custom_kernel_gated_delta_step__` — the recurrent step, 48 per round.
- `custom_kernel_qwen35_gated_delta_step_mid__` — the `Qwen35.swift` clone.
- `custom_kernel_qwen35_gated_delta_replay_state__` — the replay/rollback path.
- `custom_kernel_qwen35_packed_gdn_prework__` — causal conv1d, q/k norm, gates.

`buffers` prints the per-command-buffer table. One command buffer is one
measured GPU interval, so this table needs no solver and no identifiability
argument. It is the primary evidence for the in-situ geometry.

`gdn` audits the recurrent state itself: the `[48,128,128]` fp32 state that is
snapshotted and rolled back on every speculative round.

## 3. Which fields are direct and which are modelled

Direct measurements, from Metal's own clock:

- `gpu_busy_ns`, `gpu_span_ns`, `gpu_idle_ns` per snapshot.
- per-buffer `gpu_ns` behind `buffers`.
- per-kernel `gpu_ns` and `buffers` behind `kernels` (isolated geometry only).
- dispatch counts everywhere. These are exact integers.

Model outputs, computed by `research/e95_verify_census.py` from the model
geometry and the affine-4 group-64 packing:

- every `MB/disp` and `GB/s` column. The byte counts are derived, not observed.
  They are checked by construction: the per-layer sums reproduce the
  organiser's 14,412,349,440-byte weight stream exactly, but they are still a
  model of what the kernel reads.
- everything printed by `model`, `rowcost`, `step` and `fixed`. These solve a
  least-squares or difference system over widths. A residual there can mean
  the model is wrong, not that the hardware is slow.
- the per-class split in `fixed`, including the 85.1 % GDN share and the
  6.8x-over-DRAM reading. E95 flagged this split as a named hypothesis with a
  stated instrument limitation, not as ledger fact.

`counts` is a third category: an exact integer model of dispatch counts per
width. A residual there means the model is wrong rather than noisy.

Host wall clock is invalid in every census geometry, because the census lock
serialises every dispatch. Host `dispatch_ns` is encode cost and is never a
duration.

## 4. The open discrepancy you are inheriting

E95's `fixed` mode gave a GDN recurrent-step series that **falls** with the
verify width M:

| M | GDN step, us/round (modelled) |
| --- | --- |
| 3 | 9493.8 |
| 4 | 8585.5 |
| 5 | 8112.6 |
| 6 | 6492.8 |
| 8 | 7104.0 |
| 9 | 6768.3 |

Linear slope about -428.3 us per row. The kernel body loops `for (int t = 0;
t < T; ++t)` over the row count, so the true cost cannot fall as rows are
added. Either per-row GDN cost is being absorbed into the `c*M` term of the
width model, or the per-class residual split is noisy. The isolated
per-dispatch table in step 2 settles this without a solver: measure
`custom_kernel_gated_delta_step__` microseconds per dispatch at M = 3, 5 and 9
and check whether it rises with M.

A census leg is never a gate-qualified timing leg. It sets
`MLXFAST_LOCAL_COOL_GATE=0` and the census swizzle serialises every dispatch.
Report `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
verbatim.
