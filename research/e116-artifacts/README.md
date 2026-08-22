# E116 artifacts

`harness=local` on every file. No number in this directory is a ranked or
official score. Every timed leg carries `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false` under the
standing counterbalanced exception.

The report that reads these files is `research/e116-results.md`.

Host `ip-10-231-2-12.ec2.internal`, Apple M4 Pro, `AGXG16SDevice`, 48 GiB.
Base `67fedb4adb4cb0ec757f870ec8093617ca1e5620`.
Measurement worker `4b90ff2251d93714bf95699c4d67540663492ed8c99b5acc31558ef0bc445e3c`.

## Headline

```
e116_wide_qmv_pct_to_leg_pct_transfer = 0.6070   95 % CI [0.5843, 0.6297]
                        standing value  0.615
```

| file | what it holds | key fields |
|---|---|---|
| `rung1-dose-rate.json` | the dose cell, its rate and its resident bytes | `dose_unit_us`, `dose_cell`, `resident_bytes` |
| `rung1-qmv-share-64tok.json` | the 64-token census share, per width | `frames.mtp_round.wide_qmv_share` |
| `row-digest-512.json` | the 512-token row-evidence digest and its three controls | `digest`, `controls` |
| `rung0b-round-switch-witness.json` | one witness line per round, zero width-1 lines | `round_alignment_verified`, `failures` |
| `rung2-absorption.json` | `alpha`, both estimators, the null arm, temperatures | `alpha`, `alpha_ci95` |
| `rung3-transfer.json` | the dose ladder, `beta`, `alpha x beta` and the composition | `round_to_leg_alpha_times_beta`, `composed_kernel_percent_to_leg_percent` |
| `rung4-qmv-share-512.json` | the 512-token realised-width census | `frames.mtp_round.wide_qmv_us_per_round` |
| `rung4-reprice.json` | every standing kernel arm at the measured transfer | `arms`, `xv4_cross_check` |
| `cleanup-selector-defect-witness.json` | the dispatch entry-point defect and its witness | `dispatches_by_entry` |
| `instruments.patch` | restores every deleted research instrument | see below |

## Reproduction

```bash
research/e116_rung1.sh                        # dose rate and the 64-token share
research/e116_rung1_exact512.sh               # the row digest and its controls
research/e116_rung2.sh 411.86                 # alpha
research/e116_rung3.sh 411.86 1.177260429936175   # beta and the ladder
research/e116_rung4_census.sh                 # the 512-token realised-width census

python3 research/e116_transfer_report.py research/out/e116r3-ladder/b[123]-* \
  --dose-unit-us 411.86 --alpha 1.177260429936175 --alpha-half-width 0.06345 \
  --wide-qmv-us-per-leg 10400273.764 \
  --json research/e116-artifacts/rung3-transfer.json

python3 research/e116_reprice.py --transfer 0.6070 \
  --transfer-lo 0.5843 --transfer-hi 0.6297 \
  --json research/e116-artifacts/rung4-reprice.json
```

## `instruments.patch`

The patch restores every research instrument that the cleanup commit deleted:
the E58 dispatch census and its install hook, the eight `phase(...)` calls,
`beginRound`, `endRound`, `fireTax`, `censusAccepted`, the `forcedDrafts`
override, the E116 round dose and `E71WidthTaxCensusTests.swift`.

```bash
git apply --check research/e116-artifacts/instruments.patch ; echo "exit=$?"
```

It was applied, built with `swift build -c release --force-resolved-versions`
and reverted with `git apply -R` before it was committed, so it is not a patch
that has never been run. `research/e95-artifacts/e95-census-instrument.patch`
is the counter-example: it no longer applies, because it references an
`E90GPUIntervals` type that was later removed.

The patch also carries a fix that is deliberately **not** in `Sources/`.
`swizzleDispatch` never told the ledger which selector it had intercepted, so a
census row could not say whether `grid` counted threads or threadgroups. The
shape key now ends with `entry=threads` or `entry=groups`.

Reproduce the witness with the patch applied:

```bash
git apply research/e116-artifacts/instruments.patch
swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker
research/e116_census_leg.sh entry-witness realised 8 0 0
python3 research/e116_entry_witness.py entry-witness
git apply -R research/e116-artifacts/instruments.patch
```

A plain `swift build -c release --force-resolved-versions` writes
`.build/release/mlxfast-runtime-worker` and does **not** refresh the
`.build-worker` binary the leg scripts run. Check `worker_sha256` in the leg's
`meta.txt` before trusting a census.

## W&B

Entity `wandb-applied-ai-team`, project `qwen38-mlx-challenge-senpai`, group
`e116-measured-transfer`.

| session | run id |
|---|---|
| `rung1-dose-calibration` | `p29kdppq` |
| `rung1-exactness-512` | `41vvabw6` |
| `rung2-absorption` | `7ex6rk98` |
| `rung3-dose-ladder` | `94zn6dxl` |
| `rung4-census-and-reprice` | `7juaip0i` |
