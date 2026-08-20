# E80 rung 0a: a compile-only spill ladder that is not a `mc*` arm

## Question

Every E76 arm that fails device parity is a multi-chunk (`mc*`) arm, and every
one of those pairs carries at least 144 bytes of `applegpu_g16s` spill. Every
pair that passes carries at most 48 bytes. Spill and the chunk rewrite are
therefore perfectly confounded: the census cannot say whether the chunk rewrite
is wrong or whether a large spill frame is mis-handled on this generation.

This rung asks one compile-only question: can a numerically neutral arm that is
otherwise the shipped body be pushed to at least 144 bytes of g16s spill? If it
can, a later device-parity run of that arm separates the two explanations.

## Answer

Yes. `ballast16` at NA = 5 carries **224 bytes** of `applegpu_g16s` spill while
staying structurally and numerically identical to `plain`. That is the same
spill frame as `mc3` at NA = 4, which fails device parity.

## Identity tuple

| field | value |
| --- | --- |
| base commit | `84dc616db9aaede7bdd78241bc758baddd1121d5` |
| branch | `qwen-edward/e80-per-kernel-gpu-time-census` |
| host | `ip-10-231-2-12`, Apple M4 Pro, macOS 26.5.2 |
| toolchain | `Apple metal version 32023.883 (metalfe-32023.883)`, target `air64-apple-darwin25.5.0` |
| translation unit | `jit_string` (the `mlx-generated/*.cpp` preamble concatenation the scored worker compiles) |
| compile flags | `-std=metal4.0 -O2 -fno-fast-math` |
| generated header | `research/generated/e76_wide_arms.h`, sha256 `48de6144c288022b...`, 38 arms |
| source body | `Vendor/.../kernels/quantized.h` lines 968-1066, unchanged |
| harness | compile-only: `xcrun -sdk macosx metal` then `xcrun metal-tt -arch applegpu_g16s\|applegpu_g17s`. No GPU was used. |

## Commands run

```bash
python3 research/e76_wide_gen.py
python3 research/e76_wide_gen.py --check
python3 research/e76_rung1_census.py --na 3 4 5 6 --out /tmp/e80-rung0a-census.json
python3 research/e76_emit_arms.py --na 6 --out /tmp/e80-arms-na6.metal
python3 research/e76_emit_arms.py --na 5 --out /tmp/e80-arms-na5.metal
xcrun -sdk macosx metal -std=metal4.0 -O2 -fno-fast-math \
    -c /tmp/e80-arms-na5.metal -o /tmp/e80-arms-na5.air
cp /tmp/e80-rung0a-census.json research/e80-artifacts/rung0a-spill-ladder.json
```

The census wrote to `/tmp` and was copied afterwards, so
`research/e76-artifacts/rung1.json` is unchanged.

## The construction

`ballast<N>` adds `N` `float` values to the shipped body and changes nothing
else. The row block stays at the shipped `rows_per_simd = 4`, the staging, the
qdot expression tree, the k accumulation order and the `simd_sum` reduction are
the shipped text.

1. Before the k loop: `float ballast[N]`, `ballast[b] = float(b + 1)`.
2. At the end of each k block: `ballast[b] += float(k + b)`. This is a
   loop-carried floating-point recurrence, so the allocator must keep all `N`
   values live across the whole loop, including the register peak.
3. After the k loop: `ballast_zero = sum_b (0.0f * ballast[b])`,
   `ballast_unit = 1.0f + ballast_zero`, and the readout stores
   `simd_sum(acc[r][m]) * ballast_unit`.

The fold is bit-exact by construction, not by approximation:

* `ballast[b]` is a sum of small non-negative integers, so it is finite.
* `0.0f * finite` is `+0.0f` or `-0.0f`, and a sum of signed zeros that starts
  at `+0.0f` is `+0.0f` under round-to-nearest, so `ballast_zero` is `+0.0f`.
* `1.0f + (+0.0f)` is exactly `1.0f`, and `x * 1.0f` returns `x` for every bit
  pattern, including both signed zeros, subnormals and infinities.

`-fno-fast-math` is what makes the values live rather than dead. Without `nnan`
and `ninf` the compiler may not fold `0.0f * ballast[b]` to zero, because
`ballast[b]` could be an infinity or a NaN as far as the compiler knows, and
without `nsz` it may not fold `1.0f + ballast_zero` to `1.0f`. The device
harness `research/e69_cell_ab.m` sets `[opts setFastMathEnabled:NO]`
(line 289), which matches `-fno-fast-math` in `research/agx_crossarch.py`
(line 217), so the lever survives on the device.

The ballast loops carry `#pragma clang loop unroll(full)` so the array is
promoted to scalars. Left as an addressed `thread` array it becomes a stack
object, which is thread-memory traffic rather than allocator spill and would
not test the same thing.

**Positive control that the lever engaged:** `e76_ballast16_na5` emits
`text_sha8 = 1eba0549` while `e76_plain_na5` emits `0a5810b4`. If the compiler
had folded the ballast away, the two would be byte-identical, as `plain` and
`shipped` are. The register and spill deltas below are therefore real.

## New arms: register and spill census

Both architectures, NA = 3..6, translation unit `jit_string`. Spill is in bytes.

| arm | NA | g16s regs | g16s spill B | g17s regs | g17s spill B |
| --- | --- | --- | --- | --- | --- |
| `ballast8` | 3 | 92 | 0 | 91 | 0 |
| `ballast8` | 4 | 96 | 0 | 99 | 0 |
| `ballast8` | 5 | 96 | 16 | 103 | 0 |
| `ballast8` | 6 | 96 | 48 | 119 | 0 |
| `ballast16` | 3 | 94 | 0 | 104 | 0 |
| `ballast16` | 4 | 95 | 0 | 101 | 0 |
| `ballast16` | 5 | 96 | **224** | 111 | 0 |
| `ballast16` | 6 | 96 | 80 | 126 | 16 |
| `ballast24` | 3 | 96 | 16 | 118 | 0 |
| `ballast24` | 4 | 96 | **176** | 107 | 0 |
| `ballast24` | 5 | 96 | **272** | 119 | 0 |
| `ballast24` | 6 | 96 | 112 | 126 | 48 |
| `ballast32` | 3 | 96 | 0 | 111 | 0 |
| `ballast32` | 4 | 96 | **208** | 115 | 0 |
| `ballast32` | 5 | 96 | **288** | 126 | 240 |
| `ballast32` | 6 | 96 | 128 | 126 | 48 |
| `ballast48` | 3 | 96 | **192** | 126 | 16 |
| `ballast48` | 4 | 96 | **288** | 126 | 192 |
| `ballast48` | 5 | 96 | **368** | 126 | 304 |
| `ballast48` | 6 | 96 | **208** | 126 | 112 |
| `lazywfall` | 3 | 65 | 0 | 74 | 0 |
| `lazywfall` | 4 | 88 | 0 | 96 | 0 |
| `lazywfall` | 5 | 90 | 0 | 97 | 0 |
| `lazywfall` | 6 | 96 | 0 | 106 | 0 |
| `lazysbfall` | 3 | 85 | 0 | 95 | 0 |
| `lazysbfall` | 4 | 89 | 0 | 98 | 0 |
| `lazysbfall` | 5 | 91 | 0 | 98 | 0 |
| `lazysbfall` | 6 | 95 | 48 | 104 | 48 |
| `fallballast16` | 3 | 96 | 32 | 113 | 0 |
| `fallballast16` | 4 | 96 | 32 | 117 | 0 |
| `fallballast16` | 5 | 96 | 32 | 116 | 0 |
| `fallballast16` | 6 | 96 | 80 | 124 | 48 |
| `fallballast32` | 3 | 96 | 96 | 126 | 16 |
| `fallballast32` | 4 | 96 | 96 | 126 | 16 |
| `fallballast32` | 5 | 96 | 96 | 126 | 16 |
| `fallballast32` | 6 | 96 | **144** | 126 | 96 |

## Controls, same census

| kernel | NA | g16s regs | g16s spill B | g17s regs | g17s spill B | device parity |
| --- | --- | --- | --- | --- | --- | --- |
| `shipped` | 3 | 93 | 0 | 90 | 0 | reference |
| `shipped` | 4 | 94 | 0 | 91 | 0 | reference |
| `shipped` | 5 | 95 | 0 | 98 | 0 | reference |
| `shipped` | 6 | 96 | 16 | 111 | 0 | reference |
| `plain` | 3 | 93 | 0 | 90 | 0 | pass |
| `plain` | 4 | 94 | 0 | 91 | 0 | pass |
| `plain` | 5 | 95 | 0 | 98 | 0 | pass |
| `plain` | 6 | 96 | 16 | 111 | 0 | pass |
| `fall` | 6 | 95 | 48 | 108 | 48 | pass |
| `mc2` | 3 | 96 | 144 | 116 | 0 | FAIL |
| `mc2` | 4 | 96 | 176 | 122 | 0 | FAIL |
| `mc2` | 5 | 96 | 320 | 126 | 240 | FAIL |
| `mc2` | 6 | 96 | 352 | 126 | 288 | FAIL |
| `mc3` | 3 | 93 | 0 | 90 | 0 | pass (one chunk covers NA = 3, text equals `plain`) |
| `mc3` | 4 | 96 | 224 | 126 | 144 | FAIL |
| `mc3` | 5 | 96 | 272 | 126 | 176 | FAIL |
| `mc3` | 6 | 96 | 320 | 126 | 224 | FAIL |

`shipped_na6` reproduces the rung-1 control exactly: 96 registers and 16 bytes
on g16s, 111 registers and 0 bytes on g17s. `plain == shipped` passed on both
architectures at all four widths.

## The parity candidate

**`ballast16` at NA = 5.**

* g16s: 96 registers, **224 bytes** of spill. This is at least 144 bytes, and it
  equals the `mc3` NA = 4 spill frame exactly, which fails device parity.
* g17s: 111 registers, **0 bytes** of spill. The arm spills on the local
  generation only, which is the wanted asymmetry: parity is measured locally on
  g16s.
* It is the smallest ballast count that reaches 144 bytes anywhere in the
  ladder, and NA = 5 is inside the region where `mc2`, `mc3` and `mc4` all fail.
* It keeps the shipped row block, the shipped staging, the shipped
  `vec<float, NA>` layout, the shipped expression trees and the shipped
  reduction. The only difference from `plain` is the ballast.

Two controls make the device run interpretable, and both should run in the same
batch:

* `ballast8` at NA = 5: same family, same width, only 16 bytes of g16s spill.
  A pass here plus a fail on `ballast16` isolates spill, not the construction.
* `ballast16` at NA = 3 or NA = 4: the same arm with 0 bytes of spill. A pass
  here proves the ballast fold is bit-neutral on the device.

Backup candidates if a g17s spill is also wanted: `ballast32` at NA = 5
(288 g16s, 240 g17s) or `ballast48` at NA = 5 (368 g16s, 304 g17s).

## Command line for the later device-parity run

`research/e76_session.sh --mode parity` ignores `--arms` and runs every arm in
batches, which is 4 batches per width now that the generator holds 38 arms. For
one arm at one width, call the harness directly. `plain` must be first: the
harness uses `pso[0]` as the parity reference
(`research/e69_cell_ab.m` line 341).

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p /tmp/e80-build
python3 research/e76_wide_gen.py --check
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o /tmp/e80-build/e76_cell_ab research/e69_cell_ab.m
python3 research/e76_emit_arms.py --na 5 --out /tmp/e80-build/arms_na5.metal
MLXFAST_MACMON_BIN="${HOME}/bin/macmon" /tmp/e80-build/e76_cell_ab \
  --prefix e76_cell_ \
  --source /tmp/e80-build/arms_na5.metal \
  --na 5 --reps 1 --warmup-reps 0 --target-bytes 1e8 \
  --arms plain,ballast8,ballast16 \
  --out research/e80-artifacts/rung0a-parity-na5-ballast.json
```

Read `shapes[*].parity_differing_vs_plain` in the output. This is a correctness
run, not a timing run, so it needs no cool gate and no W&B run, exactly like the
E76 parity mode.

Optional neutrality control at a width where `ballast16` does not spill:

```bash
python3 research/e76_emit_arms.py --na 4 --out /tmp/e80-build/arms_na4.metal
MLXFAST_MACMON_BIN="${HOME}/bin/macmon" /tmp/e80-build/e76_cell_ab \
  --prefix e76_cell_ \
  --source /tmp/e80-build/arms_na4.metal \
  --na 4 --reps 1 --warmup-reps 0 --target-bytes 1e8 \
  --arms plain,ballast16 \
  --out research/e80-artifacts/rung0a-parity-na4-ballast.json
```

## What each device outcome would mean

| `ballast16` NA = 5 | `ballast8` NA = 5 and `ballast16` NA = 4 | conclusion |
| --- | --- | --- |
| FAIL | pass | Large g16s spill alone breaks parity. The `mc*` failures are not evidence against the chunk rewrite, and the register-pressure axis becomes the thing to control. |
| pass | pass | Spill alone is not sufficient. The `mc*` failures come from the chunk rewrite itself, and the 144-byte correlation is a coincidence of the search grid. |
| FAIL | FAIL | The ballast construction is not bit-neutral on the device. The rung is invalid and says nothing about spill. |

## Reproduce

```bash
python3 research/e76_wide_gen.py --check
python3 research/e76_rung1_census.py --na 3 4 5 6 --out /tmp/replay.json
python3 -c "
import json
a = json.load(open('research/e80-artifacts/rung0a-spill-ladder.json'))['census']
b = json.load(open('/tmp/replay.json'))['census']
print('identical:', a == b)"
```
