# E208 r1 — ship form and the gated 512-token confirmation

`harness=local`. Apple M4 Pro. This run used the **real 40 C cool gate**, unlike
the r0 Stage-1 ABBA session. It is still not an official or ranked score:
`official_score=false`, `rankable=false`, and the reference rows are
candidate-generated.

## 1. Ship form

The scored surface is now the minimal two-file change. The
`DARKBLOOM_E208_QMV_ARM` switch, the `stagedWide9` variant, its four `_w9` JIT
kernels and every arm reference are gone from `Sources/`, `Tests/` and
`Vendor/` (RULE 198).

| file | change |
| --- | --- |
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | `segmentedVerifyDepthCap = 7 -> 8` |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | staged plan entry `(9, 3) -> (9, 5)`, the doc comment, and the plan witness |

Diffstat against base `a5e039a4`: 2 files, 18 insertions, 5 deletions.

### Why no kernel-name plumbing was needed

MLX `CustomKernel::eval_gpu` keys its custom-kernel library cache on the kernel
name **and** the source text, and calls `clear_library(name_)` when the source
for a known name changes. `library_map_` is process-local, and there is no
on-disk cache keyed by name. The shipped kernel names therefore carry the new
source safely and need no version bump. The r0 measurement branch needed the
`_w9` names only because it held both plans inside one process at once.

### Base note verified, not assumed

`git diff --stat a5e039a4 846a2033 -- Sources/ Vendor/ mtp-head.manifest.json`
is empty. The scored surface is byte-identical between the new required base and
the r0 experiment base, so the Stage-1 statistics still apply.

## 2. Stage-0 gate after the strip

`E208Wide9QMVTests` still falsifies. With `stagedWide9` removed, the gate
compiles the legacy `(9, 3)` partition **from the shipped source** using the
same single substitution `planTable` proves is the only textual difference
between the two plans, and compares the shipped `(9, 5)` plan against it. The
perturbed positive control is unchanged.

## 3. Gated 512-token `--local-submit`

Command:

```bash
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
MLXFAST_QWEN_MTP_HEAD_DIR=~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run \
  ./benchmark-qwen-mtp.sh --local-submit
```

| field | value |
| --- | --- |
| candidate | `3d6d03a9e0615ec33b5c68913964c73715254950` |
| base | `a5e039a42636995bee421ba8892537edf64b05bf` |
| worker sha256 | `3fac22b3bcc60744...` |
| dirty submitted paths | 0 |
| cool gate | real 40 C gate **passed** at 38.1 C after 20 s (entry 45.4 C) |
| `gate_qualified_for_timing` | true (real gate, unlike r0) |
| head provenance sha256 | `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` |
| `uses_pinned_mtp_head` | true |

### Correctness and accounting

| field | value |
| --- | --- |
| `all_tokens_matched` | **true** |
| `residual_divergence_count` | **0** |
| serial leg rows | 512 / 512 checked |
| MTP leg rows | **597 / 597 checked** (RULE 179 closure) |
| `mtp-verify` | rows=513, seed_tokens=512, self_consistent=true, replayed 1 row bit-identically, chain_contradictions=0 |
| public drift tripwire | passed |
| `passed` | true |

### Timing

| field | value |
| --- | --- |
| decode tokens | 512 |
| `mtp_depth` | 8 |
| rounds | 76 |
| serial s/token | 0.0732146818190813 |
| **candidate MTP s/token** | **0.02945630089379847** |
| local ratio | 2.485535508448565 |
| accepted draft rate | 0.836852207293666 |
| effective mean draft len | 6.855263157894737 |

### Cross-check against the r0 Stage-1 ON arm

The gated leg reproduces the ungated ON arm closely, and the trajectory
invariants match to every printed digit.

| quantity | r0 Stage-1 ON (ungated) | r1 gated | agreement |
| --- | --- | --- | --- |
| candidate MTP s/token | 0.0293173 (mean of 2) | 0.0294563 | +0.47% |
| rounds | 76 | 76 | exact |
| accepted draft rate | 0.83685 | 0.836852 | exact |
| proposed / round (EDL) | 6.8553 | 6.855263 | exact |
| rows checked | 597/597 | 597/597 | exact |

The round count, acceptance rate and draft length are identical because the
kernel change cannot touch proposals, acceptance or the schedule. That is the
same trajectory-identity property that made the r0 paired statistics valid, now
confirmed on a gated run and a different thermal state.

This single gated leg confirms exactness, integrity and the gated absolute. It
does **not** replace the r0 ABBA statistics, which remain the causal evidence
for the effect size.

## 4. Contract checks on the ship form (base `a5e039a4`)

- `senpai/verify-ranked-score-boundary.sh`: PASS
- `senpai/validate-assignment-scope.sh`: OK, 2 submitted paths
- `senpai/check-editable-budget.sh`: `source=2657893/3000000 headroom=342107 growth=798/262144`
- `python3 research/twin_audit.py`: OK, 29 runtime-effective twins

Candidate growth fell from 3278 bytes on the r0 measurement branch to **798
bytes**, which is the RULE 198 strip measured in bytes.

## 5. Swift test suite

756 tests in 80 suites, **41 issues under 10 distinct test names** — the
recorded campaign floor. Not green, and not claimed as green. Zero new
failures. Failing suites: `QwenMTPTrackNamingTests` (11),
`QwenMTPHeadDeclarationTests` (6), `QwenMTPScoringSemanticsTests` (5),
`E130WiredResidencySlackTests` (1). Both QMV suites pass.

## 6. Honest limits

- The reference rows are candidate-generated, so this run cannot prove a match
  against the organizer's hidden reference.
- Both local legs use the same candidate build.
- The 512-token window completed with no `notBegun` defect and a closed ledger,
  so fixed-window continuation held. The log does not tell me whether an EOS
  token actually appeared inside this window, so I do not claim the post-EOS
  branch was exercised on this fixture.
- The ranked projection remains an EXTRAPOLATED desk model. It was not measured.
